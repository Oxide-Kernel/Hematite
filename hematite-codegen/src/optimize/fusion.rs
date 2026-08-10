// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! T4.2a — Operator fusion pass.
//!
//! Pattern-list matching over the static parsed graph (NOT a
//! post-dominator algorithm).  The engine is memory-bound: every fused op
//! eliminates a full SRAM intermediate round-trip, so the pass greedily
//! merges elementwise/activation epilogues into the kernel that produces
//! their input.  Output is a [`FusedSchedule`]: one [`FusedGroup`] per
//! emitted kernel call, in original execution order, with every absorbed
//! op and eliminated tensor recorded.  The T4.1 emitter wiring task threads
//! this into `emit_model`; T4.2b (arena) and T4.2c (layout) consume the
//! same op list.
//!
//! ## Pattern priority (per plan T4.2a)
//!
//! * **(a)** conv/depthwise/fc + bias + requantize + activation epilogue →
//!   one fused kernel call.  The activation is either the op's own
//!   `fused_activation` field (preferred when present, codes 1=RELU /
//!   3=RELU6) or a following standalone RELU(19)/RELU6(21)/HARD_SWISH(117)
//!   absorbed with a `HasOneUse` guard.  Unsupported field codes (2, 4, 5)
//!   pass through untouched — never claimed.
//! * **(b)** elementwise chains (`add→relu→mul→hardswish`) → one
//!   register-held scalar loop, zero intermediate stores (net-new vs TFLM/
//!   ESP-DL which fuse only inside kernels).
//! * **(c)** residual-add groups `conv→add(residual)→relu` one pass,
//!   in-place (liveness-proven by T4.2b; fusion records the structure).
//! * **(d)** pool/softmax absorbs a PRECEDING mul/sub (softmax absorbs at
//!   zero cost — it normalizes; the mul scale folds into the consumer's
//!   input math, the sub constant folds into the input offset).
//! * **(e)** requantize-scale constant folding: `conv→mul` where the mul is
//!   a pure scale change (all-ones constant operand, zero offsets) folds to
//!   one per-channel multiply-shift `quantize_multiplier(s_in·s_w/s_out)`
//!   at compile time.
//!
//! ## Safety guards (invariants)
//!
//! * An intermediate tensor is fused only when it has exactly one consumer
//!   op and is not a model input/output (`HasOneUse`).
//! * Every absorbed op is consumed exactly once (the pass marks `consumed`
//!   after each group forms).
//! * Fused groups never move an op earlier than its inputs allow: chain
//!   operands and residual tensors must already exist at the anchor's
//!   execution position (see [`Ctx::available_before`]).
//!
//! ## Op-code note (verified against the vendored v23.1-era schema)
//!
//! RELU = 19, RELU6 = 21, HARD_SWISH = 117.  Codes 101/102/103 are
//! ABS/SPLIT_V/UNIQUE in this schema era and must NEVER be treated as
//! activations.
//
// Dead-code warnings are expected at T4.2a — the T4.1 emitter wiring task
// consumes these descriptors.
#![allow(dead_code)]

use crate::flatbuffer::{ParsedModel, ParsedOp, ParsedOptions};

// BuiltinOperator codes (TFLite v23.1-era schema, resolved by T4.0).
const ADD: i32 = 0;
const AVERAGE_POOL_2D: i32 = 1;
const CONV_2D: i32 = 3;
const DEPTHWISE_CONV_2D: i32 = 4;
const FULLY_CONNECTED: i32 = 9;
const MAX_POOL_2D: i32 = 17;
const MUL: i32 = 18;
const RELU: i32 = 19;
const RELU6: i32 = 21;
const SOFTMAX: i32 = 25;
const SUB: i32 = 41;
const HARD_SWISH: i32 = 117;

// `fused_activation_function` field values (ActivationFunctionType).
const ACT_NONE: i8 = 0;
const ACT_RELU: i8 = 1;
const ACT_RELU6: i8 = 3;

// ---------------------------------------------------------------------------
// Fused-schedule IR (self-contained, emitter-friendly)
// ---------------------------------------------------------------------------

/// Output of [`fuse`]: the fused op sequence in original execution order.
///
/// One [`FusedGroup`] per emitted kernel call.  `total_ops` is the original
/// op count so downstream passes / diagnostics can quantify the reduction.
#[derive(Clone, Debug)]
pub(crate) struct FusedSchedule {
    /// Fused groups, ordered by their anchor op's execution position.
    pub(crate) groups: Vec<FusedGroup>,
    /// Number of ops in the original model (`model.ops().len()`).
    pub(crate) total_ops: usize,
}

impl FusedSchedule {
    /// Number of original ops absorbed into a group (eliminated kernel calls).
    pub(crate) fn fused_op_count(&self) -> usize {
        self.groups.iter().map(|g| g.absorbed_ops.len()).sum()
    }

    /// Number of kernel calls the fused schedule emits.
    pub(crate) fn emitted_op_count(&self) -> usize {
        self.groups.len()
    }
}

/// One emitted kernel call: the anchor op plus everything fused into it.
#[derive(Clone, Debug)]
pub(crate) struct FusedGroup {
    /// Index of the anchor op in `ParsedModel::ops()` (the op that remains
    /// as a kernel call: conv-family, first elementwise-of-chain, pool,
    /// softmax, or an untouched op).
    pub(crate) anchor_op_index: usize,
    /// Anchor's resolved `builtin_code`.
    pub(crate) anchor_builtin: i32,
    /// Effective kernel inputs after folds (pattern (d) substitutes the
    /// pre-elementwise tensor for the anchor's original input).
    pub(crate) inputs: Vec<u32>,
    /// Tensor the group produces (the last absorbed op's output, or the
    /// anchor's output when nothing is absorbed).
    pub(crate) output_tensor: u32,
    /// Original op indices absorbed into this group (eliminated as kernel
    /// calls).
    pub(crate) absorbed_ops: Vec<usize>,
    /// Intermediate tensor indices eliminated (no SRAM round-trip).
    pub(crate) eliminated_tensors: Vec<u32>,
    /// Fused activation epilogue (pattern (a); also set by pattern (c)'s
    /// trailing relu).  `None` = identity.
    pub(crate) activation: Option<FusedActivation>,
    /// Absorbed elementwise chain ops (pattern (b)) executed in one
    /// register-held scalar loop, in order.
    pub(crate) elementwise_chain: Vec<AbsorbedElementwise>,
    /// Residual-add group (pattern (c)): the residual tensor is accumulated
    /// into the anchor's output in the same pass (in-place).
    pub(crate) residual_add: Option<ResidualAdd>,
    /// Input fold (pattern (d)): a preceding mul/sub absorbed into a
    /// pool/softmax's input handling.
    pub(crate) input_fold: Option<InputFold>,
    /// Constant-folded requantize (pattern (e)): one per-channel
    /// multiply-shift replacing kernel-requantize + scale-change requantize.
    pub(crate) folded_requantize: Option<FoldedRequantize>,
    /// T1.1 correctness tier: `true` for T2 (proof-obligated) groups.
    /// Input-folds (pattern (d)) and requantize-folds (pattern (e)) are
    /// algebraically transformative — NOT semantics-preserving — so their
    /// bit-exactness depends on a per-model fused==unfused verification
    /// passing before the composed kernel may be used.  T1.2 consumes this
    /// flag: a `true` group is emitted per-op until verification passes.
    pub(crate) requires_verification: bool,
}

impl FusedGroup {
    /// A group that is just the anchor op (nothing absorbed).
    fn anchor(op_index: usize, op: &ParsedOp<'_>) -> Self {
        FusedGroup {
            anchor_op_index: op_index,
            anchor_builtin: op.builtin_code,
            inputs: op.inputs.clone(),
            output_tensor: op.outputs.first().copied().unwrap_or(u32::MAX),
            absorbed_ops: Vec::new(),
            eliminated_tensors: Vec::new(),
            activation: None,
            elementwise_chain: Vec::new(),
            residual_add: None,
            input_fold: None,
            folded_requantize: None,
            requires_verification: false,
        }
    }
}

/// A fused activation epilogue with its TFLM quantized clamp range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FusedActivationKind {
    Relu,
    Relu6,
    HardSwish,
}

/// Activation epilogue params for the emitted kernel call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FusedActivation {
    pub(crate) kind: FusedActivationKind,
    /// `quantized_activation_min` — TFLM `CalculateActivationRangeQuantized`.
    pub(crate) quantized_min: i32,
    /// `quantized_activation_max`.
    pub(crate) quantized_max: i32,
}

/// Kind of an op absorbed into an elementwise chain (pattern (b)).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ElementwiseKind {
    Add,
    Mul,
    Sub,
    Relu,
    Relu6,
    HardSwish,
}

/// TFLM int8 requantize parameters for one fused elementwise / residual-add
/// step (T1.1).  Mirrors the per-op elementwise math emitted in generate.rs
/// (`left_shift = 20` twice-max scaling for ADD/SUB, a single
/// `QuantizeMultiplier` output ratio for MUL, the activation output ratio
/// for activation steps) so the composed emitter (T1.2) can reproduce each
/// step bit-exactly from the operand tensors.  Steps are NEVER collapsed —
/// every step keeps its own multiplier/shift/offset triple.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StepRequantize {
    /// `left_shift` — 20 for ADD/SUB (twice-max scaling), 0 for MUL and
    /// activation steps.
    pub(crate) left_shift: i32,
    /// ADD/SUB: `QuantizeMultiplier(input1_scale / twice_max)`; else 0.
    pub(crate) input1_multiplier: i32,
    pub(crate) input1_shift: i32,
    /// ADD/SUB: `QuantizeMultiplier(input2_scale / twice_max)`; else 0.
    pub(crate) input2_multiplier: i32,
    pub(crate) input2_shift: i32,
    /// Output-ratio multiplier: ADD/SUB
    /// `twice_max / (2^left_shift · output_scale)`; MUL
    /// `input1_scale · input2_scale / output_scale`; activation steps
    /// `input_scale / output_scale`.
    pub(crate) output_multiplier: i32,
    pub(crate) output_shift: i32,
    /// Input-1 offset — `-zero_point` for ADD/SUB/MUL steps, `+zero_point`
    /// for activation steps (each follows the respective generate.rs
    /// per-op emission convention).
    pub(crate) input1_offset: i32,
    /// Input-2 offset — `-zero_point` of the operand/residual tensor
    /// (0 for activation steps).
    pub(crate) input2_offset: i32,
    /// Output offset — `+zero_point` of the step's output tensor.
    pub(crate) output_offset: i32,
}

impl StepRequantize {
    /// Requantize params for one chain/residual-add step computed from the
    /// parsed tensors: `in1` is the running tensor (a chain) or the anchor
    /// output (a residual add), `in2` the operand/residual (`u32::MAX` for
    /// activation steps), `out` the step's output tensor.
    fn elementwise(
        ctx: &Ctx<'_>,
        in1: u32,
        in2: u32,
        out: u32,
        kind: ElementwiseKind,
    ) -> Self {
        let in1_scale = f64::from(ctx.scale_of(in1).unwrap_or(1.0));
        let in2_scale = f64::from(ctx.scale_of(in2).unwrap_or(1.0));
        let out_scale = f64::from(ctx.scale_of(out).unwrap_or(1.0));
        let (left_shift, i1m, i1s, i2m, i2s, om, os) = match kind {
            ElementwiseKind::Add | ElementwiseKind::Sub => {
                let twice_max = 2.0 * in1_scale.max(in2_scale);
                let (a, b) = quantize_multiplier(in1_scale / twice_max);
                let (c, d) = quantize_multiplier(in2_scale / twice_max);
                let (e, f) = quantize_multiplier(twice_max / ((1i32 << 20) as f64 * out_scale));
                (20, a, b, c, d, e, f)
            }
            ElementwiseKind::Mul => {
                let (e, f) = quantize_multiplier(in1_scale * in2_scale / out_scale);
                (0, 0, 0, 0, 0, e, f)
            }
            ElementwiseKind::Relu | ElementwiseKind::Relu6 | ElementwiseKind::HardSwish => {
                let (e, f) = quantize_multiplier(in1_scale / out_scale);
                (0, 0, 0, 0, 0, e, f)
            }
        };
        let is_activation =
            matches!(kind, ElementwiseKind::Relu | ElementwiseKind::Relu6 | ElementwiseKind::HardSwish);
        StepRequantize {
            left_shift,
            input1_multiplier: i1m,
            input1_shift: i1s,
            input2_multiplier: i2m,
            input2_shift: i2s,
            output_multiplier: om,
            output_shift: os,
            input1_offset: if is_activation {
                ctx.zp_of(in1).unwrap_or(0) as i32
            } else {
                -(ctx.zp_of(in1).unwrap_or(0) as i32)
            },
            input2_offset: if is_activation {
                0
            } else {
                -(ctx.zp_of(in2).unwrap_or(0) as i32)
            },
            output_offset: ctx.zp_of(out).unwrap_or(0) as i32,
        }
    }
}

/// One op absorbed into a register-held elementwise loop (pattern (b)).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AbsorbedElementwise {
    /// Original op index.
    pub(crate) op_index: usize,
    pub(crate) kind: ElementwiseKind,
    /// The non-running input tensor (`u32::MAX` for activations).  The
    /// running value is produced by the previous chain step in registers.
    pub(crate) operand_tensor: u32,
    /// Quant params of this op's output tensor (the chain result scale).
    pub(crate) output_scale: f32,
    pub(crate) output_zero_point: i64,
    /// T1.1: this step's TFLM requantize params (twice-max triple for
    /// ADD/SUB, output ratio for MUL/activations) — carried per step so the
    /// emitter reproduces the exact rounding sequence, never collapsed.
    pub(crate) requantize: StepRequantize,
}

/// Residual-add group (pattern (c)).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResidualAdd {
    /// Original ADD op index (absorbed).
    pub(crate) op_index: usize,
    /// The residual source tensor accumulated into the anchor's output.
    pub(crate) residual_tensor: u32,
    /// Quant params of the ADD's output tensor (written in-place).
    pub(crate) output_scale: f32,
    pub(crate) output_zero_point: i64,
    /// T1.1: the ADD's TFLM requantize params (input1 = anchor output,
    /// input2 = residual) carried so the composed kernel reproduces the
    /// exact TFLM Add rounding sequence.
    pub(crate) requantize: StepRequantize,
}

/// Input fold for a pool/softmax (pattern (d)).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct InputFold {
    /// Original MUL/SUB op index (absorbed).
    pub(crate) op_index: usize,
    /// `MUL` (18) or `SUB` (41).
    pub(crate) builtin: i32,
    /// The pre-elementwise tensor the consumer kernel now reads instead of
    /// the elementwise output.
    pub(crate) folded_input_tensor: u32,
    /// The constant operand tensor (`input[1]` of the absorbed op).
    pub(crate) operand_tensor: u32,
    /// MUL: real-domain scale ratio `s_out / s_in` the mul applies
    /// (`folded_scale = output_scale / input_scale`).  SUB: the real-domain
    /// constant `c = s_operand·(q_operand − zp_operand)` subtracted from the
    /// input.  The emitter folds this into the consumer's scale/offset math.
    pub(crate) folded_scale: f32,
    /// Zero point of the folded input tensor (for the emitter's offset math).
    pub(crate) input_zero_point: i64,
}

/// Constant-folded requantize (pattern (e)).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FoldedRequantize {
    /// Original MUL op index (absorbed — the scale-change requantize).
    pub(crate) op_index: usize,
    /// Final output scale after folding (the mul's output scale).
    pub(crate) output_scale: f32,
    /// Final output zero point (the mul's output zero point).
    pub(crate) output_zero_point: i64,
    /// Per-channel `(multiplier, shift)` pairs replacing the two-step
    /// requantize: `quantize_multiplier(s_in·s_w[c]/s_out_final)`.
    pub(crate) multipliers: Vec<(i32, i32)>,
}

// ---------------------------------------------------------------------------
// Pass context
// ---------------------------------------------------------------------------

/// Per-model fusion context: static consumer counts, fold-target marks, and
/// the consumed-op ledger.
struct Ctx<'a> {
    model: &'a ParsedModel<'a>,
    /// Per-tensor count of consumer ops (index-aligned with `tensors()`).
    consumers: Vec<u32>,
    /// Op indices marked as pattern-(d) fold targets (never standalone).
    fold_targets: Vec<bool>,
    /// Ops already absorbed into a group (no double-absorption).
    consumed: Vec<bool>,
}

impl Ctx<'_> {
    /// `HasOneUse` guard: exactly one consumer op, and the tensor is not a
    /// model input or output (those must survive as real tensors).
    fn has_one_use(&self, t: u32) -> bool {
        if t == u32::MAX {
            return false;
        }
        let ti = t as usize;
        if ti >= self.consumers.len() || self.consumers[ti] != 1 {
            return false;
        }
        !self.model.inputs().contains(&t) && !self.model.outputs().contains(&t)
    }

    /// The single consumer op of `t`, under the `HasOneUse` guard.
    fn next_consumer(&self, t: u32) -> Option<usize> {
        if !self.has_one_use(t) {
            return None;
        }
        self.model.ops().iter().position(|op| op.inputs.contains(&t))
    }

    /// Op index producing `t` (`None` for constants/model inputs).
    fn producer(&self, t: u32) -> Option<usize> {
        self.model.ops().iter().position(|op| op.outputs.contains(&t))
    }

    /// True when `t` is a constant buffer (flash-resident data).
    fn is_constant(&self, t: u32) -> bool {
        self.model
            .tensor_by_index(t as usize)
            .and_then(|tt| self.model.buffer_data(tt))
            .is_some()
    }

    /// Liveness guard: `t` already exists when the op at `op_index`
    /// executes (constant or model input, or produced strictly earlier).
    fn available_before(&self, t: u32, op_index: usize) -> bool {
        if self.is_constant(t) || self.model.inputs().contains(&t) {
            return true;
        }
        self.producer(t).is_some_and(|p| p < op_index)
    }

    fn scale_of(&self, t: u32) -> Option<f32> {
        self.model
            .tensor_by_index(t as usize)
            .and_then(|tt| tt.quant.as_ref())
            .map(|q| q.scale)
    }

    fn zp_of(&self, t: u32) -> Option<i64> {
        self.model
            .tensor_by_index(t as usize)
            .and_then(|tt| tt.quant.as_ref())
            .map(|q| q.zero_point)
    }

    /// Conv-family weight scales: per-channel scales when present, else the
    /// single per-tensor scale.
    fn weight_scales(&self, weight: u32) -> Option<Vec<f32>> {
        let q = self.model.tensor_by_index(weight as usize)?.quant.as_ref()?;
        if let Some(pc) = &q.per_channel {
            Some(pc.scales.clone())
        } else {
            Some(vec![q.scale])
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Fuse the whole model into a [`FusedSchedule`] — a pure function over the
/// static IR, deterministic (iterates ops in execution order), no global
/// state.
pub(crate) fn fuse(model: &ParsedModel<'_>) -> FusedSchedule {
    let consumers = consumer_counts(model);
    let mut ctx = Ctx {
        model,
        consumers,
        fold_targets: Vec::new(),
        consumed: vec![false; model.ops().len()],
    };
    // Pattern (d) pre-pass: mark mul/sub ops whose single consumer is a
    // pool/softmax so they never form standalone groups.  The marking uses
    // the exact same predicate as the consumer-side absorption, so a marked
    // op can never be orphaned.
    ctx.fold_targets = (0..model.ops().len())
        .map(|i| try_input_fold(&ctx, i).is_some())
        .collect();

    let mut groups = Vec::with_capacity(model.ops().len());
    for i in 0..model.ops().len() {
        if ctx.consumed[i] || ctx.fold_targets[i] {
            continue;
        }
        let code = model.ops()[i].builtin_code;
        let mut group = if is_conv_family(code) {
            try_fuse_conv(&ctx, i)
        } else if is_elementwise(code) {
            try_fuse_elementwise_chain(&ctx, i)
        } else if is_pool_or_softmax(code) {
            try_fuse_input_fold(&ctx, i)
        } else {
            FusedGroup::anchor(i, &model.ops()[i])
        };
        for &absorbed in &group.absorbed_ops {
            ctx.consumed[absorbed] = true;
        }
        // T1.1: pattern-(d) input-folds and pattern-(e) requantize-folds are
        // algebraically transformative (NOT semantics-preserving) — they are
        // proof-obligated T2 groups, tagged here for the emitter so T1.2 can
        // keep them per-op until a fused==unfused verification passes.
        group.requires_verification =
            group.input_fold.is_some() || group.folded_requantize.is_some();
        groups.push(group);
    }

    FusedSchedule { groups, total_ops: model.ops().len() }
}

fn consumer_counts(model: &ParsedModel<'_>) -> Vec<u32> {
    let mut counts = vec![0u32; model.tensors().len()];
    for op in model.ops() {
        for &t in &op.inputs {
            if t != u32::MAX {
                if let Some(c) = counts.get_mut(t as usize) {
                    *c += 1;
                }
            }
        }
    }
    counts
}

// ---------------------------------------------------------------------------
// Pattern (d) — input fold
// ---------------------------------------------------------------------------

/// Compute the pattern-(d) input fold for a MUL/SUB op whose output feeds a
/// pool or softmax.  `None` when any gate fails — the same function marks
/// fold targets in the pre-pass and absorbs them on the consumer side, so a
/// marked op is always absorbed (never orphaned).
///
/// * MUL folds `folded_scale = s_out / s_in` (the real-domain ratio the mul
///   applies), which the consumer's scale math absorbs at zero extra cost.
/// * SUB folds the real-domain constant `c = s_operand·(q_operand − zp)`
///   read from the constant operand buffer (requires the operand to carry
///   quantization).
fn try_input_fold(ctx: &Ctx<'_>, op_index: usize) -> Option<InputFold> {
    let op = &ctx.model.ops()[op_index];
    if op.builtin_code != MUL && op.builtin_code != SUB {
        return None;
    }
    let out = *op.outputs.first().unwrap_or(&u32::MAX);
    if !ctx.has_one_use(out) {
        return None;
    }
    let consumer = ctx.next_consumer(out)?;
    let consumer_code = ctx.model.ops()[consumer].builtin_code;
    if !matches!(consumer_code, AVERAGE_POOL_2D | MAX_POOL_2D | SOFTMAX) {
        return None;
    }
    let in_t = *op.inputs.first().unwrap_or(&u32::MAX);
    let operand = *op.inputs.get(1).unwrap_or(&u32::MAX);
    if in_t == u32::MAX {
        return None;
    }

    let input_zp = ctx.zp_of(in_t).unwrap_or(0);
    let folded_scale = if op.builtin_code == MUL {
        let s_out = ctx.scale_of(out)?;
        let s_in = ctx.scale_of(in_t)?;
        if s_in == 0.0 {
            return None;
        }
        s_out / s_in
    } else {
        let t = ctx.model.tensor_by_index(operand as usize)?;
        let data = ctx.model.buffer_data(t)?;
        let first = *data.first()?;
        let s = ctx.scale_of(operand)?;
        let zp = ctx.zp_of(operand).unwrap_or(0);
        s * (f32::from(first as i8) - zp as f32)
    };

    Some(InputFold {
        op_index,
        builtin: op.builtin_code,
        folded_input_tensor: in_t,
        operand_tensor: operand,
        folded_scale,
        input_zero_point: input_zp,
    })
}

// ---------------------------------------------------------------------------
// Pattern (a) / (c) / (e) — conv-family anchor
// ---------------------------------------------------------------------------

fn try_fuse_conv(ctx: &Ctx<'_>, op_index: usize) -> FusedGroup {
    let op = &ctx.model.ops()[op_index];
    let out0 = op.outputs.first().copied().unwrap_or(u32::MAX);
    let mut g = FusedGroup::anchor(op_index, op);

    // (c) residual-add: conv → add(residual) → [relu] — most specific.
    if let Some(add_idx) = ctx.next_consumer(out0) {
        let add_op = &ctx.model.ops()[add_idx];
        if add_op.builtin_code == ADD && try_fuse_residual(ctx, op_index, add_idx, &mut g) {
            return g;
        }
    }

    // (a) prefer the explicit fused_activation field when present.
    if let Some(kind) = field_activation(op) {
        g.activation = Some(activation_from(ctx, kind, out0));
        return g;
    }

    // (e) requantize-scale fold: conv → pure-scale mul → [activation].
    let mut out = out0;
    if let Some(mul_idx) = ctx.next_consumer(out) {
        if let Some(fr) = try_fold_requantize(ctx, op_index, mul_idx) {
            let mul_op = &ctx.model.ops()[mul_idx];
            let mul_out = mul_op.outputs.first().copied().unwrap_or(u32::MAX);
            g.folded_requantize = Some(fr);
            g.absorbed_ops.push(mul_idx);
            g.eliminated_tensors.push(out);
            g.output_tensor = mul_out;
            // The mul's own fused activation (if any) becomes the epilogue.
            if let Some(kind) = field_activation(mul_op) {
                g.activation = Some(activation_from(ctx, kind, mul_out));
                return g;
            }
            out = mul_out;
        }
    }

    // (a) absorb a following standalone activation.  T1.1 correctness tier:
    // a standalone RELU/RELU6/HARD_SWISH kernel is a FULL requantize
    // (`x + input_offset; max(v, 0); multiply_by_quantized_multiplier;
    // + output_offset` — hematite-ref activation.rs), NOT a pure clamp.
    // Absorbing it as an `activation_range` clamp (see `activation_from`) is
    // exact only when the activation's output quant is IDENTICAL to the
    // anchor's output quant (scale AND zero point): equal scales with
    // differing zps diverge — unfused `max(x, 0) + zp_out` vs fused
    // `clamp(x, zp_out, 127)`.  When the gate fails the group is NOT
    // formed; the activation falls back to a separate per-op call.
    if let Some(act_idx) = ctx.next_consumer(out) {
        if let Some(kind) = activation_kind(ctx.model.ops()[act_idx].builtin_code) {
            let act_out = ctx.model.ops()[act_idx].outputs.first().copied().unwrap_or(u32::MAX);
            if quant_identity(ctx, act_out, out) {
                g.activation = Some(activation_from(ctx, kind, act_out));
                g.absorbed_ops.push(act_idx);
                g.eliminated_tensors.push(out);
                g.output_tensor = act_out;
            }
        }
    }

    g
}

/// Pattern (c): fuse `conv → add(residual) → [relu]` into one pass.  The
/// conv writes the add's output tensor in place; the epilogue accumulates
/// the residual and applies the relu.  Returns `true` when fused.
///
/// The residual must be a real computed tensor (not a model input/output,
/// not a constant) whose producer runs strictly before the conv — otherwise
/// the fused epilogue would read a tensor that does not exist yet.
fn try_fuse_residual(
    ctx: &Ctx<'_>,
    conv_idx: usize,
    add_idx: usize,
    g: &mut FusedGroup,
) -> bool {
    let conv_out = g.output_tensor;
    let add_op = &ctx.model.ops()[add_idx];
    let add_out = add_op.outputs.first().copied().unwrap_or(u32::MAX);

    let residual = match add_op.inputs.iter().copied().find(|&t| t != conv_out) {
        Some(r) => r,
        None => return false,
    };
    if ctx.model.inputs().contains(&residual) || ctx.model.outputs().contains(&residual) {
        return false;
    }
    if ctx.is_constant(residual) {
        return false;
    }
    match ctx.producer(residual) {
        Some(p) if p < conv_idx => {}
        _ => return false, // residual not yet available at the conv's position
    }

    // T1.1: the ADD's per-step requantize params (input1 = the anchor
    // output, input2 = the residual) reproduce the exact TFLM Add rounding.
    let requantize = StepRequantize::elementwise(
        ctx,
        conv_out,
        residual,
        add_out,
        ElementwiseKind::Add,
    );

    g.residual_add = Some(ResidualAdd {
        op_index: add_idx,
        residual_tensor: residual,
        output_scale: ctx.scale_of(add_out).unwrap_or(1.0),
        output_zero_point: ctx.zp_of(add_out).unwrap_or(0),
        requantize,
    });
    g.absorbed_ops.push(add_idx);
    g.eliminated_tensors.push(conv_out);
    g.output_tensor = add_out;

    // Absorb a trailing relu/relu6/hard-swish as the fused activation.
    // Same T1.1 quant-identity gate as the conv path: the standalone
    // activation is a full requantize, so the clamp is exact only when the
    // activation output quant equals the add output quant.
    if let Some(act_idx) = ctx.next_consumer(add_out) {
        if let Some(kind) = activation_kind(ctx.model.ops()[act_idx].builtin_code) {
            let act_out = ctx.model.ops()[act_idx].outputs.first().copied().unwrap_or(u32::MAX);
            if quant_identity(ctx, act_out, add_out) {
                g.activation = Some(activation_from(ctx, kind, act_out));
                g.absorbed_ops.push(act_idx);
                g.eliminated_tensors.push(add_out);
                g.output_tensor = act_out;
            }
        }
    }
    true
}

/// Pattern (e): fold a `conv → mul` pure-scale change into one per-channel
/// requantize.  Exactness requires the conv output zero point to be 0 (so
/// the mul's `input1_offset` is 0) and the operand to be an all-ones
/// constant at scale 1.0, zero point 0 — a pure scale change with no data
/// shift.  The folded multiplier is `quantize_multiplier(s_in·s_w/s_out)`
/// where `s_out` is the mul's output scale.
fn try_fold_requantize(ctx: &Ctx<'_>, conv_idx: usize, mul_idx: usize) -> Option<FoldedRequantize> {
    let mul = &ctx.model.ops()[mul_idx];
    if mul.builtin_code != MUL || mul.inputs.len() < 2 {
        return None;
    }
    let conv_out = mul.inputs[0];
    let operand = mul.inputs[1];

    // Exactness gates (double-rounding-free composition).
    if ctx.zp_of(conv_out) != Some(0) {
        return None;
    }
    if ctx.zp_of(operand) != Some(0) || (ctx.scale_of(operand)? - 1.0).abs() > 1e-6 {
        return None;
    }
    let operand_t = ctx.model.tensor_by_index(operand as usize)?;
    let data = ctx.model.buffer_data(operand_t)?;
    if data.iter().any(|&b| b != 1) {
        return None; // not all-ones → data-dependent scaling, not a requantize
    }

    // Metis G-3: the fold may only claim exactness when the mul's output
    // scale equals the conv output scale (`s_conv_out == s_mul_out`, i.e.
    // the mul's own requantize multiplier is the identity).  A real scale
    // change makes the mul perform its OWN `QuantizeMultiplier` rounding
    // AFTER the conv's — two-stage rounding is NOT equivalent to the single
    // folded requantize, so that case is never claimed here (the mul stays
    // a separate per-op call; only a passing T5.1 fused==unfused
    // verification could ever allow a scale-changing composed kernel).
    let s_mul_out = ctx.scale_of(mul.outputs.first().copied().unwrap_or(u32::MAX))?;
    if ctx.scale_of(conv_out)? != s_mul_out {
        return None;
    }

    let conv = &ctx.model.ops()[conv_idx];
    let weight = *conv.inputs.get(1)?;
    let s_in = ctx.scale_of(*conv.inputs.first()?)?;
    let multipliers = ctx
        .weight_scales(weight)?
        .iter()
        .map(|&sw| quantize_multiplier(f64::from(s_in * sw / s_mul_out)))
        .collect();

    Some(FoldedRequantize {
        op_index: mul_idx,
        output_scale: s_mul_out,
        output_zero_point: ctx.zp_of(mul.outputs[0]).unwrap_or(0),
        multipliers,
    })
}

// ---------------------------------------------------------------------------
// Pattern (b) — elementwise chain anchor
// ---------------------------------------------------------------------------

/// Fuse a run of elementwise/activation ops following an ADD/MUL/SUB anchor
/// into one register-held scalar loop.  Each absorbed op's output is the
/// next op's running input (HasOneUse), and every non-running operand must
/// already exist when the anchor executes.
fn try_fuse_elementwise_chain(ctx: &Ctx<'_>, op_index: usize) -> FusedGroup {
    let op = &ctx.model.ops()[op_index];
    let mut g = FusedGroup::anchor(op_index, op);
    let mut running = g.output_tensor;

    while let Some(next_idx) = ctx.next_consumer(running) {
        let next = &ctx.model.ops()[next_idx];
        let Some(kind) = chain_kind(next.builtin_code) else { break };
        if next
            .inputs
            .iter()
            .any(|&t| t != running && !ctx.available_before(t, op_index))
        {
            break; // an operand would not exist at the anchor's position
        }
        let operand = next
            .inputs
            .iter()
            .copied()
            .find(|&t| t != running)
            .unwrap_or(u32::MAX);
        let out = next.outputs.first().copied().unwrap_or(u32::MAX);

        // T1.1: carry this step's TFLM requantize params — each chain step
        // keeps its own multiplier/shift/offset triple (never collapsed).
        let requantize = StepRequantize::elementwise(ctx, running, operand, out, kind);

        g.elementwise_chain.push(AbsorbedElementwise {
            op_index: next_idx,
            kind,
            operand_tensor: operand,
            output_scale: ctx.scale_of(out).unwrap_or(1.0),
            output_zero_point: ctx.zp_of(out).unwrap_or(0),
            requantize,
        });
        g.absorbed_ops.push(next_idx);
        g.eliminated_tensors.push(running);
        g.output_tensor = out;
        running = out;
    }

    g
}

// ---------------------------------------------------------------------------
// Pattern (d) — pool/softmax anchor
// ---------------------------------------------------------------------------

/// A pool or softmax absorbs a preceding mul/sub marked as a fold target:
/// the consumer reads the pre-elementwise tensor and folds the mul scale
/// ratio (or sub constant) into its input math.
fn try_fuse_input_fold(ctx: &Ctx<'_>, op_index: usize) -> FusedGroup {
    let op = &ctx.model.ops()[op_index];
    let mut g = FusedGroup::anchor(op_index, op);
    let input0 = op.inputs.first().copied().unwrap_or(u32::MAX);

    let Some(prod) = ctx.producer(input0) else { return g };
    // The fold target may already be owned by a conv (pattern (e)) or an
    // elementwise chain (pattern (b)) that ran earlier in execution order —
    // then the pool reads its output normally and must not double-absorb.
    if ctx.consumed.get(prod).copied().unwrap_or(false) {
        return g;
    }
    let Some(fold) = try_input_fold(ctx, prod) else { return g };

    g.inputs[0] = fold.folded_input_tensor;
    g.input_fold = Some(fold);
    g.absorbed_ops.push(prod);
    g.eliminated_tensors.push(input0);
    g
}

// ---------------------------------------------------------------------------
// Activation helpers
// ---------------------------------------------------------------------------

/// Map a standalone op code to a fused-activation kind.  Only the verified
/// schema codes 19/21/117; 101/102/103 (ABS/SPLIT_V/UNIQUE) are never
/// activations.
fn activation_kind(code: i32) -> Option<FusedActivationKind> {
    match code {
        RELU => Some(FusedActivationKind::Relu),
        RELU6 => Some(FusedActivationKind::Relu6),
        HARD_SWISH => Some(FusedActivationKind::HardSwish),
        _ => None,
    }
}

/// The explicit `fused_activation` field, restricted to the supported kinds
/// (1 = RELU, 3 = RELU6).  NONE (0) and unsupported codes (2, 4, 5) map to
/// `None` and pass through untouched.
fn field_activation(op: &ParsedOp<'_>) -> Option<FusedActivationKind> {
    let fused = fused_activation_field(op.options.as_ref())?;
    match fused {
        ACT_RELU => Some(FusedActivationKind::Relu),
        ACT_RELU6 => Some(FusedActivationKind::Relu6),
        _ => None,
    }
}

fn fused_activation_field(options: Option<&ParsedOptions>) -> Option<i8> {
    match options? {
        ParsedOptions::Conv2D { fused_activation, .. }
        | ParsedOptions::DepthwiseConv2D { fused_activation, .. }
        | ParsedOptions::FullyConnected { fused_activation, .. }
        | ParsedOptions::Pool2D { fused_activation, .. }
        | ParsedOptions::Add { fused_activation, .. }
        | ParsedOptions::Sub { fused_activation, .. }
        | ParsedOptions::Mul { fused_activation } => Some(*fused_activation),
        _ => None,
    }
}

/// T1.1 quant-identity gate: tensors `a` and `b` share the same scale AND
/// the same zero point.  Exact `f32` equality (not epsilon) — two distinct
/// scale values define distinct quantization grids, so any requantize
/// between them is a rounding step a pure clamp cannot reproduce.  Tensors
/// without quantization never pass.
fn quant_identity(ctx: &Ctx<'_>, a: u32, b: u32) -> bool {
    match (ctx.scale_of(a), ctx.zp_of(a), ctx.scale_of(b), ctx.zp_of(b)) {
        (Some(sa), Some(za), Some(sb), Some(zb)) => sa == sb && za == zb,
        _ => false,
    }
}

/// Build a [`FusedActivation`] from the activation's own output tensor quant
/// params — the range the fused epilogue must clamp to.
fn activation_from(ctx: &Ctx<'_>, kind: FusedActivationKind, out_tensor: u32) -> FusedActivation {
    let (min, max) = activation_range(kind, ctx.scale_of(out_tensor), ctx.zp_of(out_tensor));
    FusedActivation { kind, quantized_min: min, quantized_max: max }
}

/// TFLM `CalculateActivationRangeQuantized` for int8:
///
/// * RELU → `[max(-128, round(zp + 0/scale)), 127]`
/// * RELU6 → `[max(-128, round(zp)), min(127, round(zp + 6/scale))]`
/// * HARD_SWISH → `[-128, 127]`
fn activation_range(kind: FusedActivationKind, scale: Option<f32>, zp: Option<i64>) -> (i32, i32) {
    const QMIN: i32 = -128;
    const QMAX: i32 = 127;
    let scale = match scale {
        Some(s) if s > 0.0 => f64::from(s),
        _ => return (QMIN, QMAX),
    };
    let zp = zp.unwrap_or(0);
    let quantize = |real: f64| (zp as f64 + (real / scale).round()) as i32;
    match kind {
        FusedActivationKind::Relu => (QMIN.max(quantize(0.0)), QMAX),
        FusedActivationKind::Relu6 => (QMIN.max(quantize(0.0)), QMAX.min(quantize(6.0))),
        FusedActivationKind::HardSwish => (QMIN, QMAX),
    }
}

// ---------------------------------------------------------------------------
// Classification helpers
// ---------------------------------------------------------------------------

fn is_conv_family(code: i32) -> bool {
    matches!(code, CONV_2D | DEPTHWISE_CONV_2D | FULLY_CONNECTED)
}

fn is_elementwise(code: i32) -> bool {
    matches!(code, ADD | MUL | SUB)
}

fn is_pool_or_softmax(code: i32) -> bool {
    matches!(code, AVERAGE_POOL_2D | MAX_POOL_2D | SOFTMAX)
}

fn chain_kind(code: i32) -> Option<ElementwiseKind> {
    match code {
        ADD => Some(ElementwiseKind::Add),
        MUL => Some(ElementwiseKind::Mul),
        SUB => Some(ElementwiseKind::Sub),
        RELU => Some(ElementwiseKind::Relu),
        RELU6 => Some(ElementwiseKind::Relu6),
        HARD_SWISH => Some(ElementwiseKind::HardSwish),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Host-side quantize_multiplier (local copy — hematite-int8 is not a dep)
// ---------------------------------------------------------------------------

/// TFLM `QuantizeMultiplier`: f64 scale → (Q0.31 multiplier, shift) pair.
///
/// Mirrors `hematite-int8`'s host-gated implementation (frexp decomposition,
/// significand in [0.5, 1.0), carry fix at exactly 2^31, tiny scales flushed
/// to (0, 0)).  Host-side only — this proc-macro crate never ships device
/// code.
fn quantize_multiplier(scale: f64) -> (i32, i32) {
    if scale == 0.0 {
        return (0, 0);
    }
    let (sig, mut shift) = frexp(scale);
    let mut q_fixed = (sig * (1u64 << 31) as f64 + 0.5) as i64;
    if q_fixed == (1i64 << 31) {
        q_fixed /= 2;
        shift += 1;
    }
    if shift < -31 {
        return (0, 0);
    }
    (q_fixed as i32, shift)
}

/// Decompose a float64 into significand in `[0.5, 1.0)` and a binary
/// exponent (`std::frexp` semantics via IEEE 754 bit manipulation).
fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 {
        return (0.0, 0);
    }
    let bits = x.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let mantissa = bits & 0x000f_ffff_ffff_ffff;
    let sign = bits & 0x8000_0000_0000_0000;
    let frexp_significand_bits = sign | 0x3fe0_0000_0000_0000u64 | mantissa;
    (f64::from_bits(frexp_significand_bits), exponent - 1022)
}

// ---------------------------------------------------------------------------
// Unit tests — in-crate only (proc-macro restriction, E0477)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod opt_fusion {
    use super::*;
    use crate::flatbuffer;

    // ------------------------------------------------------------------
    // Minimal flatbuffer serializer for hand-built TFLite test models
    // (same proven Fb/Fv approach as layout.rs, extended with
    // quantization tables and BuiltinOptions tables).
    //
    // A table = i32 soffset to its vtable, then field values inline; the
    // vtable holds [vtable_len, table_size, per-field offsets].  Vectors,
    // strings, and tables are referenced by uoffset (relative u32) that
    // must point FORWARD — the builder emits each referencing table first
    // and patches its uoffset slots once the targets exist.  Slot
    // numbering follows schema.fbs (see flatbuffer.rs for the walker).
    // ------------------------------------------------------------------
    mod fb {
        /// A table field value: raw inline bytes, or a forward-uoffset to a
        /// target patched in after the target is emitted.
        pub(super) enum Fv {
            Raw(Vec<u8>),
            Ref,
        }

        #[derive(Clone, Copy)]
        pub(super) struct RefSlot {
            pub(super) field_pos: usize,
        }

        pub(super) struct Fb {
            pub(super) bytes: Vec<u8>,
        }

        impl Fb {
            pub(super) fn new() -> Self {
                let mut bytes = Vec::new();
                bytes.extend_from_slice(&[0u8; 8]); // root uoffset + identifier
                bytes[4..8].copy_from_slice(b"TFL3");
                Self { bytes }
            }

            pub(super) fn pos(&self) -> usize {
                self.bytes.len()
            }

            pub(super) fn align4(&mut self) {
                while self.bytes.len() % 4 != 0 {
                    self.bytes.push(0);
                }
            }

            pub(super) fn table(&mut self, fields: &[(u32, Fv)]) -> (usize, Vec<RefSlot>) {
                self.align4();
                let table_pos = self.pos();
                let mut field_offs = Vec::with_capacity(fields.len());
                let mut table_size = 4u32;
                for (_, v) in fields {
                    field_offs.push(table_size);
                    table_size += match v {
                        Fv::Raw(b) => b.len() as u32,
                        Fv::Ref => 4,
                    };
                }
                self.bytes.extend_from_slice(&[0u8; 4]);
                let mut ref_slots = Vec::new();
                for ((_, v), off) in fields.iter().zip(&field_offs) {
                    match v {
                        Fv::Raw(b) => self.bytes.extend_from_slice(b),
                        Fv::Ref => {
                            let field_pos = table_pos + *off as usize;
                            self.bytes.extend_from_slice(&0u32.to_le_bytes());
                            ref_slots.push(RefSlot { field_pos });
                        }
                    }
                }
                self.align4();
                let vt_pos = self.pos();
                let nfields = fields
                    .iter()
                    .map(|(i, _)| *i as usize)
                    .max()
                    .map_or(0, |m| m + 1);
                let vt_len = u16::try_from(nfields + 2).expect("vtable length fits u16");
                let table_size = u16::try_from(table_size).expect("table size fits u16");
                self.bytes.extend_from_slice(&(vt_len * 2).to_le_bytes());
                self.bytes.extend_from_slice(&table_size.to_le_bytes());
                let mut vt = vec![0u16; nfields];
                for ((i, _), off) in fields.iter().zip(&field_offs) {
                    vt[*i as usize] = u16::try_from(*off).expect("field offset fits u16");
                }
                for o in vt {
                    self.bytes.extend_from_slice(&o.to_le_bytes());
                }
                let soff = (table_pos as u32).wrapping_sub(vt_pos as u32);
                self.bytes[table_pos..table_pos + 4].copy_from_slice(&soff.to_le_bytes());
                (table_pos, ref_slots)
            }

            pub(super) fn patch_ref(&mut self, slot: &RefSlot, target: usize) {
                let rel = u32::try_from(target - slot.field_pos)
                    .expect("uoffset target must follow its referencing field");
                self.bytes[slot.field_pos..slot.field_pos + 4].copy_from_slice(&rel.to_le_bytes());
            }

            pub(super) fn patch_vec_elem(&mut self, vec_pos: usize, elem_idx: usize, target: usize) {
                let elem_pos = vec_pos + 4 + elem_idx * 4;
                let rel = u32::try_from(target - elem_pos)
                    .expect("uoffset target must follow the vector");
                self.bytes[elem_pos..elem_pos + 4].copy_from_slice(&rel.to_le_bytes());
            }

            pub(super) fn vec_u32(&mut self, elems: &[u32]) -> usize {
                self.align4();
                let p = self.pos();
                self.bytes.extend_from_slice(&(elems.len() as u32).to_le_bytes());
                for e in elems {
                    self.bytes.extend_from_slice(&e.to_le_bytes());
                }
                p
            }

            pub(super) fn vec_i32(&mut self, elems: &[i32]) -> usize {
                self.align4();
                let p = self.pos();
                self.bytes.extend_from_slice(&(elems.len() as u32).to_le_bytes());
                for e in elems {
                    self.bytes.extend_from_slice(&e.to_le_bytes());
                }
                p
            }

            pub(super) fn vec_f32(&mut self, elems: &[f32]) -> usize {
                self.align4();
                let p = self.pos();
                self.bytes.extend_from_slice(&(elems.len() as u32).to_le_bytes());
                for e in elems {
                    self.bytes.extend_from_slice(&e.to_le_bytes());
                }
                p
            }

            pub(super) fn vec_i64(&mut self, elems: &[i64]) -> usize {
                self.align4();
                let p = self.pos();
                self.bytes.extend_from_slice(&(elems.len() as u32).to_le_bytes());
                for e in elems {
                    self.bytes.extend_from_slice(&e.to_le_bytes());
                }
                p
            }

            pub(super) fn vec_bytes(&mut self, data: &[u8]) -> usize {
                self.align4();
                let p = self.pos();
                self.bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
                self.bytes.extend_from_slice(data);
                p
            }

            pub(super) fn string(&mut self, s: &str) -> usize {
                self.align4();
                let p = self.pos();
                self.bytes.extend_from_slice(&(s.len() as u32).to_le_bytes());
                self.bytes.extend_from_slice(s.as_bytes());
                self.bytes.push(0);
                p
            }

            pub(super) fn finish(mut self, root: usize) -> Vec<u8> {
                self.bytes[0..4].copy_from_slice(&(root as u32).to_le_bytes());
                self.bytes
            }
        }
    }

    use fb::{Fb, Fv};

    // ------------------------------------------------------------------
    // Hand-built model spec (tensors with quant, ops with options)
    // ------------------------------------------------------------------

    /// Quantization of a hand-built test tensor.
    struct BuildQuant {
        scale: f32,
        zp: i64,
        /// `(scales, zero_points, quantized_dimension)` when per-channel.
        per_channel: Option<(Vec<f32>, Vec<i64>, i32)>,
    }

    impl BuildQuant {
        fn tensor(scale: f32, zp: i64) -> Self {
            BuildQuant { scale, zp, per_channel: None }
        }

        fn channel(scales: Vec<f32>, zps: Vec<i64>, dim: i32) -> Self {
            BuildQuant {
                scale: scales[0],
                zp: zps[0],
                per_channel: Some((scales, zps, dim)),
            }
        }
    }

    struct BuildTensor {
        shape: Vec<i32>,
        name: &'static str,
        data: Vec<u8>,
        quant: Option<BuildQuant>,
    }

    impl BuildTensor {
        fn activation(shape: &[i32], name: &'static str) -> Self {
            BuildTensor { shape: shape.to_vec(), name, data: Vec::new(), quant: None }
        }

        fn constant(shape: &[i32], name: &'static str, data: Vec<u8>) -> Self {
            BuildTensor { shape: shape.to_vec(), name, data, quant: None }
        }

        fn constant_quantized(
            shape: &[i32],
            name: &'static str,
            data: Vec<u8>,
            quant: BuildQuant,
        ) -> Self {
            BuildTensor { shape: shape.to_vec(), name, data, quant: Some(quant) }
        }

        fn quantized(shape: &[i32], name: &'static str, quant: BuildQuant) -> Self {
            BuildTensor { shape: shape.to_vec(), name, data: Vec::new(), quant: Some(quant) }
        }
    }

    /// Which BuiltinOptions table an op carries.
    #[derive(Clone, Copy)]
    enum OptionsKind {
        Conv2d,
        Depthwise,
        Fc,
        Pool,
        Softmax,
        Add,
        Sub,
        Mul,
    }

    struct BuildOptions {
        kind: OptionsKind,
        fused: i8,
    }

    struct BuildOp {
        builtin_code: i32,
        inputs: Vec<u32>,
        outputs: Vec<u32>,
        options: Option<BuildOptions>,
    }

    impl BuildOp {
        fn plain(code: i32, inputs: Vec<u32>, outputs: Vec<u32>) -> Self {
            BuildOp { builtin_code: code, inputs, outputs, options: None }
        }

        fn fused(code: i32, inputs: Vec<u32>, outputs: Vec<u32>, kind: OptionsKind, fused: i8) -> Self {
            BuildOp { builtin_code: code, inputs, outputs, options: Some(BuildOptions { kind, fused }) }
        }
    }

    /// Emit a `BuiltinOptions` table (scalar fields only).
    fn options_table(fb: &mut Fb, opt: &BuildOptions) -> (usize, Vec<fb::RefSlot>) {
        let i32b = |v: i32| v.to_le_bytes().to_vec();
        let f32b = |v: f32| v.to_le_bytes().to_vec();
        let byte = |v: i8| vec![v as u8];
        match opt.kind {
            OptionsKind::Conv2d => fb.table(&[
                (0, Fv::Raw(byte(0))), // padding SAME
                (1, Fv::Raw(i32b(1))),
                (2, Fv::Raw(i32b(1))),
                (3, Fv::Raw(byte(opt.fused))),
                (4, Fv::Raw(i32b(1))),
                (5, Fv::Raw(i32b(1))),
            ]),
            OptionsKind::Depthwise => fb.table(&[
                (0, Fv::Raw(byte(0))),
                (1, Fv::Raw(i32b(1))),
                (2, Fv::Raw(i32b(1))),
                (3, Fv::Raw(i32b(1))), // depth_multiplier
                (4, Fv::Raw(byte(opt.fused))),
                (5, Fv::Raw(i32b(1))),
                (6, Fv::Raw(i32b(1))),
            ]),
            OptionsKind::Fc => fb.table(&[
                (0, Fv::Raw(byte(opt.fused))),
                (1, Fv::Raw(byte(0))), // weights_format
                (2, Fv::Raw(vec![0u8])), // keep_num_dims
            ]),
            OptionsKind::Pool => fb.table(&[
                (0, Fv::Raw(byte(0))),
                (1, Fv::Raw(i32b(1))),
                (2, Fv::Raw(i32b(1))),
                (3, Fv::Raw(i32b(2))), // filter 2x2
                (4, Fv::Raw(i32b(2))),
                (5, Fv::Raw(byte(opt.fused))),
            ]),
            OptionsKind::Softmax => fb.table(&[(0, Fv::Raw(f32b(1.0)))]),
            OptionsKind::Add | OptionsKind::Sub => fb.table(&[
                (0, Fv::Raw(byte(opt.fused))),
                (1, Fv::Raw(vec![0u8])), // pot_scale_int16
            ]),
            OptionsKind::Mul => fb.table(&[(0, Fv::Raw(byte(opt.fused)))]),
        }
    }

    /// Emit a `QuantizationParameters` table for one tensor, returning the
    /// table position (its own scale/zp slots are patched internally).  The
    /// table is emitted first so its uoffset slots point forward to the
    /// scale/zp vectors written after it.
    fn quant_table(fb: &mut Fb, q: &BuildQuant) -> usize {
        let (scales, zps) = match &q.per_channel {
            Some((sc, z, _)) => (sc.clone(), z.clone()),
            None => (vec![q.scale], vec![q.zp]),
        };
        let mut fields = vec![
            (2u32, Fv::Ref), // scale
            (3u32, Fv::Ref), // zero_point
        ];
        if let Some((_, _, dim)) = &q.per_channel {
            fields.push((6, Fv::Raw(dim.to_le_bytes().to_vec())));
        }
        let (qpos, slots) = fb.table(&fields);
        let scale_vec = fb.vec_f32(&scales);
        let zp_vec = fb.vec_i64(&zps);
        fb.patch_ref(&slots[0], scale_vec);
        fb.patch_ref(&slots[1], zp_vec);
        qpos
    }

    /// Assemble a complete TFLite model flatbuffer.  All tensors are INT8
    /// (type byte 9).  Emission order: every uoffset points forward, so
    /// referencing tables are emitted before the data they reference and
    /// the uoffset slots are patched as the targets appear.
    fn build_model(
        tensors: Vec<BuildTensor>,
        ops: Vec<BuildOp>,
        inputs: Vec<u32>,
        outputs: Vec<u32>,
    ) -> Vec<u8> {
        let mut fb = Fb::new();

        let mut buffer_indices = vec![0u32; tensors.len()];
        let mut buffer_datas: Vec<Vec<u8>> = vec![Vec::new()];
        for (i, t) in tensors.iter().enumerate() {
            if !t.data.is_empty() {
                buffer_indices[i] =
                    u32::try_from(buffer_datas.len()).expect("buffer count fits u32");
                buffer_datas.push(t.data.clone());
            }
        }
        let buffer_count = buffer_datas.len();

        // 1. Model table (lowest address).
        let (model, slots) = fb.table(&[(1, Fv::Ref), (2, Fv::Ref), (4, Fv::Ref)]);
        let [s_opcodes, s_subgraphs, s_buffers] = [slots[0], slots[1], slots[2]];

        // 2. Subgraphs vector then the Subgraph table.
        let subgraphs_vec = fb.vec_u32(&[0u32; 1]);
        let (subgraph, slots) =
            fb.table(&[(0, Fv::Ref), (1, Fv::Ref), (2, Fv::Ref), (3, Fv::Ref)]);
        let [s_tensors, s_inputs, s_outputs, s_operators] =
            [slots[0], slots[1], slots[2], slots[3]];
        fb.patch_vec_elem(subgraphs_vec, 0, subgraph);

        // 3. Table-reference vectors (contents patched once tables exist).
        let tensors_vec = fb.vec_u32(&vec![0u32; tensors.len()]);
        let operators_vec = fb.vec_u32(&vec![0u32; ops.len()]);
        let opcodes_vec = fb.vec_u32(&vec![0u32; ops.len()]);
        let buffers_vec = fb.vec_u32(&vec![0u32; buffer_count]);

        // 4. Tensor tables — slots: shape, type, buffer_index, name (+quant).
        let mut tensor_slots = Vec::with_capacity(tensors.len());
        for (i, t) in tensors.iter().enumerate() {
            let mut fields = vec![
                (0u32, Fv::Ref),
                (1u32, Fv::Raw(vec![9u8])), // TensorType.INT8
                (2u32, Fv::Raw(buffer_indices[i].to_le_bytes().to_vec())),
                (3u32, Fv::Ref),
            ];
            if t.quant.is_some() {
                fields.push((4u32, Fv::Ref));
            }
            let (tp, slots) = fb.table(&fields);
            tensor_slots.push((tp, slots));
        }
        for (i, (tp, _)) in tensor_slots.iter().enumerate() {
            fb.patch_vec_elem(tensors_vec, i, *tp);
        }
        fb.patch_ref(&s_tensors, tensors_vec);

        // 5. Shape vectors, name strings, quant tables; patch tensor tables.
        for (i, t) in tensors.iter().enumerate() {
            let shape_pos = fb.vec_i32(&t.shape);
            let name_pos = fb.string(t.name);
            let quant_pos = t.quant.as_ref().map(|q| quant_table(&mut fb, q));
            let slots = &tensor_slots[i].1;
            fb.patch_ref(&slots[0], shape_pos);
            fb.patch_ref(&slots[1], name_pos);
            if let Some(qpos) = quant_pos {
                fb.patch_ref(&slots[2], qpos);
            }
        }

        // 6. Operator tables — slots: opcode, inputs, outputs (+options).
        let mut op_slots = Vec::with_capacity(ops.len());
        for (i, _op) in ops.iter().enumerate() {
            let opcode_index = u32::try_from(i).expect("op count fits u32");
            let mut fields = vec![
                (0u32, Fv::Raw(opcode_index.to_le_bytes().to_vec())),
                (1u32, Fv::Ref),
                (2u32, Fv::Ref),
            ];
            if ops[i].options.is_some() {
                fields.push((3u32, Fv::Raw(vec![1u8]))); // discriminator ≠ NONE
                fields.push((4u32, Fv::Ref));
            }
            let (op_pos, slots) = fb.table(&fields);
            op_slots.push((op_pos, slots));
        }
        for (i, (op_pos, _)) in op_slots.iter().enumerate() {
            fb.patch_vec_elem(operators_vec, i, *op_pos);
        }
        fb.patch_ref(&s_operators, operators_vec);

        // 7. Per-op input/output vectors, then options tables.
        for (i, op) in ops.iter().enumerate() {
            let inputs_vec = fb.vec_u32(&op.inputs);
            let outputs_vec = fb.vec_u32(&op.outputs);
            let slots = &op_slots[i].1;
            fb.patch_ref(&slots[0], inputs_vec);
            fb.patch_ref(&slots[1], outputs_vec);
        }
        for (i, op) in ops.iter().enumerate() {
            if let Some(opt) = &op.options {
                let (opos, _) = options_table(&mut fb, opt);
                let slots = &op_slots[i].1;
                fb.patch_ref(&slots[2], opos);
            }
        }

        // 8. One OperatorCode table per op (deprecated_builtin_code, field 0).
        let mut opcode_positions = Vec::with_capacity(ops.len());
        for op in &ops {
            let code = u8::try_from(op.builtin_code).expect("test opcode fits a byte");
            let (cp, _) = fb.table(&[(0u32, Fv::Raw(vec![code]))]);
            opcode_positions.push(cp);
        }
        for (i, cp) in opcode_positions.iter().enumerate() {
            fb.patch_vec_elem(opcodes_vec, i, *cp);
        }

        // 9. Buffer tables, then buffer data vectors.
        let mut buffer_positions = Vec::with_capacity(buffer_count);
        for _ in 0..buffer_count {
            let (bp, slots) = fb.table(&[(0u32, Fv::Ref)]);
            buffer_positions.push((bp, slots[0]));
        }
        for (i, (bp, _)) in buffer_positions.iter().enumerate() {
            fb.patch_vec_elem(buffers_vec, i, *bp);
        }
        fb.patch_ref(&s_buffers, buffers_vec);
        for (i, data) in buffer_datas.iter().enumerate() {
            let data_pos = fb.vec_bytes(data);
            fb.patch_ref(&buffer_positions[i].1, data_pos);
        }

        // 10. Subgraph inputs/outputs, patch the model root.
        let inputs_vec = fb.vec_u32(&inputs);
        let outputs_vec = fb.vec_u32(&outputs);
        fb.patch_ref(&s_inputs, inputs_vec);
        fb.patch_ref(&s_outputs, outputs_vec);
        fb.patch_ref(&s_opcodes, opcodes_vec);
        fb.patch_ref(&s_subgraphs, subgraphs_vec);
        fb.finish(model)
    }

    /// Build and parse a model, keeping the bytes alive for the borrow.
    /// `Box::leak` pins the fixture bytes for the process lifetime (tests
    /// only), letting the parsed model borrow them as `'static`.
    macro_rules! model {
        ($tensors:expr, $ops:expr, $inputs:expr, $outputs:expr) => {{
            let bytes: &'static [u8] =
                Box::leak(build_model($tensors, $ops, $inputs, $outputs).into_boxed_slice());
            let model = flatbuffer::parse(bytes).expect("hand-built model must parse");
            (bytes, model)
        }};
    }

    // Hand-computed TFLM QuantizeMultiplier results for the pattern-(e)
    // assertions (independent of the implementation under test).
    const QM_4_0: (i32, i32) = (1073741824, 3); // quantize_multiplier(4.0)
    const QM_2_0: (i32, i32) = (1073741824, 2); // quantize_multiplier(2.0)
    const QM_1_0: (i32, i32) = (1073741824, 1); // quantize_multiplier(1.0)
    const QM_HALF: (i32, i32) = (1073741824, 0); // quantize_multiplier(0.5)

    // ------------------------------------------------------------------
    // Pattern (a) — conv-family + activation epilogue
    // ------------------------------------------------------------------

    /// A minimal conv→relu graph with quantized tensors.  The conv output
    /// quant equals the activation output quant (T1.1 quant identity) so the
    /// standalone-activation absorption is exact and the tests exercise the
    /// clamp-range computation; the mismatch cases use
    /// [`conv_relu_model_quants`].
    fn conv_relu_model(
        relu_code: i32,
        act_scale: f32,
        act_zp: i64,
    ) -> (&'static [u8], flatbuffer::ParsedModel<'static>) {
        conv_relu_model_quants(relu_code, act_scale, act_zp, act_scale, act_zp)
    }

    /// A conv→activation graph with INDEPENDENT conv-output and
    /// activation-output quants (the scale/zero-point gate cases).
    fn conv_relu_model_quants(
        relu_code: i32,
        conv_scale: f32,
        conv_zp: i64,
        act_scale: f32,
        act_zp: i64,
    ) -> (&'static [u8], flatbuffer::ParsedModel<'static>) {
        model!(
            vec![
                BuildTensor::quantized(&[1, 4, 4, 8], "input", BuildQuant::tensor(1.0, 0)),
                BuildTensor::constant(&[16, 1, 1, 8], "conv/weights", vec![0u8; 16 * 8]),
                BuildTensor::constant(&[16], "conv/bias", vec![0u8; 16 * 4]),
                BuildTensor::quantized(
                    &[1, 4, 4, 16],
                    "conv/output",
                    BuildQuant::tensor(conv_scale, conv_zp),
                ),
                BuildTensor::quantized(
                    &[1, 4, 4, 16],
                    "act/output",
                    BuildQuant::tensor(act_scale, act_zp),
                ),
            ],
            vec![
                BuildOp::fused(CONV_2D, vec![0, 1, 2], vec![3], OptionsKind::Conv2d, ACT_NONE),
                BuildOp::plain(relu_code, vec![3], vec![4]),
            ],
            vec![0],
            vec![4]
        )
    }

    #[test]
    fn fuse_conv_absorbs_following_relu() {
        let (_bytes, model) = conv_relu_model(RELU, 1.0, 0);
        let schedule = fuse(&model);

        assert_eq!(schedule.total_ops, 2);
        assert_eq!(schedule.groups.len(), 1, "conv+relu must be one emitted call");
        assert_eq!(schedule.fused_op_count(), 1);

        let g = &schedule.groups[0];
        assert_eq!(g.anchor_op_index, 0);
        assert_eq!(g.anchor_builtin, CONV_2D);
        assert_eq!(g.absorbed_ops, vec![1]);
        assert_eq!(g.eliminated_tensors, vec![3]);
        assert_eq!(g.output_tensor, 4);

        let act = g.activation.expect("relu must fuse into the epilogue");
        assert_eq!(act.kind, FusedActivationKind::Relu);
        // CalculateActivationRangeQuantized(RELU, scale=1, zp=0) → [0, 127].
        assert_eq!((act.quantized_min, act.quantized_max), (0, 127));
        assert!(
            !g.requires_verification,
            "activation absorption is T1 semantics-preserving"
        );
    }

    #[test]
    fn fuse_conv_relu6_range_is_quantized_six() {
        let (_bytes, model) = conv_relu_model(RELU6, 1.0, 0);
        let schedule = fuse(&model);
        let act = schedule.groups[0].activation.expect("relu6 must fuse");
        assert_eq!(act.kind, FusedActivationKind::Relu6);
        // RELU6 → [max(-128, zp), min(127, round(zp + 6/scale))] = [0, 6].
        assert_eq!((act.quantized_min, act.quantized_max), (0, 6));
    }

    #[test]
    fn fuse_conv_hard_swish_is_full_range() {
        let (_bytes, model) = conv_relu_model(HARD_SWISH, 1.0, 0);
        let schedule = fuse(&model);
        let act = schedule.groups[0].activation.expect("hard swish must fuse");
        assert_eq!(act.kind, FusedActivationKind::HardSwish);
        assert_eq!((act.quantized_min, act.quantized_max), (-128, 127));
    }

    #[test]
    fn activation_range_honors_positive_zero_point() {
        let (_bytes, model) = conv_relu_model(RELU, 1.0, 5);
        let schedule = fuse(&model);
        let act = schedule.groups[0].activation.expect("relu must fuse");
        // quantize(0.0) = zp + round(0/1) = 5 → [max(-128, 5), 127] = [5, 127].
        assert_eq!((act.quantized_min, act.quantized_max), (5, 127));
    }

    #[test]
    fn scale_changing_activation_not_absorbed() {
        // conv output scale 0.25 vs activation output scale 1.0: the
        // standalone relu is a full requantize, so absorbing it as an
        // `activation_range` clamp would NOT be bit-exact.  The T1.1
        // scale-identity gate refuses → the relu stays a separate call.
        let (_bytes, model) = conv_relu_model_quants(RELU, 0.25, 0, 1.0, 0);
        let schedule = fuse(&model);
        assert_eq!(
            schedule.groups.len(),
            2,
            "scale-changing relu must stay a separate call"
        );
        let conv = &schedule.groups[0];
        assert!(
            conv.activation.is_none(),
            "standalone relu must not be absorbed as a clamp"
        );
        assert!(conv.absorbed_ops.is_empty());
        assert!(conv.eliminated_tensors.is_empty());
        assert_eq!(conv.output_tensor, 3);
        assert_eq!(schedule.groups[1].anchor_op_index, 1, "relu keeps its own call");
        assert_eq!(schedule.fused_op_count(), 0);
    }

    #[test]
    fn zero_point_only_difference_activation_not_absorbed() {
        // Equal scales (1.0), differing zero points (conv 0, relu 5): the
        // unfused relu computes `max(x, 0) + 5` while the fused clamp would
        // give `clamp(x, 5, 127)` — divergent, so absorption must be refused.
        let (_bytes, model) = conv_relu_model_quants(RELU, 1.0, 0, 1.0, 5);
        let schedule = fuse(&model);
        assert_eq!(
            schedule.groups.len(),
            2,
            "zero-point-changing relu must stay a separate call"
        );
        let conv = &schedule.groups[0];
        assert!(conv.activation.is_none());
        assert!(conv.absorbed_ops.is_empty());
        assert_eq!(schedule.groups[1].anchor_op_index, 1);
        assert_eq!(schedule.fused_op_count(), 0);
    }

    #[test]
    fn fuse_depthwise_and_fc_activations() {
        let (_bytes, model) = model!(
            vec![
                BuildTensor::quantized(&[1, 4, 4, 8], "input", BuildQuant::tensor(1.0, 0)),
                BuildTensor::constant(&[1, 3, 3, 16], "dw/weights", vec![0u8; 1 * 3 * 3 * 16]),
                BuildTensor::constant(&[16], "dw/bias", vec![0u8; 16 * 4]),
                BuildTensor::quantized(&[1, 2, 2, 16], "dw/output", BuildQuant::tensor(0.5, 0)),
                BuildTensor::quantized(&[1, 2, 2, 16], "act/output", BuildQuant::tensor(0.5, 0)),
            ],
            vec![
                BuildOp::fused(DEPTHWISE_CONV_2D, vec![0, 1, 2], vec![3], OptionsKind::Depthwise, ACT_NONE),
                BuildOp::plain(RELU, vec![3], vec![4]),
            ],
            vec![0],
            vec![4]
        );
        let schedule = fuse(&model);
        assert_eq!(schedule.groups.len(), 1);
        assert_eq!(schedule.groups[0].absorbed_ops, vec![1]);
        assert!(schedule.groups[0].activation.is_some());

        // FullyConnected → relu6.
        let (_bytes, model) = model!(
            vec![
                BuildTensor::quantized(&[1, 8], "input", BuildQuant::tensor(1.0, 0)),
                BuildTensor::constant(&[4, 8], "fc/weights", vec![0u8; 4 * 8]),
                BuildTensor::constant(&[4], "fc/bias", vec![0u8; 4 * 4]),
                BuildTensor::quantized(&[1, 4], "fc/output", BuildQuant::tensor(0.5, 0)),
                BuildTensor::quantized(&[1, 4], "act/output", BuildQuant::tensor(0.5, 0)),
            ],
            vec![
                BuildOp::fused(FULLY_CONNECTED, vec![0, 1, 2], vec![3], OptionsKind::Fc, ACT_NONE),
                BuildOp::plain(RELU6, vec![3], vec![4]),
            ],
            vec![0],
            vec![4]
        );
        let schedule = fuse(&model);
        assert_eq!(schedule.groups.len(), 1);
        assert!(matches!(
            schedule.groups[0].activation,
            Some(FusedActivation { kind: FusedActivationKind::Relu6, .. })
        ));
    }

    #[test]
    fn explicit_fused_activation_field_preferred_over_standalone_relu() {
        // Conv carries fused_activation=RELU; a redundant standalone relu
        // follows.  The explicit field wins: the conv group records the
        // activation and the standalone relu is NOT absorbed.
        let (_bytes, model) = model!(
            vec![
                BuildTensor::quantized(&[1, 4, 4, 8], "input", BuildQuant::tensor(1.0, 0)),
                BuildTensor::constant(&[16, 1, 1, 8], "conv/weights", vec![0u8; 16 * 8]),
                BuildTensor::constant(&[16], "conv/bias", vec![0u8; 16 * 4]),
                BuildTensor::quantized(&[1, 4, 4, 16], "conv/output", BuildQuant::tensor(1.0, 0)),
                BuildTensor::quantized(&[1, 4, 4, 16], "relu/output", BuildQuant::tensor(1.0, 0)),
            ],
            vec![
                BuildOp::fused(CONV_2D, vec![0, 1, 2], vec![3], OptionsKind::Conv2d, ACT_RELU),
                BuildOp::plain(RELU, vec![3], vec![4]),
            ],
            vec![0],
            vec![4]
        );
        let schedule = fuse(&model);
        assert_eq!(schedule.groups.len(), 2, "standalone relu stays a separate call");
        let conv = &schedule.groups[0];
        assert!(conv.absorbed_ops.is_empty(), "no standalone activation absorbed");
        let act = conv.activation.expect("field activation recorded");
        assert_eq!(act.kind, FusedActivationKind::Relu);
        assert_eq!((act.quantized_min, act.quantized_max), (0, 127));
        // The standalone relu keeps its own group (anchor op 1).
        assert_eq!(schedule.groups[1].anchor_op_index, 1);
    }

    #[test]
    fn unsupported_fused_field_codes_pass_through() {
        // fused_activation = TANH(4): not in {1, 3} → the pass never claims it.
        let (_bytes, model) = model!(
            vec![
                BuildTensor::quantized(&[1, 4, 4, 8], "input", BuildQuant::tensor(1.0, 0)),
                BuildTensor::constant(&[16, 1, 1, 8], "conv/weights", vec![0u8; 16 * 8]),
                BuildTensor::constant(&[16], "conv/bias", vec![0u8; 16 * 4]),
                BuildTensor::quantized(&[1, 4, 4, 16], "conv/output", BuildQuant::tensor(0.25, 0)),
            ],
            vec![BuildOp::fused(CONV_2D, vec![0, 1, 2], vec![3], OptionsKind::Conv2d, 4)],
            vec![0],
            vec![3]
        );
        let schedule = fuse(&model);
        assert_eq!(schedule.groups.len(), 1);
        assert!(schedule.groups[0].activation.is_none());
        assert!(schedule.groups[0].absorbed_ops.is_empty());
    }

    #[test]
    fn abs_is_not_treated_as_activation() {
        // BuiltinOperator 101 = ABS in this schema era, NOT relu.  It must
        // never be absorbed as a fused activation.
        let (_bytes, model) = model!(
            vec![
                BuildTensor::quantized(&[1, 4, 4, 8], "input", BuildQuant::tensor(1.0, 0)),
                BuildTensor::constant(&[16, 1, 1, 8], "conv/weights", vec![0u8; 16 * 8]),
                BuildTensor::constant(&[16], "conv/bias", vec![0u8; 16 * 4]),
                BuildTensor::quantized(&[1, 4, 4, 16], "conv/output", BuildQuant::tensor(0.25, 0)),
                BuildTensor::quantized(&[1, 4, 4, 16], "abs/output", BuildQuant::tensor(0.25, 0)),
            ],
            vec![
                BuildOp::fused(CONV_2D, vec![0, 1, 2], vec![3], OptionsKind::Conv2d, ACT_NONE),
                BuildOp::plain(101, vec![3], vec![4]),
            ],
            vec![0],
            vec![4]
        );
        let schedule = fuse(&model);
        assert_eq!(schedule.groups.len(), 2, "ABS stays a separate call");
        assert!(schedule.groups[0].activation.is_none());
        assert!(schedule.groups[0].absorbed_ops.is_empty());
    }

    // ------------------------------------------------------------------
    // Pattern (b) — elementwise chains
    // ------------------------------------------------------------------

    #[test]
    fn fuse_elementwise_chain_add_relu_mul_hardswish() {
        let (_bytes, model) = model!(
            vec![
                BuildTensor::quantized(&[1, 4], "a", BuildQuant::tensor(1.0, 0)),
                BuildTensor::quantized(&[1, 4], "b", BuildQuant::tensor(1.0, 0)),
                BuildTensor::quantized(&[1, 4], "add/out", BuildQuant::tensor(1.0, 0)),
                BuildTensor::quantized(&[1, 4], "relu/out", BuildQuant::tensor(1.0, 0)),
                BuildTensor::constant(&[1], "mul/scale", vec![1, 1, 1, 1]),
                BuildTensor::quantized(&[1, 4], "mul/out", BuildQuant::tensor(0.5, 0)),
                BuildTensor::quantized(&[1, 4], "hs/out", BuildQuant::tensor(0.5, 0)),
            ],
            vec![
                BuildOp::plain(ADD, vec![0, 1], vec![2]),
                BuildOp::plain(RELU, vec![2], vec![3]),
                BuildOp::plain(MUL, vec![3, 4], vec![5]),
                BuildOp::plain(HARD_SWISH, vec![5], vec![6]),
            ],
            vec![0, 1],
            vec![6]
        );
        let schedule = fuse(&model);

        assert_eq!(schedule.total_ops, 4);
        assert_eq!(schedule.groups.len(), 1, "whole chain is one emitted call");
        assert_eq!(schedule.fused_op_count(), 3);

        let g = &schedule.groups[0];
        assert_eq!(g.anchor_op_index, 0);
        assert_eq!(g.anchor_builtin, ADD);
        assert_eq!(g.absorbed_ops, vec![1, 2, 3]);
        assert_eq!(g.eliminated_tensors, vec![2, 3, 5]);
        assert_eq!(g.output_tensor, 6);

        let chain = &g.elementwise_chain;
        assert_eq!(chain.len(), 3);
        assert_eq!(chain[0].kind, ElementwiseKind::Relu);
        assert_eq!(chain[0].operand_tensor, u32::MAX);
        assert_eq!(chain[1].kind, ElementwiseKind::Mul);
        assert_eq!(chain[1].operand_tensor, 4);
        assert!((chain[1].output_scale - 0.5).abs() < 1e-6);
        assert_eq!(chain[2].kind, ElementwiseKind::HardSwish);
        assert_eq!(chain[2].operand_tensor, u32::MAX);
        assert!(
            !g.requires_verification,
            "elementwise chains are T1 semantics-preserving"
        );
    }

    #[test]
    fn two_step_chain_keeps_per_step_requantize_params() {
        // sub → mul → mul: two consecutive scale-changing muls (× ones@1.0,
        // 1.0 → 0.25 → 0.5).  Each chain step requantizes with NON-identity
        // scales and must carry its OWN output ratio — step 0 is
        // qm(1.0·1.0/0.25) = qm(4.0), step 1 is qm(0.25·1.0/0.5) = qm(0.5),
        // never collapsed into a single multiplier.
        let (_bytes, model) = model!(
            vec![
                BuildTensor::quantized(&[1, 4], "a", BuildQuant::tensor(1.0, 0)),
                BuildTensor::quantized(&[1, 4], "b", BuildQuant::tensor(0.5, 0)),
                BuildTensor::quantized(&[1, 4], "sub/out", BuildQuant::tensor(1.0, 0)),
                BuildTensor::constant_quantized(
                    &[1],
                    "ones",
                    vec![1, 1, 1, 1],
                    BuildQuant::tensor(1.0, 0),
                ),
                BuildTensor::quantized(&[1, 4], "mul1/out", BuildQuant::tensor(0.25, 0)),
                BuildTensor::quantized(&[1, 4], "mul2/out", BuildQuant::tensor(0.5, 0)),
            ],
            vec![
                BuildOp::plain(SUB, vec![0, 1], vec![2]),
                BuildOp::plain(MUL, vec![2, 3], vec![4]),
                BuildOp::plain(MUL, vec![4, 3], vec![5]),
            ],
            vec![0, 1],
            vec![5]
        );
        let schedule = fuse(&model);
        assert_eq!(schedule.groups.len(), 1, "three-op chain is one emitted call");
        let g = &schedule.groups[0];
        let chain = &g.elementwise_chain;
        assert_eq!(chain.len(), 2);

        // Step 0 (MUL): output ratio 1.0·1.0/0.25 = 4.0; offsets zero.
        let s0 = chain[0].requantize;
        assert_eq!(s0.left_shift, 0);
        assert_eq!((s0.output_multiplier, s0.output_shift), QM_4_0);
        assert_eq!(s0.input1_multiplier, 0);
        assert_eq!(s0.input2_multiplier, 0);
        assert_eq!(s0.input1_offset, 0);
        assert_eq!(s0.input2_offset, 0);
        assert_eq!(s0.output_offset, 0);

        // Step 1 (MUL): output ratio 0.25·1.0/0.5 = 0.5 — a DIFFERENT
        // multiplier than step 0, carried per step (never collapsed).
        let s1 = chain[1].requantize;
        assert_eq!(s1.left_shift, 0);
        assert_eq!((s1.output_multiplier, s1.output_shift), QM_HALF);
        assert_eq!(s1.input1_multiplier, 0);
        assert_eq!(s1.input2_multiplier, 0);
        assert_ne!(
            (s0.output_multiplier, s0.output_shift),
            (s1.output_multiplier, s1.output_shift),
            "steps must keep distinct requantize params (no collapsing)"
        );
        assert!(
            !g.requires_verification,
            "elementwise chains are T1 semantics-preserving"
        );
    }

    #[test]
    fn elementwise_chain_stops_at_multi_consumer() {
        // add→relu, then relu's output feeds TWO ops → the chain stops after
        // relu; the two consumers stay separate calls.
        let (_bytes, model) = model!(
            vec![
                BuildTensor::quantized(&[1, 4], "a", BuildQuant::tensor(1.0, 0)),
                BuildTensor::quantized(&[1, 4], "b", BuildQuant::tensor(1.0, 0)),
                BuildTensor::quantized(&[1, 4], "add/out", BuildQuant::tensor(1.0, 0)),
                BuildTensor::quantized(&[1, 4], "relu/out", BuildQuant::tensor(1.0, 0)),
                BuildTensor::constant(&[1], "mul/scale", vec![1, 1, 1, 1]),
                BuildTensor::quantized(&[1, 4], "mul/out", BuildQuant::tensor(0.5, 0)),
                BuildTensor::quantized(&[1, 4], "c", BuildQuant::tensor(1.0, 0)),
                BuildTensor::quantized(&[1, 4], "add2/out", BuildQuant::tensor(1.0, 0)),
            ],
            vec![
                BuildOp::plain(ADD, vec![0, 1], vec![2]),
                BuildOp::plain(RELU, vec![2], vec![3]),
                BuildOp::plain(MUL, vec![3, 4], vec![5]),
                BuildOp::plain(ADD, vec![3, 6], vec![7]),
            ],
            vec![0, 1, 6],
            vec![5, 7]
        );
        let schedule = fuse(&model);

        assert_eq!(schedule.total_ops, 4);
        assert_eq!(schedule.groups.len(), 3, "chain folds only add+relu");
        assert_eq!(schedule.fused_op_count(), 1);
        assert_eq!(schedule.groups[0].elementwise_chain.len(), 1);
        assert_eq!(schedule.groups[0].absorbed_ops, vec![1]);
        // The two consumers of relu's output remain standalone anchors.
        assert_eq!(schedule.groups[1].anchor_op_index, 2);
        assert_eq!(schedule.groups[2].anchor_op_index, 3);
    }

    #[test]
    fn elementwise_chain_refuses_late_operand() {
        // add→mul where the mul's operand is produced AFTER the add anchor:
        // the fused loop would read a tensor that does not exist yet.
        let (_bytes, model) = model!(
            vec![
                BuildTensor::quantized(&[1, 4], "a", BuildQuant::tensor(1.0, 0)),
                BuildTensor::quantized(&[1, 4], "b", BuildQuant::tensor(1.0, 0)),
                BuildTensor::quantized(&[1, 4], "add/out", BuildQuant::tensor(1.0, 0)),
                BuildTensor::constant(&[16, 1, 1, 4], "conv/weights", vec![0u8; 16 * 4]),
                BuildTensor::constant(&[16], "conv/bias", vec![0u8; 16 * 4]),
                BuildTensor::quantized(&[1, 4, 4, 16], "conv/out", BuildQuant::tensor(0.5, 0)),
                BuildTensor::quantized(&[1, 4], "mul/out", BuildQuant::tensor(0.5, 0)),
            ],
            vec![
                BuildOp::plain(ADD, vec![0, 1], vec![2]),
                BuildOp::fused(CONV_2D, vec![0, 3, 4], vec![5], OptionsKind::Conv2d, ACT_NONE),
                BuildOp::plain(MUL, vec![2, 5], vec![6]),
            ],
            vec![0, 1],
            vec![6]
        );
        let schedule = fuse(&model);
        let add_group = &schedule.groups[0];
        assert_eq!(add_group.anchor_builtin, ADD);
        assert!(add_group.elementwise_chain.is_empty(), "late operand must block the chain");
        assert!(add_group.absorbed_ops.is_empty());
    }

    // ------------------------------------------------------------------
    // Pattern (c) — residual-add groups
    // ------------------------------------------------------------------

    #[test]
    fn fuse_residual_add_conv_relu() {
        // residual producer (conv, op0) runs before the conv (op1); the add
        // (op2) and relu (op3) fuse into op1's one-pass epilogue.
        let (_bytes, model) = model!(
            vec![
                BuildTensor::quantized(&[1, 4, 4, 8], "in0", BuildQuant::tensor(1.0, 0)),
                BuildTensor::constant(&[8, 1, 1, 8], "w0", vec![0u8; 8 * 8]),
                BuildTensor::constant(&[8], "b0", vec![0u8; 8 * 4]),
                BuildTensor::quantized(&[1, 4, 4, 8], "residual", BuildQuant::tensor(0.5, 0)),
                BuildTensor::quantized(&[1, 4, 4, 8], "in1", BuildQuant::tensor(1.0, 0)),
                BuildTensor::constant(&[16, 1, 1, 8], "w1", vec![0u8; 16 * 8]),
                BuildTensor::constant(&[16], "b1", vec![0u8; 16 * 4]),
                BuildTensor::quantized(&[1, 4, 4, 16], "conv/out", BuildQuant::tensor(0.5, 0)),
                BuildTensor::quantized(&[1, 4, 4, 16], "add/out", BuildQuant::tensor(0.5, 0)),
                BuildTensor::quantized(&[1, 4, 4, 16], "relu/out", BuildQuant::tensor(0.5, 0)),
            ],
            vec![
                BuildOp::fused(CONV_2D, vec![0, 1, 2], vec![3], OptionsKind::Conv2d, ACT_NONE),
                BuildOp::fused(CONV_2D, vec![4, 5, 6], vec![7], OptionsKind::Conv2d, ACT_NONE),
                BuildOp::plain(ADD, vec![7, 3], vec![8]),
                BuildOp::plain(RELU, vec![8], vec![9]),
            ],
            vec![0, 4],
            vec![9]
        );
        let schedule = fuse(&model);

        assert_eq!(schedule.total_ops, 4);
        // op0 (residual producer) stays a call; op1+add+relu fuse into one.
        assert_eq!(schedule.groups.len(), 2);
        assert_eq!(schedule.fused_op_count(), 2);

        let fused = &schedule.groups[1];
        assert_eq!(fused.anchor_op_index, 1);
        let ra = fused.residual_add.as_ref().expect("residual add must fuse");
        assert_eq!(ra.op_index, 2);
        assert_eq!(ra.residual_tensor, 3);
        assert!((ra.output_scale - 0.5).abs() < 1e-6);
        assert_eq!(fused.absorbed_ops, vec![2, 3]);
        assert_eq!(fused.eliminated_tensors, vec![7, 8]);
        assert_eq!(fused.output_tensor, 9);
        let act = fused.activation.expect("trailing relu fuses");
        assert_eq!((act.quantized_min, act.quantized_max), (0, 127));
    }

    #[test]
    fn residual_add_refused_when_residual_late_or_model_io() {
        // (a) Residual produced AFTER the conv → no fold for that conv.
        let (_bytes, m) = model!(
            vec![
                BuildTensor::quantized(&[1, 4, 4, 8], "in0", BuildQuant::tensor(1.0, 0)),
                BuildTensor::constant(&[8, 1, 1, 8], "w0", vec![0u8; 8 * 8]),
                BuildTensor::constant(&[8], "b0", vec![0u8; 8 * 4]),
                BuildTensor::quantized(&[1, 4, 4, 8], "conv0/out", BuildQuant::tensor(0.5, 0)),
                BuildTensor::quantized(&[1, 4, 4, 8], "in1", BuildQuant::tensor(1.0, 0)),
                BuildTensor::constant(&[8, 1, 1, 8], "w1", vec![0u8; 8 * 8]),
                BuildTensor::constant(&[8], "b1", vec![0u8; 8 * 4]),
                BuildTensor::quantized(&[1, 4, 4, 8], "residual", BuildQuant::tensor(0.5, 0)),
                BuildTensor::quantized(&[1, 4, 4, 8], "add/out", BuildQuant::tensor(0.5, 0)),
                BuildTensor::quantized(&[1, 4, 4, 8], "relu/out", BuildQuant::tensor(0.5, 0)),
            ],
            vec![
                BuildOp::fused(CONV_2D, vec![0, 1, 2], vec![3], OptionsKind::Conv2d, ACT_NONE),
                BuildOp::fused(CONV_2D, vec![4, 5, 6], vec![7], OptionsKind::Conv2d, ACT_NONE),
                BuildOp::plain(ADD, vec![3, 7], vec![8]),
                BuildOp::plain(RELU, vec![8], vec![9]),
            ],
            vec![0, 4],
            vec![9]
        );
        let schedule = fuse(&m);
        // conv0's residual candidate (t7) is produced at op1 ≥ op0 → refused.
        let conv0 = &schedule.groups[0];
        assert_eq!(conv0.anchor_op_index, 0);
        assert!(conv0.residual_add.is_none(), "late residual must not fuse");
        assert!(conv0.absorbed_ops.is_empty());

        // (b) Residual is a model input → refused.
        let (_bytes, m) = model!(
            vec![
                BuildTensor::quantized(&[1, 4, 4, 8], "residual_in", BuildQuant::tensor(1.0, 0)),
                BuildTensor::quantized(&[1, 4, 4, 8], "in1", BuildQuant::tensor(1.0, 0)),
                BuildTensor::constant(&[16, 1, 1, 8], "w1", vec![0u8; 16 * 8]),
                BuildTensor::constant(&[16], "b1", vec![0u8; 16 * 4]),
                BuildTensor::quantized(&[1, 4, 4, 16], "conv/out", BuildQuant::tensor(0.5, 0)),
                BuildTensor::quantized(&[1, 4, 4, 16], "add/out", BuildQuant::tensor(0.5, 0)),
            ],
            vec![
                BuildOp::fused(CONV_2D, vec![1, 2, 3], vec![4], OptionsKind::Conv2d, ACT_NONE),
                BuildOp::plain(ADD, vec![4, 0], vec![5]),
            ],
            vec![0, 1],
            vec![5]
        );
        let schedule = fuse(&m);
        assert!(schedule.groups[0].residual_add.is_none(), "model-input residual must not fuse");
    }

    // ------------------------------------------------------------------
    // Pattern (d) — pool/softmax absorbs preceding mul/sub
    // ------------------------------------------------------------------

    #[test]
    fn pool_absorbs_preceding_mul_scale() {
        let (_bytes, model) = model!(
            vec![
                BuildTensor::quantized(&[1, 4, 4, 8], "x", BuildQuant::tensor(1.0, 0)),
                BuildTensor::constant(&[1], "mul/scale", vec![1, 1, 1, 1]),
                BuildTensor::quantized(&[1, 4, 4, 8], "mul/out", BuildQuant::tensor(0.5, 0)),
                BuildTensor::quantized(&[1, 2, 2, 8], "pool/out", BuildQuant::tensor(0.5, 0)),
            ],
            vec![
                BuildOp::plain(MUL, vec![0, 1], vec![2]),
                BuildOp::fused(AVERAGE_POOL_2D, vec![2], vec![3], OptionsKind::Pool, ACT_NONE),
            ],
            vec![0],
            vec![3]
        );
        let schedule = fuse(&model);

        assert_eq!(schedule.total_ops, 2);
        assert_eq!(schedule.groups.len(), 1, "mul must not form its own call");
        assert_eq!(schedule.fused_op_count(), 1);

        let g = &schedule.groups[0];
        assert_eq!(g.anchor_op_index, 1);
        assert_eq!(g.anchor_builtin, AVERAGE_POOL_2D);
        let fold = g.input_fold.as_ref().expect("mul must fold into the pool");
        assert_eq!(fold.op_index, 0);
        assert_eq!(fold.builtin, MUL);
        assert_eq!(fold.folded_input_tensor, 0);
        assert_eq!(fold.operand_tensor, 1);
        // folded_scale = s_mul_out / s_in = 0.5 / 1.0.
        assert!((fold.folded_scale - 0.5).abs() < 1e-6);
        assert_eq!(fold.input_zero_point, 0);
        assert_eq!(g.inputs[0], 0, "kernel reads the pre-mul tensor");
        assert_eq!(g.eliminated_tensors, vec![2]);
        assert!(
            g.requires_verification,
            "input folds are algebraically transformative T2 groups"
        );
    }

    #[test]
    fn softmax_absorbs_preceding_sub_constant() {
        // sub(x, q=10) with operand scale 0.1, zp 0 → the folded real-domain
        // constant is c = 0.1·(10 − 0) = 1.0, absorbed into the softmax's
        // input offset math.
        let (_bytes, model) = model!(
            vec![
                BuildTensor::quantized(&[1, 8], "x", BuildQuant::tensor(1.0, 0)),
                BuildTensor::constant_quantized(
                    &[1],
                    "sub/const",
                    vec![10u8],
                    BuildQuant::tensor(0.1, 0),
                ),
                BuildTensor::quantized(&[1, 8], "sub/out", BuildQuant::tensor(1.0, 0)),
                BuildTensor::quantized(&[1, 8], "softmax/out", BuildQuant::tensor(1.0, 0)),
            ],
            vec![
                BuildOp::plain(SUB, vec![0, 1], vec![2]),
                BuildOp::fused(SOFTMAX, vec![2], vec![3], OptionsKind::Softmax, 0),
            ],
            vec![0],
            vec![3]
        );
        let schedule = fuse(&model);
        assert_eq!(schedule.total_ops, 2);
        assert_eq!(schedule.groups.len(), 1, "sub must not form its own call");
        assert_eq!(schedule.fused_op_count(), 1);

        let g = &schedule.groups[0];
        let fold = g.input_fold.as_ref().expect("sub must fold into softmax");
        assert_eq!(fold.op_index, 0);
        assert_eq!(fold.builtin, SUB);
        assert_eq!(fold.folded_input_tensor, 0, "softmax reads the pre-sub tensor");
        assert_eq!(fold.operand_tensor, 1);
        assert!((fold.folded_scale - 1.0).abs() < 1e-4, "constant = 0.1·10");
        assert_eq!(g.inputs[0], 0);
        assert_eq!(g.eliminated_tensors, vec![2]);
    }

    #[test]
    fn mul_with_two_consumers_not_absorbed() {
        // mul feeds BOTH a pool and an add → HasOneUse fails → no fold; the
        // mul keeps its own call.
        let (_bytes, model) = model!(
            vec![
                BuildTensor::quantized(&[1, 4, 4, 8], "x", BuildQuant::tensor(1.0, 0)),
                BuildTensor::constant(&[1], "mul/scale", vec![1, 1, 1, 1]),
                BuildTensor::quantized(&[1, 4, 4, 8], "mul/out", BuildQuant::tensor(0.5, 0)),
                BuildTensor::quantized(&[1, 2, 2, 8], "pool/out", BuildQuant::tensor(0.5, 0)),
                BuildTensor::quantized(&[1, 4, 4, 8], "y", BuildQuant::tensor(1.0, 0)),
                BuildTensor::quantized(&[1, 4, 4, 8], "add/out", BuildQuant::tensor(1.0, 0)),
            ],
            vec![
                BuildOp::plain(MUL, vec![0, 1], vec![2]),
                BuildOp::fused(AVERAGE_POOL_2D, vec![2], vec![3], OptionsKind::Pool, ACT_NONE),
                BuildOp::plain(ADD, vec![2, 4], vec![5]),
            ],
            vec![0, 4],
            vec![3, 5]
        );
        let schedule = fuse(&model);
        assert_eq!(schedule.groups.len(), 3, "no op may be absorbed");
        assert!(schedule.groups[1].input_fold.is_none(), "pool must not fold a multi-use mul");
        assert!(schedule.groups[1].absorbed_ops.is_empty());
        assert_eq!(schedule.fused_op_count(), 0);
    }

    // ------------------------------------------------------------------
    // Pattern (e) — requantize-scale constant folding
    // ------------------------------------------------------------------

    #[test]
    fn fuse_conv_mul_requantize_single_multiply_shift() {
        // conv (per-channel w scales [0.5, 0.25], s_in 1.0) → mul by an
        // all-ones constant at scale 1.0 → mul output scale 0.25, EQUAL to
        // the conv output scale (Metis G-3 identity — the mul's own
        // requantize multiplier is the identity, so the fold is exact).
        // Folded per-channel multipliers: qm(1·0.5/0.25)=qm(2.0),
        // qm(1·0.25/0.25)=qm(1.0).
        let (_bytes, model) = model!(
            vec![
                BuildTensor::quantized(&[1, 4, 4, 8], "input", BuildQuant::tensor(1.0, 0)),
                BuildTensor::constant_quantized(
                    &[2, 1, 1, 8],
                    "conv/weights",
                    vec![0u8; 2 * 8],
                    BuildQuant::channel(vec![0.5, 0.25], vec![0, 0], 0),
                ),
                BuildTensor::constant(&[2], "conv/bias", vec![0u8; 2 * 4]),
                BuildTensor::quantized(&[1, 4, 4, 2], "conv/out", BuildQuant::tensor(0.25, 0)),
                BuildTensor::constant_quantized(&[1], "ones", vec![1, 1, 1, 1], BuildQuant::tensor(1.0, 0)),
                BuildTensor::quantized(&[1, 4, 4, 2], "mul/out", BuildQuant::tensor(0.25, 0)),
            ],
            vec![
                BuildOp::fused(CONV_2D, vec![0, 1, 2], vec![3], OptionsKind::Conv2d, ACT_NONE),
                BuildOp::plain(MUL, vec![3, 4], vec![5]),
            ],
            vec![0],
            vec![5]
        );
        let schedule = fuse(&model);

        assert_eq!(schedule.total_ops, 2);
        assert_eq!(schedule.groups.len(), 1, "scale-mul must fold into the conv");
        assert_eq!(schedule.fused_op_count(), 1);

        let g = &schedule.groups[0];
        let fr = g.folded_requantize.as_ref().expect("requantize must fold");
        assert_eq!(fr.op_index, 1);
        assert!((fr.output_scale - 0.25).abs() < 1e-6);
        assert_eq!(fr.output_zero_point, 0);
        assert_eq!(fr.multipliers, vec![QM_2_0, QM_1_0]);
        assert_eq!(g.absorbed_ops, vec![1]);
        assert_eq!(g.eliminated_tensors, vec![3]);
        assert_eq!(g.output_tensor, 5);
        assert!(
            g.requires_verification,
            "requantize folds are algebraically transformative T2 groups"
        );
    }

    #[test]
    fn requantize_fold_composes_with_activation_absorption() {
        // conv → scale-mul → relu: the mul folds AND the relu fuses.  The
        // mul output scale equals the conv output scale (G-3 identity), and
        // the relu output quant equals the mul output quant (T1.1 identity)
        // so the activation absorption stays exact.
        let (_bytes, model) = model!(
            vec![
                BuildTensor::quantized(&[1, 4, 4, 8], "input", BuildQuant::tensor(1.0, 0)),
                BuildTensor::constant_quantized(
                    &[2, 1, 1, 8],
                    "conv/weights",
                    vec![0u8; 2 * 8],
                    BuildQuant::channel(vec![0.5, 0.25], vec![0, 0], 0),
                ),
                BuildTensor::constant(&[2], "conv/bias", vec![0u8; 2 * 4]),
                BuildTensor::quantized(&[1, 4, 4, 2], "conv/out", BuildQuant::tensor(0.25, 0)),
                BuildTensor::constant_quantized(&[1], "ones", vec![1, 1, 1, 1], BuildQuant::tensor(1.0, 0)),
                BuildTensor::quantized(&[1, 4, 4, 2], "mul/out", BuildQuant::tensor(0.25, 0)),
                BuildTensor::quantized(&[1, 4, 4, 2], "relu/out", BuildQuant::tensor(0.25, 0)),
            ],
            vec![
                BuildOp::fused(CONV_2D, vec![0, 1, 2], vec![3], OptionsKind::Conv2d, ACT_NONE),
                BuildOp::plain(MUL, vec![3, 4], vec![5]),
                BuildOp::plain(RELU, vec![5], vec![6]),
            ],
            vec![0],
            vec![6]
        );
        let schedule = fuse(&model);
        assert_eq!(schedule.groups.len(), 1);
        let g = &schedule.groups[0];
        assert!(g.folded_requantize.is_some());
        let act = g.activation.expect("relu must fuse after the fold");
        assert_eq!((act.quantized_min, act.quantized_max), (0, 127));
        assert_eq!(g.absorbed_ops, vec![1, 2]);
        assert_eq!(g.output_tensor, 6);
        assert!(g.requires_verification, "requantize fold is a T2 group");
    }

    #[test]
    fn requantize_fold_refused_when_conv_out_zp_nonzero() {
        // conv output zp = -1 → the mul input1_offset is non-zero → the fold
        // would not be a pure scale change; must not fire.
        let (_bytes, model) = model!(
            vec![
                BuildTensor::quantized(&[1, 4, 4, 8], "input", BuildQuant::tensor(1.0, 0)),
                BuildTensor::constant(&[2, 1, 1, 8], "conv/weights", vec![0u8; 2 * 8]),
                BuildTensor::constant(&[2], "conv/bias", vec![0u8; 2 * 4]),
                BuildTensor::quantized(&[1, 4, 4, 2], "conv/out", BuildQuant::tensor(0.25, -1)),
                BuildTensor::constant(&[1], "ones", vec![1, 1, 1, 1]),
                BuildTensor::quantized(&[1, 4, 4, 2], "mul/out", BuildQuant::tensor(0.125, 0)),
            ],
            vec![
                BuildOp::fused(CONV_2D, vec![0, 1, 2], vec![3], OptionsKind::Conv2d, ACT_NONE),
                BuildOp::plain(MUL, vec![3, 4], vec![5]),
            ],
            vec![0],
            vec![5]
        );
        let schedule = fuse(&model);
        assert!(schedule.groups[0].folded_requantize.is_none(), "non-zero zp must block the fold");
        assert_eq!(schedule.groups.len(), 2);
        assert_eq!(schedule.fused_op_count(), 0);
    }

    // ------------------------------------------------------------------
    // Generic guards and structure
    // ------------------------------------------------------------------

    #[test]
    fn no_fuse_when_conv_output_has_two_consumers() {
        let (_bytes, model) = model!(
            vec![
                BuildTensor::quantized(&[1, 4, 4, 8], "input", BuildQuant::tensor(1.0, 0)),
                BuildTensor::constant(&[16, 1, 1, 8], "conv/weights", vec![0u8; 16 * 8]),
                BuildTensor::constant(&[16], "conv/bias", vec![0u8; 16 * 4]),
                BuildTensor::quantized(&[1, 4, 4, 16], "conv/out", BuildQuant::tensor(0.25, 0)),
                BuildTensor::quantized(&[1, 4, 4, 16], "relu/out", BuildQuant::tensor(0.25, 0)),
                BuildTensor::constant(&[1], "mul/scale", vec![1, 1, 1, 1]),
                BuildTensor::quantized(&[1, 4, 4, 16], "mul/out", BuildQuant::tensor(0.25, 0)),
            ],
            vec![
                BuildOp::fused(CONV_2D, vec![0, 1, 2], vec![3], OptionsKind::Conv2d, ACT_NONE),
                BuildOp::plain(RELU, vec![3], vec![4]),
                BuildOp::plain(MUL, vec![3, 5], vec![6]),
            ],
            vec![0],
            vec![4, 6]
        );
        let schedule = fuse(&model);
        assert_eq!(schedule.groups.len(), 3, "no fusion across a two-consumer intermediate");
        assert!(schedule.groups[0].activation.is_none());
        assert!(schedule.groups[0].absorbed_ops.is_empty());
        assert_eq!(schedule.fused_op_count(), 0);
    }

    #[test]
    fn no_fuse_when_intermediate_is_model_output() {
        let (_bytes, model) = model!(
            vec![
                BuildTensor::quantized(&[1, 4, 4, 8], "input", BuildQuant::tensor(1.0, 0)),
                BuildTensor::constant(&[16, 1, 1, 8], "conv/weights", vec![0u8; 16 * 8]),
                BuildTensor::constant(&[16], "conv/bias", vec![0u8; 16 * 4]),
                BuildTensor::quantized(&[1, 4, 4, 16], "conv/out", BuildQuant::tensor(0.25, 0)),
                BuildTensor::quantized(&[1, 4, 4, 16], "relu/out", BuildQuant::tensor(0.25, 0)),
            ],
            vec![
                BuildOp::fused(CONV_2D, vec![0, 1, 2], vec![3], OptionsKind::Conv2d, ACT_NONE),
                BuildOp::plain(RELU, vec![3], vec![4]),
            ],
            vec![0],
            vec![3, 4]
        );
        let schedule = fuse(&model);
        assert_eq!(schedule.groups.len(), 2, "model-output intermediate must not be eliminated");
        assert!(schedule.groups[0].absorbed_ops.is_empty());
        assert!(schedule.groups[0].activation.is_none());
        assert_eq!(schedule.fused_op_count(), 0);
    }

    #[test]
    fn unsupported_structure_left_untouched() {
        // reshape → conv: neither op matches a fusion pattern; both stay.
        let (_bytes, model) = model!(
            vec![
                BuildTensor::quantized(&[1, 8, 8, 4], "input", BuildQuant::tensor(1.0, 0)),
                BuildTensor::quantized(&[1, 4, 16, 4], "reshaped", BuildQuant::tensor(1.0, 0)),
                BuildTensor::constant(&[16, 1, 1, 4], "conv/weights", vec![0u8; 16 * 4]),
                BuildTensor::constant(&[16], "conv/bias", vec![0u8; 16 * 4]),
                BuildTensor::quantized(&[1, 4, 16, 16], "conv/out", BuildQuant::tensor(0.25, 0)),
            ],
            vec![
                BuildOp::plain(22, vec![0], vec![1]), // RESHAPE
                BuildOp::fused(CONV_2D, vec![1, 2, 3], vec![4], OptionsKind::Conv2d, ACT_NONE),
            ],
            vec![0],
            vec![4]
        );
        let schedule = fuse(&model);
        assert_eq!(schedule.groups.len(), 2);
        assert_eq!(schedule.fused_op_count(), 0);
        assert!(schedule.groups[1].activation.is_none());
    }

    #[test]
    fn empty_model_yields_empty_schedule() {
        let (_bytes, model) = model!(vec![], vec![], vec![], vec![]);
        let schedule = fuse(&model);
        assert_eq!(schedule.total_ops, 0);
        assert!(schedule.groups.is_empty());
        assert_eq!(schedule.emitted_op_count(), 0);
        assert_eq!(schedule.fused_op_count(), 0);
    }
}
