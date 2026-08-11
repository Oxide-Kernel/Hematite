// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! T0.2 — fused-pattern profile over the 6 real zoo models (test-only).
//!
//! `fuse()` has no proc-macro-exportable surface (E0477 — proc-macro crates
//! can only export macro items), so this profiling harness is an in-crate
//! `#[cfg(test)]` module, declared as `#[cfg(test)] pub(crate) mod profile;`
//! in `optimize/mod.rs`.  It never compiles into the proc-macro's emit path
//! and adds nothing to the public API.
//!
//! The test [`profile_zoo_models`] parses all 6 models with the **same**
//! `flatbuffer::parse` entry point the proc-macro uses (`lib.rs:41-68`
//! `parse_and_emit` → `flatbuffer::parse(&data)`), runs `fusion::fuse`, and
//! tabulates the fusion opportunity per model.  It prints the table to stdout
//! (`cargo test -p hematite-codegen -- --nocapture profile_zoo_models`) and
//! writes the same content via `std::fs` (host-only, mirroring the
//! `tools/generate_goldens` precedent) to
//! `local-notes/evidence/composed-kernels/fused-profile.md` (CARGO_MANIFEST_DIR
//! relative).  The numbers in that file SET the plan's wave-6 speed targets
//! (T6.2/T6.3), so every column is computed from the real `fuse()` runs — no
//! hand-typed table.
//!
//! # Arena numbers
//!
//! * **Unfused** — `arena::plan_arena(&model)` (arena.rs:238): the real
//!   `hematite-memory::liveness_plan` over the original op schedule.
//! * **Fused estimate** — the SAME planner run over a **reduced schedule**
//!   derived from the [`FusedSchedule`]: one synthetic op per group whose
//!   reads = the group's effective inputs (folds already substituted) plus
//!   the residual tensor and every elementwise-chain operand, and whose write
//!   = the group's output tensor.  Eliminated tensors are produced/consumed
//!   by no op in the reduced schedule, so the planner excludes them
//!   naturally — this is exactly the op list T1.2 will thread through
//!   `emit_model`, not a hand-waved subtraction.  The OpInfo contract mapping
//!   (filter `u32::MAX`/dangling, `u16` casts, `in_place`) mirrors
//!   `arena.rs:143-188` (`build_schedule`) verbatim; keep the two in sync.
//!
//! # SIMD-eligibility column
//!
//! Computed by the **T4.1 parity-tested host mirror** (`crate::eligibility` —
//! one fn per s3 gate, asserted equal to the s3 gates over the spec corpus +
//! widened grids in-crate). Every cell cites the s3 gate it reflects; the
//! runtime-only halves of engagement (pointer 16B alignment, scratch sizing)
//! are not host-visible and are noted per cell.

#![allow(dead_code)]

use std::path::Path;

use crate::eligibility as mir;
use crate::flatbuffer::{self, ParsedModel, ParsedOptions, ParsedTensor, TensorType};
use hematite_core::op_params::{
    ElementwiseChainParams, ElementwiseChainStep, ElementwiseKind, ElementwiseParams,
    FusedActivation, Padding, PoolParams,
};
use hematite_memory::{liveness_plan, ArenaPlan, OpInfo, MAX_IO_PER_OP, MAX_TENSORS};

use super::arena::{self, ArenaError};
use super::fusion::{fuse, ElementwiseKind as FusionElementwiseKind, FusedGroup, FusedSchedule};

// BuiltinOperator codes — fusion.rs's consts are private; re-declared here
// (values from the vendored v23.1-era schema, verified by T4.0).
const ADD: i32 = 0;
const AVERAGE_POOL_2D: i32 = 1;
const CONV_2D: i32 = 3;
const DEPTHWISE_CONV_2D: i32 = 4;
const FULLY_CONNECTED: i32 = 9;
const MAX_POOL_2D: i32 = 17;
const MUL: i32 = 18;
const SOFTMAX: i32 = 25;
const SUB: i32 = 41;

// ---------------------------------------------------------------------------
// Model corpus
// ---------------------------------------------------------------------------

/// One zoo model: display name, repo-relative path, embedded bytes, and the
/// risk flags this profile must carry (from the executed-TFLM goldens
/// evidence — models/zoo/README.md, DEFERRED_MODELS.md, simd-zoo-hardening
/// learnings).
struct ModelSpec {
    name: &'static str,
    path: &'static str,
    risk: &'static str,
    bytes: &'static [u8],
}

const SINE_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../models/sine.tflite"
));
const HELLO_WORLD_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../models/zoo/sine_regression/hello_world_int8.tflite"
));
const KWS_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../models/zoo/keyword_spotting/kws_micro_speech_int8.tflite"
));
const ANOMALY_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../models/zoo/anomaly_detect/anomaly_detect_int8.tflite"
));
const PERSON_DETECT_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../models/zoo/person_detect_vww/person_detect_int8.tflite"
));
const MOBILENET_V2_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../models/zoo/mobilenetv2_cls/mobilenet_v2_1.0_224_int8.tflite"
));

const MODELS: &[ModelSpec] = &[
    ModelSpec {
        name: "sine",
        path: "models/sine.tflite",
        risk: "none — single FC",
        bytes: SINE_BYTES,
    },
    ModelSpec {
        name: "hello_world",
        path: "models/zoo/sine_regression/hello_world_int8.tflite",
        risk: "none — 3 FC, bit-exact golden",
        bytes: HELLO_WORLD_BYTES,
    },
    ModelSpec {
        name: "kws_micro_speech",
        path: "models/zoo/keyword_spotting/kws_micro_speech_int8.tflite",
        risk: "depthwise dm=8 (out_c=2·in_c) blocks dm==1 SIMD gate; 7 ms ESP-DL bar structurally unreachable (PROJECT_LOG.md:796-799)",
        bytes: KWS_BYTES,
    },
    ModelSpec {
        name: "anomaly_detect",
        path: "models/zoo/anomaly_detect/anomaly_detect_int8.tflite",
        risk: "±1 LSB variance — 210/640 elements differ from executed-TFLM golden (gemmlowp double-rounding vs hematite single-rounding; DEFERRED_MODELS §6)",
        bytes: ANOMALY_BYTES,
    },
    ModelSpec {
        name: "person_detect",
        path: "models/zoo/person_detect_vww/person_detect_int8.tflite",
        risk: "device-stack-gated — generated predict allocas ~232 KB vs ~65 KB stack (SKIP reason=stack; QEMU-only golden fnv 0x6962079d)",
        bytes: PERSON_DETECT_BYTES,
    },
    ModelSpec {
        name: "mobilenet_v2_1.0_224",
        path: "models/zoo/mobilenetv2_cls/mobilenet_v2_1.0_224_int8.tflite",
        risk: "18 PAD ops (PAD never fuses — no pattern covers it); 984/1000 golden deltas (890 PAD-fill class); PSRAM-gated on device (PSRAM: 0 bytes)",
        bytes: MOBILENET_V2_BYTES,
    },
];

// ---------------------------------------------------------------------------
// Per-model analysis
// ---------------------------------------------------------------------------

/// Pattern buckets a group may fall into.  Counting is NON-exclusive — a
/// single group can be, e.g., residual-add AND activation-epilogue
/// (fusion.rs:193-250 fields decide).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PatternCounts {
    activation_epilogue: usize,
    elementwise_chain: usize,
    residual_add: usize,
    input_fold: usize,
    requant_fold: usize,
}

/// SIMD eligibility of a group's anchor kernel — computed by the T4.1
/// parity-tested mirror (crate::eligibility); each answer cites the s3 gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SimdEst {
    /// Anchor kernel is SIMD-eligible for these shapes per the cited gate.
    Simd,
    /// In-scope op whose gate FAILS for these shapes (scalar dispatch).
    Scalar,
    /// Anchor has no SIMD path in the C2/C3 composed-kernel scope.
    NoSimdPath,
}

impl SimdEst {
    fn label(self) -> &'static str {
        match self {
            SimdEst::Simd => "SIMD",
            SimdEst::Scalar => "scalar",
            SimdEst::NoSimdPath => "n/a",
        }
    }
}

struct GroupRow {
    anchor_idx: usize,
    anchor_builtin: i32,
    pattern_tags: String,
    elim_tensors: usize,
    elim_bytes: usize,
    simd: SimdEst,
    simd_note: String,
}

struct ModelProfile {
    name: &'static str,
    ops: usize,
    groups: usize,
    fused_ops: usize,
    emitted_ops: usize,
    patterns: PatternCounts,
    elim_tensors: usize,
    elim_bytes: usize,
    simd_groups: usize,
    arena_unfused: Result<usize, String>,
    arena_fused: Result<usize, String>,
    risk: &'static str,
    group_rows: Vec<GroupRow>,
}

fn analyze(spec: &ModelSpec) -> ModelProfile {
    let model = flatbuffer::parse(spec.bytes).unwrap_or_else(|e| {
        panic!("{} failed to parse: {e}", spec.path)
    });
    let schedule = fuse(&model);

    let mut patterns = PatternCounts::default();
    let mut elim_tensors = 0usize;
    let mut elim_bytes = 0usize;
    let mut simd_groups = 0usize;
    let mut group_rows = Vec::with_capacity(schedule.groups.len());

    for g in &schedule.groups {
        if g.activation.is_some() {
            patterns.activation_epilogue += 1;
        }
        if !g.elementwise_chain.is_empty() {
            patterns.elementwise_chain += 1;
        }
        if g.residual_add.is_some() {
            patterns.residual_add += 1;
        }
        if g.input_fold.is_some() {
            patterns.input_fold += 1;
        }
        if g.folded_requantize.is_some() {
            patterns.requant_fold += 1;
        }
        elim_tensors += g.eliminated_tensors.len();
        for &t in &g.eliminated_tensors {
            if let Some(tt) = model.tensor_by_index(t as usize) {
                elim_bytes += tensor_byte_size(tt);
            }
        }

        let (simd, simd_note) = simd_eligibility(&model, g);
        if simd == SimdEst::Simd {
            simd_groups += 1;
        }
        group_rows.push(GroupRow {
            anchor_idx: g.anchor_op_index,
            anchor_builtin: g.anchor_builtin,
            pattern_tags: pattern_tags(g),
            elim_tensors: g.eliminated_tensors.len(),
            elim_bytes: g
                .eliminated_tensors
                .iter()
                .filter_map(|&t| model.tensor_by_index(t as usize))
                .map(tensor_byte_size)
                .sum(),
            simd,
            simd_note,
        });
    }

    let arena_unfused = arena::plan_arena(&model)
        .map(|p| p.peak_arena_bytes)
        .map_err(|e| e.to_string());
    let arena_fused = plan_fused_arena(&model, &schedule)
        .map(|p| p.peak_arena_bytes)
        .map_err(|e| e.to_string());

    ModelProfile {
        name: spec.name,
        ops: schedule.total_ops,
        groups: schedule.groups.len(),
        fused_ops: schedule.fused_op_count(),
        emitted_ops: schedule.emitted_op_count(),
        patterns,
        elim_tensors,
        elim_bytes,
        simd_groups,
        arena_unfused,
        arena_fused,
        risk: spec.risk,
        group_rows,
    }
}

/// Compact per-group pattern tags (`-` = anchor-only group).
fn pattern_tags(g: &FusedGroup) -> String {
    let mut tags = Vec::new();
    if g.activation.is_some() {
        tags.push("act-epilogue");
    }
    if !g.elementwise_chain.is_empty() {
        tags.push("chain");
    }
    if g.residual_add.is_some() {
        tags.push("residual-add");
    }
    if g.input_fold.is_some() {
        tags.push("input-fold");
    }
    if g.folded_requantize.is_some() {
        tags.push("requant-fold");
    }
    if tags.is_empty() {
        "-".to_string()
    } else {
        tags.join("+")
    }
}

// ---------------------------------------------------------------------------
// Fused arena estimate (reduced schedule over the real planner)
// ---------------------------------------------------------------------------

/// Run `hematite-memory::liveness_plan` over the fused op list.
///
/// One synthetic `OpInfo` per group; the contract mapping (filter
/// `u32::MAX`/dangling indices, `u16` casts, `in_place`) mirrors
/// `arena.rs::build_schedule` — the two must stay in sync.
fn plan_fused_arena(
    model: &ParsedModel<'_>,
    schedule: &FusedSchedule,
) -> Result<ArenaPlan, ArenaError> {
    let tensor_count = model.tensors().len();
    if tensor_count > MAX_TENSORS {
        return Err(ArenaError::TooManyTensors { count: tensor_count });
    }

    let mut infos = Vec::with_capacity(schedule.groups.len());
    for g in &schedule.groups {
        // Effective reads: the group's (fold-substituted) inputs plus every
        // tensor the fused kernel actually reads beyond the anchor's own
        // inputs — the residual (pattern c) and each chain operand (pattern b).
        let mut reads: Vec<u32> = g.inputs.clone();
        if let Some(ra) = &g.residual_add {
            reads.push(ra.residual_tensor);
        }
        for cw in &g.elementwise_chain {
            if cw.operand_tensor != u32::MAX {
                reads.push(cw.operand_tensor);
            }
        }
        let inputs: Vec<u32> = reads
            .iter()
            .copied()
            .filter(|&t| t != u32::MAX && (t as usize) < tensor_count)
            .collect();
        let outputs: Vec<u32> = if g.output_tensor != u32::MAX
            && (g.output_tensor as usize) < tensor_count
        {
            vec![g.output_tensor]
        } else {
            Vec::new()
        };
        if inputs.len() > MAX_IO_PER_OP {
            return Err(ArenaError::TooManyInputs { op: g.anchor_op_index, count: inputs.len() });
        }
        if outputs.len() > MAX_IO_PER_OP {
            return Err(ArenaError::TooManyOutputs { op: g.anchor_op_index, count: outputs.len() });
        }

        let mut input_ids = [0u16; MAX_IO_PER_OP];
        for (slot, &t) in inputs.iter().enumerate() {
            input_ids[slot] = t as u16;
        }
        let mut output_ids = [0u16; MAX_IO_PER_OP];
        for (slot, &t) in outputs.iter().enumerate() {
            output_ids[slot] = t as u16;
        }
        let in_place =
            !inputs.is_empty() && !outputs.is_empty() && outputs.first() == inputs.first();

        infos.push(OpInfo {
            op_kind: u16::try_from(g.anchor_builtin).unwrap_or(0),
            input_ids,
            input_count: inputs.len() as u8,
            output_ids,
            output_count: outputs.len() as u8,
            in_place,
        });
    }

    let sizes: Vec<usize> = model.tensors().iter().map(tensor_byte_size).collect();
    let model_input_ids: Vec<u16> = model
        .inputs()
        .iter()
        .copied()
        .filter(|&t| (t as usize) < tensor_count)
        .map(|t| t as u16)
        .collect();
    let model_output_ids: Vec<u16> = model
        .outputs()
        .iter()
        .copied()
        .filter(|&t| (t as usize) < tensor_count)
        .map(|t| t as u16)
        .collect();

    liveness_plan(&infos, &sizes, &model_input_ids, &model_output_ids, arena::MAX_INTERNAL, None)
        .map_err(ArenaError::Layout)
}

// ---------------------------------------------------------------------------
// Tensor sizing
// ---------------------------------------------------------------------------

/// Byte size of a tensor (shape product × element size), mirroring
/// `arena.rs::tensor_byte_size`.  Zero/negative dims → 0; saturated math.
fn tensor_byte_size(t: &ParsedTensor<'_>) -> usize {
    let elems: u64 = t
        .shape
        .iter()
        .fold(1u64, |acc, &d| if d <= 0 { 0 } else { acc.saturating_mul(d as u64) });
    let elem = match t.tensor_type {
        TensorType::Float32 | TensorType::Uint32 | TensorType::Int32 | TensorType::Complex64 => 4,
        TensorType::Float16 | TensorType::Int16 | TensorType::Uint16 => 2,
        TensorType::Int8 | TensorType::Uint8 | TensorType::Bool => 1,
        TensorType::Int64 | TensorType::Uint64 | TensorType::Float64 => 8,
        TensorType::Complex128 => 16,
        TensorType::String
        | TensorType::Variant
        | TensorType::Resource
        | TensorType::Int4
        | TensorType::Unknown => 0,
    };
    (elems.saturating_mul(elem)) as usize
}

/// Flat element count of a shape (0 on dynamic/negative dims).
fn flat_prod(shape: &[i32]) -> usize {
    shape
        .iter()
        .fold(1usize, |acc, &d| if d <= 0 { 0 } else { acc.saturating_mul(d as usize) })
}

/// Channel count of a NHWC tensor = its last shape dim.
fn last_dim(shape: &[i32]) -> usize {
    shape.last().copied().filter(|&d| d > 0).map(|d| d as usize).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// SIMD eligibility — the T4.1 parity-tested host mirror (crate::eligibility).
// Every anchor routes through the SAME gates the s3 dispatchers check; the
// runtime-only halves of engagement (16B pointer alignment, scratch sizing,
// n % 16) are not host-visible and are called out per cell.
// ---------------------------------------------------------------------------

fn tensor_zp(t: &ParsedTensor<'_>) -> i32 {
    t.quant.as_ref().map(|q| q.zero_point as i32).unwrap_or(0)
}

fn tensor_scale(t: &ParsedTensor<'_>) -> Option<f64> {
    t.quant
        .as_ref()
        .filter(|q| q.scale.is_finite() && q.scale > 0.0)
        .map(|q| f64::from(q.scale))
}

/// `input_offset` exactly as the s3 dispatchers receive it: the negative of
/// the input tensor's zero point (TFLite convention).
fn input_offset_of(model: &ParsedModel<'_>, t: u32) -> i32 {
    model
        .tensor_by_index(t as usize)
        .map(tensor_zp)
        .map(|zp| -zp)
        .unwrap_or(0)
}

/// `std::frexp` semantics — replicates generate.rs:747-758 (the profile is
/// test-only; the parity-tested mirror lives in eligibility.rs).
fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 {
        return (0.0, 0);
    }
    let bits = x.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let mantissa = bits & 0x000f_ffff_ffff_ffff;
    let sign = bits & 0x8000_0000_0000_0000;
    let frexp_exponent = exponent - 1022;
    let frexp_significand_bits = sign | 0x3fe0_0000_0000_0000u64 | mantissa;
    (f64::from_bits(frexp_significand_bits), frexp_exponent)
}

/// TFLM `QuantizeMultiplier` — replicates generate.rs:730-744 (semantics
/// copied from hematite-int8).
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

/// TFLM `CalculateActivationRangeQuantized` — replicates generate.rs:847-861.
fn act_range(act: i8, out_scale: f64, out_zp: i32) -> (i32, i32) {
    const QMIN: i32 = -128;
    const QMAX: i32 = 127;
    if out_scale <= 0.0 {
        return (QMIN, QMAX);
    }
    match act {
        1 => (out_zp.max(QMIN), QMAX),
        3 => (
            out_zp.max(QMIN),
            (out_zp + (6.0 / out_scale).round() as i32).min(QMAX),
        ),
        _ => (QMIN, QMAX),
    }
}

/// The elementwise quant view of one elementwise op — replicates
/// generate.rs:1961-1996 (`elementwise_quant`): offsets from tensor zero
/// points, multipliers via the TFLM fixed-point quantize.
fn elementwise_params_view(
    in1: &ParsedTensor<'_>,
    in2: &ParsedTensor<'_>,
    out: &ParsedTensor<'_>,
    kind: ElementwiseKind,
    num_elements: i32,
) -> Option<ElementwiseParams> {
    let in1_scale = tensor_scale(in1)?;
    let in2_scale = tensor_scale(in2)?;
    let out_scale = tensor_scale(out)?;
    let (left_shift, i1m, i1s, i2m, i2s, om, os) = match kind {
        ElementwiseKind::Add | ElementwiseKind::Sub => {
            let twice_max = 2.0 * in1_scale.max(in2_scale);
            let ls = 20i32;
            let (a, b) = quantize_multiplier(in1_scale / twice_max);
            let (c, d) = quantize_multiplier(in2_scale / twice_max);
            let (e, f) = quantize_multiplier(twice_max / ((1i32 << ls) as f64 * out_scale));
            (ls, a, b, c, d, e, f)
        }
        ElementwiseKind::Mul => {
            let (e, f) = quantize_multiplier(in1_scale * in2_scale / out_scale);
            (0, 0, 0, 0, 0, e, f)
        }
        ElementwiseKind::Relu | ElementwiseKind::Relu6 | ElementwiseKind::HardSwish => {
            return None
        }
    };
    Some(ElementwiseParams {
        num_elements,
        input1_offset: -tensor_zp(in1),
        input2_offset: -tensor_zp(in2),
        output_offset: tensor_zp(out),
        output_multiplier: om,
        output_shift: os,
        left_shift,
        input1_multiplier: i1m,
        input1_shift: i1s,
        input2_multiplier: i2m,
        input2_shift: i2s,
        quantized_activation_min: i8::MIN as i32,
        quantized_activation_max: i8::MAX as i32,
    })
}

fn pool_params_view(
    model: &ParsedModel<'_>,
    g: &FusedGroup,
    anchor: &crate::flatbuffer::ParsedOp<'_>,
) -> Option<PoolParams> {
    let in_t = g.inputs.first().and_then(|&t| model.tensor_by_index(t as usize))?;
    let out_t = model.tensor_by_index(g.output_tensor as usize)?;
    let (padding, stride_w, stride_h, filter_w, filter_h, fused_activation) =
        match anchor.options.as_ref() {
            Some(ParsedOptions::Pool2D {
                padding,
                stride_w,
                stride_h,
                filter_w,
                filter_h,
                fused_activation,
            }) => (*padding, *stride_w, *stride_h, *filter_w, *filter_h, *fused_activation),
            _ => (0, 1, 1, 2, 2, 0),
        };
    let ch = last_dim(&in_t.shape) as i32;
    Some(PoolParams {
        input_shape: [1, in_t.shape.get(1).copied().unwrap_or(1), in_t.shape.get(2).copied().unwrap_or(1), ch],
        output_shape: [
            1,
            out_t.shape.get(1).copied().unwrap_or(1),
            out_t.shape.get(2).copied().unwrap_or(1),
            last_dim(&out_t.shape) as i32,
        ],
        filter_width: filter_w,
        filter_height: filter_h,
        stride_width: stride_w,
        stride_height: stride_h,
        padding: if padding == 1 { Padding::Valid } else { Padding::Same },
        activation: match fused_activation {
            1 => FusedActivation::Relu,
            3 => FusedActivation::Relu6,
            _ => FusedActivation::None,
        },
        // The pool gate ignores the clamp range (T3.1 widened) — the values
        // are carried for completeness only.
        quantized_activation_min: i8::MIN as i32,
        quantized_activation_max: i8::MAX as i32,
    })
}

/// The absorbed input fold's elementwise params, derived from the fold op's
/// own tensors exactly as the decomposition emits them (fused.rs:794-810's
/// `fold_elementwise_params` mapping; the codegen `InputFold` IR carries only
/// the real-domain ratio, so the quant pairs are re-derived via
/// `elementwise_params_view` over the absorbed op).
fn fold_params_view(
    model: &ParsedModel<'_>,
    fold: &super::fusion::InputFold,
) -> Option<ElementwiseParams> {
    let op = model.ops().get(fold.op_index)?;
    let in1 = model.tensor_by_index(fold.folded_input_tensor as usize)?;
    let in2 = model.tensor_by_index(fold.operand_tensor as usize)?;
    let out = model.tensor_by_index(*op.outputs.first()? as usize)?;
    let kind = match fold.builtin {
        18 => ElementwiseKind::Mul,
        41 => ElementwiseKind::Sub,
        _ => return None,
    };
    let num_elements = flat_prod(&in1.shape) as i32;
    elementwise_params_view(in1, in2, out, kind, num_elements)
}

/// Chain anchor elementwise params (step 0) — replicates the
/// `emit_fused_chain` step-0 derivation (generate.rs:2264-2341): the anchor
/// op's own quant + its fused-activation clamp.
fn chain_anchor_step(
    model: &ParsedModel<'_>,
    g: &FusedGroup,
    anchor: &crate::flatbuffer::ParsedOp<'_>,
    num_elements: i32,
) -> Option<ElementwiseChainStep<'static>> {
    let kind = match g.anchor_builtin {
        0 => ElementwiseKind::Add,
        18 => ElementwiseKind::Mul,
        41 => ElementwiseKind::Sub,
        _ => return None,
    };
    let in1_t = *anchor.inputs.first()?;
    let in2_t = *anchor.inputs.get(1)?;
    let out_t = *anchor.outputs.first()?;
    let in1 = model.tensor_by_index(in1_t as usize)?;
    let in2 = model.tensor_by_index(in2_t as usize)?;
    let out = model.tensor_by_index(out_t as usize)?;
    let fused_activation = match anchor.options.as_ref() {
        Some(ParsedOptions::Add { fused_activation, .. })
        | Some(ParsedOptions::Sub { fused_activation, .. })
        | Some(ParsedOptions::Mul { fused_activation }) => *fused_activation,
        _ => 0,
    };
    let out_scale = tensor_scale(out)?;
    let (amin, amax) = act_range(fused_activation, out_scale, tensor_zp(out));
    let q = elementwise_params_view(in1, in2, out, kind, num_elements)?;
    Some(ElementwiseChainStep {
        kind,
        operand: Some(&[]),
        input1_offset: q.input1_offset,
        input2_offset: q.input2_offset,
        output_offset: q.output_offset,
        output_multiplier: q.output_multiplier,
        output_shift: q.output_shift,
        left_shift: q.left_shift,
        input1_multiplier: q.input1_multiplier,
        input1_shift: q.input1_shift,
        input2_multiplier: q.input2_multiplier,
        input2_shift: q.input2_shift,
        quantized_activation_min: amin,
        quantized_activation_max: amax,
    })
}

/// Absorbed chain steps (1..) — the fusion IR's `StepRequantize` carries the
/// full per-step elementwise fields; the clamp comes from the step op's fused
/// activation + the step's carried output quant (chain_step_act_range,
/// generate.rs:2410-2429). The mirror's chain gate only tests operand
/// PRESENCE, so `Some(&[])` stands in for the constant operand bytes.
fn chain_absorbed_steps(
    model: &ParsedModel<'_>,
    g: &FusedGroup,
) -> Vec<ElementwiseChainStep<'static>> {
    g.elementwise_chain
        .iter()
        .map(|absorbed| {
            let kind = match absorbed.kind {
                FusionElementwiseKind::Add => ElementwiseKind::Add,
                FusionElementwiseKind::Mul => ElementwiseKind::Mul,
                FusionElementwiseKind::Sub => ElementwiseKind::Sub,
                FusionElementwiseKind::Relu => ElementwiseKind::Relu,
                FusionElementwiseKind::Relu6 => ElementwiseKind::Relu6,
                FusionElementwiseKind::HardSwish => ElementwiseKind::HardSwish,
            };
            let fused = model
                .ops()
                .get(absorbed.op_index)
                .and_then(|op| match op.options.as_ref() {
                    Some(ParsedOptions::Add { fused_activation, .. })
                    | Some(ParsedOptions::Sub { fused_activation, .. })
                    | Some(ParsedOptions::Mul { fused_activation }) => Some(*fused_activation),
                    _ => None,
                })
                .unwrap_or(0);
            let (amin, amax) =
                act_range(fused, f64::from(absorbed.output_scale), absorbed.output_zero_point as i32);
            let rq = &absorbed.requantize;
            ElementwiseChainStep {
                kind,
                operand: if absorbed.operand_tensor == u32::MAX { None } else { Some(&[]) },
                input1_offset: rq.input1_offset,
                input2_offset: rq.input2_offset,
                output_offset: rq.output_offset,
                output_multiplier: rq.output_multiplier,
                output_shift: rq.output_shift,
                left_shift: rq.left_shift,
                input1_multiplier: rq.input1_multiplier,
                input1_shift: rq.input1_shift,
                input2_multiplier: rq.input2_multiplier,
                input2_shift: rq.input2_shift,
                quantized_activation_min: amin,
                quantized_activation_max: amax,
            }
        })
        .collect()
}

fn simd_eligibility(model: &ParsedModel<'_>, g: &FusedGroup) -> (SimdEst, String) {
    let anchor = &model.ops()[g.anchor_op_index];
    match g.anchor_builtin {
        CONV_2D => {
            let Some(w) = anchor.inputs.get(1).and_then(|&t| model.tensor_by_index(t as usize))
            else {
                return (SimdEst::NoSimdPath, "conv weight tensor missing".into());
            };
            let Some(input) = anchor.inputs.first().and_then(|&t| model.tensor_by_index(t as usize))
            else {
                return (SimdEst::NoSimdPath, "conv input tensor missing".into());
            };
            let Some(out) = model.tensor_by_index(g.output_tensor as usize) else {
                return (SimdEst::NoSimdPath, "conv output tensor missing".into());
            };
            let in_c = last_dim(&w.shape);
            let out_c = last_dim(&out.shape);
            let (fh, fw) = (
                w.shape.get(1).copied().unwrap_or(0),
                w.shape.get(2).copied().unwrap_or(0),
            );
            let (sw, sh, dw, dh) = match anchor.options.as_ref() {
                Some(ParsedOptions::Conv2D {
                    stride_w,
                    stride_h,
                    dilation_w,
                    dilation_h,
                    ..
                }) => (*stride_w, *stride_h, *dilation_w, *dilation_h),
                _ => (1, 1, 1, 1),
            };
            let input_offset = input_offset_of(model, anchor.inputs[0]);
            let (in_h, in_w) = (
                input.shape.get(1).copied().unwrap_or(1) as usize,
                input.shape.get(2).copied().unwrap_or(1) as usize,
            );
            let (out_h, out_w) = (
                out.shape.get(1).copied().unwrap_or(1) as usize,
                out.shape.get(2).copied().unwrap_or(1) as usize,
            );
            if fh == 1 && fw == 1 {
                if mir::conv1x1_dispatch_eligible(
                    in_c, out_c, sh, sw, dh, dw, in_h, in_w, out_h, out_w,
                ) {
                    (
                        SimdEst::Simd,
                        "conv1x1: mirror of conv1x1_accx_dispatch (conv1x1.rs:214-224); ptr-align/scratch runtime-only".into(),
                    )
                } else {
                    (
                        SimdEst::Scalar,
                        "conv1x1: mirror conv1x1_dispatch_eligible fails (conv1x1.rs:214-224)".into(),
                    )
                }
            } else if mir::conv3x3_dispatch_eligible(in_c, out_c, fh, fw, dh, dw, input_offset) {
                (
                    SimdEst::Simd,
                    "conv3x3: mirror of conv3x3_accx_dispatch (conv3x3.rs:128-139); ptr-align/scratch runtime-only".into(),
                )
            } else {
                (
                    SimdEst::Scalar,
                    "conv3x3: mirror conv3x3_dispatch_eligible fails (conv3x3.rs:128-139)".into(),
                )
            }
        }
        DEPTHWISE_CONV_2D => {
            let Some(input) = anchor.inputs.first().and_then(|&t| model.tensor_by_index(t as usize))
            else {
                return (SimdEst::NoSimdPath, "depthwise input tensor missing".into());
            };
            let Some(w) = anchor.inputs.get(1).and_then(|&t| model.tensor_by_index(t as usize))
            else {
                return (SimdEst::NoSimdPath, "depthwise weight tensor missing".into());
            };
            let Some(out) = model.tensor_by_index(g.output_tensor as usize) else {
                return (SimdEst::NoSimdPath, "depthwise output tensor missing".into());
            };
            let in_c = last_dim(&input.shape);
            let out_c = last_dim(&out.shape);
            let (fh, fw) = (
                w.shape.get(1).copied().unwrap_or(0),
                w.shape.get(2).copied().unwrap_or(0),
            );
            let (dm, dw, dh) = match anchor.options.as_ref() {
                Some(ParsedOptions::DepthwiseConv2D {
                    depth_multiplier,
                    dilation_w,
                    dilation_h,
                    ..
                }) => (*depth_multiplier, *dilation_w, *dilation_h),
                _ => (1, 1, 1),
            };
            let input_offset = input_offset_of(model, anchor.inputs[0]);
            if mir::depthwise_dispatch_eligible(
                in_c, out_c, dm, fh, fw, dh, dw, input_offset,
            ) {
                (
                    SimdEst::Simd,
                    "depthwise: mirror of depthwise_accx_dispatch (depthwise.rs:310-324); ptr-align/scratch runtime-only".into(),
                )
            } else {
                (
                    SimdEst::Scalar,
                    "depthwise: mirror depthwise_dispatch_eligible fails (depthwise.rs:310-324)".into(),
                )
            }
        }
        FULLY_CONNECTED => {
            let in_dim = g
                .inputs
                .first()
                .and_then(|&t| model.tensor_by_index(t as usize))
                .map(|t| flat_prod(&t.shape))
                .unwrap_or(0);
            let out_dim = model
                .tensor_by_index(g.output_tensor as usize)
                .map(|t| flat_prod(&t.shape))
                .unwrap_or(0);
            if mir::fc_dispatch_eligible(in_dim, out_dim) {
                (
                    SimdEst::Simd,
                    "fc: mirror of fc_accx_dispatch (gemm.rs:137, accx_eligible_1x1_padded); ptr-align/scratch runtime-only".into(),
                )
            } else {
                (
                    SimdEst::Scalar,
                    "fc: mirror fc_dispatch_eligible fails (gemm.rs:137)".into(),
                )
            }
        }
        AVERAGE_POOL_2D | MAX_POOL_2D => {
            let Some(pool) = pool_params_view(model, g, anchor) else {
                return (SimdEst::NoSimdPath, "pool tensor/options missing".into());
            };
            let pool_ok = mir::simd_eligible_pool(&pool);
            let fold_ok = match &g.input_fold {
                None => true,
                Some(fold) => match fold_params_view(model, fold) {
                    Some(ep) => match fold.builtin {
                        18 => mir::simd_eligible_mul(&ep).is_some(),
                        41 => mir::simd_eligible_add_sub(&ep),
                        _ => false,
                    },
                    None => false,
                },
            };
            if pool_ok && fold_ok {
                (
                    SimdEst::Simd,
                    "pool: mirror simd_eligible_pool (pool.rs:1171-1208) + fold_simd_exact (fused.rs:677-684); ptr-align runtime-only".into(),
                )
            } else if pool_ok {
                (
                    SimdEst::Scalar,
                    "pool: mirror simd_eligible_pool OK but fold_simd_exact fails (fused.rs:677-684)".into(),
                )
            } else {
                (
                    SimdEst::Scalar,
                    "pool: mirror simd_eligible_pool fails (pool.rs:1171-1208)".into(),
                )
            }
        }
        SOFTMAX => {
            let row_size = g
                .inputs
                .first()
                .and_then(|&t| model.tensor_by_index(t as usize))
                .map(|t| last_dim(&t.shape) as i32)
                .unwrap_or(0);
            if mir::softmax_row_simd_eligible(row_size) {
                (
                    SimdEst::Simd,
                    "softmax: mirror row_size>=16 (softmax.rs:383-387); ptr-align/scratch runtime-only".into(),
                )
            } else {
                (
                    SimdEst::Scalar,
                    "softmax: mirror softmax_row_simd_eligible fails (softmax.rs:383-387)".into(),
                )
            }
        }
        ADD | SUB | MUL => {
            let num_elements = g
                .inputs
                .first()
                .and_then(|&t| model.tensor_by_index(t as usize))
                .map(|t| flat_prod(&t.shape) as i32)
                .unwrap_or(0);
            let Some(anchor_step) = chain_anchor_step(model, g, anchor, num_elements) else {
                return (SimdEst::NoSimdPath, "chain anchor params not derivable".into());
            };
            let mut steps = vec![anchor_step];
            steps.extend(chain_absorbed_steps(model, g));
            let params = ElementwiseChainParams {
                num_elements,
                steps: &steps,
            };
            if mir::chain_simd_eligible(&params) {
                (
                    SimdEst::Simd,
                    "elementwise chain: mirror chain_simd_eligible (fused.rs:486-502); n%16/ptr-align runtime-only".into(),
                )
            } else {
                (
                    SimdEst::Scalar,
                    "elementwise chain: mirror chain_simd_eligible fails (fused.rs:486-502)".into(),
                )
            }
        }
        _ => (
            SimdEst::NoSimdPath,
            "no composed SIMD kernel in C2/C3 scope (data movement / pad / reshape / transpose / mean)".into(),
        ),
    }
}

// ---------------------------------------------------------------------------
// Markdown + stdout rendering
// ---------------------------------------------------------------------------

fn arena_cell(r: &Result<usize, String>) -> String {
    match r {
        Ok(bytes) => format!("{bytes}"),
        Err(e) => format!("ERR({e})"),
    }
}

fn render(profiles: &[ModelProfile]) -> String {
    let mut s = String::new();
    s.push_str("<!-- Generated by hematite-codegen optimize::profile::profile_zoo_models (T0.2). Do not edit by hand. -->\n");
    s.push_str("# Fused-pattern profile — zoo models (T0.2)\n\n");
    s.push_str(
        "Produced by `cargo test -p hematite-codegen -- --nocapture profile_zoo_models`: every \
         model is parsed with `flatbuffer::parse` (the proc-macro's own entry point, lib.rs:41-68) \
         and fused with `fusion::fuse()`. Column semantics are pinned here so T4.2 \
         (`selector-output.md`) and T6.x speed targets can cite them.\n\n",
    );

    // ── Main table ────────────────────────────────────────────────────────
    s.push_str("## Per-model summary\n\n");
    s.push_str("| Model | ops | groups (emitted) | fused ops | act-epilogue | elementwise-chain | residual-add | input-fold | requant-fold | eliminated_tensors | eliminated_bytes | SIMD-elig | arena unfused (B) | arena fused est (B) | saved calls | saved bytes | risk flags |\n");
    s.push_str("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    for p in profiles {
        let row = format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}/{} | {} | {} | {} | {} | {} |\n",
            p.name,
            p.ops,
            p.groups,
            p.fused_ops,
            p.patterns.activation_epilogue,
            p.patterns.elementwise_chain,
            p.patterns.residual_add,
            p.patterns.input_fold,
            p.patterns.requant_fold,
            p.elim_tensors,
            p.elim_bytes,
            p.simd_groups,
            p.groups,
            arena_cell(&p.arena_unfused),
            arena_cell(&p.arena_fused),
            p.fused_ops, // saved kernel calls == fused_op_count (absorbed ops)
            p.elim_bytes,
            p.risk,
        );
        s.push_str(&row);
    }
    s.push('\n');
    s.push_str(
        "* `ops` = original op count (`FusedSchedule.total_ops`); `groups (emitted)` = \
         `FusedSchedule.groups.len()` (= `emitted_op_count`); `fused ops` = \
         `fused_op_count` (absorbed ops = eliminated kernel calls).\n\
         * Pattern counts are NON-exclusive buckets — one group can satisfy several \
         (e.g. residual-add AND activation-epilogue).\n\
         * `act-epilogue` counts groups whose kernel carries an activation epilogue — either \
         via the anchor's own `fused_activation` field (already present in the UNFUSED kernel \
         call, zero added savings) or via an absorbed standalone activation (counted in \
         `fused ops`). `saved calls` / `fused ops` are the ground truth for eliminated kernel \
         calls.\n\
         * `eliminated_bytes` = sum of the byte sizes of all `eliminated_tensors` (SRAM \
         resident bytes no longer produced; write+read SRAM traffic saved = 2× this).\n\
         * `SIMD-elig` = groups whose anchor kernel the s3 gates engage — \
         computed by the T4.1 parity-tested host mirror (`crate::eligibility`, \
         asserted == the s3 gates over the spec corpus + grids in-crate); per-group \
         cells in the detail sections cite each gate. Runtime-only halves of \
         engagement (pointer 16B alignment, scratch sizing, n % 16) are not \
         host-visible.\n\
         * `arena unfused` = `arena::plan_arena` peak; `arena fused est` = the same \
         planner over the reduced fused schedule (see module docs). `ERR(...)` = planner \
         rejection (e.g. OutOfBudget > 512 KiB).\n",
    );

    // ── Per-model detail ──────────────────────────────────────────────────
    s.push_str("\n## Per-group detail\n\n");
    for p in profiles {
        s.push_str(&format!("### {}\n\n", p.name));
        s.push_str("| group | anchor op | anchor builtin | pattern(s) | elim tensors | elim bytes | SIMD |\n");
        s.push_str("|---|---|---|---|---|---|---|\n");
        for (i, r) in p.group_rows.iter().enumerate() {
            s.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} ({}) |\n",
                i,
                r.anchor_idx,
                r.anchor_builtin,
                r.pattern_tags,
                r.elim_tensors,
                r.elim_bytes,
                r.simd.label(),
                r.simd_note,
            ));
        }
        s.push('\n');
    }

    // ── Method + risk notes ───────────────────────────────────────────────
    s.push_str("## Method\n\n");
    s.push_str(
        "- **Parser/IR**: `flatbuffer::parse(&bytes)` — identical entry point to the \
         proc-macro's `parse_and_emit` (lib.rs:41-68); `include_bytes!` with \
         CARGO_MANIFEST_DIR-relative paths mirrors the in-crate precedent \
         (arena.rs:509-521 `plan_arena_on_real_sine_model`).\n\
         - **Fusion**: `fusion::fuse(&model)` — deterministic, no global state.\n\
         - **Fused arena**: reduced-schedule run of `hematite-memory::liveness_plan`; \
         OpInfo mapping mirrors `arena.rs::build_schedule`. The number is the planner's \
         answer for the fused op list T1.2 will emit — not a subtraction estimate.\n\
         - **SIMD column**: the T4.1 parity-tested host eligibility mirror \
         (`crate::eligibility` — one fn per s3 gate, asserted equal to the s3 \
         gates over the spec corpus + widened grids in-crate). Cells route the \
         anchor's real shapes through the same gates the s3 dispatchers check; \
         runtime pointer alignment / scratch sizing / n%16 are not host-visible.\n\
         - **Not fabricated**: every cell above comes from the real `fuse()` run; a model \
         with zero fused groups would still be a valid row.\n",
    );

    s
}

fn evidence_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("local-notes")
        .join("evidence")
        .join("composed-kernels")
        .join("fused-profile.md")
}

// ---------------------------------------------------------------------------
// The test
// ---------------------------------------------------------------------------

#[test]
fn profile_zoo_models() {
    let profiles: Vec<ModelProfile> = MODELS.iter().map(analyze).collect();

    // Hard invariants — all 6 models parsed + fused without panic, and the
    // schedule totals are self-consistent (fused + emitted == ops).
    assert_eq!(profiles.len(), 6, "all 6 zoo models profiled");
    for p in &profiles {
        assert_eq!(p.ops, p.fused_ops + p.emitted_ops, "{}: fused + emitted == original ops", p.name);
        assert!(
            p.elim_tensors <= p.fused_ops,
            "{}: eliminated tensors ({}) cannot exceed absorbed ops ({})",
            p.name,
            p.elim_tensors,
            p.fused_ops
        );
    }

    let markdown = render(&profiles);

    // stdout — visible with `cargo test -p hematite-codegen -- --nocapture profile_zoo_models`.
    println!("===== fused-pattern profile over zoo models (T0.2) =====");
    println!("{markdown}");

    // Evidence file (host-side only, test-only module).
    let path = evidence_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create local-notes/evidence/composed-kernels");
    }
    std::fs::write(&path, &markdown).unwrap_or_else(|e| {
        panic!("failed to write {}: {e}", path.display())
    });
    println!("wrote {}", path.display());
}
