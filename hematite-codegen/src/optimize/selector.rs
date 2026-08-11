// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! T4.2 — static rule-tier composed-kernel selector (compile-time only).
//!
//! Maps every [`FusedGroup`] to the highest-priority **eligible** composed
//! kernel, host-side, using only the T4.1 parity-tested mirror
//! (`crate::eligibility`) — never the s3 gates themselves (not visible at
//! macro time).
//!
//! # Rule tier (priority order)
//!
//! 1. Conv-family composed — `fused_conv2d` (CONV_2D anchor with an absorbed
//!    residual-ADD and/or trailing activation).
//! 2. Chain composed — `fused_elementwise_chain` (ADD/SUB/MUL anchor with an
//!    absorbed elementwise chain).
//! 3. Pool-fold composed — `fused_pool_with_fold` (pool anchor with an
//!    absorbed MUL/SUB input fold).
//! 4. Per-op (no composed call).
//!
//! T2 groups (`requires_verification == true` on the group — input folds,
//! requant folds) resolve to **per-op for now**: their bit-exactness depends
//! on a per-model fused==unfused verification passing, and the W5 wave flips
//! them once that verification exists.
//!
//! # Discipline
//!
//! * **Never silently select a composed kernel the mirror says is
//!   ineligible.** A composed call whose composed SIMD path cannot engage
//!   would be a silent scalar fallback at runtime (the `fused_*` dispatch
//!   falls through to the per-op decomposition when its SIMD gate fails).
//!   Such a group resolves to per-op instead, with the gate failure recorded
//!   in [`Selection::reason`].
//! * The composed-kernel SIMD gate for each family is the SAME gate the s3
//!   `fused_*` dispatch checks:
//!   - `fused_conv2d` routes the anchor through `conv1x1_accx_dispatch` /
//!     `conv3x3_accx_dispatch` (fused.rs:531-585) — the anchor conv's own
//!     dispatch gate, computed by [`simd_eligibility`]'s conv branch.  The
//!     residual-ADD + activation run in the register-held
//!     [`fused_epilogue`](hematite-s3) bit-exactly for ANY params.
//!   - `fused_elementwise_chain` requires [`eligibility::chain_simd_eligible`]
//!     over anchor + absorbed steps (fused.rs:486-502).
//!   - `fused_pool_with_fold` requires [`eligibility::fused_pool_fold_simd_eligible`]
//!     (fused.rs:695-703).
//!
//!   Runtime-only halves (pointer 16B alignment, scratch sizing, `n % 16`)
//!   are not host-visible and are noted per cell.
//!
//! # Graph-input 16B alignment (staging decision)
//!
//! The caller's `input` slice alignment is unknowable at codegen, and the
//! s3 conv1x1 SIMD path falls back to scalar on `in_ptr % 16 != 0`
//! (conv1x1.rs:284-286).  When the model's FIRST emitted kernel is
//! SIMD-eligible per the mirror, [`input_staging_decision`] stages the input
//! region into a 16B-aligned intermediate (the layout.rs:220-229 repad
//! precedent) — default YES.  When the first kernel is scalar anyway, no
//! staging (recorded).
//!
//! # SIMD-eligibility estimates
//!
//! [`simd_eligibility`] (moved here from the T0.2 test-only profile so the
//! emit path and the profile share ONE copy) computes each anchor's estimate
//! from the T4.1 mirror over the parsed model — the same numbers the W0
//! fused-profile pins.

use crate::eligibility as mir;
use crate::flatbuffer::{ParsedModel, ParsedOptions, ParsedTensor};
use hematite_core::op_params::{
    ElementwiseChainParams, ElementwiseChainStep, ElementwiseKind, ElementwiseParams,
    FusedActivation, Padding, PoolParams,
};

use super::fusion::{FusedGroup, InputFold};

// BuiltinOperator codes (values from the vendored v23.1-era schema, verified
// by T4.0; fusion.rs's consts are private, so re-declared here).
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
// Selection IR
// ---------------------------------------------------------------------------

/// Which composed `FusedKernelBackend` call replaces the anchor's per-op call
/// (moved from generate.rs — the T1.2 emitter classifies groups with the
/// same three tiers; the selector now decides eligibility before emitting).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ComposedKind {
    /// CONV_2D anchor with an absorbed residual-ADD and/or trailing
    /// activation (fusion patterns (c) / (a)) → `fused_conv2d`.
    Conv,
    /// ADD/MUL/SUB anchor with an absorbed elementwise chain (pattern (b))
    /// → `fused_elementwise_chain`.
    Chain,
    /// Pool anchor with an absorbed MUL/SUB input fold (pattern (d))
    /// → `fused_pool_with_fold`.
    PoolFold,
}

/// One group's selector verdict: the composed kernel to emit, or per-op.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum GroupSelection {
    Composed(ComposedKind),
    PerOp,
}

/// Full per-group decision: emitted tier + evidence + mirror estimate.
#[derive(Clone, Debug)]
pub(crate) struct Selection {
    pub(crate) kernel: GroupSelection,
    /// Why: cites the s3 gate for the chosen tier, or the fallback reason.
    /// Consumed by the W0 selector-output evidence (test-only profile).
    #[allow(dead_code)]
    pub(crate) reason: String,
    /// Mirror SIMD estimate of the SELECTED tier (composed tiers are Simd by
    /// construction — the selector never composes an ineligible group).
    /// Consumed by the W0 selector-output evidence (test-only profile).
    #[allow(dead_code)]
    pub(crate) simd: SimdEst,
}

// ---------------------------------------------------------------------------
// Per-group SIMD-eligibility estimate (shared with the W0 profile)
// ---------------------------------------------------------------------------

/// SIMD eligibility of a group's anchor kernel — computed by the T4.1
/// parity-tested mirror (crate::eligibility); each answer cites the s3 gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SimdEst {
    /// Anchor kernel is SIMD-eligible for these shapes per the cited gate.
    Simd,
    /// In-scope op whose gate FAILS for these shapes (scalar dispatch).
    Scalar,
    /// Anchor has no SIMD path in the C2/C3 composed-kernel scope.
    NoSimdPath,
}

impl SimdEst {
    /// Used by the test-only W0 profile render (selector itself compares
    /// against `Simd` directly).
    #[allow(dead_code)]
    pub(crate) fn label(self) -> &'static str {
        match self {
            SimdEst::Simd => "SIMD",
            SimdEst::Scalar => "scalar",
            SimdEst::NoSimdPath => "n/a",
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor sizing / quant views (moved from the T0.2 profile so the emit-path
// selector and the test-only profile share ONE copy)
// ---------------------------------------------------------------------------

/// Flat element count of a shape (0 on dynamic/negative dims).
pub(crate) fn flat_prod(shape: &[i32]) -> usize {
    shape
        .iter()
        .fold(1usize, |acc, &d| if d <= 0 { 0 } else { acc.saturating_mul(d as usize) })
}

/// Channel count of a NHWC tensor = its last shape dim.
pub(crate) fn last_dim(shape: &[i32]) -> usize {
    shape.last().copied().filter(|&d| d > 0).map(|d| d as usize).unwrap_or(0)
}

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

/// `std::frexp` semantics — replicates generate.rs:747-758 (the parity-tested
/// mirror lives in eligibility.rs; this is the quant-view glue).
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
    fold: &InputFold,
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
    use super::fusion::ElementwiseKind as FusionElementwiseKind;

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

/// Per-group SIMD-eligibility estimate — the T4.1 parity-tested host mirror
/// (crate::eligibility) over the anchor's shapes.  Every anchor routes
/// through the SAME gates the s3 dispatchers check; the runtime-only halves
/// of engagement (16B pointer alignment, scratch sizing, n % 16) are not
/// host-visible and are called out per cell.
pub(crate) fn simd_eligibility(model: &ParsedModel<'_>, g: &FusedGroup) -> (SimdEst, String) {
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
// The selector — rule-tier composed-kernel decision per group
// ---------------------------------------------------------------------------

/// Structural conv-family candidate: a CONV_2D anchor that absorbed anything
/// (residual-ADD and/or trailing activation — the T1.2 composed_kind arm).
fn conv_candidate(group: &FusedGroup) -> bool {
    group.anchor_builtin == CONV_2D
        && (group.residual_add.is_some() || !group.absorbed_ops.is_empty())
}

/// Structural chain candidate: an ADD/SUB/MUL anchor with an absorbed
/// elementwise chain (the T1.2 composed_kind arm).
fn chain_candidate(group: &FusedGroup) -> bool {
    matches!(group.anchor_builtin, ADD | SUB | MUL) && !group.elementwise_chain.is_empty()
}

/// Structural pool-fold candidate: a pool anchor with an absorbed MUL/SUB
/// input fold (the T1.2 composed_kind arm).
fn pool_fold_candidate(group: &FusedGroup) -> bool {
    matches!(group.anchor_builtin, AVERAGE_POOL_2D | MAX_POOL_2D) && group.input_fold.is_some()
}

/// True when the group structurally matches ANY composed-kernel pattern
/// (regardless of eligibility).  Used by the W0 selector-output evidence
/// (test-only profile) to prove no eligible candidate is silently left
/// per-op.
#[allow(dead_code)]
pub(crate) fn has_composed_candidate(group: &FusedGroup) -> bool {
    conv_candidate(group) || chain_candidate(group) || pool_fold_candidate(group)
}

/// Map one fused group to the highest-priority ELIGIBLE composed kernel.
///
/// Priority: conv-family composed > chain composed > pool-fold composed >
/// per-op.  A composed candidate is taken ONLY when the mirror says its
/// composed SIMD path engages — an ineligible candidate resolves to per-op
/// (bit-exact, never a silent scalar composed call).  T2 groups
/// (`requires_verification`) resolve to per-op for now (W5 flips them).
///
/// The structural candidate arms are mutually exclusive by anchor builtin
/// (conv anchors carry residual/activation, elementwise anchors carry
/// chains, pool anchors carry folds — fusion.rs), so the priority order
/// only matters for a group matching multiple arms; the order is tested
/// below with a contrived multi-pattern group.
pub(crate) fn select_kernel(model: &ParsedModel<'_>, group: &FusedGroup) -> Selection {
    if group.requires_verification {
        let (simd, note) = simd_eligibility(model, group);
        return Selection {
            kernel: GroupSelection::PerOp,
            reason: format!(
                "T2 group (requires_verification) — per-op until the fused==unfused verification passes (W5 flips); {note}"
            ),
            simd,
        };
    }

    // Priority 1 — conv-family composed (`fused_conv2d`).
    if conv_candidate(group) {
        let (simd, note) = simd_eligibility(model, group);
        if simd == SimdEst::Simd {
            return Selection {
                kernel: GroupSelection::Composed(ComposedKind::Conv),
                reason: note,
                simd,
            };
        }
        return Selection {
            kernel: GroupSelection::PerOp,
            reason: format!(
                "conv composed mirror-ineligible — per-op, no silent scalar composed call: {note}"
            ),
            simd,
        };
    }

    // Priority 2 — chain composed (`fused_elementwise_chain`).
    if chain_candidate(group) {
        let (simd, note) = simd_eligibility(model, group);
        if simd == SimdEst::Simd {
            return Selection {
                kernel: GroupSelection::Composed(ComposedKind::Chain),
                reason: note,
                simd,
            };
        }
        return Selection {
            kernel: GroupSelection::PerOp,
            reason: format!(
                "chain composed mirror-ineligible — per-op, no silent scalar composed call: {note}"
            ),
            simd,
        };
    }

    // Priority 3 — pool-fold composed (`fused_pool_with_fold`).
    if pool_fold_candidate(group) {
        let (simd, note) = simd_eligibility(model, group);
        if simd == SimdEst::Simd {
            return Selection {
                kernel: GroupSelection::Composed(ComposedKind::PoolFold),
                reason: note,
                simd,
            };
        }
        return Selection {
            kernel: GroupSelection::PerOp,
            reason: format!(
                "pool-fold composed mirror-ineligible — per-op, no silent scalar composed call: {note}"
            ),
            simd,
        };
    }

    // No composed pattern — the ordinary per-op tier.
    let (simd, note) = simd_eligibility(model, group);
    Selection { kernel: GroupSelection::PerOp, reason: note, simd }
}

// ---------------------------------------------------------------------------
// Graph-input 16B-alignment staging decision
// ---------------------------------------------------------------------------

/// The T4.2 input-staging decision: when the model's FIRST emitted kernel is
/// SIMD-eligible per the mirror, the caller's input bytes are staged into a
/// 16B-aligned intermediate (the caller slice's alignment is unknowable at
/// codegen; the s3 conv1x1 SIMD path falls back to scalar on
/// `in_ptr % 16 != 0` — conv1x1.rs:284-286).  Default YES when the first
/// kernel benefits.
#[derive(Clone, Debug)]
pub(crate) struct StagingDecision {
    /// Stage the input region into a 16B-aligned intermediate.
    pub(crate) stage: bool,
    /// Staged copy size in bytes (the whole model-input region).  Consumed
    /// by the W0 selector-output evidence (test-only profile).
    #[allow(dead_code)]
    pub(crate) bytes: usize,
    /// Anchor op index of the first emitted kernel the decision was made for.
    #[allow(dead_code)]
    pub(crate) first_anchor: usize,
    /// Why: cites the first kernel's mirror gate (or the no-benefit reason).
    #[allow(dead_code)]
    pub(crate) reason: String,
}

/// Decide the input-staging for a schedule: `first` is the first EMITTED
/// group (the first-layer kernel the caller input feeds).
pub(crate) fn input_staging_decision(
    model: &ParsedModel<'_>,
    first: &FusedGroup,
) -> StagingDecision {
    let bytes: usize = model
        .inputs()
        .iter()
        .filter_map(|&t| model.tensor_by_index(t as usize))
        .map(|t| flat_prod(&t.shape))
        .sum();
    let (simd, note) = simd_eligibility(model, first);
    let stage = simd == SimdEst::Simd;
    let reason = if stage {
        // Micro-cost: a word-copy memcpy moves ~4 B/cycle on Xtensa, so the
        // staged copy is ~bytes/4 cycles — e.g. 16 B ≈ 4 cycles per predict,
        // negligible vs the first kernel's own work (thousands of cycles).
        format!(
            "stage {bytes} B into a 16B-aligned intermediate (default YES) — first kernel SIMD-eligible: {note}; copy ≈ {bytes} B ≈ {} cycles at ~4 B/cyc (word memcpy), ~1-time per predict",
            bytes.div_ceil(4)
        )
    } else {
        format!("no staging — first kernel does not benefit: {note}")
    };
    StagingDecision { stage, bytes, first_anchor: first.anchor_op_index, reason }
}

// ---------------------------------------------------------------------------
// Selector unit tests — synthetic groups over real zoo models (the group
// fields are all pub(crate), so tests mutate real fused groups to build
// contrived multi-pattern / T2 / ineligible scenarios).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flatbuffer;
    use crate::optimize::fusion::{
        fuse, AbsorbedElementwise, ElementwiseKind as FusionElementwiseKind, InputFold,
        ResidualAdd, StepRequantize,
    };

    const SINE_TFLITE: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../models/sine.tflite"
    ));
    const PERSON_DETECT_TFLITE: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../models/zoo/person_detect_vww/person_detect_int8.tflite"
    ));

    fn parse_sine() -> crate::flatbuffer::ParsedModel<'static> {
        flatbuffer::parse(SINE_TFLITE).expect("sine parses")
    }

    fn parse_person_detect() -> crate::flatbuffer::ParsedModel<'static> {
        flatbuffer::parse(PERSON_DETECT_TFLITE).expect("person_detect parses")
    }

    /// A no-op StepRequantize (all fields zeroed — only presence matters for
    /// the selector's candidate/eligibility paths; the quant views are
    /// re-derived from the model tensors, not these fields).
    fn zero_requantize() -> StepRequantize {
        StepRequantize {
            left_shift: 0,
            input1_multiplier: 0,
            input1_shift: 0,
            input2_multiplier: 0,
            input2_shift: 0,
            output_multiplier: 0,
            output_shift: 0,
            input1_offset: 0,
            input2_offset: 0,
            output_offset: 0,
        }
    }

    fn dummy_residual_add() -> ResidualAdd {
        ResidualAdd {
            op_index: 0,
            residual_tensor: 0,
            output_scale: 1.0,
            output_zero_point: 0,
            requantize: zero_requantize(),
        }
    }

    fn dummy_chain_step() -> AbsorbedElementwise {
        AbsorbedElementwise {
            op_index: 0,
            kind: FusionElementwiseKind::Add,
            operand_tensor: 0,
            output_scale: 1.0,
            output_zero_point: 0,
            requantize: zero_requantize(),
        }
    }

    fn dummy_input_fold(builtin: i32) -> InputFold {
        InputFold {
            op_index: 0,
            builtin,
            folded_input_tensor: 0,
            operand_tensor: 0,
            folded_scale: 1.0,
            input_zero_point: 0,
        }
    }

    /// Priority rule: a CONV_2D anchor that carries residual-ADD, chain AND
    /// input-fold patterns at once (contrived — real fusion partitions these
    /// patterns by anchor family) must resolve to the conv-family composed
    /// kernel, never to chain or fold.  Uses person_detect's SIMD-eligible
    /// conv1x1 anchor (group 2) so the conv tier actually engages.
    #[test]
    fn priority_order_picks_conv_family_over_chain_and_fold() {
        let model = parse_person_detect();
        let schedule = fuse(&model);
        let mut group = schedule.groups[2].clone();
        assert_eq!(group.anchor_builtin, 3, "group 2 must be a CONV_2D anchor");
        group.residual_add = Some(dummy_residual_add());
        group.elementwise_chain = vec![dummy_chain_step()];
        group.input_fold = Some(dummy_input_fold(18));

        let sel = select_kernel(&model, &group);
        assert_eq!(
            sel.kernel,
            GroupSelection::Composed(ComposedKind::Conv),
            "conv-family must win over chain/fold: {sel:?}"
        );
        assert_eq!(sel.simd, SimdEst::Simd, "composed tiers are Simd by construction");
    }

    /// T2 groups (requires_verification) resolve to per-op even when every
    /// composed pattern is present — W5 flips them after verification.
    #[test]
    fn t2_group_resolves_per_op() {
        let model = parse_sine();
        let schedule = fuse(&model);
        let mut group = schedule.groups[0].clone();
        group.residual_add = Some(dummy_residual_add());
        group.requires_verification = true;

        let sel = select_kernel(&model, &group);
        assert_eq!(sel.kernel, GroupSelection::PerOp, "T2 group must stay per-op");
        assert!(sel.reason.contains("T2"), "reason must name the T2 tier: {}", sel.reason);
        assert!(sel.reason.contains("W5"), "reason must point at W5: {}", sel.reason);
    }

    /// A conv-family candidate whose anchor conv fails the mirror gate must
    /// fall to per-op — never a silent scalar composed call.  sine's FC op
    /// reshaped as a CONV_2D anchor: its weight shape is not 1x1/3x3, so
    /// both dispatch gates fail.
    #[test]
    fn ineligible_conv_candidate_falls_to_per_op() {
        let model = parse_sine();
        let schedule = fuse(&model);
        let mut group = schedule.groups[0].clone();
        group.anchor_builtin = 3; // CONV_2D
        group.residual_add = Some(dummy_residual_add());

        let sel = select_kernel(&model, &group);
        assert_eq!(sel.kernel, GroupSelection::PerOp, "ineligible conv must not compose");
        assert_eq!(sel.simd, SimdEst::Scalar);
        assert!(sel.reason.contains("mirror-ineligible"), "reason: {}", sel.reason);
    }

    /// An SIMD-eligible pool anchor whose absorbed fold is NOT in the
    /// provably-exact subset falls to per-op (the composed pool-fold SIMD
    /// gate is pool gate AND fold_simd_exact).  person_detect's group 27
    /// pool passes the pool gate; a real ADD's quant is not identity, so
    /// fold_simd_exact fails.
    #[test]
    fn eligible_pool_with_inexact_fold_falls_to_per_op() {
        let model = parse_person_detect();
        let schedule = fuse(&model);
        let mut group = schedule.groups[27].clone();
        assert_eq!(group.anchor_builtin, 1, "group 27 must be an average-pool anchor");
        group.input_fold = Some(dummy_input_fold(18)); // MUL fold

        let sel = select_kernel(&model, &group);
        assert_eq!(sel.kernel, GroupSelection::PerOp, "inexact fold must not compose");
        assert!(sel.reason.contains("fold_simd_exact"), "reason: {}", sel.reason);
    }

    /// An SIMD-eligible pool WITHOUT a composed pattern stays per-op — but
    /// not silently: the selection records the SIMD estimate.  (The composed
    /// pool-fold SIMD gate itself — pool gate + fold_simd_exact — is
    /// parity-tested in eligibility.rs over the spec corpus; the positive
    /// fold path needs identity-quant tensors the zoo models don't carry,
    /// so it is locked by that gate test + the conv-family composed test
    /// below, which share the same select-kernel plumbing.)
    #[test]
    fn eligible_ordinary_group_stays_per_op_with_recorded_estimate() {
        let model = parse_person_detect();
        let schedule = fuse(&model);
        let group = &schedule.groups[27];
        assert_eq!(group.anchor_builtin, 1, "group 27 must be an average-pool anchor");
        assert!(group.input_fold.is_none(), "person_detect has no folds");

        let sel = select_kernel(&model, group);
        assert_eq!(sel.kernel, GroupSelection::PerOp, "no composed pattern → per-op");
        assert_eq!(sel.simd, SimdEst::Simd, "the pool itself is SIMD-eligible");
        assert!(sel.reason.contains("simd_eligible_pool"), "reason: {}", sel.reason);
    }

    /// Staging decision: stage when the FIRST kernel is SIMD-eligible
    /// (sine's FC — 1 input byte), skip when it is scalar (person_detect's
    /// conv3x3) — and the copy size is the whole model-input region.
    #[test]
    fn staging_decision_matches_first_kernel_eligibility() {
        let sine = parse_sine();
        let sched = fuse(&sine);
        let d = input_staging_decision(&sine, &sched.groups[0]);
        assert!(d.stage, "sine first FC is SIMD-eligible: {d:?}");
        assert_eq!(d.bytes, 1, "sine input region is 1 byte");
        assert_eq!(d.first_anchor, 0);
        assert!(d.reason.contains("16B-aligned"), "reason: {}", d.reason);

        let pd = parse_person_detect();
        let sched = fuse(&pd);
        let d = input_staging_decision(&pd, &sched.groups[0]);
        assert!(!d.stage, "person_detect first conv3x3 is scalar: {d:?}");
        assert_eq!(d.bytes, 96 * 96 * 3, "person_detect input region is 96x96x3");
        assert!(d.reason.contains("does not benefit"), "reason: {}", d.reason);
    }
}
