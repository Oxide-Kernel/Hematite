// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! [`FusedKernelBackend`] for [`S3Backend`] — the composed kernels (T2.2 +
//! T2.3).
//!
//! # What is fused here
//!
//! Three composed kernels live in this module:
//!
//! * [`fused_conv2d`](Self::fused_conv2d) (T2.2) — conv → residual-ADD →
//!   activation as ONE kernel call. On real silicon the anchor conv runs
//!   through the existing ACCX SIMD dispatch (`conv1x1_accx_dispatch` /
//!   `conv3x3_accx_dispatch` — the two conv-family paths reachable from a
//!   [`FusedConvParams`] anchor; the trait's anchor type is `Conv2DParams`,
//!   so a DEPTHWISE/FULLY_CONNECTED anchor never reaches this entry) into
//!   the i32 accumulator, and the residual-ADD + activation epilogue runs
//!   per element WITHOUT materializing the conv output to memory — the conv
//!   output i8 value is held in a register and fed straight into the
//!   two-stage TFLM Add rounding.
//! * [`fused_elementwise_chain`](Self::fused_elementwise_chain) (T2.3) — N
//!   elementwise steps as ONE kernel call, with each step's own requantize
//!   preserved. On real silicon, when EVERY step is SIMD-eligible under the
//!   per-step gates, the chain runs in ONE pass over `num_elements`: 16-wide
//!   chunks are vector-loaded from `src`, the running value stays in i32
//!   REGISTER lanes between steps (each step's per-op requantize applied in
//!   the lanes via the host-compilable [`chain_step_apply`] — the exact
//!   fixed-point sequence of the per-op kernel the decomposition would run,
//!   NEVER materialized to memory), and the final i8 chunk is vector-stored
//!   to `dst`. Zero intermediate stores.
//! * [`fused_pool_with_fold`](Self::fused_pool_with_fold) (T2.4) — pool +
//!   MUL/SUB input fold + activation epilogue as ONE kernel call. On real
//!   silicon, when the anchor pool passes the existing pool SIMD gate AND
//!   the fold is in the provably-exact subset ([`fused_pool_fold_simd_eligible`]),
//!   the fold materializes into scratch (per-op `mul`/`sub`, SIMD when the
//!   fold's own elementwise gates hold), the pool SIMD kernel reads the
//!   staged scratch directly, and the activation epilogue is applied
//!   register-held in place. Every other group falls back to the
//!   decomposition.
//!
//! # Bit-exact contract
//!
//! The reference decomposition (`hematite-ref/src/fused.rs`) is THE oracle.
//! The fused SIMD epilogue math reproduces the EXACT per-op fixed-point
//! sequences per element. On any ineligible path the trait methods fall back
//! to the decomposition through the existing `S3Backend` per-op methods — so
//! the trait methods are ALWAYS correct (host, QEMU, and ineligible device
//! shapes all take the fallback).
//!
//! # QEMU gating
//!
//! Both SIMD dispatches are `#[cfg(all(target_arch = "xtensa", not(feature
//! = "qemu")))]` — the same gate the conv-family dispatches use (QEMU's
//! TIE728 emulation is broken). Under `qemu` and on host the trait methods
//! run the decomposition, bit-exact.
//!
//! # Scratch
//!
//! * `fused_conv2d` needs no scratch beyond the anchor conv's own need
//!   (`S3Backend::conv2d_scratch_size`): the residual tensor is read in
//!   place and the conv output is register-held. See `backend.rs
//!   ::fused_conv2d_scratch_need`.
//! * `fused_elementwise_chain` needs ZERO scratch: the chain keeps the
//!   running value in register lanes and reads step operands (model constant
//!   tensors) in place. The decomposition forwards no scratch either (the
//!   per-op elementwise/activation kernels take `&mut []`). See `backend.rs
//!   ::fused_elementwise_chain_scratch_need` (== 0, asserted by tests).
//! * `fused_pool_with_fold` needs the fold staging region (the fold output
//!   tensor bytes, `num_elements`, padded up to the pool SIMD kernel's
//!   16-byte multiple) — the pool itself consumes no scratch on either side.
//!   See `backend.rs ::fused_pool_with_fold_scratch_need`.

use hematite_core::op_params::{
    ActivationEpilogueParams, ActivationParams, ComposedActivation, ElementwiseChainParams,
    ElementwiseChainStep, ElementwiseKind, ElementwiseParams, FoldedPoolParams, FusedConvParams,
    PoolInputFold, PoolKind,
};
#[cfg(test)]
use hematite_core::op_params::PoolParams;
use hematite_core::{FusedKernelBackend, KernelBackend, KernelError};
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
use hematite_int8::{multiply_by_quantized_multiplier, saturating_cast};

#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
use crate::activations::relu_simd_eligible_params;
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
use crate::elementwise::{simd_eligible_add_sub, simd_eligible_mul};

use crate::backend::S3Backend;

#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
#[inline]
fn clamp(v: i32, lo: i32, hi: i32) -> i32 {
    if v > hi {
        hi
    } else if v < lo {
        lo
    } else {
        v
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host-compilable fused epilogue (the SIMD-path arithmetic core)
// ─────────────────────────────────────────────────────────────────────────────
//
// Compiled on the device (used by the fused dispatches in conv1x1.rs /
// conv3x3.rs) and under `#[cfg(test)]` on host (used by the unit tests);
// never compiled into host release builds.

/// Per-element fused-epilogue params — the SIMD-path core, host-compilable so
/// the exact fixed-point sequence is unit-tested on the host even though the
/// ACCX dispatch that feeds it is device-only.
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
pub(crate) struct FusedConvAccxParams<'a> {
    /// Anchor conv per-channel requantize pairs (the conv stage, exactly as
    /// the standalone conv epilogue applies them).
    pub bias: &'a [i32],
    pub conv_multipliers: &'a [i32],
    pub conv_shifts: &'a [i32],
    pub conv_output_offset: i32,
    pub conv_act_min: i32,
    pub conv_act_max: i32,
    /// `input1_offset = −(params.output_zero_point)` — the add's input1 is
    /// the anchor output tensor (zp `params.output_zero_point`).
    pub input1_offset: i32,
    /// Residual tensor data (element-aligned with `dst`); `None` = no ADD.
    pub residual: Option<&'a [i8]>,
    /// `input2_offset = −(residual_zero_point)`.
    pub input2_offset: i32,
    /// The add's two-stage TFLM Add rounding params (`ResidualAddParams`).
    pub res_i1_mult: i32,
    pub res_i1_shift: i32,
    pub res_i2_mult: i32,
    pub res_i2_shift: i32,
    pub left_shift: i32,
    pub out_mult: i32,
    pub out_shift: i32,
    /// `output_offset = residual.output_zero_point` (the add output tensor zp).
    pub out_offset: i32,
    /// Trailing activation epilogue (`ComposedActivation`).
    pub act_kind: ComposedActivation,
    pub act_input_offset: i32,
    pub act_output_offset: i32,
    pub act_mult: i32,
    pub act_shift: i32,
    /// `quantized_activation_max` — the ReLU6 clamp bound (`quantized_six`).
    pub act_max: i32,
}

#[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
impl<'a> FusedConvAccxParams<'a> {
    /// Build the epilogue params from the fused params (device dispatch entry).
    #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
    fn build(params: &'a FusedConvParams<'a>, bias: &'a [i32]) -> Self {
        let (residual, input2_offset, i1m, i1s, i2m, i2s, ls, om, os, out_off) =
            match &params.residual {
                Some(res) => (
                    Some(res.residual_data),
                    -(res.residual_zero_point as i32),
                    res.input1_multiplier,
                    res.input1_shift,
                    res.input2_multiplier,
                    res.input2_shift,
                    res.left_shift,
                    res.output_multiplier,
                    res.output_shift,
                    res.output_zero_point as i32,
                ),
                None => (None, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            };
        Self {
            bias,
            conv_multipliers: params.conv.output_multiplier_per_channel,
            conv_shifts: params.conv.output_shift_per_channel,
            conv_output_offset: params.conv.output_offset,
            conv_act_min: params.conv.quantized_activation_min,
            conv_act_max: params.conv.quantized_activation_max,
            input1_offset: -(params.output_zero_point as i32),
            residual,
            input2_offset,
            res_i1_mult: i1m,
            res_i1_shift: i1s,
            res_i2_mult: i2m,
            res_i2_shift: i2s,
            left_shift: ls,
            out_mult: om,
            out_shift: os,
            out_offset: out_off,
            act_kind: params.activation.kind,
            act_input_offset: params.activation.input_offset,
            act_output_offset: params.activation.output_offset,
            act_mult: params.activation.output_multiplier,
            act_shift: params.activation.output_shift,
            act_max: params.activation.quantized_activation_max,
        }
    }
}

/// `multiply_by_quantized_multiplier` with the two proven identity fast forms
/// hoisted (mirrors `requantize_1x1`'s fast paths — bit-identical).
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
#[inline]
fn req(acc: i32, mult: i32, shift: i32) -> i32 {
    if mult == 1 << 30 && shift == 1 {
        // Identity scale: scaled == acc.
        acc
    } else if mult == 1 << 30 && shift == 0 {
        // (acc + 1) >> 1 via the overflow-free 32-bit form (proven
        // bit-identical to the i64 reference for every i32 acc).
        (acc >> 1).wrapping_add(acc & 1)
    } else {
        multiply_by_quantized_multiplier(acc, mult, shift)
    }
}

/// Apply ONE absorbed trailing activation to a single i32 value, returning
/// the requantized result (NOT yet saturating-cast — the caller stores it).
///
/// This is the `fused_epilogue`'s Stage 3, factored out so the pool-fold
/// epilogue (T2.4) reuses the exact same register math instead of
/// re-implementing it. Bit-exact vs the per-op activation kernels
/// (`apply_activation` forwards to `backend.relu`/`relu6`/`hard_swish`) for
/// every kind — proven by the T2.2 epilogue tests for the conv anchor and by
/// the pool-fold golden matrix for the pool anchor.
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
#[inline]
fn composed_activation_apply(
    kind: ComposedActivation,
    input_offset: i32,
    output_offset: i32,
    output_multiplier: i32,
    output_shift: i32,
    act_max: i32,
    v: i32,
) -> i32 {
    match kind {
        ComposedActivation::None => v,
        ComposedActivation::Relu => {
            let val = v + input_offset;
            let act = val.max(0);
            req(act, output_multiplier, output_shift) + output_offset
        }
        ComposedActivation::Relu6 => {
            let val = v + input_offset;
            clamp(val, 0, act_max) + output_offset
        }
        ComposedActivation::HardSwish => {
            // The DOWNGRADED s3 formula (activations.rs): integer rational
            // x·ReLU6(x+3)/6 with ±3 correction — NO fixed-point. Xtensa
            // has no SIMD integer division, so this per-lane scalar tail is
            // bit-exact vs the s3 scalar `hard_swish`.
            let x = v + input_offset;
            let relu6_arg = clamp(x + 3, 0, 6);
            let product = x * relu6_arg;
            let result = if product >= 0 {
                (product + 3) / 6
            } else {
                (product - 3) / 6
            };
            result + output_offset
        }
    }
}

/// Apply the absorbed trailing activation epilogue IN PLACE over a whole
/// tensor, register-held (`composed_activation_apply` per element) — the
/// pool-fold SIMD path's epilogue. Identity (`None`) short-circuits.
///
/// Bit-exact vs `apply_activation` (the per-op kernels) by construction —
/// the same per-element math, same `saturating_cast`.
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
fn apply_composed_activation_inplace(a: &ActivationEpilogueParams, buf: &mut [i8]) {
    if a.kind == ComposedActivation::None {
        return;
    }
    for v in buf.iter_mut() {
        let r = composed_activation_apply(
            a.kind,
            a.input_offset,
            a.output_offset,
            a.output_multiplier,
            a.output_shift,
            a.quantized_activation_max,
            i32::from(*v),
        );
        *v = saturating_cast(r);
    }
}

/// Apply the fused epilogue to one output pixel's i32 accumulators (already
/// folded for `input_offset`), writing `accs.len()` elements to
/// `output[out_base ..]`.
///
/// This is the EXACT fixed-point sequence of the `hematite-ref/src/fused.rs`
/// decomposition, reproduced register-held (the conv output i8 is never
/// written to memory): conv requantize → clamp + saturating_cast → two-stage
/// TFLM Add (per-input multiplier roundings, i32 sum, final requantize) →
/// activation epilogue.
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
#[inline]
pub(crate) fn fused_epilogue(
    fp: &FusedConvAccxParams<'_>,
    output: &mut [i8],
    accs: &[i32],
    out_base: usize,
) {
    let n = accs.len();
    for oc in 0..n {
        // Stage 1 — the anchor conv epilogue, exactly as `conv2d` writes its
        // output tensor (per-channel requantize, output_offset, clamp,
        // saturating_cast to i8). The add reads this i8 value, so a clamped /
        // saturated conv output propagates.
        let acc = accs[oc] + fp.bias[oc];
        let conv_scaled = req(acc, fp.conv_multipliers[oc], fp.conv_shifts[oc]);
        let conv_out =
            saturating_cast(clamp(conv_scaled + fp.conv_output_offset, fp.conv_act_min, fp.conv_act_max));

        let mut v = conv_out as i32;

        // Stage 2 — the absorbed residual-ADD (two-stage TFLM Add rounding),
        // in place on the conv output.
        if let Some(res) = fp.residual {
            let mut val1 = conv_out as i32 + fp.input1_offset;
            let mut val2 = res[out_base + oc] as i32 + fp.input2_offset;
            if fp.left_shift > 0 {
                val1 <<= fp.left_shift;
                val2 <<= fp.left_shift;
            }
            if fp.res_i1_mult != 1 << 30 || fp.res_i1_shift != 1 {
                val1 = multiply_by_quantized_multiplier(val1, fp.res_i1_mult, fp.res_i1_shift);
            }
            if fp.res_i2_mult != 1 << 30 || fp.res_i2_shift != 1 {
                val2 = multiply_by_quantized_multiplier(val2, fp.res_i2_mult, fp.res_i2_shift);
            }
            let sum = val1 + val2;
            let scaled = multiply_by_quantized_multiplier(sum, fp.out_mult, fp.out_shift);
            // The add's activation range is the full int8 range.
            v = clamp(scaled + fp.out_offset, i8::MIN as i32, i8::MAX as i32);
        }

        // Stage 3 — the absorbed trailing activation epilogue, in place
        // (the factored [`composed_activation_apply`] — shared with the
        // T2.4 pool-fold epilogue).
        v = composed_activation_apply(
            fp.act_kind,
            fp.act_input_offset,
            fp.act_output_offset,
            fp.act_mult,
            fp.act_shift,
            fp.act_max,
            v,
        );

        output[out_base + oc] = saturating_cast(v);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fused elementwise chain (T2.3) — the register-held chain math
// ─────────────────────────────────────────────────────────────────────────────
//
// Compiled on the device (used by the fused chain dispatch below) and under
// `#[cfg(test)]` on host (used by the unit tests); never compiled into host
// release builds. This is the T2.3 analog of `fused_epilogue`: the exact
// fixed-point sequence of every per-op kernel the decomposition runs,
// reproduced per step on the running i32 value.

/// Apply ONE chain step to the running i8-range i32 value, returning the
/// step's output as an i8-range i32 (never an i8, so the caller holds it in
/// register lanes between steps).
///
/// Mirrors the EXACT sequence of the per-op s3 scalar kernel the
/// decomposition forwards to (`add`/`mul`/`sub` in elementwise.rs, `relu`/
/// `relu6`/`hard_swish` in activations.rs), including the step's own
/// requantize (input1/input2/output multipliers+shifts, left_shift, offsets,
/// quantized-range clamp) and the final `saturating_cast` — so the running
/// value is always i8-range, exactly as the decomposition's `dst[i]` would be.
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
#[inline]
fn chain_step_apply(step: &ElementwiseChainStep<'_>, running: i32, operand: i8) -> i32 {
    match step.kind {
        ElementwiseKind::Add | ElementwiseKind::Sub => {
            let mut val1 = running + step.input1_offset;
            let mut val2 = i32::from(operand) + step.input2_offset;
            let shift_factor = if step.left_shift >= 0 {
                1i32 << step.left_shift
            } else {
                1
            };
            val1 *= shift_factor;
            val2 *= shift_factor;
            if step.input1_multiplier != 1i32 << 30 || step.input1_shift != 1 {
                val1 = multiply_by_quantized_multiplier(
                    val1, step.input1_multiplier, step.input1_shift);
            }
            if step.input2_multiplier != 1i32 << 30 || step.input2_shift != 1 {
                val2 = multiply_by_quantized_multiplier(
                    val2, step.input2_multiplier, step.input2_shift);
            }
            let raw = if step.kind == ElementwiseKind::Add {
                val1 + val2
            } else {
                val1 - val2
            };
            let scaled = multiply_by_quantized_multiplier(raw, step.output_multiplier, step.output_shift);
            let with_offset = scaled + step.output_offset;
            i32::from(saturating_cast(clamp(
                with_offset,
                step.quantized_activation_min,
                step.quantized_activation_max,
            )))
        }
        ElementwiseKind::Mul => {
            let val1 = running + step.input1_offset;
            let val2 = i32::from(operand) + step.input2_offset;
            let product = val1 * val2;
            let scaled = multiply_by_quantized_multiplier(product, step.output_multiplier, step.output_shift);
            let with_offset = scaled + step.output_offset;
            i32::from(saturating_cast(clamp(
                with_offset,
                step.quantized_activation_min,
                step.quantized_activation_max,
            )))
        }
        ElementwiseKind::Relu => {
            let val = running + step.input1_offset;
            let act = val.max(0);
            let scaled = multiply_by_quantized_multiplier(act, step.output_multiplier, step.output_shift);
            i32::from(saturating_cast(scaled + step.output_offset))
        }
        ElementwiseKind::Relu6 => {
            // The s3 relu6 kernel clamps to `quantized_six`; the S3Backend
            // adapter forwards `quantized_activation_max` as that bound.
            let val = running + step.input1_offset;
            let act = clamp(val, 0, step.quantized_activation_max);
            i32::from(saturating_cast(act + step.output_offset))
        }
        ElementwiseKind::HardSwish => {
            // The DOWNGRADED s3 formula (activations.rs): integer rational
            // x·ReLU6(x+3)/6 with ±3 round-half-away correction.
            let x = running + step.input1_offset;
            let relu6_arg = clamp(x + 3, 0, 6);
            let product = x * relu6_arg;
            let result = if product >= 0 {
                (product + 3) / 6
            } else {
                (product - 3) / 6
            };
            i32::from(saturating_cast(result + step.output_offset))
        }
    }
}

/// Execute the ENTIRE chain on one element's src i8 value, register-held.
///
/// Bit-exact vs the `RefBackend` decomposition by construction: every step
/// applies the exact per-op kernel sequence ([`chain_step_apply`]) and the
/// running value between steps is the decomposition's `dst[i]` i8 value as
/// an i32. Used by the host unit tests to prove the register math equals the
/// decomposition, and by the device dispatch's chunk loop.
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
#[inline]
fn fused_chain_element(params: &ElementwiseChainParams<'_>, src_val: i8, element: usize) -> i8 {
    let mut running = i32::from(src_val);
    for step in params.steps.iter() {
        let operand = step.operand.map_or(0, |op| op[element]);
        running = chain_step_apply(step, running, operand);
    }
    running as i8
}

/// Chain-level SIMD-eligibility gate (host-compilable so the unit tests can
/// prove WHICH chains engage).
///
/// The fused chain SIMD path engages ONLY when EVERY step is SIMD-eligible
/// under the per-step gates: `simd_eligible_add_sub` / `simd_eligible_mul`
/// (elementwise.rs — the gates that prove the raw TIE728 add/sub/mul kernels
/// bit-exact vs the scalar per-op kernels) and `relu_simd_eligible_params`
/// (activations.rs). Relu6 and HardSwish have no SIMD kernels yet (T3.2) — a
/// chain containing either falls back to the decomposition. An Add/Mul/Sub
/// step must carry its operand (the same invariant the decomposition
/// `expect`s). Therefore, TODAY, a chain SIMD-engages only when every step is
/// an identity-quant-affine Add/Sub/Mul or an identity-param Relu.
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
fn chain_simd_eligible(params: &ElementwiseChainParams<'_>) -> bool {
    if params.steps.is_empty() {
        return false;
    }
    params.steps.iter().all(|step| match step.kind {
        ElementwiseKind::Add | ElementwiseKind::Sub => {
            step.operand.is_some()
                && simd_eligible_add_sub(&step_elementwise_params(step, params.num_elements))
        }
        ElementwiseKind::Mul => {
            step.operand.is_some()
                && simd_eligible_mul(&step_elementwise_params(step, params.num_elements)).is_some()
        }
        ElementwiseKind::Relu => relu_simd_eligible_params(&step_activation_params(step)),
        ElementwiseKind::Relu6 | ElementwiseKind::HardSwish => false,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Device fused-ACCX dispatch — NEVER compiled on host or under `feature="qemu"`.
// ─────────────────────────────────────────────────────────────────────────────

/// Length-eligibility gate for the fused SIMD path (the anchor conv's own
/// ACCX gate is checked inside the dispatched functions; this covers the
/// fused-side slices).
#[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
fn fused_eligible(fp: &FusedConvAccxParams<'_>, out_c: usize, out_len: usize) -> bool {
    fp.bias.len() >= out_c
        && fp.conv_multipliers.len() >= out_c
        && fp.conv_shifts.len() >= out_c
        && fp.residual.map_or(true, |r| r.len() >= out_len)
}

/// Attempt the fused SIMD path for `fused_conv2d`, returning `Ok(true)` when
/// handled and `Ok(false)` when ineligible (the trait method falls through to
/// the per-op decomposition).
///
/// The anchor conv runs through the SAME ACCX dispatch `S3Backend::conv2d`
/// would use (1×1 filter → `conv1x1_accx_dispatch`, everything else →
/// `conv3x3_accx_dispatch` — the two conv-family paths reachable from a
/// `Conv2DParams` anchor). The dispatch accumulates into the i32 scratch
/// accumulators and then calls [`fused_epilogue`] instead of the standalone
/// requantize, so the residual-ADD + activation are fused into the one SIMD
/// pass.
#[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
fn fused_conv2d_accx(
    src: &[i8],
    weight: &[i8],
    bias: &[i32],
    params: &FusedConvParams<'_>,
    dst: &mut [i8],
    scratch: &mut [u8],
) -> Result<bool, KernelError> {
    let conv = &params.conv;
    let out_c = conv.output_shape[3].max(0) as usize;
    let fp = FusedConvAccxParams::build(params, bias);
    if !fused_eligible(&fp, out_c, dst.len()) {
        return Ok(false);
    }

    if conv.filter_shape[1] == 1 && conv.filter_shape[2] == 1 {
        let mut accx_ctx = crate::conv1x1::Conv1x1AccxCtx {
            input: src,
            weights: weight,
            bias,
            params: conv,
            output: dst,
            scratch,
        };
        // The uniform hint only feeds the non-fused `requantize_1x1` branch;
        // the fused epilogue does its own per-channel requantize.
        crate::conv1x1::conv1x1_accx_dispatch(&mut accx_ctx, (0, i32::MIN), Some(&fp))
    } else {
        let mut accx_ctx = crate::conv3x3::Conv3x3AccxCtx {
            input: src,
            weights: weight,
            bias,
            params: conv,
            output: dst,
            scratch,
        };
        crate::conv3x3::conv3x3_accx_dispatch(&mut accx_ctx, Some(&fp))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Device fused elementwise-chain dispatch — NEVER compiled on host or under
// `feature="qemu"`.
// ─────────────────────────────────────────────────────────────────────────────

/// Attempt the fused SIMD path for `fused_elementwise_chain`, returning
/// `Ok(true)` when handled and `Ok(false)` when ineligible (the trait method
/// falls through to the per-op decomposition).
///
/// The chain runs in ONE pass over `num_elements`: 16-element chunks are
/// vector-loaded from `src`, the running value stays in i32 REGISTER lanes
/// between steps (each step's own requantize applied in the lanes via
/// [`chain_step_apply`] — the decomposition's per-op sequence, never
/// materialized to memory), and the final i8 chunk is vector-stored to `dst`.
/// Zero intermediate stores.
///
/// Engagement: EVERY step must pass its per-step SIMD gate
/// ([`chain_simd_eligible`]), `num_elements % 16 == 0`, and every tensor
/// pointer 16-byte aligned. Relu6/HardSwish steps (no SIMD yet, T3.2) and
/// mixed-eligibility chains fall back to the decomposition — always bit-exact.
#[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
fn fused_elementwise_chain_simd(
    src: &[i8],
    params: &ElementwiseChainParams<'_>,
    dst: &mut [i8],
) -> Result<bool, KernelError> {
    let n = params.num_elements as usize;
    // Length validation — mirrors the decomposition's per-op shape checks
    // (each per-op kernel rejects non-`num_elements` slices with
    // ShapeMismatch before writing anything).
    if src.len() != n || dst.len() != n {
        return Err(KernelError::ShapeMismatch);
    }
    for step in params.steps.iter() {
        if let Some(op) = step.operand {
            if op.len() != n {
                return Err(KernelError::ShapeMismatch);
            }
        }
    }

    if n == 0 || n % 16 != 0 || !chain_simd_eligible(params) {
        return Ok(false);
    }
    if (src.as_ptr() as usize) % 16 != 0 || (dst.as_mut_ptr() as usize) % 16 != 0 {
        return Ok(false);
    }
    for step in params.steps.iter() {
        if let Some(op) = step.operand {
            if (op.as_ptr() as usize) % 16 != 0 {
                return Ok(false);
            }
        }
    }

    let n_chunks = n / 16;
    for chunk in 0..n_chunks {
        let base = chunk * 16;
        // Vector-load step 0's input1 (src) into the running i32 lanes.
        let mut lanes = [0i32; 16];
        for l in 0..16 {
            lanes[l] = i32::from(src[base + l]);
        }
        // Steps in sequence; the running value stays in the i32 lanes.
        for step in params.steps.iter() {
            let operand_chunk: Option<&[i8]> = step.operand.map(|op| &op[base..base + 16]);
            for l in 0..16 {
                let operand = operand_chunk.map_or(0, |c| c[l]);
                lanes[l] = chain_step_apply(step, lanes[l], operand);
            }
        }
        // Vector-store the final i8 chunk to dst.
        for l in 0..16 {
            dst[base + l] = saturating_cast(lanes[l]);
        }
    }
    Ok(true)
}

// ─────────────────────────────────────────────────────────────────────────────
// Fused pool-with-fold (T2.4) — the fold SIMD eligibility gate
// ─────────────────────────────────────────────────────────────────────────────
//
// The fused `fused_pool_with_fold` SIMD path materializes the absorbed fold
// into scratch (the per-op `mul`/`sub`, which SIMD-dispatches when the fold's
// own elementwise gates hold), runs the EXISTING pool SIMD kernel reading the
// staged scratch directly, and applies the activation epilogue register-held
// (`apply_composed_activation_inplace`). It engages ONLY when every stage is
// bit-exact under the established per-op gates:

/// Fold-level SIMD eligibility: the absorbed MUL/SUB fold's materialization
/// is bit-exact under the per-op elementwise SIMD gates (T2.4 provably-exact
/// subset — "single-rounding-equivalent" folds only).
///
/// * MUL fold — `simd_eligible_mul`: zero offsets, full-range clamp,
///   `output_multiplier == 1<<30`, `output_shift <= 1` — the raw-product +
///   power-of-two-shift form whose output is bit-exact vs the scalar kernel.
/// * SUB fold — `simd_eligible_add_sub`: zero offsets, full-range clamp,
///   `left_shift <= 0`, identity `(1<<30, 1)` pairs everywhere — the raw
///   int8-subtract form.
///
/// Any other fold (non-identity quant-affine MUL scale, two-stage TFLM SUB
/// rounding with `left_shift = 20`, non-zero zero points) is NOT proven
/// single-rounding-exact and falls back to the decomposition — the unfused
/// per-op sequence, always correct.
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
fn fold_simd_exact(fold: &PoolInputFold<'_>) -> bool {
    let ep = fold_elementwise_params(fold);
    match fold.builtin {
        18 /* MUL */ => simd_eligible_mul(&ep).is_some(),
        41 /* SUB */ => simd_eligible_add_sub(&ep),
        _ => false,
    }
}

/// Pool-fold SIMD eligibility: the anchor pool passes the existing pool SIMD
/// gate (`simd_eligible_pool` — 2×2/stride-2/pad-0, channels % 16, full-range
/// clamp, the only shapes the TIE728 `*_22c1` kernels are bit-exact for) AND
/// the fold is in the provably-exact subset ([`fold_simd_exact`]).
///
/// Host-compilable so the unit tests can pin WHICH composed groups engage.
/// When `params.fold` is `None` only the pool gate applies (a plain
/// pool+activation composed call — trivially exact).
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
fn fused_pool_fold_simd_eligible(params: &FoldedPoolParams<'_>) -> bool {
    if crate::pool::simd_eligible_pool(&params.pool).is_none() {
        return false;
    }
    match &params.fold {
        None => true,
        Some(fold) => fold_simd_exact(fold),
    }
}

/// Attempt the fused SIMD path for `fused_pool_with_fold`, returning
/// `Ok(true)` when handled and `Ok(false)` when ineligible (the trait method
/// falls through to the per-op decomposition).
///
/// Engagement: [`fused_pool_fold_simd_eligible`] (pool gate + provably-exact
/// fold subset) and the fold staging fits in `scratch`. The fold materializes
/// into scratch (`scratch_as_i8` — the per-op `mul`/`sub`, which SIMD-engage
/// through the elementwise gates when eligible and aligned), the pool reads
/// the staged scratch through the EXISTING pool SIMD dispatch (its own
/// alignment check gates the final scalar-vs-SIMD split), and the activation
/// epilogue is applied register-held in place. The decomposition is
/// functionally identical but keeps the activation as a separate per-op
/// kernel call; both are bit-exact.
#[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
fn fused_pool_with_fold_simd(
    backend: &mut S3Backend,
    src: &[i8],
    params: &FoldedPoolParams<'_>,
    dst: &mut [i8],
    scratch: &mut [u8],
) -> Result<bool, KernelError> {
    if !fused_pool_fold_simd_eligible(params) {
        return Ok(false);
    }
    let n = match &params.fold {
        Some(fold) => fold.num_elements as usize,
        None => (params.pool.input_shape[1] as usize)
            * (params.pool.input_shape[2] as usize)
            * (params.pool.input_shape[3] as usize),
    };
    if scratch.len() < n {
        return Err(KernelError::ScratchTooSmall);
    }

    let intermediate: &[i8] = match &params.fold {
        Some(fold) => {
            let buf = unsafe { scratch_as_i8(&mut scratch[..n]) };
            let ep = fold_elementwise_params(fold);
            match fold.builtin {
                18 /* MUL */ => backend.mul(src, fold.operand_data, &ep, buf)?,
                41 /* SUB */ => backend.sub(src, fold.operand_data, &ep, buf)?,
                _ => return Ok(false),
            }
            buf
        }
        None => src,
    };

    match params.pool_kind {
        PoolKind::Average => backend.average_pool_2d(intermediate, &params.pool, dst)?,
        PoolKind::Max => backend.max_pool_2d(intermediate, &params.pool, dst)?,
    }

    // Activation epilogue — register-held in place, the factored
    // `fused_epilogue` Stage-3 math (bit-exact vs `apply_activation`).
    apply_composed_activation_inplace(&params.activation, dst);
    Ok(true)
}

// ─────────────────────────────────────────────────────────────────────────────
// Fallback decomposition (host, QEMU, and ineligible device shapes)
// ─────────────────────────────────────────────────────────────────────────────

/// Shared view of `buf` that may alias a mutable reborrow of the same buffer
/// in the same call (in-place elementwise/activation chaining).
///
/// # Safety
///
/// The callee must read `input[i]` strictly before writing `output[i]` for
/// every element — true of every hematite-s3 elementwise / activation kernel
/// (the same single alias helper contract as `hematite-ref/src/fused.rs`).
unsafe fn alias_input<'b>(buf: &mut [i8]) -> &'b [i8] {
    core::slice::from_raw_parts(buf.as_ptr(), buf.len())
}

/// Reinterpret `buf` (u8) as `&mut [i8]` — the pool-fold intermediate.
///
/// # Safety
///
/// `u8` and `i8` have identical layout (size 1, alignment 1), and the buffer
/// is only used as scratch between the fold op and the pool.
unsafe fn scratch_as_i8<'b>(buf: &mut [u8]) -> &'b mut [i8] {
    core::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut i8, buf.len())
}

/// Build the `ElementwiseParams` for an absorbed pool input fold (MUL/SUB),
/// exactly as the decomposition forwards them to the per-op `mul`/`sub`
/// kernels (full-range clamp — the fold output feeds the pool raw). Shared by
/// the decomposition and the SIMD eligibility predicate.
fn fold_elementwise_params(fold: &PoolInputFold<'_>) -> ElementwiseParams {
    ElementwiseParams {
        num_elements: fold.num_elements,
        input1_offset: -(fold.input_zero_point as i32),
        input2_offset: -(fold.operand_zero_point as i32),
        output_offset: fold.output_zero_point as i32,
        output_multiplier: fold.output_multiplier,
        output_shift: fold.output_shift,
        left_shift: fold.left_shift,
        input1_multiplier: fold.input1_multiplier,
        input1_shift: fold.input1_shift,
        input2_multiplier: fold.input2_multiplier,
        input2_shift: fold.input2_shift,
        quantized_activation_min: i8::MIN as i32,
        quantized_activation_max: i8::MAX as i32,
    }
}

/// Build the `ElementwiseParams` for one chain step (Add / Mul / Sub).
fn step_elementwise_params(
    step: &ElementwiseChainStep<'_>,
    num_elements: i32,
) -> ElementwiseParams {
    ElementwiseParams {
        num_elements,
        input1_offset: step.input1_offset,
        input2_offset: step.input2_offset,
        output_offset: step.output_offset,
        output_multiplier: step.output_multiplier,
        output_shift: step.output_shift,
        left_shift: step.left_shift,
        input1_multiplier: step.input1_multiplier,
        input1_shift: step.input1_shift,
        input2_multiplier: step.input2_multiplier,
        input2_shift: step.input2_shift,
        quantized_activation_min: step.quantized_activation_min,
        quantized_activation_max: step.quantized_activation_max,
    }
}

/// Build the standalone-activation params for one chain activation step.
fn step_activation_params<'a>(step: &ElementwiseChainStep<'_>) -> ActivationParams<'a> {
    ActivationParams {
        input_offset: step.input1_offset,
        output_offset: step.output_offset,
        output_multiplier: step.output_multiplier,
        output_shift: step.output_shift,
        quantized_activation_min: step.quantized_activation_min,
        quantized_activation_max: step.quantized_activation_max,
        input_multiplier: 0,
        input_left_shift: 0,
        input_range_radius: 0,
        output_multiplier_alpha: 0,
        output_shift_alpha: 0,
        output_multiplier_identity: 0,
        output_shift_identity: 0,
        alpha_offset: 0,
        alpha_data: &[],
        output_multiplier_1: 0,
        output_shift_1: 0,
        output_multiplier_2: 0,
        output_shift_2: 0,
        reluish_multiplier_fixedpoint_int16: 0,
        reluish_multiplier_exponent: 0,
        output_multiplier_fixedpoint_int16: 0,
        output_multiplier_exponent: 0,
    }
}

/// Apply the trailing activation epilogue in place on `buf`.
fn apply_activation(
    backend: &mut S3Backend,
    kind: ComposedActivation,
    params: &ActivationParams<'_>,
    buf: &mut [i8],
) -> Result<(), KernelError> {
    match kind {
        ComposedActivation::None => Ok(()),
        ComposedActivation::Relu => backend.relu(unsafe { alias_input(buf) }, params, buf),
        ComposedActivation::Relu6 => backend.relu6(unsafe { alias_input(buf) }, params, buf),
        ComposedActivation::HardSwish => {
            backend.hard_swish(unsafe { alias_input(buf) }, params, buf)
        }
    }
}

/// Build the standalone-activation params for an epilogue from the fused
/// params, exactly as the unfused emitter would emit the activation op.
fn activation_params<'a>(a: &ActivationEpilogueParams) -> ActivationParams<'a> {
    ActivationParams {
        input_offset: a.input_offset,
        output_offset: a.output_offset,
        output_multiplier: a.output_multiplier,
        output_shift: a.output_shift,
        quantized_activation_min: a.quantized_activation_min,
        quantized_activation_max: a.quantized_activation_max,
        input_multiplier: 0,
        input_left_shift: 0,
        input_range_radius: 0,
        output_multiplier_alpha: 0,
        output_shift_alpha: 0,
        output_multiplier_identity: 0,
        output_shift_identity: 0,
        alpha_offset: 0,
        alpha_data: &[],
        output_multiplier_1: 0,
        output_shift_1: 0,
        output_multiplier_2: 0,
        output_shift_2: 0,
        reluish_multiplier_fixedpoint_int16: 0,
        reluish_multiplier_exponent: 0,
        output_multiplier_fixedpoint_int16: 0,
        output_multiplier_exponent: 0,
    }
}

/// The per-op decomposition of `fused_conv2d` (bit-exact by construction): the
/// anchor `conv2d`, then the absorbed residual-ADD (two-stage TFLM Add) in
/// place, then the activation epilogue — the EXACT sequence
/// `hematite-ref/src/fused.rs` runs.
fn fused_conv2d_decompose(
    backend: &mut S3Backend,
    src: &[i8],
    weight: &[i8],
    bias: &[i32],
    params: &FusedConvParams<'_>,
    dst: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    backend.conv2d(src, weight, bias, &params.conv, dst, scratch)?;

    if let Some(res) = &params.residual {
        let add_params = ElementwiseParams {
            num_elements: dst.len() as i32,
            input1_offset: -(params.output_zero_point as i32),
            input2_offset: -(res.residual_zero_point as i32),
            output_offset: res.output_zero_point as i32,
            output_multiplier: res.output_multiplier,
            output_shift: res.output_shift,
            left_shift: res.left_shift,
            input1_multiplier: res.input1_multiplier,
            input1_shift: res.input1_shift,
            input2_multiplier: res.input2_multiplier,
            input2_shift: res.input2_shift,
            quantized_activation_min: i8::MIN as i32,
            quantized_activation_max: i8::MAX as i32,
        };
        backend.add(unsafe { alias_input(dst) }, res.residual_data, &add_params, dst)?;
    }

    let ap = activation_params(&params.activation);
    match params.activation.kind {
        ComposedActivation::None => Ok(()),
        ComposedActivation::Relu => backend.relu(unsafe { alias_input(dst) }, &ap, dst),
        ComposedActivation::Relu6 => backend.relu6(unsafe { alias_input(dst) }, &ap, dst),
        ComposedActivation::HardSwish => backend.hard_swish(unsafe { alias_input(dst) }, &ap, dst),
    }
}

impl FusedKernelBackend for S3Backend {
    fn fused_conv2d(
        &mut self,
        src: &[i8],
        weight: &[i8],
        bias: &[i32],
        params: &FusedConvParams,
        dst: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        // Fused SIMD (device-only; QEMU-gated). On any ineligible shape the
        // dispatch returns Ok(false) and we fall through to the decomposition.
        #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
        {
            if fused_conv2d_accx(src, weight, bias, params, dst, scratch)? {
                return Ok(());
            }
        }
        fused_conv2d_decompose(self, src, weight, bias, params, dst, scratch)
    }

    fn fused_elementwise_chain(
        &mut self,
        src: &[i8],
        params: &ElementwiseChainParams,
        dst: &mut [i8],
    ) -> Result<(), KernelError> {
        // Fused SIMD chain (device-only; QEMU-gated). On any ineligible chain
        // (host, QEMU, non-%16 length, relu6/hard_swish or mixed-eligibility
        // steps, misaligned tensors) the dispatch returns Ok(false) and we
        // fall through to the decomposition — always bit-exact.
        #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
        {
            if fused_elementwise_chain_simd(src, params, dst)? {
                return Ok(());
            }
        }
        for (idx, step) in params.steps.iter().enumerate() {
            let input1 = if idx == 0 {
                src
            } else {
                unsafe { alias_input(dst) }
            };
            match step.kind {
                ElementwiseKind::Add => self.add(
                    input1,
                    step.operand.expect("ADD step must carry its operand"),
                    &step_elementwise_params(step, params.num_elements),
                    dst,
                )?,
                ElementwiseKind::Mul => self.mul(
                    input1,
                    step.operand.expect("MUL step must carry its operand"),
                    &step_elementwise_params(step, params.num_elements),
                    dst,
                )?,
                ElementwiseKind::Sub => self.sub(
                    input1,
                    step.operand.expect("SUB step must carry its operand"),
                    &step_elementwise_params(step, params.num_elements),
                    dst,
                )?,
                ElementwiseKind::Relu => {
                    self.relu(input1, &step_activation_params(step), dst)?
                }
                ElementwiseKind::Relu6 => {
                    self.relu6(input1, &step_activation_params(step), dst)?
                }
                ElementwiseKind::HardSwish => {
                    self.hard_swish(input1, &step_activation_params(step), dst)?
                }
            }
        }
        Ok(())
    }

    fn fused_pool_with_fold(
        &mut self,
        src: &[i8],
        params: &FoldedPoolParams,
        dst: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        // Fused SIMD (device-only; QEMU-gated). On any ineligible group
        // (non-SIMD pool shape, fold outside the provably-exact subset,
        // misaligned pointers) the dispatch returns Ok(false) and we fall
        // through to the decomposition — always bit-exact.
        #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
        {
            if fused_pool_with_fold_simd(self, src, params, dst, scratch)? {
                return Ok(());
            }
        }

        let intermediate: &[i8] = match &params.fold {
            Some(fold) => {
                let n = fold.num_elements as usize;
                if scratch.len() < n {
                    return Err(KernelError::ScratchTooSmall);
                }
                let buf = unsafe { scratch_as_i8(&mut scratch[..n]) };
                let ep = fold_elementwise_params(fold);
                match fold.builtin {
                    18 /* MUL */ => self.mul(src, fold.operand_data, &ep, buf)?,
                    41 /* SUB */ => self.sub(src, fold.operand_data, &ep, buf)?,
                    _ => return Err(KernelError::Unsupported),
                }
                buf
            }
            None => src,
        };

        match params.pool_kind {
            PoolKind::Average => self.average_pool_2d(intermediate, &params.pool, dst)?,
            PoolKind::Max => self.max_pool_2d(intermediate, &params.pool, dst)?,
        }

        apply_activation(self, params.activation.kind, &activation_params(&params.activation), dst)
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::boxed::Box;
    use alloc::vec;
    use alloc::vec::Vec;
    use super::*;
    use hematite_core::op_params::Padding;

    /// A 1×1 anchor conv + residual-ADD + activation epilogue, with a
    /// non-identity residual scale (the host test matrix's prototype shape).
    fn test_params<'a>(
        residual: Option<&'a [i8]>,
        activation: ComposedActivation,
    ) -> FusedConvParams<'a> {
        use hematite_core::op_params::Conv2DParams;
        FusedConvParams {
            conv: Conv2DParams {
                input_shape: [1, 4, 4, 16],
                filter_shape: [16, 1, 1, 16],
                output_shape: [1, 4, 4, 16],
                padding: Padding::Same,
                stride_width: 1,
                stride_height: 1,
                dilation_width_factor: 1,
                dilation_height_factor: 1,
                input_offset: 0,
                weights_offset: 0,
                output_offset: 5,
                output_multiplier_per_channel: &[1 << 30; 16],
                output_shift_per_channel: &[0; 16],
                quantized_activation_min: -128,
                quantized_activation_max: 127,
            },
            output_scale: 0.5,
            output_zero_point: 5,
            output_multiplier_per_channel: &[1 << 30; 16],
            output_shift_per_channel: &[0; 16],
            residual: residual.map(|data| {
                use hematite_core::op_params::ResidualAddParams;
                ResidualAddParams {
                    residual_data: data,
                    residual_scale: 0.3,
                    residual_zero_point: -3,
                    output_scale: 0.4,
                    output_zero_point: 1,
                    // StepRequantize pairs for s1=0.5, s2=0.3, s_out=0.4:
                    // input1 = QuantizeMultiplier(0.5/1.0) = (1<<30, 0)
                    // input2 = QuantizeMultiplier(0.3/1.0) = (1288490189, -1)
                    // output = QuantizeMultiplier(1.0/(2^20·0.4)) = (1342177280, -18)
                    input1_multiplier: 1 << 30,
                    input1_shift: 0,
                    input2_multiplier: 1_288_490_189,
                    input2_shift: -1,
                    left_shift: 20,
                    output_multiplier: 1_342_177_280,
                    output_shift: -18,
                }
            }),
            activation: ActivationEpilogueParams {
                kind: activation,
                input_offset: -5,
                output_offset: 2,
                output_multiplier: 1_342_177_280, // QuantizeMultiplier(0.5/0.4)
                output_shift: 1,
                quantized_activation_min: 0,
                quantized_activation_max: 127,
            },
        }
    }

    /// Replicate the ACCX accumulation (raw bias-free dot products — the
    /// kernel never sees bias; the epilogue adds it). Host stand-in for the
    /// SIMD accumulation feeding [`fused_epilogue`].
    fn scalar_conv_accs(
        src: &[i8],
        weights: &[i8],
        conv: &hematite_core::op_params::Conv2DParams<'_>,
    ) -> Vec<i32> {
        let in_c = conv.input_shape[3] as usize;
        let out_c = conv.output_shape[3] as usize;
        let n = conv.output_shape[1] as usize * conv.output_shape[2] as usize;
        let mut accs = vec![0i32; n * out_c];
        for px in 0..n {
            for oc in 0..out_c {
                let mut acc: i32 = 0;
                for ic in 0..in_c {
                    let i_val = i32::from(src[px * in_c + ic]);
                    let w_val = i32::from(weights[oc * in_c + ic]);
                    acc += (i_val + conv.input_offset) * w_val;
                }
                accs[px * out_c + oc] = acc;
            }
        }
        accs
    }

    /// The fused SIMD-path epilogue must be bit-exact vs the full trait
    /// method's decomposition (conv2d → add → activation) when fed the same
    /// accumulators — the on-device SIMD path's arithmetic, proven on host.
    #[test]
    fn fused_epilogue_matches_decomposition_bit_exact() {
        let src: Vec<i8> = (0..256).map(|i| ((i as i64 * 37) % 251 - 125) as i8).collect();
        let weights: Vec<i8> = (0..16 * 16).map(|i| ((i as i64 * 13) % 255 - 127) as i8).collect();
        let bias: Vec<i32> = (0..16).map(|i| (i as i32) * 37 - 500).collect();
        let residual: Vec<i8> = (0..256).map(|i| ((i as i64 * 7) % 200 - 100) as i8).collect();

        for (kind, label) in [
            (ComposedActivation::None, "none"),
            (ComposedActivation::Relu, "relu"),
            (ComposedActivation::Relu6, "relu6"),
            (ComposedActivation::HardSwish, "hard_swish"),
        ] {
            let params = test_params(Some(&residual), kind);
            let mut decomposed = vec![0i8; 256];
            let mut scratch = vec![0u8; 16 * 4];
            let mut backend = S3Backend;
            backend
                .fused_conv2d(&src, &weights, &bias, &params, &mut decomposed, &mut scratch)
                .expect("decomposition runs");

            // Feed the epilogue the SAME i32 accs the scalar conv produces.
            let accs = scalar_conv_accs(&src, &weights, &params.conv);
            let fp = FusedConvAccxParams {
                bias: &bias,
                conv_multipliers: params.conv.output_multiplier_per_channel,
                conv_shifts: params.conv.output_shift_per_channel,
                conv_output_offset: params.conv.output_offset,
                conv_act_min: params.conv.quantized_activation_min,
                conv_act_max: params.conv.quantized_activation_max,
                input1_offset: -(params.output_zero_point as i32),
                residual: Some(&residual),
                input2_offset: -(-3),
                res_i1_mult: 1 << 30,
                res_i1_shift: 0,
                res_i2_mult: 1_288_490_189,
                res_i2_shift: -1,
                left_shift: 20,
                out_mult: 1_342_177_280,
                out_shift: -18,
                out_offset: 1,
                act_kind: kind,
                act_input_offset: -5,
                act_output_offset: 2,
                act_mult: 1_342_177_280,
                act_shift: 1,
                act_max: 127,
            };
            let mut fused = vec![0i8; 256];
            // The epilogue consumes ONE output pixel's accumulators per call
            // (the dispatch loops pixels); replicate that loop here.
            for px in 0..16 {
                fused_epilogue(&fp, &mut fused, &accs[px * 16..px * 16 + 16], px * 16);
            }

            for (i, (&g, &w)) in fused.iter().zip(decomposed.iter()).enumerate() {
                assert_eq!(
                    g, w,
                    "{label} idx {i}: fused epilogue {g} != decomposition {w}"
                );
            }
        }
    }

    /// The same bit-exactness without a residual (activation-only epilogue).
    #[test]
    fn fused_epilogue_no_residual_matches_decomposition_bit_exact() {
        let src: Vec<i8> = (0..256).map(|i| ((i as i64 * 41) % 233 - 116) as i8).collect();
        let weights: Vec<i8> = (0..16 * 16).map(|i| ((i as i64 * 19) % 241 - 120) as i8).collect();
        let bias: Vec<i32> = (0..16).map(|i| (i as i32) * 11 - 100).collect();

        let params = test_params(None, ComposedActivation::Relu6);
        let mut decomposed = vec![0i8; 256];
        let mut scratch = vec![0u8; 16 * 4];
        let mut backend = S3Backend;
        backend
            .fused_conv2d(&src, &weights, &bias, &params, &mut decomposed, &mut scratch)
            .expect("decomposition runs");

        let accs = scalar_conv_accs(&src, &weights, &params.conv);
        let fp = FusedConvAccxParams {
            bias: &bias,
            conv_multipliers: params.conv.output_multiplier_per_channel,
            conv_shifts: params.conv.output_shift_per_channel,
            conv_output_offset: params.conv.output_offset,
            conv_act_min: params.conv.quantized_activation_min,
            conv_act_max: params.conv.quantized_activation_max,
            input1_offset: -(params.output_zero_point as i32),
            residual: None,
            input2_offset: 0,
            res_i1_mult: 0,
            res_i1_shift: 0,
            res_i2_mult: 0,
            res_i2_shift: 0,
            left_shift: 0,
            out_mult: 0,
            out_shift: 0,
            out_offset: 0,
            act_kind: ComposedActivation::Relu6,
            act_input_offset: -5,
            act_output_offset: 2,
            act_mult: 0,
            act_shift: 0,
            act_max: 127,
        };
        let mut fused = vec![0i8; 256];
        for px in 0..16 {
            fused_epilogue(&fp, &mut fused, &accs[px * 16..px * 16 + 16], px * 16);
        }

        assert_eq!(fused, decomposed, "no-residual relu6 epilogue must match");
    }

    // ── T2.3 — fused elementwise chain ────────────────────────────────────

    /// One chain step with explicit quant params (mirrors the field layout of
    /// `ElementwiseChainStep`).
    #[allow(clippy::too_many_arguments)]
    fn test_step(
        kind: ElementwiseKind,
        operand: Option<&'static [i8]>,
        input1_offset: i32,
        input2_offset: i32,
        output_offset: i32,
        output_multiplier: i32,
        output_shift: i32,
        left_shift: i32,
        input1_multiplier: i32,
        input1_shift: i32,
        input2_multiplier: i32,
        input2_shift: i32,
        quantized_activation_min: i32,
        quantized_activation_max: i32,
    ) -> ElementwiseChainStep<'static> {
        ElementwiseChainStep {
            kind,
            operand,
            input1_offset,
            input2_offset,
            output_offset,
            output_multiplier,
            output_shift,
            left_shift,
            input1_multiplier,
            input1_shift,
            input2_multiplier,
            input2_shift,
            quantized_activation_min,
            quantized_activation_max,
        }
    }

    /// The identity add/sub step (the SIMD-eligible form).
    fn identity_add_step(operand: Option<&'static [i8]>) -> ElementwiseChainStep<'static> {
        test_step(
            ElementwiseKind::Add,
            operand,
            0, 0, 0,
            1 << 30, 1,
            0,
            1 << 30, 1,
            1 << 30, 1,
            i8::MIN as i32, i8::MAX as i32,
        )
    }

    /// A NON-identity add step — same two-stage TFLM Add rounding shape the
    /// golden matrix uses (input2 scale 0.3 vs input1 0.5, output 0.4).
    fn non_identity_add_step(operand: Option<&'static [i8]>) -> ElementwiseChainStep<'static> {
        test_step(
            ElementwiseKind::Add,
            operand,
            -5, 3, 1,
            1_342_177_280, -18, // QuantizeMultiplier(2·max/(2^20·0.4))
            20,
            1 << 30, 0, // input1 = QuantizeMultiplier(0.5/1.0)
            1_288_490_189, -1, // input2 = QuantizeMultiplier(0.3/1.0)
            i8::MIN as i32, i8::MAX as i32,
        )
    }

    /// The identity relu step (SIMD-eligible).
    fn identity_relu_step() -> ElementwiseChainStep<'static> {
        test_step(
            ElementwiseKind::Relu,
            None,
            0, 0, 0,
            1 << 30, 1,
            0,
            0, 0,
            0, 0,
            0, 127,
        )
    }

    /// A non-identity relu step (output ratio 0.5/0.4, zp offsets).
    fn non_identity_relu_step() -> ElementwiseChainStep<'static> {
        test_step(
            ElementwiseKind::Relu,
            None,
            -1, 0, 2,
            1_342_177_280, 1,
            0,
            0, 0,
            0, 0,
            0, 127,
        )
    }

    /// The identity mul step (output pair (1<<30, 0) — SIMD-eligible).
    fn identity_mul_step(operand: Option<&'static [i8]>) -> ElementwiseChainStep<'static> {
        test_step(
            ElementwiseKind::Mul,
            operand,
            0, 0, 0,
            1 << 30, 0,
            0,
            0, 0,
            0, 0,
            i8::MIN as i32, i8::MAX as i32,
        )
    }

    /// A hard_swish step (NEVER SIMD-eligible today — T3.2).
    fn hard_swish_step() -> ElementwiseChainStep<'static> {
        test_step(
            ElementwiseKind::HardSwish,
            None,
            -3, 0, -1,
            0, 0,
            0,
            0, 0,
            0, 0,
            0, 127,
        )
    }

    /// The register-held chain math must be bit-exact vs the full trait
    /// method's decomposition (host → per-op scalar kernels) for the same
    /// params — the on-device SIMD path's arithmetic, proven on host.
    fn assert_register_math_matches(params: &ElementwiseChainParams<'_>, src: &[i8]) {
        let mut decomposed = vec![0i8; src.len()];
        let mut backend = S3Backend;
        backend
            .fused_elementwise_chain(src, params, &mut decomposed)
            .expect("decomposition runs");
        let mut fused = vec![0i8; src.len()];
        for (i, &s) in src.iter().enumerate() {
            fused[i] = fused_chain_element(params, s, i);
        }
        assert_eq!(
            fused, decomposed,
            "register-held chain must equal the decomposition element-for-element"
        );
    }

    #[test]
    fn fused_chain_register_math_matches_decomposition_bit_exact() {
        let src: Vec<i8> = (0..256).map(|i| ((i as i64 * 37) % 251 - 125) as i8).collect();
        let op_a: &'static [i8] = Box::leak(
            (0..256).map(|i| ((i as i64 * 11) % 199 - 99) as i8).collect::<Vec<_>>().into_boxed_slice(),
        );
        let op_b: &'static [i8] = Box::leak(
            (0..256).map(|i| ((i as i64 * 29) % 217 - 108) as i8).collect::<Vec<_>>().into_boxed_slice(),
        );

        // (a) the plan's canonical 4-op chain: add + relu + mul + hard_swish,
        //     all NON-identity scales.
        let steps: Vec<ElementwiseChainStep<'static>> = vec![
            non_identity_add_step(Some(op_a)),
            non_identity_relu_step(),
            identity_mul_step(Some(op_b)),
            hard_swish_step(),
        ];
        let params = ElementwiseChainParams {
            num_elements: 256,
            steps: &steps,
        };
        assert_register_math_matches(&params, &src);

        // (b) a 2-op add + relu chain with non-identity offsets.
        let steps: Vec<ElementwiseChainStep<'static>> =
            vec![non_identity_add_step(Some(op_a)), non_identity_relu_step()];
        let params = ElementwiseChainParams {
            num_elements: 256,
            steps: &steps,
        };
        assert_register_math_matches(&params, &src);

        // (c) a mul + hard_swish chain.
        let steps: Vec<ElementwiseChainStep<'static>> =
            vec![identity_mul_step(Some(op_b)), hard_swish_step()];
        let params = ElementwiseChainParams {
            num_elements: 256,
            steps: &steps,
        };
        assert_register_math_matches(&params, &src);

        // (d) a chain containing a relu6 step (register math handles it even
        //     though the SIMD gate refuses the chain today).
        let steps: Vec<ElementwiseChainStep<'static>> = vec![
            non_identity_add_step(Some(op_a)),
            test_step(
                ElementwiseKind::Relu6,
                None,
                -1, 0, 2,
                0, 0,
                0,
                0, 0,
                0, 0,
                0, 24, // quantized six
            ),
        ];
        let params = ElementwiseChainParams {
            num_elements: 256,
            steps: &steps,
        };
        assert_register_math_matches(&params, &src);
    }

    /// The chain-level SIMD-eligibility gate must engage ONLY for all-eligible
    /// chains (identity-quant-affine add/sub/mul + identity relu), and refuse
    /// relu6/hard_swish chains, non-identity chains, operand-less binary
    /// steps, and empty chains.
    #[test]
    fn chain_simd_eligibility_gate_expectations() {
        let op_a: &'static [i8] = Box::leak(vec![0i8; 256].into_boxed_slice());

        // All-identity add + relu → ELIGIBLE (this is what SIMD-engages today).
        let steps: Vec<ElementwiseChainStep<'static>> =
            vec![identity_add_step(Some(op_a)), identity_relu_step()];
        let params = ElementwiseChainParams {
            num_elements: 256,
            steps: &steps,
        };
        assert!(chain_simd_eligible(&params), "identity add+relu must be eligible");

        // All-identity add + mul + relu → ELIGIBLE.
        let steps: Vec<ElementwiseChainStep<'static>> = vec![
            identity_add_step(Some(op_a)),
            identity_mul_step(Some(op_a)),
            identity_relu_step(),
        ];
        let params = ElementwiseChainParams {
            num_elements: 256,
            steps: &steps,
        };
        assert!(chain_simd_eligible(&params), "identity add+mul+relu must be eligible");

        // ANY hard_swish step → INELIGIBLE (no SIMD yet, T3.2).
        let steps: Vec<ElementwiseChainStep<'static>> = vec![
            identity_add_step(Some(op_a)),
            identity_relu_step(),
            hard_swish_step(),
        ];
        let params = ElementwiseChainParams {
            num_elements: 256,
            steps: &steps,
        };
        assert!(!chain_simd_eligible(&params), "chain with hard_swish must fall back");

        // ANY relu6 step → INELIGIBLE.
        let steps: Vec<ElementwiseChainStep<'static>> = vec![
            identity_add_step(Some(op_a)),
            test_step(
                ElementwiseKind::Relu6, None,
                0, 0, 0,
                0, 0,
                0,
                0, 0,
                0, 0,
                0, 24,
            ),
        ];
        let params = ElementwiseChainParams {
            num_elements: 256,
            steps: &steps,
        };
        assert!(!chain_simd_eligible(&params), "chain with relu6 must fall back");

        // A NON-identity add step (offsets / scales) → INELIGIBLE.
        let steps: Vec<ElementwiseChainStep<'static>> =
            vec![non_identity_add_step(Some(op_a)), identity_relu_step()];
        let params = ElementwiseChainParams {
            num_elements: 256,
            steps: &steps,
        };
        assert!(!chain_simd_eligible(&params), "non-identity chain must fall back");

        // A binary step WITHOUT its operand → INELIGIBLE (the decomposition
        // would `expect`-panic; the fused path must not silently compute).
        let steps: Vec<ElementwiseChainStep<'static>> =
            vec![identity_add_step(None), identity_relu_step()];
        let params = ElementwiseChainParams {
            num_elements: 256,
            steps: &steps,
        };
        assert!(!chain_simd_eligible(&params), "operand-less add must fall back");

        // An EMPTY chain → INELIGIBLE.
        let steps: Vec<ElementwiseChainStep<'static>> = Vec::new();
        let params = ElementwiseChainParams {
            num_elements: 256,
            steps: &steps,
        };
        assert!(!chain_simd_eligible(&params), "empty chain must fall back");
    }

    /// The composed chain needs no scratch (0 == 0 on both the SIMD path and
    /// the decomposition).
    #[test]
    fn fused_chain_scratch_need_is_zero() {
        let steps: Vec<ElementwiseChainStep<'static>> =
            vec![identity_add_step(Some(Box::leak(vec![0i8; 256].into_boxed_slice())))];
        let params = ElementwiseChainParams {
            num_elements: 256,
            steps: &steps,
        };
        assert_eq!(crate::backend::fused_elementwise_chain_scratch_need(&params), 0);
    }

    // ── T2.4 — fused pool-with-fold ──────────────────────────────────────

    /// The canonical SIMD-eligible pool anchor (2×2/stride-2/pad-0, channels
    /// % 16, full-range clamp — exactly `simd_eligible_pool`'s contract).
    fn test_pool_params() -> PoolParams {
        use hematite_core::op_params::FusedActivation;
        PoolParams {
            input_shape: [1, 4, 4, 16],
            output_shape: [1, 2, 2, 16],
            filter_width: 2,
            filter_height: 2,
            stride_width: 2,
            stride_height: 2,
            padding: Padding::Same,
            activation: FusedActivation::None,
            quantized_activation_min: i8::MIN as i32,
            quantized_activation_max: i8::MAX as i32,
        }
    }

    /// One fold with explicit quant params.
    #[allow(clippy::too_many_arguments)]
    fn test_fold(
        builtin: i32,
        operand: &'static [i8],
        input_zp: i64,
        operand_zp: i64,
        out_zp: i64,
        left_shift: i32,
        om: i32,
        os: i32,
        i1m: i32,
        i1s: i32,
        i2m: i32,
        i2s: i32,
    ) -> PoolInputFold<'static> {
        PoolInputFold {
            builtin,
            operand_data: operand,
            operand_zero_point: operand_zp,
            input_zero_point: input_zp,
            output_zero_point: out_zp,
            folded_scale: 1.0,
            left_shift,
            output_multiplier: om,
            output_shift: os,
            input1_multiplier: i1m,
            input1_shift: i1s,
            input2_multiplier: i2m,
            input2_shift: i2s,
            num_elements: 256,
        }
    }

    /// An identity MUL fold (the `simd_eligible_mul` contract: zero offsets,
    /// full-range clamp, `(1<<30, shift<=1)` output pair) — SIMD-engages.
    fn identity_mul_fold(operand: &'static [i8]) -> PoolInputFold<'static> {
        test_fold(18, operand, 0, 0, 0, 0, 1 << 30, 0, 0, 0, 0, 0)
    }

    /// A NON-identity MUL fold (a real scale change — two-stage rounding) —
    /// falls back to the decomposition.
    fn non_identity_mul_fold(operand: &'static [i8]) -> PoolInputFold<'static> {
        test_fold(18, operand, 5, -3, 1, 0, 1_717_986_918, -3, 0, 0, 0, 0)
    }

    /// An identity SUB fold (the `simd_eligible_add_sub` contract: zero
    /// offsets, `left_shift <= 0`, identity `(1<<30, 1)` pairs everywhere) —
    /// SIMD-engages.
    fn identity_sub_fold(operand: &'static [i8]) -> PoolInputFold<'static> {
        test_fold(41, operand, 0, 0, 0, 0, 1 << 30, 1, 1 << 30, 1, 1 << 30, 1)
    }

    /// A NON-identity SUB fold (the two-stage TFLM Add rounding with
    /// `left_shift = 20`) — falls back to the decomposition.
    fn non_identity_sub_fold(operand: &'static [i8]) -> PoolInputFold<'static> {
        test_fold(41, operand, 5, -3, 1, 20, 1_342_177_280, -18, 1 << 30, 0, 1_288_490_189, -1)
    }

    fn folded_params(fold: Option<PoolInputFold<'static>>, act: ComposedActivation) -> FoldedPoolParams<'static> {
        FoldedPoolParams {
            pool: test_pool_params(),
            pool_kind: PoolKind::Average,
            fold,
            activation: ActivationEpilogueParams {
                kind: act,
                input_offset: -5,
                output_offset: 2,
                output_multiplier: 1_342_177_280,
                output_shift: 1,
                quantized_activation_min: 0,
                quantized_activation_max: 127,
            },
        }
    }

    /// The T2-gate must engage the fused SIMD path ONLY for the provably-exact
    /// subset: pool gate + identity-quant-affine MUL/SUB folds. Everything
    /// else (non-identity folds, non-SIMD pool shapes) falls back.
    #[test]
    fn fused_pool_fold_simd_eligibility_gate_expectations() {
        let operand: &'static [i8] = Box::leak(vec![0i8; 256].into_boxed_slice());

        // No fold, SIMD-eligible pool → ELIGIBLE (plain pool+activation).
        assert!(fused_pool_fold_simd_eligible(&folded_params(None, ComposedActivation::Relu)));

        // Identity MUL fold → ELIGIBLE.
        assert!(fused_pool_fold_simd_eligible(&folded_params(
            Some(identity_mul_fold(operand)),
            ComposedActivation::None,
        )), "identity MUL fold must SIMD-engage");

        // Identity SUB fold → ELIGIBLE.
        assert!(fused_pool_fold_simd_eligible(&folded_params(
            Some(identity_sub_fold(operand)),
            ComposedActivation::None,
        )), "identity SUB fold must SIMD-engage");

        // NON-identity MUL fold (real scale change) → INELIGIBLE — not
        // proven single-rounding-exact, falls back to the decomposition.
        assert!(!fused_pool_fold_simd_eligible(&folded_params(
            Some(non_identity_mul_fold(operand)),
            ComposedActivation::None,
        )), "non-identity MUL fold must fall back");

        // NON-identity SUB fold (two-stage TFLM rounding, left_shift 20)
        // → INELIGIBLE.
        assert!(!fused_pool_fold_simd_eligible(&folded_params(
            Some(non_identity_sub_fold(operand)),
            ComposedActivation::None,
        )), "non-identity SUB fold must fall back");

        // An unsupported fold builtin → INELIGIBLE.
        assert!(!fused_pool_fold_simd_eligible(&folded_params(
            Some(test_fold(3, operand, 0, 0, 0, 0, 1 << 30, 0, 0, 0, 0, 0)),
            ComposedActivation::None,
        )), "unknown fold builtin must fall back");

        // A NON-SIMD pool shape (channels 8) rejects even an identity fold.
        let mut p = folded_params(Some(identity_mul_fold(operand)), ComposedActivation::None);
        p.pool.input_shape = [1, 4, 4, 8];
        p.pool.output_shape = [1, 2, 2, 8];
        assert!(!fused_pool_fold_simd_eligible(&p), "non-%16 pool must fall back");
    }

    /// The register-held activation epilogue (the SIMD path's tail) must be
    /// bit-exact vs `apply_activation` (the decomposition's per-op kernels)
    /// for every absorbed activation kind.
    #[test]
    fn fused_pool_fold_activation_epilogue_matches_per_op_kernels() {
        for (kind, label) in [
            (ComposedActivation::None, "none"),
            (ComposedActivation::Relu, "relu"),
            (ComposedActivation::Relu6, "relu6"),
            (ComposedActivation::HardSwish, "hard_swish"),
        ] {
            let a = folded_params(None, kind).activation;
            let data: Vec<i8> = (0..64).map(|i| ((i as i64 * 29) % 251 - 125) as i8).collect();
            let mut reg = data.clone();
            apply_composed_activation_inplace(&a, &mut reg);
            let mut per_op = data;
            apply_activation(&mut S3Backend, kind, &activation_params(&a), &mut per_op)
                .expect("per-op activation runs");
            assert_eq!(reg, per_op, "{label}: register epilogue != per-op kernel");
        }
    }

    /// The composed scratch need = fold staging bytes padded to the pool
    /// SIMD kernel's 16-byte multiple (+ the pool's own need, 0).
    #[test]
    fn fused_pool_with_fold_scratch_need_formula() {
        let operand: &'static [i8] = Box::leak(vec![0i8; 256].into_boxed_slice());
        let mut p = folded_params(Some(identity_mul_fold(operand)), ComposedActivation::None);
        assert_eq!(crate::backend::fused_pool_with_fold_scratch_need(&p), 256);

        // Non-multiple-of-16 fold staging pads up: 100 → 112.
        let mut fold = identity_mul_fold(operand);
        fold.num_elements = 100;
        p.fold = Some(fold);
        assert_eq!(crate::backend::fused_pool_with_fold_scratch_need(&p), 112);

        // No fold → no staging (the pool reads src in place).
        p.fold = None;
        assert_eq!(crate::backend::fused_pool_with_fold_scratch_need(&p), 0);
    }
}
