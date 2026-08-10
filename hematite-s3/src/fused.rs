// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! [`FusedKernelBackend`] for [`S3Backend`] — the composed conv kernel (T2.2).
//!
//! # What is fused here
//!
//! `fused_conv2d` executes conv → residual-ADD → activation as ONE kernel
//! call. On real silicon the anchor conv runs through the existing ACCX SIMD
//! dispatch (`conv1x1_accx_dispatch` / `conv3x3_accx_dispatch` — the two
//! conv-family paths reachable from a [`FusedConvParams`] anchor; the trait's
//! anchor type is `Conv2DParams`, so a DEPTHWISE/FULLY_CONNECTED anchor never
//! reaches this entry) into the i32 accumulator, and the residual-ADD +
//! activation epilogue runs per element WITHOUT materializing the conv output
//! to memory — the conv output i8 value is held in a register and fed
//! straight into the two-stage TFLM Add rounding.
//!
//! # Bit-exact contract
//!
//! The reference decomposition (`hematite-ref/src/fused.rs`) is THE oracle:
//! `conv2d` writes the i8 anchor output, then `add` applies the two-stage
//! TFLM Add rounding, then the activation epilogue. The fused SIMD epilogue
//! ([`fused_epilogue`]) reproduces that EXACT fixed-point sequence per
//! element — including the conv's own clamp + saturating_cast (the add's
//! input1 is the *i8* conv output value, so a clamped/saturated conv output
//! propagates), the per-input `multiplier/shift` single roundings, the
//! left_shift, the final requantize, and the activation epilogue. On any
//! ineligible path the trait method falls back to the decomposition through
//! the existing `S3Backend` per-op methods — so the trait method is ALWAYS
//! correct (host, QEMU, and ineligible device shapes all take the fallback).
//!
//! # QEMU gating
//!
//! The SIMD dispatch is `#[cfg(all(target_arch = "xtensa", not(feature =
//! "qemu")))]` — the same gate the conv-family dispatches use (QEMU's TIE728
//! emulation is broken). Under `qemu` and on host the trait method runs the
//! decomposition, bit-exact.
//!
//! # Scratch
//!
//! The fused path needs no scratch beyond the anchor conv's own need
//! (`S3Backend::conv2d_scratch_size`): the residual tensor is read in place
//! and the conv output is register-held. See `backend.rs
//! ::fused_conv2d_scratch_need`.

use hematite_core::op_params::{
    ActivationEpilogueParams, ActivationParams, ComposedActivation, ElementwiseChainParams,
    ElementwiseChainStep, ElementwiseKind, ElementwiseParams, FoldedPoolParams, FusedConvParams,
    PoolKind,
};
use hematite_core::{FusedKernelBackend, KernelBackend, KernelError};
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
use hematite_int8::{multiply_by_quantized_multiplier, saturating_cast};

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

        // Stage 3 — the absorbed trailing activation epilogue, in place.
        v = match fp.act_kind {
            ComposedActivation::None => v,
            ComposedActivation::Relu => {
                let val = v + fp.act_input_offset;
                let act = val.max(0);
                req(act, fp.act_mult, fp.act_shift) + fp.act_output_offset
            }
            ComposedActivation::Relu6 => {
                let val = v + fp.act_input_offset;
                clamp(val, 0, fp.act_max) + fp.act_output_offset
            }
            ComposedActivation::HardSwish => {
                // The DOWNGRADED s3 formula (activations.rs): integer rational
                // x·ReLU6(x+3)/6 with ±3 correction — NO fixed-point. Xtensa
                // has no SIMD integer division, so this per-lane scalar tail is
                // bit-exact vs the s3 scalar `hard_swish`.
                let x = v + fp.act_input_offset;
                let relu6_arg = clamp(x + 3, 0, 6);
                let product = x * relu6_arg;
                let result = if product >= 0 {
                    (product + 3) / 6
                } else {
                    (product - 3) / 6
                };
                result + fp.act_output_offset
            }
        };

        output[out_base + oc] = saturating_cast(v);
    }
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
        let intermediate: &[i8] = match &params.fold {
            Some(fold) => {
                let n = fold.num_elements as usize;
                if scratch.len() < n {
                    return Err(KernelError::ScratchTooSmall);
                }
                let buf = unsafe { scratch_as_i8(&mut scratch[..n]) };
                let ep = ElementwiseParams {
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
                };
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
}
