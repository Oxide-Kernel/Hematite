// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! [`FusedKernelBackend`] for [`RefBackend`] — the composed-kernel default
//! decompositions (T2.1).
//!
//! Every fused method is the EXACT per-op sequence the unfused emitter
//! would emit — anchor op first, then absorbed residual-add / chain steps /
//! input fold, then the trailing activation epilogue — forwarding to the
//! existing per-op kernels.  Bit-exact by construction: the composed call
//! IS the per-op calls.
//!
//! # In-place aliasing (the two unsafe helpers, and only them)
//!
//! The decompositions chain elementwise ops in place: `dst` is both the
//! running input and the output of the next step, and the pool-fold uses
//! the caller's `scratch` buffer (u8) as the i8 intermediate.  Safe Rust
//! cannot express either aliasing pattern (`&[i8]` and `&mut [i8]` views of
//! the same buffer cannot coexist), so two narrow helpers concentrate the
//! only `unsafe` in this file:
//!
//! * [`alias_input`] — a shared view of a mutable buffer, for the in-place
//!   elementwise/activation steps.  Sound because every hematite-ref
//!   elementwise / activation kernel reads `input[i]` strictly before
//!   writing `output[i]` (elementwise.rs, activation.rs), so input ==
//!   output aliasing is well-defined per element.
//! * [`scratch_as_i8`] — `&mut [u8]` → `&mut [i8]` for the fold
//!   intermediate.  Sound because `u8` and `i8` have identical layout
//!   (size 1, alignment 1) and the buffer is only used as scratch.
//!
//! Everything else in this file is safe code.

use hematite_core::op_params::{
    ActivationParams, ComposedActivation, ElementwiseChainParams,
    ElementwiseChainStep, ElementwiseKind, ElementwiseParams,
    FoldedPoolParams, FusedConvParams, PoolKind,
};
use hematite_core::{FusedKernelBackend, KernelBackend, KernelError};

use crate::RefBackend;

/// Shared view of `buf` that may alias a mutable reborrow of the same
/// buffer in the same call (in-place elementwise chaining).
///
/// # Safety
///
/// The callee must read `input[i]` before writing `output[i]` for every
/// element — true of every hematite-ref elementwise / activation kernel
/// (see module docs).
unsafe fn alias_input<'b>(buf: &mut [i8]) -> &'b [i8] {
    core::slice::from_raw_parts(buf.as_ptr(), buf.len())
}

/// Reinterpret `buf` (u8) as `&mut [i8]` — the pool-fold intermediate.
///
/// # Safety
///
/// `u8` and `i8` have identical layout (size 1, alignment 1), and the
/// buffer is only used as scratch between the fold op and the pool.
unsafe fn scratch_as_i8<'b>(buf: &mut [u8]) -> &'b mut [i8] {
    core::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut i8, buf.len())
}

/// Build the standalone-activation params for an epilogue from the fused
/// params, exactly as the unfused emitter would emit the activation op.
fn activation_params<'a>(a: &hematite_core::op_params::ActivationEpilogueParams) -> ActivationParams<'a> {
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

/// Apply the trailing activation epilogue in place on `buf`.
fn apply_activation<B: KernelBackend + ?Sized>(
    backend: &mut B,
    kind: ComposedActivation,
    params: &ActivationParams<'_>,
    buf: &mut [i8],
) -> Result<(), KernelError> {
    match kind {
        ComposedActivation::None => Ok(()),
        ComposedActivation::Relu => backend.relu(unsafe { alias_input(buf) }, params, buf),
        ComposedActivation::Relu6 => backend.relu6(unsafe { alias_input(buf) }, params, buf),
        ComposedActivation::HardSwish => backend.hard_swish(unsafe { alias_input(buf) }, params, buf),
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

impl FusedKernelBackend for RefBackend {
    fn fused_conv2d(
        &mut self,
        src: &[i8],
        weight: &[i8],
        bias: &[i32],
        params: &FusedConvParams,
        dst: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        // 1. The anchor conv, exactly as the unfused emitter would emit it.
        self.conv2d(src, weight, bias, &params.conv, dst, scratch)?;

        // 2. Absorbed residual-ADD, in place (two-stage TFLM Add rounding).
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
            self.add(unsafe { alias_input(dst) }, res.residual_data, &add_params, dst)?;
        }

        // 3. Absorbed trailing activation epilogue, in place on dst.
        apply_activation(self, params.activation.kind, &activation_params(&params.activation), dst)
    }

    fn fused_elementwise_chain(
        &mut self,
        src: &[i8],
        params: &ElementwiseChainParams,
        dst: &mut [i8],
    ) -> Result<(), KernelError> {
        for (idx, step) in params.steps.iter().enumerate() {
            // Step 0 reads src as input1; steps >= 1 read the running value
            // in dst, in place.
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
                ElementwiseKind::Relu => self.relu(input1, &step_activation_params(step), dst)?,
                ElementwiseKind::Relu6 => self.relu6(input1, &step_activation_params(step), dst)?,
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
        // 1. Absorbed input fold (MUL/SUB) materialized into scratch (the
        //    i8 intermediate), or read src directly when no fold.
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

        // 2. The anchor pool, exactly as the unfused emitter would emit it.
        match params.pool_kind {
            PoolKind::Average => self.average_pool_2d(intermediate, &params.pool, dst)?,
            PoolKind::Max => self.max_pool_2d(intermediate, &params.pool, dst)?,
        }

        // 3. Absorbed trailing activation epilogue, in place on dst.
        apply_activation(self, params.activation.kind, &activation_params(&params.activation), dst)
    }
}
