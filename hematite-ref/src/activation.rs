// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Standalone activation functions — scalar reference kernels.
//!
//! Each function implements the TFLM int8 reference activation pattern,
//! mirroring the arithmetic in `tools/generate_goldens/src/ops/activations.rs`
//! exactly.
//!
//! # Signature convention
//!
//! All kernels take `(input, params, output, scratch)` and return
//! `Result<(), KernelError>`.  The `scratch` buffer is unused by these
//! element-wise kernels and is accepted only for trait-compatibility.
//!
//! # Ops implemented
//!
//! * [`relu`] — clamp-negative-to-zero
//! * [`relu6`] — ReLU with upper bound (quantized six)
//! * [`hard_swish`] — integer rational approximation of x·relu6(x+3)/6
//! * [`leaky_relu`] — per-tensor alpha-slope negative branch
//! * [`prelu`] — per-channel alpha-slope negative branch

use hematite_core::op_params::ActivationParams;
use hematite_core::KernelError;
use hematite_int8::{multiply_by_quantized_multiplier, saturating_cast};

/// ReLU: `output = max(0, input)` in quantized space.
///
/// # Algorithm (from `tools/generate_goldens/src/ops/activations.rs`)
///
/// 1. `val = x + params.input_offset` (dequantize)
/// 2. `act = max(val, 0)` (ReLU clamp)
/// 3. `scaled = multiply_by_quantized_multiplier(act,
///     params.output_multiplier, params.output_shift)`
/// 4. `saturating_cast(scaled + params.output_offset)`
///
/// # Errors
///
/// * [`KernelError::ShapeMismatch`] if input and output slice lengths differ.
pub fn relu(
    input: &[i8],
    params: &ActivationParams,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    let n = input.len();
    if output.len() != n {
        return Err(KernelError::ShapeMismatch);
    }

    for i in 0..n {
        let val = i32::from(input[i]) + params.input_offset;
        let act = val.max(0);
        let scaled = multiply_by_quantized_multiplier(
            act,
            params.output_multiplier,
            params.output_shift,
        );
        output[i] = saturating_cast(scaled + params.output_offset);
    }

    let _ = scratch;
    Ok(())
}

/// ReLU6: `output = clamp(input, 0, quantized_six)` in quantized space.
///
/// # Parameters
///
/// * `quantized_six` — the quantized representation of 6.0 under the
///   input tensor's quantization scheme.  For symmetric quantization with
///   scale=1 and zero_point=0, this is simply `6`.
///
/// # Algorithm (from `tools/generate_goldens/src/ops/activations.rs`)
///
/// 1. `val = x + params.input_offset`
/// 2. `act = val.max(0).min(quantized_six)`
/// 3. `saturating_cast(act + params.output_offset)`
///
/// The generator skips `multiply_by_quantized_multiplier` for the relu6
/// fixture because the quantized scale is 1.0 (identity).  This kernel
/// matches that path exactly.
///
/// # Errors
///
/// * [`KernelError::ShapeMismatch`] if input and output slice lengths differ.
pub fn relu6(
    input: &[i8],
    params: &ActivationParams,
    output: &mut [i8],
    scratch: &mut [u8],
    quantized_six: i32,
) -> Result<(), KernelError> {
    let n = input.len();
    if output.len() != n {
        return Err(KernelError::ShapeMismatch);
    }

    for i in 0..n {
        let val = i32::from(input[i]) + params.input_offset;
        let act = val.clamp(0, quantized_six);
        output[i] = saturating_cast(act + params.output_offset);
    }

    let _ = scratch;
    Ok(())
}

/// HardSwish: `x · ReLU6(x+3) / 6` — integer rational approximation.
///
/// ⚠️  This is a DOWNGRADED implementation matching the golden fixture
/// provenance.  The TFLM quantized HardSwish uses a 16-bit fixed-point
/// chain with `HardSwishParams`; this kernel uses integer arithmetic with
/// sign-aware rounding.  T5.0 must upgrade to the real TFLM chain.
///
/// # Algorithm (from `tools/generate_goldens/src/ops/activations.rs`)
///
/// 1. `x_i32 = x + params.input_offset`
/// 2. `relu6_arg = clamp(x_i32 + 3, 0, 6)`
/// 3. `product = x_i32 * relu6_arg`
/// 4. `result = (product + 3) / 6` (positive) or `(product - 3) / 6` (negative)
/// 5. `saturating_cast(result + params.output_offset)`
///
/// # Errors
///
/// * [`KernelError::ShapeMismatch`] if input and output slice lengths differ.
pub fn hard_swish(
    input: &[i8],
    params: &ActivationParams,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    let n = input.len();
    if output.len() != n {
        return Err(KernelError::ShapeMismatch);
    }

    for i in 0..n {
        let x_i32 = i32::from(input[i]) + params.input_offset;

        // ReLU6(x + 3)
        let relu6_arg = (x_i32 + 3).clamp(0, 6);

        let product = x_i32 * relu6_arg;

        // Integer division with sign-aware rounding
        let result = if product >= 0 {
            (product + 3) / 6
        } else {
            (product - 3) / 6
        };

        output[i] = saturating_cast(result + params.output_offset);
    }

    let _ = scratch;
    Ok(())
}

/// LeakyReLU: `f(x) = x if x >= 0 else α·x` in quantized space.
///
/// Uses TFLM's `QuantizeLeakyRelu` pattern: two separate
/// `MultiplyByQuantizedMultiplier` branches for positive and negative
/// inputs, each with its own multiplier/shift pair.
///
/// # Algorithm (from `tools/generate_goldens/src/ops/activations.rs`)
///
/// 1. `input_value = x - params.input_offset`
/// 2. If `input_value >= 0`:
///    `unclamped = multiply_by_quantized_multiplier(input_value,
///        params.output_multiplier_identity, params.output_shift_identity)`
///    Else:
///    `unclamped = multiply_by_quantized_multiplier(input_value,
///        params.output_multiplier_alpha, params.output_shift_alpha)`
/// 3. `saturating_cast(params.output_offset + unclamped)`
///
/// # Errors
///
/// * [`KernelError::ShapeMismatch`] if input and output slice lengths differ.
pub fn leaky_relu(
    input: &[i8],
    params: &ActivationParams,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    let n = input.len();
    if output.len() != n {
        return Err(KernelError::ShapeMismatch);
    }

    for i in 0..n {
        let input_value = i32::from(input[i]) - params.input_offset;

        let unclamped = if input_value >= 0 {
            multiply_by_quantized_multiplier(
                input_value,
                params.output_multiplier_identity,
                params.output_shift_identity,
            )
        } else {
            multiply_by_quantized_multiplier(
                input_value,
                params.output_multiplier_alpha,
                params.output_shift_alpha,
            )
        };

        output[i] = saturating_cast(params.output_offset + unclamped);
    }

    let _ = scratch;
    Ok(())
}

/// PReLU: per-channel parametric ReLU.
///
/// `f(x, channel) = x if x >= 0 else α[channel]·x`, where α is a
/// per-channel int8 slope encoded in Q7 (α = alpha_q7 / 128).
///
/// # Algorithm (from `tools/generate_goldens/src/ops/activations.rs`)
///
/// 1. `input_value = x + params.input_offset`
/// 2. If `input_value >= 0`:
///    `result = multiply_by_quantized_multiplier(input_value,
///        params.output_multiplier_1, params.output_shift_1)`
///    Else:
///    `alpha_val = params.alpha_offset + alpha_data[i]`
///    `result = multiply_by_quantized_multiplier(input_value * alpha_val,
///        params.output_multiplier_2, params.output_shift_2)`
/// 3. `saturating_cast(result + params.output_offset)`
///
/// # Errors
///
/// * [`KernelError::ShapeMismatch`] if input, output, or alpha_data slice
///   lengths differ.
pub fn prelu(
    input: &[i8],
    params: &ActivationParams,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    let n = input.len();
    if output.len() != n {
        return Err(KernelError::ShapeMismatch);
    }
    if params.alpha_data.len() != n {
        return Err(KernelError::ShapeMismatch);
    }

    for i in 0..n {
        let input_value = i32::from(input[i]) + params.input_offset;

        let result = if input_value >= 0 {
            multiply_by_quantized_multiplier(
                input_value,
                params.output_multiplier_1,
                params.output_shift_1,
            )
        } else {
            let alpha_val = params.alpha_offset + i32::from(params.alpha_data[i]);
            multiply_by_quantized_multiplier(
                input_value * alpha_val,
                params.output_multiplier_2,
                params.output_shift_2,
            )
        };

        output[i] = saturating_cast(result + params.output_offset);
    }

    let _ = scratch;
    Ok(())
}
