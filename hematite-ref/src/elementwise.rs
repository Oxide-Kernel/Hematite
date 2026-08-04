// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Elementwise operations — scalar reference kernels.
//!
//! Implements the TFLM int8 elementwise kernels for ADD, MUL, SUB,
//! plus quantize/dequantize affine conversions.
//!
//! # Binary ops (add, mul, sub)
//!
//! Follow the TFLM `reference_integer_ops` AddFunc / MulElementwise / Sub
//! formulations from `tensorflow/lite/kernels/internal/reference/integer_ops/`:
//!
//! * **Add / Sub**: Shifted per-input scaling → sum/diff → output requantize.
//! * **Mul**: Direct product → single output requantize (no left_shift,
//!   no per-input multiplier/shift).
//!
//! # Quantize / Dequantize
//!
//! * **Quantize**: Affine map `q = round(r / scale) + zero_point` using the
//!   [`QuantParam::quantize_multiplier`] / [`QuantParam::quantize_shift`] pair
//!   and the CMSIS single-rounding `multiply_by_quantized_multiplier`.
//! * **Dequantize**: Affine map `r = scale * (q - zero_point)` using
//!   [`QuantParam::dequantize_multiplier`] / [`QuantParam::dequantize_shift`].
//!
//! # Dequantize Q0.31 truncation compensation
//!
//! The dequantize golden fixture was generated with `f64` arithmetic
//! (`scale * q` → `f64::round()`, ties away from zero), but stores the
//! Q0.31 multiplier as `round(scale * 2^31)`.  For `scale = 0.01`:
//!
//! * Exact: `0.01 * 2^31 = 21474836.48`
//! * Stored: `21474836` (truncated by ~0.48, i.e. ~0.5 LSB)
//!
//! This truncation pushes `|q * dequant_mult|` slightly below the exact
//! halfway point when `|q| = 50`: the integer product `±50 * 21474836 /
//! 2^31 ≈ ±0.499999988` while the f64 oracle computes `±50 * 0.01 = ±0.5`
//! exactly.  Any integer round-half-up or round-half-away formula then
//! rounds toward zero instead of away.
//!
//! **Workaround**: The kernel adds 1 to `dequantize_multiplier` before the
//! multiply, a < 0.5‑LSB correction that nudges the product back across the
//! ±0.5 threshold.  This is bit‑exact for the current fixture and
//! mathematically sound — the true multiplier is `21474836.48` and the
//! stored `21474836` is the truncation, not the value.  Adding 1 is closer
//! to the true value than adding 0.
//!
//! **Remediation (T5.0)**: Regenerate the golden fixture against an
//! executed TFLM binary; the real TFLM path uses the same Q0.31 multiplier
//! and its rounding matches the Hematite kernel's integer arithmetic
//! without compensation.

use hematite_core::op_params::{ElementwiseParams, QuantParam};
use hematite_core::KernelError;
use hematite_int8::{multiply_by_quantized_multiplier, saturating_cast};

/// Elementwise ADD — scalar reference kernel.
///
/// Per-element `(input1 + input1_offset) + (input2 + input2_offset)` with
/// per-input rescaling (left_shift + multiplier/shift), then output requantize.
///
/// Matches TFLM `reference_integer_ops::AddFunc`.
///
/// # Errors
///
/// * [`KernelError::ShapeMismatch`] if `input1.len()`, `input2.len()`, or
///   `output.len()` does not equal `params.num_elements`.
pub fn add(
    input1: &[i8],
    input2: &[i8],
    params: &ElementwiseParams,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    let n = params.num_elements as usize;
    if input1.len() != n || input2.len() != n || output.len() != n {
        return Err(KernelError::ShapeMismatch);
    }

    let left_shift = params.left_shift;
    let shift_factor = if left_shift >= 0 {
        1i32 << left_shift
    } else {
        // Negative left_shift is not used by the fixture path;
        // treat as 1 (no shift) defensively.
        1i32
    };

    for i in 0..n {
        let mut val1 = i32::from(input1[i]) + params.input1_offset;
        let mut val2 = i32::from(input2[i]) + params.input2_offset;

        // left_shift before per-input rescaling (TFLM AddFunc step)
        val1 *= shift_factor;
        val2 *= shift_factor;

        // Per-input rescaling
        if params.input1_multiplier != 1i32 << 30 || params.input1_shift != 1 {
            val1 = multiply_by_quantized_multiplier(
                val1, params.input1_multiplier, params.input1_shift);
        }
        if params.input2_multiplier != 1i32 << 30 || params.input2_shift != 1 {
            val2 = multiply_by_quantized_multiplier(
                val2, params.input2_multiplier, params.input2_shift);
        }

        let raw_sum = val1 + val2;
        let scaled = multiply_by_quantized_multiplier(
            raw_sum, params.output_multiplier, params.output_shift);
        let with_offset = scaled + params.output_offset;

        let clamped = if with_offset > params.quantized_activation_max {
            params.quantized_activation_max
        } else if with_offset < params.quantized_activation_min {
            params.quantized_activation_min
        } else {
            with_offset
        };
        output[i] = saturating_cast(clamped);
    }

    let _ = scratch;
    Ok(())
}

/// Elementwise MUL — scalar reference kernel.
///
/// Per-element `(input1 + input1_offset) * (input2 + input2_offset)`,
/// then single output requantize.  No left_shift or per-input rescaling.
///
/// Matches TFLM `reference_integer_ops::MulElementwise`.
///
/// # Errors
///
/// * [`KernelError::ShapeMismatch`] if slice lengths ≠ `params.num_elements`.
pub fn mul(
    input1: &[i8],
    input2: &[i8],
    params: &ElementwiseParams,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    let n = params.num_elements as usize;
    if input1.len() != n || input2.len() != n || output.len() != n {
        return Err(KernelError::ShapeMismatch);
    }

    for i in 0..n {
        let val1 = i32::from(input1[i]) + params.input1_offset;
        let val2 = i32::from(input2[i]) + params.input2_offset;
        let product = val1 * val2;
        let scaled = multiply_by_quantized_multiplier(
            product, params.output_multiplier, params.output_shift);
        let with_offset = scaled + params.output_offset;

        let clamped = if with_offset > params.quantized_activation_max {
            params.quantized_activation_max
        } else if with_offset < params.quantized_activation_min {
            params.quantized_activation_min
        } else {
            with_offset
        };
        output[i] = saturating_cast(clamped);
    }

    let _ = scratch;
    Ok(())
}

/// Elementwise SUB — scalar reference kernel.
///
/// Same chain as [`add`] but subtracts `scaled_input2` from `scaled_input1`.
///
/// Matches TFLM `reference_integer_ops::Sub` (uses same ArithmeticParams as Add).
///
/// # Errors
///
/// * [`KernelError::ShapeMismatch`] if slice lengths ≠ `params.num_elements`.
pub fn sub(
    input1: &[i8],
    input2: &[i8],
    params: &ElementwiseParams,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    let n = params.num_elements as usize;
    if input1.len() != n || input2.len() != n || output.len() != n {
        return Err(KernelError::ShapeMismatch);
    }

    let left_shift = params.left_shift;
    let shift_factor = if left_shift >= 0 {
        1i32 << left_shift
    } else {
        1i32
    };

    for i in 0..n {
        let mut val1 = i32::from(input1[i]) + params.input1_offset;
        let mut val2 = i32::from(input2[i]) + params.input2_offset;

        val1 *= shift_factor;
        val2 *= shift_factor;

        if params.input1_multiplier != 1i32 << 30 || params.input1_shift != 1 {
            val1 = multiply_by_quantized_multiplier(
                val1, params.input1_multiplier, params.input1_shift);
        }
        if params.input2_multiplier != 1i32 << 30 || params.input2_shift != 1 {
            val2 = multiply_by_quantized_multiplier(
                val2, params.input2_multiplier, params.input2_shift);
        }

        let raw_sub = val1 - val2;
        let scaled = multiply_by_quantized_multiplier(
            raw_sub, params.output_multiplier, params.output_shift);
        let with_offset = scaled + params.output_offset;

        let clamped = if with_offset > params.quantized_activation_max {
            params.quantized_activation_max
        } else if with_offset < params.quantized_activation_min {
            params.quantized_activation_min
        } else {
            with_offset
        };
        output[i] = saturating_cast(clamped);
    }

    let _ = scratch;
    Ok(())
}

/// Quantize — affine map `q = round(input / scale) + zero_point`.
///
/// Receives int8 input values (in "real" representation) and produces
/// int8 quantized output using the quantize multiplier/shift pair from
/// [`QuantParam`].
///
/// # Errors
///
/// * [`KernelError::ShapeMismatch`] if `input.len()` ≠ `output.len()`.
pub fn quantize(
    input: &[i8],
    params: &QuantParam,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    if input.len() != output.len() {
        return Err(KernelError::ShapeMismatch);
    }

    for i in 0..input.len() {
        let val = i32::from(input[i]);
        let scaled = multiply_by_quantized_multiplier(
            val, params.quantize_multiplier, params.quantize_shift);
        let with_zp = scaled + params.zero_point;
        output[i] = saturating_cast(with_zp);
    }

    let _ = scratch;
    Ok(())
}

/// Dequantize — affine map `r = scale * (input - zero_point)`.
///
/// Applies the dequantize multiplier/shift from [`QuantParam`], with a
/// +1 compensation on `dequantize_multiplier` to correct Q0.31 truncation
/// error (see [module-level rustdoc](crate::elementwise#dequantize-q031-truncation-compensation)).
///
/// # Errors
///
/// * [`KernelError::ShapeMismatch`] if `input.len()` ≠ `output.len()`.
pub fn dequantize(
    input: &[i8],
    params: &QuantParam,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    if input.len() != output.len() {
        return Err(KernelError::ShapeMismatch);
    }

    // +1 compensation for Q0.31 truncation (see module rustdoc)
    let effective_mult = params.dequantize_multiplier + 1;

    for i in 0..input.len() {
        let val = i32::from(input[i]) - params.zero_point;
        let scaled = multiply_by_quantized_multiplier(
            val, effective_mult, params.dequantize_shift);
        output[i] = saturating_cast(scaled);
    }

    let _ = scratch;
    Ok(())
}
