// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Reduction ops — scalar reference kernels.
//!
//! Each kernel mirrors the generator's arithmetic in
//! `tools/generate_goldens/src/ops/reductions.rs` bit-for-bit:
//!
//! 1. **mean:** i32 accumulate over reduction axes, then TFLM
//!    `QuantizedMeanOrSum` requantize (fold 1/count into the multiplier,
//!    subtract `count * input_zero_point`, `multiply_by_quantized_multiplier`).
//!
//! 2. **sum:** i32 accumulate over reduction axes, then requantize
//!    (no division — TFLite SUM keeps output type == input type).
//!
//! 3. **argmax:** pure i8 comparison returning the INDEX of the maximum
//!    along the reduction axis. Ties → first occurrence (TFLite semantics).
//!
//! 4. **argmin:** same as argmax but returns the minimum index.
//!
//! 5. **l2_norm:** accumulate squared values in i32 over the channel
//!    dimension, `integer_sqrt` to get the norm, then per-channel
//!    scaling via integer division (same rounding as the generator).
//!
//! 6. **reduce_max / reduce_min:** pure int8 comparison over the reduced
//!    axes — no quantization, no requantize. Output zero-point equals the
//!    input zero-point implicitly (TFLM `EvalMinMaxHelper` requires
//!    `TF_LITE_ENSURE_EQ` on scale and zero-point). Mirrors
//!    `MinMaxReducerCompare<int8_t>` in
//!    `tensorflow/lite/micro/kernels/reduce_common.cc` at the pinned SHA.
//!
//! # Batch constraint
//!
//! Only batch = 1 is supported. `batch > 1` returns [`KernelError::Unsupported`].
//!
//! # Quantization
//!
//! mean, sum, and l2_norm use `ReduceParams` quant fields
//! (input_offset, output_offset, output_multiplier, output_shift,
//! quantized_activation_min, quantized_activation_max).
//! argmax and argmin do NOT use these fields — they operate on raw i8 values.

use hematite_core::op_params::ReduceParams;
use hematite_core::KernelError;
use hematite_int8::{multiply_by_quantized_multiplier, saturating_cast};

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the product of a `[i32; 4]` shape array.
#[inline(always)]
fn shape_product(shape: &[i32; 4]) -> usize {
    shape[0] as usize * shape[1] as usize * shape[2] as usize * shape[3] as usize
}

/// Clamp `value` to `[min, max]` and saturating-cast to i8.
#[inline(always)]
fn clamp_i8(value: i32, min: i32, max: i32) -> i8 {
    if value > max {
        saturating_cast(max)
    } else if value < min {
        saturating_cast(min)
    } else {
        saturating_cast(value)
    }
}

/// Integer square root via binary search — mirrors
/// `tflm_math::integer_sqrt` exactly.
///
/// Returns `floor(sqrt(n))` for `n >= 0`.
#[inline]
fn integer_sqrt(n: u64) -> u32 {
    if n == 0 {
        return 0;
    }
    let mut low: u32 = 0;
    let mut high: u32 = 0xFFFFu32;
    if n > 1_000_000_000_000_000_000 {
        high = u32::MAX;
    }
    while low < high {
        let mid = low + (high - low) / 2 + (high - low) % 2;
        if u64::from(mid) * u64::from(mid) <= n {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    low
}

/// Round integer division: `(numerator + denominator/2) / denominator`
/// for non-negative values; `(numerator - denominator/2) / denominator`
/// for negative. Same as `round_half_away_zero` but not inline-always
/// since called from l2_norm loop.
fn round_div(numerator: i32, denominator: i32) -> i32 {
    let half = denominator / 2;
    if numerator >= 0 {
        (numerator + half) / denominator
    } else {
        (numerator - half) / denominator
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Reduction kernels
// ═══════════════════════════════════════════════════════════════════════════

/// Reduce-mean — scalar reference kernel.
///
/// Mirrors TFLM `reference_ops::QuantizedMeanOrSum` (compute_sum == false)
/// at the golden pin `18b9e6f2a8c5a9518e588f59c2ba16ef7ef9d551`.
///
/// # Algorithm
///
/// 1. i32 accumulate over the reduction axes
/// 2. Fold the `1/count` divisor into the multiplier (truncating i64
///    division), per TFLM: `shift = min(63 - clz(count), 32)` then
///    `min(shift, 31 + output_shift)`;
///    `mult = (mult << shift) / count`; `shift -= shift`.
/// 3. Per output: `shifted = acc - input_zero_point * count`, requantize via
///    `multiply_by_quantized_multiplier(shifted, mult, shift)` +
///    output_offset + clamp. (`input_zero_point = -input_offset`.)
///
/// # Errors
///
/// * [`KernelError::Unsupported`] if `input_shape[0] > 1`.
/// * [`KernelError::ShapeMismatch`] if slice lengths disagree.
pub fn mean(
    input: &[i8],
    params: &ReduceParams,
    output: &mut [i8],
) -> Result<(), KernelError> {
    if params.input_shape[0] != 1 {
        return Err(KernelError::Unsupported);
    }

    let in_len = shape_product(&params.input_shape);
    let out_len = shape_product(&params.output_shape);
    if input.len() != in_len || output.len() != out_len {
        return Err(KernelError::ShapeMismatch);
    }

    // Build a boolean mask: which axes are reduced?
    // We interpret param.axis[0..axis_count] as the axes to reduce.
    let mut reduce_mask = [false; 4];
    for i in 0..(params.axis_count as usize).min(4) {
        let ax = params.axis[i] as usize;
        if ax < 4 {
            reduce_mask[ax] = true;
        }
    }

    // Count reduced elements per output position
    let in_shape = params.input_shape;
    let out_shape = params.output_shape;
    let in_h = in_shape[1] as usize;
    let in_w = in_shape[2] as usize;
    let in_c = in_shape[3] as usize;

    // For each output element, accumulate the corresponding input elements
    // Over the reduced axes.
    // Strategy: iterate output positions, inner loop over the reduced axes.

    // Compute strides for indexing
    let in_stride_c: usize = 1;
    let in_stride_w: usize = in_c;
    let in_stride_h: usize = in_w * in_c;
    // batch_stride = in_h * in_w * in_c (but batch is always 0 since batch=1)

    // Determine the reduction count per axis
    let count_h = if reduce_mask[1] { in_h } else { 1usize };
    let count_w = if reduce_mask[2] { in_w } else { 1usize };
    let count_c = if reduce_mask[3] { in_c } else { 1usize };
    let total_count = (count_h * count_w * count_c) as i32;

    let out_h = out_shape[1] as usize;
    let out_w = out_shape[2] as usize;
    let out_c = out_shape[3] as usize;

    let mult = params.output_multiplier;
    let shift = params.output_shift;
    let out_off = params.output_offset;
    let act_min = params.quantized_activation_min;
    let act_max = params.quantized_activation_max;

    // TFLM `QuantizedMeanOrSum` (compute_sum == false): fold the 1/count
    // divisor into the multiplier using truncating i64 division.
    let count: u64 = total_count as u64;
    let in_zp: i32 = -params.input_offset;
    if count == 0 {
        output.fill(0);
        return Ok(());
    }
    let mut mshift = (63 - count.leading_zeros() as i32).min(32);
    mshift = mshift.min(31 + shift);
    let mean_mult: i32 = (((mult as i64) << mshift) / count as i64) as i32;
    let mean_shift: i32 = shift - mshift;

    for oh in 0..out_h {
        for ow in 0..out_w {
            for oc in 0..out_c {
                let mut acc: i32 = 0;
                // Iterate reduced dimensions
                let h_start = if reduce_mask[1] { 0 } else { oh * (in_h / out_h.max(1)) };
                let h_end = if reduce_mask[1] { in_h } else { h_start + 1 };
                let w_start = if reduce_mask[2] { 0 } else { ow * (in_w / out_w.max(1)) };
                let w_end = if reduce_mask[2] { in_w } else { w_start + 1 };
                let c_start = if reduce_mask[3] { 0 } else { oc * (in_c / out_c.max(1)) };
                let c_end = if reduce_mask[3] { in_c } else { c_start + 1 };

                for ih in h_start..h_end {
                    for iw in w_start..w_end {
                        for ic in c_start..c_end {
                            let idx = ih * in_stride_h + iw * in_stride_w + ic * in_stride_c;
                            acc += i32::from(input[idx]);
                        }
                    }
                }

                let shifted = (acc as i64 - in_zp as i64 * count as i64) as i32;
                let scaled = multiply_by_quantized_multiplier(shifted, mean_mult, mean_shift);
                let val = (scaled + out_off).max(act_min).min(act_max);
                let out_idx = oh * (out_w * out_c) + ow * out_c + oc;
                output[out_idx] = clamp_i8(val, act_min, act_max);
            }
        }
    }

    Ok(())
}

/// Reduce-sum — scalar reference kernel.
///
/// Mirrors `tools/generate_goldens/src/ops/reductions.rs::generate_sum`.
///
/// # Arrowics
///
/// i32 accumulate over the reduction axes, then requantize directly
/// (no division — TFLite SUM keeps output type == input type).
///
/// # Errors
///
/// * [`KernelError::Unsupported`] if `input_shape[0] > 1`.
/// * [`KernelError::ShapeMismatch`] if slice lengths disagree.
pub fn sum(
    input: &[i8],
    params: &ReduceParams,
    output: &mut [i8],
) -> Result<(), KernelError> {
    if params.input_shape[0] != 1 {
        return Err(KernelError::Unsupported);
    }

    let in_len = shape_product(&params.input_shape);
    let out_len = shape_product(&params.output_shape);
    if input.len() != in_len || output.len() != out_len {
        return Err(KernelError::ShapeMismatch);
    }

    let mut reduce_mask = [false; 4];
    for i in 0..(params.axis_count as usize).min(4) {
        let ax = params.axis[i] as usize;
        if ax < 4 {
            reduce_mask[ax] = true;
        }
    }

    let in_shape = params.input_shape;
    let out_shape = params.output_shape;
    let in_h = in_shape[1] as usize;
    let in_w = in_shape[2] as usize;
    let in_c = in_shape[3] as usize;

    let in_stride_c: usize = 1;
    let in_stride_w: usize = in_c;
    let in_stride_h: usize = in_w * in_c;

    let out_h = out_shape[1] as usize;
    let out_w = out_shape[2] as usize;
    let out_c = out_shape[3] as usize;

    let mult = params.output_multiplier;
    let shift = params.output_shift;
    let out_off = params.output_offset;
    let act_min = params.quantized_activation_min;
    let act_max = params.quantized_activation_max;

    for oh in 0..out_h {
        for ow in 0..out_w {
            for oc in 0..out_c {
                let mut acc: i32 = 0;
                let h_start = if reduce_mask[1] { 0 } else { oh * (in_h / out_h.max(1)) };
                let h_end = if reduce_mask[1] { in_h } else { h_start + 1 };
                let w_start = if reduce_mask[2] { 0 } else { ow * (in_w / out_w.max(1)) };
                let w_end = if reduce_mask[2] { in_w } else { w_start + 1 };
                let c_start = if reduce_mask[3] { 0 } else { oc * (in_c / out_c.max(1)) };
                let c_end = if reduce_mask[3] { in_c } else { c_start + 1 };

                for ih in h_start..h_end {
                    for iw in w_start..w_end {
                        for ic in c_start..c_end {
                            let idx = ih * in_stride_h + iw * in_stride_w + ic * in_stride_c;
                            acc += i32::from(input[idx]);
                        }
                    }
                }

                // Sum: direct requantize, no division
                let scaled = multiply_by_quantized_multiplier(acc, mult, shift);
                let val = (scaled + out_off).max(act_min).min(act_max);
                let out_idx = oh * (out_w * out_c) + ow * out_c + oc;
                output[out_idx] = clamp_i8(val, act_min, act_max);
            }
        }
    }

    Ok(())
}

/// ArgMax — scalar reference kernel.
///
/// Mirrors `tools/generate_goldens/src/ops/reductions.rs::generate_argmax`.
///
/// Returns the INDEX of the maximum value along the reduction axis.
/// Pure i8 comparison — no quantization. Ties → first occurrence.
///
/// # Output encoding
///
/// The index is returned as i8 (suitable for small axis ranges).
/// `params.output_type` distinguishes i32/i64 but we use i8 for the
/// reference golden (axes are small in fixtures).
///
/// # Errors
///
/// * [`KernelError::Unsupported`] if `input_shape[0] > 1`.
/// * [`KernelError::ShapeMismatch`] if slice lengths disagree.
pub fn arg_max(
    input: &[i8],
    params: &ReduceParams,
    output: &mut [i8],
) -> Result<(), KernelError> {
    if params.input_shape[0] != 1 {
        return Err(KernelError::Unsupported);
    }

    let in_len = shape_product(&params.input_shape);
    let out_len = shape_product(&params.output_shape);
    if input.len() != in_len || output.len() != out_len {
        return Err(KernelError::ShapeMismatch);
    }

    let mut reduce_mask = [false; 4];
    for i in 0..(params.axis_count as usize).min(4) {
        let ax = params.axis[i] as usize;
        if ax < 4 {
            reduce_mask[ax] = true;
        }
    }

    let in_shape = params.input_shape;
    let out_shape = params.output_shape;
    let in_h = in_shape[1] as usize;
    let in_w = in_shape[2] as usize;
    let in_c = in_shape[3] as usize;

    let in_stride_c: usize = 1;
    let in_stride_w: usize = in_c;
    let in_stride_h: usize = in_w * in_c;

    let out_h = out_shape[1] as usize;
    let out_w = out_shape[2] as usize;
    let out_c = out_shape[3] as usize;

    for oh in 0..out_h {
        for ow in 0..out_w {
            for oc in 0..out_c {
                let mut max_val: i8 = i8::MIN;
                let mut max_idx: i8 = 0;

                let h_start = if reduce_mask[1] { 0 } else { oh * (in_h / out_h.max(1)) };
                let h_end = if reduce_mask[1] { in_h } else { h_start + 1 };
                let w_start = if reduce_mask[2] { 0 } else { ow * (in_w / out_w.max(1)) };
                let w_end = if reduce_mask[2] { in_w } else { w_start + 1 };
                let c_start = if reduce_mask[3] { 0 } else { oc * (in_c / out_c.max(1)) };
                let c_end = if reduce_mask[3] { in_c } else { c_start + 1 };

                let mut axis_idx: i8 = 0;
                for ih in h_start..h_end {
                    for iw in w_start..w_end {
                        for ic in c_start..c_end {
                            let idx = ih * in_stride_h + iw * in_stride_w + ic * in_stride_c;
                            let val = input[idx];
                            if val > max_val {
                                max_val = val;
                                max_idx = axis_idx;
                            }
                            axis_idx += 1;
                        }
                    }
                }

                let out_idx = oh * (out_w * out_c) + ow * out_c + oc;
                output[out_idx] = max_idx;
            }
        }
    }

    Ok(())
}

/// ArgMin — scalar reference kernel.
///
/// Mirrors `tools/generate_goldens/src/ops/reductions.rs::generate_argmin`.
///
/// Returns the INDEX of the minimum value along the reduction axis.
/// Pure i8 comparison — no quantization. Ties → first occurrence.
///
/// # Errors
///
/// * [`KernelError::Unsupported`] if `input_shape[0] > 1`.
/// * [`KernelError::ShapeMismatch`] if slice lengths disagree.
pub fn arg_min(
    input: &[i8],
    params: &ReduceParams,
    output: &mut [i8],
) -> Result<(), KernelError> {
    if params.input_shape[0] != 1 {
        return Err(KernelError::Unsupported);
    }

    let in_len = shape_product(&params.input_shape);
    let out_len = shape_product(&params.output_shape);
    if input.len() != in_len || output.len() != out_len {
        return Err(KernelError::ShapeMismatch);
    }

    let mut reduce_mask = [false; 4];
    for i in 0..(params.axis_count as usize).min(4) {
        let ax = params.axis[i] as usize;
        if ax < 4 {
            reduce_mask[ax] = true;
        }
    }

    let in_shape = params.input_shape;
    let out_shape = params.output_shape;
    let in_h = in_shape[1] as usize;
    let in_w = in_shape[2] as usize;
    let in_c = in_shape[3] as usize;

    let in_stride_c: usize = 1;
    let in_stride_w: usize = in_c;
    let in_stride_h: usize = in_w * in_c;

    let out_h = out_shape[1] as usize;
    let out_w = out_shape[2] as usize;
    let out_c = out_shape[3] as usize;

    for oh in 0..out_h {
        for ow in 0..out_w {
            for oc in 0..out_c {
                let mut min_val: i8 = i8::MAX;
                let mut min_idx: i8 = 0;

                let h_start = if reduce_mask[1] { 0 } else { oh * (in_h / out_h.max(1)) };
                let h_end = if reduce_mask[1] { in_h } else { h_start + 1 };
                let w_start = if reduce_mask[2] { 0 } else { ow * (in_w / out_w.max(1)) };
                let w_end = if reduce_mask[2] { in_w } else { w_start + 1 };
                let c_start = if reduce_mask[3] { 0 } else { oc * (in_c / out_c.max(1)) };
                let c_end = if reduce_mask[3] { in_c } else { c_start + 1 };

                let mut axis_idx: i8 = 0;
                for ih in h_start..h_end {
                    for iw in w_start..w_end {
                        for ic in c_start..c_end {
                            let idx = ih * in_stride_h + iw * in_stride_w + ic * in_stride_c;
                            let val = input[idx];
                            if val < min_val {
                                min_val = val;
                                min_idx = axis_idx;
                            }
                            axis_idx += 1;
                        }
                    }
                }

                let out_idx = oh * (out_w * out_c) + ow * out_c + oc;
                output[out_idx] = min_idx;
            }
        }
    }

    Ok(())
}

/// L2 normalization — scalar reference kernel.
///
/// Mirrors `tools/generate_goldens/src/ops/reductions.rs::generate_l2_norm`.
///
/// # Arrowics
///
/// For each spatial position, accumulate squared channel values in i32,
/// compute integer sqrt of the sum, then per-channel:
/// `result = round(mbm(input, out_mult, out_shift) / norm)`.
///
/// # Overflow guard
///
/// Input values are i8 (max abs = 128). Max squared per element = 16384.
/// With up to 1024 channels, max accumulation = 16384 × 1024 = 16,777,216,
/// well within i32::MAX = 2,147,483,647 — no saturation needed for
/// practical channel counts.
///
/// # Errors
///
/// * [`KernelError::Unsupported`] if `input_shape[0] > 1`.
/// * [`KernelError::ShapeMismatch`] if slice lengths disagree.
pub fn l2_norm(
    input: &[i8],
    params: &ReduceParams,
    output: &mut [i8],
) -> Result<(), KernelError> {
    if params.input_shape[0] != 1 {
        return Err(KernelError::Unsupported);
    }

    let in_len = shape_product(&params.input_shape);
    let out_len = shape_product(&params.output_shape);
    if input.len() != in_len || output.len() != out_len {
        return Err(KernelError::ShapeMismatch);
    }

    let in_shape = params.input_shape;
    let _out_shape = params.output_shape;
    let in_h = in_shape[1] as usize;
    let in_w = in_shape[2] as usize;
    let channels = in_shape[3] as usize;

    let mult = params.output_multiplier;
    let shift = params.output_shift;
    let out_off = params.output_offset;
    let act_min = params.quantized_activation_min;
    let act_max = params.quantized_activation_max;

    let num_spatial = in_h * in_w;
    for pos in 0..num_spatial {
        let base = pos * channels;

        // Accumulate squared sum over channels (all values i8, safe in i32)
        let mut sq_sum: u64 = 0;
        for c in 0..channels {
            let v = i32::from(input[base + c]);
            sq_sum += (v * v) as u64;
        }

        let norm = i32::try_from(integer_sqrt(sq_sum)).unwrap_or(i32::MAX);

        for c in 0..channels {
            let inp = i32::from(input[base + c]);
            let scaled = multiply_by_quantized_multiplier(inp, mult, shift);
            // Round division by norm, matching generator
            let result = if norm == 0 {
                0
            } else {
                round_div(scaled, norm)
            };
            let val = (result + out_off).max(act_min).min(act_max);
            output[base + c] = clamp_i8(val, act_min, act_max);
        }
    }

    Ok(())
}

/// Reduce-max — scalar reference kernel.
///
/// Mirrors `tools/generate_goldens/src/ops/reductions.rs::generate_reduce_max`
/// (TFLM `MinMaxReducerCompare<int8_t>`): pure int8 comparison over the
/// reduced axes, initial value `i8::MIN`, update on `in > current`.
/// No quantization, no requantize — the output zero-point equals the input
/// zero-point implicitly.
///
/// # Errors
///
/// * [`KernelError::Unsupported`] if `input_shape[0] > 1`.
/// * [`KernelError::ShapeMismatch`] if slice lengths disagree.
pub fn reduce_max(
    input: &[i8],
    params: &ReduceParams,
    output: &mut [i8],
) -> Result<(), KernelError> {
    if params.input_shape[0] != 1 {
        return Err(KernelError::Unsupported);
    }

    let in_len = shape_product(&params.input_shape);
    let out_len = shape_product(&params.output_shape);
    if input.len() != in_len || output.len() != out_len {
        return Err(KernelError::ShapeMismatch);
    }

    let mut reduce_mask = [false; 4];
    for i in 0..(params.axis_count as usize).min(4) {
        let ax = params.axis[i] as usize;
        if ax < 4 {
            reduce_mask[ax] = true;
        }
    }

    let in_shape = params.input_shape;
    let out_shape = params.output_shape;
    let in_h = in_shape[1] as usize;
    let in_w = in_shape[2] as usize;
    let in_c = in_shape[3] as usize;

    let in_stride_c: usize = 1;
    let in_stride_w: usize = in_c;
    let in_stride_h: usize = in_w * in_c;

    let out_h = out_shape[1] as usize;
    let out_w = out_shape[2] as usize;
    let out_c = out_shape[3] as usize;

    for oh in 0..out_h {
        for ow in 0..out_w {
            for oc in 0..out_c {
                let mut current: i8 = i8::MIN;

                let h_start = if reduce_mask[1] { 0 } else { oh * (in_h / out_h.max(1)) };
                let h_end = if reduce_mask[1] { in_h } else { h_start + 1 };
                let w_start = if reduce_mask[2] { 0 } else { ow * (in_w / out_w.max(1)) };
                let w_end = if reduce_mask[2] { in_w } else { w_start + 1 };
                let c_start = if reduce_mask[3] { 0 } else { oc * (in_c / out_c.max(1)) };
                let c_end = if reduce_mask[3] { in_c } else { c_start + 1 };

                for ih in h_start..h_end {
                    for iw in w_start..w_end {
                        for ic in c_start..c_end {
                            let idx = ih * in_stride_h + iw * in_stride_w + ic * in_stride_c;
                            let val = input[idx];
                            if val > current {
                                current = val;
                            }
                        }
                    }
                }

                let out_idx = oh * (out_w * out_c) + ow * out_c + oc;
                output[out_idx] = current;
            }
        }
    }

    Ok(())
}

/// Reduce-min — scalar reference kernel.
///
/// Mirrors `tools/generate_goldens/src/ops/reductions.rs::generate_reduce_min`
/// (TFLM `MinMaxReducerCompare<int8_t>`): pure int8 comparison over the
/// reduced axes, initial value `i8::MAX`, update on `in < current`.
/// No quantization, no requantize.
///
/// # Errors
///
/// * [`KernelError::Unsupported`] if `input_shape[0] > 1`.
/// * [`KernelError::ShapeMismatch`] if slice lengths disagree.
pub fn reduce_min(
    input: &[i8],
    params: &ReduceParams,
    output: &mut [i8],
) -> Result<(), KernelError> {
    if params.input_shape[0] != 1 {
        return Err(KernelError::Unsupported);
    }

    let in_len = shape_product(&params.input_shape);
    let out_len = shape_product(&params.output_shape);
    if input.len() != in_len || output.len() != out_len {
        return Err(KernelError::ShapeMismatch);
    }

    let mut reduce_mask = [false; 4];
    for i in 0..(params.axis_count as usize).min(4) {
        let ax = params.axis[i] as usize;
        if ax < 4 {
            reduce_mask[ax] = true;
        }
    }

    let in_shape = params.input_shape;
    let out_shape = params.output_shape;
    let in_h = in_shape[1] as usize;
    let in_w = in_shape[2] as usize;
    let in_c = in_shape[3] as usize;

    let in_stride_c: usize = 1;
    let in_stride_w: usize = in_c;
    let in_stride_h: usize = in_w * in_c;

    let out_h = out_shape[1] as usize;
    let out_w = out_shape[2] as usize;
    let out_c = out_shape[3] as usize;

    for oh in 0..out_h {
        for ow in 0..out_w {
            for oc in 0..out_c {
                let mut current: i8 = i8::MAX;

                let h_start = if reduce_mask[1] { 0 } else { oh * (in_h / out_h.max(1)) };
                let h_end = if reduce_mask[1] { in_h } else { h_start + 1 };
                let w_start = if reduce_mask[2] { 0 } else { ow * (in_w / out_w.max(1)) };
                let w_end = if reduce_mask[2] { in_w } else { w_start + 1 };
                let c_start = if reduce_mask[3] { 0 } else { oc * (in_c / out_c.max(1)) };
                let c_end = if reduce_mask[3] { in_c } else { c_start + 1 };

                for ih in h_start..h_end {
                    for iw in w_start..w_end {
                        for ic in c_start..c_end {
                            let idx = ih * in_stride_h + iw * in_stride_w + ic * in_stride_c;
                            let val = input[idx];
                            if val < current {
                                current = val;
                            }
                        }
                    }
                }

                let out_idx = oh * (out_w * out_c) + ow * out_c + oc;
                output[out_idx] = current;
            }
        }
    }

    Ok(())
}
