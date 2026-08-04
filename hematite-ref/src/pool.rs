// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! 2D pooling — scalar reference kernel.
//!
//! Implements the TFLM int8 reference pool loop order, matching
//! `tools/generate_goldens/src/ops/pool.rs` bit-for-bit:
//!
//! 1. **AveragePool:** i32 accumulate input values, divide by pool size
//!    with round-half-away-from-zero semantics, clamp, saturating cast to i8.
//! 2. **MaxPool:** pure i8 compare over input values (no input offset),
//!    clamp to activation range.
//! 3. **GlobalAveragePool:** average over full spatial extent, same
//!    requantize as average pool.
//!
//! # Batch constraint
//!
//! Only batch = 1 is supported (per the static-shape constraint).
//! `batch > 1` returns [`KernelError::Unsupported`].
//!
//! # No offset arithmetic
//!
//! Unlike Conv2D, pooling ops do NOT apply `input_offset` or
//! `output_offset`.  The TFLM int8 reference pool path compares (max)
//! or averages raw input values directly.  `PoolParams` carries no
//! offset fields for this reason.

use hematite_core::op_params::PoolParams;
use hematite_core::KernelError;
use hematite_int8::saturating_cast;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the product of a `[i32; 4]` shape array.
#[inline(always)]
fn shape_product(shape: &[i32; 4]) -> usize {
    shape[0] as usize * shape[1] as usize * shape[2] as usize * shape[3] as usize
}

/// Round-half-away-from-zero integer division: `numerator / denominator`
/// with halves rounded away from zero.
///
/// Matches the semantics used by the pool golden generator:
///
/// * `(acc + count / 2) / count` for positive `acc`
/// * `(acc - count / 2) / count` for negative `acc`
///
/// When `count` is a power of two, this is equivalent to
/// `rounding_divide_by_pot(acc, log2(count))`.
///
/// # Panics
///
/// Panics in debug if `denominator == 0`.
#[inline(always)]
fn round_half_away_zero(numerator: i32, denominator: i32) -> i32 {
    debug_assert!(denominator > 0, "denominator must be positive");
    let half = denominator / 2;
    if numerator > 0 {
        (numerator + half) / denominator
    } else {
        (numerator - half) / denominator
    }
}

/// Clamp `value` to `[min, max]` and saturating-cast to `i8`.
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

// ═══════════════════════════════════════════════════════════════════════════
// Pooling kernels
// ═══════════════════════════════════════════════════════════════════════════

/// 2D average pooling — scalar reference kernel.
///
/// Matches the TFLM int8 `reference_integer_ops::AveragePool` loop order
/// (same as `tools/generate_goldens/src/ops/pool.rs`).
///
/// # Layouts
///
/// * `input` — NHWC `[batch=1, H, W, C]`
/// * `output` — NHWC `[batch=1, OH, OW, C]`
///
/// # Arrowics
///
/// Each output element is the round-half-away-from-zero average of its
/// filter-window input values.  No input or output offset is applied
/// (the TFLM int8 reference pool path uses raw input values).
///
/// # Errors
///
/// * [`KernelError::Unsupported`] if `input_shape[0] > 1`.
/// * [`KernelError::ShapeMismatch`] if any slice length does not match the
///   declared shapes in `params`, or if channel dimensions disagree.
pub fn average_pool_2d(
    input: &[i8],
    params: &PoolParams,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    // ── Batch constraint ────────────────────────────────────────────────
    if params.input_shape[0] != 1 {
        return Err(KernelError::Unsupported);
    }

    // ── Extract dimensions ─────────────────────────────────────────────
    let input_h = params.input_shape[1];
    let input_w = params.input_shape[2];
    let channels = params.input_shape[3];

    let filter_h = params.filter_height;
    let filter_w = params.filter_width;

    let out_h = params.output_shape[1];
    let out_w = params.output_shape[2];
    let out_c = params.output_shape[3];

    // ── Slice-length validation ─────────────────────────────────────────
    if input.len() != shape_product(&params.input_shape) {
        return Err(KernelError::ShapeMismatch);
    }
    if output.len() != shape_product(&params.output_shape) {
        return Err(KernelError::ShapeMismatch);
    }
    if channels != out_c {
        return Err(KernelError::ShapeMismatch);
    }

    // ── Compute pad from spatial-shape relationship ────────────────────
    //
    // out_dim = ((in_dim + 2 * pad - filter_dim) / stride) + 1
    // → pad = ((out_dim - 1) * stride + filter_dim - in_dim) / 2
    let pad_h = ((out_h - 1) * params.stride_height + filter_h - input_h) / 2;
    let pad_w = ((out_w - 1) * params.stride_width + filter_w - input_w) / 2;

    let output_row_stride = out_w * channels;

    // ── Average pool loop (oh → ow → oc → fy → fx) ─────────────────────
    for oh in 0..out_h {
        let in_y_origin = oh * params.stride_height - pad_h;
        let fy_start = 0i32.max(-in_y_origin);
        let fy_end = filter_h.min(input_h - in_y_origin);

        for ow in 0..out_w {
            let in_x_origin = ow * params.stride_width - pad_w;
            let fx_start = 0i32.max(-in_x_origin);
            let fx_end = filter_w.min(input_w - in_x_origin);

            for oc in 0..channels {
                let c = oc as usize;
                let mut acc: i32 = 0;
                let mut count: i32 = 0;

                for fy in fy_start..fy_end {
                    let in_y = in_y_origin + fy;
                    for fx in fx_start..fx_end {
                        let in_x = in_x_origin + fx;
                        let idx =
                            (in_y * input_w + in_x) as usize * channels as usize + c;
                        acc += i32::from(input[idx]);
                        count += 1;
                    }
                }

                let result = if count == 0 {
                    0
                } else {
                    round_half_away_zero(acc, count)
                };

                let out_idx = (oh * output_row_stride + ow * channels + oc) as usize;
                output[out_idx] =
                    clamp_i8(result, params.quantized_activation_min, params.quantized_activation_max);
            }
        }
    }

    let _ = scratch;
    Ok(())
}

/// 2D max pooling — scalar reference kernel.
///
/// Matches the TFLM int8 `reference_integer_ops::MaxPool` loop order
/// (same as `tools/generate_goldens/src/ops/pool.rs`).
///
/// # Layouts
///
/// * `input` — NHWC `[batch=1, H, W, C]`
/// * `output` — NHWC `[batch=1, OH, OW, C]`
///
/// # Arrowics
///
/// Pure i8 element-wise maximum over each filter window.  No input or
/// output offset is applied — TFLM max pool compares raw input values.
///
/// # Errors
///
/// * [`KernelError::Unsupported`] if `input_shape[0] > 1`.
/// * [`KernelError::ShapeMismatch`] if any slice length does not match.
pub fn max_pool_2d(
    input: &[i8],
    params: &PoolParams,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    // ── Batch constraint ────────────────────────────────────────────────
    if params.input_shape[0] != 1 {
        return Err(KernelError::Unsupported);
    }

    let input_h = params.input_shape[1];
    let input_w = params.input_shape[2];
    let channels = params.input_shape[3];

    let filter_h = params.filter_height;
    let filter_w = params.filter_width;

    let out_h = params.output_shape[1];
    let out_w = params.output_shape[2];
    let out_c = params.output_shape[3];

    // ── Slice-length validation ─────────────────────────────────────────
    if input.len() != shape_product(&params.input_shape) {
        return Err(KernelError::ShapeMismatch);
    }
    if output.len() != shape_product(&params.output_shape) {
        return Err(KernelError::ShapeMismatch);
    }
    if channels != out_c {
        return Err(KernelError::ShapeMismatch);
    }

    let pad_h = ((out_h - 1) * params.stride_height + filter_h - input_h) / 2;
    let pad_w = ((out_w - 1) * params.stride_width + filter_w - input_w) / 2;

    let activation_min = params.quantized_activation_min;
    let activation_max = params.quantized_activation_max;
    let output_row_stride = out_w * channels;

    for oh in 0..out_h {
        let in_y_origin = oh * params.stride_height - pad_h;
        let fy_start = 0i32.max(-in_y_origin);
        let fy_end = filter_h.min(input_h - in_y_origin);

        for ow in 0..out_w {
            let in_x_origin = ow * params.stride_width - pad_w;
            let fx_start = 0i32.max(-in_x_origin);
            let fx_end = filter_w.min(input_w - in_x_origin);

            for oc in 0..channels {
                let c = oc as usize;
                let mut max_val = i8::MIN;

                for fy in fy_start..fy_end {
                    let in_y = in_y_origin + fy;
                    for fx in fx_start..fx_end {
                        let in_x = in_x_origin + fx;
                        let idx =
                            (in_y * input_w + in_x) as usize * channels as usize + c;
                        max_val = max_val.max(input[idx]);
                    }
                }

                // Clip by activation range — the generator casts activation_min/max
                // to i8 for the comparison.
                let clamped = max_val
                    .max(activation_min as i8)
                    .min(activation_max as i8);

                let out_idx = (oh * output_row_stride + ow * channels + oc) as usize;
                output[out_idx] = clamped;
            }
        }
    }

    let _ = scratch;
    Ok(())
}

/// Global average pooling — scalar reference kernel.
///
/// Averages over the full spatial extent `[H, W]` of the input, producing
/// a `[1, 1, 1, C]` output.  Uses the same round-half-away-from-zero
/// division as [`average_pool_2d`].
///
/// # Layouts
///
/// * `input` — NHWC `[batch=1, H, W, C]`
/// * `output` — NHWC `[batch=1, 1, 1, C]` (spatial dims reduced to 1)
///
/// # Errors
///
/// * [`KernelError::Unsupported`] if `input_shape[0] > 1`.
/// * [`KernelError::ShapeMismatch`] if slice lengths or channel dims disagree.
pub fn global_average_pool_2d(
    input: &[i8],
    params: &PoolParams,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    // ── Batch constraint ────────────────────────────────────────────────
    if params.input_shape[0] != 1 {
        return Err(KernelError::Unsupported);
    }

    let input_h = params.input_shape[1];
    let input_w = params.input_shape[2];
    let channels = params.input_shape[3];

    let out_c = params.output_shape[3];

    // ── Slice-length validation ─────────────────────────────────────────
    if input.len() != shape_product(&params.input_shape) {
        return Err(KernelError::ShapeMismatch);
    }
    if output.len() != shape_product(&params.output_shape) {
        return Err(KernelError::ShapeMismatch);
    }
    if channels != out_c {
        return Err(KernelError::ShapeMismatch);
    }

    let spatial_size = input_h * input_w;

    for oc in 0..channels {
        let c = oc as usize;
        let mut acc: i32 = 0;

        for ih in 0..input_h {
            for iw in 0..input_w {
                let idx = (ih * input_w + iw) as usize * channels as usize + c;
                acc += i32::from(input[idx]);
            }
        }

        let result = if spatial_size == 0 {
            0
        } else {
            round_half_away_zero(acc, spatial_size)
        };

        output[c] = clamp_i8(result, params.quantized_activation_min, params.quantized_activation_max);
    }

    let _ = scratch;
    Ok(())
}
