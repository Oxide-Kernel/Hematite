// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! 2D convolution — scalar reference kernel.
//!
//! Implements the TFLM int8 ConvEval loop order:
//!
//! 1. i32 accumulator, init `bias[oc]`
//! 2. MAC loop over filter h/w/depth (zero-padding via bounds-check)
//! 3. Per-channel requantize via `multiply_by_quantized_multiplier`
//! 4. Add output zero-point offset
//! 5. Clamp to activation range, saturating cast to i8
//!
//! # Batch constraint
//!
//! Only batch = 1 is supported (per the static-shape constraint).
//! `batch > 1` returns [`KernelError::Unsupported`].

use hematite_core::op_params::Conv2DParams;
use hematite_core::KernelError;
use hematite_int8::{multiply_by_quantized_multiplier, saturating_cast};

/// Compute the product of a `[i32; 4]` shape array.
#[inline(always)]
fn shape_product(shape: &[i32; 4]) -> usize {
    shape[0] as usize * shape[1] as usize * shape[2] as usize * shape[3] as usize
}

/// 2D convolution — scalar reference kernel.
///
/// Matches the TFLM int8 `ConvEval` reference loop order bit-for-bit.
///
/// # Layouts
///
/// * `input` — NHWC `[batch=1, H, W, Cin]`
/// * `weights` — OHWI `[Cout, FH, FW, Cin]`
/// * `bias` — per-output-channel `[Cout]`
/// * `output` — NHWC `[batch=1, OH, OW, Cout]`
///
/// # Errors
///
/// * [`KernelError::Unsupported`] if `input_shape[0] > 1` (multi-batch not
///   supported by the static-shape reference path).
/// * [`KernelError::ShapeMismatch`] if any slice length does not match the
///   declared shapes in `params`.
pub fn conv2d(
    input: &[i8],
    weights: &[i8],
    bias: &[i32],
    params: &Conv2DParams,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    // ── Batch constraint ────────────────────────────────────────────────
    if params.input_shape[0] != 1 {
        return Err(KernelError::Unsupported);
    }

    // ── Extract dimensions (all i32 for index arithmetic) ───────────────
    let input_h = params.input_shape[1];
    let input_w = params.input_shape[2];
    let input_c = params.input_shape[3];

    let filter_h = params.filter_shape[1];
    let filter_w = params.filter_shape[2];
    let filter_ic = params.filter_shape[3];

    let out_h = params.output_shape[1];
    let out_w = params.output_shape[2];
    let out_channels = params.output_shape[3];

    // ── Slice-length validation ─────────────────────────────────────────
    if input.len() != shape_product(&params.input_shape) {
        return Err(KernelError::ShapeMismatch);
    }
    if weights.len() != shape_product(&params.filter_shape) {
        return Err(KernelError::ShapeMismatch);
    }
    if bias.len() as i32 != out_channels {
        return Err(KernelError::ShapeMismatch);
    }
    if output.len() != shape_product(&params.output_shape) {
        return Err(KernelError::ShapeMismatch);
    }

    // Channel-dimension cross-check (no dynamic broadcast — Cin must match)
    if input_c != filter_ic {
        return Err(KernelError::ShapeMismatch);
    }

    // ── Derived pad values ──────────────────────────────────────────────
    // Pad is not stored explicitly in Conv2DParams; we compute it from the
    // spatial-shape relationship:
    //
    //   output_dim = ((input_dim + 2 * pad - dilated_filter_extent) / stride) + 1
    //
    // Where dilated_filter_extent = (filter_dim - 1) * dilation + 1.
    //
    // Solving for pad:  pad = ((output_dim - 1) * stride + dilated_filter_extent - input_dim) / 2
    let dilated_filter_h = (filter_h - 1) * params.dilation_height_factor + 1;
    let dilated_filter_w = (filter_w - 1) * params.dilation_width_factor + 1;

    let pad_h = ((out_h - 1) * params.stride_height + dilated_filter_h - input_h) / 2;
    let pad_w = ((out_w - 1) * params.stride_width + dilated_filter_w - input_w) / 2;

    // ── Per-channel multiplier / shift slices ───────────────────────────
    let multipliers = params.output_multiplier_per_channel;
    let shifts = params.output_shift_per_channel;

    // ── Stride precomputation for input/indexing fast paths ──────────
    let input_row_stride = input_w * input_c;
    let filter_oc_stride = filter_h * filter_w * filter_ic;
    let filter_row_stride = filter_w * filter_ic;
    let filter_col_stride = filter_ic;
    let output_row_stride = out_w * out_channels;

    // ── Accumulation loop ───────────────────────────────────────────────
    for oh in 0..out_h {
        // Input origin for this output row (before applying filter offset).
        let input_base_h = oh * params.stride_height - pad_h;

        for ow in 0..out_w {
            let input_base_w = ow * params.stride_width - pad_w;

            for oc in 0..out_channels {
                let mut acc: i32 = bias[oc as usize];

                let filter_oc_base = oc * filter_oc_stride;

                for fh in 0..filter_h {
                    let in_h = input_base_h + fh * params.dilation_height_factor;

                    // Bounds-check height once per row
                    let row_in_bounds = in_h >= 0 && in_h < input_h;

                    for fw in 0..filter_w {
                        let in_w = input_base_w + fw * params.dilation_width_factor;

                        if row_in_bounds && in_w >= 0 && in_w < input_w {
                            let input_base = (in_h * input_row_stride + in_w * input_c) as usize;
                            let filter_base = (filter_oc_base
                                + fh * filter_row_stride
                                + fw * filter_col_stride) as usize;

                            for ic in 0..filter_ic {
                                let i_val =
                                    i32::from(input[input_base + ic as usize]);
                                let w_val =
                                    i32::from(weights[filter_base + ic as usize]);

                                acc += (i_val + params.input_offset) * w_val;
                            }
                        }
                        // else: zero-padding — skip (contribute 0 to accumulator)
                    }
                }

                // Per-channel requantize + output offset + clamp
                let multiplier = multipliers[oc as usize];
                let shift = shifts[oc as usize];
                let scaled = multiply_by_quantized_multiplier(acc, multiplier, shift);
                let with_offset = scaled + params.output_offset;

                // Clamp to fused-activation bounds, saturating cast to i8
                let clamped = if with_offset > params.quantized_activation_max {
                    params.quantized_activation_max
                } else if with_offset < params.quantized_activation_min {
                    params.quantized_activation_min
                } else {
                    with_offset
                };

                let out_idx = (oh * output_row_stride + ow * out_channels + oc) as usize;
                output[out_idx] = saturating_cast(clamped);
            }
        }
    }

    let _ = scratch; // unused by scalar reference path

    Ok(())
}
