// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Depthwise 2D convolution — scalar reference kernel.
//!
//! Implements the TFLM int8 DepthwiseConvPerChannel loop order:
//!
//! 1. i32 accumulator, init `bias[oc]`
//! 2. MAC loop over filter h/w for each (in_channel, depth_multiplier) pair
//!    (zero-padding via bounds-check)
//! 3. Per-channel requantize via `multiply_by_quantized_multiplier`
//! 4. Add output zero-point offset
//! 5. Clamp to activation range, saturating cast to i8
//!
//! # Batch constraint
//!
//! Only batch = 1 is supported (per the static-shape constraint).
//! `batch > 1` returns [`KernelError::Unsupported`].
//!
//! # Layouts
//!
//! * `input` — NHWC `[batch=1, H, W, Cin]`
//! * `weights` — `[1, FH, FW, Cin * depth_multiplier]` (channel-contiguous)
//! * `bias` — per-output-channel `[Cout]` where `Cout = Cin * depth_multiplier`
//! * `output` — NHWC `[batch=1, OH, OW, Cout]`

use hematite_core::op_params::DepthwiseConv2DParams;
use hematite_core::KernelError;
use hematite_int8::{multiply_by_quantized_multiplier, saturating_cast};

/// Compute the product of a `[i32; 4]` shape array.
#[inline(always)]
fn shape_product(shape: &[i32; 4]) -> usize {
    shape[0] as usize * shape[1] as usize * shape[2] as usize * shape[3] as usize
}

/// Depthwise 2D convolution — scalar reference kernel.
///
/// Matches the TFLM int8 `DepthwiseConvPerChannel` reference loop order.
///
/// # Errors
///
/// * [`KernelError::Unsupported`] if `input_shape[0] > 1` (multi-batch not
///   supported by the static-shape reference path).
/// * [`KernelError::ShapeMismatch`] if any slice length does not match the
///   declared shapes in `params`, or if channel dimensions are inconsistent.
pub fn depthwise_conv2d(
    input: &[i8],
    weights: &[i8],
    bias: &[i32],
    params: &DepthwiseConv2DParams,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    // ── Batch constraint ────────────────────────────────────────────────
    if params.input_shape[0] != 1 {
        return Err(KernelError::Unsupported);
    }

    // ── Extract dimensions ──────────────────────────────────────────────
    let input_h = params.input_shape[1];
    let input_w = params.input_shape[2];
    let input_c = params.input_shape[3];

    let filter_h = params.filter_shape[1];
    let filter_w = params.filter_shape[2];
    let filter_channels = params.filter_shape[3]; // = Cin * depth_multiplier

    let out_h = params.output_shape[1];
    let out_w = params.output_shape[2];
    let out_c = params.output_shape[3];

    let depth_multiplier = params.depth_multiplier;

    // ── Slice-length validation ─────────────────────────────────────────
    if input.len() != shape_product(&params.input_shape) {
        return Err(KernelError::ShapeMismatch);
    }
    if weights.len() != shape_product(&params.filter_shape) {
        return Err(KernelError::ShapeMismatch);
    }
    if bias.len() as i32 != out_c {
        return Err(KernelError::ShapeMismatch);
    }
    if output.len() != shape_product(&params.output_shape) {
        return Err(KernelError::ShapeMismatch);
    }

    // Channel-dimension cross-checks
    if input_c * depth_multiplier != out_c {
        return Err(KernelError::ShapeMismatch);
    }
    if filter_channels != out_c {
        return Err(KernelError::ShapeMismatch);
    }

    // ── Derived pad values ──────────────────────────────────────────────
    let dilated_filter_h = (filter_h - 1) * params.dilation_height_factor + 1;
    let dilated_filter_w = (filter_w - 1) * params.dilation_width_factor + 1;

    let pad_h = ((out_h - 1) * params.stride_height + dilated_filter_h - input_h) / 2;
    let pad_w = ((out_w - 1) * params.stride_width + dilated_filter_w - input_w) / 2;

    // ── Per-channel multiplier / shift slices ───────────────────────────
    let multipliers = params.output_multiplier_per_channel;
    let shifts = params.output_shift_per_channel;

    // ── Stride precomputation ───────────────────────────────────────────
    let input_row_stride = input_w * input_c;
    let filter_row_stride = filter_w * out_c;
    let filter_col_stride = out_c;
    let output_row_stride = out_w * out_c;

    // ── Accumulation loop ───────────────────────────────────────────────
    // TFLM depthwise loop order: batch → out_h → out_w → in_ch → dm → fh → fw
    // Output channel: oc = dm + in_ch * depth_multiplier
    for oh in 0..out_h {
        let input_base_h = oh * params.stride_height - pad_h;

        for ow in 0..out_w {
            let input_base_w = ow * params.stride_width - pad_w;

            for ic in 0..input_c {
                for dm in 0..depth_multiplier {
                    let oc = dm + ic * depth_multiplier;
                    let mut acc: i32 = bias[oc as usize];

                    for fh in 0..filter_h {
                        let in_h = input_base_h + fh * params.dilation_height_factor;
                        let row_in_bounds = in_h >= 0 && in_h < input_h;

                        for fw in 0..filter_w {
                            let in_w = input_base_w + fw * params.dilation_width_factor;

                            if row_in_bounds && in_w >= 0 && in_w < input_w {
                                let input_idx =
                                    (in_h * input_row_stride + in_w * input_c + ic) as usize;
                                let filter_idx =
                                    (fh * filter_row_stride + fw * filter_col_stride + oc)
                                        as usize;

                                let i_val = i32::from(input[input_idx]);
                                let w_val = i32::from(weights[filter_idx]);

                                acc += (i_val + params.input_offset) * w_val;
                            }
                            // else: zero-padding — skip (contribute 0 to accumulator)
                        }
                    }

                    // Per-channel requantize + output offset + clamp
                    let multiplier = multipliers[oc as usize];
                    let shift = shifts[oc as usize];
                    let scaled = multiply_by_quantized_multiplier(acc, multiplier, shift);
                    let with_offset = scaled + params.output_offset;

                    let clamped = if with_offset > params.quantized_activation_max {
                        params.quantized_activation_max
                    } else if with_offset < params.quantized_activation_min {
                        params.quantized_activation_min
                    } else {
                        with_offset
                    };

                    let out_idx =
                        (oh * output_row_stride + ow * out_c + oc) as usize;
                    output[out_idx] = saturating_cast(clamped);
                }
            }
        }
    }

    let _ = scratch; // unused by scalar reference path

    Ok(())
}
