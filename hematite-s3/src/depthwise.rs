// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! DepthwiseConv2D kernel — scalar fallback + TIE728 SIMD backend.
//!
//! # Bit-exact contract (Plan A4)
//!
//! | Leg | Contract | Runs on |
//! |-----|----------|---------|
//! | (a) | SIMD ≡ per-tensor TFLM golden bit-exact | Device (Phase 5) |
//! | (b) | Scalar ref ≡ per-channel TFLM golden bit-exact | **Host** (this test) |
//! | (c) | SIMD vs ref cross-check ≤1 LSB on requantize | Device (Phase 5) |
//!
//! On host (stable-aarch64-apple-darwin), only leg (b) executes. The SIMD path
//! (`#[cfg(target_arch = "xtensa")]`) is NEVER compiled on host.
//!
//! # Layouts
//!
//! * `input` — NHWC `[batch=1, H, W, Cin]`
//! * `weights` — channel-contiguous HWCN `[1, FH, FW, Cin * depth_multiplier]`
//! * `bias` — per-output-channel `[Cout]` where `Cout = Cin * depth_multiplier`
//! * `output` — NHWC `[batch=1, OH, OW, Cout]`
//!
//! # Loop order
//!
//! TFLM depthwise loop: oh → ow → ic → dm → fh → fw
//! Output channel: oc = dm + ic * depth_multiplier
//!
//! Depthwise is memory-bound (14–17× in ESP-DL). SIMD is used only for
//! activation and requantize in the epilogue; the inner (fh, fw) MAC loop
//! is scalar.

use hematite_core::op_params::DepthwiseConv2DParams;
use hematite_core::KernelError;
use hematite_int8::{multiply_by_quantized_multiplier, saturating_cast};

/// Compute the product of a `[i32; 4]` shape array.
#[inline(always)]
fn shape_product(shape: &[i32; 4]) -> usize {
    shape[0] as usize * shape[1] as usize * shape[2] as usize * shape[3] as usize
}

/// Depthwise 2D convolution — scalar kernel (host-compilable, bit-exact vs per-channel golden).
///
/// Mirrors `hematite-ref/src/depthwise_conv.rs` semantics exactly: bias-init
/// i32 accumulator, `(i_val + input_offset) * w_val` MAC over (fh, fw),
/// per-channel `multiply_by_quantized_multiplier`, output_offset, clamp,
/// saturating_cast.
///
/// Only batch=1 is supported. Batch>1 returns [`KernelError::Unsupported`].
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
    // TFLM depthwise loop order: batch → oh → ow → ic → dm → fh → fw
    // Output channel: oc = dm + ic * depth_multiplier
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

    let _ = scratch; // unused by scalar path

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// TIE728 SIMD backend — device-only (NEVER compiled on host)
// ─────────────────────────────────────────────────────────────────────────────

/// TIE728 SIMD backend for depthwise conv2d.
///
/// Depthwise is memory-bound. The SIMD path uses the vendored
/// `tie728_s8_conv2d_per_channel_result` macros from
/// `dl_tie728_s8.S` for the requantize epilogue only — the inner (fh, fw)
/// MAC loop remains scalar.
///
/// This module is cfg-gated behind `#[cfg(target_arch = "xtensa")]`.
/// ABI unverified — validate at T5.3 on device.
#[cfg(target_arch = "xtensa")]
mod depthwise_simd {
    /// TIE728 depthwise args struct.
    ///
    /// Field offsets derived from the shared result macros in
    /// `dl_tie728_s8.S`. The `tie728_s8_conv2d_per_channel_result` macro
    /// reads scale_factor_ptr from its argument (filter_channel_factor).
    /// The `tie728_s8_conv2d_per_channel_with_bias_result` macro additionally
    /// reads bias_ptr.
    ///
    /// ABI unverified — validate at T5.3 on device.
    #[repr(C)]
    #[allow(dead_code)]
    struct Tie728DepthwiseArgs {
        _pad0: [u8; 64],
        mac_shift: i32,                // +64
        bias: *const i32,              // +68
        _pad1: [u8; 32],               // +72..+103
        filter_channel_factor: *const i16, // +104
    }

    /// Include the vendored TIE728 shared macros.
    core::arch::global_asm!(
        include_str!("../src/asm/dl_tie728_s8.S"),
        include_str!("../src/asm/dl_tie728_s8_conv2d.S"),
    );
}
