// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Reduction ops — scalar fallback + TIE728 SIMD backend.
//!
//! # Bit-exact contract (Plan A4)
//!
//! | Leg | Contract | Runs on |
//! |-----|----------|---------|
//! | (a) | SIMD ≡ per-tensor TFLM golden bit-exact | Device (Phase 5) |
//! | (b) | Scalar ref ≡ per-channel TFLM golden bit-exact | **Host** (this test) |
//! | (c) | SIMD vs ref cross-check ≤1 LSB | Device (Phase 5) |
//!
//! # Ops implemented
//!
//! * [`mean`] — i32 accumulate over reduction axes, divide by count
//!   (round-half-away-from-zero), then per-tensor requantize.

use hematite_core::op_params::ReduceParams;
use hematite_core::KernelError;
use hematite_int8::{multiply_by_quantized_multiplier, saturating_cast};

/// Compute the product of a `[i32; 4]` shape array.
#[inline(always)]
fn shape_product(shape: &[i32; 4]) -> usize {
    shape[0] as usize * shape[1] as usize * shape[2] as usize * shape[3] as usize
}

/// Round-half-away-from-zero integer division.
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

/// Reduce-mean — scalar kernel.
///
/// Mirrors `hematite-ref/src/reductions.rs::mean` arithmetic exactly.
///
/// # Algorithm
///
/// 1. i32 accumulate over the reduction axes.
/// 2. Divide by count (round-half-away-from-zero).
/// 3. Requantize via `multiply_by_quantized_multiplier` + output_offset + clamp.
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

    for oh in 0..out_h {
        for ow in 0..out_w {
            for oc in 0..out_c {
                let mut acc: i32 = 0;

                let h_start = if reduce_mask[1] {
                    0
                } else {
                    oh * (in_h / out_h.max(1))
                };
                let h_end = if reduce_mask[1] {
                    in_h
                } else {
                    h_start + 1
                };
                let w_start = if reduce_mask[2] {
                    0
                } else {
                    ow * (in_w / out_w.max(1))
                };
                let w_end = if reduce_mask[2] {
                    in_w
                } else {
                    w_start + 1
                };
                let c_start = if reduce_mask[3] {
                    0
                } else {
                    oc * (in_c / out_c.max(1))
                };
                let c_end = if reduce_mask[3] {
                    in_c
                } else {
                    c_start + 1
                };

                for ih in h_start..h_end {
                    for iw in w_start..w_end {
                        for ic in c_start..c_end {
                            let idx = ih * in_stride_h + iw * in_stride_w + ic * in_stride_c;
                            acc += i32::from(input[idx]);
                        }
                    }
                }

                let averaged = if total_count == 0 {
                    0
                } else {
                    round_half_away_zero(acc, total_count)
                };
                let scaled = multiply_by_quantized_multiplier(averaged, mult, shift);
                let val = (scaled + out_off).max(act_min).min(act_max);
                let out_idx = oh * (out_w * out_c) + ow * out_c + oc;
                output[out_idx] = clamp_i8(val, act_min, act_max);
            }
        }
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// TIE728 SIMD backend — device-only (NEVER compiled on host)
// ─────────────────────────────────────────────────────────────────────────────

/// TIE728 SIMD reduction module.
///
/// **Entirely cfg-gated** — NEVER compiled on host.
///
/// ## Plan-per-op SIMD instructions
///
/// * Mean: sum → requantize, partial SIMD (scalar accumulator with
///   SIMD load/store via `ee.vld.128.ip` / `ee.vst.128.ip`).
#[cfg(target_arch = "xtensa")]
mod reduction_simd {
    /// TIE728 reduction args struct.
    ///
    /// ## ABI (unverified — T5.3 device verification required)
    #[allow(dead_code)]
    #[repr(C)]
    struct Tie728ReduceArgs {
        output: *mut i8,
        input: *const i8,
        _pad: [u8; 64], // reserved
    }

    /// Mean SIMD — partial SIMD load/store, scalar accumulator.
    ///
    /// # Safety
    ///
    /// ABI-unverified. Inline global_asm! with ee.vld.128.ip +
    /// ee.vst.128.ip.
    #[allow(dead_code)]
    unsafe fn mean_simd(_output: *mut i8, _input: *const i8, _args: &Tie728ReduceArgs) {
        core::arch::asm!(
            // Placeholder — real SIMD loop:
            // 1. ee.vld.128.ip to load 16 int8 values
            // 2. Scalar i32 accumulate (no SIMD add — overflow risk
            //    with int8 × 16 = small window)
            // 3. Scalar divide + requantize
            // 4. ee.vst.128.ip to store result
            "nop",
            clobber_abi("C"),
        );
    }
}

#[cfg(target_arch = "xtensa")]
pub use reduction_simd::mean_simd;
