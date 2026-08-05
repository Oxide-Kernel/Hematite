// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! 2D pooling — scalar fallback + TIE728 SIMD backend.
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
//! (`#[cfg(target_arch = "xtensa")]`) is NEVER compiled on host — it exists in
//! the tree for structural review and Phase 5 device verification.
//!
//! # Kernels
//!
//! * `average_pool_2d` — i32 accumulate over filter window, round-half-away-zero
//!   division, clamp to activation range. No input/output offset.
//! * `max_pool_2d` — pure i8 element-wise max over filter window, clamped to
//!   activation range. No input/output offset.
//! * `global_average_pool_2d` — average over full spatial extent [H, W].
//!
//! # Layouts
//!
//! All kernels use NHWC layout:
//! * `input` — `[batch=1, H, W, C]`
//! * `output` — `[batch=1, OH, OW, C]` (pool2d) or `[batch=1, 1, 1, C]` (global avg)

use hematite_core::op_params::PoolParams;
use hematite_core::KernelError;
use hematite_int8::saturating_cast;

/// Compute the product of a `[i32; 4]` shape array.
#[inline(always)]
fn shape_product(shape: &[i32; 4]) -> usize {
    shape[0] as usize * shape[1] as usize * shape[2] as usize * shape[3] as usize
}

/// Round-half-away-from-zero integer division: `numerator / denominator`
/// with halves rounded away from zero.
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

/// 2D average pooling — scalar kernel.
///
/// Mirrors `hematite-ref/src/pool.rs::average_pool_2d` arithmetic exactly.
/// Only batch=1 is supported.
pub fn average_pool_2d(
    input: &[i8],
    params: &PoolParams,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
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
                let mut acc: i32 = 0;
                let mut count: i32 = 0;

                for fy in fy_start..fy_end {
                    let in_y = in_y_origin + fy;
                    for fx in fx_start..fx_end {
                        let in_x = in_x_origin + fx;
                        let idx = (in_y * input_w + in_x) as usize * channels as usize + c;
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
                output[out_idx] = clamp_i8(
                    result,
                    params.quantized_activation_min,
                    params.quantized_activation_max,
                );
            }
        }
    }

    let _ = scratch;
    Ok(())
}

/// 2D max pooling — scalar kernel.
///
/// Mirrors `hematite-ref/src/pool.rs::max_pool_2d` arithmetic exactly.
/// Pure i8 element-wise max, clamped to activation range.
/// No input/output offset.
pub fn max_pool_2d(
    input: &[i8],
    params: &PoolParams,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
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
                        let idx = (in_y * input_w + in_x) as usize * channels as usize + c;
                        max_val = max_val.max(input[idx]);
                    }
                }

                let clamped = max_val.max(activation_min as i8).min(activation_max as i8);

                let out_idx = (oh * output_row_stride + ow * channels + oc) as usize;
                output[out_idx] = clamped;
            }
        }
    }

    let _ = scratch;
    Ok(())
}

/// Global average pooling — scalar kernel.
///
/// Averages over the full spatial extent `[H, W]` of the input, producing
/// a `[1, 1, 1, C]` output. Same round-half-away-from-zero division as
/// `average_pool_2d`. No input/output offset.
pub fn global_average_pool_2d(
    input: &[i8],
    params: &PoolParams,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    if params.input_shape[0] != 1 {
        return Err(KernelError::Unsupported);
    }

    let input_h = params.input_shape[1];
    let input_w = params.input_shape[2];
    let channels = params.input_shape[3];

    let out_c = params.output_shape[3];

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

        output[c] = clamp_i8(
            result,
            params.quantized_activation_min,
            params.quantized_activation_max,
        );
    }

    let _ = scratch;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// TIE728 SIMD backend — device-only (NEVER compiled on host)
// ─────────────────────────────────────────────────────────────────────────────

/// TIE728 SIMD backend for pooling ops.
///
/// This module is **entirely cfg-gated** behind `#[cfg(target_arch = "xtensa")]`
/// and is NEVER compiled on the host (stable-aarch64-apple-darwin). It exists
/// in the tree for structural review and Phase 5 device verification (T5.3).
///
/// ## Architecture
///
/// The SIMD path calls the vendored `dl_tie728_s8_max_pool2d_22c1` /
/// `dl_tie728_s8_avg_pool2d_22c1` entry points from vendored .S files
/// in `hematite-s3/src/asm/` via `global_asm!`.
///
/// Register convention (Xtensa XCC):
/// * a2 = output pointer (i8*)
/// * a3 = input pointer (i8*)
/// * a4 = args pointer (packed struct)
///
/// ## Vendored .S files
///
/// Cell `hematite-s3/src/asm/` contains:
/// * `dl_tie728_s8.S` — shared macros (pre-existing)
/// * `dl_tie728_s8_max_pool2d.S` — 4 entry points (22c1 aligned/unaligned,
///   hwc1 aligned/unaligned)
/// * `dl_tie728_s8_avg_pool2d.S` — 4 entry points (22c1 aligned/unaligned,
///   hwc1 aligned/unaligned)
///
/// All vendored from esp-dl @ 12c0616de145b704e1149c474b9a1e852e631d67 (MIT).
///
/// ## Args struct layouts (derived from vendored .S l32i offsets)
///
/// ### MaxPool 22c1 (aligned) — `dl_tie728_s8_max_pool2d_22c1`
/// * +4: input_channel (i32) — total channel count (= 1 in our fixtures)
/// * +16: input_y_offset (i32) — byte offset between input rows (input_w * input_c)
/// * +20: input_x_offset (i32) — byte offset between adjacent columns (input_c)
/// * +104: c_div_x_1 (i32) — (output elements / 16) - 1
///
/// ### AvgPool 22c1 (aligned) — `dl_tie728_s8_avg_pool2d_22c1`
/// * +4: input_channel (i32)
/// * +16: input_y_offset (i32)
/// * +20: input_x_offset (i32)
/// * +56: shift (i32) — requantize right-shift for `tie728_s8_vector_round_result`
/// * +64: avg_pool_area_inv (i8[16]) — precomputed reciprocal area packed as i8 vector
/// * +104: c_div_x_1 (i32)
///
/// ## GlobalAveragePool
///
/// No esp-dl `global_avg_pool2d` kernel exists. The scalar `global_average_pool_2d`
/// function above is the production path. No SIMD glue is provided.
///
/// ## A4 contract notes
///
/// * Leg (a): SIMD output must match a per-tensor TFLM golden (Phase 5 fixture).
/// * Leg (c): SIMD vs scalar ref cross-check tolerance ≤1 LSB on requantize.
#[cfg(target_arch = "xtensa")]
mod pool_simd {
    /// Include the vendored TIE728 shared macros and pool entry points.
    ///
    /// The shared `dl_tie728_s8.S` provides macros used by both pool files
    /// (`dl_tie728_s8_unaligned_store0`, `tie728_s8_vector_round_result`, etc.).
    core::arch::global_asm!(
        include_str!("../src/asm/dl_tie728_s8.S"),
        include_str!("../src/asm/dl_tie728_s8_max_pool2d.S"),
        include_str!("../src/asm/dl_tie728_s8_avg_pool2d.S"),
    );

    // ── Args structs — derived from vendored .S l32i offsets ──────────────

    /// Args for aligned max pool 2×2 — matches
    /// `dl_tie728_s8_max_pool2d_22c1`.
    ///
    /// ABI verified against vendored .S:
    /// * `l32i a5, a4, 16` → input_y_offset
    /// * `l32i a6, a4, 20` → input_x_offset
    /// * `l32i a11, a4, 104` → c_div_x_1
    ///
    /// Also read by hwc1/unaligned variants (not exposed yet):
    /// * `l32i a10, a4, 4` → input_channel
    /// * `l32i a8, a4, 48` → filter_height (hwc1)
    /// * `l32i a9, a4, 52` → filter_width (hwc1)
    /// * `l32i a12, a4, 60` → c_remainder (unaligned)
    ///
    /// ABI unverified on device — validate at T5.3.
    #[repr(C)]
    #[allow(dead_code)]
    struct Tie728MaxPoolArgs {
        _pad0: [u8; 4],                  // offset 0-3: unused
        input_channel: i32,              // offset 4
        _pad1: [u8; 8],                  // offset 8-15
        input_y_offset: i32,             // offset 16
        input_x_offset: i32,             // offset 20
        _pad2: [u8; 24],                 // offset 24-47
        filter_height: i32,              // offset 48
        filter_width: i32,               // offset 52
        _pad3: [u8; 4],                  // offset 56-59
        c_remainder: i32,                // offset 60
        _pad4: [u8; 40],                 // offset 64-103
        c_div_x_1: i32,                  // offset 104
    }

    impl Default for Tie728MaxPoolArgs {
        fn default() -> Self {
            Self {
                _pad0: [0u8; 4],
                input_channel: 0,
                _pad1: [0u8; 8],
                input_y_offset: 0,
                input_x_offset: 0,
                _pad2: [0u8; 24],
                filter_height: 0,
                filter_width: 0,
                _pad3: [0u8; 4],
                c_remainder: 0,
                _pad4: [0u8; 40],
                c_div_x_1: 0,
            }
        }
    }

    /// Args for aligned avg pool 2×2 — matches
    /// `dl_tie728_s8_avg_pool2d_22c1`.
    ///
    /// ABI verified against vendored .S:
    /// * `l32i a10, a4, 4` → input_channel
    /// * `l32i a5, a4, 16` → input_y_offset
    /// * `l32i a6, a4, 20` → input_x_offset
    /// * `l32i a13, a4, 56` → shift
    /// * `l32i a11, a4, 104` → c_div_x_1
    ///
    /// +64: avg_pool_area_inv (accessed via `EE.VLDBC.8 q0, a14` where
    /// `a14 = a4 + 64` — precomputed i8[16] reciprocal of filter area).
    ///
    /// ABI unverified on device — validate at T5.3.
    #[repr(C)]
    #[allow(dead_code)]
    struct Tie728AvgPoolArgs {
        _pad0: [u8; 4],                  // offset 0-3: unused
        input_channel: i32,              // offset 4
        _pad1: [u8; 8],                  // offset 8-15
        input_y_offset: i32,             // offset 16
        input_x_offset: i32,             // offset 20
        _pad2: [u8; 24],                 // offset 24-47
        filter_height: i32,              // offset 48
        filter_width: i32,               // offset 52
        shift: i32,                      // offset 56
        _pad3: [u8; 4],                  // offset 60-63
        avg_pool_area_inv: [i8; 16],     // offset 64: packed reciprocal vector
        _pad4: [u8; 24],                 // offset 80-103
        c_div_x_1: i32,                  // offset 104
    }

    impl Default for Tie728AvgPoolArgs {
        fn default() -> Self {
            Self {
                _pad0: [0u8; 4],
                input_channel: 0,
                _pad1: [0u8; 8],
                input_y_offset: 0,
                input_x_offset: 0,
                _pad2: [0u8; 24],
                filter_height: 0,
                filter_width: 0,
                shift: 0,
                _pad3: [0u8; 4],
                avg_pool_area_inv: [0i8; 16],
                _pad4: [0u8; 24],
                c_div_x_1: 0,
            }
        }
    }

    // ── SIMD kernel glue ──────────────────────────────────────────────────

    /// SIMD max pool 2×2 (aligned) — calls the vendored TIE728 entry point.
    ///
    /// Calls `dl_tie728_s8_max_pool2d_22c1`:
    /// * a2 = output (i8*)
    /// * a3 = input (i8*)
    /// * a4 = &Tie728MaxPoolArgs { input_channel, input_y_offset, input_x_offset, c_div_x_1 }
    ///
    /// # Safety
    ///
    /// This function is inherently unsafe: it calls into foreign assembly
    /// via the C ABI. ABI unverified — validate at T5.3 on device.
    ///
    /// # Preconditions (caller MUST guarantee)
    ///
    /// * Output elements must be a multiple of 16 (16-wide SIMD lanes).
    ///   The scalar fallback handles smaller/odd sizes.
    /// * All pointers must be 16-byte aligned for EE.VLD.128.IP / EE.VST.128.IP.
    /// * `input_y_offset` = input_w * input_c (row stride in bytes).
    /// * `input_x_offset` = input_c (column stride in bytes).
    #[allow(dead_code)]
    pub unsafe fn max_pool_2d_simd(
        output: *mut i8,
        input: *const i8,
        input_channel: i32,
        input_y_offset: i32,
        input_x_offset: i32,
        c_div_x_1: i32,
    ) {
        let args = Tie728MaxPoolArgs {
            input_channel,
            input_y_offset,
            input_x_offset,
            c_div_x_1,
            ..Default::default()
        };
        core::arch::asm!(
            "mov a2, {output}",
            "mov a3, {input}",
            "mov a4, {args}",
            "call8 dl_tie728_s8_max_pool2d_22c1",
            output = in(reg) output,
            input = in(reg) input,
            args = in(reg) &args,
            clobber_abi("C"),
        );
    }

    /// SIMD avg pool 2×2 (aligned) — calls the vendored TIE728 entry point.
    ///
    /// Calls `dl_tie728_s8_avg_pool2d_22c1`:
    /// * a2 = output (i8*)
    /// * a3 = input (i8*)
    /// * a4 = &Tie728AvgPoolArgs { input_channel, input_y_offset, input_x_offset, shift, avg_pool_area_inv, c_div_x_1 }
    ///
    /// # Safety
    ///
    /// Same safety contract as `max_pool_2d_simd`. ABI unverified.
    ///
    /// # Preconditions
    ///
    /// * Same alignment and size preconditions as max pool.
    /// * `shift`: requantize right-shift for the vector round-result macro.
    ///   Set to 0 for no requantize (plain round-half-away-zero divide).
    /// * `avg_pool_area_inv`: 16-byte packed i8 vector with reciprocal of
    ///   the filter area for each lane position. For a 2×2 filter (area=4):
    ///   each element = round(256.0 / 4.0) = 64 → [64; 16].
    #[allow(dead_code)]
    pub unsafe fn avg_pool_2d_simd(
        output: *mut i8,
        input: *const i8,
        input_channel: i32,
        input_y_offset: i32,
        input_x_offset: i32,
        shift: i32,
        avg_pool_area_inv: &[i8; 16],
        c_div_x_1: i32,
    ) {
        let mut area_inv = [0i8; 16];
        area_inv.copy_from_slice(avg_pool_area_inv);
        let args = Tie728AvgPoolArgs {
            input_channel,
            input_y_offset,
            input_x_offset,
            shift,
            avg_pool_area_inv: area_inv,
            c_div_x_1,
            ..Default::default()
        };
        core::arch::asm!(
            "mov a2, {output}",
            "mov a3, {input}",
            "mov a4, {args}",
            "call8 dl_tie728_s8_avg_pool2d_22c1",
            output = in(reg) output,
            input = in(reg) input,
            args = in(reg) &args,
            clobber_abi("C"),
        );
    }
}

#[cfg(target_arch = "xtensa")]
pub use pool_simd::avg_pool_2d_simd;
#[cfg(target_arch = "xtensa")]
pub use pool_simd::max_pool_2d_simd;
