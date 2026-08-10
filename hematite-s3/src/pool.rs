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
//! (`#[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]`) is NEVER
//! compiled on host and is additionally compiled out under `feature = "qemu"`
//! (the Espressif QEMU fork's TIE728 pool emulation hangs) — it exists in the
//! tree for structural review and Phase 5 device verification.
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

    // ── TIE728 SIMD dispatch (device-only; compiled out entirely on host) ──
    // `dl_tie728_s8_avg_pool2d_22c1` hardcodes a 2x2/stride-2 4-corner
    // access pattern and has no clamp field in its args struct, so it only
    // matches the scalar path when the activation range is the full int8
    // range (native saturating cast, matching the hardware's own int8
    // arithmetic). area_inv=64/shift=8 are the exact reciprocal constants
    // for a 2x2 (area=4) filter: round(2^8 / 4) = 64. Gated `not(feature =
    // "qemu")` — the QEMU TIE728 avg-pool emulation hangs.
    #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
    {
        if filter_h == 2
            && filter_w == 2
            && params.stride_height == 2
            && params.stride_width == 2
            && pad_h == 0
            && pad_w == 0
            && channels % 16 == 0
            && params.quantized_activation_min == i8::MIN as i32
            && params.quantized_activation_max == i8::MAX as i32
        {
            let in_ptr = input.as_ptr();
            let out_ptr = output.as_mut_ptr();
            if (in_ptr as usize) % 16 == 0 && (out_ptr as usize) % 16 == 0 {
                // `dl_tie728_s8_avg_pool2d_22c1` computes ONE output pixel per
                // call (looping over `channels/16` channel-groups); the driver
                // loops over the output image, stepping the input by
                // `stride * channels` per pixel (stride == 2 here, pad == 0).
                // `c_div_x_1` is the channel-group count minus one, NOT the
                // total-output count — passing `total_out/16 - 1` makes the
                // kernel walk a stride-1 window across the whole buffer.
                let mut ctx = pool_simd::AvgPoolSimdCtx::new(
                    out_ptr,
                    in_ptr,
                    (channels, input_w * channels, channels, channels / 16 - 1),
                );
                unsafe {
                    let in_row_step = (2 * input_w) * channels;
                    let out_row_step = out_w * channels;
                    for oh in 0..out_h {
                        let mut in_px = in_ptr.add((oh * in_row_step) as usize);
                        for ow in 0..out_w {
                            ctx.input = in_px;
                            pool_simd::avg_pool_2d_simd_ctx(&mut ctx);
                            ctx.output = ctx.output.add(channels as usize);
                            in_px = in_px.add((2 * channels) as usize);
                        }
                        ctx.output = out_ptr.add(((oh + 1) * out_row_step) as usize);
                    }
                }
                let _ = scratch;
                return Ok(());
            }
        }
    }

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

    // ── TIE728 SIMD dispatch (device-only; compiled out entirely on host) ──
    // Same eligibility contract as `average_pool_2d`'s dispatch above —
    // `dl_tie728_s8_max_pool2d_22c1` hardcodes 2x2/stride-2 and has no
    // clamp field, so full-range activation bounds are required. Gated
    // `not(feature = "qemu")` — the QEMU TIE728 max-pool emulation hangs.
    #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
    {
        if filter_h == 2
            && filter_w == 2
            && params.stride_height == 2
            && params.stride_width == 2
            && pad_h == 0
            && pad_w == 0
            && channels % 16 == 0
            && params.quantized_activation_min == i8::MIN as i32
            && params.quantized_activation_max == i8::MAX as i32
        {
            let in_ptr = input.as_ptr();
            let out_ptr = output.as_mut_ptr();
            if (in_ptr as usize) % 16 == 0 && (out_ptr as usize) % 16 == 0 {
                // `dl_tie728_s8_max_pool2d_22c1` computes ONE output pixel per
                // call (looping over `channels/16` channel-groups); the driver
                // loops over the output image, stepping the input by
                // `stride * channels` per pixel (stride == 2 here, pad == 0).
                // `c_div_x_1` is the channel-group count minus one, NOT the
                // total-output count — passing `total_out/16 - 1` makes the
                // kernel walk a stride-1 window across the whole buffer.
                let mut ctx = pool_simd::MaxPoolSimdCtx::new(
                    out_ptr,
                    in_ptr,
                    (channels, input_w * channels, channels, channels / 16 - 1),
                );
                unsafe {
                    let in_row_step = (2 * input_w) * channels;
                    let out_row_step = out_w * channels;
                    for oh in 0..out_h {
                        let mut in_px = in_ptr.add((oh * in_row_step) as usize);
                        ctx.output = out_ptr.add((oh * out_row_step) as usize);
                        for ow in 0..out_w {
                            ctx.input = in_px;
                            pool_simd::max_pool_2d_simd_ctx(&mut ctx);
                            ctx.output = ctx.output.add(channels as usize);
                            in_px = in_px.add((2 * channels) as usize);
                        }
                    }
                }
                let _ = scratch;
                return Ok(());
            }
        }
    }

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
/// (the dispatch into it is additionally gated `not(feature = "qemu")`, so the
/// broken QEMU TIE728 emulation is never reached) and is NEVER compiled on the
/// host (stable-aarch64-apple-darwin). It exists in the tree for structural
/// review and Phase 5 device verification (T5.3).
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
    // Include the vendored TIE728 shared macros and pool entry points.
    //
    // The shared `dl_tie728_s8.S` provides macros used by both pool files
    // (`dl_tie728_s8_unaligned_store0`, `tie728_s8_vector_round_result`, etc.).
    core::arch::global_asm!(
        include_str!("../src/asm/dl_tie728_s8.S"),
        include_str!("../src/asm/dl_tie728_s8_max_pool2d.S"),
        include_str!("../src/asm/dl_tie728_s8_avg_pool2d.S"),
    );

    // `call8 <label>` uses an 18-bit signed PC-relative encoding (±~512KB).
    // Once these entry points are actually referenced (this task's whole
    // point), the final binary's code layout can push the call site farther
    // from the vendored .S symbol than that encoding allows ("dangerous
    // relocation: call8: call target out of range"), which LLVM's Xtensa
    // backend — unlike GCC's `-mlongcalls` — does not auto-relax. The fix is
    // a register-indirect `callx8` through an ordinary Rust function-pointer
    // load, which has no range limit; the vendored .S entry point itself is
    // untouched.
    extern "C" {
        fn dl_tie728_s8_avg_pool2d_22c1();
        fn dl_tie728_s8_max_pool2d_22c1();
    }

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
        // Write only the asm-read fields (+4/+16/+20/+48/+52/+60/+104);
        // _pad bytes are never read by the asm, so leave uninitialized.
        let mut args = core::mem::MaybeUninit::<Tie728MaxPoolArgs>::uninit();
        let p = args.as_mut_ptr();
        p.cast::<u8>().add(4).cast::<i32>().write(input_channel);
        p.cast::<u8>().add(16).cast::<i32>().write(input_y_offset);
        p.cast::<u8>().add(20).cast::<i32>().write(input_x_offset);
        p.cast::<u8>().add(48).cast::<i32>().write(2); // filter_height
        p.cast::<u8>().add(52).cast::<i32>().write(2); // filter_width
        p.cast::<u8>().add(60).cast::<i32>().write(0); // c_remainder
        p.cast::<u8>().add(104).cast::<i32>().write(c_div_x_1);
        let args = unsafe { args.assume_init_ref() };
        let target = dl_tie728_s8_max_pool2d_22c1 as unsafe extern "C" fn() as usize;
        core::arch::asm!(
            "mov a10, {output}",
            "mov a11, {input}",
            "mov a12, {args}",
            "callx8 {target}",
            output = in(reg) output,
            input = in(reg) input,
            args = in(reg) args,
            target = in(reg) target,
            out("a13") _,
            out("a14") _,
            out("a15") _,
            clobber_abi("C"),
        );
    }

    /// Max-pool SIMD via a single `&mut` context — dodges the Xtensa-LLVM
    /// multi-arg-call miscompile (a repeated 6-arg `max_pool_2d_simd` call is
    /// scrambled by the backend; a 1-arg call is not — same rationale as
    /// `AvgPoolSimdCtx`).
    ///
    /// The TIE728 args struct is built ONCE here (in `new`) and reused for
    /// every pixel of the op — `dl_tie728_s8_max_pool2d_22c1` reads all its
    /// args at entry and only advances the output/input pointers, so the
    /// per-pixel cost is one `callx8` plus pointer updates. (Before the pool
    /// public-API-gap fix the args struct was rebuilt on the stack for every
    /// pixel — ~124 wrapper cycles/pixel, the entire 33240-vs-1396 gap.)
    #[allow(dead_code)]
    pub(crate) struct MaxPoolSimdCtx {
        pub(crate) args: Tie728MaxPoolArgs,
        pub(crate) output: *mut i8,
        pub(crate) input: *const i8,
    }

    impl MaxPoolSimdCtx {
        /// Build a max-pool context with the args struct materialized once.
        ///
        /// `cfg` is the `simd_eligible_pool` tuple
        /// `(input_channel, input_y_offset, input_x_offset, c_div_x_1)`.
        pub(crate) fn new(
            output: *mut i8,
            input: *const i8,
            cfg: (i32, i32, i32, i32),
        ) -> Self {
            // Plain struct literal — the MaybeUninit pointer-cast build is
            // miscompiled by the Xtensa LLVM backend, and the struct-literal
            // form is the pattern proven on device (pool.rs precedent).
            let args = Tie728MaxPoolArgs {
                input_channel: cfg.0,
                input_y_offset: cfg.1,
                input_x_offset: cfg.2,
                filter_height: 2,
                filter_width: 2,
                c_remainder: 0,
                c_div_x_1: cfg.3,
                ..Tie728MaxPoolArgs::default()
            };
            Self {
                args,
                output,
                input,
            }
        }
    }

    /// Run ONE 2×2/stride-2 max-pool pixel through the TIE728 `*_22c1` entry.
    ///
    /// Same contract as the original `max_pool_2d_simd_ctx` (single `&mut`
    /// ctx arg — dodges the Xtensa-LLVM multi-arg-call miscompile): the args
    /// struct is NOT rebuilt here — it was materialized once in
    /// [`MaxPoolSimdCtx::new`] and is read through `&ctx.args`, so the
    /// per-pixel cost is four register loads + `callx8`.
    ///
    /// The kernel advances its own a2/a3 by 16 B (invisible to the
    /// `in("a10")`/`in("a11")` operands); the caller re-applies the advance
    /// (input by `2 * channels` for stride 2, output by `channels`).
    ///
    /// # Safety
    ///
    /// `ctx.output`/`ctx.input` 16-byte aligned, buffers sized per
    /// `simd_eligible_pool` eligibility (2×2/stride-2/pad-0, channels % 16).
    #[inline(never)]
    #[allow(dead_code)]
    pub unsafe fn max_pool_2d_simd_ctx(ctx: &mut MaxPoolSimdCtx) {
        let target = dl_tie728_s8_max_pool2d_22c1 as unsafe extern "C" fn() as usize;
        core::arch::asm!(
            "callx8 a13",
            in("a10") ctx.output,
            in("a11") ctx.input,
            in("a12") &ctx.args,
            in("a13") target,
            out("a14") _,
            out("a15") _,
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
        // NOTE: an 8-arg call like this is the Xtensa-LLVM multi-arg-call
        // miscompile class (args spill to stack and get scrambled, cf. the
        // `accx` ctx refactors). Callers must
        // route through `avg_pool_2d_simd_loop` (single `&mut` ctx arg)
        // instead.
        let mut area_inv = [0i8; 16];
        area_inv.copy_from_slice(avg_pool_area_inv);
        // Write only the asm-read fields (+4/+16/+20/+56/+64/+104).
        let mut args = core::mem::MaybeUninit::<Tie728AvgPoolArgs>::uninit();
        let p = args.as_mut_ptr();
        p.cast::<u8>().add(4).cast::<i32>().write(input_channel);
        p.cast::<u8>().add(16).cast::<i32>().write(input_y_offset);
        p.cast::<u8>().add(20).cast::<i32>().write(input_x_offset);
        p.cast::<u8>().add(48).cast::<i32>().write(2); // filter_height
        p.cast::<u8>().add(52).cast::<i32>().write(2); // filter_width
        p.cast::<u8>().add(56).cast::<i32>().write(shift);
        p.cast::<u8>().add(64).cast::<i8>().copy_from_nonoverlapping(
            area_inv.as_ptr(),
            16,
        );
        p.cast::<u8>().add(104).cast::<i32>().write(c_div_x_1);
        let args = unsafe { args.assume_init_ref() };
        let target = dl_tie728_s8_avg_pool2d_22c1 as unsafe extern "C" fn() as usize;
        core::arch::asm!(
            "mov a10, {output}",
            "mov a11, {input}",
            "mov a12, {args}",
            "callx8 {target}",
            output = in(reg) output,
            input = in(reg) input,
            args = in(reg) args,
            target = in(reg) target,
            out("a13") _,
            out("a14") _,
            out("a15") _,
            clobber_abi("C"),
        );
    }

    /// Avg-pool SIMD via a single `&mut` context — dodges the Xtensa-LLVM
    /// multi-arg-call miscompile (an 8-arg `avg_pool_2d_simd` call is
    /// scrambled by the backend; a 1-arg call is not).
    ///
    /// Same as `MaxPoolSimdCtx`: the TIE728 args struct (incl. the 16-byte
    /// `avg_pool_area_inv` vector) is built ONCE in `new`, not per pixel.
    #[allow(dead_code)]
    pub(crate) struct AvgPoolSimdCtx {
        pub(crate) args: Tie728AvgPoolArgs,
        pub(crate) output: *mut i8,
        pub(crate) input: *const i8,
    }

    impl AvgPoolSimdCtx {
        /// Build an avg-pool context with the args struct materialized once.
        ///
        /// `cfg` is the `simd_eligible_pool` tuple
        /// `(input_channel, input_y_offset, input_x_offset, c_div_x_1)`.
        pub(crate) fn new(
            output: *mut i8,
            input: *const i8,
            cfg: (i32, i32, i32, i32),
        ) -> Self {
            // Plain struct literal (not MaybeUninit pointer-cast writes): the
            // latter is miscompiled by the Xtensa LLVM backend when it holds a
            // 16-byte array copy (proven on device).
            let args = Tie728AvgPoolArgs {
                input_channel: cfg.0,
                input_y_offset: cfg.1,
                input_x_offset: cfg.2,
                filter_height: 2,
                filter_width: 2,
                shift: 8,
                avg_pool_area_inv: [64i8; 16],
                c_div_x_1: cfg.3,
                ..Tie728AvgPoolArgs::default()
            };
            Self {
                args,
                output,
                input,
            }
        }
    }

    /// Run ONE 2×2/stride-2 avg-pool pixel through the TIE728 `*_22c1` entry.
    ///
    /// Same contract as the original `avg_pool_2d_simd_ctx` (single `&mut`
    /// ctx arg — dodges the Xtensa-LLVM multi-arg-call miscompile; the args
    /// struct was materialized once in [`AvgPoolSimdCtx::new`], not rebuilt
    /// here, so the per-pixel cost is four register loads + `callx8`).
    /// The kernel advances its own a2/a3 by 16 B (invisible to the `in()`
    /// operands); the caller re-applies the advance (input by
    /// `2 * channels` for stride 2, output by `channels`).
    ///
    /// # Safety
    ///
    /// `ctx.output`/`ctx.input` 16-byte aligned, buffers sized per
    /// `simd_eligible_pool` eligibility (2×2/stride-2/pad-0, channels % 16).
    #[inline(never)]
    #[allow(dead_code)]
    pub unsafe fn avg_pool_2d_simd_ctx(ctx: &mut AvgPoolSimdCtx) {
        let target = dl_tie728_s8_avg_pool2d_22c1 as unsafe extern "C" fn() as usize;
        core::arch::asm!(
            "callx8 a13",
            in("a10") ctx.output,
            in("a11") ctx.input,
            in("a12") &ctx.args,
            in("a13") target,
            out("a14") _,
            out("a15") _,
            clobber_abi("C"),
        );
    }
}

#[cfg(target_arch = "xtensa")]
pub use pool_simd::avg_pool_2d_simd;
#[cfg(target_arch = "xtensa")]
pub use pool_simd::max_pool_2d_simd;

// ── Prepared-pool fast path ──────────────────────────────────────────────

/// Shared SIMD-eligibility gate for 2×2/stride-2 pooling (both avg and max).
///
/// Host-compilable: returns `Some(cfg)` when the TIE728 `*_22c1` entry points
/// can run, so a handle can be built once and reused across calls without
/// re-running the gate. `cfg` is `(input_channel, input_y_offset, input_x_offset,
/// c_div_x_1)`.
pub(crate) fn simd_eligible_pool(
    params: &PoolParams,
) -> Option<(i32, i32, i32, i32)> {
    let input_h = params.input_shape[1];
    let input_w = params.input_shape[2];
    let channels = params.input_shape[3];
    let filter_h = params.filter_height;
    let filter_w = params.filter_width;
    let out_h = params.output_shape[1];
    let out_w = params.output_shape[2];

    let pad_h = ((out_h - 1) * params.stride_height + filter_h - input_h) / 2;
    let pad_w = ((out_w - 1) * params.stride_width + filter_w - input_w) / 2;

    if filter_h == 2
        && filter_w == 2
        && params.stride_height == 2
        && params.stride_width == 2
        && pad_h == 0
        && pad_w == 0
        && channels % 16 == 0
        && params.quantized_activation_min == i8::MIN as i32
        && params.quantized_activation_max == i8::MAX as i32
    {
        // `c_div_x_1` is the channel-group count minus one (`channels/16 - 1`),
        // NOT `total_out/16 - 1`: the 22c1 kernel computes ONE output pixel per
        // call, looping over `channels/16` channel-groups. The caller loops the
        // output image itself. (Passing `total_out/16 - 1` makes the kernel
        // walk a stride-1 window across the whole buffer — the pre-Phase-12 bug
        // that made pool SIMD output diverge from the scalar reference.)
        Some((channels, input_w * channels, channels, channels / 16 - 1))
    } else {
        None
    }
}

/// Prepared 2×2 max pool — runs the SIMD gate once at construction.
pub struct PreparedMaxPool {
    simd: Option<(i32, i32, i32, i32)>,
    params: &'static PoolParams,
}

impl PreparedMaxPool {
    /// Run the SIMD gate once; subsequent `run` calls skip it.
    pub fn new(params: &'static PoolParams) -> Result<Self, KernelError> {
        let simd = simd_eligible_pool(params)
            .filter(|_| cfg!(all(target_arch = "xtensa", not(feature = "qemu"))));
        Ok(Self { simd, params })
    }

    /// Whether the TIE728 SIMD path is active for these params.
    pub fn is_simd(&self) -> bool {
        self.simd.is_some()
    }

    /// Run max pool on `input` → `output`.
    pub fn run(
        &self,
        input: &[i8],
        output: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
        {
            if let Some(cfg) = self.simd {
                let in_ptr = input.as_ptr();
                let out_ptr = output.as_mut_ptr();
                if (in_ptr as usize) % 16 == 0 && (out_ptr as usize) % 16 == 0 {
                    // `dl_tie728_s8_max_pool2d_22c1` computes ONE output pixel
                    // per call; the shared driver loops over the output image
                    // (stride 2, pad 0 — guaranteed by `simd_eligible_pool`).
                    let channels = cfg.0 as usize;
                    let in_w = self.params.input_shape[2] as usize;
                    let out_w = self.params.output_shape[2] as usize;
                    let out_h = self.params.output_shape[1] as usize;
                    let mut ctx = pool_simd::MaxPoolSimdCtx::new(out_ptr, in_ptr, cfg);
                    unsafe {
                        let in_row_step = (2 * in_w) * channels;
                        let out_row_step = out_w * channels;
                        for oh in 0..out_h {
                            let mut in_px = in_ptr.add(oh * in_row_step);
                            ctx.output = out_ptr.add(oh * out_row_step);
                            for _ow in 0..out_w {
                                ctx.input = in_px;
                                pool_simd::max_pool_2d_simd_ctx(&mut ctx);
                                ctx.output = ctx.output.add(channels as usize);
                                in_px = in_px.add((2 * channels) as usize);
                            }
                        }
                    }
                    let _ = scratch;
                    return Ok(());
                }
            }
        }
        max_pool_2d(input, self.params, output, scratch)
    }
}

/// Prepared 2×2 average pool — runs the SIMD gate once at construction.
pub struct PreparedAvgPool {
    simd: Option<(i32, i32, i32, i32)>,
    params: &'static PoolParams,
}

impl PreparedAvgPool {
    /// Run the SIMD gate once; subsequent `run` calls skip it.
    pub fn new(params: &'static PoolParams) -> Result<Self, KernelError> {
        let simd = simd_eligible_pool(params)
            .filter(|_| cfg!(all(target_arch = "xtensa", not(feature = "qemu"))));
        Ok(Self { simd, params })
    }

    /// Whether the TIE728 SIMD path is active for these params.
    pub fn is_simd(&self) -> bool {
        self.simd.is_some()
    }

    /// Run average pool on `input` → `output`.
    pub fn run(
        &self,
        input: &[i8],
        output: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
        {
            if let Some(cfg) = self.simd {
                let in_ptr = input.as_ptr();
                let out_ptr = output.as_mut_ptr();
                if (in_ptr as usize) % 16 == 0 && (out_ptr as usize) % 16 == 0 {
                    // `dl_tie728_s8_avg_pool2d_22c1` computes ONE output pixel
                    // per call; the shared driver loops over the output image
                    // (stride 2, pad 0 — guaranteed by `simd_eligible_pool`).
                    let channels = cfg.0 as usize;
                    let in_w = self.params.input_shape[2] as usize;
                    let out_w = self.params.output_shape[2] as usize;
                    let out_h = self.params.output_shape[1] as usize;
                    let mut ctx = pool_simd::AvgPoolSimdCtx::new(out_ptr, in_ptr, cfg);
                    unsafe {
                        let in_row_step = (2 * in_w) * channels;
                        let out_row_step = out_w * channels;
                        for oh in 0..out_h {
                            let mut in_px = in_ptr.add(oh * in_row_step);
                            ctx.output = out_ptr.add(oh * out_row_step);
                            for _ow in 0..out_w {
                                ctx.input = in_px;
                                pool_simd::avg_pool_2d_simd_ctx(&mut ctx);
                                ctx.output = ctx.output.add(channels as usize);
                                in_px = in_px.add((2 * channels) as usize);
                            }
                        }
                    }
                    let _ = scratch;
                    return Ok(());
                }
            }
        }
        average_pool_2d(input, self.params, output, scratch)
    }
}
