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
//! # Generic pool SIMD (T3.1)
//!
//! The SIMD path (device-only, `#[cfg(all(target_arch = "xtensa", not(feature
//! = "qemu")))]`) is widened from the historical 2×2/stride-2/pad-0-only gate
//! to **arbitrary filter size, stride, and activation range**: `simd_eligible_pool`
//! accepts any filter ≥ 2 taps with stride ≥ 1 and no padding / no partial
//! windows (see below), and the clamp is applied as a Rust post-pass over the
//! vendored kernel's output (the `dl_tie728_s8_{max,avg}_pool2d_hwc1` entry
//! points have no clamp field).
//!
//! ## The s3/C-SIMD semantics oracle (MUST NOT equal `hematite-ref`)
//!
//! The correctness oracle for the generic kernels is the **established
//! s3/C-SIMD pool semantics that ship on silicon**, NOT `hematite-ref`'s
//! `round_half_away_zero`:
//!
//! * **max** — element-wise max over the window. For pad-0/full-range this
//!   equals the scalar reference bit-exact (device-validated: the 4×4×16
//!   `max_pool_2x2` simd-validation check PASSes). Padding (pad > 0) fills
//!   with `i8::MIN` so it never wins the max — identical to the scalar
//!   fallback's valid-tap-only loop.
//! * **avg** — the vendored fixed-point reciprocal: `acc = Σ window (in ·
//!   area_inv)` over the FULL filter area (padding contributes 0), then the
//!   hardware round `((acc >> (shift-1)) + 1) >> 1` (the
//!   `tie728_s8_vector_round_result` macro — see `dl_tie728_s8.S:175`;
//!   arithmetic shifts, no intermediate saturation). The
//!   2×2 kernel uses `area_inv = 64`, `shift = 8` (= `round(2^8/4)`); the
//!   generic path derives the same pair from the filter area. This is NOT
//!   `round_half_away_zero(acc, area)`: the shift-based rounding diverges by
//!   ±1 LSB on negative half-even window sums (device-validated: the 4×4×16
//!   `avg_pool_2x2` check is the documented ±1 known-delta, `fnv 0xd0d19a11`),
//!   and the reciprocal is exact only for power-of-two areas. The divergence
//!   is DOCUMENTED (in the pool SIMD evidence), never
//!   "fixed".
//!
//! ## Why the device path is no-padding-only
//!
//! `KernelBackend::average_pool_2d`/`max_pool_2d` carry **no scratch
//! parameter** (backend.rs forwards `&mut []`), and the codegen path emits
//! those trait methods — so the pool SIMD dispatch can never stage a
//! zero-padded input copy on device (unlike the depthwise/conv dispatches,
//! which receive scratch). The vendored `*_hwc1` kernels read the full
//! fh×fw window at the caller-positioned origin, which is out-of-bounds for
//! boundary windows — so the gate requires `pad_total = (out-1)·stride +
//! filter - in ≤ 0` (every window fully in-bounds; VALID shapes, and SAME
//! shapes whose SAME output equals the VALID output, e.g. 2×2/stride-2).
//! This also keeps the SIMD output equal to the scalar fallback on the SAME
//! backend: the avg fixed-point divides by the FULL area, so a partial
//! window (asymmetric SAME, pad_total > 0) would diverge from the scalar's
//! valid-tap count. **Pad > 0 / partial-window shapes keep the scalar
//! fallback (bit-exact vs ref)**. The host-compilable model (the
//! `generic_*_simd` test module) covers the full pad {0,1,SAME} matrix on
//! the host, and the divergence/eligibility split is recorded in the
//! evidence file.
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
    // The legacy 2×2/stride-2/pad-0/full-range path below is BYTE-IDENTICAL
    // to the pre-T3.1 dispatch (the `dl_tie728_s8_avg_pool2d_22c1` kernel and
    // its driver loop are unchanged — same fixed-point `area_inv`/`shift`,
    // now sourced from the shared gate). Shapes outside that family route to
    // the generic `*_hwc1` driver (any filter/stride, pad 0, any clamp — the
    // clamp applied as a Rust post-pass). Gated `not(feature = "qemu")` — the
    // QEMU TIE728 avg-pool emulation hangs.
    #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
    {
        if let Some(cfg) = simd_eligible_pool(params) {
            if filter_h == 2
                && filter_w == 2
                && params.stride_height == 2
                && params.stride_width == 2
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
                    let mut ctx = pool_simd::AvgPoolSimdCtx::new(out_ptr, in_ptr, &cfg);
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
            // Generic hwc1 path — any filter/stride, pad 0, any clamp.
            if avg_pool_2d_generic_simd(input, params, output, &cfg)? {
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
    // Legacy 2×2/stride-2/pad-0/full-range via the unchanged
    // `dl_tie728_s8_max_pool2d_22c1` driver; every other eligible shape
    // (any filter/stride, pad 0, any clamp) routes to the generic `*_hwc1`
    // driver. Gated `not(feature = "qemu")` — the QEMU TIE728 max-pool
    // emulation hangs.
    #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
    {
        if let Some(cfg) = simd_eligible_pool(params) {
            if filter_h == 2
                && filter_w == 2
                && params.stride_height == 2
                && params.stride_width == 2
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
                    let mut ctx = pool_simd::MaxPoolSimdCtx::new(out_ptr, in_ptr, &cfg);
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
            // Generic hwc1 path — any filter/stride, pad 0, any clamp.
            if max_pool_2d_generic_simd(input, params, output, &cfg)? {
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
        fn dl_tie728_s8_avg_pool2d_hwc1();
        fn dl_tie728_s8_max_pool2d_hwc1();
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
        /// `cfg` is the [`PoolSimdCfg`] from [`simd_eligible_pool`]. The
        /// `*_22c1` (legacy 2×2) and `*_hwc1` (generic filter) entry points
        /// share the args layout; the ctx is reused across both drivers.
        pub(crate) fn new(
            output: *mut i8,
            input: *const i8,
            cfg: &super::PoolSimdCfg,
        ) -> Self {
            // Plain struct literal — the MaybeUninit pointer-cast build is
            // miscompiled by the Xtensa LLVM backend, and the struct-literal
            // form is the pattern proven on device (pool.rs precedent).
            let args = Tie728MaxPoolArgs {
                input_channel: cfg.channels,
                input_y_offset: cfg.input_y_offset,
                input_x_offset: cfg.input_x_offset,
                filter_height: cfg.filter_h,
                filter_width: cfg.filter_w,
                c_remainder: cfg.c_remainder,
                c_div_x_1: cfg.c_div_x_1,
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
    /// (input by `stride * channels`, output by `channels`).
    ///
    /// # Safety
    ///
    /// `ctx.output`/`ctx.input` 16-byte aligned, buffers sized per
    /// `simd_eligible_pool` eligibility (pad 0, channels % 16).
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

    /// Run ONE max-pool pixel through the generic TIE728 `*_hwc1` entry —
    /// the arbitrary-filter kernel (reads `filter_height`/`filter_width`
    /// from the args at +48/+52 and walks the window with the
    /// `input_y_offset`/`input_x_offset` strides). Same single-`&mut`-ctx
    /// contract and register pinning as `max_pool_2d_simd_ctx`.
    ///
    /// # Safety
    ///
    /// `ctx.output`/`ctx.input` 16-byte aligned, buffers sized per
    /// `simd_eligible_pool` eligibility (pad 0, channels % 16); the window
    /// anchored at `ctx.input` must be fully in-bounds (the caller positions
    /// the origin).
    #[inline(never)]
    #[allow(dead_code)]
    pub unsafe fn max_pool_2d_simd_ctx_hwc1(ctx: &mut MaxPoolSimdCtx) {
        let target = dl_tie728_s8_max_pool2d_hwc1 as unsafe extern "C" fn() as usize;
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
        /// `cfg` is the [`PoolSimdCfg`] from [`simd_eligible_pool`] — the
        /// fixed-point `shift`/`area_inv` (the legacy 2×2 path gets the same
        /// `(8, [64;16])` pair it hardcoded) plus the filter dims.
        pub(crate) fn new(
            output: *mut i8,
            input: *const i8,
            cfg: &super::PoolSimdCfg,
        ) -> Self {
            // Plain struct literal (not MaybeUninit pointer-cast writes): the
            // latter is miscompiled by the Xtensa LLVM backend when it holds a
            // 16-byte array copy (proven on device).
            let args = Tie728AvgPoolArgs {
                input_channel: cfg.channels,
                input_y_offset: cfg.input_y_offset,
                input_x_offset: cfg.input_x_offset,
                filter_height: cfg.filter_h,
                filter_width: cfg.filter_w,
                shift: cfg.shift,
                avg_pool_area_inv: cfg.area_inv,
                c_div_x_1: cfg.c_div_x_1,
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

    /// Run ONE avg-pool pixel through the generic TIE728 `*_hwc1` entry —
    /// the arbitrary-filter kernel (same window walk as the max `*_hwc1`,
    /// accumulating `in · area_inv` into QACC and applying the fixed-point
    /// `shift` round). Same single-`&mut`-ctx contract and register pinning.
    ///
    /// # Safety
    ///
    /// `ctx.output`/`ctx.input` 16-byte aligned, buffers sized per
    /// `simd_eligible_pool` eligibility (pad 0, channels % 16); the window
    /// anchored at `ctx.input` must be fully in-bounds (the caller positions
    /// the origin).
    #[inline(never)]
    #[allow(dead_code)]
    pub unsafe fn avg_pool_2d_simd_ctx_hwc1(ctx: &mut AvgPoolSimdCtx) {
        let target = dl_tie728_s8_avg_pool2d_hwc1 as unsafe extern "C" fn() as usize;
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

// ── Generic hwc1 SIMD drivers (device-only) ───────────────────────────────
//
// Drive the vendored `dl_tie728_s8_{avg,max}_pool2d_hwc1` entry points —
// the arbitrary-filter kernels — once per output pixel. The caller positions
// the window origin (stride handled by the pointer arithmetic); the kernel
// walks filter_h × filter_w 16-lane vectors from there and stores one
// `channels`-byte output pixel. The clamp runs as a Rust post-pass (the
// kernels have no clamp field). `simd_eligible_pool` already guaranteed
// pad 0, channels % 16, filter ≥ 2 taps, stride ≥ 1.

/// Run the generic avg-pool hwc1 SIMD path for an eligible `cfg`.
///
/// Returns `Ok(true)` when handled, `Ok(false)` when the pointers are not
/// 16-byte aligned (the caller falls through to the scalar kernel — the
/// established alignment-gate fallback).
#[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
fn avg_pool_2d_generic_simd(
    input: &[i8],
    params: &PoolParams,
    output: &mut [i8],
    cfg: &PoolSimdCfg,
) -> Result<bool, KernelError> {
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    if (in_ptr as usize) % 16 != 0 || (out_ptr as usize) % 16 != 0 {
        return Ok(false);
    }
    let channels = cfg.channels as usize;
    let in_w = params.input_shape[2] as usize;
    let out_w = params.output_shape[2] as usize;
    let out_h = params.output_shape[1] as usize;
    let stride_h = params.stride_height.max(1) as usize;
    let stride_w = params.stride_width.max(1) as usize;
    let mut ctx = pool_simd::AvgPoolSimdCtx::new(out_ptr, in_ptr, cfg);
    unsafe {
        let in_row_step = (stride_h * in_w) * channels;
        let out_row_step = out_w * channels;
        for oh in 0..out_h {
            let mut in_px = in_ptr.add((oh * in_row_step) as usize);
            ctx.output = out_ptr.add((oh * out_row_step) as usize);
            for _ow in 0..out_w {
                ctx.input = in_px;
                pool_simd::avg_pool_2d_simd_ctx_hwc1(&mut ctx);
                if cfg.act_min != i8::MIN as i32 || cfg.act_max != i8::MAX as i32 {
                    for c in 0..channels {
                        let v = *ctx.output.add(c);
                        *ctx.output.add(c) =
                            clamp_i8(i32::from(v), cfg.act_min, cfg.act_max);
                    }
                }
                ctx.output = ctx.output.add(channels);
                in_px = in_px.add((stride_w * channels) as usize);
            }
        }
    }
    Ok(true)
}

/// Run the generic max-pool hwc1 SIMD path for an eligible `cfg`.
///
/// Same contract as [`avg_pool_2d_generic_simd`] (alignment gate, per-pixel
/// kernel calls, clamp post-pass).
#[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
fn max_pool_2d_generic_simd(
    input: &[i8],
    params: &PoolParams,
    output: &mut [i8],
    cfg: &PoolSimdCfg,
) -> Result<bool, KernelError> {
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    if (in_ptr as usize) % 16 != 0 || (out_ptr as usize) % 16 != 0 {
        return Ok(false);
    }
    let channels = cfg.channels as usize;
    let in_w = params.input_shape[2] as usize;
    let out_w = params.output_shape[2] as usize;
    let out_h = params.output_shape[1] as usize;
    let stride_h = params.stride_height.max(1) as usize;
    let stride_w = params.stride_width.max(1) as usize;
    let mut ctx = pool_simd::MaxPoolSimdCtx::new(out_ptr, in_ptr, cfg);
    unsafe {
        let in_row_step = (stride_h * in_w) * channels;
        let out_row_step = out_w * channels;
        for oh in 0..out_h {
            let mut in_px = in_ptr.add((oh * in_row_step) as usize);
            ctx.output = out_ptr.add((oh * out_row_step) as usize);
            for _ow in 0..out_w {
                ctx.input = in_px;
                pool_simd::max_pool_2d_simd_ctx_hwc1(&mut ctx);
                if cfg.act_min != i8::MIN as i32 || cfg.act_max != i8::MAX as i32 {
                    for c in 0..channels {
                        let v = *ctx.output.add(c);
                        *ctx.output.add(c) =
                            clamp_i8(i32::from(v), cfg.act_min, cfg.act_max);
                    }
                }
                ctx.output = ctx.output.add(channels);
                in_px = in_px.add((stride_w * channels) as usize);
            }
        }
    }
    Ok(true)
}

// ── Prepared-pool fast path ──────────────────────────────────────────────

/// SIMD-eligibility result for the generic pool path (T3.1) — the TIE728
/// args fields plus the avg fixed-point constants and the activation range
/// for the Rust clamp post-pass (the vendored kernels have no clamp field).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PoolSimdCfg {
    /// `input_channel` — total channel count (`% 16 == 0`).
    pub channels: i32,
    /// `input_y_offset` — byte offset between input rows (`input_w * channels`).
    pub input_y_offset: i32,
    /// `input_x_offset` — byte offset between adjacent filter columns (`channels`).
    pub input_x_offset: i32,
    /// `filter_height` / `filter_width` — the hwc1 kernel's window walk.
    pub filter_h: i32,
    pub filter_w: i32,
    /// `c_div_x_1` — channel-group count minus one (`channels / 16 - 1`).
    pub c_div_x_1: i32,
    /// `c_remainder` — trailing channels below 16 (0: the gate requires % 16).
    pub c_remainder: i32,
    /// avg only: the fixed-point `shift` for `tie728_s8_vector_round_result`.
    pub shift: i32,
    /// avg only: the packed 16-lane reciprocal `round(2^shift / area)`.
    pub area_inv: [i8; 16],
    /// Activation range for the clamp post-pass.
    pub act_min: i32,
    pub act_max: i32,
}

/// `round(2^shift / area)` — the avg-pool reciprocal (half-up).
fn round_recip(area: i32, shift: i32) -> i32 {
    let num = 1i64 << shift;
    ((num + area as i64 / 2) / area as i64) as i32
}

/// The fixed-point avg-pool reciprocal constants — the established
/// s3/C-SIMD semantics. `area_inv = round(2^shift / area)` packed into the
/// 16-lane i8 vector the vendored kernel broadcasts, with `shift` grown from
/// 8 until the reciprocal fits the i8 lane. The 2×2 filter (area 4) yields
/// the established `(shift=8, area_inv=[64;16])` pair — bit-exact with the
/// on-silicon `dl_tie728_s8_avg_pool2d_22c1` configuration. `None` when the
/// reciprocal cannot fit (area < 2 or never ≤ 127).
fn pool_area_inv(area: i32) -> Option<(i32, [i8; 16])> {
    let mut shift = 8;
    let mut inv = round_recip(area, shift);
    while inv > 127 && shift < 24 {
        shift += 1;
        inv = round_recip(area, shift);
    }
    if !(1..=127).contains(&inv) {
        return None;
    }
    Some((shift, [inv as i8; 16]))
}

/// Shared SIMD-eligibility gate for generic pooling (both avg and max).
///
/// Host-compilable: returns `Some(cfg)` when the TIE728 `*_hwc1` entry points
/// can run, so a handle can be built once and reused across calls without
/// re-running the gate.
///
/// Eligibility (T3.1 widening): any filter ≥ 2 taps, stride ≥ 1, channels
/// `% 16 == 0`, an activation range of any size (the clamp runs as a Rust
/// post-pass), and **no padding / no partial windows**: `(out-1)·stride +
/// filter - in ≤ 0` on both axes (see the module doc — the pool backend
/// delivers no scratch, so boundary windows cannot be staged on device, and
/// the fixed-point full-area divisor would diverge from the scalar
/// valid-tap count). VALID shapes satisfy this; SAME shapes only when the
/// SAME output equals the VALID output (2×2/stride-2 on even inputs — the
/// established family — included). The legacy 2×2/stride-2/full-range shapes
/// remain eligible and dispatch through the byte-identical `*_22c1` path.
pub(crate) fn simd_eligible_pool(params: &PoolParams) -> Option<PoolSimdCfg> {
    let input_h = params.input_shape[1];
    let input_w = params.input_shape[2];
    let channels = params.input_shape[3];
    let filter_h = params.filter_height;
    let filter_w = params.filter_width;
    let out_h = params.output_shape[1];
    let out_w = params.output_shape[2];

    let pad_total_h = (out_h - 1) * params.stride_height + filter_h - input_h;
    let pad_total_w = (out_w - 1) * params.stride_width + filter_w - input_w;

    if filter_h < 1
        || filter_w < 1
        || params.stride_height < 1
        || params.stride_width < 1
        || pad_total_h > 0
        || pad_total_w > 0
        || channels <= 0
        || channels % 16 != 0
    {
        return None;
    }
    let (shift, area_inv) = pool_area_inv(filter_h * filter_w)?;
    Some(PoolSimdCfg {
        channels,
        input_y_offset: input_w * channels,
        input_x_offset: channels,
        filter_h,
        filter_w,
        c_div_x_1: channels / 16 - 1,
        c_remainder: 0,
        shift,
        area_inv,
        act_min: params.quantized_activation_min,
        act_max: params.quantized_activation_max,
    })
}

/// Prepared generic max pool — runs the SIMD gate once at construction.
pub struct PreparedMaxPool {
    simd: Option<PoolSimdCfg>,
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
            if let Some(cfg) = &self.simd {
                // Legacy 2×2/stride-2/full-range shapes keep the byte-identical
                // `*_22c1` driver; everything else eligible uses the generic
                // `*_hwc1` driver (both return Ok(false) on the misaligned
                // fallback — the scalar kernel then runs).
                let is_legacy = self.params.filter_height == 2
                    && self.params.filter_width == 2
                    && self.params.stride_height == 2
                    && self.params.stride_width == 2
                    && cfg.act_min == i8::MIN as i32
                    && cfg.act_max == i8::MAX as i32;
                if is_legacy {
                    let in_ptr = input.as_ptr();
                    let out_ptr = output.as_mut_ptr();
                    if (in_ptr as usize) % 16 == 0 && (out_ptr as usize) % 16 == 0 {
                        let channels = cfg.channels as usize;
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
                } else if max_pool_2d_generic_simd(input, self.params, output, cfg)? {
                    let _ = scratch;
                    return Ok(());
                }
            }
        }
        max_pool_2d(input, self.params, output, scratch)
    }
}

/// Prepared generic average pool — runs the SIMD gate once at construction.
pub struct PreparedAvgPool {
    simd: Option<PoolSimdCfg>,
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
            if let Some(cfg) = &self.simd {
                let is_legacy = self.params.filter_height == 2
                    && self.params.filter_width == 2
                    && self.params.stride_height == 2
                    && self.params.stride_width == 2
                    && cfg.act_min == i8::MIN as i32
                    && cfg.act_max == i8::MAX as i32;
                if is_legacy {
                    let in_ptr = input.as_ptr();
                    let out_ptr = output.as_mut_ptr();
                    if (in_ptr as usize) % 16 == 0 && (out_ptr as usize) % 16 == 0 {
                        let channels = cfg.channels as usize;
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
                } else if avg_pool_2d_generic_simd(input, self.params, output, cfg)? {
                    let _ = scratch;
                    return Ok(());
                }
            }
        }
        average_pool_2d(input, self.params, output, scratch)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host-compilable generic pool SIMD models + bit-exact gate tests (T3.1)
// ─────────────────────────────────────────────────────────────────────────────
//
// The models below are the HOST-TESTED math the device `*_hwc1` asm kernels
// implement: avg uses the established s3/C-SIMD fixed-point semantics
// (`acc = Σ in·area_inv` over the FULL window, `sat8((sat8(acc >> (shift-1))
// + 1) >> 1)`, clamp) and max uses element-wise max over the window with
// `i8::MIN` padding (never wins the max — equal to the scalar fallback's
// valid-tap-only loop). Both have a 16-wide vector main loop over the channel
// groups (the TIE728 lane width) plus a scalar tail for `C % 16` — the shape
// the device gate accepts (`channels % 16 == 0`) plus the tail the model
// proves correct.
#[cfg(test)]
mod generic_pool_simd_model {
    extern crate alloc;
    extern crate std;
    use super::*;
    use alloc::format;
    use alloc::vec;
    use alloc::string::String;
    use alloc::vec::Vec;

    /// `tie728_s8_vector_round_result` (dl_tie728_s8.S:175) on one lane:
    /// `((acc >> (shift-1)) + 1) >> 1` for `shift > 0` (round-half-up in the
    /// shifted domain — arithmetic shifts, no intermediate saturation),
    /// plain `acc` for `shift == 0`. The device-validated relationship to
    /// `hematite-ref`'s `round_half_away_zero(acc, 2^shift)` is ±1 LSB on
    /// negative half-even sums (the documented `avg_pool ±1` known-delta).
    fn round_result(acc: i32, shift: i32) -> i32 {
        if shift == 0 {
            acc
        } else {
            ((acc >> (shift - 1)) + 1) >> 1
        }
    }

    /// Generic avg-pool SIMD model — the fixed-point s3/C-SIMD semantics for
    /// arbitrary filter/stride/pad (padding taps contribute 0) and any clamp.
    pub(super) fn avg_pool_2d_simd(
        input: &[i8],
        params: &PoolParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        let input_h = params.input_shape[1] as usize;
        let input_w = params.input_shape[2] as usize;
        let channels = params.input_shape[3] as usize;
        let filter_h = params.filter_height as usize;
        let filter_w = params.filter_width as usize;
        let out_h = params.output_shape[1] as usize;
        let out_w = params.output_shape[2] as usize;
        let (shift, area_inv) = pool_area_inv(filter_h as i32 * filter_w as i32)
            .ok_or(KernelError::Unsupported)?;
        let pad_h = ((params.output_shape[1] - 1) * params.stride_height
            + params.filter_height
            - params.input_shape[1])
            / 2;
        let pad_w = ((params.output_shape[2] - 1) * params.stride_width
            + params.filter_width
            - params.input_shape[2])
            / 2;
        let out_row_stride = out_w * channels;

        for oh in 0..out_h {
            let in_y_origin = oh as i32 * params.stride_height - pad_h;
            let fy_start = 0i32.max(-in_y_origin);
            let fy_end = (input_h as i32 - in_y_origin).min(filter_h as i32);
            for ow in 0..out_w {
                let in_x_origin = ow as i32 * params.stride_width - pad_w;
                let fx_start = 0i32.max(-in_x_origin);
                let fx_end = (input_w as i32 - in_x_origin).min(filter_w as i32);
                // 16-wide vector main loop over the full channel groups.
                let mut c = 0usize;
                for g in 0..channels / 16 {
                    let mut acc = [0i32; 16];
                    for fy in fy_start..fy_end {
                        let in_y = in_y_origin + fy;
                        for fx in fx_start..fx_end {
                            let in_x = in_x_origin + fx;
                            let base = (in_y * input_w as i32 + in_x) as usize * channels;
                            for l in 0..16 {
                                acc[l] += i32::from(input[base + g * 16 + l]) * i32::from(area_inv[0]);
                            }
                        }
                    }
                    for l in 0..16 {
                        let r = round_result(acc[l], shift);
                        let out_idx = (oh * out_row_stride + ow * channels + g * 16 + l) as usize;
                        output[out_idx] = clamp_i8(
                            r,
                            params.quantized_activation_min,
                            params.quantized_activation_max,
                        );
                    }
                    c = g * 16 + 16;
                }
                // Scalar tail for C % 16.
                for oc in c..channels {
                    let mut acc: i32 = 0;
                    for fy in fy_start..fy_end {
                        let in_y = in_y_origin + fy;
                        for fx in fx_start..fx_end {
                            let in_x = in_x_origin + fx;
                            let idx =
                                (in_y * input_w as i32 + in_x) as usize * channels + oc;
                            acc += i32::from(input[idx]) * i32::from(area_inv[0]);
                        }
                    }
                    let r = round_result(acc, shift);
                    let out_idx = (oh * out_row_stride + ow * channels + oc) as usize;
                    output[out_idx] = clamp_i8(
                        r,
                        params.quantized_activation_min,
                        params.quantized_activation_max,
                    );
                }
            }
        }
        Ok(())
    }

    /// Generic max-pool SIMD model — element-wise max over the window with
    /// `i8::MIN` pad fill (never wins), then clamp. Bit-exact vs the scalar
    /// fallback and `hematite-ref` for every shape.
    pub(super) fn max_pool_2d_simd(
        input: &[i8],
        params: &PoolParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        let input_h = params.input_shape[1] as usize;
        let input_w = params.input_shape[2] as usize;
        let channels = params.input_shape[3] as usize;
        let filter_h = params.filter_height as usize;
        let filter_w = params.filter_width as usize;
        let out_h = params.output_shape[1] as usize;
        let out_w = params.output_shape[2] as usize;
        let pad_h = ((params.output_shape[1] - 1) * params.stride_height
            + params.filter_height
            - params.input_shape[1])
            / 2;
        let pad_w = ((params.output_shape[2] - 1) * params.stride_width
            + params.filter_width
            - params.input_shape[2])
            / 2;
        let out_row_stride = out_w * channels;

        for oh in 0..out_h {
            let in_y_origin = oh as i32 * params.stride_height - pad_h;
            let fy_start = 0i32.max(-in_y_origin);
            let fy_end = (input_h as i32 - in_y_origin).min(filter_h as i32);
            for ow in 0..out_w {
                let in_x_origin = ow as i32 * params.stride_width - pad_w;
                let fx_start = 0i32.max(-in_x_origin);
                let fx_end = (input_w as i32 - in_x_origin).min(filter_w as i32);
                let mut c = 0usize;
                for g in 0..channels / 16 {
                    let mut lanes = [i8::MIN; 16];
                    for fy in fy_start..fy_end {
                        let in_y = in_y_origin + fy;
                        for fx in fx_start..fx_end {
                            let in_x = in_x_origin + fx;
                            let base = (in_y * input_w as i32 + in_x) as usize * channels;
                            for l in 0..16 {
                                lanes[l] = lanes[l].max(input[base + g * 16 + l]);
                            }
                        }
                    }
                    for l in 0..16 {
                        let out_idx = (oh * out_row_stride + ow * channels + g * 16 + l) as usize;
                        output[out_idx] = clamp_i8(
                            i32::from(lanes[l]),
                            params.quantized_activation_min,
                            params.quantized_activation_max,
                        );
                    }
                    c = g * 16 + 16;
                }
                for oc in c..channels {
                    let mut max_val = i8::MIN;
                    for fy in fy_start..fy_end {
                        let in_y = in_y_origin + fy;
                        for fx in fx_start..fx_end {
                            let in_x = in_x_origin + fx;
                            let idx =
                                (in_y * input_w as i32 + in_x) as usize * channels + oc;
                            max_val = max_val.max(input[idx]);
                        }
                    }
                    let out_idx = (oh * out_row_stride + ow * channels + oc) as usize;
                    output[out_idx] = clamp_i8(
                        i32::from(max_val),
                        params.quantized_activation_min,
                        params.quantized_activation_max,
                    );
                }
            }
        }
        Ok(())
    }

    /// Build a `PoolParams` for the matrix: `in` spatial `(ih, iw)` with
    /// `channels`, filter `(fh, fw)`, stride `(sh, sw)`, and `pad` in
    /// {0, 1, SAME} (`SAME` → the s3 formula's pad for `out = ceil(in/s)`).
    fn pool_params(
        ih: i32,
        iw: i32,
        channels: i32,
        fh: i32,
        fw: i32,
        sh: i32,
        sw: i32,
        pad: PadMode,
        act_min: i32,
        act_max: i32,
    ) -> PoolParams {
        let (out_h, pad_h) = match pad {
            PadMode::Zero => ((ih - fh) / sh + 1, 0),
            PadMode::One => (ih, 1),
            PadMode::Same => {
                let oh = (ih + sh - 1) / sh;
                let p = ((oh - 1) * sh + fh - ih) / 2;
                (oh, p)
            }
        };
        let (out_w, pad_w) = match pad {
            PadMode::Zero => ((iw - fw) / sw + 1, 0),
            PadMode::One => (iw, 1),
            PadMode::Same => {
                let ow = (iw + sw - 1) / sw;
                let p = ((ow - 1) * sw + fw - iw) / 2;
                (ow, p)
            }
        };
        PoolParams {
            input_shape: [1, ih, iw, channels],
            output_shape: [1, out_h, out_w, channels],
            filter_width: fw,
            filter_height: fh,
            stride_width: sw,
            stride_height: sh,
            padding: if pad_h > 0 || pad_w > 0 {
                hematite_core::op_params::Padding::Same
            } else {
                hematite_core::op_params::Padding::Valid
            },
            activation: hematite_core::op_params::FusedActivation::None,
            quantized_activation_min: act_min,
            quantized_activation_max: act_max,
        }
    }

    #[derive(Clone, Copy)]
    enum PadMode {
        Zero,
        One,
        Same,
    }

    /// Deterministic LCG `i8` pattern (full int8 range).
    fn pattern(seed: u32, n: usize) -> Vec<i8> {
        let mut out = vec![0i8; n];
        let mut x = seed;
        for v in out.iter_mut() {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *v = (x >> 16) as i8;
        }
        out
    }

    fn shape_product(shape: &[i32; 4]) -> usize {
        shape[0] as usize * shape[1] as usize * shape[2] as usize * shape[3] as usize
    }

    fn fnv1a(data: &[i8]) -> u32 {
        let mut h: u32 = 2_166_136_261;
        for &b in data {
            h ^= u32::from(b as u8);
            h = h.wrapping_mul(16_777_619);
        }
        h
    }

    /// The matrix: filter {2×2, 3×3, 5×5, global 7×7} × stride {1, 2} ×
    /// pad {0, 1, SAME} × clamp {full-range, relu} × channels {16, 24}.
    fn matrix() -> Vec<(PoolParams, String)> {
        let mut rows = Vec::new();
        for &(ih, iw, fh, fw, tag) in &[
            (8, 8, 2, 2, "2x2"),
            (8, 8, 3, 3, "3x3"),
            (8, 8, 5, 5, "5x5"),
            (7, 7, 7, 7, "global7x7"),
        ] {
            for &(sh, sw, st_tag) in &[(1, 1, "s1"), (2, 2, "s2")] {
                if (sh > 1 || sw > 1) && (fh < sh || fw < sw) {
                    continue; // degenerate: filter smaller than stride
                }
                for &(pad, pad_tag) in
                    &[(PadMode::Zero, "p0"), (PadMode::One, "p1"), (PadMode::Same, "same")]
                {
                    if pad_tag == "p1" && (fh != 3 || sh != 1) {
                        continue; // "pad 1" is the 3×3-stride-1 shape
                    }
                    for &(act_min, act_max, cl_tag) in &[
                        (i8::MIN as i32, i8::MAX as i32, "full"),
                        (0, 127, "relu"),
                    ] {
                        for &ch in &[16i32, 24] {
                            let p = pool_params(ih, iw, ch, fh, fw, sh, sw, pad, act_min, act_max);
                            rows.push((p, format!("{tag}_{st_tag}_{pad_tag}_{cl_tag}_c{ch}")));
                        }
                    }
                }
            }
        }
        rows
    }

    /// Max-pool SIMD model == `hematite-ref` scalar bit-exact across the
    /// whole matrix (max semantics are the ref semantics: window max with
    /// `i8::MIN` padding, then clamp).
    #[test]
    fn pool_generic_max_model_matches_ref_bit_exact() {
        let mut checked = 0;
        for (params, tag) in matrix() {
            let in_len = shape_product(&params.input_shape);
            let out_len = shape_product(&params.output_shape);
            let input = pattern(0xBEEF_0001 + checked as u32 * 31, in_len);
            let mut got = vec![0i8; out_len];
            let mut want = vec![0i8; out_len];
            max_pool_2d_simd(&input, &params, &mut got).expect("model max runs");
            hematite_ref::pool::max_pool_2d(&input, &params, &mut want, &mut [])
                .expect("ref max runs");
            assert_eq!(got, want, "max model != ref for {tag}");
            checked += 1;
        }
        assert!(checked >= 40, "matrix must cover the widened family");
    }

    /// Avg-pool SIMD model vs `hematite-ref` — the divergence is the
    /// DOCUMENTED s3 fixed-point semantics: for the 2×2/stride-2/pad-0
    /// shapes the model must be bit-exact vs the established 22c1 fixed-point
    /// emulation, and across the matrix the model must stay within the
    /// documented LSB bounds of the ref (the fixed-point round ±1 class; the
    /// mid-step int8 saturation widens the bound for large windows). The
    /// per-shape FNV checksums print for the evidence table.
    #[test]
    fn pool_generic_avg_model_matches_established_semantics() {
        let mut checked = 0;
        let mut max_delta = 0;
        for (params, tag) in matrix() {
            let in_len = shape_product(&params.input_shape);
            let out_len = shape_product(&params.output_shape);
            let input = pattern(0x0BAD_0001 + checked as u32 * 29, in_len);
            let mut got = vec![0i8; out_len];
            let mut want = vec![0i8; out_len];
            avg_pool_2d_simd(&input, &params, &mut got).expect("model avg runs");
            hematite_ref::pool::average_pool_2d(&input, &params, &mut want, &mut [])
                .expect("ref avg runs");
            let area = params.filter_height * params.filter_width;
            let mut delta = 0;
            for (g, w) in got.iter().zip(want.iter()) {
                delta = delta.max((i32::from(*g) - i32::from(*w)).abs());
            }
            max_delta = max_delta.max(delta);
            // Established 2×2/stride-2/pad-0/full-range shapes: the model must
            // match the 22c1 fixed-point emulation bit-exact (the test-data
            // sums stay far below the mid-step saturation threshold here).
            let legacy = params.filter_height == 2
                && params.filter_width == 2
                && params.stride_height == 2
                && params.stride_width == 2
                && params.quantized_activation_min == i8::MIN as i32
                && params.quantized_activation_max == i8::MAX as i32;
            if legacy {
                let mut emu = vec![0i8; out_len];
                established_avg_2x2_emulation(&input, &params, &mut emu).expect("emu runs");
                assert_eq!(
                    got, emu,
                    "avg model != established 2×2 fixed-point for {tag}"
                );
            }
            // Partial-window shapes (asymmetric SAME: pad_total > 0) split the
            // fixed-point FULL-area divisor from the ref's valid-tap count —
            // an unbounded, DOCUMENTED divergence (those shapes keep the
            // scalar fallback on device). Only the no-partial-window shapes
            // carry a strict LSB bound.
            let pad_total_h = (params.output_shape[1] - 1) * params.stride_height
                + params.filter_height
                - params.input_shape[1];
            let pad_total_w = (params.output_shape[2] - 1) * params.stride_width
                + params.filter_width
                - params.input_shape[2];
            if pad_total_h <= 0 && pad_total_w <= 0 {
                // Documented divergence bounds: the fixed-point round vs
                // round_half_away_zero diverges ±1 on negative half-even sums;
                // the reciprocal is exact only for power-of-two areas, so the
                // non-power-of-two windows carry a bounded systematic error
                // (area 9: 28/256 vs 1/9, …). Bounds per area are recorded in
                // the evidence file.
                let bound = match area {
                    4 => 1,
                    9 => 2,
                    25 => 3,
                    _ => 6,
                };
                assert!(
                    delta <= bound,
                    "avg model diverged {delta} LSB (> {bound}) vs ref for {tag}"
                );
            }
            std::println!(
                "t31 avg {tag}: model_fnv=0x{:08x} ref_fnv=0x{:08x} max_delta={delta}",
                fnv1a(&got),
                fnv1a(&want)
            );
            checked += 1;
        }
        std::eprintln!("t31 avg matrix: {checked} shapes, max_delta={max_delta}");
    }

    /// Direct emulation of `dl_tie728_s8_avg_pool2d_22c1` on the 2×2/stride-2
    /// shapes: `acc = Σ 4 taps · 64`, `round_result(acc, 8)`, no clamp (the
    /// established device semantics; written independently of the model so
    /// the model-vs-established comparison is not circular).
    fn established_avg_2x2_emulation(
        input: &[i8],
        params: &PoolParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        let input_w = params.input_shape[2] as usize;
        let channels = params.input_shape[3] as usize;
        let out_h = params.output_shape[1] as usize;
        let out_w = params.output_shape[2] as usize;
        let out_row_stride = out_w * channels;
        for oh in 0..out_h {
            for ow in 0..out_w {
                for c in 0..channels {
                    let mut acc: i32 = 0;
                    for (kh, kw) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)] {
                        let idx =
                            ((oh * 2 + kh) * input_w + ow * 2 + kw) * channels + c;
                        acc += i32::from(input[idx]) * 64;
                    }
                    let r = round_result(acc, 8);
                    output[(oh * out_row_stride + ow * channels + c) as usize] =
                        clamp_i8(r, i8::MIN as i32, i8::MAX as i32);
                }
            }
        }
        Ok(())
    }

    /// The 32×32×16 bench shape (spec.rs SIMD_AVGPOOL_32X32_PARAMS fill) —
    /// computes the model checksums that the evidence table compares against
    /// the documented C-SIMD numbers.
    #[test]
    fn pool_established_32x32_checksums_reported() {
        let in_len = 32 * 32 * 16;
        let out_len = 16 * 16 * 16;
        let input: Vec<i8> = (0..in_len).map(|i| ((i * 7 + 3) & 0xFF) as i8).collect();
        let params = PoolParams {
            input_shape: [1, 32, 32, 16],
            output_shape: [1, 16, 16, 16],
            filter_width: 2,
            filter_height: 2,
            stride_width: 2,
            stride_height: 2,
            padding: hematite_core::op_params::Padding::Valid,
            activation: hematite_core::op_params::FusedActivation::None,
            quantized_activation_min: i8::MIN as i32,
            quantized_activation_max: i8::MAX as i32,
        };
        let mut avg = vec![0i8; out_len];
        let mut max = vec![0i8; out_len];
        avg_pool_2d_simd(&input, &params, &mut avg).expect("model avg runs");
        max_pool_2d_simd(&input, &params, &mut max).expect("model max runs");
        let mut avg_ref = vec![0i8; out_len];
        let mut max_ref = vec![0i8; out_len];
        hematite_ref::pool::average_pool_2d(&input, &params, &mut avg_ref, &mut [])
            .expect("ref avg runs");
        hematite_ref::pool::max_pool_2d(&input, &params, &mut max_ref, &mut [])
            .expect("ref max runs");
        std::eprintln!(
            "t31 established 32x32: avg_simd=0x{:08x} avg_ref=0x{:08x} max_simd=0x{:08x} max_ref=0x{:08x}",
            fnv1a(&avg),
            fnv1a(&avg_ref),
            fnv1a(&max),
            fnv1a(&max_ref)
        );
        // The model's max must equal the ref (device-validated PASS); the
        // model's avg is the fixed-point semantics (documented ±1 class).
        assert_eq!(max, max_ref, "established 32x32 max model must match ref");
    }

    /// The widened gate: legacy 2×2 stays eligible with the exact established
    /// constants; generic filters/strides/clamps engage; pad > 0 and C % 16
    /// fall back (documented).
    #[test]
    fn pool_simd_eligible_gate_expectations() {
        use hematite_core::op_params::Padding;
        let base = |fh: i32, fw: i32, sh: i32, sw: i32, ih: i32, iw: i32, c: i32, min: i32, max: i32| {
            let (oh, ow) = ((ih - fh) / sh + 1, (iw - fw) / sw + 1);
            PoolParams {
                input_shape: [1, ih, iw, c],
                output_shape: [1, oh, ow, c],
                filter_width: fw,
                filter_height: fh,
                stride_width: sw,
                stride_height: sh,
                padding: Padding::Valid,
                activation: hematite_core::op_params::FusedActivation::None,
                quantized_activation_min: min,
                quantized_activation_max: max,
            }
        };

        // Legacy 2×2/stride-2/full-range → Some with the exact established
        // fixed-point constants (shift 8, area_inv [64; 16]).
        let legacy = base(2, 2, 2, 2, 32, 32, 16, i8::MIN as i32, i8::MAX as i32);
        let cfg = simd_eligible_pool(&legacy).expect("legacy 2x2 must be eligible");
        assert_eq!(cfg.shift, 8);
        assert_eq!(cfg.area_inv, [64i8; 16]);
        assert_eq!(cfg.c_div_x_1, 0);

        // Generic: 3×3 stride-1 VALID, 5×5 stride-1 VALID, relu clamp.
        let c3 = base(3, 3, 1, 1, 8, 8, 16, i8::MIN as i32, i8::MAX as i32);
        assert!(simd_eligible_pool(&c3).is_some(), "3x3 s1 p0 must engage");
        let c5 = base(5, 5, 1, 1, 8, 8, 16, i8::MIN as i32, i8::MAX as i32);
        assert!(simd_eligible_pool(&c5).is_some(), "5x5 s1 p0 must engage");
        let relu = base(3, 3, 1, 1, 8, 8, 16, 0, 127);
        assert!(simd_eligible_pool(&relu).is_some(), "relu clamp must engage");
        let stride2 = base(3, 3, 2, 2, 8, 8, 16, i8::MIN as i32, i8::MAX as i32);
        assert!(simd_eligible_pool(&stride2).is_some(), "3x3 s2 p0 must engage");

        // Pad > 0 / partial windows → scalar (documented: the pool backend
        // delivers no scratch, and the fixed-point full-area divisor would
        // diverge from the scalar valid-tap count).
        let same = {
            let mut p = c3;
            p.output_shape = [1, 8, 8, 16];
            p.padding = Padding::Same;
            p
        };
        assert!(simd_eligible_pool(&same).is_none(), "pad>0 must fall back");
        // 2×2/stride-2 SAME on an even input (pad_total == 0 — the established
        // family's SAME form) → eligible.
        let same_even = {
            let mut p = base(2, 2, 2, 2, 8, 8, 16, i8::MIN as i32, i8::MAX as i32);
            p.output_shape = [1, 4, 4, 16];
            p.padding = Padding::Same;
            p
        };
        assert!(
            simd_eligible_pool(&same_even).is_some(),
            "2x2 s2 SAME even must engage"
        );
        // Asymmetric SAME (2×2 stride-1, 8→8 — partial windows) → scalar.
        let asym = {
            let mut p = base(2, 2, 1, 1, 8, 8, 16, i8::MIN as i32, i8::MAX as i32);
            p.output_shape = [1, 8, 8, 16];
            p.padding = Padding::Same;
            p
        };
        assert!(simd_eligible_pool(&asym).is_none(), "asymmetric SAME must fall back");        // C % 16 → scalar.
        let c8 = base(2, 2, 2, 2, 32, 32, 8, i8::MIN as i32, i8::MAX as i32);
        assert!(simd_eligible_pool(&c8).is_none(), "C%16 must fall back");
        // area < 2 (1×1 filter) → scalar (the reciprocal cannot fit i8).
        let one = base(1, 1, 1, 1, 8, 8, 16, i8::MIN as i32, i8::MAX as i32);
        assert!(simd_eligible_pool(&one).is_none(), "1x1 must fall back");
    }
}
