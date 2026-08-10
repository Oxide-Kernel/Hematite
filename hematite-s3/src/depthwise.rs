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

    // ── SIMD dispatch (bespoke QACC depthwise kernel, bit-exact) ─────────
    #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
    {
        let mut accx_ctx = DepthwiseAccxCtx {
            input,
            weights,
            bias,
            params,
            output,
            scratch,
        };
        if depthwise_accx_dispatch(&mut accx_ctx)? {
            return Ok(());
        }
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
// Host-compilable SIMD-pipeline helpers (T3.5 — depth_multiplier > 1).
//
// The device dispatch below stages a replicated input and applies the
// input_offset fold; these helpers implement exactly that staging/fold so the
// host SIMD-model tests exercise the real device-pipeline code. Compiled on
// the device (used by the dispatch) and under `#[cfg(test)]` on host (used by
// the unit tests); never compiled into host release builds.
// ─────────────────────────────────────────────────────────────────────────────

/// Stage the real input interior into the padded/replicated staged buffer.
///
/// `dst` is the pre-filled staged buffer (fill = `-input_offset`, or 0) of
/// `padded_h×padded_w×dst_c` bytes; the real pixel at (h, w) is written at
/// ((h + pad_h), (w + pad_w)). dm==1 copies channels 1:1 (the historical
/// path); dm>1 replicates each input channel `depth_multiplier` times so
/// output channel `oc = i*dm + j` reads input channel `i` — the TFLM
/// depthwise fan-out the SIMD kernel consumes as per-lane values.
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
pub(crate) fn stage_depthwise_pixels(
    dst: &mut [u8],
    dst_c: usize,
    dst_w: usize,
    src: &[i8],
    in_h: usize,
    in_w: usize,
    in_c: usize,
    depth_multiplier: usize,
    pad_h: usize,
    pad_w: usize,
) {
    for h in 0..in_h {
        for w in 0..in_w {
            let srow = (h * in_w + w) * in_c;
            let drow = ((h + pad_h) * dst_w + (w + pad_w)) * dst_c;
            for i in 0..in_c {
                let v = src[srow + i] as u8;
                let base = drow + i * depth_multiplier;
                for j in 0..depth_multiplier {
                    dst[base + j] = v;
                }
            }
        }
    }
}

/// Phase-C input_offset fold: `acc[oc] += input_offset * wsum[oc]` in wrapping
/// i32 — bit-identical to the scalar `Σ(in + off)·w = Σ in·w + off·Σw` split
/// (the kernel produces `Σ in·w`).
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
pub(crate) fn fold_input_offset(accs: &mut [i32], wsum: &[i32], input_offset: i32) {
    for oc in 0..accs.len() {
        let v = accs[oc];
        let s = wsum[oc];
        accs[oc] = v.wrapping_add(input_offset.wrapping_mul(s));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TIE728 SIMD backend — device-only (NEVER compiled on host)
// ─────────────────────────────────────────────────────────────────────────────

/// Context for the bespoke QACC depthwise dispatch — bundled into one `&mut`
/// arg so the Xtensa LLVM backend generates a 1-arg call (multi-arg calls are
/// miscompiled on device; see the Xtensa multi-arg call miscompile class).
#[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
pub(crate) struct DepthwiseAccxCtx<'a> {
    pub input: &'a [i8],
    pub weights: &'a [i8],
    pub bias: &'a [i32],
    pub params: &'a DepthwiseConv2DParams<'a>,
    pub output: &'a mut [i8],
    pub scratch: &'a mut [u8],
}

/// Bespoke QACC SIMD dispatch for the depthwise conv kernel — device-only.
///
/// The `s8_accx_depthwise` kernel computes the exact 32-bit accumulators for
/// ONE output pixel (all `out_c` channels) from the raw HWCN weights, into
/// `scratch`; the bit-exact TFLite requantize epilogue runs in Rust. The
/// caller strides over the output image, one kernel call per pixel.
///
/// Depth multipliers > 1 (T3.5): the dispatch stages a *replicated* input —
/// each input channel `i` fanned out to `dm` output channels `i*dm .. i*dm+dm`
/// — so the silicon-proven dm==1 per-lane kernel contract applies directly to
/// the out_c-channel staged vectors; each output channel keeps its own filter
/// row and its own per-channel requantize pair (the `requantize_1x1`
/// epilogue is already per-channel). The dm==1 path (input copied 1:1, no
/// replication) is byte-for-byte unchanged.
///
/// Returns `Ok(true)` when the ACCX path handled the layer, `Ok(false)` when
/// the layer is not eligible (caller falls through to scalar).
#[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
fn depthwise_accx_dispatch(ctx: &mut DepthwiseAccxCtx<'_>) -> Result<bool, KernelError> {
    let params = ctx.params;
    let input_c = params.input_shape[3] as usize;
    let out_c = params.output_shape[3] as usize;
    let in_w = params.input_shape[2] as usize;
    let in_h = params.input_shape[1] as usize;
    let out_h = params.output_shape[1] as usize;
    let out_w = params.output_shape[2] as usize;
    let filter_h = params.filter_shape[1];
    let filter_w = params.filter_shape[2];

    let dilated_filter_h = (filter_h - 1) * params.dilation_height_factor + 1;
    let dilated_filter_w = (filter_w - 1) * params.dilation_width_factor + 1;
    let pad_h = ((params.output_shape[1] - 1) * params.stride_height + dilated_filter_h
        - params.input_shape[1])
        / 2;
    let pad_w = ((params.output_shape[2] - 1) * params.stride_width + dilated_filter_w
        - params.input_shape[2])
        / 2;

    if params.dilation_height_factor != 1
        || params.dilation_width_factor != 1
        || filter_h < 1
        || filter_w < 1
        || !crate::accx::accx_eligible_depthwise_dm(input_c, out_c, params.depth_multiplier)
    {
        return Ok(false);
    }
    // Phase C fold requires the padded fill `-input_offset` to fit in i8:
    // `input_offset` in [-127, 128] (e.g. the kws real depthwise uses
    // input_offset = +128 — its input zero point is -128). The old
    // `abs() > 127` gate wrongly rejected +128.
    if params.input_offset != 0 && (params.input_offset < -127 || params.input_offset > 128) {
        return Ok(false);
    }
    // T3.5b — arbitrary filter sizes: the tap-parameterized
    // `s8_accx_depthwise_anytap` kernel handles any filter_h/filter_w >= 1.
    // The 3x3 shape routes to the unchanged silicon-proven 6-arg kernel.
    let filter_h_u = filter_h.max(1) as usize;
    let filter_w_u = filter_w.max(1) as usize;
    let is_3x3 = filter_h == 3 && filter_w == 3;
    let taps = filter_h_u * filter_w_u;

    let stride_h = params.stride_height.max(1) as usize;
    let stride_w = params.stride_width.max(1) as usize;
    // SAME padding is asymmetric for odd totals: the scalar ref reads
    // `oh*stride - pad + fh` with a bounds-check skip, which is equivalent to
    // `pad_top = total/2` rows of zeros above and `total - pad_top` below. We
    // must pad the staged copy by the FULL total (not 2*(total/2), which drops
    // the odd leftover and would make the kernel read out of bounds).
    let pad_total_h = ((out_h as i32 - 1) * params.stride_height + dilated_filter_h
        - in_h as i32)
        .max(0) as usize;
    let pad_total_w = ((out_w as i32 - 1) * params.stride_width + dilated_filter_w
        - in_w as i32)
        .max(0) as usize;
    let pad_h = (pad_total_h / 2) as usize;
    let pad_w = (pad_total_w / 2) as usize;
    let padded_h = in_h + pad_total_h;
    let padded_w = in_w + pad_total_w;
    // Phase F — non-%16 channels: the kernel VLDs 16-channel vectors and loops
    // `out_c / 16` groups, so we zero-pad the input AND filter channel
    // dimensions up to the next multiple of 16 (same trick as the conv3x3
    // channel padding). Padded channels have zero input and zero weights, so
    // they contribute 0 to every real output channel (depthwise per-lane
    // semantics: output channel oc only sees input channel oc).
    //
    // T3.5 — depth_multiplier > 1: the kernel consumes `out_c`-channel
    // vectors, and the dispatch stages a REPLICATED input (each input channel
    // `i` fanned out to `dm` output channels `i*dm .. i*dm+dm`), so the padded
    // channel count is pad16(out_c) — for dm==1 `out_c == input_c` and this is
    // exactly the historical pad16(input_c). dm>1 always stages (replication
    // cannot run on the caller's in_c-channel input directly).
    let depth_multiplier = params.depth_multiplier.max(1) as usize;
    let dm_gt_1 = depth_multiplier > 1;
    let padded_c = ((out_c + 15) / 16) * 16;
    let needs_channel_pad = padded_c != out_c;
    let needs_pad = pad_total_h > 0 || pad_total_w > 0 || needs_channel_pad || dm_gt_1;

    //   [padded_input: padded_h*padded_w*padded_c]
    //   [padded_filter: taps*padded_c   (only when channel padding)]
    //   [accs: padded_c*4][wsum: out_c*4 (only when input_offset != 0)]
    //   [partials: padded_c*4 (only the anytap/chunked path; the 3x3
    //    silicon-proven kernel writes accs directly and needs no partials)]
    let pad_input_len = padded_h * padded_w * padded_c;
    let pad_filter_len = if needs_channel_pad { taps * padded_c } else { 0 };
    let input_offset = params.input_offset;
    let wsum_extra = if input_offset != 0 { out_c * 4 } else { 0 };
    let partials_extra = if is_3x3 { 0 } else { padded_c * 4 };
    let need = if needs_pad {
        pad_input_len + pad_filter_len + padded_c * 4 + wsum_extra + partials_extra
    } else {
        out_c * 4 + wsum_extra + partials_extra
    };
    if ctx.scratch.len() < need {
        return Ok(false);
    }

    let w_ptr = ctx.weights.as_ptr();
    let out_ptr = ctx.output.as_mut_ptr();
    let mut accs = ctx.scratch.as_mut_ptr() as *mut i32;
    if (w_ptr as usize) % 16 != 0
        || (out_ptr as usize) % 16 != 0
        || (accs as usize) % 4 != 0
    {
        return Ok(false);
    }

    // When padding, carve a zero-filled [padded_h][padded_w][padded_c] input
    // (and padded [tap][padded_c] filter when channel padding) in scratch
    // (16-byte aligned bases) and copy the real interior at (h+pad_h, w+pad_w);
    // the kernel then runs on the padded buffers with stride stepping in the
    // caller pixel loop. When no padding, use the input directly.
    let (k_in_ptr, k_w_ptr, k_pad_w, k_in_c, row_delta);
    let mut wsum: *mut i32 = core::ptr::null_mut();
    let mut partials: *mut i32 = core::ptr::null_mut();
    if needs_pad {
        let scratch_u = ctx.scratch.as_mut_ptr() as usize;
        let in_off = (scratch_u + 15) & !15;
        let w_off = in_off + pad_input_len;
        let accs_off = (w_off + pad_filter_len + 15) & !15;
        let p_in = unsafe { ctx.scratch.as_mut_ptr().add(in_off - scratch_u) };
        let p_w = if needs_channel_pad {
            unsafe { ctx.scratch.as_mut_ptr().add(w_off - scratch_u) }
        } else {
            w_ptr as *mut u8
        };
        let p_accs =
            unsafe { ctx.scratch.as_mut_ptr().add(accs_off - scratch_u) as *mut i32 };
        if input_offset != 0 {
            wsum = unsafe { ctx.scratch.as_mut_ptr().add(accs_off - scratch_u + padded_c * 4) }
                as *mut i32;
        }
        if !is_3x3 {
            // The anytap kernel WRITES per-chunk partials (never adds), so it
            // needs its own 16-aligned i32 buffer after wsum.
            let partials_base = accs_off - scratch_u + padded_c * 4 + wsum_extra;
            partials = unsafe { ctx.scratch.as_mut_ptr().add(partials_base) as *mut i32 };
        }
        // Fill the padded input with `-input_offset` (or 0) so out-of-bounds
        // taps compute `(-off)·w` and the Phase C `+off·Σw` fold cancels them
        // to 0 — matching the scalar ref's bounds-skip semantics. Padded
        // channel slots also get the fill, but their weights are zero so they
        // contribute 0 regardless.
        let fill: u8 = if input_offset != 0 { (-input_offset) as u8 } else { 0 };
        unsafe { core::ptr::write_bytes(p_in, fill, pad_input_len) };
        if dm_gt_1 {
            // T3.5 — fan out each input channel `dm` times so output channel
            // `oc = i*dm + j` reads input channel `i`. Padded channel slots
            // keep the fill; their (padded) weights are zero so they
            // contribute 0 and the Phase C fold cancels their `off·0 = 0`.
            let staged = unsafe { core::slice::from_raw_parts_mut(p_in, pad_input_len) };
            stage_depthwise_pixels(
                staged,
                padded_c,
                padded_w,
                ctx.input,
                in_h,
                in_w,
                input_c,
                depth_multiplier,
                pad_h,
                pad_w,
            );
        } else {
            let src = ctx.input.as_ptr();
            for h in 0..in_h {
                for w in 0..in_w {
                    let srow = unsafe { src.add((h * in_w + w) * input_c) };
                    let drow = unsafe {
                        p_in.add(((h + pad_h) * padded_w + (w + pad_w)) * padded_c) as *mut i8
                    };
                    unsafe { core::ptr::copy_nonoverlapping(srow, drow, input_c) };
                }
            }
        }

        // Zero-fill the padded filter [tap][padded_c], copy real channels.
        if needs_channel_pad {
            unsafe { core::ptr::write_bytes(p_w, 0, pad_filter_len) };
            for tap in 0..taps {
                let src = unsafe { w_ptr.add(tap * out_c) };
                let dst = unsafe { p_w.add(tap * padded_c) as *mut i8 };
                unsafe { core::ptr::copy_nonoverlapping(src, dst, out_c) };
            }
        }

        k_in_ptr = p_in as *const i8;
        k_w_ptr = p_w as *const i8;
        k_pad_w = padded_w;
        k_in_c = padded_c;
        row_delta = if padded_w >= filter_w_u {
            (padded_w - filter_w_u) * padded_c
        } else {
            0
        };
        accs = p_accs;
    } else {
        let in_ptr = ctx.input.as_ptr();
        if (in_ptr as usize) % 16 != 0 {
            return Ok(false);
        }
        k_in_ptr = in_ptr;
        k_w_ptr = w_ptr;
        k_pad_w = in_w;
        k_in_c = input_c;
        row_delta = if in_w >= filter_w_u {
            (in_w - filter_w_u) * input_c
        } else {
            0
        };
        if input_offset != 0 {
            wsum = unsafe { accs.add(out_c) };
        }
        if !is_3x3 {
            partials = unsafe { accs.add(out_c + wsum_extra / 4) };
        }
    }

    // Depthwise filter is [tap][oc] (HWCN); wsum[oc] = Σ_tap w[tap·out_c + oc].
    // The stride is `k_in_c` (== padded_c when channel-padded, == out_c on the
    // raw [tap][out_c] filter otherwise) so the staged layout is read correctly.
    if input_offset != 0 {
        let ws = unsafe { core::slice::from_raw_parts_mut(wsum, out_c) };
        let wv = unsafe { core::slice::from_raw_parts(k_w_ptr, taps * k_in_c) };
        crate::accx::weight_sums_depthwise(ws, wv, k_in_c);
    }

    let multipliers = params.output_multiplier_per_channel;
    let shifts = params.output_shift_per_channel;
    let act_min = params.quantized_activation_min;
    let act_max = params.quantized_activation_max;
    let out_offset = params.output_offset;
    let (uniform_mult, uniform_shift) = match crate::accx::uniform_scale(multipliers, shifts) {
        Some((m, s)) => (m, s),
        None => (0, i32::MIN),
    };

    for oh in 0..out_h {
        for ow in 0..out_w {
            let px = (oh * stride_h * k_pad_w + ow * stride_w) * k_in_c;
            let po = (oh * out_w + ow) * out_c;
            if is_3x3 {
                // Silicon-proven 3x3 path (unchanged).
                unsafe {
                    crate::accx::accx_depthwise(
                        k_in_ptr.add(px),
                        k_w_ptr,
                        accs,
                        k_in_c,
                        k_in_c,
                        row_delta,
                    );
                }
            } else {
                // T3.5b — arbitrary filter: chunked QACC passes. Each pass
                // covers ≤ 32 taps (the QACC 20-bit-lane-safe bound: 32 taps
                // of 127² = 516,128 < the ±524,287 signed-lane limit); the
                // kernel writes per-chunk partials which we add into the
                // running i32 accs (i32 wrapping adds — mathematically
                // identical to one 32-bit sum). Chunk boundaries may fall
                // mid-row, so each chunk starts at (row, col) from tap_start.
                unsafe { core::ptr::write_bytes(accs as *mut u8, 0, padded_c * 4) };
                let mut tap_start = 0;
                while tap_start < taps {
                    let chunk_taps = (taps - tap_start).min(32);
                    let row = tap_start / filter_w_u;
                    let col = tap_start % filter_w_u;
                    // SAFETY: the window base `px` plus `(row, col)` stays
                    // within the staged/raw input (`padded_h×padded_w`), and
                    // `filter_w*k_in_c + row_delta` is the exact row stride.
                    let in_ptr = unsafe {
                        k_in_ptr.add(
                            px + (row * (filter_w_u * k_in_c + row_delta) + col * k_in_c),
                        )
                    };
                    // SAFETY: the filter is `[taps][k_in_c]`; `tap_start < taps`.
                    let w_ptr = unsafe { k_w_ptr.add(tap_start * k_in_c) };
                    unsafe {
                        crate::accx::accx_depthwise_anytap(&crate::accx::AnyTapCtx {
                            input: in_ptr,
                            filter: w_ptr,
                            acc_out: partials,
                            in_c: k_in_c as u32,
                            out_c: k_in_c as u32,
                            row_delta: row_delta as u32,
                            taps: chunk_taps as u32,
                            filter_w: filter_w_u as u32,
                            col_start: col as u32,
                        });
                    }
                    let acc_slice = unsafe { core::slice::from_raw_parts_mut(accs, padded_c) };
                    let part_slice = unsafe { core::slice::from_raw_parts(partials, padded_c) };
                    for i in 0..padded_c {
                        acc_slice[i] = acc_slice[i].wrapping_add(part_slice[i]);
                    }
                    tap_start += 32;
                }
            }
            if input_offset != 0 {
                let acc_slice = unsafe { core::slice::from_raw_parts_mut(accs, out_c) };
                let ws = unsafe { core::slice::from_raw_parts(wsum, out_c) };
                fold_input_offset(acc_slice, ws, input_offset);
            }
            let acc_slice = unsafe { core::slice::from_raw_parts_mut(accs, out_c) };
            crate::accx::requantize_1x1(&mut crate::accx::ReqCtx {
                accs: acc_slice,
                bias: ctx.bias,
                multipliers,
                shifts,
                output_offset: out_offset,
                act_min,
                act_max,
                out_base: po,
                output: ctx.output,
                uniform_mult,
                uniform_shift,
            });
        }
    }
    Ok(true)
}

/// Prepared depthwise handle — runs the SIMD gate ONCE at construction, then
/// `run` only re-checks pointer alignment and dispatches.
///
/// The bespoke QACC kernel (`s8_accx_depthwise`) computes exact 32-bit
/// per-lane accumulators, so SIMD output is bit-exact vs the scalar reference.
pub struct PreparedDepthwise {
    /// Whether the bespoke QACC SIMD kernel is eligible on this target.
    accx: bool,
    params: &'static DepthwiseConv2DParams<'static>,
}

impl PreparedDepthwise {
    pub fn new(params: &'static DepthwiseConv2DParams<'static>) -> Result<Self, KernelError> {
        let input_c = params.input_shape[3] as usize;
        let out_channels = params.output_shape[3] as usize;
        let accx = cfg!(all(target_arch = "xtensa", not(feature = "qemu")))
            && crate::accx::accx_eligible_depthwise_dm(input_c, out_channels, params.depth_multiplier);
        Ok(Self { accx, params })
    }

    #[inline]
    pub fn is_simd(&self) -> bool {
        self.accx
    }

    pub fn run(
        &self,
        input: &[i8],
        weights: &[i8],
        bias: &[i32],
        output: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
        {
            let mut accx_ctx = DepthwiseAccxCtx {
                input,
                weights,
                bias,
                params: self.params,
                output,
                scratch,
            };
            if depthwise_accx_dispatch(&mut accx_ctx)? {
                return Ok(());
            }
        }
        depthwise_conv2d(input, weights, bias, self.params, output, scratch)
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;
    use super::*;
    use hematite_core::op_params::{DepthwiseConv2DParams, Padding};

    /// Host model of the `s8_accx_depthwise[_anytap]` per-lane accumulation
    /// contract on the staged buffers (T3.5 + T3.5b): `acc[oc] = Σ_tap
    /// staged_w[tap*padded_c + oc] * staged_in[base + tap*padded_c +
    /// (tap/filter_w)*row_delta + oc]` — the exact QACC lane addressing the
    /// asm uses (input advances `padded_c` per tap, `row_delta` per row —
    /// i.e. every `filter_w` taps; filter advances `padded_c` per tap). The
    /// device chunking (≤32-tap QACC passes + i32 adds) is mathematically
    /// identical to this single i32 sum, so this model is the correct oracle.
    fn kernel_model_accs(
        staged_in: &[u8],
        staged_w: &[i8],
        padded_c: usize,
        row_delta: usize,
        base: usize,
        out_c: usize,
        filter_w: usize,
    ) -> Vec<i32> {
        let mut accs = vec![0i32; out_c];
        let taps = staged_w.len() / padded_c;
        for tap in 0..taps {
            let in_off = tap * padded_c + (tap / filter_w) * row_delta;
            for oc in 0..out_c {
                let iv = i32::from(staged_in[base + in_off + oc] as i8);
                let wv = i32::from(staged_w[tap * padded_c + oc]);
                accs[oc] = accs[oc].wrapping_add(iv.wrapping_mul(wv));
            }
        }
        accs
    }

    /// Run the full device SIMD pipeline in software — real
    /// `stage_depthwise_pixels` staging, the kernel-model accumulators, the
    /// real `fold_input_offset`, and the real `requantize_1x1` epilogue —
    /// producing one output layer.
    fn simd_model_layer(
        input: &[i8],
        weights: &[i8],
        bias: &[i32],
        p: &DepthwiseConv2DParams<'_>,
    ) -> Vec<i8> {
        let in_h = p.input_shape[1] as usize;
        let in_w = p.input_shape[2] as usize;
        let in_c = p.input_shape[3] as usize;
        let out_h = p.output_shape[1] as usize;
        let out_w = p.output_shape[2] as usize;
        let out_c = p.output_shape[3] as usize;
        let dm = p.depth_multiplier.max(1) as usize;
        let stride_h = p.stride_height.max(1) as usize;
        let stride_w = p.stride_width.max(1) as usize;
        let filter_h = p.filter_shape[1].max(1) as usize;
        let filter_w = p.filter_shape[2].max(1) as usize;
        let taps = filter_h * filter_w;

        let dilated_h = (p.filter_shape[1] - 1) * p.dilation_height_factor + 1;
        let dilated_w = (p.filter_shape[2] - 1) * p.dilation_width_factor + 1;
        let pad_total_h = ((out_h as i32 - 1) * p.stride_height + dilated_h - in_h as i32)
            .max(0) as usize;
        let pad_total_w = ((out_w as i32 - 1) * p.stride_width + dilated_w - in_w as i32)
            .max(0) as usize;
        let pad_h = pad_total_h / 2;
        let pad_w = pad_total_w / 2;
        let padded_h = in_h + pad_total_h;
        let padded_w = in_w + pad_total_w;
        let padded_c = ((out_c + 15) / 16) * 16;
        let row_delta = if padded_w >= filter_w {
            (padded_w - filter_w) * padded_c
        } else {
            0
        };

        // Staged (replicated + padded) input, pre-filled like the dispatch.
        let fill: u8 = if p.input_offset != 0 { (-p.input_offset) as u8 } else { 0 };
        let mut staged_in = vec![fill; padded_h * padded_w * padded_c];
        stage_depthwise_pixels(
            &mut staged_in,
            padded_c,
            padded_w,
            input,
            in_h,
            in_w,
            in_c,
            dm,
            pad_h,
            pad_w,
        );

        // Staged filter [tap][padded_c]; zero-fill padded channels.
        let mut staged_w = vec![0i8; taps * padded_c];
        for tap in 0..taps {
            staged_w[tap * padded_c..tap * padded_c + out_c]
                .copy_from_slice(&weights[tap * out_c..(tap + 1) * out_c]);
        }

        // Per-channel weight sums (Phase C fold).
        let mut wsum = vec![0i32; out_c];
        crate::accx::weight_sums_depthwise(&mut wsum, &staged_w, padded_c);

        let multipliers = p.output_multiplier_per_channel;
        let shifts = p.output_shift_per_channel;
        let (uniform_mult, uniform_shift) = match crate::accx::uniform_scale(multipliers, shifts) {
            Some((m, s)) => (m, s),
            None => (0, i32::MIN),
        };
        let mut output = vec![0i8; out_h * out_w * out_c];
        for oh in 0..out_h {
            for ow in 0..out_w {
                let base = (oh * stride_h * padded_w + ow * stride_w) * padded_c;
                let mut accs =
                    kernel_model_accs(&staged_in, &staged_w, padded_c, row_delta, base, out_c, filter_w);
                if p.input_offset != 0 {
                    fold_input_offset(&mut accs, &wsum, p.input_offset);
                }
                crate::accx::requantize_1x1(&mut crate::accx::ReqCtx {
                    accs: &accs,
                    bias,
                    multipliers,
                    shifts,
                    output_offset: p.output_offset,
                    act_min: p.quantized_activation_min,
                    act_max: p.quantized_activation_max,
                    out_base: (oh * out_w + ow) * out_c,
                    output: &mut output,
                    uniform_mult,
                    uniform_shift,
                });
            }
        }
        output
    }

    fn per_channel_mult(n: usize) -> Vec<i32> {
        (0..n).map(|i| (1 << 30) - (i as i32) * 7919).collect()
    }

    fn per_channel_shift(n: usize) -> Vec<i32> {
        (0..n).map(|i| (i % 3) as i32).collect()
    }

    /// Deterministic pseudo-random `i8` pattern (full int8 range).
    fn pattern(seed: u32, n: usize) -> Vec<i8> {
        let mut out = vec![0i8; n];
        let mut x = seed;
        for v in out.iter_mut() {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *v = (x >> 16) as i8;
        }
        out
    }

    fn pattern_i32(seed: u32, n: usize) -> Vec<i32> {
        let mut out = vec![0i32; n];
        let mut x = seed;
        for v in out.iter_mut() {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *v = ((x >> 16) as i32) * 37 - 500;
        }
        out
    }

    /// Host bit-exact gate (T3.5): the device SIMD pipeline model must equal
    /// the independent `hematite-ref` scalar depthwise for every
    /// dm {2,4,8} × 3×3 × in_c {1,3,8,16,32} fan-out shape, across offsets and
    /// identity / non-identity per-channel multipliers — zero mismatches.
    #[test]
    fn depthwise_dm_gt1_simd_model_matches_ref_bit_exact() {
        let mut checked = 0;
        for &dm in &[2, 4, 8] {
            for &in_c in &[1, 3, 8, 16, 32] {
                let out_c = (in_c * dm) as i32;
                for (spatial, offsets) in [(&[8, 12][..], &[0, -3][..]), (&[7][..], &[5, 0][..])] {
                    for &sp in spatial {
                        for &in_off in offsets {
                            // Case A: identity uniform scale (1<<30, shift 1).
                            // Case B: non-identity per-channel mult/shift.
                            // Case C: uniform hoisted (1<<29, shift 0).
                            for mode in 0..3 {
                                let n = out_c as usize;
                                let (mults, shifts): (Vec<i32>, Vec<i32>) = match mode {
                                    0 => (vec![1 << 30; n], vec![1; n]),
                                    1 => (per_channel_mult(n), per_channel_shift(n)),
                                    _ => (vec![1 << 29; n], vec![0; n]),
                                };
                                let p = DepthwiseConv2DParams {
                                    input_shape: [1, sp, sp, in_c as i32],
                                    filter_shape: [1, 3, 3, out_c],
                                    output_shape: [1, sp, sp, out_c],
                                    padding: Padding::Same,
                                    stride_width: 1,
                                    stride_height: 1,
                                    dilation_width_factor: 1,
                                    dilation_height_factor: 1,
                                    depth_multiplier: dm,
                                    input_offset: in_off,
                                    weights_offset: 0,
                                    output_offset: if in_off == 0 { 0 } else { -10 },
                                    output_multiplier_per_channel: &mults,
                                    output_shift_per_channel: &shifts,
                                    quantized_activation_min: if mode == 1 { 0 } else { -128 },
                                    quantized_activation_max: 127,
                                };
                                let in_len = (sp * sp) as usize * in_c as usize;
                                let w_len = 9 * out_c as usize;
                                let seed = 0x51E5_0000u32 | (dm as u32 * 101 + in_c as u32);
                                let input = pattern(seed, in_len);
                                let weights = pattern(0xBEE5 + in_c as u32 * 17, w_len);
                                let bias = pattern_i32(0xBAD + out_c as u32, out_c as usize);

                                let got = simd_model_layer(&input, &weights, &bias, &p);
                                let mut want = vec![0i8; got.len()];
                                hematite_ref::depthwise_conv::depthwise_conv2d(
                                    &input,
                                    &weights,
                                    &bias,
                                    &p,
                                    &mut want,
                                    &mut [],
                                )
                                .expect("ref depthwise accepts the shape");
                                assert_eq!(
                                    got, want,
                                    "dm={dm} in_c={in_c} sp={sp} in_off={in_off} mode={mode}: \
                                     SIMD-model output must equal hematite-ref scalar"
                                );
                                checked += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(checked >= 100, "dm>1 matrix did not expand ({checked})");
    }

    /// T3.5b — the tap-parameterized anytap SIMD pipeline must equal the
    /// independent `hematite-ref` scalar depthwise for arbitrary filter sizes
    /// and arbitrary dm, across strides, padding modes, and input offsets —
    /// zero mismatches. The device chunking (≤32-tap QACC passes + i32 adds)
    /// is mathematically identical to the single-pass model, so this is the
    /// bit-exact oracle for the kws 10×8 80-tap filter.
    #[test]
    fn depthwise_anyfilter_simd_model_matches_ref_bit_exact() {
        let mut checked = 0;
        let filters: &[(i32, i32)] = &[(3, 3), (5, 5), (7, 7), (10, 8)];
        for &(fh, fw) in filters {
            for &dm in &[1, 2, 8] {
                for &in_c in &[1, 3, 8, 16] {
                    let out_c = (in_c * dm) as i32;
                    for &(sp, stride) in &[(12i32, 1i32), (14, 2)] {
                        for &pad in &[Padding::Valid, Padding::Same] {
                            let (out_h, out_w) = match pad {
                                Padding::Same => (
                                    (sp + stride - 1) / stride,
                                    (sp + stride - 1) / stride,
                                ),
                                Padding::Valid => (
                                    (sp - fh) / stride + 1,
                                    (sp - fw) / stride + 1,
                                ),
                            };
                            if out_h < 1 || out_w < 1 {
                                continue;
                            }
                            for &in_off in &[0, 3, 5, 128] {
                                for mode in 0..2 {
                                    let n = out_c as usize;
                                    let (mults, shifts): (Vec<i32>, Vec<i32>) = match mode {
                                        0 => (vec![1 << 30; n], vec![1; n]),
                                        _ => (per_channel_mult(n), per_channel_shift(n)),
                                    };
                                    let p = DepthwiseConv2DParams {
                                        input_shape: [1, sp, sp, in_c as i32],
                                        filter_shape: [1, fh, fw, out_c],
                                        output_shape: [1, out_h, out_w, out_c],
                                        padding: pad,
                                        stride_width: stride,
                                        stride_height: stride,
                                        dilation_width_factor: 1,
                                        dilation_height_factor: 1,
                                        depth_multiplier: dm,
                                        input_offset: in_off,
                                        weights_offset: 0,
                                        output_offset: if in_off == 0 { 0 } else { -10 },
                                        output_multiplier_per_channel: &mults,
                                        output_shift_per_channel: &shifts,
                                        quantized_activation_min: if mode == 1 { 0 } else { -128 },
                                        quantized_activation_max: 127,
                                    };
                                    let in_len = (sp * sp) as usize * in_c as usize;
                                    let w_len = (fh * fw) as usize * out_c as usize;
                                    let seed = 0x51E5_5B00u32 | (dm as u32 * 101 + in_c as u32);
                                    let input = pattern(seed, in_len);
                                    let weights = pattern(0x0A77 + in_c as u32 * 17 + fh as u32, w_len);
                                    let bias = pattern_i32(0xBAD + out_c as u32, out_c as usize);

                                    let got = simd_model_layer(&input, &weights, &bias, &p);
                                    let mut want = vec![0i8; got.len()];
                                    hematite_ref::depthwise_conv::depthwise_conv2d(
                                        &input,
                                        &weights,
                                        &bias,
                                        &p,
                                        &mut want,
                                        &mut [],
                                    )
                                    .expect("ref depthwise accepts the shape");
                                    assert_eq!(
                                        got, want,
                                        "fh={fh} fw={fw} dm={dm} in_c={in_c} sp={sp} \
                                         stride={stride} pad={pad:?} in_off={in_off} mode={mode}: \
                                         SIMD-model output must equal hematite-ref scalar"
                                    );
                                    checked += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(checked >= 500, "arbitrary-filter matrix did not expand ({checked})");
    }

    /// The replication staging must produce the exact TFLM fan-out: output
    /// channel `oc = i*dm + j` gets input channel `i` (all `dm` copies equal).
    #[test]
    fn stage_depthwise_pixels_replicates_fan_out() {
        let src: Vec<i8> = (0..4).map(|i| i as i8 - 2).collect(); // in_c = 4 pixels 1x1
        let mut dst = vec![0xEEu8; 8]; // dm = 2, dst_c = 8
        stage_depthwise_pixels(&mut dst, 8, 1, &src, 1, 1, 4, 2, 0, 0);
        assert_eq!(
            dst,
            vec![
                (0u8).wrapping_sub(2),
                (0u8).wrapping_sub(2),
                (1u8).wrapping_sub(2),
                (1u8).wrapping_sub(2),
                (2u8).wrapping_sub(2),
                (2u8).wrapping_sub(2),
                (3u8).wrapping_sub(2),
                (3u8).wrapping_sub(2),
            ]
        );
    }
}
