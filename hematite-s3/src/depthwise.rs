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
    // SAFETY of the `get_unchecked` calls: slice lengths were validated
    // above (input/weights/bias/output against shape_product + channel
    // cross-checks), and the in-bounds guard below guarantees `input_idx`
    // and `filter_idx` point into the validated ranges.
    let input_offset = params.input_offset;
    for oh in 0..out_h {
        let input_base_h = oh * params.stride_height - pad_h;

        for ow in 0..out_w {
            let input_base_w = ow * params.stride_width - pad_w;

            for ic in 0..input_c {
                for dm in 0..depth_multiplier {
                    let oc = dm + ic * depth_multiplier;
                    let mut acc: i32 = unsafe { *bias.get_unchecked(oc as usize) };

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

                                let i_val = i32::from(unsafe {
                                    *input.get_unchecked(input_idx)
                                });
                                let w_val = i32::from(unsafe {
                                    *weights.get_unchecked(filter_idx)
                                });

                                acc += (i_val + input_offset) * w_val;
                            }
                            // else: zero-padding — skip (contribute 0 to accumulator)
                        }
                    }

                    // Per-channel requantize + output offset + clamp
                    let multiplier = unsafe { *multipliers.get_unchecked(oc as usize) };
                    let shift = unsafe { *shifts.get_unchecked(oc as usize) };
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

/// Context for the bespoke QACC depthwise dispatch — bundled into one `&mut`
/// arg so the Xtensa LLVM backend generates a 1-arg call (multi-arg calls are
/// miscompiled on device; see the `dispatch_fc` inline regression and
/// `ReqCtx`).
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
        || filter_h != 3
        || filter_w != 3
        || !crate::accx::accx_eligible_depthwise(input_c, out_c)
    {
        return Ok(false);
    }
    // `dm > 1` (e.g. kws's ch_mult=8) is supported by broadcasting every input
    // channel `ic` to the `dm` output slots `[ic·dm, (ic+1)·dm)` in the padded
    // virtual input; the QACC kernel's `in_c == out_c` per-lane contract then
    // holds over the `out_c` virtual channels.
    let dm = params.depth_multiplier.max(1) as usize;
    if input_c.checked_mul(dm) != Some(out_c) {
        return Ok(false);
    }
    // Phase C fold requires the padded fill `-input_offset` to fit in i8.
    // `-input_offset` is representable for input_offset in [-127, 128] (the
    // common TFLite input_zero_point=-128 gives input_offset=128, whose
    // negation -128 = 0x80 fits i8). input_offset=-128 would need +128 and is
    // rejected.
    if params.input_offset < -127 || params.input_offset > 128 {
        return Ok(false);
    }

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
    // they contribute 0 to every real output channel. For `dm > 1` the virtual
    // input has `out_c` channels (each real channel broadcast `dm` times), so
    // the padding rounds `out_c`, not `input_c`.
    let padded_c = ((out_c + 15) / 16) * 16;
    let needs_channel_pad = padded_c != out_c;
    // dm>1 always stages the broadcast virtual input (never uses the raw
    // input directly), even when out_c is already a multiple of 16.
    let needs_broadcast = dm > 1;
    let needs_pad = pad_total_h > 0 || pad_total_w > 0 || needs_channel_pad || needs_broadcast;

    //   [padded_input: padded_h*padded_w*padded_c]
    //   [padded_filter: 9*padded_c   (only when channel padding)]
    //   [accs: padded_c*4][wsum: out_c*4 (only when input_offset != 0)]
    let pad_input_len = padded_h * padded_w * padded_c;
    let pad_filter_len = if needs_channel_pad { 9 * padded_c } else { 0 };
    let input_offset = params.input_offset;
    let wsum_extra = if input_offset != 0 { out_c * 4 } else { 0 };
    let need = if needs_pad {
        pad_input_len + pad_filter_len + padded_c * 4 + wsum_extra
    } else {
        out_c * 4 + wsum_extra
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
        // Fill the padded input with `-input_offset` (or 0) so out-of-bounds
        // taps compute `(-off)·w` and the Phase C `+off·Σw` fold cancels them
        // to 0 — matching the scalar ref's bounds-skip semantics. Padded
        // channel slots also get the fill, but their weights are zero so they
        // contribute 0 regardless.
        let fill: u8 = if input_offset != 0 { (-input_offset) as u8 } else { 0 };
        unsafe { core::ptr::write_bytes(p_in, fill, pad_input_len) };
        let src = ctx.input.as_ptr();
        for h in 0..in_h {
            for w in 0..in_w {
                let srow = unsafe { src.add((h * in_w + w) * input_c) };
                let drow = unsafe {
                    p_in.add(((h + pad_h) * padded_w + (w + pad_w)) * padded_c) as *mut i8
                };
                if dm == 1 {
                    unsafe { core::ptr::copy_nonoverlapping(srow, drow, input_c) };
                } else {
                    // Broadcast: virtual channel `ic·dm + d` <- real channel `ic`.
                    for ic in 0..input_c {
                        let v = unsafe { *srow.add(ic) };
                        let base = ic * dm;
                        for d in 0..dm {
                            unsafe { *drow.add(base + d) = v };
                        }
                    }
                }
            }
        }

        // Zero-fill the padded filter [tap][padded_c], copy real channels.
        if needs_channel_pad {
            unsafe { core::ptr::write_bytes(p_w, 0, pad_filter_len) };
            for tap in 0..9 {
                let src = unsafe { w_ptr.add(tap * out_c) };
                let dst = unsafe { p_w.add(tap * padded_c) as *mut i8 };
                unsafe { core::ptr::copy_nonoverlapping(src, dst, out_c) };
            }
        }

        k_in_ptr = p_in as *const i8;
        k_w_ptr = p_w as *const i8;
        k_pad_w = padded_w;
        k_in_c = padded_c;
        row_delta = if padded_w >= 3 { (padded_w - 3) * padded_c } else { 0 };
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
        row_delta = if in_w >= 3 { (in_w - 3) * input_c } else { 0 };
        if input_offset != 0 {
            wsum = unsafe { accs.add(out_c) };
        }
    }

    // Depthwise filter is [tap][oc] (HWCN); wsum[oc] = Σ_tap w[tap·out_c + oc].
    // Computed from the RAW weights (stride `out_c`) so channel padding never
    // changes the stride `weight_sums_depthwise` steps by (the padded copy is
    // `[tap][padded_c]`).
    if input_offset != 0 {
        let ws = unsafe { core::slice::from_raw_parts_mut(wsum, out_c) };
        let wv = unsafe { core::slice::from_raw_parts(w_ptr, 9 * out_c) };
        crate::accx::weight_sums_depthwise(ws, wv, out_c);
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
            if input_offset != 0 {
                for oc in 0..out_c {
                    let v = unsafe { accs.add(oc).read() };
                    let s = unsafe { wsum.add(oc).read() };
                    unsafe { accs.add(oc).write(v.wrapping_add(input_offset.wrapping_mul(s))) };
                }
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
            && crate::accx::accx_eligible_depthwise(input_c, out_channels);
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
