// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Conv2D 3×3 kernel — scalar fallback + TIE728 SIMD backend.
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
//! # Layouts
//!
//! * `input` — NHWC `[batch=1, H, W, Cin]`
//! * `weights` — OHWI `[Cout, FH, FW, Cin]`
//! * `bias` — per-output-channel `[Cout]`
//! * `output` — NHWC `[batch=1, OH, OW, Cout]`
//!
//! The scalar kernel is a general 2D convolution identical in structure to the
//! conv1x1 module — the same code handles any filter size. The SIMD backend
//! routes to `dl_tie728_s8_conv2d_33cn` for the 3×3 fast path (hardcoded
//! 9-MAC-unrolled inner loop in the vendored asm).

use hematite_core::op_params::Conv2DParams;
use hematite_core::KernelError;
use hematite_int8::{multiply_by_quantized_multiplier, saturating_cast};

/// Shared TIE728 SIMD eligibility gate for the 3×3 conv — host-compilable.
///
/// Returns
/// `Some((mac_shift, ocd8, c_div_x_1, dilation_x_offset, dilation_y_offset, use_relu))`
/// when the params qualify for `dl_tie728_s8_conv2d_33cn`, `None` otherwise.
/// Single source of truth for both the legacy `conv2d_3x3` dispatch and the
/// [`PreparedConv3x3`] handle.
#[inline]
pub(crate) fn simd_eligible_conv3x3(
    params: &Conv2DParams<'_>,
    input_c: i32,
    out_channels: i32,
    filter_h: i32,
    filter_w: i32,
    pad_h: i32,
    pad_w: i32,
) -> Option<(i32, i32, i32, i32, i32, bool)> {
    let mult = params.output_multiplier_per_channel;
    let shift = params.output_shift_per_channel;
    let mult_uniform = !mult.is_empty() && mult.iter().all(|&m| m == mult[0]) && mult[0] == 1 << 30;
    let shift_uniform = !shift.is_empty() && shift.iter().all(|&s| s == shift[0]);
    let full_range = params.quantized_activation_min == i8::MIN as i32
        && params.quantized_activation_max == i8::MAX as i32;
    let relu_range = params.quantized_activation_min == 0
        && params.quantized_activation_max == i8::MAX as i32;

    if params.input_offset == 0
        && params.output_offset == 0
        && (full_range || relu_range)
        && mult_uniform
        && shift_uniform
        && shift[0] <= 1
        && input_c % 16 == 0
        && input_c >= 16
        && out_channels % 16 == 0
        && pad_h == 0
        && pad_w == 0
        && filter_h == 3
        && filter_w == 3
    {
        let use_relu = relu_range && !full_range;
        // mac_shift: same correction as conv1x1 — for mult==1<<30 the scalar
        // reference computes (acc+1)>>1 == asm round_result(acc, mac_shift)
        // requires mac_shift == 1 - shift[0].
        //
        // dilation offsets: per the 33c16 asm comment (dl_tie728_s8_conv2d.S
        // lines 1240-1241), with no padding (pad_h==pad_w==0):
        //   dil_x = (dilation_x * in_c - in_c) * sizeof(out)
        //   dil_y = (dilation_y * in_w * in_c - in_c - dilation_x * in_c * (filter_w-1)) * sizeof(out)
        // sizeof(out)=1 for i8, in_c_pad==in_c, in_w_pad==in_w. The 11c16 tap
        // already advances input by in_c bytes, so dil_x==0 for dilation_x==1
        // and dil_y jumps the tap down to the next row. (The previous code
        // passed dilation*stride — WRONG; e.g. 32x32x64 stride1 needs dil_x=0,
        // dil_y=1856.)
        let input_w = params.input_shape[2];
        let dilation_x_offset = (params.dilation_width_factor - 1) * input_c;
        let dilation_y_offset = (params.dilation_height_factor * input_w
            - 1
            - params.dilation_width_factor * (filter_w - 1))
            * input_c;
        Some((
            1 - shift[0],
            out_channels / 16,
            input_c / 16 - 1,
            dilation_x_offset,
            dilation_y_offset,
            use_relu,
        ))
    } else {
        None
    }
}

/// Transform weights from the caller's `[oc][fh][fw][ic]` (OHWI) layout into
/// the TIE728 33cn `[g][tap][ic][lane]` SIMD layout.
///
/// The vendored `dl_tie728_s8_conv2d_33cn` computes **one output pixel per
/// call**: per 16-output-channel group `g` it runs the 33c16 macro, which
/// executes the 11c16 inner loop NINE times (one per 3×3 tap, order
/// (0,0),(0,1),(0,2),(1,0),…,(2,2)), each consuming `in_c * 16` filter bytes.
/// So the asm reads `filter[g*(9*in_c*16) + tap*(in_c*16) + ic*16 + lane]`,
/// which must equal `src[(g*16+lane)*(9*in_c) + tap*in_c + ic]` for the SIMD
/// output to match the scalar reference bit-exact. Pure no_std permutation
/// (host-compilable, unit-tested).
///
/// `src` length must be `out_channels * 9 * input_c`; `dst` must match.
/// Returns [`KernelError::ShapeMismatch`] otherwise.
pub fn transform_weights_33cn(
    input_c: usize,
    out_channels: usize,
    src: &[i8],
    dst: &mut [i8],
) -> Result<(), KernelError> {
    if input_c % 16 != 0 || out_channels % 16 != 0 {
        return Err(KernelError::ShapeMismatch);
    }
    let taps = 9;
    if src.len() != out_channels * taps * input_c || dst.len() != src.len() {
        return Err(KernelError::ShapeMismatch);
    }
    let groups = out_channels / 16;
    let per_group = taps * input_c * 16;
    for g in 0..groups {
        let g_base = g * per_group;
        for tap in 0..taps {
            let tap_base = g_base + tap * input_c * 16;
            for ic in 0..input_c {
                let ic_base = tap_base + ic * 16;
                for lane in 0..16 {
                    let src_oc = g * 16 + lane;
                    dst[ic_base + lane] = src[src_oc * (taps * input_c) + tap * input_c + ic];
                }
            }
        }
    }
    Ok(())
}


/// Context for the ACCX 3×3 conv dispatch — bundled into one `&mut` arg so the
/// Xtensa LLVM backend generates a 1-arg call (multi-arg calls are miscompiled
/// on device; see the `dispatch_fc` inline regression and `ReqCtx`).
#[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
pub(crate) struct Conv3x3AccxCtx<'a> {
    pub input: &'a [i8],
    pub weights: &'a [i8],
    pub bias: &'a [i32],
    pub params: &'a Conv2DParams<'a>,
    pub output: &'a mut [i8],
    pub scratch: &'a mut [u8],
}

/// ACCX SIMD dispatch for the 3×3 conv kernel — device-only.
///
/// The bespoke `s8_accx_conv3x3` kernel computes the exact 32-bit dot product
/// for ONE output pixel (all out_c channels) from the raw `[oc][fh][fw][ic]`
/// weights, into `scratch`; the bit-exact TFLite requantize epilogue runs in
/// Rust. The caller strides over the output image, one kernel call per pixel.
///
/// Returns `Ok(true)` when the ACCX path handled the layer, `Ok(false)` when
/// the layer is not ACCX-eligible (caller falls through to scalar).
#[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
fn conv3x3_accx_dispatch(ctx: &mut Conv3x3AccxCtx<'_>) -> Result<bool, KernelError> {
    let params = ctx.params;
    let input_c = params.input_shape[3] as usize;
    let out_c = params.output_shape[3] as usize;
    let in_h = params.input_shape[1] as usize;
    let in_w = params.input_shape[2] as usize;
    let out_h = params.output_shape[1] as usize;
    let out_w = params.output_shape[2] as usize;
    let filter_h = params.filter_shape[1];
    let filter_w = params.filter_shape[2];

    let dilated_filter_h = (filter_h - 1) * params.dilation_height_factor + 1;
    let dilated_filter_w = (filter_w - 1) * params.dilation_width_factor + 1;

    if params.dilation_height_factor != 1
        || params.dilation_width_factor != 1
        || filter_h != 3
        || filter_w != 3
        || !crate::accx::accx_eligible_3x3(input_c, out_c)
    {
        return Ok(false);
    }
    // Phase C fold requires the padded fill `-input_offset` to fit in i8.
    if params.input_offset != 0 && params.input_offset.abs() > 127 {
        return Ok(false);
    }

    // Phase A — SAME-padding + first-conv zero-padding. The ACCX kernel VLDs
    // 16-channel vectors and a 3×3 window with `row_delta` row strides, so we
    // stage a padded copy of the input in scratch whenever the layer needs
    // spatial padding (`pad_h`/`pad_w` > 0, derived from SAME semantics) or a
    // channel multiple of 16. Padded regions are all-zero, so the dot
    // products are bit-identical to the scalar conv (which skips them).
    // SAME padding is asymmetric for odd totals (see conv3x3_accx_dispatch):
    // pad_top = total/2, padded = in + total.
    let pad_total_h = ((out_h as i32 - 1) * params.stride_height + dilated_filter_h
        - in_h as i32)
        .max(0) as usize;
    let pad_total_w = ((out_w as i32 - 1) * params.stride_width + dilated_filter_w
        - in_w as i32)
        .max(0) as usize;
    let pad_h = (pad_total_h / 2) as usize;
    let pad_w = (pad_total_w / 2) as usize;
    let padded_c = ((input_c + 15) / 16) * 16;
    let padded_h = in_h + pad_total_h;
    let padded_w = in_w + pad_total_w;
    let needs_pad = pad_total_h > 0 || pad_total_w > 0 || padded_c != input_c;

    // Scratch layout when padding:
    //   [padded_input: padded_h*padded_w*padded_c][padded_weights: out_c*9*padded_c][accs: out_c*4]
    // When input_offset != 0 we also need a weight-sum buffer (out_c*4) — see
    // the fold below — carved right after `accs`.
    let pad_input_len = padded_h * padded_w * padded_c;
    let pad_weights_len = out_c * 9 * padded_c;
    let wsum_extra = if params.input_offset != 0 { out_c * 4 } else { 0 };
    let need = if needs_pad {
        pad_input_len + pad_weights_len + out_c * 4 + wsum_extra
    } else {
        out_c * 4 + wsum_extra
    };
    if ctx.scratch.len() < need {
        return Ok(false);
    }

    let in_ptr = ctx.input.as_ptr();
    let w_ptr = ctx.weights.as_ptr();
    let out_ptr = ctx.output.as_mut_ptr();
    let scratch_ptr = ctx.scratch.as_mut_ptr();
    let scratch_u = scratch_ptr as usize;

    let (k_in_ptr, k_w_ptr, accs, k_in_c, k_row_delta, k_pad_w);
    let input_offset = params.input_offset;
    let wsum: *mut i32;
    if needs_pad {
        // Padded buffers — carve from scratch at 16-byte boundaries so the
        // kernel's VLD.128 stays aligned.
        let in_off = (scratch_u + 15) & !15;
        let w_off = in_off + pad_input_len;
        let accs_off = (w_off + pad_weights_len + 15) & !15;
        let p_in: *const i8 = unsafe { scratch_ptr.add(in_off - scratch_u) }.cast::<i8>();
        let p_w: *const i8 = unsafe { scratch_ptr.add(w_off - scratch_u) }.cast::<i8>();
        let p_accs = unsafe { scratch_ptr.add(accs_off - scratch_u) } as *mut i32;
        if (accs_off - scratch_u) % 4 != 0 {
            return Ok(false);
        }
        wsum = if input_offset != 0 {
            (unsafe { scratch_ptr.add(accs_off - scratch_u + out_c * 4) }) as *mut i32
        } else {
            core::ptr::null_mut()
        };
        // Zero-fill the padded input, then copy the real interior (offset by
        // pad_h/pad_w) and its channels into the padded channel slots. When
        // `input_offset != 0` the pad border is filled with `-input_offset`
        // (i8) so out-of-bounds taps compute `(-off)·w`, which the Phase C
        // `+off·Σw` fold cancels to exactly 0 — matching the scalar ref's
        // bounds-skip semantics for padded layers.
        let fill: u8 = if input_offset != 0 {
            (-input_offset) as u8
        } else {
            0
        };
        unsafe { core::ptr::write_bytes(p_in as *mut i8, fill, pad_input_len) };
        for h in 0..in_h {
            for w in 0..in_w {
                let src = unsafe { in_ptr.add((h * in_w + w) * input_c) };
                let dst = unsafe {
                    p_in.add(((h + pad_h) * padded_w + (w + pad_w)) * padded_c) as *mut i8
                };
                unsafe { core::ptr::copy_nonoverlapping(src, dst, input_c) };
            }
        }

        // Zero-fill the padded weights [oc][tap][padded_c], copy real channels.
        unsafe { core::ptr::write_bytes(p_w as *mut i8, 0, pad_weights_len) };
        for oc in 0..out_c {
            for tap in 0..9 {
                let src = unsafe { w_ptr.add((oc * 9 + tap) * input_c) };
                let dst = unsafe { p_w.add((oc * 9 + tap) * padded_c) as *mut i8 };
                unsafe { core::ptr::copy_nonoverlapping(src, dst, input_c) };
            }
        }

        k_in_ptr = p_in;
        k_w_ptr = p_w;
        accs = p_accs;
        k_in_c = padded_c;
        k_pad_w = padded_w;
        k_row_delta = if padded_w >= 3 { (padded_w - 3) * padded_c } else { 0 };
    } else {
        if (in_ptr as usize) % 16 != 0
            || (w_ptr as usize) % 16 != 0
            || (out_ptr as usize) % 16 != 0
        {
            return Ok(false);
        }
        k_in_ptr = in_ptr;
        k_w_ptr = w_ptr;
        accs = scratch_ptr as *mut i32;
        if (accs as usize) % 4 != 0 {
            return Ok(false);
        }
        wsum = if input_offset != 0 {
            unsafe { accs.add(out_c) }
        } else {
            core::ptr::null_mut()
        };
        k_in_c = input_c;
        k_pad_w = in_w;
        k_row_delta = if in_w >= 3 { (in_w - 3) * input_c } else { 0 };
    }

    // Compute the per-channel weight sums once (they are input-independent)
    // so a non-zero `input_offset` can be folded bit-exactly: the scalar acc is
    // `Σ (in + offset)·w = Σ in·w + offset·Σw`; the kernel produced `Σ in·w`.
    if input_offset != 0 {
        let ws = unsafe { core::slice::from_raw_parts_mut(wsum, out_c) };
        let wv = unsafe { core::slice::from_raw_parts(k_w_ptr, out_c * 9 * k_in_c) };
        crate::accx::weight_sums_conv(ws, wv, 9, k_in_c, out_c);
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
    let stride_h = params.stride_height as usize;
    let stride_w = params.stride_width as usize;

    for oh in 0..out_h {
        for ow in 0..out_w {
            let px = (oh * stride_h * k_pad_w + ow * stride_w) * k_in_c;
            let po = (oh * out_w + ow) * out_c;
            unsafe {
                crate::accx::accx_conv3x3(
                    k_in_ptr.add(px),
                    k_w_ptr,
                    accs,
                    k_in_c,
                    out_c,
                    k_row_delta,
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

/// Prepared 3×3 conv handle — runs the SIMD gate ONCE at construction, then
/// `run` only re-checks pointer alignment and dispatches.
///
/// The bespoke ACCX kernel (`s8_accx_conv3x3`) computes exact 32-bit dot
/// products, so SIMD output is bit-exact vs the scalar reference.
pub struct PreparedConv3x3 {
    /// Whether the bespoke ACCX SIMD kernel is eligible on this target.
    accx: bool,
    params: &'static Conv2DParams<'static>,
}

impl PreparedConv3x3 {
    pub fn new(params: &'static Conv2DParams<'static>) -> Result<Self, KernelError> {
        let input_c = params.input_shape[3] as usize;
        let out_channels = params.output_shape[3] as usize;
        let accx = cfg!(all(target_arch = "xtensa", not(feature = "qemu")))
            && crate::accx::accx_eligible_3x3(input_c, out_channels);
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
            let mut accx_ctx = Conv3x3AccxCtx {
                input,
                weights,
                bias,
                params: self.params,
                output,
                scratch,
            };
            if conv3x3_accx_dispatch(&mut accx_ctx)? {
                return Ok(());
            }
        }
        conv2d_3x3(input, weights, bias, self.params, output, scratch)
    }
}

/// Compute the product of a `[i32; 4]` shape array.
#[inline(always)]
fn shape_product(shape: &[i32; 4]) -> usize {
    shape[0] as usize * shape[1] as usize * shape[2] as usize * shape[3] as usize
}

/// Derived vertical pad (same formula as hematite-ref conv.rs).
#[inline]
pub(crate) fn conv_pad_h(params: &Conv2DParams<'_>) -> i32 {
    let dilated_h = (params.filter_shape[1] - 1) * params.dilation_height_factor + 1;
    ((params.output_shape[1] - 1) * params.stride_height + dilated_h - params.input_shape[1]) / 2
}

/// Derived horizontal pad (same formula as hematite-ref conv.rs).
#[inline]
pub(crate) fn conv_pad_w(params: &Conv2DParams<'_>) -> i32 {
    let dilated_w = (params.filter_shape[2] - 1) * params.dilation_width_factor + 1;
    ((params.output_shape[2] - 1) * params.stride_width + dilated_w - params.input_shape[2]) / 2
}

/// Conv2D general kernel — scalar path (host-compilable, bit-exact vs per-channel golden).
///
/// This kernel is structurally identical to the 1×1 conv kernel — the bias-init
/// i32 accumulator, `(i_val + input_offset) * w_val` MAC, per-channel
/// `multiply_by_quantized_multiplier`, output_offset, clamp, saturating_cast
/// loop is the same general 2D convolution used by the conv1x1 module.
///
/// The naming (conv2d_3x3) is for discoverability — the code handles arbitrary
/// filter dimensions. Zero-padding via bounds-check.
///
/// Only batch=1 is supported. Batch>1 returns [`KernelError::Unsupported`].
pub fn conv2d_3x3(
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

    // ── Extract dimensions ──────────────────────────────────────────────
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

    // Channel-dimension cross-check
    if input_c != filter_ic {
        return Err(KernelError::ShapeMismatch);
    }

    // ── Derived pad values (same formula as hematite-ref/src/conv.rs) ──
    let dilated_filter_h = (filter_h - 1) * params.dilation_height_factor + 1;
    let dilated_filter_w = (filter_w - 1) * params.dilation_width_factor + 1;
    let pad_h = ((out_h - 1) * params.stride_height + dilated_filter_h - input_h) / 2;
    let pad_w = ((out_w - 1) * params.stride_width + dilated_filter_w - input_w) / 2;

    // ── TIE728 SIMD dispatch (device-only; compiled out entirely on host) ──
    // Bespoke ACCX kernel: exact 32-bit dot product per output pixel/channel,
    // then a bit-exact TFLite requantize in Rust. Bit-exact vs the scalar path.
    //
    // ALSO gated off under the `qemu` feature: QEMU's xtensa/esp32s3 TIE728
    // emulation does not correctly execute the TIE MAC instructions this
    // kernel depends on (confirmed by direct bisection — see
    // local-notes/notepads/hematite-nn/problems.md). QEMU builds fall through to the
    // scalar path; real hardware still gets SIMD.
    #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
    {
        let mut accx_ctx = Conv3x3AccxCtx {
            input,
            weights,
            bias,
            params,
            output,
            scratch,
        };
        if conv3x3_accx_dispatch(&mut accx_ctx)? {
            return Ok(());
        }
    }

    // ── Per-channel multiplier/shift slices ─────────────────────────────
    let multipliers = params.output_multiplier_per_channel;
    let shifts = params.output_shift_per_channel;

    // ── Stride precomputation ──────────────────────────────────────────
    let input_row_stride = input_w * input_c;
    let filter_oc_stride = filter_h * filter_w * filter_ic;
    let filter_row_stride = filter_w * filter_ic;
    let filter_col_stride = filter_ic;
    let output_row_stride = out_w * out_channels;

    // ── Accumulation loop ───────────────────────────────────────────────
    for oh in 0..out_h {
        let input_base_h = oh * params.stride_height - pad_h;
        for ow in 0..out_w {
            let input_base_w = ow * params.stride_width - pad_w;
            for oc in 0..out_channels {
                let mut acc: i32 = bias[oc as usize];
                let filter_oc_base = oc * filter_oc_stride;

                for fh in 0..filter_h {
                    let in_h = input_base_h + fh * params.dilation_height_factor;
                    let row_in_bounds = in_h >= 0 && in_h < input_h;

                    for fw in 0..filter_w {
                        let in_w = input_base_w + fw * params.dilation_width_factor;

                        if row_in_bounds && in_w >= 0 && in_w < input_w {
                            let input_base =
                                (in_h * input_row_stride + in_w * input_c) as usize;
                            let filter_base = (filter_oc_base
                                + fh * filter_row_stride
                                + fw * filter_col_stride) as usize;

                            for ic in 0..filter_ic {
                                let i_val = i32::from(input[input_base + ic as usize]);
                                let w_val = i32::from(weights[filter_base + ic as usize]);
                                acc += (i_val + params.input_offset) * w_val;
                            }
                        }
                        // else: zero-padding — contribute 0
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

                let out_idx = (oh * output_row_stride + ow * out_channels + oc) as usize;
                output[out_idx] = saturating_cast(clamped);
            }
        }
    }

    let _ = scratch; // unused by scalar path

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// TIE728 SIMD backend — device-only (NEVER compiled on host)
// ─────────────────────────────────────────────────────────────────────────────

/// TIE728 SIMD backend for 3×3 conv2d.
///
/// This module is **entirely cfg-gated** behind `#[cfg(target_arch = "xtensa")]`
/// and is NEVER compiled on the host (stable-aarch64-apple-darwin). It exists
/// in the tree for structural review and Phase 5 device verification (T5.3).
#[cfg(target_arch = "xtensa")]
mod conv3x3_simd {
    // The global_asm! invocations live inside a module per Rust safety rules.
    // On device, the linker resolves dl_tie728_s8_conv2d_33cn from the
    // vendored .S files.

    /// TIE728 args struct for the `dl_tie728_s8_conv2d_33cn` entry point.
    ///
    /// Field offsets (in bytes) — verified against
    /// `tie728_s8_conv2d_hwcn_load_args` in `dl_tie728_s8_conv2d.S`:
    ///
    /// | Offset | Field | L32I source (33cn) |
    /// |--------|-------|---------------------|
    /// | +48 | filter | a5 ← 48 |
    /// | +52 | filter_height | hwcn macro reads |
    /// | +56 | filter_width | hwcn macro reads |
    /// | +60 | filter_y_offset | hwcn path reads |
    /// | +64 | mac_shift | a8 ← 64 |
    /// | +68 | bias | a11 ← 68 |
    /// | +76 | activation_alpha | relu path: a12 ← 76 |
    /// | +80 | activation_alpha_ptr | prelu path: a12 ← 80 |
    /// | +84 | activation_shift | a13 ← 84 |
    /// | +96 | output_channel_div_8 | a7 ← 96 |
    /// | +100 | c_div_x_1 | a6 ← 100 |
    /// | +104 | filter_channel_factor | a8 ← 104 (per-channel path) |
    /// | +108 | dilation_x_offset | a9 ← 108 |
    /// | +112 | dilation_y_offset | a10 ← 112 |
    /// | +136 | c_remainder | unaligned path |
    /// | +140 | n_remainder | unaligned path |
    ///
    /// ABI unverified — validate at T5.3 on device.
    /// The +76 activation_alpha is read by the relu path; prelu reads +80
    /// instead (activation_alpha_ptr). The struct models both slots.
    #[repr(C)]
    #[allow(dead_code)]
    struct Tie728Conv33Args {
        _pad0: [u8; 48],
        filter: *const i8,             // +48
        _pad1: [u8; 12],               // +52..+63: filter_h, filter_w, y_offset
        mac_shift: i32,                // +64
        bias: *const i32,              // +68
        _pad2: [u8; 4],               // +72..+75
        activation_alpha: i32,         // +76: relu path reads
        activation_alpha_ptr: *const u8,  // +80: prelu path reads
        activation_shift: i32,         // +84
        _pad3: [u8; 8],               // +88..+95
        output_channel_div_8: i32,     // +96
        c_div_x_1: i32,               // +100
        filter_channel_factor: *const i16, // +104
        dilation_x_offset: i32,        // +108
        dilation_y_offset: i32,        // +112
        _pad4: [u8; 20],               // +116..+135
        c_remainder: i32,              // +136
        n_remainder: i32,              // +140
    }

    /// Include the vendored TIE728 shared macros and conv2d entry points.
    core::arch::global_asm!(
        include_str!("../src/asm/dl_tie728_s8.S"),
        include_str!("../src/asm/dl_tie728_s8_conv2d.S"),
    );

    /// SIMD 3×3 conv2d — calls the vendored TIE728 entry point.
    ///
    /// # Safety
    ///
    /// Calls into foreign assembly via the C ABI (a2=output, a3=input, a4=args).
    /// ABI unverified — validate at T5.3 on device.
    #[allow(dead_code)]
    unsafe fn conv2d_3x3_simd_aligned(
        output: *mut i8,
        input: *const i8,
        args: &Tie728Conv33Args,
    ) {
        core::arch::asm!(
            "mov a10, {output}",
            "mov a11, {input}",
            "mov a12, {args}",
            "call8 dl_tie728_s8_conv2d_33cn",
            output = in(reg) output,
            input  = in(reg) input,
            args   = in(reg) args,
            clobber_abi("C"),
        );
    }

    /// SIMD 3×3 conv2d with fused ReLU.
    ///
    /// # Safety
    ///
    /// Same contract as `conv2d_3x3_simd_aligned`.
    #[allow(dead_code)]
    unsafe fn conv2d_3x3_simd_relu(
        output: *mut i8,
        input: *const i8,
        args: &Tie728Conv33Args,
    ) {
        core::arch::asm!(
            "mov a10, {output}",
            "mov a11, {input}",
            "mov a12, {args}",
            "call8 dl_tie728_s8_conv2d_33cn_relu",
            output = in(reg) output,
            input  = in(reg) input,
            args   = in(reg) args,
            clobber_abi("C"),
        );
    }

    /// Build a [`Tie728Conv33Args`] and dispatch — called from the public
    /// scalar `conv2d_3x3` eligibility check in the parent module.
    ///
    /// `conv3x3_simd` and `Tie728Conv33Args` are private to this module, so
    /// the eligibility-gated caller in `conv2d_3x3` cannot reach the entry
    /// points directly; this wrapper takes plain scalar arguments and
    /// builds the struct internally.
    ///
    /// # Safety
    ///
    /// Same safety contract as `conv2d_3x3_simd_aligned` / `_relu`.
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn dispatch_3x3(
        output: *mut i8,
        input: *const i8,
        filter: *const i8,
        bias: *const i32,
        mac_shift: i32,
        output_channel_div_8: i32,
        c_div_x_1: i32,
        dilation_x_offset: i32,
        dilation_y_offset: i32,
        use_relu: bool,
    ) {
        // MaybeUninit args build — write ONLY the asm-read fields
        // (+48/+64/+68/+76/+80/+84/+96/+100/+104/+108/+112), no memset/dead
        // pad stores. The unaligned-path fields +136/+140 stay uninit (never
        // read on the aligned path).
        let mut args = core::mem::MaybeUninit::<Tie728Conv33Args>::uninit();
        let p = args.as_mut_ptr();
        p.cast::<u8>().add(48).cast::<*const i8>().write(filter);
        p.cast::<u8>().add(64).cast::<i32>().write(mac_shift);
        p.cast::<u8>().add(68).cast::<*const i32>().write(bias);
        p.cast::<u8>().add(76).cast::<i32>().write(0);
        p.cast::<u8>().add(80).cast::<*const u8>().write(core::ptr::null());
        p.cast::<u8>().add(84).cast::<i32>().write(if use_relu { 0 } else { -1 });
        p.cast::<u8>().add(96).cast::<i32>().write(output_channel_div_8);
        p.cast::<u8>().add(100).cast::<i32>().write(c_div_x_1);
        p.cast::<u8>().add(104).cast::<*const i16>().write(core::ptr::null());
        p.cast::<u8>().add(108).cast::<i32>().write(dilation_x_offset);
        p.cast::<u8>().add(112).cast::<i32>().write(dilation_y_offset);
        let args = args.assume_init_ref();
        if use_relu {
            conv2d_3x3_simd_relu(output, input, args);
        } else {
            conv2d_3x3_simd_aligned(output, input, args);
        }
    }
}
