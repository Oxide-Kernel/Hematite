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

/// Compute the product of a `[i32; 4]` shape array.
#[inline(always)]
fn shape_product(shape: &[i32; 4]) -> usize {
    shape[0] as usize * shape[1] as usize * shape[2] as usize * shape[3] as usize
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
            "mov a2, {output}",
            "mov a3, {input}",
            "mov a4, {args}",
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
            "mov a2, {output}",
            "mov a3, {input}",
            "mov a4, {args}",
            "call8 dl_tie728_s8_conv2d_33cn_relu",
            output = in(reg) output,
            input  = in(reg) input,
            args   = in(reg) args,
            clobber_abi("C"),
        );
    }
}
