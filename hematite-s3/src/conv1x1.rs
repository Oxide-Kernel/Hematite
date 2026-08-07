// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Conv2D 1×1 GEMM kernel — scalar fallback + TIE728 SIMD backend.
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
//! * `weights` — OHWI `[Cout, 1, 1, Cin]`
//! * `bias` — per-output-channel `[Cout]`
//! * `output` — NHWC `[batch=1, OH, OW, Cout]`
//!
//! Scalar fallback handles odd input channels (not a multiple of 16) where the
//! TIE728 HWC16 layout requires 16-wide SIMD lanes. The SIMD backend pads to
//! HWC16 (channel dimension rounded up to multiple of 16); the scalar fallback
//! handles the residual channels or provides the complete kernel when no SIMD.
//!
//! # ABI (TIE728 SIMD — device only)
//!
//! The SIMD path calls the vendored `dl_tie728_s8_conv2d_11cn` /
//! `dl_tie728_s8_conv2d_11cn_relu` entry points from
//! `hematite-s3/src/asm/dl_tie728_s8_conv2d.S` via `global_asm!`.
//!
//! Register convention:
//! * a2 = output pointer (i8*)
//! * a3 = input pointer (i8*)
//! * a4 = args pointer (packed struct of input/output params)
//!
//! Args struct layout (offsets in bytes):
//! * +48: filter pointer
//! * +64: mac_shift
//! * +68: bias pointer
//! * +76: activation_alpha
//! * +84: activation_shift
//! * +96: output_channel_div_8
//! * +100: c_div_x_1 (input_channel / 16 - 1)
//! * +104: filter_channel_factor (per-channel scale factor pointer)
//!
//! Key instruction sequence (from `tie728_s8_conv2d_11c16`):
//! 1. `EE.VSMULAS.S8.QACC.LD.INCP` — 16-wide int8 MAC with QACC accumulate
//!    and auto-load-increment of input/filter pointers
//! 2. `EE.SRCMB.S8.QACC` — requantize from QACC (i48 accumulator) to int8
//! 3. `EE.VRELU.S8` — fused ReLU in vector epilogue
//! 4. `EE.VST.128.IP` — 128-bit aligned store with post-increment

use hematite_core::op_params::Conv2DParams;
use hematite_core::KernelError;
use hematite_int8::{multiply_by_quantized_multiplier, saturating_cast};

/// Compute the product of a `[i32; 4]` shape array.
#[inline(always)]
fn shape_product(shape: &[i32; 4]) -> usize {
    shape[0] as usize * shape[1] as usize * shape[2] as usize * shape[3] as usize
}

/// Conv2D 1×1 GEMM — scalar kernel (host-compilable, bit-exact vs per-channel golden).
///
/// Mirrors the `hematite-ref/src/conv.rs` arithmetic exactly: bias-init i32
/// accumulator, `(i_val + input_offset) * w_val` MAC, per-channel
/// `multiply_by_quantized_multiplier`, output_offset, clamp, saturating_cast.
///
/// For a 1×1 convolution (FH=1, FW=1), there is no spatial filter loop —
/// each spatial position maps directly to the corresponding input position.
/// Zero-padding via bounds-check is retained for structural compatibility with
/// the general conv path (identical to `hematite-ref/src/conv.rs`).
///
/// Only batch=1 is supported. Batch>1 returns [`KernelError::Unsupported`].
pub fn conv2d_1x1(
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

    // ── TIE728 SIMD dispatch (device-only; compiled out entirely on host) ──
    // Eligibility per Plan A4 T5.3: zero offsets, uniform per-channel
    // multiplier/shift collapsing to the hardware's fixed-multiplier fast
    // path, 16-aligned channel counts, batch=1 (checked above), and runtime
    // 16-byte pointer alignment. Padding is structurally 0 for a 1x1/stride-1
    // conv, so it is not checked.
    //
    // ALSO gated off under the `qemu` feature: QEMU's xtensa/esp32s3 TIE728
    // emulation does not correctly execute `EE.VSMULAS.S8.QACC.LD.INCP` (the
    // fused MAC+load+increment instruction this kernel's MAC loop depends
    // on) — confirmed by direct instruction-level bisection (see
    // local-notes/notepads/hematite-nn/problems.md). This is a QEMU emulation gap,
    // not a code defect, so QEMU builds fall through to the scalar path;
    // real hardware (no `qemu` feature) still gets SIMD once T5.3 validates
    // it there.
    #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
    {
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
            && input_c % 16 == 0
            && input_c >= 16
            && out_channels % 16 == 0
        {
            let in_ptr = input.as_ptr();
            let w_ptr = weights.as_ptr();
            let b_ptr = bias.as_ptr();
            let out_ptr = output.as_mut_ptr();
            let aligned = (in_ptr as usize) % 16 == 0
                && (w_ptr as usize) % 16 == 0
                && (b_ptr as usize) % 16 == 0
                && (out_ptr as usize) % 16 == 0;

            if aligned {
                let use_relu = relu_range && !full_range;
                unsafe {
                    conv1x1_simd::dispatch_1x1(
                        out_ptr,
                        in_ptr,
                        w_ptr,
                        b_ptr,
                        shift[0],
                        out_channels / 16,
                        input_c / 16 - 1,
                        use_relu,
                    );
                }
                let _ = scratch;
                return Ok(());
            }
        }
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

    // ── Accumulation loop (identical to hematite-ref conv.rs) ───────────
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

/// TIE728 SIMD backend for 1×1 conv2d.
///
/// This module is **entirely cfg-gated** behind `#[cfg(target_arch = "xtensa")]`
/// and is NEVER compiled on the host (stable-aarch64-apple-darwin). It exists
/// in the tree for structural review and Phase 5 device verification (T5.3).
///
/// ## Architecture
///
/// The SIMD path:
/// 1. Pads input channels to multiple of 16 (HWC16 layout — zero-pad if
///    `input_c % 16 != 0`).
/// 2. Calls the vendored `dl_tie728_s8_conv2d_11cn` entry point from
///    `hematite-s3/src/asm/dl_tie728_s8_conv2d.S` via `global_asm!`.
/// 3. For inputs with odd channels (not a multiple of 16), calls the host-
///    compiled scalar `conv2d_1x1` above as a fallback.
///
/// ## A4 contract notes
///
/// * Leg (a): SIMD output must match a per-tensor TFLM golden (Phase 5 fixture
///   with per-tensor OUTPUT_MULTIPLIER/SHIFT instead of per-channel arrays).
/// * Leg (c): SIMD vs scalar ref cross-check tolerance ≤1 LSB on requantize.
///   This captures per-channel vs per-tensor quantization differences at
///   the bit level. Documented in the test file.
#[cfg(target_arch = "xtensa")]
#[path = ""]
mod conv1x1_simd {
    // The global_asm! invocations must live inside a module (per Rust safety
    // rules). On device, the linker resolves dl_tie728_s8_conv2d_11cn from the
    // vendored .S files.
    //
    // TIE728 args struct (packed, repr(C), u32-aligned):
    // Layout mirrors esp-dl's Conv2D args at dl_tie728_s8_conv2d.S
    // dl_tie728_s8_conv2d_11cn function.

    /// TIE728 args struct — matches the `void *args` pointer convention in the
    /// vendored `dl_tie728_s8_conv2d_11cn` entry point.
    ///
    /// Field offsets (in bytes) — verified against
    /// `tie728_s8_conv2d_11cn_load_args` in dl_tie728_s8_conv2d.S:
    ///
    /// | Offset | Field | Description |
    /// |--------|-------|-------------|
    /// | +48 | filter | int8_t* filter_ptr |
    /// | +64 | mac_shift | i32 requantize shift (negative → per-channel) |
    /// | +68 | bias | i32* bias_ptr |
    /// | +76 | activation_alpha | i32 (LeakyReLU alpha / ReLU flag) |
    /// | +84 | activation_shift | i32 (negative → no activation) |
    /// | +96 | output_channel_div_8 | i32 |
    /// | +100 | c_div_x_1 | i32 = input_channel / 16 - 1 |
    /// | +104 | filter_channel_factor | i16* per-channel scale factor ptr |
    #[repr(C)]
    #[allow(dead_code)]
    pub struct Tie728ConvArgs {
        // Layout verified against the vendored .S — fields are referenced by
        // byte offset (l32i a5, a4, 48 etc.), not by struct access.
        _pad0: [u8; 48],               // offset 0-47: unused by the .S
        filter: *const i8,              // offset 48
        _pad1: [u8; 12],               // offset 52-63
        mac_shift: i32,                // offset 64
        bias: *const i32,              // offset 68
        _pad2: [u8; 4],                // offset 72-75
        activation_alpha: i32,          // offset 76
        _pad3: [u8; 4],                // offset 80-83
        activation_shift: i32,          // offset 84
        _pad4: [u8; 8],                // offset 88-95
        output_channel_div_8: i32,      // offset 96
        c_div_x_1: i32,                // offset 100
        filter_channel_factor: *const i16, // offset 104
    }

    /// Include the vendored TIE728 shared macros and conv2d entry points.
    ///
    /// These two files define:
    /// * `dl_tie728_s8.S` — shared macros (requantize, bias preload, ReLU/PRelu epilogues)
    /// * `dl_tie728_s8_conv2d.S` — entry points: `dl_tie728_s8_conv2d_11cn`,
    ///   `dl_tie728_s8_conv2d_11cn_relu`, `dl_tie728_s8_conv2d_11cn_prelu`,
    ///   and unaligned variants.
    core::arch::global_asm!(
        include_str!("../src/asm/dl_tie728_s8.S"),
        include_str!("../src/asm/dl_tie728_s8_conv2d.S"),
    );

    // ── SIMD kernel glue ────────────────────────────────────────────────

    /// SIMD 1×1 conv2d — calls the vendored TIE728 entry point.
    ///
    /// # Safety
    ///
    /// This function is inherently unsafe: it calls into foreign
    /// assembly via the C ABI (a2=output, a3=input, a4=args).
    ///
    /// # Preconditions (caller MUST guarantee)
    ///
    /// * Input channels must be a multiple of 16 (HWC16 layout).
    ///   The scalar fallback handles odd channels.
    /// * All pointers must be 16-byte aligned for EE.VLD.128.IP / EE.VST.128.IP.
    /// * `input`, `weights`, `bias`, and `output` must be valid for the
    ///   declared shapes.
    /// * The `Tie728ConvArgs` struct must be correctly populated per the
    ///   vendored .S ABI.
    #[allow(dead_code)]
    pub unsafe fn conv2d_1x1_simd_aligned(
        output: *mut i8,
        input: *const i8,
        args: &Tie728ConvArgs,
    ) {
        // Procedure call standard (Xtensa XCC):
        // a2 = output (first arg)
        // a3 = input (second arg)
        // a4 = args (third arg — pointer to Tie728ConvArgs)
        //
        // The vendored .S entry points use the `entry sp, 128` convention
        // which saves the caller's registers in a 128-byte window.
        //
        // The call site below is hand-written inline asm that loads the
        // three pointer arguments into a2/a3/a4 and branches to the
        // appropriate entry point.

        core::arch::asm!(
            // Load arguments into the ABI registers.
            //
            // `dl_tie728_s8_conv2d_11cn` uses `entry sp, 128`, so it is called
            // with the windowed `call8` convention: the callee's a2/a3/a4 are
            // the caller's a10/a11/a12 (the window rotates by 8). Placing the
            // args in a2/a3/a4 would deliver garbage pointers to the callee.
            "mov a10, {output}",
            "mov a11, {input}",
            "mov a12, {args}",

            // Branch to dl_tie728_s8_conv2d_11cn (aligned, no fused activation)
            "call8 dl_tie728_s8_conv2d_11cn",

            output = in(reg) output,
            input  = in(reg) input,
            args   = in(reg) args,
            // Clobber all caller-saved registers
            clobber_abi("C"),
        );
    }

    /// SIMD 1×1 conv2d with fused ReLU — calls the vendored TIE728 entry point.
    ///
    /// # Safety
    ///
    /// Same safety contract as `conv2d_1x1_simd_aligned`. Additionally,
    /// `args.activation_alpha` and `args.activation_shift` must be set for
    /// ReLU (alpha=0 means standard ReLU; non-zero means LeakyReLU).
    #[allow(dead_code)]
    pub unsafe fn conv2d_1x1_simd_relu(
        output: *mut i8,
        input: *const i8,
        args: &Tie728ConvArgs,
    ) {
        core::arch::asm!(
            "mov a10, {output}",
            "mov a11, {input}",
            "mov a12, {args}",
            "call8 dl_tie728_s8_conv2d_11cn_relu",
            output = in(reg) output,
            input  = in(reg) input,
            args   = in(reg) args,
            clobber_abi("C"),
        );
    }

    /// Build a [`Tie728ConvArgs`] and dispatch — called from the public
    /// scalar `conv2d_1x1` eligibility check in the parent module.
    ///
    /// `Tie728ConvArgs`'s fields are private to this module, so the
    /// eligibility-gated caller in `conv2d_1x1` cannot construct the struct
    /// itself; this wrapper does it with plain scalar arguments.
    ///
    /// # Safety
    ///
    /// Same safety contract as `conv2d_1x1_simd_aligned` / `_relu`.
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn dispatch_1x1(
        output: *mut i8,
        input: *const i8,
        filter: *const i8,
        bias: *const i32,
        mac_shift: i32,
        output_channel_div_8: i32,
        c_div_x_1: i32,
        use_relu: bool,
    ) {
        let args = Tie728ConvArgs {
            _pad0: [0u8; 48],
            filter,
            _pad1: [0u8; 12],
            mac_shift,
            bias,
            _pad2: [0u8; 4],
            activation_alpha: 0,
            _pad3: [0u8; 4],
            activation_shift: if use_relu { 0 } else { -1 },
            _pad4: [0u8; 8],
            output_channel_div_8,
            c_div_x_1,
            filter_channel_factor: core::ptr::null(),
        };
        if use_relu {
            conv2d_1x1_simd_relu(output, input, &args);
        } else {
            conv2d_1x1_simd_aligned(output, input, &args);
        }
    }
}

// Re-export the SIMD entry points at the crate level so the device
// integration layer can call them without navigating the module tree.
#[cfg(target_arch = "xtensa")]
pub use conv1x1_simd::conv2d_1x1_simd_aligned;
#[cfg(target_arch = "xtensa")]
pub use conv1x1_simd::conv2d_1x1_simd_relu;
