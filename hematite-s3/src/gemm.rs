// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Fully-connected / GEMM kernel — scalar fallback + TIE728 SIMD backend.
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
//! * `input` — flat `[input_dim]` (no spatial structure)
//! * `weights` — `output_dim × input_dim` row-major
//! * `bias` — per-output-unit `[output_dim]`
//! * `output` — flat `[output_dim]`
//!
//! The scalar kernel is a flat dot product per output unit with per-channel
//! requantize — identical to the hematite-ref fully_connected kernel.
//!
//! # SIMD backend
//!
//! The GEMM core is the same as the 1×1 conv (11cN) — each output is a dot
//! product of the input vector with one row of the weight matrix. On device,
//! the `dl_tie728_s8_conv2d_11cn` entry point is reused with the spatial
//! dimensions collapsed to 1×1.

use hematite_core::op_params::FullyConnectedParams;
use hematite_core::KernelError;
use hematite_int8::{multiply_by_quantized_multiplier, saturating_cast};

/// Fully-connected layer — scalar kernel (host-compilable, bit-exact vs per-channel golden).
///
/// Mirrors `hematite-ref/src/fully_connected.rs` semantics exactly: bias-init
/// i32 accumulator, `(i_val + input_offset) * w_val` MAC over input depth,
/// per-channel `multiply_by_quantized_multiplier`, output_offset, clamp,
/// saturating_cast.
pub fn fully_connected(
    input: &[i8],
    weights: &[i8],
    bias: &[i32],
    params: &FullyConnectedParams,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    let input_dim = params.input_dim as usize;
    let output_dim = params.output_dim as usize;

    // ── Slice-length validation ─────────────────────────────────────────
    if input.len() != input_dim {
        return Err(KernelError::ShapeMismatch);
    }
    if weights.len() != output_dim * input_dim {
        return Err(KernelError::ShapeMismatch);
    }
    if bias.len() != output_dim {
        return Err(KernelError::ShapeMismatch);
    }
    if output.len() != output_dim {
        return Err(KernelError::ShapeMismatch);
    }

    // ── Per-channel multiplier / shift slices ───────────────────────────
    let multipliers = params.output_multiplier_per_channel;
    let shifts = params.output_shift_per_channel;

    // ── TIE728 SIMD dispatch (device-only; compiled out entirely on host) ──
    // Same eligibility contract as conv1x1's dispatch, for input_dim/output_dim
    // in place of Cin/Cout (an FC layer is a 1x1 conv with H=W=1).
    //
    // ALSO gated off under the `qemu` feature: this reuses conv1x1's entry
    // point (`dl_tie728_s8_conv2d_11cn`), whose `EE.VSMULAS.S8.QACC.LD.INCP`
    // MAC instruction QEMU's TIE728 emulation does not correctly execute
    // (confirmed by direct bisection — see
    // local-notes/notepads/hematite-nn/problems.md). QEMU builds fall through to
    // the scalar path; real hardware still gets SIMD.
    #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
    {
        let mult_uniform = !multipliers.is_empty()
            && multipliers.iter().all(|&m| m == multipliers[0])
            && multipliers[0] == 1 << 30;
        let shift_uniform = !shifts.is_empty() && shifts.iter().all(|&s| s == shifts[0]);
        let full_range = params.quantized_activation_min == i8::MIN as i32
            && params.quantized_activation_max == i8::MAX as i32;
        let relu_range = params.quantized_activation_min == 0
            && params.quantized_activation_max == i8::MAX as i32;

        if params.input_offset == 0
            && params.output_offset == 0
            && (full_range || relu_range)
            && mult_uniform
            && shift_uniform
            && input_dim % 16 == 0
            && input_dim >= 16
            && output_dim % 16 == 0
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
                    gemm_simd::dispatch_fc(
                        out_ptr,
                        in_ptr,
                        w_ptr,
                        b_ptr,
                        shifts[0],
                        (output_dim / 16) as i32,
                        (input_dim / 16) as i32 - 1,
                        use_relu,
                    );
                }
                let _ = scratch;
                return Ok(());
            }
        }
    }

    // ── Accumulation loop ───────────────────────────────────────────────
    // TFLM loop order: batch(=0) → oc → accum_depth
    for oc in 0..output_dim {
        let mut acc: i32 = bias[oc];

        let weight_base = oc * input_dim;
        for d in 0..input_dim {
            let i_val = i32::from(input[d]);
            let w_val = i32::from(weights[weight_base + d]);
            acc += (i_val + params.input_offset) * w_val;
        }

        // Per-channel requantize + output offset + clamp
        let multiplier = multipliers[oc];
        let shift = shifts[oc];
        let scaled = multiply_by_quantized_multiplier(acc, multiplier, shift);
        let with_offset = scaled + params.output_offset;

        let clamped = if with_offset > params.quantized_activation_max {
            params.quantized_activation_max
        } else if with_offset < params.quantized_activation_min {
            params.quantized_activation_min
        } else {
            with_offset
        };

        output[oc] = saturating_cast(clamped);
    }

    let _ = scratch; // unused by scalar path

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// TIE728 SIMD backend — device-only (NEVER compiled on host)
// ─────────────────────────────────────────────────────────────────────────────

/// TIE728 SIMD backend for fully-connected (GEMM).
///
/// The GEMM core reuses the 1×1 conv entry point (`dl_tie728_s8_conv2d_11cn`)
/// since a fully-connected layer is mathematically equivalent to a 1×1 conv
/// with spatial dimensions H=W=1.
///
/// This module is cfg-gated behind `#[cfg(target_arch = "xtensa")]`.
/// ABI unverified — validate at T5.3 on device.
#[cfg(target_arch = "xtensa")]
mod gemm_simd {
    /// TIE728 FC/GEMM args struct.
    ///
    /// Reuses the same args layout as the 1×1 conv2d entry point.
    /// See `conv1x1.rs` for the canonical Tie728ConvArgs struct and
    /// the known ABI issues at +76 vs +80 (activation_alpha vs
    /// activation_alpha_ptr).
    ///
    /// ABI unverified — validate at T5.3 on device.
    #[repr(C)]
    #[allow(dead_code)]
    struct Tie728GemmArgs {
        _pad0: [u8; 48],
        filter: *const i8,             // +48
        _pad1: [u8; 12],               // +52..+63
        mac_shift: i32,                // +64
        bias: *const i32,              // +68
        _pad2: [u8; 4],                // +72..+75
        activation_alpha: i32,         // +76: relu path reads
        activation_alpha_ptr: *const u8,  // +80: prelu path reads
        activation_shift: i32,         // +84
        _pad3: [u8; 8],                // +88..+95
        output_channel_div_8: i32,     // +96
        c_div_x_1: i32,               // +100
        filter_channel_factor: *const i16, // +104
    }

    /// Include the vendored TIE728 shared macros and conv2d entry points.
    core::arch::global_asm!(
        include_str!("../src/asm/dl_tie728_s8.S"),
        include_str!("../src/asm/dl_tie728_s8_conv2d.S"),
    );

    /// SIMD fully-connected — calls the vendored TIE728 1×1 conv entry point.
    ///
    /// An FC layer with input_dim and output_dim is equivalent to a 1×1 conv
    /// with H=W=1, Cin=input_dim, Cout=output_dim.
    ///
    /// # Safety
    ///
    /// Calls into foreign assembly via the C ABI. ABI unverified.
    #[allow(dead_code)]
    unsafe fn fc_simd_aligned(
        output: *mut i8,
        input: *const i8,
        args: &Tie728GemmArgs,
    ) {
        core::arch::asm!(
            "mov a2, {output}",
            "mov a3, {input}",
            "mov a4, {args}",
            "call8 dl_tie728_s8_conv2d_11cn",
            output = in(reg) output,
            input  = in(reg) input,
            args   = in(reg) args,
            clobber_abi("C"),
        );
    }

    /// SIMD fully-connected with fused ReLU.
    ///
    /// # Safety
    ///
    /// Same contract as `fc_simd_aligned`.
    #[allow(dead_code)]
    unsafe fn fc_simd_relu(
        output: *mut i8,
        input: *const i8,
        args: &Tie728GemmArgs,
    ) {
        core::arch::asm!(
            "mov a2, {output}",
            "mov a3, {input}",
            "mov a4, {args}",
            "call8 dl_tie728_s8_conv2d_11cn_relu",
            output = in(reg) output,
            input  = in(reg) input,
            args   = in(reg) args,
            clobber_abi("C"),
        );
    }

    /// Build a [`Tie728GemmArgs`] and dispatch — called from the public
    /// scalar `fully_connected` eligibility check in the parent module.
    ///
    /// `gemm_simd` is private, so the eligibility-gated caller in
    /// `fully_connected` cannot reach the entry points directly; this
    /// wrapper takes plain scalar arguments and builds the struct
    /// internally.
    ///
    /// # Safety
    ///
    /// Same safety contract as `fc_simd_aligned` / `fc_simd_relu`.
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn dispatch_fc(
        output: *mut i8,
        input: *const i8,
        filter: *const i8,
        bias: *const i32,
        mac_shift: i32,
        output_channel_div_8: i32,
        c_div_x_1: i32,
        use_relu: bool,
    ) {
        let args = Tie728GemmArgs {
            _pad0: [0u8; 48],
            filter,
            _pad1: [0u8; 12],
            mac_shift,
            bias,
            _pad2: [0u8; 4],
            activation_alpha: 0,
            activation_alpha_ptr: core::ptr::null(),
            activation_shift: if use_relu { 0 } else { -1 },
            _pad3: [0u8; 8],
            output_channel_div_8,
            c_div_x_1,
            filter_channel_factor: core::ptr::null(),
        };
        if use_relu {
            fc_simd_relu(output, input, &args);
        } else {
            fc_simd_aligned(output, input, &args);
        }
    }
}
