// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Bespoke ACCX conv SIMD kernels — **bit-exact** int8 convolution.
//!
//! # Why these kernels exist
//!
//! The vendored esp-dl `dl_tie728_s8_conv2d_*` per-layer path accumulates via
//! `EE.VSMULAS.S8.QACC`, whose lanes saturate at 8 bits — proven on device: a
//! single `127×127` product already clamps a lane to `0x7f`, so any realistic
//! int8 conv (accumulators reaching ±10⁵–10⁶) produces output that diverges
//! from the scalar reference (conv1x1 measured `0x61f7a941` vs ref
//! `0x0bea8225`). The S16 variant saturates at 16 bits (`0x7FFF` for 129032).
//! The data is lost *inside* QACC, so no post-processing can recover it.
//!
//! These kernels instead accumulate in a **32-bit GPR accumulator** via the
//! element-accumulator primitive:
//!
//! ```text
//!   EE.ZERO.ACCX
//!   EE.VLD.128.IP q0, filter_ptr, 16     ; 16 filter bytes
//!   EE.VLD.128.IP q1, input_ptr, 16      ; 16 input bytes
//!   EE.VMULAS.S8.ACCX q0, q1             ; accx += Σᵢ F[i]·I[i] (16-bit products)
//!   EE.SRS.ACCX gpr, 0, 0                ; exact int32 sum
//! ```
//!
//! `EE.VMULAS.S8.ACCX` is a 16-wide element-wise dot-product **reduction** into
//! a 32-bit accumulator with full 16-bit products (`127×127 = 16129` preserved,
//! verified on device: 16 MACs of 127² → 516128, exact). `EE.SRS.ACCX gpr, 0, 0`
//! extracts it exactly (the shift GPR maps through a triangular table, so 0 ⇒
//! no shift). Because the reduction is element-wise (`F[lane]·I[lane]`), the
//! **raw `[oc][ic]` weight layout works directly — no weight transform needed**.
//!
//! The kernels write raw int32 accumulators to a caller scratch buffer; the
//! Rust caller applies the per-channel requantize (bias, `mult/shift`,
//! `output_offset`, clamp, `saturating_cast`) — bit-identical to the scalar
//! reference for **any** per-channel quantization parameters and any
//! output-channel count.
//!
//! # Kernels
//!
//! * `s8_accx_conv1x1` — one input vector (1×1 conv / FC / one output pixel):
//!   `acc[oc] = Σ_ic filter[oc·in_c + ic] · input[ic]`.
//! * `s8_accx_conv3x3` — one 3×3 window (one output pixel, 9 taps, dilation):
//!   `acc[oc] = Σ_{tap} Σ_ic filter[(oc·9 + tap)·in_c + ic] · input[tap_off + ic]`.
//!
//! # Eligibility (both kernels)
//!
//! * `input_c % 16 == 0`, `input_c >= 16`, `out_c >= 1` (any)
//! * input/filter pointers 16-byte aligned (`EE.VLD.128`); acc_out 4-byte aligned
//! * caller supplies a scratch buffer of ≥ `out_c · 4` bytes for the accs
//!
//! # ABI (windowed — caller's a10..a14 = callee a2..a6)
//!
//! ```text
//! s8_accx_conv1x1(input, filter, acc_out, in_c, out_c)
//! s8_accx_conv3x3(input, filter, acc_out, in_c, out_c, row_delta)
//! ```

use hematite_int8::{multiply_by_quantized_multiplier, saturating_cast};

/// Host-compilable eligibility for the ACCX 1×1 / FC kernel.
#[inline]
pub(crate) fn accx_eligible_1x1(input_c: usize, out_c: usize) -> bool {
    input_c >= 16 && input_c % 16 == 0 && out_c >= 1
}

/// Host-compilable eligibility for the ACCX 3×3 kernel.
#[inline]
pub(crate) fn accx_eligible_3x3(input_c: usize, out_c: usize) -> bool {
    input_c >= 16 && input_c % 16 == 0 && out_c >= 1
}

/// Context for the per-channel requantize epilogue.
///
/// Bundled into a single struct passed by `&mut` because the Xtensa LLVM
/// backend miscompiles the multi-arg (9-slot) call site on device — the same
/// class of bug as the `dispatch_fc` inline regression. A 1-arg call is safe.
#[repr(C)]
pub(crate) struct ReqCtx<'a> {
    pub accs: &'a [i32],
    pub bias: &'a [i32],
    pub multipliers: &'a [i32],
    pub shifts: &'a [i32],
    pub output_offset: i32,
    pub act_min: i32,
    pub act_max: i32,
    pub out_base: usize,
    pub output: &'a mut [i8],
}

/// Shared per-channel requantize epilogue (bit-exact TFLite semantics).
///
/// Applies `bias + Σproducts`, then the per-channel fixed-point requantize,
/// `output_offset`, activation clamp and saturating cast — the exact scalar
/// reference arithmetic.
///
/// `#[inline(never)]`: when inlined, the Xtensa LLVM backend miscompiles the
/// `accs.iter()` loop bound (the write index runs one past `output.len()`,
/// panicking `index out of bounds`) — same class of bug as the earlier
/// `dispatch_fc` inline regression. Keeping this call separate is required.
#[inline(never)]
pub(crate) fn requantize_1x1(ctx: &mut ReqCtx<'_>) {
    for (oc, &raw) in ctx.accs.iter().enumerate() {
        let acc = raw + ctx.bias[oc];
        let scaled = multiply_by_quantized_multiplier(acc, ctx.multipliers[oc], ctx.shifts[oc]);
        let with_offset = scaled + ctx.output_offset;
        let clamped = if with_offset > ctx.act_max {
            ctx.act_max
        } else if with_offset < ctx.act_min {
            ctx.act_min
        } else {
            with_offset
        };
        ctx.output[ctx.out_base + oc] = saturating_cast(clamped);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Device asm glue — NEVER compiled on host or under the qemu feature.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
mod device {
    use core::arch::{asm, global_asm};

    global_asm!(include_str!("asm/s8_accx_conv1x1.S"));
    global_asm!(include_str!("asm/s8_accx_conv3x3.S"));

    /// One input vector → `out_c` raw int32 accumulators.
    ///
    /// # Safety
    /// `input`/`filter` 16-byte aligned, `acc_out` 4-byte aligned,
    /// `in_c % 16 == 0`, `in_c >= 16`, `out_c >= 1`, all buffers sized.
    pub unsafe fn accx_conv1x1(
        input: *const i8,
        filter: *const i8,
        acc_out: *mut i32,
        in_c: usize,
        out_c: usize,
    ) {
        asm!(
            "call8 s8_accx_conv1x1",
            in("a10") input,
            in("a11") filter,
            inout("a12") acc_out => _,
            in("a13") in_c,
            in("a14") out_c,
            out("a15") _,
            clobber_abi("C"),
        );
    }

    /// One 3×3 window (9 taps, dilation 1) → `out_c` raw int32 accumulators.
    ///
    /// `row_delta` = `(in_w - 3) * in_c` bytes — the offset between 3×3 rows
    /// (dilation must be 1; the horizontal tap delta is 0).
    ///
    /// # Safety
    /// `input` (window top-left tap)/`filter` 16-byte aligned, `acc_out`
    /// 4-byte aligned, `in_c % 16 == 0`, `in_c >= 16`, `out_c >= 1`, buffers
    /// sized.
    pub unsafe fn accx_conv3x3(
        input: *const i8,
        filter: *const i8,
        acc_out: *mut i32,
        in_c: usize,
        out_c: usize,
        row_delta: usize,
    ) {
        asm!(
            "call8 s8_accx_conv3x3",
            in("a10") input,
            in("a11") filter,
            inout("a12") acc_out => _,
            in("a13") in_c,
            in("a14") out_c,
            in("a15") row_delta,
            clobber_abi("C"),
        );
    }
}

#[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
pub(crate) use device::*;
