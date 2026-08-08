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

/// Shared TIE728 SIMD eligibility gate for the FC/GEMM kernel — host-compilable.
///
/// Returns `Some((mac_shift, output_channel_div_8, c_div_x_1, use_relu))` when
/// the params qualify for the reused `dl_tie728_s8_conv2d_11cn` entry point,
/// `None` otherwise. Single source of truth for both the legacy `fully_connected`
/// dispatch and the [`PreparedFc`] handle.
#[inline]
pub(crate) fn simd_eligible_fc(
    params: &FullyConnectedParams<'_>,
    input_dim: i32,
    output_dim: i32,
) -> Option<(i32, i32, i32, bool)> {
    let mult = params.output_multiplier_per_channel;
    let shift = params.output_shift_per_channel;
    let mult_uniform = !mult.is_empty()
        && mult.iter().all(|&m| m == mult[0])
        && mult[0] == 1 << 30;
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
        && input_dim % 16 == 0
        && input_dim >= 16
        && output_dim % 16 == 0
    {
        let use_relu = relu_range && !full_range;
        // See conv1x1::simd_eligible_conv1x1 for the mac_shift semantics:
        // the asm round_result(acc, mac_shift) reproduces the scalar
        // multiply_by_quantized_multiplier for mult==1<<30 only when
        // mac_shift == 1 - shift[0].
        Some((1 - shift[0], output_dim / 16, input_dim / 16 - 1, use_relu))
    } else {
        None
    }
}

/// Context for the ACCX FC/GEMM dispatch — bundled into one `&mut` arg so the
/// Xtensa LLVM backend generates a 1-arg call (multi-arg calls are miscompiled
/// on device; see the `dispatch_fc` inline regression and `ReqCtx`).
#[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
pub(crate) struct FcAccxCtx<'a> {
    pub input: &'a [i8],
    pub weights: &'a [i8],
    pub bias: &'a [i32],
    pub params: &'a FullyConnectedParams<'a>,
    pub output: &'a mut [i8],
    pub scratch: &'a mut [u8],
}

/// ACCX SIMD dispatch for the FC/GEMM kernel — device-only.
///
/// Mirrors the conv1x1 ACCX path: the bespoke `s8_accx_conv1x1` kernel
/// computes the exact 32-bit dot product per output unit into `scratch`, then
/// the bit-exact TFLite requantize epilogue runs in Rust.
///
/// Returns `Ok(true)` when the ACCX path handled the layer, `Ok(false)` when
/// the layer is not ACCX-eligible (caller falls through to scalar).
#[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
#[inline(never)]
fn fc_accx_dispatch(ctx: &mut FcAccxCtx<'_>) -> Result<bool, KernelError> {
    let params = ctx.params;
    let input_dim = params.input_dim as usize;
    let output_dim = params.output_dim as usize;

    if params.input_offset != 0
        || !crate::accx::accx_eligible_1x1(input_dim, output_dim)
    {
        return Ok(false);
    }

    let need = output_dim * 4;
    if ctx.scratch.len() < need {
        return Ok(false);
    }

    let in_ptr = ctx.input.as_ptr();
    let w_ptr = ctx.weights.as_ptr();
    let out_ptr = ctx.output.as_mut_ptr();
    let accs = ctx.scratch.as_mut_ptr() as *mut i32;
    if (in_ptr as usize) % 16 != 0
        || (w_ptr as usize) % 16 != 0
        || (out_ptr as usize) % 16 != 0
        || (accs as usize) % 4 != 0
    {
        return Ok(false);
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

    unsafe {
        crate::accx::accx_conv1x1(in_ptr, w_ptr, accs, input_dim, output_dim);
    }
    let acc_slice = unsafe { core::slice::from_raw_parts_mut(accs, output_dim) };
    crate::accx::requantize_1x1(&mut crate::accx::ReqCtx {
        accs: acc_slice,
        bias: ctx.bias,
        multipliers,
        shifts,
        output_offset: out_offset,
        act_min,
        act_max,
        out_base: 0,
        output: ctx.output,
        uniform_mult,
        uniform_shift,
    });
    Ok(true)
}

/// Prepared FC/GEMM handle — runs the SIMD eligibility gate ONCE at
/// construction, then `run` only re-checks pointer alignment and dispatches.
///
/// The bespoke ACCX kernel (`s8_accx_conv1x1`) computes exact 32-bit dot
/// products, so SIMD output is bit-exact vs the scalar reference.
pub struct PreparedFc {
    /// Whether the bespoke ACCX SIMD kernel is eligible on this target.
    accx: bool,
    params: &'static FullyConnectedParams<'static>,
}

impl PreparedFc {
    pub fn new(params: &'static FullyConnectedParams<'static>) -> Result<Self, KernelError> {
        let input_dim = params.input_dim as usize;
        let output_dim = params.output_dim as usize;
        let accx = cfg!(all(target_arch = "xtensa", not(feature = "qemu")))
            && crate::accx::accx_eligible_1x1(input_dim, output_dim);
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
            let mut accx_ctx = FcAccxCtx {
                input,
                weights,
                bias,
                params: self.params,
                output,
                scratch,
            };
            if fc_accx_dispatch(&mut accx_ctx)? {
                return Ok(());
            }
        }
        fully_connected(input, weights, bias, self.params, output, scratch)
    }
}

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
    // Bespoke ACCX kernel: exact 32-bit dot product per output unit, then a
    // bit-exact TFLite requantize in Rust. Bit-exact vs the scalar path.
    //
    // ALSO gated off under the `qemu` feature: QEMU's xtensa/esp32s3 TIE728
    // emulation does not correctly execute the TIE MAC instructions this
    // kernel depends on (confirmed by direct instruction-level bisection —
    // see local-notes/notepads/hematite-nn/problems.md). QEMU builds fall through to
    // the scalar path; real hardware still gets SIMD.
    #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
    {
        let mut accx_ctx = FcAccxCtx {
            input,
            weights,
            bias,
            params,
            output,
            scratch,
        };
        if fc_accx_dispatch(&mut accx_ctx)? {
            return Ok(());
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
            "mov a10, {output}",
            "mov a11, {input}",
            "mov a12, {args}",
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
    #[inline(never)]
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
        // MaybeUninit args build — write ONLY the asm-read fields
        // (+48/+64/+68/+76/+84/+96/+100/+104), no memset/dead pad stores.
        let mut args = core::mem::MaybeUninit::<Tie728GemmArgs>::uninit();
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
        let args = args.assume_init_ref();
        if use_relu {
            fc_simd_relu(output, input, args);
        } else {
            fc_simd_aligned(output, input, args);
        }
    }
}
