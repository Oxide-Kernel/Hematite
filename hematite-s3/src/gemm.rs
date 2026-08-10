// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Fully-connected / GEMM kernel — scalar fallback + ACCX SIMD backend.
//!
//! # Bit-exact contract (Plan A4)
//!
//! | Leg | Contract | Runs on |
//! |-----|----------|---------|
//! | (a) | SIMD ≡ per-tensor TFLM golden bit-exact | Device (Phase 5) |
//! | (b) | Scalar ref ≡ per-channel TFLM golden bit-exact | **Host** (this test) |
//! | (c) | SIMD vs ref cross-check ≤1 LSB on requantize | Device (Phase 5) |
//!
//! On host (stable-aarch64-apple-darwin), only leg (b) executes. The SIMD
//! dispatch is `#[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]` —
//! device-only (see [`fc_accx_dispatch`]); the scalar kernel below is the
//! complete bit-exact fallback on every other target.
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
//! On device, the bespoke `s8_accx_conv1x1` kernel (assembled into the crate
//! by [`crate::accx`]) computes the exact 32-bit dot product per output unit
//! into scratch, then the bit-exact TFLite requantize epilogue runs in Rust —
//! an FC layer is mathematically a 1×1 conv with H=W=1. See
//! [`fc_accx_dispatch`] for the eligibility gate.

use hematite_core::op_params::FullyConnectedParams;
use hematite_core::KernelError;
use hematite_int8::{multiply_by_quantized_multiplier, saturating_cast};

/// Context for the ACCX FC/GEMM dispatch — bundled into one `&mut` arg so the
/// Xtensa LLVM backend generates a 1-arg call (multi-arg calls are miscompiled
/// on device; see the ACCX ctx refactors in `crate::accx`).
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

    if !crate::accx::accx_eligible_1x1(input_dim, output_dim) {
        return Ok(false);
    }

    let input_offset = params.input_offset;
    let need = output_dim * 4 + if input_offset != 0 { output_dim * 4 } else { 0 };
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
    let wsum = if input_offset != 0 {
        unsafe { accs.add(output_dim) }
    } else {
        core::ptr::null_mut()
    };
    if input_offset != 0 {
        let ws = unsafe { core::slice::from_raw_parts_mut(wsum, output_dim) };
        let wv = unsafe { core::slice::from_raw_parts(w_ptr, output_dim * input_dim) };
        crate::accx::weight_sums_conv(ws, wv, 1, input_dim);
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
    if input_offset != 0 {
        for oc in 0..output_dim {
            let v = unsafe { accs.add(oc).read() };
            let s = unsafe { wsum.add(oc).read() };
            unsafe { accs.add(oc).write(v.wrapping_add(input_offset.wrapping_mul(s))) };
        }
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

    // ── ACCX SIMD dispatch (device-only; compiled out entirely on host) ──
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
