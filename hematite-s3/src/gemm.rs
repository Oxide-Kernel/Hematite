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
///
/// `uniform` is the precomputed uniform-scale hint `(mult, shift)` —
/// `i32::MIN` shift means "per-channel" (the requantize epilogue selects the
/// fast scale inline, no upfront O(n) scan; todo 16).
#[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
#[inline(never)]
fn fc_accx_dispatch(ctx: &mut FcAccxCtx<'_>, uniform: (i32, i32)) -> Result<bool, KernelError> {
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
    let (uniform_mult, uniform_shift) = uniform;

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
    /// Cached uniform-scale hint `(mult, shift)`; `i32::MIN` shift = per-channel.
    /// Computed once at construction so `run` never re-scans the per-channel
    /// arrays (the per-call cost is otherwise O(output_dim) per call).
    /// Read only by the device dispatch (host: SIMD compiled out).
    #[allow(dead_code)]
    uniform: (i32, i32),
}

impl PreparedFc {
    pub fn new(params: &'static FullyConnectedParams<'static>) -> Result<Self, KernelError> {
        let input_dim = params.input_dim as usize;
        let output_dim = params.output_dim as usize;
        let accx = cfg!(all(target_arch = "xtensa", not(feature = "qemu")))
            && crate::accx::accx_eligible_1x1(input_dim, output_dim);
        let uniform =
            crate::accx::uniform_scale(params.output_multiplier_per_channel, params.output_shift_per_channel)
                .unwrap_or((0, i32::MIN));
        Ok(Self { accx, params, uniform })
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
            if fc_accx_dispatch(&mut accx_ctx, self.uniform)? {
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

    // ── ACCX SIMD dispatch (device-only; compiled out entirely on host) ──
    // Bespoke ACCX kernel: exact 32-bit dot product per output unit, then a
    // bit-exact TFLite requantize in Rust. Bit-exact vs the scalar path.
    //
    // ALSO gated off under the `qemu` feature: QEMU's xtensa/esp32s3 TIE728
    // emulation does not correctly execute the TIE MAC instructions this
    // kernel depends on (confirmed by direct instruction-level bisection —
    // see local-notes/notepads/hematite-nn/problems.md). QEMU builds fall through to
    // the scalar path; real hardware still gets SIMD.
    //
    // `uniform_hint` is cached per params identity (todo 16): the O(output_dim)
    // uniform_scale scan runs once per unique params, so repeated public-API
    // calls skip it.
    #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
    {
        let hint = crate::accx::uniform_hint(
            params as *const _ as usize,
            params.output_multiplier_per_channel,
            params.output_shift_per_channel,
        );
        let mut accx_ctx = FcAccxCtx {
            input,
            weights,
            bias,
            params,
            output,
            scratch,
        };
        if fc_accx_dispatch(&mut accx_ctx, hint)? {
            return Ok(());
        }
    }

    let _ = scratch; // unused by the host path (dispatch is device-only)

    fully_connected_scalar(input, weights, bias, params, output)
}

/// The scalar FC kernel, kept as a separate `#[inline(never)]` function so the
/// public [`fully_connected`] dispatch frame stays thin (an inline scalar loop
/// forced the SIMD path to share a huge frame with register spills — the
/// todo-16 public-API gap). Assumes the caller validated the slice lengths.
#[inline(never)]
fn fully_connected_scalar(
    input: &[i8],
    weights: &[i8],
    bias: &[i32],
    params: &FullyConnectedParams,
    output: &mut [i8],
) -> Result<(), KernelError> {
    let input_dim = params.input_dim as usize;
    let output_dim = params.output_dim as usize;

    // ── Per-channel multiplier / shift slices ───────────────────────────
    let multipliers = params.output_multiplier_per_channel;
    let shifts = params.output_shift_per_channel;

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

    Ok(())
}
