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
//!
//! # Small / non-16 input dims (T3.6)
//!
//! The FC gate (`input_dim >= 16 && input_dim % 16 == 0`) is widened to any
//! `input_dim >= 1`: when `input_dim % 16 != 0` the dispatch stages a
//! zero-padded input copy AND a zero-padded weight copy (rows padded to the
//! next multiple of 16) in scratch at 16-byte-aligned offsets, runs the same
//! `s8_accx_conv1x1` kernel on the padded buffers, and folds the non-zero
//! `input_offset` via weight sums over the padded rows (pad lanes are zero).
//! Padded lanes contribute `0 × 0 = 0` — the output is bit-exact vs the
//! scalar reference. The staged carve mirrors the conv3x3/depthwise
//! channel-pad path (16-byte alignment; an unaligned staged copy would
//! silently fall back to scalar).

use hematite_core::op_params::FullyConnectedParams;
use hematite_core::KernelError;
use hematite_int8::{multiply_by_quantized_multiplier, saturating_cast};

/// Round a length up to the TIE728 SIMD group width (16 lanes).
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
#[inline(always)]
const fn pad16(n: usize) -> usize {
    (n + 15) & !15
}

/// Stage a zero-padded FC input (real `input[0..input_dim]`, then zeros) and
/// the zero-padded weight rows (`weights` is `output_dim × input_dim` raw
/// `[oc][ic]`; the staged copy is `output_dim × padded_dim` with each row
/// zero-filled past `input_dim`).
///
/// The staged buffers are what the device dispatch hands to `s8_accx_conv1x1`
/// when `input_dim % 16 != 0`: the kernel VLDs 16-lane vectors and strides
/// weight rows by the padded input dim, so both staged buffers must be padded
/// to a multiple of 16. Padded lanes multiply `0 × 0 = 0`, and the Phase-C
/// `input_offset` fold reads weight sums over the padded rows — pad lanes are
/// zero, so the sums equal the real per-row sums — bit-exact vs the scalar
/// `Σ (in + offset)·w` loop. Host-compilable so the unit tests exercise the
/// real device-pipeline staging.
///
/// # Panics
/// `dst_in` / `dst_w` must be exactly `padded_dim` / `output_dim * padded_dim`
/// bytes (caller-computed via [`pad16`]); this is asserted.
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
pub(crate) fn stage_fc_padded(
    dst_in: &mut [u8],
    dst_w: &mut [i8],
    input: &[i8],
    weights: &[i8],
    input_dim: usize,
    output_dim: usize,
) {
    let padded_dim = pad16(input_dim);
    assert_eq!(dst_in.len(), padded_dim, "stage_fc_padded: dst_in len");
    assert_eq!(
        dst_w.len(),
        output_dim * padded_dim,
        "stage_fc_padded: dst_w len"
    );
    for (d, &x) in dst_in[..input_dim].iter_mut().zip(input.iter()) {
        *d = x as u8; // bit-preserving i8→u8 re-interpret (VLD reads i8 lanes)
    }
    dst_in[input_dim..].fill(0);
    for oc in 0..output_dim {
        let row = &mut dst_w[oc * padded_dim..(oc + 1) * padded_dim];
        row[..input_dim].copy_from_slice(&weights[oc * input_dim..(oc + 1) * input_dim]);
        row[input_dim..].fill(0);
    }
}

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

    if !crate::accx::accx_eligible_1x1_padded(input_dim, output_dim) {
        return Ok(false);
    }

    let input_offset = params.input_offset;
    // T3.6 — small / non-16 input dims: stage a zero-padded input copy AND a
    // zero-padded weight copy (the kernel VLDs 16-lane vectors and strides
    // weight rows by the padded input dim), then run the same kernel.
    let padded_dim = pad16(input_dim);
    let needs_pad = padded_dim != input_dim;
    // Padded layout (mirrors the conv3x3/depthwise channel-pad carve):
    //   [padded input: padded_dim][padded weights: output_dim*padded_dim][accs: output_dim*4][wsum: output_dim*4 if input_offset != 0]
    let pad_input_len = padded_dim;
    let pad_weights_len = output_dim * padded_dim;
    let wsum_extra = if input_offset != 0 { output_dim * 4 } else { 0 };
    let need = if needs_pad {
        pad_input_len + pad_weights_len + output_dim * 4 + wsum_extra
    } else {
        output_dim * 4 + wsum_extra
    };
    if ctx.scratch.len() < need {
        return Ok(false);
    }

    let in_ptr = ctx.input.as_ptr();
    let w_ptr = ctx.weights.as_ptr();
    let out_ptr = ctx.output.as_mut_ptr();
    let scratch_ptr = ctx.scratch.as_mut_ptr();
    let scratch_u = scratch_ptr as usize;

    let (k_in_ptr, k_w_ptr, accs, wsum);
    if needs_pad {
        // Padded buffers — carve from scratch at 16-byte boundaries so the
        // kernel's VLD.128 stays aligned (mirrors conv3x3.rs:180-195).
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
            (unsafe { scratch_ptr.add(accs_off - scratch_u + output_dim * 4) }) as *mut i32
        } else {
            core::ptr::null_mut()
        };
        let dst_in = unsafe {
            core::slice::from_raw_parts_mut(p_in as *mut u8, pad_input_len)
        };
        let dst_w = unsafe { core::slice::from_raw_parts_mut(p_w as *mut i8, pad_weights_len) };
        stage_fc_padded(
            dst_in,
            dst_w,
            ctx.input,
            ctx.weights,
            input_dim,
            output_dim,
        );
        k_in_ptr = p_in;
        k_w_ptr = p_w;
        accs = p_accs;
    } else {
        if (in_ptr as usize) % 16 != 0
            || (w_ptr as usize) % 16 != 0
            || (out_ptr as usize) % 16 != 0
        {
            return Ok(false);
        }
        accs = scratch_ptr as *mut i32;
        if (accs as usize) % 4 != 0 {
            return Ok(false);
        }
        wsum = if input_offset != 0 {
            unsafe { accs.add(output_dim) }
        } else {
            core::ptr::null_mut()
        };
        k_in_ptr = in_ptr;
        k_w_ptr = w_ptr;
    }
    if input_offset != 0 {
        let ws = unsafe { core::slice::from_raw_parts_mut(wsum, output_dim) };
        let wv = unsafe { core::slice::from_raw_parts(k_w_ptr, output_dim * padded_dim) };
        crate::accx::weight_sums_conv(ws, wv, 1, padded_dim);
    }

    let multipliers = params.output_multiplier_per_channel;
    let shifts = params.output_shift_per_channel;
    let act_min = params.quantized_activation_min;
    let act_max = params.quantized_activation_max;
    let out_offset = params.output_offset;
    let (uniform_mult, uniform_shift) = uniform;

    unsafe {
        crate::accx::accx_conv1x1(k_in_ptr, k_w_ptr, accs, padded_dim, output_dim);
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
            && crate::accx::accx_eligible_1x1_padded(input_dim, output_dim);
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

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;
    use super::*;
    use hematite_core::op_params::FullyConnectedParams;

    /// Host model of the `s8_accx_conv1x1` accumulation contract on the
    /// staged (padded) buffers: `acc[oc] = Σ_ic staged_w[oc*padded_dim + ic] *
    /// staged_in[ic]` in wrapping i32 — the exact GPR-accumulator arithmetic
    /// the asm uses (raw dot product, no input_offset).
    fn kernel_model_accs(staged_in: &[u8], staged_w: &[i8], padded_dim: usize, out_c: usize) -> Vec<i32> {
        let mut accs = vec![0i32; out_c];
        for oc in 0..out_c {
            let mut acc: i32 = 0;
            for ic in 0..padded_dim {
                let iv = i32::from(staged_in[ic] as i8);
                let wv = i32::from(staged_w[oc * padded_dim + ic]);
                acc = acc.wrapping_add(iv.wrapping_mul(wv));
            }
            accs[oc] = acc;
        }
        accs
    }

    /// Run the full device SIMD pipeline in software — real
    /// [`stage_fc_padded`] staging, the kernel-model accumulators, the real
    /// Phase-C `input_offset` fold, and the real `requantize_1x1` epilogue —
    /// producing one FC output layer. This exercises the exact device
    /// pipeline code (pad + kernel contract + fold + requantize).
    fn simd_model_layer(
        input: &[i8],
        weights: &[i8],
        bias: &[i32],
        p: &FullyConnectedParams<'_>,
    ) -> Vec<i8> {
        let input_dim = p.input_dim as usize;
        let output_dim = p.output_dim as usize;
        let padded_dim = pad16(input_dim);
        let needs_pad = padded_dim != input_dim;

        let mut staged_in = vec![0u8; padded_dim];
        let mut staged_w = vec![0i8; output_dim * padded_dim];
        if needs_pad {
            stage_fc_padded(
                &mut staged_in,
                &mut staged_w,
                input,
                weights,
                input_dim,
                output_dim,
            );
        } else {
            for (d, &x) in staged_in[..input_dim].iter_mut().zip(input.iter()) {
                *d = x as u8; // bit-preserving i8→u8 re-interpret
            }
            staged_w.copy_from_slice(weights);
        }

        let mut accs = kernel_model_accs(&staged_in, &staged_w, padded_dim, output_dim);
        if p.input_offset != 0 {
            // Weight sums over the PADDED rows — pad lanes are zero, so these
            // equal the real per-row sums (the dispatch computes them this
            // way; mirror exactly).
            let mut wsum = vec![0i32; output_dim];
            crate::accx::weight_sums_conv(&mut wsum, &staged_w, 1, padded_dim);
            crate::depthwise::fold_input_offset(&mut accs, &wsum, p.input_offset);
        }

        let multipliers = p.output_multiplier_per_channel;
        let shifts = p.output_shift_per_channel;
        let (uniform_mult, uniform_shift) = match crate::accx::uniform_scale(multipliers, shifts) {
            Some((m, s)) => (m, s),
            None => (0, i32::MIN),
        };
        let mut output = vec![0i8; output_dim];
        crate::accx::requantize_1x1(&mut crate::accx::ReqCtx {
            accs: &accs,
            bias,
            multipliers,
            shifts,
            output_offset: p.output_offset,
            act_min: p.quantized_activation_min,
            act_max: p.quantized_activation_max,
            out_base: 0,
            output: &mut output,
            uniform_mult,
            uniform_shift,
        });
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

    /// Host bit-exact gate (T3.6): the device SIMD pipeline model (real
    /// staging + kernel-contract accumulation + real fold + real requantize)
    /// must equal the independent `hematite-ref` scalar fully_connected for
    /// every small / non-16 input dim in {1, 3, 8, 15, 17, 32} — pad and
    /// no-pad paths — across output dims, offsets, and identity /
    /// non-identity per-channel multipliers. Zero mismatches.
    #[test]
    fn fc_small_simd_model_matches_ref_bit_exact() {
        let mut checked = 0;
        for &input_dim in &[1, 3, 8, 15, 17, 32] {
            for &out_dim in &[1, 16, 128] {
                for &in_off in &[0, 5, 128] {
                    for mode in 0..3 {
                        let n = out_dim as usize;
                        let (mults, shifts): (Vec<i32>, Vec<i32>) = match mode {
                            0 => (vec![1 << 30; n], vec![1; n]),
                            1 => (per_channel_mult(n), per_channel_shift(n)),
                            _ => (vec![1 << 29; n], vec![0; n]),
                        };
                        let p = FullyConnectedParams {
                            input_dim,
                            output_dim: out_dim,
                            input_offset: in_off,
                            weights_offset: 0,
                            output_offset: if in_off == 0 { 0 } else { -10 },
                            output_multiplier_per_channel: &mults,
                            output_shift_per_channel: &shifts,
                            quantized_activation_min: if mode == 1 { 0 } else { -128 },
                            quantized_activation_max: 127,
                        };
                        let seed = 0x3C60_0000u32 | (input_dim as u32 * 31 + out_dim as u32);
                        let input = pattern(seed, input_dim as usize);
                        let weights =
                            pattern(0xD3A + input_dim as u32 * 17, input_dim as usize * n);
                        let bias = pattern_i32(0xFAC + out_dim as u32, n);

                        let got = simd_model_layer(&input, &weights, &bias, &p);
                        let mut want = vec![0i8; got.len()];
                        hematite_ref::fully_connected::fully_connected(
                            &input,
                            &weights,
                            &bias,
                            &p,
                            &mut want,
                            &mut [],
                        )
                        .expect("ref fc accepts the shape");
                        assert_eq!(
                            got, want,
                            "input_dim={input_dim} out_dim={out_dim} in_off={in_off} mode={mode}: \
                             SIMD-model output must equal hematite-ref scalar"
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked >= 160, "small-fc matrix did not expand ({checked})");
    }

    /// The staging must produce the exact padded layout the kernel consumes:
    /// input `[real | zeros]`, weight rows `[real row | zeros]`.
    #[test]
    fn stage_fc_padded_zero_fills_pad_lanes() {
        let input: Vec<i8> = (0..8).map(|i| i as i8 - 3).collect(); // input_dim = 8
        let weights: Vec<i8> = (0..32).map(|i| (i % 7) as i8 - 2).collect(); // 4 rows x 8
        let mut dst_in = vec![0xEEu8; 16];
        let mut dst_w = vec![0x7Fi8; 64]; // 4 rows x 16
        stage_fc_padded(&mut dst_in, &mut dst_w, &input, &weights, 8, 4);
        let expect_in: Vec<u8> = input.iter().map(|&x| x as u8).collect();
        assert_eq!(&dst_in[..8], &expect_in[..]);
        assert_eq!(&dst_in[8..], &[0; 8]);
        for oc in 0..4 {
            let row = &dst_w[oc * 16..(oc + 1) * 16];
            assert_eq!(&row[..8], &weights[oc * 8..(oc + 1) * 8]);
            assert_eq!(&row[8..], &[0; 8], "row {oc} pad lanes must be zero");
        }
    }
}
