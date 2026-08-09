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
    input_c >= 16 && input_c.is_multiple_of(16) && out_c >= 1
}

/// Host-compilable eligibility for the ACCX 3×3 kernel.
///
/// Any `input_c >= 1` is accepted: when `input_c % 16 != 0` (e.g. a 3-channel
/// first conv) the dispatch zero-pads input and weights in scratch up to the
/// next multiple of 16 and runs the kernel on the padded buffers, so the
/// `% 16` requirement is lifted here.
#[inline]
pub(crate) fn accx_eligible_3x3(input_c: usize, out_c: usize) -> bool {
    input_c >= 1 && out_c >= 1
}

/// Host-compilable eligibility for the bespoke QACC depthwise kernel.
///
/// Any `input_c >= 1` is accepted: when `input_c % 16 != 0` the dispatch
/// zero-pads the input and filter channel dimensions in scratch up to the
/// next multiple of 16 (the kernel VLDs 16-channel vectors and loops
/// `out_c / 16` groups), so the `% 16` requirement is lifted here. Depthwise
/// per-lane semantics require `out_c == input_c` (dm == 1).
#[inline]
pub(crate) fn accx_eligible_depthwise(input_c: usize, out_c: usize) -> bool {
    input_c >= 1 && out_c >= 1 && input_c == out_c
}

/// Compute per-output-channel weight sums into `out`.
///
/// Used to fold a non-zero `input_offset` bit-exactly: the scalar conv
/// computes `acc = Σ (in + offset)·w = Σ in·w + offset·Σw`, and the ACCX
/// kernel produces `Σ in·w`; adding `offset·wsum[oc]` (in wrapping i32, same
/// as the scalar's i32 accumulator) reproduces the reference.
///
/// `weights` is the raw `[oc][tap][ic]` layout (taps = 1 for 1×1 / FC, 9 for
/// 3×3). Host-compilable.
#[inline]
pub(crate) fn weight_sums_conv(out: &mut [i32], weights: &[i8], taps: usize, in_c: usize) {
    for (oc, o) in out.iter_mut().enumerate() {
        let mut s: i32 = 0;
        for t in 0..taps {
            let base = (oc * taps + t) * in_c;
            for ic in 0..in_c {
                s = s.wrapping_add(weights[base + ic] as i32);
            }
        }
        *o = s;
    }
}

/// Depthwise weight sums: the depthwise filter is `[tap][oc]` (HWCN), so
/// `wsum[oc] = Σ_tap weights[tap·out_c + oc]`. Host-compilable.
#[inline]
pub(crate) fn weight_sums_depthwise(out: &mut [i32], weights: &[i8], out_c: usize) {
    for (oc, o) in out.iter_mut().enumerate() {
        let mut s: i32 = 0;
        let mut t = oc;
        while t < weights.len() {
            s = s.wrapping_add(weights[t] as i32);
            t += out_c;
        }
        *o = s;
    }
}
/// Bundled into a single struct passed by `&mut` because the Xtensa LLVM
/// backend miscompiles the multi-arg (9-slot) call site on device — the same
/// class of bug as the `dispatch_fc` inline regression. A 1-arg call is safe.
///
/// `uniform_mult`/`uniform_shift` are the fast-path hint computed once by the
/// dispatcher: when all channels share the same `mult`/`shift`, `requantize_1x1`
/// hoists the fixed-point scale out of the loop (and even skips it entirely for
/// the identity pair). `uniform_shift == i32::MIN` means "per-channel" (the
/// general path).
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
    /// Uniform (mult, shift) shared by every channel, or `(0, i32::MIN)` for
    /// the general per-channel path.
    pub uniform_mult: i32,
    pub uniform_shift: i32,
}

/// Returns `Some((mult, shift))` when every channel shares the same scale.
///
/// `None` (empty or mixed) means the per-channel general path.
#[inline]
pub(crate) fn uniform_scale(multipliers: &[i32], shifts: &[i32]) -> Option<(i32, i32)> {
    let m = *multipliers.first()?;
    let s = *shifts.first()?;
    if multipliers.iter().all(|&x| x == m) && shifts.iter().all(|&x| x == s) {
        Some((m, s))
    } else {
        None
    }
}

#[inline]
fn clamp(v: i32, lo: i32, hi: i32) -> i32 {
    if v > hi {
        hi
    } else if v < lo {
        lo
    } else {
        v
    }
}

/// Shared per-channel requantize epilogue (bit-exact TFLite semantics).
///
/// Applies `bias + Σproducts`, then the per-channel fixed-point requantize,
/// `output_offset`, activation clamp and saturating cast — the exact scalar
/// reference arithmetic.
///
/// Fast paths (selected via `uniform_mult`/`uniform_shift`):
/// * `mult == 1<<30, shift == 1` — identity scale: `scaled == acc`.
/// * `mult == 1<<30, shift == 0` — `(acc + 1) >> 1`, the common identity-mult
///   bench scale (verified bit-identical to
///   `multiply_by_quantized_multiplier(acc, 1<<30, 0)` for all i32 `acc`).
/// * any other uniform pair — the fixed-point scale is hoisted (same i64
///   arithmetic, but `round`/`total_shift` and `mult` live in registers).
///
/// Lengths are validated ONCE up front; the hot loops index unchecked. This
/// removes the four per-iteration slice bounds checks the original loop paid.
///
/// `#[inline(never)]`: when inlined, the Xtensa LLVM backend miscompiles the
/// `accs.iter()` loop bound (the write index runs one past `output.len()`,
/// panicking `index out of bounds`) — same class of bug as the earlier
/// `dispatch_fc` inline regression. Keeping this call separate is required.
#[inline(never)]
pub(crate) fn requantize_1x1(ctx: &mut ReqCtx<'_>) {
    let n = ctx.accs.len();
    assert!(
        n <= ctx.bias.len() && n <= ctx.multipliers.len() && n <= ctx.shifts.len(),
        "requantize: out_c {n} exceeds bias/mult/shift len ({}/{}/{})",
        ctx.bias.len(),
        ctx.multipliers.len(),
        ctx.shifts.len()
    );
    assert!(
        ctx.out_base + n <= ctx.output.len(),
        "requantize: out_base {} + {n} > output.len {}",
        ctx.out_base,
        ctx.output.len()
    );

    let out_offset = ctx.output_offset;
    let act_min = ctx.act_min;
    let act_max = ctx.act_max;
    let out_base = ctx.out_base;

    if ctx.uniform_shift != i32::MIN {
        let mult = ctx.uniform_mult;
        let shift = ctx.uniform_shift;
        // Bespoke asm epilogue for the two bench-critical uniform scales. The
        // asm is bit-identical to the Rust branches below (incl. the
        // overflow-free `(acc+1)>>1`), so swapping it in is safe.
        #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
        if mult == 1 << 30 && (shift == 0 || shift == 1) {
            unsafe { crate::accx::requantize_uniform_asm(ctx) };
            return;
        }
        if mult == 1 << 30 && shift == 1 {
            // Identity scale: scaled == acc (no fixed-point multiply).
            for (oc, &raw) in ctx.accs.iter().enumerate() {
                let acc = raw + unsafe { *ctx.bias.get_unchecked(oc) };
                let c = clamp(acc + out_offset, act_min, act_max);
                unsafe {
                    *ctx.output.get_unchecked_mut(out_base + oc) = saturating_cast(c);
                }
            }
        } else if mult == 1 << 30 && shift == 0 {
            // (acc + 1) >> 1 — exact for all i32 acc (see doc comment).
            for (oc, &raw) in ctx.accs.iter().enumerate() {
                let acc = raw + unsafe { *ctx.bias.get_unchecked(oc) };
                let scaled = ((acc as i64 + 1) >> 1) as i32;
                let c = clamp(scaled + out_offset, act_min, act_max);
                unsafe {
                    *ctx.output.get_unchecked_mut(out_base + oc) = saturating_cast(c);
                }
            }
        } else {
            // General uniform scale — hoisted round/total_shift, same i64
            // arithmetic as multiply_by_quantized_multiplier.
            let total_shift = 31i64 - i64::from(shift);
            let round = 1i64 << (total_shift - 1);
            for (oc, &raw) in ctx.accs.iter().enumerate() {
                let acc = raw + unsafe { *ctx.bias.get_unchecked(oc) };
                let result = i64::from(acc) * i64::from(mult) + round;
                let result = result >> total_shift;
                let scaled = if result > i64::from(i32::MAX) {
                    i32::MAX
                } else if result < i64::from(i32::MIN) {
                    i32::MIN
                } else {
                    result as i32
                };
                let c = clamp(scaled + out_offset, act_min, act_max);
                unsafe {
                    *ctx.output.get_unchecked_mut(out_base + oc) = saturating_cast(c);
                }
            }
        }
    } else {
        // Per-channel mult/shift — the general path.
        for (oc, &raw) in ctx.accs.iter().enumerate() {
            let acc = raw + unsafe { *ctx.bias.get_unchecked(oc) };
            let scaled = multiply_by_quantized_multiplier(
                acc,
                unsafe { *ctx.multipliers.get_unchecked(oc) },
                unsafe { *ctx.shifts.get_unchecked(oc) },
            );
            let c = clamp(scaled + out_offset, act_min, act_max);
            unsafe {
                *ctx.output.get_unchecked_mut(out_base + oc) = saturating_cast(c);
            }
        }
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
    global_asm!(include_str!("asm/s8_requantize.S"));
    global_asm!(include_str!("asm/s8_accx_depthwise.S"));

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

    /// Bespoke asm requantize for the uniform fast path (`mult == 1<<30`,
    /// `shift` 0 or 1). Bit-identical to the Rust loop — the `(acc+1)>>1`
    /// overflow-free form (`srai(acc,1) + (acc&1)`) matches
    /// `((acc as i64 + 1) >> 1) as i32` for every i32 `acc`.
    ///
    /// # Safety
    /// `ctx` must be a valid `&mut ReqCtx` whose `uniform_mult == 1<<30` and
    /// `uniform_shift` is 0 or 1.
    pub unsafe fn requantize_uniform_asm(ctx: *mut super::ReqCtx<'_>) {
        asm!(
            "call8 s8_requantize_uniform",
            in("a10") ctx,
            out("a11") _,
            out("a12") _,
            out("a13") _,
            out("a14") _,
            out("a15") _,
            clobber_abi("C"),
        );
    }

    /// One 3×3 depthwise window (9 taps, depth-multiplier 1) → `out_c` raw
    /// int32 accumulators.
    ///
    /// Depthwise uses the per-lane `EE.VMULAS.S8.QACC` primitive (NOT the ACCX
    /// dot-product reduction, which is wrong for depthwise). Accumulators are
    /// recovered via the QACC two-pass read-back (probe-verified) and written
    /// as 16 int32 lanes per 16-channel group.
    ///
    /// `row_delta` = `(in_w - 3) * in_c` bytes — the offset between 3×3 rows.
    ///
    /// # Safety
    /// `input` (window top-left tap)/`filter` 16-byte aligned, `acc_out`
    /// 4-byte aligned, `in_c % 16 == 0`, `in_c >= 16`, `out_c % 16 == 0`,
    /// `out_c >= 16`, buffers sized.
    pub unsafe fn accx_depthwise(
        input: *const i8,
        filter: *const i8,
        acc_out: *mut i32,
        in_c: usize,
        out_c: usize,
        row_delta: usize,
    ) {
        asm!(
            "call8 s8_accx_depthwise",
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

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;
    use super::*;

    /// Reference requantize exactly as the scalar kernels compute it.
    fn ref_requantize(
        raw: i32,
        bias: i32,
        mult: i32,
        shift: i32,
        out_offset: i32,
        act_min: i32,
        act_max: i32,
    ) -> i8 {
        let acc = raw + bias;
        let scaled = multiply_by_quantized_multiplier(acc, mult, shift);
        let c = clamp(scaled + out_offset, act_min, act_max);
        saturating_cast(c)
    }

    fn run_fast_path(
        accs: &[i32],
        bias: &[i32],
        multipliers: &[i32],
        shifts: &[i32],
        out_offset: i32,
        act_min: i32,
        act_max: i32,
        uniform: Option<(i32, i32)>,
    ) -> Vec<i8> {
        let n = accs.len();
        let mut out = vec![0i8; n];
        let (uniform_mult, uniform_shift) = match uniform {
            Some((m, s)) => (m, s),
            None => (0, i32::MIN),
        };
        requantize_1x1(&mut ReqCtx {
            accs,
            bias,
            multipliers,
            shifts,
            output_offset: out_offset,
            act_min,
            act_max,
            out_base: 0,
            output: &mut out,
            uniform_mult,
            uniform_shift,
        });
        out
    }

    #[test]
    fn requantize_fast_paths_match_reference() {
        // Deterministic pseudo-random-ish accumulators spanning negative,
        // positive, saturating, and near-boundary values.
        let accs: Vec<i32> = (0..64)
            .map(|i| {
                let x = (i as i64 * 2654435761) % (1 << 20) - (1 << 19); // ~±512k
                x as i32
            })
            .collect();
        let bias: Vec<i32> = (0..64).map(|i| (i as i32) * 7919 - 100_000).collect();
        let (min, max) = (-128, 127);

        let cases: Vec<(i32, i32)> = vec![
            (1 << 30, 0), // half-round fast path
            (1 << 30, 1), // identity fast path
            (1 << 30, 2), // hoisted uniform
            (1 << 29, 1), // general uniform
            (1 << 30, -1),
            (1 << 28, 3),
        ];

        for (mult, shift) in cases {
            let muls = vec![mult; 64];
            let shifts = vec![shift; 64];
            let got = run_fast_path(&accs, &bias, &muls, &shifts, 0, min, max, Some((mult, shift)));
            for (oc, &g) in got.iter().enumerate() {
                let want = ref_requantize(accs[oc], bias[oc], mult, shift, 0, min, max);
                assert_eq!(
                    g, want,
                    "uniform ({mult},{shift}) channel {oc}: fast {g} != ref {want}"
                );
            }
        }

        // Per-channel (general) path with mixed multipliers/shifts.
        let muls: Vec<i32> = (0..64).map(|i| (1 << 30) - i * 7919).collect();
        let shifts: Vec<i32> = (0..64).map(|i| if i % 2 == 0 { 0 } else { 1 }).collect();
        let got = run_fast_path(&accs, &bias, &muls, &shifts, 7, -100, 100, None);
        for (oc, &g) in got.iter().enumerate() {
            let want = ref_requantize(accs[oc], bias[oc], muls[oc], shifts[oc], 7, -100, 100);
            assert_eq!(g, want, "per-channel channel {oc}: fast {g} != ref {want}");
        }
    }

    #[test]
    fn uniform_scale_detects_uniformity() {
        assert_eq!(uniform_scale(&[1 << 30; 64], &[0; 64]), Some((1 << 30, 0)));
        assert_eq!(uniform_scale(&[1 << 30; 64], &[1; 64]), Some((1 << 30, 1)));
        assert_eq!(uniform_scale(&[], &[]), None);
        let mut m = [1 << 30; 64];
        m[7] = 123;
        assert_eq!(uniform_scale(&m, &[0; 64]), None);
        let mut s = [0; 64];
        s[3] = 2;
        assert_eq!(uniform_scale(&[1 << 30; 64], &s), None);
    }

    /// The bespoke asm `s8_requantize_uniform` computes `(acc+1)>>1` as
    /// `srai(acc,1) + (acc&1)` — an overflow-free form (a naive add-then-srai
    /// wraps at `i32::MAX`). Prove it is bit-identical to the Rust
    /// `((acc as i64 + 1) >> 1) as i32` for every `i32` accumulator the kernel
    /// can produce, including the exact boundary values.
    #[test]
    fn half_round_overflow_free_form_matches_i64_reference() {
        let mut accs = vec![i32::MIN, i32::MIN + 1, -2, -1, 0, 1, 2, i32::MAX - 1, i32::MAX];
        // LCG sweep over the full i32 space — every lane value a real conv can
        // emit, plus bias arithmetic in i32 (which can wrap).
        let mut state: u64 = 0x9E3779B97F4A7C15;
        for _ in 0..100_000 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let v = ((state >> 33) as u32) as i32;
            let b = ((state >> 1) as u32) as i32;
            let acc = v.wrapping_add(b); // = raw + bias in i32
            accs.push(acc);
        }

        for acc in accs {
            // The asm: srai(acc,1) + (acc&1)
            let asm_form = (acc >> 1).wrapping_add(acc & 1);
            // The Rust reference fast path: ((acc as i64 + 1) >> 1) as i32
            let ref_form = ((acc as i64 + 1) >> 1) as i32;
            assert_eq!(asm_form, ref_form, "overflow-free (acc+1)>>1 diverges for acc={acc}");
        }
    }
}
