// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Reduction ops — scalar fallback + TIE728 SIMD backend.
//!
//! # Bit-exact contract (Plan A4)
//!
//! | Leg | Contract | Runs on |
//! |-----|----------|---------|
//! | (a) | SIMD ≡ per-tensor TFLM golden bit-exact | Device (Phase 5) |
//! | (b) | Scalar ref ≡ per-channel TFLM golden bit-exact | **Host** (this test) |
//! | (c) | SIMD vs ref cross-check ≤1 LSB | Device (Phase 5) |
//!
//! # Ops implemented
//!
//! * [`mean`] — i32 accumulate over reduction axes, divide by count
//!   (round-half-away-from-zero), then per-tensor requantize.

use hematite_core::op_params::ReduceParams;
use hematite_core::KernelError;
use hematite_int8::{multiply_by_quantized_multiplier, saturating_cast};

/// Compute the product of a `[i32; 4]` shape array.
#[inline(always)]
fn shape_product(shape: &[i32; 4]) -> usize {
    shape[0] as usize * shape[1] as usize * shape[2] as usize * shape[3] as usize
}

/// Round-half-away-from-zero integer division.
#[inline(always)]
fn round_half_away_zero(numerator: i32, denominator: i32) -> i32 {
    debug_assert!(denominator > 0, "denominator must be positive");
    let half = denominator / 2;
    if numerator > 0 {
        (numerator + half) / denominator
    } else {
        (numerator - half) / denominator
    }
}

/// Clamp `value` to `[min, max]` and saturating-cast to i8.
#[inline(always)]
fn clamp_i8(value: i32, min: i32, max: i32) -> i8 {
    if value > max {
        saturating_cast(max)
    } else if value < min {
        saturating_cast(min)
    } else {
        saturating_cast(value)
    }
}

/// Reduce-mean — scalar kernel.
///
/// Mirrors `hematite-ref/src/reductions.rs::mean` arithmetic exactly.
///
/// # Algorithm
///
/// 1. i32 accumulate over the reduction axes.
/// 2. Divide by count (round-half-away-from-zero).
/// 3. Requantize via `multiply_by_quantized_multiplier` + output_offset + clamp.
pub fn mean(
    input: &[i8],
    params: &ReduceParams,
    output: &mut [i8],
) -> Result<(), KernelError> {
    if params.input_shape[0] != 1 {
        return Err(KernelError::Unsupported);
    }

    let in_len = shape_product(&params.input_shape);
    let out_len = shape_product(&params.output_shape);
    if input.len() != in_len || output.len() != out_len {
        return Err(KernelError::ShapeMismatch);
    }

    // ── SIMD dispatch (device-only; compiled out entirely on host) ──
    // Bespoke QACC per-lane `s8_mean_reduce` kernel: exact int32 lane sums
    // over the reduced spatial axes, then the bit-exact round-half-away
    // division + requantize in Rust. Bit-exact vs the scalar path below.
    //
    // ALSO gated off under the `qemu` feature: QEMU's xtensa/esp32s3 TIE728
    // emulation does not correctly execute the TIE MAC instructions this
    // kernel depends on. QEMU builds fall through to the scalar path; real
    // hardware still gets SIMD.
    #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
    if mean_accx_dispatch(input, params, output)? {
        return Ok(());
    }

    // Build a boolean mask: which axes are reduced?
    let mut reduce_mask = [false; 4];
    for i in 0..(params.axis_count as usize).min(4) {
        let ax = params.axis[i] as usize;
        if ax < 4 {
            reduce_mask[ax] = true;
        }
    }

    let in_shape = params.input_shape;
    let out_shape = params.output_shape;
    let in_h = in_shape[1] as usize;
    let in_w = in_shape[2] as usize;
    let in_c = in_shape[3] as usize;

    let in_stride_c: usize = 1;
    let in_stride_w: usize = in_c;
    let in_stride_h: usize = in_w * in_c;

    let count_h = if reduce_mask[1] { in_h } else { 1usize };
    let count_w = if reduce_mask[2] { in_w } else { 1usize };
    let count_c = if reduce_mask[3] { in_c } else { 1usize };
    let total_count = (count_h * count_w * count_c) as i32;

    let out_h = out_shape[1] as usize;
    let out_w = out_shape[2] as usize;
    let out_c = out_shape[3] as usize;

    let mult = params.output_multiplier;
    let shift = params.output_shift;
    let out_off = params.output_offset;
    let act_min = params.quantized_activation_min;
    let act_max = params.quantized_activation_max;

    for oh in 0..out_h {
        for ow in 0..out_w {
            for oc in 0..out_c {
                let mut acc: i32 = 0;

                let h_start = if reduce_mask[1] {
                    0
                } else {
                    oh * (in_h / out_h.max(1))
                };
                let h_end = if reduce_mask[1] {
                    in_h
                } else {
                    h_start + 1
                };
                let w_start = if reduce_mask[2] {
                    0
                } else {
                    ow * (in_w / out_w.max(1))
                };
                let w_end = if reduce_mask[2] {
                    in_w
                } else {
                    w_start + 1
                };
                let c_start = if reduce_mask[3] {
                    0
                } else {
                    oc * (in_c / out_c.max(1))
                };
                let c_end = if reduce_mask[3] {
                    in_c
                } else {
                    c_start + 1
                };

                for ih in h_start..h_end {
                    for iw in w_start..w_end {
                        for ic in c_start..c_end {
                            let idx = ih * in_stride_h + iw * in_stride_w + ic * in_stride_c;
                            acc += i32::from(input[idx]);
                        }
                    }
                }

                let averaged = if total_count == 0 {
                    0
                } else {
                    round_half_away_zero(acc, total_count)
                };
                let scaled = multiply_by_quantized_multiplier(averaged, mult, shift);
                let val = (scaled + out_off).max(act_min).min(act_max);
                let out_idx = oh * (out_w * out_c) + ow * out_c + oc;
                output[out_idx] = clamp_i8(val, act_min, act_max);
            }
        }
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// TIE728 SIMD backend — device-only (NEVER compiled on host)
// ─────────────────────────────────────────────────────────────────────────────

/// TIE728 SIMD reduction module.
///
/// **Entirely cfg-gated** — NEVER compiled on host.
///
/// ## Mean SIMD (QACC per-lane, bit-exact)
///
/// For a MEAN that reduces the spatial axes (H,W) while keeping channels —
/// the MobileNetV2 global-average-pool shape — the `s8_mean_reduce` kernel
/// accumulates per-lane via `EE.VMULAS.S8.QACC q0, q1` (q1 = 16×0x01 ones),
/// the same primitive as the depthwise kernel, and recovers the 16 int32
/// lane sums via the verified two-pass QACC read-back. The bit-exact
/// round-half-away division + per-tensor requantize run in Rust from the
/// raw sums (identical arithmetic to the scalar path).
///
/// ## T3.4 — looped accumulation beyond the landed limits
///
/// The landed dispatch limited `positions <= 256` (a QACC 16-bit lane bound)
/// and `in_c <= 256` (the int32 accs scratch). The extension lifts both by
/// looping: positions are chunked into `<= 256`-position `s8_mean_reduce`
/// calls and channels into `<= 256`-wide passes; each call's per-lane i32
/// sums are folded into an `in_c * 4`-byte accs buffer and the kernel is
/// invoked ONCE per chunk with the SAME `s8_mean_reduce` asm. The final
/// round-half-away division + requantize run once, from the accumulated
/// sums. Per-channel i32 sums are order-independent (and `|sum| <=
/// positions * 128` never approaches i32 overflow), so the accumulated total
/// equals the scalar single-pass sum bit-exactly — the MobileNetV2 global
/// mean (7×7×1280 → 1×1×1280: positions 49, in_c 1280) now dispatches SIMD.
#[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
mod reduction_simd {
    use core::arch::{asm, global_asm};

    global_asm!(include_str!("asm/s8_mean_reduce.S"));

    /// Ones vector (16×0x01) for the QACC lane accumulate.
    #[repr(align(16))]
    struct AlignedOnes([i8; 16]);
    static ONES: AlignedOnes = AlignedOnes([1i8; 16]);

    /// MEAN-reduce `[positions, in_c]` → `in_c` int32 lane sums.
    ///
    /// # Safety
    /// `input`/`ones` 16-byte aligned, `acc_out` 4-byte aligned,
    /// `in_c % 16 == 0`, `in_c >= 16`, `positions >= 1`, buffers sized.
    pub unsafe fn mean_reduce(
        input: *const i8,
        acc_out: *mut i32,
        positions: usize,
        in_c: usize,
    ) {
        asm!(
            "call8 s8_mean_reduce",
            in("a10") input,
            in("a11") &ONES.0,
            inout("a12") acc_out => _,
            in("a13") positions,
            in("a14") in_c,
            out("a15") _,
            clobber_abi("C"),
        );
    }
}

// ── T3.4 — looped-accumulation pass plan (host-testable) ─────────────────────
//
// The landed limits (positions <= 256, in_c <= 256) are lifted by looping:
// positions and channels are processed in bounded passes whose per-lane i32
// sums fold into an `in_c * 4`-byte accs buffer, with ONE final
// round-half-away division + requantize. Per-channel sums are
// order-independent i32 additions (|sum| <= positions * 128), so the
// accumulated total equals the scalar single-pass sum bit-exactly. The same
// plan drives the device dispatch (asm per pass) and the host model (scalar
// per pass) so the host matrix proves the device chunking.

/// Maximum spatial positions per `s8_mean_reduce` call — the QACC lanes hold
/// at most `127 * positions`, a signed-16-bit bound (see the asm doc).
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
pub(crate) const MEAN_MAX_POS_PER_CALL: usize = 256;

/// Maximum channels the accs buffer holds — the stack local is
/// `(MEAN_MAX_ACC_C + 16) * 4` bytes (8 KB; `in_c * 4` = 5 KB for the mv2
/// 1280 with headroom). Shapes with more channels fall back to scalar.
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
pub(crate) const MEAN_MAX_ACC_C: usize = 2048;

/// Maximum real channels per kernel call. `in_c <= 256` runs as a single
/// channel pass; larger `in_c` splits into staged channel passes.
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
pub(crate) const MEAN_MAX_C_PASS: usize = 256;

/// Zero-padded channel-staging buffer length (bytes) — every staged pass
/// satisfies `pos_chunk * pad16(pass_c) <= MEAN_STAGE_LEN`.
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
pub(crate) const MEAN_STAGE_LEN: usize = 4096;

/// One `s8_mean_reduce` pass: sum `[pos_chunk, pass_c]` channels starting at
/// input row `pos_off`, channel `c_off`.
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct MeanPass {
    /// First spatial position.
    pub pos_off: usize,
    /// Positions in this pass (`<= MEAN_MAX_POS_PER_CALL`).
    pub pos_chunk: usize,
    /// First channel.
    pub c_off: usize,
    /// Real channels in this pass.
    pub pass_c: usize,
    /// `true` when the kernel must read a zero-padded contiguous stage
    /// instead of the NHWC input directly (padding needed, or a channel pass
    /// whose width differs from the row stride `in_c`).
    pub staged: bool,
}

/// Enumerate the looped-accumulation passes for a spatial-full
/// `[positions, in_c]` MEAN. Positions are chunked to
/// [`MEAN_MAX_POS_PER_CALL`]; `in_c <= MEAN_MAX_C_PASS` runs as one channel
/// pass (direct when `in_c % 16 == 0`, staged when padding is needed),
/// larger `in_c` splits into staged channel passes whose width keeps
/// `pos_chunk * pad16(pass_c) <= MEAN_STAGE_LEN`. Per-channel i32 sums are
/// order-independent, so any pass order reproduces the scalar sum.
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
pub(crate) fn for_each_mean_pass<F>(positions: usize, in_c: usize, mut f: F)
where
    F: FnMut(MeanPass),
{
    fn pad16(n: usize) -> usize {
        (n + 15) & !15
    }
    let mut pos_off = 0;
    while pos_off < positions {
        if in_c <= MEAN_MAX_C_PASS {
            // One channel pass per position chunk.
            let staged = in_c % 16 != 0;
            let pos_chunk = if staged {
                (MEAN_STAGE_LEN / pad16(in_c))
                    .min(MEAN_MAX_POS_PER_CALL)
                    .min(positions - pos_off)
            } else {
                (positions - pos_off).min(MEAN_MAX_POS_PER_CALL)
            };
            f(MeanPass { pos_off, pos_chunk, c_off: 0, pass_c: in_c, staged });
            pos_off += pos_chunk;
        } else {
            // Channel passes — the pass width differs from the row stride
            // `in_c`, so every pass stages a contiguous `[pos_chunk, padded]`
            // block.
            let pos_chunk = (positions - pos_off).min(MEAN_MAX_POS_PER_CALL);
            let pass_cap = ((MEAN_STAGE_LEN / pos_chunk) / 16) * 16;
            let mut c_off = 0;
            while c_off < in_c {
                let pass_c = (in_c - c_off).min(pass_cap);
                f(MeanPass { pos_off, pos_chunk, c_off, pass_c, staged: true });
                c_off += pass_c;
            }
            pos_off += pos_chunk;
        }
    }
}

/// Mean SIMD eligibility + dispatch — device-only.
///
/// Handles the MEAN-over-spatial-axes case bit-exactly:
/// * reduction axes are exactly {H, W} (fully reduced: `out_h == out_w == 1`),
///   channels preserved;
/// * `positions` unbounded — chunked to [`MEAN_MAX_POS_PER_CALL`] passes (the
///   QACC lanes hold `127 * positions` max, a signed-16-bit bound);
/// * `in_c <= MEAN_MAX_ACC_C` — the int32 accs buffer is an `in_c * 4`-byte
///   stack local; `in_c <= MEAN_MAX_C_PASS` runs as a single channel pass
///   (zero-padded when `in_c % 16 != 0`), larger `in_c` splits into staged
///   channel passes.
///
/// The looped accumulation folds each pass's lane sums into the accs buffer
/// and requantizes ONCE at the end — bit-exact vs the scalar mean (per-channel
/// i32 sums are order-independent; `|sum| <= positions * 128` never overflows).
///
/// Returns `Ok(true)` when the SIMD path handled the mean, `Ok(false)` when
/// the shape is ineligible (caller falls through to scalar).
#[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
fn mean_accx_dispatch(
    input: &[i8],
    params: &ReduceParams,
    output: &mut [i8],
) -> Result<bool, KernelError> {
    let in_h = params.input_shape[1] as usize;
    let in_w = params.input_shape[2] as usize;
    let in_c = params.input_shape[3] as usize;
    let out_h = params.output_shape[1];
    let out_w = params.output_shape[2];
    let out_c = params.output_shape[3] as usize;

    let mut reduce_mask = [false; 4];
    for i in 0..(params.axis_count as usize).min(4) {
        let ax = params.axis[i] as usize;
        if ax < 4 {
            reduce_mask[ax] = true;
        }
    }
    let spatial_full = reduce_mask[1] && reduce_mask[2] && !reduce_mask[0] && !reduce_mask[3];
    if !spatial_full || out_h != 1 || out_w != 1 || out_c != in_c {
        return Ok(false);
    }
    let positions = in_h * in_w;
    if positions < 1 || in_c < 1 || in_c > MEAN_MAX_ACC_C {
        return Ok(false);
    }

    let count = positions as i32;
    let mult = params.output_multiplier;
    let shift = params.output_shift;
    let out_off = params.output_offset;
    let act_min = params.quantized_activation_min;
    let act_max = params.quantized_activation_max;

    // Accs buffer: `in_c * 4` bytes (5 KB for the mv2 1280) + 16 guard lanes
    // for a padded tail pass's whole-16-lane-group store. EE.VLD.128 /
    // EE.VST.128 require 16-byte alignment (a misaligned VST silently zeroed
    // accs on device — task-21 finding).
    #[repr(align(16))]
    struct AlignedAccs([i32; MEAN_MAX_ACC_C + 16]);
    // Per-pass kernel destination — the kernel OVERWRITES its acc_out, so the
    // running sums live in `accs` and each pass's sums land here first, then
    // fold into `accs`.
    #[repr(align(16))]
    struct AlignedKernelOut([i32; MEAN_MAX_C_PASS + 16]);
    // Zero-padded channel stage — one stack local reused across all passes (a
    // branch-scoped local is dropped/reused before the asm reads it,
    // producing all-zero lane sums — task-18 device finding).
    #[repr(align(16))]
    struct AlignedStage([i8; MEAN_STAGE_LEN]);
    let mut accs = AlignedAccs([0i32; MEAN_MAX_ACC_C + 16]);
    let mut kernel_out = AlignedKernelOut([0i32; MEAN_MAX_C_PASS + 16]);
    let mut stage = AlignedStage([0i8; MEAN_STAGE_LEN]);

    for_each_mean_pass(positions, in_c, |pass| {
        unsafe {
            if pass.staged {
                let padded = ((pass.pass_c + 15) / 16) * 16;
                stage.0.fill(0);
                for p in 0..pass.pos_chunk {
                    core::ptr::copy_nonoverlapping(
                        input
                            .as_ptr()
                            .add((pass.pos_off + p) * in_c + pass.c_off),
                        stage.0.as_mut_ptr().add(p * padded),
                        pass.pass_c,
                    );
                }
                reduction_simd::mean_reduce(
                    stage.0.as_ptr(),
                    kernel_out.0.as_mut_ptr(),
                    pass.pos_chunk,
                    padded,
                );
            } else {
                reduction_simd::mean_reduce(
                    input.as_ptr().add(pass.pos_off * in_c),
                    kernel_out.0.as_mut_ptr(),
                    pass.pos_chunk,
                    in_c,
                );
            }
        }
        // Fold this pass's lane sums into the running accs.
        for oc in 0..pass.pass_c {
            accs.0[pass.c_off + oc] += kernel_out.0[oc];
        }
    });

    for oc in 0..in_c {
        let acc = accs.0[oc];
        let averaged = if count == 0 { 0 } else { round_half_away_zero(acc, count) };
        let scaled = multiply_by_quantized_multiplier(averaged, mult, shift);
        let val = (scaled + out_off).max(act_min).min(act_max);
        output[oc] = saturating_cast(val);
    }
    SIMD_MEAN_RAN.store(1, core::sync::atomic::Ordering::Relaxed);
    Ok(true)
}

/// Last-`mean` SIMD engagement flag (device diagnostic for simd_validation).
static SIMD_MEAN_RAN: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);

/// Whether the most recent `mean` call took the SIMD path.
///
/// Host/QEMU builds never run the SIMD kernel, so this is always `false`
/// there; on real hardware it flips to `true` after an eligible mean.
pub fn mean_took_simd() -> bool {
    #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
    {
        SIMD_MEAN_RAN.load(core::sync::atomic::Ordering::Relaxed) != 0
    }
    #[cfg(not(all(target_arch = "xtensa", not(feature = "qemu"))))]
    {
        false
    }
}

// ── T3.4 host model + bit-exact matrix (test-only) ───────────────────────────
//
// The device dispatch's looped accumulation is proven bit-exact on the host:
// the model below walks the SAME `for_each_mean_pass` plan the device uses
// (scalar per-pass sums in place of `s8_mean_reduce`), then runs the single
// round-half-away division + requantize. Comparing it against the independent
// `hematite-ref` scalar mean across the acceptance matrix proves the chunking
// never changes the per-channel sum — i32 addition is order-independent.

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    use std::vec;
    use std::vec::Vec;

    /// Host model of the looped-accumulation SIMD mean — mirrors the device
    /// dispatch: the shared [`for_each_mean_pass`] plan, a scalar per-pass
    /// sum, and one final round-half-away division + requantize.
    fn mean_looped_model(input: &[i8], params: &ReduceParams, output: &mut [i8]) {
        let in_h = params.input_shape[1] as usize;
        let in_w = params.input_shape[2] as usize;
        let in_c = params.input_shape[3] as usize;
        let positions = in_h * in_w;
        let count = positions as i32;
        let mut accs = vec![0i32; in_c];
        for_each_mean_pass(positions, in_c, |pass| {
            let base = pass.pos_off * in_c + pass.c_off;
            for p in 0..pass.pos_chunk {
                let row = &input[base + p * in_c..base + p * in_c + pass.pass_c];
                for (oc, &v) in row.iter().enumerate() {
                    accs[pass.c_off + oc] += i32::from(v);
                }
            }
        });
        let mult = params.output_multiplier;
        let shift = params.output_shift;
        let out_off = params.output_offset;
        let act_min = params.quantized_activation_min;
        let act_max = params.quantized_activation_max;
        for oc in 0..in_c {
            let averaged = if count == 0 {
                0
            } else {
                round_half_away_zero(accs[oc], count)
            };
            let scaled = multiply_by_quantized_multiplier(averaged, mult, shift);
            let val = (scaled + out_off).max(act_min).min(act_max);
            output[oc] = saturating_cast(val);
        }
    }

    /// `ReduceParams` for a spatial-full mean over `[1, positions, 1, in_c]`.
    fn mean_params(positions: usize, in_c: usize, mult: i32, shift: i32) -> ReduceParams {
        ReduceParams {
            keep_dims: false,
            axis: [1, 2, 0, 0],
            axis_count: 2,
            input_shape: [1, positions as i32, 1, in_c as i32],
            output_shape: [1, 1, 1, in_c as i32],
            output_type: 0,
            input_offset: 0,
            output_offset: 0,
            output_multiplier: mult,
            output_shift: shift,
            quantized_activation_min: -128,
            quantized_activation_max: 127,
        }
    }

    fn pattern(n: usize, seed: u32) -> Vec<i8> {
        let mut out = vec![0i8; n];
        let mut x = seed;
        for v in out.iter_mut() {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *v = (x >> 16) as i8;
        }
        out
    }

    fn assert_model_matches_ref(positions: usize, in_c: usize, mult: i32, shift: i32, tag: &str) {
        let input = pattern(positions * in_c, 0x1D4A_7EE5 ^ (positions as u32) ^ (in_c as u32));
        let params = mean_params(positions, in_c, mult, shift);
        let mut want = vec![0i8; in_c];
        let mut got = vec![0i8; in_c];
        hematite_ref::reductions::mean(&input, &params, &mut want)
            .unwrap_or_else(|e| panic!("{tag}: ref mean rejected: {e:?}"));
        mean_looped_model(&input, &params, &mut got);
        assert_eq!(
            got, want,
            "{tag}: looped-accumulation model != hematite-ref scalar mean"
        );
    }

    /// T3.4 acceptance matrix: positions {256, 1024, 62720} × in_c {16, 256,
    /// 1280} + the mv2 global-mean shape (49 × 1280), across three quant
    /// variants. Exercises the direct single-pass, position-chunked and
    /// channel-passed plans through the SHARED `for_each_mean_pass`.
    #[test]
    fn mean_extended_simd_model_matches_ref_bit_exact() {
        let quant_cases: &[(i32, i32)] = &[(1 << 30, 1), (1_717_986_918, -3), (1 << 30, 0)];
        for &positions in &[256usize, 1024] {
            for &in_c in &[16usize, 256, 1280] {
                for (qi, &(m, s)) in quant_cases.iter().enumerate() {
                    assert_model_matches_ref(
                        positions,
                        in_c,
                        m,
                        s,
                        &std::format!("positions={positions} in_c={in_c} quant={qi}"),
                    );
                }
            }
        }
        // 62720 × 1280 is an 80 MB tensor — one quant variant bounds the
        // host-test time (the accumulation semantics are quant-independent).
        for &positions in &[62720usize] {
            for &in_c in &[16usize, 256, 1280] {
                assert_model_matches_ref(positions, in_c, 1 << 30, 1, "62720-positions identity");
            }
        }
        // The mv2 global-mean shape (7×7×1280 → 1×1×1280).
        for (qi, &(m, s)) in quant_cases.iter().enumerate() {
            let input = pattern(49 * 1280, 0xBEAD_1EAF);
            let params = ReduceParams {
                keep_dims: false,
                axis: [1, 2, 0, 0],
                axis_count: 2,
                input_shape: [1, 7, 7, 1280],
                output_shape: [1, 1, 1, 1280],
                output_type: 0,
                input_offset: 0,
                output_offset: 0,
                output_multiplier: m,
                output_shift: s,
                quantized_activation_min: -128,
                quantized_activation_max: 127,
            };
            let mut want = vec![0i8; 1280];
            let mut got = vec![0i8; 1280];
            hematite_ref::reductions::mean(&input, &params, &mut want)
                .expect("mv2 global mean: ref shape");
            mean_looped_model(&input, &params, &mut got);
            assert_eq!(
                got, want,
                "mv2 49x1280 looped-accumulation model != ref at quant={qi}"
            );
        }
    }
}
