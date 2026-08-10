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

/// Mean SIMD eligibility + dispatch — device-only.
///
/// Handles the MEAN-over-spatial-axes case bit-exactly:
/// * reduction axes are exactly {H, W} (fully reduced: `out_h == out_w == 1`),
///   channels preserved;
/// * `positions = in_h * in_w <= 256` — the QACC lanes hold `127*positions`
///   max, a signed-16-bit bound (see the kernel doc);
/// * `in_c <= 256` — the int32 accs scratch fits a bounded stack local;
/// * `in_c % 16 == 0` OR `positions * padded_c <= 4096` so the zero-padded
///   channel staging fits a bounded stack local.
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
    if positions < 1 || positions > 256 || in_c < 1 || in_c > 256 {
        return Ok(false);
    }

    let padded_c = ((in_c + 15) / 16) * 16;
    let needs_channel_pad = padded_c != in_c;
    if needs_channel_pad && positions * padded_c > 4096 {
        return Ok(false);
    }
    let mut accs = [0i32; 256];
    let count = (in_h * in_w) as i32;
    let mult = params.output_multiplier;
    let shift = params.output_shift;
    let out_off = params.output_offset;
    let act_min = params.quantized_activation_min;
    let act_max = params.quantized_activation_max;

    // The padded channel stage must outlive the kernel call — a `padded`
    // local scoped to the branch below would be dropped (stack slot reused)
    // before `mean_reduce` reads it, producing a dangling input pointer
    // (task-18 device finding: all-zero lane sums).
    let mut padded_stage = [0i8; 4096];
    let k_input: *const i8;
    unsafe {
        if needs_channel_pad {
            let src = input.as_ptr();
            let dst = padded_stage.as_mut_ptr();
            for pos in 0..positions {
                core::ptr::copy_nonoverlapping(
                    src.add(pos * in_c),
                    dst.add(pos * padded_c),
                    in_c,
                );
            }
            k_input = padded_stage.as_ptr();
        } else {
            k_input = input.as_ptr();
        }
        reduction_simd::mean_reduce(k_input, accs.as_mut_ptr(), positions, padded_c);
    }

    for oc in 0..in_c {
        let acc = accs[oc];
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
