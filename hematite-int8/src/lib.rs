#![no_std]

//! # hematite-int8 — Quantization Math Primitives
//!
//! Single source of truth for all quantization arithmetic in the Hematite
//! int8 inference library. Every backend (scalar reference, SIMD) calls
//! these functions.
//!
//! ## Rounding Modes (Plan B4)
//!
//! ### Canonical mode: CMSIS single-rounding (round-half-up at the end)
//!
//! `multiply_by_quantized_multiplier` uses the CMSIS/TFLM single-rounding
//! variant: one rounding step at the end (`+ round) >> total_shift`),
//! which rounds half-way values up (toward positive infinity).
//!
//! **esp-nn's `SKIP_NUDGE` path is a faster, NON-bit-exact variant that we
//! explicitly do NOT use in the reference path.** It skips the rounding
//! nudge entirely, trading accuracy for speed on Xtensa DSPs. All Hematite
//! backends MUST use the canonical single-rounding path documented here.
//!
//! ### Canonical integer division: gemmlowp ties-away-from-zero
//!
//! `rounding_divide_by_pot` uses gemmlowp ties-away-from-zero semantics:
//! halves are rounded away from zero (not toward zero like C's truncating
//! division). This means `-39 / 2 = -20` (not `-19`), and `39 / 2 = 20`
//! (not `19`). This is DIFFERENT from C's `x / (1 << exp)` truncation.
//!
//! ## `host` Feature Gate
//!
//! `quantize_multiplier(scale: f64) -> (i32, i32)` is gated behind the
//! `host` feature. This is a **codegen/host-side helper** — it converts
//! floating-point scales to Q0.31 multiplier/shift pairs at model-load
//! time. It is **never part of device inference**.
//!
//! The default build (no features) contains **zero `f64` code paths**,
//! satisfying Final Verification Wave F1 (no floating-point compute in
//! device code). The `host` feature is only enabled in host-side tools
//! (codegen, golden generation, model loading).

use hematite_core::op_params::PerChannelQuantParam;

/// TFLM's `MultiplyByQuantizedMultiplier` — CMSIS single-rounding variant.
///
/// Per plan T1.2: a single rounding step (round-half-up at the end) rather
/// than the double-rounding SaturatingRoundingDoublingHighMul +
/// RoundingDivideByPOT path from gemmlowp.
///
/// Matches `tensorflow/lite/kernels/internal/common.cc`
/// `TFLITE_SINGLE_ROUNDING`:
///
/// ```text
/// total_shift = 31 - shift;
/// round = 1 << (total_shift - 1);
/// result = (x * multiplier + round) >> total_shift;
/// clamp to i32 range.
/// ```
///
/// The `shift` parameter is in `[-31, 30]` per TFLM's DCHECK.
/// Positive shift means "multiplier > 1" (effectively left-shifts by
/// reducing total_shift); negative shift means "multiplier < 1"
/// (right-shifts more).
///
/// ## Rounding
///
/// Round-half-up at the end: the round value (`1 << (total_shift - 1)`)
/// is added before the final right shift, biasing the result toward
/// positive infinity at the halfway boundary.
#[inline(always)]
pub fn multiply_by_quantized_multiplier(value: i32, multiplier: i32, shift: i32) -> i32 {
    let total_shift = 31i64 - i64::from(shift);
    let round = 1i64 << (total_shift - 1);
    let result = i64::from(value) * i64::from(multiplier) + round;
    let result = result >> total_shift;
    // Saturate to i32 range (the upstream DCHECKs that result fits;
    // we saturate defensively since fixture inputs are synthetic).
    if result > i64::from(i32::MAX) {
        i32::MAX
    } else if result < i64::from(i32::MIN) {
        i32::MIN
    } else {
        result as i32
    }
}

/// `RoundingDivideByPOT(x, exponent)` — from gemmlowp.
///
/// Returns the integer nearest to `x / 2^exponent`, with halves rounded
/// **away from zero** (gemmlowp ties-away-from-zero semantics).
///
/// This is DIFFERENT from C's truncating division `x / (1 << exponent)`,
/// which rounds toward zero. Gemmlowp rounds halves away from zero:
///
/// | x | exp | x / 2^exp | Result | Reason |
/// |---|---|---|---|---|
/// | -39 | 1 | -19.5 | **-20** | away from zero |
/// | 39 | 1 | 19.5 | **20** | away from zero |
/// | -38 | 2 | -9.5 | **-10** | away from zero |
/// | -36 | 3 | -4.5 | **-5** | away from zero |
/// | 4 | 1 | 2.0 | 2 | exact |
/// | -4 | 1 | -2.0 | -2 | exact |
#[inline(always)]
pub fn rounding_divide_by_pot(x: i32, exponent: i32) -> i32 {
    if exponent == 0 {
        return x;
    }
    let mask = (1i32 << exponent).wrapping_sub(1);
    let remainder = x & mask;
    let threshold = (mask >> 1) + i32::from(x < 0);
    (x >> exponent) + i32::from(remainder > threshold)
}

/// Saturating cast from i32 to i8.
///
/// Clamps the value to the `i8` range `[-128, 127]`. Used in requantize
/// epilogues to convert the i32 accumulator result to the int8 output type.
/// This is the saturating clamp TFLM uses in requantize epilogues.
#[inline(always)]
pub fn saturating_cast(value: i32) -> i8 {
    if value > 127 {
        127
    } else if value < -128 {
        -128
    } else {
        value as i8
    }
}

/// Per-channel requantize: multiply accumulator by per-channel quant params
/// and saturate to `i8`.
///
/// Computes `saturating_cast(multiply_by_quantized_multiplier(acc, multiplier, shift))`
/// where `multiplier` and `shift` are taken from `params` for the given
/// `output_channel`.
///
/// ## Contract
///
/// - `output_multiplier_per_channel` and `output_shift_per_channel` slices
///   MUST have equal length (debug_assert in dev builds).
/// - `output_channel` MUST be in bounds for both slices.
/// - This function does NOT add the output zero-point offset — kernels add
///   offsets separately.
/// - This function does NOT apply fused activation clamping — kernels
///   handle activation bounds.
///
/// If `output_channel` is out of bounds in release mode, the function
/// returns `saturating_cast(acc)` as a defensive fallback (no panic).
#[inline(always)]
pub fn requantize(acc: i32, params: &PerChannelQuantParam, output_channel: usize) -> i8 {
    debug_assert_eq!(
        params.output_multiplier_per_channel.len(),
        params.output_shift_per_channel.len(),
        "per-channel multiplier and shift slices must have equal length"
    );
    let len = params.output_multiplier_per_channel.len();
    if output_channel >= len {
        // Out of bounds: defensive fallback (no panic in release)
        return saturating_cast(acc);
    }
    let multiplier = params.output_multiplier_per_channel[output_channel];
    let shift = params.output_shift_per_channel[output_channel];
    let scaled = multiply_by_quantized_multiplier(acc, multiplier, shift);
    saturating_cast(scaled)
}

// ── Host-only: quantize_multiplier (model-load / codegen helper) ──

/// Convert a floating-point scale into a TFLM quantized multiplier + shift pair.
///
/// Matches `tflite::QuantizeMultiplier()` in
/// `tensorflow/lite/kernels/internal/quantization_util.cc`.
///
/// The multiplier is a Q0.31 fixed-point integer in `[0, 2^31)`.
/// The shift encodes the binary exponent: `effective_scale = multiplier / 2^31 * 2^shift`.
///
/// ## Feature Gate
///
/// This function is gated behind the `host` feature because it uses `f64`
/// arithmetic (IEEE 754 bit manipulation via `f64::to_bits()` /
/// `f64::from_bits()` + `f64::round()`). It is a **codegen/host-side helper**
/// — never part of device inference. The default `hematite-int8` build
/// (no features) contains zero `f64` code paths, satisfying Final
/// Verification Wave F1.
#[cfg(feature = "host")]
pub fn quantize_multiplier(scale: f64) -> (i32, i32) {
    if scale == 0.0 {
        return (0, 0);
    }
    let (sig, mut shift) = frexp(scale);
    // Convert significand to Q0.31: sig * 2^31, rounded to nearest
    // `f64::round()` / `f64::floor()` are `std`-only — use `as i64`
    // truncation after adding 0.5, which is equivalent to round() for
    // positive values (sig is always in [0.5, 1.0)).
    let mut q_fixed = (sig * (1u64 << 31) as f64 + 0.5f64) as i64;
    if q_fixed == (1i64 << 31) {
        q_fixed /= 2;
        shift += 1;
    }
    // Flush tiny multipliers to zero
    if shift < -31 {
        return (0, 0);
    }
    (q_fixed as i32, shift)
}

/// Decompose a float64 into significand (in `[0.5, 1.0)`) and an integer
/// exponent, matching `std::frexp` semantics exactly.
///
/// This is a no_std-compatible manual implementation using IEEE 754 bit
/// manipulation (`f64::to_bits()` / `f64::from_bits()`), since `libm` is
/// not a dependency and `std::frexp` is not available in `core`.
#[cfg(feature = "host")]
fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 {
        return (0.0, 0);
    }
    let bits = x.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let mantissa = bits & 0x000f_ffff_ffff_ffff;
    let sign = bits & 0x8000_0000_0000_0000;

    // frexp returns significand in [0.5, 1.0)
    // The IEEE 754 exponent is biased by 1023 and the mantissa has an implicit 1.
    // frexp exponent = (raw_exponent - 1023) + 1 = raw_exponent - 1022
    let frexp_exponent = exponent - 1022;

    // Reconstruct the significand: set exponent to 1022 (so value is in [0.5, 1.0))
    // and put back the mantissa
    let frexp_significand_bits = sign | 0x3fe0_0000_0000_0000u64 | mantissa;
    (f64::from_bits(frexp_significand_bits), frexp_exponent)
}

// ── Host-only unit tests for quantize_multiplier ──

#[cfg(all(test, feature = "host"))]
mod host_tests {
    use super::*;

    #[test]
    fn test_quantize_multiplier_exact_half() {
        // scale = 0.5 → frexp(0.5) = (0.5, 0). sig * 2^31 = 2^30 = 1073741824. shift = 0.
        let (m, s) = quantize_multiplier(0.5);
        assert_eq!(m, 1i32 << 30, "quantize_multiplier(0.5) multiplier");
        assert_eq!(s, 0, "quantize_multiplier(0.5) shift");
    }

    #[test]
    fn test_quantize_multiplier_one() {
        // scale = 1.0 → frexp(1.0) = (0.5, 1) → sig*2^31 = 2^30, shift = 1
        let (m, s) = quantize_multiplier(1.0);
        assert_eq!(m, 1i32 << 30, "quantize_multiplier(1.0) multiplier");
        assert_eq!(s, 1, "quantize_multiplier(1.0) shift");
    }

    #[test]
    fn test_quantize_multiplier_zero() {
        let (m, s) = quantize_multiplier(0.0);
        assert_eq!(m, 0);
        assert_eq!(s, 0);
    }

    #[test]
    fn test_quantize_multiplier_carry_fix() {
        // scale = 0.75 → frexp(0.75) = (0.75, 0). sig*2^31 = 1610612736. shift=0.
        let (m, s) = quantize_multiplier(0.75);
        // 0.75 * 2^31 = 1610612736 → fits in i32, no carry fix needed
        assert_eq!(m, 1610612736);
        assert_eq!(s, 0);
    }

    #[test]
    fn test_quantize_multiplier_identity_roundtrip() {
        // scale 1.0 → (1<<30, 1) → multiply_by_quantized_multiplier with those
        // params should be identity for small integers.
        let (m, s) = quantize_multiplier(1.0);
        for &x in &[-127, -64, -8, -1, 0, 1, 8, 64, 127] {
            assert_eq!(
                multiply_by_quantized_multiplier(x, m, s),
                x,
                "multiply_by_quantized_multiplier({x}, 1<<30, 1) must be identity"
            );
        }
    }

    #[test]
    fn test_quantize_multiplier_tiny_scale_flushed_to_zero() {
        // scale < 2^-32 → flushed to (0, 0)
        let (m, s) = quantize_multiplier(1e-10);
        assert_eq!(m, 0);
        assert_eq!(s, 0);
    }

    #[test]
    fn test_quantize_multiplier_large_scale() {
        // scale = 100.0 → should produce valid multiplier/shift
        let (m, s) = quantize_multiplier(100.0);
        // multiplier must be non-negative and fit in Q0.31 (carry fix
        // prevents overflow; result is always in [0, i32::MAX]).
        assert!(m >= 0, "multiplier must be non-negative, got {m}");
        // spot-check: 100.0 * 2^31 * 2^-7 ≈ 1677721600
        assert!((1i64 << 30) <= i64::from(m) && i64::from(m) <= i64::from(i32::MAX),
            "multiplier {m} not in Q0.31 range");
        // Verify effective scale ≈ 100.0
        let recovered = f64::from(m) / ((1u64 << 31) as f64) * (2.0f64.powi(s));
        assert!((recovered - 100.0).abs() < 1.0,
            "recovered scale {recovered} too far from 100.0, m={m} s={s}");
    }
}
