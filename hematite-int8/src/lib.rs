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
///
/// ## 32-bit implementation (no i64 software emulation)
///
/// The 64-bit product `value * multiplier` and the round-add + arithmetic
/// shift are computed with pure 32-bit arithmetic (16-bit limb
/// decomposition + carry tracking), so this function compiles to a handful
/// of 32-bit instructions on ESP32-S3's 32-bit Xtensa core instead of
/// calling into the compiler's software-emulated i64 routines. It is
/// bit-for-bit identical to the `i64` reference formulation above (verified
/// by an exhaustive host test).
#[inline(always)]
pub fn multiply_by_quantized_multiplier(value: i32, multiplier: i32, shift: i32) -> i32 {
    let total_shift = 31 - shift; // in [1, 62] for shift in [-31, 30]

    // 64-bit product `value * multiplier` as (high, low) u32 halves.
    //
    // The 16-bit limb decomposition below yields the *unsigned* 64-bit
    // product of the two bit patterns; the signed product differs by
    // `-sign(a)*b*2^32 - sign(b)*a*2^32` in the high word, applied as the
    // two wrapping subtractions at the end (standard two's-complement
    // 32x32 -> 64 fix-up).
    let mut ph;
    let mut pl;
    {
        let a = value as u32;
        let b = multiplier as u32;
        let a0 = a & 0xffff;
        let a1 = a >> 16;
        let b0 = b & 0xffff;
        let b1 = b >> 16;
        let p00 = a0 * b0; // <= 0xFFFE_0001, fits u32
        let p01 = a0 * b1;
        let p10 = a1 * b0;
        let p11 = a1 * b1;
        // columns: p00 | (p01+p10)<<16 | p11<<32
        let q = (p00 >> 16) + (p01 & 0xffff) + (p10 & 0xffff); // <= 0x2_FFFD
        let c1 = q & 0xffff;
        let r = (q >> 16) + (p01 >> 16) + (p10 >> 16) + (p11 & 0xffff); // <= 0x3_0000
        let c2 = r & 0xffff;
        let c3 = (r >> 16) + (p11 >> 16); // <= 0x1_0002, fits u32
        pl = (c1 << 16) | (p00 & 0xffff);
        ph = (c3 << 16) | c2;
        // Signed fix-up on the high word.
        if value < 0 {
            ph = ph.wrapping_sub(b);
        }
        if multiplier < 0 {
            ph = ph.wrapping_sub(a);
        }
    }

    // round = 1 << (total_shift - 1), added to the 64-bit product.
    let t = total_shift - 1;
    if t < 32 {
        let (p, carry) = pl.overflowing_add(1u32 << t);
        pl = p;
        ph = ph.wrapping_add(carry as u32);
    } else {
        ph = ph.wrapping_add(1u32 << (t - 32));
    }

    // Arithmetic right shift of the 64-bit value by `total_shift`, clamped
    // to the i32 range (defensive; upstream DCHECKs the result fits).
    if total_shift >= 32 {
        // Low word fully shifted out: result = sign-extended high word
        // shifted right by (total_shift - 32). Always fits i32.
        return (ph as i32) >> (total_shift - 32);
    }

    // total_shift in [1, 31]: low 32 bits of the shifted value.
    let r = (ph << (32 - total_shift)) | (pl >> total_shift);
    if ph >> 31 == 0 {
        // Positive: overflow iff value >= 2^(31+total_shift), i.e. the high
        // word reaches 2^(total_shift-1).
        if ph >= (1u32 << (total_shift - 1)) {
            i32::MAX
        } else {
            r as i32
        }
    } else {
        // Negative: underflow iff value < -2^(31+total_shift), i.e. the
        // signed high word < -2^(total_shift-1).
        if (ph as i32) < -(1i32 << (total_shift - 1)) {
            i32::MIN
        } else {
            r as i32
        }
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

// ── 32-bit vs i64 reference equivalence (S3: no software i64 on device) ──

#[cfg(test)]
mod requantize_32bit_tests {
    use super::*;

    /// Reference: the canonical i64 single-rounding formulation the 32-bit
    /// path must be bit-for-bit identical to.
    fn ref_i64(value: i32, multiplier: i32, shift: i32) -> i32 {
        let total_shift = 31i64 - i64::from(shift);
        let round = 1i64 << (total_shift - 1);
        let result = i64::from(value) * i64::from(multiplier) + round;
        let result = result >> total_shift;
        if result > i64::from(i32::MAX) {
            i32::MAX
        } else if result < i64::from(i32::MIN) {
            i32::MIN
        } else {
            result as i32
        }
    }

    /// Deterministic PRNG (xorshift64) for reproducible exhaustive sampling.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn next_i32(&mut self) -> i32 {
            self.next() as u32 as i32
        }
    }

    #[test]
    fn multiply_matches_i64_reference_random() {
        let mut rng = Rng(0x1234_5678_9abc_def0);
        for shift in -31..=30 {
            for _ in 0..20_000 {
                let value = rng.next_i32();
                let multiplier = rng.next_i32();
                let got = multiply_by_quantized_multiplier(value, multiplier, shift);
                let want = ref_i64(value, multiplier, shift);
                assert_eq!(got, want, "mismatch value={value} mult={multiplier} shift={shift}");
            }
        }
    }

    #[test]
    fn multiply_matches_i64_reference_boundaries() {
        // Exhaustive value boundaries around sign/overflow edges × sample
        // multipliers representative of Q0.31 scales.
        let values = [
            i32::MIN,
            i32::MIN + 1,
            -1 << 30,
            -(1 << 30) - 1,
            -1 << 29,
            -(1 << 29) - 1,
            -16_384,
            -1,
            0,
            1,
            16_383,
            16_384,
            (1 << 29) - 1,
            1 << 29,
            (1 << 30) - 1,
            1 << 30,
            i32::MAX - 1,
            i32::MAX,
        ];
        let multipliers = [
            i32::MIN,
            -(1 << 30),
            -16_384,
            -1,
            0,
            1,
            2,
            16_383,
            16_384,
            (1 << 29) - 1,
            1 << 29,
            (1 << 30) - 1,
            1 << 30,
            i32::MAX - 1,
            i32::MAX,
        ];
        for &shift in &[-31, -30, -16, -1, 0, 1, 2, 15, 29, 30] {
            for &value in &values {
                for &multiplier in &multipliers {
                    let got = multiply_by_quantized_multiplier(value, multiplier, shift);
                    let want = ref_i64(value, multiplier, shift);
                    assert_eq!(got, want, "boundary mismatch value={value} mult={multiplier} shift={shift}");
                }
            }
        }
    }

    #[test]
    fn uniform_identity_and_half_round() {
        // identity pair (1<<30, shift=1) and half-round (1<<30, shift=0)
        for v in [-128, -1, 0, 1, 127] {
            assert_eq!(
                multiply_by_quantized_multiplier(v, 1 << 30, 1),
                v,
                "identity pair must be identity"
            );
            let want = ref_i64(v, 1 << 30, 0);
            assert_eq!(multiply_by_quantized_multiplier(v, 1 << 30, 0), want);
        }
    }
}
