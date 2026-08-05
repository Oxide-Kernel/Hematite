//! `TFLite` Micro int8 reference arithmetic — precisely mirrors
//! tflite-micro @ 18b9e6f2a8c5a9518e588f59c2ba16ef7ef9d551
//!
//! Every function in this module is a faithful scalar reimplementation
//! of the corresponding TFLM C++ code path.
//! The multiply_by_quantized_multiplier uses the CMSIS single-rounding
//! variant per plan T1.2 (tflite/kernels/internal/common.cc,
//! TFLITE_SINGLE_ROUNDING path).
//! These are NOT the Hematite kernels — they only exist inside this
//! generator to produce reference output values.

/// Convert a floating-point scale into a TFLM quantized multiplier + shift pair.
///
/// Matches `tflite::QuantizeMultiplier()` in
/// `tensorflow/lite/kernels/internal/quantization_util.cc`.
///
/// The multiplier is a Q0.31 fixed-point integer in [0, 2^31).
/// The shift encodes the binary exponent: `effective_scale` = multiplier/2^31 * 2^shift.
pub fn quantize_multiplier(scale: f64) -> (i32, i32) {
    if scale == 0.0 {
        return (0, 0);
    }
    let q = scale;
    // Extract binary exponent via frexp: q = significand * 2^shift
    // where significand is in [0.5, 1.0)
    let (sig, mut shift) = frexp(q);
    // Convert significand to Q0.31: sig * 2^31, rounded to nearest
    let mut q_fixed = (sig * (1u64 << 31) as f64).round() as i64;
    if q_fixed == (1i64 << 31) {
        q_fixed /= 2;
        shift += 1;
    }
    // Flush tiny multipliers to zero
    if shift < -31 {
        return (0, 0);
    }
    let quantized_multiplier = q_fixed as i32;
    (quantized_multiplier, shift)
}

/// Decompose a float64 into significand (in [0.5, 1.0)) and an integer exponent,
/// matching `std::frexp` semantics exactly.
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
    let frexp_significand_bits = sign | (0x3fe0_0000_0000_0000u64) | mantissa;
    (f64::from_bits(frexp_significand_bits), frexp_exponent)
}

/// TFLM's `MultiplyByQuantizedMultiplier` — CMSIS single-rounding variant.
///
/// Per plan T1.2 line 150: a single rounding step (round-half-up at the end)
/// rather than the double-rounding SaturatingRoundingDoublingHighMul +
/// RoundingDivideByPOT path from gemmlowp.
///
/// Matches `tensorflow/lite/kernels/internal/common.cc` `TFLITE_SINGLE_ROUNDING`:
///   total_shift = 31 - shift;
///   round = 1 << (total_shift - 1);
///   result = (x * multiplier + round) >> total_shift;
///   clamp to i32 range.
///
/// The `shift` parameter is in [-31, 30] per TFLM's DCHECK.
/// Positive shift means "multiplier > 1" (effectively left-shifts by reducing
/// total_shift); negative shift means "multiplier < 1" (right-shifts more).
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
/// Returns the integer nearest to x / 2^exponent, with halves rounded away from zero.
pub fn rounding_divide_by_pot(x: i32, exponent: i32) -> i32 {
    if exponent == 0 {
        return x;
    }
    let mask = (1i32 << exponent) - 1;
    let remainder = x & mask;
    let threshold = (mask >> 1) + if x < 0 { 1 } else { 0 };
    (x >> exponent) + if remainder > threshold { 1 } else { 0 }
}

/// Clamp an i32 value to the int8 range, optionally applying fused activation bounds.
pub fn clamp_to_i8(value: i32, activation_min: i32, activation_max: i32) -> i8 {
    value.max(activation_min).min(activation_max) as i8
}

/// Count leading zeros in a u32 — mirrors TFLM's `CountLeadingZeros<uint32_t>`.
/// Returns 32 for input 0.
pub fn count_leading_zeros_u32(x: u32) -> i32 {
    if x == 0 {
        return 32;
    }
    let mut leading = 0i32;
    let mut v = x;
    // Manual count — no std::count_leading_zeros available in no_std environments.
    // This generator is host-only so we could use the intrinsic, but the manual
    // form is zero-dependency and simple.
    while (v & 0x8000_0000) == 0 {
        v <<= 1;
        leading += 1;
    }
    leading
}

/// Integer square root via binary search (no f64, no heap).
///
/// Returns `floor(sqrt(n))` for `n >= 0`. Used by L2_NORMALIZATION
/// to compute the scaling factor from accumulated squared values.
/// The identical algorithm is mirrored in `hematite-ref/src/reductions.rs`.
pub fn integer_sqrt(n: u64) -> u32 {
    if n == 0 {
        return 0;
    }
    let mut low: u32 = 0;
    let mut high: u32 = 0xFFFFu32;
    if n > 1_000_000_000_000_000_000 {
        high = u32::MAX;
    }
    while low < high {
        let mid = low + (high - low) / 2 + (high - low) % 2;
        if u64::from(mid) * u64::from(mid) <= n {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    low
}

/// Saturating left shift for positive exponents.
/// Mirrors `SaturatingRoundingMultiplyByPOT<Exponent>` for Exponent > 0 (int32 scalar).
pub fn saturating_rounding_left_shift(x: i32, exponent: i32) -> i32 {
    if exponent <= 0 {
        return x;
    }
    let threshold = (1i32 << (31 - exponent)) - 1;
    if x > threshold {
        return i32::MAX;
    }
    if x < -threshold {
        return i32::MIN;
    }
    x << exponent
}

/// Rounding half-sum: (a + b) / 2 rounded to nearest, ties away from zero.
/// Mirrors gemmlowp::RoundingHalfSum for i32.
fn rounding_half_sum(a: i32, b: i32) -> i32 {
    let a64 = i64::from(a);
    let b64 = i64::from(b);
    let sum = a64 + b64;
    let sign = if sum >= 0 { 1i64 } else { -1i64 };
    ((sum + sign) / 2) as i32
}

// ── gemmlowp Reciprocal (Newton-Raphson) ──

/// `one_over_one_plus_x_for_x_in_0_1` — 1/(1+x) for x in [0, 1).
///
/// Input `a` is Q0.31 in [0, i32::MAX].
/// Returns 1/(1+a) as Q0.31.
///
/// Uses 3 Newton-Raphson iterations seeded with 48/17 + (a+1)/2 · (-32/17),
/// matching gemmlowp's implementation exactly.
pub fn one_over_one_plus_x_for_x_in_0_1(a: i32) -> i32 {
    // half_denominator = (a + 1) / 2  in Q0.31
    let half_denom_q031 = rounding_half_sum(a, i32::MAX);

    // Q2.29 constants
    const C48_OVER_17: i32 = 1515870810;
    const CNEG32_OVER_17: i32 = -1010580540;
    const ONE_Q229: i32 = 1i32 << 29;

    // x₀ = 48/17 + half_denom · (-32/17)   (all Q2.29 after sadhg)
    let term = saturating_rounding_doubling_high_mul(half_denom_q031, CNEG32_OVER_17);
    let mut x: i32 = C48_OVER_17.wrapping_add(term);

    for _ in 0..3 {
        // half_denom (Q0.31) · x (Q2.29) → Q2.29
        let hd_x = saturating_rounding_doubling_high_mul(half_denom_q031, x);
        let one_minus_hd_x = ONE_Q229.wrapping_sub(hd_x);
        // x (Q2.29) · one_minus (Q2.29) → Q4.26, rescale<2> → Q2.29
        let correction = saturating_rounding_doubling_high_mul(x, one_minus_hd_x);
        x = x.wrapping_add(saturating_rounding_left_shift(correction, 2));
    }

    // Rescale<0>(ExactMulByPot<-1>(x)):
    // ExactMulByPot<-1>: raw unmodified (Q2.29 → Q1.29)
    // Rescale<0>: saturatingRoundingMultiplyByPOT<1> → left-shift by 1
    saturating_rounding_left_shift(x, 1)
}

/// `GetReciprocal` — mirrors TFLM's `tflite::GetReciprocal`.
///
/// Given a raw fixed-point `x` with `x_integer_digits` integer bits,
/// returns a Q0.31 reciprocal and sets `num_bits_over_unit` to the number
/// of bits the original value exceeded 1.0.
pub fn get_reciprocal(x: i32, x_integer_digits: i32, num_bits_over_unit: &mut i32) -> i32 {
    let headroom_plus_one = count_leading_zeros_u32(x as u32);
    *num_bits_over_unit = x_integer_digits - headroom_plus_one;
    let shifted_sum_minus_one = ((x as u32) << headroom_plus_one)
        .wrapping_sub(1u32 << 31) as i32;
    one_over_one_plus_x_for_x_in_0_1(shifted_sum_minus_one)
}

// ── gemmlowp primitives used by softmax ──

/// `SaturatingRoundingDoublingHighMul(a, b)` — gemmlowp VQRDMULH equivalent.
///
/// Computes the integer nearest to `(a * b) / 2^31`, doubling the product
/// (i.e., `2 * a * b / 2^32`) and rounding to nearest. Saturates to i32::MAX
/// when both operands are i32::MIN.
///
/// Matches `gemmlowp::SaturatingRoundingDoublingHighMul` in
/// `gemmlowp/fixedpoint/fixedpoint.h`.
pub fn saturating_rounding_doubling_high_mul(a: i32, b: i32) -> i32 {
    let overflow = a == b && a == i32::MIN;
    let a_64 = i64::from(a);
    let b_64 = i64::from(b);
    let ab_64 = a_64 * b_64;
    let nudge = if ab_64 >= 0 {
        1i64 << 30
    } else {
        1i64 - (1i64 << 30)
    };
    let ab_x2_high32 = ((ab_64 + nudge) / (1i64 << 31)) as i32;
    if overflow {
        i32::MAX
    } else {
        ab_x2_high32
    }
}

/// gemmlowp `MaskIfNonZero` for scalar i32 — returns all-ones if non-zero, zero otherwise.
#[inline]
fn mask_if_non_zero(a: i32) -> i32 {
    if a != 0 { -1i32 } else { 0i32 }
}

// ── gemmlowp fixed-point exponential for softmax ──
//
// These constants are Q0.31 representations verified against gemmlowp's
// GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT entries in fixedpoint.h.
// Each constant name includes the Q format exponent for clarity.

/// exp(-1/8) as Q0.31
const EXP_NEG_ONE_EIGHTH: i32 = 1895147668;
/// 1/3 as Q0.31
const ONE_THIRD: i32 = 715827883;
/// exp(-1/4) as Q0.31
const EXP_NEG_ONE_QUARTER: i32 = 1672461947;
/// exp(-1/2) as Q0.31
const EXP_NEG_ONE_HALF: i32 = 1302514674;
/// exp(-1) as Q0.31
const EXP_NEG_ONE: i32 = 790015084;
/// exp(-2) as Q0.31
const EXP_NEG_TWO: i32 = 290630308;
/// exp(-4) as Q0.31
const EXP_NEG_FOUR: i32 = 39332535;
/// exp(-8) as Q0.31
const EXP_NEG_EIGHT: i32 = 720401;
/// exp(-16) as Q0.31
const EXP_NEG_SIXTEEN: i32 = 242;

/// `exp_on_interval_between_negative_one_quarter_and_0_excl` — gemmlowp 4th-order
/// Taylor expansion of exp(x) around x = -1/8.
///
/// Input `a` is a Q0.31 value in the range [-1/4, 0).
/// Returns exp(a) as Q0.31.
///
/// Matches `gemmlowp::exp_on_interval_between_negative_one_quarter_and_0_excl`
/// in `gemmlowp/fixedpoint/fixedpoint.h`.
pub fn exp_on_interval_between_negative_one_quarter_and_0_excl(a: i32) -> i32 {
    // Change of variable: x = a + 1/8 (in Q0.31, 1/8 = 1 << 28)
    let one_eighth = 1i32 << 28; // Q0.31: 1/8
    let x = a.wrapping_add(one_eighth);

    // Evaluate Taylor series with FixedPoint arithmetic:
    //   x2 = x * x          (Q0)
    //   x3 = x2 * x         (Q0)
    //   x4 = x2 * x2        (Q0)
    //   x4_over_4 = x4 >> 2 (rounding)
    //   inner = ((x4_over_4 + x3) * 1/3 + x2) >> 1 (rounding)
    //   result = constant_term + sadhg(constant_term, x + inner)

    let x2 = saturating_rounding_doubling_high_mul(x, x);
    let x3 = saturating_rounding_doubling_high_mul(x2, x);
    let x4 = saturating_rounding_doubling_high_mul(x2, x2);

    let x4_over_4 = rounding_divide_by_pot(x4, 2);
    let t1 = x4_over_4.wrapping_add(x3);
    let t2 = saturating_rounding_doubling_high_mul(t1, ONE_THIRD);
    let t3 = t2.wrapping_add(x2);
    // SaturatingRoundingMultiplyByPOT<-1> = right shift by 1 (rounding)
    let inner = rounding_divide_by_pot(t3, 1);

    let poly = x.wrapping_add(inner);
    let term = saturating_rounding_doubling_high_mul(EXP_NEG_ONE_EIGHTH, poly);
    EXP_NEG_ONE_EIGHTH.wrapping_add(term)
}

/// `exp_on_negative_values` — gemmlowp barrel-shifter exponential for x ≤ 0.
///
/// Splits the input into integral and fractional parts (modulo 1/4), evaluates the
/// fractional part with exp_on_interval_between_negative_one_quarter_and_0_excl,
/// then multiplies in precomputed exp(-k) constants for each integral bit set.
///
/// Input `a` is in Q5.26 format (5 integer bits, 26 fractional bits).
/// `integer_bits` must be 5 for the standard softmax path.
/// Returns exp(a) as Q0.31.
///
/// Matches `gemmlowp::exp_on_negative_values<RawType, IntegerBits>`.
pub fn exp_on_negative_values(a: i32, integer_bits: i32) -> i32 {
    debug_assert!((0..=5).contains(&integer_bits));
    let fractional_bits = 31 - integer_bits;

    // one_quarter in the input's Q format: Q(integer_bits).(fractional_bits)
    // one_quarter = 1/4 * 2^fractional_bits = 2^(fractional_bits - 2)
    let one_quarter = 1i32 << (fractional_bits - 2);
    let mask = one_quarter - 1;

    // a_mod_quarter_minus_one_quarter = (a & mask) - one_quarter
    // Extracts fractional part modulo 1/4, centered around -1/4
    let a_mod = (a & mask) - one_quarter;

    // Rescale fractional part from Q(integer_bits).(fractional_bits) to Q0.31:
    // gemmlowp Rescale<0>: SaturatingRoundingMultiplyByPOT<integer_bits> — LEFT shift,
    // because moving from more integer bits to fewer integer bits means the raw value
    // must be multiplied by 2^(integer_bits) to represent the same real number.
    // Q5.26 value = a_mod_raw / 2^26. Q0.31 value = result_raw / 2^31.
    // So result_raw = a_mod_raw * 2^(31-26) = a_mod_raw * 2^5 = a_mod_raw << 5.
    let a_mod_q0 = saturating_rounding_left_shift(a_mod, integer_bits);

    let mut result = exp_on_interval_between_negative_one_quarter_and_0_excl(a_mod_q0);

    // remainder = (a_mod - a) gives the integral part of -a (since a ≤ 0)
    // Each set bit at position (fractional_bits + exponent) corresponds to
    // a factor of exp(exponent), and we multiply result by that factor.
    let remainder = a_mod - a;

    // Barrel shifter: for each integral bit, multiply by exp(-2^k) if the bit is set.
    // The bit position in the Q5.26 input that corresponds to exponent e is
    // fractional_bits + e (the fractional_bits lower bits are fractional).
    // For integer_bits=5, fractional_bits=26:
    //   exp(-1/4): check bit 26-2=24  → if set, multiply by exp(-1/4)
    //   exp(-1/2): check bit 26-1=25  → if set, multiply by exp(-1/2)
    //   exp(-1):   check bit 26+0=26  → if set, multiply by exp(-1)
    //   exp(-2):   check bit 26+1=27  → if set, multiply by exp(-2)
    //   exp(-4):   check bit 26+2=28  → if set, multiply by exp(-4)
    //   exp(-8):   check bit 26+3=29  → if set, multiply by exp(-8)
    //   exp(-16):  check bit 26+4=30  → if set, multiply by exp(-16)

    // Each barrel-shifter entry is guarded by `integer_bits > exponent`
    // so that entries beyond the available integer bits are skipped.
    macro_rules! barrel_shift {
        ($exponent:expr, $constant:ident) => {
            if integer_bits > $exponent {
                let shift = fractional_bits + $exponent;
                let bit_mask = 1i32 << shift;
                if (remainder & bit_mask) != 0 {
                    result = saturating_rounding_doubling_high_mul(result, $constant);
                }
            }
        };
    }

    barrel_shift!(-2, EXP_NEG_ONE_QUARTER);
    barrel_shift!(-1, EXP_NEG_ONE_HALF);
    barrel_shift!(0, EXP_NEG_ONE);
    barrel_shift!(1, EXP_NEG_TWO);
    barrel_shift!(2, EXP_NEG_FOUR);
    barrel_shift!(3, EXP_NEG_EIGHT);
    barrel_shift!(4, EXP_NEG_SIXTEEN);

    // Clamp to zero for very negative inputs: if integer_bits > 5, clamp at -32
    if integer_bits > 5 {
        let clamp_b = 36 - integer_bits;
        let clamp = -(1i32 << clamp_b); // -32.0 in Q5.26
        if a < clamp {
            result = 0;
        }
    }

    // Handle zero: exp(0) = 1.0 → Q0.31 representation = i32::MAX (≈0.999...)
    if mask_if_non_zero(a) == 0 {
        result = i32::MAX;
    }

    result
}

/// Per-channel requantize: accumulate → requantize → clamp → output i8.
/// Matches the TFLM conv reference kernel epilogue.
pub fn requantize_i8(
    acc: i32,
    output_multiplier: i32,
    output_shift: i32,
    output_offset: i32,
    activation_min: i32,
    activation_max: i32,
) -> i8 {
    let scaled = multiply_by_quantized_multiplier(acc, output_multiplier, output_shift);
    let with_offset = scaled + output_offset;
    clamp_to_i8(with_offset, activation_min, activation_max)
}

// ── GRU gate helpers: fixed-point sigmoid and tanh ──────────────────────────
//
// These are implemented here (T2.4), NOT in activation.rs, because GRU is
// the sole consumer of int8 sigmoid/tanh gates in this plan. The helpers use
// gemmlowp's exp_on_negative_values + one_over_one_plus_x pipeline (same
// infrastructure as softmax), producing Q0.11 output for the GRU state update.
//
// TFLM has NO GRU kernel at the pinned SHA. These implementations are
// cross-checked against embedded-nn 0.2.1 (see tools/generate_goldens/README.md
// GRU provenance note) and manually verified via self-check assertions
// (sigmoid(0)=0.5, tanh(0)=0, endpoints saturate, monotonicity).

/// Compute sigmoid for a negative Q5.26 value. Returns Q0.31.
///
/// For x <= 0: sigmoid(x) = exp(x) / (1 + exp(x)) = 1 - 1/(1+exp(x))
fn logistic_negative_q031(x_q526: i32) -> i32 {
    debug_assert!(x_q526 <= 0, "logistic_negative_q031: x must be <= 0, got {x_q526}");
    let exp_x = exp_on_negative_values(x_q526, 5);
    let one_over_one_plus_exp = one_over_one_plus_x_for_x_in_0_1(exp_x);
    // sigmoid = 1 - 1/(1+exp)
    i32::MAX - one_over_one_plus_exp
}

/// Fixed-point logistic (sigmoid): computes `1/(1+exp(-x))` in Q0.11.
///
/// Input `x_q526` is in Q5.26 format (5 integer bits, 26 fractional bits).
/// Returns i16 in [0, 2048) representing sigmoid(x) in Q0.11.
///
/// Uses gemmlowp exp_on_negative_values + one_over_one_plus_x for x < 0,
/// and the identity sigmoid(x) = 1 - sigmoid(-x) for x > 0.
pub fn logistic_i16_q011(x_q526: i32) -> i16 {
    let sig_q031 = if x_q526 >= 0 {
        if x_q526 == 0 {
            // sigmoid(0) = 0.5 exactly
            (i32::MAX / 2) + 1
        } else {
            // sigmoid(x) = 1 - sigmoid(-x) for x > 0
            let neg = if x_q526 == i32::MIN { i32::MAX } else { -x_q526 };
            i32::MAX - logistic_negative_q031(neg)
        }
    } else {
        logistic_negative_q031(x_q526)
    };
    // Q0.31 → Q0.11: right-shift by 20, rounding (ties away from zero)
    rounding_divide_by_pot(sig_q031, 20) as i16
}

/// Fixed-point tanh: computes `(exp(x)-exp(-x))/(exp(x)+exp(-x))` in Q0.11.
///
/// Input `x_q526` is in Q5.26 format. Returns i16 in [-2048, 2047]
/// representing tanh(x) in Q0.11, using the identity:
/// `tanh(x) = 2·sigmoid(2x) - 1`.
///
/// The doubling is done in Q5.26 via saturating left-shift before
/// calling `logistic_i16_q011`.
pub fn tanh_i16_q011(x_q526: i32) -> i16 {
    // 2x in Q5.26 (saturating to avoid overflow)
    let two_x = saturating_rounding_left_shift(x_q526, 1);
    let sig_2x = logistic_i16_q011(two_x);
    // tanh = 2·sigmoid(2x) - 1
    // In Q0.11: tanh = sig_2x * 2 - 2048
    (i32::from(sig_2x) * 2 - 2048) as i16
}

// ── gemmlowp scalar masks (used by logistic/tanh activation paths) ──────────

/// gemmlowp `MaskIfGreaterThan(a, b)` for scalar i32 — all-ones if `a > b`, else 0.
#[inline]
fn mask_if_greater_than(a: i32, b: i32) -> i32 {
    if a > b { -1 } else { 0 }
}

/// gemmlowp `MaskIfLessThan(a, b)` for scalar i32 — all-ones if `a < b`, else 0.
#[inline]
fn mask_if_less_than(a: i32, b: i32) -> i32 {
    if a < b { -1 } else { 0 }
}

/// gemmlowp `MaskIfZero(a)` for scalar i32 — all-ones if `a == 0`, else 0.
#[inline]
fn mask_if_zero(a: i32) -> i32 {
    if a == 0 { -1 } else { 0 }
}

/// gemmlowp `SelectUsingMask(mask, then, else)` — `(mask & then) | (~mask & else)`.
#[inline]
fn select_using_mask(mask: i32, then_val: i32, else_val: i32) -> i32 {
    (mask & then_val) | (!mask & else_val)
}

/// gemmlowp `one_minus_x_over_one_plus_x_for_x_in_0_1` — (1-x)/(1+x) for x in [0, 1).
///
/// Input `a` is Q0.31 in [0, i32::MAX]. Returns (1-a)/(1+a) as Q0.31.
/// Mirrors `gemmlowp/fixedpoint/fixedpoint.h` exactly: identical Newton-Raphson
/// loop to [`one_over_one_plus_x_for_x_in_0_1`], but the final rescale is
/// `Rescale<0>(x - F2::One())` (left-shift by 2 after subtracting 1<<29)
/// instead of `Rescale<0>(ExactMulByPot<-1>(x))`.
pub fn one_minus_x_over_one_plus_x_for_x_in_0_1(a: i32) -> i32 {
    let half_denom_q031 = rounding_half_sum(a, i32::MAX);

    const C48_OVER_17: i32 = 1515870810;
    const CNEG32_OVER_17: i32 = -1010580540;
    const ONE_Q229: i32 = 1i32 << 29;

    let term = saturating_rounding_doubling_high_mul(half_denom_q031, CNEG32_OVER_17);
    let mut x: i32 = C48_OVER_17.wrapping_add(term);

    for _ in 0..3 {
        let hd_x = saturating_rounding_doubling_high_mul(half_denom_q031, x);
        let one_minus_hd_x = ONE_Q229.wrapping_sub(hd_x);
        let correction = saturating_rounding_doubling_high_mul(x, one_minus_hd_x);
        x = x.wrapping_add(saturating_rounding_left_shift(correction, 2));
    }

    // Rescale<0>(x - F2::One()): subtract 1<<29 (Q2.29 1.0), then
    // SaturatingRoundingMultiplyByPOT<2> → left shift by 2.
    saturating_rounding_left_shift(x.wrapping_sub(ONE_Q229), 2)
}

/// gemmlowp `logistic` on a Q4.27 input — returns sigmoid(x) as Q0.31.
///
/// Mirrors `gemmlowp::logistic(FixedPoint<int32_t, 4>)` (the int8 activation
/// path used by `reference_integer_ops::Logistic` at the pinned SHA):
/// mask-and-select on sign, exp via `exp_on_negative_values(-|x|, 4)`, then
/// `1/(1+exp)` (Newton-Raphson) or its mirror, with the `x == 0 → 0.5` special
/// case.
pub fn logistic_q4_27(input_q427: i32) -> i32 {
    let mask_if_positive = mask_if_greater_than(input_q427, 0);
    let mask_if_zero = mask_if_zero(input_q427);
    let abs_input = select_using_mask(mask_if_positive, input_q427, -input_q427);

    // logistic_on_positive_values(abs) = 1/(1+exp(-abs))
    let result_if_positive =
        one_over_one_plus_x_for_x_in_0_1(exp_on_negative_values(-abs_input, 4));

    // ResultF::One() for Q0.31 = ScalarRawMax = i32::MAX
    let result_if_negative = i32::MAX.wrapping_sub(result_if_positive);

    // one_half = 0.5 in Q0.31
    let one_half = 1i32 << 30;

    select_using_mask(
        mask_if_zero,
        one_half,
        select_using_mask(mask_if_positive, result_if_positive, result_if_negative),
    )
}

/// gemmlowp `tanh` on a Q4.27 input — returns tanh(x) as Q0.31.
///
/// Mirrors `gemmlowp::tanh(FixedPoint<int32_t, 4>)` (the int8 activation path
/// used by `reference_integer_ops::Tanh` at the pinned SHA): uses the identity
/// `tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))` via
/// `one_minus_x_over_one_plus_x(exp_on_negative_values(2·|x|))`.
pub fn tanh_q4_27(input_q427: i32) -> i32 {
    let mask_if_negative = mask_if_less_than(input_q427, 0);
    let mask_if_zero = mask_if_zero(input_q427);
    let n = select_using_mask(mask_if_negative, input_q427, -input_q427);

    // ExactMulByPot<1> on Q4.27 → Q5.26 (same raw, value doubled) →
    // exp_on_negative_values with integer_bits=5.
    let t = one_minus_x_over_one_plus_x_for_x_in_0_1(exp_on_negative_values(n, 5));

    select_using_mask(
        mask_if_zero,
        0,
        select_using_mask(mask_if_negative, -t, t),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_minus_x_over_one_plus_x_landmarks() {
        // (1-x)/(1+x): x=0 → 1.0 (Q0.31 ≈ i32::MAX); x=1/3 → 0.5; x=1 → 0
        let f = one_minus_x_over_one_plus_x_for_x_in_0_1;
        // x = 0 → 1.0
        let r0 = f(0);
        assert!((r0 as i64 - i32::MAX as i64).abs() < 4096,
            "1/(1) should be ~1.0, got {r0}");
        // x = 1/3 → (2/3)/(4/3) = 0.5 → 1<<30
        let x_third = (i32::MAX as f64 / 3.0) as i32;
        let r1 = f(x_third);
        assert!((r1 - (1i32 << 30)).abs() < 4096,
            "(1-1/3)/(1+1/3)=0.5, got {r1}");
        // x ≈ 1 → ~0
        let x_near_one = i32::MAX - 1;
        let r2 = f(x_near_one);
        assert!((r2 as i64).abs() < 4096, "near-1 input → ~0, got {r2}");
    }

    #[test]
    fn test_logistic_q4_27_matches_f64() {
        // Q4.27 raw → real value = raw / 2^27. Compare against f64 sigmoid.
        for raw in [-3i64 << 27, -2 << 27, -1 << 27, -1 << 25, -1 << 23, 0,
                    1 << 23, 1 << 25, 1 << 27, 2 << 27, 3 << 27] {
            let real = raw as f64 / (1u64 << 27) as f64;
            let expect = 1.0 / (1.0 + f64::exp(-real));
            let got = logistic_q4_27(raw as i32) as f64 / (1u64 << 31) as f64;
            assert!((got - expect).abs() < 1e-3,
                "logistic({real}) = {got}, expected {expect}");
        }
    }

    #[test]
    fn test_logistic_q4_27_zero_is_half() {
        // logistic(0) = 0.5 exactly → 1<<30 in Q0.31
        assert_eq!(logistic_q4_27(0), 1i32 << 30);
    }

    #[test]
    fn test_tanh_q4_27_matches_f64() {
        for raw in [-3i64 << 27, -2 << 27, -1 << 27, -1 << 25, -1 << 23, 0,
                    1 << 23, 1 << 25, 1 << 27, 2 << 27, 3 << 27] {
            let real = raw as f64 / (1u64 << 27) as f64;
            let expect = f64::tanh(real);
            let got = tanh_q4_27(raw as i32) as f64 / (1u64 << 31) as f64;
            assert!((got - expect).abs() < 1e-3,
                "tanh({real}) = {got}, expected {expect}");
        }
    }

    #[test]
    fn test_tanh_q4_27_zero_is_zero() {
        assert_eq!(tanh_q4_27(0), 0);
    }

    #[test]
    fn test_quantize_multiplier_exact_half() {
        // scale = 0.5 → multiplier = 2^31, shift = 0 (since 0.5 = 1.0/2 = 1 * 2^-1, but
        // frexp(0.5) = (0.5, 0) because 0.5 * 2^0 = 0.5, then sig*2^31 = 2^30, shift = 0)
        let (m, s) = quantize_multiplier(0.5);
        // frexp(0.5) = (0.5, 0). sig * 2^31 = 2^30 = 1073741824. shift = 0.
        // Not 2^31 because 0.5 = 1 * 2^-1, frexp gives sig=0.5, exp=0
        assert_eq!(m, 1i32 << 30);
        assert_eq!(s, 0);
    }

    #[test]
    fn test_quantize_multiplier_one() {
        // scale = 1.0 → frexp(1.0) = (0.5, 1) → sig*2^31 = 2^30, shift = 1
        let (m, s) = quantize_multiplier(1.0);
        assert_eq!(m, 1i32 << 30);
        assert_eq!(s, 1);
    }

    #[test]
    fn test_multiply_by_quantized_multiplier_no_shift() {
        // If multiplier = 2^30 and shift = 0, then effective_scale = 2^30/2^31 * 2^0 = 0.5
        // Multiplying 100 by 0.5 should give 50
        let m = 1i32 << 30;
        let result = multiply_by_quantized_multiplier(100, m, 0);
        assert_eq!(result, 50);
    }

    #[test]
    fn test_multiply_rounding_boundary() {
        // Test that rounding boundaries work correctly:
        // 3 * 0.5 = 1.5 → round-half-up to 2
        // 2 * 0.5 = 1.0 → should be exactly 1
        let m = 1i32 << 30; // effective scale = 0.5
        assert_eq!(multiply_by_quantized_multiplier(3, m, 0), 2);
        assert_eq!(multiply_by_quantized_multiplier(2, m, 0), 1);
    }

    #[test]
    fn test_multiply_with_positive_shift() {
        // multiplier = 2^30, shift = 1 → effective scale = 2^30/2^31 * 2^1 = 1.0
        let m = 1i32 << 30;
        let result = multiply_by_quantized_multiplier(42, m, 1);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_multiply_with_negative_shift() {
        // multiplier = 2^30, shift = -1 → effective scale = 2^30/2^31 * 2^-1 = 0.25
        let m = 1i32 << 30;
        let result = multiply_by_quantized_multiplier(100, m, -1);
        assert_eq!(result, 25);
    }

    #[test]
    fn test_requantize_i8_smoke() {
        let result = requantize_i8(100, 1i32 << 30, 0, 0, -128, 127);
        assert_eq!(result, 50);
    }

    // ── Single-rounding boundary tests (plan T1.2) ──

    #[test]
    fn test_single_rounding_negative_accumulator() {
        // -50 * 0.5 = -25.0 exactly — must not drift
        let m = 1i32 << 30;
        assert_eq!(multiply_by_quantized_multiplier(-50, m, 0), -25);
    }

    #[test]
    fn test_single_rounding_boundary_negative_half_up() {
        // -1 * 0.5 = -0.5 → add-round-then-shift biases toward zero → 0
        // (This is the CMSIS/TFLM single-rounding behavior:
        //  (-1 * 2^30 + 2^30) >> 31 = 0 >> 31 = 0)
        let m = 1i32 << 30;
        assert_eq!(multiply_by_quantized_multiplier(-1, m, 0), 0);
    }

    #[test]
    fn test_single_rounding_boundary_positive_half_up() {
        // 1 * 0.5 = 0.5 → round half up → 1
        let m = 1i32 << 30;
        assert_eq!(multiply_by_quantized_multiplier(1, m, 0), 1);
    }

    #[test]
    fn test_single_rounding_negative_scale_gt_1() {
        // effective scale = 2.0 → multiplier = 2^30, shift = 2
        // (Q0.31 * 2^2 / 2^31 = 2.0)
        let m = 1i32 << 30;
        let shift = 2;
        // 5 * 2.0 = 10
        assert_eq!(multiply_by_quantized_multiplier(5, m, shift), 10);
        // -3 * 2.0 = -6
        assert_eq!(multiply_by_quantized_multiplier(-3, m, shift), -6);
    }

    // ── Identity-through-quantize (would have caught DEFECT 4) ──

    #[test]
    fn test_multiply_by_quantized_multiplier_scale_one_is_identity() {
        let (m, s) = quantize_multiplier(1.0);
        assert_eq!(m, 1i32 << 30, "quantize_multiplier(1.0) multiplier");
        assert_eq!(s, 1, "quantize_multiplier(1.0) shift");
        for &x in &[-127, -64, -8, -1, 0, 1, 8, 64, 127] {
            assert_eq!(
                multiply_by_quantized_multiplier(x, m, s),
                x,
                "multiply_by_quantized_multiplier({x}, 1<<30, 1) must be identity"
            );
        }
    }

    // ── RoundingDivideByPOT tests ──

    #[test]
    fn test_rounding_divide_by_pot_exact_zero() {
        assert_eq!(rounding_divide_by_pot(42, 0), 42);
    }

    #[test]
    fn test_rounding_divide_by_pot_positive_half_up() {
        // 3 >> 1 = 1.5 → round half up → 2
        assert_eq!(rounding_divide_by_pot(3, 1), 2);
        // 1 >> 1 = 0.5 → round half up → 1
        assert_eq!(rounding_divide_by_pot(1, 1), 1);
    }

    #[test]
    fn test_rounding_divide_by_pot_positive_exact() {
        // 4 >> 1 = 2.0 exact
        assert_eq!(rounding_divide_by_pot(4, 1), 2);
        // 8 >> 2 = 2.0 exact
        assert_eq!(rounding_divide_by_pot(8, 2), 2);
    }

    #[test]
    fn test_rounding_divide_by_pot_negative_x_neg_39_exp_1() {
        // CONCRETE FAILURE from old formula: -39/2 = -19.5
        // gemmlowp rounds ties away from zero → -20
        assert_eq!(rounding_divide_by_pot(-39, 1), -20);
    }

    #[test]
    fn test_rounding_divide_by_pot_negative_x_neg_38_exp_2() {
        // CONCRETE FAILURE from old formula: -38/4 = -9.5
        // gemmlowp rounds away from zero → -10
        assert_eq!(rounding_divide_by_pot(-38, 2), -10);
    }

    #[test]
    fn test_rounding_divide_by_pot_negative_x_neg_36_exp_3() {
        // CONCRETE FAILURE from old formula: -36/8 = -4.5
        // gemmlowp rounds away from zero → -5
        assert_eq!(rounding_divide_by_pot(-36, 3), -5);
    }

    #[test]
    fn test_rounding_divide_by_pot_negative_not_tie() {
        // -37/2 = -18.5 → away from zero → -19
        assert_eq!(rounding_divide_by_pot(-37, 1), -19);
        // -39/4 = -9.75 → away from zero → -10
        assert_eq!(rounding_divide_by_pot(-39, 2), -10);
    }

    #[test]
    fn test_rounding_divide_by_pot_negative_exact() {
        // -4 >> 1 = -2.0 exact
        assert_eq!(rounding_divide_by_pot(-4, 1), -2);
        // -8 >> 2 = -2.0 exact
        assert_eq!(rounding_divide_by_pot(-8, 2), -2);
    }

    // ── SaturatingRoundingDoublingHighMul tests ──

    #[test]
    fn test_sadhg_two_positive() {
        // sadhg(2^30, 2^30) = round(2 * (2^30 * 2^30) / 2^32) = round(2^60 / 2^31) = 2^29
        let a = 1i32 << 30;
        let result = saturating_rounding_doubling_high_mul(a, a);
        assert_eq!(result, 1i32 << 29);
    }

    #[test]
    fn test_sadhg_one_times_one() {
        // Represent 1.0 as Q0.31 = i32::MAX = 2147483647
        // sadhg(i32::MAX, i32::MAX) ≈ i32::MAX (since 1.0 * 1.0 = 1.0 in Q0.31)
        let one_q031 = i32::MAX;
        let result = saturating_rounding_doubling_high_mul(one_q031, one_q031);
        // 2 * (2^31-1)^2 / 2^32 ≈ 2 * 2^62 / 2^32 = 2^31, close to MAX
        assert!(result >= one_q031 - 1 && result <= one_q031);
    }

    #[test]
    fn test_sadhg_positive_times_negative() {
        // sadhg(2^30, -(2^30)) = round(2 * (-2^60) / 2^32) = round(-2^29) = -(2^29)
        let a = 1i32 << 30;
        let b = -(1i32 << 30);
        let result = saturating_rounding_doubling_high_mul(a, b);
        assert_eq!(result, -(1i32 << 29));
    }

    #[test]
    fn test_sadhg_min_times_min() {
        // i32::MIN * i32::MIN → overflow case, must saturate to i32::MAX
        let result = saturating_rounding_doubling_high_mul(i32::MIN, i32::MIN);
        assert_eq!(result, i32::MAX);
    }

    #[test]
    fn test_sadhg_min_times_one() {
        // i32::MIN * 1 → not the overflow case (only when a == b == MIN)
        // sadhg(i32::MIN, 1): i64 product = -2147483648, nudge = 1-(1<<30), rounds
        let result = saturating_rounding_doubling_high_mul(i32::MIN, 1);
        // Should be approximately -1 in Q0.31
        assert!(result < 0 && result > -(1i32 << 20),
            "sadhg(MIN, 1) = {result}, expected near -1 in Q0.31");
    }

    #[test]
    fn test_sadhg_small_values() {
        // sadhg(100, 200) = round(2 * 20000 / 2^32) = round(40000 / 4294967296) ≈ 0
        assert_eq!(saturating_rounding_doubling_high_mul(100, 200), 0);
        // sadhg(1000000, 2000000) = round(2 * 2e12 / 2^32) = round(4e12 / 4.29e9) ≈ 932
        let r = saturating_rounding_doubling_high_mul(1_000_000, 2_000_000);
        assert!(r > 900 && r < 1000, "sadhg(1M, 2M) = {r}, expected ~932");
    }

    // ── exp_on_negative_values f64-comparison tests ──
    // Gemmlowp's exp has a known accuracy bound: the 4th-order Taylor
    // expansion around -1/8 plus the barrel-shifter approximates exp(x)
    // with relative error within a few percent for the Q5.26 domain.
    // Tolerance: 6% relative, which is generous enough to accommodate
    // the Taylor-series truncation and barrel-shifter quantization but
    // tight enough to catch the factor-of-1.284 defect.

    /// Convert a Q0.31 fixed-point value to f64.
    fn q031_to_f64(x: i32) -> f64 {
        f64::from(x) / ((1u64 << 31) as f64)
    }

    /// Encode a real value x ≤ 0 in Q5.26: round(x * 2^26) as i32.
    fn f64_to_q526(x: f64) -> i32 {
        debug_assert!(x <= 0.0);
        (x * ((1u64 << 26) as f64)).round() as i32
    }

    fn check_exp_q526(real_x: f64, tolerance_fraction: f64) {
        debug_assert!(real_x <= 0.0);
        let q526 = f64_to_q526(real_x);
        let result_q031 = exp_on_negative_values(q526, 5);
        let approx = q031_to_f64(result_q031);
        let expected = f64::exp(real_x);
        let relative_err = if expected > 1e-9 {
            (approx - expected).abs() / expected
        } else {
            (approx - expected).abs()
        };
        assert!(
            relative_err < tolerance_fraction,
            "exp_on_negative_values(Q5.26:{q526} for real {real_x:.6}): \
             Q0.31={result_q031} ≈ {approx:.8}, expected {expected:.8}, \
             relative_err={relative_err:.6} > tol={tolerance_fraction}"
        );
    }

    #[test]
    fn test_exp_at_zero_is_one() {
        let result = exp_on_negative_values(0, 5);
        assert_eq!(result, i32::MAX, "exp(0) must be i32::MAX (Q0.31 one-point-zero)");
    }

    #[test]
    fn test_exp_vs_f64_quarter_multiples() {
        for &x in &[-0.25, -0.5, -0.75, -1.0, -2.0, -4.0, -8.0] {
            check_exp_q526(x, 0.06);
        }
    }

    #[test]
    fn test_exp_vs_f64_non_quarter_values() {
        for &x in &[-0.1, -0.4, -1.5, -3.0] {
            check_exp_q526(x, 0.06);
        }
    }

    #[test]
    fn test_exp_monotonic() {
        let mut prev = i32::MIN;
        for x_q526 in (-8i32..=0).map(|i| f64_to_q526(f64::from(i) * 0.25)) {
            let val = exp_on_negative_values(x_q526, 5);
            assert!(val > prev || (val == i32::MAX && prev == i32::MAX),
                "exp_on_negative_values monotonicity broken at Q5.26 {x_q526}");
            prev = val;
        }
    }

    // ── one_over_one_plus_x_for_x_in_0_1 tests ──

    #[test]
    fn test_one_over_one_plus_x_at_zero() {
        // 1/(1+0) = 1.0 → Q0.31 ≈ i32::MAX
        let result = one_over_one_plus_x_for_x_in_0_1(0);
        assert!(result >= i32::MAX - 2,
            "1/(1+0) must be ~1.0 in Q0.31, got {result}");
    }

    #[test]
    fn test_one_over_one_plus_x_at_half() {
        // Input: 0.5 in Q0.31 = i32::MAX/2 ≈ 1073741823
        // 1/(1+0.5) = 1/1.5 = 0.6667 → Q0.31 ≈ 0.6667 * 2^31 ≈ 1431655765
        let a = i32::MAX / 2; // ~0.5 in Q0.31
        let result = one_over_one_plus_x_for_x_in_0_1(a);
        let expected = ((1.0 / 1.5) * (1u64 << 31) as f64).round() as i32;
        let delta = (result - expected).abs();
        assert!(delta <= 500_000,
            "1/(1+0.5) delta={delta} too large: result={result}, expected≈{expected}");
    }

    #[test]
    fn test_one_over_one_plus_x_near_one() {
        // Input: 0.999 in Q0.31 ≈ 2145487630 (i32::MAX - 2000000)
        // 1/(1+0.999) = 1/1.999 ≈ 0.50025 → Q0.31 ≈ 1074266112
        let a = i32::MAX - 2_000_000;
        let result = one_over_one_plus_x_for_x_in_0_1(a);
        let expected = ((1.0 / 1.999) * (1u64 << 31) as f64).round() as i32;
        let delta = (result - expected).abs();
        assert!(delta <= 500_000,
            "1/(1+0.999) delta={delta} too large: result={result}, expected≈{expected}");
    }

    // ── get_reciprocal tests ──

    #[test]
    fn test_get_reciprocal_q1219_value() {
        // For sum=764865 in Q12.19: actual = 764865/2^19 = 1.4588
        // Reciprocal ≈ 0.6855 → Q0.31 ≈ 1472259096
        let mut nbou = 0i32;
        let result = get_reciprocal(764865, 12, &mut nbou);
        assert_eq!(nbou, 0);
        let expected = ((1.0 / 1.4588) * (1u64 << 31) as f64).round() as i32;
        let delta = (result - expected).abs();
        assert!(delta <= 2_000_000,
            "get_reciprocal(764865,12) delta={delta}: result={result}, expected≈{expected}");
    }
    // ── integer_sqrt tests ──

    #[test]
    fn test_integer_sqrt_smoke() {
        assert_eq!(integer_sqrt(0), 0);
        assert_eq!(integer_sqrt(1), 1);
        assert_eq!(integer_sqrt(2), 1);
        assert_eq!(integer_sqrt(4), 2);
        assert_eq!(integer_sqrt(9), 3);
        assert_eq!(integer_sqrt(15), 3);
        assert_eq!(integer_sqrt(16), 4);
        assert_eq!(integer_sqrt(100), 10);
        assert_eq!(integer_sqrt(10000), 100);
        // u64::MAX = 18446744073709551615; sqrt ~ 4294967295
        let n = u64::MAX;
        let s = integer_sqrt(n);
        assert!(u64::from(s) * u64::from(s) <= n);
        if s < u32::MAX {
            let s_plus = u64::from(s) + 1;
            assert!(s_plus * s_plus > n);
        } // else s == u32::MAX, s+1 squared overflows u64 — cannot check
    }

    #[test]
    fn test_softmax_pipeline_cross_check() {
        let input: [i8; 5] = [-20, -5, 0, 5, 20];
        let input_scale: f64 = 0.1;
        let q526_factor = input_scale * (1u64 << 26) as f64;

        let max_val = *input.iter().max().unwrap();
        let depth = input.len();
        let mut exps_q031 = [0i32; 5];
        let mut sum_q1219: i32 = 0;

        for i in 0..depth {
            let diff = i32::from(input[i]) - i32::from(max_val);
            let diff_q526 = (diff as f64 * q526_factor).round() as i32;
            let exp_q031 = exp_on_negative_values(diff_q526, 5);
            exps_q031[i] = exp_q031;
            let exp_q1219 = rounding_divide_by_pot(exp_q031, 12);
            sum_q1219 = sum_q1219.wrapping_add(exp_q1219);
        }

        assert!(sum_q1219 > 0);

        let mut nbou: i32 = 0;
        let shifted_scale = get_reciprocal(sum_q1219, 12, &mut nbou);
        let exponent = nbou + 23;

        let float_exps: Vec<f64> = input.iter()
            .map(|&v| f64::exp((f64::from(v) - f64::from(max_val)) * input_scale))
            .collect();
        let float_sum: f64 = float_exps.iter().sum();

        for i in 0..depth {
            let scaled_raw = saturating_rounding_doubling_high_mul(shifted_scale, exps_q031[i]);
            let unsat = rounding_divide_by_pot(scaled_raw, exponent);
            let float_val = float_exps[i] / float_sum * 256.0;
            let float_rounded = float_val.round() as i32;
            let delta = (unsat - float_rounded).abs();
            assert!(delta <= 2,
                "softmax element[{i}]: delta={delta} > 2 LSB (gemmlowp={unsat}, float={float_rounded})");
        }
    }
}
