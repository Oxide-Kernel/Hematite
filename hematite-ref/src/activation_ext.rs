// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Sigmoid / Tanh int8 activations — scalar reference kernels.
//!
//! Implements the TFLM int8 `reference_integer_ops::Logistic` and
//! `reference_integer_ops::Tanh` paths
//! (`tensorflow/lite/kernels/internal/reference/integer_ops/logistic.h` /
//! `tanh.h` at the pinned SHA), which dispatch to the gemmlowp Q4.27
//! fixed-point `logistic` / `tanh`.
//!
//! Mirrors `tools/generate_goldens/src/tflm_math.rs` bit-for-bit
//! (`logistic_q4_27`, `tanh_q4_27`, `one_minus_x_over_one_plus_x_for_x_in_0_1`,
//! and the shared `exp_on_negative_values` / `one_over_one_plus_x_for_x_in_0_1`
//! infrastructure). The same primitives also live privately in `softmax`
//! (softmax uses them on Q5.26); this module carries the Q4.27 activation
//! entry points and the `one_minus_x_over_one_plus_x` variant softmax does
//! not need, so it is self-contained rather than reaching into softmax's
//! private helpers.
//!
//! # Algorithm (sigmoid)
//!
//! 1. `input_val = x - input_offset`
//! 2. Saturate: `input_val <= -radius → -128`, `input_val >= radius → 127`.
//! 3. `input_in_q4 = MultiplyByQuantizedMultiplier(input_val,
//!    input_multiplier, input_left_shift)` — converts to Q4.27.
//! 4. `out_q0 = gemmlowp::logistic(FixedPoint4::FromRaw(input_in_q4))` —
//!    Q0.31.
//! 5. `out_q23 = RoundingDivideByPOT(out_q0, 31 - 8)` (kOutputIntegerBits=8).
//! 6. `out = clamp(out_q23 + output_offset, -128, 127)`.
//!
//! # Algorithm (tanh)
//!
//! 1–3. Identical to sigmoid.
//! 4. `out_q0 = gemmlowp::tanh(FixedPoint4::FromRaw(input_in_q4))` — Q0.31.
//! 5. `out_q24 = RoundingDivideByPOT(out_q0, 31 - 7)` (kOutputScale=7).
//! 6. `out = clamp(out_q24, -128, 127)` — **NO output offset** (tanh
//!    fixture carries `OUTPUT_OFFSET = 0`; the reference tanh adds no
//!    zero-point, unlike sigmoid which adds -128).
//!
//! # Fixed-point note
//!
//! The gemmlowp `logistic_q4_27` / `tanh_q4_27` primitives are pure
//! fixed-point (Q0.31), no floats.

use hematite_core::op_params::ActivationParams;
use hematite_core::KernelError;
use hematite_int8::{multiply_by_quantized_multiplier, rounding_divide_by_pot};

// ── Gemmlowp fixed-point exponential constants (Q0.31) ─────────────────────
//
// Verified against gemmlowp's GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT entries
// in fixedpoint.h at the pinned TFLM SHA. Identical to the sets in
// `tools/generate_goldens/src/tflm_math.rs` and `softmax.rs`.

/// exp(-1/8) as Q0.31
const EXP_NEG_ONE_EIGHTH: i32 = 1_895_147_668;
/// 1/3 as Q0.31
const ONE_THIRD: i32 = 715_827_883;
/// exp(-1/4) as Q0.31
const EXP_NEG_ONE_QUARTER: i32 = 1_672_461_947;
/// exp(-1/2) as Q0.31
const EXP_NEG_ONE_HALF: i32 = 1_302_514_674;
/// exp(-1) as Q0.31
const EXP_NEG_ONE: i32 = 790_015_084;
/// exp(-2) as Q0.31
const EXP_NEG_TWO: i32 = 290_630_308;
/// exp(-4) as Q0.31
const EXP_NEG_FOUR: i32 = 39_332_535;
/// exp(-8) as Q0.31
const EXP_NEG_EIGHT: i32 = 720_401;
/// exp(-16) as Q0.31
const EXP_NEG_SIXTEEN: i32 = 242;

// ── Gemmlowp one_over_one_plus_x constants ─────────────────────────────────

/// 48/17 as Q2.29
const C48_OVER_17: i32 = 1_515_870_810;
/// -32/17 as Q2.29
const CNEG32_OVER_17: i32 = -1_010_580_540;
/// 1.0 as Q2.29
const ONE_Q229: i32 = 1i32 << 29;

// ── Private helpers (mirror tools/generate_goldens/src/tflm_math.rs) ───────

/// `SaturatingRoundingDoublingHighMul(a, b)` — gemmlowp VQRDMULH equivalent.
///
/// Computes the integer nearest to `(a * b) / 2^31`, doubling the product
/// and rounding to nearest. Saturates to `i32::MAX` when both operands are
/// `i32::MIN`.
#[inline(always)]
fn sadhg(a: i32, b: i32) -> i32 {
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

/// Saturating left shift for positive exponents.
///
/// Mirrors `SaturatingRoundingMultiplyByPOT<Exponent>` for `Exponent > 0`
/// (i32 scalar).
#[inline(always)]
fn saturating_rounding_left_shift(x: i32, exponent: i32) -> i32 {
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

/// Rounding half-sum: `(a + b) / 2` rounded to nearest, ties away from zero.
///
/// Mirrors `gemmlowp::RoundingHalfSum` for i32.
#[inline(always)]
fn rounding_half_sum(a: i32, b: i32) -> i32 {
    let a64 = i64::from(a);
    let b64 = i64::from(b);
    let sum = a64 + b64;
    let sign = if sum >= 0 { 1i64 } else { -1i64 };
    ((sum + sign) / 2) as i32
}

/// `exp_on_interval_between_negative_one_quarter_and_0_excl` — gemmlowp
/// 4th-order Taylor expansion of `exp(x)` around `x = -1/8`.
///
/// Input `a` is a Q0.31 value in the range `[-1/4, 0)`. Returns `exp(a)` as
/// Q0.31.
fn exp_on_interval_between_negative_one_quarter_and_0_excl(a: i32) -> i32 {
    // Change of variable: x = a + 1/8 (in Q0.31, 1/8 = 1 << 28)
    let one_eighth = 1i32 << 28;
    let x = a.wrapping_add(one_eighth);

    // Evaluate Taylor series:
    //   x2 = sadhg(x, x)    (Q0)
    //   x3 = sadhg(x2, x)   (Q0)
    //   x4 = sadhg(x2, x2)  (Q0)
    //   x4_over_4 = rounding_divide_by_pot(x4, 2)
    //   inner = rounding_divide_by_pot(
    //       sadhg(x4_over_4 + x3, ONE_THIRD) + x2, 1)
    //   result = EXP_NEG_ONE_EIGHTH + sadhg(EXP_NEG_ONE_EIGHTH, x + inner)

    let x2 = sadhg(x, x);
    let x3 = sadhg(x2, x);
    let x4 = sadhg(x2, x2);

    let x4_over_4 = rounding_divide_by_pot(x4, 2);
    let t1 = x4_over_4.wrapping_add(x3);
    let t2 = sadhg(t1, ONE_THIRD);
    let t3 = t2.wrapping_add(x2);
    // SaturatingRoundingMultiplyByPOT<-1> = right shift by 1 (rounding)
    let inner = rounding_divide_by_pot(t3, 1);

    let poly = x.wrapping_add(inner);
    let term = sadhg(EXP_NEG_ONE_EIGHTH, poly);
    EXP_NEG_ONE_EIGHTH.wrapping_add(term)
}

/// `exp_on_negative_values` — gemmlowp barrel-shifter exponential for `x ≤ 0`.
///
/// Splits the input into integral and fractional parts (modulo 1/4),
/// evaluates the fractional part with
/// `exp_on_interval_between_negative_one_quarter_and_0_excl`, then multiplies
/// in precomputed `exp(-k)` constants for each integral bit set.
///
/// `integer_bits` is 4 for the logistic path (Q4.27 input) and 5 for the
/// tanh path (2·|x| rescaled to Q5.26). Returns `exp(a)` as Q0.31.
///
/// Matches `gemmlowp::exp_on_negative_values<RawType, IntegerBits>`.
fn exp_on_negative_values(a: i32, integer_bits: i32) -> i32 {
    debug_assert!((0..=5).contains(&integer_bits));
    let fractional_bits = 31 - integer_bits;

    // one_quarter in the input's Q format: 1/4 * 2^fractional_bits
    let one_quarter = 1i32 << (fractional_bits - 2);
    let mask = one_quarter - 1;

    // a_mod_quarter_minus_one_quarter = (a & mask) - one_quarter
    // Extracts fractional part modulo 1/4, centered around -1/4
    let a_mod = (a & mask) - one_quarter;

    // Rescale fractional part to Q0.31: LEFT shift by integer_bits.
    let a_mod_q0 = saturating_rounding_left_shift(a_mod, integer_bits);

    let mut result = exp_on_interval_between_negative_one_quarter_and_0_excl(a_mod_q0);

    // remainder = (a_mod - a) gives the integral part of -a (since a ≤ 0)
    let remainder = a_mod - a;

    // Barrel shifter: for each integral bit, multiply by exp(-2^k) if set.
    macro_rules! barrel_shift {
        ($exponent:expr, $constant:ident) => {
            if integer_bits > $exponent {
                let shift = fractional_bits + $exponent;
                let bit_mask = 1i32 << shift;
                if (remainder & bit_mask) != 0 {
                    result = sadhg(result, $constant);
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

    // Handle zero: exp(0) = 1.0 → Q0.31 representation = i32::MAX
    if a == 0 {
        result = i32::MAX;
    }

    result
}

/// `one_over_one_plus_x_for_x_in_0_1` — `1/(1+x)` for `x` in `[0, 1)`.
///
/// Input `a` is Q0.31 in `[0, i32::MAX]`. Returns `1/(1+a)` as Q0.31.
///
/// Uses 3 Newton-Raphson iterations seeded with `48/17 + (a+1)/2 · (-32/17)`,
/// matching gemmlowp's implementation exactly.
fn one_over_one_plus_x_for_x_in_0_1(a: i32) -> i32 {
    // half_denominator = (a + 1) / 2  in Q0.31
    let half_denom_q031 = rounding_half_sum(a, i32::MAX);

    // x₀ = 48/17 + half_denom · (-32/17)   (all Q2.29 after sadhg)
    let term = sadhg(half_denom_q031, CNEG32_OVER_17);
    let mut x: i32 = C48_OVER_17.wrapping_add(term);

    for _ in 0..3 {
        // half_denom (Q0.31) · x (Q2.29) → Q2.29
        let hd_x = sadhg(half_denom_q031, x);
        let one_minus_hd_x = ONE_Q229.wrapping_sub(hd_x);
        // x (Q2.29) · one_minus (Q2.29) → Q4.26, rescale<2> → Q2.29
        let correction = sadhg(x, one_minus_hd_x);
        x = x.wrapping_add(saturating_rounding_left_shift(correction, 2));
    }

    // Rescale<0>(ExactMulByPot<-1>(x)):
    // ExactMulByPot<-1>: raw unmodified (Q2.29 → Q1.29)
    // Rescale<0>: saturatingRoundingMultiplyByPOT<1> → left-shift by 1
    saturating_rounding_left_shift(x, 1)
}

/// `one_minus_x_over_one_plus_x_for_x_in_0_1` — `(1-x)/(1+x)` for `x in [0, 1)`.
///
/// Input `a` is Q0.31 in `[0, i32::MAX]`. Returns `(1-a)/(1+a)` as Q0.31.
///
/// Mirrors `gemmlowp/fixedpoint/fixedpoint.h` exactly: identical Newton-Raphson
/// loop to [`one_over_one_plus_x_for_x_in_0_1`], but the final rescale is
/// `Rescale<0>(x - F2::One())` (left-shift by 2 after subtracting `1 << 29`)
/// instead of `Rescale<0>(ExactMulByPot<-1>(x))`.
fn one_minus_x_over_one_plus_x_for_x_in_0_1(a: i32) -> i32 {
    let half_denom_q031 = rounding_half_sum(a, i32::MAX);

    let term = sadhg(half_denom_q031, CNEG32_OVER_17);
    let mut x: i32 = C48_OVER_17.wrapping_add(term);

    for _ in 0..3 {
        let hd_x = sadhg(half_denom_q031, x);
        let one_minus_hd_x = ONE_Q229.wrapping_sub(hd_x);
        let correction = sadhg(x, one_minus_hd_x);
        x = x.wrapping_add(saturating_rounding_left_shift(correction, 2));
    }

    // Rescale<0>(x - F2::One()): subtract 1<<29 (Q2.29 1.0), then
    // SaturatingRoundingMultiplyByPOT<2> → left shift by 2.
    saturating_rounding_left_shift(x.wrapping_sub(ONE_Q229), 2)
}

// ── Gemmlowp scalar masks (used by the Q4.27 activations) ──────────────────

/// gemmlowp `MaskIfGreaterThan(a, b)` — all-ones if `a > b`, else 0.
#[inline(always)]
fn mask_if_greater_than(a: i32, b: i32) -> i32 {
    if a > b { -1 } else { 0 }
}

/// gemmlowp `MaskIfLessThan(a, b)` — all-ones if `a < b`, else 0.
#[inline(always)]
fn mask_if_less_than(a: i32, b: i32) -> i32 {
    if a < b { -1 } else { 0 }
}

/// gemmlowp `MaskIfZero(a)` — all-ones if `a == 0`, else 0.
#[inline(always)]
fn mask_if_zero(a: i32) -> i32 {
    if a == 0 { -1 } else { 0 }
}

/// gemmlowp `SelectUsingMask(mask, then, else)` — `(mask & then) | (!mask & else)`.
#[inline(always)]
fn select_using_mask(mask: i32, then_val: i32, else_val: i32) -> i32 {
    (mask & then_val) | (!mask & else_val)
}

// ── Gemmlowp Q4.27 logistic / tanh (the int8 activation cores) ─────────────

/// gemmlowp `logistic` on a Q4.27 input — returns `sigmoid(x)` as Q0.31.
///
/// Mirrors `gemmlowp::logistic(FixedPoint<int32_t, 4>)` (the int8 activation
/// path used by `reference_integer_ops::Logistic` at the pinned SHA):
/// mask-and-select on sign, exp via `exp_on_negative_values(-|x|, 4)`, then
/// `1/(1+exp)` (Newton-Raphson) or its mirror, with the `x == 0 → 0.5` special
/// case.
fn logistic_q4_27(input_q427: i32) -> i32 {
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

/// gemmlowp `tanh` on a Q4.27 input — returns `tanh(x)` as Q0.31.
///
/// Mirrors `gemmlowp::tanh(FixedPoint<int32_t, 4>)` (the int8 activation path
/// used by `reference_integer_ops::Tanh` at the pinned SHA): uses the identity
/// `tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))` via
/// `one_minus_x_over_one_plus_x(exp_on_negative_values(2·|x|))`.
fn tanh_q4_27(input_q427: i32) -> i32 {
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

// ── Public API ──────────────────────────────────────────────────────────────

/// Sigmoid (logistic) — scalar reference kernel.
///
/// Implements the TFLM int8 `reference_integer_ops::Logistic` path
/// (gemmlowp Q4.27 logistic), mirroring the fixture generator's arithmetic
/// in `tools/generate_goldens/src/ops/activations.rs` bit-for-bit.
///
/// Quantization params (from the fixture): input scale 1/16 →
/// `input_multiplier = 2^30`, `input_left_shift = 24`,
/// `input_range_radius = 120`, `output_offset = -128`.
///
/// # Errors
///
/// * [`KernelError::ShapeMismatch`] if input and output slice lengths differ.
pub fn sigmoid(
    input: &[i8],
    params: &ActivationParams,
    output: &mut [i8],
) -> Result<(), KernelError> {
    let n = input.len();
    if output.len() != n {
        return Err(KernelError::ShapeMismatch);
    }

    let radius = params.input_range_radius;

    for i in 0..n {
        let input_val = i32::from(input[i]) - params.input_offset;

        // Saturating range check (input outside ±radius clamps directly).
        if input_val <= -radius {
            output[i] = -128;
            continue;
        }
        if input_val >= radius {
            output[i] = 127;
            continue;
        }

        // Convert to Q4.27 fixed-point.
        let input_in_q4 = multiply_by_quantized_multiplier(
            input_val,
            params.input_multiplier,
            params.input_left_shift,
        );
        // Gemmlowp logistic → Q0.31, rescale to Q8.23, add zero-point, clamp.
        let output_in_q0 = logistic_q4_27(input_in_q4);
        let output_in_q23 = rounding_divide_by_pot(output_in_q0, 31 - 8);
        let with_offset = output_in_q23 + params.output_offset;
        output[i] = with_offset.clamp(-128, 127) as i8;
    }

    Ok(())
}

/// Tanh — scalar reference kernel.
///
/// Implements the TFLM int8 `reference_integer_ops::Tanh` path (gemmlowp
/// Q4.27 tanh), mirroring the fixture generator's arithmetic in
/// `tools/generate_goldens/src/ops/activations.rs` bit-for-bit.
///
/// Same quantization params as sigmoid (input scale 1/16 →
/// `input_multiplier = 2^30`, `input_left_shift = 24`,
/// `input_range_radius = 120`), but the reference tanh applies **no output
/// offset** (unlike sigmoid's -128) — `output_offset` is ignored.
///
/// # Errors
///
/// * [`KernelError::ShapeMismatch`] if input and output slice lengths differ.
pub fn tanh(
    input: &[i8],
    params: &ActivationParams,
    output: &mut [i8],
) -> Result<(), KernelError> {
    let n = input.len();
    if output.len() != n {
        return Err(KernelError::ShapeMismatch);
    }

    let radius = params.input_range_radius;

    for i in 0..n {
        let input_val = i32::from(input[i]) - params.input_offset;

        if input_val <= -radius {
            output[i] = -128;
            continue;
        }
        if input_val >= radius {
            output[i] = 127;
            continue;
        }

        let input_in_q4 = multiply_by_quantized_multiplier(
            input_val,
            params.input_multiplier,
            params.input_left_shift,
        );
        // Gemmlowp tanh → Q0.31, rescale to Q8.24, clamp (no zero-point).
        let output_in_q0 = tanh_q4_27(input_in_q4);
        let output_in_q24 = rounding_divide_by_pot(output_in_q0, 31 - 7);
        output[i] = output_in_q24.clamp(-128, 127) as i8;
    }

    Ok(())
}
