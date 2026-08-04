// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Int8-safe softmax — scalar reference kernel.
//!
//! Mirrors the TFLM/gemmlowp fixed-point exponential algorithm used by
//! the golden fixture generator at `tools/generate_goldens/src/ops/softmax.rs`.
//!
//! # Algorithm
//!
//! 1. Max-subtract per row (numerical stability).
//! 2. `diff_min` threshold skip (below-threshold elements → output_min).
//! 3. Q5.26 input scaling via `SaturatingRoundingDoublingHighMul` +
//!    left-shift.
//! 4. Gemmlowp barrel-shifter exponential (`exp_on_negative_values`).
//! 5. Accumulate row sum in Q12.19 (`Rescale<kAccumulationIntegerBits>`).
//! 6. `GetReciprocal` via Newton-Raphson (gemmlowp `one_over_one_plus_x`).
//! 7. Normalize: `exp * 1/sum` → `RoundingDivideByPOT` + output_offset +
//!    saturating clamp.
//!
//! No floats, no exp tables, no `libm`.

use hematite_core::op_params::SoftmaxParams;
use hematite_core::KernelError;

// ── Gemmlowp fixed-point exponential constants (Q0.31) ─────────────────────
//
// Verified against gemmlowp's GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT entries
// in fixedpoint.h at the pinned TFLM SHA.

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

// ── TFLM accumulation constant ───────────────────────────────────────────

/// Number of integer bits in the softmax accumulation format (Q12.19).
const K_ACCUM_INT_BITS: i32 = 12;

// ── Private helpers (mirror tools/generate_goldens/src/tflm_math.rs) ───────

/// `SaturatingRoundingDoublingHighMul(a, b)` — gemmlowp VQRDMULH equivalent.
///
/// Computes the integer nearest to `(a * b) / 2^31`, doubling the product
/// and rounding to nearest. Saturates to `i32::MAX` when both operands are
/// `i32::MIN`.
///
/// Matches `gemmlowp::SaturatingRoundingDoublingHighMul` in
/// `gemmlowp/fixedpoint/fixedpoint.h`.
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

/// `RoundingDivideByPOT(x, exponent)` — from gemmlowp.
///
/// Returns the integer nearest to `x / 2^exponent`, with halves rounded
/// **away from zero** (gemmlowp ties-away-from-zero semantics).
#[inline(always)]
fn rounding_divide_by_pot(x: i32, exponent: i32) -> i32 {
    if exponent == 0 {
        return x;
    }
    let mask = (1i32 << exponent).wrapping_sub(1);
    let remainder = x & mask;
    let threshold = (mask >> 1) + i32::from(x < 0);
    (x >> exponent) + i32::from(remainder > threshold)
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

/// `exp_on_interval_between_negative_one_quarter_and_0_excl` — gemmlowp
/// 4th-order Taylor expansion of `exp(x)` around `x = -1/8`.
///
/// Input `a` is a Q0.31 value in the range `[-1/4, 0)`. Returns `exp(a)` as
/// Q0.31.
///
/// Matches `gemmlowp::exp_on_interval_between_negative_one_quarter_and_0_excl`
/// in `gemmlowp/fixedpoint/fixedpoint.h`.
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
/// Input `a` is in Q5.26 format (5 integer bits, 26 fractional bits).
/// `integer_bits` must be 5 for the standard softmax path.
/// Returns `exp(a)` as Q0.31.
///
/// Matches `gemmlowp::exp_on_negative_values<RawType, IntegerBits>`.
fn exp_on_negative_values(a: i32, integer_bits: i32) -> i32 {
    debug_assert!((0..=5).contains(&integer_bits));
    let fractional_bits = 31 - integer_bits;

    // one_quarter in Q5.26: 1/4 * 2^26 = 2^24
    let one_quarter = 1i32 << (fractional_bits - 2);
    let mask = one_quarter - 1;

    // a_mod_quarter_minus_one_quarter = (a & mask) - one_quarter
    // Extracts fractional part modulo 1/4, centered around -1/4
    let a_mod = (a & mask) - one_quarter;

    // Rescale fractional part from Q5.26 to Q0.31: LEFT shift by integer_bits
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

/// Count leading zeros in a `u32`.
///
/// Mirrors TFLM's `CountLeadingZeros<uint32_t>`. Returns 32 for input 0.
fn count_leading_zeros_u32(x: u32) -> i32 {
    if x == 0 {
        return 32;
    }
    let mut leading = 0i32;
    let mut v = x;
    while (v & 0x8000_0000) == 0 {
        v <<= 1;
        leading += 1;
    }
    leading
}

/// Rounding half-sum: `(a + b) / 2` rounded to nearest, ties away from zero.
///
/// Mirrors `gemmlowp::RoundingHalfSum` for i32.
fn rounding_half_sum(a: i32, b: i32) -> i32 {
    let a64 = i64::from(a);
    let b64 = i64::from(b);
    let sum = a64 + b64;
    let sign = if sum >= 0 { 1i64 } else { -1i64 };
    ((sum + sign) / 2) as i32
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

/// `GetReciprocal` — mirrors TFLM's `tflite::GetReciprocal`.
///
/// Given a raw fixed-point `x` with `x_integer_digits` integer bits, returns
/// a Q0.31 reciprocal and sets `num_bits_over_unit` to the number of bits
/// the original value exceeded 1.0.
fn get_reciprocal(x: i32, x_integer_digits: i32, num_bits_over_unit: &mut i32) -> i32 {
    let headroom_plus_one = count_leading_zeros_u32(x as u32);
    *num_bits_over_unit = x_integer_digits - headroom_plus_one;
    let shifted_sum_minus_one =
        ((x as u32) << headroom_plus_one).wrapping_sub(1u32 << 31) as i32;
    one_over_one_plus_x_for_x_in_0_1(shifted_sum_minus_one)
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Int8-safe softmax — scalar reference kernel.
///
/// Mirrors the TFLM/gemmlowp arithmetic from the golden fixture generator
/// (`tools/generate_goldens/src/ops/softmax.rs`) bit-for-bit.
///
/// # Arguments
///
/// * `input` — 1D slice of int8 logits, length `num_rows * row_size`.
/// * `params` — softmax quantization and shape parameters
///   (see [`SoftmaxParams`]).
/// * `output` — mutable slice for int8 output, same length as `input`.
/// * `scratch` — temporary byte buffer sized to hold `row_size` i32
///   intermediate values per row (i.e. `row_size * 4` bytes minimum).
///
/// # Errors
///
/// * [`KernelError::ShapeMismatch`] if `input.len() != num_rows * row_size`
///   or `output.len() != input.len()`.
pub fn softmax(
    input: &[i8],
    params: &SoftmaxParams,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    let num_rows = params.num_rows as usize;
    let row_size = params.row_size as usize;

    // ── Slice-length validation ─────────────────────────────────────────
    if input.len() != num_rows * row_size {
        return Err(KernelError::ShapeMismatch);
    }
    if output.len() != input.len() {
        return Err(KernelError::ShapeMismatch);
    }

    // ── Process each row independently ──────────────────────────────────
    for row in 0..num_rows {
        let row_start = row * row_size;
        let row_input = &input[row_start..row_start + row_size];
        let row_output = &mut output[row_start..row_start + row_size];

        // Step 1: Find max value in this row.
        let mut max_val: i32 = i32::MIN;
        {
            let mut i = 0;
            while i < row_size {
                max_val = max_val.max(i32::from(row_input[i]));
                i += 1;
            }
        }

        // Step 2: Q5.26 scaling, gemmlowp exp (Q0.31), accumulate Q12.19.
        //
        // Q5.26 conversion: sadhg(diff, input_mult) produces
        //   round(diff * input_mult / 2^31), then we left-shift by
        //   (input_left_shift + 1) to reach Q5.26.
        //   input_left_shift = LEFT_SHIFT (22) from the fixture;
        //   the +1 accounts for the sadhg implicit doubling.
        let q526_shift = params.input_left_shift + 1;
        let mut sum_q1219: i32 = 0;

        {
            let mut i = 0;
            while i < row_size {
                let diff = i32::from(row_input[i]) - max_val;
                if diff >= params.diff_min {
                    let scaled = sadhg(diff, params.input_multiplier);
                    let diff_q526 = saturating_rounding_left_shift(scaled, q526_shift);
                    let exp_q031 = exp_on_negative_values(diff_q526, 5);
                    // Rescale<kAccumulationIntegerBits>: Q0.31 → Q12.19
                    let exp_q1219 = rounding_divide_by_pot(exp_q031, K_ACCUM_INT_BITS);
                    sum_q1219 = sum_q1219.wrapping_add(exp_q1219);
                }
                i += 1;
            }
        }

        // Step 3-4: Reciprocal of sum, normalization into int8 output.
        if sum_q1219 > 0 {
            let mut num_bits_over_unit: i32 = 0;
            let shifted_scale =
                get_reciprocal(sum_q1219, K_ACCUM_INT_BITS, &mut num_bits_over_unit);
            let exponent = num_bits_over_unit + 23;

            let mut i = 0;
            while i < row_size {
                let diff = i32::from(row_input[i]) - max_val;
                if diff < params.diff_min {
                    row_output[i] = params.quantized_activation_min as i8;
                } else {
                    // Re-compute Q0.31 exponential (store in scratch for
                    // reuse within this row; recompute is cheaper than
                    // storing in the general case and avoids branching).
                    let scaled = sadhg(diff, params.input_multiplier);
                    let diff_q526 = saturating_rounding_left_shift(scaled, q526_shift);
                    let exp_q031 = exp_on_negative_values(diff_q526, 5);

                    // exp * 1/sum → Q0.31, then RoundingDivideByPOT compresses
                    let scaled_raw = sadhg(shifted_scale, exp_q031);
                    let unsat = rounding_divide_by_pot(scaled_raw, exponent);
                    let signed = unsat.wrapping_add(params.output_offset);

                    let clamped = if signed > params.quantized_activation_max {
                        params.quantized_activation_max
                    } else if signed < params.quantized_activation_min {
                        params.quantized_activation_min
                    } else {
                        signed
                    };
                    row_output[i] = clamped as i8;
                }
                i += 1;
            }
        } else {
            // Sum is zero or negative: all diffs skipped (diff_min threshold).
            // Output is output_min for every element.
            let mut i = 0;
            while i < row_size {
                row_output[i] = params.quantized_activation_min as i8;
                i += 1;
            }
        }
    }

    let _ = scratch; // unused by scalar reference path (two-pass recompute)

    Ok(())
}
