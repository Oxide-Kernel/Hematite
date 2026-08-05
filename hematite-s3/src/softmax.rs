// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Int8-safe softmax — scalar kernel (SCALAR ONLY, no SIMD).
//!
//! Per the plan (T3.3), softmax is memory-bound with 3–100 classes —
//! there is NO SIMD benefit. This module contains a single scalar fallback
//! that is also the production path on device.
//!
//! # Algorithm
//!
//! Mirrors `hematite-ref/src/softmax.rs` and the TFLM/gemmlowp
//! fixed-point exponential pipeline exactly:
//!
//! 1. Max-subtract per row (numerical stability).
//! 2. `diff_min` threshold skip (below-threshold → output_min).
//! 3. Q5.26 input scaling via `sadhg` + left-shift.
//! 4. Gemmlowp barrel-shifter exponential (`exp_on_negative_values`).
//! 5. Accumulate row sum in Q12.19.
//! 6. Newton-Raphson reciprocal (`GetReciprocal` + `one_over_one_plus_x`).
//! 7. Normalize: `exp * 1/sum` → `RoundingDivideByPOT` + output_offset
//!    + saturating clamp.
//!
//! No floats, no exp tables, no `libm`.

use hematite_core::op_params::SoftmaxParams;
use hematite_core::KernelError;

// ── Gemmlowp fixed-point exponential constants (Q0.31) ─────────────────────

const EXP_NEG_ONE_EIGHTH: i32 = 1_895_147_668;
const ONE_THIRD: i32 = 715_827_883;
const EXP_NEG_ONE_QUARTER: i32 = 1_672_461_947;
const EXP_NEG_ONE_HALF: i32 = 1_302_514_674;
const EXP_NEG_ONE: i32 = 790_015_084;
const EXP_NEG_TWO: i32 = 290_630_308;
const EXP_NEG_FOUR: i32 = 39_332_535;
const EXP_NEG_EIGHT: i32 = 720_401;
const EXP_NEG_SIXTEEN: i32 = 242;

// ── Gemmlowp one_over_one_plus_x constants ─────────────────────────────────

const C48_OVER_17: i32 = 1_515_870_810;
const CNEG32_OVER_17: i32 = -1_010_580_540;
const ONE_Q229: i32 = 1i32 << 29;

// ── TFLM accumulation constant ───────────────────────────────────────────

const K_ACCUM_INT_BITS: i32 = 12;

// ── Private helpers (mirror hematite-ref/src/softmax.rs) ───────────────────

/// `SaturatingRoundingDoublingHighMul(a, b)` — gemmlowp VQRDMULH equivalent.
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

/// Gemmlowp `exp_on_interval_between_negative_one_quarter_and_0_excl`.
fn exp_on_interval_between_negative_one_quarter_and_0_excl(a: i32) -> i32 {
    let one_eighth = 1i32 << 28;
    let x = a.wrapping_add(one_eighth);

    let x2 = sadhg(x, x);
    let x3 = sadhg(x2, x);
    let x4 = sadhg(x2, x2);

    let x4_over_4 = rounding_divide_by_pot(x4, 2);
    let t1 = x4_over_4.wrapping_add(x3);
    let t2 = sadhg(t1, ONE_THIRD);
    let t3 = t2.wrapping_add(x2);
    let inner = rounding_divide_by_pot(t3, 1);

    let poly = x.wrapping_add(inner);
    let term = sadhg(EXP_NEG_ONE_EIGHTH, poly);
    EXP_NEG_ONE_EIGHTH.wrapping_add(term)
}

/// `exp_on_negative_values` — gemmlowp barrel-shifter exponential for `x ≤ 0`.
fn exp_on_negative_values(a: i32, integer_bits: i32) -> i32 {
    debug_assert!((0..=5).contains(&integer_bits));
    let fractional_bits = 31 - integer_bits;

    let one_quarter = 1i32 << (fractional_bits - 2);
    let mask = one_quarter - 1;

    let a_mod = (a & mask) - one_quarter;
    let a_mod_q0 = saturating_rounding_left_shift(a_mod, integer_bits);

    let mut result = exp_on_interval_between_negative_one_quarter_and_0_excl(a_mod_q0);

    let remainder = a_mod - a;

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

    if a == 0 {
        result = i32::MAX;
    }

    result
}

/// Count leading zeros in a `u32`.
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
fn rounding_half_sum(a: i32, b: i32) -> i32 {
    let a64 = i64::from(a);
    let b64 = i64::from(b);
    let sum = a64 + b64;
    let sign = if sum >= 0 { 1i64 } else { -1i64 };
    ((sum + sign) / 2) as i32
}

/// `one_over_one_plus_x_for_x_in_0_1` — `1/(1+x)` for `x` in `[0, 1)`.
fn one_over_one_plus_x_for_x_in_0_1(a: i32) -> i32 {
    let half_denom_q031 = rounding_half_sum(a, i32::MAX);

    let term = sadhg(half_denom_q031, CNEG32_OVER_17);
    let mut x: i32 = C48_OVER_17.wrapping_add(term);

    for _ in 0..3 {
        let hd_x = sadhg(half_denom_q031, x);
        let one_minus_hd_x = ONE_Q229.wrapping_sub(hd_x);
        let correction = sadhg(x, one_minus_hd_x);
        x = x.wrapping_add(saturating_rounding_left_shift(correction, 2));
    }

    saturating_rounding_left_shift(x, 1)
}

/// `GetReciprocal` — mirrors TFLM's `tflite::GetReciprocal`.
fn get_reciprocal(x: i32, x_integer_digits: i32, num_bits_over_unit: &mut i32) -> i32 {
    let headroom_plus_one = count_leading_zeros_u32(x as u32);
    *num_bits_over_unit = x_integer_digits - headroom_plus_one;
    let shifted_sum_minus_one =
        ((x as u32) << headroom_plus_one).wrapping_sub(1u32 << 31) as i32;
    one_over_one_plus_x_for_x_in_0_1(shifted_sum_minus_one)
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Int8-safe softmax — scalar kernel (SCALAR ONLY).
///
/// Mirrors the TFLM/gemmlowp arithmetic from the golden fixture generator
/// bit-for-bit. Two-pass recompute (no scratch buffer needed for
/// exponential cache — scalar path recomputes exponentials in normalization
/// pass).
///
/// # Arguments
///
/// * `input` — 1D slice of int8 logits, length `num_rows * row_size`.
/// * `params` — softmax quantization and shape parameters.
/// * `output` — mutable slice for int8 output, same length as `input`.
/// * `scratch` — accepted for API compatibility, unused by scalar path.
pub fn softmax(
    input: &[i8],
    params: &SoftmaxParams,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    let num_rows = params.num_rows as usize;
    let row_size = params.row_size as usize;

    if input.len() != num_rows * row_size {
        return Err(KernelError::ShapeMismatch);
    }
    if output.len() != input.len() {
        return Err(KernelError::ShapeMismatch);
    }

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
                    let exp_q1219 = rounding_divide_by_pot(exp_q031, K_ACCUM_INT_BITS);
                    sum_q1219 = sum_q1219.wrapping_add(exp_q1219);
                }
                i += 1;
            }
        }

        // Steps 3-4: Reciprocal of sum, normalization into int8 output.
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
                    // Recompute Q0.31 exponential
                    let scaled = sadhg(diff, params.input_multiplier);
                    let diff_q526 = saturating_rounding_left_shift(scaled, q526_shift);
                    let exp_q031 = exp_on_negative_values(diff_q526, 5);

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
            let mut i = 0;
            while i < row_size {
                row_output[i] = params.quantized_activation_min as i8;
                i += 1;
            }
        }
    }

    let _ = scratch;
    Ok(())
}
