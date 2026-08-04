//! Differential + boundary tests for `hematite-int8` quantization primitives.
//!
//! This file contains local copies of the reference implementations from
//! `tools/generate_goldens/src/tflm_math.rs` (SHA `309338a`), used as the
//! bit-exact oracle against which `hematite_int8` is validated.
//!
//! The reference functions are NOT re-exported — they are duplicated here
//! so that this integration test has zero dependency on the generator crate
//! (which is a binary, not a library).

// ── Reference implementations (bit-exact mirrors of tflm_math.rs) ──

/// Reference `multiply_by_quantized_multiplier` — exactly the same arithmetic
/// as `tools/generate_goldens/src/tflm_math.rs` @ SHA 309338a.
fn ref_multiply_by_quantized_multiplier(value: i32, multiplier: i32, shift: i32) -> i32 {
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

/// Reference `rounding_divide_by_pot` — exactly the same arithmetic
/// as `tools/generate_goldens/src/tflm_math.rs` @ SHA 309338a.
fn ref_rounding_divide_by_pot(x: i32, exponent: i32) -> i32 {
    if exponent == 0 {
        return x;
    }
    let mask = (1i32 << exponent) - 1;
    let remainder = x & mask;
    let threshold = (mask >> 1) + i32::from(x < 0);
    (x >> exponent) + i32::from(remainder > threshold)
}

/// Reference `clamp_to_i8` — saturating clamp with activation bounds.
fn ref_clamp_to_i8(value: i32, activation_min: i32, activation_max: i32) -> i8 {
    value.max(activation_min).min(activation_max) as i8
}

/// Reference `requantize_i8` — full TFLM epilogue with offset + activation.
fn ref_requantize_i8(
    acc: i32,
    output_multiplier: i32,
    output_shift: i32,
    _output_offset: i32,
    activation_min: i32,
    activation_max: i32,
) -> i8 {
    let scaled = ref_multiply_by_quantized_multiplier(acc, output_multiplier, output_shift);
    // Note: output_offset is not added here (our hematite-int8 doesn't either;
    // kernels handle offsets). This reference exists purely to validate the
    // multiply + saturate path.
    ref_clamp_to_i8(scaled, activation_min, activation_max)
}

use hematite_int8::{
    multiply_by_quantized_multiplier, requantize, rounding_divide_by_pot, saturating_cast,
};
use hematite_core::op_params::PerChannelQuantParam;

// ── Differential sweep: multiply_by_quantized_multiplier ──

#[test]
fn test_multiply_differential_sweep() {
    // Values spanning most of the i32 range with step ~2^15, plus adversarial edges.
    let mut values: Vec<i32> = (-(1i32 << 24)..=(1i32 << 24))
        .step_by((1usize << 15) as usize)
        .collect();
    // Adversarial set: extremes, zero, ±1, powers of two near the edges
    let adversarial = [
        i32::MIN,
        i32::MAX,
        0,
        1,
        -1,
        1i32 << 30,
        -(1i32 << 30),
        i32::MAX - 1,
        i32::MIN + 1,
    ];
    values.extend_from_slice(&adversarial);

    let multipliers: [i32; 6] = [
        0,
        1,
        2,
        3,
        1i32 << 30, // Q0.31 half-scale
        i32::MAX,   // Q0.31 one-point-zero
    ];

    // shifts in [-31, 30] per TFLM's DCHECK
    let shifts: Vec<i32> = (-31..=30).collect();

    let mut tested = 0usize;
    for &v in &values {
        for &m in &multipliers {
            for &s in &shifts {
                let actual = multiply_by_quantized_multiplier(v, m, s);
                let expected = ref_multiply_by_quantized_multiplier(v, m, s);
                assert_eq!(
                    actual, expected,
                    "multiply_by_quantized_multiplier diverge: \
                     value={v}, multiplier={m}, shift={s}, \
                     actual={actual}, expected={expected}"
                );
                tested += 1;
            }
        }
    }
    // sanity: we tested a non-trivial number of combinations
    assert!(tested > 100_000, "only tested {tested} combinations — sweep too small");
}

// ── Boundary tests: multiply_by_quantized_multiplier ──

#[test]
fn test_multiply_no_shift() {
    let m = 1i32 << 30; // effective scale = 0.5
    assert_eq!(multiply_by_quantized_multiplier(100, m, 0), 50);
}

#[test]
fn test_multiply_rounding_boundary_half_up_positive() {
    // 1 * 0.5 = 0.5 → round half up → 1
    let m = 1i32 << 30;
    assert_eq!(multiply_by_quantized_multiplier(1, m, 0), 1);
}

#[test]
fn test_multiply_rounding_boundary_half_up_negative() {
    // -1 * 0.5 = -0.5 → round-half-up biases toward +∞ → 0
    // (CMSIS single-rounding: (-1*2^30 + 2^30) >> 31 = 0 >> 31 = 0)
    let m = 1i32 << 30;
    assert_eq!(multiply_by_quantized_multiplier(-1, m, 0), 0);
}

#[test]
fn test_multiply_rounding_positive_three_half() {
    // 3 * 0.5 = 1.5 → round-half-up → 2
    let m = 1i32 << 30;
    assert_eq!(multiply_by_quantized_multiplier(3, m, 0), 2);
}

#[test]
fn test_multiply_rounding_exact() {
    // 2 * 0.5 = 1.0 → exact
    let m = 1i32 << 30;
    assert_eq!(multiply_by_quantized_multiplier(2, m, 0), 1);
}

#[test]
fn test_multiply_rounding_negative_exact() {
    let m = 1i32 << 30;
    assert_eq!(multiply_by_quantized_multiplier(-50, m, 0), -25);
}

#[test]
fn test_multiply_with_positive_shift() {
    // multiplier=2^30, shift=1 → effective scale = 2^30/2^31 * 2^1 = 1.0
    let m = 1i32 << 30;
    assert_eq!(multiply_by_quantized_multiplier(42, m, 1), 42);
}

#[test]
fn test_multiply_with_negative_shift() {
    // multiplier=2^30, shift=-1 → effective scale = 2^30/2^31 * 2^-1 = 0.25
    let m = 1i32 << 30;
    assert_eq!(multiply_by_quantized_multiplier(100, m, -1), 25);
}

#[test]
fn test_multiply_scale_gt_1() {
    let m = 1i32 << 30;
    let shift = 2;
    assert_eq!(multiply_by_quantized_multiplier(5, m, shift), 10);
    assert_eq!(multiply_by_quantized_multiplier(-3, m, shift), -6);
}

#[test]
fn test_multiply_doubled_high_mul_boundary() {
    // Construct the exact boundary case where the product lands on .5 at the
    // rounding point per plan T1.2 B4.
    //
    // For shift=0, total_shift=31, round=2^30.
    // The rounding decision depends on (value * multiplier + 2^30) >> 31.
    // When (value * multiplier) mod 2^31 == 2^30, the product is exactly at
    // the .5 boundary, and round-half-up pushes it to the next integer.
    //
    // Example: value=3, multiplier=2^30 → product=3*2^30=3221225472
    //   3221225472 mod 2^31 = 3221225472 mod 2147483648 = 1073741824 = 2^30
    //   So (3*2^30 + 2^30) >> 31 = (4*2^30) >> 31 = 2^31 >> 31 = 1 → wait, that gives 1?
    // Let me recalculate:
    //   value=3, multiplier=2^30, shift=0
    //   total_shift = 31 - 0 = 31, round = 2^30
    //   result = (3 * 2^30 + 2^30) >> 31 = (3*2^30 + 2^30) / 2^31 = (4*2^30)/(2*2^30) = 2
    //   Yes, 3 * 0.5 = 1.5, round-half-up → 2. Correct.
    //
    // For value=1, multiplier=2^30:
    //   result = (2^30 + 2^30) >> 31 = 2^31 >> 31 = 1
    //   1 * 0.5 = 0.5, round-half-up → 1. Correct.
    let m = 1i32 << 30;
    assert_eq!(multiply_by_quantized_multiplier(3, m, 0), 2, "3*0.5=1.5→2 boundary");
    assert_eq!(multiply_by_quantized_multiplier(1, m, 0), 1, "1*0.5=0.5→1 boundary");
}

// ── rounding_divide_by_pot tests ──

#[test]
fn test_rounding_divide_by_pot_exact_zero_exp() {
    assert_eq!(rounding_divide_by_pot(42, 0), 42);
}

#[test]
fn test_rounding_divide_by_pot_positive_half_up() {
    // 3 / 2 = 1.5 → ties away from zero → 2
    assert_eq!(rounding_divide_by_pot(3, 1), 2);
    // 1 / 2 = 0.5 → ties away from zero → 1
    assert_eq!(rounding_divide_by_pot(1, 1), 1);
}

#[test]
fn test_rounding_divide_by_pot_positive_exact() {
    assert_eq!(rounding_divide_by_pot(4, 1), 2);
    assert_eq!(rounding_divide_by_pot(8, 2), 2);
}

#[test]
fn test_rounding_divide_by_pot_positive_tie_away_from_zero() {
    // 39 / 2 = 19.5 → ties away from zero → 20
    assert_eq!(rounding_divide_by_pot(39, 1), 20);
}

// ── Negative-tie regressions (plan T1.2 — verified against golden generator) ──

#[test]
fn test_rounding_divide_by_pot_neg_39_exp_1() {
    // -39 / 2 = -19.5 → gemmlowp ties AWAY from zero → -20 (NOT -19)
    assert_eq!(rounding_divide_by_pot(-39, 1), -20);
}

#[test]
fn test_rounding_divide_by_pot_neg_38_exp_2() {
    // -38 / 4 = -9.5 → gemmlowp ties away from zero → -10
    assert_eq!(rounding_divide_by_pot(-38, 2), -10);
}

#[test]
fn test_rounding_divide_by_pot_neg_36_exp_3() {
    // -36 / 8 = -4.5 → gemmlowp ties away from zero → -5
    assert_eq!(rounding_divide_by_pot(-36, 3), -5);
}

#[test]
fn test_rounding_divide_by_pot_negative_not_tie() {
    // -37 / 2 = -18.5 → away from zero → -19
    assert_eq!(rounding_divide_by_pot(-37, 1), -19);
    // -39 / 4 = -9.75 → away from zero → -10
    assert_eq!(rounding_divide_by_pot(-39, 2), -10);
}

#[test]
fn test_rounding_divide_by_pot_negative_exact() {
    assert_eq!(rounding_divide_by_pot(-4, 1), -2);
    assert_eq!(rounding_divide_by_pot(-8, 2), -2);
}

#[test]
fn test_rounding_divide_by_pot_all_negative_ties() {
    // Cross-check against the local reference for all negative tie cases
    for &(x, exp, expected) in &[(-39, 1, -20), (-38, 2, -10), (-36, 3, -5), (39, 1, 20)] {
        let actual = rounding_divide_by_pot(x, exp);
        let ref_val = ref_rounding_divide_by_pot(x, exp);
        assert_eq!(actual, expected, "rounding_divide_by_pot({x}, {exp}) = {actual}, expected {expected}");
        assert_eq!(actual, ref_val, "rounding_divide_by_pot({x}, {exp}) diverged from reference: {actual} vs {ref_val}");
    }
}

// ── rounding_divide_by_pot differential sweep ──

#[test]
fn test_rounding_divide_by_pot_differential_sweep() {
    let values: Vec<i32> = (-(1i32 << 16)..=(1i32 << 16))
        .step_by(997) // ~131 values, covers both positive and negative
        .collect();
    // Add adversarial edges
    let mut all_values = values;
    all_values.extend_from_slice(&[
        i32::MIN, i32::MIN + 1, i32::MAX, i32::MAX - 1,
        0, 1, -1, 2, -2, 3, -3,
    ]);

    // Exponents used by gemmlowp in practice (small values)
    let exponents: [i32; 5] = [0, 1, 2, 3, 4];

    let mut tested = 0usize;
    for &v in &all_values {
        for &e in &exponents {
            let actual = rounding_divide_by_pot(v, e);
            let expected = ref_rounding_divide_by_pot(v, e);
            assert_eq!(
                actual, expected,
                "rounding_divide_by_pot diverge: x={v}, exp={e}, actual={actual}, expected={expected}"
            );
            tested += 1;
        }
    }
    assert!(tested > 100, "only tested {tested} combinations — sweep too small");
}

// ── saturating_cast tests ──

#[test]
fn test_saturating_cast_in_range_values() {
    assert_eq!(saturating_cast(0), 0);
    assert_eq!(saturating_cast(42), 42);
    assert_eq!(saturating_cast(-42), -42);
    assert_eq!(saturating_cast(127), 127);
    assert_eq!(saturating_cast(-128), -128);
    assert_eq!(saturating_cast(i32::MAX), 127);
    assert_eq!(saturating_cast(i32::MIN), -128);
}

#[test]
fn test_saturating_cast_boundaries() {
    assert_eq!(saturating_cast(128), 127);
    assert_eq!(saturating_cast(-129), -128);
    assert_eq!(saturating_cast(1000), 127);
    assert_eq!(saturating_cast(-1000), -128);
}

// ── requantize tests ──

#[test]
fn test_requantize_per_channel_smoke() {
    // Two channels: ch0 has scale 0.5, ch1 has scale 1.0
    let multipliers: &[i32] = &[1i32 << 30, 1i32 << 30];
    let shifts: &[i32] = &[0, 1]; // ch0: effective 0.5, ch1: effective 1.0
    let params = PerChannelQuantParam {
        output_multiplier_per_channel: multipliers,
        output_shift_per_channel: shifts,
    };

    // acc=100: ch0 → 100*0.5=50, ch1 → 100*1.0=100
    assert_eq!(requantize(100, &params, 0), 50);
    assert_eq!(requantize(100, &params, 1), 100);
}

#[test]
fn test_requantize_saturates_to_i8() {
    let multipliers: &[i32] = &[1i32 << 30];
    let shifts: &[i32] = &[10]; // huge effective scale to force saturation
    let params = PerChannelQuantParam {
        output_multiplier_per_channel: multipliers,
        output_shift_per_channel: shifts,
    };

    assert_eq!(requantize(1000, &params, 0), 127);
    assert_eq!(requantize(-1000, &params, 0), -128);
}

#[test]
fn test_requantize_out_of_bounds_channel() {
    let multipliers: &[i32] = &[1i32 << 30];
    let shifts: &[i32] = &[0];
    let params = PerChannelQuantParam {
        output_multiplier_per_channel: multipliers,
        output_shift_per_channel: shifts,
    };

    // channel 99 is OOB → should return saturating_cast(acc)
    assert_eq!(requantize(50, &params, 99), 50);
    assert_eq!(requantize(200, &params, 99), 127);
}

#[test]
fn test_requantize_differential_against_reference() {
    // Our requantize (without offset/activation) should match the reference's
    // multiply+clamp path for the full i8 range.
    let multipliers: &[i32] = &[1i32 << 30, 1i32 << 30, 1i32 << 30];
    let shifts: &[i32] = &[0, -1, 2];
    let params = PerChannelQuantParam {
        output_multiplier_per_channel: multipliers,
        output_shift_per_channel: shifts,
    };

    let test_values: [i32; 9] = [-127, -64, -8, -1, 0, 1, 8, 64, 127];

    for ch in 0..3 {
        let m = multipliers[ch];
        let s = shifts[ch];
        for &v in &test_values {
            let actual = requantize(v, &params, ch);
            // Reference with full i8 range activation bounds (no offset)
            let expected = ref_requantize_i8(v, m, s, 0, -128, 127);
            assert_eq!(
                actual, expected,
                "requantize({v}, ch={ch}) diverge: actual={actual}, expected={expected}"
            );
        }
    }
}

#[test]
fn test_requantize_empty_slices() {
    let params = PerChannelQuantParam {
        output_multiplier_per_channel: &[],
        output_shift_per_channel: &[],
    };
    // Any channel is out of bounds → return saturating_cast(acc)
    assert_eq!(requantize(50, &params, 0), 50);
    assert_eq!(requantize(200, &params, 0), 127);
}
