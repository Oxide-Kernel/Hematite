// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Golden-vector tests for the Softmax scalar reference kernel.
//!
//! Loads TFLM-generated const-array fixtures via `include!()` and asserts
//! bit-exact output match for every golden in the corpus.
//!
//! Test naming convention: `softmax_golden_<fixture>` so that
//! `cargo test -p hematite-ref -- softmax_golden` matches all tests.

// ── Fixture includes ───────────────────────────────────────────────────────

mod softmax5 {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/softmax.rs"
    ));
}

use hematite_core::op_params::SoftmaxParams;
use hematite_ref::softmax::softmax;

/// Construct a `SoftmaxParams` from a fixture module's public consts.
///
/// Maps every fixture const to the corresponding `SoftmaxParams` field.
macro_rules! params_from_fixture {
    ($m:ident) => {{
        SoftmaxParams {
            num_rows: 1,
            row_size: $m::OUTPUT_SHAPE[3],
            input_multiplier: $m::INPUT_MULTIPLIER,
            input_left_shift: $m::LEFT_SHIFT,
            diff_min: $m::DIFF_MIN,
            input_offset: $m::INPUT_OFFSET,
            output_offset: $m::OUTPUT_OFFSET,
            quantized_activation_min: $m::OUTPUT_ACTIVATION_MIN,
            quantized_activation_max: $m::OUTPUT_ACTIVATION_MAX,
        }
    }};
}

/// Assert that `actual` matches `expected` element-for-element, printing
/// the index and values of the first mismatch.
fn assert_bit_exact(actual: &[i8], expected: &[i8], name: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{name}: output length {} != expected length {}",
        actual.len(),
        expected.len(),
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            a, e,
            "{name}: mismatch at index {i}: kernel={a}, golden={e}",
        );
    }
}

// ── Golden tests ───────────────────────────────────────────────────────────

#[test]
fn softmax_golden_5elem() {
    let params = params_from_fixture!(softmax5);
    let mut output = [0i8; 5];
    let mut scratch = [0u8; 256];
    softmax(
        &softmax5::INPUT_DATA,
        &params,
        &mut output,
        &mut scratch,
    )
    .expect("softmax kernel returned Err");
    assert_bit_exact(&output, &softmax5::EXPECTED_OUTPUT, "softmax_golden_5elem");
}

// ── Unit tests — guard the general path beyond the single golden ────────────

/// Standard params matching the golden fixture's scale (copied to avoid
/// coupling to the include! module for tests that use custom shapes).
/// `diff_min` is the TFLM-correct value for this scale/left_shift:
/// -CalculateInputRadius(5, 23) = -(31 << 26 >> 23) = -248 (see codegen).
fn standard_params(num_rows: i32, row_size: i32) -> SoftmaxParams {
    SoftmaxParams {
        num_rows,
        row_size,
        input_multiplier: 1_717_986_918,
        input_left_shift: 22,
        diff_min: -248,
        input_offset: 0,
        output_offset: -128,
        quantized_activation_min: -128,
        quantized_activation_max: 127,
    }
}

/// (a) Two-row softmax — per-row max subtraction differs between rows.
///
/// Row 0 has max = 30 (diffs [-20, -10, 0]). Row 1 has max = 60 (diffs [0, 0, 0]).
/// The two rows are processed independently: different max values → different
/// softmax distributions. Row independence is verified by running each row
/// separately and confirming bit-exact match with the 2-row output.
#[test]
fn softmax_unit_two_row_independence() {
    let input: [i8; 6] = [10, 20, 30, 60, 60, 60];
    let params = standard_params(2, 3);
    let mut output = [0i8; 6];
    let mut scratch = [0u8; 256];

    softmax(&input, &params, &mut output, &mut scratch)
        .expect("softmax kernel returned Err");

    // Row independence: run each row separately, compare.
    {
        let mut out0 = [0i8; 3];
        let params0 = standard_params(1, 3);
        softmax(&input[0..3], &params0, &mut out0, &mut scratch)
            .expect("row 0 solo failed");
        assert_eq!(&out0[..], &output[0..3],
            "two-row: row 0 must match solo run");
    }
    {
        let mut out1 = [0i8; 3];
        let params1 = standard_params(1, 3);
        softmax(&input[3..6], &params1, &mut out1, &mut scratch)
            .expect("row 1 solo failed");
        assert_eq!(&out1[..], &output[3..6],
            "two-row: row 1 must match solo run");
    }

    // Row outputs differ (different max → different distributions).
    assert_ne!(&output[0..3], &output[3..6],
        "two-row: rows with different max values must differ");

    // All outputs are in valid int8 range.
    assert!(output.iter().all(|&x| x >= -128 && x <= 127));
}

/// (b) diff_min threshold skip — elements with diff < diff_min map to
/// `quantized_activation_min` (-128).
///
/// Input = [-128, -128, 127], max = 127.
/// Diffs = [-255, -255, 0].
/// diff_min = -248 (TFLM-correct for this scale), so elements 0 and 1
/// (diff = -255 < -248) are skipped → output = -128. Element 2 (diff = 0)
/// proceeds through exponential → gets the full probability mass → softmax
/// output is the sole contributor.
///
/// Derivation for element 2:
///   exp(0) = i32::MAX (Q0.31)
///   exp_q1219 = round(i32::MAX / 2^12) = 524288
///   sum_q1219 = 524288
///   Reciprocal of 524288 with 12 integer bits → shifted ≈ 1.0 (Q0.31)
///   scaled_raw ≈ i32::MAX (Q0.31)
///   unsat = round(i32::MAX / 2^23) = 255
///   signed = 255 + (-128) = 127 → clamped to 127
#[test]
fn softmax_unit_diff_min_skip() {
    let input: [i8; 3] = [-128, -128, 127];
    let params = standard_params(1, 3);
    let mut output = [0i8; 3];
    let mut scratch = [0u8; 256];

    softmax(&input, &params, &mut output, &mut scratch)
        .expect("softmax kernel returned Err");

    // Skipped elements → output_min.
    assert_eq!(output[0], -128, "diff=-255 below diff_min must produce output_min");
    assert_eq!(output[1], -128, "diff=-255 below diff_min must produce output_min");

    // Non-skipped element should carry the full probability mass.
    // With only one contributor: softmax → max activation = 127.
    assert_eq!(output[2], 127,
        "sole non-skipped element must saturate to activation_max=127");
}

/// (c) Uniform input — all elements equal → output must be uniform.
///
/// Given input [30, 30, 30] with max = 30, every diff = 0.
/// All exponentials are identical, so after reciprocal normalization
/// every element receives 1/N of the output range. The exact value
/// depends on gemmlowp rounding, but all outputs MUST be identical.
#[test]
fn softmax_unit_uniform_input() {
    let input: [i8; 3] = [30, 30, 30];
    let params = standard_params(1, 3);
    let mut output = [0i8; 3];
    let mut scratch = [0u8; 256];

    softmax(&input, &params, &mut output, &mut scratch)
        .expect("softmax kernel returned Err");

    // All outputs must be identical (uniform input → uniform softmax).
    assert_eq!(output[0], output[1],
        "uniform input: all elements must be equal");
    assert_eq!(output[1], output[2],
        "uniform input: all elements must be equal");

    // Output within valid range.
    assert!(output[0] >= -128 && output[0] <= 127,
        "uniform output {val} outside int8 range", val = output[0]);

    // With 3 equal elements, each gets ~1/3 of the unsigned range [0,255].
    // Expected: round(255/3) = 85, then 85 - 128 = -43.
    // Gemmlowp rounding may shift this by 1-2 LSB.
    let val = output[0];
    assert!(val >= -46 && val <= -40,
        "uniform softmax for 3 elements: expected ~-43, got {val}");
}
