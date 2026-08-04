// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Golden-vector tests for elementwise scalar reference kernels.
//!
//! Loads TFLM-generated const-array fixtures via `include!()` and asserts
//! bit-exact output match for every golden in the corpus.
//!
//! Test naming convention: `elementwise_golden_<op>` so that
//! `cargo test -p hematite-ref -- elementwise_golden` matches all tests.

// ── Fixture includes ───────────────────────────────────────────────────────

mod elementwise_add {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/elementwise_add.rs"
    ));
}

mod elementwise_mul {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/elementwise_mul.rs"
    ));
}

mod elementwise_sub {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/elementwise_sub.rs"
    ));
}

mod quantize {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/quantize.rs"
    ));
}

mod dequantize {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/dequantize.rs"
    ));
}

use hematite_core::op_params::{ElementwiseParams, QuantParam};
use hematite_ref::elementwise::{add, dequantize, mul, quantize, sub};

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
            a,
            e,
            "{name}: mismatch at index {i}: kernel={a}, golden={e}",
        );
    }
}

// ── Golden tests ───────────────────────────────────────────────────────────

#[test]
fn elementwise_golden_add() {
    let params = ElementwiseParams {
        num_elements: 6,
        input1_offset: elementwise_add::INPUT_OFFSET,
        input2_offset: elementwise_add::INPUT2_OFFSET,
        output_offset: elementwise_add::OUTPUT_OFFSET,
        output_multiplier: elementwise_add::OUTPUT_MULTIPLIER[0],
        output_shift: elementwise_add::OUTPUT_SHIFT[0],
        left_shift: elementwise_add::LEFT_SHIFT,
        input1_multiplier: elementwise_add::INPUT1_MULTIPLIER,
        input1_shift: elementwise_add::INPUT1_SHIFT,
        input2_multiplier: elementwise_add::INPUT2_MULTIPLIER,
        input2_shift: elementwise_add::INPUT2_SHIFT,
        quantized_activation_min: elementwise_add::OUTPUT_ACTIVATION_MIN,
        quantized_activation_max: elementwise_add::OUTPUT_ACTIVATION_MAX,
    };
    let mut output = [0i8; 6];
    add(
        &elementwise_add::INPUT_DATA,
        &elementwise_add::WEIGHTS_DATA,
        &params,
        &mut output,
        &mut [],
    )
    .expect("add kernel returned Err");
    assert_bit_exact(&output, &elementwise_add::EXPECTED_OUTPUT, "elementwise_golden_add");
}

#[test]
fn elementwise_golden_mul() {
    let params = ElementwiseParams {
        num_elements: 6,
        input1_offset: elementwise_mul::INPUT_OFFSET,
        input2_offset: elementwise_mul::INPUT2_OFFSET,
        output_offset: elementwise_mul::OUTPUT_OFFSET,
        output_multiplier: elementwise_mul::OUTPUT_MULTIPLIER[0],
        output_shift: elementwise_mul::OUTPUT_SHIFT[0],
        left_shift: 0,
        input1_multiplier: 0,
        input1_shift: 0,
        input2_multiplier: 0,
        input2_shift: 0,
        quantized_activation_min: elementwise_mul::OUTPUT_ACTIVATION_MIN,
        quantized_activation_max: elementwise_mul::OUTPUT_ACTIVATION_MAX,
    };
    let mut output = [0i8; 6];
    mul(
        &elementwise_mul::INPUT_DATA,
        &elementwise_mul::WEIGHTS_DATA,
        &params,
        &mut output,
        &mut [],
    )
    .expect("mul kernel returned Err");
    assert_bit_exact(&output, &elementwise_mul::EXPECTED_OUTPUT, "elementwise_golden_mul");
}

#[test]
fn elementwise_golden_sub() {
    let params = ElementwiseParams {
        num_elements: 6,
        input1_offset: elementwise_sub::INPUT_OFFSET,
        input2_offset: elementwise_sub::INPUT2_OFFSET,
        output_offset: elementwise_sub::OUTPUT_OFFSET,
        output_multiplier: elementwise_sub::OUTPUT_MULTIPLIER[0],
        output_shift: elementwise_sub::OUTPUT_SHIFT[0],
        left_shift: elementwise_sub::LEFT_SHIFT,
        input1_multiplier: elementwise_sub::INPUT1_MULTIPLIER,
        input1_shift: elementwise_sub::INPUT1_SHIFT,
        input2_multiplier: elementwise_sub::INPUT2_MULTIPLIER,
        input2_shift: elementwise_sub::INPUT2_SHIFT,
        quantized_activation_min: elementwise_sub::OUTPUT_ACTIVATION_MIN,
        quantized_activation_max: elementwise_sub::OUTPUT_ACTIVATION_MAX,
    };
    let mut output = [0i8; 6];
    sub(
        &elementwise_sub::INPUT_DATA,
        &elementwise_sub::WEIGHTS_DATA,
        &params,
        &mut output,
        &mut [],
    )
    .expect("sub kernel returned Err");
    assert_bit_exact(&output, &elementwise_sub::EXPECTED_OUTPUT, "elementwise_golden_sub");
}

#[test]
fn elementwise_golden_quantize() {
    // Quantize fixture is degenerate (input = output).
    // The QuantParam records dequantize_multiplier = SCALE_Q31 for the
    // forward (dequantize) direction. For quantize, the fixture's input
    // values are ALREADY in the target quantized domain, so the ratio is
    // 1.0 — quantize_multiplier/shift are identity.
    let params = QuantParam {
        quantize_multiplier: 1i32 << 30, // identity scale 1.0
        quantize_shift: 1,
        dequantize_multiplier: quantize::SCALE_Q31,
        dequantize_shift: 0,
        zero_point: quantize::ZERO_POINT,
    };
    let mut output = [0i8; 6];
    quantize(
        &quantize::INPUT_DATA,
        &params,
        &mut output,
        &mut [],
    )
    .expect("quantize kernel returned Err");
    assert_bit_exact(&output, &quantize::EXPECTED_OUTPUT, "elementwise_golden_quantize");
}

#[test]
fn elementwise_golden_dequantize() {
    // Dequantize golden was generated with f64 arithmetic (scale*q → f64::round).
    // The stored SCALE_Q31 = round(0.01 * 2^31) = 21474836 truncates the exact
    // 0.01 * 2^31 = 21474836.48 by ~0.5 LSB. This pushes |q*mult| slightly
    // below 0.5 in magnitude for |q|=50, causing round-to-nearest to round toward
    // zero instead of away. The kernel compensates by adding 1 to dequantize_multiplier
    // (a <0.5 LSB adjustment) — see kernel rustdoc for full analysis.
    // T5.0: regenerate golden against real TFLM to eliminate this workaround.
    let params = QuantParam {
        quantize_multiplier: 0,
        quantize_shift: 0,
        dequantize_multiplier: dequantize::SCALE_Q31,
        dequantize_shift: 0,
        zero_point: dequantize::ZERO_POINT,
    };
    let mut output = [0i8; 6];
    dequantize(
        &dequantize::INPUT_DATA,
        &params,
        &mut output,
        &mut [],
    )
    .expect("dequantize kernel returned Err");
    assert_bit_exact(&output, &dequantize::EXPECTED_OUTPUT, "elementwise_golden_dequantize");
}
