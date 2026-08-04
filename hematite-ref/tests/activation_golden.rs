// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Golden-vector tests for activation scalar reference kernels.
//!
//! Loads TFLM-generated const-array fixtures via `include!()` and asserts
//! bit-exact output match for every golden in the corpus.
//!
//! Test naming: `activation_golden_<op>` so that
//! `cargo test -p hematite-ref -- activation_golden` matches all tests.

// ── Fixture includes ────────────────────────────────────────────────────────

mod relu {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/relu.rs"
    ));
}

mod relu6 {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/relu6.rs"
    ));
}

mod hard_swish {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/hard_swish.rs"
    ));
}

mod leaky_relu {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/leaky_relu.rs"
    ));
}

mod prelu {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/prelu.rs"
    ));
}

use hematite_core::op_params::ActivationParams;
use hematite_ref::activation::{hard_swish, leaky_relu, prelu, relu, relu6};

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

// ── Golden tests ────────────────────────────────────────────────────────────

#[test]
fn activation_golden_relu() {
    let params = ActivationParams {
        input_offset: relu::INPUT_ZERO_POINT,
        output_offset: relu::OUTPUT_ZERO_POINT,
        output_multiplier: relu::OUTPUT_MULTIPLIER,
        output_shift: relu::OUTPUT_SHIFT,
        quantized_activation_min: -128,
        quantized_activation_max: 127,
        input_multiplier: 0,
        input_left_shift: 0,
        input_range_radius: 0,
        output_multiplier_alpha: 0,
        output_shift_alpha: 0,
        output_multiplier_identity: 0,
        output_shift_identity: 0,
        alpha_offset: 0,
        alpha_data: &[],
        output_multiplier_1: 0,
        output_shift_1: 0,
        output_multiplier_2: 0,
        output_shift_2: 0,
        reluish_multiplier_fixedpoint_int16: 0,
        reluish_multiplier_exponent: 0,
        output_multiplier_fixedpoint_int16: 0,
        output_multiplier_exponent: 0,
    };
    let mut output = [0i8; 8];
    relu(&relu::INPUT_DATA, &params, &mut output, &mut [])
        .expect("relu kernel returned Err");
    assert_bit_exact(&output, &relu::EXPECTED_OUTPUT, "activation_golden_relu");
}

#[test]
fn activation_golden_relu6() {
    let params = ActivationParams {
        input_offset: relu6::INPUT_ZERO_POINT,
        output_offset: relu6::OUTPUT_ZERO_POINT,
        output_multiplier: 1073741824, // identity: scale=1.0
        output_shift: 1,
        quantized_activation_min: -128,
        quantized_activation_max: 127,
        input_multiplier: 0,
        input_left_shift: 0,
        input_range_radius: 0,
        output_multiplier_alpha: 0,
        output_shift_alpha: 0,
        output_multiplier_identity: 0,
        output_shift_identity: 0,
        alpha_offset: 0,
        alpha_data: &[],
        output_multiplier_1: 0,
        output_shift_1: 0,
        output_multiplier_2: 0,
        output_shift_2: 0,
        reluish_multiplier_fixedpoint_int16: 0,
        reluish_multiplier_exponent: 0,
        output_multiplier_fixedpoint_int16: 0,
        output_multiplier_exponent: 0,
    };
    let mut output = [0i8; 8];
    relu6(
        &relu6::INPUT_DATA,
        &params,
        &mut output,
        &mut [],
        relu6::QUANTIZED_SIX,
    )
    .expect("relu6 kernel returned Err");
    assert_bit_exact(&output, &relu6::EXPECTED_OUTPUT, "activation_golden_relu6");
}

#[test]
fn activation_golden_hard_swish() {
    let params = ActivationParams {
        input_offset: hard_swish::INPUT_ZERO_POINT,
        output_offset: hard_swish::OUTPUT_ZERO_POINT,
        output_multiplier: 0,
        output_shift: 0,
        quantized_activation_min: -128,
        quantized_activation_max: 127,
        input_multiplier: 0,
        input_left_shift: 0,
        input_range_radius: 0,
        output_multiplier_alpha: 0,
        output_shift_alpha: 0,
        output_multiplier_identity: 0,
        output_shift_identity: 0,
        alpha_offset: 0,
        alpha_data: &[],
        output_multiplier_1: 0,
        output_shift_1: 0,
        output_multiplier_2: 0,
        output_shift_2: 0,
        reluish_multiplier_fixedpoint_int16: 0,
        reluish_multiplier_exponent: 0,
        output_multiplier_fixedpoint_int16: 0,
        output_multiplier_exponent: 0,
    };
    let mut output = [0i8; 8];
    hard_swish(&hard_swish::INPUT_DATA, &params, &mut output, &mut [])
        .expect("hard_swish kernel returned Err");
    assert_bit_exact(
        &output,
        &hard_swish::EXPECTED_OUTPUT,
        "activation_golden_hard_swish",
    );
}

#[test]
fn activation_golden_leaky_relu() {
    let params = ActivationParams {
        input_offset: leaky_relu::INPUT_OFFSET,
        output_offset: leaky_relu::OUTPUT_OFFSET,
        output_multiplier: 0,
        output_shift: 0,
        quantized_activation_min: -128,
        quantized_activation_max: 127,
        input_multiplier: 0,
        input_left_shift: 0,
        input_range_radius: 0,
        output_multiplier_alpha: leaky_relu::OUTPUT_MULTIPLIER_ALPHA,
        output_shift_alpha: leaky_relu::OUTPUT_SHIFT_ALPHA,
        output_multiplier_identity: leaky_relu::OUTPUT_MULTIPLIER_IDENTITY,
        output_shift_identity: leaky_relu::OUTPUT_SHIFT_IDENTITY,
        alpha_offset: 0,
        alpha_data: &[],
        output_multiplier_1: 0,
        output_shift_1: 0,
        output_multiplier_2: 0,
        output_shift_2: 0,
        reluish_multiplier_fixedpoint_int16: 0,
        reluish_multiplier_exponent: 0,
        output_multiplier_fixedpoint_int16: 0,
        output_multiplier_exponent: 0,
    };
    let mut output = [0i8; 8];
    leaky_relu(&leaky_relu::INPUT_DATA, &params, &mut output, &mut [])
        .expect("leaky_relu kernel returned Err");
    assert_bit_exact(
        &output,
        &leaky_relu::EXPECTED_OUTPUT,
        "activation_golden_leaky_relu",
    );
}

#[test]
fn activation_golden_prelu() {
    let params = ActivationParams {
        input_offset: prelu::INPUT_OFFSET,
        output_offset: prelu::OUTPUT_OFFSET,
        output_multiplier: 0,
        output_shift: 0,
        quantized_activation_min: -128,
        quantized_activation_max: 127,
        input_multiplier: 0,
        input_left_shift: 0,
        input_range_radius: 0,
        output_multiplier_alpha: 0,
        output_shift_alpha: 0,
        output_multiplier_identity: 0,
        output_shift_identity: 0,
        alpha_offset: prelu::ALPHA_OFFSET,
        alpha_data: &prelu::ALPHA_DATA,
        output_multiplier_1: prelu::OUTPUT_MULTIPLIER_1,
        output_shift_1: prelu::OUTPUT_SHIFT_1,
        output_multiplier_2: prelu::OUTPUT_MULTIPLIER_2,
        output_shift_2: prelu::OUTPUT_SHIFT_2,
        reluish_multiplier_fixedpoint_int16: 0,
        reluish_multiplier_exponent: 0,
        output_multiplier_fixedpoint_int16: 0,
        output_multiplier_exponent: 0,
    };
    let mut output = [0i8; 4];
    prelu(&prelu::INPUT_DATA, &params, &mut output, &mut [])
        .expect("prelu kernel returned Err");
    assert_bit_exact(&output, &prelu::EXPECTED_OUTPUT, "activation_golden_prelu");
}
