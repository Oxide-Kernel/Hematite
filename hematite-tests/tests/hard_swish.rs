// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Per-op golden test: `KernelBackend::hard_swish` through `RefBackend`
//! (T5.1).

mod hard_swish_fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/goldens/hard_swish.rs"
    ));
}

use hematite_core::op_params::ActivationParams;
use hematite_core::KernelBackend;
use hematite_ref::RefBackend;

/// Construct an `ActivationParams` from a fixture module's public consts.
macro_rules! params_from_fixture {
    ($m:ident) => {{
        ActivationParams {
            input_offset: $m::INPUT_ZERO_POINT,
            output_offset: $m::OUTPUT_ZERO_POINT,
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
            a,
            e,
            "{name}: mismatch at index {i}: kernel={a}, golden={e}",
        );
    }
}

#[test]
fn hard_swish_golden() {
    let backend = RefBackend;
    let params = params_from_fixture!(hard_swish_fixture);
    let mut output = [0i8; 8];
    backend
        .hard_swish(&hard_swish_fixture::INPUT_DATA, &params, &mut output)
        .expect("hard_swish kernel returned Err");
    assert_bit_exact(
        &output,
        &hard_swish_fixture::EXPECTED_OUTPUT,
        "hard_swish_golden",
    );
}
