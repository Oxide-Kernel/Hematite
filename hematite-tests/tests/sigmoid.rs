// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Per-op golden test: `KernelBackend::sigmoid` through `RefBackend` (T5.1).

mod sigmoid_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/sigmoid.rs"));
}

use hematite_core::op_params::ActivationParams;
use hematite_core::KernelBackend;
use hematite_ref::RefBackend;

#[test]
fn sigmoid_golden() {
    let backend = RefBackend;
    let params = ActivationParams {
        input_offset: sigmoid_fixture::INPUT_OFFSET,
        output_offset: sigmoid_fixture::OUTPUT_OFFSET,
        output_multiplier: 0,
        output_shift: 0,
        quantized_activation_min: -128,
        quantized_activation_max: 127,
        input_multiplier: sigmoid_fixture::INPUT_MULTIPLIER,
        input_left_shift: sigmoid_fixture::INPUT_LEFT_SHIFT,
        input_range_radius: sigmoid_fixture::INPUT_RANGE_RADIUS,
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
    let mut output = [0i8; 11];
    backend
        .sigmoid(&sigmoid_fixture::INPUT_DATA, &params, &mut output)
        .expect("sigmoid kernel returned Err");
    assert_eq!(
        &output[..],
        &sigmoid_fixture::EXPECTED_OUTPUT[..],
        "sigmoid_golden: mismatch"
    );
}
