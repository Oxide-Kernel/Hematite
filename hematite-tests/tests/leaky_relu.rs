// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Per-op golden test: `KernelBackend::leaky_relu` through `RefBackend`
//! (T5.1).

mod leaky_relu_fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/goldens/leaky_relu.rs"
    ));
}

use hematite_core::op_params::ActivationParams;
use hematite_core::KernelBackend;
use hematite_ref::RefBackend;

#[test]
fn leaky_relu_golden() {
    let backend = RefBackend;
    let params = ActivationParams {
        input_offset: leaky_relu_fixture::INPUT_OFFSET,
        output_offset: leaky_relu_fixture::OUTPUT_OFFSET,
        output_multiplier: 0,
        output_shift: 0,
        quantized_activation_min: -128,
        quantized_activation_max: 127,
        input_multiplier: 0,
        input_left_shift: 0,
        input_range_radius: 0,
        output_multiplier_alpha: leaky_relu_fixture::OUTPUT_MULTIPLIER_ALPHA,
        output_shift_alpha: leaky_relu_fixture::OUTPUT_SHIFT_ALPHA,
        output_multiplier_identity: leaky_relu_fixture::OUTPUT_MULTIPLIER_IDENTITY,
        output_shift_identity: leaky_relu_fixture::OUTPUT_SHIFT_IDENTITY,
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
    backend
        .leaky_relu(&leaky_relu_fixture::INPUT_DATA, &params, &mut output)
        .expect("leaky_relu kernel returned Err");
    assert_eq!(
        &output[..],
        &leaky_relu_fixture::EXPECTED_OUTPUT[..],
        "leaky_relu_golden: mismatch"
    );
}
