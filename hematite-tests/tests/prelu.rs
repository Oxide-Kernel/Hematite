// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Per-op golden test: `KernelBackend::prelu` through `RefBackend` (T5.1).

mod prelu_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/prelu.rs"));
}

use hematite_core::op_params::ActivationParams;
use hematite_core::KernelBackend;
use hematite_ref::RefBackend;

#[test]
fn prelu_golden() {
    let backend = RefBackend;
    let params = ActivationParams {
        input_offset: prelu_fixture::INPUT_OFFSET,
        output_offset: prelu_fixture::OUTPUT_OFFSET,
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
        alpha_offset: prelu_fixture::ALPHA_OFFSET,
        alpha_data: &prelu_fixture::ALPHA_DATA,
        output_multiplier_1: prelu_fixture::OUTPUT_MULTIPLIER_1,
        output_shift_1: prelu_fixture::OUTPUT_SHIFT_1,
        output_multiplier_2: prelu_fixture::OUTPUT_MULTIPLIER_2,
        output_shift_2: prelu_fixture::OUTPUT_SHIFT_2,
        reluish_multiplier_fixedpoint_int16: 0,
        reluish_multiplier_exponent: 0,
        output_multiplier_fixedpoint_int16: 0,
        output_multiplier_exponent: 0,
    };
    let mut output = [0i8; 4];
    backend
        .prelu(&prelu_fixture::INPUT_DATA, &params, &mut output)
        .expect("prelu kernel returned Err");
    assert_eq!(
        &output[..],
        &prelu_fixture::EXPECTED_OUTPUT[..],
        "prelu_golden: mismatch"
    );
}
