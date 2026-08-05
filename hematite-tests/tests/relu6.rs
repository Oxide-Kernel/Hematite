// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Per-op golden test: `KernelBackend::relu6` through `RefBackend` (T5.1).
//!
//! The RefBackend adapter forwards `params.quantized_activation_max` as the
//! ReLU6 clamp bound (the scalar kernel's `quantized_six` parameter). This
//! test therefore sets `quantized_activation_max = QUANTIZED_SIX` — the
//! documented adapter convention.

mod relu6_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/relu6.rs"));
}

use hematite_core::op_params::ActivationParams;
use hematite_core::KernelBackend;
use hematite_ref::RefBackend;

#[test]
fn relu6_golden() {
    let backend = RefBackend;
    let params = ActivationParams {
        input_offset: relu6_fixture::INPUT_ZERO_POINT,
        output_offset: relu6_fixture::OUTPUT_ZERO_POINT,
        output_multiplier: 1i32 << 30, // identity: scale=1.0
        output_shift: 1,
        // Adapter convention: the ReLU6 clamp bound travels in
        // quantized_activation_max (QUANTIZED_SIX has no struct field).
        quantized_activation_min: -128,
        quantized_activation_max: relu6_fixture::QUANTIZED_SIX,
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
    backend
        .relu6(&relu6_fixture::INPUT_DATA, &params, &mut output)
        .expect("relu6 kernel returned Err");
    assert_eq!(
        &output[..],
        &relu6_fixture::EXPECTED_OUTPUT[..],
        "relu6_golden: mismatch"
    );
}
