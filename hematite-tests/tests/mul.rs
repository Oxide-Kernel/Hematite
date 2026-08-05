// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Per-op golden test: `KernelBackend::mul` through `RefBackend` (T5.1).

mod elementwise_mul_fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/goldens/elementwise_mul.rs"
    ));
}

use hematite_core::op_params::ElementwiseParams;
use hematite_core::KernelBackend;
use hematite_ref::RefBackend;

#[test]
fn mul_golden() {
    let backend = RefBackend;
    // The mul fixture carries no per-input rescale constants (no left_shift,
    // no input1/input2 multiplier/shift); the kernel path uses none.
    let params = ElementwiseParams {
        num_elements: 6,
        input1_offset: elementwise_mul_fixture::INPUT_OFFSET,
        input2_offset: elementwise_mul_fixture::INPUT2_OFFSET,
        output_offset: elementwise_mul_fixture::OUTPUT_OFFSET,
        output_multiplier: elementwise_mul_fixture::OUTPUT_MULTIPLIER[0],
        output_shift: elementwise_mul_fixture::OUTPUT_SHIFT[0],
        left_shift: 0,
        input1_multiplier: 0,
        input1_shift: 0,
        input2_multiplier: 0,
        input2_shift: 0,
        quantized_activation_min: elementwise_mul_fixture::OUTPUT_ACTIVATION_MIN,
        quantized_activation_max: elementwise_mul_fixture::OUTPUT_ACTIVATION_MAX,
    };
    let mut output = [0i8; 6];
    backend
        .mul(
            &elementwise_mul_fixture::INPUT_DATA,
            &elementwise_mul_fixture::WEIGHTS_DATA,
            &params,
            &mut output,
        )
        .expect("mul kernel returned Err");
    assert_eq!(
        &output[..],
        &elementwise_mul_fixture::EXPECTED_OUTPUT[..],
        "mul_golden: mismatch"
    );
}
