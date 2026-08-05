// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Per-op golden test: `KernelBackend::add` through `RefBackend` (T5.1).

mod elementwise_add_fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/goldens/elementwise_add.rs"
    ));
}

use hematite_core::op_params::ElementwiseParams;
use hematite_core::KernelBackend;
use hematite_ref::RefBackend;

#[test]
fn add_golden() {
    let backend = RefBackend;
    let params = ElementwiseParams {
        num_elements: 6,
        input1_offset: elementwise_add_fixture::INPUT_OFFSET,
        input2_offset: elementwise_add_fixture::INPUT2_OFFSET,
        output_offset: elementwise_add_fixture::OUTPUT_OFFSET,
        output_multiplier: elementwise_add_fixture::OUTPUT_MULTIPLIER[0],
        output_shift: elementwise_add_fixture::OUTPUT_SHIFT[0],
        left_shift: elementwise_add_fixture::LEFT_SHIFT,
        input1_multiplier: elementwise_add_fixture::INPUT1_MULTIPLIER,
        input1_shift: elementwise_add_fixture::INPUT1_SHIFT,
        input2_multiplier: elementwise_add_fixture::INPUT2_MULTIPLIER,
        input2_shift: elementwise_add_fixture::INPUT2_SHIFT,
        quantized_activation_min: elementwise_add_fixture::OUTPUT_ACTIVATION_MIN,
        quantized_activation_max: elementwise_add_fixture::OUTPUT_ACTIVATION_MAX,
    };
    let mut output = [0i8; 6];
    backend
        .add(
            &elementwise_add_fixture::INPUT_DATA,
            &elementwise_add_fixture::WEIGHTS_DATA,
            &params,
            &mut output,
        )
        .expect("add kernel returned Err");
    assert_eq!(
        &output[..],
        &elementwise_add_fixture::EXPECTED_OUTPUT[..],
        "add_golden: mismatch"
    );
}
