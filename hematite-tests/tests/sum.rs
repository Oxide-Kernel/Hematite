// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Per-op golden test: `KernelBackend::sum` through `RefBackend` (T5.1).

mod sum_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/sum.rs"));
}

use hematite_core::op_params::ReduceParams;
use hematite_core::KernelBackend;
use hematite_ref::RefBackend;

#[test]
fn sum_golden() {
    let backend = RefBackend;
    let params = ReduceParams {
        keep_dims: false,
        axis: [sum_fixture::AXIS_0 as i16, 0, 0, 0],
        axis_count: sum_fixture::AXIS_COUNT as i8,
        input_shape: sum_fixture::INPUT_SHAPE,
        output_shape: sum_fixture::OUTPUT_SHAPE,
        output_type: 0,
        input_offset: sum_fixture::INPUT_OFFSET,
        output_offset: sum_fixture::OUTPUT_OFFSET,
        output_multiplier: sum_fixture::OUTPUT_MULTIPLIER[0],
        output_shift: sum_fixture::OUTPUT_SHIFT[0],
        quantized_activation_min: sum_fixture::OUTPUT_ACTIVATION_MIN,
        quantized_activation_max: sum_fixture::OUTPUT_ACTIVATION_MAX,
    };
    let mut output = [0i8; 2];
    backend
        .sum(&sum_fixture::INPUT_DATA, &params, &mut output)
        .expect("sum kernel returned Err");
    assert_eq!(
        &output[..],
        &sum_fixture::EXPECTED_OUTPUT[..],
        "sum_golden: mismatch"
    );
}
