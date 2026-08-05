// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Per-op golden test: `KernelBackend::mean` through `RefBackend` (T5.1).

mod mean_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/mean.rs"));
}

use hematite_core::op_params::ReduceParams;
use hematite_core::KernelBackend;
use hematite_ref::RefBackend;

#[test]
fn mean_golden() {
    let backend = RefBackend;
    let params = ReduceParams {
        keep_dims: false,
        axis: [mean_fixture::AXIS_0 as i16, 0, 0, 0],
        axis_count: mean_fixture::AXIS_COUNT as i8,
        input_shape: mean_fixture::INPUT_SHAPE,
        output_shape: mean_fixture::OUTPUT_SHAPE,
        output_type: 0,
        input_offset: mean_fixture::INPUT_OFFSET,
        output_offset: mean_fixture::OUTPUT_OFFSET,
        output_multiplier: mean_fixture::OUTPUT_MULTIPLIER[0],
        output_shift: mean_fixture::OUTPUT_SHIFT[0],
        quantized_activation_min: mean_fixture::OUTPUT_ACTIVATION_MIN,
        quantized_activation_max: mean_fixture::OUTPUT_ACTIVATION_MAX,
    };
    let mut output = [0i8; 6];
    backend
        .mean(&mean_fixture::INPUT_DATA, &params, &mut output)
        .expect("mean kernel returned Err");
    assert_eq!(
        &output[..],
        &mean_fixture::EXPECTED_OUTPUT[..],
        "mean_golden: mismatch"
    );
}
