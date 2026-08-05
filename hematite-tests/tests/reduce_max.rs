// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Per-op golden test: `KernelBackend::reduce_max` through `RefBackend`
//! (T5.1).

mod reduce_max_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/reduce_max.rs"));
}

use hematite_core::op_params::ReduceParams;
use hematite_core::KernelBackend;
use hematite_ref::RefBackend;

#[test]
fn reduce_max_golden() {
    let backend = RefBackend;
    let params = ReduceParams {
        keep_dims: false,
        axis: [reduce_max_fixture::AXIS_0 as i16, 0, 0, 0],
        axis_count: reduce_max_fixture::AXIS_COUNT as i8,
        input_shape: reduce_max_fixture::INPUT_SHAPE,
        output_shape: reduce_max_fixture::OUTPUT_SHAPE,
        output_type: 0,
        input_offset: 0,
        output_offset: 0,
        output_multiplier: 0,
        output_shift: 0,
        quantized_activation_min: -128,
        quantized_activation_max: 127,
    };
    let mut output = [0i8; 3];
    backend
        .reduce_max(&reduce_max_fixture::INPUT_DATA, &params, &mut output)
        .expect("reduce_max kernel returned Err");
    assert_eq!(
        &output[..],
        &reduce_max_fixture::EXPECTED_OUTPUT[..],
        "reduce_max_golden: mismatch"
    );
}
