// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Per-op golden test: `KernelBackend::l2_normalization` through `RefBackend`
//! (T5.1).

mod l2_norm_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/l2_norm.rs"));
}

use hematite_core::op_params::ReduceParams;
use hematite_core::KernelBackend;
use hematite_ref::RefBackend;

#[test]
fn l2_normalization_golden() {
    let backend = RefBackend;
    let params = ReduceParams {
        keep_dims: false,
        axis: [l2_norm_fixture::AXIS_0 as i16, 0, 0, 0],
        axis_count: l2_norm_fixture::AXIS_COUNT as i8,
        input_shape: l2_norm_fixture::INPUT_SHAPE,
        output_shape: l2_norm_fixture::OUTPUT_SHAPE,
        output_type: 0,
        input_offset: l2_norm_fixture::INPUT_OFFSET,
        output_offset: l2_norm_fixture::OUTPUT_OFFSET,
        output_multiplier: l2_norm_fixture::OUTPUT_MULTIPLIER[0],
        output_shift: l2_norm_fixture::OUTPUT_SHIFT[0],
        quantized_activation_min: l2_norm_fixture::OUTPUT_ACTIVATION_MIN,
        quantized_activation_max: l2_norm_fixture::OUTPUT_ACTIVATION_MAX,
    };
    let mut output = [0i8; 4];
    backend
        .l2_normalization(&l2_norm_fixture::INPUT_DATA, &params, &mut output)
        .expect("l2_norm kernel returned Err");
    assert_eq!(
        &output[..],
        &l2_norm_fixture::EXPECTED_OUTPUT[..],
        "l2_normalization_golden: mismatch"
    );
}
