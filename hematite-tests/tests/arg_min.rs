// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Per-op golden test: `KernelBackend::arg_min` through `RefBackend` (T5.1).

mod argmin_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/argmin.rs"));
}

use hematite_core::op_params::ReduceParams;
use hematite_core::KernelBackend;
use hematite_ref::RefBackend;

#[test]
fn arg_min_golden() {
    let backend = RefBackend;
    // argmax/argmin ignore the quant fields; the fixture carries only the
    // axis consts, so the quant fields are zeroed.
    let params = ReduceParams {
        keep_dims: false,
        axis: [argmin_fixture::AXIS_0 as i16, 0, 0, 0],
        axis_count: argmin_fixture::AXIS_COUNT as i8,
        input_shape: argmin_fixture::INPUT_SHAPE,
        output_shape: argmin_fixture::OUTPUT_SHAPE,
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
        .arg_min(&argmin_fixture::INPUT_DATA, &params, &mut output)
        .expect("arg_min kernel returned Err");
    assert_eq!(
        &output[..],
        &argmin_fixture::EXPECTED_OUTPUT[..],
        "arg_min_golden: mismatch"
    );
}
