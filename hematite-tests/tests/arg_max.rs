// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Per-op golden test: `KernelBackend::arg_max` through `RefBackend` (T5.1).

mod argmax_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/argmax.rs"));
}

use hematite_core::op_params::ReduceParams;
use hematite_core::KernelBackend;
use hematite_ref::RefBackend;

#[test]
fn arg_max_golden() {
    let backend = RefBackend;
    // argmax/argmin ignore the quant fields; the fixture carries only the
    // axis consts, so the quant fields are zeroed.
    let params = ReduceParams {
        keep_dims: false,
        axis: [argmax_fixture::AXIS_0 as i16, 0, 0, 0],
        axis_count: argmax_fixture::AXIS_COUNT as i8,
        input_shape: argmax_fixture::INPUT_SHAPE,
        output_shape: argmax_fixture::OUTPUT_SHAPE,
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
        .arg_max(&argmax_fixture::INPUT_DATA, &params, &mut output)
        .expect("arg_max kernel returned Err");
    assert_eq!(
        &output[..],
        &argmax_fixture::EXPECTED_OUTPUT[..],
        "arg_max_golden: mismatch"
    );
}
