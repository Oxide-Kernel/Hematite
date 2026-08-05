// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Per-op golden test: `KernelBackend::concat` through `RefBackend` (T5.1).

mod concat_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/concat.rs"));
}

use hematite_core::op_params::{ConcatParams, FusedActivation};
use hematite_core::KernelBackend;
use hematite_ref::RefBackend;

#[test]
fn concat_golden() {
    let backend = RefBackend;
    let params = ConcatParams {
        axis: concat_fixture::AXIS,
        activation: FusedActivation::None,
        input_shape_a: concat_fixture::INPUT_SHAPE,
        // Fixture naming quirk: the second input shape/data reuse the conv
        // template names FILTER_SHAPE / WEIGHTS_DATA (see T2.3 learnings).
        input_shape_b: concat_fixture::FILTER_SHAPE,
        output_shape: concat_fixture::OUTPUT_SHAPE,
    };
    let mut output = [0i8; 4];
    backend
        .concat(
            &concat_fixture::INPUT_DATA,
            &concat_fixture::WEIGHTS_DATA,
            &params,
            &mut output,
        )
        .expect("concat kernel returned Err");
    assert_eq!(
        &output[..],
        &concat_fixture::EXPECTED_OUTPUT[..],
        "concat_golden: mismatch"
    );
}
