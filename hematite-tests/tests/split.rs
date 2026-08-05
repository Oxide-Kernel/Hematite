// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Per-op golden test: `KernelBackend::split` through `RefBackend` (T5.1).
//!
//! The trait method splits into both output slices in one call; the adapter
//! forwards `output_a` with `split_index = 0` and `output_b` with
//! `split_index = 1`.

mod split_v0 {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/split_v0.rs"));
}

mod split_v1 {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/split_v1.rs"));
}

use hematite_core::op_params::SplitParams;
use hematite_core::KernelBackend;
use hematite_ref::RefBackend;

#[test]
fn split_golden_v0() {
    let backend = RefBackend;
    let params = SplitParams {
        num_splits: split_v0::NUM_SPLITS,
        axis: split_v0::AXIS,
        input_shape: split_v0::INPUT_SHAPE,
        output_shape_a: split_v0::OUTPUT_SHAPE,
        output_shape_b: split_v1::OUTPUT_SHAPE,
    };
    let mut output_a = [0i8; 2];
    let mut output_b = [0i8; 2];
    backend
        .split(&split_v0::INPUT_DATA, &params, &mut output_a, &mut output_b)
        .expect("split kernel returned Err");
    assert_eq!(
        &output_a[..],
        &split_v0::EXPECTED_OUTPUT[..],
        "split_golden_v0: output_a mismatch"
    );
}

#[test]
fn split_golden_v1() {
    let backend = RefBackend;
    let params = SplitParams {
        num_splits: split_v1::NUM_SPLITS,
        axis: split_v1::AXIS,
        input_shape: split_v1::INPUT_SHAPE,
        output_shape_a: split_v0::OUTPUT_SHAPE,
        output_shape_b: split_v1::OUTPUT_SHAPE,
    };
    let mut output_a = [0i8; 2];
    let mut output_b = [0i8; 2];
    backend
        .split(&split_v1::INPUT_DATA, &params, &mut output_a, &mut output_b)
        .expect("split kernel returned Err");
    assert_eq!(
        &output_b[..],
        &split_v1::EXPECTED_OUTPUT[..],
        "split_golden_v1: output_b mismatch"
    );
}
