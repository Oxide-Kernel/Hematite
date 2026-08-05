// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Per-op golden test: `KernelBackend::reshape` through `RefBackend` (T5.1).
//!
//! Reshape is a flat copy in int8 TFLM (metadata-only op); the adapter
//! implements it inline.

mod reshape_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/reshape.rs"));
}

use hematite_core::op_params::ReshapeParams;
use hematite_core::KernelBackend;
use hematite_ref::RefBackend;

#[test]
fn reshape_golden() {
    let backend = RefBackend;
    let params = ReshapeParams {
        shape: reshape_fixture::OUTPUT_SHAPE,
        shape_count: 4,
    };
    let mut output = [0i8; 8];
    backend
        .reshape(&reshape_fixture::INPUT_DATA, &params, &mut output)
        .expect("reshape kernel returned Err");
    assert_eq!(
        &output[..],
        &reshape_fixture::EXPECTED_OUTPUT[..],
        "reshape_golden: mismatch"
    );
}
