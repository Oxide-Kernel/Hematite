// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Per-op golden test: `KernelBackend::slice` through `RefBackend` (T5.1).

mod slice_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/slice.rs"));
}

use hematite_core::op_params::SliceParams;
use hematite_core::KernelBackend;
use hematite_ref::RefBackend;

#[test]
fn slice_golden() {
    let backend = RefBackend;
    let params = SliceParams {
        input_shape: slice_fixture::INPUT_SHAPE,
        begin: [
            slice_fixture::BEGIN_0,
            slice_fixture::BEGIN_1,
            slice_fixture::BEGIN_2,
            slice_fixture::BEGIN_3,
        ],
        size: [
            slice_fixture::SIZE_0,
            slice_fixture::SIZE_1,
            slice_fixture::SIZE_2,
            slice_fixture::SIZE_3,
        ],
    };
    let mut output = [0i8; 4];
    backend
        .slice(&slice_fixture::INPUT_DATA, &params, &mut output)
        .expect("slice kernel returned Err");
    assert_eq!(
        &output[..],
        &slice_fixture::EXPECTED_OUTPUT[..],
        "slice_golden: mismatch"
    );
}
