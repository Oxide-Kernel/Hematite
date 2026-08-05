// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Per-op golden test: `KernelBackend::transpose` through `RefBackend`
//! (T5.1).
//!
//! Transpose is a pure data-movement op (no arithmetic); the adapter
//! implements it inline. Fixture: swap height/width dims with perm
//! [0, 2, 1, 3].

mod transpose_fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/goldens/transpose.rs"
    ));
}

use hematite_core::op_params::TransposeParams;
use hematite_core::KernelBackend;
use hematite_ref::RefBackend;

#[test]
fn transpose_golden() {
    let backend = RefBackend;
    let params = TransposeParams {
        input_shape: transpose_fixture::INPUT_SHAPE,
        perm: [
            transpose_fixture::PERM_0,
            transpose_fixture::PERM_1,
            transpose_fixture::PERM_2,
            transpose_fixture::PERM_3,
            4,
            5,
            6,
            7,
        ],
        perm_count: 4,
    };
    let mut output = [0i8; 6];
    backend
        .transpose(&transpose_fixture::INPUT_DATA, &params, &mut output)
        .expect("transpose kernel returned Err");
    assert_eq!(
        &output[..],
        &transpose_fixture::EXPECTED_OUTPUT[..],
        "transpose_golden: mismatch"
    );
}
