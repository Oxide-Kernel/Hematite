// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Per-op golden test: `KernelBackend::pad` through `RefBackend` (T5.1).

mod pad_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/pad.rs"));
}

use hematite_core::op_params::PadParams;
use hematite_core::KernelBackend;
use hematite_ref::RefBackend;

#[test]
fn pad_golden() {
    let backend = RefBackend;
    let params = PadParams {
        input_shape: pad_fixture::INPUT_SHAPE,
        output_shape: pad_fixture::OUTPUT_SHAPE,
        left_padding: [0, pad_fixture::PAD_TOP, pad_fixture::PAD_LEFT, 0],
        left_padding_count: 4,
        right_padding: [0, pad_fixture::PAD_BOTTOM, pad_fixture::PAD_RIGHT, 0],
        right_padding_count: 4,
    };
    let mut output = [0i8; 16];
    backend
        .pad(&pad_fixture::INPUT_DATA, &params, &mut output)
        .expect("pad kernel returned Err");
    assert_eq!(
        &output[..],
        &pad_fixture::EXPECTED_OUTPUT[..],
        "pad_golden: mismatch"
    );
}
