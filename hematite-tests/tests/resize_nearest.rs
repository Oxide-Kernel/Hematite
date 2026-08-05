// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Per-op golden test: `KernelBackend::resize_nearest` through `RefBackend`
//! (T5.1).

mod resize_fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/goldens/resize_nearest_neighbor.rs"
    ));
}

use hematite_core::op_params::ResizeNearestParams;
use hematite_core::KernelBackend;
use hematite_ref::RefBackend;

#[test]
fn resize_nearest_golden() {
    let backend = RefBackend;
    let params = ResizeNearestParams {
        input_shape: resize_fixture::INPUT_SHAPE,
        output_shape: resize_fixture::OUTPUT_SHAPE,
        align_corners: resize_fixture::ALIGN_CORNERS,
        half_pixel_centers: resize_fixture::HALF_PIXEL_CENTERS,
    };
    let mut output = [0i8; 16];
    backend
        .resize_nearest(&resize_fixture::INPUT_DATA, &params, &mut output)
        .expect("resize_nearest kernel returned Err");
    assert_eq!(
        &output[..],
        &resize_fixture::EXPECTED_OUTPUT[..],
        "resize_nearest_golden: mismatch"
    );
}
