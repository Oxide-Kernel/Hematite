// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Golden-vector tests for the ResizeNearestNeighbor scalar reference kernel.
//!
//! Loads TFLM-generated const-array fixtures via `include!()` and asserts
//! bit-exact output match. Additional hand-computed tests cover downscale
//! cases not present in the golden fixture.
//!
//! Test naming convention: `resize_golden_<case>` so that
//! `cargo test -p hematite-ref -- resize_golden` matches all tests.

// ── Fixture includes ───────────────────────────────────────────────────────

mod resize_nearest_neighbor {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/resize_nearest_neighbor.rs"
    ));
}

use hematite_core::op_params::ResizeNearestParams;
use hematite_ref::resize::resize_nearest_neighbor;

/// Construct a `ResizeNearestParams` from a fixture module's public consts.
macro_rules! params_from_fixture {
    ($m:ident) => {{
        ResizeNearestParams {
            input_shape: $m::INPUT_SHAPE,
            output_shape: $m::OUTPUT_SHAPE,
            align_corners: $m::ALIGN_CORNERS,
            half_pixel_centers: $m::HALF_PIXEL_CENTERS,
        }
    }};
}

/// Assert that `actual` matches `expected` element-for-element, printing
/// the index and values of the first mismatch.
fn assert_bit_exact(actual: &[i8], expected: &[i8], name: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{name}: output length {} != expected length {}",
        actual.len(),
        expected.len(),
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            a,
            e,
            "{name}: mismatch at index {i}: kernel={a}, golden={e}",
        );
    }
}

// ── Golden tests ───────────────────────────────────────────────────────────

/// Upscale 2×2 → 4×4, asymmetric/floor — golden fixture.
#[test]
fn resize_golden_upscale() {
    let params = params_from_fixture!(resize_nearest_neighbor);
    let mut output = [0i8; 16];
    resize_nearest_neighbor(
        &resize_nearest_neighbor::INPUT_DATA,
        &params,
        &mut output,
        &mut [],
    )
    .expect("resize_nearest_neighbor kernel returned Err");
    assert_bit_exact(
        &output,
        &resize_nearest_neighbor::EXPECTED_OUTPUT,
        "resize_golden_upscale",
    );
}

/// Downscale 4×4 → 2×2, asymmetric/floor — hand-computed expected output.
///
/// TFLM coordinate mapping: `src = floor(dst * in_size / out_size)`.
/// Input is a 4×4 grid of [1..16] in row-major order; the nearest-neighbor
/// picks every other row/col starting at (0,0):
///
/// ```text
/// Input 4×4:            Output 2×2 (picks marked with *):
///  [ 1*  2   3*  4 ]     [ 1   3 ]
///  [ 5   6   7   8 ]     [ 9  11 ]
///  [ 9* 10  11* 12 ]
///  [13  14  15  16 ]
/// ```
///
/// Derivation:
///   dst[0,0]: src_h = floor(0*4/2)=0, src_w = floor(0*4/2)=0 → input[0,0] = 1
///   dst[0,1]: src_h = floor(0*4/2)=0, src_w = floor(1*4/2)=2 → input[0,2] = 3
///   dst[1,0]: src_h = floor(1*4/2)=2, src_w = floor(0*4/2)=0 → input[2,0] = 9
///   dst[1,1]: src_h = floor(1*4/2)=2, src_w = floor(1*4/2)=2 → input[2,2] = 11
#[test]
fn resize_golden_downscale() {
    let input: [i8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let params = ResizeNearestParams {
        input_shape: [1, 4, 4, 1],
        output_shape: [1, 2, 2, 1],
        align_corners: 0,
        half_pixel_centers: 0,
    };
    let mut output = [0i8; 4];
    resize_nearest_neighbor(&input, &params, &mut output, &mut [])
        .expect("resize_nearest_neighbor kernel returned Err");
    assert_bit_exact(&output, &[1, 3, 9, 11], "resize_golden_downscale");
}
