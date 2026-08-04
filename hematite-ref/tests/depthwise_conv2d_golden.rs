// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Golden-vector tests for the DepthwiseConv2D scalar reference kernel.
//!
//! Loads TFLM-generated const-array fixture via `include!()` and asserts
//! bit-exact output match.
//!
//! Test naming convention: `depthwise_conv2d_golden` so that
//! `cargo test -p hematite-ref -- depthwise_conv2d_golden` matches.

// ── Fixture includes ───────────────────────────────────────────────────────

mod depthwise_fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/depthwise_conv2d.rs"
    ));
}

use hematite_core::op_params::{DepthwiseConv2DParams, Padding};
use hematite_ref::depthwise_conv::depthwise_conv2d;

/// Construct a `DepthwiseConv2DParams` from the fixture module's public consts.
macro_rules! params_from_fixture {
    ($m:ident) => {{
        let pad = if $m::PAD_WIDTH > 0 || $m::PAD_HEIGHT > 0 {
            Padding::Same
        } else {
            Padding::Valid
        };
        DepthwiseConv2DParams {
            input_shape: $m::INPUT_SHAPE,
            filter_shape: $m::FILTER_SHAPE,
            output_shape: $m::OUTPUT_SHAPE,
            padding: pad,
            stride_width: $m::STRIDE_WIDTH,
            stride_height: $m::STRIDE_HEIGHT,
            dilation_width_factor: $m::DILATION_W,
            dilation_height_factor: $m::DILATION_H,
            depth_multiplier: $m::DEPTH_MULTIPLIER,
            input_offset: $m::INPUT_OFFSET,
            weights_offset: 0,
            output_offset: $m::OUTPUT_OFFSET,
            output_multiplier_per_channel: &$m::OUTPUT_MULTIPLIER,
            output_shift_per_channel: &$m::OUTPUT_SHIFT,
            quantized_activation_min: $m::OUTPUT_ACTIVATION_MIN,
            quantized_activation_max: $m::OUTPUT_ACTIVATION_MAX,
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

#[test]
fn depthwise_conv2d_golden_3x3() {
    let params = params_from_fixture!(depthwise_fixture);
    let mut output = [0i8; 36];
    depthwise_conv2d(
        &depthwise_fixture::INPUT_DATA,
        &depthwise_fixture::WEIGHTS_DATA,
        &depthwise_fixture::BIAS_DATA,
        &params,
        &mut output,
        &mut [],
    )
    .expect("depthwise_conv2d kernel returned Err");
    assert_bit_exact(&output, &depthwise_fixture::EXPECTED_OUTPUT, "depthwise_conv2d_golden_3x3");
}
