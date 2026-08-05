// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Per-op golden test: `KernelBackend::conv2d` through `RefBackend` (T5.1).
//!
//! Loads the conv2d fixtures and asserts bit-exact RefBackend output for
//! both the 1x1 and 3x3 golden cases.

mod conv2d_1x1 {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/goldens/conv2d_1x1.rs"
    ));
}

mod conv2d_3x3 {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/goldens/conv2d_3x3.rs"
    ));
}

use hematite_core::op_params::{Conv2DParams, Padding};
use hematite_core::KernelBackend;
use hematite_ref::RefBackend;

/// Construct a `Conv2DParams` from a fixture module's public consts.
macro_rules! params_from_fixture {
    ($m:ident) => {{
        let pad = if $m::PAD_WIDTH > 0 || $m::PAD_HEIGHT > 0 {
            Padding::Same
        } else {
            Padding::Valid
        };
        Conv2DParams {
            input_shape: $m::INPUT_SHAPE,
            filter_shape: $m::FILTER_SHAPE,
            output_shape: $m::OUTPUT_SHAPE,
            padding: pad,
            stride_width: $m::STRIDE_WIDTH,
            stride_height: $m::STRIDE_HEIGHT,
            dilation_width_factor: $m::DILATION_W,
            dilation_height_factor: $m::DILATION_H,
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

#[test]
fn conv2d_golden_1x1() {
    let backend = RefBackend;
    let params = params_from_fixture!(conv2d_1x1);
    let mut output = [0i8; 8];
    backend
        .conv2d(
            &conv2d_1x1::INPUT_DATA,
            &conv2d_1x1::WEIGHTS_DATA,
            &conv2d_1x1::BIAS_DATA,
            &params,
            &mut output,
            &mut [],
        )
        .expect("conv2d kernel returned Err");
    assert_bit_exact(&output, &conv2d_1x1::EXPECTED_OUTPUT, "conv2d_golden_1x1");
}

#[test]
fn conv2d_golden_3x3() {
    let backend = RefBackend;
    let params = params_from_fixture!(conv2d_3x3);
    let mut output = [0i8; 16];
    backend
        .conv2d(
            &conv2d_3x3::INPUT_DATA,
            &conv2d_3x3::WEIGHTS_DATA,
            &conv2d_3x3::BIAS_DATA,
            &params,
            &mut output,
            &mut [],
        )
        .expect("conv2d kernel returned Err");
    assert_bit_exact(&output, &conv2d_3x3::EXPECTED_OUTPUT, "conv2d_golden_3x3");
}
