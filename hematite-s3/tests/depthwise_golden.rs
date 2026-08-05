// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Golden-vector test for the hematite-s3 DepthwiseConv2D kernel.
//!
//! # Bit-exact contract (Plan A4)
//!
//! | Leg | Contract | Runs on | Status |
//! |-----|----------|---------|--------|
//! | (a) | SIMD ≡ per-tensor TFLM golden bit-exact | Device (Phase 5) | cfg-gated |
//! | (b) | **Scalar ref ≡ per-channel TFLM golden bit-exact** | **Host** | **this test** |
//! | (c) | SIMD vs scalar cross-check ≤1 LSB on requantize | Device (Phase 5) | cfg-gated |

// ── Fixture include ─────────────────────────────────────────────────────────

mod depthwise_conv2d {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/depthwise_conv2d.rs"
    ));
}

use hematite_core::op_params::{DepthwiseConv2DParams, Padding};
use hematite_s3::depthwise::depthwise_conv2d;

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
            a, e,
            "{name}: mismatch at index {i}: kernel={a}, golden={e}",
        );
    }
}

// ── Leg (b): Host scalar golden test ─────────────────────────────────────────

#[test]
fn depthwise_golden() {
    let params = params_from_fixture!(depthwise_conv2d);
    let mut output = [0i8; 36];
    depthwise_conv2d(
        &depthwise_conv2d::INPUT_DATA,
        &depthwise_conv2d::WEIGHTS_DATA,
        &depthwise_conv2d::BIAS_DATA,
        &params,
        &mut output,
        &mut [],
    )
    .expect("depthwise_conv2d kernel returned Err");
    assert_bit_exact(
        &output,
        &depthwise_conv2d::EXPECTED_OUTPUT,
        "depthwise_golden",
    );
}

// ── Leg (a) + (c): Device-only SIMD tests (cfg-gated) ────────────────────────

#[cfg(target_arch = "xtensa")]
mod simd_tests {
    /// Leg (a): SIMD bit-exact vs per-tensor TFLM golden.
    #[test]
    #[ignore = "Phase 5 (T5.3): requires per-tensor golden fixture + real device"]
    fn depthwise_golden_simd() {}
}
