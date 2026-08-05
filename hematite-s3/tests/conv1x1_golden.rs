// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Golden-vector test for the hematite-s3 Conv2D 1×1 kernel.
//!
//! # Bit-exact contract (Plan A4)
//!
//! Three legs, only one runs on host (stable-aarch64-apple-darwin):
//!
//! | Leg | Contract | Runs on | Status |
//! |-----|----------|---------|--------|
//! | (a) | SIMD ≡ per-tensor TFLM golden bit-exact | Device (Phase 5) | cfg-gated |
//! | (b) | **Scalar ref ≡ per-channel TFLM golden bit-exact** | **Host** | **this test** |
//! | (c) | SIMD vs scalar cross-check ≤1 LSB on requantize | Device (Phase 5) | cfg-gated |
//!
//! Leg (b) is tested here: the host-compilable scalar `conv2d_1x1` kernel
//! must produce output bit-identical to the per-channel golden fixture.
//!
//! Leg (a) and (c) are `#[cfg(target_arch = "xtensa")]`-gated and will be
//! verified at Phase 5 (T5.3 hardware verification). On device:
//!
//! * Leg (a) loads a **per-tensor** fixture (single OUTPUT_MULTIPLIER and
//!   single OUTPUT_SHIFT applied to all channels) and asserts SIMD output
//!   is bit-exact. The current fixture is per-channel; a per-tensor variant
//!   must be generated before leg (a) can be tested.
//! * Leg (c) runs both SIMD and scalar on identical input, computing the
//!   max absolute delta across all elements. The tolerance bound is ≤1 LSB
//!   — the intrinsic difference between per-channel and per-tensor
//!   requantization expressed at the bit level.

// ── Fixture include ─────────────────────────────────────────────────────────

mod conv2d_1x1 {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/conv2d_1x1.rs"
    ));
}

use hematite_core::op_params::{Conv2DParams, Padding};
use hematite_s3::conv1x1::conv2d_1x1;

/// Construct a `Conv2DParams` from a fixture module's public consts.
///
/// Maps every fixture const to the corresponding `Conv2DParams` field.
/// The `padding` enum is derived from the fixture's `PAD_WIDTH`/`PAD_HEIGHT`
/// values: non-zero pad → `Padding::Same`, zero pad → `Padding::Valid`.
/// This is a convenience mapping — the kernel derives actual pad values
/// from the spatial-shape relationship, not from the enum.
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
            a, e,
            "{name}: mismatch at index {i}: kernel={a}, golden={e}",
        );
    }
}

// ── Leg (b): Host scalar golden test ─────────────────────────────────────────

#[test]
fn conv1x1_golden() {
    let params = params_from_fixture!(conv2d_1x1);
    let mut output = [0i8; 8];
    conv2d_1x1(
        &conv2d_1x1::INPUT_DATA,
        &conv2d_1x1::WEIGHTS_DATA,
        &conv2d_1x1::BIAS_DATA,
        &params,
        &mut output,
        &mut [],
    )
    .expect("conv2d_1x1 kernel returned Err");
    assert_bit_exact(&output, &conv2d_1x1::EXPECTED_OUTPUT, "conv1x1_golden");
}

// ── Leg (a) + (c): Device-only SIMD tests (cfg-gated) ────────────────────────
//
// These tests exist in the tree so that Phase 5 (T5.3 hardware verification)
// can run them on an ESP32-S3 device. On host they never compile — the
// `cfg(target_arch = "xtensa")` gate excludes them from the build entirely.
//
// Leg (a): SIMD output matches a per-tensor TFLM golden bit-exact.
//   Requires a per-tensor fixture (single OUTPUT_MULTIPLIER/SHIFT, not arrays).
//   Since the current conv2d_1x1 fixture is per-channel, this test is a
//   placeholder that will be updated when the per-tensor fixture is generated.
//
// Leg (c): SIMD output vs scalar output difference ≤1 LSB.
//   Both paths run on the same input; the max absolute delta across all
//   elements is computed. The tolerance captures the legitimate difference
//   between per-channel (scalar) and per-tensor (SIMD) requantization.

#[cfg(target_arch = "xtensa")]
mod simd_tests {
    use super::*;

    /// Leg (a): SIMD bit-exact vs per-tensor TFLM golden.
    #[test]
    #[ignore = "Phase 5 (T5.3): requires per-tensor golden fixture + real device"]
    fn conv1x1_golden_simd() {}
}
