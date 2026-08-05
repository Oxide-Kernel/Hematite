// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Golden-vector tests for hematite-s3 pooling kernels.
//!
//! # Bit-exact contract (Plan A4)
//!
//! Three legs, only (b) runs on host:
//!
//! | Leg | Contract | Runs on | Status |
//! |-----|----------|---------|--------|
//! | (a) | SIMD ≡ per-tensor TFLM golden bit-exact | Device (Phase 5) | cfg-gated |
//! | (b) | **Scalar ref ≡ per-channel TFLM golden bit-exact** | **Host** | **this test** |
//! | (c) | SIMD vs scalar cross-check ≤1 LSB | Device (Phase 5) | cfg-gated |

// ── Fixture includes ────────────────────────────────────────────────────────

mod average_pool_2d {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/average_pool_2d.rs"
    ));
}

mod max_pool_2d {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/max_pool_2d.rs"
    ));
}

use hematite_core::op_params::{FusedActivation, Padding, PoolParams};
use hematite_s3::pool;

/// Construct a `PoolParams` from a fixture module's public consts.
macro_rules! pool_params_from_fixture {
    ($m:ident) => {{
        let pad_val = if $m::PAD_WIDTH > 0 || $m::PAD_HEIGHT > 0 {
            Padding::Same
        } else {
            Padding::Valid
        };
        PoolParams {
            input_shape: $m::INPUT_SHAPE,
            output_shape: $m::OUTPUT_SHAPE,
            filter_width: $m::FILTER_WIDTH,
            filter_height: $m::FILTER_HEIGHT,
            stride_width: $m::STRIDE_WIDTH,
            stride_height: $m::STRIDE_HEIGHT,
            padding: pad_val,
            activation: FusedActivation::None,
            quantized_activation_min: $m::OUTPUT_ACTIVATION_MIN,
            quantized_activation_max: $m::OUTPUT_ACTIVATION_MAX,
        }
    }};
}

/// Assert that `actual` matches `expected` element-for-element.
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

// ── Leg (b): Host scalar golden tests ────────────────────────────────────────

#[test]
fn avg_pool_golden() {
    let params = pool_params_from_fixture!(average_pool_2d);
    let mut output = [0i8; 4];
    pool::average_pool_2d(&average_pool_2d::INPUT_DATA, &params, &mut output, &mut [])
        .expect("average_pool_2d kernel returned Err");
    assert_bit_exact(&output, &average_pool_2d::EXPECTED_OUTPUT, "avg_pool_golden");
}

#[test]
fn max_pool_golden() {
    let params = pool_params_from_fixture!(max_pool_2d);
    let mut output = [0i8; 4];
    pool::max_pool_2d(&max_pool_2d::INPUT_DATA, &params, &mut output, &mut [])
        .expect("max_pool_2d kernel returned Err");
    assert_bit_exact(&output, &max_pool_2d::EXPECTED_OUTPUT, "max_pool_golden");
}

// ── Leg (a) + (c): Device-only SIMD tests (cfg-gated) ───────────────────────

#[cfg(target_arch = "xtensa")]
mod simd_tests {
    /// Leg (a): SIMD bit-exact vs per-tensor TFLM golden.
    #[test]
    #[ignore = "Phase 5 (T5.3): requires real device"]
    fn avg_pool_golden_simd() {}

    #[test]
    #[ignore = "Phase 5 (T5.3): requires real device"]
    fn max_pool_golden_simd() {}
}
