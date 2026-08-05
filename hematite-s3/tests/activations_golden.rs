// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Golden-vector tests for hematite-s3 activation kernels.
//!
//! # Bit-exact contract (Plan A4)
//!
//! | Leg | Contract | Runs on | Status |
//! |-----|----------|---------|--------|
//! | (a) | SIMD ≡ per-tensor TFLM golden bit-exact | Device (Phase 5) | cfg-gated |
//! | (b) | **Scalar ref ≡ per-channel TFLM golden bit-exact** | **Host** | **this test** |
//! | (c) | SIMD vs scalar cross-check ≤1 LSB | Device (Phase 5) | cfg-gated |

// ── Fixture includes ────────────────────────────────────────────────────────

mod relu_fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/relu.rs"
    ));
}

mod relu6_fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/relu6.rs"
    ));
}

mod hard_swish_fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/hard_swish.rs"
    ));
}

use hematite_core::op_params::ActivationParams;
use hematite_s3::activations;

/// Construct `ActivationParams` from a relu-family fixture.
macro_rules! activation_params_from_fixture {
    ($m:ident) => {{
        ActivationParams {
            input_offset: $m::INPUT_ZERO_POINT,
            output_offset: $m::OUTPUT_ZERO_POINT,
            output_multiplier: 1073741824, // default (1.0 → Q0.31)
            output_shift: 1,
            quantized_activation_min: -128,
            quantized_activation_max: 127,
            input_multiplier: 0,
            input_left_shift: 0,
            input_range_radius: 0,
            output_multiplier_alpha: 0,
            output_shift_alpha: 0,
            output_multiplier_identity: 0,
            output_shift_identity: 0,
            alpha_offset: 0,
            alpha_data: &[],
            output_multiplier_1: 0,
            output_shift_1: 0,
            output_multiplier_2: 0,
            output_shift_2: 0,
            reluish_multiplier_fixedpoint_int16: 0,
            reluish_multiplier_exponent: 0,
            output_multiplier_fixedpoint_int16: 0,
            output_multiplier_exponent: 0,
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

// ── Leg (b): Host scalar golden tests ────────────────────────────────────────

#[test]
fn relu_golden() {
    let params = activation_params_from_fixture!(relu_fixture);
    let mut output = [0i8; 8];
    activations::relu(&relu_fixture::INPUT_DATA, &params, &mut output, &mut [])
        .expect("relu kernel returned Err");
    assert_bit_exact(&output, &relu_fixture::EXPECTED_OUTPUT, "relu_golden");
}

#[test]
fn relu6_golden() {
    let params = activation_params_from_fixture!(relu6_fixture);
    let mut output = [0i8; 8];
    activations::relu6(
        &relu6_fixture::INPUT_DATA,
        &params,
        &mut output,
        &mut [],
        relu6_fixture::QUANTIZED_SIX,
    )
    .expect("relu6 kernel returned Err");
    assert_bit_exact(&output, &relu6_fixture::EXPECTED_OUTPUT, "relu6_golden");
}

#[test]
fn hard_swish_golden() {
    let params = activation_params_from_fixture!(hard_swish_fixture);
    let mut output = [0i8; 8];
    activations::hard_swish(
        &hard_swish_fixture::INPUT_DATA,
        &params,
        &mut output,
        &mut [],
    )
    .expect("hard_swish kernel returned Err");
    assert_bit_exact(&output, &hard_swish_fixture::EXPECTED_OUTPUT, "hard_swish_golden");
}

// ── Leg (a) + (c): Device-only SIMD tests (cfg-gated) ───────────────────────

#[cfg(target_arch = "xtensa")]
mod simd_tests {
    #[test]
    #[ignore = "Phase 5 (T5.3): requires real device"]
    fn relu_golden_simd() {}

    #[test]
    #[ignore = "Phase 5 (T5.3): requires real device"]
    fn relu6_golden_simd() {}
}
