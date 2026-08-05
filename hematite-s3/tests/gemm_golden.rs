// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Golden-vector test for the hematite-s3 FullyConnected/GEMM kernel.
//!
//! # Bit-exact contract (Plan A4)
//!
//! | Leg | Contract | Runs on | Status |
//! |-----|----------|---------|--------|
//! | (a) | SIMD ≡ per-tensor TFLM golden bit-exact | Device (Phase 5) | cfg-gated |
//! | (b) | **Scalar ref ≡ per-channel TFLM golden bit-exact** | **Host** | **this test** |
//! | (c) | SIMD vs scalar cross-check ≤1 LSB on requantize | Device (Phase 5) | cfg-gated |

// ── Fixture include ─────────────────────────────────────────────────────────

mod fully_connected {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/fully_connected.rs"
    ));
}

use hematite_core::op_params::FullyConnectedParams;
use hematite_s3::gemm::fully_connected;

macro_rules! params_from_fixture {
    ($m:ident) => {{
        FullyConnectedParams {
            input_dim: $m::ACCUM_DEPTH,
            output_dim: $m::OUTPUT_DEPTH,
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
fn gemm_golden() {
    let params = params_from_fixture!(fully_connected);
    let mut output = [0i8; 3];
    fully_connected(
        &fully_connected::INPUT_DATA,
        &fully_connected::WEIGHTS_DATA,
        &fully_connected::BIAS_DATA,
        &params,
        &mut output,
        &mut [],
    )
    .expect("fully_connected kernel returned Err");
    assert_bit_exact(&output, &fully_connected::EXPECTED_OUTPUT, "gemm_golden");
}

// ── Leg (a) + (c): Device-only SIMD tests (cfg-gated) ────────────────────────

#[cfg(target_arch = "xtensa")]
mod simd_tests {
    /// Leg (a): SIMD bit-exact vs per-tensor TFLM golden.
    #[test]
    #[ignore = "Phase 5 (T5.3): requires per-tensor golden fixture + real device"]
    fn gemm_golden_simd() {}
}
