// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Golden-vector test for the hematite-s3 mean reduction kernel.
//!
//! # Bit-exact contract (Plan A4)
//!
//! | Leg | Contract | Runs on | Status |
//! |-----|----------|---------|--------|
//! | (a) | SIMD ≡ per-tensor TFLM golden bit-exact | Device (Phase 5) | cfg-gated |
//! | (b) | **Scalar ref ≡ per-channel TFLM golden bit-exact** | **Host** | **this test** |
//! | (c) | SIMD vs scalar cross-check ≤1 LSB | Device (Phase 5) | cfg-gated |

mod mean_fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/mean.rs"
    ));
}

use hematite_core::op_params::ReduceParams;
use hematite_s3::reductions::mean;

/// Construct `ReduceParams` from the mean fixture.
macro_rules! reduce_params_from_fixture {
    ($m:ident) => {{
        ReduceParams {
            keep_dims: true,
            axis: [$m::AXIS_0 as i16, 0, 0, 0],
            axis_count: $m::AXIS_COUNT as i8,
            input_shape: $m::INPUT_SHAPE,
            output_shape: $m::OUTPUT_SHAPE,
            output_type: 0,
            input_offset: $m::INPUT_OFFSET,
            output_offset: $m::OUTPUT_OFFSET,
            output_multiplier: $m::OUTPUT_MULTIPLIER[0],
            output_shift: $m::OUTPUT_SHIFT[0],
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
fn mean_golden() {
    let params = reduce_params_from_fixture!(mean_fixture);
    let mut output = [0i8; 6];
    mean(&mean_fixture::INPUT_DATA, &params, &mut output)
        .expect("mean kernel returned Err");
    assert_bit_exact(&output, &mean_fixture::EXPECTED_OUTPUT, "mean_golden");
}

// ── Leg (a) + (c): Device-only SIMD tests (cfg-gated) ───────────────────────

#[cfg(target_arch = "xtensa")]
mod simd_tests {
    #[test]
    #[ignore = "Phase 5 (T5.3): requires real device"]
    fn mean_golden_simd() {}
}
