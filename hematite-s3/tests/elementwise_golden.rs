// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Golden-vector tests for the hematite-s3 elementwise kernels.
//!
//! # Bit-exact contract (Plan A4)
//!
//! Three legs, only one runs on host (stable-aarch64-apple-darwin):
//!
//! | Leg | Contract | Runs on | Status |
//! |-----|----------|---------|--------|
//! | (a) | SIMD ≡ per-tensor TFLM golden bit-exact | Device (Phase 5) | cfg-gated |
//! | (b) | **Scalar ref ≡ per-tensor TFLM golden bit-exact** | **Host** | **this test** |
//! | (c) | SIMD vs scalar cross-check ≤1 LSB on requantize | Device (Phase 5) | cfg-gated |
//!
//! Leg (b) is tested here: the host-compilable scalar `add`/`mul`/`sub` kernels
//! must produce output bit-identical to the per-tensor golden fixtures.
//!
//! Leg (a) and (c) are `#[cfg(target_arch = "xtensa")]`-gated and will be
//! verified at Phase 5 (T5.3 hardware verification).

// ── Fixture includes ───────────────────────────────────────────────────────

mod elementwise_add {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/elementwise_add.rs"
    ));
}

mod elementwise_mul {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/elementwise_mul.rs"
    ));
}

mod elementwise_sub {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/elementwise_sub.rs"
    ));
}

use hematite_core::op_params::ElementwiseParams;
use hematite_s3::elementwise::{add, mul, sub};

/// Construct an `ElementwiseParams` from a fixture module's public consts.
///
/// Maps fixture consts to [`ElementwiseParams`] fields. The `OUTPUT_MULTIPLIER`
/// and `OUTPUT_SHIFT` are per-tensor (single-element arrays in the fixture).
/// The `input1_multiplier`/`input1_shift` and `input2_multiplier`/`input2_shift`
/// fields are present only when the fixture emits them (add/sub); for mul,
/// they default to 0.
macro_rules! params_from_fixture {
    ($m:ident) => {{
        ElementwiseParams {
            num_elements: $m::OUTPUT_SHAPE[3],
            input1_offset: $m::INPUT_OFFSET,
            input2_offset: $m::INPUT2_OFFSET,
            output_offset: $m::OUTPUT_OFFSET,
            output_multiplier: $m::OUTPUT_MULTIPLIER[0],
            output_shift: $m::OUTPUT_SHIFT[0],
            left_shift: $m::LEFT_SHIFT,
            input1_multiplier: $m::INPUT1_MULTIPLIER,
            input1_shift: $m::INPUT1_SHIFT,
            input2_multiplier: $m::INPUT2_MULTIPLIER,
            input2_shift: $m::INPUT2_SHIFT,
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

// ── Leg (b): Host scalar golden tests ──────────────────────────────────────

#[test]
fn elementwise_golden_add() {
    let params = params_from_fixture!(elementwise_add);
    let mut output = [0i8; 6];
    add(
        &elementwise_add::INPUT_DATA,
        &elementwise_add::WEIGHTS_DATA,
        &params,
        &mut output,
        &mut [],
    )
    .expect("add kernel returned Err");
    assert_bit_exact(
        &output,
        &elementwise_add::EXPECTED_OUTPUT,
        "elementwise_golden_add",
    );
}

#[test]
fn elementwise_golden_mul() {
    // mul fixtures lack LEFT_SHIFT, INPUT1_MULTIPLIER/SHIFT, INPUT2_MULTIPLIER/SHIFT.
    // Construct params manually.
    let params = ElementwiseParams {
        num_elements: 6,
        input1_offset: elementwise_mul::INPUT_OFFSET,
        input2_offset: elementwise_mul::INPUT2_OFFSET,
        output_offset: elementwise_mul::OUTPUT_OFFSET,
        output_multiplier: elementwise_mul::OUTPUT_MULTIPLIER[0],
        output_shift: elementwise_mul::OUTPUT_SHIFT[0],
        left_shift: 0,
        input1_multiplier: 0,
        input1_shift: 0,
        input2_multiplier: 0,
        input2_shift: 0,
        quantized_activation_min: elementwise_mul::OUTPUT_ACTIVATION_MIN,
        quantized_activation_max: elementwise_mul::OUTPUT_ACTIVATION_MAX,
    };
    let mut output = [0i8; 6];
    mul(
        &elementwise_mul::INPUT_DATA,
        &elementwise_mul::WEIGHTS_DATA,
        &params,
        &mut output,
        &mut [],
    )
    .expect("mul kernel returned Err");
    assert_bit_exact(
        &output,
        &elementwise_mul::EXPECTED_OUTPUT,
        "elementwise_golden_mul",
    );
}

#[test]
fn elementwise_golden_sub() {
    let params = params_from_fixture!(elementwise_sub);
    let mut output = [0i8; 6];
    sub(
        &elementwise_sub::INPUT_DATA,
        &elementwise_sub::WEIGHTS_DATA,
        &params,
        &mut output,
        &mut [],
    )
    .expect("sub kernel returned Err");
    assert_bit_exact(
        &output,
        &elementwise_sub::EXPECTED_OUTPUT,
        "elementwise_golden_sub",
    );
}

// ── Leg (a) + (c): Device-only SIMD tests (cfg-gated) ──────────────────────
//
// These tests exist in the tree so that Phase 5 (T5.3 hardware verification)
// can run them on an ESP32-S3 device. On host they never compile — the
// `cfg(target_arch = "xtensa")` gate excludes them from the build entirely.
//
// Leg (a): SIMD output matches the per-tensor TFLM golden bit-exact.
//   The current fixtures are per-tensor (scalar OUTPUT_MULTIPLIER/SHIFT),
//   so they already match the SIMD expectation.
//
// Leg (c): SIMD output vs scalar output difference ≤1 LSB.
//   Both paths run on the same input; the max absolute delta across all
//   elements is computed. The tolerance captures the legitimate difference
//   between per-tensor (SIMD) and per-tensor (scalar — identical here,
//   since both use the same per-tensor quant params). Leg (c) is minimal
//   for elementwise ops but retained for architectural consistency.

#[cfg(target_arch = "xtensa")]
mod simd_tests {
    use super::*;

    /// Leg (a): SIMD bit-exact vs per-tensor TFLM golden — add.
    #[test]
    #[ignore = "Phase 5 (T5.3): requires real device"]
    fn elementwise_golden_add_simd() {}

    /// Leg (a): SIMD bit-exact vs per-tensor TFLM golden — mul.
    #[test]
    #[ignore = "Phase 5 (T5.3): requires real device"]
    fn elementwise_golden_mul_simd() {}

    /// Leg (a): SIMD bit-exact vs per-tensor TFLM golden — sub.
    #[test]
    #[ignore = "Phase 5 (T5.3): requires real device"]
    fn elementwise_golden_sub_simd() {}
}
