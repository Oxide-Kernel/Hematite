// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Golden-vector tests for the FullyConnected scalar reference kernel.
//!
//! Loads TFLM-generated const-array fixture via `include!()` and asserts
//! bit-exact output match.
//!
//! Test naming convention: `fully_connected_golden` so that
//! `cargo test -p hematite-ref -- fully_connected_golden` matches.

// ── Fixture includes ───────────────────────────────────────────────────────

mod fc_fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/fully_connected.rs"
    ));
}

use hematite_core::op_params::FullyConnectedParams;
use hematite_ref::fully_connected::fully_connected;

/// Construct a `FullyConnectedParams` from the fixture module's public consts.
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
fn fully_connected_golden() {
    let params = params_from_fixture!(fc_fixture);
    let mut output = [0i8; 3];
    fully_connected(
        &fc_fixture::INPUT_DATA,
        &fc_fixture::WEIGHTS_DATA,
        &fc_fixture::BIAS_DATA,
        &params,
        &mut output,
        &mut [],
    )
    .expect("fully_connected kernel returned Err");
    assert_bit_exact(&output, &fc_fixture::EXPECTED_OUTPUT, "fully_connected_golden");
}
