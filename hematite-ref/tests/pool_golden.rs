// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Golden-vector tests for the Pooling scalar reference kernels.
//!
//! Loads TFLM-generated const-array fixtures via `include!()` and asserts
//! bit-exact output match for every golden in the corpus.
//!
//! Test naming convention: `pool_golden_<op>` so that
//! `cargo test -p hematite-ref -- pool_golden` matches all tests.

// ── Fixture includes ───────────────────────────────────────────────────────

mod average_pool_fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/average_pool_2d.rs"
    ));
}

mod max_pool_fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/max_pool_2d.rs"
    ));
}

use hematite_core::op_params::{FusedActivation, Padding, PoolParams};
use hematite_ref::pool::{average_pool_2d, global_average_pool_2d, max_pool_2d};

/// Construct a `PoolParams` from a fixture module's public consts.
macro_rules! params_from_fixture {
    ($m:ident) => {{
        let pad = if $m::PAD_WIDTH > 0 || $m::PAD_HEIGHT > 0 {
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
            padding: pad,
            activation: FusedActivation::None,
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
fn pool_golden_average() {
    let params = params_from_fixture!(average_pool_fixture);
    let mut output = [0i8; 4];
    average_pool_2d(
        &average_pool_fixture::INPUT_DATA,
        &params,
        &mut output,
        &mut [],
    )
    .expect("average_pool_2d kernel returned Err");
    assert_bit_exact(
        &output,
        &average_pool_fixture::EXPECTED_OUTPUT,
        "pool_golden_average",
    );
}

#[test]
fn pool_golden_max() {
    let params = params_from_fixture!(max_pool_fixture);
    let mut output = [0i8; 4];
    max_pool_2d(
        &max_pool_fixture::INPUT_DATA,
        &params,
        &mut output,
        &mut [],
    )
    .expect("max_pool_2d kernel returned Err");
    assert_bit_exact(
        &output,
        &max_pool_fixture::EXPECTED_OUTPUT,
        "pool_golden_max",
    );
}

#[test]
fn pool_golden_global_average() {
    // Hand-computed test: 2×2 single-channel input [0, 1, 2, 3] shaped [1, 2, 2, 1].
    // Global average pools over full 2×2 spatial extent.
    // Sum = 0+1+2+3 = 6. Pool size = 4.
    // Round-half-away-from-zero: 6/4 = 1.5 → 2.
    // Expected output: single-element [2].
    let params = PoolParams {
        input_shape: [1, 2, 2, 1],
        output_shape: [1, 1, 1, 1],
        filter_width: 2,
        filter_height: 2,
        stride_width: 1,
        stride_height: 1,
        padding: Padding::Valid,
        activation: FusedActivation::None,
        quantized_activation_min: -128,
        quantized_activation_max: 127,
    };
    let input: [i8; 4] = [0, 1, 2, 3];
    let mut output = [0i8; 1];
    global_average_pool_2d(&input, &params, &mut output, &mut [])
        .expect("global_average_pool_2d kernel returned Err");
    // Expected: sum=6, count=4, round-half-away-from-zero → 2.
    assert_bit_exact(&output, &[2], "pool_golden_global_average");
}
