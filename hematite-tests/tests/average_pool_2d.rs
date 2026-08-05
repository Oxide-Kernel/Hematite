// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Per-op golden test: `KernelBackend::average_pool_2d` through `RefBackend`
//! (T5.1).

mod average_pool_fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/goldens/average_pool_2d.rs"
    ));
}

use hematite_core::op_params::{FusedActivation, Padding, PoolParams};
use hematite_core::KernelBackend;
use hematite_ref::RefBackend;

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

#[test]
fn average_pool_2d_golden() {
    let backend = RefBackend;
    let params = params_from_fixture!(average_pool_fixture);
    let mut output = [0i8; 4];
    backend
        .average_pool_2d(&average_pool_fixture::INPUT_DATA, &params, &mut output)
        .expect("average_pool_2d kernel returned Err");
    assert_bit_exact(
        &output,
        &average_pool_fixture::EXPECTED_OUTPUT,
        "average_pool_2d_golden",
    );
}
