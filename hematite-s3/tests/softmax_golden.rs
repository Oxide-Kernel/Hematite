// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Golden-vector test for the hematite-s3 softmax kernel.
//!
//! Softmax is SCALAR ONLY (plan T3.3: no SIMD benefit on 3–100 classes).
//! There are no SIMD tests for softmax.

mod softmax_fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/softmax.rs"
    ));
}

use hematite_core::op_params::SoftmaxParams;
use hematite_s3::softmax::softmax;

/// Construct `SoftmaxParams` from the fixture.
macro_rules! softmax_params_from_fixture {
    ($m:ident) => {{
        SoftmaxParams {
            num_rows: $m::INPUT_SHAPE[0] * $m::INPUT_SHAPE[1] * $m::INPUT_SHAPE[2],
            row_size: $m::INPUT_SHAPE[3],
            input_multiplier: $m::INPUT_MULTIPLIER,
            input_left_shift: $m::LEFT_SHIFT,
            diff_min: $m::DIFF_MIN,
            input_offset: $m::INPUT_OFFSET,
            output_offset: $m::OUTPUT_OFFSET,
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

// ── Golden test ─────────────────────────────────────────────────────────────

#[test]
fn softmax_golden() {
    let params = softmax_params_from_fixture!(softmax_fixture);
    let mut output = [0i8; 5];
    softmax(
        &softmax_fixture::INPUT_DATA,
        &params,
        &mut output,
        &mut [],
    )
    .expect("softmax kernel returned Err");
    assert_bit_exact(&output, &softmax_fixture::EXPECTED_OUTPUT, "softmax_golden");
}
