// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Per-op golden test: `KernelBackend::softmax` through `RefBackend` (T5.1).

mod softmax5 {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/goldens/softmax.rs"
    ));
}

use hematite_core::op_params::SoftmaxParams;
use hematite_core::KernelBackend;
use hematite_ref::RefBackend;

/// Construct a `SoftmaxParams` from a fixture module's public consts.
macro_rules! params_from_fixture {
    ($m:ident) => {{
        SoftmaxParams {
            num_rows: 1,
            row_size: $m::OUTPUT_SHAPE[3],
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
fn softmax_golden_5elem() {
    let backend = RefBackend;
    let params = params_from_fixture!(softmax5);
    let mut output = [0i8; 5];
    let mut scratch = [0u8; 256];
    backend
        .softmax(&softmax5::INPUT_DATA, &params, &mut output, &mut scratch)
        .expect("softmax kernel returned Err");
    assert_bit_exact(&output, &softmax5::EXPECTED_OUTPUT, "softmax_golden_5elem");
}
