// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Golden-vector tests for data-movement scalar reference kernels.
//!
//! Loads TFLM-generated const-array fixtures via `include!()` and asserts
//! bit-exact output match for concat, split (v0, v1), pad, and slice.
//!
//! Test naming convention: `<op>_golden[_<variant>]` so that
//! `cargo test -p hematite-ref -- data_movement_golden` matches all tests.

// ── Fixture includes ───────────────────────────────────────────────────────

mod concat {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/concat.rs"
    ));
}

mod split_v0 {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/split_v0.rs"
    ));
}

mod split_v1 {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/split_v1.rs"
    ));
}

mod pad {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/pad.rs"
    ));
}

mod slice {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/slice.rs"
    ));
}

use hematite_core::op_params::{ConcatParams, FusedActivation, PadParams, SliceParams, SplitParams};
use hematite_ref::data_movement::{concat_op, pad_op, slice_op, split_op};

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
fn concat_golden() {
    let params = ConcatParams {
        axis: concat::AXIS,
        activation: FusedActivation::None,
        input_shape_a: concat::INPUT_SHAPE,
        input_shape_b: concat::FILTER_SHAPE,
        output_shape: concat::OUTPUT_SHAPE,
    };
    let mut output = [0i8; 4];
    concat_op(
        &concat::INPUT_DATA,
        &concat::WEIGHTS_DATA,
        &params,
        &mut output,
        &mut [],
    )
    .expect("concat kernel returned Err");
    assert_bit_exact(&output, &concat::EXPECTED_OUTPUT, "concat_golden");
}

#[test]
fn split_golden_v0() {
    let params = SplitParams {
        num_splits: split_v0::NUM_SPLITS,
        axis: split_v0::AXIS,
        input_shape: split_v0::INPUT_SHAPE,
        output_shape_a: split_v0::OUTPUT_SHAPE,
        output_shape_b: split_v1::OUTPUT_SHAPE,
    };
    let split_index: i32 = split_v0::SPLIT_INDEX;
    let mut output = [0i8; 2];
    split_op(
        &split_v0::INPUT_DATA,
        split_index,
        &params,
        &mut output,
        &mut [],
    )
    .expect("split kernel returned Err");
    assert_bit_exact(&output, &split_v0::EXPECTED_OUTPUT, "split_golden_v0");
}

#[test]
fn split_golden_v1() {
    let params = SplitParams {
        num_splits: split_v1::NUM_SPLITS,
        axis: split_v1::AXIS,
        input_shape: split_v1::INPUT_SHAPE,
        output_shape_a: split_v0::OUTPUT_SHAPE,
        output_shape_b: split_v1::OUTPUT_SHAPE,
    };
    let split_index: i32 = split_v1::SPLIT_INDEX;
    let mut output = [0i8; 2];
    split_op(
        &split_v1::INPUT_DATA,
        split_index,
        &params,
        &mut output,
        &mut [],
    )
    .expect("split kernel returned Err");
    assert_bit_exact(&output, &split_v1::EXPECTED_OUTPUT, "split_golden_v1");
}

#[test]
fn pad_golden() {
    let params = PadParams {
        input_shape: pad::INPUT_SHAPE,
        output_shape: pad::OUTPUT_SHAPE,
        left_padding: [0, pad::PAD_TOP, pad::PAD_LEFT, 0],
        left_padding_count: 4,
        right_padding: [0, pad::PAD_BOTTOM, pad::PAD_RIGHT, 0],
        right_padding_count: 4,
    };
    let mut output = [0i8; 16];
    pad_op(&pad::INPUT_DATA, &params, &mut output, &mut [])
        .expect("pad kernel returned Err");
    assert_bit_exact(&output, &pad::EXPECTED_OUTPUT, "pad_golden");
}

#[test]
fn slice_golden() {
    let params = SliceParams {
        input_shape: slice::INPUT_SHAPE,
        begin: [
            slice::BEGIN_0,
            slice::BEGIN_1,
            slice::BEGIN_2,
            slice::BEGIN_3,
        ],
        size: [
            slice::SIZE_0,
            slice::SIZE_1,
            slice::SIZE_2,
            slice::SIZE_3,
        ],
    };
    let mut output = [0i8; 4];
    slice_op(&slice::INPUT_DATA, &params, &mut output, &mut [])
        .expect("slice kernel returned Err");
    assert_bit_exact(&output, &slice::EXPECTED_OUTPUT, "slice_golden");
}
