// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Golden-vector tests for the reductions scalar reference kernel.
//!
//! Loads TFLM-generated const-array fixtures via `include!()` and asserts
//! bit-exact output match for all five reduction ops.
//!
//! Test naming convention: `reductions_golden_<op>` so that
//! `cargo test -p hematite-ref -- reductions_golden` matches all tests.

// ── Fixture includes ───────────────────────────────────────────────────────

mod mean_fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/mean.rs"
    ));
}

mod sum_fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/sum.rs"
    ));
}

mod argmax_fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/argmax.rs"
    ));
}

mod argmin_fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/argmin.rs"
    ));
}

mod l2_norm_fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../hematite-tests/goldens/l2_norm.rs"
    ));
}

use hematite_core::op_params::ReduceParams;
use hematite_ref::reductions;

/// Construct a `ReduceParams` from a fixture module's public consts.
/// The fixture carries quant params even for argmax/argmin (which ignore them).
macro_rules! params_from_fixture {
    ($m:ident) => {{
        // Build axis array from AXIS_0..AXIS_COUNT
        let axis_count = $m::AXIS_COUNT as usize;
        let mut axis_arr: [i16; 4] = [0i16; 4];
        // The fixture uses AXIS_0, AXIS_1, ... but our fixtures have only one axis
        // In the general case, AXIS_0 is the only one set. Read it by name.
        if axis_count > 0 {
            axis_arr[0] = $m::AXIS_0 as i16;
        }
        ReduceParams {
            keep_dims: false,
            axis: axis_arr,
            axis_count: $m::AXIS_COUNT as i8,
            input_shape: $m::INPUT_SHAPE,
            output_shape: $m::OUTPUT_SHAPE,
            output_type: 0, // i8 for golden fixtures
            input_offset: 0,
            output_offset: 0,
            output_multiplier: 1i32 << 30,
            output_shift: 0,
            quantized_activation_min: -128,
            quantized_activation_max: 127,
        }
    }};
}

/// Param source that pulls quant fields from the fixture (for mean, sum, l2_norm).
macro_rules! quant_params_from_fixture {
    ($m:ident) => {{
        let axis_count = $m::AXIS_COUNT as usize;
        let mut axis_arr: [i16; 4] = [0i16; 4];
        if axis_count > 0 {
            axis_arr[0] = $m::AXIS_0 as i16;
        }
        ReduceParams {
            keep_dims: false,
            axis: axis_arr,
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

/// Assert that `actual` matches `expected` element-for-element.
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
fn reductions_golden_mean() {
    let params = quant_params_from_fixture!(mean_fixture);
    let mut output = [0i8; 6];
    reductions::mean(
        &mean_fixture::INPUT_DATA,
        &params,
        &mut output,
    )
    .expect("mean kernel returned Err");
    assert_bit_exact(&output, &mean_fixture::EXPECTED_OUTPUT, "reductions_golden_mean");
}

#[test]
fn reductions_golden_sum() {
    let params = quant_params_from_fixture!(sum_fixture);
    let mut output = [0i8; 2];
    reductions::sum(
        &sum_fixture::INPUT_DATA,
        &params,
        &mut output,
    )
    .expect("sum kernel returned Err");
    assert_bit_exact(&output, &sum_fixture::EXPECTED_OUTPUT, "reductions_golden_sum");
}

#[test]
fn reductions_golden_argmax() {
    let params = params_from_fixture!(argmax_fixture);
    let mut output = [0i8; 3];
    reductions::arg_max(
        &argmax_fixture::INPUT_DATA,
        &params,
        &mut output,
    )
    .expect("arg_max kernel returned Err");
    assert_bit_exact(&output, &argmax_fixture::EXPECTED_OUTPUT, "reductions_golden_argmax");
}

#[test]
fn reductions_golden_argmin() {
    let params = params_from_fixture!(argmin_fixture);
    let mut output = [0i8; 3];
    reductions::arg_min(
        &argmin_fixture::INPUT_DATA,
        &params,
        &mut output,
    )
    .expect("arg_min kernel returned Err");
    assert_bit_exact(&output, &argmin_fixture::EXPECTED_OUTPUT, "reductions_golden_argmin");
}

#[test]
fn reductions_golden_l2_norm() {
    let params = quant_params_from_fixture!(l2_norm_fixture);
    let mut output = [0i8; 4];
    reductions::l2_norm(
        &l2_norm_fixture::INPUT_DATA,
        &params,
        &mut output,
    )
    .expect("l2_norm kernel returned Err");
    assert_bit_exact(&output, &l2_norm_fixture::EXPECTED_OUTPUT, "reductions_golden_l2_norm");
}
