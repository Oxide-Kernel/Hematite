// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Per-op golden test: SVDF — mirrors `hematite-ref/tests/recurrent_golden.rs`.
//!
//! NOTE (T5.1 trait-gap): the `KernelBackend::svdf` signature cannot carry
//! the output quant constants the scalar `svdf_step` kernel requires (they
//! are not fields of `SvdfParams`). `RefBackend` returns `Unsupported` for
//! this op; the bit-exact contract is exercised directly on the scalar
//! kernel.

mod svdf_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/svdf.rs"));
}

use hematite_ref::recurrent;

fn assert_bit_exact(actual: &[i8], expected: &[i8], name: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{name}: length mismatch {} vs {}",
        actual.len(),
        expected.len(),
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(a, e, "{name}: mismatch at index {i}: kernel={a}, golden={e}");
    }
}

// out_mult = quantize_multiplier(1.0) → (2^30, 1)
const SVDF_OUT_MULT: i32 = 1i32 << 30;
const SVDF_OUT_SHIFT: i32 = 1;

#[test]
fn svdf_golden() {
    let num_filters = svdf_fixture::NUM_FILTERS as usize;
    let rank = svdf_fixture::RANK as usize;
    let input_size = svdf_fixture::INPUT_SIZE as usize;
    let timesteps = 2usize;

    let mut state = svdf_fixture::INIT_STATE.to_vec();
    let mut output = Vec::with_capacity(num_filters * timesteps);

    for t in 0..timesteps {
        let inp = &svdf_fixture::INPUT_DATA[t * input_size..(t + 1) * input_size];
        let mut out_frame = vec![0i8; num_filters];
        recurrent::svdf_step(
            &mut state,
            &svdf_fixture::FEATURE_WEIGHTS_DATA,
            &svdf_fixture::TIME_WEIGHTS_DATA,
            &svdf_fixture::BIAS_DATA,
            inp,
            &mut out_frame,
            num_filters,
            rank,
            input_size,
            SVDF_OUT_MULT,
            SVDF_OUT_SHIFT,
            svdf_fixture::OUTPUT_OFFSET,
            svdf_fixture::OUTPUT_ACTIVATION_MIN,
            svdf_fixture::OUTPUT_ACTIVATION_MAX,
        )
        .expect("svdf_step returned Err");
        output.extend_from_slice(&out_frame);
    }

    assert_bit_exact(&output, &svdf_fixture::EXPECTED_OUTPUT, "svdf_golden");
}
