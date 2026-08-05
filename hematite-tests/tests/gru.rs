// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Per-op golden test: GRU — mirrors `hematite-ref/tests/recurrent_golden.rs`.
//!
//! NOTE (T5.1 trait-gap): the `KernelBackend::gru` signature cannot carry
//! the output quant constants the scalar GRU kernel requires (they are not
//! fields of `GruParams`). `RefBackend` returns `Unsupported` for this op;
//! the bit-exact contract is exercised directly on the scalar kernel.

mod gru_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/gru.rs"));
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

// gate_mult = quantize_multiplier(2^20 = 1048576) → (2^30, 21)
// out_mult   = quantize_multiplier(128/2048=0.0625) → (2^30, -3)
const GRU_GATE_MULT: i32 = 1i32 << 30;
const GRU_GATE_SHIFT: i32 = 21;
const GRU_OUT_MULT: i32 = 1i32 << 30;
const GRU_OUT_SHIFT: i32 = -3;

#[test]
fn gru_golden() {
    let num_units = gru_fixture::NUM_UNITS as usize;
    let input_size = gru_fixture::INPUT_SIZE as usize;
    let timesteps = gru_fixture::NUM_TIMESTEPS as usize;

    let mut hidden = gru_fixture::INIT_HIDDEN_STATE.to_vec();
    let mut output = Vec::with_capacity(num_units * timesteps);

    for t in 0..timesteps {
        let inp = &gru_fixture::INPUT_DATA[t * input_size..(t + 1) * input_size];
        recurrent::gru(
            inp,
            &mut hidden,
            &gru_fixture::WEIGHTS_DATA,
            &gru_fixture::RECURRENT_WEIGHTS_DATA,
            &gru_fixture::BIAS_DATA,
            GRU_GATE_MULT,
            GRU_GATE_SHIFT,
            GRU_OUT_MULT,
            GRU_OUT_SHIFT,
            gru_fixture::OUTPUT_OFFSET,
            gru_fixture::OUTPUT_ACTIVATION_MIN,
            gru_fixture::OUTPUT_ACTIVATION_MAX,
            input_size,
            num_units,
            1, // per-timestep call
        )
        .expect("gru kernel returned Err");

        // hidden is in Q0.11 i16 — requantize to i8 for the golden compare.
        let mut out_frame = vec![0i8; num_units];
        recurrent::gru_output_to_i8(
            &hidden,
            &mut out_frame,
            num_units,
            GRU_OUT_MULT,
            GRU_OUT_SHIFT,
            gru_fixture::OUTPUT_OFFSET,
            gru_fixture::OUTPUT_ACTIVATION_MIN,
            gru_fixture::OUTPUT_ACTIVATION_MAX,
        );
        output.extend_from_slice(&out_frame);
    }

    assert_bit_exact(&output, &gru_fixture::EXPECTED_OUTPUT, "gru_golden");
}
