// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Per-op golden test: LSTM — mirrors `hematite-ref/tests/recurrent_golden.rs`.
//!
//! NOTE (T5.1 trait-gap): the `KernelBackend::unidirectional_sequence_lstm`
//! signature cannot carry the fixture-specific fixed-point quant constants
//! (gate / cell-tanh / output multiplier+shift pairs) that the scalar kernel
//! requires — they are not fields of `LstmParams`. `RefBackend` therefore
//! returns `Unsupported` for this op, and this test exercises the scalar
//! kernel directly (the A4 bit-exact contract still holds).

mod lstm_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/lstm.rs"));
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

// Gate/cell quant params computed by quantize_multiplier in the generator:
//   gate_mult = quantize_multiplier(2^20 = 1048576) → (2^30, 21)
//   cell_tanh = quantize_multiplier(2^18 = 262144)  → (2^30, 19)
//   out_mult   = quantize_multiplier(256/2048=0.125) → (2^30, -2)
const GATE_MULT: i32 = 1i32 << 30;
const GATE_SHIFT: i32 = 21;
const CELL_TANH_MULT: i32 = 1i32 << 30;
const CELL_TANH_SHIFT: i32 = 19;
const LSTM_OUT_MULT: i32 = 1i32 << 30;
const LSTM_OUT_SHIFT: i32 = -2;

#[test]
fn lstm_golden() {
    let num_units = lstm_fixture::NUM_UNITS as usize;
    let input_dim = lstm_fixture::INPUT_DIM as usize;
    let timesteps = lstm_fixture::NUM_TIMESTEPS as usize;

    let mut hidden = lstm_fixture::INIT_HIDDEN_STATE.to_vec();
    let mut cell = lstm_fixture::INIT_CELL_STATE.to_vec();
    let mut output = Vec::with_capacity(num_units * timesteps);

    for t in 0..timesteps {
        let inp = &lstm_fixture::INPUT_DATA[t * input_dim..(t + 1) * input_dim];
        recurrent::lstm(
            inp,
            &mut hidden,
            &mut cell,
            &lstm_fixture::WEIGHTS_DATA,
            &lstm_fixture::RECURRENT_WEIGHTS_DATA,
            &lstm_fixture::BIAS_DATA,
            GATE_MULT,
            GATE_SHIFT,
            CELL_TANH_MULT,
            CELL_TANH_SHIFT,
            LSTM_OUT_MULT,
            LSTM_OUT_SHIFT,
            lstm_fixture::OUTPUT_OFFSET,
            lstm_fixture::OUTPUT_ACTIVATION_MIN,
            lstm_fixture::OUTPUT_ACTIVATION_MAX,
            input_dim,
            num_units,
            1, // per-timestep call (the kernel emits final hidden only)
        )
        .expect("lstm kernel returned Err");
        output.extend_from_slice(&hidden);
    }

    assert_bit_exact(&output, &lstm_fixture::EXPECTED_OUTPUT, "lstm_golden");
}
