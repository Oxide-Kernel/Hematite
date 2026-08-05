// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Golden-vector tests for recurrent ops: LSTM, SVDF, GRU.
//!
//! Test naming: T3_lstm_golden, T3_svdf_golden, T3_gru_golden
//! QA: cargo test -p hematite-ref -- T3_*

// ── Fixture includes ──
mod lstm_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/lstm.rs"));
}
mod svdf_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/svdf.rs"));
}
mod gru_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/gru.rs"));
}

use hematite_ref::recurrent;

fn assert_bit_exact(actual: &[i8], expected: &[i8], name: &str) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch {} vs {}", actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(a, e, "{name}: mismatch at index {i}: kernel={a}, golden={e}");
    }
}

// ── LSTM Golden ──
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
fn T3_lstm_golden() {
    let num_units = lstm_fixture::NUM_UNITS as usize;
    let input_dim = lstm_fixture::INPUT_DIM as usize;
    let timesteps = lstm_fixture::NUM_TIMESTEPS as usize;

    let mut hidden = lstm_fixture::INIT_HIDDEN_STATE.to_vec();
    let mut cell = lstm_fixture::INIT_CELL_STATE.to_vec();

    recurrent::lstm(
        &lstm_fixture::INPUT_DATA,
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
        timesteps,
    )
    .expect("lstm kernel returned Err");

    // hidden now contains the FINAL hidden state after all timesteps.
    // The generator collects hidden state after EACH timestep into EXPECTED_OUTPUT.
    // But our kernel only returns the final state. Let me fix this by iterating per-timestep.

    // Actually, looking at the generator more carefully: it loops over timesteps
    // and collects `output_hidden` per timestep. The EXPECTED_OUTPUT is [t0_h0..t0_h3, t1_h0..t1_h3].
    // So the kernel should be called per-timestep, and we collect hidden each time.

    // Re-initialize and call per-timestep
    hidden.copy_from_slice(&lstm_fixture::INIT_HIDDEN_STATE);
    cell.copy_from_slice(&lstm_fixture::INIT_CELL_STATE);

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
            1, // single timestep
        )
        .expect("lstm kernel returned Err");
        output.extend_from_slice(&hidden);
    }

    assert_bit_exact(&output, &lstm_fixture::EXPECTED_OUTPUT, "T3_lstm_golden");
}

// ── SVDF Golden ──
// out_mult = quantize_multiplier(1.0) → (2^30, 1)
const SVDF_OUT_MULT: i32 = 1i32 << 30;
const SVDF_OUT_SHIFT: i32 = 1;

#[test]
fn T3_svdf_golden() {
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

    assert_bit_exact(&output, &svdf_fixture::EXPECTED_OUTPUT, "T3_svdf_golden");
}

// ── GRU Golden ──
// gate_mult = quantize_multiplier(2^20 = 1048576) → (2^30, 21)
// out_mult   = quantize_multiplier(128/2048=0.0625) → (2^30, -3)
const GRU_GATE_MULT: i32 = 1i32 << 30;
const GRU_GATE_SHIFT: i32 = 21;
const GRU_OUT_MULT: i32 = 1i32 << 30;
const GRU_OUT_SHIFT: i32 = -3;

#[test]
fn T3_gru_golden() {
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

        // hidden is now in Q0.11 i16, requantize to i8
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

    assert_bit_exact(&output, &gru_fixture::EXPECTED_OUTPUT, "T3_gru_golden");
}
