// SPDX-License-Identifier: Apache-2.0
//! Recurrent golden fixtures — LSTM, SVDF, GRU.
//!
//! Arithmetic defined HERE (oracle); mirrored in hematite-ref/src/recurrent.rs.

use crate::fixture::FixtureWriter;
use crate::tflm_math;

// ═══════════════════════════════════════════════════════════════════════════════
// LSTM — unidirectional_sequence_lstm_s8 (i8 activations, i16 cell state)
// ═══════════════════════════════════════════════════════════════════════════════

pub fn generate_lstm(w: &mut FixtureWriter) {
    let input_dim: usize = 3;
    let num_units: usize = 4;
    let timesteps: usize = 2;

    let input_shape = [1i32, timesteps as i32, 1, input_dim as i32];
    let output_shape = [1i32, timesteps as i32, 1, num_units as i32];

    // Input: 2 timesteps × 3 features (deliberately mixed signs)
    let input_data: Vec<i8> = vec![
        -10, 5, 20,
        15, -8, -3,
    ];

    // Gate order: 0=input, 1=forget, 2=cell, 3=output

    // ── Input-to-hidden weights: 4 gates × num_units × input_dim ──
    let w_i: [i8; 12] = [2, -1, 3, -3, 1, -2, 1, 4, -1, -2, -3, 1];
    let w_f: [i8; 12] = [1, 2, 1, -1, 3, -1, 2, -2, 4, 1, 1, 1];
    let w_g: [i8; 12] = [3, -2, 1, 1, 1, -3, -1, -1, 2, 2, 3, -2];
    let w_o: [i8; 12] = [1, -1, 2, -2, 2, 3, 4, 1, -2, -3, -1, 1];

    let all_w: [&[i8]; 4] = [&w_i, &w_f, &w_g, &w_o];

    // ── Recurrent weights: 4 gates × num_units × num_units ──
    let r_i: [i8; 16] = [1, 2, -1, 3, -2, 1, 3, -1, 4, -2, 1, 2, -1, 3, -2, 4];
    let r_f: [i8; 16] = [2, -1, 1, -2, 3, 1, -1, 2, -2, 4, 1, -3, 1, 2, 3, -1];
    let r_g: [i8; 16] = [-3, 1, 2, -1, 1, -2, 3, 2, -1, 3, -1, 1, 2, -2, 1, 3];
    let r_o: [i8; 16] = [1, -2, 3, 1, 2, 1, -1, -3, -2, 2, 1, 4, -1, -1, 2, -2];

    let all_r: [&[i8]; 4] = [&r_i, &r_f, &r_g, &r_o];

    // ── Biases ──
    let b_i: [i32; 4] = [0, 10, -5, 5];
    let b_f: [i32; 4] = [20, 15, 10, 5];
    let b_g: [i32; 4] = [-10, 5, 0, -5];
    let b_o: [i32; 4] = [10, -5, 15, 0];

    let all_b: [&[i32]; 4] = [&b_i, &b_f, &b_g, &b_o];

    // ── Flatten all weights/biases for fixture emission ──
    let mut flat_w: Vec<i8> = Vec::new();
    let mut flat_r: Vec<i8> = Vec::new();
    let mut flat_b: Vec<i32> = Vec::new();
    for g in 0..4 {
        flat_w.extend_from_slice(all_w[g]);
        flat_r.extend_from_slice(all_r[g]);
        flat_b.extend_from_slice(all_b[g]);
    }

    // ── Quantization ──
    // Gate accumulator → Q5.26: small i32 acc (~±few hundred) needs massive
    // amplification. scale=2^20 maps acc=35 → ~0.55 real → sigmoid~0.63.
    let (gate_mult, gate_shift) = tflm_math::quantize_multiplier(1048576.0_f64); // 2^20

    // Cell state scale for tanh: i16 → Q5.26.
    // scale=2^18 maps cell=200 → ~0.78 → tanh~0.65.
    let (cell_tanh_mult, cell_tanh_shift) = tflm_math::quantize_multiplier(262144.0_f64); // 2^18

    // Hidden output scale: Q0.11 gate * tanh(cell) → i8
    // Output multiplier: maps Q0.11 value ~[0, 2048) to i8 [-128, 127]
    // scale ≈ 256/2048 = 0.125
    let (out_mult, out_shift) = tflm_math::quantize_multiplier(256.0_f64 / 2048.0_f64);
    let out_offset: i32 = 0;
    let act_min: i32 = -128;
    let act_max: i32 = 127;

    // ── Initial state ──
    let mut hidden_state: Vec<i8> = vec![0i8; num_units];
    let mut cell_state: Vec<i16> = vec![0i16; num_units];
    let init_hidden_state: Vec<i8> = hidden_state.clone();
    let init_cell_state: Vec<i16> = cell_state.clone();

    let mut output_hidden: Vec<i8> = Vec::new();

    // ── Timestep loop ──
    for t in 0..timesteps {
        let inp = &input_data[t * input_dim..(t + 1) * input_dim];

        // Gate results in Q0.11 (i16)
        let mut gate_i = [0i16; 4];
        let mut gate_f = [0i16; 4];
        let mut gate_g = [0i16; 4];
        let mut gate_o = [0i16; 4];
        let gates: [&mut [i16]; 4] = [&mut gate_i, &mut gate_f, &mut gate_g, &mut gate_o];

        for g in 0..4usize {
            let wg = all_w[g];
            let rg = all_r[g];
            let bg = all_b[g];
            let use_tanh = g == 2; // cell gate uses tanh

            for u in 0..num_units {
                let mut acc: i32 = bg[u];

                // Input-to-hidden
                for j in 0..input_dim {
                    acc += i32::from(wg[u * input_dim + j]) * i32::from(inp[j]);
                }
                // Recurrent
                for j in 0..num_units {
                    acc += i32::from(rg[u * num_units + j]) * i32::from(hidden_state[j]);
                }

                let acc_q526 = tflm_math::multiply_by_quantized_multiplier(acc, gate_mult, gate_shift);
                gates[g][u] = if use_tanh {
                    tflm_math::tanh_i16_q011(acc_q526)
                } else {
                    tflm_math::logistic_i16_q011(acc_q526)
                };
            }
        }

        // Cell state update: c_new = f * c_old + i * g (all Q0.11)
        // c_new_i32 = (f_q011 * c_old_i16 / 2048 + i_q011 * g_q011 / 2048) → i16
        for u in 0..num_units {
            let f32 = i32::from(gate_f[u]);
            let i32 = i32::from(gate_i[u]);
            let g32 = i32::from(gate_g[u]);
            let cold = i32::from(cell_state[u]);
            // f*c: Q0.11 * i16 → i32 with rounding
            let fc = tflm_math::rounding_divide_by_pot(f32 * cold, 11);
            // i*g: Q0.11 * Q0.11 → Q0.22 → downshift
            let ig = tflm_math::rounding_divide_by_pot(i32 * g32, 11);
            cell_state[u] = (fc + ig) as i16;

            // Hidden state: h_new = o * tanh(c_new)
            // Scale cell to Q5.26 for tanh, then combine
            let cell_q526 = tflm_math::multiply_by_quantized_multiplier(
                i32::from(cell_state[u]),
                cell_tanh_mult,
                cell_tanh_shift,
            );
            let tanh_c_q011 = tflm_math::tanh_i16_q011(cell_q526);
            let o32 = i32::from(gate_o[u]);
            let h_raw = tflm_math::rounding_divide_by_pot(o32 * i32::from(tanh_c_q011), 11);
            // Requantize to i8
            let h_scaled = tflm_math::multiply_by_quantized_multiplier(h_raw, out_mult, out_shift);
            let h_val = (h_scaled + out_offset).max(act_min).min(act_max);
            hidden_state[u] = h_val as i8;
        }

        output_hidden.extend_from_slice(&hidden_state);
    }

    // ── Emit fixture ──
    let mut buf = String::with_capacity(16384);
    buf.push_str(crate::fixture::SPDX_HEADER);
    buf.push('\n');
    buf.push_str(crate::fixture::PROVENANCE_NOTE);
    buf.push('\n');

    use std::fmt::Write as _;
    let _ = writeln!(buf, "pub const GOLDEN_TFLM_VERSION: &str = \"{}\";", crate::fixture::TFLM_VERSION);
    writeln!(buf, "pub const GOLDEN_PROVENANCE: &str = \"tool-internal-reference-reimplementation; NOT captured from executed TFLM; see tools/generate_goldens/README.md\";").unwrap();
    let _ = writeln!(buf);
    let _ = writeln!(buf, "pub const INPUT_SHAPE: [i32; 4] = {:?};", input_shape);
    let _ = writeln!(buf, "pub const FILTER_SHAPE: [i32; 4] = {:?};", [0i32; 4]);
    let _ = writeln!(buf, "pub const OUTPUT_SHAPE: [i32; 4] = {:?};", output_shape);
    let _ = writeln!(buf);
    let _ = writeln!(buf, "pub const NUM_UNITS: i32 = {};", num_units as i32);
    let _ = writeln!(buf, "pub const INPUT_DIM: i32 = {};", input_dim as i32);
    let _ = writeln!(buf, "pub const NUM_TIMESTEPS: i32 = {};", timesteps as i32);
    let _ = writeln!(buf);
    let _ = writeln!(buf, "pub const INPUT_OFFSET: i32 = 0;");
    let _ = writeln!(buf, "pub const OUTPUT_OFFSET: i32 = {};", out_offset);
    let _ = writeln!(buf, "pub const OUTPUT_ACTIVATION_MIN: i32 = {};", act_min);
    let _ = writeln!(buf, "pub const OUTPUT_ACTIVATION_MAX: i32 = {};", act_max);
    let _ = writeln!(buf);
    cr::emit_const_array(&mut buf, "INPUT_DATA", "i8", &input_data);
    cr::emit_const_array(&mut buf, "WEIGHTS_DATA", "i8", &flat_w);
    cr::emit_const_array(&mut buf, "RECURRENT_WEIGHTS_DATA", "i8", &flat_r);
    cr::emit_const_array(&mut buf, "BIAS_DATA", "i32", &flat_b);
    cr::emit_const_array(&mut buf, "OUTPUT_MULTIPLIER", "i32", &[out_mult]);
    cr::emit_const_array(&mut buf, "OUTPUT_SHIFT", "i32", &[out_shift]);
    cr::emit_const_array(&mut buf, "EXPECTED_OUTPUT", "i8", &output_hidden);
    cr::emit_const_array(&mut buf, "INIT_HIDDEN_STATE", "i8", &init_hidden_state);
    cr::emit_const_array(&mut buf, "INIT_CELL_STATE", "i16", &init_cell_state);

    let path = w.output_dir().join("lstm.rs");
    std::fs::write(&path, buf).expect("write fixture");
    println!("  Wrote {}", path.display());
}

// ═══════════════════════════════════════════════════════════════════════════════
// SVDF — int8 SVDF per TFLM reference arithmetic
// ═══════════════════════════════════════════════════════════════════════════════

pub fn generate_svdf(w: &mut FixtureWriter) {
    let num_filters: usize = 3;
    let rank: usize = 2;
    let input_size: usize = 4;
    let batch_size: i32 = 1;

    let input_shape = [batch_size, 1, 1, input_size as i32];
    let output_shape = [batch_size, 1, 1, num_filters as i32];

    // Input: 2 timesteps × 4 features
    let input_data: Vec<i8> = vec![
        -5, 10, -15, 20,   // t=0
        8, -12, 6, -3,     // t=1
    ];

    let timesteps: usize = 2;

    // ── Feature weights: [num_filters, input_size] ──
    let w_feat: Vec<i8> = vec![
        20, -10, 30, -20,   // filter 0
        -30, 20, 10, -10,   // filter 1
        10, -40, 20, 30,    // filter 2
    ];

    // ── Time weights: [num_filters, rank] ──
    let w_time: Vec<i8> = vec![
        40, -20,   // filter 0
        10, 30,    // filter 1
        -40, 10,   // filter 2
    ];

    // ── Bias: [num_filters] ──
    let bias: Vec<i32> = vec![50, -100, 150];

    // ── Quantization ──
    let input_offset: i32 = 0;
    let output_offset: i32 = 0;
    let act_min: i32 = -128;
    let act_max: i32 = 127;

    // Accumulator scale: i32 acc → requantize to i8 (identity for small values)
    let (out_mult, out_shift) = tflm_math::quantize_multiplier(1.0_f64);

    // ── State: activation buffer, holds last `rank` activations ──
    let mut state: Vec<i8> = vec![0i8; num_filters * rank];
    let init_state: Vec<i8> = state.clone();

    let mut output: Vec<i8> = Vec::new();

    for t in 0..timesteps {
        let inp = &input_data[t * input_size..(t + 1) * input_size];

        // Step A: Project input to feature space →
        // scratch: [num_filters] i32 accumulators
        let mut feat_acc: Vec<i32> = vec![0i32; num_filters];
        for f in 0..num_filters {
            let mut acc: i32 = 0;
            for j in 0..input_size {
                acc += i32::from(w_feat[f * input_size + j]) * i32::from(inp[j]);
            }
            feat_acc[f] = acc;
        }

        // Step B: Shift state and insert new feature activations
        // State layout: [rank × num_filters] = [num_filters * rank]
        // Shifting: each filter's rank entries shift left by 1 (oldest drops)
        for f in 0..num_filters {
            for r in (1..rank).rev() {
                state[f * rank + r] = state[f * rank + (r - 1)];
            }
            // Insert new feature activation after requantize
            let scaled = tflm_math::multiply_by_quantized_multiplier(feat_acc[f], out_mult, out_shift);
            let val = (scaled + output_offset).max(act_min).min(act_max);
            state[f * rank] = val as i8;
        }

        // Step C: Time convolution — dot state with time weights
        let mut out_frame: Vec<i8> = vec![0i8; num_filters];
        for f in 0..num_filters {
            let mut acc: i32 = bias[f];
            for r in 0..rank {
                acc += i32::from(w_time[f * rank + r]) * i32::from(state[f * rank + r]);
            }
            let scaled = tflm_math::multiply_by_quantized_multiplier(acc, out_mult, out_shift);
            let val = (scaled + output_offset).max(act_min).min(act_max);
            out_frame[f] = val as i8;
        }

        output.extend_from_slice(&out_frame);
    }

    // ── Emit fixture ──
    let mut buf = String::with_capacity(16384);
    buf.push_str(crate::fixture::SPDX_HEADER);
    buf.push('\n');
    buf.push_str(crate::fixture::PROVENANCE_NOTE);
    buf.push('\n');

    use std::fmt::Write as _;
    let _ = writeln!(buf, "pub const GOLDEN_TFLM_VERSION: &str = \"{}\";", crate::fixture::TFLM_VERSION);
    writeln!(buf, "pub const GOLDEN_PROVENANCE: &str = \"tool-internal-reference-reimplementation; NOT captured from executed TFLM; see tools/generate_goldens/README.md\";").unwrap();
    let _ = writeln!(buf);
    let _ = writeln!(buf, "pub const INPUT_SHAPE: [i32; 4] = {:?};", input_shape);
    let _ = writeln!(buf, "pub const FILTER_SHAPE: [i32; 4] = {:?};", [0i32; 4]);
    let _ = writeln!(buf, "pub const OUTPUT_SHAPE: [i32; 4] = {:?};", output_shape);
    let _ = writeln!(buf);
    let _ = writeln!(buf, "pub const NUM_FILTERS: i32 = {};", num_filters as i32);
    let _ = writeln!(buf, "pub const RANK: i32 = {};", rank as i32);
    let _ = writeln!(buf, "pub const INPUT_SIZE: i32 = {};", input_size as i32);
    let _ = writeln!(buf);
    let _ = writeln!(buf, "pub const INPUT_OFFSET: i32 = {};", input_offset);
    let _ = writeln!(buf, "pub const OUTPUT_OFFSET: i32 = {};", output_offset);
    let _ = writeln!(buf, "pub const OUTPUT_ACTIVATION_MIN: i32 = {};", act_min);
    let _ = writeln!(buf, "pub const OUTPUT_ACTIVATION_MAX: i32 = {};", act_max);
    let _ = writeln!(buf);
    cr::emit_const_array(&mut buf, "INPUT_DATA", "i8", &input_data);
    cr::emit_const_array(&mut buf, "FEATURE_WEIGHTS_DATA", "i8", &w_feat);
    cr::emit_const_array(&mut buf, "TIME_WEIGHTS_DATA", "i8", &w_time);
    cr::emit_const_array(&mut buf, "BIAS_DATA", "i32", &bias);
    cr::emit_const_array(&mut buf, "OUTPUT_MULTIPLIER", "i32", &[out_mult]);
    cr::emit_const_array(&mut buf, "OUTPUT_SHIFT", "i32", &[out_shift]);
    cr::emit_const_array(&mut buf, "EXPECTED_OUTPUT", "i8", &output);
    cr::emit_const_array(&mut buf, "INIT_STATE", "i8", &init_state);

    let path = w.output_dir().join("svdf.rs");
    std::fs::write(&path, buf).expect("write fixture");
    println!("  Wrote {}", path.display());
}

// ═══════════════════════════════════════════════════════════════════════════════
// GRU — hand-rolled gate math (NO TFLM kernel; gemmlowp logistic/tanh)
// ═══════════════════════════════════════════════════════════════════════════════

pub fn generate_gru(w: &mut FixtureWriter) {
    let input_size: usize = 2;
    let num_units: usize = 2;
    let timesteps: usize = 2;

    let input_shape = [1i32, timesteps as i32, 1, input_size as i32];
    let output_shape = [1i32, timesteps as i32, 1, num_units as i32];

    // Input: 2 timesteps × 2 features
    let input_data: Vec<i8> = vec![
        -20, 10,   // t=0
        15, -5,    // t=1
    ];

    // ── Gate order: 0=reset, 1=update, 2=candidate ──

    // Input-to-hidden weights: 3 gates × num_units × input_size
    let w_r: [i8; 4] = [2, -3, 1, 4];  // reset gate
    let w_z: [i8; 4] = [1, 2, -2, 1];  // update gate
    let w_h: [i8; 4] = [3, -1, -1, 2]; // candidate gate

    // Recurrent weights: 3 gates × num_units × num_units
    let u_r: [i8; 4] = [1, -2, 3, 1];  // reset
    let u_z: [i8; 4] = [-1, 2, 1, -3]; // update
    let u_h: [i8; 4] = [2, 1, -2, 3];  // candidate

    // Biases: 3 gates × num_units
    let b_r: [i32; 2] = [5, -5];
    let b_z: [i32; 2] = [10, -10];
    let b_h: [i32; 2] = [-5, 5];

    // ── Flatten for fixture ──
    let mut flat_w: Vec<i8> = Vec::new();
    let mut flat_u: Vec<i8> = Vec::new();
    let mut flat_b: Vec<i32> = Vec::new();
    flat_w.extend_from_slice(&w_r);
    flat_w.extend_from_slice(&w_z);
    flat_w.extend_from_slice(&w_h);
    flat_u.extend_from_slice(&u_r);
    flat_u.extend_from_slice(&u_z);
    flat_u.extend_from_slice(&u_h);
    flat_b.extend_from_slice(&b_r);
    flat_b.extend_from_slice(&b_z);
    flat_b.extend_from_slice(&b_h);

    // Gate accumulator → Q5.26: needs 2^20 amplification for i8 weights.
    let (gate_mult, gate_shift) = tflm_math::quantize_multiplier(1048576.0_f64); // 2^20

    // State output: Q0.11 i16 → i8 requantize
    let (out_mult, out_shift) = tflm_math::quantize_multiplier(128.0_f64 / 2048.0_f64);
    let out_offset: i32 = 0;
    let act_min: i32 = -128;
    let act_max: i32 = 127;

    // ── Initial state (Q0.11 i16) ──
    let mut hidden_state: Vec<i16> = vec![0i16; num_units];
    let init_hidden_state: Vec<i16> = hidden_state.clone();

    let mut output: Vec<i8> = Vec::new();

    // ── Timestep loop ──
    for t in 0..timesteps {
        let inp = &input_data[t * input_size..(t + 1) * input_size];

        let mut r_gate = [0i16; 2];
        let mut z_gate = [0i16; 2];
        let mut h_gate = [0i16; 2];

        // Reset gate
        for u in 0..num_units {
            let mut acc: i32 = b_r[u];
            for j in 0..input_size {
                acc += i32::from(w_r[u * input_size + j]) * i32::from(inp[j]);
            }
            for j in 0..num_units {
                acc += i32::from(u_r[u * num_units + j]) * i32::from(hidden_state[j]) / 2048; // h is Q0.11
            }
            let acc_q526 = tflm_math::multiply_by_quantized_multiplier(acc, gate_mult, gate_shift);
            r_gate[u] = tflm_math::logistic_i16_q011(acc_q526);
        }

        // Update gate
        for u in 0..num_units {
            let mut acc: i32 = b_z[u];
            for j in 0..input_size {
                acc += i32::from(w_z[u * input_size + j]) * i32::from(inp[j]);
            }
            for j in 0..num_units {
                acc += i32::from(u_z[u * num_units + j]) * i32::from(hidden_state[j]) / 2048;
            }
            let acc_q526 = tflm_math::multiply_by_quantized_multiplier(acc, gate_mult, gate_shift);
            z_gate[u] = tflm_math::logistic_i16_q011(acc_q526);
        }

        // Candidate gate (uses r_gate ⊙ hidden_state)
        for u in 0..num_units {
            let mut acc: i32 = b_h[u];
            for j in 0..input_size {
                acc += i32::from(w_h[u * input_size + j]) * i32::from(inp[j]);
            }
            // r ⊙ h: r in Q0.11, h in Q0.11 i16
            // r[j] * h[j] / 2048 gives h[j] in i16 but scaled by r
            for j in 0..num_units {
                let r_scaled_h = tflm_math::rounding_divide_by_pot(
                    i32::from(r_gate[j]) * i32::from(hidden_state[j]),
                    11,
                );
                acc += i32::from(u_h[u * num_units + j]) * r_scaled_h;
            }
            let acc_q526 = tflm_math::multiply_by_quantized_multiplier(acc, gate_mult, gate_shift);
            h_gate[u] = tflm_math::tanh_i16_q011(acc_q526);
        }

        // State update: new_h = (1 - z) * h_gate + z * old_h
        // In Q0.11: z ∈ [0, 2048), (1-z) = 2048 - z
        for u in 0..num_units {
            let z = i32::from(z_gate[u]);
            let one_minus_z = 2048 - z;
            let n = i32::from(h_gate[u]); // candidate in Q0.11 [-2048, 2047]
            let old = i32::from(hidden_state[u]);

            // (1-z)*n + z*old in Q0.22 → shift down by 11 → Q0.11
            let new_h_i32 = tflm_math::rounding_divide_by_pot(one_minus_z * n + z * old, 11);
            hidden_state[u] = new_h_i32 as i16;

            // Output: requantize Q0.11 → i8
            let h_scaled = tflm_math::multiply_by_quantized_multiplier(new_h_i32, out_mult, out_shift);
            let h_val = (h_scaled + out_offset).max(act_min).min(act_max);
            output.push(h_val as i8);
        }
    }

    // ── Emit fixture ──
    let mut buf = String::with_capacity(16384);
    buf.push_str(crate::fixture::SPDX_HEADER);
    buf.push('\n');
    buf.push_str(crate::fixture::PROVENANCE_NOTE);
    buf.push('\n');
    // GRU-specific provenance
    buf.push_str("// GRU PROVENANCE: TFLM has NO GRU kernel at the pinned SHA.\n");
    buf.push_str("// These goldens were generated with hand-rolled fixed-point gate math\n");
    buf.push_str("// (gemmlowp logistic/tanh via exp_on_negative_values) and verified via\n");
    buf.push_str("// manual bit-level self-check assertions (sigmoid(0)=0.5, tanh(0)=0,\n");
    buf.push_str("// endpoint saturation, monotonicity). No reference implementation exists.\n");
    buf.push_str("// See tools/generate_goldens/README.md for full provenance.\n\n");

    use std::fmt::Write as _;
    let _ = writeln!(buf, "pub const GOLDEN_TFLM_VERSION: &str = \"{}\";", crate::fixture::TFLM_VERSION);
    writeln!(buf, "pub const GOLDEN_PROVENANCE: &str = \"hand-rolled-gemmlowp-fixed-point; NO TFLM GRU kernel exists; see README\";").unwrap();
    let _ = writeln!(buf);
    let _ = writeln!(buf, "pub const INPUT_SHAPE: [i32; 4] = {:?};", input_shape);
    let _ = writeln!(buf, "pub const FILTER_SHAPE: [i32; 4] = {:?};", [0i32; 4]);
    let _ = writeln!(buf, "pub const OUTPUT_SHAPE: [i32; 4] = {:?};", output_shape);
    let _ = writeln!(buf);
    let _ = writeln!(buf, "pub const NUM_UNITS: i32 = {};", num_units as i32);
    let _ = writeln!(buf, "pub const INPUT_SIZE: i32 = {};", input_size as i32);
    let _ = writeln!(buf, "pub const NUM_TIMESTEPS: i32 = {};", timesteps as i32);
    let _ = writeln!(buf);
    let _ = writeln!(buf, "pub const INPUT_OFFSET: i32 = 0;");
    let _ = writeln!(buf, "pub const OUTPUT_OFFSET: i32 = {};", out_offset);
    let _ = writeln!(buf, "pub const OUTPUT_ACTIVATION_MIN: i32 = {};", act_min);
    let _ = writeln!(buf, "pub const OUTPUT_ACTIVATION_MAX: i32 = {};", act_max);
    let _ = writeln!(buf);

    cr::emit_const_array(&mut buf, "INPUT_DATA", "i8", &input_data);
    cr::emit_const_array(&mut buf, "WEIGHTS_DATA", "i8", &flat_w);
    cr::emit_const_array(&mut buf, "RECURRENT_WEIGHTS_DATA", "i8", &flat_u);
    cr::emit_const_array(&mut buf, "BIAS_DATA", "i32", &flat_b);
    cr::emit_const_array(&mut buf, "OUTPUT_MULTIPLIER", "i32", &[out_mult]);
    cr::emit_const_array(&mut buf, "OUTPUT_SHIFT", "i32", &[out_shift]);
    cr::emit_const_array(&mut buf, "EXPECTED_OUTPUT", "i8", &output);
    cr::emit_const_array(&mut buf, "INIT_HIDDEN_STATE", "i16", &init_hidden_state);

    let path = w.output_dir().join("gru.rs");
    std::fs::write(&path, buf).expect("write fixture");
    println!("  Wrote {}", path.display());
}

// Use crate::fixture alias for emit_const_array
mod cr {
    pub use crate::fixture::emit_const_array;
}
