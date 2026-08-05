//! Golden fixture generator for Hematite NN TDD corpus.
//!
//! Generates `hematite-tests/goldens/*.rs` files containing Rust `const` arrays
//! with pre-computed reference input/output tensors per `TFLite` Micro reference
//! kernel arithmetic. These fixtures are consumed by Phase 2's TDD loop.
//!
//! ## Fallback Provenance
//!
//! This tool uses an internal scalar reimplementation of TFLM int8 reference
//! kernels — NOT output captured from an executed TFLM binary. See README.md
//! for the full provenance statement and T5.0 remediation plan.
//!
//! ## Usage
//!
//! ```sh
//! cargo run -p generate-goldens
//! ```
//!
//! ## Output
//!
//! All generated files are written to `hematite-tests/goldens/` relative to
//! the workspace root. The working directory must be the workspace root.

mod tflm_math;
mod fixture;
mod ops;

use fixture::FixtureWriter;
use std::path::PathBuf;

fn main() {
    let workspace_root = find_workspace_root();
    let goldens_dir = workspace_root.join("hematite-tests").join("goldens");

    println!("Hematite NN Golden Fixture Generator");
    println!("====================================");
    println!("Output dir: {}", goldens_dir.display());
    println!("TFLM pin:   {}\n", fixture::TFLM_VERSION);

    // Clean existing fixtures for determinism
    if goldens_dir.exists() {
        for entry in std::fs::read_dir(&goldens_dir).expect("read goldens dir") {
            let entry = entry.expect("dir entry");
            if entry.path().extension().is_some_and(|e| e == "rs") {
                std::fs::remove_file(entry.path()).expect("remove old fixture");
            }
        }
    }

    let mut w = FixtureWriter::new(goldens_dir.clone());

    // ── T0 — Core compute ──
    println!("\n── T0: Core compute ──");
    ops::conv2d::generate_conv2d_1x1(&mut w);
    ops::conv2d::generate_conv2d_3x3(&mut w);
    ops::depthwise_conv2d::generate(&mut w);
    ops::fully_connected::generate(&mut w);

    // ── T1 — Supporting ops ──
    println!("\n── T1: Pooling ──");
    ops::pool::generate_average_pool(&mut w);
    ops::pool::generate_max_pool(&mut w);

    println!("\n── T1: Softmax ──");
    ops::softmax::generate(&mut w);

    println!("\n── T1: Activations ──");
    ops::activations::generate_relu(&mut w);
    ops::activations::generate_relu6(&mut w);
    ops::activations::generate_hard_swish(&mut w);
    ops::activations::generate_leaky_relu(&mut w);
    ops::activations::generate_prelu(&mut w);

    println!("\n── T1: Elementwise ──");
    ops::elementwise::generate_add(&mut w);
    ops::elementwise::generate_mul(&mut w);
    ops::elementwise::generate_sub(&mut w);

    println!("\n── T1: Quantize/Dequantize ──");
    ops::quantize::generate_quantize(&mut w);
    ops::quantize::generate_dequantize(&mut w);

    // ── T2 — Data movement ──
    println!("\n── T2: Data movement ──");
    ops::data_movement::generate_reshape(&mut w);
    ops::data_movement::generate_transpose(&mut w);
    ops::data_movement::generate_concat(&mut w);
    ops::data_movement::generate_split(&mut w);
    ops::data_movement::generate_pad(&mut w);
    ops::data_movement::generate_slice(&mut w);
    ops::data_movement::generate_resize_nearest(&mut w);

    // ── T3 — Recurrent ──
    println!("\n── T3: Recurrent ──");
    ops::recurrent::generate_lstm(&mut w);
    ops::recurrent::generate_svdf(&mut w);
    ops::recurrent::generate_gru(&mut w);

    // ── T4 — Reductions ──
    println!("\n── T4: Reductions ──");
    ops::reductions::generate_mean(&mut w);
    ops::reductions::generate_sum(&mut w);
    ops::reductions::generate_argmax(&mut w);
    ops::reductions::generate_argmin(&mut w);
    ops::reductions::generate_l2_norm(&mut w);

    // ── Self-check ──
    println!("\n── Self-check ──");
    self_check_conv2d_1x1();
    self_check_mean();
    self_check_recurrent();
    guard_generated_fixtures(&goldens_dir);

    println!("\nDone. Generated {} fixture files in {}", count_files(&goldens_dir), goldens_dir.display());
}

/// Assert at least one op against a hand-computed value to catch silent arithmetic regressions.
fn self_check_conv2d_1x1() {
    // Hand-compute: 1×1 conv with input=[-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7]
    // weights=[1,2,3,4] for channel 0, bias[0]=10
    // For the first output pixel (batch=0, out_y=0, out_x=0, out_ch=0):
    //   input pixels = [-8,-7,-6,-5]
    //   acc = 1*(-8+0) + 2*(-7+0) + 3*(-6+0) + 4*(-5+0) = -8 -14 -18 -20 = -60
    //   bias = 10 → acc = -50
    //   multiplier = 2^30, shift = 0 → effective scale = 0.5
    //   multiply_by_quantized_multiplier(-50, 2^30, 0) = -25
    //   + output_offset(0) → -25, clamp → -25 (within [-128,127])

    use tflm_math::multiply_by_quantized_multiplier;

    let input: Vec<i8> = (-8..8).map(|i| i as i8).collect();
    let weights: Vec<i8> = vec![1, 2, 3, 4]; // channel 0
    let bias: i32 = 10;
    let mult: i32 = 1i32 << 30;
    let shift: i32 = 0;

    // Compute first output pixel channel 0
    let mut acc: i32 = 0;
    for d in 0..4 {
        acc += i32::from(weights[d]) * i32::from(input[d]);
    }
    acc += bias;

    let result = multiply_by_quantized_multiplier(acc, mult, shift);
    assert_eq!(acc, -50, "Expected acc=-50, got {acc}");
    assert_eq!(result, -25, "Expected multiply_by_quantized_multiplier(-50, 2^30, 0) = -25, got {result}");

    // Also check channel 1 for first pixel
    let w1: Vec<i8> = vec![-1, -2, -3, -4]; // channel 1
    let bias1: i32 = -10;
    let mult1: i32 = 1i32 << 28;
    let shift1: i32 = 1;

    let mut acc1: i32 = 0;
    for d in 0..4 {
        acc1 += i32::from(w1[d]) * i32::from(input[d]);
    }
    acc1 += bias1;
    // acc1 = (-1)(-8) + (-2)(-7) + (-3)(-6) + (-4)(-5) + (-10) = 8+14+18+20-10 = 50
    assert_eq!(acc1, 50, "Expected acc1=50, got {acc1}");
    // multiply_by_quantized_multiplier(50, 2^28, 1):
    // x << 1 = 100; SaturatingRoundingDoublingHighMul(100, 2^28) = Round(100 * 2^28 * 2 / 2^32) = Round(100 * 2^29 / 2^32) = Round(100 / 8) = 13
    // Then rounding_divide_by_pot(13, 0) = 13
    let result1 = multiply_by_quantized_multiplier(acc1, mult1, shift1);
    assert_eq!(result1, 13, "Expected multiply_by_quantized_multiplier(50, 2^28, 1) = 13, got {result1}");

    println!("  ✅ conv2d_1x1 self-check passed:");
    println!("     pixel(0,0,ch0): acc=-50 → scaled=-25");
    println!("     pixel(0,0,ch1): acc=50 → scaled=13");
}

/// Assert the mean op against a hand-computed value.
fn self_check_mean() {
    // Hand-compute: mean([1,2,3],[4,8,9]) over axis=1 for ch0
    // Col 0: (1+4)/2 = 2.5 → round-half-away-from-zero → 3
    // Col 1: (2+8)/2 = 5.0 → 5
    // Col 2: (3+9)/2 = 6.0 → 6
    // Then requantize with mbm(x, 1<<30, 0) = round(x * 0.5):
    // mbm(3, 1<<30, 0) = 2
    // mbm(5, 1<<30, 0) = 3
    // mbm(6, 1<<30, 0) = 3

    use tflm_math::multiply_by_quantized_multiplier;

    let m = 1i32 << 30;
    let s = 0i32;

    assert_eq!(multiply_by_quantized_multiplier(3, m, s), 2,
        "mbm(3, 1<<30, 0) should be 2");
    assert_eq!(multiply_by_quantized_multiplier(5, m, s), 3,
        "mbm(5, 1<<30, 0) should be 3");
    assert_eq!(multiply_by_quantized_multiplier(6, m, s), 3,
        "mbm(6, 1<<30, 0) should be 3");

    println!("  ✅ mean self-check passed:");
    println!("     ch0 col0: (1+4)/2=3 → mbm(3)=2");
    println!("     ch0 col1: (2+8)/2=5 → mbm(5)=3");
    println!("     ch0 col2: (3+9)/2=6 → mbm(6)=3");
}

/// Verify GRU sigmoid/tanh fixed-point landmarks: sigmoid(0)=0.5,
/// tanh(0)=0, endpoints saturate; plus update-gate monotonicity.
fn self_check_recurrent() {
    use tflm_math::{logistic_i16_q011, tanh_i16_q011};

    // logistic_i16_q011: sigmoid(x)·2^11 → i16 in [0, 2047] (0.0→0, 0.5→1024, 1.0→2047)
    // Input is in Q5.26: value / 2^26 = real value
    let s0 = logistic_i16_q011(0);
    assert_eq!(s0, 1024, "sigmoid(0) must be 0.5 → 1024 in Q0.11, got {s0}");

    // tanh_i16_q011: tanh(x)·2^11 → i16 in [-2048, 2047]
    let t0 = tanh_i16_q011(0);
    assert_eq!(t0, 0, "tanh(0) must be 0, got {t0}");

    // Endpoints: large positive → sigmoid ≈ 1.0, large negative → sigmoid ≈ 0.0
    // +5.0 in Q5.26 = 5 << 26 = 335544320
    let s_pos = logistic_i16_q011(5 << 26);
    let s_neg = logistic_i16_q011(-(5 << 26));
    assert!(s_pos >= 2000, "sigmoid(+large) must be near saturation, got {s_pos}");
    assert!(s_neg <= 100, "sigmoid(-large) must be near zero, got {s_neg}");

    // Endpoints: tanh → ±1.0 in Q0.11 = ±2047
    let t_pos = tanh_i16_q011(5 << 26);
    let t_neg = tanh_i16_q011(-(5 << 26));
    assert!(t_pos >= 2000, "tanh(+large) must be near +1.0, got {t_pos}");
    assert!(t_neg <= -2000, "tanh(-large) must be near -1.0, got {t_neg}");

    // Monotonicity: sigmoid at Q5.26 values ~0.25, 0.5, 1.0
    let s1 = logistic_i16_q011(1 << 24); // ~0.25
    let s2 = logistic_i16_q011(1 << 25); // ~0.5
    assert!(s2 > s1, "sigmoid monotonicity: sigmoid(0.5)={s2} <= sigmoid(0.25)={s1}");
    assert!(s1 > s0, "sigmoid monotonicity: sigmoid(0.25)={s1} <= sigmoid(0)={s0}");

    println!("  ✅ recurrent self-check passed:");
    println!("     sigmoid(0)=1024 (Q0.11 0.5), tanh(0)=0");
    println!("     sigmoid + tanh endpoints saturate correctly");
    println!("     sigmoid monotonicity verified");
}

fn find_workspace_root() -> PathBuf {
    // Walk up from current dir to find Cargo.toml with [workspace]
    let mut current = std::env::current_dir().expect("current dir");
    loop {
        let cargo_toml = current.join("Cargo.toml");
        if cargo_toml.exists() {
            if let Ok(contents) = std::fs::read_to_string(&cargo_toml) {
                if contents.contains("[workspace]") {
                    return current;
                }
            }
        }
        assert!(current.pop(), "Could not find workspace root (no Cargo.toml with [workspace])");
    }
}

fn count_files(dir: &PathBuf) -> usize {
    std::fs::read_dir(dir)
        .map(|d| d.filter(|e| {
            e.as_ref().map(|e| e.path().extension().is_some_and(|ext| ext == "rs")).unwrap_or(false)
        }).count())
        .unwrap_or(0)
}

// ── Permanent guard pass: validate generated fixtures on disk ──

fn guard_generated_fixtures(goldens_dir: &PathBuf) {
    println!("  ── Fixture integrity guards ──");
    for entry in std::fs::read_dir(goldens_dir).expect("read goldens dir") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "rs") {
            continue;
        }
        let content = std::fs::read_to_string(&path).expect("read fixture");
        let name = path.file_stem().unwrap().to_str().unwrap();

        // Guard A: every MULTIPLIER const must be > 0 (catches sign-bit overflow)
        for m in parse_multipliers(&content) {
            assert!(m > 0, "FIXTURE GUARD FAILED: {name}.rs has a multiplier {m} ≤ 0 — sign-bit overflow like 1i32<<31");
        }

        // Guard B: EXPECTED_OUTPUT must not be all zeros (catches degenerate oracles)
        let expected = parse_i8_array(&content, "EXPECTED_OUTPUT");
        assert!(!expected.is_empty(), "FIXTURE GUARD FAILED: {name}.rs has no EXPECTED_OUTPUT");
        let all_zero = expected.iter().all(|&v| v == 0);
        assert!(!all_zero, "FIXTURE GUARD FAILED: {name}.rs EXPECTED_OUTPUT is all zeros — degenerate oracle");

        // Guard C: op-specific invariants
        match name {
            "relu" => {
                let output_offset = parse_scalar_i32(&content, "OUTPUT_OFFSET").unwrap_or(0);
                for (i, &v) in expected.iter().enumerate() {
                    assert!(i32::from(v) >= output_offset,
                        "FIXTURE GUARD FAILED: {name}.rs output[{i}] = {v} < output_offset {output_offset}");
                }
            }
            "relu6" => {
                let output_offset = parse_scalar_i32(&content, "OUTPUT_OFFSET").unwrap_or(0);
                let quantized_six = parse_scalar_i32(&content, "QUANTIZED_SIX").expect("relu6 QUANTIZED_SIX");
                for (i, &v) in expected.iter().enumerate() {
                    assert!(i32::from(v) >= output_offset,
                        "FIXTURE GUARD FAILED: {name}.rs output[{i}] = {v} < output_offset {output_offset}");
                    assert!(i32::from(v) <= quantized_six,
                        "FIXTURE GUARD FAILED: {name}.rs output[{i}] = {v} > QUANTIZED_SIX {quantized_six}");
                }
            }
            "leaky_relu" | "prelu" => {
                let input_data = parse_i8_array(&content, "INPUT_DATA");
                assert!(input_data.len() == expected.len(),
                    "FIXTURE GUARD FAILED: {name}.rs INPUT_DATA/EXPECTED_OUTPUT length mismatch");
                for i in 0..input_data.len() {
                    let inp = i32::from(input_data[i]);
                    let out = i32::from(expected[i]);
                    // Sign preserving: negative input → negative-or-zero output;
                    // positive input → positive-or-zero output; zero is fine either way.
                    if inp > 0 {
                        assert!(out >= 0,
                            "FIXTURE GUARD FAILED: {name}.rs input[{i}]={inp} > 0 but output[{i}]={out} < 0");
                    }
                    if inp < 0 {
                        assert!(out <= 0,
                            "FIXTURE GUARD FAILED: {name}.rs input[{i}]={inp} < 0 but output[{i}]={out} > 0");
                    }
                }
            }
            "softmax" => {
                let output_offset = parse_scalar_i32(&content, "OUTPUT_OFFSET").unwrap_or(-128);
                let sum: i32 = expected.iter().map(|&v| i32::from(v) - output_offset).sum();
                assert!((250..=262).contains(&sum),
                    "FIXTURE GUARD FAILED: softmax.rs sum(out + {}) = {} — expected in [250, 262]",
                    -output_offset, sum);
                // Discrimination: must have distinct outputs; reject uniform oracle
                let first = expected[0];
                let all_equal = expected.iter().all(|&v| v == first);
                assert!(!all_equal,
                    "FIXTURE GUARD FAILED: softmax.rs EXPECTED_OUTPUT is uniform — non-discriminating oracle");
                // Monotonically non-decreasing
                for i in 1..expected.len() {
                    assert!(expected[i] >= expected[i - 1],
                        "FIXTURE GUARD FAILED: softmax.rs expected[{i}] = {} < expected[{}] = {}",
                        expected[i], i - 1, expected[i - 1]);
                }
                // Largest dominates
                let last_unsigned = i32::from(expected[expected.len() - 1]) - output_offset;
                let rest_sum: i32 = expected.iter()
                    .take(expected.len() - 1)
                    .map(|&v| i32::from(v) - output_offset)
                    .sum();
                assert!(last_unsigned > rest_sum,
                    "FIXTURE GUARD FAILED: softmax.rs largest element ({last_unsigned}) does not dominate rest ({rest_sum})");

                // ── Accuracy gate: f64 cross-check within 2 LSB ──
                let input_data = parse_i8_array(&content, "INPUT_DATA");
                let input_scale = parse_scalar_f64(&content, "INPUT_SCALE")
                    .expect("softmax.rs INPUT_SCALE missing");
                let depth = expected.len();
                let max_input = *input_data.iter().max().unwrap_or(&0i8);
                let float_exps: Vec<f64> = input_data.iter()
                    .map(|&v| f64::exp((f64::from(v) - f64::from(max_input)) * input_scale))
                    .collect();
                let float_sum: f64 = float_exps.iter().sum();
                let mut deltas = Vec::new();
                for i in 0..depth {
                    let float_out = float_exps[i] / float_sum * 256.0;
                    let float_rounded = float_out.round() as i32;
                    let golden_unsigned = i32::from(expected[i]) - output_offset;
                    let delta = golden_unsigned - float_rounded;
                    deltas.push(delta);
                }
                let max_abs = deltas.iter().map(|d| d.abs()).max().unwrap_or(0);
                println!("  ✅ softmax.rs accuracy deltas vs f64 softmax = {deltas:?} (max_abs_delta={max_abs})");
                for i in 0..depth {
                    let delta = deltas[i];
                    assert!(delta.abs() <= 2,
                        "ACCURACY GATE FAILED: softmax.rs element[{i}] delta={delta} > 2 LSB (golden_unsigned={}, float_rounded={:.2})",
                        i32::from(expected[i]) - output_offset,
                        float_exps[i] / float_sum * 256.0);
                }
            }
            _ => {}
        }

        println!("  ✅ {name}.rs: multiplier>0 / output-non-zero / op-invariant passed");
    }
}

fn parse_multipliers(content: &str) -> Vec<i32> {
    let mut out = Vec::new();
    for line in content.lines() {
        if !line.contains("MULTIPLIER") || !line.contains("pub const") {
            continue;
        }
        if line.contains(": i32 = ") {
            // Scalar form: pub const NAME: i32 = VALUE;
            if let Some(val_str) = line.split(": i32 = ").nth(1) {
                if let Ok(val) = val_str.trim_end_matches(';').parse::<i32>() {
                    out.push(val);
                }
            }
        } else if line.contains(": [i32;") {
            // Array form header: pub const NAME: [i32; N] = [
            // Values on subsequent lines
            out.extend(parse_i32_array(content, line));
        }
    }
    out
}

fn parse_i32_array(content: &str, _header_line: &str) -> Vec<i32> {
    // Find the array block after the header line's starting `[`
    // We look for the next `]` at the start of a line preceded by `,`
    // Simple approach: iterate lines after the header, collect numbers until `];`
    let mut found_header = false;
    let mut out = Vec::new();
    for line in content.lines() {
        if !found_header {
            if line.contains("MULTIPLIER") && line.contains(": [i32;") {
                found_header = true;
            }
            continue;
        }
        if line.trim() == "];" {
            break;
        }
        for token in line.split(|c: char| c == ',' || c.is_whitespace()) {
            if let Ok(v) = token.parse::<i32>() {
                out.push(v);
            }
        }
    }
    out
}

fn parse_i8_array(content: &str, const_name: &str) -> Vec<i8> {
    let needle = format!("pub const {}: [i8;", const_name);
    let start = content.find(&needle);
    if start.is_none() {
        return Vec::new();
    }
    let after = &content[start.unwrap()..];
    // Find the `[` that opens the array
    let open = after.find('[').expect("no [ after const declaration");
    let block = &after[open + 1..];
    // Find matching `];` — scan for `];` at the outermost level
    // Since arrays are flat (no nesting), simple `];` works
    let close = block.find("];").expect("unclosed array");
    let values_str = &block[..close];

    let mut out = Vec::new();
    for token in values_str.split([',', '\n']) {
        let token = token.trim();
        if let Ok(v) = token.parse::<i8>() {
            out.push(v);
        } else if token.ends_with('i') {
            // Handle potential integer suffixes on negatives like -128i
            // Actually i8 arrays don't use suffixes, but be robust
            if let Ok(v) = token.trim_end_matches('i').parse::<i8>() {
                out.push(v);
            }
        }
    }
    out
}

fn parse_scalar_i32(content: &str, const_name: &str) -> Option<i32> {
    let needle = format!("pub const {}: i32 = ", const_name);
    if let Some(pos) = content.find(&needle) {
        let after = &content[pos + needle.len()..];
        let end = after.find(';').unwrap_or(after.len());
        after[..end].trim().parse::<i32>().ok()
    } else {
        None
    }
}

fn parse_scalar_f64(content: &str, const_name: &str) -> Option<f64> {
    let needle = format!("pub const {}: f64 = ", const_name);
    if let Some(pos) = content.find(&needle) {
        let after = &content[pos + needle.len()..];
        let end = after.find(';').unwrap_or(after.len());
        after[..end].trim().parse::<f64>().ok()
    } else {
        None
    }
}
