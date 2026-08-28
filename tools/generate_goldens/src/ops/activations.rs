//! Activation golden fixtures — TFLM int8 reference activations.
//!
//! Each activation emits its quantization parameters explicitly.
//! Where an op is trivial under symmetric quantization (zero_point=0),
//! the simplification is exact and documented.
//! HardSwish uses a DOWNGRADED provenance (documented in the project notes).

use crate::tflm_math;
use crate::fixture::FixtureWriter;

/// ReLU: output = max(input_zero_point, input)
///
/// With symmetric quantization (input_zero_point=0, output_zero_point=0,
/// output_multiplier=1.0, output_shift=0), this simplifies to max(0, input).
/// The simplification is EXACT under these parameters per TFLM ReluParams.
pub fn generate_relu(w: &mut FixtureWriter) {
    let input_shape = [1i32, 1, 1, 8];
    let output_shape = [1i32, 1, 1, 8];
    let input: Vec<i8> = vec![-10, -5, -1, 0, 1, 5, 10, 127];
    let input_zero_point: i32 = 0;
    let output_zero_point: i32 = 0;
    let (output_multiplier, output_shift) = tflm_math::quantize_multiplier(1.0);

    let output: Vec<i8> = input.iter().map(|&x| {
        let val = i32::from(x) + input_zero_point;
        let act = val.max(0); // symmetric: 0 = quantized zero
        let scaled = tflm_math::multiply_by_quantized_multiplier(act, output_multiplier, output_shift);
        (scaled + output_zero_point).clamp(-128, 127) as i8
    }).collect();

    w.write_simple("relu", &input_shape, &output_shape, &input, &output,
        &[("input_zero_point", input_zero_point),
          ("output_zero_point", output_zero_point),
          ("output_multiplier", output_multiplier),
          ("output_shift", output_shift)],
        "// ReLU: output = max(0, input) — exact under symmetric quantization (zero_point=0, scale=1).");
}

/// ReLU6: output = clamp(input, 0, quantized_6)
///
/// With symmetric quantization (input_zero_point=0, output_zero_point=0,
/// scale=1), quantize(6.0) = round(6.0/1.0) + 0 = 6.
/// The simplification clamp(x, 0, 6) is EXACT under these parameters.
pub fn generate_relu6(w: &mut FixtureWriter) {
    let input_shape = [1i32, 1, 1, 8];
    let output_shape = [1i32, 1, 1, 8];
    let input: Vec<i8> = vec![-10, -5, 0, 3, 6, 10, 50, 127];
    let input_zero_point: i32 = 0;
    let output_zero_point: i32 = 0;
    // quantize(6.0) with scale=1, zero_point=0: round(6.0/1.0) + 0 = 6
    let quantized_six: i32 = 6;

    let output: Vec<i8> = input.iter().map(|&x| {
        let val = i32::from(x) + input_zero_point;
        let act = val.max(0).min(quantized_six);
        (act + output_zero_point).clamp(-128, 127) as i8
    }).collect();

    w.write_simple("relu6", &input_shape, &output_shape, &input, &output,
        &[("input_zero_point", input_zero_point),
          ("output_zero_point", output_zero_point),
          ("quantized_six", quantized_six)],
        "// ReLU6: output = clamp(input, 0, 6) — exact under symmetric quantization (zero_point=0, scale=1).");
}

/// HardSwish: x * ReLU6(x+3) / 6
///
/// ⚠️  GOLDEN_PROVENANCE DOWNGRADED.
/// The TFLM quantized HardSwish uses a 16-bit fixed-point chain with
/// SaturatingDoublingHighMul (round-to-zero), SaturatingRoundingDoublingHighMul,
/// and SaturatingLeftShift, operating on int16 input with HardSwishParams.
/// This fixture uses integer arithmetic with explicit division and rounding.
/// NOT bit-exact against executed TFLM HardSwish.
pub fn generate_hard_swish(w: &mut FixtureWriter) {
    let input_shape = [1i32, 1, 1, 8];
    let output_shape = [1i32, 1, 1, 8];
    let input: Vec<i8> = vec![-10, -5, -3, -1, 0, 3, 6, 10];
    let input_zero_point: i32 = 0;
    let output_zero_point: i32 = 0;

    let output: Vec<i8> = input.iter().map(|&x| {
        let x_i32 = i32::from(x) + input_zero_point;
        let relu6_arg = (x_i32 + 3).clamp(0, 6);
        let product = x_i32 * relu6_arg;
        let result = if product >= 0 {
            (product + 3) / 6
        } else {
            (product - 3) / 6
        };
        (result + output_zero_point).clamp(-128, 127) as i8
    }).collect();

    let mut buf = String::with_capacity(8192);
    buf.push_str(crate::fixture::SPDX_HEADER);
    buf.push('\n');
    buf.push_str(crate::fixture::PROVENANCE_NOTE);
    buf.push('\n');

    use std::fmt::Write as FmtWrite;
    let _ = writeln!(buf, "/// TFLite Micro pin that defines this golden corpus.");
    let _ = writeln!(buf, "pub const GOLDEN_TFLM_VERSION: &str = \"{}\";\n", crate::fixture::TFLM_VERSION);

    let _ = writeln!(buf, "/// Provenance of these golden values — DOWNGRADED for hard_swish.");
    let _ = writeln!(buf, "/// Deviation: uses integer division with sign-aware rounding instead of");
    let _ = writeln!(buf, "/// TFLM's 16-bit SaturatingDoublingHighMul/RoundingDivideByPOT chain.");
    let _ = writeln!(buf, "/// NOT bit-exact against executed TFLM HardSwish.");
    let _ = writeln!(buf, "/// T5.0 must implement the TFLM HardSwishParams chain with gemmlowp 16-bit primitives.");
    let _ = writeln!(buf, "pub const GOLDEN_PROVENANCE: &str = \"DOWNGRADED: integer-div-rational-approx; NOT TFLM-faithful\";\n");

    let _ = writeln!(buf, "pub const INPUT_SHAPE: [i32; 4] = {:?};", input_shape);
    let _ = writeln!(buf, "pub const OUTPUT_SHAPE: [i32; 4] = {:?};\n", output_shape);

    let _ = writeln!(buf, "pub const INPUT_ZERO_POINT: i32 = {};", input_zero_point);
    let _ = writeln!(buf, "pub const OUTPUT_ZERO_POINT: i32 = {};\n", output_zero_point);

    let _ = writeln!(buf, "// HardSwish: x * ReLU6(x+3) / 6 (integer rational approximation)");

    crate::fixture::emit_const_array(&mut buf, "INPUT_DATA", "i8", &input);
    crate::fixture::emit_const_array(&mut buf, "EXPECTED_OUTPUT", "i8", &output);

    let path = w.output_dir().join("hard_swish.rs");
    std::fs::write(&path, buf).expect("write fixture");
    println!("  Wrote {}", path.display());
}

/// LeakyReLU: f(x) = MultiplyByQuantizedMultiplier(x, id_mult, id_shift) if x >= 0
///                    else MultiplyByQuantizedMultiplier(x, alpha_mult, alpha_shift)
///
/// Implements the TFLM QuantizeLeakyRelu pattern from
/// tensorflow/lite/kernels/internal/reference/leaky_relu.h.
/// All quantization params are emitted as consts.
pub fn generate_leaky_relu(w: &mut FixtureWriter) {
    let input_shape = [1i32, 1, 1, 8];
    let output_shape = [1i32, 1, 1, 8];
    let input: Vec<i8> = vec![-50, -10, -5, -1, 0, 1, 5, 10];

    let input_offset: i32 = 0;
    let output_offset: i32 = 0;
    // Identity path: scale=1.0
    let (output_multiplier_identity, output_shift_identity) = tflm_math::quantize_multiplier(1.0);
    // Alpha path: alpha=0.2
    let output_multiplier_alpha: i32 = 1717986918; // 0.8 * 2^31 rounded
    let output_shift_alpha: i32 = -2;

    let output: Vec<i8> = input.iter().map(|&x| {
        let input_value = i32::from(x) - input_offset;
        let mut unclamped_output = output_offset;
        if input_value >= 0 {
            unclamped_output += tflm_math::multiply_by_quantized_multiplier(
                input_value, output_multiplier_identity, output_shift_identity);
        } else {
            unclamped_output += tflm_math::multiply_by_quantized_multiplier(
                input_value, output_multiplier_alpha, output_shift_alpha);
        }
        unclamped_output.clamp(-128, 127) as i8
    }).collect();

    w.write_simple("leaky_relu", &input_shape, &output_shape, &input, &output,
        &[("input_offset", input_offset),
          ("output_offset", output_offset),
          ("output_multiplier_identity", output_multiplier_identity),
          ("output_shift_identity", output_shift_identity),
          ("output_multiplier_alpha", output_multiplier_alpha),
          ("output_shift_alpha", output_shift_alpha)],
        "// LeakyReLU: TFLM QuantizeLeakyRelu pattern — MultiplyByQuantizedMultiplier per-branch.");
}

/// PReLU: f(x) = MultiplyByQuantizedMultiplier(x, mult_1, shift_1) if x >= 0
///                else MultiplyByQuantizedMultiplier(x * alpha[i], mult_2, shift_2)
///
/// Implements the TFLM Prelu reference from
/// tensorflow/lite/kernels/internal/reference/prelu.h.
/// Alpha values are per-channel int8 with their own zero_point.
pub fn generate_prelu(w: &mut FixtureWriter) {
    let input_shape = [1i32, 1, 1, 4];
    let output_shape = [1i32, 1, 1, 4];
    let input: Vec<i8> = vec![-20, -5, 10, -3];

    let input_offset: i32 = 0;
    let alpha_offset: i32 = 0;
    let output_offset: i32 = 0;
    // Identity path (positive branch): scale=1.0
    let (output_multiplier_1, output_shift_1) = tflm_math::quantize_multiplier(1.0);
    // Alpha path (negative branch): alpha is Q7-encoded (alpha * 128), so the
    // requantize must undo that scaling with scale = 1/128.
    let (output_multiplier_2, output_shift_2) = tflm_math::quantize_multiplier(1.0 / 128.0);

    // Alpha slopes encoded as Q7 (alpha_value = alpha * 128):
    // alpha=0.25 → 32, alpha=0.1 → 13, alpha=0.15 → 19, alpha=0.5 → 64
    let alpha_data: [i8; 4] = [32, 13, 19, 64];

    let output: Vec<i8> = input.iter().enumerate().map(|(i, &x)| {
        let input_value = i32::from(x) + input_offset;
        let output_value = if input_value >= 0 {
            tflm_math::multiply_by_quantized_multiplier(
                input_value, output_multiplier_1, output_shift_1)
        } else {
            let alpha_value = alpha_offset + i32::from(alpha_data[i]);
            tflm_math::multiply_by_quantized_multiplier(
                input_value * alpha_value, output_multiplier_2, output_shift_2)
        };
        (output_value + output_offset).clamp(-128, 127) as i8
    }).collect();

    // PReLU writes its own fixture to include ALPHA_DATA array
    // Also need to emit ALPHA_DATA. Use write instead of write_simple.
    // Actually, write_simple only emits INPUT_DATA and EXPECTED_OUTPUT.
    // We need a custom approach for alpha_data. Use write method.

    // Write the file with alpha data included
    let mut buf = String::with_capacity(8192);
    buf.push_str(crate::fixture::SPDX_HEADER);
    buf.push('\n');
    buf.push_str(crate::fixture::PROVENANCE_NOTE);
    buf.push('\n');

    use std::fmt::Write as FmtWrite;
    let _ = writeln!(buf, "/// TFLite Micro pin that defines this golden corpus.");
    let _ = writeln!(buf, "pub const GOLDEN_TFLM_VERSION: &str = \"{}\";\n", crate::fixture::TFLM_VERSION);

    let _ = writeln!(buf, "/// Provenance of these golden values.");
    let _ = writeln!(buf, "pub const GOLDEN_PROVENANCE: &str = \"tool-internal-reference-reimplementation; NOT captured from executed TFLM; see tools/generate_goldens/README.md\";\n");

    let _ = writeln!(buf, "pub const INPUT_SHAPE: [i32; 4] = {:?};", input_shape);
    let _ = writeln!(buf, "pub const OUTPUT_SHAPE: [i32; 4] = {:?};\n", output_shape);

    let _ = writeln!(buf, "pub const INPUT_OFFSET: i32 = {};", input_offset);
    let _ = writeln!(buf, "pub const ALPHA_OFFSET: i32 = {};", alpha_offset);
    let _ = writeln!(buf, "pub const OUTPUT_OFFSET: i32 = {};", output_offset);
    let _ = writeln!(buf, "pub const OUTPUT_MULTIPLIER_1: i32 = {};", output_multiplier_1);
    let _ = writeln!(buf, "pub const OUTPUT_SHIFT_1: i32 = {};", output_shift_1);
    let _ = writeln!(buf, "pub const OUTPUT_MULTIPLIER_2: i32 = {};", output_multiplier_2);
    let _ = writeln!(buf, "pub const OUTPUT_SHIFT_2: i32 = {};\n", output_shift_2);

    let _ = writeln!(buf, "// PReLU: TFLM Prelu reference — per-channel alpha (int8), MultiplyByQuantizedMultiplier chain.");

    crate::fixture::emit_const_array(&mut buf, "INPUT_DATA", "i8", &input);
    crate::fixture::emit_const_array(&mut buf, "ALPHA_DATA", "i8", &alpha_data);
    crate::fixture::emit_const_array(&mut buf, "EXPECTED_OUTPUT", "i8", &output);

    let path = w.output_dir().join("prelu.rs");
    std::fs::write(&path, buf).expect("write fixture");
    println!("  Wrote {}", path.display());
}

/// Sigmoid (int8): TFLM `reference_integer_ops::Logistic` path.
///
/// Algorithm (pinned SHA `tensorflow/lite/kernels/internal/reference/integer_ops/logistic.h`):
/// 1. `input = input_data[i] - input_zero_point`
/// 2. Saturing: `input <= -input_range_radius → -128`, `input >= input_range_radius → 127`
/// 3. `input_in_q4 = MultiplyByQuantizedMultiplier(input, input_multiplier, input_left_shift)`
///    — converts to Q4.27 fixed-point.
/// 4. `output_in_q0 = gemmlowp::logistic(FixedPoint4::FromRaw(input_in_q4)).raw()` — Q0.31.
/// 5. `output_in_q23 = RoundingDivideByPOT(output_in_q0, 31 - 8)` — rescale (kOutputIntegerBits=8).
/// 6. `output = clamp(output_in_q23 + kOutputZeroPoint(-128), -128, 127)`.
///
/// Quantization params chosen to match a TFLite int8 LOGISTIC node with
/// input_scale = 1/16 (covers the Q4.27 ±8 real range in int8 [-128,127]):
/// - input_real_multiplier = input_scale · 2^(31-4) = 2^23
/// - frexp(2^23) → (0.5, 24) → input_multiplier = round(0.5·2^31) = 2^30, input_left_shift = 24
/// - input_range_radius = floor(15 · 2^27 / 2^24) = 120
pub fn generate_sigmoid(w: &mut FixtureWriter) {
    let input_shape = [1i32, 1, 1, 11];
    let output_shape = [1i32, 1, 1, 11];

    let input_scale = 1.0f64 / 16.0;
    // input_real_multiplier = input_scale * 2^(31-4) = 2^23
    let input_real_multiplier = input_scale * ((1i64 << 27) as f64);
    let (input_multiplier, input_left_shift) = tflm_math::quantize_multiplier(input_real_multiplier);
    // CalculateInputRadius(4, input_left_shift, 31) = floor(15 * 2^27 / 2^shift)
    let input_range_radius =
        ((15i64 * (1i64 << 27)) >> input_left_shift) as i32;

    let input_zero_point: i32 = 0;
    let output_zero_point: i32 = -128; // kOutputZeroPoint = int8::min

    // Real values: ±7.5, ±5, ±2.5, ±1.25, ±0.625, 0 (scale 1/16)
    let input: Vec<i8> = vec![-120, -80, -40, -20, -10, 0, 10, 20, 40, 80, 120];

    let output: Vec<i8> = input.iter().map(|&x| {
        let input_val = i32::from(x) - input_zero_point;
        if input_val <= -input_range_radius {
            return -128i8;
        }
        if input_val >= input_range_radius {
            return 127i8;
        }
        let input_in_q4 = tflm_math::multiply_by_quantized_multiplier(
            input_val, input_multiplier, input_left_shift);
        let output_in_q0 = tflm_math::logistic_q4_27(input_in_q4);
        let output_in_q23 = tflm_math::rounding_divide_by_pot(output_in_q0, 31 - 8);
        (output_in_q23 + output_zero_point).clamp(-128, 127) as i8
    }).collect();

    w.write_simple("sigmoid", &input_shape, &output_shape, &input, &output,
        &[("input_offset", input_zero_point),
          ("output_offset", output_zero_point),
          ("input_multiplier", input_multiplier),
          ("input_left_shift", input_left_shift),
          ("input_range_radius", input_range_radius)],
        "// Sigmoid: TFLM reference_integer_ops::Logistic — gemmlowp logistic on Q4.27 input, RoundingDivideByPOT(31-8), zero_point=-128.");
}

/// Tanh (int8): TFLM `reference_integer_ops::Tanh` path.
///
/// Algorithm (pinned SHA `tensorflow/lite/kernels/internal/reference/integer_ops/tanh.h`):
/// 1. `input = input_data[i] - input_zero_point`
/// 2. Saturing: `input <= -input_range_radius → -128`, `input >= input_range_radius → 127`
/// 3. `input_in_q4 = MultiplyByQuantizedMultiplier(input, input_multiplier, input_shift)`
///    — converts to Q4.27 fixed-point.
/// 4. `output_in_q0 = gemmlowp::tanh(FixedPoint4::FromRaw(input_in_q4)).raw()` — Q0.31.
/// 5. `output_in_q24 = RoundingDivideByPOT(output_in_q0, 31 - 7)` — rescale (kOutputScale=7).
/// 6. `output = clamp(output_in_q24, -128, 127)` — NO zero-point offset for tanh.
///
/// Same quantization params as sigmoid (input_scale = 1/16 → multiplier 2^30,
/// shift 24, radius 120).
pub fn generate_tanh(w: &mut FixtureWriter) {
    let input_shape = [1i32, 1, 1, 11];
    let output_shape = [1i32, 1, 1, 11];

    let input_scale = 1.0f64 / 16.0;
    let input_real_multiplier = input_scale * ((1i64 << 27) as f64);
    let (input_multiplier, input_left_shift) = tflm_math::quantize_multiplier(input_real_multiplier);
    let input_range_radius =
        ((15i64 * (1i64 << 27)) >> input_left_shift) as i32;

    let input_zero_point: i32 = 0;

    let input: Vec<i8> = vec![-120, -80, -40, -20, -10, 0, 10, 20, 40, 80, 120];

    let output: Vec<i8> = input.iter().map(|&x| {
        let input_val = i32::from(x) - input_zero_point;
        if input_val <= -input_range_radius {
            return -128i8;
        }
        if input_val >= input_range_radius {
            return 127i8;
        }
        let input_in_q4 = tflm_math::multiply_by_quantized_multiplier(
            input_val, input_multiplier, input_left_shift);
        let output_in_q0 = tflm_math::tanh_q4_27(input_in_q4);
        let output_in_q24 = tflm_math::rounding_divide_by_pot(output_in_q0, 31 - 7);
        output_in_q24.clamp(-128, 127) as i8
    }).collect();

    w.write_simple("tanh", &input_shape, &output_shape, &input, &output,
        &[("input_offset", input_zero_point),
          ("output_offset", 0),
          ("input_multiplier", input_multiplier),
          ("input_left_shift", input_left_shift),
          ("input_range_radius", input_range_radius)],
        "// Tanh: TFLM reference_integer_ops::Tanh — gemmlowp tanh on Q4.27 input, RoundingDivideByPOT(31-7), no zero-point offset.");
}
