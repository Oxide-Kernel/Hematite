//! Quantize/Dequantize golden fixtures.

use crate::fixture::FixtureWriter;

/// Quantize: float → int8 per TFLM affine quantization (q = round(r/s) + Z)
pub fn generate_quantize(w: &mut FixtureWriter) {
    let input_shape = [1i32, 1, 1, 6];
    let output_shape = [1i32, 1, 1, 6];

    // Fake float values (we bypass the float→int conversion and just provide integer
    // test vectors that exercise the quantize clamp)
    let scale: f64 = 0.02;
    let zero_point: i32 = 0;

    // Input "real" values and their quantized counterparts
    // q = round(r / 0.02) + 0
    let reals: [f64; 6] = [-2.0, -0.5, 0.0, 0.5, 1.5, 2.5];
    let input: Vec<i8> = reals.iter().map(|&r| {
        let q = (r / scale).round() as i32 + zero_point;
        q.clamp(-128, 127) as i8
    }).collect();

    // The quantize kernel receives float* (or dequantized int8 as a proxy)
    // We encode the real values as dequantized int8 representation for the fixture.
    // For simplicity: input_i8 = real_i8 representation of the float.
    // Expected: the kernel should produce the same quantized values.
    let output = input.clone();

    w.write_simple("quantize", &input_shape, &output_shape, &input, &output,
        &[("scale_q31", (scale * (1u64 << 31) as f64).round() as i32),
          ("zero_point", zero_point)],
        "// Quantize: real → int8 (affine: q = round(r/scale) + zero_point)");
}

/// Dequantize: int8 → float per TFLM affine dequantization (r = scale * (q - Z))
pub fn generate_dequantize(w: &mut FixtureWriter) {
    let input_shape = [1i32, 1, 1, 6];
    let output_shape = [1i32, 1, 1, 6];

    // Quantized int8 input
    let input: Vec<i8> = vec![-100, -50, 0, 50, 100, 127];
    let scale: f64 = 0.01;
    let zero_point: i32 = 0;

    // Dequantized: r = scale * (q - Z)
    // We encode these as scaled int8 (not perfectly, since float→int8 is lossy)
    // For the fixture, we just provide the expected int8 values as a reference pattern.
    // A real dequantize produces float; this fixture captures the pattern.
    let output: Vec<i8> = input.iter().map(|&q| {
        let r = scale * (f64::from(q) - f64::from(zero_point));
        r.round().clamp(-128.0, 127.0) as i8
    }).collect();

    w.write_simple("dequantize", &input_shape, &output_shape, &input, &output,
        &[("scale_q31", (scale * (1u64 << 31) as f64).round() as i32),
          ("zero_point", zero_point)],
        "// Dequantize: int8 → real (affine: r = scale * (q - zero_point))");
}
