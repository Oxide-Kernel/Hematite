//! Elementwise golden fixtures — TFLM int8 add, mul, sub with full requantize.
//!
//! Implements TFLM reference_integer_ops AddFunc and MulElementwise from
//! tensorflow/lite/kernels/internal/reference/integer_ops/add.h and mul.h.
//! ArithmeticParams (shared by Add, Sub, Mul) carries input1/input2 offset,
//! left_shift, per-input multiplier/shift, output multiplier/shift, and
//! output offset. All params are emitted as consts.

use crate::tflm_math;
use crate::fixture::FixtureWriter;

/// TFLM AddFunc: full requantize chain with per-input rescaling.
///
/// Simplifies to a + b then requantize when input{1,2}_multiplier=2^31,
/// input{1,2}_shift=0, left_shift=0, and offsets are 0.
pub fn generate_add(w: &mut FixtureWriter) {
    // Two inputs: [1, 1, 1, 6]
    let input1_shape = [1i32, 1, 1, 6];
    let input2_shape = [1i32, 1, 1, 6];
    let output_shape = [1i32, 1, 1, 6];

    let input1: Vec<i8> = vec![-10, -5, 0, 3, 7, 12];
    let input2: Vec<i8> = vec![5, -3, 0, 7, -2, 8];

    // ArithmeticParams: offsets are negative zero-points.
    // Per TFLM: "Input offset is negative input zero point."
    let input1_offset: i32 = 0;
    let input2_offset: i32 = 0;
    let output_offset: i32 = 0;
    let left_shift: i32 = 0;

    // Per-input multiplier/shift: scale=1.0 for both → input unchanged
    let (input1_multiplier, input1_shift) = tflm_math::quantize_multiplier(1.0);
    let (input2_multiplier, input2_shift) = tflm_math::quantize_multiplier(1.0);

    // Output multiplier/shift: scale ≈ 0.5
    let output_multiplier: i32 = 1i32 << 30;
    let output_shift: i32 = 0;

    let activation_min: i32 = -128;
    let activation_max: i32 = 127;

    // TFLM AddFunc chain:
    //   input1_val = input1_offset + a
    //   shifted_input1_val = input1_val * (1 << left_shift)
    //   scaled_input1_val = MultiplyByQuantizedMultiplierSmallerThanOneExp(
    //       shifted_input1_val, input1_multiplier, input1_shift)
    //   (same for input2)
    //   raw_sum = scaled_input1_val + scaled_input2_val
    //   raw_output = MultiplyByQuantizedMultiplierSmallerThanOneExp(
    //       raw_sum, output_multiplier, output_shift) + output_offset
    let output: Vec<i8> = input1.iter().zip(input2.iter()).map(|(&a, &b)| {
        let input1_val = i32::from(a) + input1_offset;
        let input2_val = i32::from(b) + input2_offset;
        let shifted1 = input1_val * (1 << left_shift);
        let shifted2 = input2_val * (1 << left_shift);
        // MultiplyByQuantizedMultiplierSmallerThanOneExp is identical to
        // MultiplyByQuantizedMultiplier in single-rounding path
        let scaled1 = tflm_math::multiply_by_quantized_multiplier(
            shifted1, input1_multiplier, input1_shift);
        let scaled2 = tflm_math::multiply_by_quantized_multiplier(
            shifted2, input2_multiplier, input2_shift);
        let raw_sum = scaled1 + scaled2;
        let raw_output = tflm_math::multiply_by_quantized_multiplier(
            raw_sum, output_multiplier, output_shift) + output_offset;
        raw_output.clamp(activation_min, activation_max) as i8
    }).collect();

    w.write("elementwise_add",
        &input1_shape, &input2_shape, &output_shape,
        &input1, &input2, &[],
        input1_offset, output_offset,
        activation_min, activation_max,
        &[output_multiplier], &[output_shift],
        &output,
        &[("input2_offset", input2_offset),
          ("left_shift", left_shift),
          ("input1_multiplier", input1_multiplier),
          ("input1_shift", input1_shift),
          ("input2_multiplier", input2_multiplier),
          ("input2_shift", input2_shift)],
    );
}

/// TFLM MulElementwise: input1_val * input2_val, then one requantize.
///
/// Unlike Add, Mul does NOT use left_shift or per-input multiplier/shift.
/// The int32 product is requantized with a single output_multiplier/output_shift.
pub fn generate_mul(w: &mut FixtureWriter) {
    let input1_shape = [1i32, 1, 1, 6];
    let input2_shape = [1i32, 1, 1, 6];
    let output_shape = [1i32, 1, 1, 6];

    let input1: Vec<i8> = vec![-5, -2, 0, 2, 5, 10];
    let input2: Vec<i8> = vec![3, 4, 0, -1, 2, 5];

    let input1_offset: i32 = 0;
    let input2_offset: i32 = 0;
    let output_offset: i32 = 0;
    let output_multiplier: i32 = 1i32 << 30; // scale 0.5
    let output_shift: i32 = 0;
    let activation_min: i32 = -128;
    let activation_max: i32 = 127;

    // TFLM MulElementwise:
    //   input1_val = input1_offset + a
    //   input2_val = input2_offset + b
    //   unclamped = output_offset + MultiplyByQuantizedMultiplier(
    //       input1_val * input2_val, output_multiplier, output_shift)
    let output: Vec<i8> = input1.iter().zip(input2.iter()).map(|(&a, &b)| {
        let input1_val = i32::from(a) + input1_offset;
        let input2_val = i32::from(b) + input2_offset;
        let product = input1_val * input2_val;
        let scaled = tflm_math::multiply_by_quantized_multiplier(
            product, output_multiplier, output_shift);
        let out = scaled + output_offset;
        out.clamp(activation_min, activation_max) as i8
    }).collect();

    w.write("elementwise_mul",
        &input1_shape, &input2_shape, &output_shape,
        &input1, &input2, &[],
        input1_offset, output_offset,
        activation_min, activation_max,
        &[output_multiplier], &[output_shift],
        &output,
        &[("input2_offset", input2_offset)],
    );
}

/// TFLM Sub: mirrors AddFunc with subtraction instead of addition.
///
/// Sub uses the same ArithmeticParams as Add (with left_shift + per-input
/// multiplier/shift), and subtracts scaled_input2 from scaled_input1.
pub fn generate_sub(w: &mut FixtureWriter) {
    let input1_shape = [1i32, 1, 1, 6];
    let input2_shape = [1i32, 1, 1, 6];
    let output_shape = [1i32, 1, 1, 6];

    let input1: Vec<i8> = vec![10, 5, 0, -3, -7, -10];
    let input2: Vec<i8> = vec![3, 7, 0, -1, 2, 5];

    let input1_offset: i32 = 0;
    let input2_offset: i32 = 0;
    let output_offset: i32 = 0;
    let left_shift: i32 = 0;

    // Per-input scale=1.0
    let (input1_multiplier, input1_shift) = tflm_math::quantize_multiplier(1.0);
    let (input2_multiplier, input2_shift) = tflm_math::quantize_multiplier(1.0);

    // Output scale=1.0
    let (output_multiplier, output_shift) = tflm_math::quantize_multiplier(1.0);

    let activation_min: i32 = -128;
    let activation_max: i32 = 127;

    let output: Vec<i8> = input1.iter().zip(input2.iter()).map(|(&a, &b)| {
        let input1_val = i32::from(a) + input1_offset;
        let input2_val = i32::from(b) + input2_offset;
        let shifted1 = input1_val * (1 << left_shift);
        let shifted2 = input2_val * (1 << left_shift);
        let scaled1 = tflm_math::multiply_by_quantized_multiplier(
            shifted1, input1_multiplier, input1_shift);
        let scaled2 = tflm_math::multiply_by_quantized_multiplier(
            shifted2, input2_multiplier, input2_shift);
        let raw_sub = scaled1 - scaled2;
        let raw_output = tflm_math::multiply_by_quantized_multiplier(
            raw_sub, output_multiplier, output_shift) + output_offset;
        raw_output.clamp(activation_min, activation_max) as i8
    }).collect();

    w.write("elementwise_sub",
        &input1_shape, &input2_shape, &output_shape,
        &input1, &input2, &[],
        input1_offset, output_offset,
        activation_min, activation_max,
        &[output_multiplier], &[output_shift],
        &output,
        &[("input2_offset", input2_offset),
          ("left_shift", left_shift),
          ("input1_multiplier", input1_multiplier),
          ("input1_shift", input1_shift),
          ("input2_multiplier", input2_multiplier),
          ("input2_shift", input2_shift)],
    );
}
