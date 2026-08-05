//! Reduction golden fixtures — mean, sum, argmax, argmin, l2_norm.
//!
//! Each op's arithmetic is defined HERE (the oracle) and mirrored
//! identically in `hematite-ref/src/reductions.rs` (the kernel).
//! The golden test proves bit-exact equality.

use crate::fixture::FixtureWriter;
use crate::tflm_math;

/// Reduce mean over axis.
///
/// Arithmetic: i32 accumulate over reduction axis, divide by count
/// (round-half-away-from-zero), then per-channel requantize via
/// `multiply_by_quantized_multiplier`.
///
/// ## Divisor encoding
///
/// The output_multiplier/shift in this fixture encode ONLY the
/// requantize scale — they DO NOT include the 1/count factor.
/// The kernel divides by count BEFORE calling mbm, matching
/// the TFLM reference `Mean` int8 semantics (two-step:
/// round(acc/count), then mbm).
pub fn generate_mean(w: &mut FixtureWriter) {
    // Input [1, 2, 3, 2] → reduce axis=1 (height) → output [1, 1, 3, 2]
    let input_shape = [1i32, 2, 3, 2];
    let output_shape = [1i32, 1, 3, 2];

    // Non-degenerate: two channels, 2 rows, 3 cols
    // Channel 0: [[1, 2, 3], [4, 8, 9]] — mean over rows
    // Channel 1: [[10, 20, 30], [40, 50, 60]]
    let input: Vec<i8> = vec![
        1, 10, // h=0,w=0: ch0=1, ch1=10
        2, 20, // h=0,w=1
        3, 30, // h=0,w=2
        4, 40, // h=1,w=0: ch0=4, ch1=40
        8, 50, // h=1,w=1
        9, 60, // h=1,w=2
    ];

    let input_h = 2i32;
    let input_w = 3i32;
    let channels = 2i32;
    let count = input_h; // reduction over height, so count = 2

    let output_multiplier: i32 = 1i32 << 30; // scale = 0.5
    let output_shift: i32 = 0;
    let input_offset: i32 = 0;
    let output_offset: i32 = 0;
    let activation_min: i32 = -128;
    let activation_max: i32 = 127;

    let mut output: Vec<i8> = vec![0i8; 6]; // 1 × 3 × 2

    for w in 0..input_w {
        for c in 0..channels {
            let mut acc: i32 = 0;
            for h in 0..input_h {
                let idx = (h * input_w + w) as usize * channels as usize + c as usize;
                acc += i32::from(input[idx]);
            }
            // Divide by count (round-half-away-from-zero)
            let averaged = if count == 0 {
                0
            } else if acc > 0 {
                (acc + count / 2) / count
            } else {
                (acc - count / 2) / count
            };
            // Requantize
            let scaled = tflm_math::multiply_by_quantized_multiplier(averaged, output_multiplier, output_shift);
            let val = (scaled + output_offset).max(activation_min).min(activation_max);
            let out_idx = (w * channels + c) as usize;
            output[out_idx] = val as i8;
        }
    }

    // Extra params: axis this fixture reduces over (axis=1)
    let extra_params: &[(&str, i32)] = &[
        ("AXIS_0", 1),
        ("AXIS_COUNT", 1),
    ];

    w.write("mean",
        &input_shape, &[0; 4], &output_shape,
        &input, &[], &[],
        input_offset, output_offset, activation_min, activation_max,
        &[output_multiplier], &[output_shift],
        &output,
        extra_params,
    );
}

/// Reduce sum over axis.
///
/// Arithmetic: i32 accumulate over reduction axis, then requantize
/// (no division — sum is just the accumulated total).
/// Output type == input type (int8), so it requantizes per TFLite SUM semantics.
pub fn generate_sum(w: &mut FixtureWriter) {
    // Input [1, 2, 2, 1] → reduce axis=1 → output [1, 1, 2, 1]
    let input_shape = [1i32, 2, 2, 1];
    let output_shape = [1i32, 1, 2, 1];

    let input: Vec<i8> = vec![
        1, 2,   // h=0: [1, 2]
        10, 20, // h=1: [10, 20]
    ];

    let input_h = 2i32;
    let input_w = 2i32;
    let channels = 1i32;

    let output_multiplier: i32 = 1i32 << 30; // scale = 0.5
    let output_shift: i32 = 0;
    let input_offset: i32 = 0;
    let output_offset: i32 = 0;
    let activation_min: i32 = -128;
    let activation_max: i32 = 127;

    let mut output: Vec<i8> = vec![0i8; 2]; // 1 × 1 × 2 × 1

    for w in 0..input_w {
        let mut acc: i32 = 0;
        for h in 0..input_h {
            let idx = (h * input_w + w) as usize * channels as usize;
            acc += i32::from(input[idx]);
        }
        let scaled = tflm_math::multiply_by_quantized_multiplier(acc, output_multiplier, output_shift);
        let val = (scaled + output_offset).max(activation_min).min(activation_max);
        output[w as usize] = val as i8;
    }

    let extra_params: &[(&str, i32)] = &[
        ("AXIS_0", 1),
        ("AXIS_COUNT", 1),
    ];

    w.write("sum",
        &input_shape, &[0; 4], &output_shape,
        &input, &[], &[],
        input_offset, output_offset, activation_min, activation_max,
        &[output_multiplier], &[output_shift],
        &output,
        extra_params,
    );
}

/// ArgMax over axis.
///
/// Pure int comparison — no quantization. Output is i8 indices.
/// Ties → first occurrence (TFLite semantics).
pub fn generate_argmax(w: &mut FixtureWriter) {
    // Input [1, 2, 3, 1] → reduce axis=1 → output [1, 1, 3, 1]
    // Find which row has the max per column
    let input_shape = [1i32, 2, 3, 1];
    let output_shape = [1i32, 1, 3, 1];

    // Row 0: [5, 1, 7], Row 1: [3, 9, 2]
    let input: Vec<i8> = vec![
        5, 1, 7,  // h=0: w=0→5, w=1→1, w=2→7
        3, 9, 2,  // h=1: w=0→3, w=1→9, w=2→2
    ];

    let input_w = 3i32;

    let mut output: Vec<i8> = vec![0i8; 3];

    for w in 0..input_w {
        let val0 = input[w as usize];
        let val1 = input[(input_w + w) as usize];
        // TFLite: first occurrence on tie
        if val0 >= val1 {
            output[w as usize] = 0i8;
        } else {
            output[w as usize] = 1i8;
        }
    }

    w.write_simple("argmax", &input_shape, &output_shape, &input, &output,
        &[("AXIS_0", 1), ("AXIS_COUNT", 1)],
        "// ArgMax: find index of maximum along axis 1");
}

/// ArgMin over axis.
///
/// Pure int comparison — no quantization. Output is i8 indices.
/// Ties → first occurrence (TFLite semantics).
pub fn generate_argmin(w: &mut FixtureWriter) {
    // Same input as argmax
    let input_shape = [1i32, 2, 3, 1];
    let output_shape = [1i32, 1, 3, 1];

    let input: Vec<i8> = vec![
        5, 1, 7,
        3, 9, 2,
    ];

    let input_w = 3i32;

    let mut output: Vec<i8> = vec![0i8; 3];

    for w in 0..input_w {
        let val0 = input[w as usize];
        let val1 = input[(input_w + w) as usize];
        if val0 <= val1 {
            output[w as usize] = 0i8;
        } else {
            output[w as usize] = 1i8;
        }
    }

    w.write_simple("argmin", &input_shape, &output_shape, &input, &output,
        &[("AXIS_0", 1), ("AXIS_COUNT", 1)],
        "// ArgMin: find index of minimum along axis 1");
}

/// L2 normalization along the last axis (channel dimension).
///
/// Arithmetic:
/// 1. For each spatial position: accumulate squared values in i32
///    (overflow guard: input values are i8, so max abs = 128, max
///    squared per element = 16384; with up to 1024 channels, max
///    i32 accumulation = 16384 × 1024 = 16,777,216, well within
///    i32::MAX = 2,147,483,647 — no saturation needed for practical
///    channel counts).
/// 2. `integer_sqrt(squared_sum) → u32` to get the norm.
/// 3. Per channel: `multiply_by_quantized_multiplier(input, output_multiplier, output_shift)`
///    divided by the norm via integer arithmetic.
///
/// The output_multiplier/shift encode the OUTPUT scale (2.0 here),
/// and the kernel divides the result by the norm at each position.
pub fn generate_l2_norm(w: &mut FixtureWriter) {
    // Input [1, 1, 2, 2] — two spatial positions, 2 channels each
    // Position 0: channels [3, 4], norm = sqrt(9+16) = 5
    // Position 1: channels [6, 8], norm = sqrt(36+64) = 10
    let input_shape = [1i32, 1, 2, 2];
    let output_shape = [1i32, 1, 2, 2];

    let input: Vec<i8> = vec![
        3, 4,  // h=0,w=0: ch0=3, ch1=4
        6, 8,  // h=0,w=1: ch0=6, ch1=8
    ];

    let channels = 2i32;
    let num_positions = 2i32;

    // Output scale = 2.0 → quantize_multiplier(2.0) = (1<<30, 2)
    let (output_multiplier, output_shift) = tflm_math::quantize_multiplier(2.0);
    let input_offset: i32 = 0;
    let output_offset: i32 = 0;
    let activation_min: i32 = -128;
    let activation_max: i32 = 127;

    let mut output: Vec<i8> = vec![0i8; 4]; // 1 × 2 × 2

    for pos in 0..num_positions {
        // Accumulate squared sum
        let base = (pos * channels) as usize;
        let sq0: i64 = i64::from(input[base]) * i64::from(input[base]);
        let sq1: i64 = i64::from(input[base + 1]) * i64::from(input[base + 1]);
        let sq_sum: u64 = (sq0 + sq1) as u64; // squares are non-negative

        let norm = tflm_math::integer_sqrt(sq_sum);

        for c in 0..channels {
            let inp = i32::from(input[base + c as usize]);
            // output = mbm(inp, output_mult, output_shift) * scale_after / norm
            // but to stay in pure integer: mbm(inp * K, output_mult, output_shift) / norm
            // where K is a scaling constant. Or equivalently:
            // result = round(inp * output_mult * 2^output_shift / 2^31 / norm * scale_after)
            // Use: temp = multiply_by_quantized_multiplier(inp, output_multiplier, output_shift)
            //       result = round(temp * scale_to_norm / norm)
            // We'll encode scale_to_norm as a fixed multiplier.
            //
            // Simplest integer path:
            //   scaled = mbm(inp, output_multiplier, output_shift)  (i32)
            //   ratio = (scaled << 15) / norm                       (fixed-point Q15.16)
            // But division loses precision.
            //
            // Better: compute the output directly using the SAME arithmetic
            // the kernel will use — i32 accumulate, integer_sqrt, then
            // output = saturating_cast(mbm(inp * output_mul, scaled_before_norm_mul, shift))
            //
            // Since the kernel MUST mirror this, let's use:
            // output = (inp * output_multiplier_l2 * K) / (norm * 2^something)
            // encoded as a single mbm call with a derived multiplier.

            // Compute: result = round(inp / norm * output_scale)
            //   = round(inp * output_scale / norm)
            //   = mbm(inp, output_multiplier_for_position, output_shift_for_position)
            // where output_multiplier_for_position = quantize_multiplier(output_scale / norm)

            // Since quantize_multiplier uses f64 and we want pure int,
            // compute output directly:
            //   numerator = inp * output_multiplier_l2 (where multiplier_l2 encodes output_scale)
            //   result = rounding_divide_by_pot(rounding_half_sum(...), ...)
            //
            // Simplest correct approach: use the exact same two-step
            // arithmetic the kernel will use.
            //
            // Step A: mbm(inp, output_mult, output_shift) → scaled (i32)
            // Step B: round(scaled / norm) → use rounding_divide_by_pot for power-of-2 norms,
            //         else manual rounding division.
            // Since our norms (5, 10) are NOT powers of 2, we use:
            //   result = (scaled + norm/2) / norm  (for scaled >= 0)
            //   result = (scaled - norm/2) / norm  (for scaled < 0)

            let scaled = tflm_math::multiply_by_quantized_multiplier(inp, output_multiplier, output_shift);
            let n = i32::try_from(norm).unwrap_or(i32::MAX);
            let result = if scaled >= 0 {
                (scaled + n / 2) / n
            } else {
                (scaled - n / 2) / n
            };
            let val = (result + output_offset).max(activation_min).min(activation_max);
            output[base + c as usize] = val as i8;
        }
    }

    let extra_params: &[(&str, i32)] = &[
        ("AXIS_0", 3), // reduce last axis (channel dimension = 3 in NHWC)
        ("AXIS_COUNT", 1),
    ];

    w.write("l2_norm",
        &input_shape, &[0; 4], &output_shape,
        &input, &[], &[],
        input_offset, output_offset, activation_min, activation_max,
        &[output_multiplier], &[output_shift],
        &output,
        extra_params,
    );
}

/// ReduceMax over axis.
///
/// Pure int8 comparison — no quantization, output scale/zp must equal input
/// (TFLM `EvalMinMaxHelper` `TF_LITE_ENSURE_EQ` on scale and zp).
/// Initial value = `lowest()` (-128); compare `(in > current) ? in : current`.
/// Mirrors `MinMaxReducerCompare<int8_t>` in
/// `tensorflow/lite/micro/kernels/reduce_common.cc` at the pinned SHA.
pub fn generate_reduce_max(w: &mut FixtureWriter) {
    let input_shape = [1i32, 2, 3, 1];
    let output_shape = [1i32, 1, 3, 1];

    let input: Vec<i8> = vec![
        5, 1, 7,
        -3, 9, -2,
    ];

    let input_w = 3i32;

    let mut output: Vec<i8> = vec![0i8; 3];

    for w_i in 0..input_w {
        let idx = w_i as usize;
        let mut current: i8 = i8::MIN;
        for h in 0..2 {
            let v = input[(h * input_w + w_i) as usize];
            if v > current {
                current = v;
            }
        }
        output[idx] = current;
    }

    w.write_simple("reduce_max", &input_shape, &output_shape, &input, &output,
        &[("AXIS_0", 1), ("AXIS_COUNT", 1)],
        "// ReduceMax: max of elements along axis 1 (pure int8 comparison, no requantize)");
}

/// ReduceMin over axis.
///
/// Pure int8 comparison — no quantization. Initial value = `max()` (127);
/// compare `(in < current) ? in : current`. Mirrors
/// `MinMaxReducerCompare<int8_t>` in `reduce_common.cc` at the pinned SHA.
pub fn generate_reduce_min(w: &mut FixtureWriter) {
    let input_shape = [1i32, 2, 3, 1];
    let output_shape = [1i32, 1, 3, 1];

    let input: Vec<i8> = vec![
        5, 1, 7,
        -3, 9, -2,
    ];

    let input_w = 3i32;

    let mut output: Vec<i8> = vec![0i8; 3];

    for w_i in 0..input_w {
        let idx = w_i as usize;
        let mut current: i8 = i8::MAX;
        for h in 0..2 {
            let v = input[(h * input_w + w_i) as usize];
            if v < current {
                current = v;
            }
        }
        output[idx] = current;
    }

    w.write_simple("reduce_min", &input_shape, &output_shape, &input, &output,
        &[("AXIS_0", 1), ("AXIS_COUNT", 1)],
        "// ReduceMin: min of elements along axis 1 (pure int8 comparison, no requantize)");
}
