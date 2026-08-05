//! MatMul golden fixture — TFLM `reference_ops::BatchMatMul` int8 path.
//!
//! Mirrors `tensorflow/lite/kernels/internal/reference/batch_matmul.h`
//! (the `FullyConnectedParams` overload at the pinned SHA), which is what
//! `tensorflow/lite/micro/kernels/batch_matmul.cc` `EvalInt8` dispatches to.

use crate::fixture::FixtureWriter;
use crate::tflm_math;

/// MatMul (int8): `output[i][j] = Σ_k (A[i][k] + input_offset) · (B[k][j] + weights_offset)`
///
/// Arithmetic (pinned SHA `tensorflow/lite/kernels/internal/reference/batch_matmul.h`,
/// FullyConnectedParams overload, lines 211–284):
/// 1. `total = Σ_k (lhs_val + filter_offset) · (rhs_val + input_offset)` — i32 accumulate.
/// 2. `total_scaled = MultiplyByQuantizedMultiplier(total, output_multiplier, output_shift)`
///    — per-tensor requantize.
/// 3. `total_scaled += output_offset`; clamp to `[output_activation_min, max]`.
///
/// The micro kernel passes (rhs, lhs) so the "weights" tensor (second input, B)
/// receives `weights_offset` and the "activations" tensor (first input, A)
/// receives `input_offset`. TFLM's int8 BatchMatMul path has **no bias term**
/// (the FullyConnectedParams overload has no bias argument), so the fixture
/// emits a zero bias array; the kernel must add it as identity.
///
/// Shape: A = [2, 4], B = [4, 3] → output [2, 3]. adj_x = false, adj_y = false
/// (no transposition; both tensors row-major in the natural multiply order).
pub fn generate_matmul(w: &mut FixtureWriter) {
    // A (2×4), B (4×3). Hand-verifiable small integers; negatives exercise
    // the single-rounding path with signed accumulators.
    let a: Vec<i8> = vec![-8, -7, -6, -5, 1, 2, 3, 4];
    let b: Vec<i8> = vec![
        1, 2, -1,
        -1, 1, 2,
        2, -1, 1,
        1, 1, -1,
    ];

    let m = 2i32;
    let k = 4i32;
    let n = 3i32;

    let input_offset: i32 = 0;
    let weights_offset: i32 = 0;
    let output_offset: i32 = 0;
    let activation_min: i32 = -128;
    let activation_max: i32 = 127;
    let (output_multiplier, output_shift) = tflm_math::quantize_multiplier(0.5);

    let mut output: Vec<i8> = vec![0i8; (m * n) as usize];

    for i in 0..m {
        for j in 0..n {
            let mut total: i32 = 0;
            for kk in 0..k {
                let a_val = i32::from(a[(i * k + kk) as usize]);
                let b_val = i32::from(b[(kk * n + j) as usize]);
                total += (a_val + input_offset) * (b_val + weights_offset);
            }
            let scaled =
                tflm_math::multiply_by_quantized_multiplier(total, output_multiplier, output_shift);
            let val = (scaled + output_offset).max(activation_min).min(activation_max);
            output[(i * n + j) as usize] = val as i8;
        }
    }

    // Zero bias: TFLM int8 BatchMatMul has no bias term in the reference path.
    let bias: Vec<i32> = vec![0, 0, 0];

    let extra_params: &[(&str, i32)] = &[
        ("M", m),
        ("N", n),
        ("K", k),
        ("ADJ_X", 0),
        ("ADJ_Y", 0),
        ("WEIGHTS_OFFSET", weights_offset),
    ];

    let input_shape = [1i32, 1, 2, 4];
    let filter_shape = [1i32, 1, 4, 3];
    let output_shape = [1i32, 1, 2, 3];

    w.write("matmul",
        &input_shape, &filter_shape, &output_shape,
        &a, &b, &bias,
        input_offset, output_offset, activation_min, activation_max,
        &[output_multiplier], &[output_shift],
        &output,
        extra_params,
    );
}
