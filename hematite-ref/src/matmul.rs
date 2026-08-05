// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Batch matrix multiply (MatMul) — scalar reference kernel.
//!
//! Implements the TFLM int8 `BatchMatMul` reference path
//! (`reference_ops::BatchMatMul`, the `FullyConnectedParams` overload in
//! `tensorflow/lite/kernels/internal/reference/batch_matmul.h` at the pinned
//! SHA), which is what `tensorflow/lite/micro/kernels/batch_matmul.cc`
//! `EvalInt8` dispatches to.
//!
//! # Algorithm
//!
//! 1. `total = Σ_k (lhs[k] + input_offset) · (rhs[k] + weights_offset)` —
//!    i32 accumulate (optionally seeded with `bias[oc]`, which is a
//!    **zero array** for the reference BatchMatMul path — the
//!    FullyConnectedParams overload has no bias term, so the fixture emits
//!    zeros and the add is identity).
//! 2. `scaled = multiply_by_quantized_multiplier(total,
//!    output_multiplier, output_shift)` — per-tensor requantize.
//! 3. `scaled += output_offset`; clamp to `[quantized_activation_min, max]`.
//!
//! Mirrors `tools/generate_goldens/src/ops/matmul.rs` bit-for-bit.

use hematite_core::op_params::MatMulParams;
use hematite_core::KernelError;
use hematite_int8::{multiply_by_quantized_multiplier, saturating_cast};

/// Batch matrix multiply — scalar reference kernel.
///
/// Computes `output = adj_x(A) · adj_y(B)` where `A` is `input` (m×k or
/// k×m if `adj_x`) and `B` is `weights` (k×n or n×k if `adj_y`). The
/// accumulate is `(a + input_offset) · (b + weights_offset)` summed over
/// the inner dimension `k`, then per-tensor requantized and clamped.
///
/// `bias` is added to each output column as the accumulator seed (identity
/// for the reference path, whose fixture emits an all-zero bias array).
///
/// # Errors
///
/// * [`KernelError::ShapeMismatch`] if `input.len() != m·k`,
///   `weights.len() != k·n`, `bias.len() != n`, or
///   `output.len() != m·n` (using the adj-adjusted shapes).
pub fn matmul(
    input: &[i8],
    weights: &[i8],
    bias: &[i32],
    params: &MatMulParams,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    let m = params.m as usize;
    let n = params.n as usize;
    let k = params.k as usize;

    // Element counts are adj-invariant: A always has m·k entries
    // ([m, k] or its transpose), B always has k·n ([k, n] or its transpose).
    let a_len = m * k;
    let b_len = k * n;
    let out_len = m * n;
    if input.len() != a_len {
        return Err(KernelError::ShapeMismatch);
    }
    if weights.len() != b_len {
        return Err(KernelError::ShapeMismatch);
    }
    if bias.len() != n {
        return Err(KernelError::ShapeMismatch);
    }
    if output.len() != out_len {
        return Err(KernelError::ShapeMismatch);
    }

    for i in 0..m {
        for j in 0..n {
            let mut total: i32 = bias[j];

            for kk in 0..k {
                let a_val = if params.adj_x {
                    input[kk * m + i]
                } else {
                    input[i * k + kk]
                };
                let b_val = if params.adj_y {
                    weights[j * k + kk]
                } else {
                    weights[kk * n + j]
                };
                total +=
                    (i32::from(a_val) + params.input_offset) * (i32::from(b_val) + params.weights_offset);
            }

            // Per-tensor requantize + output offset + clamp.
            let scaled =
                multiply_by_quantized_multiplier(total, params.output_multiplier, params.output_shift);
            let with_offset = scaled + params.output_offset;

            let clamped = if with_offset > params.quantized_activation_max {
                params.quantized_activation_max
            } else if with_offset < params.quantized_activation_min {
                params.quantized_activation_min
            } else {
                with_offset
            };

            output[i * n + j] = saturating_cast(clamped);
        }
    }

    let _ = scratch; // unused by scalar reference path

    Ok(())
}
