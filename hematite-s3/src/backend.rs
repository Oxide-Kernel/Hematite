// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! [`S3Backend`] — the ESP32-S3 optimized backend.
//!
//! A thin adapter that implements [`KernelBackend`] by forwarding every
//! compute-heavy trait method to the bespoke TIE728 SIMD kernels in this
//! crate (conv, depthwise, gemm, softmax, pooling, activations, elementwise),
//! and every other method to the scalar reference kernels in `hematite-ref`.
//!
//! The s3 kernels auto-dispatch to SIMD on real hardware
//! (`cfg(target_arch = "xtensa")` without the `qemu` feature) and fall back
//! to an internal scalar path otherwise, so this adapter produces the fastest
//! available execution on every target while staying bit-exact with the
//! reference.
//!
//! # Scratch contract
//!
//! The SIMD conv/depthwise/gemm paths stage padded input/weight copies and
//! i32 accumulators in `scratch`; if the provided scratch is too small they
//! silently fall back to the scalar path (output stays correct, just slower).
//! [`KernelBackend::conv2d_scratch_size`] / [`depthwise_conv2d_scratch_size`]
//! / [`softmax_scratch_size`] return the exact s3 requirements so callers can
//! size their scratch; callers that always pass a large (>= 32 KiB),
//! 16-byte-aligned scratch buffer are guaranteed the SIMD path.
//!
//! # Ops NOT wired (return [`KernelError::Unsupported`])
//!
//! `unidirectional_sequence_lstm` / `svdf` / `gru` — same trait-signature gap
//! as [`RefBackend`](hematite_ref::RefBackend::unidirectional_sequence_lstm):
//! the scalar recurrent kernels require fixture-specific quant constants not
//! carried by the params structs. None of the zoo models use these.

use hematite_core::op_params::{
    ActivationParams, ConcatParams, Conv2DParams, DepthwiseConv2DParams,
    ElementwiseParams, FullyConnectedParams, GruParams, LstmParams,
    MatMulParams, PadParams, PoolParams, QuantParam, ReduceParams,
    ReshapeParams, ResizeNearestParams, SliceParams, SoftmaxParams,
    SplitParams, SvdfParams, TransposeParams,
};
use hematite_core::{KernelBackend, KernelError};

use crate::activations;
use crate::conv1x1;
use crate::conv3x3;
use crate::depthwise;
use crate::elementwise;
use crate::gemm;
use crate::pool;
use crate::reductions;
use crate::softmax;

use hematite_ref::activation;
use hematite_ref::activation_ext;
use hematite_ref::conv;
use hematite_ref::data_movement;
use hematite_ref::matmul;
use hematite_ref::resize;

/// The ESP32-S3 optimized backend.
///
/// Stateless unit struct — all kernels are pure functions of their slices
/// and params. Const-constructible: `let backend = S3Backend;`.
pub struct S3Backend;

/// Product of a `[i32; 4]` shape array.
#[inline(always)]
fn shape_product(shape: &[i32; 4]) -> usize {
    shape[0] as usize * shape[1] as usize * shape[2] as usize * shape[3] as usize
}

/// Compute the NHWC row-major strides of a `[i32; 4]` shape.
#[inline(always)]
fn nhwc_strides(shape: &[i32; 4]) -> [usize; 4] {
    [
        shape[1] as usize * shape[2] as usize * shape[3] as usize,
        shape[2] as usize * shape[3] as usize,
        shape[3] as usize,
        1,
    ]
}

/// Decode a flat NHWC linear index into 4D coordinates.
#[inline(always)]
fn decode_4d(idx: usize, shape: &[i32; 4], strides: &[usize; 4]) -> [usize; 4] {
    [
        (idx / strides[0]) % shape[0] as usize,
        (idx / strides[1]) % shape[1] as usize,
        (idx / strides[2]) % shape[2] as usize,
        (idx / strides[3]) % shape[3] as usize,
    ]
}

/// Encode 4D coordinates into a flat NHWC linear index.
#[inline(always)]
fn encode_4d(coords: [usize; 4], strides: &[usize; 4]) -> usize {
    coords[0] * strides[0] + coords[1] * strides[1] + coords[2] * strides[2] + coords[3]
}

/// General 4D transpose (scatter formulation) — identical to
/// `hematite-ref/src/backend.rs::transpose_impl`.
#[inline]
fn transpose_impl(
    input: &[i8],
    params: &TransposeParams,
    output: &mut [i8],
) -> Result<(), KernelError> {
    let in_shape = params.input_shape;
    let perm_count = usize::from(params.perm_count.max(0) as u8);

    // Effective permutation: identity for dims past perm_count.
    let perm: [usize; 4] = [
        if 0 < perm_count { params.perm[0] as usize } else { 0 },
        if 1 < perm_count { params.perm[1] as usize } else { 1 },
        if 2 < perm_count { params.perm[2] as usize } else { 2 },
        if 3 < perm_count { params.perm[3] as usize } else { 3 },
    ];
    for &p in &perm {
        if p >= 4 {
            return Err(KernelError::ShapeMismatch);
        }
    }

    let out_shape = [in_shape[perm[0]], in_shape[perm[1]], in_shape[perm[2]], in_shape[perm[3]]];

    if input.len() != shape_product(&in_shape) || output.len() != shape_product(&out_shape) {
        return Err(KernelError::ShapeMismatch);
    }

    let in_strides = nhwc_strides(&in_shape);
    let out_strides = nhwc_strides(&out_shape);

    for (idx, &val) in input.iter().enumerate() {
        let coords = decode_4d(idx, &in_shape, &in_strides);
        let out_coords = [coords[perm[0]], coords[perm[1]], coords[perm[2]], coords[perm[3]]];
        let out_idx = encode_4d(out_coords, &out_strides);
        output[out_idx] = val;
    }

    Ok(())
}

impl KernelBackend for S3Backend {
    // ── Tier0 — Core compute ────────────────────────────────────────────

    /// 1×1 filters route to the bespoke `conv2d_1x1` kernel; any other
    /// filter routes to `conv2d_3x3` (which handles arbitrary filter dims
    /// via bounds-check zero-pad). Both are bit-exact with the scalar conv;
    /// `Unsupported` (batch != 1) falls back to the reference kernel.
    fn conv2d(
        &self,
        input: &[i8],
        weights: &[i8],
        bias: &[i32],
        params: &Conv2DParams,
        output: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        let is_1x1 = params.filter_shape[1] == 1 && params.filter_shape[2] == 1;
        let result = if is_1x1 {
            conv1x1::conv2d_1x1(input, weights, bias, params, output, scratch)
        } else {
            conv3x3::conv2d_3x3(input, weights, bias, params, output, scratch)
        };
        match result {
            Ok(()) => Ok(()),
            Err(KernelError::Unsupported) => {
                conv::conv2d(input, weights, bias, params, output, scratch)
            }
            Err(e) => Err(e),
        }
    }

    fn depthwise_conv2d(
        &self,
        input: &[i8],
        weights: &[i8],
        bias: &[i32],
        params: &DepthwiseConv2DParams,
        output: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        match depthwise::depthwise_conv2d(input, weights, bias, params, output, scratch) {
            Ok(()) => Ok(()),
            Err(KernelError::Unsupported) => {
                hematite_ref::depthwise_conv::depthwise_conv2d(
                    input, weights, bias, params, output, scratch,
                )
            }
            Err(e) => Err(e),
        }
    }

    fn fully_connected(
        &self,
        input: &[i8],
        weights: &[i8],
        bias: &[i32],
        params: &FullyConnectedParams,
        output: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        match gemm::fully_connected(input, weights, bias, params, output, scratch) {
            Ok(()) => Ok(()),
            Err(KernelError::Unsupported) => {
                hematite_ref::fully_connected::fully_connected(
                    input, weights, bias, params, output, scratch,
                )
            }
            Err(e) => Err(e),
        }
    }

    /// Forwarded to the reference BatchMatMul kernel (TFLM int8 reference
    /// path, per-tensor requantize).
    fn matmul(
        &self,
        input: &[i8],
        weights: &[i8],
        bias: &[i32],
        params: &MatMulParams,
        output: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        matmul::matmul(input, weights, bias, params, output, scratch)
    }

    // ── Tier1 — Pooling ─────────────────────────────────────────────────

    fn average_pool_2d(
        &self,
        input: &[i8],
        params: &PoolParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        pool::average_pool_2d(input, params, output, &mut [])
    }

    fn max_pool_2d(
        &self,
        input: &[i8],
        params: &PoolParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        pool::max_pool_2d(input, params, output, &mut [])
    }

    // ── Tier1 — Softmax ─────────────────────────────────────────────────

    fn softmax(
        &self,
        input: &[i8],
        params: &SoftmaxParams,
        output: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        softmax::softmax(input, params, output, scratch)
    }

    // ── Tier1 — Standalone activations ──────────────────────────────────

    fn relu(
        &self,
        input: &[i8],
        params: &ActivationParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        activations::relu(input, params, output, &mut [])
    }

    /// Forwards `params.quantized_activation_max` as the ReLU6 clamp bound —
    /// same trait-signature adaptation as the reference backend.
    fn relu6(
        &self,
        input: &[i8],
        params: &ActivationParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        activations::relu6(input, params, output, &mut [], params.quantized_activation_max)
    }

    fn hard_swish(
        &self,
        input: &[i8],
        params: &ActivationParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        activations::hard_swish(input, params, output, &mut [])
    }

    fn sigmoid(
        &self,
        input: &[i8],
        params: &ActivationParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        activation_ext::sigmoid(input, params, output)
    }

    fn tanh(
        &self,
        input: &[i8],
        params: &ActivationParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        activation_ext::tanh(input, params, output)
    }

    fn leaky_relu(
        &self,
        input: &[i8],
        params: &ActivationParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        activation::leaky_relu(input, params, output, &mut [])
    }

    fn prelu(
        &self,
        input: &[i8],
        params: &ActivationParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        activation::prelu(input, params, output, &mut [])
    }

    // ── Tier1 — Elementwise ─────────────────────────────────────────────

    fn add(
        &self,
        input1: &[i8],
        input2: &[i8],
        params: &ElementwiseParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        match elementwise::add(input1, input2, params, output, &mut []) {
            Ok(()) => Ok(()),
            Err(e) => Err(e),
        }
    }

    fn mul(
        &self,
        input1: &[i8],
        input2: &[i8],
        params: &ElementwiseParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        match elementwise::mul(input1, input2, params, output, &mut []) {
            Ok(()) => Ok(()),
            Err(e) => Err(e),
        }
    }

    fn sub(
        &self,
        input1: &[i8],
        input2: &[i8],
        params: &ElementwiseParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        match elementwise::sub(input1, input2, params, output, &mut []) {
            Ok(()) => Ok(()),
            Err(e) => Err(e),
        }
    }

    // ── Tier1 — Quantize / Dequantize ───────────────────────────────────

    fn quantize(
        &self,
        input: &[i8],
        params: &QuantParam,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        hematite_ref::elementwise::quantize(input, params, output, &mut [])
    }

    fn dequantize(
        &self,
        input: &[i8],
        params: &QuantParam,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        hematite_ref::elementwise::dequantize(input, params, output, &mut [])
    }

    // ── Tier2 — Data movement ───────────────────────────────────────────

    /// Reshape is a flat copy — same inline implementation as the reference.
    fn reshape(
        &self,
        input: &[i8],
        params: &ReshapeParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        let _ = params;
        if input.len() != output.len() {
            return Err(KernelError::ShapeMismatch);
        }
        output.copy_from_slice(input);
        Ok(())
    }

    fn transpose(
        &self,
        input: &[i8],
        params: &TransposeParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        transpose_impl(input, params, output)
    }

    fn concat(
        &self,
        input_a: &[i8],
        input_b: &[i8],
        params: &ConcatParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        data_movement::concat_op(input_a, input_b, params, output, &mut [])
    }

    fn split(
        &self,
        input: &[i8],
        params: &SplitParams,
        output_a: &mut [i8],
        output_b: &mut [i8],
    ) -> Result<(), KernelError> {
        data_movement::split_op(input, 0, params, output_a, &mut [])?;
        data_movement::split_op(input, 1, params, output_b, &mut [])
    }

    fn pad(
        &self,
        input: &[i8],
        params: &PadParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        data_movement::pad_op(input, params, output, &mut [])
    }

    fn slice(
        &self,
        input: &[i8],
        params: &SliceParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        data_movement::slice_op(input, params, output, &mut [])
    }

    fn resize_nearest(
        &self,
        input: &[i8],
        params: &ResizeNearestParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        resize::resize_nearest_neighbor(input, params, output, &mut [])
    }

    // ── Tier3 — Recurrent ───────────────────────────────────────────────

    /// Not wired: same trait-signature gap as the reference backend (the
    /// scalar recurrent kernels need fixture-specific quant constants).
    #[allow(clippy::too_many_arguments)]
    fn unidirectional_sequence_lstm(
        &self,
        _input: &[i8],
        _input_to_input_weights: &[i8],
        _input_to_forget_weights: &[i8],
        _input_to_cell_weights: &[i8],
        _input_to_output_weights: &[i8],
        _recurrent_to_input_weights: &[i8],
        _recurrent_to_forget_weights: &[i8],
        _recurrent_to_cell_weights: &[i8],
        _recurrent_to_output_weights: &[i8],
        _input_gate_bias: &[i32],
        _forget_gate_bias: &[i32],
        _cell_bias: &[i32],
        _output_gate_bias: &[i32],
        _params: &LstmParams,
        _output: &mut [i8],
        _cell_state: &mut [i16],
        _hidden_state: &mut [i8],
        _scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported)
    }

    #[allow(clippy::too_many_arguments)]
    fn svdf(
        &self,
        _input: &[i8],
        _weights_feature: &[i8],
        _weights_time: &[i16],
        _bias: &[i32],
        _params: &SvdfParams,
        _output: &mut [i8],
        _hidden_state: &mut [i16],
        _scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported)
    }

    #[allow(clippy::too_many_arguments)]
    fn gru(
        &self,
        _input: &[i8],
        _reset_gate_weights: &[i8],
        _update_gate_weights: &[i8],
        _candidate_weights: &[i8],
        _biases: &[i32],
        _params: &GruParams,
        _output: &mut [i8],
        _hidden_state: &mut [i16],
        _scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported)
    }

    // ── Tier4 — Reductions ──────────────────────────────────────────────

    fn mean(
        &self,
        input: &[i8],
        params: &ReduceParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        reductions::mean(input, params, output)
    }

    fn sum(
        &self,
        input: &[i8],
        params: &ReduceParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        hematite_ref::reductions::sum(input, params, output)
    }

    fn reduce_max(
        &self,
        input: &[i8],
        params: &ReduceParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        hematite_ref::reductions::reduce_max(input, params, output)
    }

    fn reduce_min(
        &self,
        input: &[i8],
        params: &ReduceParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        hematite_ref::reductions::reduce_min(input, params, output)
    }

    fn arg_max(
        &self,
        input: &[i8],
        params: &ReduceParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        hematite_ref::reductions::arg_max(input, params, output)
    }

    fn arg_min(
        &self,
        input: &[i8],
        params: &ReduceParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        hematite_ref::reductions::arg_min(input, params, output)
    }

    fn l2_normalization(
        &self,
        input: &[i8],
        params: &ReduceParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        hematite_ref::reductions::l2_norm(input, params, output)
    }

    // ── Scratch-size associated functions ───────────────────────────────

    /// Exact s3 SIMD scratch need for a conv. Mirrors the `need` computed in
    /// `conv1x1_accx_dispatch` / `conv3x3_accx_dispatch`.
    fn conv2d_scratch_size(params: &Conv2DParams) -> usize {
        let in_h = params.input_shape[1] as usize;
        let in_w = params.input_shape[2] as usize;
        let in_c = params.input_shape[3] as usize;
        let out_c = params.output_shape[3] as usize;
        let filter_h = params.filter_shape[1] as usize;
        let filter_w = params.filter_shape[2] as usize;
        let wsum_extra = if params.input_offset != 0 { out_c * 4 } else { 0 };

        if filter_h == 1 && filter_w == 1 {
            return out_c * 4 + wsum_extra;
        }

        let dilated_filter_h = (filter_h - 1) * params.dilation_height_factor as usize + 1;
        let dilated_filter_w = (filter_w - 1) * params.dilation_width_factor as usize + 1;
        let pad_total_h = ((params.output_shape[1] as i32 - 1) * params.stride_height
            + dilated_filter_h as i32
            - in_h as i32)
            .max(0) as usize;
        let pad_total_w = ((params.output_shape[2] as i32 - 1) * params.stride_width
            + dilated_filter_w as i32
            - in_w as i32)
            .max(0) as usize;
        let padded_c = ((in_c + 15) / 16) * 16;
        let padded_h = in_h + pad_total_h;
        let padded_w = in_w + pad_total_w;
        let needs_pad = pad_total_h > 0 || pad_total_w > 0 || padded_c != in_c;
        if needs_pad {
            padded_h * padded_w * padded_c + out_c * 9 * padded_c + out_c * 4 + wsum_extra
        } else {
            out_c * 4 + wsum_extra
        }
    }

    /// Exact s3 SIMD scratch need for a depthwise conv. Mirrors the `need`
    /// computed in `depthwise_accx_dispatch`.
    fn depthwise_conv2d_scratch_size(params: &DepthwiseConv2DParams) -> usize {
        let in_h = params.input_shape[1] as usize;
        let in_w = params.input_shape[2] as usize;
        let in_c = params.input_shape[3] as usize;
        let out_c = params.output_shape[3] as usize;
        let filter_h = params.filter_shape[1] as usize;
        let filter_w = params.filter_shape[2] as usize;
        let out_h = params.output_shape[1] as usize;
        let out_w = params.output_shape[2] as usize;
        let wsum_extra = if params.input_offset != 0 { out_c * 4 } else { 0 };

        let dilated_filter_h = (filter_h - 1) * params.dilation_height_factor as usize + 1;
        let dilated_filter_w = (filter_w - 1) * params.dilation_width_factor as usize + 1;
        let pad_total_h = ((out_h as i32 - 1) * params.stride_height
            + dilated_filter_h as i32
            - in_h as i32)
            .max(0) as usize;
        let pad_total_w = ((out_w as i32 - 1) * params.stride_width
            + dilated_filter_w as i32
            - in_w as i32)
            .max(0) as usize;
        let padded_c = ((in_c + 15) / 16) * 16;
        let needs_channel_pad = padded_c != in_c;
        let needs_pad = pad_total_h > 0 || pad_total_w > 0 || needs_channel_pad;
        if needs_pad {
            let padded_h = in_h + pad_total_h;
            let padded_w = in_w + pad_total_w;
            let pad_filter_len = if needs_channel_pad { 9 * padded_c } else { 0 };
            padded_h * padded_w * padded_c + pad_filter_len + padded_c * 4 + wsum_extra
        } else {
            out_c * 4 + wsum_extra
        }
    }

    /// s3 softmax SIMD needs `row_size * 4` bytes (i32 exp cache per row),
    /// 4-byte aligned.
    fn softmax_scratch_size(params: &SoftmaxParams) -> usize {
        params.row_size as usize * 4
    }
}
