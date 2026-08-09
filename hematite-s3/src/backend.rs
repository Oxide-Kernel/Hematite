// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! [`S3Backend`] — the ESP32-S3 optimized backend.
//!
//! A thin adapter that implements [`KernelBackend`] by forwarding every
//! trait method to the standalone s3 kernels in this crate (see the per-op
//! modules: [`conv1x1`], [`conv3x3`], [`depthwise`], [`gemm`], [`pool`],
//! [`softmax`], [`activations`], [`elementwise`], [`reductions`]).
//!
//! SIMD dispatch happens INSIDE the free functions (gated
//! `#[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]` for the conv
//! family and `#[cfg(target_arch = "xtensa")]` for the legacy elementwise/
//! pool/relu dispatch), so this adapter is pure wiring: on real silicon the
//! forwarded calls take the ACCX/TIE728 SIMD path, on the host they take the
//! scalar fallback — output is bit-exact either way.
//!
//! # Ops NOT wired (return [`KernelError::Unsupported`])
//!
//! s3 has no kernel (scalar or SIMD) for these ops, and the plan (todo 3 of
//! `local-notes/plans/simd-zoo-hardening.md`) mandates `Unsupported` — honest
//! failure rather than a silent wrong answer:
//!
//! | Trait method | Reason |
//! |---|---|
//! | `matmul`, `sigmoid`, `tanh`, `leaky_relu`, `prelu`, `quantize`, `dequantize`, `reshape`, `transpose`, `concat`, `split`, `pad`, `slice`, `resize_nearest`, `unidirectional_sequence_lstm`, `svdf`, `gru`, `sum`, `reduce_max`, `reduce_min`, `arg_max`, `arg_min`, `l2_normalization` | No s3 kernel exists for the op. |
//!
//! # Scratch sizes
//!
//! The trait's `*_scratch_size` associated functions default to `0`; the s3
//! SIMD paths need real scratch (see per-kernel `need` formulas below), so
//! `conv2d`, `depthwise_conv2d`, and `softmax` override them. Short scratch
//! does NOT error in the s3 kernels — the ACCX/softmax dispatch returns
//! `Ok(false)` and falls through to the scalar path — so the overrides exist
//! to guarantee the SIMD path can actually engage (and to stay correct once
//! codegen starts consulting `B::*_scratch_size` at macro time).

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

/// The ESP32-S3 optimized backend.
///
/// Stateless unit struct — all kernels are pure functions of their slices
/// and params. Const-constructible: `let backend = S3Backend;`.
pub struct S3Backend;

/// Round a channel count up to the next multiple of 16.
#[inline(always)]
fn pad16(n: usize) -> usize {
    n.div_ceil(16) * 16
}

/// Scratch bytes needed by the 1×1 ACCX path (`conv1x1_accx_dispatch`):
/// `out_c * 4` i32 accumulators, plus an `out_c * 4` weight-sum buffer when
/// `input_offset != 0`.
#[inline(always)]
fn conv1x1_scratch_need(params: &Conv2DParams) -> usize {
    let out_c = params.output_shape[3].max(0) as usize;
    let wsum = if params.input_offset != 0 { out_c * 4 } else { 0 };
    out_c * 4 + wsum
}

/// Scratch bytes needed by the 3×3 ACCX path (`conv3x3_accx_dispatch`).
///
/// When the layer needs spatial padding or channel padding to a multiple of
/// 16, the kernel stages a padded input copy, a padded weight copy, and the
/// i32 accumulator buffer in scratch; otherwise only the accumulator buffer
/// (plus the weight-sum buffer when `input_offset != 0`) is needed.
#[inline(always)]
fn conv3x3_scratch_need(params: &Conv2DParams) -> usize {
    let in_h = params.input_shape[1].max(0) as usize;
    let in_w = params.input_shape[2].max(0) as usize;
    let in_c = params.input_shape[3].max(0) as usize;
    let out_h = params.output_shape[1].max(0) as usize;
    let out_w = params.output_shape[2].max(0) as usize;
    let out_c = params.output_shape[3].max(0) as usize;
    let filter_h = params.filter_shape[1];
    let filter_w = params.filter_shape[2];

    let dilated_filter_h = (filter_h - 1) * params.dilation_height_factor + 1;
    let dilated_filter_w = (filter_w - 1) * params.dilation_width_factor + 1;
    let pad_total_h =
        ((out_h as i32 - 1) * params.stride_height + dilated_filter_h - in_h as i32).max(0) as usize;
    let pad_total_w =
        ((out_w as i32 - 1) * params.stride_width + dilated_filter_w - in_w as i32).max(0) as usize;
    let padded_c = pad16(in_c);
    let needs_pad = pad_total_h > 0 || pad_total_w > 0 || padded_c != in_c;

    let wsum = if params.input_offset != 0 { out_c * 4 } else { 0 };
    if needs_pad {
        let pad_input_len = (in_h + pad_total_h) * (in_w + pad_total_w) * padded_c;
        let pad_weights_len = out_c * 9 * padded_c;
        pad_input_len + pad_weights_len + out_c * 4 + wsum
    } else {
        out_c * 4 + wsum
    }
}

/// Scratch bytes needed by the depthwise ACCX path
/// (`depthwise_accx_dispatch`): a padded input copy, a padded filter copy
/// (only when channel-padding), the accumulator buffer, and the optional
/// weight-sum buffer.
#[inline(always)]
fn depthwise_scratch_need(params: &DepthwiseConv2DParams) -> usize {
    let in_h = params.input_shape[1].max(0) as usize;
    let in_w = params.input_shape[2].max(0) as usize;
    let in_c = params.input_shape[3].max(0) as usize;
    let out_h = params.output_shape[1].max(0) as usize;
    let out_w = params.output_shape[2].max(0) as usize;
    let out_c = params.output_shape[3].max(0) as usize;
    let filter_h = params.filter_shape[1];
    let filter_w = params.filter_shape[2];

    let dilated_filter_h = (filter_h - 1) * params.dilation_height_factor + 1;
    let dilated_filter_w = (filter_w - 1) * params.dilation_width_factor + 1;
    let pad_total_h =
        ((out_h as i32 - 1) * params.stride_height + dilated_filter_h - in_h as i32).max(0) as usize;
    let pad_total_w =
        ((out_w as i32 - 1) * params.stride_width + dilated_filter_w - in_w as i32).max(0) as usize;
    let padded_c = pad16(in_c);
    let needs_channel_pad = padded_c != in_c;
    let needs_pad = pad_total_h > 0 || pad_total_w > 0 || needs_channel_pad;

    let wsum = if params.input_offset != 0 { out_c * 4 } else { 0 };
    if needs_pad {
        let pad_input_len = (in_h + pad_total_h) * (in_w + pad_total_w) * padded_c;
        let pad_filter_len = if needs_channel_pad { 9 * padded_c } else { 0 };
        pad_input_len + pad_filter_len + padded_c * 4 + wsum
    } else {
        out_c * 4 + wsum
    }
}

impl KernelBackend for S3Backend {
    // ── Tier0 — Core compute ────────────────────────────────────────────

    fn conv2d(
        &self,
        input: &[i8],
        weights: &[i8],
        bias: &[i32],
        params: &Conv2DParams,
        output: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        // A 1×1 filter (FH=1, FW=1) dispatches to the dedicated 1×1 kernel;
        // every other filter shape (including 3×3) goes to the general
        // conv3x3 module, whose kernel handles arbitrary filter dimensions.
        if params.filter_shape[1] == 1 && params.filter_shape[2] == 1 {
            conv1x1::conv2d_1x1(input, weights, bias, params, output, scratch)
        } else {
            conv3x3::conv2d_3x3(input, weights, bias, params, output, scratch)
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
        depthwise::depthwise_conv2d(input, weights, bias, params, output, scratch)
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
        gemm::fully_connected(input, weights, bias, params, output, scratch)
    }

    fn matmul(
        &self,
        _input: &[i8],
        _weights: &[i8],
        _bias: &[i32],
        _params: &MatMulParams,
        _output: &mut [i8],
        _scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported)
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
    /// the s3 kernel takes `quantized_six` as an extra parameter
    /// (`QUANTIZED_SIX` is not a field of [`ActivationParams`]), mirroring
    /// RefBackend's own adaptation.
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
        _input: &[i8],
        _params: &ActivationParams,
        _output: &mut [i8],
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported)
    }

    fn tanh(
        &self,
        _input: &[i8],
        _params: &ActivationParams,
        _output: &mut [i8],
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported)
    }

    fn leaky_relu(
        &self,
        _input: &[i8],
        _params: &ActivationParams,
        _output: &mut [i8],
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported)
    }

    fn prelu(
        &self,
        _input: &[i8],
        _params: &ActivationParams,
        _output: &mut [i8],
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported)
    }

    // ── Tier1 — Elementwise ─────────────────────────────────────────────

    fn add(
        &self,
        input1: &[i8],
        input2: &[i8],
        params: &ElementwiseParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        elementwise::add(input1, input2, params, output, &mut [])
    }

    fn mul(
        &self,
        input1: &[i8],
        input2: &[i8],
        params: &ElementwiseParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        elementwise::mul(input1, input2, params, output, &mut [])
    }

    fn sub(
        &self,
        input1: &[i8],
        input2: &[i8],
        params: &ElementwiseParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        elementwise::sub(input1, input2, params, output, &mut [])
    }

    // ── Tier1 — Quantize / Dequantize ───────────────────────────────────

    fn quantize(
        &self,
        _input: &[i8],
        _params: &QuantParam,
        _output: &mut [i8],
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported)
    }

    fn dequantize(
        &self,
        _input: &[i8],
        _params: &QuantParam,
        _output: &mut [i8],
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported)
    }

    // ── Tier2 — Data movement ───────────────────────────────────────────

    fn reshape(
        &self,
        _input: &[i8],
        _params: &ReshapeParams,
        _output: &mut [i8],
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported)
    }

    fn transpose(
        &self,
        _input: &[i8],
        _params: &TransposeParams,
        _output: &mut [i8],
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported)
    }

    fn concat(
        &self,
        _input_a: &[i8],
        _input_b: &[i8],
        _params: &ConcatParams,
        _output: &mut [i8],
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported)
    }

    fn split(
        &self,
        _input: &[i8],
        _params: &SplitParams,
        _output_a: &mut [i8],
        _output_b: &mut [i8],
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported)
    }

    fn pad(
        &self,
        _input: &[i8],
        _params: &PadParams,
        _output: &mut [i8],
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported)
    }

    fn slice(
        &self,
        _input: &[i8],
        _params: &SliceParams,
        _output: &mut [i8],
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported)
    }

    fn resize_nearest(
        &self,
        _input: &[i8],
        _params: &ResizeNearestParams,
        _output: &mut [i8],
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported)
    }

    // ── Tier3 — Recurrent ───────────────────────────────────────────────

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

    /// Scalar mean — the SIMD mean decision is todo 18 of the plan; the
    /// scalar `reductions::mean` is bit-exact vs the reference.
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
        _input: &[i8],
        _params: &ReduceParams,
        _output: &mut [i8],
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported)
    }

    fn reduce_max(
        &self,
        _input: &[i8],
        _params: &ReduceParams,
        _output: &mut [i8],
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported)
    }

    fn reduce_min(
        &self,
        _input: &[i8],
        _params: &ReduceParams,
        _output: &mut [i8],
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported)
    }

    fn arg_max(
        &self,
        _input: &[i8],
        _params: &ReduceParams,
        _output: &mut [i8],
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported)
    }

    fn arg_min(
        &self,
        _input: &[i8],
        _params: &ReduceParams,
        _output: &mut [i8],
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported)
    }

    fn l2_normalization(
        &self,
        _input: &[i8],
        _params: &ReduceParams,
        _output: &mut [i8],
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported)
    }

    // ── Scratch-size associated functions ───────────────────────────────

    /// Dispatches on the same 1×1-vs-general filter test as [`conv2d`](Self::conv2d).
    fn conv2d_scratch_size(params: &Conv2DParams) -> usize {
        if params.filter_shape[1] == 1 && params.filter_shape[2] == 1 {
            conv1x1_scratch_need(params)
        } else {
            conv3x3_scratch_need(params)
        }
    }

    fn depthwise_conv2d_scratch_size(params: &DepthwiseConv2DParams) -> usize {
        depthwise_scratch_need(params)
    }

    /// The softmax SIMD path caches one `i32` exp value per row element:
    /// `row_size * 4` bytes (`softmax_simd` requires `scratch.len() >=
    /// row_size * 4` to engage).
    fn softmax_scratch_size(params: &SoftmaxParams) -> usize {
        (params.row_size.max(0) as usize) * 4
    }

    // `lstm_scratch_size` / `svdf_scratch_size` / `gru_scratch_size` keep
    // the trait default of 0 — the recurrent ops are `Unsupported` on this
    // backend, so no scratch is ever allocated for them.
}

/// A cheap sanity check that the shape-product and scratch formulas match
/// the kernels' own `need` computations for a canonical 3×3 SAME layer.
#[cfg(test)]
mod tests {
    use super::*;

    fn conv_params_3x3_same() -> Conv2DParams<'static> {
        Conv2DParams {
            input_shape: [1, 16, 16, 32],
            filter_shape: [64, 3, 3, 32],
            output_shape: [1, 16, 16, 64],
            padding: hematite_core::op_params::Padding::Same,
            stride_width: 1,
            stride_height: 1,
            dilation_width_factor: 1,
            dilation_height_factor: 1,
            input_offset: -128,
            weights_offset: 0,
            output_offset: 0,
            output_multiplier_per_channel: &[0],
            output_shift_per_channel: &[0],
            quantized_activation_min: -128,
            quantized_activation_max: 127,
        }
    }

    #[test]
    fn scratch_need_matches_kernel_formulas() {
        // 3×3 SAME 16×16×32: padded input (16+2)×(16+2)×32, padded weights
        // 64×9×32, accs 64×4, wsum 64×4 → the kernel's exact `need`.
        let p = conv_params_3x3_same();
        let expect = (18 * 18 * 32) + (64 * 9 * 32) + (64 * 4) + (64 * 4);
        assert_eq!(conv3x3_scratch_need(&p), expect);
        assert_eq!(S3Backend::conv2d_scratch_size(&p), expect);

        // 1×1 layer: accs out_c*4 + wsum (input_offset != 0).
        let mut p11 = conv_params_3x3_same();
        p11.filter_shape = [64, 1, 1, 32];
        assert_eq!(conv1x1_scratch_need(&p11), 64 * 4 + 64 * 4);
        assert_eq!(S3Backend::conv2d_scratch_size(&p11), 64 * 4 + 64 * 4);
    }
}
