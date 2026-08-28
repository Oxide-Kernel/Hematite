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
//! s3 has no kernel (scalar or SIMD) for these ops, and the project plan
//! mandates `Unsupported` — honest failure rather than a silent wrong answer:
//!
//! | Trait method | Reason |
//! |---|---|
//! | `matmul`, `sigmoid`, `tanh`, `leaky_relu`, `prelu`, `quantize`, `dequantize`, `unidirectional_sequence_lstm`, `svdf`, `gru`, `sum`, `reduce_max`, `reduce_min`, `arg_max`, `arg_min`, `l2_normalization` | No s3 kernel exists for the op. |
//!
//! # Data movement (plan todo 25 amendment)
//!
//! `reshape` / `transpose` / `concat` / `split` / `pad` / `slice` /
//! `resize_nearest` forward to the scalar kernels in [`data_movement`]
//! (added by the todo-25 amendment so the zoo models whose op sequences
//! contain RESHAPE=22 / TRANSPOSE=39 / PAD=34 can run through
//! `Model::<S3Backend>`). The kernels are bit-exact copies of the
//! `hematite-ref` scalar semantics — pure data movement, no arithmetic.
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
    ElementwiseChainParams, ElementwiseParams, FoldedPoolParams, FusedConvParams, FullyConnectedParams, GruParams,
    LstmParams, MatMulParams, PadParams, PoolParams, QuantParam, ReduceParams,
    ReshapeParams, ResizeNearestParams, SliceParams, SoftmaxParams,
    SplitParams, SvdfParams, TransposeParams,
};
use hematite_core::{KernelBackend, KernelError};

use crate::activations;
use crate::conv1x1;
use crate::conv3x3;
use crate::data_movement;
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

/// Scratch bytes needed by the 1×1 ACCX path (`conv1x1_accx_dispatch`).
///
/// T3.3 — when `input_c` is not a multiple of 16 the dispatch stages a
/// zero-padded input copy (`pixels * pad16(input_c)` bytes — every NHWC
/// pixel padded to the next multiple of 16) AND a zero-padded weight copy
/// (`out_c * pad16(input_c)` bytes — the kernel strides weight rows by the
/// padded channel count) in scratch at 16-byte-aligned offsets, plus the i32
/// accumulator buffer and the optional weight-sum buffer. The canonical
/// runtime formula — kept in sync with `conv1x1_scratch_need_codegen`
/// (hematite-codegen/src/generate.rs) and the dispatch's own `need`
/// computation in `conv1x1.rs`.
#[inline(always)]
fn conv1x1_scratch_need(params: &Conv2DParams) -> usize {
    let in_c = params.input_shape[3].max(0) as usize;
    let out_c = params.output_shape[3].max(0) as usize;
    let pixels = (params.input_shape[1].max(0) as usize) * (params.input_shape[2].max(0) as usize);
    let padded_c = pad16(in_c);
    let wsum = if params.input_offset != 0 { out_c * 4 } else { 0 };
    if padded_c != in_c {
        pixels * padded_c + out_c * padded_c + out_c * 4 + wsum
    } else {
        out_c * 4 + wsum
    }
}

/// Scratch bytes needed by the FC/GEMM ACCX path
/// (`fc_accx_dispatch` in `gemm.rs`).
///
/// When `input_dim` is not a multiple of 16 the dispatch stages a zero-padded
/// input copy (`pad16(input_dim)` bytes) AND a zero-padded weight copy
/// (`output_dim * pad16(input_dim)` bytes — the kernel strides weight rows by
/// the padded input dim) in scratch at 16-byte-aligned offsets, plus the i32
/// accumulator buffer and the optional weight-sum buffer (T3.6). The canonical
/// runtime formula — kept in sync with `fc_scratch_need_codegen`
/// (hematite-codegen/src/generate.rs) and the dispatch's own `need`
/// computation in `gemm.rs`.
#[inline(always)]
pub fn fc_scratch_need(params: &FullyConnectedParams) -> usize {
    let input_dim = params.input_dim.max(0) as usize;
    let out_dim = params.output_dim.max(0) as usize;
    let padded_dim = pad16(input_dim);
    let wsum = if params.input_offset != 0 { out_dim * 4 } else { 0 };
    if padded_dim != input_dim {
        padded_dim + out_dim * padded_dim + out_dim * 4 + wsum
    } else {
        out_dim * 4 + wsum
    }
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
///
/// T3.5 — depth_multiplier > 1: the kernel consumes `out_c`-channel vectors
/// and the dispatch stages a REPLICATED input (each input channel fanned out
/// to `dm` output channels), so the padded channel count is `pad16(out_c)` —
/// for dm==1 `out_c == input_c` and this equals the historical `pad16(in_c)`.
/// dm>1 always stages (replication cannot run on the caller's `in_c`-channel
/// input), so `needs_pad` is forced on for dm>1.
///
/// T3.5b — arbitrary filter sizes: the tap-parameterized anytap kernel needs
/// a `taps * padded_c` padded filter when channel-padding (taps = fh*fw, not
/// 9), and an extra `padded_c * 4` partial-accumulator buffer on the non-3x3
/// path (the anytap kernel WRITES per-chunk partials the caller adds into the
/// running accs — 3x3 keeps writing accs directly). Kept in sync with
/// `depthwise_scratch_need_codegen` (hematite-codegen/src/generate.rs) and
/// the dispatch's own `need` computation in `depthwise.rs`.
#[inline(always)]
fn depthwise_scratch_need(params: &DepthwiseConv2DParams) -> usize {
    let in_h = params.input_shape[1].max(0) as usize;
    let in_w = params.input_shape[2].max(0) as usize;
    let out_h = params.output_shape[1].max(0) as usize;
    let out_w = params.output_shape[2].max(0) as usize;
    let out_c = params.output_shape[3].max(0) as usize;
    let filter_h = params.filter_shape[1];
    let filter_w = params.filter_shape[2];
    let is_3x3 = filter_h == 3 && filter_w == 3;
    let taps = (filter_h.max(0) as usize) * (filter_w.max(0) as usize);

    let dilated_filter_h = (filter_h - 1) * params.dilation_height_factor + 1;
    let dilated_filter_w = (filter_w - 1) * params.dilation_width_factor + 1;
    let pad_total_h =
        ((out_h as i32 - 1) * params.stride_height + dilated_filter_h - in_h as i32).max(0) as usize;
    let pad_total_w =
        ((out_w as i32 - 1) * params.stride_width + dilated_filter_w - in_w as i32).max(0) as usize;
    let padded_c = pad16(out_c);
    let needs_channel_pad = padded_c != out_c;
    let dm_gt_1 = params.depth_multiplier > 1;
    let needs_pad = pad_total_h > 0 || pad_total_w > 0 || needs_channel_pad || dm_gt_1;
    // T4 — in_c == 1 arbitrary-filter layers use the single-channel broadcast
    // kernel: the staged input is single-channel (padded_h*padded_w), not
    // padded_c-wide. Mirrors depthwise.rs `use_bc1`.
    let input_c = params.input_shape[3].max(0) as usize;
    let use_bc1 = !is_3x3 && input_c == 1;

    let wsum = if params.input_offset != 0 { out_c * 4 } else { 0 };
    let partials = if is_3x3 { 0 } else { padded_c * 4 };
    if needs_pad {
        let pad_input_len = if use_bc1 {
            (in_h + pad_total_h) * (in_w + pad_total_w)
        } else {
            (in_h + pad_total_h) * (in_w + pad_total_w) * padded_c
        };
        let pad_filter_len = if needs_channel_pad { taps * padded_c } else { 0 };
        // +48: worst-case alignment padding in the dispatch carve (in_off,
        // w_off, accs_off each round up to a 16-byte boundary).
        pad_input_len + pad_filter_len + padded_c * 4 + wsum + partials + 48
    } else {
        out_c * 4 + wsum + partials
    }
}

/// Scratch bytes needed by the composed `fused_conv2d` SIMD path (T2.2).
///
/// EQUALS the anchor conv's own need ([`S3Backend::conv2d_scratch_size`]):
/// the fused epilogue reads the residual tensor in place (it is a model
/// constant / computed tensor provided as a slice — never staged) and holds
/// the conv output in registers (never materialized), so no additional
/// staging beyond the conv's padded-input/weights + accumulator + weight-sum
/// buffers is required. The `RefBackend`-style decomposition forwards the same
/// scratch to `conv2d` (hematite-ref/src/fused.rs), so the composed need ==
/// the conv need is the parity invariant asserted by the T1.4
/// `composed_scratch_parity` test.
///
/// Kept in sync with `fused_conv2d_scratch_need_codegen`
/// (hematite-codegen/src/generate.rs).
#[inline(always)]
pub fn fused_conv2d_scratch_need(params: &FusedConvParams<'_>) -> usize {
    S3Backend::conv2d_scratch_size(&params.conv)
}

/// Scratch bytes needed by the composed `fused_elementwise_chain` SIMD path
/// (T2.3): **ZERO**.
///
/// The fused chain keeps the running value in i32 register lanes between
/// steps (never materialized) and reads the step operands (model constant
/// tensors) in place — nothing is staged. The decomposition forwards no
/// scratch either (the per-op elementwise/activation kernels take
/// `&mut []`), so `0 == 0` is the parity invariant asserted by
/// `tests/fused_chain_golden.rs::fused_elementwise_chain_needs_no_scratch`.
///
/// No codegen mirror exists because the codegen chain emitter already reports
/// the literal `0` (`emit_fused_chain` → `scratch: 0`), so the mirror would
/// be `0 == 0` by construction.
#[inline(always)]
pub fn fused_elementwise_chain_scratch_need(_params: &ElementwiseChainParams<'_>) -> usize {
    0
}

/// Scratch bytes needed by the composed `fused_pool_with_fold` (T2.4): the
/// fold staging region — the fold output tensor bytes (`num_elements`, the
/// fold input = pool input flat count) padded up to the pool SIMD kernel's
/// 16-byte multiple — plus the pool's own scratch need (0: the s3 pool
/// kernels ignore scratch). No fold → 0 (the pool reads `src` in place).
///
/// The decomposition (hematite-ref/src/fused.rs) materializes the absorbed
/// fold into exactly `num_elements` scratch bytes and the pool then needs 0
/// more; the `pad16` gives the TIE728 `*_22c1` input-pointer alignment head
/// room so a 16-aligned scratch base keeps the staged fold SIMD-readable.
///
/// Kept in sync with `fused_pool_fold_scratch_need_codegen`
/// (hematite-codegen/src/generate.rs) — the T1.4 parity invariant.
#[inline(always)]
pub fn fused_pool_with_fold_scratch_need(params: &FoldedPoolParams<'_>) -> usize {
    match &params.fold {
        Some(fold) => pad16(fold.num_elements.max(0) as usize),
        None => 0,
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
        input: &[i8],
        params: &ReshapeParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        data_movement::reshape(input, params, output)
    }

    fn transpose(
        &self,
        input: &[i8],
        params: &TransposeParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        data_movement::transpose(input, params, output)
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

    /// Splits both output slices in one call, mirroring RefBackend's
    /// split-index adaptation (split_index 0 → `output_a`, 1 → `output_b`).
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
        data_movement::resize_nearest_neighbor(input, params, output, &mut [])
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

    #[test]
    fn depthwise_dm_gt1_scratch_need_matches_kernel_formula() {
        // dm=2, 3×3 SAME, in_c=8 -> out_c=16 (kws-style fan-out). dm>1 forces
        // the staged path: padded input 14×14×16, no channel pad (16 % 16 == 0),
        // accs 16×4, no wsum (input_offset 0).
        let p = DepthwiseConv2DParams {
            input_shape: [1, 12, 12, 8],
            filter_shape: [1, 3, 3, 16],
            output_shape: [1, 12, 12, 16],
            padding: hematite_core::op_params::Padding::Same,
            stride_width: 1,
            stride_height: 1,
            dilation_width_factor: 1,
            dilation_height_factor: 1,
            depth_multiplier: 2,
            input_offset: 0,
            weights_offset: 0,
            output_offset: 0,
            output_multiplier_per_channel: &[0; 16],
            output_shift_per_channel: &[0; 16],
            quantized_activation_min: -128,
            quantized_activation_max: 127,
        };
        let expect = (14 * 14 * 16) + 0 + (16 * 4) + 48;
        assert_eq!(depthwise_scratch_need(&p), expect);
        assert_eq!(S3Backend::depthwise_conv2d_scratch_size(&p), expect);

        // dm=4, in_c=3 -> out_c=12 (non-%16): staged filter 9×16 + channel pad.
        let mut p2 = p;
        p2.input_shape = [1, 12, 12, 3];
        p2.filter_shape = [1, 3, 3, 12];
        p2.output_shape = [1, 12, 12, 12];
        p2.depth_multiplier = 4;
        p2.output_multiplier_per_channel = &[0; 12];
        p2.output_shift_per_channel = &[0; 12];
        let expect2 = (14 * 14 * 16) + (9 * 16) + (16 * 4) + 48;
        assert_eq!(depthwise_scratch_need(&p2), expect2);
        assert_eq!(S3Backend::depthwise_conv2d_scratch_size(&p2), expect2);

        // Non-zero input_offset adds the per-channel weight-sum buffer.
        let mut p3 = p2;
        p3.input_offset = -3;
        let expect3 = (14 * 14 * 16) + (9 * 16) + (16 * 4) + (12 * 4) + 48;
        assert_eq!(depthwise_scratch_need(&p3), expect3);
        assert_eq!(S3Backend::depthwise_conv2d_scratch_size(&p3), expect3);

        // T3.5b — arbitrary filter (kws 10×8, dm=8, in_c=1 -> out_c=8,
        // stride 2 SAME): padded filter uses taps (80) instead of 9, plus the
        // anytap partial-accumulator buffer (padded_c*4).
        let p4 = DepthwiseConv2DParams {
            input_shape: [1, 49, 40, 1],
            filter_shape: [1, 10, 8, 8],
            output_shape: [1, 25, 20, 8],
            padding: hematite_core::op_params::Padding::Same,
            stride_width: 2,
            stride_height: 2,
            dilation_width_factor: 1,
            dilation_height_factor: 1,
            depth_multiplier: 8,
            input_offset: 0,
            weights_offset: 0,
            output_offset: 0,
            output_multiplier_per_channel: &[0; 8],
            output_shift_per_channel: &[0; 8],
            quantized_activation_min: -128,
            quantized_activation_max: 127,
        };
        // padded_h 58, padded_w 46, padded_c 16; staged filter 80*16; accs 16*4;
        // partials 16*4. T4 — in_c == 1 triggers the single-channel broadcast
        // path, so the staged INPUT is single-channel (58*46, not 58*46*16).
        let expect4 = (58 * 46) + (80 * 16) + (16 * 4) + (16 * 4) + 48;
        assert_eq!(depthwise_scratch_need(&p4), expect4);
        assert_eq!(S3Backend::depthwise_conv2d_scratch_size(&p4), expect4);
    }

    /// T3.3 — the conv1x1 padded-path scratch need formula must match the
    /// dispatch layout in `conv1x1.rs::conv1x1_accx_dispatch` for pad and
    /// no-pad shapes.
    #[test]
    fn conv1x1_scratch_need_matches_padded_dispatch_layout() {
        fn p(in_c: i32, out_c: i32, spatial: i32, input_offset: i32) -> Conv2DParams<'static> {
            Conv2DParams {
                input_shape: [1, spatial, spatial, in_c],
                filter_shape: [out_c, 1, 1, in_c],
                output_shape: [1, spatial, spatial, out_c],
                padding: hematite_core::op_params::Padding::Same,
                stride_width: 1,
                stride_height: 1,
                dilation_width_factor: 1,
                dilation_height_factor: 1,
                input_offset,
                weights_offset: 0,
                output_offset: 0,
                output_multiplier_per_channel: &[0],
                output_shift_per_channel: &[0],
                quantized_activation_min: -128,
                quantized_activation_max: 127,
            }
        }
        // No pad (input_c % 16 == 0): accs + wsum only.
        assert_eq!(conv1x1_scratch_need(&p(16, 64, 1, 0)), 64 * 4);
        assert_eq!(conv1x1_scratch_need(&p(32, 8, 1, 0)), 8 * 4);
        assert_eq!(conv1x1_scratch_need(&p(16, 64, 1, 5)), 64 * 4 + 64 * 4);

        // Pad (input_c % 16 != 0): padded input (pixels × pad16(in_c)) +
        // padded weights + accs (+ wsum when input_offset != 0). in_c 3 →
        // pad16 = 16, 1 pixel.
        assert_eq!(conv1x1_scratch_need(&p(3, 16, 1, 0)), 16 + 16 * 16 + 16 * 4);
        assert_eq!(
            conv1x1_scratch_need(&p(3, 16, 1, 128)),
            16 + 16 * 16 + 16 * 4 + 16 * 4
        );
        // in_c 3, 4×4 spatial → 16 pixels × 16 padded channels.
        assert_eq!(conv1x1_scratch_need(&p(3, 16, 4, 0)), 16 * 16 + 16 * 16 + 16 * 4);
        // in_c 17 → pad16 = 32.
        assert_eq!(conv1x1_scratch_need(&p(17, 4, 1, 0)), 32 + 4 * 32 + 4 * 4);
    }

    /// T3.6 — the FC padded-path scratch need formula must match the dispatch
    /// layout in `gemm.rs::fc_accx_dispatch` for pad and no-pad shapes.
    #[test]
    fn fc_scratch_need_matches_padded_dispatch_layout() {
        fn p(input_dim: i32, output_dim: i32, input_offset: i32) -> FullyConnectedParams<'static> {
            FullyConnectedParams {
                input_dim,
                output_dim,
                input_offset,
                weights_offset: 0,
                output_offset: 0,
                output_multiplier_per_channel: &[0],
                output_shift_per_channel: &[0],
                quantized_activation_min: -128,
                quantized_activation_max: 127,
            }
        }
        // No pad (input_dim % 16 == 0): accs + wsum only.
        assert_eq!(fc_scratch_need(&p(16, 64, 0)), 64 * 4);
        assert_eq!(fc_scratch_need(&p(32, 8, 0)), 8 * 4);
        assert_eq!(fc_scratch_need(&p(16, 64, 5)), 64 * 4 + 64 * 4);

        // Pad (input_dim % 16 != 0): padded input + padded weights + accs
        // (+ wsum when input_offset != 0). input_dim 1 → pad16 = 16.
        assert_eq!(fc_scratch_need(&p(1, 16, 0)), 16 + 16 * 16 + 16 * 4);
        assert_eq!(fc_scratch_need(&p(1, 16, 128)), 16 + 16 * 16 + 16 * 4 + 16 * 4);
        // input_dim 8 → pad16 = 16; anomaly_detect's 8→128 gated-out FC.
        assert_eq!(fc_scratch_need(&p(8, 128, 128)), 16 + 128 * 16 + 128 * 4 + 128 * 4);
        // input_dim 15 → pad16 = 16; 17 → pad16 = 32.
        assert_eq!(fc_scratch_need(&p(15, 3, 0)), 16 + 3 * 16 + 3 * 4);
        assert_eq!(fc_scratch_need(&p(17, 4, 0)), 32 + 4 * 32 + 4 * 4);
        // input_dim 1, output_dim 1 (sine-family shape).
        assert_eq!(fc_scratch_need(&p(1, 1, 0)), 16 + 16 + 4);
    }
}
