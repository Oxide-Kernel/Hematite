// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! hematite-core — foundational types and traits for the Hematite embedded NN
//! library.
//!
//! This crate is **zero-dependency, `no_std`** and provides:
//!
//! * [`KernelBackend`] — the trait every backend implements
//! * [`FusedKernelBackend`] — the composed-kernel entry points (T2.1)
//! * [`KernelError`] — the error type all kernel methods return
//! * [`op_params`] — operator parameter structs mirroring TFLite Micro

#![no_std]

pub mod op_params;

pub use op_params::{
    ActivationEpilogueParams, ActivationParams, ComposedActivation, ConcatParams,
    Conv2DParams, DepthwiseConv2DParams, ElementwiseChainParams,
    ElementwiseChainStep, ElementwiseKind, ElementwiseParams, FoldedPoolParams,
    FullyConnectedParams, FusedActivation, FusedConvParams, GruParams,
    LstmParams, MatMulParams, PadParams, Padding, PerChannelQuantParam,
    PoolInputFold, PoolKind, PoolParams, QuantParam, ReduceParams,
    ReshapeParams, ResizeNearestParams, ResidualAddParams, SliceParams,
    SoftmaxParams, SplitParams, SvdfParams, TransposeParams,
};

/// Error type returned by every [`KernelBackend`] method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelError {
    /// Input, weight, bias, or output slice lengths do not match the
    /// expected shape for the given params.
    ShapeMismatch,
    /// The scratch buffer is too small for the operation.
    ScratchTooSmall,
    /// The requested op / configuration is not supported by this
    /// backend (e.g. unsupported fused activation, dilation > 1 on a
    /// SIMD backend that only handles dilation = 1).
    Unsupported,
}

/// The contract every backend implements.
///
/// All model-inference code is generic over `B: KernelBackend`.  Op
/// methods are **required** (no default body) and every method carries
/// rustdoc invariants about expected slice lengths, tensor layouts, and
/// the [`KernelError`] variants it may return.
///
/// Scratch-size associated functions return `usize` and default to `0`
/// so that a reference implementation that needs no scratch can still
/// implement the trait without boilerplate.
pub trait KernelBackend {
    // ── Tier0 — Core compute ────────────────────────────────────────────

    /// 2D convolution  (int8 input, int8 weights, i32 bias, int8 output).
    ///
    /// # Layout
    /// All slices are in NHWC row-major order (batch * height * width *
    /// channels), with batch = 1.
    ///
    /// # Expected lengths
    /// * `input.len()`  == input_height * input_width * input_channels
    /// * `weights.len()` == output_channels * filter_height * filter_width * input_channels
    /// * `bias.len()`    == params.output_multiplier_per_channel.len()
    /// * `output.len()`  == output_height * output_width * output_channels
    /// * `scratch.len()` >= [`conv2d_scratch_size`](Self::conv2d_scratch_size)
    ///
    /// # Errors
    /// Returns [`KernelError::ShapeMismatch`] if any slice length is
    /// wrong; [`KernelError::ScratchTooSmall`] if `scratch` is
    /// insufficient.
    fn conv2d(
        &self,
        input: &[i8],
        weights: &[i8],
        bias: &[i32],
        params: &Conv2DParams,
        output: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError>;

    /// Depthwise 2D convolution.
    ///
    /// Layout and expected lengths follow [`conv2d`](Self::conv2d) with:
    /// * `weights.len()` == filter_height * filter_width * input_channels * params.depth_multiplier
    /// * `output.len()`  == output_height * output_width * input_channels * params.depth_multiplier
    fn depthwise_conv2d(
        &self,
        input: &[i8],
        weights: &[i8],
        bias: &[i32],
        params: &DepthwiseConv2DParams,
        output: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError>;

    /// Fully-connected (dense) layer.
    ///
    /// # Layout
    /// * `input`  — NHWC batch=1, flat size = `input_dim`
    /// * `weights` — row-major `[output_dim][input_dim]`
    /// * `bias`    — `[output_dim]`
    /// * `output`  — `[output_dim]`
    fn fully_connected(
        &self,
        input: &[i8],
        weights: &[i8],
        bias: &[i32],
        params: &FullyConnectedParams,
        output: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError>;

    /// Batch matrix multiply (used as MatMul / GEMM).
    ///
    /// Equivalent to `output = adj_x(A) * adj_y(B)` with full
    /// quantization.  Per-tensor (not per-channel) requantize.
    fn matmul(
        &self,
        input: &[i8],
        weights: &[i8],
        bias: &[i32],
        params: &MatMulParams,
        output: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError>;

    // ── Tier1 — Pooling ─────────────────────────────────────────────────

    /// Average-pool 2D  (int8 → int32 accumulate → requantize → clamp).
    fn average_pool_2d(
        &self,
        input: &[i8],
        params: &PoolParams,
        output: &mut [i8],
    ) -> Result<(), KernelError>;

    /// Max-pool 2D  (pure int8 comparison, no requantize).
    fn max_pool_2d(
        &self,
        input: &[i8],
        params: &PoolParams,
        output: &mut [i8],
    ) -> Result<(), KernelError>;

    // ── Tier1 — Softmax ─────────────────────────────────────────────────

    /// Int8-safe softmax  (max-subtract, polynomial exp, reciprocal,
    /// clamp).
    ///
    /// Uses the fields in [`SoftmaxParams`] — output scale =
    /// `INPUT_SCALE`, output zero-point = `i8::MIN` (the standard
    /// TFLM int8 softmax convention).
    fn softmax(
        &self,
        input: &[i8],
        params: &SoftmaxParams,
        output: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError>;

    // ── Tier1 — Standalone activations ──────────────────────────────────

    /// ReLU: `max(input_offset, x)` → requantize → clamp.
    fn relu(
        &self,
        input: &[i8],
        params: &ActivationParams,
        output: &mut [i8],
    ) -> Result<(), KernelError>;

    /// ReLU6: `clamp(x, input_offset, quantized_activation_max)`.
    fn relu6(
        &self,
        input: &[i8],
        params: &ActivationParams,
        output: &mut [i8],
    ) -> Result<(), KernelError>;

    /// Hard Swish (int8): `x * ReLU6(x + 3) / 6` in quantized space.
    fn hard_swish(
        &self,
        input: &[i8],
        params: &ActivationParams,
        output: &mut [i8],
    ) -> Result<(), KernelError>;

    /// Sigmoid (logistic): polynomial or LUT-based int8 approximation.
    fn sigmoid(
        &self,
        input: &[i8],
        params: &ActivationParams,
        output: &mut [i8],
    ) -> Result<(), KernelError>;

    /// Tanh: polynomial or LUT-based int8 approximation.
    fn tanh(
        &self,
        input: &[i8],
        params: &ActivationParams,
        output: &mut [i8],
    ) -> Result<(), KernelError>;

    /// Leaky ReLU: `x >= input_offset ? identity(x) : alpha(x)` in
    /// quantized space, using `output_multiplier_identity /
    /// output_shift_identity` and `output_multiplier_alpha /
    /// output_shift_alpha`.
    fn leaky_relu(
        &self,
        input: &[i8],
        params: &ActivationParams,
        output: &mut [i8],
    ) -> Result<(), KernelError>;

    /// PReLU (parametric ReLU): per-channel alpha, two-branch
    /// requantize (`output_multiplier_1/shift_1` for positive,
    /// `output_multiplier_2/shift_2` for negative).
    ///
    /// `params.alpha_data.len()` must equal the number of input
    /// channels.
    fn prelu(
        &self,
        input: &[i8],
        params: &ActivationParams,
        output: &mut [i8],
    ) -> Result<(), KernelError>;

    // ── Tier1 — Elementwise ─────────────────────────────────────────────

    /// Elementwise add: `output = input1 + input2` in quantized space
    /// (broadcast NOT supported — shapes must match).
    fn add(
        &self,
        input1: &[i8],
        input2: &[i8],
        params: &ElementwiseParams,
        output: &mut [i8],
    ) -> Result<(), KernelError>;

    /// Elementwise multiply.
    fn mul(
        &self,
        input1: &[i8],
        input2: &[i8],
        params: &ElementwiseParams,
        output: &mut [i8],
    ) -> Result<(), KernelError>;

    /// Elementwise subtract: `output = input1 - input2`.
    fn sub(
        &self,
        input1: &[i8],
        input2: &[i8],
        params: &ElementwiseParams,
        output: &mut [i8],
    ) -> Result<(), KernelError>;

    // ── Tier1 — Quantize / Dequantize ───────────────────────────────────

    /// Quantize float-like (int8 reinterpreted) → int8.
    fn quantize(
        &self,
        input: &[i8],
        params: &QuantParam,
        output: &mut [i8],
    ) -> Result<(), KernelError>;

    /// Dequantize int8 → int8 (affine map).
    fn dequantize(
        &self,
        input: &[i8],
        params: &QuantParam,
        output: &mut [i8],
    ) -> Result<(), KernelError>;

    // ── Tier2 — Data movement ───────────────────────────────────────────

    /// Reshape (metadata change, zero-copy where possible).
    fn reshape(
        &self,
        input: &[i8],
        params: &ReshapeParams,
        output: &mut [i8],
    ) -> Result<(), KernelError>;

    /// Transpose (general N-dimensional permutation).
    fn transpose(
        &self,
        input: &[i8],
        params: &TransposeParams,
        output: &mut [i8],
    ) -> Result<(), KernelError>;

    /// Concatenate two tensors along `params.axis`.
    fn concat(
        &self,
        input_a: &[i8],
        input_b: &[i8],
        params: &ConcatParams,
        output: &mut [i8],
    ) -> Result<(), KernelError>;

    /// Split one tensor into two output slices along `params.axis`.
    fn split(
        &self,
        input: &[i8],
        params: &SplitParams,
        output_a: &mut [i8],
        output_b: &mut [i8],
    ) -> Result<(), KernelError>;

    /// Pad (constant-value padding, zero-padding by default).
    fn pad(
        &self,
        input: &[i8],
        params: &PadParams,
        output: &mut [i8],
    ) -> Result<(), KernelError>;

    /// Slice (crop a sub-volume from the input).
    fn slice(
        &self,
        input: &[i8],
        params: &SliceParams,
        output: &mut [i8],
    ) -> Result<(), KernelError>;

    /// Resize-nearest-neighbor (asymmetric, floor, int8 copy).
    fn resize_nearest(
        &self,
        input: &[i8],
        params: &ResizeNearestParams,
        output: &mut [i8],
    ) -> Result<(), KernelError>;

    // ── Tier3 — Recurrent ───────────────────────────────────────────────

    /// Unidirectional sequence LSTM  (int8 activations, i16 cell state).
    ///
    /// # Layout
    /// * Input: flattened NHWC, time-major or batch-major per
    ///   `params.time_major`.
    /// * All gate weight tensors are `[num_units, input_dim]`
    ///   (row-major) or `[num_units, num_units]` for recurrent
    ///   weights.
    /// * `cell_state: &mut [i16]` (i16 for fixed-point cell; length =
    ///   `num_units`).
    /// * `hidden_state: &mut [i8]` (length = `num_units`).
    #[allow(clippy::too_many_arguments)]
    fn unidirectional_sequence_lstm(
        &self,
        input: &[i8],
        input_to_input_weights: &[i8],
        input_to_forget_weights: &[i8],
        input_to_cell_weights: &[i8],
        input_to_output_weights: &[i8],
        recurrent_to_input_weights: &[i8],
        recurrent_to_forget_weights: &[i8],
        recurrent_to_cell_weights: &[i8],
        recurrent_to_output_weights: &[i8],
        input_gate_bias: &[i32],
        forget_gate_bias: &[i32],
        cell_bias: &[i32],
        output_gate_bias: &[i32],
        params: &LstmParams,
        output: &mut [i8],
        cell_state: &mut [i16],
        hidden_state: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError>;

    /// SVDF (Singular Value Decomposition Filter).
    ///
    /// * `weights_feature`: `[num_filters][input_dim]`
    /// * `weights_time`: `[svdf_rank][num_filters]`
    /// * `hidden_state`: `&mut [i16]` — the internal time-state.
    #[allow(clippy::too_many_arguments)]
    fn svdf(
        &self,
        input: &[i8],
        weights_feature: &[i8],
        weights_time: &[i16],
        bias: &[i32],
        params: &SvdfParams,
        output: &mut [i8],
        hidden_state: &mut [i16],
        scratch: &mut [u8],
    ) -> Result<(), KernelError>;

    /// GRU (custom op, goldens via embedded-nn cross-check).
    ///
    /// * `reset_gate_weights`, `update_gate_weights`,
    ///   `candidate_weights`: `[num_units, input_dim + num_units]`
    ///   (input + recurrent concatenated).
    /// * `biases`: concatenated `[reset_bias, update_bias,
    ///   candidate_bias]` in `i32`.
    /// * `hidden_state`: `&mut [i16]` — int16 hidden state.
    #[allow(clippy::too_many_arguments)]
    fn gru(
        &self,
        input: &[i8],
        reset_gate_weights: &[i8],
        update_gate_weights: &[i8],
        candidate_weights: &[i8],
        biases: &[i32],
        params: &GruParams,
        output: &mut [i8],
        hidden_state: &mut [i16],
        scratch: &mut [u8],
    ) -> Result<(), KernelError>;

    // ── Tier4 — Reductions ──────────────────────────────────────────────

    /// Mean (reduce by averaging along axes, requantize).
    fn mean(
        &self,
        input: &[i8],
        params: &ReduceParams,
        output: &mut [i8],
    ) -> Result<(), KernelError>;

    /// Sum (reduce by addition along axes).
    fn sum(
        &self,
        input: &[i8],
        params: &ReduceParams,
        output: &mut [i8],
    ) -> Result<(), KernelError>;

    /// Reduce-max.
    fn reduce_max(
        &self,
        input: &[i8],
        params: &ReduceParams,
        output: &mut [i8],
    ) -> Result<(), KernelError>;

    /// Reduce-min.
    fn reduce_min(
        &self,
        input: &[i8],
        params: &ReduceParams,
        output: &mut [i8],
    ) -> Result<(), KernelError>;

    /// Arg-max (returns index of maximum along axis; output is `i32`
    /// or `i64` per `params.output_type`).
    fn arg_max(
        &self,
        input: &[i8],
        params: &ReduceParams,
        output: &mut [i8],
    ) -> Result<(), KernelError>;

    /// Arg-min.
    fn arg_min(
        &self,
        input: &[i8],
        params: &ReduceParams,
        output: &mut [i8],
    ) -> Result<(), KernelError>;

    /// L2 normalization (non-quantized float path NOT exposed;
    /// int8-compatible reduction path).
    fn l2_normalization(
        &self,
        input: &[i8],
        params: &ReduceParams,
        output: &mut [i8],
    ) -> Result<(), KernelError>;

    // ── Scratch-size associated functions ───────────────────────────────

    /// Scratch bytes needed by [`conv2d`](Self::conv2d).
    fn conv2d_scratch_size(params: &Conv2DParams) -> usize {
        let _ = params;
        0
    }

    /// Scratch bytes needed by
    /// [`depthwise_conv2d`](Self::depthwise_conv2d).
    fn depthwise_conv2d_scratch_size(params: &DepthwiseConv2DParams) -> usize {
        let _ = params;
        0
    }

    /// Scratch bytes needed by [`softmax`](Self::softmax).
    fn softmax_scratch_size(params: &SoftmaxParams) -> usize {
        let _ = params;
        0
    }

    /// Scratch bytes needed by unidirectional sequence LSTM.
    fn lstm_scratch_size(params: &LstmParams) -> usize {
        let _ = params;
        0
    }

    /// Scratch bytes needed by SVDF.
    fn svdf_scratch_size(params: &SvdfParams) -> usize {
        let _ = params;
        0
    }

    /// Scratch bytes needed by GRU.
    fn gru_scratch_size(params: &GruParams) -> usize {
        let _ = params;
        0
    }
}

/// Composed-kernel entry points (T2.1): fused op groups emitted as ONE
/// kernel call by the composed emitter.
///
/// The fused methods are PURELY ADDITIVE — they sit alongside (never
/// modify) [`KernelBackend`] and default to nothing: a backend opts in by
/// implementing this trait.  Each composed call is the sum of the exact
/// per-op sequence the unfused emitter would emit (anchor op first, then
/// absorbed residual-add / chain steps / input fold, then the trailing
/// activation epilogue), so a backend can implement it by simply forwarding
/// to its own per-op methods and be **bit-exact by construction**.
///
/// The composed param structs (see [`op_params`]: [`FusedConvParams`],
/// [`ElementwiseChainParams`], [`FoldedPoolParams`]) carry the anchor's
/// per-op params EXACTLY as the unfused emitter would emit them, plus the
/// fusion-side data the composed kernel needs; the T1.2 emitter derives
/// them from the T1.1 fusion IR (`FusedGroup` and friends in
/// `hematite-codegen/src/optimize/fusion.rs`).
///
/// # Safety contract of the decomposition
///
/// The reference decomposition performs in-place elementwise chaining
/// (`dst` as both running input and output).  This is only sound because
/// every hematite-ref elementwise / activation kernel reads `input[i]`
/// strictly before writing `output[i]` — see the single documented alias
/// helper in `hematite-ref/src/fused.rs`.
pub trait FusedKernelBackend: KernelBackend {
    /// Fused CONV_2D + optional residual-ADD + optional activation
    /// epilogue, in one kernel call.
    ///
    /// # Decomposition (unfused per-op sequence, bit-exact by construction)
    ///
    /// 1. `conv2d(src, weight, bias, &params.conv, dst, scratch)`
    /// 2. if `params.residual.is_some()`:
    ///    `add(dst, residual_data, add_params, dst)` — two-stage TFLM Add
    ///    rounding (per-input multipliers first, then i32 sum, then final
    ///    requantize), in-place.
    /// 3. if `params.activation.kind != None`: the matching standalone
    ///    activation op (relu / relu6 / hard_swish) in-place on `dst`.
    ///
    /// The embedded `conv` params carry the anchor conv's OWN baked
    /// activation range (never the absorbed epilogue's) so the conv call
    /// matches the unfused emission 1:1.
    ///
    /// # Errors
    ///
    /// Returns the first per-op error ([`KernelError::ShapeMismatch`] /
    /// [`KernelError::ScratchTooSmall`] / [`KernelError::Unsupported`]).
    fn fused_conv2d(
        &mut self,
        src: &[i8],
        weight: &[i8],
        bias: &[i32],
        params: &FusedConvParams,
        dst: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError>;

    /// Fused elementwise chain (anchor op + absorbed steps) in one pass.
    ///
    /// # Decomposition (unfused per-op sequence, bit-exact by construction)
    ///
    /// Steps execute in order; step 0 reads `src` as input1, steps ≥ 1 read
    /// `dst` in-place as input1 (the running value):
    ///
    /// * `Add` / `Mul` / `Sub` → `add` / `mul` / `sub` with the step's
    ///   `ElementwiseParams`-derived fields and `step.operand` as input2
    ///   (model constant tensors — never alias `dst`).
    /// * `Relu` / `Relu6` / `HardSwish` → the matching standalone activation
    ///   op in-place.
    ///
    /// # Errors
    ///
    /// Returns the first per-op error.
    fn fused_elementwise_chain(
        &mut self,
        src: &[i8],
        params: &ElementwiseChainParams,
        dst: &mut [i8],
    ) -> Result<(), KernelError>;

    /// Fused pool + optional input fold (MUL/SUB) + optional activation
    /// epilogue, in one kernel call.
    ///
    /// # Decomposition (unfused per-op sequence, bit-exact by construction)
    ///
    /// 1. if `params.fold.is_some()`: the fold op (`mul` for builtin 18,
    ///    `sub` for builtin 41) applied to `src` and `fold.operand_data`,
    ///    writing into `scratch` reinterpreted as the i8 intermediate
    ///    (`scratch` is provided exactly for this).
    /// 2. `average_pool_2d` / `max_pool_2d` (per `params.pool_kind`)
    ///    reading the intermediate (or `src` directly when no fold).
    /// 3. if `params.activation.kind != None`: the matching standalone
    ///    activation op in-place on `dst`.
    ///
    /// # Errors
    ///
    /// Returns the first per-op error.
    fn fused_pool_with_fold(
        &mut self,
        src: &[i8],
        params: &FoldedPoolParams,
        dst: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError>;
}
