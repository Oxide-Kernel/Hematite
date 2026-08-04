// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Operator parameter structs — one per op from the in-scope tier table.
//!
//! Each struct mirrors the corresponding TFLite Micro C parameter struct,
//! with field names converted to Rust snake_case.  Together they form the
//! typed API surface that every `KernelBackend` implementation consumes.
//!
//! # Shape convention
//!
//! All 4D shape arrays use **NHWC** element order:
//! `[batch(=1), height, width, channels]`.  Filter (weight) shapes use
//! **OHWI**: `[output_channels, filter_height, filter_width, input_channels]`.
//! Each shape field's rustdoc states its element order explicitly.
//!
//! # Shared types
//!
//! * [`Padding`] — TFLM `PaddingType` / `TfLitePadding`
//! * [`FusedActivation`] — TFLM `FusedActivationFunctionType` / `TfLiteFusedActivation`
//! * [`QuantParam`] — quantize / dequantize multiplier + shift + zero-point
//! * [`PerChannelQuantParam`] — per-channel multiplier + shift for
//!   `requantize(acc, &PerChannelQuantParam, channel) -> i8`

// ── Shared supporting types ────────────────────────────────────────────────

/// Padding strategy, matching TFLM `PaddingType` and `TfLitePadding`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Padding {
    /// No padding — output spatial dimensions shrink.
    Valid,
    /// Zero-pad so output spatial dimensions equal input (stride 1).
    Same,
}

/// Fused activation function, matching TFLM
/// `FusedActivationFunctionType` and `TfLiteFusedActivation`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusedActivation {
    /// No fused activation.
    None,
    /// `max(0, x)` fused into epilogue.
    Relu,
    /// `clamp(x, 0, 6 * scale)` fused into epilogue.
    Relu6,
    /// `clamp(x, -1, 1)` fused into epilogue.
    Relu1,
}

/// Quantization parameters for quantize / dequantize ops.
///
/// Uses `(multiplier, shift)` pairs (not bare Q0.31) so that the quantize
/// direction — which encodes the reciprocal scale, typically > 1.0 — is
/// representable.  Corresponds to TFLM `DequantizationParams`.
///
/// * Quantize:   `output[i] = multiply_by_quantized_multiplier(input[i],
///                quantize_multiplier, quantize_shift) + zero_point`
/// * Dequantize: `output[i] = multiply_by_quantized_multiplier(
///                input[i] - zero_point, dequantize_multiplier,
///                dequantize_shift)`
#[derive(Clone, Copy, Debug)]
pub struct QuantParam {
    /// Q0.31 multiplier for the quantize direction (`1 / scale`).
    pub quantize_multiplier: i32,
    /// Right-shift for [`quantize_multiplier`](Self::quantize_multiplier).
    pub quantize_shift: i32,
    /// Q0.31 multiplier for the dequantize direction (`scale`).
    pub dequantize_multiplier: i32,
    /// Right-shift for [`dequantize_multiplier`](Self::dequantize_multiplier).
    pub dequantize_shift: i32,
    /// Quantization zero-point.
    pub zero_point: i32,
}

/// Per-channel quantization parameters.
///
/// Carries one multiplier and shift per output channel, matching how
/// TFLM's int8 conv / depthwise / FC kernels use
/// `output_multiplier_per_channel` / `output_shift_per_channel`.
///
/// The `requantize` signature from `hematite-int8` is:
///
/// ```ignore
/// fn requantize(acc: i32, params: &PerChannelQuantParam, output_channel: usize) -> i8
/// ```
#[derive(Clone, Copy, Debug)]
pub struct PerChannelQuantParam<'a> {
    /// Per-channel output multiplier (Q0.31 format).
    pub output_multiplier_per_channel: &'a [i32],
    /// Per-channel output right-shift.
    pub output_shift_per_channel: &'a [i32],
}

// ── Tier 0 — Core compute ──────────────────────────────────────────────────

/// Parameters for CONV_2D.
///
/// Mirrors TFLM `ConvParams` (types.h) and `TfLiteConvParams`
/// (builtin_op_data.h).
#[derive(Clone, Debug)]
pub struct Conv2DParams<'a> {
    /// Input tensor shape in NHWC: `[batch=1, height, width, input_channels]`.
    pub input_shape: [i32; 4],
    /// Filter shape in OHWI: `[output_channels, filter_height, filter_width, input_channels]`.
    pub filter_shape: [i32; 4],
    /// Output tensor shape in NHWC: `[batch=1, output_height, output_width, output_channels]`.
    pub output_shape: [i32; 4],
    /// Padding strategy (same / valid).
    pub padding: Padding,
    /// Stride along width dimension.
    pub stride_width: i32,
    /// Stride along height dimension.
    pub stride_height: i32,
    /// Dilation factor along width.
    pub dilation_width_factor: i32,
    /// Dilation factor along height.
    pub dilation_height_factor: i32,
    /// Input zero-point offset.
    pub input_offset: i32,
    /// Weights zero-point offset.
    pub weights_offset: i32,
    /// Output zero-point offset.
    pub output_offset: i32,
    /// Per-channel output multiplier (Q0.31).
    pub output_multiplier_per_channel: &'a [i32],
    /// Per-channel output right-shift.
    pub output_shift_per_channel: &'a [i32],
    /// Clamp lower bound for fused activation.
    pub quantized_activation_min: i32,
    /// Clamp upper bound for fused activation.
    pub quantized_activation_max: i32,
}

/// Parameters for DEPTHWISE_CONV_2D.
///
/// Mirrors TFLM `DepthwiseParams` (types.h) and
/// `TfLiteDepthwiseConvParams` (builtin_op_data.h).
#[derive(Clone, Debug)]
pub struct DepthwiseConv2DParams<'a> {
    /// Input tensor shape in NHWC.
    pub input_shape: [i32; 4],
    /// Depthwise filter shape: `[1, filter_height, filter_width, input_channels * depth_multiplier]`
    /// (1 output channel per filter entry, channel-multiplier semantics).
    pub filter_shape: [i32; 4],
    /// Output tensor shape in NHWC.
    pub output_shape: [i32; 4],
    pub padding: Padding,
    pub stride_width: i32,
    pub stride_height: i32,
    pub dilation_width_factor: i32,
    pub dilation_height_factor: i32,
    /// Channel multiplier (`output_channels = input_channels * depth_multiplier`).
    pub depth_multiplier: i32,
    pub input_offset: i32,
    pub weights_offset: i32,
    pub output_offset: i32,
    pub output_multiplier_per_channel: &'a [i32],
    pub output_shift_per_channel: &'a [i32],
    pub quantized_activation_min: i32,
    pub quantized_activation_max: i32,
}

/// Parameters for FULLY_CONNECTED.
///
/// Mirrors TFLM `FullyConnectedParams` (types.h) and
/// `TfLiteFullyConnectedParams` (builtin_op_data.h).
#[derive(Clone, Debug)]
pub struct FullyConnectedParams<'a> {
    /// Number of input elements (the accumulation depth).
    pub input_dim: i32,
    /// Number of output elements (must equal `output_multiplier_per_channel.len()`).
    pub output_dim: i32,
    pub input_offset: i32,
    pub weights_offset: i32,
    pub output_offset: i32,
    /// Per-channel output multiplier (Q0.31).
    pub output_multiplier_per_channel: &'a [i32],
    /// Per-channel output right-shift.
    pub output_shift_per_channel: &'a [i32],
    pub quantized_activation_min: i32,
    pub quantized_activation_max: i32,
}

/// Parameters for BATCH_MATMUL (used as MatMul in TFLite graphs).
///
/// Mirrors TFLM `TfLiteBatchMatMulParams` (builtin_op_data.h).
#[derive(Clone, Debug)]
pub struct MatMulParams {
    /// Rows of output matrix (= rows of A if `!adj_x`, else cols of A).
    pub m: i32,
    /// Columns of output matrix (= cols of B if `!adj_y`, else rows of B).
    pub n: i32,
    /// Inner dimension (= cols of A if `!adj_x`, else rows of A; = rows of B if `!adj_y`, else cols of B).
    pub k: i32,
    /// Whether to transpose the LHS (A) matrix.
    pub adj_x: bool,
    /// Whether to transpose the RHS (B) matrix.
    pub adj_y: bool,
    pub input_offset: i32,
    pub weights_offset: i32,
    pub output_offset: i32,
    /// Per-tensor (not per-channel) output multiplier.
    pub output_multiplier: i32,
    /// Per-tensor output right-shift.
    pub output_shift: i32,
    pub quantized_activation_min: i32,
    pub quantized_activation_max: i32,
}

// ── Tier1 — Pooling ────────────────────────────────────────────────────────

/// Parameters for AVERAGE_POOL_2D and MAX_POOL_2D.
///
/// Mirrors TFLM `PoolParams` (types.h) and `TfLitePoolParams`
/// (builtin_op_data.h).
#[derive(Clone, Debug)]
pub struct PoolParams {
    /// Input tensor shape in NHWC: `[batch=1, height, width, channels]`.
    pub input_shape: [i32; 4],
    /// Output tensor shape in NHWC.
    pub output_shape: [i32; 4],
    /// Filter spatial extent (width).
    pub filter_width: i32,
    /// Filter spatial extent (height).
    pub filter_height: i32,
    /// Stride along width.
    pub stride_width: i32,
    /// Stride along height.
    pub stride_height: i32,
    /// Padding strategy.
    pub padding: Padding,
    /// Fused activation applied after pooling.
    pub activation: FusedActivation,
    pub quantized_activation_min: i32,
    pub quantized_activation_max: i32,
}

// ── Tier1 — Softmax ────────────────────────────────────────────────────────

/// Parameters for SOFTMAX (int8-only path).
///
/// Mirrors TFLM `SoftmaxParams` (types.h), trimmed to the fields
/// needed by the int8 reference kernel (no LUT pointers, no float
/// `beta` — int8 softmax always uses beta=1.0).
#[derive(Clone, Debug)]
pub struct SoftmaxParams {
    /// Number of independent softmax rows (batch * height * width in
    /// the typical NHWC layout).
    pub num_rows: i32,
    /// Number of elements per softmax row (the channel dimension).
    pub row_size: i32,
    /// Multiplier applied to logits after max-subtract.
    pub input_multiplier: i32,
    /// Bit-shift applied to logits after max-subtract.
    pub input_left_shift: i32,
    /// Exponentials below `diff_min` are skipped (set to 0).
    pub diff_min: i32,
    /// Input zero-point.
    pub input_offset: i32,
    /// Output zero-point (int8 softmax output = i8::MIN).
    pub output_offset: i32,
    pub quantized_activation_min: i32,
    pub quantized_activation_max: i32,
}

// ── Tier1 — Standalone activations ─────────────────────────────────────────

/// Parameters for all standalone activation ops: RELU, RELU6, HARD_SWISH,
/// SIGMOID, TANH, LEAKY_RELU, PRELU.
///
/// Mirrors a union of TFLM `ReluParams`, `LeakyReluParams`,
/// `PreluParams`, `HardSwishParams`, `LogisticParams`, and
/// `TanhParams` (types.h).  Each op uses a subset of these fields.
#[derive(Clone, Debug)]
pub struct ActivationParams<'a> {
    /// Input zero-point offset (all ops).
    pub input_offset: i32,
    /// Output zero-point offset (relu family, leaky_relu, prelu).
    pub output_offset: i32,
    /// Output multiplier for the identity branch (relu/relu6/leaky_relu).
    pub output_multiplier: i32,
    /// Output shift for the identity branch.
    pub output_shift: i32,
    /// Activation clamp lower bound.
    pub quantized_activation_min: i32,
    /// Activation clamp upper bound.
    pub quantized_activation_max: i32,

    // ── sigmoid / tanh (LogisticParams / TanhParams) ──
    pub input_multiplier: i32,
    pub input_left_shift: i32,
    pub input_range_radius: i32,

    // ── leaky_relu (LeakyReluParams) ──
    pub output_multiplier_alpha: i32,
    pub output_shift_alpha: i32,
    pub output_multiplier_identity: i32,
    pub output_shift_identity: i32,

    // ── prelu (PreluParams) ──
    pub alpha_offset: i32,
    /// Per-channel alpha slope in int8 (Q7).
    pub alpha_data: &'a [i8],
    /// Multiplier for positive branch (output_multiplier_1).
    pub output_multiplier_1: i32,
    /// Shift for positive branch (output_shift_1).
    pub output_shift_1: i32,
    /// Multiplier for negative branch (output_multiplier_2).
    pub output_multiplier_2: i32,
    /// Shift for negative branch (output_shift_2).
    pub output_shift_2: i32,

    // ── hard_swish (HardSwishParams) ──
    /// `HardSwishParams::reluish_multiplier_fixedpoint_int16`
    pub reluish_multiplier_fixedpoint_int16: i16,
    pub reluish_multiplier_exponent: i32,
    /// `HardSwishParams::output_multiplier_fixedpoint_int16`
    pub output_multiplier_fixedpoint_int16: i16,
    pub output_multiplier_exponent: i32,
}

// ── Tier1 — Elementwise ────────────────────────────────────────────────────

/// Parameters for ADD, MUL, SUB.
///
/// Mirrors TFLM `ArithmeticParams` (types.h).
#[derive(Clone, Debug)]
pub struct ElementwiseParams {
    /// Total number of elements in each input and output slice
    /// (broadcast NOT supported — all three slices must have identical length).
    pub num_elements: i32,
    /// Input-1 zero-point offset.
    pub input1_offset: i32,
    /// Input-2 zero-point offset.
    pub input2_offset: i32,
    /// Output zero-point offset.
    pub output_offset: i32,
    /// Per-tensor output multiplier (Q0.31).
    pub output_multiplier: i32,
    /// Per-tensor output right-shift.
    pub output_shift: i32,
    /// Left-shift applied before requantize (add / sub).
    pub left_shift: i32,
    /// Input-1 multiplier (add / sub).
    pub input1_multiplier: i32,
    /// Input-1 shift (add / sub).
    pub input1_shift: i32,
    /// Input-2 multiplier (add / sub).
    pub input2_multiplier: i32,
    /// Input-2 shift (add / sub).
    pub input2_shift: i32,
    pub quantized_activation_min: i32,
    pub quantized_activation_max: i32,
}

// ── Tier2 — Data movement ──────────────────────────────────────────────────

/// Parameters for RESHAPE.
///
/// Mirrors TFLM `ReshapeParams` (types.h).
#[derive(Clone, Debug)]
pub struct ReshapeParams {
    /// Target shape (up to 4 dims per the plan's static-shape constraint).
    pub shape: [i32; 4],
    /// Number of valid dimensions in `shape`.
    pub shape_count: i8,
}

/// Parameters for TRANSPOSE.
///
/// Mirrors TFLM `TransposeParams` (types.h).
#[derive(Clone, Debug)]
pub struct TransposeParams {
    /// Input tensor shape in NHWC: `[batch=1, height, width, channels]`.
    pub input_shape: [i32; 4],
    /// Permutation array.
    pub perm: [i32; 8],
    /// Number of valid entries in `perm`.
    pub perm_count: i8,
}

/// Parameters for CONCATENATION.
///
/// Mirrors TFLM `ConcatenationParams` (types.h) and
/// `TfLiteConcatenationParams` (builtin_op_data.h).
#[derive(Clone, Debug)]
pub struct ConcatParams {
    /// Axis along which to concatenate.
    pub axis: i32,
    /// Fused activation (typically None for concat).
    pub activation: FusedActivation,
    /// Shape of the first input tensor in NHWC.
    pub input_shape_a: [i32; 4],
    /// Shape of the second input tensor in NHWC.
    pub input_shape_b: [i32; 4],
    /// Shape of the output tensor in NHWC.
    pub output_shape: [i32; 4],
}

/// Parameters for SPLIT.
///
/// Mirrors TFLM `SplitParams` (types.h) and
/// `TfLiteSplitParams` / `TfLiteSplitVParams` (builtin_op_data.h).
#[derive(Clone, Debug)]
pub struct SplitParams {
    /// Number of output slices.
    pub num_splits: i32,
    /// Axis along which to split.
    pub axis: i32,
    /// Shape of the input tensor in NHWC.
    pub input_shape: [i32; 4],
    /// Shape of the first output slice in NHWC.
    pub output_shape_a: [i32; 4],
    /// Shape of the second output slice in NHWC.
    pub output_shape_b: [i32; 4],
}

/// Parameters for PAD.
///
/// Mirrors TFLM `PadParams` (types.h).  `TfLitePadParams` /
/// `TfLitePadV2Params` are empty; the pad sizes come from a runtime
/// input tensor, so they are represented here as fixed arrays.
#[derive(Clone, Debug)]
pub struct PadParams {
    /// Input tensor shape in NHWC.
    pub input_shape: [i32; 4],
    /// Output tensor shape in NHWC.
    pub output_shape: [i32; 4],
    /// Number of prepended elements per dimension (top / left style).
    pub left_padding: [i32; 4],
    /// Count of valid entries in `left_padding`.
    pub left_padding_count: i8,
    /// Number of appended elements per dimension.
    pub right_padding: [i32; 4],
    /// Count of valid entries in `right_padding`.
    pub right_padding_count: i8,
}

/// Parameters for SLICE.
///
/// Mirrors TFLM `SliceParams` (types.h).
#[derive(Clone, Debug)]
pub struct SliceParams {
    /// Input tensor shape in NHWC.
    pub input_shape: [i32; 4],
    /// Start indices.
    pub begin: [i32; 4],
    /// Slice sizes.
    pub size: [i32; 4],
}

/// Parameters for RESIZE_NEAREST_NEIGHBOR.
///
/// Mirrors TFLM `ResizeNearestNeighborParams` (types.h) and
/// `TfLiteResizeNearestNeighborParams` (builtin_op_data.h).
#[derive(Clone, Debug)]
pub struct ResizeNearestParams {
    /// Input tensor shape in NHWC.
    pub input_shape: [i32; 4],
    /// Output tensor shape in NHWC.
    pub output_shape: [i32; 4],
    /// Align corners mode (bool, stored as i32 for compatibility with
    /// TFLM flatbuffer encoding).
    pub align_corners: i32,
    /// Half-pixel-centers mode.
    pub half_pixel_centers: i32,
}

// ── Tier3 — Recurrent ──────────────────────────────────────────────────────

/// Parameters for UNIDIRECTIONAL_SEQUENCE_LSTM.
///
/// Combines TFLM `TfLiteUnidirectionalSequenceLSTMParams`
/// (builtin_op_data.h) and `LstmCellParams` (types.h).
///
/// Note: `cell_clip` and `proj_clip` from upstream are `f32` fields
/// that the int8 inference path never reads (clipping in the integer
/// path is an integer clamp).  They are excluded per the plan's
/// Must-NOT-Have list (no f32 compute in device code).
#[derive(Clone, Debug)]
pub struct LstmParams {
    /// Fused activation for input / forget / output gates.
    pub activation: FusedActivation,
    /// If true, first dimension is time (not batch).
    pub time_major: bool,
    /// Number of LSTM hidden units (must equal cell_state / hidden_state length).
    pub num_units: i32,
    /// Input dimension per timestep.
    pub input_dim: i32,
    /// Weights zero-point for the gate kernels.
    pub weights_zero_point: i32,
    /// Accumulator multiplier.
    pub accum_multiplier: i32,
    /// Accumulator shift.
    pub accum_shift: i32,
    /// State integer bits for fixed-point cell state.
    pub state_integer_bits: i32,
}

/// Parameters for SVDF (Singular Value Decomposition Filter).
///
/// Mirrors TFLM `TfLiteSVDFParams` (builtin_op_data.h).
#[derive(Clone, Debug)]
pub struct SvdfParams {
    /// Rank (number of SVDF time-frames, typ. 1).
    pub rank: i32,
    /// Fused activation applied to output.
    pub activation: FusedActivation,
    /// Number of SVDF filters (the inner dimension of the feature-weight
    /// matrix).
    pub num_filters: i32,
}

/// Parameters for GRU (Gated Recurrent Unit).
///
/// TFLM has no GRU kernel; GRU goldens come from an `embedded-nn`
/// cross-check.  This struct encodes the gate quantization and shape
/// parameters needed by a custom GRU implementation.
#[derive(Clone, Debug)]
pub struct GruParams {
    /// Hidden-state dimension.
    pub num_units: i32,
    /// Input dimension.
    pub input_size: i32,
    /// Activation for update / reset gates (typically sigmoid).
    pub gate_activation: FusedActivation,
    /// Activation for candidate hidden state (typically tanh).
    pub candidate_activation: FusedActivation,
}

// ── Tier4 — Reductions ─────────────────────────────────────────────────────

/// Parameters for reduction ops: MEAN, SUM, REDUCE_MAX, REDUCE_MIN,
/// ARG_MAX, ARG_MIN, L2_NORMALIZATION.
///
/// Combines TFLM `MeanParams` (types.h) and
/// `TfLiteReducerParams` (builtin_op_data.h).
#[derive(Clone, Debug)]
pub struct ReduceParams {
    /// Whether to keep reduced dimensions as size-1.
    pub keep_dims: bool,
    /// Reduction axes (up to 4 axes for 4D tensors).
    pub axis: [i16; 4],
    /// Number of valid axes in `axis`.
    pub axis_count: i8,
    /// Input tensor shape in NHWC.
    pub input_shape: [i32; 4],
    /// For ARG_MAX / ARG_MIN: the output element type (`kTfLiteInt32`
    /// or `kTfLiteInt64`).
    pub output_type: i32,
}
