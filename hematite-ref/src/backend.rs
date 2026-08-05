// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! [`RefBackend`] — the scalar int8 reference backend.
//!
//! A thin adapter that implements [`KernelBackend`] by forwarding every
//! trait method to the standalone scalar reference kernel in this crate
//! (see the per-op modules: [`conv`], [`pool`], [`activation`], ...).
//!
//! This adapter is the trait-level "wiring" that lets model-inference code
//! generic over `B: KernelBackend` run against the scalar reference. It was
//! added by T5.1 ("Per-op TDD test crate — all ops wired"); every kernel's
//! math lives in its own module and is not touched here.
//!
//! # Ops NOT wired (return [`KernelError::Unsupported`])
//!
//! | Trait method            | Reason                                                                 |
//! |-------------------------|------------------------------------------------------------------------|
//! | `matmul`                | No scalar kernel and no golden fixture exist (T0.5 corpus gap).         |
//! | `sigmoid` / `tanh`      | No scalar kernel and no golden fixture exist (not in T0.5 corpus).      |
//! | `reduce_max`/`reduce_min` | No scalar kernel and no golden fixture exist (T2.5 generated only mean/sum/argmax/argmin/l2_norm). |
//! | `unidirectional_sequence_lstm` / `svdf` / `gru` | The trait signatures cannot carry the fixture-specific fixed-point quant constants (gate/cell/output multiplier+shift pairs) that the scalar recurrent kernels require; these live in the per-op tests, not in `LstmParams`/`SvdfParams`/`GruParams`. The recurrent kernels are still bit-exact tested via direct free-function calls in `hematite-tests/tests/`. |
//!
//! # Trait-signature adaptations
//!
//! * **`relu6`** — the scalar kernel takes `quantized_six` as an extra
//!   parameter (`QUANTIZED_SIX` is not a field of [`ActivationParams`]).
//!   The trait method has no such parameter, so the adapter forwards
//!   `params.quantized_activation_max` as the clamp bound. Callers must set
//!   `quantized_activation_max = QUANTIZED_SIX` when building params.
//! * **`split`** — the trait splits into two output slices in one call; the
//!   scalar kernel writes one output per `split_index`. The adapter forwards
//!   `output_a` with `split_index = 0` and `output_b` with `split_index = 1`.
//! * **`reshape` / `transpose`** — no standalone scalar kernel exists yet
//!   (Phase-2 `data_movement.rs` only covers concat/split/pad/slice). Both are
//!   pure data movement with no arithmetic, so the adapter implements them
//!   inline (see [`transpose_impl`]). These should be promoted to
//!   standalone kernels in `data_movement.rs` by a Phase-2 completion task.

use hematite_core::op_params::{
    ActivationParams, ConcatParams, Conv2DParams, DepthwiseConv2DParams,
    ElementwiseParams, FullyConnectedParams, GruParams, LstmParams,
    MatMulParams, PadParams, PoolParams, QuantParam, ReduceParams,
    ReshapeParams, ResizeNearestParams, SliceParams, SoftmaxParams,
    SplitParams, SvdfParams, TransposeParams,
};
use hematite_core::{KernelBackend, KernelError};

use crate::activation;
use crate::conv;
use crate::data_movement;
use crate::depthwise_conv;
use crate::elementwise;
use crate::fully_connected;
use crate::pool;
use crate::reductions;
use crate::resize;
use crate::softmax;

/// The scalar int8 reference backend.
///
/// Stateless unit struct — all kernels are pure functions of their slices
/// and params. Const-constructible: `let backend = RefBackend;`.
pub struct RefBackend;

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

/// General 4D transpose (scatter formulation).
///
/// `output[coords[perm[0]], coords[perm[1]], coords[perm[2]], coords[perm[3]]] =
/// input[coords]` for every input position in NHWC row-major order — the
/// same `output[i] = input[perm applied to coords]` mapping TFLM uses.
///
/// `perm` entries beyond `perm_count` default to identity. This is a pure
/// data-movement operation: no arithmetic, no offset, no requantize.
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

    // Output shape is the input shape permuted.
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

impl KernelBackend for RefBackend {
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
        conv::conv2d(input, weights, bias, params, output, scratch)
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
        depthwise_conv::depthwise_conv2d(input, weights, bias, params, output, scratch)
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
        fully_connected::fully_connected(input, weights, bias, params, output, scratch)
    }

    /// Not wired: no scalar `matmul` kernel and no golden fixture exist
    /// (see module docs). Model-level codegen does not emit MatMul for the
    /// in-scope zoo models (fully_connected covers the dense case).
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
        activation::relu(input, params, output, &mut [])
    }

    /// Forwards `params.quantized_activation_max` as the ReLU6 clamp bound —
    /// see module docs ("Trait-signature adaptations").
    fn relu6(
        &self,
        input: &[i8],
        params: &ActivationParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        activation::relu6(input, params, output, &mut [], params.quantized_activation_max)
    }

    fn hard_swish(
        &self,
        input: &[i8],
        params: &ActivationParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        activation::hard_swish(input, params, output, &mut [])
    }

    /// Not wired: no scalar sigmoid kernel and no golden fixture exist.
    fn sigmoid(
        &self,
        _input: &[i8],
        _params: &ActivationParams,
        _output: &mut [i8],
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported)
    }

    /// Not wired: no scalar tanh kernel and no golden fixture exist.
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
        input: &[i8],
        params: &QuantParam,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        elementwise::quantize(input, params, output, &mut [])
    }

    fn dequantize(
        &self,
        input: &[i8],
        params: &QuantParam,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        elementwise::dequantize(input, params, output, &mut [])
    }

    // ── Tier2 — Data movement ───────────────────────────────────────────

    /// Reshape is a flat copy: TFLM's int8 `Reshape` is a metadata-only op
    /// (same underlying buffer, new logical shape). Implemented inline here —
    /// see module docs.
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

    /// Transpose — general 4D permutation. Implemented inline here — see
    /// module docs.
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

    /// Splits both output slices in one call — see module docs
    /// ("Trait-signature adaptations").
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

    /// Not wired: the trait signature cannot carry the fixture-specific
    /// cell-tanh / output fixed-point quant constants the scalar LSTM kernel
    /// requires (see module docs). Tested bit-exact via direct free-function
    /// calls in `hematite-tests/tests/lstm.rs`.
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

    /// Not wired: same trait-signature gap as LSTM — the scalar `svdf_step`
    /// requires quant constants not carried by [`SvdfParams`]. Tested
    /// bit-exact via direct free-function calls in
    /// `hematite-tests/tests/svdf.rs`.
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

    /// Not wired: same trait-signature gap — the scalar GRU kernel requires
    /// quant constants not carried by [`GruParams`]. Tested bit-exact via
    /// direct free-function calls in `hematite-tests/tests/gru.rs`.
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
        reductions::sum(input, params, output)
    }

    /// Not wired: no scalar `reduce_max` kernel and no golden fixture exist
    /// (T2.5 generated only mean/sum/argmax/argmin/l2_norm).
    fn reduce_max(
        &self,
        _input: &[i8],
        _params: &ReduceParams,
        _output: &mut [i8],
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported)
    }

    /// Not wired: no scalar `reduce_min` kernel and no golden fixture exist
    /// (same T2.5 corpus gap as `reduce_max`).
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
        input: &[i8],
        params: &ReduceParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        reductions::arg_max(input, params, output)
    }

    fn arg_min(
        &self,
        input: &[i8],
        params: &ReduceParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        reductions::arg_min(input, params, output)
    }

    fn l2_normalization(
        &self,
        input: &[i8],
        params: &ReduceParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        reductions::l2_norm(input, params, output)
    }
}
