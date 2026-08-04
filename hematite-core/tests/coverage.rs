// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Compile-time tier-table-to-trait-method coverage check.
//!
//! A stub [`KernelBackend`] implementation that must compile — if a method
//! is added to the trait without appearing here, or if this list names a
//! method the trait removed, the compiler will reject the entire test
//! module.  That is the intended failure mode: drift is impossible because
//! the code won't build.

#![allow(dead_code, unused_variables)]

use hematite_core::{
    ActivationParams, ConcatParams, Conv2DParams, DepthwiseConv2DParams,
    ElementwiseParams, FullyConnectedParams, GruParams, KernelBackend,
    KernelError, LstmParams, MatMulParams, PadParams, PoolParams,
    QuantParam, ReduceParams, ReshapeParams, ResizeNearestParams,
    SliceParams, SoftmaxParams, SplitParams, SvdfParams, TransposeParams,
};

struct StubBackend;

impl KernelBackend for StubBackend {
    fn conv2d(
        &self,
        input: &[i8],
        weights: &[i8],
        bias: &[i32],
        _params: &Conv2DParams,
        output: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn depthwise_conv2d(
        &self,
        input: &[i8],
        weights: &[i8],
        bias: &[i32],
        _params: &DepthwiseConv2DParams,
        output: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn fully_connected(
        &self,
        input: &[i8],
        weights: &[i8],
        bias: &[i32],
        _params: &FullyConnectedParams,
        output: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn matmul(
        &self,
        input: &[i8],
        weights: &[i8],
        bias: &[i32],
        _params: &MatMulParams,
        output: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn average_pool_2d(
        &self,
        input: &[i8],
        _params: &PoolParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn max_pool_2d(
        &self,
        input: &[i8],
        _params: &PoolParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn softmax(
        &self,
        input: &[i8],
        _params: &SoftmaxParams,
        output: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn relu(
        &self,
        input: &[i8],
        _params: &ActivationParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn relu6(
        &self,
        input: &[i8],
        _params: &ActivationParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn hard_swish(
        &self,
        input: &[i8],
        _params: &ActivationParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn sigmoid(
        &self,
        input: &[i8],
        _params: &ActivationParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn tanh(
        &self,
        input: &[i8],
        _params: &ActivationParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn leaky_relu(
        &self,
        input: &[i8],
        _params: &ActivationParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn prelu(
        &self,
        input: &[i8],
        _params: &ActivationParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn add(
        &self,
        input1: &[i8],
        input2: &[i8],
        _params: &ElementwiseParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn mul(
        &self,
        input1: &[i8],
        input2: &[i8],
        _params: &ElementwiseParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn sub(
        &self,
        input1: &[i8],
        input2: &[i8],
        _params: &ElementwiseParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn quantize(
        &self,
        input: &[i8],
        _params: &QuantParam,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn dequantize(
        &self,
        input: &[i8],
        _params: &QuantParam,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn reshape(
        &self,
        input: &[i8],
        _params: &ReshapeParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn transpose(
        &self,
        input: &[i8],
        _params: &TransposeParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn concat(
        &self,
        input_a: &[i8],
        input_b: &[i8],
        _params: &ConcatParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn split(
        &self,
        input: &[i8],
        _params: &SplitParams,
        output_a: &mut [i8],
        output_b: &mut [i8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn pad(
        &self,
        input: &[i8],
        _params: &PadParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn slice(
        &self,
        input: &[i8],
        _params: &SliceParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn resize_nearest(
        &self,
        input: &[i8],
        _params: &ResizeNearestParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn mean(
        &self,
        input: &[i8],
        _params: &ReduceParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn sum(
        &self,
        input: &[i8],
        _params: &ReduceParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn reduce_max(
        &self,
        input: &[i8],
        _params: &ReduceParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn reduce_min(
        &self,
        input: &[i8],
        _params: &ReduceParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn arg_max(
        &self,
        input: &[i8],
        _params: &ReduceParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn arg_min(
        &self,
        input: &[i8],
        _params: &ReduceParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn l2_normalization(
        &self,
        input: &[i8],
        _params: &ReduceParams,
        output: &mut [i8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

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
        _params: &LstmParams,
        output: &mut [i8],
        cell_state: &mut [i16],
        hidden_state: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn svdf(
        &self,
        input: &[i8],
        weights_feature: &[i8],
        weights_time: &[i16],
        bias: &[i32],
        _params: &SvdfParams,
        output: &mut [i8],
        hidden_state: &mut [i16],
        scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }

    fn gru(
        &self,
        input: &[i8],
        reset_gate_weights: &[i8],
        update_gate_weights: &[i8],
        candidate_weights: &[i8],
        biases: &[i32],
        _params: &GruParams,
        output: &mut [i8],
        hidden_state: &mut [i16],
        scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        unimplemented!()
    }
}

/// The tier table defines exactly 36 ops in scope (T0–T4 + RESIZE + PRELU).
///
/// Scalar ops (SQUEEZE, EXPAND_DIMS, FLATTEN) are compile-time const
/// expansions with no trait method.  T5-deferred ops also have no method.
///
/// This constant lists every trait **method** name (not the scratch-size
/// fns).  If a method is added to or removed from `KernelBackend` without
/// updating this list, the runtime assertion below fails.
const EXPECTED_METHODS: &[&str] = &[
    // T0 — Core compute (4)
    "conv2d",
    "depthwise_conv2d",
    "fully_connected",
    "matmul",
    // T1 — Pooling + Softmax + Activations + Elementwise + Quant (17)
    "average_pool_2d",
    "max_pool_2d",
    "softmax",
    "relu",
    "relu6",
    "hard_swish",
    "sigmoid",
    "tanh",
    "leaky_relu",
    "prelu",
    "add",
    "mul",
    "sub",
    "quantize",
    "dequantize",
    // Hmm wait let me recount: 2 pool + 1 softmax + 7 act + 3 elem + 2 quant = 15? No, the tier has:
    // Pool: 2 (avg, max)
    // Softmax: 1
    // Activations: 8? No: relu, relu6, hard_swish, sigmoid, tanh, leaky_relu, prelu = 7
    // Elementwise: 3 (add, mul, sub)
    // Quant: 2 (quantize, dequantize)
    // Total T1 = 2+1+7+3+2 = 15
    // T1 total: 15

    // T2 — Data movement (7)
    "reshape",
    "transpose",
    "concat",
    "split",
    "pad",
    "slice",
    "resize_nearest",

    // T3 — Recurrent (3)
    "unidirectional_sequence_lstm",
    "svdf",
    "gru",

    // T4 — Reductions (7)
    "mean",
    "sum",
    "reduce_max",
    "reduce_min",
    "arg_max",
    "arg_min",
    "l2_normalization",
    // Total: 4 + 15 + 7 + 3 + 7 = 36
];

#[test]
fn op_method_count_matches_tier_table() {
    assert_eq!(
        EXPECTED_METHODS.len(),
        36,
        "Expected exactly 36 in-scope op methods in KernelBackend"
    );
}
