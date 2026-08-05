// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Macro smoke test — applies `#[model]` to a struct referencing sine.tflite
//! and runs the generated `Model<B>` against a mock `KernelBackend`.
//!
//! The macro expansion compiles as part of this integration test (the only
//! way to compile+run generated code from a proc-macro crate), then
//! `predict` dispatches the straight-line op sequence through the mock.

use std::cell::RefCell;
use std::rc::Rc;

use hematite_codegen::model;
use hematite_core::op_params::{
    ActivationParams, ConcatParams, Conv2DParams, DepthwiseConv2DParams,
    ElementwiseParams, FullyConnectedParams, GruParams, LstmParams, MatMulParams,
    PadParams, PoolParams, QuantParam, ReduceParams, ReshapeParams,
    ResizeNearestParams, SliceParams, SoftmaxParams, SplitParams, SvdfParams,
    TransposeParams,
};
use hematite_core::{KernelBackend, KernelError};

/// Smoke test struct annotated with the `#[model]` proc-macro.
///
/// Path is relative to the tests/ crate's `CARGO_MANIFEST_DIR` (the
/// `hematite-codegen/` directory), so `../models/sine.tflite` resolves to the
/// workspace `models/` directory.
#[model("../models/sine.tflite")]
pub struct SineModel;

/// A recorded op call: method name plus the tensor slice lengths the kernel
/// was handed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Call {
    op: &'static str,
    in_len: usize,
    out_len: usize,
}

/// Mock backend: records every dispatched call; only `fully_connected`
/// computes (TFLM int8 semantics) so the sine model is verified numerically.
#[derive(Default)]
struct MockBackend {
    log: Rc<RefCell<Vec<Call>>>,
}

/// TFLM `MultiplyByQuantizedMultiplier` — CMSIS single-rounding (mirrors
/// `hematite-int8`), used by the mock's fully_connected.
fn mbm(value: i32, multiplier: i32, shift: i32) -> i32 {
    let total_shift = 31i64 - i64::from(shift);
    let round = 1i64 << (total_shift - 1);
    let result = (i64::from(value) * i64::from(multiplier) + round) >> total_shift;
    result.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

/// TFLM int8 fully-connected reference loop (bias-init acc, MAC with input
/// offset, per-channel requantize, output offset, clamp).
fn fc_math(
    input: &[i8],
    weights: &[i8],
    bias: &[i32],
    params: &FullyConnectedParams,
    output: &mut [i8],
) -> Result<(), KernelError> {
    let input_dim = params.input_dim as usize;
    let output_dim = params.output_dim as usize;
    if input.len() != input_dim
        || weights.len() != output_dim * input_dim
        || bias.len() != output_dim
        || output.len() != output_dim
    {
        return Err(KernelError::ShapeMismatch);
    }
    for oc in 0..output_dim {
        let mut acc: i32 = bias[oc];
        for d in 0..input_dim {
            acc += (i32::from(input[d]) + params.input_offset) * i32::from(weights[oc * input_dim + d]);
        }
        let scaled = mbm(
            acc,
            params.output_multiplier_per_channel[oc],
            params.output_shift_per_channel[oc],
        );
        let with_off = scaled + params.output_offset;
        let clamped = with_off.clamp(params.quantized_activation_min, params.quantized_activation_max);
        output[oc] = clamped as i8;
    }
    Ok(())
}

/// `macro_rules` over the 36-method `KernelBackend` trait: every method
/// records its name; `fully_connected` additionally runs the real int8 math.
macro_rules! mock_backend {
    ($ty:ident) => {
        impl KernelBackend for $ty {
            fn fully_connected(
                &self,
                input: &[i8],
                weights: &[i8],
                bias: &[i32],
                params: &FullyConnectedParams,
                output: &mut [i8],
                _scratch: &mut [u8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "fully_connected", in_len: input.len(), out_len: output.len() });
                fc_math(input, weights, bias, params, output)
            }
            fn conv2d(
                &self,
                input: &[i8],
                _weights: &[i8],
                _bias: &[i32],
                _params: &Conv2DParams,
                output: &mut [i8],
                _scratch: &mut [u8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "conv2d", in_len: input.len(), out_len: output.len() });
                Ok(())
            }
            fn depthwise_conv2d(
                &self,
                input: &[i8],
                _weights: &[i8],
                _bias: &[i32],
                _params: &DepthwiseConv2DParams,
                output: &mut [i8],
                _scratch: &mut [u8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "depthwise_conv2d", in_len: input.len(), out_len: output.len() });
                Ok(())
            }
            fn matmul(
                &self,
                input: &[i8],
                _weights: &[i8],
                _bias: &[i32],
                _params: &MatMulParams,
                output: &mut [i8],
                _scratch: &mut [u8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "matmul", in_len: input.len(), out_len: output.len() });
                Ok(())
            }
            fn average_pool_2d(
                &self,
                input: &[i8],
                _params: &PoolParams,
                output: &mut [i8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "average_pool_2d", in_len: input.len(), out_len: output.len() });
                Ok(())
            }
            fn max_pool_2d(
                &self,
                input: &[i8],
                _params: &PoolParams,
                output: &mut [i8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "max_pool_2d", in_len: input.len(), out_len: output.len() });
                Ok(())
            }
            fn softmax(
                &self,
                input: &[i8],
                _params: &SoftmaxParams,
                output: &mut [i8],
                _scratch: &mut [u8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "softmax", in_len: input.len(), out_len: output.len() });
                Ok(())
            }
            fn relu(
                &self,
                input: &[i8],
                _params: &ActivationParams,
                output: &mut [i8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "relu", in_len: input.len(), out_len: output.len() });
                Ok(())
            }
            fn relu6(
                &self,
                input: &[i8],
                _params: &ActivationParams,
                output: &mut [i8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "relu6", in_len: input.len(), out_len: output.len() });
                Ok(())
            }
            fn hard_swish(
                &self,
                input: &[i8],
                _params: &ActivationParams,
                output: &mut [i8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "hard_swish", in_len: input.len(), out_len: output.len() });
                Ok(())
            }
            fn sigmoid(
                &self,
                input: &[i8],
                _params: &ActivationParams,
                output: &mut [i8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "sigmoid", in_len: input.len(), out_len: output.len() });
                Ok(())
            }
            fn tanh(
                &self,
                input: &[i8],
                _params: &ActivationParams,
                output: &mut [i8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "tanh", in_len: input.len(), out_len: output.len() });
                Ok(())
            }
            fn leaky_relu(
                &self,
                input: &[i8],
                _params: &ActivationParams,
                output: &mut [i8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "leaky_relu", in_len: input.len(), out_len: output.len() });
                Ok(())
            }
            fn prelu(
                &self,
                input: &[i8],
                _params: &ActivationParams,
                output: &mut [i8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "prelu", in_len: input.len(), out_len: output.len() });
                Ok(())
            }
            fn add(
                &self,
                input1: &[i8],
                input2: &[i8],
                _params: &ElementwiseParams,
                output: &mut [i8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "add", in_len: input1.len().max(input2.len()), out_len: output.len() });
                Ok(())
            }
            fn mul(
                &self,
                input1: &[i8],
                input2: &[i8],
                _params: &ElementwiseParams,
                output: &mut [i8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "mul", in_len: input1.len().max(input2.len()), out_len: output.len() });
                Ok(())
            }
            fn sub(
                &self,
                input1: &[i8],
                input2: &[i8],
                _params: &ElementwiseParams,
                output: &mut [i8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "sub", in_len: input1.len().max(input2.len()), out_len: output.len() });
                Ok(())
            }
            fn quantize(
                &self,
                input: &[i8],
                _params: &QuantParam,
                output: &mut [i8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "quantize", in_len: input.len(), out_len: output.len() });
                Ok(())
            }
            fn dequantize(
                &self,
                input: &[i8],
                _params: &QuantParam,
                output: &mut [i8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "dequantize", in_len: input.len(), out_len: output.len() });
                Ok(())
            }
            fn reshape(
                &self,
                input: &[i8],
                _params: &ReshapeParams,
                output: &mut [i8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "reshape", in_len: input.len(), out_len: output.len() });
                Ok(())
            }
            fn transpose(
                &self,
                input: &[i8],
                _params: &TransposeParams,
                output: &mut [i8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "transpose", in_len: input.len(), out_len: output.len() });
                Ok(())
            }
            fn concat(
                &self,
                input_a: &[i8],
                input_b: &[i8],
                _params: &ConcatParams,
                output: &mut [i8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "concat", in_len: input_a.len() + input_b.len(), out_len: output.len() });
                Ok(())
            }
            fn split(
                &self,
                input: &[i8],
                _params: &SplitParams,
                output_a: &mut [i8],
                output_b: &mut [i8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "split", in_len: input.len(), out_len: output_a.len() + output_b.len() });
                Ok(())
            }
            fn pad(
                &self,
                input: &[i8],
                _params: &PadParams,
                output: &mut [i8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "pad", in_len: input.len(), out_len: output.len() });
                Ok(())
            }
            fn slice(
                &self,
                input: &[i8],
                _params: &SliceParams,
                output: &mut [i8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "slice", in_len: input.len(), out_len: output.len() });
                Ok(())
            }
            fn resize_nearest(
                &self,
                input: &[i8],
                _params: &ResizeNearestParams,
                output: &mut [i8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "resize_nearest", in_len: input.len(), out_len: output.len() });
                Ok(())
            }
            fn mean(
                &self,
                input: &[i8],
                _params: &ReduceParams,
                output: &mut [i8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "mean", in_len: input.len(), out_len: output.len() });
                Ok(())
            }
            fn sum(
                &self,
                input: &[i8],
                _params: &ReduceParams,
                output: &mut [i8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "sum", in_len: input.len(), out_len: output.len() });
                Ok(())
            }
            fn reduce_max(
                &self,
                input: &[i8],
                _params: &ReduceParams,
                output: &mut [i8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "reduce_max", in_len: input.len(), out_len: output.len() });
                Ok(())
            }
            fn reduce_min(
                &self,
                input: &[i8],
                _params: &ReduceParams,
                output: &mut [i8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "reduce_min", in_len: input.len(), out_len: output.len() });
                Ok(())
            }
            fn arg_max(
                &self,
                input: &[i8],
                _params: &ReduceParams,
                output: &mut [i8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "arg_max", in_len: input.len(), out_len: output.len() });
                Ok(())
            }
            fn arg_min(
                &self,
                input: &[i8],
                _params: &ReduceParams,
                output: &mut [i8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "arg_min", in_len: input.len(), out_len: output.len() });
                Ok(())
            }
            fn l2_normalization(
                &self,
                input: &[i8],
                _params: &ReduceParams,
                output: &mut [i8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "l2_normalization", in_len: input.len(), out_len: output.len() });
                Ok(())
            }
            fn unidirectional_sequence_lstm(
                &self,
                input: &[i8],
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
                output: &mut [i8],
                _cell_state: &mut [i16],
                _hidden_state: &mut [i8],
                _scratch: &mut [u8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "unidirectional_sequence_lstm", in_len: input.len(), out_len: output.len() });
                Ok(())
            }
            fn svdf(
                &self,
                input: &[i8],
                _weights_feature: &[i8],
                _weights_time: &[i16],
                _bias: &[i32],
                _params: &SvdfParams,
                output: &mut [i8],
                _hidden_state: &mut [i16],
                _scratch: &mut [u8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "svdf", in_len: input.len(), out_len: output.len() });
                Ok(())
            }
            fn gru(
                &self,
                input: &[i8],
                _reset_gate_weights: &[i8],
                _update_gate_weights: &[i8],
                _candidate_weights: &[i8],
                _biases: &[i32],
                _params: &GruParams,
                output: &mut [i8],
                _hidden_state: &mut [i16],
                _scratch: &mut [u8],
            ) -> Result<(), KernelError> {
                self.log.borrow_mut().push(Call { op: "gru", in_len: input.len(), out_len: output.len() });
                Ok(())
            }
        }
    };
}

mock_backend!(MockBackend);

#[test]
fn sine_model_predict_with_mock_backend() {
    let log = Rc::new(RefCell::new(Vec::new()));
    let backend = MockBackend { log: log.clone() };
    let model = Model::new(backend);

    assert_eq!(Model::<MockBackend>::input_len(), 1);
    assert_eq!(Model::<MockBackend>::output_len(), 1);

    // Numeric: output = round((51 * x - 3) / 128) with CMSIS single-rounding.
    assert_eq!(model.predict(&[2]), [1]);
    assert_eq!(model.predict(&[0]), [0]);
    assert_eq!(model.predict(&[5]), [2]);

    // Call sequence: exactly one op, dispatched through the backend.
    let calls = log.borrow();
    assert_eq!(calls.len(), 3, "expected 3 fully_connected calls, got {calls:?}");
    assert!(calls.iter().all(|c| c.op == "fully_connected"));
    assert_eq!(calls[0].in_len, 1);
    assert_eq!(calls[0].out_len, 1);
    drop(calls);

    // predict_with_scratch: sized scratch array, caller-provided output.
    let mut out_buf = [0i8; 1];
    let mut scratch = [0u8; 0];
    let r = model.predict_with_scratch(&[5], &mut out_buf, &mut scratch);
    assert_eq!(r, Ok(()));
    assert_eq!(out_buf, [2]);
}

#[test]
fn sine_model_compiles() {
    // If the macro expanded without compile_error, compilation itself is the
    // test; this function ensures the test runner has something to execute.
    let _ = SineModel;
}
