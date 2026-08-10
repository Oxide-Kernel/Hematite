// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Composed-vs-decomposed equality tests (T2.1) for the RefBackend
//! `FusedKernelBackend` default decompositions.
//!
//! Each test runs a composed fused call against the equivalent explicit
//! per-op sequence on the SAME real kernels and asserts element equality —
//! the decompositions must be bit-exact by construction.  Every case uses
//! NON-identity scales / offsets so a mis-wired field mapping (wrong
//! offset, wrong order, broken in-place aliasing) changes the output.
//!
//! Test naming convention: `fused_<op>_<case>` so that
//! `cargo test -p hematite-ref -- fused_` matches all tests.

use hematite_core::op_params::{
    ActivationEpilogueParams, ActivationParams, ComposedActivation,
    Conv2DParams, ElementwiseChainParams, ElementwiseChainStep,
    ElementwiseKind, ElementwiseParams, FoldedPoolParams, FusedConvParams,
    Padding, PoolInputFold, PoolKind, PoolParams, ResidualAddParams,
};
use hematite_core::{FusedKernelBackend, KernelBackend};
use hematite_ref::RefBackend;

// ── Helpers ────────────────────────────────────────────────────────────────

/// Build the standalone-activation params the unfused emitter would emit
/// for a relu / relu6 / hard_swish op.
fn act(
    input_offset: i32,
    output_offset: i32,
    output_multiplier: i32,
    output_shift: i32,
    min: i32,
    max: i32,
) -> ActivationParams<'static> {
    ActivationParams {
        input_offset,
        output_offset,
        output_multiplier,
        output_shift,
        quantized_activation_min: min,
        quantized_activation_max: max,
        input_multiplier: 0,
        input_left_shift: 0,
        input_range_radius: 0,
        output_multiplier_alpha: 0,
        output_shift_alpha: 0,
        output_multiplier_identity: 0,
        output_shift_identity: 0,
        alpha_offset: 0,
        alpha_data: &[],
        output_multiplier_1: 0,
        output_shift_1: 0,
        output_multiplier_2: 0,
        output_shift_2: 0,
        reluish_multiplier_fixedpoint_int16: 0,
        reluish_multiplier_exponent: 0,
        output_multiplier_fixedpoint_int16: 0,
        output_multiplier_exponent: 0,
    }
}

fn assert_eq_i8(composed: &[i8], per_op: &[i8]) {
    assert_eq!(
        composed.len(),
        per_op.len(),
        "length mismatch: composed {} vs per-op {}",
        composed.len(),
        per_op.len()
    );
    for (i, (c, p)) in composed.iter().zip(per_op.iter()).enumerate() {
        assert_eq!(
            c, p,
            "element {i} differs: composed {c} vs per-op {p} (not bit-exact)"
        );
    }
}

// ── Test 1: 4-op elementwise chain (anchor Add + relu + mul + hard_swish) ──

#[test]
fn fused_chain_add_relu_mul_hardswish() {
    let n: i32 = 16;
    let src: [i8; 16] = [-100, -50, -1, 0, 1, 2, 3, 10, 20, 30, 40, 50, 60, 70, 80, 90];
    let add_operand: [i8; 16] = [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1];
    let mul_operand: [i8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

    // Step 0 — anchor Add: src (zp −3) + add_operand (zp 5) → out (zp 2),
    // twice-max scaling (left_shift 20) with non-identity per-input pairs.
    let step0 = ElementwiseChainStep {
        kind: ElementwiseKind::Add,
        operand: Some(&add_operand),
        input1_offset: 3,   // −(−3)
        input2_offset: -5,  // −(5)
        output_offset: 2,   // +(2)
        output_multiplier: 1_073_741_824, // 2^30 — output ratio (non-identity)
        output_shift: 1,
        left_shift: 20,
        input1_multiplier: 1_073_741_824, // 2^30, shift 2 ≠ identity (2^30, 1)
        input1_shift: 2,
        input2_multiplier: 536_870_912, // 2^29
        input2_shift: 1,
        quantized_activation_min: -128,
        quantized_activation_max: 127,
    };
    // Step 1 — Relu: running zp = −(step0 output_offset) = −2.
    let step1 = ElementwiseChainStep {
        kind: ElementwiseKind::Relu,
        operand: None,
        input1_offset: 2,
        input2_offset: 0,
        output_offset: 3,
        output_multiplier: 1_610_612_736, // 2^30 + 2^29 — non-identity requantize
        output_shift: 2,
        left_shift: 0,
        input1_multiplier: 0,
        input1_shift: 0,
        input2_multiplier: 0,
        input2_shift: 0,
        quantized_activation_min: -128,
        quantized_activation_max: 127,
    };
    // Step 2 — Mul: running zp = −3, operand zp 4, output zp 1 (scale change).
    let step2 = ElementwiseChainStep {
        kind: ElementwiseKind::Mul,
        operand: Some(&mul_operand),
        input1_offset: 3,   // −(−3)
        input2_offset: -4,  // −(4)
        output_offset: 1,   // +(1)
        output_multiplier: 536_870_912, // 2^29 — output ratio (non-identity)
        output_shift: 1,
        left_shift: 0,
        input1_multiplier: 0,
        input1_shift: 0,
        input2_multiplier: 0,
        input2_shift: 0,
        quantized_activation_min: -128,
        quantized_activation_max: 127,
    };
    // Step 3 — HardSwish: running zp = −1, output zp 5.
    let step3 = ElementwiseChainStep {
        kind: ElementwiseKind::HardSwish,
        operand: None,
        input1_offset: 1,
        input2_offset: 0,
        output_offset: 5,
        output_multiplier: 0,
        output_shift: 0,
        left_shift: 0,
        input1_multiplier: 0,
        input1_shift: 0,
        input2_multiplier: 0,
        input2_shift: 0,
        quantized_activation_min: -128,
        quantized_activation_max: 127,
    };

    let steps = [step0.clone(), step1.clone(), step2.clone(), step3.clone()];
    let chain_params = ElementwiseChainParams {
        num_elements: n,
        steps: &steps,
    };

    // Explicit per-op sequence with separate intermediate buffers.
    let mut backend = RefBackend;
    let mut b1 = [0i8; 16];
    let mut b2 = [0i8; 16];
    let mut b3 = [0i8; 16];
    let mut b4 = [0i8; 16];
    let ep0 = ElementwiseParams {
        num_elements: n,
        input1_offset: step0.input1_offset,
        input2_offset: step0.input2_offset,
        output_offset: step0.output_offset,
        output_multiplier: step0.output_multiplier,
        output_shift: step0.output_shift,
        left_shift: step0.left_shift,
        input1_multiplier: step0.input1_multiplier,
        input1_shift: step0.input1_shift,
        input2_multiplier: step0.input2_multiplier,
        input2_shift: step0.input2_shift,
        quantized_activation_min: step0.quantized_activation_min,
        quantized_activation_max: step0.quantized_activation_max,
    };
    let ep2 = ElementwiseParams {
        num_elements: n,
        input1_offset: step2.input1_offset,
        input2_offset: step2.input2_offset,
        output_offset: step2.output_offset,
        output_multiplier: step2.output_multiplier,
        output_shift: step2.output_shift,
        left_shift: step2.left_shift,
        input1_multiplier: step2.input1_multiplier,
        input1_shift: step2.input1_shift,
        input2_multiplier: step2.input2_multiplier,
        input2_shift: step2.input2_shift,
        quantized_activation_min: step2.quantized_activation_min,
        quantized_activation_max: step2.quantized_activation_max,
    };
    let ap1 = act(
        step1.input1_offset,
        step1.output_offset,
        step1.output_multiplier,
        step1.output_shift,
        step1.quantized_activation_min,
        step1.quantized_activation_max,
    );
    let ap3 = act(step3.input1_offset, step3.output_offset, 0, 0, -128, 127);

    backend
        .add(&src, &add_operand, &ep0, &mut b1)
        .expect("anchor add");
    backend.relu(&b1, &ap1, &mut b2).expect("relu");
    backend.mul(&b2, &mul_operand, &ep2, &mut b3).expect("mul");
    backend.hard_swish(&b3, &ap3, &mut b4).expect("hard_swish");

    // Composed chain in one buffer, in-place chaining.
    let mut dst = [0i8; 16];
    backend
        .fused_elementwise_chain(&src, &chain_params, &mut dst)
        .expect("fused chain");

    assert_eq_i8(&dst, &b4);
}

// ── Test 2: residual-add with non-identity residual scale ──────────────────

#[test]
fn fused_conv_residual_relu6() {
    // 1x1 conv over 2x2 spatial, 1 channel; non-trivial offsets + a
    // non-identity per-channel requantize pair.
    let conv = Conv2DParams {
        input_shape: [1, 2, 2, 1],
        filter_shape: [1, 1, 1, 1],
        output_shape: [1, 2, 2, 1],
        padding: Padding::Valid,
        stride_width: 1,
        stride_height: 1,
        dilation_width_factor: 1,
        dilation_height_factor: 1,
        input_offset: 5,
        weights_offset: 1,
        output_offset: -7,
        output_multiplier_per_channel: &[1_610_612_736],
        output_shift_per_channel: &[-1],
        quantized_activation_min: -128,
        quantized_activation_max: 127,
    };
    let src: [i8; 4] = [-10, -3, 0, 12];
    let weight: [i8; 1] = [3];
    let bias: [i32; 1] = [2];
    let residual: [i8; 4] = [5, -7, 2, 9];

    let res = ResidualAddParams {
        residual_data: &residual,
        residual_scale: 0.75,
        residual_zero_point: 3,
        output_scale: 0.5,
        output_zero_point: -2,
        input1_multiplier: 1_073_741_824, // 2^30, shift 2 — non-identity
        input1_shift: 2,
        input2_multiplier: 536_870_912, // 2^29
        input2_shift: 1,
        left_shift: 20,
        output_multiplier: 536_870_912, // non-identity output ratio
        output_shift: 1,
    };
    let fused = FusedConvParams {
        conv: conv.clone(),
        output_scale: 0.5,
        output_zero_point: -7,
        output_multiplier_per_channel: &[1_610_612_736],
        output_shift_per_channel: &[-1],
        residual: Some(res.clone()),
        activation: ActivationEpilogueParams {
            kind: ComposedActivation::Relu6,
            input_offset: 2,  // −(add output zp −2)
            output_offset: 0,
            output_multiplier: 0,
            output_shift: 0,
            quantized_activation_min: 0,
            quantized_activation_max: 12, // quantized six at scale 0.5
        },
    };

    // Explicit per-op sequence: conv → add → relu6, separate buffers.
    let mut backend = RefBackend;
    let mut b1 = [0i8; 4];
    let mut b2 = [0i8; 4];
    let mut b3 = [0i8; 4];
    let mut scratch = [0u8; 64];
    backend
        .conv2d(&src, &weight, &bias, &conv, &mut b1, &mut scratch)
        .expect("conv");
    let add_params = ElementwiseParams {
        num_elements: 4,
        input1_offset: -(fused.output_zero_point as i32), // −(conv out zp −7)
        input2_offset: -(res.residual_zero_point as i32), // −(3)
        output_offset: res.output_zero_point as i32,      // −2
        output_multiplier: res.output_multiplier,
        output_shift: res.output_shift,
        left_shift: res.left_shift,
        input1_multiplier: res.input1_multiplier,
        input1_shift: res.input1_shift,
        input2_multiplier: res.input2_multiplier,
        input2_shift: res.input2_shift,
        quantized_activation_min: -128,
        quantized_activation_max: 127,
    };
    backend.add(&b1, &residual, &add_params, &mut b2).expect("add");
    let ap = act(fused.activation.input_offset, 0, 0, 0, 0, 12);
    backend.relu6(&b2, &ap, &mut b3).expect("relu6");

    // Composed call.
    let mut dst = [0i8; 4];
    backend
        .fused_conv2d(&src, &weight, &bias, &fused, &mut dst, &mut scratch)
        .expect("fused conv2d");

    assert_eq_i8(&dst, &b3);
}

// ── Test 3: MUL pool fold (avg pool, folded_scale != 1) ────────────────────

#[test]
fn fused_pool_fold_mul_average() {
    let pool = PoolParams {
        input_shape: [1, 4, 4, 1],
        output_shape: [1, 2, 2, 1],
        filter_width: 2,
        filter_height: 2,
        stride_width: 2,
        stride_height: 2,
        padding: Padding::Valid,
        activation: hematite_core::op_params::FusedActivation::None,
        quantized_activation_min: -128,
        quantized_activation_max: 127,
    };
    let src: [i8; 16] = [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7];
    let operand: [i8; 16] = [1, 2, 3, 4, 2, 1, 2, 3, 4, 3, 2, 1, 1, 2, 3, 4];

    let fold = PoolInputFold {
        builtin: 18, // MUL
        operand_data: &operand,
        operand_zero_point: 3,
        input_zero_point: -2,
        output_zero_point: 1,
        folded_scale: 0.5, // s_out / s_in != 1
        left_shift: 0,
        output_multiplier: 536_870_912, // 2^29 — non-identity scaling
        output_shift: 1,
        input1_multiplier: 0,
        input1_shift: 0,
        input2_multiplier: 0,
        input2_shift: 0,
        num_elements: 16,
    };
    let folded = FoldedPoolParams {
        pool: pool.clone(),
        pool_kind: PoolKind::Average,
        fold: Some(fold.clone()),
        activation: ActivationEpilogueParams {
            kind: ComposedActivation::None,
            input_offset: 0,
            output_offset: 0,
            output_multiplier: 0,
            output_shift: 0,
            quantized_activation_min: 0,
            quantized_activation_max: 0,
        },
    };

    // Explicit per-op sequence: mul → average_pool_2d, separate buffers.
    let mut backend = RefBackend;
    let mut b1 = [0i8; 16];
    let mut b2 = [0i8; 4];
    let mul_params = ElementwiseParams {
        num_elements: fold.num_elements,
        input1_offset: -(fold.input_zero_point as i32),
        input2_offset: -(fold.operand_zero_point as i32),
        output_offset: fold.output_zero_point as i32,
        output_multiplier: fold.output_multiplier,
        output_shift: fold.output_shift,
        left_shift: fold.left_shift,
        input1_multiplier: fold.input1_multiplier,
        input1_shift: fold.input1_shift,
        input2_multiplier: fold.input2_multiplier,
        input2_shift: fold.input2_shift,
        quantized_activation_min: -128,
        quantized_activation_max: 127,
    };
    backend.mul(&src, &operand, &mul_params, &mut b1).expect("mul");
    backend.average_pool_2d(&b1, &pool, &mut b2).expect("avg pool");

    // Composed call with scratch as the i8 intermediate.
    let mut dst = [0i8; 4];
    let mut scratch = [0u8; 32];
    backend
        .fused_pool_with_fold(&src, &folded, &mut dst, &mut scratch)
        .expect("fused pool with fold");

    assert_eq_i8(&dst, &b2);
}

// ── Test 4: pool without fold + trailing activation ────────────────────────

#[test]
fn fused_pool_plain_relu() {
    let pool = PoolParams {
        input_shape: [1, 4, 4, 1],
        output_shape: [1, 2, 2, 1],
        filter_width: 2,
        filter_height: 2,
        stride_width: 2,
        stride_height: 2,
        padding: Padding::Valid,
        activation: hematite_core::op_params::FusedActivation::None,
        quantized_activation_min: -128,
        quantized_activation_max: 127,
    };
    let src: [i8; 16] = [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7];
    let folded = FoldedPoolParams {
        pool: pool.clone(),
        pool_kind: PoolKind::Average,
        fold: None,
        activation: ActivationEpilogueParams {
            kind: ComposedActivation::Relu,
            input_offset: 1, // −(src zp −1)
            output_offset: 2,
            output_multiplier: 1_073_741_824, // non-identity requantize
            output_shift: 1,
            quantized_activation_min: 0,
            quantized_activation_max: 127,
        },
    };

    // Explicit per-op sequence: pool → relu.
    let mut backend = RefBackend;
    let mut b1 = [0i8; 4];
    let mut b2 = [0i8; 4];
    backend.average_pool_2d(&src, &pool, &mut b1).expect("avg pool");
    let ap = act(
        folded.activation.input_offset,
        folded.activation.output_offset,
        folded.activation.output_multiplier,
        folded.activation.output_shift,
        0,
        127,
    );
    backend.relu(&b1, &ap, &mut b2).expect("relu");

    let mut dst = [0i8; 4];
    let mut scratch = [0u8; 8];
    backend
        .fused_pool_with_fold(&src, &folded, &mut dst, &mut scratch)
        .expect("fused plain pool");

    assert_eq_i8(&dst, &b2);
}
