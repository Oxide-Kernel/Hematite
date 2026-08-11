// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Host bit-exact tests for `S3Backend::fused_elementwise_chain` (T2.3).
//!
//! On the host the SIMD chain dispatch is compiled out, so
//! `fused_elementwise_chain` runs the per-op decomposition (each step's own
//! requantize preserved: step 0 reads `src` as input1, later steps read the
//! running value in `dst` in-place) — the same sequence
//! `hematite-ref/src/fused.rs` runs. Comparing the two backends element-equal
//! proves the trait method is correct on every fallback path (every chain in
//! this matrix carries NON-identity scales; the mixed-eligibility rows also
//! fall back on device). The SIMD-path register math is separately proven
//! bit-exact by the in-crate
//! `fused::tests::fused_chain_register_math_matches_decomposition_bit_exact`,
//! and WHICH chains SIMD-engage today is pinned by
//! `fused::tests::chain_simd_eligibility_gate_expectations`.

use hematite_core::op_params::{ElementwiseChainParams, ElementwiseChainStep, ElementwiseKind};
use hematite_core::FusedKernelBackend;
use hematite_ref::RefBackend;
use hematite_s3::backend::S3Backend;

/// `tflite::QuantizeMultiplier` (host mirror — this test crate never pulls
/// f64 into the device build).
fn quantize_multiplier(scale: f64) -> (i32, i32) {
    if scale == 0.0 {
        return (0, 0);
    }
    let (sig, mut shift) = frexp(scale);
    let mut q_fixed = (sig * (1u64 << 31) as f64 + 0.5f64) as i64;
    if q_fixed == (1i64 << 31) {
        q_fixed /= 2;
        shift += 1;
    }
    if shift < -31 {
        return (0, 0);
    }
    (q_fixed as i32, shift)
}

fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 {
        return (0.0, 0);
    }
    let bits = x.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let mantissa = bits & 0x000f_ffff_ffff_ffff;
    let sign = bits & 0x8000_0000_0000_0000;
    let frexp_exponent = exponent - 1022;
    let frexp_significand_bits = sign | 0x3fe0_0000_0000_0000u64 | mantissa;
    (f64::from_bits(frexp_significand_bits), frexp_exponent)
}

fn full_range() -> (i32, i32) {
    (i8::MIN as i32, i8::MAX as i32)
}

/// A binary Add/Sub step: two-stage TFLM Add rounding pairs derived from the
/// scales exactly as the T1.1 fusion IR derives them (left_shift = 20,
/// input_i = QuantizeMultiplier(s_i/twice_max), output =
/// QuantizeMultiplier(twice_max/(2^20·s_out))).
#[allow(clippy::too_many_arguments)]
fn binary_step(
    kind: ElementwiseKind,
    operand: &'static [i8],
    s_in1: f64,
    s_in2: f64,
    s_out: f64,
    zp_in1: i32,
    zp_in2: i32,
    zp_out: i32,
    act: (i32, i32),
) -> ElementwiseChainStep<'static> {
    let twice_max = 2.0 * s_in1.max(s_in2);
    let (i1m, i1s) = quantize_multiplier(s_in1 / twice_max);
    let (i2m, i2s) = quantize_multiplier(s_in2 / twice_max);
    let (om, os) = quantize_multiplier(twice_max / ((1i32 << 20) as f64 * s_out));
    ElementwiseChainStep {
        kind,
        operand: Some(operand),
        input1_offset: -zp_in1,
        input2_offset: -zp_in2,
        output_offset: zp_out,
        output_multiplier: om,
        output_shift: os,
        left_shift: 20,
        input1_multiplier: i1m,
        input1_shift: i1s,
        input2_multiplier: i2m,
        input2_shift: i2s,
        quantized_activation_min: act.0,
        quantized_activation_max: act.1,
    }
}

/// A Mul step: single output requantize `QuantizeMultiplier(s_in1·s_in2/s_out)`,
/// no per-input rescaling.
#[allow(clippy::too_many_arguments)]
fn mul_step(
    operand: &'static [i8],
    s_in1: f64,
    s_in2: f64,
    s_out: f64,
    zp_in1: i32,
    zp_in2: i32,
    zp_out: i32,
    act: (i32, i32),
) -> ElementwiseChainStep<'static> {
    let (om, os) = quantize_multiplier(s_in1 * s_in2 / s_out);
    ElementwiseChainStep {
        kind: ElementwiseKind::Mul,
        operand: Some(operand),
        input1_offset: -zp_in1,
        input2_offset: -zp_in2,
        output_offset: zp_out,
        output_multiplier: om,
        output_shift: os,
        left_shift: 0,
        input1_multiplier: 0,
        input1_shift: 0,
        input2_multiplier: 0,
        input2_shift: 0,
        quantized_activation_min: act.0,
        quantized_activation_max: act.1,
    }
}

/// A Relu step: `max(0, x + input_offset)` then requantize by the output
/// ratio `QuantizeMultiplier(s_in/s_out)`.
fn relu_step(
    s_in: f64,
    s_out: f64,
    zp_in: i32,
    zp_out: i32,
) -> ElementwiseChainStep<'static> {
    let (om, os) = quantize_multiplier(s_in / s_out);
    ElementwiseChainStep {
        kind: ElementwiseKind::Relu,
        operand: None,
        input1_offset: -zp_in,
        input2_offset: 0,
        output_offset: zp_out,
        output_multiplier: om,
        output_shift: os,
        left_shift: 0,
        input1_multiplier: 0,
        input1_shift: 0,
        input2_multiplier: 0,
        input2_shift: 0,
        quantized_activation_min: 0,
        quantized_activation_max: 127,
    }
}

/// A HardSwish step — the DOWNGRADED integer rational formula (the ratio
/// fields are ignored by the scalar kernel).
fn hard_swish_step(s_in: f64, s_out: f64, zp_in: i32, zp_out: i32) -> ElementwiseChainStep<'static> {
    let (om, os) = quantize_multiplier(s_in / s_out);
    ElementwiseChainStep {
        kind: ElementwiseKind::HardSwish,
        operand: None,
        input1_offset: -zp_in,
        input2_offset: 0,
        output_offset: zp_out,
        output_multiplier: om,
        output_shift: os,
        left_shift: 0,
        input1_multiplier: 0,
        input1_shift: 0,
        input2_multiplier: 0,
        input2_shift: 0,
        quantized_activation_min: 0,
        quantized_activation_max: 127,
    }
}

fn pattern(seed: u64, n: usize) -> Vec<i8> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (x >> 33) as i8
        })
        .collect()
}

fn run_case(name: &'static str, steps: Vec<ElementwiseChainStep<'static>>, src: &[i8]) {
    let params = ElementwiseChainParams {
        num_elements: src.len() as i32,
        steps: &steps,
    };
    let mut s3_out = vec![0i8; src.len()];
    let mut ref_out = vec![0i8; src.len()];
    let mut s3 = S3Backend;
    s3.fused_elementwise_chain(src, &params, &mut s3_out)
        .expect("s3 fused_elementwise_chain must succeed");
    let mut r = RefBackend;
    r.fused_elementwise_chain(src, &params, &mut ref_out)
        .expect("ref fused_elementwise_chain must succeed");
    assert_eq!(
        s3_out, ref_out,
        "{name}: S3Backend fused_elementwise_chain != RefBackend decomposition"
    );
}

#[test]
fn fused_elementwise_chain_matches_ref_decomposition_matrix() {
    const N: usize = 256;
    let src: Vec<i8> = pattern(0xABCD_1234, N);
    let op_a: &'static [i8] = Box::leak(pattern(0x0BAD_F00D, N).into_boxed_slice());
    let op_b: &'static [i8] = Box::leak(pattern(0xDEAD_BEEF, N).into_boxed_slice());
    let op_c: &'static [i8] = Box::leak(pattern(0xFEED_FACE, N).into_boxed_slice());

    // (1) The plan's canonical 4-op chain: add + relu + mul + hard_swish —
    //     every step NON-identity (add's per-input roundings, relu's ratio,
    //     mul's product requantize, hard_swish's downgraded formula).
    run_case(
        "canonical_4op_add_relu_mul_hardswish",
        vec![
            binary_step(ElementwiseKind::Add, op_a, 0.5, 0.3, 0.4, 5, -3, 1, full_range()),
            relu_step(0.4, 0.2, 1, 2),
            mul_step(op_b, 0.2, 0.05, 0.1, -2, 0, -3, full_range()),
            hard_swish_step(0.1, 0.03, 3, 1),
        ],
        &src,
    );

    // (2) A 2-op add + relu chain.
    run_case(
        "add_relu",
        vec![
            binary_step(ElementwiseKind::Add, op_a, 0.25, 0.125, 0.2, 4, -2, -3, full_range()),
            relu_step(0.2, 0.1, -3, 1),
        ],
        &src,
    );

    // (3) A mul + hard_swish chain.
    run_case(
        "mul_hardswish",
        vec![
            mul_step(op_b, 1.0, 0.5, 0.75, 0, -1, 2, full_range()),
            hard_swish_step(0.75, 0.2, -2, 0),
        ],
        &src,
    );

    // (4) A sub + relu + mul chain with strongly NON-identity zero points
    //     (src zp -127, mid zp 100) — stresses the offset paths.
    run_case(
        "sub_relu_mul_nonidentity_zp",
        vec![
            binary_step(ElementwiseKind::Sub, op_c, 0.02, 0.05, 0.03, -127, 7, 100, full_range()),
            relu_step(0.03, 0.015, -100, 42),
            mul_step(op_a, 0.015, 0.1, 0.04, -42, -8, 9, full_range()),
        ],
        &src,
    );

    // (5) SIMD eligibility differs per step: step 0 is an IDENTITY add
    //     (SIMD-eligible today) and step 1 a NON-identity relu (ineligible)
    //     → the chain falls back, still bit-exact. Identity add pairs are
    //     the (1<<30, 1) identity with zero offsets (the simd_eligible_add_sub
    //     gate's exact contract).
    let identity_add = ElementwiseChainStep {
        kind: ElementwiseKind::Add,
        operand: Some(op_b),
        input1_offset: 0,
        input2_offset: 0,
        output_offset: 0,
        output_multiplier: 1 << 30,
        output_shift: 1,
        left_shift: 0,
        input1_multiplier: 1 << 30,
        input1_shift: 1,
        input2_multiplier: 1 << 30,
        input2_shift: 1,
        quantized_activation_min: i8::MIN as i32,
        quantized_activation_max: i8::MAX as i32,
    };
    run_case(
        "mixed_eligibility_identity_add_nonidentity_relu",
        vec![identity_add, relu_step(0.5, 0.3, 0, 4)],
        &src,
    );

    // (6) A relu6 chain step (no SIMD at all today — T3.2; the decomposition
    //     forwards the relu6 clamp bound via quantized_activation_max).
    run_case(
        "add_relu6",
        vec![
            binary_step(ElementwiseKind::Add, op_a, 0.5, 0.3, 0.4, 5, -3, 1, full_range()),
            ElementwiseChainStep {
                kind: ElementwiseKind::Relu6,
                operand: None,
                input1_offset: -1,
                input2_offset: 0,
                output_offset: 2,
                output_multiplier: 0,
                output_shift: 0,
                left_shift: 0,
                input1_multiplier: 0,
                input1_shift: 0,
                input2_multiplier: 0,
                input2_shift: 0,
                quantized_activation_min: 0,
                quantized_activation_max: 24, // quantized six @ scale 0.25
            },
        ],
        &src,
    );
}

/// The composed chain needs NO scratch — the SIMD path keeps the running
/// value in register lanes and reads operands in place, and the decomposition
/// forwards no scratch either (the per-op elementwise/activation kernels take
/// `&mut []`). The codegen emitter reports the literal 0 for chains too, so
/// the T1.4 parity invariant is `0 == 0`.
#[test]
fn fused_elementwise_chain_needs_no_scratch() {
    let steps: Vec<ElementwiseChainStep<'static>> = vec![mul_step(
        Box::leak(vec![0i8; 64].into_boxed_slice()),
        0.5, 0.25, 0.4, 2, -1, 3, full_range(),
    )];
    let params = ElementwiseChainParams {
        num_elements: 64,
        steps: &steps,
    };
    assert_eq!(
        hematite_s3::backend::fused_elementwise_chain_scratch_need(&params),
        0
    );
}
