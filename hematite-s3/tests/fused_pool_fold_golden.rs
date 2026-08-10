// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Host bit-exact tests for `S3Backend::fused_pool_with_fold` (T2.4).
//!
//! On the host the fused SIMD dispatch is compiled out, so
//! `fused_pool_with_fold` runs the per-op decomposition (fold materialized
//! into scratch → pool → activation) — the same sequence
//! `hematite-ref/src/fused.rs` runs. Comparing the two backends element-equal
//! proves the trait method is correct on every path; WHICH fold shapes the
//! SIMD path engages vs falls back is pinned by the in-crate
//! `fused::tests::fused_pool_fold_simd_eligibility_gate_expectations`.
//!
//! Matrix: avg/max pool × fold{MUL,SUB} × {enabled, disabled}-by-T2-subset ×
//! activation{relu, relu6, hard_swish, none}. The "enabled" folds are the
//! provably-exact subset (identity-quant-affine — zero offsets, full-range
//! clamp, `(1<<30, ·)` output pairs) — the shapes that SIMD-engage on device;
//! the "disabled" folds carry non-identity quant (a real MUL scale change /
//! the two-stage TFLM SUB rounding) and fall back to the decomposition. Both
//! legs must equal the RefBackend oracle.

use hematite_core::op_params::{
    ActivationEpilogueParams, ComposedActivation, FoldedPoolParams, FusedActivation, Padding,
    PoolInputFold, PoolKind, PoolParams,
};
use hematite_core::FusedKernelBackend;
use hematite_ref::RefBackend;
use hematite_s3::backend::S3Backend;

/// The canonical SIMD-eligible pool anchor: 2×2/stride-2/SAME, channels % 16,
/// full-range clamp — exactly `pool::simd_eligible_pool`'s contract.
fn pool_params() -> PoolParams {
    PoolParams {
        input_shape: [1, 4, 4, 16],
        output_shape: [1, 2, 2, 16],
        filter_width: 2,
        filter_height: 2,
        stride_width: 2,
        stride_height: 2,
        padding: Padding::Same,
        activation: FusedActivation::None,
        quantized_activation_min: i8::MIN as i32,
        quantized_activation_max: i8::MAX as i32,
    }
}

#[allow(clippy::too_many_arguments)]
fn fold(
    builtin: i32,
    operand: &'static [i8],
    input_zp: i64,
    operand_zp: i64,
    out_zp: i64,
    left_shift: i32,
    om: i32,
    os: i32,
    i1m: i32,
    i1s: i32,
    i2m: i32,
    i2s: i32,
) -> PoolInputFold<'static> {
    PoolInputFold {
        builtin,
        operand_data: operand,
        operand_zero_point: operand_zp,
        input_zero_point: input_zp,
        output_zero_point: out_zp,
        folded_scale: 1.0,
        left_shift,
        output_multiplier: om,
        output_shift: os,
        input1_multiplier: i1m,
        input1_shift: i1s,
        input2_multiplier: i2m,
        input2_shift: i2s,
        num_elements: 256,
    }
}

/// ENABLED-by-T2-subset MUL fold: zero offsets, full-range clamp,
/// `(1<<30, 0)` output pair — `simd_eligible_mul`'s exact contract.
fn identity_mul_fold(operand: &'static [i8]) -> PoolInputFold<'static> {
    fold(18, operand, 0, 0, 0, 0, 1 << 30, 0, 0, 0, 0, 0)
}

/// DISABLED MUL fold: a real scale change (`om/os` non-identity, non-zero
/// zero points) — falls back to the decomposition.
fn non_identity_mul_fold(operand: &'static [i8]) -> PoolInputFold<'static> {
    fold(18, operand, 5, -3, 1, 0, 1_717_986_918, -3, 0, 0, 0, 0)
}

/// ENABLED-by-T2-subset SUB fold: zero offsets, `left_shift <= 0`, identity
/// `(1<<30, 1)` pairs everywhere — `simd_eligible_add_sub`'s exact contract.
fn identity_sub_fold(operand: &'static [i8]) -> PoolInputFold<'static> {
    fold(41, operand, 0, 0, 0, 0, 1 << 30, 1, 1 << 30, 1, 1 << 30, 1)
}

/// DISABLED SUB fold: the two-stage TFLM Add rounding (`left_shift = 20`,
/// non-identity per-input pairs) — falls back to the decomposition.
fn non_identity_sub_fold(operand: &'static [i8]) -> PoolInputFold<'static> {
    fold(41, operand, 5, -3, 1, 20, 1_342_177_280, -18, 1 << 30, 0, 1_288_490_189, -1)
}

fn activation(kind: ComposedActivation) -> ActivationEpilogueParams {
    match kind {
        ComposedActivation::None => ActivationEpilogueParams {
            kind,
            input_offset: 0,
            output_offset: 0,
            output_multiplier: 0,
            output_shift: 0,
            quantized_activation_min: -128,
            quantized_activation_max: 127,
        },
        ComposedActivation::Relu => ActivationEpilogueParams {
            kind,
            input_offset: -5,
            output_offset: 2,
            output_multiplier: 1_342_177_280,
            output_shift: 1,
            quantized_activation_min: 0,
            quantized_activation_max: 127,
        },
        ComposedActivation::Relu6 => ActivationEpilogueParams {
            kind,
            input_offset: -1,
            output_offset: 3,
            output_multiplier: 0,
            output_shift: 0,
            quantized_activation_min: 0,
            quantized_activation_max: 24, // quantized six @ scale 0.25
        },
        ComposedActivation::HardSwish => ActivationEpilogueParams {
            kind,
            input_offset: -3,
            output_offset: 1,
            output_multiplier: 0,
            output_shift: 0,
            quantized_activation_min: 0,
            quantized_activation_max: 127,
        },
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

fn run_case(name: &str, kind: PoolKind, fold: PoolInputFold<'static>, act: ComposedActivation) {
    let src: Vec<i8> = pattern(0xABCD_1234, 256);
    let params = FoldedPoolParams {
        pool: pool_params(),
        pool_kind: kind,
        fold: Some(fold),
        activation: activation(act),
    };
    let scratch_need = hematite_s3::backend::fused_pool_with_fold_scratch_need(&params);
    assert_eq!(scratch_need, 256, "{name}: fold staging need");

    let mut s3_out = vec![0i8; 64];
    let mut ref_out = vec![0i8; 64];
    let mut scratch = vec![0u8; scratch_need];

    let mut s3 = S3Backend;
    s3.fused_pool_with_fold(&src, &params, &mut s3_out, &mut scratch)
        .expect("s3 fused_pool_with_fold must succeed");
    let mut r = RefBackend;
    r.fused_pool_with_fold(&src, &params, &mut ref_out, &mut scratch)
        .expect("ref fused_pool_with_fold must succeed");

    assert_eq!(
        s3_out, ref_out,
        "{name}: S3Backend fused_pool_with_fold != RefBackend decomposition"
    );
}

#[test]
fn fused_pool_with_fold_matches_ref_decomposition_matrix() {
    let operand: &'static [i8] = Box::leak(pattern(0x0BAD_F00D, 256).into_boxed_slice());
    let mut rows = 0;

    for (kind, kind_name) in [(PoolKind::Average, "avg"), (PoolKind::Max, "max")] {
        for (fold, fold_name, enabled) in [
            (identity_mul_fold(operand), "mul_identity", true),
            (non_identity_mul_fold(operand), "mul_scale", false),
            (identity_sub_fold(operand), "sub_identity", true),
            (non_identity_sub_fold(operand), "sub_twostage", false),
        ] {
            for (act, act_name) in [
                (ComposedActivation::None, "none"),
                (ComposedActivation::Relu, "relu"),
                (ComposedActivation::Relu6, "relu6"),
                (ComposedActivation::HardSwish, "hard_swish"),
            ] {
                let enabled_label = if enabled { "SIMD-eligible" } else { "fallback" };
                run_case(
                    &format!("{kind_name}_{fold_name}_{act_name} ({enabled_label})"),
                    kind,
                    fold.clone(),
                    act,
                );
                rows += 1;
            }
        }
    }
    assert_eq!(rows, 32, "matrix must cover 32 rows");
}
