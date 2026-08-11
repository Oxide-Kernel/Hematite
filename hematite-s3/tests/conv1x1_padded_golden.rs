// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Host bit-exact tests for the channel-padded conv1x1 path (T3.3).
//!
//! On the host the ACCX SIMD path is compiled out, so `S3Backend::conv2d`
//! runs the scalar `conv2d_1x1` kernel — the same entry point a device
//! dispatch falls back to on any ineligible shape. Comparing the two
//! backends element-equal with scratch sized exactly by
//! `S3Backend::conv2d_scratch_size` (the T3.3 padded formula) proves every
//! padded shape is accepted, the scratch budget is sufficient for the
//! dispatch, and the scalar semantics match `hematite-ref` bit-exact.
//!
//! The padded SIMD PIPELINE itself (real staging + kernel-contract
//! accumulation + input_offset fold + requantize) is proven bit-exact on the
//! host by the in-crate
//! `conv1x1::tests::conv1x1_small_simd_model_matches_ref_bit_exact`, and on
//! device by `simd_validation.rs::check_conv1x1_padded_3ch_matches_ref`.
//!
//! Matrix: input_c {1, 3, 8, 15, 17, 32} × spatial {1×1, 4×4, 2×5} × offsets
//! {0, 5, 128} × uniform/per-channel requant modes.

use hematite_core::op_params::{Conv2DParams, Padding};
use hematite_core::KernelBackend;
use hematite_ref::RefBackend;
use hematite_s3::backend::S3Backend;

struct Case {
    input_c: i32,
    h: i32,
    w: i32,
    out_c: i32,
    input_offset: i32,
    mode: usize,
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

fn run_case(c: &Case) {
    let pixels = (c.h * c.w) as usize;
    let n = c.out_c as usize;
    let (mults, shifts): (Vec<i32>, Vec<i32>) = match c.mode {
        // Uniform identity pair (1<<30, 1).
        0 => (
            vec![1 << 30; n],
            vec![1; n],
        ),
        // Per-channel mixed mult/shift — the general requantize path.
        1 => (
            (0..n).map(|i| (1 << 30) - (i as i32) * 7919).collect(),
            (0..n).map(|i| (i % 3) as i32).collect(),
        ),
        // Uniform non-identity scale (1<<29, 0) — the hoisted-uniform path.
        _ => (
            vec![1 << 29; n],
            vec![0; n],
        ),
    };

    let params = Conv2DParams {
        input_shape: [1, c.h, c.w, c.input_c],
        filter_shape: [c.out_c, 1, 1, c.input_c],
        output_shape: [1, c.h, c.w, c.out_c],
        padding: Padding::Same,
        stride_width: 1,
        stride_height: 1,
        dilation_width_factor: 1,
        dilation_height_factor: 1,
        input_offset: c.input_offset,
        weights_offset: 0,
        output_offset: if c.input_offset == 0 { 0 } else { -10 },
        output_multiplier_per_channel: &mults,
        output_shift_per_channel: &shifts,
        quantized_activation_min: if c.mode == 1 { 0 } else { -128 },
        quantized_activation_max: 127,
    };

    let src = pattern(
        0x1C0_0000u64 | (c.input_c as u64 * 131 + c.h as u64 * 17 + c.w as u64),
        pixels * c.input_c as usize,
    );
    let weights = pattern(0xE3A + c.input_c as u64 * 17, c.input_c as usize * n);
    let bias: Vec<i32> = (0..n).map(|i| (i as i32) * 37 - 500).collect();

    let mut s3_out = vec![0i8; pixels * n];
    let mut ref_out = vec![0i8; pixels * n];
    let scratch_need = S3Backend::conv2d_scratch_size(&params);
    let mut scratch = vec![0u8; scratch_need];

    let s3 = S3Backend;
    s3.conv2d(&src, &weights, &bias, &params, &mut s3_out, &mut scratch)
        .expect("s3 conv2d must succeed");
    let r = RefBackend;
    r.conv2d(&src, &weights, &bias, &params, &mut ref_out, &mut scratch)
        .expect("ref conv2d must succeed");

    assert_eq!(
        s3_out, ref_out,
        "in_c={} {}x{} out_c={} offset={} mode={}: S3Backend conv2d != RefBackend",
        c.input_c, c.h, c.w, c.out_c, c.input_offset, c.mode
    );
}

#[test]
fn conv1x1_padded_s3_matches_ref_bit_exact() {
    let mut checked = 0;
    for &input_c in &[1, 3, 8, 15, 17, 32] {
        for &(h, w) in &[(1, 1), (4, 4), (2, 5)] {
            for &input_offset in &[0, 5, 128] {
                for mode in 0..3 {
                    run_case(&Case {
                        input_c,
                        h,
                        w,
                        out_c: 16,
                        input_offset,
                        mode,
                    });
                    checked += 1;
                }
            }
        }
    }
    assert!(checked >= 150, "conv1x1 padded matrix did not expand ({checked})");
}

/// The T3.3 padded scratch formula must cover the dispatch's staged layout
/// (padded input + padded weights + accs + optional wsum) so a device build
/// with exactly this scratch engages the SIMD path.
#[test]
fn conv1x1_padded_scratch_covers_staged_layout() {
    // in_c 3, 4×4 spatial, out_c 16, offset 0: 16 px × 16 ch staged input +
    // 16×16 staged weights + 16×4 accs.
    let params = Conv2DParams {
        input_shape: [1, 4, 4, 3],
        filter_shape: [16, 1, 1, 3],
        output_shape: [1, 4, 4, 16],
        padding: Padding::Same,
        stride_width: 1,
        stride_height: 1,
        dilation_width_factor: 1,
        dilation_height_factor: 1,
        input_offset: 0,
        weights_offset: 0,
        output_offset: 0,
        output_multiplier_per_channel: &[1 << 30; 16],
        output_shift_per_channel: &[1; 16],
        quantized_activation_min: -128,
        quantized_activation_max: 127,
    };
    assert_eq!(
        S3Backend::conv2d_scratch_size(&params),
        16 * 16 + 16 * 16 + 16 * 4
    );
}
