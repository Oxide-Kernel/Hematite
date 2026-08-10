// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Host bit-exact tests for `S3Backend::fused_conv2d` (T2.2).
//!
//! On the host the ACCX SIMD path is compiled out, so `fused_conv2d` runs the
//! per-op decomposition (conv2d → add → activation) — the same sequence
//! `hematite-ref/src/fused.rs` runs. Comparing the two backends element-equal
//! proves the trait method is correct on every fallback path; the SIMD-path
//! epilogue arithmetic is separately proven bit-exact by the in-crate
//! `fused::tests::fused_epilogue_matches_decomposition_bit_exact`.
//!
//! Matrix: conv1x1 / conv3x3 / depthwise-shaped anchors × residual /
//! non-residual × relu / relu6 / hard_swish / none epilogue × non-identity
//! residual scale (residual_scale != conv output scale).

use hematite_core::op_params::{
    ActivationEpilogueParams, ComposedActivation, Conv2DParams, FusedConvParams, Padding,
    ResidualAddParams,
};
use hematite_core::{FusedKernelBackend, KernelBackend};
use hematite_ref::RefBackend;
use hematite_s3::backend::S3Backend;

/// `tflite::QuantizeMultiplier` (host mirror of `hematite-int8`'s `host`
/// feature helper — this test crate never pulls f64 into the device build).
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

/// The two-stage TFLM Add requantize pairs, derived from the scales exactly
/// as the T1.1 fusion IR derives them (`StepRequantize`, fusion.rs):
/// `input1 = QuantizeMultiplier(s1/twice_max)`,
/// `input2 = QuantizeMultiplier(s2/twice_max)`,
/// `output = QuantizeMultiplier(twice_max/(2^20·s_out))`, `left_shift = 20`.
#[derive(Clone, Copy)]
struct AddPairs {
    input1_multiplier: i32,
    input1_shift: i32,
    input2_multiplier: i32,
    input2_shift: i32,
    left_shift: i32,
    output_multiplier: i32,
    output_shift: i32,
}

fn add_pairs(s1: f64, s2: f64, s_out: f64) -> AddPairs {
    let twice_max = 2.0 * s1.max(s2);
    let (i1m, i1s) = quantize_multiplier(s1 / twice_max);
    let (i2m, i2s) = quantize_multiplier(s2 / twice_max);
    let (om, os) = quantize_multiplier(twice_max / ((1i32 << 20) as f64 * s_out));
    AddPairs {
        input1_multiplier: i1m,
        input1_shift: i1s,
        input2_multiplier: i2m,
        input2_shift: i2s,
        left_shift: 20,
        output_multiplier: om,
        output_shift: os,
    }
}

/// The activation epilogue params, derived exactly as the T1.2 emitter derives
/// them (`quantize_multiplier(scale_in / scale_out)` for relu; relu6 carries
/// the quantized-six clamp; hard_swish ignores the ratio fields).
fn activation_params(
    kind: ComposedActivation,
    input_zp: i32,
    output_zp: i32,
    scale_in: f64,
    scale_out: f64,
) -> ActivationEpilogueParams {
    let (am, ash) = quantize_multiplier(scale_in / scale_out);
    ActivationEpilogueParams {
        kind,
        input_offset: -input_zp,
        output_offset: output_zp,
        output_multiplier: am,
        output_shift: ash,
        quantized_activation_min: 0,
        quantized_activation_max: 127,
    }
}

struct ConvSpec {
    input_shape: [i32; 4],
    filter_shape: [i32; 4],
    output_shape: [i32; 4],
    input_offset: i32,
    out_zp: i32,
    conv_out_zp: i32,
    out_scale: f64,
    residual_scale: f64,
    residual_zp: i32,
    add_out_scale: f64,
    add_out_zp: i32,
    act: ComposedActivation,
    act_in_zp: i32,
    act_out_zp: i32,
    act_out_scale: f64,
    residual: bool,
    name: &'static str,
}

fn conv_params(spec: &ConvSpec) -> Conv2DParams<'static> {
    let out_c = spec.filter_shape[0] as usize;
    let muls: Vec<i32> = vec![1 << 30; out_c];
    let shifts: Vec<i32> = vec![0; out_c];
    // Leak the const arrays — the params are static-shaped literals.
    let muls: &'static [i32] = Box::leak(muls.into_boxed_slice());
    let shifts: &'static [i32] = Box::leak(shifts.into_boxed_slice());
    Conv2DParams {
        input_shape: spec.input_shape,
        filter_shape: spec.filter_shape,
        output_shape: spec.output_shape,
        padding: Padding::Same,
        stride_width: 1,
        stride_height: 1,
        dilation_width_factor: 1,
        dilation_height_factor: 1,
        input_offset: spec.input_offset,
        weights_offset: 0,
        output_offset: spec.conv_out_zp,
        output_multiplier_per_channel: muls,
        output_shift_per_channel: shifts,
        quantized_activation_min: -128,
        quantized_activation_max: 127,
    }
}

fn fused_params(spec: &ConvSpec, residual_data: &'static [i8]) -> FusedConvParams<'static> {
    let pairs = add_pairs(spec.out_scale, spec.residual_scale, spec.add_out_scale);
    let out_c = spec.filter_shape[0] as usize;
    let muls: &'static [i32] = Box::leak(vec![1 << 30; out_c].into_boxed_slice());
    let shifts: &'static [i32] = Box::leak(vec![0; out_c].into_boxed_slice());
    FusedConvParams {
        conv: conv_params(spec),
        output_scale: spec.out_scale as f32,
        output_zero_point: spec.out_zp as i64,
        output_multiplier_per_channel: muls,
        output_shift_per_channel: shifts,
        residual: if spec.residual {
            Some(ResidualAddParams {
                residual_data,
                residual_scale: spec.residual_scale as f32,
                residual_zero_point: spec.residual_zp as i64,
                output_scale: spec.add_out_scale as f32,
                output_zero_point: spec.add_out_zp as i64,
                input1_multiplier: pairs.input1_multiplier,
                input1_shift: pairs.input1_shift,
                input2_multiplier: pairs.input2_multiplier,
                input2_shift: pairs.input2_shift,
                left_shift: pairs.left_shift,
                output_multiplier: pairs.output_multiplier,
                output_shift: pairs.output_shift,
            })
        } else {
            None
        },
        activation: activation_params(
            spec.act,
            spec.act_in_zp,
            spec.act_out_zp,
            spec.out_scale,
            spec.act_out_scale,
        ),
    }
}

fn shape_product(shape: &[i32; 4]) -> usize {
    shape[0] as usize * shape[1] as usize * shape[2] as usize * shape[3] as usize
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

fn run_case(spec: &ConvSpec) {
    let in_len = shape_product(&spec.input_shape);
    let w_len = shape_product(&spec.filter_shape);
    let out_len = shape_product(&spec.output_shape);
    let src = pattern(0xABCD_1234, in_len);
    let weights = pattern(0xDEAD_BEEF, w_len);
    let bias: Vec<i32> = (0..spec.filter_shape[0] as usize)
        .map(|i| (i as i32) * 37 - 500)
        .collect();
    let residual: Vec<i8> = pattern(0x0BAD_F00D, out_len);
    let params = fused_params(spec, Box::leak(residual.into_boxed_slice()));

    let mut s3_out = vec![0i8; out_len];
    let mut ref_out = vec![0i8; out_len];
    let scratch_need = hematite_s3::backend::fused_conv2d_scratch_need(&params);
    let mut scratch = vec![0u8; scratch_need];

    let mut s3 = S3Backend;
    s3.fused_conv2d(&src, &weights, &bias, &params, &mut s3_out, &mut scratch)
        .expect("s3 fused_conv2d must succeed");
    let mut r = RefBackend;
    r.fused_conv2d(&src, &weights, &bias, &params, &mut ref_out, &mut scratch)
        .expect("ref fused_conv2d must succeed");

    assert_eq!(
        s3_out, ref_out,
        "{}: S3Backend fused_conv2d != RefBackend decomposition",
        spec.name
    );
}

#[test]
fn fused_conv2d_matches_ref_decomposition_matrix() {
    let specs: Vec<ConvSpec> = vec![
        // 1×1 anchor + residual + ReLU (mv2 prototype; residual scale != conv
        // output scale exercises the per-input roundings).
        ConvSpec {
            input_shape: [1, 4, 4, 16],
            filter_shape: [16, 1, 1, 16],
            output_shape: [1, 4, 4, 16],
            input_offset: 0,
            out_zp: 5,
            conv_out_zp: 5,
            out_scale: 0.5,
            residual_scale: 0.3,
            residual_zp: -3,
            add_out_scale: 0.4,
            add_out_zp: 1,
            act: ComposedActivation::Relu,
            act_in_zp: 1,
            act_out_zp: 2,
            act_out_scale: 0.2,
            residual: true,
            name: "conv1x1_residual_relu",
        },
        // 1×1 anchor + residual + HardSwish (downgraded integer formula).
        ConvSpec {
            input_shape: [1, 4, 4, 16],
            filter_shape: [16, 1, 1, 16],
            output_shape: [1, 4, 4, 16],
            input_offset: 0,
            out_zp: -2,
            conv_out_zp: -2,
            out_scale: 0.02,
            residual_scale: 0.05,
            residual_zp: 7,
            add_out_scale: 0.03,
            add_out_zp: 3,
            act: ComposedActivation::HardSwish,
            act_in_zp: 3,
            act_out_zp: -1,
            act_out_scale: 0.015,
            residual: true,
            name: "conv1x1_residual_hard_swish",
        },
        // 3×3 anchor (in_c 8, non-%16 → SIMD channel-padding path on device)
        // + residual + ReLU6.
        ConvSpec {
            input_shape: [1, 6, 6, 8],
            filter_shape: [8, 3, 3, 8],
            output_shape: [1, 6, 6, 8],
            input_offset: 0,
            out_zp: 4,
            conv_out_zp: 4,
            out_scale: 0.25,
            residual_scale: 0.125,
            residual_zp: 0,
            add_out_scale: 0.2,
            add_out_zp: 2,
            act: ComposedActivation::Relu6,
            act_in_zp: 2,
            act_out_zp: 0,
            act_out_scale: 0.2,
            residual: true,
            name: "conv3x3_residual_relu6",
        },
        // 3×3 anchor + residual, no activation epilogue.
        ConvSpec {
            input_shape: [1, 6, 6, 8],
            filter_shape: [8, 3, 3, 8],
            output_shape: [1, 6, 6, 8],
            input_offset: 0,
            out_zp: 0,
            conv_out_zp: 0,
            out_scale: 1.0,
            residual_scale: 0.5,
            residual_zp: -5,
            add_out_scale: 0.8,
            add_out_zp: -1,
            act: ComposedActivation::None,
            act_in_zp: 0,
            act_out_zp: 0,
            act_out_scale: 0.8,
            residual: true,
            name: "conv3x3_residual_none",
        },
        // 3×3 anchor with a non-zero input_offset (Phase C fold) + residual
        // + ReLU; conv.output_offset intentionally differs from
        // params.output_zero_point to exercise the independent offset paths.
        ConvSpec {
            input_shape: [1, 6, 6, 16],
            filter_shape: [16, 3, 3, 16],
            output_shape: [1, 6, 6, 16],
            input_offset: -3,
            out_zp: 6,
            conv_out_zp: 5,
            out_scale: 0.1,
            residual_scale: 0.025,
            residual_zp: 2,
            add_out_scale: 0.05,
            add_out_zp: 4,
            act: ComposedActivation::Relu,
            act_in_zp: 4,
            act_out_zp: -2,
            act_out_scale: 0.04,
            residual: true,
            name: "conv3x3_off3_residual_relu",
        },
        // Depthwise-shaped anchor (in_c 1 → out_c 8, kws-style fan-out shape
        // through the conv3x3 path) + residual + ReLU.
        ConvSpec {
            input_shape: [1, 6, 6, 1],
            filter_shape: [8, 3, 3, 1],
            output_shape: [1, 6, 6, 8],
            input_offset: 0,
            out_zp: 1,
            conv_out_zp: 1,
            out_scale: 0.5,
            residual_scale: 0.4,
            residual_zp: -1,
            add_out_scale: 0.45,
            add_out_zp: 0,
            act: ComposedActivation::Relu,
            act_in_zp: 0,
            act_out_zp: 1,
            act_out_scale: 0.3,
            residual: true,
            name: "depthwise_shape_residual_relu",
        },
        // 3×3 anchor, NO residual, ReLU epilogue.
        ConvSpec {
            input_shape: [1, 6, 6, 8],
            filter_shape: [8, 3, 3, 8],
            output_shape: [1, 6, 6, 8],
            input_offset: 0,
            out_zp: 0,
            conv_out_zp: 0,
            out_scale: 1.0,
            residual_scale: 0.5,
            residual_zp: 0,
            add_out_scale: 1.0,
            add_out_zp: 0,
            act: ComposedActivation::Relu,
            act_in_zp: 0,
            act_out_zp: 0,
            act_out_scale: 1.0,
            residual: false,
            name: "conv3x3_no_residual_relu",
        },
        // 1×1 anchor, NO residual, HardSwish epilogue.
        ConvSpec {
            input_shape: [1, 4, 4, 16],
            filter_shape: [16, 1, 1, 16],
            output_shape: [1, 4, 4, 16],
            input_offset: 0,
            out_zp: -3,
            conv_out_zp: -3,
            out_scale: 0.05,
            residual_scale: 0.05,
            residual_zp: 0,
            add_out_scale: 0.05,
            add_out_zp: 0,
            act: ComposedActivation::HardSwish,
            act_in_zp: -3,
            act_out_zp: 2,
            act_out_scale: 0.02,
            residual: false,
            name: "conv1x1_no_residual_hard_swish",
        },
    ];

    assert_eq!(specs.len(), 8, "matrix must cover 8 rows");
    for spec in &specs {
        run_case(spec);
    }
}

/// The composed scratch need must equal the anchor conv's own need — the T1.4
/// parity invariant (the fused epilogue stages nothing beyond the conv).
#[test]
fn fused_scratch_need_equals_conv_need() {
    let conv = Conv2DParams {
        input_shape: [1, 6, 6, 8],
        filter_shape: [8, 3, 3, 8],
        output_shape: [1, 6, 6, 8],
        padding: Padding::Same,
        stride_width: 1,
        stride_height: 1,
        dilation_width_factor: 1,
        dilation_height_factor: 1,
        input_offset: 0,
        weights_offset: 0,
        output_offset: 0,
        output_multiplier_per_channel: &[1 << 30; 8],
        output_shift_per_channel: &[0; 8],
        quantized_activation_min: -128,
        quantized_activation_max: 127,
    };
    let residual = [0i8; 288];
    let conv_need = S3Backend::conv2d_scratch_size(&conv);
    let params = FusedConvParams {
        conv,
        output_scale: 1.0,
        output_zero_point: 0,
        output_multiplier_per_channel: &[1 << 30; 8],
        output_shift_per_channel: &[0; 8],
        residual: Some(ResidualAddParams {
            residual_data: &residual,
            residual_scale: 0.5,
            residual_zero_point: 0,
            output_scale: 0.8,
            output_zero_point: 0,
            input1_multiplier: 1 << 30,
            input1_shift: 0,
            input2_multiplier: 1 << 30,
            input2_shift: 0,
            left_shift: 20,
            output_multiplier: 1 << 30,
            output_shift: 0,
        }),
        activation: activation_params(ComposedActivation::None, 0, 0, 1.0, 1.0),
    };
    assert_eq!(hematite_s3::backend::fused_conv2d_scratch_need(&params), conv_need);
}
