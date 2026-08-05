// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! S3 backend golden tests (T5.1) — feature-gated consolidation.
//!
//! Run with: `cargo test -p hematite-tests --features hematite-s3`
//!
//! The `hematite-s3` backend ships host-compilable scalar kernels plus
//! `#[cfg(target_arch = "xtensa")]`-gated TIE728 SIMD glue that can never
//! run on the host. These tests exercise the host-compilable surface of the
//! s3 backend bit-exact against the golden corpus (A4 leg b). The SIMD
//! kernels themselves are verified on device at T5.3.
//!
//! Coverage mirrors the per-op golden tests in this same directory and the
//! `hematite-s3/tests/*_golden.rs` files (deliberate duplication — this is
//! the T5.1 consolidation point; see learnings.md).

#![cfg(feature = "hematite-s3")]

// ── Fixture includes ───────────────────────────────────────────────────────

mod conv2d_1x1 {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/conv2d_1x1.rs"));
}
mod conv2d_3x3 {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/conv2d_3x3.rs"));
}
mod depthwise_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/depthwise_conv2d.rs"));
}
mod fc_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/fully_connected.rs"));
}
mod average_pool_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/average_pool_2d.rs"));
}
mod max_pool_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/max_pool_2d.rs"));
}
mod softmax_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/softmax.rs"));
}
mod relu_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/relu.rs"));
}
mod relu6_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/relu6.rs"));
}
mod hard_swish_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/hard_swish.rs"));
}
mod elementwise_add_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/elementwise_add.rs"));
}
mod elementwise_mul_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/elementwise_mul.rs"));
}
mod elementwise_sub_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/elementwise_sub.rs"));
}
mod mean_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/mean.rs"));
}

use hematite_core::op_params::{
    ActivationParams, Conv2DParams, DepthwiseConv2DParams, ElementwiseParams,
    FusedActivation, FullyConnectedParams, Padding, PoolParams, ReduceParams,
    SoftmaxParams,
};

// ── Shared helpers ─────────────────────────────────────────────────────────

/// Assert that `actual` matches `expected` element-for-element, printing
/// the index and values of the first mismatch.
fn assert_bit_exact(actual: &[i8], expected: &[i8], name: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{name}: output length {} != expected length {}",
        actual.len(),
        expected.len(),
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            a,
            e,
            "{name}: mismatch at index {i}: kernel={a}, golden={e}",
        );
    }
}

/// Conv2D / Depthwise / FC-style padding enum from a fixture's PAD consts.
fn padding_from_pads(pad_w: i32, pad_h: i32) -> Padding {
    if pad_w > 0 || pad_h > 0 {
        Padding::Same
    } else {
        Padding::Valid
    }
}

// ── Conv / Depthwise / Gemm ────────────────────────────────────────────────

#[test]
fn s3_conv1x1_golden() {
    let params = Conv2DParams {
        input_shape: conv2d_1x1::INPUT_SHAPE,
        filter_shape: conv2d_1x1::FILTER_SHAPE,
        output_shape: conv2d_1x1::OUTPUT_SHAPE,
        padding: padding_from_pads(conv2d_1x1::PAD_WIDTH, conv2d_1x1::PAD_HEIGHT),
        stride_width: conv2d_1x1::STRIDE_WIDTH,
        stride_height: conv2d_1x1::STRIDE_HEIGHT,
        dilation_width_factor: conv2d_1x1::DILATION_W,
        dilation_height_factor: conv2d_1x1::DILATION_H,
        input_offset: conv2d_1x1::INPUT_OFFSET,
        weights_offset: 0,
        output_offset: conv2d_1x1::OUTPUT_OFFSET,
        output_multiplier_per_channel: &conv2d_1x1::OUTPUT_MULTIPLIER,
        output_shift_per_channel: &conv2d_1x1::OUTPUT_SHIFT,
        quantized_activation_min: conv2d_1x1::OUTPUT_ACTIVATION_MIN,
        quantized_activation_max: conv2d_1x1::OUTPUT_ACTIVATION_MAX,
    };
    let mut output = [0i8; 8];
    hematite_s3::conv1x1::conv2d_1x1(
        &conv2d_1x1::INPUT_DATA,
        &conv2d_1x1::WEIGHTS_DATA,
        &conv2d_1x1::BIAS_DATA,
        &params,
        &mut output,
        &mut [],
    )
    .expect("s3 conv2d_1x1 returned Err");
    assert_bit_exact(&output, &conv2d_1x1::EXPECTED_OUTPUT, "s3_conv1x1_golden");
}

#[test]
fn s3_conv3x3_golden() {
    let params = Conv2DParams {
        input_shape: conv2d_3x3::INPUT_SHAPE,
        filter_shape: conv2d_3x3::FILTER_SHAPE,
        output_shape: conv2d_3x3::OUTPUT_SHAPE,
        padding: padding_from_pads(conv2d_3x3::PAD_WIDTH, conv2d_3x3::PAD_HEIGHT),
        stride_width: conv2d_3x3::STRIDE_WIDTH,
        stride_height: conv2d_3x3::STRIDE_HEIGHT,
        dilation_width_factor: conv2d_3x3::DILATION_W,
        dilation_height_factor: conv2d_3x3::DILATION_H,
        input_offset: conv2d_3x3::INPUT_OFFSET,
        weights_offset: 0,
        output_offset: conv2d_3x3::OUTPUT_OFFSET,
        output_multiplier_per_channel: &conv2d_3x3::OUTPUT_MULTIPLIER,
        output_shift_per_channel: &conv2d_3x3::OUTPUT_SHIFT,
        quantized_activation_min: conv2d_3x3::OUTPUT_ACTIVATION_MIN,
        quantized_activation_max: conv2d_3x3::OUTPUT_ACTIVATION_MAX,
    };
    let mut output = [0i8; 16];
    hematite_s3::conv3x3::conv2d_3x3(
        &conv2d_3x3::INPUT_DATA,
        &conv2d_3x3::WEIGHTS_DATA,
        &conv2d_3x3::BIAS_DATA,
        &params,
        &mut output,
        &mut [],
    )
    .expect("s3 conv2d_3x3 returned Err");
    assert_bit_exact(&output, &conv2d_3x3::EXPECTED_OUTPUT, "s3_conv3x3_golden");
}

#[test]
fn s3_depthwise_golden() {
    let params = DepthwiseConv2DParams {
        input_shape: depthwise_fixture::INPUT_SHAPE,
        filter_shape: depthwise_fixture::FILTER_SHAPE,
        output_shape: depthwise_fixture::OUTPUT_SHAPE,
        padding: padding_from_pads(
            depthwise_fixture::PAD_WIDTH,
            depthwise_fixture::PAD_HEIGHT,
        ),
        stride_width: depthwise_fixture::STRIDE_WIDTH,
        stride_height: depthwise_fixture::STRIDE_HEIGHT,
        dilation_width_factor: depthwise_fixture::DILATION_W,
        dilation_height_factor: depthwise_fixture::DILATION_H,
        depth_multiplier: depthwise_fixture::DEPTH_MULTIPLIER,
        input_offset: depthwise_fixture::INPUT_OFFSET,
        weights_offset: 0,
        output_offset: depthwise_fixture::OUTPUT_OFFSET,
        output_multiplier_per_channel: &depthwise_fixture::OUTPUT_MULTIPLIER,
        output_shift_per_channel: &depthwise_fixture::OUTPUT_SHIFT,
        quantized_activation_min: depthwise_fixture::OUTPUT_ACTIVATION_MIN,
        quantized_activation_max: depthwise_fixture::OUTPUT_ACTIVATION_MAX,
    };
    let mut output = [0i8; 36];
    hematite_s3::depthwise::depthwise_conv2d(
        &depthwise_fixture::INPUT_DATA,
        &depthwise_fixture::WEIGHTS_DATA,
        &depthwise_fixture::BIAS_DATA,
        &params,
        &mut output,
        &mut [],
    )
    .expect("s3 depthwise_conv2d returned Err");
    assert_bit_exact(
        &output,
        &depthwise_fixture::EXPECTED_OUTPUT,
        "s3_depthwise_golden",
    );
}

#[test]
fn s3_fully_connected_golden() {
    let params = FullyConnectedParams {
        input_dim: fc_fixture::ACCUM_DEPTH,
        output_dim: fc_fixture::OUTPUT_DEPTH,
        input_offset: fc_fixture::INPUT_OFFSET,
        weights_offset: 0,
        output_offset: fc_fixture::OUTPUT_OFFSET,
        output_multiplier_per_channel: &fc_fixture::OUTPUT_MULTIPLIER,
        output_shift_per_channel: &fc_fixture::OUTPUT_SHIFT,
        quantized_activation_min: fc_fixture::OUTPUT_ACTIVATION_MIN,
        quantized_activation_max: fc_fixture::OUTPUT_ACTIVATION_MAX,
    };
    let mut output = [0i8; 3];
    hematite_s3::gemm::fully_connected(
        &fc_fixture::INPUT_DATA,
        &fc_fixture::WEIGHTS_DATA,
        &fc_fixture::BIAS_DATA,
        &params,
        &mut output,
        &mut [],
    )
    .expect("s3 fully_connected returned Err");
    assert_bit_exact(
        &output,
        &fc_fixture::EXPECTED_OUTPUT,
        "s3_fully_connected_golden",
    );
}

// ── Pool ───────────────────────────────────────────────────────────────────

/// Construct a `PoolParams` from a fixture module's public consts.
macro_rules! pool_params_from_fixture {
    ($m:ident) => {{
        PoolParams {
            input_shape: $m::INPUT_SHAPE,
            output_shape: $m::OUTPUT_SHAPE,
            filter_width: $m::FILTER_WIDTH,
            filter_height: $m::FILTER_HEIGHT,
            stride_width: $m::STRIDE_WIDTH,
            stride_height: $m::STRIDE_HEIGHT,
            padding: padding_from_pads($m::PAD_WIDTH, $m::PAD_HEIGHT),
            activation: FusedActivation::None,
            quantized_activation_min: $m::OUTPUT_ACTIVATION_MIN,
            quantized_activation_max: $m::OUTPUT_ACTIVATION_MAX,
        }
    }};
}

#[test]
fn s3_average_pool_golden() {
    let params = pool_params_from_fixture!(average_pool_fixture);
    let mut output = [0i8; 4];
    hematite_s3::pool::average_pool_2d(
        &average_pool_fixture::INPUT_DATA,
        &params,
        &mut output,
        &mut [],
    )
    .expect("s3 average_pool_2d returned Err");
    assert_bit_exact(
        &output,
        &average_pool_fixture::EXPECTED_OUTPUT,
        "s3_average_pool_golden",
    );
}

#[test]
fn s3_max_pool_golden() {
    let params = pool_params_from_fixture!(max_pool_fixture);
    let mut output = [0i8; 4];
    hematite_s3::pool::max_pool_2d(&max_pool_fixture::INPUT_DATA, &params, &mut output, &mut [])
        .expect("s3 max_pool_2d returned Err");
    assert_bit_exact(
        &output,
        &max_pool_fixture::EXPECTED_OUTPUT,
        "s3_max_pool_golden",
    );
}

// ── Activations ────────────────────────────────────────────────────────────

fn identity_activation_params(input_offset: i32, output_offset: i32) -> ActivationParams<'static> {
    ActivationParams {
        input_offset,
        output_offset,
        output_multiplier: 1i32 << 30,
        output_shift: 1,
        quantized_activation_min: -128,
        quantized_activation_max: 127,
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

#[test]
fn s3_relu_golden() {
    let params = identity_activation_params(relu_fixture::INPUT_ZERO_POINT, relu_fixture::OUTPUT_ZERO_POINT);
    let mut output = [0i8; 8];
    hematite_s3::activations::relu(&relu_fixture::INPUT_DATA, &params, &mut output, &mut [])
        .expect("s3 relu returned Err");
    assert_bit_exact(&output, &relu_fixture::EXPECTED_OUTPUT, "s3_relu_golden");
}

#[test]
fn s3_relu6_golden() {
    let params = identity_activation_params(relu6_fixture::INPUT_ZERO_POINT, relu6_fixture::OUTPUT_ZERO_POINT);
    let mut output = [0i8; 8];
    hematite_s3::activations::relu6(
        &relu6_fixture::INPUT_DATA,
        &params,
        &mut output,
        &mut [],
        relu6_fixture::QUANTIZED_SIX,
    )
    .expect("s3 relu6 returned Err");
    assert_bit_exact(&output, &relu6_fixture::EXPECTED_OUTPUT, "s3_relu6_golden");
}

#[test]
fn s3_hard_swish_golden() {
    let params = identity_activation_params(
        hard_swish_fixture::INPUT_ZERO_POINT,
        hard_swish_fixture::OUTPUT_ZERO_POINT,
    );
    let mut output = [0i8; 8];
    hematite_s3::activations::hard_swish(&hard_swish_fixture::INPUT_DATA, &params, &mut output, &mut [])
        .expect("s3 hard_swish returned Err");
    assert_bit_exact(
        &output,
        &hard_swish_fixture::EXPECTED_OUTPUT,
        "s3_hard_swish_golden",
    );
}

// ── Elementwise ────────────────────────────────────────────────────────────

#[test]
fn s3_add_golden() {
    let params = ElementwiseParams {
        num_elements: 6,
        input1_offset: elementwise_add_fixture::INPUT_OFFSET,
        input2_offset: elementwise_add_fixture::INPUT2_OFFSET,
        output_offset: elementwise_add_fixture::OUTPUT_OFFSET,
        output_multiplier: elementwise_add_fixture::OUTPUT_MULTIPLIER[0],
        output_shift: elementwise_add_fixture::OUTPUT_SHIFT[0],
        left_shift: elementwise_add_fixture::LEFT_SHIFT,
        input1_multiplier: elementwise_add_fixture::INPUT1_MULTIPLIER,
        input1_shift: elementwise_add_fixture::INPUT1_SHIFT,
        input2_multiplier: elementwise_add_fixture::INPUT2_MULTIPLIER,
        input2_shift: elementwise_add_fixture::INPUT2_SHIFT,
        quantized_activation_min: elementwise_add_fixture::OUTPUT_ACTIVATION_MIN,
        quantized_activation_max: elementwise_add_fixture::OUTPUT_ACTIVATION_MAX,
    };
    let mut output = [0i8; 6];
    hematite_s3::elementwise::add(
        &elementwise_add_fixture::INPUT_DATA,
        &elementwise_add_fixture::WEIGHTS_DATA,
        &params,
        &mut output,
        &mut [],
    )
    .expect("s3 add returned Err");
    assert_bit_exact(
        &output,
        &elementwise_add_fixture::EXPECTED_OUTPUT,
        "s3_add_golden",
    );
}

#[test]
fn s3_mul_golden() {
    let params = ElementwiseParams {
        num_elements: 6,
        input1_offset: elementwise_mul_fixture::INPUT_OFFSET,
        input2_offset: elementwise_mul_fixture::INPUT2_OFFSET,
        output_offset: elementwise_mul_fixture::OUTPUT_OFFSET,
        output_multiplier: elementwise_mul_fixture::OUTPUT_MULTIPLIER[0],
        output_shift: elementwise_mul_fixture::OUTPUT_SHIFT[0],
        left_shift: 0,
        input1_multiplier: 0,
        input1_shift: 0,
        input2_multiplier: 0,
        input2_shift: 0,
        quantized_activation_min: elementwise_mul_fixture::OUTPUT_ACTIVATION_MIN,
        quantized_activation_max: elementwise_mul_fixture::OUTPUT_ACTIVATION_MAX,
    };
    let mut output = [0i8; 6];
    hematite_s3::elementwise::mul(
        &elementwise_mul_fixture::INPUT_DATA,
        &elementwise_mul_fixture::WEIGHTS_DATA,
        &params,
        &mut output,
        &mut [],
    )
    .expect("s3 mul returned Err");
    assert_bit_exact(
        &output,
        &elementwise_mul_fixture::EXPECTED_OUTPUT,
        "s3_mul_golden",
    );
}

#[test]
fn s3_sub_golden() {
    let params = ElementwiseParams {
        num_elements: 6,
        input1_offset: elementwise_sub_fixture::INPUT_OFFSET,
        input2_offset: elementwise_sub_fixture::INPUT2_OFFSET,
        output_offset: elementwise_sub_fixture::OUTPUT_OFFSET,
        output_multiplier: elementwise_sub_fixture::OUTPUT_MULTIPLIER[0],
        output_shift: elementwise_sub_fixture::OUTPUT_SHIFT[0],
        left_shift: elementwise_sub_fixture::LEFT_SHIFT,
        input1_multiplier: elementwise_sub_fixture::INPUT1_MULTIPLIER,
        input1_shift: elementwise_sub_fixture::INPUT1_SHIFT,
        input2_multiplier: elementwise_sub_fixture::INPUT2_MULTIPLIER,
        input2_shift: elementwise_sub_fixture::INPUT2_SHIFT,
        quantized_activation_min: elementwise_sub_fixture::OUTPUT_ACTIVATION_MIN,
        quantized_activation_max: elementwise_sub_fixture::OUTPUT_ACTIVATION_MAX,
    };
    let mut output = [0i8; 6];
    hematite_s3::elementwise::sub(
        &elementwise_sub_fixture::INPUT_DATA,
        &elementwise_sub_fixture::WEIGHTS_DATA,
        &params,
        &mut output,
        &mut [],
    )
    .expect("s3 sub returned Err");
    assert_bit_exact(
        &output,
        &elementwise_sub_fixture::EXPECTED_OUTPUT,
        "s3_sub_golden",
    );
}

// ── Softmax / Mean ─────────────────────────────────────────────────────────

#[test]
fn s3_softmax_golden() {
    let params = SoftmaxParams {
        num_rows: 1,
        row_size: softmax_fixture::OUTPUT_SHAPE[3],
        input_multiplier: softmax_fixture::INPUT_MULTIPLIER,
        input_left_shift: softmax_fixture::LEFT_SHIFT,
        diff_min: softmax_fixture::DIFF_MIN,
        input_offset: softmax_fixture::INPUT_OFFSET,
        output_offset: softmax_fixture::OUTPUT_OFFSET,
        quantized_activation_min: softmax_fixture::OUTPUT_ACTIVATION_MIN,
        quantized_activation_max: softmax_fixture::OUTPUT_ACTIVATION_MAX,
    };
    let mut output = [0i8; 5];
    let mut scratch = [0u8; 256];
    hematite_s3::softmax::softmax(&softmax_fixture::INPUT_DATA, &params, &mut output, &mut scratch)
        .expect("s3 softmax returned Err");
    assert_bit_exact(
        &output,
        &softmax_fixture::EXPECTED_OUTPUT,
        "s3_softmax_golden",
    );
}

#[test]
fn s3_mean_golden() {
    let params = ReduceParams {
        keep_dims: false,
        axis: [mean_fixture::AXIS_0 as i16, 0, 0, 0],
        axis_count: mean_fixture::AXIS_COUNT as i8,
        input_shape: mean_fixture::INPUT_SHAPE,
        output_shape: mean_fixture::OUTPUT_SHAPE,
        output_type: 0,
        input_offset: mean_fixture::INPUT_OFFSET,
        output_offset: mean_fixture::OUTPUT_OFFSET,
        output_multiplier: mean_fixture::OUTPUT_MULTIPLIER[0],
        output_shift: mean_fixture::OUTPUT_SHIFT[0],
        quantized_activation_min: mean_fixture::OUTPUT_ACTIVATION_MIN,
        quantized_activation_max: mean_fixture::OUTPUT_ACTIVATION_MAX,
    };
    let mut output = [0i8; 6];
    hematite_s3::reductions::mean(&mean_fixture::INPUT_DATA, &params, &mut output)
        .expect("s3 mean returned Err");
    assert_bit_exact(&output, &mean_fixture::EXPECTED_OUTPUT, "s3_mean_golden");
}
