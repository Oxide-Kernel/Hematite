// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! T1.4 — cross-crate scratch parity, end-to-end through the real `#[model]`
//! macro: the macro-time `SCRATCH_LEN` const of every annotated model must
//! equal the runtime `S3Backend` scratch need computed from the model's op
//! shapes. The runtime formulas are canonical — a divergence is a codegen
//! mirror bug.
//!
//! The mirror fns themselves are not reachable from this test crate (E0774:
//! proc-macro crates export only macros), so the parity is asserted through
//! the one artifact codegen and s3 share: `Model::SCRATCH_LEN` vs the runtime
//! `*_scratch_need` formulas over the models' documented op shapes. The same
//! corpus is swept against every mirror fn in-crate (generate.rs,
//! `scratch_parity_spec_corpus_and_grids` + `composed_scratch_parity`).

use hematite_core::op_params::{DepthwiseConv2DParams, FullyConnectedParams, Padding};
use hematite_core::KernelBackend;
use hematite_s3::backend::{fc_scratch_need, S3Backend};

mod sine {
    use hematite_codegen::model;
    #[model("../models/sine.tflite")]
    pub struct M;
}

mod hello_world {
    use hematite_codegen::model;
    #[model("../models/zoo/sine_regression/hello_world_int8.tflite")]
    pub struct M;
}

mod kws {
    use hematite_codegen::model;
    #[model("../models/zoo/keyword_spotting/kws_micro_speech_int8.tflite")]
    pub struct M;
}

mod mobilenet_v2_fused {
    use hematite_codegen::model;
    #[model("../models/zoo/mobilenetv2_cls/mobilenet_v2_1.0_224_int8.tflite")]
    pub struct M;
}

mod mobilenet_v2_unfused {
    use hematite_codegen::model_unfused;
    #[model_unfused("../models/zoo/mobilenetv2_cls/mobilenet_v2_1.0_224_int8.tflite")]
    pub struct M;
}

fn fc_params(input_dim: i32, output_dim: i32, input_offset: i32) -> FullyConnectedParams<'static> {
    FullyConnectedParams {
        input_dim,
        output_dim,
        input_offset,
        weights_offset: 0,
        output_offset: 0,
        // The scratch formulas read dims/offsets only — slice contents are
        // irrelevant to `fc_scratch_need`.
        output_multiplier_per_channel: &[],
        output_shift_per_channel: &[],
        quantized_activation_min: -128,
        quantized_activation_max: 127,
    }
}

/// The kws depthwise layer — the REAL corpus row (tflite-verified,
/// hematite-benchmarks/src/spec.rs `SIMD_DEPTHWISE_KWS_10X8_PARAMS`).
fn kws_depthwise_params() -> DepthwiseConv2DParams<'static> {
    DepthwiseConv2DParams {
        input_shape: [1, 49, 40, 1],
        filter_shape: [1, 10, 8, 8],
        output_shape: [1, 25, 20, 8],
        padding: Padding::Same,
        stride_width: 2,
        stride_height: 2,
        dilation_width_factor: 1,
        dilation_height_factor: 1,
        depth_multiplier: 8,
        input_offset: 128,
        weights_offset: 0,
        output_offset: -128,
        output_multiplier_per_channel: &[],
        output_shift_per_channel: &[],
        quantized_activation_min: 0,
        quantized_activation_max: 127,
    }
}

/// sine.tflite — 1 FULLY_CONNECTED op: input [1] (zp 0 → input_offset 0),
/// weights [1,1], bias [1], output [1] (shapes from the flatbuffer parse test
/// in src/lib.rs). The T3.6 pad16 small-shape path: input_dim 1 is zero-
/// padded to 16 in scratch, so need = 16 + 1·16 + 1·4 = 36.
#[test]
fn sine_scratch_len_matches_runtime() {
    let _ = sine::M;
    let need = fc_scratch_need(&fc_params(1, 1, 0));
    assert_eq!(
        sine::SCRATCH_LEN, need,
        "sine: macro SCRATCH_LEN {} != runtime fc_scratch_need {need}",
        sine::SCRATCH_LEN
    );
}

/// hello_world_int8.tflite — 3 FULLY_CONNECTED ops (tflite-micro sine
/// regression; op table verified from the model bytes): fc [1,1]→[1,16],
/// fc [1,16]→[1,16], fc [1,16]→[1,1], all with non-zero input offsets
/// (spec rows SIMD_FC_1X16 / SIMD_FC_16X1 document offset 128). The first
/// layer's pad16 widening (16 + 16·16 + 16·4 + 16·4 = 400) dominates.
#[test]
fn hello_world_scratch_len_matches_runtime() {
    let _ = hello_world::M;
    let need = [
        fc_scratch_need(&fc_params(1, 16, 128)),
        fc_scratch_need(&fc_params(16, 16, 128)),
        fc_scratch_need(&fc_params(16, 1, 128)),
    ]
    .into_iter()
    .max()
    .unwrap();
    assert_eq!(
        hello_world::SCRATCH_LEN, need,
        "hello_world: macro SCRATCH_LEN {} != max runtime fc_scratch_need {need}",
        hello_world::SCRATCH_LEN
    );
}

/// kws_micro_speech_int8.tflite — 4 ops (op table verified from the model
/// bytes): reshape [1,1960]→[1,49,40,1] (scratch 0), the depthwise 10×8 dm8
/// S2 layer (the corpus row — 44128 bytes: padded input 58·46·16, 80-tap
/// padded filter 80·16, accs 16·4, wsum 8·4, anytap partials 16·4), fc
/// 4000→4 (32), softmax row_size 4 (16). The depthwise dominates.
#[test]
fn kws_scratch_len_matches_runtime() {
    let _ = kws::M;
    let depthwise = S3Backend::depthwise_conv2d_scratch_size(&kws_depthwise_params());
    let fc = fc_scratch_need(&fc_params(4000, 4, 128));
    let softmax = 4 * 4;
    let need = depthwise.max(fc).max(softmax);
    assert_eq!(
        kws::SCRATCH_LEN, need,
        "kws: macro SCRATCH_LEN {} != max runtime need {need} (depthwise {depthwise}, fc {fc}, softmax {softmax})",
        kws::SCRATCH_LEN
    );
}

/// mobilenet_v2_1.0_224 — the only zoo model with composed groups (10
/// fused_conv2d residual-add groups). The fused emission reports the conv
/// anchor's own need for each composed group (`emit_fused_conv` →
/// `conv.em.scratch`) and skips the absorbed ops (scratch 0), so the fused
/// SCRATCH_LEN must equal the unfused per-op max — composed groups never
/// under-size scratch. (The composed need == runtime conv formula equality is
/// asserted in-crate by `composed_scratch_parity`.)
#[test]
fn mobilenet_v2_fused_scratch_len_equals_unfused() {
    let _ = (mobilenet_v2_fused::M, mobilenet_v2_unfused::M);
    assert_eq!(
        mobilenet_v2_fused::SCRATCH_LEN,
        mobilenet_v2_unfused::SCRATCH_LEN,
        "mobilenet_v2: fused SCRATCH_LEN {} != unfused {} — a composed group changed the macro-time max",
        mobilenet_v2_fused::SCRATCH_LEN,
        mobilenet_v2_unfused::SCRATCH_LEN,
    );
}
