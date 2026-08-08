// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Per-kernel benchmark definitions (plan T5.3).
//!
//! # Shapes
//!
//! The table mirrors the ember-esp-nn benchmark shapes named in the plan
//! (T5.3 line 308):
//!
//! * `conv_s8 8×8,64×3×3×3` — 8×8 spatial input, 64×3×3×3 filter
//! * `depthwise_conv_s8 18×18,1×3×3×16` — 18×18 input, 1×3×3×16 filter
//! * `fc_s8 271 row, 3 out ch` — 271-element input, 3 output channels
//! * `conv 1×1 64×1×1×64` — the column-2 acceptance shape (beat 15.57×)
//!
//! plus additional ESP-DL / MobileNetV2-style shapes (first conv, depthwise
//! block, 1×1 projection, classification head, softmax, global average pool).
//!
//! # Column provenance (MUST NOT invent numbers)
//!
//! Column 1 (speedup vs our scalar-Rust ref) and the raw three-column timing
//! are **measured on device** — never pre-filled.  Column 2 (ember-esp-nn
//! optimized-C) and column 3 (ESP-DL ANSI-C) require absolute cycle counts
//! sourced from those projects' public benchmark tables at device bring-up;
//! until sourced they render as `—` in the report.  The only documented
//! comparison *target* carried here is the plan's 15.57× column-2 bar and the
//! 10× internal bar (T3.0) — both attributed to the plan in `source` fields.

use hematite_core::op_params::{
    ActivationParams, Conv2DParams, DepthwiseConv2DParams, ElementwiseParams,
    FullyConnectedParams, FusedActivation, Padding, PoolParams, SoftmaxParams,
};
use hematite_core::KernelError;

/// Per-channel requantize constants (Q0.31 0.5 and shift 0).  Exact values do
/// not affect cycle timing — the per-channel requantize cost is identical.
/// They only need to be length-correct so the kernels' slice validation and
/// per-channel indexing hold.
const fn mults<const N: usize>() -> [i32; N] {
    [1 << 30; N]
}
const fn shifts<const N: usize>() -> [i32; N] {
    [0; N]
}

const MULT_3: [i32; 3] = mults::<3>();
const SHIFT_3: [i32; 3] = shifts::<3>();
const MULT_16: [i32; 16] = mults::<16>();
const SHIFT_16: [i32; 16] = shifts::<16>();
const MULT_32: [i32; 32] = mults::<32>();
const SHIFT_32: [i32; 32] = shifts::<32>();
const MULT_64: [i32; 64] = mults::<64>();
const SHIFT_64: [i32; 64] = shifts::<64>();
const MULT_1000: [i32; 1000] = mults::<1000>();
const SHIFT_1000: [i32; 1000] = shifts::<1000>();

/// Element count for the SIMD relu / elementwise bench rows (n % 16 == 0).
const RELU_NUM_ELEMENTS: usize = 256;

/// Working-set memory tier label — every report row must state SRAM or PSRAM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryTier {
    /// Working set resident in internal SRAM (≤ ~260 KB for the bench arena).
    Sram,
    /// Working set (weights or activations) placed in external PSRAM.
    Psram,
}

impl MemoryTier {
    /// Report label.
    pub const fn label(self) -> &'static str {
        match self {
            MemoryTier::Sram => "SRAM",
            MemoryTier::Psram => "PSRAM",
        }
    }
}

/// The s3 kernel entry point used for this row (the kernel calling convention
/// is exactly the public free function the s3 crate exports — no invented
/// ABI; see issues.md "Tie728ConvArgs ABI" note).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpKind {
    Conv2d1x1,
    Conv2d3x3,
    DepthwiseConv2d,
    FullyConnected,
    Softmax,
    AvgPool,
    MaxPool,
    Relu,
    Add,
    Mul,
    Sub,
}

/// A documented competitor / acceptance baseline for a row.
///
/// `cycles` is `None` until sourced from the competitor project at device
/// bring-up — the MUST-NOT-invent-numbers rule.  `target_speedup_x100` is a
/// plan-attributed acceptance target (×100 fixed point, e.g. 1557 = 15.57×).
pub struct CompetitorBaseline {
    /// Competitor name, e.g. "ember-esp-nn optimized-C".
    pub name: &'static str,
    /// Competitor absolute cycle count for the same shape — `None` until
    /// sourced (no unverified comparison numbers).
    pub cycles: Option<u64>,
    /// Plan-attributed acceptance target in ×100 fixed point.
    pub target_speedup_x100: Option<u32>,
    /// Provenance / source citation for the numbers carried above.
    pub source: &'static str,
}

/// Typed parameters for a kernel row.
pub enum KernelParams {
    Conv(&'static Conv2DParams<'static>),
    Depthwise(&'static DepthwiseConv2DParams<'static>),
    Fc(&'static FullyConnectedParams<'static>),
    Softmax(&'static SoftmaxParams),
    Pool(&'static PoolParams),
    Activation(&'static ActivationParams<'static>),
    Elementwise(&'static ElementwiseParams),
}

/// A single per-kernel benchmark row.
pub struct KernelSpec {
    /// Row label (kernel + shape).
    pub name: &'static str,
    /// Working-set tier (SRAM / PSRAM).
    pub tier: MemoryTier,
    /// s3 kernel entry point.
    pub op: OpKind,
    /// Typed kernel parameters.
    pub params: KernelParams,
    /// Comparison baseline (columns 2/3 of the report) — see provenance note.
    pub reference: Option<CompetitorBaseline>,
    /// Extra provenance / context shown in the report footer.
    pub note: &'static str,
}

// ── ember-esp-nn shapes (plan T5.3 line 308) ───────────────────────────────

const EMBER_CONV_8X8_PARAMS: Conv2DParams<'static> = Conv2DParams {
    input_shape: [1, 8, 8, 3],
    filter_shape: [64, 3, 3, 3],
    output_shape: [1, 8, 8, 64],
    padding: Padding::Same,
    stride_width: 1,
    stride_height: 1,
    dilation_width_factor: 1,
    dilation_height_factor: 1,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_64,
    output_shift_per_channel: &SHIFT_64,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

const EMBER_DEPTHWISE_18X18_PARAMS: DepthwiseConv2DParams<'static> = DepthwiseConv2DParams {
    input_shape: [1, 18, 18, 16],
    filter_shape: [1, 3, 3, 16],
    output_shape: [1, 18, 18, 16],
    padding: Padding::Same,
    stride_width: 1,
    stride_height: 1,
    dilation_width_factor: 1,
    dilation_height_factor: 1,
    depth_multiplier: 1,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_16,
    output_shift_per_channel: &SHIFT_16,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

const EMBER_FC_271_PARAMS: FullyConnectedParams<'static> = FullyConnectedParams {
    input_dim: 271,
    output_dim: 3,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_3,
    output_shift_per_channel: &SHIFT_3,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

const EMBER_CONV_1X1_64_PARAMS: Conv2DParams<'static> = Conv2DParams {
    input_shape: [1, 1, 1, 64],
    filter_shape: [64, 1, 1, 64],
    output_shape: [1, 1, 1, 64],
    padding: Padding::Same,
    stride_width: 1,
    stride_height: 1,
    dilation_width_factor: 1,
    dilation_height_factor: 1,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_64,
    output_shift_per_channel: &SHIFT_64,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

// ── ESP-DL / MobileNetV2-style shapes ──────────────────────────────────────

const MV2_FIRST_CONV_224_PARAMS: Conv2DParams<'static> = Conv2DParams {
    input_shape: [1, 224, 224, 3],
    filter_shape: [32, 3, 3, 3],
    output_shape: [1, 112, 112, 32],
    padding: Padding::Same,
    stride_width: 2,
    stride_height: 2,
    dilation_width_factor: 1,
    dilation_height_factor: 1,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_32,
    output_shift_per_channel: &SHIFT_32,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

const MV2_DEPTHWISE_112_PARAMS: DepthwiseConv2DParams<'static> = DepthwiseConv2DParams {
    input_shape: [1, 112, 112, 32],
    filter_shape: [1, 3, 3, 32],
    output_shape: [1, 112, 112, 32],
    padding: Padding::Same,
    stride_width: 1,
    stride_height: 1,
    dilation_width_factor: 1,
    dilation_height_factor: 1,
    depth_multiplier: 1,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_32,
    output_shift_per_channel: &SHIFT_32,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

const MV2_POINTWISE_112_PARAMS: Conv2DParams<'static> = Conv2DParams {
    input_shape: [1, 112, 112, 32],
    filter_shape: [64, 1, 1, 32],
    output_shape: [1, 112, 112, 64],
    padding: Padding::Same,
    stride_width: 1,
    stride_height: 1,
    dilation_width_factor: 1,
    dilation_height_factor: 1,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_64,
    output_shift_per_channel: &SHIFT_64,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

const MV2_FC_HEAD_PARAMS: Conv2DParams<'static> = Conv2DParams {
    input_shape: [1, 1, 1, 1280],
    filter_shape: [1000, 1, 1, 1280],
    output_shape: [1, 1, 1, 1000],
    padding: Padding::Same,
    stride_width: 1,
    stride_height: 1,
    dilation_width_factor: 1,
    dilation_height_factor: 1,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_1000,
    output_shift_per_channel: &SHIFT_1000,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

const SOFTMAX_1X1000_PARAMS: SoftmaxParams = SoftmaxParams {
    num_rows: 1,
    row_size: 1000,
    input_multiplier: 1_717_986_918, // quantize_multiplier(0.1), from the softmax golden
    input_left_shift: 22,
    diff_min: -128,
    input_offset: 0,
    output_offset: -128,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

const AVGPOOL_7X7_1280_PARAMS: PoolParams = PoolParams {
    input_shape: [1, 7, 7, 1280],
    output_shape: [1, 1, 1, 1280],
    filter_width: 7,
    filter_height: 7,
    stride_width: 7,
    stride_height: 7,
    padding: Padding::Valid,
    activation: FusedActivation::None,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

// ── SIMD-eligible SRAM rows (per-op C-SIMD comparison, Phase A) ────────────

/// 3×3 conv with VALID padding (pad=0) and channel counts %16 — the only
/// configuration the TIE728 `dl_tie728_s8_conv2d_33cn` gate accepts.
const SIMD_CONV3X3_32X32_PARAMS: Conv2DParams<'static> = Conv2DParams {
    input_shape: [1, 32, 32, 64],
    filter_shape: [64, 3, 3, 64],
    output_shape: [1, 30, 30, 64],
    padding: Padding::Valid,
    stride_width: 1,
    stride_height: 1,
    dilation_width_factor: 1,
    dilation_height_factor: 1,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_64,
    output_shift_per_channel: &SHIFT_64,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

/// FC with input_dim 256 and output_dim 64 (both %16) — fires the TIE728
/// `dl_tie728_s8_conv2d_11cn` path (the same entry point conv1x1 uses).
const SIMD_FC_256X64_PARAMS: FullyConnectedParams<'static> = FullyConnectedParams {
    input_dim: 256,
    output_dim: 64,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_64,
    output_shift_per_channel: &SHIFT_64,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

/// 2×2 max-pool, stride 2, VALID, channels 16 (%16) — fires
/// `dl_tie728_s8_max_pool2d_22c1` (hardcoded 2x2/stride-2 pattern).
const SIMD_MAXPOOL_32X32_PARAMS: PoolParams = PoolParams {
    input_shape: [1, 32, 32, 16],
    output_shape: [1, 16, 16, 16],
    filter_width: 2,
    filter_height: 2,
    stride_width: 2,
    stride_height: 2,
    padding: Padding::Valid,
    activation: FusedActivation::None,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

/// 2×2 avg-pool, stride 2, VALID, channels 16 (%16) — fires
/// `dl_tie728_s8_avg_pool2d_22c1`.
const SIMD_AVGPOOL_32X32_PARAMS: PoolParams = PoolParams {
    input_shape: [1, 32, 32, 16],
    output_shape: [1, 16, 16, 16],
    filter_width: 2,
    filter_height: 2,
    stride_width: 2,
    stride_height: 2,
    padding: Padding::Valid,
    activation: FusedActivation::None,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

/// ReLU over 256 elements with the identity requantize pair
/// (mult=1<<30, shift=1) and zero offsets — fires `dl_tie728_s8_relu_11c`
/// (the Phase-0-fixed dispatch gate in `hematite-s3::activations::relu`).
const SIMD_RELU_256_PARAMS: ActivationParams<'static> = ActivationParams {
    input_offset: 0,
    output_offset: 0,
    output_multiplier: 1 << 30,
    output_shift: 1,
    quantized_activation_min: 0,
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
};

/// Elementwise ADD over 256 elements with the full identity contract
/// (zero offsets, no left shift, all (mult,shift) pairs == (1<<30,1), full
/// int8 activation range) — fires `dl_tie728_s8_add_w1_16_w2_16`.
const SIMD_ADD_256_PARAMS: ElementwiseParams = ElementwiseParams {
    num_elements: 256,
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
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

/// Elementwise MUL over 256 elements with zero offsets and the identity
/// output pair (mult=1<<30, shift≤1) — fires `dl_tie728_s8_mul_w1_16_w2_16`
/// with `mul_shift = 1 - output_shift`.
const SIMD_MUL_256_PARAMS: ElementwiseParams = ElementwiseParams {
    num_elements: 256,
    input1_offset: 0,
    input2_offset: 0,
    output_offset: 0,
    output_multiplier: 1 << 30,
    output_shift: 0,
    left_shift: 0,
    input1_multiplier: 1 << 30,
    input1_shift: 1,
    input2_multiplier: 1 << 30,
    input2_shift: 1,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

/// Elementwise SUB over 256 elements with the same identity contract as ADD
/// — fires `dl_tie728_s8_sub_w1_16_w2_16`.
const SIMD_SUB_256_PARAMS: ElementwiseParams = ElementwiseParams {
    num_elements: 256,
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
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

/// The full per-kernel benchmark table (order = report row order).
pub const fn kernel_specs() -> &'static [KernelSpec] {
    &[
        KernelSpec {
            name: "conv_s8 8x8,64x3x3x3 (ember-esp-nn)",
            tier: MemoryTier::Sram,
            op: OpKind::Conv2d3x3,
            params: KernelParams::Conv(&EMBER_CONV_8X8_PARAMS),
            reference: Some(CompetitorBaseline {
                name: "ember-esp-nn optimized-C",
                cycles: None,
                target_speedup_x100: None,
                source: "plan T5.3: shape 'conv_s8 8×8,64×3×3×3'; absolute cycle count to be sourced from ember-esp-nn repo at bring-up",
            }),
            note: "Shape parse: input [1,8,8,3], filter [64,3,3,3]. Verify against ember-esp-nn README table at bring-up.",
        },
        KernelSpec {
            name: "depthwise_conv_s8 18x18,1x3x3x16 (ember-esp-nn)",
            tier: MemoryTier::Sram,
            op: OpKind::DepthwiseConv2d,
            params: KernelParams::Depthwise(&EMBER_DEPTHWISE_18X18_PARAMS),
            reference: Some(CompetitorBaseline {
                name: "ember-esp-nn optimized-C",
                cycles: None,
                target_speedup_x100: None,
                source: "plan T5.3: shape 'depthwise_conv_s8 18×18,1×3×3×16'",
            }),
            note: "Depth multiplier 1, stride 1, SAME.",
        },
        KernelSpec {
            name: "fc_s8 271row,3out (ember-esp-nn)",
            tier: MemoryTier::Sram,
            op: OpKind::FullyConnected,
            params: KernelParams::Fc(&EMBER_FC_271_PARAMS),
            reference: Some(CompetitorBaseline {
                name: "ember-esp-nn optimized-C",
                cycles: None,
                target_speedup_x100: None,
                source: "plan T5.3: shape 'fc_s8 271 row, 3 out ch'",
            }),
            note: "input_dim 271, output_dim 3 (matmul 3×271).",
        },
        KernelSpec {
            name: "conv1x1_s8 64x1x1x64 (ember-esp-nn 15.57x bar)",
            tier: MemoryTier::Sram,
            op: OpKind::Conv2d1x1,
            params: KernelParams::Conv(&EMBER_CONV_1X1_64_PARAMS),
            reference: Some(CompetitorBaseline {
                name: "ember-esp-nn optimized-C",
                cycles: None,
                target_speedup_x100: Some(1557),
                source: "plan T5.3 line 309: 'Column-2: beat ember-esp-nn's 15.57× on conv 1×1 64×1×1×64'",
            }),
            note: "Column-2 acceptance bar: our speedup vs our scalar ref must exceed 15.57×.",
        },
        KernelSpec {
            name: "conv3x3_s8 32x32,64x3x3x64 VALID (SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::Conv2d3x3,
            params: KernelParams::Conv(&SIMD_CONV3X3_32X32_PARAMS),
            reference: None,
            note: "TIE728 33cn SIMD row: pad=0 (VALID), channels %16, mult/shift uniform.",
        },
        KernelSpec {
            name: "fc_s8 256row,64out (SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::FullyConnected,
            params: KernelParams::Fc(&SIMD_FC_256X64_PARAMS),
            reference: None,
            note: "TIE728 11cn SIMD row: input_dim 256 (%16) x output_dim 64 (%16).",
        },
        KernelSpec {
            name: "max_pool_s8 2x2x16 (SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::MaxPool,
            params: KernelParams::Pool(&SIMD_MAXPOOL_32X32_PARAMS),
            reference: None,
            note: "TIE728 max_pool2d_22c1 SIMD row: 2x2 stride2 VALID, channels 16.",
        },
        KernelSpec {
            name: "avg_pool_s8 2x2x16 (SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::AvgPool,
            params: KernelParams::Pool(&SIMD_AVGPOOL_32X32_PARAMS),
            reference: None,
            note: "TIE728 avg_pool2d_22c1 SIMD row: 2x2 stride2 VALID, channels 16.",
        },
        KernelSpec {
            name: "relu_s8 256 (SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::Relu,
            params: KernelParams::Activation(&SIMD_RELU_256_PARAMS),
            reference: None,
            note: "TIE728 relu_11c SIMD row: identity requantize (1<<30,1), n=256.",
        },
        KernelSpec {
            name: "add_s8 256 (SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::Add,
            params: KernelParams::Elementwise(&SIMD_ADD_256_PARAMS),
            reference: None,
            note: "TIE728 add_w1_16_w2_16 SIMD row: identity contract, n=256.",
        },
        KernelSpec {
            name: "mul_s8 256 (SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::Mul,
            params: KernelParams::Elementwise(&SIMD_MUL_256_PARAMS),
            reference: None,
            note: "TIE728 mul_w1_16_w2_16 SIMD row: output pair (1<<30,0), n=256.",
        },
        KernelSpec {
            name: "sub_s8 256 (SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::Sub,
            params: KernelParams::Elementwise(&SIMD_SUB_256_PARAMS),
            reference: None,
            note: "TIE728 sub_w1_16_w2_16 SIMD row: identity contract, n=256.",
        },
        KernelSpec {
            name: "conv_s8 224x224x3->32 (ESP-DL/MobileNetV2 first layer)",
            tier: MemoryTier::Psram,
            op: OpKind::Conv2d3x3,
            params: KernelParams::Conv(&MV2_FIRST_CONV_224_PARAMS),
            reference: Some(CompetitorBaseline {
                name: "ESP-DL ANSI-C",
                cycles: None,
                target_speedup_x100: None,
                source: "ESP-DL public benchmark claims 26–77× (C-vs-C, plan T5.3 line 308) — column 3 is reported separately, never conflated with column 1",
            }),
            note: "Stride 2, SAME. Working set ~552 KB (150 KB in + 401 KB out) → PSRAM row.",
        },
        KernelSpec {
            name: "depthwise_s8 112x112x32 (MobileNetV2 block)",
            tier: MemoryTier::Psram,
            op: OpKind::DepthwiseConv2d,
            params: KernelParams::Depthwise(&MV2_DEPTHWISE_112_PARAMS),
            reference: None,
            note: "MobileNetV2 depthwise-separable block, stride 1.",
        },
        KernelSpec {
            name: "conv1x1_s8 112x112,32->64 (MobileNetV2 projection)",
            tier: MemoryTier::Psram,
            op: OpKind::Conv2d1x1,
            params: KernelParams::Conv(&MV2_POINTWISE_112_PARAMS),
            reference: None,
            note: "1×1 pointwise projection, output 802 KB → PSRAM row.",
        },
        KernelSpec {
            name: "conv1x1_s8 1x1x1280->1000 (MobileNetV2 head)",
            tier: MemoryTier::Psram,
            op: OpKind::Conv2d1x1,
            params: KernelParams::Conv(&MV2_FC_HEAD_PARAMS),
            reference: None,
            note: "1.28 MB weight tensor → PSRAM row.",
        },
        KernelSpec {
            name: "softmax_s8 1x1000",
            tier: MemoryTier::Sram,
            op: OpKind::Softmax,
            params: KernelParams::Softmax(&SOFTMAX_1X1000_PARAMS),
            reference: None,
            note: "T3.3: softmax is memory-bound, scalar-only.",
        },
        KernelSpec {
            name: "avg_pool_s8 7x7x1280 (global)",
            tier: MemoryTier::Sram,
            op: OpKind::AvgPool,
            params: KernelParams::Pool(&AVGPOOL_7X7_1280_PARAMS),
            reference: None,
            note: "Global average pool, filter 7×7 stride 7.",
        },
    ]
}

/// Buffer slice lengths for a spec — computed from the spec shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpecLayout {
    pub input_len: usize,
    pub weights_len: usize,
    pub bias_len: usize,
    pub output_len: usize,
    /// Byte length of the TIE728-layout weight transform region — equal to
    /// `weights_len` for the SIMD-capable weighted ops (1×1/3×3 conv, fc),
    /// 0 otherwise. The prepared/SIMD path needs weights in the asm's
    /// `[g][ic][lane]` / `[g][tap][ic][lane]` layout, which is produced by
    /// `transform_bufs` once per spec (a model-build-time op, kept outside the
    /// timed window).
    pub transform_len: usize,
}

fn shape_product(shape: &[i32; 4]) -> usize {
    shape[0] as usize * shape[1] as usize * shape[2] as usize * shape[3] as usize
}

/// Compute the buffer layout a spec needs.
pub fn layout(spec: &KernelSpec) -> SpecLayout {
    match &spec.params {
        KernelParams::Conv(p) => SpecLayout {
            input_len: shape_product(&p.input_shape),
            weights_len: shape_product(&p.filter_shape),
            bias_len: p.filter_shape[0] as usize,
            output_len: shape_product(&p.output_shape),
            transform_len: shape_product(&p.filter_shape),
        },
        KernelParams::Depthwise(p) => SpecLayout {
            input_len: shape_product(&p.input_shape),
            weights_len: shape_product(&p.filter_shape),
            bias_len: p.output_shape[3] as usize,
            output_len: shape_product(&p.output_shape),
            transform_len: 0,
        },
        KernelParams::Fc(p) => SpecLayout {
            input_len: p.input_dim as usize,
            weights_len: p.input_dim as usize * p.output_dim as usize,
            bias_len: p.output_dim as usize,
            output_len: p.output_dim as usize,
            transform_len: p.input_dim as usize * p.output_dim as usize,
        },
        KernelParams::Softmax(p) => SpecLayout {
            input_len: p.num_rows as usize * p.row_size as usize,
            weights_len: 0,
            bias_len: 0,
            output_len: p.num_rows as usize * p.row_size as usize,
            transform_len: 0,
        },
        KernelParams::Pool(p) => SpecLayout {
            input_len: shape_product(&p.input_shape),
            weights_len: 0,
            bias_len: 0,
            output_len: shape_product(&p.output_shape),
            transform_len: 0,
        },
        // Relu acts on a single flat int8 tensor: input_len == output_len,
        // no weights or bias. The element count is carried in the spec name
        // but the slices' lengths are what the kernel validates.
        KernelParams::Activation(p) => {
            // Relu has no num_elements field — the caller's input slice
            // length drives the kernel; the spec row fixes it at 256 for
            // SIMD eligibility. We encode it via a fixed constant.
            let n = RELU_NUM_ELEMENTS;
            SpecLayout {
                input_len: n,
                weights_len: 0,
                bias_len: 0,
                output_len: n,
                transform_len: 0,
            }
        }
        // Elementwise ops consume input1=input slice, input2=weights slice
        // (both length n), produce n outputs.
        KernelParams::Elementwise(p) => SpecLayout {
            input_len: p.num_elements as usize,
            weights_len: p.num_elements as usize,
            bias_len: 0,
            output_len: p.num_elements as usize,
            transform_len: 0,
        },
    }
}

/// Mutable slices backing a kernel invocation.
pub struct SpecBufs<'a> {
    pub input: &'a mut [i8],
    pub weights: &'a mut [i8],
    pub bias: &'a mut [i32],
    pub output: &'a mut [i8],
    /// TIE728-layout transformed weights (see `SpecLayout::transform_len`).
    /// Empty when the op needs no transform.
    pub transformed: &'a mut [i8],
}

/// Reinterpret a byte slice as an int8 slice.
///
/// # Safety
///
/// `u8` and `i8` have identical size/alignment/layout; the reborrow keeps the
/// same lifetime and mutability.  Sound whenever the source borrow is valid.
unsafe fn cast_i8(s: &mut [u8]) -> &mut [i8] {
    core::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut i8, s.len())
}

/// Carve the four kernel buffers out of an arena for one spec.
///
/// Offsets are aligned so input/weights/output sit on 16-byte boundaries (the
/// SIMD path requires 16-byte alignment for `EE.VLD.128` / `EE.VST.128`) and
/// the bias on a 16-byte boundary too (the TIE728 SIMD kernels gate on a
/// 16-aligned bias pointer — `conv1x1.rs`/`conv3x3.rs` check `b_ptr % 16 == 0`).
/// The layout is `[input][pad][weights][pad][bias][pad][output][pad][transformed]`.
/// Returns
/// [`KernelError::ScratchTooSmall`] when the arena cannot hold the working
/// set.
///
/// This function is **host-compiled and tested** so the device firmware can
/// rely on it without a device build (issues.md: every device-only code path
/// needs a host-side proof).
pub fn carve_into<'a>(arena: &'a mut [u8], lay: &SpecLayout) -> Result<SpecBufs<'a>, KernelError> {
    let align16 = |o: usize| o.div_ceil(16) * 16;

    // Byte ranges (bias is stored as `lay.bias_len` i32 values).
    let in_start = 0usize;
    let in_end = in_start + lay.input_len;
    let w_start = align16(in_end);
    let w_end = w_start + lay.weights_len;
    let b_start = align16(w_end);
    let b_end = b_start + lay.bias_len * 4;
    let o_start = align16(b_end);
    let o_end = o_start + lay.output_len;
    let t_start = align16(o_end);
    let t_end = t_start + lay.transform_len;

    if t_end > arena.len() {
        return Err(KernelError::ScratchTooSmall);
    }

    // Sequential disjoint splits: input, then weights, then bias region, then
    // output, then the transformed-weights region.  The bias region is
    // re-cast to `&mut [i32]`.
    let (input_region, rest) = arena.split_at_mut(w_start);
    let input = unsafe { cast_i8(&mut input_region[in_start..in_end]) };

    let (weights_region, rest) = rest.split_at_mut(b_start - w_start);
    let weights = unsafe { cast_i8(&mut weights_region[..lay.weights_len]) };

    let (bias_region, rest) = rest.split_at_mut(o_start - b_start);
    // SAFETY: `bias_region` starts at a 4-byte-aligned offset and holds
    // exactly `bias_len * 4` bytes; the resulting slice is the sole mutable
    // view of that memory (it never overlaps the input/weights/output
    // regions, whose ranges are disjoint by construction).
    let bias = unsafe {
        core::slice::from_raw_parts_mut(bias_region.as_mut_ptr() as *mut i32, lay.bias_len)
    };

    let (output_region, rest) = rest.split_at_mut(lay.output_len);
    let output = unsafe { cast_i8(&mut output_region[..]) };

    let (transform_region, _after) = rest.split_at_mut(lay.transform_len);
    let transformed = unsafe { cast_i8(&mut transform_region[..]) };

    Ok(SpecBufs {
        input,
        weights,
        bias,
        output,
        transformed,
    })
}

/// Fill buffers with a deterministic, position-varying pattern so MAC values
/// are non-trivial (timing is pattern-independent, but a flat zero fill would
/// hide data-dependence bugs on first bring-up).
pub fn fill_pattern(bufs: &mut SpecBufs<'_>) {
    for (i, v) in bufs.input.iter_mut().enumerate() {
        *v = (i.wrapping_mul(7).wrapping_add(3) & 0xFF) as i8;
    }
    for (i, v) in bufs.weights.iter_mut().enumerate() {
        *v = (i.wrapping_mul(13).wrapping_add(11) & 0xFF) as i8;
    }
    for (i, v) in bufs.bias.iter_mut().enumerate() {
        *v = i.wrapping_mul(17) as i32 - 8;
    }
    for v in bufs.output.iter_mut() {
        *v = 0;
    }
}

/// Transform `bufs.weights` into the TIE728 SIMD layout in `bufs.transformed`
/// for the SIMD-capable weighted ops (1×1/3×3 conv, fc); no-op otherwise.
///
/// This is the **model-build-time** weight permutation that makes the SIMD
/// output bit-identical to the scalar reference (the vendored asm consumes the
/// filter in `[g][ic][lane]` / `[g][tap][ic][lane]` order, not the caller's
/// `[oc][ic]` / `[oc][fh][fw][ic]` OHWI order). It is kept OUTSIDE the timed
/// benchmark window so the measured prepared/SIMD cost is pure kernel cost.
/// Host-compilable and unit-tested.
pub fn transform_bufs(spec: &KernelSpec, bufs: &mut SpecBufs<'_>) -> Result<(), KernelError> {
    if bufs.transformed.is_empty() {
        return Ok(());
    }
    match &spec.params {
        KernelParams::Conv(p) => {
            let input_c = p.filter_shape[3] as usize;
            let out_channels = p.filter_shape[0] as usize;
            if p.filter_shape[1] == 1 && p.filter_shape[2] == 1 {
                hematite_s3::conv1x1::transform_weights_11cn(
                    input_c,
                    out_channels,
                    bufs.weights,
                    bufs.transformed,
                )
            } else {
                hematite_s3::conv3x3::transform_weights_33cn(
                    input_c,
                    out_channels,
                    bufs.weights,
                    bufs.transformed,
                )
            }
        }
        KernelParams::Fc(p) => hematite_s3::conv1x1::transform_weights_11cn(
            p.input_dim as usize,
            p.output_dim as usize,
            bufs.weights,
            bufs.transformed,
        ),
        _ => Ok(()),
    }
}

/// Dispatch a spec through the exact public s3 kernel free function the crate
/// exports (device-only — the same functions run on host in the cross-check
/// test, see below).
///
/// # ABI
///
/// The calling convention is **the s3 public API as written** —
/// `hematite-s3::conv1x1::conv2d_1x1`, `conv3x3::conv2d_3x3`,
/// `depthwise::depthwise_conv2d`, `gemm::fully_connected`,
/// `softmax::softmax`, `pool::average_pool_2d`.  No invented entry points
/// (issues.md: Tie728ConvArgs ABI lesson).
#[cfg(target_arch = "xtensa")]
pub fn run_kernel(spec: &KernelSpec, bufs: &mut SpecBufs<'_>, scratch: &mut [u8]) -> Result<(), KernelError> {
    match spec.op {
        OpKind::Conv2d1x1 => {
            let p = params_conv(spec);
            hematite_s3::conv1x1::conv2d_1x1(bufs.input, bufs.weights, bufs.bias, p, bufs.output, scratch)
        }        OpKind::Conv2d3x3 => {
            let p = params_conv(spec);
            hematite_s3::conv3x3::conv2d_3x3(bufs.input, bufs.weights, bufs.bias, p, bufs.output, scratch)
        }
        OpKind::DepthwiseConv2d => {
            let p = params_depthwise(spec);
            hematite_s3::depthwise::depthwise_conv2d(bufs.input, bufs.weights, bufs.bias, p, bufs.output, scratch)
        }
        OpKind::FullyConnected => {
            let p = params_fc(spec);
            hematite_s3::gemm::fully_connected(bufs.input, bufs.weights, bufs.bias, p, bufs.output, scratch)
        }
        OpKind::Softmax => {
            let p = params_softmax(spec);
            hematite_s3::softmax::softmax(bufs.input, p, bufs.output, scratch)
        }
        OpKind::AvgPool => {
            let p = params_pool(spec);
            hematite_s3::pool::average_pool_2d(bufs.input, p, bufs.output, scratch)
        }
        OpKind::MaxPool => {
            let p = params_pool(spec);
            hematite_s3::pool::max_pool_2d(bufs.input, p, bufs.output, scratch)
        }
        OpKind::Relu => {
            let p = params_activation(spec);
            hematite_s3::activations::relu(bufs.input, p, bufs.output, scratch)
        }
        OpKind::Add => {
            let p = params_elementwise(spec);
            hematite_s3::elementwise::add(bufs.input, bufs.weights, p, bufs.output, scratch)
        }
        OpKind::Mul => {
            let p = params_elementwise(spec);
            hematite_s3::elementwise::mul(bufs.input, bufs.weights, p, bufs.output, scratch)
        }
        OpKind::Sub => {
            let p = params_elementwise(spec);
            hematite_s3::elementwise::sub(bufs.input, bufs.weights, p, bufs.output, scratch)
        }
    }
}

/// A prepared kernel handle: the SIMD eligibility gate is run ONCE at
/// construction (`prepare_kernel`), then `run` only re-checks pointer
/// alignment and dispatches to the TIE728 entry — closing the wrapper
/// overhead the firmware measures on the legacy public-API path.
///
/// Host-compilable: on host every handle is non-SIMD and `run` falls through
/// to the scalar kernel, so the prepared path is bit-exact vs the scalar ref
/// in host tests. On device (`xtensa`, non-qemu) the SIMD gate fires for the
/// `SIMD_*` bench rows.
#[allow(clippy::large_enum_variant)]
pub enum PreparedKernel {
    Conv1x1(hematite_s3::conv1x1::PreparedConv1x1),
    Conv3x3(hematite_s3::conv3x3::PreparedConv3x3),
    Fc(hematite_s3::gemm::PreparedFc),
    MaxPool(hematite_s3::pool::PreparedMaxPool),
    AvgPool(hematite_s3::pool::PreparedAvgPool),
    Relu(hematite_s3::activations::PreparedRelu),
    Add(hematite_s3::elementwise::PreparedAdd),
    Mul(hematite_s3::elementwise::PreparedMul),
    Sub(hematite_s3::elementwise::PreparedSub),
    /// Ops with no SIMD path (depthwise, softmax): just run the public API.
    Scalar,
}

impl PreparedKernel {
    /// Whether this handle will dispatch to a TIE728 SIMD entry on this
    /// target (device-only; always `false` on host, where SIMD is compiled
    /// out). Used by the firmware to decide whether to pre-transform weights
    /// into the SIMD layout before the timed window.
    pub fn is_simd(&self) -> bool {
        match self {
            PreparedKernel::Conv1x1(h) => h.is_simd(),
            PreparedKernel::Conv3x3(h) => h.is_simd(),
            PreparedKernel::Fc(h) => h.is_simd(),
            PreparedKernel::MaxPool(h) => h.is_simd(),
            PreparedKernel::AvgPool(h) => h.is_simd(),
            PreparedKernel::Relu(h) => h.is_simd(),
            PreparedKernel::Add(h) => h.is_simd(),
            PreparedKernel::Mul(h) => h.is_simd(),
            PreparedKernel::Sub(h) => h.is_simd(),
            PreparedKernel::Scalar => false,
        }
    }

    /// Run the prepared kernel. `spec` is only consulted for the `Scalar`
    /// fallback (depthwise/softmax); the handle variants dispatch directly.
    pub fn run(
        &self,
        spec: &KernelSpec,
        bufs: &mut SpecBufs<'_>,
        scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        match self {
            PreparedKernel::Conv1x1(h) => {
                h.run(bufs.input, bufs.weights, bufs.bias, bufs.output, scratch)
            }
            PreparedKernel::Conv3x3(h) => {
                h.run(bufs.input, bufs.weights, bufs.bias, bufs.output, scratch)
            }
            PreparedKernel::Fc(h) => {
                h.run(bufs.input, bufs.weights, bufs.bias, bufs.output, scratch)
            }
            PreparedKernel::MaxPool(h) => h.run(bufs.input, bufs.output, scratch),
            PreparedKernel::AvgPool(h) => h.run(bufs.input, bufs.output, scratch),
            PreparedKernel::Relu(h) => h.run(bufs.input, bufs.output, scratch),
            PreparedKernel::Add(h) => {
                h.run(bufs.input, bufs.weights, bufs.output, scratch)
            }
            PreparedKernel::Mul(h) => {
                h.run(bufs.input, bufs.weights, bufs.output, scratch)
            }
            PreparedKernel::Sub(h) => {
                h.run(bufs.input, bufs.weights, bufs.output, scratch)
            }
            PreparedKernel::Scalar => run_kernel_scalar(spec, bufs, scratch),
        }
    }
}

/// Host/device-compilable scalar dispatch for the `PreparedKernel::Scalar`
/// fallback (depthwise, softmax — ops with no SIMD path). Mirrors
/// `run_kernel` but never touches the (xtensa-gated) `run_kernel`/`params_*`.
fn run_kernel_scalar(
    spec: &KernelSpec,
    bufs: &mut SpecBufs<'_>,
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    match spec.op {
        OpKind::Conv2d1x1 => {
            let p = match spec.params {
                KernelParams::Conv(p) => p,
                _ => return Err(KernelError::Unsupported),
            };
            hematite_s3::conv1x1::conv2d_1x1(bufs.input, bufs.weights, bufs.bias, p, bufs.output, scratch)
        }
        OpKind::Conv2d3x3 => {
            let p = match spec.params {
                KernelParams::Conv(p) => p,
                _ => return Err(KernelError::Unsupported),
            };
            hematite_s3::conv3x3::conv2d_3x3(bufs.input, bufs.weights, bufs.bias, p, bufs.output, scratch)
        }
        OpKind::DepthwiseConv2d => {
            let p = match spec.params {
                KernelParams::Depthwise(p) => p,
                _ => return Err(KernelError::Unsupported),
            };
            hematite_s3::depthwise::depthwise_conv2d(bufs.input, bufs.weights, bufs.bias, p, bufs.output, scratch)
        }
        OpKind::FullyConnected => {
            let p = match spec.params {
                KernelParams::Fc(p) => p,
                _ => return Err(KernelError::Unsupported),
            };
            hematite_s3::gemm::fully_connected(bufs.input, bufs.weights, bufs.bias, p, bufs.output, scratch)
        }
        OpKind::Softmax => {
            let p = match spec.params {
                KernelParams::Softmax(p) => p,
                _ => return Err(KernelError::Unsupported),
            };
            hematite_s3::softmax::softmax(bufs.input, p, bufs.output, scratch)
        }
        OpKind::AvgPool => {
            let p = match spec.params {
                KernelParams::Pool(p) => p,
                _ => return Err(KernelError::Unsupported),
            };
            hematite_s3::pool::average_pool_2d(bufs.input, p, bufs.output, scratch)
        }
        OpKind::MaxPool => {
            let p = match spec.params {
                KernelParams::Pool(p) => p,
                _ => return Err(KernelError::Unsupported),
            };
            hematite_s3::pool::max_pool_2d(bufs.input, p, bufs.output, scratch)
        }
        OpKind::Relu => {
            let p = match spec.params {
                KernelParams::Activation(p) => p,
                _ => return Err(KernelError::Unsupported),
            };
            hematite_s3::activations::relu(bufs.input, p, bufs.output, scratch)
        }
        OpKind::Add => {
            let p = match spec.params {
                KernelParams::Elementwise(p) => p,
                _ => return Err(KernelError::Unsupported),
            };
            hematite_s3::elementwise::add(bufs.input, bufs.weights, p, bufs.output, scratch)
        }
        OpKind::Mul => {
            let p = match spec.params {
                KernelParams::Elementwise(p) => p,
                _ => return Err(KernelError::Unsupported),
            };
            hematite_s3::elementwise::mul(bufs.input, bufs.weights, p, bufs.output, scratch)
        }
        OpKind::Sub => {
            let p = match spec.params {
                KernelParams::Elementwise(p) => p,
                _ => return Err(KernelError::Unsupported),
            };
            hematite_s3::elementwise::sub(bufs.input, bufs.weights, p, bufs.output, scratch)
        }
    }
}

/// Build a [`PreparedKernel`] from a spec — runs the SIMD gate once.
///
/// Uses the `&'static` params stored in `spec.params` directly (host- and
/// device-compilable, unlike the xtensa-gated `params_*` accessors).
pub fn prepare_kernel(spec: &KernelSpec) -> Result<PreparedKernel, KernelError> {
    match spec.op {
        OpKind::Conv2d1x1 => match spec.params {
            KernelParams::Conv(p) => Ok(PreparedKernel::Conv1x1(
                hematite_s3::conv1x1::PreparedConv1x1::new(p)?,
            )),
            _ => Err(KernelError::Unsupported),
        },
        OpKind::Conv2d3x3 => match spec.params {
            KernelParams::Conv(p) => Ok(PreparedKernel::Conv3x3(
                hematite_s3::conv3x3::PreparedConv3x3::new(p)?,
            )),
            _ => Err(KernelError::Unsupported),
        },
        OpKind::FullyConnected => match spec.params {
            KernelParams::Fc(p) => Ok(PreparedKernel::Fc(
                hematite_s3::gemm::PreparedFc::new(p)?,
            )),
            _ => Err(KernelError::Unsupported),
        },
        OpKind::MaxPool => match spec.params {
            KernelParams::Pool(p) => Ok(PreparedKernel::MaxPool(
                hematite_s3::pool::PreparedMaxPool::new(p)?,
            )),
            _ => Err(KernelError::Unsupported),
        },
        OpKind::AvgPool => match spec.params {
            KernelParams::Pool(p) => Ok(PreparedKernel::AvgPool(
                hematite_s3::pool::PreparedAvgPool::new(p)?,
            )),
            _ => Err(KernelError::Unsupported),
        },
        OpKind::Relu => match spec.params {
            KernelParams::Activation(p) => Ok(PreparedKernel::Relu(
                hematite_s3::activations::PreparedRelu::new(p)?,
            )),
            _ => Err(KernelError::Unsupported),
        },
        OpKind::Add => match spec.params {
            KernelParams::Elementwise(p) => Ok(PreparedKernel::Add(
                hematite_s3::elementwise::PreparedAdd::new(p)?,
            )),
            _ => Err(KernelError::Unsupported),
        },
        OpKind::Mul => match spec.params {
            KernelParams::Elementwise(p) => Ok(PreparedKernel::Mul(
                hematite_s3::elementwise::PreparedMul::new(p)?,
            )),
            _ => Err(KernelError::Unsupported),
        },
        OpKind::Sub => match spec.params {
            KernelParams::Elementwise(p) => Ok(PreparedKernel::Sub(
                hematite_s3::elementwise::PreparedSub::new(p)?,
            )),
            _ => Err(KernelError::Unsupported),
        },
        OpKind::DepthwiseConv2d | OpKind::Softmax => Ok(PreparedKernel::Scalar),
    }
}

#[cfg(target_arch = "xtensa")]
fn params_conv(spec: &KernelSpec) -> &'static Conv2DParams<'static> {
    match spec.params {
        KernelParams::Conv(p) => p,
        _ => panic!("spec.op conv requires KernelParams::Conv"),
    }
}
#[cfg(target_arch = "xtensa")]
fn params_depthwise(spec: &KernelSpec) -> &'static DepthwiseConv2DParams<'static> {
    match spec.params {
        KernelParams::Depthwise(p) => p,
        _ => panic!("spec.op depthwise requires KernelParams::Depthwise"),
    }
}
#[cfg(target_arch = "xtensa")]
fn params_fc(spec: &KernelSpec) -> &'static FullyConnectedParams<'static> {
    match spec.params {
        KernelParams::Fc(p) => p,
        _ => panic!("spec.op fc requires KernelParams::Fc"),
    }
}
#[cfg(target_arch = "xtensa")]
fn params_softmax(spec: &KernelSpec) -> &'static SoftmaxParams {
    match spec.params {
        KernelParams::Softmax(p) => p,
        _ => panic!("spec.op softmax requires KernelParams::Softmax"),
    }
}
#[cfg(target_arch = "xtensa")]
fn params_pool(spec: &KernelSpec) -> &'static PoolParams {
    match spec.params {
        KernelParams::Pool(p) => p,
        _ => panic!("spec.op pool requires KernelParams::Pool"),
    }
}
#[cfg(target_arch = "xtensa")]
fn params_activation(spec: &KernelSpec) -> &'static ActivationParams<'static> {
    match spec.params {
        KernelParams::Activation(p) => p,
        _ => panic!("spec.op activation requires KernelParams::Activation"),
    }
}
#[cfg(target_arch = "xtensa")]
fn params_elementwise(spec: &KernelSpec) -> &'static ElementwiseParams {
    match spec.params {
        KernelParams::Elementwise(p) => p,
        _ => panic!("spec.op elementwise requires KernelParams::Elementwise"),
    }
}

/// Dispatch a spec through the matching `hematite-ref` scalar kernel — the
/// column-1 baseline, measured on device (never a pre-filled number).
#[cfg(target_arch = "xtensa")]
pub fn run_ref_kernel(
    spec: &KernelSpec,
    bufs: &mut SpecBufs<'_>,
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    match spec.op {
        OpKind::Conv2d1x1 | OpKind::Conv2d3x3 => {
            let p = params_conv(spec);
            hematite_ref::conv::conv2d(bufs.input, bufs.weights, bufs.bias, p, bufs.output, scratch)
        }
        OpKind::DepthwiseConv2d => {
            let p = params_depthwise(spec);
            hematite_ref::depthwise_conv::depthwise_conv2d(
                bufs.input, bufs.weights, bufs.bias, p, bufs.output, scratch,
            )
        }
        OpKind::FullyConnected => {
            let p = params_fc(spec);
            hematite_ref::fully_connected::fully_connected(
                bufs.input, bufs.weights, bufs.bias, p, bufs.output, scratch,
            )
        }
        OpKind::Softmax => {
            let p = params_softmax(spec);
            hematite_ref::softmax::softmax(bufs.input, p, bufs.output, scratch)
        }
        OpKind::AvgPool => {
            let p = params_pool(spec);
            hematite_ref::pool::average_pool_2d(bufs.input, p, bufs.output, scratch)
        }
        OpKind::MaxPool => {
            let p = params_pool(spec);
            hematite_ref::pool::max_pool_2d(bufs.input, p, bufs.output, scratch)
        }
        OpKind::Relu => {
            let p = params_activation(spec);
            hematite_ref::activation::relu(bufs.input, p, bufs.output, scratch)
        }
        OpKind::Add => {
            let p = params_elementwise(spec);
            hematite_ref::elementwise::add(bufs.input, bufs.weights, p, bufs.output, scratch)
        }
        OpKind::Mul => {
            let p = params_elementwise(spec);
            hematite_ref::elementwise::mul(bufs.input, bufs.weights, p, bufs.output, scratch)
        }
        OpKind::Sub => {
            let p = params_elementwise(spec);
            hematite_ref::elementwise::sub(bufs.input, bufs.weights, p, bufs.output, scratch)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run the s3 kernel for a spec and return its output.
    fn run_s3(spec: &KernelSpec, bufs: &mut SpecBufs<'_>, scratch: &mut [u8]) -> Result<Vec<i8>, String> {
        let out = bufs.output.to_vec();
        let r = match spec.op {
            OpKind::Conv2d1x1 => {
                let p = match spec.params {
                    KernelParams::Conv(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                hematite_s3::conv1x1::conv2d_1x1(bufs.input, bufs.weights, bufs.bias, p, bufs.output, scratch)
            }
            OpKind::Conv2d3x3 => {
                let p = match spec.params {
                    KernelParams::Conv(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                hematite_s3::conv3x3::conv2d_3x3(bufs.input, bufs.weights, bufs.bias, p, bufs.output, scratch)
            }
            OpKind::DepthwiseConv2d => {
                let p = match spec.params {
                    KernelParams::Depthwise(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                hematite_s3::depthwise::depthwise_conv2d(bufs.input, bufs.weights, bufs.bias, p, bufs.output, scratch)
            }
            OpKind::FullyConnected => {
                let p = match spec.params {
                    KernelParams::Fc(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                hematite_s3::gemm::fully_connected(bufs.input, bufs.weights, bufs.bias, p, bufs.output, scratch)
            }
            OpKind::Softmax => {
                let p = match spec.params {
                    KernelParams::Softmax(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                hematite_s3::softmax::softmax(bufs.input, p, bufs.output, scratch)
            }
            OpKind::AvgPool => {
                let p = match spec.params {
                    KernelParams::Pool(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                hematite_s3::pool::average_pool_2d(bufs.input, p, bufs.output, scratch)
            }
            OpKind::MaxPool => {
                let p = match spec.params {
                    KernelParams::Pool(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                hematite_s3::pool::max_pool_2d(bufs.input, p, bufs.output, scratch)
            }
            OpKind::Relu => {
                let p = match spec.params {
                    KernelParams::Activation(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                hematite_s3::activations::relu(bufs.input, p, bufs.output, scratch)
            }
            OpKind::Add => {
                let p = match spec.params {
                    KernelParams::Elementwise(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                hematite_s3::elementwise::add(bufs.input, bufs.weights, p, bufs.output, scratch)
            }
            OpKind::Mul => {
                let p = match spec.params {
                    KernelParams::Elementwise(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                hematite_s3::elementwise::mul(bufs.input, bufs.weights, p, bufs.output, scratch)
            }
            OpKind::Sub => {
                let p = match spec.params {
                    KernelParams::Elementwise(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                hematite_s3::elementwise::sub(bufs.input, bufs.weights, p, bufs.output, scratch)
            }
        };
        let out_after = bufs.output.to_vec();
        bufs.output.copy_from_slice(&out);
        match r {
            Ok(()) => Ok(out_after),
            Err(e) => Err(format!("s3 kernel error: {e:?}")),
        }
    }

    /// Run the hematite-ref scalar kernel for the same spec (the canonical
    /// bit-exact reference) and return its output.
    fn run_ref(spec: &KernelSpec, bufs: &mut SpecBufs<'_>, scratch: &mut [u8]) -> Result<Vec<i8>, String> {
        let out = bufs.output.to_vec();
        let r = match spec.op {
            OpKind::Conv2d1x1 | OpKind::Conv2d3x3 => {
                let p = match spec.params {
                    KernelParams::Conv(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                hematite_ref::conv::conv2d(bufs.input, bufs.weights, bufs.bias, p, bufs.output, scratch)
            }
            OpKind::DepthwiseConv2d => {
                let p = match spec.params {
                    KernelParams::Depthwise(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                hematite_ref::depthwise_conv::depthwise_conv2d(bufs.input, bufs.weights, bufs.bias, p, bufs.output, scratch)
            }
            OpKind::FullyConnected => {
                let p = match spec.params {
                    KernelParams::Fc(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                hematite_ref::fully_connected::fully_connected(bufs.input, bufs.weights, bufs.bias, p, bufs.output, scratch)
            }
            OpKind::Softmax => {
                let p = match spec.params {
                    KernelParams::Softmax(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                hematite_ref::softmax::softmax(bufs.input, p, bufs.output, scratch)
            }
            OpKind::AvgPool => {
                let p = match spec.params {
                    KernelParams::Pool(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                hematite_ref::pool::average_pool_2d(bufs.input, p, bufs.output, scratch)
            }
            OpKind::MaxPool => {
                let p = match spec.params {
                    KernelParams::Pool(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                hematite_ref::pool::max_pool_2d(bufs.input, p, bufs.output, scratch)
            }
            OpKind::Relu => {
                let p = match spec.params {
                    KernelParams::Activation(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                hematite_ref::activation::relu(bufs.input, p, bufs.output, scratch)
            }
            OpKind::Add => {
                let p = match spec.params {
                    KernelParams::Elementwise(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                hematite_ref::elementwise::add(bufs.input, bufs.weights, p, bufs.output, scratch)
            }
            OpKind::Mul => {
                let p = match spec.params {
                    KernelParams::Elementwise(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                hematite_ref::elementwise::mul(bufs.input, bufs.weights, p, bufs.output, scratch)
            }
            OpKind::Sub => {
                let p = match spec.params {
                    KernelParams::Elementwise(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                hematite_ref::elementwise::sub(bufs.input, bufs.weights, p, bufs.output, scratch)
            }
        };
        let out_after = bufs.output.to_vec();
        bufs.output.copy_from_slice(&out);
        match r {
            Ok(()) => Ok(out_after),
            Err(e) => Err(format!("ref kernel error: {e:?}")),
        }
    }

    /// Every spec must satisfy its own kernel's shape validation and produce
    /// bit-identical output between the s3 scalar path and the hematite-ref
    /// reference.  This is the host-side proof that the bench entry points use
    /// the real s3 ABI (issues.md: "do NOT invent a new ABI").
    #[test]
    fn every_spec_shapes_are_valid_and_s3_matches_ref_bit_exact() {
        let specs = kernel_specs();
        assert!(!specs.is_empty(), "bench table must not be empty");
        for spec in specs {
            let lay = layout(spec);
            let mut input = vec![0i8; lay.input_len];
            let mut weights = vec![0i8; lay.weights_len];
            let mut bias = vec![0i32; lay.bias_len];
            let mut output = vec![0i8; lay.output_len];
            let mut transformed = vec![0i8; lay.transform_len];
            let mut scratch = vec![0u8; 0];
            {
                let mut bufs = SpecBufs {
                    input: &mut input,
                    weights: &mut weights,
                    bias: &mut bias,
                    output: &mut output,
                    transformed: &mut transformed,
                };
                fill_pattern(&mut bufs);
                let s3_out = run_s3(spec, &mut bufs, &mut scratch)
                    .unwrap_or_else(|e| panic!("{}: s3 kernel rejected the spec: {e}", spec.name));
                let ref_out = run_ref(spec, &mut bufs, &mut scratch)
                    .unwrap_or_else(|e| panic!("{}: ref kernel rejected the spec: {e}", spec.name));
                assert_eq!(
                    s3_out, ref_out,
                    "{}: s3 scalar output must be bit-identical to hematite-ref",
                    spec.name
                );
            }
        }
    }

    #[test]
    fn every_spec_prepared_matches_ref_bit_exact() {
        let specs = kernel_specs();
        for spec in specs {
            let lay = layout(spec);
            let mut input = vec![0i8; lay.input_len];
            let mut weights = vec![0i8; lay.weights_len];
            let mut bias = vec![0i32; lay.bias_len];
            let mut output = vec![0i8; lay.output_len];
            let mut transformed = vec![0i8; lay.transform_len];
            let mut scratch = vec![0u8; 0];
            {
                let mut bufs = SpecBufs {
                    input: &mut input,
                    weights: &mut weights,
                    bias: &mut bias,
                    output: &mut output,
                    transformed: &mut transformed,
                };
                let prepared = prepare_kernel(spec)
                    .unwrap_or_else(|e| panic!("{}: prepare_kernel failed: {e:?}", spec.name));
                fill_pattern(&mut bufs);
                prepared
                    .run(spec, &mut bufs, &mut scratch)
                    .unwrap_or_else(|e| panic!("{}: prepared kernel rejected: {e:?}", spec.name));
                let prepared_out = bufs.output.to_vec();
                fill_pattern(&mut bufs);
                let ref_out = run_ref(spec, &mut bufs, &mut scratch)
                    .unwrap_or_else(|e| panic!("{}: ref kernel rejected: {e}", spec.name));
                assert_eq!(
                    prepared_out, ref_out,
                    "{}: prepared output must be bit-identical to hematite-ref",
                    spec.name
                );
            }
        }
    }

    /// Emulate the bespoke ACCX SIMD reduction in Rust for the SIMD-eligible
    /// conv/fc rows and assert the result is bit-identical to the scalar ref.
    ///
    /// This is the host-side proof that the ACCX kernels (raw `[oc][ic]` /
    /// `[oc][fh][fw][ic]` weights, element-wise 16-wide reduction into a 32-bit
    /// accumulator) plus the bit-exact requantize epilogue make the SIMD output
    /// equal the scalar reference — no device needed. The emulation mirrors the
    /// asm exactly: `acc[oc] = bias[oc] + Σ filter[oc*stride*in_c + k]·input[k]`
    /// (1×1/FC) or the 9-tap 3×3 form, then per-channel requantize + clamp.
    #[test]
    fn accx_emulation_matches_ref_bit_exact() {
        use hematite_int8::{multiply_by_quantized_multiplier, saturating_cast};
        for spec in kernel_specs() {
            let lay = layout(spec);
            let mut input = vec![0i8; lay.input_len];
            let mut weights = vec![0i8; lay.weights_len];
            let mut bias = vec![0i32; lay.bias_len];
            let mut output = vec![0i8; lay.output_len];
            let mut transformed = vec![0i8; lay.transform_len];
            let mut scratch = vec![0u8; 0];
            {
                let mut bufs = SpecBufs {
                    input: &mut input,
                    weights: &mut weights,
                    bias: &mut bias,
                    output: &mut output,
                    transformed: &mut transformed,
                };
                fill_pattern(&mut bufs);
                let ref_out = run_ref(spec, &mut bufs, &mut scratch)
                    .unwrap_or_else(|e| panic!("{}: ref kernel rejected: {e}", spec.name));

                // Only rows the ACCX gate accepts are SIMD-eligible weighted
                // ops; the emulation must mirror the asm for exactly those.
                let (in_c, out_c, taps) = match &spec.params {
                    KernelParams::Conv(p) if p.filter_shape[1] == 1 && p.filter_shape[2] == 1 => {
                        (p.filter_shape[3] as usize, p.filter_shape[0] as usize, 1)
                    }
                    KernelParams::Conv(p) => {
                        (p.filter_shape[3] as usize, p.filter_shape[0] as usize, 9)
                    }
                    KernelParams::Fc(p) => {
                        (p.input_dim as usize, p.output_dim as usize, 1)
                    }
                    _ => continue,
                };
                if !(in_c >= 16 && in_c % 16 == 0 && out_c >= 1) {
                    continue;
                }
                let mut emu = vec![0i8; lay.output_len];
                match &spec.params {
                    KernelParams::Conv(p) => {
                        let in_h = p.input_shape[1] as usize;
                        let in_w = p.input_shape[2] as usize;
                        let out_h = p.output_shape[1] as usize;
                        let out_w = p.output_shape[2] as usize;
                        let stride_h = p.stride_height as usize;
                        let stride_w = p.stride_width as usize;
                        let dil_h = p.dilation_height_factor as usize;
                        let dil_w = p.dilation_width_factor as usize;
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                for oc in 0..out_c {
                                    let mut acc: i64 = 0;
                                    for tap in 0..taps {
                                        let (kh, kw) = if taps == 9 {
                                            (tap / 3, tap % 3)
                                        } else {
                                            (0, 0)
                                        };
                                        let ih = oh * stride_h + kh * dil_h;
                                        let iw = ow * stride_w + kw * dil_w;
                                        if ih >= in_h || iw >= in_w {
                                            continue;
                                        }
                                        for ic in 0..in_c {
                                            let in_idx = (ih * in_w + iw) * in_c + ic;
                                            // RAW [oc][tap][ic] — asm filter[(oc*taps+tap)*in_c+ic]
                                            let w_idx = (oc * taps + tap) * in_c + ic;
                                            acc += bufs.weights[w_idx] as i64
                                                * bufs.input[in_idx] as i64;
                                        }
                                    }
                                    let acc32 = (bufs.bias[oc] as i64 + acc).clamp(
                                        i32::MIN as i64,
                                        i32::MAX as i64,
                                    ) as i32;
                                    let scaled = multiply_by_quantized_multiplier(
                                        acc32,
                                        p.output_multiplier_per_channel[oc],
                                        p.output_shift_per_channel[oc],
                                    );
                                    let clamped = (scaled + p.output_offset).clamp(
                                        p.quantized_activation_min,
                                        p.quantized_activation_max,
                                    );
                                    emu[(oh * out_w + ow) * out_c + oc] = saturating_cast(clamped);
                                }
                            }
                        }
                    }
                    KernelParams::Fc(p) => {
                        for oc in 0..out_c {
                            let mut acc: i64 = 0;
                            for ic in 0..in_c {
                                // RAW [oc][ic] — asm filter[oc*in_c+ic]
                                acc += bufs.weights[oc * in_c + ic] as i64 * bufs.input[ic] as i64;
                            }
                            let acc32 =
                                (bufs.bias[oc] as i64 + acc).clamp(i32::MIN as i64, i32::MAX as i64)
                                    as i32;
                            let scaled = multiply_by_quantized_multiplier(
                                acc32,
                                p.output_multiplier_per_channel[oc],
                                p.output_shift_per_channel[oc],
                            );
                            let clamped = (scaled + p.output_offset).clamp(
                                p.quantized_activation_min,
                                p.quantized_activation_max,
                            );
                            emu[oc] = saturating_cast(clamped);
                        }
                    }
                    _ => unreachable!(),
                }
                assert_eq!(
                    emu, ref_out,
                    "{}: ACCX SIMD emulation must equal scalar ref bit-exact",
                    spec.name
                );
            }
        }
    }

    #[test]
    fn layout_matches_shape_products() {
        let specs = kernel_specs();
        for spec in specs {
            let lay = layout(spec);
            // Every row must have non-zero input and output.
            assert!(lay.input_len > 0, "{}: zero input", spec.name);
            assert!(lay.output_len > 0, "{}: zero output", spec.name);
        }
    }

    #[test]
    fn every_spec_has_a_tier_label() {
        for spec in kernel_specs() {
            let label = spec.tier.label();
            assert!(label == "SRAM" || label == "PSRAM", "{}: bad tier", spec.name);
        }
    }

    /// The arena carve the device firmware uses must produce valid, aligned,
    /// disjoint buffers for every spec and pass both kernels' shape checks.
    #[test]
    fn carve_into_works_for_every_spec() {
        // A 4 MB arena mirrors the firmware's PSRAM arena.
        let mut arena = vec![0u8; 4 * 1024 * 1024];
        for spec in kernel_specs() {
            let lay = layout(spec);
            let mut bufs = carve_into(&mut arena, &lay).expect("arena must hold every spec");
            fill_pattern(&mut bufs);
            let mut scratch = [0u8; 0];
            assert!(
                run_s3(spec, &mut bufs, &mut scratch).is_ok(),
                "{}: carved buffers rejected by s3 kernel",
                spec.name
            );
            // Alignment: input/weights/output/bias 16-byte.
            assert_eq!(bufs.input.as_ptr() as usize % 16, 0);
            assert_eq!(bufs.weights.as_ptr() as usize % 16, 0);
            assert_eq!(bufs.bias.as_ptr() as usize % 16, 0);
            assert_eq!(bufs.output.as_ptr() as usize % 16, 0);
        }
    }

    #[test]
    fn carve_into_rejects_oversized_specs() {
        let mut small = vec![0u8; 64];
        let lay = SpecLayout {
            input_len: 100,
            weights_len: 0,
            bias_len: 0,
            output_len: 0,
            transform_len: 0,
        };
        assert!(matches!(
            carve_into(&mut small, &lay),
            Err(KernelError::ScratchTooSmall)
        ));
    }

    #[test]
    fn carve_into_offsets_are_disjoint() {
        let mut arena = vec![0u8; 4096];
        let lay = SpecLayout {
            input_len: 7,
            weights_len: 11,
            bias_len: 3,
            output_len: 5,
            transform_len: 2,
        };
        let bufs = carve_into(&mut arena, &lay).expect("fits");
        let (io, wo, oo) = (
            bufs.input.as_ptr() as usize,
            bufs.weights.as_ptr() as usize,
            bufs.output.as_ptr() as usize,
        );
        assert!(io + 7 <= wo, "input must not overlap weights");
        assert!(wo + 11 <= oo, "weights must not overlap output");
        // Bias region lies between weights and output.
        let bo = bufs.bias.as_ptr() as usize;
        assert!(wo + 11 <= bo && bo + 3 * 4 <= oo, "bias must sit between weights and output");
    }
}
