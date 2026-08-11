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
    ActivationEpilogueParams, ActivationParams, ComposedActivation, Conv2DParams,
    DepthwiseConv2DParams, ElementwiseChainParams, ElementwiseChainStep, ElementwiseKind,
    ElementwiseParams, FoldedPoolParams, FusedConvParams, FullyConnectedParams,
    FusedActivation, Padding, PoolInputFold, PoolKind, PoolParams, ResidualAddParams, SoftmaxParams,
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
const MULT_8: [i32; 8] = mults::<8>();
const SHIFT_8: [i32; 8] = shifts::<8>();
const MULT_12: [i32; 12] = mults::<12>();
const SHIFT_12: [i32; 12] = shifts::<12>();
const MULT_16: [i32; 16] = mults::<16>();
const SHIFT_16: [i32; 16] = shifts::<16>();
const MULT_32: [i32; 32] = mults::<32>();
const SHIFT_32: [i32; 32] = shifts::<32>();
const MULT_64: [i32; 64] = mults::<64>();
const SHIFT_64: [i32; 64] = shifts::<64>();
const MULT_128: [i32; 128] = mults::<128>();
const SHIFT_128: [i32; 128] = shifts::<128>();
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
    Relu6,
    HardSwish,
    Add,
    Mul,
    Sub,
    /// Composed CONV_2D + residual-ADD + activation epilogue (T2.2) — the
    /// anchor conv's own shape; the epilogue reads the residual const tensor.
    FusedConv2d,
    /// Composed elementwise chain (T2.3) — N steps, each step's own
    /// requantize preserved, register-held between steps on the SIMD path.
    FusedElementwiseChain,
    /// Composed pool + MUL/SUB input fold + activation epilogue (T2.4) — the
    /// anchor pool's own shape; the fold operand is a const tensor embedded
    /// in the params.
    FusedPoolFold,
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
    FusedConv(&'static FusedConvParams<'static>),
    FusedChain(&'static ElementwiseChainParams<'static>),
    FusedPool(&'static FoldedPoolParams<'static>),
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
    // TFLM diff_min = -CalculateInputRadius(5, left_shift)
    // (softmax_common.cc @ 18b9e6f2a8c5a9518e588f59c2ba16ef7ef9d551);
    // radius = floor(31 * 2^26 / 2^(input_left_shift + 1)) — the +1 matches
    // TFLM's stored shift (26+s) vs our 25+s convention → -(31 << 26 >> 23) = -248.
    diff_min: -248,
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

/// 3×3 conv with SAME padding (pad=1 on a 3x3 filter), stride 1, channels %16 —
/// exercises the Phase A spatial zero-pad SIMD path in `conv3x3_accx_dispatch`.
const SIMD_CONV3X3_SAME_PARAMS: Conv2DParams<'static> = Conv2DParams {
    input_shape: [1, 16, 16, 32],
    filter_shape: [32, 3, 3, 32],
    output_shape: [1, 16, 16, 32],
    padding: Padding::Same,
    stride_width: 1,
    stride_height: 1,
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

/// Same SAME-pad 3×3 conv as `SIMD_CONV3X3_SAME_PARAMS` but with a non-zero
/// `input_offset` — exercises the Phase C weight-sum fold
/// (`Σ(in+off)·w = Σin·w + off·Σw`).
const SIMD_CONV3X3_SAME_OFF_PARAMS: Conv2DParams<'static> = Conv2DParams {
    input_shape: [1, 16, 16, 32],
    filter_shape: [32, 3, 3, 32],
    output_shape: [1, 16, 16, 32],
    padding: Padding::Same,
    stride_width: 1,
    stride_height: 1,
    dilation_width_factor: 1,
    dilation_height_factor: 1,
    input_offset: 3,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_32,
    output_shift_per_channel: &SHIFT_32,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

/// Stride-2 SAME depthwise (MobileNetV2 downsampling block): the bespoke QACC
/// depthwise dispatch zero-pads the input spatially and strides in the pixel
/// loop (Phase B).
const SIMD_DEPTHWISE_S2_SAME_PARAMS: DepthwiseConv2DParams<'static> = DepthwiseConv2DParams {
    input_shape: [1, 12, 12, 16],
    filter_shape: [1, 3, 3, 16],
    output_shape: [1, 6, 6, 16],
    padding: Padding::Same,
    stride_width: 2,
    stride_height: 2,
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

/// Same stride-2 SAME depthwise as `SIMD_DEPTHWISE_S2_SAME_PARAMS` but with a
/// negative non-zero `input_offset` (Phase C fold).
const SIMD_DEPTHWISE_S2_SAME_OFF_PARAMS: DepthwiseConv2DParams<'static> = DepthwiseConv2DParams {
    input_shape: [1, 12, 12, 16],
    filter_shape: [1, 3, 3, 16],
    output_shape: [1, 6, 6, 16],
    padding: Padding::Same,
    stride_width: 2,
    stride_height: 2,
    dilation_width_factor: 1,
    dilation_height_factor: 1,
    depth_multiplier: 1,
    input_offset: -3,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_16,
    output_shift_per_channel: &SHIFT_16,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

/// Stride-1 SAME depthwise with a NON-%16 channel count (12 channels → padded
/// to 16 in scratch). Phase F: the dispatch zero-pads the input and filter
/// channel dimensions, so any `in_c >= 1` is SIMD-eligible.
const SIMD_DEPTHWISE_NON16_PARAMS: DepthwiseConv2DParams<'static> = DepthwiseConv2DParams {
    input_shape: [1, 12, 12, 12],
    filter_shape: [1, 3, 3, 12],
    output_shape: [1, 12, 12, 12],
    padding: Padding::Same,
    stride_width: 1,
    stride_height: 1,
    dilation_width_factor: 1,
    dilation_height_factor: 1,
    depth_multiplier: 1,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_12,
    output_shift_per_channel: &SHIFT_12,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

/// depth_multiplier = 2 depthwise (T3.5): in_c 8 → out_c 16, 3×3 SAME. The
/// dm>1 SIMD dispatch stages a replicated input (each input channel fanned
/// out to dm output channels) and per-channel requantizes — bit-exact vs ref.
const SIMD_DEPTHWISE_DM2_PARAMS: DepthwiseConv2DParams<'static> = DepthwiseConv2DParams {
    input_shape: [1, 12, 12, 8],
    filter_shape: [1, 3, 3, 16],
    output_shape: [1, 12, 12, 16],
    padding: Padding::Same,
    stride_width: 1,
    stride_height: 1,
    dilation_width_factor: 1,
    dilation_height_factor: 1,
    depth_multiplier: 2,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_16,
    output_shift_per_channel: &SHIFT_16,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

/// depth_multiplier = 4 depthwise (T3.5): in_c 8 → out_c 32.
const SIMD_DEPTHWISE_DM4_PARAMS: DepthwiseConv2DParams<'static> = DepthwiseConv2DParams {
    input_shape: [1, 12, 12, 8],
    filter_shape: [1, 3, 3, 32],
    output_shape: [1, 12, 12, 32],
    padding: Padding::Same,
    stride_width: 1,
    stride_height: 1,
    dilation_width_factor: 1,
    dilation_height_factor: 1,
    depth_multiplier: 4,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_32,
    output_shift_per_channel: &SHIFT_32,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

/// depth_multiplier = 8 depthwise (T3.5): in_c 8 → out_c 64 — the KWS
/// keyword-spotting fan-out shape family that drives the 12.3× model gap
/// (kws 12,983,503 → 1,059,889 cyc vs ESP-NN; ESP-NN-relative bar:
/// < 1,059,889 cyc / 4 ms, user-verified 2026-08-10 — Scope table).
const SIMD_DEPTHWISE_DM8_PARAMS: DepthwiseConv2DParams<'static> = DepthwiseConv2DParams {
    input_shape: [1, 12, 12, 8],
    filter_shape: [1, 3, 3, 64],
    output_shape: [1, 12, 12, 64],
    padding: Padding::Same,
    stride_width: 1,
    stride_height: 1,
    dilation_width_factor: 1,
    dilation_height_factor: 1,
    depth_multiplier: 8,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_64,
    output_shift_per_channel: &SHIFT_64,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

/// T3.5b — the REAL kws depthwise layer (tflite-verified): input
/// [1,49,40,1], filter [1,10,8,8] (80 taps), output [1,25,20,8] (Relu),
/// stride 2 SAME, depth_multiplier 8, input_offset +128 (the depthwise input
/// is the first conv's output with zero point -128 — the Phase-C fold uses
/// fill -128 which fits in i8). The tap-parameterized anytap SIMD kernel
/// dispatches this shape in 3 chunked QACC passes (32+32+16 taps; the QACC
/// 20-bit-lane-safe bound).
const SIMD_DEPTHWISE_KWS_10X8_PARAMS: DepthwiseConv2DParams<'static> = DepthwiseConv2DParams {
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
    output_multiplier_per_channel: &MULT_8,
    output_shift_per_channel: &SHIFT_8,
    quantized_activation_min: 0,
    quantized_activation_max: 127,
};

/// T3.5b — arbitrary 5×5 filter, dm=1 (in_c 8 → out_c 8), stride 1 SAME.
const SIMD_DEPTHWISE_5X5_DM1_PARAMS: DepthwiseConv2DParams<'static> = DepthwiseConv2DParams {
    input_shape: [1, 12, 12, 8],
    filter_shape: [1, 5, 5, 8],
    output_shape: [1, 12, 12, 8],
    padding: Padding::Same,
    stride_width: 1,
    stride_height: 1,
    dilation_width_factor: 1,
    dilation_height_factor: 1,
    depth_multiplier: 1,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_8,
    output_shift_per_channel: &SHIFT_8,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

/// T3.5b — arbitrary 5×5 filter, dm=8 (in_c 1 → out_c 8), stride 1 SAME.
const SIMD_DEPTHWISE_5X5_DM8_PARAMS: DepthwiseConv2DParams<'static> = DepthwiseConv2DParams {
    input_shape: [1, 12, 12, 1],
    filter_shape: [1, 5, 5, 8],
    output_shape: [1, 12, 12, 8],
    padding: Padding::Same,
    stride_width: 1,
    stride_height: 1,
    dilation_width_factor: 1,
    dilation_height_factor: 1,
    depth_multiplier: 8,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_8,
    output_shift_per_channel: &SHIFT_8,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

/// T3.5b — arbitrary 7×7 filter, dm=1 (in_c 8 → out_c 8), stride 2 SAME.
const SIMD_DEPTHWISE_7X7_DM1_PARAMS: DepthwiseConv2DParams<'static> = DepthwiseConv2DParams {
    input_shape: [1, 14, 14, 8],
    filter_shape: [1, 7, 7, 8],
    output_shape: [1, 7, 7, 8],
    padding: Padding::Same,
    stride_width: 2,
    stride_height: 2,
    dilation_width_factor: 1,
    dilation_height_factor: 1,
    depth_multiplier: 1,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_8,
    output_shift_per_channel: &SHIFT_8,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

/// T3.5b — arbitrary 7×7 filter, dm=8 (in_c 1 → out_c 8), stride 2 SAME.
const SIMD_DEPTHWISE_7X7_DM8_PARAMS: DepthwiseConv2DParams<'static> = DepthwiseConv2DParams {
    input_shape: [1, 14, 14, 1],
    filter_shape: [1, 7, 7, 8],
    output_shape: [1, 7, 7, 8],
    padding: Padding::Same,
    stride_width: 2,
    stride_height: 2,
    dilation_width_factor: 1,
    dilation_height_factor: 1,
    depth_multiplier: 8,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_8,
    output_shift_per_channel: &SHIFT_8,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

/// T3.5b — non-square 3×5 filter, dm=2 (in_c 8 → out_c 16), stride 1 SAME.
const SIMD_DEPTHWISE_3X5_DM2_PARAMS: DepthwiseConv2DParams<'static> = DepthwiseConv2DParams {
    input_shape: [1, 12, 12, 8],
    filter_shape: [1, 3, 5, 16],
    output_shape: [1, 12, 12, 16],
    padding: Padding::Same,
    stride_width: 1,
    stride_height: 1,
    dilation_width_factor: 1,
    dilation_height_factor: 1,
    depth_multiplier: 2,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_16,
    output_shift_per_channel: &SHIFT_16,
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

/// Same FC as `SIMD_FC_256X64_PARAMS` but with a non-zero `input_offset`
/// (Phase C fold).
const SIMD_FC_256X64_OFF_PARAMS: FullyConnectedParams<'static> = FullyConnectedParams {
    input_dim: 256,
    output_dim: 64,
    input_offset: 5,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_64,
    output_shift_per_channel: &SHIFT_64,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

/// FC 1→1, input_offset 0 — the sine model's single-FC shape (T3.6). The
/// input_dim 1 is zero-padded to 16 in scratch, then the TIE728 11cn path
/// runs (pad-in-scratch widening).
const SIMD_FC_1X1_PARAMS: FullyConnectedParams<'static> = FullyConnectedParams {
    input_dim: 1,
    output_dim: 1,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &[1 << 30],
    output_shift_per_channel: &[0],
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

/// FC 1→16, input_offset 128 — hello_world's first dense layer (T3.6). The
/// gated-out shape (input_dim 1 < 16) now dispatches SIMD via pad-in-scratch.
const SIMD_FC_1X16_PARAMS: FullyConnectedParams<'static> = FullyConnectedParams {
    input_dim: 1,
    output_dim: 16,
    input_offset: 128,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_16,
    output_shift_per_channel: &SHIFT_16,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

/// FC 16→1, input_offset 128 — hello_world's final dense layer (T3.6).
/// input_dim 16 (%16) needs no pad; output_dim 1 exercises the small-out
/// path.
const SIMD_FC_16X1_PARAMS: FullyConnectedParams<'static> = FullyConnectedParams {
    input_dim: 16,
    output_dim: 1,
    input_offset: 128,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &[1 << 30],
    output_shift_per_channel: &[0],
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

/// FC 8→128, input_offset 128 — anomaly_detect's gated-out 6th dense layer
/// (T3.6). input_dim 8 is zero-padded to 16 in scratch.
const SIMD_FC_8X128_PARAMS: FullyConnectedParams<'static> = FullyConnectedParams {
    input_dim: 8,
    output_dim: 128,
    input_offset: 128,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_128,
    output_shift_per_channel: &SHIFT_128,
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

// ── T3.1 generic-pool rows — the widened matrix ──────────────────────────────
//
// filter {2×2, 3×3, 5×5, global 7×7} × stride {1, 2} × pad {0, 1, SAME} ×
// clamp {full-range, relu}. The device SIMD path (`simd_eligible_pool`)
// engages for the no-padding / no-partial-window shapes (pad_total ≤ 0:
// VALID rows and 2×2/stride-2 SAME); padded rows run the scalar fallback on
// device (bit-exact vs ref — the pool backend delivers no scratch for
// spatial padding staging) and are model-verified on the host. The avg
// fixed-point-vs-ref divergence is documented in
// `local-notes/evidence/composed-kernels/t31-pool.md`.

/// 3×3 stride-1 VALID (pad_total 0) — the generic hwc1 SIMD path (avg).
const POOL_3X3_S1_VALID_PARAMS: PoolParams = PoolParams {
    input_shape: [1, 8, 8, 16],
    output_shape: [1, 6, 6, 16],
    filter_width: 3,
    filter_height: 3,
    stride_width: 1,
    stride_height: 1,
    padding: Padding::Valid,
    activation: FusedActivation::None,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

/// Same as `POOL_3X3_S1_VALID_PARAMS` with a relu-range clamp (0..127) —
/// the generic driver's Rust clamp post-pass.
const POOL_3X3_S1_VALID_RELU_PARAMS: PoolParams = PoolParams {
    input_shape: [1, 8, 8, 16],
    output_shape: [1, 6, 6, 16],
    filter_width: 3,
    filter_height: 3,
    stride_width: 1,
    stride_height: 1,
    padding: Padding::Valid,
    activation: FusedActivation::None,
    quantized_activation_min: 0,
    quantized_activation_max: 127,
};

/// 3×3 stride-1 SAME (pad 1) — model-verified on host; scalar on device.
const POOL_3X3_S1_SAME_PARAMS: PoolParams = PoolParams {
    input_shape: [1, 8, 8, 16],
    output_shape: [1, 8, 8, 16],
    filter_width: 3,
    filter_height: 3,
    stride_width: 1,
    stride_height: 1,
    padding: Padding::Same,
    activation: FusedActivation::None,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

/// 3×3 stride-2 SAME on 12×12 (pad_total 1 — asymmetric SAME, partial
/// windows) — model-verified on host; scalar on device.
const POOL_3X3_S2_SAME_PARAMS: PoolParams = PoolParams {
    input_shape: [1, 12, 12, 16],
    output_shape: [1, 6, 6, 16],
    filter_width: 3,
    filter_height: 3,
    stride_width: 2,
    stride_height: 2,
    padding: Padding::Same,
    activation: FusedActivation::None,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

/// 5×5 stride-1 SAME on 12×12 (pad 2) — model-verified on host; scalar on
/// device.
const POOL_5X5_S1_SAME_PARAMS: PoolParams = PoolParams {
    input_shape: [1, 12, 12, 16],
    output_shape: [1, 12, 12, 16],
    filter_width: 5,
    filter_height: 5,
    stride_width: 1,
    stride_height: 1,
    padding: Padding::Same,
    activation: FusedActivation::None,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

/// 5×5 stride-2 SAME on 14×14 (pad_total 3, pad 1) — model-verified on
/// host; scalar on device.
const POOL_5X5_S2_SAME_PARAMS: PoolParams = PoolParams {
    input_shape: [1, 14, 14, 16],
    output_shape: [1, 7, 7, 16],
    filter_width: 5,
    filter_height: 5,
    stride_width: 2,
    stride_height: 2,
    padding: Padding::Same,
    activation: FusedActivation::None,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

/// Global 7×7 avg/max pool (pad_total 0) — the generic hwc1 path on a
/// 1×1 output.
const POOL_7X7_GLOBAL_PARAMS: PoolParams = PoolParams {
    input_shape: [1, 7, 7, 16],
    output_shape: [1, 1, 1, 16],
    filter_width: 7,
    filter_height: 7,
    stride_width: 7,
    stride_height: 7,
    padding: Padding::Valid,
    activation: FusedActivation::None,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

/// 3×3 stride-1 VALID with 24 channels — the model's C%16 scalar tail
/// (host-verified); scalar on device (the device gate requires C % 16).
const POOL_3X3_S1_VALID_C24_PARAMS: PoolParams = PoolParams {
    input_shape: [1, 8, 8, 24],
    output_shape: [1, 6, 6, 24],
    filter_width: 3,
    filter_height: 3,
    stride_width: 1,
    stride_height: 1,
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

// ── T3.2 widened elementwise + activation rows ───────────────────────────────
//
// Non-identity offset/multiplier pairs exercise the widened per-lane SIMD
// model (arbitrary quant-affine chains); relu6 / hard_swish exercise the new
// activation lane models. All n = 256 (multiple of 16 — the lane-model vector
// main loop engages on device with aligned arena buffers).

/// Elementwise ADD over 256 elements with the two-stage TFLM Add rounding
/// shape (non-zero zero points, left_shift 20, per-input multipliers, output
/// requantize) — NOT identity-eligible, engages the T3.2 widened lane model.
const SIMD_ADD_256_NONIDENTITY_PARAMS: ElementwiseParams = ElementwiseParams {
    num_elements: 256,
    input1_offset: -5,
    input2_offset: 3,
    output_offset: 1,
    output_multiplier: 1_342_177_280,
    output_shift: -18,
    left_shift: 20,
    input1_multiplier: 1 << 30,
    input1_shift: 0,
    input2_multiplier: 1_288_490_189,
    input2_shift: -1,
    quantized_activation_min: -32,
    quantized_activation_max: 96,
};

/// Elementwise SUB over 256 elements — same non-identity shape as the ADD row.
const SIMD_SUB_256_NONIDENTITY_PARAMS: ElementwiseParams = ElementwiseParams {
    num_elements: 256,
    input1_offset: -5,
    input2_offset: 3,
    output_offset: 1,
    output_multiplier: 1_342_177_280,
    output_shift: -18,
    left_shift: 20,
    input1_multiplier: 1 << 30,
    input1_shift: 0,
    input2_multiplier: 1_288_490_189,
    input2_shift: -1,
    quantized_activation_min: -32,
    quantized_activation_max: 96,
};

/// Elementwise MUL over 256 elements with non-zero offsets and a real product
/// requantize (scale change) — engages the T3.2 widened lane model.
const SIMD_MUL_256_NONIDENTITY_PARAMS: ElementwiseParams = ElementwiseParams {
    num_elements: 256,
    input1_offset: 2,
    input2_offset: -3,
    output_offset: -7,
    output_multiplier: 1_717_986_918,
    output_shift: -3,
    left_shift: 0,
    input1_multiplier: 0,
    input1_shift: 0,
    input2_multiplier: 0,
    input2_shift: 0,
    quantized_activation_min: -16,
    quantized_activation_max: 111,
};

/// ReLU6 over 256 elements — the widened lane model clamps to
/// `quantized_activation_max` forwarded as `quantized_six` (= 24 at
/// scale 0.25). No requantize (output scale 1.0).
const SIMD_RELU6_256_PARAMS: ActivationParams<'static> = ActivationParams {
    input_offset: -1,
    output_offset: 2,
    output_multiplier: 0,
    output_shift: 0,
    quantized_activation_min: 0,
    quantized_activation_max: 24,
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

/// HardSwish over 256 elements — the DOWNGRADED integer rational formula
/// (x·ReLU6(x+3)/6, ±3 round-half correction), pinned by the goldens. The
/// widened lane model reproduces it bit-exact (per-lane /6 scalar tail).
const SIMD_HARD_SWISH_256_PARAMS: ActivationParams<'static> = ActivationParams {
    input_offset: -3,
    output_offset: 1,
    output_multiplier: 0,
    output_shift: 0,
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

// ── Composed fused-conv rows (T2.2) ──────────────────────────────────────────
//
// The `fused_conv2d` composed kernel: anchor conv + residual-ADD + activation
// epilogue in ONE SIMD pass. The rows exercise the two conv-family SIMD paths
// reachable from a `Conv2DParams` anchor (1×1 and general/3×3). Quant pairs
// are scale-derived exactly as the T1.2 emitter derives them (StepRequantize):
// `input1 = QuantizeMultiplier(s1/twice_max)`, `input2 =
// QuantizeMultiplier(s2/twice_max)`, `output =
// QuantizeMultiplier(twice_max/(2^20·s_out))` with `left_shift = 20`; the
// residual scale deliberately differs from the conv output scale so the
// per-input roundings are non-identity.

/// Deterministic const residual pattern (LCG, full int8 range) — the residual
/// is a model constant tensor, so it is embedded in the params const.
const fn residual_pattern<const N: usize>(seed: u64) -> [i8; N] {
    let mut out = [0i8; N];
    let mut x = seed;
    let mut i = 0;
    while i < N {
        x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        out[i] = (x >> 33) as i8;
        i += 1;
    }
    out
}

const FUSED_RESIDUAL_256: [i8; 256] = residual_pattern::<256>(0xF0E1_D2C3);
const FUSED_RESIDUAL_288: [i8; 288] = residual_pattern::<288>(0x0BAD_F00D);

/// Anchor conv for the 1×1 fused row — full-range activation (the residual
/// block's conv output feeds the ADD, not a fused activation).
const SIMD_FUSED_CONV1X1_CONV: Conv2DParams<'static> = Conv2DParams {
    input_shape: [1, 4, 4, 16],
    filter_shape: [16, 1, 1, 16],
    output_shape: [1, 4, 4, 16],
    padding: Padding::Same,
    stride_width: 1,
    stride_height: 1,
    dilation_width_factor: 1,
    dilation_height_factor: 1,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 5,
    output_multiplier_per_channel: &MULT_16,
    output_shift_per_channel: &SHIFT_16,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

/// Anchor conv for the 3×3 fused row — non-%16 channels (8) exercise the
/// channel-padded staging path on device.
const SIMD_FUSED_CONV3X3_CONV: Conv2DParams<'static> = Conv2DParams {
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
    output_offset: -2,
    output_multiplier_per_channel: &MULT_8,
    output_shift_per_channel: &SHIFT_8,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

/// Fused 1×1 anchor: in [1,4,4,16] → out [1,4,4,16], residual + ReLU. The
/// conv1x1 ACCX SIMD path (in_c 16, %16) — the MobileNetV2 pointwise
/// residual-block prototype. Conv scale 0.5 / residual 0.3 / add-out 0.4.
const FUSED_CONV1X1_RESIDUAL_RELU_PARAMS: FusedConvParams<'static> = FusedConvParams {
    conv: SIMD_FUSED_CONV1X1_CONV,
    output_scale: 0.5,
    output_zero_point: 5,
    output_multiplier_per_channel: &MULT_16,
    output_shift_per_channel: &SHIFT_16,
    residual: Some(ResidualAddParams {
        residual_data: &FUSED_RESIDUAL_256,
        residual_scale: 0.3,
        residual_zero_point: -3,
        output_scale: 0.4,
        output_zero_point: 1,
        input1_multiplier: 1 << 30,
        input1_shift: 0,
        input2_multiplier: 1_288_490_189,
        input2_shift: -1,
        left_shift: 20,
        output_multiplier: 1_342_177_280,
        output_shift: -18,
    }),
    activation: ActivationEpilogueParams {
        kind: ComposedActivation::Relu,
        input_offset: -1,
        output_offset: 2,
        output_multiplier: 1_342_177_280,
        output_shift: 1,
        quantized_activation_min: 0,
        quantized_activation_max: 127,
    },
};

/// Fused 3×3 anchor: in [1,6,6,8] → out [1,6,6,8] (non-%16 channels →
/// channel-padded staging on device), residual + HardSwish. Conv scale 0.02 /
/// residual 0.05 / add-out 0.03.
const FUSED_CONV3X3_RESIDUAL_HSWISH_PARAMS: FusedConvParams<'static> = FusedConvParams {
    conv: SIMD_FUSED_CONV3X3_CONV,
    output_scale: 0.02,
    output_zero_point: -2,
    output_multiplier_per_channel: &MULT_8,
    output_shift_per_channel: &SHIFT_8,
    residual: Some(ResidualAddParams {
        residual_data: &FUSED_RESIDUAL_288,
        residual_scale: 0.05,
        residual_zero_point: 7,
        output_scale: 0.03,
        output_zero_point: 3,
        input1_multiplier: 1_717_986_918,
        input1_shift: -2,
        input2_multiplier: 1 << 30,
        input2_shift: 0,
        left_shift: 20,
        output_multiplier: 1_789_569_707,
        output_shift: -18,
    }),
    activation: ActivationEpilogueParams {
        kind: ComposedActivation::HardSwish,
        input_offset: -3,
        output_offset: -1,
        output_multiplier: 1_431_655_765,
        output_shift: 0,
        quantized_activation_min: 0,
        quantized_activation_max: 127,
    },
};

// ── Composed fused elementwise-chain row (T2.3) ──────────────────────────────
//
// The `fused_elementwise_chain` composed kernel: N elementwise steps, each
// step's own requantize preserved (steps are NEVER collapsed). Quant pairs
// are scale-derived exactly as the T1.2 emitter derives them (StepRequantize):
// add/sub use the two-stage TFLM Add rounding (left_shift 20, input_i =
// QuantizeMultiplier(s_i/twice_max), output =
// QuantizeMultiplier(twice_max/(2^20·s_out))); mul uses the single product
// requantize QuantizeMultiplier(s_in1·s_in2/s_out); activations use their
// output ratio. Every scale here is non-identity, so the chain's per-step
// roundings are exercised. On device TODAY this row runs the decomposition
// (the hard_swish step has no SIMD yet, T3.2 — chains SIMD-engage only when
// EVERY step is identity-eligible, see `fused::chain_simd_eligible`); the
// chain-runtime < sum-of-per-op-runtimes measurement is T6.x.

/// Deterministic const chain operand pattern (LCG, full int8 range) — model
/// constant tensors, embedded in the params const (same as the fused-conv
/// residual).
const fn chain_operand<const N: usize>(seed: u64) -> [i8; N] {
    let mut out = [0i8; N];
    let mut x = seed;
    let mut i = 0;
    while i < N {
        x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        out[i] = (x >> 33) as i8;
        i += 1;
    }
    out
}

const CHAIN_OPERAND_A: [i8; 256] = chain_operand::<256>(0x0BAD_F00D);
const CHAIN_OPERAND_B: [i8; 256] = chain_operand::<256>(0xDEAD_BEEF);

/// The plan's canonical 4-op chain: add + relu + mul + hard_swish over 256
/// elements, every step with NON-identity scales/offsets (add: input1
/// 0.5/input2 0.3/output 0.4, zps 5/-3/1; relu: 0.4→0.2, zp 1→2; mul:
/// 0.2·0.05→0.1, zps -2/0→-3; hard_swish: 0.1→0.03, zp 3→1).
const FUSED_CHAIN_4OP_STEPS: [ElementwiseChainStep<'static>; 4] = [
    ElementwiseChainStep {
        kind: ElementwiseKind::Add,
        operand: Some(&CHAIN_OPERAND_A),
        input1_offset: -5,
        input2_offset: 3,
        output_offset: 1,
        output_multiplier: 1_342_177_280,
        output_shift: -18,
        left_shift: 20,
        input1_multiplier: 1 << 30,
        input1_shift: 0,
        input2_multiplier: 1_288_490_189,
        input2_shift: -1,
        quantized_activation_min: i8::MIN as i32,
        quantized_activation_max: i8::MAX as i32,
    },
    ElementwiseChainStep {
        kind: ElementwiseKind::Relu,
        operand: None,
        input1_offset: -1,
        input2_offset: 0,
        output_offset: 2,
        output_multiplier: 1_073_741_824,
        output_shift: 2,
        left_shift: 0,
        input1_multiplier: 0,
        input1_shift: 0,
        input2_multiplier: 0,
        input2_shift: 0,
        quantized_activation_min: 0,
        quantized_activation_max: 127,
    },
    ElementwiseChainStep {
        kind: ElementwiseKind::Mul,
        operand: Some(&CHAIN_OPERAND_B),
        input1_offset: 2,
        input2_offset: 0,
        output_offset: -3,
        output_multiplier: 1_717_986_918,
        output_shift: -3,
        left_shift: 0,
        input1_multiplier: 0,
        input1_shift: 0,
        input2_multiplier: 0,
        input2_shift: 0,
        quantized_activation_min: i8::MIN as i32,
        quantized_activation_max: i8::MAX as i32,
    },
    ElementwiseChainStep {
        kind: ElementwiseKind::HardSwish,
        operand: None,
        input1_offset: -3,
        input2_offset: 0,
        output_offset: 1,
        output_multiplier: 1_789_569_707,
        output_shift: 2,
        left_shift: 0,
        input1_multiplier: 0,
        input1_shift: 0,
        input2_multiplier: 0,
        input2_shift: 0,
        quantized_activation_min: 0,
        quantized_activation_max: 127,
    },
];

const FUSED_CHAIN_4OP_PARAMS: ElementwiseChainParams<'static> = ElementwiseChainParams {
    num_elements: 256,
    steps: &FUSED_CHAIN_4OP_STEPS,
};

// ── Composed pool-with-fold row (T2.4) ───────────────────────────────────────
//
// The `fused_pool_with_fold` composed kernel: anchor pool + absorbed MUL/SUB
// input fold materialized into scratch + activation epilogue. The row uses an
// IDENTITY MUL fold — zero offsets, full-range clamp, `(1<<30, 0)` output
// pair — which is in the provably-exact subset (`fused::fold_simd_exact`), so
// on device the fold SIMD-engages via the elementwise gates and the pool SIMD
// reads the staged scratch directly (`fused::fused_pool_fold_simd_eligible`).

/// Anchor pool for the composed pool-fold row — 2×2/stride-2/SAME, channels
/// % 16, full-range clamp: exactly `pool::simd_eligible_pool`'s contract.
const FUSED_POOL_ANCHOR: PoolParams = PoolParams {
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
};

/// Deterministic const fold operand (LCG, full int8 range) — a model constant
/// tensor, embedded in the params const (same as the fused-conv residual).
const FOLD_OPERAND_256: [i8; 256] = chain_operand::<256>(0xF00D_CAFE);

/// Identity MUL fold (the `simd_eligible_mul` gate's contract) — in the
/// provably-exact T2.4 subset, so this group SIMD-engages on device.
const FUSED_POOL_MUL_FOLD: PoolInputFold<'static> = PoolInputFold {
    builtin: 18,
    operand_data: &FOLD_OPERAND_256,
    operand_zero_point: 0,
    input_zero_point: 0,
    output_zero_point: 0,
    folded_scale: 1.0,
    left_shift: 0,
    output_multiplier: 1 << 30,
    output_shift: 0,
    input1_multiplier: 0,
    input1_shift: 0,
    input2_multiplier: 0,
    input2_shift: 0,
    num_elements: 256,
};

const FUSED_POOL_FOLD_PARAMS: FoldedPoolParams<'static> = FoldedPoolParams {
    pool: FUSED_POOL_ANCHOR,
    pool_kind: PoolKind::Average,
    fold: Some(FUSED_POOL_MUL_FOLD),
    activation: ActivationEpilogueParams {
        kind: ComposedActivation::Relu,
        input_offset: -5,
        output_offset: 2,
        output_multiplier: 1_342_177_280,
        output_shift: 1,
        quantized_activation_min: 0,
        quantized_activation_max: 127,
    },
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
            name: "conv3x3_s8 16x16,32x3x3x32 SAME (SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::Conv2d3x3,
            params: KernelParams::Conv(&SIMD_CONV3X3_SAME_PARAMS),
            reference: None,
            note: "Phase A spatial zero-pad SIMD row: SAME padding (pad=1), channels %16.",
        },
        KernelSpec {
            name: "conv3x3_s8 16x16,32x3x3x32 SAME off3 (SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::Conv2d3x3,
            params: KernelParams::Conv(&SIMD_CONV3X3_SAME_OFF_PARAMS),
            reference: None,
            note: "Phase C non-zero input_offset fold row (input_offset=3).",
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
            name: "fc_s8 256row,64out off5 (SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::FullyConnected,
            params: KernelParams::Fc(&SIMD_FC_256X64_OFF_PARAMS),
            reference: None,
            note: "Phase C non-zero input_offset fold row (input_offset=5).",
        },
        KernelSpec {
            name: "fc_s8 1row,1out (sine, T3.6)",
            tier: MemoryTier::Sram,
            op: OpKind::FullyConnected,
            params: KernelParams::Fc(&SIMD_FC_1X1_PARAMS),
            reference: Some(CompetitorBaseline {
                name: "ESP-NN optimized (sine)",
                cycles: None,
                target_speedup_x100: None,
                source: "plan composed-kernels Scope table (user-verified 2026-08-10): sine 618 → 190 cyc (3.3x). Measured Hematite row lands in T6.x (on-device).",
            }),
            note: "T3.6 sine-family row: input_dim 1 zero-padded to 16 in scratch (pad-in-scratch widening).",
        },
        KernelSpec {
            name: "fc_s8 1row,16out off128 (hello_world FC1, T3.6)",
            tier: MemoryTier::Sram,
            op: OpKind::FullyConnected,
            params: KernelParams::Fc(&SIMD_FC_1X16_PARAMS),
            reference: Some(CompetitorBaseline {
                name: "ESP-NN optimized (hello_world)",
                cycles: None,
                target_speedup_x100: None,
                source: "plan composed-kernels Scope table (user-verified 2026-08-10): hello_world 10,314 → 4,675 cyc (2.2x). Measured Hematite row lands in T6.x (on-device).",
            }),
            note: "T3.6 hello_world first dense: input_dim 1 (<16, previously gated out) zero-padded to 16; non-zero input_offset fold over padded rows.",
        },
        KernelSpec {
            name: "fc_s8 16row,1out off128 (hello_world FC3, T3.6)",
            tier: MemoryTier::Sram,
            op: OpKind::FullyConnected,
            params: KernelParams::Fc(&SIMD_FC_16X1_PARAMS),
            reference: None,
            note: "T3.6 hello_world final dense: input_dim 16 (%16, no pad), output_dim 1.",
        },
        KernelSpec {
            name: "fc_s8 8row,128out off128 (anomaly FC6, T3.6)",
            tier: MemoryTier::Sram,
            op: OpKind::FullyConnected,
            params: KernelParams::Fc(&SIMD_FC_8X128_PARAMS),
            reference: Some(CompetitorBaseline {
                name: "ESP-NN optimized (anomaly_detect)",
                cycles: None,
                target_speedup_x100: None,
                source: "plan composed-kernels Scope table (user-verified 2026-08-10): anomaly 28,550,253 → 7,758,145 cyc (3.7x). Measured Hematite row lands in T6.x (on-device).",
            }),
            note: "T3.6 anomaly_detect 6th dense (the only gated-out FC): input_dim 8 zero-padded to 16; non-zero input_offset fold over padded rows.",
        },
        KernelSpec {
            name: "depthwise_s8 12x12,16x3x3x16 S2 SAME (SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::DepthwiseConv2d,
            params: KernelParams::Depthwise(&SIMD_DEPTHWISE_S2_SAME_PARAMS),
            reference: None,
            note: "Phase B stride-2 SAME depthwise SIMD row: spatial zero-pad + stride-2 pixel loop.",
        },
        KernelSpec {
            name: "depthwise_s8 12x12,16x3x3x16 S2 SAME off-3 (SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::DepthwiseConv2d,
            params: KernelParams::Depthwise(&SIMD_DEPTHWISE_S2_SAME_OFF_PARAMS),
            reference: None,
            note: "Phase C non-zero input_offset fold row (input_offset=-3).",
        },
        KernelSpec {
            name: "depthwise_s8 12x12,12x3x3x12 SAME non16 (SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::DepthwiseConv2d,
            params: KernelParams::Depthwise(&SIMD_DEPTHWISE_NON16_PARAMS),
            reference: None,
            note: "Phase F non-%16 channel row: 12 channels zero-padded to 16 in scratch.",
        },
        KernelSpec {
            name: "depthwise_s8 12x12,8x3x3x16 dm2 (SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::DepthwiseConv2d,
            params: KernelParams::Depthwise(&SIMD_DEPTHWISE_DM2_PARAMS),
            reference: None,
            note: "T3.5 depth_multiplier=2 row: each input channel fans out to 2 output channels (replicated-input staging + per-channel requantize).",
        },
        KernelSpec {
            name: "depthwise_s8 12x12,8x3x3x32 dm4 (SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::DepthwiseConv2d,
            params: KernelParams::Depthwise(&SIMD_DEPTHWISE_DM4_PARAMS),
            reference: None,
            note: "T3.5 depth_multiplier=4 row: in_c 8 → out_c 32, 3×3 SAME.",
        },
        KernelSpec {
            name: "depthwise_s8 12x12,8x3x3x64 dm8 (SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::DepthwiseConv2d,
            params: KernelParams::Depthwise(&SIMD_DEPTHWISE_DM8_PARAMS),
            reference: Some(CompetitorBaseline {
                name: "ESP-NN optimized (kws depthwise fan-out)",
                cycles: None,
                target_speedup_x100: None,
                source: "plan composed-kernels Scope table (user-verified 2026-08-10): kws 12,983,503 → 1,059,889 cyc / 54 → 4 ms (12.3×); ESP-NN-relative bar = beat 1,059,889 cyc / 4 ms. Measured Hematite row lands in T6.x (on-device).",
            }),
            note: "T3.5 depth_multiplier=8 row — the KWS keyword-spotting fan-out shape family (dm=8).",
        },
        KernelSpec {
            name: "depthwise_s8 kws 49x40,1x10x8x8 dm8 S2 (SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::DepthwiseConv2d,
            params: KernelParams::Depthwise(&SIMD_DEPTHWISE_KWS_10X8_PARAMS),
            reference: Some(CompetitorBaseline {
                name: "ESP-NN optimized (kws real depthwise, s16 path)",
                cycles: None,
                target_speedup_x100: None,
                source: "plan composed-kernels Scope table (user-verified 2026-08-10): kws 12,983,503 → 1,059,889 cyc / 54 → 4 ms (12.3×); ESP-NN-relative bar = beat 1,059,889 cyc / 4 ms. Measured Hematite row lands in T6.x (on-device).",
            }),
            note: "T3.5b the REAL kws depthwise (tflite-verified [1,10,8,8], 80 taps, stride 2, dm 8, input_offset +128): the tap-parameterized anytap kernel runs it in 3 chunked QACC passes (32+32+16) — SIMD ENGAGES.",
        },
        KernelSpec {
            name: "depthwise_s8 12x12,8x5x5x8 dm1 S1 (SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::DepthwiseConv2d,
            params: KernelParams::Depthwise(&SIMD_DEPTHWISE_5X5_DM1_PARAMS),
            reference: None,
            note: "T3.5b arbitrary 5×5 filter, dm=1, stride 1 SAME (25-tap anytap path).",
        },
        KernelSpec {
            name: "depthwise_s8 12x12,1x5x5x8 dm8 S1 (SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::DepthwiseConv2d,
            params: KernelParams::Depthwise(&SIMD_DEPTHWISE_5X5_DM8_PARAMS),
            reference: None,
            note: "T3.5b arbitrary 5×5 filter, dm=8 fan-out (in_c 1 → out_c 8), stride 1 SAME.",
        },
        KernelSpec {
            name: "depthwise_s8 14x14,8x7x7x8 dm1 S2 (SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::DepthwiseConv2d,
            params: KernelParams::Depthwise(&SIMD_DEPTHWISE_7X7_DM1_PARAMS),
            reference: None,
            note: "T3.5b arbitrary 7×7 filter, dm=1, stride 2 SAME (49-tap anytap path).",
        },
        KernelSpec {
            name: "depthwise_s8 14x14,1x7x7x8 dm8 S2 (SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::DepthwiseConv2d,
            params: KernelParams::Depthwise(&SIMD_DEPTHWISE_7X7_DM8_PARAMS),
            reference: None,
            note: "T3.5b arbitrary 7×7 filter, dm=8 fan-out (in_c 1 → out_c 8), stride 2 SAME.",
        },
        KernelSpec {
            name: "depthwise_s8 12x12,8x3x5x16 dm2 S1 (SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::DepthwiseConv2d,
            params: KernelParams::Depthwise(&SIMD_DEPTHWISE_3X5_DM2_PARAMS),
            reference: None,
            note: "T3.5b non-square 3×5 filter, dm=2 (in_c 8 → out_c 16), stride 1 SAME (15-tap anytap path).",
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
            name: "avg_pool_s8 3x3 s1 p0 (T3.1 SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::AvgPool,
            params: KernelParams::Pool(&POOL_3X3_S1_VALID_PARAMS),
            reference: None,
            note: "T3.1 generic avg-pool: 3x3 stride-1 VALID (pad_total 0) — the hwc1 SIMD path engages on device; fixed-point vs ref documented in t31-pool.md.",
        },
        KernelSpec {
            name: "max_pool_s8 3x3 s1 p0 (T3.1 SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::MaxPool,
            params: KernelParams::Pool(&POOL_3X3_S1_VALID_PARAMS),
            reference: None,
            note: "T3.1 generic max-pool: 3x3 stride-1 VALID — hwc1 SIMD engages; max semantics equal ref bit-exact.",
        },
        KernelSpec {
            name: "avg_pool_s8 3x3 s1 p0 relu (T3.1 SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::AvgPool,
            params: KernelParams::Pool(&POOL_3X3_S1_VALID_RELU_PARAMS),
            reference: None,
            note: "T3.1 generic avg-pool with relu-range clamp (0..127) — the Rust clamp post-pass over the hwc1 output.",
        },
        KernelSpec {
            name: "max_pool_s8 3x3 s1 p0 relu (T3.1 SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::MaxPool,
            params: KernelParams::Pool(&POOL_3X3_S1_VALID_RELU_PARAMS),
            reference: None,
            note: "T3.1 generic max-pool with relu-range clamp — hwc1 SIMD + clamp post-pass.",
        },
        KernelSpec {
            name: "avg_pool_s8 3x3 s1 SAME (T3.1)",
            tier: MemoryTier::Sram,
            op: OpKind::AvgPool,
            params: KernelParams::Pool(&POOL_3X3_S1_SAME_PARAMS),
            reference: None,
            note: "T3.1 widened shape (pad 1): model-verified on host (avg fixed-point vs ref documented in t31-pool.md); scalar on device — the pool backend delivers no scratch for spatial padding.",
        },
        KernelSpec {
            name: "max_pool_s8 3x3 s1 SAME (T3.1)",
            tier: MemoryTier::Sram,
            op: OpKind::MaxPool,
            params: KernelParams::Pool(&POOL_3X3_S1_SAME_PARAMS),
            reference: None,
            note: "T3.1 widened shape (pad 1): max model == ref bit-exact (host-verified); scalar on device.",
        },
        KernelSpec {
            name: "avg_pool_s8 3x3 s2 SAME (T3.1)",
            tier: MemoryTier::Sram,
            op: OpKind::AvgPool,
            params: KernelParams::Pool(&POOL_3X3_S2_SAME_PARAMS),
            reference: None,
            note: "T3.1 widened shape (asymmetric SAME, partial windows): model-verified on host; scalar on device.",
        },
        KernelSpec {
            name: "max_pool_s8 3x3 s2 SAME (T3.1)",
            tier: MemoryTier::Sram,
            op: OpKind::MaxPool,
            params: KernelParams::Pool(&POOL_3X3_S2_SAME_PARAMS),
            reference: None,
            note: "T3.1 widened shape (asymmetric SAME): max model == ref (host-verified); scalar on device.",
        },
        KernelSpec {
            name: "avg_pool_s8 5x5 s1 SAME (T3.1)",
            tier: MemoryTier::Sram,
            op: OpKind::AvgPool,
            params: KernelParams::Pool(&POOL_5X5_S1_SAME_PARAMS),
            reference: None,
            note: "T3.1 widened shape (5x5, pad 2): model-verified on host; scalar on device.",
        },
        KernelSpec {
            name: "max_pool_s8 5x5 s1 SAME (T3.1)",
            tier: MemoryTier::Sram,
            op: OpKind::MaxPool,
            params: KernelParams::Pool(&POOL_5X5_S1_SAME_PARAMS),
            reference: None,
            note: "T3.1 widened shape (5x5, pad 2): max model == ref (host-verified); scalar on device.",
        },
        KernelSpec {
            name: "avg_pool_s8 5x5 s2 SAME (T3.1)",
            tier: MemoryTier::Sram,
            op: OpKind::AvgPool,
            params: KernelParams::Pool(&POOL_5X5_S2_SAME_PARAMS),
            reference: None,
            note: "T3.1 widened shape (5x5 stride-2 SAME, pad 1): model-verified on host; scalar on device.",
        },
        KernelSpec {
            name: "max_pool_s8 5x5 s2 SAME (T3.1)",
            tier: MemoryTier::Sram,
            op: OpKind::MaxPool,
            params: KernelParams::Pool(&POOL_5X5_S2_SAME_PARAMS),
            reference: None,
            note: "T3.1 widened shape (5x5 stride-2 SAME): max model == ref (host-verified); scalar on device.",
        },
        KernelSpec {
            name: "avg_pool_s8 7x7 global (T3.1 SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::AvgPool,
            params: KernelParams::Pool(&POOL_7X7_GLOBAL_PARAMS),
            reference: None,
            note: "T3.1 global avg-pool 7x7x16 (pad_total 0) — the hwc1 SIMD path on a 1x1 output.",
        },
        KernelSpec {
            name: "max_pool_s8 7x7 global (T3.1 SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::MaxPool,
            params: KernelParams::Pool(&POOL_7X7_GLOBAL_PARAMS),
            reference: None,
            note: "T3.1 global max-pool 7x7x16 — hwc1 SIMD engages.",
        },
        KernelSpec {
            name: "avg_pool_s8 3x3 s1 p0 c24 (T3.1 tail)",
            tier: MemoryTier::Sram,
            op: OpKind::AvgPool,
            params: KernelParams::Pool(&POOL_3X3_S1_VALID_C24_PARAMS),
            reference: None,
            note: "T3.1 C%16 scalar tail (24 channels): host model-verified; scalar on device (device gate requires C % 16).",
        },
        KernelSpec {
            name: "max_pool_s8 3x3 s1 p0 c24 (T3.1 tail)",
            tier: MemoryTier::Sram,
            op: OpKind::MaxPool,
            params: KernelParams::Pool(&POOL_3X3_S1_VALID_C24_PARAMS),
            reference: None,
            note: "T3.1 C%16 scalar tail (24 channels): host model-verified; scalar on device.",
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
            name: "add_s8 256 non-identity offsets (T3.2)",
            tier: MemoryTier::Sram,
            op: OpKind::Add,
            params: KernelParams::Elementwise(&SIMD_ADD_256_NONIDENTITY_PARAMS),
            reference: None,
            note: "T3.2 widened lane-model SIMD row: two-stage TFLM Add rounding (zps -5/3/1, left_shift 20, per-input multipliers, output requantize, clamped range). NOT identity-eligible — engages the 16-wide per-lane requantize model on device.",
        },
        KernelSpec {
            name: "sub_s8 256 non-identity offsets (T3.2)",
            tier: MemoryTier::Sram,
            op: OpKind::Sub,
            params: KernelParams::Elementwise(&SIMD_SUB_256_NONIDENTITY_PARAMS),
            reference: None,
            note: "T3.2 widened lane-model SIMD row: same non-identity two-stage rounding shape as the ADD row.",
        },
        KernelSpec {
            name: "mul_s8 256 non-identity offsets (T3.2)",
            tier: MemoryTier::Sram,
            op: OpKind::Mul,
            params: KernelParams::Elementwise(&SIMD_MUL_256_NONIDENTITY_PARAMS),
            reference: None,
            note: "T3.2 widened lane-model SIMD row: non-zero offsets + real product requantize (scale change). NOT identity-eligible — engages the 16-wide per-lane requantize model on device.",
        },
        KernelSpec {
            name: "relu6_s8 256 (T3.2 SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::Relu6,
            params: KernelParams::Activation(&SIMD_RELU6_256_PARAMS),
            reference: None,
            note: "T3.2 relu6 lane-model SIMD row: clamp to quantized_six = 24 (quantized_activation_max forwarded), zps -1/2. Vector min/max clamps in 16-wide lanes.",
        },
        KernelSpec {
            name: "hard_swish_s8 256 (T3.2 SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::HardSwish,
            params: KernelParams::Activation(&SIMD_HARD_SWISH_256_PARAMS),
            reference: None,
            note: "T3.2 hard_swish lane-model SIMD row: DOWNGRADED integer rational formula (goldens-pinned, NOT the TFLM fixed-point chain); per-lane /6 scalar tail (no SIMD integer division on Xtensa).",
        },
        KernelSpec {
            name: "fused_conv1x1_s8 4x4x16 residual+relu (T2.2)",
            tier: MemoryTier::Sram,
            op: OpKind::FusedConv2d,
            params: KernelParams::FusedConv(&FUSED_CONV1X1_RESIDUAL_RELU_PARAMS),
            reference: None,
            note: "Composed conv1x1 + residual-ADD + ReLU in one pass (ACCX conv path; residual read in place, conv output register-held). The mv2 pointwise residual-block prototype.",
        },
        KernelSpec {
            name: "fused_conv3x3_s8 6x6x8 residual+hardswish (T2.2)",
            tier: MemoryTier::Sram,
            op: OpKind::FusedConv2d,
            params: KernelParams::FusedConv(&FUSED_CONV3X3_RESIDUAL_HSWISH_PARAMS),
            reference: None,
            note: "Composed 3x3 conv (non-%16 channels → channel-padded staging) + residual-ADD + HardSwish epilogue (downgraded integer formula).",
        },
        KernelSpec {
            name: "fused_chain_s8 4op add+relu+mul+hardswish (T2.3)",
            tier: MemoryTier::Sram,
            op: OpKind::FusedElementwiseChain,
            params: KernelParams::FusedChain(&FUSED_CHAIN_4OP_PARAMS),
            reference: None,
            note: "Composed elementwise chain (add+relu+mul+hard_swish, non-identity scales, n=256). On device today this chain runs the decomposition (hard_swish has no SIMD yet, T3.2 — chains SIMD-engage only when EVERY step is identity-eligible). The chain-runtime < sum-of-per-op-runtimes measurement is T6.x.",
        },
        KernelSpec {
            name: "fused_pool_s8 4x4x16 avg+mulfold+relu (T2.4)",
            tier: MemoryTier::Sram,
            op: OpKind::FusedPoolFold,
            params: KernelParams::FusedPool(&FUSED_POOL_FOLD_PARAMS),
            reference: None,
            note: "Composed avg-pool + identity MUL input fold + ReLU epilogue (T2.4). The fold is in the provably-exact subset (zero offsets, (1<<30, 0) output pair) so on device the fold SIMD-engages via the elementwise gates and the pool SIMD reads the staged scratch directly.",
        },
        KernelSpec {
            name: "softmax_s8 1x1000 (SIMD)",
            tier: MemoryTier::Sram,
            op: OpKind::Softmax,
            params: KernelParams::Softmax(&SOFTMAX_1X1000_PARAMS),
            reference: None,
            note: "Phase E: VMAX find-max + exp cache; row_size 1000.",
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
        KernelParams::Activation(_p) => {
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
        // The fused conv consumes the anchor conv's input/weights/bias and
        // produces the conv output shape. The residual is a const tensor
        // embedded in the params — no buffer region needed. transform_len 0:
        // the fused ACCX path reads the RAW `[oc][ic]` weights (no TIE728
        // 11cn/33cn permutation).
        KernelParams::FusedConv(p) => SpecLayout {
            input_len: shape_product(&p.conv.input_shape),
            weights_len: shape_product(&p.conv.filter_shape),
            bias_len: p.conv.filter_shape[0] as usize,
            output_len: shape_product(&p.conv.output_shape),
            transform_len: 0,
        },
        // The fused elementwise chain consumes a single flat int8 input
        // (`src`, `num_elements` elements) and produces `num_elements` i8
        // outputs. The step operands are model constant tensors embedded in
        // the params — no buffer region needed (same as the fused-conv
        // residual).
        KernelParams::FusedChain(p) => SpecLayout {
            input_len: p.num_elements as usize,
            weights_len: 0,
            bias_len: 0,
            output_len: p.num_elements as usize,
            transform_len: 0,
        },
        // The composed pool-fold consumes the anchor pool's input and produces
        // the pool output shape. The fold operand is a const tensor embedded
        // in the params — no buffer region needed (same as the fused-conv
        // residual and the chain operands).
        KernelParams::FusedPool(p) => SpecLayout {
            input_len: shape_product(&p.pool.input_shape),
            weights_len: 0,
            bias_len: 0,
            output_len: shape_product(&p.pool.output_shape),
            transform_len: 0,
        },
    }
}

/// Scratch bytes a spec's kernel needs at runtime (host tests allocate
/// exactly this much). 0 for every row except the composed pool-fold, whose
/// decomposition stages the absorbed fold into scratch
/// (`fused_pool_with_fold_scratch_need` — the codegen `emit_fused_pool_fold`
/// value).
pub fn spec_scratch_need(spec: &KernelSpec) -> usize {
    match &spec.params {
        KernelParams::FusedPool(p) => {
            hematite_s3::backend::fused_pool_with_fold_scratch_need(p)
        }
        _ => 0,
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
        OpKind::Relu6 => {
            let p = params_activation(spec);
            hematite_s3::activations::relu6(bufs.input, p, bufs.output, scratch, p.quantized_activation_max)
        }
        OpKind::HardSwish => {
            let p = params_activation(spec);
            hematite_s3::activations::hard_swish(bufs.input, p, bufs.output, scratch)
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
        OpKind::FusedConv2d => {
            let p = params_fused_conv(spec);
            let mut backend = hematite_s3::backend::S3Backend;
            hematite_core::FusedKernelBackend::fused_conv2d(
                &mut backend, bufs.input, bufs.weights, bufs.bias, p, bufs.output, scratch,
            )
        }
        OpKind::FusedElementwiseChain => {
            let p = params_fused_chain(spec);
            let mut backend = hematite_s3::backend::S3Backend;
            hematite_core::FusedKernelBackend::fused_elementwise_chain(
                &mut backend, bufs.input, p, bufs.output,
            )
        }
        OpKind::FusedPoolFold => {
            let p = params_fused_pool(spec);
            let mut backend = hematite_s3::backend::S3Backend;
            hematite_core::FusedKernelBackend::fused_pool_with_fold(
                &mut backend, bufs.input, p, bufs.output, scratch,
            )
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
    Depthwise(hematite_s3::depthwise::PreparedDepthwise),
    MaxPool(hematite_s3::pool::PreparedMaxPool),
    AvgPool(hematite_s3::pool::PreparedAvgPool),
    Relu(hematite_s3::activations::PreparedRelu),
    Add(hematite_s3::elementwise::PreparedAdd),
    Mul(hematite_s3::elementwise::PreparedMul),
    Sub(hematite_s3::elementwise::PreparedSub),
    /// Ops with no SIMD path (softmax): just run the public API.
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
            PreparedKernel::Depthwise(h) => h.is_simd(),
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
            PreparedKernel::Depthwise(h) => {
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
        OpKind::Relu6 => {
            let p = match spec.params {
                KernelParams::Activation(p) => p,
                _ => return Err(KernelError::Unsupported),
            };
            hematite_s3::activations::relu6(bufs.input, p, bufs.output, scratch, p.quantized_activation_max)
        }
        OpKind::HardSwish => {
            let p = match spec.params {
                KernelParams::Activation(p) => p,
                _ => return Err(KernelError::Unsupported),
            };
            hematite_s3::activations::hard_swish(bufs.input, p, bufs.output, scratch)
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
        OpKind::FusedConv2d => {
            let p = match spec.params {
                KernelParams::FusedConv(p) => p,
                _ => return Err(KernelError::Unsupported),
            };
            let mut backend = hematite_s3::backend::S3Backend;
            hematite_core::FusedKernelBackend::fused_conv2d(
                &mut backend, bufs.input, bufs.weights, bufs.bias, p, bufs.output, scratch,
            )
        }
        OpKind::FusedElementwiseChain => {
            let p = match spec.params {
                KernelParams::FusedChain(p) => p,
                _ => return Err(KernelError::Unsupported),
            };
            let mut backend = hematite_s3::backend::S3Backend;
            hematite_core::FusedKernelBackend::fused_elementwise_chain(
                &mut backend, bufs.input, p, bufs.output,
            )
        }
        OpKind::FusedPoolFold => {
            let p = match spec.params {
                KernelParams::FusedPool(p) => p,
                _ => return Err(KernelError::Unsupported),
            };
            let mut backend = hematite_s3::backend::S3Backend;
            hematite_core::FusedKernelBackend::fused_pool_with_fold(
                &mut backend, bufs.input, p, bufs.output, scratch,
            )
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
        // relu6/hard_swish have no prepared handle: the public `relu6`/
        // `hard_swish` kernels dispatch the T3.2 widened lane model
        // internally (via S3Backend), so the Scalar slot runs them.
        OpKind::Relu6 | OpKind::HardSwish => match spec.params {
            KernelParams::Activation(_) => Ok(PreparedKernel::Scalar),
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
        OpKind::DepthwiseConv2d => match spec.params {
            KernelParams::Depthwise(p) => Ok(PreparedKernel::Depthwise(
                hematite_s3::depthwise::PreparedDepthwise::new(p)?,
            )),
            _ => Err(KernelError::Unsupported),
        },
        // The fused conv has no prepared handle: `fused_conv2d` internally
        // dispatches the anchor conv's ACCX SIMD path (via S3Backend), so the
        // Scalar slot (which runs the public trait method) gets SIMD on device.
        OpKind::FusedConv2d => match spec.params {
            KernelParams::FusedConv(_) => Ok(PreparedKernel::Scalar),
            _ => Err(KernelError::Unsupported),
        },
        // The fused chain has no prepared handle either: `fused_elementwise_chain`
        // internally dispatches the register-held chain SIMD path (via
        // S3Backend), so the Scalar slot runs the public trait method.
        OpKind::FusedElementwiseChain => match spec.params {
            KernelParams::FusedChain(_) => Ok(PreparedKernel::Scalar),
            _ => Err(KernelError::Unsupported),
        },
        // The fused pool-fold has no prepared handle either: `fused_pool_with_fold`
        // internally dispatches the fold elementwise SIMD + the pool SIMD (via
        // S3Backend), so the Scalar slot runs the public trait method.
        OpKind::FusedPoolFold => match spec.params {
            KernelParams::FusedPool(_) => Ok(PreparedKernel::Scalar),
            _ => Err(KernelError::Unsupported),
        },
        OpKind::Softmax => Ok(PreparedKernel::Scalar),
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
#[cfg(target_arch = "xtensa")]
fn params_fused_conv(spec: &KernelSpec) -> &'static FusedConvParams<'static> {
    match spec.params {
        KernelParams::FusedConv(p) => p,
        _ => panic!("spec.op fused_conv2d requires KernelParams::FusedConv"),
    }
}

#[cfg(target_arch = "xtensa")]
fn params_fused_chain(spec: &KernelSpec) -> &'static ElementwiseChainParams<'static> {
    match spec.params {
        KernelParams::FusedChain(p) => p,
        _ => panic!("spec.op fused_elementwise_chain requires KernelParams::FusedChain"),
    }
}

#[cfg(target_arch = "xtensa")]
fn params_fused_pool(spec: &KernelSpec) -> &'static FoldedPoolParams<'static> {
    match spec.params {
        KernelParams::FusedPool(p) => p,
        _ => panic!("spec.op fused_pool_with_fold requires KernelParams::FusedPool"),
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
        OpKind::Relu6 => {
            let p = params_activation(spec);
            hematite_ref::activation::relu6(bufs.input, p, bufs.output, scratch, p.quantized_activation_max)
        }
        OpKind::HardSwish => {
            let p = params_activation(spec);
            hematite_ref::activation::hard_swish(bufs.input, p, bufs.output, scratch)
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
        OpKind::FusedConv2d => {
            let p = params_fused_conv(spec);
            let mut backend = hematite_ref::RefBackend;
            hematite_core::FusedKernelBackend::fused_conv2d(
                &mut backend, bufs.input, bufs.weights, bufs.bias, p, bufs.output, scratch,
            )
        }
        OpKind::FusedElementwiseChain => {
            let p = params_fused_chain(spec);
            let mut backend = hematite_ref::RefBackend;
            hematite_core::FusedKernelBackend::fused_elementwise_chain(
                &mut backend, bufs.input, p, bufs.output,
            )
        }
        OpKind::FusedPoolFold => {
            let p = params_fused_pool(spec);
            let mut backend = hematite_ref::RefBackend;
            hematite_core::FusedKernelBackend::fused_pool_with_fold(
                &mut backend, bufs.input, p, bufs.output, scratch,
            )
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
            OpKind::Relu6 => {
                let p = match spec.params {
                    KernelParams::Activation(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                hematite_s3::activations::relu6(bufs.input, p, bufs.output, scratch, p.quantized_activation_max)
            }
            OpKind::HardSwish => {
                let p = match spec.params {
                    KernelParams::Activation(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                hematite_s3::activations::hard_swish(bufs.input, p, bufs.output, scratch)
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
            OpKind::FusedConv2d => {
                let p = match spec.params {
                    KernelParams::FusedConv(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                let mut backend = hematite_s3::backend::S3Backend;
                hematite_core::FusedKernelBackend::fused_conv2d(
                    &mut backend, bufs.input, bufs.weights, bufs.bias, p, bufs.output, scratch,
                )
            }
            OpKind::FusedElementwiseChain => {
                let p = match spec.params {
                    KernelParams::FusedChain(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                let mut backend = hematite_s3::backend::S3Backend;
                hematite_core::FusedKernelBackend::fused_elementwise_chain(
                    &mut backend, bufs.input, p, bufs.output,
                )
            }
            OpKind::FusedPoolFold => {
                let p = match spec.params {
                    KernelParams::FusedPool(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                let mut backend = hematite_s3::backend::S3Backend;
                hematite_core::FusedKernelBackend::fused_pool_with_fold(
                    &mut backend, bufs.input, p, bufs.output, scratch,
                )
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
            OpKind::Relu6 => {
                let p = match spec.params {
                    KernelParams::Activation(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                hematite_ref::activation::relu6(bufs.input, p, bufs.output, scratch, p.quantized_activation_max)
            }
            OpKind::HardSwish => {
                let p = match spec.params {
                    KernelParams::Activation(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                hematite_ref::activation::hard_swish(bufs.input, p, bufs.output, scratch)
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
            OpKind::FusedConv2d => {
                let p = match spec.params {
                    KernelParams::FusedConv(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                let mut backend = hematite_ref::RefBackend;
                hematite_core::FusedKernelBackend::fused_conv2d(
                    &mut backend, bufs.input, bufs.weights, bufs.bias, p, bufs.output, scratch,
                )
            }
            OpKind::FusedElementwiseChain => {
                let p = match spec.params {
                    KernelParams::FusedChain(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                let mut backend = hematite_ref::RefBackend;
                hematite_core::FusedKernelBackend::fused_elementwise_chain(
                    &mut backend, bufs.input, p, bufs.output,
                )
            }
            OpKind::FusedPoolFold => {
                let p = match spec.params {
                    KernelParams::FusedPool(p) => p,
                    _ => return Err("op/params mismatch".into()),
                };
                let mut backend = hematite_ref::RefBackend;
                hematite_core::FusedKernelBackend::fused_pool_with_fold(
                    &mut backend, bufs.input, p, bufs.output, scratch,
                )
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
            let mut scratch = vec![0u8; spec_scratch_need(spec)];
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
            let mut scratch = vec![0u8; spec_scratch_need(spec)];
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
            let mut scratch = vec![0u8; spec_scratch_need(spec)];
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
                let is_depthwise = matches!(spec.params, KernelParams::Depthwise(_));
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
                    KernelParams::Depthwise(p) => {
                        (
                            p.input_shape[3] as usize,
                            p.output_shape[3] as usize,
                            (p.filter_shape[1].max(1) * p.filter_shape[2].max(1)) as usize,
                        )
                    }
                    _ => continue,
                };
                let eligible = if is_depthwise {
                    match &spec.params {
                        KernelParams::Depthwise(p) => {
                            in_c >= 1
                                && out_c >= 1
                                && out_c == in_c * p.depth_multiplier.max(1) as usize
                        }
                        _ => unreachable!(),
                    }
                } else {
                    match &spec.params {
                        // T3.6 — FC accepts any input_dim >= 1: small /
                        // non-16 input dims are zero-padded in scratch.
                        KernelParams::Fc(_) => in_c >= 1 && out_c >= 1,
                        _ => in_c >= 16 && in_c % 16 == 0 && out_c >= 1,
                    }
                };
                if !eligible {
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
                        // Mirror the dispatch's SAME-pad derivation (same formula
                        // as conv3x3.rs conv3x3_accx_dispatch). The ACCX kernel
                        // runs on a zero-padded copy, so out-of-padded-bounds taps
                        // contribute 0 — equivalent to skipping them here.
                        let dilated_h = (p.filter_shape[1] - 1) * p.dilation_height_factor + 1;
                        let dilated_w = (p.filter_shape[2] - 1) * p.dilation_width_factor + 1;
                        let pad_h = (((out_h as i32 - 1) * p.stride_height + dilated_h - p.input_shape[1])
                            / 2)
                            .max(0) as usize;
                        let pad_w = (((out_w as i32 - 1) * p.stride_width + dilated_w - p.input_shape[2])
                            / 2)
                            .max(0) as usize;
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
                                        let ih = (oh * stride_h + kh * dil_h) as i32 - pad_h as i32;
                                        let iw = (ow * stride_w + kw * dil_w) as i32 - pad_w as i32;
                                        if ih < 0 || iw < 0 || ih as usize >= in_h || iw as usize >= in_w
                                        {
                                            continue;
                                        }
                                        let ih = ih as usize;
                                        let iw = iw as usize;
                                        for ic in 0..in_c {
                                            let in_idx = (ih * in_w + iw) * in_c + ic;
                                            // RAW [oc][tap][ic] — asm filter[(oc*taps+tap)*in_c+ic]
                                            let w_idx = (oc * taps + tap) * in_c + ic;
                                            acc += bufs.weights[w_idx] as i64
                                                * (bufs.input[in_idx] as i64 + p.input_offset as i64);
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
                        // T3.6 — mirror the dispatch's pad-in-scratch: for
                        // input_dim % 16 != 0 the SIMD kernel runs on a
                        // zero-padded input (real lanes then zeros) and
                        // zero-padded weight rows (pad lanes zero), then the
                        // input_offset fold reads weight sums over the padded
                        // rows. Padded lanes contribute 0×0 = 0 — this equals
                        // the scalar `Σ (in + off)·w` exactly.
                        let padded_dim = in_c.div_ceil(16) * 16;
                        for oc in 0..out_c {
                            let mut raw: i64 = 0;
                            let mut wsum: i64 = 0;
                            for ic in 0..padded_dim {
                                let w = if ic < in_c {
                                    bufs.weights[oc * in_c + ic] as i64
                                } else {
                                    0
                                };
                                let x = if ic < in_c { bufs.input[ic] as i64 } else { 0 };
                                raw += x * w;
                                wsum += w;
                            }
                            let acc = raw + p.input_offset as i64 * wsum;
                            let acc32 = (bufs.bias[oc] as i64 + acc)
                                .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
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
                    KernelParams::Depthwise(p) => {
                        // TFLM depthwise fan-out (T3.5): output channel
                        // `oc = i*dm + j` reads input channel `i`; weights are
                        // HWCN [tap][out_c], so `w_idx = tap*out_c + oc`. The
                        // SIMD dispatch stages a replicated input so the
                        // per-lane kernel contract equals this accumulation.
                        let in_h = p.input_shape[1] as usize;
                        let in_w = p.input_shape[2] as usize;
                        let out_h = p.output_shape[1] as usize;
                        let out_w = p.output_shape[2] as usize;
                        let dm = p.depth_multiplier.max(1) as usize;
                        let fw = p.filter_shape[2].max(1) as usize;
                        let stride_h = p.stride_height as usize;
                        let stride_w = p.stride_width as usize;
                        let dilated_h = (p.filter_shape[1] - 1) * p.dilation_height_factor + 1;
                        let dilated_w = (p.filter_shape[2] - 1) * p.dilation_width_factor + 1;
                        let pad_h = (((out_h as i32 - 1) * p.stride_height + dilated_h
                            - p.input_shape[1])
                            / 2)
                            .max(0) as usize;
                        let pad_w = (((out_w as i32 - 1) * p.stride_width + dilated_w
                            - p.input_shape[2])
                            / 2)
                            .max(0) as usize;
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                #[allow(clippy::needless_range_loop)]
                                for oc in 0..out_c {
                                    let ic = oc / dm;
                                    let mut acc: i64 = 0;
                                    for tap in 0..taps {
                                        let (kh, kw) = (tap / fw, tap % fw);
                                        let ih = (oh * stride_h + kh) as i32 - pad_h as i32;
                                        let iw = (ow * stride_w + kw) as i32 - pad_w as i32;
                                        if ih < 0
                                            || iw < 0
                                            || ih as usize >= in_h
                                            || iw as usize >= in_w
                                        {
                                            continue;
                                        }
                                        let in_idx = (ih as usize * in_w + iw as usize) * in_c + ic;
                                        let w_idx = tap * out_c + oc;
                                        acc += bufs.weights[w_idx] as i64
                                            * (bufs.input[in_idx] as i64 + p.input_offset as i64);
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
            let mut scratch = vec![0u8; spec_scratch_need(spec)];
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
