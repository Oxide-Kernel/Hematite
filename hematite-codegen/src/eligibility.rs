// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! T4.1 — host-side mirror of every SIMD-eligibility gate in `hematite-s3`.
//!
//! The future composed-kernel selector (T4.2) and the W0 fused-profile
//! (`optimize::profile`) must decide SIMD eligibility **host-side / at macro
//! time**. The gates themselves live in `hematite-s3` (all `pub(crate)` or
//! private) — the proc-macro crate cannot depend on `hematite-s3` at macro
//! time, and the gate fns have no cross-crate surface (the `Prepared*`
//! handles AND their `is_simd()` with the device cfg, so on host they always
//! report `false`). This module therefore keeps a **self-contained copy** of
//! every gate the selector/profile use, kept in lockstep by the in-crate
//! parity tests below (`parity_*`).
//!
//! # Discipline
//!
//! * Every mirror fn mirrors EXACTLY ONE s3 gate/dispatch predicate at HEAD
//!   (commit 4ecd1ac) and cites its source fn + file:line in its doc comment.
//! * **s3 is canonical**: on any mirror/s3 divergence the MIRROR is fixed —
//!   never the s3 gate, never the s3 kernels.
//! * The mirror takes the same shape inputs the s3 gate takes (the
//!   `hematite-core` op-param types, already a regular dependency of this
//!   crate) and returns the same `bool`/`Option`.
//! * Runtime-only halves of the real engagement (pointer 16B alignment,
//!   `scratch.len()`, `n % 16`/`n >= 16` element counts) are NOT host-visible
//!   from a macro and are excluded — each mirror documents which runtime
//!   half remains.
//! * The parity test (`simd_eligibility_parity_spec_corpus_and_grids`)
//!   asserts mirror == the verbatim gate transcription (`s3_ref`, below) for
//!   every spec-corpus shape and a widened grid. The transcription is the
//!   s3-side oracle; it is independently pinned by s3's own in-crate
//!   gate-expectation tests (pool.rs:1854, fused.rs:1499/1695).

#![allow(dead_code)]

use hematite_core::op_params::{
    ActivationParams, ElementwiseChainParams, ElementwiseChainStep, ElementwiseKind,
    ElementwiseParams, FoldedPoolParams, PoolInputFold, PoolParams,
};

// ---------------------------------------------------------------------------
// ACCX conv-family gates (mirrors hematite-s3/src/accx.rs)
// ---------------------------------------------------------------------------

/// Mirrors `accx_eligible_1x1` (hematite-s3/src/accx.rs:64-66): the strict
/// unpadded 1×1/FC gate. Runtime half not visible here: 16B-aligned
/// input/filter pointers.
#[inline]
pub(crate) fn accx_eligible_1x1(input_c: usize, out_c: usize) -> bool {
    input_c >= 16 && input_c.is_multiple_of(16) && out_c >= 1
}

/// Mirrors `accx_eligible_1x1_padded` (hematite-s3/src/accx.rs:86-88): the
/// widened gate accepting any `input_c >= 1` — small/non-%16 input dims are
/// zero-padded in scratch (T3.6 FC, T3.3 conv1x1). Runtime half not visible
/// here: the pad-in-scratch staging needs `scratch >= need` bytes.
#[inline]
pub(crate) fn accx_eligible_1x1_padded(input_c: usize, out_c: usize) -> bool {
    input_c >= 1 && out_c >= 1
}

/// Mirrors `accx_eligible_3x3` (hematite-s3/src/accx.rs:97-99): any
/// `input_c >= 1` — non-%16 channels are zero-padded in scratch.
#[inline]
pub(crate) fn accx_eligible_3x3(input_c: usize, out_c: usize) -> bool {
    input_c >= 1 && out_c >= 1
}

/// Mirrors `accx_eligible_depthwise_dm` (hematite-s3/src/accx.rs:114-121):
/// depth-multiplier-aware depthwise gate — the fan-out shape invariant
/// `out_c == input_c * dm` (dm clamped to >= 1) must hold.
#[inline]
pub(crate) fn accx_eligible_depthwise_dm(
    input_c: usize,
    out_c: usize,
    depth_multiplier: i32,
) -> bool {
    let dm = depth_multiplier.max(1) as usize;
    input_c >= 1 && out_c >= 1 && out_c == input_c * dm
}

// ---------------------------------------------------------------------------
// Dispatch-level conv-family predicates (the FULL shape predicate the s3
// dispatchers check — the raw accx gates above are only part of it)
// ---------------------------------------------------------------------------

/// Mirrors `conv1x1_accx_dispatch`'s shape gate (hematite-s3/src/conv1x1.rs:
/// 214-224): stride 1, dilation 1, `out_h == in_h` / `out_w == in_w` (no
/// spatial padding), and the widened padded gate. Runtime halves not visible
/// here: scratch sizing, 16B alignment.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn conv1x1_dispatch_eligible(
    input_c: usize,
    out_c: usize,
    stride_h: i32,
    stride_w: i32,
    dil_h: i32,
    dil_w: i32,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> bool {
    stride_h == 1
        && stride_w == 1
        && dil_h == 1
        && dil_w == 1
        && in_h == out_h
        && in_w == out_w
        && accx_eligible_1x1_padded(input_c, out_c)
}

/// Mirrors `conv3x3_accx_dispatch`'s shape gate (hematite-s3/src/conv3x3.rs:
/// 128-139): dilation 1, filter exactly 3×3, the widened 3×3 gate, and the
/// Phase-C fold bound `input_offset.abs() <= 127` (the padded fill
/// `-input_offset` must fit i8). Runtime halves: scratch sizing, alignment.
#[inline]
pub(crate) fn conv3x3_dispatch_eligible(
    input_c: usize,
    out_c: usize,
    filter_h: i32,
    filter_w: i32,
    dil_h: i32,
    dil_w: i32,
    input_offset: i32,
) -> bool {
    if dil_h != 1
        || dil_w != 1
        || filter_h != 3
        || filter_w != 3
        || !accx_eligible_3x3(input_c, out_c)
    {
        return false;
    }
    if input_offset != 0 && input_offset.abs() > 127 {
        return false;
    }
    true
}

/// Mirrors `depthwise_accx_dispatch`'s shape gate (hematite-s3/src/
/// depthwise.rs:310-324): dilation 1, filter `>= 1x1` (anytap path, T3.5b),
/// the dm-aware gate, and the Phase-C fold bound `input_offset` in
/// [-127, 128]. Runtime halves: scratch sizing, alignment.
// The `manual_range_contains` form is kept verbatim to the s3 source
// (depthwise.rs:322) — the parity transcription must stay byte-identical.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::manual_range_contains)]
#[inline]
pub(crate) fn depthwise_dispatch_eligible(
    input_c: usize,
    out_c: usize,
    depth_multiplier: i32,
    filter_h: i32,
    filter_w: i32,
    dil_h: i32,
    dil_w: i32,
    input_offset: i32,
) -> bool {
    if dil_h != 1
        || dil_w != 1
        || filter_h < 1
        || filter_w < 1
        || !accx_eligible_depthwise_dm(input_c, out_c, depth_multiplier)
    {
        return false;
    }
    if input_offset != 0 && (input_offset < -127 || input_offset > 128) {
        return false;
    }
    true
}

/// FC dispatch gate — mirrors `fc_accx_dispatch` (hematite-s3/src/gemm.rs:
/// 137), which uses the widened padded gate `accx_eligible_1x1_padded`
/// (T3.6 small-shape FC SIMD; input_dim 1/8 etc. pad to 16 in scratch).
/// Note: the FC dispatch has NO input_offset bound (unlike conv3x3).
#[inline]
pub(crate) fn fc_dispatch_eligible(input_dim: usize, output_dim: usize) -> bool {
    accx_eligible_1x1_padded(input_dim, output_dim)
}

// ---------------------------------------------------------------------------
// Pool gate (mirrors hematite-s3/src/pool.rs)
// ---------------------------------------------------------------------------

/// `round(2^shift / area)` — mirrors `round_recip` (pool.rs:1130-1133).
#[inline]
fn pool_round_recip(area: i32, shift: i32) -> i32 {
    let num = 1i64 << shift;
    ((num + area as i64 / 2) / area as i64) as i32
}

/// The fixed-point avg-pool reciprocal — mirrors `pool_area_inv`
/// (pool.rs:1142-1153). `None` when the reciprocal cannot fit an i8 lane
/// (area < 3 or > 512 at the shift 8..24 search window).
#[inline]
fn pool_area_inv(area: i32) -> Option<(i32, [i8; 16])> {
    let mut shift = 8;
    let mut inv = pool_round_recip(area, shift);
    while inv > 127 && shift < 24 {
        shift += 1;
        inv = pool_round_recip(area, shift);
    }
    if !(1..=127).contains(&inv) {
        return None;
    }
    Some((shift, [inv as i8; 16]))
}

/// Mirrors `simd_eligible_pool` (hematite-s3/src/pool.rs:1171-1208): the
/// generic TIE728 `*_hwc1` gate — any filter/stride with **no padding and no
/// partial windows** (`(out-1)·stride + filter - in <= 0` on both axes),
/// channels `> 0` and `% 16 == 0`, and the avg reciprocal must fit an i8
/// lane. Returns `true` exactly when the s3 gate returns `Some`; the s3
/// `PoolSimdCfg` content (shift/area_inv/offsets/clamp) is selector-
/// irrelevant. The activation range is NOT part of the gate (T3.1 widened —
/// the clamp runs as a Rust post-pass). Runtime halves: 16B-aligned
/// input/output pointers.
#[inline]
pub(crate) fn simd_eligible_pool(params: &PoolParams) -> bool {
    let input_h = params.input_shape[1];
    let input_w = params.input_shape[2];
    let channels = params.input_shape[3];
    let filter_h = params.filter_height;
    let filter_w = params.filter_width;
    let out_h = params.output_shape[1];
    let out_w = params.output_shape[2];

    let pad_total_h = (out_h - 1) * params.stride_height + filter_h - input_h;
    let pad_total_w = (out_w - 1) * params.stride_width + filter_w - input_w;

    if filter_h < 1
        || filter_w < 1
        || params.stride_height < 1
        || params.stride_width < 1
        || pad_total_h > 0
        || pad_total_w > 0
        || channels <= 0
        || channels % 16 != 0
    {
        return false;
    }
    pool_area_inv(filter_h * filter_w).is_some()
}

// ---------------------------------------------------------------------------
// Elementwise gates (mirrors hematite-s3/src/elementwise.rs)
// ---------------------------------------------------------------------------

/// Mirrors `simd_eligible_add_sub` (elementwise.rs:698-709): the raw
/// TIE728 add/sub is bit-exact only for the identity quant-affine contract —
/// zero offsets, full-range clamp, `left_shift <= 0`, and every
/// `(multiplier, shift)` pair at the identity `(1<<30, 1)`.
#[inline]
pub(crate) fn simd_eligible_add_sub(params: &ElementwiseParams) -> bool {
    let identity = |m: i32, s: i32| m == 1 << 30 && s == 1;
    params.input1_offset == 0
        && params.input2_offset == 0
        && params.output_offset == 0
        && params.quantized_activation_min == i8::MIN as i32
        && params.quantized_activation_max == i8::MAX as i32
        && params.left_shift <= 0
        && identity(params.input1_multiplier, params.input1_shift)
        && identity(params.input2_multiplier, params.input2_shift)
        && identity(params.output_multiplier, params.output_shift)
}

/// Mirrors `simd_eligible_mul` (elementwise.rs:715-728): raw int8 product +
/// power-of-two-shift requantize — zero offsets, full-range clamp,
/// `output_multiplier == 1<<30`, `output_shift <= 1`. Returns the same
/// `Some(1 - output_shift)` the s3 gate returns.
#[inline]
pub(crate) fn simd_eligible_mul(params: &ElementwiseParams) -> Option<i32> {
    if params.input1_offset == 0
        && params.input2_offset == 0
        && params.output_offset == 0
        && params.quantized_activation_min == i8::MIN as i32
        && params.quantized_activation_max == i8::MAX as i32
        && params.output_multiplier == 1 << 30
        && params.output_shift <= 1
    {
        Some(1 - params.output_shift)
    } else {
        None
    }
}

/// Mirrors `simd_eligible_add_sub_widened` (elementwise.rs:757-759): the
/// T3.2 per-lane lane-model gate — true for EVERY param combination (the
/// 16-wide lane loop reproduces the exact scalar arithmetic). This is what
/// the standalone add/sub dispatch actually checks today (elementwise.rs:
/// 123) alongside `n >= 16` + alignment (runtime).
#[inline]
pub(crate) fn simd_eligible_add_sub_widened(_params: &ElementwiseParams) -> bool {
    true
}

/// Mirrors `simd_eligible_mul_widened` (elementwise.rs:764-766) — same
/// contract as the add/sub widened gate.
#[inline]
pub(crate) fn simd_eligible_mul_widened(_params: &ElementwiseParams) -> bool {
    true
}

// ---------------------------------------------------------------------------
// Activation gates (mirrors hematite-s3/src/activations.rs)
// ---------------------------------------------------------------------------

/// Mirrors `relu_simd_eligible_params` (activations.rs:37-42): the raw
/// TIE728 relu is bit-exact only for the identity contract — zero offsets,
/// `output_multiplier == 1<<30`, `output_shift == 1`. Runtime half: `n % 16
/// == 0` + `n >= 16` + alignment.
#[inline]
pub(crate) fn relu_simd_eligible_params(params: &ActivationParams<'_>) -> bool {
    params.input_offset == 0
        && params.output_offset == 0
        && params.output_multiplier == 1 << 30
        && params.output_shift == 1
}

/// Mirrors `relu6_simd_eligible_params` (activations.rs:292-294): the T3.2
/// widened per-lane model — true for EVERY param combination.
#[inline]
pub(crate) fn relu6_simd_eligible_params(_params: &ActivationParams<'_>) -> bool {
    true
}

/// Mirrors `hard_swish_simd_eligible_params` (activations.rs:300-302):
/// true for EVERY param combination (downgraded formula per-lane).
#[inline]
pub(crate) fn hard_swish_simd_eligible_params(_params: &ActivationParams<'_>) -> bool {
    true
}

// ---------------------------------------------------------------------------
// Softmax gate (mirrors hematite-s3/src/softmax.rs)
// ---------------------------------------------------------------------------

/// Mirrors the softmax SIMD gate (softmax.rs:383-387): `row_size >= 16`.
/// Runtime halves not visible here: 16B-aligned input pointer, 4B-aligned
/// scratch with `row_size * 4` bytes.
#[inline]
pub(crate) fn softmax_row_simd_eligible(row_size: i32) -> bool {
    row_size >= 16
}

// ---------------------------------------------------------------------------
// Fused chain gate (mirrors hematite-s3/src/fused.rs)
// ---------------------------------------------------------------------------

/// Per-step elementwise view — mirrors `step_elementwise_params`
/// (fused.rs:813-832): the step's fields as an `ElementwiseParams`.
#[inline]
fn chain_step_elementwise_view(
    step: &ElementwiseChainStep<'_>,
    num_elements: i32,
) -> ElementwiseParams {
    ElementwiseParams {
        num_elements,
        input1_offset: step.input1_offset,
        input2_offset: step.input2_offset,
        output_offset: step.output_offset,
        output_multiplier: step.output_multiplier,
        output_shift: step.output_shift,
        left_shift: step.left_shift,
        input1_multiplier: step.input1_multiplier,
        input1_shift: step.input1_shift,
        input2_multiplier: step.input2_multiplier,
        input2_shift: step.input2_shift,
        quantized_activation_min: step.quantized_activation_min,
        quantized_activation_max: step.quantized_activation_max,
    }
}

/// Per-step activation view — mirrors `step_activation_params`
/// (fused.rs:835-861): the step's fields as an `ActivationParams` (only the
/// relu-gate-relevant fields are non-zero).
#[inline]
fn chain_step_activation_view<'a>(step: &ElementwiseChainStep<'a>) -> ActivationParams<'a> {
    ActivationParams {
        input_offset: step.input1_offset,
        output_offset: step.output_offset,
        output_multiplier: step.output_multiplier,
        output_shift: step.output_shift,
        quantized_activation_min: step.quantized_activation_min,
        quantized_activation_max: step.quantized_activation_max,
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

/// Mirrors `chain_simd_eligible` (fused.rs:486-502): the fused-chain SIMD
/// path engages ONLY when every step is eligible under the per-step gates —
/// `simd_eligible_add_sub` / `simd_eligible_mul` (identity quant-affine) for
/// Add/Sub/Mul steps (which must carry their operand) and
/// `relu_simd_eligible_params` for Relu steps; Relu6/HardSwish steps have no
/// chain SIMD yet (T3.2) — any such step falls back. Empty chains are
/// ineligible. Runtime halves: `n % 16 == 0`, all pointers 16B-aligned.
#[inline]
pub(crate) fn chain_simd_eligible(params: &ElementwiseChainParams<'_>) -> bool {
    if params.steps.is_empty() {
        return false;
    }
    params.steps.iter().all(|step| match step.kind {
        ElementwiseKind::Add | ElementwiseKind::Sub => {
            step.operand.is_some()
                && simd_eligible_add_sub(&chain_step_elementwise_view(step, params.num_elements))
        }
        ElementwiseKind::Mul => {
            step.operand.is_some()
                && simd_eligible_mul(&chain_step_elementwise_view(step, params.num_elements)).is_some()
        }
        ElementwiseKind::Relu => relu_simd_eligible_params(&chain_step_activation_view(step)),
        ElementwiseKind::Relu6 | ElementwiseKind::HardSwish => false,
    })
}

// ---------------------------------------------------------------------------
// Fused pool-fold gate (mirrors hematite-s3/src/fused.rs)
// ---------------------------------------------------------------------------

/// Fold-elementwise view — mirrors `fold_elementwise_params`
/// (fused.rs:794-810): the absorbed MUL/SUB fold's params exactly as the
/// decomposition forwards them to the per-op kernels (full-range clamp).
#[inline]
fn fold_elementwise_view(fold: &PoolInputFold<'_>) -> ElementwiseParams {
    ElementwiseParams {
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
        quantized_activation_min: i8::MIN as i32,
        quantized_activation_max: i8::MAX as i32,
    }
}

/// Mirrors `fold_simd_exact` (fused.rs:677-684): the absorbed fold is
/// bit-exact under the per-op elementwise gates — MUL via `simd_eligible_mul`
/// (raw-product + power-of-two-shift), SUB via `simd_eligible_add_sub` (raw
/// int8 subtract). Any other fold falls back.
#[inline]
pub(crate) fn fold_simd_exact(fold: &PoolInputFold<'_>) -> bool {
    let ep = fold_elementwise_view(fold);
    match fold.builtin {
        18 /* MUL */ => simd_eligible_mul(&ep).is_some(),
        41 /* SUB */ => simd_eligible_add_sub(&ep),
        _ => false,
    }
}

/// Mirrors `fused_pool_fold_simd_eligible` (fused.rs:695-703): the anchor
/// pool passes the pool SIMD gate AND the fold (if any) is in the
/// provably-exact subset. `None` fold → only the pool gate applies.
#[inline]
pub(crate) fn fused_pool_fold_simd_eligible(params: &FoldedPoolParams<'_>) -> bool {
    if !simd_eligible_pool(&params.pool) {
        return false;
    }
    match &params.fold {
        None => true,
        Some(fold) => fold_simd_exact(fold),
    }
}

// ---------------------------------------------------------------------------
// Parity tests — mirror == s3 gate (verbatim transcription) over the spec
// corpus and widened grids. T1.4 scratch-parity discipline (0213f1b): the
// runtime side is canonical; a divergence is a MIRROR bug.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use hematite_core::op_params::{
        ActivationEpilogueParams, ComposedActivation, Conv2DParams, DepthwiseConv2DParams,
        FusedActivation, FullyConnectedParams, Padding, PoolKind, SoftmaxParams,
    };

    /// Verbatim transcriptions of the s3 gate fns at HEAD (4ecd1ac), the
    /// s3-side oracle of the parity tests below.
    ///
    /// The s3 gates are `pub(crate)`/private with no cross-crate surface
    /// (the `Prepared*` handles AND `is_simd()` with the device cfg, so on
    /// host they always return `false`), so the parity test cannot call them
    /// directly; these copies are kept byte-for-byte with the s3 source.
    /// They are independently pinned by s3's own in-crate gate-expectation
    /// tests (pool.rs:1854 `pool_simd_eligible_gate_expectations`,
    /// fused.rs:1499 `chain_simd_eligibility_gate_expectations`, fused.rs:
    /// 1695 `fused_pool_fold_simd_eligibility_gate_expectations`).
    ///
    /// When an s3 gate changes: update the transcription AND the mirror in
    /// the SAME commit (the parity test then still passes) — never leave
    /// them divergent.
    mod s3_ref {
        use super::super::*;

        /// accx.rs:64-66 @ 4ecd1ac.
        pub fn accx_eligible_1x1(input_c: usize, out_c: usize) -> bool {
            input_c >= 16 && input_c.is_multiple_of(16) && out_c >= 1
        }

        /// accx.rs:86-88 @ 4ecd1ac.
        pub fn accx_eligible_1x1_padded(input_c: usize, out_c: usize) -> bool {
            input_c >= 1 && out_c >= 1
        }

        /// accx.rs:97-99 @ 4ecd1ac.
        pub fn accx_eligible_3x3(input_c: usize, out_c: usize) -> bool {
            input_c >= 1 && out_c >= 1
        }

        /// accx.rs:114-121 @ 4ecd1ac.
        pub fn accx_eligible_depthwise_dm(
            input_c: usize,
            out_c: usize,
            depth_multiplier: i32,
        ) -> bool {
            let dm = depth_multiplier.max(1) as usize;
            input_c >= 1 && out_c >= 1 && out_c == input_c * dm
        }

        /// conv1x1.rs:214-224 @ 4ecd1ac.
        #[allow(clippy::too_many_arguments)]
        pub fn conv1x1_dispatch_eligible(
            input_c: usize,
            out_c: usize,
            stride_h: i32,
            stride_w: i32,
            dil_h: i32,
            dil_w: i32,
            in_h: usize,
            in_w: usize,
            out_h: usize,
            out_w: usize,
        ) -> bool {
            stride_h == 1
                && stride_w == 1
                && dil_h == 1
                && dil_w == 1
                && in_h == out_h
                && in_w == out_w
                && accx_eligible_1x1_padded(input_c, out_c)
        }

        /// conv3x3.rs:128-139 @ 4ecd1ac.
        pub fn conv3x3_dispatch_eligible(
            input_c: usize,
            out_c: usize,
            filter_h: i32,
            filter_w: i32,
            dil_h: i32,
            dil_w: i32,
            input_offset: i32,
        ) -> bool {
            if dil_h != 1
                || dil_w != 1
                || filter_h != 3
                || filter_w != 3
                || !accx_eligible_3x3(input_c, out_c)
            {
                return false;
            }
            if input_offset != 0 && input_offset.abs() > 127 {
                return false;
            }
            true
        }

        /// depthwise.rs:310-324 @ 4ecd1ac.
        #[allow(clippy::too_many_arguments)]
        #[allow(clippy::manual_range_contains)]
        pub fn depthwise_dispatch_eligible(
            input_c: usize,
            out_c: usize,
            depth_multiplier: i32,
            filter_h: i32,
            filter_w: i32,
            dil_h: i32,
            dil_w: i32,
            input_offset: i32,
        ) -> bool {
            if dil_h != 1
                || dil_w != 1
                || filter_h < 1
                || filter_w < 1
                || !accx_eligible_depthwise_dm(input_c, out_c, depth_multiplier)
            {
                return false;
            }
            if input_offset != 0 && (input_offset < -127 || input_offset > 128) {
                return false;
            }
            true
        }

        /// pool.rs:1130-1133 @ 4ecd1ac.
        fn round_recip(area: i32, shift: i32) -> i32 {
            let num = 1i64 << shift;
            ((num + area as i64 / 2) / area as i64) as i32
        }

        /// pool.rs:1142-1153 @ 4ecd1ac.
        fn pool_area_inv(area: i32) -> Option<(i32, [i8; 16])> {
            let mut shift = 8;
            let mut inv = round_recip(area, shift);
            while inv > 127 && shift < 24 {
                shift += 1;
                inv = round_recip(area, shift);
            }
            if !(1..=127).contains(&inv) {
                return None;
            }
            Some((shift, [inv as i8; 16]))
        }

        /// pool.rs:1171-1208 @ 4ecd1ac (as a bool: `Some` ⇔ `true`; the cfg
        /// content is selector-irrelevant).
        pub fn simd_eligible_pool(params: &PoolParams) -> bool {
            let input_h = params.input_shape[1];
            let input_w = params.input_shape[2];
            let channels = params.input_shape[3];
            let filter_h = params.filter_height;
            let filter_w = params.filter_width;
            let out_h = params.output_shape[1];
            let out_w = params.output_shape[2];

            let pad_total_h = (out_h - 1) * params.stride_height + filter_h - input_h;
            let pad_total_w = (out_w - 1) * params.stride_width + filter_w - input_w;

            if filter_h < 1
                || filter_w < 1
                || params.stride_height < 1
                || params.stride_width < 1
                || pad_total_h > 0
                || pad_total_w > 0
                || channels <= 0
                || channels % 16 != 0
            {
                return false;
            }
            pool_area_inv(filter_h * filter_w).is_some()
        }

        /// elementwise.rs:698-709 @ 4ecd1ac.
        pub fn simd_eligible_add_sub(params: &ElementwiseParams) -> bool {
            let identity = |m: i32, s: i32| m == 1 << 30 && s == 1;
            params.input1_offset == 0
                && params.input2_offset == 0
                && params.output_offset == 0
                && params.quantized_activation_min == i8::MIN as i32
                && params.quantized_activation_max == i8::MAX as i32
                && params.left_shift <= 0
                && identity(params.input1_multiplier, params.input1_shift)
                && identity(params.input2_multiplier, params.input2_shift)
                && identity(params.output_multiplier, params.output_shift)
        }

        /// elementwise.rs:715-728 @ 4ecd1ac.
        pub fn simd_eligible_mul(params: &ElementwiseParams) -> Option<i32> {
            if params.input1_offset == 0
                && params.input2_offset == 0
                && params.output_offset == 0
                && params.quantized_activation_min == i8::MIN as i32
                && params.quantized_activation_max == i8::MAX as i32
                && params.output_multiplier == 1 << 30
                && params.output_shift <= 1
            {
                Some(1 - params.output_shift)
            } else {
                None
            }
        }

        /// elementwise.rs:757-759 @ 4ecd1ac.
        pub fn simd_eligible_add_sub_widened(_params: &ElementwiseParams) -> bool {
            true
        }

        /// elementwise.rs:764-766 @ 4ecd1ac.
        pub fn simd_eligible_mul_widened(_params: &ElementwiseParams) -> bool {
            true
        }

        /// activations.rs:37-42 @ 4ecd1ac.
        pub fn relu_simd_eligible_params(params: &ActivationParams<'_>) -> bool {
            params.input_offset == 0
                && params.output_offset == 0
                && params.output_multiplier == 1 << 30
                && params.output_shift == 1
        }

        /// activations.rs:292-294 @ 4ecd1ac.
        pub fn relu6_simd_eligible_params(_params: &ActivationParams<'_>) -> bool {
            true
        }

        /// activations.rs:300-302 @ 4ecd1ac.
        pub fn hard_swish_simd_eligible_params(_params: &ActivationParams<'_>) -> bool {
            true
        }

        /// softmax.rs:383-387 @ 4ecd1ac.
        pub fn softmax_row_simd_eligible(row_size: i32) -> bool {
            row_size >= 16
        }

        /// fused.rs:813-832 @ 4ecd1ac.
        pub fn step_elementwise_params(
            step: &ElementwiseChainStep<'_>,
            num_elements: i32,
        ) -> ElementwiseParams {
            ElementwiseParams {
                num_elements,
                input1_offset: step.input1_offset,
                input2_offset: step.input2_offset,
                output_offset: step.output_offset,
                output_multiplier: step.output_multiplier,
                output_shift: step.output_shift,
                left_shift: step.left_shift,
                input1_multiplier: step.input1_multiplier,
                input1_shift: step.input1_shift,
                input2_multiplier: step.input2_multiplier,
                input2_shift: step.input2_shift,
                quantized_activation_min: step.quantized_activation_min,
                quantized_activation_max: step.quantized_activation_max,
            }
        }

        /// fused.rs:835-861 @ 4ecd1ac.
        pub fn step_activation_params<'a>(step: &ElementwiseChainStep<'_>) -> ActivationParams<'a> {
            ActivationParams {
                input_offset: step.input1_offset,
                output_offset: step.output_offset,
                output_multiplier: step.output_multiplier,
                output_shift: step.output_shift,
                quantized_activation_min: step.quantized_activation_min,
                quantized_activation_max: step.quantized_activation_max,
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

        /// fused.rs:486-502 @ 4ecd1ac.
        pub fn chain_simd_eligible(params: &ElementwiseChainParams<'_>) -> bool {
            if params.steps.is_empty() {
                return false;
            }
            params.steps.iter().all(|step| match step.kind {
                ElementwiseKind::Add | ElementwiseKind::Sub => {
                    step.operand.is_some()
                        && simd_eligible_add_sub(&step_elementwise_params(
                            step,
                            params.num_elements,
                        ))
                }
                ElementwiseKind::Mul => {
                    step.operand.is_some()
                        && simd_eligible_mul(&step_elementwise_params(
                            step,
                            params.num_elements,
                        ))
                        .is_some()
                }
                ElementwiseKind::Relu => relu_simd_eligible_params(&step_activation_params(step)),
                ElementwiseKind::Relu6 | ElementwiseKind::HardSwish => false,
            })
        }

        /// fused.rs:794-810 @ 4ecd1ac.
        pub fn fold_elementwise_params(fold: &PoolInputFold<'_>) -> ElementwiseParams {
            ElementwiseParams {
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
                quantized_activation_min: i8::MIN as i32,
                quantized_activation_max: i8::MAX as i32,
            }
        }

        /// fused.rs:677-684 @ 4ecd1ac.
        pub fn fold_simd_exact(fold: &PoolInputFold<'_>) -> bool {
            let ep = fold_elementwise_params(fold);
            match fold.builtin {
                18 /* MUL */ => simd_eligible_mul(&ep).is_some(),
                41 /* SUB */ => simd_eligible_add_sub(&ep),
                _ => false,
            }
        }

        /// fused.rs:695-703 @ 4ecd1ac.
        pub fn fused_pool_fold_simd_eligible(params: &FoldedPoolParams<'_>) -> bool {
            if !simd_eligible_pool(&params.pool) {
                return false;
            }
            match &params.fold {
                None => true,
                Some(fold) => fold_simd_exact(fold),
            }
        }
    }

    // ------------------------------------------------------------------
    // Param builders (grids)
    // ------------------------------------------------------------------

    fn tflite_out_dim(in_dim: i32, filter: i32, stride: i32, padding: Padding) -> i32 {
        match padding {
            Padding::Same => (in_dim + stride - 1) / stride,
            Padding::Valid => {
                if in_dim >= filter {
                    (in_dim - filter) / stride + 1
                } else {
                    0
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn conv_params(
        fh: i32,
        fw: i32,
        in_c: i32,
        out_c: i32,
        in_h: i32,
        in_w: i32,
        out_h: i32,
        out_w: i32,
        stride: i32,
        padding: Padding,
        offset: i32,
    ) -> Conv2DParams<'static> {
        Conv2DParams {
            input_shape: [1, in_h, in_w, in_c],
            filter_shape: [out_c, fh, fw, in_c],
            output_shape: [1, out_h, out_w, out_c],
            padding,
            stride_width: stride,
            stride_height: stride,
            dilation_width_factor: 1,
            dilation_height_factor: 1,
            input_offset: offset,
            weights_offset: 0,
            output_offset: 0,
            output_multiplier_per_channel: &[],
            output_shift_per_channel: &[],
            quantized_activation_min: -128,
            quantized_activation_max: 127,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn depthwise_params(
        dm: i32,
        in_c: i32,
        fh: i32,
        fw: i32,
        in_h: i32,
        in_w: i32,
        stride: i32,
        padding: Padding,
        offset: i32,
    ) -> DepthwiseConv2DParams<'static> {
        let out_c = in_c * dm;
        DepthwiseConv2DParams {
            input_shape: [1, in_h, in_w, in_c],
            filter_shape: [1, fh, fw, out_c],
            output_shape: [
                1,
                tflite_out_dim(in_h, fh, stride, padding),
                tflite_out_dim(in_w, fw, stride, padding),
                out_c,
            ],
            padding,
            stride_width: stride,
            stride_height: stride,
            dilation_width_factor: 1,
            dilation_height_factor: 1,
            depth_multiplier: dm,
            input_offset: offset,
            weights_offset: 0,
            output_offset: 0,
            output_multiplier_per_channel: &[],
            output_shift_per_channel: &[],
            quantized_activation_min: -128,
            quantized_activation_max: 127,
        }
    }

    fn fc_params(input_dim: i32, output_dim: i32, input_offset: i32) -> FullyConnectedParams<'static> {
        FullyConnectedParams {
            input_dim,
            output_dim,
            input_offset,
            weights_offset: 0,
            output_offset: 0,
            output_multiplier_per_channel: &[],
            output_shift_per_channel: &[],
            quantized_activation_min: -128,
            quantized_activation_max: 127,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn pool_params(
        _kind: &str,
        fh: i32,
        fw: i32,
        stride: i32,
        in_h: i32,
        in_w: i32,
        ch: i32,
        padding: Padding,
    ) -> PoolParams {
        let out_h = tflite_out_dim(in_h, fh, stride, padding);
        let out_w = tflite_out_dim(in_w, fw, stride, padding);
        PoolParams {
            input_shape: [1, in_h, in_w, ch],
            output_shape: [1, out_h, out_w, ch],
            filter_width: fw,
            filter_height: fh,
            stride_width: stride,
            stride_height: stride,
            padding,
            activation: FusedActivation::None,
            quantized_activation_min: i8::MIN as i32,
            quantized_activation_max: i8::MAX as i32,
        }
    }

    fn elementwise_base(num_elements: i32) -> ElementwiseParams {
        ElementwiseParams {
            num_elements,
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
        }
    }

    fn relu_base() -> ActivationParams<'static> {
        ActivationParams {
            input_offset: 0,
            output_offset: 0,
            output_multiplier: 1 << 30,
            output_shift: 1,
            quantized_activation_min: i8::MIN as i32,
            quantized_activation_max: i8::MAX as i32,
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

    /// Identity-quant-affine chain step (Add/Sub/Mul with an operand).
    fn identity_step(kind: ElementwiseKind, operand: bool) -> ElementwiseChainStep<'static> {
        ElementwiseChainStep {
            kind,
            operand: if operand { Some(&[]) } else { None },
            input1_offset: 0,
            input2_offset: 0,
            output_offset: 0,
            output_multiplier: 1 << 30,
            output_shift: if kind == ElementwiseKind::Mul { 0 } else { 1 },
            left_shift: 0,
            input1_multiplier: 1 << 30,
            input1_shift: 1,
            input2_multiplier: 1 << 30,
            input2_shift: 1,
            quantized_activation_min: i8::MIN as i32,
            quantized_activation_max: i8::MAX as i32,
        }
    }

    fn relu_step() -> ElementwiseChainStep<'static> {
        ElementwiseChainStep {
            kind: ElementwiseKind::Relu,
            operand: None,
            input1_offset: 0,
            input2_offset: 0,
            output_offset: 0,
            output_multiplier: 1 << 30,
            output_shift: 1,
            left_shift: 0,
            input1_multiplier: 0,
            input1_shift: 0,
            input2_multiplier: 0,
            input2_shift: 0,
            quantized_activation_min: i8::MIN as i32,
            quantized_activation_max: i8::MAX as i32,
        }
    }

    fn pool_fold(
        builtin: i32,
        input_zp: i64,
        operand_zp: i64,
        output_zp: i64,
        om: i32,
        os: i32,
        left_shift: i32,
    ) -> PoolInputFold<'static> {
        PoolInputFold {
            builtin,
            operand_data: &[],
            operand_zero_point: operand_zp,
            input_zero_point: input_zp,
            output_zero_point: output_zp,
            folded_scale: 1.0,
            left_shift,
            output_multiplier: om,
            output_shift: os,
            input1_multiplier: 1 << 30,
            input1_shift: 1,
            input2_multiplier: 1 << 30,
            input2_shift: 1,
            num_elements: 256,
        }
    }

    fn folded_pool_params(
        pool: PoolParams,
        fold: Option<PoolInputFold<'static>>,
    ) -> FoldedPoolParams<'static> {
        FoldedPoolParams {
            pool,
            pool_kind: PoolKind::Average,
            fold,
            activation: ActivationEpilogueParams {
                kind: ComposedActivation::None,
                input_offset: 0,
                output_offset: 0,
                output_multiplier: 1 << 30,
                output_shift: 1,
                quantized_activation_min: i8::MIN as i32,
                quantized_activation_max: i8::MAX as i32,
            },
        }
    }

    // ------------------------------------------------------------------
    // Per-family parity checks
    // ------------------------------------------------------------------

    fn check_conv(p: &Conv2DParams<'_>, checked: &mut usize, label: &str) {
        let in_c = p.input_shape[3].max(0) as usize;
        let out_c = p.output_shape[3].max(0) as usize;
        let (in_h, in_w, out_h, out_w) = (
            p.input_shape[1].max(0) as usize,
            p.input_shape[2].max(0) as usize,
            p.output_shape[1].max(0) as usize,
            p.output_shape[2].max(0) as usize,
        );
        let (fh, fw) = (p.filter_shape[1], p.filter_shape[2]);

        // Raw accx gates.
        let pairs = [
            ("accx_eligible_1x1", super::accx_eligible_1x1(in_c, out_c), s3_ref::accx_eligible_1x1(in_c, out_c)),
            ("accx_eligible_1x1_padded", super::accx_eligible_1x1_padded(in_c, out_c), s3_ref::accx_eligible_1x1_padded(in_c, out_c)),
            ("accx_eligible_3x3", super::accx_eligible_3x3(in_c, out_c), s3_ref::accx_eligible_3x3(in_c, out_c)),
        ];
        for (name, mirror, oracle) in pairs {
            assert_eq!(mirror, oracle, "conv {label}: {name} mirror {mirror} != s3 {oracle}");
        }

        // Dispatch-level predicate — the s3 conv dispatch routes by filter:
        // 1x1 → conv1x1_accx_dispatch, everything else → conv3x3_accx_dispatch
        // (fused.rs:546-568).
        let (mirror, oracle) = if fh == 1 && fw == 1 {
            (
                super::conv1x1_dispatch_eligible(
                    in_c, out_c, p.stride_height, p.stride_width,
                    p.dilation_height_factor, p.dilation_width_factor,
                    in_h, in_w, out_h, out_w,
                ),
                s3_ref::conv1x1_dispatch_eligible(
                    in_c, out_c, p.stride_height, p.stride_width,
                    p.dilation_height_factor, p.dilation_width_factor,
                    in_h, in_w, out_h, out_w,
                ),
            )
        } else {
            (
                super::conv3x3_dispatch_eligible(
                    in_c, out_c, fh, fw,
                    p.dilation_height_factor, p.dilation_width_factor, p.input_offset,
                ),
                s3_ref::conv3x3_dispatch_eligible(
                    in_c, out_c, fh, fw,
                    p.dilation_height_factor, p.dilation_width_factor, p.input_offset,
                ),
            )
        };
        assert_eq!(
            mirror, oracle,
            "conv {label}: dispatch mirror {mirror} != s3 dispatch {oracle} (in_c={in_c} out_c={out_c} f={fh}x{fw} off={})",
            p.input_offset
        );
        *checked += 1;
    }

    fn check_depthwise(p: &DepthwiseConv2DParams<'_>, checked: &mut usize, label: &str) {
        let in_c = p.input_shape[3].max(0) as usize;
        let out_c = p.output_shape[3].max(0) as usize;
        let (fh, fw) = (p.filter_shape[1], p.filter_shape[2]);
        let mirror = super::depthwise_dispatch_eligible(
            in_c, out_c, p.depth_multiplier, fh, fw,
            p.dilation_height_factor, p.dilation_width_factor, p.input_offset,
        );
        let oracle = s3_ref::depthwise_dispatch_eligible(
            in_c, out_c, p.depth_multiplier, fh, fw,
            p.dilation_height_factor, p.dilation_width_factor, p.input_offset,
        );
        assert_eq!(
            mirror, oracle,
            "depthwise {label}: mirror {mirror} != s3 {oracle} (in_c={in_c} out_c={out_c} dm={} f={fh}x{fw} off={})",
            p.depth_multiplier, p.input_offset
        );
        *checked += 1;
    }

    fn check_fc(p: &FullyConnectedParams<'_>, checked: &mut usize, label: &str) {
        let idim = p.input_dim.max(0) as usize;
        let odim = p.output_dim.max(0) as usize;
        let mirror = super::fc_dispatch_eligible(idim, odim);
        let oracle = s3_ref::accx_eligible_1x1_padded(idim, odim);
        assert_eq!(
            mirror, oracle,
            "fc {label}: mirror {mirror} != s3 {oracle} (in={idim} out={odim} off={})",
            p.input_offset
        );
        *checked += 1;
    }

    fn check_pool(p: &PoolParams, checked: &mut usize, label: &str) {
        let mirror = super::simd_eligible_pool(p);
        let oracle = s3_ref::simd_eligible_pool(p);
        assert_eq!(
            mirror, oracle,
            "pool {label}: mirror {mirror} != s3 {oracle} (in={:?} out={:?} f={}x{} s={}/{})",
            p.input_shape, p.output_shape, p.filter_height, p.filter_width,
            p.stride_height, p.stride_width
        );
        *checked += 1;
    }

    fn check_activation(p: &ActivationParams<'_>, checked: &mut usize, label: &str) {
        let mirror = super::relu_simd_eligible_params(p);
        let oracle = s3_ref::relu_simd_eligible_params(p);
        assert_eq!(mirror, oracle, "relu {label}: mirror {mirror} != s3 {oracle}");
        // relu6 / hard_swish gates (always true) must agree too.
        assert_eq!(
            super::relu6_simd_eligible_params(p),
            s3_ref::relu6_simd_eligible_params(p),
            "relu6 {label}: widened gate mismatch"
        );
        assert_eq!(
            super::hard_swish_simd_eligible_params(p),
            s3_ref::hard_swish_simd_eligible_params(p),
            "hard_swish {label}: widened gate mismatch"
        );
        *checked += 1;
    }

    fn check_elementwise(p: &ElementwiseParams, checked: &mut usize, label: &str) {
        // Both gates are asserted for every row — the corpus rows include
        // identity AND non-identity (T3.2) contracts.
        let m = super::simd_eligible_add_sub(p);
        let o = s3_ref::simd_eligible_add_sub(p);
        assert_eq!(m, o, "add_sub {label}: mirror {m} != s3 {o} (off {}/{}/{} ls {} om {} os {})",
            p.input1_offset, p.input2_offset, p.output_offset, p.left_shift,
            p.output_multiplier, p.output_shift);
        let m = super::simd_eligible_mul(p);
        let o = s3_ref::simd_eligible_mul(p);
        assert_eq!(m, o, "mul {label}: mirror {m:?} != s3 {o:?}");
        assert_eq!(
            super::simd_eligible_add_sub_widened(p),
            s3_ref::simd_eligible_add_sub_widened(p),
            "add_sub_widened {label}: mismatch"
        );
        assert_eq!(
            super::simd_eligible_mul_widened(p),
            s3_ref::simd_eligible_mul_widened(p),
            "mul_widened {label}: mismatch"
        );
        *checked += 1;
    }

    fn check_softmax(p: &SoftmaxParams, checked: &mut usize, label: &str) {
        let mirror = super::softmax_row_simd_eligible(p.row_size);
        let oracle = s3_ref::softmax_row_simd_eligible(p.row_size);
        assert_eq!(
            mirror, oracle,
            "softmax {label}: mirror {mirror} != s3 {oracle} (row_size {})",
            p.row_size
        );
        *checked += 1;
    }

    fn check_chain(p: &ElementwiseChainParams<'_>, checked: &mut usize, label: &str) {
        let mirror = super::chain_simd_eligible(p);
        let oracle = s3_ref::chain_simd_eligible(p);
        assert_eq!(
            mirror, oracle,
            "chain {label}: mirror {mirror} != s3 {oracle} ({} steps)",
            p.steps.len()
        );
        *checked += 1;
    }

    fn check_fused_pool(p: &FoldedPoolParams<'_>, checked: &mut usize, label: &str) {
        let mirror = super::fused_pool_fold_simd_eligible(p);
        let oracle = s3_ref::fused_pool_fold_simd_eligible(p);
        assert_eq!(
            mirror, oracle,
            "pool-fold {label}: mirror {mirror} != s3 {oracle} (fold {:?})",
            p.fold.as_ref().map(|f| f.builtin)
        );
        *checked += 1;
    }

    // ------------------------------------------------------------------
    // (a) Spec-corpus sweep
    // ------------------------------------------------------------------

    #[test]
    fn parity_spec_corpus() {
        use hematite_benchmarks::spec::kernel_specs;
        use hematite_benchmarks::spec::KernelParams;

        let mut checked = 0;
        let (mut n_conv, mut n_dw, mut n_fc, mut n_pool) = (0, 0, 0, 0);
        let (mut n_act, mut n_ew, mut n_sm, mut n_chain, mut n_pf, mut n_fused_conv) = (0, 0, 0, 0, 0, 0);

        for spec in kernel_specs() {
            match &spec.params {
                KernelParams::Conv(p) => {
                    check_conv(p, &mut checked, spec.name);
                    n_conv += 1;
                }
                KernelParams::Depthwise(p) => {
                    check_depthwise(p, &mut checked, spec.name);
                    n_dw += 1;
                }
                KernelParams::Fc(p) => {
                    check_fc(p, &mut checked, spec.name);
                    n_fc += 1;
                }
                KernelParams::Softmax(p) => {
                    check_softmax(p, &mut checked, spec.name);
                    n_sm += 1;
                }
                KernelParams::Pool(p) => {
                    check_pool(p, &mut checked, spec.name);
                    n_pool += 1;
                }
                KernelParams::Activation(p) => {
                    check_activation(p, &mut checked, spec.name);
                    n_act += 1;
                }
                KernelParams::Elementwise(p) => {
                    check_elementwise(p, &mut checked, spec.name);
                    n_ew += 1;
                }
                KernelParams::FusedChain(p) => {
                    check_chain(p, &mut checked, spec.name);
                    n_chain += 1;
                }
                KernelParams::FusedPool(p) => {
                    check_fused_pool(p, &mut checked, spec.name);
                    n_pf += 1;
                }
                KernelParams::FusedConv(p) => {
                    // The composed conv runs the anchor conv's own dispatch
                    // (fused.rs:546-568), so the anchor conv's dispatch gate
                    // applies unchanged.
                    check_conv(&p.conv, &mut checked, spec.name);
                    n_fused_conv += 1;
                }
                KernelParams::Reduce(_) => {
                    // MEAN has no SIMD-eligibility gate in the selector's
                    // scope (T3.4 looped accumulation; engagement is a
                    // positions/in_c shape range, not a gate fn).
                }
            }
        }

        assert!(
            n_conv >= 13 && n_dw >= 12 && n_fc >= 6 && n_sm >= 1 && n_pool >= 15 && n_act >= 3
                && n_ew >= 5 && n_chain >= 1 && n_pf >= 1 && n_fused_conv >= 2,
            "eligibility corpus shrank unexpectedly: conv={n_conv} dw={n_dw} fc={n_fc} sm={n_sm} \
             pool={n_pool} act={n_act} ew={n_ew} chain={n_chain} poolfold={n_pf} fusedconv={n_fused_conv}"
        );
        assert!(checked >= 55, "eligibility corpus sweep too small ({checked})");
    }

    // ------------------------------------------------------------------
    // (b) Widened grids
    // ------------------------------------------------------------------

    #[test]
    fn parity_widened_grids() {
        let mut checked = 0usize;

        // conv1x1: out_c × spatial × in_c × stride × offset — the padded
        // small in_c family (1/3/8/15/17), the %16 no-pad family, stride 1
        // (eligible when in==out) and stride 2 (ineligible).
        for &out_c in &[1, 16, 64] {
            for &spatial in &[1, 14, 56] {
                for &in_c in &[1, 3, 8, 15, 16, 17, 32, 64, 128] {
                    for &stride in &[1, 2] {
                        for &offset in &[0, 5, 128] {
                            let out_h = tflite_out_dim(spatial, 1, stride, Padding::Same);
                            let p = conv_params(1, 1, in_c, out_c, spatial, spatial, out_h, out_h, stride, Padding::Same, offset);
                            check_conv(&p, &mut checked, "grid-conv1x1");
                        }
                    }
                }
            }
        }

        // conv3x3: in_c × out_c × hw × pad × stride × offset — offsets 127
        // (eligible) and 128 (rejected by the Phase-C fold bound).
        for &in_c in &[3, 16, 32, 64] {
            for &out_c in &[16, 32, 64] {
                for &hw in &[8, 16, 32] {
                    for &padding in &[Padding::Valid, Padding::Same] {
                        for &stride in &[1, 2] {
                            for &offset in &[0, 5, 127, 128] {
                                let out_h = tflite_out_dim(hw, 3, stride, padding);
                                let p = conv_params(3, 3, in_c, out_c, hw, hw, out_h, out_h, stride, padding, offset);
                                check_conv(&p, &mut checked, "grid-conv3x3");
                            }
                        }
                    }
                }
            }
        }

        // depthwise: dm 1/2/8 × in_c × filter (3x3 + 10x8 anytap) × spatial ×
        // stride × offset — offsets 128 (accepted) and 129 (rejected).
        for &dm in &[1, 2, 8] {
            for &in_c in &[1, 3, 8, 16, 32] {
                for &(fh, fw) in &[(3, 3), (10, 8)] {
                    for &spatial in &[12, 49] {
                        for &stride in &[1, 2] {
                            for &offset in &[0, 3, 128, 129] {
                                let p = depthwise_params(dm, in_c, fh, fw, spatial, spatial, stride, Padding::Same, offset);
                                check_depthwise(&p, &mut checked, "grid-depthwise");
                            }
                        }
                    }
                }
            }
        }

        // fc: input_dim × output_dim × offset (pad16 small shapes + %16).
        for &input_dim in &[1, 3, 8, 15, 16, 17, 32, 640] {
            for &output_dim in &[1, 16, 128] {
                for &offset in &[0, 5, 128] {
                    let p = fc_params(input_dim, output_dim, offset);
                    check_fc(&p, &mut checked, "grid-fc");
                }
            }
        }

        // pool: kind × filter (incl. 1x1/2x1 whose reciprocal cannot fit an
        // i8 lane) × stride × pad × ch × hw — the SAME rows where
        // pad_total > 0 / partial windows appear must say false.
        for &kind in &["avg", "max"] {
            for &(fh, fw) in &[(2, 2), (3, 3), (5, 5), (1, 1), (2, 1)] {
                for &stride in &[1, 2] {
                    for &padding in &[Padding::Valid, Padding::Same] {
                        for &ch in &[8, 16, 32] {
                            for &hw in &[8, 16] {
                                let p = pool_params(kind, fh, fw, stride, hw, hw, ch, padding);
                                check_pool(&p, &mut checked, "grid-pool");
                            }
                        }
                    }
                }
            }
        }

        // elementwise: identity + non-identity offsets/multipliers/shifts/
        // left_shift/clamp mutations — both gates on every variant.
        type EwMut = Box<dyn Fn(&mut ElementwiseParams)>;
        let mutations: Vec<(&str, EwMut)> = vec![
            ("identity", Box::new(|_| {})),
            ("in1_off", Box::new(|p| p.input1_offset = 5)),
            ("in2_off", Box::new(|p| p.input2_offset = -3)),
            ("out_off", Box::new(|p| p.output_offset = 3)),
            ("left_shift_20", Box::new(|p| p.left_shift = 20)),
            ("in1_mult", Box::new(|p| p.input1_multiplier = 1 << 29)),
            ("out_mult", Box::new(|p| p.output_multiplier = 1 << 29)),
            ("out_shift_2", Box::new(|p| p.output_shift = 2)),
            ("out_shift_0", Box::new(|p| p.output_shift = 0)),
            ("relu_clamp", Box::new(|p| {
                p.quantized_activation_min = 0;
                p.quantized_activation_max = 127;
            })),
        ];
        for (name, mutate) in mutations {
            let mut p = elementwise_base(256);
            mutate(&mut p);
            check_elementwise(&p, &mut checked, &format!("grid-ew-{name}"));
        }

        // relu: identity + each identity-field mutated.
        type ReluMut = Box<dyn Fn(&mut ActivationParams<'static>)>;
        let relu_cases: Vec<(&str, ReluMut)> = vec![
            ("identity", Box::new(|_| {})),
            ("in_off", Box::new(|p| p.input_offset = 1)),
            ("out_off", Box::new(|p| p.output_offset = 1)),
            ("out_mult", Box::new(|p| p.output_multiplier = 1 << 29)),
            ("out_shift", Box::new(|p| p.output_shift = 2)),
            ("all_off", Box::new(|p| {
                p.input_offset = 128;
                p.output_offset = -128;
                p.output_multiplier = 1 << 30;
                p.output_shift = 1;
            })),
        ];
        for (name, mutate) in relu_cases {
            let mut p = relu_base();
            mutate(&mut p);
            check_activation(&p, &mut checked, &format!("grid-relu-{name}"));
        }

        // softmax: row sizes across the 16 boundary.
        for &row_size in &[1, 4, 8, 15, 16, 17, 32, 1960] {
            let p = SoftmaxParams {
                num_rows: 1,
                row_size,
                input_multiplier: 0,
                input_left_shift: 0,
                diff_min: 0,
                input_offset: 0,
                output_offset: -128,
                quantized_activation_min: -128,
                quantized_activation_max: 127,
            };
            check_softmax(&p, &mut checked, "grid-softmax");
        }

        // fused chains: identity / mixed / ineligible step mixes.
        let chains: Vec<(&str, Vec<ElementwiseChainStep<'static>>)> = vec![
            ("identity-add", vec![identity_step(ElementwiseKind::Add, true)]),
            (
                "identity-add-relu",
                vec![identity_step(ElementwiseKind::Add, true), relu_step()],
            ),
            (
                "identity-add-mul-relu",
                vec![
                    identity_step(ElementwiseKind::Add, true),
                    identity_step(ElementwiseKind::Mul, true),
                    relu_step(),
                ],
            ),
            ("identity-sub", vec![identity_step(ElementwiseKind::Sub, true)]),
            (
                "identity-mul-shift1",
                vec![{
                    let mut s = identity_step(ElementwiseKind::Mul, true);
                    s.output_shift = 1;
                    s
                }],
            ),
            (
                "mul-shift2",
                vec![{
                    let mut s = identity_step(ElementwiseKind::Mul, true);
                    s.output_shift = 2;
                    s
                }],
            ),
            ("relu6-step", vec![{
                let mut s = relu_step();
                s.kind = ElementwiseKind::Relu6;
                s
            }]),
            ("hard-swish-step", vec![{
                let mut s = relu_step();
                s.kind = ElementwiseKind::HardSwish;
                s
            }]),
            (
                "non-identity-add",
                vec![{
                    let mut s = identity_step(ElementwiseKind::Add, true);
                    s.input1_offset = 5;
                    s
                }],
            ),
            (
                "non-identity-relu",
                vec![{
                    let mut s = relu_step();
                    s.input1_offset = 3;
                    s
                }],
            ),
            ("operand-less-add", vec![identity_step(ElementwiseKind::Add, false)]),
            ("empty", vec![]),
            (
                "add-left-shift-20",
                vec![{
                    let mut s = identity_step(ElementwiseKind::Add, true);
                    s.left_shift = 20;
                    s
                }],
            ),
            (
                "non-identity-mul",
                vec![{
                    let mut s = identity_step(ElementwiseKind::Mul, true);
                    s.output_multiplier = 1 << 29;
                    s
                }],
            ),
        ];
        for (name, steps) in chains {
            let p = ElementwiseChainParams {
                num_elements: 256,
                steps: &steps,
            };
            check_chain(&p, &mut checked, &format!("grid-chain-{name}"));
        }

        // fused pool-fold: eligible/ineligible pools × fold presence/kind.
        let pools: Vec<(&str, PoolParams)> = vec![
            // 2×2/stride-2 SAME on even input → pad_total == 0 → eligible.
            ("even-2x2-s2", pool_params("avg", 2, 2, 2, 8, 8, 16, Padding::Same)),
            // 3×3 VALID → pad_total == 0, area 9 → eligible.
            ("valid-3x3", pool_params("avg", 3, 3, 1, 8, 8, 16, Padding::Valid)),
            // 2×2/stride-1 SAME → pad_total 1 → ineligible.
            ("asym-2x2-s1", pool_params("max", 2, 2, 1, 8, 8, 16, Padding::Same)),
            // channels % 16 != 0 → ineligible.
            ("c8-pool", pool_params("avg", 2, 2, 2, 8, 8, 8, Padding::Valid)),
        ];
        let folds: Vec<(&str, Option<PoolInputFold<'static>>)> = vec![
            ("no-fold", None),
            ("mul-identity", Some(pool_fold(18, 0, 0, 0, 1 << 30, 0, 0))),
            ("mul-nonidentity", Some(pool_fold(18, 0, 0, 0, 1 << 29, 0, 0))),
            ("sub-identity", Some(pool_fold(41, 0, 0, 0, 1 << 30, 1, 0))),
            ("sub-nonidentity", Some(pool_fold(41, 5, 0, 0, 1 << 30, 1, 0))),
            ("builtin-other", Some(pool_fold(0, 0, 0, 0, 1 << 30, 1, 0))),
        ];
        for (pname, pool) in &pools {
            for (fname, fold) in &folds {
                let p = folded_pool_params(pool.clone(), fold.clone());
                check_fused_pool(&p, &mut checked, &format!("grid-poolfold-{pname}-{fname}"));
            }
        }

        assert!(
            checked >= 1800,
            "widened eligibility grid did not expand ({checked} shapes)"
        );
    }

    // ------------------------------------------------------------------
    // Pool-specific boundary pin (independent of the corpus) — the
    // widened-gate rows whose exact s3 expectations are pinned in-crate at
    // pool.rs:1854 must hold for the mirror too.
    // ------------------------------------------------------------------

    #[test]
    fn pool_mirror_matches_s3_gate_expectations() {
        let base = |fh: i32, fw: i32, sh: i32, sw: i32, ih: i32, iw: i32, c: i32| {
            let (oh, ow) = ((ih - fh) / sh + 1, (iw - fw) / sw + 1);
            PoolParams {
                input_shape: [1, ih, iw, c],
                output_shape: [1, oh, ow, c],
                filter_width: fw,
                filter_height: fh,
                stride_width: sw,
                stride_height: sh,
                padding: Padding::Valid,
                activation: FusedActivation::None,
                quantized_activation_min: i8::MIN as i32,
                quantized_activation_max: i8::MAX as i32,
            }
        };
        let expect = |p: &PoolParams, want: bool, what: &str| {
            assert_eq!(
                super::simd_eligible_pool(p),
                want,
                "{what}: mirror must say {want}"
            );
        };
        expect(&base(2, 2, 2, 2, 32, 32, 16), true, "legacy 2x2 s2 p0");
        expect(&base(3, 3, 1, 1, 8, 8, 16), true, "3x3 s1 p0");
        expect(&base(5, 5, 1, 1, 8, 8, 16), true, "5x5 s1 p0");
        expect(&base(3, 3, 2, 2, 8, 8, 16), true, "3x3 s2 p0");
        expect(&base(2, 2, 2, 2, 32, 32, 8), false, "C%16");
        expect(&base(1, 1, 1, 1, 8, 8, 16), false, "1x1 (reciprocal)");
        expect(&base(2, 1, 1, 1, 8, 8, 16), false, "2x1 (reciprocal)");
        let mut same = base(3, 3, 1, 1, 8, 8, 16);
        same.output_shape = [1, 8, 8, 16];
        same.padding = Padding::Same;
        expect(&same, false, "3x3 SAME pad>0");
        let mut same_even = base(2, 2, 2, 2, 8, 8, 16);
        same_even.output_shape = [1, 4, 4, 16];
        same_even.padding = Padding::Same;
        expect(&same_even, true, "2x2 s2 SAME even (pad_total 0)");
        let mut asym = base(2, 2, 1, 1, 8, 8, 16);
        asym.output_shape = [1, 8, 8, 16];
        asym.padding = Padding::Same;
        expect(&asym, false, "asymmetric SAME partial windows");
    }
}
