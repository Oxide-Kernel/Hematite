// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! MobileNetV2-style end-to-end CNN model benchmark (model B, "mv2mini").
//!
//! The SAME 7-layer graph the standard-ESP-NN baseline runs
//! (`benchmarks/espnn-baseline`), MobileNetV2-style building blocks:
//!
//! | layer | op                         | in -> out                 | act     |
//! |-------|----------------------------|---------------------------|---------|
//! | L1    | conv3x3 stride1 VALID      | 16x16x3  -> 14x14x32      | 0..127  |
//! | L2    | maxpool 2x2 s2             | 14x14x32 -> 7x7x32        | -128/127|
//! | L3    | depthwise 3x3 stride1 VALID| 7x7x32   -> 5x5x32  dm=1 | 0..127  |
//! | L4    | conv1x1                    | 5x5x32   -> 5x5x64        | 0..127  |
//! | L5    | depthwise 3x3 stride1 VALID| 5x5x64   -> 3x3x64  dm=1 | 0..127  |
//! | L6    | conv1x1                    | 3x3x64   -> 3x3x128       | 0..127  |
//! | L7    | fully-connected            | 1152     -> 16            | -128/127|
//!
//! Both stacks use the identical deterministic fill pattern
//! (`input i*7+3, weights i*13+11, bias i*17-8`) and identical flat-tensor
//! weight layouts: conv OHWI `[oc][kh][kw][ic]`, depthwise HWCN `[kh][kw][oc]`
//! (dm=1), fc OC-major `[oc][i]`. A bit-exact output match therefore proves
//! the two stacks compute the same function on the same model.
//!
//! The reference is `hematite-ref`; the target is `hematite-s3` (ACCX SIMD
//! where eligible: L1 conv3x3 has in_c=3 → scalar fallback, L3/L5 depthwise is
//! scalar-only by design, L4/L6 conv1x1 + L7 fc fire ACCX, L2 maxpool fires
//! the TIE728 pool SIMD). The host test asserts s3 == ref at every layer and
//! that the final FNV-1a equals the ESP-NN baseline's device-verified value
//! `0x7f23eb05`.

use hematite_core::op_params::{
    Conv2DParams, DepthwiseConv2DParams, FusedActivation, FullyConnectedParams, Padding, PoolParams,
};

pub const M1_IN_H: usize = 16;
pub const M1_IN_W: usize = 16;
pub const M1_IN_C: usize = 3;
pub const M1_OUT_H: usize = 14;
pub const M1_OUT_W: usize = 14;
pub const M1_OUT_C: usize = 32;
pub const M2_OUT_H: usize = 7;
pub const M2_OUT_W: usize = 7;
pub const M2_OUT_C: usize = 32;
pub const M3_OUT_H: usize = 5;
pub const M3_OUT_W: usize = 5;
pub const M3_OUT_C: usize = 32;
pub const M4_OUT_H: usize = 5;
pub const M4_OUT_W: usize = 5;
pub const M4_OUT_C: usize = 64;
pub const M5_OUT_H: usize = 3;
pub const M5_OUT_W: usize = 3;
pub const M5_OUT_C: usize = 64;
pub const M6_OUT_H: usize = 3;
pub const M6_OUT_W: usize = 3;
pub const M6_OUT_C: usize = 128;
pub const M7_IN_DIM: usize = M6_OUT_H * M6_OUT_W * M6_OUT_C; // 1152
pub const M7_OUT_C: usize = 16;

/// Deterministic fill pattern shared with the ESP-NN baseline C harness.
pub fn fill_pattern_mv2(
    input: &mut [i8],
    l1w: &mut [i8],
    l1b: &mut [i32],
    l3w: &mut [i8],
    l3b: &mut [i32],
    l4w: &mut [i8],
    l4b: &mut [i32],
    l5w: &mut [i8],
    l5b: &mut [i32],
    l6w: &mut [i8],
    l6b: &mut [i32],
    l7w: &mut [i8],
    l7b: &mut [i32],
) {
    for (i, b) in input.iter_mut().enumerate() {
        *b = ((i * 7 + 3) & 0xFF) as i8;
    }
    for (i, b) in l1w.iter_mut().enumerate() {
        *b = ((i * 13 + 11) & 0xFF) as i8;
    }
    for (i, b) in l1b.iter_mut().enumerate() {
        *b = (i as i32) * 17 - 8;
    }
    for (i, b) in l3w.iter_mut().enumerate() {
        *b = ((i * 13 + 11) & 0xFF) as i8;
    }
    for (i, b) in l3b.iter_mut().enumerate() {
        *b = (i as i32) * 17 - 8;
    }
    for (i, b) in l4w.iter_mut().enumerate() {
        *b = ((i * 13 + 11) & 0xFF) as i8;
    }
    for (i, b) in l4b.iter_mut().enumerate() {
        *b = (i as i32) * 17 - 8;
    }
    for (i, b) in l5w.iter_mut().enumerate() {
        *b = ((i * 13 + 11) & 0xFF) as i8;
    }
    for (i, b) in l5b.iter_mut().enumerate() {
        *b = (i as i32) * 17 - 8;
    }
    for (i, b) in l6w.iter_mut().enumerate() {
        *b = ((i * 13 + 11) & 0xFF) as i8;
    }
    for (i, b) in l6b.iter_mut().enumerate() {
        *b = (i as i32) * 17 - 8;
    }
    for (i, b) in l7w.iter_mut().enumerate() {
        *b = ((i * 13 + 11) & 0xFF) as i8;
    }
    for (i, b) in l7b.iter_mut().enumerate() {
        *b = (i as i32) * 17 - 8;
    }
}

// ── Layer parameters (shared by the s3 and ref runners) ───────────────────

static MULT_16: [i32; M7_OUT_C] = [1 << 30; M7_OUT_C];
static SHIFT_16: [i32; M7_OUT_C] = [0; M7_OUT_C];
static MULT_32: [i32; M1_OUT_C] = [1 << 30; M1_OUT_C];
static SHIFT_32: [i32; M1_OUT_C] = [0; M1_OUT_C];
static MULT_64: [i32; M4_OUT_C] = [1 << 30; M4_OUT_C];
static SHIFT_64: [i32; M4_OUT_C] = [0; M4_OUT_C];
static MULT_128: [i32; M6_OUT_C] = [1 << 30; M6_OUT_C];
static SHIFT_128: [i32; M6_OUT_C] = [0; M6_OUT_C];

pub static M1_PARAMS: Conv2DParams<'static> = Conv2DParams {
    input_shape: [1, M1_IN_H as i32, M1_IN_W as i32, M1_IN_C as i32],
    filter_shape: [M1_OUT_C as i32, 3, 3, M1_IN_C as i32],
    output_shape: [1, M1_OUT_H as i32, M1_OUT_W as i32, M1_OUT_C as i32],
    padding: Padding::Valid,
    stride_width: 1,
    stride_height: 1,
    dilation_width_factor: 1,
    dilation_height_factor: 1,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_32,
    output_shift_per_channel: &SHIFT_32,
    quantized_activation_min: 0,
    quantized_activation_max: i8::MAX as i32,
};

pub static M2_PARAMS: PoolParams = PoolParams {
    input_shape: [1, M1_OUT_H as i32, M1_OUT_W as i32, M1_OUT_C as i32],
    output_shape: [1, M2_OUT_H as i32, M2_OUT_W as i32, M2_OUT_C as i32],
    filter_width: 2,
    filter_height: 2,
    stride_width: 2,
    stride_height: 2,
    padding: Padding::Valid,
    activation: FusedActivation::None,
    quantized_activation_min: i8::MIN as i32,
    quantized_activation_max: i8::MAX as i32,
};

pub static M3_PARAMS: DepthwiseConv2DParams<'static> = DepthwiseConv2DParams {
    input_shape: [1, M2_OUT_H as i32, M2_OUT_W as i32, M2_OUT_C as i32],
    filter_shape: [1, 3, 3, M3_OUT_C as i32],
    output_shape: [1, M3_OUT_H as i32, M3_OUT_W as i32, M3_OUT_C as i32],
    padding: Padding::Valid,
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
    quantized_activation_min: 0,
    quantized_activation_max: i8::MAX as i32,
};

pub static M4_PARAMS: Conv2DParams<'static> = Conv2DParams {
    input_shape: [1, M3_OUT_H as i32, M3_OUT_W as i32, M3_OUT_C as i32],
    filter_shape: [M4_OUT_C as i32, 1, 1, M3_OUT_C as i32],
    output_shape: [1, M4_OUT_H as i32, M4_OUT_W as i32, M4_OUT_C as i32],
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
    quantized_activation_min: 0,
    quantized_activation_max: i8::MAX as i32,
};

pub static M5_PARAMS: DepthwiseConv2DParams<'static> = DepthwiseConv2DParams {
    input_shape: [1, M4_OUT_H as i32, M4_OUT_W as i32, M4_OUT_C as i32],
    filter_shape: [1, 3, 3, M5_OUT_C as i32],
    output_shape: [1, M5_OUT_H as i32, M5_OUT_W as i32, M5_OUT_C as i32],
    padding: Padding::Valid,
    stride_width: 1,
    stride_height: 1,
    dilation_width_factor: 1,
    dilation_height_factor: 1,
    depth_multiplier: 1,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_64,
    output_shift_per_channel: &SHIFT_64,
    quantized_activation_min: 0,
    quantized_activation_max: i8::MAX as i32,
};

pub static M6_PARAMS: Conv2DParams<'static> = Conv2DParams {
    input_shape: [1, M5_OUT_H as i32, M5_OUT_W as i32, M5_OUT_C as i32],
    filter_shape: [M6_OUT_C as i32, 1, 1, M5_OUT_C as i32],
    output_shape: [1, M6_OUT_H as i32, M6_OUT_W as i32, M6_OUT_C as i32],
    padding: Padding::Valid,
    stride_width: 1,
    stride_height: 1,
    dilation_width_factor: 1,
    dilation_height_factor: 1,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_128,
    output_shift_per_channel: &SHIFT_128,
    quantized_activation_min: 0,
    quantized_activation_max: i8::MAX as i32,
};

pub static M7_PARAMS: FullyConnectedParams<'static> = FullyConnectedParams {
    input_dim: M7_IN_DIM as i32,
    output_dim: M7_OUT_C as i32,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_16,
    output_shift_per_channel: &SHIFT_16,
    quantized_activation_min: i8::MIN as i32,
    quantized_activation_max: i8::MAX as i32,
};

/// Model intermediate + weight buffers (flat tensors, 16-aligned arenas).
pub struct Mv2Buffers<'a> {
    pub input: &'a mut [i8],
    pub l1out: &'a mut [i8],
    pub l2out: &'a mut [i8],
    pub l3out: &'a mut [i8],
    pub l4out: &'a mut [i8],
    pub l5out: &'a mut [i8],
    pub l6out: &'a mut [i8],
    pub out: &'a mut [i8],
    pub l1w: &'a mut [i8],
    pub l1b: &'a mut [i32],
    pub l3w: &'a mut [i8],
    pub l3b: &'a mut [i32],
    pub l4w: &'a mut [i8],
    pub l4b: &'a mut [i32],
    pub l5w: &'a mut [i8],
    pub l5b: &'a mut [i32],
    pub l6w: &'a mut [i8],
    pub l6b: &'a mut [i32],
    pub l7w: &'a mut [i8],
    pub l7b: &'a mut [i32],
}

/// Total arena bytes needed by [`carve_mv2_into`].
pub const MV2_ARENA_BYTES: usize = {
    M1_IN_H * M1_IN_W * M1_IN_C // input 768
        + M1_OUT_H * M1_OUT_W * M1_OUT_C // l1out 6272
        + M2_OUT_H * M2_OUT_W * M2_OUT_C // l2out 1568
        + M3_OUT_H * M3_OUT_W * M3_OUT_C // l3out 800
        + M4_OUT_H * M4_OUT_W * M4_OUT_C // l4out 1600
        + M5_OUT_H * M5_OUT_W * M5_OUT_C // l5out 576
        + M6_OUT_H * M6_OUT_W * M6_OUT_C // l6out 1152
        + M7_OUT_C // out 16
        + M1_OUT_C * 3 * 3 * M1_IN_C // l1w 432
        + M1_OUT_C * 4 // l1b
        + 3 * 3 * M3_OUT_C // l3w 288
        + M3_OUT_C * 4 // l3b
        + M4_OUT_C * M3_OUT_C // l4w 2048
        + M4_OUT_C * 4 // l4b
        + 3 * 3 * M5_OUT_C // l5w 576
        + M5_OUT_C * 4 // l5b
        + M6_OUT_C * M5_OUT_C // l6w 8192
        + M6_OUT_C * 4 // l6b
        + M7_IN_DIM * M7_OUT_C // l7w 18432
        + M7_OUT_C * 4 // l7b
};

/// Carve all model tensors out of a flat 16-aligned arena (sequential,
/// disjoint, 16-byte-aligned boundaries — same discipline as `spec::carve_into`).
pub fn carve_mv2_into<'a>(arena: &'a mut [u8]) -> Result<Mv2Buffers<'a>, hematite_core::KernelError> {
    let in_len = M1_IN_H * M1_IN_W * M1_IN_C;
    let l1out_len = M1_OUT_H * M1_OUT_W * M1_OUT_C;
    let l2out_len = M2_OUT_H * M2_OUT_W * M2_OUT_C;
    let l3out_len = M3_OUT_H * M3_OUT_W * M3_OUT_C;
    let l4out_len = M4_OUT_H * M4_OUT_W * M4_OUT_C;
    let l5out_len = M5_OUT_H * M5_OUT_W * M5_OUT_C;
    let l6out_len = M6_OUT_H * M6_OUT_W * M6_OUT_C;
    let l1w_len = M1_OUT_C * 3 * 3 * M1_IN_C;
    let l3w_len = 3 * 3 * M3_OUT_C;
    let l4w_len = M4_OUT_C * M3_OUT_C;
    let l5w_len = 3 * 3 * M5_OUT_C;
    let l6w_len = M6_OUT_C * M5_OUT_C;
    let l7w_len = M7_IN_DIM * M7_OUT_C;

    let (input_r, rest) = arena.split_at_mut(in_len);
    let (l1out_r, rest) = rest.split_at_mut(l1out_len);
    let (l2out_r, rest) = rest.split_at_mut(l2out_len);
    let (l3out_r, rest) = rest.split_at_mut(l3out_len);
    let (l4out_r, rest) = rest.split_at_mut(l4out_len);
    let (l5out_r, rest) = rest.split_at_mut(l5out_len);
    let (l6out_r, rest) = rest.split_at_mut(l6out_len);
    let (out_r, rest) = rest.split_at_mut(M7_OUT_C);
    let (l1w_r, rest) = rest.split_at_mut(l1w_len);
    let (l1b_r, rest) = rest.split_at_mut(M1_OUT_C * 4);
    let (l3w_r, rest) = rest.split_at_mut(l3w_len);
    let (l3b_r, rest) = rest.split_at_mut(M3_OUT_C * 4);
    let (l4w_r, rest) = rest.split_at_mut(l4w_len);
    let (l4b_r, rest) = rest.split_at_mut(M4_OUT_C * 4);
    let (l5w_r, rest) = rest.split_at_mut(l5w_len);
    let (l5b_r, rest) = rest.split_at_mut(M5_OUT_C * 4);
    let (l6w_r, rest) = rest.split_at_mut(l6w_len);
    let (l6b_r, rest) = rest.split_at_mut(M6_OUT_C * 4);
    let (l7w_r, rest) = rest.split_at_mut(l7w_len);
    let (l7b_r, _after) = rest.split_at_mut(M7_OUT_C * 4);

    // SAFETY: `&mut [u8]` → `&mut [i8]` is a safe transmute of the element
    // type (same size, same alignment); each region is the sole mutable view
    // of disjoint arena bytes.
    let cast_i8 = |s: &mut [u8]| unsafe { core::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut i8, s.len()) };

    // SAFETY: bias regions are 4-byte-aligned and hold exactly N*4 bytes.
    let l1b = unsafe { core::slice::from_raw_parts_mut(l1b_r.as_mut_ptr() as *mut i32, M1_OUT_C) };
    let l3b = unsafe { core::slice::from_raw_parts_mut(l3b_r.as_mut_ptr() as *mut i32, M3_OUT_C) };
    let l4b = unsafe { core::slice::from_raw_parts_mut(l4b_r.as_mut_ptr() as *mut i32, M4_OUT_C) };
    let l5b = unsafe { core::slice::from_raw_parts_mut(l5b_r.as_mut_ptr() as *mut i32, M5_OUT_C) };
    let l6b = unsafe { core::slice::from_raw_parts_mut(l6b_r.as_mut_ptr() as *mut i32, M6_OUT_C) };
    let l7b = unsafe { core::slice::from_raw_parts_mut(l7b_r.as_mut_ptr() as *mut i32, M7_OUT_C) };

    Ok(Mv2Buffers {
        input: cast_i8(input_r),
        l1out: cast_i8(l1out_r),
        l2out: cast_i8(l2out_r),
        l3out: cast_i8(l3out_r),
        l4out: cast_i8(l4out_r),
        l5out: cast_i8(l5out_r),
        l6out: cast_i8(l6out_r),
        out: cast_i8(out_r),
        l1w: cast_i8(l1w_r),
        l1b,
        l3w: cast_i8(l3w_r),
        l3b,
        l4w: cast_i8(l4w_r),
        l4b,
        l5w: cast_i8(l5w_r),
        l5b,
        l6w: cast_i8(l6w_r),
        l6b,
        l7w: cast_i8(l7w_r),
        l7b,
    })
}

/// Run the full 7-layer mv2mini model through `hematite-s3` (ACCX SIMD where
/// eligible, scalar fallback for L1/L3/L5).
pub fn run_mv2_s3(bufs: &mut Mv2Buffers<'_>, scratch: &mut [u8]) -> Result<(), hematite_core::KernelError> {
    hematite_s3::conv3x3::conv2d_3x3(bufs.input, bufs.l1w, bufs.l1b, &M1_PARAMS, bufs.l1out, scratch)?;
    hematite_s3::pool::max_pool_2d(bufs.l1out, &M2_PARAMS, bufs.l2out, scratch)?;
    hematite_s3::depthwise::depthwise_conv2d(bufs.l2out, bufs.l3w, bufs.l3b, &M3_PARAMS, bufs.l3out, scratch)?;
    hematite_s3::conv1x1::conv2d_1x1(bufs.l3out, bufs.l4w, bufs.l4b, &M4_PARAMS, bufs.l4out, scratch)?;
    hematite_s3::depthwise::depthwise_conv2d(bufs.l4out, bufs.l5w, bufs.l5b, &M5_PARAMS, bufs.l5out, scratch)?;
    hematite_s3::conv1x1::conv2d_1x1(bufs.l5out, bufs.l6w, bufs.l6b, &M6_PARAMS, bufs.l6out, scratch)?;
    hematite_s3::gemm::fully_connected(bufs.l6out, bufs.l7w, bufs.l7b, &M7_PARAMS, bufs.out, scratch)
}

/// Run the full 7-layer mv2mini model through the `hematite-ref` scalar reference.
pub fn run_mv2_ref(bufs: &mut Mv2Buffers<'_>, scratch: &mut [u8]) -> Result<(), hematite_core::KernelError> {
    hematite_ref::conv::conv2d(bufs.input, bufs.l1w, bufs.l1b, &M1_PARAMS, bufs.l1out, scratch)?;
    hematite_ref::pool::max_pool_2d(bufs.l1out, &M2_PARAMS, bufs.l2out, scratch)?;
    hematite_ref::depthwise_conv::depthwise_conv2d(bufs.l2out, bufs.l3w, bufs.l3b, &M3_PARAMS, bufs.l3out, scratch)?;
    hematite_ref::conv::conv2d(bufs.l3out, bufs.l4w, bufs.l4b, &M4_PARAMS, bufs.l4out, scratch)?;
    hematite_ref::depthwise_conv::depthwise_conv2d(bufs.l4out, bufs.l5w, bufs.l5b, &M5_PARAMS, bufs.l5out, scratch)?;
    hematite_ref::conv::conv2d(bufs.l5out, bufs.l6w, bufs.l6b, &M6_PARAMS, bufs.l6out, scratch)?;
    hematite_ref::fully_connected::fully_connected(bufs.l6out, bufs.l7w, bufs.l7b, &M7_PARAMS, bufs.out, scratch)
}

/// FNV-1a (sign-extending, matching the firmware report + ESP-NN baseline).
pub fn fnv1a(data: &[i8]) -> u32 {
    let mut h: u32 = 2166136261;
    for &b in data {
        h ^= b as u32;
        h = h.wrapping_mul(16777619);
    }
    h
}

/// Per-layer intermediate checksums (matches the ESP-NN baseline's
/// `dump_layer_checksums_mv2` — used to localize divergence layer-by-layer).
pub struct LayerChecksumsMv2 {
    pub l1: u32,
    pub l2: u32,
    pub l3: u32,
    pub l4: u32,
    pub l5: u32,
    pub l6: u32,
    pub out: u32,
}

pub fn layer_checksums_mv2(bufs: &Mv2Buffers<'_>) -> LayerChecksumsMv2 {
    LayerChecksumsMv2 {
        l1: fnv1a(bufs.l1out),
        l2: fnv1a(bufs.l2out),
        l3: fnv1a(bufs.l3out),
        l4: fnv1a(bufs.l4out),
        l5: fnv1a(bufs.l5out),
        l6: fnv1a(bufs.l6out),
        out: fnv1a(bufs.out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mv2_s3_matches_ref_and_espnn_checksum() {
        let mut arena = vec![0u8; MV2_ARENA_BYTES];
        let mut bufs = carve_mv2_into(&mut arena).unwrap();
        let mut scratch = [0u8; 4096];

        fill_pattern_mv2(
            bufs.input, bufs.l1w, bufs.l1b, bufs.l3w, bufs.l3b, bufs.l4w, bufs.l4b, bufs.l5w,
            bufs.l5b, bufs.l6w, bufs.l6b, bufs.l7w, bufs.l7b,
        );
        run_mv2_ref(&mut bufs, &mut scratch).unwrap();
        let ref_chk = fnv1a(bufs.out);
        let ref_layers = layer_checksums_mv2(&bufs);

        fill_pattern_mv2(
            bufs.input, bufs.l1w, bufs.l1b, bufs.l3w, bufs.l3b, bufs.l4w, bufs.l4b, bufs.l5w,
            bufs.l5b, bufs.l6w, bufs.l6b, bufs.l7w, bufs.l7b,
        );
        run_mv2_s3(&mut bufs, &mut scratch).unwrap();
        let s3_chk = fnv1a(bufs.out);
        let s3_layers = layer_checksums_mv2(&bufs);

        assert_eq!(ref_chk, s3_chk, "s3 model output must equal ref bit-exact");
        assert_eq!(ref_layers.l1, s3_layers.l1);
        assert_eq!(ref_layers.l2, s3_layers.l2);
        assert_eq!(ref_layers.l3, s3_layers.l3);
        assert_eq!(ref_layers.l4, s3_layers.l4);
        assert_eq!(ref_layers.l5, s3_layers.l5);
        assert_eq!(ref_layers.l6, s3_layers.l6);
        assert_eq!(ref_layers.out, s3_layers.out);

        // Device-verified ESP-NN baseline output checksum.
        assert_eq!(
            s3_chk, 0x7f23eb05,
            "model final output must match the ESP-NN baseline 0x7f23eb05 (got 0x{:08x})",
            s3_chk
        );
    }
}
