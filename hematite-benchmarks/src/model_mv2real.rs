// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! MobileNetV2-style end-to-end CNN model benchmark (model C, "mv2real").
//!
//! The SAME 6-layer graph the standard-ESP-NN baseline runs
//! (`benchmarks/espnn-baseline`), structured like a real MobileNetV2 trunk:
//!
//! | layer | op                            | in -> out             | act     |
//! |-------|-------------------------------|-----------------------|---------|
//! | L1    | conv3x3 stride2 SAME          | 16x16x3  -> 8x8x32    | 0..127  |
//! | L2    | depthwise 3x3 stride1 SAME    | 8x8x32   -> 8x8x32    | 0..127  |
//! | L3    | conv1x1                       | 8x8x32   -> 8x8x64    | 0..127  |
//! | L4    | depthwise 3x3 stride2 SAME    | 8x8x64   -> 4x4x64    | 0..127  |
//! | L5    | conv1x1                       | 4x4x64   -> 4x4x128   | 0..127  |
//! | L6    | fully-connected               | 2048     -> 16        | -128/127|
//!
//! Unlike the earlier model B (mv2mini, all VALID), model C uses **SAME
//! padding and stride-2** layers throughout — exactly the configuration that
//! historically fell back to scalar in Hematite. Every layer here is SIMD
//! eligible in `hematite-s3` after the Phase A/B/C work: L1 conv3x3 SAME
//! stride2 (in_c=3 zero-padded to 16), L2/L4 depthwise SAME (stride 1/2),
//! L3/L5 conv1x1, L6 fc.
//!
//! Both stacks use the identical deterministic fill pattern
//! (`input i*7+3, weights i*13+11, bias i*17-8`) and identical flat-tensor
//! weight layouts: conv OHWI `[oc][kh][kw][ic]`, depthwise HWCN `[kh][kw][oc]`
//! (dm=1), fc OC-major `[oc][i]`. A bit-exact output match therefore proves
//! the two stacks compute the same function on the same model.
//!
//! The reference is `hematite-ref`; the target is `hematite-s3` (ACCX SIMD
//! everywhere). The host test asserts s3 == ref at every layer; the device
//! capture cross-checks the final FNV-1a against the ESP-NN baseline.

use hematite_core::op_params::{
    Conv2DParams, DepthwiseConv2DParams, FullyConnectedParams, Padding,
};

pub const C1_IN_H: usize = 16;
pub const C1_IN_W: usize = 16;
pub const C1_IN_C: usize = 3;
pub const C1_OUT_H: usize = 8;
pub const C1_OUT_W: usize = 8;
pub const C1_OUT_C: usize = 32;
pub const C2_OUT_H: usize = 8;
pub const C2_OUT_W: usize = 8;
pub const C2_OUT_C: usize = 32;
pub const C3_OUT_H: usize = 8;
pub const C3_OUT_W: usize = 8;
pub const C3_OUT_C: usize = 64;
pub const C4_OUT_H: usize = 4;
pub const C4_OUT_W: usize = 4;
pub const C4_OUT_C: usize = 64;
pub const C5_OUT_H: usize = 4;
pub const C5_OUT_W: usize = 4;
pub const C5_OUT_C: usize = 128;
pub const C6_IN_DIM: usize = C5_OUT_H * C5_OUT_W * C5_OUT_C; // 2048
pub const C6_OUT_C: usize = 16;

/// Deterministic fill pattern shared with the ESP-NN baseline C harness.
pub fn fill_pattern_mv2real(
    input: &mut [i8],
    l1w: &mut [i8],
    l1b: &mut [i32],
    l2w: &mut [i8],
    l2b: &mut [i32],
    l3w: &mut [i8],
    l3b: &mut [i32],
    l4w: &mut [i8],
    l4b: &mut [i32],
    l5w: &mut [i8],
    l5b: &mut [i32],
    l6w: &mut [i8],
    l6b: &mut [i32],
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
    for (i, b) in l2w.iter_mut().enumerate() {
        *b = ((i * 13 + 11) & 0xFF) as i8;
    }
    for (i, b) in l2b.iter_mut().enumerate() {
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
}

// ── Layer parameters (shared by the s3 and ref runners) ───────────────────

static MULT_16: [i32; C6_OUT_C] = [1 << 30; C6_OUT_C];
static SHIFT_16: [i32; C6_OUT_C] = [0; C6_OUT_C];
static MULT_32: [i32; C1_OUT_C] = [1 << 30; C1_OUT_C];
static SHIFT_32: [i32; C1_OUT_C] = [0; C1_OUT_C];
static MULT_64: [i32; C3_OUT_C] = [1 << 30; C3_OUT_C];
static SHIFT_64: [i32; C3_OUT_C] = [0; C3_OUT_C];
static MULT_128: [i32; C5_OUT_C] = [1 << 30; C5_OUT_C];
static SHIFT_128: [i32; C5_OUT_C] = [0; C5_OUT_C];

pub static C1_PARAMS: Conv2DParams<'static> = Conv2DParams {
    input_shape: [1, C1_IN_H as i32, C1_IN_W as i32, C1_IN_C as i32],
    filter_shape: [C1_OUT_C as i32, 3, 3, C1_IN_C as i32],
    output_shape: [1, C1_OUT_H as i32, C1_OUT_W as i32, C1_OUT_C as i32],
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
    quantized_activation_min: 0,
    quantized_activation_max: i8::MAX as i32,
};

pub static C2_PARAMS: DepthwiseConv2DParams<'static> = DepthwiseConv2DParams {
    input_shape: [1, C1_OUT_H as i32, C1_OUT_W as i32, C1_OUT_C as i32],
    filter_shape: [1, 3, 3, C2_OUT_C as i32],
    output_shape: [1, C2_OUT_H as i32, C2_OUT_W as i32, C2_OUT_C as i32],
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
    quantized_activation_min: 0,
    quantized_activation_max: i8::MAX as i32,
};

pub static C3_PARAMS: Conv2DParams<'static> = Conv2DParams {
    input_shape: [1, C2_OUT_H as i32, C2_OUT_W as i32, C2_OUT_C as i32],
    filter_shape: [C3_OUT_C as i32, 1, 1, C2_OUT_C as i32],
    output_shape: [1, C3_OUT_H as i32, C3_OUT_W as i32, C3_OUT_C as i32],
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

pub static C4_PARAMS: DepthwiseConv2DParams<'static> = DepthwiseConv2DParams {
    input_shape: [1, C3_OUT_H as i32, C3_OUT_W as i32, C3_OUT_C as i32],
    filter_shape: [1, 3, 3, C4_OUT_C as i32],
    output_shape: [1, C4_OUT_H as i32, C4_OUT_W as i32, C4_OUT_C as i32],
    padding: Padding::Same,
    stride_width: 2,
    stride_height: 2,
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

pub static C5_PARAMS: Conv2DParams<'static> = Conv2DParams {
    input_shape: [1, C4_OUT_H as i32, C4_OUT_W as i32, C4_OUT_C as i32],
    filter_shape: [C5_OUT_C as i32, 1, 1, C4_OUT_C as i32],
    output_shape: [1, C5_OUT_H as i32, C5_OUT_W as i32, C5_OUT_C as i32],
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

pub static C6_PARAMS: FullyConnectedParams<'static> = FullyConnectedParams {
    input_dim: C6_IN_DIM as i32,
    output_dim: C6_OUT_C as i32,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_16,
    output_shift_per_channel: &SHIFT_16,
    quantized_activation_min: i8::MIN as i32,
    quantized_activation_max: i8::MAX as i32,
};

/// Model intermediate + weight buffers (flat tensors, 16-aligned arenas).
pub struct Mv2RealBuffers<'a> {
    pub input: &'a mut [i8],
    pub l1out: &'a mut [i8],
    pub l2out: &'a mut [i8],
    pub l3out: &'a mut [i8],
    pub l4out: &'a mut [i8],
    pub l5out: &'a mut [i8],
    pub out: &'a mut [i8],
    pub l1w: &'a mut [i8],
    pub l1b: &'a mut [i32],
    pub l2w: &'a mut [i8],
    pub l2b: &'a mut [i32],
    pub l3w: &'a mut [i8],
    pub l3b: &'a mut [i32],
    pub l4w: &'a mut [i8],
    pub l4b: &'a mut [i32],
    pub l5w: &'a mut [i8],
    pub l5b: &'a mut [i32],
    pub l6w: &'a mut [i8],
    pub l6b: &'a mut [i32],
}

/// Total arena bytes needed by [`carve_mv2real_into`].
pub const MV2REAL_ARENA_BYTES: usize = {
    C1_IN_H * C1_IN_W * C1_IN_C // input 768
        + C1_OUT_H * C1_OUT_W * C1_OUT_C // l1out 2048
        + C2_OUT_H * C2_OUT_W * C2_OUT_C // l2out 2048
        + C3_OUT_H * C3_OUT_W * C3_OUT_C // l3out 4096
        + C4_OUT_H * C4_OUT_W * C4_OUT_C // l4out 1024
        + C5_OUT_H * C5_OUT_W * C5_OUT_C // l5out 2048
        + C6_OUT_C // out 16
        + C1_OUT_C * 3 * 3 * C1_IN_C // l1w 864
        + C1_OUT_C * 4 // l1b
        + 3 * 3 * C2_OUT_C // l2w 288
        + C2_OUT_C * 4 // l2b
        + C3_OUT_C * C2_OUT_C // l3w 2048
        + C3_OUT_C * 4 // l3b
        + 3 * 3 * C4_OUT_C // l4w 576
        + C4_OUT_C * 4 // l4b
        + C5_OUT_C * C4_OUT_C // l5w 8192
        + C5_OUT_C * 4 // l5b
        + C6_IN_DIM * C6_OUT_C // l6w 32768
        + C6_OUT_C * 4 // l6b
};

/// Carve all model tensors out of a flat 16-aligned arena (sequential,
/// disjoint, 16-byte-aligned boundaries — same discipline as `spec::carve_into`).
pub fn carve_mv2real_into<'a>(
    arena: &'a mut [u8],
) -> Result<Mv2RealBuffers<'a>, hematite_core::KernelError> {
    let in_len = C1_IN_H * C1_IN_W * C1_IN_C;
    let l1out_len = C1_OUT_H * C1_OUT_W * C1_OUT_C;
    let l2out_len = C2_OUT_H * C2_OUT_W * C2_OUT_C;
    let l3out_len = C3_OUT_H * C3_OUT_W * C3_OUT_C;
    let l4out_len = C4_OUT_H * C4_OUT_W * C4_OUT_C;
    let l5out_len = C5_OUT_H * C5_OUT_W * C5_OUT_C;
    let l1w_len = C1_OUT_C * 3 * 3 * C1_IN_C;
    let l2w_len = 3 * 3 * C2_OUT_C;
    let l3w_len = C3_OUT_C * C2_OUT_C;
    let l4w_len = 3 * 3 * C4_OUT_C;
    let l5w_len = C5_OUT_C * C4_OUT_C;
    let l6w_len = C6_IN_DIM * C6_OUT_C;

    let (input_r, rest) = arena.split_at_mut(in_len);
    let (l1out_r, rest) = rest.split_at_mut(l1out_len);
    let (l2out_r, rest) = rest.split_at_mut(l2out_len);
    let (l3out_r, rest) = rest.split_at_mut(l3out_len);
    let (l4out_r, rest) = rest.split_at_mut(l4out_len);
    let (l5out_r, rest) = rest.split_at_mut(l5out_len);
    let (out_r, rest) = rest.split_at_mut(C6_OUT_C);
    let (l1w_r, rest) = rest.split_at_mut(l1w_len);
    let (l1b_r, rest) = rest.split_at_mut(C1_OUT_C * 4);
    let (l2w_r, rest) = rest.split_at_mut(l2w_len);
    let (l2b_r, rest) = rest.split_at_mut(C2_OUT_C * 4);
    let (l3w_r, rest) = rest.split_at_mut(l3w_len);
    let (l3b_r, rest) = rest.split_at_mut(C3_OUT_C * 4);
    let (l4w_r, rest) = rest.split_at_mut(l4w_len);
    let (l4b_r, rest) = rest.split_at_mut(C4_OUT_C * 4);
    let (l5w_r, rest) = rest.split_at_mut(l5w_len);
    let (l5b_r, rest) = rest.split_at_mut(C5_OUT_C * 4);
    let (l6w_r, rest) = rest.split_at_mut(l6w_len);
    let (l6b_r, _after) = rest.split_at_mut(C6_OUT_C * 4);

    // SAFETY: `&mut [u8]` → `&mut [i8]` is a safe transmute of the element
    // type (same size, same alignment); each region is the sole mutable view
    // of disjoint arena bytes.
    let cast_i8 = |s: &mut [u8]| unsafe { core::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut i8, s.len()) };

    // SAFETY: bias regions are 4-byte-aligned and hold exactly N*4 bytes.
    let l1b = unsafe { core::slice::from_raw_parts_mut(l1b_r.as_mut_ptr() as *mut i32, C1_OUT_C) };
    let l2b = unsafe { core::slice::from_raw_parts_mut(l2b_r.as_mut_ptr() as *mut i32, C2_OUT_C) };
    let l3b = unsafe { core::slice::from_raw_parts_mut(l3b_r.as_mut_ptr() as *mut i32, C3_OUT_C) };
    let l4b = unsafe { core::slice::from_raw_parts_mut(l4b_r.as_mut_ptr() as *mut i32, C4_OUT_C) };
    let l5b = unsafe { core::slice::from_raw_parts_mut(l5b_r.as_mut_ptr() as *mut i32, C5_OUT_C) };
    let l6b = unsafe { core::slice::from_raw_parts_mut(l6b_r.as_mut_ptr() as *mut i32, C6_OUT_C) };

    Ok(Mv2RealBuffers {
        input: cast_i8(input_r),
        l1out: cast_i8(l1out_r),
        l2out: cast_i8(l2out_r),
        l3out: cast_i8(l3out_r),
        l4out: cast_i8(l4out_r),
        l5out: cast_i8(l5out_r),
        out: cast_i8(out_r),
        l1w: cast_i8(l1w_r),
        l1b,
        l2w: cast_i8(l2w_r),
        l2b,
        l3w: cast_i8(l3w_r),
        l3b,
        l4w: cast_i8(l4w_r),
        l4b,
        l5w: cast_i8(l5w_r),
        l5b,
        l6w: cast_i8(l6w_r),
        l6b,
    })
}

/// Run the full 6-layer mv2real model through `hematite-s3` (ACCX SIMD).
pub fn run_mv2real_s3(
    bufs: &mut Mv2RealBuffers<'_>,
    scratch: &mut [u8],
) -> Result<(), hematite_core::KernelError> {
    hematite_s3::conv3x3::conv2d_3x3(bufs.input, bufs.l1w, bufs.l1b, &C1_PARAMS, bufs.l1out, scratch)?;
    hematite_s3::depthwise::depthwise_conv2d(bufs.l1out, bufs.l2w, bufs.l2b, &C2_PARAMS, bufs.l2out, scratch)?;
    hematite_s3::conv1x1::conv2d_1x1(bufs.l2out, bufs.l3w, bufs.l3b, &C3_PARAMS, bufs.l3out, scratch)?;
    hematite_s3::depthwise::depthwise_conv2d(bufs.l3out, bufs.l4w, bufs.l4b, &C4_PARAMS, bufs.l4out, scratch)?;
    hematite_s3::conv1x1::conv2d_1x1(bufs.l4out, bufs.l5w, bufs.l5b, &C5_PARAMS, bufs.l5out, scratch)?;
    hematite_s3::gemm::fully_connected(bufs.l5out, bufs.l6w, bufs.l6b, &C6_PARAMS, bufs.out, scratch)
}

/// Run the full 6-layer mv2real model through the `hematite-ref` scalar reference.
pub fn run_mv2real_ref(
    bufs: &mut Mv2RealBuffers<'_>,
    scratch: &mut [u8],
) -> Result<(), hematite_core::KernelError> {
    hematite_ref::conv::conv2d(bufs.input, bufs.l1w, bufs.l1b, &C1_PARAMS, bufs.l1out, scratch)?;
    hematite_ref::depthwise_conv::depthwise_conv2d(bufs.l1out, bufs.l2w, bufs.l2b, &C2_PARAMS, bufs.l2out, scratch)?;
    hematite_ref::conv::conv2d(bufs.l2out, bufs.l3w, bufs.l3b, &C3_PARAMS, bufs.l3out, scratch)?;
    hematite_ref::depthwise_conv::depthwise_conv2d(bufs.l3out, bufs.l4w, bufs.l4b, &C4_PARAMS, bufs.l4out, scratch)?;
    hematite_ref::conv::conv2d(bufs.l4out, bufs.l5w, bufs.l5b, &C5_PARAMS, bufs.l5out, scratch)?;
    hematite_ref::fully_connected::fully_connected(bufs.l5out, bufs.l6w, bufs.l6b, &C6_PARAMS, bufs.out, scratch)
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
/// `dump_layer_checksums_mv2real` — used to localize divergence layer-by-layer).
pub struct LayerChecksumsMv2Real {
    pub l1: u32,
    pub l2: u32,
    pub l3: u32,
    pub l4: u32,
    pub l5: u32,
    pub out: u32,
}

pub fn layer_checksums_mv2real(bufs: &Mv2RealBuffers<'_>) -> LayerChecksumsMv2Real {
    LayerChecksumsMv2Real {
        l1: fnv1a(bufs.l1out),
        l2: fnv1a(bufs.l2out),
        l3: fnv1a(bufs.l3out),
        l4: fnv1a(bufs.l4out),
        l5: fnv1a(bufs.l5out),
        out: fnv1a(bufs.out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mv2real_s3_matches_ref_bit_exact() {
        let mut arena = vec![0u8; MV2REAL_ARENA_BYTES];
        let mut bufs = carve_mv2real_into(&mut arena).unwrap();
        let mut scratch = [0u8; 32768];

        fill_pattern_mv2real(
            bufs.input, bufs.l1w, bufs.l1b, bufs.l2w, bufs.l2b, bufs.l3w, bufs.l3b, bufs.l4w,
            bufs.l4b, bufs.l5w, bufs.l5b, bufs.l6w, bufs.l6b,
        );
        run_mv2real_ref(&mut bufs, &mut scratch).unwrap();
        let ref_chk = fnv1a(bufs.out);
        let ref_layers = layer_checksums_mv2real(&bufs);

        fill_pattern_mv2real(
            bufs.input, bufs.l1w, bufs.l1b, bufs.l2w, bufs.l2b, bufs.l3w, bufs.l3b, bufs.l4w,
            bufs.l4b, bufs.l5w, bufs.l5b, bufs.l6w, bufs.l6b,
        );
        run_mv2real_s3(&mut bufs, &mut scratch).unwrap();
        let s3_chk = fnv1a(bufs.out);
        let s3_layers = layer_checksums_mv2real(&bufs);

        assert_eq!(ref_chk, s3_chk, "s3 model output must equal ref bit-exact");
        assert_eq!(ref_layers.l1, s3_layers.l1);
        assert_eq!(ref_layers.l2, s3_layers.l2);
        assert_eq!(ref_layers.l3, s3_layers.l3);
        assert_eq!(ref_layers.l4, s3_layers.l4);
        assert_eq!(ref_layers.l5, s3_layers.l5);
        assert_eq!(ref_layers.out, s3_layers.out);
    }
}
