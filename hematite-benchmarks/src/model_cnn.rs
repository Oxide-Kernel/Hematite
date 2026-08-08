// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Small end-to-end CNN model benchmark.
//!
//! The SAME 4-layer graph the standard-ESP-NN baseline runs
//! (`benchmarks/espnn-baseline`): conv3x3 32x32x16→30x30x16 (stride 1,
//! VALID, relu 0..127) → maxpool 2x2 s2 → conv1x1 15x15x16→15x15x32
//! (relu 0..127) → fully-connected 7200→16. Both stacks use the identical
//! deterministic fill pattern (`input i*7+3, weights i*13+11, bias i*17-8`)
//! and the identical flat-tensor weight layouts (OHWI for conv, OC-major for
//! fc), so a bit-exact output match proves the two stacks compute the same
//! function on the same model.
//!
//! The reference scalar implementation is `hematite-ref`; the target
//! implementation is `hematite-s3` (ACCX kernels on device, scalar fallback
//! on host). The host test asserts both produce the same final output and
//! that the FNV-1a checksum equals the ESP-NN baseline's device-verified
//! value `0x75eb32f5`.

use hematite_core::op_params::{Conv2DParams, FusedActivation, FullyConnectedParams, Padding, PoolParams};

pub const L1_IN_H: usize = 32;
pub const L1_IN_W: usize = 32;
pub const L1_IN_C: usize = 16;
pub const L1_OUT_H: usize = 30;
pub const L1_OUT_W: usize = 30;
pub const L1_OUT_C: usize = 16;
pub const L2_OUT_H: usize = 15;
pub const L2_OUT_W: usize = 15;
pub const L2_OUT_C: usize = 16;
pub const L3_OUT_H: usize = 15;
pub const L3_OUT_W: usize = 15;
pub const L3_OUT_C: usize = 32;
pub const L4_IN_DIM: usize = L3_OUT_H * L3_OUT_W * L3_OUT_C; // 7200
pub const L4_OUT_C: usize = 16;

/// Deterministic fill pattern shared with the ESP-NN baseline C harness
/// (benchmarks/espnn-baseline/main/main.c) and the per-kernel spec.rs.
pub fn fill_pattern_cnn(
    input: &mut [i8],
    l1w: &mut [i8],
    l1b: &mut [i32],
    l3w: &mut [i8],
    l3b: &mut [i32],
    l4w: &mut [i8],
    l4b: &mut [i32],
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
}

// ── Layer parameters (shared by the s3 and ref runners) ───────────────────

static MULT_16: [i32; L1_OUT_C] = [1 << 30; L1_OUT_C];
static SHIFT_16: [i32; L1_OUT_C] = [0; L1_OUT_C];
static MULT_32: [i32; L3_OUT_C] = [1 << 30; L3_OUT_C];
static SHIFT_32: [i32; L3_OUT_C] = [0; L3_OUT_C];

pub static L1_PARAMS: Conv2DParams<'static> = Conv2DParams {
    input_shape: [1, L1_IN_H as i32, L1_IN_W as i32, L1_IN_C as i32],
    filter_shape: [L1_OUT_C as i32, 3, 3, L1_IN_C as i32],
    output_shape: [1, L1_OUT_H as i32, L1_OUT_W as i32, L1_OUT_C as i32],
    padding: Padding::Valid,
    stride_width: 1,
    stride_height: 1,
    dilation_width_factor: 1,
    dilation_height_factor: 1,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_16,
    output_shift_per_channel: &SHIFT_16,
    quantized_activation_min: 0,
    quantized_activation_max: i8::MAX as i32,
};

pub static L2_PARAMS: PoolParams = PoolParams {
    input_shape: [1, L1_OUT_H as i32, L1_OUT_W as i32, L1_OUT_C as i32],
    output_shape: [1, L2_OUT_H as i32, L2_OUT_W as i32, L2_OUT_C as i32],
    filter_width: 2,
    filter_height: 2,
    stride_width: 2,
    stride_height: 2,
    padding: Padding::Valid,
    activation: FusedActivation::None,
    quantized_activation_min: i8::MIN as i32,
    quantized_activation_max: i8::MAX as i32,
};

pub static L3_PARAMS: Conv2DParams<'static> = Conv2DParams {
    input_shape: [1, L2_OUT_H as i32, L2_OUT_W as i32, L2_OUT_C as i32],
    filter_shape: [L3_OUT_C as i32, 1, 1, L2_OUT_C as i32],
    output_shape: [1, L3_OUT_H as i32, L3_OUT_W as i32, L3_OUT_C as i32],
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

pub static L4_PARAMS: FullyConnectedParams<'static> = FullyConnectedParams {
    input_dim: L4_IN_DIM as i32,
    output_dim: L4_OUT_C as i32,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_16,
    output_shift_per_channel: &SHIFT_16,
    quantized_activation_min: i8::MIN as i32,
    quantized_activation_max: i8::MAX as i32,
};

/// Model intermediate + weight buffers (flat tensors, 16-aligned arenas).
pub struct CnnBuffers<'a> {
    pub input: &'a mut [i8],
    pub l1out: &'a mut [i8],
    pub l2out: &'a mut [i8],
    pub l3out: &'a mut [i8],
    pub out: &'a mut [i8],
    pub l1w: &'a mut [i8],
    pub l1b: &'a mut [i32],
    pub l3w: &'a mut [i8],
    pub l3b: &'a mut [i32],
    pub l4w: &'a mut [i8],
    pub l4b: &'a mut [i32],
}

/// Total arena bytes needed by [`carve_cnn_into`].
pub const CNN_ARENA_BYTES: usize = {
    // All tensor lengths are multiples of 16, so sequential 16-aligned
    // boundaries are exact and no padding is required.
    L1_IN_H * L1_IN_W * L1_IN_C
        + L1_OUT_H * L1_OUT_W * L1_OUT_C
        + L2_OUT_H * L2_OUT_W * L2_OUT_C
        + L3_OUT_H * L3_OUT_W * L3_OUT_C
        + L4_OUT_C
        + L1_OUT_C * 3 * 3 * L1_IN_C
        + L1_OUT_C * 4
        + L3_OUT_C * L2_OUT_C
        + L3_OUT_C * 4
        + L4_IN_DIM * L4_OUT_C
        + L4_OUT_C * 4
};

/// Carve all model tensors out of a flat 16-aligned arena (sequential,
/// disjoint, 16-byte-aligned boundaries — same discipline as `spec::carve_into`).
pub fn carve_cnn_into<'a>(arena: &'a mut [u8]) -> Result<CnnBuffers<'a>, hematite_core::KernelError> {
    let align16 = |o: usize| o.div_ceil(16) * 16;
    let in_len = L1_IN_H * L1_IN_W * L1_IN_C;
    let l1out_len = L1_OUT_H * L1_OUT_W * L1_OUT_C;
    let l2out_len = L2_OUT_H * L2_OUT_W * L2_OUT_C;
    let l3out_len = L3_OUT_H * L3_OUT_W * L3_OUT_C;
    let l1w_len = L1_OUT_C * 3 * 3 * L1_IN_C;
    let l3w_len = L3_OUT_C * L2_OUT_C;
    let l4w_len = L4_IN_DIM * L4_OUT_C;

    let (input_r, rest) = arena.split_at_mut(in_len);
    let (l1out_r, rest) = rest.split_at_mut(l1out_len);
    let (l2out_r, rest) = rest.split_at_mut(l2out_len);
    let (l3out_r, rest) = rest.split_at_mut(l3out_len);
    let (out_r, rest) = rest.split_at_mut(L4_OUT_C);
    let (l1w_r, rest) = rest.split_at_mut(l1w_len);
    let (l1b_r, rest) = rest.split_at_mut(L1_OUT_C * 4);
    let (l3w_r, rest) = rest.split_at_mut(l3w_len);
    let (l3b_r, rest) = rest.split_at_mut(L3_OUT_C * 4);
    let (l4w_r, rest) = rest.split_at_mut(l4w_len);
    let (l4b_r, _after) = rest.split_at_mut(L4_OUT_C * 4);

    let _ = align16; // boundaries are 16-aligned by construction (all lens are 16-multiples)

    // SAFETY: `&mut [u8]` → `&mut [i8]` is a safe transmute of the element
    // type (same size, same alignment); each region is the sole mutable view
    // of disjoint arena bytes.
    let cast_i8 = |s: &mut [u8]| unsafe { core::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut i8, s.len()) };

    // SAFETY: bias regions are 4-byte-aligned and hold exactly N*4 bytes.
    let l1b = unsafe { core::slice::from_raw_parts_mut(l1b_r.as_mut_ptr() as *mut i32, L1_OUT_C) };
    let l3b = unsafe { core::slice::from_raw_parts_mut(l3b_r.as_mut_ptr() as *mut i32, L3_OUT_C) };
    let l4b = unsafe { core::slice::from_raw_parts_mut(l4b_r.as_mut_ptr() as *mut i32, L4_OUT_C) };

    Ok(CnnBuffers {
        input: cast_i8(input_r),
        l1out: cast_i8(l1out_r),
        l2out: cast_i8(l2out_r),
        l3out: cast_i8(l3out_r),
        out: cast_i8(out_r),
        l1w: cast_i8(l1w_r),
        l1b,
        l3w: cast_i8(l3w_r),
        l3b,
        l4w: cast_i8(l4w_r),
        l4b,
    })
}

/// Run the full 4-layer model through `hematite-s3` (ACCX SIMD on device,
/// scalar fallback on host).
pub fn run_cnn_s3(bufs: &mut CnnBuffers<'_>, scratch: &mut [u8]) -> Result<(), hematite_core::KernelError> {
    hematite_s3::conv3x3::conv2d_3x3(
        bufs.input,
        bufs.l1w,
        bufs.l1b,
        &L1_PARAMS,
        bufs.l1out,
        scratch,
    )?;
    hematite_s3::pool::max_pool_2d(bufs.l1out, &L2_PARAMS, bufs.l2out, scratch)?;
    hematite_s3::conv1x1::conv2d_1x1(
        bufs.l2out,
        bufs.l3w,
        bufs.l3b,
        &L3_PARAMS,
        bufs.l3out,
        scratch,
    )?;
    hematite_s3::gemm::fully_connected(
        bufs.l3out,
        bufs.l4w,
        bufs.l4b,
        &L4_PARAMS,
        bufs.out,
        scratch,
    )
}

/// Run the full 4-layer model through the `hematite-ref` scalar reference.
pub fn run_cnn_ref(bufs: &mut CnnBuffers<'_>, scratch: &mut [u8]) -> Result<(), hematite_core::KernelError> {
    hematite_ref::conv::conv2d(
        bufs.input,
        bufs.l1w,
        bufs.l1b,
        &L1_PARAMS,
        bufs.l1out,
        scratch,
    )?;
    hematite_ref::pool::max_pool_2d(bufs.l1out, &L2_PARAMS, bufs.l2out, scratch)?;
    hematite_ref::conv::conv2d(
        bufs.l2out,
        bufs.l3w,
        bufs.l3b,
        &L3_PARAMS,
        bufs.l3out,
        scratch,
    )?;
    hematite_ref::fully_connected::fully_connected(
        bufs.l3out,
        bufs.l4w,
        bufs.l4b,
        &L4_PARAMS,
        bufs.out,
        scratch,
    )
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
/// `dump_layer_checksums` — used to localize divergence layer-by-layer).
pub struct LayerChecksums {
    pub l1: u32,
    pub l2: u32,
    pub l3: u32,
    pub out: u32,
}

pub fn layer_checksums(bufs: &CnnBuffers<'_>) -> LayerChecksums {
    LayerChecksums {
        l1: fnv1a(bufs.l1out),
        l2: fnv1a(bufs.l2out),
        l3: fnv1a(bufs.l3out),
        out: fnv1a(bufs.out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cnn_s3_matches_ref_and_espnn_checksum() {
        let mut arena = vec![0u8; CNN_ARENA_BYTES];
        let mut bufs = carve_cnn_into(&mut arena).unwrap();
        let mut scratch = [0u8; 4096];

        fill_pattern_cnn(bufs.input, bufs.l1w, bufs.l1b, bufs.l3w, bufs.l3b, bufs.l4w, bufs.l4b);
        run_cnn_ref(&mut bufs, &mut scratch).unwrap();
        let ref_chk = fnv1a(bufs.out);
        let ref_layers = layer_checksums(&bufs);

        fill_pattern_cnn(bufs.input, bufs.l1w, bufs.l1b, bufs.l3w, bufs.l3b, bufs.l4w, bufs.l4b);
        run_cnn_s3(&mut bufs, &mut scratch).unwrap();
        let s3_chk = fnv1a(bufs.out);
        let s3_layers = layer_checksums(&bufs);

        assert_eq!(ref_chk, s3_chk, "s3 model output must equal ref bit-exact");
        assert_eq!(ref_layers.l1, s3_layers.l1);
        assert_eq!(ref_layers.l2, s3_layers.l2);
        assert_eq!(ref_layers.l3, s3_layers.l3);
        assert_eq!(ref_layers.out, s3_layers.out);

        // Device-verified ESP-NN baseline output checksum.
        assert_eq!(
            s3_chk, 0x75eb32f5,
            "model final output must match the ESP-NN baseline 0x75eb32f5 (got 0x{:08x})",
            s3_chk
        );
    }
}
