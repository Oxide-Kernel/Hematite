// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Fit-size FC model — proves flash→SRAM weight staging on device.
//!
//! A chain of four fully-connected layers whose weights are declared as
//! immutable `static` consts, which the Xtensa linker places in flash-backed
//! DROM (0x3c...). The ACCX fc dispatch (`gemm::fully_connected` →
//! `fc_accx_dispatch`) detects a DROM weights pointer and stages each distinct
//! layer's weights into the caller's persistent scratch buffer ONCE (on the
//! warmup predict call, which is untimed); the timed calls then read the SRAM
//! copy instead of streaming flash — the measured win is ~96x (2.5 M cyc for an
//! 80 KiB DROM stream vs 26 K from SRAM on a 640→128 layer).
//!
//! Total weights here: 32 KiB + 16 KiB + 8 KiB + 1 KiB = 57 KiB, comfortably
//! inside a 64 KiB scratch region. The board has no PSRAM, so this is the
//! honest "fit-size" demonstration of the staging feature (the anomaly model's
//! 258 KiB of weights cannot be staged on this board).
//!
//! The 4-layer graph:
//!
//! ```text
//!   L1: fc 256 → 128     W1 [128][256] = 32 KiB
//!   L2: fc 128 → 128     W2 [128][128] = 16 KiB
//!   L3: fc 128 → 64      W3 [64][128]  = 8 KiB
//!   L4: fc 64  → 16      W4 [16][64]   = 1 KiB
//! ```
//!
//! All layers use `input_offset 0` (no wsum fold), uniform scale
//! (`mult = 1<<30`, `shift = 0`), full-range activation. Deterministic fill
//! matches `spec.rs`: input `i*7+3`, weights `i*13+11` (the static consts here
//! are a fixed nonzero pattern; both ref and s3 read the SAME bytes, so
//! bit-exactness holds regardless of the value), bias `i*17-8`.

use hematite_core::op_params::FullyConnectedParams;

pub const L1_IN: usize = 256;
pub const L1_OUT: usize = 128;
pub const L2_OUT: usize = 128;
pub const L3_OUT: usize = 64;
pub const L4_OUT: usize = 16;

/// Weights — immutable statics → flash-backed DROM (the staging target).
/// Nonzero const init (7) so the linker keeps them in `.rodata`/DROM rather
/// than zero-initializing into SRAM `.bss`.
pub static W1: [i8; L1_IN * L1_OUT] = [7; L1_IN * L1_OUT];
pub static W2: [i8; L1_OUT * L2_OUT] = [7; L1_OUT * L2_OUT];
pub static W3: [i8; L2_OUT * L3_OUT] = [7; L2_OUT * L3_OUT];
pub static W4: [i8; L3_OUT * L4_OUT] = [7; L3_OUT * L4_OUT];

/// Biases (small; SRAM placement is fine).
pub static B1: [i32; L1_OUT] = [0; L1_OUT];
pub static B2: [i32; L2_OUT] = [0; L2_OUT];
pub static B3: [i32; L3_OUT] = [0; L3_OUT];
pub static B4: [i32; L4_OUT] = [0; L4_OUT];

pub static MULT_L1: [i32; L1_OUT] = [1 << 30; L1_OUT];
pub static SHIFT_L1: [i32; L1_OUT] = [0; L1_OUT];
pub static MULT_L2: [i32; L2_OUT] = [1 << 30; L2_OUT];
pub static SHIFT_L2: [i32; L2_OUT] = [0; L2_OUT];
pub static MULT_L3: [i32; L3_OUT] = [1 << 30; L3_OUT];
pub static SHIFT_L3: [i32; L3_OUT] = [0; L3_OUT];
pub static MULT_L4: [i32; L4_OUT] = [1 << 30; L4_OUT];
pub static SHIFT_L4: [i32; L4_OUT] = [0; L4_OUT];

pub static L1_PARAMS: FullyConnectedParams<'static> = FullyConnectedParams {
    input_dim: L1_IN as i32,
    output_dim: L1_OUT as i32,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_L1,
    output_shift_per_channel: &SHIFT_L1,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};
pub static L2_PARAMS: FullyConnectedParams<'static> = FullyConnectedParams {
    input_dim: L1_OUT as i32,
    output_dim: L2_OUT as i32,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_L2,
    output_shift_per_channel: &SHIFT_L2,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};
pub static L3_PARAMS: FullyConnectedParams<'static> = FullyConnectedParams {
    input_dim: L2_OUT as i32,
    output_dim: L3_OUT as i32,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_L3,
    output_shift_per_channel: &SHIFT_L3,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};
pub static L4_PARAMS: FullyConnectedParams<'static> = FullyConnectedParams {
    input_dim: L3_OUT as i32,
    output_dim: L4_OUT as i32,
    input_offset: 0,
    weights_offset: 0,
    output_offset: 0,
    output_multiplier_per_channel: &MULT_L4,
    output_shift_per_channel: &SHIFT_L4,
    quantized_activation_min: -128,
    quantized_activation_max: 127,
};

/// Total bytes needed for the activation tensors + SRAM weight staging
/// buffers + scratch region.
pub const FIT_ARENA_BYTES: usize = L1_IN
    + L1_OUT
    + L2_OUT
    + L3_OUT
    + L4_OUT
    + (L1_IN * L1_OUT)
    + (L1_OUT * L2_OUT)
    + (L2_OUT * L3_OUT)
    + (L3_OUT * L4_OUT)
    + 64 * 1024;

/// Carved activation/scratch buffers (activations first, then the SRAM weight
/// staging buffers, then scratch).
pub struct FitBuffers<'a> {
    pub input: &'a mut [i8],
    pub l1: &'a mut [i8],
    pub l2: &'a mut [i8],
    pub l3: &'a mut [i8],
    pub out: &'a mut [i8],
    /// SRAM copies of the DROM consts W1..W4 — filled once by
    /// [`stage_fit_weights`] before the timed loop (the "weights resident in
    /// SRAM at model load" demonstration).
    pub sw1: &'a mut [i8],
    pub sw2: &'a mut [i8],
    pub sw3: &'a mut [i8],
    pub sw4: &'a mut [i8],
    pub scratch: &'a mut [u8],
}

/// Sequential split of `arena` into the activation tensors + SRAM weight
/// staging buffers + scratch. All lengths are 16-aligned (arena base is
/// 16-aligned, every split size is a 16-multiple), so the kernel's `VLD.128`
/// alignment holds for the staged weights too. Returns `Err` when too small.
pub fn carve_fit_into(arena: &mut [u8]) -> Result<FitBuffers<'_>, hematite_core::KernelError> {
    use hematite_core::KernelError;
    unsafe fn cast_i8(s: &mut [u8]) -> &mut [i8] {
        core::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut i8, s.len())
    }
    let (input, rest) = arena.split_at_mut(L1_IN);
    let (l1, rest) = rest.split_at_mut(L1_OUT);
    let (l2, rest) = rest.split_at_mut(L2_OUT);
    let (l3, rest) = rest.split_at_mut(L3_OUT);
    let (out, rest) = rest.split_at_mut(L4_OUT);
    let (sw1, rest) = rest.split_at_mut(L1_IN * L1_OUT);
    let (sw2, rest) = rest.split_at_mut(L1_OUT * L2_OUT);
    let (sw3, rest) = rest.split_at_mut(L2_OUT * L3_OUT);
    let (sw4, scratch) = rest.split_at_mut(L3_OUT * L4_OUT);
    if scratch.len() < 64 * 1024 {
        return Err(KernelError::ShapeMismatch);
    }
    Ok(FitBuffers {
        input: unsafe { cast_i8(input) },
        l1: unsafe { cast_i8(l1) },
        l2: unsafe { cast_i8(l2) },
        l3: unsafe { cast_i8(l3) },
        out: unsafe { cast_i8(out) },
        sw1: unsafe { cast_i8(sw1) },
        sw2: unsafe { cast_i8(sw2) },
        sw3: unsafe { cast_i8(sw3) },
        sw4: unsafe { cast_i8(sw4) },
        scratch,
    })
}

/// Deterministic input fill matching the spec.rs pattern.
pub fn fill_fit_input(input: &mut [i8]) {
    for (i, v) in input.iter_mut().enumerate() {
        *v = ((i * 7 + 3) & 0xFF) as i8;
    }
}

/// Copy the DROM consts W1..W4 into the SRAM staging buffers — the once-at-
/// model-load weight residency step that the timing loop then reads from.
/// Returns a 4-tuple of the SRAM weight slices for the runner.
pub fn stage_fit_weights<'a>(
    b: &'a mut FitBuffers<'_>,
) -> (&'a mut [i8], &'a mut [i8], &'a mut [i8], &'a mut [i8]) {
    b.sw1.copy_from_slice(&W1);
    b.sw2.copy_from_slice(&W2);
    b.sw3.copy_from_slice(&W3);
    b.sw4.copy_from_slice(&W4);
    (b.sw1, b.sw2, b.sw3, b.sw4)
}

/// Run the 4-layer chain via hematite-s3 (ACCX SIMD on device, scalar on host),
/// reading the DROM consts directly (the flash-bound control).
pub fn run_fit_s3(b: &mut FitBuffers<'_>) -> Result<(), hematite_core::KernelError> {
    hematite_s3::gemm::fully_connected(b.input, &W1, &B1, &L1_PARAMS, b.l1, b.scratch)?;
    hematite_s3::gemm::fully_connected(b.l1, &W2, &B2, &L2_PARAMS, b.l2, b.scratch)?;
    hematite_s3::gemm::fully_connected(b.l2, &W3, &B3, &L3_PARAMS, b.l3, b.scratch)?;
    hematite_s3::gemm::fully_connected(b.l3, &W4, &B4, &L4_PARAMS, b.out, b.scratch)
}

/// Run the 4-layer chain via hematite-s3 with the weights taken from the SRAM
/// staging buffers (the staged/accelerated path — same bytes as W1..W4).
pub fn run_fit_s3_staged(b: &mut FitBuffers<'_>) -> Result<(), hematite_core::KernelError> {
    hematite_s3::gemm::fully_connected(b.input, b.sw1, &B1, &L1_PARAMS, b.l1, b.scratch)?;
    hematite_s3::gemm::fully_connected(b.l1, b.sw2, &B2, &L2_PARAMS, b.l2, b.scratch)?;
    hematite_s3::gemm::fully_connected(b.l2, b.sw3, &B3, &L3_PARAMS, b.l3, b.scratch)?;
    hematite_s3::gemm::fully_connected(b.l3, b.sw4, &B4, &L4_PARAMS, b.out, b.scratch)
}

/// Run the 4-layer chain via hematite-ref (scalar reference).
pub fn run_fit_ref(b: &mut FitBuffers<'_>) -> Result<(), hematite_core::KernelError> {
    hematite_ref::fully_connected::fully_connected(b.input, &W1, &B1, &L1_PARAMS, b.l1, b.scratch)?;
    hematite_ref::fully_connected::fully_connected(b.l1, &W2, &B2, &L2_PARAMS, b.l2, b.scratch)?;
    hematite_ref::fully_connected::fully_connected(b.l2, &W3, &B3, &L3_PARAMS, b.l3, b.scratch)?;
    hematite_ref::fully_connected::fully_connected(b.l3, &W4, &B4, &L4_PARAMS, b.out, b.scratch)
}

/// FNV-1a over i8 data (sign-extending — matches firmware.rs fnv1a).
pub fn fnv1a(data: &[i8]) -> u32 {
    let mut h: u32 = 2166136261;
    for &b in data {
        h ^= b as u32;
        h = h.wrapping_mul(16777619);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_model_ref_matches_s3_bit_exact() {
        let mut arena = vec![0u8; FIT_ARENA_BYTES];
        let mut b = carve_fit_into(&mut arena).unwrap();
        fill_fit_input(b.input);
        run_fit_ref(&mut b).unwrap();
        let ref_fnv = fnv1a(b.out);
        fill_fit_input(b.input);
        run_fit_s3(&mut b).unwrap();
        assert_eq!(fnv1a(b.out), ref_fnv, "fit model s3 output must match ref");
        // The staged (SRAM-weight) path must be bit-identical to ref too.
        fill_fit_input(b.input);
        let _ = stage_fit_weights(&mut b);
        run_fit_s3_staged(&mut b).unwrap();
        assert_eq!(
            fnv1a(b.out),
            ref_fnv,
            "fit model staged-s3 output must match ref"
        );
    }
}
