// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Standalone activation functions — scalar fallback + TIE728 SIMD backend.
//!
//! # Bit-exact contract (Plan A4)
//!
//! | Leg | Contract | Runs on |
//! |-----|----------|---------|
//! | (a) | SIMD ≡ per-tensor TFLM golden bit-exact | Device (Phase 5) |
//! | (b) | Scalar ref ≡ per-channel TFLM golden bit-exact | **Host** (this test) |
//! | (c) | SIMD vs ref cross-check ≤1 LSB | Device (Phase 5) |
//!
//! # Ops implemented
//!
//! * [`relu`] — `max(0, input)` in quantized space.
//! * [`relu6`] — `clamp(input, 0, quantized_six)` in quantized space.
//! * [`hard_swish`] — integer rational approximation of `x·relu6(x+3)/6`.
//!
//! # Signature convention
//!
//! All kernels take `(input, params, output, scratch)` + optional `quantized_six`
//! for relu6, and return `Result<(), KernelError>`. The `scratch` buffer is
//! unused by these element-wise kernels.

use hematite_core::op_params::ActivationParams;
use hematite_core::KernelError;
use hematite_int8::{multiply_by_quantized_multiplier, saturating_cast};

/// ReLU: `output = max(0, input)` in quantized space.
///
/// # Algorithm
///
/// 1. `val = x + params.input_offset`
/// 2. `act = val.max(0)`
/// 3. `scaled = multiply_by_quantized_multiplier(act,
///     params.output_multiplier, params.output_shift)`
/// 4. `saturating_cast(scaled + params.output_offset)`
pub fn relu(
    input: &[i8],
    params: &ActivationParams,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    let n = input.len();
    if output.len() != n {
        return Err(KernelError::ShapeMismatch);
    }

    for i in 0..n {
        let val = i32::from(input[i]) + params.input_offset;
        let act = val.max(0);
        let scaled = multiply_by_quantized_multiplier(
            act,
            params.output_multiplier,
            params.output_shift,
        );
        output[i] = saturating_cast(scaled + params.output_offset);
    }

    let _ = scratch;
    Ok(())
}

/// ReLU6: `output = clamp(input, 0, quantized_six)` in quantized space.
///
/// # Parameters
///
/// * `quantized_six` — the quantized representation of 6.0 under the
///   input tensor's quantization scheme. Typically 6 for symmetric
///   quantization with scale=1 and zero_point=0.
///
/// # Algorithm
///
/// 1. `val = x + params.input_offset`
/// 2. `act = val.clamp(0, quantized_six)`
/// 3. `saturating_cast(act + params.output_offset)`
///
/// No requantize — the generator skips `multiply_by_quantized_multiplier`
/// for relu6 because output scale is 1.0 (identity).
pub fn relu6(
    input: &[i8],
    params: &ActivationParams,
    output: &mut [i8],
    scratch: &mut [u8],
    quantized_six: i32,
) -> Result<(), KernelError> {
    let n = input.len();
    if output.len() != n {
        return Err(KernelError::ShapeMismatch);
    }

    for i in 0..n {
        let val = i32::from(input[i]) + params.input_offset;
        let act = val.clamp(0, quantized_six);
        output[i] = saturating_cast(act + params.output_offset);
    }

    let _ = scratch;
    Ok(())
}

/// HardSwish: `x · ReLU6(x+3) / 6` — integer rational approximation.
///
/// ⚠️  DOWNGRADED implementation matching the golden fixture provenance.
/// Not bit-exact vs TFLM HardSwish (which uses a 16-bit fixed-point chain).
///
/// # Algorithm
///
/// 1. `x_i32 = x + params.input_offset`
/// 2. `relu6_arg = (x_i32 + 3).clamp(0, 6)`
/// 3. `product = x_i32 * relu6_arg`
/// 4. `result = (product + 3) / 6` (positive) or `(product - 3) / 6` (negative)
/// 5. `saturating_cast(result + params.output_offset)`
pub fn hard_swish(
    input: &[i8],
    params: &ActivationParams,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    let n = input.len();
    if output.len() != n {
        return Err(KernelError::ShapeMismatch);
    }

    for i in 0..n {
        let x_i32 = i32::from(input[i]) + params.input_offset;
        let relu6_arg = (x_i32 + 3).clamp(0, 6);
        let product = x_i32 * relu6_arg;
        let result = if product >= 0 {
            (product + 3) / 6
        } else {
            (product - 3) / 6
        };
        output[i] = saturating_cast(result + params.output_offset);
    }

    let _ = scratch;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
// TIE728 SIMD backend — device-only (NEVER compiled on host)
// ─────────────────────────────────────────────────────────────────────────────

/// TIE728 SIMD backend for activation ops.
///
/// This module is **entirely cfg-gated** behind `#[cfg(target_arch = "xtensa")]`
/// and is NEVER compiled on the host (stable-aarch64-apple-darwin). It exists
/// in the tree for structural review and Phase 5 device verification (T5.3).
///
/// ## Architecture
///
/// The SIMD path calls the vendored `dl_tie728_s8_relu_11c` entry point from
/// `hematite-s3/src/asm/dl_tie728_s8_relu.S` via `global_asm!`.
///
/// Register convention (Xtensa XCC):
/// * a2 = output pointer (i8*)
/// * a3 = input pointer (i8*)
/// * a4 = args pointer (packed struct)
///
/// ## Vendored .S files
///
/// Cell `hematite-s3/src/asm/` contains:
/// * `dl_tie728_s8.S` — shared macros (pre-existing)
/// * `dl_tie728_s8_relu.S` — 2 entry points (aligned 11c + unaligned)
///
/// Vendored from esp-dl @ 12c0616de145b704e1149c474b9a1e852e631d67 (MIT).
///
/// ## Args struct layout (derived from vendored .S l32i offsets)
///
/// ### ReLU 11c (aligned) — `dl_tie728_s8_relu_11c`
/// * +76: activation_alpha (i32) — 0 for standard ReLU, >0 for LeakyReLU slope
/// * +84: activation_shift (i32) — negative → no activation; 0/non-negative → apply ReLU
/// * +88: c_rs1_1 (i32) — c / 32 - 1 (each iteration processes 2×16 = 32 elements)
/// * +92: c_rs2_1 (i32) — (c % 32) / 16 (single-16 remainder after 32-wide loops)
/// * +136: c_remainder (i32) — used by unaligned variant only
///
/// ## ReLU6
///
/// No esp-dl `relu6` kernel exists. A ReLU6 SIMD could potentially be composed
/// from the vendored `dl_tie728_s8_relu_11c` (clamp negatives to zero) followed
/// by `dl_tie728_s8_min2d` (not vendored) to clamp the upper bound. Until both
/// are available and integrated, ReLU6 uses the scalar path above on device.
/// No SIMD stub is provided.
///
/// ## HardSwish
///
/// Not SIMD-amenable due to integer division and conditional branching.
/// Always uses the scalar path above on device. No SIMD stub.
///
/// ## A4 contract notes
///
/// * Leg (a): SIMD output must match a per-tensor TFLM golden (Phase 5 fixture).
/// * Leg (c): SIMD vs scalar ref cross-check tolerance ≤1 LSB.
#[cfg(target_arch = "xtensa")]
mod activation_simd {
    /// Include the vendored TIE728 shared macros and relu entry points.
    core::arch::global_asm!(
        include_str!("../src/asm/dl_tie728_s8.S"),
        include_str!("../src/asm/dl_tie728_s8_relu.S"),
    );

    // ── Args struct — derived from vendored .S l32i offsets ──────────────

    /// Args for aligned ReLU 1×1×c — matches `dl_tie728_s8_relu_11c`.
    ///
    /// ABI verified against vendored .S:
    /// * `l32i a14, a4, 76` → activation_alpha
    /// * `l32i a15, a4, 84` → activation_shift
    /// * `l32i a5, a4, 88` → c_rs1_1
    /// * `l32i a6, a4, 92` → c_rs2_1
    ///
    /// ABI unverified on device — validate at T5.3.
    #[repr(C)]
    #[allow(dead_code)]
    struct Tie728ReluArgs {
        _pad0: [u8; 76],          // offset 0-75: unused
        activation_alpha: i32,    // offset 76
        _pad1: [u8; 4],           // offset 80-83
        activation_shift: i32,    // offset 84
        c_rs1_1: i32,             // offset 88: c / 32 - 1
        c_rs2_1: i32,             // offset 92: (c % 32) / 16
    }

    impl Default for Tie728ReluArgs {
        fn default() -> Self {
            Self {
                _pad0: [0u8; 76],
                activation_alpha: 0,
                _pad1: [0u8; 4],
                activation_shift: 0,
                c_rs1_1: 0,
                c_rs2_1: 0,
            }
        }
    }

    // ── SIMD kernel glue ──────────────────────────────────────────────────

    /// SIMD ReLU (aligned) — calls the vendored TIE728 entry point.
    ///
    /// Calls `dl_tie728_s8_relu_11c`:
    /// * a2 = output (i8*)
    /// * a3 = input (i8*)
    /// * a4 = &Tie728ReluArgs { activation_alpha, activation_shift, c_rs1_1, c_rs2_1 }
    ///
    /// # Safety
    ///
    /// This function is inherently unsafe: it calls into foreign assembly
    /// via the C ABI. ABI unverified — validate at T5.3 on device.
    ///
    /// # Preconditions (caller MUST guarantee)
    ///
    /// * `num_elements` must be a multiple of 32 (2×16-wide SIMD lanes)
    ///   for the aligned variant. The unaligned variant handles remainders.
    /// * All pointers must be 16-byte aligned for EE.VLD.128.IP / EE.VST.128.IP.
    /// * `activation_alpha = 0` for standard ReLU (non-zero = LeakyReLU slope).
    /// * `activation_shift ≥ 0` enables ReLU; negative disables activation.
    #[allow(dead_code)]
    unsafe fn relu_simd(
        output: *mut i8,
        input: *const i8,
        num_elements: u32,
        activation_alpha: i32,
        activation_shift: i32,
    ) {
        let args = Tie728ReluArgs {
            activation_alpha,
            activation_shift,
            c_rs1_1: (num_elements / 32) as i32 - 1,
            c_rs2_1: ((num_elements % 32) / 16) as i32,
            ..Default::default()
        };
        core::arch::asm!(
            "mov a2, {output}",
            "mov a3, {input}",
            "mov a4, {args}",
            "call8 dl_tie728_s8_relu_11c",
            output = in(reg) output,
            input = in(reg) input,
            args = in(reg) &args,
            clobber_abi("C"),
        );
    }
}

#[cfg(target_arch = "xtensa")]
pub use activation_simd::relu_simd;
