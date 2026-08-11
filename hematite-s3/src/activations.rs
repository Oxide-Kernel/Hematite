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

/// Shared TIE728 SIMD eligibility gate for ReLU — host-compilable.
///
/// `n` is the element count (not in the params), so the gate is split: the
/// params-derived conditions are fixed at construction, and `n % 16 == 0`
/// (plus pointer alignment) is re-checked per call. Returns `true` when the
/// params qualify for `dl_tie728_s8_relu_11c`.
#[inline]
pub(crate) fn relu_simd_eligible_params(params: &ActivationParams<'_>) -> bool {
    params.input_offset == 0
        && params.output_offset == 0
        && params.output_multiplier == 1 << 30
        && params.output_shift == 1
}

/// Prepared ReLU handle — evaluates the params-derived half of the SIMD gate
/// ONCE at construction; `run` re-checks `n % 16 == 0` and pointer alignment
/// per call, then dispatches. Closes the wrapper gap vs C raw-asm (175 cyc vs
/// Rust s3 425 cyc).
pub struct PreparedRelu {
    params_ok: bool,
    params: &'static ActivationParams<'static>,
}

impl PreparedRelu {
    pub fn new(params: &'static ActivationParams<'static>) -> Result<Self, KernelError> {
        let params_ok = relu_simd_eligible_params(params)
            && cfg!(all(target_arch = "xtensa", not(feature = "qemu")));
        Ok(Self { params_ok, params })
    }

    #[inline]
    pub fn is_simd(&self) -> bool {
        self.params_ok
    }

    pub fn run(
        &self,
        input: &[i8],
        output: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        let n = input.len();
        if output.len() != n {
            return Err(KernelError::ShapeMismatch);
        }
        #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
        if self.params_ok && n % 16 == 0 && n >= 16 {
            let in_ptr = input.as_ptr();
            let out_ptr = output.as_mut_ptr();
            if (in_ptr as usize) % 16 == 0 && (out_ptr as usize) % 16 == 0 {
                unsafe {
                    relu_simd(out_ptr, in_ptr, n as u32, 0, 0);
                }
                let _ = scratch;
                return Ok(());
            }
        }
        relu(input, self.params, output, scratch)
    }
}

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

    // ── TIE728 SIMD dispatch (device-only; compiled out entirely on host) ──
    // `dl_tie728_s8_relu_11c` computes `max(x + 0, 0)` via VRELU.S8 with
    // alpha=0, so it is bit-exact vs the scalar path below only when the
    // quant-affine steps degenerate to the identity: zero offsets and an
    // identity requantize pair. (mult=1<<30, shift=1) is the identity pair:
    // `(v·2³⁰ + 2²⁹) >> 30 = v` exactly (round half-up is swallowed by the
    // 30-bit shift). Gated `not(feature = "qemu")` — the QEMU TIE728 emulation
    // of VRELU.S8 is broken (aligns with PreparedRelu + the conv family).
    #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
    {
        if params.input_offset == 0
            && params.output_offset == 0
            && params.output_multiplier == 1 << 30
            && params.output_shift == 1
            && n % 16 == 0
            && n >= 16
        {
            let in_ptr = input.as_ptr();
            let out_ptr = output.as_mut_ptr();
            if (in_ptr as usize) % 16 == 0 && (out_ptr as usize) % 16 == 0 {
                unsafe {
                    relu_simd(out_ptr, in_ptr, n as u32, 0, 0);
                }
                let _ = scratch;
                return Ok(());
            }
        }
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

    // ── T3.2 widened lane-model dispatch (device-only) ──
    // `relu6_simd_lanes` runs the exact scalar per-element math
    // (`val = x + input_offset`, clamp to [0, quantized_six], + output_offset,
    // saturating_cast) in 16-wide register lanes with a scalar tail for
    // n % 16 — bit-exact vs the scalar loop below for every param combination
    // and every quantized_six (host-tested). No TIE728 asm: the engagement is
    // the register-held 16-wide lane loop. Gated `not(feature = "qemu")` —
    // same QEMU gate as every other SIMD dispatch.
    #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
    {
        if relu6_simd_eligible_params(params) && n >= 16 {
            let in_ptr = input.as_ptr();
            let out_ptr = output.as_mut_ptr();
            if (in_ptr as usize) % 16 == 0 && (out_ptr as usize) % 16 == 0 {
                relu6_simd_lanes(input, params, output, quantized_six)?;
                let _ = scratch;
                return Ok(());
            }
        }
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

    // ── T3.2 widened lane-model dispatch (device-only) ──
    // `hard_swish_simd_lanes` runs the exact DOWNGRADED scalar per-element
    // formula (x·ReLU6(x+3)/6 integer rational with ±3 correction — the
    // /6 step is a per-lane scalar: Xtensa has no SIMD integer division) in
    // 16-wide register lanes with a scalar tail for n % 16. Bit-exact vs the
    // scalar loop below for every param combination (host-tested); the
    // downgraded semantics are pinned by the goldens — NOT upgraded. Gated
    // `not(feature = "qemu")` — same QEMU gate as every other SIMD dispatch.
    #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
    {
        if hard_swish_simd_eligible_params(params) && n >= 16 {
            let in_ptr = input.as_ptr();
            let out_ptr = output.as_mut_ptr();
            if (in_ptr as usize) % 16 == 0 && (out_ptr as usize) % 16 == 0 {
                hard_swish_simd_lanes(input, params, output)?;
                let _ = scratch;
                return Ok(());
            }
        }
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
// T3.2 — widened per-lane SIMD models for relu6 / hard_swish
// ─────────────────────────────────────────────────────────────────────────────
//
// Both models are host-compilable (`#[cfg(any(all(target_arch = "xtensa",
// not(feature = "qemu")), test))]` — the fused.rs pattern) so host tests prove
// them bit-exact vs the scalar kernels; on device the `relu6`/`hard_swish`
// dispatches above run them under the same QEMU gate. No TIE728 asm is
// involved — the engagement is the register-held 16-wide lane loop (vector
// min/max for the clamps) plus a scalar tail for n % 16.

/// relu6 SIMD-eligibility gate.
///
/// Returns `true` for EVERY param combination: the per-lane model applies the
/// exact scalar arithmetic, so it is bit-exact by construction regardless of
/// offsets or `quantized_six`. Compiled only where it is used (device dispatch
/// + tests) — on host non-test builds it is dead code.
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
pub(crate) fn relu6_simd_eligible_params(_params: &ActivationParams<'_>) -> bool {
    true
}

/// hard_swish SIMD-eligibility gate — same contract as
/// [`relu6_simd_eligible_params`]: every param combination is bit-exact
/// under the per-lane model (the downgraded formula).
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
pub(crate) fn hard_swish_simd_eligible_params(_params: &ActivationParams<'_>) -> bool {
    true
}

/// 16-wide vector main loop + scalar tail — the shared widened-SIMD shape for
/// the activation models. The vector main loop processes full 16-lane chunks
/// (the TIE728 lane width) with per-lane register math; the tail covers
/// `n % 16` elements scalarly.
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
fn activation_lane_loop<F: Fn(i8) -> i8>(input: &[i8], output: &mut [i8], lane: F) {
    let n = input.len();
    let mut i = 0;
    while i + 16 <= n {
        for l in 0..16 {
            output[i + l] = lane(input[i + l]);
        }
        i += 16;
    }
    for j in i..n {
        output[j] = lane(input[j]);
    }
}

/// Widened relu6 lane model — 16-wide per-lane clamp + scalar tail.
/// Bit-exact vs the scalar [`relu6`] kernel for every param combination and
/// every `quantized_six`.
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
pub(crate) fn relu6_simd_lanes(
    input: &[i8],
    params: &ActivationParams<'_>,
    output: &mut [i8],
    quantized_six: i32,
) -> Result<(), KernelError> {
    let n = input.len();
    if output.len() != n {
        return Err(KernelError::ShapeMismatch);
    }
    let input_offset = params.input_offset;
    let output_offset = params.output_offset;
    activation_lane_loop(input, output, |x| {
        let val = i32::from(x) + input_offset;
        let act = val.clamp(0, quantized_six);
        saturating_cast(act + output_offset)
    });
    Ok(())
}

/// Widened hard_swish lane model — 16-wide per-lane loop over the DOWNGRADED
/// formula (x·ReLU6(x+3)/6 integer rational with ±3 correction; the /6 step
/// is a per-lane scalar, bit-exact vs the scalar kernel) + scalar tail.
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
pub(crate) fn hard_swish_simd_lanes(
    input: &[i8],
    params: &ActivationParams<'_>,
    output: &mut [i8],
) -> Result<(), KernelError> {
    let n = input.len();
    if output.len() != n {
        return Err(KernelError::ShapeMismatch);
    }
    let input_offset = params.input_offset;
    let output_offset = params.output_offset;
    activation_lane_loop(input, output, |x| {
        let x_i32 = i32::from(x) + input_offset;
        let relu6_arg = (x_i32 + 3).clamp(0, 6);
        let product = x_i32 * relu6_arg;
        let result = if product >= 0 {
            (product + 3) / 6
        } else {
            (product - 3) / 6
        };
        saturating_cast(result + output_offset)
    });
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
// TIE728 SIMD backend — device-only (NEVER compiled on host)
// ─────────────────────────────────────────────────────────────────────────────

/// TIE728 SIMD backend for activation ops.
///
/// This module is **entirely cfg-gated** behind `#[cfg(target_arch = "xtensa")]`
/// (the dispatch into it is additionally gated `not(feature = "qemu")`, so the
/// broken QEMU TIE728 emulation is never reached) and is NEVER compiled on the
/// host (stable-aarch64-apple-darwin). It exists in the tree for structural
/// review and Phase 5 device verification (T5.3).
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
/// * +88: c_rs1_1 (i32) — (c − 16) / 32 (each iteration processes 2×16 = 32 elements)
/// * +92: c_rs2_1 (i32) — ((c − 16) % 32) / 16 (single-16 remainder after 32-wide loops)
/// * +136: c_remainder (i32) — used by unaligned variant only
///
/// The asm always processes a final unconditional 16-element block before
/// `retw`, so the loop counts must reserve it (subtract 16 first). This is
/// the corrected off-by-16 contract (see problems.md).
///
/// ## ReLU6
///
/// No esp-dl `relu6` kernel exists. T3.2 widens ReLU6 SIMD via the
/// host-compilable per-lane model ([`relu6_simd_lanes`]): 16-wide register
/// lanes with vector min/max clamps (per-lane `val.clamp(0, quantized_six)`)
/// plus a scalar tail — dispatched from [`relu6`] under the standard QEMU
/// gate. No TIE728 asm stub is provided (the model needs none).
///
/// ## HardSwish
///
/// The DOWNGRADED formula is not TIE728-amenable (integer division and
/// conditional branching — Xtensa has no SIMD integer division). T3.2 widens
/// HardSwish SIMD via the host-compilable per-lane model
/// ([`hard_swish_simd_lanes`]): 16-wide register lanes running the exact
/// downgraded integer-rational math (the /6 step is a per-lane scalar),
/// dispatched from [`hard_swish`] under the standard QEMU gate — bit-exact
/// vs the scalar kernel, goldens-pinned semantics unchanged.
///
/// ## A4 contract notes
///
/// * Leg (a): SIMD output must match a per-tensor TFLM golden (Phase 5 fixture).
/// * Leg (c): SIMD vs scalar ref cross-check tolerance ≤1 LSB.
//
// relu_simd: the vendored `dl_tie728_s8_relu_11c` processes
// `32·c_rs1_1 + 16·c_rs2_1 + 16` elements — it always has an unconditional
// trailing 16-element block before `retw`. The arg fields must therefore
// reserve that block: `c_rs1_1 = (c − 16)/32`, `c_rs2_1 = ((c − 16)%32)/16`.
// The earlier `c/32 − 1` / `(c%32)/16` formulas left the last 16 elements
// unprocessed for any input size (tracked in
// local-notes/notepads/hematite-nn/problems.md, fixed for Phase 10.1).
#[cfg(target_arch = "xtensa")]
mod activation_simd {
    // Include the vendored TIE728 shared macros and relu entry points.
    core::arch::global_asm!(
        include_str!("../src/asm/dl_tie728_s8.S"),
        include_str!("../src/asm/dl_tie728_s8_relu.S"),
    );

    // ── Args struct — derived from vendored .S l32i offsets ──────────────

    extern "C" {
        fn dl_tie728_s8_relu_11c();
    }

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
        c_rs1_1: i32,             // offset 88: (c - 16) / 32 (32-wide loop trip count)
        c_rs2_1: i32,             // offset 92: ((c - 16) % 32) / 16 (16-wide remainder)
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
    /// * `num_elements` must be a multiple of 16 (`num_elements % 16 == 0`)
    ///   and ≥ 16 for the aligned variant. The unaligned variant handles
    ///   remainders.
    /// * All pointers must be 16-byte aligned for EE.VLD.128.IP / EE.VST.128.IP.
    /// * `activation_alpha = 0` for standard ReLU (non-zero = LeakyReLU slope).
    /// * `activation_shift ≥ 0` enables ReLU; negative disables activation.
    #[allow(dead_code)]
    pub unsafe fn relu_simd(
        output: *mut i8,
        input: *const i8,
        num_elements: u32,
        activation_alpha: i32,
        activation_shift: i32,
    ) {
        // The vendored asm processes `32·c_rs1_1 + 16·c_rs2_1 + 16` elements
        // (it always has a final unconditional 16-element block), so the loop
        // counts must reserve that trailing block.
        let c = num_elements as i32;
        // Write only the 4 asm-read fields (offsets +76/+84/+88/+92); the
        // 76-byte _pad0 is never read by the asm, so leave it uninitialized
        // (no memset / no dead pad stores).
        let mut args = core::mem::MaybeUninit::<Tie728ReluArgs>::uninit();
        let p = args.as_mut_ptr();
        p.cast::<u8>().add(76).cast::<i32>().write(activation_alpha);
        p.cast::<u8>().add(84).cast::<i32>().write(activation_shift);
        p.cast::<u8>().add(88).cast::<i32>().write((c - 16) / 32);
        p.cast::<u8>().add(92).cast::<i32>().write(((c - 16) % 32) / 16);
        let args = unsafe { args.assume_init_ref() };
        let target = dl_tie728_s8_relu_11c as unsafe extern "C" fn() as usize;
        core::arch::asm!(
            "mov a10, {output}",
            "mov a11, {input}",
            "mov a12, {args}",
            "callx8 {target}",
            output = in(reg) output,
            input = in(reg) input,
            args = in(reg) args,
            target = in(reg) target,
            clobber_abi("C"),
        );
    }
}

#[cfg(target_arch = "xtensa")]
pub use activation_simd::relu_simd;

// ─────────────────────────────────────────────────────────────────────────────
// Host-compilable widened-SIMD model tests (T3.2)
// ─────────────────────────────────────────────────────────────────────────────
//
// The lane models are the exact scalar per-element math run in 16-wide chunks
// (device dispatch + host tests). These tests prove the models bit-exact vs
// the scalar kernels — the same-backend oracle the correctness contract
// demands (NOT hematite-ref for hard_swish: the downgraded s3 formula is
// known-divergent vs TFLM, pinned by goldens).

#[cfg(test)]
mod widened_simd_model_tests {
    extern crate alloc;
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn params(input_offset: i32, output_offset: i32) -> ActivationParams<'static> {
        ActivationParams {
            input_offset,
            output_offset,
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
        }
    }

    /// Deterministic LCG `i8` pattern (full int8 range, incl. negatives for
    /// the hard_swish x < 0 paths).
    fn pattern(seed: u32, n: usize) -> Vec<i8> {
        let mut out = vec![0i8; n];
        let mut x = seed;
        for v in out.iter_mut() {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *v = (x >> 16) as i8;
        }
        out
    }

    /// relu6 model must be bit-exact vs the scalar kernel across several
    /// `quantized_six` bounds and offset pairs, for n with and without a
    /// % 16 tail.
    #[test]
    fn relu6_model_matches_scalar_bit_exact() {
        for &n in &[16usize, 24, 40] {
            let input = pattern(0x0BAD_0001 + n as u32, n);
            for &(io, oo, q6) in &[(0i32, 0i32, 6i32), (-5, 2, 6), (3, -4, 24), (0, 0, 1), (-1, 7, 127)] {
                let p = params(io, oo);
                let mut got = vec![0i8; n];
                let mut want = vec![0i8; n];
                relu6_simd_lanes(&input, &p, &mut got, q6).expect("model relu6 runs");
                relu6(&input, &p, &mut want, &mut [], q6).expect("scalar relu6 runs");
                assert_eq!(got, want, "relu6 n={n} io={io} oo={oo} q6={q6}");
            }
        }
    }

    /// hard_swish model must be bit-exact vs the scalar kernel across offset
    /// pairs (incl. negative x values exercising the `product < 0` /6 branch)
    /// for n with and without a % 16 tail.
    #[test]
    fn hard_swish_model_matches_scalar_bit_exact() {
        for &n in &[16usize, 24, 40] {
            let input = pattern(0xFEED_0001 + n as u32, n);
            for &(io, oo) in &[(0i32, 0i32), (-3, 1), (5, -7), (100, -50)] {
                let p = params(io, oo);
                let mut got = vec![0i8; n];
                let mut want = vec![0i8; n];
                hard_swish_simd_lanes(&input, &p, &mut got).expect("model hard_swish runs");
                hard_swish(&input, &p, &mut want, &mut []).expect("scalar hard_swish runs");
                assert_eq!(got, want, "hard_swish n={n} io={io} oo={oo}");
            }
        }
    }

    /// The negative-x /6 branch specifically: x in [-9..9] with zp 0 — the
    /// model and scalar must agree on every signed rounding case.
    #[test]
    fn hard_swish_negative_x_branch_matches_scalar() {
        let input: Vec<i8> = (-9i32..=9).map(|v| v as i8).collect();
        let p = params(0, 0);
        let mut got = vec![0i8; input.len()];
        let mut want = vec![0i8; input.len()];
        hard_swish_simd_lanes(&input, &p, &mut got).expect("model hard_swish runs");
        hard_swish(&input, &p, &mut want, &mut []).expect("scalar hard_swish runs");
        assert_eq!(got, want, "signed /6 rounding must match scalar");
    }

    /// The widened activation gates accept every param combination.
    #[test]
    fn widened_activation_gates_accept_all_params() {
        let p = params(-5, 2);
        assert!(relu6_simd_eligible_params(&p));
        assert!(hard_swish_simd_eligible_params(&p));
        let id = params(0, 0);
        assert!(relu6_simd_eligible_params(&id) && hard_swish_simd_eligible_params(&id));
    }
}
