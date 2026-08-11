// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Elementwise operations — scalar fallback + TIE728 SIMD backend.
//!
//! # Bit-exact contract (Plan A4)
//!
//! | Leg | Contract | Runs on |
//! |-----|----------|---------|
//! | (a) | SIMD ≡ per-tensor TFLM golden bit-exact | Device (Phase 5) |
//! | (b) | **Scalar ref ≡ per-tensor TFLM golden bit-exact** | **Host** (this test) |
//! | (c) | SIMD vs ref cross-check ≤1 LSB on requantize | Device (Phase 5) |
//!
//! On host (stable-aarch64-apple-darwin), only leg (b) executes. The SIMD path
//! (`#[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]`) is NEVER
//! compiled on host and is additionally compiled out under `feature = "qemu"`
//! (the Espressif QEMU fork's TIE728 emulation of VADDS/VSUBS/VMULAS is broken
//! — crashes/silent wrong/hangs) — it exists in the tree for structural review
//! and Phase 5 device verification.
//!
//! # Binary ops (add, mul, sub)
//!
//! Follow the TFLM `reference_integer_ops` AddFunc / MulElementwise / Sub
//! formulations from `tensorflow/lite/kernels/internal/reference/integer_ops/`:
//!
//! * **Add / Sub**: Shifted per-input scaling → sum/diff → output requantize.
//! * **Mul**: Direct product → single output requantize (no left_shift,
//!   no per-input multiplier/shift).
//!
//! # SIMD dispatch (T3.2)
//!
//! Two device paths per op, tried in order: the raw TIE728 kernel
//! (`dl_tie728_s8_{add,sub,mul}_w1_16_w2_16`) under the **identity**
//! quant-affine contract (zero offsets, identity `(1<<30, 1)` pairs, full
//! range, n % 16), then the T3.2 **widened per-lane model** — a
//! host-compilable 16-wide register-lane loop that reproduces the exact
//! scalar per-element math (input offsets → left_shift scaling → conditional
//! per-input requantize → i32 sum/product → output requantize → offset →
//! clamp → saturating_cast) with a scalar tail for `n % 16`. The widened
//! model is bit-exact for EVERY param combination (host-tested below) and
//! engages for any n ≥ 16 with 16-aligned buffers.

use hematite_core::op_params::ElementwiseParams;
use hematite_core::KernelError;
use hematite_int8::{multiply_by_quantized_multiplier, saturating_cast};

/// Elementwise ADD — scalar kernel (host-compilable, bit-exact vs per-tensor golden).
///
/// Per-element `(input1 + input1_offset) + (input2 + input2_offset)` with
/// per-input rescaling (left_shift + multiplier/shift), then output requantize.
///
/// Matches TFLM `reference_integer_ops::AddFunc`.
///
/// # Errors
///
/// * [`KernelError::ShapeMismatch`] if `input1.len()`, `input2.len()`, or
///   `output.len()` does not equal `params.num_elements`.
pub fn add(
    input1: &[i8],
    input2: &[i8],
    params: &ElementwiseParams,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    let n = params.num_elements as usize;
    if input1.len() != n || input2.len() != n || output.len() != n {
        return Err(KernelError::ShapeMismatch);
    }

    // ── TIE728 SIMD dispatch (device-only; compiled out entirely on host) ──
    // Two device paths, tried in order:
    //
    // (1) Raw TIE728 `add_simd_aligned` (identity contract): computes a raw
    //     int8 add with no offset/rescale/requantize step at all (see the
    //     struct doc above — AddSubAlignedArgs carries only `length`). It is
    //     bit-exact vs the scalar path below ONLY when every quant-affine
    //     step degenerates to the identity: zero offsets, no left_shift
    //     scaling, and both the per-input and output (multiplier, shift)
    //     pairs at (1<<30, 1) — the same identity pair the scalar loop itself
    //     already special-cases below.
    //
    // (2) T3.2 widened lane model (arbitrary offsets/multipliers): the
    //     16-wide per-lane requantize model (`add_simd_lanes`) reproduces the
    //     exact scalar per-element math register-held, with a scalar tail for
    //     n % 16 — bit-exact for EVERY param combination (host-tested).
    //
    // Gated `not(feature = "qemu")` — the QEMU TIE728 emulation of VADDS
    // crashes, so SIMD dispatch must be impossible under `feature = "qemu"`
    // (scalar-only there).
    #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
    {
        let identity = |m: i32, s: i32| m == 1 << 30 && s == 1;
        if params.input1_offset == 0
            && params.input2_offset == 0
            && params.output_offset == 0
            && params.quantized_activation_min == i8::MIN as i32
            && params.quantized_activation_max == i8::MAX as i32
            && params.left_shift <= 0
            && identity(params.input1_multiplier, params.input1_shift)
            && identity(params.input2_multiplier, params.input2_shift)
            && identity(params.output_multiplier, params.output_shift)
            && n % 16 == 0
        {
            let in1_ptr = input1.as_ptr();
            let in2_ptr = input2.as_ptr();
            let out_ptr = output.as_mut_ptr();
            if (in1_ptr as usize) % 16 == 0
                && (in2_ptr as usize) % 16 == 0
                && (out_ptr as usize) % 16 == 0
            {
                unsafe {
                    add_simd_aligned(out_ptr, in1_ptr, in2_ptr, n as u32);
                }
                let _ = scratch;
                return Ok(());
            }
        }
        // (2) Widened lane-model path — any offsets/multipliers. The gate is
        // the params-derived half (always true: the model is bit-exact for
        // every param combination); n ≥ 16 and 16-aligned pointers are the
        // per-call half. Misaligned buffers fall through to the scalar
        // kernel (the established alignment-gate fallback).
        if simd_eligible_add_sub_widened(params) && n >= 16 {
            let in1_ptr = input1.as_ptr();
            let in2_ptr = input2.as_ptr();
            let out_ptr = output.as_mut_ptr();
            if (in1_ptr as usize) % 16 == 0
                && (in2_ptr as usize) % 16 == 0
                && (out_ptr as usize) % 16 == 0
            {
                add_simd_lanes(input1, input2, params, output)?;
                let _ = scratch;
                return Ok(());
            }
        }
    }

    let left_shift = params.left_shift;
    let shift_factor = if left_shift >= 0 {
        1i32 << left_shift
    } else {
        1i32
    };

    for i in 0..n {
        let mut val1 = i32::from(input1[i]) + params.input1_offset;
        let mut val2 = i32::from(input2[i]) + params.input2_offset;

        // left_shift before per-input rescaling (TFLM AddFunc step)
        val1 *= shift_factor;
        val2 *= shift_factor;

        // Per-input rescaling
        if params.input1_multiplier != 1i32 << 30 || params.input1_shift != 1 {
            val1 = multiply_by_quantized_multiplier(
                val1, params.input1_multiplier, params.input1_shift);
        }
        if params.input2_multiplier != 1i32 << 30 || params.input2_shift != 1 {
            val2 = multiply_by_quantized_multiplier(
                val2, params.input2_multiplier, params.input2_shift);
        }

        let raw_sum = val1 + val2;
        let scaled = multiply_by_quantized_multiplier(
            raw_sum, params.output_multiplier, params.output_shift);
        let with_offset = scaled + params.output_offset;

        let clamped = if with_offset > params.quantized_activation_max {
            params.quantized_activation_max
        } else if with_offset < params.quantized_activation_min {
            params.quantized_activation_min
        } else {
            with_offset
        };
        output[i] = saturating_cast(clamped);
    }

    let _ = scratch;
    Ok(())
}

/// Elementwise MUL — scalar kernel (host-compilable, bit-exact vs per-tensor golden).
///
/// Per-element `(input1 + input1_offset) * (input2 + input2_offset)`,
/// then single output requantize.  No left_shift or per-input rescaling.
///
/// Matches TFLM `reference_integer_ops::MulElementwise`.
///
/// # Errors
///
/// * [`KernelError::ShapeMismatch`] if slice lengths ≠ `params.num_elements`.
pub fn mul(
    input1: &[i8],
    input2: &[i8],
    params: &ElementwiseParams,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    let n = params.num_elements as usize;
    if input1.len() != n || input2.len() != n || output.len() != n {
        return Err(KernelError::ShapeMismatch);
    }

    // ── TIE728 SIMD dispatch (device-only; compiled out entirely on host) ──
    // Two device paths, tried in order:
    //
    // (1) Raw TIE728 `mul_simd_aligned` (identity contract): computes
    //     `round((input1[i] * input2[i]) >> mul_shift)` with no offsets.
    //     With output_multiplier fixed at 1<<30,
    //     `multiply_by_quantized_multiplier(product, 1<<30, output_shift)`
    //     reduces to `round(product >> (1 - output_shift))`, so
    //     `mul_shift = 1 - output_shift` reproduces the scalar path exactly;
    //     `output_shift <= 1` is exactly what keeps that `mul_shift`
    //     non-negative.
    //
    // (2) T3.2 widened lane model (arbitrary offsets/multipliers): the
    //     16-wide per-lane requantize model (`mul_simd_lanes`) reproduces the
    //     exact scalar per-element math register-held, with a scalar tail for
    //     n % 16 — bit-exact for EVERY param combination (host-tested).
    //
    // Gated `not(feature = "qemu")` — the QEMU TIE728 VMULAS emulation hangs.
    #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
    {
        if params.input1_offset == 0
            && params.input2_offset == 0
            && params.output_offset == 0
            && params.quantized_activation_min == i8::MIN as i32
            && params.quantized_activation_max == i8::MAX as i32
            && params.output_multiplier == 1 << 30
            && params.output_shift <= 1
            && n % 16 == 0
        {
            let in1_ptr = input1.as_ptr();
            let in2_ptr = input2.as_ptr();
            let out_ptr = output.as_mut_ptr();
            if (in1_ptr as usize) % 16 == 0
                && (in2_ptr as usize) % 16 == 0
                && (out_ptr as usize) % 16 == 0
            {
                let mul_shift = 1 - params.output_shift;
                unsafe {
                    mul_simd_aligned(out_ptr, in1_ptr, in2_ptr, n as u32, mul_shift);
                }
                let _ = scratch;
                return Ok(());
            }
        }
        // (2) Widened lane-model path — any offsets/multipliers (same gate
        // shape as the add/sub dispatch: params half + per-call n/alignment).
        if simd_eligible_mul_widened(params) && n >= 16 {
            let in1_ptr = input1.as_ptr();
            let in2_ptr = input2.as_ptr();
            let out_ptr = output.as_mut_ptr();
            if (in1_ptr as usize) % 16 == 0
                && (in2_ptr as usize) % 16 == 0
                && (out_ptr as usize) % 16 == 0
            {
                mul_simd_lanes(input1, input2, params, output)?;
                let _ = scratch;
                return Ok(());
            }
        }
    }

    for i in 0..n {
        let val1 = i32::from(input1[i]) + params.input1_offset;
        let val2 = i32::from(input2[i]) + params.input2_offset;
        let product = val1 * val2;
        let scaled = multiply_by_quantized_multiplier(
            product, params.output_multiplier, params.output_shift);
        let with_offset = scaled + params.output_offset;

        let clamped = if with_offset > params.quantized_activation_max {
            params.quantized_activation_max
        } else if with_offset < params.quantized_activation_min {
            params.quantized_activation_min
        } else {
            with_offset
        };
        output[i] = saturating_cast(clamped);
    }

    let _ = scratch;
    Ok(())
}

/// Elementwise SUB — scalar kernel (host-compilable, bit-exact vs per-tensor golden).
///
/// Same chain as [`add`] but subtracts `scaled_input2` from `scaled_input1`.
///
/// Matches TFLM `reference_integer_ops::Sub` (uses same ArithmeticParams as Add).
///
/// # Errors
///
/// * [`KernelError::ShapeMismatch`] if slice lengths ≠ `params.num_elements`.
pub fn sub(
    input1: &[i8],
    input2: &[i8],
    params: &ElementwiseParams,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    let n = params.num_elements as usize;
    if input1.len() != n || input2.len() != n || output.len() != n {
        return Err(KernelError::ShapeMismatch);
    }

    // ── TIE728 SIMD dispatch (device-only; compiled out entirely on host) ──
    // Two device paths, tried in order:
    //
    // (1) Raw TIE728 `sub_simd_aligned` (identity contract): computes a raw
    //     int8 subtract with no offset/rescale/requantize step — bit-exact
    //     vs the scalar path below ONLY under the same identity contract as
    //     `add`'s dispatch (zero offsets, no left_shift scaling, identity
    //     (1<<30, 1) pairs everywhere, full-range clamp, n % 16).
    //
    // (2) T3.2 widened lane model (arbitrary offsets/multipliers): the
    //     16-wide per-lane requantize model (`sub_simd_lanes`) reproduces the
    //     exact scalar per-element math register-held, with a scalar tail for
    //     n % 16 — bit-exact for EVERY param combination (host-tested).
    //
    // Gated `not(feature = "qemu")` — the QEMU TIE728 VSUBS emulation is
    // silently wrong.
    #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
    {
        let identity = |m: i32, s: i32| m == 1 << 30 && s == 1;
        if params.input1_offset == 0
            && params.input2_offset == 0
            && params.output_offset == 0
            && params.quantized_activation_min == i8::MIN as i32
            && params.quantized_activation_max == i8::MAX as i32
            && params.left_shift <= 0
            && identity(params.input1_multiplier, params.input1_shift)
            && identity(params.input2_multiplier, params.input2_shift)
            && identity(params.output_multiplier, params.output_shift)
            && n % 16 == 0
        {
            let in1_ptr = input1.as_ptr();
            let in2_ptr = input2.as_ptr();
            let out_ptr = output.as_mut_ptr();
            if (in1_ptr as usize) % 16 == 0
                && (in2_ptr as usize) % 16 == 0
                && (out_ptr as usize) % 16 == 0
            {
                unsafe {
                    sub_simd_aligned(out_ptr, in1_ptr, in2_ptr, n as u32);
                }
                let _ = scratch;
                return Ok(());
            }
        }
        // (2) Widened lane-model path — any offsets/multipliers (same gate
        // shape as the add dispatch: params half + per-call n/alignment).
        if simd_eligible_add_sub_widened(params) && n >= 16 {
            let in1_ptr = input1.as_ptr();
            let in2_ptr = input2.as_ptr();
            let out_ptr = output.as_mut_ptr();
            if (in1_ptr as usize) % 16 == 0
                && (in2_ptr as usize) % 16 == 0
                && (out_ptr as usize) % 16 == 0
            {
                sub_simd_lanes(input1, input2, params, output)?;
                let _ = scratch;
                return Ok(());
            }
        }
    }

    let left_shift = params.left_shift;
    let shift_factor = if left_shift >= 0 {
        1i32 << left_shift
    } else {
        1i32
    };

    for i in 0..n {
        let mut val1 = i32::from(input1[i]) + params.input1_offset;
        let mut val2 = i32::from(input2[i]) + params.input2_offset;

        val1 *= shift_factor;
        val2 *= shift_factor;

        if params.input1_multiplier != 1i32 << 30 || params.input1_shift != 1 {
            val1 = multiply_by_quantized_multiplier(
                val1, params.input1_multiplier, params.input1_shift);
        }
        if params.input2_multiplier != 1i32 << 30 || params.input2_shift != 1 {
            val2 = multiply_by_quantized_multiplier(
                val2, params.input2_multiplier, params.input2_shift);
        }

        let raw_sub = val1 - val2;
        let scaled = multiply_by_quantized_multiplier(
            raw_sub, params.output_multiplier, params.output_shift);
        let with_offset = scaled + params.output_offset;

        let clamped = if with_offset > params.quantized_activation_max {
            params.quantized_activation_max
        } else if with_offset < params.quantized_activation_min {
            params.quantized_activation_min
        } else {
            with_offset
        };
        output[i] = saturating_cast(clamped);
    }

    let _ = scratch;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// TIE728 SIMD backend — device-only (NEVER compiled on host)
// ─────────────────────────────────────────────────────────────────────────────

/// TIE728 SIMD backend for elementwise ops.
///
/// This module is **entirely cfg-gated** behind `#[cfg(target_arch = "xtensa")]`
/// (the dispatch into it is additionally gated `not(feature = "qemu")`, so the
/// broken QEMU TIE728 emulation is never reached) and is NEVER compiled on the
/// host (stable-aarch64-apple-darwin). It exists in the tree for structural
/// review and Phase 5 device verification (T5.3).
///
/// ## Architecture
///
/// The SIMD path calls the vendored `dl_tie728_s8_add_w1_16_w2_16` /
/// `dl_tie728_s8_mul_w1_16_w2_16` / `dl_tie728_s8_sub_w1_16_w2_16` entry
/// points from vendored .S files in `hematite-s3/src/asm/` via `global_asm!`.
///
/// Register convention (Xtensa XCC):
/// * a2 = output pointer (i8*)
/// * a3 = input1 pointer (i8*)
/// * a4 = input2 pointer (i8*)
/// * a5 = args pointer (packed struct)
///
/// ## Vendored .S files
///
/// Cell `hematite-s3/src/asm/` contains:
/// * `dl_tie728_s8.S` — shared macros (pre-existing)
/// * `dl_tie728_s8_add.S` — 6 add entry points (aligned/unaligned, broadcast)
/// * `dl_tie728_s8_mul.S` — 6 mul entry points (aligned/unaligned, broadcast)
/// * `dl_tie728_s8_sub.S` — 6 sub entry points (aligned/unaligned, broadcast)
///
/// All vendored from esp-dl @ 12c0616de145b704e1149c474b9a1e852e631d67 (MIT).
///
/// ## Args struct layouts (derived from vendored .S l32i offsets)
///
/// ### Add/Sub (aligned w1_16_w2_16)
/// * +44: length (u32) — total element count (not div-16)
///
/// ### Add/Sub (unaligned)
/// * +64: c_div_x_1 (i32) — number of full 16-byte chunks minus 1
/// * +76: c_remainder (u32) — remainder elements (0–15)
///
/// ### Mul (aligned w1_16_w2_16)
/// * +64: c_div_x_1 (i32) — number of full 16-byte chunks minus 1
/// * +80: mul_shift (i32) — right-shift for round+requantize
///
/// ### Mul (unaligned)
/// * +64: c_div_x_1 (i32)
/// * +76: c_remainder (u32)
/// * +80: mul_shift (i32)
///
/// ## A4 contract notes
///
/// * Leg (a): SIMD output must match a per-tensor TFLM golden (Phase 5 fixture
///   with per-tensor OUTPUT_MULTIPLIER/SHIFT).
/// * Leg (c): SIMD vs scalar ref cross-check tolerance ≤1 LSB on requantize.
///
/// These SIMD kernels do NOT handle quantization offsets (input1_offset,
/// input2_offset, output_offset) or per-input rescaling. They compute raw
/// int8 add/mul/sub. The calling layer is responsible for quant-affine
/// preprocessing and postprocessing.
#[cfg(target_arch = "xtensa")]
mod elementwise_simd {
    // Include the vendored TIE728 shared macros and elementwise entry points.
    //
    // The shared `dl_tie728_s8.S` provides macros used by all three
    // elementwise files (`dl_tie728_s8_unaligned_store0`,
    // `tie728_s8_vector_round_result`, etc.).
    core::arch::global_asm!(
        include_str!("../src/asm/dl_tie728_s8.S"),
        include_str!("../src/asm/dl_tie728_s8_add.S"),
        include_str!("../src/asm/dl_tie728_s8_mul.S"),
        include_str!("../src/asm/dl_tie728_s8_sub.S"),
    );

    // ── Args structs — derived from vendored .S l32i offsets ──────────────

    /// Args for aligned add/sub — matches `dl_tie728_s8_add_w1_16_w2_16`
    /// and `dl_tie728_s8_sub_w1_16_w2_16`.
    ///
    /// ABI verified against vendored .S at +44:
    /// `l32i a6, a5, 44` → length, then `srai a5, a6, 4` → loop count.
    ///
    /// ABI unverified on device — validate at T5.3.
    #[repr(C)]
    #[allow(dead_code)]
    struct AddSubAlignedArgs {
        _pad0: [u8; 44],       // offset 0-43: unused by these entry points
        length: u32,           // offset 44: total element count
    }

    impl Default for AddSubAlignedArgs {
        fn default() -> Self {
            Self {
                _pad0: [0u8; 44],
                length: 0,
            }
        }
    }

    /// Args for aligned mul — matches `dl_tie728_s8_mul_w1_16_w2_16`.
    ///
    /// ABI verified against vendored .S:
    /// * `l32i a6, a5, 64` → c_div_x_1
    /// * `l32i a7, a5, 80` → mul_shift
    ///
    /// ABI unverified on device — validate at T5.3.
    #[repr(C)]
    #[allow(dead_code)]
    struct MulAlignedArgs {
        _pad0: [u8; 64],       // offset 0-63: unused
        c_div_x_1: i32,        // offset 64: (num_elements / 16) - 1
        _pad1: [u8; 12],       // offset 68-79
        mul_shift: i32,        // offset 80: requantize right-shift
    }

    impl Default for MulAlignedArgs {
        fn default() -> Self {
            Self {
                _pad0: [0u8; 64],
                c_div_x_1: 0,
                _pad1: [0u8; 12],
                mul_shift: 0,
            }
        }
    }

    // ── SIMD kernel glue ──────────────────────────────────────────────────

    /// SIMD elementwise add (aligned) — calls the vendored TIE728 entry point.
    ///
    /// Calls `dl_tie728_s8_add_w1_16_w2_16`:
    /// * a2 = output (i8*)
    /// * a3 = input1 (i8*)
    /// * a4 = input2 (i8*)
    /// * a5 = &AddSubAlignedArgs { length: num_elements }
    ///
    /// # Safety
    ///
    /// This function is inherently unsafe: it calls into foreign assembly
    /// via the C ABI. ABI unverified — validate at T5.3 on device.
    ///
    /// # Preconditions (caller MUST guarantee)
    ///
    /// * `num_elements` must be a multiple of 16 (16-wide SIMD lanes).
    /// * All pointers must be 16-byte aligned for EE.VLD.128.IP / EE.VST.128.IP.
    ///
    /// # Register-hazard note (device finding, task 8)
    ///
    /// The previous `mov a10,{output}` template style is unsafe: LLVM may
    /// allocate an `in(reg)` operand to a register the template itself
    /// overwrites first (observed: `input2`→a10 and `args`→a11, so the
    /// template's own `mov a10`/`mov a11` clobbered them and the kernel
    /// received `a12=output` / `a13=input1` — it then read the "length" from
    /// input1+44 (garbage) and wrote 16-byte chunks across DRAM until it hit
    /// unmapped memory, corrupting the defmt `RTT_ENCODER.taken` flag on the
    /// way — the task-5 "defmt logger taken reentrantly" panic is a symptom of
    /// that walk). Same fix as `avg_pool_2d_simd_ctx`: pinned-register
    /// operands, no `mov` template, plain struct literal, `#[inline(never)]`.
    #[allow(dead_code)]
    #[inline(never)]
    pub unsafe fn add_simd_aligned(
        output: *mut i8,
        input1: *const i8,
        input2: *const i8,
        num_elements: u32,
    ) {
        // Plain struct literal — the MaybeUninit pointer-cast build is
        // miscompiled by the Xtensa LLVM backend (pool.rs precedent).
        let args = AddSubAlignedArgs {
            length: num_elements,
            ..Default::default()
        };
        core::arch::asm!(
            "call8 dl_tie728_s8_add_w1_16_w2_16",
            in("a10") output,
            in("a11") input1,
            in("a12") input2,
            in("a13") &args,
            clobber_abi("C"),
        );
    }

    /// SIMD elementwise mul (aligned) — calls the vendored TIE728 entry point.
    ///
    /// Calls `dl_tie728_s8_mul_w1_16_w2_16`:
    /// * a2 = output (i8*)
    /// * a3 = input1 (i8*)
    /// * a4 = input2 (i8*)
    /// * a5 = &MulAlignedArgs { c_div_x_1, mul_shift }
    ///
    /// # Safety
    ///
    /// Same safety contract as `add_simd_aligned`. ABI unverified.
    ///
    /// # Preconditions
    ///
    /// * `num_elements` must be a multiple of 16 and ≥ 16.
    /// * All pointers 16-byte aligned.
    /// * `mul_shift`: right-shift for requantize rounding
    ///   (`tie728_s8_vector_round_result` macro). Set to 0 for no shift.
    ///
    /// Pinned-register operands (no `mov` template) + plain struct literal —
    /// same register-hazard fix as `add_simd_aligned` (task-8 device finding).
    #[allow(dead_code)]
    #[inline(never)]
    pub unsafe fn mul_simd_aligned(
        output: *mut i8,
        input1: *const i8,
        input2: *const i8,
        num_elements: u32,
        mul_shift: i32,
    ) {
        let args = MulAlignedArgs {
            c_div_x_1: (num_elements / 16) as i32 - 1,
            mul_shift,
            ..Default::default()
        };
        core::arch::asm!(
            "call8 dl_tie728_s8_mul_w1_16_w2_16",
            in("a10") output,
            in("a11") input1,
            in("a12") input2,
            in("a13") &args,
            clobber_abi("C"),
        );
    }

    /// SIMD elementwise sub (aligned) — calls the vendored TIE728 entry point.
    ///
    /// Calls `dl_tie728_s8_sub_w1_16_w2_16`:
    /// * a2 = output (i8*)
    /// * a3 = input1 (i8*)
    /// * a4 = input2 (i8*)
    /// * a5 = &AddSubAlignedArgs { length: num_elements }
    ///
    /// # Safety
    ///
    /// Same safety contract as `add_simd_aligned`. ABI unverified.
    ///
    /// # Preconditions
    ///
    /// * `num_elements` must be a multiple of 16.
    /// * All pointers 16-byte aligned.
    ///
    /// Pinned-register operands (no `mov` template) + plain struct literal —
    /// same register-hazard fix as `add_simd_aligned` (task-8 device finding).
    #[allow(dead_code)]
    #[inline(never)]
    pub unsafe fn sub_simd_aligned(
        output: *mut i8,
        input1: *const i8,
        input2: *const i8,
        num_elements: u32,
    ) {
        let args = AddSubAlignedArgs {
            length: num_elements,
            ..Default::default()
        };
        core::arch::asm!(
            "call8 dl_tie728_s8_sub_w1_16_w2_16",
            in("a10") output,
            in("a11") input1,
            in("a12") input2,
            in("a13") &args,
            clobber_abi("C"),
        );
    }
}

// Re-export the SIMD entry points at the crate level.
#[cfg(target_arch = "xtensa")]
pub use elementwise_simd::add_simd_aligned;
#[cfg(target_arch = "xtensa")]
pub use elementwise_simd::mul_simd_aligned;
#[cfg(target_arch = "xtensa")]
pub use elementwise_simd::sub_simd_aligned;

// ── Prepared-elementwise fast path ───────────────────────────────────────

/// Shared SIMD-eligibility gate for add/sub (identity quant-affine chain).
///
/// Host-compilable: returns `true` when `dl_tie728_s8_{add,sub}_w1_16_w2_16`
/// produces output bit-exact vs the scalar kernel (raw int8 add/sub with no
/// offset/rescale/requantize, which only matches the scalar when every
/// quant-affine step degenerates to the identity pair `(1<<30, 1)`).
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

/// Shared SIMD-eligibility gate for mul (raw int8 product + fixed requantize).
///
/// Returns `Some(mul_shift)` when `dl_tie728_s8_mul_w1_16_w2_16` is bit-exact
/// vs the scalar kernel. `mul_shift = 1 - output_shift`.
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

// ─────────────────────────────────────────────────────────────────────────────
// T3.2 — widened per-lane SIMD models (host-compilable lane math)
// ─────────────────────────────────────────────────────────────────────────────
//
// The raw TIE728 dispatch above is bit-exact only for the identity quant-affine
// contracts. T3.2 widens elementwise SIMD to ARBITRARY input/output offsets and
// multipliers via a 16-wide per-lane requantize model: each lane runs the exact
// scalar per-element math (input offsets → left_shift scaling → conditional
// per-input `multiply_by_quantized_multiplier` (skipped iff (1<<30, 1)) → i32
// sum/product → output requantize → output_offset → clamp → saturating_cast)
// register-held — the same lane sequence `fused::chain_step_apply` applies to
// chain steps — with a scalar tail for n % 16.
//
// The models are host-compilable (`#[cfg(any(all(target_arch = "xtensa",
// not(feature = "qemu")), test))]` — the fused.rs pattern) so host tests prove
// them bit-exact vs the scalar kernels; on device the `add`/`mul`/`sub`
// dispatches above run them under the same QEMU gate. No TIE728 asm is
// involved — the engagement is the register-held 16-wide lane loop.

/// Widened add/sub SIMD-eligibility gate — host-compilable.
///
/// Returns `true` for EVERY param combination: the per-lane model applies the
/// exact scalar arithmetic, so it is bit-exact by construction regardless of
/// offsets, multipliers, shifts, left_shift, or activation range. The gate
/// exists so the `Prepared*` handles can evaluate the params-derived half once
/// at construction; the per-call half (n ≥ 16, 16-aligned pointers) is
/// re-checked in `run`.
pub(crate) fn simd_eligible_add_sub_widened(_params: &ElementwiseParams) -> bool {
    true
}

/// Widened mul SIMD-eligibility gate — host-compilable. Same contract as
/// [`simd_eligible_add_sub_widened`]: every param combination is bit-exact
/// under the per-lane model.
pub(crate) fn simd_eligible_mul_widened(_params: &ElementwiseParams) -> bool {
    true
}

/// One add/sub lane — the exact scalar per-element math (identical to the
/// `add`/`sub` scalar loops below, factored per-lane for the 16-wide model).
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
#[inline]
fn add_sub_lane(input1: i8, input2: i8, params: &ElementwiseParams, is_sub: bool) -> i8 {
    let mut val1 = i32::from(input1) + params.input1_offset;
    let mut val2 = i32::from(input2) + params.input2_offset;

    let shift_factor = if params.left_shift >= 0 {
        1i32 << params.left_shift
    } else {
        1i32
    };
    val1 *= shift_factor;
    val2 *= shift_factor;

    if params.input1_multiplier != 1i32 << 30 || params.input1_shift != 1 {
        val1 = multiply_by_quantized_multiplier(
            val1, params.input1_multiplier, params.input1_shift);
    }
    if params.input2_multiplier != 1i32 << 30 || params.input2_shift != 1 {
        val2 = multiply_by_quantized_multiplier(
            val2, params.input2_multiplier, params.input2_shift);
    }

    let raw = if is_sub { val1 - val2 } else { val1 + val2 };
    let scaled = multiply_by_quantized_multiplier(
        raw, params.output_multiplier, params.output_shift);
    let with_offset = scaled + params.output_offset;

    let clamped = if with_offset > params.quantized_activation_max {
        params.quantized_activation_max
    } else if with_offset < params.quantized_activation_min {
        params.quantized_activation_min
    } else {
        with_offset
    };
    saturating_cast(clamped)
}

/// One mul lane — the exact scalar per-element math (identical to the `mul`
/// scalar loop, factored per-lane for the 16-wide model).
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
#[inline]
fn mul_lane(input1: i8, input2: i8, params: &ElementwiseParams) -> i8 {
    let val1 = i32::from(input1) + params.input1_offset;
    let val2 = i32::from(input2) + params.input2_offset;
    let product = val1 * val2;
    let scaled = multiply_by_quantized_multiplier(
        product, params.output_multiplier, params.output_shift);
    let with_offset = scaled + params.output_offset;

    let clamped = if with_offset > params.quantized_activation_max {
        params.quantized_activation_max
    } else if with_offset < params.quantized_activation_min {
        params.quantized_activation_min
    } else {
        with_offset
    };
    saturating_cast(clamped)
}

/// 16-wide vector main loop + scalar tail — the shared widened-SIMD shape for
/// all three elementwise models. The vector main loop processes full 16-lane
/// chunks (the TIE728 lane width) with per-lane register math; the tail covers
/// `n % 16` elements scalarly.
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
fn lane_loop(
    input1: &[i8],
    input2: &[i8],
    params: &ElementwiseParams,
    output: &mut [i8],
    lane: fn(i8, i8, &ElementwiseParams) -> i8,
) {
    let n = params.num_elements as usize;
    let mut i = 0;
    while i + 16 <= n {
        for l in 0..16 {
            output[i + l] = lane(input1[i + l], input2[i + l], params);
        }
        i += 16;
    }
    for j in i..n {
        output[j] = lane(input1[j], input2[j], params);
    }
}

/// Widened ADD lane model — 16-wide per-lane requantize + scalar tail.
/// Bit-exact vs the scalar [`add`] kernel for every param combination.
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
pub(crate) fn add_simd_lanes(
    input1: &[i8],
    input2: &[i8],
    params: &ElementwiseParams,
    output: &mut [i8],
) -> Result<(), KernelError> {
    let n = params.num_elements as usize;
    if input1.len() != n || input2.len() != n || output.len() != n {
        return Err(KernelError::ShapeMismatch);
    }
    lane_loop(input1, input2, params, output, |a, b, p| {
        add_sub_lane(a, b, p, false)
    });
    Ok(())
}

/// Widened SUB lane model — same contract as [`add_simd_lanes`] with
/// `val1 - val2`.
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
pub(crate) fn sub_simd_lanes(
    input1: &[i8],
    input2: &[i8],
    params: &ElementwiseParams,
    output: &mut [i8],
) -> Result<(), KernelError> {
    let n = params.num_elements as usize;
    if input1.len() != n || input2.len() != n || output.len() != n {
        return Err(KernelError::ShapeMismatch);
    }
    lane_loop(input1, input2, params, output, |a, b, p| {
        add_sub_lane(a, b, p, true)
    });
    Ok(())
}

/// Widened MUL lane model — same contract as [`add_simd_lanes`].
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
pub(crate) fn mul_simd_lanes(
    input1: &[i8],
    input2: &[i8],
    params: &ElementwiseParams,
    output: &mut [i8],
) -> Result<(), KernelError> {
    let n = params.num_elements as usize;
    if input1.len() != n || input2.len() != n || output.len() != n {
        return Err(KernelError::ShapeMismatch);
    }
    lane_loop(input1, input2, params, output, |a, b, p| mul_lane(a, b, p));
    Ok(())
}

/// Prepared elementwise add — runs the SIMD gate once at construction.
pub struct PreparedAdd {
    simd: bool,
    widened: bool,
    params: &'static ElementwiseParams,
}

impl PreparedAdd {
    /// Run the SIMD gate once; subsequent `run` calls skip it.
    pub fn new(params: &'static ElementwiseParams) -> Result<Self, KernelError> {
        let device = cfg!(all(target_arch = "xtensa", not(feature = "qemu")));
        let simd = simd_eligible_add_sub(params) && device;
        let widened = simd_eligible_add_sub_widened(params) && device;
        Ok(Self { simd, widened, params })
    }

    /// Whether the TIE728 SIMD path is active for these params.
    pub fn is_simd(&self) -> bool {
        self.simd || self.widened
    }

    /// Run elementwise add on `input1` + `input2` → `output`.
    pub fn run(
        &self,
        input1: &[i8],
        input2: &[i8],
        output: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
        {
            let n = self.params.num_elements as usize;
            if self.simd && n % 16 == 0 {
                let in1_ptr = input1.as_ptr();
                let in2_ptr = input2.as_ptr();
                let out_ptr = output.as_mut_ptr();
                if (in1_ptr as usize) % 16 == 0
                    && (in2_ptr as usize) % 16 == 0
                    && (out_ptr as usize) % 16 == 0
                {
                    unsafe {
                        add_simd_aligned(out_ptr, in1_ptr, in2_ptr, n as u32);
                    }
                    let _ = scratch;
                    return Ok(());
                }
            }
            if self.widened && n >= 16 {
                let in1_ptr = input1.as_ptr();
                let in2_ptr = input2.as_ptr();
                let out_ptr = output.as_mut_ptr();
                if (in1_ptr as usize) % 16 == 0
                    && (in2_ptr as usize) % 16 == 0
                    && (out_ptr as usize) % 16 == 0
                {
                    add_simd_lanes(input1, input2, self.params, output)?;
                    let _ = scratch;
                    return Ok(());
                }
            }
        }
        add(input1, input2, self.params, output, scratch)
    }
}

/// Prepared elementwise mul — runs the SIMD gate once at construction.
pub struct PreparedMul {
    mul_shift: Option<i32>,
    widened: bool,
    params: &'static ElementwiseParams,
}

impl PreparedMul {
    /// Run the SIMD gate once; subsequent `run` calls skip it.
    pub fn new(params: &'static ElementwiseParams) -> Result<Self, KernelError> {
        let device = cfg!(all(target_arch = "xtensa", not(feature = "qemu")));
        let mul_shift = simd_eligible_mul(params).filter(|_| device);
        let widened = simd_eligible_mul_widened(params) && device;
        Ok(Self { mul_shift, widened, params })
    }

    /// Whether the TIE728 SIMD path is active for these params.
    pub fn is_simd(&self) -> bool {
        self.mul_shift.is_some() || self.widened
    }

    /// Run elementwise mul on `input1` * `input2` → `output`.
    pub fn run(
        &self,
        input1: &[i8],
        input2: &[i8],
        output: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
        {
            let n = self.params.num_elements as usize;
            if let Some(mul_shift) = self.mul_shift {
                if n % 16 == 0 {
                    let in1_ptr = input1.as_ptr();
                    let in2_ptr = input2.as_ptr();
                    let out_ptr = output.as_mut_ptr();
                    if (in1_ptr as usize) % 16 == 0
                        && (in2_ptr as usize) % 16 == 0
                        && (out_ptr as usize) % 16 == 0
                    {
                        unsafe {
                            mul_simd_aligned(out_ptr, in1_ptr, in2_ptr, n as u32, mul_shift);
                        }
                        let _ = scratch;
                        return Ok(());
                    }
                }
            }
            if self.widened && n >= 16 {
                let in1_ptr = input1.as_ptr();
                let in2_ptr = input2.as_ptr();
                let out_ptr = output.as_mut_ptr();
                if (in1_ptr as usize) % 16 == 0
                    && (in2_ptr as usize) % 16 == 0
                    && (out_ptr as usize) % 16 == 0
                {
                    mul_simd_lanes(input1, input2, self.params, output)?;
                    let _ = scratch;
                    return Ok(());
                }
            }
        }
        mul(input1, input2, self.params, output, scratch)
    }
}

/// Prepared elementwise sub — runs the SIMD gate once at construction.
pub struct PreparedSub {
    simd: bool,
    widened: bool,
    params: &'static ElementwiseParams,
}

impl PreparedSub {
    /// Run the SIMD gate once; subsequent `run` calls skip it.
    pub fn new(params: &'static ElementwiseParams) -> Result<Self, KernelError> {
        let device = cfg!(all(target_arch = "xtensa", not(feature = "qemu")));
        let simd = simd_eligible_add_sub(params) && device;
        let widened = simd_eligible_add_sub_widened(params) && device;
        Ok(Self { simd, widened, params })
    }

    /// Whether the TIE728 SIMD path is active for these params.
    pub fn is_simd(&self) -> bool {
        self.simd || self.widened
    }

    /// Run elementwise sub on `input1` − `input2` → `output`.
    pub fn run(
        &self,
        input1: &[i8],
        input2: &[i8],
        output: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
        {
            let n = self.params.num_elements as usize;
            if self.simd && n % 16 == 0 {
                let in1_ptr = input1.as_ptr();
                let in2_ptr = input2.as_ptr();
                let out_ptr = output.as_mut_ptr();
                if (in1_ptr as usize) % 16 == 0
                    && (in2_ptr as usize) % 16 == 0
                    && (out_ptr as usize) % 16 == 0
                {
                    unsafe {
                        sub_simd_aligned(out_ptr, in1_ptr, in2_ptr, n as u32);
                    }
                    let _ = scratch;
                    return Ok(());
                }
            }
            if self.widened && n >= 16 {
                let in1_ptr = input1.as_ptr();
                let in2_ptr = input2.as_ptr();
                let out_ptr = output.as_mut_ptr();
                if (in1_ptr as usize) % 16 == 0
                    && (in2_ptr as usize) % 16 == 0
                    && (out_ptr as usize) % 16 == 0
                {
                    sub_simd_lanes(input1, input2, self.params, output)?;
                    let _ = scratch;
                    return Ok(());
                }
            }
        }
        sub(input1, input2, self.params, output, scratch)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host-compilable widened-SIMD model tests (T3.2)
// ─────────────────────────────────────────────────────────────────────────────
//
// The lane models are the exact scalar per-element math run in 16-wide chunks
// (device dispatch + host tests). These tests prove the models bit-exact vs
// the scalar kernels across a non-identity offset/multiplier sweep — the
// same-backend oracle the correctness contract demands (NOT hematite-ref for
// hard_swish; here the scalar kernels ARE the oracle).

#[cfg(test)]
mod widened_simd_model_tests {
    extern crate alloc;
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Non-identity elementwise params — the two-stage TFLM Add rounding
    /// shape (per-input roundings, left_shift 20, output requantize) plus a
    /// clamped activation range and non-zero zero points.
    fn non_identity_add_params(n: i32) -> ElementwiseParams {
        ElementwiseParams {
            num_elements: n,
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
        }
    }

    /// Non-identity mul params — a real scale change (product requantize).
    fn non_identity_mul_params(n: i32) -> ElementwiseParams {
        ElementwiseParams {
            num_elements: n,
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
        }
    }

    /// Deterministic LCG `i8` pattern (full int8 range).
    fn pattern(seed: u32, n: usize) -> Vec<i8> {
        let mut out = vec![0i8; n];
        let mut x = seed;
        for v in out.iter_mut() {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *v = (x >> 16) as i8;
        }
        out
    }

    /// The non-identity sweep: every model must be bit-exact vs the scalar
    /// kernel for lengths {16, 24, 32, 48} (24 exercises the n % 16 scalar
    /// tail) across the param family {identity, non-identity add/sub, mul}.
    #[test]
    fn widened_models_match_scalar_kernels_bit_exact() {
        let mut checked = 0;
        for &n in &[16usize, 24, 32, 48] {
            let input1 = pattern(0xAAAA_0001 + n as u32, n);
            let input2 = pattern(0x5555_0002 + n as u32, n);

            // Identity params — the raw-TIE728 contract (model must match
            // too, since identity params also pass the widened gate).
            let id_add = ElementwiseParams {
                num_elements: n as i32,
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
            };
            let mut got = vec![0i8; n];
            let mut want = vec![0i8; n];
            add_simd_lanes(&input1, &input2, &id_add, &mut got).expect("model add runs");
            add(&input1, &input2, &id_add, &mut want, &mut []).expect("scalar add runs");
            assert_eq!(got, want, "identity add n={n}");

            // Non-identity add/sub (two-stage rounding, clamped range).
            let p = non_identity_add_params(n as i32);
            let mut got = vec![0i8; n];
            let mut want = vec![0i8; n];
            add_simd_lanes(&input1, &input2, &p, &mut got).expect("model add runs");
            add(&input1, &input2, &p, &mut want, &mut []).expect("scalar add runs");
            assert_eq!(got, want, "non-identity add n={n}");
            let mut got = vec![0i8; n];
            let mut want = vec![0i8; n];
            sub_simd_lanes(&input1, &input2, &p, &mut got).expect("model sub runs");
            sub(&input1, &input2, &p, &mut want, &mut []).expect("scalar sub runs");
            assert_eq!(got, want, "non-identity sub n={n}");

            // Non-identity mul (offsets + product requantize + clamped range).
            let pm = non_identity_mul_params(n as i32);
            let mut got = vec![0i8; n];
            let mut want = vec![0i8; n];
            mul_simd_lanes(&input1, &input2, &pm, &mut got).expect("model mul runs");
            mul(&input1, &input2, &pm, &mut want, &mut []).expect("scalar mul runs");
            assert_eq!(got, want, "non-identity mul n={n}");

            checked += 4;
        }
        assert!(checked >= 16, "sweep must cover all n × param families");
    }

    /// The widened gates accept EVERY param combination (identity and
    /// non-identity alike) — the model is bit-exact by construction — while
    /// the identity gates keep their strict contracts (fused.rs depends on
    /// them).
    #[test]
    fn widened_gates_accept_all_params() {
        let p = non_identity_add_params(256);
        assert!(simd_eligible_add_sub_widened(&p), "non-identity add must engage");
        assert!(!simd_eligible_add_sub(&p), "identity gate must still refuse");
        let pm = non_identity_mul_params(256);
        assert!(simd_eligible_mul_widened(&pm), "non-identity mul must engage");
        assert!(simd_eligible_mul(&pm).is_none(), "identity mul gate must still refuse");
        let id = ElementwiseParams {
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
            quantized_activation_min: i8::MIN as i32,
            quantized_activation_max: i8::MAX as i32,
        };
        assert!(simd_eligible_add_sub_widened(&id) && simd_eligible_mul_widened(&id));
    }

    /// Saturation path: extreme offsets push `x + offset` beyond i8 range —
    /// the i32 lane math must match the scalar's i32 arithmetic (no int8
    /// saturation anywhere in the offset/scale chain).
    #[test]
    fn widened_models_match_scalar_with_extreme_offsets() {
        let n = 48usize;
        let input1 = pattern(0xDEAD_0011, n);
        let input2 = pattern(0xBEEF_0022, n);
        let p = ElementwiseParams {
            num_elements: n as i32,
            input1_offset: 127,
            input2_offset: -127,
            output_offset: 100,
            output_multiplier: 1_342_177_280,
            output_shift: -18,
            left_shift: 20,
            input1_multiplier: 1 << 30,
            input1_shift: 0,
            input2_multiplier: 1_288_490_189,
            input2_shift: -1,
            quantized_activation_min: i8::MIN as i32,
            quantized_activation_max: i8::MAX as i32,
        };
        let mut got = vec![0i8; n];
        let mut want = vec![0i8; n];
        add_simd_lanes(&input1, &input2, &p, &mut got).expect("model add runs");
        add(&input1, &input2, &p, &mut want, &mut []).expect("scalar add runs");
        assert_eq!(got, want, "extreme-offset add must stay i32-exact");
        let mut got = vec![0i8; n];
        let mut want = vec![0i8; n];
        mul_simd_lanes(&input1, &input2, &p, &mut got).expect("model mul runs");
        mul(&input1, &input2, &p, &mut want, &mut []).expect("scalar mul runs");
        assert_eq!(got, want, "extreme-offset mul must stay i32-exact");
    }
}
