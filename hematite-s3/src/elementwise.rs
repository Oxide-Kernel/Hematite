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
    // `add_simd_aligned` computes a raw int8 add with no offset/rescale/
    // requantize step at all (see the struct doc above — AddSubAlignedArgs
    // carries only `length`). It is bit-exact vs the scalar path below ONLY
    // when every quant-affine step degenerates to the identity: zero
    // offsets, no left_shift scaling, and both the per-input and output
    // (multiplier, shift) pairs at (1<<30, 1) — the same identity pair the
    // scalar loop itself already special-cases below. Gated `not(feature =
    // "qemu")` — the QEMU TIE728 emulation of VADDS crashes, so SIMD dispatch
    // must be impossible under `feature = "qemu"` (scalar-only there).
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
    // `mul_simd_aligned` computes `round((input1[i] * input2[i]) >> mul_shift)`
    // with no offsets. With output_multiplier fixed at 1<<30,
    // `multiply_by_quantized_multiplier(product, 1<<30, output_shift)` reduces
    // to `round(product >> (1 - output_shift))`, so `mul_shift = 1 -
    // output_shift` reproduces the scalar path exactly; `output_shift <= 1`
    // is exactly what keeps that `mul_shift` non-negative. Gated
    // `not(feature = "qemu")` — the QEMU TIE728 VMULAS emulation hangs.
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
    // Same identity contract as `add`'s dispatch above — `sub_simd_aligned`
    // computes a raw int8 subtract with no offset/rescale/requantize step.
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
    /// Include the vendored TIE728 shared macros and elementwise entry points.
    ///
    /// The shared `dl_tie728_s8.S` provides macros used by all three
    /// elementwise files (`dl_tie728_s8_unaligned_store0`,
    /// `tie728_s8_vector_round_result`, etc.).
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
    #[allow(dead_code)]
    pub unsafe fn add_simd_aligned(
        output: *mut i8,
        input1: *const i8,
        input2: *const i8,
        num_elements: u32,
    ) {
        // Only the length field (@44) is read by the asm.
        let mut args = core::mem::MaybeUninit::<AddSubAlignedArgs>::uninit();
        args.as_mut_ptr()
            .cast::<u8>()
            .add(44)
            .cast::<u32>()
            .write(num_elements);
        let args = unsafe { args.assume_init_ref() };
        core::arch::asm!(
            "mov a10, {output}",
            "mov a11, {input1}",
            "mov a12, {input2}",
            "mov a13, {args}",
            "call8 dl_tie728_s8_add_w1_16_w2_16",
            output = in(reg) output,
            input1 = in(reg) input1,
            input2 = in(reg) input2,
            args = in(reg) args,
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
    #[allow(dead_code)]
    pub unsafe fn mul_simd_aligned(
        output: *mut i8,
        input1: *const i8,
        input2: *const i8,
        num_elements: u32,
        mul_shift: i32,
    ) {
        // Only c_div_x_1 (@64) and mul_shift (@80) are read by the asm.
        let mut args = core::mem::MaybeUninit::<MulAlignedArgs>::uninit();
        let p = args.as_mut_ptr();
        p.cast::<u8>().add(64).cast::<i32>().write((num_elements / 16) as i32 - 1);
        p.cast::<u8>().add(80).cast::<i32>().write(mul_shift);
        let args = unsafe { args.assume_init_ref() };
        core::arch::asm!(
            "mov a10, {output}",
            "mov a11, {input1}",
            "mov a12, {input2}",
            "mov a13, {args}",
            "call8 dl_tie728_s8_mul_w1_16_w2_16",
            output = in(reg) output,
            input1 = in(reg) input1,
            input2 = in(reg) input2,
            args = in(reg) args,
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
    #[allow(dead_code)]
    pub unsafe fn sub_simd_aligned(
        output: *mut i8,
        input1: *const i8,
        input2: *const i8,
        num_elements: u32,
    ) {
        // Only the length field (@44) is read by the asm.
        let mut args = core::mem::MaybeUninit::<AddSubAlignedArgs>::uninit();
        args.as_mut_ptr()
            .cast::<u8>()
            .add(44)
            .cast::<u32>()
            .write(num_elements);
        let args = unsafe { args.assume_init_ref() };
        core::arch::asm!(
            "mov a10, {output}",
            "mov a11, {input1}",
            "mov a12, {input2}",
            "mov a13, {args}",
            "call8 dl_tie728_s8_sub_w1_16_w2_16",
            output = in(reg) output,
            input1 = in(reg) input1,
            input2 = in(reg) input2,
            args = in(reg) args,
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

/// Prepared elementwise add — runs the SIMD gate once at construction.
pub struct PreparedAdd {
    simd: bool,
    params: &'static ElementwiseParams,
}

impl PreparedAdd {
    /// Run the SIMD gate once; subsequent `run` calls skip it.
    pub fn new(params: &'static ElementwiseParams) -> Result<Self, KernelError> {
        let simd = simd_eligible_add_sub(params)
            && cfg!(all(target_arch = "xtensa", not(feature = "qemu")));
        Ok(Self { simd, params })
    }

    /// Whether the TIE728 SIMD path is active for these params.
    pub fn is_simd(&self) -> bool {
        self.simd
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
            let n = self.params.num_elements as usize;            if self.simd && n % 16 == 0 {
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
        }
        add(input1, input2, self.params, output, scratch)
    }
}

/// Prepared elementwise mul — runs the SIMD gate once at construction.
pub struct PreparedMul {
    mul_shift: Option<i32>,
    params: &'static ElementwiseParams,
}

impl PreparedMul {
    /// Run the SIMD gate once; subsequent `run` calls skip it.
    pub fn new(params: &'static ElementwiseParams) -> Result<Self, KernelError> {
        let mul_shift =
            simd_eligible_mul(params).filter(|_| cfg!(all(target_arch = "xtensa", not(feature = "qemu"))));
        Ok(Self { mul_shift, params })
    }

    /// Whether the TIE728 SIMD path is active for these params.
    pub fn is_simd(&self) -> bool {
        self.mul_shift.is_some()
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
            let n = self.params.num_elements as usize;            if let Some(mul_shift) = self.mul_shift {
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
        }
        mul(input1, input2, self.params, output, scratch)
    }
}

/// Prepared elementwise sub — runs the SIMD gate once at construction.
pub struct PreparedSub {
    simd: bool,
    params: &'static ElementwiseParams,
}

impl PreparedSub {
    /// Run the SIMD gate once; subsequent `run` calls skip it.
    pub fn new(params: &'static ElementwiseParams) -> Result<Self, KernelError> {
        let simd = simd_eligible_add_sub(params)
            && cfg!(all(target_arch = "xtensa", not(feature = "qemu")));
        Ok(Self { simd, params })
    }

    /// Whether the TIE728 SIMD path is active for these params.
    pub fn is_simd(&self) -> bool {
        self.simd
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
            let n = self.params.num_elements as usize;            if self.simd && n % 16 == 0 {
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
        }
        sub(input1, input2, self.params, output, scratch)
    }
}
