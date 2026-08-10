// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Conv2D 1×1 GEMM kernel — scalar fallback + ACCX SIMD backend.
//!
//! # Bit-exact contract (Plan A4)
//!
//! | Leg | Contract | Runs on |
//! |-----|----------|---------|
//! | (a) | SIMD ≡ per-tensor TFLM golden bit-exact | Device (Phase 5) |
//! | (b) | Scalar ref ≡ per-channel TFLM golden bit-exact | **Host** (this test) |
//! | (c) | SIMD vs ref cross-check ≤1 LSB on requantize | Device (Phase 5) |
//!
//! On host (stable-aarch64-apple-darwin), only leg (b) executes. The SIMD
//! dispatch is `#[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]` —
//! device-only (see [`conv1x1_accx_dispatch`]); the scalar kernel below is the
//! complete bit-exact fallback on every other target.
//!
//! # Layouts
//!
//! * `input` — NHWC `[batch=1, H, W, Cin]`
//! * `weights` — OHWI `[Cout, 1, 1, Cin]`
//! * `bias` — per-output-channel `[Cout]`
//! * `output` — NHWC `[batch=1, OH, OW, Cout]`
//!
//! The scalar kernel is the complete implementation on host/QEMU. On device
//! the ACCX dispatch (`EE.VMULAS.S8.ACCX`, i32 GPR accumulators — the
//! bit-exact TFLite requantize epilogue runs in Rust) handles eligible
//! aligned layers, falling back to this scalar kernel otherwise; output is
//! bit-exact either way.

//! # ACCX SIMD (device only)
//!
//! The bespoke `s8_accx_conv1x1.S` kernel (assembled into the crate by
//! [`crate::accx`] via `global_asm!`) accumulates into i32 GPR accumulators,
//! so the bit-exact requantize epilogue runs in Rust — no fused-asm
//! requantize, no arg-struct ABI. See [`conv1x1_accx_dispatch`] for the
//! eligibility gate (zero offsets, 16-aligned channel counts, runtime 16-byte
//! pointer alignment, sufficient scratch).

use hematite_core::op_params::Conv2DParams;
use hematite_core::KernelError;
use hematite_int8::{multiply_by_quantized_multiplier, saturating_cast};

/// Compute the product of a `[i32; 4]` shape array.
#[inline(always)]
fn shape_product(shape: &[i32; 4]) -> usize {
    shape[0] as usize * shape[1] as usize * shape[2] as usize * shape[3] as usize
}

/// Transform weights from the caller's `[oc][ic]` (OHWI) layout into the
/// TIE728 11cn `[g][ic][lane]` SIMD layout.
///
/// The vendored asm consumes the filter sequentially as, per 16-output-channel
/// group `g`, `(c_div_x_1 + 1)` chunks of 16 bytes — the `k`-th VSMULAS
/// multiplies input byte `k` with filter vector `k` (see the 11c16 macro). So
/// the asm reads `filter[g * (in_c * 16) + ic * 16 + lane]`, which must equal
/// `weights[(g * 16 + lane) * in_c + ic]` for the SIMD output to match the
/// scalar reference bit-exact. This is a pure no_std permutation
/// (host-compilable, unit-tested).
///
/// `src` length must be `in_c * out_channels`; `dst` length must match.
/// Returns [`KernelError::ShapeMismatch`] otherwise.
pub fn transform_weights_11cn(
    input_c: usize,
    out_channels: usize,
    src: &[i8],
    dst: &mut [i8],
) -> Result<(), KernelError> {
    if !input_c.is_multiple_of(16) || !out_channels.is_multiple_of(16) {
        return Err(KernelError::ShapeMismatch);
    }
    if src.len() != input_c * out_channels || dst.len() != src.len() {
        return Err(KernelError::ShapeMismatch);
    }
    let groups = out_channels / 16;
    for g in 0..groups {
        let g_base = g * input_c * 16;
        for ic in 0..input_c {
            let ic_base = g_base + ic * 16;
            for lane in 0..16 {
                let src_oc = g * 16 + lane;
                dst[ic_base + lane] = src[src_oc * input_c + ic];
            }
        }
    }
    Ok(())
}

/// Context for the ACCX 1×1 conv dispatch — bundled into one `&mut` arg so the
/// Xtensa LLVM backend generates a 1-arg call (multi-arg calls are miscompiled
/// on device; see the Xtensa multi-arg call miscompile class).
#[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
pub(crate) struct Conv1x1AccxCtx<'a> {
    pub input: &'a [i8],
    pub weights: &'a [i8],
    pub bias: &'a [i32],
    pub params: &'a Conv2DParams<'a>,
    pub output: &'a mut [i8],
    pub scratch: &'a mut [u8],
}

/// Attempt the bespoke ACCX 1×1 conv SIMD path (bit-exact), returning
/// `Ok(true)` if handled, `Ok(false)` if ineligible / unaligned / insufficient
/// scratch (caller falls through to the scalar kernel).
///
/// Device-only. Uses `EE.VMULAS.S8.ACCX` (32-bit element accumulator, full
/// 16-bit products) so the output is bit-identical to the scalar reference for
/// any per-channel `mult`/`shift`/offset/activation and any `out_c`.
///
/// When `fused` is `Some`, the pixel loop runs the composed conv+residual-ADD+
/// activation epilogue ([`crate::fused::fused_epilogue`]) on the raw i32
/// accumulators instead of the standalone requantize — the T2.2 fused-conv
/// SIMD path. The `input_offset` fold still runs first in both branches.
///
/// Eligibility: `input_offset == 0`, stride/dilation 1, `out_h == in_h`,
/// `out_w == in_w` (pad 0), `in_c % 16 == 0`, `in_c >= 16`, `out_c >= 1`,
/// all pointers 16-byte aligned, `scratch >= out_c * 4`.
///
/// `uniform` is the precomputed uniform-scale hint `(mult, shift)` —
/// `i32::MIN` shift means "per-channel" (the requantize epilogue selects the
/// fast scale inline, no upfront O(n) scan). The prepared handles cache the
/// hint at construction; the public free functions pass `(0, i32::MIN)`. It is
/// unused by the fused branch.
#[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
pub(crate) fn conv1x1_accx_dispatch(
    ctx: &mut Conv1x1AccxCtx<'_>,
    uniform: (i32, i32),
    fused: Option<&crate::fused::FusedConvAccxParams<'_>>,
) -> Result<bool, KernelError> {
    let params = ctx.params;
    let input_c = params.input_shape[3] as usize;
    let out_c = params.output_shape[3] as usize;
    let in_h = params.input_shape[1] as usize;
    let in_w = params.input_shape[2] as usize;
    let out_h = params.output_shape[1] as usize;
    let out_w = params.output_shape[2] as usize;

    if params.stride_height != 1
        || params.stride_width != 1
        || params.dilation_height_factor != 1
        || params.dilation_width_factor != 1
        || in_h != out_h
        || in_w != out_w
        || !crate::accx::accx_eligible_1x1(input_c, out_c)
    {
        return Ok(false);
    }

    let input_offset = params.input_offset;
    let need = out_c * 4 + if input_offset != 0 { out_c * 4 } else { 0 };
    if ctx.scratch.len() < need {
        return Ok(false);
    }

    let in_ptr = ctx.input.as_ptr();
    let w_ptr = ctx.weights.as_ptr();
    let out_ptr = ctx.output.as_mut_ptr();
    let accs = ctx.scratch.as_mut_ptr() as *mut i32;
    if (in_ptr as usize) % 16 != 0
        || (w_ptr as usize) % 16 != 0
        || (out_ptr as usize) % 16 != 0
        || (accs as usize) % 4 != 0
    {
        return Ok(false);
    }
    let wsum = if input_offset != 0 {
        unsafe { accs.add(out_c) }
    } else {
        core::ptr::null_mut()
    };
    if input_offset != 0 {
        let ws = unsafe { core::slice::from_raw_parts_mut(wsum, out_c) };
        let wv = unsafe { core::slice::from_raw_parts(w_ptr, out_c * input_c) };
        crate::accx::weight_sums_conv(ws, wv, 1, input_c);
    }

    let multipliers = params.output_multiplier_per_channel;
    let shifts = params.output_shift_per_channel;
    let act_min = params.quantized_activation_min;
    let act_max = params.quantized_activation_max;
    let out_offset = params.output_offset;
    let (uniform_mult, uniform_shift) = uniform;

    for oh in 0..out_h {
        for ow in 0..out_w {
            let px_in = (oh * in_w + ow) * input_c;
            let px_out = (oh * out_w + ow) * out_c;
            unsafe {
                crate::accx::accx_conv1x1(
                    in_ptr.add(px_in),
                    w_ptr,
                    accs,
                    input_c,
                    out_c,
                );
            }
            if input_offset != 0 {
                for oc in 0..out_c {
                    let v = unsafe { accs.add(oc).read() };
                    let s = unsafe { wsum.add(oc).read() };
                    unsafe { accs.add(oc).write(v.wrapping_add(input_offset.wrapping_mul(s))) };
                }
            }
            let acc_slice = unsafe { core::slice::from_raw_parts_mut(accs, out_c) };
            match fused {
                Some(fp) => {
                    // T2.2 fused path: composed conv+residual-ADD+activation
                    // epilogue on the raw accumulators (conv output register-
                    // held, never materialized).
                    crate::fused::fused_epilogue(fp, ctx.output, acc_slice, px_out);
                }
                None => {
                    crate::accx::requantize_1x1(&mut crate::accx::ReqCtx {
                        accs: acc_slice,
                        bias: ctx.bias,
                        multipliers,
                        shifts,
                        output_offset: out_offset,
                        act_min,
                        act_max,
                        out_base: px_out,
                        output: ctx.output,
                        uniform_mult,
                        uniform_shift,
                    });
                }
            }
        }
    }
    Ok(true)
}

/// Prepared 1×1 conv handle — runs the SIMD eligibility gate ONCE at
/// construction, then `run` only re-checks pointer alignment and dispatches.
///
/// This closes the wrapper-overhead gap vs C (raw-asm 472 cyc vs Rust public
/// API 2628 cyc): the per-call path is reduced to 4 alignment ANDs + a
/// MaybeUninit args build + the call8, instead of re-validating shapes and
/// re-scanning the per-channel multiplier/shift arrays every call.
///
/// When the gate fails (or on host where the SIMD backend is compiled out),
/// `run` falls through to the scalar kernel — output is bit-exact either way.
pub struct PreparedConv1x1 {
    /// Whether the bespoke ACCX SIMD kernel is eligible on this target.
    accx: bool,
    params: &'static Conv2DParams<'static>,
    /// Cached uniform-scale hint `(mult, shift)`; `i32::MIN` shift = per-channel.
    /// Computed once at construction so `run` never re-scans the per-channel
    /// arrays (the per-call cost is otherwise O(out_c) per kernel invocation).
    /// Read only by the device dispatch (host: SIMD compiled out).
    #[allow(dead_code)]
    uniform: (i32, i32),
}

impl PreparedConv1x1 {
    /// Evaluate the ACCX SIMD gate for the given layer params.
    ///
    /// `input_c` / `out_channels` are taken from the params by the caller; this
    /// keeps the constructor free of shape-product slicing so it stays cheap
    /// (used once per layer at model build time).
    pub fn new(params: &'static Conv2DParams<'static>) -> Result<Self, KernelError> {
        let input_c = params.input_shape[3] as usize;
        let out_channels = params.output_shape[3] as usize;
        let accx = cfg!(all(target_arch = "xtensa", not(feature = "qemu")))
            && crate::accx::accx_eligible_1x1(input_c, out_channels);
        let uniform =
            crate::accx::uniform_scale(params.output_multiplier_per_channel, params.output_shift_per_channel)
                .unwrap_or((0, i32::MIN));
        Ok(Self { accx, params, uniform })
    }

    /// Whether this layer is SIMD-eligible on the current target.
    #[inline]
    pub fn is_simd(&self) -> bool {
        self.accx
    }

    /// Run the 1×1 conv — ACCX SIMD when eligible and aligned, scalar otherwise.
    pub fn run(
        &self,
        input: &[i8],
        weights: &[i8],
        bias: &[i32],
        output: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
        {
            let mut accx_ctx = Conv1x1AccxCtx {
                input,
                weights,
                bias,
                params: self.params,
                output,
                scratch,
            };
            if conv1x1_accx_dispatch(&mut accx_ctx, self.uniform, None)? {
                return Ok(());
            }
        }
        conv2d_1x1(input, weights, bias, self.params, output, scratch)
    }
}

/// Conv2D 1×1 GEMM — scalar kernel (host-compilable, bit-exact vs per-channel golden).
///
/// Mirrors the `hematite-ref/src/conv.rs` arithmetic exactly: bias-init i32
/// accumulator, `(i_val + input_offset) * w_val` MAC, per-channel
/// `multiply_by_quantized_multiplier`, output_offset, clamp, saturating_cast.
///
/// For a 1×1 convolution (FH=1, FW=1), there is no spatial filter loop —
/// each spatial position maps directly to the corresponding input position.
/// Zero-padding via bounds-check is retained for structural compatibility with
/// the general conv path (identical to `hematite-ref/src/conv.rs`).
///
/// Only batch=1 is supported. Batch>1 returns [`KernelError::Unsupported`].
pub fn conv2d_1x1(
    input: &[i8],
    weights: &[i8],
    bias: &[i32],
    params: &Conv2DParams,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    // ── Batch constraint ────────────────────────────────────────────────
    if params.input_shape[0] != 1 {
        return Err(KernelError::Unsupported);
    }

    // ── Slice-length validation ─────────────────────────────────────────
    if input.len() != shape_product(&params.input_shape) {
        return Err(KernelError::ShapeMismatch);
    }
    if weights.len() != shape_product(&params.filter_shape) {
        return Err(KernelError::ShapeMismatch);
    }
    if bias.len() as i32 != params.output_shape[3] {
        return Err(KernelError::ShapeMismatch);
    }
    if output.len() != shape_product(&params.output_shape) {
        return Err(KernelError::ShapeMismatch);
    }
    if params.input_shape[3] != params.filter_shape[3] {
        return Err(KernelError::ShapeMismatch);
    }

    // ── ACCX SIMD dispatch (device-only; compiled out entirely on host) ──
    // Eligibility per Plan A4 T5.3: zero offsets, uniform per-channel
    // multiplier/shift collapsing to the hardware's fixed-multiplier fast
    // path, 16-aligned channel counts, batch=1 (checked above), and runtime
    // 16-byte pointer alignment. Padding is structurally 0 for a 1x1/stride-1
    // conv, so it is not checked.
    //
    // ALSO gated off under the `qemu` feature: QEMU's xtensa/esp32s3 TIE728
    // emulation does not correctly execute `EE.VSMULAS.S8.QACC.LD.INCP` (the
    // fused MAC+load+increment instruction this kernel's MAC loop depends
    // on) — confirmed by direct instruction-level bisection (see
    // local-notes/notepads/hematite-nn/problems.md). This is a QEMU emulation gap,
    // not a code defect, so QEMU builds fall through to the scalar path;
    // real hardware (no `qemu` feature) still gets SIMD once T5.3 validates
    // it there.
    //
    // `uniform_hint` is cached per params identity (todo 16): the O(out_c)
    // uniform_scale scan runs once per unique params, so repeated public-API
    // calls (model layers across predicts, the bench's N>=10 window) skip it.
    #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
    {
        let hint = crate::accx::uniform_hint(
            params as *const _ as usize,
            params.output_multiplier_per_channel,
            params.output_shift_per_channel,
        );
        let mut accx_ctx = Conv1x1AccxCtx {
            input,
            weights,
            bias,
            params,
            output,
            scratch,
        };
        if conv1x1_accx_dispatch(&mut accx_ctx, hint, None)? {
            return Ok(());
        }
    }

    let _ = scratch; // unused by the host path (dispatch is device-only)

    conv2d_1x1_scalar(input, weights, bias, params, output)
}

/// The scalar 1×1 conv kernel, kept as a separate `#[inline(never)]` function
/// so the public [`conv2d_1x1`] dispatch frame stays thin (an inline scalar
/// loop forced the SIMD path to share a huge frame with heavy register
/// spills — the todo-16 public-API gap).
///
/// Assumes the caller already validated the slice lengths (batch, input,
/// weights, bias, output, channel cross-check).
#[inline(never)]
fn conv2d_1x1_scalar(
    input: &[i8],
    weights: &[i8],
    bias: &[i32],
    params: &Conv2DParams,
    output: &mut [i8],
) -> Result<(), KernelError> {
    // ── Extract dimensions ──────────────────────────────────────────────
    let input_h = params.input_shape[1];
    let input_w = params.input_shape[2];
    let input_c = params.input_shape[3];

    let filter_h = params.filter_shape[1];
    let filter_w = params.filter_shape[2];
    let filter_ic = params.filter_shape[3];

    let out_h = params.output_shape[1];
    let out_w = params.output_shape[2];
    let out_channels = params.output_shape[3];

    // ── Derived pad values (same formula as hematite-ref/src/conv.rs) ──
    let dilated_filter_h = (filter_h - 1) * params.dilation_height_factor + 1;
    let dilated_filter_w = (filter_w - 1) * params.dilation_width_factor + 1;
    let pad_h = ((out_h - 1) * params.stride_height + dilated_filter_h - input_h) / 2;
    let pad_w = ((out_w - 1) * params.stride_width + dilated_filter_w - input_w) / 2;

    // ── Per-channel multiplier/shift slices ─────────────────────────────
    let multipliers = params.output_multiplier_per_channel;
    let shifts = params.output_shift_per_channel;

    // ── Stride precomputation ──────────────────────────────────────────
    let input_row_stride = input_w * input_c;
    let filter_oc_stride = filter_h * filter_w * filter_ic;
    let filter_row_stride = filter_w * filter_ic;
    let filter_col_stride = filter_ic;
    let output_row_stride = out_w * out_channels;

    // ── Accumulation loop (identical to hematite-ref conv.rs) ───────────
    for oh in 0..out_h {
        let input_base_h = oh * params.stride_height - pad_h;
        for ow in 0..out_w {
            let input_base_w = ow * params.stride_width - pad_w;
            for oc in 0..out_channels {
                let mut acc: i32 = bias[oc as usize];
                let filter_oc_base = oc * filter_oc_stride;

                for fh in 0..filter_h {
                    let in_h = input_base_h + fh * params.dilation_height_factor;
                    let row_in_bounds = in_h >= 0 && in_h < input_h;

                    for fw in 0..filter_w {
                        let in_w = input_base_w + fw * params.dilation_width_factor;

                        if row_in_bounds && in_w >= 0 && in_w < input_w {
                            let input_base =
                                (in_h * input_row_stride + in_w * input_c) as usize;
                            let filter_base = (filter_oc_base
                                + fh * filter_row_stride
                                + fw * filter_col_stride) as usize;

                            for ic in 0..filter_ic {
                                let i_val = i32::from(input[input_base + ic as usize]);
                                let w_val = i32::from(weights[filter_base + ic as usize]);
                                acc += (i_val + params.input_offset) * w_val;
                            }
                        }
                        // else: zero-padding — contribute 0
                    }
                }

                // Per-channel requantize + output offset + clamp
                let multiplier = multipliers[oc as usize];
                let shift = shifts[oc as usize];
                let scaled = multiply_by_quantized_multiplier(acc, multiplier, shift);
                let with_offset = scaled + params.output_offset;

                let clamped = if with_offset > params.quantized_activation_max {
                    params.quantized_activation_max
                } else if with_offset < params.quantized_activation_min {
                    params.quantized_activation_min
                } else {
                    with_offset
                };

                let out_idx = (oh * output_row_stride + ow * out_channels + oc) as usize;
                output[out_idx] = saturating_cast(clamped);
            }
        }
    }

    Ok(())
}
