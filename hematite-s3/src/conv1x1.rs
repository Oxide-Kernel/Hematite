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
//!
//! # Small / non-16 input channels (T3.3)
//!
//! The strict `accx_eligible_1x1` gate (`input_c >= 16 && input_c % 16 == 0`)
//! is KEPT as the unpadded fast path. In addition, the dispatch now widens to
//! ANY `input_c >= 1` (via `accx_eligible_1x1_padded`): when
//! `input_c % 16 != 0` the dispatch stages a zero-padded input copy (each
//! NHWC pixel padded to the next multiple of 16) AND a zero-padded weight copy
//! (rows padded to the padded channel count) in scratch at 16-byte-aligned
//! offsets, runs the same `s8_accx_conv1x1` kernel on the padded buffers, and
//! folds the non-zero `input_offset` via weight sums over the padded rows (pad
//! lanes are zero). Padded lanes contribute `0 × 0 = 0` — the output is
//! bit-exact vs the scalar reference. The staged carve mirrors the
//! conv3x3/depthwise channel-pad path and T3.6's fc pad (16-byte alignment; an
//! unaligned staged copy would silently fall back to scalar).

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

/// Round a channel count up to the TIE728 SIMD group width (16 lanes).
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
#[inline(always)]
const fn pad16(n: usize) -> usize {
    (n + 15) & !15
}

/// Stage a zero-padded NHWC input copy (each of `pixels` pixels: real
/// `input[p*input_c .. p*input_c+input_c]`, then zeros to `padded_c`) and a
/// zero-padded weight copy (`weights` is `out_c × input_c` raw `[oc][ic]`; the
/// staged copy is `out_c × padded_c` with each row zero-filled past
/// `input_c`).
///
/// The staged buffers are what the device dispatch hands to
/// `s8_accx_conv1x1` when `input_c % 16 != 0`: the kernel VLDs 16-lane
/// vectors and strides weight rows by the padded channel count, so both staged
/// buffers must be padded to a multiple of 16. Padded lanes multiply
/// `0 × 0 = 0`, and the Phase-C `input_offset` fold reads weight sums over the
/// padded rows — pad lanes are zero, so the sums equal the real per-row
/// sums — bit-exact vs the scalar `Σ (in + offset)·w` loop. Host-compilable so
/// the unit tests exercise the real device-pipeline staging.
///
/// # Panics
/// `dst_in` / `dst_w` must be exactly `pixels * padded_c` /
/// `out_c * padded_c` bytes (caller-computed via [`pad16`]); this is asserted.
#[cfg(any(all(target_arch = "xtensa", not(feature = "qemu")), test))]
pub(crate) fn stage_conv1x1_padded(
    dst_in: &mut [u8],
    dst_w: &mut [i8],
    input: &[i8],
    weights: &[i8],
    input_c: usize,
    out_c: usize,
    pixels: usize,
) {
    let padded_c = pad16(input_c);
    assert_eq!(
        dst_in.len(),
        pixels * padded_c,
        "stage_conv1x1_padded: dst_in len"
    );
    assert_eq!(
        dst_w.len(),
        out_c * padded_c,
        "stage_conv1x1_padded: dst_w len"
    );
    dst_in.fill(0);
    for p in 0..pixels {
        let dst = &mut dst_in[p * padded_c..p * padded_c + input_c];
        for (d, &x) in dst.iter_mut().zip(input[p * input_c..(p + 1) * input_c].iter()) {
            *d = x as u8; // bit-preserving i8→u8 re-interpret (VLD reads i8 lanes)
        }
    }
    for oc in 0..out_c {
        let row = &mut dst_w[oc * padded_c..(oc + 1) * padded_c];
        row[..input_c].copy_from_slice(&weights[oc * input_c..(oc + 1) * input_c]);
        row[input_c..].fill(0);
    }
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
/// Eligibility: stride/dilation 1, `out_h == in_h`, `out_w == in_w` (pad 0),
/// `in_c >= 1`, `out_c >= 1`, all pointers 16-byte aligned (direct path) or
/// sufficient scratch for the padded staging (T3.3), `scratch >= out_c * 4`.
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
        // T3.3 — the strict `accx_eligible_1x1` gate stays the unpadded fast
        // path; the padded gate accepts ANY `input_c >= 1` and stages a
        // zero-padded copy below when `input_c % 16 != 0`.
        || !crate::accx::accx_eligible_1x1_padded(input_c, out_c)
    {
        return Ok(false);
    }

    let input_offset = params.input_offset;
    // T3.3 — small / non-16 input channels: stage a zero-padded input copy AND
    // a zero-padded weight copy (the kernel VLDs 16-lane vectors and strides
    // weight rows by the padded channel count), then run the same kernel.
    // Mirrors the conv3x3/depthwise channel-pad carve and T3.6's fc pad.
    let padded_c = pad16(input_c);
    let needs_pad = padded_c != input_c;
    let pixels = in_h * in_w;
    // Padded layout (mirrors gemm.rs fc + conv3x3.rs):
    //   [padded input: pixels*padded_c][padded weights: out_c*padded_c][accs: out_c*4][wsum: out_c*4 if input_offset != 0]
    let pad_input_len = pixels * padded_c;
    let pad_weights_len = out_c * padded_c;
    let wsum_extra = if input_offset != 0 { out_c * 4 } else { 0 };
    let need = if needs_pad {
        pad_input_len + pad_weights_len + out_c * 4 + wsum_extra
    } else {
        out_c * 4 + wsum_extra
    };
    if ctx.scratch.len() < need {
        return Ok(false);
    }

    let in_ptr = ctx.input.as_ptr();
    let w_ptr = ctx.weights.as_ptr();
    let out_ptr = ctx.output.as_mut_ptr();
    let scratch_ptr = ctx.scratch.as_mut_ptr();
    let scratch_u = scratch_ptr as usize;

    let (k_in_ptr, k_w_ptr, accs, wsum, k_in_c);
    if needs_pad {
        // Padded buffers — carve from scratch at 16-byte boundaries so the
        // kernel's VLD.128 stays aligned (mirrors conv3x3.rs:190-195 and
        // gemm.rs's fc carve).
        let in_off = (scratch_u + 15) & !15;
        let w_off = in_off + pad_input_len;
        let accs_off = (w_off + pad_weights_len + 15) & !15;
        let p_in: *const i8 = unsafe { scratch_ptr.add(in_off - scratch_u) }.cast::<i8>();
        let p_w: *const i8 = unsafe { scratch_ptr.add(w_off - scratch_u) }.cast::<i8>();
        let p_accs = unsafe { scratch_ptr.add(accs_off - scratch_u) } as *mut i32;
        if (accs_off - scratch_u) % 4 != 0 {
            return Ok(false);
        }
        wsum = if input_offset != 0 {
            (unsafe { scratch_ptr.add(accs_off - scratch_u + out_c * 4) }) as *mut i32
        } else {
            core::ptr::null_mut()
        };
        let dst_in =
            unsafe { core::slice::from_raw_parts_mut(p_in as *mut u8, pad_input_len) };
        let dst_w = unsafe { core::slice::from_raw_parts_mut(p_w as *mut i8, pad_weights_len) };
        stage_conv1x1_padded(dst_in, dst_w, ctx.input, ctx.weights, input_c, out_c, pixels);
        k_in_ptr = p_in;
        k_w_ptr = p_w;
        accs = p_accs;
        k_in_c = padded_c;
    } else {
        if (in_ptr as usize) % 16 != 0
            || (w_ptr as usize) % 16 != 0
            || (out_ptr as usize) % 16 != 0
        {
            return Ok(false);
        }
        accs = scratch_ptr as *mut i32;
        if (accs as usize) % 4 != 0 {
            return Ok(false);
        }
        wsum = if input_offset != 0 {
            unsafe { accs.add(out_c) }
        } else {
            core::ptr::null_mut()
        };
        k_in_ptr = in_ptr;
        k_w_ptr = w_ptr;
        k_in_c = input_c;
    }
    if input_offset != 0 {
        let ws = unsafe { core::slice::from_raw_parts_mut(wsum, out_c) };
        let wv = unsafe { core::slice::from_raw_parts(k_w_ptr, out_c * k_in_c) };
        crate::accx::weight_sums_conv(ws, wv, 1, k_in_c);
    }

    let multipliers = params.output_multiplier_per_channel;
    let shifts = params.output_shift_per_channel;
    let act_min = params.quantized_activation_min;
    let act_max = params.quantized_activation_max;
    let out_offset = params.output_offset;
    let (uniform_mult, uniform_shift) = uniform;

    for oh in 0..out_h {
        for ow in 0..out_w {
            let px_in = (oh * in_w + ow) * k_in_c;
            let px_out = (oh * out_w + ow) * out_c;
            unsafe {
                crate::accx::accx_conv1x1(
                    k_in_ptr.add(px_in),
                    k_w_ptr,
                    accs,
                    k_in_c,
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
    if needs_pad {
        CONV1X1_PADDED_RAN.store(1, core::sync::atomic::Ordering::Relaxed);
    }
    Ok(true)
}

/// Last-padded-1×1-conv SIMD engagement flag (device diagnostic for
/// simd_validation). Mirrors `reductions::mean`'s `SIMD_MEAN_RAN` pattern.
///
/// `allow(dead_code)`: written only from the device-gated dispatch and read
/// only from the device-gated getter below, so host builds see it as unused.
#[allow(dead_code)]
static CONV1X1_PADDED_RAN: core::sync::atomic::AtomicU8 =
    core::sync::atomic::AtomicU8::new(0);

/// Whether the most recent 1×1 conv with a padded channel count (input_c not
/// a multiple of 16) took the SIMD path.
///
/// Host/QEMU builds never run the SIMD kernel, so this is always `false`
/// there; on real hardware it flips to `true` after an eligible padded conv
/// (the eligibility mirror alone cannot see runtime pointer alignment — the
/// atomic is set only when the staged SIMD dispatch actually ran).
pub fn conv1x1_padded_took_simd() -> bool {
    #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
    {
        CONV1X1_PADDED_RAN.load(core::sync::atomic::Ordering::Relaxed) != 0
    }
    #[cfg(not(all(target_arch = "xtensa", not(feature = "qemu"))))]
    {
        false
    }
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
        // T3.3 — SIMD engages for the strict direct gate (`accx_eligible_1x1`,
        // input_c >= 16 && %16) OR the widened pad-in-scratch gate
        // (`accx_eligible_1x1_padded`, any input_c >= 1 — the dispatch stages
        // a zero-padded copy when input_c % 16 != 0). Both gates stay
        // unchanged; this is the union of the two dispatch branches.
        let accx = cfg!(all(target_arch = "xtensa", not(feature = "qemu")))
            && (crate::accx::accx_eligible_1x1(input_c, out_channels)
                || crate::accx::accx_eligible_1x1_padded(input_c, out_channels));
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

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;
    use super::*;
    use hematite_core::op_params::{Conv2DParams, Padding};

    /// Host model of the `s8_accx_conv1x1` accumulation contract on the
    /// staged (padded) buffers: `acc[oc] = Σ_ic staged_w[oc*padded_c + ic] *
    /// staged_in[pixel*padded_c + ic]` in wrapping i32 — the exact
    /// GPR-accumulator arithmetic the asm uses (raw dot product, no
    /// input_offset).
    fn kernel_model_accs(
        staged_in: &[u8],
        staged_w: &[i8],
        padded_c: usize,
        out_c: usize,
        pixel: usize,
    ) -> Vec<i32> {
        let mut accs = vec![0i32; out_c];
        for oc in 0..out_c {
            let mut acc: i32 = 0;
            for ic in 0..padded_c {
                let iv = i32::from(staged_in[pixel * padded_c + ic] as i8);
                let wv = i32::from(staged_w[oc * padded_c + ic]);
                acc = acc.wrapping_add(iv.wrapping_mul(wv));
            }
            accs[oc] = acc;
        }
        accs
    }

    /// Run the full device SIMD pipeline in software — real
    /// [`stage_conv1x1_padded`] staging, the kernel-model accumulators, the
    /// real Phase-C `input_offset` fold, and the real `requantize_1x1`
    /// epilogue — producing one 1×1 conv output layer. This exercises the
    /// exact device pipeline code (pad + kernel contract + fold + requantize).
    fn simd_model_layer(
        input: &[i8],
        weights: &[i8],
        bias: &[i32],
        p: &Conv2DParams<'_>,
    ) -> Vec<i8> {
        let in_c = p.input_shape[3] as usize;
        let out_c = p.output_shape[3] as usize;
        let pixels = (p.input_shape[1] * p.input_shape[2]) as usize;
        let padded_c = pad16(in_c);
        let needs_pad = padded_c != in_c;

        let mut staged_in = vec![0u8; pixels * padded_c];
        let mut staged_w = vec![0i8; out_c * padded_c];
        if needs_pad {
            stage_conv1x1_padded(
                &mut staged_in,
                &mut staged_w,
                input,
                weights,
                in_c,
                out_c,
                pixels,
            );
        } else {
            for p in 0..pixels {
                for (d, &x) in staged_in[p * in_c..(p + 1) * in_c]
                    .iter_mut()
                    .zip(input[p * in_c..(p + 1) * in_c].iter())
                {
                    *d = x as u8; // bit-preserving i8→u8 re-interpret
                }
            }
            staged_w.copy_from_slice(weights);
        }

        let multipliers = p.output_multiplier_per_channel;
        let shifts = p.output_shift_per_channel;
        let (uniform_mult, uniform_shift) = match crate::accx::uniform_scale(multipliers, shifts) {
            Some((m, s)) => (m, s),
            None => (0, i32::MIN),
        };
        let mut output = vec![0i8; pixels * out_c];
        for px in 0..pixels {
            let mut accs = kernel_model_accs(&staged_in, &staged_w, padded_c, out_c, px);
            if p.input_offset != 0 {
                // Weight sums over the PADDED rows — pad lanes are zero, so
                // these equal the real per-row sums (the dispatch computes
                // them this way; mirror exactly).
                let mut wsum = vec![0i32; out_c];
                crate::accx::weight_sums_conv(&mut wsum, &staged_w, 1, padded_c);
                crate::depthwise::fold_input_offset(&mut accs, &wsum, p.input_offset);
            }
            crate::accx::requantize_1x1(&mut crate::accx::ReqCtx {
                accs: &accs,
                bias,
                multipliers,
                shifts,
                output_offset: p.output_offset,
                act_min: p.quantized_activation_min,
                act_max: p.quantized_activation_max,
                out_base: px * out_c,
                output: &mut output,
                uniform_mult,
                uniform_shift,
            });
        }
        output
    }

    fn per_channel_mult(n: usize) -> Vec<i32> {
        (0..n).map(|i| (1 << 30) - (i as i32) * 7919).collect()
    }

    fn per_channel_shift(n: usize) -> Vec<i32> {
        (0..n).map(|i| (i % 3) as i32).collect()
    }

    /// Deterministic pseudo-random `i8` pattern (full int8 range).
    fn pattern(seed: u32, n: usize) -> Vec<i8> {
        let mut out = vec![0i8; n];
        let mut x = seed;
        for v in out.iter_mut() {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *v = (x >> 16) as i8;
        }
        out
    }

    fn pattern_i32(seed: u32, n: usize) -> Vec<i32> {
        let mut out = vec![0i32; n];
        let mut x = seed;
        for v in out.iter_mut() {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *v = ((x >> 16) as i32) * 37 - 500;
        }
        out
    }

    /// Host bit-exact gate (T3.3): the device SIMD pipeline model (real
    /// staging + kernel-contract accumulation + real fold + real requantize)
    /// must equal the independent `hematite-ref` scalar conv2d for every
    /// small / non-16 input_c in {1, 3, 8, 15, 17, 32} — pad and no-pad
    /// paths — across spatial shapes, offsets, and identity / non-identity
    /// per-channel multipliers. Zero mismatches.
    #[test]
    fn conv1x1_small_simd_model_matches_ref_bit_exact() {
        let mut checked = 0;
        for &input_c in &[1, 3, 8, 15, 17, 32] {
            for &(h, w) in &[(1, 1), (4, 4), (2, 5)] {
                for &in_off in &[0, 5, 128] {
                    for mode in 0..3 {
                        let out_c = 16i32;
                        let n = out_c as usize;
                        let (mults, shifts): (Vec<i32>, Vec<i32>) = match mode {
                            0 => (vec![1 << 30; n], vec![1; n]),
                            1 => (per_channel_mult(n), per_channel_shift(n)),
                            _ => (vec![1 << 29; n], vec![0; n]),
                        };
                        let pixels = h * w;
                        let p = Conv2DParams {
                            input_shape: [1, h, w, input_c],
                            filter_shape: [out_c, 1, 1, input_c],
                            output_shape: [1, h, w, out_c],
                            padding: Padding::Same,
                            stride_width: 1,
                            stride_height: 1,
                            dilation_width_factor: 1,
                            dilation_height_factor: 1,
                            input_offset: in_off,
                            weights_offset: 0,
                            output_offset: if in_off == 0 { 0 } else { -10 },
                            output_multiplier_per_channel: &mults,
                            output_shift_per_channel: &shifts,
                            quantized_activation_min: if mode == 1 { 0 } else { -128 },
                            quantized_activation_max: 127,
                        };
                        let seed = 0x1C00_0000u32
                            | (input_c as u32 * 131 + h as u32 * 17 + w as u32);
                        let input = pattern(seed, pixels as usize * input_c as usize);
                        let weights =
                            pattern(0xE3A + input_c as u32 * 17, input_c as usize * n);
                        let bias = pattern_i32(0xFAC + out_c as u32, n);

                        let got = simd_model_layer(&input, &weights, &bias, &p);
                        let mut want = vec![0i8; got.len()];
                        hematite_ref::conv::conv2d(
                            &input,
                            &weights,
                            &bias,
                            &p,
                            &mut want,
                            &mut [],
                        )
                        .expect("ref conv2d accepts the shape");
                        assert_eq!(
                            got, want,
                            "input_c={input_c} h={h} w={w} in_off={in_off} mode={mode}: \
                             SIMD-model output must equal hematite-ref scalar"
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked >= 150, "small-conv matrix did not expand ({checked})");
    }

    /// The staging must produce the exact padded layout the kernel consumes:
    /// input pixels `[real | zeros]`, weight rows `[real row | zeros]`.
    #[test]
    fn stage_conv1x1_padded_zero_fills_pad_lanes() {
        let input: Vec<i8> = (0..16).map(|i| i as i8 - 3).collect(); // 2 pixels x 8 ch
        let weights: Vec<i8> = (0..32).map(|i| (i % 7) as i8 - 2).collect(); // 4 rows x 8
        let mut dst_in = vec![0xEEu8; 2 * 16];
        let mut dst_w = vec![0x7Fi8; 64]; // 4 rows x 16
        stage_conv1x1_padded(&mut dst_in, &mut dst_w, &input, &weights, 8, 4, 2);
        for p in 0..2 {
            let expect: Vec<u8> = input[p * 8..(p + 1) * 8].iter().map(|&x| x as u8).collect();
            assert_eq!(&dst_in[p * 16..p * 16 + 8], &expect[..]);
            assert_eq!(&dst_in[p * 16 + 8..(p + 1) * 16], &[0; 8], "pixel {p} pad lanes");
        }
        for oc in 0..4 {
            let row = &dst_w[oc * 16..(oc + 1) * 16];
            assert_eq!(&row[..8], &weights[oc * 8..(oc + 1) * 8]);
            assert_eq!(&row[8..], &[0; 8], "row {oc} pad lanes must be zero");
        }
    }
}
