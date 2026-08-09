// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! TIE728 SIMD correctness validation for every SIMD-accelerated kernel group
//! — companion to [`crate::model_validation`], same reporting style
//! (`PASS`/`FAIL` + fnv1a).
//!
//! * elementwise `add`/`mul`/`sub` and pool `avg`/`max` (the kernel groups
//!   that were NOT previously proven broken under QEMU);
//! * `relu`, `softmax`, `depthwise`, `conv1x1`, `conv3x3`, `fc`/`gemm`,
//!   `mean` — added in the simd-zoo-hardening Wave 2 (todo 6). These drive
//!   the PUBLIC s3 dispatch functions (the same entry points a model calls),
//!   so on real hardware the SIMD path is exercised; the `qemu` feature is
//!   gated off inside those dispatch functions (`EE.VSMULAS.S8.QACC.LD.INCP`
//!   is unemulated by this QEMU fork — see
//!   `local-notes/notepads/hematite-nn/problems.md`).
//!
//! # Call-site note (hardware-only suite)
//!
//! This suite is device-only: under `feature = "qemu"` the SIMD dispatch is
//! unreachable (scalar fallback everywhere) and QEMU's TIE728 emulation is
//! broken for the compute ops — the caller (firmware.rs) must skip
//! `validate_all` under the `qemu` feature. The first real-silicon run is
//! the Wave 2 device sweep; on QEMU this suite is non-terminating (see the
//! per-op bisection evidence below).
//!
//! # Result (see `local-notes/notepads/hematite-nn/problems.md`, same date, for the
//! full per-op bisection evidence)
//!
//! Every one of the 5 checks below fails under this QEMU fork
//! (`esp_develop_9.2.2_20260417`) — **not a code defect in this validation
//! module or in `hematite-s3`'s dispatch gates**, but a broader QEMU TIE728
//! emulation gap than the already-known `EE.VSMULAS.S8.QACC.LD.INCP` issue:
//!
//! * `add` (`EE.VADDS.S8`) — hard crash (double-exception fault storm,
//!   confirmed via an isolated hand-replica of the exact instruction
//!   sequence, independent of any Rust glue).
//! * `sub` (`EE.VSUBS.S8`) — does NOT crash, but silently computes the
//!   WRONG result (observed: output buffer stays all-zero instead of the
//!   saturated difference).
//! * `mul` (`EE.VMULAS.S8.QACC(.LD.IP)`) — never returns (crash or long
//!   hang, inconsistent across runs, but always fails to complete).
//! * `avg_pool` (`EE.VMULAS.S8.QACC.LD.IP`/`.LD.XP`) — hard crash.
//! * `max_pool` (`EE.VMAX.S8.LD.INCP`) — genuine infinite hang (confirmed
//!   still running after 55s wall time, no reboot signature).
//!
//! Because the FIRST op tested in any given boot crashes/hangs the whole
//! firmware, a single QEMU run cannot exercise all 5 checks — each result
//! above was captured from a SEPARATE isolated run (see
//! `tools/qemu-baseline/simd_dbg_{add,sub,mul,avgpool,maxpool}*.log`).
//! `validate_all` below still runs the full, intended 11-check suite (3
//! sizes × 3 elementwise ops + 2 pool ops) in its natural order — this is
//! the CORRECT code for real hardware or a fixed QEMU; it is simply
//! non-terminating on THIS QEMU build.
//!
//! # Ground truth
//!
//! The comparison target is the INDEPENDENT `hematite-ref` scalar kernel
//! ([`hematite_ref::elementwise`], [`hematite_ref::pool`]) — not
//! `hematite_s3`'s own scalar fallback, which lives in the same file as the
//! dispatch gate under test and would make the comparison circular.
//!
//! # Eligibility (mirrors the dispatch gates in `hematite-s3/src/{elementwise,pool}.rs`)
//!
//! * add/sub: `input1_offset = input2_offset = output_offset = 0`,
//!   full-range activation bounds, `left_shift <= 0`, and (input1/input2/output)
//!   `(multiplier, shift) = (1<<30, 1)` — the exact identity pair the scalar
//!   loop itself special-cases (`multiply_by_quantized_multiplier(x, 1<<30, 1)`
//!   is an exact identity — see `hematite-int8`'s doc/derivation), so the raw
//!   SIMD add/sub instructions and the scalar identity path must match
//!   bit-for-bit. Lengths 16 / 32 / 48 exercise the single-chunk and
//!   multi-chunk (`c_div_x_1` > 0) cases.
//! * mul: same offset/bounds contract, `output_multiplier = 1<<30`,
//!   `output_shift = 1` ⇒ `mul_shift = 1 - output_shift = 0` (no shift) —
//!   also an exact identity on the requantize step.
//! * pool: exactly 2×2 filter / stride 2 / no padding / channels a multiple
//!   of 16 / full-range activation bounds (pool's SIMD args struct has no
//!   clamp field at all, so a non-full range would silently diverge).

use hematite_core::op_params::{
    ActivationParams, Conv2DParams, DepthwiseConv2DParams, ElementwiseParams,
    FullyConnectedParams, FusedActivation, Padding, PoolParams, ReduceParams, SoftmaxParams,
};

// ── Reporting (mirrors `model_validation.rs::{fnv1a, compare, report}`) ────

fn fnv1a(data: &[i8]) -> u32 {
    let mut h: u32 = 2_166_136_261;
    for &b in data {
        h ^= u32::from(b as u8);
        h = h.wrapping_mul(16_777_619);
    }
    h
}

struct CheckResult {
    name: &'static str,
    pass: bool,
    mismatch: Option<(usize, i8, i8)>,
    fnv: u32,
}

fn compare(name: &'static str, got: &[i8], want: &[i8]) -> CheckResult {
    let fnv = fnv1a(got);
    if got.len() != want.len() {
        return CheckResult {
            name,
            pass: false,
            mismatch: Some((0, 0, 0)),
            fnv,
        };
    }
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        if g != w {
            return CheckResult {
                name,
                pass: false,
                mismatch: Some((i, g, w)),
                fnv,
            };
        }
    }
    CheckResult {
        name,
        pass: true,
        mismatch: None,
        fnv,
    }
}

fn report(r: &CheckResult) {
    match (r.pass, r.mismatch) {
        (true, _) => {
            crate::firmware::firmware_log!("simd {}: PASS (fnv=0x{:08x})", r.name, r.fnv);
        }
        (false, Some((i, g, w))) => {
            crate::firmware::firmware_log!(
                "simd {}: FAIL at idx {}: got={} want={} (fnv=0x{:08x})",
                r.name,
                i,
                g,
                w,
                r.fnv,
            );
        }
        (false, None) => {
            crate::firmware::firmware_log!(
                "simd {}: FAIL (length mismatch) (fnv=0x{:08x})",
                r.name,
                r.fnv
            );
        }
    }
}

/// Deterministic LCG-based `i8` pattern generator — no real randomness
/// needed, just reproducible test data covering the full int8 range
/// (including boundary/saturation values) without depending on `std`.
const fn make_pattern<const N: usize>(seed: u32) -> [i8; N] {
    let mut out = [0i8; N];
    let mut x = seed;
    let mut i = 0;
    while i < N {
        x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        out[i] = (x >> 16) as i8;
        i += 1;
    }
    out
}

/// 16-byte-aligned buffer wrapper — required by every SIMD entry point's
/// pointer-alignment precondition (`EE.VLD.128.IP` / `EE.VST.128.IP`).
#[repr(align(16))]
struct Aligned<const N: usize>([i8; N]);

/// 16-byte-aligned byte scratch buffer — the conv-family / softmax SIMD
/// dispatches stage padded copies, accumulator banks and exp caches in
/// `scratch`; 16-alignment lets their internal carves round sub-offsets.
#[repr(align(16))]
struct AlignedBytes<const N: usize>([u8; N]);

/// Deterministic per-channel bias fill (non-zero, small enough to keep the
/// i32 accumulators far from overflow; identical inputs for both kernels).
const fn bias_pattern<const N: usize>() -> [i32; N] {
    let mut out = [0i32; N];
    let mut i = 0;
    while i < N {
        out[i] = (i as i32) * 37 - 500;
        i += 1;
    }
    out
}

// ── Per-channel requantize slices (Q0.31 0.5, shift 0 — spec.rs fill style) ─

const MULT_16: [i32; 16] = [1 << 30; 16];
const SHIFT_16: [i32; 16] = [0; 16];
const MULT_32: [i32; 32] = [1 << 30; 32];
const SHIFT_32: [i32; 32] = [0; 32];
const MULT_64: [i32; 64] = [1 << 30; 64];
const SHIFT_64: [i32; 64] = [0; 64];

// ── Elementwise: add / mul / sub at N = 16 / 32 / 48 ────────────────────────

enum ElemOp {
    Add,
    Mul,
    Sub,
}

fn elementwise_params(n: i32) -> ElementwiseParams {
    ElementwiseParams {
        num_elements: n,
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
    }
}

fn run_elementwise_check<const N: usize>(op: ElemOp, name: &'static str) {
    let input1 = Aligned(make_pattern::<N>(0x1234_5678));
    let input2 = Aligned(make_pattern::<N>(0x9ABC_DEF1));
    let mut want = Aligned([0i8; N]);
    let mut got = Aligned([0i8; N]);
    let params = elementwise_params(N as i32);

    match op {
        ElemOp::Add => {
            hematite_ref::elementwise::add(&input1.0, &input2.0, &params, &mut want.0, &mut [])
                .expect("harness: ref add shape");
            unsafe {
                hematite_s3::elementwise::add_simd_aligned(
                    got.0.as_mut_ptr(),
                    input1.0.as_ptr(),
                    input2.0.as_ptr(),
                    N as u32,
                );
            }
        }
        ElemOp::Sub => {
            hematite_ref::elementwise::sub(&input1.0, &input2.0, &params, &mut want.0, &mut [])
                .expect("harness: ref sub shape");
            unsafe {
                hematite_s3::elementwise::sub_simd_aligned(
                    got.0.as_mut_ptr(),
                    input1.0.as_ptr(),
                    input2.0.as_ptr(),
                    N as u32,
                );
            }
        }
        ElemOp::Mul => {
            hematite_ref::elementwise::mul(&input1.0, &input2.0, &params, &mut want.0, &mut [])
                .expect("harness: ref mul shape");
            let mul_shift = 1 - params.output_shift;
            unsafe {
                hematite_s3::elementwise::mul_simd_aligned(
                    got.0.as_mut_ptr(),
                    input1.0.as_ptr(),
                    input2.0.as_ptr(),
                    N as u32,
                    mul_shift,
                );
            }
        }
    }

    report(&compare(name, &got.0, &want.0));
}

// ── Pool: avg / max, 2×2 stride-2 no-pad, 4x4x16 -> 2x2x16 ──────────────────

fn pool_params() -> PoolParams {
    PoolParams {
        input_shape: [1, 4, 4, 16],
        output_shape: [1, 2, 2, 16],
        filter_width: 2,
        filter_height: 2,
        stride_width: 2,
        stride_height: 2,
        padding: Padding::Valid,
        activation: FusedActivation::None,
        quantized_activation_min: i8::MIN as i32,
        quantized_activation_max: i8::MAX as i32,
    }
}

fn run_avg_pool_check() {
    let input = Aligned(make_pattern::<256>(0x0BAD_C0DE));
    let mut want = Aligned([0i8; 64]);
    let mut got = Aligned([0i8; 64]);
    let params = pool_params();

    hematite_ref::pool::average_pool_2d(&input.0, &params, &mut want.0, &mut [])
        .expect("harness: ref average_pool_2d shape");

    let channels = 16i32;
    let input_w = 4i32;
    let total_out = 2 * 2 * channels;
    let area_inv = [64i8; 16]; // round(2^8 / 4) for a 2x2 (area=4) filter
    unsafe {
        hematite_s3::pool::avg_pool_2d_simd(
            got.0.as_mut_ptr(),
            input.0.as_ptr(),
            channels,
            input_w * channels,
            channels,
            8,
            &area_inv,
            total_out / 16 - 1,
        );
    }

    report(&compare("avg_pool_2x2", &got.0, &want.0));
}

fn run_max_pool_check() {
    let input = Aligned(make_pattern::<256>(0xFEED_FACE));
    let mut want = Aligned([0i8; 64]);
    let mut got = Aligned([0i8; 64]);
    let params = pool_params();

    hematite_ref::pool::max_pool_2d(&input.0, &params, &mut want.0, &mut [])
        .expect("harness: ref max_pool_2d shape");

    let channels = 16i32;
    let input_w = 4i32;
    let total_out = 2 * 2 * channels;
    unsafe {
        hematite_s3::pool::max_pool_2d_simd(
            got.0.as_mut_ptr(),
            input.0.as_ptr(),
            channels,
            input_w * channels,
            channels,
            total_out / 16 - 1,
        );
    }

    report(&compare("max_pool_2x2", &got.0, &want.0));
}

// ── Relu: 256 elems, identity requantize (spec.rs SIMD_RELU_256_PARAMS) ─────

fn check_relu_simd_matches_ref() {
    let input = Aligned(make_pattern::<256>(0xCAFE_BEEF));
    let mut want = Aligned([0i8; 256]);
    let mut got = Aligned([0i8; 256]);
    let params = ActivationParams {
        input_offset: 0,
        output_offset: 0,
        output_multiplier: 1 << 30,
        output_shift: 1,
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
    };

    hematite_ref::activation::relu(&input.0, &params, &mut want.0, &mut [])
        .expect("harness: ref relu shape");
    hematite_s3::activations::relu(&input.0, &params, &mut got.0, &mut [])
        .expect("harness: s3 relu shape");

    report(&compare("relu_256", &got.0, &want.0));
}

// ── Softmax: 1x1000 (spec.rs SOFTMAX_1X1000_PARAMS) ─────────────────────────

fn check_softmax_simd_matches_ref() {
    let input = Aligned(make_pattern::<1000>(0x50F7_4A11));
    let mut want = Aligned([0i8; 1000]);
    let mut got = Aligned([0i8; 1000]);
    let mut scratch = AlignedBytes([0u8; 4000]);
    let params = SoftmaxParams {
        num_rows: 1,
        row_size: 1000,
        input_multiplier: 1_717_986_918, // quantize_multiplier(0.1), softmax golden
        input_left_shift: 22,
        diff_min: -248, // TFLM -CalculateInputRadius(5, 23) — see spec.rs SOFTMAX_1X1000_PARAMS
        input_offset: 0,
        output_offset: -128,
        quantized_activation_min: -128,
        quantized_activation_max: 127,
    };

    hematite_ref::softmax::softmax(&input.0, &params, &mut want.0, &mut scratch.0)
        .expect("harness: ref softmax shape");
    hematite_s3::softmax::softmax(&input.0, &params, &mut got.0, &mut scratch.0)
        .expect("harness: s3 softmax shape");

    report(&compare("softmax_1x1000", &got.0, &want.0));
}

// ── Depthwise: 12x12x16 stride-2 SAME (spec.rs SIMD_DEPTHWISE_S2_SAME_PARAMS) ─

const DEPTHWISE_IN: usize = 12 * 12 * 16;
const DEPTHWISE_W: usize = 1 * 3 * 3 * 16;
const DEPTHWISE_OUT: usize = 6 * 6 * 16;

fn check_depthwise_simd_matches_ref() {
    let input = Aligned(make_pattern::<DEPTHWISE_IN>(0xD3A7_51E5));
    let weights = Aligned(make_pattern::<DEPTHWISE_W>(0xBEE5_7EED));
    let bias = bias_pattern::<16>();
    let mut want = Aligned([0i8; DEPTHWISE_OUT]);
    let mut got = Aligned([0i8; DEPTHWISE_OUT]);
    let mut scratch = AlignedBytes([0u8; 4096]);
    let params = DepthwiseConv2DParams {
        input_shape: [1, 12, 12, 16],
        filter_shape: [1, 3, 3, 16],
        output_shape: [1, 6, 6, 16],
        padding: Padding::Same,
        stride_width: 2,
        stride_height: 2,
        dilation_width_factor: 1,
        dilation_height_factor: 1,
        depth_multiplier: 1,
        input_offset: 0,
        weights_offset: 0,
        output_offset: 0,
        output_multiplier_per_channel: &MULT_16,
        output_shift_per_channel: &SHIFT_16,
        quantized_activation_min: -128,
        quantized_activation_max: 127,
    };

    hematite_ref::depthwise_conv::depthwise_conv2d(
        &input.0,
        &weights.0,
        &bias,
        &params,
        &mut want.0,
        &mut scratch.0,
    )
    .expect("harness: ref depthwise shape");
    hematite_s3::depthwise::depthwise_conv2d(
        &input.0,
        &weights.0,
        &bias,
        &params,
        &mut got.0,
        &mut scratch.0,
    )
    .expect("harness: s3 depthwise shape");

    report(&compare("depthwise_12x12x16_s2", &got.0, &want.0));
}

// ── Conv 1x1: 64x1x1x64 (spec.rs EMBER_CONV_1X1_64_PARAMS) ─────────────────

const CONV1X1_IN: usize = 1 * 1 * 64;
const CONV1X1_W: usize = 64 * 1 * 1 * 64;
const CONV1X1_OUT: usize = 1 * 1 * 64;

fn check_conv1x1_simd_matches_ref() {
    let input = Aligned(make_pattern::<CONV1X1_IN>(0xC0FF_EE11));
    let weights = Aligned(make_pattern::<CONV1X1_W>(0x1CE5_51CE));
    let bias = bias_pattern::<64>();
    let mut want = Aligned([0i8; CONV1X1_OUT]);
    let mut got = Aligned([0i8; CONV1X1_OUT]);
    let mut scratch = AlignedBytes([0u8; 512]);
    let params = Conv2DParams {
        input_shape: [1, 1, 1, 64],
        filter_shape: [64, 1, 1, 64],
        output_shape: [1, 1, 1, 64],
        padding: Padding::Same,
        stride_width: 1,
        stride_height: 1,
        dilation_width_factor: 1,
        dilation_height_factor: 1,
        input_offset: 0,
        weights_offset: 0,
        output_offset: 0,
        output_multiplier_per_channel: &MULT_64,
        output_shift_per_channel: &SHIFT_64,
        quantized_activation_min: -128,
        quantized_activation_max: 127,
    };

    hematite_ref::conv::conv2d(
        &input.0,
        &weights.0,
        &bias,
        &params,
        &mut want.0,
        &mut scratch.0,
    )
    .expect("harness: ref conv1x1 shape");
    hematite_s3::conv1x1::conv2d_1x1(
        &input.0,
        &weights.0,
        &bias,
        &params,
        &mut got.0,
        &mut scratch.0,
    )
    .expect("harness: s3 conv1x1 shape");

    report(&compare("conv1x1_64x1x1x64", &got.0, &want.0));
}

// ── Conv 3x3: 16x16x32 SAME (spec.rs SIMD_CONV3X3_SAME_PARAMS) ─────────────

const CONV3X3_IN: usize = 16 * 16 * 32;
const CONV3X3_W: usize = 32 * 3 * 3 * 32;
const CONV3X3_OUT: usize = 16 * 16 * 32;

fn check_conv3x3_simd_matches_ref() {
    let input = Aligned(make_pattern::<CONV3X3_IN>(0x3EE7_5EED));
    let weights = Aligned(make_pattern::<CONV3X3_W>(0xACC5_10A));
    let bias = bias_pattern::<32>();
    let mut want = Aligned([0i8; CONV3X3_OUT]);
    let mut got = Aligned([0i8; CONV3X3_OUT]);
    let mut scratch = AlignedBytes([0u8; 20480]);
    let params = Conv2DParams {
        input_shape: [1, 16, 16, 32],
        filter_shape: [32, 3, 3, 32],
        output_shape: [1, 16, 16, 32],
        padding: Padding::Same,
        stride_width: 1,
        stride_height: 1,
        dilation_width_factor: 1,
        dilation_height_factor: 1,
        input_offset: 0,
        weights_offset: 0,
        output_offset: 0,
        output_multiplier_per_channel: &MULT_32,
        output_shift_per_channel: &SHIFT_32,
        quantized_activation_min: -128,
        quantized_activation_max: 127,
    };

    hematite_ref::conv::conv2d(
        &input.0,
        &weights.0,
        &bias,
        &params,
        &mut want.0,
        &mut scratch.0,
    )
    .expect("harness: ref conv3x3 shape");
    hematite_s3::conv3x3::conv2d_3x3(
        &input.0,
        &weights.0,
        &bias,
        &params,
        &mut got.0,
        &mut scratch.0,
    )
    .expect("harness: s3 conv3x3 shape");

    report(&compare("conv3x3_16x16x32_same", &got.0, &want.0));
}

// ── FC: 256x64 (spec.rs SIMD_FC_256X64_PARAMS) ─────────────────────────────

const FC_W: usize = 64 * 256;

fn check_fc_simd_matches_ref() {
    let input = Aligned(make_pattern::<256>(0xFAC1_5ED7));
    let weights = Aligned(make_pattern::<FC_W>(0xD07_5C0DE));
    let bias = bias_pattern::<64>();
    let mut want = Aligned([0i8; 64]);
    let mut got = Aligned([0i8; 64]);
    let mut scratch = AlignedBytes([0u8; 512]);
    let params = FullyConnectedParams {
        input_dim: 256,
        output_dim: 64,
        input_offset: 0,
        weights_offset: 0,
        output_offset: 0,
        output_multiplier_per_channel: &MULT_64,
        output_shift_per_channel: &SHIFT_64,
        quantized_activation_min: -128,
        quantized_activation_max: 127,
    };

    hematite_ref::fully_connected::fully_connected(
        &input.0,
        &weights.0,
        &bias,
        &params,
        &mut want.0,
        &mut scratch.0,
    )
    .expect("harness: ref fc shape");
    hematite_s3::gemm::fully_connected(
        &input.0,
        &weights.0,
        &bias,
        &params,
        &mut got.0,
        &mut scratch.0,
    )
    .expect("harness: s3 fc shape");

    report(&compare("fc_256x64", &got.0, &want.0));
}

// ── Mean: [1,2,2,4] over H,W axes → [1,1,1,4] ──────────────────────────────
//
// s3 `reductions::mean` is scalar for now (the SIMD mean kernel is a later
// wave); the check lives here so the reduction is covered by the suite the
// moment the SIMD dispatch lands.

fn check_mean_simd_matches_ref() {
    let input = Aligned(make_pattern::<16>(0x1D4A_7EE5));
    let mut want = Aligned([0i8; 4]);
    let mut got = Aligned([0i8; 4]);
    let params = ReduceParams {
        keep_dims: false,
        axis: [1, 2, 0, 0],
        axis_count: 2,
        input_shape: [1, 2, 2, 4],
        output_shape: [1, 1, 1, 4],
        output_type: 0,
        input_offset: 0,
        output_offset: 0,
        output_multiplier: 1 << 30,
        output_shift: 1,
        quantized_activation_min: -128,
        quantized_activation_max: 127,
    };

    hematite_ref::reductions::mean(&input.0, &params, &mut want.0)
        .expect("harness: ref mean shape");
    hematite_s3::reductions::mean(&input.0, &params, &mut got.0)
        .expect("harness: s3 mean shape");

    report(&compare("mean_hw_2x2x4", &got.0, &want.0));
}

/// Run all SIMD correctness checks. Called from the firmware boot flow
/// right after [`crate::model_validation::validate_all`] (same rationale:
/// every PASS/FAIL line must print even if a later benchmark row panics).
///
/// See the module doc for why a single QEMU run currently cannot exercise
/// all 11 checks below (the first one crashes/hangs the whole firmware) —
/// this is nonetheless the correct, intended full suite for real hardware
/// or a fixed QEMU build.
pub fn validate_all() {
    crate::firmware::firmware_log!(
        "=== SIMD CORRECTNESS (elementwise + pool + relu/softmax/depthwise/conv/fc/mean, vs hematite-ref) ==="
    );
    run_elementwise_check::<16>(ElemOp::Add, "add_n16");
    run_elementwise_check::<32>(ElemOp::Add, "add_n32");
    run_elementwise_check::<48>(ElemOp::Add, "add_n48");
    run_elementwise_check::<16>(ElemOp::Mul, "mul_n16");
    run_elementwise_check::<32>(ElemOp::Mul, "mul_n32");
    run_elementwise_check::<48>(ElemOp::Mul, "mul_n48");
    run_elementwise_check::<16>(ElemOp::Sub, "sub_n16");
    run_elementwise_check::<32>(ElemOp::Sub, "sub_n32");
    run_elementwise_check::<48>(ElemOp::Sub, "sub_n48");
    run_avg_pool_check();
    run_max_pool_check();
    check_relu_simd_matches_ref();
    check_softmax_simd_matches_ref();
    check_depthwise_simd_matches_ref();
    check_conv1x1_simd_matches_ref();
    check_conv3x3_simd_matches_ref();
    check_fc_simd_matches_ref();
    check_mean_simd_matches_ref();
    crate::firmware::firmware_log!("=== SIMD CORRECTNESS DONE ===");
}
