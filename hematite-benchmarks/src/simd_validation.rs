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
//!   `mean` — added in the simd-zoo-hardening Wave 2 (todo 6). ALL 18 checks
//!   drive the PUBLIC s3 dispatch functions (the same entry points a model
//!   calls), so on real hardware the SIMD path is exercised; the `qemu`
//!   feature is gated off inside those dispatch functions
//!   (`EE.VSMULAS.S8.QACC.LD.INCP` is unemulated by this QEMU fork — see
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
//! The checks pass params through the PUBLIC dispatch functions, which engage
//! SIMD only when these gates hold:
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
//!
//! On device the SIMD path engages for every check (gates above + 16-aligned
//! `Aligned<>` buffers). The raw `*_simd_aligned` / `*_pool_2d_simd` helpers
//! are NOT called directly: their inline-asm register delivery was the
//! first-silicon corruption finding (task 8 — see
//! `hematite-s3/src/elementwise.rs` register-hazard note), and the pool raw
//! helpers are the Xtensa-LLVM multi-arg-call scramble class. The public
//! dispatch routes through the device-proven paths (`avg_pool_2d_simd_ctx`).

use hematite_core::op_params::{
    ActivationEpilogueParams, ActivationParams, ComposedActivation, Conv2DParams,
    DepthwiseConv2DParams, ElementwiseChainParams, ElementwiseChainStep, ElementwiseKind,
    ElementwiseParams, FoldedPoolParams, FusedConvParams, FullyConnectedParams,
    FusedActivation, Padding, PoolInputFold, PoolKind, PoolParams, ReduceParams,
    ResidualAddParams, SoftmaxParams,
};
use hematite_core::FusedKernelBackend;

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
            crate::firmware::uart0_log!("simd {}: PASS (fnv=0x{:08x})", r.name, r.fnv);
        }
        (false, Some((i, g, w))) => {
            crate::firmware::uart0_log!(
                "simd {}: FAIL at idx {}: got={} want={} (fnv=0x{:08x})",
                r.name,
                i,
                g,
                w,
                r.fnv,
            );
        }
        (false, None) => {
            crate::firmware::uart0_log!(
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

/// Carve `len` bytes from the SRAM bench arena at `*off`, advancing `*off`.
///
/// The conv/softmax/depthwise/fc checks' buffers live here — NOT on the
/// stack. With all 18 checks inlined, validate_all's frame pushes SP into
/// the arena's top bytes; a stack-local buffer whose slot straddles the
/// arena end writes pattern bytes past it into bss, clobbering the defmt
/// `RTT_ENCODER.taken` flag ("defmt logger taken reentrantly" — task-8/18
/// device findings). The arena is unused during validation (the kernel
/// benches carve it afterwards).
///
/// SAFETY: single-threaded firmware; each check carves once, sequentially.
/// The base is 16-aligned; callers whose kernels require 16-aligned inputs
/// must keep `*off` at a multiple of 16 (all check carve sizes are).
fn arena_carve(off: &mut usize, len: usize) -> &'static mut [u8] {
    // SAFETY: single-threaded firmware; sequential check execution; the
    // returned slice aliases arena memory only for the current check's
    // lifetime, matching the accepted conv3x3 carve pattern.
    unsafe {
        let arena = &mut *core::ptr::addr_of_mut!(crate::firmware::SRAM_ARENA);
        let p = arena.0.as_mut_ptr().add(*off);
        *off += len;
        core::slice::from_raw_parts_mut(p, len)
    }
}

/// i8 variant of [`arena_carve`] — kernels take `&mut [i8]` tensors.
fn arena_carve_i8(off: &mut usize, len: usize) -> &'static mut [i8] {
    // SAFETY: same contract as `arena_carve`; the carve base is 16-aligned
    // and every check carve size is a multiple of 16, so the i8 re-interpret
    // is alignment-preserving.
    unsafe { core::slice::from_raw_parts_mut(arena_carve(off, len).as_mut_ptr().cast(), len) }
}

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
const MULT_8: [i32; 8] = [1 << 30; 8];
const SHIFT_8: [i32; 8] = [0; 8];
const MULT_32: [i32; 32] = [1 << 30; 32];
const SHIFT_32: [i32; 32] = [0; 32];
const MULT_64: [i32; 64] = [1 << 30; 64];
const SHIFT_64: [i32; 64] = [0; 64];
const MULT_128: [i32; 128] = [1 << 30; 128];
const SHIFT_128: [i32; 128] = [0; 128];

// ── Elementwise: add / mul / sub at N = 16 / 32 / 48 ────────────────────────

#[derive(Clone, Copy)]
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

    // Both sides go through the PUBLIC dispatch: hematite-ref is the oracle,
    // hematite-s3's `add`/`mul`/`sub` take the SIMD path on device (identity
    // quant-affine params + 16-aligned Aligned<> buffers satisfy every
    // eligibility gate). The raw `*_simd_aligned` helpers are NOT called
    // here — their inline-asm register delivery is the latent hazard found
    // on first silicon (see hematite-s3/src/elementwise.rs, task-8 note).
    match op {
        ElemOp::Add => {
            hematite_ref::elementwise::add(&input1.0, &input2.0, &params, &mut want.0, &mut [])
                .expect("harness: ref add shape");
            hematite_s3::elementwise::add(&input1.0, &input2.0, &params, &mut got.0, &mut [])
                .expect("harness: s3 add shape");
        }
        ElemOp::Sub => {
            hematite_ref::elementwise::sub(&input1.0, &input2.0, &params, &mut want.0, &mut [])
                .expect("harness: ref sub shape");
            hematite_s3::elementwise::sub(&input1.0, &input2.0, &params, &mut got.0, &mut [])
                .expect("harness: s3 sub shape");
        }
        ElemOp::Mul => {
            hematite_ref::elementwise::mul(&input1.0, &input2.0, &params, &mut want.0, &mut [])
                .expect("harness: ref mul shape");
            hematite_s3::elementwise::mul(&input1.0, &input2.0, &params, &mut got.0, &mut [])
                .expect("harness: s3 mul shape");
        }
    }

    report(&compare(name, &got.0, &want.0));
}

// ── T3.2 widened elementwise: non-identity offsets/multipliers (256) ────────

/// The two-stage TFLM Add rounding params (non-zero zps, left_shift 20,
/// per-input multipliers, output requantize, clamped range) — NOT
/// identity-eligible, so on device these checks engage the widened 16-wide
/// per-lane model (`add_simd_lanes`/`mul_simd_lanes`/`sub_simd_lanes`).
fn non_identity_elementwise_params(n: i32, op: ElemOp) -> ElementwiseParams {
    let (input1_offset, input2_offset, output_offset, om, os, ls, i1m, i1s, i2m, i2s, act_min, act_max) =
        match op {
            ElemOp::Add | ElemOp::Sub => (
                -5, 3, 1, 1_342_177_280, -18, 20, 1 << 30, 0, 1_288_490_189, -1, -32, 96,
            ),
            ElemOp::Mul => (
                2, -3, -7, 1_717_986_918, -3, 0, 0, 0, 0, 0, -16, 111,
            ),
        };
    ElementwiseParams {
        num_elements: n,
        input1_offset,
        input2_offset,
        output_offset,
        output_multiplier: om,
        output_shift: os,
        left_shift: ls,
        input1_multiplier: i1m,
        input1_shift: i1s,
        input2_multiplier: i2m,
        input2_shift: i2s,
        quantized_activation_min: act_min,
        quantized_activation_max: act_max,
    }
}

fn run_elementwise_non_identity_check<const N: usize>(op: ElemOp, name: &'static str) {
    let input1 = Aligned(make_pattern::<N>(0xABCD_0001));
    let input2 = Aligned(make_pattern::<N>(0x1234_5678));
    let mut want = Aligned([0i8; N]);
    let mut got = Aligned([0i8; N]);
    let params = non_identity_elementwise_params(N as i32, op);

    // Both sides go through the PUBLIC dispatch: hematite-ref is the oracle,
    // hematite-s3's `add`/`mul`/`sub` take the widened lane-model path on
    // device (non-identity params + 16-aligned Aligned<> buffers satisfy the
    // T3.2 gate: any params, n ≥ 16, 16-aligned pointers).
    match op {
        ElemOp::Add => {
            hematite_ref::elementwise::add(&input1.0, &input2.0, &params, &mut want.0, &mut [])
                .expect("harness: ref non-identity add shape");
            hematite_s3::elementwise::add(&input1.0, &input2.0, &params, &mut got.0, &mut [])
                .expect("harness: s3 non-identity add shape");
        }
        ElemOp::Sub => {
            hematite_ref::elementwise::sub(&input1.0, &input2.0, &params, &mut want.0, &mut [])
                .expect("harness: ref non-identity sub shape");
            hematite_s3::elementwise::sub(&input1.0, &input2.0, &params, &mut got.0, &mut [])
                .expect("harness: s3 non-identity sub shape");
        }
        ElemOp::Mul => {
            hematite_ref::elementwise::mul(&input1.0, &input2.0, &params, &mut want.0, &mut [])
                .expect("harness: ref non-identity mul shape");
            hematite_s3::elementwise::mul(&input1.0, &input2.0, &params, &mut got.0, &mut [])
                .expect("harness: s3 non-identity mul shape");
        }
    }

    report(&compare(name, &got.0, &want.0));
}

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
    // PUBLIC dispatch (SIMD on device — the pool dispatch routes through the
    // device-proven `avg_pool_2d_simd_ctx`). The raw 8-arg `avg_pool_2d_simd`
    // helper is NOT called here (Xtensa-LLVM multi-arg-call scramble class).
    hematite_s3::pool::average_pool_2d(&input.0, &params, &mut got.0, &mut [])
        .expect("harness: s3 average_pool_2d shape");

    report(&compare("avg_pool_2x2", &got.0, &want.0));
}

fn run_max_pool_check() {
    let input = Aligned(make_pattern::<256>(0xFEED_FACE));
    let mut want = Aligned([0i8; 64]);
    let mut got = Aligned([0i8; 64]);
    let params = pool_params();

    hematite_ref::pool::max_pool_2d(&input.0, &params, &mut want.0, &mut [])
        .expect("harness: ref max_pool_2d shape");
    // PUBLIC dispatch (see run_avg_pool_check note).
    hematite_s3::pool::max_pool_2d(&input.0, &params, &mut got.0, &mut [])
        .expect("harness: s3 max_pool_2d shape");

    report(&compare("max_pool_2x2", &got.0, &want.0));
}

// ── Generic pool (T3.1): 3×3 stride-1 VALID 8x8x16 -> 6x6x16 ────────────────
//
// The widened `simd_eligible_pool` gate accepts any filter/stride with
// pad_total ≤ 0, so these shapes dispatch the `*_hwc1` SIMD kernels on
// device. Max semantics equal the ref bit-exact (PASS expected); avg runs
// the documented fixed-point semantics (the ±1 known-delta class — see
// `hematite-s3/src/pool.rs` module doc and `local-notes/evidence/composed-kernels/
// t31-pool.md`). The relu-clamp shape exercises the Rust clamp post-pass.

const GEN_POOL_IN: usize = 8 * 8 * 16;
const GEN_POOL_OUT: usize = 6 * 6 * 16;

fn generic_pool_params(act_min: i32) -> PoolParams {
    PoolParams {
        input_shape: [1, 8, 8, 16],
        output_shape: [1, 6, 6, 16],
        filter_width: 3,
        filter_height: 3,
        stride_width: 1,
        stride_height: 1,
        padding: Padding::Valid,
        activation: FusedActivation::None,
        quantized_activation_min: act_min,
        quantized_activation_max: 127,
    }
}

fn run_generic_max_pool_checks() {
    for (act_min, name) in [(i8::MIN as i32, "max_pool_3x3_s1"), (0, "max_pool_3x3_s1_relu")] {
        let input = Aligned(make_pattern::<GEN_POOL_IN>(0x1A7E_5EED));
        let mut want = Aligned([0i8; GEN_POOL_OUT]);
        let mut got = Aligned([0i8; GEN_POOL_OUT]);
        let params = generic_pool_params(act_min);

        hematite_ref::pool::max_pool_2d(&input.0, &params, &mut want.0, &mut [])
            .expect("harness: ref generic max_pool_2d shape");
        hematite_s3::pool::max_pool_2d(&input.0, &params, &mut got.0, &mut [])
            .expect("harness: s3 generic max_pool_2d shape");

        report(&compare(name, &got.0, &want.0));
    }
}

fn run_generic_avg_pool_check() {
    let input = Aligned(make_pattern::<GEN_POOL_IN>(0x2BAD_F00D));
    let mut want = Aligned([0i8; GEN_POOL_OUT]);
    let mut got = Aligned([0i8; GEN_POOL_OUT]);
    let params = generic_pool_params(i8::MIN as i32);

    hematite_ref::pool::average_pool_2d(&input.0, &params, &mut want.0, &mut [])
        .expect("harness: ref generic average_pool_2d shape");
    hematite_s3::pool::average_pool_2d(&input.0, &params, &mut got.0, &mut [])
        .expect("harness: s3 generic average_pool_2d shape");

    // The avg fixed-point semantics may diverge ±1 LSB from ref (the
    // documented known-delta class, same as `avg_pool_2x2`).
    report(&compare("avg_pool_3x3_s1", &got.0, &want.0));
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

// ── T3.2 relu6 / hard_swish: the widened activation lane models (256) ───────

fn activation_params(input_offset: i32, output_offset: i32, act_max: i32) -> ActivationParams<'static> {
    ActivationParams {
        input_offset,
        output_offset,
        output_multiplier: 0,
        output_shift: 0,
        quantized_activation_min: 0,
        quantized_activation_max: act_max,
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

/// relu6 — the widened lane model (vector min/max clamps, 16-wide) vs ref.
/// `quantized_six` is forwarded from `quantized_activation_max` (the S3Backend
/// adapter's contract).
fn check_relu6_simd_matches_ref() {
    let input = Aligned(make_pattern::<256>(0x6E51_0FE1));
    let mut want = Aligned([0i8; 256]);
    let mut got = Aligned([0i8; 256]);
    let params = activation_params(-1, 2, 24); // quantized six @ scale 0.25

    hematite_ref::activation::relu6(&input.0, &params, &mut want.0, &mut [], 24)
        .expect("harness: ref relu6 shape");
    hematite_s3::activations::relu6(&input.0, &params, &mut got.0, &mut [], 24)
        .expect("harness: s3 relu6 shape");

    report(&compare("relu6_256", &got.0, &want.0));
}

/// hard_swish — the DOWNGRADED integer rational formula (goldens-pinned, NOT
/// the TFLM fixed-point chain). The s3 scalar and the lane model reproduce
/// the same downgraded math; hematite-ref implements the same downgraded
/// formula (same-backend semantics — the known TFLM divergence is pinned by
/// the goldens, not "fixed" here).
fn check_hard_swish_simd_matches_ref() {
    let input = Aligned(make_pattern::<256>(0xD0D0_0D0D));
    let mut want = Aligned([0i8; 256]);
    let mut got = Aligned([0i8; 256]);
    let params = activation_params(-3, 1, 127);

    hematite_ref::activation::hard_swish(&input.0, &params, &mut want.0, &mut [])
        .expect("harness: ref hard_swish shape");
    hematite_s3::activations::hard_swish(&input.0, &params, &mut got.0, &mut [])
        .expect("harness: s3 hard_swish shape");

    report(&compare("hard_swish_256", &got.0, &want.0));
}

// ── Softmax: 1x1000 (spec.rs SOFTMAX_1X1000_PARAMS) ─────────────────────────

fn check_softmax_simd_matches_ref() {
    // Buffers carved from the SRAM bench arena (unused during validation) —
    // NOT stack locals: with all 18 checks inlined, validate_all's frame
    // pushes SP into the arena's top bytes and a stack-local buffer whose
    // slot straddles the arena end writes pattern bytes past it into the bss
    // region where the defmt RTT_ENCODER.taken flag lives (device finding,
    // task 18 — the same fix class as the conv3x3 carve in task 8).
    let mut off = 0;
    let input = arena_carve_i8(&mut off, 1000);
    let want = arena_carve_i8(&mut off, 1000);
    let got = arena_carve_i8(&mut off, 1000);
    let scratch = arena_carve(&mut off, 4000);
    input.copy_from_slice(&make_pattern::<1000>(0x50F7_4A11));
    want.fill(0);
    got.fill(0);
    scratch.fill(0);
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

    hematite_ref::softmax::softmax(input, &params, want, scratch)
        .expect("harness: ref softmax shape");
    hematite_s3::softmax::softmax(input, &params, got, scratch)
        .expect("harness: s3 softmax shape");

    report(&compare("softmax_1x1000", got, want));
}

// ── Depthwise: 12x12x16 stride-2 SAME (spec.rs SIMD_DEPTHWISE_S2_SAME_PARAMS) ─

const DEPTHWISE_IN: usize = 12 * 12 * 16;
const DEPTHWISE_W: usize = 1 * 3 * 3 * 16;
const DEPTHWISE_OUT: usize = 6 * 6 * 16;

fn check_depthwise_simd_matches_ref() {
    // Buffers carved from the SRAM bench arena — NOT stack locals (see
    // `arena_carve` for the task-18 OOB device finding).
    let mut off = 0;
    let input = arena_carve_i8(&mut off, DEPTHWISE_IN);
    let weights = arena_carve_i8(&mut off, DEPTHWISE_W);
    let want = arena_carve_i8(&mut off, DEPTHWISE_OUT);
    let got = arena_carve_i8(&mut off, DEPTHWISE_OUT);
    let scratch = arena_carve(&mut off, 4096);
    input.copy_from_slice(&make_pattern::<DEPTHWISE_IN>(0xD3A7_51E5));
    weights.copy_from_slice(&make_pattern::<DEPTHWISE_W>(0xBEE5_7EED));
    let bias = bias_pattern::<16>();
    want.fill(0);
    got.fill(0);
    scratch.fill(0);
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
        input,
        weights,
        &bias,
        &params,
        want,
        scratch,
    )
    .expect("harness: ref depthwise shape");
    hematite_s3::depthwise::depthwise_conv2d(
        input,
        weights,
        &bias,
        &params,
        got,
        scratch,
    )
    .expect("harness: s3 depthwise shape");

    report(&compare("depthwise_12x12x16_s2", got, want));
}

// ── Depthwise dm=8 (T3.5): 12x12x8 -> 12x12x64, 3x3 SAME ────────────────────

const DEPTHWISE_DM8_IN: usize = 12 * 12 * 8;
const DEPTHWISE_DM8_W: usize = 1 * 3 * 3 * 64;
const DEPTHWISE_DM8_OUT: usize = 12 * 12 * 64;

fn check_depthwise_dm8_simd_matches_ref() {
    // Buffers carved from the SRAM bench arena — NOT stack locals (see
    // `arena_carve` for the task-18 OOB device finding).
    let mut off = 0;
    let input = arena_carve_i8(&mut off, DEPTHWISE_DM8_IN);
    let weights = arena_carve_i8(&mut off, DEPTHWISE_DM8_W);
    let want = arena_carve_i8(&mut off, DEPTHWISE_DM8_OUT);
    let got = arena_carve_i8(&mut off, DEPTHWISE_DM8_OUT);
    // dm=8 stages a replicated 14×14×64 padded input (12,544 B) + 256 B accs.
    let scratch = arena_carve(&mut off, 16384);
    input.copy_from_slice(&make_pattern::<DEPTHWISE_DM8_IN>(0xDD8A_0F00));
    weights.copy_from_slice(&make_pattern::<DEPTHWISE_DM8_W>(0x0F8A_00FF));
    let bias = bias_pattern::<64>();
    want.fill(0);
    got.fill(0);
    scratch.fill(0);
    let params = DepthwiseConv2DParams {
        input_shape: [1, 12, 12, 8],
        filter_shape: [1, 3, 3, 64],
        output_shape: [1, 12, 12, 64],
        padding: Padding::Same,
        stride_width: 1,
        stride_height: 1,
        dilation_width_factor: 1,
        dilation_height_factor: 1,
        depth_multiplier: 8,
        input_offset: 0,
        weights_offset: 0,
        output_offset: 0,
        output_multiplier_per_channel: &MULT_64,
        output_shift_per_channel: &SHIFT_64,
        quantized_activation_min: -128,
        quantized_activation_max: 127,
    };

    hematite_ref::depthwise_conv::depthwise_conv2d(
        input,
        weights,
        &bias,
        &params,
        want,
        scratch,
    )
    .expect("harness: ref depthwise dm8 shape");
    hematite_s3::depthwise::depthwise_conv2d(
        input,
        weights,
        &bias,
        &params,
        got,
        scratch,
    )
    .expect("harness: s3 depthwise dm8 shape");

    report(&compare("depthwise_12x12x8_dm8", got, want));
}

// ── Depthwise kws real 10×8 (T3.5b): 49x40x1 -> 25x20x8, stride 2 SAME ─────

const DEPTHWISE_KWS_IN: usize = 49 * 40 * 1;
const DEPTHWISE_KWS_W: usize = 1 * 10 * 8 * 8;
const DEPTHWISE_KWS_OUT: usize = 25 * 20 * 8;

fn check_depthwise_kws10x8_simd_matches_ref() {
    // Buffers carved from the SRAM bench arena — NOT stack locals (see
    // `arena_carve` for the task-18 OOB device finding).
    let mut off = 0;
    // 1960 is not a multiple of 16 — carve 1968 and slice so the arena base
    // stays 16-aligned for the subsequent carves.
    let input = &mut arena_carve_i8(&mut off, (DEPTHWISE_KWS_IN + 15) & !15)[..DEPTHWISE_KWS_IN];
    let weights = arena_carve_i8(&mut off, DEPTHWISE_KWS_W);
    let want = arena_carve_i8(&mut off, DEPTHWISE_KWS_OUT);
    let got = arena_carve_i8(&mut off, DEPTHWISE_KWS_OUT);
    // dm=8 staged path: padded 58x46x16 input (42,688 B) + staged filter
    // 80x16 (1,280 B) + accs 16x4 + anytap partials 16x4 → 44,096 B.
    let scratch = arena_carve(&mut off, 65536);
    fill_pattern_slice(0x4B57_51E5, input);
    fill_pattern_slice(0x0F8A_5B00, weights);
    let bias = bias_pattern::<8>();
    want.fill(0);
    got.fill(0);
    scratch.fill(0);
    let params = DepthwiseConv2DParams {
        input_shape: [1, 49, 40, 1],
        filter_shape: [1, 10, 8, 8],
        output_shape: [1, 25, 20, 8],
        padding: Padding::Same,
        stride_width: 2,
        stride_height: 2,
        dilation_width_factor: 1,
        dilation_height_factor: 1,
        depth_multiplier: 8,
        input_offset: 128,
        weights_offset: 0,
        output_offset: -128,
        output_multiplier_per_channel: &MULT_8,
        output_shift_per_channel: &SHIFT_8,
        quantized_activation_min: 0,
        quantized_activation_max: 127,
    };

    hematite_ref::depthwise_conv::depthwise_conv2d(
        input,
        weights,
        &bias,
        &params,
        want,
        scratch,
    )
    .expect("harness: ref kws depthwise shape");
    hematite_s3::depthwise::depthwise_conv2d(
        input,
        weights,
        &bias,
        &params,
        got,
        scratch,
    )
    .expect("harness: s3 kws depthwise shape");

    report(&compare("depthwise_kws_49x40x1_10x8_dm8", got, want));
}

// ── Conv 1x1: 64x1x1x64 (spec.rs EMBER_CONV_1X1_64_PARAMS) ─────────────────

const CONV1X1_IN: usize = 1 * 1 * 64;
const CONV1X1_W: usize = 64 * 1 * 1 * 64;
const CONV1X1_OUT: usize = 1 * 1 * 64;

fn check_conv1x1_simd_matches_ref() {
    // Buffers carved from the SRAM bench arena — NOT stack locals (see
    // `arena_carve` for the task-18 OOB device finding).
    let mut off = 0;
    let input = arena_carve_i8(&mut off, CONV1X1_IN);
    let weights = arena_carve_i8(&mut off, CONV1X1_W);
    let want = arena_carve_i8(&mut off, CONV1X1_OUT);
    let got = arena_carve_i8(&mut off, CONV1X1_OUT);
    let scratch = arena_carve(&mut off, 512);
    input.copy_from_slice(&make_pattern::<CONV1X1_IN>(0xC0FF_EE11));
    weights.copy_from_slice(&make_pattern::<CONV1X1_W>(0x1CE5_51CE));
    let bias = bias_pattern::<64>();
    want.fill(0);
    got.fill(0);
    scratch.fill(0);
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

    hematite_ref::conv::conv2d(input, weights, &bias, &params, want, scratch)
        .expect("harness: ref conv1x1 shape");
    hematite_s3::conv1x1::conv2d_1x1(input, weights, &bias, &params, got, scratch)
        .expect("harness: s3 conv1x1 shape");

    report(&compare("conv1x1_64x1x1x64", got, want));
}

// ── Conv 1x1 channel-padded (T3.3): 3ch → 16ch, pad-in-scratch ──────────────
//
// input_c 3 (< 16, non-%16) gates out of the strict `% 16` gate and now
// dispatches SIMD via zero-pad-in-scratch on device. The engagement flag
// (`conv1x1_padded_took_simd`) asserts the padded SIMD path ACTUALLY ran —
// the eligibility mirror alone cannot see runtime pointer alignment.

const CONV1X1_PAD3_IN: usize = 1 * 1 * 3;
const CONV1X1_PAD3_W: usize = 16 * 1 * 1 * 3;
const CONV1X1_PAD3_OUT: usize = 1 * 1 * 16;

fn check_conv1x1_padded_3ch_matches_ref() {
    let mut off = 0;
    let input = arena_carve_i8(&mut off, (CONV1X1_PAD3_IN + 15) & !15);
    let weights = arena_carve_i8(&mut off, (CONV1X1_PAD3_W + 15) & !15);
    let want = arena_carve_i8(&mut off, (CONV1X1_PAD3_OUT + 15) & !15);
    let got = arena_carve_i8(&mut off, (CONV1X1_PAD3_OUT + 15) & !15);
    // Staged padded input (1 px × 16) + padded weights (16×16) + accs (16×4)
    // + wsum (16×4) = 592 bytes; carve 1024 for head room.
    let scratch = arena_carve(&mut off, 1024);
    let input = &mut input[..CONV1X1_PAD3_IN];
    let weights = &mut weights[..CONV1X1_PAD3_W];
    let want = &mut want[..CONV1X1_PAD3_OUT];
    let got = &mut got[..CONV1X1_PAD3_OUT];
    fill_pattern_slice(0x3C1_A0, input);
    fill_pattern_slice(0x5EED_0FF, weights);
    let bias = bias_pattern::<16>();
    want.fill(0);
    got.fill(0);
    scratch.fill(0);
    let params = Conv2DParams {
        input_shape: [1, 1, 1, 3],
        filter_shape: [16, 1, 1, 3],
        output_shape: [1, 1, 1, 16],
        padding: Padding::Same,
        stride_width: 1,
        stride_height: 1,
        dilation_width_factor: 1,
        dilation_height_factor: 1,
        input_offset: 128,
        weights_offset: 0,
        output_offset: -10,
        output_multiplier_per_channel: &MULT_16,
        output_shift_per_channel: &SHIFT_16,
        quantized_activation_min: -128,
        quantized_activation_max: 127,
    };

    hematite_ref::conv::conv2d(input, weights, &bias, &params, want, scratch)
        .expect("harness: ref conv1x1 padded shape");
    hematite_s3::conv1x1::conv2d_1x1(input, weights, &bias, &params, got, scratch)
        .expect("harness: s3 conv1x1 padded shape");

    let simd = if hematite_s3::conv1x1::conv1x1_padded_took_simd() {
        "SIMD"
    } else {
        "scalar"
    };
    crate::firmware::uart0_log!("simd conv1x1_padded_3ch path: {}", simd);
    report(&compare("conv1x1_padded_3ch", got, want));
}

// ── Conv 3x3: 16x16x32 SAME (spec.rs SIMD_CONV3X3_SAME_PARAMS) ─────────────

const CONV3X3_IN: usize = 16 * 16 * 32;
const CONV3X3_W: usize = 32 * 3 * 3 * 32;
const CONV3X3_OUT: usize = 16 * 16 * 32;

fn check_conv3x3_simd_matches_ref() {
    // 53 KB of buffers are carved from the SRAM bench arena (unused during
    // validation; the kernel benches carve it afterwards — same pattern as
    // spec.rs `carve_into`). They must NOT be stack locals: with all 18
    // checks inlined, validate_all's frame was 0xf270 bytes and the conv3x3
    // check's SP landed below `_stack_end_cpu0`, faulting with an exception
    // (device finding, task 8). They must NOT be statics either: a +53 KB
    // `.bss` hoist shrank the 65 KB stack (stack.x gives the stack whatever
    // remains above bss) to 12 KB and the firmware faulted at boot.
    // SAFETY: single-threaded firmware; each check runs once, sequentially.
    let arena = unsafe { &mut *core::ptr::addr_of_mut!(crate::firmware::SRAM_ARENA) };
    let mut off = 0usize;
    // Base is 16-aligned and every size is a multiple of 16 → all carves are
    // 16-aligned (the SIMD dispatch gate + VLD.128 requirement).
    macro_rules! take_i8 {
        ($len:expr) => {{
            let p = unsafe { arena.0.as_mut_ptr().add(off) }.cast::<i8>();
            off += $len;
            unsafe { core::slice::from_raw_parts_mut(p, $len) }
        }};
    }
    let input = take_i8!(CONV3X3_IN);
    let weights = take_i8!(CONV3X3_W);
    let want = take_i8!(CONV3X3_OUT);
    let got = take_i8!(CONV3X3_OUT);
    let scratch = &mut arena.0[off..off + 20480];
    input.copy_from_slice(&make_pattern::<CONV3X3_IN>(0x3EE7_5EED));
    weights.copy_from_slice(&make_pattern::<CONV3X3_W>(0xACC5_10A));
    let bias = bias_pattern::<32>();
    want.fill(0);
    got.fill(0);
    scratch.fill(0);
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
        input,
        weights,
        &bias,
        &params,
        want,
        scratch,
    )
    .expect("harness: ref conv3x3 shape");
    hematite_s3::conv3x3::conv2d_3x3(
        input,
        weights,
        &bias,
        &params,
        got,
        scratch,
    )
    .expect("harness: s3 conv3x3 shape");

    report(&compare("conv3x3_16x16x32_same", got, want));
}

// ── FC: 256x64 (spec.rs SIMD_FC_256X64_PARAMS) ─────────────────────────────

const FC_W: usize = 64 * 256;

fn check_fc_simd_matches_ref() {
    // Buffers carved from the SRAM bench arena — NOT stack locals (see
    // `arena_carve` for the task-18 OOB device finding).
    let mut off = 0;
    let input = arena_carve_i8(&mut off, 256);
    let weights = arena_carve_i8(&mut off, FC_W);
    let want = arena_carve_i8(&mut off, 64);
    let got = arena_carve_i8(&mut off, 64);
    let scratch = arena_carve(&mut off, 512);
    input.copy_from_slice(&make_pattern::<256>(0xFAC1_5ED7));
    weights.copy_from_slice(&make_pattern::<FC_W>(0xD07_5C0DE));
    let bias = bias_pattern::<64>();
    want.fill(0);
    got.fill(0);
    scratch.fill(0);
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
        input,
        weights,
        &bias,
        &params,
        want,
        scratch,
    )
    .expect("harness: ref fc shape");
    hematite_s3::gemm::fully_connected(input, weights, &bias, &params, got, scratch)
        .expect("harness: s3 fc shape");

    report(&compare("fc_256x64", got, want));
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

    let simd = if hematite_s3::reductions::mean_took_simd() {
        "SIMD"
    } else {
        "scalar"
    };
    crate::firmware::uart0_log!("simd mean path: {}", simd);
    report(&compare("mean_hw_2x2x4", &got.0, &want.0));
}

// ── Mean (T3.4): mv2 global 7x7x1280 (in_c > 256) ───────────────────────────
//
// The shape that actually bites the landed gate: positions 49 (single
// position pass) but in_c 1280 > 256 — the looped-accumulation dispatch
// folds staged channel passes into the in_c*4-byte accs buffer and
// requantizes once. The full mv2 model stays PSRAM-gated (board reports
// PSRAM: 0 bytes); this is the kernel row standalone (T5.2 consumes).

const MEAN_MV2_IN: usize = 7 * 7 * 1280;
const MEAN_MV2_OUT: usize = 1280;

fn check_mean_mv2_global_simd_matches_ref() {
    // 64 KB of buffers carved from the SRAM bench arena — NOT stack locals
    // (see `arena_carve` for the task-18 OOB device finding). The 62 KB
    // input is filled at runtime (`fill_pattern_slice`) — a `make_pattern`
    // const would materialize a 62 KB stack temporary.
    let mut off = 0;
    let input = arena_carve_i8(&mut off, MEAN_MV2_IN);
    let want = arena_carve_i8(&mut off, MEAN_MV2_OUT);
    let got = arena_carve_i8(&mut off, MEAN_MV2_OUT);
    fill_pattern_slice(0xBEAD_1EAF, input);
    want.fill(0);
    got.fill(0);
    let params = ReduceParams {
        keep_dims: false,
        axis: [1, 2, 0, 0],
        axis_count: 2,
        input_shape: [1, 7, 7, 1280],
        output_shape: [1, 1, 1, 1280],
        output_type: 0,
        input_offset: 0,
        output_offset: -128,
        output_multiplier: 1 << 30,
        output_shift: 1,
        quantized_activation_min: -128,
        quantized_activation_max: 127,
    };

    hematite_ref::reductions::mean(input, &params, want).expect("harness: ref mv2 global mean");
    hematite_s3::reductions::mean(input, &params, got).expect("harness: s3 mv2 global mean");

    let simd = if hematite_s3::reductions::mean_took_simd() {
        "SIMD"
    } else {
        "scalar"
    };
    crate::firmware::uart0_log!("simd mean 7x7x1280 path: {}", simd);
    report(&compare("mean_mv2_7x7x1280", got, want));
}

// ── FC small / non-16 input dims (T3.6 pad-in-scratch widening) ─────────────
//
// The zoo FC-bound shapes: sine 1→1, hello_world 1→16 / 16→1,
// anomaly_detect 8→128 — input_dims in {1,3,8,15} gate out of the old
// `% 16` gate and now dispatch SIMD via zero-pad-in-scratch; 17 and 32 cover
// the no-pad boundary and the pad-to-32 case. The host bit-exact matrix in
// gemm.rs (`fc_small_simd_model_matches_ref_bit_exact`) mirrors this.

/// Fill `out` with the LCG pattern (same generator as `make_pattern`).
fn fill_pattern_slice(seed: u32, out: &mut [i8]) {
    let mut x = seed;
    for v in out.iter_mut() {
        x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        *v = (x >> 16) as i8;
    }
}

// ── Fused conv2d (T2.2): conv + residual-ADD + activation in one pass ───────

const FUSED_CONV1X1_IN: usize = 4 * 4 * 16;
const FUSED_CONV1X1_W: usize = 16 * 1 * 1 * 16;
const FUSED_CONV1X1_OUT: usize = 4 * 4 * 16;

fn check_fused_conv2d_simd_matches_ref() {
    // Buffers carved from the SRAM bench arena — NOT stack locals (see
    // `arena_carve` for the task-18 OOB device finding).
    let mut off = 0;
    let input = arena_carve_i8(&mut off, FUSED_CONV1X1_IN);
    let weights = arena_carve_i8(&mut off, FUSED_CONV1X1_W);
    let want = arena_carve_i8(&mut off, FUSED_CONV1X1_OUT);
    let got = arena_carve_i8(&mut off, FUSED_CONV1X1_OUT);
    let residual = arena_carve_i8(&mut off, FUSED_CONV1X1_OUT);
    let scratch = arena_carve(&mut off, 2048);
    fill_pattern_slice(0xC0FF_EE11, input);
    fill_pattern_slice(0x1CE5_51CE, weights);
    fill_pattern_slice(0xABAD_1DEA, residual);
    let bias = bias_pattern::<16>();
    want.fill(0);
    got.fill(0);
    scratch.fill(0);
    let params = FusedConvParams {
        conv: Conv2DParams {
            input_shape: [1, 4, 4, 16],
            filter_shape: [16, 1, 1, 16],
            output_shape: [1, 4, 4, 16],
            padding: Padding::Same,
            stride_width: 1,
            stride_height: 1,
            dilation_width_factor: 1,
            dilation_height_factor: 1,
            input_offset: 0,
            weights_offset: 0,
            output_offset: 5,
            output_multiplier_per_channel: &MULT_16,
            output_shift_per_channel: &SHIFT_16,
            quantized_activation_min: -128,
            quantized_activation_max: 127,
        },
        output_scale: 0.5,
        output_zero_point: 5,
        output_multiplier_per_channel: &MULT_16,
        output_shift_per_channel: &SHIFT_16,
        residual: Some(ResidualAddParams {
            residual_data: residual,
            residual_scale: 0.3,
            residual_zero_point: -3,
            output_scale: 0.4,
            output_zero_point: 1,
            input1_multiplier: 1 << 30,
            input1_shift: 0,
            input2_multiplier: 1_288_490_189,
            input2_shift: -1,
            left_shift: 20,
            output_multiplier: 1_342_177_280,
            output_shift: -18,
        }),
        activation: ActivationEpilogueParams {
            kind: ComposedActivation::Relu,
            input_offset: -1,
            output_offset: 2,
            output_multiplier: 1_342_177_280,
            output_shift: 1,
            quantized_activation_min: 0,
            quantized_activation_max: 127,
        },
    };

    let mut ref_backend = hematite_ref::RefBackend;
    ref_backend
        .fused_conv2d(input, weights, &bias, &params, want, scratch)
        .expect("harness: ref fused_conv2d shape");
    let mut s3_backend = hematite_s3::backend::S3Backend;
    s3_backend
        .fused_conv2d(input, weights, &bias, &params, got, scratch)
        .expect("harness: s3 fused_conv2d shape");

    report(&compare("fused_conv1x1_residual_relu", got, want));
}

// ── Fused elementwise chain (T2.3): add + relu, register-held SIMD ───────────

fn check_fused_chain_simd_matches_ref() {
    // Buffers carved from the SRAM bench arena — NOT stack locals (see
    // `arena_carve` for the task-18 OOB device finding). The chain is
    // IDENTITY-eligible (zero offsets, identity quant pairs) so the fused
    // chain SIMD path ENGAGES on device; the operand is 16-aligned so the
    // operand-chunk loads satisfy the alignment gate.
    let mut off = 0;
    let input = arena_carve_i8(&mut off, 256);
    let operand = arena_carve_i8(&mut off, 256);
    let want = arena_carve_i8(&mut off, 256);
    let got = arena_carve_i8(&mut off, 256);
    fill_pattern_slice(0xC0FF_EE11, input);
    fill_pattern_slice(0xABAD_1DEA, operand);
    want.fill(0);
    got.fill(0);

    // Step 0: identity ADD (src + operand, saturating i8) — the
    // `simd_eligible_add_sub` gate's exact contract. Step 1: identity ReLU —
    // the `relu_simd_eligible_params` gate's exact contract. Every step is
    // SIMD-eligible, so the register-held chain path engages on device.
    let steps: [ElementwiseChainStep<'_>; 2] = [
        ElementwiseChainStep {
            kind: ElementwiseKind::Add,
            operand: Some(operand),
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
        },
        ElementwiseChainStep {
            kind: ElementwiseKind::Relu,
            operand: None,
            input1_offset: 0,
            input2_offset: 0,
            output_offset: 0,
            output_multiplier: 1 << 30,
            output_shift: 1,
            left_shift: 0,
            input1_multiplier: 0,
            input1_shift: 0,
            input2_multiplier: 0,
            input2_shift: 0,
            quantized_activation_min: 0,
            quantized_activation_max: 127,
        },
    ];
    let params = ElementwiseChainParams {
        num_elements: 256,
        steps: &steps,
    };

    let mut ref_backend = hematite_ref::RefBackend;
    ref_backend
        .fused_elementwise_chain(input, &params, want)
        .expect("harness: ref fused chain shape");
    let mut s3_backend = hematite_s3::backend::S3Backend;
    s3_backend
        .fused_elementwise_chain(input, &params, got)
        .expect("harness: s3 fused chain shape");

    report(&compare("fused_chain_add_relu_256", got, want));
}

// ── Fused pool-with-fold (T2.4): avg-pool + MUL fold + relu epilogue ─────────

fn check_fused_pool_fold_simd_matches_ref() {
    // Buffers carved from the SRAM bench arena — NOT stack locals (same as
    // the other fused checks). The fold is IDENTITY-eligible (zero offsets,
    // `(1<<30, 0)` output pair — `fused::fold_simd_exact`) and the pool is
    // SIMD-eligible (2×2/stride-2, channels % 16, full-range clamp), so the
    // composed SIMD path ENGAGES on device: the fold SIMD-engages via the
    // elementwise gates into the staged scratch and the pool SIMD reads it
    // directly (16-aligned arena carve).
    const IN: usize = 4 * 4 * 16;
    const OUT: usize = 2 * 2 * 16;
    let mut off = 0;
    let input = arena_carve_i8(&mut off, IN);
    let operand = arena_carve_i8(&mut off, IN);
    let want = arena_carve_i8(&mut off, OUT);
    let got = arena_carve_i8(&mut off, OUT);
    let scratch = arena_carve(&mut off, 256);
    fill_pattern_slice(0xC0FF_EE11, input);
    fill_pattern_slice(0xABAD_1DEA, operand);
    want.fill(0);
    got.fill(0);
    scratch.fill(0);

    let params = FoldedPoolParams {
        pool: PoolParams {
            input_shape: [1, 4, 4, 16],
            output_shape: [1, 2, 2, 16],
            filter_width: 2,
            filter_height: 2,
            stride_width: 2,
            stride_height: 2,
            padding: Padding::Same,
            activation: FusedActivation::None,
            quantized_activation_min: i8::MIN as i32,
            quantized_activation_max: i8::MAX as i32,
        },
        pool_kind: PoolKind::Average,
        fold: Some(PoolInputFold {
            builtin: 18,
            operand_data: operand,
            operand_zero_point: 0,
            input_zero_point: 0,
            output_zero_point: 0,
            folded_scale: 1.0,
            left_shift: 0,
            output_multiplier: 1 << 30,
            output_shift: 0,
            input1_multiplier: 0,
            input1_shift: 0,
            input2_multiplier: 0,
            input2_shift: 0,
            num_elements: 256,
        }),
        activation: ActivationEpilogueParams {
            kind: ComposedActivation::Relu,
            input_offset: -5,
            output_offset: 2,
            output_multiplier: 1_342_177_280,
            output_shift: 1,
            quantized_activation_min: 0,
            quantized_activation_max: 127,
        },
    };

    let mut ref_backend = hematite_ref::RefBackend;
    ref_backend
        .fused_pool_with_fold(input, &params, want, scratch)
        .expect("harness: ref fused pool-fold shape");
    let mut s3_backend = hematite_s3::backend::S3Backend;
    s3_backend
        .fused_pool_with_fold(input, &params, got, scratch)
        .expect("harness: s3 fused pool-fold shape");

    report(&compare("fused_pool_avg_mulfold_relu", got, want));
}

fn check_fc_small_simd_matches_ref() {
    let shapes: [((usize, usize, i32), &'static str); 7] = [
        ((1, 1, 0), "fc_small_1x1"),      // sine-family
        ((1, 16, 128), "fc_small_1x16"),  // hello_world FC1 (gated out before T3.6)
        ((3, 16, 128), "fc_small_3x16"),
        ((8, 128, 128), "fc_small_8x128"), // anomaly_detect FC6 (gated out before T3.6)
        ((15, 8, 128), "fc_small_15x8"),
        ((17, 8, 0), "fc_small_17x8"),    // pad to 32
        ((32, 8, 0), "fc_small_32x8"),    // no-pad path
    ];
    // Max padded scratch: input_dim 8 → pad16 16, out 128:
    // 16 + 128*16 + 128*4 + 128*4 = 3088 (a multiple of 16).
    const SCRATCH: usize = 16 + 128 * 16 + 128 * 4 + 128 * 4;
    let mut off = 0;
    for &((input_dim, output_dim, input_offset), name) in &shapes {
        // Carve at 16-aligned offsets (round up each region); the kernels
        // receive exact-length sub-slices that start at the aligned base.
        let in_len = (input_dim + 15) & !15;
        let w_len = ((input_dim * output_dim) + 15) & !15;
        let out_len = (output_dim + 15) & !15;
        let mut input = &mut arena_carve_i8(&mut off, in_len)[..input_dim];
        let mut weights = &mut arena_carve_i8(&mut off, w_len)[..input_dim * output_dim];
        let want = &mut arena_carve_i8(&mut off, out_len)[..output_dim];
        let got = &mut arena_carve_i8(&mut off, out_len)[..output_dim];
        let scratch = arena_carve(&mut off, SCRATCH);
        fill_pattern_slice(0xA11_5ED, input);
        fill_pattern_slice(0xD07_5C0DE, weights);
        want.fill(0);
        got.fill(0);
        scratch.fill(0);
        let mut bias = [0i32; 128];
        for (i, b) in bias.iter_mut().enumerate() {
            *b = (i as i32) * 37 - 500;
        }
        let params = FullyConnectedParams {
            input_dim: input_dim as i32,
            output_dim: output_dim as i32,
            input_offset,
            weights_offset: 0,
            output_offset: 0,
            output_multiplier_per_channel: &MULT_128[..output_dim],
            output_shift_per_channel: &SHIFT_128[..output_dim],
            quantized_activation_min: -128,
            quantized_activation_max: 127,
        };

        hematite_ref::fully_connected::fully_connected(
            input,
            weights,
            &bias[..output_dim],
            &params,
            want,
            scratch,
        )
        .expect("harness: ref small fc shape");
        hematite_s3::gemm::fully_connected(input, weights, &bias[..output_dim], &params, got, scratch)
            .expect("harness: s3 small fc shape");

        report(&compare(name, got, want));
    }
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
    crate::firmware::uart0_log!(
        "=== SIMD CORRECTNESS (elementwise + pool + relu/softmax/depthwise/conv/fc/mean/fused, vs hematite-ref) ==="
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
    // T3.2 — widened per-lane model with non-identity offsets/multipliers.
    run_elementwise_non_identity_check::<256>(ElemOp::Add, "add_n256_nonidentity");
    run_elementwise_non_identity_check::<256>(ElemOp::Mul, "mul_n256_nonidentity");
    run_elementwise_non_identity_check::<256>(ElemOp::Sub, "sub_n256_nonidentity");
    run_avg_pool_check();
    run_max_pool_check();
    run_generic_avg_pool_check();
    run_generic_max_pool_checks();
    check_relu_simd_matches_ref();
    // T3.2 — the widened relu6 / hard_swish activation lane models.
    check_relu6_simd_matches_ref();
    check_hard_swish_simd_matches_ref();
    check_softmax_simd_matches_ref();
    check_depthwise_simd_matches_ref();
    check_depthwise_dm8_simd_matches_ref();
    check_depthwise_kws10x8_simd_matches_ref();
    check_conv1x1_simd_matches_ref();
    check_conv1x1_padded_3ch_matches_ref();
    check_conv3x3_simd_matches_ref();
    check_fc_simd_matches_ref();
    check_fc_small_simd_matches_ref();
    check_mean_simd_matches_ref();
    check_mean_mv2_global_simd_matches_ref();
    check_fused_conv2d_simd_matches_ref();
    check_fused_chain_simd_matches_ref();
    check_fused_pool_fold_simd_matches_ref();
    crate::firmware::uart0_log!("=== SIMD CORRECTNESS DONE ===");
}
