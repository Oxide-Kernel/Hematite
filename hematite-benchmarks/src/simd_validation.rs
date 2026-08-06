// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! TIE728 SIMD correctness validation for the kernel groups that were NOT
//! previously proven broken under QEMU (elementwise `add`/`mul`/`sub`, pool
//! `avg`/`max`) — companion to [`crate::model_validation`], same reporting
//! style (`PASS`/`FAIL` + fnv1a).
//!
//! `conv1x1`/`conv3x3`/`gemm` are deliberately excluded here: their SIMD
//! dispatch is already gated off under the `qemu` feature (proven broken —
//! `EE.VSMULAS.S8.QACC.LD.INCP` is unemulated by this QEMU fork; see
//! `local-notes/notepads/hematite-nn/problems.md`).
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

use hematite_core::op_params::{ElementwiseParams, FusedActivation, Padding, PoolParams};

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

/// Run all SIMD correctness checks. Called from the firmware boot flow
/// right after [`crate::model_validation::validate_all`] (same rationale:
/// every PASS/FAIL line must print even if a later benchmark row panics).
///
/// See the module doc for why a single QEMU run currently cannot exercise
/// all 11 checks below (the first one crashes/hangs the whole firmware) —
/// this is nonetheless the correct, intended full suite for real hardware
/// or a fixed QEMU build.
pub fn validate_all() {
    crate::firmware::firmware_log!("=== SIMD CORRECTNESS (elementwise + pool, vs hematite-ref) ===");
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
    crate::firmware::firmware_log!("=== SIMD CORRECTNESS DONE ===");
}
