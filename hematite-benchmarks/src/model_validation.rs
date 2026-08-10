// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Model validation against executed-TFLite goldens (plan T5.2 validation).
//!
//! Compiles each runnable zoo `.tflite` model via `#[model]` (the T4.1
//! `Model<B>` device-side bridge) and runs it on the device (or under QEMU)
//! with the deterministic golden [`crate`] inputs, comparing the output
//! element-for-element against [`EXPECTED_OUTPUT`] and printing
//! `model <name>: PASS (fnv=0x....)` or `FAIL at idx k: got=G want=W`.
//!
//! This is a **validation** section, not a benchmark: it runs before the
//! kernel rows so every PASS/FAIL line prints even if a later row panics
//! (the MobileNetV2 PSRAM row's "arena too small" panic stays last).
//!
//! # Per-model module isolation
//!
//! The `#[model]` macro emits `Model<B>` plus `INPUT_LEN`, `OUTPUT_LEN`,
//! `SCRATCH_LEN` and model weight consts at the module scope; the golden
//! `include!` emits `MODEL_PATH`, `INPUT_SHAPE`, `OUTPUT_SHAPE`,
//! `INPUT_DATA`, `EXPECTED_OUTPUT`. Both would collide at file scope, so
//! each model gets its own module with a nested `golden` submodule.
//!
//! # Honest limits (QEMU: 512 KB SRAM, no PSRAM, ~70 KB device stack)
//!
//! * sine / hello_world / kws — small, fit SRAM + stack; the proven
//!   bit-exact models, asserted element-for-element.
//! * anomaly_detect — hematite output (fnv 0xf2a76cd6) differs from the
//!   T10-regenerated executed-TFLM golden on 210/640 elements by exactly
//!   ±1 (documented single-vs-double rounding, DEFERRED_MODELS.md §6) —
//!   reported as FAIL, never masked.
//! * person_detect — predict allocas 232 KB of intermediates; the ~65 KB
//!   device stack region cannot hold it and the arena-stack SP switch
//!   faults on real silicon (window-underflow; QEMU is lenient) — SKIP
//!   reason=stack on device, run on the dedicated 256 KB arena stack
//!   ([`crate::firmware::run_on_arena_stack`]) under QEMU (task-5
//!   evidence: task-5-device-s3-models.log).
//! * mobilenet_v2 (224×224×3, PSRAM-tier) — needs PSRAM; this board has
//!   none. Reported as an honest SKIP (cannot fit SRAM), never faked.
//!
//! All output goes through [`crate::firmware::uart0_log`] (UART0 only — no
//! defmt/RTT: the defmt logger is not reentrant across exceptions and its
//! reentrancy panic masked the simd_validation root cause in task 5; RTT is
//! unreadable on this board anyway).

use hematite_codegen::model;
use hematite_ref::RefBackend;
use hematite_s3::backend::S3Backend;

/// FNV-1a 32-bit checksum (mirrors the C baseline's `out_checksum`):
/// seed 2166136261, prime 16777619, over the raw output bytes.
fn fnv1a(data: &[u8]) -> u32 {
    let mut h: u32 = 2_166_136_261;
    for &b in data {
        h ^= u32::from(b);
        h = h.wrapping_mul(16_777_619);
    }
    h
}

/// One validation result.
struct ModelResult {
    name: &'static str,
    pass: bool,
    mismatch: Option<(usize, i8, i8)>,
    fnv: u32,
}

/// Compare `got` against `expected` element-for-element.
fn compare(name: &'static str, got: &[i8], expected: &[i8]) -> ModelResult {
    let fnv = fnv1a(bytemuck_cast(got));
    if got.len() != expected.len() {
        return ModelResult {
            name,
            pass: false,
            mismatch: Some((0, got[0], expected[0])),
            fnv,
        };
    }
    for (i, (&g, &w)) in got.iter().zip(expected.iter()).enumerate() {
        if g != w {
            return ModelResult {
                name,
                pass: false,
                mismatch: Some((i, g, w)),
                fnv,
            };
        }
    }
    ModelResult {
        name,
        pass: true,
        mismatch: None,
        fnv,
    }
}

/// Reinterpret `&[i8]` as `&[u8]` for FNV-1a (same bits).
fn bytemuck_cast(v: &[i8]) -> &[u8] {
    // SAFETY: i8 and u8 have identical layout; this is a plain re-view.
    unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) }
}

/// Print one model result.
fn report(r: &ModelResult) {
    match (r.pass, r.mismatch) {
        (true, _) => {
            crate::firmware::uart0_log!("model {}: PASS (fnv=0x{:08x})", r.name, r.fnv);
        }
        (false, Some((i, g, w))) => {
            crate::firmware::uart0_log!(
                "model {}: FAIL at idx {}: got={} want={} (fnv=0x{:08x})",
                r.name,
                i,
                g,
                w,
                r.fnv,
            );
        }
        (false, None) => {
            crate::firmware::uart0_log!("model {}: FAIL (fnv=0x{:08x})", r.name, r.fnv);
        }
    }
}

// ── Model 1: sine regression ────────────────────────────────────────────────

mod model_sine {
    use super::*;
    pub mod golden {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../hematite-tests/goldens/models/sine.rs"
        ));
    }
    #[model("../models/sine.tflite")]
    pub struct SineModel;
}

/// Carve `len` bytes of scratch from the 256 KB SRAM bench arena.
///
/// Validation runs before the kernel benches carve the arena (boot order), so
/// the arena is free here. Generated `predict()` would size its own stack
/// scratch from the model's real `SCRATCH_LEN` (up to ~34 KB for kws), which
/// overflows the ~65 KB device stack once validation frames are counted — the
/// validation path therefore goes through `predict_with_scratch` with an
/// arena-backed buffer.
fn carve_scratch(len: usize) -> &'static mut [u8] {
    let base = unsafe { core::ptr::addr_of_mut!(crate::firmware::SRAM_ARENA) as *mut u8 };
    let n = len.min(256 * 1024);
    unsafe { core::slice::from_raw_parts_mut(base, n) }
}

fn validate_sine() {
    let m = model_sine::Model::<RefBackend>::new(RefBackend);
    let mut out = [0i8; model_sine::OUTPUT_LEN];
    let scratch = carve_scratch(model_sine::SCRATCH_LEN);
    let _ = m.predict_with_scratch(&model_sine::golden::INPUT_DATA, &mut out, scratch);
    report(&compare("sine", &out, &model_sine::golden::EXPECTED_OUTPUT));
}

// ── Model 2: hello_world (sine_regression zoo) ──────────────────────────────

mod model_hello_world {
    use super::*;
    pub mod golden {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../hematite-tests/goldens/models/hello_world_int8.rs"
        ));
    }
    #[model("../models/zoo/sine_regression/hello_world_int8.tflite")]
    pub struct HelloWorldModel;
}

fn validate_hello_world() {
    let m = model_hello_world::Model::<RefBackend>::new(RefBackend);
    let mut out = [0i8; model_hello_world::OUTPUT_LEN];
    let scratch = carve_scratch(model_hello_world::SCRATCH_LEN);
    let _ = m.predict_with_scratch(&model_hello_world::golden::INPUT_DATA, &mut out, scratch);
    report(&compare(
        "hello_world_int8",
        &out,
        &model_hello_world::golden::EXPECTED_OUTPUT,
    ));
}

// ── Model 3: keyword spotting ───────────────────────────────────────────────

mod model_kws {
    use super::*;
    pub mod golden {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../hematite-tests/goldens/models/kws_micro_speech_int8.rs"
        ));
    }
    #[model("../models/zoo/keyword_spotting/kws_micro_speech_int8.tflite")]
    pub struct KwsModel;
}

fn validate_kws() {
    let m = model_kws::Model::<RefBackend>::new(RefBackend);
    let mut out = [0i8; model_kws::OUTPUT_LEN];
    let scratch = carve_scratch(model_kws::SCRATCH_LEN);
    let _ = m.predict_with_scratch(&model_kws::golden::INPUT_DATA, &mut out, scratch);
    report(&compare(
        "kws_micro_speech_int8",
        &out,
        &model_kws::golden::EXPECTED_OUTPUT,
    ));
}

// ── Model 4: anomaly detection ──────────────────────────────────────────────

mod model_anomaly {
    use super::*;
    pub mod golden {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../hematite-tests/goldens/models/anomaly_detect_int8.rs"
        ));
    }
    #[model("../models/zoo/anomaly_detect/anomaly_detect_int8.tflite")]
    pub struct AnomalyModel;
}

fn validate_anomaly() {
    let m = model_anomaly::Model::<RefBackend>::new(RefBackend);
    let mut out = [0i8; model_anomaly::OUTPUT_LEN];
    let scratch = carve_scratch(model_anomaly::SCRATCH_LEN);
    let _ = m.predict_with_scratch(&model_anomaly::golden::INPUT_DATA, &mut out, scratch);
    report(&compare(
        "anomaly_detect_int8",
        &out,
        &model_anomaly::golden::EXPECTED_OUTPUT,
    ));
}

// ── Model 5: person detect (VWW) — KNOWN-DIVERGENT / stack-borderline ───────

mod model_person_detect {
    use super::*;
    pub mod golden {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../hematite-tests/goldens/models/person_detect_int8.rs"
        ));
    }
    #[model("../models/zoo/person_detect_vww/person_detect_int8.tflite")]
    pub struct PersonDetectModel;
}

fn validate_person_detect() {
    // The generated predict allocas 232 KB of intermediates (0x38ac0 in the
    // ELF) — the ~65 KB device stack region cannot hold it, and the dedicated
    // arena-stack switch faults on real silicon (window-underflow; QEMU only,
    // see firmware::run_on_arena_stack). Honest SKIP with the reason +
    // rerun condition; never weakened. Under QEMU the predict runs on the
    // dedicated arena stack and is checked against the golden.
    #[cfg(feature = "qemu")]
    {
        let out = crate::firmware::run_on_arena_stack(|| {
            let m = model_person_detect::Model::<RefBackend>::new(RefBackend);
            m.predict(&model_person_detect::golden::INPUT_DATA)
        });
        report(&compare(
            "person_detect_int8",
            &out,
            &model_person_detect::golden::EXPECTED_OUTPUT,
        ));
    }
    #[cfg(not(feature = "qemu"))]
    {
        crate::firmware::uart0_log!(
            "model person_detect_int8: SKIP reason=stack rerun_condition=codegen-intermediates-off-stack"
        );
    }
}

// ── Model 6: MobileNetV2 — PSRAM tier, honest SKIP under QEMU ───────────────

mod model_mobilenet {
    use super::*;
    pub mod golden {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../hematite-tests/goldens/models/mobilenet_v2_1.0_224_int8.rs"
        ));
    }
    #[model("../models/zoo/mobilenetv2_cls/mobilenet_v2_1.0_224_int8.tflite")]
    pub struct MobilenetModel;
}

fn validate_mobilenet() {
    // MobileNetV2 224×224 needs ~4 MB PSRAM (input + all intermediates as
    // stack locals); QEMU has no PSRAM and the device SRAM is ~512 KB with a
    // ~70 KB stack. It cannot run here — report an honest SKIP, never fake.
    crate::firmware::uart0_log!(
        "model mobilenet_v2_1.0_224_int8: SKIP (needs PSRAM; not present under QEMU)"
    );
}

/// Run all model validations. Called first from the firmware boot flow.
pub fn validate_all() {
    crate::firmware::uart0_log!("=== MODEL VALIDATION (executed-TFLite goldens) ===");
    validate_sine();
    validate_hello_world();
    validate_kws();
    validate_anomaly();
    validate_person_detect();
    validate_mobilenet();
    crate::firmware::uart0_log!("=== MODEL VALIDATION DONE ===");
}

// ── Model::<S3Backend> validation (plan todo 5, Wave 1) ─────────────────────
//
// Every zoo model re-runs via `Model::<S3Backend>` (same `#[model]` structs)
// and reports PASS only when the output matches BOTH the `RefBackend` output
// (relative wiring gate — on the device the forwarding takes the real
// TIE728/ACCX SIMD paths) and the executed-TFLite golden (absolute gate for
// the bit-exact models). Rows use the Metis F10 record shape so
// `benchmarks/zoo-results/` is a verbatim transcription:
// `PASS <model> <backend> <fnv1a>` / `SKIP <model> reason=<r> rerun_condition=<c>`.
//
// mobilenet_v2 needs ~4 MB PSRAM (no PSRAM on this board — `PSRAM: 0 bytes`,
// task-1 probe): explicit SKIP, never faked. person_detect's intermediates
// are stack locals on the firmware main stack; a stack overflow panics the
// run and must be re-tiered to SKIP reason=stack (recorded), never weakened.

/// First mismatch between `got` and `want` (len mismatch → idx 0).
fn golden_mismatch(got: &[i8], want: &[i8]) -> Option<(usize, i8, i8)> {
    if got.len() != want.len() {
        return Some((0, got[0], want[0]));
    }
    got.iter()
        .zip(want.iter())
        .enumerate()
        .find(|(_, (g, w))| g != w)
        .map(|(i, (g, w))| (i, *g, *w))
}

/// Run one zoo model through S3Backend and report against ref + golden.
fn report_s3(name: &'static str, s3_out: &[i8], ref_out: &[i8], golden: &[i8]) {
    let fnv = fnv1a(bytemuck_cast(s3_out));
    let ref_match = s3_out == ref_out;
    let golden_mismatch = golden_mismatch(s3_out, golden);
    match (ref_match, golden_mismatch) {
        (true, None) => {
            crate::firmware::uart0_log!(
                "model {} [s3]: PASS (fnv=0x{:08x}; matches ref, matches golden)",
                name,
                fnv
            );
        }
        (_, Some((i, g, w))) => {
            crate::firmware::uart0_log!(
                "model {} [s3]: FAIL at idx {}: got={} want={} (fnv=0x{:08x}; ref_match={}, golden_match=false)",
                name,
                i,
                g,
                w,
                fnv,
                ref_match,
            );
        }
        (false, None) => {
            crate::firmware::uart0_log!(
                "model {} [s3]: FAIL (fnv=0x{:08x}; ref_match=false, golden_match=true)",
                name,
                fnv
            );
        }
    }
}

fn validate_s3_sine() {
    let mut s3_out = [0i8; model_sine::OUTPUT_LEN];
    {
        let scratch = carve_scratch(model_sine::SCRATCH_LEN);
        let _ = model_sine::Model::<S3Backend>::new(S3Backend)
            .predict_with_scratch(&model_sine::golden::INPUT_DATA, &mut s3_out, scratch);
    }
    let mut ref_out = [0i8; model_sine::OUTPUT_LEN];
    {
        let scratch = carve_scratch(model_sine::SCRATCH_LEN);
        let _ = model_sine::Model::<RefBackend>::new(RefBackend)
            .predict_with_scratch(&model_sine::golden::INPUT_DATA, &mut ref_out, scratch);
    }
    report_s3(
        "sine",
        &s3_out,
        &ref_out,
        &model_sine::golden::EXPECTED_OUTPUT,
    );
}

fn validate_s3_hello_world() {
    let mut s3_out = [0i8; model_hello_world::OUTPUT_LEN];
    {
        let scratch = carve_scratch(model_hello_world::SCRATCH_LEN);
        let _ = model_hello_world::Model::<S3Backend>::new(S3Backend)
            .predict_with_scratch(&model_hello_world::golden::INPUT_DATA, &mut s3_out, scratch);
    }
    let mut ref_out = [0i8; model_hello_world::OUTPUT_LEN];
    {
        let scratch = carve_scratch(model_hello_world::SCRATCH_LEN);
        let _ = model_hello_world::Model::<RefBackend>::new(RefBackend)
            .predict_with_scratch(&model_hello_world::golden::INPUT_DATA, &mut ref_out, scratch);
    }
    report_s3(
        "hello_world_int8",
        &s3_out,
        &ref_out,
        &model_hello_world::golden::EXPECTED_OUTPUT,
    );
}

fn validate_s3_kws() {
    let mut s3_out = [0i8; model_kws::OUTPUT_LEN];
    {
        let scratch = carve_scratch(model_kws::SCRATCH_LEN);
        let _ = model_kws::Model::<S3Backend>::new(S3Backend)
            .predict_with_scratch(&model_kws::golden::INPUT_DATA, &mut s3_out, scratch);
    }
    let mut ref_out = [0i8; model_kws::OUTPUT_LEN];
    {
        let scratch = carve_scratch(model_kws::SCRATCH_LEN);
        let _ = model_kws::Model::<RefBackend>::new(RefBackend)
            .predict_with_scratch(&model_kws::golden::INPUT_DATA, &mut ref_out, scratch);
    }
    report_s3(
        "kws_micro_speech_int8",
        &s3_out,
        &ref_out,
        &model_kws::golden::EXPECTED_OUTPUT,
    );
}

fn validate_s3_anomaly() {
    let mut s3_out = [0i8; model_anomaly::OUTPUT_LEN];
    {
        let scratch = carve_scratch(model_anomaly::SCRATCH_LEN);
        let _ = model_anomaly::Model::<S3Backend>::new(S3Backend)
            .predict_with_scratch(&model_anomaly::golden::INPUT_DATA, &mut s3_out, scratch);
    }
    let mut ref_out = [0i8; model_anomaly::OUTPUT_LEN];
    {
        let scratch = carve_scratch(model_anomaly::SCRATCH_LEN);
        let _ = model_anomaly::Model::<RefBackend>::new(RefBackend)
            .predict_with_scratch(&model_anomaly::golden::INPUT_DATA, &mut ref_out, scratch);
    }
    report_s3(
        "anomaly_detect_int8",
        &s3_out,
        &ref_out,
        &model_anomaly::golden::EXPECTED_OUTPUT,
    );
}

fn validate_s3_person_detect() {
    // Same 232 KB alloca as the RefBackend predict; same SKIP-on-device /
    // run-on-arena-stack-under-QEMU split as validate_person_detect.
    #[cfg(feature = "qemu")]
    {
        let (s3_out, ref_out) = crate::firmware::run_on_arena_stack(|| {
            let s3 = model_person_detect::Model::<S3Backend>::new(S3Backend)
                .predict(&model_person_detect::golden::INPUT_DATA);
            let refb = model_person_detect::Model::<RefBackend>::new(RefBackend)
                .predict(&model_person_detect::golden::INPUT_DATA);
            (s3, refb)
        });
        report_s3(
            "person_detect_int8",
            &s3_out,
            &ref_out,
            &model_person_detect::golden::EXPECTED_OUTPUT,
        );
    }
    #[cfg(not(feature = "qemu"))]
    {
        crate::firmware::uart0_log!(
            "model person_detect_int8 [s3]: SKIP reason=stack rerun_condition=codegen-intermediates-off-stack"
        );
    }
}

fn validate_s3_mobilenet() {
    // MobileNetV2 224×224 needs ~4 MB PSRAM; this board has none (`PSRAM: 0
    // bytes`). Honest SKIP with the Metis F10 record format — never fake.
    crate::firmware::uart0_log!(
        "model mobilenet_v2_1.0_224_int8 [s3]: SKIP reason=no-psram rerun_condition=board-with-PSRAM"
    );
}

/// Run all zoo models through `Model::<S3Backend>` (plan todo 5). Called
/// from the firmware boot flow alongside [`validate_all`].
pub fn validate_all_s3() {
    crate::firmware::uart0_log!("=== MODEL VALIDATION S3 (Model::<S3Backend> vs ref + golden) ===");
    validate_s3_sine();
    validate_s3_hello_world();
    validate_s3_kws();
    validate_s3_anomaly();
    validate_s3_person_detect();
    validate_s3_mobilenet();
    crate::firmware::uart0_log!("=== MODEL VALIDATION S3 DONE ===");
}
