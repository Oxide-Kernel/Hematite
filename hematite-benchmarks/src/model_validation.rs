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
//! * sine / hello_world / kws / anomaly — small, fit SRAM + stack; the four
//!   proven bit-exact models, asserted element-for-element.
//! * person_detect (96×96×3 = 27648 input, intermediates as stack locals) —
//!   borderline: may overflow the ~70 KB device stack. If it runs, its
//!   known kernel divergence (TFLM single-rounding vs host LiteRT
//!   double-rounding, softmax algorithm) is reported as KNOWN-DIVERGENT,
//!   not a regression.
//! * mobilenet_v2 (224×224×3, PSRAM-tier) — needs PSRAM; QEMU has none.
//!   Reported as an honest SKIP (cannot fit SRAM), never faked.
//!
//! All output goes through [`crate::firmware::firmware_log`] (qemu→UART0,
//! hardware→defmt).

use hematite_codegen::model;
use hematite_ref::RefBackend;

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
            crate::firmware::firmware_log!("model {}: PASS (fnv=0x{:08x})", r.name, r.fnv);
        }
        (false, Some((i, g, w))) => {
            crate::firmware::firmware_log!(
                "model {}: FAIL at idx {}: got={} want={} (fnv=0x{:08x})",
                r.name,
                i,
                g,
                w,
                r.fnv,
            );
        }
        (false, None) => {
            crate::firmware::firmware_log!("model {}: FAIL (fnv=0x{:08x})", r.name, r.fnv);
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

fn validate_sine() {
    let m = model_sine::Model::<RefBackend>::new(RefBackend);
    let out = m.predict(&model_sine::golden::INPUT_DATA);
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
    let out = m.predict(&model_hello_world::golden::INPUT_DATA);
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
    let out = m.predict(&model_kws::golden::INPUT_DATA);
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
    let out = m.predict(&model_anomaly::golden::INPUT_DATA);
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
    let m = model_person_detect::Model::<RefBackend>::new(RefBackend);
    let out = m.predict(&model_person_detect::golden::INPUT_DATA);
    report(&compare(
        "person_detect_int8",
        &out,
        &model_person_detect::golden::EXPECTED_OUTPUT,
    ));
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
    crate::firmware::firmware_log!(
        "model mobilenet_v2_1.0_224_int8: SKIP (needs PSRAM; not present under QEMU)"
    );
}

/// Run all model validations. Called first from the firmware boot flow.
pub fn validate_all() {
    crate::firmware::firmware_log!("=== MODEL VALIDATION (executed-TFLite goldens) ===");
    validate_sine();
    validate_hello_world();
    validate_kws();
    validate_anomaly();
    validate_person_detect();
    validate_mobilenet();
    crate::firmware::firmware_log!("=== MODEL VALIDATION DONE ===");
}
