// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Model-level zoo compilation + inference (plan T5.2).
//!
//! Each module compiles a REAL public int8 `.tflite` model via `#[model]`
//! (the T4.1 `Model<B>` device-side invocation bridge) and asserts the
//! generated straight-line op sequence produces bit-exact output against
//! the executed-TFLite golden captured by `tools/generate_goldens`
//! (ai-edge-litert 2.1.6).
//!
//! The `#[model]` attribute must live in its own module per model: the
//! expansion emits a `Model<B>` wrapper plus model-scoped consts
//! (`INPUT_LEN`, `WEIGHTS_0`, ...) that would collide at file scope.
//!
//! Model substitution rationale: the plan's named 18-model list is not
//! obtainable as `.tflite` (esp-dl = proprietary `.espdl` only; the
//! edge-ml-model-zoo repo has no binaries — see
//! `models/zoo/DEFERRED_MODELS.md`). These public int8 models cover the
//! same op families: person detection (VWW), keyword spotting, image
//! classification (MobileNetV2 224²), anomaly detection autoencoder, and
//! sine regression.
//!
//! ## Bit-exactness vs the executed-TFLite golden
//!
//! Four models assert bit-exact output (sine, hello_world, keyword
//! spotting, anomaly detection). Two models compile and execute but are
//! NOT bit-exact (person_detect, mobilenet_v2): their conv chains hit
//! rounding-boundary cases where the hematite kernels (TFLM single-rounding
//! `MultiplyByQuantizedMultiplier` semantics) differ ±1 from the host
//! ai-edge-litert reference kernels (double-rounding), and their softmax
//! outputs diverge where the LiteRT int8 softmax algorithm differs from the
//! TFLM reference (wide-dynamic-range logits). Root cause + fix path in
//! `models/zoo/DEFERRED_MODELS.md` and `local-notes/notepads/hematite-nn/problems.md`.

use hematite_ref::RefBackend;
use hematite_tests::goldens::models;

#[cfg(feature = "hematite-s3")]
use hematite_s3::backend::S3Backend;

/// Assert `actual` matches `expected` element-for-element, naming the
/// index and values of the first mismatch.
fn assert_bit_exact(actual: &[i8], expected: &[i8], name: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{name}: output length {} != expected length {}",
        actual.len(),
        expected.len(),
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            a,
            e,
            "{name}: mismatch at index {i}: model={a}, golden={e}",
        );
    }
}

fn flat_len(shape: &[i32]) -> usize {
    shape.iter().map(|&d| d as usize).product()
}

/// Run `f` on a thread with a large stack. The generated straight-line
/// inference holds every intermediate tensor as a live stack local (the
/// arena pass is not yet integrated into the emitter), so multi-megabyte
/// models exceed the default 2 MiB test-thread stack. The panic message
/// from a failed assertion inside the closure is propagated by `join`.
fn on_large_stack(f: impl FnOnce() + Send + 'static) {
    let handle = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(f)
        .expect("spawn inference thread");
    match handle.join() {
        Ok(()) => {}
        Err(payload) => {
            let msg = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| format!("{payload:?}"));
            panic!("inference panicked: {msg}");
        }
    }
}

// ── Sine smoke — Model<I,O> contract on a trivial model ────────────────────

mod models_sine_smoke {
    use super::*;
    use hematite_codegen::model;

    #[model("../models/sine.tflite")]
    pub struct SineModel;

    #[test]
    fn sine_model_contract_and_bit_exact() {
        let _ = SineModel;
        on_large_stack(|| {
            let model = Model::new(RefBackend);

            assert_eq!(Model::<RefBackend>::input_len(), 1);
            assert_eq!(Model::<RefBackend>::output_len(), 1);

            let out = model.predict(&models::sine::INPUT_DATA);
            assert_bit_exact(&out, &models::sine::EXPECTED_OUTPUT, "sine_smoke");

            let mut out_buf = [0i8; 1];
            let mut scratch = [0u8; 0];
            let r = model.predict_with_scratch(&models::sine::INPUT_DATA, &mut out_buf, &mut scratch);
            assert_eq!(r, Ok(()));
            assert_bit_exact(&out_buf, &models::sine::EXPECTED_OUTPUT, "sine_smoke_scratch");

            assert_eq!(models::sine::MODEL_PATH, "models/sine.tflite");
        });
    }
}

// ── Hello world (sine regression) — 3× FullyConnected, tflite-micro ────────

mod models_hello_world {
    use super::*;
    use hematite_codegen::model;

    #[model("../models/zoo/sine_regression/hello_world_int8.tflite")]
    pub struct HelloWorldModel;

    #[test]
    fn hello_world_predict_bit_exact() {
        let _ = HelloWorldModel;
        on_large_stack(|| {
            let model = Model::new(RefBackend);
            let out = model.predict(&models::hello_world_int8::INPUT_DATA);
            assert_bit_exact(&out, &models::hello_world_int8::EXPECTED_OUTPUT, "hello_world");
            assert_eq!(Model::<RefBackend>::input_len(), flat_len(&models::hello_world_int8::INPUT_SHAPE));
            assert_eq!(Model::<RefBackend>::output_len(), flat_len(&models::hello_world_int8::OUTPUT_SHAPE));
            assert_eq!(
                models::hello_world_int8::MODEL_PATH,
                "models/zoo/sine_regression/hello_world_int8.tflite"
            );
        });
    }
}

// ── Keyword spotting (KWS family, matches keyword_spotting_v1) ─────────────

mod models_keyword_spotting {
    use super::*;
    use hematite_codegen::model;

    #[model("../models/zoo/keyword_spotting/kws_micro_speech_int8.tflite")]
    pub struct KeywordSpottingModel;

    #[test]
    fn kws_predict_bit_exact() {
        let _ = KeywordSpottingModel;
        on_large_stack(|| {
            let model = Model::new(RefBackend);
            let out = model.predict(&models::kws_micro_speech_int8::INPUT_DATA);
            assert_bit_exact(&out, &models::kws_micro_speech_int8::EXPECTED_OUTPUT, "kws");
            assert_eq!(Model::<RefBackend>::input_len(), flat_len(&models::kws_micro_speech_int8::INPUT_SHAPE));
            assert_eq!(Model::<RefBackend>::output_len(), flat_len(&models::kws_micro_speech_int8::OUTPUT_SHAPE));
        });
    }
}

// ── Anomaly detection autoencoder (matches anomaly_detect_v2) ──────────────

mod models_anomaly_detect {
    use super::*;
    use hematite_codegen::model;

    #[model("../models/zoo/anomaly_detect/anomaly_detect_int8.tflite")]
    pub struct AnomalyDetectModel;

    #[test]
    fn anomaly_detect_predict_bit_exact() {
        let _ = AnomalyDetectModel;
        on_large_stack(|| {
            let model = Model::new(RefBackend);
            let out = model.predict(&models::anomaly_detect_int8::INPUT_DATA);
            assert_bit_exact(&out, &models::anomaly_detect_int8::EXPECTED_OUTPUT, "anomaly_detect");
            assert_eq!(Model::<RefBackend>::input_len(), flat_len(&models::anomaly_detect_int8::INPUT_SHAPE));
            assert_eq!(Model::<RefBackend>::output_len(), flat_len(&models::anomaly_detect_int8::OUTPUT_SHAPE));
        });
    }
}

// ── Person detection (VWW, matches person_detect_v2) ───────────────────────
//
// Compiles + executes through the emitter (conv/depthwise/pool/reshape/fc/
// softmax), but is NOT asserted bit-exact: the host LiteRT reference
// kernels use double-rounding requantization and a different int8 softmax,
// so the model output differs from the TFLM-semantics hematite kernels at
// rounding boundaries. See module docs + DEFERRED_MODELS.md.

mod models_person_detect {
    use super::*;
    use hematite_codegen::model;

    #[model("../models/zoo/person_detect_vww/person_detect_int8.tflite")]
    pub struct PersonDetectModel;

    #[test]
    fn person_detect_compiles_and_executes() {
        let _ = PersonDetectModel;
        on_large_stack(|| {
            assert_eq!(Model::<RefBackend>::input_len(), flat_len(&models::person_detect_int8::INPUT_SHAPE));
            assert_eq!(Model::<RefBackend>::output_len(), flat_len(&models::person_detect_int8::OUTPUT_SHAPE));
            let model = Model::new(RefBackend);
            let _out = model.predict(&models::person_detect_int8::INPUT_DATA);
        });
    }
}

// ── Image classification (MobileNetV2 224², matches imagenet_cls) ──────────
//
// Compiles + executes through the emitter (transpose/pad/conv/depthwise/add/
// mean/reshape/fc/softmax — the widest op set in the zoo), but is NOT
// asserted bit-exact: the PAD kernel fills with raw 0 while LiteRT pads with
// the input zero point, and the conv/softmax rounding differences noted in
// the module docs apply. See DEFERRED_MODELS.md.

mod models_mobilenet_v2 {
    use super::*;
    use hematite_codegen::model;

    #[model("../models/zoo/mobilenetv2_cls/mobilenet_v2_1.0_224_int8.tflite")]
    pub struct MobileNetV2Model;

    #[test]
    fn mobilenet_v2_compiles_and_executes() {
        let _ = MobileNetV2Model;
        on_large_stack(|| {
            assert_eq!(Model::<RefBackend>::input_len(), flat_len(&models::mobilenet_v2_1_0_224_int8::INPUT_SHAPE));
            assert_eq!(Model::<RefBackend>::output_len(), flat_len(&models::mobilenet_v2_1_0_224_int8::OUTPUT_SHAPE));
            let model = Model::new(RefBackend);
            let _out = model.predict(&models::mobilenet_v2_1_0_224_int8::INPUT_DATA);
        });
    }
}

// ── Model::<S3Backend> zoo validation on host (plan todo 4, Wave 1) ─────────
//
// Every zoo model is instantiated as `Model::<S3Backend>` to prove the
// S3Backend trait wiring is correct on host. On the host the s3 kernels take
// the scalar fallbacks (the TIE728 SIMD glue is `#[cfg(target_arch =
// "xtensa")]`-gated), and the s3 scalar kernels implement the same TFLM-pinned
// semantics as the reference kernels — so wherever the op set is fully wired,
// `Model::<S3Backend>` output must equal `Model::<RefBackend>` output
// element-for-element (the RELATIVE check, and for the bit-exact models also
// the executed-TFLite golden).
//
// Op-set reality check (flatbuffer dump of each model's operator sequence):
//   sine            [9 FC]                         — fully wired on S3
//   hello_world     [9 FC ×3]                      — fully wired on S3
//   anomaly_detect  [9 FC ×10]                     — fully wired on S3
//   kws             [22 RESHAPE, 4, 9, 25]         — RESHAPE is op #1
//   person_detect   [3/4 convs ×27, 1, 22, 9, 25]  — RESHAPE near the tail
//   mobilenet_v2    [39 TRANSPOSE, 34 PAD ×18, …, 40, 22, 9] — TRANSPOSE op #1
//
// The committed S3Backend (e064e7b) returns `KernelError::Unsupported` for
// the data-movement ops it has no kernel for (reshape/transpose/pad — see the
// status matrix in `local-notes/evidence/simd-zoo-hardening/task-3-s3backend.log`),
// and `Model::predict` swallows the error (output left zeroed). So the three
// models whose op sequence includes an unwired op cannot produce an s3 output
// on the host; their tests assert the honest contract instead —
// `predict_with_scratch` returns `Err(KernelError::Unsupported)` at the exact
// unwired op, never a silent wrong answer. The RefBackend side of each test
// mirrors the existing RefBackend-only test (bit-exact where the golden
// applies, compile-and-execute otherwise).

// ── Sine via S3Backend — fully wired (FC only) ─────────────────────────────

#[cfg(feature = "hematite-s3")]
mod models_sine_s3_backend {
    use super::*;
    use hematite_codegen::model;

    #[model("../models/sine.tflite")]
    pub struct SineModelS3;

    #[test]
    fn sine_predict_bit_exact_via_s3() {
        let _ = SineModelS3;
        on_large_stack(|| {
            assert_eq!(Model::<S3Backend>::input_len(), 1);
            assert_eq!(Model::<S3Backend>::output_len(), 1);

            let s3_out = Model::<S3Backend>::new(S3Backend).predict(&models::sine::INPUT_DATA);
            let ref_out = Model::<RefBackend>::new(RefBackend).predict(&models::sine::INPUT_DATA);

            // Absolute: matches the executed-TFLite golden (same as RefBackend).
            assert_bit_exact(&s3_out, &models::sine::EXPECTED_OUTPUT, "sine_s3_golden");
            // Relative (the critical check): s3 == ref element-for-element.
            assert_bit_exact(&s3_out, &ref_out, "sine_s3_vs_ref");
        });
    }
}

// ── Hello world via S3Backend — fully wired (3× FC) ────────────────────────

#[cfg(feature = "hematite-s3")]
mod models_hello_world_s3_backend {
    use super::*;
    use hematite_codegen::model;

    #[model("../models/zoo/sine_regression/hello_world_int8.tflite")]
    pub struct HelloWorldModelS3;

    #[test]
    fn hello_world_predict_bit_exact_via_s3() {
        let _ = HelloWorldModelS3;
        on_large_stack(|| {
            let s3_out = Model::<S3Backend>::new(S3Backend)
                .predict(&models::hello_world_int8::INPUT_DATA);
            let ref_out = Model::<RefBackend>::new(RefBackend)
                .predict(&models::hello_world_int8::INPUT_DATA);

            assert_bit_exact(
                &s3_out,
                &models::hello_world_int8::EXPECTED_OUTPUT,
                "hello_world_s3_golden",
            );
            assert_bit_exact(&s3_out, &ref_out, "hello_world_s3_vs_ref");
        });
    }
}

// ── Keyword spotting via S3Backend — RESHAPE (op 22) is Unsupported ────────
//
// The model's operator sequence starts with RESHAPE (code 22), which the
// committed S3Backend has no kernel for. The RefBackend path keeps its
// bit-exact golden contract (mirrored below); the S3 path asserts the honest
// `Unsupported` failure — `predict` would swallow it into a zeroed output, so
// the contract is checked through `predict_with_scratch`.

#[cfg(feature = "hematite-s3")]
mod models_keyword_spotting_s3_backend {
    use super::*;
    use hematite_codegen::model;

    #[model("../models/zoo/keyword_spotting/kws_micro_speech_int8.tflite")]
    pub struct KeywordSpottingModelS3;

    #[test]
    fn kws_predict_via_s3() {
        let _ = KeywordSpottingModelS3;
        on_large_stack(|| {
            // RefBackend: unchanged bit-exact contract vs the golden.
            let ref_out = Model::<RefBackend>::new(RefBackend)
                .predict(&models::kws_micro_speech_int8::INPUT_DATA);
            assert_bit_exact(
                &ref_out,
                &models::kws_micro_speech_int8::EXPECTED_OUTPUT,
                "kws_s3_ref_golden",
            );

            // S3Backend: op #1 is RESHAPE (22) — no s3 kernel, honest Err.
            let s3 = Model::<S3Backend>::new(S3Backend);
            let mut out_buf = [0i8; Model::<S3Backend>::output_len()];
            let mut scratch = [0u8; 65536];
            let r = s3.predict_with_scratch(
                &models::kws_micro_speech_int8::INPUT_DATA,
                &mut out_buf,
                &mut scratch,
            );
            assert_eq!(
                r,
                Err(::hematite_core::KernelError::Unsupported),
                "kws_s3: model op sequence starts with RESHAPE (22); S3Backend must \
                 report Unsupported, not silently produce a zeroed output",
            );
        });
    }
}

// ── Anomaly detection via S3Backend — fully wired (10× FC) ─────────────────

#[cfg(feature = "hematite-s3")]
mod models_anomaly_detect_s3_backend {
    use super::*;
    use hematite_codegen::model;

    #[model("../models/zoo/anomaly_detect/anomaly_detect_int8.tflite")]
    pub struct AnomalyDetectModelS3;

    #[test]
    fn anomaly_detect_predict_bit_exact_via_s3() {
        let _ = AnomalyDetectModelS3;
        on_large_stack(|| {
            let s3_out = Model::<S3Backend>::new(S3Backend)
                .predict(&models::anomaly_detect_int8::INPUT_DATA);
            let ref_out = Model::<RefBackend>::new(RefBackend)
                .predict(&models::anomaly_detect_int8::INPUT_DATA);

            assert_bit_exact(
                &s3_out,
                &models::anomaly_detect_int8::EXPECTED_OUTPUT,
                "anomaly_detect_s3_golden",
            );
            assert_bit_exact(&s3_out, &ref_out, "anomaly_detect_s3_vs_ref");
        });
    }
}

// ── Person detection via S3Backend — RESHAPE (op 22) is Unsupported ────────
//
// 27 convs + average pool execute on S3, then the RESHAPE (code 22) returns
// `Unsupported` — honest failure, and the reason the model is NOT bit-exact
// capable on the S3 backend yet (T10/T11 execute-TFLM goldens + data-movement
// kernels). RefBackend side mirrors `models_person_detect` (compile+execute).

#[cfg(feature = "hematite-s3")]
mod models_person_detect_s3_backend {
    use super::*;
    use hematite_codegen::model;

    #[model("../models/zoo/person_detect_vww/person_detect_int8.tflite")]
    pub struct PersonDetectModelS3;

    #[test]
    fn person_detect_predict_via_s3() {
        let _ = PersonDetectModelS3;
        on_large_stack(|| {
            assert_eq!(Model::<S3Backend>::input_len(), flat_len(&models::person_detect_int8::INPUT_SHAPE));
            assert_eq!(Model::<S3Backend>::output_len(), flat_len(&models::person_detect_int8::OUTPUT_SHAPE));

            // RefBackend: full model executes (known ±1 golden divergence —
            // rounding provenance; see module docs).
            let _ref_out = Model::<RefBackend>::new(RefBackend)
                .predict(&models::person_detect_int8::INPUT_DATA);

            // S3Backend: convs run, then RESHAPE (22) → honest Err.
            let s3 = Model::<S3Backend>::new(S3Backend);
            let mut out_buf = [0i8; Model::<S3Backend>::output_len()];
            let mut scratch = [0u8; 65536];
            let r = s3.predict_with_scratch(
                &models::person_detect_int8::INPUT_DATA,
                &mut out_buf,
                &mut scratch,
            );
            assert_eq!(
                r,
                Err(::hematite_core::KernelError::Unsupported),
                "person_detect_s3: model op sequence contains RESHAPE (22); S3Backend \
                 must report Unsupported, not silently produce a zeroed output",
            );
        });
    }
}

// ── MobileNetV2 via S3Backend — TRANSPOSE/PAD/RESHAPE are Unsupported ──────
//
// The model's op sequence starts with TRANSPOSE (39) then 18× PAD (34) and a
// tail RESHAPE (22) — none have s3 kernels. RefBackend side mirrors
// `models_mobilenet_v2` (compile+execute); the S3 side asserts the honest
// `Unsupported` at the very first op.

#[cfg(feature = "hematite-s3")]
mod models_mobilenet_v2_s3_backend {
    use super::*;
    use hematite_codegen::model;

    #[model("../models/zoo/mobilenetv2_cls/mobilenet_v2_1.0_224_int8.tflite")]
    pub struct MobileNetV2ModelS3;

    #[test]
    fn mobilenet_v2_predict_via_s3() {
        let _ = MobileNetV2ModelS3;
        on_large_stack(|| {
            assert_eq!(Model::<S3Backend>::input_len(), flat_len(&models::mobilenet_v2_1_0_224_int8::INPUT_SHAPE));
            assert_eq!(Model::<S3Backend>::output_len(), flat_len(&models::mobilenet_v2_1_0_224_int8::OUTPUT_SHAPE));

            // RefBackend: full model executes (PAD zero-fill + rounding
            // divergence vs golden; see module docs).
            let _ref_out = Model::<RefBackend>::new(RefBackend)
                .predict(&models::mobilenet_v2_1_0_224_int8::INPUT_DATA);

            // S3Backend: op #1 is TRANSPOSE (39) → honest Err.
            let s3 = Model::<S3Backend>::new(S3Backend);
            let mut out_buf = [0i8; Model::<S3Backend>::output_len()];
            let mut scratch = [0u8; 65536];
            let r = s3.predict_with_scratch(
                &models::mobilenet_v2_1_0_224_int8::INPUT_DATA,
                &mut out_buf,
                &mut scratch,
            );
            assert_eq!(
                r,
                Err(::hematite_core::KernelError::Unsupported),
                "mobilenet_v2_s3: model op sequence starts with TRANSPOSE (39); \
                 S3Backend must report Unsupported, not silently produce a zeroed output",
            );
        });
    }
}
