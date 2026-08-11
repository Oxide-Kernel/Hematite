// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Model-level benchmarks (plan T5.3 / simd-zoo-hardening todo 19).
//!
//! # Structure — model path is a parameter
//!
//! This module separates the **benchmark definition** (the spec: name, model
//! path, reference bar, memory tier) from the **runner** (the thing that
//! executes an inference).
//!
//! The runner is abstracted behind [`ModelRunner`], whose shape mirrors the
//! generated-code API that `hematite-codegen`'s `#[model("path.tflite")]`
//! macro emits (`Model<B>` with `input_len()`, `output_len()`,
//! `predict_with_scratch`).  Todo 19 wires a concrete generated `Model<B>` for
//! every zoo model into [`ModelRunner`] (one small adapter per model, see the
//! `zoo_runners` module — compiled behind the `model-validation` feature that
//! links `hematite-codegen`): the spec's `path` is now the literal `#[model]`
//! attribute argument and the harness below runs unchanged.
//!
//! The bench buffers (input / output / scratch) are caller-provided; the
//! device firmware carves them from the SRAM/PSRAM arena with
//! [`carve_model_bufs`] — the same carve pattern `bench_kernel` uses.
//!
//! # Reference bars (B2 — single-core, documented sources; T6.3 re-tier)
//!
//! | Model | Bar | Source |
//! |-------|-----|--------|
//! | MobileNetV2 224×224 | **1294.5 ms** single-core — **hold-as-documented** | ESP-DL reference, plan T5.3 (B2). The 856 ms figure is the DUAL-core number — never the bar. PSRAM-gated on this board (`PSRAM: 0 bytes`, PROJECT_LOG.md:721): the model row is a PSRAM-gated follow-up, the bar itself is untouched. |
//! | KWS keyword_spotting_v1 (1×1960) | **4 ms** — ESP-NN-relative via T3.5b (target < 1,059,889 cyc / 4 ms) | RE-TIERED from the 7 ms ESP-DL bar: PROJECT_LOG.md:796-799 documented the original 54 ms / 7 ms structural bound (dm=8 depthwise); T3.5b's anytap dm>1 depthwise now SIMD-engages (`depthwise_kws_49x40x1_10x8_dm8` on-device PASS, t61-device.log:77; spec row `t35b-depthwise-anyfilter.md:109-121`) |
//!
//! Internal speedup floors (×100 fixed point): **conv1x1 ≥ 15.57× vs scalar-ref**
//! (1557, spec.rs:1681 column-2 bar — holds) and **conv SIMD ≥ 10× vs scalar**
//! (1000, T3.0 column-1 floor — holds for all zoo models).
//!
//! `reference_ms_tenths` stores these in ×10 fixed point (12945 = 1294.5 ms,
//! 40 = 4.0 ms) so the pass/fail comparison is pure integer math — no f32
//! anywhere.

use crate::spec::MemoryTier;
use crate::timing::{run_repeated, BenchmarkConfig, Clock, RunLog, RunSummary};

/// A model-level benchmark definition.  `path` is the model-path parameter
/// (relative to the crate that carries the `.tflite`; resolved at T5.2 when
/// the file lands).
pub struct ModelBenchSpec {
    /// Row label.
    pub name: &'static str,
    /// Zoo model path (the `#[model]` attribute argument at T5.2 wiring time).
    pub path: &'static str,
    /// Flat input element count.
    pub input_len: usize,
    /// Flat output element count.
    pub output_len: usize,
    /// Working-set memory tier.
    pub tier: MemoryTier,
    /// Documented reference bar in ×10 fixed-point milliseconds (e.g. 12945
    /// = 1294.5 ms).  `None` = no bar (smoke rows).
    pub reference_ms_tenths: Option<u32>,
    /// Provenance citation for the bar (MUST-NOT-invent-numbers rule).
    pub source: &'static str,
}

/// MobileNetV2 224×224 — **single-core** acceptance bar.
///
/// B2 resolves the bar to 1294.5 ms (ESP-DL single-core).  The 856 ms figure
/// is dual-core and is NEVER the comparison bar (plan T5.3 line 308).
///
/// `path` is the real zoo tflite (todo 19 — the old
/// `models/mobilenet_v2.tflite` placeholder never existed).  `output_len` is
/// the model's real 1000-class logits head (golden `EXPECTED_OUTPUT: [i8; 1000]`),
/// not the stale 1001.
pub const MOBILENETV2_SPEC: ModelBenchSpec = ModelBenchSpec {
    name: "MobileNetV2 224x224 (single-core bar 1294.5 ms)",
    path: "models/zoo/mobilenetv2_cls/mobilenet_v2_1.0_224_int8.tflite",
    input_len: 224 * 224 * 3,
    output_len: 1000,
    tier: MemoryTier::Psram,
    reference_ms_tenths: Some(12945), // 1294.5 ms — hold-as-documented (T6.3)
    source: "plan T5.3 line 308 (B2): ESP-DL single-core reference 1294.5 ms; 856 ms is DUAL-core, NOT the bar. PSRAM-gated on this board: PSRAM: 0 bytes (PROJECT_LOG.md:721) — model row is a PSRAM-gated follow-up (head-to-head.md §speed-closure (a))",
};

/// KWS — kws_micro_speech_int8, input 1×1960.
///
/// `path` is the real zoo tflite (todo 19 — the old
/// `models/keyword_spotting_v1.tflite` placeholder never existed).
/// `output_len` is the model's real 4-class output (golden
/// `EXPECTED_OUTPUT: [i8; 4]`), not the stale "12 classes pending" guess.
pub const KWS_SPEC: ModelBenchSpec = ModelBenchSpec {
    name: "KWS kws_micro_speech_int8 (bar 4 ms — ESP-NN-relative, T3.5b)",
    path: "models/zoo/keyword_spotting/kws_micro_speech_int8.tflite",
    input_len: 1960, // 1×1960 per plan T5.3
    output_len: 4,
    tier: MemoryTier::Sram,
    reference_ms_tenths: Some(40), // 4.0 ms — RE-TIERED (was 70 = 7 ms ESP-DL)
    source: "plan T5.3 line 308 (7 ms ESP-DL) RE-TIERED to ESP-NN-relative target < 1,059,889 cyc / 4 ms: PROJECT_LOG.md:796-799 (original 54 ms / 7 ms structural bound, dm=8); T3.5b anytap dm>1 depthwise SIMD-engages, depthwise_kws_49x40x1_10x8_dm8 on-device PASS t61-device.log:77; spec row t35b-depthwise-anyfilter.md:109-121",
};

/// Sine smoke model — shipped in `models/` since T0, usable today as an
/// end-to-end wiring proof once a runner exists (no bar).
pub const SINE_SPEC: ModelBenchSpec = ModelBenchSpec {
    name: "sine (smoke — model present since T0)",
    path: "models/sine.tflite",
    input_len: 1,
    output_len: 1,
    tier: MemoryTier::Sram,
    reference_ms_tenths: None,
    source: "workspace models/sine.tflite, used by hematite-codegen model_smoke test",
};

/// hello_world (sine_regression zoo) — 1→1 smoke row (no bar).
pub const HELLO_WORLD_SPEC: ModelBenchSpec = ModelBenchSpec {
    name: "hello_world_int8 (sine_regression zoo)",
    path: "models/zoo/sine_regression/hello_world_int8.tflite",
    input_len: 1,
    output_len: 1,
    tier: MemoryTier::Sram,
    reference_ms_tenths: None,
    source: "workspace models/zoo/sine_regression/hello_world_int8.tflite (smoke — no bar)",
};

/// person_detect (VWW) — 96×96×3→2.
///
/// Timed row is a SKIP on this board: the generated `predict` allocas ~232 KB
/// of stack intermediates vs the ~65 KB device stack (todo-5 finding) — the
/// SKIP record is the honest outcome, never a weakened run.
pub const PERSON_DETECT_SPEC: ModelBenchSpec = ModelBenchSpec {
    name: "person_detect_int8 (VWW 96x96x3, SKIP on device: stack)",
    path: "models/zoo/person_detect_vww/person_detect_int8.tflite",
    input_len: 96 * 96 * 3,
    output_len: 2,
    tier: MemoryTier::Sram,
    reference_ms_tenths: None,
    source: "workspace models/zoo/person_detect_vww/person_detect_int8.tflite (SKIP on device: 232 KB stack alloca vs ~65 KB stack)",
};

/// anomaly_detect — 640→640 smoke row (no bar).
pub const ANOMALY_SPEC: ModelBenchSpec = ModelBenchSpec {
    name: "anomaly_detect_int8 (640->640)",
    path: "models/zoo/anomaly_detect/anomaly_detect_int8.tflite",
    input_len: 640,
    output_len: 640,
    tier: MemoryTier::Sram,
    reference_ms_tenths: None,
    source: "workspace models/zoo/anomaly_detect/anomaly_detect_int8.tflite (smoke — no bar)",
};

/// The full model-level benchmark registry (report row order).
pub const fn model_bench_specs() -> &'static [ModelBenchSpec] {
    &[
        MOBILENETV2_SPEC,
        KWS_SPEC,
        SINE_SPEC,
        HELLO_WORLD_SPEC,
        PERSON_DETECT_SPEC,
        ANOMALY_SPEC,
    ]
}

/// Internal speedup acceptance floors, ×100 fixed point (T6.3 re-tier).
///
/// These mirror the spec.rs `CompetitorBaseline::target_speedup_x100` bars
/// (conv1x1 `Some(1557)`, spec.rs:1681) and the T3.0 column-1 floor.  Kept
/// here so the model-level docs and the report footer render the same
/// plan-attributed bars as the per-kernel rows — no invented numbers.
pub mod speedup_bars {
    /// conv1x1 64×1×1×64 vs scalar-ref — column-2 acceptance bar
    /// **15.57×** (spec.rs:1681, plan T5.3 line 309; recorded hold in
    /// head-to-head.md §6.8).  Retained as-is by T6.3 — the on-device SIMD
    /// row measured 3201 cyc public-API (ESPRESSIF_VS_HEMATITE.md:227).
    pub const CONV1X1_VS_SCALAR_X100: u32 = 1557;
    /// conv SIMD vs scalar-ref — T3.0 column-1 floor **10×** (head-to-head.md
    /// §6.8).  Holds for all zoo models: Model C measures 22× vs its scalar
    /// ref 14,489,859 cyc (head-to-head.md §5.1); every SIMD-engaged layer is
    /// above the floor per ESPRESSIF_VS_HEMATITE.md §5.2.
    pub const SIMD_VS_SCALAR_FLOOR_X100: u32 = 1000;
}

/// Abstract inference runner — mirrors the generated `Model<B>` API shape
/// (`input_len` / `output_len` / `predict`).  A T5.2 adapter implements this
/// over the `#[model]`-generated code.
pub trait ModelRunner {
    /// Flat input element count.
    fn input_len(&self) -> usize;
    /// Flat output element count.
    fn output_len(&self) -> usize;
    /// Run one full inference.
    ///
    /// Returns `true` on success, `false` on kernel error.  The benchmark
    /// still records the cycles (the run is real work either way) and the
    /// firmware flags rows whose runner failed.
    fn predict(&mut self, input: &[i8], output: &mut [i8], scratch: &mut [u8]) -> bool;
}

/// Benchmark a model runner: warm-up + N ≥ 10 timed inferences (C3).
///
/// Buffers are caller-provided (device is no-heap; the firmware carves them
/// from the SRAM/PSRAM arenas).
pub fn run_model_bench<C: Clock, R: ModelRunner>(
    clock: &mut C,
    runner: &mut R,
    input: &[i8],
    output: &mut [i8],
    scratch: &mut [u8],
    cfg: &BenchmarkConfig,
) -> RunLog {
    run_repeated(
        clock,
        &mut || {
            // Discard the result: a failing inference still exercises the
            // full dispatch path and its cycles are real work.  Rows are
            // flagged for interpretation, not dropped.
            let _ = runner.predict(input, output, scratch);
        },
        cfg,
    )
}

/// Convert a wall-clock delta in ns to ×10 fixed-point milliseconds
/// (integer math; 12945000000 ns → 12945 = 1294.5 ms).
pub fn wall_ms_tenths(wall_ns: u64) -> u64 {
    wall_ns * 10 / 1_000_000
}

/// 16-byte-aligned byte offsets of a model bench's
/// `[input][pad][output][pad][scratch]` buffer layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelBufLayout {
    pub input_off: usize,
    pub output_off: usize,
    pub scratch_off: usize,
    /// Total bytes consumed (`scratch_off + scratch_len`, 16-aligned).
    pub total: usize,
}

/// Compute the aligned model-bench buffer layout for `arena_len` bytes.
///
/// Mirrors `spec::carve_into`: every region sits on a 16-byte boundary (the
/// SIMD path requires 16-byte alignment for `EE.VLD.128` / `EE.VST.128`).
/// Returns `None` when the arena cannot hold the working set.
pub fn carve_model_layout(
    input_len: usize,
    output_len: usize,
    scratch_len: usize,
    arena_len: usize,
) -> Option<ModelBufLayout> {
    let align16 = |o: usize| o.div_ceil(16) * 16;
    let input_off = 0usize;
    let output_off = align16(input_len);
    let scratch_off = align16(output_off.checked_add(output_len)?);
    let total = align16(scratch_off.checked_add(scratch_len)?);
    if total > arena_len {
        return None;
    }
    Some(ModelBufLayout {
        input_off,
        output_off,
        scratch_off,
        total,
    })
}

/// Mutable slices backing one model inference (caller-provided buffers).
pub struct ModelBufs<'a> {
    pub input: &'a mut [i8],
    pub output: &'a mut [i8],
    pub scratch: &'a mut [u8],
}

/// Carve a model bench's input/output/scratch buffers out of an arena.
///
/// Offsets are 16-byte aligned per [`carve_model_layout`]; the layout is
/// `[input][pad][output][pad][scratch]`.  Returns `None` when the arena
/// cannot hold the working set.
pub fn carve_model_bufs<'a>(
    arena: &'a mut [u8],
    input_len: usize,
    output_len: usize,
    scratch_len: usize,
) -> Option<ModelBufs<'a>> {
    let lay = carve_model_layout(input_len, output_len, scratch_len, arena.len())?;
    let (input_region, rest) = arena.split_at_mut(lay.output_off);
    let input = unsafe { cast_i8(&mut input_region[lay.input_off..lay.input_off + input_len]) };
    let (output_region, rest) = rest.split_at_mut(lay.scratch_off - lay.output_off);
    let output = unsafe { cast_i8(&mut output_region[..output_len]) };
    let (scratch_region, _) = rest.split_at_mut(scratch_len);
    Some(ModelBufs {
        input,
        output,
        scratch: &mut scratch_region[..],
    })
}

/// Reinterpret a byte slice as an int8 slice.
///
/// # Safety
///
/// `u8` and `i8` have identical size/alignment/layout; the reborrow keeps the
/// same lifetime and mutability.  Sound whenever the source borrow is valid.
unsafe fn cast_i8(s: &mut [u8]) -> &mut [i8] {
    core::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut i8, s.len())
}

/// Fill a bench input with the deterministic ramp pattern used across the
/// benchmark suite (position-varying so MAC values are non-trivial).
pub fn fill_input_pattern(input: &mut [i8]) {
    for (i, v) in input.iter_mut().enumerate() {
        *v = (i.wrapping_mul(7).wrapping_add(3) & 0xFF) as i8;
    }
}

/// Evaluate a measured run against a documented reference bar.
///
/// Both sides are ×10 fixed-point ms (integer).  `true` = measured median
/// wall time ≤ the bar (pass).
pub fn passes_reference_bar(summary: &RunSummary, bar_tenths: u32) -> bool {
    wall_ms_tenths(summary.median_wall_ns) <= u64::from(bar_tenths)
}

// ── Real zoo-model runners (plan simd-zoo-hardening todo 19) ────────────────
//
// Every registry spec's `path` is wired to a `#[model]`-generated
// `Model::<S3Backend>` through one small adapter.  The generated code needs
// `hematite-codegen` (the `model-validation` feature) and the firmware arena
// for its buffers — so this module is feature-gated and the adapters are the
// only concrete `ModelRunner` impls beyond the host-test `FakeRunner`.

/// Real-zoo `ModelRunner` adapters over `#[model]`-generated `Model::<S3Backend>`.
#[cfg(feature = "model-validation")]
pub mod zoo_runners {
    use super::{ModelBenchSpec, ModelRunner};
    use hematite_codegen::model;
    use hematite_s3::backend::S3Backend;

    // Each model gets its own module: the `#[model]` macro emits module-scope
    // `Model<B>` plus `INPUT_LEN`/`OUTPUT_LEN`/`SCRATCH_LEN` consts, which
    // would collide at file scope (same isolation model_validation.rs uses).
    // The adapter exposes the slice-based `ModelRunner` API over the
    // generated array API with caller-provided scratch
    // (`predict_with_scratch` — no internal buffer).

    mod model_sine {
        use super::*;
        #[model("../models/sine.tflite")]
        pub struct SineModel;
        pub struct SineRunner {
            model: Model<S3Backend>,
        }
        impl SineRunner {
            pub fn new() -> Self {
                Self { model: Model::new(S3Backend) }
            }
            pub fn input_len(&self) -> usize {
                INPUT_LEN
            }
            pub fn output_len(&self) -> usize {
                OUTPUT_LEN
            }
            pub fn scratch_len(&self) -> usize {
                SCRATCH_LEN
            }
            pub fn run(&mut self, input: &[i8], output: &mut [i8], scratch: &mut [u8]) -> bool {
                let input_arr: &[i8; INPUT_LEN] = match input.try_into() {
                    Ok(a) => a,
                    Err(_) => return false,
                };
                let output_arr: &mut [i8; OUTPUT_LEN] = match output.try_into() {
                    Ok(b) => b,
                    Err(_) => return false,
                };
                self.model.predict_with_scratch(input_arr, output_arr, scratch).is_ok()
            }
        }
    }

    mod model_hello_world {
        use super::*;
        #[model("../models/zoo/sine_regression/hello_world_int8.tflite")]
        pub struct HelloWorldModel;
        pub struct HelloWorldRunner {
            model: Model<S3Backend>,
        }
        impl HelloWorldRunner {
            pub fn new() -> Self {
                Self { model: Model::new(S3Backend) }
            }
            pub fn input_len(&self) -> usize {
                INPUT_LEN
            }
            pub fn output_len(&self) -> usize {
                OUTPUT_LEN
            }
            pub fn scratch_len(&self) -> usize {
                SCRATCH_LEN
            }
            pub fn run(&mut self, input: &[i8], output: &mut [i8], scratch: &mut [u8]) -> bool {
                let input_arr: &[i8; INPUT_LEN] = match input.try_into() {
                    Ok(a) => a,
                    Err(_) => return false,
                };
                let output_arr: &mut [i8; OUTPUT_LEN] = match output.try_into() {
                    Ok(b) => b,
                    Err(_) => return false,
                };
                self.model.predict_with_scratch(input_arr, output_arr, scratch).is_ok()
            }
        }
    }

    mod model_kws {
        use super::*;
        #[model("../models/zoo/keyword_spotting/kws_micro_speech_int8.tflite")]
        pub struct KwsModel;
        pub struct KwsRunner {
            model: Model<S3Backend>,
        }
        impl KwsRunner {
            pub fn new() -> Self {
                Self { model: Model::new(S3Backend) }
            }
            pub fn input_len(&self) -> usize {
                INPUT_LEN
            }
            pub fn output_len(&self) -> usize {
                OUTPUT_LEN
            }
            pub fn scratch_len(&self) -> usize {
                SCRATCH_LEN
            }
            pub fn run(&mut self, input: &[i8], output: &mut [i8], scratch: &mut [u8]) -> bool {
                let input_arr: &[i8; INPUT_LEN] = match input.try_into() {
                    Ok(a) => a,
                    Err(_) => return false,
                };
                let output_arr: &mut [i8; OUTPUT_LEN] = match output.try_into() {
                    Ok(b) => b,
                    Err(_) => return false,
                };
                self.model.predict_with_scratch(input_arr, output_arr, scratch).is_ok()
            }
        }
    }

    mod model_anomaly {
        use super::*;
        #[model("../models/zoo/anomaly_detect/anomaly_detect_int8.tflite")]
        pub struct AnomalyModel;
        pub struct AnomalyRunner {
            model: Model<S3Backend>,
        }
        impl AnomalyRunner {
            pub fn new() -> Self {
                Self { model: Model::new(S3Backend) }
            }
            pub fn input_len(&self) -> usize {
                INPUT_LEN
            }
            pub fn output_len(&self) -> usize {
                OUTPUT_LEN
            }
            pub fn scratch_len(&self) -> usize {
                SCRATCH_LEN
            }
            pub fn run(&mut self, input: &[i8], output: &mut [i8], scratch: &mut [u8]) -> bool {
                let input_arr: &[i8; INPUT_LEN] = match input.try_into() {
                    Ok(a) => a,
                    Err(_) => return false,
                };
                let output_arr: &mut [i8; OUTPUT_LEN] = match output.try_into() {
                    Ok(b) => b,
                    Err(_) => return false,
                };
                self.model.predict_with_scratch(input_arr, output_arr, scratch).is_ok()
            }
        }
    }

    mod model_person_detect {
        use super::*;
        #[model("../models/zoo/person_detect_vww/person_detect_int8.tflite")]
        pub struct PersonDetectModel;
        pub struct PersonDetectRunner {
            model: Model<S3Backend>,
        }
        impl PersonDetectRunner {
            pub fn new() -> Self {
                Self { model: Model::new(S3Backend) }
            }
            pub fn input_len(&self) -> usize {
                INPUT_LEN
            }
            pub fn output_len(&self) -> usize {
                OUTPUT_LEN
            }
            pub fn scratch_len(&self) -> usize {
                SCRATCH_LEN
            }
            pub fn run(&mut self, input: &[i8], output: &mut [i8], scratch: &mut [u8]) -> bool {
                let input_arr: &[i8; INPUT_LEN] = match input.try_into() {
                    Ok(a) => a,
                    Err(_) => return false,
                };
                let output_arr: &mut [i8; OUTPUT_LEN] = match output.try_into() {
                    Ok(b) => b,
                    Err(_) => return false,
                };
                self.model.predict_with_scratch(input_arr, output_arr, scratch).is_ok()
            }
        }
    }

    mod model_mobilenet {
        use super::*;
        #[model("../models/zoo/mobilenetv2_cls/mobilenet_v2_1.0_224_int8.tflite")]
        pub struct MobilenetModel;
        pub struct MobilenetRunner {
            model: Model<S3Backend>,
        }
        impl MobilenetRunner {
            pub fn new() -> Self {
                Self { model: Model::new(S3Backend) }
            }
            pub fn input_len(&self) -> usize {
                INPUT_LEN
            }
            pub fn output_len(&self) -> usize {
                OUTPUT_LEN
            }
            pub fn scratch_len(&self) -> usize {
                SCRATCH_LEN
            }
            pub fn run(&mut self, input: &[i8], output: &mut [i8], scratch: &mut [u8]) -> bool {
                let input_arr: &[i8; INPUT_LEN] = match input.try_into() {
                    Ok(a) => a,
                    Err(_) => return false,
                };
                let output_arr: &mut [i8; OUTPUT_LEN] = match output.try_into() {
                    Ok(b) => b,
                    Err(_) => return false,
                };
                self.model.predict_with_scratch(input_arr, output_arr, scratch).is_ok()
            }
        }
    }

    /// A constructed zoo-model runner (all six models; the firmware decides
    /// per-spec SKIPs — person_detect stack, mobilenet_v2 PSRAM — before
    /// timing).
    pub enum ZooRunner {
        Sine(model_sine::SineRunner),
        HelloWorld(model_hello_world::HelloWorldRunner),
        Kws(model_kws::KwsRunner),
        Anomaly(model_anomaly::AnomalyRunner),
        PersonDetect(model_person_detect::PersonDetectRunner),
        Mobilenet(model_mobilenet::MobilenetRunner),
    }

    impl ZooRunner {
        pub fn input_len(&self) -> usize {
            match self {
                ZooRunner::Sine(r) => r.input_len(),
                ZooRunner::HelloWorld(r) => r.input_len(),
                ZooRunner::Kws(r) => r.input_len(),
                ZooRunner::Anomaly(r) => r.input_len(),
                ZooRunner::PersonDetect(r) => r.input_len(),
                ZooRunner::Mobilenet(r) => r.input_len(),
            }
        }
        pub fn output_len(&self) -> usize {
            match self {
                ZooRunner::Sine(r) => r.output_len(),
                ZooRunner::HelloWorld(r) => r.output_len(),
                ZooRunner::Kws(r) => r.output_len(),
                ZooRunner::Anomaly(r) => r.output_len(),
                ZooRunner::PersonDetect(r) => r.output_len(),
                ZooRunner::Mobilenet(r) => r.output_len(),
            }
        }
        pub fn scratch_len(&self) -> usize {
            match self {
                ZooRunner::Sine(r) => r.scratch_len(),
                ZooRunner::HelloWorld(r) => r.scratch_len(),
                ZooRunner::Kws(r) => r.scratch_len(),
                ZooRunner::Anomaly(r) => r.scratch_len(),
                ZooRunner::PersonDetect(r) => r.scratch_len(),
                ZooRunner::Mobilenet(r) => r.scratch_len(),
            }
        }
    }

    impl ModelRunner for ZooRunner {
        fn input_len(&self) -> usize {
            self.input_len()
        }
        fn output_len(&self) -> usize {
            self.output_len()
        }
        fn predict(&mut self, input: &[i8], output: &mut [i8], scratch: &mut [u8]) -> bool {
            match self {
                ZooRunner::Sine(r) => r.run(input, output, scratch),
                ZooRunner::HelloWorld(r) => r.run(input, output, scratch),
                ZooRunner::Kws(r) => r.run(input, output, scratch),
                ZooRunner::Anomaly(r) => r.run(input, output, scratch),
                ZooRunner::PersonDetect(r) => r.run(input, output, scratch),
                ZooRunner::Mobilenet(r) => r.run(input, output, scratch),
            }
        }
    }

    /// Construct the runner for a registry spec.
    ///
    /// Panics on an unwired spec path — a registry entry without a runner is
    /// a build-time wiring error, not a runtime question.
    pub fn zoo_runner_for(spec: &ModelBenchSpec) -> ZooRunner {
        match spec.path {
            "models/sine.tflite" => ZooRunner::Sine(model_sine::SineRunner::new()),
            "models/zoo/sine_regression/hello_world_int8.tflite" => {
                ZooRunner::HelloWorld(model_hello_world::HelloWorldRunner::new())
            }
            "models/zoo/keyword_spotting/kws_micro_speech_int8.tflite" => {
                ZooRunner::Kws(model_kws::KwsRunner::new())
            }
            "models/zoo/anomaly_detect/anomaly_detect_int8.tflite" => {
                ZooRunner::Anomaly(model_anomaly::AnomalyRunner::new())
            }
            "models/zoo/person_detect_vww/person_detect_int8.tflite" => {
                ZooRunner::PersonDetect(model_person_detect::PersonDetectRunner::new())
            }
            "models/zoo/mobilenetv2_cls/mobilenet_v2_1.0_224_int8.tflite" => {
                ZooRunner::Mobilenet(model_mobilenet::MobilenetRunner::new())
            }
            other => panic!("zoo_runner_for: no runner wired for spec path '{other}'"),
        }
    }
}

/// Unfused-arm real-zoo `ModelRunner` adapters (T6.1 fused-vs-unfused delta).
///
/// The fused `#[model]` adapters above emit composed kernel calls for T1
/// groups (residual/chain/epilogue).  This module emits the SAME model with
/// `#[model_unfused]` — the plain per-op sequence with no fusion schedule —
/// so timing `Model::<S3Backend>` through both arms isolates the cycle cost
/// of fusion on real silicon.  For sine / hello_world / kws / anomaly the W0
/// profile found ZERO composed groups, so fused == unfused emission (the
/// delta is expected ≈ 0 and proves the fused dispatch adds no overhead);
/// mobilenet_v2 has 10 residual groups but is PSRAM-gated on this board
/// (PSRAM: 0 bytes) and person_detect is stack-gated — neither has an
/// unfused arm wired here (unreachable on this board, see
/// `firmware::bench_zoo_model` SKIP guards).
#[cfg(feature = "model-validation")]
pub mod zoo_unfused_runners {
    use super::{ModelBenchSpec, ModelRunner};
    use hematite_codegen::model_unfused;
    use hematite_s3::backend::S3Backend;

    mod model_sine_unfused {
        use super::*;
        #[model_unfused("../models/sine.tflite")]
        pub struct SineModel;
        pub struct SineRunner {
            model: Model<S3Backend>,
        }
        impl SineRunner {
            pub fn new() -> Self {
                Self { model: Model::new(S3Backend) }
            }
            pub fn input_len(&self) -> usize {
                INPUT_LEN
            }
            pub fn output_len(&self) -> usize {
                OUTPUT_LEN
            }
            pub fn scratch_len(&self) -> usize {
                SCRATCH_LEN
            }
            pub fn run(&mut self, input: &[i8], output: &mut [i8], scratch: &mut [u8]) -> bool {
                let input_arr: &[i8; INPUT_LEN] = match input.try_into() {
                    Ok(a) => a,
                    Err(_) => return false,
                };
                let output_arr: &mut [i8; OUTPUT_LEN] = match output.try_into() {
                    Ok(b) => b,
                    Err(_) => return false,
                };
                self.model.predict_with_scratch(input_arr, output_arr, scratch).is_ok()
            }
        }
    }

    mod model_hello_world_unfused {
        use super::*;
        #[model_unfused("../models/zoo/sine_regression/hello_world_int8.tflite")]
        pub struct HelloWorldModel;
        pub struct HelloWorldRunner {
            model: Model<S3Backend>,
        }
        impl HelloWorldRunner {
            pub fn new() -> Self {
                Self { model: Model::new(S3Backend) }
            }
            pub fn input_len(&self) -> usize {
                INPUT_LEN
            }
            pub fn output_len(&self) -> usize {
                OUTPUT_LEN
            }
            pub fn scratch_len(&self) -> usize {
                SCRATCH_LEN
            }
            pub fn run(&mut self, input: &[i8], output: &mut [i8], scratch: &mut [u8]) -> bool {
                let input_arr: &[i8; INPUT_LEN] = match input.try_into() {
                    Ok(a) => a,
                    Err(_) => return false,
                };
                let output_arr: &mut [i8; OUTPUT_LEN] = match output.try_into() {
                    Ok(b) => b,
                    Err(_) => return false,
                };
                self.model.predict_with_scratch(input_arr, output_arr, scratch).is_ok()
            }
        }
    }

    mod model_kws_unfused {
        use super::*;
        #[model_unfused("../models/zoo/keyword_spotting/kws_micro_speech_int8.tflite")]
        pub struct KwsModel;
        pub struct KwsRunner {
            model: Model<S3Backend>,
        }
        impl KwsRunner {
            pub fn new() -> Self {
                Self { model: Model::new(S3Backend) }
            }
            pub fn input_len(&self) -> usize {
                INPUT_LEN
            }
            pub fn output_len(&self) -> usize {
                OUTPUT_LEN
            }
            pub fn scratch_len(&self) -> usize {
                SCRATCH_LEN
            }
            pub fn run(&mut self, input: &[i8], output: &mut [i8], scratch: &mut [u8]) -> bool {
                let input_arr: &[i8; INPUT_LEN] = match input.try_into() {
                    Ok(a) => a,
                    Err(_) => return false,
                };
                let output_arr: &mut [i8; OUTPUT_LEN] = match output.try_into() {
                    Ok(b) => b,
                    Err(_) => return false,
                };
                self.model.predict_with_scratch(input_arr, output_arr, scratch).is_ok()
            }
        }
    }

    mod model_anomaly_unfused {
        use super::*;
        #[model_unfused("../models/zoo/anomaly_detect/anomaly_detect_int8.tflite")]
        pub struct AnomalyModel;
        pub struct AnomalyRunner {
            model: Model<S3Backend>,
        }
        impl AnomalyRunner {
            pub fn new() -> Self {
                Self { model: Model::new(S3Backend) }
            }
            pub fn input_len(&self) -> usize {
                INPUT_LEN
            }
            pub fn output_len(&self) -> usize {
                OUTPUT_LEN
            }
            pub fn scratch_len(&self) -> usize {
                SCRATCH_LEN
            }
            pub fn run(&mut self, input: &[i8], output: &mut [i8], scratch: &mut [u8]) -> bool {
                let input_arr: &[i8; INPUT_LEN] = match input.try_into() {
                    Ok(a) => a,
                    Err(_) => return false,
                };
                let output_arr: &mut [i8; OUTPUT_LEN] = match output.try_into() {
                    Ok(b) => b,
                    Err(_) => return false,
                };
                self.model.predict_with_scratch(input_arr, output_arr, scratch).is_ok()
            }
        }
    }

    /// An unfused-arm zoo-model runner (closure-capable models only).
    pub enum ZooUnfusedRunner {
        Sine(model_sine_unfused::SineRunner),
        HelloWorld(model_hello_world_unfused::HelloWorldRunner),
        Kws(model_kws_unfused::KwsRunner),
        Anomaly(model_anomaly_unfused::AnomalyRunner),
    }

    impl ZooUnfusedRunner {
        pub fn input_len(&self) -> usize {
            match self {
                ZooUnfusedRunner::Sine(r) => r.input_len(),
                ZooUnfusedRunner::HelloWorld(r) => r.input_len(),
                ZooUnfusedRunner::Kws(r) => r.input_len(),
                ZooUnfusedRunner::Anomaly(r) => r.input_len(),
            }
        }
        pub fn output_len(&self) -> usize {
            match self {
                ZooUnfusedRunner::Sine(r) => r.output_len(),
                ZooUnfusedRunner::HelloWorld(r) => r.output_len(),
                ZooUnfusedRunner::Kws(r) => r.output_len(),
                ZooUnfusedRunner::Anomaly(r) => r.output_len(),
            }
        }
        pub fn scratch_len(&self) -> usize {
            match self {
                ZooUnfusedRunner::Sine(r) => r.scratch_len(),
                ZooUnfusedRunner::HelloWorld(r) => r.scratch_len(),
                ZooUnfusedRunner::Kws(r) => r.scratch_len(),
                ZooUnfusedRunner::Anomaly(r) => r.scratch_len(),
            }
        }
    }

    impl ModelRunner for ZooUnfusedRunner {
        fn input_len(&self) -> usize {
            self.input_len()
        }
        fn output_len(&self) -> usize {
            self.output_len()
        }
        fn predict(&mut self, input: &[i8], output: &mut [i8], scratch: &mut [u8]) -> bool {
            match self {
                ZooUnfusedRunner::Sine(r) => r.run(input, output, scratch),
                ZooUnfusedRunner::HelloWorld(r) => r.run(input, output, scratch),
                ZooUnfusedRunner::Kws(r) => r.run(input, output, scratch),
                ZooUnfusedRunner::Anomaly(r) => r.run(input, output, scratch),
            }
        }
    }

    /// Construct the unfused-arm runner for a registry spec (closure-capable
    /// models only).  Panics on an unwired path — same wiring-error contract
    /// as `zoo_runners::zoo_runner_for`; the firmware's SKIP guards keep
    /// person_detect / mobilenet_v2 out of this function on this board.
    pub fn zoo_unfused_runner_for(spec: &ModelBenchSpec) -> ZooUnfusedRunner {
        match spec.path {
            "models/sine.tflite" => {
                ZooUnfusedRunner::Sine(model_sine_unfused::SineRunner::new())
            }
            "models/zoo/sine_regression/hello_world_int8.tflite" => {
                ZooUnfusedRunner::HelloWorld(model_hello_world_unfused::HelloWorldRunner::new())
            }
            "models/zoo/keyword_spotting/kws_micro_speech_int8.tflite" => {
                ZooUnfusedRunner::Kws(model_kws_unfused::KwsRunner::new())
            }
            "models/zoo/anomaly_detect/anomaly_detect_int8.tflite" => {
                ZooUnfusedRunner::Anomaly(model_anomaly_unfused::AnomalyRunner::new())
            }
            other => panic!("zoo_unfused_runner_for: no unfused runner wired for spec path '{other}'"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::{summarize, FakeClock};

    struct FakeRunner {
        input_len: usize,
        output_len: usize,
    }

    impl ModelRunner for FakeRunner {
        fn input_len(&self) -> usize {
            self.input_len
        }
        fn output_len(&self) -> usize {
            self.output_len
        }
        fn predict(&mut self, _input: &[i8], output: &mut [i8], _scratch: &mut [u8]) -> bool {
            output.fill(1);
            true
        }
    }

    #[test]
    fn registry_entries_are_sane() {
        let specs = model_bench_specs();
        assert!(!specs.is_empty());
        for s in specs {
            assert!(!s.path.is_empty());
            assert!(s.input_len > 0);
            assert!(s.output_len > 0);
            assert!(!s.source.is_empty());
            // Every documented bar must have a source citation.
            if s.reference_ms_tenths.is_some() {
                assert!(s.source.starts_with("plan T5.3") || s.source.starts_with("workspace"));
            }
        }
    }

    #[test]
    fn bars_are_the_plan_numbers() {
        // B2: 1294.5 ms single-core, NEVER 856 (dual-core) — hold-as-documented
        // (PSRAM-gated on this board, PROJECT_LOG.md:721).
        assert_eq!(MOBILENETV2_SPEC.reference_ms_tenths, Some(12945));
        assert_ne!(MOBILENETV2_SPEC.reference_ms_tenths, Some(8560));
        // KWS: RE-TIERED (T6.3) from 7 ms ESP-DL to the ESP-NN-relative
        // 4 ms target (< 1,059,889 cyc / 4 ms, T3.5b).
        assert_eq!(KWS_SPEC.reference_ms_tenths, Some(40));
        assert_ne!(KWS_SPEC.reference_ms_tenths, Some(70));
        // Internal speedup floors (×100 fixed point): 15.57× conv1x1 bar
        // (spec.rs:1681) + 10× vs-scalar floor (T3.0).
        assert_eq!(speedup_bars::CONV1X1_VS_SCALAR_X100, 1557);
        assert_eq!(speedup_bars::SIMD_VS_SCALAR_FLOOR_X100, 1000);
    }

    #[test]
    fn zoo_specs_carry_real_paths_and_lens() {
        // Todo 19: the dead placeholder paths are gone; every spec path is a
        // real workspace tflite (the same string the `#[model]` attribute
        // compiles, modulo the leading `../`).
        assert_eq!(
            KWS_SPEC.path,
            "models/zoo/keyword_spotting/kws_micro_speech_int8.tflite"
        );
        assert_eq!(KWS_SPEC.output_len, 4); // real tflite output (golden [i8; 4])
        assert_eq!(
            MOBILENETV2_SPEC.path,
            "models/zoo/mobilenetv2_cls/mobilenet_v2_1.0_224_int8.tflite"
        );
        assert_eq!(MOBILENETV2_SPEC.output_len, 1000); // real tflite output (golden [i8; 1000])
        assert_eq!(
            PERSON_DETECT_SPEC.path,
            "models/zoo/person_detect_vww/person_detect_int8.tflite"
        );
        assert_eq!(PERSON_DETECT_SPEC.input_len, 96 * 96 * 3);
        assert_eq!(PERSON_DETECT_SPEC.output_len, 2);
        assert_eq!(
            ANOMALY_SPEC.path,
            "models/zoo/anomaly_detect/anomaly_detect_int8.tflite"
        );
        assert_eq!(ANOMALY_SPEC.input_len, 640);
        assert_eq!(ANOMALY_SPEC.output_len, 640);
        assert_eq!(
            HELLO_WORLD_SPEC.path,
            "models/zoo/sine_regression/hello_world_int8.tflite"
        );
        assert_eq!(HELLO_WORLD_SPEC.input_len, 1);
        assert_eq!(HELLO_WORLD_SPEC.output_len, 1);
        // Registry covers all six zoo models + sine.
        assert_eq!(model_bench_specs().len(), 6);
    }

    #[test]
    fn carve_model_layout_aligns_and_fits() {
        let l = carve_model_layout(1960, 4, 1024, 64 * 1024).expect("fits");
        assert_eq!(l.input_off % 16, 0);
        assert_eq!(l.output_off % 16, 0);
        assert_eq!(l.scratch_off % 16, 0);
        assert_eq!(l.total % 16, 0);
        assert!(l.output_off >= 1960);
        assert!(l.scratch_off >= l.output_off + 4);
        // Too-small arena → None.
        assert!(carve_model_layout(1960, 4, 1024, 100).is_none());
        // Zero scratch (generated SCRATCH_LEN can be 0) is fine.
        let l0 = carve_model_layout(1, 1, 0, 64).expect("fits");
        assert!(l0.total <= 64);
    }

    #[test]
    fn carve_model_bufs_slices_are_disjoint() {
        let mut arena = [0u8; 4096];
        let mut bufs = carve_model_bufs(&mut arena, 1960, 4, 1024).expect("fits");
        fill_input_pattern(bufs.input);
        bufs.output[0] = 42;
        bufs.scratch[0] = 7;
        assert_eq!(bufs.input[0], 3);
        assert_eq!(bufs.output[0], 42);
        assert_eq!(bufs.scratch[0], 7);
        // Regions never overlap.
        assert!((bufs.output.as_ptr() as usize) >= (bufs.input.as_ptr() as usize) + 1960);
    }

    #[test]
    fn model_bench_records_warmup_plus_10() {
        let mut clock = FakeClock::new(100, 1_000_000); // 1 ms per read step
        let mut runner = FakeRunner {
            input_len: 1,
            output_len: 1,
        };
        let input = [0i8; 1];
        let mut output = [0i8; 1];
        let mut scratch = [0u8; 0];
        let log = run_model_bench(
            &mut clock,
            &mut runner,
            &input,
            &mut output,
            &mut scratch,
            &BenchmarkConfig::default(),
        );
        assert_eq!(log.len(), 10);
        let summary = summarize(&log).expect("log not empty");
        // The wall clock is sampled once per edge → 1 ms per run.
        assert_eq!(wall_ms_tenths(summary.median_wall_ns), 10);
        assert!(output.iter().all(|&v| v == 1));
    }

    #[test]
    fn bar_comparison_is_integer() {
        let log = RunLog::new();
        let _ = &log;
        // Wall 7.0 ms vs 7 ms bar → pass; 7.1 ms → fail.
        assert!(wall_ms_tenths(7_000_000) == 70);
        assert!(wall_ms_tenths(7_100_000) == 71);
    }
}
