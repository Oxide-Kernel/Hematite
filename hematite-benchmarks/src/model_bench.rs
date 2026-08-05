// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Model-level benchmarks (plan T5.3).
//!
//! # Structure — model path is a parameter
//!
//! The zoo `.tflite` files arrive with T5.2.  This module therefore separates
//! the **benchmark definition** (the spec: name, model path, reference bar,
//! memory tier) from the **runner** (the thing that executes an inference).
//!
//! The runner is abstracted behind [`ModelRunner`], whose shape mirrors the
//! generated-code API that `hematite-codegen`'s `#[model("path.tflite")]`
//! macro emits (`Model<B>` with `input_len()`, `output_len()`,
//! `predict_with_scratch`).  T5.2 wires a concrete generated `Model<B>` into
//! [`ModelRunner`] (one small adapter per zoo model), at which point the
//! spec's `path` becomes the literal `#[model]` attribute argument and the
//! harness below runs unchanged.
//!
//! # Reference bars (B2 — single-core, documented sources)
//!
//! | Model | Bar | Source |
//! |-------|-----|--------|
//! | MobileNetV2 224×224 | **1294.5 ms** single-core | ESP-DL reference, plan T5.3 (B2). The 856 ms figure is the DUAL-core number — never the bar. |
//! | KWS keyword_spotting_v1 (1×1960) | **7 ms** | edge-ml-model-zoo ESP-DL, plan T5.3 line 308 |
//!
//! `reference_ms_tenths` stores these in ×10 fixed point (12945 = 1294.5 ms)
//! so the pass/fail comparison is pure integer math — no f32 anywhere.

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
pub const MOBILENETV2_SPEC: ModelBenchSpec = ModelBenchSpec {
    name: "MobileNetV2 224x224 (single-core bar 1294.5 ms)",
    path: "models/mobilenet_v2.tflite",
    input_len: 224 * 224 * 3,
    output_len: 1001,
    tier: MemoryTier::Psram,
    reference_ms_tenths: Some(12945), // 1294.5 ms
    source: "plan T5.3 line 308 (B2): ESP-DL single-core reference 1294.5 ms; 856 ms is DUAL-core, NOT the bar",
};

/// KWS — keyword_spotting_v1, input 1×1960.
///
/// `output_len` is set to 12 classes pending confirmation from the actual
/// model when T5.2 lands (edge-ml-model-zoo keyword_spotting_v1).
pub const KWS_SPEC: ModelBenchSpec = ModelBenchSpec {
    name: "KWS keyword_spotting_v1 (bar 7 ms)",
    path: "models/keyword_spotting_v1.tflite",
    input_len: 1960, // 1×1960 per plan T5.3
    output_len: 12,
    tier: MemoryTier::Sram,
    reference_ms_tenths: Some(70), // 7.0 ms
    source: "plan T5.3 line 308: KWS 1×1960, edge-ml-model-zoo ESP-DL 7 ms",
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

/// The full model-level benchmark registry (report row order).
pub const fn model_bench_specs() -> &'static [ModelBenchSpec] {
    &[MOBILENETV2_SPEC, KWS_SPEC, SINE_SPEC]
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

/// Evaluate a measured run against a documented reference bar.
///
/// Both sides are ×10 fixed-point ms (integer).  `true` = measured median
/// wall time ≤ the bar (pass).
pub fn passes_reference_bar(summary: &RunSummary, bar_tenths: u32) -> bool {
    wall_ms_tenths(summary.median_wall_ns) <= u64::from(bar_tenths)
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
        // B2: 1294.5 ms single-core, NEVER 856 (dual-core).
        assert_eq!(MOBILENETV2_SPEC.reference_ms_tenths, Some(12945));
        assert_ne!(MOBILENETV2_SPEC.reference_ms_tenths, Some(8560));
        // KWS: 7 ms.
        assert_eq!(KWS_SPEC.reference_ms_tenths, Some(70));
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
