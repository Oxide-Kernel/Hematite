// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Real-model zoo benchmarks: Hematite (`Model<S3Backend>` via
//! `predict_with_arena`) vs the scalar `RefBackend`, using the real int8
//! weights and real golden inputs from the `.tflite` zoo models.
//!
//! Each model is compiled once via `#[model]` in [`crate::model_validation`]
//! and reused here (avoids re-expanding the multi-MB weight consts). The
//! benchmark runs both backends through the generated `predict_with_arena`
//! entry point, which carves all intermediates from a caller-provided arena
//! instead of the stack — so even the large models run on the device.
//!
//! Memory tiers: a model's region (aligned input copy + `ARENA_LEN` arena +
//! `SCRATCH_NEED` SIMD scratch) is carved from SRAM when it fits, else PSRAM,
//! via [`crate::firmware::arena_for`]. The reported tier is the actual one.
//!
//! `predict_with_arena` returns [`KernelError::ScratchTooSmall`] loudly if the
//! scratch region is too small for SIMD — a single warmup call validates this
//! and panics with an honest message rather than silently falling back.

use hematite_ref::RefBackend;
use hematite_s3::backend::S3Backend;

use crate::firmware::{arena_for, emit_row, fnv1a, firmware_log, psram_available, SRAM_ARENA_BYTES};
use crate::guardrails::StackCanary;
use crate::report::{row_from_summary, speedup_x100};
use crate::timing::{run_repeated, summarize, BenchmarkConfig, RealClock, CPU_HZ_240MHZ};

/// Per-model SIMD scratch requirement (the s3 conv kernels stage a padded
/// copy of the input in scratch: `padded_h * padded_w * padded_c`; channel
/// padding rounds `in_c` up to a multiple of 16).
const SCRATCH_NEED_SMALL: usize = 65536;
/// person_detect: first conv 96×96×3→96×96×8 stages a padded 98×98×16
/// (≈150 KB) input copy.
const SCRATCH_NEED_PERSON: usize = 262144;
/// mobilenet_v2: first conv 224×224×3→112×112×32 stages a padded 225×225×16
/// (≈810 KB) input copy; later SAME-padded convs need 200–420 KB.
const SCRATCH_NEED_MOBILENET: usize = 1048576;

/// Benchmark one zoo model on the device, reporting cycles for the S3 and
/// Ref backends plus the fnv1a of the final output.
macro_rules! bench_zoo_model {
    ($fn_name:ident, $mod_path:ident, $label:literal, $scratch:expr) => {
        pub fn $fn_name(clock: &mut RealClock, canary: &mut StackCanary) {
            use crate::model_validation::$mod_path as m;

            let input_bytes = m::INPUT_LEN;
            let arena_len = m::ARENA_LEN;
            let scratch_need: usize = $scratch;

            // 16-aligned carve plan within one region.
            let arena_off = (input_bytes + 15) & !15;
            let scratch_off = arena_off + arena_len; // ARENA_LEN is 16-aligned
            let total = scratch_off + scratch_need;

            // Models whose region exceeds SRAM need PSRAM. On boards without
            // it, skip honestly rather than panicking in arena_for.
            if total > SRAM_ARENA_BYTES && !psram_available() {
                firmware_log!(
                    "zoo {}: SKIP (region {} bytes requires PSRAM, not present on this board)",
                    $label,
                    total,
                );
                return;
            }

            let region = arena_for(total);
            let tier: &'static str = if total <= SRAM_ARENA_BYTES { "SRAM" } else { "PSRAM" };

            // Copy the real golden input, then re-view the sub-regions.
            let input_src =
                unsafe { core::slice::from_raw_parts(m::golden::INPUT_DATA.as_ptr() as *const u8, input_bytes) };
            region[..input_bytes].copy_from_slice(input_src);
            let input: &[i8; m::INPUT_LEN] =
                unsafe { &*(region.as_ptr() as *const [i8; m::INPUT_LEN]) };
            let arena: &mut [i8] = unsafe {
                core::slice::from_raw_parts_mut(region[arena_off..].as_mut_ptr() as *mut i8, arena_len)
            };
            let scratch: &mut [u8] = &mut region[scratch_off..scratch_off + scratch_need];
            let mut output = [0i8; m::OUTPUT_LEN];

            let cfg = BenchmarkConfig::default();

            // S3 backend (the claim being benchmarked).
            let s3 = m::Model::<S3Backend>::new(S3Backend);
            s3.predict_with_arena(input, &mut output, arena, scratch)
                .unwrap_or_else(|e| panic!("zoo {}: S3 predict_with_arena: {e:?}", $label));
            let s3_fnv = fnv1a(&output);
            let s3_log = run_repeated(clock, &mut || {
                let _ = s3.predict_with_arena(input, &mut output, arena, scratch);
            }, &cfg);
            let s3_sum = match summarize(&s3_log) {
                Some(s) => s,
                None => return,
            };

            // Scalar reference backend (col1 baseline).
            let r = m::Model::<RefBackend>::new(RefBackend);
            r.predict_with_arena(input, &mut output, arena, scratch)
                .unwrap_or_else(|e| panic!("zoo {}: Ref predict_with_arena: {e:?}", $label));
            let ref_fnv = fnv1a(&output);
            let ref_log = run_repeated(clock, &mut || {
                let _ = r.predict_with_arena(input, &mut output, arena, scratch);
            }, &cfg);
            let ref_sum = match summarize(&ref_log) {
                Some(s) => s,
                None => return,
            };

            let col1 = speedup_x100(ref_sum.median_cycles, s3_sum.median_cycles);
            let mut row = row_from_summary($label, tier, &s3_sum, CPU_HZ_240MHZ, None, None, "zoo");
            row.speedups[0].speedup = Some(col1);
            emit_row(&row, ref_fnv, s3_fnv);
            firmware_log!(
                "zoo {}: out_fnv(ref/s3)=0x{:08x}/0x{:08x} (equal={})",
                $label,
                ref_fnv,
                s3_fnv,
                ref_fnv == s3_fnv,
            );

            if let Err(e) = canary.verify() {
                panic!("zoo {}: stack canary corrupted: {}", $label, e.describe());
            }
        }
    };
}

bench_zoo_model!(
    bench_sine_regression,
    model_sine,
    "sine_regression_int8",
    SCRATCH_NEED_SMALL
);
bench_zoo_model!(
    bench_hello_world,
    model_hello_world,
    "hello_world_int8",
    SCRATCH_NEED_SMALL
);
bench_zoo_model!(
    bench_keyword_spotting,
    model_kws,
    "kws_micro_speech_int8",
    SCRATCH_NEED_SMALL
);
bench_zoo_model!(
    bench_anomaly_detect,
    model_anomaly,
    "anomaly_detect_int8",
    SCRATCH_NEED_SMALL
);
bench_zoo_model!(
    bench_person_detect,
    model_person_detect,
    "person_detect_vww_int8",
    SCRATCH_NEED_PERSON
);
bench_zoo_model!(
    bench_mobilenet_v2,
    model_mobilenet,
    "mobilenet_v2_1.0_224_int8",
    SCRATCH_NEED_MOBILENET
);

/// Run all zoo-model benchmarks. Real board only (large models need PSRAM).
pub fn run_zoo_benches(clock: &mut RealClock, canary: &mut StackCanary) {
    firmware_log!("=== ZOO MODEL BENCH (real weights + golden inputs) ===");
    bench_sine_regression(clock, canary);
    bench_hello_world(clock, canary);
    bench_keyword_spotting(clock, canary);
    bench_anomaly_detect(clock, canary);
    bench_person_detect(clock, canary);
    bench_mobilenet_v2(clock, canary);
    firmware_log!("=== ZOO MODEL BENCH DONE ===");
}
