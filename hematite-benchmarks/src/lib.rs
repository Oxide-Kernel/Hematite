// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! hematite-benchmarks — ESP32-S3 hardware benchmark suite (plan T5.3 / T5.3a).
//!
//! Crate layout (Phase 3 cfg-gating convention — device code is gated behind
//! `cfg(target_arch = "xtensa")`, everything else is host-compilable and
//! host-tested):
//!
//! * [`timing`] — CCOUNT/wall-clock timing methodology (warm-up, N ≥ 10,
//!   min + median, integer-only conversions).
//! * [`spec`] — per-kernel benchmark table (ember-esp-nn + ESP-DL/MobileNetV2
//!   shapes), buffer layouts, and the s3 dispatch (the real public ABI).
//! * [`model_bench`] — model-level benchmark registry + harness (model path
//!   is a parameter; zoo `.tflite` files land with T5.2).
//! * [`report`] — three-column raw format (cycles / ms@240 MHz / wall-ms),
//!   three speedup columns, SRAM/PSRAM tier labels, reference bars.
//! * [`guardrails`] — C3 methodology guardrails (boot profile, CCOUNT
//!   calibration, stack canary, watchdog policy) — pure, host-tested logic.
//! * [`firmware`] — device-only ESP32-S3 firmware (esp-hal + defmt/RTT).

#![cfg_attr(target_arch = "xtensa", no_std)]
// Xtensa inline asm (CCOUNT read in timing.rs) needs the experimental-arch
// gate on the esp-rs fork toolchain — same gate as hematite-s3.
#![cfg_attr(target_arch = "xtensa", feature(asm_experimental_arch))]

pub mod guardrails;
pub mod model_bench;
pub mod model_cnn;
pub mod model_mv2;
pub mod model_mv2real;
pub mod report;
pub mod spec;
pub mod timing;

#[cfg(all(target_arch = "xtensa", feature = "model-validation"))]
pub mod model_validation;

#[cfg(all(target_arch = "xtensa", feature = "model-validation"))]
pub mod simd_validation;

#[cfg(target_arch = "xtensa")]
pub mod firmware;
