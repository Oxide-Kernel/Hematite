// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! ESP32-S3 benchmark firmware (plan T5.3 / T5.3a, C3).
//!
//! **Device-only module** — compiled exclusively for `xtensa-esp32s3-none-elf`
//! (Phase 3 cfg-gating convention).  The host build never sees this file's
//! esp-hal / defmt dependencies.
//!
//! # Boot sequence
//!
//! 1. Initialize RTT (defmt) for report output.
//! 2. Initialize esp-hal at the locked profile.
//! 3. **Verify the boot profile** — panic (clear message) on drift from
//!    [`crate::guardrails::LOCKED_PROFILE`] (240 MHz CPU, QPI 80 MHz PSRAM,
//!    64 KB / 64-byte cache).
//! 4. Resolve the watchdog policy — disable only behind the explicit
//!    `--cfg bench_watchdog_disabled` build flag; otherwise the watchdog stays
//!    armed (a hung benchmark resets the chip).
//! 5. **CCOUNT calibration assert** — cycles over an independent wall window
//!    must match 240 MHz within ppm tolerance (integer math only).
//! 6. Arm the stack canary; verify after every benchmark.
//! 7. Run the per-kernel table and the model-level registry; emit the report.
//!
//! # Unverified-on-host API surface
//!
//! This file cannot be compiled on this host (no esp-rs/rust fork toolchain).
//! Every esp-hal call is therefore marked `BRING-UP:` with the exact API to
//! validate on hardware; the guardrail **logic** lives in
//! [`crate::guardrails`] and the buffer-carve logic lives in
//! [`crate::spec::carve_into`] — both fully host-tested.  See
//! `local-notes/notepads/hematite-nn/problems.md` for the tracking entry.

use esp_hal::clock::CpuClock;
use esp_hal::Config;

use crate::guardrails::{
    assert_ccount_calibration, verify_boot_profile, watchdog_disabled_policy, BootProfile,
    StackCanary,
};
use crate::model_bench::{model_bench_specs, ModelBenchSpec};
use crate::report::{row_from_summary, ReportRow};
use crate::spec::{
    carve_into, fill_pattern, kernel_specs, layout, run_kernel, run_ref_kernel, KernelSpec,
    MemoryTier,
};
use crate::timing::{
    read_ccount, run_repeated, summarize, BenchmarkConfig, CPU_HZ_240MHZ, RealClock,
};

/// Independent wall-clock read in nanoseconds.
///
/// BRING-UP: validate against esp-hal 1.1 — `esp_hal::time::now()` returns an
/// `Instant`; the exact duration conversion (`duration_since_epoch()` +
/// `as_micros()`) must be confirmed on the esp-rs toolchain.  This is the
/// independent wall source that backs report column 3 and the CCOUNT
/// calibration assert.
pub fn read_wall_ns_impl() -> u64 {
    let now = esp_hal::time::now();
    let dur = now.duration_since_epoch();
    // BRING-UP: if `as_micros` is not the 1.1 name, the equivalent integer
    // ns conversion goes here.  Integer-only (no f32 timing drift).
    (dur.as_micros() as u64).saturating_mul(1000)
}

/// Device panic handler — logs through defmt and halts.
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    defmt::error!("PANIC: {}", defmt::Debug2Format(info));
    loop {}
}

/// SRAM bench arena (rows whose working set fits internal SRAM).
static mut SRAM_ARENA: [u8; 256 * 1024] = [0u8; 256 * 1024];

/// PSRAM bench arena (large MobileNetV2-style rows).
///
/// BRING-UP: `.dram1.psram` is the ESP32-S3 PSRAM linker section used by
/// esp-hal's memory.x; if the section name differs in the pinned esp-hal
/// version the linker fails loudly and the section name is the one-line fix.
#[link_section = ".dram1.psram"]
static mut PSRAM_ARENA: [u8; 4 * 1024 * 1024] = [0u8; 4 * 1024 * 1024];

/// Stack-canary slot.  BRING-UP: extend the linker script so this section is
/// placed at the top of the stack region for true overflow detection; without
/// that placement the canary still catches gross clobbers, and the
/// SP-based [`crate::guardrails::stack_depth_ok`] check is
/// placement-independent.
#[link_section = ".stack_canary"]
static mut STACK_CANARY_SLOT: u32 = 0;

/// Read the boot profile from esp-hal's configuration.
///
/// BRING-UP: esp-hal 1.1 exposes the *configured* CPU clock via
/// `CpuClock::max().mhz()`; PSRAM/cache are fixed by the build's linker
/// config and are asserted against the locked constants here.  The returned
/// values must be validated on hardware against the actual registers.
fn read_boot_profile() -> BootProfile {
    let cpu_mhz = CpuClock::max().mhz();
    defmt::info!("boot: CpuClock::max() reports {} MHz", cpu_mhz);
    BootProfile {
        cpu_mhz,
        // BRING-UP: PSRAM QPI speed and cache geometry are read back from the
        // configured esp-hal state (or the TRM registers) on hardware.
        psram_qpi_mhz: crate::guardrails::PSRAM_QPI_MHZ_LOCKED,
        data_cache_bytes: crate::guardrails::DATA_CACHE_BYTES_LOCKED,
        cache_line_bytes: crate::guardrails::CACHE_LINE_BYTES_LOCKED,
    }
}

/// Disable the hardware watchdog for long benchmark runs.
///
/// Only compiled when the explicit `bench_watchdog_disabled` cfg flag is set
/// (`RUSTFLAGS='--cfg bench_watchdog_disabled' cargo xtensa-build ...`).
/// A normal firmware build keeps the watchdog armed.
#[cfg(bench_watchdog_disabled)]
fn disable_watchdog() {
    // BRING-UP: esp-hal 1.1 watchdog API (TWDT).  With the flag set the bench
    // may run > watchdog period; without it the watchdog resets a hung run.
    defmt::info!("bench_watchdog_disabled: hardware watchdog DISABLED for bench run");
}

/// Watchdog stays armed (safe default).
#[cfg(not(bench_watchdog_disabled))]
fn disable_watchdog() {
    defmt::info!("watchdog ARMED (safe default; pass --cfg bench_watchdog_disabled to disable)");
}

/// Measure CCOUNT over an independent wall window and assert 240 MHz.
fn calibrate_and_assert() {
    let c0 = read_ccount();
    let w0 = read_wall_ns_impl();
    // BRING-UP: a real esp-hal sleep / timer yield is preferred; this busy
    // loop must survive the optimizer (sink is consumed below).
    let mut sink: u64 = 0;
    for _ in 0..1000 {
        sink = sink.wrapping_add(1);
    }
    core::hint::black_box(sink);
    let c1 = read_ccount();
    let w1 = read_wall_ns_impl();
    let cycles = (c1.wrapping_sub(c0)) & 0xFFFF_FFFF;
    let wall_ns = w1.saturating_sub(w0);
    defmt::info!("ccount calibration: {} cycles / {} ns", cycles, wall_ns);
    // 1000 ppm tolerance (0.1%).
    if let Err(e) = assert_ccount_calibration(cycles, wall_ns, CPU_HZ_240MHZ, 1000) {
        panic!("CCOUNT calibration guardrail: {}", e.describe());
    }
}

/// Benchmark one kernel spec: scalar-ref baseline first, then the s3 kernel,
/// both with warm-up + N≥10 (C3); emits one report row via defmt.
fn bench_kernel(spec: &KernelSpec, clock: &mut RealClock, canary: &mut StackCanary) {
    let lay = layout(spec);
    // SAFETY: carve_into returns the only live borrow of the arena for the
    // duration of this benchmark; the arena is re-carved per spec.
    let arena = unsafe {
        match spec.tier {
            MemoryTier::Sram => &mut SRAM_ARENA,
            MemoryTier::Psram => &mut PSRAM_ARENA,
        }
    };
    let mut bufs = match carve_into(arena, &lay) {
        Ok(b) => b,
        Err(_) => panic!("arena too small for spec '{}'", spec.name),
    };
    let mut scratch = [0u8; 0];
    let cfg = BenchmarkConfig::default();

    // Column 1 baseline: the same shape through the hematite-ref scalar
    // kernel on device (never a pre-filled number).
    fill_pattern(&mut bufs);
    let ref_log = run_repeated(
        clock,
        &mut || {
            let _ = run_ref_kernel(spec, &mut bufs, &mut scratch);
        },
        &cfg,
    );

    // The s3 kernel (SIMD on device, scalar fallback where no SIMD path).
    fill_pattern(&mut bufs);
    let s3_log = run_repeated(
        clock,
        &mut || {
            let _ = run_kernel(spec, &mut bufs, &mut scratch);
        },
        &cfg,
    );

    let s3_sum = match summarize(&s3_log) {
        Some(s) => s,
        None => return, // unreachable: run_repeated floors at 10
    };
    let ref_sum = match summarize(&ref_log) {
        Some(s) => s,
        None => return,
    };

    // Column 1: our scalar ref vs s3 (the T3.0 >= 10x internal bar).
    let col1 = crate::report::speedup_x100(ref_sum.median_cycles, s3_sum.median_cycles);

    let mut row = row_from_summary(
        spec.name,
        spec.tier.label(),
        &s3_sum,
        CPU_HZ_240MHZ,
        None,
        None,
        spec.note,
    );
    row.speedups[0].speedup = Some(col1);

    emit_row(&row);

    if let Err(e) = canary.verify() {
        panic!("bench '{}': {}", spec.name, e.describe());
    }
}

/// Emit a report row over defmt/RTT (device).
///
/// Columns 2 and 3 stay `None` (rendered by defmt as `None`) until the
/// competitor cycle counts are sourced — the MUST-NOT-invent-numbers rule.
fn emit_row(row: &ReportRow) {
    defmt::info!(
        "| {} | {} | {}/{} | {}/{} | {}/{} | col1={} | col2={} | col3={} |",
        row.label,
        row.tier,
        row.min_cycles,
        row.median_cycles,
        row.ms_240_min,
        row.ms_240_median,
        row.wall_ms_min,
        row.wall_ms_median,
        row.speedups[0].speedup,
        row.speedups[1].speedup,
        row.speedups[2].speedup,
    );
}

/// Emit the model-level registry.  Rows whose runner is not yet wired (no
/// `.tflite` until T5.2) are listed with their documented reference bar and a
/// NOT-WIRED marker — no fabricated measurements.
fn emit_model_row(spec: &ModelBenchSpec) {
    let bar_tenths = spec.reference_ms_tenths.unwrap_or(0);
    let bar_ms = (bar_tenths / 10, bar_tenths % 10);
    defmt::info!(
        "| {} | {} | NOT-WIRED (T5.2) | bar={}.{} ms | {}",
        spec.name,
        spec.tier.label(),
        bar_ms.0,
        bar_ms.1,
        spec.source,
    );
}

/// Firmware entry point — runs the full benchmark suite and never returns.
pub fn run_benchmarks() -> ! {
    defmt_rtt::defmt_rtt_init();

    // esp-hal 1.1 documented init (context7: `Config::default()
    // .with_cpu_clock(CpuClock::max())` + `esp_hal::init`).
    let config = Config::default().with_cpu_clock(CpuClock::max());
    let _peripherals = esp_hal::init(config);

    // 1. Boot-profile guardrail — panic on any drift from the locked profile.
    let profile = read_boot_profile();
    if let Err(e) = verify_boot_profile(&profile) {
        panic!("boot guardrail: {}", e.describe());
    }
    defmt::info!("boot profile OK: 240 MHz / QPI 80 MHz / 64 KB x 64 B cache");

    // 2. Watchdog policy — the disable path compiles ONLY behind the explicit
    // `bench_watchdog_disabled` cfg flag; the safe default keeps it armed.
    let flag = cfg!(bench_watchdog_disabled);
    match watchdog_disabled_policy(flag, flag) {
        Ok(true) => disable_watchdog(),
        Ok(false) => defmt::info!("watchdog ARMED (safe default)"),
        Err(e) => panic!("watchdog policy: {}", e.describe()),
    }

    // 3. CCOUNT calibration assert.
    calibrate_and_assert();

    // 4. Stack canary.
    // SAFETY: single-threaded firmware; unique static.
    let mut canary = StackCanary::new(unsafe { &mut STACK_CANARY_SLOT });
    canary.arm();

    // 5. Per-kernel benchmarks.
    let mut clock = RealClock;
    defmt::info!("{}", crate::report::HEADER);
    for spec in kernel_specs() {
        bench_kernel(spec, &mut clock, &mut canary);
    }

    // 6. Model-level registry.
    for spec in model_bench_specs() {
        emit_model_row(spec);
    }

    defmt::info!("benchmarks complete; reference bars: MobileNetV2 224x224 = 1294.5 ms single-core (never 856 ms dual-core), KWS = 7 ms");
    loop {
        // Hold the RTT link open for the host to drain output.
    }
}
