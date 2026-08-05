// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! hematite-benchmarks binary.
//!
//! * **Device** (`xtensa-esp32s3-none-elf`): firmware entry point that runs
//!   the full benchmark suite (CCOUNT + RTT/defmt) on ESP32-S3 hardware.
//! * **Host** (any other target): prints the report *template* and the
//!   documented reference bars.  It never fabricates measurements — there is
//!   no hardware on the host.

#![cfg_attr(target_arch = "xtensa", no_std, no_main)]

// ── Device firmware (ESP32-S3, xtensa target) ─────────────────────────

/// ESP32-S3 entry point — runs the benchmark suite, never returns.
///
/// Flash via probe-rs / espflash; results stream over RTT (defmt) or, under
/// the `qemu` feature, over UART0.
///
/// `#[xtensa_lx_rt::entry]` generates the reset vector that calls this
/// function — this roots the Reset → main chain so the firmware survives
/// `--gc-sections` even when no `esp_hal::init` call pulls the rt machinery
/// (the qemu build bypasses init).
#[cfg(target_arch = "xtensa")]
#[xtensa_lx_rt::entry]
fn main() -> ! {
    hematite_benchmarks::firmware::run_benchmarks()
}

// ── Host placeholder ──────────────────────────────────────────────────

/// Host build: show the report template and reference bars.  No measured
/// numbers are printed — this is a deliverable, not a run.
#[cfg(not(target_arch = "xtensa"))]
fn main() {
    use hematite_benchmarks::model_bench::model_bench_specs;
    use hematite_benchmarks::report::{render_report, row_from_summary};
    use hematite_benchmarks::spec::kernel_specs;
    use hematite_benchmarks::timing::RunSummary;

    println!("hematite-benchmarks — report template (host build; NO measurements on this host)");
    println!();
    println!("Reference bars (plan T5.3, B2 — single-core):");
    println!("  MobileNetV2 224x224 : 1294.5 ms single-core (the 856 ms figure is DUAL-core — never the bar)");
    println!("  KWS keyword_spotting_v1 : 7 ms");
    println!();

    // Kernel table rows (zeroed summary → template columns).
    let zero = RunSummary {
        n: 0,
        min_cycles: 0,
        median_cycles: 0,
        min_wall_ns: 0,
        median_wall_ns: 0,
    };
    let rows = kernel_specs()
        .iter()
        .map(|spec| {
            row_from_summary(
                spec.name,
                spec.tier.label(),
                &zero,
                hematite_benchmarks::timing::CPU_HZ_240MHZ,
                None,
                None,
                spec.note,
            )
        })
        .collect::<Vec<_>>();
    println!("Per-kernel benchmarks (columns filled only by a device run):");
    print!("{}", render_report(&rows));

    println!();
    println!("Model-level registry (wired at T5.2 when the .tflite files land):");
    for spec in model_bench_specs() {
        let bar = spec
            .reference_ms_tenths
            .map(|t| format!("{}.{} ms", t / 10, t % 10))
            .unwrap_or_else(|| "—".to_string());
        println!("  {:<48} {}  bar={:<10} {}", spec.name, spec.tier.label(), bar, spec.path);
    }
    println!();
    println!("To run on hardware:");
    println!("  cargo run --release --target xtensa-esp32s3-none-elf -p hematite-benchmarks");
    println!("(requires the esp-rs/rust fork toolchain + an ESP32-S3; results over RTT/defmt)");
    println!("Bench-mode watchdog disable requires the explicit flag:");
    println!("  RUSTFLAGS='--cfg bench_watchdog_disabled' cargo xtensa-build -p hematite-benchmarks");
}
