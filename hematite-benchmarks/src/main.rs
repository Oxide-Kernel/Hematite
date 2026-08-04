//! hematite-benchmarks — ESP32-S3 device firmware for per-op CCOUNT benchmarks.
//!
//! Benchmarks run on physical ESP32-S3 hardware via probe-rs with RTT/defmt.
//! They measure cycle counts for every kernel and compare against ember-esp-nn
//! and ESP-DL baselines.
//!
//! Host builds produce only a placeholder binary so that `cargo check --workspace`
//! passes without the esp-rs/rust fork toolchain.

#![cfg_attr(target_arch = "xtensa", no_std, no_main)]

// ── Device firmware (ESP32-S3, xtensa target) ─────────────────────────

/// ESP32-S3 entry point — runs on hardware via probe-rs.
///
/// TODO(T5.3 / T5.3a): Wire in the RTT/defmt/CCOUNT benchmark harness.
/// The real harness per-op CCOUNT readings, dumps results over RTT,
/// and compares against ember-esp-nn + ESP-DL numbers.
#[cfg(target_arch = "xtensa")]
#[no_mangle]
pub extern "C" fn main() -> ! {
    loop {}
}

// ── Host placeholder (any non-xtensa target) ──────────────────────────

/// Placeholder binary so the workspace remains `cargo check`-able on the
/// host toolchain.  Real benchmarks require:
///   - ESP32-S3 hardware
///   - The esp-rs/rust fork toolchain (espup install)
#[cfg(not(target_arch = "xtensa"))]
fn main() {
    eprintln!(
        "hematite-benchmarks: benchmarks require ESP32-S3 hardware.\n\
         Install the esp-rs/rust fork: https://github.com/esp-rs/rust-build"
    );
}
