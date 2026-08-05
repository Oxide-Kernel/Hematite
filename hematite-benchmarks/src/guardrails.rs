// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Benchmark-firmware methodology guardrails (plan T5.3a, C3).
//!
//! All guardrail **logic** is pure, `no_std`-safe and host-testable — the
//! device firmware (`crate::firmware`) feeds real esp-hal values in and
//! panics on failure.  Cycle counts are meaningless without a locked,
//! documented measurement protocol; these checks enforce that protocol.
//!
//! # Locked boot profile
//!
//! | Parameter | Locked value |
//! |-----------|--------------|
//! | CPU frequency | 240 MHz |
//! | PSRAM | QPI 80 MHz |
//! | Data cache | 64 KB, 64-byte line |
//!
//! The firmware refuses to run (panics with a clear message) if the actual
//! profile drifts from [`LOCKED_PROFILE`].
//!
//! # Timing methodology guardrails
//!
//! * **CCOUNT calibration assert** — before any measurement, the firmware
//!   compares CCOUNT ticks against an independent wall period and verifies
//!   the ratio is 240 MHz within a tolerance (integer parts-per-million math).
//! * **No f32 timing drift** — every conversion in the timing path
//!   (`crate::timing`, `crate::report`) is integer.  This module contains
//!   zero floating point.

/// Expected CPU frequency in MHz.
pub const CPU_MHZ_LOCKED: u32 = 240;
/// Expected PSRAM interface speed in MHz (QPI mode).
pub const PSRAM_QPI_MHZ_LOCKED: u32 = 80;
/// Expected data-cache size in bytes.
pub const DATA_CACHE_BYTES_LOCKED: u32 = 64 * 1024;
/// Expected cache line size in bytes.
pub const CACHE_LINE_BYTES_LOCKED: u32 = 64;

/// The locked boot profile the firmware must run under (C3).
pub const LOCKED_PROFILE: BootProfile = BootProfile {
    cpu_mhz: CPU_MHZ_LOCKED,
    psram_qpi_mhz: PSRAM_QPI_MHZ_LOCKED,
    data_cache_bytes: DATA_CACHE_BYTES_LOCKED,
    cache_line_bytes: CACHE_LINE_BYTES_LOCKED,
};

/// Actual clock / cache configuration read at firmware boot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BootProfile {
    /// Configured CPU frequency in MHz.
    pub cpu_mhz: u32,
    /// Configured PSRAM interface speed in MHz (QPI mode).
    pub psram_qpi_mhz: u32,
    /// Configured data-cache size in bytes.
    pub data_cache_bytes: u32,
    /// Configured cache line size in bytes.
    pub cache_line_bytes: u32,
}

/// Guardrail failures — every variant carries the actual and expected value
/// so the panic message is actionable on hardware.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuardrailError {
    /// CPU is not running at the locked 240 MHz.
    CpuClockMismatch { actual_mhz: u32, expected_mhz: u32 },
    /// PSRAM is not running at QPI 80 MHz.
    PsramMismatch { actual_qpi_mhz: u32, expected_qpi_mhz: u32 },
    /// Data cache size differs from 64 KB.
    DataCacheMismatch { actual_bytes: u32, expected_bytes: u32 },
    /// Cache line size differs from 64 bytes.
    CacheLineMismatch { actual_bytes: u32, expected_bytes: u32 },
    /// CCOUNT ran at a rate outside the ppm tolerance vs the wall clock.
    CcountCalibrationFailed { measured_hz: u64, expected_hz: u64, tolerance_ppm: u64 },
    /// The stack canary word was overwritten (stack overflow suspected).
    StackCanaryViolated,
    /// The benchmark consumed more stack than the budgeted region.
    StackDepthExceeded { consumed_bytes: usize, budget_bytes: usize },
    /// The watchdog policy was requested but the firmware build does not
    /// carry the explicit `bench_watchdog_disabled` cfg flag.
    WatchdogPolicyNotEnabled,
}

impl GuardrailError {
    /// Human-readable description used in the device panic message.
    pub const fn describe(&self) -> &'static str {
        match self {
            GuardrailError::CpuClockMismatch { .. } => {
                "CPU frequency mismatch: benchmark profile requires 240 MHz"
            }
            GuardrailError::PsramMismatch { .. } => {
                "PSRAM config mismatch: benchmark profile requires QPI 80 MHz"
            }
            GuardrailError::DataCacheMismatch { .. } => {
                "data-cache config mismatch: benchmark profile requires 64 KB"
            }
            GuardrailError::CacheLineMismatch { .. } => {
                "cache-line config mismatch: benchmark profile requires 64-byte lines"
            }
            GuardrailError::CcountCalibrationFailed { .. } => {
                "CCOUNT calibration failed: ticks/sec diverges from 240 MHz"
            }
            GuardrailError::StackCanaryViolated => "stack canary overwritten — overflow suspected",
            GuardrailError::StackDepthExceeded { .. } => "benchmark exceeded its stack budget",
            GuardrailError::WatchdogPolicyNotEnabled => {
                "bench-mode watchdog disable requires --cfg bench_watchdog_disabled"
            }
        }
    }
}

/// Verify the actual boot profile matches the locked profile exactly.
///
/// The firmware panics with the returned error's description; the firmware
/// never starts measuring on a drifted profile (C3).
pub fn verify_boot_profile(actual: &BootProfile) -> Result<(), GuardrailError> {
    if actual.cpu_mhz != LOCKED_PROFILE.cpu_mhz {
        return Err(GuardrailError::CpuClockMismatch {
            actual_mhz: actual.cpu_mhz,
            expected_mhz: LOCKED_PROFILE.cpu_mhz,
        });
    }
    if actual.psram_qpi_mhz != LOCKED_PROFILE.psram_qpi_mhz {
        return Err(GuardrailError::PsramMismatch {
            actual_qpi_mhz: actual.psram_qpi_mhz,
            expected_qpi_mhz: LOCKED_PROFILE.psram_qpi_mhz,
        });
    }
    if actual.data_cache_bytes != LOCKED_PROFILE.data_cache_bytes {
        return Err(GuardrailError::DataCacheMismatch {
            actual_bytes: actual.data_cache_bytes,
            expected_bytes: LOCKED_PROFILE.data_cache_bytes,
        });
    }
    if actual.cache_line_bytes != LOCKED_PROFILE.cache_line_bytes {
        return Err(GuardrailError::CacheLineMismatch {
            actual_bytes: actual.cache_line_bytes,
            expected_bytes: LOCKED_PROFILE.cache_line_bytes,
        });
    }
    Ok(())
}

/// Assert that CCOUNT ticks at the expected rate.
///
/// `cycles` were counted over a `wall_ns` window measured by an independent
/// clock.  The measured rate is `cycles * 1e9 / wall_ns`; its error vs
/// `expected_hz` (in ppm) must stay within `tolerance_ppm`.  **Integer-only
/// math** — the ppm comparison is done in fixed point (this is the guardrail
/// against f32 timing drift).
pub fn assert_ccount_calibration(
    cycles: u64,
    wall_ns: u64,
    expected_hz: u64,
    tolerance_ppm: u64,
) -> Result<(), GuardrailError> {
    if wall_ns == 0 || expected_hz == 0 {
        // A zero-length window or zero expected rate is a broken measurement,
        // not a calibration.
        return Err(GuardrailError::CcountCalibrationFailed {
            measured_hz: 0,
            expected_hz,
            tolerance_ppm,
        });
    }
    let measured_hz = cycles.saturating_mul(1_000_000_000) / wall_ns;
    let err_hz = measured_hz.abs_diff(expected_hz);
    // err_ppm = err_hz * 1e6 / expected_hz — saturating so an overflow is
    // treated as "way out of tolerance" (correct direction).
    let err_ppm = err_hz.saturating_mul(1_000_000) / expected_hz;
    if err_ppm <= tolerance_ppm {
        Ok(())
    } else {
        Err(GuardrailError::CcountCalibrationFailed {
            measured_hz,
            expected_hz,
            tolerance_ppm,
        })
    }
}

/// Stack-canary word pattern.
pub const CANARY_PATTERN: u32 = 0x5A5A_5A5A;

/// Stack canary — a word placed at the edge of the benchmark stack region.
///
/// On device the slot lives in a dedicated linker section adjacent to the
/// stack guard area (see `crate::firmware`); the firmware arms it before the
/// benchmark loop and verifies it after every run.  An overwritten canary
/// means a kernel clobbered memory past the stack budget (stack overflow).
pub struct StackCanary {
    slot: &'static mut u32,
}

impl StackCanary {
    /// Wrap a canary storage slot.
    pub fn new(slot: &'static mut u32) -> Self {
        StackCanary { slot }
    }

    /// Write the canary pattern into the slot.
    pub fn arm(&mut self) {
        *self.slot = CANARY_PATTERN;
    }

    /// Verify the canary is still intact.
    pub fn verify(&self) -> Result<(), GuardrailError> {
        if *self.slot == CANARY_PATTERN {
            Ok(())
        } else {
            Err(GuardrailError::StackCanaryViolated)
        }
    }
}

/// Check that a benchmark's stack consumption stayed within its budget.
///
/// `entry_sp` is the stack pointer captured before the benchmark, `current_sp`
/// after (Xtensa ABI: the stack grows down, so the current SP is lower).
/// `budget_bytes` is the reserved region.  Pure arithmetic — host-testable.
pub fn stack_depth_ok(entry_sp: usize, current_sp: usize, budget_bytes: usize) -> Result<(), GuardrailError> {
    if current_sp > entry_sp {
        // Stack grew *up*?  Not possible on Xtensa — treat as a fault.
        return Err(GuardrailError::StackDepthExceeded {
            consumed_bytes: 0,
            budget_bytes,
        });
    }
    let consumed = entry_sp - current_sp;
    if consumed <= budget_bytes {
        Ok(())
    } else {
        Err(GuardrailError::StackDepthExceeded {
            consumed_bytes: consumed,
            budget_bytes,
        })
    }
}

/// Watchdog policy for benchmark runs.
///
/// By default the hardware watchdog stays **armed** (a hung benchmark resets
/// the chip — the safe default).  Only an explicit `--cfg
/// bench_watchdog_disabled` build flag (a firmware build intended purely for
/// long benchmark runs) disables it.  This function is the single source of
/// truth; the firmware refuses to proceed if the disabled policy is requested
/// but the flag is absent.
pub fn watchdog_disabled_policy(flag_present: bool, requested: bool) -> Result<bool, GuardrailError> {
    if requested && !flag_present {
        return Err(GuardrailError::WatchdogPolicyNotEnabled);
    }
    Ok(requested && flag_present)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locked_profile_verifies() {
        assert_eq!(verify_boot_profile(&LOCKED_PROFILE), Ok(()));
    }

    #[test]
    fn each_drift_fails_with_specific_error() {
        let mut p = LOCKED_PROFILE;
        p.cpu_mhz = 160;
        assert_eq!(
            verify_boot_profile(&p),
            Err(GuardrailError::CpuClockMismatch { actual_mhz: 160, expected_mhz: 240 })
        );
        let mut p = LOCKED_PROFILE;
        p.psram_qpi_mhz = 40;
        assert!(matches!(verify_boot_profile(&p), Err(GuardrailError::PsramMismatch { .. })));
        let mut p = LOCKED_PROFILE;
        p.data_cache_bytes = 32 * 1024;
        assert!(matches!(verify_boot_profile(&p), Err(GuardrailError::DataCacheMismatch { .. })));
        let mut p = LOCKED_PROFILE;
        p.cache_line_bytes = 32;
        assert!(matches!(verify_boot_profile(&p), Err(GuardrailError::CacheLineMismatch { .. })));
    }

    #[test]
    fn ccount_calibration_passes_at_exact_rate() {
        // 240,000,000 cycles over 1,000,000,000 ns = exactly 240 MHz.
        assert_eq!(assert_ccount_calibration(240_000_000, 1_000_000_000, 240_000_000, 1000), Ok(()));
    }

    #[test]
    fn ccount_calibration_rejects_drift() {
        // 10% fast: 264M cycles in 1 s → 264 MHz, way outside 1000 ppm.
        assert!(matches!(
            assert_ccount_calibration(264_000_000, 1_000_000_000, 240_000_000, 1000),
            Err(GuardrailError::CcountCalibrationFailed { .. })
        ));
    }

    #[test]
    fn ccount_calibration_tolerance_is_ppm() {
        // 240,000,240 cycles in 1 s = +1 ppm → within 10 ppm tolerance.
        assert_eq!(assert_ccount_calibration(240_000_240, 1_000_000_000, 240_000_000, 10), Ok(()));
        // +11 ppm → outside 10 ppm.
        assert!(assert_ccount_calibration(240_002_640, 1_000_000_000, 240_000_000, 10).is_err());
    }

    #[test]
    fn zero_length_calibration_window_fails() {
        assert!(assert_ccount_calibration(0, 0, 240_000_000, 1000).is_err());
    }

    #[test]
    fn canary_roundtrip() {
        static mut SLOT: u32 = 0;
        // SAFETY: single-threaded test, exclusive access to the static.
        let mut canary = StackCanary::new(unsafe { &mut SLOT });
        canary.arm();
        assert_eq!(canary.verify(), Ok(()));
        // Clobber the canary → violated.
        unsafe { SLOT = 0xDEAD_BEEF };
        assert_eq!(canary.verify(), Err(GuardrailError::StackCanaryViolated));
    }

    #[test]
    fn stack_depth_check() {
        // Entry SP 0x2000, current 0x1800 → consumed 0x800 = 2048.
        assert!(stack_depth_ok(0x2000, 0x1800, 4096).is_ok());
        assert!(stack_depth_ok(0x2000, 0x1800, 1024).is_err());
        // Stack growing up is a fault.
        assert!(stack_depth_ok(0x1800, 0x2000, 4096).is_err());
    }

    #[test]
    fn watchdog_policy_requires_explicit_flag() {
        // Default: watchdog stays armed, policy disabled is fine.
        assert_eq!(watchdog_disabled_policy(false, false), Ok(false));
        // Requesting disable without the flag → refused.
        assert_eq!(
            watchdog_disabled_policy(false, true),
            Err(GuardrailError::WatchdogPolicyNotEnabled)
        );
        // Explicit flag present → disabled allowed.
        assert_eq!(watchdog_disabled_policy(true, true), Ok(true));
    }
}
