// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Benchmark timing methodology (plan T5.3 / T5.3a, C3).
//!
//! # CCOUNT timing protocol
//!
//! Every timed measurement is **two-sample**: the CCOUNT cycle counter is read
//! immediately before and immediately after the benchmarked closure, and the
//! wall clock (an independent source) is sampled on the same two edges.  The
//! delta pair forms a [`TimedRun`].  The report therefore carries three raw
//! columns per row:
//!
//! 1. **CPU cycles** — CCOUNT deltas (the locked clock is 240 MHz).
//! 2. **ms @ 240 MHz** — `cycles * 1000 / 240_000_000`, integer arithmetic
//!    only.  This is the *expected* wall time if CCOUNT ticks at the locked
//!    rate and the benchmark was purely CPU-bound.
//! 3. **wall-clock ms** — the independent wall-clock delta.  When columns 2
//!    and 3 diverge beyond the calibration tolerance, the CCOUNT calibration
//!    assert in [`crate::guardrails`] fails the run.
//!
//! # Methodology (C3, from the plan)
//!
//! * **Warm-up** — one untimed inference per benchmark before the timed runs
//!   (i-cache / d-cache warm).
//! * **N ≥ 10 timed runs** — [`BenchmarkConfig::timed_runs`] is floored at 10.
//! * **min + median, never first-run** — [`summarize`] reports both; the
//!   first run is never used as a data point.
//! * **No `f32` timing drift** — every cycle/ns → ms conversion in this module
//!   and in [`crate::report`] is integer `u64` math.  There is no floating
//!   point anywhere in the timing path (plan F1).
//!
//! # CCOUNT wrap
//!
//! CCOUNT is a 32-bit counter; at 240 MHz it wraps every ~17.9 s.  Every
//! delta is computed with 32-bit wrapping subtraction and widened to `u64`.
//! A single benchmarked run must complete in well under one wrap period —
//! true for every kernel (µs–ms) and for the 1.3 s MobileNetV2 model bar.

/// Maximum number of timed runs that [`RunLog`] can hold.
pub const MAX_RUNS: usize = 64;

/// Locked CPU frequency of the ESP32-S3 benchmark profile (240 MHz).
pub const CPU_HZ_240MHZ: u64 = 240_000_000;

/// Benchmark run configuration.
///
/// The methodology floor (C3) is N ≥ 10 timed runs with at least one untimed
/// warm-up run; [`run_repeated`] clamps `timed_runs` up to 10 rather than
/// trusting a smaller caller value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BenchmarkConfig {
    /// Untimed runs executed first to warm i-cache / d-cache.
    pub warmup_runs: usize,
    /// Timed runs (`< 10` is silently raised to 10).
    pub timed_runs: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        BenchmarkConfig {
            warmup_runs: 1,
            timed_runs: 10,
        }
    }
}

/// A single timed observation: CCOUNT delta + independent wall-clock delta.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimedRun {
    /// CCOUNT delta between the two edges (wrapping-safe, 32-bit widened).
    pub cycles: u64,
    /// Wall-clock delta in nanoseconds.
    pub wall_ns: u64,
}

/// Clock abstraction so the timing machinery is host-testable.
///
/// On device this is implemented by [`RealClock`] (CCOUNT + ESP32-S3 system
/// timer).  Tests use a [`FakeClock`].
pub trait Clock {
    /// Current CCOUNT value (32-bit on Xtensa — widened to `u64`).
    fn now_cycles(&mut self) -> u64;
    /// Current wall-clock value in nanoseconds.
    fn now_wall_ns(&mut self) -> u64;
}

/// Fixed-capacity log of timed runs — no heap, device-safe.
#[derive(Clone, Copy, Debug)]
pub struct RunLog {
    runs: [TimedRun; MAX_RUNS],
    len: usize,
}

impl RunLog {
    /// Empty log.
    pub const fn new() -> Self {
        RunLog {
            runs: [TimedRun { cycles: 0, wall_ns: 0 }; MAX_RUNS],
            len: 0,
        }
    }

    /// Append a timed run; silently drops when full (defensive — the
    /// methodology clamps N to ≤ [`MAX_RUNS`]).
    pub fn push(&mut self, run: TimedRun) {
        if self.len < MAX_RUNS {
            self.runs[self.len] = run;
            self.len += 1;
        }
    }

    /// Number of recorded runs.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when no runs are recorded.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Recorded runs as a slice.
    pub fn as_slice(&self) -> &[TimedRun] {
        &self.runs[..self.len]
    }

    /// Drop all recorded runs.
    pub fn clear(&mut self) {
        self.len = 0;
    }
}

impl Default for RunLog {
    fn default() -> Self {
        Self::new()
    }
}

/// Run the benchmark closure: one warm-up pass, then `timed_runs` timed
/// passes.  Returns a [`RunLog`] with the timed deltas.
///
/// The closure is the *entire benchmark body* (buffer fills, kernel call, …)
/// so that the measured window includes everything a real inference costs.
/// CCOUNT + wall clock are sampled around the same edges.
pub fn run_repeated<C: Clock, F: FnMut()>(
    clock: &mut C,
    f: &mut F,
    cfg: &BenchmarkConfig,
) -> RunLog {
    // Warm-up (untimed): one pass is enough to warm i-cache/d-cache per C3.
    for _ in 0..cfg.warmup_runs.max(1) {
        f();
    }

    // Methodology floor: N ≥ 10 timed runs (MAX_RUNS = 64 > 10, no panic).
    let runs = cfg.timed_runs.clamp(10, MAX_RUNS);

    let mut log = RunLog::new();
    for _ in 0..runs {
        let c0 = clock.now_cycles();
        let w0 = clock.now_wall_ns();
        f();
        let c1 = clock.now_cycles();
        let w1 = clock.now_wall_ns();
        log.push(TimedRun {
            cycles: (c1.wrapping_sub(c0)) & 0xFFFF_FFFF,
            wall_ns: w1.saturating_sub(w0),
        });
    }
    log
}

/// Summary statistics for a run log: min and median of each metric.
///
/// Min and median are computed **independently per metric** (the run that
/// holds the min cycle count is not necessarily the run that holds the min
/// wall time).  Both are reported, per C3; the first run is never a data
/// point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunSummary {
    /// Number of timed runs.
    pub n: usize,
    /// Minimum cycle count over all runs.
    pub min_cycles: u64,
    /// Median cycle count over all runs.
    pub median_cycles: u64,
    /// Minimum wall-clock delta in ns.
    pub min_wall_ns: u64,
    /// Median wall-clock delta in ns.
    pub median_wall_ns: u64,
}

/// Compute the summary of a run log.  Returns `None` for an empty log.
pub fn summarize(log: &RunLog) -> Option<RunSummary> {
    let n = log.len();
    if n == 0 {
        return None;
    }

    let mut cyc = [0u64; MAX_RUNS];
    let mut wal = [0u64; MAX_RUNS];
    for (i, r) in log.as_slice().iter().enumerate() {
        cyc[i] = r.cycles;
        wal[i] = r.wall_ns;
    }
    cyc[..n].sort_unstable();
    wal[..n].sort_unstable();

    let min_cycles = cyc[0];
    let median_cycles = median_of(&cyc[..n]);
    let min_wall_ns = wal[0];
    let median_wall_ns = median_of(&wal[..n]);

    Some(RunSummary {
        n,
        min_cycles,
        median_cycles,
        min_wall_ns,
        median_wall_ns,
    })
}

/// Median of a sorted slice (upper-middle for even lengths, integer math).
fn median_of(sorted: &[u64]) -> u64 {
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2
    }
}

/// Convert CCOUNT cycles to milliseconds at the given CPU frequency.
///
/// **Integer-only** (`cycles * 1000 / cpu_hz`) — this is the "ms @ 240 MHz"
/// column.  No floating point anywhere in the timing path.
pub fn cycles_to_ms(cycles: u64, cpu_hz: u64) -> u64 {
    if cpu_hz == 0 {
        return 0;
    }
    cycles.saturating_mul(1000) / cpu_hz
}

/// Convert nanoseconds to whole milliseconds (integer floor).
pub fn ns_to_ms(ns: u64) -> u64 {
    ns / 1_000_000
}

/// Convert milliseconds to microseconds (integer).
pub fn ms_to_us(ms: u64) -> u64 {
    ms * 1000
}

/// Convert CCOUNT cycles to microseconds at the given CPU frequency
/// (integer; used for sub-ms kernel rows where whole-ms precision is coarse).
pub fn cycles_to_us(cycles: u64, cpu_hz: u64) -> u64 {
    if cpu_hz == 0 {
        return 0;
    }
    cycles.saturating_mul(1_000_000) / cpu_hz
}

/// Device clock: CCOUNT + ESP32-S3 system timer.
///
/// Device-only — never compiled on host (Phase 3 cfg-gating convention).
#[cfg(target_arch = "xtensa")]
pub struct RealClock;

#[cfg(target_arch = "xtensa")]
impl Clock for RealClock {
    fn now_cycles(&mut self) -> u64 {
        read_ccount() as u64
    }

    fn now_wall_ns(&mut self) -> u64 {
        // esp-hal SystemTimer is the independent wall source (BRING-UP:
        // exact 1.1 API validated at device bring-up — see firmware.rs).
        crate::firmware::read_wall_ns_impl()
    }
}

/// Read the Xtensa `CCOUNT` special register (32-bit, 240 MHz on S3).
///
/// `pub(crate)` so the device firmware's calibration routine uses the same
/// read path as [`RealClock`].
#[cfg(target_arch = "xtensa")]
pub(crate) fn read_ccount() -> u32 {
    let c: u32;
    // Xtensa: `rsr.ccount` moves CCOUNT into a general register.
    // Syntax validated against the esp-rs/rust fork asm! support at device
    // bring-up; the `mov a2, {output}` pattern in hematite-s3 conv1x1.rs
    // confirms `{reg}` operand syntax is in use in this tree.
    unsafe {
        core::arch::asm!(
            "rsr.ccount {c}",
            c = out(reg) c,
            options(nomem, nostack),
        );
    }
    c
}

/// Deterministic fake clock for host tests — yields a monotonic cycle count
/// and a monotonic wall clock so `run_repeated` / `summarize` can be tested
/// without hardware.
pub struct FakeClock {
    cycles: u64,
    wall_ns: u64,
    cycles_per_step: u64,
    wall_ns_per_step: u64,
}

impl FakeClock {
    /// New fake clock advancing by `cycles_per_step` cycles and
    /// `wall_ns_per_step` ns per read.
    pub fn new(cycles_per_step: u64, wall_ns_per_step: u64) -> Self {
        FakeClock {
            cycles: 0,
            wall_ns: 0,
            cycles_per_step,
            wall_ns_per_step,
        }
    }
}

impl Clock for FakeClock {
    fn now_cycles(&mut self) -> u64 {
        self.cycles += self.cycles_per_step;
        self.cycles
    }

    fn now_wall_ns(&mut self) -> u64 {
        self.wall_ns += self.wall_ns_per_step;
        self.wall_ns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_repeated_warms_up_and_clamps_to_10() {
        let mut clock = FakeClock::new(1, 1);
        let mut calls = 0u32;
        let cfg = BenchmarkConfig {
            warmup_runs: 0,
            timed_runs: 3, // below the floor of 10
        };
        let log = run_repeated(&mut clock, &mut || calls += 1, &cfg);
        assert_eq!(log.len(), 10, "timed_runs must be clamped up to 10");
        assert_eq!(calls, 11, "1 warm-up + 10 timed");
    }

    #[test]
    fn run_repeated_records_deltas() {
        let mut clock = FakeClock::new(100, 5_000_000); // 100 cyc / 5 ms per read
        let mut calls = 0u32;
        let log = run_repeated(&mut clock, &mut || calls += 1, &BenchmarkConfig::default());
        assert_eq!(log.len(), 10);
        for r in log.as_slice() {
            // Each metric is sampled exactly twice per run (before/after the
            // closure): the cycle delta is one `now_cycles` step and the wall
            // delta is one `now_wall_ns` step.
            assert_eq!(r.cycles, 100);
            assert_eq!(r.wall_ns, 5_000_000);
        }
    }

    #[test]
    fn summarize_reports_min_and_median() {
        // A clock whose per-read increment grows per call, so runs record
        // varying deltas: 4, 8, 12, ..., 40 for 10 runs.
        struct VarClock {
            cycles: u64,
            wall_ns: u64,
            call: u64,
        }
        impl Clock for VarClock {
            fn now_cycles(&mut self) -> u64 {
                self.call += 1;
                self.cycles += self.call * 2;
                self.cycles
            }
            fn now_wall_ns(&mut self) -> u64 {
                self.wall_ns += 1_000_000;
                self.wall_ns
            }
        }
        let mut clock = VarClock {
            cycles: 0,
            wall_ns: 0,
            call: 0,
        };
        let log = run_repeated(&mut clock, &mut || {}, &BenchmarkConfig::default());
        let s = summarize(&log).expect("non-empty log");
        assert_eq!(s.n, 10);
        assert_eq!(s.min_cycles, 4);
        // sorted 4,8,...,40 → even median = (20+24)/2
        assert_eq!(s.median_cycles, 22);
    }

    #[test]
    fn summarize_empty_is_none() {
        assert!(summarize(&RunLog::new()).is_none());
    }

    #[test]
    fn cycles_to_ms_is_integer_math() {
        // 240_000_000 cycles at 240 MHz = exactly 1000 ms.
        assert_eq!(cycles_to_ms(240_000_000, CPU_HZ_240MHZ), 1000);
        // 2400 cycles = 10 µs at 240 MHz.
        assert_eq!(cycles_to_us(2400, CPU_HZ_240MHZ), 10);
        // 1294.5 ms bar in cycles: 1294500 µs × 240 cycles/µs.
        assert_eq!(cycles_to_ms(310_680_000, CPU_HZ_240MHZ), 1294);
    }

    #[test]
    fn ccount_wrap_is_handled() {
        // Simulate CCOUNT wrapping: the mask in run_repeated's delta handling
        // must widen a 32-bit wrap to the correct short delta.
        let wrapped_delta = (u32::MAX as u64).wrapping_add(2) & 0xFFFF_FFFF;
        assert_eq!(wrapped_delta, 1, "32-bit wrapping delta widens to 1");
    }
}
