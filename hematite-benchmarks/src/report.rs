// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Benchmark report format (plan T5.3 — three-column raw format, three
//! speedup columns, memory-tier labels, reference bars).
//!
//! # Raw columns per row (measured, never pre-filled)
//!
//! 1. **CPU cycles** — min / median of N ≥ 10 CCOUNT deltas.
//! 2. **ms @ 240 MHz** — integer `cycles * 1000 / 240_000_000`.
//! 3. **wall-clock ms** — independent wall-clock min / median.
//!
//! # Speedup columns
//!
//! * Column 1 — **vs our scalar-Rust ref** (T3.0 internal bar ≥ 10× on conv
//!   SIMD).  Computed from device measurements of the same shape through the
//!   s3 kernel and the `hematite-ref` scalar kernel.
//! * Column 2 — **vs ember-esp-nn optimized-C** (the direct competitor
//!   baseline; the plan's 15.57× column-2 bar on `conv 1×1 64×1×1×64`).
//!   Requires ember-esp-nn absolute cycle counts, sourced at device bring-up.
//! * Column 3 — **vs ESP-DL ANSI-C** (26–77×, C-vs-C — reported separately,
//!   NEVER conflated with column 1).
//!
//! Columns 2/3 render `—` until the competitor cycle counts are sourced from
//! their public benchmark tables (MUST-NOT-invent-numbers rule).  Every row
//! carries its `SRAM` / `PSRAM` working-set label.

use crate::timing::{cycles_to_ms, ns_to_ms, RunSummary};

/// Fixed point: ×100 speedup (1557 = 15.57×).
pub type SpeedupX100 = u64;

/// Compute `baseline_cycles / measured_cycles` in ×100 fixed point.
///
/// `baseline_cycles` is the comparison basis (our scalar ref, or a competitor
/// cycle count); `measured_cycles` is the s3 kernel / model.  Integer math.
pub fn speedup_x100(baseline_cycles: u64, measured_cycles: u64) -> SpeedupX100 {
    baseline_cycles
        .saturating_mul(100)
        .checked_div(measured_cycles)
        .unwrap_or(0)
}

/// One speedup column of a report row.
pub struct SpeedupCol {
    /// Column name (shown in the header).
    pub name: &'static str,
    /// Speedup in ×100 fixed point, or `None` when the baseline is unsourced.
    pub speedup: Option<SpeedupX100>,
    /// Provenance note for the baseline.
    pub note: &'static str,
}

/// A fully-formed report row.  Pure data — the same struct is consumed by the
/// host string renderer and printed field-by-field by the device firmware via
/// defmt.
pub struct ReportRow {
    /// Kernel / model label.
    pub label: &'static str,
    /// `SRAM` or `PSRAM` working-set label (every row).
    pub tier: &'static str,
    /// Number of timed runs.
    pub n: usize,
    /// Raw column 1: CCOUNT cycles (min / median).
    pub min_cycles: u64,
    pub median_cycles: u64,
    /// Raw column 2: ms @ 240 MHz (min / median).
    pub ms_240_min: u64,
    pub ms_240_median: u64,
    /// Raw column 3: wall-clock ms (min / median).
    pub wall_ms_min: u64,
    pub wall_ms_median: u64,
    /// Three speedup columns.
    pub speedups: [SpeedupCol; 3],
    /// Documented reference bar in ×10 ms (row-level, model benches), or
    /// `None` for rows without a bar.
    pub bar_tenths: Option<u32>,
    /// Provenance of the bar.
    pub bar_source: Option<&'static str>,
    /// Extra context.
    pub note: &'static str,
}

/// Build the raw-measurement fields of a row from a run summary.
///
/// `cpu_hz` is the locked 240 MHz; `bar_tenths`/`bar_source` pass through to
/// the row.  The speedup columns are filled by the caller (device measures
/// scalar + s3 and fills column 1; columns 2/3 stay `None` until sourced).
pub fn row_from_summary(
    label: &'static str,
    tier: &'static str,
    summary: &RunSummary,
    cpu_hz: u64,
    bar_tenths: Option<u32>,
    bar_source: Option<&'static str>,
    note: &'static str,
) -> ReportRow {
    ReportRow {
        label,
        tier,
        n: summary.n,
        min_cycles: summary.min_cycles,
        median_cycles: summary.median_cycles,
        ms_240_min: cycles_to_ms(summary.min_cycles, cpu_hz),
        ms_240_median: cycles_to_ms(summary.median_cycles, cpu_hz),
        wall_ms_min: ns_to_ms(summary.min_wall_ns),
        wall_ms_median: ns_to_ms(summary.median_wall_ns),
        speedups: [
            SpeedupCol {
                name: "vs scalar-Rust ref",
                speedup: None,
                note: "T3.0 internal bar: conv SIMD >= 10x",
            },
            SpeedupCol {
                name: "vs ember-esp-nn",
                speedup: None,
                note: "15.57x column-2 bar on conv1x1 64x1x1x64 (plan T5.3)",
            },
            SpeedupCol {
                name: "vs ESP-DL ANSI-C",
                speedup: None,
                note: "26-77x C-vs-C; separate from column 1 (plan T5.3)",
            },
        ],
        bar_tenths,
        bar_source,
        note,
    }
}

/// Header line for the report table.
pub const HEADER: &str =
    "| row | tier | cycles(min/med) | ms@240MHz(min/med) | wall-ms(min/med) | col1 x100 | col2 x100 | col3 x100 | bar(ms) |";

/// Render the full report as a string (host tooling / tests only; the device
/// firmware prints rows field-by-field via defmt, not through this function).
#[cfg(not(target_arch = "xtensa"))]
pub fn render_report(rows: &[ReportRow]) -> String {
    use core::fmt::Write;
    let mut out = String::new();
    out.push_str("hematite-benchmarks — ESP32-S3 (CCOUNT @ 240 MHz, N>=10, warm-up=1)\n");
    out.push_str(HEADER);
    out.push('\n');
    for r in rows {
        let _ = writeln!(
            out,
            "| {} | {} | {}/{} | {}/{} | {}/{} | {} | {} | {} | {} |",
            r.label,
            r.tier,
            r.min_cycles,
            r.median_cycles,
            r.ms_240_min,
            r.ms_240_median,
            r.wall_ms_min,
            r.wall_ms_median,
            fmt_opt_speedup(r.speedups[0].speedup),
            fmt_opt_speedup(r.speedups[1].speedup),
            fmt_opt_speedup(r.speedups[2].speedup),
            fmt_opt_bar(r.bar_tenths),
        );
    }
    out
}

/// Render a ×100 speedup as "15.57x" or "—" when unsourced.
#[cfg(not(target_arch = "xtensa"))]
fn fmt_opt_speedup(s: Option<SpeedupX100>) -> String {
    match s {
        Some(v) => format!("{}.{:02}x", v / 100, v % 100),
        None => "—".to_string(),
    }
}

/// Render a ×10 ms bar as "1294.5" or "—".
#[cfg(not(target_arch = "xtensa"))]
fn fmt_opt_bar(b: Option<u32>) -> String {
    match b {
        Some(v) => format!("{}.{}", v / 10, v % 10),
        None => "—".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speedup_math_is_integer_fixed_point() {
        // 64 cycles vs 4 cycles = 16.00x.
        assert_eq!(speedup_x100(64, 4), 1600);
        // 15.57x on the 64x1x1x64 row means: 1557 x100.
        assert_eq!(speedup_x100(1557, 100), 1557);
        // Zero measured → 0 (never div-by-zero).
        assert_eq!(speedup_x100(100, 0), 0);
    }

    #[test]
    fn row_carries_tier_label_and_raw_columns() {
        let summary = RunSummary {
            n: 10,
            min_cycles: 240_000,
            median_cycles: 250_000,
            min_wall_ns: 999_000,
            median_wall_ns: 1_045_000,
        };
        let row = row_from_summary(
            "conv_s8 8x8,64x3x3x3",
            "SRAM",
            &summary,
            crate::timing::CPU_HZ_240MHZ,
            None,
            None,
            "",
        );
        // 240_000 cycles @ 240 MHz = 1 ms exactly.
        assert_eq!(row.ms_240_min, 1);
        assert_eq!(row.wall_ms_min, 0); // 999 µs → 0 whole ms
        assert_eq!(row.wall_ms_median, 1); // 1.045 ms → 1
        assert_eq!(row.tier, "SRAM");
        assert_eq!(row.n, 10);
        // Speedups start unsourced.
        assert!(row.speedups.iter().all(|c| c.speedup.is_none()));
    }

    #[test]
    fn report_renders_all_rows_with_tiers() {
        let summary = RunSummary {
            n: 10,
            min_cycles: 10,
            median_cycles: 12,
            min_wall_ns: 50_000,
            median_wall_ns: 60_000,
        };
        let mut row = row_from_summary("k", "PSRAM", &summary, crate::timing::CPU_HZ_240MHZ, Some(70), Some("plan T5.3"), "");
        row.speedups[0].speedup = Some(1000); // 10.00x — the T3.0 bar
        let out = render_report(&[row]);
        assert!(out.contains("| k | PSRAM |"));
        assert!(out.contains("10.00x"));
        assert!(out.contains("7.0")); // 70 ×10 ms → "7.0"
        assert!(out.contains("col1 x100"));
    }

    #[test]
    fn unsourced_speedups_render_as_em_dash() {
        let summary = RunSummary {
            n: 10,
            min_cycles: 10,
            median_cycles: 12,
            min_wall_ns: 50_000,
            median_wall_ns: 60_000,
        };
        let row = row_from_summary("k", "SRAM", &summary, crate::timing::CPU_HZ_240MHZ, None, None, "");
        let out = render_report(&[row]);
        // No unverified numbers anywhere in the template.
        assert!(!out.contains("ember"));
        assert!(!out.contains("15.57x"));
    }
}
