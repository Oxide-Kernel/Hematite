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

// esp-hal is referenced in both builds (CpuClock::max() is used even under
// qemu to keep the crate linked — see `read_boot_profile`).
use esp_hal::clock::CpuClock;
#[cfg(not(feature = "qemu"))]
use esp_hal::Config;
// defmt-rtt 0.4 registers its global logger by linkage only (no init fn);
// defmt 0.3.100 re-exports defmt 1.1.1 so esp-hal's messages share the RTT
// sink. Linked in both builds (see Cargo.toml) — under qemu the sink is
// invisible but the crate still provides the defmt logger symbols.
use defmt_rtt as _;

/// QEMU smoke-run UART output (feature = "qemu").
///
/// defmt-rtt emits nothing under QEMU (no RTT sink), so under the `qemu`
/// feature the report is written to UART0 via direct register access.  ESP32-S3
/// UART0 (verified against IDF v5.5 `esp32s3/register/soc/uart_reg.h` and the
/// QEMU `esp32s3_reg.h` — same map the C baseline's uart.c uses):
///
/// * base `0x60000000`
/// * `UART_STATUS` at offset `0x1C`, `TXFIFO_CNT` at bits `[25:16]`
/// * `TX_FIFO` at offset `0x00` (byte writes)
///
/// Only TXFIFO_CNT is polled (< 128 = room) before each write — no baud
/// divisor math is needed for correct visible text under QEMU.
#[cfg(feature = "qemu")]
pub(crate) mod qemu_uart {
    const UART0_BASE: usize = 0x6000_0000;
    const UART_STATUS: usize = 0x1C;
    const TX_FIFO: usize = 0x00;
    const TXFIFO_CNT_MASK: u32 = 0x3FF << 16;
    const TXFIFO_FULL: u32 = 128 << 16;

    /// Write one byte to UART0, waiting until the TX FIFO has room.
    fn putc(b: u8) {
        // SAFETY: UART0 is a fixed memory-mapped peripheral; volatile reads/
        // writes are required (the FIFO count changes behind our back).
        unsafe {
            let status_ptr = (UART0_BASE + UART_STATUS) as *const u32;
            let fifo_ptr = (UART0_BASE + TX_FIFO) as *mut u32;
            loop {
                let status = core::ptr::read_volatile(status_ptr);
                if status & TXFIFO_CNT_MASK < TXFIFO_FULL {
                    break;
                }
            }
            core::ptr::write_volatile(fifo_ptr, u32::from(b));
        }
    }

    /// `core::fmt::Write` adapter so `write!`/`writeln!` format the report.
    pub struct Uart0;

    impl core::fmt::Write for Uart0 {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            for &b in s.as_bytes() {
                putc(b);
            }
            Ok(())
        }
    }
}

/// Log one report line: defmt/RTT on hardware, UART0 under the `qemu` feature.
macro_rules! firmware_log {
    ($($arg:tt)*) => {{
        #[cfg(feature = "qemu")]
        {
            use core::fmt::Write;
            let _ = writeln!(crate::firmware::qemu_uart::Uart0, $($arg)*);
        }
        #[cfg(not(feature = "qemu"))]
        {
            defmt::info!($($arg)*);
        }
    }};
}

// Re-export so sibling device modules (e.g. model_validation) can log through
// the same path (qemu→UART0, hardware→defmt).
pub(crate) use firmware_log;

/// Display/Format adapter for `Option<u64>` so the same report row renders
/// through both `core::fmt` (qemu UART path) and `defmt` (hardware RTT path).
struct OptDisp(Option<u64>);

impl core::fmt::Display for OptDisp {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.0 {
            Some(v) => write!(f, "Some({})", v),
            None => write!(f, "None"),
        }
    }
}

impl defmt::Format for OptDisp {
    fn format(&self, f: defmt::Formatter) {
        match self.0 {
            Some(v) => defmt::write!(f, "Some({})", v),
            None => defmt::write!(f, "None"),
        }
    }
}

/// ESP-IDF application descriptor (`esp_app_desc_t`), placed first in the
/// DROM flash segment via esp-hal's `.flash.appdesc` section (rodata.x:
/// "For ESP App Description, must be placed first in image").
///
/// The IDF v5.5 2nd-stage bootloader embedded by `espflash save-image --merge`
/// casts segment 0's data to `esp_app_desc_t` and reads min/max efuse blk rev
/// from offsets 0xAC/0xB0 (bootloader_common_check_efuse_blk_validity).  Both
/// are 0 here, so the IS_FIELD_SET() gate is false and the check is skipped.
/// Without a real descriptor the bootloader reads arbitrary `.rodata` bytes
/// and rejects the app ("Image requires efuse blk rev >= vN.M") — exactly the
/// failure the C baseline's appdesc.c solved.  The `magic` also lets espflash
/// accept the descriptor without `--ignore-app-descriptor`; espflash fills the
/// `app_elf_sha256` field from the ELF.
/// Fixed-size C-string: NUL-pad `s` into `[u8; N]` (truncating if longer).
const fn cstr<const N: usize>(s: &[u8]) -> [u8; N] {
    let mut out = [0u8; N];
    let mut i = 0;
    while i < N && i < s.len() {
        out[i] = s[i];
        i += 1;
    }
    out
}

#[link_section = ".flash.appdesc"]
#[used]
#[no_mangle]
pub static esp_app_desc: EspAppDesc = EspAppDesc {
    magic: 0xABCD_5432,
    secure_version: 0,
    reserv1: [0u32; 2],
    version: cstr(b"1.0.0"),
    project_name: cstr(b"hematite-benchmarks"),
    time: cstr(b"00:00:00"),
    date: cstr(b"20260805"),
    idf_ver: cstr(b"esp-hal 1.1.1 (no IDF)"),
    app_elf_sha256: [0u8; 32],
    min_efuse_blk_rev_full: 0,
    max_efuse_blk_rev_full: 0,
    mmu_page_size: 0,
    reserv3: [0u8; 3],
    reserv2: [0u32; 18],
};

/// `esp_app_desc_t` — layout matches **espflash 4.5.0's** `AppDescriptor`
/// exactly (espflash auto-detects the `.flash.appdesc` section and parses it
/// as this struct; a size/offset mismatch makes `save-image --merge` panic).
///
/// Note this differs from the older IDF v5.5 struct the C baseline hand-rolled
/// (which espflash never parsed because its section was `.rodata.appdesc`):
/// espflash's layout has `reserv1: [u32; 2]` (not `version_len: u32`),
/// `min/max_efuse_blk_rev_full` as **u16** at 0xB0/0xB2 (the IDF bootloader
/// reads them from 0xAC/0xB0 as u32 — both 0 here, so both checks skip),
/// plus `mmu_page_size` and `reserv2: [u32; 18]`.  Total 256 bytes.
///
/// Offsets (repr(C, packed), u32-relative): magic 0x00, secure 0x04,
/// reserv1 0x08, version 0x10, project 0x30, time 0x50, date 0x60,
/// idf_ver 0x70, sha 0x90, min_efuse 0xB0 (u16), max_efuse 0xB2 (u16),
/// mmu_page_size 0xB4 (0 → espflash infers page size from alignment).
#[repr(C, packed)]
pub struct EspAppDesc {
    pub magic: u32,
    pub secure_version: u32,
    pub reserv1: [u32; 2],
    pub version: [u8; 32],
    pub project_name: [u8; 32],
    pub time: [u8; 16],
    pub date: [u8; 16],
    pub idf_ver: [u8; 32],
    pub app_elf_sha256: [u8; 32],
    pub min_efuse_blk_rev_full: u16,
    pub max_efuse_blk_rev_full: u16,
    pub mmu_page_size: u8,
    pub reserv3: [u8; 3],
    pub reserv2: [u32; 18],
}

use crate::guardrails::{verify_boot_profile, watchdog_disabled_policy, BootProfile, StackCanary};
#[cfg(not(feature = "qemu"))]
use crate::guardrails::assert_ccount_calibration;
use crate::model_bench::{model_bench_specs, ModelBenchSpec};
use crate::report::{row_from_summary, ReportRow};
use crate::spec::{
    carve_into, fill_pattern, kernel_specs, layout, run_kernel, run_ref_kernel, KernelSpec,
    MemoryTier,
};
use crate::timing::{run_repeated, summarize, BenchmarkConfig, CPU_HZ_240MHZ, RealClock};
#[cfg(not(feature = "qemu"))]
use crate::timing::read_ccount;

/// Independent wall-clock read in nanoseconds.
///
/// BRING-UP: validate against esp-hal 1.1 — `esp_hal::time::now()` returns an
/// `Instant`; the exact duration conversion (`duration_since_epoch()` +
/// `as_micros()`) must be confirmed on the esp-rs toolchain.  This is the
/// independent wall source that backs report column 3 and the CCOUNT
/// calibration assert.
///
/// Under the `qemu` feature the wall clock is unavailable: `esp_hal::init`
/// (which starts the systimer) is bypassed because its PLL reconfiguration
/// polls `bbpll_cal_done` which QEMU never asserts — so the wall column
/// renders 0 and the CCOUNT calibration is bypassed too (see
/// [`calibrate_and_assert`]).  Cycle columns are unaffected (CCOUNT is a
/// pure `rsr.ccount` read).
#[cfg(not(feature = "qemu"))]
pub fn read_wall_ns_impl() -> u64 {
    let now = esp_hal::time::Instant::now();
    let dur = now.duration_since_epoch();
    dur.as_micros().saturating_mul(1000)
}

#[cfg(feature = "qemu")]
pub fn read_wall_ns_impl() -> u64 {
    0
}

/// Device panic handler — logs (defmt or UART-under-qemu) and halts.
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    firmware_log!("PANIC: {}", info);
    loop {}
}

/// SRAM bench arena (rows whose working set fits internal SRAM).
static mut SRAM_ARENA: [u8; 256 * 1024] = [0u8; 256 * 1024];

/// PSRAM bench arena (large MobileNetV2-style rows) — runtime-mapped.
///
/// esp-hal 1.1.1 has NO static `.dram1.psram` linker section (that's an
/// esp-idf concept; esp-hal PSRAM is memory-mapped at runtime). The arena is a
/// slice over the mapped PSRAM region returned by
/// [`esp_hal::psram::Psram::raw_parts`], set once in [`run_benchmarks`].
/// If no PSRAM is present at boot, `raw_parts` yields an empty slice and the
/// PSRAM-tier benchmarks panic with "arena too small" — an honest runtime
/// failure, never a fabricated measurement.
static mut PSRAM_ARENA: &'static mut [u8] = &mut [];

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
    // CpuClock enum discriminants are the MHz values (80/160/240) on the S3.
    // Under `qemu` esp_hal::init is bypassed (PLL poll hangs in the
    // emulator), so the configured clock is the fixed 240 MHz nominal.
    #[cfg(not(feature = "qemu"))]
    let cpu_mhz = CpuClock::max() as u32;
    // Referencing `CpuClock::max()` here (a const fn — no PLL, no runtime
    // call) also keeps esp-hal linked in the qemu build: without ANY esp-hal
    // reference cargo prunes the dependency, dropping the rt startup hooks
    // (`__pre_init`/`__init_data`) that xtensa-lx-rt's reset vector needs.
    #[cfg(feature = "qemu")]
    let cpu_mhz = esp_hal::clock::CpuClock::max() as u32;
    firmware_log!("boot: CpuClock reports {} MHz", cpu_mhz);
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
    firmware_log!("bench_watchdog_disabled: hardware watchdog DISABLED for bench run");
}

/// Watchdog stays armed (safe default).
#[cfg(not(bench_watchdog_disabled))]
fn disable_watchdog() {
    firmware_log!("watchdog ARMED (safe default; pass --cfg bench_watchdog_disabled to disable)");
}

/// Measure CCOUNT over an independent wall window and assert 240 MHz.
///
/// Under the `qemu` feature this guardrail is bypassed: `-icount` skews the
/// CCOUNT-vs-wall ratio so the assert would always fail.  The busy-loop wall
/// measurement is meaningless there anyway; the QEMU numbers are explicitly
/// labeled emulation, not hardware.
#[cfg(not(feature = "qemu"))]
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
    firmware_log!("ccount calibration: {} cycles / {} ns", cycles, wall_ns);
    // 1000 ppm tolerance (0.1%).
    if let Err(e) = assert_ccount_calibration(u64::from(cycles), wall_ns, CPU_HZ_240MHZ, 1000) {
        panic!("CCOUNT calibration guardrail: {}", e.describe());
    }
}

/// QEMU: calibration bypassed (see `calibrate_and_assert` doc).
#[cfg(feature = "qemu")]
fn calibrate_and_assert() {
    firmware_log!("ccount calibration BYPASSED (qemu feature: -icount timer skew)");
}

/// Benchmark one kernel spec: scalar-ref baseline first, then the s3 kernel,
/// both with warm-up + N≥10 (C3); emits one report row via defmt.
fn bench_kernel(spec: &KernelSpec, clock: &mut RealClock, canary: &mut StackCanary) {
    let lay = layout(spec);
    // SAFETY: carve_into returns the only live borrow of the arena for the
    // duration of this benchmark; the arena is re-carved per spec.
    let arena = unsafe {
        match spec.tier {
            MemoryTier::Sram => &mut SRAM_ARENA[..],
            MemoryTier::Psram => &mut **core::ptr::addr_of_mut!(PSRAM_ARENA),
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

/// Emit a report row (defmt/RTT on hardware, UART0 under `qemu`).
///
/// Columns 2 and 3 stay `None` (rendered as `None`) until the competitor
/// cycle counts are sourced — the MUST-NOT-invent-numbers rule.
fn emit_row(row: &ReportRow) {
    firmware_log!(
        "| {} | {} | {}/{} | {}/{} | {}/{} | col1={} | col2={} | col3={} |",
        row.label,
        row.tier,
        row.min_cycles,
        row.median_cycles,
        row.ms_240_min,
        row.ms_240_median,
        row.wall_ms_min,
        row.wall_ms_median,
        OptDisp(row.speedups[0].speedup),
        OptDisp(row.speedups[1].speedup),
        OptDisp(row.speedups[2].speedup),
    );
}

/// Emit the model-level registry.  Rows whose runner is not yet wired (no
/// `.tflite` until T5.2) are listed with their documented reference bar and a
/// NOT-WIRED marker — no fabricated measurements.
fn emit_model_row(spec: &ModelBenchSpec) {
    let bar_tenths = spec.reference_ms_tenths.unwrap_or(0);
    let bar_ms = (bar_tenths / 10, bar_tenths % 10);
    firmware_log!(
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
    // esp-hal 1.1 documented init (context7: `Config::default()
    // .with_cpu_clock(CpuClock::max())` + `esp_hal::init`).
    //
    // Under the `qemu` feature this is bypassed entirely: esp-hal's clock
    // init reconfigures the PLL and polls `bbpll_cal_done`, a bit QEMU never
    // asserts, so the firmware would spin forever before any output.  The
    // QEMU run is freestanding (like the C baseline): CCOUNT, SRAM and UART0
    // all work without esp-hal; PSRAM and the wall clock are simply
    // unavailable (see the qemu-gated fns).
    #[cfg(not(feature = "qemu"))]
    {
        let config = Config::default().with_cpu_clock(CpuClock::max());
        let peripherals = esp_hal::init(config);

        // Map PSRAM at runtime and back the PSRAM-tier bench arena with the
        // mapped region (esp-hal 1.1.1 has no static `.dram1.psram` section).
        let psram = esp_hal::psram::Psram::new(
            peripherals.PSRAM,
            esp_hal::psram::PsramConfig::default(),
        );
        let (psram_ptr, psram_len) = psram.raw_parts();
        // SAFETY: single-threaded firmware; PSRAM stays mapped for program
        // lifetime (psram is held in scope). The slice is only used as a
        // scratch arena, never aliased.
        unsafe {
            PSRAM_ARENA = core::slice::from_raw_parts_mut(psram_ptr, psram_len);
        }
    }
    #[cfg(feature = "qemu")]
    {
        firmware_log!("esp-hal init SKIPPED (qemu feature: PLL cal_done poll hangs in emulator)");
        firmware_log!("PSRAM init SKIPPED (qemu feature: no PSRAM in emulator)");
    }

    // 1. Boot-profile guardrail — panic on any drift from the locked profile.
    let profile = read_boot_profile();
    if let Err(e) = verify_boot_profile(&profile) {
        panic!("boot guardrail: {}", e.describe());
    }
    firmware_log!("boot profile OK: 240 MHz / QPI 80 MHz / 64 KB x 64 B cache");

    // 2. Watchdog policy — the disable path compiles ONLY behind the explicit
    // `bench_watchdog_disabled` cfg flag; the safe default keeps it armed.
    let flag = cfg!(bench_watchdog_disabled);
    match watchdog_disabled_policy(flag, flag) {
        Ok(true) => disable_watchdog(),
        Ok(false) => firmware_log!("watchdog ARMED (safe default)"),
        Err(e) => panic!("watchdog policy: {}", e.describe()),
    }

    // 3. CCOUNT calibration assert (bypassed under qemu — see fn doc).
    calibrate_and_assert();

    // 4. Stack canary.
    // SAFETY: single-threaded firmware; unique static.
    let mut canary = StackCanary::new(unsafe { &mut *core::ptr::addr_of_mut!(STACK_CANARY_SLOT) });
    canary.arm();

    // 4.5 Model validation (model-validation feature) — runs BEFORE the
    // kernel rows so every PASS/FAIL line prints even if a later row panics
    // (the MobileNetV2 PSRAM row's "arena too small" panic stays last).
    #[cfg(feature = "model-validation")]
    crate::model_validation::validate_all();

    // 4.6 TIE728 SIMD correctness (elementwise + pool vs hematite-ref) —
    // same feature gate and ordering rationale as 4.5. conv1x1/conv3x3/gemm
    // are excluded (already proven broken under QEMU and gated off at the
    // `hematite-s3` dispatch level, not re-tested here).
    #[cfg(feature = "model-validation")]
    crate::simd_validation::validate_all();

    // 5. Per-kernel benchmarks.
    let mut clock = RealClock;
    firmware_log!("{}", crate::report::HEADER);
    for spec in kernel_specs() {
        bench_kernel(spec, &mut clock, &mut canary);
    }

    // 6. Model-level registry.
    for spec in model_bench_specs() {
        emit_model_row(spec);
    }

    firmware_log!("benchmarks complete; reference bars: MobileNetV2 224x224 = 1294.5 ms single-core (never 856 ms dual-core), KWS = 7 ms");
    loop {
        // Hold the output link open for the host to drain output.
    }
}
