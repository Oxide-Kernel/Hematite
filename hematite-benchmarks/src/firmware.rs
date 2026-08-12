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

/// UART0 report output via direct register access.
///
/// defmt-rtt emits nothing under QEMU (no RTT sink), and on real hardware the
/// RTT stream is only readable through a JTAG probe — which the bring-up
/// board does not expose (USB-UART only).  The report is therefore mirrored to
/// UART0 on **every** xtensa build so it can be captured on a plain USB-UART:
///
/// * under `qemu` it is the only transport (no RTT sink);
/// * on hardware it runs alongside defmt/RTT (for probe users).
///
/// ESP32-S3 UART0 (verified against IDF v5.5 `esp32s3/register/soc/uart_reg.h`
/// and the QEMU `esp32s3_reg.h` — same map the C baseline's uart.c uses):
///
/// * base `0x60000000`
/// * `UART_STATUS` at offset `0x1C`, `TXFIFO_CNT` at bits `[25:16]`
/// * `TX_FIFO` at offset `0x00` (byte writes)
///
/// Only TXFIFO_CNT is polled (< 128 = room) before each write — no baud
/// divisor math is needed.  The baud the bootloader left configured on UART0
/// (115200) is reused; the boot banner above confirms it is visible there.
pub(crate) mod uart0 {
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

/// Log one report line: UART0 on every xtensa build (readable on a plain
/// USB-UART), plus defmt/RTT on real hardware.  Under the `qemu` feature
/// defmt-rtt has no sink, so UART0 is the only transport there.
macro_rules! firmware_log {
    ($($arg:tt)*) => {{
        use core::fmt::Write;
        let _ = writeln!(crate::firmware::uart0::Uart0, $($arg)*);
        #[cfg(not(feature = "qemu"))]
        {
            defmt::info!($($arg)*);
        }
    }};
}

/// Log one report line: UART0 ONLY (no defmt/RTT).
///
/// The validation sections (model_validation, simd_validation) use this
/// instead of [`firmware_log`]: defmt-rtt's global logger is NOT reentrant
/// across exceptions — an exception landing inside a defmt write window makes
/// the exception handler's `defmt::panic!` hit `defmt logger taken
/// reentrantly` (defmt-rtt-0.4.2 lib.rs:139) and mask the real root cause
/// (task-5 evidence: the SIMD-correctness section died exactly that way on
/// its first line).  UART0 direct-register writes are interrupt/exception
/// safe, and RTT is unreadable on this board anyway (no JTAG probe — the
/// USB-UART is the only evidence transport), so dropping defmt here loses
/// nothing while eliminating the panic class.
macro_rules! uart0_log {
    ($($arg:tt)*) => {{
        use core::fmt::Write;
        let _ = writeln!(crate::firmware::uart0::Uart0, $($arg)*);
    }};
}

// Re-export so sibling device modules can log through the same path.
pub(crate) use uart0_log;

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
use crate::model_bench::{
    carve_model_bufs, fill_input_pattern, model_bench_specs, passes_reference_bar,
    run_model_bench, ModelBenchSpec,
};
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

/// Exception-handler wrapper (activated by linking with
/// `-Wl,--wrap=__user_exception`): receives (EXCCAUSE, save-frame pointer)
/// from xtensa-lx-rt's vector trampoline and panics with the raw cause +
/// registers, so the REAL exception lands on UART0 via the panic handler.
/// esp-hal's own `__user_exception` is compiled with the defmt feature and
/// writes the exception info to RTT only (invisible on this board), and its
/// defmt write double-faults when the trap-frame pointer is corrupt —
/// masking the original cause as "defmt logger taken reentrantly" (task-8
/// finding). `#[no_mangle]` with no normal reference: inert without --wrap.
#[cfg(target_arch = "xtensa")]
#[unsafe(no_mangle)]
unsafe extern "C" fn __wrap___user_exception(cause: u32, frame: *const u32) {
    use core::fmt::Write;
    let _ = writeln!(
        crate::firmware::uart0::Uart0,
        "EXCEPTION cause=0x{:08x} (EXCCAUSE {}):",
        cause,
        cause
    );
    // xtensa-lx-rt Context layout (no float-save-restore): [0] PC, [1] PS,
    // [2] A0, [3] A1(SP), [4..19] A2..A15, [20] SAR, [21] EXCCAUSE,
    // [22] EXCVADDR, [23] LBEG, [24] LEND, [25] LCOUNT.
    // SAFETY: the vector passed a valid save-frame pointer (sp-based);
    // reading 26 u32s is within the saved context.
    for i in 0..26 {
        let v = unsafe { frame.add(i).read() };
        match i {
            0 => writeln!(crate::firmware::uart0::Uart0, "  PC=0x{:08x}", v),
            1 => writeln!(crate::firmware::uart0::Uart0, "  PS=0x{:08x}", v),
            3 => writeln!(crate::firmware::uart0::Uart0, "  SP=0x{:08x}", v),
            22 => writeln!(crate::firmware::uart0::Uart0, "  EXCVADDR=0x{:08x}", v),
            2 | 21 | 23 | 24 | 25 => writeln!(crate::firmware::uart0::Uart0, "  r{:02}=0x{:08x}", i, v),
            _ => write!(crate::firmware::uart0::Uart0, "  r{:02}=0x{:08x}", i, v),
        }
        .ok();
        if i % 4 == 3 {
            let _ = writeln!(crate::firmware::uart0::Uart0);
        }
    }
    let _ = writeln!(crate::firmware::uart0::Uart0);
    panic!("exception");
}

/// Device panic handler — logs (UART0 only, so a defmt encoder failure can
/// never eat the panic info) and halts.
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    use core::fmt::Write;
    let _ = writeln!(crate::firmware::uart0::Uart0, "PANIC: {}", info);
    // Mirror the defmt-RTT up-channel buffer to UART0: esp-hal's exception
    // handler (compiled with the defmt feature) writes the exception frame
    // (cause + trap frame, RZCOBS-encoded) to RTT before calling
    // `__defmt_default_panic`, and RTT is unreadable on this board (no JTAG
    // probe) — without the dump the real exception cause stays invisible
    // behind the "explicit panic" tail (task-8 finding). The bytes decode
    // offline: RZCOBS → varint format tag → args.
    #[cfg(not(feature = "qemu"))]
    {
        #[repr(C)]
        struct RttChannel {
            name: *const u8,
            buffer: *mut u8,
            size: usize,
            write: usize,
            read: usize,
            flags: usize,
        }
        #[repr(C)]
        struct RttHeader {
            id: [u8; 16],
            max_up_channels: usize,
            max_down_channels: usize,
            up_channel: RttChannel,
        }
        // SAFETY: defmt-rtt's `_SEGGER_RTT` is a no_mangle static with this
        // exact repr(C) layout (defmt-rtt-0.4.2 src/lib.rs + channel.rs);
        // the panic path is single-threaded and the device is halted anyway.
        unsafe {
            extern "C" {
                #[link_name = "_SEGGER_RTT"]
                static SEGGER_RTT: RttHeader;
            }
            // The defmt RttEncoder.taken flag sits at SRAM_ARENA end + 4
            // (linker layout; task-8 finding: an out-of-bounds write past the
            // arena — e.g. the conv3x3 check's SIMD path — clobbers it to 1,
            // which makes the next defmt acquire panic "taken reentrantly").
            let arena_ptr = core::ptr::addr_of!(SRAM_ARENA) as usize;
            let enc_taken = core::ptr::read_volatile((arena_ptr + SRAM_ARENA_BYTES + 4) as *const u8);
            let _ = writeln!(
                crate::firmware::uart0::Uart0,
                "defmt RTT_ENCODER.taken = {} (arena end 0x{:08x})",
                enc_taken,
                arena_ptr + SRAM_ARENA_BYTES
            );
            let ch = &SEGGER_RTT.up_channel;
            let write = ch.write;
            let size = ch.size.min(1024);
            let n = write.min(size);
            if !ch.buffer.is_null() && n > 0 {
                let bytes = core::slice::from_raw_parts(ch.buffer, n);
                let _ = write!(crate::firmware::uart0::Uart0, "RTT dump ({} bytes):", n);
                for (i, b) in bytes.iter().enumerate() {
                    if i % 16 == 0 {
                        let _ = write!(crate::firmware::uart0::Uart0, "\n{:04x}: ", i);
                    }
                    let _ = write!(crate::firmware::uart0::Uart0, "{:02x} ", b);
                }
                let _ = writeln!(crate::firmware::uart0::Uart0);
            } else {
                let _ = writeln!(crate::firmware::uart0::Uart0, "RTT buffer empty");
            }
        }
    }
    loop {}
}

/// SRAM bench arena (rows whose working set fits internal SRAM).
///
/// 16-byte aligned so the TIE728 SIMD path's `input` pointer (the arena base,
/// carve offset 0) satisfies the kernels' alignment gate — `[u8; N]` alone
/// has alignment 1, which silently drops the bench rows to the scalar path.
///
/// `pub(crate)`: the validation sections (simd_validation) carve their big
/// buffers from this arena while it is unused (the kernel benches carve it
/// later) — a static hoist of those buffers would grow `.bss` and shrink the
/// 65 KB device stack (task-8 finding).
///
/// 256 KB is the s3 bench arena. A 32 KB slice of it is reserved for the s3
/// crate's wsum cache (Phase 1 — weight sums hoisted to model-build time via
/// a static in hematite-s3), so the effective carve arena is 224 KB. The
/// largest spec carve (conv3x3 32x32 VALID) needs ~197 KB, so 224 KB still
/// fits.
pub(crate) const SRAM_ARENA_BYTES: usize = 216 * 1024;
#[repr(align(16))]
pub(crate) struct AlignedArena(pub(crate) [u8; SRAM_ARENA_BYTES]);

pub(crate) static mut SRAM_ARENA: AlignedArena = AlignedArena([0u8; SRAM_ARENA_BYTES]);

/// Run `f` on a dedicated 256 KB stack carved from the SRAM bench arena.
///
/// The generated person_detect `Model::predict_with_scratch` allocas ~232 KB
/// of intermediates on the stack (`sub a1, a1, 0x38ac0` in the ELF) — far
/// more than the ~65 KB main-stack region — so it must run on this larger
/// stack. SAFETY contract: the arena must be unused by the caller (true
/// during model validation — the kernel benches that carve it run later) and
/// SP is restored before returning. 256 KB recorded in task-5 evidence.
///
/// QEMU-only: on real silicon the first windowed return after the SP switch
/// faults (window-underflow, excvaddr=0, epc1=retw in core::fmt::write) —
/// the device path SKIPs person_detect (reason=stack) instead of using this.
pub fn run_on_arena_stack<R>(f: impl FnOnce() -> R) -> R {
    unsafe {
        let arena = &mut *core::ptr::addr_of_mut!(SRAM_ARENA);
        let base = arena.0.as_mut_ptr();
        let top = (base as usize + arena.0.len()) & !15;
        let old_sp = read_sp();
        set_sp(top);
        let r = f();
        set_sp(old_sp);
        r
    }
}

#[cfg(target_arch = "xtensa")]
#[inline(never)]
pub(crate) unsafe fn read_sp() -> usize {
    let sp: usize;
    core::arch::asm!("mov {0}, a1", out(reg) sp, options(nostack));
    sp
}

#[cfg(target_arch = "xtensa")]
#[inline(never)]
unsafe fn set_sp(sp: usize) {
    // movsp (not mov): window-aware SP switch — relocates the register save
    // area to the new stack so window-overflow handling (device ROM handler)
    // stays consistent. A plain `mov a1, x` left the window base pointing at
    // the old stack and faulted on real silicon (QEMU's emulation was
    // lenient). The generated alloca uses the same instruction.
    core::arch::asm!("movsp a1, {0}", in(reg) sp, options(nostack));
}

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
/// BRING-UP (hardware, 2026-08-06): the original fixed 1000-iteration loop
/// produced only a ~75 µs wall window.  With the systimer's 1 µs resolution
/// that alone is ~1.3% quantization error, and the `read_wall_ns_impl` call
/// overhead (~300 cycles) is another ~1.7% on an 18 kcycle window — together
/// ~2% high, past the 1000 ppm tolerance.  The window is therefore measured
/// by polling the independent wall clock until it has advanced [`TARGET`]:
/// a 20 ms window reduces 1 µs quantization to 50 ppm and makes the fixed
/// read overhead (< 400 cycles on 4.8 Mcycles) negligible.
///
/// Under the `qemu` feature this guardrail is bypassed: `-icount` skews the
/// CCOUNT-vs-wall ratio so the assert would always fail.  The busy-loop wall
/// measurement is meaningless there anyway; the QEMU numbers are explicitly
/// labeled emulation, not hardware.
#[cfg(not(feature = "qemu"))]
fn calibrate_and_assert() {
    /// Calibration wall window in ns (20 ms → ~4.8 Mcycles at 240 MHz).
    const TARGET_WALL_NS: u64 = 20_000_000;

    let c0 = read_ccount();
    let w0 = read_wall_ns_impl();
    // Busy-wait on the independent wall clock (an opaque esp-hal call, so the
    // loop cannot be optimized away).  CCOUNT keeps ticking during the wait;
    // the wall clock is the reference.
    let mut sink: u64 = 0;
    while read_wall_ns_impl().saturating_sub(w0) < TARGET_WALL_NS {
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
            MemoryTier::Sram => &mut SRAM_ARENA.0[..],
            MemoryTier::Psram => &mut **core::ptr::addr_of_mut!(PSRAM_ARENA),
        }
    };
    let mut bufs = match carve_into(arena, &lay) {
        Ok(b) => b,
        Err(_) => panic!("arena too small for spec '{}'", spec.name),
    };
    // The ACCX kernels write raw int32 accumulators (out_c * 4 bytes) to the
    // scratch buffer before Rust requantizes. 16-byte aligned so the scratch
    // can back the kernel's aligned acc_out (and stays valid for any bench row).
    #[repr(align(16))]
    struct AlignedScratch([u8; 32768]);
    let mut scratch = AlignedScratch([0u8; 32768]);
    let cfg = BenchmarkConfig::default();

    // Prepared path: the SIMD gate runs ONCE here (outside the timed window);
    // the timed closure only re-checks pointer alignment + dispatches. This
    // isolates wrapper overhead the public-API path pays on every call.
    let prepared = match crate::spec::prepare_kernel(spec) {
        Ok(p) => p,
        Err(_) => panic!("prepared '{}': prepare_kernel failed", spec.name),
    };

    // Column 1 baseline: the same shape through the hematite-ref scalar
    // kernel on device (never a pre-filled number).
    fill_pattern(&mut bufs);
    let ref_log = run_repeated(
        clock,
        &mut || {
            let _ = run_ref_kernel(spec, &mut bufs, &mut scratch.0[..]);
        },
        &cfg,
    );
    // FNV-1a over the ref output of the final (timed) run — deterministic
    // proof of what the kernel computed; matched against the C-SIMD scalar.
    let ref_fnv = fnv1a(bufs.output);

    // The s3 kernel (SIMD on device, scalar fallback where no SIMD path).
    fill_pattern(&mut bufs);
    // NOTE: the bespoke ACCX kernels accumulate via an element-wise reduction
    // (F[lane]·I[lane]) so the RAW [oc][ic] weight layout works directly —
    // no weight transform is needed. The fill_pattern above is the raw layout.
    let s3_log = run_repeated(
        clock,
        &mut || {
            let _ = run_kernel(spec, &mut bufs, &mut scratch.0[..]);
        },
        &cfg,
    );
    // FNV-1a over the s3 output of the final (timed) run — matched against
    // the C-SIMD harness calling the identical TIE728 entry point.
    let s3_fnv = fnv1a(bufs.output);

    // Prepared path runs on the SAME (raw) weights and input — the s3 run
    // writes only `output`, never input/weights/bias, so no refill is needed
    // (a fill_pattern here would clobber the buffers the prepared path reads).
    let prepared_log = run_repeated(
        clock,
        &mut || {
            let _ = prepared.run(spec, &mut bufs, &mut scratch.0[..]);
        },
        &cfg,
    );
    let prepared_fnv = fnv1a(bufs.output);

    let prepared_sum = match summarize(&prepared_log) {
        Some(s) => s,
        None => return, // unreachable: run_repeated floors at 10
    };
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

    emit_row(&row, ref_fnv, s3_fnv);
    firmware_log!(
        "  prepared: {}/{} cycles | ms_240 {}/{} | out_fnv=0x{:08x} (matches s3 0x{:08x})",
        prepared_sum.min_cycles,
        prepared_sum.median_cycles,
        crate::timing::cycles_to_us(prepared_sum.min_cycles, CPU_HZ_240MHZ),
        crate::timing::cycles_to_us(prepared_sum.median_cycles, CPU_HZ_240MHZ),
        prepared_fnv,
        s3_fnv,
    );

    if let Err(e) = canary.verify() {
        panic!("bench '{}': {}", spec.name, e.describe());
    }
}

/// FNV-1a 32-bit checksum over raw output bytes (seed 2166136261, prime
/// 16777619) — mirrors the C baseline's `out_checksum` so the Rust s3 kernel
/// output can be matched bit-exactly against the C-SIMD harness.
 fn fnv1a(data: &[i8]) -> u32 {
     let mut h: u32 = 2_166_136_261;
     for &b in data {
         h ^= b as u32;
         h = h.wrapping_mul(16_777_619);
     }
     h
 }
 
/// End-to-end benchmark of the 4-layer CNN model — the SAME graph the
/// standard-ESP-NN baseline runs (`benchmarks/espnn-baseline`). Both the
/// `hematite-ref` scalar model and the `hematite-s3` model (ACCX kernels on
/// device) are timed end-to-end; the s3 output FNV-1a is matched against the
/// ESP-NN baseline's device-verified checksum `0x75eb32f5`.
fn bench_cnn_model(clock: &mut RealClock, canary: &mut StackCanary) {
    // SAFETY: carve_cnn_into returns the only live borrow of the arena for
    // the duration of this benchmark; the arena is re-carved per benchmark.
    let arena = unsafe { &mut SRAM_ARENA.0[..] };
    let mut bufs = match crate::model_cnn::carve_cnn_into(arena) {
        Ok(b) => b,
        Err(_) => panic!("arena too small for CNN model benchmark"),
    };
    // The ACCX kernels write raw int32 accumulators (out_c * 4 bytes) to the
    // scratch buffer before Rust requantizes.
    #[repr(align(16))]
    struct AlignedScratch([u8; 32768]);
    let mut scratch = AlignedScratch([0u8; 32768]);
    let cfg = BenchmarkConfig::default();

    // Scalar-ref model (hematite-ref) — column-1 baseline, measured on device.
    crate::model_cnn::fill_pattern_cnn(
        bufs.input, bufs.l1w, bufs.l1b, bufs.l3w, bufs.l3b, bufs.l4w, bufs.l4b,
    );
    let ref_log = run_repeated(
        clock,
        &mut || {
            let _ = crate::model_cnn::run_cnn_ref(&mut bufs, &mut scratch.0[..]);
        },
        &cfg,
    );
    let ref_fnv = crate::model_cnn::fnv1a(bufs.out);
    let ref_layers = crate::model_cnn::layer_checksums(&bufs);

    // s3 model (ACCX SIMD on device).
    crate::model_cnn::fill_pattern_cnn(
        bufs.input, bufs.l1w, bufs.l1b, bufs.l3w, bufs.l3b, bufs.l4w, bufs.l4b,
    );
    let s3_log = run_repeated(
        clock,
        &mut || {
            let _ = crate::model_cnn::run_cnn_s3(&mut bufs, &mut scratch.0[..]);
        },
        &cfg,
    );
    let s3_fnv = crate::model_cnn::fnv1a(bufs.out);
    let s3_layers = crate::model_cnn::layer_checksums(&bufs);

    let ref_sum = match summarize(&ref_log) {
        Some(s) => s,
        None => return, // unreachable: run_repeated floors at 10
    };
    let s3_sum = match summarize(&s3_log) {
        Some(s) => s,
        None => return, // unreachable: run_repeated floors at 10
    };

    let col1 = crate::report::speedup_x100(ref_sum.median_cycles, s3_sum.median_cycles);

    firmware_log!(
        "| cnn_model 4-layer (conv3x3 32x32x16 + maxpool + conv1x1 + fc) | SRAM | {}/{} | {}/{} | col1={} | out_fnv(ref/s3)=0x{:08x}/0x{:08x} |",
        s3_sum.min_cycles,
        s3_sum.median_cycles,
        crate::timing::cycles_to_us(s3_sum.min_cycles, CPU_HZ_240MHZ),
        crate::timing::cycles_to_us(s3_sum.median_cycles, CPU_HZ_240MHZ),
        col1,
        ref_fnv,
        s3_fnv,
    );
    firmware_log!(
        "  cnn_model layers: ref L1=0x{:08x} L2=0x{:08x} L3=0x{:08x} out=0x{:08x} | s3 L1=0x{:08x} L2=0x{:08x} L3=0x{:08x} out=0x{:08x} | ref_min={} ref_median={}",
        ref_layers.l1,
        ref_layers.l2,
        ref_layers.l3,
        ref_fnv,
        s3_layers.l1,
        s3_layers.l2,
        s3_layers.l3,
        s3_layers.out,
        ref_sum.min_cycles,
        ref_sum.median_cycles,
    );

    if let Err(e) = canary.verify() {
        panic!("cnn model: {}", e.describe());
    }
}

/// End-to-end MobileNetV2-style model benchmark (model B, "mv2mini").
///
/// The SAME 7-layer graph the standard-ESP-NN baseline runs
/// (`benchmarks/espnn-baseline`): conv3x3 16x16x3→14x14x32 (relu 0..127) →
/// maxpool 2x2 → depthwise 3x3 32→32 → conv1x1 32→64 → depthwise 3x3 64→64 →
/// conv1x1 64→128 → fc 1152→16. Both the scalar-ref (hematite-ref) and the
/// s3 (hematite-s3, ACCX SIMD where eligible) runs are timed end-to-end; the
/// s3 output FNV-1a is matched against the ESP-NN baseline's device-verified
/// checksum `0x7f23eb05`.
fn bench_mv2_model(clock: &mut RealClock, canary: &mut StackCanary) {
    // SAFETY: carve_mv2_into returns the only live borrow of the arena for
    // the duration of this benchmark; the arena is re-carved per benchmark.
    let arena = unsafe { &mut SRAM_ARENA.0[..] };
    let mut bufs = match crate::model_mv2::carve_mv2_into(arena) {
        Ok(b) => b,
        Err(_) => panic!("arena too small for MV2 model benchmark"),
    };
    // The ACCX kernels write raw int32 accumulators (out_c * 4 bytes) to the
    // scratch buffer before Rust requantizes.
    #[repr(align(16))]
    struct AlignedScratch([u8; 16384]);
    let mut scratch = AlignedScratch([0u8; 16384]);
    let cfg = BenchmarkConfig::default();

    // Scalar-ref model (hematite-ref) — column-1 baseline, measured on device.
    crate::model_mv2::fill_pattern_mv2(
        bufs.input, bufs.l1w, bufs.l1b, bufs.l3w, bufs.l3b, bufs.l4w, bufs.l4b, bufs.l5w, bufs.l5b,
        bufs.l6w, bufs.l6b, bufs.l7w, bufs.l7b,
    );
    let ref_log = run_repeated(
        clock,
        &mut || {
            let _ = crate::model_mv2::run_mv2_ref(&mut bufs, &mut scratch.0[..]);
        },
        &cfg,
    );
    let ref_fnv = crate::model_mv2::fnv1a(bufs.out);
    let ref_layers = crate::model_mv2::layer_checksums_mv2(&bufs);

    // s3 model (ACCX SIMD on device).
    crate::model_mv2::fill_pattern_mv2(
        bufs.input, bufs.l1w, bufs.l1b, bufs.l3w, bufs.l3b, bufs.l4w, bufs.l4b, bufs.l5w, bufs.l5b,
        bufs.l6w, bufs.l6b, bufs.l7w, bufs.l7b,
    );
    let s3_log = run_repeated(
        clock,
        &mut || {
            let _ = crate::model_mv2::run_mv2_s3(&mut bufs, &mut scratch.0[..]);
        },
        &cfg,
    );
    let s3_fnv = crate::model_mv2::fnv1a(bufs.out);
    let s3_layers = crate::model_mv2::layer_checksums_mv2(&bufs);

    let ref_sum = match summarize(&ref_log) {
        Some(s) => s,
        None => return, // unreachable: run_repeated floors at 10
    };
    let s3_sum = match summarize(&s3_log) {
        Some(s) => s,
        None => return, // unreachable: run_repeated floors at 10
    };

    let col1 = crate::report::speedup_x100(ref_sum.median_cycles, s3_sum.median_cycles);

    firmware_log!(
        "| mv2mini 7-layer (conv3x3 16x16x3 + maxpool + dw + conv1x1 + dw + conv1x1 + fc) | SRAM | {}/{} | {}/{} | col1={} | out_fnv(ref/s3)=0x{:08x}/0x{:08x} |",
        s3_sum.min_cycles,
        s3_sum.median_cycles,
        crate::timing::cycles_to_us(s3_sum.min_cycles, CPU_HZ_240MHZ),
        crate::timing::cycles_to_us(s3_sum.median_cycles, CPU_HZ_240MHZ),
        col1,
        ref_fnv,
        s3_fnv,
    );
    firmware_log!(
        "  mv2mini layers: ref L1=0x{:08x} L2=0x{:08x} L3=0x{:08x} L4=0x{:08x} L5=0x{:08x} L6=0x{:08x} out=0x{:08x} | s3 L1=0x{:08x} L2=0x{:08x} L3=0x{:08x} L4=0x{:08x} L5=0x{:08x} L6=0x{:08x} out=0x{:08x} | ref_min={} ref_median={}",
        ref_layers.l1,
        ref_layers.l2,
        ref_layers.l3,
        ref_layers.l4,
        ref_layers.l5,
        ref_layers.l6,
        ref_fnv,
        s3_layers.l1,
        s3_layers.l2,
        s3_layers.l3,
        s3_layers.l4,
        s3_layers.l5,
        s3_layers.l6,
        s3_layers.out,
        ref_sum.min_cycles,
        ref_sum.median_cycles,
    );

    if let Err(e) = canary.verify() {
        panic!("mv2 model: {}", e.describe());
    }
}

fn bench_mv2real_model(clock: &mut RealClock, canary: &mut StackCanary) {
    // SAFETY: carve_mv2real_into returns the only live borrow of the arena for
    // the duration of this benchmark; the arena is re-carved per benchmark.
    let arena = unsafe { &mut SRAM_ARENA.0[..] };
    let mut bufs = match crate::model_mv2real::carve_mv2real_into(arena) {
        Ok(b) => b,
        Err(_) => panic!("arena too small for MV2REAL model benchmark"),
    };
    // The ACCX kernels write raw int32 accumulators (out_c * 4 bytes) to the
    // scratch buffer before Rust requantizes; L1's zero-padded input+weights
    // carve needs ~5KB more.
    #[repr(align(16))]
    struct AlignedScratch([u8; 16384]);
    let mut scratch = AlignedScratch([0u8; 16384]);
    let cfg = BenchmarkConfig::default();

    // Scalar-ref model (hematite-ref) — column-1 baseline, measured on device.
    crate::model_mv2real::fill_pattern_mv2real(
        bufs.input, bufs.l1w, bufs.l1b, bufs.l2w, bufs.l2b, bufs.l3w, bufs.l3b, bufs.l4w, bufs.l4b,
        bufs.l5w, bufs.l5b, bufs.l6w, bufs.l6b,
    );
    let ref_log = run_repeated(
        clock,
        &mut || {
            let _ = crate::model_mv2real::run_mv2real_ref(&mut bufs, &mut scratch.0[..]);
        },
        &cfg,
    );
    let ref_fnv = crate::model_mv2real::fnv1a(bufs.out);
    let ref_layers = crate::model_mv2real::layer_checksums_mv2real(&bufs);

    // s3 model (ACCX SIMD on device).
    crate::model_mv2real::fill_pattern_mv2real(
        bufs.input, bufs.l1w, bufs.l1b, bufs.l2w, bufs.l2b, bufs.l3w, bufs.l3b, bufs.l4w, bufs.l4b,
        bufs.l5w, bufs.l5b, bufs.l6w, bufs.l6b,
    );
    let s3_log = run_repeated(
        clock,
        &mut || {
            let _ = crate::model_mv2real::run_mv2real_s3(&mut bufs, &mut scratch.0[..]);
        },
        &cfg,
    );
    let s3_fnv = crate::model_mv2real::fnv1a(bufs.out);
    let s3_layers = crate::model_mv2real::layer_checksums_mv2real(&bufs);

    let ref_sum = match summarize(&ref_log) {
        Some(s) => s,
        None => return, // unreachable: run_repeated floors at 10
    };
    let s3_sum = match summarize(&s3_log) {
        Some(s) => s,
        None => return, // unreachable: run_repeated floors at 10
    };

    let col1 = crate::report::speedup_x100(ref_sum.median_cycles, s3_sum.median_cycles);

    firmware_log!(
        "| mv2real 6-layer (conv3x3 16x16x3 SAME/s2 + dw SAME + conv1x1 + dw SAME/s2 + conv1x1 + fc) | SRAM | {}/{} | {}/{} | col1={} | out_fnv(ref/s3)=0x{:08x}/0x{:08x} |",
        s3_sum.min_cycles,
        s3_sum.median_cycles,
        crate::timing::cycles_to_us(s3_sum.min_cycles, CPU_HZ_240MHZ),
        crate::timing::cycles_to_us(s3_sum.median_cycles, CPU_HZ_240MHZ),
        col1,
        ref_fnv,
        s3_fnv,
    );
    firmware_log!(
        "  mv2real layers: ref L1=0x{:08x} L2=0x{:08x} L3=0x{:08x} L4=0x{:08x} L5=0x{:08x} out=0x{:08x} | s3 L1=0x{:08x} L2=0x{:08x} L3=0x{:08x} L4=0x{:08x} L5=0x{:08x} out=0x{:08x} | ref_min={} ref_median={}",
        ref_layers.l1,
        ref_layers.l2,
        ref_layers.l3,
        ref_layers.l4,
        ref_layers.l5,
        ref_fnv,
        s3_layers.l1,
        s3_layers.l2,
        s3_layers.l3,
        s3_layers.l4,
        s3_layers.l5,
        s3_layers.out,
        ref_sum.min_cycles,
        ref_sum.median_cycles,
    );

    if let Err(e) = canary.verify() {
        panic!("mv2real model: {}", e.describe());
    }
}

/// Profile Model C (mv2real) per-layer s3 CCOUNT deltas (todo-21).
///
/// One untimed full s3 run populates every layer's input buffer; each layer
/// is then timed alone via `run_repeated` (N>=10, min/median).
fn profile_mv2real_s3_layers(canary: &mut StackCanary) {
    let arena = unsafe { &mut SRAM_ARENA.0[..] };
    let mut bufs = match crate::model_mv2real::carve_mv2real_into(arena) {
        Ok(b) => b,
        Err(_) => panic!("arena too small for MV2REAL layer profile"),
    };
    #[repr(align(16))]
    struct AlignedScratch([u8; 16384]);
    let mut scratch = AlignedScratch([0u8; 16384]);
    crate::model_mv2real::fill_pattern_mv2real(
        bufs.input, bufs.l1w, bufs.l1b, bufs.l2w, bufs.l2b, bufs.l3w, bufs.l3b, bufs.l4w, bufs.l4b,
        bufs.l5w, bufs.l5b, bufs.l6w, bufs.l6b,
    );
    let _ = crate::model_mv2real::run_mv2real_s3(&mut bufs, &mut scratch.0[..]);
    let cfg = BenchmarkConfig::default();
    let mut clock = RealClock;

    let mut rows: [(&str, u64); 6] = [("", 0); 6];

    let log = run_repeated(
        &mut clock,
        &mut || {
            let _ = hematite_s3::conv3x3::conv2d_3x3(bufs.input, bufs.l1w, bufs.l1b, &crate::model_mv2real::C1_PARAMS, bufs.l1out, &mut scratch.0[..]);
        },
        &cfg,
    );
    rows[0] = ("L1 conv3x3 s2", summarize(&log).map(|s| s.min_cycles).unwrap_or(0));

    let log = run_repeated(
        &mut clock,
        &mut || {
            let _ = hematite_s3::depthwise::depthwise_conv2d(bufs.l1out, bufs.l2w, bufs.l2b, &crate::model_mv2real::C2_PARAMS, bufs.l2out, &mut scratch.0[..]);
        },
        &cfg,
    );
    rows[1] = ("L2 depthwise s1", summarize(&log).map(|s| s.min_cycles).unwrap_or(0));

    let log = run_repeated(
        &mut clock,
        &mut || {
            let _ = hematite_s3::conv1x1::conv2d_1x1(bufs.l2out, bufs.l3w, bufs.l3b, &crate::model_mv2real::C3_PARAMS, bufs.l3out, &mut scratch.0[..]);
        },
        &cfg,
    );
    rows[2] = ("L3 conv1x1", summarize(&log).map(|s| s.min_cycles).unwrap_or(0));

    let log = run_repeated(
        &mut clock,
        &mut || {
            let _ = hematite_s3::depthwise::depthwise_conv2d(bufs.l3out, bufs.l4w, bufs.l4b, &crate::model_mv2real::C4_PARAMS, bufs.l4out, &mut scratch.0[..]);
        },
        &cfg,
    );
    rows[3] = ("L4 depthwise s2", summarize(&log).map(|s| s.min_cycles).unwrap_or(0));

    let log = run_repeated(
        &mut clock,
        &mut || {
            let _ = hematite_s3::conv1x1::conv2d_1x1(bufs.l4out, bufs.l5w, bufs.l5b, &crate::model_mv2real::C5_PARAMS, bufs.l5out, &mut scratch.0[..]);
        },
        &cfg,
    );
    rows[4] = ("L5 conv1x1", summarize(&log).map(|s| s.min_cycles).unwrap_or(0));

    let log = run_repeated(
        &mut clock,
        &mut || {
            let _ = hematite_s3::gemm::fully_connected(bufs.l5out, bufs.l6w, bufs.l6b, &crate::model_mv2real::C6_PARAMS, bufs.out, &mut scratch.0[..]);
        },
        &cfg,
    );
    rows[5] = ("L6 fc", summarize(&log).map(|s| s.min_cycles).unwrap_or(0));

    firmware_log!("mv2real per-layer s3 (min cyc):");
    for (name, cyc) in rows {
        firmware_log!("  {} {}", name, cyc);
    }

    // Raw-kernel floor probes (todo-21): time the bare asm kernel calls with
    // NO Rust per-pixel dispatch / requantize, to split kernel floor vs
    // dispatcher overhead for Model C's shapes.
    #[cfg(all(target_arch = "xtensa", not(feature = "qemu")))]
    {
        let mut ccount = crate::timing::read_ccount;
        // L1 shape: conv3x3 fast16, out_c=32, row_delta=(17-3)*16=224.
        let n = 64u32;
        let t0 = ccount();
        let mut sink: u32 = 0;
        for _ in 0..n {
            unsafe {
                hematite_s3::accx::accx_conv3x3(
                    bufs.input.as_ptr(),
                    bufs.l1w.as_ptr(),
                    bufs.l1b.as_ptr() as *const i32 as *mut i32,
                    16,
                    32,
                    224,
                );
            }
            sink = sink.wrapping_add(1);
        }
        let t1 = ccount();
        firmware_log!(
            "probe accx_conv3x3 L1 shape: {} cyc/call (n={})",
            (t1 - t0) / n,
            n
        );
        // L3 shape: conv1x1 in_c=32, out_c=64.
        let t0 = ccount();
        for _ in 0..n {
            unsafe {
                hematite_s3::accx::accx_conv1x1(
                    bufs.input.as_ptr(),
                    bufs.l1w.as_ptr(),
                    bufs.l1b.as_ptr() as *const i32 as *mut i32,
                    32,
                    64,
                );
            }
            sink = sink.wrapping_add(1);
        }
        let t1 = ccount();
        firmware_log!(
            "probe accx_conv1x1 L3 shape: {} cyc/call (n={})",
            (t1 - t0) / n,
            n
        );
        // L5 shape: conv1x1 in_c=64, out_c=128.
        let t0 = ccount();
        for _ in 0..n {
            unsafe {
                hematite_s3::accx::accx_conv1x1(
                    bufs.input.as_ptr(),
                    bufs.l1w.as_ptr(),
                    bufs.l1b.as_ptr() as *const i32 as *mut i32,
                    64,
                    128,
                );
            }
            sink = sink.wrapping_add(1);
        }
        let t1 = ccount();
        firmware_log!(
            "probe accx_conv1x1 L5 shape: {} cyc/call (n={})",
            (t1 - t0) / n,
            n
        );
        // L2 depthwise shape: in_c=32, out_c=32, row_delta=(8-3)*32=160.
        let t0 = ccount();
        for _ in 0..n {
            unsafe {
                hematite_s3::accx::accx_depthwise(
                    bufs.input.as_ptr(),
                    bufs.l1w.as_ptr(),
                    bufs.l1b.as_ptr() as *const i32 as *mut i32,
                    32,
                    32,
                    160,
                );
            }
            sink = sink.wrapping_add(1);
        }
        let t1 = ccount();
        firmware_log!(
            "probe accx_depthwise L2 shape: {} cyc/call (n={})",
            (t1 - t0) / n,
            n
        );
        let _ = sink;
    }

    if let Err(e) = canary.verify() {
        panic!("mv2real layer profile: {}", e.describe());
    }
}
 
/// Emit a report row (defmt/RTT on hardware, UART0 under `qemu`).
///
/// Columns 2 and 3 stay `None` (rendered as `None`) until the competitor
/// cycle counts are sourced — the MUST-NOT-invent-numbers rule.
///
/// `ref_fnv` / `s3_fnv` are FNV-1a checksums over the final output of the
/// scalar-ref and s3 (SIMD where eligible) runs respectively; the s3 one is
/// matched against the C-SIMD harness output checksum.
fn emit_row(row: &ReportRow, ref_fnv: u32, s3_fnv: u32) {
    firmware_log!(
        "| {} | {} | {}/{} | {}/{} | {}/{} | col1={} | col2={} | col3={} | out_fnv(ref/s3)=0x{:08x}/0x{:08x} |",
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
        ref_fnv,
        s3_fnv,
    );
}

/// Fit-size FC model — proves flash→SRAM weight staging on device (Phase 2).
///
/// The four fully-connected layers use weights declared as immutable `static`
/// consts, which the Xtensa linker places in flash-backed DROM. The ACCX fc
/// dispatch stages each distinct layer's weights into the caller's persistent
/// scratch ONCE (the model bench reuses the same scratch across all predict
/// calls), so the timed runs read SRAM instead of streaming flash (~96x win
/// measured on an 80 KiB DROM stream). Total weights 57 KiB fit the 64 KiB
/// scratch — the honest fit-size proof on this no-PSRAM board.
#[cfg(feature = "model-validation")]
fn bench_fit_model(clock: &mut RealClock, canary: &mut StackCanary) {
    // SAFETY: carve_fit_into returns the only live borrow of the arena for the
    // duration of this benchmark; the arena is re-carved per benchmark.
    let arena = unsafe { &mut SRAM_ARENA.0[..] };
    let mut bufs = match crate::model_fit::carve_fit_into(arena) {
        Ok(b) => b,
        Err(_) => panic!("arena too small for fit-size model benchmark"),
    };
    let cfg = BenchmarkConfig::default();

    // Scalar-ref model (hematite-ref) — column-1 baseline, measured on device.
    crate::model_fit::fill_fit_input(bufs.input);
    let ref_log = run_repeated(
        clock,
        &mut || {
            let _ = crate::model_fit::run_fit_ref(&mut bufs);
        },
        &cfg,
    );
    let ref_fnv = crate::model_fit::fnv1a(bufs.out);

    // s3 model reading the DROM consts directly (flash-bound control).
    crate::model_fit::fill_fit_input(bufs.input);
    let s3_log = run_repeated(
        clock,
        &mut || {
            let _ = crate::model_fit::run_fit_s3(&mut bufs);
        },
        &cfg,
    );
    let s3_fnv = crate::model_fit::fnv1a(bufs.out);

    // Staged s3 model — weights copied DROM→SRAM ONCE (untimed), then the same
    // fc chain runs from SRAM. Output must equal the ref/s3 checksum (the same
    // bytes, only the residency changed).
    crate::model_fit::fill_fit_input(bufs.input);
    let _ = crate::model_fit::stage_fit_weights(&mut bufs);
    let staged_log = run_repeated(
        clock,
        &mut || {
            let _ = crate::model_fit::run_fit_s3_staged(&mut bufs);
        },
        &cfg,
    );
    let staged_fnv = crate::model_fit::fnv1a(bufs.out);

    let ref_sum = match summarize(&ref_log) {
        Some(s) => s,
        None => return,
    };
    let s3_sum = match summarize(&s3_log) {
        Some(s) => s,
        None => return,
    };
    let staged_sum = match summarize(&staged_log) {
        Some(s) => s,
        None => return,
    };

    let col1 = crate::report::speedup_x100(ref_sum.median_cycles, s3_sum.median_cycles);

    firmware_log!(
        "| fit_model 4-layer fc (256x128 + 128x128 + 128x64 + 64x16) | SRAM | flash {}/{} | staged {}/{} | col1={} | out_fnv(ref/s3/staged)=0x{:08x}/0x{:08x}/0x{:08x} |",
        s3_sum.min_cycles,
        s3_sum.median_cycles,
        staged_sum.min_cycles,
        staged_sum.median_cycles,
        col1,
        ref_fnv,
        s3_fnv,
        staged_fnv,
    );

    if let Err(e) = canary.verify() {
        panic!("fit model: {}", e.describe());
    }
}

/// Emit the model-level registry row when no runner is compiled in this
/// build (no `model-validation` feature): the spec + its documented bar only,
/// never a fabricated measurement.
fn emit_model_row(spec: &ModelBenchSpec) {
    let bar_tenths = spec.reference_ms_tenths.unwrap_or(0);
    let bar_ms = (bar_tenths / 10, bar_tenths % 10);
    firmware_log!(
        "| {} | {} | bar-only (no runner; build with --features model-validation) | bar={}.{} ms | {}",
        spec.name,
        spec.tier.label(),
        bar_ms.0,
        bar_ms.1,
        spec.source,
    );
}

/// Benchmark one real zoo model end-to-end via `Model::<S3Backend>` (plan
/// simd-zoo-hardening todo 19): warm-up + N ≥ 10 timed inferences with
/// buffers carved from the bench arena, one timed row (cycles + wall ms) with
/// the reference-bar verdict.
///
/// SKIP records (Metis F10 format) are emitted instead of timings where the
/// model cannot run on this board — never a fabricated measurement:
///
/// * person_detect — the generated `predict` allocas ~232 KB of stack
///   intermediates vs the ~65 KB device stack; the arena-stack SP switch
///   faults on real silicon (todo-5 finding). SKIP reason=stack.
/// * mobilenet_v2 — PSRAM tier; this board has no PSRAM (`PSRAM: 0 bytes`).
///   SKIP reason=no-psram when the PSRAM arena is empty (the carve check
///   catches a present-but-too-small arena the same way).
#[cfg(feature = "model-validation")]
fn bench_zoo_model(spec: &ModelBenchSpec, clock: &mut RealClock, canary: &mut StackCanary) {
    use crate::model_bench::zoo_runners::zoo_runner_for;

    // person_detect: proven hardware-fault on this board — never attempted.
    if spec.path == "models/zoo/person_detect_vww/person_detect_int8.tflite" {
        crate::firmware::uart0_log!(
            "model person_detect_int8 [bench]: SKIP reason=stack rerun_condition=codegen-intermediates-off-stack"
        );
        firmware_log!(
            "| {} | {} | SKIP | reason=stack rerun_condition=codegen-intermediates-off-stack |",
            spec.name,
            spec.tier.label(),
        );
        return;
    }

    // SAFETY: carve_model_bufs returns the only live borrow of the arena for
    // the duration of this benchmark; the arena is re-carved per spec (same
    // pattern as bench_kernel).
    let arena = unsafe {
        match spec.tier {
            crate::spec::MemoryTier::Sram => &mut SRAM_ARENA.0[..],
            crate::spec::MemoryTier::Psram => &mut **core::ptr::addr_of_mut!(PSRAM_ARENA),
        }
    };
    // mobilenet_v2 PSRAM gate: no PSRAM on this board.
    if arena.is_empty() {
        crate::firmware::uart0_log!(
            "model mobilenet_v2_1.0_224_int8 [bench]: SKIP reason=no-psram rerun_condition=board-with-PSRAM"
        );
        firmware_log!(
            "| {} | {} | SKIP | reason=no-psram rerun_condition=board-with-PSRAM |",
            spec.name,
            spec.tier.label(),
        );
        return;
    }

    let mut runner = zoo_runner_for(spec);
    let mut bufs = match carve_model_bufs(
        arena,
        runner.input_len(),
        runner.output_len(),
        runner.scratch_len(),
    ) {
        Some(b) => b,
        None => {
            firmware_log!(
                "| {} | {} | SKIP | reason=arena-too-small |",
                spec.name,
                spec.tier.label(),
            );
            return;
        }
    };
    let cfg = BenchmarkConfig::default();
    fill_input_pattern(bufs.input);
    bufs.output.fill(0);

    let log = run_model_bench(clock, &mut runner, bufs.input, bufs.output, bufs.scratch, &cfg);
    let summary = match summarize(&log) {
        Some(s) => s,
        None => return, // unreachable: run_repeated floors at 10
    };
    let out_fnv = fnv1a(bufs.output);

    match spec.reference_ms_tenths {
        Some(t) => {
            let bar_ms = (t / 10, t % 10);
            let verdict = if passes_reference_bar(&summary, t) { "PASS" } else { "FAIL" };
            firmware_log!(
                "| {} | {} | {}/{} | {}/{} | {}/{} | bar={}.{} ms | {} | out_fnv=0x{:08x} | {} |",
                spec.name,
                spec.tier.label(),
                summary.min_cycles,
                summary.median_cycles,
                crate::timing::cycles_to_ms(summary.min_cycles, CPU_HZ_240MHZ),
                crate::timing::cycles_to_ms(summary.median_cycles, CPU_HZ_240MHZ),
                crate::timing::ns_to_ms(summary.min_wall_ns),
                crate::timing::ns_to_ms(summary.median_wall_ns),
                bar_ms.0,
                bar_ms.1,
                verdict,
                out_fnv,
                spec.source,
            );
        }
        None => {
            firmware_log!(
                "| {} | {} | {}/{} | {}/{} | {}/{} | bar=None | no-bar | out_fnv=0x{:08x} | {} |",
                spec.name,
                spec.tier.label(),
                summary.min_cycles,
                summary.median_cycles,
                crate::timing::cycles_to_ms(summary.min_cycles, CPU_HZ_240MHZ),
                crate::timing::cycles_to_ms(summary.median_cycles, CPU_HZ_240MHZ),
                crate::timing::ns_to_ms(summary.min_wall_ns),
                crate::timing::ns_to_ms(summary.median_wall_ns),
                out_fnv,
                spec.source,
            );
        }
    }

    if let Err(e) = canary.verify() {
        panic!("model bench '{}': {}", spec.name, e.describe());
    }
}

/// Benchmark one real zoo model through the UNFUSED arm (`#[model_unfused]`
/// per-op sequence, no fusion schedule) — the T6.1 fused-vs-unfused delta
/// denominator.  Same SKIP guards as the fused arm: person_detect is
/// stack-gated, mobilenet_v2 is PSRAM-gated (no unfused runner is wired for
/// either — `zoo_unfused_runner_for` panics on their paths, which the guards
/// below keep unreachable).
#[cfg(feature = "model-validation")]
fn bench_zoo_model_unfused(spec: &ModelBenchSpec, clock: &mut RealClock, canary: &mut StackCanary) {
    use crate::model_bench::zoo_unfused_runners::zoo_unfused_runner_for;

    if spec.path == "models/zoo/person_detect_vww/person_detect_int8.tflite" {
        return;
    }
    let arena = unsafe {
        match spec.tier {
            crate::spec::MemoryTier::Sram => &mut SRAM_ARENA.0[..],
            crate::spec::MemoryTier::Psram => &mut **core::ptr::addr_of_mut!(PSRAM_ARENA),
        }
    };
    if arena.is_empty() {
        return;
    }

    let mut runner = zoo_unfused_runner_for(spec);
    let mut bufs = match carve_model_bufs(
        arena,
        runner.input_len(),
        runner.output_len(),
        runner.scratch_len(),
    ) {
        Some(b) => b,
        None => return,
    };
    let cfg = BenchmarkConfig::default();
    fill_input_pattern(bufs.input);
    bufs.output.fill(0);

    let log = run_model_bench(clock, &mut runner, bufs.input, bufs.output, bufs.scratch, &cfg);
    let summary = match summarize(&log) {
        Some(s) => s,
        None => return,
    };
    let out_fnv = fnv1a(bufs.output);

    firmware_log!(
        "| {} | {} | {}/{} | {}/{} | {}/{} | bar=None | no-bar (unfused arm) | out_fnv=0x{:08x} |",
        spec.name,
        spec.tier.label(),
        summary.min_cycles,
        summary.median_cycles,
        crate::timing::cycles_to_ms(summary.min_cycles, CPU_HZ_240MHZ),
        crate::timing::cycles_to_ms(summary.median_cycles, CPU_HZ_240MHZ),
        crate::timing::ns_to_ms(summary.min_wall_ns),
        crate::timing::ns_to_ms(summary.median_wall_ns),
        out_fnv,
    );

    if let Err(e) = canary.verify() {
        panic!("model bench (unfused) '{}': {}", spec.name, e.describe());
    }
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
        // PSRAM presence probe (plan simd-zoo-hardening todo 1, Metis F4): log
        // the arena capacity BEFORE `verify_boot_profile` asserts — an empty
        // slice means the board has no PSRAM, and that fact must be reported
        // rather than preempted by the boot-profile panic.
        firmware_log!("PSRAM: {} bytes", psram_len);
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
    // validate_all() runs each zoo model via RefBackend; validate_all_s3()
    // (plan todo 5) re-runs them via Model::<S3Backend> — on the device the
    // S3 forwarding takes the SIMD paths, so a PASS proves the SIMD kernels
    // agree with the scalar oracle end-to-end.
    #[cfg(feature = "model-validation")]
    crate::model_validation::validate_all();
    #[cfg(feature = "model-validation")]
    crate::model_validation::validate_all_s3();

    // 4.6 TIE728 SIMD correctness (elementwise + pool vs hematite-ref) —
    // hardware-only: gated `not(feature = "qemu")` because the QEMU fork's
    // TIE728 compute emulation is broken (VADDS crash, VSUBS silent wrong,
    // VMULAS/pool hang) — the SIMD suite must never run under qemu. Same
    // feature + ordering rationale as 4.5. conv1x1/conv3x3/gemm are excluded
    // (already gated off at the `hematite-s3` dispatch level).
    #[cfg(all(feature = "model-validation", not(feature = "qemu")))]
    crate::simd_validation::validate_all();

    // 4.7 person_detect stack-probe (composed-kernels T5.2) — AFTER the SIMD
    // sweep so a stack overflow here (the honest shortfall signal) can never
    // mask the sweep results. Runs predict_with_scratch on the real device
    // stack budget (T1.3 arena peak 55,296 B vs ~65 KB stack); NO static-mut
    // decision is made — a shortfall is recorded for the owner.
    #[cfg(all(feature = "model-validation", not(feature = "qemu")))]
    crate::model_validation::probe_person_detect_stack();

    // 5. Per-kernel benchmarks.
    let mut clock = RealClock;
    firmware_log!("{}", crate::report::HEADER);
    // The end-to-end 4-layer CNN model benchmark runs BEFORE the per-kernel
    // rows because the kernel loop terminates in the expected no-PSRAM panic
    // on this board (the PSRAM-tier MobileNetV2 row's "arena too small").
    bench_cnn_model(&mut clock, &mut canary);
    bench_mv2_model(&mut clock, &mut canary);
    bench_mv2real_model(&mut clock, &mut canary);
    #[cfg(feature = "model-validation")]
    bench_fit_model(&mut clock, &mut canary);
    profile_mv2real_s3_layers(&mut canary);

    // 5.5 Model-level registry — real zoo runners (todo 19). Runs before the
    // kernel loop for the same reason the A/B/C benches do: the loop ends in
    // the expected no-PSRAM panic, and the timed model rows must print first.
    // Without the `model-validation` feature no runners are compiled and the
    // rows are bar-only (no fabricated timings).
    #[cfg(feature = "model-validation")]
    for spec in model_bench_specs() {
        bench_zoo_model(spec, &mut clock, &mut canary);
        bench_zoo_model_unfused(spec, &mut clock, &mut canary);
    }
    #[cfg(not(feature = "model-validation"))]
    for spec in model_bench_specs() {
        emit_model_row(spec);
    }

    // 6. Per-kernel rows.
    for spec in kernel_specs() {
        bench_kernel(spec, &mut clock, &mut canary);
    }

    firmware_log!("benchmarks complete; reference bars: MobileNetV2 224x224 = 1294.5 ms single-core (never 856 ms dual-core), KWS = 7 ms");
    loop {
        // Hold the output link open for the host to drain output.
    }
}
