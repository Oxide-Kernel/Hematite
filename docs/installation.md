---
title: Installation
---

# Installation

Hematite is a Rust workspace of crates. Everything is host-buildable
with a standard Rust toolchain **except** the device firmware, which
needs the Espressif Xtensa toolchain.

!!! note "Two tiers of installation"

    - **Host-only** (recommended first): compile-time model compilation
      + `RefBackend` inference + the full test suite. Works on stable Rust.
    - **Device** (ESP32-S3 firmware): adds the esp-rs Xtensa Rust fork via
      `espup`, an ESP32-S3 board, and `espflash`/`esptool.py`.

## Hardware prerequisites

- An **ESP32-S3** dev board. Hematite has been validated on an ESP32-S3
  rev v0.2, 8 MB flash, **no PSRAM**.
  - Models larger than ~SRAM capacity (person_detect_vww, mobilenet_v2
    224×224) **require PSRAM** — the benchmark suite SKIPs them honestly
    on a no-PSRAM board.
- A USB-serial connection (or USB-JTAG) for flashing and console output.

## 1. Rust toolchain

The workspace pins a toolchain in `rust-toolchain.toml`; the pinned
channel is the **esp-rs Xtensa fork** (`esp`), which is a full Rust
nightly and works for host builds too.

### Host-only (no device)

If you only want host builds, install the pinned toolchain, or override
locally with stable:

```sh
# Use the esp toolchain (pinned in rust-toolchain.toml):
rustup toolchain install esp --profile minimal
# ...or force stable for host-only work (not needed — the fork IS a
# full toolchain; this is just for machines without the fork):
rustup override set stable 2>/dev/null || true
```

### Device builds — `espup` (Espressif Xtensa fork)

For on-device firmware you need the esp-rs fork toolchain:

```sh
cargo install espup
espup install
# then, in every shell that builds device firmware:
source ~/export-esp.sh
```

This installs `xtensa-esp32s3-none-elf` as a built-in target of the `esp`
channel — no `rustup target add` needed.

## 2. Add Hematite to your project

`hematite-core`, `hematite-ref`, and `hematite-s3` are the runtime crates;
`hematite-codegen` is the proc-macro you use directly:

```toml
[dependencies]
hematite-core = "0.1"
hematite-ref = "0.1"
hematite-s3 = "0.1"
hematite-codegen = "0.1"

# For firmware builds (device):
esp-hal = { version = "1.1", default-features = false, features = ["esp32s3", "defmt", "unstable", "rt", "exception-handler"] }
```

!!! note "esp-hal defaults"

    Hematite's workspace uses `esp-hal` with `default-features = false`
    and explicitly re-enables `rt`/`exception-handler`. This drops the
    `float-save-restore` default (ESP32-S3 has no FPU) — important both to
    save flash and to avoid a QEMU double-exception hang (`rur.fcr` with
    `XCHAL_HAVE_FP` misreported). See
    [run-under-qemu](how-to/run-under-qemu.md).

### From this repository

Clone and run the suite (host):

```sh
git clone https://github.com/Oxide-Kernel/Hematite.git
cd Hematite
cargo test --workspace
```

## 3. Board flashing (device)

The firmware lives in `hematite-benchmarks` (benchmark + validation
firmware) — it is not a library, but it is the reference on-device
harness. To flash this repo's board (ESP32-S3 rev v0.2, **permanently
flash-encrypted**):

```sh
source ~/export-esp.sh

# Host tests (no device):
cargo test -p hematite-benchmarks --lib

# Device firmware:
cargo build --release -Zbuild-std=core,alloc \
  --target xtensa-esp32s3-none-elf -p hematite-benchmarks

# A plaintext write will NOT boot on a flash-encrypted board — you MUST
# use the encrypted write path:
esptool.py write_flash --encrypt 0x0 build/bl.bin 0x8000 build/partitions.csv ...
```

!!! warning

    `espflash` has no flash-encryption support. On a permanently
    flash-encrypted board (`SPI_BOOT_CRYPT_CNT` set), a plaintext write
    will not boot. Use `esptool.py write_flash --encrypt` (see
    [benchmark methodology](benchmarks/methodology.md) for the full
    pipeline).

## 4. Verify your install

```sh
cargo test -p hematite-core -p hematite-int8 -p hematite-ref  # semantics + oracle
cargo test -p hematite-codegen                                 # macro + fusion + selector
cargo test -p hematite-s3                                      # s3 kernels (host-compiled)
```

All tests run on the host; the SIMD paths are compiled out on non-Xtensa
by `cfg` gating, and the scalar fallback produces **bit-identical**
output — see [validate-bit-exactness](how-to/validate-bit-exactness.md).

Next: [Quickstart](quickstart.md).