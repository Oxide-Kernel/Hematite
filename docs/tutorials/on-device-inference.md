---
title: Tutorial — On-Device Inference
---

# Tutorial: On-device inference on the ESP32-S3

This tutorial flashes a Hematite model onto an ESP32-S3. The device runs
the **`S3Backend`** — the bespoke Xtensa TIE728 SIMD kernels.

!!! warning "Prerequisites"

    See [Installation](../installation.md) — you need `espup`, the esp-rs
    Xtensa fork toolchain, and `esptool.py` (flashing an encrypted board
    requires the `--encrypt` path).

## 1. The firmware shape

A Hematite device program is a normal `no_std` esp-hal firmware whose
`main` runs the generated model:

```rust
#![no_std]
#![no_main]

use esp_hal::clock::CpuClock;
use esp_hal::main;
use hematite_codegen::model;
use hematite_s3::S3Backend;

#[model("models/zoo/sine_regression/hello_world_int8.tflite")]
pub struct HelloWorld;

#[main]
fn main() -> ! {
    esp_hal::init(esp_hal::Config::default().with_cpu_clock(CpuClock::max()));

    let mut model = HelloWorld::<S3Backend>::new(S3Backend);
    let input = [0i8; HelloWorld::<S3Backend>::input_len()];

    loop {
        let output = model.predict(&input);
        // ... route output[0] to UART / an LED pattern / a command path ...
    }
}
```

## 2. Building

```sh
source ~/export-esp.sh

cargo build --release -Zbuild-std=core,alloc \
  --target xtensa-esp32s3-none-elf -p your_crate
```

On the `xtensa-esp32s3-none-elf` target, `S3Backend` compiles the
`cfg(all(target_arch = "xtensa", not(feature = "qemu")))` SIMD kernels —
the real ACCX/TIE728 assembly — and dispatches shapes through them.

## 3. Flashing

For a normal (non-encrypted) board:

```sh
espflash flash --monitor target/xtensa-esp32s3-none-elf/release/your_crate
```

For this repo's reference board (ESP32-S3 rev v0.2, **permanently
flash-encrypted** — `SPI_BOOT_CRYPT_CNT=0x7`), `espflash` cannot write;
use the encrypted esptool path:

```sh
esptool.py write_flash --encrypt 0x0 path/to/bin
```

!!! danger "Never write plaintext to an encrypted board"

    A plaintext write will not boot. Always use `--encrypt`. See
    [benchmark methodology](../benchmarks/methodology.md) for the full
    reference pipeline.

## 4. Choosing the input source

The firmware harness (`hematite-benchmarks`) shows the patterns to
reuse:

- **Static input**: a `const` array (the model golden input) — what the
  validation firmware does.
- **Sensor→inference**: copy the sensor buffer into the `[i8; INPUT_LEN]`
  array, call `predict`, act on the output.

`predict_with_scratch` keeps all scratch caller-owned for static-memory
designs:

```rust
let mut out = [0i8; HelloWorld::<S3Backend>::output_len()];
let mut scratch = [0u8; HelloWorld::<S3Backend>::SCRATCH_LEN];
model.predict_with_scratch(&input, &mut out, &mut scratch)
    .expect("scratch sized correctly");
```

## 5. Which backend to use when

| Situation | Backend |
|---|---|
| On ESP32-S3, SIMD wanted | `S3Backend` |
| On ESP32-S3, validate against reference | `S3Backend` vs `RefBackend` — must be bit-equal |
| On host (tests, CI) | either — both compile on host, bit-equal |
| Custom port / exotic op set | your own `KernelBackend` (see [custom-backend](custom-backend.md)) |

## 6. Memory footprint

Device builds are static: weights live in flash (DROM), intermediates in
the generated arena (stack or `ARENA_LEN`), per-op scratch in
`SCRATCH_LEN`. No heap. Models whose arenas exceed the device stack
(person_detect_vww, mobilenet_v2 224×224) need **PSRAM** — see
[memory model](../architecture/memory-model.md) and the benchmark
[SKIP rationale](../benchmarks/zoo-models.md).

Next: [custom-backend](custom-backend.md).