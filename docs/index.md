---
title: Hematite — int8 neural network inference for ESP32-S3
---

# Hematite

**A pure-Rust, `no_std`, int8 neural-network inference engine for the
ESP32-S3**, built from scratch and validated bit-exact against a scalar
reference on real hardware.

Hematite compiles TFLite models at **compile time** into straight-line
Rust, then runs them on **bespoke Xtensa TIE728 SIMD kernels** — no
vendored Espressif kernels, no runtime allocation, no C.

```rust
use hematite_codegen::model;

#[model("models/zoo/sine_regression/hello_world_int8.tflite")]
pub struct MyModel;

fn main() {
    let output = MyModel::<hematite_s3::S3Backend>::new(hematite_s3::S3Backend)
        .predict(&[0i8; 1]);
    assert_eq!(output, models::sine::EXPECTED_OUTPUT);
}
```

## Why Hematite?

- **Bit-exact by construction.** Every SIMD kernel's output equals the
  scalar reference exactly (same `out_fnv` FNV-1a checksum), verified
  on-device for every kernel row and every zoo-model layer.
- **A compiler, not a kernel library.** `#[model]` reads the `.tflite`
  at build time and emits typed inference code; you never hand-wire
  kernel calls.
- **100% bespoke SIMD assembly.** The ACCX/GPR-accumulator kernels are
  written from scratch — including a reverse-engineered QACC depthwise
  read-back — and sit at the TIE728 MAC-issue floor.
- **Zero runtime allocation.** Stack arrays and a compile-time-planned
  arena; static analysis shows no `Vec`/`Box`/`alloc` anywhere in the
  device path.
- **Honest methodology.** Every benchmark row carries ISO timestamp +
  commit ID + full cycles on both stacks; failures are reported, never
  fabricated.

## Where to go next

| I want to… | Go here |
|---|---|
| Understand what Hematite is and its design | [Overview](overview.md) |
| Set up the toolchain and try it | [Installation](installation.md) → [Quickstart](quickstart.md) |
| See a complete minimal project | Host: [Host inference](tutorials/host-inference.md) · Device: [On-device inference](tutorials/on-device-inference.md) |
| Learn the API layer by layer | [Architecture](architecture/index.md) |
| See benchmark numbers and methodology | [Benchmarks](benchmarks/index.md) |
| Understand how it differs from ESP-NN | [vs. ESP-NN](comparison/vs-esp-nn.md) |
| Contribute | [Contributing](contributing.md) |

## Project status

Phase 20: composed-kernels (fusion + shape-flex SIMD + selector) complete;
real-silicon run-1 validated. See [Benchmarks](benchmarks/index.md) for the
current on-device numbers.

## License

Apache-2.0. See [LICENSE](https://github.com/Oxide-Kernel/Hematite/blob/main/LICENSE).