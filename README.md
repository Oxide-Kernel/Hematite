# Hematite

**A pure-Rust, `no_std`, int8 neural-network inference engine for the
ESP32-S3** — compile-time TFLite model compilation backed by bespoke
Xtensa TIE728 SIMD kernels, bit-exact against TensorFlow Lite Micro
semantics.

```toml
[dependencies]
hematite-core = "0.1"
hematite-ref = "0.1"
hematite-s3 = "0.1"
hematite-codegen = "0.1"
```

```rust
use hematite_codegen::model;
use hematite_s3::S3Backend;

#[model("models/sine.tflite")]
pub struct SineModel;

fn main() {
    let mut model = SineModel::<S3Backend>::new(S3Backend);
    let output = model.predict(&[0i8; SineModel::<S3Backend>::input_len()]);
    println!("{output:?}");
}
```

## Highlights

- **Bit-exact by construction.** Every SIMD kernel's output equals the
  scalar reference exactly (same `out_fnv` FNV-1a checksum) — verified
  on-device for every kernel row and every zoo-model layer.
- **A compiler, not a kernel library.** `#[model]` reads the `.tflite`
  at build time and emits typed straight-line Rust; you never hand-wire
  kernel calls.
- **100% bespoke SIMD assembly.** ACCX/GPR-accumulator kernels written
  from scratch (including a reverse-engineered QACC depthwise
  read-back) — at the TIE728 MAC-issue floor, and correct where vendor
  kernels saturate their 8-bit lanes.
- **Zero runtime allocation.** No `alloc`, no heap: stack arrays plus a
  compile-time-planned liveness arena. `no_std` in the device path.
- **Honest benchmarks.** Every row carries ISO timestamp + commit +
  full cycles on both stacks; failures are reported, never fabricated.

## Benchmarks (device, ESP32-S3 rev v0.2 @ 240 MHz)

Synthetic end-to-end models vs the standard ESP-NN stack:

| Model | ESP-NN | **Hematite** | Hematite wins |
|---|---|---|---|
| A — 4-layer CNN | 2,630,401 cyc | **1,686,922 cyc** | 1.56× |
| B — MobileNetV2-style 7-layer | 994,782 cyc | **763,105 cyc** | 1.30× |
| C — real MobileNetV2 (SAME + stride-2) | 655,303 cyc | **650,773 cyc** | 1.01× |

Zoo models (post-Phase-20, on device): sine **800**, hello_world
**6,240**, kws **1,787,766**, anomaly **16,986,217** cycles — all
bit-exact vs executed-TFLM goldens. See
[Docs: Benchmarks](https://hematite.readthedocs.io/en/latest/benchmarks/).

## How it differs from ESP-NN

Not just faster — **architecturally different**: a model compiler vs. a
hand-wired kernel library, a stated bit-exactness invariant vs. kernels
that can silently diverge (measured: ESP-NN's anomaly output differs ±1
from its own golden; Hematite matches the executed-TFLM reference
exactly). Read the full story:
[Docs: vs. ESP-NN](https://hematite.readthedocs.io/en/latest/comparison/vs-esp-nn/).

## Documentation

The full documentation site (getting started, tutorials, the
layer-by-layer architecture, benchmarks, methodology, contributing) is
hosted at **[hematite.readthedocs.io](https://hematite.readthedocs.io/)**
(mkdocs source in `docs/`).

| Crate | Layer | Role |
|---|---|---|
| `hematite-core` | L0 | `KernelBackend` / `FusedKernelBackend` contract, op params |
| `hematite-int8` | L0 | TFLM-exact int8 quantization math |
| `hematite-ref` | L1 | scalar reference backend — the golden oracle |
| `hematite-s3` | L2 | ESP32-S3 TIE728 SIMD kernels — the fast path |
| `hematite-memory` | L3 | compile-time liveness arena planner |
| `hematite-codegen` | L3 | `#[model]` proc-macro: TFLite → straight-line Rust |
| `hematite-tests` | L4 | golden-corpus tests |
| `hematite-benchmarks` | L4 | on-device benchmark + validation firmware |

## Building & running

Requires the `espup`-installed esp-rs Xtensa toolchain (for device
builds) — host builds work with a standard Rust toolchain:

```sh
# host tests (no hardware):
cargo test --workspace

# docs site:
pip install -r requirements-docs.txt && mkdocs serve

# device firmware (xtensa toolchain required):
cargo build --release -Zbuild-std=core,alloc \
  --target xtensa-esp32s3-none-elf -p hematite-benchmarks
```

> **Important:** flashing a flash-encrypted board requires
> `esptool.py write_flash --encrypt` — see
> [Docs: Installation](https://hematite.readthedocs.io/en/latest/installation/).

## Engineering history

`PROJECT_LOG.md` documents the full journey — hardware bring-up,
the C-SIMD bit-exact cross-language match, the bespoke ACCX kernels,
the from-silicon QACC read-back, the fast-path optimizations, the
ESP-NN head-to-head, and Phases 0–20.

## License

Apache-2.0. See [LICENSE](LICENSE).