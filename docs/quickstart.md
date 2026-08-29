---
title: Quickstart
---

# Quickstart

Compile a TFLite model and run inference — on the **host**, no hardware
needed. This exercises the full compile-time pipeline: flatbuffer parse →
memory planning → op fusion → straight-line codegen → inference.

## 1. A minimal project

```toml
# Cargo.toml
[package]
name = "hematite-hello"
version = "0.1.0"
edition = "2021"

[dependencies]
hematite-core = { path = "../../hematite-core" }
hematite-ref = { path = "../../hematite-ref" }
hematite-codegen = { path = "../../hematite-codegen" }
```

The model file is read **at compile time**; it must exist in the repo.

```rust
// src/main.rs
use hematite_codegen::model;
use hematite_ref::RefBackend;

#[model("models/sine.tflite")]
pub struct SineModel;

fn main() {
    // Predict requires B: FusedKernelBackend — both RefBackend and
    // S3Backend implement it.
    let mut model = SineModel::<RefBackend>::new(RefBackend);

    // The generated API exposes the exact I/O sizes:
    assert_eq!(SineModel::<RefBackend>::input_len(), 1);
    assert_eq!(SineModel::<RefBackend>::output_len(), 1);

    let input = [0i8; 1];
    let output = model.predict(&input);

    println!("sine(0) = {}", output[0]);

    // predict_with_scratch: no heap, caller-provided scratch (SCRATCH_LEN
    // is generated per model):
    let mut out_buf = [0i8; 1];
    let mut scratch = [0u8; SineModel::<RefBackend>::SCRATCH_LEN];
    model.predict_with_scratch(&input, &mut out_buf, &mut scratch)
        .expect("scratch sized correctly");
}
```

!!! note "Receiver is `&mut self` for fused models"

    `predict`/`predict_with_scratch` take `&mut self` when the model has
    composed (fused) groups, and `&self` otherwise — declare the model
    `let mut` to be safe across model variants. The receiver is chosen by
    the macro; `new` is a `const fn`.

## 2. What you get from `#[model]`

For `models/sine.tflite` (a 1→1 fully-connected network) the macro
generates, alongside `SineModel`:

```rust
impl<B: FusedKernelBackend> SineModel<B> {
    pub const fn new(backend: B) -> Self;
    pub const fn input_len() -> usize;               // 1
    pub const fn output_len() -> usize;              // 1
    pub fn predict(&self / &mut self, input: &[i8; INPUT_LEN]) -> [i8; OUTPUT_LEN];
    pub fn predict_with_scratch(
        &self / &mut self,
        input: &[i8; INPUT_LEN],
        output: &mut [i8; OUTPUT_LEN],
        scratch: &mut [u8],
    ) -> Result<(), KernelError>;
}

pub const INPUT_LEN: usize;    // 1
pub const OUTPUT_LEN: usize;   // 1
pub const SCRATCH_LEN: usize;  // macro-time max per-op kernel scratch
pub const ARENA_LEN: usize;    // liveness-planned intermediates peak (0 = per-tensor stack)
```

`predict` internally allocates `[u8; SCRATCH_LEN]` + the arena `[i8;
ARENA_LEN]` on the stack (no heap anywhere). `predict_with_scratch` takes
caller memory for the per-op kernel scratch instead, for tight static-memory
designs.

## 3. Running it

```sh
cargo run
# sine(0) = 0
```

The output byte is **bit-exact** vs. the TFLM golden — `predict` on
`RefBackend` is the scalar reference implementation of TFLite Micro's
int8 semantics.

## 4. Switching to the accelerated backend

Not on an ESP32-S3? The same code compiles on the host with `S3Backend`,
whose SIMD kernels are `cfg`-compiled out → scalar fallback, **bit-identical
output**:

```rust
use hematite_s3::S3Backend;

let mut model = SineModel::<S3Backend>::new(S3Backend);
let output = model.predict(&input);
assert_eq!(output, output); /* same on any backend */
```

## 5. What's next

- **Real models**: try a zoo model — [keyword_spotting](benchmarks/zoo-models.md)
  (depthwise + fc + softmax) exercises far more of the op surface.
- **On-device**: see [On-device inference](tutorials/on-device-inference.md)
  for the ESP32-S3 firmware side.
- **Understand what the macro emits**: [Layer 3 — codegen](architecture/layer-3-codegen.md).