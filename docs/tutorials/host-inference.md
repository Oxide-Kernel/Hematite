---
title: Tutorial — Host Inference
---

# Tutorial: Host inference from a TFLite model

This tutorial walks through compiling a real zoo model and running it on
the host with both backends. It builds on the [Quickstart](../quickstart.md)
concepts — here we use a model with depthwise conv, fully-connected and
softmax ops: **keyword spotting** (`kws_micro_speech_int8.tflite`).

!!! info "No hardware is required for this tutorial."

    Everything runs on your host with a standard Rust toolchain.

## 1. Project setup

```toml
[package]
name = "hematite-tutorial-kws"
version = "0.1.0"
edition = "2021"

[dependencies]
hematite-core = { path = "../../hematite-core" }
hematite-ref = { path = "../../hematite-ref" }
hematite-s3 = { path = "../../hematite-s3" }
hematite-codegen = { path = "../../hematite-codegen" }
```

The model file is resolved at **compile time** — keep the `.tflite` in
your repo (or adjust the `#[model]` path).

## 2. Declare and run the model

```rust
use hematite_codegen::model;

#[model("models/zoo/keyword_spotting/kws_micro_speech_int8.tflite")]
pub struct KeywordSpotting;

fn main() {
    // Input is [1, 1960]; output is [1, 4] (yes / no / silent / unknown).
    use hematite_ref::RefBackend;

    let mut model = KeywordSpotting::<RefBackend>::new(RefBackend);
    let input = [0i8; KeywordSpotting::<RefBackend>::input_len()];

    let output = model.predict(&input);
    println!("kws predict → {:?}", output);
}
```

```sh
cargo run
# kws predict → [-127, -127, 63, -127]
```

## 3. The four output dimensions

The KWS model outputs four logit-like values (after softmax, healthy
probabilities). The argmax index is the predicted keyword class:

```rust
let predicted = output.iter().enumerate().max_by_key(|(_, &v)| v).unwrap().0;
match predicted {
    0 => println!("silence"),
    1 => println!("unknown word"),
    2 => println!("** YES **"),
    3 => println!("** NO **"),
    _ => unreachable!(),
}
```

## 4. Verifying bit-exactness with the reference

Hematite's core invariant: **every backend produces bit-identical output
for the same model + input.** The ref backend is the golden oracle:

```rust
use hematite_ref::RefBackend;
use hematite_s3::S3Backend;

let mut ref_model = KeywordSpotting::<RefBackend>::new(RefBackend);
let mut s3_model = KeywordSpotting::<S3Backend>::new(S3Backend);
let input = [0i8; 1960];

let ref_out = ref_model.predict(&input);
let s3_out = s3_model.predict(&input);
assert_eq!(ref_out, s3_out, "backends must agree bit-for-bit");
```

On a host build, `S3Backend`'s SIMD code is compiled out (`cfg`-gated on
`target_arch = "xtensa"`), so it falls back to scalar kernels — the same
code path that runs on device when the SIMD gate rejects a shape. The
invariant holds on real silicon too; see
[validate-bit-exactness](../how-to/validate-bit-exactness.md).

## 5. What's happening under the hood

`cargo build` runs the `#[model]` macro, which:

1. parses the TFLite flatbuffer (hand-rolled byte-offset walker),
2. builds the op schedule,
3. plans a **liveness-based arena** for intermediates
   (`ARENA_LEN`, see [memory model](../architecture/memory-model.md)),
4. applies the **fusion selector** — adjacent ops that can be composed
   into a single `FusedKernelBackend` call are fused (bit-exact by
   construction; see [Layer 3 — codegen](../architecture/layer-3-codegen.md)),
5. emits straight-line Rust: `INPUT_LEN`/`OUTPUT_LEN`, `SCRATCH_LEN`,
   `ARENA_LEN`, and a `Model<B>` wrapper.

No interpreter loop, no dynamic dispatch, no heap.

## 6. Next steps

- [On-device inference](on-device-inference.md) — flash this model to an
  ESP32-S3 and run on the SIMD backend.
- [Custom backend](custom-backend.md) — implement your own
  `KernelBackend` (port, test harness, or exotic op set).