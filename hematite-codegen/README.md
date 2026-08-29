# hematite-codegen

Compile-time **TFLite → straight-line Rust** model compiler for
**Hematite** — a pure-Rust, `no_std` int8 neural-network inference engine
for the ESP32-S3.

`#[model("path.tflite")]` runs at build time:

1. parses the flatbuffer with a hand-rolled byte-offset walker,
2. plans memory (liveness arena → `ARENA_LEN`),
3. selects composed-kernel groups (fusion) via a host-side SIMD
   eligibility mirror,
4. emits typed inference code generic over `B: FusedKernelBackend` —
   `predict` / `predict_with_scratch`, plus `INPUT_LEN`, `OUTPUT_LEN`,
   `SCRATCH_LEN`, `ARENA_LEN`.

The same generated code runs bit-identically on `hematite-ref`
(scalar host) and `hematite-s3` (TIE728 SIMD device). Test-support
variants: `model_unfused`, `model_stack`, `model_unstaged`,
`model_force_t2`.

```rust
use hematite_codegen::model;

#[model("models/sine.tflite")]
pub struct SineModel;
```

Full documentation: <https://hematite.readthedocs.io/>.

License: Apache-2.0