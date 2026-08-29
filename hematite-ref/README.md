# hematite-ref

The **scalar reference backend** for **Hematite** — a pure-Rust, `no_std`
int8 neural-network inference engine for the ESP32-S3.

`RefBackend` implements every `KernelBackend` method with straightforward
scalar math that mirrors TFLite Micro's int8 semantics exactly. It is the
**golden oracle**: every SIMD kernel, composed path, and generated model
is validated against it, bit-for-bit.

- One op family per module (`conv.rs`, `depthwise_conv.rs`, `pool.rs`,
  `activation.rs`, `elementwise.rs`, `softmax.rs`, `data_movement.rs`,
  `reductions.rs`, `fused.rs`).
- Implements `FusedKernelBackend` by decomposition — bit-exact by
  construction (the reference for the composed-kernel equivalence gates).

Full documentation: <https://hematite.readthedocs.io/>.

License: Apache-2.0