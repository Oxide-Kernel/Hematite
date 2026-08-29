---
title: Layer 1 — Reference Oracle
---

# Layer 1 — Reference oracle (`hematite-ref`)

The **golden answer**. `RefBackend` is a scalar implementation of every
`KernelBackend` method — straightforward, obviously-correct math that
implements TFLite Micro's int8 semantics exactly.

## What it is

```rust
pub struct RefBackend;      // zero-sized, stateless

impl KernelBackend for RefBackend {
    // every method forwards to a standalone scalar kernel fn
    // (conv.rs, pool.rs, fully_connected.rs, softmax.rs, ...)
}
```

The scalar kernel functions are the *spec*: each op family lives in its
own module (`conv.rs`, `depthwise_conv.rs`, `pool.rs`, `activation.rs`,
`elementwise.rs`, `softmax.rs`, `data_movement.rs`, `reductions.rs`).
`RefBackend` is pure wiring — the math is in the standalone fns.

## What it is for

1. **The correctness oracle.** Every SIMD kernel is validated against it:
   same input → same output bytes, byte-for-byte on the device.
2. **The host/test backend.** `cargo test` runs models through
   `RefBackend` — no hardware, no SIMD; the golden-corpus tests all use
   it.
3. **A reference implementation of `FusedKernelBackend`.** Its fused
   methods forward to its own per-op methods — the canonical demonstration
   that "decompose = bit-exact" (used by the equivalence harness).
4. **The teaching example.** Reading `hematite-ref` is the fastest way to
   understand what each op does in the TFLM int8 convention.

## Standalone kernel surface

Besides the trait impl, the crate exposes the kernels as free functions
(`conv2d`, `depthwise_conv2d`, `softmax`, `matmul`, `pad_op`,
`concat_op`, …) so they can be tested and called directly.

## Design notes (honesty)

- **`Unsupported` is explicit.** Like `S3Backend`, `RefBackend` returns
  `KernelError::Unsupported` for ops the trait signature cannot carry
  (the recurrent ops need fixture-specific fixed-point constants outside
  `LstmParams`/`SvdfParams`/`GruParams`). Those kernels exist and are
  bit-exact-tested via direct free-function calls in
  `hematite-tests/tests/` — but they are not wired through the trait.
- **Pure data movement is inline.** `reshape`/`transpose` have no
  standalone kernel yet; the adapter implements them inline (no
  arithmetic, so correctness is trivial).

## Relationship to the other layers

- Implements the **L0** contract.
- Proves **L2** (`S3Backend`): the two must produce identical bytes.
- Anchors **L4** tests: goldens are asserted against `RefBackend` output.

Next: [Layer 2 — s3 backend](layer-2-s3-backend.md).