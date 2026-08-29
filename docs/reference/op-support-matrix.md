---
title: Reference — Op Support Matrix
---

# Op support matrix

What every op does and which backends implement it. "SIMD" means the
op has a **bespoke TIE728 SIMD path** in `hematite-s3`; "scalar" means
the op runs the (bit-exact) scalar kernel in the same backend.

Legend: ✅ implemented · ◐ scalar fallback only on s3 · ❌
`KernelError::Unsupported` · — not applicable.

## Core compute

| Op | `RefBackend` | `S3Backend` | Notes |
|---|---|---|---|
| `conv2d` 1×1 | ✅ | ✅ SIMD | channel-padded variant for in_c<16 / non-%16 |
| `conv2d` 3×3 | ✅ | ✅ SIMD | incl. SAME / stride-2 |
| `depthwise_conv2d` | ✅ | ✅ SIMD | dm=1 and dm>1, anytap, single-channel bc1 broadcast |
| `fully_connected` | ✅ | ✅ SIMD | tiny-fc fast path; inline requantize |
| `matmul` | ✅ | ❌ | trait-wired in ref only |

## Pooling & softmax

| Op | `RefBackend` | `S3Backend` | Notes |
|---|---|---|---|
| `average_pool_2d` | ✅ | ✅ SIMD | generic filter/stride/pad gate (±1 known-delta documented) |
| `max_pool_2d` | ✅ | ✅ SIMD | |
| `softmax` | ✅ | ✅ SIMD | TFLM int8 convention |

## Activations

| Op | `RefBackend` | `S3Backend` | Notes |
|---|---|---|---|
| `relu` | ✅ | ✅ SIMD | |
| `relu6` | ✅ | ✅ SIMD | |
| `hard_swish` | ✅ | ✅ SIMD | |
| `sigmoid` | ✅ | ❌ | |
| `tanh` | ✅ | ❌ | |
| `leaky_relu` | ✅ | ❌ | |
| `prelu` | ✅ | ❌ | |

## Elementwise

| Op | `RefBackend` | `S3Backend` | Notes |
|---|---|---|---|
| `add` | ✅ | ✅ SIMD | |
| `mul` | ✅ | ✅ SIMD | |
| `sub` | ✅ | ✅ SIMD | |

## Quantize

| Op | `RefBackend` | `S3Backend` | Notes |
|---|---|---|---|
| `quantize` | ✅ | ❌ | |
| `dequantize` | ✅ | ❌ | |

## Data movement

| Op | `RefBackend` | `S3Backend` | Notes |
|---|---|---|---|
| `reshape` | ✅ | ✅ scalar | |
| `transpose` | ✅ | ✅ scalar | |
| `concat` | ✅ | ✅ scalar | |
| `split` | ✅ | ✅ scalar | |
| `pad` | ✅ | ✅ scalar | zero-point fill is the documented T10 follow-up |
| `slice` | ✅ | ✅ scalar | |
| `resize_nearest` | ✅ | ✅ scalar | |

## Recurrent

| Op | `RefBackend` | `S3Backend` | Notes |
|---|---|---|---|
| `unidirectional_sequence_lstm` | ◐ (kernel + tests, not trait-wired) | ❌ | trait signature can't carry fixture quant consts |
| `svdf` | ◐ (same) | ❌ | |
| `gru` | ◐ (same) | ❌ | |

## Reductions

| Op | `RefBackend` | `S3Backend` | Notes |
|---|---|---|---|
| `mean` | ✅ | ✅ SIMD | extended in_c > 256 |
| `sum` | ✅ | ❌ | |
| `reduce_max` | ✅ | ❌ | |
| `reduce_min` | ✅ | ❌ | |
| `arg_max` | ✅ | ❌ | |
| `arg_min` | ✅ | ❌ | |
| `l2_normalization` | ✅ | ❌ | |

## Composed (fused) ops

| Op | `RefBackend` | `S3Backend` |
|---|---|---|
| `fused_conv2d` | ✅ (decomposed = bit-exact) | ✅ SIMD |
| `fused_elementwise_chain` | ✅ (decomposed) | ✅ SIMD |
| `fused_pool_with_fold` | ✅ (decomposed) | ✅ SIMD |

## Reading the matrix

- **`RefBackend` runs everything the trait can express** — that's its job
  (the golden oracle).
- **`S3Backend` is honest**: ops without an s3 kernel return
  `KernelError::Unsupported` — never a silent wrong answer.
- "scalar" rows on `S3Backend` mean the Shape-flex SIMD coverage doesn't
  extend to that op yet — pure data movement with no arithmetic
  (copy/permute) is scalar by design.

For the authoritative current list, read the `Unsupported` tables in
`hematite-s3/src/backend.rs` and `hematite-ref/src/backend.rs`.