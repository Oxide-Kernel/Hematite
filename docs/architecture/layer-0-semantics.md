---
title: Layer 0 — Semantics
---

# Layer 0 — Semantics (`hematite-core` + `hematite-int8`)

The foundation. This layer defines **what an int8 inference op means** —
independent of any hardware or kernel implementation.

## `hematite-core`: the contract

**Zero-dependency, `no_std`.** Two trait definitions and the parameter
types that annotate every op.

### `KernelBackend`

The contract every backend implements. Op methods are grouped in
"tiers":

| Tier | Ops |
|---|---|
| Tier 0 — core compute | `conv2d`, `depthwise_conv2d`, `fully_connected`, `matmul` |
| Tier 1 — pool / softmax / activations / elementwise / quantize | `average_pool_2d`, `max_pool_2d`, `softmax`, `relu`, `relu6`, `hard_swish`, `sigmoid`, `tanh`, `leaky_relu`, `prelu`, `add`, `mul`, `sub`, `quantize`, `dequantize` |
| Tier 2 — data movement | `reshape`, `transpose`, `concat`, `split`, `pad`, `slice`, `resize_nearest` |
| Tier 3 — recurrent | `unidirectional_sequence_lstm`, `svdf`, `gru` |
| Tier 4 — reductions | `mean`, `sum`, `reduce_max`, `reduce_min`, `arg_max`, `arg_min`, `l2_normalization` |
| Scratch sizes | `conv2d_scratch_size`, `depthwise_conv2d_scratch_size`, `softmax_scratch_size`, `lstm_scratch_size`, `svdf_scratch_size`, `gru_scratch_size` |

**The rules every method follows:**

1. **Exact slice layouts** are documented per op (NHWC row-major, batch=1
   for conv; `[output_dim][input_dim]` for fc; etc.).
2. **Every method returns `Result<(), KernelError>`** — never panics on
   bad input, never silently produces wrong data:
   - `ShapeMismatch` — slice lengths don't match the params
   - `ScratchTooSmall` — scratch too small for the op
   - `Unsupported` — this backend doesn't implement the op/shape
3. **Scratch-size associated functions default to `0`** so a no-scratch
   backend implements the trait without boilerplate; SIMD backends
   override them.

### `FusedKernelBackend`

The composed-kernel contract (phase-19). **Purely additive** — it sits
alongside `KernelBackend`, and each fused method documents its exact
decomposition into per-op calls:

- `fused_conv2d` = `conv2d` + optional residual `add` + optional
  activation epilogue (one call, one SIMD pass on s3)
- `fused_elementwise_chain` = anchor elementwise op + absorbed steps
  (register-held chains on s3)
- `fused_pool_with_fold` = optional fold (`mul`/`sub`) + `average_pool_2d`
  / `max_pool_2d` + activation

**Implementing it by forwarding to your own per-op methods is bit-exact by
construction** — the decomposition *is* the semantics. See
[custom-backend](../tutorials/custom-backend.md).

### `KernelError`

The single error type: `ShapeMismatch`, `ScratchTooSmall`, `Unsupported`.

### `op_params`

30+ parameter structs mirroring TFLite Micro: `Conv2DParams`,
`DepthwiseConv2DParams`, `FullyConnectedParams`, `MatMulParams`,
`PoolParams`, `SoftmaxParams`, `ActivationParams`, `ElementwiseParams`,
data-movement params, recurrent params, `ReduceParams`, plus the fused
family (`FusedConvParams`, `ElementwiseChainParams`,
`FoldedPoolParams`, `ResidualAddParams`, …) and the composed-kind &
activation enums.

## `hematite-int8`: the math

The small, shared quantization primitives — **the same functions every
backend's requantize epilogue call**:

| Function | Meaning |
|---|---|
| `multiply_by_quantized_multiplier(value, mult, shift)` | TFLM single-rounding 32×32→requantize; **pure 32-bit math** (16-bit limb decomposition) — no i64 software emulation on Xtensa |
| `rounding_divide_by_pot(x, exponent)` | TFLM `RoundingDivideByPOT` |
| `saturating_cast(value)` | TFLM i32→i8 saturation |
| `requantize(acc, params, channel)` | full per-channel requantize for a channel |
| `quantize_multiplier(scale)` | float scale → `(multiplier, shift)` |

Bit-exactness of the 32-bit `multiply_by_quantized_multiplier` vs. the
i64 reference is proven by an exhaustive host test (1.24M samples,
boundary edges, identity/half-round) — see `hematite-int8` tests.

## Why this layer matters

- **Backends are interchangeable** — the model code is generic over
  `B: KernelBackend`; only the implementation differs.
- **Correctness is checkable** — a backend is "right" iff its output
  equals the reference backend's output. That's an equality test, not a
  leap of faith.
- **The contract is the docs** — every slice length and error condition
  lives here, so backends and the codegen can be developed
  independently.

Next: [Layer 1 — Reference](layer-1-reference.md).