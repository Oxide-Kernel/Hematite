---
title: Tutorial — Writing a Custom Backend
---

# Tutorial: Writing a custom `KernelBackend`

Hematite's inference code is generic over `B: KernelBackend` — every
model is a function over the **trait contract**, not over a concrete
backend. Writing your own backend means implementing that contract. This
tutorial builds a minimal-but-honest backend.

## 1. The trait contract

From `hematite-core`:

```rust
pub trait KernelBackend {
    // Tier0 — core compute
    fn conv2d(&self, input: &[i8], weights: &[i8], bias: &[i32],
              params: &Conv2DParams, output: &mut [i8], scratch: &mut [u8])
        -> Result<(), KernelError>;
    fn fully_connected(&self, /* ... */) -> Result<(), KernelError>;
    // ... depthwise, matmul, pool, softmax, activations, elementwise,
    //     data movement, recurrent, reductions ...

    // Scratch-size associated functions (default 0):
    fn conv2d_scratch_size(params: &Conv2DParams) -> usize { 0 }
    // ...
}
```

Every method carries documented invariants about slice lengths and which
[`KernelError`] variants it may return. Return
`KernelError::Unsupported` for ops you deliberately don't implement —
**never** a silent wrong answer.

## 2. The minimal honest backend

A backend that implements just one op and rejects everything else
explicitly:

```rust
use hematite_core::op_params::*;
use hematite_core::{KernelBackend, KernelError};

/// A deliberately tiny backend: only fully-connected works,
/// everything else is an explicit Unsupported.
pub struct TinyBackend;

impl KernelBackend for TinyBackend {
    fn fully_connected(
        &self,
        input: &[i8],
        weights: &[i8],
        bias: &[i32],
        params: &FullyConnectedParams,
        output: &mut [i8],
        scratch: &mut [u8],
    ) -> Result<(), KernelError> {
        let in_dim = params.input_dim as usize;
        let out_dim = params.output_dim as usize;
        if input.len() != in_dim || weights.len() != out_dim * in_dim
            || bias.len() != out_dim || output.len() != out_dim {
            return Err(KernelError::ShapeMismatch);
        }
        // int8 dot product with TFLM requant semantics — see
        // hematite-int8 for the exact math.
        for oc in 0..out_dim {
            let mut acc: i32 = bias[oc];
            for ic in 0..in_dim {
                acc += (input[ic] as i32) * (weights[oc * in_dim + ic] as i32);
            }
            // ... scale acc by (multiplier >> shift), round, clamp ...
            // output[oc] = <TFLM-exact requantize>;
        }
        Ok(())
    }

    fn matmul(
        &self, _: &[i8], _: &[i8], _: &[i32],
        _: &MatMulParams, _: &mut [i8], _: &mut [u8],
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported)   // honest: not implemented
    }
}
```

!!! note "Required methods need real implementations"

    `KernelBackend` methods have **no default bodies** (except scratch
    sizes). A real backend must implement every method — usually by
    forwarding to a shared kernel library (that's exactly what
    `hematite-ref::RefBackend` does) or returning
    `KernelError::Unsupported` per-op. The sketch above shows the shape;
    copy the per-op math from `hematite-ref` or delegate to it.

## 3. Composed kernels are opt-in

`FusedKernelBackend` is **additive**: it changes nothing about
`KernelBackend`, and every fused method has a documented *decomposition*
— the exact per-op sequence it equals. A backend that implements it by
**forwarding to its own per-op methods is bit-exact by construction**:

```rust
use hematite_core::{FusedKernelBackend, KernelBackend, KernelError};

impl FusedKernelBackend for TinyBackend {
    fn fused_conv2d(&mut self, src, weight, bias, params, dst, scratch)
        -> Result<(), KernelError> {
        // 1. self.conv2d(...)                 — anchor
        // 2. if residual: self.add(...)        — TFLM two-stage rounding
        // 3. if activation: self.relu/relu6/hard_swish(...)
        // Bit-exact vs. the unfused sequence by construction.
        todo!()
    }
    // fused_elementwise_chain, fused_pool_with_fold — same pattern
}
```

## 4. Using your backend with `#[model]`

```rust
#[model("models/sine.tflite")]
pub struct SineModel;

let mut model = SineModel::<TinyBackend>::new(TinyBackend);
let out = model.predict(&[0i8; 1]);
```

The generated code does not care which backend runs it — the call chain
is identical; only the trait implementation differs.

## 5. Design rules for backends

1. **Bit-exactness is the contract.** A backend may be faster or slower,
   but for a given model + input it must produce the same bytes as the
   reference. This is what makes "swap the backend" safe.
2. **`Unsupported` beats a wrong answer.** If a shape/op isn't
   implemented, return `KernelError::Unsupported` — never fabricate.
3. **Honor scratch sizes.** If your SIMD path needs scratch, override the
   `*_scratch_size` associated functions so `predict_with_scratch`
   callers size correctly. (Returning `Ok(false)` from a dispatch is the
   s3 idiom for "gate says scalar fallback" — see
   [Layer 2 — s3 backend](../architecture/layer-2-s3-backend.md).)
4. **Validate slice lengths before touching memory.** Every method
   documents its expected lengths; return `ShapeMismatch` early.

## 6. Where to look for the model implementation

- `hematite-ref` — scalar reference kernels, one module per op family
  (`conv.rs`, `pool.rs`, `fully_connected.rs`, …)
- `hematite-int8` — the quantization math
  (`multiply_by_quantized_multiplier`, `saturating_cast`) shared by all
  backends
- `hematite-s3` — the SIMD backend: eligibility gates, scratch formulas,
  dispatch pattern