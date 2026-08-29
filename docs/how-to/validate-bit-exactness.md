---
title: How-To — Validate Bit-Exactness
---

# How to validate bit-exactness

Hematite's central invariant: **for the same model + input, every backend
produces the same output bytes.** This guide explains how that claim is
enforced, so you can trust it — and re-verify it yourself.

## The layers of proof

| Tier | Where | Who runs it |
|---|---|---|
| Host unit tests | `cargo test` (each crate) | CI / anyone |
| Golden corpus | `hematite-tests` — `#[model]` + `assert_bit_exact` vs executed-TFLM goldens | CI / host |
| Fused == unfused | codegen equivalence harness (all 6 zoo models) | CI / host |
| Scratch parity | codegen `SCRATCH_LEN` vs s3 `*_scratch_need` | CI / host |
| Device SIMD sweep | firmware `simd_validation.rs` — SIMD kernel vs scalar oracle per op | on-device |
| Device model validation | `model_validation.rs` — zoo models through `S3Backend` vs ref + golden | on-device |

## What "bit-exact" means here

- **Kernel:** SIMD kernel output == scalar reference output, element
  for element — checked by `out_fnv` (FNV-1a over the output buffer, 32-bit,
  sign-extending) equality AND by element-wise compare in the suites.
- **Model:** `Model::<S3Backend>::predict` output ==
  `Model::<RefBackend>::predict` == the executed-TFLM golden, byte for byte.
- **Fused:** composed-kernel emission == the equivalent per-op
  emission (the T1.2 equivalence gate).

## Re-verifying on the host (no hardware)

```sh
cargo test --workspace
# includes the golden corpus: sine, hello_world, kws, anomaly,
# person_detect, mobilenet_v2 — each asserting bit-exact vs EXPECTED_OUTPUT
```

Every test runs on `RefBackend`; the s3 crate's host-compiled scalar
fallback is exercised by `hematite-tests --features hematite-s3`.

## Re-verifying on the device

The firmware's `run_benchmarks` boots and runs:

1. **model validation** — sine / hello_world / kws / anomaly through
   `Model::<S3Backend>` vs ref + executed-TFLM golden (PASS/FAIL rows
   with FNV).
2. **SIMD correctness sweep** — ~40 per-op checks: each SIMD kernel's
   output vs the scalar oracle (`out_fnv` equal), covering the widened
   gates (dm>1 depthwise, small FC, generic pool, non-identity
   elementwise, extended mean, padded conv1x1).

```text
=== SIMD CORRECTNESS ===
conv1x1 64[1x1]x64: PASS 0x0bea8225 == 0x0bea8225
conv3x3 32x32x64 VALID: PASS 0x0a181085 == 0x0a181085
...
```

## Known, documented divergences (never hidden)

Two classes of output-difference exist in the suite — each documented,
pinned, and *intentionally not* "fixed" away:

1. **pool ±1 LSB (avg-pool fixed-point semantics)** — the generic pool
   shift-based rounding diverges from `round_half_away_zero` by ±1 on
   negative half-even window sums. Device-validated; classified as a
   known-delta class (ref is the spec; the SIMD row reports the ref
   checksum on firmware).
2. **PAD zero-point / rounding class on mv2** — PAD fill semantics and a
   rounding class in the executed-TFLM golden comparison (984/1000
   deltas; 890 PAD-fill |d|≥3, max 60; 94 rounding). On hold pending the
   T10 param-plumbing work; only the **relative** ref↔s3 and
   fused↔unfused claims hold for mv2 — never "bit-exact vs TFLM" for it.

Neither is masked: rows print `FAIL` with got/want + both checksums.

## Proving *your* backend is bit-exact

Implement `KernelBackend` (see
[custom-backend](../tutorials/custom-backend.md)), then:

```rust
use hematite_ref::RefBackend;

// model over the SAME generated code, two backends:
let ref_out = MyModel::<RefBackend>::new(RefBackend).predict(&input);
let your_out = MyModel::<YourBackend>::new(YourBackend).predict(&input);
assert_eq!(ref_out, your_out);   // byte-for-byte
```

That's the whole test — the generated code is backend-agnostic, so
correctness of your backend *is* equality with the reference.

## Related

- [Layer 0 — semantics](../architecture/layer-0-semantics.md) — the contract
- [Layer 4 — validation](../architecture/layer-4-validation.md) — the evidence
- [Benchmark methodology](../benchmarks/methodology.md) — ledger rules