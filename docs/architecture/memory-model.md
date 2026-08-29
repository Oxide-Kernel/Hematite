---
title: Memory Model
---

# Memory model

Hematite is **zero-allocation at runtime**: no `alloc`, no heap, no
`Vec`/`Box` anywhere in the device path (static analysis: zero counts).
All memory is either compile-time constants, stack arrays, or a
compile-time-planned arena.

## Where bytes live

| Kind | Place | Decided by | Example |
|---|---|---|---|
| Weights / biases | **flash (DROM)** — `&'static [i8]` consts | compiler | generated weight arrays |
| Model intermediates | **stack arena** — `[i8; ARENA_LEN]` or per-tensor stack locals | macro-time liveness planner | activations between ops |
| Per-op kernel scratch | **caller scratch** — `[u8; SCRATCH_LEN]` | macro-time max over ops | padded SIMD input copies |
| I/O buffers | caller-provided (`predict_with_scratch`) or stack | generated API | model input/output |
| Device arena (bench/firmware) | static SRAM / PSRAM arena | firmware layout | hematite-benchmarks |

## The arena (`ARENA_LEN`)

`hematite-memory::liveness_plan` runs at **macro time** over the op
schedule (a USMP-style liveness planner, 255 tensors, 64 scratch
blocks). It computes a `ArenaPlan` — offsets per tensor and the peak →
`ARENA_LEN`. Reuse is by liveness: tensors not simultaneously live share
space.

- Models that fit: intermediates live in one `#[repr(C, align(16))]`
  `[i8; ARENA_LEN]` stack local.
- Models that don't (a single tensor exceeds the 512 KiB budget — e.g.
  mobilenet_v2's first 224×224×32 activation ≈ 1.6 MiB): the emitter
  falls back to **per-tensor stack locals** — bit-exact, just larger.

## Scratch (`SCRATCH_LEN`)

Separate from the arena. SIMD kernels stage padded copies of inputs
and weights in scratch; the per-op `*_scratch_size` formulas live in
`hematite-s3` and are **mirrored at macro time** by `hematite-codegen`
(scratch-parity tested). `SCRATCH_LEN` is the max over ops.

```rust
let mut scratch = [0u8; MyModel::SCRATCH_LEN];   // sized by the macro
model.predict_with_scratch(&input, &mut output, &mut scratch)?;
```

## The 16-byte alignment rule

SIMD (`EE.*`) loads require 16-byte-aligned buffers. Generated code
aligns:

- the arena and per-tensor stack locals (`#[repr(C, align(16))]`),
- the graph input staging copy,
- kernel scratch carves (each carve re-aligns to a 16-byte boundary).

## Device memory tiers (firmware)

On the bench firmware, model memory is carved from a **static SRAM
arena** (216 KiB after reserving the s3 wsum cache), falling back to
**PSRAM** when a model doesn't fit:

| Model | Tier on ref board | Why |
|---|---|---|
| sine / hello_world / kws / anomaly | SRAM | fit the arena |
| person_detect_vww | SKIP reason=stack | arena stack budget |
| mobilenet_v2 224×224 | SKIP reason=no-psram | needs ~4 MiB |

A PSRAM probe runs at boot (`psram_probe_range`); rows gate on it and
report honest SKIPs otherwise. See
[benchmarks/zoo-models](../benchmarks/zoo-models.md).

## Weight residency (flash vs SRAM)

Weights are DROM consts in flash. Streaming large weight sets from DROM
is flash-latency-bound: an 80 KiB weight stream measured ~96× slower
from DROM than from SRAM (2.5M vs 26K cycles on a 640→128 fc). This is
the documented root cause of the residual zoo-model gap vs ESP-NN on
this no-PSRAM board, and it's **not a kernel problem** — staging the
same weights into SRAM once (a future `PreparedModel` API; demonstrated
by the fit-model bench as **145× device speedup**) removes it. See
[vs. ESP-NN](../comparison/vs-esp-nn.md).