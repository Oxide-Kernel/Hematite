---
title: Contributing
---

# Contributing

Hematite is open source under Apache-2.0. Contributions of all kinds —
bug reports, benchmarks, docs, code, model coverage, new backends (other
chips!) — are welcome.

## Ground rules

1. **The bit-exact invariant is sacred.** A change that makes any SIMD
   kernel, composed path, or generated emission produce output that
   differs from the scalar reference — for the same model + input — is
   a regression, not an optimization. Every PR that touches kernels
   must run (and pass) the equivalence + golden suites.
2. **Never fabricate benchmark numbers.** Every measured row carries the
   ledger format below. SKIP/FAIL rows are honest records, never
   omissions.
3. **Honest `Unsupported` over silent wrong answers.** New ops can be
   added as `KernelError::Unsupported` and filled in later.
4. **`no_std` + zero allocation is a constraint**, not an accident. No
   `alloc`/`Vec`/`Box` in the device path (static analysis gates this);
   no dynamic shapes.
5. **Docs are code.** Documentation changes that restate behavior (API,
   benchmarks, methodology) must match the code and the ledgers exactly.

## Development setup

```sh
git clone https://github.com/Oxide-Kernel/Hematite.git
cd Hematite

# host tests (no hardware):
cargo test --workspace

# docs site (MkDocs/Material):
python3 -m venv .docs-venv
.docs-venv/bin/pip install -r requirements-docs.txt
.docs-venv/bin/mkdocs serve    # http://127.0.0.1:8000
```

## Where things live

| Area | Location |
|---|---|
| API contract (L0) | `hematite-core`, `hematite-int8` |
| Reference backend (L1) | `hematite-ref` |
| SIMD backend (L2) | `hematite-s3` (Rust + `asm/*.S`) |
| Model compiler (L3) | `hematite-codegen`, `hematite-memory` |
| Golden tests (L4) | `hematite-tests` |
| Device firmware/bench (L4) | `hematite-benchmarks` + `benchmarks/` |
| Docs | `docs/` (this site) + `PROJECT_LOG.md` history |
| Tools | `tools/generate_goldens` (golden corpus), `tools/qemu-runner`, `tools/tflm-goldens` |

See [Architecture](architecture/index.md) for the layer guide.

## Testing before you open a PR

```sh
cargo test --workspace            # unit + integration + goldens (host)
cargo test -p hematite-tests --features hematite-s3   # s3 host scalar fallback
cargo clippy --workspace -- -W clippy::pedantic        # CI runs this
cargo xtensa-check --workspace    # device target check (esp toolchain)
mkdir -p target/evidence && cargo test -p hematite-codegen -- --nocapture
```

New codegen/scratch behavior: run the scratch-parity + fused-equivalence
tests (`hematite-codegen`), and mirror any scratch-formula change in
**both** `hematite-s3` and the codegen mirror (parity-tested).

## Benchmark ledger format

Any new measured claim must be added in this exact shape (it's what
makes the numbers trustworthy):

```text
| <row> | <ISO timestamp> | <commit> | <full Hematite cycles min/med> |
| <full C-stack cycles min/med> | <speedup ratio> | <config: model, tier, freq> |
```

Follow the "same-conditions rule" (identical model file, input bytes,
memory tier, CPU frequency on both stacks). Deltas-only rows are
rejected in review.

## On-device changes

- Needs the esp-rs Xtensa toolchain (`espup`), an ESP32-S3.
- The reference board is flash-encrypted — always
  `esptool.py write_flash --encrypt` (see
  [methodology](benchmarks/methodology.md)).
- No-PSRAM board: PSRAM-requiring model rows must be honest
  `SKIP reason=no-psram rerun_condition=board-with-PSRAM`.

## Docs conventions

- Markdown, MkDocs Material (fenced code, admonitions, tables).
- One page, one job; keep pages under ~150 lines where possible.
- Every claim about numbers links to the ledger row or committed doc.
- Rust code samples must compile against the current API
  (the quickstart/tutorial crates are pinned by CI where feasible).

## Getting help

Open an issue for bugs/benchmarks; discussion in PRs for design. For
anything unclear, ask before building — the invariants above are strict
by design.

Thank you for contributing to Hematite!