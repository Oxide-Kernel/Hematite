// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Compile-time model optimization passes (T4.2).

pub(crate) mod fusion;
pub(crate) mod layout;
pub(crate) mod arena;
// T4.2 — static rule-tier composed-kernel selector + input-staging decision
// (consumed by the generate.rs emit path and the W0 profile).
pub(crate) mod selector;

// T0.2 — test-only fused-pattern profile over the real zoo models (writes
// local-notes/evidence/composed-kernels/fused-profile.md).  Never compiled outside
// `cargo test` — keeps std/fs out of the proc-macro's non-test build.
#[cfg(test)]
pub(crate) mod profile;

// T5.1 — test-only static pins for the fused==unfused equivalence harness
// (the evidence manifest numbers, the fixture's T2 group structure, and the
// default-vs-forced selector verdicts).  Never compiled outside `cargo test`.
#[cfg(test)]
pub(crate) mod equivalence;
