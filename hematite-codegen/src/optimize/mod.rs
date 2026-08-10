// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Compile-time model optimization passes (T4.2).

pub(crate) mod fusion;
pub(crate) mod layout;
pub(crate) mod arena;

// T0.2 — test-only fused-pattern profile over the real zoo models (writes
// local-notes/evidence/composed-kernels/fused-profile.md).  Never compiled outside
// `cargo test` — keeps std/fs out of the proc-macro's non-test build.
#[cfg(test)]
pub(crate) mod profile;
