// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! T5.1 — static pins for the fused==unfused equivalence harness (test-only).
//!
//! The runtime harness (tests/fused_equivalence.rs) writes
//! `local-notes/evidence/composed-kernels/fused-equivalence.md`; its per-model
//! static columns (composed group count, T2 group count, final emitted
//! mode) are carried as a manifest in that test file.  THIS module is the
//! tripwire: it re-runs the REAL `fuse()` + `selector::select_kernel`
//! (default AND T2-forced) over the same six zoo models plus the T5.1
//! divergence fixture and asserts every manifest number, so the evidence
//! table can never drift from the compiler's actual decisions.
//!
//! It also pins the fixture's T2 structure — the load-bearing claim of the
//! auto-unfuse proof:
//!
//! * the fixture has exactly ONE fused group: the MAX_POOL anchor with an
//!   absorbed MUL input fold (`requires_verification == true`);
//! * the fold is MUL with `folded_scale = 0.5` (non-identity) and
//!   identity-quant params — the T4.1 mirror gate says SIMD-eligible, so
//!   ONLY the T2 flag stands between it and a composed emission;
//! * the default selector verdict is PerOp (T2 -> per-op, the W5-safe
//!   default), while the T5.1 forced view (T2 gate open) yields
//!   `Composed(PoolFold)` — the exact flip the harness proves at runtime.

#![allow(dead_code)]

use std::path::Path;

use crate::flatbuffer::{self, ParsedModel};
use crate::optimize::fusion::{self, FusedGroup};
use crate::optimize::selector::{self, ComposedKind, GroupSelection};

/// The manifest the evidence table carries — every number asserted below
/// against real `fuse()`/`select_kernel` runs.
pub(crate) struct ModelManifest {
    /// Groups whose default selector verdict is composed.
    pub(crate) composed_count: usize,
    /// Groups tagged `requires_verification` (T2).
    pub(crate) t2_count: usize,
    /// Structural composed-candidate groups (any pattern arm).
    pub(crate) candidate_count: usize,
}

impl ModelManifest {
    pub(crate) fn final_mode(&self) -> &'static str {
        match (self.composed_count, self.candidate_count) {
            (0, _) => "all-per-op",
            (c, cand) if c == cand => "all-composed",
            _ => "partial",
        }
    }
}

const ZOO_MODELS: &[(&str, &str)] = &[
    ("sine", "../models/sine.tflite"),
    ("hello_world", "../models/zoo/sine_regression/hello_world_int8.tflite"),
    ("kws_micro_speech", "../models/zoo/keyword_spotting/kws_micro_speech_int8.tflite"),
    ("anomaly_detect", "../models/zoo/anomaly_detect/anomaly_detect_int8.tflite"),
    ("person_detect", "../models/zoo/person_detect_vww/person_detect_int8.tflite"),
    ("mobilenet_v2", "../models/zoo/mobilenetv2_cls/mobilenet_v2_1.0_224_int8.tflite"),
];

/// The T5.1 divergence fixture (tools/generate_t2_fold_fixture.py) — inside
/// this crate's `tests/` tree, CARGO_MANIFEST_DIR-relative.
const FIXTURE_PATH: &str = "tests/fixtures/t2_fold_divergence_int8.tflite";

fn load_bytes(crate_relative: &str) -> &'static [u8] {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(crate_relative);
    // The bytes must be 'static for `ParsedModel<'a>`; leak the parse buffer
    // (test-only, bounded — the zoo models are a few MB at most).
    let data = std::fs::read(&path).unwrap_or_else(|e| {
        panic!("missing model file {}: {e}", path.display())
    });
    Box::leak(data.into_boxed_slice())
}

fn parse(bytes: &'static [u8]) -> ParsedModel<'static> {
    flatbuffer::parse(bytes).expect("model must parse")
}

/// One group's T2-forced selector verdict — the same view tweak
/// `generate.rs::fused_plan` applies for `#[model_force_t2]` (clone with
/// the flag cleared, `selector::select_kernel` itself untouched).
fn forced_selection(model: &ParsedModel<'_>, group: &FusedGroup) -> selector::Selection {
    let mut view = group.clone();
    view.requires_verification = false;
    selector::select_kernel(model, &view)
}

/// Compute a model's manifest + per-group verdict summary.
fn analyze(bytes: &'static [u8]) -> (ModelManifest, Vec<(usize, bool, GroupSelection, GroupSelection)>) {
    let model = parse(bytes);
    let schedule = fusion::fuse(&model);
    let mut composed = 0usize;
    let mut t2 = 0usize;
    let mut candidates = 0usize;
    let mut rows = Vec::new();
    for (gi, g) in schedule.groups.iter().enumerate() {
        let sel = selector::select_kernel(&model, g);
        let forced = forced_selection(&model, g);
        if sel.kernel != GroupSelection::PerOp {
            composed += 1;
        }
        if g.requires_verification {
            t2 += 1;
        }
        if selector::has_composed_candidate(g) {
            candidates += 1;
        }
        rows.push((gi, g.requires_verification, sel.kernel, forced.kernel));
    }
    (
        ModelManifest {
            composed_count: composed,
            t2_count: t2,
            candidate_count: candidates,
        },
        rows,
    )
}

#[test]
fn zoo_manifest_pins() {
    // Expected manifest — the W0 fused-profile numbers (only mobilenet_v2
    // has composed groups: 10 residual-adds, all T1).  The evidence table
    // carries these; a fuse()/selector change breaks THIS pin first.
    let expected: &[(&str, usize, usize, usize, &str)] = &[
        ("sine", 0, 0, 0, "all-per-op"),
        ("hello_world", 0, 0, 0, "all-per-op"),
        ("kws_micro_speech", 0, 0, 0, "all-per-op"),
        ("anomaly_detect", 0, 0, 0, "all-per-op"),
        ("person_detect", 0, 0, 0, "all-per-op"),
        ("mobilenet_v2", 10, 0, 10, "all-composed"),
    ];
    assert_eq!(ZOO_MODELS.len(), expected.len(), "zoo manifest rows");
    for ((name, rel), (exp_name, exp_composed, exp_t2, exp_cand, exp_mode)) in
        ZOO_MODELS.iter().zip(expected.iter())
    {
        let bytes = load_bytes(rel);
        let (m, rows) = analyze(bytes);
        assert_eq!(name, exp_name);
        assert_eq!(m.composed_count, *exp_composed, "{name}: composed count");
        assert_eq!(m.t2_count, *exp_t2, "{name}: T2 group count");
        assert_eq!(m.candidate_count, *exp_cand, "{name}: candidate count");
        assert_eq!(m.final_mode(), *exp_mode, "{name}: final emitted mode");
        for (gi, is_t2, sel, forced) in &rows {
            assert!(!is_t2, "{name}: group {gi} must not be T2 (no input/requant folds in zoo)");
            assert_eq!(
                sel, forced,
                "{name}: group {gi} — zoo groups are T1, forced view must not differ"
            );
        }
    }
}

#[test]
fn fixture_t2_group_structure_pin() {
    let bytes = load_bytes(FIXTURE_PATH);
    let model = parse(bytes);
    let schedule = fusion::fuse(&model);

    assert_eq!(schedule.groups.len(), 1, "fixture fuses to exactly ONE group");
    let group = &schedule.groups[0];

    // Anchor is the MAX_POOL (op 1); the MUL (op 0) is absorbed as the fold.
    assert_eq!(group.anchor_builtin, 17, "fixture anchor is MAX_POOL_2D");
    assert_eq!(group.absorbed_ops, vec![0], "fixture absorbs the MUL");
    assert_eq!(group.eliminated_tensors, vec![2], "fixture eliminates MUL output");
    assert!(group.requires_verification, "input-fold groups are T2");

    // The fold: MUL with non-identity folded_scale = s_out / s_in = 0.5.
    let fold = group.input_fold.as_ref().expect("fixture group carries the input fold");
    assert_eq!(fold.builtin, 18, "fold is MUL");
    assert!((fold.folded_scale - 0.5).abs() < 1e-6, "folded_scale = 0.5 (non-identity)");

    // Default selector: T2 -> per-op (the W5-safe default the production
    // emit path uses — no composed T2 emission).
    let default_sel = selector::select_kernel(&model, group);
    assert_eq!(
        default_sel.kernel,
        GroupSelection::PerOp,
        "fixture T2 group must resolve per-op by default"
    );
    assert!(
        default_sel.reason.contains("T2 group"),
        "default reason cites the T2 gate: {}",
        default_sel.reason
    );

    // Forced view (T5.1): the T2 gate open -> the structural pool-fold
    // candidate + the T4.1 mirror (pool 2x2/stride-2/SAME, identity-quant
    // fold) engage -> composed.  This is the W5 flip surface.
    let forced_sel = forced_selection(&model, group);
    assert_eq!(
        forced_sel.kernel,
        GroupSelection::Composed(ComposedKind::PoolFold),
        "forced T2 view must select fused_pool_with_fold"
    );
    assert_eq!(
        forced_sel.simd,
        selector::SimdEst::Simd,
        "the mirror must deem the fixture's composed path SIMD-eligible"
    );

    // The divergence source: the MUL carries fused RELU (clamp [0,127] in
    // the per-op emission) which the composed fold materialization drops
    // (fold params carry no activation; decomposition clamps full-range).
    let mul_op = &model.ops()[0];
    assert_eq!(mul_op.builtin_code, 18);
    match &mul_op.options {
        Some(flatbuffer::ParsedOptions::Mul { fused_activation }) => {
            assert_eq!(*fused_activation, 1, "fixture MUL must carry fused RELU");
        }
        other => panic!("expected Mul options, got {other:?}"),
    }
}
