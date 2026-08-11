// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! T1.2/T5.1 fused-vs-unfused equivalence harness — every zoo model compiled
//! three ways: `#[model]` (fused emission honoring the T4.2a schedule),
//! `#[model_unfused]` (plain per-op emission), and — for the T5.1 synthetic
//! T2-divergence fixture — `#[model_force_t2]` (the T2 `requires_verification`
//! gate forced open).  All arms run through the `RefBackend` decomposition,
//! asserting element-equal outputs and identical FNV-1a checksums.
//!
//! * The five no-composed zoo models: fused emission is token-identical to
//!   the per-op emission (the T4.2 input staging applies identically to both
//!   arms).  The test still asserts equality at runtime.
//! * mobilenet_v2_1.0_224 is the ONLY zoo model with composed groups (10
//!   residual-add groups per the W0 profile) — its fused-vs-unfused
//!   equality is the real gate on the composed param derivation.  Its
//!   intermediates are stack locals (~4 MB unfused), so both runs happen on
//!   a dedicated 128 MB-stack thread.
//! * The T5.1 fixture (`tests/fixtures/t2_fold_divergence_int8.tflite`) is
//!   MUL(fused RELU) → MAX_POOL: a T2 pool input-fold group whose COMPOSED
//!   emission genuinely diverges from the per-op sequence (the fold
//!   materialization drops the MUL's RELU clamp — the fold params carry no
//!   activation).  The harness proves the W5 flip path end-to-end:
//!   forced-composed → the equivalence check FAILS (never silently
//!   accepted) → auto-unfuse (the default T2→per-op re-emission) → the
//!   model passes element-equal.
//!
//! `RefBackend::fused_*` decompositions are the exact per-op sequences
//! (hematite-ref/src/fused.rs), so any divergence here is a bug in the
//! emitted composed params, never in the reference.
//!
//! The evidence writer ([`fused_equivalence_evidence`]) runs all 6 zoo
//! models + the fixture and writes `local-notes/evidence/composed-kernels/
//! fused-equivalence.md` (CARGO_MANIFEST_DIR-relative, the fused-profile.md
//! precedent); its static manifest columns are pinned against the real
//! `fuse()`/`selector` runs by the in-crate
//! `optimize::equivalence::*` tests.

use hematite_ref::RefBackend;

/// FNV-1a 32-bit checksum (seed 2166136261, prime 16777619) over raw bytes —
/// mirrors model_validation.rs so the numbers are comparable.
fn fnv1a(data: &[u8]) -> u32 {
    let mut h: u32 = 2_166_136_261;
    for &b in data {
        h ^= u32::from(b);
        h = h.wrapping_mul(16_777_619);
    }
    h
}

fn check_equivalence(name: &str, fused: &[i8], unfused: &[i8]) {
    assert_eq!(
        fused.len(),
        unfused.len(),
        "{name}: fused/unfused output lengths differ"
    );
    if let Some((i, (a, b))) = fused
        .iter()
        .zip(unfused.iter())
        .enumerate()
        .find(|(_, (a, b))| a != b)
    {
        panic!(
            "{name}: fused vs unfused diverge at idx {i}: fused={a} unfused={b} (fnv fused=0x{:08x} unfused=0x{:08x})",
            fnv1a(bytemuck_cast(fused)),
            fnv1a(bytemuck_cast(unfused)),
        );
    }
    let fused_fn = fnv1a(bytemuck_cast(fused));
    let unfused_fn = fnv1a(bytemuck_cast(unfused));
    assert_eq!(fused_fn, unfused_fn, "{name}: FNV-1a mismatch");
}

/// Reinterpret `&[i8]` as `&[u8]` for FNV-1a (same bits).
fn bytemuck_cast(v: &[i8]) -> &[u8] {
    // SAFETY: i8 and u8 have identical layout; this is a plain re-view.
    unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) }
}

/// Deterministic int8 fill (covers the full range, never all-zero).
fn fill<const N: usize>() -> [i8; N] {
    let mut a = [0i8; N];
    for (i, x) in a.iter_mut().enumerate() {
        *x = ((i as i32 * 7 + 13) % 251 - 125) as i8;
    }
    a
}

/// Fixture-specific fill (two graph inputs, 256 elements each): input0 all
/// -1, input1 (the MUL operand) all +5 — every fold result is -5, so the
/// per-op arm (RELU clamp [0,127]) pools 0 while the forced-composed arm
/// (fold materialization, full-range clamp) pools -5: guaranteed divergence
/// in every pool window.
fn t2_fixture_fill() -> [i8; 512] {
    let mut a = [0i8; 512];
    for x in a[..256].iter_mut() {
        *x = -1;
    }
    for x in a[256..].iter_mut() {
        *x = 5;
    }
    a
}

// Each macro expansion emits `Model<B>` + `INPUT_LEN`/`OUTPUT_LEN`/
// `SCRATCH_LEN` at module scope, so every model gets a nested fused /
// unfused pair of submodules.

mod sine {
    pub mod fused {
        use hematite_codegen::model;
        #[model("../models/sine.tflite")]
        pub struct M;
    }
    pub mod unfused {
        use hematite_codegen::model_unfused;
        #[model_unfused("../models/sine.tflite")]
        pub struct M;
    }
}

mod hello_world {
    pub mod fused {
        use hematite_codegen::model;
        #[model("../models/zoo/sine_regression/hello_world_int8.tflite")]
        pub struct M;
    }
    pub mod unfused {
        use hematite_codegen::model_unfused;
        #[model_unfused("../models/zoo/sine_regression/hello_world_int8.tflite")]
        pub struct M;
    }
}

mod kws {
    pub mod fused {
        use hematite_codegen::model;
        #[model("../models/zoo/keyword_spotting/kws_micro_speech_int8.tflite")]
        pub struct M;
    }
    pub mod unfused {
        use hematite_codegen::model_unfused;
        #[model_unfused("../models/zoo/keyword_spotting/kws_micro_speech_int8.tflite")]
        pub struct M;
    }
}

mod anomaly {
    pub mod fused {
        use hematite_codegen::model;
        #[model("../models/zoo/anomaly_detect/anomaly_detect_int8.tflite")]
        pub struct M;
    }
    pub mod unfused {
        use hematite_codegen::model_unfused;
        #[model_unfused("../models/zoo/anomaly_detect/anomaly_detect_int8.tflite")]
        pub struct M;
    }
}

mod person_detect {
    pub mod fused {
        use hematite_codegen::model;
        #[model("../models/zoo/person_detect_vww/person_detect_int8.tflite")]
        pub struct M;
    }
    pub mod unfused {
        use hematite_codegen::model_unfused;
        #[model_unfused("../models/zoo/person_detect_vww/person_detect_int8.tflite")]
        pub struct M;
    }
}

mod mobilenet_v2 {
    pub mod fused {
        use hematite_codegen::model;
        #[model("../models/zoo/mobilenetv2_cls/mobilenet_v2_1.0_224_int8.tflite")]
        pub struct M;
    }
    pub mod unfused {
        use hematite_codegen::model_unfused;
        #[model_unfused("../models/zoo/mobilenetv2_cls/mobilenet_v2_1.0_224_int8.tflite")]
        pub struct M;
    }
}

// T5.1 synthetic T2-divergence fixture: MUL(const, fused RELU) -> MAX_POOL
// (tests/fixtures/t2_fold_divergence_int8.tflite, built by
// tools/generate_t2_fold_fixture.py).  Three arms:
//   * fused     — `#[model]` default: the T2 pool-fold group resolves
//                 per-op (T4.2 selector) — the W5-safe emitted mode.
//   * unfused   — `#[model_unfused]` per-op reference.
//   * forced_t2 — `#[model_force_t2]`: the T2 `requires_verification` gate
//                 forced open, so the pool-fold group emits composed —
//                 the W5 flip surface the harness proves.
mod t2_fixture {
    pub mod fused {
        use hematite_codegen::model;
        #[model("tests/fixtures/t2_fold_divergence_int8.tflite")]
        pub struct M;
    }
    pub mod unfused {
        use hematite_codegen::model_unfused;
        #[model_unfused("tests/fixtures/t2_fold_divergence_int8.tflite")]
        pub struct M;
    }
    pub mod forced_t2 {
        use hematite_codegen::model_force_t2;
        #[model_force_t2("tests/fixtures/t2_fold_divergence_int8.tflite")]
        pub struct M;
    }
}

#[test]
fn sine_fused_equals_unfused() {
    let _ = (sine::fused::M, sine::unfused::M);
    let fused = sine::fused::Model::<RefBackend>::new(RefBackend)
        .predict(&fill::<{ sine::fused::INPUT_LEN }>());
    let unfused = sine::unfused::Model::<RefBackend>::new(RefBackend)
        .predict(&fill::<{ sine::unfused::INPUT_LEN }>());
    check_equivalence("sine", &fused, &unfused);
}

#[test]
fn hello_world_fused_equals_unfused() {
    let _ = (hello_world::fused::M, hello_world::unfused::M);
    let fused = hello_world::fused::Model::<RefBackend>::new(RefBackend)
        .predict(&fill::<{ hello_world::fused::INPUT_LEN }>());
    let unfused = hello_world::unfused::Model::<RefBackend>::new(RefBackend)
        .predict(&fill::<{ hello_world::unfused::INPUT_LEN }>());
    check_equivalence("hello_world", &fused, &unfused);
}

#[test]
fn kws_fused_equals_unfused() {
    let _ = (kws::fused::M, kws::unfused::M);
    let fused = kws::fused::Model::<RefBackend>::new(RefBackend)
        .predict(&fill::<{ kws::fused::INPUT_LEN }>());
    let unfused = kws::unfused::Model::<RefBackend>::new(RefBackend)
        .predict(&fill::<{ kws::unfused::INPUT_LEN }>());
    check_equivalence("kws_micro_speech", &fused, &unfused);
}

#[test]
fn anomaly_fused_equals_unfused() {
    let _ = (anomaly::fused::M, anomaly::unfused::M);
    let fused = anomaly::fused::Model::<RefBackend>::new(RefBackend)
        .predict(&fill::<{ anomaly::fused::INPUT_LEN }>());
    let unfused = anomaly::unfused::Model::<RefBackend>::new(RefBackend)
        .predict(&fill::<{ anomaly::unfused::INPUT_LEN }>());
    check_equivalence("anomaly_detect", &fused, &unfused);
}

#[test]
fn person_detect_fused_equals_unfused() {
    let _ = (person_detect::fused::M, person_detect::unfused::M);
    // ~232 KB of intermediate allocas — fits the default test-thread stack.
    let fused = person_detect::fused::Model::<RefBackend>::new(RefBackend)
        .predict(&fill::<{ person_detect::fused::INPUT_LEN }>());
    let unfused = person_detect::unfused::Model::<RefBackend>::new(RefBackend)
        .predict(&fill::<{ person_detect::unfused::INPUT_LEN }>());
    check_equivalence("person_detect", &fused, &unfused);
}

#[test]
fn mobilenet_v2_fused_equals_unfused() {
    let _ = (mobilenet_v2::fused::M, mobilenet_v2::unfused::M);
    // The 10 composed residual-add groups eliminate 10 intermediates, but
    // the remaining intermediates are stack locals summing to ~4 MB per
    // run — spawn a dedicated large-stack thread for both arms.
    let handle = std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(|| {
            let fused = mobilenet_v2::fused::Model::<RefBackend>::new(RefBackend)
                .predict(&fill::<{ mobilenet_v2::fused::INPUT_LEN }>());
            let unfused = mobilenet_v2::unfused::Model::<RefBackend>::new(RefBackend)
                .predict(&fill::<{ mobilenet_v2::unfused::INPUT_LEN }>());
            (fused, unfused)
        })
        .expect("mobilenet_v2 thread spawn");
    let (fused, unfused) = handle.join().expect("mobilenet_v2 thread join");
    check_equivalence("mobilenet_v2_1.0_224", &fused, &unfused);
}

// ---------------------------------------------------------------------------
// T5.1 — T2 auto-unfuse path (synthetic divergence fixture)
// ---------------------------------------------------------------------------

/// First divergent index + values, or `None` when the arms are element-equal.
fn divergence_at(name: &str, a: &[i8], b: &[i8]) -> Option<(usize, i8, i8)> {
    assert_eq!(a.len(), b.len(), "{name}: arm lengths differ");
    a.iter().zip(b.iter()).enumerate().find(|(_, (x, y))| x != y).map(|(i, (x, y))| (i, *x, *y))
}

/// Run the fixture's forced-composed and unfused arms on the harness fill.
fn run_t2_fixture_forced() -> (Vec<i8>, Vec<i8>) {
    let _ = (t2_fixture::forced_t2::M, t2_fixture::unfused::M);
    let mut forced = t2_fixture::forced_t2::Model::<RefBackend>::new(RefBackend);
    let forced_out = forced.predict(&t2_fixture_fill());
    let unfused = t2_fixture::unfused::Model::<RefBackend>::new(RefBackend)
        .predict(&t2_fixture_fill());
    (forced_out.to_vec(), unfused.to_vec())
}

/// (a) The W5 flip arm: with the T2 group FORCED composed, the equivalence
/// check must FAIL — the composed fold materialization drops the MUL's fused
/// RELU clamp (fold params carry no activation), so a window of negative
/// fold results max-pools to 0 per-op but to a negative value composed.
/// The harness catches the divergence — it is never silently accepted.
#[test]
fn t2_fixture_forced_composed_divergence_caught() {
    let (forced, unfused) = run_t2_fixture_forced();
    match divergence_at("t2_fixture forced-composed", &forced, &unfused) {
        Some((i, f, u)) => {
            let ff = fnv1a(bytemuck_cast(&forced));
            let uf = fnv1a(bytemuck_cast(&unfused));
            assert_ne!(
                ff, uf,
                "t2_fixture: fnv must differ when outputs diverge (idx {i}: forced={f} unfused={u})"
            );
        }
        None => panic!(
            "t2_fixture: forced-composed emission MUST diverge from per-op \
             (the T2 pool-fold group must not be silently accepted); \
             fnv forced=0x{:08x} unfused=0x{:08x}",
            fnv1a(bytemuck_cast(&forced)),
            fnv1a(bytemuck_cast(&unfused)),
        ),
    }
}

/// (b) The auto-unfuse recovery + (d) the safe default: the harness
/// re-emits the fixture's T2 group per-op (the default `#[model]`
/// selection — T2→per-op) and the model passes element-equal with identical
/// FNV-1a.  The forced arm must DIFFER from the default arm, proving the
/// default emitted NO composed T2 call.
#[test]
fn t2_fixture_auto_unfuse_recovers_and_default_safe() {
    let _ = t2_fixture::fused::M;
    let fused = t2_fixture::fused::Model::<RefBackend>::new(RefBackend)
        .predict(&t2_fixture_fill());
    let unfused = t2_fixture::unfused::Model::<RefBackend>::new(RefBackend)
        .predict(&t2_fixture_fill());
    // The auto-unfused re-emission == per-op, element-equal.
    check_equivalence("t2_fixture auto-unfused (default fused)", &fused, &unfused);

    let (forced, _) = run_t2_fixture_forced();
    assert_ne!(
        forced, fused,
        "t2_fixture: default fused must NOT contain the composed T2 call \
         (the forced-composed arm diverges, the default per-op arm does not)"
    );
    assert_ne!(
        fnv1a(bytemuck_cast(&forced)),
        fnv1a(bytemuck_cast(&fused)),
        "t2_fixture: forced/default FNV must differ (composed call present only when forced)"
    );
}

// ---------------------------------------------------------------------------
// T5.1 — evidence: local-notes/evidence/composed-kernels/fused-equivalence.md
// ---------------------------------------------------------------------------

/// Static manifest carried by the evidence table — pinned against the real
/// `fuse()`/`selector` runs by the in-crate `optimize::equivalence::*`
/// tests (zoo_manifest_pins / fixture_t2_group_structure_pin).  Columns:
/// (composed groups, T2 groups, composed candidates) under the DEFAULT
/// selector.
const MANIFEST: &[(&str, usize, usize, usize)] = &[
    ("sine", 0, 0, 0),
    ("hello_world", 0, 0, 0),
    ("kws_micro_speech", 0, 0, 0),
    ("anomaly_detect", 0, 0, 0),
    ("person_detect", 0, 0, 0),
    ("mobilenet_v2_1.0_224", 10, 0, 10),
];

fn final_mode(composed: usize, candidates: usize) -> &'static str {
    match (composed, candidates) {
        (0, _) => "all-per-op",
        (c, cand) if c == cand => "all-composed",
        _ => "partial",
    }
}

fn evidence_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("local-notes")
        .join("evidence")
        .join("composed-kernels")
        .join("fused-equivalence.md")
}

/// One zoo-model row measured by the evidence run.
struct ZooRow {
    name: &'static str,
    composed: usize,
    t2: usize,
    candidates: usize,
    fused_fnv: u32,
    unfused_fnv: u32,
    pass: bool,
}

/// (c)+(3) The evidence writer: runs every zoo model fused vs unfused and
/// the fixture across all three arms, then writes
/// `local-notes/evidence/composed-kernels/fused-equivalence.md` — per-model rows
/// (composed group count, per-group pass/fail, auto-unfused groups, final
/// emitted mode) + the fixture row proving the T2 auto-unfuse path.
#[test]
fn fused_equivalence_evidence() {
    // ── Zoo rows (mobilenet_v2 on a large-stack thread) ─────────────────
    let mut zoo = Vec::new();
    for (name, composed, t2, candidates) in MANIFEST {
        let (fused, unfused): (Vec<i8>, Vec<i8>) = match *name {
            "mobilenet_v2_1.0_224" => {
                let h = std::thread::Builder::new()
                    .stack_size(128 * 1024 * 1024)
                    .spawn(|| {
                        let f = mobilenet_v2::fused::Model::<RefBackend>::new(RefBackend)
                            .predict(&fill::<{ mobilenet_v2::fused::INPUT_LEN }>());
                        let u = mobilenet_v2::unfused::Model::<RefBackend>::new(RefBackend)
                            .predict(&fill::<{ mobilenet_v2::unfused::INPUT_LEN }>());
                        (f.to_vec(), u.to_vec())
                    })
                    .expect("mobilenet_v2 evidence thread spawn");
                h.join().expect("mobilenet_v2 evidence thread join")
            }
            "sine" => {
                let f = sine::fused::Model::<RefBackend>::new(RefBackend)
                    .predict(&fill::<{ sine::fused::INPUT_LEN }>());
                let u = sine::unfused::Model::<RefBackend>::new(RefBackend)
                    .predict(&fill::<{ sine::unfused::INPUT_LEN }>());
                (f.to_vec(), u.to_vec())
            }
            "hello_world" => {
                let f = hello_world::fused::Model::<RefBackend>::new(RefBackend)
                    .predict(&fill::<{ hello_world::fused::INPUT_LEN }>());
                let u = hello_world::unfused::Model::<RefBackend>::new(RefBackend)
                    .predict(&fill::<{ hello_world::unfused::INPUT_LEN }>());
                (f.to_vec(), u.to_vec())
            }
            "kws_micro_speech" => {
                let f = kws::fused::Model::<RefBackend>::new(RefBackend)
                    .predict(&fill::<{ kws::fused::INPUT_LEN }>());
                let u = kws::unfused::Model::<RefBackend>::new(RefBackend)
                    .predict(&fill::<{ kws::unfused::INPUT_LEN }>());
                (f.to_vec(), u.to_vec())
            }
            "anomaly_detect" => {
                let f = anomaly::fused::Model::<RefBackend>::new(RefBackend)
                    .predict(&fill::<{ anomaly::fused::INPUT_LEN }>());
                let u = anomaly::unfused::Model::<RefBackend>::new(RefBackend)
                    .predict(&fill::<{ anomaly::unfused::INPUT_LEN }>());
                (f.to_vec(), u.to_vec())
            }
            "person_detect" => {
                let f = person_detect::fused::Model::<RefBackend>::new(RefBackend)
                    .predict(&fill::<{ person_detect::fused::INPUT_LEN }>());
                let u = person_detect::unfused::Model::<RefBackend>::new(RefBackend)
                    .predict(&fill::<{ person_detect::unfused::INPUT_LEN }>());
                (f.to_vec(), u.to_vec())
            }
            other => panic!("unknown manifest row {other}"),
        };
        // The 6/6 gate is enforced here too — evidence never records a lie.
        check_equivalence(name, &fused, &unfused);
        zoo.push(ZooRow {
            name,
            composed: *composed,
            t2: *t2,
            candidates: *candidates,
            fused_fnv: fnv1a(bytemuck_cast(&fused)),
            unfused_fnv: fnv1a(bytemuck_cast(&unfused)),
            pass: true,
        });
    }

    // ── Fixture arms ─────────────────────────────────────────────────────
    let fixture_fused = t2_fixture::fused::Model::<RefBackend>::new(RefBackend)
        .predict(&t2_fixture_fill());
    let fixture_unfused = t2_fixture::unfused::Model::<RefBackend>::new(RefBackend)
        .predict(&t2_fixture_fill());
    check_equivalence("t2_fixture (default fused)", &fixture_fused, &fixture_unfused);
    let (fixture_forced, _) = run_t2_fixture_forced();
    let fixture_diverged = divergence_at("t2_fixture", &fixture_forced, &fixture_unfused).is_some();
    assert!(fixture_diverged, "evidence run must reproduce the forced divergence");

    // ── Render ───────────────────────────────────────────────────────────
    let mut s = String::new();
    s.push_str("# Fused-vs-Unfused Equivalence over the Zoo (T5.1)\n\n");
    s.push_str("Every zoo model compiled fused (`#[model]`) and unfused (`#[model_unfused]`),\n");
    s.push_str("run through `RefBackend`, asserting element-equal int8 outputs + identical\n");
    s.push_str("FNV-1a.  The synthetic T2-divergence fixture proves the auto-unfuse path:\n");
    s.push_str("a T2 group forced composed is caught (never silently accepted), then re-emitted\n");
    s.push_str("per-op.  Auto-generated by `tests/fused_equivalence.rs::fused_equivalence_evidence`;\n");
    s.push_str("the static manifest columns are pinned by the in-crate\n");
    s.push_str("`optimize::equivalence::zoo_manifest_pins` / `fixture_t2_group_structure_pin` tests.\n\n");
    s.push_str("## Per-model table\n\n");
    s.push_str("| model | composed groups | T2 groups | candidates | final emitted mode | fused==unfused | FNV-1a fused | FNV-1a unfused |\n");
    s.push_str("|---|---|---|---|---|---|---|---|\n");
    for r in &zoo {
        s.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | 0x{:08x} | 0x{:08x} |\n",
            r.name,
            r.composed,
            r.t2,
            r.candidates,
            final_mode(r.composed, r.candidates),
            if r.pass { "PASS" } else { "FAIL" },
            r.fused_fnv,
            r.unfused_fnv,
        ));
    }
    s.push('\n');
    s.push_str("Notes:\n");
    s.push_str("- mobilenet_v2 is the only zoo model with composed groups (10 residual-add,\n");
    s.push_str("  all T1 — per the W0 fused-profile); its row is the gate on the composed\n");
    s.push_str("  param derivation.  Zoo models carry ZERO T2 groups (no input/requant folds),\n");
    s.push_str("  so no zoo row can ever auto-unfuse.\n");
    s.push_str("- `final emitted mode`: all-composed = every composed candidate is composed;\n");
    s.push_str("  all-per-op = no composed call; partial = mixed.  Computed from the manifest\n");
    s.push_str("  columns (pinned against `fuse()`/`select_kernel`).\n\n");
    s.push_str("## T2 auto-unfuse proof (synthetic fixture, MUL(fused RELU) → MAX_POOL)\n\n");
    s.push_str("| arm | composed T2 groups | equivalence | auto-unfused | final emitted mode | FNV-1a |\n");
    s.push_str("|---|---|---|---|---|---|\n");
    s.push_str(&format!(
        "| `#[model_force_t2]` (T2 gate forced open) | 1 (pool input-fold) | **FAIL — divergence caught** | yes → per-op re-emission | all-composed (forced) | 0x{:08x} |\n",
        fnv1a(bytemuck_cast(&fixture_forced)),
    ));
    s.push_str(&format!(
        "| `#[model]` (default — T2→per-op) | 0 | PASS | — | all-per-op | 0x{:08x} |\n",
        fnv1a(bytemuck_cast(&fixture_fused)),
    ));
    s.push_str(&format!(
        "| `#[model_unfused]` (reference) | 0 | PASS (reference) | — | all-per-op | 0x{:08x} |\n",
        fnv1a(bytemuck_cast(&fixture_unfused)),
    ));
    s.push('\n');
    s.push_str("The forced arm's composed fold materialization drops the MUL's fused RELU\n");
    s.push_str("clamp (the fold params carry no activation; the decomposition clamps\n");
    s.push_str("full-range), so a window of negative fold results max-pools to 0 per-op but\n");
    s.push_str("negative composed — a genuine divergence the harness catches.  The default\n");
    s.push_str("selector resolves the same group per-op (T4.2 `requires_verification` gate),\n");
    s.push_str("which IS the auto-unfused re-emission — element-equal, FNV-1a identical.\n");

    println!("===== fused-vs-unfused equivalence evidence (T5.1) =====");
    println!("{s}");

    let path = evidence_path();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).expect("evidence dir");
    }
    std::fs::write(&path, s).expect("fused-equivalence.md write");
    println!("wrote {}", path.display());
}
