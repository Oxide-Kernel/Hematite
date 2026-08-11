// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! T1.2 fused-vs-unfused equivalence gate — every zoo model compiled twice:
//! `#[model]` (fused emission honoring the T4.2a schedule) vs
//! `#[model_unfused]` (plain per-op emission), both run through the
//! `RefBackend` decomposition, asserting element-equal outputs and identical
//! FNV-1a checksums.
//!
//! * The other five zoo models have zero composed groups: their fused
//!   emission is token-identical to the per-op emission (the T4.2 input
//!   staging applies identically to both arms).  The test still asserts
//!   equality at runtime.
//! * mobilenet_v2_1.0_224 is the ONLY model with composed groups (10
//!   residual-add groups per the W0 profile) — its fused-vs-unfused
//!   equality is the real gate on the composed param derivation.  Its
//!   intermediates are stack locals (~4 MB unfused), so both runs happen on
//!   a dedicated 128 MB-stack thread.
//!
//! `RefBackend::fused_*` decompositions are the exact per-op sequences
//! (hematite-ref/src/fused.rs), so any divergence here is a bug in the
//! emitted composed params, never in the reference.

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
