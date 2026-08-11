// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! T4.2 staged-vs-unstaged input gate — the graph-input 16B-staging path
//! must be bit-exact.
//!
//! When a model's first kernel is SIMD-eligible the T4.2 selector stages the
//! caller input region into a `#[repr(C, align(16))]` local
//! (`STAGED_INPUT`) before the first call — the caller's slice alignment is
//! unknowable at codegen, and the s3 conv1x1 SIMD path silently falls back
//! to scalar on `in_ptr % 16 != 0` (conv1x1.rs:284-286).  `#[model]`
//! (staged) vs `#[model_unstaged]` (raw caller slice) must produce
//! IDENTICAL outputs: staging only copies bytes, never changes them.
//!
//! All three staging models are FC-first (sine 1 B, hello_world 1 B,
//! anomaly_detect 640 B — the W0 selector-output rows).  kws/person_detect/
//! mobilenet_v2 do NOT stage (first kernel scalar / no SIMD path), so their
//! staged and unstaged arms are structurally identical — checked too.

use hematite_ref::RefBackend;

fn fnv1a(data: &[u8]) -> u32 {
    let mut h: u32 = 2_166_136_261;
    for &b in data {
        h ^= u32::from(b);
        h = h.wrapping_mul(16_777_619);
    }
    h
}

fn check_identical(name: &str, staged: &[i8], unstaged: &[i8]) {
    assert_eq!(
        staged.len(),
        unstaged.len(),
        "{name}: staged/unstaged output lengths differ"
    );
    if let Some((i, (a, b))) = staged
        .iter()
        .zip(unstaged.iter())
        .enumerate()
        .find(|(_, (a, b))| a != b)
    {
        panic!(
            "{name}: staged vs unstaged diverge at idx {i}: staged={a} unstaged={b} (fnv staged=0x{:08x} unstaged=0x{:08x})",
            fnv1a(bytemuck(staged)),
            fnv1a(bytemuck(unstaged)),
        );
    }
    assert_eq!(
        fnv1a(bytemuck(staged)),
        fnv1a(bytemuck(unstaged)),
        "{name}: FNV-1a mismatch"
    );
}

fn bytemuck(v: &[i8]) -> &[u8] {
    // SAFETY: i8 and u8 have identical layout; a plain re-view.
    unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) }
}

fn fill<const N: usize>() -> [i8; N] {
    let mut a = [0i8; N];
    for (i, x) in a.iter_mut().enumerate() {
        *x = ((i as i32 * 7 + 13) % 251 - 125) as i8;
    }
    a
}

// Staged (default `#[model]`) vs unstaged (`#[model_unstaged]`) pairs —
// both arms fused, only the input staging differs.
macro_rules! pair {
    ($mod:ident, $path:literal) => {
        mod $mod {
            pub mod staged {
                use hematite_codegen::model;
                #[model($path)]
                pub struct M;
            }
            pub mod unstaged {
                use hematite_codegen::model_unstaged;
                #[model_unstaged($path)]
                pub struct M;
            }
        }
    };
}

pair!(sine, "../models/sine.tflite");
pair!(hello_world, "../models/zoo/sine_regression/hello_world_int8.tflite");
pair!(anomaly, "../models/zoo/anomaly_detect/anomaly_detect_int8.tflite");
pair!(kws, "../models/zoo/keyword_spotting/kws_micro_speech_int8.tflite");
pair!(person_detect, "../models/zoo/person_detect_vww/person_detect_int8.tflite");
pair!(mobilenet_v2, "../models/zoo/mobilenetv2_cls/mobilenet_v2_1.0_224_int8.tflite");

#[test]
fn sine_staged_equals_unstaged() {
    let _ = (sine::staged::M, sine::unstaged::M);
    let s = sine::staged::Model::<RefBackend>::new(RefBackend)
        .predict(&fill::<{ sine::staged::INPUT_LEN }>());
    let u = sine::unstaged::Model::<RefBackend>::new(RefBackend)
        .predict(&fill::<{ sine::unstaged::INPUT_LEN }>());
    check_identical("sine", &s, &u);
}

#[test]
fn hello_world_staged_equals_unstaged() {
    let _ = (hello_world::staged::M, hello_world::unstaged::M);
    let s = hello_world::staged::Model::<RefBackend>::new(RefBackend)
        .predict(&fill::<{ hello_world::staged::INPUT_LEN }>());
    let u = hello_world::unstaged::Model::<RefBackend>::new(RefBackend)
        .predict(&fill::<{ hello_world::unstaged::INPUT_LEN }>());
    check_identical("hello_world", &s, &u);
}

#[test]
fn anomaly_staged_equals_unstaged() {
    let _ = (anomaly::staged::M, anomaly::unstaged::M);
    let s = anomaly::staged::Model::<RefBackend>::new(RefBackend)
        .predict(&fill::<{ anomaly::staged::INPUT_LEN }>());
    let u = anomaly::unstaged::Model::<RefBackend>::new(RefBackend)
        .predict(&fill::<{ anomaly::unstaged::INPUT_LEN }>());
    check_identical("anomaly_detect", &s, &u);
}

#[test]
fn kws_staged_equals_unstaged() {
    let _ = (kws::staged::M, kws::unstaged::M);
    // No staging decision (first op builtin 22) — both arms identical.
    let s = kws::staged::Model::<RefBackend>::new(RefBackend)
        .predict(&fill::<{ kws::staged::INPUT_LEN }>());
    let u = kws::unstaged::Model::<RefBackend>::new(RefBackend)
        .predict(&fill::<{ kws::unstaged::INPUT_LEN }>());
    check_identical("kws_micro_speech", &s, &u);
}

#[test]
fn person_detect_staged_equals_unstaged() {
    let _ = (person_detect::staged::M, person_detect::unstaged::M);
    // No staging decision (first conv3x3 is scalar per the mirror).
    let s = person_detect::staged::Model::<RefBackend>::new(RefBackend)
        .predict(&fill::<{ person_detect::staged::INPUT_LEN }>());
    let u = person_detect::unstaged::Model::<RefBackend>::new(RefBackend)
        .predict(&fill::<{ person_detect::unstaged::INPUT_LEN }>());
    check_identical("person_detect", &s, &u);
}

#[test]
fn mobilenet_v2_staged_equals_unstaged() {
    let _ = (mobilenet_v2::staged::M, mobilenet_v2::unstaged::M);
    // No staging decision (first op PAD) — 4 MB stack, dedicated thread.
    let handle = std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(|| {
            let s = mobilenet_v2::staged::Model::<RefBackend>::new(RefBackend)
                .predict(&fill::<{ mobilenet_v2::staged::INPUT_LEN }>());
            let u = mobilenet_v2::unstaged::Model::<RefBackend>::new(RefBackend)
                .predict(&fill::<{ mobilenet_v2::unstaged::INPUT_LEN }>());
            (s, u)
        })
        .expect("mobilenet_v2 thread spawn");
    let (s, u) = handle.join().expect("mobilenet_v2 thread join");
    check_identical("mobilenet_v2_1.0_224", &s, &u);
}
