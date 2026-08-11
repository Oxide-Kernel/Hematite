// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! T1.3 arena-vs-stack bit-exactness gate — every model compiled twice:
//! `#[model]` (intermediates in ONE liveness-arena `[i8; ARENA_LEN]` local,
//! indexed at const offsets via `split_at_mut`) vs `#[model_stack]`
//! (per-tensor stack arrays, the pre-T1.3 layout), both run through the
//! `RefBackend` decomposition, asserting element-equal outputs and
//! identical FNV-1a checksums.
//!
//! The two arms share the same fusion schedule and the same per-op /
//! composed param math — only intermediate STORAGE differs, so any output
//! divergence is a bug in the arena borrows (slice overlap or stale data),
//! never in the kernels.

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

fn check_equal(name: &str, arena: &[i8], stack: &[i8]) {
    assert_eq!(
        arena.len(),
        stack.len(),
        "{name}: arena/stack output lengths differ"
    );
    if let Some((i, (a, b))) = arena
        .iter()
        .zip(stack.iter())
        .enumerate()
        .find(|(_, (a, b))| a != b)
    {
        panic!(
            "{name}: arena vs stack diverge at idx {i}: arena={a} stack={b} \
             (fnv arena=0x{:08x} stack=0x{:08x})",
            fnv1a(bytemuck_cast(arena)),
            fnv1a(bytemuck_cast(stack)),
        );
    }
    assert_eq!(fnv1a(bytemuck_cast(arena)), fnv1a(bytemuck_cast(stack)), "{name}: FNV-1a mismatch");
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

mod sine {
    pub mod arena {
        use hematite_codegen::model;
        #[model("../models/sine.tflite")]
        pub struct M;
    }
    pub mod stack {
        use hematite_codegen::model_stack;
        #[model_stack("../models/sine.tflite")]
        pub struct M;
    }
}

mod hello_world {
    pub mod arena {
        use hematite_codegen::model;
        #[model("../models/zoo/sine_regression/hello_world_int8.tflite")]
        pub struct M;
    }
    pub mod stack {
        use hematite_codegen::model_stack;
        #[model_stack("../models/zoo/sine_regression/hello_world_int8.tflite")]
        pub struct M;
    }
}

mod kws {
    pub mod arena {
        use hematite_codegen::model;
        #[model("../models/zoo/keyword_spotting/kws_micro_speech_int8.tflite")]
        pub struct M;
    }
    pub mod stack {
        use hematite_codegen::model_stack;
        #[model_stack("../models/zoo/keyword_spotting/kws_micro_speech_int8.tflite")]
        pub struct M;
    }
}

mod anomaly {
    pub mod arena {
        use hematite_codegen::model;
        #[model("../models/zoo/anomaly_detect/anomaly_detect_int8.tflite")]
        pub struct M;
    }
    pub mod stack {
        use hematite_codegen::model_stack;
        #[model_stack("../models/zoo/anomaly_detect/anomaly_detect_int8.tflite")]
        pub struct M;
    }
}

#[test]
fn sine_arena_equals_stack() {
    // sine has no intermediates: ARENA_LEN == 0 on both arms.
    assert_eq!(sine::arena::ARENA_LEN, 0);
    let arena = sine::arena::Model::<RefBackend>::new(RefBackend)
        .predict(&fill::<{ sine::arena::INPUT_LEN }>());
    let stack = sine::stack::Model::<RefBackend>::new(RefBackend)
        .predict(&fill::<{ sine::stack::INPUT_LEN }>());
    check_equal("sine", &arena, &stack);
}

#[test]
fn hello_world_arena_equals_stack() {
    assert!(hello_world::arena::ARENA_LEN > 0);
    let arena = hello_world::arena::Model::<RefBackend>::new(RefBackend)
        .predict(&fill::<{ hello_world::arena::INPUT_LEN }>());
    let stack = hello_world::stack::Model::<RefBackend>::new(RefBackend)
        .predict(&fill::<{ hello_world::stack::INPUT_LEN }>());
    check_equal("hello_world", &arena, &stack);
}

#[test]
fn kws_arena_equals_stack() {
    assert!(kws::arena::ARENA_LEN > 0);
    let arena = kws::arena::Model::<RefBackend>::new(RefBackend)
        .predict(&fill::<{ kws::arena::INPUT_LEN }>());
    let stack = kws::stack::Model::<RefBackend>::new(RefBackend)
        .predict(&fill::<{ kws::stack::INPUT_LEN }>());
    check_equal("kws_micro_speech", &arena, &stack);
}

#[test]
fn anomaly_arena_equals_stack() {
    assert!(anomaly::arena::ARENA_LEN > 0);
    let arena = anomaly::arena::Model::<RefBackend>::new(RefBackend)
        .predict(&fill::<{ anomaly::arena::INPUT_LEN }>());
    let stack = anomaly::stack::Model::<RefBackend>::new(RefBackend)
        .predict(&fill::<{ anomaly::stack::INPUT_LEN }>());
    check_equal("anomaly_detect", &arena, &stack);
}
