// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Integration tests for the USMP-style arena planner.
//!
//! Covers:
//! - KWS model schedule (peak ≤ 60 KB)
//! - Determinism
//! - Property test: 600 random schedules, no live-tensor overlap
//! - In-place ops
//! - Model I/O exclusion
//! - PSRAM split
//! - Residual-read safety

extern crate hematite_memory;

use hematite_memory::{
    liveness_plan, ArenaPlan, LayoutError, OpInfo, ScratchLayout,
    MAX_IO_PER_OP, OFFSET_NONE,
};

// ── Tiny xorshift PRNG (no external deps) ──────────────────────────────────

struct XorShift {
    state: u64,
}

impl XorShift {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_u16(&mut self, max: u16) -> u16 {
        if max == 0 {
            return 0;
        }
        (self.next() % (max as u64)) as u16
    }

    fn next_size(&mut self, min: usize, max: usize) -> usize {
        let range = max - min + 1;
        min + (self.next() as usize % range)
    }
}

// ── KWS model schedule test ────────────────────────────────────────────────

/// A realistic keyword-spotting model schedule:
/// 8 ops (conv/depthwise/pool/fc/softmax), 10 tensors,
/// peak arena must be ≤ 60 KB for 512 KB SRAM.
#[test]
fn kws_schedule_peak_within_60kb() {
    // Tensor byte sizes (10 tensors: 0=model input, 9=model output)
    let tensor_byte_sizes: [usize; 10] = [
        1960,  // 0: model input (excluded)
        32000, // 1: conv output
        16640, // 2: depthwise output
        8960,  // 3: conv output
        2560,  // 4: depthwise output
        768,   // 5: conv output
        128,   // 6: pool output
        64,    // 7: fc output
        12,    // 8: intermediate
        12,    // 9: model output (excluded)
    ];

    let ops: [OpInfo; 8] = [
        OpInfo { op_kind: 0, input_ids: [0, 0, 0, 0], input_count: 1,
                 output_ids: [1, 0, 0, 0], output_count: 1, in_place: false },
        OpInfo { op_kind: 1, input_ids: [1, 0, 0, 0], input_count: 1,
                 output_ids: [2, 0, 0, 0], output_count: 1, in_place: false },
        OpInfo { op_kind: 2, input_ids: [2, 0, 0, 0], input_count: 1,
                 output_ids: [3, 0, 0, 0], output_count: 1, in_place: false },
        OpInfo { op_kind: 3, input_ids: [3, 0, 0, 0], input_count: 1,
                 output_ids: [4, 0, 0, 0], output_count: 1, in_place: false },
        OpInfo { op_kind: 4, input_ids: [4, 0, 0, 0], input_count: 1,
                 output_ids: [5, 0, 0, 0], output_count: 1, in_place: false },
        OpInfo { op_kind: 5, input_ids: [5, 0, 0, 0], input_count: 1,
                 output_ids: [6, 0, 0, 0], output_count: 1, in_place: false },
        OpInfo { op_kind: 6, input_ids: [6, 0, 0, 0], input_count: 1,
                 output_ids: [7, 0, 0, 0], output_count: 1, in_place: false },
        OpInfo { op_kind: 7, input_ids: [7, 0, 0, 0], input_count: 1,
                 output_ids: [8, 9, 0, 0], output_count: 2, in_place: false },
    ];

    let plan = liveness_plan(
        &ops, &tensor_byte_sizes, &[0], &[9],
        512 * 1024, None,
    ).expect("KWS plan should succeed");

    assert!(
        plan.peak_arena_bytes <= 60_000,
        "peak arena {} exceeds 60KB budget", plan.peak_arena_bytes
    );

    // Model I/O excluded
    assert_eq!(plan.offsets[0], OFFSET_NONE, "model input excluded");
    assert_eq!(plan.offsets[9], OFFSET_NONE, "model output excluded");

    // Intermediates allocated
    assert_ne!(plan.offsets[1], OFFSET_NONE);
    assert_ne!(plan.offsets[8], OFFSET_NONE);

    // No PSRAM needed
    assert!(plan.psram_split.is_none());

    // All arena offsets 16B aligned
    for tid in 1..9 {
        if plan.offsets[tid] != OFFSET_NONE {
            assert_eq!(plan.offsets[tid] % 16, 0,
                "tensor {} offset {} not 16B aligned", tid, plan.offsets[tid]);
        }
    }
}

// ── Determinism ────────────────────────────────────────────────────────────

#[test]
fn deterministic_output() {
    let tensor_sizes: [usize; 5] = [1024, 4096, 2048, 512, 256];
    let ops: [OpInfo; 4] = [
        OpInfo { op_kind: 0, input_ids: [0, 0, 0, 0], input_count: 1,
                 output_ids: [1, 0, 0, 0], output_count: 1, in_place: false },
        OpInfo { op_kind: 1, input_ids: [1, 0, 0, 0], input_count: 1,
                 output_ids: [2, 0, 0, 0], output_count: 1, in_place: false },
        OpInfo { op_kind: 2, input_ids: [2, 0, 0, 0], input_count: 1,
                 output_ids: [3, 0, 0, 0], output_count: 1, in_place: false },
        OpInfo { op_kind: 3, input_ids: [3, 0, 0, 0], input_count: 1,
                 output_ids: [4, 0, 0, 0], output_count: 1, in_place: false },
    ];

    let p1 = liveness_plan(&ops, &tensor_sizes, &[0], &[4], 65536, None).unwrap();
    let p2 = liveness_plan(&ops, &tensor_sizes, &[0], &[4], 65536, None).unwrap();

    assert_eq!(p1.peak_arena_bytes, p2.peak_arena_bytes);
    for tid in 0..5 {
        assert_eq!(p1.offsets[tid], p2.offsets[tid],
            "non-deterministic offset for tensor {}", tid);
    }
}

// ── Property test ──────────────────────────────────────────────────────────

#[test]
fn property_no_live_tensor_overlap() {
    const NUM_ITERS: usize = 600;
    let mut rng = XorShift::new(0xDEAD_BEEF_CAFE_BABE);

    for iter in 0..NUM_ITERS {
        let (plan, ops, n_ops, tensor_sizes, n_tensors, inputs, _ni, outputs, _no) =
            generate_random_plan(&mut rng);

        let n_t = n_tensors;

        for a in 0..n_t {
            if plan.offsets[a] == OFFSET_NONE {
                continue;
            }
            for b in (a + 1)..n_t {
                if plan.offsets[b] == OFFSET_NONE {
                    continue;
                }
                // Check spatial overlap
                let a_start = plan.offsets[a];
                let a_end = a_start.saturating_add(tensor_sizes[a]);
                let b_start = plan.offsets[b];
                let b_end = b_start.saturating_add(tensor_sizes[b]);

                if a_start >= b_end || b_start >= a_end {
                    continue; // no spatial overlap
                }

                // Spatial overlap — check temporal
                let (a_fw, a_lr) = tensor_interval(a, &ops[..n_ops], &inputs[.._ni as usize], &outputs[.._no as usize]);
                let (b_fw, b_lr) = tensor_interval(b, &ops[..n_ops], &inputs[.._ni as usize], &outputs[.._no as usize]);

                // In-place ops legitimately share offsets — check if one is
                // an in-place output of the other
                let is_in_place_related = is_in_place_pair(a, b, &ops[..n_ops]);

                if is_in_place_related {
                    continue; // in-place sharing is allowed
                }

                let temporal = a_fw <= b_lr && b_fw <= a_lr;

                assert!(
                    !temporal,
                    "iter {}: tensors {} and {} overlap spatially ([{}, {}) vs [{}, {})) and temporally ([{}, {}] vs [{}, {}])",
                    iter, a, b,
                    a_start, a_end, b_start, b_end,
                    a_fw, a_lr, b_fw, b_lr,
                );
            }
        }
    }
}

fn tensor_interval(
    tid: usize,
    ops: &[OpInfo],
    inputs: &[u16],
    outputs: &[u16],
) -> (usize, usize) {
    let num_ops = ops.len();
    let mut fw = usize::MAX;
    let mut lr: usize = 0;

    for &id in inputs {
        if id as usize == tid {
            fw = 0;
            break;
        }
    }

    for (op_idx, op) in ops.iter().enumerate() {
        for i in 0..op.output_count as usize {
            if op.output_ids[i] as usize == tid {
                if op_idx < fw {
                    fw = op_idx;
                }
                lr = lr.max(op_idx);
            }
        }
        for i in 0..op.input_count as usize {
            if op.input_ids[i] as usize == tid {
                lr = lr.max(op_idx);
            }
        }
    }

    for &id in outputs {
        if id as usize == tid {
            lr = lr.max(num_ops);
            break;
        }
    }

    // In-place adjustment
    for op in ops {
        if op.in_place && op.output_count > 0 && op.output_ids[0] as usize == tid {
            let in_id = op.input_ids[0] as usize;
            let (in_fw, _) = tensor_interval(in_id, ops, inputs, outputs);
            if in_fw < fw {
                fw = in_fw;
            }
        }
    }

    (fw, lr)
}

fn is_in_place_pair(a: usize, b: usize, ops: &[OpInfo]) -> bool {
    // Build in-place map: output → input
    let mut ip_map = [u16::MAX; 64];
    for op in ops {
        if op.in_place && op.input_count > 0 && op.output_count > 0 {
            let out = op.output_ids[0] as usize;
            let inp = op.input_ids[0] as usize;
            if out < 64 && inp < 64 {
                ip_map[out] = inp as u16;
            }
        }
    }
    let root_a = in_place_root(a, &ip_map);
    let root_b = in_place_root(b, &ip_map);
    root_a == root_b
}

fn in_place_root(mut tid: usize, ip_map: &[u16; 64]) -> usize {
    // Walk up the in-place chain to the root
    let mut seen = 0u64;
    let mask = 1u64 << (tid as u64);
    seen |= mask;
    while tid < 64 && ip_map[tid] != u16::MAX {
        tid = ip_map[tid] as usize;
        let m = 1u64 << (tid as u64);
        if seen & m != 0 {
            break; // cycle guard
        }
        seen |= m;
    }
    tid
}

// ── In-place op test ────────────────────────────────────────────────────────

#[test]
fn in_place_op_reuses_input_offset() {
    let tensor_sizes: [usize; 4] = [1024, 4096, 4096, 128];
    let ops: [OpInfo; 2] = [
        OpInfo { op_kind: 0, input_ids: [0, 0, 0, 0], input_count: 1,
                 output_ids: [1, 0, 0, 0], output_count: 1, in_place: false },
        OpInfo { op_kind: 1, input_ids: [1, 0, 0, 0], input_count: 1,
                 output_ids: [2, 0, 0, 0], output_count: 1, in_place: true },
    ];

    let plan = liveness_plan(&ops, &tensor_sizes, &[0], &[3], 65536, None).unwrap();

    assert_eq!(plan.offsets[1], plan.offsets[2],
        "in-place output must reuse input offset");
    assert_ne!(plan.offsets[1], OFFSET_NONE);
}

// ── Model I/O exclusion ─────────────────────────────────────────────────────

#[test]
fn model_input_output_excluded_from_arena() {
    let tensor_sizes: [usize; 5] = [1024, 4096, 2048, 512, 256];
    let ops: [OpInfo; 3] = [
        OpInfo { op_kind: 0, input_ids: [0, 0, 0, 0], input_count: 1,
                 output_ids: [1, 0, 0, 0], output_count: 1, in_place: false },
        OpInfo { op_kind: 1, input_ids: [1, 0, 0, 0], input_count: 1,
                 output_ids: [2, 3, 0, 0], output_count: 2, in_place: false },
        OpInfo { op_kind: 2, input_ids: [2, 0, 0, 0], input_count: 1,
                 output_ids: [4, 0, 0, 0], output_count: 1, in_place: false },
    ];

    let plan = liveness_plan(&ops, &tensor_sizes, &[0], &[4], 65536, None).unwrap();

    assert_eq!(plan.offsets[0], OFFSET_NONE, "model input excluded");
    assert_eq!(plan.offsets[4], OFFSET_NONE, "model output excluded");
    assert_ne!(plan.offsets[1], OFFSET_NONE, "intermediate allocated");
}

// ── PSRAM split tests ──────────────────────────────────────────────────────

#[test]
fn psram_split_spills_overflow() {
    // Two large tensors, 6KB SRAM budget, 12KB PSRAM
    let tensor_sizes: [usize; 4] = [0, 6000, 5000, 256];
    let ops: [OpInfo; 2] = [
        OpInfo { op_kind: 0, input_ids: [0, 0, 0, 0], input_count: 1,
                 output_ids: [1, 0, 0, 0], output_count: 1, in_place: false },
        OpInfo { op_kind: 1, input_ids: [1, 0, 0, 0], input_count: 1,
                 output_ids: [2, 3, 0, 0], output_count: 2, in_place: false },
    ];

    let plan = liveness_plan(
        &ops, &tensor_sizes, &[0], &[3],
        6000, Some(12000),
    ).unwrap();

    let pool = plan.psram_split.expect("should have PSRAM split");
    assert!(pool.total_bytes > 0, "should have spilled tensors");
    assert!(plan.peak_arena_bytes <= 6000,
        "SRAM peak {} exceeds budget after split", plan.peak_arena_bytes);

    let mut any_psram = false;
    for tid in 0..4 {
        if pool.tensor_mask & (1u64 << tid) != 0 {
            any_psram = true;
            assert_eq!(plan.offsets[tid], OFFSET_NONE,
                "PSRAM tensor {} at OFFSET_NONE", tid);
        }
    }
    assert!(any_psram, "at least one tensor in PSRAM");
}

#[test]
fn psram_split_infeasible_errors() {
    let tensor_sizes: [usize; 3] = [0, 5000, 4000];
    let ops: [OpInfo; 1] = [
        OpInfo { op_kind: 0, input_ids: [0, 0, 0, 0], input_count: 1,
                 output_ids: [1, 2, 0, 0], output_count: 2, in_place: false },
    ];
    let result = liveness_plan(&ops, &tensor_sizes, &[0], &[2], 3000, Some(1000));
    assert!(result.is_err(), "should fail when both budgets exhausted");
}

#[test]
fn psram_no_spill_when_fits() {
    let tensor_sizes: [usize; 4] = [0, 1024, 2048, 256];
    let ops: [OpInfo; 2] = [
        OpInfo { op_kind: 0, input_ids: [0, 0, 0, 0], input_count: 1,
                 output_ids: [1, 0, 0, 0], output_count: 1, in_place: false },
        OpInfo { op_kind: 1, input_ids: [1, 0, 0, 0], input_count: 1,
                 output_ids: [2, 3, 0, 0], output_count: 2, in_place: false },
    ];
    let plan = liveness_plan(&ops, &tensor_sizes, &[0], &[3], 65536, Some(65536)).unwrap();
    assert!(plan.psram_split.is_none(), "no PSRAM when everything fits");
}

// ── Residual-read safety ────────────────────────────────────────────────────

#[test]
fn residual_read_safety() {
    // t1 lives ops 0-2, t2 lives ops 1-1 → overlapping intervals, must not share offset
    let tensor_sizes: [usize; 5] = [0, 1024, 2048, 512, 256];
    let ops: [OpInfo; 3] = [
        OpInfo { op_kind: 0, input_ids: [0, 0, 0, 0], input_count: 1,
                 output_ids: [1, 0, 0, 0], output_count: 1, in_place: false },
        OpInfo { op_kind: 1, input_ids: [0, 0, 0, 0], input_count: 1,
                 output_ids: [2, 0, 0, 0], output_count: 1, in_place: false },
        OpInfo { op_kind: 2, input_ids: [1, 0, 0, 0], input_count: 1,
                 output_ids: [3, 4, 0, 0], output_count: 2, in_place: false },
    ];

    let plan = liveness_plan(&ops, &tensor_sizes, &[0], &[4], 65536, None).unwrap();

    let o1 = plan.offsets[1];
    let o2 = plan.offsets[2];
    assert_ne!(o1, OFFSET_NONE);
    assert_ne!(o2, OFFSET_NONE);

    // They must not spatially overlap
    let t1_end = o1.saturating_add(tensor_sizes[1]);
    let t2_end = o2.saturating_add(tensor_sizes[2]);
    let overlap = !(o1 >= t2_end || o2 >= t1_end);
    assert!(!overlap, "overlapping-liveness tensors must not overlap spatially");
}

// ── Const compatibility ─────────────────────────────────────────────────────

#[test]
fn const_opinfo_construction() {
    const OP: OpInfo = OpInfo {
        op_kind: 42,
        input_ids: [3, 5, 0, 0],
        input_count: 2,
        output_ids: [7, 0, 0, 0],
        output_count: 1,
        in_place: false,
    };
    assert_eq!(OP.op_kind, 42);
    assert_eq!(OP.input_ids[0], 3);
    assert_eq!(OP.input_ids[1], 5);
    assert_eq!(OP.input_count, 2);
    assert_eq!(OP.output_ids[0], 7);
    assert_eq!(OP.output_count, 1);
}

#[test]
fn const_scratch_layout_construction() {
    const SL: ScratchLayout = ScratchLayout::new();
    assert_eq!(SL.peak(), 0);
}

// ── ScratchLayout functional tests ──────────────────────────────────────────

#[test]
fn scratch_layout_allocate_and_peak() {
    let mut sl = ScratchLayout::new();
    assert_eq!(sl.peak(), 0);

    let a = sl.allocate(0, 128, 8).unwrap();
    assert_eq!(a % 16, 0, "16B aligned");
    let b = sl.allocate(a + 128, 64, 32).unwrap();
    assert_eq!(b % 32, 0, "32B aligned");

    assert!(sl.peak() >= a + 128 + 64);
    sl.reset();
    assert_eq!(sl.peak(), 0);
}

#[test]
fn scratch_layout_no_space_error() {
    let mut sl = ScratchLayout::new();
    for _ in 0..64 {
        sl.allocate(0, 16, 16).unwrap();
    }
    assert_eq!(sl.allocate(0, 16, 16), Err(LayoutError::NoSpace));
}

// ── Edge cases ──────────────────────────────────────────────────────────────

#[test]
fn empty_schedule_zero_peak() {
    let plan = liveness_plan(&[], &[], &[], &[], 65536, None).unwrap();
    assert_eq!(plan.peak_arena_bytes, 0);
    assert_eq!(plan.tensor_count, 0);
}

#[test]
fn single_op_schedule() {
    let sizes: [usize; 3] = [0, 1024, 256];
    let ops: [OpInfo; 1] = [
        OpInfo { op_kind: 0, input_ids: [0, 0, 0, 0], input_count: 1,
                 output_ids: [1, 2, 0, 0], output_count: 2, in_place: false },
    ];
    // t2 is model output → excluded; only t1 (1024B) needs arena
    let plan = liveness_plan(&ops, &sizes, &[0], &[2], 65536, None).unwrap();
    assert!(plan.peak_arena_bytes >= 1024);
    assert_eq!(plan.offsets[2], OFFSET_NONE, "model output excluded");
}

#[test]
fn oversized_tensor_errors() {
    let sizes: [usize; 3] = [0, 5000, 256];
    let ops: [OpInfo; 1] = [
        OpInfo { op_kind: 0, input_ids: [0, 0, 0, 0], input_count: 1,
                 output_ids: [1, 2, 0, 0], output_count: 2, in_place: false },
    ];
    assert_eq!(
        liveness_plan(&ops, &sizes, &[0], &[2], 3000, None),
        Err(LayoutError::Oversized)
    );
}

#[test]
fn out_of_budget_errors() {
    // Two simultaneously-live intermediate tensors (both outputs of same op),
    // each 2000 bytes, budget only 3000 → can't fit both
    let sizes: [usize; 4] = [0, 2000, 2000, 256];
    let ops: [OpInfo; 1] = [
        OpInfo { op_kind: 0, input_ids: [0, 0, 0, 0], input_count: 1,
                 output_ids: [1, 2, 0, 0], output_count: 2, in_place: false },
    ];
    let result = liveness_plan(&ops, &sizes, &[0], &[3], 3000, None);
    assert_eq!(result, Err(LayoutError::OutOfBudget),
        "two 2000-byte simultaneous tensors should not fit in 3000-byte budget");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Random plan generator (for property test)
// ═══════════════════════════════════════════════════════════════════════════

const GEN_MAX_TENSORS: usize = 16;
const GEN_MAX_OPS: usize = 12;

fn generate_random_plan(
    rng: &mut XorShift,
) -> (
    ArenaPlan,
    [OpInfo; GEN_MAX_OPS], usize,           // ops, n_ops
    [usize; GEN_MAX_TENSORS], usize,       // sizes, n_tensors
    [u16; 4], u8,                          // inputs, n_inputs
    [u16; 4], u8,                          // outputs, n_outputs
) {
    let mut ops = [OpInfo {
        op_kind: 0, input_ids: [0; MAX_IO_PER_OP], input_count: 0,
        output_ids: [0; MAX_IO_PER_OP], output_count: 0, in_place: false,
    }; GEN_MAX_OPS];
    let mut tensor_sizes = [0usize; GEN_MAX_TENSORS];

    let n_tensors: u16 = (rng.next() % 9 + 4) as u16; // 4-12 tensors
    let n_ops: usize = (rng.next() % 8 + 3) as usize;   // 3-10 ops

    // Byte sizes
    for tid in 1..n_tensors as usize {
        tensor_sizes[tid] = rng.next_size(16, 65536);
    }

    let mut next_out: u16 = 1;

    for op_idx in 0..n_ops {
        let n_inputs: u8 = (rng.next() % 2 + 1) as u8; // 1-2
        let mut in_ids = [0u16; MAX_IO_PER_OP];
        for i in 0..n_inputs as usize {
            let pool = if next_out > 1 { next_out } else { 1 };
            let pick = rng.next_u16(pool);
            in_ids[i] = if pick == 0 && op_idx == 0 { 0 } else { pick.max(1) };
        }

        let n_outputs: u8 = (rng.next() % 2 + 1) as u8;
        let mut out_ids = [0u16; MAX_IO_PER_OP];
        for i in 0..n_outputs as usize {
            out_ids[i] = next_out;
            next_out += 1;
            if next_out >= n_tensors {
                next_out = n_tensors - 1;
            }
        }

        ops[op_idx] = OpInfo {
            op_kind: op_idx as u16,
            input_ids: in_ids,
            input_count: n_inputs,
            output_ids: out_ids,
            output_count: n_outputs,
            in_place: rng.next() % 3 == 0 && n_inputs > 0 && n_outputs > 0,
        };
    }

    // Cap tensor count
    let used_tensors = next_out as usize;
    let nt = used_tensors.max(n_tensors as usize).min(GEN_MAX_TENSORS);

    let inputs = [0u16, 0, 0, 0];
    let last_out = ops[n_ops - 1].output_ids[0].min((nt - 1) as u16);
    let outputs = [last_out, 0, 0, 0];

    let plan = match liveness_plan(
        &ops[..n_ops], &tensor_sizes[..nt], &inputs[..1], &outputs[..1],
        512 * 1024, None,
    ) {
        Ok(p) => p,
        Err(_) => liveness_plan(
            &ops[..n_ops], &tensor_sizes[..nt], &inputs[..1], &outputs[..1],
            1024 * 1024, None,
        ).expect("random plan failed even with 1MB budget"),
    };

    (plan, ops, n_ops, tensor_sizes, nt, inputs, 1, outputs, 1)
}
