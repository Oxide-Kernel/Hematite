// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! T4.2b — liveness-based arena allocation (USMP-style).
//!
//! Builds an [`OpInfo`] schedule from the parsed TFLite graph and delegates
//! the allocation to `hematite-memory`'s [`liveness_plan`] (T1.3) — the
//! single source of truth for the arena algorithm.  This module never
//! re-implements liveness; it only performs graph extraction and contract
//! validation.
//!
//! The baseline without arena allocation is a separate 16-byte-aligned
//! stack array per intermediate tensor.  Reuse through liveness coalescing
//! is what gives the expected 35–60 % SRAM reduction (MobileNetV2-small@96²
//! ≈ 306 KiB → ≈ 180 KiB; KWS micro 20 → 12.5 KiB; CMSIS-NN partial im2col
//! 332 → 133 KiB).
//!
//! # PSRAM pool split (optional)
//!
//! [`plan_arena`] calls [`liveness_plan`] with `psram_budget: None`, so an
//! arena that does not fit SRAM errors with [`LayoutError::OutOfBudget`].
//! To enable the PSRAM pool path, pass `Some(budget)` instead:
//! [`liveness_plan`] spills the largest live tensors to PSRAM and returns an
//! [`ArenaPlan`] whose `psram_split` (`total_bytes` + per-tensor
//! `tensor_mask` bitmask) describes the spilled pool — spilled tensors get
//! [`OFFSET_NONE`] in `offsets`.  The T4.1 emitter can declare a second
//! PSRAM region from that description.
//!
//! # T4.1 wiring
//!
//! [`plan_arena`] is the seam the T4.1 emitter consumes: `ArenaPlan.offsets`
//! maps tensor id → arena byte offset (const-usable — plain `[usize; 64]`
//! entries) and `ArenaPlan.peak_arena_bytes` sizes
//! `static mut ARENA: [u8; …]`.  Model inputs/outputs and constant tensors
//! are excluded by the planner (caller-owned / flash-resident memory), never
//! by this module.
//!
//! This module is not yet wired into `optimize/mod.rs` — the orchestrator's
//! wiring task declares `pub(crate) mod arena;` after all T4.2 passes land.
//! Dead-code warnings are expected until then (same convention as
//! `flatbuffer.rs` / `layout.rs`).
#![allow(dead_code)]

use std::fmt;

use crate::flatbuffer::{ParsedModel, ParsedOp, ParsedTensor, TensorType};
use hematite_memory::{liveness_plan, ArenaPlan, LayoutError, OpInfo, MAX_IO_PER_OP, MAX_TENSORS};

/// SRAM budget for the arena in bytes (512 KiB per plan D1).  Passed as
/// `max_internal` to [`liveness_plan`]; a single tensor larger than this
/// errors with [`LayoutError::Oversized`] when no PSRAM split is enabled.
pub(crate) const MAX_INTERNAL: usize = 512 * 1024;

/// Errors surfaced by [`plan_arena`].
///
/// Codegen-side contract violations (model width beyond the planner's
/// fixed-size arrays) and the planner's own [`LayoutError`] fold into one
/// type so the T4.1 macro can `compile_error!` on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ArenaError {
    /// More tensors than the planner's fixed-size arrays support.
    TooManyTensors { count: usize },
    /// An op has more non-optional inputs than [`MAX_IO_PER_OP`].
    TooManyInputs { op: usize, count: usize },
    /// An op has more outputs than [`MAX_IO_PER_OP`].
    TooManyOutputs { op: usize, count: usize },
    /// The planner rejected the schedule (oversized tensor / out of budget).
    Layout(LayoutError),
}

impl fmt::Display for ArenaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArenaError::TooManyTensors { count } => write!(
                f,
                "arena planner supports at most {MAX_TENSORS} tensors per subgraph, model has {count}"
            ),
            ArenaError::TooManyInputs { op, count } => write!(
                f,
                "op {op} has {count} inputs; arena planner supports at most {MAX_IO_PER_OP} per op"
            ),
            ArenaError::TooManyOutputs { op, count } => write!(
                f,
                "op {op} has {count} outputs; arena planner supports at most {MAX_IO_PER_OP} per op"
            ),
            ArenaError::Layout(err) => write!(f, "arena plan failed: {err:?}"),
        }
    }
}

/// Element size in bytes for a tensor type in the int8 inference arena.
///
/// Unsupported / non-inference types (strings, variants, Int4, unknown)
/// return 0, which the planner filters out of the arena (never allocated);
/// T4.1 rejects those types at emission time.
fn element_size(t: TensorType) -> usize {
    match t {
        TensorType::Float32 | TensorType::Uint32 | TensorType::Int32 | TensorType::Complex64 => 4,
        TensorType::Float16 | TensorType::Int16 | TensorType::Uint16 => 2,
        TensorType::Int8 | TensorType::Uint8 | TensorType::Bool => 1,
        TensorType::Int64 | TensorType::Uint64 | TensorType::Float64 => 8,
        TensorType::Complex128 => 16,
        TensorType::String
        | TensorType::Variant
        | TensorType::Resource
        | TensorType::Int4
        | TensorType::Unknown => 0,
    }
}

/// Byte size of a tensor: shape product × element size.
///
/// A zero or negative dimension yields 0 (zero-size tensors are excluded
/// from the arena by the planner); saturating arithmetic caps runaway
/// products at the planner's `Oversized` error rather than overflowing.
fn tensor_byte_size(t: &ParsedTensor<'_>) -> usize {
    let elems: u64 = t
        .shape
        .iter()
        .fold(1u64, |acc, &dim| if dim <= 0 { 0 } else { acc.saturating_mul(dim as u64) });
    (elems.saturating_mul(element_size(t.tensor_type) as u64)) as usize
}

/// Byte sizes for every tensor in `tensors`, index-aligned with the model.
fn tensor_byte_sizes(tensors: &[ParsedTensor<'_>]) -> Vec<usize> {
    tensors.iter().map(tensor_byte_size).collect()
}

/// Build the [`OpInfo`] schedule for the parsed graph (execution order).
///
/// Contract mapping to the T4.0 IR:
///
/// * `input_ids` / `output_ids` — tensor indices, filtered of `u32::MAX`
///   optional-input placeholders and of dangling references (indices ≥
///   `tensor_count`, which cannot be real tensors).  Both filters keep ids
///   inside the planner's `u16` / `MAX_TENSORS` domain without wraparound.
/// * `in_place` — `true` when `outputs[0] == inputs[0]` (the op overwrites
///   its first input), per the planner's in-place contract.  Residual reads
///   (a tensor consumed by multiple ops) need no marking — the planner's
///   liveness intervals handle them.
/// * `op_kind` — the resolved `builtin_code` (0 when unresolved/custom),
///   opaque to the planner.
fn build_schedule(
    tensors: &[ParsedTensor<'_>],
    ops: &[ParsedOp<'_>],
) -> Result<Vec<OpInfo>, ArenaError> {
    let tensor_count = tensors.len();
    let mut schedule = Vec::with_capacity(ops.len());
    for (op_idx, op) in ops.iter().enumerate() {
        let inputs: Vec<u32> = op
            .inputs
            .iter()
            .copied()
            .filter(|&t| t != u32::MAX && (t as usize) < tensor_count)
            .collect();
        let outputs: Vec<u32> =
            op.outputs.iter().copied().filter(|&t| (t as usize) < tensor_count).collect();
        if inputs.len() > MAX_IO_PER_OP {
            return Err(ArenaError::TooManyInputs { op: op_idx, count: inputs.len() });
        }
        if outputs.len() > MAX_IO_PER_OP {
            return Err(ArenaError::TooManyOutputs { op: op_idx, count: outputs.len() });
        }

        let mut input_ids = [0u16; MAX_IO_PER_OP];
        let mut output_ids = [0u16; MAX_IO_PER_OP];
        for (slot, &t) in inputs.iter().enumerate() {
            input_ids[slot] = t as u16;
        }
        for (slot, &t) in outputs.iter().enumerate() {
            output_ids[slot] = t as u16;
        }

        // In-place: the first output reuses the first input's arena slot.
        let in_place = !inputs.is_empty()
            && !outputs.is_empty()
            && outputs.first() == inputs.first();

        schedule.push(OpInfo {
            op_kind: u16::try_from(op.builtin_code).unwrap_or(0),
            input_ids,
            input_count: inputs.len() as u8,
            output_ids,
            output_count: outputs.len() as u8,
            in_place,
        });
    }
    Ok(schedule)
}

/// Model I/O tensor ids as `u16` (the planner's index domain).
///
/// Indices ≥ `tensor_count` are dangling references (malformed model) and
/// are dropped so a large index cannot wrap into the valid range on the
/// `u16` cast.  Model input/output exclusion itself is the planner's job —
/// the planner forces those tensors to [`OFFSET_NONE`].
fn model_io_ids(indices: &[u32], tensor_count: usize) -> Vec<u16> {
    indices
        .iter()
        .copied()
        .filter(|&t| (t as usize) < tensor_count)
        .map(|t| t as u16)
        .collect()
}

/// Run the arena planner on raw graph pieces.
///
/// Split out of [`plan_arena`] so the in-crate tests can hand-build graphs
/// through the pinned pub(crate) IR types (`ParsedTensor`/`ParsedOp` fields
/// are pub(crate); `ParsedModel`'s are not, so it has no test constructor).
#[allow(clippy::too_many_arguments)]
fn plan_from_pieces(
    tensors: &[ParsedTensor<'_>],
    ops: &[ParsedOp<'_>],
    model_inputs: &[u32],
    model_outputs: &[u32],
    max_internal: usize,
    psram_budget: Option<usize>,
) -> Result<ArenaPlan, ArenaError> {
    let tensor_count = tensors.len();
    if tensor_count > MAX_TENSORS {
        return Err(ArenaError::TooManyTensors { count: tensor_count });
    }

    let schedule = build_schedule(tensors, ops)?;
    let sizes = tensor_byte_sizes(tensors);
    let model_input_ids = model_io_ids(model_inputs, tensor_count);
    let model_output_ids = model_io_ids(model_outputs, tensor_count);

    // `psram_budget: None` — the PSRAM pool split is a documented future
    // path (module docs).  Planner errors propagate verbatim to the macro.
    liveness_plan(&schedule, &sizes, &model_input_ids, &model_output_ids, max_internal, psram_budget)
        .map_err(ArenaError::Layout)
}

/// Compute the arena layout for a parsed model.
///
/// Builds the [`OpInfo`] schedule from the T4.0 IR and calls
/// `hematite-memory`'s [`liveness_plan`].  See the module docs for the
/// PSRAM split path and the T4.1 wiring contract.
pub(crate) fn plan_arena(model: &ParsedModel<'_>) -> Result<ArenaPlan, ArenaError> {
    plan_from_pieces(
        model.tensors(),
        model.ops(),
        model.inputs(),
        model.outputs(),
        MAX_INTERNAL,
        None,
    )
}

/// Compute the arena layout with an explicit SRAM budget and optional PSRAM
/// split.
///
/// Used by the T4.1 emitter's arena entry point ([`emit_model`]'s
/// `predict_with_arena`), which needs a budget large enough that no
/// intermediate spills to PSRAM (`None` = single flat region the caller owns)
/// — the plan's `offsets` are then all valid in one caller-provided arena.
pub(crate) fn plan_arena_internal(
    model: &ParsedModel<'_>,
    max_internal: usize,
    psram_budget: Option<usize>,
) -> Result<ArenaPlan, ArenaError> {
    plan_from_pieces(
        model.tensors(),
        model.ops(),
        model.inputs(),
        model.outputs(),
        max_internal,
        psram_budget,
    )
}

// ---------------------------------------------------------------------------
// Unit tests — in-crate only (proc-macro restriction)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flatbuffer::parse;
    use hematite_memory::OFFSET_NONE;

    const INT8: TensorType = TensorType::Int8;
    const INT32: TensorType = TensorType::Int32;

    fn tensor(name: &'static str, shape: &[i32], ty: TensorType) -> ParsedTensor<'static> {
        ParsedTensor { name, shape: shape.to_vec(), tensor_type: ty, quant: None, buffer_index: 0 }
    }

    fn op(inputs: &[u32], outputs: &[u32]) -> ParsedOp<'static> {
        ParsedOp {
            opcode_index: 0,
            builtin_code: 9, // FULLY_CONNECTED — opaque to the planner
            inputs: inputs.to_vec(),
            outputs: outputs.to_vec(),
            options: None,
            custom_options: &[],
        }
    }

    fn align16(x: usize) -> usize {
        (x + 15) & !15
    }

    /// Independent liveness oracle: recompute each tensor's
    /// `[first_written, last_read]` interval from the schedule and assert no
    /// two simultaneously-live tensors occupy overlapping arena ranges
    /// (16-byte-aligned sizes, mirroring the planner's collision model).
    /// Not used by the production path — a test-only cross-check.
    fn assert_no_live_overlap(
        plan: &ArenaPlan,
        schedule: &[OpInfo],
        sizes: &[usize],
        model_inputs: &[u16],
        model_outputs: &[u16],
    ) {
        let mut fw = [usize::MAX; MAX_TENSORS];
        let mut lr = [0usize; MAX_TENSORS];
        for &id in model_inputs {
            fw[id as usize] = 0;
        }
        for (op_idx, op) in schedule.iter().enumerate() {
            for k in 0..op.input_count as usize {
                lr[op.input_ids[k] as usize] = lr[op.input_ids[k] as usize].max(op_idx);
            }
            for k in 0..op.output_count as usize {
                let t = op.output_ids[k] as usize;
                fw[t] = fw[t].min(op_idx);
                lr[t] = lr[t].max(op_idx);
            }
        }
        for &id in model_outputs {
            lr[id as usize] = lr[id as usize].max(schedule.len());
        }

        let n = plan.tensor_count as usize;
        for a in 0..n {
            for b in (a + 1)..n {
                let (oa, ob) = (plan.offsets[a], plan.offsets[b]);
                if oa == OFFSET_NONE || ob == OFFSET_NONE {
                    continue;
                }
                let (sa, sb) = (
                    align16(sizes.get(a).copied().unwrap_or(0)),
                    align16(sizes.get(b).copied().unwrap_or(0)),
                );
                if sa == 0 || sb == 0 || fw[a] == usize::MAX || fw[b] == usize::MAX {
                    continue;
                }
                let live_together = fw[a] <= lr[b] && fw[b] <= lr[a];
                let ranges_overlap = oa < ob + sb && ob < oa + sa;
                assert!(
                    !(live_together && ranges_overlap),
                    "tensors {a} (offset {oa}, size {sa}, live [{},{}]) and {b} \
                     (offset {ob}, size {sb}, live [{},{}]) overlap",
                    fw[a],
                    lr[a],
                    fw[b],
                    lr[b]
                );
            }
        }
    }

    #[test]
    fn arena_offsets_chain_graph_no_overlap() {
        // t0 (input) → op0 → t1 → op1 → t2 → op2 → t3 (output).  t1 and t2
        // are live simultaneously at op1 (closed intervals [0,1] and [1,2]
        // overlap), so they must NOT share an offset: greedy-by-size places
        // t1 at 0 and t2 at 16 (equal 16-byte sizes, tiebreak by tensor id).
        let tensors = vec![
            tensor("input", &[1, 16], INT8),
            tensor("t1", &[1, 16], INT8),
            tensor("t2", &[1, 16], INT8),
            tensor("output", &[1, 16], INT8),
        ];
        let ops = vec![op(&[0], &[1]), op(&[1], &[2]), op(&[2], &[3])];
        let schedule = build_schedule(&tensors, &ops).expect("schedule builds");
        let sizes = tensor_byte_sizes(&tensors);
        let plan =
            plan_from_pieces(&tensors, &ops, &[0], &[3], MAX_INTERNAL, None).expect("3-op chain fits the budget");

        assert_eq!(plan.offsets[1], 0);
        assert_eq!(plan.offsets[2], 16);
        assert_eq!(plan.peak_arena_bytes, 32);
        assert_eq!(plan.tensor_count, 4);
        assert_no_live_overlap(&plan, &schedule, &sizes, &[0], &[3]);
    }

    #[test]
    fn arena_offsets_odd_sizes_stay_16b_aligned() {
        // Odd byte sizes force the planner to round every allocation up to
        // 16 bytes: t3 (100 B → 112 B aligned) is largest and lands at 0,
        // t2 (20 B → 32 B) collides live-wise with t3 and lands at 112, t1
        // (12 B) is dead before t3 is born and coalesces onto 0.
        let tensors = vec![
            tensor("input", &[1, 16], INT8),
            tensor("t1", &[12], INT8),
            tensor("t2", &[20], INT8),
            tensor("t3", &[100], INT8),
            tensor("output", &[16], INT8),
        ];
        let ops =
            vec![op(&[0], &[1]), op(&[1], &[2]), op(&[2], &[3]), op(&[3], &[4])];
        let plan =
            plan_from_pieces(&tensors, &ops, &[0], &[4], MAX_INTERNAL, None).expect("odd-size graph fits budget");

        for &off in plan.offsets.iter() {
            if off != OFFSET_NONE {
                assert_eq!(off % 16, 0, "offset {off} not 16-byte aligned");
            }
        }
        assert_eq!(plan.peak_arena_bytes % 16, 0);
        assert_eq!(plan.offsets[3], 0);
        assert_eq!(plan.offsets[2], 112);
        assert_eq!(plan.offsets[1], 0);
        assert_eq!(plan.peak_arena_bytes, 144);
    }

    #[test]
    fn arena_offsets_in_place_op_single_slot() {
        // op1 overwrites its input (inputs [1] == outputs [1]): the builder
        // marks it in_place and the tensor gets exactly one arena slot.
        let tensors = vec![
            tensor("input", &[1, 16], INT8),
            tensor("t1", &[1, 16], INT8),
            tensor("output", &[1, 16], INT8),
        ];
        let ops = vec![op(&[0], &[1]), op(&[1], &[1]), op(&[1], &[2])];
        let schedule = build_schedule(&tensors, &ops).expect("schedule builds");

        assert!(schedule[1].in_place, "op1 must be marked in_place");
        assert!(!schedule[0].in_place && !schedule[2].in_place);
        assert_eq!(schedule[1].input_ids[0], 1);
        assert_eq!(schedule[1].output_ids[0], 1);
        assert_eq!(schedule[1].op_kind, 9);

        let plan =
            plan_from_pieces(&tensors, &ops, &[0], &[2], MAX_INTERNAL, None).expect("in-place graph fits budget");
        assert_eq!(plan.offsets[1], 0, "in-place tensor gets exactly one offset");
        assert_eq!(plan.peak_arena_bytes, 16, "no duplicate allocation for the in-place output");
    }

    #[test]
    fn arena_offsets_model_io_stay_out_of_arena() {
        // t0 (model input) is re-read at op2 (residual); t3 (model output).
        // Neither may occupy arena bytes — the planner forces both to
        // OFFSET_NONE.
        let tensors = vec![
            tensor("input", &[1, 16], INT8),
            tensor("t1", &[1, 16], INT8),
            tensor("t2", &[1, 16], INT8),
            tensor("output", &[1, 16], INT8),
        ];
        let ops = vec![op(&[0], &[1]), op(&[1], &[2]), op(&[0, 2], &[3])];
        let plan =
            plan_from_pieces(&tensors, &ops, &[0], &[3], MAX_INTERNAL, None).expect("I/O-exclusion graph fits budget");

        assert_eq!(plan.offsets[0], OFFSET_NONE, "model input never lives in the arena");
        assert_eq!(plan.offsets[3], OFFSET_NONE, "model output never lives in the arena");
        assert_eq!(plan.offsets[1], 0);
        assert_eq!(plan.offsets[2], 16);
    }

    #[test]
    fn arena_offsets_budget_errors_propagate() {
        // Two 300 KiB intermediates live simultaneously (chain intervals
        // [0,1] and [1,2] overlap at op1) → 600 KiB peak > 512 KiB budget →
        // the planner's OutOfBudget surfaces through ArenaError::Layout.
        let tensors = vec![
            tensor("input", &[307_200], INT8),
            tensor("t1", &[307_200], INT8),
            tensor("t2", &[307_200], INT8),
            tensor("output", &[1], INT8),
        ];
        let ops = vec![op(&[0], &[1]), op(&[1], &[2]), op(&[2], &[3])];
        let err = plan_from_pieces(&tensors, &ops, &[0], &[3], MAX_INTERNAL, None)
            .expect_err("600 KiB live peak exceeds the 512 KiB budget");
        assert_eq!(err, ArenaError::Layout(LayoutError::OutOfBudget));
        assert!(err.to_string().contains("arena plan failed"));

        // A single tensor larger than the whole budget → Oversized.
        let tensors = vec![
            tensor("input", &[1], INT8),
            tensor("huge", &[614_400], INT8),
            tensor("output", &[1], INT8),
        ];
        let ops = vec![op(&[0], &[1]), op(&[1], &[2])];
        let err = plan_from_pieces(&tensors, &ops, &[0], &[2], MAX_INTERNAL, None)
            .expect_err("single tensor over the 512 KiB budget");
        assert_eq!(err, ArenaError::Layout(LayoutError::Oversized));
    }

    #[test]
    fn arena_offsets_contract_violations_error_before_planning() {
        // 257 tensors exceed the planner's MAX_TENSORS arrays.
        let many_tensors: Vec<ParsedTensor<'static>> =
            (0..257).map(|_| tensor("t", &[1], INT8)).collect();
        let ops = vec![op(&[0], &[1])];
        let err = plan_from_pieces(&many_tensors, &ops, &[0], &[1], MAX_INTERNAL, None)
            .expect_err("257 tensors exceed MAX_TENSORS");
        assert_eq!(err, ArenaError::TooManyTensors { count: 257 });

        // An op with 5 real inputs exceeds MAX_IO_PER_OP.
        let tensors = vec![
            tensor("input", &[1], INT8),
            tensor("a", &[1], INT8),
            tensor("b", &[1], INT8),
            tensor("c", &[1], INT8),
            tensor("d", &[1], INT8),
            tensor("e", &[1], INT8),
            tensor("output", &[1], INT8),
        ];
        let ops = vec![op(&[1, 2, 3, 4, 5], &[6])];
        let err = plan_from_pieces(&tensors, &ops, &[0], &[6], MAX_INTERNAL, None)
            .expect_err("5-input op exceeds MAX_IO_PER_OP");
        assert_eq!(err, ArenaError::TooManyInputs { op: 0, count: 5 });
        assert!(err.to_string().contains("op 0 has 5 inputs"));
    }

    #[test]
    fn arena_offsets_int32_bias_sized_four_bytes() {
        // INT32 [6] must size to 24 bytes (aligned 32), not 6 bytes: with
        // correct sizing t1 occupies [0,32) and pushes t2 to offset 32
        // (peak 48); a wrong 1-byte element size would give peak 32.
        let tensors = vec![
            tensor("input", &[1, 16], INT8),
            tensor("t1", &[6], INT32),
            tensor("t2", &[1, 16], INT8),
            tensor("output", &[1, 16], INT8),
        ];
        let ops = vec![op(&[0], &[1]), op(&[1], &[2]), op(&[2], &[3])];
        let plan = plan_from_pieces(&tensors, &ops, &[0], &[3], MAX_INTERNAL, None).expect("graph fits budget");
        assert_eq!(plan.offsets[1], 0, "int32 bias sized 4 bytes per element");
        assert_eq!(plan.offsets[2], 32);
        assert_eq!(plan.peak_arena_bytes, 48);
    }

    #[test]
    fn plan_arena_on_real_sine_model() {
        // sine.tflite: one FC op over 4 tensors — input/output (excluded)
        // plus constant weights/bias (never written → excluded).  No live
        // intermediates → empty arena, exercising the real ParsedModel path.
        let bytes = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/sine.tflite"));
        let model = parse(bytes).expect("sine.tflite parses");
        let plan = plan_arena(&model).expect("sine model plans within budget");
        assert_eq!(plan.peak_arena_bytes, 0);
        assert_eq!(plan.tensor_count, 4);
        for off in plan.offsets.iter().take(4) {
            assert_eq!(*off, OFFSET_NONE);
        }
    }
}
