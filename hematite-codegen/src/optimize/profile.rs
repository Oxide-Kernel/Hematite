// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! T0.2 — fused-pattern profile over the 6 real zoo models (test-only).
//!
//! `fuse()` has no proc-macro-exportable surface (E0477 — proc-macro crates
//! can only export macro items), so this profiling harness is an in-crate
//! `#[cfg(test)]` module, declared as `#[cfg(test)] pub(crate) mod profile;`
//! in `optimize/mod.rs`.  It never compiles into the proc-macro's emit path
//! and adds nothing to the public API.
//!
//! The test [`profile_zoo_models`] parses all 6 models with the **same**
//! `flatbuffer::parse` entry point the proc-macro uses (`lib.rs:41-68`
//! `parse_and_emit` → `flatbuffer::parse(&data)`), runs `fusion::fuse`, and
//! tabulates the fusion opportunity per model.  It prints the table to stdout
//! (`cargo test -p hematite-codegen -- --nocapture profile_zoo_models`) and
//! writes the same content via `std::fs` (host-only, mirroring the
//! `tools/generate_goldens` precedent) to
//! `local-notes/evidence/composed-kernels/fused-profile.md` (CARGO_MANIFEST_DIR
//! relative).  The numbers in that file SET the plan's wave-6 speed targets
//! (T6.2/T6.3), so every column is computed from the real `fuse()` runs — no
//! hand-typed table.
//!
//! # Arena numbers
//!
//! * **Unfused** — `arena::plan_arena(&model)` (arena.rs:238): the real
//!   `hematite-memory::liveness_plan` over the original op schedule.
//! * **Fused estimate** — the SAME planner run over a **reduced schedule**
//!   derived from the [`FusedSchedule`]: one synthetic op per group whose
//!   reads = the group's effective inputs (folds already substituted) plus
//!   the residual tensor and every elementwise-chain operand, and whose write
//!   = the group's output tensor.  Eliminated tensors are produced/consumed
//!   by no op in the reduced schedule, so the planner excludes them
//!   naturally — this is exactly the op list T1.2 will thread through
//!   `emit_model`, not a hand-waved subtraction.  The OpInfo contract mapping
//!   (filter `u32::MAX`/dangling, `u16` casts, `in_place`) mirrors
//!   `arena.rs:143-188` (`build_schedule`) verbatim; keep the two in sync.
//!
//! # SIMD-eligibility column
//!
//! Labeled **APPROX pending T4.1** — the parity-tested host eligibility
//! mirror (T4.1, wave 4) does not exist yet.  This module uses a rough
//! per-anchor estimate from the known s3 gates, each cell citing its gate
//! source (accx.rs:64-89, pool.rs:869-902, elementwise.rs:606-635,
//! softmax.rs:383-387).  Do not treat it as the T4.1 mirror.

#![allow(dead_code)]

use std::path::Path;

use crate::flatbuffer::{self, ParsedModel, ParsedOptions, ParsedTensor, TensorType};
use hematite_memory::{liveness_plan, ArenaPlan, OpInfo, MAX_IO_PER_OP, MAX_TENSORS};

use super::arena::{self, ArenaError};
use super::fusion::{fuse, FusedGroup, FusedSchedule};

// BuiltinOperator codes — fusion.rs's consts are private; re-declared here
// (values from the vendored v23.1-era schema, verified by T4.0).
const ADD: i32 = 0;
const AVERAGE_POOL_2D: i32 = 1;
const CONV_2D: i32 = 3;
const DEPTHWISE_CONV_2D: i32 = 4;
const FULLY_CONNECTED: i32 = 9;
const MAX_POOL_2D: i32 = 17;
const MUL: i32 = 18;
const SOFTMAX: i32 = 25;
const SUB: i32 = 41;

// ---------------------------------------------------------------------------
// Model corpus
// ---------------------------------------------------------------------------

/// One zoo model: display name, repo-relative path, embedded bytes, and the
/// risk flags this profile must carry (from the executed-TFLM goldens
/// evidence — models/zoo/README.md, DEFERRED_MODELS.md, simd-zoo-hardening
/// learnings).
struct ModelSpec {
    name: &'static str,
    path: &'static str,
    risk: &'static str,
    bytes: &'static [u8],
}

const SINE_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../models/sine.tflite"
));
const HELLO_WORLD_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../models/zoo/sine_regression/hello_world_int8.tflite"
));
const KWS_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../models/zoo/keyword_spotting/kws_micro_speech_int8.tflite"
));
const ANOMALY_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../models/zoo/anomaly_detect/anomaly_detect_int8.tflite"
));
const PERSON_DETECT_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../models/zoo/person_detect_vww/person_detect_int8.tflite"
));
const MOBILENET_V2_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../models/zoo/mobilenetv2_cls/mobilenet_v2_1.0_224_int8.tflite"
));

const MODELS: &[ModelSpec] = &[
    ModelSpec {
        name: "sine",
        path: "models/sine.tflite",
        risk: "none — single FC",
        bytes: SINE_BYTES,
    },
    ModelSpec {
        name: "hello_world",
        path: "models/zoo/sine_regression/hello_world_int8.tflite",
        risk: "none — 3 FC, bit-exact golden",
        bytes: HELLO_WORLD_BYTES,
    },
    ModelSpec {
        name: "kws_micro_speech",
        path: "models/zoo/keyword_spotting/kws_micro_speech_int8.tflite",
        risk: "depthwise dm=8 (out_c=2·in_c) blocks dm==1 SIMD gate; 7 ms ESP-DL bar structurally unreachable (PROJECT_LOG.md:796-799)",
        bytes: KWS_BYTES,
    },
    ModelSpec {
        name: "anomaly_detect",
        path: "models/zoo/anomaly_detect/anomaly_detect_int8.tflite",
        risk: "±1 LSB variance — 210/640 elements differ from executed-TFLM golden (gemmlowp double-rounding vs hematite single-rounding; DEFERRED_MODELS §6)",
        bytes: ANOMALY_BYTES,
    },
    ModelSpec {
        name: "person_detect",
        path: "models/zoo/person_detect_vww/person_detect_int8.tflite",
        risk: "device-stack-gated — generated predict allocas ~232 KB vs ~65 KB stack (SKIP reason=stack; QEMU-only golden fnv 0x6962079d)",
        bytes: PERSON_DETECT_BYTES,
    },
    ModelSpec {
        name: "mobilenet_v2_1.0_224",
        path: "models/zoo/mobilenetv2_cls/mobilenet_v2_1.0_224_int8.tflite",
        risk: "18 PAD ops (PAD never fuses — no pattern covers it); 984/1000 golden deltas (890 PAD-fill class); PSRAM-gated on device (PSRAM: 0 bytes)",
        bytes: MOBILENET_V2_BYTES,
    },
];

// ---------------------------------------------------------------------------
// Per-model analysis
// ---------------------------------------------------------------------------

/// Pattern buckets a group may fall into.  Counting is NON-exclusive — a
/// single group can be, e.g., residual-add AND activation-epilogue
/// (fusion.rs:193-250 fields decide).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PatternCounts {
    activation_epilogue: usize,
    elementwise_chain: usize,
    residual_add: usize,
    input_fold: usize,
    requant_fold: usize,
}

/// Rough SIMD eligibility of a group's anchor kernel (APPROX — T4.1 mirror
/// does not exist yet; each answer cites the s3 gate it mirrors).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SimdEst {
    /// Anchor kernel is SIMD-eligible for these shapes per the cited gate.
    Simd,
    /// In-scope op whose gate FAILS for these shapes (scalar dispatch).
    Scalar,
    /// Anchor has no SIMD path in the C2/C3 composed-kernel scope.
    NoSimdPath,
}

impl SimdEst {
    fn label(self) -> &'static str {
        match self {
            SimdEst::Simd => "SIMD",
            SimdEst::Scalar => "scalar",
            SimdEst::NoSimdPath => "n/a",
        }
    }
}

struct GroupRow {
    anchor_idx: usize,
    anchor_builtin: i32,
    pattern_tags: String,
    elim_tensors: usize,
    elim_bytes: usize,
    simd: SimdEst,
    simd_note: &'static str,
}

struct ModelProfile {
    name: &'static str,
    ops: usize,
    groups: usize,
    fused_ops: usize,
    emitted_ops: usize,
    patterns: PatternCounts,
    elim_tensors: usize,
    elim_bytes: usize,
    simd_groups: usize,
    arena_unfused: Result<usize, String>,
    arena_fused: Result<usize, String>,
    risk: &'static str,
    group_rows: Vec<GroupRow>,
}

fn analyze(spec: &ModelSpec) -> ModelProfile {
    let model = flatbuffer::parse(spec.bytes).unwrap_or_else(|e| {
        panic!("{} failed to parse: {e}", spec.path)
    });
    let schedule = fuse(&model);

    let mut patterns = PatternCounts::default();
    let mut elim_tensors = 0usize;
    let mut elim_bytes = 0usize;
    let mut simd_groups = 0usize;
    let mut group_rows = Vec::with_capacity(schedule.groups.len());

    for g in &schedule.groups {
        if g.activation.is_some() {
            patterns.activation_epilogue += 1;
        }
        if !g.elementwise_chain.is_empty() {
            patterns.elementwise_chain += 1;
        }
        if g.residual_add.is_some() {
            patterns.residual_add += 1;
        }
        if g.input_fold.is_some() {
            patterns.input_fold += 1;
        }
        if g.folded_requantize.is_some() {
            patterns.requant_fold += 1;
        }
        elim_tensors += g.eliminated_tensors.len();
        for &t in &g.eliminated_tensors {
            if let Some(tt) = model.tensor_by_index(t as usize) {
                elim_bytes += tensor_byte_size(tt);
            }
        }

        let (simd, simd_note) = approx_simd(&model, g);
        if simd == SimdEst::Simd {
            simd_groups += 1;
        }
        group_rows.push(GroupRow {
            anchor_idx: g.anchor_op_index,
            anchor_builtin: g.anchor_builtin,
            pattern_tags: pattern_tags(g),
            elim_tensors: g.eliminated_tensors.len(),
            elim_bytes: g
                .eliminated_tensors
                .iter()
                .filter_map(|&t| model.tensor_by_index(t as usize))
                .map(tensor_byte_size)
                .sum(),
            simd,
            simd_note,
        });
    }

    let arena_unfused = arena::plan_arena(&model)
        .map(|p| p.peak_arena_bytes)
        .map_err(|e| e.to_string());
    let arena_fused = plan_fused_arena(&model, &schedule)
        .map(|p| p.peak_arena_bytes)
        .map_err(|e| e.to_string());

    ModelProfile {
        name: spec.name,
        ops: schedule.total_ops,
        groups: schedule.groups.len(),
        fused_ops: schedule.fused_op_count(),
        emitted_ops: schedule.emitted_op_count(),
        patterns,
        elim_tensors,
        elim_bytes,
        simd_groups,
        arena_unfused,
        arena_fused,
        risk: spec.risk,
        group_rows,
    }
}

/// Compact per-group pattern tags (`-` = anchor-only group).
fn pattern_tags(g: &FusedGroup) -> String {
    let mut tags = Vec::new();
    if g.activation.is_some() {
        tags.push("act-epilogue");
    }
    if !g.elementwise_chain.is_empty() {
        tags.push("chain");
    }
    if g.residual_add.is_some() {
        tags.push("residual-add");
    }
    if g.input_fold.is_some() {
        tags.push("input-fold");
    }
    if g.folded_requantize.is_some() {
        tags.push("requant-fold");
    }
    if tags.is_empty() {
        "-".to_string()
    } else {
        tags.join("+")
    }
}

// ---------------------------------------------------------------------------
// Fused arena estimate (reduced schedule over the real planner)
// ---------------------------------------------------------------------------

/// Run `hematite-memory::liveness_plan` over the fused op list.
///
/// One synthetic `OpInfo` per group; the contract mapping (filter
/// `u32::MAX`/dangling indices, `u16` casts, `in_place`) mirrors
/// `arena.rs::build_schedule` — the two must stay in sync.
fn plan_fused_arena(
    model: &ParsedModel<'_>,
    schedule: &FusedSchedule,
) -> Result<ArenaPlan, ArenaError> {
    let tensor_count = model.tensors().len();
    if tensor_count > MAX_TENSORS {
        return Err(ArenaError::TooManyTensors { count: tensor_count });
    }

    let mut infos = Vec::with_capacity(schedule.groups.len());
    for g in &schedule.groups {
        // Effective reads: the group's (fold-substituted) inputs plus every
        // tensor the fused kernel actually reads beyond the anchor's own
        // inputs — the residual (pattern c) and each chain operand (pattern b).
        let mut reads: Vec<u32> = g.inputs.clone();
        if let Some(ra) = &g.residual_add {
            reads.push(ra.residual_tensor);
        }
        for cw in &g.elementwise_chain {
            if cw.operand_tensor != u32::MAX {
                reads.push(cw.operand_tensor);
            }
        }
        let inputs: Vec<u32> = reads
            .iter()
            .copied()
            .filter(|&t| t != u32::MAX && (t as usize) < tensor_count)
            .collect();
        let outputs: Vec<u32> = if g.output_tensor != u32::MAX
            && (g.output_tensor as usize) < tensor_count
        {
            vec![g.output_tensor]
        } else {
            Vec::new()
        };
        if inputs.len() > MAX_IO_PER_OP {
            return Err(ArenaError::TooManyInputs { op: g.anchor_op_index, count: inputs.len() });
        }
        if outputs.len() > MAX_IO_PER_OP {
            return Err(ArenaError::TooManyOutputs { op: g.anchor_op_index, count: outputs.len() });
        }

        let mut input_ids = [0u16; MAX_IO_PER_OP];
        for (slot, &t) in inputs.iter().enumerate() {
            input_ids[slot] = t as u16;
        }
        let mut output_ids = [0u16; MAX_IO_PER_OP];
        for (slot, &t) in outputs.iter().enumerate() {
            output_ids[slot] = t as u16;
        }
        let in_place =
            !inputs.is_empty() && !outputs.is_empty() && outputs.first() == inputs.first();

        infos.push(OpInfo {
            op_kind: u16::try_from(g.anchor_builtin).unwrap_or(0),
            input_ids,
            input_count: inputs.len() as u8,
            output_ids,
            output_count: outputs.len() as u8,
            in_place,
        });
    }

    let sizes: Vec<usize> = model.tensors().iter().map(tensor_byte_size).collect();
    let model_input_ids: Vec<u16> = model
        .inputs()
        .iter()
        .copied()
        .filter(|&t| (t as usize) < tensor_count)
        .map(|t| t as u16)
        .collect();
    let model_output_ids: Vec<u16> = model
        .outputs()
        .iter()
        .copied()
        .filter(|&t| (t as usize) < tensor_count)
        .map(|t| t as u16)
        .collect();

    liveness_plan(&infos, &sizes, &model_input_ids, &model_output_ids, arena::MAX_INTERNAL, None)
        .map_err(ArenaError::Layout)
}

// ---------------------------------------------------------------------------
// Tensor sizing
// ---------------------------------------------------------------------------

/// Byte size of a tensor (shape product × element size), mirroring
/// `arena.rs::tensor_byte_size`.  Zero/negative dims → 0; saturated math.
fn tensor_byte_size(t: &ParsedTensor<'_>) -> usize {
    let elems: u64 = t
        .shape
        .iter()
        .fold(1u64, |acc, &d| if d <= 0 { 0 } else { acc.saturating_mul(d as u64) });
    let elem = match t.tensor_type {
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
    };
    (elems.saturating_mul(elem)) as usize
}

/// Flat element count of a shape (0 on dynamic/negative dims).
fn flat_prod(shape: &[i32]) -> usize {
    shape
        .iter()
        .fold(1usize, |acc, &d| if d <= 0 { 0 } else { acc.saturating_mul(d as usize) })
}

/// Channel count of a NHWC tensor = its last shape dim.
fn last_dim(shape: &[i32]) -> usize {
    shape.last().copied().filter(|&d| d > 0).map(|d| d as usize).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// APPROX SIMD eligibility (T4.1 will replace this with the parity-tested
// host mirror; every branch cites the s3 gate it approximates)
// ---------------------------------------------------------------------------

fn approx_simd(model: &ParsedModel<'_>, g: &FusedGroup) -> (SimdEst, &'static str) {
    match g.anchor_builtin {
        CONV_2D => {
            let anchor = &model.ops()[g.anchor_op_index];
            match anchor.inputs.get(1).and_then(|&t| model.tensor_by_index(t as usize)) {
                Some(w) => {
                    let in_c = last_dim(&w.shape);
                    let is_1x1 = w.shape.get(1) == Some(&1) && w.shape.get(2) == Some(&1);
                    if is_1x1 {
                        if in_c >= 16 && in_c.is_multiple_of(16) {
                            (
                                SimdEst::Simd,
                                "conv1x1: accx_eligible_1x1 in_c>=16 && %16 (accx.rs:64-66)",
                            )
                        } else {
                            (
                                SimdEst::Scalar,
                                "conv1x1: accx_eligible_1x1 fails (accx.rs:64-66)",
                            )
                        }
                    } else {
                        (
                            SimdEst::Simd,
                            "conv3x3: accx_eligible_3x3 any in_c>=1, pad-in-scratch (accx.rs:75-77)",
                        )
                    }
                }
                None => (SimdEst::NoSimdPath, "conv weight tensor missing"),
            }
        }
        DEPTHWISE_CONV_2D => {
            let anchor = &model.ops()[g.anchor_op_index];
            let w = anchor.inputs.get(1).and_then(|&t| model.tensor_by_index(t as usize));
            let out = model.tensor_by_index(g.output_tensor as usize);
            match (w, out) {
                (Some(w), Some(o)) => {
                    let in_c = last_dim(&w.shape);
                    let out_c = last_dim(&o.shape);
                    if in_c >= 1 && out_c >= 1 && in_c == out_c {
                        (
                            SimdEst::Simd,
                            "depthwise: accx_eligible_depthwise in_c==out_c (accx.rs:87-89)",
                        )
                    } else {
                        (
                            SimdEst::Scalar,
                            "depthwise: accx_eligible_depthwise fails — dm!=1 (accx.rs:87-89)",
                        )
                    }
                }
                _ => (SimdEst::NoSimdPath, "depthwise tensor missing"),
            }
        }
        FULLY_CONNECTED => {
            let in_c = g
                .inputs
                .first()
                .and_then(|&t| model.tensor_by_index(t as usize))
                .map(|t| flat_prod(&t.shape))
                .unwrap_or(0);
            if in_c >= 16 && in_c.is_multiple_of(16) {
                (
                    SimdEst::Simd,
                    "fc: accx_eligible_1x1 input_dim>=16 && %16 (accx.rs:64-66)",
                )
            } else {
                (
                    SimdEst::Scalar,
                    "fc: accx_eligible_1x1 fails (accx.rs:64-66)",
                )
            }
        }
        AVERAGE_POOL_2D | MAX_POOL_2D => approx_pool(model, g),
        SOFTMAX => {
            let row_size = g
                .inputs
                .first()
                .and_then(|&t| model.tensor_by_index(t as usize))
                .map(|t| last_dim(&t.shape))
                .unwrap_or(0);
            // Gate (softmax.rs:383-387): row_size >= 16 + pointer alignment
            // (not host-checkable — T4.1 mirror handles it).
            if row_size >= 16 {
                (SimdEst::Simd, "softmax: row_size>=16 (softmax.rs:383-387); ptr-align at runtime")
            } else {
                (SimdEst::Scalar, "softmax: row_size<16 (softmax.rs:383-387)")
            }
        }
        ADD | SUB | MUL => (
            // Elementwise-chain anchors: per-step quant-identity gate
            // (elementwise.rs:606-635 simd_eligible_add_sub/mul).  The zoo
            // models' chain steps are not checked here — T4.1 mirror's job.
            SimdEst::Simd,
            "elementwise chain: per-step quant-identity gate, CONDITIONAL (elementwise.rs:606-635)",
        ),
        _ => (SimdEst::NoSimdPath, "no composed SIMD kernel in C2/C3 scope (data movement / pad / reshape / transpose / mean)"),
    }
}

/// Pool gate mirror — `simd_eligible_pool` (pool.rs:869-902): filter 2×2,
/// stride 2, pad 0, channels % 16 == 0, full-range clamp.
fn approx_pool(model: &ParsedModel<'_>, g: &FusedGroup) -> (SimdEst, &'static str) {
    let anchor = &model.ops()[g.anchor_op_index];
    let opts = match anchor.options.as_ref() {
        Some(ParsedOptions::Pool2D {
            padding,
            stride_w,
            stride_h,
            filter_w,
            filter_h,
            fused_activation: _,
        }) => (*padding, *stride_w, *stride_h, *filter_w, *filter_h),
        _ => return (SimdEst::NoSimdPath, "pool options missing"),
    };
    let (_padding, stride_w, stride_h, filter_w, filter_h) = opts;
    let in_t = g.inputs.first().and_then(|&t| model.tensor_by_index(t as usize));
    let out_t = model.tensor_by_index(g.output_tensor as usize);
    let (in_shape, out_shape) = match (in_t, out_t) {
        (Some(i), Some(o)) => (&i.shape, &o.shape),
        _ => return (SimdEst::NoSimdPath, "pool tensor missing"),
    };
    let (in_h, in_w, ch) = (
        in_shape.get(1).copied().unwrap_or(1),
        in_shape.get(2).copied().unwrap_or(1),
        last_dim(in_shape),
    );
    let (out_h, out_w) = (
        out_shape.get(1).copied().unwrap_or(1),
        out_shape.get(2).copied().unwrap_or(1),
    );
    let pad_h = ((out_h - 1) * stride_h + filter_h - in_h) / 2;
    let pad_w = ((out_w - 1) * stride_w + filter_w - in_w) / 2;
    if filter_h == 2 && filter_w == 2 && stride_h == 2 && stride_w == 2 && pad_h == 0 && pad_w == 0
        && ch >= 16 && ch.is_multiple_of(16)
    {
        // NOTE: the s3 gate also requires full-range clamp
        // (quantized_activation_min/max == ±128) — not checked here (APPROX).
        (
            SimdEst::Simd,
            "pool: 2x2/stride-2/pad-0/ch%16 (pool.rs:869-902); clamp-range not checked (APPROX)",
        )
    } else {
        (SimdEst::Scalar, "pool: simd_eligible_pool fails (pool.rs:869-902)")
    }
}

// ---------------------------------------------------------------------------
// Markdown + stdout rendering
// ---------------------------------------------------------------------------

fn arena_cell(r: &Result<usize, String>) -> String {
    match r {
        Ok(bytes) => format!("{bytes}"),
        Err(e) => format!("ERR({e})"),
    }
}

fn render(profiles: &[ModelProfile]) -> String {
    let mut s = String::new();
    s.push_str("<!-- Generated by hematite-codegen optimize::profile::profile_zoo_models (T0.2). Do not edit by hand. -->\n");
    s.push_str("# Fused-pattern profile — zoo models (T0.2)\n\n");
    s.push_str(
        "Produced by `cargo test -p hematite-codegen -- --nocapture profile_zoo_models`: every \
         model is parsed with `flatbuffer::parse` (the proc-macro's own entry point, lib.rs:41-68) \
         and fused with `fusion::fuse()`. Column semantics are pinned here so T4.2 \
         (`selector-output.md`) and T6.x speed targets can cite them.\n\n",
    );

    // ── Main table ────────────────────────────────────────────────────────
    s.push_str("## Per-model summary\n\n");
    s.push_str("| Model | ops | groups (emitted) | fused ops | act-epilogue | elementwise-chain | residual-add | input-fold | requant-fold | eliminated_tensors | eliminated_bytes | SIMD-elig (APPROX) | arena unfused (B) | arena fused est (B) | saved calls | saved bytes | risk flags |\n");
    s.push_str("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    for p in profiles {
        let row = format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}/{} | {} | {} | {} | {} | {} |\n",
            p.name,
            p.ops,
            p.groups,
            p.fused_ops,
            p.patterns.activation_epilogue,
            p.patterns.elementwise_chain,
            p.patterns.residual_add,
            p.patterns.input_fold,
            p.patterns.requant_fold,
            p.elim_tensors,
            p.elim_bytes,
            p.simd_groups,
            p.groups,
            arena_cell(&p.arena_unfused),
            arena_cell(&p.arena_fused),
            p.fused_ops, // saved kernel calls == fused_op_count (absorbed ops)
            p.elim_bytes,
            p.risk,
        );
        s.push_str(&row);
    }
    s.push('\n');
    s.push_str(
        "* `ops` = original op count (`FusedSchedule.total_ops`); `groups (emitted)` = \
         `FusedSchedule.groups.len()` (= `emitted_op_count`); `fused ops` = \
         `fused_op_count` (absorbed ops = eliminated kernel calls).\n\
         * Pattern counts are NON-exclusive buckets — one group can satisfy several \
         (e.g. residual-add AND activation-epilogue).\n\
         * `act-epilogue` counts groups whose kernel carries an activation epilogue — either \
         via the anchor's own `fused_activation` field (already present in the UNFUSED kernel \
         call, zero added savings) or via an absorbed standalone activation (counted in \
         `fused ops`). `saved calls` / `fused ops` are the ground truth for eliminated kernel \
         calls.\n\
         * `eliminated_bytes` = sum of the byte sizes of all `eliminated_tensors` (SRAM \
         resident bytes no longer produced; write+read SRAM traffic saved = 2× this).\n\
         * `SIMD-elig (APPROX)` = groups whose anchor kernel the known s3 gates would \
         engage — APPROX pending T4.1's parity-tested host mirror; per-group cells in the \
         detail sections cite each gate.\n\
         * `arena unfused` = `arena::plan_arena` peak; `arena fused est` = the same \
         planner over the reduced fused schedule (see module docs). `ERR(...)` = planner \
         rejection (e.g. OutOfBudget > 512 KiB).\n",
    );

    // ── Per-model detail ──────────────────────────────────────────────────
    s.push_str("\n## Per-group detail\n\n");
    for p in profiles {
        s.push_str(&format!("### {}\n\n", p.name));
        s.push_str("| group | anchor op | anchor builtin | pattern(s) | elim tensors | elim bytes | SIMD (APPROX) |\n");
        s.push_str("|---|---|---|---|---|---|---|\n");
        for (i, r) in p.group_rows.iter().enumerate() {
            s.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} ({}) |\n",
                i,
                r.anchor_idx,
                r.anchor_builtin,
                r.pattern_tags,
                r.elim_tensors,
                r.elim_bytes,
                r.simd.label(),
                r.simd_note,
            ));
        }
        s.push('\n');
    }

    // ── Method + risk notes ───────────────────────────────────────────────
    s.push_str("## Method\n\n");
    s.push_str(
        "- **Parser/IR**: `flatbuffer::parse(&bytes)` — identical entry point to the \
         proc-macro's `parse_and_emit` (lib.rs:41-68); `include_bytes!` with \
         CARGO_MANIFEST_DIR-relative paths mirrors the in-crate precedent \
         (arena.rs:509-521 `plan_arena_on_real_sine_model`).\n\
         - **Fusion**: `fusion::fuse(&model)` — deterministic, no global state.\n\
         - **Fused arena**: reduced-schedule run of `hematite-memory::liveness_plan`; \
         OpInfo mapping mirrors `arena.rs::build_schedule`. The number is the planner's \
         answer for the fused op list T1.2 will emit — not a subtraction estimate.\n\
         - **SIMD column**: APPROX — the T4.1 (wave 4) parity-tested host eligibility \
         mirror does not exist yet; cells mirror the cited s3 gates by shape only \
         (runtime pointer alignment is not host-visible).\n\
         - **Not fabricated**: every cell above comes from the real `fuse()` run; a model \
         with zero fused groups would still be a valid row.\n",
    );

    s
}

fn evidence_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("local-notes")
        .join("evidence")
        .join("composed-kernels")
        .join("fused-profile.md")
}

// ---------------------------------------------------------------------------
// The test
// ---------------------------------------------------------------------------

#[test]
fn profile_zoo_models() {
    let profiles: Vec<ModelProfile> = MODELS.iter().map(analyze).collect();

    // Hard invariants — all 6 models parsed + fused without panic, and the
    // schedule totals are self-consistent (fused + emitted == ops).
    assert_eq!(profiles.len(), 6, "all 6 zoo models profiled");
    for p in &profiles {
        assert_eq!(p.ops, p.fused_ops + p.emitted_ops, "{}: fused + emitted == original ops", p.name);
        assert!(
            p.elim_tensors <= p.fused_ops,
            "{}: eliminated tensors ({}) cannot exceed absorbed ops ({})",
            p.name,
            p.elim_tensors,
            p.fused_ops
        );
    }

    let markdown = render(&profiles);

    // stdout — visible with `cargo test -p hematite-codegen -- --nocapture profile_zoo_models`.
    println!("===== fused-pattern profile over zoo models (T0.2) =====");
    println!("{markdown}");

    // Evidence file (host-side only, test-only module).
    let path = evidence_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create local-notes/evidence/composed-kernels");
    }
    std::fs::write(&path, &markdown).unwrap_or_else(|e| {
        panic!("failed to write {}: {e}", path.display())
    });
    println!("wrote {}", path.display());
}
