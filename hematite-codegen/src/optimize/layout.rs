// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! T4.2c — Layout / alignment pass.
//!
//! Decides a per-tensor layout attribute (`NHWC | NHWC_pad16`) via a tiny
//! cost model, inserts repad descriptors at layout boundaries, and folds
//! weight-buffer TRANSPOSE/RESHAPE ops into compile-time re-indexing so the
//! runtime op disappears.  Output is consumed by the T4.1 emitter (wiring
//! task).
//!
//! ## Why channel padding matters
//!
//! The TIE728 128-bit SIMD loads force low-4-bits-aligned addresses; an
//! unaligned load costs `EE.LD.128.USAR → VLD.128 → SRC.Q` — three
//! instructions plus register pressure.  The conv inner loop processes input
//! channels in 16-lane groups (`c_div_x_1 = C/16 − 1`), so an activation
//! tensor whose channel count is a multiple of 16 yields clean `EE.VLD.128`
//! streams with no partial-lane tail.
//!
//! ## Cost model — the two thresholds
//!
//! A tensor's channel count `C` is padded up to a multiple of
//! [`SIMD_LANES`] **only when**:
//!
//! 1. `C >= SIMD_LANES` — the tensor is already lane-scale (padding a 3- or
//!    8-channel first layer would be all overhead), AND
//! 2. `3 * (padded - C) <= C` — the padding delta is at most one third of
//!    the original channel count (the [`MAX_PAD_OVERHEAD`] bound).
//!
//! The bound is chosen so that **C=24 → 32 (+33%) is accepted** — an
//! 8-lane one-time cost buys a clean lane stream for the rest of the
//! network — while **C=3 → 16 (+433%) is never** applied: a first-layer
//! 224²×3 tensor would balloon ≈150 KiB → ≈800 KiB to shave a single scalar
//! tail.  In practice only C ∈ [24, 32) (plus already-aligned tensors, where
//! padding is free) is padded.
//!
//! ## Repad boundaries
//!
//! A repad node is a **descriptor, not a runtime op**: the emitter allocates
//! the padded intermediate and every consumer reads it with the padded row
//! stride — often this is zero-copy (the next kernel reads the same buffer
//! with a different stride).  The only boundary the pass emits is a graph
//! input whose channels pass the cost model but whose bytes arrive from the
//! caller unpadded.
//!
//! ## Transpose elimination
//!
//! A TRANSPOSE (or pure-relabel RESHAPE) whose output feeds the **weight**
//! input (position 1) of a conv/depthwise/fc is folded: the constant weight
//! buffer is re-indexed at compile time (XLA-style "user transposes are
//! ignored") and the runtime op is dropped.  Foldable only when the
//! permutation is a pure bijective reindexing (NHWC↔HWCN and friends) of the
//! consumer's expected weight rank; lossy or layout-changing permutations are
//! never folded.
//
// Dead-code warnings are expected at T4.2c — the T4.1 emitter wiring task
// consumes these descriptors.
#![allow(dead_code)]

use crate::flatbuffer::{ParsedModel, ParsedOp, ParsedOptions, ParsedTensor};

/// SIMD lane width in elements for TIE728 128-bit loads (16 × int8).
pub(crate) const SIMD_LANES: u32 = 16;

/// Maximum channel-padding overhead as a fraction of the unpadded channel
/// count.  A tensor is padded only when `delta / channels <= 1/3`, written
/// in exact integer form as `3 * delta <= channels`.
///
/// * C=24 → 32: delta/channels = 8/24 = **+33% — accepted** (one-time
///   8-lane cost buys a clean SIMD stream for the rest of the network).
/// * C=3 → 16: delta/channels = 13/3 = **+433% — never**: a first-layer
///   224²×3 tensor would balloon ≈150 KiB → ≈800 KiB to shave one scalar
///   tail on a single op.
pub(crate) const MAX_PAD_OVERHEAD: u32 = 3;

// BuiltinOperator codes (TFLite v23.1-era schema, resolved by T4.0).
const CONV_2D: i32 = 3;
const DEPTHWISE_CONV_2D: i32 = 4;
const FULLY_CONNECTED: i32 = 9;
const TRANSPOSE: i32 = 15;
const RESHAPE: i32 = 22;

/// Per-tensor channel layout attribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ChannelLayout {
    /// NHWC with the channel count as-is — the base layout.
    Nhwc,
    /// NHWC with channels padded to a multiple of [`SIMD_LANES`].  Assigned
    /// when the cost model accepts the padding (delta ≤ 1/3 of C) or when C
    /// is already lane-aligned (padding is free).  `effective_channels % 16
    /// == 0` always holds.
    NhwcPad16,
}

/// Layout decision for one tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TensorLayout {
    pub(crate) tensor_index: u32,
    pub(crate) layout: ChannelLayout,
    /// Original channel count from the model (last NHWC dim).
    pub(crate) channels: u32,
    /// Allocated channel count: `== channels` for `Nhwc`, the next multiple
    /// of [`SIMD_LANES`] for `NhwcPad16`.  The emitter sizes allocations
    /// from this — a plain const, so padded sizes are compile-time
    /// expressible.
    pub(crate) effective_channels: u32,
}

/// A layout-boundary repad descriptor — NOT a runtime op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RepadNode {
    /// Tensor whose unpadded bytes must be re-materialized (or re-strided)
    /// into a padded intermediate.
    pub(crate) tensor_index: u32,
    /// Padded channel count of the destination allocation.
    pub(crate) target_channels: u32,
    /// Why this boundary exists (emitter-facing note).
    pub(crate) note: &'static str,
}

/// A folded TRANSPOSE/RESHAPE — the runtime op disappears.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TransposeFold {
    /// Index of the TRANSPOSE/RESHAPE op dropped from the runtime schedule.
    pub(crate) op_index: u32,
    /// Constant weight tensor whose buffer is re-indexed at compile time.
    pub(crate) weight_index: u32,
    /// Axis permutation (output axis → input axis, the TRANSPOSE `perm`).
    /// The emitter re-indexes the flat weight buffer with
    /// `dst[i0, i1, ...] = src[perm[0], perm[1], ...]`.  Empty for RESHAPE
    /// folds (a pure relabel needs no reordering).
    pub(crate) permutation: Vec<usize>,
    /// Shape the re-indexed weight is consumed with (the folded op's output
    /// shape).  The emitter lays out the re-indexed const array from this.
    pub(crate) weight_shape: Vec<i32>,
    pub(crate) note: &'static str,
}

/// Output of [`decide_layouts`]: everything the emitter needs to allocate
/// padded intermediates, re-index folded weights, and skip folded ops.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LayoutPlan {
    /// One entry per model tensor, index-aligned with `model.tensors()`.
    pub(crate) layouts: Vec<TensorLayout>,
    /// Repad boundaries (producer layout ≠ consumer layout).
    pub(crate) repads: Vec<RepadNode>,
    /// Weight transposes/reshapes folded into compile-time re-indexing.
    pub(crate) folds: Vec<TransposeFold>,
}

/// The tiny cost model — see the module docs for the two thresholds.
pub(crate) fn layout_pad_decision(channels: u32) -> ChannelLayout {
    if channels == 0 {
        return ChannelLayout::Nhwc;
    }
    let delta = padded_channels(channels) - channels;
    if delta == 0 {
        // Already lane-aligned: Pad16 is free.
        ChannelLayout::NhwcPad16
    } else if channels >= SIMD_LANES && MAX_PAD_OVERHEAD * delta <= channels {
        // delta <= channels / 3, integer form of the MAX_PAD_OVERHEAD bound.
        ChannelLayout::NhwcPad16
    } else {
        ChannelLayout::Nhwc
    }
}

/// Pad a channel count up to a multiple of [`SIMD_LANES`] (identity when
/// already aligned).
pub(crate) fn padded_channels(channels: u32) -> u32 {
    channels.div_ceil(SIMD_LANES) * SIMD_LANES
}

/// Compute per-tensor layouts, repad boundaries, and weight-transpose folds
/// for the whole model.
pub(crate) fn decide_layouts(model: &ParsedModel<'_>) -> LayoutPlan {
    let graph_inputs = model.inputs();
    let graph_outputs = model.outputs();
    let mut layouts = Vec::with_capacity(model.tensors().len());
    let mut repads = Vec::new();

    for (index, tensor) in model.tensors().iter().enumerate() {
        let index = u32::try_from(index).expect("model tensor count fits u32");
        let channels = channels_of(tensor);
        let is_graph_output = graph_outputs.contains(&index);
        let is_constant = model.buffer_data(tensor).is_some();
        let is_consumed_as_weight = model.ops().iter().any(|op| {
            weight_rank_for(op.builtin_code).is_some() && op.inputs.get(1) == Some(&index)
        });
        let is_nhwc_4d = tensor.shape.len() == 4;

        // Only rank-4 *activation* tensors get the pad16 cost model.
        // Constants are fixed-size buffers (esp-nn contract: "Filter can be
        // unaligned"); graph outputs keep the caller-visible shape; tensors
        // consumed as a conv-family weight input are weights, not
        // activations.  Graph *inputs* DO get the cost model — the layout
        // records what the consumers want, and the repad node below records
        // that the external caller supplies unpadded bytes.
        let layout = if !is_constant && !is_graph_output && !is_consumed_as_weight && is_nhwc_4d {
            layout_pad_decision(channels)
        } else {
            ChannelLayout::Nhwc
        };
        let effective_channels = match layout {
            ChannelLayout::Nhwc => channels,
            ChannelLayout::NhwcPad16 => padded_channels(channels),
        };
        layouts.push(TensorLayout {
            tensor_index: index,
            layout,
            channels,
            effective_channels,
        });

        // Repad boundary: a graph input is produced unpadded by the external
        // caller; when the cost model wants pad16 the emitter allocates a
        // padded intermediate and copies the input bytes into it before the
        // first kernel.  Skipped when padding is free (already aligned).
        if layout == ChannelLayout::NhwcPad16
            && effective_channels > channels
            && graph_inputs.contains(&index)
        {
            repads.push(RepadNode {
                tensor_index: index,
                target_channels: effective_channels,
                note: "graph input produced unpadded; consumer reads pad16",
            });
        }
    }

    LayoutPlan { layouts, repads, folds: find_weight_folds(model) }
}

/// Channel count of a tensor in NHWC layout — the last shape element.
/// Rank-0/1 tensors (scalars, bias vectors) have no channel dim to pad.
fn channels_of(tensor: &ParsedTensor<'_>) -> u32 {
    tensor
        .shape
        .last()
        .and_then(|&c| u32::try_from(c.max(0)).ok())
        .unwrap_or(0)
}

/// Expected constant weight rank for each conv-family op: conv/depthwise
/// weights are 4-D (`[O,H,W,I]` / `[1,FH,FW,IC]`), FC weights are 2-D.
fn weight_rank_for(op_code: i32) -> Option<usize> {
    match op_code {
        CONV_2D | DEPTHWISE_CONV_2D => Some(4),
        FULLY_CONNECTED => Some(2),
        _ => None,
    }
}

/// Find every TRANSPOSE/RESHAPE whose output feeds a conv-family weight
/// input directly (immediate predecessor in execution order).
fn find_weight_folds(model: &ParsedModel<'_>) -> Vec<TransposeFold> {
    let mut folds = Vec::new();
    for (index, op) in model.ops().iter().enumerate() {
        if op.builtin_code != TRANSPOSE && op.builtin_code != RESHAPE {
            continue;
        }
        let index = u32::try_from(index).expect("model op count fits u32");
        if let Some(fold) = try_fold_weight_layout_op(model, index, op) {
            folds.push(fold);
        }
    }
    folds
}

/// Attempt to fold one layout op into the weight embed.  Returns `None`
/// (op stays in the runtime schedule) when any foldability gate fails.
fn try_fold_weight_layout_op(
    model: &ParsedModel<'_>,
    op_index: u32,
    op: &ParsedOp<'_>,
) -> Option<TransposeFold> {
    // The layout op must directly precede a conv-family op and feed its
    // *weight* input (position 1).
    let consumer = model.ops().get(op_index as usize + 1)?;
    let weight_rank = weight_rank_for(consumer.builtin_code)?;
    let out = *op.outputs.first()?;
    if consumer.inputs.get(1) != Some(&out) {
        return None;
    }

    // Only constants can be re-indexed at compile time; a runtime-produced
    // "weight" cannot.
    let src = *op.inputs.first()?;
    let src_tensor = model.tensor_by_index(src as usize)?;
    model.buffer_data(src_tensor)?;
    let out_tensor = model.tensor_by_index(out as usize)?;
    let weight_shape = out_tensor.shape.clone();
    if weight_shape.len() != weight_rank {
        return None;
    }

    match op.builtin_code {
        TRANSPOSE => {
            let permutation = read_transpose_permutation(model, op, weight_shape.len())?;
            Some(TransposeFold {
                op_index,
                weight_index: src,
                permutation,
                weight_shape,
                note: "weight TRANSPOSE folded into compile-time buffer re-index",
            })
        }
        RESHAPE => {
            let new_shape = read_reshape_shape(model, op)?;
            if new_shape.len() != weight_rank || new_shape.iter().any(|&d| d <= 0) {
                return None;
            }
            // Pure relabel: identical non-unit dims (in order) mean flat
            // row-major order is unchanged — no element moves.  A reshape
            // that merges or splits channels would be
            // layout-changing-with-loss and is never folded.
            if strip_unit_dims(&src_tensor.shape) != strip_unit_dims(&new_shape) {
                return None;
            }
            Some(TransposeFold {
                op_index,
                weight_index: src,
                permutation: Vec::new(),
                weight_shape: new_shape,
                note: "weight RESHAPE folded as a pure shape relabel",
            })
        }
        _ => None,
    }
}

/// Read the TRANSPOSE permutation from the constant `inputs[1]` tensor and
/// validate it is a bijection over `0..rank` (a pure reindex — a duplicate
/// or out-of-range axis would be a lossy gather, not a transpose).
fn read_transpose_permutation(
    model: &ParsedModel<'_>,
    op: &ParsedOp<'_>,
    rank: usize,
) -> Option<Vec<usize>> {
    let perm_tensor = model.tensor_by_index(*op.inputs.get(1)? as usize)?;
    let data = model.buffer_data(perm_tensor)?;
    if data.len() != rank * 4 {
        return None;
    }
    let mut perm = Vec::with_capacity(rank);
    for chunk in data.chunks_exact(4) {
        let axis = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let axis = usize::try_from(axis).ok()?;
        if axis >= rank {
            return None;
        }
        perm.push(axis);
    }
    let mut seen = vec![false; rank];
    for &axis in &perm {
        if seen[axis] {
            return None;
        }
        seen[axis] = true;
    }
    Some(perm)
}

/// Read the RESHAPE target shape from the options table, falling back to the
/// constant shape tensor at `inputs[1]`.
fn read_reshape_shape(model: &ParsedModel<'_>, op: &ParsedOp<'_>) -> Option<Vec<i32>> {
    if let Some(ParsedOptions::Reshape { new_shape }) = &op.options {
        return Some(new_shape.clone());
    }
    let shape_tensor = model.tensor_by_index(*op.inputs.get(1)? as usize)?;
    let data = model.buffer_data(shape_tensor)?;
    if data.len() % 4 != 0 {
        return None;
    }
    Some(
        data.chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

/// Drop size-1 dims: two shapes are a pure relabel iff their unit-stripped
/// sequences are identical.
fn strip_unit_dims(shape: &[i32]) -> Vec<i32> {
    shape.iter().copied().filter(|&d| d != 1).collect()
}

// ---------------------------------------------------------------------------
// Unit tests — in-crate only (proc-macro restriction)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flatbuffer;

    // ------------------------------------------------------------------
    // Minimal flatbuffer serializer for hand-built TFLite test models.
    //
    // A table = i32 soffset to its vtable, then field values inline; the
    // vtable holds [vtable_len, table_size, per-field offsets].  Vectors,
    // strings, and tables are referenced by uoffset (relative u32) that must
    // point FORWARD (flatbuffers rule — the walker does plain `pos + off`).
    // The builder therefore emits each referencing table first and patches
    // its uoffset slots once the targets exist.  Slot numbering follows
    // schema.fbs (see flatbuffer.rs for the walker).
    // ------------------------------------------------------------------
    mod fb {
        /// A table field value: raw inline bytes, or a forward-uoffset to a
        /// target patched in after the target is emitted.
        pub(super) enum Fv {
            Raw(Vec<u8>),
            /// Placeholder for a forward reference (patched via `patch_ref`).
            Ref,
        }

        /// A patching slot for one `Fv::Ref` field inside a table.
        #[derive(Clone, Copy)]
        pub(super) struct RefSlot {
            pub(super) field_pos: usize,
        }

        pub(super) struct Fb {
            pub(super) bytes: Vec<u8>,
        }

        impl Fb {
            pub(super) fn new() -> Self {
                let mut bytes = Vec::new();
                bytes.extend_from_slice(&[0u8; 8]); // root uoffset + identifier
                bytes[4..8].copy_from_slice(b"TFL3");
                Self { bytes }
            }

            pub(super) fn pos(&self) -> usize {
                self.bytes.len()
            }

            pub(super) fn align4(&mut self) {
                while self.bytes.len() % 4 != 0 {
                    self.bytes.push(0);
                }
            }

            /// Emit a table and return its position plus one [`RefSlot`] per
            /// `Fv::Ref` field (in field order).
            pub(super) fn table(&mut self, fields: &[(u32, Fv)]) -> (usize, Vec<RefSlot>) {
                self.align4();
                let table_pos = self.pos();
                // Pass 1: each field's offset from the table start.
                let mut field_offs = Vec::with_capacity(fields.len());
                let mut table_size = 4u32; // soffset slot
                for (_, v) in fields {
                    field_offs.push(table_size);
                    table_size += match v {
                        Fv::Raw(b) => b.len() as u32,
                        Fv::Ref => 4,
                    };
                }
                // Pass 2: soffset placeholder + field values (Ref slots start
                // zeroed and are patched later).
                self.bytes.extend_from_slice(&[0u8; 4]);
                let mut ref_slots = Vec::new();
                for ((_, v), off) in fields.iter().zip(&field_offs) {
                    match v {
                        Fv::Raw(b) => self.bytes.extend_from_slice(b),
                        Fv::Ref => {
                            let field_pos = table_pos + *off as usize;
                            self.bytes.extend_from_slice(&0u32.to_le_bytes());
                            ref_slots.push(RefSlot { field_pos });
                        }
                    }
                }
                // Vtable.
                self.align4();
                let vt_pos = self.pos();
                let nfields = fields
                    .iter()
                    .map(|(i, _)| *i as usize)
                    .max()
                    .map_or(0, |m| m + 1);
                let vt_len = u16::try_from(nfields + 2).expect("vtable length fits u16");
                let table_size = u16::try_from(table_size).expect("table size fits u16");
                self.bytes.extend_from_slice(&(vt_len * 2).to_le_bytes());
                self.bytes.extend_from_slice(&table_size.to_le_bytes());
                let mut vt = vec![0u16; nfields];
                for ((i, _), off) in fields.iter().zip(&field_offs) {
                    vt[*i as usize] = u16::try_from(*off).expect("field offset fits u16");
                }
                for o in vt {
                    self.bytes.extend_from_slice(&o.to_le_bytes());
                }
                // Patch the soffset (table_pos - vtable_pos).
                let soff = (table_pos as u32).wrapping_sub(vt_pos as u32);
                self.bytes[table_pos..table_pos + 4].copy_from_slice(&soff.to_le_bytes());
                (table_pos, ref_slots)
            }

            /// Patch a forward reference now that its target exists.
            pub(super) fn patch_ref(&mut self, slot: &RefSlot, target: usize) {
                let rel = u32::try_from(target - slot.field_pos)
                    .expect("uoffset target must follow its referencing field");
                self.bytes[slot.field_pos..slot.field_pos + 4].copy_from_slice(&rel.to_le_bytes());
            }

            /// Patch one element of a u32 vector to be a forward uoffset.
            pub(super) fn patch_vec_elem(&mut self, vec_pos: usize, elem_idx: usize, target: usize) {
                let elem_pos = vec_pos + 4 + elem_idx * 4;
                let rel =
                    u32::try_from(target - elem_pos).expect("uoffset target must follow the vector");
                self.bytes[elem_pos..elem_pos + 4].copy_from_slice(&rel.to_le_bytes());
            }

            pub(super) fn vec_u32(&mut self, elems: &[u32]) -> usize {
                self.align4();
                let p = self.pos();
                self.bytes.extend_from_slice(&(elems.len() as u32).to_le_bytes());
                for e in elems {
                    self.bytes.extend_from_slice(&e.to_le_bytes());
                }
                p
            }

            pub(super) fn vec_i32(&mut self, elems: &[i32]) -> usize {
                self.align4();
                let p = self.pos();
                self.bytes.extend_from_slice(&(elems.len() as u32).to_le_bytes());
                for e in elems {
                    self.bytes.extend_from_slice(&e.to_le_bytes());
                }
                p
            }

            pub(super) fn vec_bytes(&mut self, data: &[u8]) -> usize {
                self.align4();
                let p = self.pos();
                self.bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
                self.bytes.extend_from_slice(data);
                p
            }

            pub(super) fn string(&mut self, s: &str) -> usize {
                self.align4();
                let p = self.pos();
                self.bytes.extend_from_slice(&(s.len() as u32).to_le_bytes());
                self.bytes.extend_from_slice(s.as_bytes());
                self.bytes.push(0); // NUL terminator (schema-correct, unused by walker)
                p
            }

            /// Patch the root uoffset in bytes 0..4 and return the model.
            pub(super) fn finish(mut self, root: usize) -> Vec<u8> {
                self.bytes[0..4].copy_from_slice(&(root as u32).to_le_bytes());
                self.bytes
            }
        }
    }

    use fb::{Fb, Fv};

    /// BuiltinOperator code used only by the test graphs (TFLite v23.1-era).
    const MUL: i32 = 18;

    /// A hand-built test tensor.
    struct BuildTensor {
        shape: Vec<i32>,
        name: &'static str,
        /// Constant buffer bytes (empty = activation/intermediate).
        data: Vec<u8>,
    }

    impl BuildTensor {
        fn activation(shape: &[i32], name: &'static str) -> Self {
            Self { shape: shape.to_vec(), name, data: Vec::new() }
        }

        fn constant(shape: &[i32], name: &'static str, data: Vec<u8>) -> Self {
            Self { shape: shape.to_vec(), name, data }
        }
    }

    /// A hand-built test op (opcode_index = its position in the op list).
    struct BuildOp {
        builtin_code: i32,
        inputs: Vec<u32>,
        outputs: Vec<u32>,
    }

    /// Assemble a complete TFLite model flatbuffer: operator_codes, one
    /// subgraph, buffers.  All tensors are INT8 (type byte 9).
    ///
    /// Emission order: every uoffset points forward, so referencing tables
    /// are emitted before the data they reference and the uoffset slots are
    /// patched as the targets appear.
    fn build_model(
        tensors: Vec<BuildTensor>,
        ops: Vec<BuildOp>,
        inputs: Vec<u32>,
        outputs: Vec<u32>,
    ) -> Vec<u8> {
        let mut fb = Fb::new();

        // Buffers: index 0 is the empty sentinel; one buffer per tensor with
        // data.
        let mut buffer_indices = vec![0u32; tensors.len()];
        let mut buffer_datas: Vec<Vec<u8>> = vec![Vec::new()];
        for (i, t) in tensors.iter().enumerate() {
            if !t.data.is_empty() {
                buffer_indices[i] =
                    u32::try_from(buffer_datas.len()).expect("buffer count fits u32");
                buffer_datas.push(t.data.clone());
            }
        }
        let buffer_count = buffer_datas.len();

        // 1. Model table (lowest address) — uoffset slots for the three
        // vectors below.
        let (model, slots) = fb.table(&[(1, Fv::Ref), (2, Fv::Ref), (4, Fv::Ref)]);
        let [s_opcodes, s_subgraphs, s_buffers] = [slots[0], slots[1], slots[2]];

        // 2. Subgraphs vector then the Subgraph table (the vector's element
        // must point forward to the table, which must in turn point forward
        // to its vectors).
        let subgraphs_vec = fb.vec_u32(&[0u32; 1]);
        let (subgraph, slots) =
            fb.table(&[(0, Fv::Ref), (1, Fv::Ref), (2, Fv::Ref), (3, Fv::Ref)]);
        let [s_tensors, s_inputs, s_outputs, s_operators] =
            [slots[0], slots[1], slots[2], slots[3]];
        fb.patch_vec_elem(subgraphs_vec, 0, subgraph);

        // 3. Table-reference vectors (contents patched once tables exist).
        let tensors_vec = fb.vec_u32(&vec![0u32; tensors.len()]);
        let operators_vec = fb.vec_u32(&vec![0u32; ops.len()]);
        let opcodes_vec = fb.vec_u32(&vec![0u32; ops.len()]);
        let buffers_vec = fb.vec_u32(&vec![0u32; buffer_count]);

        // 4. Tensor tables — uoffset slots for shape and name.
        let mut tensor_positions = Vec::with_capacity(tensors.len());
        for (i, _t) in tensors.iter().enumerate() {
            let idx = u32::try_from(i).expect("tensor count fits u32");
            let (tp, slots) = fb.table(&[
                (0u32, Fv::Ref),
                (1u32, Fv::Raw(vec![9u8])), // TensorType.INT8
                (2u32, Fv::Raw(buffer_indices[i].to_le_bytes().to_vec())),
                (3u32, Fv::Ref),
            ]);
            tensor_positions.push((idx, tp, slots[0], slots[1]));
        }
        for (i, (_, tp, _, _)) in tensor_positions.iter().enumerate() {
            fb.patch_vec_elem(tensors_vec, i, *tp);
        }
        fb.patch_ref(&s_tensors, tensors_vec);

        // 5. Shape vectors and name strings, then patch the tensor tables.
        for (i, t) in tensors.iter().enumerate() {
            let shape_pos = fb.vec_i32(&t.shape);
            let name_pos = fb.string(t.name);
            let (_, _, shape_slot, name_slot) = &tensor_positions[i];
            fb.patch_ref(shape_slot, shape_pos);
            fb.patch_ref(name_slot, name_pos);
        }

        // 6. Operator tables — uoffset slots for input/output vectors.
        let mut op_positions = Vec::with_capacity(ops.len());
        for (i, _op) in ops.iter().enumerate() {
            let opcode_index = u32::try_from(i).expect("op count fits u32");
            let (op_pos, slots) = fb.table(&[
                (0u32, Fv::Raw(opcode_index.to_le_bytes().to_vec())),
                (1u32, Fv::Ref),
                (2u32, Fv::Ref),
            ]);
            op_positions.push((op_pos, slots[0], slots[1]));
        }
        for (i, (op_pos, _, _)) in op_positions.iter().enumerate() {
            fb.patch_vec_elem(operators_vec, i, *op_pos);
        }
        fb.patch_ref(&s_operators, operators_vec);

        // 7. Per-op input/output index vectors, then patch the op tables.
        for (i, op) in ops.iter().enumerate() {
            let inputs_vec = fb.vec_u32(&op.inputs);
            let outputs_vec = fb.vec_u32(&op.outputs);
            let (_, in_slot, out_slot) = &op_positions[i];
            fb.patch_ref(in_slot, inputs_vec);
            fb.patch_ref(out_slot, outputs_vec);
        }

        // 8. One OperatorCode table per op (deprecated_builtin_code, field 0).
        let mut opcode_positions = Vec::with_capacity(ops.len());
        for op in &ops {
            let code = u8::try_from(op.builtin_code).expect("test opcode fits a byte");
            let (cp, _) = fb.table(&[(0u32, Fv::Raw(vec![code]))]);
            opcode_positions.push(cp);
        }
        for (i, cp) in opcode_positions.iter().enumerate() {
            fb.patch_vec_elem(opcodes_vec, i, *cp);
        }

        // 9. Buffer tables — uoffset slot for the data vector.
        let mut buffer_positions = Vec::with_capacity(buffer_count);
        for _ in 0..buffer_count {
            let (bp, slots) = fb.table(&[(0u32, Fv::Ref)]);
            buffer_positions.push((bp, slots[0]));
        }
        for (i, (bp, _)) in buffer_positions.iter().enumerate() {
            fb.patch_vec_elem(buffers_vec, i, *bp);
        }
        fb.patch_ref(&s_buffers, buffers_vec);

        // 10. Buffer data vectors (highest address), then patch the buffer
        // tables.
        for (i, data) in buffer_datas.iter().enumerate() {
            let data_pos = fb.vec_bytes(data);
            let (_, data_slot) = &buffer_positions[i];
            fb.patch_ref(data_slot, data_pos);
        }

        // Patch the remaining subgraph fields and the root.
        let inputs_vec = fb.vec_u32(&inputs);
        let outputs_vec = fb.vec_u32(&outputs);
        fb.patch_ref(&s_inputs, inputs_vec);
        fb.patch_ref(&s_outputs, outputs_vec);
        fb.patch_ref(&s_opcodes, opcodes_vec);
        fb.patch_ref(&s_subgraphs, subgraphs_vec);
        fb.finish(model)
    }

    /// Build and parse a model, keeping the bytes alive for the borrow.
    /// `Box::leak` pins the fixture bytes for the process lifetime (tests
    /// only), which lets the parsed model borrow them as `'static`.
    macro_rules! model {
        ($tensors:expr, $ops:expr, $inputs:expr, $outputs:expr) => {{
            let bytes: &'static [u8] =
                Box::leak(build_model($tensors, $ops, $inputs, $outputs).into_boxed_slice());
            let model = flatbuffer::parse(bytes).expect("hand-built model must parse");
            (bytes, model)
        }};
    }

    // ------------------------------------------------------------------
    // Decision-rule unit tests (QA gate: cargo test -- layout_pad_decision)
    // ------------------------------------------------------------------

    #[test]
    fn layout_pad_decision() {
        // C=3 first-layer input: never padded (+433% would be ≈150→800 KiB).
        assert_eq!(super::layout_pad_decision(3), ChannelLayout::Nhwc);
        // C=24: +33% sits exactly on the MAX_PAD_OVERHEAD bound — accepted.
        assert_eq!(super::layout_pad_decision(24), ChannelLayout::NhwcPad16);
        assert_eq!(super::padded_channels(24), 32);
        // C=31: +3.2% — accepted.
        assert_eq!(super::layout_pad_decision(31), ChannelLayout::NhwcPad16);
        // C=23: +39% above the bound — rejected.
        assert_eq!(super::layout_pad_decision(23), ChannelLayout::Nhwc);
        // C=17: +88% — rejected.
        assert_eq!(super::layout_pad_decision(17), ChannelLayout::Nhwc);
        // Already lane-aligned: unchanged, Pad16 is free.
        assert_eq!(super::layout_pad_decision(16), ChannelLayout::NhwcPad16);
        assert_eq!(super::padded_channels(16), 16);
        assert_eq!(super::layout_pad_decision(64), ChannelLayout::NhwcPad16);
        assert_eq!(super::padded_channels(64), 64);
    }

    #[test]
    fn decide_layouts_pads_c24_activation_only() {
        // MobileNetV2-style head: 224²×24 input → 32-channel conv.
        let (_bytes, model) = model!(
            vec![
                BuildTensor::activation(&[1, 224, 224, 24], "input"),
                BuildTensor::constant(&[32, 3, 3, 24], "conv/weights", vec![0u8; 32 * 3 * 3 * 24]),
                BuildTensor::constant(&[32], "conv/bias", vec![0u8; 32]),
                BuildTensor::activation(&[1, 224, 224, 32], "conv/output"),
            ],
            vec![BuildOp { builtin_code: CONV_2D, inputs: vec![0, 1, 2], outputs: vec![3] }],
            vec![0],
            vec![3]
        );
        let plan = decide_layouts(&model);

        assert_eq!(plan.layouts.len(), 4);
        // C=24 activation → pad16 (effective 32).
        assert_eq!(plan.layouts[0].layout, ChannelLayout::NhwcPad16);
        assert_eq!(plan.layouts[0].effective_channels, 32);
        // Constant weights are never padded.
        assert_eq!(plan.layouts[1].layout, ChannelLayout::Nhwc);
        assert_eq!(plan.layouts[2].layout, ChannelLayout::Nhwc);
        // Graph output keeps the caller-visible shape (C=32, unchanged).
        assert_eq!(plan.layouts[3].layout, ChannelLayout::Nhwc);
        assert_eq!(plan.layouts[3].effective_channels, 32);

        // No folds in a plain conv chain.
        assert!(plan.folds.is_empty());
    }

    #[test]
    fn repad_boundary_at_graph_input() {
        // Boundary: consumer (conv) requires pad16, producer (external
        // caller) provides unpadded NHWC bytes → exactly one repad node.
        let (_bytes, model) = model!(
            vec![
                BuildTensor::activation(&[1, 224, 224, 24], "input"),
                BuildTensor::constant(&[32, 3, 3, 24], "conv/weights", vec![0u8; 32 * 3 * 3 * 24]),
                BuildTensor::constant(&[32], "conv/bias", vec![0u8; 32]),
                BuildTensor::activation(&[1, 224, 224, 32], "conv/output"),
            ],
            vec![BuildOp { builtin_code: CONV_2D, inputs: vec![0, 1, 2], outputs: vec![3] }],
            vec![0],
            vec![3]
        );
        let plan = decide_layouts(&model);

        assert_eq!(plan.repads.len(), 1);
        assert_eq!(plan.repads[0].tensor_index, 0);
        assert_eq!(plan.repads[0].target_channels, 32);
    }

    #[test]
    fn no_repad_when_graph_input_already_aligned() {
        // C=32 input: padding is free (effective == channels), so there is
        // no boundary and no copy.
        let (_bytes, model) = model!(
            vec![
                BuildTensor::activation(&[1, 224, 224, 32], "input"),
                BuildTensor::constant(&[64, 3, 3, 32], "conv/weights", vec![0u8; 64 * 3 * 3 * 32]),
                BuildTensor::constant(&[64], "conv/bias", vec![0u8; 64]),
                BuildTensor::activation(&[1, 224, 224, 64], "conv/output"),
            ],
            vec![BuildOp { builtin_code: CONV_2D, inputs: vec![0, 1, 2], outputs: vec![3] }],
            vec![0],
            vec![3]
        );
        let plan = decide_layouts(&model);

        assert!(plan.repads.is_empty());
        assert_eq!(plan.layouts[0].effective_channels, 32); // unchanged
    }

    // ------------------------------------------------------------------
    // Transpose-elimination tests
    // ------------------------------------------------------------------

    #[test]
    fn transpose_before_conv_folds_into_weight_reindex() {
        // Weight [I=24, H=3, W=3, O=32] permuted [3,1,2,0] → [O,H,W,I]
        // ([32,3,3,24]): the classic NHWC-ize.  The runtime TRANSPOSE must
        // disappear; the descriptor carries the re-index math.
        let perm = [3i32, 1, 2, 0].map(|x| x.to_le_bytes()).concat();
        let (_bytes, model) = model!(
            vec![
                BuildTensor::constant(&[24, 3, 3, 32], "conv/weights/src", vec![0u8; 24 * 3 * 3 * 32]),
                BuildTensor::constant(&[4], "transpose/perm", perm),
                BuildTensor::activation(&[32, 3, 3, 24], "conv/weights"),
                BuildTensor::activation(&[1, 8, 8, 24], "input"),
                BuildTensor::constant(&[32], "conv/bias", vec![0u8; 32]),
                BuildTensor::activation(&[1, 8, 8, 32], "conv/output"),
            ],
            vec![
                BuildOp { builtin_code: TRANSPOSE, inputs: vec![0, 1], outputs: vec![2] },
                BuildOp { builtin_code: CONV_2D, inputs: vec![3, 2, 4], outputs: vec![5] },
            ],
            vec![3],
            vec![5]
        );
        let plan = decide_layouts(&model);

        assert_eq!(plan.folds.len(), 1);
        let fold = &plan.folds[0];
        assert_eq!(fold.op_index, 0);
        assert_eq!(fold.weight_index, 0);
        assert_eq!(fold.permutation, vec![3, 1, 2, 0]);
        assert_eq!(fold.weight_shape, vec![32, 3, 3, 24]);
        // dst[o,h,w,i] = src[perm[0],perm[1],perm[2],perm[3]] = src[i,h,w,o]
        // — a pure reindexing (NHWC↔HWCN style), no element loss.
    }

    #[test]
    fn reshape_relabel_folds_as_pure_shape_change() {
        // Moving a unit dim is a pure relabel: no element reorders, so the
        // runtime op drops and only the shape descriptor changes.
        let shape = [3i32, 3, 16, 1].map(|x| x.to_le_bytes()).concat();
        let (_bytes, model) = model!(
            vec![
                BuildTensor::constant(&[1, 3, 3, 16], "conv/weights/src", vec![0u8; 1 * 3 * 3 * 16]),
                BuildTensor::constant(&[4], "reshape/shape", shape),
                BuildTensor::activation(&[3, 3, 16, 1], "conv/weights"),
                BuildTensor::activation(&[1, 8, 8, 16], "input"),
                BuildTensor::constant(&[32], "conv/bias", vec![0u8; 32]),
                BuildTensor::activation(&[1, 8, 8, 32], "conv/output"),
            ],
            vec![
                BuildOp { builtin_code: RESHAPE, inputs: vec![0, 1], outputs: vec![2] },
                BuildOp { builtin_code: CONV_2D, inputs: vec![3, 2, 4], outputs: vec![5] },
            ],
            vec![3],
            vec![5]
        );
        let plan = decide_layouts(&model);

        assert_eq!(plan.folds.len(), 1);
        let fold = &plan.folds[0];
        assert_eq!(fold.op_index, 0);
        assert_eq!(fold.weight_index, 0);
        assert!(fold.permutation.is_empty()); // no reordering needed
        assert_eq!(fold.weight_shape, vec![3, 3, 16, 1]);
    }

    #[test]
    fn transpose_not_folded_for_non_weight_or_lossy_perms() {
        // (a) TRANSPOSE feeding the conv *activation* input is left alone.
        let perm_act = [0i32, 3, 1, 2].map(|x| x.to_le_bytes()).concat();
        let (_bytes, m_activation) = model!(
            vec![
                BuildTensor::activation(&[1, 8, 8, 24], "act/src"),
                BuildTensor::constant(&[4], "transpose/perm", perm_act),
                BuildTensor::activation(&[1, 24, 8, 8], "act/transposed"),
                BuildTensor::constant(&[32, 3, 3, 24], "conv/weights", vec![0u8; 32 * 3 * 3 * 24]),
                BuildTensor::constant(&[32], "conv/bias", vec![0u8; 32]),
                BuildTensor::activation(&[1, 8, 8, 32], "conv/output"),
            ],
            vec![
                BuildOp { builtin_code: TRANSPOSE, inputs: vec![0, 1], outputs: vec![2] },
                BuildOp { builtin_code: CONV_2D, inputs: vec![2, 3, 4], outputs: vec![5] },
            ],
            vec![0],
            vec![5]
        );
        assert!(decide_layouts(&m_activation).folds.is_empty());

        // (b) A 3-D weight transpose feeding a 4-D consumer is not a pure
        // 4-D weight reindex.
        let perm3 = [2i32, 1, 0].map(|x| x.to_le_bytes()).concat();
        let (_bytes, m_3d) = model!(
            vec![
                BuildTensor::constant(&[3, 3, 24], "conv/weights/src", vec![0u8; 3 * 3 * 24]),
                BuildTensor::constant(&[3], "transpose/perm", perm3),
                BuildTensor::activation(&[24, 3, 3], "conv/weights"),
                BuildTensor::activation(&[1, 8, 8, 24], "input"),
                BuildTensor::constant(&[32], "conv/bias", vec![0u8; 32]),
                BuildTensor::activation(&[1, 8, 8, 32], "conv/output"),
            ],
            vec![
                BuildOp { builtin_code: TRANSPOSE, inputs: vec![0, 1], outputs: vec![2] },
                BuildOp { builtin_code: CONV_2D, inputs: vec![3, 2, 4], outputs: vec![5] },
            ],
            vec![3],
            vec![5]
        );
        assert!(decide_layouts(&m_3d).folds.is_empty());

        // (c) A lossy permutation (duplicate axis) is a gather, not a
        // reindex — never folded.
        let perm_lossy = [0i32, 0, 1, 2].map(|x| x.to_le_bytes()).concat();
        let (_bytes, m_lossy) = model!(
            vec![
                BuildTensor::constant(&[24, 3, 3, 32], "conv/weights/src", vec![0u8; 24 * 3 * 3 * 32]),
                BuildTensor::constant(&[4], "transpose/perm", perm_lossy),
                BuildTensor::activation(&[24, 3, 3, 32], "conv/weights"),
                BuildTensor::activation(&[1, 8, 8, 24], "input"),
                BuildTensor::constant(&[32], "conv/bias", vec![0u8; 32]),
                BuildTensor::activation(&[1, 8, 8, 32], "conv/output"),
            ],
            vec![
                BuildOp { builtin_code: TRANSPOSE, inputs: vec![0, 1], outputs: vec![2] },
                BuildOp { builtin_code: CONV_2D, inputs: vec![3, 2, 4], outputs: vec![5] },
            ],
            vec![3],
            vec![5]
        );
        assert!(decide_layouts(&m_lossy).folds.is_empty());

        // (d) A layout-changing RESHAPE (merges H*W) is not a pure relabel.
        let shape = [1i32, 1, 9, 16].map(|x| x.to_le_bytes()).concat();
        let (_bytes, m_reshape) = model!(
            vec![
                BuildTensor::constant(&[1, 3, 3, 16], "conv/weights/src", vec![0u8; 1 * 3 * 3 * 16]),
                BuildTensor::constant(&[4], "reshape/shape", shape),
                BuildTensor::activation(&[1, 1, 9, 16], "conv/weights"),
                BuildTensor::activation(&[1, 8, 8, 16], "input"),
                BuildTensor::constant(&[32], "conv/bias", vec![0u8; 32]),
                BuildTensor::activation(&[1, 8, 8, 32], "conv/output"),
            ],
            vec![
                BuildOp { builtin_code: RESHAPE, inputs: vec![0, 1], outputs: vec![2] },
                BuildOp { builtin_code: CONV_2D, inputs: vec![3, 2, 4], outputs: vec![5] },
            ],
            vec![3],
            vec![5]
        );
        assert!(decide_layouts(&m_reshape).folds.is_empty());

        // (e) A TRANSPOSE not immediately followed by its consumer is not
        // folded ("directly precedes" gate).
        let perm_e = [3i32, 1, 2, 0].map(|x| x.to_le_bytes()).concat();
        let (_bytes, m_stray) = model!(
            vec![
                BuildTensor::constant(&[24, 3, 3, 32], "conv/weights/src", vec![0u8; 24 * 3 * 3 * 32]),
                BuildTensor::constant(&[4], "transpose/perm", perm_e),
                BuildTensor::activation(&[32, 3, 3, 24], "conv/weights"),
                BuildTensor::constant(&[1], "noop/scale", vec![1u8]),
                BuildTensor::activation(&[1, 8, 8, 24], "input"),
                BuildTensor::constant(&[32], "conv/bias", vec![0u8; 32]),
                BuildTensor::activation(&[1, 8, 8, 32], "conv/output"),
            ],
            vec![
                BuildOp { builtin_code: TRANSPOSE, inputs: vec![0, 1], outputs: vec![2] },
                BuildOp { builtin_code: MUL, inputs: vec![2, 3], outputs: vec![2] },
                BuildOp { builtin_code: CONV_2D, inputs: vec![4, 2, 5], outputs: vec![6] },
            ],
            vec![4],
            vec![6]
        );
        assert!(decide_layouts(&m_stray).folds.is_empty());
    }

}
