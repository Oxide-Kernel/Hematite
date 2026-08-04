// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! hematite-memory — USMP-style arena allocator for Hematite int8 inference.
//!
//! # Overview
//!
//! Hematite is a compile-time inference library for the ESP32-S3.  All
//! intermediate tensors are statically sized at codegen time.  This crate
//! pre-sizes the scratch buffer and hands out offsets so that the device
//! **never allocates at runtime**.
//!
//! # USMP-style strategy
//!
//! USMP (Unified Static Memory Planner) determines which intermediate
//! tensors can share the same memory because their liveness intervals do
//! not overlap.  Given a fixed op schedule, [`liveness_plan`] computes
//! each tensor's live range, then coalesces non-overlapping buffers
//! using greedy-by-size first-fit allocation with 16‑byte alignment.
//!
//! # Ownership contract with codegen (T4.2b)
//!
//! The `hematite-codegen` crate in Phase 4 builds [`OpInfo`] schedules
//! from parsed TFLite models and calls [`liveness_plan`] to compute
//! arena layouts.  **Codegen must never re-implement the liveness
//! algorithm** — this crate is the single source of truth.
//!
//! # 16‑byte alignment
//!
//! All offset math enforces 16‑byte alignment.  The Phase 3 SIMD kernels
//! (`hematite-s3`, TIE728 instructions) require 128‑bit aligned loads —
//! unaligned addresses trigger 3‑instruction fixup sequences.
//!
//! # KWS 60 KB budget
//!
//! The canonical KWS (keyword-spotting) model has a few conv/depthwise
//! layers, ~10–20 ops, with activation tensors in the 4–64 KB range.
//! The plan's QA line asserts peak arena ≤ 60 KB for a 512 KB SRAM
//! target, leaving ≥ 452 KB for model weights, runtime, and OS.

#![no_std]

// ── Constants ───────────────────────────────────────────────────────────────

/// Maximum number of inputs or outputs per operation.
///
/// TFLite ops typically have at most 3 inputs (e.g. conv: input,
/// filter, bias) and 1 output.  4 slots provides headroom for ops
/// with optional bias or residual connections.
pub const MAX_IO_PER_OP: usize = 4;

/// Maximum number of tensors in a single model subgraph.
///
/// TFLite subgraphs allow up to 64 tensors per subgraph.  This cap
/// keeps stack usage bounded and matches the `offsets` array width
/// in [`ArenaPlan`].
pub const MAX_TENSORS: usize = 64;

/// Maximum number of blocks tracked by a [`ScratchLayout`].
pub const MAX_SCRATCH_BLOCKS: usize = 64;

/// Minimum alignment enforced on every arena allocation.
/// SIMD kernels (TIE728) require 128‑bit aligned loads.
const MIN_ALIGN: usize = 16;

// ── OpInfo ──────────────────────────────────────────────────────────────────

/// Describes a single operation in the execution schedule.
///
/// All arrays are fixed-size; use the `*_count` fields to indicate
/// how many slots are populated.  This struct is constructible in
/// `const` context — codegen bakes schedules into the firmware binary.
///
/// # Import contract (T4.2b)
///
/// `hematite-codegen` builds `OpInfo` schedules from parsed TFLite
/// models and passes them to [`liveness_plan`].  The codegen must
/// supply:
///
/// * `input_ids` / `input_count` — tensor ids consumed by this op
/// * `output_ids` / `output_count` — tensor ids produced by this op
/// * `in_place` — if `true`, `output_ids[0]` reuses `input_ids[0]`'s
///   arena slot (the op writes its result over the first input)
///
/// Additionally, `liveness_plan` requires `tensor_byte_sizes`,
/// `model_input_indices`, and `model_output_indices` passed as
/// separate slices — these describe global tensor properties rather
/// than per-op state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpInfo {
    /// Operation kind tag (codegen-assigned; planner treats as opaque).
    pub op_kind: u16,

    /// Tensor IDs consumed by this op.
    pub input_ids: [u16; MAX_IO_PER_OP],
    /// Number of valid entries in `input_ids`.
    pub input_count: u8,

    /// Tensor IDs produced by this op.
    pub output_ids: [u16; MAX_IO_PER_OP],
    /// Number of valid entries in `output_ids`.
    pub output_count: u8,

    /// If `true`, `output_ids[0]` reuses `input_ids[0]`'s arena slot.
    /// In-place ops update a buffer without requiring a new allocation.
    pub in_place: bool,
}

// ── LayoutError ─────────────────────────────────────────────────────────────

/// Errors returned by the arena planner and scratch allocator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutError {
    /// A single tensor's size exceeds `max_internal` (SRAM budget)
    /// and PSRAM split is not available.
    Oversized,

    /// Peak arena usage exceeds `max_internal` and PSRAM split
    /// could not bring it within budget (either `psram_budget` was
    /// `None` or the PSRAM pool is also exhausted).
    OutOfBudget,

    /// The scratch layout has no free block slots left.
    NoSpace,
}

// ── PsramPool ───────────────────────────────────────────────────────────────

/// Result of splitting oversized or spill tensors into PSRAM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PsramPool {
    /// Total bytes moved to PSRAM.
    pub total_bytes: usize,
    /// Bitmask: bit i set ⇒ tensor i lives in PSRAM.
    pub tensor_mask: u64,
}

// ── ArenaPlan ───────────────────────────────────────────────────────────────

/// Result of the liveness-based arena allocation pass.
///
/// `offsets[i]` is the arena byte offset for tensor `i`, or
/// [`OFFSET_NONE`] if tensor `i` is kept out of the arena (model
/// input/output buffers live in caller-owned memory, or tensor
/// spilled to PSRAM).
///
/// The generated code declares:
///
/// ```ignore
/// static mut ARENA: [u8; PLAN.peak_arena_bytes] = [0u8; PLAN.peak_arena_bytes];
/// ```
///
/// and accesses each intermediate tensor as
/// `&mut ARENA[PLAN.offsets[tensor_id]..]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaPlan {
    /// Peak arena usage in bytes (always 16‑byte aligned).
    pub peak_arena_bytes: usize,

    /// Per-tensor byte offsets within the arena.
    ///
    /// `offsets[i] == OFFSET_NONE` marks tensors kept out of the
    /// arena (model inputs/outputs, or tensors spilled to PSRAM).
    pub offsets: [usize; MAX_TENSORS],

    /// Number of model tensors (informs codegen how many `offsets`
    /// entries are meaningful).
    pub tensor_count: u8,

    /// PSRAM pool allocation, if `psram_budget` was `Some(_)` and
    /// tensors were spilled to PSRAM.
    pub psram_split: Option<PsramPool>,
}

/// Sentinel for [`ArenaPlan::offsets`] entries that do not live in
/// the arena.
pub const OFFSET_NONE: usize = usize::MAX;

// ── ScratchLayout ───────────────────────────────────────────────────────────

/// Internal block record for [`ScratchLayout`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ScratchBlock {
    offset: usize,
    size: usize,
}

/// Fixed-capacity bump allocator for scratch buffers.
///
/// Allocates blocks sequentially, tracking the high-water mark via
/// [`peak`](ScratchLayout::peak).  [`reset`](ScratchLayout::reset)
/// clears all blocks for reuse.
///
/// # Capacity
///
/// Tracks at most [`MAX_SCRATCH_BLOCKS`] (64) allocations.  The
/// backing array is const-constructible.
///
/// # Example
///
/// ```rust
/// use hematite_memory::{ScratchLayout, LayoutError};
///
/// let mut sl = ScratchLayout::new();
/// let a = sl.allocate(0, 128, 16).unwrap();
/// let b = sl.allocate(a + 128, 64, 16).unwrap();
/// assert_eq!(sl.peak(), a + 128 + 64);
/// sl.reset();
/// assert_eq!(sl.peak(), 0);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct ScratchLayout {
    blocks: [ScratchBlock; MAX_SCRATCH_BLOCKS],
    block_count: usize,
}

impl ScratchLayout {
    /// Create an empty scratch layout.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            blocks: [ScratchBlock {
                offset: 0,
                size: 0,
            }; MAX_SCRATCH_BLOCKS],
            block_count: 0,
        }
    }

    /// Allocate a block of `size` bytes with `align` alignment,
    /// starting the search at `offset`.
    ///
    /// The returned offset satisfies `ret % max(16, align) == 0`.
    /// Sixteen‑byte alignment is always applied on top of the
    /// requested alignment.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError::NoSpace`] if the block table is full.
    pub fn allocate(
        &mut self,
        offset: usize,
        size: usize,
        align: usize,
    ) -> Result<usize, LayoutError> {
        if self.block_count >= MAX_SCRATCH_BLOCKS {
            return Err(LayoutError::NoSpace);
        }
        let effective_align = if align > MIN_ALIGN { align } else { MIN_ALIGN };
        let aligned = align_up(offset, effective_align);

        self.blocks[self.block_count] = ScratchBlock {
            offset: aligned,
            size,
        };
        self.block_count += 1;

        Ok(aligned)
    }

    /// Return the high-water offset (end of the furthest-placed block).
    #[must_use]
    pub fn peak(&self) -> usize {
        let mut max_end: usize = 0;
        for i in 0..self.block_count {
            let blk = &self.blocks[i];
            let end = blk.offset.saturating_add(blk.size);
            if end > max_end {
                max_end = end;
            }
        }
        max_end
    }

    /// Clear all recorded blocks (reset to empty layout).
    pub fn reset(&mut self) {
        self.block_count = 0;
    }
}

impl Default for ScratchLayout {
    fn default() -> Self {
        Self::new()
    }
}

// ── Liveness internal types ─────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
struct Cand {
    tid: u16,
    size: usize,
    fw: usize,
    lr: usize,
}

#[derive(Clone, Copy)]
struct SpillEntry {
    tid: u16,
    size: usize,
}

impl SpillEntry {
    const fn empty() -> Self {
        Self { tid: 0, size: 0 }
    }
}

// ── Liveness algorithm ──────────────────────────────────────────────────────

/// Compute arena allocation for a fixed op schedule.
///
/// # Algorithm
///
/// 1. **Liveness intervals.**  Scan the schedule to find each
///    tensor's `[first_written, last_read]` op index range.  Model
///    inputs are treated as written before op 0; model outputs are
///    treated as read after the last op.  For in‑place ops, the
///    output inherits the input's first-write index.
///
/// 2. **Filter.**  Exclude model inputs and outputs — they live in
///    caller-owned buffers.  Also exclude tensors with zero byte
///    size or that are never written.
///
/// 3. **Sort.**  Order remaining tensors by byte size descending
///    (greedy‑by‑size).  Break ties by tensor id for determinism.
///
/// 4. **Coalesce.**  First-fit: for each tensor, scan arena slots
///    for the earliest offset where no already‑placed live tensor
///    collides.  Collision means: offset ranges overlap **and**
///    liveness intervals overlap.  All offsets round up to 16.
///
/// 5. **PSRAM split** (optional).  If peak exceeds `max_internal`
///    and `psram_budget` is `Some(b)`, spills tensors to PSRAM
///    (largest first) until SRAM peak fits within budget or PSRAM
///    is exhausted.  Spilled tensors get [`OFFSET_NONE`].
///
/// # Determinism
///
/// Identical inputs always produce an identical `ArenaPlan`.  No
/// hash maps, no randomness — fully deterministic.
///
/// # Errors
///
/// Returns [`LayoutError::Oversized`] if any single tensor exceeds
/// `max_internal` and PSRAM split is not available.
/// Returns [`LayoutError::OutOfBudget`] if peak exceeds budget
/// after PSRAM split attempts.
pub fn liveness_plan(
    op_schedule: &[OpInfo],
    tensor_byte_sizes: &[usize],
    model_input_indices: &[u16],
    model_output_indices: &[u16],
    max_internal: usize,
    psram_budget: Option<usize>,
) -> Result<ArenaPlan, LayoutError> {
    if op_schedule.is_empty() {
        return Ok(ArenaPlan {
            peak_arena_bytes: 0,
            offsets: [OFFSET_NONE; MAX_TENSORS],
            tensor_count: 0,
            psram_split: None,
        });
    }

    let num_ops = op_schedule.len();
    let last_op = num_ops.saturating_sub(1);

    // Step 1: liveness intervals
    let mut first_written = [OFFSET_NONE; MAX_TENSORS];
    let mut last_read = [0_usize; MAX_TENSORS];
    let mut max_tid: usize = 0;

    for &id in model_input_indices {
        let idx = id as usize;
        if idx < MAX_TENSORS {
            first_written[idx] = 0;
            if idx > max_tid {
                max_tid = idx;
            }
        }
    }

    for (op_idx, op) in op_schedule.iter().enumerate() {
        for i in 0..op.input_count as usize {
            let tid = op.input_ids[i] as usize;
            if tid < MAX_TENSORS {
                last_read[tid] = last_read[tid].max(op_idx);
                if tid > max_tid {
                    max_tid = tid;
                }
            }
        }
        for i in 0..op.output_count as usize {
            let tid = op.output_ids[i] as usize;
            if tid < MAX_TENSORS {
                let fw = first_written[tid];
                if fw == OFFSET_NONE || op_idx < fw {
                    first_written[tid] = op_idx;
                }
                last_read[tid] = last_read[tid].max(op_idx);
                if tid > max_tid {
                    max_tid = tid;
                }
            }
        }
    }

    for &id in model_output_indices {
        let idx = id as usize;
        if idx < MAX_TENSORS {
            last_read[idx] = last_read[idx].max(last_op + 1);
            if idx > max_tid {
                max_tid = idx;
            }
        }
    }

    // In-place: output inherits input's first_written.
    // Iterate to fixed-point because in-place chains (a→b→c) can
    // form when multiple in-place ops compose.
    loop {
        let mut changed = false;
        for op in op_schedule {
            if op.in_place && op.input_count > 0 && op.output_count > 0 {
                let in_id = op.input_ids[0] as usize;
                let out_id = op.output_ids[0] as usize;
                if in_id < MAX_TENSORS && out_id < MAX_TENSORS {
                    let in_fw = first_written[in_id];
                    let out_fw = first_written[out_id];
                    if in_fw != OFFSET_NONE && (out_fw == OFFSET_NONE || in_fw < out_fw) {
                        first_written[out_id] = in_fw;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    let tensor_count = if max_tid > 0 { max_tid + 1 } else { 0 };

    // Step 2: model I/O exclusion
    let mut is_model_input = [false; MAX_TENSORS];
    let mut is_model_output = [false; MAX_TENSORS];
    for &id in model_input_indices {
        let idx = id as usize;
        if idx < MAX_TENSORS {
            is_model_input[idx] = true;
        }
    }
    for &id in model_output_indices {
        let idx = id as usize;
        if idx < MAX_TENSORS {
            is_model_output[idx] = true;
        }
    }

    // Step 3: candidate list
    let mut cands: [Option<Cand>; MAX_TENSORS] = [None; MAX_TENSORS];
    let mut n_cands: usize = 0;

    for tid in 0..tensor_count {
        if is_model_input[tid] || is_model_output[tid] {
            continue;
        }
        let size = tensor_byte_sizes.get(tid).copied().unwrap_or(0);
        if size == 0 || size == OFFSET_NONE {
            continue;
        }
        let fw = first_written[tid];
        if fw == OFFSET_NONE {
            continue;
        }
        if size > max_internal && psram_budget.is_none() {
            return Err(LayoutError::Oversized);
        }
        cands[n_cands] = Some(Cand {
            tid: tid as u16,
            size,
            fw,
            lr: last_read[tid],
        });
        n_cands += 1;
    }

    sort_candidates_by_size(&mut cands[..n_cands]);

    // Step 4: in-place mapping
    let mut ip_map: [u16; MAX_TENSORS] = [u16::MAX; MAX_TENSORS];
    for op in op_schedule {
        if op.in_place && op.input_count > 0 && op.output_count > 0 {
            let out_id = op.output_ids[0] as usize;
            let in_id = op.input_ids[0] as usize;
            if out_id < MAX_TENSORS && in_id < MAX_TENSORS {
                ip_map[out_id] = in_id as u16;
            }
        }
    }

    // Step 5: first-fit coalescing
    // placed: (tensor_id, offset, size, first_written, last_read)
    let mut placed: [(u16, usize, usize, usize, usize); MAX_TENSORS] =
        [(0, 0, 0, 0, 0); MAX_TENSORS];
    let mut n_placed: usize = 0;
    let mut offsets = [OFFSET_NONE; MAX_TENSORS];

    // Track tensors explicitly spilled to PSRAM during first-fit
    let mut psram_candidates: [u16; MAX_TENSORS] = [0; MAX_TENSORS];
    let mut n_psram: usize = 0;

    for cand_opt in cands.iter().take(n_cands) {
        let c = match cand_opt {
            Some(c) => *c,
            None => continue,
        };

        let ip_input = ip_map[c.tid as usize];
        if ip_input != u16::MAX {
            let input_off = offsets[ip_input as usize];
            if input_off != OFFSET_NONE {
                let input_sz = tensor_byte_sizes
                    .get(ip_input as usize)
                    .copied()
                    .unwrap_or(0);
                let actual_sz = if c.size > 0 { c.size } else { input_sz };
                let aligned_sz = align_up(actual_sz, MIN_ALIGN);
                // Check that reusing input's offset does not collide with
                // OTHER already-placed tensors (excluding the input itself).
                if !collides_placed(input_off, aligned_sz, c.fw, c.lr, &placed, n_placed, ip_input)
                {
                    offsets[c.tid as usize] = input_off;
                    placed[n_placed] = (c.tid, input_off, aligned_sz, c.fw, c.lr);
                    n_placed += 1;
                    continue;
                }
                // Collision — fall through to normal allocation
            }
        }

        let aligned_sz = align_up(c.size, MIN_ALIGN);
        let mut off: usize = 0;
        let mut found = false;

        while off.saturating_add(aligned_sz) <= max_internal {
            let aoff = align_up(off, MIN_ALIGN);
            if !collides_placed(aoff, aligned_sz, c.fw, c.lr, &placed, n_placed, u16::MAX) {
                offsets[c.tid as usize] = aoff;
                placed[n_placed] = (c.tid, aoff, aligned_sz, c.fw, c.lr);
                n_placed += 1;
                found = true;
                break;
            }
            off = aoff.saturating_add(MIN_ALIGN);
        }

        if !found {
            if psram_budget.is_some() {
                offsets[c.tid as usize] = OFFSET_NONE;
                psram_candidates[n_psram] = c.tid;
                n_psram += 1;
            } else {
                return Err(LayoutError::OutOfBudget);
            }
        }
    }

    // Step 6: peak
    let raw_peak = placed_peak(&placed, n_placed);
    let mut peak = align_up(raw_peak, MIN_ALIGN);

    // Enforce model I/O → OFFSET_NONE
    for &id in model_input_indices {
        let idx = id as usize;
        if idx < MAX_TENSORS {
            offsets[idx] = OFFSET_NONE;
        }
    }
    for &id in model_output_indices {
        let idx = id as usize;
        if idx < MAX_TENSORS {
            offsets[idx] = OFFSET_NONE;
        }
    }

    // Step 7: PSRAM split
    let psram_split = if let Some(budget) = psram_budget {
        // If peak exceeds max_internal, try to spill placed tensors
        if peak > max_internal {
            let result = psram_spill_from_placed(
                &mut offsets,
                &mut placed,
                &mut n_placed,
                &mut peak,
                max_internal,
                budget,
                0,
                0,
            )?;
            // result = (spilled_total, spilled_mask)
            // Add to psram_candidates
            let (extra_total, extra_mask) = result;
            // Merge with psram_candidates below...
            build_psram_pool(
                &psram_candidates[..n_psram],
                tensor_byte_sizes,
                extra_total,
                extra_mask,
                budget,
            )?
        } else if n_psram > 0 {
            // Peak fits but some tensors were explicitly spilled
            build_psram_pool(
                &psram_candidates[..n_psram],
                tensor_byte_sizes,
                0,
                0,
                budget,
            )?
        } else {
            None
        }
    } else if peak > max_internal {
        return Err(LayoutError::OutOfBudget);
    } else {
        None
    };

    Ok(ArenaPlan {
        peak_arena_bytes: peak,
        offsets,
        tensor_count: tensor_count as u8,
        psram_split,
    })
}

// ── Internal helpers ────────────────────────────────────────────────────────

#[inline]
const fn align_up(x: usize, align: usize) -> usize {
    (x.wrapping_add(align.wrapping_sub(1))) & !align.wrapping_sub(1)
}

fn placed_peak(placed: &[(u16, usize, usize, usize, usize)], n: usize) -> usize {
    let mut max_end: usize = 0;
    for &(_tid, off, sz, _fw, _lr) in placed.iter().take(n) {
        let end = off.saturating_add(sz);
        if end > max_end {
            max_end = end;
        }
    }
    max_end
}

fn sort_candidates_by_size(cands: &mut [Option<Cand>]) {
    let n = cands.len();
    if n < 2 {
        return;
    }
    for i in 1..n {
        let key = cands[i];
        let (key_size, key_tid) = match key {
            Some(c) => (c.size, c.tid),
            None => continue,
        };
        let mut j = i;
        while j > 0 {
            let prev = cands[j - 1];
            let (prev_size, prev_tid) = match prev {
                Some(c) => (c.size, c.tid),
                None => break,
            };
            let should_swap = key_size > prev_size
                || (key_size == prev_size && key_tid < prev_tid);
            if should_swap {
                cands[j] = cands[j - 1];
                j -= 1;
            } else {
                break;
            }
        }
        cands[j] = key;
    }
}

fn collides_placed(
    offset: usize,
    size: usize,
    fw: usize,
    lr: usize,
    placed: &[(u16, usize, usize, usize, usize)],
    n_placed: usize,
    exclude_tid: u16,
) -> bool {
    let end = offset.saturating_add(size);
    for &(tid, p_off, p_sz, p_fw, p_lr) in placed.iter().take(n_placed) {
        if p_sz == 0 {
            continue;
        }
        if tid == exclude_tid {
            continue;
        }
        let p_end = p_off.saturating_add(p_sz);
        if offset >= p_end || end <= p_off {
            continue;
        }
        if fw <= p_lr && p_fw <= lr {
            return true;
        }
    }
    false
}

/// Spill tensors from the placed set until SRAM peak fits within
/// `max_internal`.  Returns the total bytes spilled and the tensor
/// bitmask.  Does NOT check PSRAM budget — the caller must do that.
#[allow(clippy::too_many_arguments)]
fn psram_spill_from_placed(
    offsets: &mut [usize; MAX_TENSORS],
    placed: &mut [(u16, usize, usize, usize, usize); MAX_TENSORS],
    n_placed: &mut usize,
    peak: &mut usize,
    max_internal: usize,
    psram_budget: usize,
    mut psram_total: usize,
    mut psram_mask: u64,
) -> Result<(usize, u64), LayoutError> {
    let mut spillable: [SpillEntry; MAX_TENSORS] = [SpillEntry::empty(); MAX_TENSORS];
    let mut n_spill: usize = 0;

    for &(tid, _off, sz, _fw, _lr) in placed.iter().take(*n_placed) {
        if sz > 0 {
            spillable[n_spill] = SpillEntry { tid, size: sz };
            n_spill += 1;
        }
    }

    // Sort by size descending
    for i in 1..n_spill {
        let key = spillable[i];
        let mut j = i;
        while j > 0 && spillable[j - 1].size < key.size {
            spillable[j] = spillable[j - 1];
            j -= 1;
        }
        spillable[j] = key;
    }

    for entry in spillable.iter().take(n_spill) {
        let current_peak = placed_peak(placed, *n_placed);
        if current_peak <= max_internal {
            break;
        }

        let tid = entry.tid;
        let size = entry.size;

        if psram_total.saturating_add(size) > psram_budget {
            continue;
        }

        let mut write: usize = 0;
        for read in 0..*n_placed {
            if placed[read].0 == tid {
                continue;
            }
            if placed[read].2 > 0 {
                if write != read {
                    placed[write] = placed[read];
                }
                write += 1;
            }
        }
        *n_placed = write;

        psram_total = psram_total.saturating_add(size);
        psram_mask |= 1u64 << (tid as u64);

        let tidx = tid as usize;
        if tidx < MAX_TENSORS {
            offsets[tidx] = OFFSET_NONE;
        }
    }

    let final_peak = placed_peak(placed, *n_placed);
    if final_peak > max_internal {
        return Err(LayoutError::OutOfBudget);
    }
    *peak = align_up(final_peak, MIN_ALIGN);

    Ok((psram_total, psram_mask))
}

/// Build a PsramPool from spilled tensor candidates.  Computes total
/// PSRAM bytes, checks against budget, returns `None` if nothing was
/// spilled.
fn build_psram_pool(
    psram_candidates: &[u16],
    tensor_byte_sizes: &[usize],
    extra_total: usize,
    extra_mask: u64,
    psram_budget: usize,
) -> Result<Option<PsramPool>, LayoutError> {
    let mut total = extra_total;
    let mut mask = extra_mask;

    for &tid in psram_candidates {
        let idx = tid as usize;
        let size = tensor_byte_sizes.get(idx).copied().unwrap_or(0);
        if size == 0 || size == OFFSET_NONE {
            continue;
        }
        // Don't double-count — if already in extra_mask, skip
        if mask & (1u64 << (tid as u64)) != 0 {
            continue;
        }
        total = total.saturating_add(size);
        mask |= 1u64 << (tid as u64);
    }

    if total > psram_budget {
        return Err(LayoutError::OutOfBudget);
    }

    Ok(if total > 0 {
        Some(PsramPool {
            total_bytes: total,
            tensor_mask: mask,
        })
    } else {
        None
    })
}
