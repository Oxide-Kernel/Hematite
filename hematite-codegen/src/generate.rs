// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! T4.1 — straight-line code emitter: [`ParsedModel`] → typed `KernelBackend`
//! call sequence + `Model<B>` wrapper.
//!
//! The emitter runs at **compile time of the consumer crate** (host-side,
//! inside the proc-macro).  All device-side math is precomputed here and
//! emitted as integer consts — the generated code contains no `f32`/`f64`
//! arithmetic, no `unsafe`, no heap allocation, and no panic paths.
//!
//! ## Quantization contract (mirrors `hematite-tests/goldens/*.rs`)
//!
//! * Per-channel output multipliers: `quantize_multiplier(in_scale ·
//!   filter_scale[oc] / out_scale)` — the TFLM `QuantizeMultiplier` f64
//!   (frexp) formula, copied from `hematite-int8` (host-only; this crate is
//!   not a dependency of the proc-macro).
//! * Fused activation: `CalculateActivationRangeQuantized` against the
//!   output tensor's scale/zero-point (0=NONE, 1=RELU, 3=RELU6; 2/4/5 →
//!   NONE).
//! * Offsets: `input_offset = -in_zp`, `output_offset = out_zp`,
//!   `weights_offset = w_zp` — the kernels consume raw int8 weights plus the
//!   params zero-point, exactly like the golden fixtures (all fixtures use
//!   zp = 0, so these reduce to 0 there).
//! * Scratch: a **macro-time** const, `SCRATCH_LEN`, computed as the max over
//!   ops of the per-op scratch need recomputed from the parsed op params (the
//!   S3 backend's `conv1x1_scratch_need` / `conv3x3_scratch_need` /
//!   `depthwise_scratch_need` / `softmax_scratch_size` formulas, mirrored in
//!   this file — see the `scratch_need_*` helpers below).  Generated code
//!   sizes the scratch array with this const — never `[0u8; B::scratch()]`
//!   inside a `const fn` (unstable const-trait-call trap).
//!
//! ## T1.3 — liveness arena
//!
//! When `optimize::arena::plan_arena` succeeds for the model (every
//! intermediate fits `MAX_INTERNAL`), the per-tensor stack arrays collapse
//! into ONE `#[repr(C, align(16))] struct Arena { data: [i8; ARENA_LEN] }`
//! local sized to the planner's liveness peak, indexed at compile-time
//! offsets.  Each op call borrows its disjoint slices via nested
//! `split_at_mut` (safe, no `unsafe` in generated code); the planner
//! guarantees a single op's input/output slices never overlap because they
//! are simultaneously live.  Models the planner rejects (mobilenet_v2: a
//! single 224×224×32 activation exceeds the 512 KiB budget) keep per-tensor
//! stack emission — bit-exact, just larger (evidence:
//! `local-notes/evidence/composed-kernels/t13-arena.md`).

use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use crate::flatbuffer::{ParsedModel, ParsedOp, ParsedOptions, ParsedTensor, QuantInfo, TensorType};
use crate::optimize::arena::plan_arena;
use crate::optimize::fusion::{
    AbsorbedElementwise, ElementwiseKind as FusedStepKind, FusedActivationKind, FusedGroup,
    FusedSchedule,
};
use hematite_memory::{ArenaPlan, OFFSET_NONE};

/// Emit the full model wrapper for `subgraph[0]` of a parsed model — the
/// plain **unfused** per-op straight-line sequence (no fusion schedule).
///
/// This is the T4.1 emission, kept reachable as the unfused reference of the
/// T1.2 fused-vs-unfused equivalence gate (and by `#[model_unfused]`).
pub(crate) fn emit_model(model: &ParsedModel) -> Result<TokenStream, String> {
    emit_model_with(model, None)
}

/// Emit the full model wrapper for `subgraph[0]`, honoring a fused schedule
/// (T1.2 wiring of the T4.2a fusion pass).
///
/// Groups whose anchor has a composed kernel emission (see [`composed_kind`])
/// collapse the anchor + absorbed ops into ONE `FusedKernelBackend` composed
/// call; the absorbed ops and their eliminated intermediate tensors vanish
/// from the emitted code.  Everything else — including every T2 group
/// (`requires_verification == true`) — emits exactly as the unfused
/// [`emit_model`], so a model with zero composed groups emits byte-identical
/// code through both entry points.
pub(crate) fn emit_model_fused(
    model: &ParsedModel,
    schedule: &FusedSchedule,
) -> Result<TokenStream, String> {
    emit_model_with(model, Some(schedule))
}

/// T1.3 test arm — fused emission with the liveness arena DISABLED
/// (per-tensor stack arrays, exactly the pre-T1.3 layout): the `stack` arm
/// of the arena-vs-stack bit-exactness gate (`#[model_stack]`).
pub(crate) fn emit_model_stack_fused(
    model: &ParsedModel,
    schedule: &FusedSchedule,
) -> Result<TokenStream, String> {
    emit_model_with_options(model, Some(schedule), false)
}

/// Shared emission core: `schedule: None` is the unfused path; `Some` routes
/// composed groups through the `fused_*` backend calls (T1.2).
fn emit_model_with(
    model: &ParsedModel,
    schedule: Option<&FusedSchedule>,
) -> Result<TokenStream, String> {
    emit_model_with_options(model, schedule, true)
}

/// `arena_enabled: false` forces per-tensor stack emission even when the
/// planner succeeds — the `#[model_stack]` test arm.
fn emit_model_with_options(
    model: &ParsedModel,
    schedule: Option<&FusedSchedule>,
    arena_enabled: bool,
) -> Result<TokenStream, String> {
    if model.subgraph_count() != 1 {
        return Err(format!(
            "model has {} subgraphs; only single-subgraph models are supported",
            model.subgraph_count()
        ));
    }
    let tensors = model.tensors();
    if tensors.is_empty() {
        return Err("model subgraph[0] has no tensors".into());
    }
    let ops = model.ops();
    let plan = match schedule {
        Some(s) => Some(fused_plan(s, ops.len(), tensors.len())),
        None => None,
    };
    let inputs = model.inputs();
    let outputs = model.outputs();
    if inputs.is_empty() {
        return Err("model subgraph[0] has no input tensors".into());
    }
    if outputs.is_empty() {
        return Err("model subgraph[0] has no output tensors".into());
    }

    // ── Storage classification for every tensor index ──────────────────────
    // Model inputs occupy a contiguous region of the caller's `input` array;
    // model outputs a contiguous region of `output`.  Buffered tensors
    // (weights/biases) become emitted `const` arrays.  Everything else is an
    // intermediate stack array.
    let mut storage: Vec<Storage> = vec![Storage::Const; tensors.len()];
    let mut input_len = 0usize;
    for &t in inputs {
        let tensor = tensor_at(tensors, t)?;
        check_int8(tensor)?;
        let len = flat_len(&tensor.shape)?;
        storage[t as usize] = Storage::Input { start: input_len, len };
        input_len += len;
    }
    let mut output_len = 0usize;
    for &t in outputs {
        let tensor = tensor_at(tensors, t)?;
        check_int8(tensor)?;
        let len = flat_len(&tensor.shape)?;
        storage[t as usize] = Storage::Output { start: output_len, len };
        output_len += len;
    }
    for (t, tensor) in tensors.iter().enumerate() {
        if let Storage::Const = storage[t] {
            if model.buffer_data(tensor).is_none() {
                storage[t] = Storage::Tensor { idx: t };
            }
        }
    }

    // ── T1.3: liveness arena over the intermediates ────────────────────────
    // One `#[repr(C, align(16))] struct Arena { data: [i8; ARENA_LEN] }`
    // local replaces the per-tensor stack arrays whenever the planner
    // succeeds AND covers every emitted intermediate.  On planner rejection
    // (mobilenet_v2: a single 224²×32 activation exceeds MAX_INTERNAL) the
    // emitter falls back to per-tensor stack locals — bit-exact, just
    // larger (see local-notes/evidence/composed-kernels/t13-arena.md).
    let arena_plan = if arena_enabled { arena_plan_for(model, &storage) } else { None };
    let lens: Option<Vec<usize>> = match &arena_plan {
        Some(_) => Some(
            tensors
                .iter()
                .map(|t| flat_len(&t.shape))
                .collect::<Result<Vec<usize>, String>>()?,
        ),
        None => None,
    };
    let arena_peak = arena_plan.as_ref().map(|p| p.peak_arena_bytes).unwrap_or(0);

    // ── Emit per-op consts + calls (straight-line, execution order) ────────
    // With a schedule, each op is either (a) an anchor of a composed group
    // → one `fused_*` call, (b) absorbed into a composed group → skipped, or
    // (c) an ordinary op → its per-op call exactly as the unfused emitter.
    let mut consts: Vec<TokenStream> = Vec::new();
    let mut calls: Vec<TokenStream> = Vec::new();
    let mut scratch_max = 0usize;
    for (i, op) in ops.iter().enumerate() {
        if plan.as_ref().is_some_and(|p| p.absorbed[i]) {
            continue;
        }
        let mut actx = ArenaCtx::new(arena_plan.as_ref(), lens.as_deref().unwrap_or(&[]));
        actx.op = i;
        let em = match plan.as_ref().and_then(|p| p.anchor_group[i]) {
            Some(gi) => {
                let group = &schedule.expect("anchor_group implies a schedule").groups[gi];
                match composed_kind(group) {
                    Some(ComposedKind::Conv) => {
                        emit_fused_conv(model, &storage, &mut actx, i, op, group)?
                    }
                    Some(ComposedKind::Chain) => {
                        emit_fused_chain(model, &storage, &mut actx, i, op, group)?
                    }
                    Some(ComposedKind::PoolFold) => {
                        emit_fused_pool_fold(model, &storage, &mut actx, i, op, group)?
                    }
                    None => emit_op(model, &storage, &mut actx, i, op)?,
                }
            }
            None => emit_op(model, &storage, &mut actx, i, op)?,
        };
        consts.extend(em.consts);
        calls.push(actx.wrap(em.call));
        scratch_max = scratch_max.max(em.scratch);
    }

    // ── Intermediate storage: arena local OR per-tensor stack arrays ───────
    // Arena mode: every intermediate lives in `Arena.data` at a const
    // offset (the planner covers all of them — `arena_plan_for` checked).
    // Fallback mode: per-tensor `#[repr(C, align(16))]` stack locals, as
    // before T1.3.  Eliminated tensors (absorbed into composed groups) get
    // no storage in either mode.
    let mut tensor_types: Vec<TokenStream> = Vec::new();
    let mut tensor_locals: Vec<TokenStream> = Vec::new();
    let arena_ty: Option<TokenStream> = match &arena_plan {
        Some(p) if p.peak_arena_bytes > 0 => {
            let peak = p.peak_arena_bytes;
            Some(quote! {
                /// Intermediate-tensor arena (T1.3): ONE 16-byte-aligned
                /// array sized to the liveness peak; every intermediate
                /// indexes it at a compile-time offset.
                #[repr(C, align(16))]
                struct Arena {
                    data: [i8; #peak],
                }
            })
        }
        _ => None,
    };
    let arena_local: Option<TokenStream> = match &arena_plan {
        Some(p) if p.peak_arena_bytes > 0 => {
            let peak = p.peak_arena_bytes;
            Some(quote! {
                let mut arena = Arena { data: [0i8; #peak] };
            })
        }
        _ => None,
    };
    if arena_plan.is_none() {
        for s in &storage {
            if let Storage::Tensor { idx } = s {
                if plan.as_ref().is_some_and(|p| p.eliminated[*idx]) {
                    continue;
                }
                let len = flat_len(&tensors[*idx].shape)?;
                let ty = Ident::new(&format!("TENSOR_{idx}"), proc_macro2::Span::call_site());
                let var = Ident::new(&format!("tensor_{idx}"), proc_macro2::Span::call_site());
                tensor_types.push(quote! {
                    #[repr(C, align(16))]
                    struct #ty {
                        data: [i8; #len],
                    }
                });
                tensor_locals.push(quote! {
                    let mut #var = #ty { data: [0i8; #len] };
                });
            }
        }
    }

    let input_len_ts = input_len;
    let output_len_ts = output_len;
    let scratch_ts = scratch_max;
    // Composed `fused_*` calls need `&mut self` (the `FusedKernelBackend`
    // receiver); models without composed groups keep the `&self` receiver so
    // existing call sites (non-`mut` bindings) compile unchanged.
    let composed = plan.as_ref().is_some_and(|p| p.anchor_group.iter().any(|g| g.is_some()));
    let receiver = if composed { quote!(&mut self) } else { quote!(&self) };
    let backend_bind = if calls.is_empty() {
        quote! { let _ = &self.backend; }
    } else if composed {
        quote! { let backend = &mut self.backend; }
    } else {
        quote! { let backend = &self.backend; }
    };

    let wrapper = quote! {
        /// Generated inference model — typed I/O bridge for `subgraph[0]`.
        ///
        /// `B` is any [`::hematite_core::FusedKernelBackend`]
        /// implementation (which extends [`::hematite_core::KernelBackend`]);
        /// the straight-line op sequence dispatches through it, with fused
        /// groups going through the composed `fused_*` entry points.
        pub struct Model<B> {
            backend: B,
        }

        impl<B> Model<B> {
            /// Construct the model around a backend.
            pub const fn new(backend: B) -> Self {
                Model { backend }
            }

            /// Number of input bytes (i8 elements) the model consumes.
            pub const fn input_len() -> usize {
                INPUT_LEN
            }

            /// Number of output bytes (i8 elements) the model produces.
            pub const fn output_len() -> usize {
                OUTPUT_LEN
            }
        }

        impl<B: ::hematite_core::FusedKernelBackend> Model<B> {
            /// Run inference with an internally allocated scratch buffer.
            ///
            /// No panic paths: on error the output array is left zeroed.
            pub fn predict(#receiver, input: &[i8; INPUT_LEN]) -> [i8; OUTPUT_LEN] {
                let mut output = [0i8; OUTPUT_LEN];
                let mut scratch = [0u8; SCRATCH_LEN];
                let _ = self.predict_with_scratch(input, &mut output, &mut scratch);
                output
            }

            /// Run inference with caller-provided scratch.
            ///
            /// # Errors
            ///
            /// * [`::hematite_core::KernelError::ScratchTooSmall`] if
            ///   `scratch.len()` is below [`SCRATCH_LEN`] — the macro-time
            ///   max over ops of their documented scratch need.
            ///
            /// The intermediate-tensor arena is ALWAYS the internal
            /// [`ARENA_LEN`]-byte stack local: the caller's `scratch` is
            /// `&mut [u8]` and the arena is `[i8; ARENA_LEN]`, and reusing
            /// it would require a byte-type cast that generated code must
            /// never contain (T1.3 evidence).  `scratch` covers only the
            /// per-op kernel scratch.
            pub fn predict_with_scratch(
                #receiver,
                input: &[i8; INPUT_LEN],
                output: &mut [i8; OUTPUT_LEN],
                scratch: &mut [u8],
            ) -> Result<(), ::hematite_core::KernelError> {
                if scratch.len() < SCRATCH_LEN {
                    return Err(::hematite_core::KernelError::ScratchTooSmall);
                }
                #arena_local
                #(#tensor_locals)*
                #backend_bind
                #(#calls)*
                Ok(())
            }
        }
    };

    Ok(quote! {
        pub const INPUT_LEN: usize = #input_len_ts;
        pub const OUTPUT_LEN: usize = #output_len_ts;
        pub const SCRATCH_LEN: usize = #scratch_ts;
        /// Intermediate-tensor arena size in bytes (liveness peak, T1.3);
        /// 0 when the model fell back to per-tensor stack emission.
        pub const ARENA_LEN: usize = #arena_peak;
        #(#consts)*
        #arena_ty
        #(#tensor_types)*
        #wrapper
    })
}

// ---------------------------------------------------------------------------
// Fused-schedule wiring (T1.2)
// ---------------------------------------------------------------------------

/// Which composed `FusedKernelBackend` call replaces the anchor's per-op call.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ComposedKind {
    /// CONV_2D anchor with an absorbed residual-ADD and/or trailing
    /// activation (fusion patterns (c) / (a)) → `fused_conv2d`.
    Conv,
    /// ADD/MUL/SUB anchor with an absorbed elementwise chain (pattern (b))
    /// → `fused_elementwise_chain`.
    Chain,
    /// Pool anchor with an absorbed MUL/SUB input fold (pattern (d))
    /// → `fused_pool_with_fold`.
    PoolFold,
}

/// Classify a fused group for the T1.2 emitter.
///
/// Only **T1 groups** (`requires_verification == false`) are composed — T2
/// groups (input folds, requantize folds) are algebraically transformative
/// and stay per-op until a fused==unfused verification passes.  Only
/// CONV_2D anchors compose as convs: [`FusedConvParams`] carries
/// `Conv2DParams`, so a DEPTHWISE/FULLY_CONNECTED anchor with absorbed ops
/// falls back to per-op emission (bit-exact, unchanged).  Softmax anchors
/// never compose (their fold has no `PoolParams` representation).
fn composed_kind(group: &FusedGroup) -> Option<ComposedKind> {
    if group.requires_verification {
        return None;
    }
    match group.anchor_builtin {
        3 if group.residual_add.is_some() || !group.absorbed_ops.is_empty() => {
            Some(ComposedKind::Conv)
        }
        0 | 18 | 41 if !group.elementwise_chain.is_empty() => Some(ComposedKind::Chain),
        1 | 17 if group.input_fold.is_some() => Some(ComposedKind::PoolFold),
        _ => None,
    }
}

/// Per-op / per-tensor decisions derived from the schedule once per model.
struct FusedPlan {
    /// Op index → group index in `FusedSchedule::groups` (composed groups
    /// only; `None` for ordinary ops and T2 groups).
    anchor_group: Vec<Option<usize>>,
    /// Op indices absorbed into composed groups (skip their own emission).
    absorbed: Vec<bool>,
    /// Tensor indices eliminated by composed groups (no stack array).
    eliminated: Vec<bool>,
}

fn fused_plan(schedule: &FusedSchedule, op_count: usize, tensor_count: usize) -> FusedPlan {
    let mut anchor_group = vec![None; op_count];
    let mut absorbed = vec![false; op_count];
    let mut eliminated = vec![false; tensor_count];
    for (gi, group) in schedule.groups.iter().enumerate() {
        if composed_kind(group).is_none() {
            continue;
        }
        anchor_group[group.anchor_op_index] = Some(gi);
        for &a in &group.absorbed_ops {
            absorbed[a] = true;
        }
        for &t in &group.eliminated_tensors {
            if (t as usize) < tensor_count {
                eliminated[t as usize] = true;
            }
        }
    }
    FusedPlan { anchor_group, absorbed, eliminated }
}

// ---------------------------------------------------------------------------
// T1.3 — liveness-arena emission
// ---------------------------------------------------------------------------

/// Compute the T1.3 arena plan for a model, or fall back to per-tensor
/// stack emission when the planner rejects it.
///
/// Fallback triggers: planner error (mobilenet_v2's single 224²×32
/// activation exceeds `MAX_INTERNAL`) or incomplete coverage (any
/// intermediate the emitter will materialize has no arena slot — the
/// planner keeps zero-size / never-written tensors out).  The fallback
/// emits exactly the pre-T1.3 per-tensor code, bit-exact.
fn arena_plan_for(model: &ParsedModel<'_>, storage: &[Storage]) -> Option<ArenaPlan> {
    let plan = plan_arena(model).ok()?;
    if plan.peak_arena_bytes == 0 {
        return None;
    }
    for (t, s) in storage.iter().enumerate() {
        if matches!(s, Storage::Tensor { .. }) && plan.offsets[t] == OFFSET_NONE {
            return None;
        }
    }
    Some(plan)
}

/// Per-call arena slice bookkeeping (T1.3).
///
/// The arena is ONE `[i8; ARENA_LEN]` local; each op call borrows the
/// disjoint slices it touches through nested `split_at_mut` (safe, const
/// offsets).  The planner guarantees a single op's slices never overlap:
/// its inputs and output are simultaneously live at that op.  [`wrap`]
/// prefixes the call statement with the borrows.
struct ArenaCtx<'a> {
    /// Arena emission active (plan succeeded and covers all intermediates).
    on: bool,
    /// Tensor idx → arena byte offset.
    offsets: &'a [usize],
    /// Tensor flat lens (byte counts), index-aligned with `offsets`.
    lens: &'a [usize],
    /// (slice ident, offset, len) registered for the current call.
    regions: Vec<(Ident, usize, usize)>,
    /// Op index — names the slice idents uniquely.
    op: usize,
}

impl<'a> ArenaCtx<'a> {
    fn new(plan: Option<&'a ArenaPlan>, lens: &'a [usize]) -> Self {
        let (on, offsets) = match plan {
            Some(p) => (true, &p.offsets[..]),
            None => (false, &[][..]),
        };
        Self { on, offsets, lens, regions: Vec::new(), op: 0 }
    }

    /// Arena off: every slice lookup returns `None`, so callers emit the
    /// per-tensor `tensor_N.data` expressions (fallback mode).
    fn inactive() -> Self {
        Self { on: false, offsets: &[], lens: &[], regions: Vec::new(), op: 0 }
    }

    /// Register the arena slice for tensor `t`, deduped by (offset, len) so
    /// repeated references to one tensor share a borrow.  `None` when the
    /// arena is off or the tensor is not arena-backed (caller falls back to
    /// the per-tensor local).
    fn slice(&mut self, t: usize) -> Option<(Ident, usize, usize)> {
        if !self.on {
            return None;
        }
        let off = *self.offsets.get(t)?;
        if off == OFFSET_NONE {
            return None;
        }
        let len = *self.lens.get(t)?;
        if let Some(existing) = self.regions.iter().find(|(_, o, l)| *o == off && *l == len) {
            return Some(existing.clone());
        }
        let ident = Ident::new(
            &format!("arena_slice_{}_{}", self.op, self.regions.len()),
            proc_macro2::Span::call_site(),
        );
        let entry = (ident, off, len);
        self.regions.push(entry.clone());
        Some(entry)
    }

    /// Prefix a call with the nested `split_at_mut` borrows its slices need.
    ///
    /// Regions are sorted by offset; each is carved as `(gap, slice)` from
    /// the remaining tail, so the slice idents are disjoint borrows of
    /// `arena.data`.  All split indices are compile-time consts within
    /// `[0, ARENA_LEN)` — no panic paths.
    fn wrap(&self, call: TokenStream) -> TokenStream {
        if !self.on || self.regions.is_empty() {
            return call;
        }
        let mut regions = self.regions.clone();
        regions.sort_by_key(|(_, off, _)| *off);
        let last = regions.len() - 1;
        let mut lets: Vec<TokenStream> = Vec::with_capacity(regions.len() * 2);
        let mut prev_end = 0usize;
        for (k, (ident, off, len)) in regions.iter().enumerate() {
            let gap = if k == 0 { *off } else { off - prev_end };
            if k == 0 {
                lets.push(quote!(let (_, rest) = arena.data.split_at_mut(#gap);));
            } else {
                lets.push(quote!(let (_, rest) = rest.split_at_mut(#gap);));
            }
            if k == last {
                lets.push(quote!(let (#ident, _) = rest.split_at_mut(#len);));
            } else {
                lets.push(quote!(let (#ident, rest) = rest.split_at_mut(#len);));
            }
            prev_end = off + len;
        }
        quote!({ #(#lets)* #call })
    }
}

/// `&tensor_N.data` for an intermediate tensor — the composed params'
/// runtime operand slices (the residual, chain operands) must point at the
/// generated storage.  Fusion guarantees the residual is a computed
/// intermediate (never a const / model I/O).
fn tensor_ref_expr(
    ctx: &mut ArenaCtx,
    storage: &[Storage],
    t: usize,
) -> Result<TokenStream, String> {
    match storage.get(t).ok_or_else(|| format!("tensor index {t} out of range"))? {
        Storage::Tensor { idx } => match ctx.slice(*idx) {
            Some((ident, _, _)) => Ok(quote!(&#ident[..])),
            None => {
                let var = Ident::new(&format!("tensor_{idx}"), proc_macro2::Span::call_site());
                Ok(quote!(&#var.data))
            }
        },
        _ => Err(format!(
            "tensor {t} must be an intermediate stack array for composed params"
        )),
    }
}

/// Runtime operand slice for a composed param struct: an emitted const for
/// buffer-backed tensors, the arena slice / `&tensor_N.data` for
/// intermediates, `&input[..]` for model inputs.  (The T4.1 per-op emitter
/// rejects constant operands — "deferred to T4.2a fusion" — the composed
/// paths are where they land.)
fn operand_data(
    model: &ParsedModel,
    ctx: &mut ArenaCtx,
    storage: &[Storage],
    t: u32,
    name: &Ident,
) -> Result<(TokenStream, Vec<TokenStream>), String> {
    match storage
        .get(t as usize)
        .ok_or_else(|| format!("tensor index {t} out of range"))?
    {
        Storage::Const => {
            let c = weight_const(model, t, name)?;
            Ok((quote!(&#name.0), vec![c]))
        }
        Storage::Tensor { idx } => match ctx.slice(*idx) {
            Some((ident, _, _)) => Ok((quote!(&#ident[..]), Vec::new())),
            None => {
                let var = Ident::new(&format!("tensor_{idx}"), proc_macro2::Span::call_site());
                Ok((quote!(&#var.data), Vec::new()))
            }
        },
        Storage::Input { start, len } => {
            let end = start + len;
            Ok((quote!(&input[#start..#end]), Vec::new()))
        }
        Storage::Output { .. } => Err(format!(
            "tensor {t} is a model output used as a composed operand"
        )),
    }
}

/// `FusedActivationKind` → `ComposedActivation` token (the epilogue enum).
fn composed_activation_enum(kind: FusedActivationKind) -> TokenStream {
    match kind {
        FusedActivationKind::Relu => {
            quote!(::hematite_core::op_params::ComposedActivation::Relu)
        }
        FusedActivationKind::Relu6 => {
            quote!(::hematite_core::op_params::ComposedActivation::Relu6)
        }
        FusedActivationKind::HardSwish => {
            quote!(::hematite_core::op_params::ComposedActivation::HardSwish)
        }
    }
}

/// Fusion `ElementwiseKind` → `op_params::ElementwiseKind` token.
fn chain_step_kind_enum(kind: FusedStepKind) -> TokenStream {
    match kind {
        FusedStepKind::Add => quote!(::hematite_core::op_params::ElementwiseKind::Add),
        FusedStepKind::Mul => quote!(::hematite_core::op_params::ElementwiseKind::Mul),
        FusedStepKind::Sub => quote!(::hematite_core::op_params::ElementwiseKind::Sub),
        FusedStepKind::Relu => quote!(::hematite_core::op_params::ElementwiseKind::Relu),
        FusedStepKind::Relu6 => quote!(::hematite_core::op_params::ElementwiseKind::Relu6),
        FusedStepKind::HardSwish => {
            quote!(::hematite_core::op_params::ElementwiseKind::HardSwish)
        }
    }
}

// ---------------------------------------------------------------------------
// Storage classification
// ---------------------------------------------------------------------------

/// Where a tensor's data lives in the generated code.
#[derive(Clone)]
enum Storage {
    /// A slice of the caller's `input: &[i8; INPUT_LEN]` array.
    Input { start: usize, len: usize },
    /// A slice of the caller's `output: &mut [i8; OUTPUT_LEN]` array.
    Output { start: usize, len: usize },
    /// An emitted `const` array (weights/biases — source only).
    Const,
    /// A `#[repr(C, align(16))]` stack array (intermediate tensor).
    Tensor { idx: usize },
}

/// Expression naming a tensor as a kernel *input* (immutable `&[i8]`).
fn src_expr(
    ctx: &mut ArenaCtx,
    storage: &[Storage],
    t: usize,
) -> Result<TokenStream, String> {
    match storage.get(t).ok_or_else(|| format!("tensor index {t} out of range"))? {
        Storage::Input { start, len } => {
            let end = start + len;
            Ok(quote!(&input[#start..#end]))
        }
        Storage::Tensor { idx } => match ctx.slice(*idx) {
            Some((ident, _, _)) => Ok(quote!(&#ident[..])),
            None => {
                let var = Ident::new(&format!("tensor_{idx}"), proc_macro2::Span::call_site());
                Ok(quote!(&#var.data))
            }
        },
        Storage::Const => Err(format!(
            "tensor {t} is a constant (buffer-backed) but is used as a runtime op input; \
             constant data inputs are not supported in T4.1 (deferred to T4.2a fusion)"
        )),
        Storage::Output { .. } => Err(format!("tensor {t} is a model output used as an op input")),
    }
}

/// Expression naming a tensor as a kernel *output* (`&mut [i8]`).
fn dst_expr(
    ctx: &mut ArenaCtx,
    storage: &[Storage],
    t: usize,
) -> Result<TokenStream, String> {
    match storage.get(t).ok_or_else(|| format!("tensor index {t} out of range"))? {
        Storage::Output { start, len } => {
            let end = start + len;
            Ok(quote!(&mut output[#start..#end]))
        }
        Storage::Tensor { idx } => match ctx.slice(*idx) {
            Some((ident, _, _)) => Ok(quote!(&mut #ident[..])),
            None => {
                let var = Ident::new(&format!("tensor_{idx}"), proc_macro2::Span::call_site());
                Ok(quote!(&mut #var.data))
            }
        },
        _ => Err(format!("tensor {t} cannot be an op output (model input or const)")),
    }
}

// ---------------------------------------------------------------------------
// Host-side quantization math (TFLM `QuantizeMultiplier`, f64 — macro-time)
// ---------------------------------------------------------------------------

/// Convert a float scale into a TFLM quantized-multiplier + shift pair.
///
/// Semantics copied from `hematite-int8` (`#[cfg(feature = "host")]`
/// `quantize_multiplier`), which mirrors
/// `tflite::QuantizeMultiplier` in `quantization_util.cc`: frexp the scale,
/// round the significand to Q0.31, carry on overflow, flush tiny scales.
fn quantize_multiplier(scale: f64) -> (i32, i32) {
    if scale == 0.0 {
        return (0, 0);
    }
    let (sig, mut shift) = frexp(scale);
    let mut q_fixed = (sig * (1u64 << 31) as f64 + 0.5) as i64;
    if q_fixed == (1i64 << 31) {
        q_fixed /= 2;
        shift += 1;
    }
    if shift < -31 {
        return (0, 0);
    }
    (q_fixed as i32, shift)
}

/// `std::frexp` semantics via IEEE 754 bit manipulation (`no_std`-safe).
fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 {
        return (0.0, 0);
    }
    let bits = x.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let mantissa = bits & 0x000f_ffff_ffff_ffff;
    let sign = bits & 0x8000_0000_0000_0000;
    let frexp_exponent = exponent - 1022;
    let frexp_significand_bits = sign | 0x3fe0_0000_0000_0000u64 | mantissa;
    (f64::from_bits(frexp_significand_bits), frexp_exponent)
}

// ---------------------------------------------------------------------------
// Shape / quantization helpers
// ---------------------------------------------------------------------------

fn tensor_at<'a>(tensors: &'a [ParsedTensor<'a>], t: u32) -> Result<&'a ParsedTensor<'a>, String> {
    tensors
        .get(t as usize)
        .ok_or_else(|| format!("tensor index {t} out of range"))
}

/// Flat element count of a tensor shape (rejects dynamic/zero dims).
fn flat_len(shape: &[i32]) -> Result<usize, String> {
    let mut n: i64 = 1;
    for &d in shape {
        if d <= 0 {
            return Err(format!(
                "dynamic or non-positive shape dimension {d} is not supported (T4.1 static shapes)"
            ));
        }
        n *= i64::from(d);
    }
    Ok(n as usize)
}

/// Pad a tensor shape to NHWC rank 4 (left-pad with 1s for rank < 4).
fn shape4(shape: &[i32]) -> Result<[i32; 4], String> {
    if shape.len() > 4 {
        return Err(format!(
            "tensor rank {} exceeds the T4.1 static-shape limit of 4",
            shape.len()
        ));
    }
    let mut out = [1i32; 4];
    let base = 4 - shape.len();
    for (i, &d) in shape.iter().enumerate() {
        if d <= 0 {
            return Err(format!("dynamic or non-positive shape dimension {d} is not supported"));
        }
        out[base + i] = d;
    }
    Ok(out)
}

/// Tensor scale as `f64`, validating positivity/finiteness.
fn tensor_scale(t: &ParsedTensor) -> Result<f64, String> {
    match &t.quant {
        Some(q) if q.scale.is_finite() && q.scale > 0.0 => Ok(f64::from(q.scale)),
        Some(q) => Err(format!("tensor {} has invalid scale {}", t.name, q.scale)),
        None => Err(format!("tensor {} has no quantization", t.name)),
    }
}

/// Tensor zero-point as `i32`.
fn tensor_zp(t: &ParsedTensor) -> Result<i32, String> {
    match &t.quant {
        Some(q) => Ok(q.zero_point as i32),
        None => Err(format!("tensor {} has no quantization", t.name)),
    }
}

/// Per-channel scale vector of length `n` (broadcasts per-tensor quant).
fn channel_scales(quant: Option<&QuantInfo>, n: usize) -> Result<Vec<f64>, String> {
    match quant {
        None => Err("tensor has no quantization".into()),
        Some(q) => {
            if let Some(pc) = &q.per_channel {
                if pc.scales.len() == n {
                    Ok(pc.scales.iter().map(|&s| f64::from(s)).collect())
                } else if pc.scales.len() == 1 {
                    Ok(vec![f64::from(pc.scales[0]); n])
                } else {
                    Err(format!(
                        "per-channel scales length {} != expected output channels {n}",
                        pc.scales.len()
                    ))
                }
            } else {
                Ok(vec![f64::from(q.scale); n])
            }
        }
    }
}

/// TFLM `CalculateActivationRangeQuantized` against the output tensor.
///
/// Fused codes: 0=NONE, 1=RELU, 3=RELU6.  2 (RELU_N1_TO_1), 4 (TANH),
/// 5 (SIGN_BIT) are treated as NONE per the T4.1 dispatch scope.
fn act_range(act: i8, out_scale: f64, out_zp: i32) -> (i32, i32) {
    const QMIN: i32 = -128;
    const QMAX: i32 = 127;
    if out_scale <= 0.0 {
        return (QMIN, QMAX);
    }
    match act {
        1 => (out_zp.max(QMIN), QMAX),
        3 => (
            out_zp.max(QMIN),
            (out_zp + (6.0 / out_scale).round() as i32).min(QMAX),
        ),
        _ => (QMIN, QMAX),
    }
}

fn padding_enum(padding: i8) -> Result<TokenStream, String> {
    match padding {
        0 => Ok(quote!(::hematite_core::op_params::Padding::Same)),
        1 => Ok(quote!(::hematite_core::op_params::Padding::Valid)),
        p => Err(format!("unknown padding value {p} (expected 0=SAME, 1=VALID)")),
    }
}

fn fused_activation_enum(act: i8) -> TokenStream {
    match act {
        1 => quote!(::hematite_core::op_params::FusedActivation::Relu),
        3 => quote!(::hematite_core::op_params::FusedActivation::Relu6),
        2 => quote!(::hematite_core::op_params::FusedActivation::Relu1),
        _ => quote!(::hematite_core::op_params::FusedActivation::None),
    }
}

fn check_int8(t: &ParsedTensor) -> Result<(), String> {
    if t.tensor_type != TensorType::Int8 {
        return Err(format!(
            "model I/O tensor '{}' is {:?}, not INT8 (T4.1 typed I/O bridge requires int8)",
            t.name, t.tensor_type
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Const-array builders
// ---------------------------------------------------------------------------

fn i8_lit(bytes: &[u8]) -> TokenStream {
    let vals: Vec<TokenStream> = bytes
        .iter()
        .map(|&b| {
            let v = b as i8;
            quote!(#v)
        })
        .collect();
    quote!([ #(#vals),* ])
}

fn i32_le_lit(bytes: &[u8]) -> Result<TokenStream, String> {
    if !bytes.len().is_multiple_of(4) {
        return Err(format!("i32 buffer length {} not divisible by 4", bytes.len()));
    }
    let vals: Vec<TokenStream> = bytes
        .chunks(4)
        .map(|c| {
            let v = i32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            quote!(#v)
        })
        .collect();
    Ok(quote!([ #(#vals),* ]))
}

/// Emit an int8 weight/alpha const guaranteed 16-byte aligned.
///
/// The ACCX/FC SIMD dispatch gates on `(w_ptr as usize) % 16 == 0` — a plain
/// `const W: [i8; N]` has alignment 1, so zoo-model weights (emitted consts)
/// silently fell back to scalar while synthetic bench weights (carved into the
/// 16-aligned arena) engaged SIMD. Wrapping in a `#[repr(C, align(16))]`
/// struct forces the const data's alignment; call sites reference `&W.0`.
fn const_i8(name: &Ident, bytes: &[u8], len: usize) -> TokenStream {
    let vals = i8_lit(bytes);
    let ty = Ident::new(&format!("{name}Ty"), proc_macro2::Span::call_site());
    quote! {
        #[repr(C, align(16))]
        struct #ty([i8; #len]);
        const #name: #ty = #ty(#vals);
    }
}

fn const_i32(name: &Ident, values: &[i32]) -> TokenStream {
    let vals: Vec<TokenStream> = values.iter().map(|&v| quote!(#v)).collect();
    let n = values.len();
    quote!(const #name: [i32; #n] = [ #(#vals),* ];)
}

/// Token stream for a `[i32; 4]` array literal (arrays do not impl
/// `ToTokens` in quote).
fn arr4(v: [i32; 4]) -> TokenStream {
    let [a, b, c, d] = v;
    quote!([#a, #b, #c, #d])
}

/// Token stream for a `[i32; 8]` array literal.
fn arr8(v: [i32; 8]) -> TokenStream {
    let [a, b, c, d, e, f, g, h] = v;
    quote!([#a, #b, #c, #d, #e, #f, #g, #h])
}

/// Weight const for op `i`: the int8 filter buffer (raw bytes → i8).
fn weight_const(model: &ParsedModel, tensor_idx: u32, name: &Ident) -> Result<TokenStream, String> {
    let tensor = model
        .tensor_by_index(tensor_idx as usize)
        .ok_or_else(|| format!("weight tensor index {tensor_idx} out of range"))?;
    let bytes = model.buffer_data(tensor).ok_or_else(|| {
        format!(
            "weight tensor '{}' has no buffer data; constant weights are required in T4.1",
            tensor.name
        )
    })?;
    let len = flat_len(&tensor.shape)?;
    if bytes.len() != len {
        return Err(format!(
            "weight tensor '{}' buffer length {} != shape size {len}",
            tensor.name,
            bytes.len()
        ));
    }
    Ok(const_i8(name, bytes, len))
}

/// Bias const for op `i`: int32 LE buffer, zero-padded when absent/optional.
fn bias_const(
    model: &ParsedModel,
    tensor_idx: u32,
    expected_len: usize,
    name: &Ident,
) -> Result<TokenStream, String> {
    if tensor_idx == u32::MAX {
        return Ok(quote!(const #name: [i32; #expected_len] = [0; #expected_len];));
    }
    let tensor = model
        .tensor_by_index(tensor_idx as usize)
        .ok_or_else(|| format!("bias tensor index {tensor_idx} out of range"))?;
    match model.buffer_data(tensor) {
        Some(bytes) => {
            if bytes.len() != expected_len * 4 {
                return Err(format!(
                    "bias tensor '{}' buffer length {} != expected {}",
                    tensor.name,
                    bytes.len(),
                    expected_len * 4
                ));
            }
            let vals = i32_le_lit(bytes)?;
            Ok(quote!(const #name: [i32; #expected_len] = #vals;))
        }
        None => Ok(quote!(const #name: [i32; #expected_len] = [0; #expected_len];)),
    }
}

// ---------------------------------------------------------------------------
// Per-op emitters
// ---------------------------------------------------------------------------

struct OpEmission {
    consts: Vec<TokenStream>,
    call: TokenStream,
    /// Macro-time scratch bytes this op needs (mirrors the S3 backend's
    /// `*_scratch_size` formulas; 0 when the op never touches scratch).
    scratch: usize,
}

/// Quantization context shared by the conv-family (conv/depthwise/FC).
struct ConvQuant {
    input_offset: i32,
    weights_offset: i32,
    output_offset: i32,
    multipliers: Vec<i32>,
    shifts: Vec<i32>,
    act_min: i32,
    act_max: i32,
}

fn conv_quant(
    input: &ParsedTensor,
    weights: &ParsedTensor,
    output: &ParsedTensor,
    out_channels: usize,
    fused_activation: i8,
) -> Result<ConvQuant, String> {
    let in_scale = tensor_scale(input)?;
    let out_scale = tensor_scale(output)?;
    let out_zp = tensor_zp(output)?;
    let w_scales = channel_scales(weights.quant.as_ref(), out_channels)?;
    let mut multipliers = Vec::with_capacity(out_channels);
    let mut shifts = Vec::with_capacity(out_channels);
    for w in &w_scales {
        let real = in_scale * w / out_scale;
        let (m, s) = quantize_multiplier(real);
        multipliers.push(m);
        shifts.push(s);
    }
    let (act_min, act_max) = act_range(fused_activation, out_scale, out_zp);
    Ok(ConvQuant {
        input_offset: -tensor_zp(input)?,
        weights_offset: tensor_zp(weights)?,
        output_offset: out_zp,
        multipliers,
        shifts,
        act_min,
        act_max,
    })
}

// ── Macro-time scratch-need computation ────────────────────────────────────
//
// The `KernelBackend` trait's `*_scratch_size` associated functions cannot be
// called from this proc-macro (it runs at host compile time against a generic
// `B`). Instead we recompute the S3 backend's documented scratch formulas
// (hematite-s3/src/backend.rs, `conv1x1_scratch_need`,
// `conv3x3_scratch_need`, `depthwise_scratch_need`, `softmax_scratch_size`)
// directly from the parsed op params, which the macro always has. Backends
// that ignore scratch (e.g. `RefBackend`) are unaffected — a larger
// `SCRATCH_LEN` only sizes a stack array they never read. Keep these formulas
// in sync with backend.rs.

/// Round a channel count up to the TIE728 SIMD group width (16 lanes).
const fn pad16(c: usize) -> usize {
    (c + 15) & !15
}

/// Scratch bytes for a 1×1 conv (`conv1x1_scratch_need`).
fn conv1x1_scratch_need_codegen(out_c: usize, input_offset: i32) -> usize {
    let wsum = if input_offset != 0 { out_c * 4 } else { 0 };
    out_c * 4 + wsum
}

/// Scratch bytes for the FC/GEMM path (`fc_scratch_need` in backend.rs).
///
/// T3.6 — small / non-16 input dims: when `input_dim` is not a multiple of
/// 16 the FC dispatch stages a zero-padded input copy (`pad16(input_dim)`
/// bytes) AND a zero-padded weight copy (`output_dim * pad16(input_dim)`
/// bytes) in scratch, plus the i32 accumulator buffer and the optional
/// weight-sum buffer. Keep in sync with `hematite-s3/src/backend.rs`.
fn fc_scratch_need_codegen(input_dim: usize, out_dim: usize, input_offset: i32) -> usize {
    let padded_dim = pad16(input_dim);
    let wsum = if input_offset != 0 { out_dim * 4 } else { 0 };
    if padded_dim != input_dim {
        padded_dim + out_dim * padded_dim + out_dim * 4 + wsum
    } else {
        out_dim * 4 + wsum
    }
}

/// Scratch bytes for the general conv path (`conv3x3_scratch_need`).
#[allow(clippy::too_many_arguments)]
fn conv3x3_scratch_need_codegen(
    in_h: usize,
    in_w: usize,
    in_c: usize,
    out_h: usize,
    out_w: usize,
    out_c: usize,
    filter_h: i32,
    filter_w: i32,
    stride_h: i32,
    stride_w: i32,
    dil_h: i32,
    dil_w: i32,
    input_offset: i32,
) -> usize {
    let dilated_filter_h = (filter_h - 1) * dil_h + 1;
    let dilated_filter_w = (filter_w - 1) * dil_w + 1;
    let pad_total_h =
        ((out_h as i32 - 1) * stride_h + dilated_filter_h - in_h as i32).max(0) as usize;
    let pad_total_w =
        ((out_w as i32 - 1) * stride_w + dilated_filter_w - in_w as i32).max(0) as usize;
    let padded_c = pad16(in_c);
    let needs_pad = pad_total_h > 0 || pad_total_w > 0 || padded_c != in_c;
    let wsum = if input_offset != 0 { out_c * 4 } else { 0 };
    if needs_pad {
        let pad_input_len = (in_h + pad_total_h) * (in_w + pad_total_w) * padded_c;
        let pad_weights_len = out_c * 9 * padded_c;
        pad_input_len + pad_weights_len + out_c * 4 + wsum
    } else {
        out_c * 4 + wsum
    }
}

/// Scratch bytes for the depthwise path (`depthwise_scratch_need`).
///
/// T3.5 — depth_multiplier > 1: the kernel consumes `out_c`-channel vectors
/// and the dispatch stages a REPLICATED input, so the padded channel count is
/// `pad16(out_c)` (== `pad16(in_c)` for dm==1) and dm>1 always forces the
/// staged path.
///
/// T3.5b — arbitrary filter sizes: the padded filter uses `taps = fh*fw`
/// rows (not 9), and the non-3x3 anytap path needs an extra `pad16(out_c)*4`
/// partial-accumulator buffer. Keep in sync with `hematite-s3/src/backend.rs`.
#[allow(clippy::too_many_arguments)]
pub fn depthwise_scratch_need_codegen(
    in_h: usize,
    in_w: usize,
    in_c: usize,
    out_h: usize,
    out_w: usize,
    out_c: usize,
    filter_h: i32,
    filter_w: i32,
    stride_h: i32,
    stride_w: i32,
    dil_h: i32,
    dil_w: i32,
    depth_multiplier: i32,
    input_offset: i32,
) -> usize {
    let is_3x3 = filter_h == 3 && filter_w == 3;
    let taps = (filter_h.max(0) as usize) * (filter_w.max(0) as usize);
    let dilated_filter_h = (filter_h - 1) * dil_h + 1;
    let dilated_filter_w = (filter_w - 1) * dil_w + 1;
    let pad_total_h =
        ((out_h as i32 - 1) * stride_h + dilated_filter_h - in_h as i32).max(0) as usize;
    let pad_total_w =
        ((out_w as i32 - 1) * stride_w + dilated_filter_w - in_w as i32).max(0) as usize;
    // dm==1: `out_c == in_c`, so pad16(out_c) == pad16(in_c) (historical).
    let padded_c = pad16(if depth_multiplier > 1 { out_c } else { in_c });
    let needs_channel_pad = padded_c != out_c;
    let needs_pad =
        pad_total_h > 0 || pad_total_w > 0 || needs_channel_pad || depth_multiplier > 1;
    let wsum = if input_offset != 0 { out_c * 4 } else { 0 };
    let partials = if is_3x3 { 0 } else { padded_c * 4 };
    if needs_pad {
        let pad_input_len = (in_h + pad_total_h) * (in_w + pad_total_w) * padded_c;
        let pad_filter_len = if needs_channel_pad { taps * padded_c } else { 0 };
        pad_input_len + pad_filter_len + padded_c * 4 + wsum + partials
    } else {
        out_c * 4 + wsum + partials
    }
}

fn emit_op(
    model: &ParsedModel,
    storage: &[Storage],
    ctx: &mut ArenaCtx,
    i: usize,
    op: &ParsedOp,
) -> Result<OpEmission, String> {
    match op.builtin_code {
        0 => emit_elementwise(model, storage, ctx, i, op, quote!(add), ElementwiseKind::AddSub),
        1 => emit_pool(model, storage, ctx, i, op, quote!(average_pool_2d)),
        3 => emit_conv2d(model, storage, ctx, i, op).map(|c| c.em),
        4 => emit_depthwise(model, storage, ctx, i, op),
        9 => emit_fully_connected(model, storage, ctx, i, op),
        17 => emit_pool(model, storage, ctx, i, op, quote!(max_pool_2d)),
        18 => emit_elementwise(model, storage, ctx, i, op, quote!(mul), ElementwiseKind::Mul),
        19 => emit_activation(model, storage, ctx, i, op, quote!(relu), 1),
        21 => emit_activation(model, storage, ctx, i, op, quote!(relu6), 3),
        22 => emit_reshape(model, storage, ctx, i, op),
        25 => emit_softmax(model, storage, ctx, i, op),
        34 => emit_pad(model, storage, ctx, i, op),
        39 => emit_transpose(model, storage, ctx, i, op),
        40 => emit_mean(model, storage, ctx, i, op),
        41 => emit_elementwise(model, storage, ctx, i, op, quote!(sub), ElementwiseKind::AddSub),
        54 => emit_prelu(model, storage, ctx, i, op),
        97 => emit_resize(model, storage, ctx, i, op),
        98 => emit_leaky_relu(model, storage, ctx, i, op),
        117 => emit_activation(model, storage, ctx, i, op, quote!(hard_swish), 0),
        code => Err(format!(
            "unsupported operator (builtin_code={code}) in subgraph[0]; T4.1 dispatches only the \
             in-scope op set (conv2d, depthwise_conv2d, fully_connected, average_pool_2d, \
             max_pool_2d, softmax, relu, relu6, hard_swish, leaky_relu, prelu, add, sub, mul, \
             mean, reshape, resize_nearest); this opcode is gated behind the T5 feature wave \
             (see local-notes/plans/hematite-nn.md, T5: extended op support)"
        )),
    }
}

fn emit_conv2d(model: &ParsedModel, storage: &[Storage], ctx: &mut ArenaCtx, i: usize, op: &ParsedOp) -> Result<ConvEmission, String> {
    let opts = match op.options.as_ref() {
        Some(ParsedOptions::Conv2D {
            padding,
            stride_w,
            stride_h,
            dilation_w,
            dilation_h,
            fused_activation,
        }) => (*padding, *stride_w, *stride_h, *dilation_w, *dilation_h, *fused_activation),
        other => return Err(format!("op {i}: expected Conv2D options, got {other:?}")),
    };
    let (padding, stride_w, stride_h, dilation_w, dilation_h, fused_activation) = opts;
    let in_t = *op.inputs.first().ok_or("conv2d missing input tensor")?;
    let w_t = *op.inputs.get(1).ok_or("conv2d missing weights tensor")?;
    let b_t = op.inputs.get(2).copied().unwrap_or(u32::MAX);
    let out_t = *op.outputs.first().ok_or("conv2d missing output tensor")?;
    let input = tensor_at(model.tensors(), in_t)?;
    let weights = tensor_at(model.tensors(), w_t)?;
    let output = tensor_at(model.tensors(), out_t)?;
    let input_raw = shape4(&input.shape)?;
    let input_shape = arr4(input_raw);
    let filter_raw = shape4(&weights.shape)?;
    let filter_shape = arr4(filter_raw);
    let output_raw = shape4(&output.shape)?;
    let output_shape = arr4(output_raw);
    let out_channels = filter_raw[0] as usize;
    let q = conv_quant(input, weights, output, out_channels, fused_activation)?;
    let input_offset = q.input_offset;
    let weights_offset = q.weights_offset;
    let output_offset = q.output_offset;
    let act_min = q.act_min;
    let act_max = q.act_max;
    let padding = padding_enum(padding)?;

    let w_name = Ident::new(&format!("WEIGHTS_{i}"), proc_macro2::Span::call_site());
    let b_name = Ident::new(&format!("BIAS_{i}"), proc_macro2::Span::call_site());
    let m_name = Ident::new(&format!("CONV2D_MULT_{i}"), proc_macro2::Span::call_site());
    let s_name = Ident::new(&format!("CONV2D_SHIFT_{i}"), proc_macro2::Span::call_site());
    let p_name = Ident::new(&format!("CONV2D_PARAMS_{i}"), proc_macro2::Span::call_site());

    let weights_c = weight_const(model, w_t, &w_name)?;
    let bias_c = bias_const(model, b_t, out_channels, &b_name)?;
    let mult_c = const_i32(&m_name, &q.multipliers);
    let shift_c = const_i32(&s_name, &q.shifts);
    let params_c = quote! {
        const #p_name: ::hematite_core::op_params::Conv2DParams<'static> =
            ::hematite_core::op_params::Conv2DParams {
                input_shape: #input_shape,
                filter_shape: #filter_shape,
                output_shape: #output_shape,
                padding: #padding,
                stride_width: #stride_w,
                stride_height: #stride_h,
                dilation_width_factor: #dilation_w,
                dilation_height_factor: #dilation_h,
                input_offset: #input_offset,
                weights_offset: #weights_offset,
                output_offset: #output_offset,
                output_multiplier_per_channel: &#m_name,
                output_shift_per_channel: &#s_name,
                quantized_activation_min: #act_min,
                quantized_activation_max: #act_max,
            };
    };
    let src = src_expr(ctx, storage, in_t as usize)?;
    let dst = dst_expr(ctx, storage, out_t as usize)?;
    let call = quote! {
        backend.conv2d(#src, &#w_name.0, &#b_name, &#p_name, #dst, scratch)?;
    };
    let scratch = if filter_raw[1] == 1 && filter_raw[2] == 1 {
        conv1x1_scratch_need_codegen(out_channels, input_offset)
    } else {
        conv3x3_scratch_need_codegen(
            input_raw[1].max(0) as usize,
            input_raw[2].max(0) as usize,
            input_raw[3].max(0) as usize,
            output_raw[1].max(0) as usize,
            output_raw[2].max(0) as usize,
            output_raw[3].max(0) as usize,
            filter_raw[1],
            filter_raw[2],
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            input_offset,
        )
    };
    Ok(ConvEmission {
        em: OpEmission {
            consts: vec![weights_c, bias_c, mult_c, shift_c, params_c],
            call,
            scratch,
        },
        weight: w_name,
        bias: b_name,
        mult: m_name,
        shift: s_name,
        params: p_name,
    })
}

fn emit_depthwise(model: &ParsedModel, storage: &[Storage], ctx: &mut ArenaCtx, i: usize, op: &ParsedOp) -> Result<OpEmission, String> {
    let opts = match op.options.as_ref() {
        Some(ParsedOptions::DepthwiseConv2D {
            padding,
            stride_w,
            stride_h,
            depth_multiplier,
            dilation_w,
            dilation_h,
            fused_activation,
        }) => (
            *padding,
            *stride_w,
            *stride_h,
            *depth_multiplier,
            *dilation_w,
            *dilation_h,
            *fused_activation,
        ),
        other => return Err(format!("op {i}: expected DepthwiseConv2D options, got {other:?}")),
    };
    let (padding, stride_w, stride_h, depth_multiplier, dilation_w, dilation_h, fused_activation) = opts;
    let in_t = *op.inputs.first().ok_or("depthwise missing input tensor")?;
    let w_t = *op.inputs.get(1).ok_or("depthwise missing weights tensor")?;
    let b_t = op.inputs.get(2).copied().unwrap_or(u32::MAX);
    let out_t = *op.outputs.first().ok_or("depthwise missing output tensor")?;
    let input = tensor_at(model.tensors(), in_t)?;
    let weights = tensor_at(model.tensors(), w_t)?;
    let output = tensor_at(model.tensors(), out_t)?;
    let input_raw = shape4(&input.shape)?;
    let input_shape = arr4(input_raw);
    let filter_raw = shape4(&weights.shape)?;
    let filter_shape = arr4(filter_raw);
    let output_raw = shape4(&output.shape)?;
    let output_shape = arr4(output_raw);
    let out_channels = filter_raw[3] as usize;
    let q = conv_quant(input, weights, output, out_channels, fused_activation)?;
    let input_offset = q.input_offset;
    let weights_offset = q.weights_offset;
    let output_offset = q.output_offset;
    let act_min = q.act_min;
    let act_max = q.act_max;
    let padding = padding_enum(padding)?;

    let w_name = Ident::new(&format!("WEIGHTS_{i}"), proc_macro2::Span::call_site());
    let b_name = Ident::new(&format!("BIAS_{i}"), proc_macro2::Span::call_site());
    let m_name = Ident::new(&format!("DEPTHWISE_MULT_{i}"), proc_macro2::Span::call_site());
    let s_name = Ident::new(&format!("DEPTHWISE_SHIFT_{i}"), proc_macro2::Span::call_site());
    let p_name = Ident::new(&format!("DEPTHWISE_PARAMS_{i}"), proc_macro2::Span::call_site());

    let weights_c = weight_const(model, w_t, &w_name)?;
    let bias_c = bias_const(model, b_t, out_channels, &b_name)?;
    let mult_c = const_i32(&m_name, &q.multipliers);
    let shift_c = const_i32(&s_name, &q.shifts);
    let params_c = quote! {
        const #p_name: ::hematite_core::op_params::DepthwiseConv2DParams<'static> =
            ::hematite_core::op_params::DepthwiseConv2DParams {
                input_shape: #input_shape,
                filter_shape: #filter_shape,
                output_shape: #output_shape,
                padding: #padding,
                stride_width: #stride_w,
                stride_height: #stride_h,
                dilation_width_factor: #dilation_w,
                dilation_height_factor: #dilation_h,
                depth_multiplier: #depth_multiplier,
                input_offset: #input_offset,
                weights_offset: #weights_offset,
                output_offset: #output_offset,
                output_multiplier_per_channel: &#m_name,
                output_shift_per_channel: &#s_name,
                quantized_activation_min: #act_min,
                quantized_activation_max: #act_max,
            };
    };
    let src = src_expr(ctx, storage, in_t as usize)?;
    let dst = dst_expr(ctx, storage, out_t as usize)?;
    let call = quote! {
        backend.depthwise_conv2d(#src, &#w_name.0, &#b_name, &#p_name, #dst, scratch)?;
    };
    let scratch = depthwise_scratch_need_codegen(
        input_raw[1].max(0) as usize,
        input_raw[2].max(0) as usize,
        input_raw[3].max(0) as usize,
        output_raw[1].max(0) as usize,
        output_raw[2].max(0) as usize,
        output_raw[3].max(0) as usize,
        filter_raw[1],
        filter_raw[2],
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        depth_multiplier,
        input_offset,
    );
    Ok(OpEmission {
        consts: vec![weights_c, bias_c, mult_c, shift_c, params_c],
        call,
        scratch,
    })
}

fn emit_fully_connected(model: &ParsedModel, storage: &[Storage], ctx: &mut ArenaCtx, i: usize, op: &ParsedOp) -> Result<OpEmission, String> {
    let opts = match op.options.as_ref() {
        Some(ParsedOptions::FullyConnected {
            fused_activation,
            keep_num_dims,
            ..
        }) => (*fused_activation, *keep_num_dims),
        other => return Err(format!("op {i}: expected FullyConnected options, got {other:?}")),
    };
    let (fused_activation, keep_num_dims) = opts;
    let in_t = *op.inputs.first().ok_or("fully_connected missing input tensor")?;
    let w_t = *op.inputs.get(1).ok_or("fully_connected missing weights tensor")?;
    let b_t = op.inputs.get(2).copied().unwrap_or(u32::MAX);
    let out_t = *op.outputs.first().ok_or("fully_connected missing output tensor")?;
    let input = tensor_at(model.tensors(), in_t)?;
    let weights = tensor_at(model.tensors(), w_t)?;
    let output = tensor_at(model.tensors(), out_t)?;

    let (batches, input_dim, output_dim) = fc_dimensions(input, output, keep_num_dims)?;
    let out_channels = output_dim as usize;
    let q = conv_quant(input, weights, output, out_channels, fused_activation)?;
    let input_offset = q.input_offset;
    let weights_offset = q.weights_offset;
    let output_offset = q.output_offset;
    let act_min = q.act_min;
    let act_max = q.act_max;

    let w_name = Ident::new(&format!("WEIGHTS_{i}"), proc_macro2::Span::call_site());
    let b_name = Ident::new(&format!("BIAS_{i}"), proc_macro2::Span::call_site());
    let m_name = Ident::new(&format!("FC_MULT_{i}"), proc_macro2::Span::call_site());
    let s_name = Ident::new(&format!("FC_SHIFT_{i}"), proc_macro2::Span::call_site());
    let p_name = Ident::new(&format!("FC_PARAMS_{i}"), proc_macro2::Span::call_site());

    let weights_c = weight_const(model, w_t, &w_name)?;
    let bias_c = bias_const(model, b_t, out_channels, &b_name)?;
    let mult_c = const_i32(&m_name, &q.multipliers);
    let shift_c = const_i32(&s_name, &q.shifts);
    let params_c = quote! {
        const #p_name: ::hematite_core::op_params::FullyConnectedParams<'static> =
            ::hematite_core::op_params::FullyConnectedParams {
                input_dim: #input_dim,
                output_dim: #output_dim,
                input_offset: #input_offset,
                weights_offset: #weights_offset,
                output_offset: #output_offset,
                output_multiplier_per_channel: &#m_name,
                output_shift_per_channel: &#s_name,
                quantized_activation_min: #act_min,
                quantized_activation_max: #act_max,
            };
    };
    let src = src_expr(ctx, storage, in_t as usize)?;
    let dst = dst_expr(ctx, storage, out_t as usize)?;
    let call = quote! {
        backend.fully_connected(#src, &#w_name.0, &#b_name, &#p_name, #dst, scratch)?;
    };
    let _ = batches;
    let out_dim = output_dim as usize;
    let in_dim = input_dim as usize;
    let fc_scratch = fc_scratch_need_codegen(in_dim, out_dim, input_offset);
    Ok(OpEmission {
        consts: vec![weights_c, bias_c, mult_c, shift_c, params_c],
        call,
        scratch: fc_scratch,
    })
}

/// Per-op CONV_2D emission plus the const names the composed
/// [`emit_fused_conv`] re-references (`FusedConvParams` embeds the anchor's
/// `Conv2DParams` and per-channel slices).
struct ConvEmission {
    em: OpEmission,
    weight: Ident,
    bias: Ident,
    mult: Ident,
    shift: Ident,
    params: Ident,
}

/// TFLM FC shape math: `input_dim` = flattened non-batch dims (or last dim
/// when `keep_num_dims`), `output_dim` = output elements per batch.
fn fc_dimensions(
    input: &ParsedTensor,
    output: &ParsedTensor,
    keep_num_dims: bool,
) -> Result<(i32, i32, i32), String> {
    let input_shape = &input.shape;
    let output_len = flat_len(&output.shape)? as i64;
    let (batches, input_dim): (i64, i64) = if keep_num_dims {
        let last = *input_shape.last().ok_or("empty input shape")?;
        (flat_len(&input_shape[..input_shape.len() - 1])? as i64, i64::from(last))
    } else if input_shape.len() == 1 {
        (1, i64::from(input_shape[0]))
    } else {
        let batch = i64::from(input_shape[0]);
        let mut rest: i64 = 1;
        for &d in &input_shape[1..] {
            if d <= 0 {
                return Err(format!("dynamic or non-positive shape dimension {d} is not supported"));
            }
            rest *= i64::from(d);
        }
        (batch, rest)
    };
    if batches <= 0 || output_len % batches != 0 {
        return Err(format!(
            "fully_connected batch size {batches} does not divide output length {output_len}"
        ));
    }
    let output_dim = output_len / batches;
    Ok((batches as i32, input_dim as i32, output_dim as i32))
}

fn emit_pool(
    model: &ParsedModel,
    storage: &[Storage],
    ctx: &mut ArenaCtx,
    i: usize,
    op: &ParsedOp,
    method: TokenStream,
) -> Result<OpEmission, String> {
    let opts = match op.options.as_ref() {
        Some(ParsedOptions::Pool2D {
            padding,
            stride_w,
            stride_h,
            filter_w,
            filter_h,
            fused_activation,
        }) => (*padding, *stride_w, *stride_h, *filter_w, *filter_h, *fused_activation),
        other => return Err(format!("op {i}: expected Pool2D options, got {other:?}")),
    };
    let (padding, stride_w, stride_h, filter_w, filter_h, fused_activation) = opts;
    let in_t = *op.inputs.first().ok_or("pool missing input tensor")?;
    let out_t = *op.outputs.first().ok_or("pool missing output tensor")?;
    let input = tensor_at(model.tensors(), in_t)?;
    let output = tensor_at(model.tensors(), out_t)?;
    let input_shape = arr4(shape4(&input.shape)?);
    let output_shape = arr4(shape4(&output.shape)?);
    let out_scale = tensor_scale(output)?;
    let out_zp = tensor_zp(output)?;
    let (act_min, act_max) = act_range(fused_activation, out_scale, out_zp);
    let padding = padding_enum(padding)?;
    let activation = fused_activation_enum(fused_activation);

    let p_name = Ident::new(&format!("POOL_PARAMS_{i}"), proc_macro2::Span::call_site());
    let params_c = quote! {
        const #p_name: ::hematite_core::op_params::PoolParams =
            ::hematite_core::op_params::PoolParams {
                input_shape: #input_shape,
                output_shape: #output_shape,
                filter_width: #filter_w,
                filter_height: #filter_h,
                stride_width: #stride_w,
                stride_height: #stride_h,
                padding: #padding,
                activation: #activation,
                quantized_activation_min: #act_min,
                quantized_activation_max: #act_max,
            };
    };
    let src = src_expr(ctx, storage, in_t as usize)?;
    let dst = dst_expr(ctx, storage, out_t as usize)?;
    let call = quote! {
        backend.#method(#src, &#p_name, #dst)?;
    };
    Ok(OpEmission {
        consts: vec![params_c],
        call,
        scratch: 0,
    })
}

fn emit_softmax(model: &ParsedModel, storage: &[Storage], ctx: &mut ArenaCtx, i: usize, op: &ParsedOp) -> Result<OpEmission, String> {
    // TFLM int8 softmax only supports beta = 1.0; a SoftmaxOptions table
    // carrying exactly that (the converter default) is accepted and ignored.
    match op.options.as_ref() {
        None => {}
        Some(ParsedOptions::Softmax { beta }) if (beta - 1.0).abs() < 1e-6 => {}
        Some(other) => {
            return Err(format!(
                "op {i}: softmax beta != 1.0 is not supported (T4.1 dispatch scope), got {other:?}"
            ));
        }
    }
    let in_t = *op.inputs.first().ok_or("softmax missing input tensor")?;
    let out_t = *op.outputs.first().ok_or("softmax missing output tensor")?;
    let input = tensor_at(model.tensors(), in_t)?;
    let output = tensor_at(model.tensors(), out_t)?;
    let shape = shape4(&input.shape)?;
    let row_size = shape[3];
    let num_rows = (flat_len(&input.shape)? as i32) / row_size;
    let in_scale = tensor_scale(input)?;
    let in_zp = tensor_zp(input)?;
    let (m, s) = quantize_multiplier(in_scale);
    // Q5.26 logit scaling: sadhg(diff, m) << (input_left_shift + 1) must equal
    // round(diff * in_scale * 2^26) → input_left_shift = 25 + s.
    let input_left_shift = 25 + s;

    // TFLM-correct `diff_min` (was hardcoded -128). TFLM softmax_common.cc
    // @ 18b9e6f2a8c5a9518e588f59c2ba16ef7ef9d551:
    //   diff_min = -CalculateInputRadius(kScaledDiffIntegerBits=5, left_shift),
    // with CalculateInputRadius (quantization_util.cc, same SHA)
    //   = floor((2^5 - 1) * 2^(31 - 5) / 2^left_shift) = floor(31 * 2^26 / 2^ls).
    // TFLM's shift comes from QuantizeMultiplier(in_scale * 2^26) → `26 + s`,
    // while we store `25 + s` (the extra +1 lives in the kernel's sadhg doubling
    // shift), so the radius uses `input_left_shift + 1` to match TFLM exactly:
    let diff_min = -(((31i64) << 26) >> (input_left_shift + 1)) as i32;

    let p_name = Ident::new(&format!("SOFTMAX_PARAMS_{i}"), proc_macro2::Span::call_site());
    let params_c = quote! {
        const #p_name: ::hematite_core::op_params::SoftmaxParams =
            ::hematite_core::op_params::SoftmaxParams {
                num_rows: #num_rows,
                row_size: #row_size,
                input_multiplier: #m,
                input_left_shift: #input_left_shift,
                diff_min: #diff_min,
                input_offset: #in_zp,
                output_offset: -128,
                quantized_activation_min: -128,
                quantized_activation_max: 127,
            };
    };
    let src = src_expr(ctx, storage, in_t as usize)?;
    let dst = dst_expr(ctx, storage, out_t as usize)?;
    let call = quote! {
        backend.softmax(#src, &#p_name, #dst, scratch)?;
    };
    let _ = output;
    Ok(OpEmission {
        consts: vec![params_c],
        call,
        scratch: (row_size.max(0) as usize) * 4,
    })
}

/// Standalone RELU (fused 1), RELU6 (fused 3), HARD_SWISH (fused NONE).
fn emit_activation(
    model: &ParsedModel,
    storage: &[Storage],
    ctx: &mut ArenaCtx,
    i: usize,
    op: &ParsedOp,
    method: TokenStream,
    fused: i8,
) -> Result<OpEmission, String> {
    let in_t = *op.inputs.first().ok_or("activation missing input tensor")?;
    let out_t = *op.outputs.first().ok_or("activation missing output tensor")?;
    let input = tensor_at(model.tensors(), in_t)?;
    let output = tensor_at(model.tensors(), out_t)?;
    if flat_len(&output.shape)? != flat_len(&input.shape)? {
        return Err(format!("op {i}: activation input/output element counts differ"));
    }
    let in_scale = tensor_scale(input)?;
    let out_scale = tensor_scale(output)?;
    let in_zp = tensor_zp(input)?;
    let out_zp = tensor_zp(output)?;
    let (output_multiplier, output_shift) = quantize_multiplier(in_scale / out_scale);
    let (act_min, act_max) = act_range(fused, out_scale, out_zp);

    let p_name = Ident::new(&format!("ACTIVATION_PARAMS_{i}"), proc_macro2::Span::call_site());
    let params_c = quote! {
        const #p_name: ::hematite_core::op_params::ActivationParams<'static> =
            ::hematite_core::op_params::ActivationParams {
                input_offset: #in_zp,
                output_offset: #out_zp,
                output_multiplier: #output_multiplier,
                output_shift: #output_shift,
                quantized_activation_min: #act_min,
                quantized_activation_max: #act_max,
                input_multiplier: 0,
                input_left_shift: 0,
                input_range_radius: 0,
                output_multiplier_alpha: 0,
                output_shift_alpha: 0,
                output_multiplier_identity: 0,
                output_shift_identity: 0,
                alpha_offset: 0,
                alpha_data: &[],
                output_multiplier_1: 0,
                output_shift_1: 0,
                output_multiplier_2: 0,
                output_shift_2: 0,
                reluish_multiplier_fixedpoint_int16: 0,
                reluish_multiplier_exponent: 0,
                output_multiplier_fixedpoint_int16: 0,
                output_multiplier_exponent: 0,
            };
    };
    let src = src_expr(ctx, storage, in_t as usize)?;
    let dst = dst_expr(ctx, storage, out_t as usize)?;
    let call = quote! {
        backend.#method(#src, &#p_name, #dst)?;
    };
    Ok(OpEmission {
        consts: vec![params_c],
        call,
        scratch: 0,
    })
}

fn emit_leaky_relu(
    model: &ParsedModel,
    storage: &[Storage],
    ctx: &mut ArenaCtx,
    i: usize,
    op: &ParsedOp,
) -> Result<OpEmission, String> {
    let alpha = match op.options.as_ref() {
        Some(ParsedOptions::LeakyRelu { alpha }) => f64::from(*alpha),
        other => return Err(format!("op {i}: expected LeakyRelu options, got {other:?}")),
    };
    let in_t = *op.inputs.first().ok_or("leaky_relu missing input tensor")?;
    let out_t = *op.outputs.first().ok_or("leaky_relu missing output tensor")?;
    let input = tensor_at(model.tensors(), in_t)?;
    let output = tensor_at(model.tensors(), out_t)?;
    let in_scale = tensor_scale(input)?;
    let out_scale = tensor_scale(output)?;
    let in_zp = tensor_zp(input)?;
    let out_zp = tensor_zp(output)?;
    let (id_mult, id_shift) = quantize_multiplier(in_scale / out_scale);
    let (al_mult, al_shift) = quantize_multiplier(in_scale * alpha / out_scale);

    let p_name = Ident::new(&format!("ACTIVATION_PARAMS_{i}"), proc_macro2::Span::call_site());
    let params_c = quote! {
        const #p_name: ::hematite_core::op_params::ActivationParams<'static> =
            ::hematite_core::op_params::ActivationParams {
                input_offset: #in_zp,
                output_offset: #out_zp,
                output_multiplier: 0,
                output_shift: 0,
                quantized_activation_min: -128,
                quantized_activation_max: 127,
                input_multiplier: 0,
                input_left_shift: 0,
                input_range_radius: 0,
                output_multiplier_alpha: #al_mult,
                output_shift_alpha: #al_shift,
                output_multiplier_identity: #id_mult,
                output_shift_identity: #id_shift,
                alpha_offset: 0,
                alpha_data: &[],
                output_multiplier_1: 0,
                output_shift_1: 0,
                output_multiplier_2: 0,
                output_shift_2: 0,
                reluish_multiplier_fixedpoint_int16: 0,
                reluish_multiplier_exponent: 0,
                output_multiplier_fixedpoint_int16: 0,
                output_multiplier_exponent: 0,
            };
    };
    let src = src_expr(ctx, storage, in_t as usize)?;
    let dst = dst_expr(ctx, storage, out_t as usize)?;
    let call = quote! {
        backend.leaky_relu(#src, &#p_name, #dst)?;
    };
    Ok(OpEmission {
        consts: vec![params_c],
        call,
        scratch: 0,
    })
}

fn emit_prelu(
    model: &ParsedModel,
    storage: &[Storage],
    ctx: &mut ArenaCtx,
    i: usize,
    op: &ParsedOp,
) -> Result<OpEmission, String> {
    if !matches!(op.options.as_ref(), None | Some(ParsedOptions::Prelu)) {
        return Err(format!("op {i}: unexpected options for prelu"));
    }
    let in_t = *op.inputs.first().ok_or("prelu missing input tensor")?;
    let a_t = *op.inputs.get(1).ok_or("prelu missing alpha tensor")?;
    let out_t = *op.outputs.first().ok_or("prelu missing output tensor")?;
    let input = tensor_at(model.tensors(), in_t)?;
    let alpha = tensor_at(model.tensors(), a_t)?;
    let output = tensor_at(model.tensors(), out_t)?;
    let in_scale = tensor_scale(input)?;
    let out_scale = tensor_scale(output)?;
    let in_zp = tensor_zp(input)?;
    let out_zp = tensor_zp(output)?;
    let alpha_zp = tensor_zp(alpha)?;
    let alpha_scale = tensor_scale(alpha)?;
    let (m1, s1) = quantize_multiplier(in_scale / out_scale);
    let (m2, s2) = quantize_multiplier(in_scale * alpha_scale / out_scale);

    let a_name = Ident::new(&format!("ALPHA_{i}"), proc_macro2::Span::call_site());
    let p_name = Ident::new(&format!("ACTIVATION_PARAMS_{i}"), proc_macro2::Span::call_site());
    let alpha_c = weight_const(model, a_t, &a_name)?;
    let params_c = quote! {
        const #p_name: ::hematite_core::op_params::ActivationParams<'static> =
            ::hematite_core::op_params::ActivationParams {
                input_offset: #in_zp,
                output_offset: #out_zp,
                output_multiplier: 0,
                output_shift: 0,
                quantized_activation_min: -128,
                quantized_activation_max: 127,
                input_multiplier: 0,
                input_left_shift: 0,
                input_range_radius: 0,
                output_multiplier_alpha: 0,
                output_shift_alpha: 0,
                output_multiplier_identity: 0,
                output_shift_identity: 0,
                alpha_offset: #alpha_zp,
                alpha_data: &#a_name.0,
                output_multiplier_1: #m1,
                output_shift_1: #s1,
                output_multiplier_2: #m2,
                output_shift_2: #s2,
                reluish_multiplier_fixedpoint_int16: 0,
                reluish_multiplier_exponent: 0,
                output_multiplier_fixedpoint_int16: 0,
                output_multiplier_exponent: 0,
            };
    };
    let src = src_expr(ctx, storage, in_t as usize)?;
    let dst = dst_expr(ctx, storage, out_t as usize)?;
    let call = quote! {
        backend.prelu(#src, &#p_name, #dst)?;
    };
    Ok(OpEmission {
        consts: vec![alpha_c, params_c],
        call,
        scratch: 0,
    })
}

#[derive(Clone, Copy, PartialEq)]
enum ElementwiseKind {
    AddSub,
    Mul,
}

/// TFLM int8 elementwise requantize triple (the `ElementwiseParams` quant
/// fields), shared by the per-op emission and the composed chain / pool-fold
/// param derivation (T1.2) so both use the identical rounding sequence.
struct ElementwiseQuant {
    in1_off: i32,
    in2_off: i32,
    out_off: i32,
    left_shift: i32,
    input1_multiplier: i32,
    input1_shift: i32,
    input2_multiplier: i32,
    input2_shift: i32,
    output_multiplier: i32,
    output_shift: i32,
}

/// ADD/SUB: twice_max scaling with `left_shift = 20` (add_common.cc); MUL:
/// single output ratio.  Offsets follow the per-op emission convention
/// (`-zp` inputs, `+zp` output).
fn elementwise_quant(
    input1: &ParsedTensor,
    input2: &ParsedTensor,
    output: &ParsedTensor,
    kind: ElementwiseKind,
) -> Result<ElementwiseQuant, String> {
    let in1_scale = tensor_scale(input1)?;
    let in2_scale = tensor_scale(input2)?;
    let out_scale = tensor_scale(output)?;
    let (left_shift, i1m, i1s, i2m, i2s, om, os) = match kind {
        ElementwiseKind::AddSub => {
            let twice_max = 2.0 * in1_scale.max(in2_scale);
            let ls = 20i32;
            let (a, b) = quantize_multiplier(in1_scale / twice_max);
            let (c, d) = quantize_multiplier(in2_scale / twice_max);
            let (e, f) = quantize_multiplier(twice_max / ((1i32 << ls) as f64 * out_scale));
            (ls, a, b, c, d, e, f)
        }
        ElementwiseKind::Mul => {
            let (e, f) = quantize_multiplier(in1_scale * in2_scale / out_scale);
            (0, 0, 0, 0, 0, e, f)
        }
    };
    Ok(ElementwiseQuant {
        in1_off: -tensor_zp(input1)?,
        in2_off: -tensor_zp(input2)?,
        out_off: tensor_zp(output)?,
        left_shift,
        input1_multiplier: i1m,
        input1_shift: i1s,
        input2_multiplier: i2m,
        input2_shift: i2s,
        output_multiplier: om,
        output_shift: os,
    })
}

fn emit_elementwise(
    model: &ParsedModel,
    storage: &[Storage],
    ctx: &mut ArenaCtx,
    i: usize,
    op: &ParsedOp,
    method: TokenStream,
    kind: ElementwiseKind,
) -> Result<OpEmission, String> {
    let fused_activation = match op.options.as_ref() {
        Some(ParsedOptions::Add { fused_activation, .. })
        | Some(ParsedOptions::Sub { fused_activation, .. })
        | Some(ParsedOptions::Mul { fused_activation }) => *fused_activation,
        None => 0,
        other => return Err(format!("op {i}: unexpected elementwise options, got {other:?}")),
    };
    let in1_t = *op.inputs.first().ok_or("elementwise missing input1 tensor")?;
    let in2_t = *op.inputs.get(1).ok_or("elementwise missing input2 tensor")?;
    let out_t = *op.outputs.first().ok_or("elementwise missing output tensor")?;
    let input1 = tensor_at(model.tensors(), in1_t)?;
    let input2 = tensor_at(model.tensors(), in2_t)?;
    let output = tensor_at(model.tensors(), out_t)?;
    let num_elements = flat_len(&input1.shape)? as i32;
    if flat_len(&input2.shape)? != num_elements as usize || flat_len(&output.shape)? != num_elements as usize {
        return Err(format!(
            "op {i}: elementwise input1/input2/output element counts must match"
        ));
    }
    let q = elementwise_quant(input1, input2, output, kind)?;
    let out_scale = tensor_scale(output)?;
    let out_off = tensor_zp(output)?;
    let (act_min, act_max) = act_range(fused_activation, out_scale, out_off);
    let q_in1_off = q.in1_off;
    let q_in2_off = q.in2_off;
    let q_out_off = q.out_off;
    let q_left_shift = q.left_shift;
    let q_i1m = q.input1_multiplier;
    let q_i1s = q.input1_shift;
    let q_i2m = q.input2_multiplier;
    let q_i2s = q.input2_shift;
    let q_om = q.output_multiplier;
    let q_os = q.output_shift;

    let p_name = Ident::new(&format!("ELEMENTWISE_PARAMS_{i}"), proc_macro2::Span::call_site());
    let params_c = quote! {
        const #p_name: ::hematite_core::op_params::ElementwiseParams =
            ::hematite_core::op_params::ElementwiseParams {
                num_elements: #num_elements,
                input1_offset: #q_in1_off,
                input2_offset: #q_in2_off,
                output_offset: #q_out_off,
                output_multiplier: #q_om,
                output_shift: #q_os,
                left_shift: #q_left_shift,
                input1_multiplier: #q_i1m,
                input1_shift: #q_i1s,
                input2_multiplier: #q_i2m,
                input2_shift: #q_i2s,
                quantized_activation_min: #act_min,
                quantized_activation_max: #act_max,
            };
    };
    let src1 = src_expr(ctx, storage, in1_t as usize)?;
    let src2 = src_expr(ctx, storage, in2_t as usize)?;
    let dst = dst_expr(ctx, storage, out_t as usize)?;
    let call = quote! {
        backend.#method(#src1, #src2, &#p_name, #dst)?;
    };
    Ok(OpEmission {
        consts: vec![params_c],
        call,
        scratch: 0,
    })
}

// ---------------------------------------------------------------------------
// Composed-kernel emitters (T1.2 — one `fused_*` call per composed group)
// ---------------------------------------------------------------------------

/// Emit the composed CONV_2D call for a group with an absorbed residual-ADD
/// and/or trailing activation (patterns (c) / (a)).
///
/// The `FusedConvParams` (T2.1) is built as a runtime `let` inside
/// `predict_with_scratch`: every value is a macro-time literal except the
/// anchor's consts (re-referenced) and the residual slice
/// (`&tensor_N.data` — the residual is a computed intermediate produced
/// before the conv, per the fusion `HasOneUse` guard).
fn emit_fused_conv(
    model: &ParsedModel,
    storage: &[Storage],
    ctx: &mut ArenaCtx,
    i: usize,
    op: &ParsedOp,
    group: &FusedGroup,
) -> Result<OpEmission, String> {
    // The per-op conv emission inside the composed path registers slices
    // for the anchor's OWN src/dst — discarded with the call; only its
    // consts and scratch are reused.  A throwaway ctx keeps those slices
    // out of the fused call's region list (the fused call re-registers the
    // slices it actually references).
    let conv = emit_conv2d(model, storage, &mut ArenaCtx::inactive(), i, op)?;
    let in_t = *op.inputs.first().ok_or("conv2d missing input tensor")?;
    let out_t = *op.outputs.first().ok_or("conv2d missing output tensor")?;
    let output = tensor_at(model.tensors(), out_t)?;
    let out_scale = tensor_scale(output)? as f32;
    let out_zp = tensor_zp(output)? as i64;

    let residual_ts = match &group.residual_add {
        Some(res) => {
            let r_t = tensor_at(model.tensors(), res.residual_tensor)?;
            let r_expr = tensor_ref_expr(ctx, storage, res.residual_tensor as usize)?;
            let r_scale = tensor_scale(r_t)? as f32;
            let r_zp = tensor_zp(r_t)? as i64;
            let r_out_scale = res.output_scale;
            let r_out_zp = res.output_zero_point;
            let rq = &res.requantize;
            let rq_i1m = rq.input1_multiplier;
            let rq_i1s = rq.input1_shift;
            let rq_i2m = rq.input2_multiplier;
            let rq_i2s = rq.input2_shift;
            let rq_ls = rq.left_shift;
            let rq_om = rq.output_multiplier;
            let rq_os = rq.output_shift;
            quote! {
                Some(::hematite_core::op_params::ResidualAddParams {
                    residual_data: #r_expr,
                    residual_scale: #r_scale,
                    residual_zero_point: #r_zp,
                    output_scale: #r_out_scale,
                    output_zero_point: #r_out_zp,
                    input1_multiplier: #rq_i1m,
                    input1_shift: #rq_i1s,
                    input2_multiplier: #rq_i2m,
                    input2_shift: #rq_i2s,
                    left_shift: #rq_ls,
                    output_multiplier: #rq_om,
                    output_shift: #rq_os,
                })
            }
        }
        None => quote!(None),
    };

    let (act_kind, act_in_off, act_out_off, act_mult, act_shift, act_min, act_max) =
        match &group.activation {
            Some(act) => {
                let act_idx = *group
                    .absorbed_ops
                    .last()
                    .ok_or_else(|| format!("group anchor {i}: activation without absorbed op"))?;
                let act_op = model
                    .ops()
                    .get(act_idx)
                    .ok_or_else(|| format!("group anchor {i}: absorbed op {act_idx} out of range"))?;
                let a_in_t = *act_op.inputs.first().ok_or("activation missing input tensor")?;
                let a_out_t = *act_op.outputs.first().ok_or("activation missing output tensor")?;
                let a_in = tensor_at(model.tensors(), a_in_t)?;
                let a_out = tensor_at(model.tensors(), a_out_t)?;
                let (am, ash) = quantize_multiplier(tensor_scale(a_in)? / tensor_scale(a_out)?);
                (
                    composed_activation_enum(act.kind),
                    -tensor_zp(a_in)?,
                    tensor_zp(a_out)?,
                    am,
                    ash,
                    act.quantized_min,
                    act.quantized_max,
                )
            }
            None => (
                quote!(::hematite_core::op_params::ComposedActivation::None),
                0i32,
                0i32,
                0i32,
                0i32,
                -128i32,
                127i32,
            ),
        };

    let src = src_expr(ctx, storage, in_t as usize)?;
    let dst = dst_expr(ctx, storage, group.output_tensor as usize)?;
    let conv_params = &conv.params;
    let conv_mult = &conv.mult;
    let conv_shift = &conv.shift;
    let conv_weight = &conv.weight;
    let conv_bias = &conv.bias;
    let p = Ident::new(&format!("FUSED_CONV_PARAMS_{i}"), proc_macro2::Span::call_site());
    let call = quote! {
        let #p = ::hematite_core::op_params::FusedConvParams {
            conv: #conv_params,
            output_scale: #out_scale,
            output_zero_point: #out_zp,
            output_multiplier_per_channel: &#conv_mult,
            output_shift_per_channel: &#conv_shift,
            residual: #residual_ts,
            activation: ::hematite_core::op_params::ActivationEpilogueParams {
                kind: #act_kind,
                input_offset: #act_in_off,
                output_offset: #act_out_off,
                output_multiplier: #act_mult,
                output_shift: #act_shift,
                quantized_activation_min: #act_min,
                quantized_activation_max: #act_max,
            },
        };
        backend.fused_conv2d(#src, &#conv_weight.0, &#conv_bias, &#p, #dst, scratch)?;
    };
    Ok(OpEmission {
        consts: conv.em.consts,
        call,
        scratch: conv.em.scratch,
    })
}

/// Emit the composed elementwise-chain call (pattern (b)): step 0 is the
/// ANCHOR elementwise op itself (quant derived from its own tensors, the
/// same math the per-op emitter uses), steps 1.. are the absorbed ops
/// (kind + operand + `StepRequantize` carried by the T1.1 IR).  Steps are
/// NEVER collapsed.
fn emit_fused_chain(
    model: &ParsedModel,
    storage: &[Storage],
    ctx: &mut ArenaCtx,
    i: usize,
    op: &ParsedOp,
    group: &FusedGroup,
) -> Result<OpEmission, String> {
    let anchor_kind: FusedStepKind = match op.builtin_code {
        0 => FusedStepKind::Add,
        18 => FusedStepKind::Mul,
        41 => FusedStepKind::Sub,
        code => {
            return Err(format!(
                "op {i}: elementwise-chain anchor has unsupported builtin_code {code}"
            ))
        }
    };
    let fused_activation = match op.options.as_ref() {
        Some(ParsedOptions::Add { fused_activation, .. })
        | Some(ParsedOptions::Sub { fused_activation, .. })
        | Some(ParsedOptions::Mul { fused_activation }) => *fused_activation,
        None => 0,
        other => return Err(format!("op {i}: unexpected elementwise options, got {other:?}")),
    };
    let in1_t = *op.inputs.first().ok_or("elementwise missing input1 tensor")?;
    let in2_t = *op.inputs.get(1).ok_or("elementwise missing input2 tensor")?;
    let out_t = *op.outputs.first().ok_or("elementwise missing output tensor")?;
    let input1 = tensor_at(model.tensors(), in1_t)?;
    let input2 = tensor_at(model.tensors(), in2_t)?;
    let output = tensor_at(model.tensors(), out_t)?;
    let num_elements = flat_len(&input1.shape)? as i32;
    if flat_len(&input2.shape)? != num_elements as usize
        || flat_len(&output.shape)? != num_elements as usize
    {
        return Err(format!(
            "op {i}: elementwise input1/input2/output element counts must match"
        ));
    }
    let local_kind = match anchor_kind {
        FusedStepKind::Add | FusedStepKind::Sub => ElementwiseKind::AddSub,
        FusedStepKind::Mul => ElementwiseKind::Mul,
        _ => unreachable!("anchor is ADD/SUB/MUL"),
    };
    let q = elementwise_quant(input1, input2, output, local_kind)?;
    let out_scale = tensor_scale(output)?;
    let (act_min, act_max) = act_range(fused_activation, out_scale, q.out_off);
    let (anchor_operand, anchor_operand_consts) = operand_data(
        model,
        ctx,
        storage,
        in2_t,
        &Ident::new(&format!("CHAIN_OPERAND_{i}_0"), proc_macro2::Span::call_site()),
    )?;

    let mut steps = Vec::with_capacity(group.elementwise_chain.len() + 1);
    let mut consts = anchor_operand_consts;
    let anchor_kind_enum = chain_step_kind_enum(anchor_kind);
    let q_in1_off = q.in1_off;
    let q_in2_off = q.in2_off;
    let q_out_off = q.out_off;
    let q_ls = q.left_shift;
    let q_i1m = q.input1_multiplier;
    let q_i1s = q.input1_shift;
    let q_i2m = q.input2_multiplier;
    let q_i2s = q.input2_shift;
    let q_om = q.output_multiplier;
    let q_os = q.output_shift;
    steps.push(quote! {
        ::hematite_core::op_params::ElementwiseChainStep {
            kind: #anchor_kind_enum,
            operand: #anchor_operand,
            input1_offset: #q_in1_off,
            input2_offset: #q_in2_off,
            output_offset: #q_out_off,
            output_multiplier: #q_om,
            output_shift: #q_os,
            left_shift: #q_ls,
            input1_multiplier: #q_i1m,
            input1_shift: #q_i1s,
            input2_multiplier: #q_i2m,
            input2_shift: #q_i2s,
            quantized_activation_min: #act_min,
            quantized_activation_max: #act_max,
        }
    });
    for (k, absorbed) in group.elementwise_chain.iter().enumerate() {
        let k1 = k + 1;
        let kind = chain_step_kind_enum(absorbed.kind);
        let (op_expr, op_consts) = match absorbed.operand_tensor {
            u32::MAX => (quote!(None), Vec::new()),
            t => {
                let (e, c) = operand_data(
        model,
        ctx,
        storage,
                    t,
                    &Ident::new(&format!("CHAIN_OPERAND_{i}_{k1}"), proc_macro2::Span::call_site()),
                )?;
                (quote!(Some(#e)), c)
            }
        };
        consts.extend(op_consts);
        let rq = &absorbed.requantize;
        let (amin, amax) = chain_step_act_range(model, absorbed)?;
        let rq_in1_off = rq.input1_offset;
        let rq_in2_off = rq.input2_offset;
        let rq_out_off = rq.output_offset;
        let rq_ls = rq.left_shift;
        let rq_i1m = rq.input1_multiplier;
        let rq_i1s = rq.input1_shift;
        let rq_i2m = rq.input2_multiplier;
        let rq_i2s = rq.input2_shift;
        let rq_om = rq.output_multiplier;
        let rq_os = rq.output_shift;
        steps.push(quote! {
            ::hematite_core::op_params::ElementwiseChainStep {
                kind: #kind,
                operand: #op_expr,
                input1_offset: #rq_in1_off,
                input2_offset: #rq_in2_off,
                output_offset: #rq_out_off,
                output_multiplier: #rq_om,
                output_shift: #rq_os,
                left_shift: #rq_ls,
                input1_multiplier: #rq_i1m,
                input1_shift: #rq_i1s,
                input2_multiplier: #rq_i2m,
                input2_shift: #rq_i2s,
                quantized_activation_min: #amin,
                quantized_activation_max: #amax,
            }
        });
    }
    let src = src_expr(ctx, storage, in1_t as usize)?;
    let dst = dst_expr(ctx, storage, group.output_tensor as usize)?;
    let p = Ident::new(&format!("CHAIN_PARAMS_{i}"), proc_macro2::Span::call_site());
    let call = quote! {
        let #p = ::hematite_core::op_params::ElementwiseChainParams {
            num_elements: #num_elements,
            steps: &[ #(#steps),* ],
        };
        backend.fused_elementwise_chain(#src, &#p, #dst)?;
    };
    Ok(OpEmission {
        consts,
        call,
        scratch: 0,
    })
}

/// Clamp range of an absorbed chain step: the step op's own
/// `fused_activation` field against its output tensor quant (mirrors the
/// per-op elementwise emission; identity when the field is NONE).
fn chain_step_act_range(
    model: &ParsedModel,
    absorbed: &AbsorbedElementwise,
) -> Result<(i32, i32), String> {
    let op = model
        .ops()
        .get(absorbed.op_index)
        .ok_or_else(|| format!("absorbed op {} out of range", absorbed.op_index))?;
    let fused = match op.options.as_ref() {
        Some(ParsedOptions::Add { fused_activation, .. })
        | Some(ParsedOptions::Sub { fused_activation, .. })
        | Some(ParsedOptions::Mul { fused_activation }) => *fused_activation,
        _ => 0,
    };
    Ok(act_range(
        fused,
        f64::from(absorbed.output_scale),
        absorbed.output_zero_point as i32,
    ))
}

/// Emit the composed pool-with-fold call (pattern (d)) for a pool anchor.
///
/// Reaches emission only once an input-fold group's fused==unfused
/// verification passes (T2 groups stay per-op); the composed params embed
/// the anchor `PoolParams` (re-referencing `emit_pool`'s const) plus the
/// fold's per-op elementwise quant.
fn emit_fused_pool_fold(
    model: &ParsedModel,
    storage: &[Storage],
    ctx: &mut ArenaCtx,
    i: usize,
    op: &ParsedOp,
    group: &FusedGroup,
) -> Result<OpEmission, String> {
    let fold = group
        .input_fold
        .as_ref()
        .ok_or_else(|| format!("op {i}: pool-fold group without input_fold"))?;
    let pool_kind = match op.builtin_code {
        1 => quote!(::hematite_core::op_params::PoolKind::Average),
        17 => quote!(::hematite_core::op_params::PoolKind::Max),
        code => {
            return Err(format!(
                "op {i}: pool-fold anchor has unsupported builtin_code {code}"
            ))
        }
    };
    let pool = emit_pool(model, storage, ctx, i, op, quote!(average_pool_2d))?;
    let p_name = Ident::new(&format!("POOL_PARAMS_{i}"), proc_macro2::Span::call_site());

    let fold_op = model
        .ops()
        .get(fold.op_index)
        .ok_or_else(|| format!("op {i}: fold op {} out of range", fold.op_index))?;
    let fold_out_t = *fold_op
        .outputs
        .first()
        .ok_or_else(|| format!("op {i}: fold op missing output tensor"))?;
    let fold_out = tensor_at(model.tensors(), fold_out_t)?;
    let in1 = tensor_at(model.tensors(), fold.folded_input_tensor)?;
    let in2 = tensor_at(model.tensors(), fold.operand_tensor)?;
    let q = elementwise_quant(
        in1,
        in2,
        fold_out,
        if fold.builtin == 18 { ElementwiseKind::Mul } else { ElementwiseKind::AddSub },
    )?;
    let num_elements = flat_len(&in1.shape)? as i32;
    let (operand, operand_consts) = operand_data(
        model,
        ctx,
        storage,
        fold.operand_tensor,
        &Ident::new(&format!("FOLD_OPERAND_{i}"), proc_macro2::Span::call_site()),
    )?;
    let op_zp = tensor_zp(in2)? as i64;
    let in_zp = fold.input_zero_point;
    let out_zp = tensor_zp(fold_out)? as i64;
    let folded_scale = fold.folded_scale;
    let fold_builtin = fold.builtin;
    let q_ls = q.left_shift;
    let q_om = q.output_multiplier;
    let q_os = q.output_shift;
    let q_i1m = q.input1_multiplier;
    let q_i1s = q.input1_shift;
    let q_i2m = q.input2_multiplier;
    let q_i2s = q.input2_shift;

    let src = src_expr(ctx, storage, fold.folded_input_tensor as usize)?;
    let dst = dst_expr(ctx, storage, group.output_tensor as usize)?;
    let p = Ident::new(&format!("FUSED_POOL_PARAMS_{i}"), proc_macro2::Span::call_site());
    let call = quote! {
        let #p = ::hematite_core::op_params::FoldedPoolParams {
            pool: #p_name,
            pool_kind: #pool_kind,
            fold: Some(::hematite_core::op_params::PoolInputFold {
                builtin: #fold_builtin,
                operand_data: #operand,
                operand_zero_point: #op_zp,
                input_zero_point: #in_zp,
                output_zero_point: #out_zp,
                folded_scale: #folded_scale,
                left_shift: #q_ls,
                output_multiplier: #q_om,
                output_shift: #q_os,
                input1_multiplier: #q_i1m,
                input1_shift: #q_i1s,
                input2_multiplier: #q_i2m,
                input2_shift: #q_i2s,
                num_elements: #num_elements,
            }),
            activation: ::hematite_core::op_params::ActivationEpilogueParams {
                kind: ::hematite_core::op_params::ComposedActivation::None,
                input_offset: 0,
                output_offset: 0,
                output_multiplier: 0,
                output_shift: 0,
                quantized_activation_min: -128,
                quantized_activation_max: 127,
            },
        };
        backend.fused_pool_with_fold(#src, &#p, #dst, scratch)?;
    };
    let mut consts = pool.consts;
    consts.extend(operand_consts);
    Ok(OpEmission {
        consts,
        call,
        scratch: num_elements.max(0) as usize,
    })
}

fn emit_mean(model: &ParsedModel, storage: &[Storage], ctx: &mut ArenaCtx, i: usize, op: &ParsedOp) -> Result<OpEmission, String> {
    let (axis, keep_dims) = match op.options.as_ref() {
        Some(ParsedOptions::Mean { axis, keep_dims }) => (axis.clone(), *keep_dims),
        other => return Err(format!("op {i}: expected Mean options, got {other:?}")),
    };
    if axis.is_empty() || axis.len() > 4 {
        return Err(format!("op {i}: mean axis count {} out of range (1..=4)", axis.len()));
    }
    let axis_count = axis.len() as i8;
    let mut axis_arr = [0i16; 4];
    for (k, &a) in axis.iter().enumerate() {
        axis_arr[k] = i16::try_from(a).map_err(|_| format!("op {i}: mean axis {a} out of i16 range"))?;
    }
    let in_t = *op.inputs.first().ok_or("mean missing input tensor")?;
    let out_t = *op.outputs.first().ok_or("mean missing output tensor")?;
    let input = tensor_at(model.tensors(), in_t)?;
    let output = tensor_at(model.tensors(), out_t)?;
    let input_shape = arr4(shape4(&input.shape)?);
    let output_shape = arr4(shape4(&output.shape)?);
    let in_scale = tensor_scale(input)?;
    let out_scale = tensor_scale(output)?;
    let in_off = -tensor_zp(input)?;
    let out_off = tensor_zp(output)?;
    let (om, os) = quantize_multiplier(in_scale / out_scale);

    let p_name = Ident::new(&format!("REDUCE_PARAMS_{i}"), proc_macro2::Span::call_site());
    let axis_arr_0 = axis_arr[0];
    let axis_arr_1 = axis_arr[1];
    let axis_arr_2 = axis_arr[2];
    let axis_arr_3 = axis_arr[3];
    let params_c = quote! {
        const #p_name: ::hematite_core::op_params::ReduceParams =
            ::hematite_core::op_params::ReduceParams {
                keep_dims: #keep_dims,
                axis: [#axis_arr_0, #axis_arr_1, #axis_arr_2, #axis_arr_3],
                axis_count: #axis_count,
                input_shape: #input_shape,
                output_shape: #output_shape,
                output_type: 0,
                input_offset: #in_off,
                output_offset: #out_off,
                output_multiplier: #om,
                output_shift: #os,
                quantized_activation_min: -128,
                quantized_activation_max: 127,
            };
    };
    let src = src_expr(ctx, storage, in_t as usize)?;
    let dst = dst_expr(ctx, storage, out_t as usize)?;
    let call = quote! {
        backend.mean(#src, &#p_name, #dst)?;
    };
    Ok(OpEmission {
        consts: vec![params_c],
        call,
        scratch: 0,
    })
}

fn emit_reshape(model: &ParsedModel, storage: &[Storage], ctx: &mut ArenaCtx, i: usize, op: &ParsedOp) -> Result<OpEmission, String> {
    let out_t = *op.outputs.first().ok_or("reshape missing output tensor")?;
    let in_t = *op.inputs.first().ok_or("reshape missing input tensor")?;
    let output = tensor_at(model.tensors(), out_t)?;
    // ReshapeOptions may carry a target with 0 (copy-from-input) or -1
    // (infer-from-count) entries; the model's output tensor shape IS the
    // resolved shape, so fall back to it whenever the target is not fully
    // static-positive.
    let target: Vec<i32> = match op.options.as_ref() {
        Some(ParsedOptions::Reshape { new_shape })
            if !new_shape.is_empty() && new_shape.iter().all(|&d| d > 0) =>
        {
            new_shape.clone()
        }
        _ => output.shape.clone(),
    };
    if target.len() > 4 {
        return Err(format!("op {i}: reshape target rank {} exceeds 4", target.len()));
    }
    let mut shape = [0i32; 4];
    for (k, &d) in target.iter().enumerate() {
        if d <= 0 {
            return Err(format!("op {i}: reshape dim {d} is not supported (static shapes only)"));
        }
        shape[k] = d;
    }
    let shape_count = target.len() as i8;
    let shape = arr4(shape);
    let p_name = Ident::new(&format!("RESHAPE_PARAMS_{i}"), proc_macro2::Span::call_site());
    let params_c = quote! {
        const #p_name: ::hematite_core::op_params::ReshapeParams =
            ::hematite_core::op_params::ReshapeParams {
                shape: #shape,
                shape_count: #shape_count,
            };
    };
    let src = src_expr(ctx, storage, in_t as usize)?;
    let dst = dst_expr(ctx, storage, out_t as usize)?;
    let call = quote! {
        backend.reshape(#src, &#p_name, #dst)?;
    };
    Ok(OpEmission {
        consts: vec![params_c],
        call,
        scratch: 0,
    })
}

/// PAD — the padding amounts come from a const int32 tensor of shape
/// `[rank, 2]` (`[before, after]` per dim); the kernel pads with value 0.
fn emit_pad(model: &ParsedModel, storage: &[Storage], ctx: &mut ArenaCtx, i: usize, op: &ParsedOp) -> Result<OpEmission, String> {
    let in_t = *op.inputs.first().ok_or("pad missing input tensor")?;
    let pad_t = *op.inputs.get(1).ok_or("pad missing padding tensor")?;
    let out_t = *op.outputs.first().ok_or("pad missing output tensor")?;
    let input = tensor_at(model.tensors(), in_t)?;
    let padding = tensor_at(model.tensors(), pad_t)?;
    let output = tensor_at(model.tensors(), out_t)?;
    let rank = input.shape.len();
    if !(1..=4).contains(&rank) {
        return Err(format!("op {i}: pad input rank {rank} outside the static-shape range 1..=4"));
    }
    let bytes = model.buffer_data(padding).ok_or_else(|| {
        format!("op {i}: padding tensor '{}' has no const buffer", padding.name)
    })?;
    if bytes.len() != rank * 2 * 4 {
        return Err(format!(
            "op {i}: padding tensor '{}' buffer length {} != rank*2*4 = {}",
            padding.name,
            bytes.len(),
            rank * 2 * 4
        ));
    }
    let mut left = [0i32; 4];
    let mut right = [0i32; 4];
    for d in 0..rank {
        let lo = d * 8;
        left[d] = i32::from_le_bytes([bytes[lo], bytes[lo + 1], bytes[lo + 2], bytes[lo + 3]]);
        right[d] = i32::from_le_bytes([bytes[lo + 4], bytes[lo + 5], bytes[lo + 6], bytes[lo + 7]]);
    }
    let input_shape = arr4(shape4(&input.shape)?);
    let output_shape = arr4(shape4(&output.shape)?);
    let left = arr4(left);
    let right = arr4(right);
    let count = rank as i8;

    let p_name = Ident::new(&format!("PAD_PARAMS_{i}"), proc_macro2::Span::call_site());
    let params_c = quote! {
        const #p_name: ::hematite_core::op_params::PadParams =
            ::hematite_core::op_params::PadParams {
                input_shape: #input_shape,
                output_shape: #output_shape,
                left_padding: #left,
                left_padding_count: #count,
                right_padding: #right,
                right_padding_count: #count,
            };
    };
    let src = src_expr(ctx, storage, in_t as usize)?;
    let dst = dst_expr(ctx, storage, out_t as usize)?;
    let call = quote! {
        backend.pad(#src, &#p_name, #dst)?;
    };
    Ok(OpEmission {
        consts: vec![params_c],
        call,
        scratch: 0,
    })
}

/// TRANSPOSE — the permutation comes from a const int32 tensor of length
/// `rank`; the kernel computes the output shape from the permuted input.
fn emit_transpose(model: &ParsedModel, storage: &[Storage], ctx: &mut ArenaCtx, i: usize, op: &ParsedOp) -> Result<OpEmission, String> {
    let in_t = *op.inputs.first().ok_or("transpose missing input tensor")?;
    let perm_t = *op.inputs.get(1).ok_or("transpose missing perm tensor")?;
    let out_t = *op.outputs.first().ok_or("transpose missing output tensor")?;
    let input = tensor_at(model.tensors(), in_t)?;
    let perm_tensor = tensor_at(model.tensors(), perm_t)?;
    let _ = tensor_at(model.tensors(), out_t)?;
    let rank = input.shape.len();
    if !(1..=4).contains(&rank) {
        return Err(format!("op {i}: transpose input rank {rank} outside the static-shape range 1..=4"));
    }
    let bytes = model.buffer_data(perm_tensor).ok_or_else(|| {
        format!("op {i}: perm tensor '{}' has no const buffer", perm_tensor.name)
    })?;
    if bytes.len() != rank * 4 {
        return Err(format!(
            "op {i}: perm tensor '{}' buffer length {} != rank*4 = {}",
            perm_tensor.name,
            bytes.len(),
            rank * 4
        ));
    }
    let mut perm = [0i32; 8];
    let mut perm_count = 0i8;
    for (d, slot) in perm.iter_mut().take(rank).enumerate() {
        let lo = d * 4;
        let p = i32::from_le_bytes([bytes[lo], bytes[lo + 1], bytes[lo + 2], bytes[lo + 3]]);
        if p < 0 || p as usize >= rank {
            return Err(format!("op {i}: transpose perm entry {p} out of range for rank {rank}"));
        }
        *slot = p;
        perm_count += 1;
    }
    let input_shape = arr4(shape4(&input.shape)?);
    let perm = arr8(perm);

    let p_name = Ident::new(&format!("TRANSPOSE_PARAMS_{i}"), proc_macro2::Span::call_site());
    let params_c = quote! {
        const #p_name: ::hematite_core::op_params::TransposeParams =
            ::hematite_core::op_params::TransposeParams {
                input_shape: #input_shape,
                perm: #perm,
                perm_count: #perm_count,
            };
    };
    let src = src_expr(ctx, storage, in_t as usize)?;
    let dst = dst_expr(ctx, storage, out_t as usize)?;
    let call = quote! {
        backend.transpose(#src, &#p_name, #dst)?;
    };
    Ok(OpEmission {
        consts: vec![params_c],
        call,
        scratch: 0,
    })
}

fn emit_resize(model: &ParsedModel, storage: &[Storage], ctx: &mut ArenaCtx, i: usize, op: &ParsedOp) -> Result<OpEmission, String> {    let (align_corners, half_pixel_centers) = match op.options.as_ref() {
        Some(ParsedOptions::ResizeNearest {
            align_corners,
            half_pixel_centers,
        }) => (*align_corners, *half_pixel_centers),
        other => return Err(format!("op {i}: expected ResizeNearest options, got {other:?}")),
    };
    let in_t = *op.inputs.first().ok_or("resize missing input tensor")?;
    let out_t = *op.outputs.first().ok_or("resize missing output tensor")?;
    let input = tensor_at(model.tensors(), in_t)?;
    let output = tensor_at(model.tensors(), out_t)?;
    let input_shape = arr4(shape4(&input.shape)?);
    let output_shape = arr4(shape4(&output.shape)?);

    let p_name = Ident::new(&format!("RESIZE_PARAMS_{i}"), proc_macro2::Span::call_site());
    let params_c = quote! {
        const #p_name: ::hematite_core::op_params::ResizeNearestParams =
            ::hematite_core::op_params::ResizeNearestParams {
                input_shape: #input_shape,
                output_shape: #output_shape,
                align_corners: #align_corners as i32,
                half_pixel_centers: #half_pixel_centers as i32,
            };
    };
    let src = src_expr(ctx, storage, in_t as usize)?;
    let dst = dst_expr(ctx, storage, out_t as usize)?;
    let call = quote! {
        backend.resize_nearest(#src, &#p_name, #dst)?;
    };
    Ok(OpEmission {
        consts: vec![params_c],
        call,
        scratch: 0,
    })
}

// ---------------------------------------------------------------------------
// In-crate unit tests (proc-macro restriction: integration tests can only
// invoke the macro — every codegen test lives in-file).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flatbuffer;

    const SINE_TFLITE: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../models/sine.tflite"
    ));

    /// All six zoo models, same paths as the fused-pattern profile
    /// (optimize/profile.rs) and model_validation.rs.
    const HELLO_WORLD_TFLITE: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../models/zoo/sine_regression/hello_world_int8.tflite"
    ));
    const KWS_TFLITE: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../models/zoo/keyword_spotting/kws_micro_speech_int8.tflite"
    ));
    const ANOMALY_TFLITE: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../models/zoo/anomaly_detect/anomaly_detect_int8.tflite"
    ));
    const PERSON_DETECT_TFLITE: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../models/zoo/person_detect_vww/person_detect_int8.tflite"
    ));
    const MOBILENET_V2_TFLITE: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../models/zoo/mobilenetv2_cls/mobilenet_v2_1.0_224_int8.tflite"
    ));

    /// T1.3 — the emitted `ARENA_LEN` const must equal the planner's peak
    /// for every arena-emitting model, the arena must be 16-byte aligned,
    /// and NO `unsafe` / `static mut` may leak into either emission mode.
    /// mobilenet_v2 falls back to per-tensor arrays (a single 224²×32
    /// activation exceeds MAX_INTERNAL) — its ARENA_LEN is 0.
    #[test]
    fn emitted_arena_len_matches_plan_peak() {
        use crate::optimize::arena::plan_arena;
        let cases: [(&str, &[u8], usize); 6] = [
            ("sine", SINE_TFLITE, 0), // no intermediates
            ("hello_world", HELLO_WORLD_TFLITE, 32),
            ("kws", KWS_TFLITE, 5_968),
            ("anomaly_detect", ANOMALY_TFLITE, 272),
            ("person_detect", PERSON_DETECT_TFLITE, 55_296),
            ("mobilenet_v2", MOBILENET_V2_TFLITE, 0), // fallback: Oversized
        ];
        for (name, bytes, expected) in cases {
            let model = flatbuffer::parse(bytes).unwrap_or_else(|e| panic!("{name}: parse: {e}"));
            let peak = plan_arena(&model).map(|p| p.peak_arena_bytes).unwrap_or(0);
            assert_eq!(peak, expected, "{name}: plan peak (t13-arena.md)");
            let ts = emit_model(&model).unwrap_or_else(|e| panic!("{name}: emit: {e}"));
            let s: String = ts.to_string().chars().filter(|c| !c.is_whitespace()).collect();
            assert!(
                s.contains(&format!("ARENA_LEN:usize={peak}usize")),
                "{name}: ARENA_LEN const wrong: {s}"
            );
            let arena_mode = peak > 0;
            assert_eq!(
                s.contains("structArena{data:[i8;"),
                arena_mode,
                "{name}: arena struct presence wrong"
            );
            assert_eq!(
                s.contains("split_at_mut"),
                arena_mode,
                "{name}: arena borrows presence wrong"
            );
            assert!(
                s.contains("align(16))"),
                "{name}: 16-byte alignment missing (SIMD gate)"
            );
            assert!(!s.contains("unsafe"), "{name}: unsafe leaked into generated code");
            assert!(!s.contains("staticmut"), "{name}: static mut leaked into generated code");
            if name == "mobilenet_v2" {
                assert!(
                    s.contains("structTENSOR_"),
                    "mobilenet_v2: fallback must keep per-tensor arrays"
                );
            }
        }
    }

    /// T1.3 — the stack arm (`emit_model_stack`) must NOT emit the arena for
    /// a model that arena-emits, and must stay unsafe/static-mut-free.
    #[test]
    fn stack_emission_has_no_arena() {
        let model = flatbuffer::parse(KWS_TFLITE).expect("kws parses");
        let schedule = crate::optimize::fusion::fuse(&model);
        let ts = emit_model_stack_fused(&model, &schedule).expect("kws stack emits");
        let s: String = ts.to_string().chars().filter(|c| !c.is_whitespace()).collect();
        assert!(!s.contains("structArena"), "stack arm must not emit the arena");
        assert!(s.contains("ARENA_LEN:usize=0usize"), "stack arm: ARENA_LEN must be 0");
        assert!(s.contains("structTENSOR_"), "stack arm must emit per-tensor arrays");
        assert!(!s.contains("unsafe"), "stack arm: unsafe leaked");
        assert!(!s.contains("staticmut"), "stack arm: static mut leaked");
    }

    /// T1.2 regression: for a model with zero composed groups the fused
    /// emission must be byte-identical (token-identical) to the unfused
    /// per-op emission — every one of the 5 non-mv2 zoo models qualifies.
    #[test]
    fn fused_emission_byte_identical_when_no_composed_groups() {
        for (name, bytes) in [
            ("sine", SINE_TFLITE),
            ("hello_world", HELLO_WORLD_TFLITE),
            ("kws", KWS_TFLITE),
            ("anomaly_detect", ANOMALY_TFLITE),
            ("person_detect", PERSON_DETECT_TFLITE),
        ] {
            let model = flatbuffer::parse(bytes).unwrap_or_else(|e| panic!("{name}: parse: {e}"));
            let schedule = crate::optimize::fusion::fuse(&model);
            assert!(
                schedule.groups.iter().all(|g| composed_kind(g).is_none()),
                "{name}: expected zero composed groups"
            );
            let fused = emit_model_fused(&model, &schedule)
                .unwrap_or_else(|e| panic!("{name}: fused emit: {e}"))
                .to_string();
            let unfused = emit_model(&model)
                .unwrap_or_else(|e| panic!("{name}: unfused emit: {e}"))
                .to_string();
            assert_eq!(fused, unfused, "{name}: fused emission diverged from per-op");
        }
    }

    /// W0 profile pins (fused-profile.md): mv2 84 ops → 74 groups, 10
    /// residual-add groups, 10 eliminated tensors — the ONLY composed
    /// groups in the zoo.  The fused emission collapses each to one
    /// `fused_conv2d` call and stays unsafe-free.
    #[test]
    fn mobilenet_fused_schedule_matches_w0_profile() {
        let model = flatbuffer::parse(MOBILENET_V2_TFLITE).expect("mv2 parses");
        let schedule = crate::optimize::fusion::fuse(&model);
        assert_eq!(schedule.total_ops, 84);
        assert_eq!(schedule.groups.len(), 74);
        assert_eq!(schedule.fused_op_count(), 10);
        let residual_groups = schedule
            .groups
            .iter()
            .filter(|g| g.residual_add.is_some())
            .count();
        assert_eq!(residual_groups, 10);
        let eliminated: usize = schedule.groups.iter().map(|g| g.eliminated_tensors.len()).sum();
        assert_eq!(eliminated, 10);
        assert!(
            schedule.groups.iter().all(|g| !g.requires_verification),
            "mv2: all groups must be T1 (no verification obligation)"
        );
        let composed = schedule
            .groups
            .iter()
            .filter(|g| composed_kind(g).is_some())
            .count();
        assert_eq!(composed, 10);

        let fused = emit_model_fused(&model, &schedule).expect("mv2 fused emits");
        let fused_s: String = fused.to_string().chars().filter(|c| !c.is_whitespace()).collect();
        let unfused = emit_model(&model).expect("mv2 unfused emits");
        let unfused_s: String = unfused.to_string().chars().filter(|c| !c.is_whitespace()).collect();

        assert_eq!(fused_s.matches("backend.fused_conv2d(").count(), 10);
        assert!(!unfused_s.contains("backend.fused_conv2d("));
        assert!(!fused_s.contains("unsafe"), "generated code must be unsafe-free");
        assert!(!unfused_s.contains("unsafe"), "generated code must be unsafe-free");
    }

    #[test]
    fn sine_model_emits_fc_call_sequence() {
        let model = flatbuffer::parse(SINE_TFLITE).expect("sine parses");
        let ts = emit_model(&model).expect("sine emits");
        // `to_string()` inserts spaces between tokens — compare whitespace-free.
        let s: String = ts.to_string().chars().filter(|c| !c.is_whitespace()).collect();

        // Model<I, O> wrapper with the typed I/O bridge.
        assert!(s.contains("structModel<B>"), "Model wrapper missing");
        assert!(s.contains("fnpredict_with_scratch"), "predict_with_scratch missing");
        assert!(s.contains("fnpredict("), "predict missing");
        assert!(s.contains("constfnnew("), "new missing");
        assert!(s.contains("constfninput_len"), "input_len missing");
        assert!(s.contains("constfnoutput_len"), "output_len missing");

        // I/O sized from tensor shapes (input [1], output [1]).
        assert!(s.contains("INPUT_LEN:usize=1usize"));
        assert!(s.contains("OUTPUT_LEN:usize=1usize"));

        // Scratch computed at macro time — FC 1→1, input_offset 0. T3.6 pad
        // path: input_dim 1 pads to 16 (padded input 16 + padded weights
        // 1×16 + accs 1×4 = 36), mirroring `fc_scratch_need` in backend.rs.
        assert!(s.contains("SCRATCH_LEN:usize=36usize"), "scratch len wrong: {s}");

        // Weight/bias consts from buffer bytes. Weights are wrapped in a
        // `#[repr(C, align(16))]` struct so the ACCX/FC SIMD `w_ptr % 16 == 0`
        // alignment gate engages (see `const_i8`).
        assert!(s.contains("structWEIGHTS_0Ty([i8;1usize])"), "weight wrapper missing: {s}");
        assert!(s.contains("constWEIGHTS_0:WEIGHTS_0Ty=WEIGHTS_0Ty([51i8])"), "weight const wrong: {s}");
        assert!(s.contains("BIAS_0:[i32;1usize]=[-3i32]"), "bias const wrong: {s}");

        // FC params: input_dim=1, output_dim=1, per-channel quant
        // in_scale*w_scale/out_scale = 0.1*0.0078125/0.1 = 1/128 → (2^30, -6).
        assert!(s.contains("input_dim:1i32"));
        assert!(s.contains("output_dim:1i32"));
        assert!(s.contains("FC_MULT_0:[i32;1usize]=[1073741824i32]"));
        assert!(s.contains("FC_SHIFT_0:[i32;1usize]=[-6i32]"));

        // Dispatch through KernelBackend, slices into the typed I/O arrays.
        assert!(s.contains("backend.fully_connected("));
        assert!(s.contains("&input[0usize..1usize]"));
        assert!(s.contains("&mutoutput[0usize..1usize]"));

        // No unsafe or heap in the generated code.
        assert!(!s.contains("unsafe"), "generated code must be unsafe-free");
        assert!(!s.contains("Vec<"), "generated code must not allocate");
    }

    #[test]
    fn unsupported_op_reports_t5_gate() {
        let bytes = minimal_model_bytes(200); // extended/custom opcode
        let model = flatbuffer::parse(&bytes).expect("minimal model parses");
        let err = emit_model(&model).expect_err("unsupported op must error");
        assert!(err.contains("builtin_code=200"), "error must name the op: {err}");
        assert!(err.contains("T5"), "error must point at the T5 gate: {err}");
    }

    #[test]
    fn malformed_op_input_errors_not_panics() {
        // FULLY_CONNECTED (9) on a model whose operator has no options table
        // and lists no weights input — the emitter must error, not panic.
        let bytes = minimal_model_bytes(9);
        let model = flatbuffer::parse(&bytes).expect("minimal model parses");
        let err = emit_model(&model).expect_err("malformed FC must error");
        assert!(err.contains("op 0"), "expected per-op error: {err}");
    }

    #[test]
    fn standalone_relu_emits_activation_params() {
        let bytes = minimal_model_bytes(19); // RELU
        let model = flatbuffer::parse(&bytes).expect("minimal model parses");
        let ts = emit_model(&model).expect("relu emits");
        let s: String = ts.to_string().chars().filter(|c| !c.is_whitespace()).collect();
        assert!(s.contains("backend.relu("));
        assert!(s.contains("ACTIVATION_PARAMS_0"));
        assert!(s.contains("quantized_activation_min:0i32"), "relu clamps at output zp: {s}");
        assert!(s.contains("quantized_activation_max:127i32"));
    }

    /// Assemble a minimal valid TFLite model by hand: 2 int8 tensors
    /// (scale 1.0, zp 0), one operator of `op_code` fed by tensor 0 → 1.
    /// Byte positions are hard offsets into the returned buffer.
    fn minimal_model_bytes(op_code: i32) -> Vec<u8> {
        let mut b = Vec::with_capacity(432);
        fn u16v(b: &mut Vec<u8>, v: u16) {
            b.extend_from_slice(&v.to_le_bytes());
        }
        fn u32v(b: &mut Vec<u8>, v: u32) {
            b.extend_from_slice(&v.to_le_bytes());
        }
        fn i32v(b: &mut Vec<u8>, v: i32) {
            b.extend_from_slice(&v.to_le_bytes());
        }
        fn i64v(b: &mut Vec<u8>, v: i64) {
            b.extend_from_slice(&v.to_le_bytes());
        }

        // 0..8: root uoffset (Model table at 24) + TFL3 identifier.
        u32v(&mut b, 24);
        b.extend_from_slice(b"TFL3");
        // 8..24: Model vtable — 8 slots (16 bytes) so the Model table starts
        // at exactly 24, matching every hard-coded offset below.
        u16v(&mut b, 16); // vtable_len
        u16v(&mut b, 16); // table_size
        u16v(&mut b, 0); // f0 version absent
        u16v(&mut b, 4); // f1 operator_codes at table+4
        u16v(&mut b, 8); // f2 subgraphs at table+8
        u16v(&mut b, 0); // f3 description absent
        u16v(&mut b, 12); // f4 buffers at table+12
        u16v(&mut b, 0); // f5 padding slot
        // 24..40: Model table — uoffsets relative to their field positions.
        i32v(&mut b, 16); // SOffsetT → vtable at 8
        u32v(&mut b, 12); // f1 → opcodes vector at 40
        u32v(&mut b, 40); // f2 → subgraphs vector at 72
        u32v(&mut b, 328); // f4 → buffers vector at 364
        // 40..48: operator_codes vector.
        u32v(&mut b, 1); // len
        u32v(&mut b, 16); // → OperatorCode table at 60
        // 48..60: OperatorCode vtable.
        u16v(&mut b, 12);
        u16v(&mut b, 12);
        u16v(&mut b, 0);
        u16v(&mut b, 0);
        u16v(&mut b, 0);
        u16v(&mut b, 8); // f3 builtin_code at table+8
        // 60..72: OperatorCode table (extended builtin_code path).
        i32v(&mut b, 12);
        u32v(&mut b, 0); // padding to f3
        i32v(&mut b, op_code);
        // 72..80: subgraphs vector.
        u32v(&mut b, 1);
        u32v(&mut b, 20); // → Subgraph table at 96
        // 80..96: Subgraph vtable — 8 slots (16 bytes) so the Subgraph table
        // starts at exactly 96.
        u16v(&mut b, 16);
        u16v(&mut b, 20);
        u16v(&mut b, 4); // f0 tensors
        u16v(&mut b, 8); // f1 inputs
        u16v(&mut b, 12); // f2 outputs
        u16v(&mut b, 16); // f3 operators
        u16v(&mut b, 0); // f4 padding slot
        u16v(&mut b, 0); // f5 padding slot
        // 96..116: Subgraph table.
        i32v(&mut b, 16);
        u32v(&mut b, 16); // f0 → tensors vector at 116
        u32v(&mut b, 288); // f1 → inputs vector at 392
        u32v(&mut b, 292); // f2 → outputs vector at 400
        u32v(&mut b, 200); // f3 → operators vector at 312
        // 116..128: tensors vector.
        u32v(&mut b, 2);
        u32v(&mut b, 24); // → tensor0 table at 144
        u32v(&mut b, 112); // → tensor1 table at 236
        // 128..144: tensor0 vtable (fields 0, 1, 2, 4).
        u16v(&mut b, 16);
        u16v(&mut b, 20);
        u16v(&mut b, 4); // f0 shape
        u16v(&mut b, 8); // f1 type
        u16v(&mut b, 12); // f2 buffer
        u16v(&mut b, 0); // f3 name absent
        u16v(&mut b, 16); // f4 quantization
        u16v(&mut b, 0); // f5 unused
        // 144..164: tensor0 table.
        i32v(&mut b, 16);
        u32v(&mut b, 16); // f0 → shape vector at 164
        b.push(9); // f1 INT8
        b.extend_from_slice(&[0, 0, 0]);
        u32v(&mut b, 0); // f2 buffer 0 (empty sentinel)
        u32v(&mut b, 24); // f4 → quantization table at 184
        // 164..172: tensor0 shape vector.
        u32v(&mut b, 1);
        i32v(&mut b, 1);
        // 172..184: quantization vtable (fields 2, 3).
        u16v(&mut b, 12);
        u16v(&mut b, 16);
        u16v(&mut b, 0);
        u16v(&mut b, 0);
        u16v(&mut b, 8); // f2 scale at table+8
        u16v(&mut b, 12); // f3 zero_point at table+12
        // 184..200: quantization table.
        i32v(&mut b, 12);
        u32v(&mut b, 0); // padding
        u32v(&mut b, 8); // f2 → scale vector at 200
        u32v(&mut b, 12); // f3 → zero_point vector at 208
        // 200..208: scale vector.
        u32v(&mut b, 1);
        b.extend_from_slice(&1.0f32.to_le_bytes());
        // 208..216: zero_point vector.
        u32v(&mut b, 1);
        i64v(&mut b, 0);
        // 216..232: tensor1 vtable (mirror of tensor0).
        u16v(&mut b, 16);
        u16v(&mut b, 20);
        u16v(&mut b, 4);
        u16v(&mut b, 8);
        u16v(&mut b, 12);
        u16v(&mut b, 0);
        u16v(&mut b, 16);
        u16v(&mut b, 0);
        // 232..252: tensor1 table.
        i32v(&mut b, 16);
        u32v(&mut b, 16); // f0 → shape vector at 256
        b.push(9); // f1 INT8
        b.extend_from_slice(&[0, 0, 0]);
        u32v(&mut b, 0); // f2 buffer 0
        u32v(&mut b, 24); // f4 → quantization table at 276 (vtable precedes it)
        // 252..260: tensor1 shape vector.
        u32v(&mut b, 1);
        i32v(&mut b, 1);
        // 260..272: quantization vtable.
        u16v(&mut b, 12);
        u16v(&mut b, 16);
        u16v(&mut b, 0);
        u16v(&mut b, 0);
        u16v(&mut b, 8); // f2 scale at table+8
        u16v(&mut b, 12); // f3 zero_point at table+12
        // 272..288: quantization table.
        i32v(&mut b, 12);
        u32v(&mut b, 0);
        u32v(&mut b, 8); // → scale vector at 288
        u32v(&mut b, 12); // → zero_point vector at 296
        // 288..296: scale vector.
        u32v(&mut b, 1);
        b.extend_from_slice(&1.0f32.to_le_bytes());
        // 296..312: zero_point vector.
        u32v(&mut b, 1);
        i64v(&mut b, 0);
        // 312..320: operators vector.
        u32v(&mut b, 1);
        u32v(&mut b, 16); // → Operator table at 332
        // 320..332: Operator vtable — 6 slots (12 bytes) so the Operator
        // table starts at exactly 332.
        u16v(&mut b, 12);
        u16v(&mut b, 16);
        u16v(&mut b, 4); // f0 opcode_index
        u16v(&mut b, 8); // f1 inputs
        u16v(&mut b, 12); // f2 outputs
        u16v(&mut b, 0); // f3 padding slot
        // 332..348: Operator table.
        i32v(&mut b, 12);
        u32v(&mut b, 0); // f0 opcode_index 0
        u32v(&mut b, 8); // f1 → inputs vector at 348
        u32v(&mut b, 12); // f2 → outputs vector at 356
        // 348..356: operator inputs vector.
        u32v(&mut b, 1);
        u32v(&mut b, 0);
        // 356..364: operator outputs vector.
        u32v(&mut b, 1);
        u32v(&mut b, 1);
        // 364..376: buffers vector — buf0/buf1 tables at 380/388 (4-byte
        // empty vtables precede them).
        u32v(&mut b, 2);
        u32v(&mut b, 12); // → buffer0 table at 380
        u32v(&mut b, 16); // → buffer1 table at 388
        // 376..380: buffer0 vtable (empty table).
        u16v(&mut b, 4);
        u16v(&mut b, 8);
        // 380..384: buffer0 table.
        i32v(&mut b, 4);
        // 384..388: buffer1 vtable.
        u16v(&mut b, 4);
        u16v(&mut b, 8);
        // 388..392: buffer1 table.
        i32v(&mut b, 4);
        // 392..400: subgraph inputs vector.
        u32v(&mut b, 1);
        u32v(&mut b, 0);
        // 400..408: subgraph outputs vector.
        u32v(&mut b, 1);
        u32v(&mut b, 1);
        b
    }

    /// Cross-crate scratch parity (T3.5 / T3.5b / T1.4 gate): the macro-time
    /// mirror `depthwise_scratch_need_codegen` must equal the runtime
    /// `S3Backend::depthwise_conv2d_scratch_size` for the dm>1 fan-out shapes,
    /// the dm==1 corpus, AND arbitrary filter sizes (3×3, 5×5, 7×7, kws 10×8)
    /// — the runtime formula is canonical; a mismatch is a mirror bug.
    #[test]
    fn depthwise_scratch_mirror_matches_s3_backend() {
        use hematite_core::op_params::{DepthwiseConv2DParams, Padding};
        use hematite_core::KernelBackend;
        use hematite_s3::backend::S3Backend;

        fn params(
            dm: i32,
            in_c: i32,
            spatial: i32,
            input_offset: i32,
            fh: i32,
            fw: i32,
            stride: i32,
        ) -> DepthwiseConv2DParams<'static> {
            let out_c = in_c * dm;
            DepthwiseConv2DParams {
                input_shape: [1, spatial, spatial, in_c],
                filter_shape: [1, fh, fw, out_c],
                output_shape: [1, spatial, spatial, out_c],
                padding: Padding::Same,
                stride_width: stride,
                stride_height: stride,
                dilation_width_factor: 1,
                dilation_height_factor: 1,
                depth_multiplier: dm,
                input_offset,
                weights_offset: 0,
                output_offset: 0,
                // The scratch-size formula reads shapes/offsets only — slice
                // contents (and length) are irrelevant to `depthwise_scratch_need`.
                output_multiplier_per_channel: &[],
                output_shift_per_channel: &[],
                quantized_activation_min: -128,
                quantized_activation_max: 127,
            }
        }

        fn codegen_need(p: &DepthwiseConv2DParams<'_>) -> usize {
            depthwise_scratch_need_codegen(
                p.input_shape[1].max(0) as usize,
                p.input_shape[2].max(0) as usize,
                p.input_shape[3].max(0) as usize,
                p.output_shape[1].max(0) as usize,
                p.output_shape[2].max(0) as usize,
                p.output_shape[3].max(0) as usize,
                p.filter_shape[1],
                p.filter_shape[2],
                p.stride_height,
                p.stride_width,
                p.dilation_height_factor,
                p.dilation_width_factor,
                p.depth_multiplier,
                p.input_offset,
            )
        }

        let mut checked = 0;
        for &dm in &[1, 2, 4, 8] {
            for &in_c in &[1, 3, 8, 16, 32] {
                for &spatial in &[8, 12, 14] {
                    for &offset in &[0, -3] {
                        let p = params(dm, in_c, spatial, offset, 3, 3, 1);
                        let mirror = codegen_need(&p);
                        let runtime = S3Backend::depthwise_conv2d_scratch_size(&p);
                        assert_eq!(
                            mirror, runtime,
                            "depthwise dm={dm} in_c={in_c} spatial={spatial} offset={offset}: \
                             codegen mirror {mirror} != S3Backend need {runtime}"
                        );
                        checked += 1;
                    }
                }
            }
        }
        // T3.5b — arbitrary filters (5×5, 7×7, kws 10×8) × dm {1,8} ×
        // strides {1,2}, with and without input_offset.
        for &(fh, fw) in &[(5, 5), (7, 7), (10, 8)] {
            for &dm in &[1, 8] {
                for &in_c in &[1, 8] {
                    for &stride in &[1, 2] {
                        for &offset in &[0, 3, 128] {
                            let p = params(dm, in_c, 14, offset, fh, fw, stride);
                            let mirror = codegen_need(&p);
                            let runtime = S3Backend::depthwise_conv2d_scratch_size(&p);
                            assert_eq!(
                                mirror, runtime,
                                "depthwise fh={fh} fw={fw} dm={dm} in_c={in_c} stride={stride} \
                                 offset={offset}: codegen mirror {mirror} != S3Backend need {runtime}"
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(checked, 192, "parity matrix did not expand ({checked})");
    }

    /// Cross-crate scratch parity (T3.6): the macro-time mirror
    /// `fc_scratch_need_codegen` must equal the runtime
    /// `hematite_s3::backend::fc_scratch_need` for every small / non-16 FC
    /// shape (pad and no-pad paths). The runtime formula is canonical — a
    /// mismatch is a mirror bug.
    #[test]
    fn fc_scratch_mirror_matches_s3_backend() {
        use hematite_core::op_params::FullyConnectedParams;

        fn params(
            input_dim: i32,
            output_dim: i32,
            input_offset: i32,
        ) -> FullyConnectedParams<'static> {
            FullyConnectedParams {
                input_dim,
                output_dim,
                input_offset,
                weights_offset: 0,
                output_offset: 0,
                // The scratch-size formula reads input/output dims + offset
                // only — slice contents are irrelevant to `fc_scratch_need`.
                output_multiplier_per_channel: &[],
                output_shift_per_channel: &[],
                quantized_activation_min: -128,
                quantized_activation_max: 127,
            }
        }

        let mut checked = 0;
        for &input_dim in &[1, 3, 8, 15, 16, 17, 32, 128, 640] {
            for &output_dim in &[1, 3, 8, 16, 128] {
                for &offset in &[0, 5, 128] {
                    let p = params(input_dim, output_dim, offset);
                    let mirror = fc_scratch_need_codegen(
                        input_dim.max(0) as usize,
                        output_dim.max(0) as usize,
                        p.input_offset,
                    );
                    let runtime = hematite_s3::backend::fc_scratch_need(&p);
                    assert_eq!(
                        mirror, runtime,
                        "fc input_dim={input_dim} out_dim={output_dim} offset={offset}: \
                         codegen mirror {mirror} != S3Backend need {runtime}"
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked >= 100, "fc parity matrix did not expand");
    }
}
