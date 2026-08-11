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
//!   ops of the documented per-op scratch need (every `*_scratch_size` in the
//!   current kernel set defaults to `0` → `SCRATCH_LEN = 0`).  Generated code
//!   sizes the scratch array with this const — never `[0u8; B::scratch()]`
//!   inside a `const fn` (unstable const-trait-call trap).

use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use crate::flatbuffer::{ParsedModel, ParsedOp, ParsedOptions, ParsedTensor, QuantInfo, TensorType};

/// Emit the full model wrapper for `subgraph[0]` of a parsed model.
///
/// The op list is read from `model.ops()` (execution order); a later wiring
/// task (T4.2a fusion) can pass a preprocessed schedule by refactoring the
/// single `let ops = model.ops();` line into a parameter.
pub(crate) fn emit_model(model: &ParsedModel) -> Result<TokenStream, String> {
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

    // ── Emit per-op consts + calls (straight-line, execution order) ────────
    // Two passes over the same op list: the stack path (per-tensor struct
    // arrays) and the arena path (tensors carved from a caller arena at
    // liveness offsets).  Consts are emitted once (identical for both).
    let mut consts: Vec<TokenStream> = Vec::new();
    let mut stack_calls: Vec<TokenStream> = Vec::new();
    let mut arena_calls: Vec<TokenStream> = Vec::new();
    let mut arena_scratch_checks: Vec<TokenStream> = Vec::new();
    let mut scratch_max = 0usize;
    for (i, op) in ops.iter().enumerate() {
        let em_stack = emit_op(model, &storage, i, op, TensorMode::Stack)?;
        consts.extend(em_stack.consts);
        stack_calls.push(em_stack.call);
        let em_arena = emit_op(model, &storage, i, op, TensorMode::Arena)?;
        arena_calls.push(em_arena.call);
        arena_scratch_checks.push(em_arena.scratch_check);
        scratch_max = scratch_max.max(em_stack.scratch.max(em_arena.scratch));
    }

    // ── Intermediate storage: stack types + arena layout ───────────────────
    let mut tensor_types: Vec<TokenStream> = Vec::new();
    let mut tensor_locals: Vec<TokenStream> = Vec::new();
    let mut arena_locals: Vec<TokenStream> = Vec::new();
    let mut arena_offsets: Vec<usize> = vec![0usize; tensors.len()];
    let mut arena_len = 0usize;
    let mut arena_plan_ok = false;
    for s in &storage {
        if let Storage::Tensor { idx } = s {
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
    if let Ok(plan) = crate::optimize::arena::plan_arena_internal(model, usize::MAX / 4, None) {
        let all_slotted = storage.iter().all(|s| match s {
            Storage::Tensor { idx } => plan.offsets[*idx] != hematite_memory::OFFSET_NONE,
            _ => true,
        });
        if all_slotted {
            for s in &storage {
                if let Storage::Tensor { idx } = s {
                    arena_offsets[*idx] = plan.offsets[*idx];
                }
            }
            arena_len = plan.peak_arena_bytes;
            arena_plan_ok = true;
        }
    }
    if !arena_plan_ok {
        // Sequential 16-aligned offsets — correct without liveness coalescing
        // (larger arena, but no shared slots).  Used when the planner rejects
        // the schedule (e.g. > MAX_TENSORS intermediates) or a tensor has no
        // arena slot.
        let mut cursor = 0usize;
        for s in &storage {
            if let Storage::Tensor { idx } = s {
                let len = flat_len(&tensors[*idx].shape)?;
                cursor = (cursor + 15) & !15;
                arena_offsets[*idx] = cursor;
                cursor += (len + 15) & !15;
            }
        }
        arena_len = cursor;
    }
    for s in &storage {
        if let Storage::Tensor { idx } = s {
            let var = Ident::new(&format!("tensor_{idx}"), proc_macro2::Span::call_site());
            let off = arena_offsets[*idx];
            let len = flat_len(&tensors[*idx].shape)?;
            arena_locals.push(quote! {
                // SAFETY: `arena` is a single caller-owned buffer of at least
                // `ARENA_LEN` bytes; `#off..#off+#len` lies wholly inside it
                // (offsets are 16-aligned, within the plan's peak footprint).
                let #var: &mut [i8] = unsafe {
                    core::slice::from_raw_parts_mut(arena.as_mut_ptr().add(#off), #len)
                };
            });
        }
    }

    let input_len_ts = input_len;
    let output_len_ts = output_len;
    let scratch_ts = scratch_max;
    let arena_len_ts = arena_len;
    let stack_bind = if stack_calls.is_empty() {
        quote! { let _ = &self.backend; }
    } else {
        quote! { let backend = &self.backend; }
    };
    let arena_bind = if arena_calls.is_empty() {
        quote! { let _ = &self.backend; }
    } else {
        quote! { let backend = &self.backend; }
    };

    let wrapper = quote! {
        /// Generated inference model — typed I/O bridge for `subgraph[0]`.
        ///
        /// `B` is any [`::hematite_core::KernelBackend`] implementation; the
        /// straight-line op sequence dispatches through it.
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

        impl<B: ::hematite_core::KernelBackend> Model<B> {
            /// Run inference with an internally allocated scratch buffer.
            ///
            /// No panic paths: on error the output array is left zeroed.
            pub fn predict(&self, input: &[i8; INPUT_LEN]) -> [i8; OUTPUT_LEN] {
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
            pub fn predict_with_scratch(
                &self,
                input: &[i8; INPUT_LEN],
                output: &mut [i8; OUTPUT_LEN],
                scratch: &mut [u8],
            ) -> Result<(), ::hematite_core::KernelError> {
                if scratch.len() < SCRATCH_LEN {
                    return Err(::hematite_core::KernelError::ScratchTooSmall);
                }
                #(#tensor_locals)*
                #stack_bind
                #(#stack_calls)*
                Ok(())
            }

            /// Run inference with intermediates carved from a caller-provided
            /// arena.
            ///
            /// Unlike [`predict_with_scratch`](Self::predict_with_scratch)
            /// (per-tensor stack arrays), every intermediate tensor is carved
            /// from `arena` at liveness-coalesced 16-aligned offsets, so a
            /// single `ARENA_LEN`-byte buffer (SRAM or PSRAM) replaces
            /// hundreds of kilobytes of stack.  `arena` must be at least
            /// [`ARENA_LEN`] bytes and 16-byte aligned.
            ///
            /// # Errors
            ///
            /// * [`::hematite_core::KernelError::ScratchTooSmall`] if
            ///   `arena.len() < ARENA_LEN` or `scratch.len() < SCRATCH_LEN`.
            pub fn predict_with_arena(
                &self,
                input: &[i8; INPUT_LEN],
                output: &mut [i8; OUTPUT_LEN],
                arena: &mut [i8],
                scratch: &mut [u8],
            ) -> Result<(), ::hematite_core::KernelError> {
                if arena.len() < ARENA_LEN {
                    return Err(::hematite_core::KernelError::ScratchTooSmall);
                }
                // Runtime scratch requirement from the backend's `*_scratch_size`
                // associated fns (constant params → const-foldable per op).  This
                // guarantees the backend's SIMD paths get enough scratch or the
                // call fails loudly instead of silently falling back to scalar.
                let mut need = 0usize;
                #(#arena_scratch_checks)*
                if scratch.len() < need {
                    return Err(::hematite_core::KernelError::ScratchTooSmall);
                }
                #(#arena_locals)*
                #arena_bind
                #(#arena_calls)*
                Ok(())
            }
        }
    };

    Ok(quote! {
        pub const INPUT_LEN: usize = #input_len_ts;
        pub const OUTPUT_LEN: usize = #output_len_ts;
        pub const SCRATCH_LEN: usize = #scratch_ts;
        pub const ARENA_LEN: usize = #arena_len_ts;
        /// 16-byte-aligned wrapper for weight statics — SIMD kernels gate on
        /// `weights.as_ptr() % 16 == 0`; bare `const [i8; N]` arrays are only
        /// 1-byte aligned in flash.
        #[repr(align(16))]
        #[allow(dead_code)]
        struct WeightAlign<const N: usize>([i8; N]);
        #(#consts)*
        #(#tensor_types)*
        #wrapper
    })
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

/// How intermediate tensors are allocated in the generated code.
#[derive(Clone, Copy)]
enum TensorMode {
    /// Per-tensor `#[repr(C, align(16))]` stack arrays (`tensor_N.data`).
    Stack,
    /// Slices carved from a caller-provided `arena: &mut [i8]` at liveness
    /// offsets (`tensor_N: &mut [i8]`, `&*tensor_N` / `tensor_N`).
    Arena,
}

/// Expression naming a tensor as a kernel *input* (immutable `&[i8]`).
fn src_expr(storage: &[Storage], t: usize, mode: TensorMode) -> Result<TokenStream, String> {
    match storage.get(t).ok_or_else(|| format!("tensor index {t} out of range"))? {
        Storage::Input { start, len } => {
            let end = start + len;
            Ok(quote!(&input[#start..#end]))
        }
        Storage::Tensor { idx } => {
            let var = Ident::new(&format!("tensor_{idx}"), proc_macro2::Span::call_site());
            match mode {
                TensorMode::Stack => Ok(quote!(&#var.data)),
                TensorMode::Arena => Ok(quote!(&*#var)),
            }
        }
        Storage::Const => Err(format!(
            "tensor {t} is a constant (buffer-backed) but is used as a runtime op input; \
             constant data inputs are not supported in T4.1 (deferred to T4.2a fusion)"
        )),
        Storage::Output { .. } => Err(format!("tensor {t} is a model output used as an op input")),
    }
}

/// Expression naming a tensor as a kernel *output* (`&mut [i8]`).
fn dst_expr(storage: &[Storage], t: usize, mode: TensorMode) -> Result<TokenStream, String> {
    match storage.get(t).ok_or_else(|| format!("tensor index {t} out of range"))? {
        Storage::Output { start, len } => {
            let end = start + len;
            Ok(quote!(&mut output[#start..#end]))
        }
        Storage::Tensor { idx } => {
            let var = Ident::new(&format!("tensor_{idx}"), proc_macro2::Span::call_site());
            match mode {
                TensorMode::Stack => Ok(quote!(&mut #var.data)),
                TensorMode::Arena => Ok(quote!(#var)),
            }
        }
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
    let vals = i8_lit(bytes);
    // `static` (not `const`) wrapped in a `#[repr(align(16))]` struct so the
    // symbol is 16-byte aligned in flash: the SIMD kernels gate on
    // `weights.as_ptr() % 16 == 0`, and a bare `const [i8; N]` array has
    // alignment 1. `&#name.0` points at field offset 0 → aligned.
    Ok(quote!(static #name: WeightAlign<#len> = WeightAlign(#vals);))
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
    /// Documented scratch need for this op (kernel `*_scratch_size` default 0).
    scratch: usize,
    /// Statement advancing a `need` accumulator to the backend's runtime
    /// scratch requirement for this op's params (only the arena path emits
    /// these — see [`emit_model`]).
    scratch_check: TokenStream,
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

fn emit_op(
    model: &ParsedModel,
    storage: &[Storage],
    i: usize,
    op: &ParsedOp,
    mode: TensorMode,
) -> Result<OpEmission, String> {
    match op.builtin_code {
        0 => emit_elementwise(model, storage, i, op, quote!(add), ElementwiseKind::AddSub, mode),
        1 => emit_pool(model, storage, i, op, quote!(average_pool_2d), mode),
        3 => emit_conv2d(model, storage, i, op, mode),
        4 => emit_depthwise(model, storage, i, op, mode),
        9 => emit_fully_connected(model, storage, i, op, mode),
        17 => emit_pool(model, storage, i, op, quote!(max_pool_2d), mode),
        18 => emit_elementwise(model, storage, i, op, quote!(mul), ElementwiseKind::Mul, mode),
        19 => emit_activation(model, storage, i, op, quote!(relu), 1, mode),
        21 => emit_activation(model, storage, i, op, quote!(relu6), 3, mode),
        22 => emit_reshape(model, storage, i, op, mode),
        25 => emit_softmax(model, storage, i, op, mode),
        34 => emit_pad(model, storage, i, op, mode),
        39 => emit_transpose(model, storage, i, op, mode),
        40 => emit_mean(model, storage, i, op, mode),
        41 => emit_elementwise(model, storage, i, op, quote!(sub), ElementwiseKind::AddSub, mode),
        54 => emit_prelu(model, storage, i, op, mode),
        97 => emit_resize(model, storage, i, op, mode),
        98 => emit_leaky_relu(model, storage, i, op, mode),
        117 => emit_activation(model, storage, i, op, quote!(hard_swish), 0, mode),
        code => Err(format!(
            "unsupported operator (builtin_code={code}) in subgraph[0]; T4.1 dispatches only the \
             in-scope op set (conv2d, depthwise_conv2d, fully_connected, average_pool_2d, \
             max_pool_2d, softmax, relu, relu6, hard_swish, leaky_relu, prelu, add, sub, mul, \
             mean, reshape, resize_nearest); this opcode is gated behind the T5 feature wave \
             (see local-notes/plans/hematite-nn.md, T5: extended op support)"
        )),
    }
}

fn emit_conv2d(
    model: &ParsedModel,
    storage: &[Storage],
    i: usize,
    op: &ParsedOp,
    mode: TensorMode,
) -> Result<OpEmission, String> {
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
    let input_shape = arr4(shape4(&input.shape)?);
    let filter_raw = shape4(&weights.shape)?;
    let filter_shape = arr4(filter_raw);
    let output_shape = arr4(shape4(&output.shape)?);
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
    let src = src_expr(storage, in_t as usize, mode)?;
    let dst = dst_expr(storage, out_t as usize, mode)?;
    let call = quote! {
        backend.conv2d(#src, &(#w_name.0), &#b_name, &#p_name, #dst, scratch)?;
    };
    Ok(OpEmission {
        consts: vec![weights_c, bias_c, mult_c, shift_c, params_c],
        call,
        scratch: 0,
        scratch_check: quote!(need = need.max(B::conv2d_scratch_size(&#p_name));),
    })
}

fn emit_depthwise(
    model: &ParsedModel,
    storage: &[Storage],
    i: usize,
    op: &ParsedOp,
    mode: TensorMode,
) -> Result<OpEmission, String> {
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
    let input_shape = arr4(shape4(&input.shape)?);
    let filter_raw = shape4(&weights.shape)?;
    let filter_shape = arr4(filter_raw);
    let output_shape = arr4(shape4(&output.shape)?);
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
    let src = src_expr(storage, in_t as usize, mode)?;
    let dst = dst_expr(storage, out_t as usize, mode)?;
    let call = quote! {
        backend.depthwise_conv2d(#src, &(#w_name.0), &#b_name, &#p_name, #dst, scratch)?;
    };
    Ok(OpEmission {
        consts: vec![weights_c, bias_c, mult_c, shift_c, params_c],
        call,
        scratch: 0,
        scratch_check: quote!(need = need.max(B::depthwise_conv2d_scratch_size(&#p_name));),
    })
}

fn emit_fully_connected(
    model: &ParsedModel,
    storage: &[Storage],
    i: usize,
    op: &ParsedOp,
    mode: TensorMode,
) -> Result<OpEmission, String> {
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
    let src = src_expr(storage, in_t as usize, mode)?;
    let dst = dst_expr(storage, out_t as usize, mode)?;
    let call = quote! {
        backend.fully_connected(#src, &(#w_name.0), &#b_name, &#p_name, #dst, scratch)?;
    };
    let _ = batches;
    Ok(OpEmission {
        consts: vec![weights_c, bias_c, mult_c, shift_c, params_c],
        call,
        scratch: 0,
        scratch_check: TokenStream::new(),
    })
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
    i: usize,
    op: &ParsedOp,
    method: TokenStream,
    mode: TensorMode,
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
    let src = src_expr(storage, in_t as usize, mode)?;
    let dst = dst_expr(storage, out_t as usize, mode)?;
    let call = quote! {
        backend.#method(#src, &#p_name, #dst)?;
    };
    Ok(OpEmission {
        consts: vec![params_c],
        call,
        scratch: 0,
        scratch_check: TokenStream::new(),
    })
}

fn emit_softmax(
    model: &ParsedModel,
    storage: &[Storage],
    i: usize,
    op: &ParsedOp,
    mode: TensorMode,
) -> Result<OpEmission, String> {
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
    let row_size = shape[3] as i32;
    let num_rows = (flat_len(&input.shape)? as i32) / row_size;
    let in_scale = tensor_scale(input)?;
    let in_zp = tensor_zp(input)?;
    let (m, s) = quantize_multiplier(in_scale);
    // Q5.26 logit scaling: sadhg(diff, m) << (input_left_shift + 1) must equal
    // round(diff * in_scale * 2^26) → input_left_shift = 25 + s.
    let input_left_shift = 25 + s;

    let p_name = Ident::new(&format!("SOFTMAX_PARAMS_{i}"), proc_macro2::Span::call_site());
    let params_c = quote! {
        const #p_name: ::hematite_core::op_params::SoftmaxParams =
            ::hematite_core::op_params::SoftmaxParams {
                num_rows: #num_rows,
                row_size: #row_size,
                input_multiplier: #m,
                input_left_shift: #input_left_shift,
                diff_min: -128,
                input_offset: #in_zp,
                output_offset: -128,
                quantized_activation_min: -128,
                quantized_activation_max: 127,
            };
    };
    let src = src_expr(storage, in_t as usize, mode)?;
    let dst = dst_expr(storage, out_t as usize, mode)?;
    let call = quote! {
        backend.softmax(#src, &#p_name, #dst, scratch)?;
    };
    let _ = output;
    Ok(OpEmission {
        consts: vec![params_c],
        call,
        scratch: 0,
        scratch_check: quote!(need = need.max(B::softmax_scratch_size(&#p_name));),
    })
}

/// Standalone RELU (fused 1), RELU6 (fused 3), HARD_SWISH (fused NONE).
fn emit_activation(
    model: &ParsedModel,
    storage: &[Storage],
    i: usize,
    op: &ParsedOp,
    method: TokenStream,
    fused: i8,
    mode: TensorMode,
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
    let src = src_expr(storage, in_t as usize, mode)?;
    let dst = dst_expr(storage, out_t as usize, mode)?;
    let call = quote! {
        backend.#method(#src, &#p_name, #dst)?;
    };
    Ok(OpEmission {
        consts: vec![params_c],
        call,
        scratch: 0,
        scratch_check: TokenStream::new(),
    })
}

fn emit_leaky_relu(
    model: &ParsedModel,
    storage: &[Storage],
    i: usize,
    op: &ParsedOp,
    mode: TensorMode,
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
    let src = src_expr(storage, in_t as usize, mode)?;
    let dst = dst_expr(storage, out_t as usize, mode)?;
    let call = quote! {
        backend.leaky_relu(#src, &#p_name, #dst)?;
    };
    Ok(OpEmission {
        consts: vec![params_c],
        call,
        scratch: 0,
        scratch_check: TokenStream::new(),
    })
}

fn emit_prelu(
    model: &ParsedModel,
    storage: &[Storage],
    i: usize,
    op: &ParsedOp,
    mode: TensorMode,
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
                alpha_data: &(#a_name.0),
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
    let src = src_expr(storage, in_t as usize, mode)?;
    let dst = dst_expr(storage, out_t as usize, mode)?;
    let call = quote! {
        backend.prelu(#src, &#p_name, #dst)?;
    };
    Ok(OpEmission {
        consts: vec![alpha_c, params_c],
        call,
        scratch: 0,
        scratch_check: TokenStream::new(),
    })
}

#[derive(Clone, Copy, PartialEq)]
enum ElementwiseKind {
    AddSub,
    Mul,
}

fn emit_elementwise(
    model: &ParsedModel,
    storage: &[Storage],
    i: usize,
    op: &ParsedOp,
    method: TokenStream,
    kind: ElementwiseKind,
    mode: TensorMode,
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
    let in1_scale = tensor_scale(input1)?;
    let in2_scale = tensor_scale(input2)?;
    let out_scale = tensor_scale(output)?;
    let in1_off = -tensor_zp(input1)?;
    let in2_off = -tensor_zp(input2)?;
    let out_off = tensor_zp(output)?;
    let (act_min, act_max) = act_range(fused_activation, out_scale, out_off);

    let (left_shift, i1m, i1s, i2m, i2s, om, os) = match kind {
        ElementwiseKind::AddSub => {
            // TFLM int8 Add/Sub (add_common.cc): twice_max scaling with
            // left_shift = 20; output ratio carries 2^left_shift.
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

    let p_name = Ident::new(&format!("ELEMENTWISE_PARAMS_{i}"), proc_macro2::Span::call_site());
    let params_c = quote! {
        const #p_name: ::hematite_core::op_params::ElementwiseParams =
            ::hematite_core::op_params::ElementwiseParams {
                num_elements: #num_elements,
                input1_offset: #in1_off,
                input2_offset: #in2_off,
                output_offset: #out_off,
                output_multiplier: #om,
                output_shift: #os,
                left_shift: #left_shift,
                input1_multiplier: #i1m,
                input1_shift: #i1s,
                input2_multiplier: #i2m,
                input2_shift: #i2s,
                quantized_activation_min: #act_min,
                quantized_activation_max: #act_max,
            };
    };
    let src1 = src_expr(storage, in1_t as usize, mode)?;
    let src2 = src_expr(storage, in2_t as usize, mode)?;
    let dst = dst_expr(storage, out_t as usize, mode)?;
    let call = quote! {
        backend.#method(#src1, #src2, &#p_name, #dst)?;
    };
    Ok(OpEmission {
        consts: vec![params_c],
        call,
        scratch: 0,
        scratch_check: TokenStream::new(),
    })
}

fn emit_mean(
    model: &ParsedModel,
    storage: &[Storage],
    i: usize,
    op: &ParsedOp,
    mode: TensorMode,
) -> Result<OpEmission, String> {
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
    let src = src_expr(storage, in_t as usize, mode)?;
    let dst = dst_expr(storage, out_t as usize, mode)?;
    let call = quote! {
        backend.mean(#src, &#p_name, #dst)?;
    };
    Ok(OpEmission {
        consts: vec![params_c],
        call,
        scratch: 0,
        scratch_check: TokenStream::new(),
    })
}

fn emit_reshape(
    model: &ParsedModel,
    storage: &[Storage],
    i: usize,
    op: &ParsedOp,
    mode: TensorMode,
) -> Result<OpEmission, String> {
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
    let src = src_expr(storage, in_t as usize, mode)?;
    let dst = dst_expr(storage, out_t as usize, mode)?;
    let call = quote! {
        backend.reshape(#src, &#p_name, #dst)?;
    };
    Ok(OpEmission {
        consts: vec![params_c],
        call,
        scratch: 0,
        scratch_check: TokenStream::new(),
    })
}

/// PAD — the padding amounts come from a const int32 tensor of shape
/// `[rank, 2]` (`[before, after]` per dim); the kernel pads with value 0.
fn emit_pad(
    model: &ParsedModel,
    storage: &[Storage],
    i: usize,
    op: &ParsedOp,
    mode: TensorMode,
) -> Result<OpEmission, String> {
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
    let src = src_expr(storage, in_t as usize, mode)?;
    let dst = dst_expr(storage, out_t as usize, mode)?;
    let call = quote! {
        backend.pad(#src, &#p_name, #dst)?;
    };
    Ok(OpEmission {
        consts: vec![params_c],
        call,
        scratch: 0,
        scratch_check: TokenStream::new(),
    })
}

/// TRANSPOSE — the permutation comes from a const int32 tensor of length
/// `rank`; the kernel computes the output shape from the permuted input.
fn emit_transpose(
    model: &ParsedModel,
    storage: &[Storage],
    i: usize,
    op: &ParsedOp,
    mode: TensorMode,
) -> Result<OpEmission, String> {
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
    let src = src_expr(storage, in_t as usize, mode)?;
    let dst = dst_expr(storage, out_t as usize, mode)?;
    let call = quote! {
        backend.transpose(#src, &#p_name, #dst)?;
    };
    Ok(OpEmission {
        consts: vec![params_c],
        call,
        scratch: 0,
        scratch_check: TokenStream::new(),
    })
}

fn emit_resize(
    model: &ParsedModel,
    storage: &[Storage],
    i: usize,
    op: &ParsedOp,
    mode: TensorMode,
) -> Result<OpEmission, String> {    let (align_corners, half_pixel_centers) = match op.options.as_ref() {
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
    let src = src_expr(storage, in_t as usize, mode)?;
    let dst = dst_expr(storage, out_t as usize, mode)?;
    let call = quote! {
        backend.resize_nearest(#src, &#p_name, #dst)?;
    };
    Ok(OpEmission {
        consts: vec![params_c],
        call,
        scratch: 0,
        scratch_check: TokenStream::new(),
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

    #[test]
    fn sine_model_emits_fc_call_sequence() {
        let model = flatbuffer::parse(SINE_TFLITE).expect("sine parses");
        let ts = emit_model(&model).expect("sine emits");
        // `to_string()` inserts spaces between tokens — compare whitespace-free.
        let s: String = ts.to_string().chars().filter(|c| !c.is_whitespace()).collect();

        // Model<I, O> wrapper with the typed I/O bridge.
        assert!(s.contains("structModel<B>"), "Model wrapper missing");
        assert!(s.contains("fnpredict_with_scratch"), "predict_with_scratch missing");
        assert!(s.contains("fnpredict_with_arena"), "predict_with_arena missing");
        assert!(s.contains("fnpredict("), "predict missing");
        assert!(s.contains("constfnnew("), "new missing");
        assert!(s.contains("constfninput_len"), "input_len missing");
        assert!(s.contains("constfnoutput_len"), "output_len missing");

        // I/O sized from tensor shapes (input [1], output [1]).
        assert!(s.contains("pubconstINPUT_LEN:usize=1usize"));
        assert!(s.contains("pubconstOUTPUT_LEN:usize=1usize"));

        // Scratch computed at macro time — kernels' *_scratch_size default 0.
        assert!(s.contains("pubconstSCRATCH_LEN:usize=0usize"));

        // Arena — sine has no intermediates (fc reads I/O + consts directly),
        // so the arena footprint is 0 and no carve locals are emitted.
        assert!(s.contains("pubconstARENA_LEN:usize=0usize"));

        // Weight/bias consts from buffer bytes.
        assert!(s.contains("staticWEIGHTS_0:WeightAlign<1usize>=WeightAlign([51i8])"), "weight const wrong: {s}");
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
}
