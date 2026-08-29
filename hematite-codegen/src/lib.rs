// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! hematite-codegen — proc-macro: TFLite → straight-line backend-dispatched code.
//!
//! Applies `#[model("path.tflite")]` to an empty struct.  At compile time the
//! macro reads the flatbuffer, walks it with a hand-rolled byte-offset walker,
//! and emits inference code dispatched through the `KernelBackend` trait.
//!
//! ## Crate structure
//!
//! * [`model`] — the proc-macro attribute entry point (only public item).
//! * `flatbuffer` — hand-rolled TFLite flatbuffer parser and IR types
//!   (consumed by T4.1 generate, T4.2a fusion, T4.2b arena, T4.2c layout).
//!
//! Because this is a proc-macro crate, only the `#[proc_macro_attribute]`
//! function is publicly exported.  All parser infrastructure lives in
//! `pub(crate)` modules visible to sibling source files.

use proc_macro::TokenStream;

pub(crate) mod flatbuffer;
pub(crate) mod generate;
// T4.2a fusion now compiles (see optimize/fusion.rs) — optimize module
// restored into the shared build per the gating comment below.
pub(crate) mod optimize;
// T4.1 — host-side mirror of every s3 SIMD-eligibility gate (the selector
// and the W0 fused-profile consume it; parity-tested in-crate).
pub(crate) mod eligibility;

/// Parses the `#[model("path.tflite")]` attribute, reads and validates the
/// TFLite model at compile time, then emits the typed inference code for
/// `subgraph[0]` alongside the annotated item.
///
/// The emitted code honors the T4.2a fusion schedule (T1.2): composed
/// groups collapse to single `FusedKernelBackend` calls; T2 groups and
/// ordinary ops emit the straight-line per-op sequence.
#[proc_macro_attribute]
pub fn model(attr: TokenStream, item: TokenStream) -> TokenStream {
    let proc_item = proc_macro2::TokenStream::from(item);
    parse_and_emit_impl(attr, proc_item, true, true, true, false).into()
}

/// Test-support attribute: identical to [`model`], but emits the UNFUSED
/// per-op straight-line sequence (no fusion schedule) — the unfused arm of
/// the T1.2 fused-vs-unfused equivalence gate.
#[proc_macro_attribute]
pub fn model_unfused(attr: TokenStream, item: TokenStream) -> TokenStream {
    let proc_item = proc_macro2::TokenStream::from(item);
    parse_and_emit_impl(attr, proc_item, false, true, true, false).into()
}

/// Test-support attribute: identical to [`model`] (fused schedule honored)
/// but with the T1.3 liveness arena DISABLED — intermediates are per-tensor
/// stack arrays, the `stack` arm of the arena-vs-stack bit-exactness gate.
#[proc_macro_attribute]
pub fn model_stack(attr: TokenStream, item: TokenStream) -> TokenStream {
    let proc_item = proc_macro2::TokenStream::from(item);
    parse_and_emit_impl(attr, proc_item, true, false, true, false).into()
}

/// Test-support attribute: identical to [`model`] (fused schedule honored,
/// arena enabled) but with the T4.2 graph-input 16B staging DISABLED — the
/// unstaged arm of the staged-vs-unstaged bit-exactness gate
/// (`tests/staged_input.rs`).
#[proc_macro_attribute]
pub fn model_unstaged(attr: TokenStream, item: TokenStream) -> TokenStream {
    let proc_item = proc_macro2::TokenStream::from(item);
    parse_and_emit_impl(attr, proc_item, true, true, false, false).into()
}

/// Test-support attribute: identical to [`model`] (fused schedule honored,
/// arena enabled, staging honored) but with the T2
/// `requires_verification` gate FORCED OPEN — T2 groups (input folds,
/// requantize folds) are emitted composed whenever the T4.2 selector's
/// structural + mirror gates pass, exactly as the W5 flip would.
///
/// T5.1 uses this as the "prove the auto-unfuse path" arm: a T2 group
/// forced composed must FAIL the fused==unfused equivalence check when its
/// composed semantics genuinely diverge from per-op, and the harness then
/// re-emits that model's T2 groups per-op (the default `#[model]` selection
/// — never silently accepted).  TEST-ONLY: unreachable from plain `#[model]`
/// usage (which always passes `force_t2 = false`).
#[proc_macro_attribute]
pub fn model_force_t2(attr: TokenStream, item: TokenStream) -> TokenStream {
    let proc_item = proc_macro2::TokenStream::from(item);
    parse_and_emit_impl(attr, proc_item, true, true, true, true).into()
}

/// Read + parse the model and route through the emitter, all within one
/// scope so the parsed model (which borrows the file bytes) stays alive
/// through emission.  `fused: true` precomputes the fusion schedule and
/// passes it to the emitter (T1.2); `false` emits per-op only.
/// `arena: true` enables the T1.3 liveness arena for intermediates;
/// `false` forces per-tensor stack arrays (`#[model_stack]` test arm).
/// `stage_input: true` honors the T4.2 graph-input 16B-staging decision.
/// `force_t2: true` (`#[model_force_t2]` test arm, T5.1) opens the T2
/// `requires_verification` gate so T2 groups may emit composed — the W5
/// flip surface the fused==unfused harness proves.  Never set by
/// `#[model]` / `#[model_unfused]` / `#[model_stack]` / `#[model_unstaged]`.
fn parse_and_emit_impl(
    attr: TokenStream,
    proc_item: proc_macro2::TokenStream,
    fused: bool,
    arena: bool,
    stage_input: bool,
    force_t2: bool,
) -> proc_macro2::TokenStream {
    let path = match model_path_from_attr(&attr) {
        Ok(p) => p,
        Err(msg) => return compile_error_with_item(msg, proc_item),
    };
    let data = match std::fs::read(&path) {
        Ok(d) => d,
        Err(e) => {
            return compile_error_with_item(format!("cannot read {}: {e}", path.display()), proc_item);
        }
    };
    let model = match flatbuffer::parse(&data) {
        Ok(m) => m,
        Err(e) => {
            return compile_error_with_item(
                format!(
                    "TFLite parse error in {}: {e}\n\
                     ─ help: verify the model is a valid TFLite file with 'TFL3' identifier",
                    path.display()
                ),
                proc_item,
            );
        }
    };
    let emitted = if fused {
        let schedule = optimize::fusion::fuse(&model);
        if arena {
            generate::emit_model_fused_with_policy(&model, &schedule, stage_input, force_t2)
        } else {
            generate::emit_model_stack_fused_with_policy(&model, &schedule, stage_input, force_t2)
        }
    } else {
        generate::emit_model(&model)
    };
    match emitted {
        Ok(generated) => {
            quote::quote! {
                #generated
                #proc_item
            }
        }
        Err(msg) => compile_error_with_item(msg, proc_item),
    }
}

fn compile_error_with_item(msg: String, proc_item: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    quote::quote! {
        compile_error!(#msg);
        #proc_item
    }
}

/// Resolve the attribute string literal to a filesystem path, relative to the
/// consumer crate's `CARGO_MANIFEST_DIR`.
fn model_path_from_attr(attr: &TokenStream) -> Result<std::path::PathBuf, String> {
    let attr_str = attr.to_string();
    let lit: syn::LitStr = syn::parse_str(&attr_str)
        .map_err(|e| format!("expected string literal: {e}"))?;

    let path_str = lit.value();
    let cargo_dir = std::env::var("CARGO_MANIFEST_DIR")
        .unwrap_or_else(|_| ".".to_string());
    Ok(std::path::Path::new(&cargo_dir).join(&path_str))
}

// ---------------------------------------------------------------------------
// Unit tests — in-crate only (proc-macro restriction)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::flatbuffer;

    const SINE_TFLITE: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../models/sine.tflite"
    ));

    /// A real `Conv2DOptions` table emitted by the reference flatbuffers
    /// Python Builder (flatbuffers 25.12.19) — padding=VALID(1),
    /// stride_w=2, stride_h=3, fused=RELU(1), dilation_w=2, dilation_h=2.
    /// The byte sequence IS the known-good reference wire format.
    const CONV2D_OPTIONS_BYTES: &[u8] = &[
        0x14, 0x00, 0x00, 0x00, 0x10, 0x00, 0x1c, 0x00, 0x07, 0x00, 0x08, 0x00,
        0x0c, 0x00, 0x13, 0x00, 0x14, 0x00, 0x18, 0x00, 0x10, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
    ];

    /// Probe model (flatbuffers Python Builder): three operators —
    /// op0: extended builtin_code 200 (≥127) with a Conv2DOptions table;
    /// op1: CUSTOM (32) with discriminator NONE + custom_options bytes;
    /// op2: MEAN (40) with axis resolved from input tensor 2's buffer.
    const PROBE_MODEL_BYTES: &[u8] = &[
        0x18, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x00, 0x00, 0x0e, 0x00,
        0x14, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x10, 0x00,
        0x0e, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x38, 0x00, 0x00, 0x00,
        0x68, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
        0x24, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x06, 0x00, 0x08, 0x00, 0x04, 0x00, 0x06, 0x00, 0x00, 0x00,
        0xd0, 0x01, 0x00, 0x00, 0xfc, 0xff, 0xff, 0xff, 0x04, 0x00, 0x04, 0x00,
        0x04, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x28, 0x00, 0x00, 0x00,
        0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xf0, 0xff, 0xff, 0xff,
        0x28, 0x00, 0x00, 0x00, 0xf8, 0xff, 0xff, 0xff, 0x20, 0x00, 0x00, 0x00,
        0x0c, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00,
        0x0c, 0x00, 0x00, 0x00, 0xc8, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x18, 0x00, 0x04, 0x00,
        0x08, 0x00, 0x0c, 0x00, 0x10, 0x00, 0x14, 0x00, 0x0e, 0x00, 0x00, 0x00,
        0xc8, 0x00, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00,
        0x24, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
        0x6d, 0x61, 0x69, 0x6e, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x03, 0x00, 0x00, 0x00, 0x5c, 0x00, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00,
        0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x10, 0x00, 0x04, 0x00,
        0x08, 0x00, 0x0c, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
        0x58, 0x00, 0x00, 0x00, 0x60, 0x00, 0x00, 0x00, 0x10, 0x00, 0x16, 0x00,
        0x04, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00,
        0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x4c, 0x00, 0x00, 0x00,
        0x40, 0x00, 0x00, 0x00, 0x28, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x00,
        0x14, 0x00, 0x00, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0f, 0x00, 0x10, 0x00,
        0x0e, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x01, 0xe8, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
        0xab, 0xcd, 0xef, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
        0x78, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
        0xa0, 0xff, 0xff, 0xff, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09,
        0x02, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00,
        0x70, 0x72, 0x6f, 0x62, 0x65, 0x5f, 0x61, 0x78, 0x69, 0x73, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x10, 0x00,
        0x04, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x0c, 0x00, 0x00, 0x00,
        0x1c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09, 0x04, 0x00, 0x00, 0x00,
        0x09, 0x00, 0x00, 0x00, 0x70, 0x72, 0x6f, 0x62, 0x65, 0x5f, 0x6f, 0x75,
        0x74, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x0c, 0x00, 0x14, 0x00, 0x04, 0x00, 0x0b, 0x00, 0x0c, 0x00, 0x10, 0x00,
        0x0c, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09,
        0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
        0x70, 0x72, 0x6f, 0x62, 0x65, 0x5f, 0x69, 0x6e, 0x00, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x1c, 0x00, 0x07, 0x00, 0x08, 0x00,
        0x0c, 0x00, 0x13, 0x00, 0x14, 0x00, 0x18, 0x00, 0x10, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
    ];

    #[test]
    fn test_parse_sine_model() {
        let model = flatbuffer::parse(SINE_TFLITE).expect("sine.tflite should parse");
        assert_eq!(model.subgraph_count(), 1);
        assert_eq!(model.inputs(), &[0]);
        assert_eq!(model.outputs(), &[3]);
        assert_eq!(model.tensors().len(), 4);
        assert_eq!(model.ops().len(), 1);
        assert_eq!(model.buffers().len(), 4);

        let op = &model.ops()[0];
        assert_eq!(op.opcode_index, 0);
        assert_eq!(op.builtin_code, 9); // FULLY_CONNECTED
        assert_eq!(op.inputs, vec![0, 1, 2]);
        assert_eq!(op.outputs, vec![3]);
        assert!(op.custom_options.is_empty());
        match &op.options {
            Some(flatbuffer::ParsedOptions::FullyConnected {
                fused_activation,
                weights_format,
                keep_num_dims,
            }) => {
                assert_eq!(*fused_activation, 0); // NONE
                assert_eq!(*weights_format, 0); // DEFAULT
                assert!(!*keep_num_dims);
            }
            other => panic!("expected FullyConnected options, got {other:?}"),
        }

        // Weight tensor (input 1): buffer contains the int8 weight 51.
        let weight = model.tensor_by_index(1).expect("weight tensor");
        assert_eq!(weight.name, "fc/weights");
        assert_eq!(weight.shape, vec![1, 1]);
        assert_eq!(weight.tensor_type, flatbuffer::TensorType::Int8);
        let weight_data = model.buffer_data(weight).expect("weight buffer");
        assert_eq!(weight_data[0], 51);
        let wq = weight.quant.as_ref().expect("weight quant");
        assert!((wq.scale - 0.007_812_5).abs() < 1e-6); // 1/128
        assert_eq!(wq.zero_point, 0);
        assert!(wq.per_channel.is_none());

        // Bias tensor (input 2): int32 -3.
        let bias = model.tensor_by_index(2).expect("bias tensor");
        let bias_data = model.buffer_data(bias).expect("bias buffer");
        let bias_val =
            i32::from_le_bytes([bias_data[0], bias_data[1], bias_data[2], bias_data[3]]);
        assert_eq!(bias_val, -3);

        // Input tensor: INT8, scale 0.1, empty buffer (None from buffer_data).
        let input = model.tensor_by_index(0).expect("input tensor");
        assert_eq!(input.name, "input");
        assert_eq!(input.shape, vec![1]);
        assert_eq!(input.tensor_type, flatbuffer::TensorType::Int8);
        assert!(model.buffer_data(input).is_none());
        let iq = input.quant.as_ref().expect("input quant");
        assert!((iq.scale - 0.1).abs() < 1e-6);
        assert_eq!(iq.zero_point, 0);

        // Output tensor: INT8, scale 0.1.
        let output = model.tensor_by_index(3).expect("output tensor");
        assert_eq!(output.name, "output");
        assert_eq!(output.tensor_type, flatbuffer::TensorType::Int8);
        let oq = output.quant.as_ref().expect("output quant");
        assert!((oq.scale - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_conv2d_options_byte_decode() {
        // Root uoffset: the table starts at byte 20 (bytes 0..4 of the fixture).
        let root =
            u32::from_le_bytes([CONV2D_OPTIONS_BYTES[0], CONV2D_OPTIONS_BYTES[1], CONV2D_OPTIONS_BYTES[2], CONV2D_OPTIONS_BYTES[3]]) as usize;
        assert_eq!(root, 20);

        let decoded = flatbuffer::parse_conv2d_options(CONV2D_OPTIONS_BYTES, root);
        match decoded {
            flatbuffer::ParsedOptions::Conv2D {
                padding,
                stride_w,
                stride_h,
                dilation_w,
                dilation_h,
                fused_activation,
            } => {
                assert_eq!(padding, 1); // VALID
                assert_eq!(stride_w, 2);
                assert_eq!(stride_h, 3);
                assert_eq!(dilation_w, 2);
                assert_eq!(dilation_h, 2);
                assert_eq!(fused_activation, 1); // RELU

                // Byte-for-byte: the raw bytes at each vtable-recorded field
                // offset match the decoded values exactly.
                let pad_off =
                    flatbuffer::table_field(CONV2D_OPTIONS_BYTES, root, 0).expect("padding");
                assert_eq!(CONV2D_OPTIONS_BYTES[pad_off], 0x01);
                let sw_off =
                    flatbuffer::table_field(CONV2D_OPTIONS_BYTES, root, 1).expect("stride_w");
                assert_eq!(&CONV2D_OPTIONS_BYTES[sw_off..sw_off + 4], &2i32.to_le_bytes());
                let sh_off =
                    flatbuffer::table_field(CONV2D_OPTIONS_BYTES, root, 2).expect("stride_h");
                assert_eq!(&CONV2D_OPTIONS_BYTES[sh_off..sh_off + 4], &3i32.to_le_bytes());
                let fa_off = flatbuffer::table_field(CONV2D_OPTIONS_BYTES, root, 3)
                    .expect("fused_activation");
                assert_eq!(CONV2D_OPTIONS_BYTES[fa_off], 0x01);
                let dw_off =
                    flatbuffer::table_field(CONV2D_OPTIONS_BYTES, root, 4).expect("dilation_w");
                assert_eq!(&CONV2D_OPTIONS_BYTES[dw_off..dw_off + 4], &2i32.to_le_bytes());
                let dh_off =
                    flatbuffer::table_field(CONV2D_OPTIONS_BYTES, root, 5).expect("dilation_h");
                assert_eq!(&CONV2D_OPTIONS_BYTES[dh_off..dh_off + 4], &2i32.to_le_bytes());
            }
            other => panic!("expected Conv2D options, got {other:?}"),
        }
    }

    #[test]
    fn test_probe_model_extended_opcode_and_custom() {
        let model = flatbuffer::parse(PROBE_MODEL_BYTES).expect("probe model should parse");
        assert_eq!(model.subgraph_count(), 1);
        assert_eq!(model.ops().len(), 3);

        // op0: extended builtin_code 200 resolved via opcodes table field 3,
        // options fall back to Custom(raw table bytes).
        let op0 = &model.ops()[0];
        assert_eq!(op0.builtin_code, 200);
        match &op0.options {
            Some(flatbuffer::ParsedOptions::Custom(bytes)) => assert!(!bytes.is_empty()),
            other => panic!("expected Custom options, got {other:?}"),
        }

        // op1: CUSTOM(32), discriminator NONE → no options; custom_options kept.
        let op1 = &model.ops()[1];
        assert_eq!(op1.builtin_code, 32);
        assert!(op1.options.is_none());
        assert_eq!(op1.custom_options, &[0xab, 0xcd, 0xef]);

        // op2: MEAN(40) synthesized with axis read from input tensor 2's buffer.
        let op2 = &model.ops()[2];
        assert_eq!(op2.builtin_code, 40);
        match &op2.options {
            Some(flatbuffer::ParsedOptions::Mean { axis, keep_dims }) => {
                assert_eq!(*axis, vec![1]);
                assert!(!*keep_dims);
            }
            other => panic!("expected Mean options, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_corrupted_bytes() {
        let result = flatbuffer::parse(&[0u8; 4]);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("too short"));
    }

    #[test]
    fn test_parse_bad_identifier() {
        let mut bad = vec![0u8; 100];
        bad[0..4].copy_from_slice(&12u32.to_le_bytes());
        bad[4..8].copy_from_slice(b"XXXX");
        let result = flatbuffer::parse(&bad);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("not \"TFL3\""));
    }

    #[test]
    fn test_parse_truncated_buffers_vector() {
        // Model root with a buffers vector whose declared length runs past the
        // end of the buffer → descriptive Truncated error.
        let mut buf = vec![0u8; 40];
        buf[0..4].copy_from_slice(&28u32.to_le_bytes()); // root offset
        buf[4..8].copy_from_slice(b"TFL3");
        // vtable at 12 for the Model table at 28
        buf[12..14].copy_from_slice(&8u16.to_le_bytes()); // vtable_len
        buf[14..16].copy_from_slice(&8u16.to_le_bytes()); // table_size
        buf[16..18].copy_from_slice(&4u16.to_le_bytes()); // field[0] at table+4
        let table_pos = 28usize;
        let soff = (table_pos - 12) as i32;
        buf[28..32].copy_from_slice(&soff.to_le_bytes());
        // field[0] at table+4=32: uoffset pointing to a fake buffers vector at 33
        buf[32..36].copy_from_slice(&1u32.to_le_bytes());
        // Vector at 33: len=1000 with truncated data
        buf[33..37].copy_from_slice(&1000u32.to_le_bytes());

        let result = flatbuffer::parse_buffers(&buf, 33);
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_vtable() {
        let mut buf = vec![0u8; 20];
        let soff = -(100i32);
        buf[0..4].copy_from_slice(&soff.to_le_bytes());
        let result = flatbuffer::table_field(&buf, 0, 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_vtable_missing_field_defaults() {
        // A Conv2DOptions table whose vtable omits most fields: only stride_w
        // is present (field 1) — every other field must fall back to its
        // schema default.
        let mut buf = Vec::new();
        buf.extend_from_slice(&[0u8; 4]);
        buf.extend_from_slice(&12u16.to_le_bytes()); // vtable_len
        buf.extend_from_slice(&16u16.to_le_bytes()); // table_size
        buf.extend_from_slice(&4u16.to_le_bytes()); // field[0] = padding at table+4
        buf.extend_from_slice(&8u16.to_le_bytes()); // field[1] = stride_w at table+8
        buf.extend_from_slice(&0u16.to_le_bytes()); // field[2] absent
        buf.extend_from_slice(&0u16.to_le_bytes()); // field[3] absent

        let table_pos = 4 + 12;
        let soff = table_pos as i32 - 4;
        buf.extend_from_slice(&soff.to_le_bytes()); // SOffsetT
        buf.push(0u8); // padding (unused)
        buf.extend_from_slice(&[0u8; 3]);
        buf.extend_from_slice(&3i32.to_le_bytes()); // stride_w = 3

        let result = flatbuffer::parse_conv2d_options(&buf, table_pos);
        match result {
            flatbuffer::ParsedOptions::Conv2D {
                padding,
                stride_w,
                stride_h,
                dilation_w,
                dilation_h,
                fused_activation,
            } => {
                assert_eq!(padding, 0); // SAME (default)
                assert_eq!(stride_w, 3);
                assert_eq!(stride_h, 1); // default
                assert_eq!(dilation_w, 1); // default
                assert_eq!(dilation_h, 1); // default
                assert_eq!(fused_activation, 0); // NONE (default)
            }
            other => panic!("expected Conv2D, got {other:?}"),
        }
    }

    #[test]
    fn test_quantization_per_tensor() {
        // Vtable at 4: vt_len=12 (fields 0-3), table_size=20, fields 2 and 3
        // present; vectors placed immediately after the table region.
        let mut buf = Vec::new();
        buf.extend_from_slice(&[0u8; 4]);
        buf.extend_from_slice(&12u16.to_le_bytes()); // vtable_len
        buf.extend_from_slice(&20u16.to_le_bytes()); // table_size
        buf.extend_from_slice(&0u16.to_le_bytes()); // field[0] absent
        buf.extend_from_slice(&0u16.to_le_bytes()); // field[1] absent
        buf.extend_from_slice(&4u16.to_le_bytes()); // field[2] scale
        buf.extend_from_slice(&12u16.to_le_bytes()); // field[3] zero_point

        let table_pos: usize = 4 + 12; // 16
        let soff = table_pos as i32 - 4;
        buf.extend_from_slice(&soff.to_le_bytes()); // SOffsetT (4B)

        let scale_vec_pos = table_pos + 20; // 36 — first vector after the table
        let scale_uoff = (scale_vec_pos - (table_pos + 4)) as u32; // field[2] at table+4
        buf.extend_from_slice(&scale_uoff.to_le_bytes());
        buf.extend_from_slice(&[0u8; 4]); // pad to field[3] at table+12
        let zp_vec_pos = scale_vec_pos + 8; // 44
        let zp_uoff = (zp_vec_pos - (table_pos + 12)) as u32; // field[3] at table+12
        buf.extend_from_slice(&zp_uoff.to_le_bytes());
        buf.extend_from_slice(&[0u8; 4]); // pad to table_size (table region 16..36)

        // Scale vector at 36: len=1, val=0.5
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0.5f32.to_le_bytes());
        // ZP vector at 44: len=1, val=-128
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&(-128i64).to_le_bytes());

        let quant =
            flatbuffer::parse_quantization_table(&buf, table_pos).expect("should parse quantization");
        assert!((quant.scale - 0.5).abs() < 0.001);
        assert_eq!(quant.zero_point, -128);
        assert!(quant.per_channel.is_none());
    }

    #[test]
    fn test_quantization_per_channel() {
        // Vtable at 4: vt_len=16 (fields 0-5), table_size=16; fields 2, 3 and 5
        // present.  Two scales/zero-points → per-channel QuantInfo.
        let mut buf = Vec::new();
        buf.extend_from_slice(&[0u8; 4]);
        buf.extend_from_slice(&16u16.to_le_bytes()); // vtable_len
        buf.extend_from_slice(&16u16.to_le_bytes()); // table_size
        buf.extend_from_slice(&0u16.to_le_bytes()); // field[0] absent
        buf.extend_from_slice(&0u16.to_le_bytes()); // field[1] absent
        buf.extend_from_slice(&4u16.to_le_bytes()); // field[2] scale
        buf.extend_from_slice(&8u16.to_le_bytes()); // field[3] zero_point
        buf.extend_from_slice(&0u16.to_le_bytes()); // field[4] absent
        buf.extend_from_slice(&12u16.to_le_bytes()); // field[5] quantized_dimension

        let table_pos: usize = 4 + 16; // 20
        let soff = table_pos as i32 - 4;
        buf.extend_from_slice(&soff.to_le_bytes()); // SOffsetT

        let scale_vec_pos = table_pos + 16; // 36
        let scale_uoff = (scale_vec_pos - (table_pos + 4)) as u32; // field[2] at table+4
        buf.extend_from_slice(&scale_uoff.to_le_bytes());
        let zp_vec_pos = scale_vec_pos + 12; // 48
        let zp_uoff = (zp_vec_pos - (table_pos + 8)) as u32; // field[3] at table+8
        buf.extend_from_slice(&zp_uoff.to_le_bytes());
        buf.extend_from_slice(&0i32.to_le_bytes()); // field[5] quantized_dimension = 0 (table+12)

        // Scale vector at 36: len=2, [0.5, 1.0]
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&0.5f32.to_le_bytes());
        buf.extend_from_slice(&1.0f32.to_le_bytes());
        // ZP vector at 48: len=2, [-128, 0]
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&(-128i64).to_le_bytes());
        buf.extend_from_slice(&0i64.to_le_bytes());

        let quant = flatbuffer::parse_quantization_table(&buf, table_pos)
            .expect("should parse per-channel quantization");
        assert!((quant.scale - 0.5).abs() < 0.001);
        assert_eq!(quant.zero_point, -128);
        let pc = quant.per_channel.expect("per_channel should be set");
        assert_eq!(pc.scales, vec![0.5, 1.0]);
        assert_eq!(pc.zero_points, vec![-128, 0]);
        assert_eq!(pc.quantized_dimension, 0);
    }
}
