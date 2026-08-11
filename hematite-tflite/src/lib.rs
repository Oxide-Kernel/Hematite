// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Hand-rolled TFLite flatbuffer byte-offset walker.
//!
//! Zero dependency on the `flatbuffers` crate.  Walks tables, vectors,
//! strings, and the `BuiltinOptions` union by raw byte offsets against the
//! TFLite schema vendored at `tools/tflite-schema/schema.fbs` (tensorflow
//! v2.14.1, flatbuffers v23.1-era).
//!
//! ## Parsed-model IR — stable API surface
//!
//! These types are consumed by T4.1 (generate), T4.2a (fusion),
//! T4.2b (arena), and T4.2c (layout).  **Do not rename or re-signature
//! without coordinating those tasks.**
//!
//! ## Wire-format notes (v23.1-era schema)
//!
//! * A TFLite file is a flatbuffer rooted at the uoffset in bytes `0..4`,
//!   with the `TFL3` file identifier in bytes `4..8`.
//! * Tables begin with a `SOffsetT` (i32) pointing *backward* to their
//!   vtable; vtable slots `4 + 2*i` hold per-field relative offsets into the
//!   table (0 = field absent → schema default).
//! * A union field is encoded as **two** vtable slots: the discriminator
//!   (u8, `builtin_options_type`) at slot 3 and the union value (uoffset to
//!   the options table, `builtin_options`) at slot 4.  `custom_options`
//!   (vector of bytes) is slot 5.
//! * `BuiltinOperator` codes are a stable explicit enum; dispatch on the
//!   **resolved** code from the `OperatorCode` table (field 3, int32 — the
//!   extended path for codes ≥ 127 — falling back to the deprecated byte at
//!   field 0 for legacy files).
//!
//! Dead-code warnings are expected at T4.0 — T4.1 consumes every field.
//
// allow: SIZE_OK — single-responsibility flatbuffer decoder; the core walker
// is ~300 LOC (per plan T4.0 spec) and the remainder is a homogeneous table
// of mechanical per-options field extractors that cannot be reviewed
// meaningfully in smaller files.
#![allow(dead_code)]

use std::fmt;

// ---------------------------------------------------------------------------
// Parsed-model IR
// ---------------------------------------------------------------------------

/// A parsed TFLite model, borrowing from the original flatbuffer bytes.
#[derive(Clone, Debug)]
pub struct ParsedModel<'a> {
    bytes: &'a [u8],
    subgraph_count: u32,
    inputs: Vec<u32>,
    outputs: Vec<u32>,
    tensors: Vec<ParsedTensor<'a>>,
    ops: Vec<ParsedOp<'a>>,
    buffers: Vec<ParsedBuffer<'a>>,
}

impl<'a> ParsedModel<'a> {
    /// Number of subgraphs (subgraph[0] is parsed).
    pub fn subgraph_count(&self) -> u32 {
        self.subgraph_count
    }

    /// Tensor indices fed into subgraph[0] (its `inputs` vector).
    pub fn inputs(&self) -> &[u32] {
        &self.inputs
    }

    /// Tensor indices produced by subgraph[0] (its `outputs` vector).
    pub fn outputs(&self) -> &[u32] {
        &self.outputs
    }

    /// All tensors of subgraph[0].
    pub fn tensors(&self) -> &[ParsedTensor<'a>] {
        &self.tensors
    }

    /// All operators of subgraph[0], in execution order.
    pub fn ops(&self) -> &[ParsedOp<'a>] {
        &self.ops
    }

    /// All model buffers (index 0 is the empty sentinel).
    pub fn buffers(&self) -> &[ParsedBuffer<'a>] {
        &self.buffers
    }

    /// Look up a tensor by index; `None` if out of range.
    pub fn tensor_by_index(&self, index: usize) -> Option<&ParsedTensor<'a>> {
        self.tensors.get(index)
    }

    /// Raw bytes backing `tensor`'s buffer; `None` for empty buffers
    /// (including the buffer-0 sentinel used by intermediates).
    pub fn buffer_data(&self, tensor: &ParsedTensor<'a>) -> Option<&'a [u8]> {
        self.buffers
            .get(tensor.buffer_index as usize)
            .and_then(|b| if b.data.is_empty() { None } else { Some(b.data) })
    }
}

/// A single operator in subgraph[0], in execution order.
#[derive(Clone, Debug)]
pub struct ParsedOp<'a> {
    /// Index into the model's `operator_codes` table.
    pub opcode_index: u32,
    /// Resolved `BuiltinOperator` code (via `operator_codes`); `-1` for
    /// custom operators whose code could not be resolved.  Extended codes
    /// (≥ 127) come from the `OperatorCode.builtin_code` int32 field.
    pub builtin_code: i32,
    /// Input tensor indices (optional inputs are `u32::MAX`).
    pub inputs: Vec<u32>,
    /// Output tensor indices.
    pub outputs: Vec<u32>,
    /// Decoded `BuiltinOptions` union value; `None` when the union
    /// discriminator is `NONE` and no variant is synthesized from the code.
    pub options: Option<ParsedOptions>,
    /// Raw `custom_options` byte vector (preserved verbatim, zero-copy).
    pub custom_options: &'a [u8],
}

/// Operator-specific options decoded from the `BuiltinOptions` union.
///
/// Dispatch is keyed on the resolved [`ParsedOp::builtin_code`] (stable
/// across schema revisions); the union discriminator only gates whether an
/// options table is present.
#[derive(Clone, Debug)]
pub enum ParsedOptions {
    /// `Conv2DOptions` — `padding`, `stride_w/h`, `dilation_w/h_factor`,
    /// `fused_activation_function`.
    Conv2D {
        padding: i8,
        stride_w: i32,
        stride_h: i32,
        dilation_w: i32,
        dilation_h: i32,
        fused_activation: i8,
    },
    /// `DepthwiseConv2DOptions` — as `Conv2D` plus `depth_multiplier`.
    DepthwiseConv2D {
        padding: i8,
        stride_w: i32,
        stride_h: i32,
        depth_multiplier: i32,
        dilation_w: i32,
        dilation_h: i32,
        fused_activation: i8,
    },
    /// `FullyConnectedOptions`.
    FullyConnected {
        fused_activation: i8,
        weights_format: i8,
        keep_num_dims: bool,
    },
    /// `Pool2DOptions` (average + max pooling share the table).
    Pool2D {
        padding: i8,
        stride_w: i32,
        stride_h: i32,
        filter_w: i32,
        filter_h: i32,
        fused_activation: i8,
    },
    /// `SoftmaxOptions`.
    Softmax { beta: f32 },
    /// `ReshapeOptions`.
    Reshape { new_shape: Vec<i32> },
    /// `AddOptions`.
    Add { fused_activation: i8, pot_scale_int16: bool },
    /// `SubOptions`.
    Sub { fused_activation: i8, pot_scale_int16: bool },
    /// `MulOptions`.
    Mul { fused_activation: i8 },
    /// MEAN — has no options table in the v23.1-era schema; `axis` is
    /// resolved from the `inputs[1]` tensor's buffer, `keep_dims` from the
    /// optional `MeanOptions` table (or `false` when absent).
    Mean { axis: Vec<i32>, keep_dims: bool },
    /// `ResizeNearestNeighborOptions`.
    ResizeNearest {
        align_corners: bool,
        half_pixel_centers: bool,
    },
    /// `LeakyReluOptions`.
    LeakyRelu { alpha: f32 },
    /// PRELU — no options table in the v23.1-era schema.
    Prelu,
    /// PAD / PADV2 — no options table (`TfLitePadParams` is empty); the
    /// padding amounts live in the `inputs[1]` const tensor's buffer.
    Pad,
    /// TRANSPOSE — no options table; the permutation lives in the
    /// `inputs[1]` const tensor's buffer.
    Transpose,
    /// Any op whose options are not decoded here: the raw options-table
    /// bytes (empty when no options table was present).
    Custom(Vec<u8>),
}

/// A tensor of subgraph[0].
#[derive(Clone, Debug)]
pub struct ParsedTensor<'a> {
    pub name: &'a str,
    pub shape: Vec<i32>,
    pub tensor_type: TensorType,
    pub quant: Option<QuantInfo>,
    pub buffer_index: u32,
}

/// TFLite `TensorType` enum values (v23.1-era numbering — `INT8 = 9`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorType {
    Float32,
    Float16,
    Int32,
    Uint8,
    Int64,
    String,
    Bool,
    Int16,
    Complex64,
    Int8,
    Float64,
    Complex128,
    Uint64,
    Resource,
    Variant,
    Uint32,
    Uint16,
    Int4,
    /// Any value beyond the vendored schema's enum.
    Unknown,
}

impl TensorType {
    pub fn from_byte(b: i8) -> Self {
        match b {
            0 => Self::Float32,
            1 => Self::Float16,
            2 => Self::Int32,
            3 => Self::Uint8,
            4 => Self::Int64,
            5 => Self::String,
            6 => Self::Bool,
            7 => Self::Int16,
            8 => Self::Complex64,
            9 => Self::Int8,
            10 => Self::Float64,
            11 => Self::Complex128,
            12 => Self::Uint64,
            13 => Self::Resource,
            14 => Self::Variant,
            15 => Self::Uint32,
            16 => Self::Uint16,
            17 => Self::Int4,
            _ => Self::Unknown,
        }
    }
}

/// Per-tensor or per-channel quantization of a [`ParsedTensor`].
#[derive(Clone, Debug)]
pub struct QuantInfo {
    /// First scale (per-tensor scale, or channel-0 scale for per-channel).
    pub scale: f32,
    /// First zero point.
    pub zero_point: i64,
    /// Present when the `scale`/`zero_point` vectors have length > 1.
    pub per_channel: Option<PerChannel>,
}

/// Per-channel quantization parameters.
#[derive(Clone, Debug)]
pub struct PerChannel {
    pub scales: Vec<f32>,
    pub zero_points: Vec<i64>,
    pub quantized_dimension: usize,
}

/// A model buffer containing raw byte data (index 0 is the empty sentinel).
#[derive(Clone, Debug)]
pub struct ParsedBuffer<'a> {
    pub data: &'a [u8],
}

/// Error returned by [`parse`].
#[derive(Clone, Debug)]
pub enum ParseError {
    TooShort,
    BadIdentifier,
    BadField { context: String },
    Truncated { context: String },
    BadRootOffset,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::TooShort => write!(f, "buffer too short for TFLite model"),
            ParseError::BadIdentifier => write!(f, "file identifier is not \"TFL3\""),
            ParseError::BadField { context } => write!(f, "bad field: {context}"),
            ParseError::Truncated { context } => write!(f, "truncated: {context}"),
            ParseError::BadRootOffset => write!(f, "root offset points beyond buffer"),
        }
    }
}

// ---------------------------------------------------------------------------
// Low-level flatbuffer primitives
// ---------------------------------------------------------------------------

fn u32_at(buf: &[u8], pos: usize) -> Option<u32> {
    let b = buf.get(pos..pos + 4)?;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn i32_at(buf: &[u8], pos: usize) -> Option<i32> {
    let b = buf.get(pos..pos + 4)?;
    Some(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn u16_at(buf: &[u8], pos: usize) -> Option<u16> {
    let b = buf.get(pos..pos + 2)?;
    Some(u16::from_le_bytes([b[0], b[1]]))
}

fn i8_at(buf: &[u8], pos: usize) -> Option<i8> {
    buf.get(pos).copied().map(|v| v as i8)
}

fn f32_at(buf: &[u8], pos: usize) -> Option<f32> {
    let b = buf.get(pos..pos + 4)?;
    Some(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn i64_at(buf: &[u8], pos: usize) -> Option<i64> {
    let b = buf.get(pos..pos + 8)?;
    Some(i64::from_le_bytes([
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
    ]))
}

/// Absolute position of `field_idx` within the table at `table_pos`, or
/// `None` when the vtable is missing/truncated or the field slot is 0.
pub fn table_field(buf: &[u8], table_pos: usize, field_idx: usize) -> Option<usize> {
    let vtable_soff = i32_at(buf, table_pos)? as usize;
    let vtable_pos = table_pos.wrapping_sub(vtable_soff);
    let vt_len = u16_at(buf, vtable_pos)? as usize;
    let field_off_pos = vtable_pos + 4 + field_idx * 2;
    if field_off_pos + 2 > vtable_pos + vt_len {
        return None;
    }
    let field_off = u16_at(buf, field_off_pos)? as usize;
    if field_off == 0 {
        return None;
    }
    Some(table_pos + field_off)
}

fn vector_header(buf: &[u8], pos: usize) -> Option<(u32, usize)> {
    let len = u32_at(buf, pos)?;
    Some((len, pos + 4))
}

/// Resolve a uoffset at `pos` into an absolute position (`None` for 0).
fn uoffset_at(buf: &[u8], pos: usize) -> Option<usize> {
    let off = u32_at(buf, pos)? as usize;
    if off == 0 {
        return None;
    }
    Some(pos + off)
}

fn string_at(buf: &[u8], pos: usize) -> Option<&str> {
    let str_pos = uoffset_at(buf, pos)?;
    let len = u32_at(buf, str_pos)? as usize;
    let raw = buf.get(str_pos + 4..str_pos + 4 + len)?;
    std::str::from_utf8(raw).ok()
}

fn read_index_vector(buf: &[u8], pos: usize) -> Option<Vec<u32>> {
    let vec_pos = uoffset_at(buf, pos)?;
    let (len, elem_pos) = vector_header(buf, vec_pos)?;
    let mut out = Vec::with_capacity(len as usize);
    for i in 0..len as usize {
        out.push(u32_at(buf, elem_pos + i * 4)?);
    }
    Some(out)
}

fn read_i32_vector(buf: &[u8], pos: usize) -> Option<Vec<i32>> {
    let vec_pos = uoffset_at(buf, pos)?;
    let (len, elem_pos) = vector_header(buf, vec_pos)?;
    let mut out = Vec::with_capacity(len as usize);
    for i in 0..len as usize {
        out.push(i32_at(buf, elem_pos + i * 4)?);
    }
    Some(out)
}

fn read_f32_vector(buf: &[u8], pos: usize) -> Option<Vec<f32>> {
    let vec_pos = uoffset_at(buf, pos)?;
    let (len, elem_pos) = vector_header(buf, vec_pos)?;
    let mut out = Vec::with_capacity(len as usize);
    for i in 0..len as usize {
        out.push(f32_at(buf, elem_pos + i * 4)?);
    }
    Some(out)
}

fn read_i64_vector(buf: &[u8], pos: usize) -> Option<Vec<i64>> {
    let vec_pos = uoffset_at(buf, pos)?;
    let (len, elem_pos) = vector_header(buf, vec_pos)?;
    let mut out = Vec::with_capacity(len as usize);
    for i in 0..len as usize {
        out.push(i64_at(buf, elem_pos + i * 8)?);
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Model parsing
// ---------------------------------------------------------------------------

/// Parse a TFLite Model flatbuffer.  Zero-copy: buffer data are `&[u8]`
/// views into the original `bytes` slice.
pub fn parse(bytes: &[u8]) -> Result<ParsedModel<'_>, ParseError> {
    if bytes.len() < 12 {
        return Err(ParseError::TooShort);
    }
    let root_off = u32_at(bytes, 0).ok_or(ParseError::TooShort)? as usize;
    if root_off >= bytes.len() {
        return Err(ParseError::BadRootOffset);
    }
    if bytes.get(4..8) != Some(b"TFL3") {
        return Err(ParseError::BadIdentifier);
    }

    let opcodes_field = table_field(bytes, root_off, 1)
        .ok_or(ParseError::Truncated { context: "operator_codes".into() })?;
    let subgraphs_field = table_field(bytes, root_off, 2)
        .ok_or(ParseError::Truncated { context: "subgraphs".into() })?;
    let buffers_field = table_field(bytes, root_off, 4)
        .ok_or(ParseError::Truncated { context: "buffers".into() })?;

    let opcodes_off = uoffset_at(bytes, opcodes_field)
        .ok_or(ParseError::Truncated { context: "operator_codes vector".into() })?;
    let subgraphs_off = uoffset_at(bytes, subgraphs_field)
        .ok_or(ParseError::Truncated { context: "subgraphs vector".into() })?;
    let buffers_off = uoffset_at(bytes, buffers_field)
        .ok_or(ParseError::Truncated { context: "buffers vector".into() })?;

    let opcodes = parse_opcodes(bytes, opcodes_off)?;
    let buffers = parse_buffers_inner(bytes, buffers_off)?;
    let (subgraph_count, inputs, outputs, tensors, ops) =
        parse_subgraph(bytes, subgraphs_off, &opcodes, &buffers)?;

    Ok(ParsedModel { bytes, subgraph_count, inputs, outputs, tensors, ops, buffers })
}

/// Resolve the `BuiltinOperator` codes for every `OperatorCode` table.
///
/// The extended path (codes ≥ 127) lives in `builtin_code` (field 3, int32);
/// legacy files store the code in `deprecated_builtin_code` (field 0, byte).
fn parse_opcodes(bytes: &[u8], opcodes_off: usize) -> Result<Vec<i32>, ParseError> {
    let (len, elem_pos) = vector_header(bytes, opcodes_off)
        .ok_or(ParseError::Truncated { context: "opcodes vector".into() })?;
    let mut out = Vec::with_capacity(len as usize);
    for i in 0..len as usize {
        let table_pos = uoffset_at(bytes, elem_pos + i * 4)
            .ok_or(ParseError::Truncated { context: format!("opcode table {i}") })?;
        let code = if let Some(pos) = table_field(bytes, table_pos, 3) {
            i32_at(bytes, pos).ok_or(ParseError::Truncated {
                context: format!("opcode[{i}].builtin_code"),
            })?
        } else if let Some(pos) = table_field(bytes, table_pos, 0) {
            i8_at(bytes, pos).ok_or(ParseError::Truncated {
                context: format!("opcode[{i}].deprecated_builtin_code"),
            })? as i32
        } else {
            // Neither field present: both schema fields default to 0 (ADD),
            // and the flatbuffers builder omits fields equal to their default.
            // Legacy-encoded models (code only in `deprecated_builtin_code`)
            // omit it for ADD. Resolve to the schema default.
            0
        };
        out.push(code);
    }
    Ok(out)
}

/// Parsed contents of subgraph[0] plus the subgraph count.
type SubgraphResult<'a> = (
    u32,
    Vec<u32>,
    Vec<u32>,
    Vec<ParsedTensor<'a>>,
    Vec<ParsedOp<'a>>,
);

fn parse_subgraph<'a>(
    bytes: &'a [u8],
    subgraphs_off: usize,
    opcodes: &[i32],
    buffers: &[ParsedBuffer<'a>],
) -> Result<SubgraphResult<'a>, ParseError> {
    let (subgraph_count, elem_pos) = vector_header(bytes, subgraphs_off)
        .ok_or(ParseError::Truncated { context: "subgraphs vector".into() })?;
    let sg_pos = uoffset_at(bytes, elem_pos)
        .ok_or(ParseError::Truncated { context: "subgraph[0]".into() })?;

    let tensors_off = table_field(bytes, sg_pos, 0)
        .and_then(|p| uoffset_at(bytes, p))
        .ok_or(ParseError::Truncated { context: "subgraph.tensors".into() })?;
    let operators_off = table_field(bytes, sg_pos, 3)
        .and_then(|p| uoffset_at(bytes, p))
        .ok_or(ParseError::Truncated { context: "subgraph.operators".into() })?;

    let tensors = parse_tensors(bytes, tensors_off)?;
    let inputs = table_field(bytes, sg_pos, 1)
        .and_then(|p| read_index_vector(bytes, p))
        .unwrap_or_default();
    let outputs = table_field(bytes, sg_pos, 2)
        .and_then(|p| read_index_vector(bytes, p))
        .unwrap_or_default();
    let ops = parse_operators(bytes, operators_off, opcodes, &tensors, buffers)?;
    Ok((subgraph_count, inputs, outputs, tensors, ops))
}

fn parse_tensors<'a>(
    bytes: &'a [u8],
    tensors_off: usize,
) -> Result<Vec<ParsedTensor<'a>>, ParseError> {
    let (len, elem_pos) = vector_header(bytes, tensors_off)
        .ok_or(ParseError::Truncated { context: "tensors vector".into() })?;
    let mut out = Vec::with_capacity(len as usize);
    for i in 0..len as usize {
        let t_pos = uoffset_at(bytes, elem_pos + i * 4)
            .ok_or(ParseError::Truncated { context: format!("tensor[{i}]") })?;
        let shape = table_field(bytes, t_pos, 0)
            .and_then(|p| read_i32_vector(bytes, p))
            .unwrap_or_default();
        let tensor_type = table_field(bytes, t_pos, 1)
            .and_then(|p| i8_at(bytes, p))
            .map(TensorType::from_byte)
            .unwrap_or(TensorType::Unknown);
        let buffer_index = table_field(bytes, t_pos, 2)
            .and_then(|p| u32_at(bytes, p))
            .unwrap_or(0);
        let name = table_field(bytes, t_pos, 3)
            .and_then(|p| string_at(bytes, p))
            .unwrap_or("");
        let quant = table_field(bytes, t_pos, 4).and_then(|p| parse_quantization(bytes, p));
        out.push(ParsedTensor { name, shape, tensor_type, quant, buffer_index });
    }
    Ok(out)
}

fn parse_operators<'a>(
    bytes: &'a [u8],
    operators_off: usize,
    opcodes: &[i32],
    tensors: &[ParsedTensor<'a>],
    buffers: &[ParsedBuffer<'a>],
) -> Result<Vec<ParsedOp<'a>>, ParseError> {
    let (len, elem_pos) = vector_header(bytes, operators_off)
        .ok_or(ParseError::Truncated { context: "operators vector".into() })?;
    let mut out = Vec::with_capacity(len as usize);
    for i in 0..len as usize {
        let op_pos = uoffset_at(bytes, elem_pos + i * 4)
            .ok_or(ParseError::Truncated { context: format!("operator[{i}]") })?;

        let opcode_index = table_field(bytes, op_pos, 0)
            .and_then(|p| u32_at(bytes, p))
            .unwrap_or(0);
        let inputs = table_field(bytes, op_pos, 1)
            .and_then(|p| read_index_vector(bytes, p))
            .unwrap_or_default();
        let outputs = table_field(bytes, op_pos, 2)
            .and_then(|p| read_index_vector(bytes, p))
            .unwrap_or_default();
        let builtin_code = opcodes.get(opcode_index as usize).copied().unwrap_or(-1);
        let discriminator = table_field(bytes, op_pos, 3)
            .and_then(|p| i8_at(bytes, p))
            .unwrap_or(0);
        let options = parse_builtin_options(
            bytes,
            op_pos,
            discriminator,
            builtin_code,
            &inputs,
            tensors,
            buffers,
        )?;
        let custom_options: &[u8] = table_field(bytes, op_pos, 5)
            .and_then(|p| uoffset_at(bytes, p))
            .and_then(|vec_pos| {
                let (clen, data_pos) = vector_header(bytes, vec_pos)?;
                bytes.get(data_pos..data_pos + clen as usize)
            })
            .unwrap_or(&[]);

        out.push(ParsedOp {
            opcode_index,
            builtin_code,
            inputs,
            outputs,
            options,
            custom_options,
        });
    }
    Ok(out)
}

/// `pub` for test access.
pub fn parse_buffers(
    bytes: &[u8],
    buffers_off: usize,
) -> Result<Vec<ParsedBuffer<'_>>, ParseError> {
    parse_buffers_inner(bytes, buffers_off)
}

fn parse_buffers_inner<'a>(
    bytes: &'a [u8],
    buffers_off: usize,
) -> Result<Vec<ParsedBuffer<'a>>, ParseError> {
    let (len, elem_pos) = vector_header(bytes, buffers_off)
        .ok_or(ParseError::Truncated { context: "buffers vector".into() })?;
    let mut out = Vec::with_capacity(len as usize);
    for i in 0..len as usize {
        let buf_pos = uoffset_at(bytes, elem_pos + i * 4)
            .ok_or(ParseError::Truncated { context: format!("buffer[{i}]") })?;
        let data = table_field(bytes, buf_pos, 0)
            .and_then(|p| uoffset_at(bytes, p))
            .and_then(|vec_pos| {
                let (dlen, dpos) = vector_header(bytes, vec_pos)?;
                bytes.get(dpos..dpos + dlen as usize)
            })
            .unwrap_or(&[]);
        out.push(ParsedBuffer { data });
    }
    Ok(out)
}

/// Resolve a `quantization` union-adjacent field to a [`QuantInfo`].
/// `pub` for test access.
pub fn parse_quantization(bytes: &[u8], field_pos: usize) -> Option<QuantInfo> {
    let q_pos = uoffset_at(bytes, field_pos)?;
    parse_quantization_table(bytes, q_pos)
}

/// Parse a `QuantizationParameters` table at the resolved table position.
/// `pub` for test access.
///
/// Vtable slot layout (schema.fbs `QuantizationParameters`):
/// 0 min, 1 max, 2 scale, 3 zero_point, 4 details_type (union discriminator,
/// auto-generated by the flatbuffers compiler), 5 details value, 6
/// quantized_dimension.
pub fn parse_quantization_table(bytes: &[u8], q_pos: usize) -> Option<QuantInfo> {
    let scales = table_field(bytes, q_pos, 2).and_then(|p| read_f32_vector(bytes, p));
    let zero_points = table_field(bytes, q_pos, 3).and_then(|p| read_i64_vector(bytes, p));
    let quant_dim = table_field(bytes, q_pos, 6)
        .and_then(|p| i32_at(bytes, p))
        .unwrap_or(0) as usize;

    match (&scales, &zero_points) {
        (Some(s), Some(z)) if s.len() > 1 => Some(QuantInfo {
            scale: s.first().copied().unwrap_or(1.0),
            zero_point: z.first().copied().unwrap_or(0),
            per_channel: Some(PerChannel {
                scales: s.clone(),
                zero_points: z.clone(),
                quantized_dimension: quant_dim,
            }),
        }),
        (Some(s), Some(z)) if !s.is_empty() => Some(QuantInfo {
            scale: s[0],
            zero_point: z.first().copied().unwrap_or(0),
            per_channel: None,
        }),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// BuiltinOptions union decoder
// ---------------------------------------------------------------------------

/// Decode the `BuiltinOptions` union for one operator.
///
/// Slot 3 of the Operator table is the union discriminator
/// (`builtin_options_type`); slot 4 is the options-table uoffset.  Dispatch
/// to the field decoder is keyed on the resolved `builtin_code`.
fn parse_builtin_options<'a>(
    bytes: &[u8],
    op_pos: usize,
    discriminator: i8,
    builtin_code: i32,
    inputs: &[u32],
    tensors: &[ParsedTensor<'a>],
    buffers: &[ParsedBuffer<'a>],
) -> Result<Option<ParsedOptions>, ParseError> {
    if discriminator == 0 {
        // `BuiltinOptions.NONE` — no options table.  Synthesize the variants
        // whose parameters live outside the union.
        return Ok(match builtin_code {
            40 => Some(ParsedOptions::Mean {
                axis: mean_axis(inputs, tensors, buffers),
                keep_dims: false,
            }),
            54 => Some(ParsedOptions::Prelu),
            34 => Some(ParsedOptions::Pad),
            39 => Some(ParsedOptions::Transpose),
            _ => None,
        });
    }

    let options_field = table_field(bytes, op_pos, 4).ok_or(ParseError::Truncated {
        context: "operator.builtin_options field".into(),
    })?;
    let table_pos = uoffset_at(bytes, options_field).ok_or(ParseError::Truncated {
        context: "operator.builtin_options uoffset".into(),
    })?;

    Ok(Some(match builtin_code {
        3 => parse_conv2d_options(bytes, table_pos),
        4 => parse_depthwise_options(bytes, table_pos),
        9 => parse_fc_options(bytes, table_pos),
        1 | 17 => parse_pool_options(bytes, table_pos),
        25 => parse_softmax_options(bytes, table_pos),
        22 => parse_reshape_options(bytes, table_pos),
        0 => parse_addsub_options(bytes, table_pos, false),
        41 => parse_addsub_options(bytes, table_pos, true),
        18 => parse_mul_options(bytes, table_pos),
        40 => ParsedOptions::Mean {
            axis: mean_axis(inputs, tensors, buffers),
            keep_dims: table_field(bytes, table_pos, 0)
                .and_then(|p| bytes.get(p).map(|&v| v != 0))
                .unwrap_or(false),
        },
        54 => ParsedOptions::Prelu,
        97 => parse_resize_nearest_options(bytes, table_pos),
        98 => parse_leaky_relu_options(bytes, table_pos),
        _ => ParsedOptions::Custom(raw_table_bytes(bytes, table_pos)),
    }))
}

/// MEAN's reduction axes live in the `inputs[1]` tensor's buffer (int32).
fn mean_axis<'a>(
    inputs: &[u32],
    tensors: &[ParsedTensor<'a>],
    buffers: &[ParsedBuffer<'a>],
) -> Vec<i32> {
    let Some(&t) = inputs.get(1) else {
        return vec![];
    };
    let Some(tensor) = tensors.get(t as usize) else {
        return vec![];
    };
    let Some(buffer) = buffers.get(tensor.buffer_index as usize) else {
        return vec![];
    };
    buffer
        .data
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Raw bytes of an options table (used for the `Custom` fallback).
fn raw_table_bytes(bytes: &[u8], table_pos: usize) -> Vec<u8> {
    let vt_pos = i32_at(bytes, table_pos).map(|s| table_pos.wrapping_sub(s as usize));
    let table_size = vt_pos
        .and_then(|p| u16_at(bytes, p + 2))
        .map(|s| s as usize);
    match table_size {
        Some(sz) => bytes.get(table_pos..table_pos + sz).unwrap_or(&[]).to_vec(),
        None => Vec::new(),
    }
}

/// Decode a `Conv2DOptions` table at `table_pos`.  `pub` for tests.
pub fn parse_conv2d_options(bytes: &[u8], t: usize) -> ParsedOptions {
    ParsedOptions::Conv2D {
        padding: table_field(bytes, t, 0).and_then(|p| i8_at(bytes, p)).unwrap_or(0),
        stride_w: table_field(bytes, t, 1).and_then(|p| i32_at(bytes, p)).unwrap_or(1),
        stride_h: table_field(bytes, t, 2).and_then(|p| i32_at(bytes, p)).unwrap_or(1),
        fused_activation: table_field(bytes, t, 3).and_then(|p| i8_at(bytes, p)).unwrap_or(0),
        dilation_w: table_field(bytes, t, 4).and_then(|p| i32_at(bytes, p)).unwrap_or(1),
        dilation_h: table_field(bytes, t, 5).and_then(|p| i32_at(bytes, p)).unwrap_or(1),
    }
}

fn parse_depthwise_options(bytes: &[u8], t: usize) -> ParsedOptions {
    ParsedOptions::DepthwiseConv2D {
        padding: table_field(bytes, t, 0).and_then(|p| i8_at(bytes, p)).unwrap_or(0),
        stride_w: table_field(bytes, t, 1).and_then(|p| i32_at(bytes, p)).unwrap_or(1),
        stride_h: table_field(bytes, t, 2).and_then(|p| i32_at(bytes, p)).unwrap_or(1),
        depth_multiplier: table_field(bytes, t, 3).and_then(|p| i32_at(bytes, p)).unwrap_or(1),
        fused_activation: table_field(bytes, t, 4).and_then(|p| i8_at(bytes, p)).unwrap_or(0),
        dilation_w: table_field(bytes, t, 5).and_then(|p| i32_at(bytes, p)).unwrap_or(1),
        dilation_h: table_field(bytes, t, 6).and_then(|p| i32_at(bytes, p)).unwrap_or(1),
    }
}

fn parse_fc_options(bytes: &[u8], t: usize) -> ParsedOptions {
    ParsedOptions::FullyConnected {
        fused_activation: table_field(bytes, t, 0).and_then(|p| i8_at(bytes, p)).unwrap_or(0),
        weights_format: table_field(bytes, t, 1).and_then(|p| i8_at(bytes, p)).unwrap_or(0),
        keep_num_dims: table_field(bytes, t, 2)
            .and_then(|p| bytes.get(p).map(|&v| v != 0))
            .unwrap_or(false),
    }
}

fn parse_pool_options(bytes: &[u8], t: usize) -> ParsedOptions {
    ParsedOptions::Pool2D {
        padding: table_field(bytes, t, 0).and_then(|p| i8_at(bytes, p)).unwrap_or(0),
        stride_w: table_field(bytes, t, 1).and_then(|p| i32_at(bytes, p)).unwrap_or(1),
        stride_h: table_field(bytes, t, 2).and_then(|p| i32_at(bytes, p)).unwrap_or(1),
        filter_w: table_field(bytes, t, 3).and_then(|p| i32_at(bytes, p)).unwrap_or(1),
        filter_h: table_field(bytes, t, 4).and_then(|p| i32_at(bytes, p)).unwrap_or(1),
        fused_activation: table_field(bytes, t, 5).and_then(|p| i8_at(bytes, p)).unwrap_or(0),
    }
}

fn parse_softmax_options(bytes: &[u8], t: usize) -> ParsedOptions {
    ParsedOptions::Softmax {
        beta: table_field(bytes, t, 0).and_then(|p| f32_at(bytes, p)).unwrap_or(1.0),
    }
}

fn parse_reshape_options(bytes: &[u8], t: usize) -> ParsedOptions {
    ParsedOptions::Reshape {
        new_shape: table_field(bytes, t, 0)
            .and_then(|p| read_i32_vector(bytes, p))
            .unwrap_or_default(),
    }
}

fn parse_addsub_options(bytes: &[u8], t: usize, is_sub: bool) -> ParsedOptions {
    let fused_activation = table_field(bytes, t, 0).and_then(|p| i8_at(bytes, p)).unwrap_or(0);
    let pot_scale_int16 = table_field(bytes, t, 1)
        .and_then(|p| bytes.get(p).map(|&v| v != 0))
        .unwrap_or(true);
    if is_sub {
        ParsedOptions::Sub { fused_activation, pot_scale_int16 }
    } else {
        ParsedOptions::Add { fused_activation, pot_scale_int16 }
    }
}

fn parse_mul_options(bytes: &[u8], t: usize) -> ParsedOptions {
    ParsedOptions::Mul {
        fused_activation: table_field(bytes, t, 0).and_then(|p| i8_at(bytes, p)).unwrap_or(0),
    }
}

fn parse_resize_nearest_options(bytes: &[u8], t: usize) -> ParsedOptions {
    ParsedOptions::ResizeNearest {
        align_corners: table_field(bytes, t, 0)
            .and_then(|p| bytes.get(p).map(|&v| v != 0))
            .unwrap_or(false),
        half_pixel_centers: table_field(bytes, t, 1)
            .and_then(|p| bytes.get(p).map(|&v| v != 0))
            .unwrap_or(false),
    }
}

fn parse_leaky_relu_options(bytes: &[u8], t: usize) -> ParsedOptions {
    ParsedOptions::LeakyRelu {
        alpha: table_field(bytes, t, 0).and_then(|p| f32_at(bytes, p)).unwrap_or(0.01),
    }
}

// ---------------------------------------------------------------------------
// Unit tests — in-crate only (proc-macro restriction)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-crafted `QuantizationParameters` table with `quantized_dimension`
    /// written to vtable slot 6 (per the vendored schema: 0 min, 1 max,
    /// 2 scale, 3 zero_point, 4 details_type, 5 details value, 6
    /// quantized_dimension).  The value 1 can only be read from slot 6.
    #[test]
    fn quantized_dimension_from_slot_6() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&[0u8; 4]);
        buf.extend_from_slice(&18u16.to_le_bytes()); // vtable_len
        buf.extend_from_slice(&20u16.to_le_bytes()); // table_size
        buf.extend_from_slice(&0u16.to_le_bytes()); // field[0] min absent
        buf.extend_from_slice(&0u16.to_le_bytes()); // field[1] max absent
        buf.extend_from_slice(&4u16.to_le_bytes()); // field[2] scale at table+4
        buf.extend_from_slice(&12u16.to_le_bytes()); // field[3] zero_point at table+12
        buf.extend_from_slice(&0u16.to_le_bytes()); // field[4] details_type absent
        buf.extend_from_slice(&0u16.to_le_bytes()); // field[5] details value absent
        buf.extend_from_slice(&16u16.to_le_bytes()); // field[6] quantized_dimension at table+16

        let table_pos: usize = 4 + 18; // 22
        let soff = table_pos as i32 - 4;
        buf.extend_from_slice(&soff.to_le_bytes()); // SOffsetT

        let scale_vec_pos = table_pos + 20; // 42 — first vector after the table
        let scale_uoff = (scale_vec_pos - (table_pos + 4)) as u32; // field[2] at table+4
        buf.extend_from_slice(&scale_uoff.to_le_bytes());
        buf.extend_from_slice(&[0u8; 4]); // pad to field[3] at table+12
        let zp_vec_pos = scale_vec_pos + 12; // 54
        let zp_uoff = (zp_vec_pos - (table_pos + 12)) as u32; // field[3] at table+12
        buf.extend_from_slice(&zp_uoff.to_le_bytes());
        buf.extend_from_slice(&1i32.to_le_bytes()); // field[6] quantized_dimension = 1 (table+16)

        // Scale vector at 42: len=2, [0.5, 1.0]
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&0.5f32.to_le_bytes());
        buf.extend_from_slice(&1.0f32.to_le_bytes());
        // ZP vector at 54: len=2, [-128, 0]
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&(-128i64).to_le_bytes());
        buf.extend_from_slice(&0i64.to_le_bytes());

        let quant = parse_quantization_table(&buf, table_pos)
            .expect("should parse per-channel quantization");
        assert!((quant.scale - 0.5).abs() < 0.001);
        assert_eq!(quant.zero_point, -128);
        let pc = quant.per_channel.expect("per_channel should be set");
        assert_eq!(pc.scales, vec![0.5, 1.0]);
        assert_eq!(pc.zero_points, vec![-128, 0]);
        assert_eq!(pc.quantized_dimension, 1);
    }

    /// An `OperatorCode` table whose vtable omits BOTH `builtin_code` (field
    /// 3) and `deprecated_builtin_code` (field 0) resolves to the schema
    /// default ADD (0). Legacy-encoded models (the xtensa
    /// `pytorch_to_tflite` artifacts) omit the field entirely for ADD, since
    /// flatbuffers drops fields equal to their default.
    #[test]
    fn opcode_missing_fields_defaults_to_add() {
        // vector at 0: len=1, element uoffset at 4 → table at 12 (V=8)
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        // vtable at 8: vt_len=4 (no field slots), table_size=12
        buf.extend_from_slice(&4u16.to_le_bytes());
        buf.extend_from_slice(&12u16.to_le_bytes());
        // table at 12: SOffsetT → vtable at 8
        buf.extend_from_slice(&4i32.to_le_bytes());
        let codes = parse_opcodes(&buf, 0).expect("should parse");
        assert_eq!(codes, vec![0]); // ADD
    }

    /// A legacy-encoded opcode table (code in `deprecated_builtin_code`,
    /// field 3 absent) resolves from field 0.
    #[test]
    fn opcode_legacy_deprecated_field() {
        // vector at 0: len=1, element uoffset at 4 → table at 14 (V=10)
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&10u32.to_le_bytes());
        // vtable at 8: vt_len=6 (one field slot), table_size=12, field[0] at table+4
        buf.extend_from_slice(&6u16.to_le_bytes());
        buf.extend_from_slice(&12u16.to_le_bytes());
        buf.extend_from_slice(&4u16.to_le_bytes());
        // table at 14: SOffsetT → vtable at 8, then deprecated code 34 (PAD)
        buf.extend_from_slice(&6i32.to_le_bytes());
        buf.push(34u8);
        buf.extend_from_slice(&[0u8; 3]);
        let codes = parse_opcodes(&buf, 0).expect("should parse");
        assert_eq!(codes, vec![34]); // PAD
    }
}
