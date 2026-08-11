// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Emit the ESP-NN baseline C harness for the zoo tflite models (Phase 3).
//!
//! For each of the 5 runnable zoo models (hello_world, kws, anomaly_detect,
//! person_detect, mobilenet_v2) this tool parses the `.tflite` (via
//! `hematite-tflite`) plus the golden `INPUT_DATA` (from
//! `hematite-tests/goldens/models/*.rs`), computes the same TFLM
//! quantization math as `hematite-codegen/src/generate.rs`, and emits C
//! into `benchmarks/espnn-baseline/main/zoo_gen/`:
//!
//! * `zoo_common.h` — shared harness (fnv1a, ccount timing, run_bench,
//!   scalar PAD/TRANSPOSE/SUB fallbacks).
//! * `zoo_<model>.c` — per model: 16-aligned const weights/biases/
//!   qmult/qshift, tensor slots carved from a caller-provided memory arena
//!   (liveness offsets from `hematite-memory::liveness_plan`), per-op
//!   esp_nn kernel calls, `zoo_<model>_run(mem, scratch)`.
//! * `zoo_runner.c` — per-model benchmark rows (N=10 min/median CCOUNT,
//!   fnv1a), PSRAM heap allocation for person_detect/mobilenet.
//!
//! Invocation: `cargo run -p espnn-codegen` from the workspace root.

use std::path::{Path, PathBuf};

use hematite_memory::{liveness_plan, ArenaPlan, OpInfo, OFFSET_NONE};

use hematite_tflite::{ParsedModel, ParsedOp, ParsedOptions, ParsedTensor, QuantInfo};

// ---------------------------------------------------------------------------
// Model registry
// ---------------------------------------------------------------------------

struct ModelSpec {
    /// C symbol prefix (also output file stem).
    name: &'static str,
    /// Workspace-relative tflite path.
    tflite: &'static str,
    /// Workspace-relative golden .rs path (INPUT_DATA source).
    golden: &'static str,
    /// Scratch bytes to reserve for the model (conv/dw/softmax staging).
    scratch_bytes: usize,
}

const MODELS: &[ModelSpec] = &[
    ModelSpec {
        name: "hello_world",
        tflite: "models/zoo/sine_regression/hello_world_int8.tflite",
        golden: "hematite-tests/goldens/models/hello_world_int8.rs",
        scratch_bytes: 8 * 1024,
    },
    ModelSpec {
        name: "kws",
        tflite: "models/zoo/keyword_spotting/kws_micro_speech_int8.tflite",
        golden: "hematite-tests/goldens/models/kws_micro_speech_int8.rs",
        scratch_bytes: 8 * 1024,
    },
    ModelSpec {
        name: "anomaly",
        tflite: "models/zoo/anomaly_detect/anomaly_detect_int8.tflite",
        golden: "hematite-tests/goldens/models/anomaly_detect_int8.rs",
        scratch_bytes: 8 * 1024,
    },
    ModelSpec {
        name: "person_detect",
        tflite: "models/zoo/person_detect_vww/person_detect_int8.tflite",
        golden: "hematite-tests/goldens/models/person_detect_int8.rs",
        scratch_bytes: 512 * 1024,
    },
    ModelSpec {
        name: "mobilenet",
        tflite: "models/zoo/mobilenetv2_cls/mobilenet_v2_1.0_224_int8.tflite",
        golden: "hematite-tests/goldens/models/mobilenet_v2_1.0_224_int8.rs",
        scratch_bytes: 2 * 1024 * 1024,
    },
];

// ---------------------------------------------------------------------------
// TFLM quant math (mirrors hematite-codegen/src/generate.rs exactly)
// ---------------------------------------------------------------------------

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

fn flat_len(shape: &[i32]) -> Result<usize, String> {
    let mut n: i64 = 1;
    for &d in shape {
        if d <= 0 {
            return Err(format!("dynamic or non-positive shape dimension {d}"));
        }
        n *= i64::from(d);
    }
    Ok(n as usize)
}

fn shape4(shape: &[i32]) -> Result<[i32; 4], String> {
    if shape.len() > 4 {
        return Err(format!("tensor rank {} exceeds 4", shape.len()));
    }
    let mut out = [1i32; 4];
    let base = 4 - shape.len();
    for (i, &d) in shape.iter().enumerate() {
        if d <= 0 {
            return Err(format!("dynamic or non-positive shape dimension {d}"));
        }
        out[base + i] = d;
    }
    Ok(out)
}

fn tensor_scale(t: &ParsedTensor) -> Result<f64, String> {
    match &t.quant {
        Some(q) if q.scale.is_finite() && q.scale > 0.0 => Ok(f64::from(q.scale)),
        Some(q) => Err(format!("tensor {} has invalid scale {}", t.name, q.scale)),
        None => Err(format!("tensor {} has no quantization", t.name)),
    }
}

fn tensor_zp(t: &ParsedTensor) -> Result<i32, String> {
    match &t.quant {
        Some(q) => Ok(q.zero_point as i32),
        None => Err(format!("tensor {} has no quantization", t.name)),
    }
}

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

/// TFLM `QuantizedMeanOrSum` multiplier adaptation (compute_sum == false).
fn mean_adapt(mult: i32, shift: i32, count: u64) -> (i32, i32) {
    if count == 0 {
        return (0, shift);
    }
    let mut mshift = (63 - count.leading_zeros() as i32).min(32);
    mshift = mshift.min(31 + shift);
    let mean_mult = (((mult as i64) << mshift) / count as i64) as i32;
    let mean_shift = shift - mshift;
    (mean_mult, mean_shift)
}

// ---------------------------------------------------------------------------
// Golden INPUT_DATA parsing
// ---------------------------------------------------------------------------

fn parse_i8_array(content: &str, const_name: &str) -> Result<Vec<i8>, String> {
    let start_marker = format!("pub const {const_name}: [i8;");
    let start = content
        .find(&start_marker)
        .ok_or_else(|| format!("missing `{start_marker}`"))?;
    // The value array begins after `] = [`. Find `] = [` to skip the type.
    let assign = content[start..]
        .find("] = [")
        .ok_or_else(|| format!("missing `] = [` after `{const_name}`"))?
        + start;
    let bracket = assign + 4; // point at the `[` after `] = `
    let close = content[bracket..]
        .find("];")
        .ok_or_else(|| format!("missing `];` after `{const_name}`"))?
        + bracket;
    let body = &content[bracket + 1..close];
    let mut out = Vec::new();
    for tok in body.split(|c: char| c == ',' || c.is_whitespace()) {
        if tok.is_empty() {
            continue;
        }
        let v: i8 = tok
            .trim()
            .parse()
            .map_err(|e| format!("bad i8 literal `{tok}` in {const_name}: {e}"))?;
        out.push(v);
    }
    if out.is_empty() {
        return Err(format!("{const_name}: no values parsed"));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// C emission helpers
// ---------------------------------------------------------------------------

struct C {
    buf: String,
}

impl C {
    fn new() -> Self {
        C { buf: String::new() }
    }

    fn line(&mut self, s: &str) {
        self.buf.push_str(s);
        self.buf.push('\n');
    }

    fn blank(&mut self) {
        self.buf.push('\n');
    }

    fn fmt(&mut self, args: std::fmt::Arguments<'_>) {
        self.buf.push_str(&args.to_string());
    }

    fn push(&mut self, s: &str) {
        self.buf.push_str(s);
    }
}

fn c_bytes(data: &[u8], per_line: usize) -> String {
    let mut out = String::new();
    for (i, b) in data.iter().enumerate() {
        if i > 0 && i % per_line == 0 {
            out.push('\n');
        }
        out.push_str(&format!("{}, ", b));
    }
    out.push('\n');
    out
}

fn c_i32s(vals: &[i32], per_line: usize) -> String {
    let mut out = String::new();
    for (i, v) in vals.iter().enumerate() {
        if i > 0 && i % per_line == 0 {
            out.push('\n');
        }
        out.push_str(&format!("{v},"));
        if (i + 1) % per_line != 0 {
            out.push(' ');
        }
    }
    out.push('\n');
    out
}

/// Tensor classification for the C harness.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Slot {
    Input,
    Output,
    Const,
    Arena(usize),
}

/// Per-model emission state.
struct ModelCtx<'a> {
    spec: &'a ModelSpec,
    model: &'a ParsedModel<'a>,
    input: Vec<i8>,
    in_t: u32,
    out_t: u32,
    arena: ArenaPlan,
    /// slot[tid]
    slots: Vec<Slot>,
    /// flat element count per tensor (i8).
    sizes: Vec<usize>,
    /// const tensors emitted so far (tid -> emitted).
    emitted_consts: Vec<bool>,
    /// collected op call statements (emitted in the run function).
    op_calls: Vec<String>,
}

fn main() {
    let ws = find_workspace_root().expect("workspace root not found");
    let out_dir = ws.join("benchmarks/espnn-baseline/main/zoo_gen");
    std::fs::create_dir_all(&out_dir).expect("create zoo_gen dir");

    let mut common = C::new();
    emit_common(&mut common);

    for spec in MODELS {
        let tflite = std::fs::read(ws.join(spec.tflite)).unwrap_or_else(|e| {
            panic!("read {}: {e}", spec.tflite);
        });
        let model = hematite_tflite::parse(&tflite)
            .unwrap_or_else(|e| panic!("parse {}: {e}", spec.tflite));
        let golden = std::fs::read_to_string(ws.join(spec.golden))
            .unwrap_or_else(|e| panic!("read {}: {e}", spec.golden));
        let input = parse_i8_array(&golden, "INPUT_DATA")
            .unwrap_or_else(|e| panic!("{}: {e}", spec.golden));
        let out = emit_model(spec, &model, input).unwrap_or_else(|e| {
            panic!("emit {}: {e}", spec.tflite);
        });
        let c_path = out_dir.join(format!("zoo_{}.c", spec.name));
        std::fs::write(&c_path, out).expect("write zoo model c");
        eprintln!("emitted {}", c_path.display());
    }

    let mut runner = C::new();
    emit_runner(&mut runner);
    let r_path = out_dir.join("zoo_runner.c");
    std::fs::write(&r_path, runner.buf).expect("write zoo_runner.c");

    let h_path = out_dir.join("zoo_common.h");
    std::fs::write(&h_path, common.buf).expect("write zoo_common.h");
    eprintln!("emitted {} / {}", h_path.display(), r_path.display());
    eprintln!("done: {} models", MODELS.len());
}

// ---------------------------------------------------------------------------
// Workspace discovery
// ---------------------------------------------------------------------------

fn find_workspace_root() -> Result<PathBuf, String> {
    let mut dir = std::env::current_dir().map_err(|e| e.to_string())?;
    loop {
        if dir.join("Cargo.toml").exists() {
            let content = std::fs::read_to_string(dir.join("Cargo.toml")).map_err(|e| e.to_string())?;
            if content.contains("[workspace]") {
                return Ok(dir);
            }
        }
        if !dir.pop() {
            break;
        }
    }
    Err("no workspace root found (walked to filesystem root)".into())
}

// ---------------------------------------------------------------------------
// Shared harness header
// ---------------------------------------------------------------------------

fn emit_common(c: &mut C) {
    c.line("// SPDX-License-Identifier: Apache-2.0");
    c.line("// Generated by tools/espnn_codegen — do not edit by hand.");
    c.blank();
    c.line("#pragma once");
    c.line("#include <stdint.h>");
    c.line("#include <stddef.h>");
    c.line("#include <string.h>");
    c.blank();
    c.line("static inline uint32_t zoo_ccount(void) {");
    c.line("    uint32_t c;");
    c.line("    asm volatile(\"rsr.ccount %0\" : \"=r\"(c));");
    c.line("    return c;");
    c.line("}");
    c.blank();
    c.line("static inline uint32_t zoo_fnv1a(const int8_t *data, int len) {");
    c.line("    uint32_t h = 2166136261u;");
    c.line("    for (int i = 0; i < len; i++) { h ^= (uint32_t)(int8_t)data[i]; h *= 16777619u; }");
    c.line("    return h;");
    c.line("}");
    c.blank();
    c.line("// Scalar requantize: round_doubling_high_mul + rounding_divide_by_pot (TFLM).");
    c.line("static inline int32_t zoo_sadhg(int32_t a, int32_t b) {");
    c.line("    int64_t p = ((int64_t)a * b + (1 << 30)) >> 31;");
    c.line("    if (p > INT32_MAX) p = INT32_MAX;");
    c.line("    if (p < INT32_MIN) p = INT32_MIN;");
    c.line("    return (int32_t)p;");
    c.line("}");
    c.line("static inline int32_t zoo_rdbp(int32_t x, int shift) {");
    c.line("    if (shift > 0) return ((x + (1 << (shift - 1))) >> shift);");
    c.line("    return x << (-shift);");
    c.line("}");
    c.line("static inline int32_t zoo_mbm(int32_t x, int32_t mult, int shift) {");
    c.line("    int32_t r = zoo_sadhg(x, mult);");
    c.line("    if (shift < 0) return zoo_rdbp(r, -shift);");
    c.line("    return r << shift;");
    c.line("}");
    c.blank();
    c.line("// Scalar PAD fallback (esp_nn has no pad kernel). Pads with 0.");
    c.line("static void zoo_pad(const int8_t *src, int8_t *dst, const int32_t in_shape[4],");
    c.line("                    const int32_t pad[4][2], int rank) {");
    c.line("    int in[4] = {1, 1, 1, 1};");
    c.line("    for (int i = 0; i < rank; i++) in[i] = in_shape[i];");
    c.line("    int out[4];");
    c.line("    for (int i = 0; i < 4; i++) out[i] = in[i] + pad[i][0] + pad[i][1];");
    c.line("    int is[4] = {1,1,1,1}, os[4] = {1,1,1,1};");
    c.line("    for (int i = 3; i >= 0; i--) { if (i < 3) { is[i] = is[i+1]*in[i+1]; os[i] = os[i+1]*out[i+1]; } }");
    c.line("    for (int i0 = 0; i0 < out[0]; i0++)");
    c.line("    for (int i1 = 0; i1 < out[1]; i1++)");
    c.line("    for (int i2 = 0; i2 < out[2]; i2++)");
    c.line("    for (int i3 = 0; i3 < out[3]; i3++) {");
    c.line("        int j0 = i0 - pad[0][0], j1 = i1 - pad[1][0], j2 = i2 - pad[2][0], j3 = i3 - pad[3][0];");
    c.line("        int8_t v = 0;");
    c.line("        if (j0 >= 0 && j0 < in[0] && j1 >= 0 && j1 < in[1] && j2 >= 0 && j2 < in[2] && j3 >= 0 && j3 < in[3])");
    c.line("            v = src[j0*is[0] + j1*is[1] + j2*is[2] + j3*is[3]];");
    c.line("        dst[i0*os[0] + i1*os[1] + i2*os[2] + i3*os[3]] = v;");
    c.line("    }");
    c.line("}");
    c.blank();
    c.line("// Scalar TRANSPOSE fallback (esp_nn has no transpose kernel).");
    c.line("static void zoo_transpose(const int8_t *src, int8_t *dst, const int32_t in_shape[4],");
    c.line("                         const int32_t perm[4], int rank) {");
    c.line("    int in[4] = {1, 1, 1, 1};");
    c.line("    for (int i = 0; i < rank; i++) in[i] = in_shape[i];");
    c.line("    int p[4] = {0, 1, 2, 3};");
    c.line("    for (int i = 0; i < rank; i++) p[i] = perm[i];");
    c.line("    int out[4];");
    c.line("    for (int i = 0; i < 4; i++) out[i] = in[p[i]];");
    c.line("    int is[4] = {1,1,1,1}, os[4] = {1,1,1,1};");
    c.line("    for (int i = 3; i >= 0; i--) { if (i < 3) { is[i] = is[i+1]*in[i+1]; os[i] = os[i+1]*out[i+1]; } }");
    c.line("    for (int i0 = 0; i0 < in[0]; i0++)");
    c.line("    for (int i1 = 0; i1 < in[1]; i1++)");
    c.line("    for (int i2 = 0; i2 < in[2]; i2++)");
    c.line("    for (int i3 = 0; i3 < in[3]; i3++) {");
    c.line("        int o = i0*is[0] + i1*is[1] + i2*is[2] + i3*is[3];");
    c.line("        int oi[4] = {i0, i1, i2, i3};");
    c.line("        dst[oi[p[0]]*os[0] + oi[p[1]]*os[1] + oi[p[2]]*os[2] + oi[p[3]]*os[3]] = src[o];");
    c.line("    }");
    c.line("}");
    c.blank();
    c.line("// Scalar SUB fallback (esp_nn has add/mul only).");
    c.line("static void zoo_sub(const int8_t *a, const int8_t *b, int size,");
    c.line("                   int32_t a_off, int32_t b_off, int32_t a_mult, int32_t b_mult,");
    c.line("                   int32_t a_shift, int32_t b_shift, int32_t left_shift,");
    c.line("                   int8_t *out, int32_t out_off, int32_t out_mult, int32_t out_shift,");
    c.line("                   int32_t act_min, int32_t act_max) {");
    c.line("    for (int i = 0; i < size; i++) {");
    c.line("        int32_t x = (int32_t)a[i] + a_off;");
    c.line("        int32_t y = (int32_t)b[i] + b_off;");
    c.line("        x = zoo_sadhg(x, a_mult); y = zoo_sadhg(y, b_mult);");
    c.line("        if (a_shift < 0) x = zoo_rdbp(x, -a_shift); else x <<= a_shift;");
    c.line("        if (b_shift < 0) y = zoo_rdbp(y, -b_shift); else y <<= b_shift;");
    c.line("        int32_t v = ((x - y) << left_shift);");
    c.line("        v = zoo_sadhg(v, out_mult);");
    c.line("        if (out_shift < 0) v = zoo_rdbp(v, -out_shift); else v <<= out_shift;");
    c.line("        v += out_off;");
    c.line("        if (v < act_min) v = act_min;");
    c.line("        if (v > act_max) v = act_max;");
    c.line("        out[i] = (int8_t)v;");
    c.line("    }");
    c.line("}");
    c.blank();
    c.line("// Scalar RELU fallback (in-place).");
    c.line("static void zoo_relu(int8_t *data, int size, int32_t out_zp) {");
    c.line("    int32_t lo = out_zp > -128 ? out_zp : -128;");
    c.line("    for (int i = 0; i < size; i++) { if (data[i] < lo) data[i] = (int8_t)lo; }");
    c.line("}");
    c.blank();
}

// ---------------------------------------------------------------------------
// Per-model emission
// ---------------------------------------------------------------------------

fn emit_model<'a>(
    spec: &'a ModelSpec,
    model: &'a ParsedModel<'a>,
    input: Vec<i8>,
) -> Result<String, String> {
    let in_t = model.inputs()[0];
    let out_t = model.outputs()[0];

    // --- sizes + arena ---
    let n_tensors = model.tensors().len();
    let mut sizes = vec![0usize; n_tensors];
    for (tid, t) in model.tensors().iter().enumerate() {
        sizes[tid] = flat_len(&t.shape)?;
    }

    let mut schedule = Vec::new();
    for op in model.ops() {
        let mut info = OpInfo {
            op_kind: 0,
            input_ids: [u16::MAX; 4],
            input_count: 0,
            output_ids: [u16::MAX; 4],
            output_count: 0,
            in_place: false,
        };
        let mut count = 0u8;
        for &i in &op.inputs {
            if count >= 4 {
                break;
            }
            if i != u32::MAX {
                info.input_ids[count as usize] = i as u16;
                count += 1;
            }
        }
        info.input_count = count;
        count = 0;
        for &o in &op.outputs {
            if count >= 4 {
                break;
            }
            info.output_ids[count as usize] = o as u16;
            count += 1;
        }
        info.output_count = count;
        if !op.outputs.is_empty() && !op.inputs.is_empty() && op.outputs[0] == op.inputs[0] {
            info.in_place = true;
        }
        schedule.push(info);
    }

    let in_ids: Vec<u16> = model.inputs().iter().map(|&x| x as u16).collect();
    let out_ids: Vec<u16> = model.outputs().iter().map(|&x| x as u16).collect();
    let arena = liveness_plan(&schedule, &sizes, &in_ids, &out_ids, usize::MAX / 4, None)
        .map_err(|e| format!("liveness_plan: {e:?}"))?;

    // --- classify tensors ---
    let mut slots = vec![Slot::Arena(0); n_tensors];
    for tid in 0..n_tensors {
        let t = &model.tensors()[tid];
        let is_in = model.inputs().contains(&(tid as u32));
        let is_out = model.outputs().contains(&(tid as u32));
        let has_const = model.buffer_data(t).is_some();
        if is_in {
            slots[tid] = Slot::Input;
        } else if is_out {
            slots[tid] = Slot::Output;
        } else if has_const && sizes[tid] > 0 {
            slots[tid] = Slot::Const;
        } else if tid < arena.offsets.len() && arena.offsets[tid] != OFFSET_NONE {
            slots[tid] = Slot::Arena(arena.offsets[tid]);
        } else {
            return Err(format!(
                "tensor {tid} ({}) has no arena slot (offset NONE) and is not const",
                t.name
            ));
        }
    }

    let mut ctx = ModelCtx {
        spec,
        model,
        input,
        in_t,
        out_t,
        arena,
        slots,
        sizes,
        emitted_consts: vec![false; n_tensors],
        op_calls: Vec::new(),
    };

    let c = build_model_c(&mut ctx)?;
    Ok(c)
}

fn align16(x: usize) -> usize {
    (x + 15) & !15
}

fn build_model_c(ctx: &mut ModelCtx) -> Result<String, String> {
    let mut c = C::new();
    let n = ctx.spec.name;

    c.line("// SPDX-License-Identifier: Apache-2.0");
    c.line(&format!(
        "// Generated by tools/espnn_codegen from {}",
        ctx.spec.tflite
    ));
    c.line("#include <stdint.h>");
    c.line("#include <string.h>");
    c.line("#include \"zoo_common.h\"");
    c.line("#include \"esp_nn.h\"");
    c.blank();

    let in_len = ctx.sizes[ctx.in_t as usize];
    let out_len = ctx.sizes[ctx.out_t as usize];
    let in_off = 0usize;
    let out_off = align16(in_len);
    let arena_base = out_off + align16(out_len);
    let mem_bytes = arena_base + ctx.arena.peak_arena_bytes;

    c.line(&format!("// liveness arena: peak {} bytes", ctx.arena.peak_arena_bytes));
    c.line(&format!(
        "const uint32_t ZOO_{}_MEM_BYTES = {};",
        n.to_uppercase(),
        mem_bytes
    ));
    c.line(&format!(
        "const uint32_t ZOO_{}_IN_LEN = {};",
        n.to_uppercase(),
        in_len
    ));
    c.line(&format!(
        "const uint32_t ZOO_{}_OUT_LEN = {};",
        n.to_uppercase(),
        out_len
    ));
    c.blank();

    // Input (golden).
    c.line(&format!(
        "static const int8_t __attribute__((aligned(16))) {}_input[{}] = {{",
        n,
        in_len
    ));
    c.push(&c_bytes(&to_i8s(&ctx.input), 16));
    c.line("};");
    c.blank();

    // Emit each op.
    for (i, op) in ctx.model.ops().iter().enumerate() {
        emit_op(ctx, &mut c, i, op)?;
    }

    // Run function.
    c.blank();
    c.line(&format!("void zoo_{n}_run(int8_t *mem, int8_t *scratch) {{"));
    c.line(&format!(
        "    int8_t *t_in = mem + {in_off}; int8_t *t_out = mem + {out_off}; int8_t *t_arena = mem + {arena_base};"
    ));
    c.line("    (void)t_in; (void)t_out; (void)t_arena; (void)scratch;");
    c.blank();
    c.line(&format!("    memcpy(t_in, {n}_input, {in_len});"));
    c.blank();
    for s in &ctx.op_calls {
        c.line(&format!("    {s}"));
    }
    c.line("}");
    c.blank();

    Ok(c.buf)
}

fn to_i8s(v: &[i8]) -> Vec<u8> {
    v.iter().map(|&x| x as u8).collect()
}

fn emit_op(
    ctx: &mut ModelCtx,
    c: &mut C,
    i: usize,
    op: &ParsedOp,
) -> Result<(), String> {
    match op.builtin_code {
        3 => emit_conv(ctx, c, i, op),
        4 => emit_depthwise(ctx, c, i, op),
        9 => emit_fc(ctx, c, i, op),
        1 => emit_pool(ctx, c, i, op, true),
        17 => emit_pool(ctx, c, i, op, false),
        25 => emit_softmax(ctx, c, i, op),
        0 => emit_add(ctx, c, i, op),
        41 => emit_sub(ctx, c, i, op),
        18 => emit_mul(ctx, c, i, op),
        40 => emit_mean(ctx, c, i, op),
        19 => emit_relu(ctx, c, i, op),
        21 => emit_relu6(ctx, c, i, op),
        22 => emit_reshape(ctx, c, i, op),
        34 => emit_pad_op(ctx, c, i, op),
        39 => emit_transpose_op(ctx, c, i, op),
        code => Err(format!("op {i}: unsupported builtin_code {code}")),
    }
}

fn ref_slot<'a>(ctx: &'a ModelCtx, tid: u32) -> String {
    match ctx.slots[tid as usize] {
        Slot::Input => format!("t_in"),
        Slot::Output => format!("t_out"),
        Slot::Const => format!("const_{}", tid),
        Slot::Arena(off) => format!("(t_arena + {off})"),
    }
}

fn len_of(ctx: &ModelCtx, tid: u32) -> usize {
    ctx.sizes[tid as usize]
}

fn emit_const_tensor(ctx: &mut ModelCtx, c: &mut C, tid: u32, is_i32: bool) -> Result<(), String> {
    if ctx.emitted_consts[tid as usize] {
        return Ok(());
    }
    ctx.emitted_consts[tid as usize] = true;
    let t = ctx
        .model
        .tensor_by_index(tid as usize)
        .ok_or_else(|| format!("tensor {tid} missing"))?;
    let data = ctx
        .model
        .buffer_data(t)
        .ok_or_else(|| format!("tensor {tid} not const"))?;
    let name = format!("const_{}", tid);
    if is_i32 {
        // i32 buffer (e.g. paddings, perm, bias): little-endian i32s.
        let vals: Vec<i32> = data
            .chunks_exact(4)
            .map(|ch| i32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]))
            .collect();
        c.line(&format!(
            "static const int32_t __attribute__((aligned(16))) {name}[{}] = {{",
            vals.len()
        ));
        c.push(&c_i32s(&vals, 8));
        c.line("};");
    } else {
        c.line(&format!(
            "static const int8_t __attribute__((aligned(16))) {name}[{}] = {{",
            data.len()
        ));
        c.push(&c_bytes(data, 16));
        c.line("};");
    }
    c.blank();
    Ok(())
}

fn emit_conv(ctx: &mut ModelCtx, c: &mut C, i: usize, op: &ParsedOp) -> Result<(), String> {
    let (padding, sw, sh, dw, dh, fused) = match op.options.as_ref() {
        Some(ParsedOptions::Conv2D {
            padding,
            stride_w,
            stride_h,
            dilation_w,
            dilation_h,
            fused_activation,
        }) => (
            *padding,
            *stride_w,
            *stride_h,
            *dilation_w,
            *dilation_h,
            *fused_activation,
        ),
        other => return Err(format!("op {i}: expected Conv2D options, got {other:?}")),
    };
    let in_t = op.inputs[0];
    let w_t = op.inputs[1];
    let b_t = op.inputs.get(2).copied().unwrap_or(u32::MAX);
    let out_t = op.outputs[0];
    let input = ctx
        .model
        .tensor_by_index(in_t as usize)
        .ok_or("conv input missing")?;
    let weights = ctx
        .model
        .tensor_by_index(w_t as usize)
        .ok_or("conv weights missing")?;
    let output = ctx
        .model
        .tensor_by_index(out_t as usize)
        .ok_or("conv output missing")?;

    emit_const_tensor(ctx, c, w_t, false)?;
    if b_t != u32::MAX {
        emit_const_tensor(ctx, c, b_t, true)?;
    }
    emit_const_tensor(ctx, c, w_t, false)?;

    let in_s4 = shape4(&input.shape)?;
    let f_raw = shape4(&weights.shape)?;
    let out_s4 = shape4(&output.shape)?;
    let out_c = f_raw[0] as usize;
    let (ih, iw, ic) = (in_s4[1], in_s4[2], in_s4[3]);
    let (kh, kw, _kic) = (f_raw[1], f_raw[2], f_raw[3]);
    let (oh, ow) = (out_s4[1], out_s4[2]);
    let q = conv_quant(input, weights, output, out_c, fused)?;
    if q.weights_offset != 0 {
        return Err(format!(
            "op {i}: conv weights zero point {} != 0 (esp_nn assumes 0)",
            q.weights_offset
        ));
    }
    let kh_eff = (kh - 1) * dh + 1;
    let kw_eff = (kw - 1) * dw + 1;
    let (pad_h, pad_w) = if padding == 0 {
        (
            ((oh - 1) * sh + kh_eff - ih).max(0) / 2,
            ((ow - 1) * sw + kw_eff - iw).max(0) / 2,
        )
    } else {
        (0, 0)
    };
    let m_name = format!("qmult_{i}");
    let s_name = format!("qshift_{i}");
    c.line(&format!(
        "static const int32_t __attribute__((aligned(16))) {m_name}[{out_c}] = {{"
    ));
    c.push(&c_i32s(&q.multipliers, 8));
    c.line("};");
    c.line(&format!(
        "static const int32_t __attribute__((aligned(16))) {s_name}[{out_c}] = {{"
    ));
    c.push(&c_i32s(&q.shifts, 8));
    c.line("};");
    c.blank();

    let src = ref_slot(ctx, in_t);
    let dst = ref_slot(ctx, out_t);
    let wref = ref_slot(ctx, w_t);
    let bref = if b_t != u32::MAX {
        ref_slot(ctx, b_t)
    } else {
        format!("NULL")
    };
    ctx.op_calls.push(format!(
        "{{ data_dims_t din = {{.width={iw},.height={ih},.channels={ic},.extra=1}}; \
          data_dims_t dfil = {{.width={kw},.height={kh},.channels={ic},.extra={out_c}}}; \
          data_dims_t dout = {{.width={ow},.height={oh},.channels={out_c},.extra=1}}; \
          conv_params_t cp = {{.in_offset={},.out_offset={},.stride={{.width={sw},.height={sh}}},.padding={{.width={pad_w},.height={pad_h}}},.dilation={{.width={dw},.height={dh}}},.activation={{.min={},.max={}}}}}; \
          quant_data_t qd = {{.shift={s_name},.mult={m_name}}}; \
          esp_nn_set_conv_scratch_buf(scratch); \
          esp_nn_conv_s8(&din, {src}, &dfil, {wref}, {bref}, &dout, {dst}, &cp, &qd); }}",
        q.input_offset, q.output_offset, q.act_min, q.act_max
    ));
    Ok(())
}

fn emit_depthwise(ctx: &mut ModelCtx, c: &mut C, i: usize, op: &ParsedOp) -> Result<(), String> {
    let (padding, sw, sh, dm, dw, dh, fused) = match op.options.as_ref() {
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
    if dm != 1 && dm != 4 && dm != 8 {
        return Err(format!(
            "op {i}: depth_multiplier {dm} unsupported by esp_nn S3 depthwise (supports 1/4/8)"
        ));
    }
    let in_t = op.inputs[0];
    let w_t = op.inputs[1];
    let b_t = op.inputs.get(2).copied().unwrap_or(u32::MAX);
    let out_t = op.outputs[0];
    let input = ctx
        .model
        .tensor_by_index(in_t as usize)
        .ok_or("dw input missing")?;
    let weights = ctx
        .model
        .tensor_by_index(w_t as usize)
        .ok_or("dw weights missing")?;
    let output = ctx
        .model
        .tensor_by_index(out_t as usize)
        .ok_or("dw output missing")?;

    emit_const_tensor(ctx, c, w_t, false)?;
    if b_t != u32::MAX {
        emit_const_tensor(ctx, c, b_t, true)?;
    }

    let in_s4 = shape4(&input.shape)?;
    let f_raw = shape4(&weights.shape)?;
    let out_s4 = shape4(&output.shape)?;
    let out_c = f_raw[3] as usize;
    let (ih, iw, ic) = (in_s4[1], in_s4[2], in_s4[3]);
    let (kh, kw) = (f_raw[1], f_raw[2]);
    let (oh, ow) = (out_s4[1], out_s4[2]);
    let q = conv_quant(input, weights, output, out_c, fused)?;
    if q.weights_offset != 0 {
        return Err(format!(
            "op {i}: dw weights zero point {} != 0 (esp_nn assumes 0)",
            q.weights_offset
        ));
    }
    let kh_eff = (kh - 1) * dh + 1;
    let kw_eff = (kw - 1) * dw + 1;
    let (pad_h, pad_w) = if padding == 0 {
        (
            ((oh - 1) * sh + kh_eff - ih).max(0) / 2,
            ((ow - 1) * sw + kw_eff - iw).max(0) / 2,
        )
    } else {
        (0, 0)
    };
    let m_name = format!("qmult_{i}");
    let s_name = format!("qshift_{i}");
    c.line(&format!(
        "static const int32_t __attribute__((aligned(16))) {m_name}[{out_c}] = {{"
    ));
    c.push(&c_i32s(&q.multipliers, 8));
    c.line("};");
    c.line(&format!(
        "static const int32_t __attribute__((aligned(16))) {s_name}[{out_c}] = {{"
    ));
    c.push(&c_i32s(&q.shifts, 8));
    c.line("};");
    c.blank();

    let src = ref_slot(ctx, in_t);
    let dst = ref_slot(ctx, out_t);
    let wref = ref_slot(ctx, w_t);
    let bref = if b_t != u32::MAX {
        ref_slot(ctx, b_t)
    } else {
        format!("NULL")
    };
    ctx.op_calls.push(format!(
        "{{ data_dims_t din = {{.width={iw},.height={ih},.channels={ic},.extra=1}}; \
          data_dims_t dfil = {{.width={kw},.height={kh},.channels={ic},.extra={out_c}}}; \
          data_dims_t dout = {{.width={ow},.height={oh},.channels={out_c},.extra=1}}; \
          dw_conv_params_t dp = {{.in_offset={},.out_offset={},.ch_mult={dm},.stride={{.width={sw},.height={sh}}},.padding={{.width={pad_w},.height={pad_h}}},.dilation={{.width={dw},.height={dh}}},.activation={{.min={},.max={}}}}}; \
          quant_data_t qd = {{.shift={s_name},.mult={m_name}}}; \
          esp_nn_set_depthwise_conv_scratch_buf(scratch); \
          esp_nn_depthwise_conv_s8(&din, {src}, &dfil, {wref}, {bref}, &dout, {dst}, &dp, &qd); }}",
        q.input_offset, q.output_offset, q.act_min, q.act_max
    ));
    Ok(())
}

fn emit_fc(ctx: &mut ModelCtx, c: &mut C, i: usize, op: &ParsedOp) -> Result<(), String> {
    let fused = match op.options.as_ref() {
        Some(ParsedOptions::FullyConnected {
            fused_activation, ..
        }) => *fused_activation,
        other => return Err(format!("op {i}: expected FullyConnected options, got {other:?}")),
    };
    let in_t = op.inputs[0];
    let w_t = op.inputs[1];
    let b_t = op.inputs.get(2).copied().unwrap_or(u32::MAX);
    let out_t = op.outputs[0];
    let input = ctx
        .model
        .tensor_by_index(in_t as usize)
        .ok_or("fc input missing")?;
    let weights = ctx
        .model
        .tensor_by_index(w_t as usize)
        .ok_or("fc weights missing")?;
    let output = ctx
        .model
        .tensor_by_index(out_t as usize)
        .ok_or("fc output missing")?;

    emit_const_tensor(ctx, c, w_t, false)?;
    if b_t != u32::MAX {
        emit_const_tensor(ctx, c, b_t, true)?;
    }

    let row_len = flat_len(&input.shape)?;
    let out_c = flat_len(&output.shape)?;
    let q = conv_quant(input, weights, output, out_c, fused)?;
    if q.weights_offset != 0 {
        return Err(format!(
            "op {i}: fc weights zero point {} != 0 (esp_nn assumes 0)",
            q.weights_offset
        ));
    }
    let m_name = format!("qmult_{i}");
    let s_name = format!("qshift_{i}");
    c.line(&format!(
        "static const int32_t __attribute__((aligned(16))) {m_name}[{out_c}] = {{"
    ));
    c.push(&c_i32s(&q.multipliers, 8));
    c.line("};");
    c.line(&format!(
        "static const int32_t __attribute__((aligned(16))) {s_name}[{out_c}] = {{"
    ));
    c.push(&c_i32s(&q.shifts, 8));
    c.line("};");
    c.blank();

    let src = ref_slot(ctx, in_t);
    let dst = ref_slot(ctx, out_t);
    let wref = ref_slot(ctx, w_t);
    let bref = if b_t != u32::MAX {
        ref_slot(ctx, b_t)
    } else {
        format!("NULL")
    };
    ctx.op_calls.push(format!(
        "esp_nn_fully_connected_per_ch_s8({src}, {}, {row_len}, {wref}, 0, {bref}, {dst}, {out_c}, {}, {s_name}, {m_name}, {}, {});",
        q.input_offset, q.output_offset, q.act_min, q.act_max
    ));
    Ok(())
}

fn emit_pool(ctx: &mut ModelCtx, c: &mut C, i: usize, op: &ParsedOp, is_avg: bool) -> Result<(), String> {
    let (padding, sw, sh, fw, fh, fused) = match op.options.as_ref() {
        Some(ParsedOptions::Pool2D {
            padding,
            stride_w,
            stride_h,
            filter_w,
            filter_h,
            fused_activation,
        }) => (
            *padding,
            *stride_w,
            *stride_h,
            *filter_w,
            *filter_h,
            *fused_activation,
        ),
        other => return Err(format!("op {i}: expected Pool2D options, got {other:?}")),
    };
    let in_t = op.inputs[0];
    let out_t = op.outputs[0];
    let input = ctx
        .model
        .tensor_by_index(in_t as usize)
        .ok_or("pool input missing")?;
    let output = ctx
        .model
        .tensor_by_index(out_t as usize)
        .ok_or("pool output missing")?;
    let in_s4 = shape4(&input.shape)?;
    let out_s4 = shape4(&output.shape)?;
    let (ih, iw, ic) = (in_s4[1], in_s4[2], in_s4[3]);
    let (oh, ow) = (out_s4[1], out_s4[2]);
    let out_scale = tensor_scale(output)?;
    let out_zp = tensor_zp(output)?;
    let (act_min, act_max) = act_range(fused, out_scale, out_zp);
    let (pad_h, pad_w) = if padding == 0 {
        (
            ((oh - 1) * sh + fh - ih).max(0) / 2,
            ((ow - 1) * sw + fw - iw).max(0) / 2,
        )
    } else {
        (0, 0)
    };
    let src = ref_slot(ctx, in_t);
    let dst = ref_slot(ctx, out_t);
    let kernel = if is_avg { "esp_nn_avg_pool_s8" } else { "esp_nn_max_pool_s8" };
    ctx.op_calls.push(format!(
        "{kernel}({src}, {iw}, {ih}, {dst}, {ow}, {oh}, {sw}, {sh}, {fw}, {fh}, {pad_w}, {pad_h}, {act_min}, {act_max}, {ic});"
    ));
    Ok(())
}

fn emit_softmax(ctx: &mut ModelCtx, c: &mut C, i: usize, op: &ParsedOp) -> Result<(), String> {
    let beta = match op.options.as_ref() {
        Some(ParsedOptions::Softmax { beta }) => *beta,
        other => return Err(format!("op {i}: expected Softmax options, got {other:?}")),
    };
    if beta != 1.0 {
        return Err(format!("op {i}: softmax beta {beta} != 1.0"));
    }
    let in_t = op.inputs[0];
    let out_t = op.outputs[0];
    let input = ctx
        .model
        .tensor_by_index(in_t as usize)
        .ok_or("softmax input missing")?;
    let in_s4 = shape4(&input.shape)?;
    let row_size = in_s4[3] as usize;
    let num_rows = flat_len(&input.shape)? / row_size;
    let in_scale = tensor_scale(input)?;
    let (m, s) = quantize_multiplier(in_scale);
    let shift = 26 + s;
    let src = ref_slot(ctx, in_t);
    let dst = ref_slot(ctx, out_t);
    ctx.op_calls.push(format!(
        "{{ esp_nn_set_softmax_scratch_buf(scratch); esp_nn_softmax_s8({src}, {num_rows}, {row_size}, {m}, {shift}, -128, {dst}); }}"
    ));
    Ok(())
}

fn emit_add(ctx: &mut ModelCtx, c: &mut C, i: usize, op: &ParsedOp) -> Result<(), String> {
    emit_addsub(ctx, c, i, op, false)
}

fn emit_sub(ctx: &mut ModelCtx, c: &mut C, i: usize, op: &ParsedOp) -> Result<(), String> {
    emit_addsub(ctx, c, i, op, true)
}

fn emit_addsub(ctx: &mut ModelCtx, c: &mut C, i: usize, op: &ParsedOp, is_sub: bool) -> Result<(), String> {
    let fused = match op.options.as_ref() {
        Some(ParsedOptions::Add {
            fused_activation, ..
        })
        | Some(ParsedOptions::Sub {
            fused_activation, ..
        }) => *fused_activation,
        other => return Err(format!("op {i}: expected Add/Sub options, got {other:?}")),
    };
    let in1_t = op.inputs[0];
    let in2_t = op.inputs[1];
    let out_t = op.outputs[0];
    let in1 = ctx
        .model
        .tensor_by_index(in1_t as usize)
        .ok_or("add input1 missing")?;
    let in2 = ctx
        .model
        .tensor_by_index(in2_t as usize)
        .ok_or("add input2 missing")?;
    let output = ctx
        .model
        .tensor_by_index(out_t as usize)
        .ok_or("add output missing")?;
    let in1_scale = tensor_scale(in1)?;
    let in2_scale = tensor_scale(in2)?;
    let out_scale = tensor_scale(output)?;
    let out_zp = tensor_zp(output)?;
    let twice_max = 2.0 * in1_scale.max(in2_scale);
    let ls = 20;
    let (i1m, i1s) = quantize_multiplier(in1_scale / twice_max);
    let (i2m, i2s) = quantize_multiplier(in2_scale / twice_max);
    let (om, os) = quantize_multiplier(twice_max / ((1 << 20) as f64 * out_scale));
    let in1_off = -tensor_zp(in1)?;
    let in2_off = -tensor_zp(in2)?;
    let out_off = out_zp;
    let (act_min, act_max) = act_range(fused, out_scale, out_zp);
    let size = len_of(ctx, in1_t);
    let src1 = ref_slot(ctx, in1_t);
    let src2 = ref_slot(ctx, in2_t);
    let dst = ref_slot(ctx, out_t);
    if is_sub {
        ctx.op_calls.push(format!(
            "zoo_sub({src1}, {src2}, {size}, {in1_off}, {in2_off}, {i1m}, {i2m}, {i1s}, {i2s}, {ls}, {dst}, {out_off}, {om}, {os}, {act_min}, {act_max});"
        ));
    } else {
        ctx.op_calls.push(format!(
            "esp_nn_add_elementwise_s8({src1}, {src2}, {in1_off}, {in2_off}, {i1m}, {i2m}, {i1s}, {i2s}, {ls}, {dst}, {out_off}, {om}, {os}, {act_min}, {act_max}, {size});"
        ));
    }
    Ok(())
}

fn emit_mul(ctx: &mut ModelCtx, c: &mut C, i: usize, op: &ParsedOp) -> Result<(), String> {
    let fused = match op.options.as_ref() {
        Some(ParsedOptions::Mul {
            fused_activation, ..
        }) => *fused_activation,
        other => return Err(format!("op {i}: expected Mul options, got {other:?}")),
    };
    let in1_t = op.inputs[0];
    let in2_t = op.inputs[1];
    let out_t = op.outputs[0];
    let in1 = ctx
        .model
        .tensor_by_index(in1_t as usize)
        .ok_or("mul input1 missing")?;
    let in2 = ctx
        .model
        .tensor_by_index(in2_t as usize)
        .ok_or("mul input2 missing")?;
    let output = ctx
        .model
        .tensor_by_index(out_t as usize)
        .ok_or("mul output missing")?;
    let in1_scale = tensor_scale(in1)?;
    let in2_scale = tensor_scale(in2)?;
    let out_scale = tensor_scale(output)?;
    let out_zp = tensor_zp(output)?;
    let (om, os) = quantize_multiplier(in1_scale * in2_scale / out_scale);
    let in1_off = -tensor_zp(in1)?;
    let in2_off = -tensor_zp(in2)?;
    let out_off = out_zp;
    let (act_min, act_max) = act_range(fused, out_scale, out_zp);
    let size = len_of(ctx, in1_t);
    let src1 = ref_slot(ctx, in1_t);
    let src2 = ref_slot(ctx, in2_t);
    let dst = ref_slot(ctx, out_t);
    ctx.op_calls.push(format!(
        "esp_nn_mul_elementwise_s8({src1}, {src2}, {in1_off}, {in2_off}, {dst}, {out_off}, {om}, {os}, {act_min}, {act_max}, {size});"
    ));
    Ok(())
}

fn emit_mean(ctx: &mut ModelCtx, c: &mut C, i: usize, op: &ParsedOp) -> Result<(), String> {
    let (axis, keep_dims) = match op.options.as_ref() {
        Some(ParsedOptions::Mean { axis, keep_dims }) => (axis.clone(), *keep_dims),
        other => return Err(format!("op {i}: expected Mean options, got {other:?}")),
    };
    let in_t = op.inputs[0];
    let out_t = op.outputs[0];
    let input = ctx
        .model
        .tensor_by_index(in_t as usize)
        .ok_or("mean input missing")?;
    let output = ctx
        .model
        .tensor_by_index(out_t as usize)
        .ok_or("mean output missing")?;
    let in_s4 = shape4(&input.shape)?;
    let out_s4 = shape4(&output.shape)?;
    let in_scale = tensor_scale(input)?;
    let out_scale = tensor_scale(output)?;
    let out_zp = tensor_zp(output)?;
    let in_zp = tensor_zp(input)?;
    let (mult, shift) = quantize_multiplier(in_scale / out_scale);
    // Count = product of reduced dims. Only NHWC H,W reduction supported
    // for esp_nn_mean_nhwc_s8 (height*width per channel).
    let (h, w, ch) = (in_s4[1], in_s4[2], in_s4[3]);
    let (oh, ow) = (out_s4[1], out_s4[2]);
    let _ = keep_dims;
    let reduced_hw = oh < h || ow < w;
    if !reduced_hw && axis.len() == 1 && axis[0] == 3 {
        // channel-wise mean: not supported via esp_nn_mean_nhwc_s8
        return Err(format!("op {i}: channel-axis mean unsupported"));
    }
    let count = if axis.contains(&1) && axis.contains(&2) {
        (h as u64) * (w as u64)
    } else if axis.contains(&1) {
        h as u64
    } else if axis.contains(&2) {
        w as u64
    } else {
        return Err(format!("op {i}: mean over axis {axis:?} not supported"));
    };
    let (mean_mult, mean_shift) = mean_adapt(mult, shift, count);
    let src = ref_slot(ctx, in_t);
    let dst = ref_slot(ctx, out_t);
    ctx.op_calls.push(format!(
        "esp_nn_mean_nhwc_s8({src}, {dst}, {h}, {w}, {ch}, {in_zp}, {out_zp}, {mean_mult}, {mean_shift});"
    ));
    Ok(())
}

fn emit_relu(ctx: &mut ModelCtx, c: &mut C, i: usize, op: &ParsedOp) -> Result<(), String> {
    let in_t = op.inputs[0];
    let out_t = op.outputs[0];
    let input = ctx
        .model
        .tensor_by_index(in_t as usize)
        .ok_or("relu input missing")?;
    let out_zp = tensor_zp(input)?;
    let size = len_of(ctx, in_t);
    let src = ref_slot(ctx, in_t);
    let dst = ref_slot(ctx, out_t);
    if in_t == out_t {
        ctx.op_calls.push(format!("zoo_relu({src}, {size}, {out_zp});"));
    } else {
        ctx.op_calls.push(format!(
            "{{ memcpy({dst}, {src}, {size}); zoo_relu({dst}, {size}, {out_zp}); }}"
        ));
    }
    Ok(())
}

fn emit_relu6(ctx: &mut ModelCtx, c: &mut C, i: usize, op: &ParsedOp) -> Result<(), String> {
    let in_t = op.inputs[0];
    let out_t = op.outputs[0];
    let size = len_of(ctx, in_t);
    let src = ref_slot(ctx, in_t);
    let dst = ref_slot(ctx, out_t);
    if in_t == out_t {
        ctx.op_calls.push(format!("esp_nn_relu6_s8({src}, {size});"));
    } else {
        ctx.op_calls.push(format!(
            "{{ memcpy({dst}, {src}, {size}); esp_nn_relu6_s8({dst}, {size}); }}"
        ));
    }
    Ok(())
}

fn emit_reshape(ctx: &mut ModelCtx, c: &mut C, i: usize, op: &ParsedOp) -> Result<(), String> {
    let in_t = op.inputs[0];
    let out_t = op.outputs[0];
    let in_len = len_of(ctx, in_t);
    let out_len = len_of(ctx, out_t);
    let src = ref_slot(ctx, in_t);
    let dst = ref_slot(ctx, out_t);
    if in_len != out_len {
        return Err(format!(
            "op {i}: reshape {in_len} -> {out_len} element count mismatch"
        ));
    }
    ctx.op_calls.push(format!("memcpy({dst}, {src}, {in_len});"));
    Ok(())
}

fn emit_pad_op(ctx: &mut ModelCtx, c: &mut C, i: usize, op: &ParsedOp) -> Result<(), String> {
    let in_t = op.inputs[0];
    let pad_t = op.inputs[1];
    let out_t = op.outputs[0];
    let input = ctx
        .model
        .tensor_by_index(in_t as usize)
        .ok_or("pad input missing")?;
    let pad_tensor = ctx
        .model
        .tensor_by_index(pad_t as usize)
        .ok_or("pad amounts missing")?;
    let out_shape = ctx
        .model
        .tensor_by_index(out_t as usize)
        .ok_or("pad output missing")?
        .shape
        .clone();
    emit_const_tensor(ctx, c, pad_t, true)?;
    let rank = input.shape.len();
    let padref = ref_slot(ctx, pad_t);
    let src = ref_slot(ctx, in_t);
    let dst = ref_slot(ctx, out_t);
    let in_shape = shape4(&input.shape)?;
    let out_s4 = shape4(&out_shape)?;
    let _ = out_s4;
    // The paddings buffer is int32 [rank][2].
    ctx.op_calls.push(format!(
        "{{ int32_t in_sh[4] = {{{}, {}, {}, {}}}; zoo_pad({src}, {dst}, in_sh, (const int32_t(*)[2]){padref}, {rank}); }}",
        in_shape[0], in_shape[1], in_shape[2], in_shape[3]
    ));
    Ok(())
}

fn emit_transpose_op(ctx: &mut ModelCtx, c: &mut C, i: usize, op: &ParsedOp) -> Result<(), String> {
    let in_t = op.inputs[0];
    let perm_t = op.inputs[1];
    let out_t = op.outputs[0];
    let input = ctx
        .model
        .tensor_by_index(in_t as usize)
        .ok_or("transpose input missing")?;
    emit_const_tensor(ctx, c, perm_t, true)?;
    let rank = input.shape.len();
    let permref = ref_slot(ctx, perm_t);
    let src = ref_slot(ctx, in_t);
    let dst = ref_slot(ctx, out_t);
    let in_shape = shape4(&input.shape)?;
    ctx.op_calls.push(format!(
        "{{ int32_t in_sh[4] = {{{}, {}, {}, {}}}; zoo_transpose({src}, {dst}, in_sh, {permref}, {rank}); }}",
        in_shape[0], in_shape[1], in_shape[2], in_shape[3]
    ));
    Ok(())
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

fn emit_runner(c: &mut C) {
    c.line("// SPDX-License-Identifier: Apache-2.0");
    c.line("// Generated by tools/espnn_codegen — do not edit by hand.");
    c.blank();
    c.line("#include <stdio.h>");
    c.line("#include <stdint.h>");
    c.line("#include <stdlib.h>");
    c.line("#include <string.h>");
    c.line("#include <stdbool.h>");
    c.line("#include \"esp_heap_caps.h\"");
    c.line("#include \"zoo_common.h\"");
    c.blank();
    for m in MODELS {
        let n = m.name;
        let N = n.to_uppercase();
        c.line(&format!("void zoo_{n}_run(int8_t *mem, int8_t *scratch);"));
        c.line(&format!("extern const uint32_t ZOO_{N}_MEM_BYTES, ZOO_{N}_IN_LEN, ZOO_{N}_OUT_LEN;"));
    }
    c.blank();
    c.line("#define TIMED_RUNS 10");
    c.blank();
    c.line("static void zoo_run_bench(const char *label, void (*fn)(int8_t *, int8_t *),");
    c.line("                          int8_t *mem, int8_t *scratch, uint32_t out_off,");
    c.line("                          uint32_t out_len,");
    c.line("                          uint32_t *min_cycles, uint32_t *median_cycles, uint32_t *fnv) {");
    c.line("    uint32_t runs[TIMED_RUNS];");
    c.line("    fn(mem, scratch); // warmup");
    c.line("    for (int r = 0; r < TIMED_RUNS; r++) {");
    c.line("        uint32_t t0 = zoo_ccount();");
    c.line("        fn(mem, scratch);");
    c.line("        uint32_t t1 = zoo_ccount();");
    c.line("        runs[r] = t1 - t0;");
    c.line("    }");
    c.line("    for (int i = 1; i < TIMED_RUNS; i++) {");
    c.line("        uint32_t v = runs[i]; int j = i - 1;");
    c.line("        while (j >= 0 && runs[j] > v) { runs[j + 1] = runs[j]; j--; }");
    c.line("        runs[j + 1] = v;");
    c.line("    }");
    c.line("    *min_cycles = runs[0];");
    c.line("    *median_cycles = (runs[TIMED_RUNS / 2 - 1] + runs[TIMED_RUNS / 2]) / 2;");
    c.line("    *fnv = zoo_fnv1a(mem + out_off, (int)out_len);");
    c.line("    (void)label;");
    c.line("}");
    c.blank();
    c.line("void zoo_run_all(void) {");
    c.line("    printf(\"=== ESP-NN ZOO MODEL BENCH (real weights + golden inputs) ===\\n\");");
    for m in MODELS {
        let n = m.name;
        let N = n.to_uppercase();
        c.line(&format!("    do {{"));
        c.line(&format!(
            "        uint32_t mem_bytes = ZOO_{N}_MEM_BYTES; uint32_t scratch_bytes = {};",
            m.scratch_bytes
        ));
        c.line(&format!(
            "        uint32_t out_off = (ZOO_{N}_IN_LEN + 15) & ~15u;"
        ));
        c.line(&format!(
            "        bool big = (mem_bytes + scratch_bytes) > 300*1024;"
        ));
        c.line(&format!(
            "        int8_t *mem = big ? (int8_t*)heap_caps_malloc(mem_bytes + scratch_bytes, MALLOC_CAP_SPIRAM) : (int8_t*)malloc(mem_bytes + scratch_bytes);"
        ));
        c.line(&format!(
            "        if (!mem) {{ printf(\"zoo {n}: OOM (%u)\\n\", (unsigned)(mem_bytes + scratch_bytes)); break; }}"
        ));
        c.line(&format!("        int8_t *scr = mem + mem_bytes;"));
        c.line(&format!("        uint32_t minc = 0, medc = 0, fnv = 0;"));
        c.line(&format!(
            "        zoo_run_bench(\"{n}\", zoo_{n}_run, mem, scr, out_off, ZOO_{N}_OUT_LEN, &minc, &medc, &fnv);"
        ));
        c.line(&format!(
            "        printf(\"| zoo {n} | %s | %u/%u | out_fnv=0x%08x |\\n\", big ? \"PSRAM\" : \"SRAM\", (unsigned)minc, (unsigned)medc, (unsigned)fnv);"
        ));
        c.line(&format!("        free(mem);"));
        c.line(&format!("    }} while (0);"));
    }
    c.line("    printf(\"=== ESP-NN ZOO MODEL BENCH DONE ===\\n\");");
    c.line("}");
    c.blank();
}
