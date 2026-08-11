//! Host-side isolation: does esp_nn's fully_connected math match hematite-ref
//! (TFLM-exact) for the anomaly_detect model?  Runs the 10-fc chain both ways
//! and compares per-layer outputs.
//!
//! Usage: `cargo run -p espnn-codegen --bin anomaly_diff`
//!
//! Reimplements the esp_nn s8-fast path + s16 asm path in Rust and compares
//! against `hematite_ref::fully_connected` per layer.

use hematite_core::op_params::FullyConnectedParams;
use hematite_core::KernelError;
use hematite_tflite::parse;

// ── TFLM quant math (mirrors generate.rs / espnn_codegen main.rs) ──────────

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

fn flat_len(shape: &[i32]) -> usize {
    shape.iter().map(|&d| d as usize).product()
}

fn tensor_scale(t: &hematite_tflite::ParsedTensor) -> f64 {
    f64::from(t.quant.as_ref().unwrap().scale)
}

fn tensor_zp(t: &hematite_tflite::ParsedTensor) -> i32 {
    t.quant.as_ref().unwrap().zero_point as i32
}

fn channel_scales(quant: &Option<hematite_tflite::QuantInfo>, n: usize) -> Vec<f64> {
    match quant {
        None => vec![0.0; n],
        Some(q) => {
            if let Some(pc) = &q.per_channel {
                if pc.scales.len() == n {
                    pc.scales.iter().map(|&s| f64::from(s)).collect()
                } else if pc.scales.len() == 1 {
                    vec![f64::from(pc.scales[0]); n]
                } else {
                    vec![f64::from(q.scale); n]
                }
            } else {
                vec![f64::from(q.scale); n]
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

fn conv_quant(
    input: &hematite_tflite::ParsedTensor,
    weights: &hematite_tflite::ParsedTensor,
    output: &hematite_tflite::ParsedTensor,
    out_c: usize,
    fused: i8,
) -> (i32, i32, Vec<i32>, Vec<i32>, i32, i32) {
    let in_scale = tensor_scale(input);
    let out_scale = tensor_scale(output);
    let out_zp = tensor_zp(output);
    let w_scales = channel_scales(&weights.quant, out_c);
    let mut mults = Vec::with_capacity(out_c);
    let mut shifts = Vec::with_capacity(out_c);
    for w in &w_scales {
        let (m, s) = quantize_multiplier(in_scale * w / out_scale);
        mults.push(m);
        shifts.push(s);
    }
    let (act_min, act_max) = act_range(fused, out_scale, out_zp);
    (
        -tensor_zp(input),
        tensor_zp(weights),
        mults,
        shifts,
        act_min,
        act_max,
    )
}

// ── esp_nn requant helpers (TFLM-exact signed path) ─────────────────────────

fn sat_round_doubling_high_mul(a: i32, b: i32) -> i32 {
    // esp_nn common_functions.h L91-110 (NOT gemmlowp): sign-dependent
    // nudge + esp_nn_pick_sat_high32_of64 double-rounding for negatives.
    let overflow = a == i32::MIN && b == i32::MIN;
    let mut nudge_val: i64 = 1 << 30;
    if (a < 0) ^ (b < 0) {
        nudge_val = 1 - nudge_val;
    }
    let mult = (a as i64) * (b as i64) + nudge_val;
    // esp_nn_pick_sat_high32_of64(mult): sign = mult>>63; to_add = sign & (2^31-1)
    let sign = (mult >> 63) as i64;
    let to_add = sign & ((1i64 << 31) - 1);
    let result = ((mult + to_add) >> 31) as i32;
    if overflow {
        i32::MAX
    } else {
        result
    }
}

fn div_by_power_of_two(x: i32, exp: i32) -> i32 {
    // esp_nn common_functions.h L124-138 (NOT the fast/half-away version):
    // remainder-mask + threshold (+1 if result<0), round-up only when remainder > threshold.
    if exp <= 0 {
        return x << (-exp);
    }
    let mask = (1i64 << exp) - 1;
    let remainder = (x as i64) & mask;
    let result = (x >> exp) as i64;
    let threshold = (mask >> 1) + if result < 0 { 1 } else { 0 };
    let mut r = result;
    if remainder > threshold {
        r += 1;
    }
    r as i32
}

fn esp_nn_multiply_by_quantized_mult(x: i32, mult: i32, shift: i32) -> i32 {
    // esp_nn common_functions.h L140 exact path
    let left_shift = if shift > 0 { shift } else { 0 };
    let right_shift = if shift > 0 { 0 } else { -shift };
    let shifted = x.wrapping_shl(left_shift as u32);
    let result = sat_round_doubling_high_mul(shifted, mult);
    div_by_power_of_two(result, right_shift)
}

// ── esp_nn fc reimplementations ─────────────────────────────────────────────

fn esp_nn_fc_s8_fast(
    input: &[i8],
    weights: &[i8],
    bias: &[i32],
    in_off: i32,
    out_off: i32,
    mults: &[i32],
    shifts: &[i32],
    act_min: i32,
    act_max: i32,
    out_c: usize,
    input_dim: usize,
) -> Vec<i8> {
    let mut out = vec![0i8; out_c];
    for oc in 0..out_c {
        let row = &weights[oc * input_dim..(oc + 1) * input_dim];
        let filter_sum: i32 = row.iter().map(|&w| i32::from(w)).sum();
        let mut acc: i32 = bias[oc];
        acc += if in_off != 0 { filter_sum * in_off } else { 0 };
        for d in 0..input_dim {
            acc += i32::from(input[d]) * i32::from(row[d]);
        }
        let scaled = esp_nn_multiply_by_quantized_mult(acc, mults[oc], shifts[oc]);
        let with_offset = scaled + out_off;
        let clamped = with_offset.clamp(act_min, act_max);
        out[oc] = clamped as i8;
    }
    out
}

fn esp_nn_fc_s16(
    input: &[i8],
    weights: &[i8],
    bias: &[i32],
    in_off: i32,
    out_off: i32,
    mults: &[i32],
    shifts: &[i32],
    act_min: i32,
    act_max: i32,
    out_c: usize,
    input_dim: usize,
) -> Vec<i8> {
    // s16 asm: (input + in_off) * (filter + filter_off) lanes → int32 acc
    let mut out = vec![0i8; out_c];
    for oc in 0..out_c {
        let row = &weights[oc * input_dim..(oc + 1) * input_dim];
        let mut acc: i32 = bias[oc];
        for d in 0..input_dim {
            let i = i32::from(input[d]) + in_off;
            let w = i32::from(row[d]);
            acc += i * w;
        }
        let scaled = esp_nn_multiply_by_quantized_mult(acc, mults[oc], shifts[oc]);
        let with_offset = scaled + out_off;
        let clamped = with_offset.clamp(act_min, act_max);
        out[oc] = clamped as i8;
    }
    out
}

// ── golden input parser (from hematite-tests/goldens/models/anomaly_detect_int8.rs) ──

fn parse_i8_array(content: &str, const_name: &str) -> Vec<i8> {
    let marker = format!("pub const {const_name}: [i8;");
    let start = content.find(&marker).unwrap_or_else(|| panic!("no {const_name}"));
    let bracket = content[start..].find(']').unwrap() + start;
    // content[bracket] == ']' of "[i8; N]"
    let assign = content[bracket..].find('=').unwrap() + bracket;
    let open = content[assign..].find('[').unwrap() + assign + 1;
    let close = content[open..].find("];").unwrap() + open;
    let body = &content[open..close];
    body.split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<i8>().unwrap())
        .collect()
}

fn fnv1a(data: &[i8]) -> u32 {
    let mut h: u32 = 2166136261;
    for &b in data {
        h ^= b as u32;
        h = h.wrapping_mul(16777619);
    }
    h
}

fn as_i8(b: &[u8]) -> &[i8] {
    unsafe { std::slice::from_raw_parts(b.as_ptr() as *const i8, b.len()) }
}

fn main() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    let model_path = root.join("models/zoo/anomaly_detect/anomaly_detect_int8.tflite");
    let golden_path = root.join("hematite-tests/goldens/models/anomaly_detect_int8.rs");
    let tflite = std::fs::read(&model_path).expect("read tflite");
    let golden = std::fs::read_to_string(&golden_path).expect("read golden");
    let input = parse_i8_array(&golden, "INPUT_DATA");
    let expected = parse_i8_array(&golden, "EXPECTED_OUTPUT");
    assert_eq!(input.len(), 640);
    assert_eq!(expected.len(), 640);

    let model = parse(&tflite).expect("parse model");
    let tensors = model.tensors();
    let ops = model.ops();
    println!("ops: {}", ops.len());
    for (i, op) in ops.iter().enumerate() {
        println!("op {i}: code={} inputs={:?} outputs={:?}", op.builtin_code, op.inputs, op.outputs);
    }

    // Execute fc chain with hematite-ref (TFLM authoritative)
    let mut state: Vec<i8> = input.clone();
    let mut state_s8: Vec<i8> = input.clone();
    let mut state_s16: Vec<i8> = input.clone();
    let mut scratch = vec![0u8; 65536];
    let mut mismatch_any = false;
    let mut fc_index = 0usize;
    for (i, op) in ops.iter().enumerate() {
        let in_t = *op.inputs.first().unwrap() as usize;
        let w_t = *op.inputs.get(1).unwrap() as usize;
        let b_t = op.inputs.get(2).map(|&t| t as usize);
        let out_t = *op.outputs.first().unwrap() as usize;
        let t_in = &tensors[in_t];
        let t_w = &tensors[w_t];
        let t_out = &tensors[out_t];
        let input_dim = flat_len(&t_in.shape);
        let out_c = flat_len(&t_out.shape);
        let weights = as_i8(model.buffer_data(t_w).unwrap());
        let bias_bytes = b_t.map(|b| model.buffer_data(&tensors[b]).unwrap()).unwrap_or(&[]);
        let bias: Vec<i32> = bias_bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let fused = match &op.options {
            Some(hematite_tflite::ParsedOptions::FullyConnected { fused_activation, .. }) => *fused_activation,
            _ => 0,
        };
        let (in_off, _w_off, mults, shifts, act_min, act_max) =
            conv_quant(t_in, t_w, t_out, out_c, fused);

        // hematite-ref
        let params = FullyConnectedParams {
            input_dim: input_dim as i32,
            output_dim: out_c as i32,
            input_offset: in_off,
            weights_offset: 0,
            output_offset: tensor_zp(t_out),
            output_multiplier_per_channel: &mults,
            output_shift_per_channel: &shifts,
            quantized_activation_min: act_min,
            quantized_activation_max: act_max,
        };
        let mut ref_out = vec![0i8; out_c];
        let res: Result<(), KernelError> = hematite_ref::fully_connected::fully_connected(
            &state,
            weights,
            &bias,
            &params,
            &mut ref_out,
            &mut scratch,
        );
        res.expect("ref fc");

        // esp_nn reimplementations
        let s8 = esp_nn_fc_s8_fast(
            &state, weights, &bias, in_off, tensor_zp(t_out), &mults, &shifts,
            act_min, act_max, out_c, input_dim,
        );
        let s16 = esp_nn_fc_s16(
            &state, weights, &bias, in_off, tensor_zp(t_out), &mults, &shifts,
            act_min, act_max, out_c, input_dim,
        );

        let eq_ref = s8 == ref_out;
        let eq16 = s16 == ref_out;
        if !eq_ref || !eq16 {
            mismatch_any = true;
            // find first differing index
            let first = ref_out
                .iter()
                .zip(s8.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(usize::MAX);
            println!(
                "  fc{fc_index} (op {i}) dims {input_dim}x{out_c} in_off={in_off} out_off={} \
                 mult[0]={} shift[0]={}: s8_fast==ref:{eq_ref} s16==ref:{eq16} first_diff@{first}",
                tensor_zp(t_out),
                mults[0],
                shifts[0],
            );
            if !eq_ref && first != usize::MAX {
                println!(
                    "    ref[{}]={} s8[{}]={} s16[{}]={}",
                    first,
                    ref_out[first],
                    first,
                    s8[first],
                    first,
                    s16[first]
                );
            }
        } else {
            println!(
                "  fc{fc_index} (op {i}) dims {input_dim}x{out_c} in_off={in_off} out_off={} OK (s8_fast==ref==s16)",
                tensor_zp(t_out)
            );
        }
        state = ref_out;
        state_s8 = s8;
        state_s16 = s16;
        fc_index += 1;
    }

    let final_fnv = fnv1a(&state);
    let esp_fnv = fnv1a(&state_s8);
    let esp16_fnv = fnv1a(&state_s16);
    println!("\nfinal fnv (ref chain): 0x{final_fnv:08x}");
    println!("expected (host golden): 0x{:08x}", fnv1a(&expected));
    println!("esp_nn s8-fast chain  : 0x{esp_fnv:08x}");
    println!("esp_nn s16 chain      : 0x{esp16_fnv:08x}");
    println!("on-device ESP-NN      : 0x16213cfa");
    if !mismatch_any {
        println!("ALL FC LAYERS MATCH: esp_nn algorithm == hematite-ref on this model");
    } else {
        println!("MISMATCH FOUND: esp_nn algorithm diverges from hematite-ref on anomaly");
    }
}
