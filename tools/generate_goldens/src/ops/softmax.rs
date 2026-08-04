//! Softmax golden fixture — int8 softmax with gemmlowp fixed-point exponential.
//!
//! Uses `exp_on_negative_values` from gemmlowp's fixedpoint library (4th-order Taylor
//! expansion around -1/8 + barrel shifter with 7 exponential multipliers), plus TFLM's
//! kAccumulationIntegerBits=12 accumulation and GetReciprocal-based normalization.
//!
//! Mirroring of `tflite::reference_ops::Softmax<int8_t, int8_t>` at the pinned SHA.
//! Not bit-exact against executed TFLM, but uses the genuine gemmlowp exponential
//! and reciprocal algorithm that TFLM's integer softmax relies on.

use crate::fixture::FixtureWriter;
use crate::tflm_math;

pub fn generate(w: &mut FixtureWriter) {
    let input_shape = [1i32, 1, 1, 5];
    let output_shape = [1i32, 1, 1, 5];

    let input: Vec<i8> = vec![-20, -5, 0, 5, 20];

    let input_scale: f64 = 0.1;
    let (input_multiplier, input_shift) = tflm_math::quantize_multiplier(input_scale);
    let left_shift: i32 = 22;
    let diff_min: i32 = -(1i32 << 7);

    let depth = 5usize;

    // TFLM constants: kScaledDiffIntegerBits = 5, kAccumulationIntegerBits = 12.
    // Exponential is Q0.31; accumulation downshifts each exp by 12 → Q12.19.
    const K_ACCUM_INT_BITS: i32 = 12;

    // Step 1: Find max.
    let mut max_val: i32 = i32::MIN;
    for &v in &input {
        max_val = max_val.max(i32::from(v));
    }

    // Step 2: Compute Q5.26 diffs, gemmlowp exp in Q0.31, accumulate in Q12.19.
    let q526_factor = input_scale * (1u64 << 26) as f64;
    let mut exps_q031: [i32; 5] = [0i32; 5];
    let mut sum_q1219: i32 = 0i32;

    for i in 0..depth {
        let diff = i32::from(input[i]) - max_val;
        if diff < diff_min {
            continue;
        }
        let diff_q526 = (diff as f64 * q526_factor).round() as i32;
        let exp_q031 = tflm_math::exp_on_negative_values(diff_q526, 5);
        exps_q031[i] = exp_q031;
        // Rescale<kAccumulationIntegerBits>: Q0.31 → Q12.19 (right-shift by 12, rounding)
        let exp_q1219 = tflm_math::rounding_divide_by_pot(exp_q031, K_ACCUM_INT_BITS);
        sum_q1219 = sum_q1219.wrapping_add(exp_q1219);
    }

    // Step 3: Reciprocal via TFLM's GetReciprocal + one_over_one_plus_x.
    let output_zero_point: i32 = -128;
    let mut output: Vec<i8> = vec![0i8; depth];

    if sum_q1219 > 0 {
        let mut num_bits_over_unit: i32 = 0;
        let shifted_scale =
            tflm_math::get_reciprocal(sum_q1219, K_ACCUM_INT_BITS, &mut num_bits_over_unit);
        let exponent = num_bits_over_unit + 23;

        for i in 0..depth {
            let diff = i32::from(input[i]) - max_val;
            if diff < diff_min {
                output[i] = i8::MIN;
                continue;
            }
            // exp · 1/sum in Q0.31, then right-shift by exponent to compress into int8 range.
            let scaled_raw =
                tflm_math::saturating_rounding_doubling_high_mul(shifted_scale, exps_q031[i]);
            let unsat_out = tflm_math::rounding_divide_by_pot(scaled_raw, exponent);
            let signed_out = unsat_out.wrapping_add(output_zero_point);
            output[i] = signed_out.clamp(-128, 127) as i8;
        }
    }

    // ── Emit fixture ──
    let mut buf = String::with_capacity(16384);
    buf.push_str(crate::fixture::SPDX_HEADER);
    buf.push('\n');
    buf.push_str(crate::fixture::PROVENANCE_NOTE);
    buf.push('\n');

    use std::fmt::Write as FmtWrite;
    let _ = writeln!(buf, "/// TFLite Micro pin that defines this golden corpus.");
    let _ = writeln!(
        buf,
        "pub const GOLDEN_TFLM_VERSION: &str = \"{}\";\n",
        crate::fixture::TFLM_VERSION
    );

    // Standard provenance (upgraded from DOWNGRADED)
    let _ = writeln!(
        buf,
        "/// Provenance of these golden values."
    );
    let _ = writeln!(
        buf,
        "pub const GOLDEN_PROVENANCE: &str = \"tool-internal-reference-reimplementation; NOT captured from executed TFLM; see tools/generate_goldens/README.md\";\n"
    );

    let _ = writeln!(buf, "pub const INPUT_SHAPE: [i32; 4] = {:?};", input_shape);
    let _ = writeln!(buf, "pub const FILTER_SHAPE: [i32; 4] = {:?};", [0i32; 4]);
    let _ = writeln!(buf, "pub const OUTPUT_SHAPE: [i32; 4] = {:?};\n", output_shape);

    // Softmax-specific quantization consts (for T2.2c helper)
    let _ = writeln!(buf, "/// De-quantization scale for softmax logits.");
    let _ = writeln!(
        buf,
        "pub const INPUT_SCALE: f64 = {:?};\n",
        input_scale
    );
    let _ = writeln!(buf, "pub const INPUT_MULTIPLIER: i32 = {};", input_multiplier);
    let _ = writeln!(buf, "pub const INPUT_SHIFT: i32 = {};", input_shift);
    let _ = writeln!(buf, "pub const LEFT_SHIFT: i32 = {};", left_shift);
    let _ = writeln!(buf, "pub const DIFF_MIN: i32 = {};\n", diff_min);

    let _ = writeln!(buf, "pub const INPUT_OFFSET: i32 = 0;");
    let _ = writeln!(buf, "pub const OUTPUT_OFFSET: i32 = {};", output_zero_point);
    let _ = writeln!(buf, "pub const OUTPUT_ACTIVATION_MIN: i32 = -128;");
    let _ = writeln!(buf, "pub const OUTPUT_ACTIVATION_MAX: i32 = 127;\n");

    crate::fixture::emit_const_array(&mut buf, "INPUT_DATA", "i8", &input);
    crate::fixture::emit_const_array(&mut buf, "WEIGHTS_DATA", "i8", &[] as &[i8]);
    crate::fixture::emit_const_array(&mut buf, "BIAS_DATA", "i32", &[] as &[i32]);
    crate::fixture::emit_const_array(&mut buf, "OUTPUT_MULTIPLIER", "i32", &[input_multiplier]);
    crate::fixture::emit_const_array(&mut buf, "OUTPUT_SHIFT", "i32", &[input_shift]);
    crate::fixture::emit_const_array(&mut buf, "EXPECTED_OUTPUT", "i8", &output);

    let path = w.output_dir().join("softmax.rs");
    std::fs::write(&path, buf).expect("write fixture");
    println!("  Wrote {}", path.display());
}
