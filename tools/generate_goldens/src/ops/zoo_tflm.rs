//! Regenerate model goldens + the hard_swish op fixture from EXECUTED
//! tflite-micro harness output (tools/tflm-goldens, pinned SHA).
//!
//! Consumes the harness stdout format (see tools/tflm-goldens/main.cc):
//!
//! ```text
//! == MODEL <name> ==
//! input_shape: 1 1960
//! output_shape: 1 4
//! output: -64,-64,-64,-64
//! fnv1a: 0x3a24ca75 (975504501)
//! ```
//!
//! Rules (todo T10):
//! * EXPECTED_OUTPUT values come from the harness output VERBATIM.
//! * Every case's printed FNV-1a is cross-checked against a recompute — a
//!   mismatch aborts (the generator never consumes corrupt harness output).
//! * Model goldens are rewritten ONLY when the executed-TFLM FNV-1a differs
//!   from the current golden's EXPECTED_OUTPUT hash. sine / hello_world /
//!   kws / person_detect match -> left untouched (no churn).
//! * The hard_swish op fixture is always re-tiered (DOWNGRADED
//!   integer-rational approximation -> executed-TFLM provenance).
//! * Inputs and shapes come from the PRE-EXISTING goldens (unchanged); only
//!   EXPECTED_OUTPUT + the provenance banner are rewritten.

use crate::fixture::FixtureWriter;
use std::path::Path;

const FNV_OFFSET_BASIS: u32 = 2166136261;
const FNV_PRIME: u32 = 16777619;

/// FNV-1a 32-bit over raw output bytes (i8 -> u8), identical to
/// hematite-benchmarks/src/model_validation.rs::fnv1a.
fn fnv1a_i8(values: &[i8]) -> u32 {
    let mut h = FNV_OFFSET_BASIS;
    for &v in values {
        h ^= v as u8 as u32;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

struct HarnessCase {
    name: String,
    input_shape: Vec<i32>,
    output_shape: Vec<i32>,
    output: Vec<i8>,
    fnv1a: u32,
}

/// Parse the harness stdout into per-case records; verifies each printed
/// FNV-1a against a recompute over the parsed output bytes.
fn parse_harness(text: &str) -> Vec<HarnessCase> {
    let mut cases = Vec::new();
    let mut cur: Option<HarnessCase> = None;

    for line in text.lines() {
        let line = line.trim();
        if let Some(name) = line.strip_prefix("== MODEL ").and_then(|s| s.strip_suffix(" ==")) {
            if let Some(c) = cur.take() {
                cases.push(c);
            }
            cur = Some(HarnessCase {
                name: name.to_string(),
                input_shape: Vec::new(),
                output_shape: Vec::new(),
                output: Vec::new(),
                fnv1a: 0,
            });
            continue;
        }
        let Some(c) = cur.as_mut() else { continue };
        if let Some(rest) = line.strip_prefix("input_shape:") {
            c.input_shape = rest.split_whitespace().filter_map(|t| t.parse().ok()).collect();
        } else if let Some(rest) = line.strip_prefix("output_shape:") {
            c.output_shape = rest.split_whitespace().filter_map(|t| t.parse().ok()).collect();
        } else if let Some(rest) = line.strip_prefix("output:") {
            c.output = rest
                .split(',')
                .filter_map(|t| t.trim().parse::<i8>().ok())
                .collect();
        } else if let Some(rest) = line.strip_prefix("fnv1a:") {
            let hex = rest.split_whitespace().next().unwrap_or("");
            c.fnv1a = u32::from_str_radix(hex.trim_start_matches("0x"), 16)
                .unwrap_or_else(|_| panic!("unparseable fnv1a line: {line:?}"));
        }
    }
    if let Some(c) = cur.take() {
        cases.push(c);
    }

    for c in &cases {
        assert!(
            c.fnv1a == fnv1a_i8(&c.output),
            "harness integrity: {} printed fnv1a 0x{:08x} != recompute 0x{:08x}",
            c.name,
            c.fnv1a,
            fnv1a_i8(&c.output)
        );
    }
    cases
}

/// Parse `pub const NAME: [i8; N] = [ ... ];` — anchors on the `= [` that
/// opens the VALUES array (the first `[` belongs to the `[i8; N]` type).
fn parse_i8_const(text: &str, name: &str) -> Vec<i8> {
    let needle = format!("pub const {name}: [i8;");
    let start = text
        .find(&needle)
        .unwrap_or_else(|| panic!("const {name} not found"));
    let after = &text[start + needle.len()..];
    let eq = after
        .find("= [")
        .unwrap_or_else(|| panic!("const {name}: no = [ array"));
    let block = &after[eq + 3..];
    let close = block
        .find("];")
        .unwrap_or_else(|| panic!("const {name}: no array close"));
    block[..close]
        .split([',', '\n'])
        .filter_map(|t| t.trim().parse::<i8>().ok())
        .collect()
}

/// Parse `pub const NAME: [i32; N] = [ ... ];` (same anchoring as above).
fn parse_i32_shape(text: &str, name: &str) -> Vec<i32> {
    let needle = format!("pub const {name}: [i32;");
    let start = text
        .find(&needle)
        .unwrap_or_else(|| panic!("const {name} not found"));
    let after = &text[start + needle.len()..];
    let eq = after
        .find("= [")
        .unwrap_or_else(|| panic!("const {name}: no = [ array"));
    let block = &after[eq + 3..];
    let close = block
        .find("];")
        .unwrap_or_else(|| panic!("const {name}: no array close"));
    block[..close]
        .split([',', '\n'])
        .filter_map(|t| t.trim().parse::<i32>().ok())
        .collect()
}

fn parse_str_const(text: &str, name: &str) -> String {
    let needle = format!(r#"pub const {name}: &str = ""#);
    let start = text
        .find(&needle)
        .unwrap_or_else(|| panic!("const {name} not found"));
    let rest = &text[start + needle.len()..];
    let end = rest
        .find('"')
        .unwrap_or_else(|| panic!("const {name}: unterminated string"));
    rest[..end].to_string()
}

/// Model stems in harness order; each must exist as goldens/models/<stem>.rs.
const MODEL_STEMS: [&str; 6] = [
    "sine",
    "hello_world_int8",
    "kws_micro_speech_int8",
    "anomaly_detect_int8",
    "person_detect_int8",
    "mobilenet_v2_1.0_224_int8",
];

/// Entry point for `cargo run -p generate-goldens -- tflm-regen <file>`.
pub fn regen_from_tflm_harness(w: &mut FixtureWriter, workspace_root: &Path, harness_file: &Path) {
    let text = std::fs::read_to_string(harness_file)
        .unwrap_or_else(|e| panic!("cannot read harness output {}: {e}", harness_file.display()));
    let cases = parse_harness(&text);
    let by_name: std::collections::HashMap<&str, &HarnessCase> =
        cases.iter().map(|c| (c.name.as_str(), c)).collect();

    println!("\n── Model goldens: executed-TFLM regeneration (todo T10) ──");

    for stem in MODEL_STEMS {
        let case = by_name
            .get(stem)
            .unwrap_or_else(|| panic!("harness output has no case for {stem}"));
        let path = workspace_root
            .join("hematite-tests")
            .join("goldens")
            .join("models")
            .join(format!("{stem}.rs"));
        let golden = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));

        let input_data = parse_i8_const(&golden, "INPUT_DATA");
        let expected = parse_i8_const(&golden, "EXPECTED_OUTPUT");
        let input_shape = parse_i32_shape(&golden, "INPUT_SHAPE");
        let output_shape = parse_i32_shape(&golden, "OUTPUT_SHAPE");
        let model_path = parse_str_const(&golden, "MODEL_PATH");

        // Shape consts are metadata captured from the original LiteRT run and
        // may differ in rank from the model tensors (e.g. sine's golden
        // OUTPUT_SHAPE is [1,1] while the model output tensor is rank-1 [1]);
        // the invariant that matters is the element count (byte count of the
        // .bin the harness consumed).
        assert_eq!(
            input_shape.iter().product::<i32>(),
            case.input_shape.iter().product::<i32>(),
            "{stem}: harness input element count != golden INPUT_SHAPE product"
        );
        assert_eq!(
            output_shape.iter().product::<i32>(),
            case.output_shape.iter().product::<i32>(),
            "{stem}: harness output element count != golden OUTPUT_SHAPE product"
        );
        assert_eq!(
            input_data.len(),
            case.input_shape.iter().product::<i32>() as usize,
            "{stem}: harness input shape does not match INPUT_DATA len"
        );

        let golden_hash = fnv1a_i8(&expected);
        if golden_hash == case.fnv1a {
            println!("  {stem}: MATCH (golden 0x{golden_hash:08x} == executed TFLM) — untouched");
            continue;
        }
        println!(
            "  {stem}: DIFF golden 0x{golden_hash:08x} vs executed TFLM 0x{:08x} — regenerating",
            case.fnv1a
        );
        w.write_model_tflm(
            stem,
            &input_shape,
            &output_shape,
            &input_data,
            &case.output,
            &model_path,
        );
    }

    println!("\n── hard_swish op fixture: re-tier (DOWNGRADED -> executed TFLM) ──");
    let case = by_name
        .get("hard_swish")
        .unwrap_or_else(|| panic!("harness output has no hard_swish case"));
    let path = workspace_root
        .join("hematite-tests")
        .join("goldens")
        .join("hard_swish.rs");
    let golden = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let input_data = parse_i8_const(&golden, "INPUT_DATA");
    assert_eq!(case.input_shape, [1, 1, 1, 8], "hard_swish: harness shape mismatch");
    w.write_hard_swish_tflm(&input_data, &case.output);
    println!(
        "  hard_swish: executed TFLM fnv1a 0x{:08x}, {} values",
        case.fnv1a,
        case.output.len()
    );
}
