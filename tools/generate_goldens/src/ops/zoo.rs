//! Per-model golden generation — extends the generator to emit model-level
//! fixtures captured from an executed TFLite interpreter.
//!
//! For each runnable `.tflite` model discovered under the workspace, this
//! module invokes `zoo/run_model.py` (ai-edge-litert interpreter, project
//! venv at `zoo/.venv`) as a subprocess, captures the exact int8 output, and
//! writes it via `FixtureWriter::write_model`.
//!
//! The 18-model zoo (plan T5.2) is NOT runnable through this path as of T5.0:
//! all 15 esp-dl artifacts are the proprietary `.espdl` format (no `.tflite`
//! exists anywhere in esp-dl), and the 6 edge-ml models are not publicly
//! available. Those barriers are documented in
//! `models/zoo/DEFERRED_MODELS.md`. The mechanism is proven end-to-end with
//! `models/sine.tflite` (the workspace's only runnable TFLite model).

use crate::fixture::FixtureWriter;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Scan `models/` (workspace root) for runnable `.tflite` models and emit a
/// golden fixture for each. Skips `.espdl` artifacts (format barrier, logged).
pub fn generate_model_goldens(w: &mut FixtureWriter, workspace_root: &Path) {
    let models_dir = workspace_root.join("models");

    let mut tflite_models: Vec<PathBuf> = Vec::new();
    collect_tflite(&models_dir, &mut tflite_models);

    println!("\n── Model goldens (executed TFLite) ──");
    if tflite_models.is_empty() {
        println!("  ⚠️  No runnable .tflite models found under models/. The zoo is");
        println!("     .espdl-only (see models/zoo/DEFERRED_MODELS.md).");
    }

    for model in &tflite_models {
        let rel_path = model.strip_prefix(workspace_root).unwrap_or(model);
        let rel = rel_path.to_string_lossy();
        match capture_model(w, workspace_root, model, &rel) {
            Ok(()) => {}
            Err(e) => {
                println!("  ⚠️  Skipped {rel}: {e}");
            }
        }
    }

    let zoo = workspace_root.join("models").join("zoo");
    if zoo.exists() {
        let count = count_espdl(&zoo);
        println!("  ({count} .espdl zoo artifacts present — not runnable via TFLite)");
    }
}

fn collect_tflite(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.ends_with("zoo") {
                // zoo contains .espdl only; skip recursion (no .tflite exists)
                continue;
            }
            collect_tflite(&path, out);
        } else if path.extension().is_some_and(|e| e == "tflite") {
            out.push(path);
        }
    }
}

fn count_espdl(dir: &Path) -> usize {
    let mut total = 0usize;
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            total += count_espdl(&path);
        } else if path.extension().is_some_and(|x| x == "espdl") {
            total += 1;
        }
    }
    total
}

fn capture_model(
    w: &mut FixtureWriter,
    workspace_root: &Path,
    model: &Path,
    rel: &str,
) -> Result<(), String> {
    let harness = workspace_root
        .join("tools")
        .join("generate_goldens")
        .join("zoo")
        .join("run_model.py");
    let venv_py = workspace_root
        .join("tools")
        .join("generate_goldens")
        .join("zoo")
        .join(".venv")
        .join("bin")
        .join("python3");
    let python = if venv_py.exists() {
        venv_py
    } else {
        PathBuf::from("python3")
    };

    let tmp_out = std::env::temp_dir().join(format!(
        "hematite_model_{}.txt",
        model.file_stem().and_then(|s| s.to_str()).unwrap_or("model")
    ));

    let status = Command::new(&python)
        .arg(&harness)
        .arg(model)
        .arg(&tmp_out)
        .status()
        .map_err(|e| format!("spawn interpreter: {e}"))?;

    if !status.success() {
        return Err(format!(
            "interpreter exited {status:?} (ai-edge-litert missing? install zoo/.venv via pip)"
        ));
    }

    let text = std::fs::read_to_string(&tmp_out).map_err(|e| format!("read output: {e}"))?;
    let mut lines = text.lines();
    let runtime = lines.next().unwrap_or("unknown").to_string();
    let in_shape: Vec<i32> = lines
        .next()
        .unwrap_or("")
        .split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect();
    let out_shape: Vec<i32> = lines
        .next()
        .unwrap_or("")
        .split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect();
    let zp: i32 = lines.next().unwrap_or("0").parse().unwrap_or(0);
    let _dtype = lines.next().unwrap_or("int8").to_string();
    let out_data: Vec<i8> = lines
        .next()
        .unwrap_or("")
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();

    if out_data.is_empty() {
        return Err("output data empty".into());
    }

    let n_in: usize = in_shape.iter().product::<i32>() as usize;
    let input_data = vec![zp as i8; n_in];

    let name = model.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    w.write_model(
        name,
        &in_shape,
        &out_shape,
        &input_data,
        &out_data,
        &runtime,
        rel,
    );
    Ok(())
}
