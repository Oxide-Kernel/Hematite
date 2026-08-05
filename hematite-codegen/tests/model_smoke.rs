// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Macro smoke test — applies `#[model]` to a struct referencing sine.tflite.
//!
//! The macro expansion verifies the model parses without error.  T4.1 will
//! replace the minimal expansion with real code emission.

use hematite_codegen::model;

/// Smoke test struct annotated with the `#[model]` proc-macro.
///
/// At compile time this reads `models/sine.tflite`, parses the flatbuffer,
/// and emits a const assertion proving the parse succeeded.
///
/// Path is relative to the **tests/** crate's `CARGO_MANIFEST_DIR`
/// (which is the `hematite-codegen/` directory), so `../models/sine.tflite`
/// resolves to the workspace `models/` directory.
#[model("../models/sine.tflite")]
pub struct SineModel;

#[test]
fn sine_model_compiles() {
    // If the macro expanded without compile_error, compilation itself
    // is the test.  This function ensures the test runner has something
    // to execute.
    let _ = SineModel;
}
