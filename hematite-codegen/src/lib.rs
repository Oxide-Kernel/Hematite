//! hematite-codegen — proc-macro: TFLite → straight-line backend-dispatched code.
//!
//! Applies `#[model("path.tflite")]` to an empty struct.  At compile time the
//! macro reads the flatbuffer, walks it with a hand-rolled byte-offset walker,
//! applies a JAX-style graph optimization pass, and emits inference code
//! dispatched through the `KernelBackend` trait.

use proc_macro::TokenStream;

/// Attribute macro entry point — `#[model("path.tflite")]` on a struct.
///
/// Reads the `.tflite` model at compile time and generates straight-line,
/// backend-dispatched inference code.  Returns the input item unchanged for
/// now — the real flatbuffer byte-offset walker + JAX optimisation pass
/// lands in T4.0.
///
/// TODO(T4.0): Hand-rolled ~300-LOC byte-offset walker with NO `flatbuffers`
/// runtime dependency.  Parse the model, apply graph optimisations, emit
/// backed-dispatched inference code.
#[proc_macro_attribute]
pub fn model(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Stub: return the input item unchanged.
    item
}
