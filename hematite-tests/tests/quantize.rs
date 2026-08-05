// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Per-op golden test: `KernelBackend::quantize` through `RefBackend` (T5.1).
//!
//! The quantize fixture is degenerate (input = output): the fixture's input
//! values are already in the target quantized domain, so the quantize
//! direction multiplier is identity (scale 1.0).

mod quantize_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/quantize.rs"));
}

use hematite_core::op_params::QuantParam;
use hematite_core::KernelBackend;
use hematite_ref::RefBackend;

#[test]
fn quantize_golden() {
    let backend = RefBackend;
    let params = QuantParam {
        quantize_multiplier: 1i32 << 30, // identity scale 1.0
        quantize_shift: 1,
        dequantize_multiplier: quantize_fixture::SCALE_Q31,
        dequantize_shift: 0,
        zero_point: quantize_fixture::ZERO_POINT,
    };
    let mut output = [0i8; 6];
    backend
        .quantize(&quantize_fixture::INPUT_DATA, &params, &mut output)
        .expect("quantize kernel returned Err");
    assert_eq!(
        &output[..],
        &quantize_fixture::EXPECTED_OUTPUT[..],
        "quantize_golden: mismatch"
    );
}
