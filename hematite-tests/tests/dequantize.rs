// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Per-op golden test: `KernelBackend::dequantize` through `RefBackend`
//! (T5.1).
//!
//! The dequantize golden was generated with f64 arithmetic; the stored
//! SCALE_Q31 truncates the exact scale by ~0.5 LSB and the kernel adds a +1
//! compensation to `dequantize_multiplier` (see the kernel rustdoc). T5.0
//! must regenerate this fixture against a real TFLM binary.

mod dequantize_fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/goldens/dequantize.rs"
    ));
}

use hematite_core::op_params::QuantParam;
use hematite_core::KernelBackend;
use hematite_ref::RefBackend;

#[test]
fn dequantize_golden() {
    let backend = RefBackend;
    let params = QuantParam {
        quantize_multiplier: 0,
        quantize_shift: 0,
        dequantize_multiplier: dequantize_fixture::SCALE_Q31,
        dequantize_shift: 0,
        zero_point: dequantize_fixture::ZERO_POINT,
    };
    let mut output = [0i8; 6];
    backend
        .dequantize(&dequantize_fixture::INPUT_DATA, &params, &mut output)
        .expect("dequantize kernel returned Err");
    assert_eq!(
        &output[..],
        &dequantize_fixture::EXPECTED_OUTPUT[..],
        "dequantize_golden: mismatch"
    );
}
