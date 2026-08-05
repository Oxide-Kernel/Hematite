// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Per-op golden test: `KernelBackend::matmul` through `RefBackend` (T5.1).

mod matmul_fixture {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/goldens/matmul.rs"));
}

use hematite_core::op_params::MatMulParams;
use hematite_core::KernelBackend;
use hematite_ref::RefBackend;

#[test]
fn matmul_golden() {
    let backend = RefBackend;
    let params = MatMulParams {
        m: matmul_fixture::M,
        n: matmul_fixture::N,
        k: matmul_fixture::K,
        adj_x: matmul_fixture::ADJ_X != 0,
        adj_y: matmul_fixture::ADJ_Y != 0,
        input_offset: matmul_fixture::INPUT_OFFSET,
        weights_offset: matmul_fixture::WEIGHTS_OFFSET,
        output_offset: matmul_fixture::OUTPUT_OFFSET,
        output_multiplier: matmul_fixture::OUTPUT_MULTIPLIER[0],
        output_shift: matmul_fixture::OUTPUT_SHIFT[0],
        quantized_activation_min: matmul_fixture::OUTPUT_ACTIVATION_MIN,
        quantized_activation_max: matmul_fixture::OUTPUT_ACTIVATION_MAX,
    };
    let mut output = [0i8; 6];
    backend
        .matmul(
            &matmul_fixture::INPUT_DATA,
            &matmul_fixture::WEIGHTS_DATA,
            &matmul_fixture::BIAS_DATA,
            &params,
            &mut output,
            &mut [],
        )
        .expect("matmul kernel returned Err");
    assert_eq!(
        &output[..],
        &matmul_fixture::EXPECTED_OUTPUT[..],
        "matmul_golden: mismatch"
    );
}
