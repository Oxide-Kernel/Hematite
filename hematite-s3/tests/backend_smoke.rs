// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Host compile-only smoke tests proving `S3Backend` implements
//! [`KernelBackend`]: the generic bound (the exact one codegen's
//! `Model<B>` uses) and the `Unsupported` contract for ops without an s3
//! kernel.
//!
//! Note: `&dyn KernelBackend` trait objects are NOT possible — the trait's
//! six no-`self` `*_scratch_size` associated functions make it dyn-
//! incompatible (a trait property that applies to every backend, RefBackend
//! included). The generic-bound form is the compatible proof.

use hematite_core::op_params::ActivationParams;
use hematite_core::{KernelBackend, KernelError};
use hematite_s3::backend::S3Backend;

/// Proves `S3Backend` satisfies the `B: KernelBackend` generic bound used by
/// `Model::<S3Backend>` (codegen emits `Model<B>` with that exact bound).
fn assert_backend<B: KernelBackend>(_: &B) {}

#[test]
fn s3_backend_satisfies_kernel_backend_bound() {
    let backend = S3Backend;
    assert_backend(&backend);
}

#[test]
fn unsupported_ops_return_kernel_error_unsupported() {
    let backend = S3Backend;
    let params = ActivationParams {
        input_offset: 0,
        output_offset: -128,
        output_multiplier: 0,
        output_shift: 0,
        quantized_activation_min: -128,
        quantized_activation_max: 127,
        input_multiplier: 0,
        input_left_shift: 0,
        input_range_radius: 0,
        output_multiplier_alpha: 0,
        output_shift_alpha: 0,
        output_multiplier_identity: 0,
        output_shift_identity: 0,
        alpha_offset: 0,
        alpha_data: &[],
        output_multiplier_1: 0,
        output_shift_1: 0,
        output_multiplier_2: 0,
        output_shift_2: 0,
        reluish_multiplier_fixedpoint_int16: 0,
        reluish_multiplier_exponent: 0,
        output_multiplier_fixedpoint_int16: 0,
        output_multiplier_exponent: 0,
    };
    let mut out = [0i8; 4];
    assert_eq!(
        backend.sigmoid(&[0i8; 4], &params, &mut out),
        Err(KernelError::Unsupported)
    );
    assert_eq!(
        backend.tanh(&[0i8; 4], &params, &mut out),
        Err(KernelError::Unsupported)
    );
}
