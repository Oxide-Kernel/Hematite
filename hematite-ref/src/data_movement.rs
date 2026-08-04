// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Data-movement scalar reference kernels — concat, split, pad, slice.
//!
//! Implements TFLM int8 reference-layout copy kernels for the four
//! runtime data-movement ops that have existing golden fixtures.
//!
//! # Batch constraint
//!
//! Only batch = 1 is supported (per the static-shape constraint).
//! `batch > 1` returns [`KernelError::Unsupported`].

use hematite_core::op_params::{ConcatParams, PadParams, SliceParams, SplitParams};
use hematite_core::KernelError;

/// Compute the product of a `[i32; 4]` shape array.
#[inline(always)]
fn shape_product(shape: &[i32; 4]) -> usize {
    shape[0] as usize * shape[1] as usize * shape[2] as usize * shape[3] as usize
}

/// Compute the product of dims `axis+1 .. 4` (elements per step at `axis`).
#[inline(always)]
fn inner_stride(shape: &[i32; 4], axis: i32) -> usize {
    let mut s: usize = 1;
    for d in (axis + 1)..4 {
        s *= shape[d as usize] as usize;
    }
    s
}

/// Compute the product of dims `0 .. axis` (count of outer iteration groups).
#[inline(always)]
fn outer_count(shape: &[i32; 4], axis: i32) -> usize {
    let mut c: usize = 1;
    for d in 0..axis {
        c *= shape[d as usize] as usize;
    }
    c
}

// ── concat ──────────────────────────────────────────────────────────────────

/// Concatenate two NHWC int8 tensors along an axis.
///
/// # Layouts
///
/// * `input_a` — NHWC `[batch=1, Ha, W, C]`
/// * `input_b` — NHWC `[batch=1, Hb, W, C]` (same shape except on `axis`)
/// * `output` — NHWC `[batch=1, Ha+Hb, W, C]` (or adjusted per `axis`)
///
/// # Errors
///
/// * [`KernelError::Unsupported`] if `input_shape_a[0] > 1`.
/// * [`KernelError::ShapeMismatch`] if any slice length does not match
///   the declared shapes in `params`.
pub fn concat_op(
    input_a: &[i8],
    input_b: &[i8],
    params: &ConcatParams,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    let _ = scratch;
    let axis = params.axis;

    // ── Batch constraint ────────────────────────────────────────────────
    if params.input_shape_a[0] != 1 {
        return Err(KernelError::Unsupported);
    }

    // ── Validate slice lengths ──────────────────────────────────────────
    let expected_len_a = shape_product(&params.input_shape_a);
    let expected_len_b = shape_product(&params.input_shape_b);
    let expected_out = shape_product(&params.output_shape);

    if input_a.len() != expected_len_a
        || input_b.len() != expected_len_b
        || output.len() != expected_out
    {
        return Err(KernelError::ShapeMismatch);
    }

    // ── Concatenation: copy outer-slice groups ──────────────────────────
    let oc = outer_count(&params.output_shape, axis);
    let is = inner_stride(&params.output_shape, axis);
    let size_a = params.input_shape_a[axis as usize] as usize * is;
    let size_b = params.input_shape_b[axis as usize] as usize * is;

    for g in 0..oc {
        let out_base = g * (size_a + size_b);
        let src_a_base = g * size_a;
        let src_b_base = g * size_b;
        output[out_base..out_base + size_a]
            .copy_from_slice(&input_a[src_a_base..src_a_base + size_a]);
        output[out_base + size_a..out_base + size_a + size_b]
            .copy_from_slice(&input_b[src_b_base..src_b_base + size_b]);
    }

    Ok(())
}

// ── split ───────────────────────────────────────────────────────────────────

/// Split an NHWC int8 tensor along an axis into `num_splits` equal parts and
/// return the `split_index`-th output slice.
///
/// # Layouts
///
/// * `input` — NHWC `[batch=1, H, W, C]`
/// * `output` — NHWC `[batch=1, H/num_splits, W, C]` (adjusted per `axis`)
///
/// # Errors
///
/// * [`KernelError::Unsupported`] if `input_shape[0] > 1`.
/// * [`KernelError::ShapeMismatch`] if any slice length does not match
///   the declared shapes in `params`.
pub fn split_op(
    input: &[i8],
    split_index: i32,
    params: &SplitParams,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    let _ = scratch;
    let axis = params.axis;

    // ── Batch constraint ────────────────────────────────────────────────
    if params.input_shape[0] != 1 {
        return Err(KernelError::Unsupported);
    }

    // ── Validate slice lengths ──────────────────────────────────────────
    let expected_in = shape_product(&params.input_shape);
    // Pick the output shape for the requested split index
    let output_shape = if split_index == 0 {
        params.output_shape_a
    } else {
        params.output_shape_b
    };
    let expected_out = shape_product(&output_shape);

    if input.len() != expected_in || output.len() != expected_out {
        return Err(KernelError::ShapeMismatch);
    }

    // ── Split: copy the requested slice ─────────────────────────────────
    let oc = outer_count(&params.input_shape, axis);
    let is = inner_stride(&params.input_shape, axis);
    let split_elems = (params.input_shape[axis as usize] / params.num_splits) as usize;
    let chunk_size = split_elems * is;
    let full_slice = params.input_shape[axis as usize] as usize * is;

    for g in 0..oc {
        let src_base = g * full_slice + split_index as usize * chunk_size;
        let out_base = g * chunk_size;
        output[out_base..out_base + chunk_size]
            .copy_from_slice(&input[src_base..src_base + chunk_size]);
    }

    Ok(())
}

// ── pad ─────────────────────────────────────────────────────────────────────

/// Zero-pad an NHWC int8 tensor.
///
/// Mirrors TFLM `Pad` with constant value 0.
///
/// # Layouts
///
/// * `input` — NHWC `[batch=1, H_in, W_in, C]`
/// * `output` — NHWC `[batch=1, H_out, W_out, C]`
///
/// `left_padding` / `right_padding` specify the number of elements to
/// prepend / append per dimension (batch, height, width, channels).
///
/// # Errors
///
/// * [`KernelError::Unsupported`] if `input_shape[0] > 1`.
/// * [`KernelError::ShapeMismatch`] if any slice length does not match
///   the declared shapes in `params`.
pub fn pad_op(
    input: &[i8],
    params: &PadParams,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    let _ = scratch;

    // ── Batch constraint ────────────────────────────────────────────────
    if params.input_shape[0] != 1 {
        return Err(KernelError::Unsupported);
    }

    // ── Validate slice lengths ──────────────────────────────────────────
    let expected_in = shape_product(&params.input_shape);
    let expected_out = shape_product(&params.output_shape);

    if input.len() != expected_in || output.len() != expected_out {
        return Err(KernelError::ShapeMismatch);
    }

    // ── Zero-fill output, then copy input at the padded offsets ─────────
    output.fill(0);

    let h_in = params.input_shape[1] as usize;
    let w_in = params.input_shape[2] as usize;
    let w_out = params.output_shape[2] as usize;
    let c = params.input_shape[3] as usize;

    let pad_t = params.left_padding[1] as usize;
    let pad_l = params.left_padding[2] as usize;

    for ih in 0..h_in {
        let src_row = ih * w_in * c;
        let dst_row = (ih + pad_t) * w_out * c + pad_l * c;
        output[dst_row..dst_row + w_in * c]
            .copy_from_slice(&input[src_row..src_row + w_in * c]);
    }

    Ok(())
}

// ── slice ───────────────────────────────────────────────────────────────────

/// Slice (crop) an NHWC int8 tensor given `begin` and `size`.
///
/// # Layouts
///
/// * `input` — NHWC `[batch=1, H_in, W_in, C]`
/// * `output` — NHWC `[batch=1, size[1], size[2], size[3]]`
///
/// # Errors
///
/// * [`KernelError::Unsupported`] if `input_shape[0] > 1`.
/// * [`KernelError::ShapeMismatch`] if any slice length does not match
///   the declared shapes in `params`.
pub fn slice_op(
    input: &[i8],
    params: &SliceParams,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    let _ = scratch;

    // ── Batch constraint ────────────────────────────────────────────────
    if params.input_shape[0] != 1 {
        return Err(KernelError::Unsupported);
    }

    // ── Validate slice lengths ──────────────────────────────────────────
    let expected_in = shape_product(&params.input_shape);
    let expected_out =
        params.size[0] as usize * params.size[1] as usize * params.size[2] as usize * params.size[3] as usize;

    if input.len() != expected_in || output.len() != expected_out {
        return Err(KernelError::ShapeMismatch);
    }

    // ── Slice: copy each row of the crop ────────────────────────────────
    let w_in = params.input_shape[2] as usize;
    let w_out = params.size[2] as usize;
    let h_out = params.size[1] as usize;
    let c = params.input_shape[3] as usize;

    let begin_h = params.begin[1] as usize;
    let begin_w = params.begin[2] as usize;

    for oh in 0..h_out {
        let src_row = (begin_h + oh) * w_in * c + begin_w * c;
        let dst_row = oh * w_out * c;
        output[dst_row..dst_row + w_out * c]
            .copy_from_slice(&input[src_row..src_row + w_out * c]);
    }

    Ok(())
}
