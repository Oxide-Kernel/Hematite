// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! ResizeNearestNeighbor — scalar reference kernel.
//!
//! Implements the TFLM int8 `ResizeNearestNeighbor` algorithm: per-output-pixel
//! coordinate mapping via integer arithmetic (`src = floor(dst * in_size / out_size)`),
//! followed by an i8 copy — no value interpolation, no floating-point.
//!
//! # Coordinate mode
//!
//! Only the asymmetric/floor mode (`align_corners = false`,
//! `half_pixel_centers = false`) is supported. All 13 Resize nodes in the
//! Hematite model zoo use this mode.
//!
//! # Batch constraint
//!
//! Only batch = 1 is supported (per the static-shape constraint).
//! `batch > 1` returns [`KernelError::Unsupported`].

use hematite_core::op_params::ResizeNearestParams;
use hematite_core::KernelError;

/// Compute the product of a `[i32; 4]` shape array.
#[inline(always)]
fn shape_product(shape: &[i32; 4]) -> usize {
    shape[0] as usize * shape[1] as usize * shape[2] as usize * shape[3] as usize
}

/// ResizeNearestNeighbor int8 — asymmetric/floor coordinate mapping.
///
/// Each output pixel at `(oh, ow, c)` is a direct i8 copy from the source
/// pixel at `(floor(oh * in_h / out_h), floor(ow * in_w / out_w), c)`.
/// No value arithmetic, no interpolation — this is an index-copy kernel.
///
/// # Coordinate mapping
///
/// Uses the TFLM `ResizeNearestNeighbor` integer formulation:
///
/// ```text
/// src_h = floor(out_h * input_height / output_height)
/// src_w = floor(out_w * input_width / output_width)
/// ```
///
/// Integer division truncates toward zero, which is equivalent to `floor`
/// for the non-negative coordinates this kernel produces. The mapping is
/// exact for both upscale and downscale cases.
///
/// # Layouts
///
/// * `input` — NHWC `[batch=1, H, W, C]`
/// * `output` — NHWC `[batch=1, OH, OW, C]`
///
/// # Errors
///
/// * [`KernelError::Unsupported`] if `input_shape[0] > 1` (multi-batch not
///   supported), or if `align_corners != 0` or `half_pixel_centers != 0`
///   (only asymmetric/floor mode is implemented).
/// * [`KernelError::ShapeMismatch`] if any slice length does not match the
///   declared shapes in `params`, or if input/output channel dimensions differ.
pub fn resize_nearest_neighbor(
    input: &[i8],
    params: &ResizeNearestParams,
    output: &mut [i8],
    scratch: &mut [u8],
) -> Result<(), KernelError> {
    // ── Batch constraint ────────────────────────────────────────────────
    if params.input_shape[0] != 1 {
        return Err(KernelError::Unsupported);
    }

    // ── Semantic mode constraint — only asymmetric/floor ─────────────────
    if params.align_corners != 0 || params.half_pixel_centers != 0 {
        return Err(KernelError::Unsupported);
    }

    // ── Extract dimensions (all i32 for index arithmetic) ───────────────
    let input_h = params.input_shape[1];
    let input_w = params.input_shape[2];
    let channels = params.input_shape[3];

    let output_h = params.output_shape[1];
    let output_w = params.output_shape[2];
    let output_c = params.output_shape[3];

    // ── Slice-length validation ─────────────────────────────────────────
    if input.len() != shape_product(&params.input_shape) {
        return Err(KernelError::ShapeMismatch);
    }
    if output.len() != shape_product(&params.output_shape) {
        return Err(KernelError::ShapeMismatch);
    }

    // Channel-dimension cross-check (no broadcast — C must match)
    if channels != output_c {
        return Err(KernelError::ShapeMismatch);
    }

    // ── Stride precomputation ───────────────────────────────────────────
    let input_row_stride = input_w * channels;
    let output_row_stride = output_w * channels;

    // ── Nearest-neighbor copy loop ──────────────────────────────────────
    // TFLM coordinate mapping:
    //   src_h = floor(oh * input_h / output_h)
    //   src_w = floor(ow * input_w / output_w)
    //
    // Integer division truncates toward zero, which equals floor for all
    // non-negative indices produced by this loop.
    for oh in 0..output_h {
        let src_h = (oh * input_h / output_h) as usize;
        let input_row_base = src_h * input_row_stride as usize;

        for ow in 0..output_w {
            let src_w = (ow * input_w / output_w) as usize;
            let input_pixel_base = input_row_base + src_w * channels as usize;

            let output_idx = (oh * output_row_stride + ow * channels) as usize;

            output[output_idx..output_idx + channels as usize]
                .copy_from_slice(&input[input_pixel_base..input_pixel_base + channels as usize]);
        }
    }

    let _ = scratch; // unused by scalar reference path

    Ok(())
}
