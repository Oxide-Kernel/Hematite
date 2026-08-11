// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Data-movement scalar kernels — reshape, transpose, concat, split, pad,
//! slice, resize_nearest.
//!
//! Scalar copies of the `hematite-ref` data-movement kernels (plan todo 25
//! amendment: S3Backend must run the zoo models whose op sequence includes
//! RESHAPE=22 / TRANSPOSE=39 / PAD=34). These are pure data movement — no
//! arithmetic, no rounding — so the s3 copies are bit-exact against the
//! reference by construction. The logic mirrors
//! `hematite-ref/src/data_movement.rs`, `hematite-ref/src/resize.rs`, and
//! the inline `transpose_impl` / `reshape` in
//! `hematite-ref/src/backend.rs`; the ONLY documented divergence is the PAD
//! fill-value discussion below (which changes nothing about the fill this
//! crate actually performs).
//!
//! # Batch constraint
//!
//! Only batch = 1 is supported (per the static-shape constraint).
//! `batch > 1` returns [`KernelError::Unsupported`].
//!
//! # PAD fill value — TFLM semantics and the params limitation
//!
//! TFLM `tensorflow/lite/micro/kernels/pad.cc` @ pinned SHA
//! `18b9e6f2a8c5a9518e588f59c2ba16ef7ef9d551`, `kTfLiteInt8` branch
//! (verbatim):
//!
//! ```cpp
//! int8_t pad_value;
//! if (constant_values == nullptr) {
//!   pad_value = static_cast<uint8_t>(data->output_zero_point);
//! } else {
//!   pad_value = *tflite::micro::GetTensorData<int8_t>(constant_values);
//! }
//! ```
//!
//! So TFLM fills pads with the output tensor's zero point when the model
//! has no constant-values tensor (mobilenet_v2's 18 PAD ops are this case;
//! output zp == input zp == −14, and executed TFLM fills −14 — evidence:
//! `local-notes/evidence/simd-zoo-hardening/task-10-goldens.log` §9(b)).
//!
//! **Hematite's [`PadParams`] carries NO zero point** and the
//! [`KernelBackend::pad`] trait method has no pad-value argument (codegen
//! emits `backend.pad(src, &PAD_PARAMS_i, dst)`), so the s3 kernel — like
//! [`hematite-ref`]'s `pad_op` — cannot know the zero point. Implementing
//! the TFLM fill requires param plumbing (a zero-point field on
//! `PadParams` + codegen emission), which is the T10-flagged follow-up
//! (task-10 log §9(b): "needs param plumbing"), OUT OF SCOPE for this
//! amendment (its file list does not include `hematite-core`/codegen).
//! Filling the TFLM value here would ALSO break the relative `s3 == ref`
//! model gate (mobilenet_v2's PAD fill propagates through the conv chain —
//! 0-vs-−14 drives 861/1000 output deltas), so this copy fills raw 0,
//! bit-exact with the reference. The mv2 divergence vs the executed-TFLM
//! golden therefore remains a documented PAD/rounding divergence shared by
//! BOTH backends; the zero-point fill is tracked in
//! `local-notes/notepads/simd-zoo-hardening/learnings.md` and the plan follow-up.

use hematite_core::op_params::{
    ConcatParams, PadParams, ReshapeParams, ResizeNearestParams, SliceParams,
    SplitParams, TransposeParams,
};
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

// ── reshape ────────────────────────────────────────────────────────────────

/// Reshape is a flat copy: TFLM's int8 `Reshape` is a metadata-only op
/// (same underlying buffer, new logical shape). Mirrors the inline
/// `KernelBackend::reshape` of `hematite-ref` (flat copy + length check).
pub fn reshape(
    input: &[i8],
    _params: &ReshapeParams,
    output: &mut [i8],
) -> Result<(), KernelError> {
    if input.len() != output.len() {
        return Err(KernelError::ShapeMismatch);
    }
    output.copy_from_slice(input);
    Ok(())
}

// ── transpose ──────────────────────────────────────────────────────────────

/// Compute the NHWC row-major strides of a `[i32; 4]` shape.
#[inline(always)]
fn nhwc_strides(shape: &[i32; 4]) -> [usize; 4] {
    [
        shape[1] as usize * shape[2] as usize * shape[3] as usize,
        shape[2] as usize * shape[3] as usize,
        shape[3] as usize,
        1,
    ]
}

/// Decode a flat NHWC linear index into 4D coordinates.
#[inline(always)]
fn decode_4d(idx: usize, shape: &[i32; 4], strides: &[usize; 4]) -> [usize; 4] {
    [
        (idx / strides[0]) % shape[0] as usize,
        (idx / strides[1]) % shape[1] as usize,
        (idx / strides[2]) % shape[2] as usize,
        (idx / strides[3]) % shape[3] as usize,
    ]
}

/// Encode 4D coordinates into a flat NHWC linear index.
#[inline(always)]
fn encode_4d(coords: [usize; 4], strides: &[usize; 4]) -> usize {
    coords[0] * strides[0] + coords[1] * strides[1] + coords[2] * strides[2] + coords[3]
}

/// General 4D transpose (scatter formulation).
///
/// `output[coords[perm[0]], coords[perm[1]], coords[perm[2]], coords[perm[3]]] =
/// input[coords]` for every input position in NHWC row-major order — the
/// same `output[i] = input[perm applied to coords]` mapping TFLM uses.
///
/// `perm` entries beyond `perm_count` default to identity. This is a pure
/// data-movement operation: no arithmetic, no offset, no requantize.
/// Mirrors `hematite-ref`'s inline `transpose_impl` exactly.
pub fn transpose(
    input: &[i8],
    params: &TransposeParams,
    output: &mut [i8],
) -> Result<(), KernelError> {
    let in_shape = params.input_shape;
    let perm_count = usize::from(params.perm_count.max(0) as u8);

    // Effective permutation: identity for dims past perm_count.
    let perm: [usize; 4] = [
        if 0 < perm_count { params.perm[0] as usize } else { 0 },
        if 1 < perm_count { params.perm[1] as usize } else { 1 },
        if 2 < perm_count { params.perm[2] as usize } else { 2 },
        if 3 < perm_count { params.perm[3] as usize } else { 3 },
    ];
    for &p in &perm {
        if p >= 4 {
            return Err(KernelError::ShapeMismatch);
        }
    }

    // Output shape is the input shape permuted.
    let out_shape = [in_shape[perm[0]], in_shape[perm[1]], in_shape[perm[2]], in_shape[perm[3]]];

    if input.len() != shape_product(&in_shape) || output.len() != shape_product(&out_shape) {
        return Err(KernelError::ShapeMismatch);
    }

    let in_strides = nhwc_strides(&in_shape);
    let out_strides = nhwc_strides(&out_shape);

    for (idx, &val) in input.iter().enumerate() {
        let coords = decode_4d(idx, &in_shape, &in_strides);
        let out_coords = [coords[perm[0]], coords[perm[1]], coords[perm[2]], coords[perm[3]]];
        let out_idx = encode_4d(out_coords, &out_strides);
        output[out_idx] = val;
    }

    Ok(())
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
/// Mirrors TFLM `Pad` with constant value 0 — see the module docs for the
/// full TFLM fill-value analysis (`output_zero_point` vs raw 0) and why the
/// s3 copy keeps raw 0 (params carry no zero point; the relative `s3 == ref`
/// gate requires identical fill to [`hematite-ref`]'s `pad_op`).
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
    // TFLM pads with the output zero point when no constant-values tensor
    // exists (pad.cc @ pinned SHA — see module docs). PadParams carries no
    // zero point, so like hematite-ref this fills raw 0; the zero-point
    // fill is the documented T10 follow-up (param plumbing) and the
    // s3==ref model gate requires this identical fill.
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

// ── resize_nearest ──────────────────────────────────────────────────────────

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

    let _ = scratch; // unused by scalar data-movement path

    Ok(())
}
