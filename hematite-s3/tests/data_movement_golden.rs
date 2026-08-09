// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hematite Contributors.
//
//! Golden-vector tests for the hematite-s3 data-movement kernels
//! (plan todo 25 amendment).
//!
//! Each test drives the s3 free kernel and the `RefBackend` trait method
//! (the scalar oracle) with identical params + input and asserts
//! element-for-element bit-exact equality. These ops are pure data
//! movement — no arithmetic, no rounding — so bit-exactness is by
//! construction; the tests lock it against regressions.

use hematite_core::op_params::{
    ConcatParams, FusedActivation, PadParams, ReshapeParams, ResizeNearestParams,
    SliceParams, SplitParams, TransposeParams,
};
use hematite_core::KernelBackend;
use hematite_ref::RefBackend;
use hematite_s3::data_movement;

fn assert_bit_exact(actual: &[i8], expected: &[i8], name: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{name}: output length {} != expected length {}",
        actual.len(),
        expected.len(),
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            a, e,
            "{name}: mismatch at index {i}: s3={a}, ref={e}",
        );
    }
}

/// Deterministic int8 ramp centered on a nonzero zero point (the PAD case
/// needs data that is not confusable with the raw-0 fill).
fn make_pattern(len: usize, zero_point: i32) -> Vec<i8> {
    (0..len)
        .map(|i| {
            let v = (i as i32 * 7 + 3) % 128 - 64 + zero_point;
            v.clamp(-128, 127) as i8
        })
        .collect()
}

// ── reshape ─────────────────────────────────────────────────────────────────

#[test]
fn reshape_golden() {
    let input = make_pattern(24, 0);
    let params = ReshapeParams { shape: [1, 2, 12, 1], shape_count: 4 };
    let mut s3_out = [0i8; 24];
    let mut ref_out = [0i8; 24];

    data_movement::reshape(&input, &params, &mut s3_out).expect("s3 reshape returned Err");
    RefBackend.reshape(&input, &params, &mut ref_out).expect("ref reshape returned Err");
    assert_bit_exact(&s3_out, &ref_out, "reshape [1,2,3,4] -> [1,2,12,1]");

    // Length mismatch must be reported, not truncated.
    let mut short = [0i8; 12];
    assert_eq!(
        data_movement::reshape(&input, &params, &mut short),
        Err(hematite_core::KernelError::ShapeMismatch),
        "reshape with mismatched output length must Err"
    );
}

// ── transpose ───────────────────────────────────────────────────────────────

#[test]
fn transpose_golden_perm3() {
    // 3-dim perm with perm_count = 3: dim 3 defaults to identity.
    let input = make_pattern(24, 0);
    let params = TransposeParams {
        input_shape: [1, 2, 3, 4],
        perm: [2, 0, 1, 3, 0, 0, 0, 0],
        perm_count: 3,
    };
    let mut s3_out = [0i8; 24];
    let mut ref_out = [0i8; 24];

    data_movement::transpose(&input, &params, &mut s3_out).expect("s3 transpose returned Err");
    RefBackend.transpose(&input, &params, &mut ref_out).expect("ref transpose returned Err");
    assert_bit_exact(&s3_out, &ref_out, "transpose [1,2,3,4] perm [2,0,1] (count 3)");

    // Sanity: the permuted output is not an identity copy.
    assert_ne!(&s3_out[..], &input[..], "transpose must actually permute");
}

#[test]
fn transpose_golden_perm4() {
    // Full 4-dim permutation, the mobilenet_v2 op #1 shape class.
    let input = make_pattern(24, 0);
    let params = TransposeParams {
        input_shape: [1, 2, 3, 4],
        perm: [2, 3, 0, 1, 0, 0, 0, 0],
        perm_count: 4,
    };
    let mut s3_out = [0i8; 24];
    let mut ref_out = [0i8; 24];

    data_movement::transpose(&input, &params, &mut s3_out).expect("s3 transpose returned Err");
    RefBackend.transpose(&input, &params, &mut ref_out).expect("ref transpose returned Err");
    assert_bit_exact(&s3_out, &ref_out, "transpose [1,2,3,4] perm [2,3,0,1]");
}

// ── pad ─────────────────────────────────────────────────────────────────────

#[test]
fn pad_golden() {
    // [1,3,3,1] padded 1 on each side -> [1,5,5,1] (total pad 2 per dim).
    // Nonzero zero-point data (-14 class, like mobilenet_v2's PAD inputs) so
    // the copied interior is distinguishable from the raw-0 fill border.
    let input = make_pattern(9, -14);
    let params = PadParams {
        input_shape: [1, 3, 3, 1],
        output_shape: [1, 5, 5, 1],
        left_padding: [0, 1, 1, 0],
        left_padding_count: 4,
        right_padding: [0, 1, 1, 0],
        right_padding_count: 4,
    };
    let mut s3_out = [0i8; 25];
    let mut ref_out = [0i8; 25];

    data_movement::pad_op(&input, &params, &mut s3_out, &mut [])
        .expect("s3 pad returned Err");
    RefBackend.pad(&input, &params, &mut ref_out).expect("ref pad returned Err");
    assert_bit_exact(&s3_out, &ref_out, "pad [1,3,3,1] total pad 2, zp-class data");

    // The 1-element border must be the raw-0 fill, interior must be the data.
    assert_eq!(s3_out[0], 0, "pad fill border must be 0");
    assert_eq!(s3_out[12], input[4], "pad interior must carry the input");
}

// ── slice ───────────────────────────────────────────────────────────────────

#[test]
fn slice_golden() {
    let input = make_pattern(16, 0);
    let params = SliceParams {
        input_shape: [1, 4, 4, 1],
        begin: [0, 1, 1, 0],
        size: [1, 2, 2, 1],
    };
    let mut s3_out = [0i8; 4];
    let mut ref_out = [0i8; 4];

    data_movement::slice_op(&input, &params, &mut s3_out, &mut [])
        .expect("s3 slice returned Err");
    RefBackend.slice(&input, &params, &mut ref_out).expect("ref slice returned Err");
    assert_bit_exact(&s3_out, &ref_out, "slice [1,4,4,1] begin [0,1,1,0] size [1,2,2,1]");

    // begin [1,1] of a 4x4 row-major grid: rows 1..3, cols 1..3.
    assert_eq!(s3_out[0], input[5], "slice corner must equal input[5]");
}

// ── concat ──────────────────────────────────────────────────────────────────

#[test]
fn concat_golden_axis_h() {
    let input_a = make_pattern(24, 0);
    let input_b = make_pattern(24, 0);
    let params = ConcatParams {
        axis: 1,
        activation: FusedActivation::None,
        input_shape_a: [1, 2, 3, 4],
        input_shape_b: [1, 2, 3, 4],
        output_shape: [1, 4, 3, 4],
    };
    let mut s3_out = [0i8; 48];
    let mut ref_out = [0i8; 48];

    data_movement::concat_op(&input_a, &input_b, &params, &mut s3_out, &mut [])
        .expect("s3 concat returned Err");
    RefBackend.concat(&input_a, &input_b, &params, &mut ref_out).expect("ref concat returned Err");
    assert_bit_exact(&s3_out, &ref_out, "concat axis=1 [1,2,3,4]+[1,2,3,4]");
}

#[test]
fn concat_golden_axis_c() {
    // Channel-axis concat with unequal C on the two inputs.
    let input_a = make_pattern(12, 0);
    let input_b = make_pattern(18, 0);
    let params = ConcatParams {
        axis: 3,
        activation: FusedActivation::None,
        input_shape_a: [1, 2, 3, 2],
        input_shape_b: [1, 2, 3, 3],
        output_shape: [1, 2, 3, 5],
    };
    let mut s3_out = [0i8; 30];
    let mut ref_out = [0i8; 30];

    data_movement::concat_op(&input_a, &input_b, &params, &mut s3_out, &mut [])
        .expect("s3 concat returned Err");
    RefBackend.concat(&input_a, &input_b, &params, &mut ref_out).expect("ref concat returned Err");
    assert_bit_exact(&s3_out, &ref_out, "concat axis=3 [1,2,3,2]+[1,2,3,3]");
}

// ── split ───────────────────────────────────────────────────────────────────

#[test]
fn split_golden_axis_h() {
    let input = make_pattern(48, 0);
    let params = SplitParams {
        num_splits: 2,
        axis: 1,
        input_shape: [1, 4, 3, 4],
        output_shape_a: [1, 2, 3, 4],
        output_shape_b: [1, 2, 3, 4],
    };
    let mut s3_a = [0i8; 24];
    let mut s3_b = [0i8; 24];
    let mut ref_a = [0i8; 24];
    let mut ref_b = [0i8; 24];

    data_movement::split_op(&input, 0, &params, &mut s3_a, &mut []).expect("s3 split 0 Err");
    data_movement::split_op(&input, 1, &params, &mut s3_b, &mut []).expect("s3 split 1 Err");
    RefBackend.split(&input, &params, &mut ref_a, &mut ref_b).expect("ref split Err");
    assert_bit_exact(&s3_a, &ref_a, "split axis=1 output_a");
    assert_bit_exact(&s3_b, &ref_b, "split axis=1 output_b");

    // Both halves together reconstruct the input (row-major H split).
    let mut joined = [0i8; 48];
    joined[..24].copy_from_slice(&s3_a);
    joined[24..].copy_from_slice(&s3_b);
    assert_eq!(&joined[..], &input[..], "split axis=1 halves must reconstruct input");
}

#[test]
fn split_golden_axis_c() {
    let input = make_pattern(24, 0);
    let params = SplitParams {
        num_splits: 2,
        axis: 3,
        input_shape: [1, 2, 3, 4],
        output_shape_a: [1, 2, 3, 2],
        output_shape_b: [1, 2, 3, 2],
    };
    let mut s3_a = [0i8; 12];
    let mut s3_b = [0i8; 12];
    let mut ref_a = [0i8; 12];
    let mut ref_b = [0i8; 12];

    data_movement::split_op(&input, 0, &params, &mut s3_a, &mut []).expect("s3 split 0 Err");
    data_movement::split_op(&input, 1, &params, &mut s3_b, &mut []).expect("s3 split 1 Err");
    RefBackend.split(&input, &params, &mut ref_a, &mut ref_b).expect("ref split Err");
    assert_bit_exact(&s3_a, &ref_a, "split axis=3 output_a");
    assert_bit_exact(&s3_b, &ref_b, "split axis=3 output_b");

    // First pixel of each half = first two channels of input pixel 0.
    assert_eq!(s3_a[0], input[0]);
    assert_eq!(s3_b[0], input[2]);
}

// ── resize_nearest ──────────────────────────────────────────────────────────

#[test]
fn resize_nearest_golden_upscale() {
    // 2x upscale [1,2,2,1] -> [1,4,4,1] (asymmetric/floor mode).
    let input = make_pattern(4, 0);
    let params = ResizeNearestParams {
        input_shape: [1, 2, 2, 1],
        output_shape: [1, 4, 4, 1],
        align_corners: 0,
        half_pixel_centers: 0,
    };
    let mut s3_out = [0i8; 16];
    let mut ref_out = [0i8; 16];

    data_movement::resize_nearest_neighbor(&input, &params, &mut s3_out, &mut [])
        .expect("s3 resize returned Err");
    RefBackend.resize_nearest(&input, &params, &mut ref_out).expect("ref resize returned Err");
    assert_bit_exact(&s3_out, &ref_out, "resize_nearest upscale 2x");

    // Nearest-neighbor: floor mapping tiles each input pixel into 2x2
    // output blocks — rows 0-1 repeat input rows 0-1, rows 2-3 input rows
    // 2-3 (input = [in0, in1, in2, in3]).
    assert_eq!(s3_out[0], input[0]);
    assert_eq!(s3_out[5], input[0]);
    assert_eq!(s3_out[10], input[3]);
    assert_eq!(s3_out[15], input[3]);
}

#[test]
fn resize_nearest_golden_downscale() {
    // 2x downscale [1,4,4,1] -> [1,2,2,1] (floor mapping).
    let input = make_pattern(16, 0);
    let params = ResizeNearestParams {
        input_shape: [1, 4, 4, 1],
        output_shape: [1, 2, 2, 1],
        align_corners: 0,
        half_pixel_centers: 0,
    };
    let mut s3_out = [0i8; 4];
    let mut ref_out = [0i8; 4];

    data_movement::resize_nearest_neighbor(&input, &params, &mut s3_out, &mut [])
        .expect("s3 resize returned Err");
    RefBackend.resize_nearest(&input, &params, &mut ref_out).expect("ref resize returned Err");
    assert_bit_exact(&s3_out, &ref_out, "resize_nearest downscale 2x");
}
