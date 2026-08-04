//! Data movement golden fixtures — reshape, transpose, concat, split, pad, slice, `resize_nearest_neighbor`.

use crate::fixture::FixtureWriter;

pub fn generate_reshape(w: &mut FixtureWriter) {
    // Reshape [1,2,2,2] → [1,1,4,2] — same underlying buffer, different shape
    let input_shape = [1i32, 2, 2, 2];
    let output_shape = [1i32, 1, 4, 2];
    let input: Vec<i8> = (0..8).map(|i| i as i8).collect();
    // Output is identical data — reshape just changes the logical shape
    let output = input.clone(); // same data, different shape interpretation

    w.write_simple("reshape", &input_shape, &output_shape, &input, &output, &[],
        "// Reshape: same data, different logical shape [1,2,2,2] → [1,1,4,2]");
}

pub fn generate_transpose(w: &mut FixtureWriter) {
    // Transpose [1,2,3,1] → perm [0,2,1,3] → [1,3,2,1]
    // NHWC layout: batch dim unchanged
    let input_shape = [1i32, 2, 3, 1];
    let output_shape = [1i32, 3, 2, 1];
    let perm = [0i32, 2, 1, 3]; // swap height and width

    let input: Vec<i8> = vec![
        10, 20, 30, // row 0
        40, 50, 60, // row 1
    ];

    // Transpose: output[b, i, j, c] = input[b, j, i, c]
    // [0,1,0,0] ← input[0,0,1,0] = 20
    // [0,2,0,0] ← input[0,0,2,0] = 30
    // [0,0,1,0] ← input[0,1,0,0] = 40
    // [0,1,1,0] ← input[0,1,1,0] = 50
    // [0,2,1,0] ← input[0,1,2,0] = 60
    let output: Vec<i8> = vec![10, 40, 20, 50, 30, 60];

    w.write_simple("transpose", &input_shape, &output_shape, &input, &output,
        &[("perm_0", perm[0]), ("perm_1", perm[1]), ("perm_2", perm[2]), ("perm_3", perm[3])],
        "// Transpose: swap dims [0,2,1,3] (NHWC height/width swap)");
}

pub fn generate_concat(w: &mut FixtureWriter) {
    // Concatenate two [1,2,1] tensors along axis 1 → [1,4,1]
    // We encode as NHWC [1,2,1,1] + [1,2,1,1] → [1,4,1,1]
    let input1_shape = [1i32, 2, 1, 1];
    let input2_shape = [1i32, 2, 1, 1];
    let output_shape = [1i32, 4, 1, 1];
    let axis: i32 = 1; // concat along height

    let input1: Vec<i8> = vec![10, 20];
    let input2: Vec<i8> = vec![30, 40];

    let output: Vec<i8> = vec![10, 20, 30, 40];

    // Use write directly
    w.write("concat",
        &input1_shape, &input2_shape, &output_shape,
        &input1, &input2, &[],
        0, 0, -128, 127,
        &[], &[],
        &output,
        &[("axis", axis)],
    );
}

pub fn generate_split(w: &mut FixtureWriter) {
    // Split [1,4,1] along axis 1 into two [1,2,1] tensors
    let input_shape = [1i32, 4, 1, 1];
    // First split output
    let output_shape = [1i32, 2, 1, 1];
    let axis: i32 = 1;
    let num_splits: i32 = 2;

    let input: Vec<i8> = vec![1, 2, 3, 4];
    let split0: Vec<i8> = vec![1, 2];
    let split1: Vec<i8> = vec![3, 4];

    // Use the passed-in writer for both — its output_dir is already absolute.
    w.write_simple("split_v0", &input_shape, &output_shape, &input, &split0,
        &[("axis", axis), ("split_index", 0), ("num_splits", num_splits)],
        "// Split: partition [1,4,1] along axis 1 into 2 parts — output 0");
    w.write_simple("split_v1", &input_shape, &output_shape, &input, &split1,
        &[("axis", axis), ("split_index", 1), ("num_splits", num_splits)],
        "// Split: partition [1,4,1] along axis 1 into 2 parts — output 1");
}

pub fn generate_pad(w: &mut FixtureWriter) {
    // Pad [1,2,2,1] → [1,4,4,1] with zero-padding
    let input_shape = [1i32, 2, 2, 1];
    let output_shape = [1i32, 4, 4, 1];
    // Pad: before(1,1), after(1,1) on height; before(1,1), after(1,1) on width
    let pad_top = 1;
    let pad_bottom = 1;
    let pad_left = 1;
    let pad_right = 1;
    let pad_value: i8 = 0;

    let input: Vec<i8> = vec![1, 2, 3, 4]; // 2×2

    let mut output = vec![pad_value; 16];
    // Copy input into center [1:3, 1:3]
    for y in 0..2 {
        for x in 0..2 {
            output[(y + 1) * 4 + (x + 1)] = input[y * 2 + x];
        }
    }

    w.write_simple("pad", &input_shape, &output_shape, &input, &output,
        &[("pad_top", pad_top), ("pad_bottom", pad_bottom),
          ("pad_left", pad_left), ("pad_right", pad_right),
          ("pad_value", i32::from(pad_value))],
        "// Pad: zero-pad [2,2] → [4,4] with pad=(1,1,1,1)");
}

pub fn generate_slice(w: &mut FixtureWriter) {
    // Slice [1,4,4,1] → [1,2,2,1] (crop center 2×2)
    let input_shape = [1i32, 4, 4, 1];
    let output_shape = [1i32, 2, 2, 1];
    let begin: [i32; 4] = [0, 1, 1, 0]; // start at [batch=0, y=1, x=1, ch=0]
    let size: [i32; 4] = [1, 2, 2, 1];

    let input: Vec<i8> = (0..16).map(|i| i as i8).collect();

    // Extract 2×2 from center: rows 1-2, cols 1-2
    // Row 1: 4,5,6,7 → cols 1-2: 5,6
    // Row 2: 8,9,10,11 → cols 1-2: 9,10
    let output: Vec<i8> = vec![5, 6, 9, 10];

    w.write_simple("slice", &input_shape, &output_shape, &input, &output,
        &[("begin_0", begin[0]), ("begin_1", begin[1]), ("begin_2", begin[2]), ("begin_3", begin[3]),
          ("size_0", size[0]), ("size_1", size[1]), ("size_2", size[2]), ("size_3", size[3])],
        "// Slice: crop center 2×2 from 4×4 input");
}

pub fn generate_resize_nearest(w: &mut FixtureWriter) {
    // ResizeNearestNeighbor: scale [1,2,2,1] → [1,4,4,1] (2× upscale)
    // Mode: nearest, asymmetric, floor
    let input_shape = [1i32, 2, 2, 1];
    let output_shape = [1i32, 4, 4, 1];

    let input: Vec<i8> = vec![10, 20, 30, 40]; // 2×2

    let input_height = 2i32;
    let input_width = 2i32;
    let output_height = 4i32;
    let output_width = 4i32;

    // Nearest-neighbor upscale: output[y][x] = input[floor(y * 2/4)][floor(x * 2/4)]
    let mut output = vec![0i8; 16];
    for out_y in 0..output_height {
        for out_x in 0..output_width {
            let in_y = (out_y * input_height / output_height).min(input_height - 1);
            let in_x = (out_x * input_width / output_width).min(input_width - 1);
            output[(out_y * output_width + out_x) as usize] = input[(in_y * input_width + in_x) as usize];
        }
    }
    // Expected: each input pixel maps to a 2×2 block
    // [10,10,20,20]
    // [10,10,20,20]
    // [30,30,40,40]
    // [30,30,40,40]

    w.write_simple("resize_nearest_neighbor", &input_shape, &output_shape, &input, &output,
        &[("align_corners", 0), ("half_pixel_centers", 0)],
        "// ResizeNearestNeighbor: 2× upscale, asymmetric, floor");
}
