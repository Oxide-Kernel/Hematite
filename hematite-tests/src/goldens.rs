//! Golden fixture re-exports — each module embeds a generated const-array fixture.
//! Used by Phase 2 TDD tests via `use hematite_tests::goldens::conv2d`.

#![allow(dead_code)]

pub mod conv2d_1x1 {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/conv2d_1x1.rs"));
}

pub mod conv2d_3x3 {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/conv2d_3x3.rs"));
}

pub mod depthwise_conv2d {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/depthwise_conv2d.rs"));
}

pub mod fully_connected {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/fully_connected.rs"));
}

pub mod average_pool_2d {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/average_pool_2d.rs"));
}

pub mod max_pool_2d {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/max_pool_2d.rs"));
}

pub mod softmax {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/softmax.rs"));
}

pub mod relu {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/relu.rs"));
}

pub mod relu6 {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/relu6.rs"));
}

pub mod hard_swish {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/hard_swish.rs"));
}

pub mod leaky_relu {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/leaky_relu.rs"));
}

pub mod prelu {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/prelu.rs"));
}

pub mod elementwise_add {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/elementwise_add.rs"));
}

pub mod elementwise_mul {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/elementwise_mul.rs"));
}

pub mod elementwise_sub {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/elementwise_sub.rs"));
}

pub mod quantize {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/quantize.rs"));
}

pub mod dequantize {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/dequantize.rs"));
}

pub mod reshape {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/reshape.rs"));
}

pub mod transpose {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/transpose.rs"));
}

pub mod concat {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/concat.rs"));
}

pub mod split_v0 {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/split_v0.rs"));
}

pub mod split_v1 {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/split_v1.rs"));
}

pub mod pad {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/pad.rs"));
}

pub mod slice {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/slice.rs"));
}

pub mod resize_nearest_neighbor {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/resize_nearest_neighbor.rs"));
}

// T0 — MatMul (BatchMatMul reference path)
pub mod matmul {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/matmul.rs"));
}

// T1 — standalone activations added in T5.0
pub mod sigmoid {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/sigmoid.rs"));
}

pub mod tanh {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/tanh.rs"));
}

// T3 — Recurrent
pub mod lstm {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/lstm.rs"));
}

pub mod svdf {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/svdf.rs"));
}

pub mod gru {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/gru.rs"));
}

// T4 — Reductions
pub mod mean {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/mean.rs"));
}

pub mod sum {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/sum.rs"));
}

pub mod argmax {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/argmax.rs"));
}

pub mod argmin {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/argmin.rs"));
}

pub mod l2_norm {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/l2_norm.rs"));
}

pub mod reduce_max {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/reduce_max.rs"));
}

pub mod reduce_min {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/reduce_min.rs"));
}

// ── Model goldens (captured from executed TFLite interpreter) ──
pub mod models {
    pub mod sine {
        include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../hematite-tests/goldens/models/sine.rs"));
    }
}
