#![no_std]

//! hematite-s3 — ESP32-S3 optimized backend.

pub mod conv1x1;
pub mod elementwise;
pub mod conv3x3;
pub mod depthwise;
pub mod gemm;
pub mod pool;
pub mod softmax;
pub mod activations;
pub mod reductions;
