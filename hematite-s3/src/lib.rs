#![no_std]
// Xtensa inline asm is still experimental in the esp-rs fork (stock rustc gate
// #93335). The SIMD glue in this crate uses `core::arch::asm!`/`global_asm!` to
// call the vendored esp-dl TIE728 entry points, so the gate is required when
// compiling for the device. cfg_attr keeps it off the host toolchain (stable).
#![cfg_attr(target_arch = "xtensa", feature(asm_experimental_arch))]

//! hematite-s3 — ESP32-S3 optimized backend.

pub mod backend;
pub mod conv1x1;
pub mod accx;
pub mod elementwise;
pub mod fused;
pub mod conv3x3;
pub mod data_movement;
pub mod depthwise;
pub mod gemm;
pub mod pool;
pub mod softmax;
pub mod activations;
pub mod reductions;
