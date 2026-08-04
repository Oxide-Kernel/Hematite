#![no_std]

//! hematite-ref — reference implementation of the Hematite NN engine.

pub mod conv;
pub mod resize;
pub mod depthwise_conv;
pub mod fully_connected;
pub mod pool;
pub mod activation;
pub mod softmax;
pub mod elementwise;
pub mod data_movement;
pub mod reductions;
