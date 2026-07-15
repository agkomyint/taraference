//! NVRTC device source assembled from small `.cu` fragments.

/// Full CUDA source compiled once at model load.
pub const SOURCE: &str = concat!(
    include_str!("common.cu"),
    include_str!("gemm.cu"),
    include_str!("gemv.cu"),
    include_str!("embed.cu"),
    include_str!("ops.cu"),
);
