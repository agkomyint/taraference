//! NVRTC device source assembled from small `.cu` fragments.
//!
//! Attention backends: one file under `attn/` each. Wire them in
//! [`crate::cuda::decode::REGISTRY`] and list the file here.
//! Delete a failed experiment = remove include + registry row + file.

/// Full CUDA source compiled once at model load.
pub const SOURCE: &str = concat!(
    include_str!("common.cu"),
    include_str!("gemm.cu"),
    include_str!("gemv.cu"),
    include_str!("embed.cu"),
    include_str!("ops.cu"),
    // ── attn backends (order free; symbols must match REGISTRY) ──
    include_str!("attn/fast_v1.cu"),
    include_str!("attn/fast_v2.cu"),
    include_str!("attn/flash.cu"),
    include_str!("attn/basic.cu"),
    include_str!("attn/online.cu"),
);
