//! Per-layer key/value cache (IEEE f16 bits as u16 — half the HBM of f32).

use cudarc::driver::CudaSlice;

pub struct CudaKv {
    /// K cache [max_seq, n_kv_heads * head_dim] as f16 bit patterns.
    pub(crate) k: Vec<CudaSlice<u16>>,
    /// V cache [max_seq, n_kv_heads * head_dim] as f16 bit patterns.
    pub(crate) v: Vec<CudaSlice<u16>>,
    pub len: usize,
    pub max_seq: usize,
}

impl CudaKv {
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Bytes per token across all layers (K+V, f16).
    pub fn bytes_per_token(&self, n_layer: usize, stride: usize) -> usize {
        n_layer * stride * 2 * 2 // K+V, 2 bytes each
    }
}
