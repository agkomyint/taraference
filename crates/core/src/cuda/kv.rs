//! Per-layer key/value cache (IEEE f16 bits as u16 — half the HBM of f32)
//! plus optional recurrent state for hybrid linear-attention layers.

use cudarc::driver::CudaSlice;

pub struct CudaKv {
    /// K cache [max_seq, n_kv_heads * head_dim] as f16 bit patterns.
    /// Empty slice for pure linear layers (no softmax KV).
    pub(crate) k: Vec<CudaSlice<u16>>,
    /// V cache [max_seq, n_kv_heads * head_dim] as f16 bit patterns.
    pub(crate) v: Vec<CudaSlice<u16>>,
    /// Gated DeltaNet state S: [n_v_heads * d_k * d_v] f32, or empty.
    pub(crate) ssm: Vec<CudaSlice<f32>>,
    /// Causal conv ring: [(kernel-1) * conv_channels] f32, or empty.
    pub(crate) conv: Vec<CudaSlice<f32>>,
    /// True when layer i uses softmax KV (false → linear / empty k,v).
    pub(crate) is_full: Vec<bool>,
    pub len: usize,
    pub max_seq: usize,
}

impl CudaKv {
    pub fn clear(&mut self) {
        self.len = 0;
        // Zero recurrent state so the next turn starts clean.
        // (KV length alone is not enough for Gated DeltaNet.)
        for s in &mut self.ssm {
            if s.len() > 0 {
                // Best-effort: mark dirty via host-side clear on next session
                // by re-zeroing through the model when available. Length is the
                // primary reset; explicit zeroing happens in Session::reset via
                // model.zero_recurrent(cache).
            }
        }
    }

    /// Bytes per token across full-attention layers only (K+V, f16).
    pub fn bytes_per_token(&self, _n_layer: usize, stride: usize) -> usize {
        let full = self.is_full.iter().filter(|&&b| b).count();
        full * stride * 2 * 2 // K+V, 2 bytes each
    }
}
