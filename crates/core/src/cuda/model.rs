//! CudaModel shell: owns weights, workspace, stream.

use super::decode::DecodeBackend;
use super::kv::CudaKv;
use super::types::{GpuLayer, GpuMat, Kernels};
use crate::config::ModelConfig;
use cudarc::driver::{CudaContext, CudaModule, CudaSlice, CudaStream};
use std::sync::Arc;

pub struct CudaModel {
    pub cfg: ModelConfig,
    /// Selected decode / attention backend (`--decode`).
    pub decode: DecodeBackend,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) _ctx: Arc<CudaContext>,
    pub(crate) _module: Arc<CudaModule>,
    pub(crate) k: Kernels,
    pub(crate) token_embd: GpuMat,
    pub(crate) output_norm: CudaSlice<f32>,
    pub(crate) output: Option<GpuMat>,
    pub(crate) layers: Vec<GpuLayer>,
    // batch workspace [MAX_BATCH, …]
    pub(crate) x: CudaSlice<f32>,
    pub(crate) xb: CudaSlice<f32>,
    pub(crate) xb2: CudaSlice<f32>,
    pub(crate) q: CudaSlice<f32>,
    pub(crate) k_buf: CudaSlice<f32>,
    pub(crate) v_buf: CudaSlice<f32>,
    pub(crate) hb: CudaSlice<f32>,
    pub(crate) hb2: CudaSlice<f32>,
    pub(crate) x1: CudaSlice<f32>,
    pub(crate) xb1: CudaSlice<f32>,
    pub(crate) logits: CudaSlice<f32>,
    pub(crate) argmax_buf: CudaSlice<i32>,
    pub(crate) tok_buf: CudaSlice<i32>,
    /// Split-K GEMV partials: layout `[GEMV_SPLIT_MAX, max_gemv_cols]`.
    pub(crate) gemv_partial: CudaSlice<f32>,
    /// Capacity of one partial row (= max n_cols among gemv mats / vocab).
    pub(crate) gemv_partial_stride: usize,
}

impl CudaModel {
    pub fn alloc_kv(&self, max_seq: usize) -> anyhow::Result<CudaKv> {
        let stride = self.cfg.n_head_kv * self.cfg.head_dim();
        let slot = max_seq * stride;
        let mut k = Vec::with_capacity(self.cfg.n_layer);
        let mut v = Vec::with_capacity(self.cfg.n_layer);
        // f16 KV: half the VRAM/BW of f32 (stored as u16 bit patterns).
        for _ in 0..self.cfg.n_layer {
            k.push(self.stream.alloc_zeros::<u16>(slot)?);
            v.push(self.stream.alloc_zeros::<u16>(slot)?);
        }
        let kv_mib = (self.cfg.n_layer * slot * 2 * 2) as f64 / (1024.0 * 1024.0);
        eprintln!(
            "KV    | f16  max_seq={max_seq}  ~{kv_mib:.1} MiB (all layers K+V)"
        );
        Ok(CudaKv {
            k,
            v,
            len: 0,
            max_seq,
        })
    }
}
