//! CudaModel shell: owns weights, workspace, stream.

use super::decode::DecodeBackend;
use super::kv::CudaKv;
use super::types::{GpuLayer, GpuMat, Kernels};
use crate::config::ModelConfig;
use cudarc::driver::{CudaContext, CudaGraph, CudaModule, CudaSlice, CudaStream};
use std::sync::Arc;

/// `CudaGraph` is not `Send` in cudarc; we only use it under the engine mutex.
pub(crate) struct SendCudaGraph(pub CudaGraph);

// SAFETY: inference is single-threaded behind `Mutex<InferenceEngine>`; graph
// is never shared across threads concurrently.
unsafe impl Send for SendCudaGraph {}
unsafe impl Sync for SendCudaGraph {}

pub struct CudaModel {
    pub cfg: ModelConfig,
    /// Selected decode / attention backend (`--decode`).
    pub decode: DecodeBackend,
    /// Device name from CUDA (e.g. `Tesla T4`, `NVIDIA GeForce RTX 3050 Ti …`).
    pub gpu_name: String,
    /// Compute capability major (e.g. 7 for T4, 8 for 3050 Ti).
    pub compute_major: i32,
    /// Compute capability minor (e.g. 5 → sm_75 with major 7).
    pub compute_minor: i32,
    /// NVRTC target used for this process (e.g. `sm_75`).
    pub nvrtc_arch: String,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) _ctx: Arc<CudaContext>,
    pub(crate) _module: Arc<CudaModule>,
    pub(crate) k: Kernels,
    pub(crate) token_embd: GpuMat,
    pub(crate) output_norm: CudaSlice<f32>,
    pub(crate) output: Option<GpuMat>,
    /// One full-vocabulary special-token column retained when an approximate
    /// active-vocabulary prefix is enabled. This keeps ChatML termination
    /// reachable without paying for the entire output head.
    pub(crate) output_special: Option<GpuMat>,
    pub(crate) output_special_id: Option<u32>,
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
    pub(crate) special_logit: CudaSlice<f32>,
    pub(crate) logits_batch: CudaSlice<f32>,
    pub(crate) argmax_buf: CudaSlice<i32>,
    pub(crate) argmax_batch: CudaSlice<i32>,
    pub(crate) tok_buf: CudaSlice<i32>,
    /// Decode activation quantization arena, reused by fused Q4 projections.
    pub(crate) q8_x: CudaSlice<i8>,
    pub(crate) q8_d: CudaSlice<f32>,
    /// Split-K GEMV partials: layout `[GEMV_SPLIT_MAX, max_gemv_cols]`.
    pub(crate) gemv_partial: CudaSlice<f32>,
    /// Capacity of one partial row (= max n_cols among gemv mats / vocab).
    pub(crate) gemv_partial_stride: usize,
    /// Device pos0 for single-token decode / CUDA graphs.
    pub(crate) d_pos0: CudaSlice<i32>,
    /// Device token id for single-token embed / CUDA graphs.
    pub(crate) d_token: CudaSlice<i32>,
    /// Flash-decoding partial workspace: n_head * n_split * (2 + head_dim).
    pub(crate) flash_partial: CudaSlice<f32>,
    /// Captured single-token decode graph (replay after updating d_pos0/d_token).
    pub(crate) decode_graph: Option<SendCudaGraph>,
    /// Graph capture attempted (success or permanent fail).
    pub(crate) graph_tried: bool,
    /// When true, attempt capture after first single-token decode.
    pub cuda_graph: bool,
    /// True once a graph is live and replaying.
    pub graph_active: bool,
}

impl CudaModel {
    pub fn set_cuda_graph(&mut self, enabled: bool) {
        self.cuda_graph = enabled;
        if !enabled {
            self.decode_graph = None;
            self.graph_tried = false;
            self.graph_active = false;
        }
    }

    pub fn graph_active(&self) -> bool {
        self.graph_active
    }

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
        eprintln!("KV    | f16  max_seq={max_seq}  ~{kv_mib:.1} MiB (all layers K+V)");
        Ok(CudaKv {
            k,
            v,
            len: 0,
            max_seq,
        })
    }

    /// Drop CUDA graph (e.g. after backend switch — not used mid-session).
    pub fn invalidate_decode_graph(&mut self) {
        self.decode_graph = None;
        self.graph_tried = false;
    }
}
