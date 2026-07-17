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
    /// Qwen3.5: fused Q-gate half / linear z / GDN scratch.
    pub(crate) gate_buf: CudaSlice<f32>,
    pub(crate) gdn_q: CudaSlice<f32>,
    pub(crate) gdn_k: CudaSlice<f32>,
    pub(crate) gdn_v: CudaSlice<f32>,
    pub(crate) gdn_z: CudaSlice<f32>,
    pub(crate) gdn_alpha: CudaSlice<f32>,
    pub(crate) gdn_beta: CudaSlice<f32>,
    pub(crate) gdn_conv: CudaSlice<f32>,
    pub(crate) gdn_out: CudaSlice<f32>,
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
        let mut ssm = Vec::with_capacity(self.cfg.n_layer);
        let mut conv = Vec::with_capacity(self.cfg.n_layer);
        let mut is_full = Vec::with_capacity(self.cfg.n_layer);
        let mut full_bytes = 0usize;
        let mut ssm_bytes = 0usize;
        for li in 0..self.cfg.n_layer {
            if self.cfg.is_linear_layer(li) {
                is_full.push(false);
                // Minimal 1-element placeholders so index is always valid.
                k.push(self.stream.alloc_zeros::<u16>(1)?);
                v.push(self.stream.alloc_zeros::<u16>(1)?);
                let se = self.cfg.ssm_state_elems().max(1);
                let ce = self.cfg.ssm_conv_state_elems().max(1);
                ssm.push(self.stream.alloc_zeros::<f32>(se)?);
                conv.push(self.stream.alloc_zeros::<f32>(ce)?);
                ssm_bytes += (se + ce) * 4;
            } else {
                is_full.push(true);
                k.push(self.stream.alloc_zeros::<u16>(slot)?);
                v.push(self.stream.alloc_zeros::<u16>(slot)?);
                ssm.push(self.stream.alloc_zeros::<f32>(1)?);
                conv.push(self.stream.alloc_zeros::<f32>(1)?);
                full_bytes += slot * 2 * 2;
            }
        }
        let kv_mib = full_bytes as f64 / (1024.0 * 1024.0);
        let ssm_mib = ssm_bytes as f64 / (1024.0 * 1024.0);
        if self.cfg.is_hybrid() {
            eprintln!(
                "KV    | hybrid f16 full-attn ~{kv_mib:.1} MiB + GDN state ~{ssm_mib:.1} MiB  max_seq={max_seq}"
            );
        } else {
            eprintln!("KV    | f16  max_seq={max_seq}  ~{kv_mib:.1} MiB (all layers K+V)");
        }
        Ok(CudaKv {
            k,
            v,
            ssm,
            conv,
            is_full,
            len: 0,
            max_seq,
        })
    }

    /// Zero recurrent (GDN + conv) state for all linear layers.
    pub fn zero_recurrent(&mut self, cache: &mut CudaKv) -> anyhow::Result<()> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        for li in 0..self.cfg.n_layer {
            if !self.cfg.is_linear_layer(li) {
                continue;
            }
            let se = self.cfg.ssm_state_elems() as i32;
            let ce = self.cfg.ssm_conv_state_elems() as i32;
            if se > 0 {
                unsafe {
                    self.stream
                        .launch_builder(&self.k.zero_f32)
                        .arg(&mut cache.ssm[li])
                        .arg(&se)
                        .launch(LaunchConfig::for_num_elems(se as u32))?;
                }
            }
            if ce > 0 {
                unsafe {
                    self.stream
                        .launch_builder(&self.k.zero_f32)
                        .arg(&mut cache.conv[li])
                        .arg(&ce)
                        .launch(LaunchConfig::for_num_elems(ce as u32))?;
                }
            }
        }
        self.stream.synchronize()?;
        Ok(())
    }

    /// Drop CUDA graph (e.g. after backend switch — not used mid-session).
    pub fn invalidate_decode_graph(&mut self) {
        self.decode_graph = None;
        self.graph_tried = false;
    }
}
