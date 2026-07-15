//! High-level inference engine: load model once, run completions.

use crate::chat::{format_chatml, ChatMessage};
use crate::cuda::{CudaKv, CudaModel, DecodeBackend};
use crate::session::{Session, SessionOptions, TurnStats};
use crate::tokenizer::Tokenizer;
use crate::ModelConfig;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use taraference_gguf::GgufFile;

/// Configuration for loading an engine.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub model_path: PathBuf,
    pub max_seq: usize,
    pub max_new: usize,
    pub decode: DecodeBackend,
    pub default_system: String,
    /// Attempt CUDA graph capture for single-token decode after warm-up.
    pub cuda_graph: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            max_seq: 5000,
            max_new: 512,
            decode: DecodeBackend::default(),
            default_system: "You are a helpful assistant.".into(),
            cuda_graph: true,
        }
    }
}

/// Owns GPU model + tokenizer + one KV arena. One engine per process (single CUDA stream).
pub struct InferenceEngine {
    model: CudaModel,
    tok: Tokenizer,
    /// Reused across requests (cleared per stateless completion).
    kv: CudaKv,
    pub model_path: PathBuf,
    /// GGUF file stem — the only model this process serves (OpenAI `model` id).
    pub model_id: String,
    pub max_seq: usize,
    pub max_new: usize,
    pub default_system: String,
    pub weight_gib: f64,
}

impl InferenceEngine {
    pub fn load(cfg: EngineConfig) -> Result<Self> {
        let path = cfg.model_path;
        eprintln!("loading {} …", path.display());
        let gguf = GgufFile::open(&path).with_context(|| format!("open {}", path.display()))?;
        let weight_gib = gguf.total_tensor_bytes() as f64 / (1024.0 * 1024.0 * 1024.0);
        let tok = Tokenizer::from_gguf(&gguf)?;
        let mut model = CudaModel::load_with(&gguf, cfg.decode)?;
        model.set_cuda_graph(cfg.cuda_graph);
        let max_seq = cfg.max_seq.min(model.cfg.n_ctx);
        let kv = model.alloc_kv(max_seq)?;
        let model_id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("taraference")
            .to_string();
        eprintln!(
            "flags | cuda_graph={} | decode={}",
            cfg.cuda_graph,
            cfg.decode.name()
        );

        Ok(Self {
            model,
            tok,
            kv,
            model_path: path,
            model_id,
            max_seq,
            max_new: cfg.max_new,
            default_system: cfg.default_system,
            weight_gib,
        })
    }

    pub fn load_path(
        model: impl AsRef<Path>,
        decode: DecodeBackend,
        max_seq: usize,
        max_new: usize,
    ) -> Result<Self> {
        Self::load(EngineConfig {
            model_path: model.as_ref().to_path_buf(),
            max_seq,
            max_new,
            decode,
            ..EngineConfig::default()
        })
    }

    pub fn load_with_flags(
        model: impl AsRef<Path>,
        decode: DecodeBackend,
        max_seq: usize,
        max_new: usize,
        cuda_graph: bool,
    ) -> Result<Self> {
        Self::load(EngineConfig {
            model_path: model.as_ref().to_path_buf(),
            max_seq,
            max_new,
            decode,
            cuda_graph,
            ..EngineConfig::default()
        })
    }

    pub fn cfg(&self) -> &ModelConfig {
        &self.model.cfg
    }

    pub fn decode(&self) -> DecodeBackend {
        self.model.decode
    }

    /// GPU name from CUDA (for profile logs across multi-GPU setups).
    pub fn gpu_name(&self) -> &str {
        &self.model.gpu_name
    }

    /// Compute capability `(major, minor)` of the loaded device.
    pub fn compute_capability(&self) -> (i32, i32) {
        (self.model.compute_major, self.model.compute_minor)
    }

    /// NVRTC `--gpu-architecture` used when compiling kernels (e.g. `sm_75`).
    pub fn nvrtc_arch(&self) -> &str {
        &self.model.nvrtc_arch
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    pub fn model_mut(&mut self) -> &mut CudaModel {
        &mut self.model
    }

    /// Interactive / multi-turn session (CLI REPL + profile).
    /// Uses the engine's shared KV (reset first for a clean chat).
    pub fn session(&mut self, opts: SessionOptions) -> Session<'_> {
        self.kv.clear();
        Session::with_kv(&mut self.model, &self.tok, &mut self.kv, opts)
    }

    /// Stateless OpenAI-style completion: full message list → one assistant reply.
    /// Resets KV each call (client is source of truth for history).
    pub fn chat_completion(
        &mut self,
        messages: &[ChatMessage],
        max_tokens: Option<usize>,
    ) -> Result<TurnStats> {
        self.chat_completion_stream(messages, max_tokens, |_| {})
    }

    /// Same as [`Self::chat_completion`], invoking `on_token` for each decoded piece (SSE).
    pub fn chat_completion_stream<F>(
        &mut self,
        messages: &[ChatMessage],
        max_tokens: Option<usize>,
        on_token: F,
    ) -> Result<TurnStats>
    where
        F: FnMut(&str),
    {
        if messages.is_empty() {
            anyhow::bail!("messages must not be empty");
        }
        let max_new = max_tokens.unwrap_or(self.max_new).max(1);
        let prompt = format_chatml(messages, Some(&self.default_system));
        let opts = SessionOptions {
            max_new,
            system: self.default_system.clone(),
            print_stream: false,
            print_stats: false,
        };
        self.kv.clear();
        let mut session = Session::with_kv(&mut self.model, &self.tok, &mut self.kv, opts);
        session.complete_prompt_stream(&prompt, on_token)
    }
}
