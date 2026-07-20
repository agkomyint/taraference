//! High-level inference engine: load model once, run completions.

use crate::chat::{format_chatml, ChatMessage};
use crate::backend::BackendKind;
use crate::cuda::{CudaKv, CudaModel, DecodeBackend};
use crate::session::{Session, SessionOptions, TurnStats};
use crate::tokenizer::Tokenizer;
use crate::ModelConfig;
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use taraference_gguf::GgufFile;

fn dir_bytes(dir: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            if let Ok(m) = e.metadata() {
                if m.is_file() {
                    total += m.len();
                }
            }
        }
    }
    total
}

/// Relative tokenizer GGUF candidates searched after env + pack-local paths.
fn default_tokenizer_candidates() -> &'static [&'static str] {
    &[
        "models/tara-sprint-80m-Q8_0.gguf",
        "models/tara-sprint-50m-Q8_0.gguf",
        "models/tara-sprint-150m-Q8_0.gguf",
    ]
}

/// Tokenizer GGUF for MoE packs: `TARAFER_TOKENIZER_GGUF`, pack-local `tokenizer.gguf`,
/// or a Tara-Sprint GGUF under `models/`.
fn resolve_tokenizer_gguf(pack_dir: &Path) -> Result<PathBuf> {
    if let Ok(p) = std::env::var("TARAFER_TOKENIZER_GGUF") {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return Ok(pb);
        }
        bail!("TARAFER_TOKENIZER_GGUF not a file: {}", pb.display());
    }
    let local = pack_dir.join("tokenizer.gguf");
    if local.is_file() {
        return Ok(local);
    }
    for rel in default_tokenizer_candidates() {
        let c = PathBuf::from(rel);
        if c.is_file() {
            return Ok(c);
        }
    }
    bail!(
        "MoE pack needs a tokenizer GGUF. Set TARAFER_TOKENIZER_GGUF=path/to/tara-sprint-*.gguf \
         (same vocab as training) or place tokenizer.gguf next to meta.json"
    );
}

#[cfg(test)]
mod tokenizer_resolve_tests {
    use super::default_tokenizer_candidates;

    #[test]
    fn default_candidates_are_relative_only() {
        for c in default_tokenizer_candidates() {
            assert!(!c.contains(':'), "no machine-local absolute paths: {c}");
            assert!(
                c.starts_with("models/"),
                "expected models/ relative path, got {c}"
            );
        }
    }
}

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
    /// Architecture-specific accelerator selected for this model.
    pub backend_kind: BackendKind,
}

impl InferenceEngine {
    pub fn load(cfg: EngineConfig) -> Result<Self> {
        let path = cfg.model_path.clone();
        eprintln!("loading {} …", path.display());

        // Directory with meta.json → Tara MoE Q8 pack (sparse experts).
        if path.is_dir() {
            return Self::load_moe_pack(cfg, path);
        }

        let gguf = GgufFile::open(&path).with_context(|| format!("open {}", path.display()))?;
        let weight_gib = gguf.total_tensor_bytes() as f64 / (1024.0 * 1024.0 * 1024.0);
        let tok = Tokenizer::from_gguf(&gguf)?;
        let mut model = CudaModel::load_with(&gguf, cfg.decode)?;
        model.set_cuda_graph(cfg.cuda_graph);
        let backend_kind = BackendKind::from_config(&model.cfg);
        let requested_max_seq = cfg.max_seq.min(model.cfg.n_ctx);
        let max_seq = backend_kind.max_seq(cfg.max_seq, model.cfg.n_ctx, weight_gib);
        if max_seq != requested_max_seq {
            eprintln!(
                "backend | {} max_seq {requested_max_seq} → {max_seq} (weights {weight_gib:.2} GiB; set TARAFER_LONG_CTX=1 to keep)",
                backend_kind.name()
            );
        }
        eprintln!("backend | {}", backend_kind.name());
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
            backend_kind,
        })
    }

    /// Load Tara MoE Q8 pack directory + tokenizer from a sibling/env GGUF.
    fn load_moe_pack(cfg: EngineConfig, path: PathBuf) -> Result<Self> {
        let meta = path.join("meta.json");
        if !meta.is_file() {
            bail!(
                "directory {} is not a Tara MoE pack (missing meta.json)",
                path.display()
            );
        }
        let mut model = CudaModel::load_tara_moe_pack(&path, cfg.decode)?;
        // Device top-k + packed experts → fixed launch graph (real routing, no FIXED cheat).
        model.set_cuda_graph(cfg.cuda_graph);

        let tok_gguf = resolve_tokenizer_gguf(&path)?;
        eprintln!("tokenizer | {}", tok_gguf.display());
        let tok_file =
            GgufFile::open(&tok_gguf).with_context(|| format!("open tokenizer {}", tok_gguf.display()))?;
        let tok = Tokenizer::from_gguf(&tok_file)?;
        if tok.tokens.len() != model.cfg.n_vocab {
            eprintln!(
                "warn | tokenizer vocab {} != model n_vocab {} (pack may still run if ids overlap)",
                tok.tokens.len(),
                model.cfg.n_vocab
            );
        }

        let weight_gib = dir_bytes(&path) as f64 / (1024.0 * 1024.0 * 1024.0);
        let backend_kind = BackendKind::from_config(&model.cfg);
        debug_assert!(matches!(
            backend_kind,
            BackendKind::TaraMoe | BackendKind::Tara141
        ));
        let max_seq = backend_kind.max_seq(cfg.max_seq, model.cfg.n_ctx, weight_gib);
        eprintln!("backend | {}", backend_kind.name());
        let kv = model.alloc_kv(max_seq)?;
        let model_id = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("tara-moe")
            .to_string();
        eprintln!(
            "flags | cuda_graph={} | moe_device_topk={} | decode={} | weight_gib={weight_gib:.3}",
            cfg.cuda_graph,
            model.cfg.router_top_k,
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
            backend_kind,
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
    /// Greedy (temperature 0) unless the caller uses
    /// [`Self::chat_completion_stream_sampled`].
    pub fn chat_completion_stream<F>(
        &mut self,
        messages: &[ChatMessage],
        max_tokens: Option<usize>,
        on_token: F,
    ) -> Result<TurnStats>
    where
        F: FnMut(&str),
    {
        let defaults = SessionOptions::default();
        self.chat_completion_stream_with(
            messages,
            max_tokens,
            defaults.temperature,
            defaults.top_p,
            defaults.top_k,
            defaults.repetition_penalty,
            defaults.seed,
            on_token,
        )
    }

    /// Stateless sampled completion for CLI/server clients that request quality
    /// decoding instead of the greedy benchmark path.
    pub fn chat_completion_stream_sampled<F>(
        &mut self,
        messages: &[ChatMessage],
        max_tokens: Option<usize>,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        repetition_penalty: f32,
        seed: u64,
        on_token: F,
    ) -> Result<TurnStats>
    where
        F: FnMut(&str),
    {
        self.chat_completion_stream_with(
            messages,
            max_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            seed,
            on_token,
        )
    }

    fn chat_completion_stream_with<F>(
        &mut self,
        messages: &[ChatMessage],
        max_tokens: Option<usize>,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        repetition_penalty: f32,
        seed: u64,
        on_token: F,
    ) -> Result<TurnStats>
    where
        F: FnMut(&str),
    {
        if messages.is_empty() {
            anyhow::bail!("messages must not be empty");
        }
        let max_new = max_tokens.unwrap_or(self.max_new).max(1);
        // Server defaults to non-thinking (Qwen3.5 small default).
        let enable_thinking = false;
        let prompt = format_chatml(messages, Some(&self.default_system), enable_thinking);
        let opts = SessionOptions {
            max_new,
            system: self.default_system.clone(),
            print_stream: false,
            print_stats: false,
            enable_thinking,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            seed,
        };
        self.kv.clear();
        let mut session = Session::with_kv(&mut self.model, &self.tok, &mut self.kv, opts);
        session.complete_prompt_stream(&prompt, on_token)
    }
}
