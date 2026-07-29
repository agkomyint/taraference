//! taraference core: GGUF load, CUDA forward, chat session, tokenizer.
//!
//! Layers:
//! - **inference** — [`engine::InferenceEngine`], [`session::Session`], CUDA path
//! - **chat** — message types + ChatML formatting (shared by CLI + server)

pub mod backend;
pub mod chat;
pub mod config;
pub mod cuda;
pub mod engine;
pub mod quant;
pub mod sampler;
pub mod session;
pub mod tokenizer;

pub use backend::BackendKind;
pub use chat::{
    assistant_generation_prompt, format_chatml, format_chatml_for_model, ChatMessage, ChatRole,
    CHATML_ASSISTANT_PROMPT,
};
pub use config::ModelConfig;
pub use cuda::{CudaKv, CudaModel, DecodeBackend};
pub use engine::{EngineConfig, InferenceEngine};
pub use sampler::{sample_logits, SamplingOptions};
pub use session::{Session, SessionOptions, StopReason, TurnStats};
pub use tokenizer::Tokenizer;
