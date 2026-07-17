//! taraference core: GGUF load, CUDA forward, chat session, tokenizer.
//!
//! Layers:
//! - **inference** — [`engine::InferenceEngine`], [`session::Session`], CUDA path
//! - **chat** — message types + ChatML formatting (shared by CLI + server)

pub mod chat;
pub mod config;
pub mod cuda;
pub mod engine;
pub mod quant;
pub mod session;
pub mod tokenizer;

pub use chat::{assistant_generation_prompt, format_chatml, ChatMessage, ChatRole};
pub use config::ModelConfig;
pub use cuda::{CudaKv, CudaModel, DecodeBackend};
pub use engine::{EngineConfig, InferenceEngine};
pub use session::{Session, SessionOptions, StopReason, TurnStats};
pub use tokenizer::Tokenizer;
