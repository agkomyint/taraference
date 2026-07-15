//! GGUF (GPT-Generated Unified Format) loader.
//!
//! Spec reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
//!
//! This crate maps the file and exposes metadata + tensor descriptors.
//! Weights stay memory-mapped; the engine repacks them into GPU buffers later.

mod error;
mod reader;
mod types;
mod value;

pub use error::GgufError;
pub use reader::GgufFile;
pub use types::{GgmlType, GgufTensorInfo};
pub use value::Value;
