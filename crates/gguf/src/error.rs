use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("I/O error for {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("invalid GGUF magic (got 0x{0:08X}, expected 0x46554747 'GGUF')")]
    BadMagic(u32),

    #[error("unsupported GGUF version {0} (supported: 2, 3)")]
    UnsupportedVersion(u32),

    #[error("unexpected end of file while reading {context}")]
    UnexpectedEof { context: &'static str },

    #[error("invalid UTF-8 in GGUF string ({context})")]
    InvalidUtf8 { context: &'static str },

    #[error("unknown metadata value type id {0}")]
    UnknownValueType(u32),

    #[error("unknown GGML tensor type id {0}")]
    UnknownTensorType(u32),

    #[error("metadata key not found: {0}")]
    MissingMetadata(String),

    #[error("tensor not found: {0}")]
    MissingTensor(String),

    #[error("malformed GGUF: {0}")]
    Malformed(String),
}

pub type Result<T> = std::result::Result<T, GgufError>;
