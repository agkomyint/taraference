//! Pluggable decode backends for A/B testing attention (and future matmul) paths.
//!
//! Add a new variant here + a CUDA kernel + a branch in [`launch_attn`].
//! CLI: `--decode fast|basic|online`

use anyhow::{bail, Result};
use std::fmt;
use std::str::FromStr;

/// Named decode optimization bundle (attention path today; extend as needed).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DecodeBackend {
    /// Parallel softmax attention (default — current best general path).
    #[default]
    Fast,
    /// Serial softmax baseline (original simple kernel).
    Basic,
    /// Online softmax for single-token decode; multi-token prefill uses Fast.
    Online,
}

impl DecodeBackend {
    pub const ALL: &[DecodeBackend] = &[
        DecodeBackend::Fast,
        DecodeBackend::Basic,
        DecodeBackend::Online,
    ];

    pub fn name(self) -> &'static str {
        match self {
            DecodeBackend::Fast => "fast",
            DecodeBackend::Basic => "basic",
            DecodeBackend::Online => "online",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            DecodeBackend::Fast => {
                "f16 KV + tiled online attn (fixed smem, no scores[ctx]) — default"
            }
            DecodeBackend::Basic => {
                "f16 KV + serial softmax scores[ctx] — baseline for A/B"
            }
            DecodeBackend::Online => {
                "f16 KV + online decode (1 tok); prefill uses tiled fast"
            }
        }
    }

    pub fn list_help() -> String {
        Self::ALL
            .iter()
            .map(|b| format!("{} — {}", b.name(), b.description()))
            .collect::<Vec<_>>()
            .join("; ")
    }
}

impl fmt::Display for DecodeBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for DecodeBackend {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "fast" | "default" | "parallel" => Ok(DecodeBackend::Fast),
            "basic" | "baseline" | "serial" => Ok(DecodeBackend::Basic),
            "online" | "flash" => Ok(DecodeBackend::Online),
            other => bail!(
                "unknown --decode {other:?}; choose one of: {}",
                Self::ALL
                    .iter()
                    .map(|b| b.name())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}
