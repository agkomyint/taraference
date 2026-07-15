//! Decode backend registry — **the only place** that lists A/B attention paths.
//!
//! # Add a new backend (e.g. `fastv3`)
//! 1. Add `kernels/attn/fast_v3.cu` with `extern "C" __global__ void attn_fast_v3(...)`
//! 2. `include_str!` it in `kernels/mod.rs`
//! 3. Append one [`DecodeSpec`] to [`REGISTRY`] below
//! 4. Rebuild + `--profile --decode fastv3`
//!
//! # Delete a backend that did not improve
//! 1. Remove its row from [`REGISTRY`]
//! 2. Remove its `include_str!` from `kernels/mod.rs`
//! 3. Delete `kernels/attn/<file>.cu`
//!
//! Host launch is **data-driven** from [`AttnLaunch`] — no new match arms in `layer.rs`
//! for normal causal backends; flash has a dedicated arm for partial+reduce.

use anyhow::{bail, Result};
use std::fmt;
use std::str::FromStr;

// ── launch recipes (shared host code) ───────────────────────────────────────

/// How shared memory size is computed for a standard causal attn kernel.
#[derive(Clone, Copy, Debug)]
pub enum SmemRule {
    /// `(head_dim + seq_len) * 4` — grows with context (v1-style).
    HeadPlusSeq,
    /// `seq_len * 4` — basic baseline.
    SeqOnly,
    /// `(head_dim + tile) * 4` — fixed tile (v2-style; lean smem for occupancy).
    HeadPlusTile { tile: u32 },
    /// `head_dim * 2 * 4` — online decode (Q + reduce).
    HeadTimes2,
}

impl SmemRule {
    pub fn bytes(self, head_dim: usize, seq_len: usize) -> u32 {
        match self {
            SmemRule::HeadPlusSeq => ((head_dim + seq_len.max(1)) * 4) as u32,
            SmemRule::SeqOnly => (seq_len.max(1) * 4) as u32,
            SmemRule::HeadPlusTile { tile } => ((head_dim + tile as usize) * 4) as u32,
            SmemRule::HeadTimes2 => (head_dim * 2 * 4) as u32,
        }
    }
}

/// Max sequence splits for flash-decoding (must match flash.cu FLASH_MAX_SPLIT).
pub const FLASH_MAX_SPLIT: u32 = 8;
/// Fixed splits for decode (CUDA-graph stable launch shape).
pub const FLASH_DECODE_SPLIT: u32 = 4;

/// Host launch shape for an attention kernel.
#[derive(Clone, Copy, Debug)]
pub enum AttnLaunch {
    /// Causal multi-query: grid `(n_head, n_q)`, args `pos0, n_q`.
    /// CUDA signature: `attn_*(q,k,v,out, n_head,n_kv,hd, pos0, n_q, scale)`.
    Causal {
        kernel: &'static str,
        /// Device-pos0 variant for CUDA graphs (`*_d`).
        kernel_d: Option<&'static str>,
        smem: SmemRule,
        block_threads: u32,
    },
    /// Flash-decoding: partial over splits + reduce. Prefill uses `prefill_as`.
    Flash {
        partial: &'static str,
        partial_d: &'static str,
        reduce: &'static str,
        smem: SmemRule,
        block_threads: u32,
        prefill_as: &'static str,
        n_split: u32,
    },
    /// Single-token online decode: grid `(n_head)`, block `head_dim`, arg `seq_len`.
    /// Prefill (`n_tok > 1`) uses another registry name.
    OnlineDecode {
        kernel: &'static str,
        /// Registry `name` used for multi-token prefill.
        prefill_as: &'static str,
        max_head_dim: usize,
    },
}

/// One pluggable `--decode` option.
#[derive(Clone, Copy, Debug)]
pub struct DecodeSpec {
    /// CLI primary name (`fastv2`, …).
    pub name: &'static str,
    /// Extra CLI aliases.
    pub aliases: &'static [&'static str],
    /// One-line description for help / profile.
    pub description: &'static str,
    /// How to launch (and which CUDA symbols).
    pub launch: AttnLaunch,
    /// Exactly one entry should be `true` — becomes [`Default`].
    pub is_default: bool,
}

// ═══════════════════════════════════════════════════════════════════════════
// REGISTRY — add / remove backends here only
// ═══════════════════════════════════════════════════════════════════════════

/// All decode backends. Order = help list order. Default = the `is_default` row.
pub static REGISTRY: &[DecodeSpec] = &[
    DecodeSpec {
        name: "fastv2",
        aliases: &["v2", "tiled"],
        description: "v2: f16 KV + tiled online attn + CUDA-graph path — recommended",
        launch: AttnLaunch::Causal {
            kernel: "attn_fast_v2",
            kernel_d: Some("attn_fast_v2_d"),
            smem: SmemRule::HeadPlusTile { tile: 64 },
            block_threads: 128,
        },
        is_default: true,
    },
    DecodeSpec {
        name: "flash",
        aliases: &["flashdec", "fd"],
        description: "flash-decoding: split KV + reduce (helps long-ctx drop)",
        launch: AttnLaunch::Flash {
            partial: "attn_flash_partial",
            partial_d: "attn_flash_partial_d",
            reduce: "attn_flash_reduce",
            smem: SmemRule::HeadPlusTile { tile: 64 },
            block_threads: 128,
            prefill_as: "fastv2",
            n_split: FLASH_DECODE_SPLIT,
        },
        is_default: false,
    },
    DecodeSpec {
        name: "fast",
        aliases: &["fastv1", "v1", "parallel"],
        description: "v1: f16 KV + parallel softmax scores[ctx] smem — A/B baseline",
        launch: AttnLaunch::Causal {
            kernel: "attn_fast_v1",
            kernel_d: None,
            smem: SmemRule::HeadPlusSeq,
            block_threads: 128,
        },
        is_default: false,
    },
    DecodeSpec {
        name: "basic",
        aliases: &["baseline", "serial"],
        description: "f16 KV + serial softmax scores[ctx] — slow baseline",
        launch: AttnLaunch::Causal {
            kernel: "attn_basic_f32",
            kernel_d: None,
            smem: SmemRule::SeqOnly,
            block_threads: 128,
        },
        is_default: false,
    },
    DecodeSpec {
        name: "online",
        aliases: &[],
        description: "f16 KV + online decode (1 tok); prefill uses fastv2",
        launch: AttnLaunch::OnlineDecode {
            kernel: "attn_online_f32",
            prefill_as: "fastv2",
            max_head_dim: 256,
        },
        is_default: false,
    },
];

// ── handle used everywhere (CLI, model, profile) ────────────────────────────

/// Selected decode backend (index into [`REGISTRY`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DecodeBackend {
    idx: u8,
}

impl DecodeBackend {
    /// All registered backends.
    pub fn all() -> impl Iterator<Item = DecodeBackend> {
        (0..REGISTRY.len()).map(|i| DecodeBackend { idx: i as u8 })
    }

    pub fn spec(self) -> &'static DecodeSpec {
        &REGISTRY[self.idx as usize]
    }

    pub fn name(self) -> &'static str {
        self.spec().name
    }

    pub fn description(self) -> &'static str {
        self.spec().description
    }

    pub fn list_help() -> String {
        REGISTRY
            .iter()
            .map(|s| {
                let mark = if s.is_default { " [default]" } else { "" };
                format!("{} — {}{mark}", s.name, s.description)
            })
            .collect::<Vec<_>>()
            .join("; ")
    }

    /// Unique CUDA kernel symbols needed by the full registry (for load).
    pub fn required_kernel_symbols() -> Vec<&'static str> {
        let mut out = Vec::new();
        for s in REGISTRY {
            match s.launch {
                AttnLaunch::Causal {
                    kernel, kernel_d, ..
                } => {
                    push_unique(&mut out, kernel);
                    if let Some(kd) = kernel_d {
                        push_unique(&mut out, kd);
                    }
                }
                AttnLaunch::Flash {
                    partial,
                    partial_d,
                    reduce,
                    ..
                } => {
                    push_unique(&mut out, partial);
                    push_unique(&mut out, partial_d);
                    push_unique(&mut out, reduce);
                }
                AttnLaunch::OnlineDecode { kernel, .. } => push_unique(&mut out, kernel),
            }
        }
        // Prefill fallbacks + graph variants always useful.
        push_unique(&mut out, "attn_fast_v2");
        push_unique(&mut out, "attn_fast_v2_d");
        out
    }

    /// Resolve by primary name or alias.
    pub fn parse_name(s: &str) -> Result<Self> {
        let key = s.trim().to_ascii_lowercase();
        if key == "default" {
            return Ok(Self::default());
        }
        for (i, spec) in REGISTRY.iter().enumerate() {
            if spec.name == key || spec.aliases.iter().any(|a| *a == key) {
                return Ok(DecodeBackend { idx: i as u8 });
            }
        }
        bail!(
            "unknown --decode {s:?}; choose one of: {}",
            REGISTRY
                .iter()
                .map(|sp| sp.name)
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

fn push_unique(v: &mut Vec<&'static str>, s: &'static str) {
    if !v.contains(&s) {
        v.push(s);
    }
}

impl Default for DecodeBackend {
    fn default() -> Self {
        for (i, s) in REGISTRY.iter().enumerate() {
            if s.is_default {
                return DecodeBackend { idx: i as u8 };
            }
        }
        DecodeBackend { idx: 0 }
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
        Self::parse_name(s)
    }
}

/// Look up a registry entry by primary name (for prefill fallback).
pub fn find_by_name(name: &str) -> Option<&'static DecodeSpec> {
    REGISTRY.iter().find(|s| s.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_marked() {
        let d = DecodeBackend::default();
        assert!(d.spec().is_default);
        assert_eq!(d.name(), "fastv2");
    }

    #[test]
    fn parse_aliases() {
        assert_eq!(
            DecodeBackend::parse_name("fastv2").unwrap().name(),
            "fastv2"
        );
        assert_eq!(DecodeBackend::parse_name("v1").unwrap().name(), "fast");
        assert_eq!(
            DecodeBackend::parse_name("default").unwrap().name(),
            DecodeBackend::default().name()
        );
    }

    #[test]
    fn symbols_cover_registry() {
        let syms = DecodeBackend::required_kernel_symbols();
        assert!(syms.contains(&"attn_fast_v2"));
        assert!(syms.contains(&"attn_flash_partial"));
        assert!(syms.contains(&"attn_online_f32"));
    }
}
