//! Architecture-family dispatch and backend-specific runtime policy.
//!
//! Keep model-family decisions out of the generic engine. CUDA implementations
//! can move behind these boundaries incrementally without changing the CLI/API.

use crate::ModelConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Tara141,
    TaraMoe,
    Qwen35,
    Qwen2,
    LlamaDense,
    OtherDense,
}

impl BackendKind {
    pub fn from_config(cfg: &ModelConfig) -> Self {
        if cfg.architecture == "tara_moe_141" {
            return Self::Tara141;
        }
        if cfg.is_moe() {
            return Self::TaraMoe;
        }
        match cfg.architecture.as_str() {
            "qwen35" | "qwen3_5" => Self::Qwen35,
            "qwen2" | "qwen2_5" => Self::Qwen2,
            "llama" | "llama2" | "llama3" => Self::LlamaDense,
            _ => Self::OtherDense,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Tara141 => "tara-1.4.1-moe",
            Self::TaraMoe => "tara-moe",
            Self::Qwen35 => "qwen3.5-hybrid",
            Self::Qwen2 => "qwen2-dense",
            Self::LlamaDense => "llama-dense",
            Self::OtherDense => "generic-dense",
        }
    }

    /// Apply only policy owned by this architecture family.
    pub fn max_seq(self, requested: usize, model_ctx: usize, weight_gib: f64) -> usize {
        let max_seq = requested.min(model_ctx);
        match self {
            // Qwen3.5 expands some quantized weights on load. On small GPUs a
            // large KV arena can remove clock/allocator headroom.
            Self::Qwen35
                if weight_gib > 2.0
                    && max_seq > 1024
                    && std::env::var_os("TARAFER_LONG_CTX").is_none() =>
            {
                1024
            }
            _ => max_seq,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{LayerKind, RouterWeightMode};

    fn cfg(architecture: &str, experts: usize) -> ModelConfig {
        ModelConfig {
            architecture: architecture.into(),
            n_vocab: 32,
            n_embd: 16,
            n_layer: 1,
            n_head: 1,
            n_head_kv: 1,
            attention_head_dim: 16,
            no_rope_layer_interval: 0,
            n_ff: 32,
            n_ctx: 4096,
            rope_theta: 10_000.0,
            rms_eps: 1e-6,
            rope_dim: 16,
            fused_q_gate: false,
            ssm_conv_kernel: 0,
            ssm_state_size: 0,
            ssm_n_k_heads: 0,
            ssm_n_v_heads: 0,
            ssm_inner_size: 0,
            layer_kinds: vec![LayerKind::FullAttention],
            n_experts: experts,
            router_top_k: usize::from(experts > 0),
            router_weight_mode: RouterWeightMode::SelectedSoftmax,
            expert_ff: 32,
            num_dense_layers: usize::from(experts == 0),
        }
    }

    #[test]
    fn dispatches_architecture_families() {
        assert_eq!(BackendKind::from_config(&cfg("qwen35", 0)), BackendKind::Qwen35);
        assert_eq!(BackendKind::from_config(&cfg("qwen2", 0)), BackendKind::Qwen2);
        assert_eq!(BackendKind::from_config(&cfg("llama", 2)), BackendKind::TaraMoe);
        assert_eq!(
            BackendKind::from_config(&cfg("tara_moe_141", 4)),
            BackendKind::Tara141
        );
    }

    #[test]
    fn tara_policy_does_not_inherit_qwen_context_cap() {
        assert_eq!(BackendKind::TaraMoe.max_seq(4096, 4096, 3.0), 4096);
    }
}
