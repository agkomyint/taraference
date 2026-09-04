//! OpenAI Chat Completions JSON shapes (subset we implement).

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    /// Accepted for OpenAI clients; server always uses the loaded GGUF.
    #[serde(default)]
    #[allow(dead_code)]
    pub model: String,
    pub input: EmbeddingInput,
    #[serde(default)]
    pub encoding_format: Option<String>,
    #[serde(default)]
    pub dimensions: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingObject {
    pub object: &'static str,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    /// Accepted for OpenAI clients; server always uses the loaded GGUF.
    #[serde(default)]
    #[allow(dead_code)]
    pub model: String,
    pub messages: Vec<ChatMessageIn>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// OpenAI newer alias for max_tokens.
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessageIn {
    pub role: String,
    #[serde(default)]
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessageIn,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Debug, Serialize)]
pub struct PromptTokensDetails {
    pub cached_tokens: u32,
}

/// Streaming chunk (SSE `data:` payload).
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

#[derive(Debug, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorBody,
}

#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub type_: String,
    pub code: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_and_multiple_embedding_inputs() {
        let one: EmbeddingRequest =
            serde_json::from_str(r#"{"model":"embed","input":"hello","dimensions":256}"#).unwrap();
        assert!(matches!(one.input, EmbeddingInput::One(ref value) if value == "hello"));
        assert_eq!(one.dimensions, Some(256));

        let many: EmbeddingRequest =
            serde_json::from_str(r#"{"input":["hello","สวัสดี"],"encoding_format":"float"}"#)
                .unwrap();
        assert!(matches!(many.input, EmbeddingInput::Many(ref values) if values.len() == 2));
    }

    #[test]
    fn reports_openai_compatible_cached_prompt_tokens() {
        let usage = Usage {
            prompt_tokens: 120,
            completion_tokens: 20,
            total_tokens: 140,
            prompt_tokens_details: Some(PromptTokensDetails { cached_tokens: 80 }),
        };
        let value = serde_json::to_value(usage).unwrap();
        assert_eq!(value["prompt_tokens"], 120);
        assert_eq!(value["prompt_tokens_details"]["cached_tokens"], 80);
    }
}
