//! Chat message types and Qwen2 chat-template formatting.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl ChatRole {
    pub fn as_str(self) -> &'static str {
        match self {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
        }
    }
}

/// Qwen3 / Qwen3.5 generation-prompt suffix for thinking vs non-thinking.
///
/// Matches the official chat template:
/// - thinking on:  `<|im_start|>assistant\n<think>\n`  (model fills reasoning)
/// - thinking off: `<|im_start|>assistant\n<think>\n\n</think>\n\n` (skip think)
pub fn assistant_generation_prompt(enable_thinking: bool) -> &'static str {
    if enable_thinking {
        "<|im_start|>assistant\n<think>\n"
    } else {
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }
}

/// Build a Qwen2 / ChatML prompt from OpenAI-style messages.
///
/// Ends with the Qwen3.5 assistant generation prompt (see
/// [`assistant_generation_prompt`]). If there is no system message, a default
/// is injected when `default_system` is set.
///
/// `enable_thinking` selects thinking vs non-thinking mode (Qwen3.5 small
/// models default to non-thinking).
pub fn format_chatml(
    messages: &[ChatMessage],
    default_system: Option<&str>,
    enable_thinking: bool,
) -> String {
    let has_system = messages.iter().any(|m| m.role == ChatRole::System);
    let mut out = String::new();

    if !has_system {
        if let Some(sys) = default_system {
            out.push_str("<|im_start|>system\n");
            out.push_str(sys);
            out.push_str("<|im_end|>\n");
        }
    }

    for m in messages {
        out.push_str("<|im_start|>");
        out.push_str(m.role.as_str());
        out.push('\n');
        out.push_str(&m.content);
        out.push_str("<|im_end|>\n");
    }

    // Continue as assistant (OpenAI chat completions generation target).
    if messages
        .last()
        .map(|m| m.role != ChatRole::Assistant)
        .unwrap_or(true)
    {
        out.push_str(assistant_generation_prompt(enable_thinking));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatml_injects_default_system() {
        let msgs = [ChatMessage::user("hi")];
        let s = format_chatml(&msgs, Some("You are helpful."), false);
        assert!(s.starts_with("<|im_start|>system\nYou are helpful."));
        assert!(s.contains("<|im_start|>user\nhi"));
        assert!(s.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn chatml_thinking_opens_think_block() {
        let msgs = [ChatMessage::user("hi")];
        let s = format_chatml(&msgs, None, true);
        assert!(s.ends_with("<|im_start|>assistant\n<think>\n"));
        assert!(!s.contains("</think>"));
    }
}
