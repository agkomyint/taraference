//! OpenAI-compatible HTTP server (`--serve`).

mod openai;

use anyhow::{Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use openai::{
    ChatChoice, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatMessageIn,
    ChunkChoice, Delta, ErrorBody, ErrorResponse, ModelObject, ModelsResponse, Usage,
};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use taraference_core::{ChatMessage, ChatRole, InferenceEngine, StopReason};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::{error, info, warn};
use uuid::Uuid;

/// Shared GPU engine (serialized — single CUDA context).
#[derive(Clone)]
struct AppState {
    engine: Arc<Mutex<InferenceEngine>>,
}

pub async fn run(engine: InferenceEngine, port: u16) -> Result<()> {
    let model_id = engine.model_id.clone();
    let max_seq = engine.max_seq;
    let max_new = engine.max_new;
    let decode = engine.decode().name();
    let weight_gib = engine.weight_gib;

    let state = AppState {
        engine: Arc::new(Mutex::new(engine)),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!(
        %addr,
        model = %model_id,
        decode,
        max_seq,
        max_new,
        weight_gib = format!("{weight_gib:.2}"),
        "OpenAI-compatible server ready"
    );
    info!("  GET  /health");
    info!("  GET  /v1/models");
    info!("  POST /v1/chat/completions  (stream=true|false)");
    info!(
        "  curl example:  curl http://127.0.0.1:{port}/v1/chat/completions -H \"Content-Type: application/json\" -d '{{\"model\":\"{model_id}\",\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}],\"stream\":true}}'"
    );

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("bind {addr}"))?;
    info!(%addr, "listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("server")?;
    info!("server stopped");
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    info!("Ctrl-C received, shutting down…");
}

async fn health() -> impl IntoResponse {
    tracing::debug!("GET /health");
    Json(serde_json::json!({ "status": "ok" }))
}

async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    let (id, created) = {
        let eng = state.engine.lock().expect("engine lock");
        (eng.model_id.clone(), now_secs())
    };
    info!(model = %id, "GET /v1/models");
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelObject {
            id,
            object: "model",
            created,
            owned_by: "taraference".into(),
        }],
    })
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let t_req = Instant::now();
    let n_msgs = req.messages.len();
    let last_user = last_user_preview(&req.messages);
    let max_tokens = req.max_tokens.or(req.max_completion_tokens);
    let stream = req.stream.unwrap_or(false);
    // One process = one GGUF; request `model` is ignored (always the loaded weights).
    let served_id = {
        let eng = state.engine.lock().expect("engine lock");
        eng.model_id.clone()
    };

    info!(
        stream,
        n_msgs,
        max_tokens = ?max_tokens,
        model = %served_id,
        last_user = %last_user,
        "POST /v1/chat/completions"
    );

    if req.messages.is_empty() {
        warn!("reject: empty messages");
        return api_err(StatusCode::BAD_REQUEST, "messages must not be empty");
    }

    let messages = match map_messages(&req.messages) {
        Ok(m) => m,
        Err(e) => {
            warn!(error = %e, "reject: bad messages");
            return api_err(StatusCode::BAD_REQUEST, &e);
        }
    };

    if stream {
        return stream_chat(state, messages, max_tokens, served_id, t_req).into_response();
    }

    // Non-streaming: GPU work off the async worker.
    let result = tokio::task::spawn_blocking(move || {
        let wait = Instant::now();
        let mut eng = state.engine.lock().expect("engine lock");
        let lock_ms = wait.elapsed().as_secs_f64() * 1000.0;
        if lock_ms > 5.0 {
            info!(lock_ms = format!("{lock_ms:.0}"), "waited for GPU lock");
        }
        let model_id = eng.model_id.clone();
        let gen = Instant::now();
        let stats = eng.chat_completion(&messages, max_tokens)?;
        let gen_ms = gen.elapsed().as_secs_f64() * 1000.0;
        Ok::<_, anyhow::Error>((model_id, stats, gen_ms, lock_ms))
    })
    .await;

    let (model_id, stats, gen_ms, lock_ms) = match result {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            error!(error = %format!("{e:#}"), "completion failed");
            return api_err(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e:#}"));
        }
        Err(e) => {
            error!(error = %e, "join error");
            return api_err(StatusCode::INTERNAL_SERVER_ERROR, &format!("join: {e}"));
        }
    };

    let finish = match stats.stop {
        StopReason::Eos | StopReason::Empty => "stop",
        StopReason::MaxNew => "length",
    };
    let wall_ms = t_req.elapsed().as_secs_f64() * 1000.0;
    let reply_preview = truncate(&stats.reply, 80);

    info!(
        model = %model_id,
        stream = false,
        finish,
        prompt_tok = stats.prompt_tokens,
        gen_tok = stats.gen_tokens,
        prefill_ms = format!("{:.0}", stats.prefill_ms),
        decode_tps = format!("{:.1}", stats.decode_tps),
        gen_ms = format!("{gen_ms:.0}"),
        lock_ms = format!("{lock_ms:.0}"),
        wall_ms = format!("{wall_ms:.0}"),
        reply = %reply_preview,
        "completion ok"
    );

    let resp = ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion",
        created: now_secs(),
        model: model_id,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessageIn {
                role: "assistant".into(),
                content: stats.reply,
            },
            finish_reason: Some(finish.into()),
        }],
        usage: Usage {
            prompt_tokens: stats.prompt_tokens as u32,
            completion_tokens: stats.gen_tokens as u32,
            total_tokens: (stats.prompt_tokens + stats.gen_tokens) as u32,
        },
    };

    (StatusCode::OK, Json(resp)).into_response()
}

/// OpenAI SSE stream: `data: {json}\n\n` … `data: [DONE]\n\n`
fn stream_chat(
    state: AppState,
    messages: Vec<ChatMessage>,
    max_tokens: Option<usize>,
    model_id: String,
    t_req: Instant,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<String>(64);
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = now_secs();
    let id_log = id.clone();

    tokio::task::spawn_blocking(move || {
        let send = |payload: String| {
            // Drop send errors if client disconnected.
            let _ = tx.blocking_send(payload);
        };

        let wait = Instant::now();
        let mut eng = match state.engine.lock() {
            Ok(g) => g,
            Err(e) => {
                error!(error = %e, "engine lock poisoned");
                send(format!(
                    "{{\"error\":{{\"message\":\"lock: {e}\",\"type\":\"server_error\"}}}}"
                ));
                return;
            }
        };
        let lock_ms = wait.elapsed().as_secs_f64() * 1000.0;
        if lock_ms > 5.0 {
            info!(
                id = %id_log,
                lock_ms = format!("{lock_ms:.0}"),
                "waited for GPU lock"
            );
        }

        info!(
            id = %id_log,
            model = %model_id,
            stream = true,
            "generation start"
        );

        // First chunk: role
        let first = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model_id.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: Some("assistant".into()),
                    content: Some(String::new()),
                },
                finish_reason: None,
            }],
        };
        if let Ok(s) = serde_json::to_string(&first) {
            send(s);
        }

        let gen = Instant::now();
        let mut streamed_chars = 0usize;
        let result = eng.chat_completion_stream(&messages, max_tokens, |piece| {
            streamed_chars += piece.len();
            let chunk = ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_id.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: None,
                        content: Some(piece.to_string()),
                    },
                    finish_reason: None,
                }],
            };
            if let Ok(s) = serde_json::to_string(&chunk) {
                send(s);
            }
        });

        match result {
            Ok(stats) => {
                let finish = match stats.stop {
                    StopReason::Eos | StopReason::Empty => "stop",
                    StopReason::MaxNew => "length",
                };
                let gen_ms = gen.elapsed().as_secs_f64() * 1000.0;
                let wall_ms = t_req.elapsed().as_secs_f64() * 1000.0;
                let reply_preview = truncate(&stats.reply, 80);
                info!(
                    id = %id_log,
                    model = %model_id,
                    stream = true,
                    finish,
                    prompt_tok = stats.prompt_tokens,
                    gen_tok = stats.gen_tokens,
                    prefill_ms = format!("{:.0}", stats.prefill_ms),
                    decode_tps = format!("{:.1}", stats.decode_tps),
                    gen_ms = format!("{gen_ms:.0}"),
                    lock_ms = format!("{lock_ms:.0}"),
                    wall_ms = format!("{wall_ms:.0}"),
                    streamed_chars,
                    reply = %reply_preview,
                    "completion ok"
                );
                let end = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model_id,
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta {
                            role: None,
                            content: None,
                        },
                        finish_reason: Some(finish.into()),
                    }],
                };
                if let Ok(s) = serde_json::to_string(&end) {
                    send(s);
                }
                send("[DONE]".into());
            }
            Err(e) => {
                error!(id = %id_log, error = %format!("{e:#}"), "stream completion failed");
                send(format!(
                    "{{\"error\":{{\"message\":\"{e:#}\",\"type\":\"server_error\"}}}}"
                ));
            }
        }
    });

    let stream = ReceiverStream::new(rx).map(|data| Ok::<_, Infallible>(Event::default().data(data)));
    Sse::new(stream).keep_alive(KeepAlive::default())
}

fn map_messages(raw: &[ChatMessageIn]) -> Result<Vec<ChatMessage>, String> {
    let mut out = Vec::with_capacity(raw.len());
    for m in raw {
        let role = match m.role.to_ascii_lowercase().as_str() {
            "system" => ChatRole::System,
            "user" => ChatRole::User,
            "assistant" => ChatRole::Assistant,
            other => {
                return Err(format!(
                    "unsupported role {other:?} (use system|user|assistant)"
                ))
            }
        };
        out.push(ChatMessage {
            role,
            content: m.content.clone(),
        });
    }
    Ok(out)
}

fn api_err(status: StatusCode, message: &str) -> Response {
    warn!(%status, %message, "API error response");
    let body = ErrorResponse {
        error: ErrorBody {
            message: message.into(),
            type_: "invalid_request_error".into(),
            code: None,
        },
    };
    (status, Json(body)).into_response()
}

fn last_user_preview(messages: &[ChatMessageIn]) -> String {
    messages
        .iter()
        .rev()
        .find(|m| m.role.eq_ignore_ascii_case("user"))
        .map(|m| truncate(&m.content, 60))
        .unwrap_or_else(|| "(no user message)".into())
}

fn truncate(s: &str, max_chars: usize) -> String {
    let mut it = s.chars();
    let head: String = it.by_ref().take(max_chars).collect();
    if it.next().is_some() {
        format!("{head}…")
    } else {
        head
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
