//! Runtime profiler: sample CPU/GPU while multi-turn chat runs, print a rich report,
//! and save timestamped logs under `profile-logs/` for before/after comparison.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use taraference_core::{DecodeBackend, ModelConfig, StopReason, TurnStats};

/// Cap generated tokens per turn during profile (comparable runs).
pub const PROFILE_MAX_NEW: usize = 128;

/// Directory (cwd-relative) for saved profile reports.
pub const PROFILE_LOG_DIR: &str = "profile-logs";

/// Realistic multi-turn script (context grows like a real chat).
pub const MULTI_TURN_SCRIPT: &[&str] = &[
    "hi, who are you?",
    "what can you help me with in one sentence?",
    "ok give me 3 bullet ideas for a weekend project",
    "expand on the second idea a bit more",
    "summarize everything we talked about so far",
];

#[derive(Debug, Clone, Default)]
struct Sample {
    t_ms: f64,
    cpu_pct: f32,
    gpu_util: f32,
    gpu_mem_util: f32,
    gpu_mem_used_mb: f32,
    gpu_mem_total_mb: f32,
    gpu_power_w: f32,
    gpu_temp_c: f32,
    gpu_clock_mhz: f32,
    gpu_mem_clock_mhz: f32,
}

#[derive(Debug, Default)]
struct SampleBuf {
    items: Vec<Sample>,
}

#[derive(Debug, Clone)]
pub struct TurnRow {
    pub index: usize,
    pub user: String,
    pub stats: TurnStats,
}

/// Static context for the report header.
pub struct ProfileMeta {
    pub model_path: String,
    pub cfg: ModelConfig,
    pub weight_gib: f64,
    pub max_seq: usize,
    pub max_new: usize,
    pub sample_interval_ms: u64,
    pub decode: DecodeBackend,
}

pub struct Profiler {
    stop: Arc<AtomicBool>,
    buf: Arc<Mutex<SampleBuf>>,
    handle: Option<thread::JoinHandle<()>>,
    t0: Instant,
}

impl Profiler {
    pub fn start(interval_ms: u64) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let buf = Arc::new(Mutex::new(SampleBuf::default()));
        let stop_c = Arc::clone(&stop);
        let buf_c = Arc::clone(&buf);
        let t0 = Instant::now();
        let handle = thread::spawn(move || {
            let interval = Duration::from_millis(interval_ms.max(50));
            while !stop_c.load(Ordering::Relaxed) {
                let mut s = Sample {
                    t_ms: t0.elapsed().as_secs_f64() * 1000.0,
                    ..Default::default()
                };
                if let Some(g) = sample_gpu() {
                    s.gpu_util = g.0;
                    s.gpu_mem_util = g.1;
                    s.gpu_mem_used_mb = g.2;
                    s.gpu_mem_total_mb = g.3;
                    s.gpu_power_w = g.4;
                    s.gpu_temp_c = g.5;
                    s.gpu_clock_mhz = g.6;
                    s.gpu_mem_clock_mhz = g.7;
                }
                s.cpu_pct = sample_cpu().unwrap_or(0.0);
                if let Ok(mut b) = buf_c.lock() {
                    b.items.push(s);
                }
                thread::sleep(interval);
            }
        });
        Self {
            stop,
            buf,
            handle: Some(handle),
            t0,
        }
    }

    /// Build report, print it, save under `profile-logs/`, return log path.
    pub fn stop_and_report(
        &mut self,
        turns: &[TurnRow],
        mode: &str,
        meta: &ProfileMeta,
    ) -> PathBuf {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        let elapsed = self.t0.elapsed().as_secs_f64();
        let samples = self
            .buf
            .lock()
            .map(|b| b.items.clone())
            .unwrap_or_default();

        let n_s = samples.len().max(1) as f32;
        let avg = |f: fn(&Sample) -> f32| samples.iter().map(f).sum::<f32>() / n_s;
        let maxf = |f: fn(&Sample) -> f32| {
            samples.iter().map(f).fold(f32::NEG_INFINITY, |a, b| a.max(b))
        };
        let minf = |f: fn(&Sample) -> f32| {
            samples.iter().map(f).fold(f32::INFINITY, |a, b| a.min(b))
        };

        let cpu_avg = avg(|s| s.cpu_pct);
        let cpu_max = maxf(|s| s.cpu_pct);
        let cpu_min = minf(|s| s.cpu_pct);
        let gpu_avg = avg(|s| s.gpu_util);
        let gpu_max = maxf(|s| s.gpu_util);
        let gpu_min = minf(|s| s.gpu_util);
        let gmem_util_avg = avg(|s| s.gpu_mem_util);
        let gmem_avg = avg(|s| s.gpu_mem_used_mb);
        let gmem_max = maxf(|s| s.gpu_mem_used_mb);
        let gmem_min = minf(|s| s.gpu_mem_used_mb);
        let gmem_tot = samples
            .iter()
            .map(|s| s.gpu_mem_total_mb)
            .fold(0.0_f32, f32::max);
        let pwr_avg = avg(|s| s.gpu_power_w);
        let pwr_max = maxf(|s| s.gpu_power_w);
        let pwr_min = minf(|s| s.gpu_power_w);
        let tmp_avg = avg(|s| s.gpu_temp_c);
        let tmp_max = maxf(|s| s.gpu_temp_c);
        let tmp_min = minf(|s| s.gpu_temp_c);
        let clk_avg = avg(|s| s.gpu_clock_mhz);
        let clk_max = maxf(|s| s.gpu_clock_mhz);
        let mclk_avg = avg(|s| s.gpu_mem_clock_mhz);

        let total_prompt: usize = turns.iter().map(|t| t.stats.prompt_tokens).sum();
        let total_gen: usize = turns.iter().map(|t| t.stats.gen_tokens).sum();
        let total_prefill_ms: f64 = turns.iter().map(|t| t.stats.prefill_ms).sum();
        let total_decode_ms: f64 = turns.iter().map(|t| t.stats.decode_ms).sum();
        let total_tokenize_ms: f64 = turns.iter().map(|t| t.stats.tokenize_ms).sum();
        let total_wall_ms: f64 = turns.iter().map(|t| t.stats.wall_ms).sum();
        let overall_decode_tps = if total_decode_ms > 0.0 {
            total_gen as f64 / (total_decode_ms / 1000.0)
        } else {
            0.0
        };
        let overall_prefill_tps = if total_prefill_ms > 0.0 {
            total_prompt as f64 / (total_prefill_ms / 1000.0)
        } else {
            0.0
        };
        let util_eff = if gpu_avg > 1.0 {
            overall_decode_tps / gpu_avg as f64
        } else {
            0.0
        };
        let final_ctx = turns.last().map(|t| t.stats.ctx_len).unwrap_or(0);
        let first_prefill = turns.first().map(|t| t.stats.prefill_ms).unwrap_or(0.0);
        let later_prefill_avg = if turns.len() > 1 {
            turns[1..].iter().map(|t| t.stats.prefill_ms).sum::<f64>()
                / (turns.len() - 1) as f64
        } else {
            0.0
        };
        let avg_ttft = turns.iter().map(|t| t.stats.ttft_ms).sum::<f64>() / turns.len().max(1) as f64;
        let max_ttft = turns
            .iter()
            .map(|t| t.stats.ttft_ms)
            .fold(0.0_f64, f64::max);
        let min_ttft = turns
            .iter()
            .map(|t| t.stats.ttft_ms)
            .fold(f64::INFINITY, f64::min);
        let hit_cap = turns.iter().filter(|t| t.stats.hit_max_new).count();
        let eos_stops = turns
            .iter()
            .filter(|t| t.stats.stop == StopReason::Eos)
            .count();
        let empty_stops = turns
            .iter()
            .filter(|t| t.stats.stop == StopReason::Empty)
            .count();

        let prefill_share = if total_wall_ms > 0.0 {
            100.0 * total_prefill_ms / total_wall_ms
        } else {
            0.0
        };
        let decode_share = if total_wall_ms > 0.0 {
            100.0 * total_decode_ms / total_wall_ms
        } else {
            0.0
        };

        let weight_bytes = meta.weight_gib * (1024.0 * 1024.0 * 1024.0);
        let est_gbps = overall_decode_tps * weight_bytes / 1e9;
        let ref_bw = 192.0;
        let bw_pct = 100.0 * est_gbps / ref_bw;

        let ctx_pct = if meta.max_seq > 0 {
            100.0 * final_ctx as f64 / meta.max_seq as f64
        } else {
            0.0
        };
        let vram_free = (gmem_tot - gmem_max).max(0.0);
        let vram_pct = if gmem_tot > 0.0 {
            100.0 * gmem_max / gmem_tot
        } else {
            0.0
        };

        let (decode_tps_first, decode_tps_last) = (
            turns.first().map(|t| t.stats.decode_tps).unwrap_or(0.0),
            turns.last().map(|t| t.stats.decode_tps).unwrap_or(0.0),
        );
        let decode_drop_pct = if decode_tps_first > 0.0 {
            100.0 * (1.0 - decode_tps_last / decode_tps_first)
        } else {
            0.0
        };

        let ms_per_tok_decode = if total_gen > 0 {
            total_decode_ms / total_gen as f64
        } else {
            0.0
        };
        let ms_per_tok_prefill = if total_prompt > 0 {
            total_prefill_ms / total_prompt as f64
        } else {
            0.0
        };

        let ctx_growth: Vec<usize> = turns
            .windows(2)
            .map(|w| w[1].stats.ctx_len.saturating_sub(w[0].stats.ctx_len))
            .collect();
        let avg_ctx_growth = if !ctx_growth.is_empty() {
            ctx_growth.iter().sum::<usize>() as f64 / ctx_growth.len() as f64
        } else {
            0.0
        };

        let host = sample_host_mem();
        let sample_span = samples
            .last()
            .map(|s| s.t_ms)
            .unwrap_or(0.0)
            .max(0.0);

        let stamp = local_timestamp();
        let mut report = String::with_capacity(8 * 1024);
        let mut line = |s: String| {
            report.push_str(&s);
            report.push('\n');
        };

        line(String::new());
        line("════════════════════════════════════════════════════════════════".into());
        line(format!("  PROFILE REPORT  ({mode})"));
        line(format!("  saved_at  {stamp}"));
        line("════════════════════════════════════════════════════════════════".into());
        line("  CONFIG".into());
        line(format!("    model     {}", meta.model_path));
        line(format!(
            "    arch      {}  L={}  d={}  heads={}/{}  ff={}  vocab={}",
            meta.cfg.architecture,
            meta.cfg.n_layer,
            meta.cfg.n_embd,
            meta.cfg.n_head,
            meta.cfg.n_head_kv,
            meta.cfg.n_ff,
            meta.cfg.n_vocab
        ));
        line(format!(
            "    weights   {:.2} GiB Q  |  model n_ctx={}  session max_seq={}  max_new={}",
            meta.weight_gib, meta.cfg.n_ctx, meta.max_seq, meta.max_new
        ));
        line(format!(
            "    decode    {} — {}",
            meta.decode.name(),
            meta.decode.description()
        ));
        line("    kv        f16 (half bandwidth vs f32)".into());
        line(format!(
            "    sample    every {} ms  |  {} samples over {:.2}s wall (span {:.0} ms)",
            meta.sample_interval_ms,
            samples.len(),
            elapsed,
            sample_span
        ));
        line("────────────────────────────────────────────────────────────────".into());
        line("  PER-TURN".into());
        line(format!(
            "  {:>4} {:>7} {:>7} {:>6} {:>5} {:>5} {:>5} {:>5} {:>6} {:>4} {}",
            "turn", "prefill", "decode", "ttft", "pTok", "gTok", "ctx0", "ctx", "stop", "cap", "user"
        ));
        for t in turns {
            let user_short = if t.user.chars().count() > 36 {
                let s: String = t.user.chars().take(36).collect();
                format!("{s}…")
            } else {
                t.user.clone()
            };
            let stop = match t.stats.stop {
                StopReason::Eos => "eos",
                StopReason::MaxNew => "max",
                StopReason::Empty => "empty",
            };
            let cap = if t.stats.hit_max_new { "Y" } else { "-" };
            line(format!(
                "  {:>4} {:>5.0}ms {:>5.1}t/s {:>5.0} {:>5} {:>5} {:>5} {:>5} {:>6} {:>4} {}",
                t.index + 1,
                t.stats.prefill_ms,
                t.stats.decode_tps,
                t.stats.ttft_ms,
                t.stats.prompt_tokens,
                t.stats.gen_tokens,
                t.stats.ctx_before,
                t.stats.ctx_len,
                stop,
                cap,
                user_short
            ));
        }
        line("────────────────────────────────────────────────────────────────".into());
        line("  LATENCY".into());
        line(format!(
            "    TTFT      avg {avg_ttft:.0} ms  min {min_ttft:.0}  max {max_ttft:.0}  (user→first token)"
        ));
        line(format!(
            "    tokenize  total {total_tokenize_ms:.1} ms  ({:.2} ms/turn)",
            total_tokenize_ms / turns.len().max(1) as f64
        ));
        line(format!(
            "    ms/token  prefill {ms_per_tok_prefill:.1}  decode {ms_per_tok_decode:.1}"
        ));
        line("────────────────────────────────────────────────────────────────".into());
        line("  THROUGHPUT".into());
        line(format!(
            "    prefill  {total_prefill_ms:>7.0} ms | {total_prompt:>5} tok | {overall_prefill_tps:>6.1} tok/s | {prefill_share:>5.1}% of turn-time"
        ));
        line(format!(
            "    decode   {total_decode_ms:>7.0} ms | {total_gen:>5} tok | {overall_decode_tps:>6.1} tok/s | {decode_share:>5.1}% of turn-time"
        ));
        line(format!(
            "    wall     {total_wall_ms:>7.0} ms sum turns | wall-clock run {:.2}s",
            elapsed
        ));
        line(format!(
            "    decode   first-turn {decode_tps_first:.1} t/s → last-turn {decode_tps_last:.1} t/s  (drop {decode_drop_pct:.0}%)"
        ));
        if turns.len() > 1 {
            line(format!(
                "    prefill  first {first_prefill:.0} ms | later avg {later_prefill_avg:.0} ms"
            ));
        }
        line(format!(
            "    est weight BW  {est_gbps:.1} GB/s  (~{bw_pct:.0}% of 192 GB/s ref)  [decode × model size]"
        ));
        line("────────────────────────────────────────────────────────────────".into());
        line("  CONTEXT / MEMORY".into());
        line(format!(
            "    ctx final {final_ctx} / {}  ({ctx_pct:.1}% full)  avg growth/turn {avg_ctx_growth:.0} tok",
            meta.max_seq
        ));
        if samples.iter().any(|s| s.gpu_mem_total_mb > 0.0) {
            line(format!(
                "    VRAM   used avg {gmem_avg:.0}  min {gmem_min:.0}  peak {gmem_max:.0} / {gmem_tot:.0} MiB  ({vram_pct:.0}% peak)"
            ));
            line(format!(
                "    VRAM   free at peak ~{vram_free:.0} MiB  |  mem-controller util avg {gmem_util_avg:.1}%"
            ));
        }
        if let Some((used, total)) = host {
            line(format!(
                "    host RAM  ~{used:.0} / {total:.0} MiB used (system snapshot)"
            ));
        }
        line("────────────────────────────────────────────────────────────────".into());
        line("  GENERATION QUALITY FLAGS".into());
        line(format!(
            "    stop: eos={eos_stops}  hit_max_new={hit_cap}  empty={empty_stops}  / {} turns",
            turns.len()
        ));
        if hit_cap > 0 {
            line(format!(
                "    ⚠ {hit_cap} turn(s) hit max_new={} — reply may be truncated mid-sentence",
                meta.max_new
            ));
        }
        line("────────────────────────────────────────────────────────────────".into());
        line("  CPU".into());
        line(format!(
            "    util   min {cpu_min:5.1}%  avg {cpu_avg:5.1}%  max {cpu_max:5.1}%"
        ));
        line("  GPU".into());
        if samples.iter().any(|s| s.gpu_mem_total_mb > 0.0) {
            line(format!(
                "    compute  min {gpu_min:5.1}%  avg {gpu_avg:5.1}%  max {gpu_max:5.1}%"
            ));
            line(format!(
                "    power    min {pwr_min:5.1} W  avg {pwr_avg:5.1} W  max {pwr_max:5.1} W"
            ));
            line(format!(
                "    temp     min {tmp_min:5.0} °C avg {tmp_avg:5.0} °C max {tmp_max:5.0} °C"
            ));
            line(format!(
                "    clocks   core avg {clk_avg:.0} MHz (max {clk_max:.0})  mem avg {mclk_avg:.0} MHz"
            ));
            line(format!(
                "    efficiency  {util_eff:.3} decode_tok/s per GPU%   (higher = more tokens for same util)"
            ));
            if pwr_max > 0.0 {
                line(format!(
                    "    tok/J (rough)  {:.2} decode tok per joule  [gen_tok / (avg_W × decode_s)]",
                    total_gen as f64 / (pwr_avg as f64 * (total_decode_ms / 1000.0)).max(1e-6)
                ));
            }
        } else {
            line("    (nvidia-smi unavailable — check PATH / driver)".into());
        }
        line("────────────────────────────────────────────────────────────────".into());
        line("  READOUT / HINTS".into());
        if vram_pct < 70.0 && gmem_tot > 0.0 {
            line(format!(
                "    • VRAM headroom ~{vram_free:.0} MiB — not memory-capacity bound; speed ≠ fill VRAM."
            ));
        }
        if gpu_avg > 85.0 && est_gbps < ref_bw * 0.25 {
            line(
                "    • GPU util high but est weight BW low vs peak → kernels busy but not streaming weights efficiently (or util metric saturated)."
                    .into(),
            );
        }
        if decode_drop_pct > 30.0 {
            line(format!(
                "    • Decode dropped {decode_drop_pct:.0}% as ctx grew → attention/KV path dominates long chats."
            ));
        }
        if turns.len() > 1 && later_prefill_avg > first_prefill * 0.85 {
            line(
                "    • Later-turn prefill still heavy → each new token pays attention over long KV (expected) or rework batching."
                    .into(),
            );
        }
        if cpu_avg > 45.0 {
            line("    • CPU elevated → host launch/sync/copy may steal decode tok/s.".into());
        }
        if tmp_max > 85.0 {
            line(format!(
                "    • GPU hot ({tmp_max:.0}°C) — possible thermal throttle; watch clocks."
            ));
        }
        if hit_cap > 0 {
            line("    • Raise max_new (or profile cap) if answers look cut off.".into());
        }
        if ctx_pct > 80.0 {
            line(format!(
                "    • Context nearly full ({ctx_pct:.0}%) — risk of failures / need /reset."
            ));
        }
        if gpu_avg < 40.0 && overall_decode_tps > 0.0 {
            line("    • Low GPU util → launch-bound or CPU-bound; fuse kernels / less sync.".into());
        } else if gpu_avg > 85.0 {
            line("    • High GPU util → optimize matmul/attention bandwidth next.".into());
        }
        line("════════════════════════════════════════════════════════════════".into());

        let model_slug = model_slug(&meta.model_path);

        // Machine-readable SUMMARY block (easy to diff between runs).
        line("  SUMMARY_KV".into());
        line(format!("    stamp={stamp}"));
        line(format!("    model={model_slug}"));
        line(format!("    model_path={}", meta.model_path));
        line(format!("    decode_backend={}", meta.decode.name()));
        line(format!("    mode={mode}"));
        line(format!("    overall_decode_tps={overall_decode_tps:.3}"));
        line(format!("    overall_prefill_tps={overall_prefill_tps:.3}"));
        line(format!("    decode_tps_first={decode_tps_first:.3}"));
        line(format!("    decode_tps_last={decode_tps_last:.3}"));
        line(format!("    decode_drop_pct={decode_drop_pct:.2}"));
        line(format!("    total_prefill_ms={total_prefill_ms:.2}"));
        line(format!("    total_decode_ms={total_decode_ms:.2}"));
        line(format!("    total_gen={total_gen}"));
        line(format!("    total_prompt={total_prompt}"));
        line(format!("    final_ctx={final_ctx}"));
        line(format!("    wall_s={elapsed:.3}"));
        line(format!("    first_prefill_ms={first_prefill:.2}"));
        line(format!("    later_prefill_avg_ms={later_prefill_avg:.2}"));
        line("════════════════════════════════════════════════════════════════".into());

        print!("{report}");

        // Compare only against the previous run of the *same* model.
        let prev = read_latest_for_model(&model_slug);
        let path = save_profile_log(&report, meta, &stamp, &model_slug);
        if let Some(ref p) = path {
            eprintln!("\nprofile log → {}", p.display());
            // Global latest + per-model latest (fair A/B).
            let _ = fs::write(Path::new(PROFILE_LOG_DIR).join("latest.txt"), &report);
            let _ = fs::write(
                Path::new(PROFILE_LOG_DIR).join(format!("latest_{model_slug}.txt")),
                &report,
            );
            if let Some(prev_text) = prev {
                print_compare(&prev_text, &report, &model_slug);
            }
        } else {
            eprintln!("\nprofile log: failed to write under {PROFILE_LOG_DIR}/");
        }

        path.unwrap_or_else(|| PathBuf::from(PROFILE_LOG_DIR))
    }
}

/// Short filesystem-safe name from a model path
/// (`models/Qwen2.5-3B-Instruct-Q4_K_M.gguf` → `Qwen2.5-3B-Instruct-Q4_K_M`).
fn model_slug(model_path: &str) -> String {
    let name = Path::new(model_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    let mut out = String::with_capacity(name.len());
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "model".into()
    } else {
        // Keep filenames readable; trim very long stems.
        if out.len() > 80 {
            out.truncate(80);
        }
        out
    }
}

fn local_timestamp() -> String {
    // Prefer local wall clock via PowerShell (Windows); fall back to unix seconds.
    #[cfg(windows)]
    {
        if let Ok(out) = Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                "Get-Date -Format 'yyyy-MM-dd_HH-mm-ss'",
            ])
            .output()
        {
            if out.status.success() {
                let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !s.is_empty() {
                    return s;
                }
            }
        }
    }
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("unix_{secs}")
}

fn save_profile_log(
    report: &str,
    meta: &ProfileMeta,
    stamp: &str,
    model_slug: &str,
) -> Option<PathBuf> {
    fs::create_dir_all(PROFILE_LOG_DIR).ok()?;
    // profile_<stamp>_<model>_<decode>.txt
    let fname = format!(
        "profile_{stamp}_{model_slug}_{}.txt",
        meta.decode.name()
    );
    let path = Path::new(PROFILE_LOG_DIR).join(&fname);
    // Avoid clobber if two runs share the same second.
    let path = if path.exists() {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Path::new(PROFILE_LOG_DIR).join(format!(
            "profile_{stamp}_{model_slug}_{}_{}.txt",
            meta.decode.name(),
            secs
        ))
    } else {
        path
    };
    let mut f = fs::File::create(&path).ok()?;
    f.write_all(report.as_bytes()).ok()?;
    // One-line index for quick grepping / filtering by model.
    let idx = Path::new(PROFILE_LOG_DIR).join("index.csv");
    let header_needed = !idx.exists();
    if let Ok(mut idxf) = fs::OpenOptions::new().create(true).append(true).open(&idx) {
        if header_needed {
            let _ = writeln!(
                idxf,
                "stamp,model,decode,file,overall_decode_tps,decode_tps_first,decode_tps_last,decode_drop_pct,final_ctx"
            );
        }
        let (od, f1, fl, drop, ctx) = parse_summary_metrics(report);
        let _ = writeln!(
            idxf,
            "{},{},{},{},{:.3},{:.3},{:.3},{:.2},{}",
            stamp,
            model_slug,
            meta.decode.name(),
            path.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
            od,
            f1,
            fl,
            drop,
            ctx
        );
    }
    Some(path)
}

/// Previous run for this model only (`latest_<model>.txt`).
fn read_latest_for_model(model_slug: &str) -> Option<String> {
    let p = Path::new(PROFILE_LOG_DIR).join(format!("latest_{model_slug}.txt"));
    fs::read_to_string(p).ok()
}

fn parse_summary_metrics(report: &str) -> (f64, f64, f64, f64, usize) {
    let mut od = 0.0;
    let mut f1 = 0.0;
    let mut fl = 0.0;
    let mut drop = 0.0;
    let mut ctx = 0usize;
    for line in report.lines() {
        let t = line.trim();
        if let Some(v) = t.strip_prefix("overall_decode_tps=") {
            od = v.parse().unwrap_or(0.0);
        } else if let Some(v) = t.strip_prefix("decode_tps_first=") {
            f1 = v.parse().unwrap_or(0.0);
        } else if let Some(v) = t.strip_prefix("decode_tps_last=") {
            fl = v.parse().unwrap_or(0.0);
        } else if let Some(v) = t.strip_prefix("decode_drop_pct=") {
            drop = v.parse().unwrap_or(0.0);
        } else if let Some(v) = t.strip_prefix("final_ctx=") {
            ctx = v.parse().unwrap_or(0);
        }
    }
    (od, f1, fl, drop, ctx)
}

fn print_compare(prev: &str, curr: &str, model_slug: &str) {
    let (pod, pf1, pfl, pdrop, pctx) = parse_summary_metrics(prev);
    let (cod, cf1, cfl, cdrop, cctx) = parse_summary_metrics(curr);
    if pod == 0.0 && pf1 == 0.0 {
        return;
    }
    let d = |a: f64, b: f64| b - a;
    let pct = |a: f64, b: f64| {
        if a.abs() < 1e-9 {
            0.0
        } else {
            100.0 * (b - a) / a
        }
    };
    println!();
    println!("════════════════════════════════════════════════════════════════");
    println!("  vs PREVIOUS same model  ({model_slug})");
    println!("  (latest_{model_slug}.txt)");
    println!("────────────────────────────────────────────────────────────────");
    println!(
        "    overall decode  {pod:.1} → {cod:.1} t/s   ({:+.1}  {:+.0}%)",
        d(pod, cod),
        pct(pod, cod)
    );
    println!(
        "    first-turn      {pf1:.1} → {cf1:.1} t/s   ({:+.1}  {:+.0}%)",
        d(pf1, cf1),
        pct(pf1, cf1)
    );
    println!(
        "    last-turn       {pfl:.1} → {cfl:.1} t/s   ({:+.1}  {:+.0}%)",
        d(pfl, cfl),
        pct(pfl, cfl)
    );
    println!(
        "    decode drop %   {pdrop:.0} → {cdrop:.0}     ({:+.1})",
        d(pdrop, cdrop)
    );
    println!("    final ctx       {pctx} → {cctx}");
    println!("════════════════════════════════════════════════════════════════");
}

/// util, mem_util, mem_used, mem_total, power, temp, sm_clock, mem_clock
fn sample_gpu() -> Option<(f32, f32, f32, f32, f32, f32, f32, f32)> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let line = String::from_utf8_lossy(&out.stdout);
    let line = line.lines().next()?.trim();
    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
    if parts.len() < 8 {
        // fallback shorter query
        return sample_gpu_basic();
    }
    let parse = |i: usize| parts[i].parse::<f32>().unwrap_or(0.0);
    Some((
        parse(0),
        parse(1),
        parse(2),
        parse(3),
        parse(4),
        parse(5),
        parse(6),
        parse(7),
    ))
}

fn sample_gpu_basic() -> Option<(f32, f32, f32, f32, f32, f32, f32, f32)> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let line = String::from_utf8_lossy(&out.stdout);
    let line = line.lines().next()?.trim();
    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
    if parts.len() < 6 {
        return None;
    }
    let parse = |i: usize| parts[i].parse::<f32>().unwrap_or(0.0);
    Some((
        parse(0),
        parse(1),
        parse(2),
        parse(3),
        parse(4),
        parse(5),
        0.0,
        0.0,
    ))
}

fn sample_cpu() -> Option<f32> {
    #[cfg(windows)]
    {
        let out = Command::new("wmic")
            .args(["cpu", "get", "loadpercentage", "/value"])
            .output()
            .ok()?;
        let s = String::from_utf8_lossy(&out.stdout);
        for line in s.lines() {
            if let Some(v) = line.strip_prefix("LoadPercentage=") {
                return v.trim().parse().ok();
            }
        }
        None
    }
    #[cfg(not(windows))]
    {
        None
    }
}

/// (used_mb, total_mb) system memory snapshot via wmic (Windows).
fn sample_host_mem() -> Option<(f32, f32)> {
    #[cfg(windows)]
    {
        let out = Command::new("wmic")
            .args([
                "OS",
                "get",
                "FreePhysicalMemory,TotalVisibleMemorySize",
                "/value",
            ])
            .output()
            .ok()?;
        let s = String::from_utf8_lossy(&out.stdout);
        let mut free_kb = None;
        let mut total_kb = None;
        for line in s.lines() {
            if let Some(v) = line.strip_prefix("FreePhysicalMemory=") {
                free_kb = v.trim().parse::<f32>().ok();
            }
            if let Some(v) = line.strip_prefix("TotalVisibleMemorySize=") {
                total_kb = v.trim().parse::<f32>().ok();
            }
        }
        let free = free_kb?;
        let total = total_kb?;
        let used = (total - free) / 1024.0;
        let total_mb = total / 1024.0;
        Some((used, total_mb))
    }
    #[cfg(not(windows))]
    {
        None
    }
}
