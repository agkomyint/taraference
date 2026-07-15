//! Runtime profiler: sample CPU/GPU while multi-turn chat runs, print a report.

use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use taraference_core::TurnStats;

/// Cap generated tokens per turn during profile (comparable runs).
pub const PROFILE_MAX_NEW: usize = 128;

/// Realistic multi-turn script: greeting → short Q → follow-up → longer ask → recap.
/// Mimics how people actually chat (context grows; later turns prefill less but decode still full).
pub const MULTI_TURN_SCRIPT: &[&str] = &[
    "hi, who are you?",
    "what can you help me with in one sentence?",
    "ok give me 3 bullet ideas for a weekend project",
    "expand on the second idea a bit more",
    "summarize everything we talked about so far",
];

/// Single-turn fallback if user passes `--prompt` with `--profile`.
pub const PROFILE_PROMPT: &str =
    "Explain in one short paragraph how a transformer attention layer works.";

#[derive(Debug, Clone, Default)]
struct Sample {
    cpu_pct: f32,
    gpu_util: f32,
    gpu_mem_util: f32,
    gpu_mem_used_mb: f32,
    gpu_mem_total_mb: f32,
    gpu_power_w: f32,
    gpu_temp_c: f32,
}

#[derive(Debug, Default)]
struct SampleBuf {
    items: Vec<Sample>,
}

/// One turn in the multi-turn bench.
#[derive(Debug, Clone)]
pub struct TurnRow {
    pub index: usize,
    pub user: String,
    pub stats: TurnStats,
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
        let handle = thread::spawn(move || {
            let interval = Duration::from_millis(interval_ms.max(50));
            while !stop_c.load(Ordering::Relaxed) {
                let mut s = Sample::default();
                if let Some(g) = sample_gpu() {
                    s.gpu_util = g.0;
                    s.gpu_mem_util = g.1;
                    s.gpu_mem_used_mb = g.2;
                    s.gpu_mem_total_mb = g.3;
                    s.gpu_power_w = g.4;
                    s.gpu_temp_c = g.5;
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
            t0: Instant::now(),
        }
    }

    pub fn stop_and_report(&mut self, turns: &[TurnRow], mode: &str) {
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
            samples.iter().map(f).fold(0.0_f32, |a, b| a.max(b))
        };

        let cpu_avg = avg(|s| s.cpu_pct);
        let cpu_max = maxf(|s| s.cpu_pct);
        let gpu_avg = avg(|s| s.gpu_util);
        let gpu_max = maxf(|s| s.gpu_util);
        let gmem_avg = avg(|s| s.gpu_mem_used_mb);
        let gmem_max = maxf(|s| s.gpu_mem_used_mb);
        let gmem_tot = samples.last().map(|s| s.gpu_mem_total_mb).unwrap_or(0.0);
        let pwr_avg = avg(|s| s.gpu_power_w);
        let pwr_max = maxf(|s| s.gpu_power_w);
        let tmp_max = maxf(|s| s.gpu_temp_c);

        let total_prompt: usize = turns.iter().map(|t| t.stats.prompt_tokens).sum();
        let total_gen: usize = turns.iter().map(|t| t.stats.gen_tokens).sum();
        let total_prefill_ms: f64 = turns.iter().map(|t| t.stats.prefill_ms).sum();
        let total_decode_ms: f64 = turns.iter().map(|t| t.stats.decode_ms).sum();
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

        println!();
        println!("══════════════════════════════════════════════════════════");
        println!("  PROFILE REPORT  ({mode})");
        println!("══════════════════════════════════════════════════════════");
        println!(
            "  turns: {} | samples: {} over {:.2}s wall",
            turns.len(),
            samples.len(),
            elapsed
        );
        println!("──────────────────────────────────────────────────────────");
        println!("  PER-TURN (real chat shape: ctx grows each turn)");
        println!(
            "  {:>4}  {:>8}  {:>8}  {:>7}  {:>7}  {:>6}  {}",
            "turn", "prefill", "decode", "p_tok", "g_tok", "ctx", "user"
        );
        for t in turns {
            let user_short = if t.user.len() > 42 {
                format!("{}…", &t.user[..42])
            } else {
                t.user.clone()
            };
            println!(
                "  {:>4}  {:>6.0}ms  {:>5.1} t/s  {:>7}  {:>7}  {:>6}  {}",
                t.index + 1,
                t.stats.prefill_ms,
                t.stats.decode_tps,
                t.stats.prompt_tokens,
                t.stats.gen_tokens,
                t.stats.ctx_len,
                user_short
            );
        }
        println!("──────────────────────────────────────────────────────────");
        println!("  AGGREGATE SPEED");
        println!(
            "    prefill  {:>7.0} ms total | {:>5} prompt tok | {:>6.1} tok/s overall",
            total_prefill_ms, total_prompt, overall_prefill_tps
        );
        println!(
            "    decode   {:>7.0} ms total | {:>5} gen tok    | {:>6.1} tok/s overall",
            total_decode_ms, total_gen, overall_decode_tps
        );
        println!(
            "    wall     {:>7.0} ms sum of turns | final ctx {}",
            total_wall_ms, final_ctx
        );
        if turns.len() > 1 {
            println!(
                "    first-turn prefill {first_prefill:.0} ms | later turns avg prefill {later_prefill_avg:.0} ms"
            );
            println!(
                "    (later turns should be cheaper prefill if only new user tokens are fed)"
            );
        }
        println!("──────────────────────────────────────────────────────────");
        println!("  CPU");
        println!("    util    avg {cpu_avg:5.1}%  max {cpu_max:5.1}%");
        println!("  GPU");
        if samples.iter().any(|s| s.gpu_mem_total_mb > 0.0) {
            println!("    compute avg {gpu_avg:5.1}%  max {gpu_max:5.1}%");
            println!(
                "    memory  avg {gmem_avg:6.0} / {gmem_tot:.0} MiB  peak {gmem_max:.0} MiB"
            );
            println!("    power   avg {pwr_avg:5.1} W  max {pwr_max:5.1} W");
            println!("    temp    max {tmp_max:5.0} °C");
            println!(
                "    decode_tok/s per GPU%  {util_eff:.2}  (higher ⇒ more tokens for same util)"
            );
        } else {
            println!("    (nvidia-smi unavailable — install driver tools or check PATH)");
        }
        println!("──────────────────────────────────────────────────────────");
        println!("  READOUT");
        if turns.len() > 1 && later_prefill_avg > first_prefill * 0.8 {
            println!(
                "    Later-turn prefill not much cheaper → check multi-turn KV (should not re-prefill whole history)."
            );
        }
        if final_ctx > 0 && gmem_max > 0.0 {
            println!(
                "    Context grew to {final_ctx} tokens; VRAM peak {gmem_max:.0} MiB — watch KV growth on long chats."
            );
        }
        if gpu_avg < 40.0 && overall_decode_tps > 0.0 {
            println!("    GPU util low while decoding → kernel/launch bound.");
        } else if gpu_avg > 85.0 {
            println!("    GPU util high → near bandwidth/compute limit; optimize matmul/quant.");
        } else {
            println!("    Mixed util — Nsight next for kernel hotspots.");
        }
        if cpu_avg > 50.0 {
            println!("    CPU busy → host sync/copy/launch may cost decode tok/s.");
        }
        println!("══════════════════════════════════════════════════════════");
    }
}

fn sample_gpu() -> Option<(f32, f32, f32, f32, f32, f32)> {
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
