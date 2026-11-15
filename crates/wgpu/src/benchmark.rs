use anyhow::{bail, ensure, Result};
use serde_json::json;
use std::{io::Write, path::Path, time::Instant};
use taraference_gguf::GgufFile;
use taraference_wgpu::{runner::prompt_tokens, WgpuModel};

enum Backend {
    Wgpu(WgpuModel),
    #[cfg(feature = "cuda-compare")]
    Cuda {
        model: Box<taraference_core::CudaModel>,
        kv: taraference_core::CudaKv,
    },
}
impl Backend {
    fn forward(&mut self, ids: &[u32]) -> Result<u32> {
        match self {
            Self::Wgpu(m) => m.forward_greedy(ids),
            #[cfg(feature = "cuda-compare")]
            Self::Cuda { model, kv } => model.forward_greedy(ids, kv),
        }
    }
    fn clear(&mut self) {
        match self {
            Self::Wgpu(m) => m.clear(),
            #[cfg(feature = "cuda-compare")]
            Self::Cuda { kv, .. } => kv.clear(),
        }
    }
    fn adapter(&self) -> String {
        match self {
            Self::Wgpu(m) => m.adapter().into(),
            #[cfg(feature = "cuda-compare")]
            Self::Cuda { model, .. } => model.gpu_name.clone(),
        }
    }
    fn logits(&self) -> Result<Vec<f32>> {
        match self {
            Self::Wgpu(m) => m.current_logits(),
            #[cfg(feature = "cuda-compare")]
            Self::Cuda { model, .. } => Ok(model
                .current_logits()?
                .into_iter()
                .map(|(_, v)| v)
                .collect()),
        }
    }
    fn graph_active(&self) -> bool {
        match self {
            Self::Wgpu(_) => false,
            #[cfg(feature = "cuda-compare")]
            Self::Cuda { model, .. } => model.graph_active,
        }
    }
    fn prefill_batch(&self) -> Option<usize> {
        match self {
            Self::Wgpu(model) => Some(model.prefill_batch()),
            #[cfg(feature = "cuda-compare")]
            Self::Cuda { .. } => None,
        }
    }
    fn kernel_settings(&self) -> serde_json::Value {
        match self {
            Self::Wgpu(model) => model.kernel_settings(),
            #[cfg(feature = "cuda-compare")]
            Self::Cuda { .. } => serde_json::Value::Null,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    path: &Path,
    ctx: usize,
    prompt: &str,
    count: usize,
    runs: usize,
    warmup: usize,
    backend: &str,
    report: &Path,
) -> Result<()> {
    ensure!(
        runs > 0 && count >= 2 && warmup >= 1,
        "require runs>=1, max-new>=2 and warmup>=1"
    );
    ensure!(
        std::env::var_os("TARAFER_VOCAB_LIMIT").is_none(),
        "unset TARAFER_VOCAB_LIMIT: comparison requires full vocabulary"
    );
    let start = Instant::now();
    let file = GgufFile::open(path)?;
    let (tok, ids) = prompt_tokens(&file, prompt, false)?;
    ensure!(ids.len() + count - 1 <= ctx, "benchmark exceeds context");
    let cfg = taraference_core::ModelConfig::from_gguf(&file)?;
    ensure!(
        cfg.architecture == "qwen2" && (1..=2048).contains(&ctx),
        "comparison supports Qwen2/Qwen2.5 and context<=2048"
    );
    let mut runtime = match backend {
        "wgpu" => Backend::Wgpu(WgpuModel::load(&file, ctx)?),
        #[cfg(feature = "cuda-compare")]
        "cuda" => {
            let mut model = taraference_core::CudaModel::load(&file)?;
            model.set_cuda_graph(true);
            let kv = model.alloc_kv(ctx)?;
            Backend::Cuda {
                model: Box::new(model),
                kv,
            }
        }
        _ => {
            bail!("CUDA comparison requires --features cuda-compare; wgpu itself requires no CUDA")
        }
    };
    let load_ms = start.elapsed().as_secs_f64() * 1000.;
    // Warm all operations and the CUDA decode graph before measured runs.
    for _ in 0..warmup {
        runtime.clear();
        let mut next = runtime.forward(&ids)?;
        for _ in 1..count {
            next = runtime.forward(&[next])?;
        }
    }
    let mut measurements = Vec::new();
    for i in 0..runs {
        runtime.clear();
        let start = Instant::now();
        let mut next = runtime.forward(&ids)?;
        let ttft_ms = start.elapsed().as_secs_f64() * 1000.;
        let mut generated = vec![next];
        let decode = Instant::now();
        for _ in 1..count {
            next = runtime.forward(&[next])?;
            generated.push(next);
        }
        let decode_s = decode.elapsed().as_secs_f64();
        let decode_tps = (count - 1) as f64 / decode_s;
        let prefill_tps = ids.len() as f64 / (ttft_ms / 1000.);
        eprintln!("{backend} run {}: TTFT {ttft_ms:.2} ms, prefill {prefill_tps:.2} tok/s, decode {decode_tps:.2} tok/s", i + 1);
        measurements.push(json!({"ttft_ms":ttft_ms,"prefill_tps":prefill_tps,"decode_tps":decode_tps,"decode_ms":decode_s*1000.,"decode_steps":count-1,"generated_tokens":count,"token_ids":generated,"reply":tok.decode(&generated)}));
    }
    // Independent teacher-forced continuation: both backends see exactly the same
    // tokens even when small numeric differences change free-running generation.
    let teacher = tok.encode(
        "The capital of France is Paris. Two plus two equals four.",
        false,
    );
    ensure!(
        ids.len() + 8 <= ctx && teacher.len() >= 8,
        "verification context too short"
    );
    runtime.clear();
    runtime.forward(&ids)?;
    if let Some(parent) = report.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let logits_path = report.with_extension("logits.bin");
    let mut logits_file = std::io::BufWriter::new(std::fs::File::create(&logits_path)?);
    let mut positions = Vec::new();
    for step in 0..=8 {
        if step > 0 {
            runtime.forward(&[teacher[step - 1]])?;
        }
        if ![0, 1, 4, 8].contains(&step) {
            continue;
        }
        let logits = runtime.logits()?;
        ensure!(
            logits.len() == cfg.n_vocab && logits.iter().all(|v| v.is_finite()),
            "non-finite or truncated {backend} logits"
        );
        for value in &logits {
            logits_file.write_all(&value.to_le_bytes())?;
        }
        let mut order: Vec<_> = (0..logits.len()).collect();
        order.select_nth_unstable_by(5, |&a, &b| logits[b].total_cmp(&logits[a]));
        order.truncate(5);
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        positions.push(json!({"teacher_tokens":step,"context_tokens":ids.len()+step,"top5":order}));
    }
    logits_file.flush()?;
    let result = json!({"model":path.file_name().unwrap_or_default().to_string_lossy(),"model_bytes":std::fs::metadata(path)?.len(),"backend":backend,"adapter":runtime.adapter(),"context":ctx,"prompt":prompt,"prompt_ids":ids,"prompt_tokens":ids.len(),"full_vocabulary":cfg.n_vocab,"load_ms":load_ms,"warmup_runs":warmup,"runs":measurements,"cuda_graph_active":runtime.graph_active(),"wgpu_prefill_batch":runtime.prefill_batch(),"wgpu_kernel_settings":runtime.kernel_settings(),"verification":{"teacher_ids":teacher[..8].to_vec(),"positions":positions,"logits_file":logits_path},"build_profile":if cfg!(debug_assertions) {"debug"} else {"release"},"precision":if backend=="wgpu" {"packed GGUF values with lossless device repacking, fp32 activations, packed fp16 KV"} else {"existing Taraference CUDA defaults (DP4A where selected), fp16 KV"}});
    std::fs::write(report, serde_json::to_vec_pretty(&result)?)?;
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
