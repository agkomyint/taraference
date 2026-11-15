//! Warm ONNX Runtime CUDA timings for the shipped BiRefNet Lite model, as the
//! baseline for `wgpu_parity`.
//!
//! cargo run --release -p taraference-background-removal --example ort_timing -- \
//!     MODEL.onnx FIXTURES
use anyhow::{anyhow, Context, Result};
use ort::{ep, session::Session, value::Tensor};
use std::{path::Path, time::Instant};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(args.len() == 3, "usage: ort_timing MODEL.onnx FIXTURES");
    let bytes = std::fs::read(Path::new(&args[2]).join("input.bin")).context("read input.bin")?;
    let input: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let size = ((input.len() / 3) as f64).sqrt() as usize;
    let started = Instant::now();
    let cuda = ep::CUDA::default()
        .with_arena_extend_strategy(ep::ArenaExtendStrategy::SameAsRequested)
        .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Heuristic)
        .build()
        .error_on_failure();
    let mut session = Session::builder()
        .map_err(|e| anyhow!(e.to_string()))?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Disable)
        .map_err(|e| anyhow!(e.to_string()))?
        .with_execution_providers([cuda])
        .map_err(|e| anyhow!(e.to_string()))?
        .commit_from_file(&args[1])
        .map_err(|e| anyhow!(e.to_string()))?;
    eprintln!("ORT CUDA session: {:.2} s", started.elapsed().as_secs_f64());
    let name = session.inputs()[0].name().to_owned();
    for run in 1..=5 {
        let tensor = Tensor::from_array(([1usize, 3, size, size], input.clone()))?;
        let started = Instant::now();
        let outputs = session.run(ort::inputs![name.as_str() => tensor])?;
        let output = outputs.values().last().context("no outputs")?;
        let (_, logits) = output.try_extract_tensor::<f32>()?;
        let foreground = logits.iter().filter(|v| **v > 0.0).count();
        eprintln!(
            "ORT forward {size}x{size} run {run}: {:.2} s ({foreground} foreground pixels)",
            started.elapsed().as_secs_f64()
        );
    }
    Ok(())
}
