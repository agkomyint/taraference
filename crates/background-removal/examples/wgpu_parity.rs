//! Compares the native wgpu BiRefNet Lite port with PyTorch fixtures from
//! `tools/image/birefnet_wgpu_fixtures.py`.
//!
//! cargo run --release -p taraference-background-removal --features wgpu \
//!     --example wgpu_parity -- MODEL.safetensors FIXTURES [deform]
#[cfg(feature = "wgpu")]
fn main() -> anyhow::Result<()> {
    use anyhow::{ensure, Context};
    use std::{collections::HashMap, path::Path, time::Instant};
    use taraference_background_removal::wgpu_birefnet::NativeBiRefNet;

    const DEFORM: &str = "decoder.decoder_block4.dec_att.aspp_deforms.1.atrous_conv";
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        matches!(args.len(), 3 | 4),
        "usage: wgpu_parity MODEL.safetensors FIXTURES [deform]"
    );
    let dir = Path::new(&args[2]);
    let manifest = std::fs::read_to_string(dir.join("manifest.txt"))
        .with_context(|| format!("read {}", dir.join("manifest.txt").display()))?;
    let shapes: HashMap<&str, Vec<usize>> = manifest
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let name = fields.next()?;
            Some((name, fields.filter_map(|d| d.parse().ok()).collect()))
        })
        .collect();
    let fixture = |name: &str| -> anyhow::Result<Vec<f32>> {
        let bytes = std::fs::read(dir.join(format!("{name}.bin")))?;
        ensure!(bytes.len() % 4 == 0, "invalid f32 fixture {name}");
        Ok(bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect())
    };
    fn compare(name: &str, got: &[f32], reference: &[f32], limit: f64) -> anyhow::Result<bool> {
        ensure!(
            got.len() == reference.len() && !got.is_empty(),
            "{name}: {} values, reference has {}",
            got.len(),
            reference.len()
        );
        let (mut dot, mut gg, mut rr, mut err, mut max) = (0f64, 0f64, 0f64, 0f64, 0f32);
        for (&g, &r) in got.iter().zip(reference) {
            ensure!(g.is_finite() && r.is_finite(), "{name}: nonfinite output");
            dot += g as f64 * r as f64;
            gg += (g as f64).powi(2);
            rr += (r as f64).powi(2);
            err += ((g - r) as f64).powi(2);
            max = max.max((g - r).abs());
        }
        let cosine = dot / (gg * rr).sqrt();
        let nrmse = (err / rr.max(1e-30)).sqrt();
        let pass = cosine >= 0.999 && nrmse <= limit;
        eprintln!(
            "{} {name}: cosine={cosine:.7}, relative_rmse={nrmse:.7}, max_abs={max:.6}",
            if pass { "ok  " } else { "FAIL" }
        );
        Ok(pass)
    }

    let started = Instant::now();
    let model = NativeBiRefNet::load(Path::new(&args[1]))?;
    eprintln!(
        "BiRefNet Lite (upstream fp32 safetensors) on {}: load {:.2} s",
        model.adapter,
        started.elapsed().as_secs_f64()
    );
    let mut pass = true;
    let dims: [usize; 4] = shapes
        .get("deform_in")
        .and_then(|dims| dims.as_slice().try_into().ok())
        .context("manifest has no deform_in")?;
    let got = model.deformable_conv(DEFORM, &fixture("deform_in")?, dims)?;
    pass &= compare(
        "deformable conv 3x3 + BN",
        &got,
        &fixture("deform_out")?,
        0.01,
    )?;
    if args.len() == 4 {
        ensure!(args[3] == "deform", "unknown mode {:?}", args[3]);
    } else {
        let size = *shapes
            .get("input")
            .and_then(|dims| dims.last())
            .context("manifest has no input")?;
        let input = fixture("input")?;
        let started = Instant::now();
        let trace = model.trace(&input, size)?;
        eprintln!(
            "traced forward {size}x{size}: {:.2} s (includes shader compilation and readbacks)",
            started.elapsed().as_secs_f64()
        );
        for (name, got) in &trace {
            let limit = if name == "logits" { 0.02 } else { 0.01 };
            pass &= compare(name, got, &fixture(name)?, limit)?;
        }
        let (_, logits) = trace.last().context("empty trace")?;
        let reference = fixture("logits")?;
        let both = logits
            .iter()
            .zip(&reference)
            .filter(|(a, b)| **a > 0.0 && **b > 0.0)
            .count();
        let either = logits
            .iter()
            .zip(&reference)
            .filter(|(a, b)| **a > 0.0 || **b > 0.0)
            .count();
        if either > 0 {
            let iou = both as f64 / either as f64;
            eprintln!("mask IoU vs PyTorch: {iou:.5} ({either} foreground pixels)");
            pass &= iou >= 0.99;
        } else {
            eprintln!("mask IoU vs PyTorch: both masks are empty");
        }
        for run in 1..=2 {
            let started = Instant::now();
            model.logits(&input, size)?;
            eprintln!(
                "forward {size}x{size} run {run}: {:.2} s",
                started.elapsed().as_secs_f64()
            );
        }
    }
    ensure!(pass, "parity failed");
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
fn main() {
    eprintln!("enable --features wgpu");
    std::process::exit(1);
}
