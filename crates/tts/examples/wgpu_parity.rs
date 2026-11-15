//! cargo run -p taraference-tts --features wgpu --example wgpu_parity -- MODEL_DIR FIXTURES
#[cfg(feature = "wgpu")]
fn main() -> anyhow::Result<()> {
    use std::path::Path;
    use taraference_tts::wgpu_f5::NativeF5;
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(args.len() == 3, "usage: wgpu_parity MODEL_DIR FIXTURES");
    let dir = Path::new(&args[2]);
    let read = |name: &str| -> anyhow::Result<Vec<f32>> {
        let b = std::fs::read(dir.join(format!("{name}.bin")))?;
        anyhow::ensure!(b.len() % 4 == 0, "invalid f32 fixture {name}");
        Ok(b.chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect())
    };
    let bytes = std::fs::read(dir.join("ids.bin"))?;
    anyhow::ensure!(bytes.len() % 8 == 0, "invalid i64 fixture");
    let ids: Vec<i64> = bytes
        .chunks_exact(8)
        .map(|v| i64::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let model = NativeF5::load(Path::new(&args[1]))?;
    fn compare(name: &str, got: &[f32], reference: &[f32], limit: f64) -> anyhow::Result<()> {
        anyhow::ensure!(
            got.len() == reference.len() && !got.is_empty(),
            "{name}: shape mismatch"
        );
        let mut dot = 0.;
        let mut gg = 0.;
        let mut rr = 0.;
        let mut err = 0.;
        let mut max = 0f32;
        for (&g, &r) in got.iter().zip(reference) {
            anyhow::ensure!(g.is_finite() && r.is_finite(), "{name}: nonfinite output");
            dot += g as f64 * r as f64;
            gg += (g as f64).powi(2);
            rr += (r as f64).powi(2);
            err += ((g - r) as f64).powi(2);
            max = max.max((g - r).abs());
        }
        let cosine = dot / (gg * rr).sqrt();
        let nrmse = (err / rr.max(1e-30)).sqrt();
        eprintln!("{name}: cosine={cosine:.7}, relative_rmse={nrmse:.7}, max_abs={max:.6}");
        anyhow::ensure!(cosine >= 0.985 && nrmse <= limit, "{name}: parity failed");
        Ok(())
    }
    let velocity = model.velocity(&read("x")?, &read("cond")?, &ids, 0.3, 2.0)?;
    compare(
        "DiT velocity (F5-Myanmar-TTS v2)",
        &velocity,
        &read("velocity")?,
        0.05,
    )?;
    if dir.join("sample.bin").is_file() {
        let sample = model.sample(&read("x")?, &read("cond")?, &ids, &read("timesteps")?, 2.0)?;
        compare(
            "F5 final mel (Euler sampler)",
            &sample,
            &read("sample")?,
            0.05,
        )?;
    }
    let (re, im) = model.spectrogram(&read("mel")?, ids.len())?;
    compare("Vocos real", &re, &read("re")?, 0.01)?;
    compare("Vocos imaginary", &im, &read("im")?, 0.01)?;
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
fn main() {
    eprintln!("enable --features wgpu");
    std::process::exit(1);
}
