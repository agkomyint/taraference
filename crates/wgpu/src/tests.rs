use super::*;
use taraference_core::quant::{f16_to_f32, f32_to_f16};

#[test]
fn q6_scale_repack_preserves_payload_and_exact_coefficients() -> Result<()> {
    let mut raw = Vec::new();
    for bits in [
        0u16, 0x8000, 1, 0x03ff, 0x0400, 0x3800, 0x3c00, 0x7bff, 0xfbff,
    ] {
        for start in (0..256).step_by(16) {
            let mut block = [0u8; 210];
            for (i, b) in block[..192].iter_mut().enumerate() {
                *b = (i * 73 + 19) as u8;
            }
            for i in 0..16 {
                block[192 + i] = (start + i) as u8;
            }
            block[208..210].copy_from_slice(&bits.to_le_bytes());
            raw.extend(block);
        }
    }
    let packed = repack_q6_scales(&raw)?;
    for (source, dest) in raw.chunks_exact(210).zip(packed.chunks_exact(256)) {
        assert_eq!(&source[..192], &dest[..192]);
        let d = f16_to_f32(u16::from_le_bytes([source[208], source[209]])) as f64;
        for i in 0..16 {
            let actual =
                f32::from_le_bytes(dest[192 + i * 4..196 + i * 4].try_into().unwrap()) as f64;
            assert_eq!(actual, d * (source[192 + i] as i8 as f64));
        }
    }
    assert!(repack_q6_scales(&[0]).is_err());
    let mut bad = [0u8; 210];
    bad[208..].copy_from_slice(&0x7c00u16.to_le_bytes());
    assert!(repack_q6_scales(&bad).is_err());
    Ok(())
}

fn reference(kind: u32, data: &[u8]) -> Vec<f32> {
    let half = |i| f16_to_f32(u16::from_le_bytes([data[i], data[i + 1]]));
    if kind == 0 {
        return data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
    }
    if kind == 1 {
        return (0..data.len()).step_by(2).map(half).collect();
    }
    let (bytes, values) = match kind {
        2 => (18, 32),
        8 => (34, 32),
        12 => (144, 256),
        13 => (176, 256),
        14 => (210, 256),
        _ => unreachable!(),
    };
    let mut out = Vec::new();
    for block in data.chunks_exact(bytes) {
        let h = |i| f16_to_f32(u16::from_le_bytes([block[i], block[i + 1]]));
        let mut row = vec![0.; values];
        match kind {
            2 => {
                for j in 0..16 {
                    row[j] = h(0) * ((block[2 + j] & 15) as i32 - 8) as f32;
                    row[j + 16] = h(0) * ((block[2 + j] >> 4) as i32 - 8) as f32;
                }
            }
            8 => {
                for j in 0..32 {
                    row[j] = h(0) * (block[2 + j] as i8 as f32);
                }
            }
            12 | 13 => {
                let scales = &block[4..16];
                let qs = if kind == 12 {
                    &block[16..]
                } else {
                    &block[48..]
                };
                for pair in 0..4 {
                    for part in 0..2 {
                        let j = pair * 2 + part;
                        let (sc, min) = if j < 4 {
                            (scales[j] & 63, scales[j + 4] & 63)
                        } else {
                            (
                                (scales[j + 4] & 15) | ((scales[j - 4] >> 6) << 4),
                                (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
                            )
                        };
                        for l in 0..32 {
                            let mut q = ((qs[pair * 32 + l] >> (part * 4)) & 15) as u32;
                            if kind == 13 && block[16 + l] & (1 << j) != 0 {
                                q += 16;
                            }
                            row[pair * 64 + part * 32 + l] =
                                h(0) * sc as f32 * q as f32 - h(2) * min as f32;
                        }
                    }
                }
            }
            14 => {
                for half in 0..2 {
                    for l in 0..32 {
                        let ql = &block[half * 64..];
                        let high = block[128 + half * 32 + l];
                        let quants = [
                            (ql[l] & 15) | ((high & 3) << 4),
                            (ql[l + 32] & 15) | (((high >> 2) & 3) << 4),
                            (ql[l] >> 4) | (((high >> 4) & 3) << 4),
                            (ql[l + 32] >> 4) | (((high >> 6) & 3) << 4),
                        ];
                        for part in 0..4 {
                            let scale = block[192 + half * 8 + part * 2 + l / 16] as i8 as f32;
                            row[half * 128 + part * 32 + l] =
                                h(208) * scale * (quants[part] as i32 - 32) as f32;
                        }
                    }
                }
            }
            _ => unreachable!(),
        }
        out.extend(row);
    }
    out
}

#[test]
#[ignore = "requires a hardware wgpu adapter; run explicitly on the target GPU"]
fn packed_weight_unpack_and_gemv_match_cpu() -> Result<()> {
    let gpu = Gpu::new()?;
    for (kind, source_kind) in [
        (0, 0),
        (1, 1),
        (2, 2),
        (8, 8),
        (12, 12),
        (13, 13),
        (14, 14),
        (114, 14),
    ] {
        // Exercise K-tail masking as well as the M/token tails. K-block formats
        // require 256 columns; F32/F16 can have odd widths, Q4_0/Q8_0 need only 32.
        let width = match source_kind {
            0 | 1 => 273usize,
            2 | 8 => 96,
            _ => 256,
        };
        let rows = 7usize;
        let bytes = match source_kind {
            0 => width * rows * 4,
            1 => width * rows * 2,
            2 => width * rows / 32 * 18,
            8 => width * rows / 32 * 34,
            12 => width * rows / 256 * 144,
            13 => width * rows / 256 * 176,
            14 => width * rows / 256 * 210,
            _ => unreachable!(),
        };
        let mut data: Vec<u8> = (0..bytes).map(|i| ((i * 73 + 19) % 256) as u8).collect();
        match source_kind {
            0 => {
                data = (0..width * rows)
                    .flat_map(|i| ((i as f32 * 0.13).sin()).to_le_bytes())
                    .collect()
            }
            1 => {
                data = (0..width * rows)
                    .flat_map(|i| f32_to_f16((i as f32 * 0.13).sin()).to_le_bytes())
                    .collect()
            }
            _ => {
                let block = match source_kind {
                    2 => 18,
                    8 => 34,
                    12 => 144,
                    13 => 176,
                    14 => 210,
                    _ => unreachable!(),
                };
                for b in data.chunks_exact_mut(block) {
                    let off = if source_kind == 14 { 208 } else { 0 };
                    b[off..off + 2].copy_from_slice(&f32_to_f16(0.125).to_le_bytes());
                    if source_kind == 12 || source_kind == 13 {
                        b[2..4].copy_from_slice(&f32_to_f16(0.25).to_le_bytes());
                    }
                }
            }
        }
        let expected = reference(source_kind, &data);
        if kind == 114 {
            data = repack_q6_scales(&data)?;
        }
        let weight = gpu.upload("quant fixture", &data)?;
        let out = gpu.buffer("unpacked", width * rows);
        let mut p = [0u32; 16];
        p[0] = (width * rows) as u32;
        p[2] = kind;
        let op = gpu.op(
            "unpack",
            Some(&weight),
            None,
            None,
            &out,
            None,
            p,
            [(p[0]).div_ceil(128), 1, 1],
        );
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        gpu.encode(&mut encoder, &[op]);
        gpu.queue.submit([encoder.finish()]);
        let actual: Vec<f32> = gpu
            .read(&out, width * rows)?
            .into_iter()
            .map(f32::from_bits)
            .collect();
        for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
            ensure!(
                (a - e).abs() < 1e-5 * e.abs().max(1.),
                "unpack kind={kind}, element={i}, {a} != {e}"
            );
        }
        let input: Vec<f32> = (0..width).map(|i| (i as f32 * 0.17).sin()).collect();
        let a = gpu.upload("input", bytemuck::cast_slice(&input))?;
        let result = gpu.buffer("gemv", rows);
        p[0] = width as u32;
        p[1] = rows as u32;
        let op = gpu.op(
            "gemv",
            Some(&weight),
            Some(&a),
            None,
            &result,
            None,
            p,
            [(rows as u32).div_ceil(4), 1, 1],
        );
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        gpu.encode(&mut encoder, &[op]);
        gpu.queue.submit([encoder.finish()]);
        let actual: Vec<f32> = gpu
            .read(&result, rows)?
            .into_iter()
            .map(f32::from_bits)
            .collect();
        for (row, &got) in actual.iter().enumerate() {
            let expected: f32 = expected[row * width..(row + 1) * width]
                .iter()
                .zip(&input)
                .map(|(a, b)| a * b)
                .sum();
            ensure!(
                (got - expected).abs() < 0.003 + expected.abs() * 1e-5,
                "GEMV kind={kind}, row={row}: {got} != {expected}"
            );
        }
        eprintln!("packing + GEMV kind {kind}: pass");
        if kind == 12 {
            let mut up_data = data.clone();
            for block in up_data.chunks_exact_mut(144) {
                for v in &mut block[16..] {
                    *v ^= 0xa5;
                }
            }
            let up_expected = reference(12, &up_data);
            let up_weight = gpu.upload("paired up weights", &up_data)?;
            let fused = gpu.buffer("paired SwiGLU", rows);
            let op = gpu.op_extra(
                "gate_up_q4k",
                Some(&weight),
                Some(&a),
                None,
                &fused,
                None,
                p,
                [(rows as u32).div_ceil(4), 1, 1],
                Some(&up_weight),
            );
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            gpu.encode(&mut encoder, &[op]);
            gpu.queue.submit([encoder.finish()]);
            let actual: Vec<f32> = gpu
                .read(&fused, rows)?
                .into_iter()
                .map(f32::from_bits)
                .collect();
            for row in 0..rows {
                let dot = |weights: &[f32]| -> f32 {
                    weights[row * width..(row + 1) * width]
                        .iter()
                        .zip(&input)
                        .map(|(w, x)| w * x)
                        .sum()
                };
                let gate = dot(&expected);
                let e = (gate / (1. + (-gate).exp())) * dot(&up_expected);
                ensure!(
                    (actual[row] - e).abs() < 0.05 + e.abs() * 1e-5,
                    "paired gate/up row {row}: {} != {e}",
                    actual[row]
                );
            }
            eprintln!("paired Q4_K gate/up + SwiGLU: pass");
        }
        // Tail tiles, multiple tokens, bias and in-place residual fusion.
        let count = 17usize;
        let inputs: Vec<f32> = (0..count * width)
            .map(|i| (i as f32 * 0.071).sin())
            .collect();
        let bias: Vec<f32> = (0..rows).map(|i| i as f32 * 0.125).collect();
        let initial: Vec<f32> = (0..count * rows).map(|i| i as f32 * 0.01).collect();
        let batch_input = gpu.upload("GEMM inputs", bytemuck::cast_slice(&inputs))?;
        let bias_buffer = gpu.upload("bias", bytemuck::cast_slice(&bias))?;
        let batch_out = gpu.buffer("GEMM output", count * rows);
        gpu.queue
            .write_buffer(&batch_out, 0, bytemuck::cast_slice(&initial));
        gpu.queue.write_buffer(
            &gpu.state,
            0,
            bytemuck::cast_slice(&[0u32, 0, count as u32, 0]),
        );
        p[10] = 1;
        p[11] = 1;
        let op = gpu.op(
            "gemv",
            Some(&weight),
            Some(&batch_input),
            Some(&bias_buffer),
            &batch_out,
            None,
            p,
            [(rows as u32).div_ceil(4), 1, 1],
        );
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        gpu.encode_count(&mut encoder, &[op], count as u32);
        gpu.queue.submit([encoder.finish()]);
        let got: Vec<f32> = gpu
            .read(&batch_out, count * rows)?
            .into_iter()
            .map(f32::from_bits)
            .collect();
        for t in 0..count {
            for row in 0..rows {
                let dot: f32 = expected[row * width..(row + 1) * width]
                    .iter()
                    .zip(&inputs[t * width..(t + 1) * width])
                    .map(|(w, x)| w * x)
                    .sum();
                let e = dot + bias[row] + initial[t * rows + row];
                ensure!(
                    (got[t * rows + row] - e).abs() < 0.003 + e.abs() * 1e-5,
                    "GEMM kind={kind}, token={t}, row={row}: {} != {e}",
                    got[t * rows + row]
                );
            }
        }
        eprintln!("batched GEMM + bias/residual kind {kind}: pass");
    }
    Ok(())
}

#[test]
#[ignore = "requires Qwen2.5-3B baseline weights and a hardware GPU"]
fn batched_prefill_matches_sequential_across_boundaries() -> Result<()> {
    let file = GgufFile::open("../../models/Qwen2.5-3B-Instruct-Q4_K_M.gguf")?;
    let tok = taraference_core::Tokenizer::from_gguf(&file)?;
    let text = "Sunlight contains many colors. Air molecules scatter blue light more strongly than red light. ".repeat(14);
    let ids = tok.encode(&text, false);
    let mut model = WgpuModel::load(&file, 512)?;
    for length in [1, 31, 32, 33, 65, 182] {
        let mut results = Vec::new();
        for batch in [1, 32] {
            model.clear();
            model.prefill_batch = batch;
            let first = model.forward_greedy(&ids[..length])?;
            let prompt = model.current_logits()?;
            model.forward_greedy(&[first])?;
            results.push((first, prompt, model.current_logits()?));
            ensure!(
                model.len == length + 1,
                "cache length differs after batch {batch}"
            );
        }
        ensure!(
            results[0].0 == results[1].0,
            "first-token mismatch at prompt length {length}"
        );
        for (a, b) in [
            (&results[0].1, &results[1].1),
            (&results[0].2, &results[1].2),
        ] {
            let error: f64 = a
                .iter()
                .zip(b)
                .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
                .sum();
            let norm: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum();
            let relative = (error / norm).sqrt();
            ensure!(
                relative < 0.005,
                "batch/sequential logits drift {relative} at prompt length {length}"
            );
        }
        eprintln!("prefill boundary {length}, decode continuation and cache reset: pass");
    }
    model.clear();
    ensure!(
        model.forward_greedy(&[]).is_err() && model.len == 0,
        "empty input changed cache"
    );
    ensure!(
        model.forward_greedy(&[model.cfg.n_vocab as u32]).is_err() && model.len == 0,
        "invalid token changed cache"
    );
    ensure!(
        model.forward_greedy(&vec![1; 513]).is_err() && model.len == 0,
        "oversized input changed cache"
    );
    Ok(())
}
