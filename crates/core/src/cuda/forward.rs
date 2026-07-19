//! Chunked prefill / decode entry points + CUDA graph capture.

use super::kv::CudaKv;
use super::layer::ChunkDims;
use super::model::CudaModel;
use super::types::{MAX_BATCH, MAX_VERIFY_TOKENS};
use anyhow::{bail, Context, Result};
use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::PushKernelArg;

fn cuda_graph_debug(message: &str) {
    if std::env::var_os("TARAFER_CUDA_GRAPH_DEBUG").is_some() {
        eprintln!("{message}");
    }
}

impl CudaModel {
    /// Copy the current next-token logits to the host for quality sampling.
    /// Greedy benchmarks keep using the device argmax path and pay no copy cost.
    pub fn current_logits(&self) -> Result<Vec<(u32, f32)>> {
        let active = self
            .output
            .as_ref()
            .map_or(self.token_embd.n_cols, |m| m.n_cols);
        let host = self.stream.clone_dtoh(&self.logits)?;
        let mut out: Vec<(u32, f32)> = host
            .into_iter()
            .take(active)
            .enumerate()
            .map(|(id, logit)| (id as u32, logit))
            .collect();
        if let Some(id) = self.output_special_id {
            let special = self.stream.clone_dtoh(&self.special_logit)?;
            if let Some(&logit) = special.first() {
                out.push((id, logit));
            }
        }
        Ok(out)
    }

    /// Run tokens through the model; return greedy next-token id.
    pub fn forward_greedy(&mut self, tokens: &[u32], cache: &mut CudaKv) -> Result<u32> {
        if tokens.is_empty() {
            bail!("empty tokens");
        }
        if cache.len + tokens.len() > cache.max_seq {
            bail!("context full");
        }

        if tokens.len() == 1 {
            return self.forward_decode_one(tokens[0], cache);
        }

        // Tara MoE Q4 packs (and sparse MoE prefill) use n_tok=1 chunks.
        let max_chunk = if self.cfg.is_moe() { 1 } else { MAX_BATCH };
        let mut offset = 0usize;
        while offset < tokens.len() {
            let n = (tokens.len() - offset).min(max_chunk);
            let chunk = &tokens[offset..offset + n];
            let pos0 = cache.len + offset;
            let is_last = offset + n == tokens.len();
            self.forward_chunk(chunk, pos0, is_last, false, cache)?;
            offset += n;
        }
        cache.len += tokens.len();

        // clone_dtoh device-synchronizes; no extra stream.synchronize needed.
        let idx = self.stream.clone_dtoh(&self.argmax_buf)?;
        Ok(self.map_output_id(idx[0] as u32))
    }

    /// Single-token decode with optional CUDA graph replay.
    fn forward_decode_one(&mut self, token: u32, cache: &mut CudaKv) -> Result<u32> {
        if cache.len + 1 > cache.max_seq {
            bail!("context full");
        }
        let pos0 = cache.len;

        if self.graph_active {
            // Update device-side scalars, then replay the captured kernel sequence.
            self.stream
                .memcpy_htod(&[pos0 as i32], &mut self.d_pos0)
                .context("upload d_pos0")?;
            self.stream
                .memcpy_htod(&[token as i32], &mut self.d_token)
                .context("upload d_token")?;
            self.decode_graph
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("graph_active but no graph"))?
                .0
                .launch()
                .context("cuda graph launch")?;
        } else {
            // Eager path (host scalars). Always correct; used before capture and if graphs off.
            self.forward_chunk(&[token], pos0, true, false, cache)?;
            if self.cuda_graph && !self.graph_tried {
                self.try_capture_decode_graph(cache)?;
            }
        }

        cache.len += 1;
        // clone_dtoh already device-synchronizes; skip extra stream.synchronize().
        let idx = self.stream.clone_dtoh(&self.argmax_buf)?;
        Ok(self.map_output_id(idx[0] as u32))
    }

    fn map_output_id(&self, id: u32) -> u32 {
        let active = self
            .output
            .as_ref()
            .map_or(self.token_embd.n_cols, |m| m.n_cols) as u32;
        if id == active {
            self.output_special_id.unwrap_or(id)
        } else {
            id
        }
    }

    /// Capture a single-token decode graph. Call after at least one eager decode.
    fn try_capture_decode_graph(&mut self, cache: &mut CudaKv) -> Result<()> {
        self.graph_tried = true;
        if !self.cuda_graph {
            return Ok(());
        }
        if cache.max_seq < 2 {
            cuda_graph_debug("CUDA graph | skipped (max_seq < 2)");
            return Ok(());
        }

        let safe_pos = (cache.max_seq - 1) as i32;
        let dummy_tok = 0i32;
        let flags0 = unsafe { std::mem::transmute::<u32, CUgraphInstantiate_flags>(0) };

        // Drain ALL context streams (null + non-blocking) so capture has no cross-stream deps.
        self._ctx
            .synchronize()
            .context("ctx synchronize before graph capture")?;
        self.stream
            .memcpy_htod(&[safe_pos], &mut self.d_pos0)
            .context("graph d_pos0")?;
        self.stream
            .memcpy_htod(&[dummy_tok], &mut self.d_token)
            .context("graph d_token")?;
        self.stream.synchronize().context("param upload sync")?;
        self._ctx.synchronize().context("ctx sync after params")?;

        // Probe: can we capture *any* kernel on this stream?
        let probe_ok = (|| -> Result<()> {
            self._ctx.synchronize()?;
            self.stream
                .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
            // Single tiny kernel — no multi-buffer deps.
            let n_vocab = self.cfg.n_vocab as i32;
            unsafe {
                self.stream
                    .launch_builder(&self.k.argmax)
                    .arg(&self.logits)
                    .arg(&n_vocab)
                    .arg(&mut self.argmax_buf)
                    .launch(cudarc::driver::LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
            let g = self
                .stream
                .end_capture(flags0)?
                .ok_or_else(|| anyhow::anyhow!("empty probe graph"))?;
            drop(g);
            Ok(())
        })();
        if let Err(e) = probe_ok {
            let _ = self.stream.end_capture(flags0);
            cuda_graph_debug(&format!(
                "CUDA graph | probe failed: {e:#} (eager decode continues; try --no-cuda-graph)"
            ));
            return Ok(());
        }

        // Hybrid GDN/conv state is a single recurrent buffer (not per-position).
        // Capture runs a dummy token that would corrupt live state — snapshot first.
        let hybrid = self.cfg.is_hybrid();
        let mut ssm_snap: Vec<cudarc::driver::CudaSlice<f32>> = Vec::new();
        let mut conv_snap: Vec<cudarc::driver::CudaSlice<f32>> = Vec::new();
        if hybrid {
            for li in 0..self.cfg.n_layer {
                if !self.cfg.is_linear_layer(li) {
                    ssm_snap.push(self.stream.alloc_zeros::<f32>(1)?);
                    conv_snap.push(self.stream.alloc_zeros::<f32>(1)?);
                    continue;
                }
                let se = self.cfg.ssm_state_elems().max(1);
                let ce = self.cfg.ssm_conv_state_elems().max(1);
                let mut ss = self.stream.alloc_zeros::<f32>(se)?;
                let mut cs = self.stream.alloc_zeros::<f32>(ce)?;
                self.stream
                    .memcpy_dtod(&cache.ssm[li], &mut ss)
                    .context("snapshot ssm")?;
                self.stream
                    .memcpy_dtod(&cache.conv[li], &mut cs)
                    .context("snapshot conv")?;
                ssm_snap.push(ss);
                conv_snap.push(cs);
            }
            self.stream.synchronize().context("snapshot sync")?;
        }

        // Full single-token decode capture (device-pos kernels read d_pos0/d_token).
        self._ctx.synchronize().ok();
        if let Err(e) = self
            .stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        {
            cuda_graph_debug(&format!(
                "CUDA graph | begin_capture failed: {e:?} (eager continues)"
            ));
            return Ok(());
        }
        let capture_result = (|| -> Result<()> {
            self.forward_chunk(&[0u32], safe_pos as usize, true, true, cache)?;
            let g = self
                .stream
                .end_capture(flags0)
                .context("end_capture")?
                .ok_or_else(|| anyhow::anyhow!("empty CUDA graph"))?;
            g.upload().context("graph upload")?;
            self.decode_graph = Some(super::model::SendCudaGraph(g));
            self.graph_active = true;
            cuda_graph_debug("CUDA graph | OK — single-token decode will REPLAY (faster launches)");
            Ok(())
        })();
        if let Err(e) = capture_result {
            let _ = self.stream.end_capture(flags0);
            self.decode_graph = None;
            self.graph_active = false;
            cuda_graph_debug(&format!(
                "CUDA graph | full capture failed: {e:#} (eager decode continues)"
            ));
        }

        // Restore recurrent state so the next real decode continues correctly.
        if hybrid {
            for li in 0..self.cfg.n_layer {
                if !self.cfg.is_linear_layer(li) {
                    continue;
                }
                self.stream
                    .memcpy_dtod(&ssm_snap[li], &mut cache.ssm[li])
                    .context("restore ssm")?;
                self.stream
                    .memcpy_dtod(&conv_snap[li], &mut cache.conv[li])
                    .context("restore conv")?;
            }
            self.stream.synchronize().context("restore sync")?;
            if self.graph_active {
                cuda_graph_debug("CUDA graph | hybrid recurrent state restored after capture");
            }
        }
        Ok(())
    }

    /// Multi-token forward returning greedy argmax at each position.
    /// Retained as the verification primitive for a future model-based speculator.
    pub fn forward_greedy_all(&mut self, tokens: &[u32], cache: &mut CudaKv) -> Result<Vec<u32>> {
        if tokens.is_empty() {
            bail!("empty tokens");
        }
        if cache.len + tokens.len() > cache.max_seq {
            bail!("context full");
        }
        // Multi-token path invalidates a single-token graph (pos/token geometry differs).
        // Keep graph; we only use it for n_tok==1.

        let mut out = Vec::with_capacity(tokens.len());
        let mut offset = 0usize;
        while offset < tokens.len() {
            let n = (tokens.len() - offset).min(MAX_VERIFY_TOKENS);
            let chunk = &tokens[offset..offset + n];
            let pos0 = cache.len + offset;
            self.forward_chunk(chunk, pos0, false, false, cache)?;
            out.extend(self.logits_for_rows(n)?);
            offset += n;
        }
        cache.len += tokens.len();
        Ok(out)
    }

    fn logits_for_rows(&mut self, n_tok: usize) -> Result<Vec<u32>> {
        if n_tok == 0 || n_tok > MAX_VERIFY_TOKENS {
            bail!("verification batch {n_tok} exceeds {MAX_VERIFY_TOKENS}");
        }
        let n_embd = self.cfg.n_embd as i32;
        let n_tok_i = n_tok as i32;
        let eps = self.cfg.rms_eps;
        unsafe {
            self.stream
                .launch_builder(&self.k.rms_norm)
                .arg(&self.x)
                .arg(&self.output_norm)
                .arg(&mut self.xb)
                .arg(&n_embd)
                .arg(&n_tok_i)
                .arg(&eps)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (n_tok as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        use super::matmul::gemm;
        if let Some(ref ow) = self.output {
            gemm(
                &self.stream,
                &self.k,
                ow,
                &self.xb,
                &mut self.logits_batch,
                n_tok_i,
                &mut self.gemv_partial,
                self.gemv_partial_stride,
            )?;
        } else {
            gemm(
                &self.stream,
                &self.k,
                &self.token_embd,
                &self.xb,
                &mut self.logits_batch,
                n_tok_i,
                &mut self.gemv_partial,
                self.gemv_partial_stride,
            )?;
        }
        let n_vocab = self
            .output
            .as_ref()
            .map_or(self.token_embd.n_cols, |m| m.n_cols) as i32;
        unsafe {
            self.stream
                .launch_builder(&self.k.argmax_rows)
                .arg(&self.logits_batch)
                .arg(&n_vocab)
                .arg(&n_tok_i)
                .arg(&mut self.argmax_batch)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (n_tok as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        self.stream.synchronize()?;
        let ids = self.stream.clone_dtoh(&self.argmax_batch)?;
        Ok(ids[..n_tok].iter().map(|&id| id as u32).collect())
    }

    fn forward_chunk(
        &mut self,
        tokens: &[u32],
        pos0: usize,
        compute_logits: bool,
        use_device_pos: bool,
        cache: &mut CudaKv,
    ) -> Result<()> {
        let n_tok_u = tokens.len();
        let head_dim = self.cfg.head_dim();
        let n_kv_heads = self.cfg.n_head_kv;
        let d = ChunkDims {
            n_tok: n_tok_u as i32,
            n_tok_u,
            n_embd: self.cfg.n_embd as i32,
            n_embd_u: self.cfg.n_embd,
            n_ff_u: self.cfg.n_ff,
            n_head: self.cfg.n_head as i32,
            n_head_u: self.cfg.n_head,
            n_kv: n_kv_heads as i32,
            n_kv_heads,
            head_dim,
            hd: head_dim as i32,
            stride: (n_kv_heads * head_dim) as i32,
            stride_u: n_kv_heads * head_dim,
            pos0,
            pos0_i: pos0 as i32,
            eps: self.cfg.rms_eps,
            theta: self.cfg.rope_theta,
            scale: (head_dim as f32).sqrt().recip(),
            use_device_pos: use_device_pos && n_tok_u == 1,
        };

        if d.use_device_pos {
            self.embed_one_device()?;
        } else {
            let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            self.embed_batch(&ids)?;
        }

        for li in 0..self.layers.len() {
            self.run_layer(li, &d, cache)?;
        }
        if compute_logits {
            self.logits_from_last(&d)?;
        }
        Ok(())
    }
}
