//! Chunked prefill / decode entry points + CUDA graph capture.

use super::kv::CudaKv;
use super::layer::ChunkDims;
use super::model::CudaModel;
use super::types::MAX_BATCH;
use anyhow::{bail, Context, Result};
use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::PushKernelArg;

impl CudaModel {
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

        let mut offset = 0usize;
        while offset < tokens.len() {
            let n = (tokens.len() - offset).min(MAX_BATCH);
            let chunk = &tokens[offset..offset + n];
            let pos0 = cache.len + offset;
            let is_last = offset + n == tokens.len();
            self.forward_chunk(chunk, pos0, is_last, false, cache)?;
            offset += n;
        }
        cache.len += tokens.len();

        self.stream.synchronize().context("cuda sync")?;
        let idx = self.stream.clone_dtoh(&self.argmax_buf)?;
        Ok(idx[0] as u32)
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
        self.stream.synchronize().context("cuda sync")?;
        let idx = self.stream.clone_dtoh(&self.argmax_buf)?;
        Ok(idx[0] as u32)
    }

    /// Capture a single-token decode graph. Call after at least one eager decode.
    fn try_capture_decode_graph(&mut self, cache: &mut CudaKv) -> Result<()> {
        self.graph_tried = true;
        if !self.cuda_graph {
            return Ok(());
        }
        if cache.max_seq < 2 {
            eprintln!("CUDA graph | skipped (max_seq < 2)");
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
            eprintln!(
                "CUDA graph | probe failed: {e:#} (eager decode continues; try --no-cuda-graph)"
            );
            return Ok(());
        }

        // Full single-token decode capture (device-pos kernels read d_pos0/d_token).
        self._ctx.synchronize().ok();
        if let Err(e) = self
            .stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        {
            eprintln!("CUDA graph | begin_capture failed: {e:?} (eager continues)");
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
            eprintln!("CUDA graph | OK — single-token decode will REPLAY (faster launches)");
            Ok(())
        })();
        if let Err(e) = capture_result {
            let _ = self.stream.end_capture(flags0);
            self.decode_graph = None;
            self.graph_active = false;
            eprintln!("CUDA graph | full capture failed: {e:#} (eager decode continues)");
        }
        Ok(())
    }

    /// Multi-token forward returning greedy argmax at each position.
    /// Retained as the verification primitive for a future model-based speculator.
    pub fn forward_greedy_all(
        &mut self,
        tokens: &[u32],
        cache: &mut CudaKv,
    ) -> Result<Vec<u32>> {
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
            let n = (tokens.len() - offset).min(MAX_BATCH);
            let chunk = &tokens[offset..offset + n];
            let pos0 = cache.len + offset;
            self.forward_chunk(chunk, pos0, false, false, cache)?;
            for t in 0..n {
                self.logits_at_row(t, n)?;
                self.stream.synchronize()?;
                let idx = self.stream.clone_dtoh(&self.argmax_buf)?;
                out.push(idx[0] as u32);
            }
            offset += n;
        }
        cache.len += tokens.len();
        Ok(out)
    }

    fn logits_at_row(&mut self, row: usize, n_tok: usize) -> Result<()> {
        let n_embd = self.cfg.n_embd as i32;
        let n_embd_u = self.cfg.n_embd;
        let row_i = row as i32;
        let _ = n_tok;
        unsafe {
            // copy_last with n_tok = row+1 copies x[row] into x1.
            self.stream
                .launch_builder(&self.k.copy_last)
                .arg(&self.x)
                .arg(&mut self.x1)
                .arg(&(row_i + 1))
                .arg(&n_embd)
                .launch(cudarc::driver::LaunchConfig::for_num_elems(n_embd_u as u32))?;
            let one = 1i32;
            let eps = self.cfg.rms_eps;
            self.stream
                .launch_builder(&self.k.rms_norm)
                .arg(&self.x1)
                .arg(&self.output_norm)
                .arg(&mut self.xb1)
                .arg(&n_embd)
                .arg(&one)
                .arg(&eps)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        use super::matmul::{gemv, try_gemv_global_q8, GemvResidual};
        if let Some(ref ow) = self.output {
            if !try_gemv_global_q8(
                &self.stream,
                &self.k,
                ow,
                &self.xb1,
                &mut self.logits,
                None,
                GemvResidual::None,
                &mut self.q8_x,
                &mut self.q8_d,
                &mut self.gemv_partial,
                self.gemv_partial_stride,
            )? {
                gemv(
                    &self.stream, &self.k, ow, &self.xb1, &mut self.logits, None,
                    GemvResidual::None, &mut self.gemv_partial, self.gemv_partial_stride,
                )?;
            }
        } else {
            if !try_gemv_global_q8(
                &self.stream,
                &self.k,
                &self.token_embd,
                &self.xb1,
                &mut self.logits,
                None,
                GemvResidual::None,
                &mut self.q8_x,
                &mut self.q8_d,
                &mut self.gemv_partial,
                self.gemv_partial_stride,
            )? {
                gemv(
                    &self.stream, &self.k, &self.token_embd, &self.xb1,
                    &mut self.logits, None, GemvResidual::None,
                    &mut self.gemv_partial, self.gemv_partial_stride,
                )?;
            }
        }
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
        Ok(())
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
