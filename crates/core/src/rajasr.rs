//! Native CUDA speech-to-text for Raj-0.6B (Qwen3-ASR architecture):
//! Whisper-style 128-mel frontend -> conv/transformer audio encoder (this file,
//! simple f32 kernels) -> Qwen3-0.6B decoder (the regular CUDA engine fed with
//! precomputed embeddings).

use crate::quant::f16_to_f32;
use crate::tokenizer::Tokenizer;
use crate::CudaModel;
use anyhow::{anyhow, bail, Context, Result};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::path::Path;
use std::sync::Arc;
use taraference_gguf::{GgmlType, GgufFile};

pub const SAMPLE_RATE: usize = 16_000;
const N_FFT: usize = 400;
const HOP: usize = 160;
const N_MELS: usize = 128;
const CHUNK_FRAMES: usize = 100;
const TOK_PER_CHUNK: usize = 13;
const WINDOW_TOK: usize = 104;

const ID_IM_END: u32 = 151645;
const ID_ENDOFTEXT: u32 = 151643;
const ID_ASR_TEXT: u32 = 151704;
const ID_AUDIO_PAD: u32 = 151676;

// ───────────────────────── mel frontend (WhisperFeatureExtractor) ─────────────────────────

fn hz_to_mel(f: f64) -> f64 {
    let min_log_hz = 1000.0;
    let min_log_mel = 15.0;
    let logstep = (6.4f64).ln() / 27.0;
    if f >= min_log_hz {
        min_log_mel + (f / min_log_hz).ln() / logstep
    } else {
        3.0 * f / 200.0
    }
}

fn mel_to_hz(m: f64) -> f64 {
    let min_log_hz = 1000.0;
    let min_log_mel = 15.0;
    let logstep = (6.4f64).ln() / 27.0;
    if m >= min_log_mel {
        min_log_hz * ((m - min_log_mel) * logstep).exp()
    } else {
        200.0 * m / 3.0
    }
}

fn mel_filters() -> Vec<f32> {
    let n_freq = N_FFT / 2 + 1;
    let fft_freqs: Vec<f64> = (0..n_freq).map(|i| i as f64 * 8000.0 / (n_freq - 1) as f64).collect();
    let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(8000.0));
    let pts: Vec<f64> = (0..N_MELS + 2)
        .map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (N_MELS + 1) as f64))
        .collect();
    let mut fb = vec![0f32; N_MELS * n_freq];
    for m in 0..N_MELS {
        let (f0, f1, f2) = (pts[m], pts[m + 1], pts[m + 2]);
        let enorm = 2.0 / (f2 - f0);
        for (k, &f) in fft_freqs.iter().enumerate() {
            let up = (f - f0) / (f1 - f0);
            let down = (f2 - f) / (f2 - f1);
            fb[m * n_freq + k] = (up.min(down).max(0.0) * enorm) as f32;
        }
    }
    fb
}

/// Log-mel features, laid out `[N_MELS][frames]` (mel-major). Returns (features, frames).
pub fn log_mel(samples: &[f32]) -> (Vec<f32>, usize) {
    let n = samples.len();
    let pad = N_FFT / 2;
    let reflect = |i: isize| -> f32 {
        let mut i = i;
        let last = n as isize - 1;
        if i < 0 {
            i = -i;
        }
        if i > last {
            i = 2 * last - i;
        }
        samples[i.clamp(0, last) as usize]
    };
    let padded: Vec<f32> = (0..n + 2 * pad).map(|i| reflect(i as isize - pad as isize)).collect();
    let frames = n / HOP; // stft frames minus the dropped last one
    let n_freq = N_FFT / 2 + 1;
    let window: Vec<f32> = (0..N_FFT)
        .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos()) as f32)
        .collect();
    let (cos_t, sin_t): (Vec<f32>, Vec<f32>) = (0..n_freq * N_FFT)
        .map(|idx| {
            let (k, t) = (idx / N_FFT, idx % N_FFT);
            let ang = 2.0 * std::f64::consts::PI * ((k * t) % N_FFT) as f64 / N_FFT as f64;
            (ang.cos() as f32, ang.sin() as f32)
        })
        .unzip();
    let fb = mel_filters();
    use rayon::prelude::*;
    let per_frame: Vec<Vec<f32>> = (0..frames)
        .into_par_iter()
        .map(|f| {
            let mut buf = [0f32; N_FFT];
            let mut power = vec![0f32; n_freq];
            for t in 0..N_FFT {
                buf[t] = padded[f * HOP + t] * window[t];
            }
            for k in 0..n_freq {
                let (mut re, mut im) = (0f32, 0f32);
                let (c, s) = (&cos_t[k * N_FFT..(k + 1) * N_FFT], &sin_t[k * N_FFT..(k + 1) * N_FFT]);
                for t in 0..N_FFT {
                    re += buf[t] * c[t];
                    im -= buf[t] * s[t];
                }
                power[k] = re * re + im * im;
            }
            (0..N_MELS)
                .map(|m| {
                    let row = &fb[m * n_freq..(m + 1) * n_freq];
                    row.iter().zip(&power).map(|(a, b)| a * b).sum::<f32>().max(1e-10).log10()
                })
                .collect()
        })
        .collect();
    let mut out = vec![0f32; N_MELS * frames];
    for (f, col) in per_frame.iter().enumerate() {
        for m in 0..N_MELS {
            out[m * frames + f] = col[m];
        }
    }
    let mx = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    for v in &mut out {
        *v = (v.max(mx - 8.0) + 4.0) / 4.0;
    }
    (out, frames)
}

fn conv_len(mut l: usize) -> usize {
    for _ in 0..3 {
        l = (l - 1) / 2 + 1;
    }
    l
}

/// Number of audio embedding tokens produced for `frames` mel frames.
pub fn audio_tokens(frames: usize) -> usize {
    let full = frames / CHUNK_FRAMES;
    let rem = frames % CHUNK_FRAMES;
    full * TOK_PER_CHUNK + if rem > 0 { conv_len(rem) } else { 0 }
}

// ───────────────────────── CUDA audio encoder ─────────────────────────

const KERNELS: &str = r#"
__device__ __forceinline__ float gelu_f(float x){ return 0.5f*x*(1.f+erff(x*0.70710678118f)); }

extern "C" __global__ void conv3x3s2_gelu(const float* in,const float* w,const float* b,float* out,
    int Cin,int H,int W,int Cout,int Ho,int Wo){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx>=Cout*Ho*Wo) return;
  int ox=idx%Wo; int oy=(idx/Wo)%Ho; int co=idx/(Wo*Ho);
  float acc=b[co];
  for(int ci=0;ci<Cin;ci++){
    for(int ky=0;ky<3;ky++){
      int iy=oy*2-1+ky; if(iy<0||iy>=H) continue;
      for(int kx=0;kx<3;kx++){
        int ix=ox*2-1+kx; if(ix<0||ix>=W) continue;
        acc+=in[(ci*H+iy)*W+ix]*w[((co*Cin+ci)*3+ky)*3+kx];
      }
    }
  }
  out[idx]=gelu_f(acc);
}

// out3 [C,F,Wo] -> feats rows [tok_off + t][c*F+f], t < vt
extern "C" __global__ void reorder(const float* in,float* feats,int CF,int Wo,int vt,int tok_off){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx>=vt*CF) return;
  int t=idx/CF; int cf=idx%CF;
  feats[(size_t)(tok_off+t)*CF+cf]=in[(size_t)cf*Wo+t];
}

#define TS 16
extern "C" __global__ void linear(const float* X,const float* W,const float* b,float* Y,int M,int N,int K,int act){
  __shared__ float xs[TS][TS+1];
  __shared__ float ws[TS][TS+1];
  int tx=threadIdx.x, ty=threadIdx.y;
  int row=blockIdx.y*TS+ty; int col=blockIdx.x*TS+tx;
  float acc=0.f;
  for(int k0=0;k0<K;k0+=TS){
    xs[ty][tx]= row<M ? X[(size_t)row*K+k0+tx] : 0.f;
    int wn=blockIdx.x*TS+ty;
    ws[ty][tx]= wn<N ? W[(size_t)wn*K+k0+tx] : 0.f;
    __syncthreads();
    #pragma unroll
    for(int kk=0;kk<TS;kk++) acc+=xs[ty][kk]*ws[tx][kk];
    __syncthreads();
  }
  if(row<M&&col<N){
    float v=acc+(b?b[col]:0.f);
    if(act==1) v=gelu_f(v);
    Y[(size_t)row*N+col]=v;
  }
}

extern "C" __global__ void layernorm(const float* x,const float* g,const float* b,float* y,int n,float eps){
  int r=blockIdx.x; const float* xr=x+(size_t)r*n; float* yr=y+(size_t)r*n;
  __shared__ float red[2][256];
  float s=0.f,s2=0.f;
  for(int i=threadIdx.x;i<n;i+=blockDim.x){ float v=xr[i]; s+=v; s2+=v*v; }
  red[0][threadIdx.x]=s; red[1][threadIdx.x]=s2; __syncthreads();
  for(int o=blockDim.x/2;o>0;o>>=1){
    if(threadIdx.x<o){ red[0][threadIdx.x]+=red[0][threadIdx.x+o]; red[1][threadIdx.x]+=red[1][threadIdx.x+o]; }
    __syncthreads();
  }
  float mean=red[0][0]/n; float var=red[1][0]/n-mean*mean;
  float inv=rsqrtf(var+eps);
  for(int i=threadIdx.x;i<n;i+=blockDim.x) yr[i]=(xr[i]-mean)*inv*g[i]+b[i];
}

extern "C" __global__ void add_inplace(float* dst,const float* src,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dst[i]+=src[i];
}

extern "C" __global__ void add_pos(float* x,const float* pos,int ntok,int d,int per){
  int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=ntok*d) return;
  int t=idx/d; int c=idx%d;
  x[idx]+=pos[(size_t)(t%per)*d+c];
}

extern "C" __global__ void attn_window(const float* q,const float* k,const float* v,float* o,
    int ntok,int win,int d,float scale){
  int h=blockIdx.x, i=blockIdx.y;
  int ws=(i/win)*win; int we=min(ws+win,ntok); int wl=we-ws;
  __shared__ float sc[128]; __shared__ float qs[64];
  int t=threadIdx.x;
  if(t<64) qs[t]=q[(size_t)i*d+h*64+t];
  __syncthreads();
  if(t<wl){
    const float* kp=k+(size_t)(ws+t)*d+h*64; float s=0.f;
    for(int j=0;j<64;j++) s+=qs[j]*kp[j];
    sc[t]=s*scale;
  }
  __syncthreads();
  if(t==0){
    float m=-1e30f; for(int j=0;j<wl;j++) m=fmaxf(m,sc[j]);
    float sum=0.f; for(int j=0;j<wl;j++){ sc[j]=expf(sc[j]-m); sum+=sc[j]; }
    for(int j=0;j<wl;j++) sc[j]/=sum;
  }
  __syncthreads();
  if(t<64){
    float a=0.f;
    for(int j=0;j<wl;j++) a+=sc[j]*v[(size_t)(ws+j)*d+h*64+t];
    o[(size_t)i*d+h*64+t]=a;
  }
}
"#;

fn tensor_f32(g: &GgufFile, name: &str) -> Result<Vec<f32>> {
    let info = g.tensor(name).ok_or_else(|| anyhow!("mmproj missing tensor {name}"))?;
    let data = g.tensor_data(info);
    let n = info.n_elements() as usize;
    Ok(match info.ggml_type {
        GgmlType::F32 => data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        GgmlType::F16 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        GgmlType::Q8_0 => {
            use rayon::prelude::*;
            let mut out = vec![0f32; n];
            out.par_chunks_mut(32).zip(data.par_chunks_exact(34)).for_each(|(o, blk)| {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for (dst, &q) in o.iter_mut().zip(&blk[2..34]) {
                    *dst = (q as i8) as f32 * d;
                }
            });
            out
        }
        other => bail!("{name}: unsupported type {}", other.name()),
    })
}

struct Linear {
    w: CudaSlice<f32>,
    b: Option<CudaSlice<f32>>,
    n: usize,
    k: usize,
}

struct EncLayer {
    ln1: (CudaSlice<f32>, CudaSlice<f32>),
    ln2: (CudaSlice<f32>, CudaSlice<f32>),
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    up: Linear,
    down: Linear,
}

pub struct AudioEncoder {
    stream: Arc<CudaStream>,
    f: std::collections::HashMap<&'static str, cudarc::driver::CudaFunction>,
    conv: [(CudaSlice<f32>, CudaSlice<f32>); 3],
    conv_out: Linear,
    pos: CudaSlice<f32>,
    layers: Vec<EncLayer>,
    post_ln: (CudaSlice<f32>, CudaSlice<f32>),
    proj1: Linear,
    proj2: Linear,
    d: usize,
    heads: usize,
    ff: usize,
    out_dim: usize,
}

impl AudioEncoder {
    pub fn load(mmproj: &Path) -> Result<Self> {
        let g = GgufFile::open(mmproj).with_context(|| format!("open {}", mmproj.display()))?;
        let arch = g.meta_str("general.architecture").unwrap_or("");
        if arch != "clip" || g.meta_str("clip.audio.projector_type") != Some("qwen3a") {
            bail!("{} is not a qwen3a audio mmproj", mmproj.display());
        }
        let meta = |k: &str| -> Result<usize> {
            g.meta_u64(&format!("clip.audio.{k}")).map(|v| v as usize).ok_or_else(|| anyhow!("mmproj missing clip.audio.{k}"))
        };
        let (d, heads, ff, n_layers, out_dim) = (
            meta("embedding_length")?,
            meta("attention.head_count")?,
            meta("feed_forward_length")?,
            meta("block_count")?,
            meta("projection_dim")?,
        );
        if d / heads != 64 {
            bail!("unsupported audio head_dim {}", d / heads);
        }
        let ctx = CudaContext::new(0).context("CudaContext")?;
        let stream = ctx.default_stream();
        let ptx = cudarc::nvrtc::compile_ptx(KERNELS).context("nvrtc compile (audio encoder)")?;
        let module = ctx.load_module(ptx)?;
        let mut f = std::collections::HashMap::new();
        for name in [
            "conv3x3s2_gelu",
            "reorder",
            "linear",
            "layernorm",
            "add_inplace",
            "add_pos",
            "attn_window",
        ] {
            f.insert(name, module.load_function(name)?);
        }
        let up = |name: &str| -> Result<CudaSlice<f32>> { Ok(stream.clone_htod(&tensor_f32(&g, name)?)?) };
        let lin = |name: &str, n: usize, k: usize, bias: bool| -> Result<Linear> {
            let w = up(&format!("{name}.weight"))?;
            let b = if bias { Some(up(&format!("{name}.bias"))?) } else { None };
            Ok(Linear { w, b, n, k })
        };
        let conv = [
            (up("a.conv2d.1.weight")?, up("a.conv2d.1.bias")?),
            (up("a.conv2d.2.weight")?, up("a.conv2d.2.bias")?),
            (up("a.conv2d.3.weight")?, up("a.conv2d.3.bias")?),
        ];
        let conv_out = lin("a.conv_out", d, 480 * 16, false)?;
        let pos = up("a.position_embd.weight")?;
        let mut layers = Vec::new();
        for i in 0..n_layers {
            let p = format!("a.blk.{i}");
            layers.push(EncLayer {
                ln1: (up(&format!("{p}.ln1.weight"))?, up(&format!("{p}.ln1.bias"))?),
                ln2: (up(&format!("{p}.ln2.weight"))?, up(&format!("{p}.ln2.bias"))?),
                q: lin(&format!("{p}.attn_q"), d, d, true)?,
                k: lin(&format!("{p}.attn_k"), d, d, true)?,
                v: lin(&format!("{p}.attn_v"), d, d, true)?,
                o: lin(&format!("{p}.attn_out"), d, d, true)?,
                up: lin(&format!("{p}.ffn_up"), ff, d, true)?,
                down: lin(&format!("{p}.ffn_down"), d, ff, true)?,
            });
        }
        let post_ln = (up("a.post_ln.weight")?, up("a.post_ln.bias")?);
        let proj1 = lin("mm.a.mlp.1", d, d, true)?;
        let proj2 = lin("mm.a.mlp.2", out_dim, d, true)?;
        Ok(Self { stream, f, conv, conv_out, pos, layers, post_ln, proj1, proj2, d, heads, ff, out_dim })
    }

    fn linear(&self, l: &Linear, x: &CudaSlice<f32>, y: &mut CudaSlice<f32>, m: usize, gelu: bool) -> Result<()> {
        let (mi, ni, ki, act) = (m as i32, l.n as i32, l.k as i32, gelu as i32);
        let null_bias;
        let bias = match &l.b {
            Some(b) => b,
            None => {
                null_bias = self.stream.alloc_zeros::<f32>(l.n)?;
                &null_bias
            }
        };
        unsafe {
            self.stream
                .launch_builder(&self.f["linear"])
                .arg(x)
                .arg(&l.w)
                .arg(bias)
                .arg(y)
                .arg(&mi)
                .arg(&ni)
                .arg(&ki)
                .arg(&act)
                .launch(LaunchConfig {
                    grid_dim: ((l.n as u32).div_ceil(16), (m as u32).div_ceil(16), 1),
                    block_dim: (16, 16, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    fn layernorm(&self, g: &(CudaSlice<f32>, CudaSlice<f32>), x: &CudaSlice<f32>, y: &mut CudaSlice<f32>, m: usize) -> Result<()> {
        let (n, eps) = (self.d as i32, 1e-5f32);
        unsafe {
            self.stream
                .launch_builder(&self.f["layernorm"])
                .arg(x)
                .arg(&g.0)
                .arg(&g.1)
                .arg(y)
                .arg(&n)
                .arg(&eps)
                .launch(LaunchConfig { grid_dim: (m as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })?;
        }
        Ok(())
    }

    fn add(&self, dst: &mut CudaSlice<f32>, src: &CudaSlice<f32>, n: usize) -> Result<()> {
        let ni = n as i32;
        unsafe {
            self.stream
                .launch_builder(&self.f["add_inplace"])
                .arg(dst)
                .arg(src)
                .arg(&ni)
                .launch(LaunchConfig::for_num_elems(n as u32))?;
        }
        Ok(())
    }

    /// Encode log-mel features (`[N_MELS][frames]`) to `[n_tok][1024]` decoder-space embeddings.
    pub fn encode(&self, mel: &[f32], frames: usize) -> Result<(Vec<f32>, usize)> {
        let (d, ff, heads, out_dim) = (self.d, self.ff, self.heads, self.out_dim);
        let n_tok = audio_tokens(frames);
        if n_tok == 0 {
            bail!("audio too short");
        }
        let s = &self.stream;
        let mut feats = s.alloc_zeros::<f32>(n_tok * 480 * 16)?;
        let n_chunks = frames.div_ceil(CHUNK_FRAMES);
        let mut tok_off = 0usize;
        for c in 0..n_chunks {
            let c0 = c * CHUNK_FRAMES;
            let valid = (frames - c0).min(CHUNK_FRAMES);
            let mut img = vec![0f32; N_MELS * CHUNK_FRAMES];
            for m in 0..N_MELS {
                img[m * CHUNK_FRAMES..m * CHUNK_FRAMES + valid]
                    .copy_from_slice(&mel[m * frames + c0..m * frames + c0 + valid]);
            }
            let mut cur = s.clone_htod(&img)?;
            let (mut h, mut w, mut cin) = (N_MELS, CHUNK_FRAMES, 1usize);
            for (wt, bs) in &self.conv {
                let (ho, wo) = ((h - 1) / 2 + 1, (w - 1) / 2 + 1);
                let mut out = s.alloc_zeros::<f32>(480 * ho * wo)?;
                let (cin_i, h_i, w_i, co_i, ho_i, wo_i) =
                    (cin as i32, h as i32, w as i32, 480i32, ho as i32, wo as i32);
                unsafe {
                    s.launch_builder(&self.f["conv3x3s2_gelu"])
                        .arg(&cur)
                        .arg(wt)
                        .arg(bs)
                        .arg(&mut out)
                        .arg(&cin_i)
                        .arg(&h_i)
                        .arg(&w_i)
                        .arg(&co_i)
                        .arg(&ho_i)
                        .arg(&wo_i)
                        .launch(LaunchConfig::for_num_elems((480 * ho * wo) as u32))?;
                }
                cur = out;
                (h, w, cin) = (ho, wo, 480);
            }
            debug_assert_eq!((h, w), (16, TOK_PER_CHUNK));
            let vt = conv_len(valid);
            let (cf, wo, vt_i, off_i) = ((480 * 16) as i32, w as i32, vt as i32, tok_off as i32);
            unsafe {
                s.launch_builder(&self.f["reorder"])
                    .arg(&cur)
                    .arg(&mut feats)
                    .arg(&cf)
                    .arg(&wo)
                    .arg(&vt_i)
                    .arg(&off_i)
                    .launch(LaunchConfig::for_num_elems((vt * 480 * 16) as u32))?;
            }
            tok_off += vt;
        }
        debug_assert_eq!(tok_off, n_tok);

        let mut x = s.alloc_zeros::<f32>(n_tok * d)?;
        self.linear(&self.conv_out, &feats, &mut x, n_tok, false)?;
        drop(feats);
        {
            let (nt, di, per) = (n_tok as i32, d as i32, TOK_PER_CHUNK as i32);
            unsafe {
                s.launch_builder(&self.f["add_pos"])
                    .arg(&mut x)
                    .arg(&self.pos)
                    .arg(&nt)
                    .arg(&di)
                    .arg(&per)
                    .launch(LaunchConfig::for_num_elems((n_tok * d) as u32))?;
            }
        }
        let mut xn = s.alloc_zeros::<f32>(n_tok * d)?;
        let mut q = s.alloc_zeros::<f32>(n_tok * d)?;
        let mut k = s.alloc_zeros::<f32>(n_tok * d)?;
        let mut v = s.alloc_zeros::<f32>(n_tok * d)?;
        let mut ao = s.alloc_zeros::<f32>(n_tok * d)?;
        let mut tmp = s.alloc_zeros::<f32>(n_tok * d)?;
        let mut ffn = s.alloc_zeros::<f32>(n_tok * ff)?;
        for l in &self.layers {
            self.layernorm(&l.ln1, &x, &mut xn, n_tok)?;
            self.linear(&l.q, &xn, &mut q, n_tok, false)?;
            self.linear(&l.k, &xn, &mut k, n_tok, false)?;
            self.linear(&l.v, &xn, &mut v, n_tok, false)?;
            let (nt, win, di, scale) = (n_tok as i32, WINDOW_TOK as i32, d as i32, 1.0f32 / 8.0);
            unsafe {
                s.launch_builder(&self.f["attn_window"])
                    .arg(&q)
                    .arg(&k)
                    .arg(&v)
                    .arg(&mut ao)
                    .arg(&nt)
                    .arg(&win)
                    .arg(&di)
                    .arg(&scale)
                    .launch(LaunchConfig {
                        grid_dim: (heads as u32, n_tok as u32, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
            self.linear(&l.o, &ao, &mut tmp, n_tok, false)?;
            self.add(&mut x, &tmp, n_tok * d)?;
            self.layernorm(&l.ln2, &x, &mut xn, n_tok)?;
            self.linear(&l.up, &xn, &mut ffn, n_tok, true)?;
            self.linear(&l.down, &ffn, &mut tmp, n_tok, false)?;
            self.add(&mut x, &tmp, n_tok * d)?;
        }
        self.layernorm(&self.post_ln, &x, &mut xn, n_tok)?;
        self.linear(&self.proj1, &xn, &mut tmp, n_tok, true)?;
        let mut out = s.alloc_zeros::<f32>(n_tok * out_dim)?;
        self.linear(&self.proj2, &tmp, &mut out, n_tok, false)?;
        s.synchronize()?;
        Ok((s.clone_dtoh(&out)?, n_tok))
    }
}

// ───────────────────────── end-to-end transcription ─────────────────────────

pub struct RajResult {
    pub language: String,
    pub text: String,
    pub audio_seconds: f32,
    pub audio_tokens: usize,
    pub text_tokens: usize,
    pub mel_ms: f64,
    pub encode_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
}

/// Models kept resident (encoder + decoder + tokenizer) so each request pays only compute.
pub struct RajSession {
    enc: AudioEncoder,
    dec: CudaModel,
    tok: Tokenizer,
    kv: crate::cuda::CudaKv,
    pub load_ms: f64,
}

const SESSION_MAX_SEQ: usize = 1536;

impl RajSession {
    pub fn load(model: &Path, mmproj: &Path) -> Result<Self> {
        let t = std::time::Instant::now();
        let enc = AudioEncoder::load(mmproj)?;
        let gguf = GgufFile::open(model).with_context(|| format!("open {}", model.display()))?;
        let tok = Tokenizer::from_gguf(&gguf)?;
        let mut dec = CudaModel::load(&gguf)?;
        dec.set_cuda_graph(true);
        let kv = dec.alloc_kv(SESSION_MAX_SEQ)?;
        Ok(Self { enc, dec, tok, kv, load_ms: t.elapsed().as_secs_f64() * 1e3 })
    }

    /// Transcribe 16 kHz mono samples.
    pub fn transcribe(&mut self, samples: &[f32], max_new: usize) -> Result<RajResult> {
        use std::time::Instant;
        if samples.len() < SAMPLE_RATE / 10 {
            bail!("audio shorter than 0.1 s");
        }
        let t = Instant::now();
        let (mel, frames) = log_mel(samples);
        let mel_ms = t.elapsed().as_secs_f64() * 1e3;

        let t = Instant::now();
        let (audio, n_audio) = self.enc.encode(&mel, frames)?;
        let encode_ms = t.elapsed().as_secs_f64() * 1e3;

        if n_audio + max_new + 64 > SESSION_MAX_SEQ {
            bail!("audio too long ({n_audio} tokens); max about {} s", (SESSION_MAX_SEQ - max_new - 64) / 13);
        }
        let (dec, tok) = (&mut self.dec, &self.tok);
        // One KV cache for the session: the decode CUDA graph is bound to its buffers.
        let kv = &mut self.kv;
        kv.len = 0;
        let prefix = tok.encode("<|im_start|>system
<|im_end|>
<|im_start|>user
<|audio_start|>", false);
        let suffix = tok.encode("<|audio_end|><|im_end|>
<|im_start|>assistant
", false);

        let t = Instant::now();
        dec.forward_greedy(&prefix, kv)?;
        dec.forward_greedy_embeds(&audio, kv)?;
        let mut next = dec.forward_greedy(&suffix, kv)?;
        let prefill_ms = t.elapsed().as_secs_f64() * 1e3;

        let t = Instant::now();
        let mut ids: Vec<u32> = Vec::new();
        while ids.len() < max_new {
            if next == ID_IM_END || next == ID_ENDOFTEXT {
                break;
            }
            ids.push(next);
            next = dec.forward_greedy(&[next], kv)?;
        }
        let decode_ms = t.elapsed().as_secs_f64() * 1e3;

        let (lang_ids, text_ids) = match ids.iter().position(|&i| i == ID_ASR_TEXT) {
            Some(p) => (&ids[..p], &ids[p + 1..]),
            None => (&ids[..0], &ids[..]),
        };
        let language = tok.decode(lang_ids).trim().trim_start_matches("language").trim().to_string();
        Ok(RajResult {
            language,
            text: tok.decode(text_ids).trim().to_string(),
            audio_seconds: samples.len() as f32 / SAMPLE_RATE as f32,
            audio_tokens: n_audio,
            text_tokens: text_ids.len(),
            mel_ms,
            encode_ms,
            prefill_ms,
            decode_ms,
        })
    }
}

/// One-shot helper: load, transcribe, drop.
pub fn transcribe(model: &Path, mmproj: &Path, samples: &[f32], max_new: usize) -> Result<RajResult> {
    RajSession::load(model, mmproj)?.transcribe(samples, max_new)
}
