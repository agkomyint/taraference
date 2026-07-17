// Qwen3.5 hybrid: Gated DeltaNet linear attention helpers + partial RoPE / Q-gate.

// ---------------------------------------------------------------------------
// Elementwise helpers
// ---------------------------------------------------------------------------

extern "C" __global__ void sigmoid_f32(float* __restrict__ x, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        float v = x[i];
        x[i] = 1.f / (1.f + expf(-v));
    }
}

// out = softplus(x + bias) * a   (a is already -exp(A_log) in GGUF)
// For n_tok==1: n == n_v, bias/a length n_v.
extern "C" __global__ void softplus_bias_scale_f32(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    const float* __restrict__ a,
    float* __restrict__ out,
    int n
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        float z = x[i] + bias[i];
        // softplus
        float sp = (z > 20.f) ? z : log1pf(expf(z));
        out[i] = sp * a[i];
    }
}

// Multi-token: x/out layout [n_tok, n_v], bias/a length n_v (broadcast).
extern "C" __global__ void softplus_bias_scale_rows_f32(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    const float* __restrict__ a,
    float* __restrict__ out,
    int n_v,
    int n_tok
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int n = n_v * n_tok;
    if (i < n) {
        int h = i % n_v;
        float z = x[i] + bias[h];
        float sp = (z > 20.f) ? z : log1pf(expf(z));
        out[i] = sp * a[h];
    }
}

// Device-to-device copy of n floats.
extern "C" __global__ void copy_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) dst[i] = src[i];
}

// g = exp(alpha) for decay in (0, 1]
extern "C" __global__ void exp_f32(float* __restrict__ x, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) x[i] = expf(x[i]);
}

// L2-normalize each head: x[t, h, d] layout [n_tok, n_heads, head_dim]
extern "C" __global__ void l2_norm_heads_f32(
    float* __restrict__ x,
    int n_heads,
    int head_dim,
    int n_tok,
    float eps
) {
    int h = (int)blockIdx.x;
    int t = (int)blockIdx.y;
    if (h >= n_heads || t >= n_tok) return;
    float* base = x + ((size_t)t * (size_t)n_heads + (size_t)h) * (size_t)head_dim;
    int tid = (int)threadIdx.x;
    __shared__ float buf[256];
    float ss = 0.f;
    for (int i = tid; i < head_dim; i += (int)blockDim.x) {
        float v = base[i];
        ss += v * v;
    }
    buf[tid] = ss;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] += buf[tid + s];
        __syncthreads();
    }
    float inv = rsqrtf(buf[0] + eps);
    for (int i = tid; i < head_dim; i += (int)blockDim.x) base[i] *= inv;
}

// Gated RMSNorm: out = rms_norm(x, w) * silu(gate)
// x, gate: [n_tok, n_heads, head_dim] (contiguous)
extern "C" __global__ void gated_rms_norm_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ gate,
    float* __restrict__ out,
    int n_heads,
    int head_dim,
    int n_tok,
    float eps
) {
    int h = (int)blockIdx.x;
    int t = (int)blockIdx.y;
    if (h >= n_heads || t >= n_tok) return;
    size_t off = ((size_t)t * (size_t)n_heads + (size_t)h) * (size_t)head_dim;
    const float* xb = x + off;
    const float* gb = gate + off;
    float* ob = out + off;
    int tid = (int)threadIdx.x;
    __shared__ float buf[256];
    float ss = 0.f;
    for (int i = tid; i < head_dim; i += (int)blockDim.x) {
        float v = xb[i];
        ss += v * v;
    }
    buf[tid] = ss;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] += buf[tid + s];
        __syncthreads();
    }
    float scale = rsqrtf(buf[0] / (float)head_dim + eps);
    for (int i = tid; i < head_dim; i += (int)blockDim.x) {
        float g = gb[i];
        float silu = g / (1.f + expf(-g));
        ob[i] = xb[i] * scale * w[i] * silu;
    }
}

// Interleaved Q|gate → compact Q and gate (per head: [q|g] → q, g)
// in: [n_tok, n_heads * 2 * head_dim]
// q_out: [n_tok, n_heads * head_dim]
// g_out: [n_tok, n_heads * head_dim]
//
// mode=0: per-head interleaved [Q|G][Q|G]... (HF / llama.cpp default)
// mode=1: contiguous halves [Q_all | G_all]
extern "C" __global__ void split_q_gate_interleaved_f32(
    const float* __restrict__ in,
    float* __restrict__ q_out,
    float* __restrict__ g_out,
    int n_heads,
    int head_dim,
    int n_tok,
    int mode
) {
    int h = (int)blockIdx.x;
    int t = (int)blockIdx.y;
    if (h >= n_heads || t >= n_tok) return;
    int tid = (int)threadIdx.x;
    size_t out_base = ((size_t)t * (size_t)n_heads + (size_t)h) * (size_t)head_dim;
    size_t row = (size_t)t * (size_t)n_heads * (size_t)(2 * head_dim);
    if (mode == 1) {
        size_t q_off = row + (size_t)h * (size_t)head_dim;
        size_t g_off = row + (size_t)n_heads * (size_t)head_dim + (size_t)h * (size_t)head_dim;
        for (int i = tid; i < head_dim; i += (int)blockDim.x) {
            q_out[out_base + i] = in[q_off + i];
            g_out[out_base + i] = in[g_off + i];
        }
    } else {
        size_t in_base = row + (size_t)h * (size_t)(2 * head_dim);
        for (int i = tid; i < head_dim; i += (int)blockDim.x) {
            q_out[out_base + i] = in[in_base + i];
            g_out[out_base + i] = in[in_base + head_dim + i];
        }
    }
}

// attn_out *= sigmoid(gate)  both [n_tok, n_heads * head_dim]
extern "C" __global__ void mul_sigmoid_f32(
    float* __restrict__ x,
    const float* __restrict__ gate,
    int n
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        float g = gate[i];
        x[i] *= 1.f / (1.f + expf(-g));
    }
}

// QK RMS-norm + partial RoPE (only first n_rot dims; neox pairs within n_rot)
extern "C" __global__ void qk_rms_norm_partial_rope_neox_f32(
    float* __restrict__ x,
    const float* __restrict__ w,
    int n_heads,
    int head_dim,
    int n_rot,
    int pos0,
    int n_tok,
    float theta,
    float eps
) {
    int h = (int)blockIdx.x;
    int t = (int)blockIdx.y;
    if (h >= n_heads || t >= n_tok) return;
    float* base = x + (size_t)t * (size_t)(n_heads * head_dim) + h * head_dim;
    int tid = (int)threadIdx.x;
    __shared__ float sums[256];
    float ss = 0.f;
    for (int i = tid; i < head_dim; i += (int)blockDim.x) {
        float v = base[i];
        ss += v * v;
    }
    sums[tid] = ss;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sums[tid] += sums[tid + s];
        __syncthreads();
    }
    float scale = rsqrtf(sums[0] / (float)head_dim + eps);
    // apply weight + scale
    for (int i = tid; i < head_dim; i += (int)blockDim.x) {
        base[i] = base[i] * scale * w[i];
    }
    __syncthreads();
    int pos = pos0 + t;
    int half = n_rot / 2;
    if (half < 1) return;
    for (int i = tid; i < half; i += (int)blockDim.x) {
        float freq = powf(theta, -2.f * (float)i / (float)n_rot);
        float ang = (float)pos * freq;
        float c = cosf(ang), s = sinf(ang);
        float x0 = base[i];
        float x1 = base[i + half];
        base[i] = x0 * c - x1 * s;
        base[i + half] = x0 * s + x1 * c;
    }
}

extern "C" __global__ void qk_rms_norm_partial_rope_neox_f32_d(
    float* __restrict__ x,
    const float* __restrict__ w,
    int n_heads,
    int head_dim,
    int n_rot,
    const int* __restrict__ pos0_ptr,
    int n_tok,
    float theta,
    float eps
) {
    int pos0 = pos0_ptr[0];
    int h = (int)blockIdx.x;
    int t = (int)blockIdx.y;
    if (h >= n_heads || t >= n_tok) return;
    float* base = x + (size_t)t * (size_t)(n_heads * head_dim) + h * head_dim;
    int tid = (int)threadIdx.x;
    __shared__ float sums[256];
    float ss = 0.f;
    for (int i = tid; i < head_dim; i += (int)blockDim.x) {
        float v = base[i];
        ss += v * v;
    }
    sums[tid] = ss;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sums[tid] += sums[tid + s];
        __syncthreads();
    }
    float scale = rsqrtf(sums[0] / (float)head_dim + eps);
    for (int i = tid; i < head_dim; i += (int)blockDim.x) {
        base[i] = base[i] * scale * w[i];
    }
    __syncthreads();
    int pos = pos0 + t;
    int half = n_rot / 2;
    if (half < 1) return;
    for (int i = tid; i < half; i += (int)blockDim.x) {
        float freq = powf(theta, -2.f * (float)i / (float)n_rot);
        float ang = (float)pos * freq;
        float c = cosf(ang), s = sinf(ang);
        float x0 = base[i];
        float x1 = base[i + half];
        base[i] = x0 * c - x1 * s;
        base[i + half] = x0 * s + x1 * c;
    }
}

// ---------------------------------------------------------------------------
// Fuse: beta = sigmoid(beta_raw); decay = exp(softplus(alpha_raw + dt) * a)
// n_v heads (decode and multi-token rows when n_tok==1 uses n = n_v).
// ---------------------------------------------------------------------------
extern "C" __global__ void gdn_prep_decay_beta_f32(
    float* __restrict__ alpha,   // in: raw alpha proj; out: exp(softplus)
    float* __restrict__ beta,    // in: raw beta proj; out: sigmoid
    const float* __restrict__ dt_bias,
    const float* __restrict__ a,  // already -exp(A_log)
    int n_v,
    int n_tok
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int n = n_v * n_tok;
    if (i >= n) return;
    int h = i % n_v;
    float b = beta[i];
    beta[i] = 1.f / (1.f + expf(-b));
    float z = alpha[i] + dt_bias[h];
    float sp = (z > 20.f) ? z : log1pf(expf(z));
    alpha[i] = expf(sp * a[h]);
}

// ---------------------------------------------------------------------------
// Causal depthwise conv1d (kernel K), with ring state of length K-1.
// qkv_mixed: [n_tok, channels] row-major
// weight (GGUF/ggml): dims [kernel, channels] with ne0=kernel contiguous
//   → weight(k, c) at index k + c * kernel  (matches llama.cpp ssm_conv)
// state: [(K-1) * channels] as state[k * channels + c] (oldest first)
// out: [n_tok, channels]
// ---------------------------------------------------------------------------
extern "C" __global__ void causal_conv1d_f32(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ state,
    float* __restrict__ out,
    int channels,
    int kernel,
    int n_tok
) {
    int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= channels) return;
    int km1 = kernel - 1;
    // load history into registers (K<=8)
    float hist[8];
    for (int k = 0; k < km1 && k < 8; k++) {
        hist[k] = state[(size_t)k * (size_t)channels + c];
    }
    for (int t = 0; t < n_tok; t++) {
        float acc = 0.f;
        // older history first; weight layout is ggml ne0-major (kernel fast)
        for (int k = 0; k < km1 && k < 8; k++) {
            acc += weight[(size_t)k + (size_t)c * (size_t)kernel] * hist[k];
        }
        float xt = x[(size_t)t * (size_t)channels + c];
        acc += weight[(size_t)(kernel - 1) + (size_t)c * (size_t)kernel] * xt;
        // SiLU
        out[(size_t)t * (size_t)channels + c] = acc / (1.f + expf(-acc));
        // shift history
        for (int k = 0; k < km1 - 1 && k < 7; k++) hist[k] = hist[k + 1];
        if (km1 > 0) hist[km1 - 1] = xt;
    }
    // write back state
    for (int k = 0; k < km1 && k < 8; k++) {
        state[(size_t)k * (size_t)channels + c] = hist[k];
    }
}

// Decode-optimized: n_tok==1 causal conv + SiLU (same math as causal_conv1d_f32).
extern "C" __global__ void causal_conv1d_one_f32(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ state,
    float* __restrict__ out,
    int channels,
    int kernel
) {
    int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= channels) return;
    int km1 = kernel - 1;
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < 3; k++) {
        if (k < km1) {
            acc += weight[(size_t)k + (size_t)c * (size_t)kernel]
                 * state[(size_t)k * (size_t)channels + c];
        }
    }
    float xt = x[c];
    acc += weight[(size_t)(kernel - 1) + (size_t)c * (size_t)kernel] * xt;
    out[c] = acc / (1.f + expf(-acc));
    // shift ring: drop oldest, append xt
    for (int k = 0; k < km1 - 1; k++) {
        state[(size_t)k * (size_t)channels + c] =
            state[(size_t)(k + 1) * (size_t)channels + c];
    }
    if (km1 > 0) {
        state[(size_t)(km1 - 1) * (size_t)channels + c] = xt;
    }
}

// ---------------------------------------------------------------------------
// Gated DeltaNet recurrent step over a sequence (token-serial, head-parallel).
//
// q: [n_tok, n_k_heads, d_k]  — will be repeated to n_v_heads if needed
// k: same
// v: [n_tok, n_v_heads, d_v]
// decay g: [n_tok, n_v_heads]  already exp(α) in (0,1]
// beta:    [n_tok, n_v_heads]  sigmoid
// state S: [n_v_heads, d_k, d_v]  updated in place
// out:     [n_tok, n_v_heads, d_v]
//
// When n_k_heads != n_v_heads, k/q heads are repeated (GQA-style).
// ---------------------------------------------------------------------------
extern "C" __global__ void gated_delta_rule_seq_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ decay,
    const float* __restrict__ beta,
    float* __restrict__ state,
    float* __restrict__ out,
    int n_k_heads,
    int n_v_heads,
    int d_k,
    int d_v,
    int n_tok
) {
    int h = (int)blockIdx.x;
    if (h >= n_v_heads) return;
    int tid = (int)threadIdx.x;
    // GQA head map: match llama.cpp ggml_gated_delta_net / ggml_repeat
    // tiling: v-head h → k-head (h % n_k). When n_v == n_k this is identity.
    // (HF training uses repeat_interleave; GGUF+llama path uses tile/modulo.)
    int hk = (n_k_heads > 0) ? (h % n_k_heads) : 0;

    // state for this head: d_k x d_v, row-major
    float* S = state + (size_t)h * (size_t)d_k * (size_t)d_v;

    for (int t = 0; t < n_tok; t++) {
        float g = decay[(size_t)t * (size_t)n_v_heads + h];
        float b = beta[(size_t)t * (size_t)n_v_heads + h];
        const float* qt = q + ((size_t)t * (size_t)n_k_heads + (size_t)hk) * (size_t)d_k;
        const float* kt = k + ((size_t)t * (size_t)n_k_heads + (size_t)hk) * (size_t)d_k;
        const float* vt = v + ((size_t)t * (size_t)n_v_heads + (size_t)h) * (size_t)d_v;
        float* ot = out + ((size_t)t * (size_t)n_v_heads + (size_t)h) * (size_t)d_v;

        // 1) S *= g
        for (int i = tid; i < d_k * d_v; i += (int)blockDim.x) {
            S[i] *= g;
        }
        __syncthreads();

        // 2) kv_mem[j] = sum_i S[i,j] * k[i]
        // 3) delta[j] = (v[j] - kv_mem[j]) * beta
        // 4) S[i,j] += k[i] * delta[j]
        // 5) o[j] = sum_i S[i,j] * q[i]
        // Parallelize over d_v columns.
        for (int j = tid; j < d_v; j += (int)blockDim.x) {
            float kv_mem = 0.f;
            for (int i = 0; i < d_k; i++) {
                kv_mem += S[(size_t)i * (size_t)d_v + j] * kt[i];
            }
            float delta = (vt[j] - kv_mem) * b;
            for (int i = 0; i < d_k; i++) {
                S[(size_t)i * (size_t)d_v + j] += kt[i] * delta;
            }
            float o = 0.f;
            float qscale = rsqrtf((float)d_k);
            for (int i = 0; i < d_k; i++) {
                o += S[(size_t)i * (size_t)d_v + j] * (qt[i] * qscale);
            }
            ot[j] = o;
        }
        __syncthreads();
    }
}

// Fast decode path: n_tok==1, d_k==d_v (Qwen3.5). Shared k/q, unrolled S.
// grid: n_v_heads, block: 128 (covers d_v=128 with 1 column per thread).
extern "C" __global__ void gated_delta_rule_one_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ decay,
    const float* __restrict__ beta,
    float* __restrict__ state,
    float* __restrict__ out,
    int n_k_heads,
    int n_v_heads,
    int d_state
) {
    int h = (int)blockIdx.x;
    if (h >= n_v_heads) return;
    int tid = (int)threadIdx.x;
    // Match llama.cpp fused GDN: h_k = head % H_k (tile / modulo GQA).
    int hk = (n_k_heads > 0) ? (h % n_k_heads) : 0;
    float* S = state + (size_t)h * (size_t)d_state * (size_t)d_state;
    const float* qt = q + (size_t)hk * (size_t)d_state;
    const float* kt = k + (size_t)hk * (size_t)d_state;
    const float* vt = v + (size_t)h * (size_t)d_state;
    float* ot = out + (size_t)h * (size_t)d_state;

    __shared__ float sk[256];
    __shared__ float sq[256];
    float scale = rsqrtf((float)d_state);
    for (int i = tid; i < d_state; i += (int)blockDim.x) {
        sk[i] = kt[i];
        sq[i] = qt[i] * scale;
    }
    float g = decay[h];
    float b = beta[h];
    __syncthreads();

    int n_s = d_state * d_state;
    for (int i = tid; i < n_s; i += (int)blockDim.x) {
        S[i] *= g;
    }
    __syncthreads();

    if (tid < d_state) {
        int j = tid;
        float kv_mem = 0.f;
        #pragma unroll 8
        for (int i = 0; i < d_state; i++) {
            kv_mem += S[(size_t)i * (size_t)d_state + j] * sk[i];
        }
        float delta = (vt[j] - kv_mem) * b;
        float o = 0.f;
        #pragma unroll 8
        for (int i = 0; i < d_state; i++) {
            float s = S[(size_t)i * (size_t)d_state + j] + sk[i] * delta;
            S[(size_t)i * (size_t)d_state + j] = s;
            o += s * sq[i];
        }
        ot[j] = o;
    }
}

// ---------------------------------------------------------------------------
// Fused decode mixer (after QKV/gate/α/β projections):
//   causal conv1d + SiLU → split Q/K/V → L2(Q), L2(K)
// grid: n_k_heads, block: 128  (Qwen3.5 d_state=128)
// V channels are partitioned across k-head blocks (no single-block bottleneck).
// ---------------------------------------------------------------------------
__device__ __forceinline__ float gdn_conv1d_channel_one(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ state,
    int c,
    int channels,
    int kernel
) {
    int km1 = kernel - 1;
    float acc = 0.f;
    // Qwen3.5 uses kernel=4 → km1=3; fully unrolled.
    #pragma unroll
    for (int k = 0; k < 3; k++) {
        if (k < km1) {
            acc += weight[(size_t)k + (size_t)c * (size_t)kernel]
                 * state[(size_t)k * (size_t)channels + c];
        }
    }
    float xt = x[c];
    acc += weight[(size_t)(kernel - 1) + (size_t)c * (size_t)kernel] * xt;
    float y = acc / (1.f + expf(-acc));
    for (int k = 0; k < km1 - 1; k++) {
        state[(size_t)k * (size_t)channels + c] =
            state[(size_t)(k + 1) * (size_t)channels + c];
    }
    if (km1 > 0) {
        state[(size_t)(km1 - 1) * (size_t)channels + c] = xt;
    }
    return y;
}

extern "C" __global__ void gdn_conv_qkvl2_one_f32(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ state,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    int n_k_heads,
    int n_v_heads,
    int d_state,
    int kernel,
    float eps
) {
    int h = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    if (h >= n_k_heads) return;

    int key_dim = n_k_heads * d_state;
    int val_dim = n_v_heads * d_state;
    int channels = key_dim * 2 + val_dim;

    // --- Q / K for this k-head (one channel per thread when blockDim >= d_state) ---
    float qv = 0.f, kv = 0.f;
    if (tid < d_state) {
        int cq = h * d_state + tid;
        int ck = key_dim + h * d_state + tid;
        qv = gdn_conv1d_channel_one(x, weight, state, cq, channels, kernel);
        kv = gdn_conv1d_channel_one(x, weight, state, ck, channels, kernel);
    }

    // --- V: partition val_dim across k-head blocks ---
    int v_chunk = (val_dim + n_k_heads - 1) / n_k_heads;
    int v0 = h * v_chunk;
    int v1 = v0 + v_chunk;
    if (v1 > val_dim) v1 = val_dim;
    for (int i = v0 + tid; i < v1; i += (int)blockDim.x) {
        int cv = 2 * key_dim + i;
        v_out[i] = gdn_conv1d_channel_one(x, weight, state, cv, channels, kernel);
    }

    // --- L2 normalize Q and K ---
    __shared__ float red[256];
    red[tid] = (tid < d_state) ? (qv * qv) : 0.f;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float invq = rsqrtf(red[0] + eps);
    __syncthreads();
    red[tid] = (tid < d_state) ? (kv * kv) : 0.f;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float invk = rsqrtf(red[0] + eps);
    if (tid < d_state) {
        q_out[(size_t)h * (size_t)d_state + tid] = qv * invq;
        k_out[(size_t)h * (size_t)d_state + tid] = kv * invk;
    }
}

// ---------------------------------------------------------------------------
// Specialized decode for d_state == 128 (Qwen3.5): fully unrolled column update.
// grid: n_v_heads, block: 128. Same math as gdn_delta_gated_one_f32.
// ---------------------------------------------------------------------------
extern "C" __global__ void gdn_delta_gated_one_d128_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ alpha_raw,
    const float* __restrict__ beta_raw,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a,
    float* __restrict__ state,
    const float* __restrict__ norm_w,
    const float* __restrict__ gate_z,
    float* __restrict__ out,
    int n_k_heads,
    int n_v_heads,
    float eps
) {
    const int d_state = 128;
    int h = (int)blockIdx.x;
    if (h >= n_v_heads) return;
    int tid = (int)threadIdx.x;
    int hk = (n_k_heads > 0) ? (h % n_k_heads) : 0;

    __shared__ float sg, sb;
    if (tid == 0) {
        float br = beta_raw[h];
        sb = 1.f / (1.f + expf(-br));
        float z = alpha_raw[h] + dt_bias[h];
        float sp = (z > 20.f) ? z : log1pf(expf(z));
        sg = expf(sp * a[h]);
    }
    __syncthreads();
    float g = sg;
    float b = sb;

    float* S = state + (size_t)h * 128u * 128u;
    const float* qt = q + (size_t)hk * 128u;
    const float* kt = k + (size_t)hk * 128u;
    const float* vt = v + (size_t)h * 128u;
    const float* gz = gate_z + (size_t)h * 128u;
    float* ot = out + (size_t)h * 128u;

    __shared__ float sk[128];
    __shared__ float sq[128];
    __shared__ float so[128];
    const float qscale = 0.08838834764f; // 1/sqrt(128)
    if (tid < 128) {
        sk[tid] = kt[tid];
        sq[tid] = qt[tid] * qscale;
    }
    __syncthreads();

    // Scale state by decay: vectorized float4 when aligned.
    {
        float4* S4 = reinterpret_cast<float4*>(S);
        for (int i = tid; i < (128 * 128) / 4; i += (int)blockDim.x) {
            float4 s = S4[i];
            s.x *= g; s.y *= g; s.z *= g; s.w *= g;
            S4[i] = s;
        }
    }
    __syncthreads();

    float o = 0.f;
    if (tid < 128) {
        int j = tid;
        float kv_mem = 0.f;
        #pragma unroll
        for (int i = 0; i < 128; i++) {
            kv_mem += S[(size_t)i * 128u + (size_t)j] * sk[i];
        }
        float delta = (vt[j] - kv_mem) * b;
        #pragma unroll
        for (int i = 0; i < 128; i++) {
            float s = S[(size_t)i * 128u + (size_t)j] + sk[i] * delta;
            S[(size_t)i * 128u + (size_t)j] = s;
            o += s * sq[i];
        }
        so[tid] = o;
    }
    __syncthreads();

    // Gated RMS
    float ss = 0.f;
    if (tid < 128) {
        float v = so[tid];
        ss = v * v;
    }
    __shared__ float red[128];
    red[tid] = ss;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float scale = rsqrtf(red[0] * (1.f / 128.f) + eps);
    if (tid < 128) {
        float gate = gz[tid];
        float silu = gate / (1.f + expf(-gate));
        ot[tid] = so[tid] * scale * norm_w[tid] * silu;
    }
}

// ---------------------------------------------------------------------------
// Fused decode: prep(decay,beta) + gated delta rule + gated RMSNorm
// grid: n_v_heads, block: 128
// alpha/beta are RAW projections; dt_bias/a applied here (skips separate prep).
// out layout: [n_v_heads, d_state]  (feeds ssm_out projection)
// ---------------------------------------------------------------------------
extern "C" __global__ void gdn_delta_gated_one_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ alpha_raw, // [n_v] raw projection
    const float* __restrict__ beta_raw,  // [n_v] raw projection
    const float* __restrict__ dt_bias,
    const float* __restrict__ a,
    float* __restrict__ state,
    const float* __restrict__ norm_w,
    const float* __restrict__ gate_z,
    float* __restrict__ out,
    int n_k_heads,
    int n_v_heads,
    int d_state,
    float eps
) {
    int h = (int)blockIdx.x;
    if (h >= n_v_heads) return;
    int tid = (int)threadIdx.x;
    int hk = (n_k_heads > 0) ? (h % n_k_heads) : 0;

    // prep decay / beta for this head (thread 0 broadcasts via shared)
    __shared__ float sg;
    __shared__ float sb;
    if (tid == 0) {
        float br = beta_raw[h];
        sb = 1.f / (1.f + expf(-br));
        float z = alpha_raw[h] + dt_bias[h];
        float sp = (z > 20.f) ? z : log1pf(expf(z));
        sg = expf(sp * a[h]);
    }
    __syncthreads();
    float g = sg;
    float b = sb;

    float* S = state + (size_t)h * (size_t)d_state * (size_t)d_state;
    const float* qt = q + (size_t)hk * (size_t)d_state;
    const float* kt = k + (size_t)hk * (size_t)d_state;
    const float* vt = v + (size_t)h * (size_t)d_state;
    const float* gz = gate_z + (size_t)h * (size_t)d_state;
    float* ot = out + (size_t)h * (size_t)d_state;

    __shared__ float sk[256];
    __shared__ float sq[256];
    __shared__ float so[256];
    float qscale = rsqrtf((float)d_state);
    for (int i = tid; i < d_state; i += (int)blockDim.x) {
        sk[i] = kt[i];
        sq[i] = qt[i] * qscale;
    }
    __syncthreads();

    int n_s = d_state * d_state;
    for (int i = tid; i < n_s; i += (int)blockDim.x) {
        S[i] *= g;
    }
    __syncthreads();

    float o = 0.f;
    if (tid < d_state) {
        int j = tid;
        float kv_mem = 0.f;
        #pragma unroll 8
        for (int i = 0; i < d_state; i++) {
            kv_mem += S[(size_t)i * (size_t)d_state + j] * sk[i];
        }
        float delta = (vt[j] - kv_mem) * b;
        #pragma unroll 8
        for (int i = 0; i < d_state; i++) {
            float s = S[(size_t)i * (size_t)d_state + j] + sk[i] * delta;
            S[(size_t)i * (size_t)d_state + j] = s;
            o += s * sq[i];
        }
        so[tid] = o;
    } else {
        so[tid] = 0.f;
    }
    __syncthreads();

    // Gated RMSNorm over head dims
    float ss = 0.f;
    for (int i = tid; i < d_state; i += (int)blockDim.x) {
        float v = so[i];
        ss += v * v;
    }
    __shared__ float red[256];
    red[tid] = ss;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float scale = rsqrtf(red[0] / (float)d_state + eps);
    if (tid < d_state) {
        float gate = gz[tid];
        float silu = gate / (1.f + expf(-gate));
        ot[tid] = so[tid] * scale * norm_w[tid] * silu;
    }
}

// ---------------------------------------------------------------------------
// Prefill: fused conv + split + L2(Q/K) for multi-token.
// grid: (channels+255)/256 for conv is separate — this fuses split+L2 only is weak.
// Instead: head-parallel over (n_k_heads, n_tok) after conv buffer exists is already
// two launches. Here we fuse split+L2 into one kernel (was split + 2× l2_norm).
// grid: (n_k_heads, n_tok), block: 128
// ---------------------------------------------------------------------------
extern "C" __global__ void gdn_split_l2_seq_f32(
    const float* __restrict__ conv,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    int n_k_heads,
    int n_v_heads,
    int d_k,
    int d_v,
    int n_tok,
    float eps
) {
    int h = (int)blockIdx.x;
    int t = (int)blockIdx.y;
    if (h >= n_k_heads || t >= n_tok) return;
    int tid = (int)threadIdx.x;
    int key_dim = n_k_heads * d_k;
    int val_dim = n_v_heads * d_v;
    int qkv_dim = key_dim * 2 + val_dim;
    const float* row = conv + (size_t)t * (size_t)qkv_dim;

    // Head 0 of each token also copies V once.
    if (h == 0) {
        float* vb = v_out + (size_t)t * (size_t)val_dim;
        for (int i = tid; i < val_dim; i += (int)blockDim.x) {
            vb[i] = row[2 * key_dim + i];
        }
    }

    float* qb = q_out + ((size_t)t * (size_t)n_k_heads + (size_t)h) * (size_t)d_k;
    float* kb = k_out + ((size_t)t * (size_t)n_k_heads + (size_t)h) * (size_t)d_k;
    const float* cq = row + (size_t)h * (size_t)d_k;
    const float* ck = row + (size_t)key_dim + (size_t)h * (size_t)d_k;

    __shared__ float red[128];
    float sq = 0.f, skv = 0.f;
    for (int i = tid; i < d_k; i += (int)blockDim.x) {
        float qv = cq[i];
        float kv = ck[i];
        qb[i] = qv;
        kb[i] = kv;
        sq += qv * qv;
        skv += kv * kv;
    }
    red[tid] = sq;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float invq = rsqrtf(red[0] + eps);
    __syncthreads();
    red[tid] = skv;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float invk = rsqrtf(red[0] + eps);
    for (int i = tid; i < d_k; i += (int)blockDim.x) {
        qb[i] *= invq;
        kb[i] *= invk;
    }
}

// Prefill: gated delta (token-serial) + gated RMSNorm fused.
// grid: n_v_heads, block: 128. d_k == d_v (Qwen3.5).
extern "C" __global__ void gdn_delta_gated_seq_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ decay,
    const float* __restrict__ beta,
    float* __restrict__ state,
    const float* __restrict__ norm_w,
    const float* __restrict__ gate_z,
    float* __restrict__ out,
    int n_k_heads,
    int n_v_heads,
    int d_state,
    int n_tok,
    float eps
) {
    int h = (int)blockIdx.x;
    if (h >= n_v_heads) return;
    int tid = (int)threadIdx.x;
    int hk = (n_k_heads > 0) ? (h % n_k_heads) : 0;
    float* S = state + (size_t)h * (size_t)d_state * (size_t)d_state;

    __shared__ float sk[256];
    __shared__ float sq[256];
    __shared__ float so[256];
    __shared__ float red[256];
    float qscale = rsqrtf((float)d_state);

    for (int t = 0; t < n_tok; t++) {
        float g = decay[(size_t)t * (size_t)n_v_heads + h];
        float b = beta[(size_t)t * (size_t)n_v_heads + h];
        const float* qt = q + ((size_t)t * (size_t)n_k_heads + (size_t)hk) * (size_t)d_state;
        const float* kt = k + ((size_t)t * (size_t)n_k_heads + (size_t)hk) * (size_t)d_state;
        const float* vt = v + ((size_t)t * (size_t)n_v_heads + (size_t)h) * (size_t)d_state;
        const float* gz = gate_z + ((size_t)t * (size_t)n_v_heads + (size_t)h) * (size_t)d_state;
        float* ot = out + ((size_t)t * (size_t)n_v_heads + (size_t)h) * (size_t)d_state;

        for (int i = tid; i < d_state; i += (int)blockDim.x) {
            sk[i] = kt[i];
            sq[i] = qt[i] * qscale;
        }
        __syncthreads();

        int n_s = d_state * d_state;
        for (int i = tid; i < n_s; i += (int)blockDim.x) {
            S[i] *= g;
        }
        __syncthreads();

        if (tid < d_state) {
            int j = tid;
            float kv_mem = 0.f;
            #pragma unroll 8
            for (int i = 0; i < d_state; i++) {
                kv_mem += S[(size_t)i * (size_t)d_state + j] * sk[i];
            }
            float delta = (vt[j] - kv_mem) * b;
            float o = 0.f;
            #pragma unroll 8
            for (int i = 0; i < d_state; i++) {
                float s = S[(size_t)i * (size_t)d_state + j] + sk[i] * delta;
                S[(size_t)i * (size_t)d_state + j] = s;
                o += s * sq[i];
            }
            so[tid] = o;
        } else {
            so[tid] = 0.f;
        }
        __syncthreads();

        float ss = 0.f;
        for (int i = tid; i < d_state; i += (int)blockDim.x) {
            float vv = so[i];
            ss += vv * vv;
        }
        red[tid] = ss;
        __syncthreads();
        for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        float scale = rsqrtf(red[0] / (float)d_state + eps);
        if (tid < d_state) {
            float gate = gz[tid];
            float silu = gate / (1.f + expf(-gate));
            ot[tid] = so[tid] * scale * norm_w[tid] * silu;
        }
        __syncthreads();
    }
}

// Split conv + L2-normalize Q and K for n_tok==1.
// grid.x = n_k_heads (each block owns one head for Q/K L2; V copied by all).
extern "C" __global__ void split_qkv_l2_one_f32(
    const float* __restrict__ conv,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    int n_k_heads,
    int n_v_heads,
    int d_k,
    int d_v,
    float eps
) {
    int h = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int key_dim = n_k_heads * d_k;
    int val_dim = n_v_heads * d_v;

    // Head 0 also scatters V (once).
    if (h == 0) {
        for (int i = tid; i < val_dim; i += (int)blockDim.x) {
            v_out[i] = conv[2 * key_dim + i];
        }
    }
    if (h >= n_k_heads) return;

    // Copy this head's Q and K from conv, then L2-normalize.
    float* qb = q_out + (size_t)h * (size_t)d_k;
    float* kb = k_out + (size_t)h * (size_t)d_k;
    const float* cq = conv + (size_t)h * (size_t)d_k;
    const float* ck = conv + (size_t)key_dim + (size_t)h * (size_t)d_k;
    __shared__ float ss[128];
    float sq = 0.f, sk = 0.f;
    for (int i = tid; i < d_k; i += (int)blockDim.x) {
        float qv = cq[i];
        float kv = ck[i];
        qb[i] = qv;
        kb[i] = kv;
        sq += qv * qv;
        sk += kv * kv;
    }
    ss[tid] = sq;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) ss[tid] += ss[tid + s];
        __syncthreads();
    }
    float invq = rsqrtf(ss[0] + eps);
    __syncthreads();
    ss[tid] = sk;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) ss[tid] += ss[tid + s];
        __syncthreads();
    }
    float invk = rsqrtf(ss[0] + eps);
    for (int i = tid; i < d_k; i += (int)blockDim.x) {
        qb[i] *= invq;
        kb[i] *= invk;
    }
}

// Zero a buffer (for session reset of recurrent state).
extern "C" __global__ void zero_f32(float* __restrict__ x, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) x[i] = 0.f;
}

// Extract Q/K/V views from conv output [n_tok, key*2+value]
// q_out [n_tok, n_k, dk], k_out same, v_out [n_tok, n_v, dv]
extern "C" __global__ void split_qkv_from_conv_f32(
    const float* __restrict__ conv,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    int n_k_heads,
    int n_v_heads,
    int d_k,
    int d_v,
    int n_tok
) {
    int t = (int)blockIdx.x;
    if (t >= n_tok) return;
    int tid = (int)threadIdx.x;
    int key_dim = n_k_heads * d_k;
    int qkv_dim = key_dim * 2 + n_v_heads * d_v;
    const float* row = conv + (size_t)t * (size_t)qkv_dim;
    float* q = q_out + (size_t)t * (size_t)key_dim;
    float* k = k_out + (size_t)t * (size_t)key_dim;
    float* v = v_out + (size_t)t * (size_t)(n_v_heads * d_v);
    for (int i = tid; i < key_dim; i += (int)blockDim.x) {
        q[i] = row[i];
        k[i] = row[key_dim + i];
    }
    for (int i = tid; i < n_v_heads * d_v; i += (int)blockDim.x) {
        v[i] = row[2 * key_dim + i];
    }
}
