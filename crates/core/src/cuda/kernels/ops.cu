// Batch RMSNorm: one block per token
extern "C" __global__ void rms_norm_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out,
    int n,
    int n_tok,
    float eps
) {
    int t = (int)blockIdx.x;
    if (t >= n_tok) return;
    const float* xt = x + (size_t)t * (size_t)n;
    float* ot = out + (size_t)t * (size_t)n;
    __shared__ float buf[256];
    float local = 0.f;
    for (int i = (int)threadIdx.x; i < n; i += (int)blockDim.x) {
        float v = xt[i];
        local += v * v;
    }
    buf[threadIdx.x] = local;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    __shared__ float scale;
    if (threadIdx.x == 0) scale = rsqrtf(buf[0] / (float)n + eps);
    __syncthreads();
    float s = scale;
    for (int i = (int)threadIdx.x; i < n; i += (int)blockDim.x)
        ot[i] = xt[i] * s * w[i];
}

extern "C" __global__ void silu_mul_f32(
    float* __restrict__ gate,
    const float* __restrict__ up,
    int n
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        float v = gate[i];
        gate[i] = (v * (1.f / (1.f + expf(-v)))) * up[i];
    }
}

extern "C" __global__ void add_f32(
    float* __restrict__ a, const float* __restrict__ b, int n
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) a[i] += b[i];
}

extern "C" __global__ void add_bias_f32(
    float* __restrict__ x, const float* __restrict__ b, int n_feat, int n_tok
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = n_feat * n_tok;
    if (i < total) x[i] += b[i % n_feat];
}

// RoPE neox batch: grid (n_heads, n_tok)
extern "C" __global__ void rope_neox_f32(
    float* __restrict__ x,
    int n_heads,
    int head_dim,
    int pos0,
    int n_tok,
    float theta
) {
    int h = (int)blockIdx.x;
    int t = (int)blockIdx.y;
    if (h >= n_heads || t >= n_tok) return;
    int half = head_dim / 2;
    int pos = pos0 + t;
    float* base = x + (size_t)t * (size_t)(n_heads * head_dim) + h * head_dim;
    for (int i = (int)threadIdx.x; i < half; i += (int)blockDim.x) {
        float freq = powf(theta, -2.f * (float)i / (float)head_dim);
        float ang = (float)pos * freq;
        float c = cosf(ang), s = sinf(ang);
        float x0 = base[i];
        float x1 = base[i + half];
        base[i] = x0 * c - x1 * s;
        base[i + half] = x0 * s + x1 * c;
    }
}

// Copy n_tok KV rows into cache starting at pos0
// src: [n_tok, stride], cache: [max_seq, stride]
extern "C" __global__ void copy_kv_f32(
    const float* __restrict__ src,
    float* __restrict__ cache,
    int pos0,
    int n_tok,
    int stride
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = n_tok * stride;
    if (i < total) {
        int t = i / stride;
        int d = i % stride;
        cache[(size_t)(pos0 + t) * (size_t)stride + d] = src[(size_t)t * (size_t)stride + d];
    }
}

// Prefill+decode attention: grid (n_head, n_q)
// q: [n_q, n_head, hd], cache K/V: [max_seq, n_kv, hd]
// queries at absolute positions pos0 .. pos0+n_q-1
extern "C" __global__ void attn_f32(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ out,
    int n_head,
    int n_kv,
    int head_dim,
    int pos0,
    int n_q,
    float scale
) {
    int h = (int)blockIdx.x;
    int qi = (int)blockIdx.y;
    if (h >= n_head || qi >= n_q) return;
    int rep = n_head / n_kv;
    int kv_h = h / rep;
    int stride = n_kv * head_dim;
    int pos = pos0 + qi;
    int seq_len = pos + 1;
    const float* qh = q + (size_t)qi * (size_t)(n_head * head_dim) + h * head_dim;
    extern __shared__ float scores[];

    for (int t = (int)threadIdx.x; t < seq_len; t += (int)blockDim.x) {
        const float* kt = k_cache + (size_t)t * (size_t)stride + kv_h * head_dim;
        float dot = 0.f;
        for (int d = 0; d < head_dim; d++) dot += qh[d] * kt[d];
        scores[t] = dot * scale;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float m = -1e30f;
        for (int t = 0; t < seq_len; t++) if (scores[t] > m) m = scores[t];
        float sum = 0.f;
        for (int t = 0; t < seq_len; t++) {
            scores[t] = expf(scores[t] - m);
            sum += scores[t];
        }
        float inv = 1.f / sum;
        for (int t = 0; t < seq_len; t++) scores[t] *= inv;
    }
    __syncthreads();

    float* oh = out + (size_t)qi * (size_t)(n_head * head_dim) + h * head_dim;
    for (int d = (int)threadIdx.x; d < head_dim; d += (int)blockDim.x) {
        float acc = 0.f;
        for (int t = 0; t < seq_len; t++) {
            const float* vt = v_cache + (size_t)t * (size_t)stride + kv_h * head_dim;
            acc += scores[t] * vt[d];
        }
        oh[d] = acc;
    }
}

extern "C" __global__ void argmax_f32(
    const float* __restrict__ x,
    int n,
    int* __restrict__ out_idx
) {
    __shared__ float sbest_v[256];
    __shared__ int sbest_i[256];
    float best_v = -1e30f;
    int best_i = 0;
    for (int i = (int)threadIdx.x; i < n; i += (int)blockDim.x) {
        float v = x[i];
        if (v > best_v) { best_v = v; best_i = i; }
    }
    sbest_v[threadIdx.x] = best_v;
    sbest_i[threadIdx.x] = best_i;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) {
            if (sbest_v[threadIdx.x + s] > sbest_v[threadIdx.x]) {
                sbest_v[threadIdx.x] = sbest_v[threadIdx.x + s];
                sbest_i[threadIdx.x] = sbest_i[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out_idx[0] = sbest_i[0];
}

// Slice last token row: src[n_tok, dim] -> dst[dim]
extern "C" __global__ void copy_last_row(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n_tok,
    int dim
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < dim) dst[i] = src[(size_t)(n_tok - 1) * (size_t)dim + i];
}
