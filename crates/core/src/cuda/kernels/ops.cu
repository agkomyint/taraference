// Elementwise + RoPE + KV(f16) + attention + argmax

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

// Store K/V as IEEE f16 bits (unsigned short) — halves attention HBM traffic.
extern "C" __global__ void copy_kv_f16(
    const float* __restrict__ src,
    unsigned short* __restrict__ cache,
    int pos0,
    int n_tok,
    int stride
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = n_tok * stride;
    if (i < total) {
        int t = i / stride;
        int d = i % stride;
        cache[(size_t)(pos0 + t) * (size_t)stride + d] =
            float_to_half_bits(src[(size_t)t * (size_t)stride + d]);
    }
}

__device__ __forceinline__ float block_max(float v) {
    __shared__ float buf[256];
    int tid = (int)threadIdx.x;
    buf[tid] = v;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] = fmaxf(buf[tid], buf[tid + s]);
        __syncthreads();
    }
    return buf[0];
}

__device__ __forceinline__ float block_sum(float v) {
    __shared__ float buf[256];
    int tid = (int)threadIdx.x;
    buf[tid] = v;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] += buf[tid + s];
        __syncthreads();
    }
    return buf[0];
}

// Load one f16 K/V head vector element.
__device__ __forceinline__ float kv_load(
    const unsigned short* __restrict__ cache,
    int t, int stride, int kv_h, int head_dim, int d
) {
    return half_to_float(
        cache[(size_t)t * (size_t)stride + (size_t)kv_h * (size_t)head_dim + d]);
}

// ---------------------------------------------------------------------------
// Fast attention (default): f16 KV + tiled online softmax (fixed smem).
// No scores[seq_len] — cost scales with ctx but bandwidth is ~2× better and
// occupancy no longer collapses as chat grows.
// grid (n_head, n_q)  block 128   smem = (head_dim + TILE) * 4
// ---------------------------------------------------------------------------
#define ATTN_TILE 64

extern "C" __global__ void attn_f32(
    const float* __restrict__ q,
    const unsigned short* __restrict__ k_cache,
    const unsigned short* __restrict__ v_cache,
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
    int tid = (int)threadIdx.x;
    int nt = (int)blockDim.x;
    int rep = n_head / n_kv;
    int kv_h = h / rep;
    int stride = n_kv * head_dim;
    int pos = pos0 + qi;
    int seq_len = pos + 1;

    extern __shared__ float smem[];
    float* qh = smem;                    // head_dim
    float* scores = smem + head_dim;     // ATTN_TILE

    const float* qsrc = q + (size_t)qi * (size_t)(n_head * head_dim) + h * head_dim;
    for (int d = tid; d < head_dim; d += nt) qh[d] = qsrc[d];
    __syncthreads();

    float m = -1e30f;
    float l = 0.f;
    // Per-thread accumulator over head dims this thread owns.
    // head_dim <= 256, block 128 → at most 2 dims per thread.
    float acc0 = 0.f, acc1 = 0.f;
    int d0 = tid;
    int d1 = tid + nt;

    for (int t0 = 0; t0 < seq_len; t0 += ATTN_TILE) {
        int tlen = seq_len - t0;
        if (tlen > ATTN_TILE) tlen = ATTN_TILE;

        // Q·K for tile
        for (int t = tid; t < tlen; t += nt) {
            float dot = 0.f;
            #pragma unroll 8
            for (int d = 0; d < head_dim; d++) {
                dot += qh[d] * kv_load(k_cache, t0 + t, stride, kv_h, head_dim, d);
            }
            scores[t] = dot * scale;
        }
        __syncthreads();

        float local_m = -1e30f;
        for (int t = tid; t < tlen; t += nt) local_m = fmaxf(local_m, scores[t]);
        float tile_m = block_max(local_m);
        float m_new = fmaxf(m, tile_m);
        float alpha = expf(m - m_new);

        float local_s = 0.f;
        for (int t = tid; t < tlen; t += nt) {
            float e = expf(scores[t] - m_new);
            scores[t] = e;
            local_s += e;
        }
        float tile_l = block_sum(local_s);

        // rescale previous acc + accumulate V
        acc0 *= alpha;
        if (d1 < head_dim) acc1 *= alpha;
        if (d0 < head_dim) {
            float a0 = 0.f, a1 = 0.f;
            for (int t = 0; t < tlen; t++) {
                float w = scores[t];
                a0 += w * kv_load(v_cache, t0 + t, stride, kv_h, head_dim, d0);
                if (d1 < head_dim)
                    a1 += w * kv_load(v_cache, t0 + t, stride, kv_h, head_dim, d1);
            }
            acc0 += a0;
            acc1 += a1;
        }

        l = l * alpha + tile_l;
        m = m_new;
        __syncthreads();
    }

    float inv = 1.f / fmaxf(l, 1e-20f);
    float* oh = out + (size_t)qi * (size_t)(n_head * head_dim) + h * head_dim;
    if (d0 < head_dim) oh[d0] = acc0 * inv;
    if (d1 < head_dim) oh[d1] = acc1 * inv;
}

// Baseline: full scores in smem (A/B). Still f16 KV.
extern "C" __global__ void attn_basic_f32(
    const float* __restrict__ q,
    const unsigned short* __restrict__ k_cache,
    const unsigned short* __restrict__ v_cache,
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
    int tid = (int)threadIdx.x;
    int nt = (int)blockDim.x;
    int rep = n_head / n_kv;
    int kv_h = h / rep;
    int stride = n_kv * head_dim;
    int pos = pos0 + qi;
    int seq_len = pos + 1;
    const float* qh = q + (size_t)qi * (size_t)(n_head * head_dim) + h * head_dim;
    extern __shared__ float scores[];

    for (int t = tid; t < seq_len; t += nt) {
        float dot = 0.f;
        for (int d = 0; d < head_dim; d++)
            dot += qh[d] * kv_load(k_cache, t, stride, kv_h, head_dim, d);
        scores[t] = dot * scale;
    }
    __syncthreads();

    if (tid == 0) {
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
    for (int d = tid; d < head_dim; d += nt) {
        float acc = 0.f;
        for (int t = 0; t < seq_len; t++)
            acc += scores[t] * kv_load(v_cache, t, stride, kv_h, head_dim, d);
        oh[d] = acc;
    }
}

// Online softmax decode: f16 KV + warp-friendly score (still one pass over seq).
// grid: n_head, block: head_dim (one thread per dim).
extern "C" __global__ void attn_online_f32(
    const float* __restrict__ q,
    const unsigned short* __restrict__ k_cache,
    const unsigned short* __restrict__ v_cache,
    float* __restrict__ out,
    int n_head,
    int n_kv,
    int head_dim,
    int seq_len,
    float scale
) {
    int h = (int)blockIdx.x;
    if (h >= n_head) return;
    int tid = (int)threadIdx.x;
    if (tid >= head_dim) return;

    int rep = n_head / n_kv;
    int kv_h = h / rep;
    int stride = n_kv * head_dim;

    extern __shared__ float sm[];
    float* qh = sm;
    float* red = sm + head_dim;

    qh[tid] = q[h * head_dim + tid];
    __syncthreads();

    float m = -1e30f;
    float l = 0.f;
    float acc = 0.f;

    for (int t = 0; t < seq_len; t++) {
        float kd = kv_load(k_cache, t, stride, kv_h, head_dim, tid);
        float vd = kv_load(v_cache, t, stride, kv_h, head_dim, tid);

        red[tid] = qh[tid] * kd;
        __syncthreads();
        for (int s = head_dim / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        float score = red[0] * scale;

        float m_new = fmaxf(m, score);
        float alpha = expf(m - m_new);
        float beta = expf(score - m_new);
        acc = acc * alpha + beta * vd;
        l = l * alpha + beta;
        m = m_new;
        __syncthreads();
    }
    out[h * head_dim + tid] = acc / fmaxf(l, 1e-20f);
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

extern "C" __global__ void copy_last_row(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n_tok,
    int dim
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < dim) dst[i] = src[(size_t)(n_tok - 1) * (size_t)dim + i];
}
