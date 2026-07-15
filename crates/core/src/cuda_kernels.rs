//! NVRTC kernels — correct ggml Q4_K/Q6_K/Q8_0 dequant + decode ops.

pub const KERNELS: &str = r#"
__device__ __forceinline__ float half_to_float(unsigned short h) {
    unsigned int sign = ((unsigned int)h >> 15) & 1u;
    unsigned int exp  = ((unsigned int)h >> 10) & 0x1fu;
    unsigned int mant = (unsigned int)h & 0x3ffu;
    unsigned int f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0u) { mant <<= 1; exp--; }
            mant &= 0x3ffu;
            f = (sign << 31) | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7f800000u | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    return __int_as_float(f);
}

__device__ __forceinline__ void get_scale_min_k4(
    int j, const unsigned char* q, unsigned char* d, unsigned char* m
) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Dequant one Q4_K column (n_rows elems, n_rows % 256 == 0) into out[0..n_rows)
__device__ void dequant_q4_k_col(
    const unsigned char* col, float* out, int n_rows
) {
    const int QK = 256;
    int nb = n_rows / QK;
    int y_i = 0;
    for (int i = 0; i < nb; i++) {
        const unsigned char* base = col + i * 144;
        float d = half_to_float((unsigned short)(base[0] | (base[1] << 8)));
        float minv = half_to_float((unsigned short)(base[2] | (base[3] << 8)));
        const unsigned char* scales = base + 4;
        const unsigned char* q = base + 16;
        int is = 0;
        for (int t = 0; t < QK / 64; t++) {
            unsigned char sc, m;
            get_scale_min_k4(is, scales, &sc, &m);
            float d1 = d * (float)sc;
            float m1 = minv * (float)m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            float d2 = d * (float)sc;
            float m2 = minv * (float)m;
            for (int l = 0; l < 32; l++)
                out[y_i + l] = d1 * (float)(q[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++)
                out[y_i + 32 + l] = d2 * (float)(q[l] >> 4) - m2;
            y_i += 64;
            q += 32;
            is += 2;
        }
    }
}

__device__ void dequant_q6_k_col(
    const unsigned char* col, float* out, int n_rows
) {
    const int QK = 256;
    int nb = n_rows / QK;
    int y_off = 0;
    for (int i = 0; i < nb; i++) {
        const unsigned char* base = col + i * 210;
        const unsigned char* ql = base;
        const unsigned char* qh = base + 128;
        const signed char* sc = (const signed char*)(base + 192);
        float d = half_to_float((unsigned short)(base[208] | (base[209] << 8)));
        int ql_i = 0, qh_i = 0, sc_i = 0;
        for (int n = 0; n < QK / 128; n++) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = (int)((ql[ql_i + l] & 0xF) | (((qh[qh_i + l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[ql_i + 32 + l] & 0xF) | (((qh[qh_i + l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[ql_i + l] >> 4) | (((qh[qh_i + l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[ql_i + 32 + l] >> 4) | (((qh[qh_i + l] >> 6) & 3) << 4)) - 32;
                out[y_off + l]       = d * (float)sc[sc_i + is] * (float)q1;
                out[y_off + 32 + l]  = d * (float)sc[sc_i + is + 2] * (float)q2;
                out[y_off + 64 + l]  = d * (float)sc[sc_i + is + 4] * (float)q3;
                out[y_off + 96 + l]  = d * (float)sc[sc_i + is + 6] * (float)q4;
            }
            y_off += 128;
            ql_i += 64;
            qh_i += 32;
            sc_i += 8;
        }
    }
}

__device__ void dequant_q8_0_col(
    const unsigned char* col, float* out, int n_rows
) {
    const int QK = 32;
    int nb = n_rows / QK;
    for (int i = 0; i < nb; i++) {
        const unsigned char* base = col + i * 34;
        float d = half_to_float((unsigned short)(base[0] | (base[1] << 8)));
        const signed char* qs = (const signed char*)(base + 2);
        int yo = i * QK;
        for (int j = 0; j < QK; j++) out[yo + j] = (float)qs[j] * d;
    }
}

// Fused GEMV: one thread = one output column (simple, correct)
extern "C" __global__ void gemv_q4_k(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_rows,
    int n_cols,
    int col_bytes
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_cols) return;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    // stream dequant + dot without full column buffer (same math as dequant_q4_k_col)
    float sum = 0.f;
    const int QK = 256;
    int nb = n_rows / QK;
    int y_i = 0;
    for (int i = 0; i < nb; i++) {
        const unsigned char* base = col + i * 144;
        float d = half_to_float((unsigned short)(base[0] | (base[1] << 8)));
        float minv = half_to_float((unsigned short)(base[2] | (base[3] << 8)));
        const unsigned char* scales = base + 4;
        const unsigned char* q = base + 16;
        int is = 0;
        for (int t = 0; t < QK / 64; t++) {
            unsigned char sc, m;
            get_scale_min_k4(is, scales, &sc, &m);
            float d1 = d * (float)sc, m1 = minv * (float)m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            float d2 = d * (float)sc, m2 = minv * (float)m;
            for (int l = 0; l < 32; l++)
                sum += (d1 * (float)(q[l] & 0xF) - m1) * x[y_i + l];
            for (int l = 0; l < 32; l++)
                sum += (d2 * (float)(q[l] >> 4) - m2) * x[y_i + 32 + l];
            y_i += 64;
            q += 32;
            is += 2;
        }
    }
    out[j] = sum;
}

extern "C" __global__ void gemv_q6_k(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_rows,
    int n_cols,
    int col_bytes
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_cols) return;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    float sum = 0.f;
    const int QK = 256;
    int nb = n_rows / QK;
    int y_off = 0;
    for (int i = 0; i < nb; i++) {
        const unsigned char* base = col + i * 210;
        const unsigned char* ql = base;
        const unsigned char* qh = base + 128;
        const signed char* sc = (const signed char*)(base + 192);
        float d = half_to_float((unsigned short)(base[208] | (base[209] << 8)));
        int ql_i = 0, qh_i = 0, sc_i = 0;
        for (int n = 0; n < QK / 128; n++) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = (int)((ql[ql_i + l] & 0xF) | (((qh[qh_i + l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[ql_i + 32 + l] & 0xF) | (((qh[qh_i + l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[ql_i + l] >> 4) | (((qh[qh_i + l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[ql_i + 32 + l] >> 4) | (((qh[qh_i + l] >> 6) & 3) << 4)) - 32;
                sum += d * (float)sc[sc_i + is] * (float)q1 * x[y_off + l];
                sum += d * (float)sc[sc_i + is + 2] * (float)q2 * x[y_off + 32 + l];
                sum += d * (float)sc[sc_i + is + 4] * (float)q3 * x[y_off + 64 + l];
                sum += d * (float)sc[sc_i + is + 6] * (float)q4 * x[y_off + 96 + l];
            }
            y_off += 128;
            ql_i += 64;
            qh_i += 32;
            sc_i += 8;
        }
    }
    out[j] = sum;
}

extern "C" __global__ void gemv_q8_0(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_rows,
    int n_cols,
    int col_bytes
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_cols) return;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    float sum = 0.f;
    const int QK = 32;
    int nb = n_rows / QK;
    for (int i = 0; i < nb; i++) {
        const unsigned char* base = col + i * 34;
        float d = half_to_float((unsigned short)(base[0] | (base[1] << 8)));
        const signed char* qs = (const signed char*)(base + 2);
        int yo = i * QK;
        for (int t = 0; t < QK; t++) sum += (float)qs[t] * d * x[yo + t];
    }
    out[j] = sum;
}

// Embed: one CTA dequants column `token` (thread 0 does full col — n_embd is small)
extern "C" __global__ void embed_q4_k(
    const unsigned char* __restrict__ table,
    float* __restrict__ out,
    int token,
    int n_rows,
    int col_bytes
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    dequant_q4_k_col(table + (size_t)token * (size_t)col_bytes, out, n_rows);
}

extern "C" __global__ void embed_q6_k(
    const unsigned char* __restrict__ table,
    float* __restrict__ out,
    int token,
    int n_rows,
    int col_bytes
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    dequant_q6_k_col(table + (size_t)token * (size_t)col_bytes, out, n_rows);
}

extern "C" __global__ void embed_q8_0(
    const unsigned char* __restrict__ table,
    float* __restrict__ out,
    int token,
    int n_rows,
    int col_bytes
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    dequant_q8_0_col(table + (size_t)token * (size_t)col_bytes, out, n_rows);
}

extern "C" __global__ void rms_norm_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out,
    int n,
    float eps
) {
    __shared__ float buf[256];
    float local = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[i];
        local += v * v;
    }
    buf[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    __shared__ float scale;
    if (threadIdx.x == 0) scale = rsqrtf(buf[0] / (float)n + eps);
    __syncthreads();
    float s = scale;
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = x[i] * s * w[i];
}

extern "C" __global__ void silu_f32(float* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = v * (1.f / (1.f + expf(-v)));
    }
}

extern "C" __global__ void add_f32(float* __restrict__ a, const float* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

extern "C" __global__ void mul_f32(float* __restrict__ a, const float* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] *= b[i];
}

extern "C" __global__ void add_bias_f32(float* __restrict__ x, const float* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += b[i];
}

// Llama/Qwen Neox RoPE: rotate pairs in first/second half of each head
extern "C" __global__ void rope_neox_f32(
    float* __restrict__ x,
    int n_heads,
    int head_dim,
    int pos,
    float theta
) {
    int h = blockIdx.x;
    if (h >= n_heads) return;
    int half = head_dim / 2;
    float* base = x + h * head_dim;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float freq = powf(theta, -((float)i) / (float)half);
        // Qwen2/llama.cpp: inv_freq[i] = theta^(-2i/head_dim) for i in 0..half
        freq = powf(theta, -2.f * (float)i / (float)head_dim);
        float ang = (float)pos * freq;
        float c = cosf(ang), s = sinf(ang);
        float x0 = base[i];
        float x1 = base[i + half];
        base[i] = x0 * c - x1 * s;
        base[i + half] = x0 * s + x1 * c;
    }
}

extern "C" __global__ void attn_decode_f32(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ out,
    int n_head,
    int n_kv,
    int head_dim,
    int seq_len,
    float scale
) {
    int h = blockIdx.x;
    if (h >= n_head) return;
    int rep = n_head / n_kv;
    int kv_h = h / rep;
    int stride = n_kv * head_dim;
    const float* qh = q + h * head_dim;
    extern __shared__ float scores[];

    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        const float* kt = k_cache + t * stride + kv_h * head_dim;
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

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.f;
        for (int t = 0; t < seq_len; t++) {
            const float* vt = v_cache + t * stride + kv_h * head_dim;
            acc += scores[t] * vt[d];
        }
        out[h * head_dim + d] = acc;
    }
}

extern "C" __global__ void copy_kv_f32(
    const float* __restrict__ src,
    float* __restrict__ cache,
    int pos,
    int stride
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < stride) cache[(size_t)pos * (size_t)stride + i] = src[i];
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
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[i];
        if (v > best_v) { best_v = v; best_i = i; }
    }
    sbest_v[threadIdx.x] = best_v;
    sbest_i[threadIdx.x] = best_i;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sbest_v[threadIdx.x + s] > sbest_v[threadIdx.x]) {
                sbest_v[threadIdx.x] = sbest_v[threadIdx.x + s];
                sbest_i[threadIdx.x] = sbest_i[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out_idx[0] = sbest_i[0];
}

// Dense embd gather: table is column-major [n_embd, n_vocab]
extern "C" __global__ void embed_f32_col(
    const float* __restrict__ table,
    float* __restrict__ out,
    int token,
    int n_embd
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_embd) out[i] = table[(size_t)token * (size_t)n_embd + i];
}
"#;
