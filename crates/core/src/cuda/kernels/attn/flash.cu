// --decode flash (flash-decoding style split over sequence)
// CUDA: attn_flash_partial + attn_flash_reduce
// REGISTRY name: "flash"
//
// Decode (n_q==1): grid (n_head, n_split) computes online-softmax partials
// over disjoint KV ranges, then a reduce merges (m, l, O). Better SM fill when
// n_head is small and seq_len is long (reduces first→last tok/s drop).
// Prefill: host falls back to fastv2 via OnlineDecode-style or Causal prefill.

#define ATTN_TILE 64
#define FLASH_MAX_SPLIT 8

// partial layout per (h,s): m, l, then O[head_dim]
// stride_o = 2 + head_dim

extern "C" __global__ void attn_flash_partial(
    const float* __restrict__ q,
    const unsigned short* __restrict__ k_cache,
    const unsigned short* __restrict__ v_cache,
    float* __restrict__ partial, // [n_head, n_split, 2+hd]
    int n_head,
    int n_kv,
    int head_dim,
    int pos0,
    int n_q,
    int n_split,
    float scale
) {
    int h = (int)blockIdx.x;
    int s = (int)blockIdx.y;
    int qi = (int)blockIdx.z; // 0 for decode
    if (h >= n_head || s >= n_split || qi >= n_q) return;
    int tid = (int)threadIdx.x;
    int nt = (int)blockDim.x;
    int rep = n_head / n_kv;
    int kv_h = h / rep;
    int stride = n_kv * head_dim;
    int pos = pos0 + qi;
    int seq_len = pos + 1;

    // Split [0, seq_len) across n_split
    int chunk = (seq_len + n_split - 1) / n_split;
    int t_begin = s * chunk;
    int t_end = t_begin + chunk;
    if (t_end > seq_len) t_end = seq_len;

    extern __shared__ float smem[];
    float* qh = smem;
    float* scores = smem + head_dim;

    const float* qsrc = q + (size_t)qi * (size_t)(n_head * head_dim) + h * head_dim;
    for (int d = tid; d < head_dim; d += nt) qh[d] = qsrc[d];
    __syncthreads();

    float m = -1e30f;
    float l = 0.f;
    float acc0 = 0.f, acc1 = 0.f;
    int d0 = tid;
    int d1 = tid + nt;

    if (t_begin < t_end) {
        for (int t0 = t_begin; t0 < t_end; t0 += ATTN_TILE) {
            int tlen = t_end - t0;
            if (tlen > ATTN_TILE) tlen = ATTN_TILE;

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
    }

    // Write partial (m, l, O) — O not yet normalized by l (store weighted sum).
    int po = 2 + head_dim;
    float* base = partial + ((size_t)h * (size_t)n_split + (size_t)s) * (size_t)po;
    if (tid == 0) {
        base[0] = m;
        base[1] = l;
    }
    __syncthreads();
    if (d0 < head_dim) base[2 + d0] = acc0;
    if (d1 < head_dim) base[2 + d1] = acc1;
}

// Device-pos variant for CUDA graphs (n_q fixed to 1).
extern "C" __global__ void attn_flash_partial_d(
    const float* __restrict__ q,
    const unsigned short* __restrict__ k_cache,
    const unsigned short* __restrict__ v_cache,
    float* __restrict__ partial,
    int n_head,
    int n_kv,
    int head_dim,
    const int* __restrict__ pos0_ptr,
    int n_q,
    int n_split,
    float scale
) {
    // Inline same logic with pos0 from device.
    int h = (int)blockIdx.x;
    int s = (int)blockIdx.y;
    int qi = (int)blockIdx.z;
    if (h >= n_head || s >= n_split || qi >= n_q) return;
    int pos0 = pos0_ptr[0];
    int tid = (int)threadIdx.x;
    int nt = (int)blockDim.x;
    int rep = n_head / n_kv;
    int kv_h = h / rep;
    int stride = n_kv * head_dim;
    int pos = pos0 + qi;
    int seq_len = pos + 1;

    int chunk = (seq_len + n_split - 1) / n_split;
    int t_begin = s * chunk;
    int t_end = t_begin + chunk;
    if (t_end > seq_len) t_end = seq_len;

    extern __shared__ float smem[];
    float* qh = smem;
    float* scores = smem + head_dim;

    const float* qsrc = q + (size_t)qi * (size_t)(n_head * head_dim) + h * head_dim;
    for (int d = tid; d < head_dim; d += nt) qh[d] = qsrc[d];
    __syncthreads();

    float m = -1e30f;
    float l = 0.f;
    float acc0 = 0.f, acc1 = 0.f;
    int d0 = tid;
    int d1 = tid + nt;

    if (t_begin < t_end) {
        for (int t0 = t_begin; t0 < t_end; t0 += ATTN_TILE) {
            int tlen = t_end - t0;
            if (tlen > ATTN_TILE) tlen = ATTN_TILE;

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
    }

    int po = 2 + head_dim;
    float* base = partial + ((size_t)h * (size_t)n_split + (size_t)s) * (size_t)po;
    if (tid == 0) {
        base[0] = m;
        base[1] = l;
    }
    __syncthreads();
    if (d0 < head_dim) base[2 + d0] = acc0;
    if (d1 < head_dim) base[2 + d1] = acc1;
}

// Merge partials: out[h] = sum_s exp(m_s - m) * O_s / sum_s exp(m_s-m)*l_s
extern "C" __global__ void attn_flash_reduce(
    const float* __restrict__ partial,
    float* __restrict__ out,
    int n_head,
    int head_dim,
    int n_split,
    int n_q
) {
    int h = (int)blockIdx.x;
    int qi = (int)blockIdx.y;
    if (h >= n_head || qi >= n_q) return;
    int tid = (int)threadIdx.x;
    int po = 2 + head_dim;

    // Find global m across splits (serial — n_split small).
    float m = -1e30f;
    for (int s = 0; s < n_split; s++) {
        const float* base = partial + ((size_t)h * (size_t)n_split + (size_t)s) * (size_t)po;
        m = fmaxf(m, base[0]);
    }
    float l = 0.f;
    float acc = 0.f;
    int d = tid;
    if (d < head_dim) {
        for (int s = 0; s < n_split; s++) {
            const float* base = partial + ((size_t)h * (size_t)n_split + (size_t)s) * (size_t)po;
            float alpha = expf(base[0] - m);
            l += alpha * base[1];
            acc += alpha * base[2 + d];
        }
        float* oh = out + (size_t)qi * (size_t)(n_head * head_dim) + h * head_dim;
        oh[d] = acc / fmaxf(l, 1e-20f);
    }
}
