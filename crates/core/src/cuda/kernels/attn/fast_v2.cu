// --decode fastv2 (default)
// CUDA: attn_fast_v2  |  REGISTRY name: "fastv2"
// f16 KV + tiled online softmax (fixed smem, no scores[ctx]).
// grid (n_head, n_q)  block 128  smem = (head_dim + ATTN_TILE) * 4

#define ATTN_TILE 64

extern "C" __global__ void attn_fast_v2(
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

    for (int t0 = 0; t0 < seq_len; t0 += ATTN_TILE) {
        int tlen = seq_len - t0;
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

    float inv = 1.f / fmaxf(l, 1e-20f);
    float* oh = out + (size_t)qi * (size_t)(n_head * head_dim) + h * head_dim;
    if (d0 < head_dim) oh[d0] = acc0 * inv;
    if (d1 < head_dim) oh[d1] = acc1 * inv;
}
