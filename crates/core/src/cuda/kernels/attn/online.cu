// --decode online
// CUDA: attn_online_f32  |  REGISTRY name: "online"
// Online softmax single-query decode; prefill uses fastv2 (see REGISTRY).
// grid: n_head, block: head_dim

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
