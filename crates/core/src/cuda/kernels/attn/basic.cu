// --decode basic
// CUDA: attn_basic_f32  |  REGISTRY name: "basic"
// Serial softmax on thread 0 — A/B baseline.
// grid (n_head, n_q)  block 128  smem = seq_len * 4

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
