// Elementwise + RoPE + KV(f16) + argmax
// Attention kernels live in kernels/attn/*.cu (one file per --decode backend).

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

/// Decode attn prep (n_tok=1): RMSNorm → xb + Q8 quantize for fused QKV GEMV.
extern "C" __global__ void attn_prep_rms_quant(
    const float* __restrict__ x,
    const float* __restrict__ attn_norm,
    float* __restrict__ xb,
    signed char* __restrict__ q8,
    float* __restrict__ d8,
    int n_embd,
    float eps
) {
    int tid = (int)threadIdx.x;
    __shared__ float red[256];
    __shared__ float scale;
    float local = 0.f;
    for (int i = tid; i < n_embd; i += (int)blockDim.x) {
        float v = x[i];
        local += v * v;
    }
    red[tid] = local;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    if (tid == 0) scale = rsqrtf(red[0] / (float)n_embd + eps);
    __syncthreads();
    float s = scale;
    for (int i = tid; i < n_embd; i += (int)blockDim.x)
        xb[i] = x[i] * s * attn_norm[i];
    __syncthreads();
    int n_blocks = n_embd >> 5;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = (int)blockDim.x >> 5;
    for (int bi = warp; bi < n_blocks; bi += nwarps) {
        int i = (bi << 5) + lane;
        float xi = xb[i];
        float amax = fabsf(xi);
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, 16));
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, 8));
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, 4));
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, 2));
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, 1));
        amax = __shfl_sync(0xffffffffu, amax, 0);
        float d = amax * (1.0f / 127.0f);
        q8[i] = amax == 0.f ? 0 : (signed char)__float2int_rn(xi / d);
        if (lane == 0) d8[bi] = d;
    }
}

/// Decode MoE prep (n_tok=1): RMSNorm → xb, device router top-k, Q8 quantize xb.
/// One block (256 threads). Replaces 3 launches per MoE layer.
extern "C" __global__ void moe_ffn_prep_rms_router_quant(
    const float* __restrict__ x,
    const float* __restrict__ ffn_norm,
    float* __restrict__ xb,
    const float* __restrict__ router,
    int* __restrict__ top_idx,
    float* __restrict__ top_w,
    signed char* __restrict__ q8,
    float* __restrict__ d8,
    int n_embd,
    int n_experts,
    int top_k,
    float eps
) {
    int tid = (int)threadIdx.x;
    __shared__ float red[256];
    __shared__ float scores[32];
    __shared__ float scale;

    // RMSNorm
    float local = 0.f;
    for (int i = tid; i < n_embd; i += (int)blockDim.x) {
        float v = x[i];
        local += v * v;
    }
    red[tid] = local;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    if (tid == 0) scale = rsqrtf(red[0] / (float)n_embd + eps);
    __syncthreads();
    float s = scale;
    for (int i = tid; i < n_embd; i += (int)blockDim.x)
        xb[i] = x[i] * s * ffn_norm[i];
    __syncthreads();

    // Router scores (E is tiny, serial over experts)
    for (int e = 0; e < n_experts; e++) {
        const float* wr = router + (size_t)e * (size_t)n_embd;
        float loc = 0.f;
        for (int i = tid; i < n_embd; i += (int)blockDim.x)
            loc += wr[i] * xb[i];
        red[tid] = loc;
        __syncthreads();
        for (int ss = (int)blockDim.x / 2; ss > 0; ss >>= 1) {
            if (tid < ss) red[tid] += red[tid + ss];
            __syncthreads();
        }
        if (tid == 0) scores[e] = red[0];
        __syncthreads();
    }
    if (tid == 0) {
        for (int k = 0; k < top_k; k++) {
            int best_i = 0;
            float best_v = -1e30f;
            for (int e = 0; e < n_experts; e++) {
                float sc = scores[e];
                if (sc > best_v) {
                    best_v = sc;
                    best_i = e;
                }
            }
            top_idx[k] = best_i;
            top_w[k] = best_v;
            scores[best_i] = -1e30f;
        }
        float max_l = top_w[0];
        for (int k = 1; k < top_k; k++)
            if (top_w[k] > max_l) max_l = top_w[k];
        float sum = 0.f;
        for (int k = 0; k < top_k; k++) {
            top_w[k] = expf(top_w[k] - max_l);
            sum += top_w[k];
        }
        float inv = sum > 0.f ? 1.f / sum : 0.f;
        for (int k = 0; k < top_k; k++) top_w[k] *= inv;
    }
    __syncthreads();

    // Q8 quantize xb (groups of 32)
    int n_blocks = n_embd >> 5;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = (int)blockDim.x >> 5;
    for (int bi = warp; bi < n_blocks; bi += nwarps) {
        int i = (bi << 5) + lane;
        float xi = xb[i];
        float amax = fabsf(xi);
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, 16));
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, 8));
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, 4));
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, 2));
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, 1));
        amax = __shfl_sync(0xffffffffu, amax, 0);
        float d = amax * (1.0f / 127.0f);
        q8[i] = amax == 0.f ? 0 : (signed char)__float2int_rn(xi / d);
        if (lane == 0) d8[bi] = d;
    }
}

extern "C" __global__ void silu_mul_f32(
    float* __restrict__ gate,
    const float* __restrict__ up,
    int n
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        float v = gate[i];
        // Fast SiLU: v * sigmoid(v); use intrinsic exp.
        gate[i] = (v / (1.f + __expf(-v))) * up[i];
    }
}

extern "C" __global__ void add_f32(
    float* __restrict__ a, const float* __restrict__ b, int n
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) a[i] += b[i];
}

/// a[i] += scale * b[i]
extern "C" __global__ void scale_add_f32(
    float* __restrict__ a, const float* __restrict__ b, float scale, int n
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) a[i] += scale * b[i];
}

/// y[e] = sum_i W[e * n_in + i] * x[i]  (dense f32 router GEMV, one block per expert row)
extern "C" __global__ void gemv_f32_rows(
    const float* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int n_rows,
    int n_in
) {
    int row = (int)blockIdx.x;
    if (row >= n_rows) return;
    const float* wr = w + (size_t)row * (size_t)n_in;
    __shared__ float buf[256];
    float local = 0.f;
    for (int i = (int)threadIdx.x; i < n_in; i += (int)blockDim.x)
        local += wr[i] * x[i];
    buf[threadIdx.x] = local;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = buf[0];
}

/// Device MoE router: scores = router @ x, write top-k indices + softmax weights.
/// One block; n_experts <= 32, top_k <= 8. Fully device-side (CUDA-graph safe).
extern "C" __global__ void moe_router_topk_f32(
    const float* __restrict__ router,
    const float* __restrict__ x,
    int* __restrict__ top_idx,
    float* __restrict__ top_w,
    int n_experts,
    int n_embd,
    int top_k
) {
    __shared__ float scores[32];
    __shared__ float red[256];
    int tid = (int)threadIdx.x;
    // Score each expert (serial over E, parallel reduce over embd — E is tiny).
    for (int e = 0; e < n_experts; e++) {
        const float* wr = router + (size_t)e * (size_t)n_embd;
        float local = 0.f;
        for (int i = tid; i < n_embd; i += (int)blockDim.x)
            local += wr[i] * x[i];
        red[tid] = local;
        __syncthreads();
        for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        if (tid == 0) scores[e] = red[0];
        __syncthreads();
    }
    if (tid == 0) {
        // Iterative top-k (n_experts <= 32).
        float taken = -1e30f;
        for (int k = 0; k < top_k; k++) {
            int best_i = 0;
            float best_v = -1e30f;
            for (int e = 0; e < n_experts; e++) {
                float s = scores[e];
                // Skip already chosen by marking after pick.
                if (s > best_v) {
                    best_v = s;
                    best_i = e;
                }
            }
            top_idx[k] = best_i;
            top_w[k] = best_v;
            scores[best_i] = -1e30f; // remove from next rounds
            (void)taken;
        }
        // Softmax over selected logits only.
        float max_l = top_w[0];
        for (int k = 1; k < top_k; k++)
            if (top_w[k] > max_l) max_l = top_w[k];
        float sum = 0.f;
        for (int k = 0; k < top_k; k++) {
            top_w[k] = expf(top_w[k] - max_l);
            sum += top_w[k];
        }
        float inv = sum > 0.f ? 1.f / sum : 0.f;
        for (int k = 0; k < top_k; k++) top_w[k] *= inv;
    }
}

/// a[i] += weights[slot] * b[i]  (device MoE residual; graph-safe)
extern "C" __global__ void scale_add_slot_f32(
    float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ weights,
    int slot,
    int n
) {
    float scale = weights[slot];
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) a[i] += scale * b[i];
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

// Device-side pos0 (for CUDA graph replay — scalar pos0 would be baked at capture).
extern "C" __global__ void rope_neox_f32_d(
    float* __restrict__ x,
    int n_heads,
    int head_dim,
    const int* __restrict__ pos0_ptr,
    int n_tok,
    float theta
) {
    int pos0 = pos0_ptr[0];
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

// Qwen3: per-head RMS normalization is applied to Q and K before RoPE.  Fuse
// both operations so architecture correctness costs no additional launch.
__device__ __forceinline__ void qk_rms_norm_rope_impl(
    float* __restrict__ base,
    const float* __restrict__ w,
    int head_dim,
    int pos,
    float theta,
    float eps
) {
    __shared__ float sums[256];
    int tid = (int)threadIdx.x;
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
    int half = head_dim / 2;
    for (int i = tid; i < half; i += (int)blockDim.x) {
        float freq = powf(theta, -2.f * (float)i / (float)head_dim);
        float ang = (float)pos * freq;
        float c = cosf(ang), s = sinf(ang);
        float x0 = base[i] * scale * w[i];
        float x1 = base[i + half] * scale * w[i + half];
        base[i] = x0 * c - x1 * s;
        base[i + half] = x0 * s + x1 * c;
    }
}

extern "C" __global__ void qk_rms_norm_rope_neox_f32(
    float* __restrict__ x,
    const float* __restrict__ w,
    int n_heads,
    int head_dim,
    int pos0,
    int n_tok,
    float theta,
    float eps
) {
    int h = (int)blockIdx.x;
    int t = (int)blockIdx.y;
    if (h >= n_heads || t >= n_tok) return;
    float* base = x + (size_t)t * (size_t)(n_heads * head_dim) + h * head_dim;
    qk_rms_norm_rope_impl(base, w, head_dim, pos0 + t, theta, eps);
}

extern "C" __global__ void qk_rms_norm_rope_neox_f32_d(
    float* __restrict__ x,
    const float* __restrict__ w,
    int n_heads,
    int head_dim,
    const int* __restrict__ pos0_ptr,
    int n_tok,
    float theta,
    float eps
) {
    int h = (int)blockIdx.x;
    int t = (int)blockIdx.y;
    if (h >= n_heads || t >= n_tok) return;
    float* base = x + (size_t)t * (size_t)(n_heads * head_dim) + h * head_dim;
    qk_rms_norm_rope_impl(base, w, head_dim, pos0_ptr[0] + t, theta, eps);
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

extern "C" __global__ void copy_kv_f16_d(
    const float* __restrict__ src,
    unsigned short* __restrict__ cache,
    const int* __restrict__ pos0_ptr,
    int n_tok,
    int stride
) {
    int pos0 = pos0_ptr[0];
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = n_tok * stride;
    if (i < total) {
        int t = i / stride;
        int d = i % stride;
        cache[(size_t)(pos0 + t) * (size_t)stride + d] =
            float_to_half_bits(src[(size_t)t * (size_t)stride + d]);
    }
}

// Shared by all attn/*.cu kernels (same TU after concat).
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

__device__ __forceinline__ float kv_load(
    const unsigned short* __restrict__ cache,
    int t, int stride, int kv_h, int head_dim, int d
) {
    return half_to_float(
        cache[(size_t)t * (size_t)stride + (size_t)kv_h * (size_t)head_dim + d]);
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

extern "C" __global__ void argmax_rows_f32(
    const float* __restrict__ x,
    int n_cols,
    int n_rows,
    int* __restrict__ out_idx
) {
    int row = (int)blockIdx.x;
    if (row >= n_rows) return;
    x += (size_t)row * (size_t)n_cols;
    __shared__ float sbest_v[256];
    __shared__ int sbest_i[256];
    float best_v = -1e30f;
    int best_i = 0;
    for (int i = (int)threadIdx.x; i < n_cols; i += (int)blockDim.x) {
        float v = x[i];
        if (v > best_v) { best_v = v; best_i = i; }
    }
    sbest_v[threadIdx.x] = best_v;
    sbest_i[threadIdx.x] = best_i;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s && sbest_v[threadIdx.x + s] > sbest_v[threadIdx.x]) {
            sbest_v[threadIdx.x] = sbest_v[threadIdx.x + s];
            sbest_i[threadIdx.x] = sbest_i[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out_idx[row] = sbest_i[0];
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
