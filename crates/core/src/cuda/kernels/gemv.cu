// Decode GEMV: bandwidth-first.
// - Stage x[] (or a row-slice) in shared memory
// - 8 warps / block → 8 output columns
// - Optional bias / residual via flags
// - Split-K path: grid (col_blocks, n_split) for better SM occupancy when n_rows is large

#define GEMV_WARPS 8
#define GEMV_THREADS (GEMV_WARPS * 32)
// Must match matmul.rs GEMV_SPLIT_MAX
#define GEMV_SPLIT_MAX 8

__device__ __forceinline__ float dot_q4_k_col_xs_range(
    const unsigned char* __restrict__ col,
    const float* __restrict__ xs,
    int bi0,
    int bi1,
    int lane
) {
    // xs is only the slice for superblocks [bi0, bi1); local index (bi-bi0)*256
    float acc = 0.f;
    for (int bi = bi0; bi < bi1; bi++) {
        const unsigned char* base = col + bi * 144;
        float d = half_to_float((unsigned short)(base[0] | (base[1] << 8)));
        float minv = half_to_float((unsigned short)(base[2] | (base[3] << 8)));
        const unsigned char* scales = base + 4;
        const unsigned char* q = base + 16;
        const float* xb = xs + (bi - bi0) * 256;
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            unsigned char sc, m;
            get_scale_min_k4(t * 2, scales, &sc, &m);
            float d1 = d * (float)sc, m1 = minv * (float)m;
            get_scale_min_k4(t * 2 + 1, scales, &sc, &m);
            float d2 = d * (float)sc, m2 = minv * (float)m;
            unsigned char qq = q[t * 32 + lane];
            acc += (d1 * (float)(qq & 0xF) - m1) * xb[t * 64 + lane];
            acc += (d2 * (float)(qq >> 4) - m2) * xb[t * 64 + 32 + lane];
        }
    }
    return warp_sum(acc);
}

__device__ __forceinline__ float dot_q4_k_col_xs(
    const unsigned char* __restrict__ col,
    const float* __restrict__ xs,
    int n_rows,
    int lane
) {
    return dot_q4_k_col_xs_range(col, xs, 0, n_rows / 256, lane);
}

__device__ __forceinline__ float dot_q6_k_col_xs_range(
    const unsigned char* __restrict__ col,
    const float* __restrict__ xs,
    int bi0,
    int bi1,
    int lane
) {
    float acc = 0.f;
    for (int bi = bi0; bi < bi1; bi++) {
        const unsigned char* base = col + bi * 210;
        const unsigned char* ql = base;
        const unsigned char* qh = base + 128;
        const signed char* sc = (const signed char*)(base + 192);
        float d = half_to_float((unsigned short)(base[208] | (base[209] << 8)));
        const float* xb = xs + (bi - bi0) * 256;
        #pragma unroll
        for (int n = 0; n < 2; n++) {
            int ql_i = n * 64, qh_i = n * 32, sc_i = n * 8;
            int y0 = n * 128;
            int is = lane / 16;
            int q1 = (int)((ql[ql_i + lane] & 0xF) | (((qh[qh_i + lane] >> 0) & 3) << 4)) - 32;
            int q2 = (int)((ql[ql_i + 32 + lane] & 0xF) | (((qh[qh_i + lane] >> 2) & 3) << 4)) - 32;
            int q3 = (int)((ql[ql_i + lane] >> 4) | (((qh[qh_i + lane] >> 4) & 3) << 4)) - 32;
            int q4 = (int)((ql[ql_i + 32 + lane] >> 4) | (((qh[qh_i + lane] >> 6) & 3) << 4)) - 32;
            acc += d * (float)sc[sc_i + is] * (float)q1 * xb[y0 + lane];
            acc += d * (float)sc[sc_i + is + 2] * (float)q2 * xb[y0 + 32 + lane];
            acc += d * (float)sc[sc_i + is + 4] * (float)q3 * xb[y0 + 64 + lane];
            acc += d * (float)sc[sc_i + is + 6] * (float)q4 * xb[y0 + 96 + lane];
        }
    }
    return warp_sum(acc);
}

__device__ __forceinline__ float dot_q6_k_col_xs(
    const unsigned char* __restrict__ col,
    const float* __restrict__ xs,
    int n_rows,
    int lane
) {
    return dot_q6_k_col_xs_range(col, xs, 0, n_rows / 256, lane);
}

// use_res: 0 = none, 1 = +residual[j], 2 = +out[j] in-place (before write).
__device__ __forceinline__ float gemv_apply_res(
    int use_res, float acc, float* out, int j, const float* residual
) {
    if (use_res == 1) acc += residual[j];
    else if (use_res == 2) acc += out[j];
    return acc;
}

// ── baseline (full rows) ───────────────────────────────────────────────────

extern "C" __global__ void gemv_q4_k(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_rows, int n_cols, int col_bytes,
    int use_bias, const float* __restrict__ bias,
    int use_res, const float* __restrict__ residual
) {
    extern __shared__ float xs[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    for (int i = tid; i < n_rows; i += GEMV_THREADS) xs[i] = x[i];
    __syncthreads();

    int j = (int)blockIdx.x * GEMV_WARPS + warp;
    if (j >= n_cols) return;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    float acc = dot_q4_k_col_xs(col, xs, n_rows, lane);
    if (lane == 0) {
        acc = gemv_apply_res(use_res, acc, out, j, residual);
        if (use_bias) acc += bias[j];
        out[j] = acc;
    }
}

extern "C" __global__ void gemv_q6_k(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_rows, int n_cols, int col_bytes,
    int use_bias, const float* __restrict__ bias,
    int use_res, const float* __restrict__ residual
) {
    extern __shared__ float xs[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    for (int i = tid; i < n_rows; i += GEMV_THREADS) xs[i] = x[i];
    __syncthreads();

    int j = (int)blockIdx.x * GEMV_WARPS + warp;
    if (j >= n_cols) return;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    float acc = dot_q6_k_col_xs(col, xs, n_rows, lane);
    if (lane == 0) {
        acc = gemv_apply_res(use_res, acc, out, j, residual);
        if (use_bias) acc += bias[j];
        out[j] = acc;
    }
}

extern "C" __global__ void gemv_q8_0(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_rows, int n_cols, int col_bytes,
    int use_bias, const float* __restrict__ bias,
    int use_res, const float* __restrict__ residual
) {
    extern __shared__ float xs[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    for (int i = tid; i < n_rows; i += GEMV_THREADS) xs[i] = x[i];
    __syncthreads();

    int j = (int)blockIdx.x * GEMV_WARPS + warp;
    if (j >= n_cols) return;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    float acc = 0.f;
    int nb = n_rows / 32;
    for (int bi = lane; bi < nb; bi += 32) {
        const unsigned char* base = col + bi * 34;
        float d = half_to_float((unsigned short)(base[0] | (base[1] << 8)));
        const signed char* qs = (const signed char*)(base + 2);
        int yo = bi * 32;
        #pragma unroll 8
        for (int t = 0; t < 32; t++) acc += (float)qs[t] * d * xs[yo + t];
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        acc = gemv_apply_res(use_res, acc, out, j, residual);
        if (use_bias) acc += bias[j];
        out[j] = acc;
    }
}

extern "C" __global__ void gemv_q5_0(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_rows, int n_cols, int col_bytes,
    int use_bias, const float* __restrict__ bias,
    int use_res, const float* __restrict__ residual
) {
    extern __shared__ float xs[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    for (int i = tid; i < n_rows; i += GEMV_THREADS) xs[i] = x[i];
    __syncthreads();

    int j = (int)blockIdx.x * GEMV_WARPS + warp;
    if (j >= n_cols) return;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    float acc = 0.f;
    int nb = n_rows / 32;
    for (int bi = lane; bi < nb; bi += 32) {
        const unsigned char* base = col + bi * 22;
        float d = half_to_float((unsigned short)(base[0] | (base[1] << 8)));
        unsigned int qh = (unsigned int)base[2]
            | ((unsigned int)base[3] << 8)
            | ((unsigned int)base[4] << 16)
            | ((unsigned int)base[5] << 24);
        const unsigned char* qs = base + 6;
        int yo = bi * 32;
        #pragma unroll
        for (int t = 0; t < 16; t++) {
            unsigned char xh0 = (unsigned char)(((qh >> t) << 4) & 0x10u);
            unsigned char xh1 = (unsigned char)(((qh >> (t + 12))) & 0x10u);
            int x0 = (int)((qs[t] & 0x0F) | xh0);
            int x1 = (int)((qs[t] >> 4) | xh1);
            acc += (float)(x0 - 16) * d * xs[yo + t];
            acc += (float)(x1 - 16) * d * xs[yo + 16 + t];
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        acc = gemv_apply_res(use_res, acc, out, j, residual);
        if (use_bias) acc += bias[j];
        out[j] = acc;
    }
}

// ── Split-K partials ───────────────────────────────────────────────────────
// grid: (ceil(n_cols/GEMV_WARPS), n_split)
// partial[s * n_cols + j] = partial dot for split s
// Superblock quant (Q4_K / Q6_K): split on 256-row superblocks.
// Group quant (Q5_0 / Q8_0): split on 32-row blocks.

extern "C" __global__ void gemv_q4_k_splitk(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ partial,
    int n_rows, int n_cols, int col_bytes, int n_split
) {
    extern __shared__ float xs[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int s = (int)blockIdx.y;
    if (s >= n_split) return;

    const int nsb = n_rows / 256;
    const int bi0 = (nsb * s) / n_split;
    const int bi1 = (nsb * (s + 1)) / n_split;
    const int row0 = bi0 * 256;
    const int n_local = (bi1 - bi0) * 256;
    for (int i = tid; i < n_local; i += GEMV_THREADS) xs[i] = x[row0 + i];
    __syncthreads();

    int j = (int)blockIdx.x * GEMV_WARPS + warp;
    if (j >= n_cols) return;
    if (bi0 >= bi1) {
        if (lane == 0) partial[(size_t)s * (size_t)n_cols + j] = 0.f;
        return;
    }
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    float acc = dot_q4_k_col_xs_range(col, xs, bi0, bi1, lane);
    if (lane == 0) partial[(size_t)s * (size_t)n_cols + j] = acc;
}

extern "C" __global__ void gemv_q6_k_splitk(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ partial,
    int n_rows, int n_cols, int col_bytes, int n_split
) {
    extern __shared__ float xs[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int s = (int)blockIdx.y;
    if (s >= n_split) return;

    const int nsb = n_rows / 256;
    const int bi0 = (nsb * s) / n_split;
    const int bi1 = (nsb * (s + 1)) / n_split;
    const int row0 = bi0 * 256;
    const int n_local = (bi1 - bi0) * 256;
    for (int i = tid; i < n_local; i += GEMV_THREADS) xs[i] = x[row0 + i];
    __syncthreads();

    int j = (int)blockIdx.x * GEMV_WARPS + warp;
    if (j >= n_cols) return;
    if (bi0 >= bi1) {
        if (lane == 0) partial[(size_t)s * (size_t)n_cols + j] = 0.f;
        return;
    }
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    float acc = dot_q6_k_col_xs_range(col, xs, bi0, bi1, lane);
    if (lane == 0) partial[(size_t)s * (size_t)n_cols + j] = acc;
}

extern "C" __global__ void gemv_q8_0_splitk(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ partial,
    int n_rows, int n_cols, int col_bytes, int n_split
) {
    extern __shared__ float xs[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int s = (int)blockIdx.y;
    if (s >= n_split) return;

    const int nb = n_rows / 32;
    const int bi0 = (nb * s) / n_split;
    const int bi1 = (nb * (s + 1)) / n_split;
    const int row0 = bi0 * 32;
    const int n_local = (bi1 - bi0) * 32;
    for (int i = tid; i < n_local; i += GEMV_THREADS) xs[i] = x[row0 + i];
    __syncthreads();

    int j = (int)blockIdx.x * GEMV_WARPS + warp;
    if (j >= n_cols) return;
    if (bi0 >= bi1) {
        if (lane == 0) partial[(size_t)s * (size_t)n_cols + j] = 0.f;
        return;
    }
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    float acc = 0.f;
    for (int bi = bi0 + lane; bi < bi1; bi += 32) {
        const unsigned char* base = col + bi * 34;
        float d = half_to_float((unsigned short)(base[0] | (base[1] << 8)));
        const signed char* qs = (const signed char*)(base + 2);
        int yo = (bi - bi0) * 32;
        #pragma unroll 8
        for (int t = 0; t < 32; t++) acc += (float)qs[t] * d * xs[yo + t];
    }
    acc = warp_sum(acc);
    if (lane == 0) partial[(size_t)s * (size_t)n_cols + j] = acc;
}

extern "C" __global__ void gemv_q5_0_splitk(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ partial,
    int n_rows, int n_cols, int col_bytes, int n_split
) {
    extern __shared__ float xs[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int s = (int)blockIdx.y;
    if (s >= n_split) return;

    const int nb = n_rows / 32;
    const int bi0 = (nb * s) / n_split;
    const int bi1 = (nb * (s + 1)) / n_split;
    const int row0 = bi0 * 32;
    const int n_local = (bi1 - bi0) * 32;
    for (int i = tid; i < n_local; i += GEMV_THREADS) xs[i] = x[row0 + i];
    __syncthreads();

    int j = (int)blockIdx.x * GEMV_WARPS + warp;
    if (j >= n_cols) return;
    if (bi0 >= bi1) {
        if (lane == 0) partial[(size_t)s * (size_t)n_cols + j] = 0.f;
        return;
    }
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    float acc = 0.f;
    for (int bi = bi0 + lane; bi < bi1; bi += 32) {
        const unsigned char* base = col + bi * 22;
        float d = half_to_float((unsigned short)(base[0] | (base[1] << 8)));
        unsigned int qh = (unsigned int)base[2]
            | ((unsigned int)base[3] << 8)
            | ((unsigned int)base[4] << 16)
            | ((unsigned int)base[5] << 24);
        const unsigned char* qs = base + 6;
        int yo = (bi - bi0) * 32;
        #pragma unroll
        for (int t = 0; t < 16; t++) {
            unsigned char xh0 = (unsigned char)(((qh >> t) << 4) & 0x10u);
            unsigned char xh1 = (unsigned char)(((qh >> (t + 12))) & 0x10u);
            int x0 = (int)((qs[t] & 0x0F) | xh0);
            int x1 = (int)((qs[t] >> 4) | xh1);
            acc += (float)(x0 - 16) * d * xs[yo + t];
            acc += (float)(x1 - 16) * d * xs[yo + 16 + t];
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) partial[(size_t)s * (size_t)n_cols + j] = acc;
}

// Sum partial[s, j] → out[j], apply bias / residual once.
extern "C" __global__ void gemv_splitk_reduce(
    const float* __restrict__ partial,
    float* __restrict__ out,
    int n_cols,
    int n_split,
    int use_bias,
    const float* __restrict__ bias,
    int use_res,
    const float* __restrict__ residual
) {
    int j = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (j >= n_cols) return;
    float acc = 0.f;
    #pragma unroll 4
    for (int s = 0; s < n_split; s++) {
        acc += partial[(size_t)s * (size_t)n_cols + j];
    }
    acc = gemv_apply_res(use_res, acc, out, j, residual);
    if (use_bias) acc += bias[j];
    out[j] = acc;
}

// ── Fused dual GEMV (decode, Q5_0): stage x once ───────────────────────────
// Used for Q+K and for gate+up (same n_rows, same quant, two outs).

__device__ __forceinline__ float dot_q5_0_col_xs(
    const unsigned char* __restrict__ col,
    const float* __restrict__ xs,
    int n_rows,
    int lane
) {
    float acc = 0.f;
    int nb = n_rows / 32;
    for (int bi = lane; bi < nb; bi += 32) {
        const unsigned char* base = col + bi * 22;
        float d = half_to_float((unsigned short)(base[0] | (base[1] << 8)));
        unsigned int qh = (unsigned int)base[2]
            | ((unsigned int)base[3] << 8)
            | ((unsigned int)base[4] << 16)
            | ((unsigned int)base[5] << 24);
        const unsigned char* qs = base + 6;
        int yo = bi * 32;
        #pragma unroll
        for (int t = 0; t < 16; t++) {
            unsigned char xh0 = (unsigned char)(((qh >> t) << 4) & 0x10u);
            unsigned char xh1 = (unsigned char)(((qh >> (t + 12))) & 0x10u);
            int x0 = (int)((qs[t] & 0x0F) | xh0);
            int x1 = (int)((qs[t] >> 4) | xh1);
            acc += (float)(x0 - 16) * d * xs[yo + t];
            acc += (float)(x1 - 16) * d * xs[yo + 16 + t];
        }
    }
    return warp_sum(acc);
}

extern "C" __global__ void gemv_q5_0_qk(
    const unsigned char* __restrict__ wq,
    const unsigned char* __restrict__ wk,
    const float* __restrict__ x,
    float* __restrict__ q,
    float* __restrict__ k,
    int n_rows, int n_q, int n_k, int col_bytes,
    int use_bias,
    const float* __restrict__ bq,
    const float* __restrict__ bk
) {
    extern __shared__ float xs[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    for (int i = tid; i < n_rows; i += GEMV_THREADS) xs[i] = x[i];
    __syncthreads();

    int j = (int)blockIdx.x * GEMV_WARPS + warp;
    int n_tot = n_q + n_k;
    if (j >= n_tot) return;

    const unsigned char* wcol;
    float* out;
    const float* bias;
    int oj;
    if (j < n_q) {
        wcol = wq + (size_t)j * (size_t)col_bytes;
        out = q;
        bias = bq;
        oj = j;
    } else {
        int jk = j - n_q;
        wcol = wk + (size_t)jk * (size_t)col_bytes;
        out = k;
        bias = bk;
        oj = jk;
    }

    float acc = dot_q5_0_col_xs(wcol, xs, n_rows, lane);
    if (lane == 0) {
        if (use_bias) acc += bias[oj];
        out[oj] = acc;
    }
}

// ── Fused dual GEMV (decode, Q4_K): stage x once (gate+up / Q+K on 3B+) ───

extern "C" __global__ void gemv_q4_k_pair(
    const unsigned char* __restrict__ wa,
    const unsigned char* __restrict__ wb,
    const float* __restrict__ x,
    float* __restrict__ out_a,
    float* __restrict__ out_b,
    int n_rows, int n_a, int n_b, int col_bytes,
    int use_bias,
    const float* __restrict__ ba,
    const float* __restrict__ bb
) {
    extern __shared__ float xs[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    for (int i = tid; i < n_rows; i += GEMV_THREADS) xs[i] = x[i];
    __syncthreads();

    int j = (int)blockIdx.x * GEMV_WARPS + warp;
    int n_tot = n_a + n_b;
    if (j >= n_tot) return;

    const unsigned char* wcol;
    float* out;
    const float* bias;
    int oj;
    if (j < n_a) {
        wcol = wa + (size_t)j * (size_t)col_bytes;
        out = out_a;
        bias = ba;
        oj = j;
    } else {
        int jb = j - n_a;
        wcol = wb + (size_t)jb * (size_t)col_bytes;
        out = out_b;
        bias = bb;
        oj = jb;
    }

    float acc = dot_q4_k_col_xs(wcol, xs, n_rows, lane);
    if (lane == 0) {
        if (use_bias) acc += bias[oj];
        out[oj] = acc;
    }
}

// Fused Q+K+V when all three are Q5_0 (stage x once; common on many Q4_K_M layers).
extern "C" __global__ void gemv_q5_0_qkv(
    const unsigned char* __restrict__ wq,
    const unsigned char* __restrict__ wk,
    const unsigned char* __restrict__ wv,
    const float* __restrict__ x,
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    int n_rows, int n_q, int n_k, int n_v, int col_bytes,
    int use_bias,
    const float* __restrict__ bq,
    const float* __restrict__ bk,
    const float* __restrict__ bv
) {
    extern __shared__ float xs[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    for (int i = tid; i < n_rows; i += GEMV_THREADS) xs[i] = x[i];
    __syncthreads();

    int j = (int)blockIdx.x * GEMV_WARPS + warp;
    int n_tot = n_q + n_k + n_v;
    if (j >= n_tot) return;

    const unsigned char* wcol;
    float* out;
    const float* bias;
    int oj;
    if (j < n_q) {
        wcol = wq + (size_t)j * (size_t)col_bytes;
        out = q;
        bias = bq;
        oj = j;
    } else if (j < n_q + n_k) {
        int jk = j - n_q;
        wcol = wk + (size_t)jk * (size_t)col_bytes;
        out = k;
        bias = bk;
        oj = jk;
    } else {
        int jv = j - n_q - n_k;
        wcol = wv + (size_t)jv * (size_t)col_bytes;
        out = v;
        bias = bv;
        oj = jv;
    }

    float acc = dot_q5_0_col_xs(wcol, xs, n_rows, lane);
    if (lane == 0) {
        if (use_bias) acc += bias[oj];
        out[oj] = acc;
    }
}
