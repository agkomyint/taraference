// Decode GEMV: bandwidth-first.
// - Stage x[] in shared memory (reused by all warps in the block)
// - 4 warps / block → 4 output columns, one weight stream each
// - Fused Q4/Q6 dequant·dot (no full float column buffer)

#define GEMV_WARPS 4
#define GEMV_THREADS (GEMV_WARPS * 32)

__device__ __forceinline__ float dot_q4_k_col_xs(
    const unsigned char* __restrict__ col,
    const float* __restrict__ xs,
    int n_rows,
    int lane
) {
    float acc = 0.f;
    const int nb = n_rows / 256;
    for (int bi = 0; bi < nb; bi++) {
        const unsigned char* base = col + bi * 144;
        float d = half_to_float((unsigned short)(base[0] | (base[1] << 8)));
        float minv = half_to_float((unsigned short)(base[2] | (base[3] << 8)));
        const unsigned char* scales = base + 4;
        const unsigned char* q = base + 16;
        const float* xb = xs + bi * 256;
        // 4×64: each lane owns one of 32 packed bytes → 2 outputs
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

__device__ __forceinline__ float dot_q6_k_col_xs(
    const unsigned char* __restrict__ col,
    const float* __restrict__ xs,
    int n_rows,
    int lane
) {
    float acc = 0.f;
    const int nb = n_rows / 256;
    for (int bi = 0; bi < nb; bi++) {
        const unsigned char* base = col + bi * 210;
        const unsigned char* ql = base;
        const unsigned char* qh = base + 128;
        const signed char* sc = (const signed char*)(base + 192);
        float d = half_to_float((unsigned short)(base[208] | (base[209] << 8)));
        const float* xb = xs + bi * 256;
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

extern "C" __global__ void gemv_q4_k(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_rows, int n_cols, int col_bytes
) {
    extern __shared__ float xs[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    // Stage activation vector once for the whole block.
    for (int i = tid; i < n_rows; i += GEMV_THREADS) {
        xs[i] = x[i];
    }
    __syncthreads();

    int j = (int)blockIdx.x * GEMV_WARPS + warp;
    if (j >= n_cols) return;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    float acc = dot_q4_k_col_xs(col, xs, n_rows, lane);
    if (lane == 0) out[j] = acc;
}

extern "C" __global__ void gemv_q6_k(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_rows, int n_cols, int col_bytes
) {
    extern __shared__ float xs[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    for (int i = tid; i < n_rows; i += GEMV_THREADS) {
        xs[i] = x[i];
    }
    __syncthreads();

    int j = (int)blockIdx.x * GEMV_WARPS + warp;
    if (j >= n_cols) return;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    float acc = dot_q6_k_col_xs(col, xs, n_rows, lane);
    if (lane == 0) out[j] = acc;
}

extern "C" __global__ void gemv_q8_0(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_rows, int n_cols, int col_bytes
) {
    extern __shared__ float xs[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    for (int i = tid; i < n_rows; i += GEMV_THREADS) {
        xs[i] = x[i];
    }
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
    if (lane == 0) out[j] = acc;
}
