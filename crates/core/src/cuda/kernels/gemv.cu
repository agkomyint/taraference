// Decode GEMV: bandwidth-first.
// - Stage x[] (or a row-slice) in shared memory
// - 32 warps / block → 32 output columns
// - Optional bias / residual via flags
// - Split-K path: grid (col_blocks, n_split) for better SM occupancy when n_rows is large

#define GEMV_WARPS 32
#define GEMV_THREADS (GEMV_WARPS * 32)
#define Q5_GEMV_WARPS 8
#define Q5_GEMV_THREADS (Q5_GEMV_WARPS * 32)
// Must match matmul.rs GEMV_SPLIT_MAX
#define GEMV_SPLIT_MAX 8

// Quantize one f32 activation row to Q8 in shared memory. Each warp owns one
// 32-value group at a time; scales are per group, matching MMVQ's Q8_1 dot path.
__device__ __forceinline__ void quantize_q8_smem(
    const float* __restrict__ x,
    signed char* __restrict__ q8,
    float* __restrict__ d8,
    int n_rows,
    int warp,
    int lane
) {
    const int nb = n_rows >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    for (int bi = warp; bi < nb; bi += warps_per_block) {
        const int i = (bi << 5) + lane;
        const float xi = x[i];
        float amax = fabsf(xi);
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, 16));
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, 8));
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, 4));
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, 2));
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, 1));
        amax = __shfl_sync(0xffffffffu, amax, 0);
        const float d = amax * (1.0f / 127.0f);
        q8[i] = amax == 0.f ? 0 : (signed char)__float2int_rn(xi / d);
        if (lane == 0) d8[bi] = d;
    }
}

// Quantize an activation once for all output-column blocks of a fused GEMV.
extern "C" __global__ void quantize_q8_global(
    const float* __restrict__ x,
    signed char* __restrict__ q8,
    float* __restrict__ d8,
    int n_rows
) {
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int bi = (int)blockIdx.x * 8 + warp;
    if (bi >= (n_rows >> 5)) return;
    const int i = (bi << 5) + lane;
    const float xi = x[i];
    float amax = fabsf(xi);
    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, 16));
    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, 8));
    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, 4));
    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, 2));
    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, 1));
    amax = __shfl_sync(0xffffffffu, amax, 0);
    const float d = amax * (1.0f / 127.0f);
    q8[i] = amax == 0.f ? 0 : (signed char)__float2int_rn(xi / d);
    if (lane == 0) d8[bi] = d;
}

// Q4_K block mapping follows ggml/llama.cpp's MIT-licensed CUDA MMVQ design;
// this implementation is specialized for taraference's column-major weights.
// Lanes 0..15 process one 256-value superblock while
// lanes 16..31 process the next; __dp4a performs four signed int8 products.
__device__ __forceinline__ float dot_q4_k_col_q8_range(
    const unsigned char* __restrict__ col,
    const signed char* __restrict__ q8,
    const float* __restrict__ d8,
    int bi0,
    int bi1,
    int lane
) {
    float acc = 0.f;
    const int iqs = 2 * (lane & 15);
    for (int bi = bi0 + (lane >> 4); bi < bi1; bi += 2) {
        const unsigned char* base = col + bi * 144;
        const float d = half_to_float((unsigned short)(base[0] | (base[1] << 8)));
        const float minv = half_to_float((unsigned short)(base[2] | (base[3] << 8)));
        const unsigned char* scales = base + 4;
        const unsigned char* qs = base + 16;

        const int q8_offset = 2 * ((iqs >> 1) >> 2); // 0, 2, 4, 6 Q8 blocks
        const int q4_offset = 16 * q8_offset + 4 * ((iqs >> 1) & 3);
        const int* q4 = reinterpret_cast<const int*>(qs + q4_offset);
        const int v0 = q4[0];
        const int v1 = q4[4];

        const unsigned short* sc16 = reinterpret_cast<const unsigned short*>(scales);
        unsigned short aux0, aux1;
        const int j = q8_offset >> 1;
        if (j < 2) {
            aux0 = sc16[j] & 0x3f3f;
            aux1 = sc16[j + 2] & 0x3f3f;
        } else {
            aux0 = ((sc16[j + 2] >> 0) & 0x0f0f) | ((sc16[j - 2] & 0xc0c0) >> 2);
            aux1 = ((sc16[j + 2] >> 4) & 0x0f0f) | ((sc16[j] & 0xc0c0) >> 2);
        }
        const unsigned char sc0 = (unsigned char)(aux0 & 0xff);
        const unsigned char sc1 = (unsigned char)(aux0 >> 8);
        const unsigned char m0 = (unsigned char)(aux1 & 0xff);
        const unsigned char m1 = (unsigned char)(aux1 >> 8);

        const int local_block = bi - bi0;
        const signed char* qb0 = q8 + (local_block * 8 + q8_offset) * 32;
        const signed char* qb1 = qb0 + 32;
        const int q8_word = (iqs >> 1) & 3;
        const int* u0 = reinterpret_cast<const int*>(qb0) + q8_word;
        const int* u1 = reinterpret_cast<const int*>(qb1) + q8_word;
        const int ua0 = u0[0], ua1 = u0[4];
        const int ub0 = u1[0], ub1 = u1[4];

        const int dot0 = __dp4a(v0 & 0x0f0f0f0f, ua0,
                         __dp4a(v1 & 0x0f0f0f0f, ua1, 0));
        const int dot1 = __dp4a((v0 >> 4) & 0x0f0f0f0f, ub0,
                         __dp4a((v1 >> 4) & 0x0f0f0f0f, ub1, 0));
        const int sum0 = __dp4a(0x01010101, ua0, __dp4a(0x01010101, ua1, 0));
        const int sum1 = __dp4a(0x01010101, ub0, __dp4a(0x01010101, ub1, 0));
        acc += d * (d8[local_block * 8 + q8_offset] * (float)(dot0 * (int)sc0)
                  + d8[local_block * 8 + q8_offset + 1] * (float)(dot1 * (int)sc1));
        acc -= minv * (d8[local_block * 8 + q8_offset] * (float)(sum0 * (int)m0)
                     + d8[local_block * 8 + q8_offset + 1] * (float)(sum1 * (int)m1));
    }
    return warp_sum(acc);
}

__device__ __forceinline__ float2 dot_q4_k_pair_q8_range(
    const unsigned char* __restrict__ col_a,
    const unsigned char* __restrict__ col_b,
    const signed char* __restrict__ q8,
    const float* __restrict__ d8,
    int bi0, int bi1, int lane
) {
    float acc[2] = {0.f, 0.f};
    const int iqs = 2 * (lane & 15);
    const int q8_offset = 2 * ((iqs >> 1) >> 2);
    const int q4_offset = 16 * q8_offset + 4 * ((iqs >> 1) & 3);
    const int q8_word = (iqs >> 1) & 3;
    for (int bi = bi0 + (lane >> 4); bi < bi1; bi += 2) {
        const int local_block = bi - bi0;
        const signed char* qb0 = q8 + (local_block * 8 + q8_offset) * 32;
        const int* u0 = reinterpret_cast<const int*>(qb0) + q8_word;
        const int* u1 = reinterpret_cast<const int*>(qb0 + 32) + q8_word;
        const int ua0 = u0[0], ua1 = u0[4];
        const int ub0 = u1[0], ub1 = u1[4];
        const int sum0 = __dp4a(0x01010101, ua0, __dp4a(0x01010101, ua1, 0));
        const int sum1 = __dp4a(0x01010101, ub0, __dp4a(0x01010101, ub1, 0));
        const float d80 = d8[local_block * 8 + q8_offset];
        const float d81 = d8[local_block * 8 + q8_offset + 1];
        const unsigned char* bases[2] = {
            col_a + bi * 144,
            col_b + bi * 144,
        };
        #pragma unroll
        for (int matrix = 0; matrix < 2; matrix++) {
            const unsigned char* base = bases[matrix];
            const float d = half_to_float((unsigned short)(base[0] | (base[1] << 8)));
            const float minv = half_to_float((unsigned short)(base[2] | (base[3] << 8)));
            const unsigned short* sc16 = reinterpret_cast<const unsigned short*>(base + 4);
            const int* q4 = reinterpret_cast<const int*>(base + 16 + q4_offset);
            const int v0 = q4[0];
            const int v1 = q4[4];
            unsigned short aux0, aux1;
            const int j = q8_offset >> 1;
            if (j < 2) {
                aux0 = sc16[j] & 0x3f3f;
                aux1 = sc16[j + 2] & 0x3f3f;
            } else {
                aux0 = ((sc16[j + 2] >> 0) & 0x0f0f) | ((sc16[j - 2] & 0xc0c0) >> 2);
                aux1 = ((sc16[j + 2] >> 4) & 0x0f0f) | ((sc16[j] & 0xc0c0) >> 2);
            }
            const int dot0 = __dp4a(v0 & 0x0f0f0f0f, ua0,
                             __dp4a(v1 & 0x0f0f0f0f, ua1, 0));
            const int dot1 = __dp4a((v0 >> 4) & 0x0f0f0f0f, ub0,
                             __dp4a((v1 >> 4) & 0x0f0f0f0f, ub1, 0));
            acc[matrix] += d * (d80 * (float)(dot0 * (int)(unsigned char)(aux0 & 0xff))
                              + d81 * (float)(dot1 * (int)(unsigned char)(aux0 >> 8)));
            acc[matrix] -= minv * (d80 * (float)(sum0 * (int)(unsigned char)(aux1 & 0xff))
                                 + d81 * (float)(sum1 * (int)(unsigned char)(aux1 >> 8)));
        }
    }
    float2 result;
    result.x = warp_sum(acc[0]);
    result.y = warp_sum(acc[1]);
    return result;
}

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

__device__ __forceinline__ float dot_q6_k_repack_q8_range(
    const unsigned char* __restrict__ col,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    int bi0,
    int bi1,
    int lane
) {
    float acc = 0.f;
    for (int bi = bi0; bi < bi1; bi++) {
        const unsigned char* base = col + bi * 276;
        const signed char* qw = reinterpret_cast<const signed char*>(base);
        const signed char* sc = reinterpret_cast<const signed char*>(base + 256);
        const float d = half_to_float((unsigned short)(base[272] | (base[273] << 8)));
        const int local = (bi - bi0) * 256;
        #pragma unroll
        for (int g = lane; g < 64; g += 32) {
            const int i = g * 4;
            const int wpack = *reinterpret_cast<const int*>(qw + i);
            const int apack = *reinterpret_cast<const int*>(xq + local + i);
            const int dot = __dp4a(wpack, apack, 0);
            acc += d * (float)sc[i >> 4] * xd[(local + i) >> 5] * (float)dot;
        }
    }
    return warp_sum(acc);
}

__device__ __forceinline__ float dot_q6_k_compact_q8_range(
    const unsigned char* __restrict__ col,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    int bi0, int bi1, int lane
) {
    float acc = 0.f;
    for (int bi = bi0; bi < bi1; bi++) {
        const unsigned char* base = col + bi * 212;
        const signed char* sc = reinterpret_cast<const signed char*>(base + 192);
        const float d = half_to_float((unsigned short)(base[208] | (base[209] << 8)));
        const int local = (bi - bi0) * 256;
        const int lane16 = lane & 15;
        const unsigned int ql = *reinterpret_cast<const unsigned int*>(base + 4 * lane);
        const int qh_word = 8 * (lane >> 4) + (lane & 7);
        unsigned int qh = *reinterpret_cast<const unsigned int*>(base + 128 + 4 * qh_word);
        qh >>= 2 * (lane16 >> 3);
        const int scale_offset = 8 * (lane >> 4) + (lane16 >> 2);
        const int q8_base = 4 * (lane >> 4) + (lane16 >> 3);
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            const unsigned int lo = (ql >> (4 * r)) & 0x0f0f0f0fu;
            const unsigned int hi = ((qh >> (4 * r)) << 4) & 0x30303030u;
            const unsigned int raw = lo | hi;
            const int wpack = __vsubss4((int)raw, 0x20202020);
            const int q8_block = q8_base + 2 * r;
            const int apack = *reinterpret_cast<const int*>(
                xq + local + 32 * q8_block + 4 * (lane & 7)
            );
            const int dot = __dp4a(wpack, apack, 0);
            acc += d * (float)sc[scale_offset + 4 * r]
                * xd[(local >> 5) + q8_block] * (float)dot;
        }
    }
    return warp_sum(acc);
}

__device__ __forceinline__ float dot_q6_k_repack_q8_global_range(
    const unsigned char* __restrict__ col,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    int bi0,
    int bi1,
    int lane
) {
    float acc = 0.f;
    for (int bi = bi0; bi < bi1; bi++) {
        const unsigned char* base = col + bi * 276;
        const signed char* qw = reinterpret_cast<const signed char*>(base);
        const signed char* sc = reinterpret_cast<const signed char*>(base + 256);
        const float d = half_to_float((unsigned short)(base[272] | (base[273] << 8)));
        const int global = bi * 256;
        #pragma unroll
        for (int g = lane; g < 64; g += 32) {
            const int i = g * 4;
            const int wpack = *reinterpret_cast<const int*>(qw + i);
            const int apack = *reinterpret_cast<const int*>(xq + global + i);
            const int dot = __dp4a(wpack, apack, 0);
            acc += d * (float)sc[i >> 4] * xd[(global + i) >> 5] * (float)dot;
        }
    }
    return warp_sum(acc);
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
    extern __shared__ unsigned char qsmem[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    signed char* xq = reinterpret_cast<signed char*>(qsmem);
    float* xd = reinterpret_cast<float*>(qsmem + n_rows);
    quantize_q8_smem(x, xq, xd, n_rows, warp, lane);
    __syncthreads();

    int j = (int)blockIdx.x * GEMV_WARPS + warp;
    if (j >= n_cols) return;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    float acc = dot_q4_k_col_q8_range(col, xq, xd, 0, n_rows / 256, lane);
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
    {
        int n4 = n_rows >> 2;
        for (int i = tid; i < n4; i += (int)blockDim.x) {
            float4 v = reinterpret_cast<const float4*>(x)[i];
            int o = i << 2;
            xs[o] = v.x; xs[o + 1] = v.y; xs[o + 2] = v.z; xs[o + 3] = v.w;
        }
        for (int i = (n4 << 2) + tid; i < n_rows; i += (int)blockDim.x) xs[i] = x[i];
    }
    __syncthreads();

    const int warps_per_block = (int)blockDim.x >> 5;
    int j = (int)blockIdx.x * warps_per_block + warp;
    if (j >= n_cols) return;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    float acc = dot_q6_k_col_xs(col, xs, n_rows, lane);
    if (lane == 0) {
        acc = gemv_apply_res(use_res, acc, out, j, residual);
        if (use_bias) acc += bias[j];
        out[j] = acc;
    }
}

extern "C" __global__ void gemv_q6_k_repack(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_rows, int n_cols, int col_bytes,
    int use_bias, const float* __restrict__ bias,
    int use_res, const float* __restrict__ residual
) {
    extern __shared__ unsigned char qsmem[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    signed char* xq = reinterpret_cast<signed char*>(qsmem);
    float* xd = reinterpret_cast<float*>(qsmem + n_rows);
    quantize_q8_smem(x, xq, xd, n_rows, warp, lane);
    __syncthreads();

    const int warps_per_block = (int)blockDim.x >> 5;
    const int j = (int)blockIdx.x * warps_per_block + warp;
    if (j >= n_cols) return;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    float acc = dot_q6_k_repack_q8_range(col, xq, xd, 0, n_rows / 256, lane);
    if (lane == 0) {
        acc = gemv_apply_res(use_res, acc, out, j, residual);
        if (use_bias) acc += bias[j];
        out[j] = acc;
    }
}

extern "C" __global__ void gemv_q4_k_global(
    const unsigned char* __restrict__ w,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ out,
    int n_rows, int n_cols, int col_bytes,
    int use_bias, const float* __restrict__ bias,
    int use_res, const float* __restrict__ residual
) {
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const int j = (int)blockIdx.x * warps_per_block + warp;
    if (j >= n_cols) return;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    float acc = dot_q4_k_col_q8_range(col, xq, xd, 0, n_rows / 256, lane);
    if (lane == 0) {
        acc = gemv_apply_res(use_res, acc, out, j, residual);
        if (use_bias) acc += bias[j];
        out[j] = acc;
    }
}

extern "C" __global__ void gemv_q6_k_repack_global(
    const unsigned char* __restrict__ w,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ out,
    int n_rows, int n_cols, int col_bytes,
    int use_bias, const float* __restrict__ bias,
    int use_res, const float* __restrict__ residual
) {
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const int j = (int)blockIdx.x * warps_per_block + warp;
    if (j >= n_cols) return;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    float acc = dot_q6_k_repack_q8_range(col, xq, xd, 0, n_rows / 256, lane);
    if (lane == 0) {
        acc = gemv_apply_res(use_res, acc, out, j, residual);
        if (use_bias) acc += bias[j];
        out[j] = acc;
    }
}

extern "C" __global__ void gemv_q6_k_compact_global(
    const unsigned char* __restrict__ w,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ out,
    int n_rows, int n_cols, int col_bytes,
    int use_bias, const float* __restrict__ bias,
    int use_res, const float* __restrict__ residual
) {
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const int j = (int)blockIdx.x * warps_per_block + warp;
    if (j >= n_cols) return;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    float acc = dot_q6_k_compact_q8_range(col, xq, xd, 0, n_rows / 256, lane);
    if (lane == 0) {
        acc = gemv_apply_res(use_res, acc, out, j, residual);
        if (use_bias) acc += bias[j];
        out[j] = acc;
    }
}

__device__ __forceinline__ int load_int_unaligned(const void* ptr) {
    int val;
    memcpy(&val, ptr, sizeof(int));
    return val;
}

// ── Block-major Q4_0 experts: layout [n_blocks][n_cols][18] ─────────────────
// Thread-per-column: for fixed bi, consecutive threads load consecutive columns
// → coalesced 18B-block streams (warp-per-col + lane-over-bi is the WRONG pattern).

__device__ __forceinline__ float dot_q4_0_bm_col_full(
    const unsigned char* __restrict__ w_bm,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    int n_cols,
    int j,
    int n_blocks
) {
    float acc = 0.f;
    // Unroll small block counts (d=640 → 20, expert_ff=1536 → 48).
    #pragma unroll 1
    for (int bi = 0; bi < n_blocks; bi++) {
        // Coalesced across threads in a warp: base_j, base_j+1, ... are 18B apart.
        const unsigned char* base = w_bm + ((size_t)bi * (size_t)n_cols + (size_t)j) * 18u;
        float dw = half_to_float((unsigned short)(base[0] | (base[1] << 8)));
        const unsigned char* qs = base + 2;
        const signed char* xv = xq + (bi << 5);
        int sum = 0;
        #pragma unroll
        for (int t = 0; t < 16; t += 4) {
            int qlo =
                (((int)(qs[t + 0] & 0x0f) - 8) & 0xff) |
                ((((int)(qs[t + 1] & 0x0f) - 8) & 0xff) << 8) |
                ((((int)(qs[t + 2] & 0x0f) - 8) & 0xff) << 16) |
                ((((int)(qs[t + 3] & 0x0f) - 8) & 0xff) << 24);
            int xlo = load_int_unaligned(xv + t);
            sum = __dp4a(qlo, xlo, sum);
            int qhi =
                (((int)(qs[t + 0] >> 4) - 8) & 0xff) |
                ((((int)(qs[t + 1] >> 4) - 8) & 0xff) << 8) |
                ((((int)(qs[t + 2] >> 4) - 8) & 0xff) << 16) |
                ((((int)(qs[t + 3] >> 4) - 8) & 0xff) << 24);
            int xhi = load_int_unaligned(xv + t + 16);
            sum = __dp4a(qhi, xhi, sum);
        }
        acc += dw * xd[bi] * (float)sum;
    }
    return acc;
}

/// Gate+up+SiLU, block-major Q4. Launch: grid=ceil(n_cols/threads), block=threads (e.g. 128).
extern "C" __global__ void gemv_q4_0_bm_expert_gate_up(
    const unsigned char* __restrict__ gate_bm,
    const unsigned char* __restrict__ up_bm,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ out_ff,
    const int* __restrict__ expert_ids,
    int slot,
    int n_rows,
    int n_cols_expert
) {
    const int j = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (j >= n_cols_expert) return;
    const int expert = expert_ids[slot];
    const int n_blocks = n_rows >> 5;
    // Per-expert BM slice: [n_blocks][n_cols_expert][18], experts consecutive.
    const size_t exp_off = (size_t)expert * (size_t)n_blocks * (size_t)n_cols_expert * 18u;
    const unsigned char* g = gate_bm + exp_off;
    const unsigned char* u = up_bm + exp_off;
    float ag = dot_q4_0_bm_col_full(g, xq, xd, n_cols_expert, j, n_blocks);
    float au = dot_q4_0_bm_col_full(u, xq, xd, n_cols_expert, j, n_blocks);
    out_ff[j] = (ag / (1.f + __expf(-ag))) * au;
}

extern "C" __global__ void gemv_q4_0_bm_expert_down_scale(
    const unsigned char* __restrict__ down_bm,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ residual_x,
    const int* __restrict__ expert_ids,
    const float* __restrict__ weights,
    int slot,
    int n_rows,
    int n_cols_expert
) {
    const int j = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (j >= n_cols_expert) return;
    const int expert = expert_ids[slot];
    const float w = weights[slot];
    const int n_blocks = n_rows >> 5;
    const size_t exp_off = (size_t)expert * (size_t)n_blocks * (size_t)n_cols_expert * 18u;
    const unsigned char* d = down_bm + exp_off;
    float acc = dot_q4_0_bm_col_full(d, xq, xd, n_cols_expert, j, n_blocks);
    residual_x[j] += w * acc;
}

// ── f16 column-major GEMV (optional expand; often more BW than Q4) ───────────
// Layout: for column j, halfs at w[j * n_rows + 0 .. n_rows).

__device__ __forceinline__ float dot_f16_col_f32(
    const unsigned short* __restrict__ col, // __half bit patterns
    const float* __restrict__ x,
    int n_rows,
    int lane
) {
    float acc = 0.f;
    for (int i = lane; i < n_rows; i += 32) {
        float w = half_to_float(col[i]);
        acc += w * x[i];
    }
    return warp_sum(acc);
}

/// Fused gate+up+SiLU for f16 packed experts (storage is u8 bytes of f16 LE).
extern "C" __global__ void gemv_f16_expert_gate_up(
    const unsigned char* __restrict__ gate_packed,
    const unsigned char* __restrict__ up_packed,
    const float* __restrict__ x,
    float* __restrict__ out_ff,
    const int* __restrict__ expert_ids,
    int slot,
    int n_rows,
    int n_cols_expert
) {
    const int expert = expert_ids[slot];
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const int j = (int)blockIdx.x * warps_per_block + warp;
    if (j >= n_cols_expert) return;
    const int packed_j = expert * n_cols_expert + j;
    const unsigned short* cg =
        reinterpret_cast<const unsigned short*>(gate_packed) + (size_t)packed_j * (size_t)n_rows;
    const unsigned short* cu =
        reinterpret_cast<const unsigned short*>(up_packed) + (size_t)packed_j * (size_t)n_rows;
    float ag = dot_f16_col_f32(cg, x, n_rows, lane);
    float au = dot_f16_col_f32(cu, x, n_rows, lane);
    if (lane == 0) {
        float silu = ag / (1.f + __expf(-ag));
        out_ff[j] = silu * au;
    }
}

/// Down f16 GEMV + scale residual. n_rows = expert_ff, n_cols_expert = n_embd.
extern "C" __global__ void gemv_f16_expert_down_scale(
    const unsigned char* __restrict__ down_packed,
    const float* __restrict__ hb,
    float* __restrict__ residual_x,
    const int* __restrict__ expert_ids,
    const float* __restrict__ weights,
    int slot,
    int n_rows,
    int n_cols_expert
) {
    const int expert = expert_ids[slot];
    const float w = weights[slot];
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const int j = (int)blockIdx.x * warps_per_block + warp;
    if (j >= n_cols_expert) return;
    const int packed_j = expert * n_cols_expert + j;
    const unsigned short* col =
        reinterpret_cast<const unsigned short*>(down_packed) + (size_t)packed_j * (size_t)n_rows;
    float acc = dot_f16_col_f32(col, hb, n_rows, lane);
    if (lane == 0) residual_x[j] += w * acc;
}

/// 4 warps/col f16 gate_up — more latency hiding for n_rows=640 / 1536.
extern "C" __global__ void gemv_f16_expert_gate_up_4w(
    const unsigned char* __restrict__ gate_packed,
    const unsigned char* __restrict__ up_packed,
    const float* __restrict__ x,
    float* __restrict__ out_ff,
    const int* __restrict__ expert_ids,
    int slot,
    int n_rows,
    int n_cols_expert
) {
    const int j = (int)blockIdx.x;
    if (j >= n_cols_expert) return;
    const int expert = expert_ids[slot];
    const int packed_j = expert * n_cols_expert + j;
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int chunk = (n_rows + 3) >> 2;
    const int i0 = warp * chunk;
    int i1 = i0 + chunk;
    if (i1 > n_rows) i1 = n_rows;
    const unsigned short* cg =
        reinterpret_cast<const unsigned short*>(gate_packed) + (size_t)packed_j * (size_t)n_rows;
    const unsigned short* cu =
        reinterpret_cast<const unsigned short*>(up_packed) + (size_t)packed_j * (size_t)n_rows;
    float ag = 0.f, au = 0.f;
    for (int i = i0 + lane; i < i1; i += 32) {
        ag += half_to_float(cg[i]) * x[i];
        au += half_to_float(cu[i]) * x[i];
    }
    ag = warp_sum(ag);
    au = warp_sum(au);
    __shared__ float pg[4], pu[4];
    if (lane == 0) {
        pg[warp] = ag;
        pu[warp] = au;
    }
    __syncthreads();
    if (warp == 0 && lane == 0) {
        float g = pg[0] + pg[1] + pg[2] + pg[3];
        float u = pu[0] + pu[1] + pu[2] + pu[3];
        out_ff[j] = (g / (1.f + __expf(-g))) * u;
    }
}

extern "C" __global__ void gemv_f16_expert_down_scale_4w(
    const unsigned char* __restrict__ down_packed,
    const float* __restrict__ hb,
    float* __restrict__ residual_x,
    const int* __restrict__ expert_ids,
    const float* __restrict__ weights,
    int slot,
    int n_rows,
    int n_cols_expert
) {
    const int j = (int)blockIdx.x;
    if (j >= n_cols_expert) return;
    const int expert = expert_ids[slot];
    const float w = weights[slot];
    const int packed_j = expert * n_cols_expert + j;
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int chunk = (n_rows + 3) >> 2;
    const int i0 = warp * chunk;
    int i1 = i0 + chunk;
    if (i1 > n_rows) i1 = n_rows;
    const unsigned short* col =
        reinterpret_cast<const unsigned short*>(down_packed) + (size_t)packed_j * (size_t)n_rows;
    float acc = 0.f;
    for (int i = i0 + lane; i < i1; i += 32)
        acc += half_to_float(col[i]) * hb[i];
    acc = warp_sum(acc);
    __shared__ float p[4];
    if (lane == 0) p[warp] = acc;
    __syncthreads();
    if (warp == 0 && lane == 0)
        residual_x[j] += w * (p[0] + p[1] + p[2] + p[3]);
}

// Q4_0: 32 vals / block. File format 18B (f16 d + 16B qs).
// Aligned decode format 32B: f16 d + 2B pad + 16B qs + pad (qs is 4-byte aligned → uint4 loads).
// Values = (nibble-8)*d. DP4A path for Ampere+.
__device__ __forceinline__ float dot_q4_0_col_q8_range_bsz(
    const unsigned char* __restrict__ col,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    int bi0,
    int bi1,
    int lane,
    int bsz
) {
    float acc = 0.f;
    for (int bi = bi0 + lane; bi < bi1; bi += 32) {
        const unsigned char* base = col + bi * bsz;
        float dw = half_to_float((unsigned short)(base[0] | (base[1] << 8)));
        const signed char* xv = xq + (bi << 5);
        int sum = 0;
        if (bsz >= 32) {
            // qs at +4 is 4-byte aligned — safe vector load
            const unsigned int* qs = reinterpret_cast<const unsigned int*>(base + 4);
            unsigned int q0 = qs[0], q1 = qs[1], q2 = qs[2], q3 = qs[3];
            unsigned int qwords[4] = {q0, q1, q2, q3};
            #pragma unroll
            for (int t = 0; t < 4; t++) {
                unsigned int qw = qwords[t];
                int qlo =
                    ((((int)(qw & 0x0fu)) - 8) & 0xff) |
                    (((((int)((qw >> 8) & 0x0fu)) - 8) & 0xff) << 8) |
                    (((((int)((qw >> 16) & 0x0fu)) - 8) & 0xff) << 16) |
                    (((((int)((qw >> 24) & 0x0fu)) - 8) & 0xff) << 24);
                int xlo = load_int_unaligned(xv + t * 4);
                sum = __dp4a(qlo, xlo, sum);
                int qhi =
                    ((((int)((qw >> 4) & 0x0fu)) - 8) & 0xff) |
                    (((((int)((qw >> 12) & 0x0fu)) - 8) & 0xff) << 8) |
                    (((((int)((qw >> 20) & 0x0fu)) - 8) & 0xff) << 16) |
                    (((((int)((qw >> 28) & 0x0fu)) - 8) & 0xff) << 24);
                int xhi = load_int_unaligned(xv + 16 + t * 4);
                sum = __dp4a(qhi, xhi, sum);
            }
        } else {
            const unsigned char* qs = base + 2;
            #pragma unroll
            for (int j = 0; j < 16; j += 4) {
                int qlo =
                    (((int)(qs[j + 0] & 0x0f) - 8) & 0xff) |
                    ((((int)(qs[j + 1] & 0x0f) - 8) & 0xff) << 8) |
                    ((((int)(qs[j + 2] & 0x0f) - 8) & 0xff) << 16) |
                    ((((int)(qs[j + 3] & 0x0f) - 8) & 0xff) << 24);
                int xlo = load_int_unaligned(xv + j);
                sum = __dp4a(qlo, xlo, sum);
                int qhi =
                    (((int)(qs[j + 0] >> 4) - 8) & 0xff) |
                    ((((int)(qs[j + 1] >> 4) - 8) & 0xff) << 8) |
                    ((((int)(qs[j + 2] >> 4) - 8) & 0xff) << 16) |
                    ((((int)(qs[j + 3] >> 4) - 8) & 0xff) << 24);
                int xhi = load_int_unaligned(xv + j + 16);
                sum = __dp4a(qhi, xhi, sum);
            }
        }
        acc += dw * xd[bi] * (float)sum;
    }
    return warp_sum(acc);
}

__device__ __forceinline__ float dot_q4_0_col_q8_range(
    const unsigned char* __restrict__ col,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    int bi0,
    int bi1,
    int lane
) {
    // Legacy 18B blocks (callers without col_bytes context).
    return dot_q4_0_col_q8_range_bsz(col, xq, xd, bi0, bi1, lane, 18);
}

__device__ __forceinline__ int q4_0_block_bytes(int n_rows, int col_bytes) {
    int nb = n_rows >> 5;
    return (nb > 0) ? (col_bytes / nb) : 18;
}

// Q4_0 GEMV with in-kernel activation quant (prefill / baseline; graph decode prefers global).
extern "C" __global__ void gemv_q4_0(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_rows, int n_cols, int col_bytes,
    int use_bias, const float* __restrict__ bias,
    int use_res, const float* __restrict__ residual
) {
    extern __shared__ unsigned char qsmem[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    signed char* xq = reinterpret_cast<signed char*>(qsmem);
    float* xd = reinterpret_cast<float*>(qsmem + n_rows);
    quantize_q8_smem(x, xq, xd, n_rows, warp, lane);
    __syncthreads();
    const int warps_per_block = (int)blockDim.x >> 5;
    const int j = (int)blockIdx.x * warps_per_block + warp;
    if (j >= n_cols) return;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    float acc = dot_q4_0_col_q8_range_bsz(col, xq, xd, 0, n_rows / 32, lane, q4_0_block_bytes(n_rows, col_bytes));
    if (lane == 0) {
        acc = gemv_apply_res(use_res, acc, out, j, residual);
        if (use_bias) acc += bias[j];
        out[j] = acc;
    }
}

extern "C" __global__ void gemv_q4_0_global(
    const unsigned char* __restrict__ w,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ out,
    int n_rows, int n_cols, int col_bytes,
    int use_bias, const float* __restrict__ bias,
    int use_res, const float* __restrict__ residual
) {
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const int j = (int)blockIdx.x * warps_per_block + warp;
    if (j >= n_cols) return;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    float acc = dot_q4_0_col_q8_range_bsz(col, xq, xd, 0, n_rows / 32, lane, q4_0_block_bytes(n_rows, col_bytes));
    if (lane == 0) {
        acc = gemv_apply_res(use_res, acc, out, j, residual);
        if (use_bias) acc += bias[j];
        out[j] = acc;
    }
}

extern "C" __global__ void gemv_q4_0_global_expert_slot(
    const unsigned char* __restrict__ w_packed,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ out,
    const int* __restrict__ expert_ids,
    int slot,
    int n_rows,
    int n_cols_expert,
    int col_bytes,
    int use_res,
    const float* __restrict__ residual
) {
    const int expert = expert_ids[slot];
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const int j = (int)blockIdx.x * warps_per_block + warp;
    if (j >= n_cols_expert) return;
    const int packed_j = expert * n_cols_expert + j;
    const unsigned char* col = w_packed + (size_t)packed_j * (size_t)col_bytes;
    float acc = dot_q4_0_col_q8_range_bsz(col, xq, xd, 0, n_rows / 32, lane, q4_0_block_bytes(n_rows, col_bytes));
    if (lane == 0) {
        acc = gemv_apply_res(use_res, acc, out, j, residual);
        out[j] = acc;
    }
}

/// Fused gate+up+SiLU*mul expert GEMV for one MoE slot.
/// Stage Q8 activation (xq + per-block scales) into dynamic smem when host
/// passes shared_mem_bytes >= n_rows + n_blocks*4 (aligned). All CTAs share
/// the same activation — smem cuts repeated global reads on short-K MoE (d=640).
__device__ __forceinline__ void stage_q8_act_smem(
    const signed char* __restrict__ xq_g,
    const float* __restrict__ xd_g,
    signed char* __restrict__ xq_s,
    float* __restrict__ xd_s,
    int n_rows,
    int n_blocks
) {
    const int tid = (int)threadIdx.x;
    const int nt = (int)blockDim.x;
    for (int i = tid; i < n_rows; i += nt) xq_s[i] = xq_g[i];
    for (int bi = tid; bi < n_blocks; bi += nt) xd_s[bi] = xd_g[bi];
    __syncthreads();
}

/// Writes out_ff[j] = silu(gate[j]) * up[j] (ready for down-proj).
/// Interleaves gate/up block loads in one K-loop for short-K latency hiding (d=640).
/// Optional dynamic smem: stage activation once per CTA (TARAFER_MOE_SMEM).
extern "C" __global__ void gemv_q4_0_global_expert_gate_up(
    const unsigned char* __restrict__ gate_packed,
    const unsigned char* __restrict__ up_packed,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ out_ff,
    float* __restrict__ out_unused,
    const int* __restrict__ expert_ids,
    int slot,
    int n_rows,
    int n_cols_expert,
    int col_bytes,
    int stage_act
) {
    const int n_blocks = n_rows >> 5;
    // When stage_act != 0, host passes align4(n_rows)+n_blocks*sizeof(float).
    extern __shared__ unsigned char act_smem[];
    signed char* xq_s = reinterpret_cast<signed char*>(act_smem);
    float* xd_s = reinterpret_cast<float*>(act_smem + ((n_rows + 3) & ~3));
    if (stage_act) stage_q8_act_smem(xq, xd, xq_s, xd_s, n_rows, n_blocks);
    const signed char* xq_use = stage_act ? xq_s : xq;
    const float* xd_use = stage_act ? xd_s : xd;
    const int expert = expert_ids[slot];
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const int j = (int)blockIdx.x * warps_per_block + warp;
    if (j >= n_cols_expert) return;
    const int packed_j = expert * n_cols_expert + j;
    const int bsz = q4_0_block_bytes(n_rows, col_bytes);
    const unsigned char* cg = gate_packed + (size_t)packed_j * (size_t)col_bytes;
    const unsigned char* cu = up_packed + (size_t)packed_j * (size_t)col_bytes;
    float ag = 0.f, au = 0.f;
    for (int bi = lane; bi < n_blocks; bi += 32) {
        const unsigned char* bg = cg + bi * bsz;
        const unsigned char* bu = cu + bi * bsz;
        float dwg = half_to_float((unsigned short)(bg[0] | (bg[1] << 8)));
        float dwu = half_to_float((unsigned short)(bu[0] | (bu[1] << 8)));
        const signed char* xv = xq_use + (bi << 5);
        const float xsc = xd_use[bi];
        int sumg = 0, sumu = 0;
        if (bsz >= 32) {
            const unsigned int* qsg = reinterpret_cast<const unsigned int*>(bg + 4);
            const unsigned int* qsu = reinterpret_cast<const unsigned int*>(bu + 4);
            #pragma unroll
            for (int t = 0; t < 4; t++) {
                unsigned int qwg = qsg[t], qwu = qsu[t];
                int xlo = load_int_unaligned(xv + t * 4);
                int xhi = load_int_unaligned(xv + 16 + t * 4);
                int qglo =
                    ((((int)(qwg & 0x0fu)) - 8) & 0xff) |
                    (((((int)((qwg >> 8) & 0x0fu)) - 8) & 0xff) << 8) |
                    (((((int)((qwg >> 16) & 0x0fu)) - 8) & 0xff) << 16) |
                    (((((int)((qwg >> 24) & 0x0fu)) - 8) & 0xff) << 24);
                int qulo =
                    ((((int)(qwu & 0x0fu)) - 8) & 0xff) |
                    (((((int)((qwu >> 8) & 0x0fu)) - 8) & 0xff) << 8) |
                    (((((int)((qwu >> 16) & 0x0fu)) - 8) & 0xff) << 16) |
                    (((((int)((qwu >> 24) & 0x0fu)) - 8) & 0xff) << 24);
                sumg = __dp4a(qglo, xlo, sumg);
                sumu = __dp4a(qulo, xlo, sumu);
                int qghi =
                    ((((int)((qwg >> 4) & 0x0fu)) - 8) & 0xff) |
                    (((((int)((qwg >> 12) & 0x0fu)) - 8) & 0xff) << 8) |
                    (((((int)((qwg >> 20) & 0x0fu)) - 8) & 0xff) << 16) |
                    (((((int)((qwg >> 28) & 0x0fu)) - 8) & 0xff) << 24);
                int quhi =
                    ((((int)((qwu >> 4) & 0x0fu)) - 8) & 0xff) |
                    (((((int)((qwu >> 12) & 0x0fu)) - 8) & 0xff) << 8) |
                    (((((int)((qwu >> 20) & 0x0fu)) - 8) & 0xff) << 16) |
                    (((((int)((qwu >> 28) & 0x0fu)) - 8) & 0xff) << 24);
                sumg = __dp4a(qghi, xhi, sumg);
                sumu = __dp4a(quhi, xhi, sumu);
            }
        } else {
            const unsigned char* qsg = bg + 2;
            const unsigned char* qsu = bu + 2;
            #pragma unroll
            for (int t = 0; t < 16; t += 4) {
                int xlo = load_int_unaligned(xv + t);
                int xhi = load_int_unaligned(xv + t + 16);
                int qglo =
                    (((int)(qsg[t + 0] & 0x0f) - 8) & 0xff) |
                    ((((int)(qsg[t + 1] & 0x0f) - 8) & 0xff) << 8) |
                    ((((int)(qsg[t + 2] & 0x0f) - 8) & 0xff) << 16) |
                    ((((int)(qsg[t + 3] & 0x0f) - 8) & 0xff) << 24);
                int qulo =
                    (((int)(qsu[t + 0] & 0x0f) - 8) & 0xff) |
                    ((((int)(qsu[t + 1] & 0x0f) - 8) & 0xff) << 8) |
                    ((((int)(qsu[t + 2] & 0x0f) - 8) & 0xff) << 16) |
                    ((((int)(qsu[t + 3] & 0x0f) - 8) & 0xff) << 24);
                sumg = __dp4a(qglo, xlo, sumg);
                sumu = __dp4a(qulo, xlo, sumu);
                int qghi =
                    (((int)(qsg[t + 0] >> 4) - 8) & 0xff) |
                    ((((int)(qsg[t + 1] >> 4) - 8) & 0xff) << 8) |
                    ((((int)(qsg[t + 2] >> 4) - 8) & 0xff) << 16) |
                    ((((int)(qsg[t + 3] >> 4) - 8) & 0xff) << 24);
                int quhi =
                    (((int)(qsu[t + 0] >> 4) - 8) & 0xff) |
                    ((((int)(qsu[t + 1] >> 4) - 8) & 0xff) << 8) |
                    ((((int)(qsu[t + 2] >> 4) - 8) & 0xff) << 16) |
                    ((((int)(qsu[t + 3] >> 4) - 8) & 0xff) << 24);
                sumg = __dp4a(qghi, xhi, sumg);
                sumu = __dp4a(quhi, xhi, sumu);
            }
        }
        ag += dwg * xsc * (float)sumg;
        au += dwu * xsc * (float)sumu;
    }
    ag = warp_sum(ag);
    au = warp_sum(au);
    if (lane == 0) {
        float silu = ag / (1.f + __expf(-ag));
        out_ff[j] = silu * au;
        (void)out_unused;
    }
}

/// Gate+up+SiLU for 32 intermediate cols per block, then Q8-quantize the tile in-smem.
/// Launch: grid = ceil(n_cols_expert/32), block = 32 warps (1024 threads).
/// Writes q8_out[32*bi + lane] and d8_out[bi] — drops a separate quantize launch before down.
extern "C" __global__ void gemv_q4_0_global_expert_gate_up_q8(
    const unsigned char* __restrict__ gate_packed,
    const unsigned char* __restrict__ up_packed,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    signed char* __restrict__ q8_out,
    float* __restrict__ d8_out,
    const int* __restrict__ expert_ids,
    int slot,
    int n_rows,
    int n_cols_expert,
    int col_bytes
) {
    // Exactly 32 warps: one output column each in this tile.
    const int expert = expert_ids[slot];
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int bi = (int)blockIdx.x; // tile index over intermediate/32
    const int j = (bi << 5) + warp;
    __shared__ float tile[32];

    float val = 0.f;
    if (j < n_cols_expert) {
        const int packed_j = expert * n_cols_expert + j;
        const unsigned char* cg = gate_packed + (size_t)packed_j * (size_t)col_bytes;
        const unsigned char* cu = up_packed + (size_t)packed_j * (size_t)col_bytes;
        float ag = dot_q4_0_col_q8_range_bsz(cg, xq, xd, 0, n_rows / 32, lane, q4_0_block_bytes(n_rows, col_bytes));
        float au = dot_q4_0_col_q8_range_bsz(cu, xq, xd, 0, n_rows / 32, lane, q4_0_block_bytes(n_rows, col_bytes));
        if (lane == 0) {
            val = (ag / (1.f + __expf(-ag))) * au;
            tile[warp] = val;
        }
    } else if (lane == 0) {
        tile[warp] = 0.f;
    }
    __syncthreads();

    // Q8 quantize the 32-wide tile (one activation block).
    float xi = tile[lane];
    float amax = fabsf(xi);
    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, 16));
    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, 8));
    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, 4));
    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, 2));
    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, 1));
    amax = __shfl_sync(0xffffffffu, amax, 0);
    const float d = amax * (1.0f / 127.0f);
    const int base = bi << 5;
    q8_out[base + lane] = (amax == 0.f || d == 0.f)
        ? 0
        : (signed char)__float2int_rn(xi / d);
    if (lane == 0) d8_out[bi] = d;
}

/// Down GEMV + scale-add residual: x[i] += weights[slot] * (W_down @ hb)[i]
extern "C" __global__ void gemv_q4_0_global_expert_down_scale(
    const unsigned char* __restrict__ down_packed,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ residual_x,
    const int* __restrict__ expert_ids,
    const float* __restrict__ weights,
    int slot,
    int n_rows,
    int n_cols_expert,
    int col_bytes,
    int stage_act
) {
    // When stage_act != 0, host passes align4(n_rows)+(n_rows/32)*4.
    const int n_blocks = n_rows >> 5;
    extern __shared__ unsigned char act_smem[];
    signed char* xq_s = reinterpret_cast<signed char*>(act_smem);
    float* xd_s = reinterpret_cast<float*>(act_smem + ((n_rows + 3) & ~3));
    if (stage_act) stage_q8_act_smem(xq, xd, xq_s, xd_s, n_rows, n_blocks);
    const signed char* xq_use = stage_act ? xq_s : xq;
    const float* xd_use = stage_act ? xd_s : xd;

    const int expert = expert_ids[slot];
    const float w = weights[slot];
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const int j = (int)blockIdx.x * warps_per_block + warp;
    if (j >= n_cols_expert) return;
    const int packed_j = expert * n_cols_expert + j;
    const unsigned char* col = down_packed + (size_t)packed_j * (size_t)col_bytes;
    float acc = dot_q4_0_col_q8_range_bsz(
        col, xq_use, xd_use, 0, n_blocks, lane, q4_0_block_bytes(n_rows, col_bytes));
    if (lane == 0) residual_x[j] += w * acc;
}

/// 4 warps per column: better BW latency hiding when n_rows/32 is modest (d=640 → 20 blocks).
/// Launch: grid=n_cols_expert, block=128 (4 warps).
extern "C" __global__ void gemv_q4_0_global_expert_gate_up_4w(
    const unsigned char* __restrict__ gate_packed,
    const unsigned char* __restrict__ up_packed,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ out_ff,
    const int* __restrict__ expert_ids,
    int slot,
    int n_rows,
    int n_cols_expert,
    int col_bytes
) {
    const int j = (int)blockIdx.x;
    if (j >= n_cols_expert) return;
    const int expert = expert_ids[slot];
    const int packed_j = expert * n_cols_expert + j;
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5; // 0..3
    const int n_blocks = n_rows / 32;
    const int chunk = (n_blocks + 3) >> 2;
    const int bi0 = warp * chunk;
    int bi1 = bi0 + chunk;
    if (bi1 > n_blocks) bi1 = n_blocks;

    const unsigned char* cg = gate_packed + (size_t)packed_j * (size_t)col_bytes;
    const unsigned char* cu = up_packed + (size_t)packed_j * (size_t)col_bytes;
    float ag = (bi0 < bi1) ? dot_q4_0_col_q8_range_bsz(cg, xq, xd, bi0, bi1, lane, q4_0_block_bytes(n_rows, col_bytes)) : 0.f;
    float au = (bi0 < bi1) ? dot_q4_0_col_q8_range_bsz(cu, xq, xd, bi0, bi1, lane, q4_0_block_bytes(n_rows, col_bytes)) : 0.f;
    __shared__ float pg[4], pu[4];
    if (lane == 0) {
        pg[warp] = ag;
        pu[warp] = au;
    }
    __syncthreads();
    if (warp == 0 && lane == 0) {
        float g = pg[0] + pg[1] + pg[2] + pg[3];
        float u = pu[0] + pu[1] + pu[2] + pu[3];
        out_ff[j] = (g / (1.f + __expf(-g))) * u;
    }
}

extern "C" __global__ void gemv_q4_0_global_expert_down_scale_4w(
    const unsigned char* __restrict__ down_packed,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ residual_x,
    const int* __restrict__ expert_ids,
    const float* __restrict__ weights,
    int slot,
    int n_rows,
    int n_cols_expert,
    int col_bytes
) {
    const int j = (int)blockIdx.x;
    if (j >= n_cols_expert) return;
    const int expert = expert_ids[slot];
    const float w = weights[slot];
    const int packed_j = expert * n_cols_expert + j;
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int n_blocks = n_rows / 32;
    const int chunk = (n_blocks + 3) >> 2;
    const int bi0 = warp * chunk;
    int bi1 = bi0 + chunk;
    if (bi1 > n_blocks) bi1 = n_blocks;
    const unsigned char* col = down_packed + (size_t)packed_j * (size_t)col_bytes;
    float acc = (bi0 < bi1) ? dot_q4_0_col_q8_range_bsz(col, xq, xd, bi0, bi1, lane, q4_0_block_bytes(n_rows, col_bytes)) : 0.f;
    __shared__ float p[4];
    if (lane == 0) p[warp] = acc;
    __syncthreads();
    if (warp == 0 && lane == 0)
        residual_x[j] += w * (p[0] + p[1] + p[2] + p[3]);
}

/// 2 warps per output column: better latency hiding on tall expert_ff (1536).
/// Launch: grid=n_cols_expert, block=64 (exactly 2 warps).
extern "C" __global__ void gemv_q4_0_global_expert_gate_up_2w(
    const unsigned char* __restrict__ gate_packed,
    const unsigned char* __restrict__ up_packed,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ out_ff,
    const int* __restrict__ expert_ids,
    int slot,
    int n_rows,
    int n_cols_expert,
    int col_bytes
) {
    const int j = (int)blockIdx.x;
    if (j >= n_cols_expert) return;
    const int expert = expert_ids[slot];
    const int packed_j = expert * n_cols_expert + j;
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5; // 0 or 1
    const int n_blocks = n_rows / 32;
    const int mid = n_blocks >> 1;
    const int bi0 = (warp == 0) ? 0 : mid;
    const int bi1 = (warp == 0) ? mid : n_blocks;

    const unsigned char* cg = gate_packed + (size_t)packed_j * (size_t)col_bytes;
    const unsigned char* cu = up_packed + (size_t)packed_j * (size_t)col_bytes;
    float ag = dot_q4_0_col_q8_range_bsz(cg, xq, xd, bi0, bi1, lane, q4_0_block_bytes(n_rows, col_bytes));
    float au = dot_q4_0_col_q8_range_bsz(cu, xq, xd, bi0, bi1, lane, q4_0_block_bytes(n_rows, col_bytes));

    __shared__ float pg[2], pu[2];
    if (lane == 0) {
        pg[warp] = ag;
        pu[warp] = au;
    }
    __syncthreads();
    if (warp == 0 && lane == 0) {
        float g = pg[0] + pg[1];
        float u = pu[0] + pu[1];
        out_ff[j] = (g / (1.f + __expf(-g))) * u;
    }
}

/// 2 warps per column for expert down + residual scale-add.
/// Launch: grid=n_cols_expert (=n_embd), block=64.
extern "C" __global__ void gemv_q4_0_global_expert_down_scale_2w(
    const unsigned char* __restrict__ down_packed,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ residual_x,
    const int* __restrict__ expert_ids,
    const float* __restrict__ weights,
    int slot,
    int n_rows,
    int n_cols_expert,
    int col_bytes
) {
    const int j = (int)blockIdx.x;
    if (j >= n_cols_expert) return;
    const int expert = expert_ids[slot];
    const float w = weights[slot];
    const int packed_j = expert * n_cols_expert + j;
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int n_blocks = n_rows / 32;
    const int mid = n_blocks >> 1;
    const int bi0 = (warp == 0) ? 0 : mid;
    const int bi1 = (warp == 0) ? mid : n_blocks;

    const unsigned char* col = down_packed + (size_t)packed_j * (size_t)col_bytes;
    float acc = dot_q4_0_col_q8_range_bsz(col, xq, xd, bi0, bi1, lane, q4_0_block_bytes(n_rows, col_bytes));
    __shared__ float p[2];
    if (lane == 0) p[warp] = acc;
    __syncthreads();
    if (warp == 0 && lane == 0) residual_x[j] += w * (p[0] + p[1]);
}

/// 2 warps per column fused Q+K+V for Q4_0 MoE packs (n_rows often 640).
/// Launch: grid = n_q+n_k+n_v, block=64.
extern "C" __global__ void gemv_q4_0_global_qkv_2w(
    const unsigned char* __restrict__ wq,
    const unsigned char* __restrict__ wk,
    const unsigned char* __restrict__ wv,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    int n_rows, int n_q, int n_k, int n_v, int col_bytes,
    int use_bias,
    const float* __restrict__ bq,
    const float* __restrict__ bk,
    const float* __restrict__ bv
) {
    const int j = (int)blockIdx.x;
    const int n_tot = n_q + n_k + n_v;
    if (j >= n_tot) return;
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int n_blocks = n_rows / 32;
    const int mid = n_blocks >> 1;
    const int bi0 = (warp == 0) ? 0 : mid;
    const int bi1 = (warp == 0) ? mid : n_blocks;

    const unsigned char* wcol;
    float* out;
    const float* bias;
    int oj;
    if (j < n_q) {
        wcol = wq + (size_t)j * (size_t)col_bytes;
        out = q; bias = bq; oj = j;
    } else if (j < n_q + n_k) {
        oj = j - n_q;
        wcol = wk + (size_t)oj * (size_t)col_bytes;
        out = k; bias = bk;
    } else {
        oj = j - n_q - n_k;
        wcol = wv + (size_t)oj * (size_t)col_bytes;
        out = v; bias = bv;
    }
    float acc = dot_q4_0_col_q8_range_bsz(wcol, xq, xd, bi0, bi1, lane, q4_0_block_bytes(n_rows, col_bytes));
    __shared__ float p[2];
    if (lane == 0) p[warp] = acc;
    __syncthreads();
    if (warp == 0 && lane == 0) {
        float a = p[0] + p[1];
        if (use_bias) a += bias[oj];
        out[oj] = a;
    }
}

// ── Q4_0 fused decode paths (MoE packs): stage activation once, multi-mat GEMV ──

/// Fused Q+K+V for Tara MoE Q4_0 packs (quantize x once globally).
extern "C" __global__ void gemv_q4_0_global_qkv(
    const unsigned char* __restrict__ wq,
    const unsigned char* __restrict__ wk,
    const unsigned char* __restrict__ wv,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    int n_rows, int n_q, int n_k, int n_v, int col_bytes,
    int use_bias,
    const float* __restrict__ bq,
    const float* __restrict__ bk,
    const float* __restrict__ bv
) {
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const int j = (int)blockIdx.x * warps_per_block + warp;
    const int n_tot = n_q + n_k + n_v;
    if (j >= n_tot) return;

    const unsigned char* wcol;
    float* out;
    const float* bias;
    int oj;
    if (j < n_q) {
        wcol = wq + (size_t)j * (size_t)col_bytes;
        out = q; bias = bq; oj = j;
    } else if (j < n_q + n_k) {
        oj = j - n_q;
        wcol = wk + (size_t)oj * (size_t)col_bytes;
        out = k; bias = bk;
    } else {
        oj = j - n_q - n_k;
        wcol = wv + (size_t)oj * (size_t)col_bytes;
        out = v; bias = bv;
    }
    float acc = dot_q4_0_col_q8_range_bsz(wcol, xq, xd, 0, n_rows / 32, lane, q4_0_block_bytes(n_rows, col_bytes));
    if (lane == 0) {
        if (use_bias) acc += bias[oj];
        out[oj] = acc;
    }
}

/// Fused dual GEMV for equal-shape Q4_0 mats (gate+up or Q+K). Quantize once.
extern "C" __global__ void gemv_q4_0_global_pair(
    const unsigned char* __restrict__ wa,
    const unsigned char* __restrict__ wb,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ out_a,
    float* __restrict__ out_b,
    int n_rows, int n_cols, int col_bytes
) {
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const int j = (int)blockIdx.x * warps_per_block + warp;
    if (j >= n_cols) return;
    const unsigned char* ca = wa + (size_t)j * (size_t)col_bytes;
    const unsigned char* cb = wb + (size_t)j * (size_t)col_bytes;
    float aa = dot_q4_0_col_q8_range_bsz(ca, xq, xd, 0, n_rows / 32, lane, q4_0_block_bytes(n_rows, col_bytes));
    float ab = dot_q4_0_col_q8_range_bsz(cb, xq, xd, 0, n_rows / 32, lane, q4_0_block_bytes(n_rows, col_bytes));
    if (lane == 0) {
        out_a[j] = aa;
        out_b[j] = ab;
    }
}

/// Dense Q4_0 gate+up+SiLU*mul (same math as expert path, no expert index).
extern "C" __global__ void gemv_q4_0_global_ffn(
    const unsigned char* __restrict__ gate,
    const unsigned char* __restrict__ up,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ out_ff,
    int n_rows, int n_cols, int col_bytes
) {
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const int j = (int)blockIdx.x * warps_per_block + warp;
    if (j >= n_cols) return;
    const unsigned char* cg = gate + (size_t)j * (size_t)col_bytes;
    const unsigned char* cu = up + (size_t)j * (size_t)col_bytes;
    float ag = dot_q4_0_col_q8_range_bsz(cg, xq, xd, 0, n_rows / 32, lane, q4_0_block_bytes(n_rows, col_bytes));
    float au = dot_q4_0_col_q8_range_bsz(cu, xq, xd, 0, n_rows / 32, lane, q4_0_block_bytes(n_rows, col_bytes));
    if (lane == 0) {
        float silu = ag / (1.f + __expf(-ag));
        out_ff[j] = silu * au;
    }
}

/// MoE expert gate+up from float x: quantize once in smem (drops host quantize launch).
extern "C" __global__ void gemv_q4_0_expert_gate_up_f32(
    const unsigned char* __restrict__ gate_packed,
    const unsigned char* __restrict__ up_packed,
    const float* __restrict__ x,
    float* __restrict__ out_ff,
    const int* __restrict__ expert_ids,
    int slot,
    int n_rows,
    int n_cols_expert,
    int col_bytes
) {
    extern __shared__ unsigned char qsmem[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    signed char* xq = reinterpret_cast<signed char*>(qsmem);
    float* xd = reinterpret_cast<float*>(qsmem + n_rows);
    quantize_q8_smem(x, xq, xd, n_rows, warp, lane);
    __syncthreads();

    const int expert = expert_ids[slot];
    const int warps_per_block = (int)blockDim.x >> 5;
    const int j = (int)blockIdx.x * warps_per_block + warp;
    if (j >= n_cols_expert) return;
    const int packed_j = expert * n_cols_expert + j;
    const unsigned char* cg = gate_packed + (size_t)packed_j * (size_t)col_bytes;
    const unsigned char* cu = up_packed + (size_t)packed_j * (size_t)col_bytes;
    float ag = dot_q4_0_col_q8_range_bsz(cg, xq, xd, 0, n_rows / 32, lane, q4_0_block_bytes(n_rows, col_bytes));
    float au = dot_q4_0_col_q8_range_bsz(cu, xq, xd, 0, n_rows / 32, lane, q4_0_block_bytes(n_rows, col_bytes));
    if (lane == 0) {
        float silu = ag / (1.f + __expf(-ag));
        out_ff[j] = silu * au;
    }
}

/// MoE expert down from float hb + residual scale-add (smem quantize once).
extern "C" __global__ void gemv_q4_0_expert_down_scale_f32(
    const unsigned char* __restrict__ down_packed,
    const float* __restrict__ hb,
    float* __restrict__ residual_x,
    const int* __restrict__ expert_ids,
    const float* __restrict__ weights,
    int slot,
    int n_rows,
    int n_cols_expert,
    int col_bytes
) {
    extern __shared__ unsigned char qsmem[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    signed char* xq = reinterpret_cast<signed char*>(qsmem);
    float* xd = reinterpret_cast<float*>(qsmem + n_rows);
    quantize_q8_smem(hb, xq, xd, n_rows, warp, lane);
    __syncthreads();

    const int expert = expert_ids[slot];
    const float w = weights[slot];
    const int warps_per_block = (int)blockDim.x >> 5;
    const int j = (int)blockIdx.x * warps_per_block + warp;
    if (j >= n_cols_expert) return;
    const int packed_j = expert * n_cols_expert + j;
    const unsigned char* col = down_packed + (size_t)packed_j * (size_t)col_bytes;
    float acc = dot_q4_0_col_q8_range_bsz(col, xq, xd, 0, n_rows / 32, lane, q4_0_block_bytes(n_rows, col_bytes));
    if (lane == 0) residual_x[j] += w * acc;
}

// Fast path for Q5→Q8 hybrid weights: activation quantized once globally.
// Q8_0 block = f16 d + 32×int8; activation groups match (32 vals, scale xd[bi]).
__device__ __forceinline__ float dot_q8_0_col_q8_range(
    const unsigned char* __restrict__ col,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    int bi0,
    int bi1,
    int lane
) {
    float acc = 0.f;
    for (int bi = bi0 + lane; bi < bi1; bi += 32) {
        const unsigned char* base = col + bi * 34;
        float dw = half_to_float((unsigned short)(base[0] | (base[1] << 8)));
        const signed char* qs = (const signed char*)(base + 2);
        const signed char* xv = xq + (bi << 5);
        int sum = 0;
        #pragma unroll
        for (int t = 0; t < 32; t += 4) {
            int w4 = load_int_unaligned(qs + t);
            int x4 = load_int_unaligned(xv + t);
            sum = __dp4a(w4, x4, sum);
        }
        acc += dw * xd[bi] * (float)sum;
    }
    return warp_sum(acc);
}

// Q8_0 × f32 (quantize x to smem, compute with hardware __dp4a int8 dot products).
extern "C" __global__ void gemv_q8_0(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_rows, int n_cols, int col_bytes,
    int use_bias, const float* __restrict__ bias,
    int use_res, const float* __restrict__ residual
) {
    extern __shared__ unsigned char qsmem[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    signed char* xq = reinterpret_cast<signed char*>(qsmem);
    float* xd = reinterpret_cast<float*>(qsmem + n_rows);
    quantize_q8_smem(x, xq, xd, n_rows, warp, lane);
    __syncthreads();

    const int warps_per_block = (int)blockDim.x >> 5;
    const int j = (int)blockIdx.x * warps_per_block + warp;
    if (j >= n_cols) return;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    float acc = dot_q8_0_col_q8_range(col, xq, xd, 0, n_rows / 32, lane);
    if (lane == 0) {
        acc = gemv_apply_res(use_res, acc, out, j, residual);
        if (use_bias) acc += bias[j];
        out[j] = acc;
    }
}

extern "C" __global__ void gemv_q8_0_global(
    const unsigned char* __restrict__ w,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ out,
    int n_rows, int n_cols, int col_bytes,
    int use_bias, const float* __restrict__ bias,
    int use_res, const float* __restrict__ residual
) {
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const int j = (int)blockIdx.x * warps_per_block + warp;
    if (j >= n_cols) return;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    float acc = dot_q8_0_col_q8_range(col, xq, xd, 0, n_rows / 32, lane);
    if (lane == 0) {
        acc = gemv_apply_res(use_res, acc, out, j, residual);
        if (use_bias) acc += bias[j];
        out[j] = acc;
    }
}

/// Fused Q8 gate+up+SiLU expert slot → out_ff = silu(gate)*up.
extern "C" __global__ void gemv_q8_0_global_expert_gate_up(
    const unsigned char* __restrict__ gate_packed,
    const unsigned char* __restrict__ up_packed,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ out_ff,
    float* __restrict__ out_unused,
    const int* __restrict__ expert_ids,
    int slot,
    int n_rows,
    int n_cols_expert,
    int col_bytes
) {
    const int expert = expert_ids[slot];
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const int j = (int)blockIdx.x * warps_per_block + warp;
    if (j >= n_cols_expert) return;
    const int packed_j = expert * n_cols_expert + j;
    const unsigned char* cg = gate_packed + (size_t)packed_j * (size_t)col_bytes;
    const unsigned char* cu = up_packed + (size_t)packed_j * (size_t)col_bytes;
    float ag = dot_q8_0_col_q8_range(cg, xq, xd, 0, n_rows / 32, lane);
    float au = dot_q8_0_col_q8_range(cu, xq, xd, 0, n_rows / 32, lane);
    if (lane == 0) {
        float silu = ag / (1.f + __expf(-ag));
        out_ff[j] = silu * au;
        (void)out_unused;
    }
}

extern "C" __global__ void gemv_q8_0_global_expert_down_scale(
    const unsigned char* __restrict__ down_packed,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ residual_x,
    const int* __restrict__ expert_ids,
    const float* __restrict__ weights,
    int slot,
    int n_rows,
    int n_cols_expert,
    int col_bytes
) {
    const int expert = expert_ids[slot];
    const float w = weights[slot];
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const int j = (int)blockIdx.x * warps_per_block + warp;
    if (j >= n_cols_expert) return;
    const int packed_j = expert * n_cols_expert + j;
    const unsigned char* col = down_packed + (size_t)packed_j * (size_t)col_bytes;
    float acc = dot_q8_0_col_q8_range(col, xq, xd, 0, n_rows / 32, lane);
    if (lane == 0) residual_x[j] += w * acc;
}

/// Sparse MoE expert GEMV: columns live in a packed buffer
/// `w_packed[expert * n_cols_expert + j]`. Expert id from device `expert_ids[slot]`.
/// CUDA-graph safe (static base ptr + device-side expert index).
extern "C" __global__ void gemv_q8_0_global_expert_slot(
    const unsigned char* __restrict__ w_packed,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ out,
    const int* __restrict__ expert_ids,
    int slot,
    int n_rows,
    int n_cols_expert,
    int col_bytes,
    int use_res,
    const float* __restrict__ residual
) {
    const int expert = expert_ids[slot];
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const int j = (int)blockIdx.x * warps_per_block + warp;
    if (j >= n_cols_expert) return;
    const int packed_j = expert * n_cols_expert + j;
    const unsigned char* col = w_packed + (size_t)packed_j * (size_t)col_bytes;
    float acc = dot_q8_0_col_q8_range(col, xq, xd, 0, n_rows / 32, lane);
    if (lane == 0) {
        acc = gemv_apply_res(use_res, acc, out, j, residual);
        out[j] = acc;
    }
}

extern "C" __global__ void gemv_q8_0_global_splitk(
    const unsigned char* __restrict__ w,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ partial,
    int n_rows, int n_cols, int col_bytes, int n_split
) {
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const int s = (int)blockIdx.y;
    if (s >= n_split) return;
    const int j = (int)blockIdx.x * warps_per_block + warp;
    if (j >= n_cols) return;
    const int nb = n_rows / 32;
    const int bi0 = (nb * s) / n_split;
    const int bi1 = (nb * (s + 1)) / n_split;
    if (bi0 >= bi1) {
        if (lane == 0) partial[(size_t)s * (size_t)n_cols + j] = 0.f;
        return;
    }
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    float acc = dot_q8_0_col_q8_range(col, xq, xd, bi0, bi1, lane);
    if (lane == 0) partial[(size_t)s * (size_t)n_cols + j] = acc;
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
    {
        int n4 = n_rows >> 2;
        for (int i = tid; i < n4; i += Q5_GEMV_THREADS) {
            float4 v = reinterpret_cast<const float4*>(x)[i];
            int o = i << 2;
            xs[o] = v.x; xs[o + 1] = v.y; xs[o + 2] = v.z; xs[o + 3] = v.w;
        }
        for (int i = (n4 << 2) + tid; i < n_rows; i += Q5_GEMV_THREADS) xs[i] = x[i];
    }
    __syncthreads();

    int j = (int)blockIdx.x * Q5_GEMV_WARPS + warp;
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

extern "C" __global__ void gemv_q5_k(
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
    for (int i = tid; i < n_rows; i += Q5_GEMV_THREADS) xs[i] = x[i];
    __syncthreads();

    int j = (int)blockIdx.x * Q5_GEMV_WARPS + warp;
    if (j >= n_cols) return;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    float acc = 0.f;
    int nb = n_rows / 256;
    for (int bi = lane; bi < nb; bi += 32) {
        acc += dot_q5_k_block_f32(col + bi * 176, xs + bi * 256);
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
    extern __shared__ unsigned char qsmem[];
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
    signed char* xq = reinterpret_cast<signed char*>(qsmem);
    float* xd = reinterpret_cast<float*>(qsmem + n_local);
    quantize_q8_smem(x + row0, xq, xd, n_local, warp, lane);
    __syncthreads();

    int j = (int)blockIdx.x * GEMV_WARPS + warp;
    if (j >= n_cols) return;
    if (bi0 >= bi1) {
        if (lane == 0) partial[(size_t)s * (size_t)n_cols + j] = 0.f;
        return;
    }
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    float acc = dot_q4_k_col_q8_range(col, xq, xd, bi0, bi1, lane);
    if (lane == 0) partial[(size_t)s * (size_t)n_cols + j] = acc;
}

extern "C" __global__ void gemv_q4_k_global_splitk(
    const unsigned char* __restrict__ w,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ partial,
    int n_rows, int n_cols, int col_bytes, int n_split
) {
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int s = (int)blockIdx.y;
    const int j = (int)blockIdx.x;
    if (s >= n_split || j >= n_cols) return;
    const int n_blocks = n_rows / 256;
    const int split_bi0 = (n_blocks * s) / n_split;
    const int split_bi1 = (n_blocks * (s + 1)) / n_split;
    const int mid = (split_bi0 + split_bi1 + 1) / 2;
    const int bi0 = warp == 0 ? split_bi0 : mid;
    const int bi1 = warp == 0 ? mid : split_bi1;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    float acc = dot_q4_k_col_q8_range(
        col, xq + bi0 * 256, xd + bi0 * 8, bi0, bi1, lane
    );
    __shared__ float second;
    if (warp == 1 && lane == 0) second = acc;
    __syncthreads();
    if (warp == 0 && lane == 0) {
        partial[(size_t)s * (size_t)n_cols + j] = acc + second;
    }
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

extern "C" __global__ void gemv_q6_k_repack_splitk(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ partial,
    int n_rows, int n_cols, int col_bytes, int n_split
) {
    extern __shared__ unsigned char qsmem[];
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
    signed char* xq = reinterpret_cast<signed char*>(qsmem);
    float* xd = reinterpret_cast<float*>(qsmem + n_local);
    quantize_q8_smem(x + row0, xq, xd, n_local, warp, lane);
    __syncthreads();

    const int j = (int)blockIdx.x * GEMV_WARPS + warp;
    if (j >= n_cols) return;
    if (bi0 >= bi1) {
        if (lane == 0) partial[(size_t)s * (size_t)n_cols + j] = 0.f;
        return;
    }
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    const float acc = dot_q6_k_repack_q8_range(col, xq, xd, bi0, bi1, lane);
    if (lane == 0) partial[(size_t)s * (size_t)n_cols + j] = acc;
}

extern "C" __global__ void gemv_q6_k_repack_global_splitk(
    const unsigned char* __restrict__ w,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ partial,
    int n_rows, int n_cols, int col_bytes, int n_split
) {
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int s = (int)blockIdx.y;
    if (s >= n_split) return;
    const int nsb = n_rows / 256;
    const int bi0 = (nsb * s) / n_split;
    const int bi1 = (nsb * (s + 1)) / n_split;
    const int j = (int)blockIdx.x * GEMV_WARPS + warp;
    if (j >= n_cols) return;
    if (bi0 >= bi1) {
        if (lane == 0) partial[(size_t)s * (size_t)n_cols + j] = 0.f;
        return;
    }
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    const float acc = dot_q6_k_repack_q8_global_range(col, xq, xd, bi0, bi1, lane);
    if (lane == 0) partial[(size_t)s * (size_t)n_cols + j] = acc;
}

extern "C" __global__ void gemv_q6_k_compact_global_splitk(
    const unsigned char* __restrict__ w,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ partial,
    int n_rows, int n_cols, int col_bytes, int n_split
) {
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int s = (int)blockIdx.y;
    if (s >= n_split) return;
    const int nsb = n_rows / 256;
    const int bi0 = (nsb * s) / n_split;
    const int bi1 = (nsb * (s + 1)) / n_split;
    const int warps_per_block = (int)blockDim.x >> 5;
    const int j = (int)blockIdx.x * warps_per_block + warp;
    if (j >= n_cols) return;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    const float acc = dot_q6_k_compact_q8_range(
        col, xq + bi0 * 256, xd + bi0 * 8, bi0, bi1, lane
    );
    if (lane == 0) partial[(size_t)s * (size_t)n_cols + j] = acc;
}

// One output column per block.  Four warps split the tall K dimension and
// reduce in shared memory, avoiding the separate partial-buffer reduction
// launch used by conventional split-K.
extern "C" __global__ void gemv_q6_k_compact_global_4way(
    const unsigned char* __restrict__ w,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ out,
    int n_rows, int n_cols, int col_bytes,
    int use_bias, const float* __restrict__ bias,
    int use_res, const float* __restrict__ residual
) {
    __shared__ float warp_acc[4];
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int j = (int)blockIdx.x;
    if (j >= n_cols) return;
    const int nsb = n_rows / 256;
    const int bi0 = (nsb * warp) / 4;
    const int bi1 = (nsb * (warp + 1)) / 4;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    const float acc = dot_q6_k_compact_q8_range(
        col, xq + bi0 * 256, xd + bi0 * 8, bi0, bi1, lane
    );
    if (lane == 0) warp_acc[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = warp_acc[0] + warp_acc[1] + warp_acc[2] + warp_acc[3];
        total = gemv_apply_res(use_res, total, out, j, residual);
        if (use_bias) total += bias[j];
        out[j] = total;
    }
}

// Multi-column Q6 compact: 8 warps → 8 output cols, xq already global.
// blockDim=256, grid=ceil(n_cols/8).
extern "C" __global__ void gemv_q6_k_compact_global_mcol(
    const unsigned char* __restrict__ w,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ out,
    int n_rows, int n_cols, int col_bytes,
    int use_bias, const float* __restrict__ bias,
    int use_res, const float* __restrict__ residual
) {
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int j = (int)blockIdx.x * 8 + warp;
    if (j >= n_cols) return;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    float acc = dot_q6_k_compact_q8_range(col, xq, xd, 0, n_rows / 256, lane);
    if (lane == 0) {
        acc = gemv_apply_res(use_res, acc, out, j, residual);
        if (use_bias) acc += bias[j];
        out[j] = acc;
    }
}

// 8-warp cooperative Q6 compact decode (blockDim=256). Better SM fill on tall FFN down.
extern "C" __global__ void gemv_q6_k_compact_global_8way(
    const unsigned char* __restrict__ w,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ out,
    int n_rows, int n_cols, int col_bytes,
    int use_bias, const float* __restrict__ bias,
    int use_res, const float* __restrict__ residual
) {
    __shared__ float warp_acc[8];
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int j = (int)blockIdx.x;
    if (j >= n_cols) return;
    const int nsb = n_rows / 256;
    const int bi0 = (nsb * warp) / 8;
    const int bi1 = (nsb * (warp + 1)) / 8;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    float acc = 0.f;
    if (bi0 < bi1) {
        acc = dot_q6_k_compact_q8_range(
            col, xq + bi0 * 256, xd + bi0 * 8, bi0, bi1, lane
        );
    }
    if (lane == 0) warp_acc[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.f;
        #pragma unroll
        for (int i = 0; i < 8; i++) total += warp_acc[i];
        total = gemv_apply_res(use_res, total, out, j, residual);
        if (use_bias) total += bias[j];
        out[j] = total;
    }
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
    for (int i = tid; i < n_local; i += Q5_GEMV_THREADS) xs[i] = x[row0 + i];
    __syncthreads();

    int j = (int)blockIdx.x * Q5_GEMV_WARPS + warp;
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

extern "C" __global__ void gemv_q5_k_splitk(
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

    const int nb = n_rows / 256;
    const int bi0 = (nb * s) / n_split;
    const int bi1 = (nb * (s + 1)) / n_split;
    const int row0 = bi0 * 256;
    const int n_local = (bi1 - bi0) * 256;
    for (int i = tid; i < n_local; i += Q5_GEMV_THREADS) xs[i] = x[row0 + i];
    __syncthreads();

    int j = (int)blockIdx.x * Q5_GEMV_WARPS + warp;
    if (j >= n_cols) return;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    float acc = 0.f;
    for (int bi = bi0 + lane; bi < bi1; bi += 32) {
        acc += dot_q5_k_block_f32(col + bi * 176, xs + (bi - bi0) * 256);
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
    for (int i = tid; i < n_rows; i += Q5_GEMV_THREADS) xs[i] = x[i];
    __syncthreads();

    int j = (int)blockIdx.x * Q5_GEMV_WARPS + warp;
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
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ out_a,
    float* __restrict__ out_b,
    int n_rows, int n_a, int n_b, int col_bytes,
    int use_bias,
    const float* __restrict__ ba,
    const float* __restrict__ bb
) {
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int j = (int)blockIdx.x;
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

    const int n_blocks = n_rows / 256;
    const int split = (n_blocks + 1) / 2;
    const int bi0 = warp == 0 ? 0 : split;
    const int bi1 = warp == 0 ? split : n_blocks;
    float acc = dot_q4_k_col_q8_range(
        wcol, xq + bi0 * 256, xd + bi0 * 8, bi0, bi1, lane
    );
    __shared__ float second;
    if (warp == 1 && lane == 0) second = acc;
    __syncthreads();
    if (warp == 0 && lane == 0) {
        acc += second;
        if (use_bias) acc += bias[oj];
        out[oj] = acc;
    }
}

// Equal-width Q4_K pair (FFN gate+up): one output row per block so the same
// two warps reuse the activation cache lines for both matrices.
extern "C" __global__ void gemv_q4_k_dual(
    const unsigned char* __restrict__ wa,
    const unsigned char* __restrict__ wb,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ out_a,
    float* __restrict__ out_b,
    int n_rows, int n_cols, int col_bytes
) {
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int j = (int)blockIdx.x;
    if (j >= n_cols) return;
    const int n_blocks = n_rows / 256;
    const int split = (n_blocks + 1) / 2;
    const int bi0 = warp == 0 ? 0 : split;
    const int bi1 = warp == 0 ? split : n_blocks;
    const signed char* xq_local = xq + bi0 * 256;
    const float* xd_local = xd + bi0 * 8;
    const float2 pair = dot_q4_k_pair_q8_range(
        wa + (size_t)j * (size_t)col_bytes,
        wb + (size_t)j * (size_t)col_bytes,
        xq_local, xd_local, bi0, bi1, lane
    );
    __shared__ float a_second;
    __shared__ float b_second;
    if (warp == 1 && lane == 0) {
        a_second = pair.x;
        b_second = pair.y;
    }
    __syncthreads();
    if (warp == 0 && lane == 0) {
        out_a[j] = pair.x + a_second;
        out_b[j] = pair.y + b_second;
    }
}

// Decode FFN gate+up with the SiLU/multiply epilogue fused into the final
// writer. This avoids materializing the separate up vector and a later launch.
extern "C" __global__ void gemv_q4_k_ffn(
    const unsigned char* __restrict__ gate_w,
    const unsigned char* __restrict__ up_w,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ out,
    int n_rows, int n_cols, int col_bytes
) {
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int j = (int)blockIdx.x;
    if (j >= n_cols) return;
    const int n_blocks = n_rows / 256;
    const int split = (n_blocks + 1) / 2;
    const int bi0 = warp == 0 ? 0 : split;
    const int bi1 = warp == 0 ? split : n_blocks;
    const float2 pair = dot_q4_k_pair_q8_range(
        gate_w + (size_t)j * (size_t)col_bytes,
        up_w + (size_t)j * (size_t)col_bytes,
        xq + bi0 * 256, xd + bi0 * 8, bi0, bi1, lane
    );
    __shared__ float gate_second;
    __shared__ float up_second;
    if (warp == 1 && lane == 0) {
        gate_second = pair.x;
        up_second = pair.y;
    }
    __syncthreads();
    if (warp == 0 && lane == 0) {
        const float gate = pair.x + gate_second;
        const float up = pair.y + up_second;
        out[j] = (gate / (1.f + __expf(-gate))) * up;
    }
}

// Ampere and newer: four warps cooperate on each equal-width gate/up row.
extern "C" __global__ void gemv_q4_k_dual_4way(
    const unsigned char* __restrict__ wa,
    const unsigned char* __restrict__ wb,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ out_a,
    float* __restrict__ out_b,
    int n_rows, int n_cols, int col_bytes
) {
    __shared__ float a_part[4];
    __shared__ float b_part[4];
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int j = (int)blockIdx.x;
    if (j >= n_cols) return;
    const int n_blocks = n_rows / 256;
    const int bi0 = (n_blocks * warp) / 4;
    const int bi1 = (n_blocks * (warp + 1)) / 4;
    const float2 pair = dot_q4_k_pair_q8_range(
        wa + (size_t)j * (size_t)col_bytes,
        wb + (size_t)j * (size_t)col_bytes,
        xq + bi0 * 256, xd + bi0 * 8, bi0, bi1, lane
    );
    if (lane == 0) {
        a_part[warp] = pair.x;
        b_part[warp] = pair.y;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        out_a[j] = a_part[0] + a_part[1] + a_part[2] + a_part[3];
        out_b[j] = b_part[0] + b_part[1] + b_part[2] + b_part[3];
    }
}

extern "C" __global__ void gemv_q4_k_ffn_4way(
    const unsigned char* __restrict__ gate_w,
    const unsigned char* __restrict__ up_w,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ out,
    int n_rows, int n_cols, int col_bytes
) {
    __shared__ float gate_part[4];
    __shared__ float up_part[4];
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int j = (int)blockIdx.x;
    if (j >= n_cols) return;
    const int n_blocks = n_rows / 256;
    const int bi0 = (n_blocks * warp) / 4;
    const int bi1 = (n_blocks * (warp + 1)) / 4;
    const float2 pair = dot_q4_k_pair_q8_range(
        gate_w + (size_t)j * (size_t)col_bytes,
        up_w + (size_t)j * (size_t)col_bytes,
        xq + bi0 * 256, xd + bi0 * 8, bi0, bi1, lane
    );
    if (lane == 0) {
        gate_part[warp] = pair.x;
        up_part[warp] = pair.y;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const float gate = gate_part[0] + gate_part[1] + gate_part[2] + gate_part[3];
        const float up = up_part[0] + up_part[1] + up_part[2] + up_part[3];
        out[j] = (gate / (1.f + __expf(-gate))) * up;
    }
}

// Multi-column fused FFN: 32 warps → 32 output columns, quantize x once in smem.
// blockDim = 1024, grid = ceil(n_cols/32). Best for wide decode FFNs.
extern "C" __global__ void gemv_q4_k_ffn_mcol(
    const unsigned char* __restrict__ gate_w,
    const unsigned char* __restrict__ up_w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_rows, int n_cols, int col_bytes
) {
    extern __shared__ unsigned char qsmem[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    signed char* xq = reinterpret_cast<signed char*>(qsmem);
    float* xd = reinterpret_cast<float*>(qsmem + n_rows);
    quantize_q8_smem(x, xq, xd, n_rows, warp, lane);
    __syncthreads();

    const int j = (int)blockIdx.x * 32 + warp;
    if (j >= n_cols) return;
    // Full-K reduction in this warp (no split-K).
    const float2 pair = dot_q4_k_pair_q8_range(
        gate_w + (size_t)j * (size_t)col_bytes,
        up_w + (size_t)j * (size_t)col_bytes,
        xq, xd, 0, n_rows / 256, lane
    );
    if (lane == 0) {
        const float gate = pair.x;
        const float up = pair.y;
        out[j] = (gate / (1.f + __expf(-gate))) * up;
    }
}

// 8-warp FFN gate+up (blockDim=256). Use when n_rows/256 is large enough.
extern "C" __global__ void gemv_q4_k_ffn_8way(
    const unsigned char* __restrict__ gate_w,
    const unsigned char* __restrict__ up_w,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ out,
    int n_rows, int n_cols, int col_bytes
) {
    __shared__ float gate_part[8];
    __shared__ float up_part[8];
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int j = (int)blockIdx.x;
    if (j >= n_cols) return;
    const int n_blocks = n_rows / 256;
    const int bi0 = (n_blocks * warp) / 8;
    const int bi1 = (n_blocks * (warp + 1)) / 8;
    float2 pair;
    pair.x = 0.f;
    pair.y = 0.f;
    if (bi0 < bi1) {
        pair = dot_q4_k_pair_q8_range(
            gate_w + (size_t)j * (size_t)col_bytes,
            up_w + (size_t)j * (size_t)col_bytes,
            xq + bi0 * 256, xd + bi0 * 8, bi0, bi1, lane
        );
    }
    if (lane == 0) {
        gate_part[warp] = pair.x;
        up_part[warp] = pair.y;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float gate = 0.f, up = 0.f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            gate += gate_part[i];
            up += up_part[i];
        }
        out[j] = (gate / (1.f + __expf(-gate))) * up;
    }
}

extern "C" __global__ void gemv_q4_k_ffn_4way_smem(
    const unsigned char* __restrict__ gate_w,
    const unsigned char* __restrict__ up_w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_rows, int n_cols, int col_bytes
) {
    extern __shared__ unsigned char qsmem[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    signed char* xq = reinterpret_cast<signed char*>(qsmem);
    float* xd = reinterpret_cast<float*>(qsmem + n_rows);
    quantize_q8_smem(x, xq, xd, n_rows, warp, lane);
    __syncthreads();

    __shared__ float gate_part[4];
    __shared__ float up_part[4];
    const int j = (int)blockIdx.x;
    if (j >= n_cols) return;
    const int n_blocks = n_rows / 256;
    const int bi0 = (n_blocks * warp) / 4;
    const int bi1 = (n_blocks * (warp + 1)) / 4;
    const float2 pair = dot_q4_k_pair_q8_range(
        gate_w + (size_t)j * (size_t)col_bytes,
        up_w + (size_t)j * (size_t)col_bytes,
        xq + bi0 * 256, xd + bi0 * 8, bi0, bi1, lane
    );
    if (lane == 0) {
        gate_part[warp] = pair.x;
        up_part[warp] = pair.y;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const float gate = gate_part[0] + gate_part[1] + gate_part[2] + gate_part[3];
        const float up = up_part[0] + up_part[1] + up_part[2] + up_part[3];
        out[j] = (gate / (1.f + __expf(-gate))) * up;
    }
}

// Fused Q+K+V for Q4_K decode: quantize/stage the hidden state once.
extern "C" __global__ void gemv_q4_k_qkv(
    const unsigned char* __restrict__ wq,
    const unsigned char* __restrict__ wk,
    const unsigned char* __restrict__ wv,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    int n_rows, int n_q, int n_k, int n_v, int col_bytes,
    int use_bias,
    const float* __restrict__ bq,
    const float* __restrict__ bk,
    const float* __restrict__ bv
) {
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int warps_per_block = (int)blockDim.x >> 5;
    const int j = (int)blockIdx.x * warps_per_block + warp;
    const int n_tot = n_q + n_k + n_v;
    if (j >= n_tot) return;

    const unsigned char* wcol;
    float* out;
    const float* bias;
    int oj;
    if (j < n_q) {
        wcol = wq + (size_t)j * (size_t)col_bytes;
        out = q; bias = bq; oj = j;
    } else if (j < n_q + n_k) {
        oj = j - n_q;
        wcol = wk + (size_t)oj * (size_t)col_bytes;
        out = k; bias = bk;
    } else {
        oj = j - n_q - n_k;
        wcol = wv + (size_t)oj * (size_t)col_bytes;
        out = v; bias = bv;
    }
    float acc = dot_q4_k_col_q8_range(wcol, xq, xd, 0, n_rows / 256, lane);
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
    for (int i = tid; i < n_rows; i += Q5_GEMV_THREADS) xs[i] = x[i];
    __syncthreads();

    int j = (int)blockIdx.x * Q5_GEMV_WARPS + warp;
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

// Fused Q+K+V when all three are Q8_0 (global activation quantize once).
extern "C" __global__ void gemv_q8_0_qkv(
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
    extern __shared__ unsigned char qsmem[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    signed char* xq = reinterpret_cast<signed char*>(qsmem);
    float* xd = reinterpret_cast<float*>(qsmem + n_rows);
    quantize_q8_smem(x, xq, xd, n_rows, warp, lane);
    __syncthreads();

    const int warps_per_block = (int)blockDim.x >> 5;
    const int j = (int)blockIdx.x * warps_per_block + warp;
    const int n_tot = n_q + n_k + n_v;
    if (j >= n_tot) return;

    const unsigned char* wcol;
    float* out;
    const float* bias;
    int oj;
    if (j < n_q) {
        wcol = wq + (size_t)j * (size_t)col_bytes;
        out = q; bias = bq; oj = j;
    } else if (j < n_q + n_k) {
        oj = j - n_q;
        wcol = wk + (size_t)oj * (size_t)col_bytes;
        out = k; bias = bk;
    } else {
        oj = j - n_q - n_k;
        wcol = wv + (size_t)oj * (size_t)col_bytes;
        out = v; bias = bv;
    }
    float acc = dot_q8_0_col_q8_range(wcol, xq, xd, 0, n_rows / 32, lane);
    if (lane == 0) {
        if (use_bias) acc += bias[oj];
        out[oj] = acc;
    }
}

// Fused 4-way GDN input projections for Q8_0.
// Matches gemv_q8_0_qkv: quantize x once in shared memory, then DP4A all columns.
extern "C" __global__ void gemv_q8_0_gdn_4way(
    const unsigned char* __restrict__ wqkv,
    const unsigned char* __restrict__ w_gate,
    const unsigned char* __restrict__ w_beta,
    const unsigned char* __restrict__ w_alpha,
    const float* __restrict__ x,
    float* __restrict__ out_qkv,
    float* __restrict__ out_gate,
    float* __restrict__ out_beta,
    float* __restrict__ out_alpha,
    int n_rows, int n_qkv, int n_gate, int n_beta, int n_alpha, int col_bytes
) {
    extern __shared__ unsigned char qsmem[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    signed char* xq = reinterpret_cast<signed char*>(qsmem);
    float* xd = reinterpret_cast<float*>(qsmem + n_rows);
    quantize_q8_smem(x, xq, xd, n_rows, warp, lane);
    __syncthreads();

    const int warps_per_block = (int)blockDim.x >> 5;
    const int j = (int)blockIdx.x * warps_per_block + warp;
    const int n_tot = n_qkv + n_gate + n_beta + n_alpha;
    if (j >= n_tot) return;

    const unsigned char* wcol;
    float* out;
    int oj;
    if (j < n_qkv) {
        wcol = wqkv + (size_t)j * (size_t)col_bytes;
        out = out_qkv; oj = j;
    } else if (j < n_qkv + n_gate) {
        oj = j - n_qkv;
        wcol = w_gate + (size_t)oj * (size_t)col_bytes;
        out = out_gate;
    } else if (j < n_qkv + n_gate + n_beta) {
        oj = j - n_qkv - n_gate;
        wcol = w_beta + (size_t)oj * (size_t)col_bytes;
        out = out_beta;
    } else {
        oj = j - n_qkv - n_gate - n_beta;
        wcol = w_alpha + (size_t)oj * (size_t)col_bytes;
        out = out_alpha;
    }
    float acc = dot_q8_0_col_q8_range(wcol, xq, xd, 0, n_rows / 32, lane);
    if (lane == 0) {
        out[oj] = acc;
    }
}

// Fused 4-way hybrid: wqkv/beta/alpha Q8_0, w_gate Q4_K. Smem-quantize x once.
extern "C" __global__ void gemv_hybrid_gdn_4way(
    const unsigned char* __restrict__ wqkv,
    const unsigned char* __restrict__ w_gate,
    const unsigned char* __restrict__ w_beta,
    const unsigned char* __restrict__ w_alpha,
    const float* __restrict__ x,
    float* __restrict__ out_qkv,
    float* __restrict__ out_gate,
    float* __restrict__ out_beta,
    float* __restrict__ out_alpha,
    int n_rows, int n_qkv, int n_gate, int n_beta, int n_alpha,
    int col_bytes_q8, int col_bytes_q4
) {
    extern __shared__ unsigned char qsmem[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    signed char* xq = reinterpret_cast<signed char*>(qsmem);
    float* xd = reinterpret_cast<float*>(qsmem + n_rows);
    quantize_q8_smem(x, xq, xd, n_rows, warp, lane);
    __syncthreads();

    const int warps_per_block = (int)blockDim.x >> 5;
    const int j = (int)blockIdx.x * warps_per_block + warp;
    const int n_tot = n_qkv + n_gate + n_beta + n_alpha;
    if (j >= n_tot) return;

    const unsigned char* wcol;
    float* out;
    int oj;
    float acc;
    if (j < n_qkv) {
        wcol = wqkv + (size_t)j * (size_t)col_bytes_q8;
        out = out_qkv; oj = j;
        acc = dot_q8_0_col_q8_range(wcol, xq, xd, 0, n_rows / 32, lane);
    } else if (j < n_qkv + n_gate) {
        oj = j - n_qkv;
        wcol = w_gate + (size_t)oj * (size_t)col_bytes_q4;
        out = out_gate;
        acc = dot_q4_k_col_q8_range(wcol, xq, xd, 0, n_rows / 256, lane);
    } else if (j < n_qkv + n_gate + n_beta) {
        oj = j - n_qkv - n_gate;
        wcol = w_beta + (size_t)oj * (size_t)col_bytes_q8;
        out = out_beta;
        acc = dot_q8_0_col_q8_range(wcol, xq, xd, 0, n_rows / 32, lane);
    } else {
        oj = j - n_qkv - n_gate - n_beta;
        wcol = w_alpha + (size_t)oj * (size_t)col_bytes_q8;
        out = out_alpha;
        acc = dot_q8_0_col_q8_range(wcol, xq, xd, 0, n_rows / 32, lane);
    }
    if (lane == 0) {
        out[oj] = acc;
    }
}

extern "C" __global__ void gemv_q8_0_qkv_splitk(
    const unsigned char* __restrict__ wq,
    const unsigned char* __restrict__ wk,
    const unsigned char* __restrict__ wv,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ partial,
    int n_rows, int n_q, int n_k, int n_v, int col_bytes, int n_split
) {
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const int s = (int)blockIdx.y;
    if (s >= n_split) return;
    const int j = (int)blockIdx.x * warps_per_block + warp;
    const int n_tot = n_q + n_k + n_v;
    if (j >= n_tot) return;
    const int nb = n_rows / 32;
    const int bi0 = (nb * s) / n_split;
    const int bi1 = (nb * (s + 1)) / n_split;
    if (bi0 >= bi1) {
        if (lane == 0) partial[(size_t)s * (size_t)n_tot + j] = 0.f;
        return;
    }
    const unsigned char* wcol;
    if (j < n_q) {
        wcol = wq + (size_t)j * (size_t)col_bytes;
    } else if (j < n_q + n_k) {
        wcol = wk + (size_t)(j - n_q) * (size_t)col_bytes;
    } else {
        wcol = wv + (size_t)(j - n_q - n_k) * (size_t)col_bytes;
    }
    float acc = dot_q8_0_col_q8_range(wcol, xq, xd, bi0, bi1, lane);
    if (lane == 0) partial[(size_t)s * (size_t)n_tot + j] = acc;
}

extern "C" __global__ void gemv_q8_0_gdn_4way_splitk(
    const unsigned char* __restrict__ wqkv,
    const unsigned char* __restrict__ w_gate,
    const unsigned char* __restrict__ w_beta,
    const unsigned char* __restrict__ w_alpha,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ partial,
    int n_rows, int n_qkv, int n_gate, int n_beta, int n_alpha, int col_bytes, int n_split
) {
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const int s = (int)blockIdx.y;
    if (s >= n_split) return;
    const int j = (int)blockIdx.x * warps_per_block + warp;
    const int n_tot = n_qkv + n_gate + n_beta + n_alpha;
    if (j >= n_tot) return;
    const int nb = n_rows / 32;
    const int bi0 = (nb * s) / n_split;
    const int bi1 = (nb * (s + 1)) / n_split;
    if (bi0 >= bi1) {
        if (lane == 0) partial[(size_t)s * (size_t)n_tot + j] = 0.f;
        return;
    }
    const unsigned char* wcol;
    if (j < n_qkv) {
        wcol = wqkv + (size_t)j * (size_t)col_bytes;
    } else if (j < n_qkv + n_gate) {
        wcol = w_gate + (size_t)(j - n_qkv) * (size_t)col_bytes;
    } else if (j < n_qkv + n_gate + n_beta) {
        wcol = w_beta + (size_t)(j - n_qkv - n_gate) * (size_t)col_bytes;
    } else {
        wcol = w_alpha + (size_t)(j - n_qkv - n_gate - n_beta) * (size_t)col_bytes;
    }
    float acc = dot_q8_0_col_q8_range(wcol, xq, xd, bi0, bi1, lane);
    if (lane == 0) partial[(size_t)s * (size_t)n_tot + j] = acc;
}

extern "C" __global__ void gemv_hybrid_gdn_4way_splitk(
    const unsigned char* __restrict__ wqkv,
    const unsigned char* __restrict__ w_gate,
    const unsigned char* __restrict__ w_beta,
    const unsigned char* __restrict__ w_alpha,
    const signed char* __restrict__ xq,
    const float* __restrict__ xd,
    float* __restrict__ partial,
    int n_rows, int n_qkv, int n_gate, int n_beta, int n_alpha,
    int col_bytes_q8, int col_bytes_q4, int n_split
) {
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const int s = (int)blockIdx.y;
    if (s >= n_split) return;
    const int j = (int)blockIdx.x * warps_per_block + warp;
    const int n_tot = n_qkv + n_gate + n_beta + n_alpha;
    if (j >= n_tot) return;

    const unsigned char* wcol;
    float acc;
    if (j < n_qkv) {
        const int nb = n_rows / 32;
        const int bi0 = (nb * s) / n_split;
        const int bi1 = (nb * (s + 1)) / n_split;
        if (bi0 >= bi1) {
            if (lane == 0) partial[(size_t)s * (size_t)n_tot + j] = 0.f;
            return;
        }
        wcol = wqkv + (size_t)j * (size_t)col_bytes_q8;
        acc = dot_q8_0_col_q8_range(wcol, xq, xd, bi0, bi1, lane);
    } else if (j < n_qkv + n_gate) {
        const int nb = n_rows / 256;
        const int bi0 = (nb * s) / n_split;
        const int bi1 = (nb * (s + 1)) / n_split;
        if (bi0 >= bi1) {
            if (lane == 0) partial[(size_t)s * (size_t)n_tot + j] = 0.f;
            return;
        }
        wcol = w_gate + (size_t)(j - n_qkv) * (size_t)col_bytes_q4;
        acc = dot_q4_k_col_q8_range(wcol, xq, xd, bi0, bi1, lane);
    } else if (j < n_qkv + n_gate + n_beta) {
        const int nb = n_rows / 32;
        const int bi0 = (nb * s) / n_split;
        const int bi1 = (nb * (s + 1)) / n_split;
        if (bi0 >= bi1) {
            if (lane == 0) partial[(size_t)s * (size_t)n_tot + j] = 0.f;
            return;
        }
        wcol = w_beta + (size_t)(j - n_qkv - n_gate) * (size_t)col_bytes_q8;
        acc = dot_q8_0_col_q8_range(wcol, xq, xd, bi0, bi1, lane);
    } else {
        const int nb = n_rows / 32;
        const int bi0 = (nb * s) / n_split;
        const int bi1 = (nb * (s + 1)) / n_split;
        if (bi0 >= bi1) {
            if (lane == 0) partial[(size_t)s * (size_t)n_tot + j] = 0.f;
            return;
        }
        wcol = w_alpha + (size_t)(j - n_qkv - n_gate - n_beta) * (size_t)col_bytes_q8;
        acc = dot_q8_0_col_q8_range(wcol, xq, xd, bi0, bi1, lane);
    }
    if (lane == 0) partial[(size_t)s * (size_t)n_tot + j] = acc;
}

extern "C" __global__ void gemv_splitk_reduce_qkv(
    const float* __restrict__ partial,
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    int n_q, int n_k, int n_v,
    int n_split,
    int use_bias,
    const float* __restrict__ bq,
    const float* __restrict__ bk,
    const float* __restrict__ bv
) {
    int j = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int n_tot = n_q + n_k + n_v;
    if (j >= n_tot) return;
    float acc = 0.f;
    #pragma unroll 4
    for (int s = 0; s < n_split; s++) {
        acc += partial[(size_t)s * (size_t)n_tot + j];
    }
    float* out;
    const float* bias;
    int oj;
    if (j < n_q) { out = q; bias = bq; oj = j; }
    else if (j < n_q + n_k) { out = k; bias = bk; oj = j - n_q; }
    else { out = v; bias = bv; oj = j - n_q - n_k; }
    if (use_bias) acc += bias[oj];
    out[oj] = acc;
}

extern "C" __global__ void gemv_splitk_reduce_gdn_4way(
    const float* __restrict__ partial,
    float* __restrict__ out_qkv,
    float* __restrict__ out_gate,
    float* __restrict__ out_beta,
    float* __restrict__ out_alpha,
    int n_qkv, int n_gate, int n_beta, int n_alpha,
    int n_split
) {
    int j = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int n_tot = n_qkv + n_gate + n_beta + n_alpha;
    if (j >= n_tot) return;
    float acc = 0.f;
    #pragma unroll 4
    for (int s = 0; s < n_split; s++) {
        acc += partial[(size_t)s * (size_t)n_tot + j];
    }
    float* out;
    int oj;
    if (j < n_qkv) { out = out_qkv; oj = j; }
    else if (j < n_qkv + n_gate) { out = out_gate; oj = j - n_qkv; }
    else if (j < n_qkv + n_gate + n_beta) { out = out_beta; oj = j - n_qkv - n_gate; }
    else { out = out_alpha; oj = j - n_qkv - n_gate - n_beta; }
    out[oj] = acc;
}
