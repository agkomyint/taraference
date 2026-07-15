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
    for (int bi = warp; bi < nb; bi += GEMV_WARPS) {
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
        #pragma unroll
        for (int g = lane; g < 64; g += 32) {
            const int i = g * 4;
            const int half = i >> 7;
            const int segment = (i & 127) >> 5;
            const int offset = i & 31;
            const unsigned int ql = *reinterpret_cast<const unsigned int*>(
                base + half * 64 + (segment & 1) * 32 + offset
            );
            const unsigned int qh = *reinterpret_cast<const unsigned int*>(
                base + 128 + half * 32 + offset
            );
            const unsigned int lo = segment < 2
                ? (ql & 0x0f0f0f0fu) : ((ql >> 4) & 0x0f0f0f0fu);
            const unsigned int hi = (qh >> (segment * 2)) & 0x03030303u;
            const unsigned int raw = lo | (hi << 4);
            const int wpack = __vsubss4((int)raw, 0x20202020);
            const int apack = *reinterpret_cast<const int*>(xq + local + i);
            const int dot = __dp4a(wpack, apack, 0);
            acc += d * (float)sc[i >> 4] * xd[(local + i) >> 5] * (float)dot;
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
    const int j = (int)blockIdx.x * GEMV_WARPS + warp;
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
    const int j = (int)blockIdx.x * GEMV_WARPS + warp;
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
    {
        int n4 = n_rows >> 2;
        for (int i = tid; i < n4; i += GEMV_THREADS) {
            float4 v = reinterpret_cast<const float4*>(x)[i];
            int o = i << 2;
            xs[o] = v.x; xs[o + 1] = v.y; xs[o + 2] = v.z; xs[o + 3] = v.w;
        }
        for (int i = (n4 << 2) + tid; i < n_rows; i += GEMV_THREADS) xs[i] = x[i];
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

    const int j = (int)blockIdx.x * GEMV_WARPS + warp;
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
