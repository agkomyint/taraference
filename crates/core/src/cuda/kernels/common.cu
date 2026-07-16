
__device__ __forceinline__ float half_to_float(unsigned short h) {
    float out;
    asm("cvt.f32.f16 %0, %1;" : "=f"(out) : "h"(h));
    return out;
}

// Round-to-nearest-even-ish f32→f16 for KV store (2× less attention bandwidth).
__device__ __forceinline__ unsigned short float_to_half_bits(float f) {
    unsigned int x = __float_as_uint(f);
    unsigned int sign = (x >> 16) & 0x8000u;
    unsigned int absx = x & 0x7fffffffu;
    if (absx == 0u) return (unsigned short)sign;
    // NaN / Inf
    if (absx >= 0x7f800000u) {
        if (absx == 0x7f800000u) return (unsigned short)(sign | 0x7c00u);
        return (unsigned short)(sign | 0x7e00u);
    }
    // Too small → 0; too large → Inf
    if (absx < 0x33000000u) return (unsigned short)sign;
    if (absx >= 0x47800000u) return (unsigned short)(sign | 0x7c00u);
    int exp = (int)((absx >> 23) & 0xff) - 127 + 15;
    unsigned int mant = absx & 0x7fffffu;
    if (exp <= 0) {
        // denormal
        if (exp < -10) return (unsigned short)sign;
        mant |= 0x800000u;
        unsigned int shift = (unsigned int)(14 - exp);
        unsigned int half_mant = mant >> shift;
        // round
        if ((mant >> (shift - 1)) & 1u) half_mant += 1u;
        return (unsigned short)(sign | half_mant);
    }
    unsigned int half = ((unsigned int)exp << 10) | (mant >> 13);
    // round bit
    if (mant & 0x1000u) half += 1u;
    return (unsigned short)(sign | half);
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

__device__ __forceinline__ float warp_sum(float v) {
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v, 8);
    v += __shfl_down_sync(0xffffffffu, v, 4);
    v += __shfl_down_sync(0xffffffffu, v, 2);
    v += __shfl_down_sync(0xffffffffu, v, 1);
    return v;
}

// ---------------------------------------------------------------------------
// Dequant one Q4_K block (256 vals) into smem cooperatively
// ---------------------------------------------------------------------------
__device__ void dequant_q4_k_block_smem(
    const unsigned char* base, float* smem, int tid, int nthreads
) {
    float d = half_to_float((unsigned short)(base[0] | (base[1] << 8)));
    float minv = half_to_float((unsigned short)(base[2] | (base[3] << 8)));
    const unsigned char* scales = base + 4;
    const unsigned char* q = base + 16;
    // 4 groups of 64
    for (int t = 0; t < 4; t++) {
        unsigned char sc, m;
        get_scale_min_k4(t * 2, scales, &sc, &m);
        float d1 = d * (float)sc, m1 = minv * (float)m;
        get_scale_min_k4(t * 2 + 1, scales, &sc, &m);
        float d2 = d * (float)sc, m2 = minv * (float)m;
        const unsigned char* qq = q + t * 32;
        for (int l = tid; l < 32; l += nthreads) {
            smem[t * 64 + l] = d1 * (float)(qq[l] & 0xF) - m1;
            smem[t * 64 + 32 + l] = d2 * (float)(qq[l] >> 4) - m2;
        }
    }
}

// Q5_0: 32 vals / block, 22 bytes (fp16 d + 4B qh + 16B qs)
__device__ __forceinline__ void dequant_q5_0_block(
    const unsigned char* base, float* out32
) {
    float d = half_to_float((unsigned short)(base[0] | (base[1] << 8)));
    unsigned int qh = (unsigned int)base[2]
        | ((unsigned int)base[3] << 8)
        | ((unsigned int)base[4] << 16)
        | ((unsigned int)base[5] << 24);
    const unsigned char* qs = base + 6;
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        unsigned char xh0 = (unsigned char)(((qh >> j) << 4) & 0x10u);
        unsigned char xh1 = (unsigned char)(((qh >> (j + 12)) ) & 0x10u);
        int x0 = (int)((qs[j] & 0x0F) | xh0);
        int x1 = (int)((qs[j] >> 4) | xh1);
        out32[j] = (float)(x0 - 16) * d;
        out32[j + 16] = (float)(x1 - 16) * d;
    }
}

// Q5_K: 256 vals / block, 176 bytes (fp16 d/dmin + 12B scales + 32B qh + 128B qs)
__device__ void dequant_q5_k_block_smem(
    const unsigned char* base, float* smem, int tid, int nthreads
) {
    float d = half_to_float((unsigned short)(base[0] | (base[1] << 8)));
    float minv = half_to_float((unsigned short)(base[2] | (base[3] << 8)));
    const unsigned char* scales = base + 4;
    const unsigned char* qh = base + 16;
    const unsigned char* qs = base + 48;
    for (int g = 0; g < 4; g++) {
        unsigned char sc0, m0, sc1, m1;
        get_scale_min_k4(2 * g, scales, &sc0, &m0);
        get_scale_min_k4(2 * g + 1, scales, &sc1, &m1);
        float d0 = d * (float)sc0, dm0 = minv * (float)m0;
        float d1 = d * (float)sc1, dm1 = minv * (float)m1;
        unsigned char hm0 = (unsigned char)(1u << (2 * g));
        unsigned char hm1 = (unsigned char)(2u << (2 * g));
        for (int l = tid; l < 32; l += nthreads) {
            unsigned char q = qs[g * 32 + l];
            int v0 = (int)(q & 0x0f) + ((qh[l] & hm0) ? 16 : 0);
            int v1 = (int)(q >> 4) + ((qh[l] & hm1) ? 16 : 0);
            smem[g * 64 + l] = d0 * (float)v0 - dm0;
            smem[g * 64 + 32 + l] = d1 * (float)v1 - dm1;
        }
    }
}

__device__ __forceinline__ float dot_q5_k_block_f32(
    const unsigned char* base, const float* x
) {
    float d = half_to_float((unsigned short)(base[0] | (base[1] << 8)));
    float minv = half_to_float((unsigned short)(base[2] | (base[3] << 8)));
    const unsigned char* scales = base + 4;
    const unsigned char* qh = base + 16;
    const unsigned char* qs = base + 48;
    float acc = 0.f;
    #pragma unroll
    for (int g = 0; g < 4; g++) {
        unsigned char sc0, m0, sc1, m1;
        get_scale_min_k4(2 * g, scales, &sc0, &m0);
        get_scale_min_k4(2 * g + 1, scales, &sc1, &m1);
        float d0 = d * (float)sc0, dm0 = minv * (float)m0;
        float d1 = d * (float)sc1, dm1 = minv * (float)m1;
        unsigned char hm0 = (unsigned char)(1u << (2 * g));
        unsigned char hm1 = (unsigned char)(2u << (2 * g));
        #pragma unroll
        for (int l = 0; l < 32; l++) {
            unsigned char q = qs[g * 32 + l];
            int v0 = (int)(q & 0x0f) + ((qh[l] & hm0) ? 16 : 0);
            int v1 = (int)(q >> 4) + ((qh[l] & hm1) ? 16 : 0);
            acc += (d0 * (float)v0 - dm0) * x[g * 64 + l];
            acc += (d1 * (float)v1 - dm1) * x[g * 64 + 32 + l];
        }
    }
    return acc;
}

__device__ void dequant_q6_k_block_smem(
    const unsigned char* base, float* smem, int tid, int nthreads
) {
    const unsigned char* ql = base;
    const unsigned char* qh = base + 128;
    const signed char* sc = (const signed char*)(base + 192);
    float d = half_to_float((unsigned short)(base[208] | (base[209] << 8)));
    for (int n = 0; n < 2; n++) {
        int ql_i = n * 64, qh_i = n * 32, sc_i = n * 8;
        int y0 = n * 128;
        for (int l = tid; l < 32; l += nthreads) {
            int is = l / 16;
            int q1 = (int)((ql[ql_i + l] & 0xF) | (((qh[qh_i + l] >> 0) & 3) << 4)) - 32;
            int q2 = (int)((ql[ql_i + 32 + l] & 0xF) | (((qh[qh_i + l] >> 2) & 3) << 4)) - 32;
            int q3 = (int)((ql[ql_i + l] >> 4) | (((qh[qh_i + l] >> 4) & 3) << 4)) - 32;
            int q4 = (int)((ql[ql_i + 32 + l] >> 4) | (((qh[qh_i + l] >> 6) & 3) << 4)) - 32;
            smem[y0 + l]      = d * (float)sc[sc_i + is] * (float)q1;
            smem[y0 + 32 + l] = d * (float)sc[sc_i + is + 2] * (float)q2;
            smem[y0 + 64 + l] = d * (float)sc[sc_i + is + 4] * (float)q3;
            smem[y0 + 96 + l] = d * (float)sc[sc_i + is + 6] * (float)q4;
        }
    }
}
