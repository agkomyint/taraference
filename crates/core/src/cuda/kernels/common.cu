
__device__ __forceinline__ float half_to_float(unsigned short h) {
    unsigned int sign = ((unsigned int)h >> 15) & 1u;
    unsigned int exp  = ((unsigned int)h >> 10) & 0x1fu;
    unsigned int mant = (unsigned int)h & 0x3ffu;
    unsigned int f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0u) { mant <<= 1; exp--; }
            mant &= 0x3ffu;
            f = (sign << 31) | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7f800000u | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    return __int_as_float(f);
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
