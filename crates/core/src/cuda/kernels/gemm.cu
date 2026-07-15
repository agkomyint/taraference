// ---------------------------------------------------------------------------
// GEMM: out[t, j] = W[:,j] · x[t, :]
// grid: n_cols   block: 32
// Dequant each QK block once → shared, then accumulate all tokens.
// ---------------------------------------------------------------------------
extern "C" __global__ void gemm_q4_k(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_rows,
    int n_cols,
    int col_bytes,
    int n_tok
) {
    int j = (int)blockIdx.x;
    if (j >= n_cols) return;
    const int tid = (int)threadIdx.x;
    const int nt = (int)blockDim.x;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    __shared__ float dq[256];

    // Process tokens in tiles to keep acc in registers
    const int TILE = 8;
    for (int t0 = 0; t0 < n_tok; t0 += TILE) {
        int tn = n_tok - t0;
        if (tn > TILE) tn = TILE;
        float acc0=0,acc1=0,acc2=0,acc3=0,acc4=0,acc5=0,acc6=0,acc7=0;
        int nb = n_rows / 256;
        for (int bi = 0; bi < nb; bi++) {
            dequant_q4_k_block_smem(col + bi * 144, dq, tid, nt);
            __syncthreads();
            int base = bi * 256;
            for (int i = tid; i < 256; i += nt) {
                float wv = dq[i];
                const float* xp = x + (size_t)t0 * (size_t)n_rows + base + i;
                if (tn > 0) acc0 += wv * xp[0 * n_rows];
                if (tn > 1) acc1 += wv * xp[1 * n_rows];
                if (tn > 2) acc2 += wv * xp[2 * n_rows];
                if (tn > 3) acc3 += wv * xp[3 * n_rows];
                if (tn > 4) acc4 += wv * xp[4 * n_rows];
                if (tn > 5) acc5 += wv * xp[5 * n_rows];
                if (tn > 6) acc6 += wv * xp[6 * n_rows];
                if (tn > 7) acc7 += wv * xp[7 * n_rows];
            }
            __syncthreads();
        }
        acc0 = warp_sum(acc0);
        acc1 = warp_sum(acc1);
        acc2 = warp_sum(acc2);
        acc3 = warp_sum(acc3);
        acc4 = warp_sum(acc4);
        acc5 = warp_sum(acc5);
        acc6 = warp_sum(acc6);
        acc7 = warp_sum(acc7);
        if (tid == 0) {
            if (tn > 0) out[(size_t)(t0 + 0) * (size_t)n_cols + j] = acc0;
            if (tn > 1) out[(size_t)(t0 + 1) * (size_t)n_cols + j] = acc1;
            if (tn > 2) out[(size_t)(t0 + 2) * (size_t)n_cols + j] = acc2;
            if (tn > 3) out[(size_t)(t0 + 3) * (size_t)n_cols + j] = acc3;
            if (tn > 4) out[(size_t)(t0 + 4) * (size_t)n_cols + j] = acc4;
            if (tn > 5) out[(size_t)(t0 + 5) * (size_t)n_cols + j] = acc5;
            if (tn > 6) out[(size_t)(t0 + 6) * (size_t)n_cols + j] = acc6;
            if (tn > 7) out[(size_t)(t0 + 7) * (size_t)n_cols + j] = acc7;
        }
        __syncthreads();
    }
}

extern "C" __global__ void gemm_q6_k(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_rows,
    int n_cols,
    int col_bytes,
    int n_tok
) {
    int j = (int)blockIdx.x;
    if (j >= n_cols) return;
    const int tid = (int)threadIdx.x;
    const int nt = (int)blockDim.x;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    __shared__ float dq[256];
    const int TILE = 8;
    for (int t0 = 0; t0 < n_tok; t0 += TILE) {
        int tn = n_tok - t0;
        if (tn > TILE) tn = TILE;
        float acc0=0,acc1=0,acc2=0,acc3=0,acc4=0,acc5=0,acc6=0,acc7=0;
        int nb = n_rows / 256;
        for (int bi = 0; bi < nb; bi++) {
            dequant_q6_k_block_smem(col + bi * 210, dq, tid, nt);
            __syncthreads();
            int base = bi * 256;
            for (int i = tid; i < 256; i += nt) {
                float wv = dq[i];
                const float* xp = x + (size_t)t0 * (size_t)n_rows + base + i;
                if (tn > 0) acc0 += wv * xp[0 * n_rows];
                if (tn > 1) acc1 += wv * xp[1 * n_rows];
                if (tn > 2) acc2 += wv * xp[2 * n_rows];
                if (tn > 3) acc3 += wv * xp[3 * n_rows];
                if (tn > 4) acc4 += wv * xp[4 * n_rows];
                if (tn > 5) acc5 += wv * xp[5 * n_rows];
                if (tn > 6) acc6 += wv * xp[6 * n_rows];
                if (tn > 7) acc7 += wv * xp[7 * n_rows];
            }
            __syncthreads();
        }
        acc0 = warp_sum(acc0); acc1 = warp_sum(acc1);
        acc2 = warp_sum(acc2); acc3 = warp_sum(acc3);
        acc4 = warp_sum(acc4); acc5 = warp_sum(acc5);
        acc6 = warp_sum(acc6); acc7 = warp_sum(acc7);
        if (tid == 0) {
            if (tn > 0) out[(size_t)(t0 + 0) * (size_t)n_cols + j] = acc0;
            if (tn > 1) out[(size_t)(t0 + 1) * (size_t)n_cols + j] = acc1;
            if (tn > 2) out[(size_t)(t0 + 2) * (size_t)n_cols + j] = acc2;
            if (tn > 3) out[(size_t)(t0 + 3) * (size_t)n_cols + j] = acc3;
            if (tn > 4) out[(size_t)(t0 + 4) * (size_t)n_cols + j] = acc4;
            if (tn > 5) out[(size_t)(t0 + 5) * (size_t)n_cols + j] = acc5;
            if (tn > 6) out[(size_t)(t0 + 6) * (size_t)n_cols + j] = acc6;
            if (tn > 7) out[(size_t)(t0 + 7) * (size_t)n_cols + j] = acc7;
        }
        __syncthreads();
    }
}

extern "C" __global__ void gemm_q8_0(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_rows,
    int n_cols,
    int col_bytes,
    int n_tok
) {
    int j = (int)blockIdx.x;
    if (j >= n_cols) return;
    const int tid = (int)threadIdx.x;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    const int TILE = 8;
    for (int t0 = 0; t0 < n_tok; t0 += TILE) {
        int tn = n_tok - t0;
        if (tn > TILE) tn = TILE;
        float acc0=0,acc1=0,acc2=0,acc3=0,acc4=0,acc5=0,acc6=0,acc7=0;
        int nb = n_rows / 32;
        for (int bi = tid; bi < nb; bi += (int)blockDim.x) {
            const unsigned char* base = col + bi * 34;
            float d = half_to_float((unsigned short)(base[0] | (base[1] << 8)));
            const signed char* qs = (const signed char*)(base + 2);
            int yo = bi * 32;
            for (int t = 0; t < 32; t++) {
                float wv = (float)qs[t] * d;
                const float* xp = x + (size_t)t0 * (size_t)n_rows + yo + t;
                if (tn > 0) acc0 += wv * xp[0 * n_rows];
                if (tn > 1) acc1 += wv * xp[1 * n_rows];
                if (tn > 2) acc2 += wv * xp[2 * n_rows];
                if (tn > 3) acc3 += wv * xp[3 * n_rows];
                if (tn > 4) acc4 += wv * xp[4 * n_rows];
                if (tn > 5) acc5 += wv * xp[5 * n_rows];
                if (tn > 6) acc6 += wv * xp[6 * n_rows];
                if (tn > 7) acc7 += wv * xp[7 * n_rows];
            }
        }
        acc0 = warp_sum(acc0); acc1 = warp_sum(acc1);
        acc2 = warp_sum(acc2); acc3 = warp_sum(acc3);
        acc4 = warp_sum(acc4); acc5 = warp_sum(acc5);
        acc6 = warp_sum(acc6); acc7 = warp_sum(acc7);
        if (tid == 0) {
            if (tn > 0) out[(size_t)(t0 + 0) * (size_t)n_cols + j] = acc0;
            if (tn > 1) out[(size_t)(t0 + 1) * (size_t)n_cols + j] = acc1;
            if (tn > 2) out[(size_t)(t0 + 2) * (size_t)n_cols + j] = acc2;
            if (tn > 3) out[(size_t)(t0 + 3) * (size_t)n_cols + j] = acc3;
            if (tn > 4) out[(size_t)(t0 + 4) * (size_t)n_cols + j] = acc4;
            if (tn > 5) out[(size_t)(t0 + 5) * (size_t)n_cols + j] = acc5;
            if (tn > 6) out[(size_t)(t0 + 6) * (size_t)n_cols + j] = acc6;
            if (tn > 7) out[(size_t)(t0 + 7) * (size_t)n_cols + j] = acc7;
        }
        __syncthreads();
    }
}

extern "C" __global__ void gemm_q5_0(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_rows,
    int n_cols,
    int col_bytes,
    int n_tok
) {
    int j = (int)blockIdx.x;
    if (j >= n_cols) return;
    const int tid = (int)threadIdx.x;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    const int TILE = 8;
    float dq[32];
    for (int t0 = 0; t0 < n_tok; t0 += TILE) {
        int tn = n_tok - t0;
        if (tn > TILE) tn = TILE;
        float acc0=0,acc1=0,acc2=0,acc3=0,acc4=0,acc5=0,acc6=0,acc7=0;
        int nb = n_rows / 32;
        for (int bi = tid; bi < nb; bi += (int)blockDim.x) {
            dequant_q5_0_block(col + bi * 22, dq);
            int yo = bi * 32;
            for (int t = 0; t < 32; t++) {
                float wv = dq[t];
                const float* xp = x + (size_t)t0 * (size_t)n_rows + yo + t;
                if (tn > 0) acc0 += wv * xp[0 * n_rows];
                if (tn > 1) acc1 += wv * xp[1 * n_rows];
                if (tn > 2) acc2 += wv * xp[2 * n_rows];
                if (tn > 3) acc3 += wv * xp[3 * n_rows];
                if (tn > 4) acc4 += wv * xp[4 * n_rows];
                if (tn > 5) acc5 += wv * xp[5 * n_rows];
                if (tn > 6) acc6 += wv * xp[6 * n_rows];
                if (tn > 7) acc7 += wv * xp[7 * n_rows];
            }
        }
        acc0 = warp_sum(acc0); acc1 = warp_sum(acc1);
        acc2 = warp_sum(acc2); acc3 = warp_sum(acc3);
        acc4 = warp_sum(acc4); acc5 = warp_sum(acc5);
        acc6 = warp_sum(acc6); acc7 = warp_sum(acc7);
        if (tid == 0) {
            if (tn > 0) out[(size_t)(t0 + 0) * (size_t)n_cols + j] = acc0;
            if (tn > 1) out[(size_t)(t0 + 1) * (size_t)n_cols + j] = acc1;
            if (tn > 2) out[(size_t)(t0 + 2) * (size_t)n_cols + j] = acc2;
            if (tn > 3) out[(size_t)(t0 + 3) * (size_t)n_cols + j] = acc3;
            if (tn > 4) out[(size_t)(t0 + 4) * (size_t)n_cols + j] = acc4;
            if (tn > 5) out[(size_t)(t0 + 5) * (size_t)n_cols + j] = acc5;
            if (tn > 6) out[(size_t)(t0 + 6) * (size_t)n_cols + j] = acc6;
            if (tn > 7) out[(size_t)(t0 + 7) * (size_t)n_cols + j] = acc7;
        }
        __syncthreads();
    }
}
