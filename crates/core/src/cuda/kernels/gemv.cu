// Fast single-token GEMV aliases (same as gemm n_tok=1) — dedicated for lower overhead
extern "C" __global__ void gemv_q4_k(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_rows, int n_cols, int col_bytes
) {
    int j = (int)blockIdx.x;
    if (j >= n_cols) return;
    const int tid = (int)threadIdx.x;
    const int nt = (int)blockDim.x;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    __shared__ float dq[256];
    float acc = 0.f;
    int nb = n_rows / 256;
    for (int bi = 0; bi < nb; bi++) {
        dequant_q4_k_block_smem(col + bi * 144, dq, tid, nt);
        __syncthreads();
        int base = bi * 256;
        for (int i = tid; i < 256; i += nt) acc += dq[i] * x[base + i];
        __syncthreads();
    }
    acc = warp_sum(acc);
    if (tid == 0) out[j] = acc;
}

extern "C" __global__ void gemv_q6_k(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_rows, int n_cols, int col_bytes
) {
    int j = (int)blockIdx.x;
    if (j >= n_cols) return;
    const int tid = (int)threadIdx.x;
    const int nt = (int)blockDim.x;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    __shared__ float dq[256];
    float acc = 0.f;
    int nb = n_rows / 256;
    for (int bi = 0; bi < nb; bi++) {
        dequant_q6_k_block_smem(col + bi * 210, dq, tid, nt);
        __syncthreads();
        int base = bi * 256;
        for (int i = tid; i < 256; i += nt) acc += dq[i] * x[base + i];
        __syncthreads();
    }
    acc = warp_sum(acc);
    if (tid == 0) out[j] = acc;
}

extern "C" __global__ void gemv_q8_0(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_rows, int n_cols, int col_bytes
) {
    int j = (int)blockIdx.x;
    if (j >= n_cols) return;
    const int tid = (int)threadIdx.x;
    const unsigned char* col = w + (size_t)j * (size_t)col_bytes;
    float acc = 0.f;
    int nb = n_rows / 32;
    for (int bi = tid; bi < nb; bi += (int)blockDim.x) {
        const unsigned char* base = col + bi * 34;
        float d = half_to_float((unsigned short)(base[0] | (base[1] << 8)));
        const signed char* qs = (const signed char*)(base + 2);
        int yo = bi * 32;
        for (int t = 0; t < 32; t++) acc += (float)qs[t] * d * x[yo + t];
    }
    acc = warp_sum(acc);
    if (tid == 0) out[j] = acc;
}
