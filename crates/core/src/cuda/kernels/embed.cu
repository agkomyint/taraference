// Decode embed: token id as kernel arg (no H→D buffer).
// *_one_d variants read token from device memory (CUDA-graph safe).

extern "C" __global__ void embed_q4_0_one(
    const unsigned char* __restrict__ table,
    float* __restrict__ out,
    int token,
    int n_rows,
    int col_bytes
) {
    const unsigned char* col = table + (size_t)token * (size_t)col_bytes;
    const int nb = n_rows / 32;
    for (int bi = (int)threadIdx.x; bi < nb; bi += (int)blockDim.x) {
        const unsigned char* base = col + bi * q4_0_block_bytes(n_rows, col_bytes);
        const float d = half_to_float((unsigned short)(base[0] | (base[1] << 8)));
        const unsigned char* qs = base + (q4_0_block_bytes(n_rows, col_bytes) >= 32 ? 4 : 2);
        const int yo = bi * 32;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const unsigned char q = qs[j];
            out[yo + j] = d * (float)((int)(q & 0x0f) - 8);
            out[yo + 16 + j] = d * (float)((int)(q >> 4) - 8);
        }
    }
}

extern "C" __global__ void embed_q4_0_one_d(
    const unsigned char* __restrict__ table,
    float* __restrict__ out,
    const int* __restrict__ token_ptr,
    int n_rows,
    int col_bytes
) {
    const int token = token_ptr[0];
    const unsigned char* col = table + (size_t)token * (size_t)col_bytes;
    const int nb = n_rows / 32;
    for (int bi = (int)threadIdx.x; bi < nb; bi += (int)blockDim.x) {
        const unsigned char* base = col + bi * q4_0_block_bytes(n_rows, col_bytes);
        const float d = half_to_float((unsigned short)(base[0] | (base[1] << 8)));
        const unsigned char* qs = base + (q4_0_block_bytes(n_rows, col_bytes) >= 32 ? 4 : 2);
        const int yo = bi * 32;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const unsigned char q = qs[j];
            out[yo + j] = d * (float)((int)(q & 0x0f) - 8);
            out[yo + 16 + j] = d * (float)((int)(q >> 4) - 8);
        }
    }
}

extern "C" __global__ void embed_q4_k_one(
    const unsigned char* __restrict__ table,
    float* __restrict__ out,
    int token,
    int n_rows,
    int col_bytes
) {
    const unsigned char* col = table + (size_t)token * (size_t)col_bytes;
    const int tid = (int)threadIdx.x;
    const int nt = (int)blockDim.x;
    __shared__ float dq[256];
    int nb = n_rows / 256;
    for (int bi = 0; bi < nb; bi++) {
        dequant_q4_k_block_smem(col + bi * 144, dq, tid, nt);
        __syncthreads();
        for (int i = tid; i < 256; i += nt) out[bi * 256 + i] = dq[i];
        __syncthreads();
    }
}

extern "C" __global__ void embed_q4_k_one_d(
    const unsigned char* __restrict__ table,
    float* __restrict__ out,
    const int* __restrict__ token_ptr,
    int n_rows,
    int col_bytes
) {
    int token = token_ptr[0];
    const unsigned char* col = table + (size_t)token * (size_t)col_bytes;
    const int tid = (int)threadIdx.x;
    const int nt = (int)blockDim.x;
    __shared__ float dq[256];
    int nb = n_rows / 256;
    for (int bi = 0; bi < nb; bi++) {
        dequant_q4_k_block_smem(col + bi * 144, dq, tid, nt);
        __syncthreads();
        for (int i = tid; i < 256; i += nt) out[bi * 256 + i] = dq[i];
        __syncthreads();
    }
}

extern "C" __global__ void embed_q6_k_one(
    const unsigned char* __restrict__ table,
    float* __restrict__ out,
    int token,
    int n_rows,
    int col_bytes
) {
    const unsigned char* col = table + (size_t)token * (size_t)col_bytes;
    const int tid = (int)threadIdx.x;
    const int nt = (int)blockDim.x;
    __shared__ float dq[256];
    int nb = n_rows / 256;
    for (int bi = 0; bi < nb; bi++) {
        dequant_q6_k_block_smem(col + bi * 210, dq, tid, nt);
        __syncthreads();
        for (int i = tid; i < 256; i += nt) out[bi * 256 + i] = dq[i];
        __syncthreads();
    }
}

extern "C" __global__ void embed_q6_k_one_d(
    const unsigned char* __restrict__ table,
    float* __restrict__ out,
    const int* __restrict__ token_ptr,
    int n_rows,
    int col_bytes
) {
    int token = token_ptr[0];
    const unsigned char* col = table + (size_t)token * (size_t)col_bytes;
    const int tid = (int)threadIdx.x;
    const int nt = (int)blockDim.x;
    __shared__ float dq[256];
    int nb = n_rows / 256;
    for (int bi = 0; bi < nb; bi++) {
        dequant_q6_k_block_smem(col + bi * 210, dq, tid, nt);
        __syncthreads();
        for (int i = tid; i < 256; i += nt) out[bi * 256 + i] = dq[i];
        __syncthreads();
    }
}

extern "C" __global__ void embed_q8_0_one(
    const unsigned char* __restrict__ table,
    float* __restrict__ out,
    int token,
    int n_rows,
    int col_bytes
) {
    const unsigned char* col = table + (size_t)token * (size_t)col_bytes;
    int nb = n_rows / 32;
    for (int bi = (int)threadIdx.x; bi < nb; bi += (int)blockDim.x) {
        const unsigned char* base = col + bi * 34;
        float d = half_to_float((unsigned short)(base[0] | (base[1] << 8)));
        const signed char* qs = (const signed char*)(base + 2);
        int yo = bi * 32;
        for (int j = 0; j < 32; j++) out[yo + j] = (float)qs[j] * d;
    }
}

extern "C" __global__ void embed_q8_0_one_d(
    const unsigned char* __restrict__ table,
    float* __restrict__ out,
    const int* __restrict__ token_ptr,
    int n_rows,
    int col_bytes
) {
    int token = token_ptr[0];
    const unsigned char* col = table + (size_t)token * (size_t)col_bytes;
    int nb = n_rows / 32;
    for (int bi = (int)threadIdx.x; bi < nb; bi += (int)blockDim.x) {
        const unsigned char* base = col + bi * 34;
        float d = half_to_float((unsigned short)(base[0] | (base[1] << 8)));
        const signed char* qs = (const signed char*)(base + 2);
        int yo = bi * 32;
        for (int j = 0; j < 32; j++) out[yo + j] = (float)qs[j] * d;
    }
}

extern "C" __global__ void embed_q5_0_one(
    const unsigned char* __restrict__ table,
    float* __restrict__ out,
    int token,
    int n_rows,
    int col_bytes
) {
    const unsigned char* col = table + (size_t)token * (size_t)col_bytes;
    int nb = n_rows / 32;
    float dq[32];
    for (int bi = (int)threadIdx.x; bi < nb; bi += (int)blockDim.x) {
        dequant_q5_0_block(col + bi * 22, dq);
        int yo = bi * 32;
        for (int j = 0; j < 32; j++) out[yo + j] = dq[j];
    }
}

extern "C" __global__ void embed_q5_0_one_d(
    const unsigned char* __restrict__ table,
    float* __restrict__ out,
    const int* __restrict__ token_ptr,
    int n_rows,
    int col_bytes
) {
    int token = token_ptr[0];
    const unsigned char* col = table + (size_t)token * (size_t)col_bytes;
    int nb = n_rows / 32;
    float dq[32];
    for (int bi = (int)threadIdx.x; bi < nb; bi += (int)blockDim.x) {
        dequant_q5_0_block(col + bi * 22, dq);
        int yo = bi * 32;
        for (int j = 0; j < 32; j++) out[yo + j] = dq[j];
    }
}

extern "C" __global__ void embed_q5_k_one(
    const unsigned char* __restrict__ table,
    float* __restrict__ out,
    int token,
    int n_rows,
    int col_bytes
) {
    const unsigned char* col = table + (size_t)token * (size_t)col_bytes;
    const int tid = (int)threadIdx.x;
    const int nt = (int)blockDim.x;
    __shared__ float dq[256];
    int nb = n_rows / 256;
    for (int bi = 0; bi < nb; bi++) {
        dequant_q5_k_block_smem(col + bi * 176, dq, tid, nt);
        __syncthreads();
        for (int i = tid; i < 256; i += nt) out[bi * 256 + i] = dq[i];
        __syncthreads();
    }
}

extern "C" __global__ void embed_q5_k_one_d(
    const unsigned char* __restrict__ table,
    float* __restrict__ out,
    const int* __restrict__ token_ptr,
    int n_rows,
    int col_bytes
) {
    int token = token_ptr[0];
    const unsigned char* col = table + (size_t)token * (size_t)col_bytes;
    const int tid = (int)threadIdx.x;
    const int nt = (int)blockDim.x;
    __shared__ float dq[256];
    int nb = n_rows / 256;
    for (int bi = 0; bi < nb; bi++) {
        dequant_q5_k_block_smem(col + bi * 176, dq, tid, nt);
        __syncthreads();
        for (int i = tid; i < 256; i += nt) out[bi * 256 + i] = dq[i];
        __syncthreads();
    }
}

// Prefill embed: one block per token, warp dequants column into out[t*n_rows]
extern "C" __global__ void embed_q4_0(
    const unsigned char* __restrict__ table,
    float* __restrict__ out,
    const int* __restrict__ tokens,
    int n_tok,
    int n_rows,
    int col_bytes
) {
    const int t = (int)blockIdx.x;
    if (t >= n_tok) return;
    const int token = tokens[t];
    const unsigned char* col = table + (size_t)token * (size_t)col_bytes;
    float* dst = out + (size_t)t * (size_t)n_rows;
    const int nb = n_rows / 32;
    for (int bi = (int)threadIdx.x; bi < nb; bi += (int)blockDim.x) {
        const unsigned char* base = col + bi * q4_0_block_bytes(n_rows, col_bytes);
        const float d = half_to_float((unsigned short)(base[0] | (base[1] << 8)));
        const unsigned char* qs = base + (q4_0_block_bytes(n_rows, col_bytes) >= 32 ? 4 : 2);
        const int yo = bi * 32;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const unsigned char q = qs[j];
            dst[yo + j] = d * (float)((int)(q & 0x0f) - 8);
            dst[yo + 16 + j] = d * (float)((int)(q >> 4) - 8);
        }
    }
}

extern "C" __global__ void embed_q4_k(
    const unsigned char* __restrict__ table,
    float* __restrict__ out,
    const int* __restrict__ tokens,
    int n_tok,
    int n_rows,
    int col_bytes
) {
    int t = (int)blockIdx.x;
    if (t >= n_tok) return;
    int token = tokens[t];
    const unsigned char* col = table + (size_t)token * (size_t)col_bytes;
    float* dst = out + (size_t)t * (size_t)n_rows;
    const int tid = (int)threadIdx.x;
    const int nt = (int)blockDim.x;
    __shared__ float dq[256];
    int nb = n_rows / 256;
    for (int bi = 0; bi < nb; bi++) {
        dequant_q4_k_block_smem(col + bi * 144, dq, tid, nt);
        __syncthreads();
        for (int i = tid; i < 256; i += nt) dst[bi * 256 + i] = dq[i];
        __syncthreads();
    }
}

extern "C" __global__ void embed_q6_k(
    const unsigned char* __restrict__ table,
    float* __restrict__ out,
    const int* __restrict__ tokens,
    int n_tok,
    int n_rows,
    int col_bytes
) {
    int t = (int)blockIdx.x;
    if (t >= n_tok) return;
    int token = tokens[t];
    const unsigned char* col = table + (size_t)token * (size_t)col_bytes;
    float* dst = out + (size_t)t * (size_t)n_rows;
    const int tid = (int)threadIdx.x;
    const int nt = (int)blockDim.x;
    __shared__ float dq[256];
    int nb = n_rows / 256;
    for (int bi = 0; bi < nb; bi++) {
        dequant_q6_k_block_smem(col + bi * 210, dq, tid, nt);
        __syncthreads();
        for (int i = tid; i < 256; i += nt) dst[bi * 256 + i] = dq[i];
        __syncthreads();
    }
}

extern "C" __global__ void embed_q8_0(
    const unsigned char* __restrict__ table,
    float* __restrict__ out,
    const int* __restrict__ tokens,
    int n_tok,
    int n_rows,
    int col_bytes
) {
    int t = (int)blockIdx.x;
    if (t >= n_tok) return;
    int token = tokens[t];
    const unsigned char* col = table + (size_t)token * (size_t)col_bytes;
    float* dst = out + (size_t)t * (size_t)n_rows;
    int nb = n_rows / 32;
    for (int bi = (int)threadIdx.x; bi < nb; bi += (int)blockDim.x) {
        const unsigned char* base = col + bi * 34;
        float d = half_to_float((unsigned short)(base[0] | (base[1] << 8)));
        const signed char* qs = (const signed char*)(base + 2);
        int yo = bi * 32;
        for (int j = 0; j < 32; j++) dst[yo + j] = (float)qs[j] * d;
    }
}

extern "C" __global__ void embed_q5_0(
    const unsigned char* __restrict__ table,
    float* __restrict__ out,
    const int* __restrict__ tokens,
    int n_tok,
    int n_rows,
    int col_bytes
) {
    int t = (int)blockIdx.x;
    if (t >= n_tok) return;
    int token = tokens[t];
    const unsigned char* col = table + (size_t)token * (size_t)col_bytes;
    float* dst = out + (size_t)t * (size_t)n_rows;
    int nb = n_rows / 32;
    float dq[32];
    for (int bi = (int)threadIdx.x; bi < nb; bi += (int)blockDim.x) {
        dequant_q5_0_block(col + bi * 22, dq);
        int yo = bi * 32;
        for (int j = 0; j < 32; j++) dst[yo + j] = dq[j];
    }
}

extern "C" __global__ void embed_q5_k(
    const unsigned char* __restrict__ table,
    float* __restrict__ out,
    const int* __restrict__ tokens,
    int n_tok,
    int n_rows,
    int col_bytes
) {
    int t = (int)blockIdx.x;
    if (t >= n_tok) return;
    int token = tokens[t];
    const unsigned char* col = table + (size_t)token * (size_t)col_bytes;
    float* dst = out + (size_t)t * (size_t)n_rows;
    const int tid = (int)threadIdx.x;
    const int nt = (int)blockDim.x;
    __shared__ float dq[256];
    int nb = n_rows / 256;
    for (int bi = 0; bi < nb; bi++) {
        dequant_q5_k_block_smem(col + bi * 176, dq, tid, nt);
        __syncthreads();
        for (int i = tid; i < 256; i += nt) dst[bi * 256 + i] = dq[i];
        __syncthreads();
    }
}
