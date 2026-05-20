#include <torch/extension.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void quantize_fp16_to_int8_kernel(
    const __half* __restrict__ in, int8_t* __restrict__ out,
    float inv_scale, int64_t total)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t s = (int64_t)blockDim.x * gridDim.x;
    for (; i < total; i += s) {
        float v = __half2float(in[i]) * inv_scale;
        int iv = (int)roundf(v);
        iv = max(-128, min(127, iv));
        out[i] = (int8_t)iv;
    }
}

__global__ void unpack_int4_to_int8_kernel(
    const uint8_t* __restrict__ packed, int8_t* __restrict__ unpacked,
    int N, int K2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K2;
    int s = blockDim.x * gridDim.x;
    int K = K2 * 2;
    for (; i < total; i += s) {
        int r = i / K2, c2 = i % K2;
        uint8_t b = packed[i];
        int8_t lo = b & 0xF; lo = (lo >= 8) ? lo - 16 : lo;
        int8_t hi = (b >> 4) & 0xF; hi = (hi >= 8) ? hi - 16 : hi;
        unpacked[r * K + 2 * c2]     = lo;
        unpacked[r * K + 2 * c2 + 1] = hi;
    }
}

__global__ void int8_wmma_matmul_kernel(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B,
    int32_t* __restrict__ C, int M_pad, int N_pad, int K)
{
    int wid = threadIdx.x / 32;
    int wm = (blockIdx.x * 2 + (wid >> 1)) * WMMA_M;
    int wn = (blockIdx.y * 2 + (wid & 1)) * WMMA_N;
    if (wm >= M_pad || wn >= N_pad) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> acc;
    wmma::fill_fragment(acc, 0);

    for (int k = 0; k < K; k += WMMA_K) {
        wmma::load_matrix_sync(af, A + wm * K + k, K);
        wmma::load_matrix_sync(bf, B + wn * K + k, K);
        wmma::mma_sync(acc, af, bf, acc);
    }
    wmma::store_matrix_sync(C + wm * N_pad + wn, acc, N_pad, wmma::mem_row_major);
}

__global__ void scale_int32_to_fp16_kernel(
    const int32_t* __restrict__ in, __half* __restrict__ out,
    float asc, const __half* __restrict__ wsc, const __half* __restrict__ bias,
    int M, int N, bool has_bias)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;
    float v = (float)in[m * N + n] * asc * __half2float(wsc[n]);
    if (has_bias) v += __half2float(bias[n]);
    out[m * N + n] = __float2half(v);
}

torch::Tensor int4_linear_cuda(
    torch::Tensor input, torch::Tensor packed_weights,
    torch::Tensor act_scale, torch::Tensor wt_scale, torch::Tensor bias)
{
    auto in2d = input.reshape({-1, input.size(-1)});
    int M = in2d.size(0), K = in2d.size(1);
    int N = packed_weights.size(0), K2 = packed_weights.size(1);
    TORCH_CHECK(K2 * 2 == K, "K mismatch");
    bool hb = (bias.numel() > 0);

    int M_pad = ((M + 15) / 16) * 16;
    int N_pad = ((N + 15) / 16) * 16;
    cudaStream_t s = 0;

    auto a_i8 = torch::zeros({M_pad, K}, torch::dtype(torch::kInt8).device(input.device()));
    {
        float isc = 1.0f / act_scale.item<float>();
        int64_t tot = (int64_t)M * K;
        int b = 256, g = std::min((int)((tot + b - 1) / b), 65535);
        quantize_fp16_to_int8_kernel<<<g, b, 0, s>>>(
            (const __half*)in2d.data_ptr<at::Half>(), a_i8.data_ptr<int8_t>(), isc, tot);
    }

    auto w_i8 = torch::zeros({N_pad, K}, torch::dtype(torch::kInt8).device(input.device()));
    {
        int b = 256, g = std::min((N * K2 + b - 1) / b, 65535);
        unpack_int4_to_int8_kernel<<<g, b, 0, s>>>(
            packed_weights.data_ptr<uint8_t>(), w_i8.data_ptr<int8_t>(), N, K2);
    }

    auto c_i32 = torch::zeros({M_pad, N_pad}, torch::dtype(torch::kInt32).device(input.device()));
    {
        int gm = (M_pad + 2 * WMMA_M - 1) / (2 * WMMA_M);
        int gn = (N_pad + 2 * WMMA_N - 1) / (2 * WMMA_N);
        int8_wmma_matmul_kernel<<<dim3(gm, gn), 128, 0, s>>>(
            a_i8.data_ptr<int8_t>(), w_i8.data_ptr<int8_t>(),
            c_i32.data_ptr<int32_t>(), M_pad, N_pad, K);
    }

    auto out = torch::empty({M, N}, torch::dtype(torch::kFloat16).device(input.device()));
    {
        float asc = act_scale.item<float>();
        scale_int32_to_fp16_kernel<<<dim3((N+31)/32, (M+31)/32), dim3(32, 32), 0, s>>>(
            c_i32.data_ptr<int32_t>(), (__half*)out.data_ptr<at::Half>(),
            asc, (const __half*)wt_scale.data_ptr<at::Half>(),
            hb ? (const __half*)bias.data_ptr<at::Half>() : nullptr, M, N, hb);
    }

    auto sz = input.sizes().vec(); sz.back() = N;
    return out.reshape(sz);
}
