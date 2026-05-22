#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

// ============================================================
// Kernel 1: dequant int4 → half2 fp16 (user's optimized version)
// ============================================================
__global__ void dequant_int4_kernel_half2(
    const uint8_t* __restrict__ packed,
    __half2* __restrict__ output,
    const __half* __restrict__ scale,
    int N, int K2)
{
    int row = blockIdx.y;
    int col2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col2 >= K2) return;

    int offset = row * K2 + col2;
    uint8_t byte = packed[offset];

    int lo = byte & 0x0f;  lo = lo >= 8 ? lo - 16 : lo;
    int hi = (byte >> 4) & 0x0f;  hi = hi >= 8 ? hi - 16 : hi;

    __half2 q = __halves2half2(__int2half_rn(lo), __int2half_rn(hi));
    __half2 s = __half2half2(scale[row]);
    output[offset] = __hmul2(q, s);
}

// ============================================================
// Kernel 2: x /= smooth_scale  (half2 vectorized)
// ============================================================
__global__ void smooth_scale_kernel_half2(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    const __half* __restrict__ inv_smooth_scale,
    int M, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * (K / 2);
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        int row = i / (K / 2);
        int col2 = i % (K / 2);
        int col = col2 * 2;
        __half2 val = *((const __half2*)(input + row * K + col));
        __half2 ss = *((const __half2*)(inv_smooth_scale + col));
        *((__half2*)(output + row * K + col)) = __hmul2(val, ss);
    }
}

// ============================================================
// Fused prepare: smooth_scale + dequant in one C++ call
// Returns: {x_prepared[M,K] fp16, w_deq[N,K] fp16}
// ============================================================
std::vector<torch::Tensor> fused_prepare_cuda(
    torch::Tensor input,            // [M, K] fp16
    torch::Tensor packed_w,         // [N, K/2] uint8
    torch::Tensor w_scale,          // [N] fp16
    torch::Tensor inv_smooth_scale  // [K] fp16 or empty
) {
    int M = input.size(0), K = input.size(1);
    int N = packed_w.size(0), K2 = packed_w.size(1);
    bool has_smooth = (inv_smooth_scale.numel() > 0);
    auto stream = at::cuda::getCurrentCUDAStream();

    // Step 1: smooth_scale
    auto x_prep = torch::empty({M, K}, input.options());
    if (has_smooth) {
        int total = M * (K / 2);
        int block = 256;
        dim3 grid((total + block - 1) / block);
        smooth_scale_kernel_half2<<<grid, block, 0, stream>>>(
            (const __half*)input.data_ptr<at::Half>(),
            (__half*)x_prep.data_ptr<at::Half>(),
            (const __half*)inv_smooth_scale.data_ptr<at::Half>(),
            M, K);
    } else {
        x_prep.copy_(input);
    }

    // Step 2: dequant
    auto w_deq = torch::empty({N, K2 * 2}, packed_w.options().dtype(torch::kHalf));
    int block_w = 256;
    dim3 grid_w((K2 + block_w - 1) / block_w, N);
    dequant_int4_kernel_half2<<<grid_w, block_w, 0, stream>>>(
        packed_w.data_ptr<uint8_t>(),
        reinterpret_cast<__half2*>(w_deq.data_ptr<at::Half>()),
        (const __half*)w_scale.data_ptr<at::Half>(),
        N, K2);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {x_prep, w_deq};
}

// ============================================================
// Standalone dequant (kept for compatibility)
// ============================================================
torch::Tensor dequant_int4_cuda(torch::Tensor packed, torch::Tensor scale) {
    int N = packed.size(0), K2 = packed.size(1);
    auto out = torch::empty({N, K2 * 2}, packed.options().dtype(torch::kHalf));
    int block = 256;
    dim3 grid((K2 + block - 1) / block, N);
    dequant_int4_kernel_half2<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<__half2*>(out.data_ptr<at::Half>()),
        (const __half*)scale.data_ptr<at::Half>(),
        N, K2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
