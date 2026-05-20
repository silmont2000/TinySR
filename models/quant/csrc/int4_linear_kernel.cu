#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

__global__ void dequant_int4_kernel_half2(
    const uint8_t* __restrict__ packed,
    __half2* __restrict__ output,
    const __half* __restrict__ scale,
    int N,
    int K2
) {
    int row = blockIdx.y;
    int col2 = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col2 >= K2) {
        return;
    }

    int offset = row * K2 + col2;
    uint8_t byte = packed[offset];

    int lo = byte & 0x0f;
    int hi = (byte >> 4) & 0x0f;

    lo = lo >= 8 ? lo - 16 : lo;
    hi = hi >= 8 ? hi - 16 : hi;

    __half2 q = __halves2half2(
        __int2half_rn(lo),
        __int2half_rn(hi)
    );

    __half2 s = __half2half2(scale[row]);

    output[offset] = __hmul2(q, s);
}

torch::Tensor dequant_int4_cuda(
    torch::Tensor packed,
    torch::Tensor scale
) {
    TORCH_CHECK(packed.is_cuda(), "packed must be CUDA tensor");
    TORCH_CHECK(scale.is_cuda(), "scale must be CUDA tensor");
    TORCH_CHECK(packed.dtype() == torch::kUInt8, "packed must be uint8");
    TORCH_CHECK(scale.dtype() == torch::kHalf, "scale must be float16");
    TORCH_CHECK(packed.is_contiguous(), "packed must be contiguous");
    TORCH_CHECK(scale.is_contiguous(), "scale must be contiguous");
    TORCH_CHECK(packed.dim() == 2, "packed must be 2D");
    TORCH_CHECK(scale.dim() == 1, "scale must be 1D");
    TORCH_CHECK(scale.size(0) == packed.size(0), "scale size mismatch");

    int N = packed.size(0);
    int K2 = packed.size(1);

    auto out = torch::empty(
        {N, K2 * 2},
        packed.options().dtype(torch::kHalf)
    );

    int block = 256;
    dim3 grid((K2 + block - 1) / block, N);

    dequant_int4_kernel_half2<<<
        grid,
        block,
        0,
        at::cuda::getCurrentCUDAStream()
    >>>(
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<__half2*>(out.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(scale.data_ptr<at::Half>()),
        N,
        K2
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
