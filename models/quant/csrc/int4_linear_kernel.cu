#include <torch/extension.h>
#include <cuda_fp16.h>

__global__ void dequant_int4_kernel(
    const uint8_t* __restrict__ packed,
    __half* __restrict__ output,
    const __half* __restrict__ scale,
    int N, int K2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K2;
    int stride = blockDim.x * gridDim.x;
    int K = K2 * 2;
    for (int i = idx; i < total; i += stride) {
        int row = i / K2, col2 = i % K2;
        uint8_t byte = packed[i];
        int8_t lo = byte & 0xF; lo = (lo >= 8) ? lo - 16 : lo;
        int8_t hi = (byte >> 4) & 0xF; hi = (hi >= 8) ? hi - 16 : hi;
        float s = __half2float(scale[row]);
        output[row * K + 2 * col2]     = __float2half((float)lo * s);
        output[row * K + 2 * col2 + 1] = __float2half((float)hi * s);
    }
}

torch::Tensor dequant_int4_cuda(
    torch::Tensor packed, torch::Tensor scale)
{
    int N = packed.size(0), K2 = packed.size(1);
    auto out = torch::empty({N, K2 * 2}, packed.options().dtype(c10::kHalf));
    int total = N * K2;
    int block = 256, grid = std::min((total + block - 1) / block, 65535);
    dequant_int4_kernel<<<grid, block>>>(
        packed.data_ptr<uint8_t>(),
        (__half*)out.data_ptr<at::Half>(),
        (const __half*)scale.data_ptr<at::Half>(),
        N, K2);
    return out;
}
