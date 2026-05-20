#include <torch/extension.h>

torch::Tensor dequant_int4_cuda(torch::Tensor packed, torch::Tensor scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequant", &dequant_int4_cuda,
          "int4→fp16 dequant for cuBLAS fp16 matmul",
          py::arg("packed"), py::arg("scale"));
}
