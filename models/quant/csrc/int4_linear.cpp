#include <torch/extension.h>

std::vector<torch::Tensor> fused_prepare_cuda(
    torch::Tensor input, torch::Tensor packed_w,
    torch::Tensor w_scale, torch::Tensor inv_smooth_scale);

torch::Tensor dequant_int4_cuda(torch::Tensor packed, torch::Tensor scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_prepare", &fused_prepare_cuda,
          "Fused smooth_scale + dequant in one call → (x_prep, w_deq)",
          py::arg("input"), py::arg("packed_w"),
          py::arg("w_scale"), py::arg("inv_smooth_scale"));
    m.def("dequant", &dequant_int4_cuda,
          "int4 → fp16 dequant",
          py::arg("packed"), py::arg("scale"));
}
