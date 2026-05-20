#include <torch/extension.h>

torch::Tensor int4_linear_cuda(
    torch::Tensor input,
    torch::Tensor packed_weights,
    torch::Tensor act_scale,
    torch::Tensor wt_scale,
    torch::Tensor bias
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &int4_linear_cuda,
          "W4A8 int4 linear via WMMA int8 Tensor Cores",
          py::arg("input"),
          py::arg("packed_weights"),
          py::arg("act_scale"),
          py::arg("wt_scale"),
          py::arg("bias"));
}
