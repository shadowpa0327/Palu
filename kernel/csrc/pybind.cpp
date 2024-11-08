#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "palu_gemm_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("batched_gemm_forward_cuda", &batched_gemm_forward_cuda);
  m.def("batched_gemm_forward_outer_cuda", &batched_gemm_forward_outer_cuda);
}