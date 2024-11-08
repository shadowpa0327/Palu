"""
Modified from https://github.com/jy-yuan/KIVI/blob/main/quant/setup.py
"""
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


extra_compile_args = {
    "cxx": [
        "-g", 
        "-O3", 
        "-fopenmp", 
        "-lgomp", 
        "-std=c++17",
        "-DENABLE_BF16"
    ],
    "nvcc": [
        # "-O0", "-G", "-g", # uncomment for debugging
        "-O3",
        "-std=c++17",
        "-DENABLE_BF16",  # TODO
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--threads=8"
    ],
}

cuda_kernel_sources = [
    "csrc/palu_gemm_cuda.cu",
    "csrc/palu_gemm_outer_cuda.cu",
    "csrc/pybind.cpp",
]

setup(
    name="palu_kernel",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="palu_kernel",
            sources=cuda_kernel_sources,
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)