from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

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
        "-O3",
        "-std=c++17",
        "-DENABLE_BF16",
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

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="palu",
    version="0.1",
    description="Palu package with CUDA extension",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="palu.palu_cuda",
            sources=[
                "palu/csrc/palu_gemm_cuda.cu",
                "palu/csrc/palu_gemm_outer_cuda.cu",
                "palu/csrc/pybind.cpp",
            ],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=requirements, # Load requirements from requirements.txt
)
