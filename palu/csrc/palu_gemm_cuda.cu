#include <cuda_fp16.h>
#include <stdio.h>
#include <torch/extension.h>
#include "palu_gemm_cuda.h"

#define BM 4              // Tile size in the M dimension (number of rows per tile)
#define BK 128            // Tile size in the K dimension (group size for quantization)
#define BN 16             // Tile size in the N dimension (number of columns per tile)
#define PACK_FACTOR 8     // Number of elements packed together (for vectorized loads)

// Kernel function to perform batched GEMM with quantized weights
__global__ void batched_gemm_kernel_quantized(
    const float4* __restrict__ A_packed,        // Packed input activations [B, M, K / PACK_FACTOR]
    const uint32_t* __restrict__ qB_packed,     // Packed quantized weights [B, N, K / PACK_FACTOR]
    const half* __restrict__ zeros,             // Zero offsets for quantization [B, N, K / group_size]
    const half* __restrict__ scaling_factors,   // Scaling factors for quantization [B, N, K / group_size]
    half* __restrict__ C,                       // Output matrix [B, M, N]
    const int B, const int M, const int N, const int K,
    const int group_size) {
    
    // Compute batch index
    const int batch_idx = blockIdx.z;

    // Tile indices along the N (columns) and M (rows) dimensions
    const int tile_idx_N = blockIdx.x; // Tile index along the N dimension
    const int tile_idx_M = blockIdx.y; // Tile index along the M dimension

    // Each thread computes one element in the output tile of size BM x BN
    const int thread_id = threadIdx.x;          // Thread index within the block
    const int thread_row = thread_id / BN;      // Row index within the tile [0, BM)
    const int thread_col = thread_id % BN;      // Column index within the tile [0, BN)

    // Compute global indices in the M and N dimensions
    const int global_row = tile_idx_M * BM + thread_row; // Global row index in the output matrix
    const int global_col = tile_idx_N * BN + thread_col; // Global column index in the output matrix

    // Bounds check to prevent out-of-bounds memory access
    if (global_row >= M || global_col >= N || batch_idx >= B)
        return;

    // Pointer to the output element computed by this thread
    half* output_ptr = C + batch_idx * (M * N) + global_row * N + global_col;

    // Initialize the partial sum for the dot product
    float partial_sum = 0.0f;

    // Number of groups along the K dimension (since we process group_size elements at a time)
    const int num_groups = K / group_size;

    // Loop over each group along the K dimension
    for (int group_idx = 0; group_idx < num_groups; group_idx++) {

        // Compute index for scaling factors and zeros
        int sf_zero_idx = batch_idx * (N * num_groups) + global_col * num_groups + group_idx;

        // Load the scaling factor and zero point for the current group
        float scaling_factor = __half2float(scaling_factors[sf_zero_idx]);
        float zero_point = __half2float(zeros[sf_zero_idx]);

        // Number of iterations within the group (since we process 32 elements per iteration)
        const int iterations = group_size / 32;

        // Loop over iterations within the group
        #pragma unroll
        for (int iter = 0; iter < iterations; iter++) {
            // Calculate offsets for qB_packed and A_packed
            int qB_offset = batch_idx * (N * K / PACK_FACTOR) +
                            global_col * (K / PACK_FACTOR) +
                            group_idx * (group_size / PACK_FACTOR) +
                            iter * 4; // 4 because we load 4 uint32_t (128 bits) at a time

            int A_offset = batch_idx * (M * K / PACK_FACTOR) +
                           global_row * (K / PACK_FACTOR) +
                           group_idx * (group_size / PACK_FACTOR) +
                           iter * 4; // Matching offset for A_packed

            // Load 128 bits (4 uint32_t) of packed quantized weights from qB_packed
            uint32_t qB_values[4];
            *((float4*)(qB_values)) = *((float4*)(qB_packed + qB_offset));

            // Process each of the 4 uint32_t values
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint32_t packed_weights = qB_values[j];

                // Load 128 bits (8 half-precision floats) of activations from A_packed
                float4 A_packed_value = A_packed[A_offset + j];
                half* A_values = reinterpret_cast<half*>(&A_packed_value); // Access as half*

                // Process 8 elements (since each uint32_t packs 8 quantized weights)
                #pragma unroll
                for (int k = 0; k < PACK_FACTOR; k++) {
                    // Extract a 4-bit quantized weight
                    float quantized_weight = static_cast<float>(packed_weights & 0xF);

                    // Dequantize the weight
                    float dequantized_weight = quantized_weight * scaling_factor + zero_point;

                    // Multiply with the activation and accumulate the result
                    partial_sum += dequantized_weight * __half2float(A_values[k]);

                    // Shift to the next 4-bit quantized weight
                    packed_weights >>= 4;
                }
            }
        }
    }

    // Store the computed partial sum as the output element
    *output_ptr = __float2half(partial_sum);
}

// Host function to launch the kernel
torch::Tensor batched_gemm_forward_cuda(
    torch::Tensor _A,                 // Input activations tensor [B, M, K]
    torch::Tensor _qB,                // Packed quantized weights tensor [B, N, K / PACK_FACTOR]
    torch::Tensor _scaling_factors,   // Scaling factors tensor [B, N, K / group_size]
    torch::Tensor _zeros,             // Zero points tensor [B, N, K / group_size]
    const int bit,                    // Bit-width for quantization (e.g., 4 bits)
    const int group_size) {           // Group size used for quantization

    // Extract input tensor dimensions
    int B = _A.size(0); // Batch size
    int M = _A.size(1); // Number of rows in A (and C)
    int K = _A.size(2); // Number of columns in A and rows in qB
    int N = _qB.size(1); // Number of columns in qB (and C)

    // Ensure that K is divisible by PACK_FACTOR and group_size
    TORCH_CHECK(K % PACK_FACTOR == 0, "K must be divisible by PACK_FACTOR");
    TORCH_CHECK(K % group_size == 0, "K must be divisible by group_size");

    // Ensure that input tensors are on CUDA
    TORCH_CHECK(_A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(_qB.is_cuda(), "Input tensor qB must be a CUDA tensor");
    TORCH_CHECK(_scaling_factors.is_cuda(), "Input tensor scaling_factors must be a CUDA tensor");
    TORCH_CHECK(_zeros.is_cuda(), "Input tensor zeros must be a CUDA tensor");

    // Cast input tensors to appropriate data types
    auto A_packed = reinterpret_cast<const float4*>(_A.data_ptr<at::Half>());
    auto qB_packed = reinterpret_cast<const uint32_t*>(_qB.data_ptr<int32_t>());
    auto zeros_ptr = reinterpret_cast<half*>(_zeros.data_ptr<at::Half>());
    auto scaling_factors_ptr = reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());

    // Create an output tensor
    auto options = torch::TensorOptions().dtype(_A.dtype()).device(_A.device());
    at::Tensor _C = torch::empty({B, M, N}, options);
    auto C_ptr = reinterpret_cast<half*>(_C.data_ptr<at::Half>());

    // Calculate grid and block dimensions for kernel launch
    dim3 blockDim(BN * BM); // Total threads per block (BM x BN)
    dim3 gridDim((N + BN - 1) / BN, // Number of blocks along the N dimension
                 (M + BM - 1) / BM, // Number of blocks along the M dimension
                 B);                 // Number of blocks along the batch dimension

    // Ensure that blockDim.x does not exceed the maximum threads per block
    TORCH_CHECK(blockDim.x <= 1024, "blockDim.x exceeds the maximum number of threads per block");

    // Launch the CUDA kernel
    batched_gemm_kernel_quantized<<<gridDim, blockDim>>>(
        A_packed, qB_packed, zeros_ptr, scaling_factors_ptr, C_ptr, B, M, N, K, group_size
    );

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    // Return the output tensor
    return _C;
}

