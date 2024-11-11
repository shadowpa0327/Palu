#include <cuda_fp16.h>
#include <stdio.h>
#include <torch/extension.h>
#include "palu_gemm_cuda.h"

#define BM 4              // Tile size in the M dimension (number of rows per tile)
#define BK 128            // Tile size in the K dimension (group size for quantization)
#define BN 8             // Tile size in the N dimension (number of columns per tile)
#define PACK_FACTOR 8     // Number of elements packed together (for vectorized loads)

// Reduce sum within the warp using the tree reduction algorithm.
__device__ __forceinline__ float warp_reduce_sum(float sum) {
  #pragma unroll
  for(int i = 4; i >= 0; i--){
    sum += __shfl_down_sync(0xffffffff, sum, 1<<i);
  }
  /*
  // Equivalent to the following tree reduction implementation:
  sum += __shfl_down_sync(0xffffffff, sum, 16);
  sum += __shfl_down_sync(0xffffffff, sum, 8);
  sum += __shfl_down_sync(0xffffffff, sum, 4);
  sum += __shfl_down_sync(0xffffffff, sum, 2);
  sum += __shfl_down_sync(0xffffffff, sum, 1);
  */
  return sum;
}


// Kernel function to perform batched GEMM with quantized weights
__global__ void batched_gemm_kernel_quantized_outer(
    const half* __restrict__ A,        // Packed input activations [B, M, K]
    const uint32_t* __restrict__ qB,     // Packed quantized weights [B, N / PACK_FACTOR, K]
    const half* __restrict__ scaling_factors,   // Scaling factors for quantization [B, N / group_size, K ]
    const half* __restrict__ zeros,             // Zero offsets for quantization [B, N / group_size, K ]
    half* __restrict__ C,                       // Output matrix [B, M, N]
    const int B, const int M, const int N, const int K,
    const int group_size) {
    
    // Batch Index
    const int batch_idx = blockIdx.z;
    //print()
    // Tile indices
    const int tile_idx_N = blockIdx.x; // Tile index along the N dimension
    const int tile_idx_M = blockIdx.y; // Tile index along the M dimension

    // Thread indices
    const int thread_id = threadIdx.x; // Thread index within the block
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    float psum[PACK_FACTOR]{};

    half* output_ptr = C + batch_idx * (M * N) + (tile_idx_M * BM + warp_id) * N + tile_idx_N * BN;

    // Loop over K dimension
    for (int k=0; k<(K+BK-1)/BK; k++){
        // Each thread load 4 uint32_t from qB == 128 bits == 32 int4
        uint32_t qB_reg[4]{}; // along K-dim
        half cscale[4]{}; // along K-dim
        half czero[4]{}; // along K-dim
        half a_reg[4]{}; // along K-dim
        
        int A_offset = batch_idx * M * K + 
                       tile_idx_M * BM * K + 
                       warp_id * K +  // each combume 1 row in A of BK elements
                       k * BK + lane_id * 4; 
                       // NOTE for myself, why lane_id * 4? 
                       // Ans: because we are loading 4 elements at a time
 
        int B_offset = batch_idx * (N / PACK_FACTOR) * K + 
                       tile_idx_N * (BN / PACK_FACTOR) * K + 
                       k * BK + lane_id * 4;

        int scale_offset = batch_idx * (N / group_size) * K + 
                           (tile_idx_N * BN) / group_size * K + 
                           k * BK + lane_id * 4;

        // FIXME: What if K is not multiple of 4?
        *((float4*)(qB_reg)) = *((float4*)(qB + B_offset));
        *((float2*)(cscale)) = *((float2*)(scaling_factors + scale_offset));
        *((float2*)(czero)) = *((float2*)(zeros + scale_offset));
        *((float2*)(a_reg)) = *((float2*)(A + A_offset));        

        #pragma unroll
        for (int i=0; i<4; i++){
            uint32_t cur_qB = qB_reg[i];
            float cur_scale = __half2float(cscale[i]);
            float cur_zero = __half2float(czero[i]);
            half cur_a = a_reg[i];
            // unpacked
            for (int j=0; j<PACK_FACTOR; j++){
                float cur_single_b = (float)(cur_qB & 0xF);
                float dequant_b = cur_single_b * cur_scale + cur_zero;
                cur_qB = cur_qB >> 4;
                psum[j] += __half2float(cur_a) * dequant_b;
            }
        }
    }
    // Write back the results
    for(int i=0; i<PACK_FACTOR; i++){
        psum[i] = warp_reduce_sum(psum[i]);
        if (lane_id == 0){
            *(output_ptr + i) = __float2half(psum[i]);
        }
    }
}



/// PyTorch binding function
torch::Tensor batched_gemm_forward_outer_cuda(
    at::Tensor fA,              // [B, M, K], fp16
    at::Tensor qB,              // [B, N_packed, K], int32
    at::Tensor scaling_factors, // [B, N / group_size, K / group_size], fp16
    at::Tensor zeros,           // [B, N / group_size, K / group_size], fp16
    const int bit,
    const int group_size              // Group size used for quantization
) {
    // Ensure the tensors are contiguous
    fA = fA.contiguous();
    qB = qB.contiguous();
    scaling_factors = scaling_factors.contiguous();
    zeros = zeros.contiguous();

    // Check inputs
    TORCH_CHECK(fA.is_cuda(), "fA must be a CUDA tensor");
    TORCH_CHECK(qB.is_cuda(), "qB must be a CUDA tensor");
    TORCH_CHECK(scaling_factors.is_cuda(), "scaling_factors must be a CUDA tensor");
    TORCH_CHECK(zeros.is_cuda(), "zeros must be a CUDA tensor");
    TORCH_CHECK(fA.dtype() == torch::kHalf, "fA must be of type torch.float16");
    TORCH_CHECK(qB.dtype() == torch::kInt32, "qB must be of type torch.int32");
    TORCH_CHECK(scaling_factors.dtype() == torch::kHalf, "scaling_factors must be of type torch.float16");
    TORCH_CHECK(zeros.dtype() == torch::kHalf, "zeros must be of type torch.float16");
    TORCH_CHECK(bit==4, "Current we only support bit width of 4");
    // Get dimensions
    int B = fA.size(0);
    int M = fA.size(1);
    int K = fA.size(2);
    int N_packed = qB.size(1);
    int N = N_packed * 8;
    // Ensure M is 4
    TORCH_CHECK(M % 4, "M must be multiple of 4");
    TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128 (alignment)");

    // Get raw pointers to the data
    const __half* fA_ptr = reinterpret_cast<const __half*>(fA.data_ptr<at::Half>());
    const uint32_t* qB_ptr = reinterpret_cast<const uint32_t*>(qB.data_ptr<int32_t>());
    const __half* scaling_factors_ptr = reinterpret_cast<const __half*>(scaling_factors.data_ptr<at::Half>());
    const __half* zeros_ptr = reinterpret_cast<const __half*>(zeros.data_ptr<at::Half>());

    // Create output tensor
    auto options = torch::TensorOptions().dtype(torch::kHalf).device(fA.device());
    at::Tensor output = torch::empty({B, M, N}, options);
    __half* output_ptr = reinterpret_cast<__half*>(output.data_ptr<at::Half>());

    // Define block and grid dimensions
    dim3 gridDim((N + BN - 1) / BN, 
                 (M + BM - 1) / BM,
                B); // (Batches, N tiles)
    dim3 blockDim(32 * BM);              // M warps per block (M=4)

    // Launch the kernel
    batched_gemm_kernel_quantized_outer<<<gridDim, blockDim>>>(
        fA_ptr, qB_ptr, scaling_factors_ptr, zeros_ptr, output_ptr, B, M, N, K, group_size
    );

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    return output;
}