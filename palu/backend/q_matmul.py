import torch
import palu_kernel

def cuda_bmm_fA_qB_inner(group_size: int,
						fA: torch.FloatTensor,
						qB: torch.IntTensor,
						scales: torch.FloatTensor,
						zeros: torch.FloatTensor,
						bits: int) -> torch.FloatTensor:
	"""
    fA is of shape (B, nh, M, K) float16
    qB is of shape (B, nh, K // feat_per_int, N) int32
    scales is of shape (B, nh, G, N) float16
    zeros is of shape (B, nh, G, N) float16
    
    groupsize is the number of inner dimension in each groups.
    G = K // groupsize
    
    Returns C of shape (B, nh, M, N) float16
    """
	assert len(fA.shape) == 4 and len(qB.shape) == 4
	B, nh, M, K = fA.shape
	#assert M == 1, "Currently only supoort M=1"
	feat_per_int = 32 // bits
	# flatten to a 3D tensor
	#print(fA.view(-1, M, K).is_contiguous())
	fA = fA.view(-1, M, K).contiguous() 
	#print(qB.view(-1, K // feat_per_int, qB.shape[-1]).transpose(1, 2).is_contiguous())
	qB = qB.view(-1, K // feat_per_int, qB.shape[-1]).transpose(1, 2).contiguous()
	flatten_B = B * nh
	scales = scales.view(flatten_B, scales.shape[-2], scales.shape[-1]).transpose(1, 2).contiguous()
	zeros = zeros.view(flatten_B, zeros.shape[-2], zeros.shape[-1]).transpose(1, 2).contiguous()
	#print(scales.shape, zeros.shape)
	assert bits in [4]
	c = palu_kernel.batched_gemm_forward_cuda(fA, qB, scales, zeros, bits, group_size)
	c = c.view(B, nh, c.shape[-2], c.shape[-1])
	return c

def cuda_bmm_fA_qB_outer(group_size: int, 
				fA: torch.FloatTensor, 
				qB: torch.IntTensor, 
				scales: torch.FloatTensor, 
				zeros: torch.FloatTensor,
				bits: int,
				mqa: bool=False) -> torch.FloatTensor:
	"""
	Compute the matrix multiplication C = query x key.
	Where key is quantized into 2-bit values.

	fA is of shape (B, nh, M, K) float16
	qB is of shape (B, nh, K, N // feat_per_int) int32
	scales is of shape (B, nh, K, G) float16
	zeros is of shape (B, nh, K, G) float16

	groupsize is the number of outer dimensions in each group.
	G = N // groupsize

	Returns C of shape (B, nh, M, N) float16
	"""    
	assert len(fA.shape) == 4 and len(qB.shape) == 4
	B, nh, M, K = fA.shape 
	feat_per_int = 32 // bits
	# flatten to a 3D tensor
	fA = fA.view(-1, M, K).contiguous()
	N = qB.shape[-1] * feat_per_int
	qB = qB.reshape(-1, K, qB.shape[-1]).transpose(1, 2).contiguous()
	# This is based on the possible BLOCK_SIZE_Ks
	# assert K % 16 == 0 and K % 32 == 0 and K % 64 == 0 and K % 128 == 0, "K must be a multiple of 16, 32, 64, and 128"
	# This is based on the possible BLOCK_SIZE_Ns
	# assert N % 16 == 0 and N % 32 == 0 and N % 64 == 0, "N must be a multiple of 16, 32, 64, 128, and 256"
	# This is based on the possible BLOCK_SIZE_Ks
	# assert group_size % 64 == 0, "groupsize must be a multiple of 64, and 128"
	flatten_B = B * nh
	# if mqa:
	# 	flatten_B = B
	scales = scales.view(flatten_B, scales.shape[-2], scales.shape[-1]).transpose(1, 2).contiguous()
	zeros = zeros.view(flatten_B, zeros.shape[-2], zeros.shape[-1]).transpose(1, 2).contiguous()
	assert bits in [4]
	#c = palu_kernel.gemv_forward_cuda_outer_dim(fA, qB, scales, zeros, bits, group_size, nh, mqa)
	c = palu_kernel.batched_gemm_forward_outer_cuda(fA, qB, scales, zeros, bits, group_size)
	c = c.view(B, nh, c.shape[-2], c.shape[-1])
	return c