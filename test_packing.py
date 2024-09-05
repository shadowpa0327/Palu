import torch
import triton
import triton.language as tl

from kernel.pytorch_reference import LlamaRotaryEmbedding, apply_rotary_pos_emb_pytorch

import random
import numpy as np


def quant_and_pack_vcache(v: torch.FloatTensor, group_size: int, bits: int):
    shape = v.shape
    assert len(shape) == 4
    assert v.shape[-1] % group_size == 0
    num_groups = shape[-1] // group_size
    new_shape = (shape[:-1] + (num_groups, group_size))
    # Quantize
    max_int = 2 ** bits - 1
    data = v.view(new_shape)
    mn = torch.min(data, dim=-1, keepdim=True)[0]
    mx = torch.max(data, dim=-1, keepdim=True)[0]
    scale = (mx - mn) / max_int
    data = data - mn
    data.div_(scale)
    data = data.clamp_(0, max_int).round_().to(torch.int32)
    data = data.view(shape)
    #print(data)
    # Pack
    code = pack_tensor(data, bits, pack_dim=3)
    #print(code)
    return code, scale, mn


def unpack_and_dequant_vcache(v_code: torch.FloatTensor, 
							  scale: torch.FloatTensor, 
							  mn: torch.FloatTensor,
							  group_size: int, 
							  bits: int,
							  ):
	assert bits in [2, 4, 8]
	assert len(v_code.shape) == 4
	data = unpack_tensor(v_code, bits, pack_dim=3)
	#print(data.shape)
	shape = data.shape
	#num_groups = shape[-1] // group_size
	#data = data.view(shape[:-1] + (num_groups, group_size,))
	#print(data.shape)
	data = data.to(torch.float16)
	data = data * scale + mn 
	#print(data.shape)
	return data.view(shape)


def pack_tensor(data, bits, pack_dim):
    # Pack
    shape = data.shape
    feat_per_int = 32 // bits
    assert bits in [2,4,8], "Only 2, 4, 8 bits are supported"
    assert shape[pack_dim] % feat_per_int == 0, "Dimension length must be divisible by number of features per int"
    # BS, nh, T, nd // 16 # 16 is for 2bit
    code = torch.zeros(shape[:pack_dim] + (shape[pack_dim] // feat_per_int,)+shape[pack_dim+1:], 
                    dtype=torch.int32, 
                    device=data.device)
    i = 0
    row = 0
    unpacked_indices = [slice(None)] * len(data.shape)
    packed_indices = [slice(None)] * len(data.shape)
    while row < code.shape[pack_dim]:
        packed_indices[pack_dim] = row
        for j in range(i, i + (32 // bits)):
            unpacked_indices[pack_dim] = j
            code[packed_indices] |= data[unpacked_indices] << (bits * (j - i))
        i += 32 // bits
        row += 1
    return code


def unpack_tensor(v_code: torch.FloatTensor, 
				  bits: int, 
				  pack_dim: int):
	assert bits in [2,4,8]
	shape = v_code.shape
	feat_per_int = 32 // bits
	new_shape = shape[:pack_dim] + (shape[pack_dim] * feat_per_int,) + shape[pack_dim+1:]
	unpacked_v_code = torch.zeros(new_shape, dtype=torch.int8, device=v_code.device)
	i = torch.arange(new_shape[pack_dim], device=v_code.device) // feat_per_int
	j = torch.arange(new_shape[pack_dim], device=v_code.device) % feat_per_int
	num = 0xFF >> (8 - bits)
	packed_indices = [slice(None)] * len(new_shape)
	packed_indices[pack_dim] = i
	if pack_dim == 2:
		unpacked_v_code = ((v_code[packed_indices] >> (j * bits)[None, None, :, None]).to(torch.int16)) & num
	elif pack_dim == 3:
		unpacked_v_code = ((v_code[packed_indices] >> (j * bits)).to(torch.int16)) & num
	else:
		raise NotImplementedError
	return unpacked_v_code


@triton.jit
def _pack_along_last_dim(
	bits: tl.constexpr,
	intensor_ptr,
	code_ptr,
	N,
	num_feats: tl.constexpr,
	feat_per_int: tl.constexpr,
	BLOCK_SIZE_N: tl.constexpr
):
	num_int_per_y_dim = num_feats // feat_per_int
	bid = tl.program_id(axis=0)
	yid = tl.program_id(axis=1)
	offs_N = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	block_start = intensor_ptr + offs_N * num_feats + yid * feat_per_int # offset of the first element at current tile
	packed = tl.zeros((BLOCK_SIZE_N,), dtype=tl.int32)
	for i in range(feat_per_int):
		ptr = block_start + i
		element = tl.load(ptr, mask=offs_N<N, other=0.)
		element = element << (i * bits)
		# Combine the value using bitwise OR
		packed = packed | element
	tl.store(code_ptr + offs_N * num_int_per_y_dim + yid, packed, mask=offs_N < N)



@triton.jit
def _minmax_along_last_dim(
	x_ptr,
	mn_ptr, mx_ptr,
	total_elements: tl.constexpr, 
	N: tl.constexpr,
	num_groups: tl.constexpr, 
	group_size: tl.constexpr,
	BLOCK_SIZE_N: tl.constexpr
):
	bid = tl.program_id(axis=0)
	offsets_b = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	offsets = offsets_b[:, None] * group_size + tl.arange(0, group_size)[None, :]
	mask = offsets < total_elements
	x = tl.load(x_ptr + offsets, mask=mask)
	mx_val = tl.max(x, axis=1)
	mn_val = tl.min(x, axis=1)
	# tl.device_print('shape', mn_val[:, None].shape)
	tl.store(mn_ptr+offsets_b, mn_val, mask=offsets_b<N*num_groups)
	tl.store(mx_ptr+offsets_b, mx_val, mask=offsets_b<N*num_groups)



	
def triton_quantize_and_pack_along_last_dim(data: torch.Tensor, group_size: int, bit: int):
	assert len(data.shape) == 4
	shape = data.shape
	B, nh, D, T = shape
	# ================== Get Scale & Zeros ===============
	assert T % group_size == 0
	num_groups = T // group_size
	new_shape = (B * nh * D, num_groups, group_size)
	scale_mn_shape = B, nh, D, num_groups
	# Quantize
	data = data.reshape(new_shape)
	mx = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
	mn = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
	BLOCK_SIZE_N = 128
	grid = lambda meta: (triton.cdiv(data.shape[0]*data.shape[1], BLOCK_SIZE_N),)
	_minmax_along_last_dim[grid](data, mn, mx,
							 data.numel(), data.shape[0], num_groups, group_size,
							 BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=8) 
	# mn = torch.min(data, dim=-1, keepdim=True)[0].squeeze(-1)
	# mx = torch.max(data, dim=-1, keepdim=True)[0].squeeze(-1)
	scale = (mx - mn) / (2 ** bit - 1)
	data = data - mn.unsqueeze(-1)
	data.div_(scale.unsqueeze(-1))
	data = data.clamp_(0, 2 ** bit - 1).round_().to(torch.int32)
	data = data.view(-1, T)
	feat_per_int = 32 // bit
	packshape = (np.prod(shape[:-1]), shape[-1] // feat_per_int,)
	code = torch.zeros(*packshape, device=data.device, dtype=torch.int32)
	grid = lambda meta: (triton.cdiv(data.shape[0], BLOCK_SIZE_N), data.shape[1] // feat_per_int,)
	_pack_along_last_dim[grid](bit, data, code, data.shape[0], 
								data.shape[1], feat_per_int, 
								BLOCK_SIZE_N=BLOCK_SIZE_N, 
								num_warps=8)
	return code.view(B, nh, D, -1), scale.reshape(scale_mn_shape), mn.reshape(scale_mn_shape)

@triton.jit
def get_freq_multi_tokens(starting_idx, theta: tl.constexpr, NB_TOKENS: tl.constexpr):
    DIM: tl.constexpr = 128  # in model, dim = self.params.dim // self.params.n_heads
    DIM_2: tl.constexpr = 64
    freqs = tl.arange(0, DIM_2) * 2
    freqs = freqs.to(tl.float32) / DIM
    freqs = tl.extra.cuda.libdevice.fast_powf(theta, freqs)
    freqs = (tl.arange(0, NB_TOKENS) + starting_idx)[:, None] / freqs[None, :]
    return tl.extra.cuda.libdevice.fast_cosf(freqs), tl.extra.cuda.libdevice.fast_sinf(freqs)



def get_configs():
    configs = []
    for block_l in [16, 32, 64, 128]:
        for block_r in [16, 32]:
            for num_warps in [1, 4, 8, 16]:
                for num_stages in [1, 2, 3]:
                    configs.append(
                        triton.Config({'BLOCK_SIZE_L': block_l, 'BLOCK_SIZE_R': block_r},
                                num_stages=num_stages, num_warps=num_warps))
    # return configs
    # return [triton.Config({'BLOCK_SIZE_L': 128, 'BLOCK_SIZE_R': 32}, num_warps=4, num_stages=3)] # for gs=4
    # return [triton.Config({'BLOCK_SIZE_L': 64, 'BLOCK_SIZE_R': 32}, num_warps=4, num_stages=3)] # for gs=2
    return [triton.Config({'BLOCK_SIZE_L': 64, 'BLOCK_SIZE_R': 32}, num_warps=4, num_stages=1)] # for gs=1

# @triton.autotune(
#     configs= get_configs(),
#     key=["seq_len"]
# )
@triton.jit
def _ab_qx_fwd(
    bits, group_size,
    # ptrs
    # a_ptr: (bs, num_heads, seq_len(assume 1), head_dim)
    # b_ptr: (num_goups, rank, head_group_size*head_dim)
    # x_ptr: (bs, num_groups, seq_len, rank)
    a_ptr, b_ptr, x_ptr, 
    scales_ptr, zeros_ptr, out_ptr,
    # strides
    stride_az, stride_aa ,stride_ad,
    stride_bz, stride_br, stride_bd,
    stride_xhg, stride_xl, stride_xr,
    stride_scales_hg, stride_scales_xl, stide_scales_g,
    stride_zeros_hg, stride_zeros_xl, stide_zeros_g,
    #NOTE(brian1009): Debug, check dequant first
    #stride_ohg, stride_ol, stride_or,
    stide_oz, stride_oa, stride_ol,
    R, D, seq_len,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    THETA: tl.constexpr,
):
    pid_h = tl.program_id(axis=0) # parallel alone heads
    pid_l = tl.program_id(axis=1) # parallel alone blocks of sequence dim
    
    HEAD_GROUPS_ID = pid_h // (32 // NUM_GROUPS)
    
    feat_per_int = 32 // bits
    offs_ds = tl.arange(0, BLOCK_SIZE_D)
    offs_rs = tl.arange(0, BLOCK_SIZE_R)
    offs_ls = (pid_l * BLOCK_SIZE_L) + tl.arange(0, BLOCK_SIZE_L)
    
    
    
    # Load A and B
    A_ptrs = a_ptr + pid_h * stride_az + (0*stride_aa + offs_ds[None, :]*stride_ad) # assume a is always (bs, nh, 1, d)
    B_ptrs = b_ptr + pid_h * stride_bz + (offs_rs[:, None]*stride_br + offs_ds[None, :]*stride_bd) 
    X_ptrs = x_ptr + HEAD_GROUPS_ID * stride_xhg + (offs_ls[:, None]*stride_xl + (offs_rs[None, :] // feat_per_int) * stride_xr) # (BLOCK_SIZE_L, BLOCK_SIZE_R)
    scales_ptr = scales_ptr + HEAD_GROUPS_ID * stride_scales_hg + (offs_ls[:, None]*stride_scales_xl) # (BLOCK_SIZE_L, 1)
    zeros_ptr = zeros_ptr + HEAD_GROUPS_ID * stride_zeros_hg + (offs_ls[:, None]*stride_zeros_xl)
    # Set output ptr
    O_ptrs = out_ptr + pid_h * stide_oz + (0*stride_oa + offs_ls[None, :] * stride_ol) # follow the shape assumption of a, we set 0*stride_oa
    # NOTE(brian1009) debug
    #O_ptrs = out_ptr + pid_h * stride_ohg + (offs_ls[:, None] * stride_ol + offs_rs[None, :] * stride_or) 
    
       
    # parameters for dequantization
    # NOTE(brian1009): Since we do not have group-wise quant yet,
    # Hence, we dont need update the scales and zeros for each rank-dimensions
    # as it is the same.    
    shifter = (offs_rs % feat_per_int) * bits
    num = 0xFF >> (8-bits)
    scales = tl.load(scales_ptr)
    zeros = tl.load(zeros_ptr)
    #zeros = (zeros * 1.0).to(tl.float16)

    xb_0 = tl.zeros((BLOCK_SIZE_L, BLOCK_SIZE_D), dtype=tl.float32)
    xb_1 = tl.zeros((BLOCK_SIZE_L, BLOCK_SIZE_D), dtype=tl.float32)
    
    for _ in range(0, tl.cdiv(R, BLOCK_SIZE_R)):
        # Load next block of B, X
        x = tl.load(X_ptrs)
        tl.static_print('x', x)
        tl.static_print('shifter', shifter)
        x = (x >> shifter[None, :] & num)
        #FIXME: Multiply this 1.0 can make execution normal Triton==3.0.0
        x = x * scales + zeros * 1.0
        x = x.to(tl.float16)   
        b_0 = tl.load(B_ptrs)
        b_1 = tl.load(B_ptrs + BLOCK_SIZE_D * stride_bd)
        # Accumulate along the rank dimension
        xb_0 += tl.dot(x, b_0)
        xb_1 += tl.dot(x, b_1)
        # Advance the pointer to the next blocks
        B_ptrs += BLOCK_SIZE_R * stride_br
        X_ptrs += (tl.cdiv(BLOCK_SIZE_R, feat_per_int)) * stride_xr
    
    xb_0 = xb_0.to(tl.float16)
    xb_1 = xb_1.to(tl.float16)
    
    # RoPE (Temp)
    start_block = pid_l * BLOCK_SIZE_L
    cos, sin = get_freq_multi_tokens(starting_idx=start_block, theta=THETA, NB_TOKENS=BLOCK_SIZE_L)
    cos = cos.to(tl.float16)
    sin = sin.to(tl.float16)

    xb_rope_0 = xb_0 * cos - xb_1 * sin
    xb_rope_1 = xb_1 * cos + xb_0 * sin
    xb_0 = xb_rope_0.to(tl.float16)
    xb_1 = xb_rope_1.to(tl.float16)
    
    
    # # GEMV
    a_0 = tl.load(A_ptrs)
    a_1 = tl.load(A_ptrs + BLOCK_SIZE_D * stride_ad)
    abx_0 = tl.sum(a_0 * xb_0, 1)
    abx_1 = tl.sum(a_1 * xb_1, 1)
    abx = abx_0 + abx_1
    tl.store(O_ptrs, abx[None, :])
    
    
    
    
    

def triton_ab_qx_rope(
    a: torch.Tensor, b: torch.Tensor, x_q: torch.Tensor, 
    x_scales: torch.Tensor, x_zeros: torch.Tensor, 
    x_bits: int, x_quant_group_size: int):
    
    
    
    assert x_bits in [4, 8]
    assert a.dim() == 3
    assert b.dim() == 3
    assert x_q.dim() == 3
    
    feat_per_int = 32 // x_bits
    num_heads, _, head_dim = a.shape
    num_heads, rank_per_head_groups, head_dim = b.shape
    num_groups, seq_len, packed_rank_per_head_groups = x_q.shape
    rank_per_head_groups = packed_rank_per_head_groups * feat_per_int
    
    out = torch.empty((num_heads, 1, seq_len), dtype=a.dtype, device=a.device)
    BLOCK_SIZE_D = 64
    NUM_GROUPS = num_groups
    grid = lambda META: (
       num_heads ,triton.cdiv(seq_len, META['BLOCK_SIZE_L']),
    )
    _ab_qx_fwd[grid](
        x_bits, x_quant_group_size,
        a, b, x_q, x_scales, x_zeros, out,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        x_q.stride(0), x_q.stride(1), x_q.stride(2),
        x_scales.stride(0), x_scales.stride(1), x_scales.stride(2),
        x_zeros.stride(0), x_zeros.stride(1), x_zeros.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        rank_per_head_groups, num_heads, seq_len,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_L = 32,
        BLOCK_SIZE_R = 64,
        # num_stages=num_stages,
        # num_warps=num_warps,
        NUM_GROUPS=NUM_GROUPS,
        THETA=10000.
    )
    return out
                     

def torch_ab_qx_rope2(a, b, x_q, x_scales, x_zeros, x_bits, x_quant_group_size):
    x_f = unpack_and_dequant_vcache(x_q, x_scales, x_zeros, x_quant_group_size, x_bits) #(num_groups, seq_len, rank_per_head_groups)
    #return x_f
    out_puts = torch.empty((a.shape[0], a.shape[1], x_f.shape[-2]), dtype=torch.float16, device=a.device)
    cos, sin = LlamaRotaryEmbedding(dim=128, end=x_q.shape[-2])
    for i in range(b.shape[0]):
        b_i = b[i]
        a_i = a[i]
        rank_groups_id = i // (32 // x_q.shape[1])
        x_f_i = x_f[:, rank_groups_id].squeeze(0)
        xb_i = x_f_i @ b_i
        xb_i_rope = apply_rotary_pos_emb_pytorch(x = xb_i, cos=cos, sin=sin)
        xb_i_rope = xb_i_rope.squeeze(0).to(torch.float16)
        axb_i = a_i @ xb_i_rope.transpose(1, 0)
        out_puts[i] = axb_i
    return out_puts
    
def test_correctness(args):
    num_heads = args.num_heads
    head_dim = args.head_dim
    total_rank = args.total_rank
    seq_len = 512
    num_groups = args.num_groups
    rank_per_groups = total_rank // num_groups
    dtype = torch.float16
    device = torch.device('cuda')
    bits = args.x_bits
    
    A = torch.randn(num_heads, 1, head_dim, dtype=dtype, device=device)
    B = torch.randn(num_heads, rank_per_groups, head_dim, dtype=dtype, device=device)
    X = torch.randn(num_groups, seq_len, rank_per_groups, dtype=dtype, device=device)
    X_q, X_scales, X_zeros = triton_quantize_and_pack_along_last_dim(X.unsqueeze(0), group_size=rank_per_groups, bit=bits)  
    out_torch2 = torch_ab_qx_rope2(A, B, X_q, X_scales, X_zeros, bits, rank_per_groups)
    out_triton = triton_ab_qx_rope(A, B, X_q.squeeze(0), X_scales.squeeze(0), X_zeros.squeeze(0), bits, rank_per_groups)
    #print(out_triton)
    print("Correctness: ", torch.allclose(out_torch2, out_triton, atol=1, rtol=1e-4))
    
def main(args):
    args.num_groups = args.num_heads // args.group_size
    args.group_rank = args.total_rank // args.num_groups
    test_correctness(args)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Argument Parser")
    parser.add_argument("--total_rank", type=int, default=1024, help="Total rank")
    parser.add_argument("--num_heads", type=int, default=32, help="Number of heads, default to 32 (llama)")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension, default to 128 (llama)")
    parser.add_argument("--group_size", type=int, default=4, help="Number of heads per group")
    parser.add_argument("--target_seq_lens", nargs="+", type=int, 
                        default=[4096, 16384, 65536, 262144], help="Target sequence lengths")
    parser.add_argument("--x_bits", type=int, default=4, help="Number of bits for quantization")
    args = parser.parse_args()
    main(args)