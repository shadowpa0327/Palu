import torch
import random
import triton
import triton.language as tl
from kernel.packing import *

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
    stride_ohg, stride_ol, stride_or,
    #stide_oz, stride_oa, stride_ol,
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
    
    offs_ds = tl.arange(0, BLOCK_SIZE_D)
    offs_rs = tl.arange(0, BLOCK_SIZE_R)
    offs_ls = (pid_l * BLOCK_SIZE_L) + tl.arange(0, BLOCK_SIZE_L)
    
    feat_per_int = 32 // bits
    
    # Load A and B
    A_ptrs = a_ptr + pid_h * stride_az + (0*stride_aa + offs_ds[None, :]*stride_ad) # assume a is always (bs, nh, 1, d)
    B_ptrs = b_ptr + pid_h * stride_bz + (offs_rs[:, None]*stride_br + offs_ds[None, :]*stride_bd)
    # Load X (quanitzed and packed in 32 bits integers)
    #X_ptrs = x_ptr + HEAD_GROUPS_ID * stride_xhg + (offs_ls[:, None]*stride_xl + (offs_rs[None, :] // feat_per_int) * stride_xr) # (BLOCK_SIZE_L, BLOCK_SIZE_R)
    # X_ptrs = x_ptr + pid_h * stride_xhg + (offs_ls[:, None]*stride_xl + (offs_rs[None, :] // feat_per_int) * stride_xr) # (BLOCK_SIZE_L, BLOCK_SIZE_R)
    # scales_ptr = scales_ptr + pid_h * stride_scales_hg + (offs_ls[:, None]*stride_scales_xl + (offs_rs[None, :] // group_size) ) # (BLOCK_SIZE_L, 1)
    # zeros_ptr = zeros_ptr + pid_h * stride_zeros_hg + (offs_ls[:, None]*stride_zeros_xl + (offs_rs[None, :] // group_size))
    
    X_ptrs = x_ptr + pid_h * stride_xhg + (offs_ls[:, None]*stride_xl + (offs_rs[None, :] // feat_per_int) * stride_xr) # (BLOCK_SIZE_L, BLOCK_SIZE_R)
    scales_ptr = scales_ptr + pid_h * stride_scales_hg + (offs_ls[:, None]*stride_scales_xl + (offs_rs[None, :] // group_size) ) # (BLOCK_SIZE_L, 1)
    zeros_ptr = zeros_ptr + pid_h * stride_zeros_hg + (offs_ls[:, None]*stride_zeros_xl + (offs_rs[None, :] // group_size))
    
    # Set output ptr
    #O_ptrs = out_ptr + pid_h * stide_oz + (0*stride_oa + offs_ls[None, :] * stride_ol) # follow the shape assumption of a, we set 0*stride_oa
    # NOTE(brian1009) debug
    O_ptrs = out_ptr + pid_h * stride_ohg + (offs_ls[:, None] * stride_ol + offs_rs[None, :] * stride_or) 
    
       
    # parameters for dequantization
    # NOTE(brian1009): Since we do not have group-wise quant yet,
    # Hence, we dont need update the scales and zeros for each rank-dimensions
    # as it is the same.    
    shifter = (offs_rs % feat_per_int) * bits
    num = 0xFF >> (8-bits)
    scales = tl.load(scales_ptr)
    zeros = tl.load(zeros_ptr)
    
    # x = tl.load(X_ptrs)
    # x = (x >> shifter[None, :] & num)
    # x = x*scales + zeros
    
    # tl.store(O_ptrs, x)

    # xb_0 = tl.zeros((BLOCK_SIZE_L, BLOCK_SIZE_D), dtype=tl.float32)
    # xb_1 = tl.zeros((BLOCK_SIZE_L, BLOCK_SIZE_D), dtype=tl.float32)
    
    for _ in range(0, tl.cdiv(R, BLOCK_SIZE_R)):
        # Load next block of B, X
        x = tl.load(X_ptrs)
        x = (x >> shifter[None, :] & num)
        x = x * scales + zeros
        tl.store(O_ptrs, x)
        O_ptrs += BLOCK_SIZE_R * stride_or
        X_ptrs += (BLOCK_SIZE_R // feat_per_int) * stride_xr
        
    #     b_0 = tl.load(B_ptrs)
    #     b_1 = tl.load(B_ptrs + BLOCK_SIZE_D * stride_bd)
    #     # Accumulate along the rank dimension
    #     xb_0 += tl.dot(x, b_0, xb_0)
    #     xb_1 += tl.dot(x, b_1, xb_1)
    #     # Advance the pointer to the next blocks
    #     B_ptrs += BLOCK_SIZE_R * stride_br
    #     X_ptrs += BLOCK_SIZE_R * stride_xr
    
    # xb_0 = xb_0.to(tl.float16)
    # xb_1 = xb_1.to(tl.float16)
    
    # RoPE (TBD)
    
    
    # # GEMV
    # a_0 = tl.load(A_ptrs)
    # a_1 = tl.load(A_ptrs + BLOCK_SIZE_D * stride_ad)
    # abx_0 = tl.sum(a_0 * xb_0, 1)
    # abx_1 = tl.sum(a_1 * xb_1, 1)
    # abx = abx_0 + abx_1
    # tl.store(O_ptrs, abx[None, :])
    
    
    
    
    

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
    
    # flatten to a 3D tensor
    #x_q = x_q.view(-1, L, R_q)
    #flatten_B = B*nh
    #x_dq = torch.empty((flatten_B, L, R_f), dtype=torch.float16, device=x_q.device)
    #out = torch.empty((num_heads, 1, seq_len), dtype=a.dtype, device=a.device)
    out = torch.zeros((num_groups, seq_len, rank_per_head_groups), dtype=torch.float16, device=a.device)
    BLOCK_SIZE_D = 64
    NUM_GROUPS = num_groups
    #print(NUM_GROUPS)
    #x_scales = x_scales.view(flatten_B, x_scales.shape[-2], x_scales.shape[-1])
    #x_zeros = x_zeros.view(flatten_B, x_zeros.shape[-2], x_zeros.shape[-1])

    
    # grid = lambda META: (
    #     num_heads, triton.cdiv(seq_len, META['BLOCK_SIZE_L'])
    # )
    grid = lambda META: (
       NUM_GROUPS ,triton.cdiv(seq_len, META['BLOCK_SIZE_L']),
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
                     

# def torch_ab_qx_rope(a, b, x_q, x_scales, x_zeros, x_bits, x_quant_group_size):
#     x_f = unpack_and_dequant_vcache(x_q, x_scales, x_zeros, x_quant_group_size, x_bits) #(num_groups, seq_len, rank_per_head_groups)
#     b_reshape = b.reshape(-1, 32 // x_f.shape[1], b.shape[-2], b.shape[-1]).transpose(1, 0)
#     xb = x_f @ b_reshape
#     xb = xb.transpose(1, 0)
#     xb = xb.reshape(b.shape[0], -1, b.shape[-1])
#     axb = a @ xb.transpose(-1, -2).to(torch.float16)
#     return axb


def torch_ab_qx_rope2(a, b, x_q, x_scales, x_zeros, x_bits, x_quant_group_size):
    x_f = unpack_and_dequant_vcache(x_q, x_scales, x_zeros, x_quant_group_size, x_bits) #(num_groups, seq_len, rank_per_head_groups)
    return x_f
    # out_puts = torch.empty((a.shape[0], a.shape[1], x_f.shape[-2]), dtype=torch.float16, device=a.device)
    # for i in range(b.shape[0]):
    #     b_i = b[i]
    #     a_i = a[i]
    #     rank_groups_id = i // (32 // x_q.shape[1])
    #     x_f_i = x_f[:, rank_groups_id].squeeze(0)
    #     xb_i = x_f_i @ b_i
    #     axb_i = a_i @ xb_i.transpose(1, 0)
    #     out_puts[i] = axb_i
    # return out_puts
    
   
    
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
    #out_torch = torch_ab_qx_rope(A, B, X_q, X_scales, X_zeros, bits, rank_per_groups)
    out_torch2 = torch_ab_qx_rope2(A, B, X_q, X_scales, X_zeros, bits, rank_per_groups)
    out_triton = triton_ab_qx_rope(A, B, X_q.squeeze(0), X_scales.squeeze(0), X_zeros.squeeze(0), bits, rank_per_groups)
    print("Correctness: ", torch.allclose(out_torch2, out_triton, atol=1e-2, rtol=1e-4))
    
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