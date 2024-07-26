import torch
import triton
import triton.language as tl
import argparse

from .pytorch_reference import LlamaRotaryEmbedding, apply_rotary_pos_emb_pytorch


def set_random_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

@triton.autotune(
    configs= get_configs(),
    key=["seq_len"]
)
@triton.jit
def _abx_fwd(
    a_ptr, b_ptr, x_ptr, out_ptr,
    stride_az, stride_aa, stride_ad,
    stride_bz, stride_br, stride_bd,
    stride_xhg, stride_xl, stride_xr,
    stride_oz, stride_oa, stride_ol,
    R, D, seq_len,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    THETA: tl.constexpr,
):
    pid_h = tl.program_id(axis=0)  # number of heads
    pid_l = tl.program_id(axis=1)  # nubmer of block along seq_length dimension
    
    # Assuming NUM_GROUPS = 4, then pid_h = 0, 1, 2, 3 will be assigned to head group 0
    HEAD_GROUPS_ID = pid_h // (32 // NUM_GROUPS) 
    offs_ds = tl.arange(0, BLOCK_SIZE_D) # same as offs_bds
    offs_rs  = tl.arange(0, BLOCK_SIZE_R)
    offs_ls = (pid_l * BLOCK_SIZE_L) + tl.arange(0, BLOCK_SIZE_L)
    
    A_ptrs = a_ptr + pid_h * stride_az + (0*stride_aa + offs_ds[None, :]*stride_ad) # assume a is always (bs, 1, d)
    B_ptrs = b_ptr + pid_h * stride_bz + (offs_rs[:, None]*stride_br + offs_ds[None, :]*stride_bd)
    X_ptrs = x_ptr + HEAD_GROUPS_ID * stride_xhg + (offs_ls[:, None]*stride_xl + offs_rs[None, :]*stride_xr)
    O_ptrs = out_ptr + pid_h * stride_oz + (0*stride_oa + offs_ls[None, :]*stride_ol)
    
    # Fix BLOCK_SIZE_D = 64, and head_dim = 128
    xb_0 = tl.zeros((BLOCK_SIZE_L, BLOCK_SIZE_D), dtype=tl.float32)
    xb_1 = tl.zeros((BLOCK_SIZE_L, BLOCK_SIZE_D), dtype=tl.float32)
    for _ in range(0, tl.cdiv(R, BLOCK_SIZE_R)):
        # Load next block of B, X
        x = tl.load(X_ptrs)
        b_0 = tl.load(B_ptrs)
        b_1 = tl.load(B_ptrs + BLOCK_SIZE_D * stride_bd)
        # Accumulate along R dimension.
        xb_0 = tl.dot(x, b_0, xb_0)
        xb_1 = tl.dot(x, b_1, xb_1)
        # Advance the pointers to next blocks
        B_ptrs += BLOCK_SIZE_R * stride_br
        X_ptrs += BLOCK_SIZE_R * stride_xr
    
    xb_0 = xb_0.to(tl.float16)
    xb_1 = xb_1.to(tl.float16)
    
    # RoPE
    start_block = pid_l * BLOCK_SIZE_L
    cos, sin = get_freq_multi_tokens(starting_idx=start_block, theta=THETA, NB_TOKENS=BLOCK_SIZE_L)
    cos = cos.to(tl.float16)
    sin = sin.to(tl.float16)

    xb_rope_0 = xb_0 * cos - xb_1 * sin
    xb_rope_1 = xb_1 * cos + xb_0 * sin
    xb_0 = xb_rope_0.to(tl.float16)
    xb_1 = xb_rope_1.to(tl.float16)

    # GEMV
    a_0 = tl.load(A_ptrs)
    a_1 = tl.load(A_ptrs + BLOCK_SIZE_D * stride_ad)
    abx_0 = tl.sum(a_0 * xb_0, 1)
    abx_1 = tl.sum(a_1 * xb_1, 1)
    abx = abx_0 + abx_1
    tl.store(O_ptrs, abx[None, :])

    
def abx(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # U x V x X
    assert a.dim() == 3
    assert b.dim() == 3
    assert x.dim() == 3

    num_heads, _, head_dim = a.shape
    num_heads,rank_per_head_groups, head_dim = b.shape
    num_groups, seq_len, rank_per_head_groups = x.shape
    # Allocate output tensor
    out = torch.empty((num_heads, 1, seq_len), dtype=x.dtype, device=x.device)
    BLOCK_SIZE_D = 64
    # BLOCK_SIZE_R = 32
    # BLOCK_SIZE_L = 128
    # num_stages = 1
    # num_warps = 8
    NUM_GROUPS = num_groups
    
    grid = lambda META: (32, triton.cdiv(seq_len, META["BLOCK_SIZE_L"]))
    _abx_fwd[grid](
        a, b, x, out,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        x.stride(0), x.stride(1), x.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        R = rank_per_head_groups,
        D = head_dim,
        seq_len = seq_len,
        BLOCK_SIZE_D = BLOCK_SIZE_D,
        # BLOCK_SIZE_L = BLOCK_SIZE_L,
        # BLOCK_SIZE_R = BLOCK_SIZE_R,
        # num_stages=num_stages,
        # num_warps=num_warps,
        NUM_GROUPS = NUM_GROUPS,
        THETA = 10000.,
    )
    return out

def torch_abx(a, b, x):
    # Input shape
    # a: (num_heads, 1, head_dim)
    # b: (num_heads, rank_per_groups, head_dim)
    # x: (num_groups, seq_len, rank_per_groups)
    
    # Recompute the key states
    # x: (num_groups, 1, seq_len, rank_per_groups)
    # b: (num_groups, group_size, rank_per_groups, head_dim)
    # xb: (num_heads, seq_len, head_dim)
    x_expand = x.unsqueeze(1)
    b_reshape = b.reshape(-1, b.shape[0] // x.shape[0], b.shape[-2], b.shape[-1])
    xb = x_expand @ b_reshape
    xb = xb.reshape(b.shape[0], -1, b.shape[-1])

    # Apply RoPE
    cos, sin = LlamaRotaryEmbedding(dim=128, end=x.shape[1])
    xb_rope = apply_rotary_pos_emb_pytorch(x=xb, cos=cos, sin=sin)
    axb = a @ xb_rope.transpose(-1, -2).to(torch.float16)
    return axb

def run_benchmark(args):
    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["seq_len"],
            x_vals=args.target_seq_lens,
            line_arg="provider",
            line_vals=["WX", "torch", "ours"],
            line_names=["WX", "Torch", "Ours"],
            styles=[("gray", "--"), ("green", "--"), ("blue", "-")],
            ylabel="us",
            plot_name=f"low-rank-rank-{args.total_rank}-group-{args.num_groups}",
            args={
                "dtype": torch.float16,
                "num_heads": args.num_heads,
                "head_dim": args.head_dim,
                "total_rank": args.total_rank,
                "num_groups": args.num_groups, # number of head groups
            },
        ))

    @triton.testing.perf_report(configs)
    def bench_low_rank(num_heads, head_dim, total_rank, seq_len, num_groups, provider, dtype=torch.float16, device="cuda"):
        rank_per_groups = total_rank // num_groups
    
        warmup = 25
        rep = 100
        A = torch.randn(num_heads, 1, head_dim, dtype=dtype, device=device)
        B = torch.randn(num_heads, rank_per_groups, head_dim, dtype=dtype, device=device)
        X = torch.randn(num_groups, seq_len, rank_per_groups, dtype=dtype, device=device)
        org_A = torch.randn(num_heads, 1, head_dim, dtype=dtype, device=device)
        org_X = torch.randn(num_heads, seq_len, head_dim, dtype=dtype, device=device)
        
        
        quantiles = [0.5, 0.2, 0.8]
        if provider == "torch":
            def fn(): return torch_abx(A, B, X)
            ms, min_ms, max_ms = triton.testing.do_bench(
                fn, quantiles=quantiles, warmup=warmup, rep=rep)
            
        if provider == "ours":
            def fn(): return abx(A, B, X)
            ms, min_ms, max_ms = triton.testing.do_bench(
                fn, quantiles=quantiles, warmup=warmup, rep=rep)

        if provider == "WX":
            def fn(): return torch.matmul(org_A, org_X.transpose(-1, -2))
            ms, min_ms, max_ms = triton.testing.do_bench(
                fn, quantiles=quantiles, warmup=warmup, rep=rep)
        
        return ms*1000, min_ms*1000, max_ms*1000

    import os
    # create a directory to store the results
    os.makedirs('results', exist_ok=True)
    bench_low_rank.run(print_data=True, show_plots=True, save_path='results/')

def run_test(args):
    num_heads = args.num_heads
    head_dim = args.head_dim
    total_rank = args.total_rank
    seq_len = 64
    num_groups = args.num_groups
    rank_per_groups = total_rank //  num_groups
    dtype = torch.float16
    device = "cuda"
    
    A = torch.randn(num_heads, 1, head_dim, dtype=dtype, device=device)
    B = torch.randn(num_heads, rank_per_groups, head_dim, dtype=dtype, device=device)
    X = torch.randn(num_groups, seq_len, rank_per_groups, dtype=dtype, device=device)

    x, xb, xb_rope, xb_rope_0, xb_rope_1, axb, cos, sin, freqs = torch_abx(A, B, X)
    ours = abx(A, B, X)

    print("Max diff: ", torch.max(torch.abs(axb - ours)))

def parse_args():
    parser = argparse.ArgumentParser(description="Argument Parser")
    parser.add_argument("--total_rank", type=int, default=2048, help="Total rank")
    parser.add_argument("--num_heads", type=int, default=32, help="Number of heads, default to 32 (llama)")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension, default to 128 (llama)")
    parser.add_argument("--group_size", type=int, default=4, help="Number of heads per group")
    parser.add_argument("--target_seq_lens", nargs="+", type=int, 
                        default=[4096, 16384, 65536, 262144], help="Target sequence lengths")
    parser.add_argument("--check", action="store_true", help="Check the correctness of the implementation")
    args = parser.parse_args()
    return args

def main(args):
    args.num_groups = args.num_heads // args.group_size
    args.group_rank = args.total_rank // args.num_groups
    print("Start benchmarking fused low-rank KV Cache Kernels...")
    print("Total Rank: ", args.total_rank)
    print("Number of Heads: ", args.num_heads)
    print("Head Dimension: ", args.head_dim)
    print("Group Size:", args.group_size)
    print("Number of Groups: ", args.num_groups)
    print("Rank per Group: ", args.group_rank)
    if args.check:
        run_test(args)
    else:
        run_benchmark(args)

if __name__ == "__main__":
    set_random_seed()
    args = parse_args()
    main(args)
    
