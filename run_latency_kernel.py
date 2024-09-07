import argparse

from kernel.abx_rope import run_benchmark, run_test

def parse_args():
    parser = argparse.ArgumentParser(description="Argument Parser")
    parser.add_argument("--total_rank", type=int, default=1024, help="Total rank")
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
    args = parse_args()
    main(args)
    
