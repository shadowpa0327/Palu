import sys
import logging
from functools import partial

import torch
import argparse

import socket
from datetime import datetime

from transformers.models.llama.modeling_llama import (
    LlamaConfig, 
    LlamaAttention
)

from transformers.cache_utils import DynamicCache

from palu.model.svd_llama.palu_llama_attention import LlamaPaluAttention
from palu.quant.quant_kv_cache import (
    ValueQuantizedCacheV2,
)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"


def trace_handler(prof: torch.profiler.profile, file_postfix="prefilling", device="cuda:0"):
   # Prefix for file names.
   host_name = socket.gethostname()
   timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   file_prefix = f"{host_name}_{timestamp}"

   # Construct the trace file.
   prof.export_chrome_trace(f"{file_prefix}_{file_postfix}.json.gz")

   # Construct the memory timeline file.
   prof.export_memory_timeline(f"{file_prefix}_{file_postfix}.html", device=device)

def build_attention(args):
    device = "cuda:0"
    dtype = torch.float16

    logging.info(f"Creating Attention, dtype: {dtype}, device: {device}")
    config = LlamaConfig()
    config.max_position_embeddings = 300000
    attention = LlamaAttention(config, layer_idx=0).to(device, dtype)
    
    return attention, config

def build_attention_palu(args):
    device = "cuda:0"
    dtype = torch.float16

    logging.info(f"Creating Attention_Palu, dtype: {dtype}, device: {device}")
    config = LlamaConfig()
    config.max_position_embeddings = 300000
    config.total_rank_k = args.rank_k
    config.total_rank_v = args.rank_v
    config.k_bits = 16
    config.v_bits = args.v_bits
    num_groups = config.num_attention_heads // args.group_size
    config.num_groups = num_groups
    logging.info(f"rank_k: {args.rank_k}, rank_v: {args.rank_v}, group_size: {args.group_size}, num_groups: {num_groups}")
    attention = LlamaAttention(config, layer_idx=0)
    attention_palu = LlamaPaluAttention.from_attention(
        attention, 
        config,
        rank_k_list=[args.rank_k // num_groups for _ in range(num_groups)],
        rank_v_list=[args.rank_v // num_groups for _ in range(num_groups)],
    ).to(device, dtype)
    attention_palu.prepared_k_merged_U()
    return attention_palu, config

def build_prefilled_kv_cache(cache_size_k, cache_size_v, dtype, device, v_bits=16, optmitized_fp16_value=False):
    cache_k = torch.randn(cache_size_k, dtype=dtype, device=device)
    cache_v = torch.randn(cache_size_v, dtype=dtype, device=device)
    if v_bits == 16:
        past_key_value = DynamicCache()
    else:
        past_key_value = ValueQuantizedCacheV2(bits=v_bits, residual_length=128)
    logging.info(f"Creating Cache (type={type(past_key_value).__name__})")
    past_key_value.update(cache_k, cache_v, 0)
    return past_key_value


def profile_tpot(model, past_key_value, batch_size=1, prompt_len=1024, repeats=100,
                 torch_profile=False, outfile=""):
    logging.info(">>> Profiling TPOT (generation stage)")
    device = next(iter(model.parameters())).device
    
    position_ids = torch.arange(prompt_len, prompt_len+1)
    hidden_dim = model.config.hidden_size
    input_token = torch.randn((batch_size, 1, hidden_dim), dtype=torch.float16, device=device) # only input 1 token at a time

    # warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.no_grad():
        with torch.cuda.stream(s):
            for _ in range(25):
                _ = model(input_token, past_key_value=past_key_value, position_ids=position_ids)
    torch.cuda.current_stream().wait_stream(s)


    def generate(new_input_token, past_key_value, position_ids):
        out = model(new_input_token, past_key_value=past_key_value, position_ids=position_ids)
        return out

    new_input_token = torch.randn((batch_size, 1, hidden_dim), dtype=torch.float16, device=device) # only input 1 token at a time
    with torch.no_grad():
        start = torch.cuda.Event(enable_timing=True) 
        end   = torch.cuda.Event(enable_timing=True) 
        start.record()
        for _ in range(repeats):
            generate(new_input_token, past_key_value=past_key_value, position_ids=position_ids)
        end.record()
        torch.cuda.synchronize()
    dur = start.elapsed_time(end)
    logging.info(f"Finished, prompt_len: {prompt_len}, latency: {dur/repeats:.2f} milliseconds")

    if torch_profile:
        outfile_postfix = f"{outfile}"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=5, active=6, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=partial(
                trace_handler, file_postfix=outfile_postfix, device="cuda:0"
            )
        ) as prof:

            with torch.no_grad():
                # (wait=1, warmup=5, active=6) , repeat=1
                for _ in range(12):
                    generate(new_input_token, past_key_value, position_ids=position_ids)
                    prof.step()


def main(args):
    bs = 1

    if args.palu:
        attention, config = build_attention_palu(args)
        attention.eval()

        num_groups = config.num_groups
        # NOTE: Assuming uniform head_dim
        group_dim_k = config.total_rank_k // config.num_groups 
        group_dim_v = config.total_rank_v // config.num_groups 
        cache_size_k = (bs, num_groups, args.prompt_len, group_dim_k)
        cache_size_v = (bs, num_groups, args.prompt_len, group_dim_v)
        
        past_key_value = build_prefilled_kv_cache(
            cache_size_k, cache_size_v,
            torch.float16, "cuda:0", args.v_bits
        )
        
        profile_tpot(attention, past_key_value, bs, args.prompt_len, args.repeats, args.torch_profile, "tpot_palu_fp16")
    else:
        attention, config = build_attention(args)
        attention.eval()

        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // num_heads
        cache_size = (bs, num_heads, args.prompt_len, head_dim)
        past_key_value = build_prefilled_kv_cache(
            cache_size, cache_size,
            torch.float16, "cuda:0", 16,
            args.enable_opt_value_cache
        )
        profile_tpot(attention, past_key_value, bs, args.prompt_len, args.repeats, args.torch_profile, "tpot_fp16")
    

if __name__ =='__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--palu', action='store_true',
        help='Whether to use PALU attention.'
    )
    parser.add_argument(
        '--rank_k', type=int, default=1024,
        help='The rank of key matrix for PALU attention.'
    )
    parser.add_argument(
        '--rank_v', type=int, default=2048,
        help='The rank of value matrix for PALU attention.'
    )
    parser.add_argument(
        '--v_bits', type=int, default=16,
        help='The number of bits for low-rank latents of Value'
    )
    parser.add_argument(
        '--group_size', type=int, default=4,
        help='The group size for PALU attention.'
    )
    parser.add_argument(
        '--repeats', type=int, default=100,
        help='The number of profiling to repeat (default: 100)'
    )
    parser.add_argument(
        '--prompt_len', type=int, default=1024,
        help='The number of input tokens to model. (default: 1024)'
    )
    parser.add_argument(
        '--torch_profile', action='store_true',
        help='Whether to launch the pytorch profiler.'
    )
    parser.add_argument( #deprecated so far
        '--enable_opt_value_cache', action='store_true',
        help='Whether to use the optimized FP16 value cache.'
    )
    
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)3d] %(message)s",
        datefmt="%d/%b/%Y %H:%M:%S",
        stream=sys.stdout)
    main(args)
