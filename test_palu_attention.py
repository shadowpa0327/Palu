import math
import torch
import torch.nn as nn
import pytest

from transformers.models.llama.modeling_llama import LlamaAttention, LlamaConfig, DynamicCache

from kernel.palu_attention import HeadwiseLowRankModule, LlamaPaluAttention, LlamaPaluAttention_noRoPE

@pytest.fixture(autouse=True)
def set_random_seed():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@pytest.fixture()
def config():
    config = LlamaConfig()
    config.group_size = 4
    config.num_groups = config.num_attention_heads // 4
    config.total_rank_k = 4096
    config.total_rank_v = 4096
    config.k_bits = 16
    config.v_bits = 16
    return config


def _set_random_seed():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test_lr_layer_init():
    bsz = 1
    seq_len = 5
    ranks = [2, 2]
    in_features = 6
    out_features = 6
    bias = False

    module = HeadwiseLowRankModule(ranks, in_features, out_features, bias)
    
    hidden_states = torch.randn(bsz, seq_len, in_features)

    # forward
    forward_output = module(hidden_states)
    
    # project_to_latent & reconstruct
    latent_states = module.project_to_latent(hidden_states)
    reconstructed_output = module.reconstruct(latent_states)

    torch.testing.assert_close(forward_output, reconstructed_output)

def test_lr_layer_from_linear():
    bsz = 1
    seq_len = 5
    ranks = [3, 3]
    in_features = 10
    out_features = 6
    bias = False

    linear = nn.Linear(in_features, out_features, bias)
    svd_linear = HeadwiseLowRankModule.from_linear(linear, ranks)
    
    inputs = torch.randn(bsz, seq_len, in_features)

    # Golden linear
    linear_output = linear(inputs)
    
    # Low-Ranl linear
    svd_linear_output = svd_linear(inputs)

    torch.testing.assert_close(linear_output, svd_linear_output)

def test_palu_attention_inherit_no_fusion(config):
    bsz = 1
    seq_len = 64

    attention = LlamaAttention(config, 0)
    palu_attention = LlamaPaluAttention.from_attention(attention, config, no_fusion=True)

    # q, o proj
    torch.testing.assert_close(attention.q_proj.weight, palu_attention.q_proj.weight)
    torch.testing.assert_close(attention.o_proj.weight, palu_attention.o_proj.weight)

    # k, v proj
    inputs = torch.randn(bsz, seq_len, config.hidden_size)
    torch.testing.assert_close(attention.k_proj(inputs), palu_attention.k_proj(inputs))
    torch.testing.assert_close(attention.v_proj(inputs), palu_attention.v_proj(inputs))

def test_palu_attention_inherit_fusion(config):
    bsz = 1
    seq_len = 64

    attention = LlamaAttention(config, 0)
    palu_attention = LlamaPaluAttention.from_attention(attention, config)

    # q, v proj
    torch.testing.assert_close(attention.q_proj.weight, palu_attention.q_proj.weight)
    
    # k, v proj
    inputs = torch.randn(bsz, seq_len, config.hidden_size)
    torch.testing.assert_close(attention.k_proj(inputs), palu_attention.k_proj(inputs))
    torch.testing.assert_close(attention.v_proj(inputs), palu_attention.v_proj(inputs))

    # o proj
    q_len = seq_len
    group_size = config.group_size
    num_heads = config.num_attention_heads
    hidden_dim = config.hidden_size
    num_groups = num_heads // group_size
    head_dim = hidden_dim // num_heads
    group_rank = config.total_rank_v // num_groups

    # original
    inputs = torch.randn(1, q_len, hidden_dim)
    attn_weight = torch.randn(1, num_heads, q_len, q_len)
    v_states = attention.v_proj(inputs).view(1, q_len, num_heads, head_dim).transpose(1, 2)
    attn_output = torch.matmul(attn_weight, v_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(1, q_len, -1)
    ori_output = attention.o_proj(attn_output)

    # fusion
    attn_weight = attn_weight.reshape(1, num_groups, q_len * group_size, q_len)
    v_h_states = palu_attention.v_proj.project_to_latent(inputs).reshape(1, q_len, num_groups, group_rank).transpose(1, 2)
    
    attn_h_output = torch.matmul(attn_weight, v_h_states)
    attn_h_output = attn_h_output.reshape(1, num_heads, q_len, group_rank)
    
    final_fused_o_output = palu_attention.o_proj(attn_h_output.transpose(1, 2).reshape(1, q_len, -1))
    torch.testing.assert_close(ori_output, final_fused_o_output, rtol=1e-3, atol=1e-3)

def test_palu_attention_inherit_fusion_no_rope(config):
    bsz = 1
    seq_len = 64

    attention = LlamaAttention(config, 0)
    palu_attention = LlamaPaluAttention_noRoPE.from_attention(attention, config)

    # attn_weight
    q_len = seq_len
    group_size = config.group_size
    num_heads = config.num_attention_heads
    hidden_dim = config.hidden_size
    num_groups = num_heads // group_size
    head_dim = hidden_dim // num_heads
    group_rank_k = config.total_rank_k // num_groups

    # original
    hidden_states = torch.randn(bsz, q_len, hidden_dim)
    query_states = attention.q_proj(hidden_states)
    key_states = attention.k_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    
    ori_attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

    # fusion
    query_states = palu_attention.q_proj(hidden_states)
    key_h_states = palu_attention.k_proj.project_to_latent(hidden_states)
    query_states = query_states.view(bsz, q_len, num_heads, group_rank_k).transpose(1, 2)
    key_h_states = key_h_states.view(bsz, q_len, num_groups, group_rank_k).transpose(1, 2)
    
    query_states = query_states.reshape(bsz, num_groups, q_len * group_size, group_rank_k)
    
    fused_attn_weights = torch.matmul(query_states, key_h_states.transpose(2, 3)) / math.sqrt(head_dim)
    fused_attn_weights = fused_attn_weights.view(bsz, num_heads, q_len, seq_len)

    torch.testing.assert_close(ori_attn_weights, fused_attn_weights, rtol=1e-3, atol=1e-3)

    # o proj
    q_len = seq_len
    group_size = config.group_size
    num_heads = config.num_attention_heads
    hidden_dim = config.hidden_size
    num_groups = num_heads // group_size
    head_dim = hidden_dim // num_heads
    group_rank = config.total_rank_v // num_groups

    # original
    inputs = torch.randn(1, q_len, hidden_dim)
    attn_weight = torch.randn(1, num_heads, q_len, q_len)
    v_states = attention.v_proj(inputs).view(1, q_len, num_heads, head_dim).transpose(1, 2)
    attn_output = torch.matmul(attn_weight, v_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(1, q_len, -1)
    ori_output = attention.o_proj(attn_output)

    # fusion
    attn_weight = attn_weight.reshape(1, num_groups, q_len * group_size, q_len)
    v_h_states = palu_attention.v_proj.project_to_latent(inputs).reshape(1, q_len, num_groups, group_rank).transpose(1, 2)
    
    attn_h_output = torch.matmul(attn_weight, v_h_states)
    attn_h_output = attn_h_output.reshape(1, num_heads, q_len, group_rank)
    
    final_fused_o_output = palu_attention.o_proj(attn_h_output.transpose(1, 2).reshape(1, q_len, -1))
    torch.testing.assert_close(ori_output, final_fused_o_output, rtol=1e-3, atol=1e-3)


def test_palu_attention_fusion(config):
    bsz = 1
    seq_len = 64
    dev = 'cuda:0'
    dtype = torch.float16

    attention = LlamaAttention(config, 0)
    palu_attention = LlamaPaluAttention.from_attention(attention, config)

    attention = attention.to(dev, dtype)
    palu_attention = palu_attention.to(dev, dtype)

    inputs = torch.randn(bsz, seq_len, config.hidden_size).to(dev, dtype)

    # Golden
    golden_output, golden_attn_weights, _ = attention(inputs, output_attentions = True)
    
    # Fusion
    fusion_output, fusion_attn_weights, _ = palu_attention(inputs, output_attentions = True, golden_kernel=True)

    torch.testing.assert_close(golden_attn_weights, fusion_attn_weights, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(golden_output, fusion_output, rtol=1e-3, atol=1e-3)

def test_palu_attention_kernel(config):
    bsz = 1
    prompt_len = 63
    seq_len = 1
    dev = 'cuda:0'
    dtype = torch.float16

    attention = LlamaAttention(config, 0)
    palu_attention = LlamaPaluAttention.from_attention(attention, config)
    
    attention = attention.to(dev, dtype)
    palu_attention = palu_attention.to(dev, dtype)
    prompt_inputs = torch.randn(bsz, prompt_len, config.hidden_size).to(dev, dtype)
    inputs = torch.randn(bsz, seq_len, config.hidden_size).to(dev, dtype)
    prompt_position_ids = torch.arange(prompt_len).unsqueeze(0)  # Shape: [1, seq_length]
    generate_position_ids = torch.arange(prompt_len, prompt_len+seq_len).unsqueeze(0)  # Shape: [1, seq_length]
    kv_cache = DynamicCache()
    palu_kv_cache = DynamicCache()

    # Prompt
    attn_output, attn_weights, kv_cache = attention(prompt_inputs, output_attentions=True, 
                                                    past_key_value=kv_cache, position_ids=prompt_position_ids)
    palu_attn_output, palu_attn_weights, palu_kv_cache = palu_attention(prompt_inputs, output_attentions=True, 
                                                                        past_key_value=palu_kv_cache, position_ids=prompt_position_ids)
   
    torch.testing.assert_close(attn_output, palu_attn_output, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(attn_weights, palu_attn_weights, rtol=1e-3, atol=1e-3)

    # Generate
    # Golden
    golden_attn_output, golden_attn_weights, _ = attention(inputs, output_attentions=True, 
                                                           past_key_value=kv_cache, position_ids=generate_position_ids)
    # Kernel
    palu_attn_output, palu_attn_weights, _ = palu_attention(inputs, output_attentions=True, 
                                                            past_key_value=palu_kv_cache, position_ids=generate_position_ids)
    
    torch.testing.assert_close(golden_attn_weights, palu_attn_weights, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(golden_attn_output, palu_attn_output, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    _set_random_seed()
    _config = LlamaConfig()
    _config.group_size = 4
    _config.num_groups = _config.num_attention_heads // 4
    _config.total_rank_k = 4096
    _config.total_rank_v = 4096
    _config.k_bits = 16
    _config.v_bits = 16
    # test_palu_attention_fusion(_config)
    # test_palu_attention_kernel(_config)
    # test_palu_attention_inherit_fusion(_config)
    test_palu_attention_inherit_fusion_no_rope(_config)
    