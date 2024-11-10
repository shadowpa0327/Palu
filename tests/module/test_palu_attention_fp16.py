import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import pytest
from palu.model.svd_llama.palu_llama_attention import LlamaPaluAttention
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.cache_utils import DynamicCache
from transformers import AutoConfig


@pytest.fixture(scope="module", autouse=True)
def set_random_seed():
    """Fixture to set a fixed random seed for reproducibility."""
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.fixture
def llama_config():
    """Fixture to initialize Llama configuration."""
    config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
    config.k_bits = 16
    config.v_bits = 16
    return config


def test_prefilling(llama_config):
    """
    Test prefilling attention with both the original and PALU attention modules.
    Verifies that the attention weights and outputs are nearly identical.
    """
    bsz, seq_len, hidden_dim = 1, 256, 4096
    attn = LlamaAttention(llama_config, layer_idx=0).half().cuda()
    palu_attn = LlamaPaluAttention.from_attention(
        module=attn,
        config=llama_config,
        rank_k_list=[512 for _ in range(8)],
        rank_v_list=[512 for _ in range(8)],
        no_fusion=True
    ).cuda()
    
    inputs = torch.rand(bsz, seq_len, hidden_dim, dtype=torch.float16).cuda()
    orig_output, orig_attn_weights, _ = attn(inputs, output_attentions=True)
    palu_output, palu_attn_weights, _ = palu_attn(inputs, output_attentions=True)

    # Assert closeness in prefilling
    torch.testing.assert_close(orig_attn_weights, palu_attn_weights, rtol=1e-3, atol=7.5e-3)
    torch.testing.assert_close(orig_output, palu_output, rtol=1e-3, atol=7.5e-3)

    #Merge the recomputation matrix into o_projection
    palu_attn.fused_v_recompute_to_o()
    orig_output, _, _ = attn(inputs)
    palu_output, _, _ = palu_attn(inputs)
    
    torch.testing.assert_close(orig_output, palu_output, rtol=1e-3, atol=1e-3)


def test_decoding(llama_config):
    """
    Test decoding with both the original and PALU attention modules.
    Verifies that output and attention weights are nearly identical.
    """
    bsz, prompt_len, decode_len = 1, 63, 1 
    device, dtype = "cuda", torch.float16

    attn = LlamaAttention(llama_config, layer_idx=0).to(device, dtype)
    palu_attn = LlamaPaluAttention.from_attention(
        module=attn,
        config=llama_config,
        rank_k_list=[512 for _ in range(8)],
        rank_v_list=[512 for _ in range(8)],
        no_fusion=False # Directly running in fusion mode
    ).to(device, dtype)
    
    prompt_inputs = torch.rand(bsz, prompt_len, llama_config.hidden_size).to(device, dtype)
    generate_inputs = torch.rand(bsz, decode_len, llama_config.hidden_size).to(device, dtype)
    prompt_position_ids = torch.arange(prompt_len).unsqueeze(0)  # Shape: [1, seq_length]
    generate_position_ids = torch.arange(prompt_len, prompt_len + decode_len).unsqueeze(0)  # Shape: [1, seq_length]
    kv_cache, palu_kv_cache = DynamicCache(), DynamicCache()

    # Run prompting
    attn_output, _, kv_cache = attn(prompt_inputs, output_attentions=False, past_key_value=kv_cache, position_ids=prompt_position_ids)
    palu_attn_output, _, palu_kv_cache = palu_attn(prompt_inputs, output_attentions=False, past_key_value=palu_kv_cache, position_ids=prompt_position_ids)

    torch.testing.assert_close(attn_output, palu_attn_output, rtol=1e-3, atol=1e-3)

    # Run generation step
    palu_attn.prepared_k_merged_U()
    attn_output, attn_weights, kv_cache = attn(generate_inputs, output_attentions=True, past_key_value=kv_cache, position_ids=generate_position_ids)
    palu_attn_output, palu_attn_weights, palu_kv_cache = palu_attn(generate_inputs, output_attentions=True, past_key_value=palu_kv_cache, position_ids=generate_position_ids)

    torch.testing.assert_close(attn_weights, palu_attn_weights, rtol=5e-3, atol=3e-2)
    torch.testing.assert_close(attn_output, palu_attn_output, rtol=5e-3, atol=3e-2)
