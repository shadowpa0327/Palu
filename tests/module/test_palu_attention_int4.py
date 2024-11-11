import warnings
import torch
import pytest
from palu.model.svd_llama.palu_llama_attention import LlamaPaluAttention
from palu.quant.quant_kv_cache import ValueQuantizedCacheV2
from palu.quant.q_packing import unpack_and_dequant_vcache
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers import AutoConfig

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
    config.v_bits = 4
    return config

# Define parameter values and custom IDs
params = [(129, 128), (256, 256), (512, 512)]
param_ids = [f"prompt_len={p[0]}, residual_length={p[1]}" for p in params]

@pytest.mark.parametrize("prompt_len, residual_length", params, ids=param_ids)
def test_decoding(llama_config, prompt_len, residual_length):
    """
    Test decoding with both the original and PALU attention modules.
    Verifies that output and attention weights are nearly identical.
    """
    bsz, decode_len = 1, 1
    device, dtype = "cuda", torch.float16

    # Initialize attention modules
    attn = LlamaAttention(llama_config, layer_idx=0).to(device, dtype)
    palu_attn = LlamaPaluAttention.from_attention(
        module=attn,
        config=llama_config,
        rank_k_list=[512 for _ in range(8)],
        rank_v_list=[512 for _ in range(8)],
        no_fusion=False  # Directly running in fusion mode
    ).to(device, dtype)
    palu_attn.prepared_k_merged_U()

    # Generate random inputs
    prompt_inputs = torch.rand(bsz, prompt_len, llama_config.hidden_size).to(device, dtype)
    generate_inputs = torch.rand(bsz, decode_len, llama_config.hidden_size).to(device, dtype)
    prompt_position_ids = torch.arange(prompt_len).unsqueeze(0)  # Shape: [1, seq_length]
    generate_position_ids = torch.arange(prompt_len, prompt_len + decode_len).unsqueeze(0)  # Shape: [1, seq_length]
    palu_kv_cache = ValueQuantizedCacheV2(bits=llama_config.v_bits, residual_length=residual_length)

    # Run prompting
    _, _, palu_kv_cache = palu_attn(
        prompt_inputs,
        output_attentions=False,
        past_key_value=palu_kv_cache,
        position_ids=prompt_position_ids
    )

    # Run generation step (Palu)
    palu_attn_output, palu_attn_weights, palu_kv_cache = palu_attn(
        generate_inputs,
        output_attentions=True,
        past_key_value=palu_kv_cache,
        position_ids=generate_position_ids
    )

    # Extract and process attention weights and value states for validation
    _, value_quant, scales, zeros, value_full = palu_kv_cache[0]
    value_quant = value_quant.transpose(2, 3)
    value_states = unpack_and_dequant_vcache(value_quant, scales, zeros, value_quant.shape[-1] * 8, 4)

    if value_full is not None:
        value_states = torch.cat([value_states, value_full], dim=-2)

    palu_attn_h_weights = palu_attn_weights.reshape(1, palu_attn.num_groups, decode_len * palu_attn.group_size, -1)
    attn_h_output_golden = torch.matmul(palu_attn_h_weights, value_states)
    attn_output_golden = attn_h_output_golden.reshape(1, palu_attn.num_heads, decode_len, -1)
    attn_output_golden = attn_output_golden.transpose(1, 2).contiguous()
    attn_output_golden = attn_output_golden.view(bsz, decode_len, -1)
    attn_output_golden = palu_attn.o_proj(attn_output_golden)

    # Assert that the outputs are close
    torch.testing.assert_close(palu_attn_output, attn_output_golden, rtol=1e-3, atol=1e-3)
