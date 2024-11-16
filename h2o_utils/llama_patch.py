import os
import pdb
import copy
import math
import numpy as np 
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List
import warnings
import types
import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings, is_flash_attn_greater_or_equal_2_10, logging, is_flash_attn_2_available
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaAttention, apply_rotary_pos_emb, _get_unpad_data, LlamaRMSNorm, LlamaPreTrainedModel, LlamaMLP, repeat_kv, LlamaLinearScalingRotaryEmbedding, LlamaDynamicNTKScalingRotaryEmbedding
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, AttentionMaskConverter
from transformers.cache_utils import Cache, DynamicCache
_CONFIG_FOR_DOC = "LlamaConfig"


def llama_h2o_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)


        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)


        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)


        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)


    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)


    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)


    past_key_value = getattr(self, "past_key_value", past_key_value)
    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)


    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)


    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        #causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    # h2o start
    #breakpoint()
    attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
    if self.attention_masks_next is not None:
        attn_weights = attn_weights * self.attention_masks_next + (1 - self.attention_masks_next) * torch.finfo(attn_weights.dtype).min

    # h2o end
    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    # h2o start
    current_scores_sum = attn_weights.sum(0).sum(1) # (heads, k-tokens)


    #Accumulate attention scores
    if not self.previous_scores == None:
        current_scores_sum[:, :-1] += self.previous_scores #(Enlarged Sequence)
    else:
        self.heavy_budget = int(self.heavy_budget_ratio * current_scores_sum.shape[-1])
        self.recent_budget = int(self.recent_budget_ratio * current_scores_sum.shape[-1])
        self.cache_budget = self.heavy_budget + self.recent_budget

    dtype_attn_weights = attn_weights.dtype
    attn_weights_devices = attn_weights.device
    assert attn_weights.shape[0] == 1
    self.previous_scores = current_scores_sum #(heads, k-tokens)
    attn_mask = torch.ones(current_scores_sum.shape[0], current_scores_sum.shape[1]+1).to(dtype_attn_weights).to(attn_weights_devices)


    attn_tokens_all = self.previous_scores.shape[-1]

    if attn_tokens_all > self.cache_budget:
        # activate most recent k-cache
        if not self.recent_budget == 0:
            attn_mask[:, :-self.recent_budget] = 0
            selected_set = self.previous_scores[:, :-self.recent_budget]
        else:
            selected_set = self.previous_scores


        if not self.heavy_budget == 0:
            _, keep_topk = selected_set.topk(k=self.heavy_budget, dim=-1, largest=True)
            attn_mask = attn_mask.scatter(-1, keep_topk, 1)


    self.attention_masks_next = attn_mask.unsqueeze(0).unsqueeze(2)


    score_mask = attn_mask[:,:-1]
    score_mask[:, -self.recent_budget:] = 1
    self.previous_scores = self.previous_scores * score_mask
    
    # h2o end
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)


    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )


    attn_output = attn_output.transpose(1, 2).contiguous()


    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)


    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)


    if not output_attentions:
        attn_weights = None


    return attn_output, attn_weights, past_key_value


def _reset_masks(self):
    self.attention_masks_next = None
    self.heavy_budget = None
    self.recent_budget = None
    self.cache_budget = None
    self.previous_scores = None

def monkey_patch_h2o(model, heavy_ratio, recent_ratio):
     for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            monkey_patch_h2o(module, heavy_ratio, recent_ratio)
        if isinstance(module, LlamaAttention):
            assert module.config._attn_implementation == 'eager', "H2O only supports eager implementation"
            module.heavy_budget_ratio = heavy_ratio
            module.recent_budget_ratio = recent_ratio
            
            module.forward = types.MethodType(llama_h2o_forward, module)
            module._reset_masks = types.MethodType(_reset_masks, module)
            module._reset_masks()

    