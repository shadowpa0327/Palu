import math
import warnings
from typing import Optional, Tuple, Union

import torch
from torch import nn

from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb, 
    LlamaAttention, 
    LlamaConfig,
)

from transformers.cache_utils import Cache

from .configuration_palu_llama import PaluLlamaConfig
from ..modules.svd_linear import HeadwiseLowRankModule

from ...backend.fused_recompute import abx as recompute_k_gemv
from ...backend.q_matmul import cuda_bmm_fA_qB_outer

from ...quant.quant_kv_cache import ValueQuantizedCacheV2

class LlamaPaluAttention(LlamaAttention):
    """
    Llama Attention with Low-Rank KV-Cache with Palu. This module inherits from
    `LlamaAttention` but change linear layer and add custom Triton kernel.
    """
    def __init__(self, config: Union[PaluLlamaConfig, LlamaConfig], 
                 rank_k_list: list[int],
                 rank_v_list: list[int],
                 layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)

        #FIXME(brian1009): The interface design here is somehow inconsistent, need to be reviewed further
        self.rank_k_list = rank_k_list
        self.rank_v_list = rank_v_list
        #NOTE(brian1009): We assume all groups sharing the same rank for now
        self.group_rank_k = self.rank_k_list[0]
        self.group_rank_v = self.rank_v_list[0]
        self.total_rank_k = sum(self.rank_k_list)
        self.total_rank_v = sum(self.rank_v_list)
        assert len(self.rank_k_list) == len(self.rank_v_list), "The number of groups for k and v should be the same so far"
        self.num_groups = len(self.rank_k_list)
        self.group_size = self.num_heads // self.num_groups
        
        self.k_bits = config.k_bits
        self.v_bits = config.v_bits
                
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = HeadwiseLowRankModule(self.rank_k_list, self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = HeadwiseLowRankModule(self.rank_v_list, self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        self.v_recompution_fused = False
        self.has_prepared_for_k_recompute_kernel = False
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        # assert self.is_prepared, "Please call palu_prepare() method before forward"
        
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_h_states = self.k_proj.project_to_latent(hidden_states)
        value_h_states = self.v_proj.project_to_latent(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_h_states = key_h_states.view(bsz, q_len, self.num_groups, self.group_rank_k).transpose(1, 2)
        value_h_states = value_h_states.view(bsz, q_len, self.num_groups, self.group_rank_v).transpose(1, 2)

        #key_h_states_quant, key_scales, key_zeros = quant_and_pack_vcache(key_h_states, self.group_rank_k, self.k_bits)
        #if self.v_bits != 16:
        #    value_h_states_quant, value_scales, value_zeros = quant_and_pack_vcache(value_h_states, self.group_rank_v, self.v_bits)
        # kv_seq_len = key_states.shape[-2]
        
        kv_seq_len = key_h_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        
        # cos, sin = self.rotary_emb(query_states, seq_len=kv_seq_len)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            #key_h_states, value_h_states = past_key_value.update(key_h_states, value_h_states, self.layer_idx)
            if self.v_bits == 16:
                key_h_states, value_h_states = past_key_value.update(
                    key_h_states, value_h_states, self.layer_idx
                )
            else: 
                # key_h_states, value_h_states_quant, value_scales, value_zeros = past_key_value.update(
                #     key_h_states, value_h_states_quant, self.layer_idx, value_scales, value_zeros
                # )
                assert isinstance(past_key_value, ValueQuantizedCacheV2), "When the value is not 16-bit, we assume past_key_value to be ValueQuantizedCacheV2"
                key_h_states, value_h_states_quant, value_scales, value_zeros, value_h_states_full = past_key_value.update(
                    key_h_states, value_h_states, self.layer_idx
                ) 
                #NOTE(brian1009): We already transposed the value_h_states_quant in the update function. 
                # Now the shape of value_h_states_quant is (bsz, num_heads, group_rank_v, seq_len) and its contiguous.
                # This is for saving the an extra contigous operation in the kernel.
                value_h_states_quant = value_h_states_quant.transpose(2, 3)
                
        if q_len > 1:
            # Prompting
            # Recompute the key states
            key_h_states = key_h_states.transpose(1, 2).reshape(bsz, kv_seq_len, self.total_rank_k)
            key_states = self.k_proj.reconstruct(key_h_states)
            key_states = key_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            # Apply RoPE after recomputing the key states
            cos, sin = self.rotary_emb(query_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        else:
            # Generating (Apply our reconsturction kernel)
            # A: (num_heads, 1, head_dim)
            # B: (num_heads, rank_per_groups, head_dim)
            # X: (num_head_groups, seq_len, rank_per_groups)
            # TODO: Optimize RoPE & sqrt(head_dim) into kernel
            # TODO: Check if sin & cos are share among different blocks
            cos, sin = self.rotary_emb(query_states, seq_len=kv_seq_len)
            #query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin, position_ids)
            assert bsz == 1, "Only support batch size 1 for now"
            A = query_states.squeeze(0)
            
            assert self.has_prepared_for_k_recompute_kernel, "Please call prepared_k_merged_U() before forward using customized Triton Kernel"
            B = self.concated_B_for_k_recompute
            #B = self.k_proj.B
            X = key_h_states.squeeze(0)
            attn_weights = recompute_k_gemv(A, B, X).unsqueeze(0) / math.sqrt(self.head_dim)

        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask


        # Upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        if self.v_recompution_fused:
            # Fusion version
            # attn_weights: (bsz, num_groups, q_len * group_size, kv_seq_len)
            assert self.v_recompution_fused, "Please call fused_v_recompute_to_o() before forward"
            attn_h_weights = attn_weights.reshape(1, self.num_groups, q_len * self.group_size, kv_seq_len)
            if self.v_bits == 16:
                attn_h_output = torch.matmul(attn_h_weights, value_h_states)
            else:
                if value_h_states_full is None:
                    value_full_length = 0
                    attn_h_output = cuda_bmm_fA_qB_outer(
                        group_size=self.group_rank_v, fA=attn_h_weights[:, :, :, :], qB=value_h_states_quant,
                        scales=value_scales, zeros=value_zeros,
                        bits = self.v_bits
                    )
                else:
                    value_full_length = value_h_states_full.shape[-2]
                    attn_h_output = cuda_bmm_fA_qB_outer(
                        group_size=self.group_rank_v, fA=attn_h_weights[:, :, :, :-value_full_length], qB=value_h_states_quant,
                        scales=value_scales, zeros=value_zeros,
                        bits = self.v_bits
                    )
                    attn_h_output += torch.matmul(attn_h_weights[:, :, :, -value_full_length:], value_h_states_full)
            # attn_h_output: (bsz, num_heads, q_len * group_size, group_rank)
            attn_output = attn_h_output.reshape(1, self.num_heads, q_len, self.group_rank_v)
        else:
            # Original version
            value_h_states = value_h_states.transpose(1, 2).reshape(bsz, kv_seq_len, self.total_rank_v)
            value_states = self.v_proj.reconstruct(value_h_states)
            value_states = value_states.view(bsz, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            attn_output = torch.matmul(attn_weights, value_states)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        
        
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
    def fused_v_recompute_to_o(self):
        """
        Fuses the `v_proj` components into `o_proj` for optimized computation.
        Creates a new `o_proj` layer with fused weights.
        """
        
        # Calculate the required dimensions for the new o_proj layer
        fused_hidden_dim_o = self.group_rank_v * self.num_heads
        new_o_proj = nn.Linear(fused_hidden_dim_o, self.hidden_size, bias=self.o_proj.bias is not None)
        new_o_weight = torch.zeros_like(new_o_proj.weight)  # Ensure same device and dtype

        # Perform fusion by iterating over groups and ranks
        total_dims_2, total_fused_dims = 0, 0
        head_dim = self.head_dim
        for group_idx in range(self.num_groups):
            total_dims = 0
            for _ in range(self.group_size):
                v_proj_U = self.v_proj.U[group_idx].weight  # Assume U_list holds the decomposed matrices
                # Perform the fusion by multiplying corresponding segments
                new_o_weight[:, total_fused_dims:total_fused_dims + self.group_rank_v] = (
                    self.o_proj.weight[:, total_dims_2:total_dims_2 + head_dim]
                    @ v_proj_U[total_dims:total_dims + head_dim, :]
                )
                total_dims += head_dim
                total_dims_2 += head_dim
                total_fused_dims += self.group_rank_v

        # Copy fused weights to new o_proj
        with torch.no_grad():
            new_o_proj.weight.copy_(new_o_weight)

        # Update the o_proj to the fused version while preserving the device and dtype
        new_o_proj = new_o_proj.to(self.o_proj.weight.device, self.o_proj.weight.dtype)
        self.o_proj = new_o_proj
        
        #TODO(brian1009): Remove the U_list to CPU to save GPU memory
        self.v_recompution_fused = True
        torch.cuda.empty_cache()
    
    def prepared_k_merged_U(self):
        U_list_T = [x.weight.data.T for x in self.k_proj.U]
        b = torch.stack(U_list_T)
        b = b.reshape(self.num_groups, self.group_rank_k, self.group_size, self.head_dim)
        b = b.transpose(1, 2)
        b = b.reshape(self.num_heads, self.rank_k_list[0], self.head_dim)
        self.concated_B_for_k_recompute = nn.Parameter(b)
        #TODO(brian1009): Remove the U_list to CPU to save GPU memory
        self.has_prepared_for_k_recompute_kernel = True
        
    @staticmethod
    def from_attention(
        module: LlamaAttention,
        config: LlamaConfig,
        rank_k_list: list[int],
        rank_v_list: list[int],
        no_fusion: bool = False,
    ):
        new_module = LlamaPaluAttention(config,
                                        rank_k_list,
                                        rank_v_list,
                                        module.layer_idx)
        new_module.q_proj = module.q_proj
        new_module.k_proj = HeadwiseLowRankModule.from_linear(module.k_proj, new_module.rank_k_list)
        new_module.v_proj = HeadwiseLowRankModule.from_linear(module.v_proj, new_module.rank_v_list)
        new_module.o_proj = module.o_proj
        
        
        # No fusion version
        if not no_fusion:
            new_module.fused_v_recompute_to_o()
        
        return new_module
