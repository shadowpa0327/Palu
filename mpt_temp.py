import math
import warnings
from typing import Optional, Union, Tuple

import torch
from torch import nn

from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.mpt.modeling_mpt import (
    MptAttention, MptConfig,
)


class HeadwiseLowRankModule(nn.Module):
    """ Headwise Low-Rank module """
    def __init__(self, ranks, in_features, out_features, bias):
        super().__init__()

        self.ranks = ranks
        self.num_groups = len(ranks)
        self.in_features = in_features
        self.out_features = out_features
        self.group_dim = out_features // self.num_groups

        if (self.group_dim * self.num_groups) != self.out_features:
            raise ValueError(
                f"out_features must be divisible by num_groups (got `out_features`: {self.out_features}"
                f" and `num_groups`: {self.num_groups})."
            )

        self.VT = nn.Linear(in_features, sum(ranks), bias=False)

        # Create the list of linear layers first
        Us = []
        for r in ranks:
            linear_layer = nn.Linear(r, self.group_dim, bias=bias)
            nn.init.normal_(linear_layer.weight)
            Us.append(linear_layer)

        self.U_list = nn.ModuleList(Us)
    
    def forward(self, hidden_states: torch.Tensor):
        """ hidden_states: Tensor of shape (batch_size, seq_len, in_features) """
        assert hidden_states.dim() == 3, f"hidden_states should have 3 dimensions, got {hidden_states.dim()}"
        
        hidden_states = self.VT(hidden_states)

        # hidden_states: Tensor of shape (batch_size, seq_len, r1 + r2 + ... )
        outputs = []
        total_ranks = 0
        for i in range(self.num_groups):
            outputs.append(self.U_list[i](hidden_states[:, :, total_ranks: total_ranks+self.ranks[i]]))
            total_ranks += self.ranks[i]

        return torch.cat(outputs, dim=-1)

    def project_to_latent(self, hidden_states: torch.Tensor):
        """ hidden_states: Tensor of shape (batch_size, seq_len, in_features) """
        assert hidden_states.dim() == 3, f"hidden_states should have 3 dimensions, got {hidden_states.dim()}"

        hidden_states = self.VT(hidden_states)

        return hidden_states
    
    def reconstruct(self, hidden_states: torch.Tensor):
        """ hidden_states: Tensor of shape (batch_size, seq_len, sum(ranks)) """
        assert hidden_states.dim() == 3, f"hidden_states should have 3 dimensions, got {hidden_states.dim()}"

        outputs = []
        total_ranks = 0
        for i in range(self.num_groups):
            outputs.append(self.U_list[i](hidden_states[:, :, total_ranks: total_ranks+self.ranks[i]]))
            total_ranks += self.ranks[i]

        return torch.cat(outputs, dim=-1)
    
    @staticmethod
    def from_linear(
        old_module: nn.Linear,
        ranks: list,
        attn_module: Union[LlamaAttention, MptAttention] = None,
    ):   
        new_module = HeadwiseLowRankModule(ranks, old_module.in_features, old_module.out_features, bias=old_module.bias is not None)
        w = old_module.weight.data.reshape(len(ranks), -1, old_module.in_features).float()

        wl = []
        wr = []
        for i in range(len(ranks)):
            l, s, r = torch.linalg.svd(w[i], full_matrices=False)
            l = l[:, 0:ranks[i]]
            s = s[0:ranks[i]]
            r = r[0:ranks[i], :]
            l = l.mul(s)

            # l: (head_dim, rank), r: (rank, hidden_size)
            wl.append(l)
            wr.append(r)

        # load to U
        for i in range(len(ranks)):
            if new_module.U_list[i].weight.data.shape != wl[i].shape:
                raise ValueError(f"{new_module.U_list[i].weight.data.shape} != {wl[i].shape}")
            new_module.U_list[i].weight.data = wl[i].contiguous()
        
        # Create B matrix for kernel
        if attn_module is not None:
            U_list_T = [x.weight.data.T for x in new_module.U_list]
            b = torch.stack(U_list_T)
            b = b.reshape(new_module.num_groups, new_module.ranks[0], attn_module.group_size, attn_module.head_dim)
            b = b.transpose(1, 2)
            b = b.reshape(attn_module.n_heads, new_module.ranks[0], attn_module.head_dim)
            new_module.B = nn.Parameter(b)

        # load to VT
        # shape (sum(ranks), hidden_size)
        VT_weight = torch.cat(wr, dim=0).contiguous()
        assert new_module.VT.weight.data.shape == VT_weight.shape
        new_module.VT.weight.data = VT_weight
        
        return new_module

class MptPaluAttention(MptAttention):
    """Multi-head self attention.
    Using torch or triton attention implemetation enables user to also use additive bias.
    """

    def __init__(self, config: MptConfig):
        super().__init__(config)
        
        self.group_size = config.group_size
        self.num_groups = config.num_groups
        self.total_rank_k = config.total_rank_k
        self.total_rank_v = config.total_rank_v
        self.k_bits = config.k_bits
        self.v_bits = config.v_bits
        self.group_rank_k = self.total_rank_k // self.num_groups
        self.group_rank_v = self.total_rank_v // self.num_groups
        self.fused_hidden_dim_q = self.group_rank_k * self.n_heads
        self.fused_hidden_dim_o = self.group_rank_v * self.n_heads
        self.rank_k_list = [self.group_rank_k for _ in range(self.num_groups)]
        self.rank_v_list = [self.group_rank_v for _ in range(self.num_groups)]

        assert config.no_bias, "Bias is not supported in PALU attention"
        self.q_proj = nn.Linear(self.hidden_size, self.fused_hidden_dim_q, bias=False)
        self.k_proj = HeadwiseLowRankModule(self.rank_k_list, self.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.v_proj = HeadwiseLowRankModule(self.rank_v_list, self.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.fused_hidden_dim_o, self.hidden_size, bias=False)


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_bias: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_length = hidden_states.shape[:2]

        mixed_qkv = self.Wqkv(hidden_states)
        query_states, key_states, value_states = mixed_qkv.chunk(3, dim=2)
        query_states = query_states.reshape(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.reshape(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.reshape(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            if len(past_key_value) != 0:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
            past_key_value = (key_states, value_states)
        else:
            past_key_value = (key_states, value_states)

        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.softmax_scale

        query_length = seq_length if past_key_value is None else seq_length + past_key_value[0].shape[2]

        if position_bias is not None:
            if len(position_bias.shape) != 3:
                raise ValueError(f"Expecting position_bias shape to be 3 dimensions, got {len(position_bias.shape)}")
            key_length = key_states.shape[-2]

            position_bias_query_index = max(0, position_bias.size(1) - query_length)
            position_bias_key_index = max(0, position_bias.size(2) - key_length)

            position_bias = position_bias[:, position_bias_query_index:, position_bias_key_index:]

            attention_scores = attention_scores + position_bias

        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask, torch.finfo(query_states.dtype).min)

        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.softmax(attention_scores.float(), dim=-1).to(value_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attn_dropout_p, training=self.training)

        context_states = torch.matmul(attn_weights, value_states)
        context_states = context_states.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, -1)
        attn_output = self.out_proj(context_states)

        return attn_output, attn_weights, past_key_value

    @staticmethod
    def from_attention(
        module: MptAttention,
        config: MptConfig,
        no_fusion: bool = False,
    ):
        # Copy Wqkv weight into q, k, v_proj
        q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        with torch.no_grad():
            q_proj.weight.copy_(module.Wqkv.weight[:module.hidden_size, :])
            k_proj.weight.copy_(module.Wqkv.weight[module.hidden_size:2*module.hidden_size, :])
            v_proj.weight.copy_(module.Wqkv.weight[2*module.hidden_size:, :])
        
        new_module = MptPaluAttention(config)
        new_module.k_proj = HeadwiseLowRankModule.from_linear(k_proj, new_module.rank_k_list)
        new_module.v_proj = HeadwiseLowRankModule.from_linear(v_proj, new_module.rank_v_list)

        # No fusion version
        if no_fusion:
            new_module.q_proj = q_proj
            new_module.out_proj = module.out_proj
            return new_module

        # Fusion version
        # TODO: We never change k, v proj into pure VT linear
        # new_module.k_proj = new_k_proj.VT
        # new_module.v_proj = new_v_proj.VT

        head_dim = module.head_dim
        group_size = config.group_size

        # Fuse k_proj.U into q_proj
        new_q_weights = []
        for i, B_group in enumerate(new_module.k_proj.U_list):
            B_group_weights = B_group.weight.view(group_size, head_dim, -1)
            for j, B_weight in enumerate(B_group_weights):
                head_id = i * group_size + j
                q_head = q_proj.weight[head_id*head_dim:(head_id+1)*head_dim, :]
                new_q_weights.append(B_weight.T @ q_head)

        with torch.no_grad():
            new_module.q_proj.weight.copy_(torch.cat(new_q_weights, dim=0))


        # Fuse v_proj.U into out_proj
        new_o_weights = []
        for i, B_group in enumerate(new_module.v_proj.U_list):
            B_group_weights = B_group.weight.view(group_size, head_dim, -1)
            for j, B_weight in enumerate(B_group_weights):
                head_id = i * group_size + j
                o_head = module.out_proj.weight[:, head_id*head_dim:(head_id+1)*head_dim]
                new_o_weights.append(o_head @ B_weight)

        with torch.no_grad():
            new_module.out_proj.weight.copy_(torch.cat(new_o_weights, dim=1))

        return new_module
