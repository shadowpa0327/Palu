import math
import warnings
from typing import Optional, Union, Tuple

import torch
from torch import nn

from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.opt.modeling_opt import (
    OPTAttention, OPTConfig,
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
        attn_module: Union[LlamaAttention, OPTAttention] = None,
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
            b = b.reshape(attn_module.num_heads, new_module.ranks[0], attn_module.head_dim)
            new_module.B = nn.Parameter(b)

        # load to VT
        # shape (sum(ranks), hidden_size)
        VT_weight = torch.cat(wr, dim=0).contiguous()
        assert new_module.VT.weight.data.shape == VT_weight.shape
        new_module.VT.weight.data = VT_weight
        
        return new_module


class OPTPaluAttention(OPTAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: OPTConfig, is_decoder: bool = False, **kwargs):
        super().__init__(config)
        
        self.group_size = config.group_size
        self.num_groups = config.num_groups
        self.total_rank_k = config.total_rank_k
        self.total_rank_v = config.total_rank_v
        self.k_bits = config.k_bits
        self.v_bits = config.v_bits
        self.group_rank_k = self.total_rank_k // self.num_groups
        self.group_rank_v = self.total_rank_v // self.num_groups
        self.fused_hidden_dim_q = self.group_rank_k * self.num_heads
        self.fused_hidden_dim_o = self.group_rank_v * self.num_heads
        self.rank_k_list = [self.group_rank_k for _ in range(self.num_groups)]
        self.rank_v_list = [self.group_rank_v for _ in range(self.num_groups)]

        self.q_proj = nn.Linear(self.embed_dim, self.fused_hidden_dim_q, bias=self.enable_bias)
        self.k_proj = HeadwiseLowRankModule(self.rank_k_list, self.embed_dim, self.num_heads * self.head_dim, bias=self.enable_bias)
        self.v_proj = HeadwiseLowRankModule(self.rank_v_list, self.embed_dim, self.num_heads * self.head_dim, bias=self.enable_bias)
        self.out_proj = nn.Linear(self.fused_hidden_dim_o, self.embed_dim, bias=self.enable_bias)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value

    @staticmethod
    def from_attention(
        module: OPTAttention,
        config: OPTConfig,
        no_fusion: bool = False,
    ):
        new_module = OPTPaluAttention(config, is_decoder=True)
        new_module.k_proj = HeadwiseLowRankModule.from_linear(module.k_proj, new_module.rank_k_list)
        new_module.v_proj = HeadwiseLowRankModule.from_linear(module.v_proj, new_module.rank_v_list)

        # No fusion version
        if no_fusion:
            new_module.q_proj = module.q_proj
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
                q_head = module.q_proj.weight[head_id*head_dim:(head_id+1)*head_dim, :]
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
