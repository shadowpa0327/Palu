from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM
)
import torch.nn as nn
import torch
from types import SimpleNamespace
from .configuration_palu_llama import PaluLlamaConfig
from .palu_llama_attention import LlamaPaluAttention
from ..modules.svd_linear import HeadwiseLowRankModule

class PaluLlamaForCausalLM(LlamaForCausalLM):
    config_class = PaluLlamaConfig
    def __init__(self, config:PaluLlamaConfig):
        self.head_wise_ranks=config.head_wise_ranks
        self.palu_attn_linear_only = config.palu_attn_linear_only
        config._attn_implementation = 'eager'
        super().__init__(config)
        self._replace_modules(self.palu_attn_linear_only)
    @staticmethod
    def get_kv_info(llama: LlamaForCausalLM, num_heads_in_lr_groups: int):
        num_lr_groups = llama.config.num_attention_heads // num_heads_in_lr_groups
        num_lr_kv_groups = llama.config.num_key_value_heads // num_heads_in_lr_groups
        head_dim = llama.config.hidden_size // llama.config.num_attention_heads
        lr_group_dims = head_dim * num_heads_in_lr_groups
        
        if num_lr_groups * num_heads_in_lr_groups != llama.config.num_attention_heads:
            raise ValueError(
                f"num_heads must be divisible by num_heads_in_lr_groups (got `num_heads`: {llama.config.num_attention_heads}"
                f" and `num_heads_in_lr_groups`: {num_heads_in_lr_groups})."
            )
    
        if num_lr_kv_groups * num_heads_in_lr_groups != llama.config.num_key_value_heads:
            raise ValueError(
                f"num_key_value_heads must be divisible by num_heads_in_lr_groups (got `num_key_value_heads`: {llama.config.num_key_value_heads}"
                f" and `num_heads_in_lr_groups`: {num_heads_in_lr_groups})."
            )

        return SimpleNamespace(
            num_lr_groups=num_lr_kv_groups,
            lr_group_dims=lr_group_dims,
        )

    
    def _replace_modules(self, linear_only=True):
        if linear_only:
            # Mode 1: Only replace the linear layers to simulate the low-rank approximation
            full_name_dict = {module: name for name, module in self.named_modules()}
            linear_info = {}
            modules = [self]
            while len(modules) > 0:
                submodule = modules.pop()
                for name, raw_linear in submodule.named_children():
                    if isinstance(raw_linear, nn.Linear):
                        full_name = full_name_dict[raw_linear]
                        linear_info[raw_linear] = {
                            "father": submodule,
                            "name": name,
                            "full_name": full_name,
                        }
                    else:
                        modules.append(raw_linear)


            for name,module in self.named_modules():
                if name in self.head_wise_ranks:
                    info=linear_info[module]
                    new_layer=HeadwiseLowRankModule(self.head_wise_ranks[name],module.in_features,module.out_features,bias=module.bias is not None)
                    setattr(info["father"], info["name"], new_layer)
        else:
            # Mode 2: Replace all the attention modules with update forward path
            #FIXME (brian1009): Could be simplified further
            full_name_dict = {module: name for name, module in self.named_modules()}
            attn_info = {}
            modules = [self]
            while len(modules) > 0:
                submodule = modules.pop()
                for name, child in submodule.named_children():
                    if isinstance(child, LlamaAttention):
                        full_name = full_name_dict[child]
                        attn_info[child] = {
                            "father": submodule,
                            "name": name,
                            "full_name": full_name,
                        }
                    else:
                        modules.append(child)

            layer_id_counter = 0
            for name, module in self.named_modules():
                if isinstance(module, LlamaAttention):
                    info = attn_info[module]
                    rank_k_lists = self.config.head_wise_ranks[info["full_name"]+".k_proj"]
                    rank_v_lists = self.config.head_wise_ranks[info["full_name"]+".v_proj"]
                    new_layer = LlamaPaluAttention(self.config, 
                                                   rank_k_list=rank_k_lists,
                                                    rank_v_list=rank_v_lists,
                                                    layer_idx=layer_id_counter)
                    setattr(info["father"], info["name"], new_layer)
                    layer_id_counter += 1
                
        torch.cuda.empty_cache()
        
        
    def prepare_for_palu_inference(self):
        # invoke fused_v_recompute_to_o and prepared_k_merged_U functions on LlamaPaluAttention
        for name, module in self.named_modules():
            if isinstance(module, LlamaPaluAttention):
                module.fused_v_recompute_to_o()
                module.prepared_k_merged_U()