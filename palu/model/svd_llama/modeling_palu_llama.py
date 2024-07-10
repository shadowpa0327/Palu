from transformers import LlamaForCausalLM
import torch.nn as nn
from types import SimpleNamespace
from .configuration_palu_llama import PaluLlamaConfig
from ..modules.svd_linear import HeadwiseLowRankModule

class PaluLlamaForCausalLM(LlamaForCausalLM):
    config_class = PaluLlamaConfig
    def __init__(self, config:PaluLlamaConfig):
        super().__init__(config)
        self.head_wise_ranks=config.head_wise_ranks

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
