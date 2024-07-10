from transformers import MistralForCausalLM
from .configuration_asvd_mistral import PaluMistralConfig
import torch.nn as nn
import torch


class HeadwiseLowRankModule(nn.Module):
    """ Headwise low rank module """

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
        
        Us = []
        for r in ranks:
            Us.append(nn.Linear(r, self.group_dim, bias=bias))

        self.U = nn.ModuleList(Us)    
    def forward(self, 
                hidden_states: torch.Tensor):
        """
            hidden_states: Tensor of shape (batch_size, seq_len, in_features)
        """
        if hidden_states.dim() != 3:
            raise ValueError(
                "Input tensor should have dimension 3."
            )

        hidden_states = self.VT(hidden_states)
        """
            hidden_states: Tensor of shape (batch_size, seq_len, r1 + r2 + ... )
        """

        outputs = []
        total_ranks = 0
        for i in range(self.num_groups):
            outputs.append(self.U[i](hidden_states[:, :, total_ranks: total_ranks+self.ranks[i]]))
            total_ranks += self.ranks[i]

        """
            outputs: [
        """
        return torch.cat(outputs, dim=-1)

class PaluMistralForCausalLM(MistralForCausalLM):
    config_class = PaluMistralConfig
    def __init__(self, config:PaluMistralConfig):
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
                
        