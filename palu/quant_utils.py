from .model.modules import HeadwiseLowRankModule
import torch.nn as nn

def configure_latent_quantizer(
        model: nn.Module, 
        n_bits:4,
        group_size=0,
        sym=True,
        clip_ratio=1.0,
        hadamard=False,
    ):
    
    for name, module in model.named_modules():
        if isinstance(module, HeadwiseLowRankModule):
            module.configure_latent_quantizer(n_bits, group_size, sym, clip_ratio, hadamard)
            
        
    