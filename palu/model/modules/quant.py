import torch
from torch import nn
from functools import partial

@torch.no_grad()
def quantize_tensor(w: torch.tensor, n_bits, group_size, sym, clip_ratio=1.0) -> torch.tensor:
    savedShape = w.shape
    assert w.dim() == 2 


    if group_size > 0:
        assert w.shape[-1] % group_size == 0
        w = w.reshape(-1, group_size) # row-major order

    assert w.dim() == 2, "Weight format: [-1, group]"
    assert n_bits < 16

    if sym:
        w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    else:
        w_max = w.amax(dim=-1, keepdim=True)
        w_min = w.amin(dim=-1, keepdim=True)

    if sym:
        q_max = (2**(n_bits-1)-1)
        q_min = (-2**(n_bits-1))
        if clip_ratio < 1.0:
            w_max = w_max * clip_ratio
        scales = w_max / q_max
        base = torch.zeros_like(scales)
    else:
        q_max = (2**(n_bits)-1)
        q_min = (0)
        if clip_ratio < 1.0:
            w_max *= clip_ratio
            w_min *= clip_ratio
        scales = (w_max-w_min).clamp(min=1e-5) / q_max
        base = torch.round(-w_min/scales).clamp_(min=q_min, max=q_max)
    w = (torch.clamp(torch.round(w / scales) + base, q_min, q_max) - base) * scales
    
    return w.reshape(savedShape)




class Quantizer(nn.Module):
    def __init__(self,
            n_bits: int, 
            group_size: int, 
            sym: bool,
            clip_ratio: float     
        ) -> None:
        super().__init__()
        self.n_bits = n_bits
        self.group_size = group_size
        self.sym = sym
        self.clip_ratio = clip_ratio
        

    @torch.no_grad()
    def forward(self, x):
        if self.n_bits >= 16:
            return x 
        
        qFunction = partial(
            quantize_tensor, 
            n_bits=self.n_bits,
            group_size=self.group_size,
            sym=self.sym,
            clip_ratio=self.clip_ratio
        )

        savedShape = x.shape
        x = x.view(-1, savedShape[-1])
        assert self.group_size == 0 or (savedShape[-1]) % self.group_size == 0, "Group size should be divisible by (dim)."

        x = qFunction(x)
        
        return x.view(savedShape)
        
    def to(self, *args, **kwargs):
        super(Quantizer, self).to(*args, **kwargs)
        return self