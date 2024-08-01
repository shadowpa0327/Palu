import torch

def LlamaRotaryEmbedding(dim: int, end: int, theta: float = 10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
    t = torch.arange(end, dtype=torch.int64).type_as(inv_freq)
    freqs = torch.outer(t, inv_freq)
    # Different from paper, but it uses a different permutation in order to obtain the same calculation
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_pytorch(x, cos, sin, unsqueeze_dim=0):
    cos = cos.unsqueeze(unsqueeze_dim).to(x.device)
    sin = sin.unsqueeze(unsqueeze_dim).to(x.device)
    x_emb = (x * cos) + (rotate_half(x) * sin)
    return x_emb
