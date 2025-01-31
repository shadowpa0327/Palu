import torch
import torch.nn as nn
import pytest
from palu.backend.fused_recompute import abx

# Define tolerance levels for comparison
ATOL = 8e-3
RTOL = 1e-3

# Set random seed for reproducibility
def set_random_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Rotary Embedding - defined here as a helper function
def LlamaRotaryEmbedding(dim: int, end: int, theta: float = 10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
    t = torch.arange(end, dtype=torch.int64).type_as(inv_freq)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_pytorch(x, cos, sin, unsqueeze_dim=0):
    cos = cos.unsqueeze(unsqueeze_dim).to(x.device)
    sin = sin.unsqueeze(unsqueeze_dim).to(x.device)
    x_emb = (x * cos) + (rotate_half(x) * sin)
    return x_emb

def torch_abx(a, b, x):
    x_expand = x.unsqueeze(1)
    b_reshape = b.reshape(-1, b.shape[0] // x.shape[0], b.shape[-2], b.shape[-1])
    xb = x_expand @ b_reshape
    xb = xb.reshape(b.shape[0], -1, b.shape[-1])
    cos, sin = LlamaRotaryEmbedding(dim=128, end=x.shape[1])
    xb_rope = apply_rotary_pos_emb_pytorch(x=xb, cos=cos, sin=sin)
    axb = a @ xb_rope.transpose(-1, -2).to(torch.float16)
    return axb

# Define test cases with varied parameters
# Each tuple represents (num_heads, head_dim, total_rank, num_groups, seq_len)
TEST_CASES = [
    (32, 128, 1024, 8, 64),
    (32, 128, 2048, 8, 64),
    (32, 128, 1024, 8, 256),
    (32, 128, 1024, 8, 1024),
    (32, 128, 1024, 8, 4096),
    # test arbitary output length
    (32, 128, 1024, 8, 65),
    (32, 128, 1024, 8, 78),
    (32, 128, 1024, 8, 4099),
    (32, 128, 1024, 8, 63)
]

@pytest.mark.parametrize("num_heads, head_dim, total_rank, num_groups, seq_len", TEST_CASES)
def test_abx(num_heads, head_dim, total_rank, num_groups, seq_len):
    """Test the abx function for various configurations."""
    set_random_seed(0)
    rank_per_groups = total_rank // num_groups
    dtype = torch.float16
    device = "cuda"
    
    # Create test tensors with configurable seq_len
    A = torch.randn(num_heads, 1, head_dim, dtype=dtype, device=device)
    B = torch.randn(num_heads, rank_per_groups, head_dim, dtype=dtype, device=device)/10
    X = torch.randn(num_groups, seq_len, rank_per_groups, dtype=dtype, device=device)/10
    
    # Run the original and custom implementations
    axb = torch_abx(A, B, X)
    ours = abx(A, B, X)

    
    attn_weights_axb = nn.functional.softmax(axb, dim=-1, dtype=torch.float32).to(torch.float16)
    attn_weights_ours = nn.functional.softmax(ours, dim=-1, dtype=torch.float32).to(torch.float16)
    # Check for correctness within tolerance
    #max_diff = torch.max(torch.abs(axb - ours))
    max_diff = torch.max(torch.abs(attn_weights_axb - attn_weights_ours))
    print(f"Max diff: {max_diff.item()}")
    assert torch.allclose(axb, ours, atol=ATOL, rtol=RTOL), f"Test failed: Max diff {max_diff.item()} exceeded tolerance"
    
    print(f"Test passed for (num_heads={num_heads}, head_dim={head_dim}, total_rank={total_rank}, num_groups={num_groups}, seq_len={seq_len}) with max diff: {max_diff.item()}")

# For manual testing without pytest, you could include a simple runner:
if __name__ == '__main__':
    set_random_seed(0)
    for case in TEST_CASES:
        num_heads, head_dim, total_rank, num_groups, seq_len = case
        print(f"Running test for (num_heads={num_heads}, head_dim={head_dim}, total_rank={total_rank}, num_groups={num_groups}, seq_len={seq_len})")
        test_abx(num_heads, head_dim, total_rank, num_groups, seq_len)
