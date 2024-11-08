import torch
import pytest
from palu.quant.q_packing import unpack_and_dequant_vcache, quant_and_pack_vcache
from palu.backend.q_matmul import cuda_bmm_fA_qB_outer

# Set the seed for reproducibility
torch.random.manual_seed(1234)

# Define tolerance levels for comparison
ATOL = 1e-1
RTOL = 1e-3

# Define test cases for different tensor shapes
# Each tuple represents (q_len, seq_len, num_heads, block_size, num_bits)
TEST_CASES = [
    (4, 1024, 8, 128, 128, 4),
]

def run_orig_matmul(attn_weights, Value):
    """ Run the original matmul implementation for verification """
    return torch.matmul(attn_weights, Value)

def run_palu_matmul(attn_weights, V_quant, V_scales, V_mn, block_size, num_bits):
    """ Run the PALU custom matmul implementation """
    return cuda_bmm_fA_qB_outer(block_size, attn_weights, V_quant, V_scales, V_mn, num_bits)

@pytest.mark.parametrize("q_len, seq_len, num_heads, grouped_rank, block_size, num_bits", TEST_CASES)
def test_palu_matmul(q_len, seq_len, num_heads, grouped_rank, block_size, num_bits):
    """Parameterized test to validate correctness across multiple input shapes."""

    # Generate random attention weights and value tensors with updated shapes
    attn_weights = torch.rand(1, num_heads, q_len, seq_len, dtype=torch.float16).cuda()
    Value = torch.rand(1, num_heads, seq_len, num_heads * grouped_rank, dtype=torch.float16).cuda()
    
    # Quantize Value tensor with the updated shape
    V_quant, V_scales, V_mn = quant_and_pack_vcache(Value, block_size, num_bits)
    
    # Dequantize to get the 'golden' reference
    V_dequant = unpack_and_dequant_vcache(V_quant, V_scales, V_mn, block_size, num_bits)
    
    # Chagne the underlying memory layout implicitly. 
    # This is to simulated the phenomenon we used in Palu's e2e.
    V_quant = V_quant.transpose(-2, -1).contiguous().transpose(-2, -1)
    V_scales = V_scales.transpose(-2, -1).contiguous().transpose(-2, -1)
    V_mn = V_mn.transpose(-2, -1).contiguous().transpose(-2, -1)
    
    # Run the original and PALU implementations
    golden_output = run_orig_matmul(attn_weights, V_dequant)
    palu_output = run_palu_matmul(attn_weights, V_quant, V_scales, V_mn, block_size, num_bits)
    
    # Check for correctness within tolerance
    test_passed = torch.allclose(palu_output, golden_output, atol=ATOL, rtol=RTOL)
    
    assert test_passed, f"Test failed for shape (q_len={q_len}, seq_len={seq_len}, num_heads={num_heads}, grouped_rank={grouped_rank})"
    print(f"Test passed for shape (q_len={q_len}, seq_len={seq_len}, num_heads={num_heads}, grouped_rank={grouped_rank})")

if __name__ == '__main__':
    # This Test is mainly for testing the quantization kernels for calculation the multiplication of attention score and value.
    for case in TEST_CASES:
        q_len, seq_len, num_heads, block_size, num_bits = case
        print(f"Running test for shape (q_len={q_len}, seq_len={seq_len}, num_heads={num_heads})")
        test_palu_matmul(q_len, seq_len, num_heads, block_size, num_bits)
