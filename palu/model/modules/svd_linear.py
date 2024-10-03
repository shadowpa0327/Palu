import torch
import torch.nn as nn
from .quant import Quantizer
from .hadamard_utils import apply_hadamard

def _per_head_whiten_decomposition_from_weight(weight, scaling_diag_matrix, rank):
    original_dtype = weight.dtype
    try:
        scaling_diag_matrix = scaling_diag_matrix.to(weight.device)
    except AttributeError:
        raise FileExistsError("Cache may not be loaded correctly")
    
    # Get the inverse of scaling_diag_matrix
    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix.to(torch.float32))

    # Multiply scaling_diag_matrix to weight matrix
    W_scale = torch.matmul(weight.to(torch.float32), scaling_diag_matrix.to(torch.float32))
    
    U, S, Vt = torch.linalg.svd(W_scale, full_matrices=False)
    
    V = torch.matmul(Vt, scaling_matrix_inv)
    
    # Low rank approximation to the target rank
    U = U[:, :rank]
    S = S[:rank]
    V = V[:rank, :]
    
    sqrtSigma = torch.sqrt(torch.diag(S))

    # Fuse the SVD components
    L = torch.matmul(U, sqrtSigma).to(original_dtype)
    R = torch.matmul(sqrtSigma, V).to(original_dtype)
    
    return L, R

def _per_head_decomposition_from_weight(weight, rank):
    original_dtype = weight.dtype
    # Get weight matrix decomposed
    U, S, Vt = torch.linalg.svd(weight.to(torch.float32), full_matrices=False)

    # Low rank approximation to the target rank
    U = U[:, :rank]
    S = S[:rank]
    Vt = Vt[:rank, :]

    sqrtSigma = torch.sqrt(torch.diag(S))
    # Fuse the SVD components
    L = torch.matmul(U, sqrtSigma).to(original_dtype)
    R = torch.matmul(sqrtSigma, Vt).to(original_dtype)
    assert torch.allclose(torch.matmul(L, R), weight, atol=1e-3), "SVD decomposition failed"
    return L, R

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
        
        
        self.quantized_latents = False
        self.latent_quantizer = None
        
    def forward(self, 
                hidden_states: torch.Tensor):
        low_rank_latents = self.project_to_latent(hidden_states)
        if self.quantized_latents:
            low_rank_latents = self.quantize_latent(low_rank_latents)
        outputs = self.reconstruct(low_rank_latents)
        return outputs
    
    
    def project_to_latent(self, hidden_states:  torch.Tensor):
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
        return hidden_states

    def reconstruct(self, low_rank_latents: torch.Tensor):
        """
            low_rank_latents: Tensor of shape (batch_size, seq_len, r1 + r2 + ... )
        """
        outputs = []
        total_ranks = 0
        for i in range(self.num_groups):
            low_rank_latent = low_rank_latents[:, :, total_ranks: total_ranks+self.ranks[i]]
            outputs.append(self.U[i](low_rank_latent))
            total_ranks += self.ranks[i]

        """
            outputs: Tensor of shape (batch_size, seq_len, out_features)
        """
        return torch.cat(outputs, dim=-1)
    
    
    def quantize_latent(self, low_rank_latents: torch.Tensor):
        """
            low_rank_latents: Tensor of shape (batch_size, seq_len, r1 + r2 + ... )
        """
        assert self.latent_quantizer is not None, "Latent quantizer is not initialized."
        fake_quantized_low_rank_latents = []
        total_ranks = 0
        for i in range(self.num_groups):
            low_rank_latent = low_rank_latents[:, :, total_ranks: total_ranks+self.ranks[i]]
            fake_quantized_low_rank_latents.append(self.latent_quantizer(low_rank_latent))
            total_ranks += self.ranks[i]

        """
            fake_quantized_low_rank_latents: Tensor of shape (batch_size, seq_len, r1 + r2 + ...)
        """
        return torch.cat(fake_quantized_low_rank_latents, dim=-1)
    
    
    def configure_latent_quantizer(self, 
        n_bits: int, 
        group_size: int, 
        sym: bool,
        clip_ratio: float,
        hadamard = False
    ):
        #self.latent_quantizer = Quantizer(n_bits, group_size, sym, clip_ratio, hadamard)
        self.latent_quantizer = Quantizer(n_bits, group_size, sym, clip_ratio)
        if hadamard:
            self.fused_hadamard_matrix()
        self.quantized_latents = True
    
    
    def fused_hadamard_matrix(self):
        total_ranks = 0
        for i in range(self.num_groups):
            # Apply Q to VT
            VT_weight_i = self.VT.weight.data[total_ranks: total_ranks+self.ranks[i], :]
            VT_weight_i = apply_hadamard(VT_weight_i.t())
            self.VT.weight.data[total_ranks: total_ranks+self.ranks[i], :] = VT_weight_i.t()
            # Apply Q^T to U
            U_weight_i = self.U[i].weight.data
            U_weight_i = apply_hadamard(U_weight_i)
            self.U[i].weight.data = U_weight_i
            
            total_ranks += self.ranks[i]
    
    @staticmethod
    def from_linear_whiten(
        old_module: nn.Linear,
        ranks: list,
    ):   
        new_module = HeadwiseLowRankModule(ranks, old_module.in_features, old_module.out_features, bias=old_module.bias is not None)
        w = old_module.weight.data.reshape(len(ranks), -1, old_module.in_features)
        # Handle the cases where the bias is not None
        if old_module.bias is not None:
            b = old_module.bias.data.reshape(len(ranks), -1)
        
        wl = []
        wr = []
        for i in range(len(ranks)):
            l, r = _per_head_whiten_decomposition_from_weight(w[i], old_module.scaling_diag_matrix, ranks[i])
            # l: (head_dim, rank), r: (rank, hidden_size)
            wl.append(l)
            wr.append(r)

        # load to U
        for i in range(len(ranks)):
            if new_module.U[i].weight.data.shape != wl[i].shape:
                raise ValueError(f"{new_module.U[i].weight.data.shape} != {wl[i].shape}")
            new_module.U[i].weight.data = wl[i].contiguous()
            # Handle the cases where the bias is not None
            if old_module.bias is not None:
                new_module.U[i].bias.data = b[i]

        # load to VT
        # shape (sum(ranks), hidden_size)
        VT_weight = torch.cat(wr, dim=0).contiguous()
        assert new_module.VT.weight.data.shape == VT_weight.shape
        new_module.VT.weight.data = VT_weight
        
        return new_module
    
    @staticmethod
    def from_linear(
        old_module: nn.Linear,
        ranks: list,
    ):
        new_module = HeadwiseLowRankModule(ranks, old_module.in_features, old_module.out_features, bias=old_module.bias is not None)
        w = old_module.weight.data.reshape(len(ranks), -1, old_module.in_features)
        if old_module.bias is not None:
            b = old_module.bias.data.reshape(len(ranks), -1)
        wl = []
        wr = []
        for i in range(len(ranks)):
            l, r = _per_head_decomposition_from_weight(w[i], ranks[i])
            # l: (head_dim, rank), r: (rank, hidden_size)
            wl.append(l)
            wr.append(r)

        # load to U
        for i in range(len(ranks)):
            if new_module.U[i].weight.data.shape != wl[i].shape:
                raise ValueError(f"{new_module.U[i].weight.data.shape} != {wl[i].shape}")
            new_module.U[i].weight.data = wl[i].contiguous()
            if old_module.bias is not None:
                new_module.U[i].bias.data = b[i]
        # load to VT
        # shape (sum(ranks), hidden_size)
        VT_weight = torch.cat(wr, dim=0).contiguous()
        assert new_module.VT.weight.data.shape == VT_weight.shape
        new_module.VT.weight.data = VT_weight
        
        return new_module