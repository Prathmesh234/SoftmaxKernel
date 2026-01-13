"""
=================================================
SOFTMAX KERNELS - HELION
=================================================
Helion softmax kernels with autotuning for H100 GPU.
Target: NVIDIA H100 SXM 80GB HBM3
"""

import torch

# Check Helion availability
try:
    import helion
    import helion.language as hl
    HELION_AVAILABLE = True
except ImportError:
    HELION_AVAILABLE = False
    print("⚠️ Helion not installed. Use: pip install git+https://github.com/pytorch/helion.git")


if HELION_AVAILABLE:
    @helion.kernel()
    def helion_softmax_simple(x: torch.Tensor) -> torch.Tensor:
        """
        Simple Helion softmax - wraps PyTorch's softmax.
        
        Helion will:
        1. Tile over the batch dimension
        2. Compile to optimized Triton code
        3. Autotune for your specific GPU and tensor shapes
        """
        n, m = x.size()
        out = torch.empty_like(x)
        for tile_n in hl.tile(n):
            out[tile_n, :] = torch.nn.functional.softmax(x[tile_n, :], dim=1)
        return out

    @helion.kernel()
    def helion_softmax_decomposed(x: torch.Tensor) -> torch.Tensor:
        """
        Decomposed Helion softmax - explicit max, exp, sum, divide.
        
        This avoids PyTorch's internal decomposition for more control.
        """
        n, m = x.size()
        out = torch.empty_like(x)
        for tile_n in hl.tile(n):
            values = x[tile_n, :]
            amax = torch.amax(values, dim=1, keepdim=True)
            exp = torch.exp(values - amax)
            sum_exp = torch.sum(exp, dim=1, keepdim=True)
            out[tile_n, :] = exp / sum_exp
        return out

    # Default export
    softmax = helion_softmax_simple
    
else:
    # Fallback when Helion is not installed
    def softmax(x: torch.Tensor) -> torch.Tensor:
        """Fallback to PyTorch when Helion is not available."""
        return torch.softmax(x, dim=-1)
