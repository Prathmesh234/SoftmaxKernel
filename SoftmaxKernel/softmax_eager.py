"""
=================================================
SOFTMAX KERNELS - EAGER (PyTorch Baseline)
=================================================
Simple PyTorch implementations for baseline comparison.
Target: NVIDIA H100 SXM 80GB HBM3
"""

import torch


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Naive PyTorch softmax showing memory inefficiency.
    
    This is intentionally NOT fused - each operation is a separate kernel.
    Memory operations:
    - x.max(dim=1): read MN, write M
    - x - x_max: read MN + M, write MN
    - torch.exp: read MN, write MN
    - sum: read MN, write M
    - divide: read MN + M, write MN
    
    Total: read 5MN + 2M, write 3MN + 2M (very inefficient!)
    """
    x_max = x.max(dim=1)[0]
    z = x - x_max[:, None]
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1)
    return numerator / denominator[:, None]


def eager_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    PyTorch's built-in softmax (optimized CUDA implementation).
    
    This is the baseline we compare against.
    """
    return torch.softmax(x, dim=-1)


# Default export
softmax = eager_softmax
