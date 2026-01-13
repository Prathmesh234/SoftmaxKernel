"""
=================================================
PATCH EMBEDDING KERNELS - HELION (OPTIMIZED)
=================================================
Optimized Helion patch embedding kernel for H100 GPU.
Target: NVIDIA H100 SXM 80GB HBM3

Uses the proven addmm pattern with optimized config from autotuning.
"""

import torch
import torch.nn.functional as F
from typing import Callable

# Check Helion availability
try:
    import helion
    import helion.language as hl
    HELION_AVAILABLE = True
except ImportError:
    HELION_AVAILABLE = False
    print("⚠️ Helion not installed. Use: pip install git+https://github.com/pytorch/helion.git")


if HELION_AVAILABLE:
    # Optimized kernel with best config from autotuning on H100
    @helion.kernel(
        config=helion.Config(
            block_sizes=[64, 128, 128],
            indexing=['pointer', 'pointer', 'tensor_descriptor', 'pointer'],
            l2_groupings=[32],
            load_eviction_policies=['', 'last', 'last'],
            loop_orders=[[0, 1]],
            num_stages=1,
            num_warps=4,
            pid_type='flat',
            range_flattens=[None, None],
            range_multi_buffers=[None, False],
            range_num_stages=[0, 3],
            range_unroll_factors=[0, 1],
            range_warp_specializes=[]
        ),
        static_shapes=True
    )
    def helion_linear_optimized(
        x: torch.Tensor,
        weight: torch.Tensor,
        epilogue: Callable[[torch.Tensor, tuple[torch.Tensor, ...]], torch.Tensor] = lambda acc, tile: acc,
    ) -> torch.Tensor:
        """
        Optimized Helion linear projection kernel.
        
        Performs: out = epilogue(x @ weight.T)
        
        Uses the best configuration found via autotuning on H100:
        - block_sizes=[64, 128, 128] (tile_m=64, tile_n=128, tile_k=128)
        - 4 warps, 1 pipeline stage
        - tensor_descriptor indexing for weights
        
        Args:
            x: Input [M, K] (flattened patches)
            weight: Weight [N, K]
            epilogue: Post-processing function for bias
        Returns:
            Output [M, N]
        """
        m, k = x.size()
        n, k2 = weight.size()
        assert k == k2, f"size mismatch {k} != {k2}"
        
        out = torch.empty([m, n], dtype=x.dtype, device=x.device)
        
        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, x[tile_m, tile_k], weight[tile_n, tile_k].T)
            out[tile_m, tile_n] = epilogue(acc, (tile_m, tile_n))
        
        return out

    # Autotuning version for finding configs on new hardware
    @helion.kernel(static_shapes=True)
    def helion_linear_autotune(
        x: torch.Tensor,
        weight: torch.Tensor,
        epilogue: Callable[[torch.Tensor, tuple[torch.Tensor, ...]], torch.Tensor] = lambda acc, tile: acc,
    ) -> torch.Tensor:
        """
        Helion linear projection kernel (autotuning version).
        
        Use this to find optimal configs for new hardware.
        Warning: Autotuning takes 5-10 minutes!
        """
        m, k = x.size()
        n, k2 = weight.size()
        assert k == k2, f"size mismatch {k} != {k2}"
        
        out = torch.empty([m, n], dtype=x.dtype, device=x.device)
        
        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, x[tile_m, tile_k], weight[tile_n, tile_k].T)
            out[tile_m, tile_n] = epilogue(acc, (tile_m, tile_n))
        
        return out


def _make_bias_epilogue(bias: torch.Tensor):
    """Create a bias epilogue function that captures the bias tensor."""
    def epilogue(acc: torch.Tensor, tile: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return acc + bias[tile[1]]
    return epilogue


def helion_patch_embedding(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    patch_size: int,
    use_autotune: bool = False
) -> torch.Tensor:
    """
    Optimized Helion patch embedding.
    
    Args:
        x: Input tensor [B, C, H, W]
        weight: Projection weight [embed_dim, C * patch_size * patch_size]
        bias: Optional bias [embed_dim]
        patch_size: Size of each patch
        use_autotune: If True, runs full autotuning (slow)
    Returns:
        Patch embeddings [B, num_patches, embed_dim]
    """
    B, C, H, W = x.shape
    embed_dim, patch_dim = weight.shape
    
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w
    
    # Unfold: [B, C, H, W] -> [B, C*P*P, num_patches] -> [B*num_patches, patch_dim]
    patches = F.unfold(x, kernel_size=patch_size, stride=patch_size)
    patches_2d = patches.transpose(1, 2).reshape(-1, patch_dim)
    
    # Select kernel
    kernel_fn = helion_linear_autotune if use_autotune else helion_linear_optimized
    
    # Apply kernel with optional bias epilogue
    if bias is not None:
        epilogue = _make_bias_epilogue(bias)
        out_2d = kernel_fn(patches_2d, weight, epilogue)
    else:
        out_2d = kernel_fn(patches_2d, weight)
    
    # Reshape: [B * num_patches, embed_dim] -> [B, num_patches, embed_dim]
    out = out_2d.view(B, num_patches, embed_dim)
    
    return out


# Export
if HELION_AVAILABLE:
    patch_embedding = helion_patch_embedding
else:
    def patch_embedding(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        patch_size: int
    ) -> torch.Tensor:
        """Fallback to PyTorch when Helion is not available."""
        B, C, H, W = x.shape
        embed_dim = weight.shape[0]
        
        conv_weight = weight.view(embed_dim, C, patch_size, patch_size)
        out = F.conv2d(x, conv_weight, bias=bias, stride=patch_size)
        out = out.flatten(2).transpose(1, 2)
        return out
