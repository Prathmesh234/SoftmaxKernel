"""
=================================================
PATCH EMBEDDING KERNELS - EAGER (PyTorch Baseline)
=================================================
Simple PyTorch implementations for baseline comparison.
Target: NVIDIA H100 SXM 80GB HBM3

PatchEmbedding converts an image [B, C, H, W] into patch tokens [B, N, D]
where N = (H/patch_size) * (W/patch_size) is the number of patches,
and D is the embedding dimension.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def naive_patch_embedding(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    patch_size: int
) -> torch.Tensor:
    """
    Naive PyTorch patch embedding showing memory inefficiency.
    
    This is intentionally NOT fused - each operation is a separate kernel.
    
    Steps:
    1. Unfold image into patches: [B, C, H, W] -> [B, C, num_patches_h, patch_size, num_patches_w, patch_size]
    2. Reshape patches: [B, num_patches, C * patch_size * patch_size]
    3. Linear projection: [B, num_patches, embed_dim]
    
    Memory operations are NOT fused, leading to extra read/write operations.
    """
    B, C, H, W = x.shape
    embed_dim = weight.shape[0]
    
    # Number of patches
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w
    
    # Step 1: Reshape to extract patches (write intermediate)
    x = x.reshape(B, C, num_patches_h, patch_size, num_patches_w, patch_size)
    
    # Step 2: Transpose and flatten patches (write intermediate)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, num_patches_h, num_patches_w, C, patch_size, patch_size]
    x = x.reshape(B, num_patches, C * patch_size * patch_size)
    
    # Step 3: Linear projection
    out = F.linear(x, weight, bias)
    
    return out


def eager_patch_embedding(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    patch_size: int
) -> torch.Tensor:
    """
    PyTorch's optimized patch embedding using Conv2d.
    
    This fuses the patch extraction and projection into a single 
    convolution operation with stride = kernel_size = patch_size.
    
    This is the standard approach used by timm and HuggingFace transformers.
    """
    B, C, H, W = x.shape
    embed_dim = weight.shape[0]
    
    # Reshape weight for conv2d: [embed_dim, C*P*P] -> [embed_dim, C, P, P]
    conv_weight = weight.view(embed_dim, C, patch_size, patch_size)
    
    # Conv2d with stride=patch_size acts as patch extraction + projection
    out = F.conv2d(x, conv_weight, bias=bias, stride=patch_size)
    
    # Reshape: [B, embed_dim, H/P, W/P] -> [B, num_patches, embed_dim]
    out = out.flatten(2).transpose(1, 2)
    
    return out


class PatchEmbeddingModule(nn.Module):
    """
    Standard Patch Embedding module for Vision Transformers.
    
    Args:
        img_size: Input image size (assumes square)
        patch_size: Size of each patch
        in_channels: Number of input channels (3 for RGB)
        embed_dim: Embedding dimension
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Projection using Conv2d
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Patch embeddings [B, num_patches, embed_dim]
        """
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


# Default export
patch_embedding = eager_patch_embedding

# TODO: Add example showing how to load CLIP weights from HuggingFace
# e.g., model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
# weights = model.vision_model.embeddings.patch_embedding.weight.data
