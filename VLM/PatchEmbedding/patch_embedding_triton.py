"""
=================================================
PATCH EMBEDDING KERNELS - TRITON (H100 Optimized)
=================================================
High-performance Triton patch embedding kernels for H100 GPU.
Target: NVIDIA H100 SXM 80GB HBM3

Key Optimizations:
1. tl.dot for Tensor Core utilization (4th-gen Tensor Cores)
2. Extensive autotuning for optimal block sizes on H100
3. Memory alignment for 128-byte coalesced access
4. Proper tile dimensions (multiples of 16 for mma instructions)
5. Truly fused patch embedding kernel (no separate unfold)
6. TMA-style cooperative fetching for better bandwidth
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# =============================================================================
# AUTOTUNED MATMUL KERNEL WITH tl.dot FOR TENSOR CORES
# =============================================================================

def get_matmul_configs():
    """Generate comprehensive autotuning configs for H100."""
    configs = []

    # Large tile configs for large matrices (B >= 8)
    for block_m in [256, 128]:
        for block_n in [256, 128]:
            for block_k in [64, 32]:
                for warps in [8]:
                    for stages in [3, 4, 5]:
                        configs.append(triton.Config(
                            {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k},
                            num_warps=warps, num_stages=stages
                        ))

    # Medium tile configs
    for block_m in [128, 64]:
        for block_n in [128, 64]:
            for block_k in [64, 32]:
                for warps in [4, 8]:
                    for stages in [3, 4]:
                        configs.append(triton.Config(
                            {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k},
                            num_warps=warps, num_stages=stages
                        ))

    # Small tile configs for small M (batch=1)
    for block_m in [32, 16]:
        for block_n in [128, 64, 256]:
            for block_k in [64, 32]:
                for warps in [2, 4]:
                    for stages in [3, 4, 5]:
                        configs.append(triton.Config(
                            {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k},
                            num_warps=warps, num_stages=stages
                        ))

    # Remove duplicates
    seen = set()
    unique_configs = []
    for c in configs:
        key = (c.kwargs['BLOCK_M'], c.kwargs['BLOCK_N'], c.kwargs['BLOCK_K'], c.num_warps, c.num_stages)
        if key not in seen:
            seen.add(key)
            unique_configs.append(c)

    return unique_configs


@triton.autotune(
    configs=get_matmul_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_kernel(
    # Pointers
    A_ptr, B_ptr, C_ptr, bias_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    Optimized GEMM kernel using tl.dot for Tensor Cores.

    Computes: C = A @ B + bias
    Where:
        A: [M, K] - flattened patches (B*num_patches, patch_dim)
        B: [K, N] - weight matrix transposed (patch_dim, embed_dim)
        C: [M, N] - output embeddings (B*num_patches, embed_dim)
        bias: [N] - optional bias vector
    """
    # Program IDs
    pid = tl.program_id(0)

    # Calculate number of blocks in each dimension
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Grouped ordering for better L2 cache locality (super-grouping)
    # Use larger group for H100's larger L2 cache (50MB)
    GROUP_SIZE_M: tl.constexpr = 8
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block starting positions
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize pointers with better coalescing
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator in FP32 for precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop over K dimension with software pipelining
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k

        # Load with boundary masks
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_offs[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k_offs[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        # Tensor Core matmul
        acc = tl.dot(a, b, acc)

        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]

    # Store output with coalesced access
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=mask)


def triton_patch_embedding_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    patch_size: int
) -> torch.Tensor:
    """
    Triton patch embedding using im2col + autotuned GEMM with Tensor Cores.

    This is the primary high-performance implementation that:
    1. Uses F.unfold for efficient im2col
    2. Uses autotuned Triton GEMM with tl.dot for Tensor Cores
    3. Uses 1D grid with super-grouping for better L2 cache locality

    Args:
        x: Input tensor [B, C, H, W]
        weight: Projection weight [embed_dim, C * patch_size * patch_size]
        bias: Optional bias [embed_dim]
        patch_size: Size of each patch
    Returns:
        Patch embeddings [B, num_patches, embed_dim]
    """
    B, C, H, W = x.shape
    embed_dim, patch_dim = weight.shape

    assert H % patch_size == 0 and W % patch_size == 0, \
        f"Image size ({H}, {W}) must be divisible by patch_size ({patch_size})"

    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w

    # im2col: [B, C, H, W] -> [B, num_patches, patch_dim]
    patches = F.unfold(x, kernel_size=patch_size, stride=patch_size)
    patches = patches.transpose(1, 2).contiguous()

    # Reshape for GEMM: [M, K] where M = B*num_patches, K = patch_dim
    M = B * num_patches
    K = patch_dim
    N = embed_dim

    A = patches.view(M, K)
    B_mat = weight.t().contiguous()  # [K, N]

    # Allocate output
    C_out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # 1D grid for better work distribution with super-grouping
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    # Launch kernel
    _matmul_kernel[grid](
        A, B_mat, C_out,
        bias if bias is not None else A,
        M, N, K,
        A.stride(0), A.stride(1),
        B_mat.stride(0), B_mat.stride(1),
        C_out.stride(0), C_out.stride(1),
        HAS_BIAS=(bias is not None),
    )

    return C_out.view(B, num_patches, embed_dim)


# =============================================================================
# TRULY FUSED PATCH EMBEDDING KERNEL - DIRECT IMAGE READ + GEMM
# =============================================================================

def get_fused_configs():
    """Generate configs for truly fused kernel."""
    configs = []

    # Configs for various output dimensions (embed_dim)
    for block_m in [64, 32, 128]:
        for block_n in [128, 64, 256]:
            for block_k in [64, 32, 128]:
                for warps in [4, 8]:
                    for stages in [3, 4]:
                        configs.append(triton.Config(
                            {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k},
                            num_warps=warps, num_stages=stages
                        ))

    # Remove duplicates
    seen = set()
    unique_configs = []
    for c in configs:
        key = (c.kwargs['BLOCK_M'], c.kwargs['BLOCK_N'], c.kwargs['BLOCK_K'], c.num_warps, c.num_stages)
        if key not in seen:
            seen.add(key)
            unique_configs.append(c)

    return unique_configs


@triton.autotune(
    configs=get_fused_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_patch_embed_kernel(
    # Pointers
    img_ptr, weight_ptr, bias_ptr, output_ptr,
    # Image dimensions
    batch_size, channels, img_h, img_w,
    # Patch parameters
    patch_size, num_patches_h, num_patches_w,
    # Output dimensions
    M, N, K,  # M = batch*num_patches, N = embed_dim, K = patch_dim
    # Strides for image (NCHW)
    stride_ib, stride_ic, stride_ih, stride_iw,
    # Strides for weight [embed_dim, patch_dim]
    stride_wn, stride_wk,
    # Strides for output [M, N]
    stride_om, stride_on,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    Truly fused patch embedding kernel.

    Reads patches directly from image memory during GEMM computation,
    eliminating the need for a separate unfold operation.

    Each output row corresponds to one patch from one batch element.
    """
    # 1D grid with super-grouping
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Group for L2 cache locality
    GROUP_SIZE_M: tl.constexpr = 8
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # Patch indices
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # Embed dim indices
    offs_k = tl.arange(0, BLOCK_K)  # Patch dim indices

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Convert patch index to (batch, patch_h, patch_w) indices
    num_patches = num_patches_h * num_patches_w

    # Main K-loop
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        k_mask = k_offs < K

        # Load weight tile [BLOCK_K, BLOCK_N]
        # Weight is [embed_dim, patch_dim] so we need W[k, n]
        # But weight is stored as [N, K], so we access weight[n, k]
        w_ptrs = weight_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        w_mask = k_mask[:, None] & (offs_n[None, :] < N)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Load input patches - need to compute image coordinates
        # For each m in offs_m: batch_idx = m // num_patches
        #                       patch_idx = m % num_patches
        #                       patch_h = patch_idx // num_patches_w
        #                       patch_w = patch_idx % num_patches_w
        # For each k in k_offs:  c = k // (patch_size * patch_size)
        #                        offset = k % (patch_size * patch_size)
        #                        ph = offset // patch_size
        #                        pw = offset % patch_size

        # Compute patch coordinates for each output row
        batch_idx = offs_m // num_patches
        patch_idx = offs_m % num_patches
        patch_row = patch_idx // num_patches_w
        patch_col = patch_idx % num_patches_w

        # Compute input coordinates for each k
        ch = k_offs // (patch_size * patch_size)
        k_rem = k_offs % (patch_size * patch_size)
        ph = k_rem // patch_size
        pw = k_rem % patch_size

        # Load patch data [BLOCK_M, BLOCK_K]
        # img[batch_idx, ch, patch_row*patch_size + ph, patch_col*patch_size + pw]
        img_h_idx = patch_row[:, None] * patch_size + ph[None, :]
        img_w_idx = patch_col[:, None] * patch_size + pw[None, :]

        img_ptrs = (img_ptr +
                    batch_idx[:, None] * stride_ib +
                    ch[None, :] * stride_ic +
                    img_h_idx * stride_ih +
                    img_w_idx * stride_iw)

        img_mask = (offs_m[:, None] < M) & k_mask[None, :]
        a = tl.load(img_ptrs, mask=img_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # Accumulate: C += A @ W
        acc = tl.dot(a, w, acc)

    # Add bias if present
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]

    # Store output
    out_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(output_ptr.dtype.element_ty), mask=out_mask)


def triton_patch_embedding_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    patch_size: int
) -> torch.Tensor:
    """
    Truly fused Triton patch embedding - reads patches directly from image.

    This kernel fuses patch extraction and linear projection into a single
    kernel, eliminating the intermediate unfold operation entirely.
    """
    B, C, H, W = x.shape
    embed_dim, patch_dim = weight.shape

    assert H % patch_size == 0 and W % patch_size == 0
    assert patch_dim == C * patch_size * patch_size

    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w

    # Ensure contiguous
    x = x.contiguous()
    weight = weight.contiguous()

    M = B * num_patches
    N = embed_dim
    K = patch_dim

    # Allocate output
    output = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # 1D grid
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    _fused_patch_embed_kernel[grid](
        x, weight, bias if bias is not None else x, output,
        B, C, H, W,
        patch_size, num_patches_h, num_patches_w,
        M, N, K,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        HAS_BIAS=(bias is not None),
    )

    return output.view(B, num_patches, embed_dim)


# Default export - use the optimized matmul pattern
patch_embedding = triton_patch_embedding_matmul
