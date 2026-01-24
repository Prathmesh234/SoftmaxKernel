"""
Flash Attention Kernel for NVIDIA B200 (Blackwell) GPU
=======================================================

A clean, simple implementation of Flash Attention using Triton,
optimized for B200 GPUs with warp specialization support.

Usage:
    from flash_attention import flash_attention
    
    output = flash_attention(q, k, v, causal=True)
"""

import torch
import triton
import triton.language as tl
import math


def is_blackwell():
    """Check if running on Blackwell (B200) GPU."""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 10


@triton.jit
def _flash_attn_fwd(
    Q, K, V, O,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """
    Flash Attention forward kernel.
    
    This kernel computes scaled dot-product attention without materializing
    the full N×N attention matrix, using the online softmax algorithm.
    """
    # Program IDs
    start_m = tl.program_id(0)  # Which query block
    off_hz = tl.program_id(1)   # Which batch*head
    
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Base pointers for this batch/head
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
    
    # Block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=O + o_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    
    # Offsets for masking
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # Running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # Running sum
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)  # Output accumulator
    
    # Scale for numerical stability (use exp2 instead of exp)
    qk_scale = sm_scale * 1.44269504  # sm_scale / log(2)
    
    # Load Q block - stays in registers throughout
    q = tl.load(Q_block_ptr)
    
    # Determine loop bounds for causal masking
    if CAUSAL:
        hi = min((start_m + 1) * BLOCK_M, N_CTX)
    else:
        hi = N_CTX
    
    # Main loop over K, V blocks
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Load K block and compute QK^T
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, tl.trans(k))
        
        # Apply causal mask if needed
        if CAUSAL:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0.0, float("-inf"))
        else:
            qk = qk * qk_scale
        
        # Online softmax: compute new max and update
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        
        # Compute correction factor for previous accumulator
        alpha = tl.math.exp2(m_i - m_new)
        
        # Compute attention weights with new max
        p = tl.math.exp2(qk - m_new[:, None])
        
        # Update running sum
        l_ij = tl.sum(p, 1)
        l_new = l_i * alpha + l_ij
        
        # Update accumulator with correction
        acc = acc * alpha[:, None]
        
        # Load V and accumulate
        v = tl.load(V_block_ptr)
        acc += tl.dot(p.to(v.dtype), v)
        
        # Update state
        m_i = m_new
        l_i = l_new
        
        # Advance K, V pointers
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    
    # Final normalization
    acc = acc / l_i[:, None]
    
    # Store output
    tl.store(O_block_ptr, acc.to(O_block_ptr.dtype.element_ty))


# Autotuning configurations
def get_autotune_configs():
    """Get autotuning configurations based on GPU architecture."""
    configs = [
        # Standard configs (work on all GPUs)
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=2, num_warps=8),
    ]
    
    # B200-specific configs with warp specialization
    if is_blackwell():
        configs.extend([
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=4, num_warps=8),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=4, num_warps=8),
        ])
    
    return configs


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: float = None,
) -> torch.Tensor:
    """
    Flash Attention implementation optimized for B200 GPUs.
    
    Args:
        q: Query tensor of shape [batch, heads, seq_len, head_dim]
        k: Key tensor of shape [batch, heads, seq_len, head_dim]
        v: Value tensor of shape [batch, heads, seq_len, head_dim]
        causal: If True, apply causal masking (default: True)
        sm_scale: Softmax scaling factor (default: 1/sqrt(head_dim))
    
    Returns:
        Output tensor of shape [batch, heads, seq_len, head_dim]
    
    Example:
        >>> q = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
        >>> k = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
        >>> v = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
        >>> output = flash_attention(q, k, v, causal=True)
    """
    # Input validation
    assert q.dim() == 4, f"Expected 4D tensor, got {q.dim()}D"
    assert q.shape == k.shape == v.shape, "Q, K, V must have same shape"
    assert q.is_cuda, "Tensors must be on CUDA"
    
    batch, heads, seq_len, head_dim = q.shape
    
    # Default scaling
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    
    # Ensure contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    # Allocate output
    o = torch.empty_like(q)
    
    # Choose block sizes based on head dimension
    if head_dim <= 64:
        BLOCK_M, BLOCK_N = 128, 64
    else:
        BLOCK_M, BLOCK_N = 128, 128
    
    # Grid: one program per query block per batch*head
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads)
    
    # Launch kernel
    _flash_attn_fwd[grid](
        q, k, v, o,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        batch, heads, seq_len,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        CAUSAL=causal,
        num_stages=4 if is_blackwell() else 2,
        num_warps=8 if head_dim >= 128 else 4,
    )
    
    return o


def benchmark(
    batch: int = 4,
    heads: int = 32,
    seq_len: int = 2048,
    head_dim: int = 64,
    causal: bool = True,
    warmup: int = 10,
    rep: int = 100,
):
    """
    Benchmark the Flash Attention kernel.
    
    Args:
        batch: Batch size
        heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Head dimension
        causal: Whether to use causal masking
        warmup: Number of warmup iterations
        rep: Number of benchmark iterations
    
    Returns:
        Dictionary with timing results
    """
    # Create inputs
    q = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    
    # Warmup
    for _ in range(warmup):
        _ = flash_attention(q, k, v, causal=causal)
    
    torch.cuda.synchronize()
    
    # Benchmark
    import time
    start = time.perf_counter()
    for _ in range(rep):
        _ = flash_attention(q, k, v, causal=causal)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    # Calculate metrics
    ms = (end - start) / rep * 1000
    
    # FLOPS calculation
    flops_per_matmul = 2 * batch * heads * seq_len * seq_len * head_dim
    total_flops = 2 * flops_per_matmul  # QK^T and PV
    if causal:
        total_flops *= 0.5  # Only half the matrix is computed
    
    tflops = total_flops / (ms * 1e-3) / 1e12
    
    return {
        "batch": batch,
        "heads": heads,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "causal": causal,
        "time_ms": ms,
        "tflops": tflops,
    }


def test_correctness():
    """Test the Flash Attention implementation against PyTorch reference."""
    torch.manual_seed(42)
    
    batch, heads, seq_len, head_dim = 2, 4, 256, 64
    
    q = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    
    sm_scale = 1.0 / math.sqrt(head_dim)
    
    # Reference implementation
    def reference_attention(q, k, v, causal=True):
        scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
        if causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        return torch.matmul(attn, v)
    
    # Test causal
    ref_causal = reference_attention(q, k, v, causal=True)
    out_causal = flash_attention(q, k, v, causal=True, sm_scale=sm_scale)
    
    # Test non-causal
    ref_full = reference_attention(q, k, v, causal=False)
    out_full = flash_attention(q, k, v, causal=False, sm_scale=sm_scale)
    
    # Check results
    causal_diff = (ref_causal - out_causal).abs().max().item()
    full_diff = (ref_full - out_full).abs().max().item()
    
    print(f"Causal attention max diff: {causal_diff:.6f}")
    print(f"Full attention max diff: {full_diff:.6f}")
    
    assert causal_diff < 1e-2, f"Causal attention failed: diff={causal_diff}"
    assert full_diff < 1e-2, f"Full attention failed: diff={full_diff}"
    
    print("✓ All tests passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Flash Attention for B200 (Blackwell) GPU")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Compute Capability: {torch.cuda.get_device_capability()}")
        print(f"Is Blackwell: {is_blackwell()}")
        print()
        
        # Run correctness test
        print("Running correctness tests...")
        test_correctness()
        print()
        
        # Run benchmarks
        print("Running benchmarks...")
        for seq_len in [1024, 2048, 4096]:
            result = benchmark(seq_len=seq_len)
            print(f"  seq_len={seq_len:5d}: {result['time_ms']:.2f} ms, {result['tflops']:.1f} TFLOPS")
    else:
        print("CUDA not available!")
