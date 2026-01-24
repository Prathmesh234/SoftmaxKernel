# Triton Attention Kernel Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [What is Triton?](#what-is-triton)
3. [What is Helion?](#what-is-helion)
4. [**NVIDIA B200 Blackwell Optimizations**](#nvidia-b200-blackwell-optimizations) ⚡ NEW
5. [GPU Memory Hierarchy](#gpu-memory-hierarchy)
6. [Flash Attention Algorithm](#flash-attention-algorithm)
7. [Core Concepts for Attention Kernels](#core-concepts-for-attention-kernels)
8. [Triton Attention Implementation](#triton-attention-implementation)
9. [Helion-Based Attention Implementation](#helion-based-attention-implementation)
10. [Autotuning](#autotuning)
11. [Installation & Requirements](#installation--requirements)
12. [Complete Code Examples](#complete-code-examples)
13. [Performance Benchmarks](#performance-benchmarks)
14. [Best Practices](#best-practices)
15. [References](#references)

---

## Introduction

This document provides comprehensive documentation for implementing optimized attention kernels using **Triton** and **Helion**. Attention mechanisms are the backbone of modern transformer architectures, but they are notoriously memory-intensive due to the O(N²) complexity of the attention matrix. Optimized kernels like Flash Attention use clever algorithmic tricks to reduce memory usage while maintaining computational efficiency.

### Why Custom Attention Kernels?

Standard attention computations suffer from memory bottlenecks because they materialize a large N × N attention score matrix, where N is the sequence length. For modern Large Language Models (LLMs), N can be thousands or even millions, making this matrix too large to fit in fast on-chip SRAM, forcing frequent data transfers to slower HBM (High Bandwidth Memory).

---

## What is Triton?

**Triton** is an open-source, Python-like programming language and compiler designed by OpenAI to simplify GPU programming while achieving performance on par with or exceeding hand-written CUDA kernels.

### Key Features

| Feature | Description |
|---------|-------------|
| **Block-Level Programming** | Work is scheduled in blocks rather than individual threads, abstracting away GPU thread management |
| **Automatic Memory Optimization** | Triton handles memory coalescing, shared memory allocation, and synchronization |
| **Python-Like Syntax** | Familiar syntax for Python developers, reducing the learning curve |
| **Automatic Vectorization** | Operations on blocks are automatically vectorized for efficiency |
| **Cross-Platform** | Works on NVIDIA, AMD, and other GPU architectures |

### Triton Programming Model

```python
import triton
import triton.language as tl

@triton.jit
def example_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance processes a block of elements
    pid = tl.program_id(axis=0)
    
    # Calculate the starting offset for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for boundary handling
    mask = offsets < n_elements
    
    # Load data from global memory to registers
    data = tl.load(input_ptr + offsets, mask=mask)
    
    # Perform computation
    result = data * 2.0
    
    # Store results back to global memory
    tl.store(output_ptr + offsets, result, mask=mask)
```

### Core Triton Language Operations

| Operation | Description | Example |
|-----------|-------------|---------|
| `tl.program_id(axis)` | Get the current program/block ID | `pid = tl.program_id(0)` |
| `tl.arange(start, end)` | Create a range of values | `offs = tl.arange(0, 128)` |
| `tl.load(ptr, mask)` | Load data from memory | `data = tl.load(ptr + offs, mask=mask)` |
| `tl.store(ptr, value, mask)` | Store data to memory | `tl.store(ptr + offs, val, mask=mask)` |
| `tl.dot(a, b)` | Matrix multiplication (uses Tensor Cores) | `c = tl.dot(a, b)` |
| `tl.sum(x, axis)` | Reduction sum | `total = tl.sum(x, axis=1)` |
| `tl.max(x, axis)` | Reduction max | `maximum = tl.max(x, axis=1)` |
| `tl.exp(x)` | Elementwise exponential | `exp_x = tl.exp(x)` |
| `tl.where(cond, a, b)` | Conditional selection | `result = tl.where(mask, a, b)` |

---

## What is Helion?

**Helion** is a Python-embedded domain-specific language (DSL) developed by the Meta PyTorch Team for authoring machine learning kernels. It compiles down to Triton, providing a higher level of abstraction while maintaining high performance.

### Helion vs. Triton

| Aspect | Triton | Helion |
|--------|--------|--------|
| **Abstraction Level** | Low-level block programming | High-level PyTorch-like |
| **Syntax** | Custom DSL with `@triton.jit` | Standard PyTorch operations |
| **Tiling** | Manual specification | Automatic with `hl.tile()` |
| **Autotuning** | Manual config definition | Extensive automatic autotuning |
| **Learning Curve** | Moderate | Lower (if you know PyTorch) |

### Helion Key Features

1. **Familiar PyTorch Syntax**: Use standard PyTorch operators like `torch.addmm`, `torch.softmax`
2. **Automatic Tiling**: The `hl.tile()` function automatically subdivides iteration space
3. **Extensive Autotuning**: Evaluates hundreds of configurations to find optimal performance
4. **TorchInductor Integration**: Leverages PyTorch 2's compilation infrastructure

### Basic Helion Example

```python
import torch
import helion
import helion.language as hl

@helion.kernel()
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    
    # Standard PyTorch code for output allocation
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    
    # Tiled execution - compiled to Triton
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    
    return out
```

---

## NVIDIA B200 Blackwell Optimizations

The **NVIDIA B200** GPU, built on the **Blackwell architecture** (CUDA Compute Capability 10.0 / SM100), introduces groundbreaking features specifically designed to accelerate attention mechanisms and transformer workloads. This section covers how to optimize your Triton kernels for maximum performance on B200 GPUs.

### B200 GPU Specifications

| Specification | Value |
|--------------|-------|
| **Architecture** | Blackwell (SM100) |
| **CUDA Compute Capability** | 10.0 |
| **Transistors** | 208 billion (dual-die) |
| **Memory** | 192 GB HBM3e |
| **Memory Bandwidth** | 8 TB/s |
| **NVLink Bandwidth** | 1.8 TB/s |
| **FP8 Tensor TFLOPS** | 4,500 (dense) / 9,000 (sparse) |
| **FP16/BF16 Tensor TFLOPS** | 2,250 (dense) / 4,500 (sparse) |
| **TDP** | Up to 1000W |

### Key Blackwell Features for Attention Kernels

#### 1. Tensor Memory (TMEM)

Blackwell introduces **Tensor Memory (TMEM)**, a dedicated on-chip memory positioned closer to the 5th-generation Tensor Cores. TMEM replaces registers for UMMA (Unified Matrix Multiply-Accumulate) instructions.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Blackwell Memory Hierarchy                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Global Memory (HBM3e)                     │ │
│  │              Size: 192 GB, Bandwidth: 8 TB/s                 │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              ▲                                    │
│                              │                                    │
│                              ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                   Shared Memory (SMEM)                       │ │
│  │              Size: 228 KB per SM                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              ▲                                    │
│                              │                                    │
│                              ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │               ⚡ Tensor Memory (TMEM) ⚡                      │ │
│  │              Size: 256 KB per SM                             │ │
│  │              Direct connection to Tensor Cores               │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              ▲                                    │
│                              │                                    │
│                              ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              5th Gen Tensor Cores (UMMA)                     │ │
│  │              Larger tile sizes, async output to TMEM         │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

#### 2. Tensor Memory Accelerator (TMA)

TMA is a dedicated hardware unit for efficient asynchronous tensor data movement:

```python
# Using TMA with Triton tensor descriptors
from triton.tools.tensor_descriptor import TensorDescriptor

def create_tma_descriptors(q, k, v, o, block_m, block_n, head_dim):
    """Create TMA-backed tensor descriptors for B200."""
    batch, heads, seq_len, d = q.shape
    y_dim = batch * heads * seq_len
    
    # Create tensor descriptors for TMA
    desc_q = TensorDescriptor(
        q, 
        shape=[y_dim, head_dim],
        strides=[head_dim, 1],
        block_shape=[block_m, head_dim]
    )
    desc_k = TensorDescriptor(
        k,
        shape=[y_dim, head_dim],
        strides=[head_dim, 1],
        block_shape=[block_n, head_dim]
    )
    desc_v = TensorDescriptor(
        v,
        shape=[y_dim, head_dim],
        strides=[head_dim, 1],
        block_shape=[block_n, head_dim]
    )
    desc_o = TensorDescriptor(
        o,
        shape=[y_dim, head_dim],
        strides=[head_dim, 1],
        block_shape=[block_m, head_dim]
    )
    
    return desc_q, desc_k, desc_v, desc_o
```

#### 3. Warp Specialization

Warp specialization is a **critical optimization for Blackwell** that assigns distinct roles to warps within a threadblock:

- **Producer Warps**: Handle memory fetches (TMA operations)
- **Consumer Warps**: Handle computation (Tensor Core operations)
- **Epilogue Warps**: Handle output writes

```python
import triton
import triton.language as tl

def is_blackwell():
    """Check if running on Blackwell GPU."""
    import torch
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10

# Blackwell-optimized configs with warp specialization
BLACKWELL_CONFIGS = [
    triton.Config(
        {'BLOCK_M': 128, 'BLOCK_N': 128}, 
        num_stages=4, 
        num_warps=8,
        num_consumer_groups=2,      # Enable warp specialization
        num_buffers_warp_spec=2,    # Shared memory buffers for warp communication
    ),
    triton.Config(
        {'BLOCK_M': 128, 'BLOCK_N': 64}, 
        num_stages=3, 
        num_warps=8,
        num_consumer_groups=2,
        num_buffers_warp_spec=2,
    ),
    triton.Config(
        {'BLOCK_M': 64, 'BLOCK_N': 128}, 
        num_stages=4, 
        num_warps=4,
        num_consumer_groups=1,
        num_buffers_warp_spec=2,
    ),
]

@triton.autotune(
    configs=BLACKWELL_CONFIGS if is_blackwell() else [],
    key=['N_CTX', 'HEAD_DIM'],
)
@triton.jit
def _attn_fwd_blackwell(
    # ... kernel parameters
    warp_specialize: tl.constexpr,  # Enable warp specialization
    IS_BLACKWELL: tl.constexpr,     # Blackwell-specific optimizations
):
    # Kernel implementation with warp specialization
    pass
```

#### 4. Enabling Warp Specialization in Attention

```python
@triton.jit
def _attn_fwd_inner_blackwell(
    acc, l_i, m_i, q,
    desc_k, desc_v,
    offset_y, dtype: tl.constexpr,
    start_m, qk_scale,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    N_CTX: tl.constexpr,
    warp_specialize: tl.constexpr,  # ⚡ Blackwell warp specialization
    IS_BLACKWELL: tl.constexpr,
):
    """Blackwell-optimized inner attention loop with warp specialization."""
    
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX
    
    offsetk_y = offset_y + lo
    offsetv_y = offset_y + lo
    
    # ⚡ Loop with warp specialization enabled
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Load K using TMA (producer warps handle this)
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)
        
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        
        # ⚡ Blackwell-specific accumulator reshape for optimal TMEM usage
        if IS_BLACKWELL and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
        else:
            acc = acc * alpha[:, None]
        
        # Load V using TMA
        v = desc_v.load([offsetv_y, 0])
        p = p.to(dtype)
        acc = tl.dot(p, v, acc)
        
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    
    return acc, l_i, m_i
```

### FlashAttention-4 for Blackwell

**FlashAttention-4 (FA4)** is specifically designed for Blackwell GPUs, achieving up to **1,605 TFLOPS/s** (71% of theoretical peak):

| Feature | FlashAttention-2 | FlashAttention-4 (B200) |
|---------|------------------|-------------------------|
| **Peak TFLOPS** | ~300 (A100) | **1,605 (B200)** |
| **Hardware Utilization** | ~60% | **71%** |
| **Memory Usage** | Shared Memory | **TMEM + SMEM** |
| **Warp Specialization** | No | **Yes** |
| **TMA Support** | Limited | **Full** |

### Complete B200-Optimized Attention Kernel

```python
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

def is_blackwell():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10

def supports_tma():
    """Check if GPU supports TMA (Hopper+ with SM >= 9.0)."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9

# ⚡ B200 Blackwell-specific configurations
def get_blackwell_configs():
    if not is_blackwell():
        return []
    
    configs = []
    for block_m in [64, 128]:
        for block_n in [64, 128]:
            for num_stages in [3, 4]:
                for num_warps in [4, 8]:
                    # Enable warp specialization for B200
                    configs.append(
                        triton.Config(
                            {'BLOCK_M': block_m, 'BLOCK_N': block_n},
                            num_stages=num_stages,
                            num_warps=num_warps,
                            num_consumer_groups=2,
                            num_buffers_warp_spec=2,
                        )
                    )
    return configs


def _pre_hook_blackwell(nargs):
    """Pre-hook to configure tensor descriptors for B200."""
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    
    nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M, HEAD_DIM]


# Standard configs for non-Blackwell GPUs
STANDARD_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
]


@triton.autotune(
    configs=get_blackwell_configs() + STANDARD_CONFIGS if is_blackwell() else STANDARD_CONFIGS,
    key=['N_CTX', 'HEAD_DIM', 'warp_specialize'],
)
@triton.jit
def flash_attention_fwd_b200(
    sm_scale,
    M,  # LSE output
    Z, H,  # Batch, heads
    desc_q, desc_k, desc_v, desc_o,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    STAGE: tl.constexpr,
    warp_specialize: tl.constexpr,
    IS_BLACKWELL: tl.constexpr,
):
    """
    FlashAttention forward kernel optimized for NVIDIA B200 Blackwell GPUs.
    
    Key B200 optimizations:
    - Warp specialization for producer/consumer separation
    - TMA-backed tensor descriptors for async memory access
    - TMEM utilization for accumulator storage
    - Larger block sizes for 5th gen Tensor Cores
    """
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    
    y_dim = Z * H * N_CTX
    
    # Create tensor descriptors (uses TMA on B200)
    desc_q = tl.make_tensor_descriptor(
        desc_q, 
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM]
    ) if not isinstance(desc_q, tl.tensor_descriptor) else desc_q
    
    desc_k = tl.make_tensor_descriptor(
        desc_k,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM]
    ) if not isinstance(desc_k, tl.tensor_descriptor) else desc_k
    
    desc_v = tl.make_tensor_descriptor(
        desc_v,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM]
    ) if not isinstance(desc_v, tl.tensor_descriptor) else desc_v
    
    desc_o = tl.make_tensor_descriptor(
        desc_o,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM]
    ) if not isinstance(desc_o, tl.tensor_descriptor) else desc_o
    
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    qk_scale = sm_scale * 1.44269504  # 1/log(2)
    
    # Load Q (stays in SRAM/TMEM)
    q = desc_q.load([qo_offset_y, 0])
    
    # Stage 1: Off-diagonal blocks
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner_blackwell(
            acc, l_i, m_i, q,
            desc_k, desc_v,
            offset_y, dtype,
            start_m, qk_scale,
            BLOCK_M, HEAD_DIM, BLOCK_N,
            4 - STAGE,
            offs_m, offs_n, N_CTX,
            warp_specialize, IS_BLACKWELL,
        )
    
    # Stage 2: Diagonal blocks (causal masking)
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner_blackwell(
            acc, l_i, m_i, q,
            desc_k, desc_v,
            offset_y, dtype,
            start_m, qk_scale,
            BLOCK_M, HEAD_DIM, BLOCK_N,
            2,
            offs_m, offs_n, N_CTX,
            warp_specialize, IS_BLACKWELL,
        )
    
    # Epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


class FlashAttentionB200(torch.autograd.Function):
    """PyTorch autograd wrapper for B200-optimized Flash Attention."""
    
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, warp_specialize=True):
        HEAD_DIM = q.shape[-1]
        assert HEAD_DIM in {64, 128, 256}, f"HEAD_DIM must be 64, 128, or 256, got {HEAD_DIM}"
        
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        
        M = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]),
            device=q.device,
            dtype=torch.float32
        )
        
        # Use TMA tensor descriptors on B200
        if supports_tma():
            y_dim = q.shape[0] * q.shape[1] * q.shape[2]
            dummy_block = [1, 1]
            
            desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
            desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
            desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
            desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=dummy_block)
        else:
            desc_q, desc_k, desc_v, desc_o = q, k, v, o
        
        # B200-specific kernel arguments
        extra_args = {}
        if is_blackwell() and warp_specialize:
            if HEAD_DIM == 128:
                extra_args["maxnreg"] = 168  # Optimal register allocation for B200
            else:
                extra_args["maxnreg"] = 80
        
        def grid(META):
            return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        
        flash_attention_fwd_b200[grid](
            sm_scale, M,
            q.shape[0], q.shape[1],
            desc_q, desc_k, desc_v, desc_o,
            N_CTX=q.shape[2],
            HEAD_DIM=HEAD_DIM,
            FP8_OUTPUT=False,
            STAGE=stage,
            warp_specialize=warp_specialize and is_blackwell(),
            IS_BLACKWELL=is_blackwell(),
            **extra_args,
        )
        
        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM
        ctx.causal = causal
        
        return o
    
    @staticmethod
    def backward(ctx, do):
        # Backward pass implementation (similar optimizations apply)
        q, k, v, o, M = ctx.saved_tensors
        # ... backward kernel calls
        raise NotImplementedError("Backward pass - implement with similar B200 optimizations")


# Convenience function
def flash_attention_b200(q, k, v, causal=True, sm_scale=None):
    """
    B200-optimized Flash Attention.
    
    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        v: Value tensor [batch, heads, seq_len, head_dim]
        causal: Whether to apply causal masking
        sm_scale: Softmax scaling factor (default: 1/sqrt(head_dim))
    
    Returns:
        Output tensor [batch, heads, seq_len, head_dim]
    """
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    
    return FlashAttentionB200.apply(q, k, v, causal, sm_scale, is_blackwell())
```

### Helion B200 Optimizations

For Helion, enable Blackwell-specific features:

```python
import torch
import helion
import helion.language as hl

@helion.kernel(
    config=helion.Config(
        block_sizes=[128, 128, 64],    # Larger tiles for B200
        num_warps=8,                    # More warps for B200 SM
        num_stages=4,                   # More pipeline stages
        indexing='tensor_descriptor',   # Use TMA on Blackwell
        range_warp_specializes=[True, True],  # Enable warp specialization
    ),
    # Force autotuning to include Blackwell-specific configs
    autotune_effort="high",
)
def flash_attention_helion_b200(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Helion Flash Attention optimized for NVIDIA B200."""
    batch, heads, seq_len, head_dim = q.shape
    output = torch.empty_like(q)
    
    for tile_b, tile_h in hl.tile([batch, heads]):
        for tile_m in hl.tile(seq_len):
            m_i = hl.full([tile_m], float('-inf'), dtype=torch.float32)
            l_i = hl.zeros([tile_m], dtype=torch.float32)
            acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
            
            q_block = q[tile_b, tile_h, tile_m, :]
            
            for tile_n in hl.tile(seq_len):
                k_block = k[tile_b, tile_h, tile_n, :]
                v_block = v[tile_b, tile_h, tile_n, :]
                
                scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale
                
                m_ij = scores.max(dim=-1).values
                m_new = torch.maximum(m_i, m_ij)
                
                alpha = torch.exp(m_i - m_new)
                p = torch.exp(scores - m_new[:, None])
                l_ij = p.sum(dim=-1)
                
                acc = acc * alpha[:, None] + torch.matmul(p, v_block)
                l_i = l_i * alpha + l_ij
                m_i = m_new
            
            output[tile_b, tile_h, tile_m, :] = acc / l_i[:, None]
    
    return output
```

### B200 Performance Expectations

| Configuration | Expected TFLOPS (FP16) | Expected TFLOPS (FP8) |
|--------------|------------------------|------------------------|
| seq_len=1024, d=64, causal | 800+ | 1,200+ |
| seq_len=2048, d=64, causal | 1,000+ | 1,400+ |
| seq_len=4096, d=64, causal | 1,200+ | 1,500+ |
| seq_len=8192, d=128, causal | 1,400+ | 1,600+ |
| seq_len=16384, d=128, causal | 1,500+ | 1,605+ |

### B200 Best Practices

1. **Always Enable Warp Specialization**
   ```python
   warp_specialize=True  # Critical for B200 performance
   ```

2. **Use TMA Tensor Descriptors**
   ```python
   indexing='tensor_descriptor'  # In Helion
   # Or use TensorDescriptor class in Triton
   ```

3. **Larger Block Sizes**
   ```python
   BLOCK_M = 128
   BLOCK_N = 128
   # B200's larger SMEM and TMEM can handle bigger tiles
   ```

4. **More Pipeline Stages**
   ```python
   num_stages=4  # Up from 2-3 on older GPUs
   ```

5. **Optimal Register Allocation**
   ```python
   maxnreg=168  # For HEAD_DIM=128 on B200
   ```

6. **Consider FP8 for Training**
   ```python
   # B200 has native FP8 support with 2x throughput
   q = q.to(torch.float8_e5m2)
   k = k.to(torch.float8_e5m2)
   v = v.to(torch.float8_e5m2)
   ```

---

## GPU Memory Hierarchy

Understanding GPU memory hierarchy is crucial for writing efficient attention kernels.

```
┌─────────────────────────────────────────────────────────────────┐
│                         GPU Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Global Memory (HBM)                       │ │
│  │              Size: 40-80 GB, Bandwidth: ~2 TB/s              │ │
│  │                    Latency: ~400 cycles                      │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              ▲                                    │
│                              │ (Slow)                             │
│                              ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                   Shared Memory (SRAM)                       │ │
│  │              Size: 48-164 KB per SM                          │ │
│  │              Bandwidth: ~19 TB/s effective                   │ │
│  │                    Latency: ~20 cycles                       │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              ▲                                    │
│                              │ (Fast)                             │
│                              ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                      Registers                               │ │
│  │              Size: ~256 KB per SM                            │ │
│  │                    Latency: ~1 cycle                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Optimization Strategy

1. **Minimize HBM Access**: Each HBM read/write is expensive
2. **Maximize Data Reuse**: Keep data in SRAM as long as possible
3. **Coalesced Access**: Ensure adjacent threads access adjacent memory locations
4. **Tiling**: Break large computations into SRAM-sized chunks

---

## Flash Attention Algorithm

Flash Attention is an I/O-aware attention algorithm that significantly reduces memory usage and improves speed by avoiding materialization of the full attention matrix.

### Standard Attention vs Flash Attention

**Standard Attention:**
```
S = Q @ K^T           # O(N²) memory for N×N matrix
P = softmax(S)        # O(N²) memory
O = P @ V             # O(N²) memory reads
```

**Flash Attention:**
```
For each block of Q:
    For each block of K, V:
        - Compute partial attention scores on-chip
        - Update running softmax statistics (online softmax)
        - Accumulate partial output
# Never materialize full N×N matrix!
```

### Online Softmax Algorithm

The key innovation is computing softmax incrementally without needing all values at once:

```python
# Traditional softmax requires two passes:
# Pass 1: Find max for numerical stability
# Pass 2: Compute exp and sum

# Online softmax (single pass with updates):
def online_softmax_update(m_prev, l_prev, acc_prev, new_block):
    """
    m_prev: previous maximum
    l_prev: previous sum of exp(x - m_prev)
    acc_prev: previous accumulated output
    new_block: new block of attention scores
    """
    m_new = max(m_prev, max(new_block))
    
    # Correction factor for previous values
    alpha = exp(m_prev - m_new)
    
    # Update accumulator with correction
    acc_new = acc_prev * alpha + exp(new_block - m_new) @ V_block
    
    # Update running sum
    l_new = l_prev * alpha + sum(exp(new_block - m_new))
    
    return m_new, l_new, acc_new
```

### Algorithm Pseudocode

```
Algorithm: Flash Attention Forward Pass
─────────────────────────────────────────

Input: Q, K, V ∈ ℝ^(N×d), block sizes Br, Bc
Output: O ∈ ℝ^(N×d)

1. Initialize O = 0, ℓ = 0, m = -∞  (all of shape N)

2. Divide Q into Tr = ⌈N/Br⌉ blocks Q₁, ..., Q_Tr
   Divide K, V into Tc = ⌈N/Bc⌉ blocks K₁, ..., K_Tc and V₁, ..., V_Tc

3. For j = 1 to Tc:
     - Load Kⱼ, Vⱼ from HBM to SRAM
     
     For i = 1 to Tr:
       - Load Qᵢ, Oᵢ, ℓᵢ, mᵢ from HBM to SRAM
       
       - Compute Sᵢⱼ = Qᵢ @ Kⱼᵀ ∈ ℝ^(Br×Bc)  (on-chip)
       
       - Compute m̃ᵢⱼ = rowmax(Sᵢⱼ) ∈ ℝ^Br
       - Compute P̃ᵢⱼ = exp(Sᵢⱼ - m̃ᵢⱼ) ∈ ℝ^(Br×Bc)
       - Compute ℓ̃ᵢⱼ = rowsum(P̃ᵢⱼ) ∈ ℝ^Br
       
       - Compute mᵢ_new = max(mᵢ, m̃ᵢⱼ)
       - Compute ℓᵢ_new = exp(mᵢ - mᵢ_new) * ℓᵢ + exp(m̃ᵢⱼ - mᵢ_new) * ℓ̃ᵢⱼ
       
       - Update Oᵢ = (ℓᵢ * exp(mᵢ - mᵢ_new) * Oᵢ + exp(m̃ᵢⱼ - mᵢ_new) * P̃ᵢⱼ @ Vⱼ) / ℓᵢ_new
       
       - Write Oᵢ, ℓᵢ_new, mᵢ_new back to HBM

4. Return O
```

---

## Core Concepts for Attention Kernels

### 1. Tiling Strategy

Tiling divides Q, K, V matrices into blocks that fit in SRAM:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Tiling for Attention                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│    Q Matrix (N × d)              K Matrix (N × d)                │
│    ┌──────────────┐              ┌──────────────┐                │
│    │ Q_block_0    │              │ K_block_0    │                │
│    ├──────────────┤              ├──────────────┤                │
│    │ Q_block_1    │              │ K_block_1    │                │
│    ├──────────────┤      @       ├──────────────┤                │
│    │ Q_block_2    │              │ K_block_2    │                │
│    ├──────────────┤              ├──────────────┤                │
│    │    ...       │              │    ...       │                │
│    └──────────────┘              └──────────────┘                │
│                                                                   │
│    Block Size Br                 Block Size Bc                   │
│    (typically 64-128)            (typically 32-128)              │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Numerical Stability

Always subtract maximum before exponentiating to prevent overflow:

```python
# Unstable:
exp(x)  # Can overflow for large x

# Stable:
exp(x - max(x))  # Always in [0, 1] range
```

### 3. Causal Masking

For autoregressive models, mask future tokens:

```python
# Create causal mask
# Position i can only attend to positions <= i
mask = offs_m[:, None] >= offs_n[None, :]
qk = tl.where(mask, qk, float('-inf'))
```

### 4. Scaling Factor

Apply the attention scaling factor:

```python
# Standard attention scaling
sm_scale = 1.0 / sqrt(head_dim)

# For numerical efficiency with exp2:
qk_scale = sm_scale * 1.44269504  # 1/log(2)
```

---

## Triton Attention Implementation

### Forward Pass Kernel Structure

```python
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q,          # Accumulators and query block
    K_block_ptr, V_block_ptr,  # Pointers to K, V blocks
    start_m, qk_scale,         # Position and scaling
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, 
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,       # 1=before diagonal, 2=diagonal, 3=no causal
    offs_m, offs_n,
    N_CTX: tl.constexpr,
):
    """Inner loop for computing attention over K, V blocks."""
    
    # Determine iteration range based on stage
    if STAGE == 1:  # Before diagonal (always visible)
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:  # Diagonal (needs masking)
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
    else:  # Non-causal (everything visible)
        lo, hi = 0, N_CTX
    
    # Iterate over K, V blocks
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Load K block and compute QK^T
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, tl.trans(k))
        
        # Apply causal mask if on diagonal
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, float('-inf'))
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        
        # Compute softmax weights
        p = tl.math.exp2(qk)
        
        # Online softmax correction
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        
        # Update accumulator with correction factor
        acc = acc * alpha[:, None]
        
        # Load V and accumulate
        v = tl.load(V_block_ptr)
        p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)
        
        # Update running statistics
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        
        # Advance pointers
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    
    return acc, l_i, m_i


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
    ],
    key=['N_CTX', 'HEAD_DIM'],
)
@triton.jit
def _attn_fwd(
    Q, K, V, O,                # Input/Output tensors
    sm_scale,                  # Softmax scaling factor
    M,                         # For storing LSE (log-sum-exp) for backward
    stride_qz, stride_qh, stride_qm, stride_qk,  # Q strides
    stride_kz, stride_kh, stride_kn, stride_kk,  # K strides
    stride_vz, stride_vh, stride_vn, stride_vk,  # V strides
    stride_oz, stride_oh, stride_om, stride_ok,  # O strides
    Z, H, N_CTX,              # Batch, heads, sequence length
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,      # 1 for causal, 3 for non-causal
):
    """Flash Attention forward pass kernel."""
    
    # Get program IDs
    start_m = tl.program_id(0)  # Which Q block
    off_hz = tl.program_id(1)   # Which batch*head
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Compute base offsets
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
    
    # Create block pointers
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
    
    # Initialize accumulators
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    # Apply scaling for exp2
    qk_scale = sm_scale * 1.44269504  # 1/log(2)
    
    # Load Q block (stays in SRAM)
    q = tl.load(Q_block_ptr)
    
    # Compute attention
    if STAGE & 1:  # Off-diagonal blocks (no masking needed)
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q,
            K_block_ptr, V_block_ptr,
            start_m, qk_scale,
            BLOCK_M, BLOCK_N, HEAD_DIM,
            4 - STAGE,
            offs_m, offs_n, N_CTX,
        )
    
    if STAGE & 2:  # Diagonal blocks (masking needed)
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q,
            K_block_ptr, V_block_ptr,
            start_m, qk_scale,
            BLOCK_M, BLOCK_N, HEAD_DIM,
            2,
            offs_m, offs_n, N_CTX,
        )
    
    # Finalize: apply normalization
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    
    # Store LSE for backward pass
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    
    # Store output
    tl.store(O_block_ptr, acc.to(tl.float16))
```

### Backward Pass Structure

The backward pass computes gradients dQ, dK, dV using the saved LSE values:

```python
@triton.jit
def _attn_bwd_preprocess(
    O, DO, Delta,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Precompute delta = rowsum(O * dO) for backward pass."""
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    
    # Load O and dO
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    
    # Compute delta
    delta = tl.sum(o * do, axis=1)
    
    # Store
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


@triton.jit
def _attn_bwd_dkdv(
    dk, dv, Q, k, v, sm_scale, DO, M, D,
    stride_tok, stride_d, H, N_CTX,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    start_n, start_m, num_steps,
    MASK: tl.constexpr,
):
    """Compute gradients dK and dV."""
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    
    curr_m = start_m
    step_m = BLOCK_M1
    
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        
        if MASK:
            mask = offs_m[None, :] >= offs_n[:, None]
            pT = tl.where(mask, pT, 0.0)
        
        do = tl.load(do_ptrs)
        
        # Compute dV
        dv += tl.dot(pT.to(tl.float16), do)
        
        # Compute dP and dS
        Di = tl.load(D + offs_m)
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        
        # Compute dK
        dk += tl.dot(dsT.to(tl.float16), tl.trans(qT))
        
        # Advance
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    
    return dk, dv
```

---

## Helion-Based Attention Implementation

Helion provides a higher-level abstraction for implementing attention:

```python
import torch
import helion
import helion.language as hl

@helion.kernel()
def flash_attention_helion(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Flash Attention implemented with Helion."""
    
    batch, heads, seq_len, head_dim = q.shape
    
    # Allocate output
    output = torch.empty_like(q)
    
    # Process each batch and head
    for tile_b, tile_h in hl.tile([batch, heads]):
        # Get slices for this batch/head
        q_slice = q[tile_b, tile_h]  # [seq_len, head_dim]
        k_slice = k[tile_b, tile_h]
        v_slice = v[tile_b, tile_h]
        
        # Tile over sequence dimension
        for tile_m in hl.tile(seq_len):
            # Initialize accumulators
            m_i = hl.full([tile_m], float('-inf'), dtype=torch.float32)
            l_i = hl.zeros([tile_m], dtype=torch.float32)
            acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
            
            # Tile over K, V
            for tile_n in hl.tile(seq_len):
                # Load Q, K blocks
                q_block = q_slice[tile_m, :]
                k_block = k_slice[tile_n, :]
                v_block = v_slice[tile_n, :]
                
                # Compute attention scores
                scores = torch.matmul(q_block, k_block.T) * sm_scale
                
                # Online softmax update
                m_ij = torch.max(scores, dim=-1).values
                m_new = torch.maximum(m_i, m_ij)
                
                alpha = torch.exp(m_i - m_new)
                beta = torch.exp(m_ij - m_new)
                
                p = torch.exp(scores - m_new[:, None])
                l_ij = torch.sum(p, dim=-1)
                
                # Update accumulators
                acc = acc * alpha[:, None] + torch.matmul(p, v_block)
                l_i = l_i * alpha + l_ij * beta
                m_i = m_new
            
            # Normalize and store
            output[tile_b, tile_h, tile_m, :] = acc / l_i[:, None]
    
    return output
```

### Helion Configuration Options

```python
@helion.kernel(
    config=helion.Config(
        block_sizes=[64, 64, 64],      # Tile sizes for each hl.tile dimension
        loop_orders=[[0, 1]],          # Iteration order
        l2_groupings=[4],              # PID grouping for L2 cache
        num_warps=8,                   # Warps per block
        num_stages=6,                  # Pipeline stages
        indexing='block_ptr',          # Memory indexing strategy
        pid_type='flat',               # Program ID mapping
    )
)
def optimized_attention(...):
    ...
```

---

## Autotuning

### Triton Autotuning

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w)
        for BM in [64, 128]
        for BN in [32, 64, 128]
        for s in [2, 3, 4]
        for w in [4, 8]
    ],
    key=['N_CTX', 'HEAD_DIM'],
    prune_configs_by={'early_config_prune': prune_invalid_configs},
)
@triton.jit
def _attn_fwd(...):
    ...


def prune_invalid_configs(configs, named_args, **kwargs):
    """Remove invalid configurations."""
    N_CTX = kwargs['N_CTX']
    STAGE = kwargs['STAGE']
    
    return [
        conf for conf in configs
        if conf.kwargs.get('BLOCK_M', 0) <= N_CTX
        and (conf.kwargs.get('BLOCK_M', 0) >= conf.kwargs.get('BLOCK_N', 0) or STAGE == 1)
    ]
```

### Helion Autotuning

Helion performs extensive autotuning automatically:

```python
# Enable autotuning (default behavior)
@helion.kernel()
def my_kernel(...):
    ...

# Skip autotuning for development
@helion.kernel(autotune_effort="none")
def my_kernel(...):
    ...

# Use specific configurations
@helion.kernel(configs=[
    helion.Config(block_sizes=[[32, 32], [16]], num_warps=4),
    helion.Config(block_sizes=[[64, 64], [32]], num_warps=8),
])
def my_kernel(...):
    ...
```

### Helion Autotuning Output Example

```
[0s] Starting DifferentialEvolutionSearch with population=40, generations=20, crossover_rate=0.8
[20s] Initial population: failed=4 min=0.0266 mid=0.1577 max=1.2390
      best=Config(block_sizes=[64, 32, 64], num_warps=4, num_stages=7, indexing='block_ptr')
[88s] Generation 3: replaced=18 min=0.0225 mid=0.0389 max=0.1085
      best=Config(block_sizes=[64, 64, 16], num_warps=4, num_stages=6, indexing='pointer')
...
[586s] Autotuning complete in 586.6s after searching 1520 configs.
```

---

## Installation & Requirements

### Triton Installation

```bash
# Install Triton (requires CUDA)
pip install triton

# For development version
pip install triton-nightly
```

### Helion Installation

```bash
# Requirements:
# - Python 3.10-3.14
# - PyTorch 2.9+
# - Triton 3.5+

# Install from PyPI
pip install helion

# Or from source
git clone https://github.com/pytorch/helion.git
cd helion
pip install -e .
```

### Verify Installation

```python
import triton
import triton.language as tl
print(f"Triton version: {triton.__version__}")

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

import helion
print("Helion installed successfully!")
```

---

## Complete Code Examples

### Example 1: Simple Triton Softmax

```python
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=float('-inf'))
    
    # Subtract max for numerical stability
    row_max = tl.max(row, axis=0)
    row_minus_max = row - row_max
    
    # Compute exp and sum
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    
    # Normalize
    softmax_output = numerator / denominator
    
    # Store
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def triton_softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    output = torch.empty_like(x)
    
    softmax_kernel[(n_rows,)](
        output,
        x,
        x.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
```

### Example 2: Flash Attention with Helion

```python
import torch
import helion
import helion.language as hl

@helion.kernel(
    autotune_effort="medium",
    print_output_code=False,
)
def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    """
    Flash Attention implementation using Helion.
    
    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        v: Value tensor [batch, heads, seq_len, head_dim]
        scale: Optional scaling factor (default: 1/sqrt(head_dim))
    
    Returns:
        Output tensor [batch, heads, seq_len, head_dim]
    """
    batch, heads, seq_len, head_dim = q.shape
    
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)
    
    output = torch.empty_like(q)
    
    for tile_b, tile_h in hl.tile([batch, heads]):
        for tile_m in hl.tile(seq_len):
            # Initialize per-row statistics
            m_i = hl.full([tile_m], float('-inf'), dtype=torch.float32)
            l_i = hl.zeros([tile_m], dtype=torch.float32)
            acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
            
            q_block = q[tile_b, tile_h, tile_m, :]
            
            for tile_n in hl.tile(seq_len):
                k_block = k[tile_b, tile_h, tile_n, :]
                v_block = v[tile_b, tile_h, tile_n, :]
                
                # QK^T
                scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale
                
                # Online softmax
                m_ij = scores.max(dim=-1).values
                m_new = torch.maximum(m_i, m_ij)
                
                p = torch.exp(scores - m_new[:, None])
                l_ij = p.sum(dim=-1)
                
                alpha = torch.exp(m_i - m_new)
                
                acc = acc * alpha[:, None] + torch.matmul(p, v_block)
                l_i = l_i * alpha + l_ij
                m_i = m_new
            
            output[tile_b, tile_h, tile_m, :] = acc / l_i[:, None]
    
    return output


# Usage
if __name__ == "__main__":
    batch, heads, seq_len, head_dim = 2, 8, 1024, 64
    
    q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    k = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    v = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    
    output = flash_attention(q, k, v)
    print(f"Output shape: {output.shape}")
```

---

## Performance Benchmarks

### Expected Performance (NVIDIA A100)

| Configuration | Forward TFLOPS | Backward TFLOPS |
|--------------|----------------|-----------------|
| seq_len=1024, d=64, causal=True | ~112 | ~60 |
| seq_len=2048, d=64, causal=True | ~137 | ~76 |
| seq_len=4096, d=64, causal=True | ~152 | ~88 |
| seq_len=8192, d=64, causal=True | ~160 | ~96 |
| seq_len=16384, d=64, causal=True | ~165 | ~100 |

### Running Benchmarks

```python
import triton
import torch

def benchmark_attention(fn, q, k, v, causal, sm_scale):
    """Benchmark attention implementation."""
    # Warmup
    for _ in range(10):
        _ = fn(q, k, v, causal, sm_scale)
    
    # Benchmark
    ms = triton.testing.do_bench(lambda: fn(q, k, v, causal, sm_scale))
    
    # Calculate TFLOPS
    batch, heads, seq_len, head_dim = q.shape
    flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * head_dim
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    
    tflops = total_flops * 1e-12 / (ms * 1e-3)
    return ms, tflops
```

---

## Best Practices

### 1. Memory Layout

```python
# Ensure contiguous memory layout
q = q.contiguous()
k = k.contiguous()
v = v.contiguous()

# Use optimal stride patterns
assert q.stride(-1) == 1, "Head dimension should be contiguous"
```

### 2. Block Size Selection

```python
# Guidelines for block sizes:
# - BLOCK_M: 64-128 (query block size)
# - BLOCK_N: 32-128 (key/value block size)
# - Should divide evenly into sequence length
# - Larger blocks = better compute/memory ratio but more register pressure
```

### 3. Numerical Precision

```python
# Use FP32 for accumulation to prevent precision loss
acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

# Convert back to FP16 only at the end
tl.store(output_ptr, acc.to(tl.float16))
```

### 4. Debugging Tips

```python
# Helion: View generated Triton code
@helion.kernel(print_output_code=True)
def my_kernel(...):
    ...

# Or via environment variable
# HELION_PRINT_OUTPUT_CODE=1 python script.py

# Triton: Use assertions
tl.static_assert(BLOCK_N <= HEAD_DIM, "BLOCK_N must not exceed HEAD_DIM")
```

### 5. Production Deployment

```python
# Cache autotuned configs for production
@helion.kernel(
    config=helion.Config(
        block_sizes=[64, 64, 64],
        num_warps=8,
        num_stages=6,
        indexing='block_ptr',
        pid_type='flat',
    )
)
def production_attention(...):
    ...
```

---

## References

### Papers

1. **Flash Attention**: Dao, T., et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

2. **Flash Attention 2**: Dao, T. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." 2023. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)

3. **Triton**: Tillet, P., et al. "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations." MAPL 2019.

### Documentation & Resources

- **Triton Official Documentation**: [https://triton-lang.org](https://triton-lang.org)
- **Helion Documentation**: [https://helionlang.com](https://helionlang.com)
- **Helion GitHub**: [https://github.com/pytorch/helion](https://github.com/pytorch/helion)
- **Flash Attention GitHub**: [https://github.com/HazyResearch/flash-attention](https://github.com/HazyResearch/flash-attention)

### Tutorials

- [Triton Fused Attention Tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [Helion Interactive Demo Notebook](https://github.com/pytorch/helion/blob/main/notebooks/softmax.ipynb)
- [PyTorch Conference 2025 Helion Talk](https://youtu.be/BW-Ht-5IxgM)

---

## Appendix: Quick Reference

### Triton Language Cheatsheet

```python
# Program identification
pid = tl.program_id(axis=0)

# Range creation
offs = tl.arange(0, BLOCK_SIZE)

# Memory operations
data = tl.load(ptr + offs, mask=mask, other=0.0)
tl.store(ptr + offs, data, mask=mask)

# Block pointers (efficient for 2D)
ptr = tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)
data = tl.load(ptr)
ptr = tl.advance(ptr, (rows, cols))

# Math operations
result = tl.dot(a, b)        # Matrix multiply
result = tl.exp(x)           # Exponential
result = tl.log(x)           # Natural log
result = tl.sum(x, axis=0)   # Sum reduction
result = tl.max(x, axis=0)   # Max reduction

# Control flow
result = tl.where(cond, a, b)

# Type conversion
x_fp16 = x.to(tl.float16)
```

### Helion Language Cheatsheet

```python
# Kernel definition
@helion.kernel()
def my_kernel(x: torch.Tensor) -> torch.Tensor:
    ...

# Tiling
for tile_i, tile_j in hl.tile([M, N]):
    ...

for tile_k in hl.tile(K):
    ...

# Tensor creation
zeros = hl.zeros([tile_i, tile_j], dtype=torch.float32)
full = hl.full([tile_i], value, dtype=torch.float32)

# Standard PyTorch operations work inside kernels
result = torch.matmul(a, b)
result = torch.softmax(x, dim=-1)
result = a + b
```

---

*Last updated: January 2026*
*Documentation version: 1.0*
