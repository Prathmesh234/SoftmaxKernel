# Helion: PyTorch Kernel DSL for Optimized Attention

## Table of Contents
1. [Introduction](#introduction)
2. [Requirements & Installation](#requirements--installation)
3. [Core Concepts](#core-concepts)
4. [Helion Language API](#helion-language-api)
5. [Attention Kernel Implementation](#attention-kernel-implementation)
6. [Autotuning System](#autotuning-system)
7. [Configuration Options](#configuration-options)
8. [Settings & Debugging](#settings--debugging)
9. [Best Practices for Attention Kernels](#best-practices-for-attention-kernels)
10. [Example: Complete Attention Kernel](#example-complete-attention-kernel)
11. [Performance Optimization Tips](#performance-optimization-tips)
12. [**🚀 NVIDIA B200/Blackwell Optimizations**](#-nvidia-b200blackwell-optimizations) ← **NEW**
13. [References](#references)

---

## Introduction

**Helion** is a Python-embedded domain-specific language (DSL) developed by Meta for authoring high-performance machine learning kernels. It compiles down to **Triton**, providing a higher-level abstraction that makes kernel development more accessible while maintaining near-optimal performance.

### Key Benefits

| Feature | Description |
|---------|-------------|
| **PyTorch-Native Syntax** | Write GPU kernels using familiar PyTorch operations |
| **Automatic Tiling** | `hl.tile` automatically handles memory tiling strategies |
| **Powerful Autotuning** | Explores thousands of configurations to find optimal performance |
| **Hardware Portability** | Same kernel works across different GPU architectures |
| **Reduced Boilerplate** | ~30 lines of Helion vs ~120 lines of Triton vs thousands in CUDA |

### How Helion Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        Helion Kernel                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  @helion.kernel()                                        │   │
│  │  def attention(q, k, v):                                 │   │
│  │      # Host code (CPU) - shape computation, allocation   │   │
│  │      for tile in hl.tile([m, n]):                        │   │
│  │          # Device code (GPU) - compiled to Triton        │   │
│  │          acc = torch.bmm(q[tile], k[tile].T)             │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   TorchInductor │
                    └────────┬────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Triton Kernel  │
                    └────────┬────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │    GPU Code     │
                    │   (PTX/AMDGCN)  │
                    └─────────────────┘
```

---

## Requirements & Installation

### System Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Operating System** | Linux | Required (no Windows/macOS support) |
| **Python** | 3.10 - 3.14 | Python environment |
| **PyTorch** | ≥ 2.9 | Core dependency |
| **Triton** | ≥ 3.5 | GPU compiler backend |
| **CUDA** | 12.x | For NVIDIA GPUs |
| **ROCm** | 7.0+ | For AMD GPUs (optional) |

### Installation Methods

#### Method 1: pip install (Recommended)
```bash
# Install PyTorch 2.9+ first
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Install Helion
pip install helion
```

#### Method 2: From Source (For Development)
```bash
git clone https://github.com/pytorch/helion.git
cd helion

# Create virtual environment (using uv recommended)
uv venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e '.[dev]'
```

#### Verify Installation
```python
import torch
import helion
import helion.language as hl

print(f"PyTorch: {torch.__version__}")
print(f"Helion: {helion.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## Core Concepts

### Kernel Structure

A Helion kernel consists of two distinct sections:

```python
import torch
import helion
import helion.language as hl

@helion.kernel()
def my_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # ╔═══════════════════════════════════════════════╗
    # ║  HOST SECTION (runs on CPU)                   ║
    # ║  - Shape computations                         ║
    # ║  - Output tensor allocation                   ║
    # ║  - Scalar parameter setup                     ║
    # ╚═══════════════════════════════════════════════╝
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    
    # ╔═══════════════════════════════════════════════╗
    # ║  DEVICE SECTION (compiled to Triton/GPU)      ║
    # ║  - Parallel tile execution                    ║
    # ║  - PyTorch operations → Triton operations     ║
    # ╚═══════════════════════════════════════════════╝
    for tile_m, tile_n in hl.tile([m, n]):
        # Operations here run on GPU
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
    
    return out
```

### Tiling Model

The `hl.tile()` function is the core abstraction that enables parallel GPU execution:

```python
# 1D Tiling
for tile_m in hl.tile(m):
    # Each tile processes a portion of dimension m
    # Tile size determined by autotuner or config

# 2D Tiling (for matrices)
for tile_m, tile_n in hl.tile([m, n]):
    # Creates 2D grid of tiles
    # Maps to GPU thread blocks

# Nested Tiling (for reductions)
for tile_m, tile_n in hl.tile([m, n]):       # Grid (parallel)
    acc = hl.zeros([tile_m, tile_n])
    for tile_k in hl.tile(k):                 # Loop (sequential)
        acc += compute(tile_k)
```

### Memory Hierarchy Awareness

```
┌──────────────────────────────────────────────────────────────┐
│                     GPU Memory Hierarchy                      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  HBM (High Bandwidth Memory) - Large, Slow              │ │
│  │  • Full tensors stored here                             │ │
│  │  • ~1.5-2 TB/s bandwidth                                │ │
│  └───────────────────────────┬─────────────────────────────┘ │
│                              │                                │
│  ┌───────────────────────────▼─────────────────────────────┐ │
│  │  SRAM (On-chip) - Small, Fast                           │ │
│  │  • Tiles loaded here for computation                    │ │
│  │  • ~19 TB/s bandwidth                                   │ │
│  │  • Helion manages this automatically via hl.tile()      │ │
│  └───────────────────────────┬─────────────────────────────┘ │
│                              │                                │
│  ┌───────────────────────────▼─────────────────────────────┐ │
│  │  Registers - Fastest                                     │ │
│  │  • Intermediate computations                             │ │
│  │  • Accumulator variables                                 │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Helion Language API

### Core Functions (`helion.language` / `hl`)

#### `hl.tile(sizes)`
Subdivides iteration space into parallel tiles.

```python
# Single dimension
for tile_m in hl.tile(m):
    pass

# Multiple dimensions
for tile_m, tile_n in hl.tile([m, n]):
    pass

# With explicit block size
block_size = hl.register_block_size()
for tile_m in hl.tile(m, block_size=block_size):
    pass
```

#### `hl.zeros(shape, dtype)`
Creates a device tensor filled with zeros.

```python
# Accumulator initialization
acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
```

#### `hl.full(shape, value, dtype)`
Creates a device tensor filled with a specified value.

```python
# Initialize with -infinity for max reduction
m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
```

#### `hl.specialize(value)`
Converts dynamic values to compile-time constants.

```python
# Specialize head dimension for better optimization
head_dim = hl.specialize(q.size(-1))
```

#### `hl.load(tensor, indices, ...)`
Explicitly load from tensor with fine-grained control.

```python
# With eviction policy
data = hl.load(tensor, [i, j], eviction_policy="first")
```

#### `hl.store(tensor, indices, value, ...)`
Explicitly store to tensor with fine-grained control.

```python
hl.store(output, [i, j], computed_value)
```

#### `hl.register_block_size()`
Registers a tunable block size dimension.

```python
block_size = hl.register_block_size()
# Now block_size can be used in hl.tile() and autotuned
```

### Supported PyTorch Operations

Helion supports most PyTorch operations via TorchInductor:

| Category | Operations |
|----------|------------|
| **Pointwise** | `add`, `sub`, `mul`, `div`, `sigmoid`, `relu`, `exp`, `exp2`, `log`, `log2`, `tanh`, `gelu` |
| **Reductions** | `sum`, `mean`, `max`, `min`, `amax`, `amin`, `softmax` |
| **Matrix Operations** | `matmul`, `bmm`, `addmm`, `baddbmm` |
| **Tensor Creation** | `zeros`, `ones`, `full`, `empty`, `zeros_like`, `ones_like`, `full_like`, `empty_like` |
| **View Operations** | `reshape`, `view`, `transpose`, `squeeze`, `unsqueeze` |
| **Comparisons** | `maximum`, `minimum`, `eq`, `ne`, `lt`, `le`, `gt`, `ge` |

---

## Attention Kernel Implementation

### Flash Attention Algorithm Overview

Flash Attention optimizes standard attention by:

1. **Tiling**: Process Q, K, V in blocks to fit in SRAM
2. **Online Softmax**: Compute softmax incrementally without materializing full attention matrix
3. **Recomputation**: Trade compute for memory to avoid storing intermediate results

```
┌────────────────────────────────────────────────────────────────┐
│                    Flash Attention Algorithm                    │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Standard Attention:                                            │
│  • Compute S = Q @ K.T         (N × N matrix - HUGE!)          │
│  • Compute P = softmax(S)      (Store full matrix)              │
│  • Compute O = P @ V           (Another matmul)                 │
│  • Memory: O(N²) - Prohibitive for long sequences              │
│                                                                 │
│  Flash Attention (Tiled):                                       │
│  • For each Q tile:                                             │
│    • For each K,V tile:                                         │
│      • Compute local attention scores                           │
│      • Update running softmax statistics                        │
│      • Accumulate weighted values                               │
│  • Memory: O(N) - Linear scaling!                              │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Key Components for Attention in Helion

#### 1. Online Softmax with Rescaling

```python
# Running maximum for numerical stability
m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
# Running sum of exponentials
l_i = torch.full_like(m_i, 1.0)
# Accumulator
acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)

for tile_n in hl.tile(n_dim):
    # Compute attention scores
    qk = torch.bmm(q, k)
    
    # Update running max
    m_ij = torch.maximum(m_i, torch.amax(qk, -1) * scale)
    
    # Compute rescaled exponential
    qk = qk * scale - m_ij[:, :, None]
    p = torch.exp2(qk)
    
    # Update running sum
    l_ij = torch.sum(p, -1)
    alpha = torch.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    
    # Rescale and accumulate
    acc = acc * alpha[:, :, None]
    acc = torch.baddbmm(acc, p, v)
    
    m_i = m_ij

# Final normalization
acc = acc / l_i[:, :, None]
```

#### 2. Scaling Factor

```python
import math

# Softmax scale for numerical stability
sm_scale = 1.0 / math.sqrt(head_dim)

# Use log2 for exp2 optimization (faster on GPU)
qk_scale = sm_scale * 1.44269504  # 1/log(2)
```

---

## Autotuning System

### How Autotuning Works

Helion uses **Differential Evolution Search** to explore the configuration space:

```
┌────────────────────────────────────────────────────────────────┐
│                    Autotuning Process                           │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Generate Initial Population (40 random configs)            │
│     └─ Evaluate performance of each                            │
│                                                                 │
│  2. Evolution Loop (20 generations):                            │
│     ├─ Crossover: Combine good configurations                  │
│     ├─ Mutation: Random modifications                          │
│     └─ Selection: Keep best performers                         │
│                                                                 │
│  3. Output: Best configuration found                            │
│                                                                 │
│  Typical search: ~1000-1500 configurations                      │
│  Typical time: 5-15 minutes (depending on kernel complexity)    │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Autotuning Example Output

```
[0s] Starting DifferentialEvolutionSearch with population=40, generations=20
[20s] Initial population: failed=4 min=0.0266 mid=0.1577 max=1.2390
      best=Config(block_sizes=[64, 32, 64], num_warps=4, ...)
[51s] Generation 2: replaced=17 min=0.0266 mid=0.0573 max=0.1331
...
[586s] Generation 19: replaced=3 min=0.0184 mid=0.0225 max=0.0287
       best=Config(block_sizes=[64, 64, 64], num_warps=8, ...)
[586s] Autotuning complete in 586.6s after searching 1520 configs.
```

### Using Autotuned Configurations

#### Option 1: Let Helion Autotune
```python
@helion.kernel()  # Will autotune on first run
def attention(q, k, v):
    ...

# First call triggers autotuning
out = attention(q, k, v)
```

#### Option 2: Hardcode Best Config
```python
@helion.kernel(config=helion.Config(
    block_sizes=[64, 64, 64],
    loop_orders=[[0, 1]],
    l2_groupings=[4],
    num_warps=8,
    num_stages=6,
    indexing='block_ptr',
    pid_type='flat'
))
def attention(q, k, v):
    ...
```

#### Option 3: Provide Multiple Configs for Quick Selection
```python
@helion.kernel(configs=[
    helion.Config(block_sizes=[[32, 32], [16]], num_warps=4),
    helion.Config(block_sizes=[[64, 64], [32]], num_warps=8),
    helion.Config(block_sizes=[[128, 128], [64]], num_warps=8),
])
def attention(q, k, v):
    ...
```

---

## Configuration Options

### Complete Config Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `block_sizes` | `list[int]` | Tile sizes for each `hl.tile()` dimension |
| `loop_orders` | `list[list[int]]` | Permutation of iteration order per `hl.tile()` with 2+ dims |
| `flatten_loops` | `list[bool]` | Flatten iteration space into single dimension |
| `range_unroll_factors` | `list[int]` | Unroll factor for each loop dimension |
| `range_num_stages` | `list[int]` | Pipeline stages for each loop dimension |
| `range_multi_buffers` | `list[bool\|None]` | Enable multi-buffering per loop |
| `range_warp_specializes` | `list[bool\|None]` | Enable warp specialization (Blackwell+) |
| `range_flattens` | `list[bool\|None]` | Flatten parameter for `tl.range()` |
| `static_ranges` | `list[bool]` | Use `tl.static_range()` for static bounds |
| `reduction_loops` | `list[int\|None]` | `None` = persistent, `int` = loop with that block size |
| `l2_groupings` | `list[int]` | PID reordering for L2 cache optimization |
| `indexing` | `str \| list[str]` | `"pointer"`, `"block_ptr"`, or `"tensor_descriptor"` |
| `pid_type` | `str` | `"flat"`, `"xyz"`, `"persistent_blocked"`, `"persistent_interleaved"` |
| `num_warps` | `int` | Number of warps per thread block |
| `num_stages` | `int` | Pipeline stages for Triton |
| `load_eviction_policies` | `list[str]` | `""`, `"first"`, or `"last"` per load |

### Indexing Strategies

```python
# Pointer indexing (default) - Most flexible
@helion.kernel(config=helion.Config(indexing="pointer"))

# Block pointer - Better for contiguous access patterns
@helion.kernel(config=helion.Config(indexing="block_ptr"))

# Tensor descriptor - Uses TMA on Hopper+ GPUs
@helion.kernel(config=helion.Config(indexing="tensor_descriptor"))

# Per-operation indexing
@helion.kernel(config=helion.Config(
    indexing=["pointer", "block_ptr", "block_ptr"]  # load1, load2, store1
))
```

### PID Types

```python
# Flat (default) - Single x-dimension grid
@helion.kernel(config=helion.Config(pid_type="flat"))

# XYZ - Multi-dimensional grid
@helion.kernel(config=helion.Config(pid_type="xyz"))

# Persistent blocked - Better SM utilization
@helion.kernel(config=helion.Config(pid_type="persistent_blocked"))

# Persistent interleaved - Alternative persistent strategy
@helion.kernel(config=helion.Config(pid_type="persistent_interleaved"))
```

---

## Settings & Debugging

### Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `HELION_AUTOTUNE_EFFORT` | `none` | Skip autotuning |
| `HELION_PRINT_OUTPUT_CODE` | `1` | Print generated Triton code |
| `HELION_PRINT_REPRO` | `1` | Print reproduction script |
| `HELION_FORCE_AUTOTUNE` | `1` | Force autotuning even with config |
| `HELION_AUTOTUNE_RANDOM_SEED` | `<int>` | Reproducible autotuning |
| `HELION_LOGS` | `all`, `+all` | Enable logging |
| `HELION_INTERPRET` | `1` | Run kernel in eager mode |
| `TRITON_INTERPRET` | `1` | Run Triton's CPU interpreter |

### Decorator Settings

```python
@helion.kernel(
    # Skip autotuning for development
    autotune_effort="none",
    
    # Print generated Triton code
    print_output_code=True,
    
    # Print reproduction script
    print_repro=True,
    
    # Enable static shapes optimization
    static_shapes=True,
    
    # Set random seed for reproducibility
    autotune_random_seed=42,
)
def my_kernel(x, y):
    ...
```

### Debugging Workflow

```python
# 1. Start with autotuning disabled for fast iteration
@helion.kernel(autotune_effort="none", print_output_code=True)
def attention(q, k, v):
    ...

# 2. Test correctness
q, k, v = [torch.randn(2, 32, 1024, 64, device="cuda") for _ in range(3)]
out = attention(q, k, v)
ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
assert torch.allclose(out, ref, atol=1e-2), "Mismatch!"

# 3. Enable autotuning for performance
@helion.kernel()  # Default: full autotuning
def attention(q, k, v):
    ...

# 4. Hardcode best config for production
@helion.kernel(config=best_config)
def attention(q, k, v):
    ...
```

### Getting Triton Code Programmatically

```python
# Bind kernel to arguments and extract Triton code
bound = attention.bind(q, k, v)
triton_code = bound.to_triton_code(config)
print(triton_code)
```

---

## Best Practices for Attention Kernels

### 1. Use Static Shapes When Possible

```python
@helion.kernel(static_shapes=True)  # Enables better optimization
def attention(q, k, v):
    ...
```

### 2. Specialize on Head Dimension

```python
head_dim = hl.specialize(q.size(-1))  # Compile-time constant
```

### 3. Use Higher Precision Accumulators

```python
# Input may be float16, but accumulate in float32
acc = hl.zeros([tile_m, tile_n, head_dim], dtype=torch.float32)

# Cast back to output dtype at the end
out[tile_m, tile_n, :] = acc.to(out.dtype)
```

### 4. Optimize for Numerical Stability

```python
# Use log2/exp2 instead of log/exp (faster on GPU)
qk_scale = sm_scale * 1.44269504  # 1/log(2)
p = torch.exp2(qk * qk_scale - m_ij[:, :, None])
```

### 5. Pre-tune for Production

```python
# Development: autotune once, save config
configs = attention.autotune(q, k, v)
best_config = configs[0]
print(f"Best: {best_config}")

# Production: use saved config
@helion.kernel(config=saved_config)
def attention_prod(q, k, v):
    ...
```

---

## Example: Complete Attention Kernel

```python
"""
Complete Flash Attention Implementation in Helion
Supports: Batch, Multi-head, Variable sequence lengths
"""

from __future__ import annotations
import math
import torch
import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def flash_attention(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> torch.Tensor:
    """
    Computes scaled dot-product attention using Flash Attention algorithm.
    
    Implements: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Args:
        q_in: Query tensor of shape [batch, heads, seq_len_q, head_dim]
        k_in: Key tensor of shape [batch, heads, seq_len_k, head_dim]
        v_in: Value tensor of shape [batch, heads, seq_len_k, head_dim]
    
    Returns:
        Output tensor of shape [batch, heads, seq_len_q, head_dim]
    """
    # =========================================
    # HOST SECTION (CPU)
    # =========================================
    
    # Extract dimensions
    m_dim = q_in.size(-2)  # Query sequence length
    n_dim = k_in.size(-2)  # Key/Value sequence length
    assert n_dim == v_in.size(-2), "K and V must have same sequence length"
    
    # Specialize head dimension for compile-time optimization
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1), "Head dimensions must match"
    
    # Reshape for batched processing: [B*H, N, D]
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)  # [B*H, D, N]
    
    # Allocate output
    out = torch.empty_like(q_view)
    
    # Scaling factors
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504  # Scale for exp2 (1/log(2))
    
    # =========================================
    # DEVICE SECTION (GPU - Compiled to Triton)
    # =========================================
    
    # Tile over batch dimension and query positions
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        # Initialize running statistics for online softmax
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        
        # Load query tile (stays in SRAM for all K,V tiles)
        q = q_view[tile_b, tile_m, :]
        
        # Iterate over key/value sequence (tiled for memory efficiency)
        for tile_n in hl.tile(v_view.size(1)):
            # Load key tile and compute attention scores: Q @ K^T
            k = k_view[tile_b, :, tile_n]
            qk = torch.bmm(q, k)  # [tile_b, tile_m, tile_n]
            
            # Online softmax: update running maximum
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            
            # Compute scaled softmax: exp2((qk * scale) - max)
            qk = qk * qk_scale - m_ij[:, :, None]
            p = torch.exp2(qk)
            
            # Update running sum
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)  # Rescaling factor
            l_i = l_i * alpha + l_ij
            
            # Rescale previous accumulator
            acc = acc * alpha[:, :, None]
            
            # Accumulate weighted values: acc += P @ V
            v = v_view[tile_b, tile_n, :]
            p = p.to(v.dtype)  # Match dtypes for matmul
            acc = torch.baddbmm(acc, p, v)
            
            # Update running max for next iteration
            m_i = m_ij
        
        # Final normalization
        m_i += torch.log2(l_i)  # Log-space normalization
        acc = acc / l_i[:, :, None]
        
        # Write output with dtype conversion
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    
    return out.view(q_in.size())


# =========================================
# DYNAMIC SHAPES VERSION
# =========================================
flash_attention_dynamic: object = helion.kernel(
    flash_attention.fn,
    configs=flash_attention.configs,
    static_shapes=False,
)


# =========================================
# TESTING & BENCHMARKING
# =========================================
def test_attention(
    batch: int = 2,
    heads: int = 32,
    seq_len: int = 1024,
    head_dim: int = 64,
    dtype: torch.dtype = torch.float16,
) -> None:
    """Test against PyTorch reference."""
    device = "cuda"
    
    # Create random inputs
    q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
    
    # Reference implementation
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    
    # Helion implementation
    out = flash_attention(q, k, v)
    
    # Verify correctness
    max_diff = (out - ref).abs().max().item()
    print(f"Max difference: {max_diff:.6f}")
    
    if dtype == torch.float16:
        assert max_diff < 1e-2, f"Results differ too much: {max_diff}"
    else:
        assert max_diff < 1e-5, f"Results differ too much: {max_diff}"
    
    print("✓ Test passed!")


def benchmark_attention(
    batch: int = 4,
    heads: int = 32,
    seq_len: int = 4096,
    head_dim: int = 128,
    dtype: torch.dtype = torch.float16,
    warmup: int = 10,
    iterations: int = 100,
) -> None:
    """Benchmark performance."""
    device = "cuda"
    
    q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
    
    # Warmup
    for _ in range(warmup):
        _ = flash_attention(q, k, v)
    torch.cuda.synchronize()
    
    # Benchmark
    import time
    start = time.perf_counter()
    for _ in range(iterations):
        _ = flash_attention(q, k, v)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    avg_time = elapsed / iterations * 1000  # ms
    print(f"Average time: {avg_time:.3f} ms")
    
    # Calculate FLOPS
    flops = 4 * batch * heads * seq_len * seq_len * head_dim  # Approximate
    tflops = flops / (avg_time / 1000) / 1e12
    print(f"Throughput: {tflops:.2f} TFLOPS")


if __name__ == "__main__":
    test_attention()
    # benchmark_attention()
```

---

## Performance Optimization Tips

### 1. Tile Size Selection

| Sequence Length | Recommended Block Sizes |
|-----------------|------------------------|
| < 512 | `[32, 32]` or `[64, 32]` |
| 512 - 2048 | `[64, 64]` or `[128, 64]` |
| > 2048 | `[128, 128]` (let autotuner decide) |

### 2. Memory Access Patterns

```python
# Good: Coalesced access (threads access adjacent memory)
for tile_m in hl.tile(m):
    x = data[tile_m, :]  # Contiguous in last dimension

# Better: Use block_ptr indexing for contiguous patterns
@helion.kernel(config=helion.Config(indexing="block_ptr"))
```

### 3. L2 Cache Optimization

```python
# Enable L2 grouping for better cache utilization
@helion.kernel(config=helion.Config(l2_groupings=[8]))
```

### 4. Pipeline Stages

```python
# More stages = better latency hiding (up to a point)
@helion.kernel(config=helion.Config(
    num_stages=4,  # 3-6 typically works well
    range_num_stages=[0, 3],  # Per-loop configuration
))
```

### 5. Warp Configuration

```python
# More warps = better occupancy (but more register pressure)
@helion.kernel(config=helion.Config(
    num_warps=8,  # 4 or 8 typically optimal
))
```

---

## 🚀 NVIDIA B200/Blackwell Optimizations

The NVIDIA B200 GPU (Blackwell architecture) introduces significant hardware features that Helion can leverage for maximum attention kernel performance. This section covers B200-specific optimizations.

### B200 Hardware Features

| Feature | Specification | Helion Support |
|---------|---------------|----------------|
| **5th Gen Tensor Cores** | FP4, FP6, FP8, BF16, TF32, FP64 | ✅ Native support |
| **Tensor Memory (TMEM)** | 256 KB per SM | ✅ Via `tensor_descriptor` indexing |
| **TMA (Tensor Memory Accelerator)** | Async 1D-5D tensor transfers | ✅ Via `tensor_descriptor` indexing |
| **Warp Specialization** | Dedicated producer/consumer warps | ✅ Via `range_warp_specializes` |
| **2nd Gen Transformer Engine** | FP8 with micro-tensor scaling | ✅ Supported |
| **HBM3e Memory** | ~8 TB/s bandwidth | Automatic |

### B200 Memory Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    NVIDIA B200 Memory Hierarchy                     │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  HBM3e (High Bandwidth Memory) - 192GB                        │ │
│  │  • ~8 TB/s bandwidth (massive improvement over H100)          │ │
│  │  • Full model weights and activations                         │ │
│  └───────────────────────────────┬───────────────────────────────┘ │
│                                  │ TMA (Tensor Memory Accelerator)  │
│  ┌───────────────────────────────▼───────────────────────────────┐ │
│  │  TMEM (Tensor Memory) - 256 KB/SM  ← NEW IN BLACKWELL         │ │
│  │  • User-managed high-speed memory                             │ │
│  │  • Optimized for tensor operations                            │ │
│  │  • Accessed via tensor_descriptor indexing                    │ │
│  └───────────────────────────────┬───────────────────────────────┘ │
│                                  │                                  │
│  ┌───────────────────────────────▼───────────────────────────────┐ │
│  │  L2 Cache - 96 MB (vs 50MB on H100)                           │ │
│  │  • Larger cache = better data reuse                           │ │
│  │  • Use l2_groupings for optimization                          │ │
│  └───────────────────────────────┬───────────────────────────────┘ │
│                                  │                                  │
│  ┌───────────────────────────────▼───────────────────────────────┐ │
│  │  Shared Memory / L1 - 256 KB/SM (configurable)                │ │
│  │  • Tiles loaded here for computation                          │ │
│  └───────────────────────────────┬───────────────────────────────┘ │
│                                  │                                  │
│  ┌───────────────────────────────▼───────────────────────────────┐ │
│  │  Registers - 256 KB/SM                                        │ │
│  │  • Accumulators, intermediate values                          │ │
│  │  • Critical for attention kernel performance                  │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Key B200 Optimizations in Helion

#### 1. Tensor Descriptor Indexing (TMA)

The Tensor Memory Accelerator enables efficient async data movement. Use `tensor_descriptor` indexing to leverage TMA:

```python
@helion.kernel(
    config=helion.Config(
        indexing="tensor_descriptor",  # Use TMA for memory access
        # ... other configs
    ),
    static_shapes=True,
)
def b200_attention(q, k, v):
    ...
```

**Benefits of TMA:**
- Async data transfer without using SM instructions
- Single thread can initiate large tensor copies
- Reduces register pressure for address computation
- Enables overlapping compute with memory access

#### 2. Warp Specialization

Blackwell supports dedicated producer/consumer warp patterns:

```python
@helion.kernel(
    config=helion.Config(
        # Enable warp specialization for the inner loop
        range_warp_specializes=[True, None],  # [outer_loop, inner_loop]
        # ... other configs
    ),
    static_shapes=True,
)
def b200_attention(q, k, v):
    for tile_m in hl.tile(M):           # Outer loop - warp specialized
        for tile_n in hl.tile(N):       # Inner loop - standard
            ...
```

**Warp Specialization Explained:**
```
┌────────────────────────────────────────────────────────────────┐
│                    Warp Specialization                          │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Traditional Execution:                                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ All Warps: Load Data → Compute → Store → Repeat           │  │
│  │            [IDLE WHILE WAITING FOR MEMORY]                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Warp Specialization (Blackwell):                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Producer Warps: [Load Q,K,V tiles via TMA continuously]   │  │
│  │                        ↓ (signal when ready)              │  │
│  │ Consumer Warps: [Compute attention on ready tiles]        │  │
│  │                 [No waiting - always have data!]          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Result: Higher Tensor Core utilization, better overlap        │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

#### 3. Persistent Kernels

For maximum SM utilization on B200:

```python
@helion.kernel(
    config=helion.Config(
        pid_type="persistent_interleaved",  # Best for B200
        # Alternative: "persistent_blocked"
    ),
)
def b200_attention(q, k, v):
    ...
```

**PID Types for B200:**
| PID Type | Description | Best For |
|----------|-------------|----------|
| `persistent_interleaved` | Warps interleave across tiles | Large batch attention |
| `persistent_blocked` | Warps process contiguous blocks | Memory-bound workloads |

#### 4. Register Pressure Management

B200 has 256KB registers per SM. Control register usage for optimal performance:

```python
@helion.kernel(
    config=helion.Config(
        # Limit max registers per warp for warp specialization
        _triton_config_maxRegAutoWS=152,  # or 192
    ),
)
def b200_attention(q, k, v):
    ...
```

#### 5. FP8/BF16 Precision

B200's 5th-gen Tensor Cores excel at lower precision:

```python
def b200_attention(q, k, v):
    # Use BF16 for inputs
    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    v = v.to(torch.bfloat16)
    
    # FP32 accumulator for numerical stability
    acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
    
    # ... attention computation ...
    
    # Cast output
    return out.to(torch.bfloat16)
```

### Complete B200-Optimized Attention Kernel

Here's the official Blackwell-optimized attention kernel from the Helion repository:

```python
"""
B200/Blackwell-Optimized Flash Attention Kernel
Achieves up to 1605 TFLOPS (71% hardware utilization)
"""

from __future__ import annotations
import math
import torch
import helion
from helion.autotuner.config_fragment import EnumFragment
import helion.language as hl


def _mul_f32x2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Vectorized F32 PTX MUL for Blackwell"""
    return hl.inline_asm_elementwise(
        """
            {
                .reg .b64 ra, rb, rc;
                mov.b64 ra, { $2, $3 };
                mov.b64 rb, { $4, $5 };
                mul.f32x2 rc, ra, rb;
                mov.b64 { $0, $1 }, rc;
            }
            """,
        "=r,=r,r,r,r,r",
        [a, b],
        dtype=torch.float32,
        is_pure=True,
        pack=2,
    )


@helion.kernel(
    configs=[
        helion.Config(
            block_sizes=[256, N],
            range_warp_specializes=[OUTER_LOOP or None, None if OUTER_LOOP else True],
            range_multi_buffers=[None, False],
            pid_type="persistent_interleaved",
            indexing="tensor_descriptor",
            num_warps=4,
            num_stages=3,
            _triton_range_id_data_partition_factor=0,
            _triton_range_value_data_partition_factor=2,
            _triton_config_maxRegAutoWS=maxreg,
        )
        for N in [64, 128]
        for OUTER_LOOP in [True]
        for maxreg in [152, 192]
    ],
    static_shapes=True,
    autotune_accuracy_check=False,
)
def blackwell_attention_kernel(
    q_in: torch.Tensor, 
    k_in: torch.Tensor, 
    v_in: torch.Tensor, 
    qk_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    B200-Optimized Flash Attention with:
    - Tensor Memory Accelerator (TMA) via tensor_descriptor
    - Warp specialization for producer/consumer pattern
    - Persistent kernels for maximum SM utilization
    - Vectorized F32x2 operations
    
    Args:
        q_in: Query [B, H, M, D]
        k_in: Key [B, H, N, D]
        v_in: Value [B, H, N, D]
        qk_scale: Attention scale factor
    
    Returns:
        output: Attention output [B, H, M, D]
        lse: Log-sum-exp for backward pass [B, H, M]
    """
    B, H, M, D = q_in.shape
    Bk, Hk, N, Dk = k_in.shape
    assert Dk == D and Bk == B and Hk == H
    
    Bv, Hv, Nv, Dv = v_in.shape
    assert Bv == B and Hv == Hk and Nv == N
    
    # Specialize dimensions for compile-time optimization
    D = hl.specialize(D)
    Dv = hl.specialize(Dv)
    
    # Flatten batch and head dimensions
    q = q_in.reshape(-1, D)
    k = k_in.reshape(-1, D)
    v = v_in.reshape(-1, Dv)
    MM = q.shape[0]
    
    # Allocate outputs
    o = q.new_empty(MM, Dv)
    lse = q.new_empty(MM, dtype=torch.float32)
    
    # Register tunable block sizes
    block_m = hl.register_block_size(M)
    block_n = hl.register_block_size(N)
    assert M % block_m == 0
    assert N % block_n == 0
    
    # Register Blackwell-specific tunables
    hl.register_tunable(
        "_triton_range_id_data_partition_factor", 
        EnumFragment(choices=(0,))
    )
    hl.register_tunable(
        "_triton_range_value_data_partition_factor", 
        EnumFragment(choices=(2,))
    )
    hl.register_tunable(
        "_triton_config_maxRegAutoWS", 
        EnumFragment(choices=(152, 192))
    )
    
    SUBTILING = True
    qk_scale = qk_scale * 1.44269504  # 1/log(2) for exp2
    
    for tile_m in hl.tile(MM, block_size=block_m):
        # Initialize running statistics
        m_i = hl.zeros([tile_m]) - float("inf")
        l_i = hl.zeros([tile_m]) + 1.0
        acc = hl.zeros([tile_m, Dv])
        q_i = q[tile_m, :]
        
        # Compute attention head offset
        start_N = tile_m.begin // M * N
        
        for tile_n in hl.tile(N, block_size=block_n):
            k_j = k[tile_n + start_N, :]
            v_j = v[tile_n + start_N, :]
            
            # Compute attention scores using hl.dot (optimized for Blackwell)
            qk = hl.dot(q_i, k_j.T, out_dtype=torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            
            p = torch.exp2(qk)
            alpha = torch.exp2(m_i - m_ij)
            l_ij = torch.sum(p, -1)
            
            # Subtiling with vectorized multiply for better performance
            if SUBTILING:
                acc0, acc1 = hl.split(
                    acc.reshape([tile_m, 2, Dv // 2]).permute(0, 2, 1)
                )
                acc0 = _mul_f32x2(acc0, alpha[:, None])
                acc1 = _mul_f32x2(acc1, alpha[:, None])
                acc = (
                    hl.join(acc0, acc1)
                    .permute(0, 2, 1)
                    .reshape(acc.size(0), acc.size(1))
                )
            else:
                acc = acc * alpha[:, None]
            
            # Accumulate with fused matmul
            p = p.to(v.dtype)
            acc = hl.dot(p, v_j, acc=acc)  # Non-transposed V - Blackwell only!
            
            l_i = l_i * alpha + l_ij
            m_i = m_ij
        
        # Finalize
        m_i += torch.log2(l_i)
        acc = acc / l_i[:, None]
        lse[tile_m] = m_i
        o[tile_m, :] = acc
    
    return o.reshape(B, H, M, Dv), lse.reshape(B, H, M)


def blackwell_attention(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """User-facing wrapper with automatic scaling."""
    qk_scale = math.sqrt(1.0 / q.shape[-1])
    return blackwell_attention_kernel(q, k, v, qk_scale)


# =========================================
# B200 BENCHMARKING
# =========================================
def benchmark_b200(
    batch: int = 4,
    heads: int = 32,
    seq_len: int = 8192,
    head_dim: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Benchmark B200-optimized attention."""
    from triton.testing import do_bench
    
    device = "cuda"
    q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
    
    # Warmup and benchmark
    dur = do_bench(lambda: blackwell_attention(q, k, v))
    
    # Calculate TFLOPS
    flops = batch * heads * seq_len * seq_len * head_dim * 4
    tflops = flops / dur * 1e-9  # dur is in ms
    
    print(f"B200 Attention Benchmark:")
    print(f"  Shape: [{batch}, {heads}, {seq_len}, {head_dim}]")
    print(f"  Dtype: {dtype}")
    print(f"  Time: {dur:.3f} ms")
    print(f"  Throughput: {tflops:.2f} TFLOPS")
    
    # B200 theoretical peak reference
    # FP16/BF16: ~2250 TFLOPS, FP8: ~4500 TFLOPS
    print(f"  Utilization: ~{tflops/2250*100:.1f}% (vs 2250 TFLOPS peak)")


if __name__ == "__main__":
    benchmark_b200()
```

### B200 Configuration Quick Reference

```python
# ═══════════════════════════════════════════════════════════════════
#                    B200 OPTIMAL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

@helion.kernel(
    configs=[
        helion.Config(
            # ─────────────────────────────────────────────────────────
            # TILE SIZES (tune based on sequence length)
            # ─────────────────────────────────────────────────────────
            block_sizes=[256, 128],      # [M_tile, N_tile] - larger for B200
            
            # ─────────────────────────────────────────────────────────
            # B200-SPECIFIC: Warp Specialization
            # ─────────────────────────────────────────────────────────
            range_warp_specializes=[True, None],  # Enable for outer loop
            
            # ─────────────────────────────────────────────────────────
            # B200-SPECIFIC: TMA via Tensor Descriptors
            # ─────────────────────────────────────────────────────────
            indexing="tensor_descriptor",  # Use TMA for async memory
            
            # ─────────────────────────────────────────────────────────
            # B200-SPECIFIC: Persistent Kernels
            # ─────────────────────────────────────────────────────────
            pid_type="persistent_interleaved",  # Maximum SM utilization
            
            # ─────────────────────────────────────────────────────────
            # PIPELINE CONFIGURATION
            # ─────────────────────────────────────────────────────────
            num_warps=4,                  # 4 works well with warp spec
            num_stages=3,                 # 3-4 for B200
            range_multi_buffers=[None, False],
            
            # ─────────────────────────────────────────────────────────
            # B200-SPECIFIC: Register Limits for Warp Specialization
            # ─────────────────────────────────────────────────────────
            _triton_config_maxRegAutoWS=152,  # 152 or 192
        ),
    ],
    static_shapes=True,           # Always enable for production
    autotune_accuracy_check=False,  # Disable for speed (verify separately)
)
def b200_optimized_kernel(...):
    ...
```

### B200 Performance Expectations

| Configuration | Sequence Length | Expected TFLOPS | Utilization |
|---------------|-----------------|-----------------|-------------|
| BF16, 4x32 heads | 4096 | ~1200-1400 | ~55-60% |
| BF16, 4x32 heads | 8192 | ~1400-1605 | ~65-71% |
| FP8, 4x32 heads | 8192 | ~2500-3000 | ~55-65% |

**Note:** FlashAttention-4 on B200 can achieve up to **1605 TFLOPS** (71% utilization) with optimal tuning!

### B200 vs H100 Comparison

| Feature | H100 | B200 | Helion Config |
|---------|------|------|---------------|
| Tensor Cores | 4th Gen | 5th Gen | - |
| TMA | Yes | Enhanced | `indexing="tensor_descriptor"` |
| TMEM | No | 256KB/SM | Automatic with tensor_descriptor |
| Warp Specialization | Limited | Full | `range_warp_specializes=[True, ...]` |
| Peak FP16/BF16 | ~990 TFLOPS | ~2250 TFLOPS | - |
| Peak FP8 | ~1980 TFLOPS | ~4500 TFLOPS | Use FP8 inputs |
| Memory BW | 3.35 TB/s | ~8 TB/s | - |

### B200 Quick Start Checklist

```
┌────────────────────────────────────────────────────────────────┐
│                B200 OPTIMIZATION CHECKLIST                      │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ☐ Use tensor_descriptor indexing for TMA                      │
│      indexing="tensor_descriptor"                               │
│                                                                 │
│  ☐ Enable warp specialization                                  │
│      range_warp_specializes=[True, None]                        │
│                                                                 │
│  ☐ Use persistent kernels                                      │
│      pid_type="persistent_interleaved"                          │
│                                                                 │
│  ☐ Set static_shapes=True                                      │
│                                                                 │
│  ☐ Use BF16 or FP8 for inputs                                  │
│      torch.bfloat16 or torch.float8_e4m3fn                      │
│                                                                 │
│  ☐ Use FP32 accumulators                                       │
│      acc = hl.zeros([...], dtype=torch.float32)                 │
│                                                                 │
│  ☐ Larger tile sizes (B200 has more SRAM)                      │
│      block_sizes=[256, 128] instead of [64, 64]                 │
│                                                                 │
│  ☐ Configure register limits                                   │
│      _triton_config_maxRegAutoWS=152                            │
│                                                                 │
│  ☐ Use hl.dot() instead of torch.bmm() for attention           │
│      qk = hl.dot(q_i, k_j.T, out_dtype=torch.float32)           │
│                                                                 │
│  ☐ Run autotuning on B200 hardware                             │
│      Configs tuned on H100 won't be optimal!                    │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## References

### Official Resources

- **Helion Documentation**: [https://helionlang.com](https://helionlang.com)
- **GitHub Repository**: [https://github.com/pytorch/helion](https://github.com/pytorch/helion)
- **PyTorch Blog**: [Helion: A DSL for High-Performance ML Kernels](https://pytorch.org/blog/helion/)
- **GPU Mode Discord**: `#helion` channel

### Related Papers & Talks

- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Flash Attention 2](https://tridao.me/publications/flash2/flash2.pdf)
- [PyTorch Conference 2025 Helion Talk](https://youtu.be/BW-Ht-5IxgM)

### Example Notebooks

- [Interactive Softmax Demo (Google Colab)](https://github.com/pytorch/helion/blob/main/notebooks/softmax.ipynb)
- [Helion Puzzles](https://helionlang.com/puzzles.html) - Build up to Flash Attention

---

## Quick Reference Card

```
┌────────────────────────────────────────────────────────────────┐
│                    HELION QUICK REFERENCE                       │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  IMPORTS:                                                       │
│    import helion                                                │
│    import helion.language as hl                                 │
│                                                                 │
│  KERNEL DECORATOR:                                              │
│    @helion.kernel()                                             │
│    @helion.kernel(config=helion.Config(...))                    │
│    @helion.kernel(static_shapes=True, autotune_effort="none")   │
│                                                                 │
│  CORE FUNCTIONS:                                                │
│    hl.tile([m, n])      - Parallel tiling                       │
│    hl.zeros([m, n])     - Create zero tensor                    │
│    hl.full([m], val)    - Create constant tensor                │
│    hl.specialize(val)   - Compile-time constant                 │
│    hl.load(t, [i, j])   - Explicit load                         │
│    hl.store(t, [i, j])  - Explicit store                        │
│                                                                 │
│  DEBUGGING:                                                     │
│    HELION_AUTOTUNE_EFFORT=none python script.py                 │
│    HELION_PRINT_OUTPUT_CODE=1 python script.py                  │
│                                                                 │
│  ATTENTION PATTERN:                                             │
│    for tile_m in hl.tile(m):                                    │
│        m_i = hl.full([tile_m], -inf)                            │
│        l_i, acc = init_accumulators()                           │
│        for tile_n in hl.tile(n):                                │
│            qk = Q @ K.T                                         │
│            m_ij = max(m_i, max(qk))                             │
│            p = exp(qk - m_ij)                                   │
│            alpha = exp(m_i - m_ij)                              │
│            l_i = l_i * alpha + sum(p)                           │
│            acc = acc * alpha + p @ V                            │
│            m_i = m_ij                                           │
│        out = acc / l_i                                          │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```
