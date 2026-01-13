"""
=================================================
Using the Winning Helion Softmax Kernel
=================================================
This script demonstrates how to use the pre-tuned
Helion softmax kernel that achieved 2,240 GB/s
(67% of H100 peak) on [4096, 32000] tensors.

Usage:
    python example_helion_winner.py
"""

import torch
import time

# Check Helion availability
try:
    import helion
    import helion.language as hl
    print("✅ Helion is available")
except ImportError:
    print("❌ Helion not installed!")
    print("   Install with: pip install git+https://github.com/pytorch/helion.git")
    exit(1)

DEVICE = torch.device("cuda")

print("=" * 60)
print("🏆 WINNING HELION SOFTMAX KERNEL")
print("=" * 60)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 60)


# =============================================================================
# METHOD 1: Let Helion auto-tune (uses cache after first run)
# =============================================================================

@helion.kernel()
def softmax_autotune(x: torch.Tensor) -> torch.Tensor:
    """
    Helion softmax with automatic autotuning.
    
    First call will autotune (~5 min for full search).
    Subsequent calls use cached configuration.
    """
    n, m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        out[tile_n, :] = torch.nn.functional.softmax(x[tile_n, :], dim=1)
    return out


# =============================================================================
# METHOD 2: Pre-tuned configuration (skip autotuning - instant!)
# =============================================================================

# This is the EXACT winning configuration from Helion's autotuning output
# for [4096, 32000] on H100.
#
# Source: Helion printed this after autotuning completed:
#   "[200s] Autotuning complete in 200.8s after searching 410 configs.
#    One can hardcode the best config and skip autotuning with: ..."
#
WINNING_CONFIG = helion.Config(
    # === Core Parameters ===
    block_sizes=[1],              # One row per thread block
    num_stages=1,                 # No software pipelining
    num_warps=32,                 # Maximum warps (1024 threads/block)
    pid_type='flat',              # Simple program ID scheduling
    
    # === Memory Access Strategy ===
    indexing=[
        'tensor_descriptor',      # TMA for input tensor (H100 HW accelerated!)
        'pointer',                # Standard pointer arithmetic
        'pointer', 
        'tensor_descriptor',      # TMA for another access
        'pointer',
        'pointer'
    ],
    
    # === L2 Cache Eviction Policies ===
    load_eviction_policies=[
        'last',                   # Keep input in L2 cache (will be reused)
        '',                       # Default policy
        'first',                  # Stream to memory (don't cache)
        'first'                   # Stream to memory
    ],
    
    # === Additional Parameters (from autotuning) ===
    range_flattens=[None],        # No loop flattening
    range_multi_buffers=[None],   # No multi-buffering
    range_num_stages=[0],         # No range pipelining
    range_unroll_factors=[0],     # No unrolling
    range_warp_specializes=[],    # No warp specialization
    reduction_loops=[None],       # Reduction fits in shared memory
)


@helion.kernel(config=WINNING_CONFIG, static_shapes=True)
def softmax_pretuned(x: torch.Tensor) -> torch.Tensor:
    """
    Pre-tuned Helion softmax - skips autotuning!
    
    Use this in production for instant kernel execution.
    Optimized for [4096, 32000] tensors (LLaMA vocab size).
    """
    n, m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        out[tile_n, :] = torch.nn.functional.softmax(x[tile_n, :], dim=1)
    return out


# =============================================================================
# HELPER: Benchmark function
# =============================================================================

def benchmark(fn, x, warmup=20, iters=100):
    """Benchmark with CUDA events."""
    # Warmup
    for _ in range(warmup):
        _ = fn(x)
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(iters):
        start.record()
        _ = fn(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    mean_ms = sum(times) / len(times)
    min_ms = min(times)
    
    # Calculate throughput
    m, n = x.shape
    bytes_total = 2 * m * n * 2  # FP16, read + write
    gbps = (bytes_total / (mean_ms / 1000)) / 1e9
    
    return mean_ms, min_ms, gbps


# =============================================================================
# MAIN: Example usage
# =============================================================================

def main():
    print("\n📌 Example: LLaMA Vocabulary Softmax [4096, 32000]")
    print("-" * 60)
    
    # Create input tensor (LLaMA vocab shape)
    batch_size = 4096
    vocab_size = 32000
    
    x = torch.randn(batch_size, vocab_size, device=DEVICE, dtype=torch.float16)
    print(f"Input shape: {x.shape}")
    print(f"Input dtype: {x.dtype}")
    print(f"Input size: {x.numel() * 2 / 1e6:.1f} MB")
    
    # ----- Method 1: PyTorch baseline -----
    print("\n1️⃣ PyTorch Eager (baseline):")
    mean, min_t, gbps = benchmark(lambda t: torch.softmax(t, dim=-1), x)
    print(f"   Mean: {mean:.4f} ms | Throughput: {gbps:.1f} GB/s")
    
    # ----- Method 2: Pre-tuned Helion -----
    print("\n2️⃣ Helion Pre-tuned (winner):")
    
    # First call compiles the kernel (fast with pre-tuned config)
    print("   Compiling kernel...", end=" ", flush=True)
    _ = softmax_pretuned(x)
    torch.cuda.synchronize()
    print("done")
    
    # Benchmark
    mean, min_t, gbps = benchmark(softmax_pretuned, x)
    print(f"   Mean: {mean:.4f} ms | Throughput: {gbps:.1f} GB/s")
    
    # Verify correctness
    ref = torch.softmax(x, dim=-1)
    helion_out = softmax_pretuned(x)
    max_diff = (helion_out - ref).abs().max().item()
    print(f"   Correctness: max diff = {max_diff:.2e} ✅" if max_diff < 1e-2 else f"   ⚠️ Max diff: {max_diff}")
    
    # ----- Calculate speedup -----
    print("\n" + "=" * 60)
    print("📊 COMPARISON")
    print("=" * 60)
    
    # Re-run both for fair comparison
    eager_mean, _, eager_gbps = benchmark(lambda t: torch.softmax(t, dim=-1), x)
    helion_mean, _, helion_gbps = benchmark(softmax_pretuned, x)
    
    speedup = eager_mean / helion_mean
    print(f"  PyTorch Eager: {eager_mean:.4f} ms ({eager_gbps:.1f} GB/s)")
    print(f"  Helion Winner: {helion_mean:.4f} ms ({helion_gbps:.1f} GB/s)")
    print(f"  Speedup: {speedup:.2f}x faster! 🚀")
    print(f"  H100 Peak Efficiency: {helion_gbps / 3350 * 100:.1f}%")
    
    # ----- Show how to use in your code -----
    print("\n" + "=" * 60)
    print("💡 HOW TO USE IN YOUR CODE")
    print("=" * 60)
    print("""
# Option 1: Import and use directly
from softmax_helion import helion_softmax_simple
output = helion_softmax_simple(input_tensor)

# Option 2: Use the pre-tuned version (recommended for production)
from example_helion_winner import softmax_pretuned
output = softmax_pretuned(input_tensor)

# Option 3: Copy the config to your own kernel
OPTIMAL_CONFIG = helion.Config(
    block_sizes=[1],
    num_warps=32,
    num_stages=1,
    pid_type='flat',
    indexing=['tensor_descriptor', 'pointer', 'pointer', 
              'tensor_descriptor', 'pointer', 'pointer'],
    load_eviction_policies=['last', '', 'first', 'first']
)

@helion.kernel(config=OPTIMAL_CONFIG, static_shapes=True)
def my_softmax(x):
    ...
""")


if __name__ == "__main__":
    main()
