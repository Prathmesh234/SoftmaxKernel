"""
=================================================
PATCH EMBEDDING BENCHMARK
=================================================
Benchmarks all patch embedding implementations on H100 GPU.
Usage: python benchmark.py
"""

import torch
import json
import argparse
from typing import Callable, Dict, Tuple

# Setup
DEVICE = torch.device("cuda")
DTYPE = torch.float16

print("=" * 70)
print("PATCH EMBEDDING KERNEL BENCHMARKS - NVIDIA H100")
print("=" * 70)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"dtype: {DTYPE}")
print("=" * 70)

# Import our kernels
from patch_embedding_eager import eager_patch_embedding, naive_patch_embedding
from patch_embedding_triton import triton_patch_embedding_matmul, triton_patch_embedding_fused

try:
    from patch_embedding_helion import patch_embedding as helion_patch_embedding, HELION_AVAILABLE
except ImportError:
    HELION_AVAILABLE = False
    helion_patch_embedding = None


def create_inputs(
    batch_size: int,
    img_size: int,
    patch_size: int,
    in_channels: int = 3,
    embed_dim: int = 768,
    dtype=DTYPE,
    device=DEVICE
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create input tensors for patch embedding."""
    x = torch.randn(batch_size, in_channels, img_size, img_size, dtype=dtype, device=device)
    patch_dim = in_channels * patch_size * patch_size
    weight = torch.randn(embed_dim, patch_dim, dtype=dtype, device=device)
    bias = torch.randn(embed_dim, dtype=dtype, device=device)
    return x, weight, bias


def benchmark(
    fn: Callable,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    patch_size: int,
    warmup: int = 50,
    iters: int = 200
) -> Tuple[float, float]:
    """
    Accurate GPU benchmarking with CUDA events.
    
    Returns: (mean_ms, min_ms)
    """
    # Warmup
    for _ in range(warmup):
        _ = fn(x, weight, bias, patch_size)
    torch.cuda.synchronize()
    
    # Benchmark using CUDA events (most accurate)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(iters):
        start.record()
        _ = fn(x, weight, bias, patch_size)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    # Remove outliers
    times = sorted(times)[5:-5]
    return sum(times) / len(times), min(times)


def throughput_gbps(
    batch_size: int,
    img_size: int,
    patch_size: int,
    in_channels: int,
    embed_dim: int,
    time_ms: float,
    dtype_bytes: int = 2  # FP16
) -> float:
    """Calculate memory throughput in GB/s."""
    num_patches = (img_size // patch_size) ** 2
    patch_dim = in_channels * patch_size * patch_size
    
    # Input: B * C * H * W
    input_bytes = batch_size * in_channels * img_size * img_size * dtype_bytes
    # Output: B * num_patches * embed_dim
    output_bytes = batch_size * num_patches * embed_dim * dtype_bytes
    # Weight: embed_dim * patch_dim
    weight_bytes = embed_dim * patch_dim * dtype_bytes
    # Bias: embed_dim
    bias_bytes = embed_dim * dtype_bytes
    
    total_bytes = input_bytes + output_bytes + weight_bytes + bias_bytes
    return (total_bytes / (time_ms / 1000)) / 1e9


def tflops(
    batch_size: int,
    img_size: int,
    patch_size: int,
    in_channels: int,
    embed_dim: int,
    time_ms: float
) -> float:
    """Calculate TFLOPS for patch embedding (linear projection)."""
    num_patches = (img_size // patch_size) ** 2
    patch_dim = in_channels * patch_size * patch_size
    
    # FLOPs for matmul: 2 * B * num_patches * patch_dim * embed_dim
    flops = 2 * batch_size * num_patches * patch_dim * embed_dim
    return (flops / (time_ms / 1000)) / 1e12


def run_benchmarks(configs: list, warmup: int = 50, iters: int = 200):
    """Run all benchmarks and return results."""
    
    # Define kernels to benchmark
    kernels = {
        "eager": eager_patch_embedding,
        "triton_matmul": triton_patch_embedding_matmul,
        "triton_fused": triton_patch_embedding_fused,
    }
    
    if HELION_AVAILABLE and helion_patch_embedding:
        kernels["helion"] = helion_patch_embedding
    
    results = {}
    
    # Print header
    kernel_names = list(kernels.keys())
    header = f"{'Config':<25}"
    for name in kernel_names:
        header += f" | {name:>14}"
    print(f"\n{header}")
    print("-" * (27 + 17 * len(kernel_names)))
    
    for config in configs:
        batch_size, img_size, patch_size, embed_dim = config
        config_str = f"B={batch_size},I={img_size},P={patch_size},D={embed_dim}"
        
        x, weight, bias = create_inputs(batch_size, img_size, patch_size, embed_dim=embed_dim)
        
        # Reference using eager
        ref = eager_patch_embedding(x, weight, bias, patch_size)
        
        row = {"config": config_str}
        row_str = f"{config_str:<25}"
        
        for name, fn in kernels.items():
            try:
                # Verify correctness
                out = fn(x, weight, bias, patch_size)
                if not torch.allclose(out, ref, rtol=1e-2, atol=1e-2):
                    max_diff = (out - ref).abs().max().item()
                    row[name] = {"error": f"incorrect (max_diff={max_diff:.4f})"}
                    row_str += f" | {'ERROR':>14}"
                    continue
                
                # Benchmark
                mean_ms, min_ms = benchmark(fn, x, weight, bias, patch_size, warmup, iters)
                gbps = throughput_gbps(batch_size, img_size, patch_size, 3, embed_dim, mean_ms)
                tflop = tflops(batch_size, img_size, patch_size, 3, embed_dim, mean_ms)
                
                row[name] = {
                    "mean_ms": round(mean_ms, 4),
                    "min_ms": round(min_ms, 4),
                    "gbps": round(gbps, 1),
                    "tflops": round(tflop, 3)
                }
                row_str += f" | {mean_ms:>10.4f} ms"
                
            except Exception as e:
                row[name] = {"error": str(e)[:50]}
                row_str += f" | {'ERROR':>14}"
        
        print(row_str)
        results[config_str] = row
        
        del x, weight, bias
        torch.cuda.empty_cache()
    
    return results


def print_summary(results: Dict):
    """Print throughput summary."""
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    
    # Get kernel names from first result
    first_result = next(iter(results.values()))
    kernel_names = [k for k in first_result.keys() if k != "config"]
    
    header = f"{'Config':<25}"
    for name in kernel_names:
        header += f" | {name:>12}"
    header += " | Winner"
    print(header)
    print("-" * (37 + 15 * len(kernel_names)))
    
    for config, data in results.items():
        row_str = f"{config:<25}"
        best_time = float('inf')
        winner = ""
        
        for name in kernel_names:
            if name in data and isinstance(data[name], dict) and "mean_ms" in data[name]:
                mean_ms = data[name]["mean_ms"]
                row_str += f" | {mean_ms:>10.4f}"
                if mean_ms < best_time:
                    best_time = mean_ms
                    winner = name
            else:
                row_str += f" | {'-':>12}"
        
        row_str += f" | {winner}"
        print(row_str)
    
    print("=" * 70)
    print("H100 HBM3 Peak: 3,350 GB/s | H100 FP16 Peak: 989 TFLOPS")


def print_tflops_summary(results: Dict):
    """Print TFLOPS summary."""
    print("\n" + "=" * 70)
    print("TFLOPS (Higher is Better)")
    print("=" * 70)
    
    first_result = next(iter(results.values()))
    kernel_names = [k for k in first_result.keys() if k != "config"]
    
    header = f"{'Config':<25}"
    for name in kernel_names:
        header += f" | {name:>12}"
    print(header)
    print("-" * (27 + 15 * len(kernel_names)))
    
    for config, data in results.items():
        row_str = f"{config:<25}"
        
        for name in kernel_names:
            if name in data and isinstance(data[name], dict) and "tflops" in data[name]:
                tflop = data[name]["tflops"]
                row_str += f" | {tflop:>10.3f}"
            else:
                row_str += f" | {'-':>12}"
        
        print(row_str)


def main():
    parser = argparse.ArgumentParser(description="Patch embedding kernel benchmarks")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=200, help="Benchmark iterations")
    parser.add_argument("--save", type=str, default="benchmark_results.json", help="Output file")
    args = parser.parse_args()
    
    # Test configurations: (batch_size, img_size, patch_size, embed_dim)
    configs = [
        # Standard ViT configurations
        (1, 224, 16, 768),      # ViT-Base single image
        (8, 224, 16, 768),      # ViT-Base batch
        (32, 224, 16, 768),     # ViT-Base larger batch
        
        # ViT-Large
        (1, 224, 16, 1024),     # ViT-Large single
        (8, 224, 16, 1024),     # ViT-Large batch
        
        # ViT-Huge
        (1, 224, 14, 1280),     # ViT-Huge single (patch=14)
        (8, 224, 14, 1280),     # ViT-Huge batch
        
        # Higher resolution
        (1, 384, 16, 768),      # ViT-Base 384
        (8, 384, 16, 768),      # ViT-Base 384 batch
        
        # Larger patches
        (8, 224, 32, 768),      # Larger patch size
        
        # Different patch sizes
        (8, 256, 8, 512),       # Smaller patches, more tokens
    ]
    
    print(f"\nWarmup: {args.warmup}, Iterations: {args.iters}")
    
    results = run_benchmarks(configs, args.warmup, args.iters)
    print_summary(results)
    print_tflops_summary(results)
    
    # Save results
    with open(args.save, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.save}")


if __name__ == "__main__":
    main()
