"""
=================================================
TRITON PATCH EMBEDDING BENCHMARK - H100 SXM5
=================================================
Benchmark for optimized Triton kernel implementations.
Usage: python benchmark_triton.py
"""

import sys
sys.path.insert(0, '..')

import torch
import json
import argparse
from typing import Callable, Tuple

# Setup
DEVICE = torch.device("cuda")
DTYPE = torch.float16

print("=" * 80)
print("TRITON PATCH EMBEDDING BENCHMARK - NVIDIA H100 SXM5")
print("=" * 80)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA: {torch.version.cuda} | PyTorch: {torch.__version__}")
print(f"dtype: {DTYPE}")
print("=" * 80)

# Import kernels
from patch_embedding_triton import triton_patch_embedding_matmul, triton_patch_embedding_fused
from patch_embedding_eager import eager_patch_embedding


def create_inputs(batch_size, img_size, patch_size, in_channels=3, embed_dim=768, dtype=DTYPE, device=DEVICE):
    """Create input tensors for patch embedding."""
    x = torch.randn(batch_size, in_channels, img_size, img_size, dtype=dtype, device=device)
    patch_dim = in_channels * patch_size * patch_size
    weight = torch.randn(embed_dim, patch_dim, dtype=dtype, device=device)
    bias = torch.randn(embed_dim, dtype=dtype, device=device)
    return x, weight, bias


def benchmark(fn: Callable, x, weight, bias, patch_size, warmup=50, iters=200) -> Tuple[float, float]:
    """Benchmark with CUDA events."""
    for _ in range(warmup):
        _ = fn(x, weight, bias, patch_size)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(iters):
        start.record()
        _ = fn(x, weight, bias, patch_size)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    times = sorted(times)[5:-5]
    return sum(times) / len(times), min(times)


def tflops(batch_size, img_size, patch_size, in_channels, embed_dim, time_ms):
    """Calculate TFLOPS."""
    num_patches = (img_size // patch_size) ** 2
    patch_dim = in_channels * patch_size * patch_size
    flops = 2 * batch_size * num_patches * patch_dim * embed_dim
    return (flops / (time_ms / 1000)) / 1e12


def throughput_gbps(batch_size, img_size, patch_size, in_channels, embed_dim, time_ms, dtype_bytes=2):
    """Calculate memory throughput in GB/s."""
    num_patches = (img_size // patch_size) ** 2
    patch_dim = in_channels * patch_size * patch_size
    input_bytes = batch_size * in_channels * img_size * img_size * dtype_bytes
    output_bytes = batch_size * num_patches * embed_dim * dtype_bytes
    weight_bytes = embed_dim * patch_dim * dtype_bytes
    bias_bytes = embed_dim * dtype_bytes
    total_bytes = input_bytes + output_bytes + weight_bytes + bias_bytes
    return (total_bytes / (time_ms / 1000)) / 1e9


def run_benchmarks(configs, warmup=50, iters=200):
    """Run benchmarks for all configs."""
    
    kernels = {
        "eager": eager_patch_embedding,
        "triton_matmul": triton_patch_embedding_matmul,
        "triton_fused": triton_patch_embedding_fused,
    }
    
    results = {}
    
    print(f"\n{'Config':<30} | {'eager':>12} | {'triton_matmul':>14} | {'triton_fused':>14} | {'Best':>12} | {'Speedup':>8}")
    print("-" * 110)
    
    for config in configs:
        batch_size, img_size, patch_size, embed_dim = config
        config_str = f"B={batch_size}, I={img_size}, P={patch_size}, D={embed_dim}"
        
        x, weight, bias = create_inputs(batch_size, img_size, patch_size, embed_dim=embed_dim)
        ref = eager_patch_embedding(x, weight, bias, patch_size)
        
        row = {"config": config_str}
        times = {}
        
        for name, fn in kernels.items():
            try:
                out = fn(x, weight, bias, patch_size)
                if not torch.allclose(out, ref, rtol=1e-2, atol=1e-2):
                    max_diff = (out - ref).abs().max().item()
                    times[name] = None
                    row[name] = {"error": f"max_diff={max_diff:.4f}"}
                else:
                    mean_ms, min_ms = benchmark(fn, x, weight, bias, patch_size, warmup, iters)
                    times[name] = mean_ms
                    row[name] = {
                        "mean_ms": round(mean_ms, 4),
                        "min_ms": round(min_ms, 4),
                        "tflops": round(tflops(batch_size, img_size, patch_size, 3, embed_dim, mean_ms), 3),
                        "gbps": round(throughput_gbps(batch_size, img_size, patch_size, 3, embed_dim, mean_ms), 1),
                    }
            except Exception as e:
                times[name] = None
                row[name] = {"error": str(e)[:40]}
        
        # Find best
        valid_times = {k: v for k, v in times.items() if v is not None}
        if valid_times:
            best = min(valid_times, key=valid_times.get)
            row["best"] = best
            if "eager" in valid_times and best != "eager":
                row["speedup"] = round(valid_times["eager"] / valid_times[best], 2)
            else:
                row["speedup"] = 1.0
        
        # Print row
        row_str = f"{config_str:<30}"
        for name in ["eager", "triton_matmul", "triton_fused"]:
            if times.get(name) is not None:
                row_str += f" | {times[name]:>10.4f}ms"
            else:
                row_str += f" | {'ERROR':>12}"
        row_str += f" | {row.get('best', 'N/A'):>12}"
        row_str += f" | {row.get('speedup', 'N/A'):>7}x"
        print(row_str)
        
        results[config_str] = row
        del x, weight, bias
        torch.cuda.empty_cache()
    
    return results


def print_summary(results):
    """Print performance summary."""
    print("\n" + "=" * 80)
    print("TFLOPS SUMMARY (Higher is Better)")
    print("=" * 80)
    
    print(f"\n{'Config':<30} | {'eager':>10} | {'triton_matmul':>14} | {'triton_fused':>14}")
    print("-" * 80)
    
    for config, data in results.items():
        row = f"{config:<30}"
        for name in ["eager", "triton_matmul", "triton_fused"]:
            if name in data and isinstance(data[name], dict) and "tflops" in data[name]:
                row += f" | {data[name]['tflops']:>10.3f}"
            else:
                row += f" | {'-':>10}"
        print(row)
    
    print("\n" + "=" * 80)
    print("H100 SXM5 Reference: FP16 Tensor Core Peak = 989 TFLOPS, HBM3 = 3,350 GB/s")
    print("=" * 80)
    
    # Compute average speedups
    speedups = [d["speedup"] for d in results.values() if "speedup" in d and d["speedup"] != 1.0]
    if speedups:
        avg_speedup = sum(speedups) / len(speedups)
        max_speedup = max(speedups)
        print(f"\n📊 Average Triton Speedup: {avg_speedup:.2f}x | Max: {max_speedup:.2f}x")
    
    # Count wins
    wins = {"eager": 0, "triton_matmul": 0, "triton_fused": 0}
    for data in results.values():
        if "best" in data:
            wins[data["best"]] = wins.get(data["best"], 0) + 1
    print(f"📈 Wins: eager={wins['eager']}, triton_matmul={wins['triton_matmul']}, triton_fused={wins['triton_fused']}")


def main():
    parser = argparse.ArgumentParser(description="Triton patch embedding benchmark")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--save", type=str, default="triton_benchmark_results.json")
    args = parser.parse_args()
    
    configs = [
        # Standard ViT configs
        (1, 224, 16, 768),      # ViT-Base single
        (8, 224, 16, 768),      # ViT-Base batch
        (32, 224, 16, 768),     # ViT-Base large batch
        
        # ViT-Large
        (1, 224, 16, 1024),
        (8, 224, 16, 1024),
        
        # ViT-Huge
        (1, 224, 14, 1280),
        (8, 224, 14, 1280),
        
        # High resolution
        (1, 384, 16, 768),
        (8, 384, 16, 768),
        
        # Different configs
        (8, 224, 32, 768),
        (8, 256, 8, 512),
    ]
    
    print(f"\nWarmup: {args.warmup}, Iterations: {args.iters}")
    
    results = run_benchmarks(configs, args.warmup, args.iters)
    print_summary(results)
    
    with open(args.save, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: {args.save}")


if __name__ == "__main__":
    main()
