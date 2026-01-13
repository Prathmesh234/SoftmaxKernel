"""
=================================================
SOFTMAX BENCHMARK
=================================================
Benchmarks all softmax implementations on H100 GPU.
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
print("SOFTMAX KERNEL BENCHMARKS - NVIDIA H100")
print("=" * 70)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"dtype: {DTYPE}")
print("=" * 70)

# Import our kernels
from softmax_eager import eager_softmax, naive_softmax
from softmax_triton import triton_softmax

try:
    from softmax_helion import softmax as helion_softmax, HELION_AVAILABLE
except ImportError:
    HELION_AVAILABLE = False
    helion_softmax = None


def benchmark(
    fn: Callable,
    x: torch.Tensor,
    warmup: int = 50,
    iters: int = 200
) -> Tuple[float, float]:
    """
    Accurate GPU benchmarking with CUDA events.
    
    Returns: (mean_ms, min_ms)
    """
    # Warmup
    for _ in range(warmup):
        _ = fn(x)
    torch.cuda.synchronize()
    
    # Benchmark using CUDA events (most accurate)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(iters):
        start.record()
        _ = fn(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    # Remove outliers
    times = sorted(times)[5:-5]
    return sum(times) / len(times), min(times)


def throughput_gbps(m: int, n: int, time_ms: float) -> float:
    """Calculate memory throughput in GB/s."""
    bytes_per_elem = 2  # FP16
    total_bytes = 2 * m * n * bytes_per_elem  # Read + Write
    return (total_bytes / (time_ms / 1000)) / 1e9


def run_benchmarks(shapes: list, warmup: int = 50, iters: int = 200):
    """Run all benchmarks and return results."""
    
    # Define kernels to benchmark
    kernels = {
        "eager": eager_softmax,
        "triton": triton_softmax,
    }
    if HELION_AVAILABLE and helion_softmax:
        kernels["helion"] = helion_softmax
    
    results = {}
    
    # Print header
    kernel_names = list(kernels.keys())
    header = f"{'Shape':<18}"
    for name in kernel_names:
        header += f" | {name:>12}"
    print(f"\n{header}")
    print("-" * (20 + 15 * len(kernel_names)))
    
    for m, n in shapes:
        shape_str = f"[{m}, {n}]"
        x = torch.randn(m, n, device=DEVICE, dtype=DTYPE)
        ref = torch.softmax(x, dim=-1)
        
        row = {"shape": shape_str, "elements": m * n}
        row_str = f"{shape_str:<18}"
        
        for name, fn in kernels.items():
            try:
                # Verify correctness
                out = fn(x)
                if not torch.allclose(out, ref, rtol=1e-2, atol=1e-2):
                    row[name] = {"error": "incorrect"}
                    row_str += f" | {'ERROR':>12}"
                    continue
                
                # Benchmark
                mean_ms, min_ms = benchmark(fn, x, warmup, iters)
                gbps = throughput_gbps(m, n, mean_ms)
                
                row[name] = {
                    "mean_ms": round(mean_ms, 4),
                    "min_ms": round(min_ms, 4),
                    "gbps": round(gbps, 1)
                }
                row_str += f" | {mean_ms:>8.4f} ms"
                
            except Exception as e:
                row[name] = {"error": str(e)}
                row_str += f" | {'ERROR':>12}"
        
        print(row_str)
        results[shape_str] = row
        
        del x
        torch.cuda.empty_cache()
    
    return results


def print_summary(results: Dict):
    """Print throughput summary."""
    print("\n" + "=" * 70)
    print("THROUGHPUT (GB/s) - Higher is Better")
    print("=" * 70)
    
    # Get kernel names from first result
    first_result = next(iter(results.values()))
    kernel_names = [k for k in first_result.keys() if k not in ["shape", "elements"]]
    
    header = f"{'Shape':<18}"
    for name in kernel_names:
        header += f" | {name:>12}"
    header += " | Winner"
    print(header)
    print("-" * (30 + 15 * len(kernel_names)))
    
    for shape, data in results.items():
        row_str = f"{shape:<18}"
        best_gbps = 0
        winner = ""
        
        for name in kernel_names:
            if name in data and isinstance(data[name], dict) and "gbps" in data[name]:
                gbps = data[name]["gbps"]
                row_str += f" | {gbps:>10.1f}"
                if gbps > best_gbps:
                    best_gbps = gbps
                    winner = name
            else:
                row_str += f" | {'-':>12}"
        
        row_str += f" | {winner}"
        print(row_str)
    
    print("=" * 70)
    print(f"H100 HBM3 Peak: 3,350 GB/s")


def main():
    parser = argparse.ArgumentParser(description="Softmax kernel benchmarks")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=200, help="Benchmark iterations")
    parser.add_argument("--save", type=str, default="benchmark_results.json", help="Output file")
    args = parser.parse_args()
    
    # Test shapes
    shapes = [
        (32, 128),        # 4K elements - tiny
        (256, 1024),      # 256K elements - small
        (1024, 1024),     # 1M elements - medium
        (2048, 2048),     # 4M elements - medium-large
        (4096, 4096),     # 16M elements - large
        (1024, 50257),    # 51M elements - GPT-2 vocab
        (4096, 32000),    # 131M elements - LLaMA vocab
    ]
    
    print(f"\nWarmup: {args.warmup}, Iterations: {args.iters}")
    
    results = run_benchmarks(shapes, args.warmup, args.iters)
    print_summary(results)
    
    # Save results
    with open(args.save, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.save}")


if __name__ == "__main__":
    main()
