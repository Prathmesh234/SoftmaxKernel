"""
=================================================
HELION FLASH ATTENTION BENCHMARK
=================================================
Benchmarks the Helion Flash Attention kernel on B200 GPU.
Usage: uv run python benchmark_helion.py
"""

import torch
import json
import argparse
from datetime import datetime

# Setup
DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16

print("=" * 70)
print("HELION FLASH ATTENTION BENCHMARK - NVIDIA B200")
print("=" * 70)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"dtype: {DTYPE}")
print("=" * 70)

# Import our kernel
from b200_attention import flash_attention


def benchmark(
    fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    warmup: int = 25,
    iters: int = 100
) -> tuple[float, float]:
    """
    Accurate GPU benchmarking with CUDA events.
    
    Returns: (mean_ms, min_ms)
    """
    # Warmup
    for _ in range(warmup):
        _ = fn(q, k, v)
    torch.cuda.synchronize()
    
    # Benchmark using CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(iters):
        start.record()
        _ = fn(q, k, v)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    # Remove outliers
    times = sorted(times)[5:-5] if len(times) > 10 else times
    return sum(times) / len(times), min(times)


def compute_tflops(batch: int, heads: int, seq_len: int, head_dim: int, time_ms: float) -> float:
    """Calculate TFLOPS for attention."""
    # Attention FLOPS: 4 * B * H * M * N * D (2 for QK, 2 for PV)
    flops = 4 * batch * heads * seq_len * seq_len * head_dim
    return flops / (time_ms / 1000) / 1e12


def run_benchmarks(configs: list, warmup: int = 25, iters: int = 100) -> dict:
    """Run benchmarks on all configurations."""
    
    results = {}
    
    # Header
    print(f"\n{'Config':<30} | {'Helion (ms)':>12} | {'PyTorch (ms)':>12} | {'Speedup':>8} | {'TFLOPS':>8}")
    print("-" * 85)
    
    for batch, heads, seq_len, head_dim in configs:
        config_str = f"[{batch}, {heads}, {seq_len}, {head_dim}]"
        
        # Create tensors
        q = torch.randn(batch, heads, seq_len, head_dim, device=DEVICE, dtype=DTYPE)
        k = torch.randn(batch, heads, seq_len, head_dim, device=DEVICE, dtype=DTYPE)
        v = torch.randn(batch, heads, seq_len, head_dim, device=DEVICE, dtype=DTYPE)
        
        # Reference (PyTorch SDPA)
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        
        try:
            # Verify correctness
            out = flash_attention(q, k, v)
            max_diff = (out - ref).abs().max().item()
            
            if max_diff > 0.1:
                print(f"{config_str:<30} | {'ERROR: incorrect':>35}")
                results[config_str] = {"error": f"max_diff={max_diff:.4f}"}
                continue
            
            # Benchmark Helion
            helion_mean, helion_min = benchmark(flash_attention, q, k, v, warmup, iters)
            
            # Benchmark PyTorch
            pytorch_mean, pytorch_min = benchmark(
                torch.nn.functional.scaled_dot_product_attention, 
                q, k, v, warmup, iters
            )
            
            # Calculate metrics
            speedup = pytorch_mean / helion_mean
            tflops = compute_tflops(batch, heads, seq_len, head_dim, helion_mean)
            
            results[config_str] = {
                "helion_mean_ms": round(helion_mean, 4),
                "helion_min_ms": round(helion_min, 4),
                "pytorch_mean_ms": round(pytorch_mean, 4),
                "pytorch_min_ms": round(pytorch_min, 4),
                "speedup": round(speedup, 2),
                "tflops": round(tflops, 2),
                "max_diff": round(max_diff, 6),
            }
            
            print(f"{config_str:<30} | {helion_mean:>10.4f} | {pytorch_mean:>10.4f} | {speedup:>7.2f}x | {tflops:>7.1f}")
            
        except Exception as e:
            print(f"{config_str:<30} | ERROR: {str(e)[:40]}")
            results[config_str] = {"error": str(e)}
        
        # Cleanup
        del q, k, v
        torch.cuda.empty_cache()
    
    return results


def print_summary(results: dict):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    valid_results = [r for r in results.values() if "tflops" in r]
    
    if not valid_results:
        print("No valid results!")
        return
    
    tflops_values = [r["tflops"] for r in valid_results]
    speedup_values = [r["speedup"] for r in valid_results]
    
    print(f"Peak TFLOPS:     {max(tflops_values):.1f}")
    print(f"Avg TFLOPS:      {sum(tflops_values)/len(tflops_values):.1f}")
    print(f"Best Speedup:    {max(speedup_values):.2f}x vs PyTorch")
    print(f"Avg Speedup:     {sum(speedup_values)/len(speedup_values):.2f}x vs PyTorch")
    print("=" * 70)
    print(f"B200 BF16 Peak:  ~2250 TFLOPS")
    print(f"Utilization:     {max(tflops_values)/2250*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Helion Flash Attention benchmark")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--save", type=str, default="benchmark_results.json", help="Output file")
    args = parser.parse_args()
    
    # Test configurations: (batch, heads, seq_len, head_dim)
    # B200-optimized shapes - larger configs that leverage high bandwidth
    configs = [
        (2, 32, 1024, 128),     # Medium - LLaMA-like
        (4, 32, 2048, 128),     # Large
        (4, 32, 4096, 128),     # Very large
        (8, 32, 2048, 128),     # Large batch
        (16, 64, 4096, 128),    # Massive (Autotuned Target)
        (8, 64, 8192, 128),     # Extreme (Long Context 8k)
    ]
    
    print(f"\nWarmup: {args.warmup}, Iterations: {args.iters}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    results = run_benchmarks(configs, args.warmup, args.iters)
    print_summary(results)
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "dtype": str(DTYPE),
        "warmup": args.warmup,
        "iters": args.iters,
        "results": results,
    }
    
    with open(args.save, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {args.save}")


if __name__ == "__main__":
    main()
