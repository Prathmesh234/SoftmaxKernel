# Test 1: B200 Helion Flash Attention Benchmark

**Date:** 2026-01-24T20:25:06  
**GPU:** NVIDIA B200  
**Dtype:** torch.bfloat16  
**Warmup:** 25 iterations  
**Benchmark:** 100 iterations  

---

## Configuration

```python
@helion.kernel(
    config=helion.Config(
        block_sizes=[1, 128, 128],
        indexing=['pointer', 'pointer', 'pointer', 'pointer'],
        l2_groupings=[2],
        load_eviction_policies=['first', 'last', 'first'],
        loop_orders=[[1, 0]],
        num_stages=2,
        num_warps=4,
        pid_type='flat',
        range_flattens=[None, True],
        range_multi_buffers=[None, False],
        range_num_stages=[0, 1],
        range_unroll_factors=[0, 1],
        range_warp_specializes=[None, None],
    ),
    static_shapes=True,
)
```

**Autotuning:** 816 configs searched in 661 seconds

---

## Results

| Config [B, H, S, D] | Helion (ms) | PyTorch (ms) | Speedup | TFLOPS | Max Diff |
|---------------------|-------------|--------------|---------|--------|----------|
| [2, 32, 1024, 128] | 0.1448 | 0.1234 | 0.85x | 237.3 | 1.95e-03 |
| [4, 32, 2048, 128] | 0.7286 | 0.7308 | **1.00x** | 377.2 | 9.77e-04 |
| [4, 32, 4096, 128] | 2.7169 | 2.8052 | **1.03x** | 404.7 | 9.77e-04 |
| [8, 32, 2048, 128] | 1.4440 | 1.4330 | 0.99x | 380.7 | 1.95e-03 |

---

## Summary

| Metric | Value |
|--------|-------|
| **Peak TFLOPS** | 404.7 |
| **Avg TFLOPS** | 350.0 |
| **Best Speedup** | 1.03x vs PyTorch |
| **Avg Speedup** | 0.97x vs PyTorch |
| **B200 BF16 Peak** | ~2250 TFLOPS |
| **Utilization** | 18.0% |

---

## Key Observations

1. **Large sequences perform best**: The 4096 sequence length achieves 1.03x speedup over PyTorch SDPA
2. **128x128 tile size**: Matches FlashAttention-4's recommended configuration for B200
3. **Competitive performance**: At parity or better than PyTorch on larger shapes
4. **Numerical accuracy**: Max difference < 2e-3, well within FP16/BF16 tolerance

---

## Next Steps

- [ ] Try warp specialization for further optimization
- [ ] Test with FP8 precision
- [ ] Explore persistent kernel configurations
- [ ] Benchmark against FlashAttention-3
