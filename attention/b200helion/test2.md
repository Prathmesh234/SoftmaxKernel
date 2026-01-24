# Test 2: B200 Helion Flash Attention (Large Batch)

**Date:** 2026-01-24T21:35:00  
**GPU:** NVIDIA B200  
**Dtype:** torch.bfloat16  

---

## Configuration

**Autotuned for:** `[16, 64, 4096, 128]`

```python
@helion.kernel(
    config=helion.Config(
        block_sizes=[2, 32, 16],  # Small tiles found by autotuner
        indexing=['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'],
        l2_groupings=[4],
        load_eviction_policies=['last', 'last', ''],
        loop_orders=[[0, 1]],
        num_stages=8,
        num_warps=2,
        pid_type='flat',
        # ...
    ),
    static_shapes=True,
)
```

---

## Results

| Config [B, H, S, D] | Helion (ms) | PyTorch (ms) | Speedup | TFLOPS |
|---------------------|-------------|--------------|---------|--------|
| [2, 32, 1024, 128] | 0.2122 | 0.1230 | 0.58x | 161.9 |
| [4, 32, 2048, 128] | 1.0717 | 0.7299 | 0.68x | 256.5 |
| [4, 32, 4096, 128] | 3.9385 | 2.8049 | 0.71x | 279.2 |
| [8, 32, 2048, 128] | 2.0359 | 1.4325 | 0.70x | 270.0 |
| **[16, 64, 4096, 128]** | **30.5381** | **21.9485** | **0.72x** | **288.0** |

### Optimization Attempt 2: Hybrid Config (TMA + Persistent)

We attempted to use `tensor_descriptor` (TMA) and `persistent_blocked` scheduling to improve performance.

| Config | Helion (ms) | PyTorch (ms) | Speedup | TFLOPS |
|--------|-------------|--------------|---------|--------|
| [2, 32, 1024, 128] | 0.1928 | 0.1231 | 0.64x | 178.2 |
| [4, 32, 4096, 128] | 4.2923 | 2.8052 | 0.65x | 256.2 |
| **[16, 64, 4096, 128]** | **32.8050** | **21.9438** | **0.67x** | **268.1** |

**Observation:** Tuning for TMA (`tensor_descriptor`) and persistent warps actually **regressed** performance compared to the simpler pointer-based config (0.67x vs 1.03x). This suggests the B200 TMA overhead requires larger tile sizes or more careful pipeline stage tuning (e.g., barriers) to be effective for this kernel.


---

## Analysis

- **Performance Regression**: The Autotuner selected small tile sizes (`32x16`) for the massive shape, likely due to register pressure or compilation memory limits.
- **TFLOPS Drop**: Peak TFLOPS dropped from ~405 (Test 1) to 288.
- **Scaling**: The kernel scales linearly but remains ~30% slower than PyTorch SDPA (which uses FlashAttention-3/4 cuDNN backend).

## Final Validated Configuration (128x128 Tiles, 4 Warps)

After reverting to the optimal configuration found in Test 1 (`128x128` tiles, `4` warps, `flat` scheduling) and running on massive scales:

| Config [B, H, S, D] | Helion (ms) | PyTorch (ms) | Speedup | TFLOPS |
|---------------------|-------------|--------------|---------|--------|
| [2, 32, 1024, 128] | 0.1470 | 0.1235 | 0.84x | 233.8 |
| [4, 32, 2048, 128] | 0.7319 | 0.7316 | 1.00x | 375.6 |
| [4, 32, 4096, 128] | 2.7222 | 2.8071 | **1.03x** | 403.9 |
| [8, 32, 2048, 128] | 1.4363 | 1.4337 | 1.00x | 382.8 |
| **[16, 64, 4096, 128]** | **20.7709** | **21.9450** | **1.06x** | **423.5** |
| **[8, 64, 8192, 128]** | **41.6737** | **43.5296** | **1.04x** | **422.1** |

**Conclusion:**
The `128x128` tile configuration with standard pointer loading is the optimal strategy for Helion on B200 at this time. It successfully beats the highly optimized PyTorch SDPA implementation (FlashAttention-3/4 backend) on large workloads, achieving a peak of **423.5 TFLOPS** and **1.06x speedup**. The efficiency gains on 8k context length confirm the kernel's robust scaling.
