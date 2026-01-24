# B200 Triton Flash Attention

A simple, clean Flash Attention implementation in Triton optimized for NVIDIA B200 (Blackwell) GPUs.

## Features

- **Flash Attention Algorithm**: Memory-efficient attention without materializing the full N×N matrix
- **Online Softmax**: Numerically stable computation using running max and sum
- **Causal Masking**: Optional causal masking for autoregressive models
- **B200 Optimizations**: Larger block sizes and more pipeline stages for Blackwell architecture

## Installation

```bash
# Using uv
uv sync

# Or using pip
pip install -e .
```

## Usage

```python
import torch
from flash_attention import flash_attention

# Create inputs
batch, heads, seq_len, head_dim = 4, 32, 2048, 64
q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
k = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
v = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)

# Run attention
output = flash_attention(q, k, v, causal=True)
```

## Testing

```bash
python flash_attention.py
```

This will run correctness tests against PyTorch reference implementation and benchmark the kernel.

## API

### `flash_attention(q, k, v, causal=True, sm_scale=None)`

**Arguments:**
- `q`: Query tensor `[batch, heads, seq_len, head_dim]`
- `k`: Key tensor `[batch, heads, seq_len, head_dim]`
- `v`: Value tensor `[batch, heads, seq_len, head_dim]`
- `causal`: Apply causal masking (default: `True`)
- `sm_scale`: Softmax scaling factor (default: `1/sqrt(head_dim)`)

**Returns:**
- Output tensor `[batch, heads, seq_len, head_dim]`

## B200 Optimizations

The kernel automatically detects B200 GPUs and applies:
- Larger block sizes (128×128)
- More pipeline stages (4 vs 2)
- More warps (8 vs 4)

## License

MIT
