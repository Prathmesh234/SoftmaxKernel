"""
=================================================
SOFTMAX KERNELS - TRITON
=================================================
Fused Triton softmax kernels for H100 GPU.
Target: NVIDIA H100 SXM 80GB HBM3
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused softmax kernel - all operations in one pass.
    
    Each program handles one row:
    1. Load row into SRAM
    2. Compute max (numerical stability)
    3. Subtract max and exponentiate
    4. Sum exponentials
    5. Divide to normalize
    6. Write back to DRAM
    
    Memory: read MN, write MN (2x better than naive!)
    """
    row_idx = tl.program_id(0)
    
    # Compute row pointers
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    # Mask for valid columns (BLOCK_SIZE may be > n_cols)
    mask = col_offsets < n_cols
    
    # Load row into SRAM
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # Numerical stability: subtract max
    row_minus_max = row - tl.max(row, axis=0)
    
    # Softmax: exp and normalize
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    # Write back to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Triton softmax wrapper.
    
    Args:
        x: Input tensor [M, N]
    Returns:
        Softmax output [M, N]
    """
    n_rows, n_cols = x.shape
    
    # Block size must be power of 2 and >= n_cols
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 65536)  # Hardware limit
    
    # Allocate output
    y = torch.empty_like(x)
    
    # Launch: one program per row
    _softmax_kernel[(n_rows,)](
        y, x,
        x.stride(0), y.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    return y


# Default export
softmax = triton_softmax
