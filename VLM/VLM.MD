Vision Encoder Path:

Patch Embedding - Already discussed. Input preparation.
Multi-Head Attention (MHA) - Core bottleneck. Operations:

Q, K, V projections (3 GEMMs)
Scaled dot-product: softmax(QK^T / sqrt(d)) @ V
Output projection (1 GEMM)
Fuse QKV projection into single kernel for bandwidth savings


Layer Normalization - Runs after every attention/MLP block:

Compute mean/variance across embedding dim
Normalize: (x - mean) / sqrt(var + eps)
Scale and shift with learned parameters
Fuse with residual addition


MLP/FFN Block - Two linear layers with activation:

GELU(xW1 + b1)W2 + b2
Fuse GELU with first GEMM output