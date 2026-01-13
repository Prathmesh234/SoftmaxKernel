Vision Encoder Path:

### 1. Patch Embedding
- Already discussed. Input preparation.
- **Key Insight**: Kernels are stateless execution engines. We can leverage pre-trained VLM models (CLIP, SigLIP) by directly loading their weights into these kernels. This avoids training from scratch while achieving optimized inference performance.

### 2. Multi-Head Attention (MHA) - Core bottleneck. Operations:
- Q, K, V projections (3 GEMMs)
- Scaled dot-product: softmax(QK^T / sqrt(d)) @ V
- Output projection (1 GEMM)
- Fuse QKV projection into single kernel for bandwidth savings

### 3. Layer Normalization - Runs after every attention/MLP block:
- Compute mean/variance across embedding dim
- Normalize: (x - mean) / sqrt(var + eps)
- Scale and shift with learned parameters
- Fuse with residual addition

### 4. MLP/FFN Block - Two linear layers with activation:
- GELU(xW1 + b1)W2 + b2
- Fuse GELU with first GEMM output

## TODO List
- [ ] **Pre-trained Weight Loading**: Show how to load HuggingFace CLIP/SigLIP weights and pass them into the `PatchEmbedding` kernels.
- [ ] **Kernel Verification**: Compare output of custom Triton/Helion kernels against a pre-trained CLIP model to ensure numerical parity.
- [ ] **MHA Implementation**: High-performance MHA kernel for H100 (FlashAttention-style).