# FlashAttention — the kernel that powers every modern LLM

> Same math as standard attention. Just computed without ever materializing the $T \times T$ attention matrix in HBM (GPU global memory).

## The bottleneck nobody told you about

Modern GPU FLOPs are basically free; **memory bandwidth** is the bottleneck. Standard attention does:

1. Compute $S = QK^T$ → write $T \times T$ matrix to HBM.
2. Read it back, compute softmax → write to HBM.
3. Read it back, multiply by V → write output to HBM.

For long $T$, you spend most time waiting on memory transfers. The $T \times T$ matrix is the killer (4096-token context = 16M entries per head, per layer).

## What FlashAttention does

Tiling + online softmax: it processes Q, K, V in blocks that fit in **on-chip SRAM**. Computes the softmax incrementally across blocks (running max + running denominator trick). Never writes the full $T \times T$ matrix anywhere.

Result: same numerical output, **2-4× faster wall-clock**, and memory usage drops from $O(T^2)$ to $O(T)$. So you can train with longer sequences on the same hardware.

## How you actually use it

```python
import torch.nn.functional as F
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

PyTorch picks FlashAttention 2 (Ampere+, e.g. T4, A100, H100) or memory-efficient kernels otherwise — automatically. You almost never call it manually.

## The "online softmax" trick (sketch)

The hard part of tiling attention is softmax — naive softmax needs the global max for numerical stability. FlashAttention keeps a running max $m$ and running sum $\ell$, updating both as it processes each new block of K, V:

$$m_\text{new} = \max(m_\text{old}, m_\text{block}),\quad \ell_\text{new} = e^{m_\text{old} - m_\text{new}} \ell_\text{old} + e^{m_\text{block} - m_\text{new}} \ell_\text{block}$$

Beautiful. Don't memorize, just appreciate.

## See also

- Dao et al., *FlashAttention*, 2022.
- *FlashAttention-2*, 2023.
- *FlashAttention-3* (H100-specific FP8 variant), 2024.
