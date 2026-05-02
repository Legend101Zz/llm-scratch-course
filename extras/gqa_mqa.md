# MQA & GQA — Cheaper KV Cache

> Goal: shrink the KV cache without hurting quality. KV cache size dominates inference memory at long contexts.

## Quick refresher

In standard MHA, each of the $h$ heads has its own $W_K, W_V$. So at inference, the KV cache stores $2 \cdot L \cdot h \cdot T \cdot d_h$ values (layers × heads × seq len × head dim, ×2 for K and V).

For a 7B-class model: 32 layers × 32 heads × 128 head_dim × 4096 tokens × 2 (K&V) × 2 bytes (fp16) ≈ **2 GB just for one user's cache**. Multiply by concurrent users and you see why LLM serving needs massive GPUs.

## MQA — Multi-Query Attention (Shazeer, 2019)

Share **one** K and V across all heads. Each head still has its own Q.

```
MHA:   h pairs of (W_K, W_V)            cache size: O(h * T)
MQA:   1 pair of (W_K, W_V) total       cache size: O(T)        h× smaller
```

Trade-off: slight quality hit. PaLM and Falcon used MQA.

## GQA — Grouped-Query Attention (Ainslie et al., 2023)

Compromise: $g$ groups of heads, each group shares K, V.

```
MHA:  h K/V pairs   (g = h)
GQA:  g K/V pairs    (1 < g < h)        cache size: O(g * T)
MQA:  1 K/V pair    (g = 1)
```

LLaMA-2 70B uses GQA with $g=8$. Quality near MHA, KV cache near MQA. Now the standard.

## In code (sketch)

```python
class GQA(nn.Module):
    def __init__(self, d, n_q_heads, n_kv_heads):
        ...
        self.q = nn.Linear(d, n_q_heads * d_h, bias=False)
        self.k = nn.Linear(d, n_kv_heads * d_h, bias=False)   # smaller
        self.v = nn.Linear(d, n_kv_heads * d_h, bias=False)   # smaller
        self.repeat = n_q_heads // n_kv_heads
    def forward(self, x):
        ...
        # repeat K, V along head axis to match Q
        k = k.repeat_interleave(self.repeat, dim=1)
        v = v.repeat_interleave(self.repeat, dim=1)
        # then standard scaled-dot-product attention
```

PyTorch's `F.scaled_dot_product_attention` handles GQA automatically when K, V have fewer heads than Q.

## Why this matters in practice

- Long-context models (100k+ tokens) are basically *only* possible with GQA/MQA + paged attention.
- vLLM, TGI, and other serving frameworks lean heavily on these.
