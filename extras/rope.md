# RoPE — Rotary Position Embedding

> The position encoding scheme used by LLaMA, DeepSeek, Qwen, Mistral. Learned absolute pos embeddings (GPT-2 style) are basically obsolete in serious models.

## The idea in one paragraph

Instead of adding a position vector to the embedding (`x + pos_emb`), **rotate Q and K** by an angle proportional to position. Different feature dimensions rotate at different frequencies. The dot product between rotated Q and K then naturally encodes *relative* position — which is what the model actually wants.

## The math (worth doing on paper once)

Pair up consecutive features into 2D blocks. For position $m$ and frequency $\theta_i$:

$$\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

So we apply a 2D rotation to each pair. Different $\theta_i = 10000^{-2i/d}$ across pairs (low frequencies for late dims, high for early — same idea as sinusoidal embeds).

The magic: $\langle R_m q, R_n k \rangle = \langle q, R_{n-m} k \rangle$. The dot product depends only on the **relative** distance $n - m$, not absolute positions. This generalizes: a model trained on sequences of length 2048 can extrapolate (with tricks like NTK / YaRN scaling) to longer.

## Minimal PyTorch

```python
def rope(x, freqs):                 # x: (..., T, d), freqs: (T, d/2)
    cos = freqs.cos()[None, :, :]
    sin = freqs.sin()[None, :, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin,
                        x1 * sin + x2 * cos], dim=-1).flatten(-2)

def precompute_freqs(d, max_seq, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))   # (d/2,)
    t = torch.arange(max_seq).float()
    return torch.outer(t, inv_freq)   # (T, d/2)
```

Apply RoPE to Q and K **after the projection**, before the attention dot product. Don't rotate V.

## Why people moved away from learned absolute embeddings

- Learned embeddings can't extrapolate past training-time `block_size`.
- Token+pos addition entangles "what" and "where" in the same vector.
- RoPE encodes relative position cleanly without extra parameters.

## See also
- Su et al., *RoFormer*, 2021.
- The `LLaMA` implementation in HuggingFace `transformers/models/llama/modeling_llama.py`.
