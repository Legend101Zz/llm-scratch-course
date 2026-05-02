# Module 7 — Mini-GPT in PyTorch (60 min)

> ⏱️ 60 minutes. Re-build everything from Modules 4-5 in PyTorch, then stack it into a real GPT.
>
> 📄 Skim `papers/language_models_are_unsupervised_multitask_learners.pdf` (GPT-2) — focus on Section 2 (Approach) and Table 2 (model sizes).

## What we're building

A nano-GPT: roughly the same architecture as GPT-2, but tiny (so it runs on CPU/Colab T4). Specs:

```
vocab_size = 65 (chars in tinyshakespeare)  or 50257 (BPE)
block_size = 128         # context length
n_embd     = 192
n_head     = 6
n_layer    = 6
dropout    = 0.2
```

About **1M parameters**. GPT-2 was 124M (smallest) — 100× bigger, same shape.

## Key design choices to internalize

### Token + position embeddings (added)

```python
tok_emb = nn.Embedding(vocab_size, n_embd)      # what is this token
pos_emb = nn.Embedding(block_size, n_embd)      # where is it in the sequence
x = tok_emb(idx) + pos_emb(positions)           # SUMMED, not concatenated
```

Why position embeddings? Pure attention is permutation-invariant — `("the", "cat")` and `("cat", "the")` would attend identically. We inject position info via a learned vector per position. Modern models use **RoPE** (rotary embeddings); GPT-2 uses learned absolute. We use the GPT-2 way for simplicity.

### Causal attention with the registered buffer

```python
self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
# in forward: scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
```

`register_buffer` means: not a parameter (no gradients), but moves with the model to GPU.

### Weight init

```python
nn.init.normal_(p, mean=0.0, std=0.02)   # GPT-2's default for most weights
nn.init.zeros_(p)                        # for biases
```

Smaller std for residual-projection weights (`std=0.02/sqrt(2*n_layer)`) — keeps activations stable as the stack deepens. Andrej discusses this.

### Generating text (autoregressive sampling)

```python
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]                      # crop context
        logits = model(idx_cond)[:, -1, :] / temperature     # last position
        if top_k is not None:
            v, _ = logits.topk(top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nxt], dim=1)
    return idx
```

Temperature < 1 → sharper (more deterministic). Top-k = sample only from k most likely. Top-p (nucleus) is similar.

## Build it

Open [`gpt.py`](gpt.py) — the starter has the file structure laid out, you fill in:
1. `CausalSelfAttention.__init__` and `.forward`
2. `Block.__init__` and `.forward`
3. `GPT.__init__` and `.forward`
4. The `generate` function

Run a quick CPU sanity check at the bottom — it should produce gibberish (untrained) but with correct shapes.

Then move to **Module 8 to actually train it on Colab**.

## ⚡ Modern PyTorch shortcut: `F.scaled_dot_product_attention`

Everything in our `CausalSelfAttention.forward` (matmul, scale, mask, softmax, dropout, matmul) is one fused PyTorch op since 2.0:

```python
import torch.nn.functional as F
y = F.scaled_dot_product_attention(
    q, k, v,                         # (B, h, T, d_h)
    dropout_p=self.dropout if self.training else 0.0,
    is_causal=True,                  # auto-applies the lower-triangular mask
)
```

Under the hood it picks the fastest kernel: **FlashAttention** on Ampere+ GPUs (A100/T4 mostly), memory-efficient attention otherwise. **2-4× faster** and uses far less memory than the manual version, *because it never materializes the full T×T attention matrix*. We keep the manual code for transparency in `gpt.py`, but in real code you should use the fused op.

## 🚀 Inference: the KV cache (the trick that makes ChatGPT fast)

Naive autoregressive generation: generate token $t+1$ → feed entire sequence $[1..t+1]$ back through all layers → recompute K, V for every past position. That's $O(T^2)$ work per token, $O(T^3)$ for the full response. Brutal.

**KV cache:** during generation, store K and V tensors per layer per head. When generating token $t+1$, only compute Q, K, V for the *new* token, append the new K, V to the cache, and run attention against the full cached K, V. Now it's $O(T)$ per token, $O(T^2)$ total.

```python
# pseudo-code, conceptual:
class CausalSelfAttention(nn.Module):
    def forward(self, x, kv_cache=None):
        q_new = self.q(x[:, -1:])    # only the new token's Q
        k_new = self.k(x[:, -1:])
        v_new = self.v(x[:, -1:])
        if kv_cache is not None:
            k = torch.cat([kv_cache.k, k_new], dim=1)
            v = torch.cat([kv_cache.v, v_new], dim=1)
        else:
            k, v = k_new, v_new
        kv_cache.update(k, v)
        # attention only between q_new (length 1) and k, v (length T)
        ...
```

**Memory cost:** for a 7B model with 32 layers, 32 heads, head_dim 128, generating 4096 tokens, the cache is roughly `2 * L * H * T * d_h * dtype_size` ≈ several GBs. This is why long-context models are memory-hungry. **MQA** and **GQA** (see `extras/`) exist mostly to shrink this cache.

We don't implement KV cache in the starter (would muddy the educational code), but knowing it exists is non-negotiable for understanding modern LLM serving.

## Reflection

- The `(B, T, C)` convention everywhere — what is each dim?
- Why crop `idx_cond` to `block_size`? What goes wrong otherwise?
- What's the difference between `model.eval()` and `torch.no_grad()`?

✅ Next: [Module 8 — Train on Colab](../08_train_colab/README.md)
