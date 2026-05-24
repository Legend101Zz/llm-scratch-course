# Module 5 — Full Transformer Block in Numpy (40 min)

> ⏱️ 40 minutes. We assemble: LayerNorm → MHA → residual → LayerNorm → FFN → residual. This is one **block**. Stack $N$ of these and you have GPT.
>
> 📄 Reference: `papers/1706.03762v7.pdf` Section 3 (full architecture). The 2017 paper uses post-norm; GPT-2 onward switched to pre-norm — we follow the modern convention.

## 1. What a Block Is, Conceptually

In Module 4 we built **attention** — a way for tokens to share information. But attention alone isn't enough. A transformer block is attention **plus** a per-token computation **plus** the plumbing (norms, residuals) that keeps deep stacks trainable.

Two sublayers, in order:

1. **Multi-Head Attention (MHA)** — tokens *talk to each other*. The "communication" step.
2. **Feed-Forward Network (FFN)** — each token *thinks alone*. The "computation" step. No mixing across positions; just a 2-layer MLP applied independently to every token vector.

> 🧠 Mental model: **attention shuffles information across positions; the FFN processes it within each position**. A transformer block alternates these — gather, then digest, gather, then digest. Stack of blocks = repeated rounds of this.

This is also why people call the running stream of token vectors the **"residual stream"**: each block reads from it, computes something, and writes back via the residual add. The stream is a kind of working memory that gets progressively refined by every block.

## 2. The Block Diagram (GPT-2 / modern "pre-norm")

```
        x  (B, T, d)
        │
        ├──────────────────────────┐
        ▼                          │
    LayerNorm                      │  residual
        │                          │  bypass
        ▼                          │
       MHA                         │
        │                          │
        ▼                          │
     ( + )◄─────────────────────────┘
        │
        ├──────────────────────────┐
        ▼                          │
    LayerNorm                      │  residual
        │                          │  bypass
        ▼                          │
       FFN                         │
        │                          │
        ▼                          │
     ( + )◄─────────────────────────┘
        │
        ▼   (to next block)
```

In code this is just:

```python
x = x + MHA(LN1(x))
x = x + FFN(LN2(x))
```

Five lines. Stack $N$ of these (N=12 for GPT-2 small, N=96 for GPT-3) and you have a language model.

## 3. Pre-Norm vs. Post-Norm — Why It Matters

The original 2017 paper put LayerNorm **after** the sublayer (post-norm):

```
post-norm:  x = LN(x + Sublayer(x))
pre-norm:   x = x + Sublayer(LN(x))
```

Looks identical. It is not. **Pre-norm is dramatically easier to train deep**, and every modern LLM (GPT-2/3, LLaMA, Mistral, DeepSeek, Claude, Gemini) uses it.

**Why?** With pre-norm, the residual path from input to output is a clean, unmodified identity. Gradients flow back along it without ever passing through a normalization (which would scale them). With post-norm, every residual stage's gradient gets multiplied by the LayerNorm Jacobian — small effects compound across 96 layers and training becomes unstable, requiring careful learning-rate warmup.

> Pre-norm preserves the "gradient highway." Post-norm puts speed bumps on it.

## 4. Why Residuals? (The Gradient Highway)

Without residuals, gradients have to flow back through dozens of non-linear layers — they vanish (or explode). With $y = x + f(x)$, gradient backflow is:

$$\frac{\partial y}{\partial x} = 1 + \frac{\partial f}{\partial x}$$

Even if $\partial f/\partial x$ is small (deep network, saturating activation, whatever), the $+1$ keeps the gradient alive. **Residuals = gradient highway.**

A second, equally important benefit: residual connections give every layer the option to **do nothing**. If the optimal thing for a layer is identity, it just learns $f(x) \approx 0$ and the residual delivers $x$ unchanged. Adding more layers without residuals can *hurt* — adding more layers *with* residuals is at worst neutral. This is what makes "deeper is better" actually achievable.

> The residual stream perspective: think of $x$ as a running write-buffer that each sublayer **adds to**, never overwrites. Layers communicate by reading from and writing to this shared stream.

## 5. LayerNorm — and Why Not BatchNorm

LayerNorm normalizes across the **feature** dimension, *per token*. For a token vector $x \in \mathbb{R}^d$:

$$\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where $\mu, \sigma^2$ are the mean and variance computed over the $d$ features of *that one token*, and $\gamma, \beta \in \mathbb{R}^d$ are learnable per-feature scale and shift.

**Why normalize at all?** Activations drift in scale across layers (some get bigger, some smaller). Without normalization, the model has to learn weights that compensate for arbitrary scale changes — wasted capacity. Normalizing forces every input to a sublayer onto a consistent scale. The learnable $\gamma, \beta$ then let the model *un-normalize* whenever that's actually useful.

**Why LayerNorm, not BatchNorm?**
- BatchNorm normalizes each feature *across the batch*. Two problems for transformers: (a) you need a big batch for stable statistics; (b) at inference the batch can be size 1, and inference behavior diverges from training.
- LayerNorm needs no batch statistics — its computation is the same at train and test time, and works for any batch size.
- Sequence lengths in NLP are variable; BatchNorm with padding becomes a mess.

**RMSNorm** (used by LLaMA, Mistral) is a simplification: drop the mean-subtraction and the bias, keep only the scale. Slightly faster, slightly worse on small models, neutral on large ones. The principle is the same.

## 6. The FFN — Per-Token "Thinking"

After attention has mixed information across positions, the FFN processes each token's vector *independently*:

$$\text{FFN}(x) = \text{GELU}(x W_1 + b_1)\, W_2 + b_2$$

Shapes: $W_1 \in \mathbb{R}^{d \times 4d}$, $W_2 \in \mathbb{R}^{4d \times d}$. The hidden width is conventionally **4×** the embedding dim.

### Why 4×?

No deep theoretical reason. Empirically, 4× hits a sweet spot of capacity vs. cost. Try smaller — the model underfits. Try bigger — diminishing returns and big memory cost. 4× has been the default since 2017 and nothing has dethroned it.

### What does the FFN actually do?

There's a useful "key-value memory" view (Geva et al., 2020): the first matrix $W_1$ projects the token vector to "keys" of size $4d$; GELU keeps only the strong activations; $W_2$ writes back a learned linear combination of "values." It looks a lot like an associative lookup. Empirically, individual neurons in $W_1$ have been found to fire on specific patterns ("French sentences," "code," "names of capital cities") — they're like memory cells.

This matters because:

- **Most parameters live in the FFN.** For GPT-2 small: ~25% of params are attention, ~67% are FFN. FLOPs split similarly. When you optimize a transformer, the FFN is the budget-eater.
- **MoE (Mixture of Experts)** replaces this single dense FFN with many specialized experts that get routed to per-token. That's the difference between a 70B dense model and a 8×7B MoE: MoE only activates ~12B params per token but has the knowledge capacity of much more.

### Why GELU and not ReLU?

ReLU has zero gradient for negative inputs — neurons can "die." GELU is a smooth approximation that keeps a small gradient flowing for slightly-negative inputs. Marginal but consistent improvement on transformers. LLaMA uses **SwiGLU**, a small variant that's empirically a bit better still — same idea.

```
   ReLU(x):     ___/        zero below 0, kink at 0
   GELU(x):    __/⌒/        smooth around 0, looks like ReLU far from 0
```

## 7. Putting It Together — A Walked-Through Forward Pass

```python
def block_forward(x, params):
    # x: (B, T, d)
    
    # --- attention sublayer ---
    a = layer_norm(x, params.ln1)        # (B, T, d) — normalize per token
    a = multi_head_attention(a, params)  # (B, T, d) — tokens share info
    x = x + a                            # (B, T, d) — residual add

    # --- FFN sublayer ---
    f = layer_norm(x, params.ln2)        # (B, T, d) — normalize again
    f = ffn(f, params)                   # (B, T, d) — per-token transform
    x = x + f                            # (B, T, d) — residual add

    return x
```

Shape never changes through a block: in `(B, T, d)`, out `(B, T, d)`. This is what makes stacking trivial — block N's output is block N+1's input, no reshaping.

## 8. The Full Transformer LM (preview)

```
tokens (B, T)
   │
   │ token embedding (lookup) + position embedding
   ▼
x  (B, T, d)
   │
   │ N × { LN → MHA → +,  LN → FFN → + }
   ▼
x  (B, T, d)
   │
   │ final LayerNorm
   ▼
   │ projection to vocab (linear, often weight-tied to token embedding)
   ▼
logits (B, T, vocab_size)
```

That's the whole architecture. Module 7 will turn this picture into actual PyTorch code.

> 📌 **Final LayerNorm.** In pre-norm models, the residual stream output of the last block is *not* normalized (no LN comes after it). So implementations always tack on one more LayerNorm before the final logit projection. Easy to forget; subtly important.

## 9. Build it

Open [`starter.py`](starter.py). Implement:
1. `LayerNorm` — mean, variance, normalize, scale-and-shift.
2. `FFN` — two linears with GELU between them.
3. `TransformerBlock` — wire LN → MHA → residual → LN → FFN → residual.

Run a forward pass and check shapes. Input `(B, T, d)` → output `(B, T, d)`.

We won't backprop in numpy (would re-derive painfully). The next module switches to PyTorch and we get autograd back.

## 10. Common Bugs

| Symptom | Cause | Fix |
|---|---|---|
| Output shape wrong | Forgot residual is *added*, not concatenated | `x + sublayer(x)`, not `concat([x, sublayer(x)])` |
| LayerNorm produces NaN | Variance is 0, divided by sqrt(0) | Add the $\epsilon$ inside the sqrt |
| Loss explodes after a few steps | Used post-norm with no LR warmup | Switch to pre-norm OR add warmup |
| Final logits look untrained even at end | Forgot the final LayerNorm before the lm_head | Add `self.ln_f = LayerNorm(d)` and apply before projection |
| FFN dominates memory | This is normal; 4× expansion is the cost of capacity | Use mixed precision, gradient checkpointing |

## 11. Reflection

1. **Remove residuals.** What happens to gradient magnitudes after 12 layers without residuals? (Compute symbolically: each layer multiplies the gradient by some Jacobian. Compounded.)
2. **4× rule.** Why is the FFN inner width 4× the embedding dim? (No deep reason; empirical sweet spot. Mention this in interviews — it's a famous "no theory, just works" choice.)
3. **Final LayerNorm.** In pre-norm, the FINAL output is unnormalized. Where do most implementations add the missing LayerNorm? Why is it easy to forget?
4. **Communication vs. computation.** Map each part of the block to either "tokens talking to each other" or "tokens thinking alone." Which is which? Could you reorder them?
5. **Param count.** For $d = 768$, 12 heads, vocab = 50257, 12 blocks — roughly how many params? Where do most of them live? (Hint: it's the FFN. Compute $W_1, W_2$ counts vs. attention's $W_Q, W_K, W_V, W_O$.)

## 12. What's NOT in this module (radar)

- **RMSNorm** instead of LayerNorm (LLaMA, Mistral) — see `extras/`.
- **SwiGLU** instead of GELU+linear FFN (LLaMA) — same shape role, slightly better.
- **Parallel blocks** (PaLM, GPT-J) — run MHA and FFN in parallel and add both, instead of in series. Slight speedup, comparable quality.
- **MoE FFNs** — replace the dense FFN with sparsely-activated experts. See `extras/06_moe.md`.

The block structure has been remarkably stable since 2018. Most "modern" tweaks are 5-10% improvements that compound at scale.

✅ Next: [Module 6 — PyTorch Crash](../06_pytorch_crash/README.md). PyTorch will give us autograd + GPU for free.
