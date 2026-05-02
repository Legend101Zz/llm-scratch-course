# Module 5 — Full Transformer Block in Numpy (40 min)

> ⏱️ 40 minutes. We assemble: LayerNorm → MHA → residual → LayerNorm → FFN → residual. This is one **block**. Stack $N$ of these and you have GPT.

## The block (GPT-2 style "pre-norm")

```
x ── LayerNorm ── MHA ──┐
│                       │
└─────── + ─────────────┘   ← residual
            │
            └── LayerNorm ── FFN ──┐
                  │                │
                  └────── + ───────┘   ← residual
                            │
                            ▼ (to next block)
```

Why pre-norm (norm BEFORE the sublayer) and not post-norm (the original 2017 paper's order)?
Empirically pre-norm trains more stably for deep stacks. GPT-2/3, LLaMA, etc. all use pre-norm.

### Why residuals?

Without residuals, gradients have to flow back through dozens of non-linear layers — they vanish or explode. With $y = x + f(x)$, the gradient flowing back includes a clean `+1` path:

$$\frac{\partial y}{\partial x} = 1 + \frac{\partial f}{\partial x}$$

So even if $\partial f/\partial x$ is small, the `1` keeps the gradient alive. **Residuals = gradient highway.**

### LayerNorm vs BatchNorm

LayerNorm normalizes across the **feature** dim (per-token). BatchNorm normalizes across the **batch** dim. LN works for variable sequence lengths and tiny batches → standard for transformers.

$$\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

$\gamma, \beta$ are learnable per-feature scale/shift.

### The FFN

Two linear layers with a non-linearity (GELU in GPT-2, ReLU originally). Inner width is **4×** the embedding dim. This is where most of the model's parameters and FLOPs live.

$$\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$$

## Build it

Open [`starter.py`](starter.py). Implement `LayerNorm`, `FFN`, and `TransformerBlock`. Run forward pass and check shapes.

We won't backprop in numpy (would re-derive everything painfully). The next module switches to PyTorch and we get autograd back.

## Reflection

- What would happen if you removed the residual connection?
- Why is the FFN width usually 4× the embedding dim? (No deep theoretical reason; empirical sweet spot. Mention this in interviews.)
- In pre-norm, the FINAL output is unnormalized. Where do most implementations add a final LayerNorm?

✅ Next: [Module 6 — PyTorch Crash](../06_pytorch_crash/README.md). PyTorch will give us autograd + GPU for free.
