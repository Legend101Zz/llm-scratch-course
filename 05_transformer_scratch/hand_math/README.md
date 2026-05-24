# `05_transformer_scratch/hand_math/` — pencil derivations

## What belongs here for Module 5

### 1. The LayerNorm backward — derive it once

Forward:
```
μ = mean(x)            over last dim, shape (d,) → scalar per row
σ² = var(x) = mean((x - μ)²)
x_norm = (x - μ) / √(σ² + ε)
y = γ · x_norm + β
```

Derive `∂L/∂x` given `∂L/∂y` (and the corresponding grads for `γ` and `β`):
- `∂L/∂γ = ∑ ∂L/∂y · x_norm`
- `∂L/∂β = ∑ ∂L/∂y`
- `∂L/∂x_norm = ∂L/∂y · γ`
- The interesting part: `∂L/∂x` is NOT just `∂L/∂x_norm / √(σ²+ε)` — because `μ` and `σ²` *themselves depend on x*. You have to chain through that.

Show the full expression. Hint: end up with something like
`∂L/∂x_i = (1/N) · (1/√(σ²+ε)) · [N · ∂L/∂x_norm_i - ∑_j ∂L/∂x_norm_j - x_norm_i · ∑_j ∂L/∂x_norm_j · x_norm_j]`.

The intuition: a perturbation to `x_i` changes the mean and variance, which propagates back through the normalization to *every* other position. This is why LayerNorm's gradient depends on all elements — even though the forward is a simple per-row formula.

This derivation is mechanically what PyTorch does for you with `nn.LayerNorm`; doing it once on paper means you *understand* the layer rather than just *use* it.

### 2. Pre-norm vs post-norm — the gradient highway argument

Sketch the gradient flow through:

**Post-norm:** `y = LN(x + Sublayer(x))`
- Backward: gradient `∂L/∂x = ∂L/∂y · ∂LN/∂(x + Sublayer) · (I + ∂Sublayer/∂x)`
- The `∂LN/∂(·)` term *attenuates* the gradient at every layer — over many layers, this compounds. Deep post-norm transformers train less stably.

**Pre-norm:** `y = x + Sublayer(LN(x))`
- Backward: gradient `∂L/∂x = ∂L/∂y · (I + ∂Sublayer/∂(LN(x)) · ∂LN/∂x)`
- The `I` term means the residual stream has a **clean identity path back through the layer**. Gradient flows unimpeded. The Sublayer's contribution gets scaled, but the highway is intact.

This is why every modern transformer (GPT-2 onwards, LLaMA, etc.) is pre-norm. Show the algebra on paper.

### 3. (Optional) Why `4×` in the FFN — key-value memory framing

A paragraph + sketch: in the Geva et al. interpretation, the FFN's first linear `W_1: d → 4d` is a *key memory* — each of the 4d hidden units is a "key" matched against the residual stream. The GELU/ReLU acts as a soft mask. The second linear `W_2: 4d → d` is the *value memory* — each unit contributes a learned vector back to the residual.

Why 4× specifically? Empirically tuned; the ratio of "memory slots" to "stream dimensions" works out around 4. Modern variants (SwiGLU) sometimes use 2.67× to compensate for the gate adding parameters.

## Format

The LayerNorm derivation in particular is long enough that you'll want to commit a clean transcription (e.g., `01_layernorm_backward.md` with LaTeX) rather than just a photo — but writing it on paper *first* is the point.
