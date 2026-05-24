# `04_attention_scratch/hand_math/` — pencil derivations

> Attention is the most-derived idea in the course. Don't skip these. Per
> [`../../CONVENTIONS.md`](../../CONVENTIONS.md) you commit at least one
> derivation; for attention, two is recommended.

## What belongs here for Module 4

### 1. The 1/√d_k scaling — derived from variance

Setup: assume the entries of `Q` and `K` are iid with mean 0 and variance 1.

For a single dot-product score `s = Q_i · K_j = Σ_k q_{i,k} · k_{j,k}`:
- `E[s] = Σ_k E[q_{i,k}] · E[k_{j,k}] = 0`
- `Var[s] = Σ_k Var[q_{i,k} · k_{j,k}] = Σ_k Var[q_{i,k}] · Var[k_{j,k}] = d_k`

(Using independence of `q_{i,k}` and `k_{j,k}` and the fact that product-of-independents variance multiplies for zero-mean factors.)

So `Var[s] = d_k`. Without scaling, scores grow with `d_k` — and for `d_k = 64`, std dev ≈ 8. Now plug into softmax:
- Softmax is shift-invariant: only score *differences* matter.
- A typical pair of scores differs by ~10 (since std=8 in each, difference has std ≈ 11).
- `softmax([0, 10, ...])` → concentrates 99.99% of mass on the largest score → near-one-hot.
- Gradients of softmax for near-one-hot outputs ≈ 0 → **vanishing gradient through attention.**

The fix: divide by `√d_k` so `Var[s/√d_k] = 1` for any `d_k`. Now score differences have std ≈ 1.4 regardless of head dim; softmax stays diffuse; gradients flow.

**Commit this derivation on paper.**

### 2. Mask-before-softmax — proof by counterexample

Show in a 2-token example what happens if you mask AFTER softmax:

Setup: scores = `[[3, 5], [3, 5]]`. The "right" causal answer should have attention `[[1, 0], [softmax([3,5])]]`.

- **Correct (mask before softmax):**
  - Mask: `[[3, -∞], [3, 5]]`
  - Softmax → `[[1, 0], [0.119, 0.881]]` — rows sum to 1 ✓
- **Wrong (mask after softmax):**
  - Softmax of `[[3, 5], [3, 5]]` → `[[0.119, 0.881], [0.119, 0.881]]`
  - Zero out upper-right → `[[0.119, 0], [0.119, 0.881]]` — **row 0 sums to 0.119, not 1.**
  - Result: token 0's output is 0.119 × V[0] instead of V[0]. Information is *lost*, not just rerouted.

The invariant that breaks: **attention rows must be probability distributions (sum to 1).** Mask-then-softmax preserves it; softmax-then-mask doesn't.

Commit this two-token worked example.

### 3. Why Q ≠ K (one-paragraph argument)

Why can't we use one shared projection `W_QK = W_Q = W_K` instead of two? Sketch on paper:
- If `Q = K`, then `Q K^⊤` is symmetric, and the score from `i → j` equals `j → i`.
- "Adjective attends to noun" should be different from "noun attends to adjective" — asymmetric relations are real and the architecture has to support them.
- One paragraph + an example (the "trophy didn't fit in the suitcase because it was too big" — *it* attends to *trophy*; *trophy* doesn't necessarily attend back to *it*).

This argues for `Q ≠ K`. By similar reasoning, `V` is decoupled from both — what gets *written* by a head can be different from what gets *matched on*.

## Format

Photo or markdown. The variance argument benefits from being typeset (greek letters) — feel free to transcribe; the *thinking* must be yours.
