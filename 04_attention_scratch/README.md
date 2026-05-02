# Module 4 — Self-Attention From Scratch (50 min)

> ⏱️ 50 minutes. **The most important idea in modern AI.** Don't rush this.
>
> 📄 Before you start: skim Section 3 of `papers/1706.03762v7.pdf` (Attention Is All You Need). Then close it. We re-derive.

## 1. The Problem Attention Solves

A bigram model only looks at the previous token. RNNs look at all previous tokens but compress them into one fixed-size hidden state — a bottleneck. We want each token to look at *all* previous tokens, with **learnable, content-dependent** weights.

> "Hey, what should I attend to in my context?" — every token, every layer.

## 2. The Q, K, V Trick

For each token's embedding $x \in \mathbb{R}^{d}$, project to three vectors:

- **Query** $q = x W_Q$ — "what am I looking for?"
- **Key**   $k = x W_K$ — "what do I offer?"
- **Value** $v = x W_V$ — "what info will I pass along?"

For a sequence of $T$ tokens, stack into matrices: $Q, K, V \in \mathbb{R}^{T \times d_k}$.

Each token's new representation is a **weighted sum of all the V's**, where the weights come from how much its Q matches every K:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

Let's read this slowly:

1. $QK^\top \in \mathbb{R}^{T \times T}$. Element $[i, j]$ = how well token $i$'s query matches token $j$'s key. Bigger = more attention.
2. $/\sqrt{d_k}$ — scaling to keep variance ~1 so softmax doesn't saturate. Without this, gradients vanish for large $d_k$.
3. softmax along the row → each row sums to 1. Now $[i, j]$ = "fraction of attention $i$ pays to $j$".
4. Multiply by $V$ → each token's new representation is a weighted blend of all values.

## 3. Causal Masking (the crucial detail for LMs)

For a *language model*, token $t$ should not see future tokens (that would leak the answer). So before softmax we set the upper-triangular part of $QK^\top$ to $-\infty$:

```
    j=0   j=1   j=2   j=3      ← keys (positions)
i=0  ✓    -∞    -∞    -∞
i=1  ✓     ✓    -∞    -∞
i=2  ✓     ✓     ✓    -∞
i=3  ✓     ✓     ✓     ✓
↑
queries
```

After softmax, those become 0. Each query only mixes from past + present. This is what makes "attention" → "**causal** self-attention" → a language model.

## 4. Multi-Head Attention

One attention "head" learns one kind of relationship. We want many in parallel: syntactic agreement, coreference, position, semantics... So we run $h$ heads in parallel, each with its own $W_Q, W_K, W_V$, each producing $d_k = d/h$ dimensional outputs, then concatenate and project:

$$\text{MHA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O$$

Each head is cheap because dims shrink to $d/h$.

## 4b. The Attention Matrix, Visualized

For a sequence of 4 tokens, after softmax over a causal mask, the attention matrix looks like:

```
              Keys  →
          tok0   tok1   tok2   tok3
        ┌─────────────────────────────┐
  tok0  │ 1.00  0.00   0.00   0.00   │   ← can only see itself
        │                             │
Q tok1  │ 0.30  0.70   0.00   0.00   │   ← sees tok0 and tok1
↓       │                             │
  tok2  │ 0.20  0.30   0.50   0.00   │   ← sees up to tok2
        │                             │
  tok3  │ 0.10  0.20   0.30   0.40   │   ← sees everything
        └─────────────────────────────┘

Each row sums to 1 (it's a softmax). Lower triangle is "real" attention,
upper triangle is forced to 0 by the mask.
```

The actual numbers come from `softmax(Q K^T / √d_k)`. The shape of the attention pattern (sharp vs. diffuse) emerges from training. Different heads end up specializing — one head might attend mostly to the previous token (a "looking-back" head), another to the start-of-sentence (a "context" head), another to the same token type (a "copy" head). This is the substance of [interpretability research](https://transformer-circuits.pub/).

```
DATA FLOW for one token (token 2, in a 4-token sequence):

   embedding x_2 ─────┬──→ W_Q ──→ q_2  ────┐
                     ├──→ W_K ──→ k_2 ────┐ │
                     └──→ W_V ──→ v_2 ──┐ │ │
                                        │ │ │
   (also k_0, k_1 from earlier tokens)  │ │ │
                                        │ │ │
                              q_2 · K^T ──┘ │ │
                                ↓           │ │
                             scale by √d_k  │ │
                                ↓           │ │
                             mask + softmax │ │
                                ↓ (1×T)     │ │
                          weighted sum of V ┘ │
                                ↓             │
                          new representation  │
                          for token 2 ────────┘
```

## 5. Build it (in numpy)

Open [`starter.py`](starter.py). You'll implement:
1. `softmax(x, axis)`
2. `single_head_attention(X, W_Q, W_K, W_V, causal=True)` — returns the new `(T, d_k)` matrix.
3. `multi_head_attention(X, heads, W_O)` — wraps the above.

Keep shapes commented above each line. **Shape-thinking is 80% of transformer engineering.** Sketch shapes on paper before coding any line.

## 6. Math Exercise (do on paper before coding)

For $T=3$, $d=2$ embeddings $X = \begin{pmatrix}1 & 0 \\ 0 & 1 \\ 1 & 1\end{pmatrix}$, $W_Q = W_K = W_V = I_2$:

- Compute $QK^\top$.
- Apply causal mask.
- Apply softmax row-wise.
- Multiply by $V$.

Confirm token 0 attends only to token 0. Token 1 attends to {0, 1}. Token 2 attends to all three.

## 7. Reflection

- Why $1/\sqrt{d_k}$? (Hint: variance of dot product of two vectors with iid unit-variance entries is $d_k$.)
- Why is masking done **before** softmax, not after?
- What if you used the same matrix for $Q$ and $K$? What property would the attention pattern have?

✅ Next: [Module 5 — Transformer Block](../05_transformer_scratch/README.md)
