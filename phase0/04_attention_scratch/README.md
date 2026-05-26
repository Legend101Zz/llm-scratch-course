# Module 4 — Self-Attention From Scratch (50 min)

> ⏱️ 50 minutes. **The most important idea in modern AI.** Don't rush this.
>
> 📄 Before you start: skim Section 3 of `papers/1706.03762v7.pdf` (Attention Is All You Need). Then close it. We re-derive.

## 1. The Problem Attention Solves

Static word embeddings (Module 3) give every word **one** vector. But the meaning of a word depends on its context:

- "river **bank**" vs. "investment **bank**"
- "I **left** the room" vs. "turn **left**"
- "the trophy didn't fit in the suitcase because **it** was too big" — does *it* refer to trophy or suitcase?

The bigram model can't help — it only sees the previous token. RNNs see all previous tokens, but compress them into a single fixed-size hidden state — a bottleneck that forgets long-range info.

**What we want:** a way for each token to *look at every other token in its context* and **rewrite its own embedding** to include the relevant pieces. The word "bank" should pull information from "river" or "investment" depending on which is nearby.

This is exactly what attention does. It's a **content-aware mixing operation**: each token's new embedding is a *weighted blend of every other token's information*, where the weights are computed from the tokens themselves.

> 🧠 **Big idea:** attention turns a static embedding into a **context-aware** one. After one attention layer, "bank" near "river" has a different vector than "bank" near "investment." That's the whole magic.

## 2. The Intuition: Soft Dictionary Lookup

Imagine you have a **Python dict** and you want to look something up:

```python
d = {"apple": 1, "banana": 2, "cherry": 3}
d["banana"]   # exact match, returns 2
```

This is "hard" lookup: you have a key, you find the matching key, you get its value. Three roles:

- **Query**: what you're searching for (`"banana"`).
- **Keys**: the labels in the dict (`"apple"`, `"banana"`, `"cherry"`).
- **Values**: the things stored under each key (`1`, `2`, `3`).

Now make it differentiable. Instead of "the key matches exactly," use **similarity** (dot product):

```
score("banana", "apple")  = 0.1
score("banana", "banana") = 0.9
score("banana", "cherry") = 0.2
```

Pass through softmax → these become weights summing to 1:

```
weights ≈ (0.20, 0.55, 0.25)
result  = 0.20·1 + 0.55·2 + 0.25·3 = 2.05
```

You got *roughly* the value for "banana," but smeared a bit across other entries. That smearing is the price of differentiability — and it's exactly what enables gradient descent to learn what the keys and queries should be.

**Self-attention is this exact operation, applied to every token in a sentence at once.** Each token plays all three roles simultaneously: it produces a query (what am I looking for?), it advertises a key (what info do I offer?), and it carries a value (what info will I pass along if attended to?).

## 3. Why Project to Q, K, V Instead of Using X Directly?

Reasonable question: we already have an embedding $x$ for each token. Why introduce three separate projections $W_Q, W_K, W_V$?

Three reasons, in increasing order of subtlety:

**(a) Separation of concerns.** A token's role as a "searcher" is different from its role as a "thing to be found." The word "it" is a *bad* key (it carries little semantic content), but it's an *active* query (it's hunting for its referent). The noun "trophy" is a *strong* key (rich semantic content) but might be a passive query at this position. Separate projections let the model learn each role independently.

**(b) Asymmetric relationships.** Without separate Q and K, attention would be symmetric: $x_i \cdot x_j = x_j \cdot x_i$. So if "it" attends strongly to "trophy," "trophy" must attend equally strongly to "it." That's wrong. Verbs attend to their subjects; subjects don't necessarily attend back. With $W_Q \neq W_K$, the matrix $QK^\top$ is **not symmetric** — relationships can be one-way.

**(c) Decoupling "what to mix" from "what to use as a similarity signal."** $K$ decides *who* gets attended to. $V$ decides *what* gets passed along. They can carry different information. A token might be findable by its syntactic role (key) but contribute its semantic content (value). Without separate $W_V$, the value would be forced to be the same vector used for matching — collapsing two distinct functions into one.

In short: three projections give the model three independent dials. With learnable weights, it figures out the right setting.

## 4. The Math, Slowly

For each token's embedding $x \in \mathbb{R}^{d}$, project to three vectors:

- **Query** $q = x W_Q$ — "what am I looking for?"
- **Key**   $k = x W_K$ — "what do I offer?"
- **Value** $v = x W_V$ — "what info will I pass along?"

For a sequence of $T$ tokens, stack into matrices: $Q, K, V \in \mathbb{R}^{T \times d_k}$.

The full attention operation:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

Reading this **piece by piece**:

### 4.1. $QK^\top \in \mathbb{R}^{T \times T}$ — the score matrix

Element $[i, j]$ = dot product of token $i$'s query with token $j$'s key. Larger values = stronger match. Each row $i$ holds *every score relevant to token $i$*.

### 4.2. Divide by $\sqrt{d_k}$ — variance correction

If $q$ and $k$ have iid components with variance 1, their dot product has variance $d_k$. So as $d_k$ grows, dot products get bigger, and softmax saturates (one entry near 1, all others near 0) — gradients vanish. Dividing by $\sqrt{d_k}$ keeps variance ≈ 1 regardless of head dimension. **This is one of those tiny details that makes training actually work.** Without it, the original Transformer paper reports "extremely small gradients."

Concrete example: with $d_k = 64$ and unit-variance entries, a typical raw score is $\sim\sqrt{64} = 8$. After softmax, an 8 vs. a 6 differs by a factor of $e^2 \approx 7.4$ — already very peaky. Dividing by 8 brings them to 1 vs. 0.75, a factor of 1.3 — much smoother.

### 4.3. Softmax row-wise — soft selection

Each row turns into a probability distribution over the keys. Row $i$, entry $j$ = "fraction of attention token $i$ pays to token $j$." Each row sums to 1.

Why softmax? Because we want a *differentiable* version of "pick the best match." Softmax is the smooth version of argmax — it concentrates weight on the largest entries but lets gradient flow through all of them.

### 4.4. Multiply by $V$ — the actual mixing

The output is a $T \times d_k$ matrix. Row $i$ is a **weighted sum of every value vector**, with weights from row $i$ of the softmax. This row is token $i$'s **new, context-enriched representation**.

> 🔑 The output of attention is a *new embedding for each position*, computed as a weighted mix of every token's value vector. The weights come from queries-vs-keys; the content comes from values.

## 5. A Worked Mini-Example: "The cat sat"

Three tokens: "the", "cat", "sat". Suppose after $W_Q$ and $W_K$ projections, the score matrix (before softmax/scaling) ends up like:

```
                  K_the   K_cat   K_sat
        ┌─────────────────────────────────┐
Q_the   │  2.0    0.5     0.1   │
Q_cat   │  1.0    3.0     0.5   │   ← "cat" mostly attends to itself
Q_sat   │  0.3    2.5     1.0   │   ← "sat" attends strongly to "cat" (its subject)
        └─────────────────────────────────┘
```

After causal mask (token can't see the future) + softmax, row "sat" might become:

```
sat: [0.10, 0.75, 0.15]
```

So sat's new vector = $0.10 \cdot v_\text{the} + 0.75 \cdot v_\text{cat} + 0.15 \cdot v_\text{sat}$. The *cat* component dominates — sat now "knows about" cat. That's how the model encodes "sat's subject is cat" — by literally injecting cat's value vector into sat's representation.

This is what people mean when they say attention makes embeddings **context-aware**. The vector for "sat" now depends on what came before. Run another attention layer and "sat" can attend to a *contextualized* version of cat — which already attended to "the" — so information propagates further with depth.

## 6. Causal Masking (the crucial detail for LMs)

For a *language model*, token $t$ should not see future tokens (that would leak the answer during training). So before softmax we set the upper-triangular part of $QK^\top$ to $-\infty$:

```
    j=0   j=1   j=2   j=3      ← keys (positions)
i=0  ✓    -∞    -∞    -∞
i=1  ✓     ✓    -∞    -∞
i=2  ✓     ✓     ✓    -∞
i=3  ✓     ✓     ✓     ✓
↑
queries
```

After softmax, $-\infty$ entries become 0. Each query only mixes from past + present. This is what makes "attention" → "**causal** self-attention" → a language model.

> ⚠️ Mask **before** softmax, not after. After softmax, zeroing entries makes rows no longer sum to 1 → broken attention distribution. Setting pre-softmax scores to $-\infty$ ensures softmax produces a valid distribution that simply assigns 0 probability to the masked positions.

## 7. Multi-Head Attention — Many Conversations at Once

One attention "head" can learn one kind of relationship. But sentences contain many simultaneous structures: syntactic agreement, coreference, semantic similarity, position-based patterns, etc. Forcing one head to capture all of these is asking too much.

**Solution:** run $h$ heads in parallel, each with its own $W_Q, W_K, W_V$. Each head produces a $d/h$-dimensional output, then concatenate and project:

$$\text{MHA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O$$

Compute cost is the same as one big head (each head is smaller by $1/h$). What you get for free is **specialization**: heads end up learning different patterns. Empirically:

- Some heads attend mostly to the **previous token** (a "looking-back" head — useful for local syntax).
- Some attend to the **first token** of the sequence (a "summary" head).
- Some attend to **the same word elsewhere** (a "copy" / "induction" head).
- Some attend to **the subject of the verb** (semantic role).

This specialization is the substance of [mechanistic interpretability research](https://transformer-circuits.pub/). Heads are the unit of analysis.

## 8. The Attention Matrix, Visualized

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

The actual numbers come from `softmax(Q K^T / √d_k)`. The shape of the attention pattern — sharp peak vs. diffuse spread — *emerges from training*; nobody tells the model where to look.

## 9. Data Flow for One Token (visual)

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

## 10. Build it (in numpy)

Open [`starter.py`](starter.py). You'll implement:
1. `softmax(x, axis)` — be careful with numerical stability (subtract max before exp).
2. `single_head_attention(X, W_Q, W_K, W_V, causal=True)` — returns the new `(T, d_k)` matrix.
3. `multi_head_attention(X, heads, W_O)` — wraps the above.

Keep shapes commented above each line. **Shape-thinking is 80% of transformer engineering.** Sketch shapes on paper before coding any line.

## 11. Math Exercise (do on paper before coding)

For $T=3$, $d=2$ embeddings $X = \begin{pmatrix}1 & 0 \\ 0 & 1 \\ 1 & 1\end{pmatrix}$, $W_Q = W_K = W_V = I_2$:

- Compute $Q K^\top$.
- Apply causal mask.
- Apply softmax row-wise.
- Multiply by $V$.

Confirm token 0 attends only to token 0. Token 1 attends to {0, 1}. Token 2 attends to all three.

## 12. Roofline — is attention memory-bound or compute-bound?

> 🛠️ **Systems lens.** Per [`../../CONVENTIONS.md`](../../CONVENTIONS.md), every hardware-touching module must answer: *what limits this op?* You should be able to do this calculation in your sleep by Phase 4 (it's the single skill `frontier-lab.md` says most directly gets people hired). Start now.

Let's pick a small concrete shape: `B = 1` (one sequence), `T = 512` (sequence length), `d = 64` (embedding dim), `h = 8` heads → `d_k = d/h = 8`. All `fp32` (4 bytes per element).

### The FLOP count (one forward pass)

| Op | What it computes | FLOPs (one head) |
|---|---|---|
| Q = X W_Q | `(T, d) @ (d, d_k)` | `2 · T · d · d_k = 2 · 512 · 64 · 8 = 524,288` |
| K = X W_K | same shape | `524,288` |
| V = X W_V | same shape | `524,288` |
| scores = Q K^⊤ | `(T, d_k) @ (d_k, T)` | `2 · T · T · d_k = 2 · 512 · 512 · 8 = 4,194,304` |
| softmax | `T·T` exps + normalize | negligible (≈ 5·T² ≈ 1.3M flops, dominated by the matmuls) |
| out = scores @ V | `(T, T) @ (T, d_k)` | `2 · T · T · d_k = 4,194,304` |

(The `2·` everywhere is the standard "1 multiply + 1 add per dot-product element" convention.)

Per head: ~10 MFLOPs. With 8 heads + the output projection `(h·d_k, d)` = `(64, 64)`:
- 8 × 10 = 80 MFLOPs from heads
- Output proj: `2 · T · d · d = 2 · 512 · 64 · 64 ≈ 4.2 MFLOPs`
- **Total: ~84 MFLOPs per forward.**

### The byte count (HBM traffic — what actually crosses to GPU memory)

Here's where it gets interesting. We need to load every tensor **at least once** and write every output **at least once**:

| Tensor | Bytes (fp32) |
|---|---|
| `X` (input)                        | `T · d · 4   = 512 · 64 · 4 = 131,072` |
| `W_Q, W_K, W_V, W_O` (weights)     | `4 · d · d · 4 = 4 · 64 · 64 · 4 = 65,536` |
| `Q, K, V` (per head)               | `3 · T · d_k · 4 = 3 · 512 · 8 · 4 = 49,152` |
| **`scores` (per head, T×T)**       | `T · T · 4 = 512 · 512 · 4 = 1,048,576` ← **dominant** |
| `attn weights` (post-softmax)      | another T·T·4 = `1,048,576` |
| Head output, output proj output    | `T · d_k · 4 + T · d · 4 = 16,384 + 131,072` |

Across 8 heads, the **scores tensor alone is 8 × 1 MB = 8 MB of reads + 8 MB of writes**.

Sum it up roughly: ~17 MB of HBM traffic for ~84 MFLOPs.

### Arithmetic intensity (AI)

$$
\text{AI} = \frac{\text{FLOPs}}{\text{bytes}} = \frac{84 \times 10^6}{17 \times 10^6} \approx 5 \text{ FLOP/byte}
$$

### The roofline knee on a T4 (free Colab)

A T4 has **~8.1 TFLOPS** peak fp32 and **~320 GB/s** HBM bandwidth.

$$
\text{knee} = \frac{8.1 \times 10^{12}}{320 \times 10^9} \approx 25 \text{ FLOP/byte}
$$

If your op's AI is **below the knee**, you are **bandwidth-bound** (waiting on HBM). If above, **compute-bound** (limited by tensor cores).

### Conclusion

**AI ≈ 5 FLOP/byte vs. T4 knee at 25.** Attention at this shape is **firmly memory-bound** — moving the scores tensor in and out of HBM is the slow step, not the math itself.

This is the **FlashAttention insight, earned not told**. The fix isn't to do fewer multiplies; it's to never materialize the full `T × T` scores tensor in HBM. Tile it. Keep it in SRAM (on-chip, ~1000× faster). Compute softmax online so you don't need to look at all of `scores` at once. That's the whole FlashAttention paper.

### How AI changes with shape

Try the same calculation for the *backward* pass, and for `T = 2048` instead of `T = 512`:

- Backward materializes `scores`, `attn`, and their gradients → more bytes per FLOP → **even more bandwidth-bound.**
- As `T` grows, the `T·T` scores tensor grows quadratically in bytes while head FLOPs grow quadratically in compute (still `2·T·T·d_k`). AI scales with `d_k`, **not** with `T`. So long-context attention stays bandwidth-bound regardless of length — the bottleneck doesn't go away as you scale up.

Generally:
- **Long context → bandwidth dominates** → FlashAttention's win grows.
- **Wider model (large d_k) → compute dominates** → you become tensor-core-limited.

This is why production inference engines work so hard on KV-cache layout (bandwidth) but not as hard on per-step matmul math (compute is already saturated).

### Worked exercise (commit to `evidence/roofline.md`)

Redo this calculation for:
- `B = 8, T = 1024, d = 256, h = 8, fp16` on an A100 (peak 312 TFLOPS fp16, 1555 GB/s HBM).

State your AI and conclusion. Compare to the T4 result above. **Why does fp16 (or bf16) sometimes flip the bound type?** *(Hint: it halves bytes-moved but doubles peak FLOPs — what happens to AI vs. the new roofline knee?)*

## 13. Reflection

1. **Scaling.** Why $1/\sqrt{d_k}$? (Hint: variance of dot product of two vectors with iid unit-variance entries is $d_k$.) What goes wrong without it for large $d_k$?
2. **Mask order.** Why is masking done **before** softmax, not after? Trace through what happens if you mask after.
3. **Q = K experiment.** What if you used the **same** matrix for $Q$ and $K$? What property would the attention pattern have? Why is that bad for modeling things like "verb attends to subject"?
4. **Context-awareness.** After one attention layer, the embedding for "bank" depends on neighbors. After two layers, it depends on neighbors-of-neighbors. How many layers does GPT-2 use? Why does depth matter for long-range reasoning?
5. **Self-attention vs. cross-attention.** What changes if $Q$ comes from one sequence and $K, V$ come from another? (This is how the original Transformer encoder-decoder talks across sequences — and how vision-language models bind text to images.)

## 14. What's NOT in this module (but should be on your radar)

These are covered in `extras/` — read after the course:
- **RoPE** — rotary position embeddings (replaces learned position vectors in modern models).
- **GQA / MQA** — sharing K/V across heads to shrink the inference KV cache.
- **FlashAttention** — fused kernel that never materializes the full $T \times T$ matrix; the reason long context is feasible.
- **Sparse / Linear attention** — sub-quadratic alternatives.

The math you just learned is unchanged in all of these. They're optimizations on top.

✅ Next: [Module 5 — Transformer Block](../05_transformer_scratch/README.md)
