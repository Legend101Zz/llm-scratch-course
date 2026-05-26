# REVIEW.md — spaced-repetition queue

> Forces recall, per Hard Rule #5. The mentor adds questions at session end; the mentor quizzes due ones at session start.
>
> **Schedule per item:** correct answer pushes the next-due date out (1d → 3d → 7d → 30d → 90d). Wrong answer resets to 1d. Mentor logs hit/miss inline.

---

## Format

```
| ID | Q | First asked | Last asked | Next due | Streak | History |
```

`Streak` = number of consecutive correct answers. `History` = compact log like `2026-05-23 ✗ | 2026-05-25 ✓ | 2026-05-28 ✓`.

The user **doesn't need to peek at the answer key** — the answer key is in the indicated source file. The mentor is the one who checks correctness.

---

## Active queue (sorted by next-due)

| ID | Q | Source | First asked | Last asked | Next due | Streak | History |
|---|---|---|---|---|---|---|---|
| R-001 | Without looking, why do attention scores get divided by √d_k? Don't restate "for stability" — derive it. | `phase0/04_attention_scratch/README.md` | (cold quiz pending) | – | 2026-05-23 | 0 | – |
| R-002 | The softmax in attention is taken over which dimension, and what would break if you flipped it (softmax over queries instead of over keys)? | `phase0/04_attention_scratch/README.md` | (cold quiz pending) | – | 2026-05-23 | 0 | – |
| R-003 | Explain the *three* reasons Q, K, and V are separate linear projections instead of one shared projection or just `X` itself. | `phase0/04_attention_scratch/README.md` | (cold quiz pending) | – | 2026-05-23 | 0 | – |
| R-004 | Why is the causal mask applied **before** softmax and not after? Demonstrate with a 2-token example what goes wrong if you flip the order. | `phase0/04_attention_scratch/README.md` | (cold quiz pending) | – | 2026-05-23 | 0 | – |
| R-005 | In a pre-norm transformer block, draw the residual stream and the LayerNorm placements. Where is the "gradient highway" and why does pre-norm protect it better than post-norm? | `phase0/05_transformer_scratch/README.md` | (cold quiz pending) | – | 2026-05-23 | 0 | – |
| R-006 | The FFN inside a transformer block expands `d` → `4d` → `d`. Why 4×? Give the empirical and the "key-value memory" framings. | `phase0/05_transformer_scratch/README.md` | (cold quiz pending) | – | 2026-05-23 | 0 | – |
| R-007 | What does LayerNorm normalize over, exactly — which axis? Why is that the right axis for transformers (vs BatchNorm)? | `phase0/05_transformer_scratch/README.md` | (cold quiz pending) | – | 2026-05-23 | 0 | – |
| R-008 | Sketch the multi-head attention forward pass with explicit shapes: input `(B, T, d)` → output `(B, T, d)`, all intermediate shapes labeled. | `phase0/04_attention_scratch/README.md` | (cold quiz pending) | – | 2026-05-23 | 0 | – |
| R-009 | Why does scalar (reverse-mode) autograd work in two passes — forward then backward — and why couldn't you just compute gradients on the forward pass? (Hint: many-to-one structure.) | `phase0/01_autograd/README.md` | (cold quiz pending) | – | 2026-05-26 | 0 | – |
| R-010 | What does the cross-entropy loss become for a single token if the model assigns probability 0.0001 to the correct answer? (Numeric answer expected.) | `phase0/00_start/README.md` | (cold quiz pending) | – | 2026-05-26 | 0 | – |

---

## Resolved (kept for audit)

(none yet)

---

## How due dates progress

| State | Action if correct | Action if wrong |
|---|---|---|
| First ask | → next due in 3 days | → next due in 1 day |
| Streak 1 | → next due in 7 days | → reset to 1 day |
| Streak 2 | → next due in 30 days | → reset to 1 day |
| Streak 3 | → next due in 90 days | → reset to 1 day |
| Streak 4 | → permanently retired to `Resolved` | → reset to 1 day |

If a question gets reset 3 times running, that's a flag the user has a real gap — mentor opens a focused mini-module to close it.
