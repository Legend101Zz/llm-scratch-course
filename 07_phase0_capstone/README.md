# Module 7 — Phase 0 capstone: a tiny GPT, assembled

> The gate to Phase 1. If everything in Modules 1–6 was *components*, this is the
> *integration*. You'll prove that a NumPy GPT (assembled from the LayerNorm,
> FFN, attention, and block you wrote by hand) matches a PyTorch GPT with the
> same architecture and weights, then train the PyTorch one on tinyshakespeare
> and watch the loss drop.
>
> This module **does not have a `starter.py`**. It's an *assembly* module:
> there's no new algorithm to derive — just the careful work of fitting
> Modules 4 and 5 into a complete language model, plus embeddings and a head.
> Read `numpy_gpt.py`, then read `torch_gpt.py`, then run `test.py`, then run
> `train.py`. If at any point a parity test fails, the bug is in Module 4 or 5
> (since this module only adds embeddings + LN + a linear projection on top).

## What you'll build

```
              ids ∈ Z^T
               │
               ▼
  ┌────────────────────────────────┐
  │ token_emb (V, d)  + pos_emb    │   ← embeddings
  │ → x ∈ R^(T, d)                 │
  └────────────┬───────────────────┘
               │
       ┌───────▼────────┐
       │  Block × N     │   ← from Module 5 (which uses Module 4)
       │  pre-LN + MHA  │
       │  pre-LN + FFN  │
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │  LayerNorm     │   ← the final LN
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │ lm_head (d, V) │   ← linear, no bias (GPT-2 convention)
       └───────┬────────┘
               │
               ▼
        logits ∈ R^(T, V)
        cross-entropy(logits, targets[shift +1])
```

That's the whole architecture. Every piece is one you've already built:
- **Embeddings + final LN + lm_head** are linear layers + a LayerNorm.
- **Block × N** is Module 5's `TransformerBlock`, imported wholesale.
- **MHA inside the block** is Module 4's `multi_head_attention`, transitively.

## The four files

| File | Purpose |
|---|---|
| `numpy_gpt.py` | The NumPy-only forward-pass GPT (no training — Value-scalar autograd is too slow at this scale, by design) |
| `torch_gpt.py` | The same architecture in PyTorch — trainable |
| `test.py` | Parity proof: identical weights → identical forward to 1e-4 |
| `train.py` | Trains the PyTorch GPT on tinyshakespeare; outputs loss curve + samples |

## Why no NumPy training?

Module 1's Value engine is *scalar* autograd — every multiplication creates a Python object. Running 1 forward pass through a 2-layer GPT on T=64 tokens with d=64 hits roughly 200,000 Value ops; backward through that takes ~30 seconds per step on a modern CPU. 2000 steps would be ~17 hours. That's *intentional* — the Value engine taught you how backprop works *algorithmically*. It's not the right tool to actually train with.

The bridge: Modules 1, 2, 4, 5 each ship a `test.py` that proves the corresponding PyTorch ops match the from-scratch NumPy ops to 1e-5. So by Phase 0 we've **earned the right to trust PyTorch** — we've checked its arithmetic against ours, op by op. The capstone uses PyTorch for the gradient pass; we know what PyTorch is doing under the hood because we built it.

## How to work this module

1. **Read `numpy_gpt.py` top to bottom.** Confirm: the GPT class is *literally* just a token table lookup + position table lookup + a list of Module 5 blocks + a final LayerNorm + a matmul. There's no new math here.
2. **Read `torch_gpt.py`** and verify the architecture is the same (just in PyTorch). The `TorchTransformerBlock` mirrors Module 5's NumPy `TransformerBlock` shape-for-shape.
3. **Run `python test.py`.** Three tests:
   - parameter counts match between numpy + torch versions
   - forward outputs match to 1e-4 with identical weights
   - the full-stack GPT preserves causality (perturbing late positions doesn't change early logits)
4. **Run `python train.py`.** Trains for 2000 steps on CPU (~3–5 min). Produces:
   - `evidence/loss_curve.png` — train + val loss vs step
   - `evidence/training_log.json` — raw numbers
   - `evidence/sample_generations.txt` — what the model generates after training

## Convergence expectation

| step | train loss (nats) | meaning |
|---|---|---|
| 0 | ~4.17 = ln(65) | uniform — random |
| 500 | ~2.4 | learning bigram-ish patterns + some structure |
| 1000 | ~1.9 | recognizing words, capital-letter conventions |
| 2000 | ~1.6 | short Shakespearean snippets emerge |

After 2000 steps, generations should look like (real example targeting; yours will vary):
```
ROMEO:
What now, my lord? I hope thou hast not yet
Forsworn the breath which mocks the day.
```
Mostly real words, mostly grammatical, sometimes coherent. **Not** a real reasoning model — this is 0.2M params on 1MB of text. The point is that the architecture works.

## How this module is "done"

Per [`../CONVENTIONS.md`](../CONVENTIONS.md):
1. `python test.py` passes all three parity tests.
2. `python train.py` completes; train loss drops below 2.0 nats by step 2000.
3. `hand_math/` contains your derivation of the full-stack causal-mask invariant (see hand_math/README.md for what specifically to derive).
4. `evidence/` contains:
   - `parity_output.txt` — captured stdout from `python test.py`
   - `loss_curve.png` — written by `train.py`
   - `training_log.json` — written by `train.py`
   - `sample_generations.txt` — written by `train.py`
   - `roofline.md` — your roofline analysis of *this whole GPT's* forward pass (extends Module 4/5's roofline to the full model)
5. You can cold-explain (mentor will quiz):
   - The full data flow from `ids: (T,)` to `logits: (T, V)`, with shapes at every step.
   - Why we need a positional embedding (and what happens if you drop it).
   - Why the final LayerNorm is *before* `lm_head`, not after.

## Reflection

Once you've trained the model, before declaring Phase 0 done:

1. **Loss baseline:** the random-init loss is `ln(V) ≈ 4.17` for 65-char vocab. Why? Derive it from the cross-entropy formula. *(Hint: a uniform distribution over V classes has entropy `ln(V)`.)*
2. **Position embedding ablation:** what happens if you set `pos_emb = 0` everywhere and re-run training? Will the model still learn? *(Hint: think about what self-attention "sees" with and without position info — and how a causal LM might partially compensate.)*
3. **Capacity:** at ~200k params, what's the model's ratio of params to training tokens (~1M chars ≈ 250k tokens after a BPE)? Compare to Chinchilla's compute-optimal ratio (~1:20). Is this overtrained, undertrained, or roughly Chinchilla-optimal?
4. **Generation quality:** if you generate with `temperature=0.0` (greedy), what do you see? With `temperature=1.5`? Why does temperature affect coherence?
5. **The block, ablation by ablation:** if you replaced the FFN with `Identity()`, would the model still learn? If you replaced the MHA with `Identity()`? Test it; record what happens in `evidence/ablations.md`.

✅ Phase 0 closes when the capstone tests pass + the reflection questions are answered in the cold quiz at the start of Phase 1.
