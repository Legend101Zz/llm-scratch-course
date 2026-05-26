# `07_phase0_capstone/evidence/` — Phase 0 closure artifacts

> The single most important `evidence/` folder in the course — this is the file
> set that proves the from-scratch components compose into a real LM that
> learns.

## What belongs here

### `parity_output.txt`
Captured output of `python test.py`. Must show:
- `param count` matches between numpy + torch
- `max abs error` < 1e-4 between numpy GPT and torch GPT forward
- `drift at positions [0..7]` = 0.0 after perturbing positions [8..15] (causal invariant holds)
- ends with: `✅ Phase 0 capstone parity tests passed.`

Capture with `python test.py > evidence/parity_output.txt 2>&1`.

### `loss_curve.png`
Written by `train.py` automatically. Train + val loss vs. step, with a dotted line at `ln(V)` for the uniform baseline. Should clearly show:
- Step 0 loss ≈ ln(65) ≈ 4.17
- Monotonic decrease (with stochastic noise)
- Train loss < 2.0 by step 2000

### `training_log.json`
Written by `train.py`. Raw `{step, train_loss, val_loss}` triples. Useful for re-plotting or comparing across runs (e.g., when you swap LayerNorm → RMSNorm in a Phase 1 ablation).

### `sample_generations.txt`
Written by `train.py`. 200-token generations from each of a few seed prompts. After 2000 steps these should be roughly word-shaped (real characters in mostly-real combinations, capital letters at line starts).

### `roofline.md` (REQUIRED)

Roofline analysis of the full forward pass — not just the attention sub-op. Decompose:
- Embedding lookup (no real FLOPs, ~MB of HBM traffic)
- Per-block: MHA + FFN (you have these from Module 4/5's roofline files)
- Final LN + lm_head matmul `(T, d) @ (d, V) = (T, V)`

For `T=64, d=64, V=65, n_layers=2`:
- Total FLOPs per forward
- Total HBM traffic
- AI = FLOPs / bytes
- Compare to your CPU's roofline (Apple M-series ≈ 100 GFLOPS / ~100 GB/s; commodity x86 similar). At this tiny size the model is almost certainly **compute-bound, not memory-bound**, which is *opposite* to attention at production sizes. Explain why in one paragraph.

### `ablations.md` (recommended — answers reflection Q5)
If you do the ablations from the README's reflection (replace MHA with Identity, replace FFN with Identity, set pos_emb to zero), commit the resulting loss curves + a paragraph each on what you learned.

## How this folder ages

Each Phase 1 training run might re-do the training with bigger models. Don't overwrite — version: `loss_curve_2k.png`, `loss_curve_10k.png`. The diff over time is part of the story.
