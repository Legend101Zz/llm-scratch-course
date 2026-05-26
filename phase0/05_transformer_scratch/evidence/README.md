# `05_transformer_scratch/evidence/` — outputs that prove the module works

## What belongs here for Module 5

### `test_output.txt`
Output of `python test.py` showing all five parity tests:
- gelu matches torch (tanh approx) to 1e-6
- layernorm matches torch to 1e-6 + has zero mean / unit std per row
- ffn matches torch to 1e-5
- block forward matches the canonical pre-norm computation to 1e-9
- residual-only block (zero W_O and W2) leaves input unchanged

### `roofline.md` (required per CONVENTIONS)
Apply the same roofline framework you used for Module 4 — but for the full block, not just attention. Decompose by sub-op:
- LN1 (~0 FLOPs, ~MB of HBM traffic)
- MHA (you already roofline'd this)
- LN2
- FFN (`(T, d) @ (d, 4d) @ (4d, d)` — much more compute than MHA but linear-not-quadratic in T)
- Residual adds

Question: which sub-op limits the block's wall-clock time at small T vs. large T? At small T, attention's `T·T·d_k` scaling makes it negligible; at large T it dominates. **What T flips the bottleneck?** Solve for the crossover analytically.

### `metrics.json` (optional)
```json
{
  "gelu_max_err": 1.1e-16,
  "layernorm_max_err": 4.4e-16,
  "ffn_max_err": 8.2e-15,
  "block_forward_match_err": 0.0,
  "residual_path_drift": 0.0,
  "param_count_T_8_d_32_h_4": 4576,
  "roofline_attention_vs_ffn_crossover_T": 256
}
```

## Maintenance

Same rule as Module 4: re-run after any refactor; commit the diff.
