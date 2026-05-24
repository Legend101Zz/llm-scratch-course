# `04_attention_scratch/evidence/` — outputs that prove the module works

## What belongs here for Module 4

### `test_output.txt`
Output of `python test.py` showing all five parity tests:
- softmax matches torch to 1e-6
- single-head causal matches torch to 1e-5
- single-head bidirectional matches torch to 1e-5
- multi-head matches torch to 1e-5
- causal mask structurally correct (output[0] drift = 0 after perturbing positions [1:])

### `roofline.md` (REQUIRED for Module 4 onward)

Per [`../../CONVENTIONS.md`](../../CONVENTIONS.md), every hardware-touching module from Module 4 forward commits a roofline calculation. The README in `../README.md` walks the calculation for `(B=1, T=512, d=64, h=8)` on a T4. For Module 4, commit your version of:

- The same calculation re-derived in your own words (don't copy the README).
- One *additional* shape: redo the math for either `(B=8, T=1024, d=256, h=8, fp16)` on an A100 (per the README's worked exercise), or any other shape where you can find a reasonable peak-TFLOPS / HBM-bandwidth number.
- Conclusion sentence: "memory-bound at AI=X vs. knee=Y → optimization target = bandwidth."

This roofline document is *the artifact* that proves you've internalized the systems lens. Phase 4's first deliverable will be doing this for ten different shapes in your sleep — start practicing here.

### `metrics.json` (optional)
```json
{
  "softmax_max_err": 1.1e-16,
  "single_head_causal_max_err": 5.5e-15,
  "multi_head_max_err": 8.9e-15,
  "causal_leakage_after_perturb": 0.0,
  "roofline_T4_T512_d64_h8_AI": 5.0,
  "roofline_T4_knee_FLOP_per_byte": 25.0,
  "conclusion": "memory-bound at this shape; FlashAttention-style fusion is the right optimization"
}
```

## Maintenance

If you refactor `solution.py` (e.g., consolidate single-head and multi-head into a batched version), re-run `test.py` and update both `test_output.txt` and `metrics.json`. The git diff is your safety net.
