# `02_neural_net/evidence/` — outputs that prove the module works

## What belongs here for Module 2

### `test_output.txt`
Output of `python test.py` showing all three parity tests passing:
- forward parity to 1e-6
- per-parameter gradient match to 1e-5
- training convergence (loss < 0.05 after 500 steps)

Capture with `python test.py > evidence/test_output.txt 2>&1`.

### `xor_loss_curve.png` (recommended)
A simple matplotlib plot of loss vs. step over 500 training steps. Helpful for noticing if the curve looks suspicious (oscillating, plateau, divergence). You can add it to `solution.py`'s `__main__` block:
```python
import matplotlib.pyplot as plt
losses = []  # append in the loop
# ...
plt.plot(losses); plt.xlabel("step"); plt.ylabel("loss"); plt.yscale("log")
plt.savefig("evidence/xor_loss_curve.png")
```

### `metrics.json` (optional)
```json
{
  "final_loss_after_500_steps": 0.00021,
  "predictions": {"00": 0, "01": 1, "10": 1, "11": 0},
  "param_count": 33,
  "max_abs_grad_err_vs_torch": 4.2e-12
}
```

## Why bother

Subtle backprop bugs in `Value.__mul__` or `Value.tanh` might still pass Module 1's `test.py` (which uses a fixed-expression check) but break in Module 2 (which composes the engine into a multi-layer network with batched accumulation). The grad-match assertion in Module 2's `test.py` is your insurance that the engine generalises.

If you ever refactor the Value engine later, re-run `test.py` here and compare the `test_output.txt` diff to confirm nothing silently regressed.
