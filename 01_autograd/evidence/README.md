# `01_autograd/evidence/` — outputs that prove the module works

> Per [`../../CONVENTIONS.md`](../../CONVENTIONS.md), every module commits its test output and any metrics worth comparing across sessions.

## What belongs here for Module 1

### `test_output.txt`
Captured output of `python test.py`. Should show the parity assertion passing:
```
forward  ours=24.704082  torch=24.704082
a.grad   ours=138.834...
b.grad   ours=645.577...
✅ all good — you have rediscovered backprop.
```
Capture with:
```bash
python test.py > evidence/test_output.txt 2>&1
```

### `metrics.json` (optional but recommended)
A small JSON capturing the numerical results of `test.py`. Future sessions can compare:
```json
{
  "forward":       24.704082312484523,
  "grad_a":        138.83381924,
  "grad_b":        645.57723367,
  "max_abs_err":   3.4e-13
}
```

### What NOT to put here
- Don't commit the `__pycache__/` directory or `.pyc` files.
- Don't commit huge profiler HTML dumps. Save a digested summary or screenshot instead.

## When this gets updated

- First time: when you finish Module 1 and `test.py` passes.
- Re-update: any time you refactor `solution.py` or `starter.py` — re-run, re-capture, re-commit.
- The git diff of `test_output.txt` shows you whether a refactor broke numerical reproducibility.
