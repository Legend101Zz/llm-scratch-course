# `06_pytorch_crash/evidence/` — outputs that prove the drills work

> Module 6 is a reference module (no from-scratch derivation), so there's no `hand_math/` folder — but the drills still need evidence on disk.

## What belongs here for Module 6

### `drills_output.txt`
Captured stdout from `python tensor_drills.py` showing all 11 drills assert-pass, **including drill 11** (the bridge drill that rebuilds Module 4 attention in PyTorch and matches the NumPy reference to 1e-5).

Capture with:
```bash
python tensor_drills.py > evidence/drills_output.txt 2>&1
```

Should end with:
```
drill 11: torch MHA vs numpy MHA max abs error = 1.42e-15
✅ drill 11: torch MHA matches numpy MHA. you have bridged from-scratch → framework.

✅ all 11 drills passed. you are PyTorch-fluent.
```

### `xor_torch_output.txt` (optional)
If you ran `xor_torch.py` to compare with Module 2's Value-based XOR — capture its output here. Useful for cross-referencing: the torch XOR should converge to the same loss curve as Module 2's Value-based XOR with matched seeds.

### Why no `hand_math/` here?

Module 6 is a PyTorch reference — every concept (tensors, autograd, nn.Module, optim) has its math in the modules that *use* it (1, 2, 4, 5). The drills are *exercises in framework fluency*, not in derivation. So convention compliance for Module 6 is: starter assertions pass + evidence captured. No hand-math required.

If you do *invent* a derivation while working through the drills (e.g. you finally understand why broadcasting works on the implicit "right-aligned dimensions" rule), feel free to add a `hand_math/` folder later — it's never bad to record an insight.
