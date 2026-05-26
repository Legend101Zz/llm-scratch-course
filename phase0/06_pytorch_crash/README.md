# Module 6 — PyTorch Crash Course (30–90 min)

> ⏱️ Aim for 30 min on first pass. Re-visit the deep-dive READMEs as reference. By the end, **no PyTorch code in this course will look unfamiliar**, and you should be able to read most production model code without help.

This module is an index. Skim this page, then drill into the topic-specific deep-dives below. Each is self-contained — you can read in any order.

| Topic | What's in it | When to read |
|---|---|---|
| **[`01_tensors.md`](01_tensors.md)** | Tensors end-to-end: dtypes, devices, indexing, reshape vs view vs permute, broadcasting, every math op you'll use | First. This is the foundation. |
| **[`02_autograd.md`](02_autograd.md)** | How autograd actually works. The dynamic graph. `requires_grad`, `detach`, `no_grad`, custom `Function`s | After tensors. Crucial for understanding training. |
| **[`03_nn_module.md`](03_nn_module.md)** | Every `nn.*` building block: Linear, Embedding, Conv, RNN, LayerNorm, Dropout, ModuleList/Dict, Sequential, Parameter, buffers, init, save/load | Before writing your own model. |
| **[`04_optim_data_gpu.md`](04_optim_data_gpu.md)** | Optimizers (SGD/Adam/AdamW), LR schedulers, Dataset/DataLoader, GPU & device management, mixed precision (AMP), `torch.compile`, profiling | When you start training. |

---

## The 10-Second Mental Model

```
   Tensor (data + dtype + device + .grad)
        │
        ├──── ops build a graph ────┐
        ▼                           │
   Forward pass                     │
        │                           │
        ▼                           │
   Loss (a scalar tensor)           │
        │                           │
        ▼ loss.backward()           │
        │                           │
   Walk graph in reverse, fill .grad on every leaf
        │
        ▼ optimizer.step()
        │
   Params updated in place
        │
        ▼ optimizer.zero_grad()
        │
   Repeat
```

That's the entire training loop. Every other PyTorch concept — `nn.Module`, optimizers, datasets — is plumbing that makes this loop ergonomic. Internalize the loop, and the library starts feeling small.

## How PyTorch Compares to Your Module 1 Engine

| Module 1 (yours)         | PyTorch                          |
|--------------------------|----------------------------------|
| `Value(2.0)`             | `torch.tensor(2.0, requires_grad=True)` |
| Python scalar            | n-dim tensor (CPU/GPU)           |
| `out._backward = ...`    | C++ autograd engine, automatic   |
| `value.backward()`       | `loss.backward()`                |
| `p.data -= lr * p.grad`  | `optimizer.step()` + `zero_grad()` |
| `Neuron`, `Layer`, `MLP` | `nn.Linear`, `nn.Module`         |

Same ideas, optimized + GPU. **If you grokked Module 1, you already understand 80% of PyTorch's design.** What's new is shape-thinking (tensors instead of scalars), the module API, and the data/optim plumbing.

---

## What "fluent in PyTorch" means

You can call yourself fluent when:

1. You can guess the **shape** of any intermediate tensor in someone's `forward` without running it.
2. You know which of `view` / `reshape` / `permute` / `transpose` / `contiguous` to reach for, and why.
3. You know the **5-step training loop** by heart: `zero_grad` → `forward` → `loss` → `backward` → `step`.
4. You can read an `nn.Module` and predict its parameters without printing them.
5. You debug a NaN or shape error within ~3 prints.
6. You know when `.detach()`, `torch.no_grad()`, and `model.eval()` apply — and can articulate the differences.

The deep-dive READMEs are designed to take you to all six.

---

## Two exercises

### Exercise A — Tensor Operation Drills (15 min)

Open [`tensor_drills.py`](tensor_drills.py). 10 progressively harder TODOs covering: creation, reduction, scaled dot-product, causal masking, softmax, head-splitting reshape, autograd verification, `nn.Linear` param accounting, and `train()`/`eval()` mode behavior. Every step has assertions — when they pass, you know it's right.

If you can do all 10 without looking at the docs or the deep-dive READMEs, you're operationally fluent.

### Exercise B — XOR in PyTorch (10 min)

Open [`xor_torch.py`](xor_torch.py). Rewrite Module 2's XOR MLP in PyTorch. Goal: feel the difference. Half the lines, GPU-ready, faster, and you no longer maintain your own backprop.

Compare line-by-line with `02_neural_net/solution.py`. The math is identical; only the engine changed.

---

## Common-failure cheatsheet (keep this open while you train)

| When... | Do... |
|---|---|
| Anything is wrong | `print(x.shape, x.dtype, x.device)` first. |
| Loss is NaN | Check `softmax`/`log` inputs, learning rate, AMP scaler. |
| `RuntimeError: shapes cannot be multiplied` | Print all shapes; sketch on paper. Usually a missing `.transpose(-2,-1)` or forgotten batch dim. |
| `expected device cuda:0 but got cpu` | Some tensor wasn't `.to(device)`'d. |
| `element 0 of tensors does not require grad` | Forgot `requires_grad=True`, or detached somewhere. |
| `view size is not compatible with input tensor's size and stride` | Tensor became non-contiguous after permute/transpose. Use `.reshape()` or `.contiguous().view()`. |
| Memory blows up | Forgot `torch.no_grad()` during eval, or didn't `del` big intermediates, or batch is too big. |
| Weights aren't moving | Forgot `optimizer.step()`, or `lr=0`, or `requires_grad=False` somewhere. |
| Suddenly different results across runs | Forgot `torch.manual_seed(...)` and `torch.cuda.manual_seed_all(...)`. |

---

## Going further (read AFTER finishing the course)

- [PyTorch official tutorials](https://pytorch.org/tutorials/) — the "60-minute blitz" is exactly this README's depth, more polished.
- [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html) — JIT-compiles your model for big speedups. Drop-in for PyTorch 2.x.
- [Edward Yang — PyTorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/) — what's actually happening under the hood.
- [Aladdin Persson's PyTorch series](https://www.youtube.com/playlist?list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz) — video walk-throughs that complement these READMEs.

✅ Next: [Module 7 — Phase 0 capstone (tiny GPT)](../07_phase0_capstone/README.md). The big one — assembles every from-scratch component into a working LM, parity-tested against PyTorch.
