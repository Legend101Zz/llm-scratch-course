# 6.2 — Autograd, Top to Bottom

> Goal: understand exactly what `.backward()` does, why it does it, and when it goes wrong. You already wrote autograd from scratch in Module 1 — this deep-dive maps that onto PyTorch.

---

## 1. The Two-Line Mental Model

PyTorch records operations into a **dynamic computation graph** as you run forward. Calling `.backward()` walks that graph in reverse and accumulates gradients into every leaf tensor that has `requires_grad=True`.

```
   Tensor with requires_grad=True
        │
        ▼  ops record themselves
   [op] ─→ [op] ─→ [op] ─→ scalar loss
                                │
                                ▼ loss.backward()
                                │
   walks back, calls each op's vjp (vector-Jacobian product)
                                │
                                ▼
   .grad accumulated on every leaf
```

This is **exactly** what your Module 1 `Value` engine did. Each op stored a `_backward` closure; `backward()` toposorted and called them in reverse. PyTorch is the same idea, in C++, with tensor support.

---

## 2. `requires_grad` — Who Tracks Gradients?

Tensors have a flag `requires_grad`. Default is `False`. Set it to `True` when you want gradients to flow back to a tensor.

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x
y.backward()
print(x.grad)        # 8.0  (= 2x + 2 at x=3)
```

Three ways tensors get `requires_grad=True`:

1. **Created with the flag**: `torch.tensor(..., requires_grad=True)`.
2. **Model parameters** (`nn.Parameter`) — automatically have it set.
3. **Derived from a tracked tensor**: any op whose input had `requires_grad=True` produces a result with `requires_grad=True`.

```python
w = torch.randn(5, requires_grad=True)
x = torch.randn(5)                 # default False
y = w * x                          # requires_grad = True (because w does)
y.requires_grad                    # True
```

### Toggling at runtime

```python
x.requires_grad_(False)            # in-place toggle (note the trailing _)
x.requires_grad = True             # also valid for leaf tensors
```

---

## 3. Leaf Tensors and the `.grad` Field

A **leaf** tensor is one created directly by you (not the output of a tracked op). Only leaves accumulate gradients. Intermediate tensors do not store `.grad` by default.

```python
x = torch.tensor(3.0, requires_grad=True)   # leaf
y = x ** 2                                  # not a leaf; requires_grad = True
z = y.sum()                                 # not a leaf
z.backward()
print(x.grad)        # 6.0
print(y.grad)        # None — intermediate tensor
```

To force an intermediate to retain its gradient (rare; usually for debugging):

```python
y.retain_grad()
z.backward()
print(y.grad)        # now populated
```

> 🔑 **`.grad` accumulates.** Every `.backward()` *adds* to existing `.grad`. That's why `optimizer.zero_grad()` exists — it clears them before each step. This is the same `+=` behavior you implemented in Module 1.

```python
x = torch.tensor(2.0, requires_grad=True)
(x ** 2).backward(); print(x.grad)    # 4.0
(x ** 2).backward(); print(x.grad)    # 8.0  ← added, not replaced
x.grad = None                          # reset
(x ** 2).backward(); print(x.grad)    # 4.0
```

---

## 4. `.backward()` — What It Actually Requires

`.backward()` only works on a **scalar** tensor. If your loss is a vector or higher, you must reduce it first (usually `.sum()` or `.mean()`).

```python
x = torch.randn(5, requires_grad=True)
y = x ** 2                  # vector
# y.backward()              # ERROR: grad can be implicitly created only for scalar outputs
y.sum().backward()          # OK
```

You *can* backward a non-scalar by passing a `gradient` argument — that's the seed for the chain rule from "above." We don't need this for normal training.

### The graph is freed by default

After `.backward()`, PyTorch deletes the graph (saves memory). Calling backward a second time on the same graph errors:

```python
y = x ** 2
y.sum().backward()
y.sum().backward()    # RuntimeError: Trying to backward through the graph a second time
```

If you actually need this (rare), pass `retain_graph=True`:

```python
y.sum().backward(retain_graph=True)
y.sum().backward()              # works now
```

**99% of the time you don't want this.** Just rebuild the forward.

---

## 5. `detach()` — Cutting the Graph

`.detach()` returns a new tensor sharing storage, but **disconnected from the graph**. No gradient flows back through it.

```python
x = torch.randn(5, requires_grad=True)
y = x ** 2
z = y.detach()              # z.requires_grad = False
(z * 3).sum().backward()    # x.grad = None (never reached x)
```

When to use:
- The target/label in a loss should be detached from the model (you don't want to backprop into your data).
- Caching activations (e.g., KV cache during inference) — you don't need grads through them.
- Implementing things like Polyak averaging, EMA, target networks in RL — the "frozen" copy is detached.

There is also `tensor.data` — older API for the same thing. Prefer `.detach()` (safer with autograd).

---

## 6. `torch.no_grad()` — Disable Tracking for a Block

`no_grad()` is a context manager that turns off graph building entirely inside it. Use it for inference / evaluation.

```python
model.eval()
with torch.no_grad():
    logits = model(inputs)              # no graph built; faster, less memory
    preds = logits.argmax(dim=-1)
```

This is **different** from `model.eval()`:
- `model.eval()` toggles dropout/batchnorm into inference mode (deterministic).
- `torch.no_grad()` disables autograd graph building.

**You almost always want both** during evaluation — they're orthogonal and complementary.

There's also `torch.inference_mode()` — slightly faster than `no_grad()`, with stronger guarantees that no autograd-related state is touched. Use it in production inference paths.

```python
with torch.inference_mode():
    logits = model(x)
```

---

## 7. The Computation Graph, Visualized

For `y = (x * 2 + 3) ** 2`:

```
   x  (leaf, requires_grad=True)
    │
    │ MulBackward (×2)
    ▼
    a = x * 2
    │
    │ AddBackward (+3)
    ▼
    b = a + 3
    │
    │ PowBackward (**2)
    ▼
    y
```

Each tensor stores a `.grad_fn` pointing to the op that created it. `y.grad_fn` is `<PowBackward0>`. Following `.next_functions` walks the graph upstream.

```python
print(y.grad_fn)                          # <PowBackward0 object>
print(y.grad_fn.next_functions)           # tuple including AddBackward
```

You don't usually need to peek inside, but knowing the graph exists makes debugging make sense.

---

## 8. Custom Autograd Functions

99% of the time you write models out of standard ops and let PyTorch handle backward. Rarely, you need a custom op (e.g., a non-differentiable layer, a special trick, or a faster kernel). The pattern:

```python
class StraightThroughEstimator(torch.autograd.Function):
    """Forward: round to nearest integer.
       Backward: pretend it was the identity (gradient passes through unchanged).
       This is how quantization-aware training works."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)         # save tensors needed in backward
        return x.round()

    @staticmethod
    def backward(ctx, grad_output):
        # we ignore the saved tensor here; just pass grad through
        return grad_output

ste = StraightThroughEstimator.apply

x = torch.randn(5, requires_grad=True)
y = ste(x)                               # rounded values
y.sum().backward()                       # grad flows through as identity
```

Two static methods: `forward(ctx, *inputs)` and `backward(ctx, *grad_outputs)`. `ctx` is a scratchpad for tensors and constants you need in backward.

---

## 9. Common Autograd Errors and What They Mean

| Error | Cause | Fix |
|---|---|---|
| `element 0 of tensors does not require grad` | You called backward on a tensor that has no graph (no `requires_grad=True` upstream, or you detached). | Check `requires_grad` on inputs; remove stray `.detach()`. |
| `Trying to backward through the graph a second time` | Called `.backward()` twice on the same forward. | Run forward again, or pass `retain_graph=True` (rarely correct). |
| `RuntimeError: leaf variable has been moved into the graph interior` | You modified a leaf tensor in place after using it in the graph. | Avoid in-place ops on parameters; use functional updates. |
| `One of the variables needed for gradient computation has been modified by an inplace operation` | An activation needed for backward was overwritten in place. | Stop using `_` ops on intermediates; copy if needed. |
| `RuntimeError: cudnn RNN backward can only be called in training mode` | Called backward while in eval mode. | `model.train()` before training. |

---

## 10. Putting It Together — Your Module 1 Engine, in PyTorch Terms

| Module 1 concept | PyTorch equivalent |
|---|---|
| `Value(2.0)` | `torch.tensor(2.0, requires_grad=True)` |
| `_op` (the operator that created this Value) | `.grad_fn` |
| `_prev` (parents in the DAG) | `.grad_fn.next_functions` |
| `_backward` closure | the C++ backward kernel registered for the op |
| `topo_sort + reverse` | walking `.grad_fn` graph in reverse |
| `self.grad += dout` | `.grad += grad_output` |
| `with no_op_tracking:` | `torch.no_grad()` |
| `value.detach()` (you didn't have one — but it'd be) | `.detach()` |

**The architecture is identical.** You wrote it for scalars; PyTorch ships it for arbitrary tensors with optimized C++/CUDA kernels.

---

## 11. Reflection

1. After `loss.backward()`, why is `intermediate.grad` `None` but `param.grad` populated?
2. You write `loss = (pred - target) ** 2; loss.backward()` and get an error. What's wrong?
3. What's the difference between `.detach()` and `torch.no_grad()`?
4. Why does `model.eval()` not also disable the graph? (Hint: sometimes you want gradients during eval, e.g., for adversarial examples.)
5. You're caching K, V tensors during generation. Should they require grad? Why or why not?

📖 Next deep-dive: [03_nn_module.md](03_nn_module.md) — every building block under `nn.*`.
