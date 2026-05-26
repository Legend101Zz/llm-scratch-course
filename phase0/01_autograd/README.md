# Module 1 — Build Your Own Autograd Engine (50 min)

> ⏱️ **50 minutes.** This is the single most important hour of the day. If you understand backprop *deeply* by the end, the rest of the course slots into place.

## 🎯 Goal

Build a `Value` class that:
- Stores a scalar number.
- Supports `+`, `*`, `**`, `tanh`, `exp`, `relu`.
- Builds a **computation graph** as you do operations.
- Has a `.backward()` method that fills in `.grad` on every node, automatically, by reverse-mode automatic differentiation.

This is essentially Andrej Karpathy's `micrograd`, re-derived. We do it because **PyTorch's autograd is the same idea, just on tensors and in C++**. If you grasp the scalar version, the tensor version is just bookkeeping.

---

## 1. The Math: Chain Rule, Slowly

Suppose $L = f(g(h(x)))$. We want $\frac{dL}{dx}$. Chain rule:

$$\frac{dL}{dx} = \frac{dL}{df} \cdot \frac{df}{dg} \cdot \frac{dg}{dh} \cdot \frac{dh}{dx}$$

In a neural net, $L$ is the loss, and $x$ is some weight buried 50 operations deep. The chain rule tells us we can compute the gradient by multiplying *local* derivatives along the path from $L$ back to $x$.

**Forward pass**: compute values left-to-right, building the graph.
**Backward pass**: starting from $L$ with $\frac{dL}{dL} = 1$, walk *backwards* and at each node apply the chain rule using each parent's local derivative.

### Worked example (do this on paper now):

Let $a = 2$, $b = -3$, $c = 10$, $d = a \cdot b$, $e = d + c$, $f = e \cdot 2$.

- Compute $f$ forward: $d = -6$, $e = 4$, $f = 8$.
- Now compute gradients backward, starting with $\frac{df}{df} = 1$:
  - $\frac{df}{de} = 2$ (because $f = 2e$).
  - $\frac{df}{dd} = \frac{df}{de} \cdot \frac{de}{dd} = 2 \cdot 1 = 2$.
  - $\frac{df}{dc} = 2 \cdot 1 = 2$.
  - $\frac{df}{da} = \frac{df}{dd} \cdot \frac{dd}{da} = 2 \cdot b = 2 \cdot (-3) = -6$.
  - $\frac{df}{db} = 2 \cdot a = 4$.

Verify: if you nudge $a$ by $+\epsilon$, $f$ should change by approximately $-6\epsilon$. ✅

This is **exactly** what `.backward()` does. We just automate it.

---

## 2. The Trick: Each Op Knows Its Local Derivative

For each operation we implement, we store a closure (`_backward`) that, **given the gradient of the output**, knows how to add to the gradient of the inputs.

```
out = a + b
# local derivatives:  d(out)/da = 1,   d(out)/db = 1
# so on backward: a.grad += 1 * out.grad ; b.grad += 1 * out.grad

out = a * b
# d(out)/da = b ; d(out)/db = a
# so on backward: a.grad += b * out.grad ; b.grad += a * out.grad

out = tanh(a)
# d(tanh(a))/da = 1 - tanh(a)^2
# so on backward: a.grad += (1 - out.data**2) * out.grad
```

The key invariant: **gradients accumulate (`+=`)**. If a node feeds into multiple children, its gradient is the sum of contributions from each.

---

## 3. Topological Order (why we sort before backprop)

We must compute a node's gradient **only after** all its children have computed theirs. So we build a topological ordering of the DAG and walk it in reverse. This is one DFS — easy.

---

## 4. Now Build It

Open [`starter.py`](starter.py). It has the skeleton with TODOs. Implement, in order:
1. `__init__`, `__add__`, `__mul__`
2. `tanh` (we'll use this for activation)
3. `_build_topo` and `backward`
4. `__pow__` (for $x^n$), `relu`, `exp`
5. The "reverse ops" (`__radd__`, `__rmul__`, `__sub__`, `__neg__`, `__truediv__`)

Then run [`test.py`](test.py) — it will check your engine matches PyTorch's autograd on a small expression. If your numbers match torch's, you've **literally rediscovered autograd**.

> 🚨 **Common bugs**:
> - Using `=` instead of `+=` on `.grad` (gradients overwrite instead of accumulate when a node feeds two children).
> - Forgetting to call `_backward` only after topo-sorting.
> - In `__pow__`, derivative is $n \cdot x^{n-1}$, not $n^{x-1}$. Sleep deprivation kills.

When tests pass, **don't peek at solution.py until they pass**. Then diff yours against mine and note differences.

---

## 5. Reflection (write in `notes.md`)

Answer in 2-3 lines each:
- Why does `.grad` accumulate with `+=` instead of being assigned?
- What happens if you forget the topological sort?
- Why is this "reverse-mode" instead of "forward-mode"? When would forward-mode win?

✅ **Move on to** [Module 2 — Build a Neural Net on top of this](../02_neural_net/README.md).
