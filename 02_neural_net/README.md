# Module 2 — Neural Net on Your Own Autograd (30 min)

> ⏱️ 30 minutes. We use the `Value` class you just built. No numpy, no torch — just your engine.
>
> 🎯 **The aha moment:** by the end, your `Value` class trains a real neural net on a real (toy) problem. Same math, same backprop, same gradient descent that runs GPT-4. Just smaller numbers.

## Goal

Build, in this order:
- `Neuron` — weighted sum of inputs + bias + activation.
- `Layer` — list of `Neuron`s.
- `MLP` — list of `Layer`s.
- Train it on **XOR** (the classic non-linear problem).

## 1. Anatomy of a Neuron

A single neuron is the dumbest "computational unit" in deep learning. It takes inputs, multiplies each by a learned weight, sums them, adds a bias, and squashes the result through a non-linear activation.

```
       x_1 ──×─── w_1 ──┐
                        │
       x_2 ──×─── w_2 ──┤
                        ├──→ Σ ──→ +b ──→ φ(·) ──→ y
       x_3 ──×─── w_3 ──┤
                        │
       x_n ──×─── w_n ──┘
```

In math:

$$y = \phi\!\left(\sum_{i=1}^{n} w_i x_i + b\right)$$

- $x_i$ — inputs (numbers we feed in).
- $w_i$ — **weights** (learnable). One per input.
- $b$ — **bias** (learnable). Lets the neuron shift its activation independently of the inputs.
- $\phi$ — **activation function**. We use `tanh` (smooth, output in $(-1, 1)$).

**Why a bias?** Without $b$, every neuron is forced to map $\mathbf{x} = 0$ to $\phi(0) = 0$. The bias lets the neuron's "decision boundary" shift away from the origin. Always have a bias.

## 2. Stacking Neurons → Layers → MLP

Multiple neurons receiving the same inputs in parallel = a **layer**. Each neuron has its own $w$'s and $b$, so the layer learns multiple different "features" of the input.

```
                  ┌──→ neuron_1 ──→ h_1
   x_1 ────────┬──┼──→ neuron_2 ──→ h_2
   x_2 ────────┼──┼──→ neuron_3 ──→ h_3
   x_3 ────────┴──┼──→ neuron_4 ──→ h_4
                  └──→  ...
                  (all neurons see all inputs)
```

A **multi-layer perceptron (MLP)** stacks these layers. The output of one layer is the input to the next:

```
   inputs ──► [ Layer 1: 4 neurons ] ──► [ Layer 2: 4 neurons ] ──► [ Layer 3: 1 neuron ] ──► output
   2 vals          (4 hidden vals)          (4 hidden vals)             (1 prediction)
```

For our XOR network: `MLP(2, [4, 4, 1])` means 2 inputs → hidden of 4 → hidden of 4 → 1 output.

**Convention:** the last layer is **linear** (no `tanh`) when the output is a regression target like 0/1. Squashing through tanh would compress (0,1) into a smaller range and slow learning.

## 3. The Math, End to End

For one training example $(x, y)$:

**Forward pass** (compute prediction $\hat{y}$):
```
h¹  = tanh(W¹ x  + b¹)        ← layer 1, 4 neurons
h²  = tanh(W² h¹ + b²)        ← layer 2, 4 neurons
ŷ   = W³ h² + b³               ← layer 3, 1 neuron, linear
```

**Loss** (mean squared error over $N$ examples):

$$\mathcal{L} = \frac{1}{N}\sum_{n=1}^{N} (\hat{y}_n - y_n)^2$$

**Backward pass** (gradients via your `Value` engine — for FREE):
- Call `loss.backward()` once.
- Every weight and bias in every layer now has `.grad` filled in.

**Update** (gradient descent):

$$\theta \leftarrow \theta - \eta \cdot \frac{\partial \mathcal{L}}{\partial \theta} \quad\quad\text{for every parameter } \theta$$

In code:
```python
for p in model.parameters():
    p.data -= lr * p.grad
```

**Zero gradients before the next step.** Because your engine accumulates with `+=`, leftover gradients from the previous step would corrupt the next update:
```python
for p in model.parameters():
    p.grad = 0.0
```

> 🔥 **The bug everyone hits once:** forgetting to zero gradients. Loss either explodes or stalls. Mark this paragraph.

## 4. Why XOR? (The Whole Point of Hidden Layers)

XOR is the historical "neural nets vs. perceptrons" benchmark. A single linear neuron **cannot** solve it. To see why, plot the four points:

```
   y=1 ●────────────○ y=0
       │            │
       │            │
       │            │
   y=0 ○────────────● y=1
       (0,0)        (1,0)
```

(`●` = target 1, `○` = target 0.)

There's no straight line that puts all the `●`'s on one side and all the `○`'s on the other. A single linear neuron defines a line — `w₁x₁ + w₂x₂ + b = 0`. Useless here.

**A hidden layer with a non-linearity changes the geometry.** Layer 1 maps the input plane into a 4-dim space where the four points *can* be separated by a line. Layer 2 then draws that line.

This is the abstract reason deep learning works at all: stack non-linear feature transformers, and any continuous function becomes representable. It's called the **Universal Approximation Theorem**, but the *practical* reason XOR works is much simpler — the hidden layer reshapes space until the problem is linearly separable.

```
   raw 2D space          after Layer 1 (4D, hard to draw)        Layer 2 separates
   ●        ○                       *imagine a hyperplane*           cleanly
                                     existing in 4D space
   ○        ●            ─────────────────────────────────►          ✓
```

### Why `tanh` and not just linear?

If every layer is linear (no $\phi$), the whole MLP collapses to a single matrix multiply:

$$W^3 (W^2 (W^1 x + b^1) + b^2) + b^3 = W' x + b'$$

— same expressiveness as one layer. **The non-linearity is what makes depth meaningful.** Without it, you have a fancy linear regression. Tanh, ReLU, GELU, sigmoid — they all work for this purpose (with different tradeoffs in speed and gradient behavior).

## 5. Walkthrough: One Training Step, In Slow Motion

Suppose at step 0 our network is randomly initialized and we feed it `x = (1, 0)` with target `y = 1`:

```
1) Forward
   x = [1, 0]
   layer1 → h¹ = tanh(W¹·x + b¹)   say [0.3, -0.5, 0.1, 0.7]   (4 values)
   layer2 → h² = tanh(W²·h¹ + b²)  say [-0.2, 0.4, 0.8, -0.1]
   layer3 → ŷ  = W³·h² + b³        say 0.42

2) Loss
   loss = (0.42 - 1)² = 0.336

3) Backward (loss.backward())
   - dL/dŷ = 2(0.42 - 1) = -1.16
   - chain rule pushes this back through layer3, layer2, layer1
   - every w_i, b_i now has a .grad value

4) Update (lr=0.1)
   - For each parameter p:  p.data -= 0.1 * p.grad
   - Net moves slightly in the direction that decreases loss for this example

5) Zero grads
   - p.grad = 0  for every p
```

Repeat for the other 3 examples (or for all 4 in a "batch" — sum their losses and backward once, which is what the solution does).

## 6. The Decision Boundary Story

After training, you can ask the network its prediction at every point in the input plane and color by output. You'll see something like:

```
               x₂
               1.0 ┌──────────────────────────────┐
                   │ ░░░░░░░░░░░░░░░░ █████████████│   ░ = predicted close to 0
                   │ ░░░░░░░░░░░░░ █████████████████│   █ = predicted close to 1
                   │ ░░░░░░░░░ █████████████████████│
                   │ ░░░░░░ █████████████████████████│
                   │ ░░░ █████████████████████████████│
                   │ ███████████████ ░░░░░░░░░░░░░░░░│
                   │ █████████████ ░░░░░░░░░░░░░░░░░░│
                   │ ██████████ ░░░░░░░░░░░░░░░░░░░░░│
                   │ ███████ ░░░░░░░░░░░░░░░░░░░░░░░░│
               0.0 └──────────────────────────────┘
                   0.0                            1.0  x₁
```

The boundary is **curved** — it has to be, to separate XOR. That curvature comes from `tanh`. The network learned to approximate a curve that splits the plane the way XOR demands.

## 7. Now build it

Open [`starter.py`](starter.py). TODOs in order:
1. `Neuron.__init__` — weights uniform in [-1, 1], bias = 0.
2. `Neuron.__call__` — sum, add bias, optionally tanh.
3. `Layer.__init__` and `MLP.__init__` — list comprehensions.
4. The training loop — forward, loss, zero, backward, update.

Then run:
```bash
python starter.py
```

You should see the loss drop from ~1.0 to <0.05 over ~500 steps. Final predictions should round to (0, 1, 1, 0). If your loss explodes or stalls, see Section 9 below.

## 8. Common Bugs & How They Look

| Symptom | Cause | Fix |
|---|---|---|
| Loss exactly stays around 0.25 | All four predictions converge to 0.5 (the average) | Net is too small / bad init / no non-linearity |
| Loss jumps wildly each step | LR too high | Divide LR by 10 |
| Loss drops to ~0.5 then plateaus | Stuck in a saddle / bad init | Re-run (different random seed) |
| Loss grows unboundedly | Forgot to zero `.grad`, or LR too high | Add `p.grad = 0` before backward; lower LR |
| Final preds are all near 0.5 | The output layer has `tanh` (compresses range) | Make the last layer linear |
| Loss never moves | Forgot to call `loss.backward()`, or forgot to update `p.data` | Re-check the loop |
| `AttributeError: 'float' object has no attribute 'data'` | Loss became a Python float, not a Value | Probably summed with the wrong type — keep everything as `Value` |

## 9. Reflection (write in `notes.md`)

Answer in 2–3 lines each — write before peeking at `solution.py`:

1. **Chain rule by hand.** Write $\partial \mathcal{L}/\partial w_i$ where $w_i$ is a weight in the **first** layer. How many factors does the chain have? Which one is the "local derivative" of `tanh`?
2. **Why does pure-linear (no tanh) fail on XOR?** Show by collapsing two linear layers into one.
3. **LR sweep.** What happens if you set lr=100? lr=0.0001? Try both — you'll see one explode, one stall. Why?
4. **Symmetry.** What if we initialized all weights to *zero* instead of random? (Hint: every neuron in a layer would compute identical outputs and receive identical gradients — the network would never break symmetry. **This is why we initialize randomly.**)
5. **MSE for classification?** Why is using MSE on a 0/1 target weird in theory but fine in practice for XOR? (We'll use proper cross-entropy for actual classification in later modules.)

## 10. What this prepares you for

By writing this from scratch:

- You've seen `Neuron → Layer → MLP` — the same hierarchy as `nn.Linear → nn.Sequential → nn.Module` in PyTorch.
- The training loop pattern (`forward → loss → zero → backward → update`) is **identical** in PyTorch — only the autograd vendor changes.
- The bugs you'll hit here (forgetting zero_grad, wrong activation on output, exploding loss) are the **exact same** bugs you'll hit when training a 7B-parameter LLM. The scale changes; the failure modes don't.

✅ Next: [Module 3 — Tokenization + Bigram](../03_tokenizer_bigram/README.md)
