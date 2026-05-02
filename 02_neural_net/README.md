# Module 2 — Neural Net on Your Own Autograd (30 min)

> ⏱️ 30 minutes. We use the `Value` class you just built. No numpy, no torch — just your engine.

## Goal

Build, in this order:
- `Neuron` — weighted sum of inputs + bias + activation.
- `Layer` — list of `Neuron`s.
- `MLP` — list of `Layer`s.
- Train it on **XOR** (the classic non-linear problem).

When the loss visibly drops, you've trained a neural network using your own autograd. That's a real moment. Stop and feel it.

## The Math (you already know this)

A neuron computes:

$$y = \phi\!\left(\sum_{i} w_i x_i + b\right)$$

where $\phi$ is an activation (we'll use `tanh` for hidden layers, no activation for the output layer in this regression-style XOR setup).

Loss for XOR (mean squared error):

$$\mathcal{L} = \frac{1}{N}\sum_{n=1}^{N} (y_n - \hat{y}_n)^2$$

Gradient descent:

$$\theta \leftarrow \theta - \eta \cdot \frac{\partial \mathcal{L}}{\partial \theta}$$

Your `Value` engine computes $\frac{\partial \mathcal{L}}{\partial \theta}$ for free. You write the forward pass, call `.backward()`, then for every parameter `p`: `p.data -= lr * p.grad`. Then **zero the gradients** before the next step (because of `+=`).

> 🔥 The most common bug: forgetting to zero gradients between steps. Loss explodes. You will do this once.

## Why XOR?

XOR is the historical "neural nets vs. perceptrons" problem. A single linear layer can't separate it (the points are not linearly separable). One hidden layer + a non-linearity can. So XOR proves your network has actually learned a non-linear function — not just memorized an average.

```
Input    Target
(0, 0) →   0
(0, 1) →   1
(1, 0) →   1
(1, 1) →   0
```

## Now build it

Open [`starter.py`](starter.py). TODOs in order. Then run:

```bash
python starter.py
```

You should see the loss drop from ~1.0 to <0.05 over a few hundred steps.

Then peek at `solution.py`. Note any differences in your training loop.

## Reflection (in `notes.md`)

- Write the chain rule for $\partial \mathcal{L}/\partial w_i$ where $w_i$ is in the first layer.
- Why does using only linear (no `tanh`) layers fail on XOR?
- What happens if you set learning rate to 100? To 0.0001?

✅ Next: [Module 3 — Tokenization + Bigram](../03_tokenizer_bigram/README.md)
