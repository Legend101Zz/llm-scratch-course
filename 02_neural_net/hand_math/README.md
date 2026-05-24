# `02_neural_net/hand_math/` — pencil derivations

> Per [`../../CONVENTIONS.md`](../../CONVENTIONS.md), every Phase-0 module needs at least one handwritten derivation here.

## What belongs in this folder for Module 2

At minimum, commit the derivation of:

### 1. MSE gradient flow through one neuron

For a single neuron `y = tanh(w · x + b)` with MSE loss `L = ½ (y - t)²`:

- Show by hand: `∂L/∂w_i = (y - t) · (1 - y²) · x_i`
  - Step 1: `∂L/∂y = y - t`
  - Step 2: `∂y/∂z = 1 - y²` where `z = w·x + b` (tanh derivative)
  - Step 3: `∂z/∂w_i = x_i`
  - Compose: chain rule → product of the three.
- Show: `∂L/∂b = (y - t) · (1 - y²)` (same chain, last term is 1)

This is the derivation your `solution.py` is mechanically computing, one Value at a time, in `backward()`. Make sure you can write it without looking.

### 2. Finite-difference check (numerical sanity)

For one parameter `w_0`, compute `(L(w_0 + ε) - L(w_0 - ε)) / (2ε)` for `ε = 1e-4` and compare to the autograd gradient. They should match to ~1e-7.

Commit either:
- A photo of the calculation on paper, or
- A short script `finite_diff_check.py` (small enough to live in `hand_math/` rather than the module root) and its output.

### 3. The decision-boundary intuition

XOR is not linearly separable. Show by hand:
- Draw the 4 XOR points in 2D.
- Try to draw a single line that puts (0,0) and (1,1) on one side and (0,1), (1,0) on the other. You can't.
- Now sketch what your trained hidden layer does: the tanh activations effectively *fold* the input space so the output layer's linear boundary works.

This visual explains why an MLP needs ≥ 1 hidden layer with a nonlinearity to solve XOR.

## Format

Photo OR transcribed-to-markdown — same rules as `01_autograd/hand_math/`. The mentor spot-checks one step cold at the start of the next session.
