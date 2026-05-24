# `01_autograd/hand_math/` — pencil derivations

> This folder is where **you** prove you derived the math, not just read it.
> Per [`../../CONVENTIONS.md`](../../CONVENTIONS.md), every Phase-0 module needs at
> least one handwritten derivation committed here. The git log is the evidence.

## What belongs in this folder for Module 1

At minimum, commit the derivation of:

### 1. The chain rule through `Value.__mul__`, `Value.__add__`, `Value.tanh`

Take a small expression, e.g.:

> `L = ((a * b) + c).tanh()`, with `a = 2`, `b = -3`, `c = 10`.

Derive on paper:
- The forward pass values at each node.
- The gradients `∂L/∂a`, `∂L/∂b`, `∂L/∂c` via the chain rule, walking the graph backwards.
- Show why `tanh.backward` uses `(1 - t²)` and where the `t²` comes from
  (`d/dx tanh(x) = 1 - tanh²(x)`).

### 2. Why reverse-mode AD is the right choice for `N params → 1 loss`

One paragraph + a sketch:
- Forward-mode AD costs `O(N)` passes for `N` inputs.
- Reverse-mode AD costs `O(M)` passes for `M` outputs.
- LLM training: `N` ≈ billions of params, `M` = 1 scalar loss → reverse-mode wins by `N/M ≈ 10⁹`×.

This is a one-line argument but it's the single most important property of the autograd
mental model. Write it down.

## How to commit

Two acceptable formats:

1. **Photo of the page** (`01_chain_rule.jpg`, `02_reverse_mode.jpg`). Phone-quality is fine; just make sure the writing is legible.
2. **Transcribed to LaTeX in markdown** (`01_chain_rule.md`, `02_reverse_mode.md`). Use a chatbot to transcribe if you want — the *derivation* is yours, the typesetting can be assisted.

Hybrid is also fine: commit the photo + a short markdown summary explaining the result.

## How the mentor checks this

At the start of the next session, the mentor will spot-pick one step and ask you to re-derive
it cold (without looking at the file). If you can't, you haven't earned it — repeat the
derivation and re-commit. (Per Hard Rule #5: force recall.)
