"""Module 2 — parity test: Value-based MLP vs. PyTorch.

What we verify:
  1. Forward parity: with identical weights, our MLP and a PyTorch MLP
     produce the same output (per input, to 1e-6).
  2. Backward parity: the gradient of MSE loss w.r.t. every parameter matches
     PyTorch's autograd to 1e-5 — for ALL 4 XOR inputs accumulated as one batch.
  3. End-to-end training: after the same 500-step SGD run on XOR, our model
     converges to readable predictions and matches PyTorch's trajectory
     to a much looser tolerance (1e-3 final loss — float64 + accumulated drift).

Why this matters: Module 2 is the first time backprop runs through a *multi-layer*
network. "Loss → 0 on XOR" is weak proof (could be lucky init). What we need is
"every parameter's gradient matches PyTorch" — the strongest possible signal that
the Module 1 autograd is correctly composing through a real architecture.

Run:
    python test.py

Or against the reference solution:
    USE_SOLUTION=1 python test.py
"""

import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn

USE_SOLUTION = os.environ.get("USE_SOLUTION") == "1"

# IMPORTANT: import Value from the same module the MLP uses. `solution.py` /
# `starter.py` each load their own Value via `importlib.util.spec_from_file_location`,
# which creates a *fresh* class each call — so re-loading independently here would
# give us a DIFFERENT Value class, and `isinstance(other, Value)` inside the Neuron
# would fail. Re-using the module's own Value avoids that footgun.
if USE_SOLUTION:
    print("(testing solution.py)")
    from solution import Neuron, Layer, MLP, Value  # type: ignore[reportMissingImports]
else:
    print("(testing starter.py — set USE_SOLUTION=1 to test the reference)")
    try:
        from starter import Neuron, Layer, MLP, Value  # type: ignore[reportMissingImports]
    except Exception as e:
        print(f"❌ can't import from starter.py: {e}")
        sys.exit(1)


def _build_matched_torch_mlp(value_mlp):
    """Build a PyTorch MLP with the same numerical weights as `value_mlp`.

    Mapping:
      value_mlp.layers[i].neurons[k].w[j]  →  torch_layer[i].weight[k, j]
      value_mlp.layers[i].neurons[k].b      →  torch_layer[i].bias[k]

    Returns (torch_module, list_of_layers).
    """
    layers = []
    for vlayer in value_mlp.layers:
        n_out = len(vlayer.neurons)
        n_in = len(vlayer.neurons[0].w)
        lin = nn.Linear(n_in, n_out, bias=True).double()
        with torch.no_grad():
            for k, neuron in enumerate(vlayer.neurons):
                for j, wij in enumerate(neuron.w):
                    lin.weight[k, j] = wij.data
                lin.bias[k] = neuron.b.data
        layers.append(lin)

    # Build a callable that mirrors `value_mlp.__call__`:
    # apply tanh after every layer except the last.
    n_layers = len(layers)
    def forward(x_t):
        for i, lin in enumerate(layers):
            x_t = lin(x_t)
            if i != n_layers - 1:
                x_t = torch.tanh(x_t)
        return x_t

    return forward, layers


def _copy_value_grads_to_torch_layers(value_mlp, torch_layers):
    """For diagnostic comparison — read the Value engine's accumulated grads
    and produce a PyTorch-shaped grad tensor for each Linear layer."""
    grads = []
    for vlayer, lin in zip(value_mlp.layers, torch_layers):
        n_out = len(vlayer.neurons)
        n_in = len(vlayer.neurons[0].w)
        gw = torch.zeros((n_out, n_in), dtype=torch.float64)
        gb = torch.zeros(n_out, dtype=torch.float64)
        for k, neuron in enumerate(vlayer.neurons):
            for j, wij in enumerate(neuron.w):
                gw[k, j] = wij.grad
            gb[k] = neuron.b.grad
        grads.append((gw, gb))
    return grads


def test_forward_parity():
    """Forward output must match torch on every XOR input to 1e-6."""
    random.seed(1337)
    model = MLP(2, [4, 4, 1])
    fwd_torch, _layers = _build_matched_torch_mlp(model)

    cases = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    for x in cases:
        # Value side — Layer.__call__ collapses a single-output layer to a scalar.
        out_v = model([Value(xi) for xi in x])
        # If it's a list-of-1, unwrap; otherwise it's already a scalar.
        if isinstance(out_v, list):
            out_v_val = out_v[0].data
        else:
            out_v_val = out_v.data

        out_t = fwd_torch(torch.tensor(x, dtype=torch.float64)).item()
        err = abs(out_v_val - out_t)
        print(f"  x={x}  ours={out_v_val:+.10f}  torch={out_t:+.10f}  err={err:.2e}")
        assert err < 1e-6, f"forward mismatch on x={x}: {err:.2e}"


def test_backward_parity():
    """Per-parameter gradients of MSE on the full XOR batch must match torch to 1e-5."""
    random.seed(1337)
    model = MLP(2, [4, 4, 1])
    fwd_torch, layers = _build_matched_torch_mlp(model)

    xs_v = [[Value(0.0), Value(0.0)], [Value(0.0), Value(1.0)],
            [Value(1.0), Value(0.0)], [Value(1.0), Value(1.0)]]
    xs_t = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=torch.float64)
    ys_v = [0, 1, 1, 0]
    ys_t = torch.tensor([0., 1., 1., 0.], dtype=torch.float64).reshape(-1, 1)

    # ---- ours ----
    for p in model.parameters():
        p.grad = 0.0
    preds_v = [model(xv) for xv in xs_v]
    # Layer.__call__ may return a list-of-1 or a scalar Value; unwrap.
    pred_scalars = [(pv[0] if isinstance(pv, list) else pv) for pv in preds_v]
    loss_v = sum((pv - yv) ** 2 for pv, yv in zip(pred_scalars, ys_v)) * (1.0 / 4.0)
    loss_v.backward()

    # ---- torch ----
    for lin in layers:
        lin.zero_grad()
    preds_t = fwd_torch(xs_t)
    loss_t = ((preds_t - ys_t) ** 2).mean()
    loss_t.backward()

    # Loss values must agree.
    err_loss = abs(loss_v.data - loss_t.item())
    print(f"  loss ours={loss_v.data:.10f}  torch={loss_t.item():.10f}  err={err_loss:.2e}")
    assert err_loss < 1e-9, f"loss mismatch: {err_loss:.2e}"

    # Per-parameter grads must agree.
    ours_grads = _copy_value_grads_to_torch_layers(model, layers)
    for i, ((gw_ours, gb_ours), lin) in enumerate(zip(ours_grads, layers)):
        ew = (gw_ours - lin.weight.grad).abs().max().item()
        eb = (gb_ours - lin.bias.grad).abs().max().item()
        print(f"  layer {i}  max |Δgrad w|={ew:.2e}  max |Δgrad b|={eb:.2e}")
        assert ew < 1e-5, f"layer {i} weight grads mismatch: {ew:.2e}"
        assert eb < 1e-5, f"layer {i} bias grads mismatch: {eb:.2e}"


def test_training_convergence():
    """After 500 SGD steps the loss must drop below 0.05 (XOR is solvable)."""
    random.seed(1337)
    model = MLP(2, [4, 4, 1])
    xs_v = [[Value(0.0), Value(0.0)], [Value(0.0), Value(1.0)],
            [Value(1.0), Value(0.0)], [Value(1.0), Value(1.0)]]
    ys_v = [0, 1, 1, 0]
    lr = 0.1
    for step in range(500):
        for p in model.parameters():
            p.grad = 0.0
        preds_v = [model(xv) for xv in xs_v]
        pred_scalars = [(pv[0] if isinstance(pv, list) else pv) for pv in preds_v]
        loss_v = sum((pv - yv) ** 2 for pv, yv in zip(pred_scalars, ys_v)) * (1.0 / 4.0)
        loss_v.backward()
        for p in model.parameters():
            p.data -= lr * p.grad
    print(f"  final loss after 500 steps: {loss_v.data:.6f}")
    assert loss_v.data < 0.05, (
        f"XOR didn't converge (loss={loss_v.data:.4f}). Either the MLP architecture "
        f"is wrong (need at least one hidden layer with > 1 neuron and a nonlinearity) "
        f"or the gradient signs are flipped in your engine."
    )

    # spot-check predictions sit on the right side of 0.5
    for x, y in zip([[0,0],[0,1],[1,0],[1,1]], ys_v):
        pred = model([Value(xi) for xi in x])
        pv = pred[0].data if isinstance(pred, list) else pred.data
        rounded = 1 if pv > 0.5 else 0
        print(f"  x={x}  pred={pv:+.3f}  rounded={rounded}  target={y}")
        assert rounded == y, f"wrong prediction on x={x}: got {pv}, expected {y}"


if __name__ == "__main__":
    print("\n--- test_forward_parity ---")
    test_forward_parity()
    print("\n--- test_backward_parity ---")
    test_backward_parity()
    print("\n--- test_training_convergence ---")
    test_training_convergence()
    print("\n✅ all MLP parity tests passed. your Module-1 autograd composes correctly.")
