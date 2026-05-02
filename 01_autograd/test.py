"""Compares your Value engine against PyTorch on a non-trivial expression."""

import torch
from starter import Value  # change to `solution` to test the reference impl


def test_against_torch():
    # ---- ours ----
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b ** 3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e ** 2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    a_ours, b_ours, g_ours = a.grad, b.grad, g.data

    # ---- torch ----
    a = torch.tensor([-4.0], requires_grad=True, dtype=torch.double)
    b = torch.tensor([2.0], requires_grad=True, dtype=torch.double)
    c = a + b
    d = a * b + b ** 3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e ** 2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    a_torch, b_torch, g_torch = a.grad.item(), b.grad.item(), g.item()

    print(f"forward  ours={g_ours:.6f}  torch={g_torch:.6f}")
    print(f"a.grad   ours={a_ours:.6f}  torch={a_torch:.6f}")
    print(f"b.grad   ours={b_ours:.6f}  torch={b_torch:.6f}")

    assert abs(g_ours - g_torch) < 1e-6, "forward mismatch"
    assert abs(a_ours - a_torch) < 1e-6, "a.grad mismatch"
    assert abs(b_ours - b_torch) < 1e-6, "b.grad mismatch"
    print("✅ all good — you have rediscovered backprop.")


if __name__ == "__main__":
    test_against_torch()
