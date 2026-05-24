"""Module 4 — parity test: NumPy MHA vs. PyTorch.

What we verify:
  1. softmax(x) matches torch.softmax to 1e-6 (forward only).
  2. single_head_attention with same weights matches a PyTorch reference impl
     (manual scaled-dot-product) to 1e-5.
  3. multi_head_attention with same weights matches a PyTorch reference
     (loop over heads + concat + W_O) to 1e-5.
  4. (Sanity) Causal mask is upper-triangular and applied BEFORE softmax —
     verified by checking attention[0] only attends to position 0.

Why these tests matter (per CONVENTIONS.md): "From scratch first, framework
second" is only meaningful if you can prove the from-scratch version matches
the framework. Without parity, "I built attention" reduces to "I wrote code
that ran" — much weaker claim.

Run:
    python test.py

Or test the reference solution:
    USE_SOLUTION=1 python test.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

USE_SOLUTION = os.environ.get("USE_SOLUTION") == "1"

if USE_SOLUTION:
    print("(testing solution.py)")
    from solution import softmax, single_head_attention, multi_head_attention
else:
    print("(testing starter.py — set USE_SOLUTION=1 to test the reference)")
    try:
        from starter import softmax, single_head_attention, multi_head_attention
    except Exception as e:
        print(f"❌ can't import from starter.py: {e}")
        sys.exit(1)


# ----- shared fixture: same shapes and same random weights -----
np.random.seed(0)
T, d, h = 8, 32, 4
d_k = d // h
X_np = np.random.randn(T, d).astype(np.float64)
heads_np = [
    (np.random.randn(d, d_k), np.random.randn(d, d_k), np.random.randn(d, d_k))
    for _ in range(h)
]
W_O_np = np.random.randn(h * d_k, d)

# torch mirrors
X_t = torch.from_numpy(X_np).double()
heads_t = [tuple(torch.from_numpy(w).double() for w in head) for head in heads_np]
W_O_t = torch.from_numpy(W_O_np).double()


def test_softmax():
    """Your softmax must match torch.softmax to 1e-6."""
    x = np.random.randn(8, 16)
    yours = softmax(x, axis=-1)
    theirs = F.softmax(torch.from_numpy(x), dim=-1).numpy()
    err = np.abs(yours - theirs).max()
    print(f"  max abs error: {err:.2e}")
    assert err < 1e-6, f"softmax mismatch: {err:.2e}"
    # also: rows must sum to 1
    assert np.allclose(yours.sum(axis=-1), 1.0)


def _torch_single_head(X, W_Q, W_K, W_V, causal=True):
    """PyTorch reference single-head attention with explicit weights, no bias."""
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    scores = Q @ K.transpose(-2, -1) / (W_K.shape[-1] ** 0.5)
    if causal:
        T_ = X.shape[-2]
        mask = torch.triu(torch.ones(T_, T_, dtype=torch.bool, device=X.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return attn @ V


def test_single_head_causal():
    """Causal single-head must match the PyTorch reference to 1e-5."""
    W_Q, W_K, W_V = heads_np[0]
    yours = single_head_attention(X_np, W_Q, W_K, W_V, causal=True)
    theirs = _torch_single_head(X_t, *heads_t[0], causal=True).numpy()
    err = np.abs(yours - theirs).max()
    print(f"  max abs error: {err:.2e}")
    assert err < 1e-5, f"single-head causal mismatch: {err:.2e}"


def test_single_head_uncausal():
    """Bidirectional (no mask) single-head must also match."""
    W_Q, W_K, W_V = heads_np[1]
    yours = single_head_attention(X_np, W_Q, W_K, W_V, causal=False)
    theirs = _torch_single_head(X_t, *heads_t[1], causal=False).numpy()
    err = np.abs(yours - theirs).max()
    print(f"  max abs error: {err:.2e}")
    assert err < 1e-5, f"single-head bidirectional mismatch: {err:.2e}"


def test_multi_head():
    """Multi-head (concat heads, then W_O) must match the PyTorch reference."""
    yours = multi_head_attention(X_np, heads_np, W_O_np)
    head_outs = [_torch_single_head(X_t, *head, causal=True) for head in heads_t]
    theirs = (torch.cat(head_outs, dim=-1) @ W_O_t).numpy()
    err = np.abs(yours - theirs).max()
    print(f"  max abs error: {err:.2e}")
    assert err < 1e-5, f"multi-head mismatch: {err:.2e}"


def test_causal_mask_structure():
    """Position 0 can only see position 0. Position 1 can see {0, 1}. Etc.

    Tests this by checking that the output at position 0 is independent of
    positions [1:]: shuffle them and confirm output[0] is unchanged.
    """
    W_Q, W_K, W_V = heads_np[0]
    out_original = single_head_attention(X_np, W_Q, W_K, W_V, causal=True)

    # Perturb positions [1:]: should leave output[0] unchanged under causal masking.
    X_perturbed = X_np.copy()
    X_perturbed[1:] += np.random.randn(*X_perturbed[1:].shape) * 10  # huge change
    out_perturbed = single_head_attention(X_perturbed, W_Q, W_K, W_V, causal=True)

    err = np.abs(out_original[0] - out_perturbed[0]).max()
    print(f"  output[0] drift after perturbing positions [1:]: {err:.2e}")
    assert err < 1e-9, (
        f"output[0] changed by {err:.2e} after perturbing later positions — "
        "this means the causal mask is not blocking future attention. "
        "Common bug: masking AFTER softmax (which sets zeros but the row no longer sums to 1)."
    )


if __name__ == "__main__":
    print("\n--- test_softmax ---")
    test_softmax()
    print("\n--- test_single_head_causal ---")
    test_single_head_causal()
    print("\n--- test_single_head_uncausal ---")
    test_single_head_uncausal()
    print("\n--- test_multi_head ---")
    test_multi_head()
    print("\n--- test_causal_mask_structure ---")
    test_causal_mask_structure()
    print("\n✅ all attention parity tests passed.")
