"""Module 6 — PyTorch tensor operation drills.

Fill in each TODO. Run the file: every assertion should pass silently.
If one fails, fix that block before moving on. No peeking at the solution
until you've struggled.

    python tensor_drills.py
"""

import math
import torch
import torch.nn as nn

torch.manual_seed(0)


# -----------------------------------------------------------------------------
# 1. Create a (4, 5) tensor of zeros; set element [2, 3] to 7.
# -----------------------------------------------------------------------------
# TODO: build x as described
x = ...  # TODO

assert x.shape == (4, 5)
assert x[2, 3].item() == 7
assert x.sum().item() == 7  # everything else is zero


# -----------------------------------------------------------------------------
# 2. Build a random tensor of shape (B=2, T=4, d=8).
#    Compute the mean across the time dim T. Resulting shape should be (B, d).
# -----------------------------------------------------------------------------
B, T, d = 2, 4, 8
x = torch.randn(B, T, d)
# TODO: mean across T
mean_x = ...  # TODO

assert mean_x.shape == (B, d)
# sanity: mean across T should equal manually averaging the slices
expected = (x[:, 0] + x[:, 1] + x[:, 2] + x[:, 3]) / T
assert torch.allclose(mean_x, expected, atol=1e-6)


# -----------------------------------------------------------------------------
# 3. Given q, k of shape (B, T, d), compute scaled dot-product scores (B, T, T).
#    Formula: scores[b, i, j] = (q[b, i] . k[b, j]) / sqrt(d)
# -----------------------------------------------------------------------------
B, T, d = 2, 5, 16
q = torch.randn(B, T, d)
k = torch.randn(B, T, d)
# TODO: compute scores
scores = ...  # TODO

assert scores.shape == (B, T, T)
# spot-check: scores[0, 1, 2] should equal q[0,1] dot k[0,2] / sqrt(d)
expected = (q[0, 1] * k[0, 2]).sum() / math.sqrt(d)
assert torch.allclose(scores[0, 1, 2], expected, atol=1e-5)


# -----------------------------------------------------------------------------
# 4. Apply a causal mask to scores: upper triangle (j > i) → -inf.
#    Hint: torch.tril returns a lower-triangular ones matrix.
# -----------------------------------------------------------------------------
# TODO: build the mask of shape (T, T) and use masked_fill on scores
masked = ...  # TODO

assert masked.shape == (B, T, T)
# anything in the strict upper triangle should now be -inf
for i in range(T):
    for j in range(i + 1, T):
        assert torch.isinf(masked[0, i, j]) and masked[0, i, j] < 0
# the diagonal and below should be unchanged
for i in range(T):
    for j in range(i + 1):
        assert torch.allclose(masked[0, i, j], scores[0, i, j])


# -----------------------------------------------------------------------------
# 5. Softmax over the keys (last) dim. Verify each row sums to 1.
# -----------------------------------------------------------------------------
import torch.nn.functional as F
# TODO: softmax along the last dim
attn = ...  # TODO

assert attn.shape == (B, T, T)
row_sums = attn.sum(dim=-1)
assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
# row 0 should attend ONLY to position 0 (only valid key) → one-hot
assert torch.allclose(attn[0, 0, 0], torch.tensor(1.0))
assert attn[0, 0, 1:].abs().sum().item() < 1e-6


# -----------------------------------------------------------------------------
# 6. Multiply attn by v of shape (B, T, d) to get the attention output (B, T, d).
# -----------------------------------------------------------------------------
v = torch.randn(B, T, d)
# TODO: compute output
out = ...  # TODO

assert out.shape == (B, T, d)
# spot-check: token 0's output is just v[0, 0] (since it only attends to itself)
assert torch.allclose(out[0, 0], v[0, 0], atol=1e-5)


# -----------------------------------------------------------------------------
# 7. Reshape (B, T, n_head * d_head) into (B, n_head, T, d_head).
#    This is the "split heads" step you'll write many times in transformers.
# -----------------------------------------------------------------------------
B, T, n_head, d_head = 2, 5, 4, 8
x = torch.randn(B, T, n_head * d_head)
# TODO: reshape via view + transpose (or permute) to (B, n_head, T, d_head)
x_split = ...  # TODO

assert x_split.shape == (B, n_head, T, d_head)
# correctness: head h's content at time t should be x[b, t, h*d_head : (h+1)*d_head]
b, h, t = 1, 2, 3
assert torch.allclose(
    x_split[b, h, t],
    x[b, t, h * d_head : (h + 1) * d_head],
    atol=1e-6,
)


# -----------------------------------------------------------------------------
# 8. Gradient of (x**3).sum() w.r.t. x. Should equal 3 * x**2 elementwise.
# -----------------------------------------------------------------------------
x = torch.randn(5, requires_grad=True)
# TODO: forward + backward
y = ...  # TODO: compute (x**3).sum()
# TODO: call backward

assert x.grad is not None
assert torch.allclose(x.grad, 3 * x.detach() ** 2, atol=1e-5)


# -----------------------------------------------------------------------------
# 9. nn.Linear(8, 8): count its params, run forward+backward, inspect .grad.
#    A Linear has weight (out, in) and bias (out,). For (8, 8): 8*8 + 8 = 72.
# -----------------------------------------------------------------------------
lin = nn.Linear(8, 8)
n_params = sum(p.numel() for p in lin.parameters())
assert n_params == 72

# Run forward + backward.
inp = torch.randn(3, 8)            # batch of 3
# TODO: forward through lin, sum the output, backward
out = ...  # TODO
loss = ...  # TODO
# TODO: zero grads, then backward

assert lin.weight.grad is not None
assert lin.weight.grad.shape == lin.weight.shape
assert lin.bias.grad is not None


# -----------------------------------------------------------------------------
# 10. Confirm Dropout behaves differently in train vs. eval mode.
# -----------------------------------------------------------------------------
class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(0.5)
    def forward(self, x):
        return self.drop(x)

m = Tiny()
x = torch.ones(1000)

# In train mode, ~half should be zeroed.
m.train()
y_train = m(x)
zero_frac_train = (y_train == 0).float().mean().item()
assert 0.3 < zero_frac_train < 0.7  # roughly 0.5

# In eval mode, dropout is off → output equals input.
m.eval()
y_eval = m(x)
assert torch.allclose(y_eval, x)


print("✅ drills 1–10 passed.")


# -----------------------------------------------------------------------------
# 11. The bridge drill: rebuild Module 4's multi-head attention in PyTorch
#     and verify it matches your NumPy implementation to 1e-5.
#
#     This drill exists because the value of "from scratch first, framework
#     second" depends on the framework version being a faithful translation,
#     not a different algorithm. Build both. Make them agree.
#
#     We use the *solution* from Module 4 as ground truth so this drill stays
#     independent of whether your starter is complete.
# -----------------------------------------------------------------------------
import os, numpy as np, importlib.util, math

# Load the Module 4 solution by path (modules don't share __init__.py).
_spec = importlib.util.spec_from_file_location(
    "m4_solution",
    os.path.join(os.path.dirname(__file__), "..", "04_attention_scratch", "solution.py"),
)
_m4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m4)

# Shared fixture.
np.random.seed(0)
T, d, h = 8, 32, 4
d_k = d // h
X_np = np.random.randn(T, d).astype(np.float64)
heads_np = [
    (np.random.randn(d, d_k), np.random.randn(d, d_k), np.random.randn(d, d_k))
    for _ in range(h)
]
W_O_np = np.random.randn(h * d_k, d)

# Ground truth: numpy MHA from Module 4.
gt = _m4.multi_head_attention(X_np, heads_np, W_O_np)

# Same weights, but as torch tensors.
X_t = torch.from_numpy(X_np).double()
heads_t = [tuple(torch.from_numpy(w).double() for w in head) for head in heads_np]
W_O_t = torch.from_numpy(W_O_np).double()


def my_mha_torch(X, heads, W_O, causal=True):
    """TODO: implement multi-head attention in pure torch ops.

    Steps (one head at a time, then concat):
      for each (W_Q, W_K, W_V) in heads:
        Q = X @ W_Q                                    # (T, d_k)
        K = X @ W_K
        V = X @ W_V
        scores = Q @ K.transpose(-2, -1) / sqrt(d_k)   # (T, T)
        if causal: mask the upper triangle to -inf
        attn = F.softmax(scores, dim=-1)
        head_out = attn @ V                            # (T, d_k)
      concat all head_outs along last dim → (T, h*d_k)
      return concat @ W_O                              # (T, d)

    Must use:
      - torch.triu / torch.ones / masked_fill for the causal mask
      - F.softmax(..., dim=-1)
      - Plain @ for matmuls (no nn.Linear — we want explicit weight handling)
    """
    head_outs = []
    Tn = X.shape[0]
    for W_Q, W_K, W_V in heads:
        # TODO: compute Q, K, V
        Q = ...  # TODO
        K = ...  # TODO
        V = ...  # TODO
        # TODO: compute scores with sqrt scaling
        scores = ...  # TODO
        # TODO: causal mask (if causal=True), using torch.triu + masked_fill
        # TODO: attn = F.softmax(scores, dim=-1)
        attn = ...  # TODO
        # TODO: head_out = attn @ V
        head_out = ...  # TODO
        head_outs.append(head_out)
    # TODO: concat and project (return torch.cat(head_outs, dim=-1) @ W_O)
    return ...  # TODO


torch_out = my_mha_torch(X_t, heads_t, W_O_t, causal=True).numpy()
err = np.abs(torch_out - gt).max()
print(f"drill 11: torch MHA vs numpy MHA max abs error = {err:.2e}")
assert err < 1e-5, (
    f"drill 11 failed (err={err:.2e}). Common causes:\n"
    "  - forgot 1/sqrt(d_k) scaling\n"
    "  - applied mask AFTER softmax (rows no longer sum to 1)\n"
    "  - softmax on the wrong dim (should be -1, i.e. keys)\n"
    "  - matmul order: scores = Q @ K.T, not K @ Q.T\n"
    "  - concat dim wrong: should be the last dim (head feature axis)"
)
print("✅ drill 11: torch MHA matches numpy MHA. you have bridged from-scratch → framework.")

print("\n✅ all 11 drills passed. you are PyTorch-fluent.")
