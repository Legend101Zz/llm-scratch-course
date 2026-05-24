"""Module 6 — tensor drills, reference solution. Open ONLY after struggle."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# 1. zeros, set [2, 3] = 7
x = torch.zeros(4, 5)
x[2, 3] = 7
assert x.shape == (4, 5) and x[2, 3].item() == 7 and x.sum().item() == 7


# 2. mean over T
B, T, d = 2, 4, 8
x = torch.randn(B, T, d)
mean_x = x.mean(dim=1)
assert mean_x.shape == (B, d)


# 3. scaled dot-product scores
B, T, d = 2, 5, 16
q = torch.randn(B, T, d)
k = torch.randn(B, T, d)
scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
assert scores.shape == (B, T, T)


# 4. causal mask
mask = torch.tril(torch.ones(T, T))
masked = scores.masked_fill(mask == 0, float("-inf"))


# 5. softmax over keys
attn = F.softmax(masked, dim=-1)
assert torch.allclose(attn.sum(dim=-1), torch.ones(B, T), atol=1e-5)


# 6. attention output
v = torch.randn(B, T, d)
out = attn @ v
assert out.shape == (B, T, d)


# 7. split heads
B, T, n_head, d_head = 2, 5, 4, 8
x = torch.randn(B, T, n_head * d_head)
x_split = x.view(B, T, n_head, d_head).transpose(1, 2)   # (B, n_head, T, d_head)
assert x_split.shape == (B, n_head, T, d_head)


# 8. autograd: d/dx (x**3).sum() = 3 x**2
x = torch.randn(5, requires_grad=True)
y = (x ** 3).sum()
y.backward()
assert torch.allclose(x.grad, 3 * x.detach() ** 2, atol=1e-5)


# 9. nn.Linear forward + backward
lin = nn.Linear(8, 8)
assert sum(p.numel() for p in lin.parameters()) == 72
inp = torch.randn(3, 8)
lin.zero_grad()
out = lin(inp)
loss = out.sum()
loss.backward()
assert lin.weight.grad.shape == lin.weight.shape


# 10. dropout train vs eval
class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(0.5)
    def forward(self, x):
        return self.drop(x)

m = Tiny()
x = torch.ones(1000)
m.train();  y_train = m(x); assert 0.3 < (y_train == 0).float().mean().item() < 0.7
m.eval();   y_eval  = m(x); assert torch.allclose(y_eval, x)

print("✅ drills 1–10 passed.")


# 11. bridge drill: torch MHA matches numpy MHA (Module 4 solution = ground truth)
import os, numpy as np, importlib.util, math
_spec = importlib.util.spec_from_file_location(
    "m4_solution",
    os.path.join(os.path.dirname(__file__), "..", "04_attention_scratch", "solution.py"),
)
_m4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m4)

np.random.seed(0)
T, d, h = 8, 32, 4
d_k = d // h
X_np = np.random.randn(T, d).astype(np.float64)
heads_np = [
    (np.random.randn(d, d_k), np.random.randn(d, d_k), np.random.randn(d, d_k))
    for _ in range(h)
]
W_O_np = np.random.randn(h * d_k, d)
gt = _m4.multi_head_attention(X_np, heads_np, W_O_np)

X_t = torch.from_numpy(X_np).double()
heads_t = [tuple(torch.from_numpy(w).double() for w in head) for head in heads_np]
W_O_t = torch.from_numpy(W_O_np).double()


def my_mha_torch(X, heads, W_O, causal=True):
    Tn = X.shape[0]
    head_outs = []
    for W_Q, W_K, W_V in heads:
        Q = X @ W_Q
        K = X @ W_K
        V = X @ W_V
        scores = Q @ K.transpose(-2, -1) / math.sqrt(W_K.shape[-1])
        if causal:
            mask = torch.triu(
                torch.ones(Tn, Tn, dtype=torch.bool, device=X.device), diagonal=1
            )
            scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        head_outs.append(attn @ V)
    return torch.cat(head_outs, dim=-1) @ W_O


torch_out = my_mha_torch(X_t, heads_t, W_O_t, causal=True).numpy()
err = np.abs(torch_out - gt).max()
print(f"drill 11: torch MHA vs numpy MHA max abs error = {err:.2e}")
assert err < 1e-5
print("✅ all 11 drills passed.")
