"""Module 5 — solution."""

import numpy as np, importlib.util, os
_spec = importlib.util.spec_from_file_location(
    "attention_solution",
    os.path.join(os.path.dirname(__file__), "..", "04_attention_scratch", "solution.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
multi_head_attention = _mod.multi_head_attention


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


class LayerNorm:
    def __init__(self, d, eps=1e-5):
        self.gamma = np.ones(d); self.beta = np.zeros(d); self.eps = eps

    def __call__(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return self.gamma * (x - mu) / np.sqrt(var + self.eps) + self.beta


class FFN:
    def __init__(self, d, d_ff=None):
        d_ff = d_ff or 4 * d
        self.W1 = np.random.randn(d, d_ff) * 0.02
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d) * 0.02
        self.b2 = np.zeros(d)

    def __call__(self, x):
        return gelu(x @ self.W1 + self.b1) @ self.W2 + self.b2


class TransformerBlock:
    def __init__(self, d, h):
        d_k = d // h
        self.heads = [(np.random.randn(d, d_k) * 0.02,
                       np.random.randn(d, d_k) * 0.02,
                       np.random.randn(d, d_k) * 0.02) for _ in range(h)]
        self.W_O = np.random.randn(h * d_k, d) * 0.02
        self.ln1 = LayerNorm(d); self.ln2 = LayerNorm(d); self.ffn = FFN(d)

    def __call__(self, x):
        x = x + multi_head_attention(self.ln1(x), self.heads, self.W_O)
        x = x + self.ffn(self.ln2(x))
        return x


if __name__ == "__main__":
    np.random.seed(0)
    T, d, h = 8, 32, 4
    x = np.random.randn(T, d)
    block = TransformerBlock(d, h)
    print("output shape:", block(x).shape)
