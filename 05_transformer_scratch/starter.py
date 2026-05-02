"""Module 5 — full transformer block (forward only, numpy)."""

import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "04_attention_scratch"))
from solution import multi_head_attention, softmax

np.random.seed(0)


def gelu(x):
    # TODO: approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
    pass


class LayerNorm:
    def __init__(self, d, eps=1e-5):
        self.gamma = np.ones(d)
        self.beta = np.zeros(d)
        self.eps = eps

    def __call__(self, x):
        # x: (T, d)
        # TODO: mean and var along last axis (keepdims=True), normalize, then scale+shift
        pass


class FFN:
    def __init__(self, d, d_ff=None):
        d_ff = d_ff or 4 * d
        self.W1 = np.random.randn(d, d_ff) * 0.02
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d) * 0.02
        self.b2 = np.zeros(d)

    def __call__(self, x):
        # TODO: gelu(x W1 + b1) W2 + b2
        pass


class TransformerBlock:
    def __init__(self, d, h):
        d_k = d // h
        self.heads = [(np.random.randn(d, d_k) * 0.02,
                       np.random.randn(d, d_k) * 0.02,
                       np.random.randn(d, d_k) * 0.02)
                      for _ in range(h)]
        self.W_O = np.random.randn(h * d_k, d) * 0.02
        self.ln1 = LayerNorm(d)
        self.ln2 = LayerNorm(d)
        self.ffn = FFN(d)

    def __call__(self, x):
        # TODO (pre-norm):
        # x = x + multi_head_attention(self.ln1(x), self.heads, self.W_O)
        # x = x + self.ffn(self.ln2(x))
        # return x
        pass


if __name__ == "__main__":
    T, d, h = 8, 32, 4
    x = np.random.randn(T, d)
    block = TransformerBlock(d, h)
    y = block(x)
    # print(y.shape)  # expect (T, d) = (8, 32)
