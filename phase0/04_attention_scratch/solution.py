"""Module 4 — solution."""

import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def single_head_attention(X, W_Q, W_K, W_V, causal=True):
    T, d = X.shape
    d_k = W_K.shape[1]
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    scores = Q @ K.T / np.sqrt(d_k)            # (T, T)
    if causal:
        mask = np.triu(np.ones((T, T)), k=1).astype(bool)
        scores = np.where(mask, -1e9, scores)
    w = softmax(scores, axis=-1)
    return w @ V                                # (T, d_v)


def multi_head_attention(X, heads_params, W_O):
    head_outs = [single_head_attention(X, *p) for p in heads_params]
    concat = np.concatenate(head_outs, axis=-1)  # (T, h*d_v)
    return concat @ W_O                          # (T, d)


if __name__ == "__main__":
    np.random.seed(0)
    T, d, d_k, h = 4, 8, 2, 4
    X = np.random.randn(T, d)
    heads = [(np.random.randn(d, d_k), np.random.randn(d, d_k), np.random.randn(d, d_k))
             for _ in range(h)]
    W_O = np.random.randn(h * d_k, d)
    out = multi_head_attention(X, heads, W_O)
    print("MHA output shape:", out.shape)
