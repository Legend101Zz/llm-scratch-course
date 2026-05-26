"""Module 4 — self-attention in numpy."""

import numpy as np
np.random.seed(0)


def softmax(x, axis=-1):
    # TODO: numerically stable softmax. Subtract max along axis before exp.
    x = x - x.max(axis=axis, keepdims =True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims =True)


def single_head_attention(X, W_Q, W_K, W_V, causal=True):
    """
    X:    (T, d)
    W_Q:  (d, d_k)
    W_K:  (d, d_k)
    W_V:  (d, d_v)
    returns: (T, d_v)
    """
    T, d = X.shape
    d_k = W_K.shape[1]

    # TODO: Q = X @ W_Q                 # (T, d_k)
    # TODO: K = X @ W_K                 # (T, d_k)
    # TODO: V = X @ W_V                 # (T, d_v)
    # TODO: scores = Q @ K.T / sqrt(d_k)  # (T, T)
    # TODO: if causal: set upper triangle (i < j part where j > i) to -inf
    #       hint: mask = np.triu(np.ones((T, T)), k=1).astype(bool); scores[mask] = -np.inf
    # TODO: weights = softmax(scores, axis=-1)
    # TODO: return weights @ V          # (T, d_v)
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    scores = Q @ K.T / np.sqrt(d_k)
    if causal:
        mask = np.triu(np.ones((T, T)), k=1).astype(bool)
        scores = np.where(mask, -1e9, scores)
    w = softmax(scores, axis=-1)
    return w @ V 

def multi_head_attention(X, heads_params, W_O):
    """
    heads_params: list of (W_Q, W_K, W_V) tuples, one per head.
    W_O: (h*d_v, d)  output projection.
    returns: (T, d)
    """
    # TODO: run each head, concat along last dim, project with W_O.
    head_outs = [single_head_attention(X, *p) for p in heads_params]
    concat = np.concatenate(head_outs, axis=-1)
    return concat @ W_O


if __name__ == "__main__":
    T, d, d_k, h = 4, 8, 2, 4
    X = np.random.randn(T, d)
    heads = [(np.random.randn(d, d_k), np.random.randn(d, d_k), np.random.randn(d, d_k))
             for _ in range(h)]
    W_O = np.random.randn(h * d_k, d)

    out = multi_head_attention(X, heads, W_O)
    print(out.shape)  # expect (T, d) = (4, 8)
