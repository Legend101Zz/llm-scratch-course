"""Module 3 — solution."""

import os, urllib.request
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "tinyshakespeare.txt")
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists(DATA):
    urllib.request.urlretrieve(URL, DATA)

text = open(DATA).read()
chars = sorted(set(text))
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
data = np.array([stoi[c] for c in text], dtype=np.int64)

N = np.ones((V, V), dtype=np.float32)  # +1 smoothing
for a, b in zip(data[:-1], data[1:]):
    N[a, b] += 1
P = N / N.sum(axis=1, keepdims=True)

# negative log-likelihood on training data
nll = -np.log(P[data[:-1], data[1:]] + 1e-12).mean()
print(f"bigram NLL: {nll:.4f}  (random would be {np.log(V):.4f})")

ix = stoi["\n"]
out = [ix]
for _ in range(300):
    ix = np.random.choice(V, p=P[ix])
    out.append(ix)
print("".join(itos[i] for i in out))
