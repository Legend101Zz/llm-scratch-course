"""Module 3 — char tokenizer + bigram model."""

import os, urllib.request, random
import numpy as np

random.seed(0)
np.random.seed(0)

DATA = os.path.join(os.path.dirname(__file__), "tinyshakespeare.txt")
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

if not os.path.exists(DATA):
    print("downloading tinyshakespeare...")
    urllib.request.urlretrieve(URL, DATA)

with open(DATA) as f:
    text = f.read()

print(f"corpus chars: {len(text):,}")
print(f"first 100:\n{text[:100]}")

# 1) build vocab
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)
print(f"vocab size: {vocab_size}")

# 2) encode whole corpus
data = np.array(encode(text), dtype=np.int64)

# 3) TODO: build bigram count matrix N of shape (vocab, vocab)
#    N[i, j] = number of times token j follows token i in `data`.
#    Hint: zip(data[:-1], data[1:])
N = np.zeros((vocab_size, vocab_size), dtype=np.float32)
# fill N here

# 4) TODO: convert counts to probabilities (row-normalize). Add a small smoothing (+1) to avoid zeros.
P = None  # shape (vocab, vocab)

# 5) sample 200 chars starting from '\n'
def sample(n=200, start="\n"):
    ix = stoi[start]
    out = [ix]
    for _ in range(n):
        # TODO: sample next index from P[ix]
        # nxt = np.random.choice(vocab_size, p=P[ix])
        # out.append(nxt)
        # ix = nxt
        pass
    return decode(out)

# print(sample())
