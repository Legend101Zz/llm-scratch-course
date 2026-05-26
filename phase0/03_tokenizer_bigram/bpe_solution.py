"""Module 3 — BPE from scratch (reference solution).

Don't open this until your `bpe_starter.py` passes `test.py`.
This is the same algorithm as the original `bpe.py`, kept in the same shape
as `bpe_starter.py` so a side-by-side comparison shows the diff cleanly.
"""

from collections import Counter


def get_pairs(tokens):
    """Adjacent symbol pairs in a list of token IDs."""
    return Counter(zip(tokens[:-1], tokens[1:]))


def merge(tokens, pair, new_id):
    """Replace every occurrence of `pair` in `tokens` with `new_id`."""
    out = []
    i = 0
    n = len(tokens)
    while i < n:
        if i < n - 1 and (tokens[i], tokens[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def train_bpe(text: str, n_merges: int = 200, verbose: bool = True):
    """Train BPE; returns (merges_dict, vocab_dict)."""
    tokens = list(text.encode("utf-8"))
    vocab = {i: bytes([i]) for i in range(256)}
    merges: dict = {}
    next_id = 256

    for step in range(n_merges):
        pairs = get_pairs(tokens)
        if not pairs:
            break
        best_pair, count = pairs.most_common(1)[0]
        if count < 2:
            break
        vocab[next_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges[best_pair] = next_id
        tokens = merge(tokens, best_pair, next_id)
        if verbose and step % 10 == 0:
            piece = vocab[next_id].decode("utf-8", errors="replace")
            print(f"merge {step:3d}: {best_pair} -> {next_id} ({piece!r}) x{count}")
        next_id += 1

    return merges, vocab


def encode(text: str, merges: dict):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        pairs = get_pairs(tokens)
        candidate = min(pairs, key=lambda p: merges.get(p, float("inf")))
        if candidate not in merges:
            break
        tokens = merge(tokens, candidate, merges[candidate])
    return tokens


def decode(ids, vocab):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


if __name__ == "__main__":
    import os, urllib.request
    PATH = os.path.join(os.path.dirname(__file__), "tinyshakespeare.txt")
    if not os.path.exists(PATH):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            PATH,
        )
    text = open(PATH).read()[:50_000]
    merges, vocab = train_bpe(text, n_merges=200)
    print(f"\n✅ final vocab size: {len(vocab)}")
    sample = "ROMEO: Hello there, my friend!"
    ids = encode(sample, merges)
    print(f"encoded: {ids}")
    print(f"decoded: {decode(ids, vocab)}")
    assert decode(ids, vocab) == sample
