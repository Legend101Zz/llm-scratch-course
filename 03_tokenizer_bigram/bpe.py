"""Tiny BPE in ~50 lines.

Run: python bpe.py
You'll watch the vocab grow merge by merge. After training, encode/decode any string.

This is exactly what `tiktoken` does — just slower and less polished.
"""

from collections import Counter


def get_pairs(tokens):
    """Adjacent symbol pairs in a list of token IDs."""
    return Counter(zip(tokens[:-1], tokens[1:]))


def merge(tokens, pair, new_id):
    """Replace every occurrence of `pair` in `tokens` with `new_id`."""
    out = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def train_bpe(text: str, n_merges: int = 100):
    """Returns (merges_dict, vocab_dict).

    merges: maps (id, id) -> new_id  (in order learned)
    vocab:  maps id -> bytes (so we can decode)
    """
    # start: byte-level. 256 base tokens, one per byte value.
    tokens = list(text.encode("utf-8"))
    vocab = {i: bytes([i]) for i in range(256)}
    merges = {}

    next_id = 256
    for step in range(n_merges):
        pairs = get_pairs(tokens)
        if not pairs:
            break
        best_pair, count = pairs.most_common(1)[0]
        if count < 2:
            break
        tokens = merge(tokens, best_pair, next_id)
        merges[best_pair] = next_id
        vocab[next_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        if step % 10 == 0:
            piece = vocab[next_id].decode("utf-8", errors="replace")
            print(f"merge {step:3d}: {best_pair} -> {next_id} ({piece!r}) x{count}")
        next_id += 1

    return merges, vocab


def encode(text: str, merges: dict):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        pairs = get_pairs(tokens)
        # find the pair with the smallest merge index (= learned earliest)
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
            PATH)
    text = open(PATH).read()[:50_000]      # train on a slice for speed

    merges, vocab = train_bpe(text, n_merges=200)
    print(f"\nfinal vocab size: {len(vocab)}")

    sample = "ROMEO: Hello there, my friend!"
    ids = encode(sample, merges)
    print(f"\nencoded: {ids}")
    print(f"len before: {len(sample.encode('utf-8'))}, after BPE: {len(ids)}")
    print(f"decoded:  {decode(ids, vocab)}")
    print(f"\nfirst 20 learned merges (each is now a single token):")
    for i, (pair, nid) in enumerate(list(merges.items())[:20]):
        piece = vocab[nid].decode("utf-8", errors="replace")
        print(f"  {nid}: {piece!r}")
