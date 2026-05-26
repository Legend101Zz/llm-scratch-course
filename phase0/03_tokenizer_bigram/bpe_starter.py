"""Module 3 — BPE from scratch (the starter).

You will implement byte-pair encoding (BPE) — the same algorithm that powers
`tiktoken` for GPT-2/3/4. About 60 lines if you do it tightly.

THE ALGORITHM (no looking at the reference):

1. Start with a base vocabulary: each unique byte (0..255) is one token.
   Why bytes, not chars? Because then *anything* — emoji, Chinese, weird Unicode —
   round-trips through UTF-8 with no out-of-vocab failures. tiktoken does this too.

2. Encode the training text as a list of byte IDs.

3. Repeat `n_merges` times:
   a. Count every adjacent pair `(a, b)` in the current token list.
   b. Find the most-frequent pair. Call it `(a*, b*)`.
   c. Mint a new token ID (the next integer ≥ 256). Add it to the vocab:
      vocab[new_id] = vocab[a*] + vocab[b*]  (concatenate the byte strings).
   d. Record the merge:  merges[(a*, b*)] = new_id.
   e. Walk the token list and replace every occurrence of `(a*, b*)` with `new_id`.

   That's it. The vocab grows by one each step; the encoded sequence gets
   *strictly shorter* whenever there's a real repeated pair.

4. To `encode(new_text)`:
   - Start with bytes of `new_text`.
   - While ≥ 2 tokens remain, find the pair in the current sequence
     whose merge was learned EARLIEST (smallest merge index).
     Apply that merge. Repeat.
   - This greedy "earliest-learned first" rule is exactly how tiktoken decodes.

5. To `decode(token_ids)`:
   - Look up `vocab[id]` for each — that's a byte string. Concatenate.
   - Decode as UTF-8.

WHY THIS WORKS (one paragraph of intuition):
Frequent character sequences like ' the' get one token; rare sequences like
'antidisestablishmentarianism' stay as many small tokens. This gives roughly
*equal probability mass per token* across the corpus, which is the property
the language model actually needs (every token contributes a similar amount
to the cross-entropy loss).

Run `python bpe_starter.py` to test.
After all TODOs are filled, run `python test.py` to compare to the reference
solution and check the encode/decode round-trip.
"""

from collections import Counter


def get_pairs(tokens):
    """Count adjacent symbol pairs in a list of token IDs.

    Args:
        tokens: list[int] — the current token sequence.

    Returns:
        Counter mapping (id_a, id_b) -> count.

    Example:
        >>> get_pairs([1, 2, 3, 1, 2])
        Counter({(1, 2): 2, (2, 3): 1, (3, 1): 1})
    """
    # TODO: return a Counter of all adjacent pairs.
    # Hint: zip(tokens[:-1], tokens[1:]).
    raise NotImplementedError


def merge(tokens, pair, new_id):
    """Replace every occurrence of `pair` in `tokens` with `new_id`.

    Args:
        tokens: list[int]
        pair: tuple[int, int] — the pair to replace
        new_id: int — the new token id

    Returns:
        list[int] — the merged token sequence.

    Example:
        >>> merge([1, 2, 3, 1, 2, 4], (1, 2), 99)
        [99, 3, 99, 4]

    CAREFUL: don't use a list comprehension naively. After you consume positions
    i and i+1 as a merge, you must skip i+1 — otherwise you may double-count.
    """
    # TODO: walk the list once; if (tokens[i], tokens[i+1]) == pair, emit new_id
    # and advance by 2; otherwise emit tokens[i] and advance by 1.
    raise NotImplementedError


def train_bpe(text: str, n_merges: int = 200, verbose: bool = True):
    """Train BPE on `text` for `n_merges` merge steps.

    Returns:
        merges: dict[(int, int)] -> int — pair-to-new-id, in learning order
        vocab:  dict[int]        -> bytes — id-to-byte-string (for decoding)

    Walks the algorithm above. Should print a line every ~10 merges so you can
    watch the vocab grow — e.g.  `merge  10: (101, 32) -> 266 ('e ') x1834`.

    Stop early if no pair appears more than once (no useful merges left).
    """
    # ---- step 1+2: byte-level base vocab + encode text ----
    # TODO: tokens = list(text.encode("utf-8"))  --- a list of ints 0..255
    # TODO: vocab = {i: bytes([i]) for i in range(256)}
    # TODO: merges = {}
    # TODO: next_id = 256
    tokens = ...
    vocab = ...
    merges = ...
    next_id = ...

    for step in range(n_merges):
        # ---- step 3a: count pairs ----
        # TODO: pairs = get_pairs(tokens)
        pairs = ...

        if not pairs:
            break

        # ---- step 3b: find the most-frequent pair ----
        # TODO: best_pair, count = pairs.most_common(1)[0]
        best_pair, count = ..., ...

        if count < 2:
            # no pair appears more than once — no useful merges left
            break

        # ---- step 3c, d: mint new id, record vocab + merge ----
        # TODO: vocab[next_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        # TODO: merges[best_pair] = next_id
        ...

        # ---- step 3e: apply the merge ----
        # TODO: tokens = merge(tokens, best_pair, next_id)
        tokens = ...

        if verbose and step % 10 == 0:
            piece = vocab[next_id].decode("utf-8", errors="replace")
            print(f"merge {step:3d}: {best_pair} -> {next_id} ({piece!r}) x{count}")

        next_id += 1

    return merges, vocab


def encode(text: str, merges: dict):
    """Encode `text` to a list of token IDs using the learned merges.

    The rule: at each step, find the pair (a, b) in the current sequence
    whose merges[(a, b)] is the smallest (= learned earliest). Apply that merge.
    Stop when no pair in the current sequence has a known merge.

    Why earliest-learned-first? Because earliest merges are the most frequent
    base patterns — applying them first guarantees the same trajectory as
    training. Try a different order and you'd encode the same string differently
    every time.
    """
    # TODO: tokens = list(text.encode("utf-8"))
    tokens = ...

    while len(tokens) >= 2:
        # TODO: pairs = get_pairs(tokens)
        # TODO: candidate = min(pairs, key=lambda p: merges.get(p, float("inf")))
        # TODO: if candidate not in merges: break
        # TODO: tokens = merge(tokens, candidate, merges[candidate])
        ...

    return tokens


def decode(ids, vocab):
    """Decode a list of token IDs back to a string.

    Concatenate the byte strings, then UTF-8 decode (with errors="replace"
    for partial UTF-8 sequences mid-token — rare in practice).
    """
    # TODO: return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")
    raise NotImplementedError


if __name__ == "__main__":
    import os, urllib.request
    PATH = os.path.join(os.path.dirname(__file__), "tinyshakespeare.txt")
    if not os.path.exists(PATH):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            PATH,
        )
    text = open(PATH).read()[:50_000]   # train on a slice for speed

    print(f"training BPE on {len(text):,} chars ({len(text.encode('utf-8')):,} bytes)...\n")
    merges, vocab = train_bpe(text, n_merges=200)

    print(f"\n✅ final vocab size: {len(vocab)} (base 256 + {len(merges)} merges)")
    print(f"   compression: {len(text.encode('utf-8'))} bytes -> "
          f"{len(encode(text, merges))} tokens "
          f"({len(text.encode('utf-8')) / max(len(encode(text, merges)), 1):.2f}× shorter)")

    sample = "ROMEO: Hello there, my friend!"
    ids = encode(sample, merges)
    print(f"\nsample: {sample!r}")
    print(f"encoded ({len(ids)} tokens): {ids}")
    print(f"decoded: {decode(ids, vocab)!r}")
    assert decode(ids, vocab) == sample, "round-trip failed!"
    print("\n✅ round-trip OK")
