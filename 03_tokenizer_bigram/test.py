"""Module 3 — parity & sanity tests.

What we verify:
  1. Your BPE round-trips: decode(encode(s, merges), vocab) == s for any UTF-8 string.
  2. Your BPE matches the reference solution byte-for-byte (same merges, same encoding).
  3. (Structural) Your BPE behaves like `tiktoken`'s gpt2 encoder on a key property:
     ASCII-only input encoded by tiktoken always decodes back to itself, and your
     trained BPE has the same property. (We can't expect *identical* IDs because
     tiktoken was trained on a much larger corpus, but the algorithm is the same.)
  4. Compression on the training corpus: 200 merges should compress tinyshakespeare
     by at least 1.3× (bytes → tokens).

Run from this directory:
    python test.py

Or, to test the reference solution itself (sanity):
    USE_SOLUTION=1 python test.py
"""

import os
import sys

# Toggle to test reference vs starter.
USE_SOLUTION = os.environ.get("USE_SOLUTION") == "1"

if USE_SOLUTION:
    print("(running tests against bpe_solution.py)")
    from bpe_solution import train_bpe, encode, decode
else:
    print("(running tests against bpe_starter.py — set USE_SOLUTION=1 to test the reference)")
    try:
        from bpe_starter import train_bpe, encode, decode
    except Exception as e:
        print(f"❌ can't import from bpe_starter.py: {e}")
        sys.exit(1)


def _get_text():
    import urllib.request
    path = os.path.join(os.path.dirname(__file__), "tinyshakespeare.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            path,
        )
    return open(path).read()


def test_roundtrip():
    """encode(decode(...)) should be identity for any UTF-8 input."""
    text = _get_text()[:20_000]
    merges, vocab = train_bpe(text, n_merges=100, verbose=False)

    cases = [
        "Hello, World!",
        "ROMEO: But soft! What light through yonder window breaks?",
        "tab\tand newline\nand quotes \"and\" 'and'.",
        "",  # empty
        "a",  # single char (no merges to apply)
        "What's in a name? That which we call a rose / By any other name would smell as sweet.",
    ]
    for s in cases:
        out = decode(encode(s, merges), vocab)
        assert out == s, f"round-trip failed: {s!r} -> {out!r}"
        print(f"  ✓ round-trip: {s[:50]!r}{'...' if len(s) > 50 else ''}")


def test_parity_with_solution():
    """Your BPE should produce the same merges and the same encoding as the reference."""
    if USE_SOLUTION:
        print("  (skipped: testing solution against itself)")
        return

    from bpe_solution import train_bpe as ref_train, encode as ref_encode
    text = _get_text()[:20_000]
    yours_merges, _yours_vocab = train_bpe(text, n_merges=50, verbose=False)
    ref_merges, _ref_vocab = ref_train(text, n_merges=50, verbose=False)

    assert list(yours_merges.items()) == list(ref_merges.items()), (
        "merges differ from reference — same input, same algorithm should give "
        "identical merges (modulo dict iteration order, which Python 3.7+ preserves "
        "insertion order so this comparison is meaningful)"
    )
    print(f"  ✓ all {len(yours_merges)} merges match the reference")

    sample = "ROMEO: Hello, my friend!"
    assert encode(sample, yours_merges) == ref_encode(sample, ref_merges)
    print(f"  ✓ encode() matches the reference on a sample string")


def test_compression():
    """After 200 merges on tinyshakespeare-50k, compression ratio ≥ 1.3×."""
    text = _get_text()[:50_000]
    merges, _vocab = train_bpe(text, n_merges=200, verbose=False)

    bytes_len = len(text.encode("utf-8"))
    tokens_len = len(encode(text, merges))
    ratio = bytes_len / tokens_len

    print(f"  bytes: {bytes_len:,}  tokens: {tokens_len:,}  ratio: {ratio:.3f}×")
    assert ratio >= 1.3, (
        f"compression ratio {ratio:.3f}× < 1.3×. "
        "Either the merges aren't being applied during encode(), or the train loop "
        "isn't actually merging the most-frequent pair."
    )


def test_structural_match_with_tiktoken():
    """Sanity: tiktoken's gpt2 encoder round-trips ASCII; so should ours.

    We do NOT expect identical token IDs — tiktoken was trained on much more text.
    But the algorithm is the same, so both should:
      - round-trip ASCII identically,
      - compress 'the the the the' to fewer tokens than 'xyzqkjwpv',
        because ' the' is a high-frequency English pattern.
    """
    try:
        import tiktoken
    except ImportError:
        print("  (skipped: tiktoken not installed — `pip install tiktoken`)")
        return

    enc = tiktoken.get_encoding("gpt2")
    text = _get_text()[:20_000]
    merges, vocab = train_bpe(text, n_merges=200, verbose=False)

    # ASCII round-trip on both
    s = "Hello, world! ROMEO: How art thou?"
    assert enc.decode(enc.encode(s)) == s
    assert decode(encode(s, merges), vocab) == s
    print(f"  ✓ both BPEs round-trip {s!r}")

    # Both should compress high-frequency English better than gibberish.
    common = "the the the the the the the the"
    gibberish = "xyz qjk pwv zzz nyu"
    yours_common = len(encode(common, merges))
    yours_gib    = len(encode(gibberish, merges))
    tik_common = len(enc.encode(common))
    tik_gib    = len(enc.encode(gibberish))
    print(f"  ours      : 'the…' = {yours_common} tok | 'xyz…' = {yours_gib} tok")
    print(f"  tiktoken  : 'the…' = {tik_common} tok | 'xyz…' = {tik_gib} tok")
    assert yours_common < yours_gib, "your BPE should compress repeated 'the' better than gibberish"
    assert tik_common < tik_gib,     "(this is a tiktoken sanity check — should always hold)"


if __name__ == "__main__":
    print("\n--- test_roundtrip ---")
    test_roundtrip()
    print("\n--- test_parity_with_solution ---")
    test_parity_with_solution()
    print("\n--- test_compression ---")
    test_compression()
    print("\n--- test_structural_match_with_tiktoken ---")
    test_structural_match_with_tiktoken()
    print("\n✅ all BPE tests passed.")
