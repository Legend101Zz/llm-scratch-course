# Module 3 — Tokenization + the Bigram Baseline (25 min)

> ⏱️ 25 minutes. Quick but important. Most LLM bugs start at the tokenizer.

## Why tokenize?

Neural nets eat numbers, not strings. We need a deterministic mapping `string ↔ list of int IDs`. Three common schemes:

| Scheme            | Vocab size | Pros                              | Cons                             |
|-------------------|------------|-----------------------------------|----------------------------------|
| Char-level        | ~100       | Simple, no OOV                    | Long sequences, weak semantics   |
| Word-level        | ~50k+      | Semantic units                    | Huge vocab, OOV explosion        |
| **BPE / subword** | 32k–100k   | Best of both, what GPT-2 uses     | Slightly more complex            |

We'll start char-level (5 lines of code), then I'll explain BPE intuitively. Production = BPE (`tiktoken` for GPT, `sentencepiece` for many others).

### BPE in one paragraph

Start with all unique bytes/chars as your vocab. Find the most frequent **adjacent pair** in the corpus. Merge it into a new token. Repeat $N$ times. Common pairs like `(t, h)` → `th`, then `(th, e)` → `the`, become single tokens. So frequent words are 1 token, rare words are several. That's why GPT-2 has ~50k tokens and "antidisestablishmentarianism" is split into many but "the" is just one.

## The Bigram Model (the dumbest LM that almost works)

Predict the next token using only the previous one. So we learn a table $P(x_t \mid x_{t-1})$ — just a `(vocab, vocab)` matrix of logits, softmaxed to probabilities.

This is a transformer with **0 attention heads, 0 layers, 1 embedding lookup** — and it is the foundation Karpathy uses to introduce GPT in his videos. After this module, every later trick (attention, multi-head, depth) is "make this baseline less dumb".

Loss is still cross-entropy:

$$\mathcal{L} = -\frac{1}{N}\sum \log p(x_{t+1} \mid x_t)$$

For a bigram model trained to convergence, **the optimal table is just the empirical bigram counts**, normalized. So you can also just *count* and compare to the trained model. They should match. (Why? Because for cross-entropy on counts, the MLE is the empirical distribution.)

### Build BPE yourself (required — not optional)

> Originally this module had BPE as a 5-min "optional, here's the code" aside.
> Under the new framing (see [`../CONVENTIONS.md`](../CONVENTIONS.md)), Phase 0 modules implement everything from scratch with a parity test. So BPE is now a proper exercise.

You'll fill in **[`bpe_starter.py`](bpe_starter.py)** — `train_bpe`, `encode`, `decode`, plus the `get_pairs` and `merge` helpers — ~60 lines total. The skeleton has detailed comments walking you through the algorithm; the docstring at the top lists the 5 algorithmic steps you need to implement.

When you can run:
```bash
python test.py
```
and see all four tests green (`test_roundtrip`, `test_parity_with_solution`, `test_compression`, `test_structural_match_with_tiktoken`), BPE is done.

Then sanity-check against GPT-2's tokenizer:
```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
print(enc.encode("Hello world!"))   # production version, 50,257 vocab
```
Your BPE won't produce identical IDs (tiktoken was trained on much more text), but the algorithm is the same — `test.py`'s `test_structural_match_with_tiktoken` checks the algorithmic equivalence (round-trip works, frequent patterns compress better than gibberish, etc.).

Reference solution: **[`bpe_solution.py`](bpe_solution.py)** — open only after the test passes. It's structured identically to `bpe_starter.py` so a side-by-side diff highlights exactly what was missing.

### Why we use *bytes*, not chars

Look closely at `bpe_starter.py`'s step 1: the base vocab is 256 byte values, not the `set(text)` of unique chars. Why?

Because then **everything UTF-8 round-trips with zero out-of-vocab failures** — emoji, Chinese, weird symbols, all of them. They start as multi-byte sequences and the BPE merges treat byte n-grams uniformly. tiktoken does this; so should you. (A char-level base vocab fails on any string with a character not seen in training.)

The merge dynamics learn the high-frequency *byte* bigrams that happen to correspond to letters first — so the first 30 merges learn things like `(116, 104) -> 'th'`, exactly like a char-level BPE would, but with no UTF-8 fragility.

### Reflection

- Why is char-level tokenization "weaker semantically" than BPE?
- For BPE, why do we merge the most *frequent* pair? Argue this from an information-theoretic angle — what's the loss function this is implicitly optimizing? *(Hint: the LM ultimately spends the same training-loss budget per token; you want each token to carry roughly equal information.)*
- A trained bigram NN converges to the count table. What does this say about cross-entropy as a loss function? *(Hint: think about what distribution minimizes the cross-entropy between the empirical distribution and `p_\theta`.)*
- Why does BPE's `encode()` apply the **earliest-learned** merge first, not the most-frequent one? What goes wrong if you applied merges in count-order? *(Hint: determinism. The same string should encode the same way every time.)*
- BPE has a known failure mode where a long word can tokenize into way more tokens than its character length would suggest. Construct an example string that breaks your BPE's compression. What's a fix in modern tokenizers like GPT-4's `cl100k_base`?

## Build it (the bigram model)

Open [`starter.py`](starter.py). You'll:
1. Load `tinyshakespeare.txt` (downloaded automatically).
2. Build char vocab + encode/decode.
3. Build a bigram count table → probabilities → sample from it.
4. (Optional, +10 min) Train the same bigram as a neural net (1 layer, no hidden units) using PyTorch — to see the two approaches converge to the same answer.

After `starter.py` works, do the BPE exercise above (`bpe_starter.py` + `test.py`).

## How this module is "done"

Per [`../CONVENTIONS.md`](../CONVENTIONS.md):
1. `starter.py` runs and prints a bigram-sampled sequence.
2. `bpe_starter.py` is filled in such that `python test.py` shows all four tests green.
3. `hand_math/` contains your derivation of why the bigram NN with cross-entropy loss converges to the empirical count table. *(One paragraph + the partial derivative is enough; the result is that the gradient of `-log p_θ(b|a)` w.r.t. `θ_{a,b}` is zero exactly when `p_θ` equals the empirical distribution.)*
4. `evidence/test_output.txt` contains the output of `python test.py` showing all green.
5. `notes.md` exists with at least one "thing I got stuck on."

✅ Next: [Module 4 — Self-Attention](../04_attention_scratch/README.md)
