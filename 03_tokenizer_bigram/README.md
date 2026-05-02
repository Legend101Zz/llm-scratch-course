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

### Build a tiny BPE yourself (5 min, optional but recommended)

Open [`bpe.py`](bpe.py) — ~50 lines. Two pieces:
1. `train_bpe(text, n_merges)` — repeatedly find the most common adjacent pair, merge it, log the new token.
2. `encode(text)` / `decode(ids)` — apply learned merges greedily; reverse via the vocab dict.

Run `python bpe.py`. Watch tokens like `('e', ' ')`, `(' ', 't')`, `('t', 'h')` get merged early — exactly the high-frequency English bigrams. After 200 merges you'll see whole words pop up as single tokens.

Compare to GPT-2's `tiktoken`:
```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
print(enc.encode("Hello world!"))   # production version, 50,257 vocab
```

## Build it

Open [`starter.py`](starter.py). You'll:
1. Load `tinyshakespeare.txt` (downloaded automatically).
2. Build char vocab + encode/decode.
3. Build a bigram count table → probabilities → sample from it.
4. (Optional, +10 min) Train the same bigram as a neural net (1 layer, no hidden units) using PyTorch — to see the two approaches converge to the same answer.

## Reflection

- Why is char-level tokenization "weaker semantically"?
- For BPE, why do we merge the most *frequent* pair? What if we merged the rarest?
- A trained bigram NN converges to the count table. What does this say about cross-entropy?

✅ Next: [Module 4 — Self-Attention](../04_attention_scratch/README.md)
