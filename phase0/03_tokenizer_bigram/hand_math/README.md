# `03_tokenizer_bigram/hand_math/` — pencil derivations

## What belongs here for Module 3

### 1. MLE → empirical count table

Prove by hand that a bigram model trained with cross-entropy loss converges to the empirical count distribution.

Setup:
- Let `θ_{a,b}` = the model's logit for token `b` given previous token `a`.
- Let `p_θ(b|a) = softmax(θ_{a,·})_b`.
- Let `C_{a,b}` = the count of bigram `(a, b)` in the corpus.
- The training loss is `L = -∑_{a,b} C_{a,b} log p_θ(b|a)`.

Derive:
- `∂L/∂θ_{a,b} = -C_{a,b} + (∑_{b'} C_{a,b'}) · p_θ(b|a)`
- Setting this to zero: `p_θ(b|a) = C_{a,b} / ∑_{b'} C_{a,b'}` — the empirical conditional distribution.

This is one of the most important results in the course because it generalises:
**a neural LM trained on cross-entropy converges to the empirical conditional distribution
given its context window.** The bigram is the trivial case; GPT is the same statement
with a much larger context.

### 2. Information-theoretic argument for "merge the most-frequent pair"

One paragraph: why BPE merges the *most frequent* adjacent pair at each step, not the rarest. The argument is roughly:
- A merged token costs 1 ID slot in the vocab.
- The slot saves `count(pair) · 1 token per occurrence` in encoded length.
- So the highest expected information-rate gain per vocab slot = highest count.

This is also why BPE tends to learn tokens of *roughly equal frequency* — once frequent pairs are merged, what was uncommon becomes more relatively common, and the next merge targets it. Equal-frequency tokens give the LM roughly equal cross-entropy budget per token, which is the property the loss function actually cares about.

### Optional 3. Why bytes, not chars

A short note on the failure mode of char-level BPE on unseen Unicode and how byte-level avoids it.

## Format

Photo or markdown. Mentor spot-checks one derivation cold.
