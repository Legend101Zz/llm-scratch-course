"""Phase 0 capstone — tiny GPT, forward-only, in pure NumPy.

This is the *assembly* step: take the LayerNorm, FFN, and TransformerBlock you
built in Module 5 (which already use the multi-head attention from Module 4) and
add the three pieces that turn a stack of blocks into a language model:

  1. Token embedding         (vocab_size → d, table lookup)
  2. Positional embedding    (max_T → d, table lookup, added to token emb)
  3. Stack of TransformerBlocks   (the Module 5 components)
  4. Final LayerNorm         (before the logit projection — the standard recipe)
  5. LM head                 (d → vocab_size, linear projection to logits)

Why forward-only in NumPy? Because Module 1's Value engine is *scalar* autograd —
running backprop through a tiny GPT with scalar ops would take many minutes per
forward pass. The Value engine taught you *how backprop works*. For training a
real (tiny) model, we use PyTorch — which we've now proved (via Modules 1–5's
parity tests) is doing the same math, just with vectorized autograd.

This file lets you assemble + run forward end-to-end in NumPy. `torch_gpt.py`
provides the matching trainable version. `test.py` proves they agree to 1e-4
forward with identical weights.

Run:
    python numpy_gpt.py

You'll see a randomly-initialized GPT predict 65 token-logits per position for a
short tinyshakespeare snippet. The logits are noise (random init) — the point of
this file is to *prove the assembly works*. Training happens in `train.py`
(PyTorch) and parity vs the trained model is in `test.py`.
"""

import os
import importlib.util
import numpy as np


# ----- pull in the Module 5 components (which transitively pull Module 4) -----
HERE = os.path.dirname(__file__)
_spec5 = importlib.util.spec_from_file_location(
    "transformer_solution",
    os.path.join(HERE, "..", "05_transformer_scratch", "solution.py"),
)
_m5 = importlib.util.module_from_spec(_spec5)
_spec5.loader.exec_module(_m5)
LayerNorm = _m5.LayerNorm
TransformerBlock = _m5.TransformerBlock


class NumpyGPT:
    """A tiny pre-norm GPT, forward-only, in NumPy.

    Architecture (matches the standard nanoGPT / GPT-2 small variant):
        ids -> emb (tok + pos) -> N blocks -> LN_f -> linear lm_head -> logits

    No dropout (numpy doesn't differentiate inference vs training cleanly here),
    no bias in the lm_head (matches GPT-2; saves vocab_size params), no weight
    tying (we could tie lm_head.weight to token_emb.T but keeping them separate
    makes the parity proof cleaner — torch_gpt mirrors this choice).
    """

    def __init__(self, vocab_size: int, d: int, h: int, n_layers: int, max_T: int, seed: int = 0):
        rng = np.random.RandomState(seed)
        std = 0.02   # GPT-2 init

        # Embeddings
        self.token_emb = rng.randn(vocab_size, d) * std        # (V, d)
        self.pos_emb = rng.randn(max_T, d) * std               # (max_T, d)

        # Reset the *global* numpy RNG before building blocks so they consume from
        # the seed deterministically (Module 5's TransformerBlock uses np.random
        # directly for its init).
        np.random.seed(seed + 1)
        self.blocks = [TransformerBlock(d, h) for _ in range(n_layers)]

        # Final LayerNorm + LM head
        self.ln_f = LayerNorm(d)
        np.random.seed(seed + 2)
        self.lm_head_W = np.random.randn(d, vocab_size) * std  # (d, V), no bias

        # Bookkeeping
        self.vocab_size = vocab_size
        self.d = d
        self.h = h
        self.n_layers = n_layers
        self.max_T = max_T

    def forward(self, ids):
        """ids: 1-D array of int token IDs of length T (T <= max_T).
        Returns logits of shape (T, vocab_size)."""
        ids = np.asarray(ids, dtype=np.int64)
        T = len(ids)
        assert T <= self.max_T, f"sequence length {T} exceeds max_T={self.max_T}"

        # Embed
        x = self.token_emb[ids] + self.pos_emb[:T]   # (T, d)

        # Stack of blocks
        for block in self.blocks:
            x = block(x)

        # Final LN + project to vocab
        x = self.ln_f(x)
        logits = x @ self.lm_head_W                  # (T, V)
        return logits

    def param_count(self):
        n = self.token_emb.size + self.pos_emb.size
        for block in self.blocks:
            n += sum(getattr(block, attr).size for attr in ("W_O",)) if hasattr(block, "W_O") else 0
            n += sum(w.size for head in block.heads for w in head)
            n += block.ln1.gamma.size + block.ln1.beta.size
            n += block.ln2.gamma.size + block.ln2.beta.size
            n += block.ffn.W1.size + block.ffn.b1.size + block.ffn.W2.size + block.ffn.b2.size
        n += self.ln_f.gamma.size + self.ln_f.beta.size
        n += self.lm_head_W.size
        return n


def _load_tinyshakes():
    import urllib.request
    path = os.path.join(HERE, "..", "03_tokenizer_bigram", "tinyshakespeare.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            path,
        )
    return open(path).read()


if __name__ == "__main__":
    text = _load_tinyshakes()
    chars = sorted(set(text))
    V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    print(f"vocab size: {V}")

    # Tiny config — enough to demonstrate the architecture, small enough to run
    # in a few seconds.
    d, h, n_layers, max_T = 64, 4, 2, 64

    model = NumpyGPT(vocab_size=V, d=d, h=h, n_layers=n_layers, max_T=max_T, seed=0)
    print(f"model params: {model.param_count():,}")

    # Run forward on a short snippet
    snippet = "ROMEO: "
    ids = [stoi[c] for c in snippet]
    logits = model.forward(ids)
    print(f"input: {snippet!r}  -> logits shape {logits.shape}")

    # Argmax-decode the next token at each position (untrained → garbage, but shows the pipe works)
    pred_ids = logits.argmax(axis=-1)
    pred_chars = "".join(itos[i] for i in pred_ids)
    print(f"argmax next-token (untrained, expect garbage): {pred_chars!r}")

    # Cross-entropy on a target = the snippet shifted by 1 (just to demonstrate the loss formula).
    if len(snippet) > 1:
        targets = ids[1:] + [ids[-1]]
        # softmax over vocab
        m = logits.max(axis=-1, keepdims=True)
        e = np.exp(logits - m)
        p = e / e.sum(axis=-1, keepdims=True)
        nll = -np.log(p[np.arange(len(ids)), targets] + 1e-12).mean()
        print(f"random-init NLL on this snippet: {nll:.3f}  (uniform baseline: {np.log(V):.3f})")
