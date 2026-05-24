"""Phase 0 capstone — parity test: NumpyGPT vs. TorchGPT.

The claim: with identical numerical weights, both models produce the same
forward output for the same input ids, to 1e-4.

This is the Phase 0 closure proof. If this test passes, you've assembled a
correct transformer LM from scratch — every component (autograd, MLP machinery,
tokenizer, attention, block, embeddings, LM head) integrates into one model
that matches PyTorch arithmetic.

What gets copied (the parity work):
  - token_emb (V, d)           ↔ Embedding(V, d).weight
  - pos_emb   (max_T, d)       ↔ Embedding(max_T, d).weight
  - per block, per head:
      W_Q (d, d_k)             ↔ heads_q[h].weight.T   (nn.Linear stores (out, in))
      W_K (d, d_k)             ↔ heads_k[h].weight.T
      W_V (d, d_k)             ↔ heads_v[h].weight.T
  - W_O (h·d_k, d)             ↔ W_O.weight.T
  - LN1, LN2, LN_f: gamma → weight, beta → bias
  - FFN W1 (d, 4d), b1 (4d)    ↔ ffn[0].weight.T, ffn[0].bias
  - FFN W2 (4d, d), b2 (d)     ↔ ffn[2].weight.T, ffn[2].bias
  - lm_head_W (d, V)           ↔ lm_head.weight.T

Run:
    python test.py
"""

import os
import math
import numpy as np
import torch

# Local modules.
from numpy_gpt import NumpyGPT
from torch_gpt import TorchGPT


def _copy_numpy_into_torch(np_model: NumpyGPT, torch_model: TorchGPT):
    """Copy every weight from np_model into torch_model so the two have identical
    numerical parameters. After this call, forward(x) must match to floating-point
    precision."""
    with torch.no_grad():
        torch_model.token_emb.weight.copy_(torch.from_numpy(np_model.token_emb).float())
        torch_model.pos_emb.weight.copy_(torch.from_numpy(np_model.pos_emb).float())

        for np_block, t_block in zip(np_model.blocks, torch_model.blocks):
            # Per-head projections. NumPy stores W_Q : (d, d_k); torch's nn.Linear
            # weight is (out, in) = (d_k, d). Transpose on copy.
            for i, (W_Q, W_K, W_V) in enumerate(np_block.heads):
                t_block.heads_q[i].weight.copy_(torch.from_numpy(W_Q.T).float())
                t_block.heads_k[i].weight.copy_(torch.from_numpy(W_K.T).float())
                t_block.heads_v[i].weight.copy_(torch.from_numpy(W_V.T).float())

            # Output projection: numpy W_O is (h*d_k, d); torch is (d, h*d_k).
            t_block.W_O.weight.copy_(torch.from_numpy(np_block.W_O.T).float())

            # LayerNorms.
            t_block.ln1.weight.copy_(torch.from_numpy(np_block.ln1.gamma).float())
            t_block.ln1.bias.copy_(torch.from_numpy(np_block.ln1.beta).float())
            t_block.ln2.weight.copy_(torch.from_numpy(np_block.ln2.gamma).float())
            t_block.ln2.bias.copy_(torch.from_numpy(np_block.ln2.beta).float())

            # FFN. nn.Sequential[0] is Linear(d, 4d), [2] is Linear(4d, d).
            t_block.ffn[0].weight.copy_(torch.from_numpy(np_block.ffn.W1.T).float())
            t_block.ffn[0].bias.copy_(torch.from_numpy(np_block.ffn.b1).float())
            t_block.ffn[2].weight.copy_(torch.from_numpy(np_block.ffn.W2.T).float())
            t_block.ffn[2].bias.copy_(torch.from_numpy(np_block.ffn.b2).float())

        # Final LN
        torch_model.ln_f.weight.copy_(torch.from_numpy(np_model.ln_f.gamma).float())
        torch_model.ln_f.bias.copy_(torch.from_numpy(np_model.ln_f.beta).float())

        # LM head: numpy (d, V); torch (V, d).
        torch_model.lm_head.weight.copy_(torch.from_numpy(np_model.lm_head_W.T).float())


def test_param_count():
    """Both models should have the same parameter count."""
    V, d, h, n_layers, max_T = 65, 64, 4, 2, 64
    np_model = NumpyGPT(V, d, h, n_layers, max_T, seed=0)
    t_model = TorchGPT(V, d, h, n_layers, max_T)

    np_count = np_model.param_count()
    t_count = sum(p.numel() for p in t_model.parameters())
    print(f"  numpy param count: {np_count:,}")
    print(f"  torch param count: {t_count:,}")
    assert np_count == t_count, (
        f"param count mismatch: numpy={np_count}, torch={t_count}. "
        "This usually means the architectures diverged (e.g., a bias was added "
        "on one side but not the other)."
    )


def test_forward_parity():
    """Forward(ids) must match to 1e-4 with identical weights."""
    torch.manual_seed(0)
    V, d, h, n_layers, max_T = 65, 64, 4, 2, 64

    np_model = NumpyGPT(V, d, h, n_layers, max_T, seed=0)
    t_model = TorchGPT(V, d, h, n_layers, max_T)
    _copy_numpy_into_torch(np_model, t_model)

    # Same input ids.
    rng = np.random.RandomState(42)
    ids = rng.randint(0, V, size=(16,))
    np_logits = np_model.forward(ids)                       # (T, V)
    t_logits, _ = t_model(torch.from_numpy(ids).long())     # (1, T, V)
    t_logits = t_logits.squeeze(0).detach().numpy()

    err = np.abs(np_logits - t_logits).max()
    print(f"  max abs error: {err:.2e}")
    # 1e-4 is the tolerance because we're going through float32 in torch.
    assert err < 1e-4, (
        f"forward mismatch: max abs error = {err:.2e}. "
        "Likely causes:\n"
        "  - weight-transpose direction wrong somewhere in _copy_numpy_into_torch\n"
        "  - GELU approximation mismatch (numpy uses tanh-approx; torch must use approximate='tanh')\n"
        "  - mask shape / placement diverges between the two blocks\n"
        "  - LayerNorm eps differs (both should use 1e-5; check Module 5's solution)"
    )


def test_causal_property_on_full_gpt():
    """The full GPT should preserve causality: perturbing positions [k+1:] must
    not change logits at position k."""
    V, d, h, n_layers, max_T = 65, 32, 4, 2, 32

    np_model = NumpyGPT(V, d, h, n_layers, max_T, seed=0)
    rng = np.random.RandomState(7)
    ids = rng.randint(0, V, size=(16,))
    logits_orig = np_model.forward(ids)

    # Perturb the second half of the sequence (positions 8..15).
    ids_perturbed = ids.copy()
    ids_perturbed[8:] = rng.randint(0, V, size=(8,))
    logits_perturbed = np_model.forward(ids_perturbed)

    # Logits at positions 0..7 must be unchanged.
    err = np.abs(logits_orig[:8] - logits_perturbed[:8]).max()
    print(f"  drift at positions [0..7] after perturbing [8..15]: {err:.2e}")
    assert err < 1e-9, (
        f"causal leak in the full GPT (drift {err:.2e}). "
        "The block's causal mask is correctly upper-triangular per Module 4, but "
        "somewhere in the stack future information is leaking back. This is a "
        "real bug — find it before promoting Phase 0."
    )


if __name__ == "__main__":
    print("\n--- test_param_count ---")
    test_param_count()
    print("\n--- test_forward_parity ---")
    test_forward_parity()
    print("\n--- test_causal_property_on_full_gpt ---")
    test_causal_property_on_full_gpt()
    print("\n✅ Phase 0 capstone parity tests passed. The from-scratch components compose into a working LM.")
