"""Phase 0 capstone — the same tiny GPT, but in PyTorch (trainable).

Why PyTorch for training? Module 1's Value engine taught us *how* backprop works
on scalars; vectorized autograd is the same math at the array level. Modules
1, 2, 4, 5 each proved (via their `test.py`) that PyTorch's ops match ours
to 1e-5 — so we now *trust* PyTorch and use it for the gradient pass during
training.

The architecture deliberately mirrors `numpy_gpt.NumpyGPT` so that, given
identical numerical weights, the two models produce identical forward outputs.
That's the parity claim verified in `test.py`.

Run on its own (no training) for a sanity check:
    python torch_gpt.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchTransformerBlock(nn.Module):
    """Pre-norm transformer block matching Module 5's NumPy class.

    Mirrors the structure (not the parameterization) of nn.TransformerEncoderLayer
    but is explicit + smaller so the parity proof is direct.
    """

    def __init__(self, d: int, h: int):
        super().__init__()
        self.d = d
        self.h = h
        self.d_k = d // h
        assert d % h == 0, "d must be divisible by h"

        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)

        # Per-head Q/K/V projections — kept as separate Linear modules so each
        # head's weights live in a 1:1 correspondence with the NumPy class's
        # `block.heads[i] = (W_Q, W_K, W_V)` triple.
        self.heads_q = nn.ModuleList([nn.Linear(d, self.d_k, bias=False) for _ in range(h)])
        self.heads_k = nn.ModuleList([nn.Linear(d, self.d_k, bias=False) for _ in range(h)])
        self.heads_v = nn.ModuleList([nn.Linear(d, self.d_k, bias=False) for _ in range(h)])

        self.W_O = nn.Linear(h * self.d_k, d, bias=False)

        # FFN: d -> 4d -> d with tanh-approx GELU (matches Module 5).
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * d, d),
        )

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d) -> (B, T, d).  Also accepts (T, d) for parity testing."""
        squeeze = (x.dim() == 2)
        if squeeze:
            x = x.unsqueeze(0)             # (1, T, d)

        B, T, _ = x.shape

        # ----- MHA -----
        h_in = self.ln1(x)
        mask = self._causal_mask(T, x.device)
        head_outs = []
        for q_lin, k_lin, v_lin in zip(self.heads_q, self.heads_k, self.heads_v):
            Q = q_lin(h_in)                # (B, T, d_k)
            K = k_lin(h_in)
            V = v_lin(h_in)
            scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
            scores = scores.masked_fill(mask, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            head_outs.append(attn @ V)     # (B, T, d_k)
        attn_out = self.W_O(torch.cat(head_outs, dim=-1))  # (B, T, d)
        x = x + attn_out

        # ----- FFN -----
        x = x + self.ffn(self.ln2(x))

        if squeeze:
            x = x.squeeze(0)
        return x


class TorchGPT(nn.Module):
    """Same architecture as `numpy_gpt.NumpyGPT` — trainable."""

    def __init__(self, vocab_size: int, d: int, h: int, n_layers: int, max_T: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(max_T, d)
        self.blocks = nn.ModuleList(
            [TorchTransformerBlock(d, h) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)  # no bias, like GPT-2
        self.vocab_size = vocab_size
        self.d = d
        self.h = h
        self.n_layers = n_layers
        self.max_T = max_T

        # GPT-2 init (matches NumpyGPT's std=0.02 default — though parity tests
        # don't depend on this, since they copy weights over explicitly).
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, ids: torch.Tensor, targets: torch.Tensor | None = None):
        """ids: (B, T) long.  targets: same shape, or None.

        If targets is None, returns logits of shape (B, T, V).
        If targets is given, also returns the mean cross-entropy loss.
        """
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        B, T = ids.shape
        assert T <= self.max_T, f"T={T} > max_T={self.max_T}"

        pos = torch.arange(T, device=ids.device)
        x = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]   # (B, T, d)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)        # (B, T, V)

        loss = None
        if targets is not None:
            # Standard next-token cross-entropy; flatten (B, T) into one batch dim.
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None):
        """Greedy / temperature sampling. ids: (B, T)."""
        for _ in range(max_new_tokens):
            ids_cond = ids[:, -self.max_T :]
            logits, _ = self.forward(ids_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_ids], dim=1)
        return ids


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    V, d, h, n_layers, max_T = 65, 64, 4, 2, 64
    model = TorchGPT(V, d, h, n_layers, max_T)
    print(f"torch model params: {count_parameters(model):,}")

    # Forward sanity
    ids = torch.randint(0, V, (1, 8))
    logits, loss = model(ids, targets=ids)   # using ids as both for the sanity check
    print(f"logits shape: {tuple(logits.shape)}  loss: {loss.item():.3f}")
    print(f"random init loss should be ≈ ln(V) = {math.log(V):.3f}")
