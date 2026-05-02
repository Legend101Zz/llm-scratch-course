"""Module 7 — Mini-GPT in PyTorch. Fill the TODOs."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------- config -----------------
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 128
    n_layer:   int = 6
    n_head:    int = 6
    n_embd:    int = 192
    dropout:   float = 0.2


# ----------------- attention -----------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head

        # Combined QKV projection — efficient (one matmul instead of three)
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # causal mask
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        # TODO:
        # qkv = self.qkv(x)                                   # (B, T, 3C)
        # q, k, v = qkv.split(C, dim=2)                       # each (B, T, C)
        # # split heads:
        # q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)   # (B, h, T, hd)
        # k = ...
        # v = ...
        # # scaled dot product:
        # att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)     # (B, h, T, T)
        # att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # att = self.attn_drop(att)
        # y = att @ v                                                    # (B, h, T, hd)
        # y = y.transpose(1, 2).contiguous().view(B, T, C)               # merge heads
        # y = self.resid_drop(self.proj(y))
        # return y
        raise NotImplementedError


# ----------------- block -----------------
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        # TODO: pre-norm residual block
        # x = x + self.attn(self.ln1(x))
        # x = x + self.mlp(self.ln2(x))
        # return x
        raise NotImplementedError


# ----------------- GPT -----------------
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # weight tying (GPT-2 trick): share embedding & head weights
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx: (B, T) int64 token ids; targets: (B, T) or None
        B, T = idx.shape
        assert T <= self.cfg.block_size

        # TODO:
        # pos = torch.arange(T, device=idx.device)
        # x = self.tok_emb(idx) + self.pos_emb(pos)              # (B, T, C)
        # x = self.drop(x)
        # for blk in self.blocks: x = blk(x)
        # x = self.ln_f(x)
        # logits = self.head(x)                                  # (B, T, V)
        # loss = None
        # if targets is not None:
        #     loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        # return logits, loss
        raise NotImplementedError


# ----------------- generation -----------------
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    model.eval()
    block_size = model.cfg.block_size
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


if __name__ == "__main__":
    cfg = GPTConfig()
    model = GPT(cfg)
    n = sum(p.numel() for p in model.parameters())
    print(f"params: {n/1e6:.2f} M")
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = generate(model, idx, 50)
    print("untrained sample shape:", out.shape)
