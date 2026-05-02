"""Self-contained Colab training script.

Drop this whole file into ONE Colab cell (with T4 GPU runtime). It will:
  - download tinyshakespeare
  - build the char tokenizer
  - define a mini-GPT (same as Module 7's solution)
  - train ~3000 steps with AdamW + grad clip + AMP
  - sample some text

If you'd rather run from a checkout: place this file alongside the course/
directory, mount /content/drive or scp the file, then `python train.py`.
"""

import os, math, time, urllib.request, sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. DATA
# ============================================================
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
PATH = "tinyshakespeare.txt"
if not os.path.exists(PATH):
    print("downloading tinyshakespeare...")
    urllib.request.urlretrieve(URL, PATH)

text = open(PATH).read()
chars = sorted(set(text))
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"corpus chars: {len(text):,} | vocab: {V}")


# ============================================================
# 2. MODEL  (same as 07_gpt_pytorch/gpt_solution.py, inlined)
# ============================================================
class GPTConfig:
    vocab_size = V
    block_size = 128
    n_layer = 6
    n_head = 6
    n_embd = 192
    dropout = 0.2


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.dropout = cfg.dropout
        # Optional: PyTorch 2.x's fused scaled_dot_product_attention is faster.
        # We keep manual code for transparency; flip to True to use the fast path.
        self.use_sdpa = True

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if self.use_sdpa and hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


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
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight    # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ============================================================
# 3. TRAINING UTILITIES
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
cfg = GPTConfig()


def get_batch(split, batch_size=64):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - cfg.block_size - 1, (batch_size,))
    x = torch.stack([d[i:i + cfg.block_size] for i in ix])
    y = torch.stack([d[i + 1:i + cfg.block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, eval_iters=50):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, l = model(x, y)
            losses[k] = l.item()
        out[split] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -cfg.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


# ============================================================
# 4. RUN
# ============================================================
torch.manual_seed(1337)
model = GPT(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"params: {n_params/1e6:.2f} M")

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))

# Mixed precision: fp16 on T4. (bf16 only on A100/H100/etc.)
use_amp = (device == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

MAX_ITERS = 3000
EVAL_EVERY = 500
t0 = time.time()
model.train()

for step in range(MAX_ITERS):
    if step % EVAL_EVERY == 0:
        losses = estimate_loss(model)
        elapsed = time.time() - t0
        print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | {elapsed:.0f}s")

    x, y = get_batch("train")
    with torch.cuda.amp.autocast(enabled=use_amp):
        _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()

print(f"\ntraining done in {time.time()-t0:.0f}s")
torch.save(model.state_dict(), "ckpt.pt")
print("saved ckpt.pt")

# sample
print("\n--- SAMPLE ---")
seed = torch.zeros((1, 1), dtype=torch.long, device=device)
out = generate(model, seed, max_new_tokens=500, temperature=0.8, top_k=40)
print(decode(out[0].tolist()))
