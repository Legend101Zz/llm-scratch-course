"""Module 9 — LoRA training demo.

Shows LoRA actually works: we take a pretrained mini-GPT (use the ckpt.pt from
Module 8 if you have it; otherwise we just train fresh on tinyshakespeare with
ALL params, then 'fine-tune' to ALL CAPS using LoRA).

The 'task': make the model emit ALL-CAPS Shakespeare. Reward signal: nope —
we just supervise on uppercased text via cross-entropy. With LoRA, only ~1%
of params trains.

Run on a T4 (or CPU for a slow demo). Best done in Colab.
"""

import os, sys, time, urllib.request
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "07_gpt_pytorch"))
from gpt_solution import GPT, GPTConfig
from lora import LoRALinear, replace_linears_with_lora, lora_trainable_params

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
PATH = os.path.join(os.path.dirname(__file__), "tinyshakespeare.txt")
if not os.path.exists(PATH):
    urllib.request.urlretrieve(URL, PATH)

text = open(PATH).read()
chars = sorted(set(text.upper() + text))     # union so both cases share vocab
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: torch.tensor([stoi[c] for c in s], dtype=torch.long)


class Cfg(GPTConfig):
    vocab_size = V; block_size = 64
    n_layer = 4; n_head = 4; n_embd = 128; dropout = 0.1


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
model = GPT(Cfg()).to(device)


def get_batch(corpus_ids, B=32, T=Cfg.block_size):
    ix = torch.randint(len(corpus_ids) - T - 1, (B,))
    x = torch.stack([corpus_ids[i:i+T] for i in ix])
    y = torch.stack([corpus_ids[i+1:i+T+1] for i in ix])
    return x.to(device), y.to(device)


# Phase 1: tiny "pretrain" on lowercase Shakespeare so model has SOMETHING.
print("=== phase 1: pretrain on normal text (full FT) ===")
data_pre = encode(text)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
for step in range(800):
    x, y = get_batch(data_pre)
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 100 == 0:
        print(f"step {step:4d}  loss {loss.item():.4f}")

# Phase 2: freeze all base params, install LoRA, fine-tune to UPPERCASE.
print("\n=== phase 2: LoRA fine-tune to ALL-CAPS ===")
for p in model.parameters(): p.requires_grad_(False)
replace_linears_with_lora(model, r=4, alpha=8)
model.to(device)
trainable = lora_trainable_params(model)
n_total = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in trainable)
print(f"trainable: {n_train:,} / total: {n_total:,}  ({100*n_train/n_total:.2f}%)")

data_caps = encode(text.upper())
opt = torch.optim.AdamW(trainable, lr=1e-3)
for step in range(400):
    x, y = get_batch(data_caps)
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step()
    if step % 50 == 0:
        print(f"step {step:4d}  loss {loss.item():.4f}")

# sample
@torch.no_grad()
def sample(prompt, n=200):
    model.eval()
    idx = encode(prompt).unsqueeze(0).to(device)
    for _ in range(n):
        idx_cond = idx[:, -Cfg.block_size:]
        logits, _ = model(idx_cond)
        probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    return "".join(itos[i] for i in idx[0].tolist())

print("\n--- sample after LoRA ---")
print(sample("ROMEO:"))
