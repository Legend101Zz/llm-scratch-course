"""Module 10 — toy GRPO loop on a tiny generation task.

Task: prompt is "Q:<digit>?A:". Reward 1 if the next-generated char == that digit.
Trains a tiny GPT to copy. Same mechanics as DeepSeek-R1's GRPO, just minimal.
"""

import copy, math, random, sys, os
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "07_gpt_pytorch"))
from gpt_solution import GPT, GPTConfig

torch.manual_seed(0); random.seed(0)

# tiny vocab: digits + punctuation
chars = list("0123456789Q:?A. \n")
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: torch.tensor([stoi[c] for c in s], dtype=torch.long)


class Cfg(GPTConfig):
    vocab_size = V; block_size = 32; n_layer = 2; n_head = 2; n_embd = 64; dropout = 0.0


device = "cuda" if torch.cuda.is_available() else "cpu"
policy = GPT(Cfg()).to(device)
ref = copy.deepcopy(policy)                          # frozen reference
for p in ref.parameters(): p.requires_grad_(False)
opt = torch.optim.AdamW(policy.parameters(), lr=1e-3)


def make_prompt():
    d = random.randint(0, 9)
    return f"Q:{d}?A:", str(d)


def sample_one(model, prompt_ids, max_new=2, temperature=1.0):
    """Sample with logprob bookkeeping."""
    model.eval()
    idx = prompt_ids.unsqueeze(0).to(device)
    logp_sum = 0.0
    new_tokens = []
    for _ in range(max_new):
        logits, _ = model(idx)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        logp = torch.log(probs.gather(-1, nxt) + 1e-12).squeeze()
        logp_sum = logp_sum + logp
        new_tokens.append(nxt.item())
        idx = torch.cat([idx, nxt], dim=1)
    return new_tokens, logp_sum, idx.squeeze(0)


def reward_fn(generated_tokens, target_char):
    text = "".join(itos[t] for t in generated_tokens)
    # +1 if first generated char matches target
    return 1.0 if (len(text) > 0 and text[0] == target_char) else 0.0


def grpo_step(G=8, eps=0.2, beta=0.02):
    prompt, target = make_prompt()
    pids = encode(prompt)

    # 1) sample group of G outputs from current policy
    samples = []  # list of (token_ids_tail, logprob_old, full_seq, reward)
    with torch.no_grad():
        for _ in range(G):
            toks, logp, seq = sample_one(policy, pids)
            r = reward_fn(toks, target)
            samples.append((toks, logp.detach(), seq, r))

    rewards = torch.tensor([s[3] for s in samples], device=device)
    if rewards.std() < 1e-8:    # all-equal → no signal this step
        return rewards.mean().item(), 0.0
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

    # 2) compute new logprobs under current policy (re-forward) + KL to ref
    policy.train()
    total_obj = 0.0
    total_kl = 0.0
    for (toks, logp_old, seq, _), A in zip(samples, advantages):
        # full seq logprobs from policy and ref, on the GENERATED positions only
        seq = seq.to(device).unsqueeze(0)
        logits_p, _ = policy(seq[:, :-1])
        logits_r, _ = ref(seq[:, :-1])
        logp_p = F.log_softmax(logits_p, dim=-1)
        logp_r = F.log_softmax(logits_r, dim=-1)
        # only the last len(toks) positions correspond to generated tokens
        gen_targets = seq[:, -len(toks):]
        gen_logp_p = logp_p[:, -len(toks):, :].gather(-1, gen_targets.unsqueeze(-1)).squeeze(-1).sum()
        gen_logp_r = logp_r[:, -len(toks):, :].gather(-1, gen_targets.unsqueeze(-1)).squeeze(-1).sum()

        ratio = torch.exp(gen_logp_p - logp_old)
        clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
        obj = torch.min(ratio * A, clipped * A)
        kl = (gen_logp_p - gen_logp_r)               # rough per-sequence KL surrogate
        total_obj = total_obj + obj
        total_kl = total_kl + kl

    loss = -(total_obj / G) + beta * (total_kl / G)
    opt.zero_grad(); loss.backward(); opt.step()
    return rewards.mean().item(), loss.item()


if __name__ == "__main__":
    for step in range(2000):
        r, l = grpo_step()
        if step % 100 == 0:
            print(f"step {step:4d}  mean_reward {r:.3f}  loss {l:.4f}")
