# Paper Reading Cheat-Sheets

Read these BEFORE / DURING the matching modules — not all at once.

| Paper                        | Read before | Sections to focus on             |
|------------------------------|-------------|----------------------------------|
| Attention Is All You Need    | Module 4    | §3 (model), §3.2 (attention)     |
| GPT-2 (Unsupervised Multitask) | Module 7  | Abstract, §2 (Approach), Table 2 |
| DeepSeek-R1                  | Module 10   | Abstract, §2.2 (GRPO), §2.3 (R1-Zero), §4 (results) |

---

## 1. Attention Is All You Need (Vaswani et al., 2017)

**One-liner:** Replace recurrence with attention; train faster, better, parallelizable.

**Key contributions:**
- Self-attention as the core operation (no RNN, no CNN).
- Multi-head attention (different heads learn different relationships).
- Positional encodings (sinusoidal in this paper; learned in GPT-2).
- Encoder–decoder architecture for translation.

**Equations to know cold:**
$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

**What changed since:** decoder-only models (GPT) dominate; encoder-decoder still good for seq2seq (T5). Pre-norm replaced post-norm. Multi-query and grouped-query attention reduce KV cache size.

**Confused readers' FAQ:**
- "Why $\sqrt{d_k}$?" — variance of dot product = $d_k$; dividing keeps softmax temperature ~constant across head sizes.
- "Why is the decoder masked?" — autoregressive: token $t$ shouldn't see $t+1, \ldots$.

---

## 2. Language Models are Unsupervised Multitask Learners (Radford et al., 2019)

**One-liner:** Scale up a transformer LM, train on diverse internet text, you get zero-shot task performance for free.

**Key contributions:**
- 1.5B-param decoder-only transformer.
- WebText dataset (curated by Reddit upvotes).
- Showed scaling makes zero-shot work — early hint at "scaling laws".
- Released the smaller variants (124M, 355M, 774M) — became the default toy model.

**Architecture details (relevant to Module 7):**
- Pre-norm (LayerNorm before sublayers, plus a final LN).
- Modified init: `std=0.02` for most weights, scaled-down for residual projections.
- Weight tying: input embedding == output projection.
- BPE tokenizer (50,257 vocab — `tiktoken` `gpt2`).

**What changed since:** GPT-3/4 are 100×–1000× bigger; instruction tuning + RLHF; longer context; mixture-of-experts.

---

## 3. DeepSeek-R1 (DeepSeek-AI, 2025)

**One-liner:** RL with rule-based rewards, applied to a strong base model, produces a reasoning model — no human preference labels needed.

**Key contributions:**
- **R1-Zero**: pure RL (GRPO) on the base model; reasoning emerges, including self-verification, backtracking, "Wait, let me check…" patterns.
- **R1**: cold-start SFT data + multi-stage RL → polished, deployable reasoner.
- **GRPO**: critic-free PPO variant; group-relative advantages.
- **Distillation**: R1 outputs as SFT data for smaller bases (Qwen, Llama) yields small, strong reasoners.

**Equations to know cold:**
- Group advantages: $A_i = (r_i - \bar r) / \sigma_r$.
- Clipped objective + KL anchor: $\min(\rho A, \text{clip}(\rho, 1-\epsilon, 1+\epsilon) A) - \beta \text{KL}[\pi_\theta\|\pi_\text{ref}]$.

**Reward design (math):**
- Accuracy reward (extract `<answer>...</answer>`, compare to ground truth).
- Format reward (did it follow `<think>...</think><answer>...</answer>`?).
- That's it. Pure rules.

**Why it shocked people:**
- No learned reward model needed for reasoning domains.
- Pure RL from base produced *emergent* metacognitive behavior.
- Distilled small models beat much larger non-reasoning models on math.

**Open questions (frontier):**
- Does this scale beyond verifiable domains (writing, dialogue)?
- How much of R1's behavior was already in the base model, just unlocked?
- What's the right reward for "good reasoning" outside math/code?

---

## How to read a deep-learning paper in 20 minutes

1. **Abstract + Figure 1 + Table 1** (3 min) — what & how much.
2. **Introduction** last paragraph (2 min) — explicit claims.
3. **Method** section, focus on equations (10 min) — re-derive the math on paper.
4. **Results** main table (3 min) — what beats what, by how much.
5. **Limitations + Conclusion** (2 min) — what they admit doesn't work.

Skip: related work, hyperparameter appendix (unless reproducing).
