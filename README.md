# 🥋 LLM From Scratch — 8-Hour Sprint Course

> Build an autograd engine, a transformer, a GPT, fine-tune it with LoRA, and implement DeepSeek-R1's GRPO — in one day. Type every line. No magic.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)

A self-paced, hands-on curriculum that takes you from **scalar autograd → modern reasoning models** in 11 modules. Designed as a **single-day sprint** for people who already know basic deep learning but want to *actually build* an LLM end-to-end.

## What you'll build

By hour 8, you will have written:

1. **Your own autograd engine** (scalar Value class, full backprop) — micrograd-style.
2. **A neural net library** on top of it — Neuron / Layer / MLP that learns XOR.
3. **A char tokenizer + a tiny BPE** — so you understand `tiktoken` from the inside.
4. **Self-attention from scratch** in numpy — Q/K/V, causal masking, multi-head.
5. **A full transformer block** — layernorm, residuals, FFN, GELU.
6. **A mini-GPT in PyTorch** (~1M params) — nanoGPT-style.
7. **Trained the GPT on Colab** — emits Shakespeare-flavored text.
8. **LoRA fine-tuning** — train ~1% of params and see it work.
9. **A toy GRPO loop** — DeepSeek-R1's RL algorithm, minimal but real.

## Why this exists

LLM theory is widely taught. Hands-on production code is everywhere. What's missing is the **middle**: a guided, friction-free path where you build *every component yourself* — including the autograd that backs everything — before reaching for PyTorch shortcuts. That gap is what this course fills.

> "What I cannot create, I do not understand." — Feynman

## Repo structure

```
.
├── README.md               ← you are here (overview + credits)
├── SCHEDULE.md             ← detailed 8-hour schedule + conceptual map
├── glossary.md             ← every term defined once (keep this open)
├── debugging.md            ← shape errors, NaN losses, OOM cheatsheet
├── whats_next.md           ← post-sprint progression (Tier 1 → 4)
├── requirements.txt
│
├── 00_start/               ← mental model + setup (15 min)
├── 01_autograd/            ← Value class + backprop (50 min)
├── 02_neural_net/          ← MLP on your autograd (30 min)
├── 03_tokenizer_bigram/    ← char + tiny BPE + bigram model (25 min)
├── 04_attention_scratch/   ← QKV attention in numpy (50 min)
├── 05_transformer_scratch/ ← full block in numpy (40 min)
├── 06_pytorch_crash/       ← PyTorch in 30 min
├── 07_gpt_pytorch/         ← mini-GPT model (60 min)
├── 08_train_colab/         ← train it on a free Colab T4 (45 min)
├── 09_finetune_lora/       ← LoRA fine-tuning end-to-end (30 min)
├── 10_deepseek_r1/         ← GRPO + reasoning (45 min)
│
├── extras/                 ← RoPE, GQA/MQA, FlashAttention, MoE,
│                             scaling laws, RLHF vs GRPO, quantization
└── papers/                 ← the 3 papers + annotated cheatsheets
```

Each module has the same structure:
- `README.md` — math + intuition + diagrams + reflection prompts.
- `starter.py` — skeleton with `# TODO` comments.
- `solution.py` — open *only after* tests pass / the timer rings.
- (where applicable) `test.py` that compares your impl to PyTorch.

## Quick start

```bash
git clone https://github.com/Legend101Zz/llm-scratch-course.git
cd llm-scratch-course
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then open [`SCHEDULE.md`](SCHEDULE.md) for the full schedule + conceptual map, or jump straight into [`00_start/`](00_start/) and start the timer.

For Module 8 onward, you'll need a free [Google Colab](https://colab.research.google.com) account (T4 GPU). Upload [`08_train_colab/train_colab.ipynb`](08_train_colab/train_colab.ipynb).

## Who this is for

✅ You know basic neural-net theory (have heard of backprop, weights, loss).
✅ You can read Python, even if PyTorch feels foreign.
✅ You'd rather *type a backprop loop* than watch the 100th lecture about it.

❌ It is **not** a gentle introduction. If gradient descent and matrix multiplication are new, do [3Blue1Brown's neural network series](https://www.3blue1brown.com/topics/neural-networks) first.

## How to use

- **Sprint mode (one day, ~8 hours):** the original challenge. Set a timer per module, type fast, peek at solutions if you blow past 1.5× the estimate.
- **Casual mode (one module per day):** read deeply, do the math by hand, attempt every reflection question.
- **Reference mode:** treat each module as standalone. Module 4 alone is a complete attention tutorial. Module 10 alone is a GRPO primer.

The course is non-linear-friendly: every module's README contains its own math derivation and intuition.

## Highlights

- **Math derived, not just stated.** Chain rule walked step by step in Module 1. Attention math derived from "tokens want to look at each other" in Module 4. GRPO objective dissected term by term in Module 10.
- **PyTorch parity tests.** Module 1 ends with `test.py` that runs your scalar autograd and PyTorch on the same expression — values match to 1e-6.
- **Real training, not toys.** Module 8 produces a model that emits readable Shakespeare-flavored text in ~5 min on a free Colab T4.
- **Modern context.** `extras/` covers RoPE, GQA, FlashAttention, MoE, Chinchilla scaling, RLHF→GRPO, quantization — the difference between a 2017 transformer and a 2026 production model.

## Inspirations & credits

This course stands on the shoulders of:

- **Andrej Karpathy** — [*Zero to Hero* playlist](https://karpathy.ai/zero-to-hero.html), [`micrograd`](https://github.com/karpathy/micrograd), [`nanoGPT`](https://github.com/karpathy/nanoGPT), [`minBPE`](https://github.com/karpathy/minbpe). The whole pedagogical philosophy of this course (scalar autograd → MLP → transformer → GPT) is a tribute to Andrej's videos. Modules 1, 2, 3, 7 directly mirror them.
- **Vaswani et al., 2017** — [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762).
- **Radford et al., 2019** — [*Language Models are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2).
- **Hu et al., 2021** — [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685).
- **Shao et al., 2024** — [*DeepSeekMath / GRPO*](https://arxiv.org/abs/2402.03300).
- **DeepSeek-AI, 2025** — [*DeepSeek-R1*](https://arxiv.org/abs/2501.12948).
- **Hoffmann et al., 2022** — [*Chinchilla scaling laws*](https://arxiv.org/abs/2203.15556).
- **Dao et al., 2022** — [*FlashAttention*](https://arxiv.org/abs/2205.14135).

Other influences: Sebastian Raschka's [*LLMs from Scratch*](https://github.com/rasbt/LLMs-from-scratch) book, Jay Alammar's [*Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/), and the [Anthropic Transformer Circuits](https://transformer-circuits.pub/) series.

## Contributing

Found a bug? Math typo? Confusing explanation? PRs welcome — please:

1. Open an issue first if it's a non-trivial change.
2. Keep the *teaching style* (intuition first, then math, then code with TODOs).
3. Don't add dependencies. The point is minimal.
4. New `extras/` pages are welcome if they cover a topic that genuinely changes how transformers are built today.

## License

[MIT](LICENSE) — fork it, remix it, teach with it. Attribution appreciated but not required.

## Star history

If this saved you a weekend, consider starring the repo — it helps others find it.

---

*Built in one focused sprint with the help of Claude. The bugs are mine; the inspiration is Karpathy's.*
