# 0 → frontier-lab — a public build log

> A from-scratch journey through the LLM stack — kernels below, agent loops above — toward genuinely contributing at a frontier lab. The repo is the artifact: every derivation, every parity test, every roofline calculation is committed evidence.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![JAX](https://img.shields.io/badge/JAX-Phase%204%2B-blue.svg)](https://docs.jax.dev/)

## What this is

A multi-month curriculum built around one thesis from [Vlad Feinberg's article](https://www.vladfeinberg.com/) (paraphrased in [`../frontier-lab.md`](../frontier-lab.md) for internal reference): the most reliable path into a frontier lab is to demonstrate a specific skill the lab needs, by working at the **edges of the stack** — kernels below (Track 1), agent loops above (Track 2) — with mathematical maturity and a public artifact that proves it.

This repo *is* that public artifact, built in real time. Each phase ends with an artifact that proves something specific. Each module ends with a handwritten derivation, a parity test, and (where applicable) a roofline calculation. The point is **proof, not a certificate.**

> **Heads up — framing pivoted 2026-05-23.** Earlier commits called this an "8-hour sprint." That framing is now retired. The course is now a deep, evidence-driven journey; the original sprint modules form the foundation of Phase 0, patched with new discipline conventions. See [`DECISIONS.md`](DECISIONS.md) D-0001 for the rationale.

## Phase map

| # | Phase | What gets built | Status |
|---|---|---|---|
| 0 | **From-scratch core** | Build the transformer end-to-end by hand, parity-tested against PyTorch. Tiny GPT trained on a toy task | 🟨 in progress |
| 1 | **Real training & fine-tuning** | GPT-2-small-ish trained on Colab, LoRA-fine-tuned, with debugging-a-broken-run exercises | ⬜ |
| 2 | **Reasoning & RL** | GRPO derived + implemented from scratch; reward-shape a small base model on a verifiable task | ⬜ |
| 3 | **Literature fluency + JAX primer** | Reconstruct the canon (Chinchilla derived by hand); JAX/Flax/Optax fluency | ⬜ |
| 4 | **Kernels (below the stack)** | Roofline → CUDA → Triton → Pallas. Capstone = **Feinberg's Exercise A** (10M JAX/TPU adder + hand Chinchilla) | ⬜ |
| 5 | **Quantization** | Walk the quality↔perf tradeoff hands-on. Capstone = **Feinberg's Exercise B** (Pallas `ragged_dot` beater) | ⬜ |
| 6 | **Inference engine (above the stack pt 1)** | nano-vLLM from scratch — KV cache → paged attention → continuous batching | ⬜ |
| 7 | **Agents (above the stack pt 2)** | Hypothesis-driven, measured agent experiment in the ADRS / autoresearch lineage | ⬜ |
| 8 | **Capstone & signaling artifact** | Public repo + handwritten derivations + screen-recorded build evidence | ⬜ |

Full arc with per-module deliverables: [`COURSE_MAP.md`](COURSE_MAP.md).

## How the journey is recorded

| File | What it is |
|---|---|
| [`JOURNEY.md`](JOURNEY.md) | The narrative table of contents — newest devlog entry at top |
| [`devlog/`](devlog/) | One entry per working session, written for an external reader |
| [`PROGRESS.md`](PROGRESS.md) | Where I am, what's next, what's blocking |
| [`SKILLS.md`](SKILLS.md) | Skill tree (0–5) with per-skill proof files |
| [`REVIEW.md`](REVIEW.md) | Spaced-repetition Qs with due dates — recall is forced, not optional |
| [`DECISIONS.md`](DECISIONS.md) | Architectural / strategic decisions with rationale |
| [`RESOURCES.md`](RESOURCES.md) | Curated, link-verified library by phase |
| [`MENTOR_LOG.md`](MENTOR_LOG.md) | Terse internal journal from the mentor (Claude) — kept public for transparency |
| [`CONVENTIONS.md`](CONVENTIONS.md) | The per-module discipline template (`hand_math/`, `evidence/`, parity, roofline) |

## Repo structure

```
.
├── README.md                ← you are here (public face)
├── COURSE_MAP.md            ← Phase 0–8 spine
├── CONVENTIONS.md           ← per-module template (the rules)
├── DECISIONS.md             ← strategic call log
├── RESOURCES.md             ← curated, verified library
├── PROGRESS.md              ← current state + next step
├── SKILLS.md                ← skill tree (0–5)
├── REVIEW.md                ← spaced-repetition queue
├── MENTOR_LOG.md            ← terse internal journal
├── JOURNEY.md               ← narrative TOC
├── devlog/                  ← one entry per session
├── glossary.md              ← terms defined once
├── debugging.md             ← failure modes cheatsheet
├── requirements.txt
│
│   ──── Phase 0 modules (from-scratch core) ────
├── 00_start/                ← mental model + setup
├── 01_autograd/             ← Value class + backprop ✅
├── 02_neural_net/           ← MLP on your autograd ✅
├── 03_tokenizer_bigram/     ← char + BPE + bigram model 🟨 (BPE patched in)
├── 04_attention_scratch/    ← QKV attention in NumPy ✅ (+ roofline section)
├── 05_transformer_scratch/  ← full block in NumPy ✅
├── 06_pytorch_crash/        ← PyTorch deep-dive (4 sub-docs + drills) ✅
├── 07_gpt_pytorch/          ← mini-GPT (to be redesigned as Phase 0 capstone)
│
│   ──── Phase 1+ modules ────
├── 08_train_colab/          ← (skeletal — Phase 1.1 will rewrite)
├── 09_finetune_lora/        ← (skeletal — Phase 1.4 will rewrite)
├── 10_deepseek_r1/          ← (skeletal — Phase 2 will rewrite)
│
├── extras/                  ← supplementary topics (RoPE, GQA, FlashAttention,
│                              MoE, scaling laws, GRPO vs RLHF, quantization)
└── papers/                  ← reference papers + cheatsheets
```

Per [`DECISIONS.md`](DECISIONS.md) D-0005, Phase 1+ module folders are designed *when we reach them*, not in advance. The existing `07–10` folders are skeletal sketches from the sprint era and will be rewritten.

## Per-module structure

Every module from Phase 0 onward follows the template in [`CONVENTIONS.md`](CONVENTIONS.md):

```
NN_module_name/
├── README.md       ← intuition + math + diagrams + roofline + reflection
├── starter.py      ← skeleton with # TODO comments
├── solution.py     ← reference impl (read only after starter passes tests)
├── test.py         ← PyTorch (or JAX) parity test
├── hand_math/      ← pencil derivations (photo or transcribed)
├── evidence/       ← test outputs, metrics, roofline calculations
└── notes.md        ← your scratch — confusions, dead ends, aha moments
```

## Quick start

```bash
git clone https://github.com/Legend101Zz/llm-scratch-course.git
cd llm-scratch-course
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then read [`COURSE_MAP.md`](COURSE_MAP.md) for the full arc, or jump into [`00_start/`](00_start/) and begin.

For Phase 1 onward you'll need [Google Colab](https://colab.research.google.com) (T4 free tier) or similar cheap-GPU access; for Phases 4–5 you'll need TPU access (Colab free tier suffices for the graded JAX exercises).

## Who this is for

✅ You can read Python and have heard of backprop / softmax / gradient descent.
✅ You want to *type every line* and have the evidence to prove you did.
✅ You'd rather feel stupid for three months than fake competence forever.

❌ It is not a gentle introduction. If matrix multiplication and the chain rule are new, do [3Blue1Brown's Neural Networks series](https://www.3blue1brown.com/topics/neural-networks) first.

❌ It is not a "watch and learn" course. Watching has been tried and failed; that's why this exists.

## How to read this repo

- **As an outsider:** start with [`JOURNEY.md`](JOURNEY.md) for the narrative, dip into the devlog entries that interest you, then [`COURSE_MAP.md`](COURSE_MAP.md) for the arc.
- **As a learner:** follow [`COURSE_MAP.md`](COURSE_MAP.md) module by module. Each module's `README.md` is self-contained (math + intuition + diagrams + roofline).
- **As a verifier:** the evidence trail is in `**/evidence/` and `**/hand_math/`. Per [`CONVENTIONS.md`](CONVENTIONS.md), a module is only "done" when all five criteria are met (parity test passes, hand-math derivation exists, evidence committed, etc.).

## Realistic expectations (from `../frontier-lab.md`)

> Doing these exercises is a strong **start**, not a shortcut past the years of signaling a traditional path provides. The realistic payoff is real skills + a public repo that demonstrates something useful and adopted — after which the choice of where to work becomes mine.

The course is long because the goal is long. The arc is years-not-months in full depth; the multi-month intensive build (Phases 0–7) is the foundation, Phase 8 (artifact + reach-out + landing) is on top of that.

## Inspirations & credits

- **Andrej Karpathy** — [*Zero to Hero*](https://karpathy.ai/zero-to-hero.html), [`micrograd`](https://github.com/karpathy/micrograd), [`nanoGPT`](https://github.com/karpathy/nanoGPT), [`minbpe`](https://github.com/karpathy/minbpe), [`nanochat`](https://github.com/karpathy/nanochat), [`autoresearch`](https://github.com/karpathy/autoresearch). The pedagogical philosophy of this course is a tribute to Andrej's videos.
- **Vlad Feinberg** — [*How to land a job at a frontier lab*](https://www.vladfeinberg.com/) (2026). The North Star.
- **Reiner Pope** — [*The math behind how LLMs are trained and served*](https://www.dwarkesh.com/p/reiner-pope) (Dwarkesh, 2026). The roofline lecture.
- **Sebastian Raschka** — [*Build a Large Language Model (From Scratch)*](https://github.com/rasbt/LLMs-from-scratch).
- **Anthropic Transformer Circuits Thread** — [transformer-circuits.pub](https://transformer-circuits.pub/) — the mechanistic-interpretability lens.

Plus the paper trail in [`RESOURCES.md`](RESOURCES.md).

## Contributing

This is a personal build log; PRs aren't expected. But if you spot a math typo, a dead link, or a confusing explanation, open an issue.

## License

[MIT](LICENSE) — fork it, remix it, teach with it. Attribution appreciated.

---

*"What I cannot create, I do not understand." — Feynman.*
*The repo is the proof.*
