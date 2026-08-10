# 0 → frontier-lab — a public build log

> My from-scratch journey through the LLM stack — kernels below, agent loops above — toward genuinely contributing at a frontier lab. The repo is the artifact: every derivation, every parity test, every roofline calculation is committed evidence.
>
> **This is a personal learning system, not a course for other people.** It's public because the trail is the signal, and because a raw honest log is worth more than a polished one. Nothing here is optimized for a reader — if you find it useful, good, but that's a side effect.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![JAX](https://img.shields.io/badge/JAX-Phase%204%2B-blue.svg)](https://docs.jax.dev/)

## What this is

A multi-month arc built around one thesis from [Vlad Feinberg's article](https://www.vladfeinberg.com/) (paraphrased in [`../frontier-lab.md`](../frontier-lab.md) for internal reference): the most reliable path into a frontier lab is to demonstrate a specific skill the lab needs, by working at the **edges of the stack** — kernels below (Track 1), agent loops above (Track 2) — with mathematical maturity and a public artifact that proves it.

This repo *is* that public artifact, built in real time. The point is **proof, not a certificate.**

> **Heads up — framing pivoted twice.** Earlier commits called this an "8-hour sprint" (retired 2026-05-23, see [`DECISIONS.md`](DECISIONS.md) D-0001). Since 2026-08-05 progress is measured in **capability claims**, not modules, against a dated April 2027 checkpoint — see [D-0008](DECISIONS.md).

## Progress is measured in claims

A claim is one falsifiable capability statement — "I can implement KV-cache inference from scratch matching HuggingFace token-for-token" — never a topic. Each has an acceptance test named up front, a 1–2 week timebox, and a shipped public artifact as its terminal event. **A claim without its artifact is not done.**

**24 claims on the checkpoint path.** Currently proven: **0**. Active: **Claim R0a** — a strided tensor library and a cache-blocked matmul, in Rust, from nothing.

> **Pivot, 2026-08-10 ([D-0009](DECISIONS.md)).** The from-scratch core is being rebuilt in **Rust with `std` only** — no ndarray, nalgebra, candle, burn, tch, tokenizers, BLAS, or autodiff crate. Own tensor, own strided memory model, own matmul, own autograd, own BPE. It ends by loading OpenAI's real GPT-2 124M checkpoint and reproducing its output token-for-token. Python `phase0/` is retained untouched as trail — superseded, not deleted. Full design: [`RUST_PHASE_0_1.md`](RUST_PHASE_0_1.md).

Phases survive as groupings; claims are the rows. Full ladder with acceptance tests: [`COURSE_MAP.md`](COURSE_MAP.md).

| Grouping | Claims | Path |
|---|---|---|
| **Phase 0–1 (Rust)** — tensor + matmul → autograd → **GPT-2 124M logit parity** | R0a, R0b, R1 | checkpoint |
| **PyTorch block** — rebuild GPT-2, 3-way parity vs Rust → training hygiene + 3 timed broken-run diagnoses | P1, P2 | checkpoint |
| **Phase 1 (PyTorch)** — LoRA → **~100M model end-to-end** on rented GPU | 5–6 | checkpoint |
| **JAX + scaling** — primer, then every scaling-book exercise | 7–9 | checkpoint |
| **Phase 4** — kernels: roofline → CUDA → Triton → Pallas. Includes **Feinberg Exercise A** | 10–15 | checkpoint |
| **Phase 5** — **Feinberg Exercise B** (Pallas `ragged_dot` beater) | 16 | checkpoint |
| **GenHash** — VQ → HiFiC repro → JPEG XL/Kodak benchmark → honest writeup | 17–20 | checkpoint |
| **Phase 6** — KV cache + prefill/decode roofline | 21 | checkpoint |
| **Phase 2** — GRPO derived cold | 22 | checkpoint |
| **Checkpoint gate** — founding-engineer depth interview, cold | 23 | checkpoint |
| Quantization proper · serving loop · real GRPO · agents · **Phase 8 framework build** | — | continuation |

## The April 2027 checkpoint

Frontier-lab readiness stays the north star. The checkpoint is a **dated waypoint on the same arc** — what must already be banked by April 2027:

- **(a)** a ~100M-param LM trained end-to-end on my own implementation, with loss curves, failure modes and cost
- **(b)** GenHash complete: HiFiC reproduced, benchmarked against JPEG XL on Kodak with my own plots, honestly positioned
- **(c)** at least one public artifact a recognizable ML person shared unprompted
- **(d)** I can pass a founding-engineer depth interview at a serious AI company cold

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
├── PHASE8_FRAMEWORK.md      ← Phase 8 plan: "build your own PyTorch" (placeholder, fleshed out at end of Phase 6)
├── devlog/                  ← one entry per session
├── glossary.md              ← terms defined once
├── debugging.md             ← failure modes cheatsheet
├── requirements.txt
│
│   ──── Phase 0 modules (from-scratch core) ────
├── phase0/
│   ├── PHASE0_CLOSURE.md          ← friendly closing checklist (read first if mid-Phase-0)
│   ├── 00_start/                  ← mental model + setup
│   ├── 01_autograd/               ← Value class + backprop ✅
│   ├── 02_neural_net/             ← MLP on your autograd ✅
│   ├── 03_tokenizer_bigram/       ← char + BPE + bigram model 🟨 (BPE patched in)
│   ├── 04_attention_scratch/      ← QKV attention in NumPy ✅ (+ roofline section)
│   ├── 05_transformer_scratch/    ← full block in NumPy ✅
│   ├── 06_pytorch_crash/          ← PyTorch deep-dive (4 sub-docs + drills) ✅
│   └── 07_phase0_capstone/        ← tiny GPT (numpy + torch) ✅ Phase 0 capstone
│
│   ──── Phase 1+ legacy stubs (sprint-era, will be rewritten per D-0006) ────
├── 07_gpt_pytorch/          ← (legacy — replaced by phase0/07_phase0_capstone/)
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
└── notes.md        ← scratch — confusions, dead ends, aha moments
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then [`COURSE_MAP.md`](COURSE_MAP.md) for the claim ladder. Mid-Claim-1? → [`phase0/PHASE0_CLOSURE.md`](phase0/PHASE0_CLOSURE.md) is the runbook from venv setup to the cold quiz.

Compute: free Colab T4 covers through Claim 5, and the free TPU tier covers Exercise A. Claim 6 (~100M) and Claims 17–19 (HiFiC + Kodak benchmark) need paid A100 time — a few hundred dollars across both. Claim 6's acceptance test requires the cost to be itemized.

## Reading the trail

The evidence is in `**/evidence/` and `**/hand_math/`. Per [`CONVENTIONS.md`](CONVENTIONS.md) a module is only done when the parity test passes, a hand-math derivation exists, and the evidence is committed — and per [D-0008](DECISIONS.md), a **claim** is only done when its acceptance test passed *and* its artifact shipped.

`PROGRESS.md` records what is **proven**, not what is scaffolded. Where it says nothing has been verified, nothing has been verified.

## Realistic expectations (from `../frontier-lab.md`)

> Doing these exercises is a strong **start**, not a shortcut past the years of signaling a traditional path provides. The realistic payoff is real skills + a public repo that demonstrates something useful and adopted — after which the choice of where to work becomes mine.

The arc is long because the goal is long — years-not-months in full depth. The April 2027 checkpoint doesn't shorten it; it dates a waypoint on it, so drifting can't be mistaken for progress.

## Inspirations & credits

- **Andrej Karpathy** — [*Zero to Hero*](https://karpathy.ai/zero-to-hero.html), [`micrograd`](https://github.com/karpathy/micrograd), [`nanoGPT`](https://github.com/karpathy/nanoGPT), [`minbpe`](https://github.com/karpathy/minbpe), [`nanochat`](https://github.com/karpathy/nanochat), [`autoresearch`](https://github.com/karpathy/autoresearch). The build-it-yourself philosophy here is a tribute to Andrej's videos.
- **Vlad Feinberg** — [*How to land a job at a frontier lab*](https://www.vladfeinberg.com/) (2026). The North Star.
- **Reiner Pope** — [*The math behind how LLMs are trained and served*](https://www.dwarkesh.com/p/reiner-pope) (Dwarkesh, 2026). The roofline lecture.
- **Sebastian Raschka** — [*Build a Large Language Model (From Scratch)*](https://github.com/rasbt/LLMs-from-scratch).
- **Anthropic Transformer Circuits Thread** — [transformer-circuits.pub](https://transformer-circuits.pub/) — the mechanistic-interpretability lens.

Plus the paper trail in [`RESOURCES.md`](RESOURCES.md).

## Contributing

This is a personal build log, not a course — PRs aren't expected and it isn't maintained for anyone else. If you spot a math error, open an issue: a wrong derivation sitting in my evidence trail is worth knowing about.

## License

[MIT](LICENSE).

---

*"What I cannot create, I do not understand." — Feynman.*
*The repo is the proof.*
