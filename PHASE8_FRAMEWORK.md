# PHASE8_FRAMEWORK.md — your own PyTorch

> **Status: placeholder.** This file is a forward commitment, not a current plan.
> It will be fleshed out incrementally as Phases 4 and 6 surface real design ideas, and fully detailed when we reach Phase 8.
> For now it locks in *what* Phase 8 is and *why*, so that all earlier phases know what they're building toward.
>
> Linked from: [COURSE_MAP.md](COURSE_MAP.md) Phase 8 · [DECISIONS.md](DECISIONS.md) D-0007.

---

## What this is

Phase 8 is a dedicated **3–4 month framework project**: build your own PyTorch-style ML library from scratch, end-to-end, then train a real (~12M-param) language model on it and publish.

This is **not** built incrementally during Phases 0–7. Those phases are pure learning — each component (kernels, inference, autograd) is built once in isolation for understanding, against PyTorch parity. Phase 8 then takes everything you learned and rebuilds it cleanly, integrated, opinionated, as a single coherent library.

The "build twice" principle: the first pass (during Phases 1–7) is messy, experimental, focused on understanding. The second pass (in Phase 8) is designed, with the benefit of knowing exactly what each piece should look like. The second pass is what gets published.

---

## Why it's structured this way

See [DECISIONS.md D-0007](DECISIONS.md) for the full reasoning. Short version:

1. **Learning-first wins.** Interweaving framework discipline across every phase would pull focus from the underlying ideas toward productization concerns. Phases 0–7 stay clean.
2. **Build twice produces a better artifact.** Karpathy's `nanoGPT` is the proof: years of building one-off models, then one clean pass produces the artifact that endures.
3. **De-risks half-finished artifacts.** A framework that's 40% built across 7 phases is bad signal. A framework that's 0% built (Phase 8 not started) just means "haven't reached the artifact phase yet."
4. **Hook decision is deferred to evidence.** No commitment to a specific unique angle (specific kernel, novel backend, minimalism, etc.) up front. You'll know by Phase 8 what's interesting about your build because you'll have built it.

---

## What goes in the framework

A normal ML library's surface area, scoped to what an LLM actually needs:

```
myframework/                  ← name TBD; placeholder
├── core/
│   ├── tensor.py             ← your own Tensor class, vectorized
│   ├── autograd.py           ← reverse-mode autograd graph
│   └── backends/
│       ├── numpy.py          ← CPU/NumPy fallback
│       └── cuda.py           ← GPU backend, kernels in kernels/
├── nn/
│   ├── module.py             ← nn.Module-equivalent base
│   ├── linear.py
│   ├── attention.py          ← MHA + flash-attention path
│   ├── norm.py               ← LayerNorm, RMSNorm
│   ├── embedding.py
│   └── transformer.py        ← composed block
├── optim/
│   ├── sgd.py
│   ├── adamw.py
│   └── schedulers.py
├── train/
│   ├── trainer.py            ← train loop + eval + checkpoint
│   └── mixed_precision.py
├── serve/
│   ├── kv_cache.py
│   ├── paged_attention.py
│   └── continuous_batch.py
├── kernels/
│   ├── cuda/
│   └── triton/
├── tests/                    ← parity vs PyTorch on every primitive
└── examples/
    └── train_12m_lm.py       ← the proof
```

---

## The 6 stages

Reproduced from [COURSE_MAP.md](COURSE_MAP.md) Phase 8:

| # | Stage | Approx duration |
|---|---|---|
| **8.1** | Core (ML side) — Tensor + autograd + nn primitives + parity tests | ~3–4 weeks |
| **8.2** | Training (ML side) — optimizers + schedulers + mixed precision + trainer | ~2–3 weeks |
| **8.3** | Kernels (GPU side) — port Phase 4 work as the GPU backend | ~3–4 weeks |
| **8.4** | Inference (GPU side) — port Phase 6 work as `framework.serve` | ~2–3 weeks |
| **8.5** | Train a real model — 12M-param LM end-to-end on your framework | ~2 weeks |
| **8.6** | Publish — README, blog, screen-recording, hook decision | ~1 week |

---

## What Phases 4 and 6 should do *now* in anticipation

You don't change what those phases build, but you change how you think about them:

- **Phase 4 (kernels):** Every CUDA / Triton / Pallas kernel you write knows it will be rewritten in Phase 8.3 as part of a real backend. So:
  - Write tests carefully (they're the spec for the Phase 8 rewrite)
  - Keep a `notes.md` of API mistakes and "what would I do differently"
  - Don't bother polishing the standalone module — Phase 8 is where polish happens
- **Phase 6 (inference):** Same posture. The nano-vLLM module is the *draft* of `framework.serve`. Optimize for learning + notes, not polish.

This actually makes Phases 4 and 6 *faster* because you're explicitly allowed to skip polish.

---

## What the 12M-param model is

The proof-of-life model that gets trained on the framework. Probably:
- ~12M parameters (matches the reference project, low enough to train on a single T4 or modest GPU)
- Trained on a clean dataset (tinyshakespeare → OpenWebText subset → maybe a small code corpus, TBD)
- Evaluated against a HuggingFace reference with identical architecture, trained on the same data, to demonstrate your framework reaches comparable loss
- Sample generations + perplexity + loss curve committed to evidence

This is what makes the artifact undeniable: "I built a framework AND trained a model on it AND it matches a reference."

---

## What "publishing" means (Phase 8.6)

The artifact only counts as published when:
- The repo has a clean README explaining what it is, how to install, how to train
- A blog post (Substack / Medium / personal) walks through the design choices
- A screen-recording (15–30 min) shows the build + the trained model running
- Twitter / X announcement with the demo (the trained model + a "trained on my own framework" caption)
- Optional: a 30-minute talk / livestream walking through the codebase

**Hook decision happens in 8.6, not earlier.** Examples from reference projects:
- tinygrad: lazy eval + new accelerator support
- micrograd: extreme minimalism (~100 lines, scalar)
- llm.c: single-file C, no dependencies
- mni-ml/framework: Rust backend + custom CUDA kernels

You may have one of these by Phase 8.6, or you may not. Either way the artifact stands.

---

## What this commits us to *now*

Nothing concrete yet. This file exists so that:
1. The user knows where the course ends and what's worth building toward
2. Phases 4 and 6 can write their code with awareness of the Phase 8 rewrite
3. The "is the course done?" question has a clean answer: no, not until Phase 8.6 ships

Phase 8 design happens **at end of Phase 6**, not now.

---

## Open questions (to be answered before Phase 8 starts)

- **Backend language for kernels.** Pure CUDA-C? Triton? Both? Rust + CUDA like the reference project? Decide at end of Phase 4.
- **The 12M-param model's task.** Generic LM, instruction-tuned, math, code? Decide at end of Phase 2.
- **JAX vs PyTorch as the parity reference.** PyTorch is the obvious choice but JAX might be cleaner for some primitives. Decide at end of Phase 3.
- **Name.** TBD. Should suggest the build philosophy — minimalism, performance, pedagogy, or whatever the hook turns out to be.
- **License.** Probably MIT (matches the course repo). Confirm at 8.6.

---

*The artifact is the proof. The framework is the artifact.*
