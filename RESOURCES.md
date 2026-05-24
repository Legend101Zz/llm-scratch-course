# RESOURCES.md — curated, link-verified library

> All URLs verified live as of **2026-05-23**. Organized by curriculum phase.
> One-line "why it matters" per entry. ✅ = live · ❌ = dead · → = superseded.
>
> When a link rots, mark it ❌ but **leave the entry** for audit trail; add the replacement underneath.

---

## Foundations (Phase 0)

- [Karpathy "Neural Networks: Zero to Hero" playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) ✅ — the canonical from-zero series; backprop through GPT and BPE. Course page: https://karpathy.ai/zero-to-hero.html.
- [karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) ✅ — companion notebooks (~22k stars).
- [karpathy/micrograd](https://github.com/karpathy/micrograd) ✅ — the scalar autograd Module 1 mirrors.
- [karpathy/makemore](https://github.com/karpathy/makemore) ✅ — bigram → MLP → WaveNet → Transformer LM lineage; mirrors Modules 2–3.
- [karpathy/minbpe](https://github.com/karpathy/minbpe) ✅ — Module 3's BPE reference target.
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) ✅ — Phase 0 / Phase 1 reference repo.
- [karpathy/nanochat](https://github.com/karpathy/nanochat) ✅ — *"ChatGPT for $100"*, the natural successor to nanoGPT and what Karpathy actually iterates on now. Phase 1+ target.
- [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) ✅ — Sebastian Raschka's book + actively-maintained code; best paced from-scratch book.
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) ✅ — Rush et al. 2022 refresh; canonical line-by-line "Attention Is All You Need" walkthrough. Open in a side tab while doing Module 4/5.
- [Lilian Weng — Lil'Log](https://lilianweng.github.io/) ✅ — must-reads: ["Why We Think" (May 2025)](https://lilianweng.github.io/posts/2025-05-01-thinking/), ["Reward Hacking in RL" (Nov 2024)](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/).
- [Jay Alammar — Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) ✅ — single best visual companion for Modules 4–5.

---

## Training & scaling (Phase 1, Phase 3, Phase 4)

- [**How To Scale Your Model** — Austin et al., DeepMind 2025](https://jax-ml.github.io/scaling-book/) ✅ — *roofline-first textbook on TPU/LLM scaling. The Phase 4 prereq textbook.* Repo: [jax-ml/scaling-book](https://github.com/jax-ml/scaling-book). **Do every exercise** before Phase 4.
- [Chinchilla — Hoffmann et al. 2022, arXiv:2203.15556](https://arxiv.org/abs/2203.15556) ✅ — compute-optimal N:D ≈ 1:20. Still canonical, now widely seen as a *lower bound* (RL-era models train 100× past Chinchilla per Pope). **Derive this by hand in Phase 3.**
- [Reiner Pope × Dwarkesh — *The math behind how LLMs are trained and served*](https://www.dwarkesh.com/p/reiner-pope) ✅ (2026-04-29) — 2h14m blackboard roofline lecture. **The single best 2 hours for the systems lens.** [Transcript gist](https://gist.github.com/dwarkeshsp/79100f0fdeed69d76241903bb0604dbe).
- [Andrej Karpathy — *The State of GPT*](https://www.youtube.com/watch?v=bZQun8Y4L2A) ✅ — pre-train → SFT → RM → RL pipeline overview; useful framing for Phases 1–2.
- [HuggingFace transformers source](https://github.com/huggingface/transformers) ✅ — read GPT-2 / LLaMA implementations as a reference; especially helpful when debugging non-convergence in Phase 1.

---

## Reasoning / RL (Phase 2)

- [DeepSeek-R1 — arXiv:2501.12948](https://arxiv.org/abs/2501.12948) ✅ — Phase 2 milestone paper. Already in `papers/`.
- [DeepSeekMath / GRPO — arXiv:2402.03300](https://arxiv.org/abs/2402.03300) ✅ — the algorithm paper. **Derive the GRPO objective by hand** in Phase 2.3.
- [huggingface/trl](https://github.com/huggingface/trl) ✅ — production-grade SFT/PPO/DPO/GRPO trainers. The "framework" target after you've built GRPO from scratch.
- [PrimeIntellect-ai/verifiers](https://github.com/PrimeIntellect-ai/verifiers) ✅ — (formerly `willccbb/verifiers`; old URL still redirects) the canonical minimal-GRPO reference repo as of 2026.
- [OpenAI Spinning Up in Deep RL](https://spinningup.openai.com/) ✅ — Achiam; the standard pedagogical RL intro. Use it for Phase 2.1 (REINFORCE → PPO).
- [Schulman et al. — PPO, arXiv:1707.06347](https://arxiv.org/abs/1707.06347) ✅ — historical PPO; understand for the comparison to GRPO.
- [Ouyang et al. — InstructGPT, arXiv:2203.02155](https://arxiv.org/abs/2203.02155) ✅ — RLHF history.

---

## GPU kernels & DSLs (Phase 4)

- [gpu-mode/lectures](https://github.com/gpu-mode/lectures) ✅ + [GPU MODE YouTube](https://www.youtube.com/channel/UCJgIbYl6C5no72a0NUAPcTA) ✅ — *the* modern CUDA/Triton reading group. Required for Phase 4. Also: [gpu-mode/resource-stream](https://github.com/gpu-mode/resource-stream).
- **PMPP** — Hwu, Kirk, El Hajj, *Programming Massively Parallel Processors: A Hands-on Approach*, **4th ed., Morgan Kaufmann 2022, ISBN 978-0323912310**. Canonical CUDA textbook. Phase 4.2's Ch.1–4 reading.
- [srush/GPU-Puzzles](https://github.com/srush/GPU-Puzzles) ✅ — Numba-based GPU puzzles; pedagogical bridge into kernel thinking.
- [srush/LLM-Training-Puzzles](https://github.com/srush/LLM-Training-Puzzles) ✅ — companion puzzle set on the 1000-H100 thought experiments.
- [Triton tutorials (official)](https://triton-lang.org/main/getting-started/tutorials/) ✅ — vector add → fused softmax → matmul → fused attention. Phase 4.6.
- Flash Attention lineage (all ✅):
  - [v1 — arXiv:2205.14135](https://arxiv.org/abs/2205.14135) (Dao 2022)
  - [v2 — arXiv:2307.08691](https://arxiv.org/abs/2307.08691) (Dao 2023)
  - [v3 — arXiv:2407.08608](https://arxiv.org/abs/2407.08608) (Hopper/FP8, 2024)
  - [v4 — arXiv:2603.05451](https://arxiv.org/abs/2603.05451) (Mar 2026, **B200 in CuTe-DSL**) — the current frontier; the FA4 paper is the best applied CuTe-DSL tutorial right now.
- [HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens) ✅ — TK 2.0 (Jan 2026) adds Blackwell + MXFP8/NVFP4. The cleanest CUDA-as-a-DSL.
- [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) ✅ — CUTLASS + CuTe-DSL home.
- Daily-kernel example repos: [hkproj/100-days-of-gpu](https://github.com/hkproj/100-days-of-gpu) ✅ and [a-hamdi/GPU](https://github.com/a-hamdi/GPU) ✅ — pick one as a template for your own kernel log.

---

## Quantization (Phase 5)

- [LLM.int8() — Dettmers et al., arXiv:2208.07339](https://arxiv.org/abs/2208.07339) ✅ — outlier-aware INT8; the clean bag of tricks. Phase 5.2 reproduction target.
- [QuIP — arXiv:2307.13304](https://arxiv.org/abs/2307.13304) ✅, [QuIP# — arXiv:2402.04396](https://arxiv.org/abs/2402.04396) ✅, [QTIP — arXiv:2406.11235](https://arxiv.org/abs/2406.11235) ✅ — Cornell-RelaxML lineage; ≤4-bit SOTA. Repo: [Cornell-RelaxML/quip-sharp](https://github.com/Cornell-RelaxML/quip-sharp).
- [AQLM — arXiv:2401.06118](https://arxiv.org/abs/2401.06118) + [Vahe1994/AQLM](https://github.com/Vahe1994/AQLM) ✅ — additive quantization, 2–3 bit Pareto-optimal.
- [PV-Tuning — arXiv:2405.14852](https://arxiv.org/abs/2405.14852) ✅ — bundles with AQLM; the post-quantization fine-tuning recipe.
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) ✅ — production INT8 / 4-bit kernels (Dettmers' lab's library).

---

## Inference & serving (Phase 6)

- [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) ✅ — ~1.2k LoC vLLM clone with paged-attn + prefix cache + tensor parallel. **The reference "mini-vLLM"** for Phase 6.5.
- [changjonathanc/flex-nano-vllm](https://github.com/changjonathanc/flex-nano-vllm) ✅ — companion to the writeup below; pure FlexAttention, no flash-attn dep. Easier to read.
- [vLLM / PagedAttention — Kwon et al., arXiv:2309.06180](https://arxiv.org/abs/2309.06180) ✅ — the paper behind it.
- ["vLLM from scratch with FlexAttention" — Jonathan Chang, Aug 2025](https://jonathanc.net/blog/vllm-flex-attention-from-scratch) ✅ — *the* tutorial writeup.
- [PyTorch — FlexAttention-for-Inference (Apr 2025)](https://pytorch.org/blog/flexattention-for-inference/) ✅ — pair with Chang's writeup.
- [SnapKV — arXiv:2404.14469](https://arxiv.org/abs/2404.14469) ✅ — observation-window KV compression, ~3.6× decode speedup. Phase 6.6.
- [vllm-project/vllm](https://github.com/vllm-project/vllm) ✅ — the production engine; read after writing your own.

---

## JAX edge (Phases 3.J, 4.7, 4.8, 5.5)

- [JAX docs](https://docs.jax.dev/) ✅ — *note: `jax.readthedocs.io` now redirects to `docs.jax.dev`; update bookmarks.*
- [Flax (NNX) docs](https://flax.readthedocs.io/) ✅ — NNX is the current API (replaces Linen for new code).
- [Optax docs](https://optax.readthedocs.io/) ✅ — JAX optimizer library.
- [Pallas docs](https://docs.jax.dev/en/latest/pallas/index.html) ✅ — JAX kernel language; has Mosaic GPU (Hopper+) and TPU backends. [Quickstart](https://docs.jax.dev/en/latest/pallas/quickstart.html). **Phase 4.7, Phase 5.5 home base.**
- [JAX AI Stack tutorials](https://docs.jaxstack.ai/) ✅ — newer one-stop tutorial set worth knowing about.

---

## Agents — track 2 (Phase 7)

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) ✅ (Mar 2026) — 630-line nanochat-as-RL-env; agents found ~20 additive training improvements that transferred zero-shot to larger models. **The reference "lab while you sleep" recipe.**
- [AlphaEvolve — arXiv:2506.13131](https://arxiv.org/abs/2506.13131) ✅ + [google-deepmind/alphaevolve_results](https://github.com/google-deepmind/alphaevolve_results) ✅ — evolutionary-coding-agent lineage.
- [FunSearch — Nature 2023](https://www.nature.com/articles/s41586-023-06924-6) ✅ — the predecessor to AlphaEvolve.
- [**Barbarians at the Gate** — arXiv:2510.06189](https://arxiv.org/abs/2510.06189) ✅ — Berkeley, Oct 2025. ADRS framework; up to 5× runtime / 50% cost wins from LLM agents in systems-research workflows. **Phase 7's write-up template.**

---

## Mechanistic interpretability (optional, mind-expanding)

- [Anthropic Transformer Circuits Thread](https://transformer-circuits.pub/) ✅ — the corpus.
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) ✅ — foundational.
- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) ✅ — the polysemantic-neurons paper.

---

## Compute (for when you need to actually run things)

- **Free:** Google Colab (T4 / TPU v2-8 free tier) — Phase 0 capstone, Phase 1.1, Exercise A.
- **Free:** Kaggle (P100, 30 hrs/week).
- **Cheap (~$1/hr):** Lambda Labs (A100), RunPod, vast.ai. Phase 1.2+ as needed.
- **TPU:** Google TPU Research Cloud — free TPU access for research/learning, application-based.

---

## Resources I considered but did not include

- *Generic "Top X LLM resources" blog lists.* Too shallow.
- *Free deep-learning courses (fast.ai, Andrew Ng's specialization).* Excellent in their own right but pre-Phase-0; if any of Phase 0 is hard, take a step back to these instead of forward.
- *Anthropic / OpenAI tutorials on prompting.* Not relevant to the from-scratch arc until Phase 7.

---

## Maintenance

- Last verified: **2026-05-23**.
- Re-verify every 3 months or when a link looks suspicious.
- When a new paper / repo lands that materially changes a phase, add it here *and* note in `MENTOR_LOG.md`.
