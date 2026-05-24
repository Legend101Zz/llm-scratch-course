# COURSE_MAP.md — the 0 → frontier-lab arc

> **The spine of the journey.** Replaces the old `SCHEDULE.md` (which framed this as an 8-hour sprint). This is a multi-month deep-evidence path, not a tour.
>
> Read alongside: [DECISIONS.md](DECISIONS.md) · [CONVENTIONS.md](CONVENTIONS.md) · [RESOURCES.md](RESOURCES.md) · [MENTOR.md](../MENTOR.md) Part C · [frontier-lab.md](../frontier-lab.md).
>
> **Status of each phase:** ⬜ not started · 🟨 in progress · ✅ complete.
> Per-module status lives in [PROGRESS.md](PROGRESS.md). This file shows the *shape* of the arc.

---

## North star (don't lose this)

The endpoint is **frontier-lab readiness** as defined in `frontier-lab.md`: mathematical maturity + a public artifact that proves a specific skill the lab needs, lived out at the **edges** of the stack (kernels below, agent loops above). The capstone signals are concrete: a 10M-param JAX transformer doing digit addition on a TPU you derived Chinchilla laws for, a Pallas kernel that beats `ragged_dot`, and a public repo with handwritten derivations and recorded build sessions backing it all up.

Everything in Phases 0–8 below routes toward that endpoint.

---

## Phase progression at a glance

| # | Phase | Goal | Output | Status |
|---|---|---|---|---|
| 0 | From-scratch core | Build the transformer end-to-end *by hand*, parity-tested against PyTorch | Tiny GPT trained on a toy task, every component derived + tested | 🟨 |
| 1 | Real training & fine-tuning | Train a real-sized model on real data, debug a non-converging run | GPT-2-small-ish trained on Colab, LoRA-fine-tuned, all loss curves committed | ⬜ |
| 2 | Reasoning & RL | Implement GRPO from scratch; reward shape a small model | Toy GRPO → real GRPO on GSM8K-tier task, with measured improvement | ⬜ |
| 3 | Literature fluency | Read + reconstruct the canon | Per-paper write-up + reconstruction (code or hand-derived figure); Chinchilla derived by hand | ⬜ |
| 4 | Kernels (below the stack) | Earn the FlashAttention insight by hitting the bottleneck yourself | CUDA → Triton → Pallas path; **Feinberg Exercise A** capstone (10M JAX/TPU adder + hand Chinchilla) | ⬜ |
| 5 | Quantization | Walk the quality↔perf tradeoff | INT8 path + LLM.int8() reproduction + **Feinberg Exercise B** capstone (Pallas `ragged_dot` beater) | ⬜ |
| 6 | Inference engine (above the stack pt 1) | Build a nano-vLLM from scratch | KV cache → paged attention → continuous batching → benchmark vs HF | ⬜ |
| 7 | Agents (above the stack pt 2) | Rigorous, measured agent experiments | Hypothesis-driven experiment with metric, write-up in the style of the Berkeley ADRS paper | ⬜ |
| 8 | Capstone | Public artifact + recorded evidence | A repo + a demo + the videos `frontier-lab.md` asks for | ⬜ |

---

# Phase 0 — From-scratch core 🟨

**Goal.** Build every component of a transformer LM by hand, derive its math by hand, parity-test against PyTorch, then assemble them into a tiny working GPT on a toy task. This phase is the foundation: every later phase assumes you can re-derive backprop, attention, and the transformer block cold.

**Prerequisite check.** You can write down the cross-entropy loss without looking; you know what a gradient is; you've heard of softmax. (If any is shaky → 3Blue1Brown's NN series, then return.)

**Phase 0 capstone.** A new module `07_phase0_capstone/` (to be created at end of this phase) that:
- Imports the NumPy `Value` engine, attention, and block from Modules 1, 4, 5.
- Stacks N=4 blocks, embeds 65-char vocab, runs forward on tinyshakespeare.
- Loss drops on a 200-step training run (NumPy autograd — slow but real).
- A second cell rebuilds the same model in PyTorch and verifies forward+backward parity.
- `evidence/loss_curve.png` and `evidence/parity_check.txt`.

| Mod | Title | Status | What it delivers | `hand_math/` scaffolded | Parity test |
|---|---|---|---|---|---|
| 01 | Autograd engine | ✅ scaffolded; user runs tests + writes hand-math | Scalar `Value` class + `backward()` over chain rule | ✅ (waiting on user derivation: ∂L/∂w through `+`, `*`, `tanh`) | ✅ exists |
| 02 | Neural net on autograd | ✅ scaffolded; user runs tests + writes hand-math | Neuron → Layer → MLP; XOR train loop | ✅ (waiting on user: MSE gradient + finite-diff check) | ✅ added: forward + per-param-gradient + training convergence vs PyTorch |
| 03 | Tokenization + BPE | ✅ scaffolded; user fills starter + runs tests | Char vocab, bigram count, **BPE from scratch** (starter+solution+test) | ✅ (waiting on user: MLE → empirical counts proof; info-theoretic argument for "merge most-frequent") | ✅ added: round-trip + parity with ref + compression + structural vs tiktoken |
| 04 | Self-attention | ✅ scaffolded; **roofline section added** | Q/K/V from scratch in NumPy, causal mask, multi-head | ✅ (waiting on user: 1/√d_k variance derivation + mask-order counterexample) | ✅ added: softmax + single-head causal + uncausal + multi-head + causal-leak check |
| 05 | Transformer block | ✅ scaffolded; user runs tests + writes hand-math | LayerNorm, residuals, FFN, GELU, full block forward in NumPy | ✅ (waiting on user: LayerNorm backward + pre-norm-vs-post-norm gradient argument) | ✅ added: gelu + LN + FFN + block forward + residual-path-zero check |
| 06 | PyTorch crash | ✅ scaffolded; **bridge drill added** | Tensors, autograd, `nn.Module`, optim/data/GPU; **11** tensor drills (drill 11 = rebuild Module 4 in PyTorch + parity) | N/A (no derivation required for reference module) | ✅ inline drill assertions (incl. drill 11 numpy↔torch parity to 1e-5) |
| 07 | **Phase 0 capstone** | ✅ scaffolded; user runs tests + trains | Tiny GPT (2 blocks × 4 heads × d=64) assembled from Modules 4/5; numpy forward + torch train; loss curve on tinyshakespeare | ✅ (waiting on user: causal-mask induction proof) | ✅ added: param-count + numpy-vs-torch forward to 1e-4 + full-stack causal invariant |

**Phase 0 exits when:** all 7 modules satisfy CONVENTIONS.md "How to know a module is done" criteria. Specifically the user must, in their venv:
1. Run each module's `test.py` and capture output to `evidence/test_output.txt`.
2. Run Module 7's `python train.py` and confirm loss drops below 2.0 nats by step 2000.
3. Produce at least one `hand_math/` derivation per module (photo or transcription).
4. Pass the cold quiz on attention + transformer block + the capstone's data flow.

After all four, mentor promotes to Phase 1.

---

# Phase 1 — Real training & fine-tuning ⬜

**Goal.** Take the Module 7 PyTorch GPT and train it on a real-sized dataset on Colab, then fine-tune it. You must be able to *debug a run that won't converge*. This phase is where pure ML practice meets reality.

**Frame.** Phase 0 proves you can build it. Phase 1 proves you can *run* it — which is a different skill.

**New idea introduced.** Training hygiene: warmup + cosine LR schedule, gradient clipping, mixed precision (bf16), AdamW with weight-decay param groups, gradient accumulation for effective batch size, evaluation at fixed intervals, checkpoint + resume, deterministic seeding.

| Mod | Title | What it delivers |
|---|---|---|
| 1.1 | nanoGPT-style pretrain on tinyshakespeare | Loss curve → 1.4 nats; readable generations; first profiling pass |
| 1.2 | Scale up — pretrain on OpenWebText subset | A ~30M-param model on a T4 in a day; perplexity logged; learning-rate ablation |
| 1.3 | Debugging non-convergence (deliberately broken runs) | Mentor breaks the run 3 ways (bad LR, missing layernorm, masking bug). You diagnose each; write `evidence/debug_log.md` |
| 1.4 | Fine-tuning + LoRA | LoRA implemented from scratch (not via `peft`); ~1% trainable params; eval improvement on a target style |
| 1.5 | Inference quality probes | Perplexity, top-k sample quality, simple eval set (e.g. tinyshakespeare held-out) |

**Capstone.** A trained-on-Colab GPT (your weights, on HuggingFace or in `evidence/`) + a LoRA adapter that demonstrably changes behavior + the debugging log.

**Phase 1 exits when:** you can take a fresh broken training run, diagnose the bug from loss-curve / gradient-norm / param-norm signals alone, in under 15 minutes.

---

# Phase 2 — Reasoning & RL ⬜

**Goal.** Implement GRPO **from scratch**, then on a real (small) base model, on a verifiable task. Walk the bridge from supervised fine-tuning to RL.

**Frame.** This is where modern reasoning lives. DeepSeek-R1 is the milestone paper; GRPO is the algorithm. PPO/InstructGPT are the historical predecessors worth understanding for the comparison.

**New idea introduced.** Policy gradient mechanics: rollout buffers, advantage normalization (the *Relative* in GRPO), reward shaping, the KL-to-reference term, why off-policy RL is unstable. Also: rule-based rewards (math correctness, code-tests-pass) vs human-preference rewards (RLHF).

| Mod | Title | What it delivers |
|---|---|---|
| 2.1 | Policy gradient primer | REINFORCE on a toy bandit, then on a CartPole-tier task (via `spinningup`) |
| 2.2 | PPO from scratch | Clipped-objective PPO on the same toy; understand the value-function role; ablate the clip |
| 2.3 | GRPO derivation + toy impl | Derive the GRPO objective by hand (advantage normalization, KL term). Toy GRPO on the Module 1.4 GPT with a synthetic reward |
| 2.4 | Real GRPO on a small base | Qwen-0.5B or similar, RL on GSM8K-style math; measured accuracy improvement |
| 2.5 | Reward shaping deep dive | Deliberately bad rewards → observe reward hacking; fix it; write up the failure modes |

**Capstone.** A small reasoning model with measurable accuracy improvement on a verifiable task, plus a written failure-mode catalog of the reward shapes that didn't work.

**Phase 2 exits when:** you can derive the GRPO objective cold and explain why the advantage normalization is what makes the *Relative* version more stable than vanilla policy gradient.

---

# Phase 3 — Literature fluency + JAX primer ⬜

**Goal.** Read the canon. Reconstruct the key result of each paper in code or in a hand-derived figure. End with a JAX primer that prepares Phases 4–5.

**Frame.** Up to this point, you've been building. Now you internalize the field's vocabulary so you can converse with researchers efficiently. You can't only show edge contributions; foundational fluency is a baseline. (Per `frontier-lab.md` "FOUNDATION".)

**Reading list — required, with reconstruction:**

| Paper | Reconstruction task |
|---|---|
| Attention Is All You Need (Vaswani 2017) | (Already done in Phase 0.) Re-read, now annotate with your Module 4/5 in the other tab |
| GPT-2 (Radford 2019) | Annotate; identify the 3 things that differ from your Module 7 |
| Chinchilla (Hoffmann 2022) | **Derive N:D ≈ 1:20 by hand from the parametric loss.** Commit photo + transcript |
| LLM.int8() (Dettmers 2022) | Reproduce the outlier-channel histogram on a small model |
| FlashAttention v1 (Dao 2022) | Read; identify the *one sentence* that motivates the whole thing (memory bandwidth). Implement standard attention + measure HBM traffic with profiler |
| DeepSeek-R1 (2025) | (Already covered in Phase 2.) Re-read; identify the 2 algorithmic ideas vs vanilla GRPO |
| LoRA (Hu 2021) | (Already done in Phase 1.) Re-read; understand the low-rank-decomposition argument |
| How to Scale Your Model (DeepMind 2025) | Do **every exercise** in this web textbook. This is the prereq for Phase 4 |

**JAX primer (the bridge for Phases 4–5):**

| Mod | What it delivers |
|---|---|
| 3.J1 | JAX basics — pure functions, `jit`, `grad`, `vmap`, `pmap`, sharding |
| 3.J2 | Flax (NNX) — model definition, parameter handling, training loop equivalents |
| 3.J3 | Optax — optimizers, schedules, gradient clipping, composability |
| 3.J4 | PyTorch ↔ JAX translation — rebuild Module 7 GPT in JAX, verify forward parity to 1e-4 |

**Capstone.** A `papers/READING_LOG.md` with one-page per-paper write-ups (your words, not summaries — what *clicked* and what stayed confusing) + the Chinchilla derivation in `hand_math/` + a JAX implementation of Module 7 that matches PyTorch forward.

**Phase 3 exits when:** you can explain the Chinchilla result from the parametric loss without looking, and you can sketch Q/K/V in JAX without consulting docs.

---

# Phase 4 — Kernels: below the stack ⬜

**Goal.** Earn the FlashAttention insight by *hitting* the memory bottleneck yourself, then write fused kernels that fix it. Path: roofline reflex → CUDA-C basics → Triton → Pallas. End with Feinberg's Exercise A.

**Frame.** This is the central frontier-lab track. The math is simple at the surface — a little algebra tells you whether you're bottlenecked on communication, FLOPs, or memory bandwidth. The hard part is systems thinking + lateral reasoning + identifying unmodelled constraints. Coding agents won't out-engineer you on a well-posed kernel question, but they won't *notice* the constraint unless told.

**New idea introduced.** GPU mental model: threads, warps, blocks, SMs, memory hierarchy (registers → SRAM → L2 → HBM), bandwidth roofline, tiling, fusion. Then the DSL escalator: CUDA-C (closest to the metal) → Triton (block-level pythonic) → Pallas (JAX-native kernels).

| Mod | Title | What it delivers |
|---|---|---|
| 4.1 | Roofline reflex | Reiner Pope lecture digested; do "How to Scale Your Model" exercises 1–N; you can roofline any op in your sleep |
| 4.2 | GPU mental model | Threads/warps/blocks/SMs from PMPP Ch.1–4; memory hierarchy diagram; first vector-add kernel in CUDA-C |
| 4.3 | Matmul → tiled matmul | Naive matmul, then tiled with shared memory. Measure speedup. Roofline both |
| 4.4 | Softmax + LayerNorm/RMSNorm kernels | Online softmax (the FlashAttention prerequisite); fused LN/RMSNorm |
| 4.5 | Naive attention → fused attention | Naive impl, **profile to find the HBM bottleneck**, then write fused attention from scratch (online softmax + tiling). Measure speedup vs HF reference |
| 4.6 | Triton tutorials walkthrough | Vector add → softmax → matmul → fused attention in Triton |
| 4.7 | Pallas tutorials walkthrough | Same ops in Pallas; understand the JAX-native abstraction |
| 4.8 | **Feinberg Exercise A — JAX/TPU adder** | 10M-param transformer in `jax + flax + optax` on free Colab TPU; hard-coded vocab (digits + `+` + `=`); trains to do up-to-3-digit addition; **Chinchilla scaling laws derived by hand for dense and MoE; documented in `hand_math/`**; screen-recorded build |

**Capstone (= 4.8).** Exercise A from `frontier-lab.md`. This is the first artifact Feinberg explicitly grades on.

**Phase 4 exits when:** you can do a full roofline analysis for any op cold; your fused attention kernel beats naive by ≥3× on a T4; Exercise A is committed with `hand_math/chinchilla_dense_moe.md` + a video recording link.

---

# Phase 5 — Quantization ⬜

**Goal.** Walk the quality↔performance tradeoff hands-on. Reproduce the LLM.int8() outlier insight. Then Feinberg's Exercise B.

**Frame.** Quantization is the second canonical edge-of-stack skill — minimal resource demand, full exposure to the quality↔performance tradeoff. The De Sa group lineage (QuIP → QuIP# → QTIP) is current state of the art for ≤4-bit; AQLM is the alternate branch.

**New idea introduced.** Outlier-aware quantization. Why a few large activations dominate the quantization error. The lattice/codebook-based methods that fix it. PV-Tuning.

| Mod | Title | What it delivers |
|---|---|---|
| 5.1 | INT8 from scratch | Quantize Module 1.2's pretrained GPT to INT8; measure perplexity degradation; first feel for the tradeoff |
| 5.2 | LLM.int8() outlier reproduction | Reproduce the outlier-channel histogram; mixed-precision recipe |
| 5.3 | QuIP / QuIP# / QTIP reading + 2-bit reproduction | Reproduce QuIP# 2-bit on a small model; understand the LDLQ + RHT decomposition |
| 5.4 | AQLM + PV-Tuning reading | Side branch: additive quantization. Reproduce on the same small model. Compare |
| 5.5 | **Feinberg Exercise B — Pallas `ragged_dot` beater** | Pallas kernel that beats `jax.lax.ragged_dot` for `F > D` by fusing up/down projections. Measured forward-pass speedup with a written explanation of *why* it's there |

**Capstone (= 5.5).** Exercise B from `frontier-lab.md`. The second artifact Feinberg grades on.

**Phase 5 exits when:** Exercise B has a measured speedup committed in `evidence/` + you can explain the cause-of-speedup in one paragraph.

---

# Phase 6 — Inference engine: above the stack pt 1 ⬜

**Goal.** Build a nano-vLLM. KV cache → paged attention → continuous batching → tiny serving loop.

**Frame.** This is where "vLLM makes sense." You've built the model; now build the *engine that runs it efficiently*. SnapKV is the natural extension (KV cache compression).

**New idea introduced.** Inference is a different beast from training: prefill vs decode phases have different roofline behavior (prefill is compute-bound, decode is bandwidth-bound). KV cache is the central data structure. Paged attention is "virtual memory for KV blocks." Continuous batching is "schedule new requests into the same forward pass as ongoing ones."

| Mod | Title | What it delivers |
|---|---|---|
| 6.1 | KV cache from scratch | Add KV cache to Module 7 GPT; measure tokens/sec speedup |
| 6.2 | Prefill vs decode | Profile both phases separately; roofline each |
| 6.3 | Paged attention | Implement block-allocated KV; reference `vllm-from-scratch` style |
| 6.4 | Continuous batching | Mini scheduler; multiple requests in one forward |
| 6.5 | nano-vLLM reproduction | Get to ~1k-line vLLM-like serving loop; validate against HuggingFace reference |
| 6.6 | SnapKV reproduction | KV compression on top of the cache; measure decode speedup |

**Capstone.** A small serving loop (`evidence/`) that runs your Phase 1.2 model at meaningfully better throughput than naive `model.generate()`, profiled and explained.

**Phase 6 exits when:** you can sketch the prefill/decode/paged-attention flow on a whiteboard cold + your nano-vLLM matches HF outputs to 1e-4 for greedy decoding.

---

# Phase 7 — Agents: above the stack pt 2 ⬜

**Goal.** Rigorous, controlled, measured agent experiments. Not "use Claude" — set up a hypothesis, define a metric, run a measured experiment, write it up.

**Frame.** Per `frontier-lab.md`: "It's setting up rigorous, controlled, technical experiments that measure how single or multiple LLM agents behave." The field doesn't have a clean path yet. Reference points: Karpathy's `autoresearch` (LLM agents finding 20 additive training tweaks for nanochat that transferred to larger models), AlphaEvolve + FunSearch (LLM-in-the-inner-loop of algorithm development), "Barbarians at the Gate" (LLM agents in systems research — up to 5× runtime wins).

**New idea introduced.** Agent loop discipline: experiment hypothesis up-front, controlled baselines, metric pre-registered, statistical significance, ablations. Treat the LLM as a measurement instrument with its own noise floor.

| Mod | Title | What it delivers |
|---|---|---|
| 7.1 | Reading the three lineages | Karpathy autoresearch, AlphaEvolve/FunSearch, Berkeley ADRS. Identify which framing fits your interest |
| 7.2 | Tooling foundation | Pick a harness: claude-agent-sdk / openai-agents / your own. Build a measured ping-pong with a baseline |
| 7.3 | Hypothesis-driven experiment design | Write the experiment doc *before* running anything: hypothesis, metric, baseline, ablation set |
| 7.4 | The experiment | Run it. Multiple seeds. Statistical significance. Plot. |
| 7.5 | Write-up + null result discipline | Even if it didn't work, write the result honestly. ADRS-paper-style write-up |

**Capstone.** A measured agent experiment with a clear hypothesis, baseline, metric, multiple seeds, and a write-up. The write-up matters as much as the result.

**Phase 7 exits when:** the experiment is run + written up + committed, *whether or not* the hypothesis held. (Hard Rule #6: honest, not flattering.)

---

# Phase 8 — Capstone & signaling artifact ⬜

**Goal.** Consolidate everything into a public artifact that demonstrates something real and reusable. Plus the recorded evidence `frontier-lab.md` asks for.

**Frame.** The deliverable is **proof, not a certificate**.

| Component | Requirement |
|---|---|
| Public repo | Cleaned, documented, README rewritten for an external audience |
| The signaling artifact | One of: a kernel that beats a published baseline, an inference optimization with measured wins, an agent experiment with a novel result, a quantization recipe with quality data |
| Handwritten derivations | Scaling-book exercises (all), Chinchilla, FlashAttention sketch |
| Build recordings | Screen-records of you building the transformer / kernels / agent experiment |
| The narrative | `JOURNEY.md` reads as a coherent story from "from-scratch autograd" to "edge-of-the-stack contribution" |

**Phase 8 exits when:** the artifact exists, you've reached out to one or more frontier labs (Feinberg explicitly offers an evaluation), and the choice of where to work becomes yours.

---

## Realistic expectations (from `frontier-lab.md`, in case I forget)

> Doing these exercises is a strong **start**, not a shortcut past the years of signaling a traditional path provides. The realistic payoff is real skills + a public repo that demonstrates something useful and adopted — after which the choice of where to work becomes mine.

Believe this. The course is long because the goal is long. Each phase has a concrete completion criterion so progress is verifiable, but the overall arc is years-not-months in *full* depth. The 2–3 month figure in MENTOR.md is the *intensive build* — Phase 8 (artifact + reach-out + landing) is on top of that.

---

## What the mentor does at session start (per `MENTOR.md` Parts D, E)

1. Reads `PROGRESS.md`, `SKILLS.md`, `REVIEW.md`, `MENTOR_LOG.md`, `DECISIONS.md`, `RESOURCES.md`, `JOURNEY.md` in full.
2. Tells you exactly where we left off.
3. **Quizzes you cold** on any `REVIEW.md` items due today + the central concept of the prior session.
4. Asks how much time you have, scopes the session to **one concrete completable deliverable**.
5. Starts work.

## What the mentor does at session end

1. Updates all memory files.
2. Writes the devlog entry in your voice (Part F).
3. Adds at least one new spaced-repetition Q to `REVIEW.md`.
4. Commits + pushes.
5. States the single most important thing to remember from this session.

---

## Next concrete step (current state, 2026-05-23 end of day)

Phase 0 is **fully scaffolded** — every module has its prose + starter + solution + parity test (where applicable) + `hand_math/` and `evidence/` READMEs. The Phase 0 capstone (`07_phase0_capstone/`) exists, has been verified to run forward in NumPy, and is ready for the user to run `test.py` + `train.py` in their venv.

**The user's next steps (autonomous):**
1. Install `requirements.txt` in a venv and run each `test.py` to capture `evidence/test_output.txt`.
2. Run `07_phase0_capstone/train.py` to produce loss curve + samples.
3. Write at least one `hand_math/` derivation per module.
4. Re-issue the cold quiz with the mentor (paused at end of session 02).

After all four → Phase 0 closes, Phase 1 begins.
