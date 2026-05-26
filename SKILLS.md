# SKILLS.md — the skill tree (0–5), with proof

> Each skill rated 0 (haven't touched it) → 5 (can reproduce, derive, debug, and explain it cold to a researcher).
> A rating must be backed by a **proof file** — a committed artifact in this repo that demonstrates the level. No rating moves up without a new artifact.
>
> The mentor updates this at session end. The user can challenge any rating.

---

## Rating scale

| Level | Meaning | What proves it |
|---|---|---|
| 0 | No exposure. | (no proof needed) |
| 1 | I've heard of it; I understand the high-level intuition. | Notes / a paragraph in a module README |
| 2 | I can use a framework abstraction for it. | A working call to `torch.nn.X` / `jax.X` in code |
| 3 | I've built a from-scratch version that passes a parity test against the framework. | `test.py` parity assertion in a module |
| 4 | I've derived the math by hand and can re-derive cold; I can debug failures. | `hand_math/` derivation + a debugged-broken-run log in `evidence/` |
| 5 | I've extended / optimized / written a new variant of it that beats a baseline, with measured evidence. | Original artifact (kernel, optimization, paper reproduction) + benchmarks in `evidence/` |

---

## Foundations

| Skill | Level | Proof |
|---|---|---|
| Backprop / scalar autograd | **3** | [`phase0/01_autograd/test.py`](01_autograd/test.py) — parity vs PyTorch on a complex expression. Hand-math derivation pending → level 4. |
| Forward-mode AD vs reverse-mode AD | 1 | Conceptual only. Distinction explained in `phase0/01_autograd/README.md`. |
| Multi-layer perceptron from scratch | **3** | [`phase0/02_neural_net/solution.py`](02_neural_net/solution.py) + [`test.py`](02_neural_net/test.py) — XOR loss → 0, **per-parameter gradients match PyTorch to 1e-5**. Hand-math pending → level 4. |
| Cross-entropy loss derivation | 2 | Stated in `phase0/00_start/README.md`; not yet derived by hand. |
| Numerical stability (log-sum-exp, softmax shift) | 1 | Implicit in `phase0/04_attention_scratch/solution.py`'s `softmax`. Not derived by hand. |
| Initialization (Xavier, Kaiming, GPT-2 std=0.02) | 1 | Discussed in `phase0/06_pytorch_crash/03_nn_module.md`. Not yet ablated experimentally. |

## Tokenization

| Skill | Level | Proof |
|---|---|---|
| Char-level tokenization | **3** | `phase0/03_tokenizer_bigram/solution.py` — encode/decode round-trip, bigram NLL computed. |
| BPE from scratch | **2** | `phase0/03_tokenizer_bigram/bpe.py` exists as reference; not yet refactored to `starter.py` + parity test. Level 3 once tests pass. |
| Comparing to `tiktoken` | 1 | A `print(tiktoken.encode(...))` line in the README. No structured comparison yet. |

## Attention & transformer architecture

| Skill | Level | Proof |
|---|---|---|
| Single-head attention from scratch (NumPy) | **3** | `phase0/04_attention_scratch/solution.py` + `test.py` parity vs PyTorch to 1e-5. |
| Multi-head attention from scratch | **3** | Same; multi-head also parity-tested to 1e-5. **Also: rebuilt in PyTorch as Module 6 drill 11** → confirms numpy↔torch round-trip. |
| Roofline analysis for attention | **2** | Worked T4 calculation committed in `phase0/04_attention_scratch/README.md` (AI=5 vs knee=25 → memory-bound). Level 3 once user commits their own version + alternate-shape calculation in `evidence/roofline.md`. |
| Causal masking | 2 | Implemented; not yet hand-derived (why mask before softmax). |
| Q/K/V split — purpose | 2 | Explained in detail in `phase0/04_attention_scratch/README.md`. Cold-quiz verification pending. |
| 1/√d_k scaling — derivation | 2 | Discussed numerically in README. Not yet hand-derived. |
| Multi-head specialization (heads as circuits) | 1 | Mentioned in README. No experimental verification (haven't visualized attention patterns). |
| LayerNorm / RMSNorm | 2 | LayerNorm impl in `phase0/05_transformer_scratch/solution.py`. RMSNorm not implemented. Backward not derived. |
| Pre-norm vs post-norm | 2 | Discussed in `phase0/05_transformer_scratch/README.md`. Cold-quiz verification pending. |
| Residual stream framing | 2 | Mentioned. No experimental ablation. |
| FFN as key-value memory (Geva et al.) | 1 | Discussed in README. Not reproduced. |
| Full transformer block forward pass | **3** | `phase0/05_transformer_scratch/solution.py` + `test.py` (gelu/LN/FFN/block all parity-tested + residual-path verified). |
| Full GPT (stack of blocks + embeddings + LM head) | **3** | `phase0/07_phase0_capstone/{numpy_gpt.py, torch_gpt.py, test.py}` — 112k-param GPT, numpy↔torch forward parity to 1e-4, full-stack causal invariant verified. Training script committed; loss-curve generation pending (user runs `train.py`). |
| Weight tying (lm_head.weight = token_emb.weight) | 1 | Discussed in `phase0/06_pytorch_crash/03_nn_module.md`. Not yet implemented in capstone (kept separate for cleaner parity proof). |

## PyTorch

| Skill | Level | Proof |
|---|---|---|
| Tensors, dtypes, devices | **2** | Walkthrough in `phase0/06_pytorch_crash/01_tensors.md`. Drills not yet completed by user. |
| `view` / `reshape` / `permute` / `contiguous` | 2 | Same. |
| Broadcasting (incl. attention shape examples) | 2 | Same. |
| `einsum` | 1 | Mentioned. Not drilled. |
| Autograd dynamic graph mental model | 2 | Walkthrough in `phase0/06_pytorch_crash/02_autograd.md`. Drills pending. |
| `requires_grad`, leaf vs intermediate | 2 | Same. |
| `.detach()` / `no_grad` / `inference_mode` | 1 | Discussed; not used in practice yet. |
| Custom `autograd.Function` + STE | 1 | Discussed in `phase0/06_pytorch_crash/02_autograd.md`; not yet implemented. |
| `nn.Module` (Parameter, buffer, train/eval) | 2 | Walkthrough in `03_nn_module.md`. |
| Optimizers (SGD / Adam / AdamW + param groups) | 1 | Discussed in `04_optim_data_gpu.md`. Not used in a real training loop yet. |
| Dataset / DataLoader / collate_fn | 1 | Discussed. Not used. |
| Mixed precision (bf16 / fp16 / GradScaler) | 1 | Discussed. Not used. |
| `torch.compile` | 1 | Discussed. Not used. |
| `torch.profiler` | 1 | Discussed. Not used. |
| `torch.distributed` (DDP / FSDP) | 0 | Not yet covered. |

## Training & evaluation

| Skill | Level | Proof |
|---|---|---|
| Training loop (forward / loss / zero_grad / backward / step) | 2 | Walkthrough in `phase0/06_pytorch_crash/04_optim_data_gpu.md`. Not yet run end-to-end on a real model. |
| Warmup + cosine LR schedule | 1 | Discussed. Not implemented. |
| Gradient clipping | 1 | Discussed. Not implemented. |
| Debugging non-convergence (LR / init / mask bugs) | 0 | Phase 1.3 deliverable. |
| LoRA fine-tuning | 1 | Sketch in `09_finetune_lora/`. Phase 1.4 will rebuild from scratch. |
| Perplexity / eval-set construction | 1 | Discussed in passing. |

## RL & reasoning

| Skill | Level | Proof |
|---|---|---|
| REINFORCE | 0 | Phase 2.1. |
| PPO from scratch | 0 | Phase 2.2. |
| GRPO derivation | 0 | Phase 2.3 hand-derivation deliverable. |
| Reward shaping + failure modes | 0 | Phase 2.5. |

## Literature fluency

| Skill | Level | Proof |
|---|---|---|
| Attention Is All You Need | 2 | Modules 4–5 are partial reconstruction; Phase 3 will close the loop. |
| GPT-2 paper | 1 | Skimmed (per `papers/`). Phase 3 reconstruction. |
| Chinchilla (parametric loss → N:D ratio) | 1 | Discussed in `extras/scaling_laws.md`. **Hand-derivation in Phase 3.5 is the level-4 gate.** |
| FlashAttention v1 | 1 | Discussed in `extras/flash_attention.md`. Phase 3.5 + Phase 4 will close. |
| LLM.int8() | 0 | Phase 5.1. |
| DeepSeek-R1 | 1 | Skimmed (per `papers/`). |
| LoRA paper | 1 | Skimmed; Phase 1.4 reconstruction. |
| How to Scale Your Model (full) | 0 | Phase 3.5 reading + exercises = Phase 4 prereq. |

## Systems / kernels

| Skill | Level | Proof |
|---|---|---|
| Roofline analysis | 0 | **Phase 4.1 reflex deliverable.** |
| GPU mental model (threads/warps/blocks/SMs/memory hierarchy) | 0 | Phase 4.2 (PMPP Ch.1–4). |
| CUDA-C basics | 0 | Phase 4.2. |
| Tiled matmul (shared memory) | 0 | Phase 4.3. |
| Online softmax | 0 | Phase 4.4 (FlashAttention prerequisite). |
| Fused attention kernel | 0 | Phase 4.5. |
| Triton | 0 | Phase 4.6. |
| Pallas | 0 | Phase 4.7. |
| Profiling (torch.profiler / nsys / Pallas bench) | 0 | Threaded through Phase 4. |

## JAX

| Skill | Level | Proof |
|---|---|---|
| JAX basics (jit, grad, vmap, pmap, sharding) | 0 | Phase 3.J1. |
| Flax (NNX) | 0 | Phase 3.J2. |
| Optax | 0 | Phase 3.J3. |
| PyTorch ↔ JAX translation | 0 | Phase 3.J4 (GPT in JAX). |

## Inference

| Skill | Level | Proof |
|---|---|---|
| KV cache | 0 | Phase 6.1. |
| Prefill vs decode | 0 | Phase 6.2. |
| Paged attention | 0 | Phase 6.3. |
| Continuous batching | 0 | Phase 6.4. |
| nano-vLLM reproduction | 0 | Phase 6.5. |
| SnapKV | 0 | Phase 6.6. |

## Quantization

| Skill | Level | Proof |
|---|---|---|
| INT8 quantization | 0 | Phase 5.1. |
| LLM.int8() outlier handling | 0 | Phase 5.2 reproduction. |
| QuIP / QuIP# / QTIP family | 0 | Phase 5.3. |
| AQLM | 0 | Phase 5.4. |

## Agents (Track 2)

| Skill | Level | Proof |
|---|---|---|
| Agent harness / tool-use loop | 0 | Phase 7.2. |
| Hypothesis-driven agent experiment design | 0 | Phase 7.3. |
| Statistical significance for agent metrics | 0 | Phase 7.4. |

---

## Headline summary

**Strongest area:** Phase 0 fundamentals — backprop, attention, transformer block, **and the full integrated GPT** all at level 3 (built from scratch + parity-tested against PyTorch). Once hand-math derivations land in `hand_math/` they climb to 4.

**Biggest gap relative to North Star:** *Everything in Track 1 (kernels) is at 0.* Roofline analysis is **the** most-cited skill in `frontier-lab.md`'s recruiter signal, and we haven't started. Phase 4 is the central work of the curriculum.

**Next level-3 unlock (just happened this session):** BPE from scratch with round-trip test; Modules 2, 4, 5 with parity tests; full GPT (capstone) with numpy↔torch parity.

**Next level-4 unlock (next session):** Hand-math derivations for autograd, attention scaling, LayerNorm in their respective `hand_math/` folders.

**Next level-5 unlock (Phase 4):** A fused attention kernel that beats naive by ≥3× on a T4, with the roofline analysis explaining *why*.
