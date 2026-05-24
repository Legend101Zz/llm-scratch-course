# DECISIONS.md — strategic & architectural calls

> Append-only log of decisions that shape the curriculum.
> Each entry: **date · decision · why · revisit when**. If a later decision overrides an earlier one, the earlier one stays for the audit trail with a `STATUS: superseded by D-NNNN`.
> Cross-references: [MENTOR.md](../MENTOR.md) · [frontier-lab.md](../frontier-lab.md) · [COURSE_MAP.md](COURSE_MAP.md) · [CONVENTIONS.md](CONVENTIONS.md).

---

## D-0001 · 2026-05-23 · Restructure (option (b) hybrid lean), not sprint-finish

**Decision.** Reframe the existing course from an *8-hour sprint* into a *multi-month phased journey* toward frontier-lab readiness. Keep the prose and code already written for Modules 1–6 (it's genuinely strong), but drop the sprint framing across `README.md` / `SCHEDULE.md` / `whats_next.md`, replace it with a phased arc (Phases 0–8 per MENTOR.md Part C), and impose course-wide discipline conventions on every module going forward.

**Why.**
- The sprint framing (timed modules, "blow past 1.5× → peek at solution") rewards touching every concept once. The 2–3 month frontier-lab goal rewards *deep evidence* per concept — handwritten derivations, parity tests, roofline analysis, ablations. The two framings actively work against each other.
- Modules 7–10 in the current tree are skeletal anyway; they need rebuilding regardless.
- The Module 1 PyTorch-parity test is the single best pedagogical move in the course. Generalizing it (parity tests on Modules 3/4/5, plus an evidence trail per module) raises the floor without re-writing what already works.
- Rejected option (a) "finish & extend" because the sprint framing is *baked into* the public README, schedule, and module READMEs — leaving it in place steers later phases shallow.
- Rejected option (c) "full rebuild" because Modules 1, 4, 5, 6 prose is the strongest pedagogy in the repo and re-writing would waste a week.

**Concretely changes:**
1. Drop sprint language from `README.md`, `SCHEDULE.md`, `whats_next.md`. Keep `SCHEDULE.md` as a legacy artifact (renamed in a future commit if useful).
2. New top-level files at the course root: `COURSE_MAP.md` (the phased spine), `CONVENTIONS.md` (the discipline patterns), `PROGRESS.md`, `SKILLS.md`, `REVIEW.md`, `MENTOR_LOG.md`, `DECISIONS.md`, `RESOURCES.md`, `JOURNEY.md`, and a `devlog/` directory.
3. Modules 1–6 stay where they are. Each is patched in place to apply the new conventions (`hand_math/`, `evidence/`, parity test, roofline section where applicable).
4. Modules 7–10 will be redesigned phase by phase as we reach them, using the COURSE_MAP arc — not scaffolded all at once. Per-phase design lets each module benefit from what we learned in the prior phase.

**Revisit when.** End of Phase 0 (when Modules 1–6 are patched and Phase 0 capstone exists). Confirm the patched modules actually feel deeper, not just decorated.

---

## D-0002 · 2026-05-23 · PyTorch through Phase 3, JAX/Pallas crossover at Phases 4–5

**Decision.** The framework strategy:

- **Phases 0–3** (from-scratch core → real training → reasoning/RL → literature) live in **PyTorch** (with the from-scratch-first NumPy stage preserved for Phases 0).
- **Phases 4–5** (GPU kernels + scaling-laws derivation + quantization) include a dedicated **JAX / Flax / Optax / Pallas** crossover, scoped specifically to satisfy Vlad Feinberg's graded Exercises A and B from `frontier-lab.md`:
  - **Exercise A**: ~10M-param transformer in `jax + flax + optax` on free Colab TPU, learning digit addition, with Chinchilla scaling derived by hand (dense vs MoE).
  - **Exercise B**: a Pallas kernel that beats `jax.lax.ragged_dot` for `F > D` by fusing up/down projections, with measured forward-pass speedup and a written explanation of *why* it's faster.
- **Phases 6–8** (inference engine, agents, capstone) stay PyTorch-first unless an artifact specifically benefits from JAX (e.g. a Pallas inference kernel).

**Why.**
- PyTorch is unambiguously the best framework to *learn* in: dynamic graph, Pythonic debugging, the ecosystem most reading material assumes. Switching framework cognitive load before mastering the ideas would waste weeks.
- Feinberg grades on JAX/Pallas. The graded exercises *are* the signaling artifact. Skipping JAX means forfeiting the article's offer.
- Compromise (split): learn ideas in PyTorch, then deliberately cross over to JAX once the math + roofline reflex is in muscle memory. Phase 1's training-loop debugging is best learned in PyTorch. Phase 4's kernels are best learned in Triton + Pallas in parallel (CUDA-C → Triton → Pallas, escalating abstraction).
- Rejected "JAX from day one" because Phase 0's numpy-then-PyTorch path is already the best pedagogical entry point and undoing it would waste the strongest existing modules.
- Rejected "PyTorch only, skip JAX" because the entire `frontier-lab.md` evidence chain (Exercise A's TPU run, Exercise B's Pallas kernel) requires JAX.

**Bridge mechanics (concrete):**
- Phase 3 closes with a "JAX primer" module (1 week): jax basics, `jit`/`vmap`/`pmap`/`grad`, `flax.nnx` linen-vs-nnx note, `optax` optimizers, the scaling-book tutorial set. No transformer yet — just framework fluency.
- Phase 4 opens with the roofline chapter from "How to Scale Your Model" + Reiner Pope's lecture + GPU mental model. CUDA-C basics in week 1, Triton in week 2, Pallas in week 3.
- Exercise A is the Phase 4 capstone. Exercise B is the Phase 5 capstone (because the `ragged_dot` fusion is intrinsically a quantization-adjacent memory-bandwidth problem).

**Revisit when.** Entering Phase 3 (decide whether the JAX primer is a separate module or threaded into the literature phase). Also revisit if PyTorch+`torch.compile` evolves to subsume what Pallas does — that would matter for Exercise B.

---

## D-0003 · 2026-05-23 · Course-wide discipline conventions (hand_math, evidence, parity, roofline)

**Decision.** Every module from this session forward — including patches to existing Modules 1–6 — follows the conventions defined in `CONVENTIONS.md`:

1. **`hand_math/`** subfolder per module, containing a photographed or transcribed pencil derivation of the module's central gradient / objective / scaling law.
2. **`evidence/`** subfolder per module, containing the test output, profiling result, or numerical comparison that *proves* the implementation works.
3. **PyTorch-parity test** for any from-scratch numerical module (Modules 1, 2 already; 3, 4, 5 to be added).
4. **Roofline section** in every README for code that runs on hardware (Modules 4 onward).
5. **Reflection questions** stay (already standard in current modules).
6. **No `solution.py` is revealed in chat until the user has genuinely attempted `starter.py`** (Hard Rule #1).

**Why.** Frontier-lab signal is *recorded evidence*, not "I read it." The conventions force the artifacts that prove the work happened. Specifically: handwritten derivations are what Feinberg asks for; parity tests close the "from scratch matches framework" loop; roofline answers "what limits this?" before optimization; evidence/ is what gets linked from the public devlog.

**Revisit when.** Phase 1 starts. By then we'll know if the convention overhead is too heavy or just right; adjust the template before scaling it across 8 phases.

---

## D-0004 · 2026-05-23 · Public repo = build log. Memory files live in the repo.

**Decision.** The `Legend101Zz/llm-scratch-course` repo doubles as the public build log per MENTOR.md Part F. The Part D memory files (`PROGRESS.md`, `SKILLS.md`, `REVIEW.md`, `MENTOR_LOG.md`, `DECISIONS.md`, `RESOURCES.md`, `JOURNEY.md`) all live **inside** the public repo and get committed each session. `MENTOR_LOG.md` is terse and reader-aware but still public — transparent learning is part of the signal.

**Why.** The point of the public artifact is that *someone reading the repo can reconstruct the journey*. Hiding memory files (or putting them in a separate private repo) would undermine that. Public memory also forces the entries to be honest but presentable, which is a better discipline than private rambling.

**Outside the repo (stays private):** `MENTOR.md`, `frontier-lab.md`, `CLAUDE.md` at `/Users/comreton/Desktop/llm-scratch/` are the mentor's operating instructions and should not be in the public artifact. They're the *charter*, not the *journey*.

**Revisit when.** If `MENTOR_LOG.md` gets uncomfortably personal, split into a `MENTOR_LOG_private.md` (gitignored) + a public `MENTOR_LOG.md`.

---

## D-0005 · 2026-05-23 · Module-by-module scaffolding, not batch-scaffolding

**Decision.** Phases 1–8 module directories will be created **one phase at a time** as we reach them, not scaffolded in advance. Today's session only scaffolds Phase 0 (which is the patched existing Modules 1–6 + a new Phase 0 capstone where we assemble them).

**Why.** Stale skeletons rot. A pre-built `08_train_colab/README.md` written today will be a worse plan than one written after Phases 1–2 have happened. Also, batch scaffolding tempts shallow filling-in later; per-phase scaffolding forces fresh design each time.

**Revisit when.** Never. This is a meta-rule about how to manage planning entropy.

---

## D-0006 · 2026-05-23 · Existing Modules 7–10 will be redesigned, not preserved

**Decision.** The current `07_gpt_pytorch/`, `08_train_colab/`, `09_finetune_lora/`, `10_deepseek_r1/` directories are skeletal sketches and will be **rewritten** when we hit their phase, not patched. They stay in the tree for now (with a banner in each README pointing to COURSE_MAP) but should not be relied on.

**Why.** They were written under the sprint framing; they don't have parity tests, hand-math, roofline analysis, or evidence trails. Rewriting under the new conventions is faster than patching.

**Exception:** `extras/` and `papers/` stay as-is. They're reference material, not core modules.

**Revisit when.** When entering each affected phase.
