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

---

## D-0007 · 2026-05-27 · Phase 8 reframed as "build your own PyTorch" (post-learning, not interwoven)

**Decision.** Phase 8 is reframed from the vague "Capstone & signaling artifact" into a concrete 3–4 month framework project: **build your own PyTorch-style ML library from scratch**, end-to-end (autograd → nn modules → optimizers → kernels → inference), train a small (~12M-param) LM on it, publish.

Crucially: the framework is **not** built incrementally as a "deposit" from each prior phase. Phases 0–7 stay pure learning. The framework happens *after* all the learning has been done, in a dedicated Phase 8 project that rebuilds (cleanly, integrated, opinionated) what was already built once (messily, in isolation, for learning) across Phases 1–7.

Phase 8 has 6 stages:
- **8.1 — Core (ML side):** vectorized `Tensor` + autograd, `nn.Module` system, primitives, parity tests.
- **8.2 — Training:** optimizers, schedulers, gradient clipping, mixed precision, checkpoint/resume, train a small transformer.
- **8.3 — Kernels (GPU side):** port Phase 4's CUDA/Triton kernels into the framework as the GPU backend.
- **8.4 — Inference:** port Phase 6's KV cache + paged attention + continuous batching into a `framework.serve` module.
- **8.5 — Train a real model:** ~12M-param LM end-to-end on your framework, evaluated.
- **8.6 — Publish:** README polish, blog post, screen-recording, Twitter. Hook decision (if any) happens *here*, post-build.

**Why.**
- **Learning-first beats artifact-first.** The user explicitly chose to keep Phases 0–7 focused on learning, not on framework infrastructure. Interweaving framework discipline across every phase would slow learning and pull attention toward productization concerns before the underlying ideas are solid.
- **Build twice = much better second time.** Phases 1–7 build each component once for learning (Phase 4 kernels, Phase 6 inference, etc.). Phase 8 builds them *again*, integrated, with the benefit of hindsight — cleaner APIs, fewer dead ends, an opinionated POV. This is the same pattern as Karpathy's `nanoGPT` (the "second pass" after years of building one-off models).
- **Reduces half-finished-artifact risk.** If the framework had to grow phase-by-phase, abandoning the course at Phase 5 would leave a half-built repo as bad signal. With Phase 8 standalone, partial completion just means "skipped the artifact," not "shipped something broken."
- **The hook decision is deferred to Phase 8.6.** No commitment now to a unique angle (specific kernel, novel backend, minimalism play). By Phase 8 the user will know what's interesting about their build and can pick a hook *from evidence*, not speculation. If no hook emerges, the artifact is still "I built a working ML framework end-to-end with proof for every piece" — which is itself signal.

**What this changes in the existing plan:**
- **No new module in Phases 1–7.** They stay exactly as in COURSE_MAP.md.
- **Phase 4 and Phase 6 awareness shift.** Each kernel/inference primitive in those phases is now written knowing "I will rewrite this cleanly in Phase 8." So: write tests carefully, write notes about API mistakes, document gotchas. Phase 4/6 code is the *draft*; Phase 8 code is the *publication*. This actually *speeds up* Phases 4 and 6 because polish isn't required.
- **Phase 8 row in COURSE_MAP rewritten** to show the 6 stages with rough duration estimates.
- **New file `PHASE8_FRAMEWORK.md`** committed now as a forward placeholder. It gets fleshed out incrementally as Phases 4 and 6 surface design ideas; full plan when we reach Phase 8.

**What this implies for total course time.**
The framework adds ~3–4 months on top of Phases 0–7. Total course time becomes ~12–18 months of focused work — which matches Feinberg's "real skills + real artifact" framing. If time has to compress, cut Phase 7 (agents) or shrink Phase 4, not Phase 8.

**Reference projects considered.**
- [xames3/slowtorch](https://github.com/xames3/slowtorch) — PyTorch reimplementation in pure Python. Pedagogical reference.
- [mni-ml/framework](https://github.com/mni-ml/framework) — Rust backend + custom CUDA kernels, trained a 12M-param LLM. The shape of artifact we're aiming for, scaled to our skill profile.

**Revisit when.** End of Phase 4 (the kernels phase). At that point we'll know if the GPU side of Phase 8 is realistically scoped at 3–4 weeks or needs more. Also revisit at end of Phase 6 (inference) for the same reason on `framework.serve`.

---

## D-0008 · 2026-08-05 · Unit of progress = capability claims; dated April 2027 checkpoint; GenHash as a capstone track

**Decision.** Restructure the course around **capability claims** rather than modules, and sequence them so a dated **April 2027 checkpoint** is banked first. Six concrete changes:

1. **Claims replace modules as the unit of progress.** A claim is one falsifiable capability statement ("I can implement KV-cache inference from scratch matching HuggingFace token-for-token"), never a topic ("understand attention"). Its acceptance test is named **up front**. Its timebox is **1–2 weeks** — longer means split. Its terminal event is a **shipped public artifact** (≤1 extra hour, raw, no polish). A claim without its artifact is not done. Phases survive as groupings; claims are the rows.
2. **The north star is unchanged.** Frontier-lab readiness per `frontier-lab.md`; Feinberg Exercises A and B remain graded capstones. GenHash joins them as a third capstone signal.
3. **A dated checkpoint is added beneath the north star** in `COURSE_MAP.md` and `CLAUDE.md`: April 2027, EF / founding-engineer readiness, goals (a)–(d). Frontier-lab-ready is a **superset** of founding-engineer-ready — the checkpoint is a waypoint on the same arc, not a substitute for it.
4. **Phase 3 is dissolved** as a standalone phase. Literature fluency becomes **one paper per week, every week**, matched to the active claim, logged to `papers/READING_LOG.md` in a fixed ≤1-page format. The JAX primer (old 3.J1–J4) becomes ordinary Claims 7–8; *How to Scale Your Model* stays the Phase-4 prerequisite as Claim 9.
5. **GenHash becomes its own claim block** (Claims 17–20) on the CHECKPOINT path.
6. **`CLAUDE.md` is rewritten as the operating contract** — role boundaries, session start/end, enforcement rules, honesty rule.

**Why claims.**
- Modules measure *exposure*; claims measure *capability*. "Finished Module 4" and "can rederive attention cold" are different facts, and only the second one survives an interview.
- Naming the acceptance test up front is what makes a claim falsifiable. A test written afterwards always describes whatever got built, which is how a course drifts while appearing to progress.
- The artifact-as-terminal-event closes the `frontier-lab.md` loop directly: the article's whole thesis is that public artifacts are the signal. Making the artifact a *completion condition* rather than a later phase means the signal accumulates continuously instead of depending on reaching Phase 8.
- The 1–2 week timebox exists because D-0007 already projected 12–18 months. Timeboxes convert that from an estimate into a control.

**Sequencing decisions and reasoning.**

*Budget.* 2026-08-05 → 2027-04-01 is 34.1 weeks. At 12–15 hrs/week that's 410–512 hours. Allocated: **32 claim-weeks + 2 weeks slack** for a single 2-week slip.

*Goal (a)'s ~100M model — reconciled with the existing 12M/30M targets.* Placed as **an extension of the Phase 1 scale-up ladder** (Claim 6, following the ~30M OpenWebText run at Claim 4), **not** as Phase 8's trained model. Reasoning: Phase 8's 12M model exists to prove *the framework works*, not to prove *scale* — it is a different claim, and it sits behind a 3–4 month framework build that would consume the entire checkpoint budget. Phase 1's ladder already terminates in a real trained model on real data, so extending it to ~100M is the cheap path. Phase 8's 12M target is left unchanged and post-checkpoint. "My own implementation" is read as: my architecture and my training loop, with PyTorch as the tensor/autograd backend — not HF `Trainer`. Writing my own autograd for a 100M run is Phase 8's job, not the checkpoint's.

*Feinberg A and B stay as early as prerequisites allow*, because they serve goal (c) and (c) has the longest lead time — an artifact cannot be shared by anyone until it exists. A (Claim 10) lands immediately after the JAX primer and the scaling-book exercises, its true prerequisites. B (Claim 16) lands immediately after kernels-core, since a Pallas fusion kernel is not writable before the roofline reflex and fused attention exist.

*GenHash is sequenced after Phase 1 and after a VQ / perceptual-loss prerequisite* (Claim 17), never before. HiFiC is a GAN-based codec — reproducing it requires working training skills, and benchmarking it requires LPIPS/FID machinery. Attempting it earlier would fail for training reasons that have nothing to do with compression. **Reproduction-first: novelty claims are banned until the Claim 19 benchmark exists.**

*What moved to CONTINUATION, and why each.*
- **Phase 8 (full framework build)** — 3–4 months standalone; its 12M model cannot serve goal (a); pulling it forward eats the budget. Post-checkpoint.
- **Phase 5.3/5.4 (QuIP#/QTIP/AQLM reproductions)** — genuinely SOTA and genuinely not required by (a)–(d). Phase 5.1/5.2 (INT8, LLM.int8()) also deferred, but flagged **first up post-checkpoint** — highest interview value of the remainder.
- **Phase 7 (agents)** — Track 2 of `frontier-lab.md`, valuable, and entirely independent of (a)–(d). Nothing in the checkpoint depends on it.
- **Phase 6 back half** (paged attention, continuous batching, nano-vLLM, SnapKV) — Claim 21 (KV cache + prefill/decode roofline) banks the interview-relevant core cheaply in 1 week; the full serving loop is a multi-week build with no checkpoint dependency.
- **Phase 2 back half** (REINFORCE → PPO → real GRPO on a base model → reward-hacking catalog) — Claim 22 banks the *derivation*, which is what goal (d) actually needs. Real GRPO on a GSM8K-tier task is a compute-heavy multi-week run.

*Goal (c) is deliberately not a claim.* No acceptance test can compel a stranger to share your work, and writing one would be dishonest. It is structured instead as ~23 shots on goal — every claim ships an artifact — with Claims 10, 16 and 20 as the highest-probability ones.

**Honest risks recorded at decision time.**
- **The budget is tight.** Three capstone-grade artifacts plus a 100M scale-up in 32 claim-weeks is aggressive. The compressible block is kernels-core (Claims 11–15); if the schedule slips, that descopes first and Exercise B goes with it. Recorded now so a future descope is a decision, not a surprise.
- **Goals (a) and (b) both require paid compute** — order a few hundred dollars of A100 time. Free Colab covers neither a ~100M run nor a HiFiC reproduction. Goal (a) asks for documented cost, so this is a deliverable rather than a hidden dependency, but the checkpoint fails without it.
- **No GenHash source material existed** anywhere in the repo, on the SSD, or in the user's notes when this decision was made. Claims 17–20 were built from the compression spine specified in the restructure brief. **Open thread:** the *learned-hashing* half of "hybrid learned hashing + generative image compression" has no defined role in the ladder. It was deliberately not invented into claims. Resolve before Claim 17 starts.

**Considered, rejected: meta-work.** Recorded here instead of built, per the no-meta-work rule:
- A claim-tracking CLI / dashboard / status script — considered, rejected: meta-work. `PROGRESS.md` is the tracker.
- A per-claim directory template + scaffolding generator — considered, rejected: meta-work. `CONVENTIONS.md` already defines the module layout.
- A CI job to verify parity tests and artifact links — considered, rejected: meta-work. Running the tests is Claim 1's acceptance test; automating it would substitute for doing it.
- A `READING_LOG.md` entry generator / template filler — considered, rejected: meta-work. The format is 5 lines; typing them is the point.
- Auto-computing "weeks remaining to checkpoint" from a dated file — considered, rejected: meta-work. The mentor states it at session start from the date.

**What this does NOT change.** Parity tests, `hand_math/`, `evidence/`, cold quizzes, the `REVIEW.md` spaced-repetition queue, the `MENTOR.md` Hard Rules and session protocol, `CONVENTIONS.md`, and `frontier-lab.md` all stand exactly as written. No code, tests, starters, solutions, or phase directories were touched by this restructure. D-0001…D-0007 remain in force; this decision supersedes none of them.

**Revisit when.** At Claim 6 (the ~100M run) — that is the first claim whose cost and duration are genuinely uncertain, and the first real test of whether the 32-week budget holds. Also revisit immediately if any claim runs past its timebox twice, which would mean the 1–2 week box is wrong for this material rather than the claims being wrong.

---

## D-0009 · 2026-08-10 · Phase 0–1 rebuilt in Rust from scratch; replaces Claims 1–4

**Decision.**
1. **Claims 1–4 are replaced by two Rust claims, R0 and R1.** The from-scratch core is rebuilt in Rust with `std` and the toolchain only — no ndarray, nalgebra, candle, burn, tch, tokenizers, BLAS, or autodiff crate. Design: [`RUST_PHASE_0_1.md`](RUST_PHASE_0_1.md).
2. **R0 = the numerical substrate**, **split into R0a and R0b to respect the 1–2 week rule** — R0a = `Scalar` trait, strided tensor, broadcasting, reductions, blocked + threaded matmul (Days 1–8); R0b = tape-based autograd, gradient checker, softmax/cross-entropy, AdamW (Days 9–14). The boundary is natural: both halves have independently falsifiable exits (a GFLOP/s table vs. the naive oracle; a gradient checker watched to fail). **R1 = the first real LLM component** (byte-level BPE + GPT-2 pre-tokenizer, attention, the full forward pass, real-checkpoint logit parity).
3. **The split is drawn at "would this exist in a physics simulator?"** — because it's the only place both halves get an independently falsifiable exit test: property tests + gradient checking for R0, logit parity for R1.
4. **Claims 5–6 stay PyTorch on rented GPU.** April-2027 goal (a) — a ~100M model trained end-to-end on my own implementation — survives unchanged.
5. **Honest estimate: 5 weeks (29 working days, ~72 hrs at 15 hrs/week), not the 2 weeks budgeted.** The overage is accounted for line by line in `RUST_PHASE_0_1.md` §3. Four cards are explicitly 2-day.
6. **Primary oracle is the real GPT-2 124M checkpoint, not PyTorch fixtures.** Load OpenAI's published weights, match logits to 1e-3, greedy-decode byte-identically.

**Why Rust replaces the Python Phase 0.** Phase 0 was scaffolded and never proven (D-0008), so nothing is lost by rebuilding it. The Rust version is strictly deeper on the axis that matters for the north star: writing the strided memory model, the blocked matmul, the threading, and the autograd tape by hand *is* the kernel track (Track 1), several months before Phase 4 formally starts. A NumPy micrograd rebuild would have proven less.

**Why ~100M cannot be a Rust target.** M4 peak is ~550 GFLOP/s fp32; a good hand-rolled blocked matmul realistically reaches 100–250. GPT-2 124M at Chinchilla-optimal tokens is ~1.9e18 FLOPs ≈ **5 months of continuous compute**. Measured, not assumed. Hence the split: **train at ~20M in Rust, prove correctness at 124M by weight-loading, rent a GPU for 100M.**

**Verification strategy** (`RUST_PHASE_0_1.md` §6) — four independent layers, because there is no numpy oracle:
1. Property tests (permute round-trip, broadcast scaling, matmul associativity), plus a deliberately naive matmul **retained forever** as the oracle for every optimised kernel.
2. Central-difference gradient checking **in `f64`** — which is why the `Scalar` trait lands on Day 1 and cannot be retrofitted. Plus a random-projection variant that catches sign errors a plain sum hides.
3. **Golden files: a narrow yes.** Exactly three PyTorch-generated fixtures (tanh-GELU, LayerNorm, softmax+cross-entropy) with the generator script committed for provenance — the three places hand-derivation is genuinely ambiguous. Rule: a fixture may only be added *after* hand-deriving the value and disagreeing with yourself.
4. **Logit parity is primary; loss-curve shape is explicitly rejected as an architectural oracle.** A wrong mask, a transposed weight, or erf-instead-of-tanh GELU all still produce a smooth decreasing loss curve. Parity is binary and unforgeable.

**Constraint rulings** (§4) — exceptions argued, not silently relaxed:
- **KEEP** the GPT-2 pre-tokenizer, scoped to a hand-written state machine for that one pattern, not a regex engine. ~1.5 days, real lesson.
- **EXCEPTION: safetensors only, never `pytorch_model.bin`.** Pickle is a stack VM — ~1 week, zero learning value, arbitrary-code-execution hazard.
- **EXCEPTION: no hand-rolled threadpool.** `std::thread::scope` + row chunking gets the full 4× in an hour.
- **DEFER explicit NEON SIMD to Phase 2.** Build blocked first, measure the auto-vectorisation gap, then decide from data.
- **KEEP** the hand-rolled PRNG — reproducible gradient checks are impossible without it.

**Design pushback recorded** (§5): build the autograd as a **tape — an arena of nodes indexed by `usize`** — not `Rc<RefCell<Node>>`. The Python object-graph transliteration compiles and then produces runtime borrow panics and leaked cycles. The arena is what the borrow checker wants *and* what reverse-mode AD actually is.

**Honest risks recorded at decision time.**
- **The training gap is the real cost of this decision.** Old Claims 2–4 were training claims (hygiene, diagnosing broken runs, predicting loss at 30M); R0/R1 are implementation claims. Swapping them directly leaves **zero training experience before Claim 6's ~100M run** — the claim that demands a pre-committed loss prediction and a cost accounting. Placeholder **R2** is recorded in `COURSE_MAP.md` to close it, and is **not yet designed** (Phase 2 was out of scope for the Rust plan). **Design R2 before starting Claim 5; if R2 is cut, restore Claims 2–4 in PyTorch.**
- **The slack is gone. Arithmetic, so it can be checked:** old Claims 1–4 = 2+1+1+2 = **6w**. New block = R0a (2w) + R0b (2w) + R1 (2w) + R2 (~4w) = **10w boxed**, or **9w on honest estimate** (the designed part is 29 working days ≈ 4.8w, boxed at 6w). Delta = **+3w on estimate, +4w if every box is consumed.** The 2-week slack does not cover either, so the checkpoint is **1–2 weeks over-committed before any slip has occurred.** Per D-0008 the designated descope target remains kernels-core (Claims 11–15), which takes Feinberg Exercise B with it. Recorded now so that descope is a decision, not a surprise.
  > **STATUS: superseded by D-0010.** R2 moved to CONTINUATION and the PyTorch block (P1 1w + P2 1w) replaced it. The corrected arithmetic is 6w → 8w, **delta +2w**, which the 2-week slack covers exactly. The checkpoint has zero slack remaining but is **not** over-committed. The figures above stand as the audit trail of what was believed on 2026-08-10 before D-0010.
- **R2's `~4w` is a budget line, not a legal claim.** It exceeds the 1–2 week rule and must be split when designed, exactly as R0 was.
  > **STATUS: superseded by D-0010.** R2 is no longer on the checkpoint path, so the 1–2 week rule no longer binds it. P1 and P2 are 1w each and comply.
- **The autograd-vs-borrow-checker day is the schedule risk.** Estimated at 2 days; it is the one card that could plausibly take 4.

**Considered, rejected: meta-work.**
- A `cargo` workspace generator / project scaffolder — considered, rejected: meta-work. `cargo new` exists.
- A custom test-runner or gradient-check reporting harness beyond the plain `GradCheckReport` struct — considered, rejected: meta-work. `cargo test` is the runner.
- A benchmark-tracking dashboard for the matmul GFLOP/s numbers — considered, rejected: meta-work. A committed markdown table is the evidence.

**What this does NOT change.** The north star, the April-2027 checkpoint goals (a)–(d), Claims 5–23, the weekly paper track, `CONVENTIONS.md`, `REVIEW.md`, the `MENTOR.md` Hard Rules, and `frontier-lab.md` all stand. The Python `phase0/` tree is **retained untouched** as trail — it is superseded, not deleted.

**Revisit when.** At the end of R0. If the substrate took more than 3.5 weeks, R1's 2-week box is wrong too and the whole Rust track needs rescoping against the checkpoint rather than absorbing the slip silently.

---

## D-0010 · 2026-08-10 · PyTorch block replaces Rust R2; Simple English adopted as the docs standard

**Decision.**
1. **R2 (train the ~20M model in Rust) moves to CONTINUATION.** Two PyTorch claims replace it: **P1** (rebuild GPT-2 in PyTorch, 3-way parity against the Rust version, 1w) and **P2** (training hygiene + 3 timed broken-run diagnoses, 1w).
2. **The constraint reverses on purpose at P1.** Everything through R1 is `std` only. From P1 onward, use any library. The Rust work proves the theory. The PyTorch block buys the tool.
3. **pandas is out of scope.** It does nothing for language-model work. Adding it would pad the week.
4. **[Simple English (ASD-STE100)](.claude/skills/simple-english/SKILL.md) is the docs standard.** The skill is vendored at `course/.claude/skills/simple-english/` so later sessions load it automatically.
5. **Three Rust reference projects are recorded in `RESOURCES.md` with a read-after gate.** Read each one only after your own version of that component passes its test.
6. **Mermaid roadmap diagrams** are added to `COURSE_MAP.md` (whole ladder) and `RUST_PHASE_0_1.md` (whole track, Phase 0, Phase 1, verification layers).

**Why the PyTorch block, and why it replaces R2.** D-0009 recorded a real hole: R0a/R0b/R1 are *implementation* claims, so swapping them for old Claims 2–4 left **zero training experience before Claim 6's ~100M run**. The PyTorch block closes that hole directly, because old Claims 2–4 were PyTorch training claims. Two facts make it cheap. The theory is already banked by the Rust work, so only the API is new. And Claims 5–6 need PyTorch fluency regardless, so this is a prerequisite rather than a detour.

**Why R2 does not survive as a checkpoint claim.** Goal (a) is a ~100M run on rented GPU, in PyTorch. Training the 20M in Rust is the purer artifact and it serves no checkpoint goal. Keeping both cost about 4 extra weeks and would have made kernels-core (Claims 11–15, taking Feinberg Exercise B with it) the near-certain descope. The Rust engine stays capable of the run — Phase 2 is a config-and-compute problem, which was the design goal from the start.

**Budget arithmetic, so it can be checked.** Old Claims 1–4 = 2+1+1+2 = **6w**. New block = R0a (2w) + R0b (2w) + R1 (2w) + P1 (1w) + P2 (1w) = **8w**. Delta = **+2w**. The 2 weeks of slack cover this exactly. **This corrects D-0009, which recorded +3w to +4w against a ~4w R2 that no longer exists.** The checkpoint is fully committed with zero slack remaining, but it is not over-committed.

**Why Simple English, and where it does not apply.** ASD-STE100 is a controlled language for maintenance documentation. Its rules — 20 words per instruction, 25 per description, simple tenses, active voice, one instruction per sentence, condition before command, no `should`/`would`/`may`/`might` — make an instruction survive one read. The skill's own **Limits** section states: do not apply it to marketing copy, blog voice or brand writing, because it deletes persuasion by design. **The LinkedIn hooks are blog voice and stay in my own voice.** So does the reasoning prose in this decision log — a decision log records argument, not procedure.

**Honest risks recorded at decision time.**
- **P1 and P2 at 1 week each are tight.** They are only credible because the theory is already banked. If P1 overruns, the cause is NumPy/PyTorch API friction, not concepts — that is a signal to drill the API, not to extend the box.
- **Zero slack remains.** Any slip now comes out of kernels-core. Recorded so that descope is a decision, not a surprise.
- **The three Rust reference projects all break the `std`-only constraint** — RustGPT and the Reddit build use `ndarray`, and llms-from-scratch-rs uses `candle`. `ndarray` starts where Day 8 ends. Reading them early imports design decisions before the problem those decisions solve is understood. Hence the read-after gate.

**Considered, rejected: meta-work.**
- A doc linter that checks STE sentence limits in CI — considered, rejected: meta-work. The skill has a self-check step.
- A script to regenerate the mermaid diagrams from the claim tables — considered, rejected: meta-work. The tables change a few times a year.
- Forking or vendoring the three Rust reference repos locally — considered, rejected: meta-work. Links plus a read-after gate is the whole value.

**What this does NOT change.** The north star, checkpoint goals (a)–(d), R0a/R0b/R1 as designed, Claims 5–23, the weekly paper track, `CONVENTIONS.md`, `REVIEW.md`, the `MENTOR.md` Hard Rules, and `frontier-lab.md` all stand. The Python `phase0/` tree stays untouched as trail. `/Volumes/Mrigesh SSD/llm-scratch/` is the pre-restructure clone and is marked stale, not deleted.

**Revisit when.** At the end of P1. If a 3-way parity table took more than 1 week with the theory already banked, then P2's box is wrong too, and the PyTorch block needs rescoping before Claim 5.
