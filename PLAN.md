# PLAN.md — restructure around weekly capability claims

> Written 2026-08-05, before any edits. Approval gate.
> Scope: docs only. No code, tests, `hand_math/`, `evidence/`, starters, or solutions are touched. No phase dirs renamed.

---

## What I found that changed the plan

1. **`CLAUDE.md`, `MENTOR.md`, `frontier-lab.md` are not in the repo.** They live one level up, deliberately (D-0004: the charter is private, the journey is public). Working layout is now `/Volumes/Mrigesh SSD/llm-scratch-2027/` with `course/` as the clone and the three private files as its siblings — so `../MENTOR.md` links resolve and `CLAUDE.md` still auto-loads as an ancestor. Your live `/Volumes/Mrigesh SSD/llm-scratch/` is untouched.
2. **No GenHash material exists** — not in the repo, not on the SSD, not in your notes. The claim block is built from the compression spine you specified. The learned-hashing half is recorded as an open thread in DECISIONS.md, not invented into claims.
3. **34.1 weeks to 2027-04-01** at 12–15 hrs/wk = 410–512 hours. Budget: **32 claim-weeks + 2 weeks slack.**
4. **Phase 3 is a phase, `papers/READING_LOG.md` is not a file yet** — the old COURSE_MAP promised it (line 141) but never created it. The weekly paper track needs it.

---

## Files I will change

### In the repo

| File | Change | Why |
|---|---|---|
| `PLAN.md` | new | This plan; final self-check appended after edits (DoD #6). |
| `COURSE_MAP.md` | rewrite | North star kept verbatim + GenHash added to capstone signals; dated April 2027 checkpoint section directly beneath it; Phase 0–8 content remapped onto a numbered claim ladder with CHECKPOINT/CONTINUATION marks and a weekly-paper column; Phase 3 dissolved into the ladder; honest current position (Phase 0 scaffolded, nothing proven); session-protocol section reworded to claims. |
| `PROGRESS.md` | edit | Conventions moved to claim granularity. **No progress fabricated** — the session log, the "what hasn't happened yet" list and both dated session records stay exactly as written. |
| `DECISIONS.md` | append only | New `D-0008` recording this restructure and the sequencing reasoning, plus the "considered, rejected: meta-work" one-liners. D-0001…D-0007 untouched. |
| `papers/READING_LOG.md` | new | Required by the weekly paper track. Fixed ≤1-page format, header only — **zero fabricated entries**. |
| `README.md` | edit | Strip student framing ("Who this is for", "As a learner", "fork it, remix it, teach with it"); phase map → claim ladder + checkpoint. Minimal diff otherwise. |
| `SCHEDULE.md` | 2-line banner | Retired sprint doc. Banner marks it historical and points at COURSE_MAP. Content preserved — it's part of the trail, not deleted. |
| `whats_next.md` | banner + opener | Same treatment; strip the "Congrats. You went from…" student opener/closer. Tier lists preserved. |

### Outside the repo (private charter)

| File | Change | Why |
|---|---|---|
| `../CLAUDE.md` | rewrite | The operating contract (CHANGE 6): role boundaries, session start/end, enforcement rules, honesty rule, checkpoint. |
| `../MENTOR.md` | minimal edits | Keep Hard Rules A1–A8 and the Part D/E protocol; reword module→claim; mark Part B (first-run bootstrap) historical so a future session can't re-bootstrap; point Part C at COURSE_MAP as authority; fix stale `~/Desktop` paths. |

### Explicitly NOT touched

All code, tests, `hand_math/`, `evidence/`, starters, solutions · `phase0/` module READMEs · `CONVENTIONS.md` · `RESOURCES.md` · `REVIEW.md` · `SKILLS.md` · `MENTOR_LOG.md` · `JOURNEY.md` · `devlog/` · `PHASE8_FRAMEWORK.md` · `PHASE0_CLOSURE.md` · `extras/` · `papers/*.pdf` · `papers/README.md` · `glossary.md` · `debugging.md` · `requirements.txt` · `LICENSE` · legacy `07_`–`10_` dirs · `frontier-lab.md`.

---

## The sequencing decisions (the substance to approve)

**Claim shape.** One falsifiable capability statement · acceptance test named up front · 1–2 week timebox · terminal event = a shipped public artifact (≤1 extra hour, raw). No artifact, not done.

**Goal (a) — the ~100M model — reconciled.** It becomes an extension of the Phase 1 scale-up ladder (1.2's ~30M → a new ~100M claim), **not** Phase 8's 12M model. Reasoning: Phase 8 is a 3–4 month framework build whose 12M model proves *the framework*, not *scale*; pulling it before April would eat the whole budget. Phase 1's ladder already ends in a trained model on real data — extending it is the cheap path to (a). Phase 8's 12M target is left unchanged, post-checkpoint.

**CHECKPOINT path** (~32 claim-weeks, blocks in order):

| Block | Wks | Serves |
|---|---|---|
| Close Phase 0 — cold-start proof (Claim 1) | 1–2 | (d) |
| Phase 1 training ladder → **~100M end-to-end, loss curves + failure modes + cost** | 7 | **(a)** |
| JAX primer (3.J1–J4) + *How to Scale Your Model* exercises | 4 | prereq |
| **Feinberg Exercise A** (10M JAX/TPU adder + hand Chinchilla) | 2 | **(c)** |
| Kernels core — roofline → GPU model → tiled matmul → online softmax → fused attention | 6 | (c)(d) |
| **Feinberg Exercise B** (Pallas `ragged_dot` beater) | 2 | **(c)** |
| **GenHash** — VQ/perceptual prereq → HiFiC repro → Kodak benchmark vs JPEG XL → honest writeup | 6 | **(b)(c)** |
| KV cache + prefill/decode roofline | 2 | (d) |
| Depth-interview rehearsal, cold | 1 | **(d)** |

Feinberg A and B sit as early as their prerequisites allow, per your instruction — A immediately after the JAX primer, B immediately after the kernels core.

**CONTINUATION** (post-checkpoint, on the frontier-lab arc): full Phase 8 framework build · Phase 5.3/5.4 QuIP#/QTIP/AQLM reproductions · Phase 7 agents · full Phase 2 RL (real GRPO on a base model; the *derivation* stays on the checkpoint path for (d)) · Phase 6 paged attention / continuous batching / nano-vLLM / SnapKV.

**Goal (c) is not a claim.** No claim can force a stranger to share your work. It's structured as ~25 shots on goal — every claim ships an artifact — with A, B and the GenHash benchmark as the three highest-probability ones.

---

## Two things I want on the record before you approve

**The budget is tight, honestly.** Three capstone-grade artifacts (Feinberg A, Feinberg B, GenHash) plus a 100M scale-up in 32 claim-weeks at 12–15 hrs/wk is aggressive. The compressible block is kernels-core; if you slip, that's what descopes first, and Exercise B goes with it. I'd rather say this now than discover it in week 22.

**Goals (a) and (b) both need paid compute.** A ~100M model at roughly Chinchilla-optimal tokens, plus a HiFiC reproduction (GAN training, slow), are past what free Colab covers. Order-of-magnitude: a few hundred dollars of A100 time across both. Goal (a) asks for documented cost, so this is a deliverable rather than a hidden problem — but it's a real prerequisite, and the checkpoint fails without it.

---

## Open flag

`SKILLS.md` and `REVIEW.md` are untouched by design. Their content is proof-backed and the claim ladder doesn't require restating it — but if you want SKILLS.md's proof gates rephrased against claims, say so and I'll add it.

---

*Approved 2026-08-05. Executed as written.*

---
---

# Final self-check (appended after execution, 2026-08-05)

## Guardrail-by-guardrail

| # | Guardrail | Verdict | Evidence |
|---|---|---|---|
| 1 | **Docs-only rewrite.** No code, tests, `hand_math/`, `evidence/`, starters, or solutions modified/moved/deleted. | ✅ **held** | `git status` shows 8 files: `COURSE_MAP.md`, `DECISIONS.md`, `PROGRESS.md`, `README.md`, `SCHEDULE.md`, `whats_next.md`, `PLAN.md` (new), `papers/READING_LOG.md` (new). Grep for `*.py`, `*.ipynb`, `hand_math/`, `evidence/`, `phase0/` in the changed set returns **zero**. |
| 2 | **No phase dirs renamed.** | ✅ **held** | No renames of any kind. `phase0/` and the legacy `07_`–`10_` dirs are byte-identical. |
| 3 | **No history or completed-work records deleted. Completed = sacred.** | ✅ **held** | `DECISIONS.md` is strictly append-only — `git diff` shows **zero** deleted lines; D-0001…D-0007 untouched. `PROGRESS.md`'s completed-work records were moved into an explicit *"History (completed work — do not edit)"* section and verified present line-by-line: both session-01 and session-02 build records, all four capstone file descriptions, all three session-log entries, and the open questions. `SCHEDULE.md` and `whats_next.md` content preserved in full under retirement banners rather than deleted. |
| 4 | **Minimal diffs; preserve what works.** | ✅ **held** | Parity tests, `hand_math/`, `evidence/`, cold quizzes, `REVIEW.md`, `CONVENTIONS.md`, `SKILLS.md`, `MENTOR_LOG.md`, `JOURNEY.md`, `devlog/`, `RESOURCES.md`, `PHASE8_FRAMEWORK.md`, `PHASE0_CLOSURE.md`, `frontier-lab.md`, `extras/`, `papers/README.md` — **all untouched**. `MENTOR.md`'s Hard Rules A1–A8 and the Part D/E protocol are intact; edits there are wording-only (module→claim) plus historical markers on Parts B and C. `DECISIONS.md` style (date · decision · why · revisit when) followed exactly in D-0008. |
| 5 | **No new tooling, scripts, CI, student templates, or infrastructure.** | ✅ **held** | Zero executable files created. `papers/READING_LOG.md` is a markdown log the old `COURSE_MAP.md:141` already promised and CHANGE 5 requires — a document, not infrastructure. Five scaffolding urges were logged in D-0008 as *"considered, rejected: meta-work"* instead of built: a claim-tracking CLI, a per-claim directory generator, a parity-test CI job, a reading-log template filler, and an auto-computed weeks-to-checkpoint counter. |
| 6 | **Strip for-other-students framing; second person = Mrigesh.** | ✅ **held** | `README.md`: removed "Who this is for", "As a learner", "fork it, remix it, teach with it", "It is not a gentle introduction"; added an explicit "personal learning system, not a course for other people" line. `SCHEDULE.md` + `whats_next.md`: retirement banners; removed "Congrats. You went from…". `MENTOR.md`: added "it is **my** learning system, not a course for other people. Never write for hypothetical students; second person means me." |
| 7 | **CHANGE 1 — north star kept, not weakened; GenHash added; dated checkpoint under it.** | ✅ **held** | North-star paragraph kept verbatim with GenHash inserted into the capstone-signal list alongside Exercises A and B. Checkpoint section sits directly beneath it in both `COURSE_MAP.md` and `CLAUDE.md`, framed as a superset relationship, with (a)–(d) each mapped to specific claims. |
| 8 | **CHANGE 2 — claims as the unit of progress; honest current position.** | ✅ **held** | 23-claim ladder; every claim has a falsifiable "I can…" statement, an acceptance test named up front, a 1–2 week box, and a required artifact. Phases retained as groupings. Current position states plainly that Phase 0 is scaffolded but **nothing is proven** — no test run, no derivation, no evidence, quiz never taken — and that Claim 1 is closing it with a cold-start proof. |
| 9 | **CHANGE 3 — checkpoint-first sequencing, recorded in DECISIONS.md.** | ✅ **held** | 32 claim-weeks + 2 slack against a measured 34.1 weeks. CHECKPOINT/CONTINUATION marked per row. Goal (a)'s ~100M reconciled explicitly as a Phase 1 scale-up extension (Claim 6), **not** Phase 8's 12M, with the reasoning in D-0008. Feinberg A and B placed at the earliest point their prerequisites allow. |
| 10 | **CHANGE 4 — GenHash as a capstone track, reproduction-first.** | ✅ **held** | Claims 17–20, sequenced after Phase 1 and after the VQ/perceptual-loss prerequisite. Includes HiFiC repro → Kodak benchmark vs JPEG XL (LPIPS/SSIM/FID + own plots) → honest positioning writeup. "Novelty claims are banned until Claim 19's benchmark exists" stated in both `COURSE_MAP.md` and D-0008. Listed in the north-star capstone signals. |
| 11 | **CHANGE 5 — weekly paper track replaces Phase 3.** | ✅ **held** | Phase 3 dissolved. Every claim row carries a paper. `papers/READING_LOG.md` created with the exact 5-part ≤1-page format. Phase 3's reading list distributed across the ladder (Chinchilla→C4, LoRA→C5, FlashAttention→C14, DeepSeek/GRPO→C22, etc.). *How to Scale Your Model* kept as the Phase-4 prerequisite (Claim 9, every exercise in pencil). JAX primer kept as ordinary Claims 7–8. |
| 12 | **CHANGE 6 — CLAUDE.md as operating contract.** | ✅ **held** | Role and the never-author-core-code boundary as an explicit two-column table; `MENTOR.md` protocol kept and reworded to claims; session start states active claim + acceptance test + days left + this week's paper + weeks to checkpoint; session end updates `PROGRESS.md` at claim granularity; all six enforcement rules present; honesty rule present. |

## Things I was tempted to add and did not

1. **A claim-status script** that parses `PROGRESS.md` and prints the active claim, days remaining, and weeks to checkpoint. Genuinely useful, unambiguously meta-work. Logged as rejected in D-0008.
2. **Rewriting the seven `phase0/` module READMEs** to strip the sprint timers ("⏱️ 50 minutes", "Sprint > perfection"), which now directly contradict the claim-timebox model. You scoped the strip to root docs; I stayed inside that. **Flagged for a future call** — this is the largest remaining pocket of retired framing.
3. **Restructuring `SKILLS.md`** so its 0–5 proof gates map onto claim numbers. Coherent, and out of scope — raised as an open flag pre-approval and left alone.
4. **Auto-generating the first `READING_LOG.md` entry** for Attention Is All You Need. That would have been fabricated progress. The log ships empty with a note that the first entry is due with Claim 1.
5. **Deleting `SCHEDULE.md`** outright as dead weight. It's history; history is sacred. Banner instead.
6. **Inventing the learned-hashing half of GenHash.** No source material exists. Recorded as an unresolved open thread in D-0008 rather than filled with plausible-sounding claims.

## One thing I could not fix inside the guardrails

`phase0/PHASE0_CLOSURE.md` — the Claim 1 runbook — contains three stale absolute paths (`cd ~/Desktop/llm-scratch/course`, lines 68, 101, 244) pointing at a directory that no longer exists. It was not in the approved edit list, so I left it. **This will break the first commands you run on Claim 1.** One-line fix when you want it.

`DECISIONS.md` D-0004 carries the same stale path, but that file is append-only history and correctly stays as written.

## Not done, by instruction

Claim 1 was **not** begun. No test was run, no derivation written, no quiz issued.
