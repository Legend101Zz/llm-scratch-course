# PROGRESS.md — where we are, what's next, what's blocking

> Updated at the end of **every** session, at **claim granularity**. The mentor reads this *first* at every session start.
>
> **Honesty rule.** This file records what is **PROVEN** — tests actually run in my venv, quiz actually passed, artifact actually shipped. Not what is scaffolded. Scaffolded ≠ done. Written ≠ verified. "Should pass" ≠ passed.

---

## Conventions for this file

Progress is measured in **claims**, not modules. See [`COURSE_MAP.md`](COURSE_MAP.md) for the ladder and [`../CLAUDE.md`](../CLAUDE.md) for the enforcement rules.

**A claim's state is one of four:**

| State | Meaning |
|---|---|
| ⬜ **not started** | No work begun. |
| 🟨 **open** | Work in progress. Exactly one claim may be open at a time. |
| ⏸ **descoped / split** | Ran past its timebox. Record what was cut and why in `DECISIONS.md`. Never silently extended. |
| ✅ **proven** | Acceptance test **run and passed** *and* the public artifact **shipped**. Both. |

**A claim is not ✅ until its artifact exists.** Work finished with no artifact shipped is still 🟨.

**Every open claim records:** claim number and statement · its acceptance test verbatim · timebox start date and days remaining · this week's paper · the one deliverable scoped for the current session.

---

## Active claim

### 🟨 Claim 1 — I can rebuild the Phase 0 transformer stack from a blank file and prove it matches PyTorch.

**Acceptance test** (fixed up front, not retrofitted):

1. Every `phase0/*/test.py` passes **in my venv**, output captured to each module's `evidence/`.
2. `phase0/07_phase0_capstone/train.py` reaches **< 2.0 nats by step 2000**, loss curve committed.
3. **≥1 `hand_math/` derivation per module** (photo or transcription).
4. Cold quiz `REVIEW.md` R-001…R-008 scored **≥6/8, with R-001, R-002 and R-004 all correct**.

**Timebox:** 2 weeks. **Not yet started — clock not running.**
**Paper:** Attention Is All You Need — re-read with my own Module 4/5 open in the other tab.
**Artifact (required to close):** a post with the loss curve and the numpy↔torch parity numbers, including anything that failed first.

**Proven so far: nothing.** See below.

---

## Where things actually stand (2026-08-05)

Phase 0 is **scaffolded, not proven**. Every module has prose, starter, solution, parity test, and `hand_math/` + `evidence/` READMEs. None of it is verified by me.

**What has NOT happened:**

- **No parity test has been run in my venv.** The torch assertions are written and were spot-verified numpy-side only on a machine without torch. They are **unverified**.
- **No `hand_math/` derivation exists.** The folders contain only the "what goes here" README.
- **No `evidence/` output exists.** No `test_output.txt`, no `metrics.json`, no `loss_curve.png`, no `roofline.md`. Same reason.
- **The cold quiz was never taken.** R-001…R-008 were posed in session 01 and paused for gap closure. Still pending.

**Claim 1 is exactly this gap.** The runbook is [`phase0/PHASE0_CLOSURE.md`](phase0/PHASE0_CLOSURE.md) — read it first.

**Everything downstream (Claims 2–23) is ⬜ not started.**

---

## Checkpoint status

**April 2027 · 34 weeks out as of 2026-08-05 · 32 claim-weeks budgeted + 2 weeks slack.**

| Goal | Banked by | State |
|---|---|---|
| (a) ~100M LM trained end-to-end on my own implementation, with loss curves, failure modes, cost | Claim 6 | ⬜ |
| (b) GenHash: HiFiC reproduced → JPEG XL/Kodak benchmark → honest writeup | Claims 17–20 | ⬜ |
| (c) A public artifact a recognizable ML person shared unprompted | structural; best shots Claims 10, 16, 20 | ⬜ |
| (d) Pass a founding-engineer depth interview cold | Claim 23 | ⬜ |

**Claims proven: 0 / 23.**

---

## Blockers / open questions

- **None blocking.** Claim 1 can start immediately — it needs only a venv and my own time.

**Open questions for me:**

- Hand-derivations as paper-photo or transcribed LaTeX? (Affects `hand_math/` file types; both are documented per module.)
- Am I OK screen-recording build sessions from Claim 1 onward? Claim 10 (Exercise A) requires a recording — practise the habit early rather than panicking then.
- **Compute budget.** Goals (a) and (b) both need paid GPU time — order a few hundred dollars of A100 hours across Claim 6 and Claims 17–19. Free Colab does not cover either. This needs answering well before Claim 4, not at Claim 6.

## Stalled / parked items

- The legacy `07_gpt_pytorch/` at repo root is sprint-era skeletal content. Per D-0006 it gets rewritten if and when its phase comes up. The active Phase 0 capstone is `phase0/07_phase0_capstone/` — ignore the root one.

---

## History (completed work — do not edit)

Kept as the audit trail. These records describe what was **built and committed**; per the honesty rule above, "scaffolded" here never means "proven."

### Bootstrap (session 01, 2026-05-23)

- ✅ `DECISIONS.md`, `CONVENTIONS.md`, `COURSE_MAP.md`, `RESOURCES.md`
- ✅ Public `README.md` rewritten; `JOURNEY.md` + first devlog entry
- ✅ Part D memory files (`PROGRESS.md`, `SKILLS.md`, `REVIEW.md`, `MENTOR_LOG.md`)
- ✅ Module 3 BPE: starter + solution + 4-test parity file
- ✅ Module 4: full roofline section (T4, FlashAttention insight earned) + 5-test parity file
- ✅ Module 5: 5-test parity file (gelu, LN, FFN, block forward, residual path)
- ✅ Module 6: drill 11 bridge — rebuilds Module 4 attention in PyTorch, asserts numpy parity

### Gap closure (session 02, 2026-05-23 — after honest audit)

- ✅ Module 2: 3-test parity file (forward, per-parameter gradients, training convergence). Closed the most glaring Phase 0 gap.
- ✅ `hand_math/` + `evidence/` folders with READMEs in every Phase-0 module (Modules 1–6) — each README specifies exactly what derivations + outputs go there.
- ✅ **`phase0/07_phase0_capstone/`** — tiny GPT assembled from Modules 4/5 components:
  - `numpy_gpt.py` — forward-only NumPy GPT (verified runs, 112k params on V=65, d=64, h=4, 2 layers)
  - `torch_gpt.py` — trainable PyTorch GPT with matching architecture
  - `test.py` — 3 parity tests (param count, forward to 1e-4, full-stack causal invariant)
  - `train.py` — 2000-step training on tinyshakespeare with loss curve + sample generation outputs
  - README + `hand_math/` + `evidence/` scaffolding

### Session log (most recent first)

- **2026-08-05 · 04** — Restructured the course around capability claims; added the dated April 2027 checkpoint; dissolved Phase 3 into a weekly paper track; added GenHash as a capstone track; rewrote `CLAUDE.md` as the operating contract. Docs only — no code, tests, `hand_math/`, `evidence/`, starters or solutions touched. See [D-0008](DECISIONS.md).
- **2026-05-23 · 02** — Foundation gap closure. After an honest audit revealed the bootstrap had written discipline conventions but not enforced them, closed three concrete gaps: Module 2 parity test, `hand_math/` + `evidence/` scaffolding across all Phase-0 modules, and the `07_phase0_capstone/` integration module (numpy ↔ torch GPT with parity test + tinyshakespeare training).
- **2026-05-23 · 01** — Bootstrap: pivoted sprint → journey. All foundation files written. Modules 3/4/5/6 patches.
- **(prior, unlogged)** — Original sprint course built (Modules 0–10 scaffolded; Modules 1–6 fleshed out deeply; Modules 7–10 skeletal).
