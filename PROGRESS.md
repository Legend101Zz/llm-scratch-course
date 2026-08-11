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

### 🟨 Claim R0a — I can implement a strided, broadcasting tensor library in Rust with a cache-blocked multithreaded matmul, and prove it correct with no reference implementation.

> Per **D-0009 (2026-08-10)**, the from-scratch core is rebuilt in **Rust, `std` only** — no ndarray, nalgebra, candle, burn, tch, tokenizers, BLAS, or autodiff crate. This replaces old Claims 1–4. Day-by-day design, verification strategy and constraint rulings: [`RUST_PHASE_0_1.md`](RUST_PHASE_0_1.md).

**Acceptance test** (fixed up front, not retrofitted):

1. Property tests green: `permute` round-trip on data *and* strides · `broadcast(a).sum() == a.sum() × factor` · `(AB)C ≈ A(BC)` · `(AB)ᵀ == BᵀAᵀ` · `A·I == A`.
2. `blocked_matches_naive` passes at **non-multiples of the block size** (129×257·257×63), in both `f32` and `f64`.
3. `parallel_matches_blocked` is **bit-identical**, and invariant across 1/2/4/8 threads.
4. Committed GFLOP/s table vs. block size and thread count, stating the measured **% of the M4's ~550 GFLOP/s fp32 peak** — with the predicted L1-derived optimal block size written down *before* measuring.

**Timebox:** 2 weeks. **Started 2026-08-11.** The clock runs. 9 working days remain, and Day 7 takes 2 of them.
**Day-by-day checklist:** [`RUST_TRACKER.md`](RUST_TRACKER.md).
**Paper:** Attention Is All You Need — re-read with my own attention code open in the other tab.
**Artifact (required to close):** a post on the blocking result — same FLOPs, same output, N× faster — with the roofline arithmetic that predicted it.

**Proven so far: nothing.** Day 0 is setup, not a capability. See below.

### R0a day state

- **Day 0 — done (2026-08-11), commit `39952d6`.** The `rust/` crate builds. `criterion 0.5` is the only dev-dependency. `rust/target/` is ignored. Mrigesh ran the commands himself.
- **Day 1 — open.** `rust/tests/day1.rs` holds the four acceptance tests from the day card, plus one test that a seed of 0 panics. The mentor wrote the tests. `cargo test --test day1` fails to compile, because `src/scalar.rs` and `src/rng.rs` do not exist. **That red is recorded, not hidden.** No implementation exists. No hand-derivation exists. No post is shipped.

---

## Where things actually stand (2026-08-05)

Phase 0 is **scaffolded, not proven**. Every module has prose, starter, solution, parity test, and `hand_math/` + `evidence/` READMEs. None of it is verified by me.

**What has NOT happened:**

- **No parity test has been run in my venv.** The torch assertions are written and were spot-verified numpy-side only on a machine without torch. They are **unverified**.
- **No `hand_math/` derivation exists.** The folders contain only the "what goes here" README.
- **No `evidence/` output exists.** No `test_output.txt`, no `metrics.json`, no `loss_curve.png`, no `roofline.md`. Same reason.
- **The cold quiz was never taken.** R-001…R-008 were posed in session 01 and paused for gap closure. Still pending.

**This gap is no longer closed in Python.** Per D-0009, because Phase 0 was only ever scaffolded, nothing is lost by rebuilding it — so the from-scratch core is being rebuilt in Rust, which is strictly deeper on the axis the north star cares about. `phase0/` and `PHASE0_CLOSURE.md` are **retained untouched as trail: superseded, not deleted.**

**Everything downstream (R0b, R1, P1, P2, Claims 5–23) is ⬜ not started.**

---

## Checkpoint status

**April 2027 · ~33 weeks out as of 2026-08-10 · 32 claim-weeks budgeted + 2 weeks slack.**

> ⚠ **The slack is fully spent, but the checkpoint is not over-committed.** Arithmetic, so it can be checked: old Claims 1–4 = 2+1+1+2 = **6w**. New block = R0a (2w) + R0b (2w) + R1 (2w) + P1 (1w) + P2 (1w) = **8w**. Delta = **+2w**, which the 2 weeks of slack cover exactly. **This corrects the earlier +3w/+4w figure**, which assumed a ~4w Rust R2 that D-0010 moved to CONTINUATION. Zero slack now remains, so any slip comes out of kernels-core (Claims 11–15), taking Feinberg Exercise B with it. Recorded so that descope stays a decision, not a surprise.

| Goal | Banked by | State |
|---|---|---|
| (a) ~100M LM trained end-to-end on my own implementation, with loss curves, failure modes, cost | Claim 6 | ⬜ |
| (b) GenHash: HiFiC reproduced → JPEG XL/Kodak benchmark → honest writeup | Claims 17–20 | ⬜ |
| (c) A public artifact a recognizable ML person shared unprompted | structural; best shots Claims 10, 16, 20 | ⬜ |
| (d) Pass a founding-engineer depth interview cold | Claim 23 | ⬜ |

**Claims proven: 0 / 24.**

---

## Blockers / open questions

- **None blocking.** R0a can start immediately — `rustc 1.91.1` is installed and it needs nothing else.

**Open questions for me:**

- ~~R2 is undesigned, and it is the training gap.~~ **Closed by D-0010.** The PyTorch block (P1, P2) fills it. R2 moved to CONTINUATION.
- **The three PyTorch fixtures need one Colab session** (tanh-GELU, LayerNorm, softmax+CE), plus the Layer-4 reference oracles (block statistics, GPT-2 logits for 3 prompts, 200 tiktoken pairs). Torch is **not installed locally** — this is a scheduled errand, due before Day 21.
- Hand-derivations as paper-photo or transcribed LaTeX? (Affects `hand_math/` file types; both are documented per module.)
- Am I OK screen-recording build sessions from R0a onward? Claim 10 (Exercise A) requires a recording — practise the habit early rather than panicking then.
- **Compute budget.** Goals (a) and (b) both need paid GPU time — order a few hundred dollars of A100 hours across Claim 6 and Claims 17–19. Free Colab does not cover either. Answer this before Claim 5, not at Claim 6.

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

- **2026-08-11 · 05** — R0a timebox started. Mrigesh completed Day 0 himself: `cargo new --lib --name rustgpt rust`, `criterion` as the single dev-dependency, `rust/target/` ignored, empty crate committed (`39952d6`). The mentor then wrote `rust/tests/day1.rs` — the Day 1 red — and wrote no implementation. The overdue `REVIEW.md` queue (R-001…R-010, all pre-Rust theory) is still unasked. Then, at Mrigesh's request, the mentor added the `days/` lesson files — `DAY_01.md` and `DAY_02.md` — and recorded the generation rule in `CLAUDE.md`, the new `AGENTS.md`, and [D-0011](DECISIONS.md). No lesson exists past Day 2, by rule.
- **2026-08-05 · 04** — Restructured the course around capability claims; added the dated April 2027 checkpoint; dissolved Phase 3 into a weekly paper track; added GenHash as a capstone track; rewrote `CLAUDE.md` as the operating contract. Docs only — no code, tests, `hand_math/`, `evidence/`, starters or solutions touched. See [D-0008](DECISIONS.md).
- **2026-05-23 · 02** — Foundation gap closure. After an honest audit revealed the bootstrap had written discipline conventions but not enforced them, closed three concrete gaps: Module 2 parity test, `hand_math/` + `evidence/` scaffolding across all Phase-0 modules, and the `07_phase0_capstone/` integration module (numpy ↔ torch GPT with parity test + tinyshakespeare training).
- **2026-05-23 · 01** — Bootstrap: pivoted sprint → journey. All foundation files written. Modules 3/4/5/6 patches.
- **(prior, unlogged)** — Original sprint course built (Modules 0–10 scaffolded; Modules 1–6 fleshed out deeply; Modules 7–10 skeletal).
