# PROGRESS.md — where we are, what's next, what's blocking

> Updated at the end of every session. The mentor reads this *first* at every session start.

---

## Current phase: **Phase 0 — From-scratch core** 🟨 (closure imminent)

## Where we left off (2026-05-23, end of session 02)

Phase 0 is fully scaffolded. The bootstrap + audit + gap-closure work in this session produced:

### Bootstrap (session 01)
- ✅ `DECISIONS.md`, `CONVENTIONS.md`, `COURSE_MAP.md`, `RESOURCES.md`
- ✅ Public `README.md` rewritten; `JOURNEY.md` + first devlog entry
- ✅ Part D memory files (`PROGRESS.md`, `SKILLS.md`, `REVIEW.md`, `MENTOR_LOG.md`)
- ✅ Module 3 BPE: starter + solution + 4-test parity file
- ✅ Module 4: full roofline section (T4, FlashAttention insight earned) + 5-test parity file
- ✅ Module 5: 5-test parity file (gelu, LN, FFN, block forward, residual path)
- ✅ Module 6: drill 11 bridge — rebuilds Module 4 attention in PyTorch, asserts numpy parity

### Gap closure (session 02 — after honest audit)
- ✅ Module 2: 3-test parity file (forward, per-parameter gradients, training convergence). Closes the most glaring Phase 0 gap.
- ✅ `hand_math/` + `evidence/` folders with READMEs in every Phase-0 module (Modules 1–6) — each README specifies exactly what derivations + outputs go there.
- ✅ **`07_phase0_capstone/`** — tiny GPT assembled from Modules 4/5 components:
  - `numpy_gpt.py` — forward-only NumPy GPT (verified runs, 112k params on V=65, d=64, h=4, 2 layers)
  - `torch_gpt.py` — trainable PyTorch GPT with matching architecture
  - `test.py` — 3 parity tests (param count, forward to 1e-4, full-stack causal invariant)
  - `train.py` — 2000-step training on tinyshakespeare with loss curve + sample generation outputs
  - README + `hand_math/` + `evidence/` scaffolding

### What hasn't happened yet
- The user **has not yet run the parity tests in their venv** (this machine doesn't have torch installed). Tests are designed correctly and have been spot-verified in numpy-only mode, but full torch parity verification is the user's first task next session.
- The cold quiz on attention + transformer block (posed in session 01) was paused for gap closure. **Re-issue at start of session 03.**
- `hand_math/` folders contain only the "what goes here" README. The actual derivations are the user's work — they get written by the user during/after working through each module.
- `evidence/` folders contain only the "what goes here" README. The actual `test_output.txt`, `metrics.json`, `loss_curve.png`, `roofline.md` files are produced by running the modules in the user's venv and capturing output.

## What's next (in order)

1. **User: install torch in venv + run all parity tests + capture `evidence/` outputs.**
   ```bash
   cd ~/Desktop/llm-scratch/course
   source .venv/bin/activate
   pip install -r requirements.txt   # if not already done
   # Run each module's test:
   for m in 01_autograd 02_neural_net 03_tokenizer_bigram 04_attention_scratch 05_transformer_scratch; do
     echo "=== $m ==="
     cd $m && USE_SOLUTION=1 python test.py > evidence/test_output.txt 2>&1
     cd ..
   done
   cd 06_pytorch_crash && python tensor_drills_solution.py > evidence/drills_output.txt 2>&1 && cd ..
   cd 07_phase0_capstone && python test.py > evidence/parity_output.txt 2>&1 && python train.py && cd ..
   ```
   Then commit + push the evidence files.

2. **User: cold quiz.** Mentor re-issues the 8-question attention + transformer block quiz from session 01. User answers without looking at READMEs. Mentor grades honestly.

3. **User: at least one `hand_math/` derivation per module.** Photo or transcription. Mentor verifies cold by asking to re-derive one step.

4. **Phase 0 promotion review.** If the parity tests pass + quiz passes + at least one hand-math derivation per module exists → promote to Phase 1.

## What's queued for the next session (after Phase 0 closes)

- **Phase 1.1**: scaffold the first real-training module (nanoGPT-style pretrain on tinyshakespeare with proper hygiene — warmup + cosine LR, gradient clipping, bf16 if available, eval intervals, checkpoint+resume).
- The existing `08_train_colab/` and `09_finetune_lora/` skeletons will be rewritten under Phase 1's hygiene rules (per D-0006).

## Blockers / open questions

- **None blocking** — the user can run the tests autonomously.
- **Open questions for the user:**
  - Will you do hand-derivations on paper-photo or transcribed LaTeX? (Affects `hand_math/` file types — both formats are documented in the per-module READMEs.)
  - Are you OK starting screen-recording build sessions from Phase 0? (Per `frontier-lab.md` evidence ask. Phase 4 graded exercises require recordings — practice early.)
  - What's your compute access beyond Colab T4? (Matters for Phase 1.2 — the scale-up to 30M params on OpenWebText subset.)

## Stalled / parked items

- The old `07_gpt_pytorch/` directory remains in the tree (sprint-era skeletal content). Per D-0006 it'll be rewritten when Phase 1 starts. **Don't be confused** by the two `07_*` directories — `07_phase0_capstone/` is the active Phase 0 closure module; `07_gpt_pytorch/` is legacy.

## Session log (most recent first)

- **2026-05-23 · 02** — Foundation gap closure. After an honest audit revealed the bootstrap had written discipline conventions but not enforced them, closed three concrete gaps: Module 2 parity test, `hand_math/` + `evidence/` scaffolding across all Phase-0 modules, and the `07_phase0_capstone/` integration module (numpy ↔ torch GPT with parity test + tinyshakespeare training).
- **2026-05-23 · 01** — Bootstrap: pivoted sprint → journey. All foundation files written. Modules 3/4/5/6 patches.
- **(prior, unlogged)** — Original sprint course built (Modules 0–10 scaffolded; Modules 1–6 fleshed out deeply; Modules 7–10 skeletal).
