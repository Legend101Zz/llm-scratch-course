# PHASE 0 — Closing Checklist (a friendly walkthrough)

> Read this top to bottom. It tells you (a) what happened in the last two sessions, (b) what *you* need to do to officially close Phase 0, and (c) what to do when something breaks.
>
> Cross-refs (in case you want them): [`COURSE_MAP.md`](COURSE_MAP.md) for the full 0→8 arc · [`CONVENTIONS.md`](CONVENTIONS.md) for the per-module rules · [`PROGRESS.md`](PROGRESS.md) for the current state.

---

## Part 1 — What just happened (the "why")

### The framing pivot

This repo used to be called "the 8-hour LLM sprint." Eleven timed modules. Touch every concept, type fast, peek at the solution if you blow past 1.5× the estimate.

**That framing was wrong for the actual goal** (landing at a frontier lab — see [`../frontier-lab.md`](../frontier-lab.md)). Sprint thinking rewards shallow touch-every-concept. Frontier-lab work rewards *deep evidence per concept* — handwritten derivations, parity tests, roofline analyses, measured ablations.

So: I retired the sprint framing and rebuilt the course as **Phase 0 → Phase 8**, a multi-month journey with deep evidence per phase. The full map lives in [`COURSE_MAP.md`](COURSE_MAP.md).

### What the last two sessions actually built

| Session 1 | Session 2 |
|---|---|
| Bootstrap | Gap closure |
| Wrote 11 foundational docs (DECISIONS, COURSE_MAP, CONVENTIONS, RESOURCES, README, JOURNEY, PROGRESS, SKILLS, REVIEW, MENTOR_LOG, devlog/2026-05-23-01) | Honest audit revealed gaps; fixed them |
| Module 3: turned BPE explanation into a real exercise (`bpe_starter.py` + `bpe_solution.py` + `test.py`) | Module 2: added `test.py` that proves your MLP matches PyTorch's gradients to 1e-5 |
| Module 4: added a worked **roofline** section (FLOPs vs bytes → memory-bound → why FlashAttention exists) + `test.py` | Created `hand_math/` + `evidence/` folders in every Phase 0 module (with READMEs telling you what to put in each) |
| Module 5: added `test.py` covering gelu, LayerNorm, FFN, full block forward, and the residual path | Built **`07_phase0_capstone/`** — the missing tiny GPT that assembles your Module 4/5 components into a real LM |
| Module 6: added drill 11 (rebuild Module 4 attention in PyTorch, assert numerical parity) | Updated all memory files to reflect new state; second devlog entry |

### Why none of this is "done" yet

Every parity test I wrote *should* pass when you run it. But "should pass" isn't the same as "did pass." This machine doesn't have PyTorch installed — so I could only verify the numpy-only parts of each test ran cleanly. **The torch parity assertions are unverified.** Verifying them in your venv is step 1 below.

---

## Part 2 — Where Phase 0 stands right now

```
course/
├── 00_start/              ✅ mental model + setup
├── 01_autograd/           ✅ scalar Value class + backprop + parity test
├── 02_neural_net/         ✅ MLP on Value engine + parity test (forward + per-param-gradient + training)
├── 03_tokenizer_bigram/   ✅ char vocab + bigram + BPE from scratch + 4-test parity file
├── 04_attention_scratch/  ✅ Q/K/V attention + roofline section + 5-test parity file
├── 05_transformer_scratch/✅ full block (LN + MHA + FFN + residuals) + 5-test parity file
├── 06_pytorch_crash/      ✅ 4 deep-dive docs + 11 drills (drill 11 = numpy↔torch bridge)
└── 07_phase0_capstone/    ✅ tiny GPT (numpy + torch) + parity test + training script  ⟵  the new piece
```

Every module also has:
- `hand_math/README.md` — explains what derivation to put in this folder (you do it on paper)
- `evidence/README.md` — explains what test outputs / metrics / roofline calcs to put here (you generate them by running tests)

These two folders are *empty of content* right now. **You populate them as you work** — they're how Phase 0 will officially "close."

---

## Part 3 — Your job (4 concrete tasks)

You need to do **four things**. Two are mostly mechanical (Tasks 1–2). Two need real thought (Tasks 3–4).

### Task 1 — Set up your environment

If you haven't already:

```bash
cd ~/Desktop/llm-scratch/course

# Create a virtual environment so dependencies don't pollute your system Python
python3 -m venv .venv

# Activate it (your terminal prompt should now show "(.venv)")
source .venv/bin/activate

# Install the dependencies (torch, numpy, tiktoken, matplotlib, transformers, datasets)
pip install -r requirements.txt
```

This takes ~5 min the first time (PyTorch is large). After this, you'll always run `source .venv/bin/activate` before doing anything in the course directory.

**Quick check it worked:**

```bash
python -c "import torch, numpy, tiktoken; print('torch', torch.__version__, '| numpy', numpy.__version__)"
```

You should see something like `torch 2.x.x | numpy 1.x.x`. If you see `ModuleNotFoundError`, the activation didn't work or `pip install` failed — fix that before proceeding.

---

### Task 2 — Run every test and capture the outputs to `evidence/`

This proves on disk that the from-scratch implementations actually match PyTorch. **This is the most important task** — it's what turns "I wrote code that runs" into "I wrote code that's verifiably correct."

#### 2a. Modules 1–5 (parity tests)

Run this from the `course/` directory (your venv must be active):

```bash
for m in 01_autograd 02_neural_net 03_tokenizer_bigram 04_attention_scratch 05_transformer_scratch; do
  echo "=== $m ==="
  (cd $m && USE_SOLUTION=1 python test.py > evidence/test_output.txt 2>&1)
  tail -3 $m/evidence/test_output.txt
done
```

What this does:
- For each module, it `cd`s into the folder, runs `test.py` (testing the *reference solution*, not your starter — we want to confirm the test infrastructure works before testing your code), and captures the full output to `evidence/test_output.txt`.
- Then it prints the last 3 lines so you can see the result inline.

**Expected output (last line of each):** something ending with `✅ all <N> ... tests passed.`

If any module fails, **stop and copy the error to me.** A failing test means there's a real bug in either my parity test logic or the solution code — I need to fix it before you keep going.

#### 2b. Module 6 (drills)

```bash
(cd 06_pytorch_crash && python tensor_drills_solution.py > evidence/drills_output.txt 2>&1)
tail -3 06_pytorch_crash/evidence/drills_output.txt
```

Expected: `✅ all 11 drills passed.`

#### 2c. Module 7 — capstone parity test

```bash
(cd 07_phase0_capstone && python test.py > evidence/parity_output.txt 2>&1)
tail -3 07_phase0_capstone/evidence/parity_output.txt
```

Expected: `✅ Phase 0 capstone parity tests passed. The from-scratch components compose into a working LM.`

#### 2d. Commit the evidence

After all tests pass:

```bash
git add -A
git commit -m "Phase 0: capture parity-test evidence in venv"
git push
```

You now have `evidence/test_output.txt` in every Phase 0 module — proof that the from-scratch implementations are byte-for-byte equivalent to PyTorch on the operations you implemented.

---

### Task 3 — Train the capstone GPT (this is the satisfying part)

```bash
(cd 07_phase0_capstone && python train.py)
```

What you'll see:
- The script downloads tinyshakespeare (if not already present), tokenizes it character-level (V=65), and trains for 2000 steps.
- Every 200 steps it prints `step XXXX/2000  train Y.YYY  val Z.ZZZ  (T.Ts elapsed)`.
- Training takes **3-5 minutes on CPU**, less on GPU. (You can pass `--device cuda` or `--device mps` if you have one.)

What you should see:

| step | train loss (nats) | what it means |
|---|---|---|
| 0 | ~4.17 | random init — uniform distribution over 65 chars (`ln 65 ≈ 4.17`) |
| 500 | ~2.4 | learning character bigrams + capitalization conventions |
| 1000 | ~1.9 | recognizing common words like 'the', 'and', 'to' |
| 2000 | ~1.6 | short Shakespeare-like phrases emerge |

When it finishes, the script writes three files to `07_phase0_capstone/evidence/`:
- `loss_curve.png` — plot of train + val loss over time
- `training_log.json` — the raw numbers
- `sample_generations.txt` — what the model writes when prompted with `"ROMEO:"`, `"\n"`, `"JULIET:"`

**Don't expect coherent prose.** This is 112k params trained on 1MB of text on CPU. The output should look "Shakespeare-shaped" — capital letters at line starts, recognizable short words, sometimes a real sentence fragment — but it's not a real LM. The point is *the loss dropped* and *the architecture works*, not that the generations are good.

Commit the evidence:

```bash
git add 07_phase0_capstone/evidence/
git commit -m "Phase 0 capstone: trained 2000 steps + saved loss curve / samples"
git push
```

---

### Task 4 — Two things only you can do (the human work)

#### 4a. Hand-math derivations (at least one per module)

Per [`CONVENTIONS.md`](CONVENTIONS.md), every Phase 0 module needs **at least one handwritten derivation** committed to its `hand_math/` folder. The point: math you've *typed* doesn't stick the way math you've *derived on paper* does. The git log of `hand_math/` is the evidence trail.

For each module, read its `hand_math/README.md` — it tells you exactly what to derive. Pick the one you find most interesting per module and do it on paper. **One sheet of paper per module is enough to start.**

Easiest order (cheapest first):

| # | Module | What to derive | Difficulty |
|---|---|---|---|
| 1 | `01_autograd` | Chain rule through `tanh` (use the example a=2, b=-3 in the README) | Easy — 1 page |
| 2 | `00_start` (skip — no derivation needed) | — | — |
| 3 | `03_tokenizer_bigram` | The MLE = empirical counts argument | Easy — 1 page |
| 4 | `02_neural_net` | MSE gradient flow through one neuron + finite-difference check | Medium — 1 page + a small Python script |
| 5 | `04_attention_scratch` | The 1/√d_k scaling from variance of the dot product | Medium — 1 page of algebra |
| 6 | `05_transformer_scratch` | Pre-norm vs post-norm gradient highway argument (or LayerNorm backward — pick one) | Medium-hard — 1-2 pages |
| 7 | `07_phase0_capstone` | Causal-mask induction proof (the stack preserves causality if every block does) | Easy — half a page |

**Format options (pick whichever you'll actually do):**
- Photo from your phone of the page → save as `hand_math/01_chain_rule.jpg` in the relevant module.
- Transcribe to markdown after you've done the work on paper → save as `hand_math/01_chain_rule.md` with LaTeX math (you can ask Claude or any chatbot to transcribe your photo to LaTeX — that's allowed; the *derivation* must be yours).

After each derivation, commit:
```bash
git add 01_autograd/hand_math/
git commit -m "hand_math: chain rule through tanh"
git push
```

#### 4b. The cold quiz

The mentor (this Claude session) posed 8 questions during session 01 about attention + the transformer block. You **paused** them to close the gaps first. They're still queued — in [`REVIEW.md`](REVIEW.md), questions R-001 through R-008.

To take it: open a new conversation with me and paste:

> *"I'm ready for the cold quiz on attention + transformer block from REVIEW.md R-001 through R-008. Don't show me the questions until I've finished module-by-module test runs."*

I'll re-issue the questions. You answer in your own words **without looking at the module READMEs**. I grade each one honestly (per Hard Rule #6 — no flattery). Wrong / partial answers schedule a re-quiz (REVIEW.md tracks due dates per question with a spaced-repetition cadence).

**Rough time:** 20-40 minutes if you can answer most cold. Faster if you can't (writing "I don't remember — would look it up" is the correct answer for anything you don't know, and it's worth more than guessing).

---

## Part 4 — Troubleshooting

### "I get `ModuleNotFoundError: No module named 'torch'`"

Your venv isn't activated, or `pip install` failed. Run:

```bash
source .venv/bin/activate
which python   # should be ~/Desktop/llm-scratch/course/.venv/bin/python
pip install -r requirements.txt
```

### "A test fails with a numerical-precision error like `max abs error: 3.2e-04`"

Open the captured `evidence/test_output.txt` and look at *which* assertion failed. If the error is just slightly above the threshold (say, 2e-4 when threshold is 1e-4) it's probably a float32-vs-float64 issue or a numerical-stability difference. **Send me the full output and I'll either tighten the implementation or loosen the threshold.** Don't paper over it by editing the threshold yourself.

### "Module 3's BPE test fails on `test_parity_with_solution`"

Means your `bpe_starter.py` is not yet filled in. You haven't done the exercise. That's expected — for closure purposes, run `USE_SOLUTION=1 python test.py` which tests the reference solution (which works). Filling in `bpe_starter.py` yourself is also part of the work, but it's "do the exercise" not "Phase 0 closure."

### "Module 7 capstone training runs slowly / loss isn't dropping"

- Check `train loss` at step 0 is approximately `ln(65) ≈ 4.17`. If it's way different, the model is initialized wrong.
- If loss is stuck around 4.0, the optimizer is broken (probably grad-clipping or lr issue).
- If loss spikes to NaN, lower the lr (`python train.py --lr 1e-3` instead of the default `3e-3`).

### "I want a faster training run"

```bash
python train.py --steps 500 --eval_every 50
```

Won't converge as well but lets you see the curve shape in ~1 min.

### "I have a Mac with Apple Silicon and want to use the GPU"

```bash
python train.py --device mps
```

Should be ~3× faster than CPU. (Linux NVIDIA: `--device cuda`.)

### Anything else

Copy the actual error and tell me. **Don't sandbag** by saying "it didn't quite work" — the specific error is what I need.

---

## Part 5 — Phase 0 closes when…

All four of these are true:

- ☐ `evidence/test_output.txt` exists and shows ✅ in every Phase 0 module (1, 2, 3, 4, 5, 6, 7).
- ☐ `07_phase0_capstone/evidence/loss_curve.png` exists and the final train loss is below 2.0 nats.
- ☐ Every Phase 0 module's `hand_math/` folder has at least one derivation file (`.jpg` or `.md`).
- ☐ You passed the cold quiz (8 questions, grade ≥ 6/8 with no R-001/R-002/R-004 fail — those three are non-negotiable).

When all four are checked, I update [`PROGRESS.md`](PROGRESS.md), bump skill levels in [`SKILLS.md`](SKILLS.md), write the Phase-0-close devlog entry, and open Phase 1.

---

## Part 6 — What Phase 1 looks like (so you know where you're heading)

Phase 1 = **real training & fine-tuning on Colab**. Five modules:

1. nanoGPT-style pretrain on tinyshakespeare with proper training hygiene (warmup + cosine LR, gradient clipping, bf16, eval intervals, checkpointing).
2. Scale up — pretrain a ~30M param model on a slice of OpenWebText. A T4 day. Watch a real model emerge.
3. **Debugging non-convergence** — I deliberately break the training run three ways. You diagnose each from loss-curve / gradient-norm / param-norm signals alone, in under 15 min. This is the skill that separates "can run a training loop" from "can train models."
4. LoRA fine-tuning from scratch (not via `peft` — implement the low-rank decomposition yourself).
5. Inference-quality probes — perplexity, top-k sample quality, eval-set construction.

We'll need to talk about your compute access before Phase 1.2 — Colab T4 is free; Phase 1.2 (the OpenWebText pretrain) takes ~24h of T4 time which is more than Colab's free tier comfortably allows. Lambda Labs A100 at $1/hr works; we'll plan when we get there.

---

**TL;DR for fast scanning:**

```
1. Activate venv + pip install -r requirements.txt
2. Run all the parity tests, capture outputs to evidence/, commit
3. Run capstone training (3-5 min), commit the loss curve + samples
4. Write at least one hand_math derivation per module (paper or LaTeX)
5. Come back to me and ask for the cold quiz
```

That's it. Welcome to Phase 0 closure.
