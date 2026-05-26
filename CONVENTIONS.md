# CONVENTIONS.md — the per-module discipline template

> Read this once. Every module in Phase 0 onward follows this template. It exists because frontier-lab readiness is recorded evidence, not "I read it."
>
> Cross-refs: [DECISIONS.md D-0003](DECISIONS.md#d-0003--2026-05-23--course-wide-discipline-conventions-hand_math-evidence-parity-roofline) · [MENTOR.md](../MENTOR.md) Hard Rules · [frontier-lab.md](../frontier-lab.md) "The author's self-consistent ask."

---

## The non-negotiable rules (Hard Rules from MENTOR.md, applied per module)

1. **DOING > WATCHING.** You implement; the mentor reviews. The mentor never reveals `solution.py` until you've genuinely attempted `starter.py`. If you ask for it early, the answer is "here's a smaller first step." (Hard Rule #1.)
2. **From scratch first, framework second.** NumPy / pure Python first, then PyTorch (then JAX in Phases 4–5). The parity test closes the loop. (Hard Rule #2.)
3. **Math derived by hand.** Pencil on paper, photographed or transcribed into the module's `hand_math/` folder. The mentor checks it. No hand-waving past a gradient, softmax, norm, attention, or scaling law. (Hard Rule #3.)
4. **Systems lens by default.** For anything that runs on hardware, a `## Roofline` section in the README answers: FLOPs? Bytes read/written? Arithmetic intensity? Memory-bound or compute-bound? *Before* you optimize. (Hard Rule #4.)
5. **Force recall.** End of every module: re-explain the central concept in your own words, re-derive the central equation. Mentor adds the question to `REVIEW.md` with a spaced-repetition due date. (Hard Rule #5.)
6. **Honest, not flattering.** If the implementation is wrong, the mentor says so. If it's right, the mentor moves on. No reflexive praise. (Hard Rule #6.)
7. **Reproducible.** Every artifact is a committed file. GPU work runs in Colab notebooks the mentor writes; you paste results back into `evidence/`. (Hard Rule #8.)

---

## The module folder layout (canonical)

```
NN_module_name/
├── README.md           ← intuition + math + diagrams + roofline (if applicable) + reflection
├── starter.py          ← skeleton with # TODO comments. You fill it in.
├── solution.py         ← reference impl. Read only after starter passes tests.
├── test.py             ← parity test: from-scratch impl vs. PyTorch (or JAX where applicable)
├── hand_math/          ← your pencil derivations
│   ├── README.md       ← what's in here and what each derivation proves
│   ├── 01_gradient.jpg ← photographed page (or 01_gradient.md if you transcribe to LaTeX)
│   └── ...
├── evidence/           ← outputs that prove the implementation works
│   ├── test_output.txt ← captured stdout from `python test.py`
│   ├── metrics.json    ← numerical comparisons (forward error, grad error, timings)
│   ├── roofline.md     ← worked roofline numbers for this module's main op (if hardware-bound)
│   └── ...
└── notes.md            ← your scratch — confusions, dead ends, "aha" moments
```

`hand_math/`, `evidence/`, and `notes.md` are created by you as you work. The mentor doesn't pre-populate them — that would defeat the point.

---

## What `hand_math/` actually requires

Per module, **at minimum one** of:
- The full derivation of the module's central gradient (e.g. Module 1: ∂L/∂w through `+`, `*`, `tanh`).
- The full derivation of the module's central objective (e.g. Module 4: why scores get scaled by 1/√d_k; e.g. Module 10: the GRPO advantage normalization).
- The full derivation of the relevant scaling law (e.g. Phase 4: Chinchilla N:D from the parametric loss, with the partial derivatives).

**Format.**
1. Pencil on paper. Yes, actually. Pencil enforces slowness.
2. Photo it (`.jpg`/`.png`) **or** transcribe to LaTeX in a `.md` (a chatbot can transcribe if needed — but the *derivation* is yours).
3. Commit the file. The git history is the evidence trail.

The mentor will spot-check by asking you to re-derive a step. If you can't, the derivation didn't stick — repeat it.

**Per `frontier-lab.md`:** for the graded exercises (Phase 4+), be ready to **show video** of you doing random subsets of these derivations on paper. Plan recording from Phase 0.

---

## What `evidence/` actually requires

Per module, **at minimum:**
1. `test_output.txt` — captured stdout from `python test.py` showing the parity assertion passed (or the metric that proves the module works, e.g. "XOR loss → 0.0001" for Module 2).
2. For any hardware-touching module from Module 4 onward: `roofline.md` — a one-page worked calculation of (FLOPs / bytes-read / bytes-written / arithmetic-intensity / bound type) for the module's main op with the chosen (B, T, d, h) shapes. Conclusion sentence: "Memory-bound at AI ≈ X FLOP/byte on a T4 (roofline knee ≈ Y FLOP/byte), so the optimization target is bandwidth, not FLOPs." (Or the opposite. State the conclusion.)
3. For Phases 1+: training logs (loss curves, eval numbers, perplexity / pass@k / whatever the phase grades on).
4. For Phase 4+: profiler output (`torch.profiler`, `nsys`, or Pallas/Triton bench), saved as JSON or text. **Don't** commit huge HTML reports — store a screenshot or a digested summary instead.

Evidence is what gets linked from the public devlog. It also gets pasted into the per-session `REVIEW.md` entry as "did the artifact land."

---

## The PyTorch-parity test pattern

For every from-scratch numerical module (NumPy → PyTorch), `test.py` must:

1. Run your from-scratch implementation on a fixed-seed input.
2. Run the equivalent PyTorch implementation on the same input.
3. Assert numerical match: forward to `1e-5` for float32 / `1e-6` for float64; gradients to the same tolerance (where backward exists).
4. Print both values side-by-side so failures are diagnosable, not just `assertion error`.

Reference template: see [`phase0/01_autograd/test.py`](phase0/01_autograd/test.py) — it compares the `Value` engine to `torch.tensor(requires_grad=True)` on a complex expression and prints `forward / a.grad / b.grad` for both.

**For modules where backward isn't implemented from scratch** (e.g. Module 5's transformer block in NumPy, where we only do forward): the parity test runs forward only, but loads the *same weights* into a PyTorch reference and checks the forward output. The PyTorch reference is what eventually becomes the Module 7 implementation, so writing it now is not wasted.

**For JAX modules (Phase 4+):** the parity test goes the *other* direction — JAX impl vs. PyTorch reference impl, same seed.

---

## The roofline section template (for any hardware-bound module)

Drop this section in the README **after** the math derivation, **before** the reflection questions. For Module 4 (attention), Module 5 (block), Module 7 (full GPT), Phase 1+ training modules, and every Phase 4+ kernel.

```markdown
## Roofline — is this memory-bound or compute-bound?

Assume `B=1, T=512, d=64, h=8, head_dim=8` (small enough to fit on a T4), `fp32`.

**FLOPs (one forward pass of this op):**
- Q,K,V projections: 3 × T × d × d = ...
- Attention scores: T × T × d_k = ...
- Output projection: T × d × d = ...
- **Total: ~X GFLOPs**

**Bytes read/written (HBM traffic):**
- Q, K, V tensors: 3 × T × d × 4 bytes = ...
- Scores: T × T × 4 bytes = ... (this is the big one)
- Output: T × d × 4 bytes = ...
- **Total: ~Y MB**

**Arithmetic intensity:** AI = FLOPs / bytes = X / Y FLOP/byte.

**T4 roofline knee:** ~8.1 TFLOPS peak / 320 GB/s HBM = **25 FLOP/byte**.

**Conclusion:** AI ≈ Z FLOP/byte vs. knee at 25. If Z < 25, the op is **memory-bound** — the optimization target is bandwidth (kernel fusion, tiling, KV cache). If Z > 25, **compute-bound** — the optimization target is FLOPs (lower precision, less wasted math).

**For attention specifically:** as T grows, the scores tensor (T×T) dominates the byte count linearly while FLOPs grow quadratically — so attention is compute-bound at large T but memory-bound at small/medium T. This is the FlashAttention insight earned, not told: the *intermediate* scores matrix is what hits HBM, so fusing softmax + matmul to keep scores in SRAM is the win.
```

Future-you will skim a hundred of these. Make them comparable across modules.

---

## The reflection questions (per module)

End every README with 3–5 reflection questions. The mentor uses them to:
- Add a spaced-repetition entry to `REVIEW.md` with a due date (1 day → 3 days → 1 week → 1 month → 3 months).
- Quiz you cold at the start of subsequent sessions.

Questions must be **non-Googleable** — they require you to reason about the implementation you just built. Not "what is softmax?" but "why does softmax in attention go over keys, not queries, and what would break if you flipped it?"

---

## Recording the work (Feinberg's evidence ask)

Per `frontier-lab.md`:

> Record yourself doing the scaling-book exercises with pencil & paper (all of them); have a chatbot transcribe to LaTeX. Be ready to show video of random subsets. Screen-record yourself manually writing the transformer code and deriving the Chinchilla laws.

**Convention.** From Phase 0 onward:
- Webcam over the paper for any hand-math session. Save the video locally (don't push to git — too large), but commit a screenshot or a transcribed LaTeX block to `hand_math/`.
- Screen-recording for any from-scratch implementation session. Same deal: video stays local, but commit a final screenshot or an `asciinema` cast if the session was short.

Practice this discipline early. By the time the Phase 4 graded exercises arrive, "be ready to show video" should be a habit, not a panic.

---

## A note on AI assistance

You can use a chatbot to:
- Transcribe handwritten LaTeX.
- Decode an error message you don't understand *after* you've stared at it for ≥10 minutes.
- Suggest a refactor *after* the implementation works.
- Cross-check a derivation you've finished.

You **may not** use a chatbot to:
- Write the implementation. (Hard Rule #1.)
- Write the derivation. (Hard Rule #3.)
- Write the reflection questions for yourself. (That defeats spaced repetition.)
- Write the parity test for you. The test is part of the discipline.

The mentor agent (this Claude session) is excluded from these prohibitions but bound by its own constraints: it will only scaffold, hint, review, and quiz — it will not produce finished implementations on demand. Begging will be refused; the response will be a smaller first step.

---

## How to know a module is done

A module is *done* when **all five** are true:

1. `starter.py` passes its own assertions when run by you (not the solution).
2. `test.py` passes (parity with PyTorch / JAX).
3. `hand_math/` contains at least one derivation for this module's central equation.
4. `evidence/` contains `test_output.txt`, `metrics.json`, and (if applicable) `roofline.md`.
5. You can re-explain the central concept *cold* the next session. (Mentor verifies via quiz.)

If any of 1–5 are missing, the module is "in progress" — `PROGRESS.md` reflects that.
