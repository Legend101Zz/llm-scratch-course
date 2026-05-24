# 2026-05-23 · 02 — Foundation gap closure (and an honest audit)

> **Phase:** 0 (closure)
> **Session length:** open-ended, immediately after session 01
> **Outcome:** Phase 0 is fully scaffolded. Three real gaps closed: Module 2 parity test, `hand_math/` + `evidence/` folders across all Phase-0 modules, and the Phase 0 capstone (`07_phase0_capstone/`).

## What I set out to do

End of session 01 I asked the mentor: *"is the foundation course actually nice and detailed?"* Casual-sounding question, but it was a test. The mentor was supposed to answer honestly (Hard Rule #6: no flattery), which meant actually auditing what was on disk against the discipline conventions I'd just written into `CONVENTIONS.md`.

Two outcomes were possible: (a) "yes it's great, ship it" — which I would have known to distrust — or (b) "here's what's actually missing." The mentor dispatched an Explore subagent to grade the work without knowing who built it, then came back with a brutally specific gap list:

- **Module 2 had no parity test** — the first multi-layer backprop in the course, the place where Module 1's autograd most needs verification, and there was no `test.py` proving the gradients matched PyTorch.
- **No `hand_math/` or `evidence/` folders existed anywhere** — CONVENTIONS.md demanded both per-module, but the discipline was *promised in writing* and not *enforced on disk*.
- **`07_phase0_capstone/` didn't exist** — COURSE_MAP.md described it as the Phase 0 closure deliverable, listed its contents, but never built it.

That last one was the one that stung. Phase 0 was claimed to be 🟨 in progress with all the components done — but the *integration* of those components into a working LM hadn't happened. So I picked option A: close the gaps before committing anything.

## What I actually built

- **Module 2 parity test** (`02_neural_net/test.py`). Three checks: (1) forward output matches a PyTorch MLP with identical weights to 1e-6 on every XOR input; (2) per-parameter gradients of MSE loss match to 1e-5 — every weight, every bias, every layer; (3) the model converges to loss < 0.05 with correct rounded predictions after 500 SGD steps. This is the test I should have written in session 01; better late than after Phase 0 closure.

- **`hand_math/` and `evidence/` folders in Modules 1–6 + the capstone.** Twelve folders, each with a README that specifies exactly what derivations / artifacts go there. They're empty of *content* (because the actual hand-derivations are work I have to do, not the mentor) but the *structure* is now real — git will track what lands when.

- **The capstone, `07_phase0_capstone/`.** Five files: `numpy_gpt.py` (the assembled forward-only NumPy GPT — 112k params, two blocks, four heads, d=64, V=65 chars), `torch_gpt.py` (the same architecture in PyTorch, trainable), `test.py` (parity proof — identical weights, forward matches to 1e-4, full-stack causal invariant holds), `train.py` (2000-step training on tinyshakespeare with loss curve + sample generation), and a README that explains the whole assembly. The NumPy GPT runs end-to-end (verified in this session — 112k params, NLL≈ln(V) at init as expected).

- **Memory file updates.** `PROGRESS.md`, `COURSE_MAP.md`, `SKILLS.md`, `MENTOR_LOG.md` all updated to reflect the new state. The capstone bumps "Full GPT" from skill-level 1 to 3.

## The wall I hit

Two real ones.

**The Value-class identity bug.** First attempt at Module 2's parity test failed mysteriously: `TypeError: float() argument must be a string or a real number, not 'Value'`. Traced it: when my test file loaded the `Value` class via `importlib.util.spec_from_file_location("autograd_solution", ...)`, and the MLP's solution.py *also* loaded it via `spec_from_file_location("autograd_solution", ...)`, **Python produced two different `Value` classes**. The MLP's neuron weights were instances of one class; the test inputs I created were instances of the other. So inside `Value.__mul__`, `isinstance(other, Value)` returned False — and the fallback `Value(other)` tried `float(value_instance)` which fails. Fix: import Value from the MLP module itself, which already does the loading once and exposes Value at module level. This is the kind of footgun that bites once and stays in muscle memory forever.

**The "should we actually train in NumPy" decision.** COURSE_MAP.md's original spec for the capstone said "trains on tinyshakespeare with NumPy autograd — slow but real." I drafted it that way, started sketching the code, and stopped. Module 1's `Value` is scalar autograd; one forward pass through a 2-layer GPT at T=64, d=64 hits roughly 200,000 scalar ops. Each op is a Python object creation. Backward through that takes 30+ seconds per step. 2000 steps would be 17 hours. That's not "slow but real" — that's "actively wrong as a teaching tool."

So I retreated and reframed: the capstone is *forward-only in NumPy* (proves the assembly works, parity-checked against PyTorch). Training happens in PyTorch (we've earned the right to trust it — Modules 1, 4, 5 each ship a parity test proving torch's arithmetic matches ours). The pedagogical line in the capstone README is: "Module 1 taught the algorithm; PyTorch's parity tests earned the right to use vectorized autograd; now we train." That's a more honest division of labor than pretending scalar autograd scales.

## The aha

The audit pattern. The mentor agent could easily have said "yes, great work, ship it" — that's the lowest-resistance path when the user just spent an hour writing a hundred files. Dispatching the audit to an independent subagent removed that temptation because the subagent hadn't done the work; it had no ego to protect. The verdict came back: "detailed on paper, partially-executed on disk."

That's exactly the failure mode the new discipline conventions were trying to prevent. CONVENTIONS.md says every module needs `hand_math/` and `evidence/`. If you read CONVENTIONS.md and then look at the modules on disk and find no such folders, the conventions are aspirational documentation, not engineering discipline. The audit-then-close-gaps loop is what turns the latter into the former.

I think this should become the standard rhythm: **before any "Phase X done" promotion, dispatch an audit agent.** The cost is a few minutes; the benefit is that the discipline gets enforced on real files, not on a wishlist.

## What's next

Three things, in order:

1. **Run the tests in my actual venv.** This session ran the numpy-only logic on every test (it passes), but the parts that require `torch` haven't been executed because this machine doesn't have it. I need to `pip install -r requirements.txt`, then run each `python test.py` and capture the output to the respective `evidence/test_output.txt`. Then run `07_phase0_capstone/train.py` and let it train for 2000 steps (~5 min on CPU). The loss curve and generated samples are the closure evidence for Phase 0.

2. **Take the cold quiz.** The mentor posed 8 questions on attention + transformer block at the end of session 01; I paused them to close the gaps first. With the capstone now built, the quiz is the actual gate to Phase 1 — re-explaining things cold means I understand the components I just assembled.

3. **Write the first `hand_math/` derivation.** Pick a module and actually derive its central equation on paper (Module 1's chain rule through `tanh` is probably the cheapest first one). Photo it, commit it. That's the first piece of physical evidence the discipline isn't aspirational.

After all three, Phase 0 closes, Phase 1 begins.

One sentence to remember from this session: **the audit-then-close-gaps loop is the discipline that makes CONVENTIONS.md real**; without it, the conventions are decoration.
