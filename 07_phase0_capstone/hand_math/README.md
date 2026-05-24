# `07_phase0_capstone/hand_math/` — derivations for the assembled model

> Most module-level math derivations live in the individual modules (1, 2, 3, 4, 5). The capstone adds *one* new derivation worth doing.

## What belongs here

### 1. The full-stack causal-mask invariant

Each block has a causal mask (Module 4). When you stack N blocks, does the causal property compose? Prove on paper:

> *"If every block respects causality (i.e. position k's output depends only on positions 0..k of its input), then the entire stack respects causality."*

This is a one-line proof by induction:
- **Base:** Block 1 outputs at position k depend only on inputs at 0..k (causal mask).
- **Step:** Assume block n's output at position k depends only on input positions 0..k. Then block n+1 reads only positions 0..k of block n's output (causal mask again), which by induction depends only on input positions 0..k. ✓

Why bother? Because the test `test_causal_property_on_full_gpt` only checks empirically that *this* model preserves causality. The induction proves it for *any* stack depth — which matters because Phase 1 will scale this to 12+ layers.

Commit this on paper (one page max).

### 2. (Optional, recommended) Cross-entropy → ln(V) baseline derivation

The "random init loss should be ≈ ln(V)" claim from the capstone training:

For a uniform distribution `p(b) = 1/V` over `V` classes, the cross-entropy with any target `t` is:
```
CE = -log p(t) = -log(1/V) = log(V)
```

This means: **at random init, before any learning, the model should output approximately log(V) nats of loss.** If it doesn't (e.g. random init gives 5.0 nats for V=65), the init is wrong — usually too-large std.

Commit the derivation + one line of intuition: "a uniformly-noisy prediction is exactly as informative as random; the loss measures information missing relative to the truth."
