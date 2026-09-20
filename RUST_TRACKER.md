# RUST_TRACKER.md — the day checklist for R0a and R0b

> Written in [Simple English](.claude/skills/simple-english/SKILL.md) (ASD-STE100).
> This file is the **checklist**. [`RUST_PHASE_0_1.md`](RUST_PHASE_0_1.md) is the **design**.
> [`days/DAY_NN.md`](days/) is the **lesson** — one self-contained file per day, with the reading built in.
> Tick the boxes here. Read the lesson there.
> **Lessons and tests for Days 1 to 14 are written.** Days 8 to 14 were written ahead on 4 September 2026, at my explicit request. Day 15 and later come one at a time.
> [`days/AUTHORING.md`](days/AUTHORING.md) is how the mentor writes them, and what is still open in the ones written ahead.

**Active claim: R0a.** I can write a strided, broadcast tensor library in Rust with a cache-blocked multithreaded matmul. I can prove it correct with no reference implementation.

**Timebox: 2 weeks. 9 working days. Day 7 takes 2 days.**

**Timebox start date: `11th August 2026`** .

| Week       | Days | Claim | What lands                                                                  |
| ---------- | ---- | ----- | --------------------------------------------------------------------------- |
| **Week 1** | 1–6  | R0a   | Trait, PRNG, strides, views, broadcast, reductions, the naive matmul oracle |
| **Week 2** | 7–8  | R0a   | Blocked matmul, scoped threads, the GFLOP/s table, **Gate R0a**             |
| **Week 3** | 9–14 | R0b   | The tape, backward, the gradient checker, cross-entropy, AdamW, **Gate R0b** |

This file covers **R0a and R0b**. R0b starts below Gate R0a. Do not open it before that gate is closed.

---

## Day 0 — setup (15 minutes, before Day 1)

Do this once. Do not skip the `criterion` line, because Day 7 needs it.

**Crate location: `course/rust/`. One library crate. The package name is `rustgpt`.**
One crate holds R0a, R0b and R1. Binaries arrive at R1 under `src/bin/`.

- [x] Run `rustc --version`. Confirm 1.91 or newer.
- [x] From `course/`, run `cargo new --lib --name rustgpt rust`.
- [x] Add `criterion = "0.5"` under `[dev-dependencies]` in `rust/Cargo.toml`.
- [x] Add `rust/target/` to `course/.gitignore`.
- [x] Run `cargo test` inside `rust/`. Confirm it builds and reports 0 tests.
- [x] Run `cargo clippy`. Confirm it is clean.
- [x] Write the current date in the **Timebox start date** line above.
- [x] Commit the empty crate. The first commit is the start of the trail.

> **CAUTION: Do not add any other dependency.** Banned for R0a, R0b and R1: `ndarray`, `nalgebra`, `candle`, `burn`, `tch`, `tokenizers`, any BLAS, any autodiff crate. `criterion` is the one exception, and it is a dev-dependency.

---

## Every session — the ritual

This block applies to every day below. It is here once, not six times.

**"You" is always Mrigesh.** Where the actor is the mentor, the step names Claude.

1. [ ] Open `PROGRESS.md` and this file. Claude states the active claim and the days left.
2. [ ] Claude quizzes you cold on the due `REVIEW.md` items and the concept of the prior day.
3. [ ] Read the prereq pages **before** you write code. The pages are on the day card.
4. [ ] Move `rust/tests/pending/dayN.rs` to `rust/tests/dayN.rs`. **Claude writes the tests. You write the code.**
5. [ ] Run the tests. Watch them fail. A test that never failed proves nothing.
6. [ ] Write the code. Make the tests pass.
7. [ ] Do the self-check on paper. Photograph the paper into `hand_math/`.
8. [ ] Record the session. Post the day hook on LinkedIn.
9. [ ] Claude updates `PROGRESS.md`, the devlog, and `REVIEW.md`.
10. [ ] Claude commits and pushes.

**The rule that makes the tests real:** write the test, watch it fail, then write the code. A test written after the code passes for the wrong reason.

---

# WEEK 1 — Days 1 to 6

## ☐ Day 1 — The `Scalar` trait and a PRNG you own · 2.5 h

📖 **[Full lesson — days/DAY_01.md](days/DAY_01.md)** · [Day card](RUST_PHASE_0_1.md#day-1--the-scalar-trait-and-a-prng-you-own)

- [ ] **Read (25 min).** _Programming Rust_ pp. 235–252. Skim only.
- [ ] Create `src/scalar.rs` and `src/rng.rs`. Declare both in `src/lib.rs`.
- [ ] Write the `Scalar` trait with the supertrait bounds and the two constants.
- [ ] Implement `Scalar for f32`.
- [ ] Implement `Scalar for f64`.
- [ ] Write `Rng::seed`. Reject a seed of 0.
- [ ] Write `Rng::next_u64` as xorshift64\*.
- [ ] Write `Rng::uniform` and `Rng::normal`. Use Box-Muller for `normal`.

**Tests**

- [ ] `same_seed_same_sequence`
- [ ] `uniform_in_range`
- [ ] `normal_moments`
- [ ] `scalar_generic_compiles`

**Self-check (on paper)**

- [ ] **Derive** why `z = sqrt(-2 ln u₁) · cos(2π u₂)` is standard normal. Start from two independent normals in polar coordinates.
- [ ] Answer why the trait needs `from_f64` **and** `to_f64` instead of a `From`/`Into` bound.

> **Stuck-signal.** If one generic function fights `the trait bound X is not satisfied` for 40 minutes, ask. That is a bounds-design problem, not a syntax problem.

- [ ] Session ritual steps 8 to 10 are done.

---

## ☐ Day 2 — Storage, shape, strides · 2.5 h

📖 **[Full lesson — days/DAY_02.md](days/DAY_02.md)** · [Day card](RUST_PHASE_0_1.md#day-2--storage-shape-strides)

- [ ] **Read (20 min).** _Programming Rust_ pp. 57–63 and pp. 90–92.
- [ ] Create `src/tensor.rs`.
- [ ] Write the `Tensor<T>` struct with `data`, `shape`, `strides` and `offset`.
- [ ] Write `contiguous_strides`.
- [ ] Write `zeros` and `from_vec`. Panic on a length mismatch.
- [ ] Write `shape`, `numel` and `is_contiguous`.
- [ ] Write `phys_index`. This function is the core of the whole library.
- [ ] Write `get`.

**Tests**

- [ ] `strides_row_major`
- [ ] `phys_index_matches_manual`
- [ ] `from_vec_rejects_bad_len`
- [ ] `numel_is_shape_product`

**Self-check (on paper)**

- [ ] **Trace** shape `[2,3,4]`. Write the strides. Find the buffer index of element `[1,0,2]`. Transpose the last two axes. Write the new strides. Confirm that the buffer does not move.
- [ ] Answer why `Rc<Vec<T>>` and not `Vec<T>`. State what each choice makes impossible.

> **Stuck-signal.** If you want to write `data: Vec<Vec<T>>`, stop. That is the Python model. Ask before you commit to it.

- [ ] Session ritual steps 8 to 10 are done.

---

## ☐ Day 3 — Zero-copy views · 3 h

📖 **[Full lesson — days/DAY_03.md](days/DAY_03.md)** · [Day card](RUST_PHASE_0_1.md#day-3--zero-copy-views-reshape-permute-transpose-slice)

- [ ] Move `tests/pending/day3.rs` to `tests/day3.rs`. Run it. Watch it fail.
- [ ] Write `strides()` and `shares_storage_with()`. The lesson section 4.1 says why.

- [ ] **Read (20 min).** _Programming Rust_ pp. 148–158.
- [ ] Write the `ShapeError` enum with its four variants.
- [ ] Write `reshape`. Return `Err(NotContiguous)` instead of a silent copy.
- [ ] Write `permute`.
- [ ] Write `transpose`.
- [ ] Write `slice`.
- [ ] Write `contiguous`. This is the only method that copies.
- [ ] Write `to_vec`. Walk the strides in logical order.

**Tests**

- [ ] `permute_roundtrip`
- [ ] `transpose_is_zero_copy`
- [ ] `reshape_noncontiguous_errors`
- [ ] `contiguous_then_reshape_ok`
- [ ] `slice_bounds`

**Self-check (on paper)**

- [ ] **Trace** shape `[2,3]` with buffer `[0,1,2,3,4,5]`. Transpose to `[3,2]`. Write the new strides. Write the elements in reading order. Show why a plain `reshape` to `[6]` gives the wrong sequence.
- [ ] Answer why `reshape` returns `Result`. Use the cost of a hidden copy inside an attention loop.

> **Stuck-signal.** If `permute` passes on shape `[2,3]` and fails on `[2,3,4]`, ask. That is a rank-generality fault in the index permutation.

- [ ] Session ritual steps 8 to 10 are done.

---

## ☐ Day 4 — Broadcast and elementwise ops · 3 h

📖 **[Full lesson — days/DAY_04.md](days/DAY_04.md)** · [Day card](RUST_PHASE_0_1.md#day-4--broadcast-and-elementwise-ops)

- [ ] Move `tests/pending/day4.rs` to `tests/day4.rs`. Run it. Watch it fail.

- [ ] **Read (25 min).** _Programming Rust_ pp. 303–312 and pp. 330–344.
- [ ] Write `broadcast_shapes`. Align the shapes from the trailing axis.
- [ ] Write `broadcast_to`. Give the broadcast axis **stride 0**.
- [ ] Write `map`.
- [ ] Write `zip_with`.
- [ ] Build `add`, `sub`, `mul`, `div` and `neg` from `map` and `zip_with`.
- [ ] Build `exp`, `ln`, `sqrt`, `tanh` and `relu` the same way.

**Tests**

- [ ] `broadcast_shapes_table`
- [ ] `broadcast_uses_stride_zero`
- [ ] `broadcast_sum_scales`
- [ ] `elementwise_against_manual`

**Self-check (on paper)**

- [ ] **Derive** the broadcast of shapes `[8,1,6]` and `[7,1]`. Write both stride vectors of the broadcast views.
- [ ] A stride-0 view has a `numel()` larger than its buffer. Name which Day 2 and Day 3 methods break. Name which ones you already guarded.

> **Stuck-signal.** If `to_vec()` on a broadcast view gives the correct length but repeats the wrong elements, ask. Your stride walk is correct. Your trailing-axis alignment is off by one.

- [ ] Session ritual steps 8 to 10 are done.

---

## ☐ Day 5 — Reductions · 3 h

📖 **[Full lesson — days/DAY_05.md](days/DAY_05.md)** · [Day card](RUST_PHASE_0_1.md#day-5--reductions)

- [ ] Move `tests/pending/day5.rs` to `tests/day5.rs`. Run it. Watch it fail.

- [ ] **Read (20 min).** _Programming Rust_ pp. 345–354.
- [ ] Write `sum_axis` with `keepdim`.
- [ ] Write `mean_axis`.
- [ ] Write `max_axis`.
- [ ] Write `sum_all`. Use pairwise summation, not a sequential fold.

**Tests**

- [ ] `sum_axis_shapes`
- [ ] `sum_axis_values`
- [ ] `sum_all_precision` ← **this test is the lesson of the day**
- [ ] `max_axis_handles_negatives`

**Self-check (on paper)**

- [ ] **Numerical trace.** In `f32`, set an accumulator to `1.6e7`. Add `1.0`. Work out the exponent and the 24-bit mantissa. State what happens. Then explain why pairwise summation of 10⁸ ones survives.
- [ ] Write the identity that shows `softmax(x) == softmax(x − c)` for any scalar `c`. State why softmax needs `max_axis`.

> **Stuck-signal.** If the reduction is correct for `axis=0` and wrong for the last axis, ask. The fault is the multi-index iteration order, not the fold.

- [ ] Session ritual steps 8 to 10 are done.

---

## ☐ Day 6 — The naive matmul, and the oracle discipline · 3 h

📖 **[Full lesson — days/DAY_06.md](days/DAY_06.md)** · [Day card](RUST_PHASE_0_1.md#day-6--the-naive-matmul-and-the-oracle-discipline)

- [ ] Move `tests/pending/day6.rs` to `tests/day6.rs`. Run it. Watch it fail.

Today you write the slowest correct matmul. **You keep it for the whole project.** It is the oracle for every faster version.

- [ ] **Read (15 min).** _Programming Rust_ pp. 178–182.
- [ ] Create `src/matmul.rs`.
- [ ] Write `matmul_naive`. Three nested loops. No cleverness.
- [ ] Add the comment `/// Reference implementation. NEVER optimise this. It is the oracle.`
- [ ] Write `matmul`, the batched version. Broadcast the batch dimensions.

**Tests**

- [ ] `matmul_2x3_3x2_by_hand`
- [ ] `matmul_identity`
- [ ] `matmul_transpose_identity`
- [ ] `matmul_associative`
- [ ] `matmul_batched_shapes`
- [ ] `matmul_dim_mismatch_errors`

**Self-check (on paper)**

- [ ] **Derive** the FLOP count for `[M,K] · [K,N]`. Then count the minimum bytes moved in `f32`. Then compute the arithmetic intensity in FLOP per byte at M=N=K=512. Against ~500 GFLOP/s and ~100 GB/s, state whether this is compute-bound or memory-bound.
- [ ] Six loop orderings exist for `i, j, k`. Name the one with the best cache behaviour for row-major storage. **Write the prediction down. You measure it tomorrow.**

> **Stuck-signal.** If batched matmul passes for equal batch sizes and fails when one is 1, ask. The fault is the batch-stride computation. Do not add a special case.

- [ ] Session ritual steps 8 to 10 are done.

---

### End of Week 1 — check before you go on

- [ ] `cargo test` is green.
- [ ] `cargo clippy` is clean.
- [ ] Six LinkedIn posts are shipped.
- [ ] Six hand-derivations sit in `hand_math/`.
- [ ] `PROGRESS.md` names the real state, not the intended state.

---

# WEEK 2 — Days 7 to 8, then the gate

## ☐ Day 7 — Blocked matmul, and the measured gap · 5 h across **2 days**

📖 **[Full lesson — days/DAY_07.md](days/DAY_07.md)** · [Day card](RUST_PHASE_0_1.md#day-7--blocked-matmul-and-the-measured-gap--2-day-card)

- [ ] Move `tests/pending/day7.rs` to `tests/day7.rs`. Run it. Watch it fail.

**Predict before you measure.** Write the predicted best block size on paper first. The prediction is part of the exit artifact.

- [ ] **Read (20 min).** _Programming Rust_ pp. 161–165. Then skim the criterion "Getting Started" page.
- [ ] Compute the predicted best block size from the 128 KB L1D of the M4. Write the number down.
- [ ] Write `matmul_blocked`.
- [ ] Add the `[[bench]]` section to `Cargo.toml` with `harness = false`.
- [ ] Write the criterion benchmark. Use `black_box`.
- [ ] Benchmark in `--release`. A debug benchmark is meaningless.
- [ ] Measure GFLOP/s across block sizes 8, 16, 32, 64 and 128, at N of 128, 512 and 1024.
- [ ] Commit the table to `evidence/`.

**Tests**

- [ ] `blocked_matches_naive` — includes 129×257 by 257×63. Tolerance 1e-4.
- [ ] `blocked_matches_naive_f64` — the same sizes. Tolerance 1e-10.

**Self-check (on paper)**

- [ ] **Derive** how many bytes of `A`, `B` and `C` stay live in the innermost blocked loop for block size `b`. Set that equal to 128 KB. Solve for the largest `b` that holds three `f32` tiles. Compare the predicted best against the measured best. If they disagree, state why.
- [ ] State the percent of the ~550 GFLOP/s fp32 peak that you reached. If it is below 15 percent, name the next bottleneck. State whether blocking can fix that bottleneck at all.

> **Stuck-signal.** If blocked runs slower than naive, ask. The cause is almost always a benchmark in debug mode, a block larger than L1, or a bounds check in the inner loop. Do not rewrite first.

- [ ] Session ritual steps 8 to 10 are done.

---

## ☐ Day 8 — Parallelism with scoped threads · 2.5 h

📖 **[Full lesson — days/DAY_08.md](days/DAY_08.md)** · [Day card](RUST_PHASE_0_1.md#day-8--parallelism-with-scoped-threads)

- [ ] Move `tests/pending/day8.rs` to `tests/day8.rs`. Run it. Watch it fail.

- [ ] **Read (30 min).** _Programming Rust_ pp. 457–466. Read the Rayon section to learn what you leave out.
- [ ] Decide the `Rc` question. `Rc` is not `Send`, so `&Tensor<T>` cannot enter a thread. Lesson section 4.2 gives the two fixes. Record the choice in the commit message.
- [ ] Write `matmul_parallel`. Partition the output **by rows**.
- [ ] Use `std::thread::scope`. Use no `Arc` and no `Mutex`.
- [ ] Predict the speedup at 10 threads against 4 threads. Write the number down.
- [ ] Measure the speedup against thread count at N=1024.
- [ ] Commit the speedup table to `evidence/`. State where scaling stops and why.

**Tests**

- [ ] `parallel_matches_blocked` — **bit-identical**, not approximate.
- [ ] `parallel_thread_count_invariant` — 1, 2, 4 and 8 threads.

**Self-check (on paper)**

- [ ] State why a partition of the output by rows is safe with no `Mutex`. State why a partition along `K` is not safe. Answer with which threads write which bytes.
- [ ] Compare your 10-thread prediction against the measurement. Explain the gap through the P-core and E-core split.

> **Stuck-signal.** If the compiler says a closure `may outlive the current function`, check which function you called. That is `thread::spawn`, not `thread::scope`. If `scope` still rejects the closure, ask.

- [ ] Session ritual steps 8 to 10 are done.

---

## ☐ GATE R0a — do not start Day 9 until all four are true

- [ ] Property tests pass. `blocked_matches_naive` passes at non-multiple sizes.
- [ ] `parallel_matches_blocked` is bit-identical and thread-count invariant.
- [ ] The GFLOP/s table is committed. It states percent of peak, and predicted against measured block size.
- [ ] The R0a LinkedIn post is shipped.

**A claim without its artifact is not done.** When all four boxes are ticked, mark R0a ✅ in `PROGRESS.md`. Then open the R0b section below.

---

# WEEK 3 — R0b · Days 9 to 14, then the gate

**Active claim after Gate R0a: R0b.** I can build a reverse-mode autograd in Rust and prove every gradient correct against an independent numerical oracle.

**Timebox: 1.5 weeks. 7 working days. Day 9 takes 2 days.**

> Do not open this section until all four Gate R0a boxes above are ticked.

---

## ☐ Day 9 — The tape: autograd architecture · 5 h across **2 days**

📖 **[Full lesson — days/DAY_09.md](days/DAY_09.md)** · [Day card](RUST_PHASE_0_1.md#day-9--the-tape-autograd-architecture--2-day-card-the-hardest-one)

**Day one is paper. Do not type until step 5 of the lesson's order of work.**

- [ ] Move `tests/pending/day9.rs` to `tests/day9.rs`. Run it. Watch it fail.
- [ ] **Read (30 min).** _Programming Rust_ pp. 114–122 and pp. 211–218. **Read p. 121 twice.**
- [ ] Write the tape for the self-check expression on paper, with the backward walk.
- [ ] Create `src/autograd.rs`. Declare it in `src/lib.rs`.
- [ ] Write `NodeId` with `Clone, Copy, PartialEq, Eq, Debug`, and `NodeId::index`.
- [ ] Write `Op` with `Clone, Debug, PartialEq`.
- [ ] Write `Tape` with the four parallel `Vec`s and the invariant comment.
- [ ] Write `new`, `len`, `is_empty`, `leaf`, `value`, `grad`, `op`, `requires_grad`.
- [ ] Write `push`, with the `requires` rule computed from the inputs.
- [ ] Write the forward helpers `add` and `mul`.
- [ ] Write `zero_grad`. Leave `backward` as `todo!()`.

**Tests**

- [ ] `tape_records_in_order`
- [ ] `nodeid_is_copy`
- [ ] `value_roundtrip`

**Self-check (on paper)**

- [ ] **Build and walk** the tape for `f = (a + b) * a` with `a = 2`, `b = 3`. Every node, its op, its inputs, its value. Then `∂f/∂a` symbolically and by a backward walk. **Keep the paper. It is Day 10's first test.**
- [ ] Answer why `backward` can require a scalar root, what differentiation of a vector output means, and what you must supply instead.

> **Stuck-signal.** If you spent an hour on `Rc<RefCell<Node>>` and now hit `already mutably borrowed`, stop. That is the wall. Switch to the arena, or ask.

- [ ] Session ritual steps 8 to 10 are done.

---

## ☐ Day 10 — Backward for elementwise and broadcast · 3 h

📖 **[Full lesson — days/DAY_10.md](days/DAY_10.md)** · [Day card](RUST_PHASE_0_1.md#day-10--backward-for-elementwise-and-broadcast)

- [ ] Move `tests/pending/day10.rs` to `tests/day10.rs`. Run it. Watch it fail.
- [ ] **Read (20 min).** _Programming Rust_ pp. 221–233.
- [ ] Write the forward helpers `neg`, `exp`, `ln`, `tanh`, `sum_axis`, `broadcast_to`, `sum_all`.
- [ ] Write `accumulate`, with the shape assertion. Use `+=`, never `=`.
- [ ] Write `unbroadcast`, with the four-step procedure from lesson section 2.6.
- [ ] Write the `backward` loop: the scalar check, the seed, the reverse range, the two skips.
- [ ] Fill the arms for `Leaf`, `Add`, `Neg`, `Mul`, `Exp`, `Ln`, `Tanh`, `Sum`, `Broadcast`.
- [ ] Confirm the `match` has **no** `_ =>` arm.

**Tests**

- [ ] `backward_matches_paper`
- [ ] `diamond_accumulates` ← **this test is the lesson of the day**
- [ ] `broadcast_backward_sums`
- [ ] `zero_grad_clears`
- [ ] `sum_backward_restores_axis`

**Self-check (on paper)**

- [ ] **Derive** `∂L/∂b₁` for `y = broadcast([b₁,b₂], [3,2])` and `L = sum(y)`. Write all six terms. Then state the general rule in one sentence.
- [ ] Draw the **smallest** graph where assignment instead of accumulation gives a wrong answer. State the wrong number and the right one.

> **Stuck-signal.** If `diamond_accumulates` fails with exactly `2x` or exactly `1`, that is the `=` against `+=` fault. Fix it yourself. Any other value, ask.

- [ ] Session ritual steps 8 to 10 are done.

---

## ☐ Day 11 — Backward for matmul · 3 h

📖 **[Full lesson — days/DAY_11.md](days/DAY_11.md)** · [Day card](RUST_PHASE_0_1.md#day-11--backward-for-matmul)

- [ ] Move `tests/pending/day11.rs` to `tests/day11.rs`. Run it. Watch it fail.
- [ ] **Do both self-checks BEFORE you write the arm.** The order is the point of the day.
- [ ] **Read (20 min).** Raschka section 3.4, pp. 64–70. For the shapes only.
- [ ] Write the `matmul` forward helper.
- [ ] Write the `MatMul` backward arm, rank 2 first.
- [ ] Add the batch handling. Transpose the **last two axes only**.
- [ ] Reduce the batch gradients with `unbroadcast`. Add no special case for batch 1.
- [ ] Confirm no `.contiguous()` sits in the backward path, or comment which kernel demands it.

**Tests**

- [ ] `matmul_backward_shapes`
- [ ] `matmul_backward_2x2_by_hand`
- [ ] `matmul_backward_batched`

**Self-check (on paper)**

- [ ] **Derive from shapes only.** Every arrangement of two of `A`, `B`, `∂L/∂C` giving `[M,K]`. Show exactly one has agreeing inner dimensions. Then the same for `∂L/∂B`.
- [ ] **Derive properly** from `Cᵢⱼ = Σₖ Aᵢₖ Bₖⱼ`, in index form. State what the shape argument alone could not have told you.

> **Stuck-signal.** If shapes are right, values are wrong, and a transpose makes a different test fail, ask. That is a batch-dimension fault, not a transpose fault.

- [ ] Session ritual steps 8 to 10 are done.

---

## ☐ Day 12 — The gradient checker · 3 h

📖 **[Full lesson — days/DAY_12.md](days/DAY_12.md)** · [Day card](RUST_PHASE_0_1.md#day-12--the-gradient-checker)

- [ ] Move `tests/pending/day12.rs` to `tests/day12.rs`. Run it. Watch it fail.
- [ ] **Read (15 min).** _Programming Rust_ pp. 312–319.
- [ ] Create `src/gradcheck.rs`. **`f64` only. No generic parameter.** Say why in the file header.
- [ ] Write `GradCheckReport` and `compare_grads`. Start here.
- [ ] Write `analytic_grads`, then `numeric_grads`. A fresh copy and a fresh tape per perturbation.
- [ ] Write `grad_check`, then the `weights` path, then `grad_check_projected`.

**Tests**

- [ ] `gradcheck_passes_for_correct_ops`
- [ ] `gradcheck_catches_injected_bug` ← **until this is red for a wrong gradient, nothing else today is proven**
- [ ] `gradcheck_projected_catches_sign_flip`

**Self-check (on paper)**

- [ ] **Derive** the central-difference error `O(h²)` and the forward-difference error `O(h)` from a Taylor expansion. Minimize `O(h²) + ε/h`. Give the best `h` and the best achievable accuracy for `f64` and for `f32`. **This is the reason Day 1 exists.**
- [ ] Write a wrong backward **rule** that passes `grad_check` on `sum(out)` and fails `grad_check_projected`. A different instance from the one in the test file. Explain why the projection catches it.

- [ ] **Break a Day 10 backward arm on purpose. Watch the checker go red. Put it back.**

> **Stuck-signal.** If the check fails near 1e-3 relative error, print the worst index and the input value there first. Then ask.

- [ ] Session ritual steps 8 to 10 are done.

---

## ☐ Day 13 — Softmax and cross-entropy · 3 h

📖 **[Full lesson — days/DAY_13.md](days/DAY_13.md)** · [Day card](RUST_PHASE_0_1.md#day-13--softmax-and-cross-entropy)

- [ ] Move `tests/pending/day13.rs` to `tests/day13.rs`. Run it. Watch it fail.
- [ ] **Read (25 min).** Raschka section 5.1.2, pp. 132–140.
- [ ] Add `LogSumExp` and `CrossEntropy` to `Op`. The build breaks. Read the error list first.
- [ ] Write `logsumexp` forward, with the max-shift. `keepdim = true`.
- [ ] Write `softmax` and `log_softmax` as compositions. No new `Op` variant.
- [ ] **Derive the `LogSumExp` backward on paper.** Write the arm. Gradient-check it before you build on it.
- [ ] **Do self-check 1 on paper, before you write the cross-entropy arm.**
- [ ] Write `cross_entropy` forward, rank 2, mean over predictions.
- [ ] Write the `CrossEntropy` backward arm, with the `1 / predictions` factor.
- [ ] Record the owed PyTorch fixture in `PROGRESS.md`. It is due before Day 21.

**Tests**

- [ ] `softmax_sums_to_one`
- [ ] `softmax_overflow_safe`
- [ ] `softmax_shift_invariant`
- [ ] `cross_entropy_uniform_is_ln_n` ← **this assertion saves you in Phase 2**
- [ ] `logsumexp_gradcheck`
- [ ] `cross_entropy_gradcheck`

**Self-check (on paper)**

- [ ] **Derive** `∂L/∂xᵢ` from `L = −log(softmax(x)[t])`, for `i == t` and `i ≠ t`. Show both collapse to `softmax(x)ᵢ − 1[i == t]`. **With no notes. The most important derivation in Phase 0.**
- [ ] Compute the expected cross-entropy of an untrained model at vocab 50257. Write one paragraph on why it is the most valuable assertion in a training loop, and the three faults it catches at step 0.

> **Stuck-signal.** If `cross_entropy_gradcheck` fails by the same constant factor everywhere, that is the batch normalisation. Fix it yourself. A varying factor, ask.

- [ ] Session ritual steps 8 to 10 are done.

---

## ☐ Day 14 — Optimizers, and the Phase 0 capstone · 3.5 h

📖 **[Full lesson — days/DAY_14.md](days/DAY_14.md)** · 🎓 **[The capstone in depth — capstone_r0b/README.md](capstone_r0b/README.md)** · [Day card](RUST_PHASE_0_1.md#day-14--optimizers-and-the-phase-0-capstone)

- [ ] Move `tests/pending/day14.rs` to `tests/day14.rs`. Run it **in release**. Watch it fail.
- [ ] **Do self-check 1 on paper before you implement AdamW.**
- [ ] **Read (20 min).** _Programming Rust_ pp. 237–245.
- [ ] Create `src/optim.rs`. Write the `Optimizer` trait with the default `zero_grad`.
- [ ] Write `Sgd` with `Sgd::new(lr, momentum)`.
- [ ] Write `AdamW` with `AdamW::new(lr, beta1, beta2, eps, wd)`. Correct **both** moments.
- [ ] Apply the decay to the **parameter**, outside the normalisation.
- [ ] Create `src/nn.rs`. Write `Linear::new` with the init variance from lesson section 2.5.
- [ ] Write `Linear::forward`, appending `(id, tensor)` for `w` then `b`.
- [ ] Gradient-check a two-layer MLP **before** you train it.
- [ ] Write the three CSVs to `evidence/`: the loss curve, the data, and the boundary grid.
- [ ] Run `capstone_r0b/plot_loss.py` and `capstone_r0b/render_boundary.py`.
- [ ] Run `render_boundary.py --best-line` and put the number in the write-up.
- [ ] Write `evidence/phase0_spiral.md` with the seed, the hyperparameters and the machine.

**Tests**

- [ ] `sgd_descends_quadratic`
- [ ] `adamw_bias_correction_first_step`
- [ ] `adamw_weight_decay_is_decoupled`
- [ ] `spiral_classification`
- [ ] `all_ops_gradchecked` ← **the R0b acceptance test**

**Self-check (on paper)**

- [ ] **Trace one AdamW step by hand.** `g = 0.1`, `m₀ = v₀ = 0`, `β₁ = 0.9`, `β₂ = 0.999`, `lr = 1e-3`, `eps = 1e-8`, `wd = 0`. Every number. Show the update is near `lr`, and state in one sentence why that is by design.
- [ ] Write the decoupled and the coupled weight-decay updates side by side. Name the term that differs. State what the `v̂` denominator does to an L2 gradient contribution.

- [ ] **Inject a fault and watch `all_ops_gradchecked` fail.** Put it back.

> **Stuck-signal.** If the spiral loss stops at about `ln(2) ≈ 0.693`, work the diagnostic ladder in [`capstone_r0b/README.md`](capstone_r0b/README.md) section 7, from rung 0. **Rung 5, overfitting 4 points, splits the whole search space in seconds.** Do not tune before it. If the ladder runs out, ask.

- [ ] Session ritual steps 8 to 10 are done.

---

## ☐ GATE R0b — do not start Phase 1 until all four are true

- [ ] `cargo test --release` is fully green, Days 1 to 14.
- [ ] `all_ops_gradchecked` passes, and **you watched it fail** after you injected a fault.
- [ ] The spiral CSVs, both plots and `evidence/phase0_spiral.md` are committed, with the seed recorded.
- [ ] The R0b LinkedIn post is shipped.

**A claim without its artifact is not done.** When all four boxes are ticked, mark R0b ✅ in `PROGRESS.md`.

---

## What comes after, in one table

Do not plan these now. They expand at their gates.

| Claim   | Days   | Cards                                                                    | Exit test                                                            |
| ------- | ------ | ------------------------------------------------------------------------ | -------------------------------------------------------------------- |
| **R1**  | 15–25  | BPE, embeddings, LayerNorm, attention, GELU, blocks, safetensors, parity | Logit parity under 1e-3 against the real GPT-2 124M checkpoint.      |
| **P1**  | 1 week | NumPy and PyTorch fluency. Rebuild GPT-2.                                | A 3-way parity table against the Rust version.                       |
| **P2**  | 1 week | Training hygiene.                                                        | Three timed diagnoses of broken runs.                                |

Day 9 is the hardest card in the project. It takes 2 days. Read section 5 of [`RUST_PHASE_0_1.md`](RUST_PHASE_0_1.md) before you start it, and build the tape, not `Rc<RefCell<_>>`.
