# RUST_TRACKER.md — the day checklist for R0a

> Written in [Simple English](.claude/skills/simple-english/SKILL.md) (ASD-STE100).
> This file is the **checklist**. [`RUST_PHASE_0_1.md`](RUST_PHASE_0_1.md) is the **explanation**.
> Open both. Tick the boxes here. Read the concept there.

**Active claim: R0a.** I can write a strided, broadcast tensor library in Rust with a cache-blocked multithreaded matmul. I can prove it correct with no reference implementation.

**Timebox: 2 weeks. 9 working days. Day 7 takes 2 days.**

**Timebox start date: `____________`** ← write the date here on Day 0. The clock starts then.

| Week | Days | What lands |
|---|---|---|
| **Week 1** | 1–6 | Trait, PRNG, strides, views, broadcast, reductions, the naive matmul oracle |
| **Week 2** | 7–8 | Blocked matmul, scoped threads, the GFLOP/s table, the gate |

This file covers **R0a only**. R0b (Days 9–14) gets its own section after you pass Gate R0a. Plan one claim at a time.

---

## Day 0 — setup (15 minutes, before Day 1)

Do this once. Do not skip the `criterion` line, because Day 7 needs it.

**Crate location: `course/rust/`. One library crate. The package name is `rustgpt`.**
One crate holds R0a, R0b and R1. Binaries arrive at R1 under `src/bin/`.

- [ ] Run `rustc --version`. Confirm 1.91 or newer.
- [ ] From `course/`, run `cargo new --lib --name rustgpt rust`.
- [ ] Add `criterion = "0.5"` under `[dev-dependencies]` in `rust/Cargo.toml`.
- [ ] Add `rust/target/` to `course/.gitignore`.
- [ ] Run `cargo test` inside `rust/`. Confirm it builds and reports 0 tests.
- [ ] Run `cargo clippy`. Confirm it is clean.
- [ ] Write the current date in the **Timebox start date** line above.
- [ ] Commit the empty crate. The first commit is the start of the trail.

> **CAUTION: Do not add any other dependency.** Banned for R0a, R0b and R1: `ndarray`, `nalgebra`, `candle`, `burn`, `tch`, `tokenizers`, any BLAS, any autodiff crate. `criterion` is the one exception, and it is a dev-dependency.

---

## Every session — the ritual

This block applies to every day below. It is here once, not six times.

**"You" is always Mrigesh.** Where the actor is the mentor, the step names Claude.

1. [ ] Open `PROGRESS.md` and this file. Claude states the active claim and the days left.
2. [ ] Claude quizzes you cold on the due `REVIEW.md` items and the concept of the prior day.
3. [ ] Read the prereq pages **before** you write code. The pages are on the day card.
4. [ ] Ask Claude for the failing test file. **Claude writes the tests. You write the code.**
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

[Day card](RUST_PHASE_0_1.md#day-1--the-scalar-trait-and-a-prng-you-own)

- [ ] **Read (25 min).** *Programming Rust* pp. 235–252. Skim only.
- [ ] Create `src/scalar.rs` and `src/rng.rs`. Declare both in `src/lib.rs`.
- [ ] Write the `Scalar` trait with the supertrait bounds and the two constants.
- [ ] Implement `Scalar for f32`.
- [ ] Implement `Scalar for f64`.
- [ ] Write `Rng::seed`. Reject a seed of 0.
- [ ] Write `Rng::next_u64` as xorshift64*.
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

[Day card](RUST_PHASE_0_1.md#day-2--storage-shape-strides)

- [ ] **Read (20 min).** *Programming Rust* pp. 57–63 and pp. 90–92.
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

[Day card](RUST_PHASE_0_1.md#day-3--zero-copy-views-reshape-permute-transpose-slice)

- [ ] **Read (20 min).** *Programming Rust* pp. 148–158.
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

[Day card](RUST_PHASE_0_1.md#day-4--broadcast-and-elementwise-ops)

- [ ] **Read (25 min).** *Programming Rust* pp. 303–312 and pp. 330–344.
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

[Day card](RUST_PHASE_0_1.md#day-5--reductions)

- [ ] **Read (20 min).** *Programming Rust* pp. 345–354.
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

[Day card](RUST_PHASE_0_1.md#day-6--the-naive-matmul-and-the-oracle-discipline)

Today you write the slowest correct matmul. **You keep it for the whole project.** It is the oracle for every faster version.

- [ ] **Read (15 min).** *Programming Rust* pp. 178–182.
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

[Day card](RUST_PHASE_0_1.md#day-7--blocked-matmul-and-the-measured-gap--2-day-card)

**Predict before you measure.** Write the predicted best block size on paper first. The prediction is part of the exit artifact.

- [ ] **Read (20 min).** *Programming Rust* pp. 161–165. Then skim the criterion "Getting Started" page.
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

[Day card](RUST_PHASE_0_1.md#day-8--parallelism-with-scoped-threads)

- [ ] **Read (30 min).** *Programming Rust* pp. 457–466. Read the Rayon section to learn what you leave out.
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

**A claim without its artifact is not done.** When all four boxes are ticked, mark R0a ✅ in `PROGRESS.md`. Then this file gets its R0b section.

---

## What comes after, in one table

Do not plan these now. They expand at their gates.

| Claim | Days | Cards | Exit test |
|---|---|---|---|
| **R0b** | 9–14 | Tape, backward, gradient checker, softmax, cross-entropy, AdamW | Every op passes `grad_check` in `f64`. You watched the checker fail. |
| **R1** | 15–25 | BPE, embeddings, LayerNorm, attention, GELU, blocks, safetensors, parity | Logit parity under 1e-3 against the real GPT-2 124M checkpoint. |
| **P1** | 1 week | NumPy and PyTorch fluency. Rebuild GPT-2. | A 3-way parity table against the Rust version. |
| **P2** | 1 week | Training hygiene. | Three timed diagnoses of broken runs. |

Day 9 is the hardest card in the project. It takes 2 days. Read section 5 of [`RUST_PHASE_0_1.md`](RUST_PHASE_0_1.md) before you start it, and build the tape, not `Rc<RefCell<_>>`.
