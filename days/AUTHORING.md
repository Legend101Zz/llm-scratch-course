# AUTHORING.md — how to write a day lesson and its tests

> **This file is for Claude, not for Mrigesh.** Read it before you write any
> `DAY_NN.md` or any `tests/dayN.rs`. It is the working method behind the
> "Day lessons" section of [`CLAUDE.md`](../../CLAUDE.md), which stays the contract.
> Where the two disagree, `CLAUDE.md` wins.
>
> Written in [Simple English](../.claude/skills/simple-english/SKILL.md) (ASD-STE100), pragmatic mode.

---

## 1. Who the reader is

One reader. Mrigesh. Write for this person and for nobody else.

- **A novice in Rust.** Assume no knowledge of traits, ownership, `Result`, closures, iterators or `Rc` beyond what an earlier day taught. Every Rust idea gets built from the start, in the lesson, in your own words.
- **Not a novice in thinking.** Do not simplify the idea. Simplify the language. A first-principles explanation is longer than a hand-wave, not shorter.
- **Here for two things at once.** The Rust, and the machine learning with the mathematics under it. A lesson that teaches only the Rust has done half the job.
- **Reads one file per day and nothing else.** The lesson replaces the book. Page numbers are an optional cross-reference, never a dependency.
- **Wants depth.** Length is not the enemy. A lesson of 400 to 600 lines is normal. Filler is the enemy.

The voice is a teacher who respects the reader: direct, concrete, no praise, no hedging. State the fact. Give the example. Name the trap.

---

## 2. The hard rules

These come from `CLAUDE.md`. Breaking one damages the project, so they are repeated here.

| Rule | Why it exists |
|---|---|
| **No function body for code Mrigesh is building.** Signatures and `todo!()` only. | He writes the tensor library, the autograd and the kernels. That is the whole repo. |
| **No answer to a self-check question.** Pose it and stop. | The self-checks are the `hand_math/` deliverable. An answered question is a deleted exercise. |
| **No text copied from a book.** Teach the material, then cite the page. | The repo is public. |
| **No checkboxes.** | `RUST_TRACKER.md` owns every tick, so one file records state. |
| **Never write a lesson past the current gate.** | A lesson written before the API is argued is a lesson written against the wrong API. |

**Test files and benchmark harnesses are yours to write.** `CLAUDE.md` says so directly. A failing test that defines an acceptance test is the mentor's job. So is a criterion harness, a plotting script and a parity check.

---

## 3. The structure of a lesson file

Keep this order. The reader learns the shape of the file once and then knows where to look.

| # | Section | What goes in it |
|---|---|---|
| 1 | **Why this day exists** | The one idea. Then a table titled *What breaks later if you get this wrong*, with one row per real model step and the day it lands. Then a mermaid map of the day. |
| 2 | **The concept, from first principles** | The domain material. Split into two numbered sections when the day has two subjects, as Day 5 splits shape rules from floating point. ASCII memory figures, worked traces, the mathematics. |
| 3 | **The Rust you need today** | Every Rust idea the day needs, taught on **unrelated data**. Then one line of book page numbers at the end. |
| 4 | **What you build today** | Signatures copied from the day card. Any signature you had to fix, with the argument for it. Design points to decide, with instructions to record them in the commit message. |
| 5 | **The tests, and the trap in each one** | The `mv` command. One table row per test, stating what that test traps. Not what it checks. What it **traps**. |
| 6 | **Order of work** | A numbered list. Name the natural stopping points and the step that unblocks the rest. |
| 7 | **Compiler errors you will meet today** | Three columns: the error, what it means, where to look. Include the logic faults that are not compiler errors, and say so. |
| 8 | **Self-check** | Copied from the day card, posed and unanswered. Say plainly that you answer neither. |
| 9 | **Stuck-signals** | One bullet per signal, each with a time limit. |
| 10 | **The post** | A link to the LinkedIn hook at the end of the day card. |
| 11 | **Done means all five** | The five exit conditions. Day 7 has six, because it has a table deliverable. |

The header block above section 1 is fixed. Copy it from `DAY_03.md` and change the numbers.

---

## 4. The teaching method

### 4.1 Build from the problem, not from the rule

Open with the thing that does not work. Then the fix. Never open with the rule.

- **Weak:** "Broadcasting aligns shapes from the trailing dimension."
- **Strong:** "You have a matrix and a row and you want to add them. Copying the row 1024 times costs 3 million writes for 3072 distinct values. Here is what it costs to not copy."

### 4.2 Worked examples go on unrelated data

This is a `CLAUDE.md` rule and it has a reason: an example on his own tensor code hands him the design.

Use money, city names, shapes, bookshelves, tickets, temperatures. The established ones in this repo are `Shape`/`Square`/`Circle` for traits, `Cents` for operators, `Bookshelf` for structs, and Indian city names for collections. Reuse them.

**The exception:** the domain material in section 2 *is* the subject, so strides, broadcast rules and matmul use real tensors. The rule applies to the **Rust** teaching in section 3.

### 4.3 Every day earns at least one diagram

- A **mermaid flowchart** for the map of the day, and for anything with a flow.
- An **ASCII figure** for memory layout, buffer contents, index traces and cache behaviour. ASCII beats mermaid for anything with a grid, and it survives in a terminal.

The mermaid palette is fixed across the repo. Use these three classes and nothing else:

```text
classDef a fill:#8b3a1a,stroke:#4d1f0d,color:#fff    rust-orange: the thing being built, or the cost
classDef b fill:#1f6f43,stroke:#0d3d24,color:#fff    green: rules, metadata, the free path
classDef c fill:#b8860b,stroke:#6b4e06,color:#fff    gold: outputs, gates, later days
```

### 4.4 The mathematics gets written out

Do not say "this is the chain rule". Write the terms.

Do not say "floats lose precision". Write the bit layout, the ULP formula and a worked case at a stated magnitude.

Where the self-check owns the derivation, teach the **tool** and pose the question. Day 5 does this: it gives the rounding rule and the ULP table, and it leaves the `1.6e7` trace and the pairwise argument to `hand_math/`.

### 4.5 Connect to the model, every day

Section 1 gets a *What breaks later* table. Every row names a real GPT-2 step and the day it arrives. This is what turns "strides" from a data-structures exercise into a thing worth three hours.

Where a forward-pass idea has a backward-pass consequence, say it. Day 4 derives that the gradient of a broadcast is a sum, because that fact is free at the time and expensive to discover on Day 10.

### 4.6 Name the trap before he hits it

Every day has two or three faults that cost an hour each. Find them and write them down:

- The `u64::MAX` division on Day 1.
- The `Vec<Vec<T>>` storage model on Day 2.
- The permute convention on Day 3.
- `numel() != data.len()` after a broadcast on Day 4.
- A `max` fold seeded at zero on Day 5.
- The unclamped edge tile on Day 7.

Put each one in section 2 where it belongs, in the compiler-error table, and in the stuck-signals with a time limit.

### 4.7 Simple English, always

Read the [skill](../.claude/skills/simple-english/SKILL.md). The four rules that matter most here:

1. 20 words per instruction, 25 per description.
2. Active voice, simple tenses, one instruction per sentence.
3. Condition before command. *"If the build fails, read the log."*
4. No `should`, `would`, `may`, `might`, `could`. Use `can`, `will`, `must`.

Code blocks, identifiers, file paths and error strings are untouchable. Leave them exact.

---

## 5. Writing the test file

### 5.1 Where the file goes

The lesson for day `N` needs `tests/dayN.rs` to exist. Two states:

- **The day is active or done.** The file lives at `rust/tests/dayN.rs`, and Cargo builds it.
- **The day is ahead.** The file lives at `rust/tests/pending/dayN.rs`. Cargo ignores it, because auto-discovery only picks up `tests/*.rs` and `tests/*/main.rs`. Section 5 of the lesson gives the `mv` command.

The staging directory is what lets a test be written ahead without breaking `cargo test` for the days in between.

`rust/tests/common/mod.rs` holds shared helpers. Cargo does not build it either, for the same reason. A test file pulls it in with `mod common;`. Declare that line **only** in files that use it, or clippy reports an unused module.

### 5.2 The rules for a test file

1. **Integration tests only.** They see the public API. If a test wants a private item, the seam is wrong. Reach it through the public API instead, and explain the trick in the file header. `tests/day2.rs` reads `phys_index` through `get` by filling the buffer with `data[i] == i`.
2. **Use the exact test names from the day card.** The card is the acceptance test and it is named before the work starts. Extra coverage goes **inside** a named test as a new block with a comment, never as a new test function.
3. **Every assertion carries a message** when the failure is not self-explaining. `assert_eq!(got, want, "get([{i},{j},{k}])")` beats a bare comparison.
4. **State the conventions the card left open**, in the file header, before the tests. The permute convention, the rank split for `matmul_naive`, and the `Result` against `Self` split for the elementwise ops all live in a header block.
5. **Mark the arguable assertions.** Where a test fixes a design decision the card left to Mrigesh, say so: *"If you decide otherwise, delete this block and tell me why."* Day 1 does this for `seed_zero_is_rejected`.
6. **Use a mixed tolerance for float comparisons.** `|got - want| <= tol * (1 + |want|)`. `assert_all_close` in `common/mod.rs` does it. A pure absolute limit is too tight for large values, and a pure relative limit divides by zero.
7. **No timing assertions, ever.** Speed is a committed table that a human reads. A flaky red teaches him to ignore red.
8. **Pick sizes that break things.** Prime and odd dimensions, sizes smaller than the block, sizes that are not multiples of anything, all-negative inputs, empty axes, rank 0, rank 4.
9. **Say what each test traps**, in a doc comment above it. The lesson's test table repeats it in one line.

### 5.3 Type-check a test file before you ship it

A test file for a day that is not built yet cannot be compiled in the repo. Compile it anyway, against a stub.

1. Copy the crate to the scratchpad.
2. Write `src/tensor.rs` and `src/matmul.rs` with **every signature and a `todo!()` body**. `todo!()` satisfies any return type, so the crate type-checks and implements nothing.
3. Copy the test files in and run `cargo test --no-run`, then `cargo clippy --tests`.
4. Fix what it finds. Delete the stub. It never enters the repo.

This is not a rule you can skip. It caught a real fault on the first run: `Result::expect_err` needs `Debug` on the success type, and `Tensor` has none. The fix was a small `assert_err!` macro in `tests/day3.rs`, Without the stub check, that error reaches Mrigesh on the morning of Day 3.

### 5.4 Verify the expected values

Hand-computed values in a test are a claim. Check them.

Use `python3` for the arithmetic tables. Do not use NumPy to design the API, and do not put NumPy anywhere near the repo. Use it as a calculator for the numbers you are about to assert, then throw the script away.

---

## 6. Before you ship a lesson

Run this list. It is short because each item has failed once.

1. Every mandatory section from the table in section 3 is present, in order.
2. No function body for library code. Search the file for `{` on a line with a signature.
3. No self-check answer. Read section 8, then search the rest of the file for its numbers and its shapes. Day 4 keeps `[8,1,6]` and `[7,1]` out of the test table for exactly this reason.
4. At least one mermaid diagram and at least one ASCII figure.
5. Every Rust example in section 3 is on unrelated data.
6. The *What breaks later* table names real model steps and real day numbers.
7. No checkboxes anywhere.
8. The Simple English self-check: the three longest sentences are under the limit, no `should`/`would`/`may`/`might`, no semicolons, every `if` at the start of its sentence.
9. The test file compiles against a stub, and clippy is clean on it.
10. `RUST_TRACKER.md` links the lesson from that day's header.

---

## 7. State of the written-ahead lessons

**Days 3 to 7 were written ahead, on 28 August 2026, at Mrigesh's explicit request.**

`CLAUDE.md` says to write one lesson at a time, at the start of that day. This batch breaks that rule on purpose, and he asked for it in those words. Two facts limit the damage:

- All five days sit inside Gate R0a, so no lesson crosses a gate.
- The tests are type-checked against a stub, so the signatures are consistent with each other.

**What a later session must know.** These lessons fix API decisions that Mrigesh has not argued with yet. Each one is marked in the lesson, and here they are in one place:

| Day | The decision | Where it is stated |
|---|---|---|
| 3 | `strides()` and `shares_storage_with()` added to the tensor, so a test can observe the zero-copy claim | `DAY_03.md` section 4.1 |
| 3 | `permute(order)` uses the NumPy convention: `order[i]` is the **source** axis | `DAY_03.md` section 2.4, and the `day3.rs` header |
| 3 | `ShapeError` lives in `src/tensor.rs` | the `day3.rs` imports |
| 3 | An empty slice, where `start == end`, is legal | `day3.rs`, `slice_bounds` |
| 4 | Two-operand ops return `Result`, one-operand ops return `Self` | `DAY_04.md` section 4, and the `day4.rs` header |
| 4 | `zip_with` broadcasts both operands before it applies the closure | the `day4.rs` header |
| 4 | `map` and `zip_with` return a fresh contiguous tensor | `DAY_04.md` section 2.4 |
| 5 | A rank-1 tensor reduced with `keepdim = false` gives rank 0 | `day5.rs`, `sum_axis_shapes` |
| 6 | `matmul_naive` is rank 2 only. `matmul` handles batches | `DAY_06.md` section 4.1, and the `day6.rs` header |
| 6 | `matmul_naive` accepts a strided input, so it reads through the index formula | `DAY_06.md` section 4.2 |
| 7 | `matmul_blocked` is rank 2 only | `DAY_07.md` section 4.1 |

If he argues one of these down, the test file changes and the lesson changes with it. That is the correct outcome. Do not defend a decision because it is already written.

---

**Days 8 to 14 were written ahead, on 4 September 2026, at Mrigesh's explicit request.**

This is the second batch written ahead, and it is a larger exception than the first: **Days 9 to 14 sit past Gate R0a, in claim R0b.** He asked for "next week till day 14" in those words, after finishing the Day 1 to 7 material. Two facts limit the damage:

- Every test file was type-checked against a stub crate, so the signatures across all seven days are consistent with each other and with the real `Tensor`, `Scalar` and `Rng` that already exist.
- Every decision the batch fixes is listed below, and each one is marked in its own lesson.

**What a later session must know.** These lessons fix API decisions that Mrigesh has not argued with yet. Here they are in one place.

| Day | The decision | Where it is stated |
|---|---|---|
| 8 | `matmul_parallel` is rank 2 only, and the comparison against `matmul_blocked` is **exact**, not `assert_all_close` | `DAY_08.md` section 2.8, and the `day8.rs` header |
| 8 | `Rc` is not `Send`, so the day forces a choice: `Rc` becomes `Arc`, or the kernel extracts plain slices before the scope. **The card does not mention this collision.** The tests are agnostic to the choice. | `DAY_08.md` section 4.2 |
| 9 | `Tape` gains `len`, `is_empty`, `op(id)` and `requires_grad(id)`, and `NodeId` gains `index()`, so a test can observe the recorded structure | `DAY_09.md` section 4.2 |
| 9 | `Op` derives `Clone, Debug, PartialEq`. `NodeId` derives `Clone, Copy, PartialEq, Eq, Debug` | the `day9.rs` header |
| 9 | **The forward helpers return a bare `NodeId` and panic on a shape error.** They are not `Result`. A shape disagreement in a model definition is a bug, not a data-dependent state. `DAY_09.md` still poses the question, and the test header marks the assumption. | the `day9.rs` header, decision 4 |
| 10 | The remaining forward helpers land on Day 10: `neg`, `exp`, `ln`, `tanh`, `sum_axis`, `broadcast_to`, and `sum_all` | `DAY_10.md` section 4.2 |
| 10 | `sum_all` reduces every axis away and gives a **rank 0** node, following the Day 5 `keepdim = false` rule | the `day10.rs` header |
| 10 | `unbroadcast` is a free function, not a method, because it needs nothing from the tape | `DAY_10.md` section 4.1 |
| 11 | An `autograd::matmul` forward helper lands with the `MatMul` arm. The card implies only the arm. | `DAY_11.md` section 4.1 |
| 12 | **`build` returns the OUTPUT node, not a scalar.** The card's comment says scalar. The projected checker must apply its own weights to the un-reduced output, so the checker owns the reduction. Without this, `grad_check_projected` cannot exist. | `DAY_12.md` section 2.8, and the `day12.rs` header |
| 12 | The checker splits into `analytic_grads`, `numeric_grads`, `compare_grads` plus the two thin entry points. The card lists two functions. The split is what lets `gradcheck_catches_injected_bug` be written through the public API instead of behind a feature flag. | `DAY_12.md` section 2.9 |
| 12 | `GradCheckReport` gains `worst_input` beside the card's `worst_index`, because an index alone does not say which input tensor | `DAY_12.md` section 4.1 |
| 13 | A `logsumexp` forward helper is public. `softmax` and `log_softmax` are compositions and add no `Op` variant. | `DAY_13.md` section 4.2 |
| 13 | `Op::LogSumExp` reduces with **keepdim = true**, so `log_softmax` is a plain broadcasting subtraction | the `day13.rs` header |
| 13 | `cross_entropy` takes rank-2 logits `[predictions, classes]`, returns the **mean**, and panics on an out-of-range target | the `day13.rs` header |
| 14 | **`Linear::forward` takes a fourth argument**, `params: &mut Vec<(NodeId, Tensor<T>)>`, and appends `w` then `b`. The card writes a three-argument version. `forward` must push the weights as leaves to get ids, and the optimizer needs exactly those ids, so they have to come back out. | `DAY_14.md` section 4.2, and the `day14.rs` header |
| 14 | `Sgd::new` and `AdamW::new` exist. The moment buffers are private, so a constructor is the only way in. | the `day14.rs` header |
| 14 | **The spiral's `TURN` is `4*pi`, not the 3.5 radians first written.** Measured: at 3.5 rad the best possible straight line scores 93 percent and the MLP passes the 99 percent bar in as few as 3 epochs, so the test proved nothing. At `4*pi` the best line scores 60.5 percent and the MLP needs 274 to 391 epochs across 8 seeds, inside a 2000 budget. | `capstone_r0b/README.md` section 2.3, and the `TURN` doc comment in `day14.rs` |
| 14 | The capstone gets its own file, `capstone_r0b/README.md`, because Day 14 cannot teach AdamW and the capstone in 3.5 hours. Two plotting tools live beside it. | `DAY_14.md` header |
| 14 | `spiral_classification` contains its training loop, which `CLAUDE.md` lists under "may not write". The test header says so, explains why, and offers to move the loop to `src/bin/spiral.rs` if he wants to own it. | the `day14.rs` header |

**The Day 13 fixture is owed, not written.** The card lists a fixture test against committed PyTorch reference values. Those fixtures do not exist. Section 6 of `RUST_PHASE_0_1.md` schedules one Colab session before Day 21. The lesson tells him to record it in `PROGRESS.md` on the day he reads it. **Do not let that slip.**

If he argues any of these down, the test file changes and the lesson changes with it. That is the correct outcome. Do not defend a decision because it is already written.

**Day 15 and later are not written.** Go back to one day at a time. Day 16 and Day 23 are 2-day cards, and Day 24 is where the three checkpoint traps in section 6 of `RUST_PHASE_0_1.md` collect.

---

## 8. The skeleton

Copy this, then fill it. Keep the header block byte-identical apart from the numbers.

```markdown
# Day N — <title from the day card>

> **Read this file. It replaces the book for today.**
> The page numbers stay as optional depth. This lesson teaches the same material in my own words.
> Tick your boxes in [`RUST_TRACKER.md`](../RUST_TRACKER.md). This file holds no checkboxes.
> The short card is [`RUST_PHASE_0_1.md` Day N](../RUST_PHASE_0_1.md#day-N--<anchor>).

**Time: <h> hours. Claim: <R0a>. File you <create|extend>: `src/<file>.rs`.**
**Your tests are written: move `rust/tests/pending/dayN.rs` to `rust/tests/dayN.rs` at the start of the session.**

---

## 1. Why this day exists
### 1.1 What breaks later if you get this wrong
### 1.2 The map of today          <- mermaid

## 2. <the concept>, from first principles
                                  <- ASCII figures, worked traces, the mathematics

## 3. The Rust you need today
                                  <- unrelated data only. Book pages at the end.

## 4. What you build today
                                  <- signatures, fixed decisions, design points to record

## 5. The tests, and the trap in each one
                                  <- the mv command, then one row per test

## 6. Order of work
                                  <- numbered, with the stopping points named

## 7. Compiler errors you will meet today

## 8. Self-check — on paper, before you close the laptop
                                  <- posed, never answered

## 9. Stuck-signals — the points where you ask
                                  <- each one with a time limit

## 10. The post

## 11. Done means all five
```
