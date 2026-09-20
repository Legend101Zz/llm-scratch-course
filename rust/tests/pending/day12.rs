//! Day 12 acceptance tests — the gradient checker.
//!
//! The mentor writes this file. You write the code that makes it pass.
//! Do not edit the assertions. If a test is wrong, say so and argue it.
//!
//! Move this file to `rust/tests/day12.rs` at the start of Day 12, then run
//! `cargo test --test day12` and watch it fail.
//!
//! Expected module layout. Add one line to `src/lib.rs`:
//!
//! ```text
//! pub mod gradcheck;   // f64 only. There is no generic parameter.
//! ```
//!
//! ## The decisions this file fixes
//!
//! 1. **`build` returns the OUTPUT node, not a scalar.** The card's comment
//!    says scalar. The projected checker has to apply its own weights to the
//!    un-reduced output, so the checker owns the reduction. `DAY_12.md`
//!    section 2.8 is the argument. If `build` reduced, `grad_check_projected`
//!    could not exist.
//!
//! 2. The API splits into `analytic_grads`, `numeric_grads`, `compare_grads`,
//!    and the two thin entry points. That split is what lets a test hand the
//!    comparison a deliberately wrong gradient without breaking the library.
//!    `DAY_12.md` section 2.9.
//!
//! 3. `weights: Option<&Tensor<f64>>`. `None` means all ones, which makes the
//!    scalar `sum(out)`. `Some(r)` makes it `sum(out * r)`.
//!
//! 4. `compare_grads` returns `passed = false` when any element exceeds the
//!    tolerance. It never panics on a mismatch. Deciding is its whole job.

mod common;

use common::randn;
use rustgpt::autograd::{
    NodeId, Tape, add, broadcast_to, exp, ln, matmul as gmatmul, mul, sum_axis, tanh,
};
use rustgpt::gradcheck::{compare_grads, grad_check, grad_check_projected, numeric_grads};
use rustgpt::matmul::matmul_naive;
use rustgpt::rng::Rng;
use rustgpt::tensor::Tensor;

const H: f64 = 1e-5;
const TOL: f64 = 1e-5;

fn t(data: Vec<f64>, shape: &[usize]) -> Tensor<f64> {
    Tensor::from_vec(data, shape)
}

// ---------------------------------------------------------------------------

/// Traps any wrong rule from Days 10 and 11.
///
/// One block per op, each with its own inputs and its own message, so a
/// failure names the op instead of the file. The `ln` inputs are drawn away
/// from zero on purpose, which is rule 1 of `DAY_12.md` section 2.6.
#[test]
fn gradcheck_passes_for_correct_ops() {
    let mut rng = Rng::seed(12_121);

    // --- add ---------------------------------------------------------------
    let inputs = vec![randn(&mut rng, &[3, 4]), randn(&mut rng, &[3, 4])];
    let r = grad_check(|tape: &mut Tape<f64>, ids: &[NodeId]| add(tape, ids[0], ids[1]), &inputs, H, TOL);
    assert!(r.passed, "add: max_rel_err {:e} at {:?}", r.max_rel_err, r.worst_index);

    // --- mul ---------------------------------------------------------------
    let inputs = vec![randn(&mut rng, &[3, 4]), randn(&mut rng, &[3, 4])];
    let r = grad_check(|tape: &mut Tape<f64>, ids: &[NodeId]| mul(tape, ids[0], ids[1]), &inputs, H, TOL);
    assert!(r.passed, "mul: max_rel_err {:e} at {:?}", r.max_rel_err, r.worst_index);

    // --- matmul, rank 2, all dimensions different --------------------------
    let inputs = vec![randn(&mut rng, &[2, 3]), randn(&mut rng, &[3, 4])];
    let r = grad_check(|tape: &mut Tape<f64>, ids: &[NodeId]| gmatmul(tape, ids[0], ids[1]), &inputs, H, TOL);
    assert!(r.passed, "matmul: max_rel_err {:e} at {:?}", r.max_rel_err, r.worst_index);

    // --- matmul, batched with a broadcast batch ----------------------------
    let inputs = vec![randn(&mut rng, &[1, 2, 3]), randn(&mut rng, &[2, 3, 2])];
    let r = grad_check(|tape: &mut Tape<f64>, ids: &[NodeId]| gmatmul(tape, ids[0], ids[1]), &inputs, H, TOL);
    assert!(r.passed, "matmul batched: max_rel_err {:e}", r.max_rel_err);

    // --- exp, on a modest range so the forward value stays sane ------------
    let inputs = vec![t(vec![-1.3, -0.4, 0.0, 0.7, 1.9, 2.5], &[2, 3])];
    let r = grad_check(|tape: &mut Tape<f64>, ids: &[NodeId]| exp(tape, ids[0]), &inputs, H, TOL);
    assert!(r.passed, "exp: max_rel_err {:e} at {:?}", r.max_rel_err, r.worst_index);

    // --- ln, strictly positive and away from zero --------------------------
    let inputs = vec![t(vec![0.4, 0.9, 1.5, 2.2, 3.7, 6.1], &[2, 3])];
    let r = grad_check(|tape: &mut Tape<f64>, ids: &[NodeId]| ln(tape, ids[0]), &inputs, H, TOL);
    assert!(r.passed, "ln: max_rel_err {:e} at {:?}", r.max_rel_err, r.worst_index);

    // --- tanh, including the saturating ends -------------------------------
    let inputs = vec![t(vec![-3.0, -0.6, 0.0, 0.5, 1.4, 3.0], &[2, 3])];
    let r = grad_check(|tape: &mut Tape<f64>, ids: &[NodeId]| tanh(tape, ids[0]), &inputs, H, TOL);
    assert!(r.passed, "tanh: max_rel_err {:e} at {:?}", r.max_rel_err, r.worst_index);

    // --- sum, both keepdim settings and both axes --------------------------
    let inputs = vec![randn(&mut rng, &[3, 4])];
    for (axis, keepdim) in [(0usize, false), (0, true), (1, false), (1, true)] {
        let r = grad_check(
            |tape: &mut Tape<f64>, ids: &[NodeId]| sum_axis(tape, ids[0], axis, keepdim),
            &inputs,
            H,
            TOL,
        );
        assert!(r.passed, "sum axis {axis} keepdim {keepdim}: {:e}", r.max_rel_err);
    }

    // --- broadcast ---------------------------------------------------------
    let inputs = vec![randn(&mut rng, &[1, 4])];
    let r = grad_check(|tape: &mut Tape<f64>, ids: &[NodeId]| broadcast_to(tape, ids[0], &[3, 4]), &inputs, H, TOL);
    assert!(r.passed, "broadcast: max_rel_err {:e}", r.max_rel_err);

    // --- a composite, so the reverse loop is exercised over many nodes -----
    let inputs = vec![randn(&mut rng, &[2, 3]), randn(&mut rng, &[3, 3])];
    let r = grad_check(
        |tape: &mut Tape<f64>, ids: &[NodeId]| {
            let p = gmatmul(tape, ids[0], ids[1]);
            let a = tanh(tape, p);
            let b = mul(tape, a, a);
            add(tape, b, a)
        },
        &inputs,
        H,
        TOL,
    );
    assert!(r.passed, "composite: max_rel_err {:e} at {:?}", r.max_rel_err, r.worst_index);

    // The projected checker must also pass on correct code, at both entry
    // points. A checker that only ever fails is as useless as one that
    // only ever passes.
    let r = grad_check_projected(
        |tape: &mut Tape<f64>, ids: &[NodeId]| mul(tape, ids[0], ids[1]),
        &inputs_pair(&mut rng),
        H,
        TOL,
        99,
    );
    assert!(r.passed, "projected on correct mul: {:e}", r.max_rel_err);
}

fn inputs_pair(rng: &mut Rng) -> Vec<Tensor<f64>> {
    vec![randn(rng, &[3, 3]), randn(rng, &[3, 3])]
}

// ---------------------------------------------------------------------------

/// Traps a checker that always passes.
///
/// A checker with no failing case is an untested test. This one hands
/// `compare_grads` the gradient that a known-wrong `mul` backward produces,
/// and asserts the report FAILS.
///
/// The injected fault is the card's: a `mul` backward that returns the
/// incoming gradient instead of `grad * other`. For `z = x*y` under a plain
/// sum, the correct `grad_x` is `y` and the buggy one is all ones.
#[test]
fn gradcheck_catches_injected_bug() {
    let mut rng = Rng::seed(3_141);

    let x: Tensor<f64> = randn(&mut rng, &[2, 3]);
    let y: Tensor<f64> = randn(&mut rng, &[2, 3]);
    let inputs = vec![x.clone(), y.clone()];

    let build = |tape: &mut Tape<f64>, ids: &[NodeId]| mul(tape, ids[0], ids[1]);

    // The truth, from the definition of the derivative.
    let numeric = numeric_grads(build, &inputs, None, H);

    // The correct analytic side agrees with it.
    let correct = grad_check(build, &inputs, H, TOL);
    assert!(correct.passed, "the correct rule must pass first: {:e}", correct.max_rel_err);

    // Now the bug: `grad` instead of `grad * other`, on both operands.
    let ones_x = t(vec![1.0; x.numel()], x.shape());
    let ones_y = t(vec![1.0; y.numel()], y.shape());
    let wrong = vec![ones_x, ones_y];

    let report = compare_grads(&wrong, &numeric, TOL);
    assert!(
        !report.passed,
        "the checker accepted a wrong gradient. max_rel_err was {:e}. \
         Until this assertion holds, every other green in this repo is worthless.",
        report.max_rel_err
    );
    assert!(
        report.max_rel_err > 1e-2,
        "a real bug must sit far outside the tolerance, not just past it. Got {:e}",
        report.max_rel_err
    );

    // A second injected fault: the right rule, scaled by 2. A factor error is
    // the most common real fault and it must not slip through either.
    let doubled: Vec<Tensor<f64>> = numeric.iter().map(|g| g.map(|v| v * 2.0)).collect();
    assert!(
        !compare_grads(&doubled, &numeric, TOL).passed,
        "the checker accepted a gradient that was twice the truth"
    );
}

// ---------------------------------------------------------------------------

/// Traps the blind spot of a plain sum.
///
/// The forward is `y = S · x` with S a 3x3 matrix whose row sums and column
/// sums are all 3, and which is NOT symmetric:
///
///     S = [[0,1,2],       Sᵀ = [[0,2,1],
///          [2,0,1],             [1,0,2],
///          [1,2,0]]             [2,1,0]]
///
/// The correct gradient is `Sᵀ · ȳ`. A backward that forgets the transpose
/// gives `S · ȳ`.
///
/// Under a plain sum, `ȳ` is all ones, so the correct answer is the column
/// sums and the wrong answer is the row sums. Both are [3,3,3], so the plain
/// check ACCEPTS the wrong rule. Under a random projection they differ, and
/// the projected check rejects it.
///
/// Self-check 2 asks you for a different instance, expressed as a wrong
/// backward rule. Do that exercise before you read this body.
#[test]
fn gradcheck_projected_catches_sign_flip() {
    let s = t(vec![0.0, 1.0, 2.0, 2.0, 0.0, 1.0, 1.0, 2.0, 0.0], &[3, 3]);
    let x = t(vec![0.7, -1.1, 0.3], &[3, 1]);
    let inputs = vec![x];

    let s_for_build = s.clone();
    let build = move |tape: &mut Tape<f64>, ids: &[NodeId]| {
        let s_node = tape.leaf(s_for_build.clone(), false);
        gmatmul(tape, s_node, ids[0])
    };

    // --- the plain reduction: the wrong rule slips through -----------------
    let ones = t(vec![1.0, 1.0, 1.0], &[3, 1]);
    let numeric_plain = numeric_grads(&build, &inputs, None, H);
    let wrong_plain = matmul_naive(&s, &ones).expect("S times ones");

    let plain = compare_grads(&[wrong_plain], &numeric_plain, TOL);
    assert!(
        plain.passed,
        "the plain sum must ACCEPT this wrong rule, or the example is not the \
         phenomenon. max_rel_err {:e}",
        plain.max_rel_err
    );

    // --- the projected reduction: the same wrong rule is caught ------------
    let r = t(vec![0.5, -1.25, 2.0], &[3, 1]);
    let numeric_proj = numeric_grads(&build, &inputs, Some(&r), H);
    let wrong_proj = matmul_naive(&s, &r).expect("S times r");

    let projected = compare_grads(&[wrong_proj], &numeric_proj, TOL);
    assert!(
        !projected.passed,
        "the projection must REJECT the wrong rule. max_rel_err {:e}",
        projected.max_rel_err
    );

    // And the correct rule passes both, so the projection is not simply
    // stricter about everything.
    assert!(grad_check(&build, &inputs, H, TOL).passed, "correct rule, plain");
    assert!(
        grad_check_projected(&build, &inputs, H, TOL, 7).passed,
        "correct rule, projected"
    );
}
