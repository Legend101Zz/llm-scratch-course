//! Day 13 acceptance tests — softmax and cross-entropy.
//!
//! The mentor writes this file. You write the code that makes it pass.
//! Do not edit the assertions. If a test is wrong, say so and argue it.
//!
//! Move this file to `rust/tests/day13.rs` at the start of Day 13, then run
//! `cargo test --test day13` and watch it fail.
//!
//! ## The decisions this file fixes
//!
//! 1. `softmax` and `log_softmax` are COMPOSITIONS, not new `Op` variants.
//!    Only `LogSumExp` and `CrossEntropy` are new nodes.
//!
//! 2. `LogSumExp { input, axis }` reduces with **keepdim = true**. That is
//!    what makes `log_softmax` a plain broadcasting subtraction with no
//!    reshape. `DAY_13.md` section 4.3, design point five.
//!
//! 3. `cross_entropy` takes rank-2 logits, `[predictions, classes]`, and a
//!    `targets` slice of length `predictions`. Flatten the leading axes
//!    before you call it. `DAY_13.md` section 2.9.
//!
//! 4. It returns the **mean** over predictions, so the gradient carries a
//!    `1 / predictions` factor. `DAY_13.md` section 2.8.
//!
//! 5. A target index outside `0..classes` panics. That is a bug in the data
//!    pipeline, not a state to carry.
//!
//! ## What is deliberately missing
//!
//! The card also lists a fixture test against committed PyTorch reference
//! values. Those fixtures do not exist yet. Section 6 of `RUST_PHASE_0_1.md`
//! schedules one Colab session before Day 21. Record it as owed in
//! `PROGRESS.md` and add the test when the fixture lands.

mod common;

use common::{assert_all_close, randn};
use rustgpt::autograd::{NodeId, Tape, cross_entropy, log_softmax, logsumexp, softmax};
use rustgpt::gradcheck::{grad_check, grad_check_projected};
use rustgpt::rng::Rng;
use rustgpt::tensor::Tensor;

const H: f64 = 1e-5;
const TOL: f64 = 1e-5;

fn t(data: Vec<f64>, shape: &[usize]) -> Tensor<f64> {
    Tensor::from_vec(data, shape)
}

/// Runs a forward-only graph and returns the value of the output node.
fn forward(build: impl Fn(&mut Tape<f64>, NodeId) -> NodeId, x: Tensor<f64>) -> Tensor<f64> {
    let mut tape: Tape<f64> = Tape::new();
    let id = tape.leaf(x, false);
    let out = build(&mut tape, id);
    tape.value(out).clone()
}

// ---------------------------------------------------------------------------

/// Traps a wrong reduction axis, and a denominator built from the wrong set.
///
/// The rank-3 case is checked along two different axes, so an implementation
/// that hardcodes the last axis fails on the first of them.
#[test]
fn softmax_sums_to_one() {
    let mut rng = Rng::seed(13_013);

    for &axis in &[0usize, 1] {
        let x: Tensor<f64> = randn(&mut rng, &[3, 5]);
        let p = forward(|tape: &mut Tape<f64>, id: NodeId| softmax(tape, id, axis), x);
        assert_eq!(p.shape(), &[3, 5], "softmax keeps the input shape");

        let sums = p.sum_axis(axis, false).expect("sum along the reduced axis");
        for (i, s) in sums.to_vec().into_iter().enumerate() {
            assert!(
                (s - 1.0).abs() < 1e-12,
                "axis {axis}, slice {i}: the probabilities sum to {s}, not 1"
            );
        }

        // And every entry is a probability.
        for (i, v) in p.to_vec().into_iter().enumerate() {
            assert!(v > 0.0 && v < 1.0, "axis {axis}, element {i} is {v}");
        }
    }

    // Rank 3, along the middle axis and the last axis.
    for &axis in &[1usize, 2] {
        let x: Tensor<f64> = randn(&mut rng, &[2, 3, 4]);
        let p = forward(|tape: &mut Tape<f64>, id: NodeId| softmax(tape, id, axis), x);
        let sums = p.sum_axis(axis, false).expect("rank 3 sum");
        for (i, s) in sums.to_vec().into_iter().enumerate() {
            assert!((s - 1.0).abs() < 1e-12, "rank 3, axis {axis}, slice {i}: {s}");
        }
    }
}

// ---------------------------------------------------------------------------

/// Traps a missing max-shift.
///
/// The naive formula returns `inf / inf = NaN` here. A shift applied to the
/// numerator only returns finite garbage, so the test checks the values as
/// well as their finiteness.
#[test]
fn softmax_overflow_safe() {
    let p = forward(
        |tape: &mut Tape<f64>, id: NodeId| softmax(tape, id, 0),
        t(vec![1000.0, 1001.0, 1002.0], &[3]),
    );

    for (i, v) in p.to_vec().into_iter().enumerate() {
        assert!(v.is_finite(), "element {i} is {v}, so the shift is missing");
    }

    // The true answer depends only on the differences 0, 1, 2.
    assert_all_close(
        &p,
        &t(
            vec![0.090_030_573_170_380_46, 0.244_728_471_054_797_64, 0.665_240_955_774_821_8],
            &[3],
        ),
        1e-12,
        "softmax([1000, 1001, 1002])",
    );

    // The other direction. Very negative logits underflow to zero in the
    // numerator, and the denominator must still be at least 1.
    let p = forward(
        |tape: &mut Tape<f64>, id: NodeId| softmax(tape, id, 0),
        t(vec![-1000.0, -1001.0, -1002.0], &[3]),
    );
    let sum: f64 = p.to_vec().iter().sum();
    assert!((sum - 1.0).abs() < 1e-12, "very negative logits sum to {sum}");

    // logsumexp on its own must survive the same input.
    let l = forward(|tape: &mut Tape<f64>, id: NodeId| logsumexp(tape, id, 0), t(vec![1000.0, 1001.0, 1002.0], &[3]));
    assert_eq!(l.shape(), &[1], "logsumexp reduces with keepdim = true");
    let got = l.to_vec()[0];
    assert!(
        (got - 1_002.407_605_964_444_4).abs() < 1e-9,
        "logsumexp([1000,1001,1002]) was {got}"
    );
}

// ---------------------------------------------------------------------------

/// Traps a shift by the wrong quantity.
///
/// Adding a constant to every logit must change nothing at all. A shift by a
/// whole-tensor constant instead of a per-row maximum shows up here as a
/// row-dependent error, because the two rows have different maxima.
#[test]
fn softmax_shift_invariant() {
    let base = t(vec![0.3, -1.2, 2.5, 40.0, 41.5, 39.0], &[2, 3]);
    let shifted = base.map(|v| v + 500.0);

    let a = forward(|tape: &mut Tape<f64>, id: NodeId| softmax(tape, id, 1), base.clone());
    let b = forward(|tape: &mut Tape<f64>, id: NodeId| softmax(tape, id, 1), shifted);
    assert_all_close(&b, &a, 1e-12, "softmax(x + 500) must equal softmax(x)");

    // log_softmax is shift invariant too, and it must equal ln(softmax(x))
    // without ever computing that quotient.
    let la = forward(|tape: &mut Tape<f64>, id: NodeId| log_softmax(tape, id, 1), base.clone());
    let want = a.ln();
    assert_all_close(&la, &want, 1e-12, "log_softmax equals ln(softmax) in value");

    // And a per-row shift by different amounts is still invariant, which a
    // single global maximum would get wrong.
    let per_row = t(vec![0.3, -1.2, 2.5, 340.0, 341.5, 339.0], &[2, 3]);
    let c = forward(|tape: &mut Tape<f64>, id: NodeId| softmax(tape, id, 1), per_row);
    let row0_a: Vec<f64> = a.to_vec()[..3].to_vec();
    let row0_c: Vec<f64> = c.to_vec()[..3].to_vec();
    for (i, (x, y)) in row0_a.iter().zip(row0_c.iter()).enumerate() {
        assert!(
            (x - y).abs() < 1e-12,
            "row 0 element {i} changed when only row 1 was shifted: {x} against {y}"
        );
    }
}

// ---------------------------------------------------------------------------

/// Traps a wrong base, a missing negation, a sum where a mean belongs, and a
/// wrong reduction, all with one number.
///
/// Uniform logits over `n` classes carry no information, so the loss is
/// exactly `ln(n)`. Several class counts, because a fault that happens to be
/// right at one `n` fails at another.
#[test]
fn cross_entropy_uniform_is_ln_n() {
    for &(n, want) in &[
        (2usize, std::f64::consts::LN_2),
        (5, 1.609_437_912_434_100_3),
        (10, std::f64::consts::LN_10),
    ] {
        // One prediction, uniform logits. Any constant works, by shift invariance.
        let logits = t(vec![0.0; n], &[1, n]);
        let mut tape: Tape<f64> = Tape::new();
        let id = tape.leaf(logits, false);
        let loss = cross_entropy(&mut tape, id, &[0]);

        assert!(
            tape.value(loss).shape().is_empty(),
            "the loss is a rank-0 scalar, n = {n}"
        );
        let got = tape.value(loss).to_vec()[0];
        assert!(
            (got - want).abs() < 1e-12,
            "uniform over {n} classes gave {got}, want ln({n}) = {want}"
        );

        // The target must not matter for uniform logits.
        let mut tape: Tape<f64> = Tape::new();
        let id = tape.leaf(t(vec![7.5; n], &[1, n]), false);
        let loss = cross_entropy(&mut tape, id, &[n - 1]);
        let got = tape.value(loss).to_vec()[0];
        assert!((got - want).abs() < 1e-12, "n = {n}, last target, got {got}");
    }

    // A batch of uniform predictions gives the same number, because the
    // reduction is a MEAN. A sum would give this times the batch size.
    let n = 10usize;
    let batch = 4usize;
    let mut tape: Tape<f64> = Tape::new();
    let id = tape.leaf(t(vec![0.0; batch * n], &[batch, n]), false);
    let loss = cross_entropy(&mut tape, id, &[0, 3, 7, 9]);
    let got = tape.value(loss).to_vec()[0];
    assert!(
        (got - std::f64::consts::LN_10).abs() < 1e-12,
        "a batch of {batch} uniform rows gave {got}, want ln(10). \
         Got {batch}x that? You summed where you must average."
    );

    // A confidently correct prediction has a loss near zero.
    let mut tape: Tape<f64> = Tape::new();
    let id = tape.leaf(t(vec![0.0, 0.0, 30.0], &[1, 3]), false);
    let loss = cross_entropy(&mut tape, id, &[2]);
    assert!(
        tape.value(loss).to_vec()[0] < 1e-11,
        "a confident correct prediction must cost almost nothing"
    );

    // A confidently WRONG prediction must be large and FINITE. The naive
    // ln(softmax(x)) form returns inf here.
    let mut tape: Tape<f64> = Tape::new();
    let id = tape.leaf(t(vec![0.0, 0.0, 800.0], &[1, 3]), false);
    let loss = cross_entropy(&mut tape, id, &[0]);
    let got = tape.value(loss).to_vec()[0];
    assert!(got.is_finite(), "a confident wrong prediction gave {got}");
    assert!((got - 800.0).abs() < 1e-6, "the loss is about the logit gap, got {got}");
}

// ---------------------------------------------------------------------------

/// Traps the new `LogSumExp` backward arm, on its own.
///
/// If this fails and `cross_entropy_gradcheck` also fails, fix this one first.
/// The cross-entropy gradient is two lines away from this one.
#[test]
fn logsumexp_gradcheck() {
    let mut rng = Rng::seed(6_006);

    for &axis in &[0usize, 1] {
        let inputs = vec![randn(&mut rng, &[3, 4])];
        let r = grad_check(
            |tape: &mut Tape<f64>, ids: &[NodeId]| logsumexp(tape, ids[0], axis),
            &inputs,
            H,
            TOL,
        );
        assert!(r.passed, "logsumexp axis {axis}: max_rel_err {:e}", r.max_rel_err);

        let r = grad_check_projected(
            |tape: &mut Tape<f64>, ids: &[NodeId]| logsumexp(tape, ids[0], axis),
            &inputs,
            H,
            TOL,
            axis as u64 + 1,
        );
        assert!(r.passed, "logsumexp projected, axis {axis}: {:e}", r.max_rel_err);
    }

    // softmax and log_softmax are compositions, so their gradients come from
    // the arms you already own. Check them anyway: a wrong keepdim in
    // `LogSumExp` shows up as a broadcast fault here and nowhere else.
    let inputs = vec![randn(&mut rng, &[3, 4])];
    for (name, f) in [
        ("softmax", 0u8),
        ("log_softmax", 1u8),
    ] {
        let r = grad_check(
            |tape: &mut Tape<f64>, ids: &[NodeId]| {
                if f == 0 {
                    softmax(tape, ids[0], 1)
                } else {
                    log_softmax(tape, ids[0], 1)
                }
            },
            &inputs,
            H,
            TOL,
        );
        assert!(r.passed, "{name}: max_rel_err {:e} at {:?}", r.max_rel_err, r.worst_index);
    }
}

// ---------------------------------------------------------------------------

/// Traps the fused cross-entropy gradient.
///
/// Two independent checks. The finite-difference oracle from Day 12, and a
/// direct comparison against `softmax(logits) − onehot(targets)` scaled by
/// `1 / predictions`. A fault that is wrong in the same way in both the rule
/// and one of the checks still fails the other.
#[test]
fn cross_entropy_gradcheck() {
    let mut rng = Rng::seed(50_257);

    let targets = [2usize, 0, 3, 1];
    let inputs = vec![randn(&mut rng, &[4, 5])];

    let r = grad_check(
        |tape: &mut Tape<f64>, ids: &[NodeId]| cross_entropy(tape, ids[0], &targets),
        &inputs,
        H,
        TOL,
    );
    assert!(
        r.passed,
        "cross_entropy: max_rel_err {:e} at {:?}. A CONSTANT factor on every \
         element is the batch normalisation, and you fix that one yourself.",
        r.max_rel_err, r.worst_index
    );

    // The analytic gradient, built directly. Independent of your backward arm.
    let logits = inputs[0].clone();
    let probs = forward(|tape: &mut Tape<f64>, id: NodeId| softmax(tape, id, 1), logits.clone());
    let scale = 1.0 / targets.len() as f64;
    let mut want = probs.to_vec();
    for (row, &tgt) in targets.iter().enumerate() {
        want[row * 5 + tgt] -= 1.0;
    }
    let want = t(want.into_iter().map(|v| v * scale).collect(), &[4, 5]);

    let mut tape: Tape<f64> = Tape::new();
    let id = tape.leaf(logits, true);
    let loss = cross_entropy(&mut tape, id, &targets);
    tape.backward(loss);
    assert_all_close(
        tape.grad(id).expect("logits have a gradient"),
        &want,
        1e-12,
        "the fused gradient against softmax minus onehot, over the batch size",
    );

    // A single prediction, so the 1/B factor is 1 and cannot hide a fault.
    let inputs = vec![randn(&mut rng, &[1, 6])];
    let r = grad_check(
        |tape: &mut Tape<f64>, ids: &[NodeId]| cross_entropy(tape, ids[0], &[4]),
        &inputs,
        H,
        TOL,
    );
    assert!(r.passed, "cross_entropy, batch of 1: {:e}", r.max_rel_err);

    // A larger batch, so a wrong factor scales differently from the case above.
    let inputs = vec![randn(&mut rng, &[8, 3])];
    let r = grad_check(
        |tape: &mut Tape<f64>, ids: &[NodeId]| {
            cross_entropy(tape, ids[0], &[0, 1, 2, 0, 1, 2, 0, 1])
        },
        &inputs,
        H,
        TOL,
    );
    assert!(r.passed, "cross_entropy, batch of 8: {:e}", r.max_rel_err);
}
