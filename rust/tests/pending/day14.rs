//! Day 14 acceptance tests — optimizers, and the Phase 0 capstone.
//!
//! The mentor writes this file. You write the code that makes it pass.
//! Do not edit the assertions. If a test is wrong, say so and argue it.
//!
//! Move this file to `rust/tests/day14.rs` at the start of Day 14, then run
//! `cargo test --test day14 --release` and watch it fail.
//!
//! **Run this file in release.** `spiral_classification` trains a real
//! network. A debug build takes minutes where release takes seconds.
//!
//! Expected module layout. Add two lines to `src/lib.rs`:
//!
//! ```text
//! pub mod nn;      // Linear
//! pub mod optim;   // Optimizer, Sgd, AdamW
//! ```
//!
//! ## The decisions this file fixes
//!
//! 1. `Sgd::new(lr, momentum)` and `AdamW::new(lr, beta1, beta2, eps, wd)`
//!    exist. The moment buffers are private, so a constructor is the only way
//!    in, and the tests need one.
//!
//! 2. The optimizer indexes its state by POSITION in the `params` slice. The
//!    caller passes the parameters in the same order every step.
//!
//! 3. **`forward` takes a fourth argument, and this differs from the card.**
//!    The card writes `forward(&self, tape, x) -> NodeId`. A forward pass has
//!    to push the layer's weights as leaves to get ids for them, and the
//!    optimizer needs exactly those ids to find the gradients. So:
//!
//!    ```text
//!    pub fn forward(&self, tape: &mut Tape<T>, x: NodeId,
//!                   params: &mut Vec<(NodeId, Tensor<T>)>) -> NodeId;
//!    ```
//!
//!    It appends `(id, tensor)` for `w`, then for `b` when present, and
//!    returns the output node. `&self` still holds: the layer is not
//!    modified, the tape and the params list are. `DAY_14.md` section 4.2.
//!
//! 4. A parameter whose gradient is `None` this step is skipped, not a panic.
//!
//! ## The one arguable thing in this file
//!
//! `spiral_classification` contains a training loop. `CLAUDE.md` says the
//! mentor does not write training loops, and this one is here because the
//! acceptance test has to be executable. The model, the optimizer, the loss
//! and the layer inside it are all yours. **If you would rather own the loop,
//! move it into `src/bin/spiral.rs` and let this test assert on the CSV it
//! writes. Say so and I will change the file.**

mod common;

use std::fs;
use std::io::Write;

use rustgpt::autograd::{NodeId, Op, Tape, add, cross_entropy, mul, sum_all, tanh};
use rustgpt::gradcheck::grad_check;
use rustgpt::nn::Linear;
use rustgpt::optim::{AdamW, Optimizer, Sgd};
use rustgpt::rng::Rng;
use rustgpt::tensor::Tensor;

fn t(data: Vec<f64>, shape: &[usize]) -> Tensor<f64> {
    Tensor::from_vec(data, shape)
}

fn scalar(v: f64) -> Tensor<f64> {
    Tensor::from_vec(vec![v], &[1])
}

// ---------------------------------------------------------------------------

/// Traps the sign of the step, and a missing learning-rate multiply.
///
/// On `f(x) = (x − 3)²` from `x = 0` with `lr = 0.1`, the error shrinks by a
/// factor of 0.8 per step, so 47 steps reach 1e-4. A step in the wrong
/// direction diverges. A missing `lr` overshoots and oscillates.
#[test]
fn sgd_descends_quadratic() {
    let mut x = scalar(0.0);
    let mut opt = Sgd::new(0.1, 0.0);

    let mut steps_taken = 0usize;
    for step in 1..=200 {
        let mut tape: Tape<f64> = Tape::new();
        let x_id = tape.leaf(x.clone(), true);
        let minus_three = tape.leaf(scalar(-3.0), false);

        let d = add(&mut tape, x_id, minus_three);
        let sq = mul(&mut tape, d, d);
        let loss = sum_all(&mut tape, sq);
        tape.backward(loss);

        let mut params = vec![(x_id, x.clone())];
        opt.step(&tape, &mut params);
        x = params[0].1.clone();

        steps_taken = step;
        if (x.to_vec()[0] - 3.0).abs() < 1e-4 {
            break;
        }
        assert!(
            x.to_vec()[0].abs() < 1e6,
            "diverged at step {step} to {}. The step has the wrong sign.",
            x.to_vec()[0]
        );
    }

    assert!(
        (x.to_vec()[0] - 3.0).abs() < 1e-4,
        "did not reach 3.0 in 200 steps. Ended at {}",
        x.to_vec()[0]
    );
    assert!(
        steps_taken < 200,
        "took every one of the 200 steps, so convergence is far too slow"
    );

    // Momentum must not break the same problem. It converges faster or
    // it oscillates once, and either way it arrives.
    let mut x = scalar(0.0);
    let mut opt = Sgd::new(0.05, 0.9);
    for _ in 0..200 {
        let mut tape: Tape<f64> = Tape::new();
        let x_id = tape.leaf(x.clone(), true);
        let minus_three = tape.leaf(scalar(-3.0), false);
        let d = add(&mut tape, x_id, minus_three);
        let sq = mul(&mut tape, d, d);
        let loss = sum_all(&mut tape, sq);
        tape.backward(loss);
        let mut params = vec![(x_id, x.clone())];
        opt.step(&tape, &mut params);
        x = params[0].1.clone();
    }
    assert!(
        (x.to_vec()[0] - 3.0).abs() < 1e-4,
        "with momentum, ended at {}",
        x.to_vec()[0]
    );
}

// ---------------------------------------------------------------------------

/// Traps a missing or misplaced bias correction.
///
/// The first AdamW step has magnitude close to `lr` at ANY gradient scale.
/// Three scales six orders of magnitude apart, and all three must give the
/// same step size. Without the correction they differ enormously.
///
/// Confirm this by hand for one scalar parameter. It is self-check 1.
#[test]
fn adamw_bias_correction_first_step() {
    let lr = 1e-3f64;

    for &g_scale in &[1e-3f64, 1.0, 1e3] {
        // L = g_scale * x, so dL/dx is exactly g_scale.
        let x = scalar(5.0);
        let mut opt = AdamW::new(lr, 0.9, 0.999, 1e-8, 0.0);

        let mut tape: Tape<f64> = Tape::new();
        let x_id = tape.leaf(x.clone(), true);
        let c = tape.leaf(scalar(g_scale), false);
        let prod = mul(&mut tape, x_id, c);
        let loss = sum_all(&mut tape, prod);
        tape.backward(loss);

        assert!(
            (tape.grad(x_id).unwrap().to_vec()[0] - g_scale).abs() < 1e-9,
            "the test's own setup is wrong: the gradient is not {g_scale}"
        );

        let mut params = vec![(x_id, x.clone())];
        opt.step(&tape, &mut params);

        let moved = (params[0].1.to_vec()[0] - 5.0).abs();
        assert!(
            (moved - lr).abs() < 1e-4 * lr,
            "first step at gradient scale {g_scale} moved {moved:e}, want about {lr:e}. \
             A step that scales with the gradient means the bias correction is missing."
        );
    }

    // The correction must be applied to BOTH moments. Applying it to one
    // gives a first step that is wrong by a large constant factor.
    let x = scalar(0.0);
    let mut opt = AdamW::new(lr, 0.9, 0.999, 1e-8, 0.0);
    let mut tape: Tape<f64> = Tape::new();
    let x_id = tape.leaf(x.clone(), true);
    let c = tape.leaf(scalar(2.0), false);
    let prod = mul(&mut tape, x_id, c);
    let loss = sum_all(&mut tape, prod);
    tape.backward(loss);
    let mut params = vec![(x_id, x)];
    opt.step(&tape, &mut params);
    let moved = params[0].1.to_vec()[0].abs();
    assert!(
        (moved / lr - 1.0).abs() < 1e-4,
        "step / lr was {}, want 1. Check that BOTH m and v are corrected.",
        moved / lr
    );
}

// ---------------------------------------------------------------------------

/// Traps decay coupled into the gradient.
///
/// A zero gradient is the one input where the two forms are unmistakably
/// different. Decoupled, the parameter shrinks by exactly `lr * wd * theta`.
/// Coupled, the decay is routed through the second moment, and the result
/// depends on the gradient history instead.
#[test]
fn adamw_weight_decay_is_decoupled() {
    let lr = 0.1f64;
    let wd = 0.5f64;
    let theta0 = 2.0f64;

    let x = scalar(theta0);
    let mut opt = AdamW::new(lr, 0.9, 0.999, 1e-8, wd);

    // L = sum(x * 0), so the gradient of x is exactly zero.
    let mut tape: Tape<f64> = Tape::new();
    let x_id = tape.leaf(x.clone(), true);
    let zero = tape.leaf(scalar(0.0), false);
    let prod = mul(&mut tape, x_id, zero);
    let loss = sum_all(&mut tape, prod);
    tape.backward(loss);

    let g = tape.grad(x_id).expect("x has a gradient").to_vec()[0];
    assert_eq!(g, 0.0, "the test's own setup is wrong: the gradient is {g}");

    let mut params = vec![(x_id, x)];
    opt.step(&tape, &mut params);

    let got = params[0].1.to_vec()[0];
    let want = theta0 - lr * wd * theta0; // 2.0 - 0.1*0.5*2.0 = 1.9
    assert!(
        (got - want).abs() < 1e-12,
        "decoupled decay at zero gradient must give exactly {want}. Got {got}."
    );

    // At wd = 0 and a zero gradient, nothing moves at all.
    let x = scalar(theta0);
    let mut opt = AdamW::new(lr, 0.9, 0.999, 1e-8, 0.0);
    let mut tape: Tape<f64> = Tape::new();
    let x_id = tape.leaf(x.clone(), true);
    let zero = tape.leaf(scalar(0.0), false);
    let prod = mul(&mut tape, x_id, zero);
    let loss = sum_all(&mut tape, prod);
    tape.backward(loss);
    let mut params = vec![(x_id, x)];
    opt.step(&tape, &mut params);
    assert!(
        (params[0].1.to_vec()[0] - theta0).abs() < 1e-12,
        "with no gradient and no decay the parameter must not move"
    );

    // Two decay steps compound multiplicatively: theta * (1 - lr*wd)^2.
    let x = scalar(theta0);
    let mut opt = AdamW::new(lr, 0.9, 0.999, 1e-8, wd);
    let mut current = x;
    for _ in 0..2 {
        let mut tape: Tape<f64> = Tape::new();
        let x_id = tape.leaf(current.clone(), true);
        let zero = tape.leaf(scalar(0.0), false);
        let prod = mul(&mut tape, x_id, zero);
        let loss = sum_all(&mut tape, prod);
        tape.backward(loss);
        let mut params = vec![(x_id, current.clone())];
        opt.step(&tape, &mut params);
        current = params[0].1.clone();
    }
    let want2 = theta0 * (1.0 - lr * wd) * (1.0 - lr * wd);
    assert!(
        (current.to_vec()[0] - want2).abs() < 1e-12,
        "two decay steps must give {want2}, got {}",
        current.to_vec()[0]
    );
}

// ---------------------------------------------------------------------------

/// Two interleaved spirals. `n` points per class, 2 classes.
///
/// Returns the features `[2n, 2]` and the class labels, length `2n`.
/// A test fixture, so this one is written out.
///
/// **`TURN` is the whole difficulty of the problem, and it is measured, not
/// guessed.** Each arm sweeps `TURN` radians while its radius grows from 0.1
/// to 1.1, and the second arm is the first rotated by pi.
///
///   TURN = 3.5 rad (0.56 turns)  ->  the best possible straight line scores
///                                    93 percent, and the MLP passes 99
///                                    percent inside 30 epochs. Proves nothing.
///   TURN = 4*pi    (2.00 turns)  ->  the best possible straight line scores
///                                    60.5 percent, barely above chance, and
///                                    the MLP needs 270 to 400 epochs.
///
/// The second setting is the one that makes the 99 percent bar evidence of a
/// working non-linear model. `capstone_r0b/README.md` section 2 has the sweep.
const TURN: f64 = 4.0 * std::f64::consts::PI;

fn spiral_data(n: usize, rng: &mut Rng) -> (Tensor<f64>, Vec<usize>) {
    let mut feats = Vec::with_capacity(4 * n);
    let mut labels = Vec::with_capacity(2 * n);

    for class in 0..2usize {
        for i in 0..n {
            let frac = i as f64 / n as f64;
            let radius = 0.1 + frac;
            let noise: f64 = rng.normal::<f64>() * 0.05;
            let theta = frac * TURN + class as f64 * std::f64::consts::PI + noise;
            feats.push(radius * theta.sin());
            feats.push(radius * theta.cos());
            labels.push(class);
        }
    }
    (Tensor::from_vec(feats, &[2 * n, 2]), labels)
}

/// Fraction of predictions whose largest logit sits at the true class.
fn accuracy(logits: &Tensor<f64>, labels: &[usize]) -> f64 {
    let classes = logits.shape()[1];
    let flat = logits.to_vec();
    let mut right = 0usize;
    for (row, &want) in labels.iter().enumerate() {
        let base = row * classes;
        let mut best = 0usize;
        for c in 1..classes {
            if flat[base + c] > flat[base + best] {
                best = c;
            }
        }
        if best == want {
            right += 1;
        }
    }
    right as f64 / labels.len() as f64
}

/// The capstone. Everything, end to end.
///
/// A 2 -> 64 -> 64 -> 2 tanh MLP, trained by your AdamW on a dataset no
/// linear model can separate, must exceed 99 percent train accuracy inside
/// 2000 epochs, and write its loss curve to `evidence/`.
///
/// The seed is fixed, so a failure is reproducible.
///
/// `cargo test` runs with the crate root as the working directory, so
/// `../evidence` is `course/evidence`.
#[test]
fn spiral_classification() {
    let mut rng = Rng::seed(2_026);
    let (x_data, labels) = spiral_data(100, &mut rng);

    let mut layers: Vec<Linear<f64>> = vec![
        Linear::new(2, 64, true, &mut rng),
        Linear::new(64, 64, true, &mut rng),
        Linear::new(64, 2, true, &mut rng),
    ];

    let mut opt = AdamW::new(3e-3, 0.9, 0.999, 1e-8, 1e-4);

    let mut curve: Vec<(usize, f64, f64)> = Vec::new();
    let mut best_acc = 0.0f64;
    let mut hit_epoch: Option<usize> = None;

    for epoch in 0..2000usize {
        let mut tape: Tape<f64> = Tape::new();
        let x = tape.leaf(x_data.clone(), false);

        // `forward` appends its own parameter leaves here, in a fixed order.
        let mut params: Vec<(NodeId, Tensor<f64>)> = Vec::new();

        let mut h = x;
        let depth = layers.len();
        for (i, layer) in layers.iter().enumerate() {
            h = layer.forward(&mut tape, h, &mut params);
            if i + 1 < depth {
                h = tanh(&mut tape, h);
            }
        }
        let logits = h;
        assert_eq!(
            params.len(),
            6,
            "3 layers with bias must contribute 6 parameter tensors"
        );

        let loss_node = cross_entropy(&mut tape, logits, &labels);
        let loss = tape.value(loss_node).to_vec()[0];

        if epoch == 0 {
            assert!(
                (loss - std::f64::consts::LN_2).abs() < 0.25,
                "step-0 loss was {loss}, want about ln(2) = 0.693. \
                 That is a forward-pass or init fault. Fix it before you tune."
            );
        }
        assert!(loss.is_finite(), "loss became {loss} at epoch {epoch}");

        tape.backward(loss_node);
        opt.step(&tape, &mut params);

        // Write the updated tensors back, in the same order `forward` used.
        let mut k = 0usize;
        for layer in layers.iter_mut() {
            layer.w = params[k].1.clone();
            k += 1;
            if layer.b.is_some() {
                layer.b = Some(params[k].1.clone());
                k += 1;
            }
        }

        let acc = accuracy(tape.value(logits), &labels);
        best_acc = best_acc.max(acc);
        if acc > 0.99 && hit_epoch.is_none() {
            hit_epoch = Some(epoch);
        }
        if epoch % 10 == 0 || acc > 0.99 {
            curve.push((epoch, loss, acc));
        }
        if let Some(e) = hit_epoch
            && epoch > e + 10
        {
            break;
        }
    }

    fs::create_dir_all("../evidence").expect("create evidence/");
    let mut f = fs::File::create("../evidence/phase0_spiral_loss.csv").expect("open csv");
    writeln!(f, "epoch,loss,train_accuracy").unwrap();
    for (e, l, a) in &curve {
        writeln!(f, "{e},{l:.8},{a:.6}").unwrap();
    }

    assert!(
        best_acc > 0.99,
        "reached only {best_acc:.4} train accuracy in 2000 epochs. \
         A loss stuck near 0.693 is a DEAD GRADIENT, not an autograd fault: \
         check the init variance first, then the learning rate."
    );
}

// ---------------------------------------------------------------------------

/// The R0b acceptance test. It traps a future you.
///
/// The match below has NO wildcard arm. Add an `Op` variant on any later day
/// and this file stops compiling, naming the variant you forgot. A test that
/// does not compile is a failing test.
///
/// **Never add `_ =>` to this match.** That single character turns a
/// compile-time guarantee into a silent missing gradient, which is the most
/// expensive class of bug in this project.
#[test]
fn all_ops_gradchecked() {
    let mut rng = Rng::seed(14_014);
    let h = 1e-5;
    let tol = 1e-5;

    // One representative value per variant. The ids are real, taken from a
    // throwaway tape, so this needs no constructor that does not exist.
    let mut probe: Tape<f64> = Tape::new();
    let a = probe.leaf(scalar(0.0), false);

    let variants: Vec<Op> = vec![
        Op::Leaf,
        Op::Add(a, a),
        Op::Mul(a, a),
        Op::MatMul(a, a),
        Op::Sum { input: a, axis: 0, keepdim: false },
        Op::Broadcast { input: a, from: vec![1, 4] },
        Op::Exp(a),
        Op::Ln(a),
        Op::Tanh(a),
        Op::Neg(a),
        Op::LogSumExp { input: a, axis: 1 },
        Op::CrossEntropy { logits: a, targets: vec![0] },
    ];

    for op in &variants {
        let passed = match op {
            Op::Leaf => true, // a leaf propagates nothing. Nothing to check.

            Op::Add(..) => grad_check(
                |tp: &mut Tape<f64>, i: &[NodeId]| add(tp, i[0], i[1]),
                &[common::randn(&mut rng, &[2, 3]), common::randn(&mut rng, &[2, 3])],
                h, tol,
            ).passed,

            Op::Mul(..) => grad_check(
                |tp: &mut Tape<f64>, i: &[NodeId]| mul(tp, i[0], i[1]),
                &[common::randn(&mut rng, &[2, 3]), common::randn(&mut rng, &[2, 3])],
                h, tol,
            ).passed,

            Op::MatMul(..) => grad_check(
                |tp: &mut Tape<f64>, i: &[NodeId]| rustgpt::autograd::matmul(tp, i[0], i[1]),
                &[common::randn(&mut rng, &[2, 3]), common::randn(&mut rng, &[3, 4])],
                h, tol,
            ).passed,

            Op::Sum { .. } => grad_check(
                |tp: &mut Tape<f64>, i: &[NodeId]| rustgpt::autograd::sum_axis(tp, i[0], 1, false),
                &[common::randn(&mut rng, &[2, 3])],
                h, tol,
            ).passed,

            Op::Broadcast { .. } => grad_check(
                |tp: &mut Tape<f64>, i: &[NodeId]| {
                    rustgpt::autograd::broadcast_to(tp, i[0], &[3, 4])
                },
                &[common::randn(&mut rng, &[1, 4])],
                h, tol,
            ).passed,

            Op::Exp(..) => grad_check(
                |tp: &mut Tape<f64>, i: &[NodeId]| rustgpt::autograd::exp(tp, i[0]),
                &[t(vec![-1.0, -0.2, 0.5, 1.4], &[2, 2])],
                h, tol,
            ).passed,

            Op::Ln(..) => grad_check(
                |tp: &mut Tape<f64>, i: &[NodeId]| rustgpt::autograd::ln(tp, i[0]),
                &[t(vec![0.5, 1.0, 2.0, 4.0], &[2, 2])],
                h, tol,
            ).passed,

            Op::Tanh(..) => grad_check(
                |tp: &mut Tape<f64>, i: &[NodeId]| tanh(tp, i[0]),
                &[t(vec![-1.5, -0.3, 0.4, 1.1], &[2, 2])],
                h, tol,
            ).passed,

            Op::Neg(..) => grad_check(
                |tp: &mut Tape<f64>, i: &[NodeId]| rustgpt::autograd::neg(tp, i[0]),
                &[common::randn(&mut rng, &[2, 3])],
                h, tol,
            ).passed,

            Op::LogSumExp { .. } => grad_check(
                |tp: &mut Tape<f64>, i: &[NodeId]| rustgpt::autograd::logsumexp(tp, i[0], 1),
                &[common::randn(&mut rng, &[2, 3])],
                h, tol,
            ).passed,

            Op::CrossEntropy { .. } => grad_check(
                |tp: &mut Tape<f64>, i: &[NodeId]| cross_entropy(tp, i[0], &[1, 0]),
                &[common::randn(&mut rng, &[2, 3])],
                h, tol,
            ).passed,
        };

        assert!(passed, "no passing gradient check for {op:?}");
    }
}
