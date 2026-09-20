//! Day 10 acceptance tests — backward for elementwise and broadcast.
//!
//! The mentor writes this file. You write the code that makes it pass.
//! Do not edit the assertions. If a test is wrong, say so and argue it.
//!
//! Move this file to `rust/tests/day10.rs` at the start of Day 10, then run
//! `cargo test --test day10` and watch it fail.
//!
//! ## The decisions this file fixes
//!
//! 1. The forward helpers from `DAY_10.md` section 4.2 exist:
//!    `neg`, `exp`, `ln`, `tanh`, `sum_axis`, `broadcast_to`, `sum_all`.
//!    They panic on a shape error, like Day 9's `add` and `mul`.
//!
//! 2. `sum_all` reduces every axis away and gives a **rank 0** node. That
//!    follows the Day 5 rule: a rank-1 tensor reduced with `keepdim = false`
//!    gives rank 0. `backward` accepts that node as a scalar root.
//!
//! 3. `broadcast_to` records the SOURCE shape in `Op::Broadcast { from }`.
//!    The backward rule cannot recover it from the result.
//!
//! 4. `zero_grad` sets every slot back to `None`, not to `Some(zeros)`.
//!    Day 10 section 2.7 argues why those are different states.
//!
//! 5. A gradient tensor has exactly the shape of its node's value. Every
//!    assertion here checks the shape before it checks the values.

mod common;

use common::assert_all_close;
use rustgpt::autograd::{Tape, add, broadcast_to, mul, sum_all, sum_axis};
use rustgpt::tensor::Tensor;

const TOL: f64 = 1e-12;

fn t(data: Vec<f64>, shape: &[usize]) -> Tensor<f64> {
    Tensor::from_vec(data, shape)
}

// ---------------------------------------------------------------------------

/// Traps the base case, against your own hand derivation from Day 9.
///
/// This is `f = (a + b) * a` with `a = 2` and `b = 3`, which is the tape you
/// walked on paper yesterday. If this disagrees with your paper, find out
/// which of the two is wrong before you touch anything else.
#[test]
fn backward_matches_paper() {
    let mut tape: Tape<f64> = Tape::new();

    let a = tape.leaf(t(vec![2.0], &[1]), true);
    let b = tape.leaf(t(vec![3.0], &[1]), true);

    let s = add(&mut tape, a, b);
    let f = mul(&mut tape, s, a);
    let root = sum_all(&mut tape, f);

    assert_eq!(tape.value(f).to_vec(), vec![10.0], "forward: (2+3)*2");
    assert!(tape.value(root).shape().is_empty(), "the root is rank 0");

    tape.backward(root);

    let ga = tape.grad(a).expect("a has a gradient");
    let gb = tape.grad(b).expect("b has a gradient");

    assert_eq!(ga.shape(), &[1], "grad(a) has the shape of a");
    assert_eq!(gb.shape(), &[1], "grad(b) has the shape of b");

    assert_all_close(ga, &t(vec![7.0], &[1]), TOL, "d f / d a");
    assert_all_close(gb, &t(vec![2.0], &[1]), TOL, "d f / d b");
}

// ---------------------------------------------------------------------------

/// Traps `=` instead of `+=`.
///
/// `x` feeds two consumers, so its gradient is the sum of two contributions.
/// A rule that assigns keeps only whichever one the reverse loop wrote last,
/// which gives exactly `2x` or exactly `1` and never the sum.
///
/// The second block is a deeper reuse, so a fix that special-cases one shape
/// of graph still fails.
#[test]
fn diamond_accumulates() {
    let x_data = vec![1.5, -2.0, 0.25];

    // z = x*x + x        dz/dx = 2x + 1
    {
        let mut tape: Tape<f64> = Tape::new();
        let x = tape.leaf(t(x_data.clone(), &[3]), true);

        let y = mul(&mut tape, x, x);
        let z = add(&mut tape, y, x);
        let root = sum_all(&mut tape, z);
        tape.backward(root);

        let want: Vec<f64> = x_data.iter().map(|v| 2.0 * v + 1.0).collect();
        let g = tape.grad(x).expect("x has a gradient");
        assert_eq!(g.shape(), &[3]);
        assert_all_close(g, &t(want, &[3]), TOL, "d(x*x + x)/dx");
    }

    // w = (x*x + x) * x = x^3 + x^2       dw/dx = 3x^2 + 2x
    // Three separate paths reach `x`. Two contributions is not enough here.
    {
        let mut tape: Tape<f64> = Tape::new();
        let x = tape.leaf(t(x_data.clone(), &[3]), true);

        let y = mul(&mut tape, x, x);
        let z = add(&mut tape, y, x);
        let w = mul(&mut tape, z, x);
        let root = sum_all(&mut tape, w);
        tape.backward(root);

        let want: Vec<f64> = x_data.iter().map(|v| 3.0 * v * v + 2.0 * v).collect();
        let g = tape.grad(x).expect("x has a gradient");
        assert_all_close(g, &t(want, &[3]), TOL, "d(x^3 + x^2)/dx");
    }
}

// ---------------------------------------------------------------------------

/// Traps the factor-of-N error.
///
/// A broadcast backward that passes the gradient through unchanged produces
/// the right SHAPE, so nothing else in the suite notices. Only the values do.
///
/// The three blocks are the three cases in `DAY_10.md` section 2.6: a size-1
/// axis, a missing leading axis, and both at once.
#[test]
fn broadcast_backward_sums() {
    // [1,4] -> [3,4]. Each source element was read 3 times.
    {
        let mut tape: Tape<f64> = Tape::new();
        let x = tape.leaf(t(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]), true);
        let y = broadcast_to(&mut tape, x, &[3, 4]);
        let root = sum_all(&mut tape, y);
        tape.backward(root);

        let g = tape.grad(x).expect("gradient");
        assert_eq!(g.shape(), &[1, 4], "the gradient keeps the SOURCE shape");
        assert_all_close(g, &t(vec![3.0; 4], &[1, 4]), TOL, "size-1 axis");
    }

    // [4] -> [3,4]. The source has no leading axis at all, so step 2 of the
    // procedure sums it away with keepdim = false.
    {
        let mut tape: Tape<f64> = Tape::new();
        let x = tape.leaf(t(vec![1.0, 2.0, 3.0, 4.0], &[4]), true);
        let y = broadcast_to(&mut tape, x, &[3, 4]);
        let root = sum_all(&mut tape, y);
        tape.backward(root);

        let g = tape.grad(x).expect("gradient");
        assert_eq!(g.shape(), &[4], "rank 1 in, rank 1 out");
        assert_all_close(g, &t(vec![3.0; 4], &[4]), TOL, "missing leading axis");
    }

    // [2,1,3] -> [2,4,3]. A middle axis of 1, with a real axis on each side.
    {
        let mut tape: Tape<f64> = Tape::new();
        let x = tape.leaf(t((1..=6).map(f64::from).collect(), &[2, 1, 3]), true);
        let y = broadcast_to(&mut tape, x, &[2, 4, 3]);
        let root = sum_all(&mut tape, y);
        tape.backward(root);

        let g = tape.grad(x).expect("gradient");
        assert_eq!(g.shape(), &[2, 1, 3]);
        assert_all_close(g, &t(vec![4.0; 6], &[2, 1, 3]), TOL, "middle size-1 axis");
    }
}

// ---------------------------------------------------------------------------

/// Traps a `zero_grad` that writes `Some(zeros)` instead of `None`.
///
/// Both look cleared from the outside. Only one lets the next `accumulate`
/// tell a fresh node from one that has already received a contribution.
#[test]
fn zero_grad_clears() {
    let mut tape: Tape<f64> = Tape::new();
    let x = tape.leaf(t(vec![2.0, 3.0], &[2]), true);
    let y = mul(&mut tape, x, x);
    let root = sum_all(&mut tape, y);

    tape.backward(root);
    assert!(tape.grad(x).is_some(), "backward filled the gradient");
    assert_all_close(
        tape.grad(x).unwrap(),
        &t(vec![4.0, 6.0], &[2]),
        TOL,
        "d(x*x)/dx before clearing",
    );

    tape.zero_grad();
    assert!(tape.grad(x).is_none(), "zero_grad restores None, not Some(zeros)");

    // And a second backward on the same tape rebuilds the same answer, so
    // clearing did not corrupt anything.
    tape.backward(root);
    assert_all_close(
        tape.grad(x).expect("gradient after the second backward"),
        &t(vec![4.0, 6.0], &[2]),
        TOL,
        "d(x*x)/dx after clearing and rerunning",
    );
}

// ---------------------------------------------------------------------------

/// Traps the `keepdim = false` reshape in the `Sum` backward.
///
/// Reducing axis 1 of a `[2,3]` gives a `[2]`. Broadcasting a `[2]` straight
/// back to `[2,3]` right-aligns the shapes and fails, or worse, succeeds with
/// the values on the wrong axis. The gradient must be reshaped to `[2,1]`
/// first. A weight that differs per row makes a wrong axis visible.
#[test]
fn sum_backward_restores_axis() {
    // s = sum(x, axis=1);  L = sum(s * w)   ->   dL/dx[i][j] = w[i]
    {
        let mut tape: Tape<f64> = Tape::new();
        let x = tape.leaf(t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]), true);
        let w = tape.leaf(t(vec![1.0, 10.0], &[2]), false);

        let s = sum_axis(&mut tape, x, 1, false);
        assert_eq!(tape.value(s).shape(), &[2], "keepdim = false drops the axis");
        assert_eq!(tape.value(s).to_vec(), vec![6.0, 15.0], "sum over axis 1");

        let weighted = mul(&mut tape, s, w);
        let root = sum_all(&mut tape, weighted);
        tape.backward(root);

        let g = tape.grad(x).expect("gradient");
        assert_eq!(g.shape(), &[2, 3]);
        assert_all_close(
            g,
            &t(vec![1.0, 1.0, 1.0, 10.0, 10.0, 10.0], &[2, 3]),
            TOL,
            "every element of row i gets w[i]",
        );
    }

    // The same reduction with keepdim = true. The rank never drops, so this
    // path needs no reshape, and it must give the identical gradient.
    {
        let mut tape: Tape<f64> = Tape::new();
        let x = tape.leaf(t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]), true);
        let w = tape.leaf(t(vec![1.0, 10.0], &[2, 1]), false);

        let s = sum_axis(&mut tape, x, 1, true);
        assert_eq!(tape.value(s).shape(), &[2, 1], "keepdim = true holds the axis");

        let weighted = mul(&mut tape, s, w);
        let root = sum_all(&mut tape, weighted);
        tape.backward(root);

        assert_all_close(
            tape.grad(x).expect("gradient"),
            &t(vec![1.0, 1.0, 1.0, 10.0, 10.0, 10.0], &[2, 3]),
            TOL,
            "keepdim = true gives the same gradient as keepdim = false",
        );
    }

    // Reducing axis 0 instead. A hardcoded last-axis reduction fails here.
    {
        let mut tape: Tape<f64> = Tape::new();
        let x = tape.leaf(t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]), true);
        let w = tape.leaf(t(vec![1.0, 10.0, 100.0], &[3]), false);

        let s = sum_axis(&mut tape, x, 0, false);
        assert_eq!(tape.value(s).to_vec(), vec![5.0, 7.0, 9.0], "sum over axis 0");

        let weighted = mul(&mut tape, s, w);
        let root = sum_all(&mut tape, weighted);
        tape.backward(root);

        assert_all_close(
            tape.grad(x).expect("gradient"),
            &t(vec![1.0, 10.0, 100.0, 1.0, 10.0, 100.0], &[2, 3]),
            TOL,
            "every element of column j gets w[j]",
        );
    }
}
