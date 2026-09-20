//! Day 11 acceptance tests — backward for matmul.
//!
//! The mentor writes this file. You write the code that makes it pass.
//! Do not edit the assertions. If a test is wrong, say so and argue it.
//!
//! Move this file to `rust/tests/day11.rs` at the start of Day 11, then run
//! `cargo test --test day11` and watch it fail.
//!
//! ## The decisions this file fixes
//!
//! 1. `autograd::matmul` is the forward helper that pushes `Op::MatMul`. It
//!    handles rank 2 and batched inputs, the same as `matmul::matmul`. It is
//!    imported here as `gmatmul` so that the reference kernel keeps its name.
//!
//! 2. The batch axes broadcast by the Day 4 rule, and their gradients reduce
//!    back with the same `unbroadcast` you wrote on Day 10.
//!
//! 3. **No test here uses a square matrix, and no test seeds the backward
//!    pass with an all-ones adjoint.** Both hide a wrong transpose. The
//!    2x2 test is the one exception on shape, and it compensates by using
//!    a non-symmetric weighting so the wrong arrangement gives a wrong
//!    number rather than the same one.

mod common;

use common::{assert_all_close, randn};
use rustgpt::autograd::{Tape, matmul as gmatmul, mul, sum_all};
use rustgpt::matmul::matmul_naive;
use rustgpt::rng::Rng;
use rustgpt::tensor::Tensor;

const TOL: f64 = 1e-9;

fn t(data: Vec<f64>, shape: &[usize]) -> Tensor<f64> {
    Tensor::from_vec(data, shape)
}

/// Pulls batch element `b` out of a rank-3 tensor as a fresh rank-2 tensor.
///
/// It reads through `to_vec`, which walks the strides in logical order, so it
/// works for a view as well as for a freshly allocated tensor.
fn batch_of(x: &Tensor<f64>, b: usize, shape: &[usize]) -> Tensor<f64> {
    let per: usize = shape.iter().product();
    let flat = x.to_vec();
    Tensor::from_vec(flat[b * per..(b + 1) * per].to_vec(), shape)
}

// ---------------------------------------------------------------------------

/// Traps the wrong transpose, in the cheapest possible way.
///
/// `M`, `K` and `N` are all different, so an arrangement with the transpose
/// on the wrong operand cannot even form a legal product. This test fires
/// before any value is compared, and its failure names the fault directly.
#[test]
fn matmul_backward_shapes() {
    let mut rng = Rng::seed(11_011);

    let a_t: Tensor<f64> = randn(&mut rng, &[2, 3]);
    let b_t: Tensor<f64> = randn(&mut rng, &[3, 4]);
    // A non-uniform weighting, so the incoming adjoint is not all ones.
    let w_t: Tensor<f64> = randn(&mut rng, &[2, 4]);

    let mut tape: Tape<f64> = Tape::new();
    let a = tape.leaf(a_t, true);
    let b = tape.leaf(b_t, true);
    let w = tape.leaf(w_t, false);

    let c = gmatmul(&mut tape, a, b);
    assert_eq!(tape.value(c).shape(), &[2, 4], "forward shape");

    let weighted = mul(&mut tape, c, w);
    let root = sum_all(&mut tape, weighted);
    tape.backward(root);

    assert_eq!(
        tape.grad(a).expect("grad_a").shape(),
        &[2, 3],
        "grad_a has the shape of a"
    );
    assert_eq!(
        tape.grad(b).expect("grad_b").shape(),
        &[3, 4],
        "grad_b has the shape of b"
    );

    // A rank-3 by rank-3 product, so the dispatch cannot be rank-2 only.
    let mut tape2: Tape<f64> = Tape::new();
    let a3 = tape2.leaf(randn(&mut rng, &[5, 2, 3]), true);
    let b3 = tape2.leaf(randn(&mut rng, &[5, 3, 4]), true);
    let w3 = tape2.leaf(randn(&mut rng, &[5, 2, 4]), false);
    let c3 = gmatmul(&mut tape2, a3, b3);
    assert_eq!(tape2.value(c3).shape(), &[5, 2, 4]);
    let weighted3 = mul(&mut tape2, c3, w3);
    let root3 = sum_all(&mut tape2, weighted3);
    tape2.backward(root3);
    assert_eq!(tape2.grad(a3).expect("grad_a3").shape(), &[5, 2, 3]);
    assert_eq!(tape2.grad(b3).expect("grad_b3").shape(), &[5, 3, 4]);
}

// ---------------------------------------------------------------------------

/// Traps an arrangement that type-checks and is wrong.
///
/// A = [[1,2],[3,4]]   B = [[5,6],[7,8]]   C = A·B = [[19,22],[43,50]]
///
/// The scalar is  L = sum(C * R)  with  R = [[1,2],[3,4]],  so the incoming
/// adjoint is R and not all ones. Neither A, B nor R is symmetric, so every
/// wrong arrangement produces a different number.
///
/// Worked by hand, then checked with a calculator:
///
///     grad_A = R · Bᵀ = [[1,2],[3,4]] · [[5,7],[6,8]] = [[17,23],[39,53]]
///     grad_B = Aᵀ · R = [[1,3],[2,4]] · [[1,2],[3,4]] = [[10,14],[14,20]]
#[test]
fn matmul_backward_2x2_by_hand() {
    let mut tape: Tape<f64> = Tape::new();

    let a = tape.leaf(t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
    let b = tape.leaf(t(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]), true);
    let r = tape.leaf(t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]), false);

    let c = gmatmul(&mut tape, a, b);
    assert_eq!(
        tape.value(c).to_vec(),
        vec![19.0, 22.0, 43.0, 50.0],
        "forward product"
    );

    let weighted = mul(&mut tape, c, r);
    let root = sum_all(&mut tape, weighted);
    tape.backward(root);

    assert_all_close(
        tape.grad(a).expect("grad_a"),
        &t(vec![17.0, 23.0, 39.0, 53.0], &[2, 2]),
        TOL,
        "grad_a",
    );
    assert_all_close(
        tape.grad(b).expect("grad_b"),
        &t(vec![10.0, 14.0, 14.0, 20.0], &[2, 2]),
        TOL,
        "grad_b",
    );

    // The same graph with an all-ones weighting. grad_A becomes the row sums
    // of Bᵀ and grad_B the column sums of A, which are DIFFERENT numbers from
    // the block above. If both blocks give the same answer, the weighting is
    // not reaching the backward pass at all.
    let mut tape2: Tape<f64> = Tape::new();
    let a2 = tape2.leaf(t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
    let b2 = tape2.leaf(t(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]), true);
    let c2 = gmatmul(&mut tape2, a2, b2);
    let root2 = sum_all(&mut tape2, c2);
    tape2.backward(root2);

    assert_all_close(
        tape2.grad(a2).expect("grad_a with ones"),
        &t(vec![11.0, 15.0, 11.0, 15.0], &[2, 2]),
        TOL,
        "grad_a under a plain sum",
    );
    assert_all_close(
        tape2.grad(b2).expect("grad_b with ones"),
        &t(vec![4.0, 4.0, 6.0, 6.0], &[2, 2]),
        TOL,
        "grad_b under a plain sum",
    );
}

// ---------------------------------------------------------------------------

/// Traps two faults at once: the batch loop, and the broadcast batch.
///
/// The expected values come from an independent route: `matmul_naive` on each
/// batch element, summed by hand. That is a different implementation from the
/// one in your backward arm, so it is a real cross-check and not a restatement.
#[test]
fn matmul_backward_batched() {
    let mut rng = Rng::seed(4_711);

    // --- equal batch sizes -------------------------------------------------
    {
        let a_t: Tensor<f64> = randn(&mut rng, &[2, 3, 4]);
        let b_t: Tensor<f64> = randn(&mut rng, &[2, 4, 5]);
        let w_t: Tensor<f64> = randn(&mut rng, &[2, 3, 5]);

        let mut tape: Tape<f64> = Tape::new();
        let a = tape.leaf(a_t.clone(), true);
        let b = tape.leaf(b_t.clone(), true);
        let w = tape.leaf(w_t.clone(), false);

        let c = gmatmul(&mut tape, a, b);
        assert_eq!(tape.value(c).shape(), &[2, 3, 5]);
        let weighted = mul(&mut tape, c, w);
        let root = sum_all(&mut tape, weighted);
        tape.backward(root);

        let ga = tape.grad(a).expect("grad_a");
        let gb = tape.grad(b).expect("grad_b");
        assert_eq!(ga.shape(), &[2, 3, 4]);
        assert_eq!(gb.shape(), &[2, 4, 5]);

        // Independent route, one batch element at a time. `batch_of` rebuilds
        // a rank-2 tensor from the flat logical order, so this cross-check
        // leans on nothing but `to_vec` and `from_vec`.
        for batch in 0..2 {
            let a_i = batch_of(&a_t, batch, &[3, 4]);
            let b_i = batch_of(&b_t, batch, &[4, 5]);
            let w_i = batch_of(&w_t, batch, &[3, 5]);

            let want_a = matmul_naive(&w_i, &b_i.transpose(0, 1).unwrap()).unwrap();
            let want_b = matmul_naive(&a_i.transpose(0, 1).unwrap(), &w_i).unwrap();

            let got_a = batch_of(ga, batch, &[3, 4]);
            let got_b = batch_of(gb, batch, &[4, 5]);

            assert_all_close(&got_a, &want_a, TOL, &format!("grad_a batch {batch}"));
            assert_all_close(&got_b, &want_b, TOL, &format!("grad_b batch {batch}"));
        }
    }

    // --- a broadcast batch: [1,3,4] against [2,4,5] ------------------------
    // `a` holds ONE matrix that took part in TWO products, so its gradient is
    // the SUM of the two per-batch gradients, reduced back to [1,3,4].
    {
        let a_t: Tensor<f64> = randn(&mut rng, &[1, 3, 4]);
        let b_t: Tensor<f64> = randn(&mut rng, &[2, 4, 5]);
        let w_t: Tensor<f64> = randn(&mut rng, &[2, 3, 5]);

        let mut tape: Tape<f64> = Tape::new();
        let a = tape.leaf(a_t.clone(), true);
        let b = tape.leaf(b_t.clone(), true);
        let w = tape.leaf(w_t.clone(), false);

        let c = gmatmul(&mut tape, a, b);
        assert_eq!(tape.value(c).shape(), &[2, 3, 5], "the batch broadcast");
        let weighted = mul(&mut tape, c, w);
        let root = sum_all(&mut tape, weighted);
        tape.backward(root);

        let ga = tape.grad(a).expect("grad_a");
        assert_eq!(
            ga.shape(),
            &[1, 3, 4],
            "grad_a reduces back to the OPERAND shape, not the broadcast shape"
        );

        // Sum the two per-batch gradients by hand.
        let mut want = vec![0.0f64; 12];
        for batch in 0..2 {
            let b_i = batch_of(&b_t, batch, &[4, 5]);
            let w_i = batch_of(&w_t, batch, &[3, 5]);
            let part = matmul_naive(&w_i, &b_i.transpose(0, 1).unwrap()).unwrap();
            for (acc, v) in want.iter_mut().zip(part.to_vec()) {
                *acc += v;
            }
        }
        assert_all_close(ga, &t(want, &[1, 3, 4]), TOL, "grad_a summed over the batch");

        // grad_b keeps its own batch axis, untouched.
        assert_eq!(tape.grad(b).expect("grad_b").shape(), &[2, 4, 5]);
    }
}
