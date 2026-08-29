//! Day 6 acceptance tests — the naive matmul, and the oracle discipline.
//!
//! The mentor writes this file. You write the code that makes it pass.
//! Do not edit the assertions. If a test is wrong, say so and argue it.
//!
//! Move this file to `rust/tests/day6.rs` at the start of Day 6, then run
//! `cargo test --test day6` and watch it fail.
//!
//! Expected module layout. Add one line to `src/lib.rs`:
//!
//! ```text
//! pub mod matmul;   // matmul_naive, matmul
//! ```
//!
//! ## The rank split, fixed here
//!
//! `matmul_naive` takes rank 2 only. It is the reference, so it stays as
//! simple as a thing can be. Any other rank is an error.
//!
//! `matmul` takes rank 2 or higher. The last two axes are the matrix. Every
//! axis before them is a batch axis, and the batch axes broadcast by the
//! Day 4 rule.
//!
//! ## The helpers
//!
//! `mod common;` pulls in `rust/tests/common/mod.rs`. Cargo does not build
//! that file as a test binary, because the file is not named `main.rs`.
//! It holds `randn`, `identity` and `assert_all_close`.
//!
//! `assert_all_close` uses a mixed tolerance: the limit is
//! `tol * (1 + |want|)`. Read the note in `common/mod.rs` for why.

mod common;

use common::{assert_all_close, identity, randn};
use rustgpt::rng::Rng;
use rustgpt::tensor::Tensor;
use rustgpt::matmul::{matmul, matmul_naive};

// ---------------------------------------------------------------------------

/// The product every linear-algebra course opens with. If you cannot get this
/// one right, nothing after it matters.
#[test]
fn matmul_2x3_3x2_by_hand() {
    // A = [[1, 2, 3],        B = [[ 7,  8],
    //      [4, 5, 6]]             [ 9, 10],
    //                             [11, 12]]
    //
    // C[0][0] = 1*7 + 2*9  + 3*11 = 58
    // C[0][1] = 1*8 + 2*10 + 3*12 = 64
    // C[1][0] = 4*7 + 5*9  + 6*11 = 139
    // C[1][1] = 4*8 + 5*10 + 6*12 = 154
    let a = Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::from_vec(vec![7.0f64, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);

    let c = matmul_naive(&a, &b).expect("2x3 times 3x2");
    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.to_vec(), vec![58.0, 64.0, 139.0, 154.0]);
    assert!(c.is_contiguous(), "a fresh result is contiguous");

    // The batched entry point must agree with the reference at rank 2.
    assert_eq!(matmul(&a, &b).unwrap().to_vec(), c.to_vec());

    // 1 by 1 times 1 by 1. The smallest case is where an off-by-one lives.
    let p = Tensor::from_vec(vec![3.0f64], &[1, 1]);
    let q = Tensor::from_vec(vec![5.0f64], &[1, 1]);
    assert_eq!(matmul_naive(&p, &q).unwrap().to_vec(), vec![15.0]);

    // A row times a column is one number. A column times a row is a matrix.
    // Same six inputs, two different shapes. This traps M and N swapped.
    let row = Tensor::from_vec(vec![1.0f64, 2.0, 3.0], &[1, 3]);
    let col = Tensor::from_vec(vec![4.0f64, 5.0, 6.0], &[3, 1]);

    let inner = matmul_naive(&row, &col).unwrap();
    assert_eq!(inner.shape(), &[1, 1]);
    assert_eq!(inner.to_vec(), vec![32.0]); // 1*4 + 2*5 + 3*6

    let outer = matmul_naive(&col, &row).unwrap();
    assert_eq!(outer.shape(), &[3, 3]);
    assert_eq!(
        outer.to_vec(),
        vec![4.0, 8.0, 12.0, 5.0, 10.0, 15.0, 6.0, 12.0, 18.0]
    );
}

/// A times the identity is A, element for element and bit for bit.
///
/// This one is exact, not approximate. Every discarded term is `x * 0.0`,
/// which is exactly 0.0, and adding 0.0 changes nothing. If this test needs
/// a tolerance, your accumulator does something you did not intend.
#[test]
fn matmul_identity() {
    let mut rng = Rng::seed(2);
    let a: Tensor<f64> = randn(&mut rng, &[5, 4]);

    let right = matmul_naive(&a, &identity::<f64>(4)).expect("A times I");
    assert_eq!(right.shape(), &[5, 4]);
    assert_eq!(right.to_vec(), a.to_vec(), "A * I must be A, exactly");

    let left = matmul_naive(&identity::<f64>(5), &a).expect("I times A");
    assert_eq!(left.to_vec(), a.to_vec(), "I * A must be A, exactly");

    // In f32 as well. The library is generic. Only the test picks a type.
    let mut rng32 = Rng::seed(3);
    let f: Tensor<f32> = randn(&mut rng32, &[3, 6]);
    assert_eq!(
        matmul_naive(&f, &identity::<f32>(6)).unwrap().to_vec(),
        f.to_vec()
    );
}

/// `(A·B)ᵀ == Bᵀ·Aᵀ`. This is the identity that every attention
/// implementation leans on, and it is the first test that feeds a
/// non-contiguous view into the kernel.
///
/// `transpose` returns a strided view. If `matmul_naive` reads the buffer
/// directly instead of going through the index formula, it fails here and
/// passes everything above.
#[test]
fn matmul_transpose_identity() {
    let mut rng = Rng::seed(4);
    let a: Tensor<f32> = randn(&mut rng, &[6, 4]);
    let b: Tensor<f32> = randn(&mut rng, &[4, 5]);

    let ab_t = matmul_naive(&a, &b).unwrap().transpose(0, 1).unwrap();
    let bt = b.transpose(0, 1).unwrap();
    let at = a.transpose(0, 1).unwrap();
    let bt_at = matmul_naive(&bt, &at).expect("B^T times A^T");

    assert_eq!(ab_t.shape(), &[5, 6]);
    assert_eq!(bt_at.shape(), &[5, 6]);
    assert_all_close(&bt_at, &ab_t, 1e-5, "(A*B)^T == B^T * A^T");

    // A strided operand and an explicit copy of it give the same product.
    // `at` is [4,6] and strided. `at.contiguous()` is [4,6] and packed.
    let at_copy = at.contiguous();
    let d: Tensor<f32> = randn(&mut rng, &[6, 3]);
    assert!(!at.is_contiguous() && at_copy.is_contiguous());
    assert_all_close(
        &matmul_naive(&at, &d).unwrap(),
        &matmul_naive(&at_copy, &d).unwrap(),
        1e-6,
        "a strided view and its copy give the same product",
    );
}

/// `(A·B)·C ≈ A·(B·C)`. Matrix multiplication is associative in exact
/// arithmetic. In floating point the two sides differ by rounding only,
/// so a wide gap here means a real bug and not a precision limit.
#[test]
fn matmul_associative() {
    let mut rng = Rng::seed(5);
    let a: Tensor<f32> = randn(&mut rng, &[8, 8]);
    let b: Tensor<f32> = randn(&mut rng, &[8, 8]);
    let c: Tensor<f32> = randn(&mut rng, &[8, 8]);

    let left = matmul_naive(&matmul_naive(&a, &b).unwrap(), &c).unwrap();
    let right = matmul_naive(&a, &matmul_naive(&b, &c).unwrap()).unwrap();
    assert_all_close(&left, &right, 1e-4, "(A*B)*C == A*(B*C)");

    // Non-square, so a wrong dimension cannot hide behind a square shape.
    let p: Tensor<f64> = randn(&mut rng, &[3, 5]);
    let q: Tensor<f64> = randn(&mut rng, &[5, 7]);
    let r: Tensor<f64> = randn(&mut rng, &[7, 2]);
    let l2 = matmul_naive(&matmul_naive(&p, &q).unwrap(), &r).unwrap();
    let r2 = matmul_naive(&p, &matmul_naive(&q, &r).unwrap()).unwrap();
    assert_eq!(l2.shape(), &[3, 2]);
    assert_all_close(&l2, &r2, 1e-12, "(P*Q)*R == P*(Q*R) in f64");
}

/// The batch axes broadcast by the Day 4 rule. The last two axes are the
/// matrix. This is the shape logic that multi-head attention runs on.
#[test]
fn matmul_batched_shapes() {
    let mut rng = Rng::seed(6);
    let a: Tensor<f64> = randn(&mut rng, &[2, 3, 4]);
    let b: Tensor<f64> = randn(&mut rng, &[2, 4, 5]);

    let c = matmul(&a, &b).expect("batched [2,3,4] times [2,4,5]");
    assert_eq!(c.shape(), &[2, 3, 5]);

    // Every batch must equal the rank 2 product of that batch. This is the
    // oracle discipline applied inside one test: the slow correct function
    // checks the general one.
    for n in 0..2 {
        let an = a.slice(0, n, n + 1).unwrap().contiguous().reshape(&[3, 4]).unwrap();
        let bn = b.slice(0, n, n + 1).unwrap().contiguous().reshape(&[4, 5]).unwrap();
        let cn = c.slice(0, n, n + 1).unwrap().contiguous().reshape(&[3, 5]).unwrap();
        assert_all_close(
            &matmul_naive(&an, &bn).unwrap(),
            &cn,
            1e-12,
            &format!("batch {n}"),
        );
    }

    // A batch of 1 stretches against a batch of 2, with no copy of A.
    let a1: Tensor<f64> = randn(&mut rng, &[1, 3, 4]);
    let c2 = matmul(&a1, &b).expect("[1,3,4] times [2,4,5]");
    assert_eq!(c2.shape(), &[2, 3, 5]);

    let a_flat = a1.contiguous().reshape(&[3, 4]).unwrap();
    for n in 0..2 {
        let bn = b.slice(0, n, n + 1).unwrap().contiguous().reshape(&[4, 5]).unwrap();
        let cn = c2.slice(0, n, n + 1).unwrap().contiguous().reshape(&[3, 5]).unwrap();
        assert_all_close(
            &matmul_naive(&a_flat, &bn).unwrap(),
            &cn,
            1e-12,
            &format!("broadcast batch {n}"),
        );
    }

    // Two batch axes, with a stretch on each side. This is the exact shape
    // that [batch, heads, seq, head_dim] takes on Day 20.
    let x: Tensor<f64> = randn(&mut rng, &[2, 1, 3, 4]);
    let y: Tensor<f64> = randn(&mut rng, &[1, 5, 4, 6]);
    assert_eq!(matmul(&x, &y).unwrap().shape(), &[2, 5, 3, 6]);

    // Rank 2 through the batched path has no batch axes at all.
    let p: Tensor<f64> = randn(&mut rng, &[3, 4]);
    let q: Tensor<f64> = randn(&mut rng, &[4, 2]);
    assert_eq!(matmul(&p, &q).unwrap().shape(), &[3, 2]);
    assert_all_close(
        &matmul(&p, &q).unwrap(),
        &matmul_naive(&p, &q).unwrap(),
        1e-12,
        "matmul must agree with the oracle at rank 2",
    );
}

/// Every shape fault is an error and never a panic. A kernel that panics
/// on a bad shape cannot be called from a `?` chain.
#[test]
fn matmul_dim_mismatch_errors() {
    let a: Tensor<f64> = Tensor::zeros(&[2, 3]);
    let b: Tensor<f64> = Tensor::zeros(&[4, 5]);

    // The inner dimensions disagree: 3 against 4.
    assert!(matmul_naive(&a, &b).is_err());
    assert!(matmul(&a, &b).is_err());

    // Rank 1 has no matrix axes.
    let v: Tensor<f64> = Tensor::zeros(&[3]);
    assert!(matmul_naive(&v, &a).is_err());
    assert!(matmul(&v, &a).is_err());
    assert!(matmul(&a, &v).is_err());

    // The reference takes rank 2 only. Batches go through `matmul`.
    let p: Tensor<f64> = Tensor::zeros(&[2, 3, 4]);
    let q: Tensor<f64> = Tensor::zeros(&[2, 4, 5]);
    assert!(matmul_naive(&p, &q).is_err(), "the oracle stays rank 2");

    // Batch axes that do not broadcast: 2 against 3.
    let r: Tensor<f64> = Tensor::zeros(&[3, 4, 5]);
    assert!(matmul(&p, &r).is_err());
}
