//! Day 7 acceptance tests — the blocked matmul.
//!
//! The mentor writes this file. You write the code that makes it pass.
//! Do not edit the assertions. If a test is wrong, say so and argue it.
//!
//! Move this file to `rust/tests/day7.rs` at the start of Day 7, then run
//! `cargo test --test day7` and watch it fail.
//!
//! ## What this file does and does not prove
//!
//! It proves **correctness only**. Blocking changes the order of the
//! additions, so it changes the rounding, so the comparison is a tolerance
//! and not an equality.
//!
//! It proves **nothing about speed**. Speed is the benchmark, and the
//! benchmark is a separate deliverable: a committed GFLOP/s table. A test
//! that asserts a timing flakes on a busy machine and teaches you to ignore
//! red. Never write one.
//!
//! ## Why the sizes look strange
//!
//! 129 by 257 times 257 by 63 is deliberate. Every dimension is a prime or
//! an odd number, and none is a multiple of any block size in the sweep.
//! Blocking faults live in the leftover edge tile, so a test that only uses
//! 64 by 64 passes while the kernel is wrong.

mod common;

use common::{assert_all_close, randn};
use rustgpt::rng::Rng;
use rustgpt::tensor::Tensor;
use rustgpt::matmul::{matmul_blocked, matmul_naive};

/// (M, K, N) for A of shape [M,K] and B of shape [K,N].
const CASES: [(usize, usize, usize); 6] = [
    (1, 1, 1),       // the degenerate case
    (7, 13, 5),      // smaller than every block size
    (64, 64, 64),    // an exact multiple of 8, 16, 32 and 64
    (129, 257, 63),  // no dimension is a multiple of any block size
    (16, 1, 16),     // K = 1, so the inner loop runs once
    (33, 8, 1),      // N = 1, so the output is one column
];

const BLOCKS: [usize; 5] = [8, 16, 32, 64, 128];

// ---------------------------------------------------------------------------

/// The blocked kernel must agree with the Day 6 oracle at every size and at
/// every block size. This is the whole reason the slow version stays in the
/// repository.
#[test]
fn blocked_matches_naive() {
    let mut rng = Rng::seed(21);

    for &(m, k, n) in &CASES {
        let a: Tensor<f32> = randn(&mut rng, &[m, k]);
        let b: Tensor<f32> = randn(&mut rng, &[k, n]);
        let want = matmul_naive(&a, &b).expect("the oracle");

        for &block in &BLOCKS {
            let got = matmul_blocked(&a, &b, block).expect("blocked");
            assert_eq!(got.shape(), want.shape(), "{m}x{k}x{n} block {block}");
            assert_all_close(&got, &want, 1e-4, &format!("{m}x{k}x{n} block {block}"));
        }
    }

    // A strided input. The kernel must read through the index formula, and
    // it must not assume that a row of A sits in one run of memory.
    let a: Tensor<f32> = randn(&mut rng, &[13, 7]);
    let at = a.transpose(0, 1).unwrap(); // [7, 13], not contiguous
    let b: Tensor<f32> = randn(&mut rng, &[13, 5]);
    assert!(!at.is_contiguous());
    let want = matmul_naive(&at, &b).unwrap();
    for &block in &BLOCKS {
        assert_all_close(
            &matmul_blocked(&at, &b, block).unwrap(),
            &want,
            1e-4,
            &format!("strided A, block {block}"),
        );
    }

    // The same shape faults as Day 6, through the new entry point.
    let p: Tensor<f32> = Tensor::zeros(&[2, 3]);
    let q: Tensor<f32> = Tensor::zeros(&[4, 5]);
    assert!(matmul_blocked(&p, &q, 16).is_err());
}

/// The same sweep in `f64`, at a tolerance 6 orders of magnitude tighter.
///
/// This test separates a real bug from a rounding difference. A kernel that
/// drops one term fails both. A kernel that only reorders the additions
/// passes both. If `f32` fails and `f64` passes, widen nothing and look at
/// your tolerance instead.
#[test]
fn blocked_matches_naive_f64() {
    let mut rng = Rng::seed(22);

    for &(m, k, n) in &CASES {
        let a: Tensor<f64> = randn(&mut rng, &[m, k]);
        let b: Tensor<f64> = randn(&mut rng, &[k, n]);
        let want = matmul_naive(&a, &b).expect("the oracle");

        for &block in &BLOCKS {
            let got = matmul_blocked(&a, &b, block).expect("blocked");
            assert_all_close(&got, &want, 1e-10, &format!("f64 {m}x{k}x{n} block {block}"));
        }
    }

    // Two different block sizes must agree with each other as well as with
    // the oracle. This traps a kernel that is wrong in the same direction
    // as your reference, which happens when you copy a bug into both.
    let a: Tensor<f64> = randn(&mut rng, &[65, 129]);
    let b: Tensor<f64> = randn(&mut rng, &[129, 33]);
    let b16 = matmul_blocked(&a, &b, 16).unwrap();
    let b64 = matmul_blocked(&a, &b, 64).unwrap();
    assert_all_close(&b16, &b64, 1e-10, "block 16 against block 64");
}
