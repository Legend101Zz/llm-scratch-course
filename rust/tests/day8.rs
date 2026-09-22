//! Day 8 acceptance tests — parallelism with scoped threads.
//!
//! The mentor writes this file. You write the code that makes it pass.
//! Do not edit the assertions. If a test is wrong, say so and argue it.
//!
//! Move this file to `rust/tests/day8.rs` at the start of Day 8, then run
//! `cargo test --test day8` and watch it fail.
//!
//! ## The conventions this file fixes
//!
//! `matmul_parallel` is rank 2 only, the same as `matmul_blocked`. It takes
//! the same `block` argument and partitions the OUTPUT ROWS across threads.
//!
//! The comparison against `matmul_blocked` is **exact**, not approximate. A
//! row partition leaves the summation order of every output element
//! unchanged, so the two kernels must produce identical bits. A small
//! difference means you partitioned along `K`. See `DAY_08.md` section 2.8.
//!
//! `threads` is a request, not a promise. The kernel is free to use fewer
//! when `M < threads`. Nothing here asserts how many threads actually ran,
//! because a timing or a thread-count assertion is not a correctness test.
//!
//! ## The helpers
//!
//! `mod common;` pulls in `rust/tests/common/mod.rs`, which Cargo does not
//! build as a test binary. It holds `randn`, `identity` and `assert_all_close`.

mod common;

use common::randn;
use rustgpt::matmul::{matmul_blocked, matmul_parallel};
use rustgpt::rng::Rng;
use rustgpt::tensor::Tensor;

// ---------------------------------------------------------------------------

/// Traps the `K` partition, and the short edge tile.
///
/// Every shape here has `M`, `K` and `N` different, so an index mix-up cannot
/// hide behind a square. 129 by 257 times 257 by 63 is a multiple of no block
/// size in the sweep, so the row bands and the tiles are both short at once.
#[test]
fn parallel_matches_blocked() {
    let mut rng = Rng::seed(80_808);

    // (M, K, N)
    let shapes = [(1, 1, 1), (7, 13, 5), (64, 64, 64), (129, 257, 63), (10, 3, 7)];

    for &(m, k, n) in &shapes {
        let a: Tensor<f32> = randn(&mut rng, &[m, k]);
        let b: Tensor<f32> = randn(&mut rng, &[k, n]);

        for &block in &[8usize, 16, 32] {
            let want = matmul_blocked(&a, &b, block).expect("blocked reference");

            for &threads in &[1usize, 2, 4, 8] {
                let got = matmul_parallel(&a, &b, block, threads)
                    .unwrap_or_else(|e| panic!("parallel {m}x{k}x{n} block {block} threads {threads}: {e}"));

                assert_eq!(
                    got.shape(),
                    want.shape(),
                    "shape at {m}x{k}x{n}, block {block}, threads {threads}"
                );
                // EXACT. Not assert_all_close. Read the header if this fails.
                assert_eq!(
                    got.to_vec(),
                    want.to_vec(),
                    "not bit-identical at {m}x{k}x{n}, block {block}, threads {threads}"
                );
            }
        }
    }

    // A strided input. `a` here is a view with swapped strides and no copy,
    // so the kernel must read through the index formula, not off a raw slice.
    let base: Tensor<f32> = randn(&mut rng, &[13, 7]);
    let a_t = base.transpose(0, 1).expect("transpose");
    let b: Tensor<f32> = randn(&mut rng, &[13, 5]);
    let want = matmul_blocked(&a_t, &b, 16).expect("blocked on a strided input");
    let got = matmul_parallel(&a_t, &b, 16, 4).expect("parallel on a strided input");
    assert_eq!(got.to_vec(), want.to_vec(), "strided input, 4 threads");

    // A shape error must still be an error, and it must not spawn anything.
    let bad: Tensor<f32> = randn(&mut rng, &[3, 3]);
    let good: Tensor<f32> = randn(&mut rng, &[4, 4]);
    assert!(
        matmul_parallel(&bad, &good, 8, 4).is_err(),
        "3x3 times 4x4 must be a shape error"
    );
}

// ---------------------------------------------------------------------------

/// Traps the row-count arithmetic.
///
/// `M = 10` is smaller than two of the thread counts on purpose. A kernel
/// that computes `M / threads` instead of `M.div_ceil(threads)` drops the
/// remainder rows and leaves them at zero, which shows here and nowhere else.
#[test]
fn parallel_thread_count_invariant() {
    let mut rng = Rng::seed(1_008);

    let a: Tensor<f64> = randn(&mut rng, &[10, 6]);
    let b: Tensor<f64> = randn(&mut rng, &[6, 9]);

    let reference = matmul_parallel(&a, &b, 8, 1).expect("one thread");

    for &threads in &[1usize, 2, 4, 8, 10, 16] {
        let got = matmul_parallel(&a, &b, 8, threads)
            .unwrap_or_else(|e| panic!("threads {threads}: {e}"));

        assert_eq!(got.shape(), &[10, 9], "shape at threads {threads}");
        assert_eq!(
            got.to_vec(),
            reference.to_vec(),
            "output changed at threads {threads}"
        );

        // No output element may be left untouched. A dropped row band is
        // exactly zero, and a real product of normal draws never is.
        assert!(
            got.to_vec().iter().all(|v| *v != 0.0),
            "an output element is exactly 0.0 at threads {threads}: a row band was dropped"
        );
    }

    // A tall matrix with a prime row count, so no thread count divides it.
    let a2: Tensor<f64> = randn(&mut rng, &[97, 5]);
    let b2: Tensor<f64> = randn(&mut rng, &[5, 11]);
    let r2 = matmul_parallel(&a2, &b2, 16, 1).expect("one thread, 97 rows");
    for &threads in &[2usize, 3, 4, 7, 8] {
        let got = matmul_parallel(&a2, &b2, 16, threads).expect("97 rows");
        assert_eq!(got.to_vec(), r2.to_vec(), "97 rows at threads {threads}");
    }
}
