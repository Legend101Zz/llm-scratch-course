//! Day 2 acceptance tests — storage, shape and strides.
//!
//! The mentor writes this file. You write the code that makes it pass.
//! Do not edit the assertions. If a test is wrong, say so and argue it.
//!
//! Expected module layout. Add one line to `src/lib.rs`:
//!
//! ```text
//! pub mod tensor;   // pub struct Tensor<T>, pub fn contiguous_strides
//! ```
//!
//! Run with `cargo test --test day2`.
//!
//! The first run does not compile, because `src/tensor.rs` does not exist.
//! That failure is the red. The compiler tells you what to build.
//!
//! ## One note on `phys_index`
//!
//! `phys_index` is private, and this file is an integration test. An integration
//! test is a separate crate, so it sees the public API only. It cannot call
//! `phys_index` and it must not.
//!
//! `phys_index_matches_manual` reaches the function through `get`. The trick is
//! the buffer: element `i` holds the value `i`. So `get` returns the buffer
//! position that it read, and the test reads `phys_index` out loud.
//!
//! Keep `phys_index` private. A test that needs a private function is a test
//! that guessed the wrong seam.

use rustgpt::tensor::{Tensor, contiguous_strides};

// ---------------------------------------------------------------------------

/// The row-major rule: the stride of an axis is the product of the shape
/// entries after it, and the last axis always has stride 1.
#[test]
fn strides_row_major() {
    assert_eq!(contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
    assert_eq!(contiguous_strides(&[3, 4]), vec![4, 1]);
    assert_eq!(contiguous_strides(&[5]), vec![1]);

    // Rank 4. A rule written for rank 3 and stopped there fails here.
    assert_eq!(contiguous_strides(&[2, 3, 4, 5]), vec![60, 20, 5, 1]);

    // One stride per axis. Rank 0 has no axes, so it has no strides.
    assert_eq!(contiguous_strides(&[]), Vec::<usize>::new());

    // A zero entry is not a special case. The product rule carries the 0 left.
    assert_eq!(contiguous_strides(&[2, 0, 4]), vec![0, 4, 1]);

    // A fresh tensor is contiguous by construction. Day 3 builds the first
    // tensor that is not, so today only the true case is reachable.
    let t: Tensor<f64> = Tensor::zeros(&[2, 3, 4]);
    assert!(t.is_contiguous());
}

/// The index formula: `offset + sum(index[i] * stride[i])`.
///
/// The buffer holds `data[i] == i`, so `get` reports the buffer position.
#[test]
fn phys_index_matches_manual() {
    let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let t = Tensor::from_vec(data, &[2, 3, 4]);

    // The three worked examples from the lesson.
    assert_eq!(t.get(&[0, 0, 0]), 0.0);
    assert_eq!(t.get(&[1, 2, 3]), 23.0); // 1*12 + 2*4 + 3*1
    assert_eq!(t.get(&[1, 0, 2]), 14.0); // 1*12 + 0*4 + 2*1

    // Every element, not three of them. This traps two axes swapped, and it
    // traps a loop that walks the strides in the wrong direction.
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                let want = (i * 12 + j * 4 + k) as f64;
                assert_eq!(t.get(&[i, j, k]), want, "get([{i}, {j}, {k}])");
            }
        }
    }

    // Rank 1 and rank 2 use the same formula, with no special case.
    let v = Tensor::from_vec(vec![10.0f64, 11.0, 12.0], &[3]);
    assert_eq!(v.get(&[2]), 12.0);

    let m = Tensor::from_vec((0..6).map(|i| i as f32).collect(), &[2, 3]);
    assert_eq!(m.get(&[1, 2]), 5.0); // 1*3 + 2*1
    assert_eq!(m.get(&[0, 1]), 1.0);
}

/// A length that does not match the shape is a bug in the caller.
/// Bugs panic. Legitimate states return `Result`, and those start on Day 3.
#[test]
#[should_panic]
fn from_vec_rejects_bad_len() {
    let _ = Tensor::from_vec(vec![0.0f64; 5], &[2, 3]);
}

/// `numel` is the product of the shape. The empty product is 1, and a zero
/// entry gives 0. Both fall out of the rule with no special case.
#[test]
fn numel_is_shape_product() {
    let a: Tensor<f64> = Tensor::zeros(&[2, 3, 4]);
    assert_eq!(a.numel(), 24);
    assert_eq!(a.shape(), &[2, 3, 4]);

    let b: Tensor<f32> = Tensor::zeros(&[5]);
    assert_eq!(b.numel(), 5);
    assert_eq!(b.shape(), &[5]);

    // A zero entry means no elements at all.
    let c: Tensor<f64> = Tensor::zeros(&[2, 0, 4]);
    assert_eq!(c.numel(), 0);

    // `zeros` gives zeros. This looks trivial and it traps an uninitialized
    // buffer built with `Vec::with_capacity` and never filled.
    let d: Tensor<f64> = Tensor::zeros(&[2, 2]);
    assert_eq!(d.get(&[0, 0]), 0.0);
    assert_eq!(d.get(&[1, 1]), 0.0);

    // Rank 0 holds exactly one element, because the empty product is 1.
    // This is the answer the product rule gives with no `if`. If you decide
    // that rank 0 must be rejected instead, delete this block and tell me why.
    let s: Tensor<f64> = Tensor::zeros(&[]);
    assert_eq!(s.numel(), 1);
    assert!(s.shape().is_empty());
    assert_eq!(s.get(&[]), 0.0);
}
