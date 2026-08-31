//! Day 3 acceptance tests — zero-copy views.
//!
//! The mentor writes this file. You write the code that makes it pass.
//! Do not edit the assertions. If a test is wrong, say so and argue it.
//!
//! Move this file to `rust/tests/day3.rs` at the start of Day 3, then run
//! `cargo test --test day3` and watch it fail.
//!
//! ## Two API additions that the day card does not list
//!
//! The card promises that `transpose` copies nothing. A promise that no test
//! can observe is not a promise. So the tensor needs two more readers:
//!
//! ```text
//! pub fn strides(&self) -> &[usize];
//! pub fn shares_storage_with(&self, other: &Self) -> bool;
//! ```
//!
//! `strides` mirrors `shape` from Day 2. `shares_storage_with` answers "same
//! allocation?" and its whole body is the `Rc::ptr_eq` call from Day 2
//! section 3.5. PyTorch ships both: `Tensor.stride()` and `Tensor.data_ptr()`.
//!
//! ## The `permute` convention, stated once
//!
//! `order[i]` names the **source** axis that becomes the new axis `i`.
//!
//! ```text
//! new_shape[i]   = old_shape[order[i]]
//! new_strides[i] = old_strides[order[i]]
//! ```
//!
//! This is the NumPy and PyTorch convention. The other reading ("axis `i` moves
//! to position `order[i]`") gives the inverse permutation, and the roundtrip
//! test below fails on it. Pick this one.

use rustgpt::tensor::{ShapeError, Tensor};

/// Assert that a call failed with one named `ShapeError` variant.
///
/// This is a macro and not a function, because a variant is a *pattern* and
/// a function cannot take a pattern as an argument. `macro_rules!` matches on
/// syntax, so `$variant:pat` accepts `ShapeError::NotContiguous` or
/// `ShapeError::BadRank { .. }` and drops it straight into a `match` arm.
///
/// `Result::expect_err` would be shorter and it needs `Debug` on the success
/// type. `Tensor` has none, and a derived one would print the whole buffer.
macro_rules! assert_err {
    ($call:expr, $variant:pat, $what:expr) => {
        match $call {
            Err($variant) => {}
            Err(other) => panic!("{}: wrong variant, got {other:?}", $what),
            Ok(_) => panic!("{}: the call succeeded and must not", $what),
        }
    };
}

// ---------------------------------------------------------------------------

/// A permutation and its inverse give back the original view, in shape,
/// in strides and in logical order.
#[test]
fn permute_roundtrip() {
    let t = Tensor::from_vec((0..24).map(|i| i as f64).collect(), &[2, 3, 4]);
    assert_eq!(t.strides(), &[12, 4, 1]);

    let p = t.permute(&[2, 0, 1]).expect("permute [2,0,1]");
    assert_eq!(p.shape(), &[4, 2, 3]);
    assert_eq!(p.strides(), &[1, 12, 4]);
    assert!(p.shares_storage_with(&t), "permute must copy nothing");

    let back = p.permute(&[1, 2, 0]).expect("permute back");
    assert_eq!(back.shape(), t.shape());
    assert_eq!(back.strides(), t.strides());
    assert_eq!(back.to_vec(), t.to_vec());

    // The elements agree under the index map, not only in bulk.
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                assert_eq!(p.get(&[k, i, j]), t.get(&[i, j, k]), "[{i},{j},{k}]");
            }
        }
    }

    // The identity permutation changes nothing.
    let same = t.permute(&[0, 1, 2]).expect("identity permute");
    assert_eq!(same.strides(), t.strides());

    // A wrong length is a rank error.
    assert_err!(
        t.permute(&[0, 1]),
        ShapeError::BadRank { .. },
        "a rank 2 order on a rank 3 tensor"
    );

    // A repeated axis and an out-of-range axis are both errors. Which variant
    // you pick is yours. Both must be `Err`, and neither may panic.
    assert!(t.permute(&[0, 0, 1]).is_err(), "axis 0 used twice");
    assert!(t.permute(&[0, 1, 3]).is_err(), "axis 3 on a rank 3 tensor");
}

/// A transpose swaps two entries of `shape` and two entries of `strides`.
/// It touches no data. This test proves the "no data" half.
#[test]
fn transpose_is_zero_copy() {
    // [[0, 1, 2],
    //  [3, 4, 5]]
    let t = Tensor::from_vec((0..6).map(|i| i as f64).collect(), &[2, 3]);
    let tt = t.transpose(0, 1).expect("transpose(0, 1)");

    assert_eq!(tt.shape(), &[3, 2]);
    assert_eq!(tt.strides(), &[1, 3]);
    assert!(tt.shares_storage_with(&t), "transpose copied the buffer");

    // The last axis now has stride 3, so the view is not contiguous.
    // This is the first tensor in the project for which that is true.
    assert!(t.is_contiguous());
    assert!(!tt.is_contiguous());

    // The same element, read through two different index paths.
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(tt.get(&[j, i]), t.get(&[i, j]), "[{i},{j}]");
        }
    }

    // `to_vec` walks the strides, so it reports logical order and not buffer
    // order. The buffer is unchanged. Only the reading order changed.
    assert_eq!(t.to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(tt.to_vec(), vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);

    // Transpose is its own inverse.
    let back = tt.transpose(0, 1).expect("transpose back");
    assert_eq!(back.shape(), t.shape());
    assert_eq!(back.strides(), t.strides());

    // The same axis twice is a no-op, not an error.
    let noop = t.transpose(1, 1).expect("transpose(1, 1)");
    assert_eq!(noop.strides(), t.strides());

    // An axis past the rank is an error, not a panic.
    assert!(t.transpose(0, 2).is_err());

    // Rank 3, to prove that transpose is not a rank 2 special case.
    let u = Tensor::from_vec((0..24).map(|i| i as f64).collect(), &[2, 3, 4]);
    let ut = u.transpose(1, 2).expect("transpose(1, 2)");
    assert_eq!(ut.shape(), &[2, 4, 3]);
    assert_eq!(ut.strides(), &[12, 1, 4]);
    assert_eq!(ut.get(&[1, 3, 2]), u.get(&[1, 2, 3]));
}

/// `reshape` on a non-contiguous view has no correct stride expression.
/// It must refuse. A silent copy inside an attention loop is the fault
/// this test exists to prevent.
#[test]
fn reshape_noncontiguous_errors() {
    let t = Tensor::from_vec((0..6).map(|i| i as f64).collect(), &[2, 3]);

    // On a contiguous tensor, reshape is a view and costs nothing.
    let r = t.reshape(&[3, 2]).expect("reshape a contiguous tensor");
    assert_eq!(r.shape(), &[3, 2]);
    assert_eq!(r.strides(), &[2, 1]);
    assert!(r.shares_storage_with(&t), "a contiguous reshape must not copy");
    assert_eq!(r.to_vec(), t.to_vec(), "reshape keeps the reading order");

    let flat = t.reshape(&[6]).expect("reshape to rank 1");
    assert_eq!(flat.strides(), &[1]);

    // The same call on a transposed view must fail.
    let tt = t.transpose(0, 1).expect("transpose");
    assert_err!(
        tt.reshape(&[6]),
        ShapeError::NotContiguous,
        "reshape of a strided view"
    );

    // A reshape that changes the element count is a different fault, and it
    // gets a different variant. 6 elements do not fit into [2,4].
    assert_err!(
        t.reshape(&[2, 4]),
        ShapeError::SizeMismatch,
        "6 elements into [2,4]"
    );
}

/// `contiguous` is the only method in the library that copies. After it,
/// the reshape that failed above succeeds, and the values are unchanged.
#[test]
fn contiguous_then_reshape_ok() {
    let t = Tensor::from_vec((0..6).map(|i| i as f64).collect(), &[2, 3]);
    let tt = t.transpose(0, 1).expect("transpose");

    let c = tt.contiguous();
    assert!(c.is_contiguous());
    assert_eq!(c.shape(), tt.shape());
    assert_eq!(c.strides(), &[2, 1]);
    assert_eq!(c.to_vec(), tt.to_vec(), "contiguous must keep logical order");
    assert!(
        !c.shares_storage_with(&tt),
        "contiguous on a strided view must build a new buffer"
    );

    let flat = c.reshape(&[6]).expect("reshape after contiguous");
    assert_eq!(flat.shape(), &[6]);
    assert_eq!(flat.to_vec(), vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);

    // `contiguous` on a tensor that is already contiguous must give the same
    // values. Whether it also copies is your call. Write the answer in the
    // commit message, because Day 20 pays for a copy you did not intend.
    let again = t.contiguous();
    assert_eq!(again.to_vec(), t.to_vec());
    assert!(again.is_contiguous());
}

/// A slice narrows one axis. It changes `shape` and `offset`, and it must
/// never change `strides`.
#[test]
fn slice_bounds() {
    let t = Tensor::from_vec((0..24).map(|i| i as f64).collect(), &[2, 3, 4]);

    let s = t.slice(1, 1, 3).expect("slice axis 1, 1..3");
    assert_eq!(s.shape(), &[2, 2, 4]);
    assert_eq!(s.strides(), t.strides(), "a slice never changes the strides");
    assert!(s.shares_storage_with(&t), "a slice must copy nothing");
    assert_eq!(s.get(&[0, 0, 0]), t.get(&[0, 1, 0]));
    assert_eq!(s.get(&[1, 1, 3]), t.get(&[1, 2, 3]));

    // A slice on axis 0 is the one that moves the offset by a whole plane.
    let plane = t.slice(0, 1, 2).expect("slice axis 0, 1..2");
    assert_eq!(plane.shape(), &[1, 3, 4]);
    assert_eq!(plane.get(&[0, 0, 0]), 12.0);
    assert_eq!(plane.to_vec(), (12..24).map(|i| i as f64).collect::<Vec<f64>>());

    // A slice of a slice composes. The offsets add.
    let inner = plane.slice(2, 2, 4).expect("slice a slice");
    assert_eq!(inner.shape(), &[1, 3, 2]);
    assert_eq!(inner.get(&[0, 0, 0]), 14.0);
    assert_eq!(inner.get(&[0, 2, 1]), 23.0);

    // start after end.
    assert!(t.slice(0, 2, 1).is_err());
    // end past the axis length. Axis 1 has length 3.
    assert!(t.slice(1, 0, 4).is_err());
    // axis past the rank.
    assert!(t.slice(3, 0, 1).is_err());

    // `start == end` gives an empty view. NumPy allows this and so do we.
    // If you decide that an empty slice is an error, delete this block and
    // tell me which caller you protected.
    let empty = t.slice(0, 1, 1).expect("start == end is legal");
    assert_eq!(empty.shape(), &[0, 3, 4]);
    assert_eq!(empty.numel(), 0);
    assert_eq!(empty.to_vec(), Vec::<f64>::new());
}
