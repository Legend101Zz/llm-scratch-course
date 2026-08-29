//! Day 5 acceptance tests — reductions.
//!
//! The mentor writes this file. You write the code that makes it pass.
//! Do not edit the assertions. If a test is wrong, say so and argue it.
//!
//! Move this file to `rust/tests/day5.rs` at the start of Day 5, then run
//! `cargo test --test day5` and watch it fail.
//!
//! ## A warning about run time
//!
//! `sum_all_precision` builds 100 million `f32` values. That is a 400 MB
//! buffer, and this one test dominates `cargo test` in a debug build.
//! While you iterate, run this:
//!
//! ```text
//! cargo test --release --test day5
//! ```
//!
//! If the debug run takes more than a minute, that is information and not a
//! fault. It says your traversal costs a lot per element. Write the number
//! down. Do not optimise today. Day 7 is where speed becomes the subject.

use rustgpt::tensor::Tensor;

// ---------------------------------------------------------------------------

/// `keepdim` decides whether the reduced axis disappears or stays as size 1.
/// It never changes a value.
#[test]
fn sum_axis_shapes() {
    let t: Tensor<f64> = Tensor::zeros(&[2, 3, 4]);

    assert_eq!(t.sum_axis(1, false).unwrap().shape(), &[2, 4]);
    assert_eq!(t.sum_axis(1, true).unwrap().shape(), &[2, 1, 4]);
    assert_eq!(t.sum_axis(0, false).unwrap().shape(), &[3, 4]);
    assert_eq!(t.sum_axis(0, true).unwrap().shape(), &[1, 3, 4]);
    assert_eq!(t.sum_axis(2, false).unwrap().shape(), &[2, 3]);
    assert_eq!(t.sum_axis(2, true).unwrap().shape(), &[2, 3, 1]);

    // mean and max follow the same shape rule. One rule, three reductions.
    assert_eq!(t.mean_axis(0, true).unwrap().shape(), &[1, 3, 4]);
    assert_eq!(t.mean_axis(0, false).unwrap().shape(), &[3, 4]);
    assert_eq!(t.max_axis(2, false).unwrap().shape(), &[2, 3]);
    assert_eq!(t.max_axis(2, true).unwrap().shape(), &[2, 3, 1]);

    // A result is a fresh tensor, so it is contiguous.
    assert!(t.sum_axis(1, true).unwrap().is_contiguous());

    // An axis past the rank is an error, not a panic.
    assert!(t.sum_axis(3, false).is_err());
    assert!(t.mean_axis(9, false).is_err());
    assert!(t.max_axis(3, true).is_err());

    // Rank 1 reduced with keepdim=false gives rank 0. The axis disappeared,
    // and there were no other axes. The empty shape is the honest answer.
    let v: Tensor<f64> = Tensor::zeros(&[5]);
    assert!(v.sum_axis(0, false).unwrap().shape().is_empty());
    assert_eq!(v.sum_axis(0, false).unwrap().numel(), 1);
    assert_eq!(v.sum_axis(0, true).unwrap().shape(), &[1]);
}

/// Values computed by hand, on both axes, at rank 2 and at rank 3.
#[test]
fn sum_axis_values() {
    // [[1, 2, 3],
    //  [4, 5, 6]]
    let t = Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    // Axis 0 disappears, so the sum runs down each column.
    assert_eq!(t.sum_axis(0, false).unwrap().to_vec(), vec![5.0, 7.0, 9.0]);
    // Axis 1 disappears, so the sum runs along each row.
    assert_eq!(t.sum_axis(1, false).unwrap().to_vec(), vec![6.0, 15.0]);
    // keepdim changes the shape and never the values.
    assert_eq!(t.sum_axis(1, true).unwrap().to_vec(), vec![6.0, 15.0]);
    assert_eq!(t.sum_axis(1, true).unwrap().shape(), &[2, 1]);

    assert_eq!(t.mean_axis(0, false).unwrap().to_vec(), vec![2.5, 3.5, 4.5]);
    assert_eq!(t.mean_axis(1, false).unwrap().to_vec(), vec![2.0, 5.0]);
    assert_eq!(t.max_axis(0, false).unwrap().to_vec(), vec![4.0, 5.0, 6.0]);
    assert_eq!(t.max_axis(1, false).unwrap().to_vec(), vec![3.0, 6.0]);

    // `sum_all` folds every element, whatever the shape.
    assert_eq!(t.sum_all(), 21.0);

    // Rank 3, where element [i,j,k] holds 12i + 4j + k.
    // The last axis is where an index walk usually breaks.
    let u = Tensor::from_vec((0..24).map(|i| i as f64).collect(), &[2, 3, 4]);

    let s2 = u.sum_axis(2, false).unwrap();
    assert_eq!(s2.shape(), &[2, 3]);
    assert_eq!(s2.to_vec(), vec![6.0, 22.0, 38.0, 54.0, 70.0, 86.0]);

    let s1 = u.sum_axis(1, false).unwrap();
    assert_eq!(s1.shape(), &[2, 4]);
    assert_eq!(s1.to_vec(), vec![12.0, 15.0, 18.0, 21.0, 48.0, 51.0, 54.0, 57.0]);

    let s0 = u.sum_axis(0, false).unwrap();
    assert_eq!(s0.shape(), &[3, 4]);
    assert_eq!(
        s0.to_vec(),
        (0..12).map(|i| (12 + 2 * i) as f64).collect::<Vec<f64>>()
    );

    assert_eq!(u.sum_all(), (0..24).map(|i| i as f64).sum::<f64>());
    assert_eq!(u.mean_axis(2, false).unwrap().to_vec()[0], 1.5);

    // A reduction must read through strides, so a view works.
    let tt = t.transpose(0, 1).unwrap();
    assert_eq!(tt.shape(), &[3, 2]);
    assert_eq!(tt.sum_axis(0, false).unwrap().to_vec(), vec![6.0, 15.0]);
    assert_eq!(tt.sum_axis(1, false).unwrap().to_vec(), vec![5.0, 7.0, 9.0]);
    assert_eq!(tt.sum_all(), 21.0);

    // A broadcast view too. Its numel is larger than its buffer.
    let row = Tensor::from_vec(vec![1.0f64, 2.0, 3.0], &[1, 3]);
    let wide = row.broadcast_to(&[4, 3]).unwrap();
    assert_eq!(wide.sum_axis(0, false).unwrap().to_vec(), vec![4.0, 8.0, 12.0]);
    assert_eq!(wide.sum_all(), 24.0);
}

/// **This test is the lesson of the day.**
///
/// A left fold over 100 million `f32` ones does not give 100 million.
/// `f32` carries a 24-bit significand. Once the accumulator passes 2^24,
/// the value 1.0 is smaller than half of one step, so `acc + 1.0` rounds
/// straight back to `acc`. The fold stops climbing and stays there for the
/// remaining 83 million additions.
///
/// Pairwise summation adds values of similar size to each other. No partial
/// sum ever gets far ahead of its partner, so no addition is ever thrown away.
#[test]
fn sum_all_precision() {
    const N: usize = 100_000_000;

    // The reference: the loop you would write first, and it is wrong.
    let mut sequential = 0.0f32;
    for _ in 0..N {
        sequential += 1.0;
    }
    assert_eq!(
        sequential, 16_777_216.0,
        "a left fold in f32 must stop dead at 2^24"
    );

    let t = Tensor::from_vec(vec![1.0f32; N], &[N]);
    let pairwise: f32 = t.sum_all();

    // 1e8 sits between 2^26 and 2^27, so one step of f32 there is 8.0.
    assert!(
        (pairwise - 1.0e8f32).abs() <= 8.0,
        "pairwise sum was {pairwise}, want 1e8 inside one step (8.0). \
         The left fold gives {sequential}, which is {} short.",
        1.0e8f32 - sequential
    );

    // The same fold in f64 is exact, because 53 bits reach far past 1e8.
    // The fault is the width of the accumulator, not the idea of adding.
    let d = Tensor::from_vec(vec![1.0f64; 1_000_000], &[1_000_000]);
    assert_eq!(d.sum_all(), 1_000_000.0f64);

    // A small sum must stay exact. A pairwise split that drops an odd
    // element passes the big test above and fails this one.
    let odd = Tensor::from_vec((1..=7).map(|i| i as f64).collect(), &[7]);
    assert_eq!(odd.sum_all(), 28.0);
    let one = Tensor::from_vec(vec![42.0f64], &[1]);
    assert_eq!(one.sum_all(), 42.0);
    let none: Tensor<f64> = Tensor::zeros(&[0]);
    assert_eq!(none.sum_all(), 0.0, "the sum of nothing is 0");
}

/// A max seeded with zero returns 0 on all-negative input, and that is wrong.
/// Attention scores after a causal mask are large and negative, so this bug
/// hides until Day 20 and then destroys every row.
#[test]
fn max_axis_handles_negatives() {
    // [[-5, -1, -9],
    //  [-2, -7, -3]]
    let t = Tensor::from_vec(vec![-5.0f64, -1.0, -9.0, -2.0, -7.0, -3.0], &[2, 3]);

    assert_eq!(t.max_axis(1, false).unwrap().to_vec(), vec![-1.0, -2.0]);
    assert_eq!(t.max_axis(0, false).unwrap().to_vec(), vec![-2.0, -1.0, -3.0]);
    assert_eq!(t.max_axis(1, true).unwrap().shape(), &[2, 1]);

    // One element along the axis. The max is that element.
    let one = Tensor::from_vec(vec![-3.0f64], &[1, 1]);
    assert_eq!(one.max_axis(0, false).unwrap().to_vec(), vec![-3.0]);
    assert_eq!(one.max_axis(1, false).unwrap().to_vec(), vec![-3.0]);

    // All values equal. The max is that value, not the first index.
    let flat = Tensor::from_vec(vec![-4.0f64; 6], &[2, 3]);
    assert_eq!(flat.max_axis(1, false).unwrap().to_vec(), vec![-4.0, -4.0]);

    // The reason `max_axis` exists, and the reason it takes `keepdim`.
    // Softmax subtracts the row max before it calls exp. Without the shift,
    // exp overflows to infinity and every probability becomes NaN.
    let logits = Tensor::from_vec(vec![1000.0f64, 1001.0, 1002.0], &[1, 3]);
    let m = logits.max_axis(1, true).unwrap();
    assert_eq!(m.shape(), &[1, 1], "keepdim exists so the max broadcasts back");
    assert_eq!(m.to_vec(), vec![1002.0]);

    let shifted = logits.sub(&m).expect("[1,3] - [1,1] must broadcast");
    assert_eq!(shifted.to_vec(), vec![-2.0, -1.0, 0.0]);

    // Without the shift.
    for v in logits.exp().to_vec() {
        assert!(v.is_infinite(), "exp(1000) overflows f64, and that is the point");
    }
    // With the shift.
    for v in shifted.exp().to_vec() {
        assert!(v.is_finite() && v > 0.0);
    }
    // And the answer is unchanged, because softmax(x) equals softmax(x - c).
    let denom: f64 = shifted.exp().to_vec().iter().sum();
    assert!((shifted.exp().get(&[0, 2]) / denom - 0.665_240_955_774_821_8).abs() < 1e-9);
}
