//! Day 4 acceptance tests — broadcast and elementwise ops.
//!
//! The mentor writes this file. You write the code that makes it pass.
//! Do not edit the assertions. If a test is wrong, say so and argue it.
//!
//! Move this file to `rust/tests/day4.rs` at the start of Day 4, then run
//! `cargo test --test day4` and watch it fail.
//!
//! ## The signatures this file calls
//!
//! The day card names the ten elementwise ops and gives no signatures.
//! Here they are, fixed:
//!
//! ```text
//! pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>, ShapeError>;
//!
//! // two operands, so a shape can fail  -> Result
//! pub fn add(&self, other: &Self) -> Result<Self, ShapeError>;
//! pub fn sub(&self, other: &Self) -> Result<Self, ShapeError>;
//! pub fn mul(&self, other: &Self) -> Result<Self, ShapeError>;
//! pub fn div(&self, other: &Self) -> Result<Self, ShapeError>;
//!
//! // one operand, so no shape can fail  -> Self
//! pub fn neg(&self)  -> Self;
//! pub fn exp(&self)  -> Self;
//! pub fn ln(&self)   -> Self;
//! pub fn sqrt(&self) -> Self;
//! pub fn tanh(&self) -> Self;
//! pub fn relu(&self) -> Self;
//! ```
//!
//! Read the split. The return type states which calls can fail. A one-operand
//! op has one shape, so there is nothing to disagree about. A two-operand op
//! can be handed two shapes that do not broadcast, and the caller must handle
//! that. The type system carries the fact, so no comment has to.
//!
//! `zip_with` broadcasts both operands to the common shape before it applies
//! the closure. `add` is one call to `zip_with`. So are the other three.
//!
//! ## What the output of `map` and `zip_with` looks like
//!
//! Both return a fresh, contiguous, offset-0 tensor. A stride-0 input axis
//! produces a real axis in the output, with real repeated values in a real
//! buffer. The stride-0 trick lives on the read side only.

use rustgpt::tensor::{Tensor, broadcast_shapes};

// ---------------------------------------------------------------------------

/// The compatibility rule: align the shapes from the trailing axis, then each
/// pair must be equal, or one of the two must be 1. A missing axis counts as 1.
#[test]
fn broadcast_shapes_table() {
    // Both sides stretch. This is the case that surprises people.
    assert_eq!(broadcast_shapes(&[3, 1], &[1, 4]).unwrap(), vec![3, 4]);

    // A missing leading axis counts as 1.
    assert_eq!(broadcast_shapes(&[5], &[3, 5]).unwrap(), vec![3, 5]);
    assert_eq!(broadcast_shapes(&[3, 5], &[5]).unwrap(), vec![3, 5]);

    // Equal shapes broadcast to themselves.
    assert_eq!(broadcast_shapes(&[2, 3], &[2, 3]).unwrap(), vec![2, 3]);

    // Rank 0 is a scalar. It broadcasts against anything.
    assert_eq!(broadcast_shapes(&[], &[2, 3]).unwrap(), vec![2, 3]);
    assert_eq!(broadcast_shapes(&[], &[]).unwrap(), Vec::<usize>::new());

    // Different ranks, with a stretch on each side at different axes.
    assert_eq!(broadcast_shapes(&[4, 1, 5], &[3, 1]).unwrap(), vec![4, 3, 5]);
    assert_eq!(broadcast_shapes(&[1, 3, 1], &[2, 1, 4]).unwrap(), vec![2, 3, 4]);

    // The shapes in your self-check are deliberately not in this table.

    // 3 against 2 is not equal, and neither one is 1.
    assert!(broadcast_shapes(&[2, 3], &[3, 2]).is_err());
    assert!(broadcast_shapes(&[2, 3], &[4]).is_err());
    assert!(broadcast_shapes(&[5, 2], &[3, 2, 4]).is_err());
}

/// Broadcast duplicates no data. It sets the stride of the stretched axis to 0,
/// so every index along that axis reads the same element.
#[test]
fn broadcast_uses_stride_zero() {
    let row = Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0], &[1, 4]);
    let big = row.broadcast_to(&[3, 4]).expect("broadcast [1,4] to [3,4]");

    assert_eq!(big.shape(), &[3, 4]);
    assert_eq!(big.strides(), &[0, 1], "the stretched axis must have stride 0");
    assert!(big.shares_storage_with(&row), "broadcast must copy nothing");
    assert_eq!(big.numel(), 12);

    // Three logical rows over four physical elements.
    for i in 0..3 {
        for j in 0..4 {
            assert_eq!(big.get(&[i, j]), row.get(&[0, j]), "[{i},{j}]");
        }
    }

    // A missing leading axis is added first, and it also gets stride 0.
    let v = Tensor::from_vec(vec![10.0f64, 20.0], &[2]);
    let vb = v.broadcast_to(&[3, 2]).expect("broadcast [2] to [3,2]");
    assert_eq!(vb.shape(), &[3, 2]);
    assert_eq!(vb.strides(), &[0, 1]);
    assert_eq!(vb.get(&[2, 1]), 20.0);

    // A middle axis stretches the same way.
    let m = Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3]);
    let mb = m.broadcast_to(&[2, 4, 3]).expect("broadcast [2,1,3] to [2,4,3]");
    assert_eq!(mb.strides(), &[3, 0, 1]);
    assert_eq!(mb.get(&[1, 3, 2]), 6.0);
    assert_eq!(mb.get(&[1, 0, 2]), 6.0);

    // A broadcast view has more elements than its buffer holds.
    assert_eq!(mb.numel(), 24);
    assert_eq!(m.numel(), 6);

    // 2 does not stretch to 3.
    assert!(v.broadcast_to(&[3, 3]).is_err());
    // Broadcast never shrinks an axis.
    assert!(big.broadcast_to(&[3, 2]).is_err());
    // Broadcast never lowers the rank.
    assert!(big.broadcast_to(&[4]).is_err());
}

/// A broadcast view holds real elements when you read it. Its sum over the
/// stretched axis is the original sum times the stretch factor.
#[test]
fn broadcast_sum_scales() {
    // `sum_all` arrives on Day 5. This test folds its own sum, so Day 4 does
    // not wait for Day 5 and does not test Day 5's precision work by accident.
    fn total(t: &Tensor<f64>) -> f64 {
        t.to_vec().iter().sum()
    }

    let a = Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0], &[1, 4]);
    let b = a.broadcast_to(&[3, 4]).expect("broadcast");

    assert_eq!(b.to_vec().len(), 12, "to_vec must report numel, not buffer len");

    let want = total(&a) * 3.0;
    let got = total(&b);
    assert!((got - want).abs() < 1e-12, "sum was {got}, want {want}");

    // The exact reading order, not only the total. A stride walk that reads
    // the buffer instead of the strides gives the right length and the
    // wrong values.
    assert_eq!(
        b.to_vec(),
        vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
    );

    // Stretch the other axis, so the repeat pattern is different.
    let col = Tensor::from_vec(vec![7.0f64, 8.0], &[2, 1]);
    let cb = col.broadcast_to(&[2, 3]).expect("broadcast [2,1] to [2,3]");
    assert_eq!(cb.to_vec(), vec![7.0, 7.0, 7.0, 8.0, 8.0, 8.0]);
    assert!((total(&cb) - total(&col) * 3.0).abs() < 1e-12);
}

/// The ten ops, against values computed by hand.
#[test]
fn elementwise_against_manual() {
    // [[1, 2, 3],
    //  [4, 5, 6]]
    let a = Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::from_vec(vec![10.0f64, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3]);

    let s = a.add(&b).expect("add");
    assert_eq!(s.shape(), &[2, 3]);
    assert!(s.is_contiguous(), "a fresh result is contiguous");
    assert_eq!(s.to_vec(), vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]);

    assert_eq!(b.sub(&a).unwrap().to_vec(), vec![9.0, 18.0, 27.0, 36.0, 45.0, 54.0]);
    assert_eq!(a.mul(&b).unwrap().to_vec(), vec![10.0, 40.0, 90.0, 160.0, 250.0, 360.0]);
    assert_eq!(b.div(&a).unwrap().to_vec(), vec![10.0; 6]);
    assert_eq!(a.neg().to_vec(), vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]);

    // `map` takes any closure over one element.
    assert_eq!(a.map(|x| x * x).to_vec(), vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0]);

    // `zip_with` is the one place the elementwise rule lives.
    // Every two-operand op is a call to it.
    let z = a.zip_with(&b, |x, y| y - x).expect("zip_with");
    assert_eq!(z.to_vec(), b.sub(&a).unwrap().to_vec());

    // relu. This traps a max written with the arguments the wrong way round,
    // and it traps a relu that returns 0 for a positive input.
    let n = Tensor::from_vec(vec![-2.0f64, -0.5, 0.0, 0.5, 2.0], &[5]);
    assert_eq!(n.relu().to_vec(), vec![0.0, 0.0, 0.0, 0.5, 2.0]);

    // exp, ln, sqrt and tanh, against values you can state without a machine.
    assert!((n.exp().get(&[4]) - std::f64::consts::E.powi(2)).abs() < 1e-12);
    assert!(n.exp().get(&[2]) == 1.0, "exp(0) is exactly 1");

    let sq = Tensor::from_vec(vec![1.0f64, 4.0, 9.0, 16.0], &[4]);
    assert_eq!(sq.sqrt().to_vec(), vec![1.0, 2.0, 3.0, 4.0]);

    let pos = Tensor::from_vec(vec![1.0f64, std::f64::consts::E], &[2]);
    assert!(pos.ln().get(&[0]).abs() < 1e-12, "ln(1) is 0");
    assert!((pos.ln().get(&[1]) - 1.0).abs() < 1e-12, "ln(e) is 1");
    // exp and ln undo each other inside float tolerance.
    assert!((sq.ln().exp().get(&[3]) - 16.0).abs() < 1e-12);

    assert!(n.tanh().get(&[2]).abs() < 1e-12, "tanh(0) is 0");
    assert!(n.tanh().get(&[4]) < 1.0, "tanh is bounded above by 1");
    assert!(n.tanh().get(&[0]) > -1.0, "tanh is bounded below by -1");
    // tanh is an odd function: tanh(-x) == -tanh(x).
    assert!((n.tanh().get(&[0]) + n.tanh().get(&[4])).abs() < 1e-12);

    // THE case this whole day exists for. A bias add with no copy of the bias.
    // GPT-2 does this after every linear layer, 48 times per forward pass.
    let x = Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let bias = Tensor::from_vec(vec![100.0f64, 200.0, 300.0], &[3]);
    let y = x.add(&bias).expect("[2,3] + [3] must broadcast");
    assert_eq!(y.shape(), &[2, 3]);
    assert_eq!(y.to_vec(), vec![101.0, 202.0, 303.0, 104.0, 205.0, 306.0]);

    // A column stretches the other way.
    let col = Tensor::from_vec(vec![10.0f64, 20.0], &[2, 1]);
    assert_eq!(x.add(&col).unwrap().to_vec(), vec![11.0, 12.0, 13.0, 24.0, 25.0, 26.0]);

    // Both sides stretch at once. [2,1] with [1,3] gives [2,3].
    let r = Tensor::from_vec(vec![1.0f64, 2.0, 3.0], &[1, 3]);
    let out = col.add(&r).expect("[2,1] + [1,3]");
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(out.to_vec(), vec![11.0, 12.0, 13.0, 21.0, 22.0, 23.0]);

    // Incompatible shapes are an error, not a panic.
    let bad = Tensor::from_vec(vec![0.0f64; 2], &[2]);
    assert!(x.add(&bad).is_err());
    assert!(x.zip_with(&bad, |p, q| p + q).is_err());

    // An elementwise op must read through strides, so a view works.
    let xt = x.transpose(0, 1).expect("transpose");
    assert_eq!(xt.shape(), &[3, 2]);
    assert_eq!(xt.map(|v| v * 2.0).to_vec(), vec![2.0, 8.0, 4.0, 10.0, 6.0, 12.0]);
    assert_eq!(xt.add(&xt).unwrap().to_vec(), vec![2.0, 8.0, 4.0, 10.0, 6.0, 12.0]);

    // f32 works too. The library is generic, and only the tests pick a type.
    let f = Tensor::from_vec(vec![1.5f32, 2.5], &[2]);
    assert_eq!(f.add(&f).unwrap().to_vec(), vec![3.0f32, 5.0]);
}
