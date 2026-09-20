//! Day 9 acceptance tests — the tape.
//!
//! The mentor writes this file. You write the code that makes it pass.
//! Do not edit the assertions. If a test is wrong, say so and argue it.
//!
//! Move this file to `rust/tests/day9.rs` at the start of Day 9, then run
//! `cargo test --test day9` and watch it fail.
//!
//! Expected module layout. Add one line to `src/lib.rs`:
//!
//! ```text
//! pub mod autograd;   // NodeId, Op, Tape, and the forward helpers
//! ```
//!
//! ## The decisions this file fixes
//!
//! 1. `NodeId` derives `Clone, Copy, PartialEq, Eq, Debug`. Its inner field
//!    is private, and `NodeId::index(self) -> usize` is the one-way door out.
//!
//! 2. `Op` derives `Clone, Debug, PartialEq`. `assert_eq!` needs the last two,
//!    and a test that cannot compare an `Op` cannot check the wiring.
//!
//! 3. `Tape` exposes `len`, `is_empty`, `op(id)` and `requires_grad(id)`
//!    beyond the card's list. `DAY_09.md` section 4.2 argues each one.
//!
//! 4. **The forward helpers panic on a shape error, and return a bare
//!    `NodeId`.** A shape disagreement inside a model is a bug in the model
//!    definition, never a data-dependent state, so it is not a `Result`.
//!    If you decide otherwise, this file changes and tell me why.
//!
//! 5. `backward` is `todo!()` today. Nothing here calls it.

use rustgpt::autograd::{Op, Tape, add, mul};
use rustgpt::tensor::Tensor;

/// A rank-1 tensor of one element, so every node here has a shape.
fn scalar(v: f64) -> Tensor<f64> {
    Tensor::from_vec(vec![v], &[1])
}

// ---------------------------------------------------------------------------

/// Traps the wiring.
///
/// A tape that stores the right values and drops the edges passes every other
/// test in this file. Only the `op()` assertions below catch it.
#[test]
fn tape_records_in_order() {
    let mut tape: Tape<f64> = Tape::new();
    assert!(tape.is_empty(), "a fresh tape holds no nodes");
    assert_eq!(tape.len(), 0);

    let a = tape.leaf(scalar(2.0), true);
    let b = tape.leaf(scalar(3.0), true);
    let c = tape.leaf(scalar(4.0), true);

    let s = add(&mut tape, a, b);
    let p = mul(&mut tape, s, c);

    // 3 leaves + 1 add + 1 mul. Each helper pushes exactly one node.
    assert_eq!(tape.len(), 5, "one node per leaf and one per op, and no more");
    assert!(!tape.is_empty());

    // Ids are assigned in construction order, starting at 0.
    assert_eq!(a.index(), 0);
    assert_eq!(b.index(), 1);
    assert_eq!(c.index(), 2);
    assert_eq!(s.index(), 3);
    assert_eq!(p.index(), 4);

    // Every input id is strictly smaller than the node that consumes it.
    // That is the property the reverse loop of Day 10 rests on.
    assert!(s.index() > a.index() && s.index() > b.index());
    assert!(p.index() > s.index() && p.index() > c.index());

    assert_eq!(*tape.op(a), Op::Leaf, "a leaf records Op::Leaf");
    assert_eq!(*tape.op(s), Op::Add(a, b), "the add points at both leaves");
    assert_eq!(
        *tape.op(p),
        Op::Mul(s, c),
        "the mul points at the ADD node and at the c leaf, not at a and b"
    );

    // The forward values are computed and stored, not deferred.
    assert_eq!(tape.value(s).to_vec(), vec![5.0]);
    assert_eq!(tape.value(p).to_vec(), vec![20.0]);
}

// ---------------------------------------------------------------------------

/// Traps a missing `#[derive(Copy)]`.
///
/// This is a compile-time assertion wearing the clothes of a runtime one.
/// Without `Copy`, the second use of `x` below is a use-after-move and the
/// file does not build. Every op takes ids and returns one, so a non-`Copy`
/// id makes composition impossible.
#[test]
fn nodeid_is_copy() {
    let mut tape: Tape<f64> = Tape::new();
    let x = tape.leaf(scalar(7.0), true);

    // `x` used twice in one expression, then twice more after it.
    let doubled = add(&mut tape, x, x);
    assert_eq!(tape.value(doubled).to_vec(), vec![14.0]);

    let squared = mul(&mut tape, x, x);
    assert_eq!(tape.value(squared).to_vec(), vec![49.0]);

    // And still usable, and still comparable.
    assert_eq!(x, x);
    assert_eq!(tape.value(x).to_vec(), vec![7.0]);

    // A plain copy into another binding, with the original still live.
    let y = x;
    assert_eq!(y.index(), x.index());
}

// ---------------------------------------------------------------------------

/// Traps an offset between an id and its slot, and the `requires` rule.
///
/// Several leaves, so an off-by-one in `leaf` cannot hide behind a
/// single-element tape.
#[test]
fn value_roundtrip() {
    let mut tape: Tape<f64> = Tape::new();

    let want = [
        Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]),
        Tensor::from_vec(vec![4.0, 5.0, 6.0, 7.0], &[2, 2]),
        Tensor::from_vec(vec![8.0], &[1]),
    ];

    let ids: Vec<_> = want
        .iter()
        .map(|t| tape.leaf(t.clone(), true))
        .collect();

    for (i, id) in ids.iter().enumerate() {
        assert_eq!(tape.value(*id).shape(), want[i].shape(), "shape of leaf {i}");
        assert_eq!(tape.value(*id).to_vec(), want[i].to_vec(), "value of leaf {i}");
    }

    // requires_grad is the caller's word for a leaf.
    let data = tape.leaf(scalar(1.0), false);
    let param = tape.leaf(scalar(1.0), true);
    assert!(!tape.requires_grad(data), "a leaf keeps the flag it was given");
    assert!(tape.requires_grad(param));

    // For a non-leaf it is COMPUTED: true when any input requires it.
    let mixed = add(&mut tape, data, param);
    assert!(
        tape.requires_grad(mixed),
        "an op requires a gradient when ANY input does"
    );

    let neither_a = tape.leaf(scalar(1.0), false);
    let neither_b = tape.leaf(scalar(1.0), false);
    let neither = add(&mut tape, neither_a, neither_b);
    assert!(
        !tape.requires_grad(neither),
        "an op over two data leaves requires no gradient"
    );

    // No backward has run, so every gradient slot is still empty.
    for id in &ids {
        assert!(tape.grad(*id).is_none(), "no gradient before backward");
    }

    // zero_grad on a tape that never ran backward is a no-op, not a panic.
    tape.zero_grad();
    for id in &ids {
        assert!(tape.grad(*id).is_none(), "still empty after zero_grad");
    }
    // 3 leaves + data + param + mixed + neither_a + neither_b + neither = 9.
    assert_eq!(tape.len(), 9, "zero_grad does not change the node count");
}
