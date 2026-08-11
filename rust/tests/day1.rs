//! Day 1 acceptance tests — the `Scalar` trait and the `Rng`.
//!
//! The mentor writes this file. You write the code that makes it pass.
//! Do not edit the assertions. If a test is wrong, say so and argue it.
//!
//! Expected module layout. Declare both modules in `src/lib.rs`:
//!
//! ```text
//! pub mod scalar;   // pub trait Scalar
//! pub mod rng;      // pub struct Rng
//! ```
//!
//! Run with `cargo test --test day1`.
//!
//! The first run does not compile, because `src/scalar.rs` and `src/rng.rs`
//! do not exist. That failure is the red. The compiler tells you what to build.

use rustgpt::rng::Rng;
use rustgpt::scalar::Scalar;

/// The generic function from the day card. It must build for `f32` and for `f64`.
/// It uses `ZERO`, `Add`, `Div` and `from_f64`, so it fails if the bounds are wrong.
fn mean<T: Scalar>(xs: &[T]) -> T {
    let mut acc = T::ZERO;
    for &x in xs {
        acc = acc + x;
    }
    acc / T::from_f64(xs.len() as f64)
}

#[test]
fn scalar_generic_compiles() {
    let m32 = mean(&[1.0f32, 2.0, 3.0, 4.0]);
    assert!((m32 - 2.5f32).abs() < 1e-6, "f32 mean was {m32}");

    let m64 = mean(&[1.0f64, 2.0, 3.0, 4.0]);
    assert!((m64 - 2.5f64).abs() < 1e-12, "f64 mean was {m64}");

    // The two constants are values, not functions.
    assert_eq!(f32::ZERO + f32::ONE, 1.0f32);
    assert_eq!(f64::ONE.to_f64(), 1.0f64);
    assert_eq!(f32::from_f64(0.5f64), 0.5f32);
}

#[test]
fn same_seed_same_sequence() {
    let mut a = Rng::seed(42);
    let mut b = Rng::seed(42);
    for i in 0..1000 {
        assert_eq!(a.next_u64(), b.next_u64(), "streams diverged at draw {i}");
    }

    // A different seed must not give the same stream. A constant generator
    // passes the test above and is still worthless.
    let mut c = Rng::seed(42);
    let mut d = Rng::seed(43);
    let differs = (0..1000).any(|_| c.next_u64() != d.next_u64());
    assert!(differs, "seed 43 gave the same first 1000 draws as seed 42");
}

#[test]
fn uniform_in_range() {
    let mut rng = Rng::seed(7);
    for i in 0..100_000 {
        let x: f32 = rng.uniform();
        assert!((0.0..1.0).contains(&x), "f32 draw {i} was {x}, outside [0, 1)");
    }

    let mut rng = Rng::seed(7);
    for i in 0..100_000 {
        let x: f64 = rng.uniform();
        assert!((0.0..1.0).contains(&x), "f64 draw {i} was {x}, outside [0, 1)");
    }

    // A generator stuck in one octave of the range also stays inside it.
    // 100k draws must touch both halves.
    let mut rng = Rng::seed(9);
    let mut low = 0;
    let mut high = 0;
    for _ in 0..100_000 {
        let x: f64 = rng.uniform();
        if x < 0.5 {
            low += 1;
        } else {
            high += 1;
        }
    }
    assert!(low > 40_000 && high > 40_000, "split was {low} low / {high} high");
}

#[test]
fn normal_moments() {
    const N: usize = 1_000_000;
    let mut rng = Rng::seed(12345);

    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    for _ in 0..N {
        let z: f64 = rng.normal();
        sum += z;
        sum_sq += z * z;
    }

    let mean = sum / N as f64;
    let var = sum_sq / N as f64 - mean * mean;
    let sd = var.sqrt();

    // 1e6 draws give a standard error near 0.001, so 0.01 is a 10-sigma band.
    // This test fails on a wrong constant, and passes on correct Box-Muller.
    assert!(mean.abs() < 0.01, "mean was {mean}, want |mean| < 0.01");
    assert!((sd - 1.0).abs() < 0.01, "sd was {sd}, want |sd - 1| < 0.01");
}

/// The day card says "reject a seed of 0" and does not say how.
/// This test assumes a panic. xorshift64* stays at 0 forever from state 0,
/// so a 0 seed is a caller error, not a value to repair silently.
/// If you decide to remap 0 to a constant instead, delete this test and tell me why.
#[test]
#[should_panic]
fn seed_zero_is_rejected() {
    let _ = Rng::seed(0);
}
