//! Shared helpers for the day test files.
//!
//! Cargo builds `tests/*.rs` as one test binary each. It does not build
//! `tests/common/mod.rs`, because the name is not `main.rs`. So this file is a
//! plain module. A test file pulls it in with `mod common;`.
//!
//! `#![allow(dead_code)]` is here because each test binary uses a different
//! subset. Without it, every unused helper is a warning in every other file.

#![allow(dead_code)]

use rustgpt::rng::Rng;
use rustgpt::scalar::Scalar;
use rustgpt::tensor::Tensor;

/// A tensor of standard normal draws, in row-major order.
///
/// The seed is the caller's, so every test that uses this is reproducible.
pub fn randn<T: Scalar>(rng: &mut Rng, shape: &[usize]) -> Tensor<T> {
    let n: usize = shape.iter().product();
    let data: Vec<T> = (0..n).map(|_| rng.normal()).collect();
    Tensor::from_vec(data, shape)
}

/// The `n` by `n` identity matrix.
pub fn identity<T: Scalar>(n: usize) -> Tensor<T> {
    let mut data = vec![T::ZERO; n * n];
    for i in 0..n {
        data[i * n + i] = T::ONE;
    }
    Tensor::from_vec(data, &[n, n])
}

/// Compare two tensors with a mixed absolute and relative tolerance.
///
/// The limit is `tol * (1 + |want|)`. A pure absolute limit is too tight for
/// large values. A pure relative limit divides by zero when `want` is 0.
/// The mixed form behaves like an absolute limit near 0, and like a relative
/// limit far from it. Every numerical library uses this shape.
pub fn assert_all_close<T: Scalar>(got: &Tensor<T>, want: &Tensor<T>, tol: f64, what: &str) {
    assert_eq!(got.shape(), want.shape(), "{what}: shape");
    let g = got.to_vec();
    let w = want.to_vec();
    for (i, (gi, wi)) in g.into_iter().zip(w).enumerate() {
        let (gi, wi) = (gi.to_f64(), wi.to_f64());
        let limit = tol * (1.0 + wi.abs());
        assert!(
            (gi - wi).abs() <= limit,
            "{what}: element {i} was {gi}, want {wi}, limit {limit:e}"
        );
    }
}
