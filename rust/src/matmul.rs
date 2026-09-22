use std::thread;

use crate::scalar::Scalar;
use crate::tensor::{broadcast_shapes, unravel_index, ShapeError, Tensor};

/// Reference implementation. NEVER optimise this. It is the oracle.
/// Rank 2 only.
pub fn matmul_naive<T: Scalar>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, ShapeError> {
    // Both inputs must be rank 2 — check each separately so the error
    // tells the caller which input was wrong.
    if a.shape().len() != 2 {
        return Err(ShapeError::BadRank {
            got: a.shape().len(),
            want: 2,
        });
    }
    if b.shape().len() != 2 {
        return Err(ShapeError::BadRank {
            got: b.shape().len(),
            want: 2,
        });
    }

    if a.shape()[1] != b.shape()[0] {
        return Err(ShapeError::SizeMismatch);
    }

    let m = a.shape()[0];
    let n = b.shape()[1];
    let k = a.shape()[1];

    let mut out: Vec<T> = vec![T::ZERO; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = T::ZERO;
            for kk in 0..k {
                sum = sum + a.get(&[i, kk]) * b.get(&[kk, j]);
            }
            out[i * n + j] = sum;
        }
    }

    Ok(Tensor::from_vec(out, &[m, n]))
}

/// Batched matrix multiply: `[..., M, K] x [..., K, N] -> [..., M, N]`.
///
/// The last two axes of each input are the matrix. Every axis before them
/// is a batch axis. Batch axes broadcast by the Day 4 rule — missing
/// leading batch dims are treated as size 1.
///
/// Rank-2 inputs work too: both batch prefixes are empty, the broadcast is
/// empty, and the function runs the oracle exactly once.
pub fn matmul<T: Scalar>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, ShapeError> {
    // 1. Both inputs need at least the matrix axes.
    if a.shape().len() < 2 {
        return Err(ShapeError::BadRank {
            got: a.shape().len(),
            want: 2,
        });
    }
    if b.shape().len() < 2 {
        return Err(ShapeError::BadRank {
            got: b.shape().len(),
            want: 2,
        });
    }

    // 2. Split each input into "batch prefix" + "matrix dims".
    //    A's matrix is [M, K], B's matrix is [K, N].
    let a_rank = a.shape().len();
    let b_rank = b.shape().len();
    let a_mat_dims = &a.shape()[a_rank - 2..];
    let b_mat_dims = &b.shape()[b_rank - 2..];

    let m = a_mat_dims[0];
    let k_a = a_mat_dims[1];
    let k_b = b_mat_dims[0];
    let n = b_mat_dims[1];

    // The K dimensions must agree — that's the dot-product axis.
    if k_a != k_b {
        return Err(ShapeError::SizeMismatch);
    }

    // 3. Broadcast the batch prefixes. Day 4 alignment-from-the-right:
    //    missing leading dims are implicitly 1.
    let a_batch = &a.shape()[..a_rank - 2];
    let b_batch = &b.shape()[..b_rank - 2];
    let batch_shape = broadcast_shapes(a_batch, b_batch)?;

    // 4. Output shape = batch_shape + [M, N]. The buffer holds
    //    prod(batch_shape) consecutive [M, N] submatrices in row-major order.
    let mut out_shape = batch_shape.clone();
    out_shape.push(m);
    out_shape.push(n);

    let batch_numel: usize = batch_shape.iter().product();
    let out_numel = batch_numel * m * n;
    let mut out_buf: Vec<T> = vec![T::ZERO; out_numel];

    // The rank of each input's batch prefix. Used to slice out the right
    // [M, K] / [K, N] chunks at each batch position.
    let a_batch_rank = a_batch.len();
    let b_batch_rank = b_batch.len();

    // 5. Run the oracle once per batch position.
    for b_lin in 0..batch_numel {
        // The full multi-dim batch index for this output slot.
        let b_idx = unravel_index(b_lin, &batch_shape);

        // Each input's batch index is the LAST r_* entries of b_idx — the
        // broadcast aligns from the right, so missing leading batch dims
        // (implicitly 1) don't contribute an index here.
        let a_batch_idx = &b_idx[batch_shape.len() - a_batch_rank..];
        let b_batch_idx = &b_idx[batch_shape.len() - b_batch_rank..];

        // Slice a down to a [1, 1, ..., 1, M, K] view, then materialise
        // (slice views are non-contiguous) and reshape to [M, K].
        //
        // When a's dim at this batch axis is 1 (it was broadcast up from 1),
        // the slice index is always 0 — there's only one slice to take.
        let mut a_view = a.clone();
        for (axis, &b_k) in a_batch_idx.iter().enumerate() {
            let k = if a.shape()[axis] == 1 { 0 } else { b_k };
            a_view = a_view.slice(axis, k, k + 1)?;
        }
        let a_mat = a_view.contiguous().reshape(&[m, k_a])?;

        // Same for b — slice, materialise, reshape to [K, N].
        let mut b_view = b.clone();
        for (axis, &b_k) in b_batch_idx.iter().enumerate() {
            let k = if b.shape()[axis] == 1 { 0 } else { b_k };
            b_view = b_view.slice(axis, k, k + 1)?;
        }
        let b_mat = b_view.contiguous().reshape(&[k_b, n])?;

        // The oracle does the arithmetic.
        let result = matmul_naive(&a_mat, &b_mat)?;

        // Place the [M, N] result into the output buffer at the right slot.
        // In row-major layout, submatrix b_lin occupies buffer positions
        // [b_lin * M * N .. (b_lin + 1) * M * N).
        let result_buf = result.to_vec();
        out_buf[b_lin * m * n..(b_lin + 1) * m * n].copy_from_slice(&result_buf);
    }

    Ok(Tensor::from_vec(out_buf, &out_shape))
}

/// Cache-blocked matmul: `[M, K] x [K, N] -> [M, N]`.
///
/// Same arithmetic as `matmul_naive` but accesses memory in `block x block`
/// tiles. While three tiles (one of A, one of B, one of C) are live, the
/// inner loops reuse them from L1 instead of going back to DRAM.
///
/// **Fast path:** when both `a` and `b` are contiguous (the common case),
/// the inner loop indexes the underlying `&[T]` slices directly. This is
/// the optimisation the Day 7 lesson points at next — see section 2.7,
/// "register blocking / packing."
///
/// **Slow path:** when either input is a transposed, sliced, or broadcast
/// view, the inner loop falls back to `a.get(&[i, k])` so the strides and
/// offset are honoured. This is what the f64 test exercises with a
/// transposed A, and it must keep working.
///
/// `block == 0` returns `SizeMismatch` — there is no sensible tile of side 0.
/// Rank-2 only, like the oracle.
pub fn matmul_blocked<T: Scalar>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    block: usize,
) -> Result<Tensor<T>, ShapeError> {
    // --- validate inputs (same checks as the oracle) ---
    if a.shape().len() != 2 {
        return Err(ShapeError::BadRank {
            got: a.shape().len(),
            want: 2,
        });
    }
    if b.shape().len() != 2 {
        return Err(ShapeError::BadRank {
            got: b.shape().len(),
            want: 2,
        });
    }
    if a.shape()[1] != b.shape()[0] {
        return Err(ShapeError::SizeMismatch);
    }
    // A tile of side 0 is degenerate — nothing to compute, and step_by(0)
    // would panic. Return an error rather than a silent zero result.
    if block == 0 {
        return Err(ShapeError::SizeMismatch);
    }

    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];

    let mut out: Vec<T> = vec![T::ZERO; m * n];

    // --- dispatch: fast path if both inputs are contiguous ---
    if a.is_contiguous() && b.is_contiguous() {
        // For contiguous rank-2 tensors of shape [M, K] and [K, N],
        // element [i, k] of A lives at a_slice[i*K + k], and element
        // [k, j] of B lives at b_slice[k*N + j]. No bounds check, no
        // stride math, no allocation per access.
        let a_slice = a.as_slice();
        let b_slice = b.as_slice();

        let mut ii = 0;
        while ii < m {
            let i_max = if ii + block < m { ii + block } else { m };
            let mut jj = 0;
            while jj < n {
                let j_max = if jj + block < n { jj + block } else { n };
                let mut kk = 0;
                while kk < k {
                    let k_max = if kk + block < k { kk + block } else { k };

                    // Inner triple loop — direct slice indexing.
                    for i in ii..i_max {
                        let a_row = &a_slice[i * k..]; // pointer to row i of A
                        for j in jj..j_max {
                            let mut sum = out[i * n + j];
                            for k_idx in kk..k_max {
                                sum = sum + a_row[k_idx] * b_slice[k_idx * n + j];
                            }
                            out[i * n + j] = sum;
                        }
                    }

                    kk += block;
                }
                jj += block;
            }
            ii += block;
        }
    } else {
        // Slow path: at least one input is a strided view. We must use
        // get() so the strides and offset are honoured. This is the
        // matmul the f64 test exercises with a transposed A.
        let mut ii = 0;
        while ii < m {
            let i_max = if ii + block < m { ii + block } else { m };
            let mut jj = 0;
            while jj < n {
                let j_max = if jj + block < n { jj + block } else { n };
                let mut kk = 0;
                while kk < k {
                    let k_max = if kk + block < k { kk + block } else { k };

                    for i in ii..i_max {
                        for j in jj..j_max {
                            let mut sum = out[i * n + j];
                            for k_idx in kk..k_max {
                                sum = sum + a.get(&[i, k_idx]) * b.get(&[k_idx, j]);
                            }
                            out[i * n + j] = sum;
                        }
                    }

                    kk += block;
                }
                jj += block;
            }
            ii += block;
        }
    }

    Ok(Tensor::from_vec(out, &[m, n]))
}

/// Row-partitioned parallel matmul: `[M, K] x [K, N] -> [M, N]`.
///
/// Same arithmetic as `matmul_blocked`, but the rows of `C` are split
/// across threads with `chunks_mut`. Each thread owns a **disjoint band
/// of whole rows** of the output, so no two threads write the same byte.
///
/// **Bit-identical to `matmul_blocked`.** A row partition leaves the
/// summation order of every output element unchanged — each thread
/// walks `k` in the same tile order as the single-threaded kernel — so
/// the rounding is identical. See `DAY_08.md` section 2.8.
///
/// **Fix B for the Rc collision.** `Rc<Vec<T>>` is `!Send`, so we cannot
/// move `&Tensor<T>` into a thread. Instead we extract plain `&[T]`
/// slices **once**, before the scope. For contiguous inputs this is a
/// free borrow (`as_slice()`); for strided inputs (transpose, slice) we
/// walk the logical view with `to_vec()`. See `DAY_08.md` section 4.2.
///
/// `block == 0` returns `SizeMismatch`. `threads == 0` returns
/// `SizeMismatch`. `threads` is a *request*: the kernel may use fewer
/// when `M < threads`. Rank-2 only, like `matmul_blocked`.
pub fn matmul_parallel<T: Scalar + Send + Sync>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    block: usize,
    threads: usize,
) -> Result<Tensor<T>, ShapeError> {
    // --- Step 1: validate inputs (same checks as the oracle) ---
    if a.shape().len() != 2 {
        return Err(ShapeError::BadRank {
            got: a.shape().len(),
            want: 2,
        });
    }
    if b.shape().len() != 2 {
        return Err(ShapeError::BadRank {
            got: b.shape().len(),
            want: 2,
        });
    }
    if a.shape()[1] != b.shape()[0] {
        return Err(ShapeError::SizeMismatch);
    }
    if block == 0 {
        return Err(ShapeError::SizeMismatch);
    }
    if threads == 0 {
        return Err(ShapeError::SizeMismatch);
    }

    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];

    // --- Step 2: row count per thread ---
    // div_ceil, not / — plain division drops remainder rows.
    let rows_per_thread = m.div_ceil(threads);

    // --- Step 3: extract &[T] (Fix B for the Rc problem) ---
    // Rc<Vec<T>> is !Send. We must get the data out as a plain &[T] before
    // any thread spawn. We do this once, OUTSIDE the scope, so it doesn't
    // happen per-thread per-tile (that would dominate the cost).
    //
    // For contiguous inputs, as_slice() borrows the underlying buffer for
    // free. For strided views, to_vec() walks the logical layout.
    // The .to_vec() at the end is a safety net: a Vec owns its data, so
    // we know it lives as long as `a_data`/`b_data` are in scope.
    let a_data: Vec<T> = if a.is_contiguous() {
        a.as_slice().to_vec()
    } else {
        a.to_vec()
    };

    let b_data: Vec<T> = if b.is_contiguous() {
        b.as_slice().to_vec()
    } else {
        b.to_vec()
    };

    // --- Step 4: allocate the output, split into row bands ---
    let mut out: Vec<T> = vec![T::ZERO; m * n];

    // --- Step 5 + 6: spawn one thread per chunk of rows ---
    //
    // We bind `&a_data` and `&b_data` to local names BEFORE the loop so
    // each closure can `move`-capture those slice references (which are
    // `Copy`) rather than the underlying `Vec<T>` (which is not). Without
    // this, the first iteration would move `a_data` into the closure and
    // the second iteration would not be able to use it.
    let a_slice: &[T] = &a_data;
    let b_slice: &[T] = &b_data;

    thread::scope(|s| {
        for (chunk_idx, chunk) in out.chunks_mut(rows_per_thread * n).enumerate() {
            // Each chunk covers `rows_per_thread` rows of the output, except the
            // last chunk which is short when M is not a multiple of rows_per.
            let row_start = chunk_idx * rows_per_thread;
            let rows_in_chunk = chunk.len() / n;
            let row_end = row_start + rows_in_chunk;

            // The closure captures:
            //   - `chunk`: &mut [T] — this thread's row band of `out`
            //   - `row_start`, `row_end`: usize — which rows of A this thread owns
            //   - `k`, `n`, `block`: usize — matrix dimensions and tile size
            //   - `a_slice`, `b_slice`: &[T] — read-only input data
            //
            // `move` moves the &mut [T] and usize captures by value. The
            // &[T] captures are Copy, so move is harmless for them.
            //
            // The function body is the same six-loop kernel as Day 7's
            // fast path, restricted to the rows [row_start, row_end).
            s.spawn(move || {
                // Six loops, restricted to this thread's rows.
                let mut ii = row_start;
                while ii < row_end {
                    let i_max = if ii + block < row_end {
                        ii + block
                    } else {
                        row_end
                    };
                    let mut jj = 0;
                    while jj < n {
                        let j_max = if jj + block < n { jj + block } else { n };
                        let mut kk = 0;
                        while kk < k {
                            let k_max = if kk + block < k { kk + block } else { k };

                            for i in ii..i_max {
                                let a_row = &a_slice[i * k..];
                                for j in jj..j_max {
                                    let mut sum = chunk[(i - row_start) * n + j];
                                    for k_idx in kk..k_max {
                                        sum = sum + a_row[k_idx] * b_slice[k_idx * n + j];
                                    }
                                    chunk[(i - row_start) * n + j] = sum;
                                }
                            }

                            kk += block;
                        }
                        jj += block;
                    }
                    ii += block;
                }
            });
        }
    });

    // --- Step 7: wrap the buffer and return ---
    Ok(Tensor::from_vec(out, &[m, n]))
}
