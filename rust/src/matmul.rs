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
/// `block == 0` returns `SizeMismatch` — there is no sensible tile of side 0.
/// Rank-2 only, like the oracle. Batched blocking is not on this card.
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

    // Six loops: three over tiles, three inside a tile.
    //
    // ii steps over rows of C in chunks of `block`.
    // jj steps over columns of C in chunks of `block`.
    // kk steps over the K dimension in chunks of `block`.
    //
    // The `min` calls clamp each inner range to the matrix boundary.
    // The last tile in any direction is short when the dimension is not
    // a multiple of `block`. This is where every blocking bug lives.
    let ii_end = m; // rename for clarity in the loops
    let jj_end = n;
    let kk_end = k;

    let mut ii = 0;
    while ii < ii_end {
        let i_max = if ii + block < ii_end { ii + block } else { ii_end };
        let mut jj = 0;
        while jj < jj_end {
            let j_max = if jj + block < jj_end { jj + block } else { jj_end };
            let mut kk = 0;
            while kk < kk_end {
                let k_max = if kk + block < kk_end { kk + block } else { kk_end };

                // Inner triple loop — the tile of C[i_max, j_max] += A[i_max, kk_max] * B[kk_max, j_max]
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

    Ok(Tensor::from_vec(out, &[m, n]))
}
