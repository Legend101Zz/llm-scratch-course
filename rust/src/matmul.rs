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
        for axis in 0..a_batch_rank {
            let b_k = a_batch_idx[axis];
            let k = if a.shape()[axis] == 1 { 0 } else { b_k };
            a_view = a_view.slice(axis, k, k + 1)?;
        }
        let a_mat = a_view.contiguous().reshape(&[m, k_a])?;

        // Same for b — slice, materialise, reshape to [K, N].
        let mut b_view = b.clone();
        for axis in 0..b_batch_rank {
            let b_k = b_batch_idx[axis];
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
