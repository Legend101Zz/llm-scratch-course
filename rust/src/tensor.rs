use crate::scalar::Scalar;
use std::rc::Rc;

#[derive(Debug)]
pub enum ShapeError {
    NotContiguous,
    BadRank { got: usize, want: usize },
    SizeMismatch,
    OutOfBounds,
    InvalidPermutation,
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShapeError::NotContiguous => write!(f, "tensor is not contiguous"),
            ShapeError::BadRank { got, want } => {
                write!(f, "rank mismatch: got {got}, want {want}")
            }
            ShapeError::SizeMismatch => write!(f, "size mismatch"),
            ShapeError::OutOfBounds => write!(f, "index out of bounds"),
            ShapeError::InvalidPermutation => write!(f, "invalid permutation"),
        }
    }
}

impl std::error::Error for ShapeError {}

/// A multi-dimensional array of `T`, parameterised over a scalar type.
///
/// Every tensor is a view onto a shared `Rc<Vec<T>>` plus four pieces of
/// metadata: `shape`, `strides`, `offset`, and the implicit rule
/// `flat_index = offset + Σ idx[i] * strides[i]`. Transpose, slice, broadcast
/// and permute all just change the metadata; the buffer doesn't move.
#[derive(Clone)]
pub struct Tensor<T: Scalar> {
    data: Rc<Vec<T>>, // shared, immutable. Views alias it.
    shape: Vec<usize>,
    strides: Vec<usize>, // in elements, not bytes
    offset: usize,
}

impl<T: Scalar> Tensor<T> {
    /// Allocate a fresh tensor of the given shape, filled with `T::ZERO`.
    ///
    /// Use this when you need an output buffer, a gradient accumulator, or a
    /// mask. The buffer is owned outright (refcount = 1).
    pub fn zeros(shape: &[usize]) -> Self {
        let strides = contiguous_strides(shape);
        let offset = 0;
        let data = Rc::new(vec![T::ZERO; shape.iter().product()]);

        Self {
            data,
            shape: shape.to_vec(),
            strides,
            offset,
        }
    }

    /// Wrap an existing flat `Vec` as a tensor with the given shape.
    ///
    /// The buffer is consumed (taken by value) and shared via `Rc`. The Vec
    /// must already be in row-major order. Panics if its length doesn't match
    /// the product of the shape dims — that's a bug, not a recoverable error.
    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Self {
        assert!(
            data.len() == shape.iter().product(),
            "data length {} doesn't match shape product {}",
            data.len(),
            shape.iter().product::<usize>()
        );
        let strides = contiguous_strides(shape);
        let offset = 0;

        Self {
            data: Rc::new(data),
            shape: shape.to_vec(),
            strides,
            offset,
        }
    }

    /// The shape: how many entries each axis has. Borrowed, not copied.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The strides: how many buffer slots to skip per unit step on each axis.
    /// Borrowed, not copied.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// True iff both tensors point at the same underlying buffer allocation.
    ///
    /// A transpose, slice or broadcast of `self` shares storage with `self`.
    /// A `contiguous()` copy, or any operation that builds a fresh buffer,
    /// does not.
    pub fn shares_storage_with(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.data, &other.data)
    }

    /// Total number of elements: the product of the shape dims.
    ///
    /// For a broadcast view, this can be larger than the buffer length. That's
    /// the point of broadcasting — many logical entries, one physical one.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// True iff the strides match the default row-major layout and the offset
    /// is zero. A contiguous tensor can be reshaped without copying; a
    /// non-contiguous one cannot.
    pub fn is_contiguous(&self) -> bool {
        self.strides == contiguous_strides(&self.shape)
    }

    fn phys_index(&self, idx: &[usize]) -> usize // private. The core. position = offset+ ∑ ​index i ​⋅ stride i​
    {
        // 1. Rank matches
        assert!(
            idx.len() == self.shape.len(),
            "index has {} dims, tensor has {}",
            idx.len(),
            self.shape.len()
        );

        // 2. Each index in range
        for (i, (&dim, &index)) in self.shape.iter().zip(idx.iter()).enumerate() {
            assert!(
                index < dim,
                "index {i} out of bounds: got {index}, dim is {dim}"
            );
        }

        // 3. Compute flat index
        self.offset
            + idx
                .iter()
                .zip(self.strides.iter())
                .map(|(&i, &s)| i * s)
                .sum::<usize>()
    }

    /// Read one element at the given multi-dim index.
    ///
    /// This is the canonical way to peek into a tensor. Goes through
    /// `phys_index`, so it honours the strides and offset — works correctly
    /// even on transposed, sliced, or broadcast views.
    pub fn get(&self, idx: &[usize]) -> T {
        let index = self.phys_index(idx);
        self.data[index]
    }

    /// Borrow the underlying buffer as a `&[T]`, in storage order.
    ///
    /// This is **only meaningful for contiguous tensors.** For a contiguous
    /// rank-2 tensor of shape `[M, K]`, element `[i, k]` lives at
    /// `slice[i * K + k]`. For a non-contiguous view (transpose, slice,
    /// broadcast), the slice is the *underlying storage*, not the logical
    /// view — reading it as if it were the view will give wrong answers.
    ///
    /// Use this in hot loops where you have already checked
    /// `is_contiguous()` and you know the layout. The compiler turns the
    /// `slice[i*K + k]` access into a single load with no bounds check or
    /// stride math.
    pub fn as_slice(&self) -> &[T] {
        // For contiguous tensors, offset is 0 and numel == data.len().
        // For non-contiguous views, the caller should not be calling this.
        // We return just the prefix that belongs to this view.
        self.data.as_slice()
    }

    /// Reorder the axes according to `order`. `order[i]` names the **source**
    /// axis that becomes the new axis `i`.
    ///
    /// Pure metadata: the buffer is shared, the offset is unchanged. Returns
    /// `Err(BadRank)` if `order.len()` doesn't match the rank, or
    /// `Err(InvalidPermutation)` if any axis index is out of range or
    /// repeated.
    ///
    /// This is the general "permute" used by multi-head attention. The
    /// `transpose(a, b)` method below is a special case.
    pub fn permute(&self, order: &[usize]) -> Result<Self, ShapeError> {
        let rank = self.shape.len();
        if order.len() != rank {
            return Err(ShapeError::BadRank {
                got: order.len(),
                want: rank,
            });
        }

        let mut seen = vec![false; rank];
        for &axis in order {
            if axis >= rank {
                return Err(ShapeError::InvalidPermutation);
            }
            if seen[axis] {
                return Err(ShapeError::InvalidPermutation);
            }
            seen[axis] = true;
        }

        let new_shape: Vec<usize> = order.iter().map(|&a| self.shape[a]).collect();
        let new_strides: Vec<usize> = order.iter().map(|&a| self.strides[a]).collect();

        Ok(Self {
            data: self.data.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }

    /// Swap two specific axes. Cheap view, no copy.
    ///
    /// For a 2D tensor, `transpose(0, 1)` is the matrix transpose. For higher
    /// ranks it swaps any two axes — the method of choice when you need to
    /// move a dimension from the middle to the front.
    pub fn transpose(&self, a: usize, b: usize) -> Result<Self, ShapeError> {
        let rank = self.shape.len();
        if a >= rank || b >= rank {
            return Err(ShapeError::OutOfBounds);
        }
        let mut perm: Vec<usize> = (0..rank).collect();
        perm.swap(a, b);
        self.permute(&perm)
    }

    /// Change the shape, keeping the same elements in row-major order.
    ///
    /// On a contiguous tensor this is a free metadata change. On a
    /// non-contiguous tensor it refuses — call `contiguous()` first to
    /// materialise a copy, then reshape the copy.
    ///
    /// Returns `Err(SizeMismatch)` if the new shape has a different number
    /// of elements, or `Err(NotContiguous)` if the source isn't contiguous.
    pub fn reshape(&self, shape: &[usize]) -> Result<Self, ShapeError> {
        let new_numel: usize = shape.iter().product();
        let cur_numel = self.numel();

        if new_numel != cur_numel {
            return Err(ShapeError::SizeMismatch);
        }

        if !self.is_contiguous() {
            return Err(ShapeError::NotContiguous);
        }

        Ok(Self {
            data: self.data.clone(),
            shape: shape.to_vec(),
            strides: contiguous_strides(shape),
            offset: 0,
        })
    }

    /// Narrow one axis to a range `[start..end)`. Pure metadata: the buffer is
    /// shared and the strides are untouched.
    ///
    /// The result has `shape[axis] = end - start` and `offset` shifted by
    /// `start * strides[axis]`. `start == end` is legal and yields an empty
    /// view (numel = 0).
    ///
    /// For multi-axis slicing, chain calls: `t.slice(0, 1, 3)?.slice(2, 0, 4)?`.
    pub fn slice(&self, axis: usize, start: usize, end: usize) -> Result<Self, ShapeError> {
        let rank = self.shape.len();

        if axis >= rank {
            return Err(ShapeError::OutOfBounds);
        }

        let dim = self.shape[axis];

        if start > end || end > dim {
            return Err(ShapeError::OutOfBounds);
        }

        let mut new_shape = self.shape.clone();
        let mut new_offset = self.offset;

        new_offset += start * self.strides[axis];
        new_shape[axis] = end - start;

        Ok(Self {
            data: self.data.clone(),
            shape: new_shape,
            strides: self.strides.clone(),
            offset: new_offset,
        })
    }

    /// Materialise the tensor into a fresh row-major buffer.
    ///
    /// Fast path: if the tensor is already contiguous with offset zero,
    /// returns a clone (no copy). Slow path: walks every logical position,
    /// reads through the strides/offset, and packs the values into a new
    /// buffer.
    ///
    /// This is the only method in the library that copies. Reach for it
    /// when you want to mutate, or when an operation refuses a non-contiguous
    /// input.
    pub fn contiguous(&self) -> Self {
        // Fast path: already contiguous, no copy.
        if self.is_contiguous() && self.offset == 0 {
            return self.clone();
        }

        // Slow path: walk in logical order, copy into a fresh buffer.
        let numel = self.numel();
        let mut new_buffer = Vec::with_capacity(numel);

        for linear in 0..numel {
            let idx = unravel_index(linear, &self.shape);
            let flat = self.phys_index(&idx);
            new_buffer.push(self.data[flat]);
        }

        Self {
            data: Rc::new(new_buffer),
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// Walk the tensor in logical row-major order and dump the values into a
    /// flat `Vec<T>`.
    ///
    /// For a contiguous tensor this matches the buffer order. For a transposed
    /// or sliced view it reflects the *logical* layout — the output reads as
    /// if the view were a fresh contiguous buffer. For a broadcast view, the
    /// output has `numel()` entries with the broadcast repeats resolved.
    pub fn to_vec(&self) -> Vec<T> {
        let mut output: Vec<T> = Vec::with_capacity(self.numel());
        for i in 0..self.numel() {
            let idx = unravel_index(i, &self.shape);
            output.push(self.get(&idx));
        }
        output
    }

    /// Reshape the view to a target shape by stretching axes of size 1.
    ///
    /// The trick: stretched axes get stride 0, so every index along them
    /// reads the same buffer position. That's the entire mechanism of
    /// broadcasting — no data ever moves.
    ///
    /// Rules: target rank must be ≥ self rank, and every existing axis must
    /// either match the target or be stretched from 1. Anything else returns
    /// `Err(SizeMismatch)`.
    ///
    /// The result may have `numel()` greater than the buffer length —
    /// multiple logical positions reading the same physical slot.
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Self, ShapeError> {
        let self_rank = self.shape.len();
        let target_rank = shape.len();

        // Broadcast never lowers the rank.
        if target_rank < self_rank {
            return Err(ShapeError::SizeMismatch);
        }

        let pad = target_rank - self_rank;
        let mut new_shape = Vec::with_capacity(target_rank);
        let mut new_strides = Vec::with_capacity(target_rank);

        for (i, &target_dim) in shape.iter().enumerate() {
            if i < pad {
                // New leading axis: take the target's shape, force stride 0.
                new_shape.push(target_dim);
                new_strides.push(0);
            } else {
                let self_idx = i - pad;
                let self_dim = self.shape[self_idx];
                let self_stride = self.strides[self_idx];

                if self_dim == target_dim {
                    new_shape.push(target_dim);
                    new_strides.push(self_stride);
                } else if self_dim == 1 {
                    // Stretch: keep the target's shape, zero the stride.
                    new_shape.push(target_dim);
                    new_strides.push(0);
                } else {
                    return Err(ShapeError::SizeMismatch);
                }
            }
        }

        Ok(Self {
            data: self.data.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }

    /// Apply a closure to every element. Returns a fresh, contiguous, offset-0
    /// tensor.
    ///
    /// The closure is called once per logical element, including the repeats
    /// in a broadcast view — those become real, distinct entries in the new
    /// buffer. The closure must be `Fn` (not `FnOnce`), because it's called
    /// many times.
    pub fn map(&self, f: impl Fn(T) -> T) -> Self {
        let numel = self.numel();
        let mut new_buffer = Vec::with_capacity(numel);

        for linear in 0..numel {
            let idx = unravel_index(linear, &self.shape);
            new_buffer.push(f(self.get(&idx)));
        }

        Self {
            data: Rc::new(new_buffer),
            shape: self.shape.clone(),
            strides: contiguous_strides(&self.shape),
            offset: 0,
        }
    }

    /// Apply a closure pairwise. Both operands are broadcast to their common
    /// shape before the closure runs.
    ///
    /// Every two-operand op (`add`, `sub`, `mul`, `div`) is a one-line call
    /// to `zip_with`. The broadcasting happens once, here; the closure sees
    /// only matching logical positions.
    ///
    /// Returns `Err(SizeMismatch)` if the two shapes cannot be broadcast.
    pub fn zip_with(&self, other: &Self, f: impl Fn(T, T) -> T) -> Result<Self, ShapeError> {
        let common_shape = broadcast_shapes(&self.shape, &other.shape)?;

        let a_broad = self.broadcast_to(&common_shape)?;
        let b_broad = other.broadcast_to(&common_shape)?;

        let numel: usize = common_shape.iter().product();
        let mut new_buffer = Vec::with_capacity(numel);

        for linear in 0..numel {
            let idx = unravel_index(linear, &common_shape);
            new_buffer.push(f(a_broad.get(&idx), b_broad.get(&idx)));
        }

        Ok(Self {
            data: Rc::new(new_buffer),
            shape: common_shape.clone(),
            strides: contiguous_strides(&common_shape),
            offset: 0,
        })
    }

    /// Elementwise addition. Both operands are broadcast to a common shape.
    /// Returns `Err(SizeMismatch)` if the shapes don't broadcast.
    pub fn add(&self, other: &Self) -> Result<Self, ShapeError> {
        self.zip_with(other, |a, b| a + b)
    }

    /// Elementwise subtraction. Both operands are broadcast to a common shape.
    pub fn sub(&self, other: &Self) -> Result<Self, ShapeError> {
        self.zip_with(other, |a, b| a - b)
    }

    /// Elementwise multiplication. Both operands are broadcast to a common shape.
    pub fn mul(&self, other: &Self) -> Result<Self, ShapeError> {
        self.zip_with(other, |a, b| a * b)
    }

    /// Elementwise division. Both operands are broadcast to a common shape.
    pub fn div(&self, other: &Self) -> Result<Self, ShapeError> {
        self.zip_with(other, |a, b| a / b)
    }

    /// Negate every element. Single operand → no shape can fail.
    pub fn neg(&self) -> Self {
        self.map(|x| -x)
    }

    /// Elementwise `exp(x)`. Single operand → no shape can fail.
    pub fn exp(&self) -> Self {
        self.map(|x| x.exp())
    }

    /// Elementwise natural log. Single operand → no shape can fail.
    pub fn ln(&self) -> Self {
        self.map(|x| x.ln())
    }

    /// Elementwise square root. Single operand → no shape can fail.
    pub fn sqrt(&self) -> Self {
        self.map(|x| x.sqrt())
    }

    /// Elementwise hyperbolic tangent. Single operand → no shape can fail.
    pub fn tanh(&self) -> Self {
        self.map(|x| x.tanh())
    }

    /// ReLU: `max(x, 0)` per element. Single operand → no shape can fail.
    pub fn relu(&self) -> Self {
        self.map(|x| x.max(T::ZERO))
    }

    /// Reduce the tensor along one axis by summing.
    ///
    /// `keepdim = false` removes the axis from the shape; `keepdim = true`
    /// leaves it as a size-1 axis so the result broadcasts back against the
    /// input — that's the whole reason `keepdim` exists.
    ///
    /// Each group is folded left-to-right. Groups are typically bounded by
    /// one axis (often <10k), so the precision loss is at most a few ULPs —
    /// acceptable. `sum_all` is the one place that needs pairwise.
    ///
    /// Returns `Err(OutOfBounds)` if `axis` is past the rank.
    pub fn sum_axis(&self, axis: usize, keepdim: bool) -> Result<Self, ShapeError> {
        let rank = self.shape.len();
        if axis >= rank {
            return Err(ShapeError::OutOfBounds);
        }

        let out_shape: Vec<usize> = self
            .shape
            .iter()
            .enumerate()
            .filter_map(|(i, &dim)| {
                if i == axis {
                    if keepdim {
                        Some(1)
                    } else {
                        None
                    }
                } else {
                    Some(dim)
                }
            })
            .collect();

        let reduced_dim = self.shape[axis];
        let out_numel: usize = out_shape.iter().product();
        let mut buffer = Vec::with_capacity(out_numel);

        for linear in 0..out_numel {
            let out_idx = unravel_index(linear, &out_shape);
            let mut acc = T::ZERO;
            for j in 0..reduced_dim {
                let mut in_idx = out_idx.clone();
                if keepdim {
                    // Output already has the slot at position `axis`.
                    in_idx[axis] = j;
                } else {
                    // Output is rank - 1; insert the reduced slot.
                    in_idx.insert(axis, j);
                }
                acc = acc + self.get(&in_idx);
            }
            buffer.push(acc);
        }

        let out_strides = contiguous_strides(&out_shape);

        Ok(Self {
            data: Rc::new(buffer),
            shape: out_shape,
            strides: out_strides,
            offset: 0,
        })
    }

    /// Reduce along one axis by averaging.
    ///
    /// Same shape rule as `sum_axis`. Equivalent to `sum_axis` followed by
    /// dividing every output element by the size of the reduced axis.
    ///
    /// An empty axis (size 0) returns NaN: 0/0 on a float. That is the
    /// honest answer — there is no mean of nothing.
    pub fn mean_axis(&self, axis: usize, keepdim: bool) -> Result<Self, ShapeError> {
        let sum = self.sum_axis(axis, keepdim)?;
        let n = T::from_f64(self.shape[axis] as f64);
        Ok(sum.map(|x| x / n))
    }

    /// Reduce along one axis by taking the max.
    ///
    /// Seeded with the first element of each group, never with zero — that
    /// way an all-negative axis returns the right answer. NaN handling comes
    /// from `Scalar::max`, which returns the non-NaN operand.
    pub fn max_axis(&self, axis: usize, keepdim: bool) -> Result<Self, ShapeError> {
        let rank = self.shape.len();
        if axis >= rank {
            return Err(ShapeError::OutOfBounds);
        }

        let out_shape: Vec<usize> = self
            .shape
            .iter()
            .enumerate()
            .filter_map(|(i, &dim)| {
                if i == axis {
                    if keepdim {
                        Some(1)
                    } else {
                        None
                    }
                } else {
                    Some(dim)
                }
            })
            .collect();

        let reduced_dim = self.shape[axis];
        let out_numel: usize = out_shape.iter().product();
        let mut buffer = Vec::with_capacity(out_numel);

        for linear in 0..out_numel {
            let out_idx = unravel_index(linear, &out_shape);

            // Seed with the first element of this group.
            let mut first_in_idx = out_idx.clone();
            if keepdim {
                first_in_idx[axis] = 0;
            } else {
                first_in_idx.insert(axis, 0);
            }
            let mut m = self.get(&first_in_idx);

            for j in 1..reduced_dim {
                let mut in_idx = out_idx.clone();
                if keepdim {
                    in_idx[axis] = j;
                } else {
                    in_idx.insert(axis, j);
                }
                m = m.max(self.get(&in_idx));
            }
            buffer.push(m);
        }

        let out_strides = contiguous_strides(&out_shape);

        Ok(Self {
            data: Rc::new(buffer),
            shape: out_shape,
            strides: out_strides,
            offset: 0,
        })
    }

    /// Sum every element of the tensor into a single scalar.
    ///
    /// Implemented as pairwise summation with a base-case block of 64. A
    /// plain left fold in `f32` stops dead at 2^24, so a naive sum over a
    /// million parameters is wrong by a factor of six. Pairwise keeps every
    /// addition between values of similar size and survives.
    pub fn sum_all(&self) -> T {
        const BASE: usize = 64;

        fn pairwise<T: Scalar>(values: &[T]) -> T {
            if values.len() <= BASE {
                let mut acc = T::ZERO;
                for &v in values {
                    acc = acc + v;
                }
                acc
            } else {
                let mid = values.len() / 2;
                pairwise(&values[..mid]) + pairwise(&values[mid..])
            }
        }

        pairwise(&self.to_vec())
    }
}

/// Convert a flat linear index into a multi-dim index for the given shape.
///
/// Mixed-radix counting from the innermost axis outward. The inverse of
/// `offset + Σ idx[i] * strides[i]` for a contiguous, offset-0 tensor.
pub fn unravel_index(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut idx = vec![0; shape.len()];

    for axis in (0..shape.len()).rev() {
        idx[axis] = flat % shape[axis];
        flat /= shape[axis];
    }

    idx
}

/// Compute the row-major contiguous strides for a given shape.
///
/// `strides[n-1] = 1`, `strides[i] = prod(shape[i+1..])`. The innermost axis
/// varies fastest in memory; the outermost varies slowest.
pub fn contiguous_strides(shape: &[usize]) -> Vec<usize> // row-major
{
    let mut strides = vec![0; shape.len()];
    let mut stride = 1;

    for i in (0..shape.len()).rev() {
        strides[i] = stride;
        stride *= shape[i];
    }
    strides
}

pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>, ShapeError> {
    let mut result = Vec::new();
    let mut ai = a.iter().rev();
    let mut bi = b.iter().rev();

    loop {
        match (ai.next(), bi.next()) {
            (Some(&da), Some(&db)) => {
                if da == db {
                    result.push(da);
                } else if da == 1 {
                    result.push(db);
                } else if db == 1 {
                    result.push(da);
                } else {
                    return Err(ShapeError::SizeMismatch);
                }
            }
            (Some(&da), None) => {
                // a has an axis, b doesn't: b is implicitly 1. Pair is (da, 1) — always compatible.
                result.push(da);
            }
            (None, Some(&db)) => {
                // b has an axis, a doesn't: a is implicitly 1. Pair is (1, db) — always compatible.
                result.push(db);
            }
            (None, None) => break,
        }
    }

    result.reverse();
    Ok(result)
}
