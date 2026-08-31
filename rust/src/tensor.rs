use crate::scalar::Scalar;
use std::ops::Range;
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
#[derive(Clone)]
pub struct Tensor<T: Scalar> {
    data: Rc<Vec<T>>, // shared, immutable. Views alias it.
    shape: Vec<usize>,
    strides: Vec<usize>, // in elements, not bytes
    offset: usize,
}

impl<T: Scalar> Tensor<T> {
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

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn shares_storage_with(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.data, &other.data)
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
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

    pub fn get(&self, idx: &[usize]) -> T {
        let index = self.phys_index(idx);
        self.data[index]
    }

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

    pub fn transpose(&self, a: usize, b: usize) -> Result<Self, ShapeError> {
        let rank = self.shape.len();
        if a >= rank || b >= rank {
            return Err(ShapeError::OutOfBounds);
        }
        let mut perm: Vec<usize> = (0..rank).collect();
        perm.swap(a, b);
        self.permute(&perm)
    }

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

    pub fn to_vec(&self) -> Vec<T> {
        let mut output: Vec<T> = Vec::with_capacity(self.numel());
        for i in 0..self.numel() {
            let idx = unravel_index(i, &self.shape);
            output.push(self.get(&idx));
        }
        output
    }
}

pub fn unravel_index(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut idx = vec![0; shape.len()];

    for axis in (0..shape.len()).rev() {
        idx[axis] = flat % shape[axis];
        flat /= shape[axis];
    }

    idx
}

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
