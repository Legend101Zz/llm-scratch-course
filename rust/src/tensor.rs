use crate::scalar::Scalar;
use std::rc::Rc;

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
