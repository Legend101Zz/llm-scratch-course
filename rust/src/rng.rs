// goal : make successive states look statistically random enough for simulation/testing.
use crate::scalar::Scalar;

const ODD_CONSTANT: u64 = 0x2545_F491_4F6C_DD1D;
const TWO_POW_53: f64 = 9_007_199_254_740_992.0;

pub struct Rng {
    state: u64,
    cache: Option<f64>,
}

impl Rng {
    pub fn seed(s: u64) -> Self {
        assert!(s != 0, "Seed must be non-zero");

        Self {
            state: s,
            cache: None,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;

        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;

        self.state = x;

        x.wrapping_mul(ODD_CONSTANT)
    }

    pub fn uniform<T: Scalar>(&mut self) -> T {
        let bits = self.next_u64() >> 11;

        let value = bits as f64 / TWO_POW_53;

        T::from_f64(value)
    }

    pub fn normal<T: Scalar>(&mut self) -> T {
        if let Some(cached) = self.cache.take() {
            return T::from_f64(cached);
        }

        let mut u1: T = self.uniform();

        while u1 == T::ZERO {
            u1 = self.uniform();
        }

        let u2: T = self.uniform();

        let r = (T::from_f64(-2.0) * u1.ln()).sqrt();

        let theta = T::from_f64(2.0) * T::PI * u2;

        let z0 = r * theta.cos();
        let z1 = r * theta.sin();

        self.cache = Some(z1.to_f64());

        z0
    }
}
