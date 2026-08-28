pub trait Scalar:
    Copy
    + PartialOrd
    + std::fmt::Debug
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;
    const PI: Self;

    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tanh(self) -> Self;
    fn abs(self) -> Self;
    fn max(self, other: Self) -> Self;
}

impl Scalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const PI: Self = std::f32::consts::PI;

    fn from_f64(x: f64) -> Self {
        x as f32
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    fn exp(self) -> Self {
        f32::exp(self)
    }
    fn ln(self) -> Self {
        f32::ln(self)
    }
    fn sin(self) -> Self {
        f32::sin(self)
    }
    fn cos(self) -> Self {
        f32::cos(self)
    }
    fn tanh(self) -> Self {
        f32::tanh(self)
    }
    fn abs(self) -> Self {
        f32::abs(self)
    }
    fn max(self, other: Self) -> Self {
        f32::max(self, other)
    }
}

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const PI: Self = std::f64::consts::PI;

    fn from_f64(x: f64) -> Self {
        x
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    fn exp(self) -> Self {
        f64::exp(self)
    }
    fn ln(self) -> Self {
        f64::ln(self)
    }
    fn sin(self) -> Self {
        f64::sin(self)
    }
    fn cos(self) -> Self {
        f64::cos(self)
    }
    fn tanh(self) -> Self {
        f64::tanh(self)
    }
    fn abs(self) -> Self {
        f64::abs(self)
    }
    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }
}
