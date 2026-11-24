use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Minimal, zero-dependency double-precision complex number.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };
    pub const I: Self = Self { re: 0.0, im: 1.0 };

    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline]
    pub const fn from_real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }

    #[inline]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    #[inline]
    pub fn norm_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }

    #[inline]
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    #[inline]
    pub fn is_zero(self) -> bool {
        self.re == 0.0 && self.im == 0.0
    }

    pub fn sqrt(self) -> Self {
        if self.im == 0.0 {
            if self.re >= 0.0 {
                return Self::new(self.re.sqrt(), 0.0);
            } else {
                return Self::new(0.0, (-self.re).sqrt());
            }
        }
        let r = self.abs();
        let re = ((r + self.re) * 0.5).max(0.0).sqrt();
        let im = ((r - self.re) * 0.5).max(0.0).sqrt();
        Self::new(re, if self.im < 0.0 { -im } else { im })
    }

    pub fn exp(self) -> Self {
        let er = self.re.exp();
        Self::new(er * self.im.cos(), er * self.im.sin())
    }

    pub fn ln(self) -> Self {
        Self::new(self.abs().ln(), self.arg())
    }

    pub fn sin(self) -> Self {
        Self::new(
            self.re.sin() * self.im.cosh(),
            self.re.cos() * self.im.sinh(),
        )
    }

    pub fn cos(self) -> Self {
        Self::new(
            self.re.cos() * self.im.cosh(),
            -self.re.sin() * self.im.sinh(),
        )
    }

    pub fn tan(self) -> Self {
        self.sin() / self.cos()
    }

    pub fn powc(self, exp: Self) -> Self {
        if exp.im == 0.0 && self.im == 0.0 && (self.re >= 0.0 || exp.re.fract() == 0.0) {
            return Self::new(self.re.powf(exp.re), 0.0);
        }
        if self.is_zero() {
            if exp.re > 0.0 {
                return Self::ZERO;
            }
            if exp.is_zero() {
                return Self::ONE;
            }
        }
        (self.ln() * exp).exp()
    }
}

impl Add for Complex {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl Sub for Complex {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul for Complex {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl Div for Complex {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let denom = rhs.norm_sq();
        Self::new(
            (self.re * rhs.re + self.im * rhs.im) / denom,
            (self.im * rhs.re - self.re * rhs.im) / denom,
        )
    }
}

impl Neg for Complex {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", format_complex(*self))
    }
}

pub fn format_real(v: f64) -> String {
    let v = if v == 0.0 || v.abs() < 1e-15 { 0.0 } else { v };
    if v.is_nan() {
        "NaN".to_string()
    } else if v.is_infinite() {
        if v.is_sign_positive() {
            "Inf".to_string()
        } else {
            "-Inf".to_string()
        }
    } else if v.fract().abs() < 1e-11 && v.abs() < 1e15 {
        format!("{}", v.round() as i64)
    } else {
        format!("{:.4}", v)
    }
}

pub fn format_complex(c: Complex) -> String {
    let im = if c.im.abs() < 1e-15 { 0.0 } else { c.im };
    if im == 0.0 {
        format_real(c.re)
    } else if c.re.abs() < 1e-15 {
        format!("{}i", format_real(im))
    } else if im > 0.0 {
        format!("{}+{}i", format_real(c.re), format_real(im))
    } else {
        format!("{}-{}i", format_real(c.re), format_real(-im))
    }
}
