use std::fmt;

use crate::complex::{Complex, format_complex, format_real};
use crate::{MatroError, Result};

/// MATLAB numeric / logical element storage class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NumericClass {
    #[default]
    Double,
    Single,
    Logical,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
}

impl NumericClass {
    pub fn name(self) -> &'static str {
        match self {
            Self::Double => "double",
            Self::Single => "single",
            Self::Logical => "logical",
            Self::Int8 => "int8",
            Self::Int16 => "int16",
            Self::Int32 => "int32",
            Self::Int64 => "int64",
            Self::Uint8 => "uint8",
            Self::Uint16 => "uint16",
            Self::Uint32 => "uint32",
            Self::Uint64 => "uint64",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "double" => Some(Self::Double),
            "single" => Some(Self::Single),
            "logical" => Some(Self::Logical),
            "int8" => Some(Self::Int8),
            "int16" => Some(Self::Int16),
            "int32" => Some(Self::Int32),
            "int64" => Some(Self::Int64),
            "uint8" => Some(Self::Uint8),
            "uint16" => Some(Self::Uint16),
            "uint32" => Some(Self::Uint32),
            "uint64" => Some(Self::Uint64),
            _ => None,
        }
    }

    #[inline]
    pub fn is_integer(self) -> bool {
        matches!(
            self,
            Self::Int8
                | Self::Int16
                | Self::Int32
                | Self::Int64
                | Self::Uint8
                | Self::Uint16
                | Self::Uint32
                | Self::Uint64
        )
    }

    #[inline]
    pub fn is_float(self) -> bool {
        matches!(self, Self::Double | Self::Single)
    }

    #[inline]
    pub fn clamp_round(self, v: f64) -> f64 {
        match self {
            Self::Double => v,
            Self::Single => (v as f32) as f64,
            Self::Logical => {
                if v != 0.0 && !v.is_nan() {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Int8 => {
                if v.is_nan() { 0.0 } else { v.round().clamp(i8::MIN as f64, i8::MAX as f64) }
            }
            Self::Int16 => {
                if v.is_nan() { 0.0 } else { v.round().clamp(i16::MIN as f64, i16::MAX as f64) }
            }
            Self::Int32 => {
                if v.is_nan() { 0.0 } else { v.round().clamp(i32::MIN as f64, i32::MAX as f64) }
            }
            Self::Int64 => {
                if v.is_nan() { 0.0 } else { v.round().clamp(i64::MIN as f64, i64::MAX as f64) }
            }
            Self::Uint8 => {
                if v.is_nan() { 0.0 } else { v.round().clamp(0.0, u8::MAX as f64) }
            }
            Self::Uint16 => {
                if v.is_nan() { 0.0 } else { v.round().clamp(0.0, u16::MAX as f64) }
            }
            Self::Uint32 => {
                if v.is_nan() { 0.0 } else { v.round().clamp(0.0, u32::MAX as f64) }
            }
            Self::Uint64 => {
                if v.is_nan() { 0.0 } else { v.round().clamp(0.0, u64::MAX as f64) }
            }
        }
    }

    pub fn bounds(self) -> Option<(f64, f64)> {
        match self {
            Self::Int8 => Some((i8::MIN as f64, i8::MAX as f64)),
            Self::Int16 => Some((i16::MIN as f64, i16::MAX as f64)),
            Self::Int32 => Some((i32::MIN as f64, i32::MAX as f64)),
            Self::Int64 => Some((i64::MIN as f64, i64::MAX as f64)),
            Self::Uint8 => Some((0.0, u8::MAX as f64)),
            Self::Uint16 => Some((0.0, u16::MAX as f64)),
            Self::Uint32 => Some((0.0, u32::MAX as f64)),
            Self::Uint64 => Some((0.0, u64::MAX as f64)),
            _ => None,
        }
    }

    /// Combine two numeric classes according to MATLAB arithmetic promotion rules.
    pub fn combine(a: Self, b: Self) -> Result<Self> {
        if a == b {
            return Ok(if a == Self::Logical { Self::Double } else { a });
        }
        match (a, b) {
            (Self::Logical, other) | (other, Self::Logical) => {
                Ok(if other == Self::Logical { Self::Double } else { other })
            }
            (Self::Double, other) | (other, Self::Double) => Ok(other),
            (x, y) if x.is_integer() && y.is_integer() => Err(MatroError::runtime(format!(
                "Integers can only be combined with integers of the same class or scalar doubles (got {} and {})",
                x.name(),
                y.name()
            ))),
            (Self::Single, x) | (x, Self::Single) if x.is_integer() => {
                Err(MatroError::runtime(format!(
                    "Cannot combine {} and single directly",
                    x.name()
                )))
            }
            _ => Ok(Self::Double),
        }
    }
}

/// Column-major N-D array supporting real (`im: None`), complex (`im: Some(...)`),
/// logical masks, `single`, and saturating MATLAB integer classes (`int8`..`uint64`).
/// 2D matrices keep `extra_dims: None` for zero overhead; 3D+ arrays store `[d3, d4, ...]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    extra_dims: Option<Vec<usize>>,
    re: Vec<f64>,
    im: Option<Vec<f64>>,
    dtype: NumericClass,
}

impl Matrix {
    /// Creates an N-D real array from column-major data and dimension list (at least 2 dims).
    pub fn from_nd_col_major(mut dims: Vec<usize>, data: Vec<f64>) -> Result<Self> {
        while dims.len() > 2 && dims.last() == Some(&1) {
            dims.pop();
        }
        while dims.len() < 2 {
            dims.push(1);
        }
        let expected: usize = dims.iter().copied().try_fold(1usize, usize::checked_mul).unwrap_or(usize::MAX);
        if expected != data.len() {
            return Err(MatroError::dimension(format!(
                "Cannot create {:?} array from {} elements",
                dims,
                data.len()
            )));
        }
        let rows = dims[0];
        let cols = dims[1];
        let extra_dims = if dims.len() > 2 {
            Some(dims[2..].to_vec())
        } else {
            None
        };
        Ok(Self {
            rows,
            cols,
            extra_dims,
            re: data,
            im: None,
            dtype: NumericClass::Double,
        })
    }

    pub fn from_nd_complex_col_major(
        mut dims: Vec<usize>,
        re: Vec<f64>,
        im: Vec<f64>,
    ) -> Result<Self> {
        while dims.len() > 2 && dims.last() == Some(&1) {
            dims.pop();
        }
        while dims.len() < 2 {
            dims.push(1);
        }
        let expected: usize = dims.iter().copied().try_fold(1usize, usize::checked_mul).unwrap_or(usize::MAX);
        if expected != re.len() || re.len() != im.len() {
            return Err(MatroError::dimension(format!(
                "Cannot create {:?} complex array from {} elements",
                dims,
                re.len()
            )));
        }
        let rows = dims[0];
        let cols = dims[1];
        let extra_dims = if dims.len() > 2 {
            Some(dims[2..].to_vec())
        } else {
            None
        };
        let has_imag = im.iter().any(|&x| x != 0.0);
        Ok(Self {
            rows,
            cols,
            extra_dims,
            re,
            im: if has_imag { Some(im) } else { None },
            dtype: NumericClass::Double,
        })
    }

    pub fn zeros_nd(dims: Vec<usize>) -> Self {
        let total: usize = dims.iter().product();
        Self::from_nd_col_major(dims, vec![0.0; total]).unwrap()
    }

    pub fn ones_nd(dims: Vec<usize>) -> Self {
        let total: usize = dims.iter().product();
        Self::from_nd_col_major(dims, vec![1.0; total]).unwrap()
    }

    pub fn dims(&self) -> Vec<usize> {
        let mut d = vec![self.rows, self.cols];
        if let Some(ext) = &self.extra_dims {
            d.extend_from_slice(ext);
        }
        d
    }

    pub fn ndims(&self) -> usize {
        2 + self.extra_dims.as_ref().map_or(0, Vec::len)
    }

    pub fn dim_size(&self, dim_0based: usize) -> usize {
        match dim_0based {
            0 => self.rows,
            1 => self.cols,
            k => self
                .extra_dims
                .as_ref()
                .and_then(|e| e.get(k - 2).copied())
                .unwrap_or(1),
        }
    }

    pub fn with_dims(mut self, dims: Vec<usize>) -> Result<Self> {
        let mut d = dims;
        while d.len() > 2 && d.last() == Some(&1) {
            d.pop();
        }
        while d.len() < 2 {
            d.push(1);
        }
        let total: usize = d.iter().product();
        if total != self.re.len() {
            return Err(MatroError::dimension(format!(
                "Cannot reshape {} elements into {:?}",
                self.re.len(),
                d
            )));
        }
        self.rows = d[0];
        self.cols = d[1];
        self.extra_dims = if d.len() > 2 { Some(d[2..].to_vec()) } else { None };
        Ok(self)
    }

    /// Creates a real matrix from column-major data.
    pub fn from_col_major(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self> {
        if rows.saturating_mul(cols) != data.len() {
            return Err(MatroError::dimension(format!(
                "Cannot create {rows}x{cols} matrix from {} elements",
                data.len()
            )));
        }
        Ok(Self {
            rows,
            cols,
            extra_dims: None,
            re: data,
            im: None,
            dtype: NumericClass::Double,
        })
    }

    /// Creates a complex matrix from column-major real and imaginary vectors.
    pub fn from_complex_col_major(
        rows: usize,
        cols: usize,
        re: Vec<f64>,
        im: Vec<f64>,
    ) -> Result<Self> {
        if rows.saturating_mul(cols) != re.len() || re.len() != im.len() {
            return Err(MatroError::dimension(format!(
                "Cannot create {rows}x{cols} complex matrix from {} / {} elements",
                re.len(),
                im.len()
            )));
        }
        let has_imag = im.iter().any(|&x| x != 0.0);
        Ok(Self {
            rows,
            cols,
            extra_dims: None,
            re,
            im: if has_imag { Some(im) } else { None },
            dtype: NumericClass::Double,
        })
    }

    /// Creates a real matrix from row-major data (convenient for literals & tests).
    pub fn from_row_major(rows: usize, cols: usize, row_data: &[f64]) -> Result<Self> {
        if rows.saturating_mul(cols) != row_data.len() {
            return Err(MatroError::dimension(format!(
                "Cannot create {rows}x{cols} matrix from {} elements",
                row_data.len()
            )));
        }
        let mut re = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                re[c * rows + r] = row_data[r * cols + c];
            }
        }
        Ok(Self {
            rows,
            cols,
            extra_dims: None,
            re,
            im: None,
            dtype: NumericClass::Double,
        })
    }

    pub fn scalar(v: f64) -> Self {
        Self {
            rows: 1,
            cols: 1,
            extra_dims: None,
            re: vec![v],
            im: None,
            dtype: NumericClass::Double,
        }
    }

    pub fn logical_scalar(b: bool) -> Self {
        Self {
            rows: 1,
            cols: 1,
            extra_dims: None,
            re: vec![if b { 1.0 } else { 0.0 }],
            im: None,
            dtype: NumericClass::Logical,
        }
    }

    pub fn complex_scalar(c: Complex) -> Self {
        if c.im == 0.0 {
            Self::scalar(c.re)
        } else {
            Self {
                rows: 1,
                cols: 1,
                extra_dims: None,
                re: vec![c.re],
                im: Some(vec![c.im]),
                dtype: NumericClass::Double,
            }
        }
    }

    pub fn empty() -> Self {
        Self {
            rows: 0,
            cols: 0,
            extra_dims: None,
            re: Vec::new(),
            im: None,
            dtype: NumericClass::Double,
        }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            extra_dims: None,
            re: vec![0.0; rows * cols],
            im: None,
            dtype: NumericClass::Double,
        }
    }

    pub fn complex_zeros(rows: usize, cols: usize) -> Self {
        let len = rows * cols;
        Self {
            rows,
            cols,
            extra_dims: None,
            re: vec![0.0; len],
            im: Some(vec![0.0; len]),
            dtype: NumericClass::Double,
        }
    }

    pub fn ones(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            extra_dims: None,
            re: vec![1.0; rows * cols],
            im: None,
            dtype: NumericClass::Double,
        }
    }

    pub fn eye(rows: usize, cols: usize) -> Self {
        let mut m = Self::zeros(rows, cols);
        let n = rows.min(cols);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    pub fn row_vec(data: Vec<f64>) -> Self {
        let cols = data.len();
        let rows = if cols == 0 { 0 } else { 1 };
        Self {
            rows,
            cols,
            extra_dims: None,
            re: data,
            im: None,
            dtype: NumericClass::Double,
        }
    }

    pub fn col_vec(data: Vec<f64>) -> Self {
        let rows = data.len();
        let cols = if rows == 0 { 0 } else { 1 };
        Self {
            rows,
            cols,
            extra_dims: None,
            re: data,
            im: None,
            dtype: NumericClass::Double,
        }
    }

    pub fn complex_row_vec(vals: Vec<Complex>) -> Self {
        let cols = vals.len();
        let rows = if cols == 0 { 0 } else { 1 };
        let has_imag = vals.iter().any(|c| c.im != 0.0);
        let re = vals.iter().map(|c| c.re).collect();
        let im = if has_imag {
            Some(vals.iter().map(|c| c.im).collect())
        } else {
            None
        };
        Self {
            rows,
            cols,
            extra_dims: None,
            re,
            im,
            dtype: NumericClass::Double,
        }
    }

    pub fn complex_col_vec(vals: Vec<Complex>) -> Self {
        let rows = vals.len();
        let cols = if rows == 0 { 0 } else { 1 };
        let has_imag = vals.iter().any(|c| c.im != 0.0);
        let re = vals.iter().map(|c| c.re).collect();
        let im = if has_imag {
            Some(vals.iter().map(|c| c.im).collect())
        } else {
            None
        };
        Self {
            rows,
            cols,
            extra_dims: None,
            re,
            im,
            dtype: NumericClass::Double,
        }
    }

    #[inline]
    pub fn dtype(&self) -> NumericClass {
        self.dtype
    }

    #[inline]
    pub fn is_logical(&self) -> bool {
        self.dtype == NumericClass::Logical
    }

    pub fn cast_to(mut self, target: NumericClass) -> Self {
        self.dtype = target;
        if target == NumericClass::Logical || target.is_integer() {
            self.im = None;
        }
        for v in &mut self.re {
            *v = target.clamp_round(*v);
        }
        if let Some(im) = &mut self.im {
            for v in im {
                *v = target.clamp_round(*v);
            }
        }
        self
    }

    pub fn into_logical(self) -> Self {
        self.cast_to(NumericClass::Logical)
    }

    pub fn with_logical(mut self, logical: bool) -> Self {
        self.dtype = if logical {
            NumericClass::Logical
        } else {
            NumericClass::Double
        };
        self
    }

    pub fn range(start: f64, step: f64, stop: f64) -> Self {
        if step == 0.0 || (step > 0.0 && start > stop) || (step < 0.0 && start < stop) {
            return Self::empty();
        }
        let eps = 1e-12 * step.abs().max(1.0);
        let count = (((stop - start) / step) + eps / step.abs()).floor() as isize + 1;
        if count <= 0 {
            return Self::empty();
        }
        let mut data = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            let val = start + (i as f64) * step;
            if (step > 0.0 && val <= stop + eps) || (step < 0.0 && val >= stop - eps) {
                data.push(val);
            }
        }
        Self::row_vec(data)
    }

    #[inline]
    pub fn is_real_non_logical_scalar(&self) -> Option<f64> {
        if self.rows == 1
            && self.cols == 1
            && self.extra_dims.is_none()
            && self.im.is_none()
            && self.dtype == NumericClass::Double
        {
            Some(self.re[0])
        } else {
            None
        }
    }

    #[inline]
    pub fn set_scalar_in_place(&mut self, val: f64) -> bool {
        if self.rows == 1
            && self.cols == 1
            && self.extra_dims.is_none()
            && self.im.is_none()
            && self.dtype == NumericClass::Double
        {
            self.re[0] = val;
            true
        } else {
            false
        }
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.re.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.re.is_empty()
    }

    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.rows == 1 && self.cols == 1 && self.extra_dims.is_none()
    }

    #[inline]
    pub fn is_complex(&self) -> bool {
        self.im.is_some()
    }

    /// Demote to pure real if all imaginary parts are zero (within 1e-15).
    pub fn normalize(mut self) -> Self {
        if let Some(im) = &self.im {
            if im.iter().all(|&x| x.abs() < 1e-15) {
                self.im = None;
            }
        }
        self
    }

    #[inline]
    pub fn as_scalar(&self) -> Option<f64> {
        if self.is_scalar() && !self.is_complex() {
            Some(self.re[0])
        } else if self.is_scalar() && self.get_c(0, 0).im.abs() < 1e-15 {
            Some(self.re[0])
        } else {
            None
        }
    }

    #[inline]
    pub fn as_complex_scalar(&self) -> Option<Complex> {
        if self.is_scalar() {
            Some(self.get_c(0, 0))
        } else {
            None
        }
    }

    #[inline]
    pub fn data(&self) -> &[f64] {
        &self.re
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.re
    }

    #[inline]
    pub fn imag_data(&self) -> Option<&[f64]> {
        self.im.as_deref()
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.re[c * self.rows + r]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        let idx = c * self.rows + r;
        self.re[idx] = val;
        if let Some(im) = &mut self.im {
            im[idx] = 0.0;
        }
    }

    #[inline]
    pub fn get_linear_c(&self, idx: usize) -> Complex {
        let im = self.im.as_ref().map_or(0.0, |v| v[idx]);
        Complex::new(self.re[idx], im)
    }

    #[inline]
    pub fn set_linear_c(&mut self, idx: usize, val: Complex) {
        self.re[idx] = val.re;
        if val.im != 0.0 && self.im.is_none() {
            self.im = Some(vec![0.0; self.re.len()]);
        }
        if let Some(im) = &mut self.im {
            im[idx] = val.im;
        }
    }

    #[inline]
    pub fn get_c(&self, r: usize, c: usize) -> Complex {
        self.get_linear_c(c * self.rows + r)
    }

    #[inline]
    pub fn set_c(&mut self, r: usize, c: usize, val: Complex) {
        self.set_linear_c(c * self.rows + r, val);
    }

    pub fn real_part(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            extra_dims: self.extra_dims.clone(),
            re: self.re.clone(),
            im: None,
            dtype: NumericClass::Double,
        }
    }

    pub fn imag_part(&self) -> Self {
        let data = match &self.im {
            Some(im) => im.clone(),
            None => vec![0.0; self.re.len()],
        };
        Self {
            rows: self.rows,
            cols: self.cols,
            extra_dims: self.extra_dims.clone(),
            re: data,
            im: None,
            dtype: NumericClass::Double,
        }
    }

    pub fn conj(&self) -> Self {
        match &self.im {
            None => self.clone(),
            Some(im) => Self {
                rows: self.rows,
                cols: self.cols,
                extra_dims: self.extra_dims.clone(),
                re: self.re.clone(),
                im: Some(im.iter().map(|&x| -x).collect()),
                dtype: NumericClass::Double,
            },
        }
    }

    pub fn map(&self, f: impl Fn(f64) -> f64) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            extra_dims: self.extra_dims.clone(),
            re: self.re.iter().map(|&x| f(x)).collect(),
            im: None,
            dtype: NumericClass::Double,
        }
    }

    pub fn map_c(&self, f: impl Fn(Complex) -> Complex) -> Self {
        let mut re = Vec::with_capacity(self.len());
        let mut im = Vec::with_capacity(self.len());
        let mut any_im = false;
        for i in 0..self.len() {
            let c = f(self.get_linear_c(i));
            re.push(c.re);
            im.push(c.im);
            if c.im != 0.0 {
                any_im = true;
            }
        }
        Self {
            rows: self.rows,
            cols: self.cols,
            extra_dims: self.extra_dims.clone(),
            re,
            im: if any_im { Some(im) } else { None },
            dtype: NumericClass::Double,
        }
    }

    pub fn map_c_to_real(&self, f: impl Fn(Complex) -> f64) -> Self {
        let re = (0..self.len()).map(|i| f(self.get_linear_c(i))).collect();
        Self {
            rows: self.rows,
            cols: self.cols,
            extra_dims: self.extra_dims.clone(),
            re,
            im: None,
            dtype: NumericClass::Double,
        }
    }

    /// Element-wise binary operation over real values with full N-D singleton broadcasting.
    pub fn elementwise(&self, other: &Self, f: impl Fn(f64, f64) -> f64) -> Result<Self> {
        if self.dims() == other.dims() {
            let re = self
                .re
                .iter()
                .zip(other.re.iter())
                .map(|(&a, &b)| f(a, b))
                .collect();
            return Ok(Self {
                rows: self.rows,
                cols: self.cols,
                extra_dims: self.extra_dims.clone(),
                re,
                im: None,
                dtype: NumericClass::Double,
            });
        }
        if let Some(a) = self.as_scalar() {
            return Ok(other.map(|b| f(a, b)));
        }
        if let Some(b) = other.as_scalar() {
            return Ok(self.map(|a| f(a, b)));
        }
        if self.extra_dims.is_none() && other.extra_dims.is_none() {
            let rows_ok = self.rows == other.rows || self.rows == 1 || other.rows == 1;
            let cols_ok = self.cols == other.cols || self.cols == 1 || other.cols == 1;
            if rows_ok && cols_ok {
                let out_rows = if self.is_empty() || other.is_empty() {
                    0
                } else {
                    self.rows.max(other.rows)
                };
                let out_cols = if self.is_empty() || other.is_empty() {
                    0
                } else {
                    self.cols.max(other.cols)
                };
                let mut out = Self::zeros(out_rows, out_cols);
                for c in 0..out_cols {
                    let c1 = if self.cols == 1 { 0 } else { c };
                    let c2 = if other.cols == 1 { 0 } else { c };
                    for r in 0..out_rows {
                        let r1 = if self.rows == 1 { 0 } else { r };
                        let r2 = if other.rows == 1 { 0 } else { r };
                        out.set(r, c, f(self.get(r1, c1), other.get(r2, c2)));
                    }
                }
                return Ok(out);
            }
            return Err(MatroError::dimension(format!(
                "Incompatible matrix dimensions: {}x{} and {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )));
        }

        // General N-D implicit broadcasting
        let d1 = self.dims();
        let d2 = other.dims();
        let nd = d1.len().max(d2.len());
        let mut out_dims = Vec::with_capacity(nd);
        let mut s1 = vec![1usize; nd];
        let mut s2 = vec![1usize; nd];
        let mut cur1 = 1usize;
        let mut cur2 = 1usize;
        for ax in 0..nd {
            let a = d1.get(ax).copied().unwrap_or(1);
            let b = d2.get(ax).copied().unwrap_or(1);
            if a != b && a != 1 && b != 1 {
                return Err(MatroError::dimension(format!(
                    "Incompatible array dimensions: {d1:?} and {d2:?}"
                )));
            }
            out_dims.push(if a == 0 || b == 0 { 0 } else { a.max(b) });
            s1[ax] = if a == 1 { 0 } else { cur1 };
            s2[ax] = if b == 1 { 0 } else { cur2 };
            cur1 = cur1.saturating_mul(a);
            cur2 = cur2.saturating_mul(b);
        }
        let total: usize = out_dims.iter().product();
        let mut re = Vec::with_capacity(total);
        let mut coord = vec![0usize; nd];
        for _ in 0..total {
            let mut idx1 = 0usize;
            let mut idx2 = 0usize;
            for ax in 0..nd {
                idx1 += coord[ax] * s1[ax];
                idx2 += coord[ax] * s2[ax];
            }
            re.push(f(self.re[idx1], other.re[idx2]));
            for ax in 0..nd {
                coord[ax] += 1;
                if coord[ax] < out_dims[ax] {
                    break;
                }
                coord[ax] = 0;
            }
        }
        Self::from_nd_col_major(out_dims, re)
    }

    /// Complex-aware element-wise binary operation with full N-D & singleton broadcasting,
    /// preserving MATLAB integer saturation and single-precision promotion rules.
    pub fn elementwise_c(&self, other: &Self, f: impl Fn(Complex, Complex) -> Complex) -> Result<Self> {
        let out_dtype = NumericClass::combine(self.dtype, other.dtype)?;
        let finish = |m: Self| -> Self {
            if out_dtype == NumericClass::Double {
                m.normalize()
            } else {
                m.cast_to(out_dtype).normalize()
            }
        };
        if self.dims() == other.dims() {
            let mut re = Vec::with_capacity(self.len());
            let mut im = Vec::with_capacity(self.len());
            let mut any_im = false;
            for i in 0..self.len() {
                let c = f(self.get_linear_c(i), other.get_linear_c(i));
                re.push(c.re);
                im.push(c.im);
                if c.im != 0.0 {
                    any_im = true;
                }
            }
            return Ok(finish(Self {
                rows: self.rows,
                cols: self.cols,
                extra_dims: self.extra_dims.clone(),
                re,
                im: if any_im { Some(im) } else { None },
                dtype: NumericClass::Double,
            }));
        }
        if let Some(a) = self.as_complex_scalar() {
            return Ok(finish(other.map_c(|b| f(a, b))));
        }
        if let Some(b) = other.as_complex_scalar() {
            return Ok(finish(self.map_c(|a| f(a, b))));
        }
        let rows_ok = self.rows == other.rows || self.rows == 1 || other.rows == 1;
        let cols_ok = self.cols == other.cols || self.cols == 1 || other.cols == 1;
        if self.extra_dims.is_none() && other.extra_dims.is_none() {
            if !(rows_ok && cols_ok) {
                return Err(MatroError::dimension(format!(
                    "Incompatible array dimensions: {:?} and {:?}",
                    self.dims(),
                    other.dims()
                )));
            }
            let out_rows = if self.is_empty() || other.is_empty() {
                0
            } else {
                self.rows.max(other.rows)
            };
            let out_cols = if self.is_empty() || other.is_empty() {
                0
            } else {
                self.cols.max(other.cols)
            };
            let mut out = Self::zeros(out_rows, out_cols);
            for c in 0..out_cols {
                let c1 = if self.cols == 1 { 0 } else { c };
                let c2 = if other.cols == 1 { 0 } else { c };
                for r in 0..out_rows {
                    let r1 = if self.rows == 1 { 0 } else { r };
                    let r2 = if other.rows == 1 { 0 } else { r };
                    out.set_c(r, c, f(self.get_c(r1, c1), other.get_c(r2, c2)));
                }
            }
            return Ok(finish(out));
        }

        // General N-D implicit broadcasting (MATLAB R2016b+)
        let d1 = self.dims();
        let d2 = other.dims();
        let nd = d1.len().max(d2.len());
        let mut out_dims = Vec::with_capacity(nd);
        let mut s1 = vec![1usize; nd];
        let mut s2 = vec![1usize; nd];
        let mut cur1 = 1usize;
        let mut cur2 = 1usize;
        for ax in 0..nd {
            let a = d1.get(ax).copied().unwrap_or(1);
            let b = d2.get(ax).copied().unwrap_or(1);
            if a != b && a != 1 && b != 1 {
                return Err(MatroError::dimension(format!(
                    "Incompatible array dimensions: {d1:?} and {d2:?}"
                )));
            }
            out_dims.push(if a == 0 || b == 0 { 0 } else { a.max(b) });
            s1[ax] = if a == 1 { 0 } else { cur1 };
            s2[ax] = if b == 1 { 0 } else { cur2 };
            cur1 = cur1.saturating_mul(a);
            cur2 = cur2.saturating_mul(b);
        }
        let total: usize = out_dims.iter().product();
        let mut re = Vec::with_capacity(total);
        let mut im = Vec::with_capacity(total);
        let mut coord = vec![0usize; nd];
        for _ in 0..total {
            let mut idx1 = 0usize;
            let mut idx2 = 0usize;
            for ax in 0..nd {
                idx1 += coord[ax] * s1[ax];
                idx2 += coord[ax] * s2[ax];
            }
            let c = f(self.get_linear_c(idx1), other.get_linear_c(idx2));
            re.push(c.re);
            im.push(c.im);
            for ax in 0..nd {
                coord[ax] += 1;
                if coord[ax] < out_dims[ax] {
                    break;
                }
                coord[ax] = 0;
            }
        }
        Self::from_nd_complex_col_major(out_dims, re, im).map(finish)
    }

    /// Matrix multiplication (real fast-path or complex).
    pub fn matmul(&self, other: &Self) -> Result<Self> {
        if let Some(a) = self.as_complex_scalar() {
            return Ok(other.map_c(|b| a * b).normalize());
        }
        if let Some(b) = other.as_complex_scalar() {
            return Ok(self.map_c(|a| a * b).normalize());
        }
        if self.cols != other.rows {
            return Err(MatroError::dimension(format!(
                "Inner matrix dimensions must agree for multiplication: {}x{} * {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )));
        }
        let m = self.rows;
        let k = self.cols;
        let n = other.cols;

        if !self.is_complex() && !other.is_complex() {
            let mut out = Self::zeros(m, n);
            let a = &self.re;
            let b = &other.re;
            let c = &mut out.re;
            const BLK: usize = 32;
            let mut j0 = 0;
            while j0 < n {
                let j_end = (j0 + BLK).min(n);
                let mut p0 = 0;
                while p0 < k {
                    let p_end = (p0 + BLK).min(k);
                    let mut i0 = 0;
                    while i0 < m {
                        let i_end = (i0 + BLK).min(m);
                        for j in j0..j_end {
                            let c_col = &mut c[j * m + i0..j * m + i_end];
                            let b_col = &b[j * k..j * k + k];
                            for p in p0..p_end {
                                let b_pj = b_col[p];
                                if b_pj != 0.0 {
                                    let a_col = &a[p * m + i0..p * m + i_end];
                                    for (dst, &src) in c_col.iter_mut().zip(a_col.iter()) {
                                        *dst += src * b_pj;
                                    }
                                }
                            }
                        }
                        i0 += BLK;
                    }
                    p0 += BLK;
                }
                j0 += BLK;
            }
            return Ok(out);
        }

        let mut out = Self::complex_zeros(m, n);
        for j in 0..n {
            for p in 0..k {
                let b_pj = other.get_c(p, j);
                if !b_pj.is_zero() {
                    for i in 0..m {
                        let cur = out.get_c(i, j);
                        out.set_c(i, j, cur + self.get_c(i, p) * b_pj);
                    }
                }
            }
        }
        Ok(out.normalize())
    }

    /// Non-conjugate transpose (`.'`).
    pub fn transpose(&self) -> Self {
        let mut out = if self.is_complex() {
            Self::complex_zeros(self.cols, self.rows)
        } else {
            Self::zeros(self.cols, self.rows)
        };
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set_c(c, r, self.get_c(r, c));
            }
        }
        out
    }

    /// Conjugate (Hermitian) transpose (`'`).
    pub fn ctranspose(&self) -> Self {
        let mut out = if self.is_complex() {
            Self::complex_zeros(self.cols, self.rows)
        } else {
            Self::zeros(self.cols, self.rows)
        };
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set_c(c, r, self.get_c(r, c).conj());
            }
        }
        out
    }

    /// Horizontal concatenation `[A, B, ...]`.
    pub fn hcat(mats: &[&Matrix]) -> Result<Self> {
        let non_empty: Vec<&Matrix> = mats.iter().copied().filter(|m| !m.is_empty()).collect();
        if non_empty.is_empty() {
            return Ok(Self::empty());
        }
        let rows = non_empty[0].rows;
        let mut total_cols = 0;
        let mut any_complex = false;
        for m in &non_empty {
            if m.rows != rows {
                return Err(MatroError::dimension(format!(
                    "Horizontal concatenation row mismatch: expected {rows}, got {}",
                    m.rows
                )));
            }
            total_cols += m.cols;
            any_complex |= m.is_complex();
        }
        let mut re = Vec::with_capacity(rows * total_cols);
        let mut im = if any_complex {
            Some(Vec::with_capacity(rows * total_cols))
        } else {
            None
        };
        for m in &non_empty {
            re.extend_from_slice(&m.re);
            if let Some(im_vec) = &mut im {
                if let Some(m_im) = &m.im {
                    im_vec.extend_from_slice(m_im);
                } else {
                    im_vec.resize(im_vec.len() + m.len(), 0.0);
                }
            }
        }
        let first_dtype = non_empty[0].dtype;
        let out_dtype = if non_empty.iter().all(|m| m.dtype == first_dtype) {
            first_dtype
        } else if let Some(non_dbl) = non_empty
            .iter()
            .map(|m| m.dtype)
            .find(|&d| d != NumericClass::Double && d != NumericClass::Logical)
        {
            non_dbl
        } else {
            NumericClass::Double
        };
        let mut out = Self {
            rows,
            cols: total_cols,
            extra_dims: None,
            re,
            im,
            dtype: NumericClass::Double,
        };
        if out_dtype != NumericClass::Double {
            out = out.cast_to(out_dtype);
        }
        Ok(out)
    }

    /// Vertical concatenation `[A; B; ...]`.
    pub fn vcat(mats: &[&Matrix]) -> Result<Self> {
        let non_empty: Vec<&Matrix> = mats.iter().copied().filter(|m| !m.is_empty()).collect();
        if non_empty.is_empty() {
            return Ok(Self::empty());
        }
        let cols = non_empty[0].cols;
        let mut total_rows = 0;
        let mut any_complex = false;
        for m in &non_empty {
            if m.cols != cols {
                return Err(MatroError::dimension(format!(
                    "Vertical concatenation column mismatch: expected {cols}, got {}",
                    m.cols
                )));
            }
            total_rows += m.rows;
            any_complex |= m.is_complex();
        }
        let mut out = if any_complex {
            Self::complex_zeros(total_rows, cols)
        } else {
            Self::zeros(total_rows, cols)
        };
        let mut row_offset = 0;
        for m in &non_empty {
            for c in 0..cols {
                for r in 0..m.rows {
                    out.set_c(row_offset + r, c, m.get_c(r, c));
                }
            }
            row_offset += m.rows;
        }
        Ok(out)
    }

    /// Solve linear system `A * X = B` (`A \ B` in MATLAB).
    /// Supports square systems (real & complex LU with partial pivoting),
    /// overdetermined real & complex least-squares (QR with SVD fallback for rank-deficient),
    /// and underdetermined real & complex minimum-norm solutions (SVD).
    pub fn solve(&self, rhs: &Self) -> Result<Self> {
        if let Some(a) = self.as_complex_scalar() {
            return Ok(rhs.map_c(|b| b / a).normalize());
        }
        if self.rows != rhs.rows {
            return Err(MatroError::dimension(format!(
                "Row count mismatch in A \\ B: {}x{} \\ {}x{}",
                self.rows, self.cols, rhs.rows, rhs.cols
            )));
        }
        // Overdetermined least squares (rows > cols): real or complex QR, with SVD fallback if rank-deficient
        if self.rows > self.cols {
            let n = self.cols;
            let m = rhs.cols;
            if let Ok((q, r)) = self.qr() {
                let qhb = q.ctranspose().matmul(rhs)?;
                let mut x = if self.is_complex() || rhs.is_complex() {
                    Self::complex_zeros(n, m)
                } else {
                    Self::zeros(n, m)
                };
                let max_diag = (0..n)
                    .map(|k| r.get_c(k, k).abs())
                    .fold(0.0_f64, f64::max);
                let tol = (self.rows.max(n) as f64) * f64::EPSILON * max_diag.max(1.0);
                let mut full_rank = max_diag > 0.0;
                if full_rank {
                    for k in (0..n).rev() {
                        let diag = r.get_c(k, k);
                        if diag.abs() <= tol {
                            full_rank = false;
                            break;
                        }
                        for j in 0..m {
                            let mut sum = qhb.get_c(k, j);
                            for i in (k + 1)..n {
                                sum = sum - r.get_c(k, i) * x.get_c(i, j);
                            }
                            x.set_c(k, j, sum / diag);
                        }
                    }
                }
                if full_rank {
                    return Ok(x.normalize());
                }
            }
            return self.solve_svd_min_norm(rhs);
        }
        // Underdetermined system (rows < cols): minimum-norm least-squares via SVD
        if self.rows < self.cols {
            return self.solve_svd_min_norm(rhs);
        }
        let n = self.rows;
        let m = rhs.cols;
        let mut a = self.clone();
        let mut b = rhs.clone();

        for k in 0..n {
            let mut pivot_row = k;
            let mut max_val = a.get_c(k, k).abs();
            for i in (k + 1)..n {
                let v = a.get_c(i, k).abs();
                if v > max_val {
                    max_val = v;
                    pivot_row = i;
                }
            }
            if max_val < 1e-15 {
                return Err(MatroError::runtime("Matrix is singular or nearly singular"));
            }
            if pivot_row != k {
                for j in k..n {
                    let tmp = a.get_c(k, j);
                    a.set_c(k, j, a.get_c(pivot_row, j));
                    a.set_c(pivot_row, j, tmp);
                }
                for j in 0..m {
                    let tmp = b.get_c(k, j);
                    b.set_c(k, j, b.get_c(pivot_row, j));
                    b.set_c(pivot_row, j, tmp);
                }
            }
            let pivot = a.get_c(k, k);
            for i in (k + 1)..n {
                let factor = a.get_c(i, k) / pivot;
                a.set_c(i, k, Complex::ZERO);
                for j in (k + 1)..n {
                    let v = a.get_c(i, j) - factor * a.get_c(k, j);
                    a.set_c(i, j, v);
                }
                for j in 0..m {
                    let v = b.get_c(i, j) - factor * b.get_c(k, j);
                    b.set_c(i, j, v);
                }
            }
        }

        let mut x = if a.is_complex() || b.is_complex() {
            Self::complex_zeros(n, m)
        } else {
            Self::zeros(n, m)
        };
        for k in (0..n).rev() {
            let diag = a.get_c(k, k);
            for j in 0..m {
                let mut sum = b.get_c(k, j);
                for i in (k + 1)..n {
                    sum = sum - a.get_c(k, i) * x.get_c(i, j);
                }
                x.set_c(k, j, sum / diag);
            }
        }
        Ok(x.normalize())
    }

    /// Minimum-norm least-squares solve `A \ B` via SVD (`V * S^+ * U' * B`).
    fn solve_svd_min_norm(&self, rhs: &Self) -> Result<Self> {
        let (u, s, v) = self.svd()?;
        let k = self.rows.min(self.cols);
        let uhb = u.ctranspose().matmul(rhs)?;
        let max_s = (0..k).map(|i| s.get(i, i).abs()).fold(0.0_f64, f64::max);
        let tol = (self.rows.max(self.cols) as f64) * f64::EPSILON * max_s;
        let m = rhs.cols;
        let mut scaled = if self.is_complex() || rhs.is_complex() || u.is_complex() || v.is_complex() {
            Self::complex_zeros(self.cols, m)
        } else {
            Self::zeros(self.cols, m)
        };
        for i in 0..k {
            let si = s.get(i, i);
            if si > tol {
                for j in 0..m {
                    scaled.set_c(i, j, uhb.get_c(i, j) / Complex::new(si, 0.0));
                }
            }
        }
        v.matmul(&scaled).map(Self::normalize)
    }

    /// Determinant (real or complex) via LU decomposition with partial pivoting.
    pub fn det_c(&self) -> Result<Complex> {
        if self.rows != self.cols {
            return Err(MatroError::dimension(format!(
                "Matrix must be square for det (got {}x{})",
                self.rows, self.cols
            )));
        }
        let n = self.rows;
        if n == 0 {
            return Ok(Complex::ONE);
        }
        let mut a = self.clone();
        let mut sign = 1.0;
        for k in 0..n {
            let mut pivot_row = k;
            let mut max_val = a.get_c(k, k).abs();
            for i in (k + 1)..n {
                let v = a.get_c(i, k).abs();
                if v > max_val {
                    max_val = v;
                    pivot_row = i;
                }
            }
            if max_val < 1e-15 {
                return Ok(Complex::ZERO);
            }
            if pivot_row != k {
                sign = -sign;
                for j in k..n {
                    let tmp = a.get_c(k, j);
                    a.set_c(k, j, a.get_c(pivot_row, j));
                    a.set_c(pivot_row, j, tmp);
                }
            }
            let pivot = a.get_c(k, k);
            for i in (k + 1)..n {
                let factor = a.get_c(i, k) / pivot;
                for j in (k + 1)..n {
                    let v = a.get_c(i, j) - factor * a.get_c(k, j);
                    a.set_c(i, j, v);
                }
            }
        }
        let mut prod = Complex::from_real(sign);
        for i in 0..n {
            prod = prod * a.get_c(i, i);
        }
        Ok(prod)
    }

    pub fn det(&self) -> Result<f64> {
        Ok(self.det_c()?.re)
    }

    /// Matrix inverse via solving `A * X = I`.
    pub fn inv(&self) -> Result<Self> {
        if self.rows != self.cols {
            return Err(MatroError::dimension(format!(
                "Matrix must be square for inv (got {}x{})",
                self.rows, self.cols
            )));
        }
        let eye = Self::eye(self.rows, self.cols);
        self.solve(&eye)
    }

    /// Matrix power `A ^ p`.
    pub fn matpow(&self, other: &Self) -> Result<Self> {
        if let (Some(a), Some(b)) = (self.as_complex_scalar(), other.as_complex_scalar()) {
            return Ok(Self::complex_scalar(a.powc(b)).normalize());
        }
        if let Some(exp) = other.as_scalar() {
            if self.rows != self.cols {
                return Err(MatroError::dimension(
                    "Matrix must be square for matrix power (^); use .^ for element-wise power",
                ));
            }
            let k = exp.round() as i64;
            if (exp - k as f64).abs() > 1e-12 {
                return Err(MatroError::runtime(
                    "Non-integer matrix powers are not yet supported; use .^ for element-wise power",
                ));
            }
            if k == 0 {
                return Ok(Self::eye(self.rows, self.cols));
            }
            let (mut base, mut n) = if k < 0 {
                (self.inv()?, (-k) as u64)
            } else {
                (self.clone(), k as u64)
            };
            let mut acc = Self::eye(self.rows, self.cols);
            while n > 0 {
                if n & 1 == 1 {
                    acc = acc.matmul(&base)?;
                }
                n >>= 1;
                if n > 0 {
                    base = base.matmul(&base)?;
                }
            }
            return Ok(acc);
        }
        Err(MatroError::dimension(
            "Invalid operands for matrix power (^); use .^ for element-wise power",
        ))
    }

    /// Right matrix division `A / B`.
    pub fn rdivide(&self, other: &Self) -> Result<Self> {
        if let Some(b) = other.as_complex_scalar() {
            return Ok(self.map_c(|a| a / b).normalize());
        }
        let sol = other.ctranspose().solve(&self.ctranspose())?;
        Ok(sol.ctranspose())
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return write!(f, "[]");
        }
        if let Some(c) = self.as_complex_scalar() {
            return write!(f, "{}", format_complex(c));
        }
        // Display 3D+ arrays page by page, matching MATLAB's `(:,:,k) =` format
        if let Some(ext) = &self.extra_dims {
            let page_size = self.rows * self.cols;
            let total_pages: usize = ext.iter().product();
            for p in 0..total_pages {
                if p > 0 {
                    writeln!(f)?;
                }
                let mut rem = p;
                let mut coord = Vec::with_capacity(ext.len());
                for &d in ext {
                    coord.push((rem % d) + 1);
                    rem /= d;
                }
                let coord_str: Vec<String> = coord.iter().map(ToString::to_string).collect();
                writeln!(f, "(:,:,{}) =", coord_str.join(","))?;
                let start = p * page_size;
                let page_re = self.re[start..start + page_size].to_vec();
                let page = if let Some(im) = &self.im {
                    let page_im = im[start..start + page_size].to_vec();
                    Matrix::from_complex_col_major(self.rows, self.cols, page_re, page_im).unwrap()
                } else {
                    Matrix::from_col_major(self.rows, self.cols, page_re).unwrap()
                };
                write!(f, "{page}")?;
            }
            return Ok(());
        }
        let is_c = self.is_complex();
        let mut col_widths = vec![0usize; self.cols];
        let mut cells = vec![vec![String::new(); self.cols]; self.rows];
        for r in 0..self.rows {
            for c in 0..self.cols {
                let s = if is_c {
                    format_complex(self.get_c(r, c))
                } else {
                    format_real(self.get(r, c))
                };
                col_widths[c] = col_widths[c].max(s.len());
                cells[r][c] = s;
            }
        }
        for r in 0..self.rows {
            if r > 0 {
                writeln!(f)?;
            }
            for c in 0..self.cols {
                write!(f, "  {:>width$}", cells[r][c], width = col_widths[c])?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_and_solve() {
        let a = Matrix::from_row_major(2, 2, &[3.0, 1.0, 1.0, 2.0]).unwrap();
        let b = Matrix::from_row_major(2, 1, &[9.0, 8.0]).unwrap();
        let x = a.solve(&b).unwrap();
        assert!((x.get(0, 0) - 2.0).abs() < 1e-10);
        assert!((x.get(1, 0) - 3.0).abs() < 1e-10);
        assert!((a.det().unwrap() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn complex_arithmetic_and_ctranspose() {
        let z = Matrix::complex_scalar(Complex::new(3.0, 4.0));
        assert_eq!(z.ctranspose().get_c(0, 0), Complex::new(3.0, -4.0));
        assert_eq!(z.transpose().get_c(0, 0), Complex::new(3.0, 4.0));
    }
}
