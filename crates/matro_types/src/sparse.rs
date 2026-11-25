//! Compressed Sparse Column (CSC) matrix with SpMV, SpGEMM, and CG / BiCGSTAB iterative solvers.

use std::collections::BTreeMap;
use std::fmt;

use crate::{Complex, MatroError, Matrix, Result};

/// Compressed Sparse Column (CSC) matrix of real or complex `f64` values (matching MATLAB's sparse storage).
#[derive(Debug, Clone, PartialEq)]
pub struct SparseMatrix {
    rows: usize,
    cols: usize,
    /// Length `cols + 1`: column `c` entries live at `col_ptr[c]..col_ptr[c + 1]`.
    col_ptr: Vec<usize>,
    /// Row index for each non-zero entry (sorted ascending within each column).
    row_idx: Vec<usize>,
    /// Real part of non-zero numeric values.
    values: Vec<f64>,
    /// Optional imaginary part of non-zero values (same length as `values` when `Some`).
    imag: Option<Vec<f64>>,
}

impl SparseMatrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            col_ptr: vec![0; cols + 1],
            row_idx: Vec::new(),
            values: Vec::new(),
            imag: None,
        }
    }

    pub fn eye(n: usize) -> Self {
        let col_ptr: Vec<usize> = (0..=n).collect();
        let row_idx: Vec<usize> = (0..n).collect();
        let values = vec![1.0; n];
        Self {
            rows: n,
            cols: n,
            col_ptr,
            row_idx,
            values,
            imag: None,
        }
    }

    #[inline]
    pub fn is_complex(&self) -> bool {
        self.imag.is_some()
    }

    #[inline]
    fn val_c(&self, idx: usize) -> Complex {
        Complex::new(
            self.values[idx],
            self.imag.as_ref().map_or(0.0, |im| im[idx]),
        )
    }

    /// Construct from 0-based `(row, col, Complex)` triplets, summing duplicate coordinates.
    pub fn from_complex_triplets(
        rows: usize,
        cols: usize,
        triplets: &[(usize, usize, Complex)],
    ) -> Result<Self> {
        let mut entries: Vec<(usize, usize, Complex)> = Vec::with_capacity(triplets.len());
        for &(r, c, v) in triplets {
            if r >= rows || c >= cols {
                return Err(MatroError::index(format!(
                    "Sparse triplet ({}, {}) out of bounds for {rows}x{cols}",
                    r + 1,
                    c + 1
                )));
            }
            if v.re != 0.0 || v.im != 0.0 {
                entries.push((c, r, v));
            }
        }
        entries.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        let mut col_ptr = vec![0usize; cols + 1];
        let mut row_idx = Vec::with_capacity(entries.len());
        let mut values = Vec::with_capacity(entries.len());
        let mut im_vals = Vec::with_capacity(entries.len());
        let mut has_imag = false;

        let mut cur_col = 0usize;
        let mut idx = 0usize;
        while idx < entries.len() {
            let (c, r, mut sum) = entries[idx];
            idx += 1;
            while idx < entries.len() && entries[idx].0 == c && entries[idx].1 == r {
                sum = sum + entries[idx].2;
                idx += 1;
            }
            if sum.re != 0.0 || sum.im != 0.0 {
                while cur_col < c {
                    cur_col += 1;
                    col_ptr[cur_col] = row_idx.len();
                }
                row_idx.push(r);
                values.push(sum.re);
                im_vals.push(sum.im);
                if sum.im != 0.0 {
                    has_imag = true;
                }
            }
        }
        while cur_col < cols {
            cur_col += 1;
            col_ptr[cur_col] = row_idx.len();
        }
        Ok(Self {
            rows,
            cols,
            col_ptr,
            row_idx,
            values,
            imag: if has_imag { Some(im_vals) } else { None },
        })
    }

    /// Construct from 0-based `(row, col, val)` triplets, summing duplicate coordinates (MATLAB convention).
    pub fn from_triplets(rows: usize, cols: usize, triplets: &[(usize, usize, f64)]) -> Result<Self> {
        let mut entries: Vec<(usize, usize, f64)> = Vec::with_capacity(triplets.len());
        for &(r, c, v) in triplets {
            if r >= rows || c >= cols {
                return Err(MatroError::index(format!(
                    "Sparse triplet ({}, {}) out of bounds for {rows}x{cols}",
                    r + 1,
                    c + 1
                )));
            }
            if v != 0.0 {
                entries.push((c, r, v));
            }
        }
        entries.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        let mut col_ptr = vec![0usize; cols + 1];
        let mut row_idx = Vec::with_capacity(entries.len());
        let mut values = Vec::with_capacity(entries.len());

        let mut cur_col = 0usize;
        let mut idx = 0usize;
        while idx < entries.len() {
            let (c, r, mut sum) = entries[idx];
            idx += 1;
            while idx < entries.len() && entries[idx].0 == c && entries[idx].1 == r {
                sum += entries[idx].2;
                idx += 1;
            }
            if sum != 0.0 {
                while cur_col < c {
                    cur_col += 1;
                    col_ptr[cur_col] = row_idx.len();
                }
                row_idx.push(r);
                values.push(sum);
            }
        }
        while cur_col < cols {
            cur_col += 1;
            col_ptr[cur_col] = row_idx.len();
        }
        Ok(Self {
            rows,
            cols,
            col_ptr,
            row_idx,
            values,
            imag: None,
        })
    }

    pub fn from_dense(m: &Matrix) -> Self {
        let mut col_ptr = Vec::with_capacity(m.cols() + 1);
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        let mut im_vals = if m.is_complex() { Some(Vec::new()) } else { None };
        let mut has_imag = false;
        col_ptr.push(0);
        for c in 0..m.cols() {
            for r in 0..m.rows() {
                let v = m.get_c(r, c);
                if v.re != 0.0 || v.im != 0.0 {
                    row_idx.push(r);
                    values.push(v.re);
                    if let Some(im) = im_vals.as_mut() {
                        im.push(v.im);
                        if v.im != 0.0 {
                            has_imag = true;
                        }
                    }
                }
            }
            col_ptr.push(row_idx.len());
        }
        Self {
            rows: m.rows(),
            cols: m.cols(),
            col_ptr,
            row_idx,
            values,
            imag: if has_imag { im_vals } else { None },
        }
    }

    pub fn to_dense(&self) -> Matrix {
        if self.is_complex() {
            let mut out = Matrix::complex_zeros(self.rows, self.cols);
            for c in 0..self.cols {
                for idx in self.col_ptr[c]..self.col_ptr[c + 1] {
                    out.set_c(self.row_idx[idx], c, self.val_c(idx));
                }
            }
            return out.normalize();
        }
        let mut out = Matrix::zeros(self.rows, self.cols);
        for c in 0..self.cols {
            for idx in self.col_ptr[c]..self.col_ptr[c + 1] {
                out.set(self.row_idx[idx], c, self.values[idx]);
            }
        }
        out
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
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Return all stored non-zero entries as 0-based `(row, col, value)` triplets in column-major order.
    pub fn triplets(&self) -> Vec<(usize, usize, f64)> {
        let mut out = Vec::with_capacity(self.values.len());
        for c in 0..self.cols {
            for idx in self.col_ptr[c]..self.col_ptr[c + 1] {
                out.push((self.row_idx[idx], c, self.values[idx]));
            }
        }
        out
    }

    /// Return all stored non-zero entries as 0-based `(row, col, Complex)` triplets in column-major order.
    pub fn triplets_c(&self) -> Vec<(usize, usize, Complex)> {
        let mut out = Vec::with_capacity(self.values.len());
        for c in 0..self.cols {
            for idx in self.col_ptr[c]..self.col_ptr[c + 1] {
                out.push((self.row_idx[idx], c, self.val_c(idx)));
            }
        }
        out
    }

    /// Column vector of all non-zero values in column-major order.
    pub fn nonzeros(&self) -> Vec<f64> {
        self.values.clone()
    }

    /// Column vector of all complex non-zero values in column-major order.
    pub fn nonzeros_c(&self) -> Vec<Complex> {
        (0..self.values.len()).map(|i| self.val_c(i)).collect()
    }

    pub fn real_part(&self) -> Self {
        let trips: Vec<(usize, usize, f64)> = self
            .triplets_c()
            .into_iter()
            .filter(|&(_, _, v)| v.re != 0.0)
            .map(|(r, c, v)| (r, c, v.re))
            .collect();
        Self::from_triplets(self.rows, self.cols, &trips).unwrap()
    }

    pub fn imag_part(&self) -> Self {
        if self.imag.is_none() {
            return Self::zeros(self.rows, self.cols);
        }
        let trips: Vec<(usize, usize, f64)> = self
            .triplets_c()
            .into_iter()
            .filter(|&(_, _, v)| v.im != 0.0)
            .map(|(r, c, v)| (r, c, v.im))
            .collect();
        Self::from_triplets(self.rows, self.cols, &trips).unwrap()
    }

    pub fn conj(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            col_ptr: self.col_ptr.clone(),
            row_idx: self.row_idx.clone(),
            values: self.values.clone(),
            imag: self.imag.as_ref().map(|im| im.iter().map(|&v| -v).collect()),
        }
    }

    /// Return sparse matrix with same sparsity pattern and all non-zeros replaced by `1.0`.
    pub fn spones(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            col_ptr: self.col_ptr.clone(),
            row_idx: self.row_idx.clone(),
            values: vec![1.0; self.values.len()],
            imag: None,
        }
    }

    pub fn get(&self, r: usize, c: usize) -> f64 {
        let start = self.col_ptr[c];
        let end = self.col_ptr[c + 1];
        match self.row_idx[start..end].binary_search(&r) {
            Ok(offset) => self.values[start + offset],
            Err(_) => 0.0,
        }
    }

    pub fn get_c(&self, r: usize, c: usize) -> Complex {
        let start = self.col_ptr[c];
        let end = self.col_ptr[c + 1];
        match self.row_idx[start..end].binary_search(&r) {
            Ok(offset) => self.val_c(start + offset),
            Err(_) => Complex::ZERO,
        }
    }

    /// Set entry `(r, c)` in-place (auto-growing dimensions if needed, removing explicit zeros).
    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        self.set_c(r, c, Complex::from_real(val));
    }

    /// Set complex entry `(r, c)` in-place.
    pub fn set_c(&mut self, r: usize, c: usize, val: Complex) {
        if r >= self.rows {
            self.rows = r + 1;
        }
        if c >= self.cols {
            let last = *self.col_ptr.last().unwrap_or(&0);
            self.col_ptr.resize(c + 2, last);
            self.cols = c + 1;
        }
        if val.im != 0.0 && self.imag.is_none() {
            self.imag = Some(vec![0.0; self.values.len()]);
        }
        let start = self.col_ptr[c];
        let end = self.col_ptr[c + 1];
        let is_zero = val.re == 0.0 && val.im == 0.0;
        match self.row_idx[start..end].binary_search(&r) {
            Ok(offset) => {
                let pos = start + offset;
                if is_zero {
                    self.row_idx.remove(pos);
                    self.values.remove(pos);
                    if let Some(im) = self.imag.as_mut() {
                        im.remove(pos);
                    }
                    for p in &mut self.col_ptr[(c + 1)..] {
                        *p -= 1;
                    }
                } else {
                    self.values[pos] = val.re;
                    if let Some(im) = self.imag.as_mut() {
                        im[pos] = val.im;
                    }
                }
            }
            Err(offset) => {
                if !is_zero {
                    let pos = start + offset;
                    self.row_idx.insert(pos, r);
                    self.values.insert(pos, val.re);
                    if let Some(im) = self.imag.as_mut() {
                        im.insert(pos, val.im);
                    }
                    for p in &mut self.col_ptr[(c + 1)..] {
                        *p += 1;
                    }
                }
            }
        }
    }

    /// Extract sparse submatrix `S(rows, cols)` in `O(nnz)` time, preserving sparsity.
    pub fn submatrix(&self, rows: &[usize], cols: &[usize]) -> Self {
        // Map each original row index -> list of output row indices (supports duplicate row selection)
        let mut row_counts = vec![0usize; self.rows];
        let mut has_dup_rows = false;
        let mut row_single = vec![usize::MAX; self.rows];
        for (out_r, &orig_r) in rows.iter().enumerate() {
            if orig_r < self.rows {
                row_counts[orig_r] += 1;
                if row_counts[orig_r] == 1 {
                    row_single[orig_r] = out_r;
                } else {
                    has_dup_rows = true;
                }
            }
        }

        let mut row_multi: Vec<Vec<usize>> = Vec::new();
        if has_dup_rows {
            row_multi.resize(self.rows, Vec::new());
            for (out_r, &orig_r) in rows.iter().enumerate() {
                if orig_r < self.rows {
                    row_multi[orig_r].push(out_r);
                }
            }
        }

        let mut col_ptr = Vec::with_capacity(cols.len() + 1);
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        let mut im_vals = if self.is_complex() { Some(Vec::new()) } else { None };
        let mut has_imag = false;
        col_ptr.push(0);

        let mut col_entries: Vec<(usize, Complex)> = Vec::new();
        for &orig_c in cols {
            col_entries.clear();
            if orig_c < self.cols {
                for idx in self.col_ptr[orig_c]..self.col_ptr[orig_c + 1] {
                    let orig_r = self.row_idx[idx];
                    let v = self.val_c(idx);
                    if has_dup_rows {
                        for &out_r in &row_multi[orig_r] {
                            col_entries.push((out_r, v));
                        }
                    } else {
                        let out_r = row_single[orig_r];
                        if out_r != usize::MAX {
                            col_entries.push((out_r, v));
                        }
                    }
                }
                col_entries.sort_unstable_by_key(|&(r, _)| r);
                for &(r, v) in &col_entries {
                    row_idx.push(r);
                    values.push(v.re);
                    if let Some(im) = im_vals.as_mut() {
                        im.push(v.im);
                        if v.im != 0.0 {
                            has_imag = true;
                        }
                    }
                }
            }
            col_ptr.push(row_idx.len());
        }

        Self {
            rows: rows.len(),
            cols: cols.len(),
            col_ptr,
            row_idx,
            values,
            imag: if has_imag { im_vals } else { None },
        }
    }

    /// `O(nnz)` column or row sum (`dim == 1` -> `1 x cols`, `dim == 2` -> `rows x 1`).
    pub fn sum_dim(&self, dim: usize) -> Matrix {
        if self.is_complex() {
            if dim == 1 {
                let mut re = vec![0.0; self.cols];
                let mut im = vec![0.0; self.cols];
                for c in 0..self.cols {
                    let mut acc = Complex::ZERO;
                    for idx in self.col_ptr[c]..self.col_ptr[c + 1] {
                        acc = acc + self.val_c(idx);
                    }
                    re[c] = acc.re;
                    im[c] = acc.im;
                }
                return Matrix::from_complex_col_major(1, self.cols, re, im)
                    .unwrap()
                    .normalize();
            } else {
                let mut re = vec![0.0; self.rows];
                let mut im = vec![0.0; self.rows];
                for idx in 0..self.values.len() {
                    let r = self.row_idx[idx];
                    let v = self.val_c(idx);
                    re[r] += v.re;
                    im[r] += v.im;
                }
                return Matrix::from_complex_col_major(self.rows, 1, re, im)
                    .unwrap()
                    .normalize();
            }
        }
        if dim == 1 {
            let mut out = vec![0.0; self.cols];
            for (c, slot) in out.iter_mut().enumerate() {
                let mut acc = 0.0;
                for idx in self.col_ptr[c]..self.col_ptr[c + 1] {
                    acc += self.values[idx];
                }
                *slot = acc;
            }
            Matrix::row_vec(out)
        } else {
            let mut out = vec![0.0; self.rows];
            for idx in 0..self.values.len() {
                out[self.row_idx[idx]] += self.values[idx];
            }
            Matrix::col_vec(out)
        }
    }

    /// Sparse Kronecker tensor product `kron(self, other)`.
    pub fn kron(&self, other: &Self) -> Self {
        let out_rows = self.rows * other.rows;
        let out_cols = self.cols * other.cols;
        let complex = self.is_complex() || other.is_complex();
        let mut col_ptr = Vec::with_capacity(out_cols + 1);
        let mut row_idx = Vec::with_capacity(self.nnz().saturating_mul(other.nnz()));
        let mut values = Vec::with_capacity(self.nnz().saturating_mul(other.nnz()));
        let mut im_vals = if complex { Some(Vec::with_capacity(self.nnz().saturating_mul(other.nnz()))) } else { None };
        let mut has_imag = false;
        col_ptr.push(0);
        for ca in 0..self.cols {
            for cb in 0..other.cols {
                for ia in self.col_ptr[ca]..self.col_ptr[ca + 1] {
                    let ra = self.row_idx[ia];
                    let va = self.val_c(ia);
                    let r_off = ra * other.rows;
                    for ib in other.col_ptr[cb]..other.col_ptr[cb + 1] {
                        let v = va * other.val_c(ib);
                        if v.re != 0.0 || v.im != 0.0 {
                            row_idx.push(r_off + other.row_idx[ib]);
                            values.push(v.re);
                            if let Some(im) = im_vals.as_mut() {
                                im.push(v.im);
                                if v.im != 0.0 {
                                    has_imag = true;
                                }
                            }
                        }
                    }
                }
                col_ptr.push(row_idx.len());
            }
        }
        Self {
            rows: out_rows,
            cols: out_cols,
            col_ptr,
            row_idx,
            values,
            imag: if has_imag { im_vals } else { None },
        }
    }

    pub fn scale(&self, alpha: f64) -> Self {
        self.scale_c(Complex::from_real(alpha))
    }

    pub fn scale_c(&self, alpha: Complex) -> Self {
        if alpha.re == 0.0 && alpha.im == 0.0 {
            return Self::zeros(self.rows, self.cols);
        }
        if !self.is_complex() && alpha.im == 0.0 {
            return Self {
                rows: self.rows,
                cols: self.cols,
                col_ptr: self.col_ptr.clone(),
                row_idx: self.row_idx.clone(),
                values: self.values.iter().map(|&v| v * alpha.re).collect(),
                imag: None,
            };
        }
        let mut values = Vec::with_capacity(self.values.len());
        let mut im_vals = Vec::with_capacity(self.values.len());
        let mut has_imag = false;
        for idx in 0..self.values.len() {
            let v = self.val_c(idx) * alpha;
            values.push(v.re);
            im_vals.push(v.im);
            if v.im != 0.0 {
                has_imag = true;
            }
        }
        Self {
            rows: self.rows,
            cols: self.cols,
            col_ptr: self.col_ptr.clone(),
            row_idx: self.row_idx.clone(),
            values,
            imag: if has_imag { Some(im_vals) } else { None },
        }
    }

    pub fn transpose(&self) -> Self {
        if self.is_complex() {
            let mut triplets = Vec::with_capacity(self.nnz());
            for c in 0..self.cols {
                for idx in self.col_ptr[c]..self.col_ptr[c + 1] {
                    triplets.push((c, self.row_idx[idx], self.val_c(idx)));
                }
            }
            return Self::from_complex_triplets(self.cols, self.rows, &triplets).unwrap();
        }
        let mut triplets = Vec::with_capacity(self.nnz());
        for c in 0..self.cols {
            for idx in self.col_ptr[c]..self.col_ptr[c + 1] {
                triplets.push((c, self.row_idx[idx], self.values[idx]));
            }
        }
        Self::from_triplets(self.cols, self.rows, &triplets).unwrap()
    }

    pub fn ctranspose(&self) -> Self {
        if !self.is_complex() {
            return self.transpose();
        }
        let mut triplets = Vec::with_capacity(self.nnz());
        for c in 0..self.cols {
            for idx in self.col_ptr[c]..self.col_ptr[c + 1] {
                triplets.push((c, self.row_idx[idx], self.val_c(idx).conj()));
            }
        }
        Self::from_complex_triplets(self.cols, self.rows, &triplets).unwrap()
    }

    /// Sparse addition / subtraction `self + beta * other`.
    pub fn add_scaled(&self, other: &Self, beta: f64) -> Result<Self> {
        self.add_scaled_c(other, Complex::from_real(beta))
    }

    /// Complex sparse addition / subtraction `self + beta * other`.
    pub fn add_scaled_c(&self, other: &Self, beta: Complex) -> Result<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MatroError::dimension(format!(
                "Sparse dimension mismatch: {}x{} vs {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )));
        }
        if !self.is_complex() && !other.is_complex() && beta.im == 0.0 {
            let mut col_ptr = Vec::with_capacity(self.cols + 1);
            let mut row_idx = Vec::new();
            let mut values = Vec::new();
            col_ptr.push(0);
            for c in 0..self.cols {
                let mut acc = BTreeMap::new();
                for idx in self.col_ptr[c]..self.col_ptr[c + 1] {
                    *acc.entry(self.row_idx[idx]).or_insert(0.0) += self.values[idx];
                }
                for idx in other.col_ptr[c]..other.col_ptr[c + 1] {
                    *acc.entry(other.row_idx[idx]).or_insert(0.0) += beta.re * other.values[idx];
                }
                for (r, v) in acc {
                    if v.abs() > 0.0 {
                        row_idx.push(r);
                        values.push(v);
                    }
                }
                col_ptr.push(row_idx.len());
            }
            return Ok(Self {
                rows: self.rows,
                cols: self.cols,
                col_ptr,
                row_idx,
                values,
                imag: None,
            });
        }
        let mut col_ptr = Vec::with_capacity(self.cols + 1);
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        let mut im_vals = Vec::new();
        let mut has_imag = false;
        col_ptr.push(0);
        for c in 0..self.cols {
            let mut acc: BTreeMap<usize, Complex> = BTreeMap::new();
            for idx in self.col_ptr[c]..self.col_ptr[c + 1] {
                let e = acc.entry(self.row_idx[idx]).or_insert(Complex::ZERO);
                *e = *e + self.val_c(idx);
            }
            for idx in other.col_ptr[c]..other.col_ptr[c + 1] {
                let e = acc.entry(other.row_idx[idx]).or_insert(Complex::ZERO);
                *e = *e + beta * other.val_c(idx);
            }
            for (r, v) in acc {
                if v.re != 0.0 || v.im != 0.0 {
                    row_idx.push(r);
                    values.push(v.re);
                    im_vals.push(v.im);
                    if v.im != 0.0 {
                        has_imag = true;
                    }
                }
            }
            col_ptr.push(row_idx.len());
        }
        Ok(Self {
            rows: self.rows,
            cols: self.cols,
            col_ptr,
            row_idx,
            values,
            imag: if has_imag { Some(im_vals) } else { None },
        })
    }

    /// Element-wise sparse product `self .* other`.
    pub fn elem_mul(&self, other: &Self) -> Result<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MatroError::dimension(format!(
                "Sparse dimension mismatch: {}x{} vs {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )));
        }
        if self.is_complex() || other.is_complex() {
            let mut triplets = Vec::new();
            for c in 0..self.cols {
                for idx in self.col_ptr[c]..self.col_ptr[c + 1] {
                    let r = self.row_idx[idx];
                    let ov = other.get_c(r, c);
                    if ov.re != 0.0 || ov.im != 0.0 {
                        triplets.push((r, c, self.val_c(idx) * ov));
                    }
                }
            }
            return Self::from_complex_triplets(self.rows, self.cols, &triplets);
        }
        let mut triplets = Vec::new();
        for c in 0..self.cols {
            for idx in self.col_ptr[c]..self.col_ptr[c + 1] {
                let r = self.row_idx[idx];
                let ov = other.get(r, c);
                if ov != 0.0 {
                    triplets.push((r, c, self.values[idx] * ov));
                }
            }
        }
        Self::from_triplets(self.rows, self.cols, &triplets)
    }

    /// Sparse matrix * dense real vector (`y = A * x`).
    pub fn matvec(&self, x: &[f64]) -> Result<Vec<f64>> {
        if self.cols != x.len() {
            return Err(MatroError::dimension(format!(
                "SpMV dimension mismatch: {}x{} * {}",
                self.rows,
                self.cols,
                x.len()
            )));
        }
        let mut y = vec![0.0; self.rows];
        for (c, &xc) in x.iter().enumerate() {
            if xc != 0.0 {
                for idx in self.col_ptr[c]..self.col_ptr[c + 1] {
                    y[self.row_idx[idx]] += self.values[idx] * xc;
                }
            }
        }
        Ok(y)
    }

    /// Sparse matrix * dense complex vector (`y = A * x`).
    pub fn matvec_c(&self, x: &[Complex]) -> Result<Vec<Complex>> {
        if self.cols != x.len() {
            return Err(MatroError::dimension(format!(
                "SpMV dimension mismatch: {}x{} * {}",
                self.rows,
                self.cols,
                x.len()
            )));
        }
        let mut y = vec![Complex::ZERO; self.rows];
        for (c, &xc) in x.iter().enumerate() {
            if xc.re != 0.0 || xc.im != 0.0 {
                for idx in self.col_ptr[c]..self.col_ptr[c + 1] {
                    let r = self.row_idx[idx];
                    y[r] = y[r] + self.val_c(idx) * xc;
                }
            }
        }
        Ok(y)
    }

    /// Sparse matrix * dense matrix (`C = A * B`), real or complex `self` and/or `rhs`.
    pub fn matmul_dense(&self, rhs: &Matrix) -> Result<Matrix> {
        if let Some(s) = rhs.as_complex_scalar() {
            let mut out = self.to_dense();
            out = out.map_c(|v| v * s).normalize();
            return Ok(out);
        }
        if self.cols != rhs.rows() {
            return Err(MatroError::dimension(format!(
                "Inner dimensions must agree: {}x{} * {}x{}",
                self.rows,
                self.cols,
                rhs.rows(),
                rhs.cols()
            )));
        }
        if self.is_complex() || rhs.is_complex() {
            let mut out = Matrix::complex_zeros(self.rows, rhs.cols());
            for j in 0..rhs.cols() {
                let col: Vec<Complex> = (0..rhs.rows()).map(|r| rhs.get_c(r, j)).collect();
                let res = self.matvec_c(&col)?;
                for (i, &v) in res.iter().enumerate() {
                    out.set_c(i, j, v);
                }
            }
            return Ok(out.normalize());
        }
        let mut out = Matrix::zeros(self.rows, rhs.cols());
        for j in 0..rhs.cols() {
            let col: Vec<f64> = (0..rhs.rows()).map(|r| rhs.get(r, j)).collect();
            let res = self.matvec(&col)?;
            for (i, &v) in res.iter().enumerate() {
                out.set(i, j, v);
            }
        }
        Ok(out)
    }

    /// Horizontal concatenation of sparse matrices (`[A, B, ...]`).
    pub fn horzcat(mats: &[&Self]) -> Result<Self> {
        if mats.is_empty() {
            return Ok(Self::zeros(0, 0));
        }
        let rows = mats[0].rows;
        let mut total_cols = 0usize;
        let complex = mats.iter().any(|m| m.is_complex());
        for m in mats {
            if m.rows != rows {
                return Err(MatroError::dimension(format!(
                    "Sparse horzcat row mismatch: {} vs {}",
                    rows, m.rows
                )));
            }
            total_cols += m.cols;
        }
        let mut col_ptr = Vec::with_capacity(total_cols + 1);
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        let mut im_vals = if complex { Some(Vec::new()) } else { None };
        col_ptr.push(0);
        for m in mats {
            for c in 0..m.cols {
                for idx in m.col_ptr[c]..m.col_ptr[c + 1] {
                    row_idx.push(m.row_idx[idx]);
                    let v = m.val_c(idx);
                    values.push(v.re);
                    if let Some(im) = im_vals.as_mut() {
                        im.push(v.im);
                    }
                }
                col_ptr.push(row_idx.len());
            }
        }
        Ok(Self {
            rows,
            cols: total_cols,
            col_ptr,
            row_idx,
            values,
            imag: im_vals,
        })
    }

    /// Vertical concatenation of sparse matrices (`[A; B; ...]`).
    pub fn vertcat(mats: &[&Self]) -> Result<Self> {
        if mats.is_empty() {
            return Ok(Self::zeros(0, 0));
        }
        let cols = mats[0].cols;
        let mut total_rows = 0usize;
        let complex = mats.iter().any(|m| m.is_complex());
        for m in mats {
            if m.cols != cols {
                return Err(MatroError::dimension(format!(
                    "Sparse vertcat column mismatch: {} vs {}",
                    cols, m.cols
                )));
            }
            total_rows += m.rows;
        }
        let mut col_ptr = Vec::with_capacity(cols + 1);
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        let mut im_vals = if complex { Some(Vec::new()) } else { None };
        col_ptr.push(0);
        for c in 0..cols {
            let mut row_offset = 0usize;
            for m in mats {
                for idx in m.col_ptr[c]..m.col_ptr[c + 1] {
                    row_idx.push(row_offset + m.row_idx[idx]);
                    let v = m.val_c(idx);
                    values.push(v.re);
                    if let Some(im) = im_vals.as_mut() {
                        im.push(v.im);
                    }
                }
                row_offset += m.rows;
            }
            col_ptr.push(row_idx.len());
        }
        Ok(Self {
            rows: total_rows,
            cols,
            col_ptr,
            row_idx,
            values,
            imag: im_vals,
        })
    }

    /// Sparse-sparse matrix multiplication (`C = A * B`), real or complex.
    pub fn matmul_sparse(&self, other: &Self) -> Result<Self> {
        if self.cols != other.rows {
            return Err(MatroError::dimension(format!(
                "Inner dimensions must agree: {}x{} * {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )));
        }
        if self.is_complex() || other.is_complex() {
            let mut col_ptr = Vec::with_capacity(other.cols + 1);
            let mut row_idx = Vec::new();
            let mut values = Vec::new();
            let mut im_vals = Vec::new();
            let mut has_imag = false;
            col_ptr.push(0);
            for j in 0..other.cols {
                let mut acc: BTreeMap<usize, Complex> = BTreeMap::new();
                for b_idx in other.col_ptr[j]..other.col_ptr[j + 1] {
                    let k = other.row_idx[b_idx];
                    let b_kj = other.val_c(b_idx);
                    for a_idx in self.col_ptr[k]..self.col_ptr[k + 1] {
                        let e = acc.entry(self.row_idx[a_idx]).or_insert(Complex::ZERO);
                        *e = *e + self.val_c(a_idx) * b_kj;
                    }
                }
                for (r, v) in acc {
                    if v.re != 0.0 || v.im != 0.0 {
                        row_idx.push(r);
                        values.push(v.re);
                        im_vals.push(v.im);
                        if v.im != 0.0 {
                            has_imag = true;
                        }
                    }
                }
                col_ptr.push(row_idx.len());
            }
            return Ok(Self {
                rows: self.rows,
                cols: other.cols,
                col_ptr,
                row_idx,
                values,
                imag: if has_imag { Some(im_vals) } else { None },
            });
        }
        let mut col_ptr = Vec::with_capacity(other.cols + 1);
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        col_ptr.push(0);
        for j in 0..other.cols {
            let mut acc = BTreeMap::new();
            for b_idx in other.col_ptr[j]..other.col_ptr[j + 1] {
                let k = other.row_idx[b_idx];
                let b_kj = other.values[b_idx];
                for a_idx in self.col_ptr[k]..self.col_ptr[k + 1] {
                    *acc.entry(self.row_idx[a_idx]).or_insert(0.0) += self.values[a_idx] * b_kj;
                }
            }
            for (r, v) in acc {
                if v != 0.0 {
                    row_idx.push(r);
                    values.push(v);
                }
            }
            col_ptr.push(row_idx.len());
        }
        Ok(Self {
            rows: self.rows,
            cols: other.cols,
            col_ptr,
            row_idx,
            values,
            imag: None,
        })
    }

    /// Check approximate symmetry (`A == A'`).
    pub fn is_symmetric(&self) -> bool {
        if self.rows != self.cols {
            return false;
        }
        for c in 0..self.cols {
            for idx in self.col_ptr[c]..self.col_ptr[c + 1] {
                let r = self.row_idx[idx];
                if (self.values[idx] - self.get(c, r)).abs() > 1e-12 {
                    return false;
                }
            }
        }
        true
    }

    /// Lower and upper half-bandwidths `(kl, ku)` in `O(nnz)` time.
    pub fn bandwidth(&self) -> (usize, usize) {
        let mut kl = 0usize;
        let mut ku = 0usize;
        for c in 0..self.cols {
            for idx in self.col_ptr[c]..self.col_ptr[c + 1] {
                let r = self.row_idx[idx];
                if r > c {
                    kl = kl.max(r - c);
                } else {
                    ku = ku.max(c - r);
                }
            }
        }
        (kl, ku)
    }

    /// $O(\text{nnz})$ sparse forward substitution for lower-triangular CSC matrices (`ku == 0`).
    pub fn solve_lower_triangular(&self, b: &[f64]) -> Option<Vec<f64>> {
        let n = self.rows;
        if self.cols != n || b.len() != n {
            return None;
        }
        let mut x = b.to_vec();
        for c in 0..n {
            let start = self.col_ptr[c];
            let end = self.col_ptr[c + 1];
            let mut diag = 0.0;
            let mut diag_pos = None;
            for idx in start..end {
                if self.row_idx[idx] == c {
                    diag = self.values[idx];
                    diag_pos = Some(idx);
                    break;
                }
            }
            if diag.abs() < 1e-30 || !diag.is_finite() {
                return None;
            }
            let xc = x[c] / diag;
            x[c] = xc;
            if xc != 0.0 {
                let first_sub = diag_pos.map_or(start, |p| p + 1);
                for idx in first_sub..end {
                    let r = self.row_idx[idx];
                    if r > c {
                        x[r] -= self.values[idx] * xc;
                    }
                }
            }
        }
        Some(x)
    }

    /// $O(\text{nnz})$ sparse back substitution for upper-triangular CSC matrices (`kl == 0`).
    pub fn solve_upper_triangular(&self, b: &[f64]) -> Option<Vec<f64>> {
        let n = self.rows;
        if self.cols != n || b.len() != n {
            return None;
        }
        let mut x = b.to_vec();
        for c in (0..n).rev() {
            let start = self.col_ptr[c];
            let end = self.col_ptr[c + 1];
            let mut diag = 0.0;
            for idx in (start..end).rev() {
                if self.row_idx[idx] == c {
                    diag = self.values[idx];
                    break;
                }
            }
            if diag.abs() < 1e-30 || !diag.is_finite() {
                return None;
            }
            let xc = x[c] / diag;
            x[c] = xc;
            if xc != 0.0 {
                for idx in start..end {
                    let r = self.row_idx[idx];
                    if r < c {
                        x[r] -= self.values[idx] * xc;
                    }
                }
            }
        }
        Some(x)
    }

    /// Zero-fill Incomplete Cholesky `IC(0)` factorization returning lower-triangular sparse `L`
    /// with the same lower-triangular sparsity pattern as `self` (`diagcomp >= 0` shifts diagonal by `diagcomp * diag(A)`).
    pub fn ichol0(&self, diagcomp: f64) -> Result<Self> {
        if self.rows != self.cols {
            return Err(MatroError::dimension("ichol: matrix must be square"));
        }
        let n = self.rows;
        if n == 0 {
            return Ok(Self::zeros(0, 0));
        }
        let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for c in 0..n {
            let mut has_diag = false;
            for idx in self.col_ptr[c]..self.col_ptr[c + 1] {
                let r = self.row_idx[idx];
                if r >= c {
                    let mut v = self.values[idx];
                    if r == c {
                        has_diag = true;
                        if diagcomp > 0.0 {
                            v += diagcomp * v.abs().max(1.0);
                        }
                    }
                    cols[c].push((r, v));
                }
            }
            if !has_diag {
                return Err(MatroError::runtime(format!(
                    "ichol: missing positive diagonal at row {}",
                    c + 1
                )));
            }
        }
        for j in 0..n {
            let diag = cols[j][0].1;
            if !diag.is_finite() || diag <= 0.0 {
                return Err(MatroError::runtime(format!(
                    "ichol: encountered nonpositive pivot ({diag:.3e}) at column {}",
                    j + 1
                )));
            }
            let ljj = diag.sqrt();
            cols[j][0].1 = ljj;
            let inv_ljj = 1.0 / ljj;
            for entry in cols[j].iter_mut().skip(1) {
                entry.1 *= inv_ljj;
            }
            let col_j: Vec<(usize, f64)> = cols[j][1..].to_vec();
            for (idx_k, &(k, lkj)) in col_j.iter().enumerate() {
                if lkj == 0.0 {
                    continue;
                }
                // Update cols[k] at matching rows i >= k in col_j[idx_k..]
                let sub_j = &col_j[idx_k..];
                let col_k = &mut cols[k];
                let mut pj = 0usize;
                let mut pk = 0usize;
                while pj < sub_j.len() && pk < col_k.len() {
                    let rj = sub_j[pj].0;
                    let rk = col_k[pk].0;
                    if rj == rk {
                        col_k[pk].1 -= sub_j[pj].1 * lkj;
                        pj += 1;
                        pk += 1;
                    } else if rj < rk {
                        pj += 1;
                    } else {
                        pk += 1;
                    }
                }
            }
        }
        let mut col_ptr = Vec::with_capacity(n + 1);
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        col_ptr.push(0);
        for col in cols {
            for (r, v) in col {
                if v != 0.0 {
                    row_idx.push(r);
                    values.push(v);
                }
            }
            col_ptr.push(row_idx.len());
        }
        Ok(Self {
            rows: n,
            cols: n,
            col_ptr,
            row_idx,
            values,
            imag: None,
        })
    }

    /// Zero-fill Incomplete LU `ILU(0)` factorization returning `(L, U)` where `L` is unit lower-triangular
    /// and `U` is upper-triangular with the sparsity pattern of `self`.
    pub fn ilu0(&self) -> Result<(Self, Self)> {
        if self.rows != self.cols {
            return Err(MatroError::dimension("ilu: matrix must be square"));
        }
        let n = self.rows;
        if n == 0 {
            return Ok((Self::zeros(0, 0), Self::zeros(0, 0)));
        }
        // Build CSR rows (sorted by column index)
        let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for c in 0..n {
            for idx in self.col_ptr[c]..self.col_ptr[c + 1] {
                rows[self.row_idx[idx]].push((c, self.values[idx]));
            }
        }
        let mut diag_pos = vec![0usize; n];
        for i in 0..n {
            match rows[i].binary_search_by_key(&i, |&(c, _)| c) {
                Ok(pos) => diag_pos[i] = pos,
                Err(pos) => {
                    rows[i].insert(pos, (i, 0.0));
                    diag_pos[i] = pos;
                }
            }
        }
        // IKJ zero-fill Gaussian elimination
        for i in 0..n {
            let d_i = diag_pos[i];
            for pos_k in 0..d_i {
                let k = rows[i][pos_k].0;
                let u_kk = rows[k][diag_pos[k]].1;
                if u_kk.abs() < 1e-30 || !u_kk.is_finite() {
                    return Err(MatroError::runtime(format!(
                        "ilu: zero pivot encountered at row {}",
                        k + 1
                    )));
                }
                let l_ik = rows[i][pos_k].1 / u_kk;
                rows[i][pos_k].1 = l_ik;
                if l_ik == 0.0 {
                    continue;
                }
                // Merge-update rows[i][pos_k+1..] against rows[k][diag_pos[k]+1..]
                let row_k_tail: Vec<(usize, f64)> = rows[k][(diag_pos[k] + 1)..].to_vec();
                let mut pi = pos_k + 1;
                let mut pk = 0usize;
                while pi < rows[i].len() && pk < row_k_tail.len() {
                    let ci = rows[i][pi].0;
                    let ck = row_k_tail[pk].0;
                    if ci == ck {
                        rows[i][pi].1 -= l_ik * row_k_tail[pk].1;
                        pi += 1;
                        pk += 1;
                    } else if ci < ck {
                        pi += 1;
                    } else {
                        pk += 1;
                    }
                }
            }
            let u_ii = rows[i][d_i].1;
            if u_ii.abs() < 1e-30 || !u_ii.is_finite() {
                return Err(MatroError::runtime(format!(
                    "ilu: zero pivot encountered at row {}",
                    i + 1
                )));
            }
        }

        let mut l_triplets = Vec::with_capacity(self.nnz() + n);
        let mut u_triplets = Vec::with_capacity(self.nnz() + n);
        for (i, row) in rows.into_iter().enumerate() {
            l_triplets.push((i, i, 1.0));
            for (c, v) in row {
                if v == 0.0 {
                    continue;
                }
                if c < i {
                    l_triplets.push((i, c, v));
                } else {
                    u_triplets.push((i, c, v));
                }
            }
        }
        Ok((
            Self::from_triplets(n, n, &l_triplets)?,
            Self::from_triplets(n, n, &u_triplets)?,
        ))
    }

    /// Gilbert-Peierls sparse left-looking LU factorization with partial pivoting:
    /// returns `(L, U, perm)` such that `A[perm[i], :] == (L * U)[i, :]`,
    /// where `L` is unit lower-triangular sparse and `U` is upper-triangular sparse.
    pub fn lu_sparse(&self) -> Result<(Self, Self, Vec<usize>)> {
        if self.rows != self.cols {
            return Err(MatroError::dimension("Sparse LU requires a square matrix"));
        }
        let n = self.rows;
        if n == 0 {
            return Ok((Self::zeros(0, 0), Self::zeros(0, 0), Vec::new()));
        }
        let mut l_cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        let mut u_cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        // pinv[orig_row] = pivot step k (0..n), or usize::MAX if unpivoted
        let mut pinv = vec![usize::MAX; n];
        let mut perm = vec![0usize; n];

        let mut x = vec![0.0; n];
        let mut mark = vec![0usize; n];
        let mut dfs_stack: Vec<(usize, usize)> = Vec::new();
        let mut topo: Vec<usize> = Vec::new();
        let mut unpivoted: Vec<usize> = Vec::new();
        let mut unpiv_mark = vec![0usize; n];
        let mut cur_mark = 0usize;

        for j in 0..n {
            cur_mark += 1;
            topo.clear();
            unpivoted.clear();

            // 1. Symbolic reachability DFS in DAG of L from nonzeros of A[:, j]
            for idx in self.col_ptr[j]..self.col_ptr[j + 1] {
                let r = self.row_idx[idx];
                x[r] += self.values[idx];
                let step = pinv[r];
                if step == usize::MAX {
                    if unpiv_mark[r] != cur_mark {
                        unpiv_mark[r] = cur_mark;
                        unpivoted.push(r);
                    }
                } else if mark[step] != cur_mark {
                    mark[step] = cur_mark;
                    dfs_stack.push((step, 0));
                    while let Some((col_k, pos)) = dfs_stack.last_mut() {
                        let k = *col_k;
                        if *pos < l_cols[k].len() {
                            let next_r = l_cols[k][*pos].0;
                            *pos += 1;
                            let next_step = pinv[next_r];
                            if next_step == usize::MAX {
                                if unpiv_mark[next_r] != cur_mark {
                                    unpiv_mark[next_r] = cur_mark;
                                    unpivoted.push(next_r);
                                }
                            } else if mark[next_step] != cur_mark {
                                mark[next_step] = cur_mark;
                                dfs_stack.push((next_step, 0));
                            }
                        } else {
                            topo.push(k);
                            dfs_stack.pop();
                        }
                    }
                }
            }

            // 2. Numeric sparse triangular solve in reverse postorder (topological order)
            for &k in topo.iter().rev() {
                let piv_r = perm[k];
                let uk = x[piv_r];
                x[piv_r] = 0.0;
                if uk != 0.0 {
                    u_cols[j].push((k, uk));
                    for &(r, lrk) in &l_cols[k] {
                        x[r] -= lrk * uk;
                        if pinv[r] == usize::MAX && unpiv_mark[r] != cur_mark {
                            unpiv_mark[r] = cur_mark;
                            unpivoted.push(r);
                        }
                    }
                }
            }

            // 3. Partial pivoting among unpivoted nonzero rows
            let mut best_row = usize::MAX;
            let mut best_abs = 0.0;
            for &r in &unpivoted {
                let a = x[r].abs();
                if a > best_abs {
                    best_abs = a;
                    best_row = r;
                }
            }
            if best_row == usize::MAX || !best_abs.is_finite() || best_abs == 0.0 {
                // Zero out any residual workspace entries and report singular
                for &r in &unpivoted {
                    x[r] = 0.0;
                }
                return Err(MatroError::runtime(format!(
                    "Sparse LU: matrix is singular at column {}",
                    j + 1
                )));
            }

            let piv_val = x[best_row];
            x[best_row] = 0.0;
            pinv[best_row] = j;
            perm[j] = best_row;
            u_cols[j].push((j, piv_val));
            u_cols[j].sort_unstable_by_key(|&(r, _)| r);

            let inv_piv = 1.0 / piv_val;
            for &r in &unpivoted {
                if r == best_row {
                    continue;
                }
                let v = x[r];
                x[r] = 0.0;
                if v != 0.0 {
                    l_cols[j].push((r, v * inv_piv));
                }
            }
        }

        // Build CSC L (with row indices remapped through pinv + unit diagonal) and CSC U
        let mut l_col_ptr = Vec::with_capacity(n + 1);
        let mut l_row_idx = Vec::new();
        let mut l_values = Vec::new();
        l_col_ptr.push(0);
        for (j, mut col) in l_cols.into_iter().enumerate() {
            for (r, _) in &mut col {
                *r = pinv[*r];
            }
            col.push((j, 1.0));
            col.sort_unstable_by_key(|&(r, _)| r);
            for (r, v) in col {
                l_row_idx.push(r);
                l_values.push(v);
            }
            l_col_ptr.push(l_row_idx.len());
        }

        let mut u_col_ptr = Vec::with_capacity(n + 1);
        let mut u_row_idx = Vec::new();
        let mut u_values = Vec::new();
        u_col_ptr.push(0);
        for col in u_cols {
            for (r, v) in col {
                u_row_idx.push(r);
                u_values.push(v);
            }
            u_col_ptr.push(u_row_idx.len());
        }

        Ok((
            Self {
                rows: n,
                cols: n,
                col_ptr: l_col_ptr,
                row_idx: l_row_idx,
                values: l_values,
                imag: None,
            },
            Self {
                rows: n,
                cols: n,
                col_ptr: u_col_ptr,
                row_idx: u_row_idx,
                values: u_values,
                imag: None,
            },
            perm,
        ))
    }

    /// $O(\text{nnz})$ complex sparse forward substitution for lower-triangular CSC matrices.
    pub fn solve_lower_triangular_c(&self, b: &[Complex]) -> Option<Vec<Complex>> {
        let n = self.rows;
        if self.cols != n || b.len() != n {
            return None;
        }
        let mut x = b.to_vec();
        for c in 0..n {
            let start = self.col_ptr[c];
            let end = self.col_ptr[c + 1];
            let mut diag = Complex::ZERO;
            let mut diag_pos = None;
            for idx in start..end {
                if self.row_idx[idx] == c {
                    diag = self.val_c(idx);
                    diag_pos = Some(idx);
                    break;
                }
            }
            let dabs = diag.abs();
            if dabs < 1e-30 || !dabs.is_finite() {
                return None;
            }
            let xc = x[c] / diag;
            x[c] = xc;
            if xc.re != 0.0 || xc.im != 0.0 {
                let first_sub = diag_pos.map_or(start, |p| p + 1);
                for idx in first_sub..end {
                    let r = self.row_idx[idx];
                    if r > c {
                        x[r] = x[r] - self.val_c(idx) * xc;
                    }
                }
            }
        }
        Some(x)
    }

    /// $O(\text{nnz})$ complex sparse back substitution for upper-triangular CSC matrices.
    pub fn solve_upper_triangular_c(&self, b: &[Complex]) -> Option<Vec<Complex>> {
        let n = self.rows;
        if self.cols != n || b.len() != n {
            return None;
        }
        let mut x = b.to_vec();
        for c in (0..n).rev() {
            let start = self.col_ptr[c];
            let end = self.col_ptr[c + 1];
            let mut diag = Complex::ZERO;
            for idx in (start..end).rev() {
                if self.row_idx[idx] == c {
                    diag = self.val_c(idx);
                    break;
                }
            }
            let dabs = diag.abs();
            if dabs < 1e-30 || !dabs.is_finite() {
                return None;
            }
            let xc = x[c] / diag;
            x[c] = xc;
            if xc.re != 0.0 || xc.im != 0.0 {
                for idx in start..end {
                    let r = self.row_idx[idx];
                    if r < c {
                        x[r] = x[r] - self.val_c(idx) * xc;
                    }
                }
            }
        }
        Some(x)
    }

    /// Complex Gilbert-Peierls sparse left-looking LU factorization with partial pivoting:
    /// returns `(L, U, perm)` such that `A[perm[i], :] == (L * U)[i, :]`.
    pub fn lu_sparse_complex(&self) -> Result<(Self, Self, Vec<usize>)> {
        if !self.is_complex() {
            return self.lu_sparse();
        }
        if self.rows != self.cols {
            return Err(MatroError::dimension("Sparse LU requires a square matrix"));
        }
        let n = self.rows;
        if n == 0 {
            return Ok((Self::zeros(0, 0), Self::zeros(0, 0), Vec::new()));
        }
        let mut l_cols: Vec<Vec<(usize, Complex)>> = vec![Vec::new(); n];
        let mut u_cols: Vec<Vec<(usize, Complex)>> = vec![Vec::new(); n];
        let mut pinv = vec![usize::MAX; n];
        let mut perm = vec![0usize; n];

        let mut x = vec![Complex::ZERO; n];
        let mut mark = vec![0usize; n];
        let mut dfs_stack: Vec<(usize, usize)> = Vec::new();
        let mut topo: Vec<usize> = Vec::new();
        let mut unpivoted: Vec<usize> = Vec::new();
        let mut unpiv_mark = vec![0usize; n];
        let mut cur_mark = 0usize;

        for j in 0..n {
            cur_mark += 1;
            topo.clear();
            unpivoted.clear();

            for idx in self.col_ptr[j]..self.col_ptr[j + 1] {
                let r = self.row_idx[idx];
                x[r] = x[r] + self.val_c(idx);
                let step = pinv[r];
                if step == usize::MAX {
                    if unpiv_mark[r] != cur_mark {
                        unpiv_mark[r] = cur_mark;
                        unpivoted.push(r);
                    }
                } else if mark[step] != cur_mark {
                    mark[step] = cur_mark;
                    dfs_stack.push((step, 0));
                    while let Some((col_k, pos)) = dfs_stack.last_mut() {
                        let k = *col_k;
                        if *pos < l_cols[k].len() {
                            let next_r = l_cols[k][*pos].0;
                            *pos += 1;
                            let next_step = pinv[next_r];
                            if next_step == usize::MAX {
                                if unpiv_mark[next_r] != cur_mark {
                                    unpiv_mark[next_r] = cur_mark;
                                    unpivoted.push(next_r);
                                }
                            } else if mark[next_step] != cur_mark {
                                mark[next_step] = cur_mark;
                                dfs_stack.push((next_step, 0));
                            }
                        } else {
                            topo.push(k);
                            dfs_stack.pop();
                        }
                    }
                }
            }

            for &k in topo.iter().rev() {
                let piv_r = perm[k];
                let uk = x[piv_r];
                x[piv_r] = Complex::ZERO;
                if uk.re != 0.0 || uk.im != 0.0 {
                    u_cols[j].push((k, uk));
                    for &(r, lrk) in &l_cols[k] {
                        x[r] = x[r] - lrk * uk;
                        if pinv[r] == usize::MAX && unpiv_mark[r] != cur_mark {
                            unpiv_mark[r] = cur_mark;
                            unpivoted.push(r);
                        }
                    }
                }
            }

            let mut best_row = usize::MAX;
            let mut best_abs = 0.0;
            for &r in &unpivoted {
                let a = x[r].abs();
                if a > best_abs {
                    best_abs = a;
                    best_row = r;
                }
            }
            if best_row == usize::MAX || !best_abs.is_finite() || best_abs == 0.0 {
                for &r in &unpivoted {
                    x[r] = Complex::ZERO;
                }
                return Err(MatroError::runtime(format!(
                    "Sparse LU: matrix is singular at column {}",
                    j + 1
                )));
            }

            let piv_val = x[best_row];
            x[best_row] = Complex::ZERO;
            pinv[best_row] = j;
            perm[j] = best_row;
            u_cols[j].push((j, piv_val));
            u_cols[j].sort_unstable_by_key(|&(r, _)| r);

            for &r in &unpivoted {
                if r == best_row {
                    continue;
                }
                let v = x[r];
                x[r] = Complex::ZERO;
                if v.re != 0.0 || v.im != 0.0 {
                    l_cols[j].push((r, v / piv_val));
                }
            }
        }

        let mut l_trips = Vec::new();
        for (j, col) in l_cols.into_iter().enumerate() {
            l_trips.push((j, j, Complex::ONE));
            for (r, v) in col {
                l_trips.push((pinv[r], j, v));
            }
        }
        let mut u_trips = Vec::new();
        for (j, col) in u_cols.into_iter().enumerate() {
            for (r, v) in col {
                u_trips.push((r, j, v));
            }
        }
        Ok((
            Self::from_complex_triplets(n, n, &l_trips)?,
            Self::from_complex_triplets(n, n, &u_trips)?,
            perm,
        ))
    }

    /// Solve complex `A * X = B` given a precomputed sparse LU factorization `(L, U, perm)`.
    pub fn solve_with_lu_c(l: &Self, u: &Self, perm: &[usize], rhs: &Matrix) -> Result<Matrix> {
        let n = perm.len();
        let mut out = Matrix::complex_zeros(n, rhs.cols());
        for j in 0..rhs.cols() {
            let mut pb = vec![Complex::ZERO; n];
            for i in 0..n {
                pb[i] = rhs.get_c(perm[i], j);
            }
            let y = l
                .solve_lower_triangular_c(&pb)
                .ok_or_else(|| MatroError::runtime("Complex sparse LU forward solve failed"))?;
            let x = u
                .solve_upper_triangular_c(&y)
                .ok_or_else(|| MatroError::runtime("Complex sparse LU back solve failed"))?;
            for (i, &v) in x.iter().enumerate() {
                out.set_c(i, j, v);
            }
        }
        Ok(out.normalize())
    }

    /// Solve `A * X = B` given a precomputed sparse LU factorization `(L, U, perm)`.
    pub fn solve_with_lu(l: &Self, u: &Self, perm: &[usize], rhs: &Matrix) -> Result<Matrix> {
        let n = perm.len();
        let mut out = Matrix::zeros(n, rhs.cols());
        for j in 0..rhs.cols() {
            let mut pb = vec![0.0; n];
            for i in 0..n {
                pb[i] = rhs.get(perm[i], j);
            }
            let y = l
                .solve_lower_triangular(&pb)
                .ok_or_else(|| MatroError::runtime("Sparse LU forward solve failed"))?;
            let x = u
                .solve_upper_triangular(&y)
                .ok_or_else(|| MatroError::runtime("Sparse LU back solve failed"))?;
            for (i, &v) in x.iter().enumerate() {
                out.set(i, j, v);
            }
        }
        Ok(out)
    }

    /// Exact Banded Cholesky factorization `A = L * L^T` for symmetric positive definite banded systems.
    fn factor_banded_cholesky(&self, bw: usize) -> Option<Vec<f64>> {
        let n = self.rows;
        let stride = bw + 1;
        let mut l = vec![0.0; n * stride];
        for c in 0..n {
            for idx in self.col_ptr[c]..self.col_ptr[c + 1] {
                let r = self.row_idx[idx];
                if r >= c && r - c <= bw {
                    l[c * stride + (r - c)] = self.values[idx];
                }
            }
        }
        for j in 0..n {
            let diag = l[j * stride];
            if !diag.is_finite() || diag <= 0.0 {
                return None;
            }
            let ljj = diag.sqrt();
            l[j * stride] = ljj;
            let max_i = (j + bw).min(n - 1);
            let inv_ljj = 1.0 / ljj;
            let col_j = j * stride;
            for i in (j + 1)..=max_i {
                l[col_j + (i - j)] *= inv_ljj;
            }
            for k in (j + 1)..=max_i {
                let lkj = l[col_j + (k - j)];
                if lkj == 0.0 {
                    continue;
                }
                let col_k = k * stride;
                for i in k..=max_i {
                    l[col_k + (i - k)] -= l[col_j + (i - j)] * lkj;
                }
            }
        }
        Some(l)
    }

    fn solve_with_banded_cholesky(l: &[f64], n: usize, bw: usize, b: &[f64]) -> Vec<f64> {
        let stride = bw + 1;
        let mut y = b.to_vec();
        for j in 0..n {
            let col_j = j * stride;
            let yj = y[j] / l[col_j];
            y[j] = yj;
            if yj != 0.0 {
                let max_i = (j + bw).min(n - 1);
                for i in (j + 1)..=max_i {
                    y[i] -= l[col_j + (i - j)] * yj;
                }
            }
        }
        for j in (0..n).rev() {
            let col_j = j * stride;
            let mut s = y[j];
            let max_i = (j + bw).min(n - 1);
            for i in (j + 1)..=max_i {
                s -= l[col_j + (i - j)] * y[i];
            }
            y[j] = s / l[col_j];
        }
        y
    }

    /// Exact Banded LU with partial pivoting (LAPACK `DGBTRF`/`DGBTRS` layout) for general banded sparse systems.
    fn factor_banded_lu(&self, kl: usize, ku: usize) -> Option<(Vec<f64>, Vec<usize>)> {
        let n = self.rows;
        let ldab = 2 * kl + ku + 1;
        let kv = kl + ku;
        let mut ab = vec![0.0; n * ldab];
        for c in 0..n {
            for idx in self.col_ptr[c]..self.col_ptr[c + 1] {
                let r = self.row_idx[idx];
                let row_slot = kv as isize + r as isize - c as isize;
                if row_slot >= 0 && (row_slot as usize) < ldab {
                    ab[c * ldab + row_slot as usize] = self.values[idx];
                }
            }
        }
        let mut ipiv = vec![0usize; n];
        for j in 0..n {
            let max_row_off = kl.min(n - 1 - j);
            let mut piv_off = 0usize;
            let mut piv_val = ab[j * ldab + kv].abs();
            for off in 1..=max_row_off {
                let v = ab[j * ldab + kv + off].abs();
                if v > piv_val {
                    piv_val = v;
                    piv_off = off;
                }
            }
            if !piv_val.is_finite() || piv_val == 0.0 {
                return None;
            }
            ipiv[j] = j + piv_off;
            let max_col = (j + ku + kl).min(n - 1);
            if piv_off != 0 {
                for k in j..=max_col {
                    let base = k * ldab;
                    let r1 = kv + j - k;
                    let r2 = r1 + piv_off;
                    ab.swap(base + r1, base + r2);
                }
            }
            let pivot = ab[j * ldab + kv];
            let inv_piv = 1.0 / pivot;
            for off in 1..=max_row_off {
                ab[j * ldab + kv + off] *= inv_piv;
            }
            for k in (j + 1)..=max_col {
                let base_k = k * ldab;
                let u_jk = ab[base_k + kv + j - k];
                if u_jk == 0.0 {
                    continue;
                }
                for off in 1..=max_row_off {
                    let m = ab[j * ldab + kv + off];
                    ab[base_k + kv + j + off - k] -= m * u_jk;
                }
            }
        }
        Some((ab, ipiv))
    }

    fn solve_with_banded_lu(
        ab: &[f64],
        ipiv: &[usize],
        n: usize,
        kl: usize,
        ku: usize,
        b: &[f64],
    ) -> Vec<f64> {
        let ldab = 2 * kl + ku + 1;
        let kv = kl + ku;
        let mut x = b.to_vec();
        for j in 0..n {
            let p = ipiv[j];
            if p != j {
                x.swap(j, p);
            }
            let xj = x[j];
            if xj != 0.0 {
                let max_off = kl.min(n - 1 - j);
                for off in 1..=max_off {
                    x[j + off] -= ab[j * ldab + kv + off] * xj;
                }
            }
        }
        for j in (0..n).rev() {
            let base = j * ldab;
            let xj = x[j] / ab[base + kv];
            x[j] = xj;
            if xj != 0.0 {
                let min_i = j.saturating_sub(kv);
                for i in min_i..j {
                    x[i] -= ab[base + kv + i - j] * xj;
                }
            }
        }
        x
    }

    /// Jacobi-preconditioned Conjugate Gradient solver for symmetric positive definite sparse systems (`pcg`).
    pub fn solve_cg(&self, b: &[f64], tol: f64, maxit: usize) -> Result<Vec<f64>> {
        if self.rows != self.cols || self.rows != b.len() {
            return Err(MatroError::dimension(
                "pcg: matrix must be square and match RHS length",
            ));
        }
        let n = self.rows;
        let b_norm = dot(b, b).sqrt().max(1e-15);
        let inv_diag: Vec<f64> = (0..n)
            .map(|i| {
                let d = self.get(i, i);
                if d.abs() > 1e-30 { 1.0 / d } else { 1.0 }
            })
            .collect();
        let mut x = vec![0.0; n];
        let mut r = b.to_vec();
        let mut z: Vec<f64> = (0..n).map(|i| r[i] * inv_diag[i]).collect();
        let mut p = z.clone();
        let mut rz_old = dot(&r, &z);

        for _ in 0..maxit {
            if dot(&r, &r).sqrt() / b_norm < tol {
                return Ok(x);
            }
            let ap = self.matvec(&p)?;
            let pap = dot(&p, &ap);
            if pap.abs() < 1e-30 {
                break;
            }
            let alpha = rz_old / pap;
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            if dot(&r, &r).sqrt() / b_norm < tol {
                return Ok(x);
            }
            for i in 0..n {
                z[i] = r[i] * inv_diag[i];
            }
            let rz_new = dot(&r, &z);
            if rz_old.abs() < 1e-30 {
                break;
            }
            let beta = rz_new / rz_old;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
            rz_old = rz_new;
        }
        Ok(x)
    }

    /// BiCGSTAB iterative solver for general non-symmetric square sparse systems.
    pub fn solve_bicgstab(&self, b: &[f64], tol: f64, maxit: usize) -> Result<Vec<f64>> {
        if self.rows != self.cols || self.rows != b.len() {
            return Err(MatroError::dimension(
                "Sparse solve: matrix must be square and match RHS length",
            ));
        }
        let n = self.rows;
        let b_norm = dot(b, b).sqrt().max(1e-15);
        let mut x = vec![0.0; n];
        let mut r = b.to_vec();
        let r_hat = r.clone();
        let mut p = vec![0.0; n];
        let mut v = vec![0.0; n];
        let mut rho_prev = 1.0;
        let mut alpha = 1.0;
        let mut omega = 1.0;

        for _ in 0..maxit {
            if dot(&r, &r).sqrt() / b_norm < tol {
                return Ok(x);
            }
            let rho = dot(&r_hat, &r);
            if rho.abs() < 1e-30 {
                break;
            }
            let beta = (rho / rho_prev) * (alpha / omega);
            for i in 0..n {
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
            }
            v = self.matvec(&p)?;
            let denom = dot(&r_hat, &v);
            if denom.abs() < 1e-30 {
                break;
            }
            alpha = rho / denom;
            let s: Vec<f64> = (0..n).map(|i| r[i] - alpha * v[i]).collect();
            if dot(&s, &s).sqrt() / b_norm < tol {
                for i in 0..n {
                    x[i] += alpha * p[i];
                }
                return Ok(x);
            }
            let t = self.matvec(&s)?;
            let tt = dot(&t, &t);
            if tt.abs() < 1e-30 {
                break;
            }
            omega = dot(&t, &s) / tt;
            for i in 0..n {
                x[i] += alpha * p[i] + omega * s[i];
                r[i] = s[i] - omega * t[i];
            }
            rho_prev = rho;
            if omega.abs() < 1e-30 {
                break;
            }
        }
        Ok(x)
    }

    /// Sparse linear solve `S \ B`: supports complex RHS, rectangular systems, exact Banded Cholesky
    /// (for banded SPD), exact Banded LU (for general banded), and Jacobi-PCG / BiCGSTAB fallback.
    pub fn solve_dense(&self, rhs: &Matrix) -> Result<Matrix> {
        if self.rows != rhs.rows() {
            return Err(MatroError::dimension(format!(
                "Sparse solve dimension mismatch: {}x{} \\ {}x{}",
                self.rows,
                self.cols,
                rhs.rows(),
                rhs.cols()
            )));
        }
        if self.is_complex() {
            if self.rows != self.cols {
                if self.rows.saturating_mul(self.cols) <= 250_000 {
                    return self.to_dense().solve(rhs);
                }
                if self.rows > self.cols {
                    let ah = self.ctranspose();
                    let aha = ah.matmul_sparse(self)?;
                    let ahb = ah.matmul_dense(rhs)?;
                    return aha.solve_dense(&ahb);
                }
                return Err(MatroError::dimension(format!(
                    "Underdetermined large complex sparse solve ({}x{}) exceeds dense fallback limit",
                    self.rows, self.cols
                )));
            }
            let n = self.rows;
            if n == 0 {
                return Ok(Matrix::zeros(0, rhs.cols()));
            }
            if n <= 16 {
                return self.to_dense().solve(rhs);
            }
            let (kl, ku) = self.bandwidth();
            if ku == 0 {
                let mut out = Matrix::complex_zeros(n, rhs.cols());
                let mut ok = true;
                for j in 0..rhs.cols() {
                    let b: Vec<Complex> = (0..n).map(|r| rhs.get_c(r, j)).collect();
                    if let Some(x) = self.solve_lower_triangular_c(&b) {
                        for (i, &v) in x.iter().enumerate() {
                            out.set_c(i, j, v);
                        }
                    } else {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    return Ok(out.normalize());
                }
            }
            if kl == 0 {
                let mut out = Matrix::complex_zeros(n, rhs.cols());
                let mut ok = true;
                for j in 0..rhs.cols() {
                    let b: Vec<Complex> = (0..n).map(|r| rhs.get_c(r, j)).collect();
                    if let Some(x) = self.solve_upper_triangular_c(&b) {
                        for (i, &v) in x.iter().enumerate() {
                            out.set_c(i, j, v);
                        }
                    } else {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    return Ok(out.normalize());
                }
            }
            let (l, u, perm) = self.lu_sparse_complex()?;
            return Self::solve_with_lu_c(&l, &u, &perm, rhs);
        }
        if rhs.is_complex() {
            let re_mat = Matrix::from_col_major(rhs.rows(), rhs.cols(), rhs.data().to_vec())?;
            let im_data = rhs
                .imag_data()
                .map(|s| s.to_vec())
                .unwrap_or_else(|| vec![0.0; rhs.len()]);
            let im_mat = Matrix::from_col_major(rhs.rows(), rhs.cols(), im_data)?;
            let x_re = self.solve_dense(&re_mat)?;
            let x_im = self.solve_dense(&im_mat)?;
            return Matrix::from_complex_col_major(
                x_re.rows(),
                x_re.cols(),
                x_re.data().to_vec(),
                x_im.data().to_vec(),
            )
            .map(Matrix::normalize);
        }
        if self.rows != self.cols {
            // Rectangular sparse system: dense QR/SVD solve for moderate sizes, normal equations otherwise
            if self.rows.saturating_mul(self.cols) <= 250_000 {
                return self.to_dense().solve(rhs);
            }
            if self.rows > self.cols {
                let at = self.transpose();
                let ata = at.matmul_sparse(self)?;
                let atb = at.matmul_dense(rhs)?;
                return ata.solve_dense(&atb);
            }
            return Err(MatroError::dimension(format!(
                "Underdetermined large sparse solve ({}x{}) exceeds dense fallback limit",
                self.rows, self.cols
            )));
        }
        let n = self.rows;
        if n == 0 {
            return Ok(Matrix::zeros(0, rhs.cols()));
        }
        if n <= 16 {
            return self.to_dense().solve(rhs);
        }

        let (kl, ku) = self.bandwidth();

        // O(nnz) exact direct triangular solves (makes ichol/ilu preconditioners blazing fast!)
        if ku == 0 {
            let mut out = Matrix::zeros(n, rhs.cols());
            let mut ok = true;
            for j in 0..rhs.cols() {
                let b: Vec<f64> = (0..n).map(|r| rhs.get(r, j)).collect();
                if let Some(x) = self.solve_lower_triangular(&b) {
                    for (i, &v) in x.iter().enumerate() {
                        out.set(i, j, v);
                    }
                } else {
                    ok = false;
                    break;
                }
            }
            if ok {
                return Ok(out);
            }
        }
        if kl == 0 {
            let mut out = Matrix::zeros(n, rhs.cols());
            let mut ok = true;
            for j in 0..rhs.cols() {
                let b: Vec<f64> = (0..n).map(|r| rhs.get(r, j)).collect();
                if let Some(x) = self.solve_upper_triangular(&b) {
                    for (i, &v) in x.iter().enumerate() {
                        out.set(i, j, v);
                    }
                } else {
                    ok = false;
                    break;
                }
            }
            if ok {
                return Ok(out);
            }
        }

        let sym = kl == ku && self.is_symmetric();

        // Fast exact direct Banded Cholesky for banded SPD matrices (e.g. 2D FEM stiffness matrices)
        if sym && kl <= 256 && n.saturating_mul(kl + 1) <= 2_000_000 {
            if let Some(l) = self.factor_banded_cholesky(kl) {
                let mut out = Matrix::zeros(n, rhs.cols());
                for j in 0..rhs.cols() {
                    let b: Vec<f64> = (0..n).map(|r| rhs.get(r, j)).collect();
                    let x = Self::solve_with_banded_cholesky(&l, n, kl, &b);
                    for (i, &val) in x.iter().enumerate() {
                        out.set(i, j, val);
                    }
                }
                return Ok(out);
            }
        }

        // Fast exact direct Banded LU with partial pivoting for general or indefinite banded matrices
        if kl + ku <= 256 && n.saturating_mul(2 * kl + ku + 1) <= 2_000_000 {
            if let Some((ab, ipiv)) = self.factor_banded_lu(kl, ku) {
                let mut out = Matrix::zeros(n, rhs.cols());
                for j in 0..rhs.cols() {
                    let b: Vec<f64> = (0..n).map(|r| rhs.get(r, j)).collect();
                    let x = Self::solve_with_banded_lu(&ab, &ipiv, n, kl, ku, &b);
                    for (i, &val) in x.iter().enumerate() {
                        out.set(i, j, val);
                    }
                }
                return Ok(out);
            }
        }

        // Exact direct Gilbert-Peierls sparse left-looking LU for general non-banded sparse matrices
        // (handles arbitrary graph / loop-closure / advection-diffusion sparsity without dense O(n^2) blowup)
        if n <= 3000 {
            if let Ok((l, u, perm)) = self.lu_sparse() {
                if let Ok(sol) = Self::solve_with_lu(&l, &u, &perm, rhs) {
                    return Ok(sol);
                }
            }
        }

        let maxit = (n * 4).max(400);
        let mut out = Matrix::zeros(self.cols, rhs.cols());
        for j in 0..rhs.cols() {
            let b: Vec<f64> = (0..rhs.rows()).map(|r| rhs.get(r, j)).collect();
            let x = if sym {
                let cand = self.solve_cg(&b, 1e-10, maxit)?;
                let ax = self.matvec(&cand)?;
                let res: f64 = (0..b.len()).map(|i| (ax[i] - b[i]).powi(2)).sum::<f64>().sqrt();
                let bn = dot(&b, &b).sqrt().max(1.0);
                if res / bn < 1e-6 {
                    cand
                } else {
                    self.solve_bicgstab(&b, 1e-10, maxit)?
                }
            } else {
                self.solve_bicgstab(&b, 1e-10, maxit)?
            };
            for (i, &val) in x.iter().enumerate() {
                out.set(i, j, val);
            }
        }
        Ok(out)
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

impl fmt::Display for SparseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.nnz() == 0 {
            return write!(f, "All zero sparse: {}x{}", self.rows, self.cols);
        }
        let limit = 24usize;
        let mut shown = 0usize;
        for c in 0..self.cols {
            for idx in self.col_ptr[c]..self.col_ptr[c + 1] {
                if shown > 0 {
                    writeln!(f)?;
                }
                if shown == limit {
                    write!(f, "  ... ({} more non-zeros)", self.nnz() - limit)?;
                    return Ok(());
                }
                let v = self.val_c(idx);
                if self.is_complex() {
                    write!(
                        f,
                        "  ({}, {})    {:.4} {:+.4}i",
                        self.row_idx[idx] + 1,
                        c + 1,
                        v.re,
                        v.im
                    )?;
                } else {
                    write!(
                        f,
                        "  ({}, {})    {:.4}",
                        self.row_idx[idx] + 1,
                        c + 1,
                        v.re
                    )?;
                }
                shown += 1;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_poisson_cg_solve() {
        // 1D Laplacian (tridiagonal SPD, n = 50)
        let n = 50usize;
        let mut triplets = Vec::new();
        for i in 0..n {
            triplets.push((i, i, 2.0));
            if i + 1 < n {
                triplets.push((i, i + 1, -1.0));
                triplets.push((i + 1, i, -1.0));
            }
        }
        let a = SparseMatrix::from_triplets(n, n, &triplets).unwrap();
        let b = Matrix::ones(n, 1);
        let x = a.solve_dense(&b).unwrap();
        let ax = a.matmul_dense(&x).unwrap();
        let err: f64 = (0..n).map(|i| (ax.get(i, 0) - 1.0).powi(2)).sum::<f64>().sqrt();
        assert!(err < 1e-8);
    }
}
