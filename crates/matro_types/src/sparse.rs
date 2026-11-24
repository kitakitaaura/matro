//! Compressed Sparse Column (CSC) matrix with SpMV, SpGEMM, and CG / BiCGSTAB iterative solvers.

use std::collections::BTreeMap;
use std::fmt;

use crate::{MatroError, Matrix, Result};

/// Compressed Sparse Column (CSC) matrix of real `f64` values (matching MATLAB's sparse storage).
#[derive(Debug, Clone, PartialEq)]
pub struct SparseMatrix {
    rows: usize,
    cols: usize,
    /// Length `cols + 1`: column `c` entries live at `col_ptr[c]..col_ptr[c + 1]`.
    col_ptr: Vec<usize>,
    /// Row index for each non-zero entry (sorted ascending within each column).
    row_idx: Vec<usize>,
    /// Non-zero numeric values.
    values: Vec<f64>,
}

impl SparseMatrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            col_ptr: vec![0; cols + 1],
            row_idx: Vec::new(),
            values: Vec::new(),
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
        }
    }

    /// Construct from 0-based `(row, col, val)` triplets, summing duplicate coordinates (MATLAB convention).
    pub fn from_triplets(rows: usize, cols: usize, triplets: &[(usize, usize, f64)]) -> Result<Self> {
        let mut by_col: Vec<BTreeMap<usize, f64>> = vec![BTreeMap::new(); cols];
        for &(r, c, v) in triplets {
            if r >= rows || c >= cols {
                return Err(MatroError::index(format!(
                    "Sparse triplet ({}, {}) out of bounds for {rows}x{cols}",
                    r + 1,
                    c + 1
                )));
            }
            if v != 0.0 {
                *by_col[c].entry(r).or_insert(0.0) += v;
            }
        }
        let mut col_ptr = Vec::with_capacity(cols + 1);
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        col_ptr.push(0);
        for col in by_col {
            for (r, v) in col {
                if v != 0.0 {
                    row_idx.push(r);
                    values.push(v);
                }
            }
            col_ptr.push(row_idx.len());
        }
        Ok(Self {
            rows,
            cols,
            col_ptr,
            row_idx,
            values,
        })
    }

    pub fn from_dense(m: &Matrix) -> Self {
        let mut col_ptr = Vec::with_capacity(m.cols() + 1);
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        col_ptr.push(0);
        for c in 0..m.cols() {
            for r in 0..m.rows() {
                let v = m.get(r, c);
                if v != 0.0 {
                    row_idx.push(r);
                    values.push(v);
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
        }
    }

    pub fn to_dense(&self) -> Matrix {
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

    pub fn get(&self, r: usize, c: usize) -> f64 {
        let start = self.col_ptr[c];
        let end = self.col_ptr[c + 1];
        match self.row_idx[start..end].binary_search(&r) {
            Ok(offset) => self.values[start + offset],
            Err(_) => 0.0,
        }
    }

    pub fn scale(&self, alpha: f64) -> Self {
        if alpha == 0.0 {
            return Self::zeros(self.rows, self.cols);
        }
        Self {
            rows: self.rows,
            cols: self.cols,
            col_ptr: self.col_ptr.clone(),
            row_idx: self.row_idx.clone(),
            values: self.values.iter().map(|&v| v * alpha).collect(),
        }
    }

    pub fn transpose(&self) -> Self {
        let mut triplets = Vec::with_capacity(self.nnz());
        for c in 0..self.cols {
            for idx in self.col_ptr[c]..self.col_ptr[c + 1] {
                triplets.push((c, self.row_idx[idx], self.values[idx]));
            }
        }
        Self::from_triplets(self.cols, self.rows, &triplets).unwrap()
    }

    /// Sparse addition / subtraction `alpha * self + beta * other`.
    pub fn add_scaled(&self, other: &Self, beta: f64) -> Result<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MatroError::dimension(format!(
                "Sparse dimension mismatch: {}x{} vs {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )));
        }
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
                *acc.entry(other.row_idx[idx]).or_insert(0.0) += beta * other.values[idx];
            }
            for (r, v) in acc {
                if v.abs() > 0.0 {
                    row_idx.push(r);
                    values.push(v);
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

    /// Sparse matrix * dense vector (`y = A * x`).
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

    /// Sparse matrix * dense matrix (`C = A * B`).
    pub fn matmul_dense(&self, rhs: &Matrix) -> Result<Matrix> {
        if let Some(s) = rhs.as_scalar() {
            return Ok(self.scale(s).to_dense());
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

    /// Sparse-sparse matrix multiplication (`C = A * B`).
    pub fn matmul_sparse(&self, other: &Self) -> Result<Self> {
        if self.cols != other.rows {
            return Err(MatroError::dimension(format!(
                "Inner dimensions must agree: {}x{} * {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )));
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

    /// Conjugate Gradient solver for symmetric positive definite sparse systems (`pcg`).
    pub fn solve_cg(&self, b: &[f64], tol: f64, maxit: usize) -> Result<Vec<f64>> {
        if self.rows != self.cols || self.rows != b.len() {
            return Err(MatroError::dimension(
                "pcg: matrix must be square and match RHS length",
            ));
        }
        let n = self.rows;
        let b_norm = dot(b, b).sqrt().max(1e-15);
        let mut x = vec![0.0; n];
        let mut r = b.to_vec();
        let mut p = r.clone();
        let mut rs_old = dot(&r, &r);

        for _ in 0..maxit {
            if rs_old.sqrt() / b_norm < tol {
                return Ok(x);
            }
            let ap = self.matvec(&p)?;
            let pap = dot(&p, &ap);
            if pap.abs() < 1e-30 {
                break;
            }
            let alpha = rs_old / pap;
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            let rs_new = dot(&r, &r);
            if rs_new.sqrt() / b_norm < tol {
                return Ok(x);
            }
            let beta = rs_new / rs_old;
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
            rs_old = rs_new;
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

    /// Sparse linear solve `S \ B`: uses CG for symmetric matrices, BiCGSTAB otherwise
    /// (falling back to dense LU for tiny matrices <= 16).
    pub fn solve_dense(&self, rhs: &Matrix) -> Result<Matrix> {
        if self.rows != self.cols || self.rows != rhs.rows() {
            return Err(MatroError::dimension(format!(
                "Sparse solve dimension mismatch: {}x{} \\ {}x{}",
                self.rows,
                self.cols,
                rhs.rows(),
                rhs.cols()
            )));
        }
        if self.rows <= 16 {
            return self.to_dense().solve(rhs);
        }
        let maxit = (self.rows * 4).max(400);
        let sym = self.is_symmetric();
        let mut out = Matrix::zeros(self.cols, rhs.cols());
        for j in 0..rhs.cols() {
            let b: Vec<f64> = (0..rhs.rows()).map(|r| rhs.get(r, j)).collect();
            let x = if sym {
                let cand = self.solve_cg(&b, 1e-10, maxit)?;
                // Verify residual (in case symmetric indefinite)
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
                write!(
                    f,
                    "  ({}, {})    {:.4}",
                    self.row_idx[idx] + 1,
                    c + 1,
                    self.values[idx]
                )?;
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
