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

    /// Column vector of all non-zero values in column-major order.
    pub fn nonzeros(&self) -> Vec<f64> {
        self.values.clone()
    }

    /// Return sparse matrix with same sparsity pattern and all non-zeros replaced by `1.0`.
    pub fn spones(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            col_ptr: self.col_ptr.clone(),
            row_idx: self.row_idx.clone(),
            values: vec![1.0; self.values.len()],
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

    /// Set entry `(r, c)` in-place (auto-growing dimensions if needed, removing explicit zeros).
    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        if r >= self.rows {
            self.rows = r + 1;
        }
        if c >= self.cols {
            let last = *self.col_ptr.last().unwrap_or(&0);
            self.col_ptr.resize(c + 2, last);
            self.cols = c + 1;
        }
        let start = self.col_ptr[c];
        let end = self.col_ptr[c + 1];
        match self.row_idx[start..end].binary_search(&r) {
            Ok(offset) => {
                let pos = start + offset;
                if val == 0.0 {
                    self.row_idx.remove(pos);
                    self.values.remove(pos);
                    for p in &mut self.col_ptr[(c + 1)..] {
                        *p -= 1;
                    }
                } else {
                    self.values[pos] = val;
                }
            }
            Err(offset) => {
                if val != 0.0 {
                    let pos = start + offset;
                    self.row_idx.insert(pos, r);
                    self.values.insert(pos, val);
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
        col_ptr.push(0);

        let mut col_entries: Vec<(usize, f64)> = Vec::new();
        for &orig_c in cols {
            col_entries.clear();
            if orig_c < self.cols {
                for idx in self.col_ptr[orig_c]..self.col_ptr[orig_c + 1] {
                    let orig_r = self.row_idx[idx];
                    let v = self.values[idx];
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
                    values.push(v);
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
        }
    }

    /// `O(nnz)` column or row sum (`dim == 1` -> `1 x cols`, `dim == 2` -> `rows x 1`).
    pub fn sum_dim(&self, dim: usize) -> Matrix {
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
        let mut col_ptr = Vec::with_capacity(out_cols + 1);
        let mut row_idx = Vec::with_capacity(self.nnz().saturating_mul(other.nnz()));
        let mut values = Vec::with_capacity(self.nnz().saturating_mul(other.nnz()));
        col_ptr.push(0);
        for ca in 0..self.cols {
            for cb in 0..other.cols {
                for ia in self.col_ptr[ca]..self.col_ptr[ca + 1] {
                    let ra = self.row_idx[ia];
                    let va = self.values[ia];
                    let r_off = ra * other.rows;
                    for ib in other.col_ptr[cb]..other.col_ptr[cb + 1] {
                        let v = va * other.values[ib];
                        if v != 0.0 {
                            row_idx.push(r_off + other.row_idx[ib]);
                            values.push(v);
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

    /// Sparse linear solve `S \ B`: uses exact Banded Cholesky (for banded SPD),
    /// exact Banded LU (for general banded), and falls back to Jacobi-PCG / BiCGSTAB.
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
        let n = self.rows;
        if n == 0 {
            return Ok(Matrix::zeros(0, rhs.cols()));
        }
        if n <= 16 {
            return self.to_dense().solve(rhs);
        }

        let (kl, ku) = self.bandwidth();
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
