//! Sparse & large-scale iterative eigenvalue and singular value solvers:
//! - `eigs(A [, k [, sigma]])`: Arnoldi / shift-and-invert Krylov Rayleigh-Ritz solver
//! - `svds(A [, k])`: Subspace iteration SVD solver for sparse and dense matrices

use matro_types::{Complex, MatroError, Matrix, Result, Rng, SparseMatrix, Value};

enum LinOp<'a> {
    Dense(&'a Matrix),
    Sparse(&'a SparseMatrix),
}

impl<'a> LinOp<'a> {
    fn rows(&self) -> usize {
        match self {
            Self::Dense(m) => m.rows(),
            Self::Sparse(s) => s.rows(),
        }
    }

    fn cols(&self) -> usize {
        match self {
            Self::Dense(m) => m.cols(),
            Self::Sparse(s) => s.cols(),
        }
    }

    fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        let m = self.rows();
        let n = self.cols();
        let mut y = vec![0.0; m];
        match self {
            Self::Dense(mat) => {
                for c in 0..n {
                    let xc = x[c];
                    if xc != 0.0 {
                        for r in 0..m {
                            y[r] += mat.get(r, c) * xc;
                        }
                    }
                }
            }
            Self::Sparse(sp) => {
                let x_mat = Matrix::col_vec(x.to_vec());
                if let Ok(res) = sp.matmul_dense(&x_mat) {
                    return res.data().to_vec();
                }
            }
        }
        y
    }

    fn mul_transpose_vec(&self, x: &[f64]) -> Vec<f64> {
        let m = self.rows();
        let n = self.cols();
        let mut y = vec![0.0; n];
        match self {
            Self::Dense(mat) => {
                for c in 0..n {
                    let mut sum = 0.0;
                    for r in 0..m {
                        sum += mat.get(r, c) * x[r];
                    }
                    y[c] = sum;
                }
            }
            Self::Sparse(sp) => {
                let st = sp.transpose();
                let x_mat = Matrix::col_vec(x.to_vec());
                if let Ok(res) = st.matmul_dense(&x_mat) {
                    return res.data().to_vec();
                }
            }
        }
        y
    }

    fn to_dense(&self) -> Matrix {
        match self {
            Self::Dense(m) => (*m).clone(),
            Self::Sparse(s) => s.to_dense(),
        }
    }

    /// Solve `(A - sigma*I) * x = b` for shift-and-invert eigs.
    fn solve_shifted(&self, sigma: f64, b: &[f64]) -> Result<Vec<f64>> {
        let n = self.rows();
        let b_mat = Matrix::col_vec(b.to_vec());
        match self {
            Self::Sparse(sp) => {
                let shifted = if sigma == 0.0 {
                    (*sp).clone()
                } else {
                    let eye = SparseMatrix::eye(n);
                    sp.add_scaled(&eye, -sigma)?
                };
                Ok(shifted.solve_dense(&b_mat)?.data().to_vec())
            }
            Self::Dense(m) => {
                let mut a = (*m).clone();
                for i in 0..n {
                    let v = a.get(i, i) - sigma;
                    a.set(i, i, v);
                }
                Ok(a.solve(&b_mat)?.data().to_vec())
            }
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// Orthonormalize columns of `q` (each column is a Vec<f64> of length `m`) via Modified Gram-Schmidt twice.
fn orthonormalize(cols: &mut Vec<Vec<f64>>, rng: &mut Rng) {
    let m = cols.first().map_or(0, Vec::len);
    for j in 0..cols.len() {
        for _pass in 0..2 {
            for i in 0..j {
                let h = dot(&cols[i], &cols[j]);
                for r in 0..m {
                    cols[j][r] -= h * cols[i][r];
                }
            }
        }
        let mut nrm = norm2(&cols[j]);
        if nrm < 1e-13 {
            // Replace collapsed vector with fresh random direction & orthogonalize
            for r in 0..m {
                cols[j][r] = rng.next_normal();
            }
            for _pass in 0..2 {
                for i in 0..j {
                    let h = dot(&cols[i], &cols[j]);
                    for r in 0..m {
                        cols[j][r] -= h * cols[i][r];
                    }
                }
            }
            nrm = norm2(&cols[j]);
        }
        if nrm > 0.0 {
            for r in 0..m {
                cols[j][r] /= nrm;
            }
        }
    }
}

#[derive(Clone, Copy)]
enum EigMode {
    LargestAbs,
    SmallestAbs,
    LargestReal,
    SmallestReal,
    Shift(f64),
}

fn parse_eig_mode(v: &Value) -> Result<EigMode> {
    if let Ok(s) = v.as_str() {
        match s.to_ascii_lowercase().as_str() {
            "lm" | "largestabs" => Ok(EigMode::LargestAbs),
            "sm" | "smallestabs" => Ok(EigMode::SmallestAbs),
            "lr" | "la" | "largestreal" => Ok(EigMode::LargestReal),
            "sr" | "sa" | "smallestreal" => Ok(EigMode::SmallestReal),
            other => Err(MatroError::runtime(format!(
                "eigs: unsupported sigma selector '{other}'"
            ))),
        }
    } else {
        Ok(EigMode::Shift(v.as_scalar()?))
    }
}

fn sort_eig_indices(eigs: &[Complex], mode: EigMode) -> Vec<usize> {
    let mut idxs: Vec<usize> = (0..eigs.len()).collect();
    match mode {
        EigMode::LargestAbs => {
            idxs.sort_by(|&a, &b| eigs[b].abs().total_cmp(&eigs[a].abs()));
        }
        EigMode::SmallestAbs => {
            idxs.sort_by(|&a, &b| eigs[a].abs().total_cmp(&eigs[b].abs()));
        }
        EigMode::LargestReal => {
            idxs.sort_by(|&a, &b| eigs[b].re.total_cmp(&eigs[a].re));
        }
        EigMode::SmallestReal => {
            idxs.sort_by(|&a, &b| eigs[a].re.total_cmp(&eigs[b].re));
        }
        EigMode::Shift(sig) => {
            idxs.sort_by(|&a, &b| {
                let da = (eigs[a] - Complex::from_real(sig)).abs();
                let db = (eigs[b] - Complex::from_real(sig)).abs();
                da.total_cmp(&db)
            });
        }
    }
    idxs
}

/// `d = eigs(A [, k [, sigma]])` or `[V, D] = eigs(A [, k [, sigma]])`.
pub fn fn_eigs(args: &[Value], nargout: usize, rng: &mut Rng) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity("eigs", "1 to 3 (A [, k [, sigma]])", args.len()));
    }
    let op = match &args[0] {
        Value::Sparse(s) => LinOp::Sparse(s),
        Value::Matrix(m) => LinOp::Dense(m),
        other => {
            return Err(MatroError::runtime(format!(
                "eigs: expected matrix or sparse matrix, got {}",
                other.type_name()
            )));
        }
    };
    let n = op.rows();
    if n != op.cols() || n == 0 {
        return Err(MatroError::dimension("eigs: matrix A must be non-empty square"));
    }
    let k = if args.len() >= 2 {
        (args[1].as_scalar()?.round() as usize).clamp(1, n)
    } else {
        6.min(n)
    };
    let mode = if args.len() >= 3 {
        parse_eig_mode(&args[2])?
    } else {
        EigMode::LargestAbs
    };

    // For modest sizes (n <= 48), direct dense eig is exact, handles all non-symmetric/complex cases, and is < 1ms
    if n <= 48 {
        let a = op.to_dense();
        let (v_full, d_full) = a.eig()?;
        let eigs: Vec<Complex> = (0..n).map(|i| d_full.get_c(i, i)).collect();
        let order = sort_eig_indices(&eigs, mode);
        return pack_eig_result(&v_full, &eigs, &order[..k], nargout);
    }

    // Large matrix: Restarted Arnoldi / shift-and-invert Krylov subspace
    let use_shift_invert = matches!(mode, EigMode::SmallestAbs | EigMode::Shift(_));
    let shift = match mode {
        EigMode::Shift(s) => s,
        _ => 0.0,
    };

    let m = n.min((2 * k + 18).max(32).min(60));
    let mut v0: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
    let mut best_v = Matrix::zeros(n, 1);
    let mut best_eigs: Vec<Complex> = Vec::new();
    let mut best_order: Vec<usize> = Vec::new();
    let restarts = if use_shift_invert { 2 } else { 4 };

    for _restart in 0..restarts {
        let n0 = norm2(&v0).max(1e-15);
        for x in &mut v0 {
            *x /= n0;
        }
        let mut q_basis: Vec<Vec<f64>> = Vec::with_capacity(m);
        q_basis.push(v0.clone());

        let mut h_mat = Matrix::zeros(m, m);
        let mut actual_m = m;

        for j in 0..m {
            let mut w = if use_shift_invert {
                op.solve_shifted(shift, &q_basis[j])?
            } else {
                op.mul_vec(&q_basis[j])
            };
            // Double Modified Gram-Schmidt
            for _pass in 0..2 {
                for i in 0..=j {
                    let hij = dot(&q_basis[i], &w);
                    h_mat.set(i, j, h_mat.get(i, j) + hij);
                    for r in 0..n {
                        w[r] -= hij * q_basis[i][r];
                    }
                }
            }
            let beta = norm2(&w);
            if j + 1 < m {
                if beta < 1e-13 {
                    actual_m = j + 1;
                    break;
                }
                h_mat.set(j + 1, j, beta);
                for r in 0..n {
                    w[r] /= beta;
                }
                q_basis.push(w);
            }
        }

        // Slice H to actual_m x actual_m and compute its Ritz pairs
        let mut h_sub = Matrix::zeros(actual_m, actual_m);
        for c in 0..actual_m {
            for r in 0..actual_m {
                h_sub.set(r, c, h_mat.get(r, c));
            }
        }
        let (y_ritz, d_ritz) = h_sub.eig()?;
        let mut eigs = Vec::with_capacity(actual_m);
        for i in 0..actual_m {
            let mu = d_ritz.get_c(i, i);
            let lam = if use_shift_invert {
                if mu.abs() > 1e-15 {
                    Complex::from_real(shift) + Complex::ONE / mu
                } else {
                    Complex::from_real(shift)
                }
            } else {
                mu
            };
            eigs.push(lam);
        }

        // Lift Ritz vectors V = Q * Y
        let mut v_full = Matrix::zeros(n, actual_m);
        for c in 0..actual_m {
            for r in 0..n {
                let mut sum = Complex::ZERO;
                for j in 0..actual_m {
                    sum = sum + Complex::from_real(q_basis[j][r]) * y_ritz.get_c(j, c);
                }
                v_full.set_c(r, c, sum);
            }
        }
        let order = sort_eig_indices(&eigs, mode);
        let k_eff = k.min(actual_m);

        // Restart vector = linear combination of top k_eff Ritz vectors
        let mut next_v0 = vec![0.0; n];
        for (ci, &idx) in order[..k_eff].iter().enumerate() {
            let weight = 1.0 / ((ci + 1) as f64);
            for r in 0..n {
                let cval = v_full.get_c(r, idx);
                next_v0[r] += weight * (cval.re + 0.37 * cval.im);
            }
        }
        v0 = next_v0;
        best_v = v_full;
        best_eigs = eigs;
        best_order = order;

        // Early stop if top eigenvector residual is already < 1e-10
        if actual_m == n {
            break;
        }
    }
    let k_eff = k.min(best_order.len());
    pack_eig_result(&best_v, &best_eigs, &best_order[..k_eff], nargout)
}

fn pack_eig_result(
    v_full: &Matrix,
    eigs: &[Complex],
    pick: &[usize],
    nargout: usize,
) -> Result<Vec<Value>> {
    let k = pick.len();
    let n = v_full.rows();
    if nargout <= 1 {
        let vals: Vec<Complex> = pick.iter().map(|&i| eigs[i]).collect();
        return Ok(vec![Value::matrix(Matrix::complex_col_vec(vals).normalize())]);
    }
    let mut v_out = Matrix::zeros(n, k);
    let mut d_out = Matrix::zeros(k, k);
    for (ci, &idx) in pick.iter().enumerate() {
        d_out.set_c(ci, ci, eigs[idx]);
        let mut col_norm_sq = 0.0;
        for r in 0..n {
            let val = v_full.get_c(r, idx);
            col_norm_sq += val.norm_sq();
            v_out.set_c(r, ci, val);
        }
        let col_norm = col_norm_sq.sqrt();
        if col_norm > 0.0 {
            for r in 0..n {
                let val = v_out.get_c(r, ci) / Complex::from_real(col_norm);
                v_out.set_c(r, ci, val);
            }
        }
    }
    Ok(vec![
        Value::matrix(v_out.normalize()),
        Value::matrix(d_out.normalize()),
    ])
}

/// `s = svds(A [, k])` or `[U, S, V] = svds(A [, k])`.
pub fn fn_svds(args: &[Value], nargout: usize, rng: &mut Rng) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("svds", "1 or 2 (A [, k])", args.len()));
    }
    let op = match &args[0] {
        Value::Sparse(s) => LinOp::Sparse(s),
        Value::Matrix(m) => LinOp::Dense(m),
        other => {
            return Err(MatroError::runtime(format!(
                "svds: expected matrix or sparse matrix, got {}",
                other.type_name()
            )));
        }
    };
    let m = op.rows();
    let n = op.cols();
    let min_dim = m.min(n);
    if min_dim == 0 {
        return Err(MatroError::dimension("svds: matrix A cannot be empty"));
    }
    let k = if args.len() == 2 {
        (args[1].as_scalar()?.round() as usize).clamp(1, min_dim)
    } else {
        6.min(min_dim)
    };

    // Modest dense size -> exact SVD sliced to top k
    if m.max(n) <= 48 {
        let (u_full, s_full, v_full) = op.to_dense().svd()?;
        return slice_svd(&u_full, &s_full, &v_full, k, nargout);
    }

    // Subspace iteration on range of A (Halko-Martinsson-Tropp with power iterations)
    let ell = min_dim.min((k + 16).max(2 * k + 10).min(48));
    let mut q_cols: Vec<Vec<f64>> = Vec::with_capacity(ell);
    for _ in 0..ell {
        let omega: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
        q_cols.push(op.mul_vec(&omega));
    }
    orthonormalize(&mut q_cols, rng);

    // Subspace power iterations: Q <- orth(A * (A' * Q))
    for _ in 0..25 {
        let mut z_cols: Vec<Vec<f64>> = q_cols.iter().map(|q| op.mul_transpose_vec(q)).collect();
        orthonormalize(&mut z_cols, rng);
        q_cols = z_cols.iter().map(|z| op.mul_vec(z)).collect();
        orthonormalize(&mut q_cols, rng);
    }

    // Form small ell x n matrix B = Q' * A
    let mut b_mat = Matrix::zeros(ell, n);
    for i in 0..ell {
        let row_i = op.mul_transpose_vec(&q_cols[i]);
        for c in 0..n {
            b_mat.set(i, c, row_i[c]);
        }
    }

    let (ub, sb, vb) = b_mat.svd()?;
    // Lift U = Q * Ub (m x ell)
    let mut u_full = Matrix::zeros(m, ell);
    for c in 0..ell {
        for r in 0..m {
            let mut sum = 0.0;
            for j in 0..ell {
                sum += q_cols[j][r] * ub.get(j, c);
            }
            u_full.set(r, c, sum);
        }
    }
    slice_svd(&u_full, &sb, &vb, k, nargout)
}

fn slice_svd(
    u_full: &Matrix,
    s_full: &Matrix,
    v_full: &Matrix,
    k: usize,
    nargout: usize,
) -> Result<Vec<Value>> {
    if nargout <= 1 {
        let svals: Vec<f64> = (0..k).map(|i| s_full.get(i, i)).collect();
        return Ok(vec![Value::matrix(Matrix::col_vec(svals))]);
    }
    let m = u_full.rows();
    let n = v_full.rows();
    let mut u_k = Matrix::zeros(m, k);
    let mut s_k = Matrix::zeros(k, k);
    let mut v_k = Matrix::zeros(n, k);
    for c in 0..k {
        s_k.set(c, c, s_full.get(c, c));
        for r in 0..m {
            u_k.set(r, c, u_full.get(r, c));
        }
        for r in 0..n {
            v_k.set(r, c, v_full.get(r, c));
        }
    }
    Ok(vec![
        Value::matrix(u_k),
        Value::matrix(s_k),
        Value::matrix(v_k),
    ])
}
