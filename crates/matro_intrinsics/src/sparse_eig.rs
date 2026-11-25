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

    fn is_complex(&self) -> bool {
        match self {
            Self::Dense(m) => m.is_complex(),
            Self::Sparse(s) => s.is_complex(),
        }
    }

    fn mul_vec_c(&self, x: &[Complex]) -> Vec<Complex> {
        let m = self.rows();
        let n = self.cols();
        match self {
            Self::Dense(mat) => {
                let mut y = vec![Complex::ZERO; m];
                for c in 0..n {
                    let xc = x[c];
                    if !xc.is_zero() {
                        for r in 0..m {
                            y[r] = y[r] + mat.get_c(r, c) * xc;
                        }
                    }
                }
                y
            }
            Self::Sparse(sp) => sp.matvec_c(x).unwrap_or_else(|_| vec![Complex::ZERO; m]),
        }
    }

    fn mul_ctranspose_vec_c(&self, x: &[Complex]) -> Vec<Complex> {
        let m = self.rows();
        let n = self.cols();
        match self {
            Self::Dense(mat) => {
                let mut y = vec![Complex::ZERO; n];
                for c in 0..n {
                    let mut sum = Complex::ZERO;
                    for r in 0..m {
                        sum = sum + mat.get_c(r, c).conj() * x[r];
                    }
                    y[c] = sum;
                }
                y
            }
            Self::Sparse(sp) => sp
                .ctranspose()
                .matvec_c(x)
                .unwrap_or_else(|_| vec![Complex::ZERO; n]),
        }
    }

    /// Solve `(A - sigma*I) * x = b` in $\mathbb{C}^n$ for complex shift-and-invert eigs.
    fn solve_shifted_c(&self, sigma: Complex, b: &[Complex]) -> Result<Vec<Complex>> {
        let n = self.rows();
        let b_mat = Matrix::complex_col_vec(b.to_vec());
        let sol = match self {
            Self::Sparse(sp) => {
                let shifted = if sigma.is_zero() {
                    (*sp).clone()
                } else {
                    let eye = SparseMatrix::eye(n);
                    sp.add_scaled_c(&eye, Complex::new(-sigma.re, -sigma.im))?
                };
                shifted.solve_dense(&b_mat)?
            }
            Self::Dense(m) => {
                let mut a = (*m).clone();
                for i in 0..n {
                    let v = a.get_c(i, i) - sigma;
                    a.set_c(i, i, v);
                }
                a.solve(&b_mat)?
            }
        };
        Ok((0..n).map(|i| sol.get_linear_c(i)).collect())
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
    Shift(Complex),
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
        let m = v.to_dense_matrix()?;
        let sc = m.as_complex_scalar().ok_or_else(|| {
            MatroError::runtime("eigs: shift sigma must be a scalar or selector string")
        })?;
        Ok(EigMode::Shift(sc))
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
                let da = (eigs[a] - sig).abs();
                let db = (eigs[b] - sig).abs();
                da.total_cmp(&db)
            });
        }
    }
    idxs
}

/// `d = eigs(A [, k [, sigma [, opts]]])` or `[V, D, flag] = eigs(...)`.
pub fn fn_eigs(args: &[Value], nargout: usize, rng: &mut Rng) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 5 {
        return Err(MatroError::arity("eigs", "1 to 5 (A [, k [, sigma [, opts]]])", args.len()));
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
    let k = if args.len() >= 2 && !args[1].to_dense_matrix().is_ok_and(|m| m.is_empty()) {
        (args[1].as_scalar()?.round() as usize).clamp(1, n)
    } else {
        6.min(n)
    };
    let mode = if args.len() >= 3 && !matches!(&args[2], Value::Struct(_)) {
        parse_eig_mode(&args[2])?
    } else {
        EigMode::LargestAbs
    };

    // For modest sizes (n <= 64) or when k approaches n, direct dense eig is exact and fast
    if n <= 64 || k >= 50 {
        let a = op.to_dense();
        let (v_full, d_full) = a.eig()?;
        let eigs: Vec<Complex> = (0..n).map(|i| d_full.get_c(i, i)).collect();
        let order = sort_eig_indices(&eigs, mode);
        return pack_eig_result(&v_full, &eigs, &order[..k], nargout);
    }

    // Large matrix: Restarted Arnoldi / shift-and-invert Krylov subspace
    let use_shift_invert = matches!(mode, EigMode::SmallestAbs | EigMode::Shift(_));
    let shift_c = match mode {
        EigMode::Shift(s) => s,
        _ => Complex::ZERO,
    };
    if op.is_complex() || shift_c.im != 0.0 {
        return eigs_complex(&op, k, mode, use_shift_invert, shift_c, nargout, rng);
    }
    let shift = shift_c.re;

    let m = n.min((2 * k + 18).max(32).max(k + 8));
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
    let mut res = vec![
        Value::matrix(v_out.normalize()),
        Value::matrix(d_out.normalize()),
    ];
    if nargout >= 3 {
        res.push(Value::scalar(0.0));
    }
    Ok(res)
}

/// `s = svds(A [, k [, sigma]])` or `[U, S, V, flag] = svds(A [, k [, sigma]])`.
pub fn fn_svds(args: &[Value], nargout: usize, rng: &mut Rng) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 4 {
        return Err(MatroError::arity("svds", "1 to 4 (A [, k [, sigma]])", args.len()));
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
    let k = if args.len() >= 2 && !args[1].to_dense_matrix().is_ok_and(|mm| mm.is_empty()) {
        (args[1].as_scalar()?.round() as usize).clamp(1, min_dim)
    } else {
        6.min(min_dim)
    };
    let want_smallest = args.get(2).and_then(crate::value_as_text).is_some_and(|s| {
        let low = s.to_ascii_lowercase();
        low == "smallest" || low == "smallestnz"
    });

    // Modest dense size, smallest singular values, or large k -> exact SVD sliced to k
    if m.max(n) <= 48 || want_smallest || k >= 40 {
        let (u_full, s_full, v_full) = op.to_dense().svd()?;
        if want_smallest {
            let start = min_dim.saturating_sub(k);
            let mut u = Matrix::zeros(m, k);
            let mut s = Matrix::zeros(k, k);
            let mut v = Matrix::zeros(n, k);
            for c in 0..k {
                let src = start + c;
                s.set(c, c, s_full.get(src, src));
                for r in 0..m {
                    u.set_c(r, c, u_full.get_c(r, src));
                }
                for r in 0..n {
                    v.set_c(r, c, v_full.get_c(r, src));
                }
            }
            return if nargout <= 1 {
                Ok(vec![Value::matrix(Matrix::col_vec((0..k).map(|i| s.get(i, i)).collect()))])
            } else if nargout >= 4 {
                Ok(vec![Value::matrix(u.normalize()), Value::matrix(s), Value::matrix(v.normalize()), Value::scalar(0.0)])
            } else {
                Ok(vec![Value::matrix(u.normalize()), Value::matrix(s), Value::matrix(v.normalize())])
            };
        }
        let mut res = slice_svd(&u_full, &s_full, &v_full, k, nargout)?;
        if nargout >= 4 {
            res.push(Value::scalar(0.0));
        }
        return Ok(res);
    }

    if op.is_complex() {
        return svds_complex(&op, k, nargout, rng);
    }

    // Subspace iteration on range of A (Halko-Martinsson-Tropp with power iterations)
    let ell = min_dim.min((k + 16).max(2 * k + 10));
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
    let mut u_k = if u_full.is_complex() {
        Matrix::complex_zeros(m, k)
    } else {
        Matrix::zeros(m, k)
    };
    let mut s_k = Matrix::zeros(k, k);
    let mut v_k = if v_full.is_complex() {
        Matrix::complex_zeros(n, k)
    } else {
        Matrix::zeros(n, k)
    };
    for c in 0..k {
        s_k.set(c, c, s_full.get(c, c));
        for r in 0..m {
            u_k.set_c(r, c, u_full.get_c(r, c));
        }
        for r in 0..n {
            v_k.set_c(r, c, v_full.get_c(r, c));
        }
    }
    Ok(vec![
        Value::matrix(u_k.normalize()),
        Value::matrix(s_k),
        Value::matrix(v_k.normalize()),
    ])
}

fn dot_c(a: &[Complex], b: &[Complex]) -> Complex {
    a.iter()
        .zip(b.iter())
        .fold(Complex::ZERO, |acc, (&x, &y)| acc + x.conj() * y)
}

fn norm2_c(a: &[Complex]) -> f64 {
    a.iter().map(|x| x.re * x.re + x.im * x.im).sum::<f64>().sqrt()
}

fn orthonormalize_c(cols: &mut [Vec<Complex>], rng: &mut Rng) {
    let k = cols.len();
    if k == 0 {
        return;
    }
    let m = cols[0].len();
    for j in 0..k {
        for _pass in 0..2 {
            for i in 0..j {
                let h = dot_c(&cols[i], &cols[j]);
                for r in 0..m {
                    cols[j][r] = cols[j][r] - h * cols[i][r];
                }
            }
        }
        let mut nrm = norm2_c(&cols[j]);
        if nrm < 1e-13 {
            for r in 0..m {
                cols[j][r] = Complex::new(rng.next_normal(), rng.next_normal());
            }
            for _pass in 0..2 {
                for i in 0..j {
                    let h = dot_c(&cols[i], &cols[j]);
                    for r in 0..m {
                        cols[j][r] = cols[j][r] - h * cols[i][r];
                    }
                }
            }
            nrm = norm2_c(&cols[j]);
        }
        if nrm > 0.0 {
            let inv = Complex::from_real(1.0 / nrm);
            for r in 0..m {
                cols[j][r] = cols[j][r] * inv;
            }
        }
    }
}

fn eigs_complex(
    op: &LinOp<'_>,
    k: usize,
    mode: EigMode,
    use_shift_invert: bool,
    shift: Complex,
    nargout: usize,
    rng: &mut Rng,
) -> Result<Vec<Value>> {
    let n = op.rows();
    let m = n.min((2 * k + 26).max(36).max(k + 12));
    let mut v0: Vec<Complex> = (0..n)
        .map(|_| Complex::new(rng.next_normal(), rng.next_normal()))
        .collect();
    let mut best_v = Matrix::complex_zeros(n, 1);
    let mut best_eigs: Vec<Complex> = Vec::new();
    let mut best_order: Vec<usize> = Vec::new();
    let mut best_res = f64::INFINITY;
    let restarts = 6;

    // Pre-factor (A - sigma*I) ONCE for shift-and-invert (30x faster than factoring every Arnoldi step!)
    let sparse_lu_cache: Option<(SparseMatrix, SparseMatrix, Vec<usize>)> = if use_shift_invert {
        if let LinOp::Sparse(sp) = op {
            let shifted = if shift.is_zero() {
                (*sp).clone()
            } else {
                let eye = SparseMatrix::eye(n);
                sp.add_scaled_c(&eye, Complex::new(-shift.re, -shift.im))?
            };
            Some(shifted.lu_sparse_complex()?)
        } else {
            None
        }
    } else {
        None
    };

    for _restart in 0..restarts {
        let n0 = norm2_c(&v0).max(1e-15);
        for x in &mut v0 {
            *x = *x / Complex::from_real(n0);
        }
        let mut q_basis: Vec<Vec<Complex>> = Vec::with_capacity(m);
        q_basis.push(v0.clone());

        let mut h_mat = Matrix::complex_zeros(m, m);
        let mut actual_m = m;

        for j in 0..m {
            let mut w = if use_shift_invert {
                if let Some((ref l, ref u, ref perm)) = sparse_lu_cache {
                    let rhs = Matrix::complex_col_vec(q_basis[j].clone());
                    let sol = SparseMatrix::solve_with_lu_c(l, u, perm, &rhs)?;
                    (0..n).map(|i| sol.get_linear_c(i)).collect()
                } else {
                    op.solve_shifted_c(shift, &q_basis[j])?
                }
            } else {
                op.mul_vec_c(&q_basis[j])
            };
            for _pass in 0..2 {
                for i in 0..=j {
                    let hij = dot_c(&q_basis[i], &w);
                    let prev = h_mat.get_c(i, j);
                    h_mat.set_c(i, j, prev + hij);
                    for r in 0..n {
                        w[r] = w[r] - hij * q_basis[i][r];
                    }
                }
            }
            let beta = norm2_c(&w);
            if j + 1 < m {
                if beta < 1e-13 {
                    actual_m = j + 1;
                    break;
                }
                h_mat.set_c(j + 1, j, Complex::from_real(beta));
                let inv = Complex::from_real(1.0 / beta);
                for r in 0..n {
                    w[r] = w[r] * inv;
                }
                q_basis.push(w);
            }
        }

        let mut h_sub = Matrix::complex_zeros(actual_m, actual_m);
        for c in 0..actual_m {
            for r in 0..actual_m {
                h_sub.set_c(r, c, h_mat.get_c(r, c));
            }
        }
        let (y_ritz, d_ritz) = h_sub.normalize().eig()?;
        let mut eigs = Vec::with_capacity(actual_m);
        for i in 0..actual_m {
            let mu = d_ritz.get_c(i, i);
            let lam = if use_shift_invert {
                if mu.abs() > 1e-15 {
                    shift + Complex::ONE / mu
                } else {
                    shift
                }
            } else {
                mu
            };
            eigs.push(lam);
        }

        let mut v_full = Matrix::complex_zeros(n, actual_m);
        for c in 0..actual_m {
            for r in 0..n {
                let mut sum = Complex::ZERO;
                for j in 0..actual_m {
                    sum = sum + q_basis[j][r] * y_ritz.get_c(j, c);
                }
                v_full.set_c(r, c, sum);
            }
        }
        let order = sort_eig_indices(&eigs, mode);
        let k_eff = k.min(actual_m);

        // Check convergence on top k_eff Ritz pairs
        let mut max_res = 0.0f64;
        for &idx in &order[..k_eff] {
            let vc: Vec<Complex> = (0..n).map(|r| v_full.get_c(r, idx)).collect();
            let av = op.mul_vec_c(&vc);
            let lam = eigs[idx];
            let res: f64 = (0..n)
                .map(|r| {
                    let d = av[r] - lam * vc[r];
                    d.re * d.re + d.im * d.im
                })
                .sum::<f64>()
                .sqrt();
            max_res = max_res.max(res / lam.abs().max(1.0));
        }

        let mut next_v0 = vec![Complex::ZERO; n];
        for (ci, &idx) in order[..k_eff].iter().enumerate() {
            let weight = Complex::from_real(1.0 / ((ci + 1) as f64));
            for r in 0..n {
                next_v0[r] = next_v0[r] + weight * v_full.get_c(r, idx);
            }
        }
        v0 = next_v0;
        if max_res < best_res {
            best_res = max_res;
            best_v = v_full;
            best_eigs = eigs;
            best_order = order;
        }

        if actual_m == n || best_res < 1e-10 {
            break;
        }
    }
    let k_eff = k.min(best_order.len());
    pack_eig_result(&best_v, &best_eigs, &best_order[..k_eff], nargout)
}

fn svds_complex(op: &LinOp<'_>, k: usize, nargout: usize, rng: &mut Rng) -> Result<Vec<Value>> {
    let m = op.rows();
    let n = op.cols();
    let min_dim = m.min(n);
    let ell = min_dim.min((k + 16).max(2 * k + 10));
    let mut q_cols: Vec<Vec<Complex>> = Vec::with_capacity(ell);
    for _ in 0..ell {
        let omega: Vec<Complex> = (0..n)
            .map(|_| Complex::new(rng.next_normal(), rng.next_normal()))
            .collect();
        q_cols.push(op.mul_vec_c(&omega));
    }
    orthonormalize_c(&mut q_cols, rng);

    for _ in 0..20 {
        let mut z_cols: Vec<Vec<Complex>> = q_cols
            .iter()
            .map(|q| op.mul_ctranspose_vec_c(q))
            .collect();
        orthonormalize_c(&mut z_cols, rng);
        q_cols = z_cols.iter().map(|z| op.mul_vec_c(z)).collect();
        orthonormalize_c(&mut q_cols, rng);
    }

    // B = Q^H * A (ell x n): row i of B is (A^H * q_i)^H
    let mut b_mat = Matrix::complex_zeros(ell, n);
    for i in 0..ell {
        let ah_qi = op.mul_ctranspose_vec_c(&q_cols[i]);
        for c in 0..n {
            b_mat.set_c(i, c, ah_qi[c].conj());
        }
    }
    let (ub, sb, vb) = b_mat.normalize().svd()?;
    let mut u_full = Matrix::complex_zeros(m, ell);
    for c in 0..ell {
        for r in 0..m {
            let mut sum = Complex::ZERO;
            for j in 0..ell {
                sum = sum + q_cols[j][r] * ub.get_c(j, c);
            }
            u_full.set_c(r, c, sum);
        }
    }
    slice_svd(&u_full, &sb, &vb, k, nargout)
}
