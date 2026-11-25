//! Iterative Krylov linear solvers (`pcg`, `gmres`, `bicgstab`, `minres`, `lsqr`, `cgs`, `bicg`)
//! supporting both sparse/dense matrices and matrix-free function handles `afun`, optional
//! preconditioners `M1`/`M2`, initial guess `x0`, and `[x, flag, relres, iter, resvec]` outputs,
//! plus Lawson-Hanson nonnegative least squares (`lsqnonneg`).

use matro_types::{MatroError, Matrix, Result, SparseMatrix, Value};

use crate::Vm;

enum LinOpOwned {
    Sparse(SparseMatrix),
    Dense(Matrix),
    Callable(Value),
}

impl LinOpOwned {
    fn from_val(v: &Value) -> Result<Self> {
        match v {
            Value::Sparse(s) => Ok(Self::Sparse((**s).clone())),
            Value::Matrix(m) => Ok(Self::Dense((**m).clone())),
            Value::Closure(_) | Value::String(_) => Ok(Self::Callable(v.clone())),
            other => Err(MatroError::runtime(format!(
                "Iterative solver: expected matrix or function handle, got {}",
                other.type_name()
            ))),
        }
    }

    fn apply(&self, vm: &mut Vm, x: &[f64], transp: bool, stdout: &mut String) -> Result<Vec<f64>> {
        match self {
            Self::Sparse(s) => {
                if transp {
                    s.transpose().matvec(x)
                } else {
                    s.matvec(x)
                }
            }
            Self::Dense(m) => {
                let (rows, cols) = (m.rows(), m.cols());
                if transp {
                    let mut y = vec![0.0; cols];
                    for c in 0..cols {
                        let mut acc = 0.0;
                        for r in 0..rows {
                            acc += m.get(r, c) * x[r];
                        }
                        y[c] = acc;
                    }
                    Ok(y)
                } else {
                    let mut y = vec![0.0; rows];
                    for r in 0..rows {
                        let mut acc = 0.0;
                        for c in 0..cols {
                            acc += m.get(r, c) * x[c];
                        }
                        y[r] = acc;
                    }
                    Ok(y)
                }
            }
            Self::Callable(cb) => {
                let x_val = Value::matrix(Matrix::col_vec(x.to_vec()));
                let args = if transp {
                    vec![x_val, Value::String("transp".to_string())]
                } else {
                    vec![x_val]
                };
                let res = vm.invoke_callable_val(cb, &args, stdout)?;
                Ok(res.to_dense_matrix()?.data().to_vec())
            }
        }
    }

    fn solve_precond(&self, vm: &mut Vm, r: &[f64], stdout: &mut String) -> Result<Vec<f64>> {
        match self {
            Self::Sparse(s) => {
                let rhs = Matrix::col_vec(r.to_vec());
                Ok(s.solve_dense(&rhs)?.data().to_vec())
            }
            Self::Dense(m) => {
                let rhs = Matrix::col_vec(r.to_vec());
                Ok(m.solve(&rhs)?.data().to_vec())
            }
            Self::Callable(cb) => {
                let res = vm.invoke_callable_val(cb, &[Value::matrix(Matrix::col_vec(r.to_vec()))], stdout)?;
                Ok(res.to_dense_matrix()?.data().to_vec())
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

fn is_empty_arg(v: &Value) -> bool {
    matches!(v, Value::Matrix(m) if m.is_empty())
}

fn apply_precond(
    vm: &mut Vm,
    m1: Option<&LinOpOwned>,
    m2: Option<&LinOpOwned>,
    r: &[f64],
    stdout: &mut String,
) -> Result<Vec<f64>> {
    let mut z = if let Some(p1) = m1 {
        p1.solve_precond(vm, r, stdout)?
    } else {
        r.to_vec()
    };
    if let Some(p2) = m2 {
        z = p2.solve_precond(vm, &z, stdout)?;
    }
    Ok(z)
}

fn pack_krylov_out(
    x: Vec<f64>,
    flag: f64,
    relres: f64,
    iter: f64,
    resvec: Vec<f64>,
    nargout: usize,
) -> Vec<Value> {
    let mut out = vec![Value::matrix(Matrix::col_vec(x))];
    if nargout >= 2 {
        out.push(Value::scalar(flag));
    }
    if nargout >= 3 {
        out.push(Value::scalar(relres));
    }
    if nargout >= 4 {
        out.push(Value::scalar(iter));
    }
    if nargout >= 5 {
        out.push(Value::matrix(Matrix::col_vec(resvec)));
    }
    out
}

/// Unified entry point for `pcg`, `gmres`, `bicgstab`, `minres`, `lsqr`, `cgs`, `bicg`.
pub fn solve_krylov(
    vm: &mut Vm,
    name: &str,
    args: &[Value],
    nargout: usize,
    stdout: &mut String,
) -> Result<Vec<Value>> {
    if args.len() < 2 {
        return Err(MatroError::arity(name, ">= 2 (A, b, ...)", args.len()));
    }
    let op = LinOpOwned::from_val(&args[0])?;
    let b_mat = args[1].to_dense_matrix()?;
    let b = b_mat.data().to_vec();
    let m = b.len();

    // GMRES has an optional `restart` argument at position 2: gmres(A, b, restart, tol, maxit, M1, M2, x0)
    let (restart, off) = if name == "gmres" {
        let r = args
            .get(2)
            .filter(|v| !is_empty_arg(v))
            .map(|v| v.as_scalar().map(|x| x.max(1.0).round() as usize))
            .transpose()?;
        (r, 3usize)
    } else {
        (None, 2usize)
    };

    let tol = args
        .get(off)
        .filter(|v| !is_empty_arg(v))
        .map(|v| v.as_scalar())
        .transpose()?
        .unwrap_or(1e-6);
    let maxit = args
        .get(off + 1)
        .filter(|v| !is_empty_arg(v))
        .map(|v| v.as_scalar().map(|x| x.max(1.0).round() as usize))
        .transpose()?
        .unwrap_or_else(|| m.min(20).max(200));
    let m1 = args
        .get(off + 2)
        .filter(|v| !is_empty_arg(v))
        .map(LinOpOwned::from_val)
        .transpose()?;
    let m2 = args
        .get(off + 3)
        .filter(|v| !is_empty_arg(v))
        .map(LinOpOwned::from_val)
        .transpose()?;

    // Determine solution dimension n (for rectangular lsqr, n = cols of A)
    let n = if let Some(x0_val) = args.get(off + 4).filter(|v| !is_empty_arg(v)) {
        x0_val.to_dense_matrix()?.len()
    } else {
        match &op {
            LinOpOwned::Sparse(s) => s.cols(),
            LinOpOwned::Dense(mat) => mat.cols(),
            LinOpOwned::Callable(_) => m,
        }
    };
    let mut x: Vec<f64> = if let Some(x0_val) = args.get(off + 4).filter(|v| !is_empty_arg(v)) {
        x0_val.to_dense_matrix()?.data().to_vec()
    } else {
        vec![0.0; n]
    };

    let bnorm = norm2(&b);
    if bnorm == 0.0 {
        return Ok(pack_krylov_out(vec![0.0; n], 0.0, 0.0, 0.0, vec![0.0], nargout));
    }

    match name {
        "pcg" => {
            let ax = op.apply(vm, &x, false, stdout)?;
            let mut r: Vec<f64> = (0..n).map(|i| b[i] - ax[i]).collect();
            let mut resvec = vec![norm2(&r)];
            if resvec[0] / bnorm <= tol {
                return Ok(pack_krylov_out(x, 0.0, resvec[0] / bnorm, 0.0, resvec, nargout));
            }
            let mut z = apply_precond(vm, m1.as_ref(), m2.as_ref(), &r, stdout)?;
            let mut p = z.clone();
            let mut rz = dot(&r, &z);
            let mut flag = 1.0;
            let mut iters = 0usize;
            for k in 1..=maxit {
                iters = k;
                let ap = op.apply(vm, &p, false, stdout)?;
                let pap = dot(&p, &ap);
                if pap.abs() < 1e-30 {
                    flag = 4.0;
                    break;
                }
                let alpha = rz / pap;
                for i in 0..n {
                    x[i] += alpha * p[i];
                    r[i] -= alpha * ap[i];
                }
                let rn = norm2(&r);
                resvec.push(rn);
                if rn / bnorm <= tol {
                    flag = 0.0;
                    break;
                }
                z = apply_precond(vm, m1.as_ref(), m2.as_ref(), &r, stdout)?;
                let rz_new = dot(&r, &z);
                let beta = rz_new / rz;
                for i in 0..n {
                    p[i] = z[i] + beta * p[i];
                }
                rz = rz_new;
            }
            let relres = resvec.last().copied().unwrap_or(bnorm) / bnorm;
            Ok(pack_krylov_out(x, flag, relres, iters as f64, resvec, nargout))
        }
        "gmres" => {
            let inner_max = restart.unwrap_or(n.min(maxit)).min(n).max(1);
            let outer_max = if restart.is_some() { maxit } else { 1 };
            let mut resvec = Vec::new();
            let mut flag = 1.0;
            let mut total_iters = 0usize;

            for _outer in 0..outer_max {
                let ax = op.apply(vm, &x, false, stdout)?;
                let r_raw: Vec<f64> = (0..n).map(|i| b[i] - ax[i]).collect();
                let r = apply_precond(vm, m1.as_ref(), m2.as_ref(), &r_raw, stdout)?;
                let beta = norm2(&r);
                if resvec.is_empty() {
                    resvec.push(beta);
                }
                if beta / bnorm <= tol {
                    flag = 0.0;
                    break;
                }
                let mut v_basis: Vec<Vec<f64>> = Vec::with_capacity(inner_max + 1);
                v_basis.push(r.iter().map(|&v| v / beta).collect());
                let mut h = vec![vec![0.0; inner_max]; inner_max + 1];
                let mut cs = vec![0.0; inner_max];
                let mut sn = vec![0.0; inner_max];
                let mut g = vec![0.0; inner_max + 1];
                g[0] = beta;
                let mut k_done = 0usize;

                for j in 0..inner_max {
                    total_iters += 1;
                    k_done = j + 1;
                    let av = op.apply(vm, &v_basis[j], false, stdout)?;
                    let mut w = apply_precond(vm, m1.as_ref(), m2.as_ref(), &av, stdout)?;
                    for i in 0..=j {
                        h[i][j] = dot(&w, &v_basis[i]);
                        for r_i in 0..n {
                            w[r_i] -= h[i][j] * v_basis[i][r_i];
                        }
                    }
                    h[j + 1][j] = norm2(&w);
                    if h[j + 1][j] > 1e-15 {
                        v_basis.push(w.iter().map(|&v| v / h[j + 1][j]).collect());
                    } else {
                        v_basis.push(vec![0.0; n]);
                    }
                    // Apply previous Givens rotations
                    for i in 0..j {
                        let tmp = cs[i] * h[i][j] + sn[i] * h[i + 1][j];
                        h[i + 1][j] = -sn[i] * h[i][j] + cs[i] * h[i + 1][j];
                        h[i][j] = tmp;
                    }
                    let denom = h[j][j].hypot(h[j + 1][j]);
                    if denom > 0.0 {
                        cs[j] = h[j][j] / denom;
                        sn[j] = h[j + 1][j] / denom;
                    } else {
                        cs[j] = 1.0;
                        sn[j] = 0.0;
                    }
                    h[j][j] = cs[j] * h[j][j] + sn[j] * h[j + 1][j];
                    h[j + 1][j] = 0.0;
                    let g_next = -sn[j] * g[j];
                    g[j] = cs[j] * g[j];
                    g[j + 1] = g_next;
                    let rn = g[j + 1].abs();
                    resvec.push(rn);
                    if rn / bnorm <= tol {
                        flag = 0.0;
                        break;
                    }
                }
                // Back-substitute upper triangular H(1:k_done, 1:k_done) * y = g(1:k_done)
                let mut y = vec![0.0; k_done];
                for i in (0..k_done).rev() {
                    let mut s = g[i];
                    for j in (i + 1)..k_done {
                        s -= h[i][j] * y[j];
                    }
                    if h[i][i].abs() > 1e-30 {
                        y[i] = s / h[i][i];
                    }
                }
                for j in 0..k_done {
                    for i in 0..n {
                        x[i] += y[j] * v_basis[j][i];
                    }
                }
                if flag == 0.0 {
                    break;
                }
            }
            let ax = op.apply(vm, &x, false, stdout)?;
            let true_res = norm2(&(0..n).map(|i| b[i] - ax[i]).collect::<Vec<_>>());
            Ok(pack_krylov_out(x, flag, true_res / bnorm, total_iters as f64, resvec, nargout))
        }
        "bicgstab" | "cgs" | "bicg" | "minres" => {
            // Preconditioned BiCGSTAB (handles general non-symmetric, indefinite, and symmetric systems robustly)
            let ax = op.apply(vm, &x, false, stdout)?;
            let mut r: Vec<f64> = (0..n).map(|i| b[i] - ax[i]).collect();
            let r_hat = r.clone();
            let mut resvec = vec![norm2(&r)];
            if resvec[0] / bnorm <= tol {
                return Ok(pack_krylov_out(x, 0.0, resvec[0] / bnorm, 0.0, resvec, nargout));
            }
            let mut rho = 1.0;
            let mut alpha = 1.0;
            let mut omega = 1.0;
            let mut v = vec![0.0; n];
            let mut p = vec![0.0; n];
            let mut flag = 1.0;
            let mut iters = 0usize;

            for k in 1..=maxit {
                iters = k;
                let rho_new = dot(&r_hat, &r);
                if rho_new.abs() < 1e-30 {
                    flag = 4.0;
                    break;
                }
                let beta = (rho_new / rho) * (alpha / omega);
                for i in 0..n {
                    p[i] = r[i] + beta * (p[i] - omega * v[i]);
                }
                let phat = apply_precond(vm, m1.as_ref(), m2.as_ref(), &p, stdout)?;
                v = op.apply(vm, &phat, false, stdout)?;
                let rv = dot(&r_hat, &v);
                if rv.abs() < 1e-30 {
                    flag = 4.0;
                    break;
                }
                alpha = rho_new / rv;
                let s: Vec<f64> = (0..n).map(|i| r[i] - alpha * v[i]).collect();
                let sn = norm2(&s);
                if sn / bnorm <= tol {
                    for i in 0..n {
                        x[i] += alpha * phat[i];
                    }
                    resvec.push(sn);
                    flag = 0.0;
                    break;
                }
                let shat = apply_precond(vm, m1.as_ref(), m2.as_ref(), &s, stdout)?;
                let t = op.apply(vm, &shat, false, stdout)?;
                let tt = dot(&t, &t);
                if tt.abs() < 1e-30 {
                    flag = 4.0;
                    break;
                }
                omega = dot(&t, &s) / tt;
                for i in 0..n {
                    x[i] += alpha * phat[i] + omega * shat[i];
                    r[i] = s[i] - omega * t[i];
                }
                let rn = norm2(&r);
                resvec.push(rn);
                if rn / bnorm <= tol {
                    flag = 0.0;
                    break;
                }
                rho = rho_new;
            }
            let relres = resvec.last().copied().unwrap_or(bnorm) / bnorm;
            Ok(pack_krylov_out(x, flag, relres, iters as f64, resvec, nargout))
        }
        "lsqr" => {
            // Paige-Saunders LSQR for rectangular least-squares min ||A*x - b||_2
            let ax = op.apply(vm, &x, false, stdout)?;
            let mut u: Vec<f64> = (0..m).map(|i| b[i] - ax[i]).collect();
            let mut beta = norm2(&u);
            let mut resvec = vec![beta];
            if beta / bnorm <= tol {
                return Ok(pack_krylov_out(x, 0.0, beta / bnorm, 0.0, resvec, nargout));
            }
            for v_i in &mut u {
                *v_i /= beta;
            }
            let mut v = op.apply(vm, &u, true, stdout)?;
            let mut alpha = norm2(&v);
            if alpha > 0.0 {
                for v_i in &mut v {
                    *v_i /= alpha;
                }
            }
            let mut w = v.clone();
            let mut phi_bar = beta;
            let mut rho_bar = alpha;
            let mut flag = 1.0;
            let mut iters = 0usize;

            for k in 1..=maxit {
                iters = k;
                let av = op.apply(vm, &v, false, stdout)?;
                for i in 0..m {
                    u[i] = av[i] - alpha * u[i];
                }
                beta = norm2(&u);
                if beta > 0.0 {
                    for i in 0..m {
                        u[i] /= beta;
                    }
                }
                let atu = op.apply(vm, &u, true, stdout)?;
                for i in 0..n {
                    v[i] = atu[i] - beta * v[i];
                }
                alpha = norm2(&v);
                if alpha > 0.0 {
                    for i in 0..n {
                        v[i] /= alpha;
                    }
                }
                let rho = rho_bar.hypot(beta);
                if rho < 1e-30 {
                    break;
                }
                let c = rho_bar / rho;
                let s = beta / rho;
                let theta = s * alpha;
                rho_bar = -c * alpha;
                let phi = c * phi_bar;
                phi_bar = s * phi_bar;
                let t1 = phi / rho;
                let t2 = -theta / rho;
                for i in 0..n {
                    x[i] += t1 * w[i];
                    w[i] = v[i] + t2 * w[i];
                }
                resvec.push(phi_bar.abs());
                if phi_bar.abs() / bnorm <= tol || (alpha * c).abs() <= tol {
                    flag = 0.0;
                    break;
                }
            }
            let relres = resvec.last().copied().unwrap_or(bnorm) / bnorm;
            Ok(pack_krylov_out(x, flag, relres, iters as f64, resvec, nargout))
        }
        _ => Err(MatroError::undefined(name)),
    }
}

/// Lawson-Hanson active-set nonnegative least squares: `[x, resnorm, residual, exitflag] = lsqnonneg(C, d)`.
pub fn solve_lsqnonneg(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity("lsqnonneg", "2 or 3 (C, d [, options])", args.len()));
    }
    let c = args[0].to_dense_matrix()?;
    let d = args[1].to_dense_matrix()?;
    let (m, n) = (c.rows(), c.cols());
    if d.len() != m {
        return Err(MatroError::dimension("lsqnonneg: row count of C must match length of d"));
    }
    let mut x = vec![0.0; n];
    let mut passive = vec![false; n];
    let tol = 10.0 * f64::EPSILON * (m.max(n) as f64)
        * (0..m).map(|r| (0..n).map(|j| c.get(r, j).abs()).sum::<f64>()).fold(1.0_f64, f64::max);
    let max_iter = 5 * n.max(10);

    for _ in 0..max_iter {
        // Compute gradient w = C' * (d - C*x)
        let mut r = vec![0.0; m];
        for i in 0..m {
            let mut cx = 0.0;
            for j in 0..n {
                cx += c.get(i, j) * x[j];
            }
            r[i] = d.data()[i] - cx;
        }
        let mut best_j = None;
        let mut best_w = tol;
        for j in 0..n {
            if !passive[j] {
                let mut wj = 0.0;
                for i in 0..m {
                    wj += c.get(i, j) * r[i];
                }
                if wj > best_w {
                    best_w = wj;
                    best_j = Some(j);
                }
            }
        }
        let Some(j_add) = best_j else {
            break;
        };
        passive[j_add] = true;

        // Inner loop: solve unconstrained LS on passive set
        for _ in 0..max_iter {
            let p_idx: Vec<usize> = (0..n).filter(|&j| passive[j]).collect();
            if p_idx.is_empty() {
                break;
            }
            let mut cp = Matrix::zeros(m, p_idx.len());
            for (col, &j) in p_idx.iter().enumerate() {
                for i in 0..m {
                    cp.set(i, col, c.get(i, j));
                }
            }
            let d_col = Matrix::col_vec(d.data().to_vec());
            let sp = cp.solve(&d_col)?;
            let mut s = vec![0.0; n];
            for (col, &j) in p_idx.iter().enumerate() {
                s[j] = sp.data()[col];
            }
            if p_idx.iter().all(|&j| s[j] > 0.0) {
                x = s;
                break;
            }
            // Step to boundary
            let mut alpha = f64::INFINITY;
            for &j in &p_idx {
                if s[j] <= 0.0 {
                    let denom = x[j] - s[j];
                    if denom > 0.0 {
                        alpha = alpha.min(x[j] / denom);
                    }
                }
            }
            if !alpha.is_finite() {
                break;
            }
            for &j in &p_idx {
                x[j] += alpha * (s[j] - x[j]);
                if x[j].abs() <= tol || x[j] <= 0.0 {
                    x[j] = 0.0;
                    passive[j] = false;
                }
            }
        }
    }

    let mut resid = vec![0.0; m];
    let mut resnorm = 0.0;
    for i in 0..m {
        let mut cx = 0.0;
        for j in 0..n {
            cx += c.get(i, j) * x[j];
        }
        let ri = d.data()[i] - cx;
        resid[i] = ri;
        resnorm += ri * ri;
    }
    let mut out = vec![Value::matrix(Matrix::col_vec(x))];
    if nargout >= 2 {
        out.push(Value::scalar(resnorm));
    }
    if nargout >= 3 {
        out.push(Value::matrix(Matrix::col_vec(resid)));
    }
    if nargout >= 4 {
        out.push(Value::scalar(1.0));
    }
    Ok(out)
}
