//! Optimization Toolbox matrix solvers (`quadprog`, `lsqlin`) — pure zero-dep Rust.

use matro_types::{MatroError, Matrix, Result, Value};

fn opt_mat(args: &[Value], idx: usize) -> Result<Option<Matrix>> {
    match args.get(idx) {
        None => Ok(None),
        Some(v) => {
            let m = v.to_dense_matrix()?;
            if m.is_empty() { Ok(None) } else { Ok(Some(m)) }
        }
    }
}

/// Solve convex QP: `min 0.5*x'*H*x + f'*x`
/// s.t. `A*x <= b`, `Aeq*x == beq`, `lb <= x <= ub`.
pub fn solve_qp_core(
    h_in: &Matrix,
    f_in: &Matrix,
    a_ineq: Option<&Matrix>,
    b_ineq: Option<&Matrix>,
    a_eq: Option<&Matrix>,
    b_eq: Option<&Matrix>,
    lb_m: Option<&Matrix>,
    ub_m: Option<&Matrix>,
) -> Result<(Matrix, f64)> {
    let n = h_in.rows();
    if h_in.cols() != n || f_in.len() != n || n == 0 {
        return Err(MatroError::dimension("quadprog: H must be n x n and f must have length n"));
    }
    // Symmetrize H
    let mut h = vec![vec![0.0; n]; n];
    for r in 0..n {
        for c in 0..n {
            h[r][c] = 0.5 * (h_in.get(r, c) + h_in.get(c, r));
        }
    }
    let f = f_in.data();

    let lb: Vec<f64> = match lb_m {
        Some(m) if m.len() == n => m.data().to_vec(),
        Some(m) if m.len() == 1 => vec![m.data()[0]; n],
        _ => vec![f64::NEG_INFINITY; n],
    };
    let ub: Vec<f64> = match ub_m {
        Some(m) if m.len() == n => m.data().to_vec(),
        Some(m) if m.len() == 1 => vec![m.data()[0]; n],
        _ => vec![f64::INFINITY; n],
    };

    let m_ineq = a_ineq.map_or(0, Matrix::rows);
    let m_eq = a_eq.map_or(0, Matrix::rows);
    let has_bounds = lb.iter().any(|v| v.is_finite()) || ub.iter().any(|v| v.is_finite());

    // Exact KKT direct solve when there are no inequality constraints and no box bounds
    if m_ineq == 0 && !has_bounds {
        let tot = n + m_eq;
        let mut kkt = Matrix::zeros(tot, tot);
        let mut rhs = Matrix::zeros(tot, 1);
        for r in 0..n {
            for c in 0..n {
                kkt.set(r, c, h[r][c]);
            }
            rhs.set(r, 0, -f[r]);
        }
        if let (Some(ae), Some(be)) = (a_eq, b_eq) {
            for i in 0..m_eq {
                for c in 0..n {
                    let v = ae.get(i, c);
                    kkt.set(n + i, c, v);
                    kkt.set(c, n + i, v);
                }
                rhs.set(n + i, 0, be.data()[i]);
            }
        }
        if let Ok(sol) = kkt.solve(&rhs) {
            let x: Vec<f64> = (0..n).map(|i| sol.get(i, 0)).collect();
            let fval = eval_qp_obj(&h, f, &x);
            return Ok((Matrix::col_vec(x), fval));
        }
    }

    // Augmented Lagrangian + Projected Nesterov Accelerated Gradient (FISTA)
    let mut x = vec![0.0f64; n];
    for i in 0..n {
        x[i] = x[i].clamp(lb[i], ub[i]);
    }
    let mut lam_eq = vec![0.0; m_eq];
    let mut lam_in = vec![0.0; m_ineq];
    let mut rho = 1.0f64;

    // Estimate Frobenius norm of H for safe Lipschitz step size
    let h_norm: f64 = h
        .iter()
        .flat_map(|row| row.iter())
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt()
        .max(1e-6);
    let a_in_norm2: f64 = a_ineq.map_or(0.0, |a| a.data().iter().map(|v| v * v).sum());
    let a_eq_norm2: f64 = a_eq.map_or(0.0, |a| a.data().iter().map(|v| v * v).sum());

    for _outer in 0..35 {
        let lipschitz = (h_norm + rho * (a_in_norm2 + a_eq_norm2)).max(1.0);
        let step = 1.0 / lipschitz;

        let mut y = x.clone();
        let mut t_nes = 1.0f64;

        for _inner in 0..400 {
            // Gradient of Augmented Lagrangian at y
            let mut g = vec![0.0; n];
            for r in 0..n {
                let mut hv = f[r];
                for c in 0..n {
                    hv += h[r][c] * y[c];
                }
                g[r] = hv;
            }
            if let (Some(ae), Some(be)) = (a_eq, b_eq) {
                for i in 0..m_eq {
                    let mut viol = -be.data()[i];
                    for c in 0..n {
                        viol += ae.get(i, c) * y[c];
                    }
                    let coef = lam_eq[i] + rho * viol;
                    for c in 0..n {
                        g[c] += coef * ae.get(i, c);
                    }
                }
            }
            if let (Some(ai), Some(bi)) = (a_ineq, b_ineq) {
                for i in 0..m_ineq {
                    let mut viol = -bi.data()[i];
                    for c in 0..n {
                        viol += ai.get(i, c) * y[c];
                    }
                    let active = (lam_in[i] + rho * viol).max(0.0);
                    if active > 0.0 {
                        for c in 0..n {
                            g[c] += active * ai.get(i, c);
                        }
                    }
                }
            }

            let mut x_next = vec![0.0; n];
            let mut diff_norm = 0.0f64;
            for i in 0..n {
                x_next[i] = (y[i] - step * g[i]).clamp(lb[i], ub[i]);
                diff_norm = diff_norm.max((x_next[i] - x[i]).abs());
            }
            // Adaptive restart if momentum opposes gradient
            let mut dot_dir = 0.0;
            for i in 0..n {
                dot_dir += (y[i] - x_next[i]) * (x_next[i] - x[i]);
            }
            if dot_dir > 0.0 {
                t_nes = 1.0;
                y = x_next.clone();
            } else {
                let t_next = 0.5 * (1.0 + (1.0 + 4.0 * t_nes * t_nes).sqrt());
                let beta = (t_nes - 1.0) / t_next;
                for i in 0..n {
                    y[i] = x_next[i] + beta * (x_next[i] - x[i]);
                }
                t_nes = t_next;
            }
            x = x_next;
            if diff_norm < 1e-10 {
                break;
            }
        }

        // Dual multiplier update & constraint violation check
        let mut max_viol = 0.0f64;
        if let (Some(ae), Some(be)) = (a_eq, b_eq) {
            for i in 0..m_eq {
                let mut viol = -be.data()[i];
                for c in 0..n {
                    viol += ae.get(i, c) * x[c];
                }
                lam_eq[i] += rho * viol;
                max_viol = max_viol.max(viol.abs());
            }
        }
        if let (Some(ai), Some(bi)) = (a_ineq, b_ineq) {
            for i in 0..m_ineq {
                let mut viol = -bi.data()[i];
                for c in 0..n {
                    viol += ai.get(i, c) * x[c];
                }
                lam_in[i] = (lam_in[i] + rho * viol).max(0.0);
                max_viol = max_viol.max(viol.max(0.0));
            }
        }
        if max_viol < 1e-7 {
            break;
        }
        rho = (rho * 3.0).min(1e7);
    }

    let fval = eval_qp_obj(&h, f, &x);
    Ok((Matrix::col_vec(x), fval))
}

fn eval_qp_obj(h: &[Vec<f64>], f: &[f64], x: &[f64]) -> f64 {
    let n = x.len();
    let mut q = 0.0;
    for r in 0..n {
        let mut hx = 0.0;
        for c in 0..n {
            hx += h[r][c] * x[c];
        }
        q += 0.5 * x[r] * hx + f[r] * x[r];
    }
    q
}

/// `[x, fval] = quadprog(H, f [, A, b [, Aeq, beq [, lb, ub]]])`
pub fn fn_quadprog(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() < 2 {
        return Err(MatroError::arity(
            "quadprog",
            ">= 2 (H, f [, A, b, Aeq, beq, lb, ub])",
            args.len(),
        ));
    }
    let h = args[0].to_dense_matrix()?;
    let f = args[1].to_dense_matrix()?;
    let a = opt_mat(args, 2)?;
    let b = opt_mat(args, 3)?;
    let aeq = opt_mat(args, 4)?;
    let beq = opt_mat(args, 5)?;
    let lb = opt_mat(args, 6)?;
    let ub = opt_mat(args, 7)?;
    let (x, fval) = solve_qp_core(
        &h,
        &f,
        a.as_ref(),
        b.as_ref(),
        aeq.as_ref(),
        beq.as_ref(),
        lb.as_ref(),
        ub.as_ref(),
    )?;
    let mut out = vec![Value::matrix(x)];
    if nargout >= 2 {
        out.push(Value::scalar(fval));
    }
    if nargout >= 3 {
        out.push(Value::scalar(1.0));
    }
    Ok(out)
}

/// `[x, resnorm, residual, exitflag] = lsqlin(C, d [, A, b [, Aeq, beq [, lb, ub]]])`
pub fn fn_lsqlin(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() < 2 {
        return Err(MatroError::arity(
            "lsqlin",
            ">= 2 (C, d [, A, b, Aeq, beq, lb, ub])",
            args.len(),
        ));
    }
    let c = args[0].to_dense_matrix()?;
    let d_in = args[1].to_dense_matrix()?;
    let d_col = Matrix::col_vec(d_in.data().to_vec());
    let ct = c.transpose();
    let h = ct.matmul(&c)?;
    let f = ct.matmul(&d_col)?.map(|v| -v);
    let a = opt_mat(args, 2)?;
    let b = opt_mat(args, 3)?;
    let aeq = opt_mat(args, 4)?;
    let beq = opt_mat(args, 5)?;
    let lb = opt_mat(args, 6)?;
    let ub = opt_mat(args, 7)?;
    let (x, _) = solve_qp_core(
        &h,
        &f,
        a.as_ref(),
        b.as_ref(),
        aeq.as_ref(),
        beq.as_ref(),
        lb.as_ref(),
        ub.as_ref(),
    )?;
    let resid = c.matmul(&x)?.elementwise(&d_col, |a, b| a - b)?;
    let resnorm: f64 = resid.data().iter().map(|v| v * v).sum();
    let mut out = vec![Value::matrix(x)];
    if nargout >= 2 {
        out.push(Value::scalar(resnorm));
    }
    if nargout >= 3 {
        out.push(Value::matrix(resid));
    }
    if nargout >= 4 {
        out.push(Value::scalar(1.0));
    }
    Ok(out)
}
