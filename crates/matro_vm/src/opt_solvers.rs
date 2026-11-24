//! Callback-driven nonlinear Optimization Toolbox solvers:
//! - `fminunc`: Quasi-Newton BFGS with central finite-difference gradient & Armijo backtracking
//! - `fmincon`: Augmented Lagrangian + Box-Projected BFGS supporting linear inequalities,
//!   linear equalities, box bounds (`lb`/`ub`), and nonlinear constraints (`[c, ceq] = nonlcon(x)`)

use matro_types::{ClosureHandle, MatroError, Matrix, Result, Value};

use crate::Vm;

fn eval_scalar_obj(
    vm: &mut Vm,
    cl: &ClosureHandle,
    x: &[f64],
    is_row: bool,
    stdout: &mut String,
) -> Result<f64> {
    let x_val = if x.len() == 1 {
        Value::scalar(x[0])
    } else if is_row {
        Value::matrix(Matrix::row_vec(x.to_vec()))
    } else {
        Value::matrix(Matrix::col_vec(x.to_vec()))
    };
    let rets = vm.call_closure(cl, &[x_val], 1, stdout)?;
    rets.into_iter()
        .next()
        .unwrap_or_else(Value::empty)
        .as_scalar()
}

fn eval_nonlcon(
    vm: &mut Vm,
    cl: &ClosureHandle,
    x: &[f64],
    is_row: bool,
    stdout: &mut String,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let x_val = if x.len() == 1 {
        Value::scalar(x[0])
    } else if is_row {
        Value::matrix(Matrix::row_vec(x.to_vec()))
    } else {
        Value::matrix(Matrix::col_vec(x.to_vec()))
    };
    let rets = vm.call_closure(cl, &[x_val], 2, stdout)?;
    let c = rets
        .first()
        .and_then(|v| v.to_dense_matrix().ok())
        .map(|m| m.data().to_vec())
        .unwrap_or_default();
    let ceq = rets
        .get(1)
        .and_then(|v| v.to_dense_matrix().ok())
        .map(|m| m.data().to_vec())
        .unwrap_or_default();
    Ok((c, ceq))
}

fn pack_x(x: Vec<f64>, is_row: bool) -> Matrix {
    if x.len() == 1 {
        Matrix::scalar(x[0])
    } else if is_row {
        Matrix::row_vec(x)
    } else {
        Matrix::col_vec(x)
    }
}

/// Quasi-Newton BFGS unconstrained minimizer: `[x, fval] = fminunc(fun, x0)`.
pub fn solve_fminunc(
    vm: &mut Vm,
    cl: &ClosureHandle,
    x0: &Matrix,
    stdout: &mut String,
) -> Result<(Matrix, f64)> {
    let n = x0.len();
    if n == 0 {
        return Err(MatroError::dimension("fminunc: x0 cannot be empty"));
    }
    let is_row = x0.rows() == 1 && x0.cols() > 1;
    let mut x = x0.data().to_vec();
    let mut f = eval_scalar_obj(vm, cl, &x, is_row, stdout)?;

    let grad_of = |vm: &mut Vm, x: &[f64], stdout: &mut String| -> Result<Vec<f64>> {
        let mut g = vec![0.0; n];
        for i in 0..n {
            let eps = 1e-6 * x[i].abs().max(1.0);
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            xp[i] += eps;
            xm[i] -= eps;
            let fp = eval_scalar_obj(vm, cl, &xp, is_row, stdout)?;
            let fm = eval_scalar_obj(vm, cl, &xm, is_row, stdout)?;
            g[i] = (fp - fm) / (2.0 * eps);
        }
        Ok(g)
    };

    let mut g = grad_of(vm, &x, stdout)?;
    // Inverse Hessian approximation H = I
    let mut h_inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        h_inv[i][i] = 1.0;
    }

    for _iter in 0..250 {
        let gnorm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if gnorm < 1e-8 {
            break;
        }
        // Search direction p = -H * g
        let mut p = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                p[i] -= h_inv[i][j] * g[j];
            }
        }
        let mut gp: f64 = g.iter().zip(p.iter()).map(|(a, b)| a * b).sum();
        if gp >= 0.0 {
            // Reset H = I and use steepest descent
            for i in 0..n {
                for j in 0..n {
                    h_inv[i][j] = if i == j { 1.0 } else { 0.0 };
                }
                p[i] = -g[i];
            }
            gp = -gnorm * gnorm;
        }

        // Armijo backtracking line search
        let mut alpha = 1.0;
        let mut x_next = x.clone();
        let mut f_next = f;
        let mut accepted = false;
        for _ in 0..30 {
            for i in 0..n {
                x_next[i] = x[i] + alpha * p[i];
            }
            f_next = eval_scalar_obj(vm, cl, &x_next, is_row, stdout)?;
            if f_next.is_finite() && f_next <= f + 1e-4 * alpha * gp {
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted {
            break;
        }
        let g_next = grad_of(vm, &x_next, stdout)?;
        let s: Vec<f64> = (0..n).map(|i| x_next[i] - x[i]).collect();
        let y: Vec<f64> = (0..n).map(|i| g_next[i] - g[i]).collect();
        let sy: f64 = s.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

        if sy > 1e-12 {
            // BFGS update: H_{k+1} = (I - rho s y') H (I - rho y s') + rho s s'
            let rho = 1.0 / sy;
            let mut hy = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    hy[i] += h_inv[i][j] * y[j];
                }
            }
            let yhy: f64 = y.iter().zip(hy.iter()).map(|(a, b)| a * b).sum();
            for i in 0..n {
                for j in 0..n {
                    h_inv[i][j] += (1.0 + rho * yhy) * rho * s[i] * s[j]
                        - rho * (s[i] * hy[j] + hy[i] * s[j]);
                }
            }
        }
        x = x_next;
        f = f_next;
        g = g_next;
    }

    Ok((pack_x(x, is_row), f))
}

fn opt_mat_arg(args: &[Value], idx: usize) -> Result<Option<Matrix>> {
    match args.get(idx) {
        None => Ok(None),
        Some(v) => {
            let m = v.to_dense_matrix()?;
            if m.is_empty() { Ok(None) } else { Ok(Some(m)) }
        }
    }
}

/// Constrained nonlinear optimization via Augmented Lagrangian + Box-Projected Quasi-Newton:
/// `[x, fval] = fmincon(fun, x0, A, b [, Aeq, beq [, lb, ub [, nonlcon]]])`
pub fn solve_fmincon(
    vm: &mut Vm,
    args: &[Value],
    stdout: &mut String,
) -> Result<(Matrix, f64)> {
    if args.len() < 4 {
        return Err(MatroError::arity(
            "fmincon",
            ">= 4 (fun, x0, A, b [, Aeq, beq, lb, ub, nonlcon])",
            args.len(),
        ));
    }
    let Value::Closure(fun_cl) = &args[0] else {
        return Err(MatroError::runtime("fmincon: first argument must be a function handle"));
    };
    let x0 = args[1].to_dense_matrix()?;
    let n = x0.len();
    let is_row = x0.rows() == 1 && x0.cols() > 1;
    let a_in = opt_mat_arg(args, 2)?;
    let b_in = opt_mat_arg(args, 3)?;
    let a_eq = opt_mat_arg(args, 4)?;
    let b_eq = opt_mat_arg(args, 5)?;
    let lb_m = opt_mat_arg(args, 6)?;
    let ub_m = opt_mat_arg(args, 7)?;
    let nonlcon_cl = match args.get(8) {
        Some(Value::Closure(cl)) => Some(cl.clone()),
        _ => None,
    };

    let lb: Vec<f64> = match &lb_m {
        Some(m) if m.len() == n => m.data().to_vec(),
        Some(m) if m.len() == 1 => vec![m.data()[0]; n],
        _ => vec![f64::NEG_INFINITY; n],
    };
    let ub: Vec<f64> = match &ub_m {
        Some(m) if m.len() == n => m.data().to_vec(),
        Some(m) if m.len() == 1 => vec![m.data()[0]; n],
        _ => vec![f64::INFINITY; n],
    };

    let mut x: Vec<f64> = x0
        .data()
        .iter()
        .enumerate()
        .map(|(i, &v)| v.clamp(lb[i], ub[i]))
        .collect();

    // Probe constraint counts
    let m_lin_in = a_in.as_ref().map_or(0, Matrix::rows);
    let m_lin_eq = a_eq.as_ref().map_or(0, Matrix::rows);
    let (m_nl_in, m_nl_eq) = if let Some(ref nl) = nonlcon_cl {
        let (c, ceq) = eval_nonlcon(vm, nl, &x, is_row, stdout)?;
        (c.len(), ceq.len())
    } else {
        (0, 0)
    };

    let mut lam_in = vec![0.0; m_lin_in + m_nl_in];
    let mut lam_eq = vec![0.0; m_lin_eq + m_nl_eq];
    let mut rho = 2.0f64;

    // Collect all inequality constraints c(x) <= 0 and equality constraints ceq(x) = 0
    let eval_constraints = |vm: &mut Vm, x: &[f64], stdout: &mut String| -> Result<(Vec<f64>, Vec<f64>)> {
        let mut c_all = Vec::with_capacity(m_lin_in + m_nl_in);
        let mut eq_all = Vec::with_capacity(m_lin_eq + m_nl_eq);
        if let (Some(a), Some(b)) = (&a_in, &b_in) {
            for i in 0..a.rows() {
                let mut s = -b.data()[i];
                for j in 0..n {
                    s += a.get(i, j) * x[j];
                }
                c_all.push(s);
            }
        }
        if let (Some(ae), Some(be)) = (&a_eq, &b_eq) {
            for i in 0..ae.rows() {
                let mut s = -be.data()[i];
                for j in 0..n {
                    s += ae.get(i, j) * x[j];
                }
                eq_all.push(s);
            }
        }
        if let Some(ref nl) = nonlcon_cl {
            let (cnl, eqnl) = eval_nonlcon(vm, nl, x, is_row, stdout)?;
            c_all.extend(cnl);
            eq_all.extend(eqnl);
        }
        Ok((c_all, eq_all))
    };

    let eval_aug_lag = |vm: &mut Vm,
                        x: &[f64],
                        lam_in: &[f64],
                        lam_eq: &[f64],
                        rho: f64,
                        stdout: &mut String|
     -> Result<f64> {
        let mut val = eval_scalar_obj(vm, fun_cl, x, is_row, stdout)?;
        let (c_all, eq_all) = eval_constraints(vm, x, stdout)?;
        for (i, &ci) in c_all.iter().enumerate() {
            let lam = lam_in.get(i).copied().unwrap_or(0.0);
            let shifted = (lam + rho * ci).max(0.0);
            val += (shifted * shifted - lam * lam) / (2.0 * rho);
        }
        for (i, &ei) in eq_all.iter().enumerate() {
            let lam = lam_eq.get(i).copied().unwrap_or(0.0);
            val += lam * ei + 0.5 * rho * ei * ei;
        }
        Ok(val)
    };

    for _outer in 0..25 {
        // Inner projected BFGS on Augmented Lagrangian subject to lb <= x <= ub
        let mut f_cur = eval_aug_lag(vm, &x, &lam_in, &lam_eq, rho, stdout)?;
        let mut h_inv = vec![vec![0.0; n]; n];
        for i in 0..n {
            h_inv[i][i] = 1.0;
        }

        let eval_grad = |vm: &mut Vm, x: &[f64], lam_in: &[f64], lam_eq: &[f64], rho: f64, stdout: &mut String| -> Result<Vec<f64>> {
            let mut g = vec![0.0; n];
            for i in 0..n {
                let eps = 1e-6 * x[i].abs().max(1.0);
                let mut xp = x.to_vec();
                let mut xm = x.to_vec();
                xp[i] += eps;
                xm[i] -= eps;
                let fp = eval_aug_lag(vm, &xp, lam_in, lam_eq, rho, stdout)?;
                let fm = eval_aug_lag(vm, &xm, lam_in, lam_eq, rho, stdout)?;
                g[i] = (fp - fm) / (2.0 * eps);
            }
            Ok(g)
        };

        let mut g = eval_grad(vm, &x, &lam_in, &lam_eq, rho, stdout)?;
        for _inner in 0..80 {
            // Projected gradient norm check
            let mut pg_norm = 0.0f64;
            for i in 0..n {
                let step_x = (x[i] - g[i]).clamp(lb[i], ub[i]) - x[i];
                pg_norm += step_x * step_x;
            }
            if pg_norm.sqrt() < 1e-7 {
                break;
            }

            // Compute BFGS step on free variables, steepest descent on active box variables
            let mut p = vec![0.0; n];
            for i in 0..n {
                let at_lb = (x[i] - lb[i]).abs() < 1e-11 && g[i] > 0.0;
                let at_ub = (ub[i] - x[i]).abs() < 1e-11 && g[i] < 0.0;
                if at_lb || at_ub {
                    p[i] = 0.0;
                } else {
                    for j in 0..n {
                        p[i] -= h_inv[i][j] * g[j];
                    }
                }
            }
            let gp: f64 = g.iter().zip(p.iter()).map(|(a, b)| a * b).sum();
            if gp >= -1e-14 {
                for i in 0..n {
                    p[i] = -g[i];
                    for j in 0..n {
                        h_inv[i][j] = if i == j { 1.0 } else { 0.0 };
                    }
                }
            }

            let mut alpha = 1.0;
            let mut x_next = x.clone();
            let mut f_next = f_cur;
            let mut accepted = false;
            for _ in 0..25 {
                for i in 0..n {
                    x_next[i] = (x[i] + alpha * p[i]).clamp(lb[i], ub[i]);
                }
                let s_dot_g: f64 = (0..n).map(|i| g[i] * (x_next[i] - x[i])).sum();
                f_next = eval_aug_lag(vm, &x_next, &lam_in, &lam_eq, rho, stdout)?;
                if f_next.is_finite() && f_next <= f_cur + 1e-4 * s_dot_g.min(0.0) {
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
            if !accepted {
                break;
            }
            let g_next = eval_grad(vm, &x_next, &lam_in, &lam_eq, rho, stdout)?;
            let s: Vec<f64> = (0..n).map(|i| x_next[i] - x[i]).collect();
            let y: Vec<f64> = (0..n).map(|i| g_next[i] - g[i]).collect();
            let sy: f64 = s.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
            if sy > 1e-12 {
                let r = 1.0 / sy;
                let mut hy = vec![0.0; n];
                for i in 0..n {
                    for j in 0..n {
                        hy[i] += h_inv[i][j] * y[j];
                    }
                }
                let yhy: f64 = y.iter().zip(hy.iter()).map(|(a, b)| a * b).sum();
                for i in 0..n {
                    for j in 0..n {
                        h_inv[i][j] += (1.0 + r * yhy) * r * s[i] * s[j]
                            - r * (s[i] * hy[j] + hy[i] * s[j]);
                    }
                }
            }
            let step_sz: f64 = s.iter().map(|v| v.abs()).fold(0.0, f64::max);
            x = x_next;
            f_cur = f_next;
            g = g_next;
            if step_sz < 1e-10 {
                break;
            }
        }

        // Update multipliers & check feasibility
        let (c_all, eq_all) = eval_constraints(vm, &x, stdout)?;
        let mut max_viol = 0.0f64;
        for (i, &ci) in c_all.iter().enumerate() {
            if i < lam_in.len() {
                lam_in[i] = (lam_in[i] + rho * ci).max(0.0);
            }
            max_viol = max_viol.max(ci.max(0.0));
        }
        for (i, &ei) in eq_all.iter().enumerate() {
            if i < lam_eq.len() {
                lam_eq[i] += rho * ei;
            }
            max_viol = max_viol.max(ei.abs());
        }
        if max_viol < 1e-6 && (m_lin_in + m_nl_in + m_lin_eq + m_nl_eq == 0 || _outer >= 2) {
            break;
        }
        rho = (rho * 4.0).min(1e7);
    }

    let final_f = eval_scalar_obj(vm, fun_cl, &x, is_row, stdout)?;
    Ok((pack_x(x, is_row), final_f))
}
