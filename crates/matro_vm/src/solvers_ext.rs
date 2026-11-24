//! Extended numerical solvers:
//! - `ode23`: Adaptive Bogacki-Shampine RK2(3) with FSAL
//! - `ode113`: Adaptive 4th-order Adams-Bashforth-Moulton PECE multistep solver
//! - `bvpinit`, `bvp4c`, `deval`: 4th-order Lobatto IIIA collocation 2-point BVP solver
//! - `pdepe`: 1D parabolic/elliptic PDE solver via conservative finite-volume method-of-lines + implicit BDF

use std::collections::BTreeMap;

use matro_types::{ClosureHandle, MatroError, Matrix, Result, Value};

use crate::Vm;

fn eval_ode_rhs(
    vm: &mut Vm,
    cl: &ClosureHandle,
    t: f64,
    y: &[f64],
    dim: usize,
    fname: &str,
    stdout: &mut String,
) -> Result<Vec<f64>> {
    let y_val = if y.len() == 1 {
        Value::scalar(y[0])
    } else {
        Value::matrix(Matrix::col_vec(y.to_vec()))
    };
    let rets = vm.call_closure(cl, &[Value::scalar(t), y_val], 1, stdout)?;
    let m = rets
        .into_iter()
        .next()
        .unwrap_or_else(Value::empty)
        .to_dense_matrix()?;
    if m.len() != dim {
        return Err(MatroError::dimension(format!(
            "{fname}: derivative function returned {} elements, expected {dim}",
            m.len()
        )));
    }
    Ok(m.data().to_vec())
}

fn pack_ode_solution(ts: &[f64], ys: &[Vec<f64>], tspan: &Matrix, dim: usize) -> (Matrix, Matrix) {
    if tspan.len() > 2 {
        // Interpolate linearly onto user-requested tspan grid
        let req = tspan.data();
        let dir = if req[req.len() - 1] >= req[0] { 1.0 } else { -1.0 };
        let mut y_mat = Matrix::zeros(req.len(), dim);
        let mut idx = 0usize;
        for (ri, &tq) in req.iter().enumerate() {
            while idx + 2 < ts.len() && dir * (ts[idx + 1] - tq) < 0.0 {
                idx += 1;
            }
            let t0 = ts[idx];
            let t1 = ts[(idx + 1).min(ts.len() - 1)];
            let alpha = if (t1 - t0).abs() > 1e-15 {
                ((tq - t0) / (t1 - t0)).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let j1 = (idx + 1).min(ys.len() - 1);
            for c in 0..dim {
                let v = ys[idx][c] + alpha * (ys[j1][c] - ys[idx][c]);
                y_mat.set(ri, c, v);
            }
        }
        return (Matrix::col_vec(req.to_vec()), y_mat);
    }
    let n = ts.len();
    let t_mat = Matrix::col_vec(ts.to_vec());
    let mut y_mat = Matrix::zeros(n, dim);
    for r in 0..n {
        for c in 0..dim {
            y_mat.set(r, c, ys[r][c]);
        }
    }
    (t_mat, y_mat)
}

/// Adaptive Bogacki-Shampine RK2(3) solver with FSAL: `[t, y] = ode23(f, tspan, y0)`.
pub fn solve_ode23(
    vm: &mut Vm,
    cl: &ClosureHandle,
    tspan: &Matrix,
    y0: &Matrix,
    stdout: &mut String,
) -> Result<(Matrix, Matrix)> {
    if tspan.len() < 2 {
        return Err(MatroError::dimension("ode23: tspan must have at least 2 elements"));
    }
    let t0 = tspan.data()[0];
    let tf = tspan.data()[tspan.len() - 1];
    let dir = if tf >= t0 { 1.0 } else { -1.0 };
    let dim = y0.len();
    if dim == 0 {
        return Err(MatroError::dimension("ode23: y0 cannot be empty"));
    }

    let mut t = t0;
    let mut y = y0.data().to_vec();
    let mut h = dir * ((tf - t0).abs() * 0.01).clamp(1e-5, 0.1);
    let rtol = 1e-4;
    let atol = 1e-6;

    let mut ts = vec![t];
    let mut ys = vec![y.clone()];
    let mut k1 = eval_ode_rhs(vm, cl, t, &y, dim, "ode23", stdout)?;

    for _ in 0..6000 {
        if dir * (tf - t) <= 1e-12 {
            break;
        }
        if dir * (t + h - tf) > 0.0 {
            h = tf - t;
        }
        let y2: Vec<f64> = (0..dim).map(|i| y[i] + 0.5 * h * k1[i]).collect();
        let k2 = eval_ode_rhs(vm, cl, t + 0.5 * h, &y2, dim, "ode23", stdout)?;

        let y3: Vec<f64> = (0..dim).map(|i| y[i] + 0.75 * h * k2[i]).collect();
        let k3 = eval_ode_rhs(vm, cl, t + 0.75 * h, &y3, dim, "ode23", stdout)?;

        let y_next: Vec<f64> = (0..dim)
            .map(|i| y[i] + h * ((2.0 / 9.0) * k1[i] + (1.0 / 3.0) * k2[i] + (4.0 / 9.0) * k3[i]))
            .collect();
        let k4 = eval_ode_rhs(vm, cl, t + h, &y_next, dim, "ode23", stdout)?;

        let mut err_norm = 0.0f64;
        for i in 0..dim {
            let err_i = h
                * ((-5.0 / 72.0) * k1[i]
                    + (1.0 / 12.0) * k2[i]
                    + (1.0 / 9.0) * k3[i]
                    - (1.0 / 8.0) * k4[i]);
            let scale = atol + rtol * y[i].abs().max(y_next[i].abs());
            err_norm = err_norm.max((err_i / scale).abs());
        }

        if err_norm <= 1.0 || h.abs() <= 1e-12 {
            t += h;
            y = y_next;
            k1 = k4;
            ts.push(t);
            ys.push(y.clone());
        }
        let factor = if err_norm == 0.0 {
            2.0
        } else {
            (0.9 * err_norm.powf(-1.0 / 3.0)).clamp(0.2, 2.5)
        };
        h *= factor;
    }

    Ok(pack_ode_solution(&ts, &ys, tspan, dim))
}

fn rk4_step(
    vm: &mut Vm,
    cl: &ClosureHandle,
    t: f64,
    y: &[f64],
    h: f64,
    dim: usize,
    stdout: &mut String,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let k1 = eval_ode_rhs(vm, cl, t, y, dim, "ode113", stdout)?;
    let y2: Vec<f64> = (0..dim).map(|i| y[i] + 0.5 * h * k1[i]).collect();
    let k2 = eval_ode_rhs(vm, cl, t + 0.5 * h, &y2, dim, "ode113", stdout)?;
    let y3: Vec<f64> = (0..dim).map(|i| y[i] + 0.5 * h * k2[i]).collect();
    let k3 = eval_ode_rhs(vm, cl, t + 0.5 * h, &y3, dim, "ode113", stdout)?;
    let y4: Vec<f64> = (0..dim).map(|i| y[i] + h * k3[i]).collect();
    let k4 = eval_ode_rhs(vm, cl, t + h, &y4, dim, "ode113", stdout)?;
    let y_next: Vec<f64> = (0..dim)
        .map(|i| y[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
        .collect();
    let f_next = eval_ode_rhs(vm, cl, t + h, &y_next, dim, "ode113", stdout)?;
    Ok((y_next, f_next))
}

/// Adaptive 4th-order Adams-Bashforth-Moulton PECE solver: `[t, y] = ode113(f, tspan, y0)`.
pub fn solve_ode113(
    vm: &mut Vm,
    cl: &ClosureHandle,
    tspan: &Matrix,
    y0: &Matrix,
    stdout: &mut String,
) -> Result<(Matrix, Matrix)> {
    if tspan.len() < 2 {
        return Err(MatroError::dimension("ode113: tspan must have at least 2 elements"));
    }
    let t0 = tspan.data()[0];
    let tf = tspan.data()[tspan.len() - 1];
    let dir = if tf >= t0 { 1.0 } else { -1.0 };
    let dim = y0.len();
    if dim == 0 {
        return Err(MatroError::dimension("ode113: y0 cannot be empty"));
    }

    let rtol = 1e-6;
    let atol = 1e-8;
    let mut t = t0;
    let mut y = y0.data().to_vec();
    let mut h = dir * ((tf - t0).abs() * 0.01).clamp(1e-5, 0.05);

    let mut ts = vec![t];
    let mut ys = vec![y.clone()];
    let mut f_hist: Vec<Vec<f64>> = vec![eval_ode_rhs(vm, cl, t, &y, dim, "ode113", stdout)?];

    for _ in 0..8000 {
        if dir * (tf - t) <= 1e-12 {
            break;
        }
        if dir * (t + h - tf) > 0.0 {
            let rem = tf - t;
            let (yn, _) = rk4_step(vm, cl, t, &y, rem, dim, stdout)?;
            t = tf;
            y = yn;
            ts.push(t);
            ys.push(y.clone());
            break;
        }

        if f_hist.len() < 4 {
            // Bootstrap multistep history with RK4
            let (yn, fn_) = rk4_step(vm, cl, t, &y, h, dim, stdout)?;
            t += h;
            y = yn;
            ts.push(t);
            ys.push(y.clone());
            f_hist.push(fn_);
            continue;
        }

        let n = f_hist.len();
        let f0 = &f_hist[n - 1];
        let f1 = &f_hist[n - 2];
        let f2 = &f_hist[n - 3];
        let f3 = &f_hist[n - 4];

        // AB4 Predictor
        let y_pred: Vec<f64> = (0..dim)
            .map(|i| y[i] + (h / 24.0) * (55.0 * f0[i] - 59.0 * f1[i] + 37.0 * f2[i] - 9.0 * f3[i]))
            .collect();
        let f_pred = eval_ode_rhs(vm, cl, t + h, &y_pred, dim, "ode113", stdout)?;

        // AM4 Corrector
        let y_corr: Vec<f64> = (0..dim)
            .map(|i| y[i] + (h / 24.0) * (9.0 * f_pred[i] + 19.0 * f0[i] - 5.0 * f1[i] + f2[i]))
            .collect();

        // Milne local error estimate
        let mut err_norm = 0.0f64;
        for i in 0..dim {
            let err_i = (19.0 / 270.0) * (y_corr[i] - y_pred[i]).abs();
            let scale = atol + rtol * y[i].abs().max(y_corr[i].abs());
            err_norm = err_norm.max(err_i / scale);
        }

        if err_norm <= 1.0 || h.abs() <= 1e-11 {
            let f_corr = eval_ode_rhs(vm, cl, t + h, &y_corr, dim, "ode113", stdout)?;
            t += h;
            y = y_corr;
            ts.push(t);
            ys.push(y.clone());
            f_hist.push(f_corr);
            if err_norm < 0.02 && h.abs() < (tf - t0).abs() * 0.05 {
                // Double step size & reset history from current point
                h *= 1.5;
                let last_f = f_hist.pop().unwrap();
                f_hist.clear();
                f_hist.push(last_f);
            }
        } else {
            // Shrink step size & restart history from current accepted state
            h *= 0.5;
            let last_f = f_hist.pop().unwrap();
            f_hist.clear();
            f_hist.push(last_f);
        }
    }

    Ok(pack_ode_solution(&ts, &ys, tspan, dim))
}

/// `solinit = bvpinit(x, yinit)` where `yinit` is either a constant vector or a function handle `@(x)`.
pub fn eval_bvpinit(
    vm: &mut Vm,
    x_mat: &Matrix,
    yinit: &Value,
    stdout: &mut String,
) -> Result<Value> {
    let xs = x_mat.data();
    let nx = xs.len();
    if nx < 2 {
        return Err(MatroError::dimension("bvpinit: mesh x must have at least 2 points"));
    }
    let y_mat = match yinit {
        Value::Closure(cl) => {
            let mut cols = Vec::with_capacity(nx);
            let mut dim = 0usize;
            for &x in xs {
                let rets = vm.call_closure(cl, &[Value::scalar(x)], 1, stdout)?;
                let m = rets
                    .into_iter()
                    .next()
                    .unwrap_or_else(Value::empty)
                    .to_dense_matrix()?;
                if dim == 0 {
                    dim = m.len();
                } else if m.len() != dim {
                    return Err(MatroError::dimension(
                        "bvpinit: yinit function must return a consistent vector length",
                    ));
                }
                cols.push(m.data().to_vec());
            }
            let mut out = Matrix::zeros(dim, nx);
            for c in 0..nx {
                for r in 0..dim {
                    out.set(r, c, cols[c][r]);
                }
            }
            out
        }
        other => {
            let v = other.to_dense_matrix()?;
            let dim = v.len();
            let mut out = Matrix::zeros(dim, nx);
            for c in 0..nx {
                for r in 0..dim {
                    out.set(r, c, v.data()[r]);
                }
            }
            out
        }
    };

    let mut map = BTreeMap::new();
    map.insert("x".to_string(), Value::matrix(Matrix::row_vec(xs.to_vec())));
    map.insert("y".to_string(), Value::matrix(y_mat));
    Ok(Value::Struct(map))
}

/// 4th-order Lobatto IIIA collocation 2-point BVP solver: `sol = bvp4c(odefun, bcfun, solinit)`.
pub fn solve_bvp4c(
    vm: &mut Vm,
    ode_cl: &ClosureHandle,
    bc_cl: &ClosureHandle,
    solinit: &Value,
    stdout: &mut String,
) -> Result<Value> {
    let Value::Struct(map) = solinit else {
        return Err(MatroError::runtime("bvp4c: solinit must be a struct from bvpinit"));
    };
    let x_mat = map
        .get("x")
        .ok_or_else(|| MatroError::runtime("bvp4c: solinit missing field 'x'"))?
        .to_dense_matrix()?;
    let y_init = map
        .get("y")
        .ok_or_else(|| MatroError::runtime("bvp4c: solinit missing field 'y'"))?
        .to_dense_matrix()?;

    let xs = x_mat.data().to_vec();
    let nx = xs.len();
    let dim = y_init.rows();
    if nx < 2 || y_init.cols() != nx || dim == 0 {
        return Err(MatroError::dimension(
            "bvp4c: solinit.y must have size [d, length(solinit.x)]",
        ));
    }

    let n_unk = dim * nx;
    let mut u = vec![0.0; n_unk];
    for c in 0..nx {
        for r in 0..dim {
            u[c * dim + r] = y_init.get(r, c);
        }
    }

    let eval_residual = |vm: &mut Vm, u_vec: &[f64], stdout: &mut String| -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
        let mut fs = Vec::with_capacity(nx);
        for i in 0..nx {
            let yi = &u_vec[i * dim..(i + 1) * dim];
            fs.push(eval_ode_rhs(vm, ode_cl, xs[i], yi, dim, "bvp4c", stdout)?);
        }
        let mut res = vec![0.0; n_unk];
        // Lobatto IIIA (Simpson) collocation equations on each subinterval
        for i in 0..(nx - 1) {
            let h = xs[i + 1] - xs[i];
            let xm = 0.5 * (xs[i] + xs[i + 1]);
            let yi = &u_vec[i * dim..(i + 1) * dim];
            let yj = &u_vec[(i + 1) * dim..(i + 2) * dim];
            let mut ym = vec![0.0; dim];
            for k in 0..dim {
                ym[k] = 0.5 * (yi[k] + yj[k]) - (h / 8.0) * (fs[i + 1][k] - fs[i][k]);
            }
            let fm = eval_ode_rhs(vm, ode_cl, xm, &ym, dim, "bvp4c", stdout)?;
            for k in 0..dim {
                res[i * dim + k] =
                    yj[k] - yi[k] - (h / 6.0) * (fs[i][k] + 4.0 * fm[k] + fs[i + 1][k]);
            }
        }
        // Boundary condition equations
        let ya = Matrix::col_vec(u_vec[0..dim].to_vec());
        let yb = Matrix::col_vec(u_vec[(nx - 1) * dim..nx * dim].to_vec());
        let bc_rets = vm.call_closure(bc_cl, &[Value::matrix(ya), Value::matrix(yb)], 1, stdout)?;
        let bc_m = bc_rets
            .into_iter()
            .next()
            .unwrap_or_else(Value::empty)
            .to_dense_matrix()?;
        if bc_m.len() != dim {
            return Err(MatroError::dimension(format!(
                "bvp4c: bcfun returned {} residuals, expected {dim}",
                bc_m.len()
            )));
        }
        for k in 0..dim {
            res[(nx - 1) * dim + k] = bc_m.data()[k];
        }
        Ok((res, fs))
    };

    let (mut r_cur, mut fs_cur) = eval_residual(vm, &u, stdout)?;
    let mut norm_cur = r_cur.iter().map(|v| v.abs()).fold(0.0f64, f64::max);

    for _iter in 0..40 {
        if norm_cur < 1e-9 {
            break;
        }
        // Finite-difference Jacobian J (n_unk x n_unk)
        let mut jac = Matrix::zeros(n_unk, n_unk);
        for j in 0..n_unk {
            let eps = 1e-7 * u[j].abs().max(1.0);
            let mut u_p = u.clone();
            u_p[j] += eps;
            let (r_p, _) = eval_residual(vm, &u_p, stdout)?;
            for i in 0..n_unk {
                jac.set(i, j, (r_p[i] - r_cur[i]) / eps);
            }
        }
        let rhs = Matrix::col_vec(r_cur.iter().map(|&v| -v).collect());
        let du = jac.solve(&rhs)?;

        // Backtracking Armijo line search
        let mut alpha = 1.0;
        let mut accepted = false;
        for _ in 0..10 {
            let u_try: Vec<f64> = (0..n_unk).map(|i| u[i] + alpha * du.data()[i]).collect();
            let (r_try, fs_try) = eval_residual(vm, &u_try, stdout)?;
            let norm_try = r_try.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
            if norm_try < norm_cur || norm_try < 1e-9 {
                u = u_try;
                r_cur = r_try;
                fs_cur = fs_try;
                norm_cur = norm_try;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted {
            break;
        }
    }

    let mut y_sol = Matrix::zeros(dim, nx);
    let mut yp_sol = Matrix::zeros(dim, nx);
    for c in 0..nx {
        for r in 0..dim {
            y_sol.set(r, c, u[c * dim + r]);
            yp_sol.set(r, c, fs_cur[c][r]);
        }
    }
    let mut out = BTreeMap::new();
    out.insert("x".to_string(), Value::matrix(Matrix::row_vec(xs)));
    out.insert("y".to_string(), Value::matrix(y_sol));
    out.insert("yp".to_string(), Value::matrix(yp_sol));
    Ok(Value::Struct(out))
}

/// Evaluate a BVP/ODE solution struct using cubic Hermite interpolation: `Sxint = deval(sol, xint)`.
pub fn eval_deval(sol: &Value, xint: &Matrix) -> Result<Value> {
    let Value::Struct(map) = sol else {
        return Err(MatroError::runtime("deval: first argument must be a solution struct"));
    };
    let xs_m = map
        .get("x")
        .ok_or_else(|| MatroError::runtime("deval: sol missing 'x'"))?
        .to_dense_matrix()?;
    let y_m = map
        .get("y")
        .ok_or_else(|| MatroError::runtime("deval: sol missing 'y'"))?
        .to_dense_matrix()?;
    let yp_m = map.get("yp").and_then(|v| v.to_dense_matrix().ok());

    let xs = xs_m.data();
    let nx = xs.len();
    let dim = y_m.rows();
    let nq = xint.len();
    let mut out = Matrix::zeros(dim, nq);

    for (qi, &xq) in xint.data().iter().enumerate() {
        let idx = if xq <= xs[0] {
            0
        } else if xq >= xs[nx - 1] {
            nx - 2
        } else {
            match xs.binary_search_by(|v| v.total_cmp(&xq)) {
                Ok(i) => i.min(nx - 2),
                Err(i) => i.saturating_sub(1).min(nx - 2),
            }
        };
        let h = xs[idx + 1] - xs[idx];
        let s = (xq - xs[idx]) / h;
        for r in 0..dim {
            let y0 = y_m.get(r, idx);
            let y1 = y_m.get(r, idx + 1);
            let val = if let Some(ref yp) = yp_m {
                let d0 = yp.get(r, idx);
                let d1 = yp.get(r, idx + 1);
                let h00 = (2.0 * s - 3.0) * s * s + 1.0;
                let h10 = ((s - 2.0) * s + 1.0) * s;
                let h01 = (-2.0 * s + 3.0) * s * s;
                let h11 = (s - 1.0) * s * s;
                h00 * y0 + h10 * h * d0 + h01 * y1 + h11 * h * d1
            } else {
                y0 + s * (y1 - y0)
            };
            out.set(r, qi, val);
        }
    }
    Ok(Value::matrix(out))
}

/// 1D parabolic/elliptic PDE solver:
/// `sol = pdepe(m, pdefun, icfun, bcfun, xmesh, tspan)`
/// Solves $c \frac{\partial u}{\partial t} = x^{-m} \frac{\partial}{\partial x}(x^m f) + s$
/// using conservative finite-volume spatial discretization and implicit Backward-Euler time stepping.
pub fn solve_pdepe(
    vm: &mut Vm,
    m_sym: f64,
    pde_cl: &ClosureHandle,
    ic_cl: &ClosureHandle,
    bc_cl: &ClosureHandle,
    xmesh: &Matrix,
    tspan: &Matrix,
    stdout: &mut String,
) -> Result<Value> {
    let xs = xmesh.data().to_vec();
    let ts = tspan.data().to_vec();
    let nx = xs.len();
    let nt = ts.len();
    if nx < 3 || nt < 2 {
        return Err(MatroError::dimension(
            "pdepe: xmesh must have >= 3 points and tspan >= 2 points",
        ));
    }
    let m_pow = m_sym.round() as i32;
    if !(0..=2).contains(&m_pow) {
        return Err(MatroError::runtime("pdepe: symmetry m must be 0, 1, or 2"));
    }

    // Evaluate initial condition on xmesh
    let mut u0_cols = Vec::with_capacity(nx);
    let mut dim = 0usize;
    for &x in &xs {
        let rets = vm.call_closure(ic_cl, &[Value::scalar(x)], 1, stdout)?;
        let m = rets
            .into_iter()
            .next()
            .unwrap_or_else(Value::empty)
            .to_dense_matrix()?;
        if dim == 0 {
            dim = m.len();
        }
        u0_cols.push(m.data().to_vec());
    }
    if dim == 0 {
        return Err(MatroError::dimension("pdepe: icfun returned empty state"));
    }

    let n_unk = nx * dim;
    let mut u = vec![0.0; n_unk];
    for i in 0..nx {
        for k in 0..dim {
            u[i * dim + k] = u0_cols[i][k];
        }
    }

    // Helper: evaluate pdefun(x, t, u_vec, dudx_vec) -> (c, f, s)
    let call_pde = |vm: &mut Vm, x: f64, t: f64, uv: &[f64], dv: &[f64], stdout: &mut String| -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let u_val = if dim == 1 {
            Value::scalar(uv[0])
        } else {
            Value::matrix(Matrix::col_vec(uv.to_vec()))
        };
        let d_val = if dim == 1 {
            Value::scalar(dv[0])
        } else {
            Value::matrix(Matrix::col_vec(dv.to_vec()))
        };
        let rets = vm.call_closure(
            pde_cl,
            &[Value::scalar(x), Value::scalar(t), u_val, d_val],
            3,
            stdout,
        )?;
        if rets.len() < 3 {
            return Err(MatroError::runtime("pdepe: pdefun must return 3 outputs [c, f, s]"));
        }
        Ok((
            rets[0].to_dense_matrix()?.data().to_vec(),
            rets[1].to_dense_matrix()?.data().to_vec(),
            rets[2].to_dense_matrix()?.data().to_vec(),
        ))
    };

    // Helper: evaluate bcfun(xl, ul, xr, ur, t) -> (pl, ql, pr, qr)
    let call_bc = |vm: &mut Vm, ul: &[f64], ur: &[f64], t: f64, stdout: &mut String| -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
        let ul_v = if dim == 1 { Value::scalar(ul[0]) } else { Value::matrix(Matrix::col_vec(ul.to_vec())) };
        let ur_v = if dim == 1 { Value::scalar(ur[0]) } else { Value::matrix(Matrix::col_vec(ur.to_vec())) };
        let rets = vm.call_closure(
            bc_cl,
            &[Value::scalar(xs[0]), ul_v, Value::scalar(xs[nx - 1]), ur_v, Value::scalar(t)],
            4,
            stdout,
        )?;
        if rets.len() < 4 {
            return Err(MatroError::runtime("pdepe: bcfun must return 4 outputs [pl, ql, pr, qr]"));
        }
        Ok((
            rets[0].to_dense_matrix()?.data().to_vec(),
            rets[1].to_dense_matrix()?.data().to_vec(),
            rets[2].to_dense_matrix()?.data().to_vec(),
            rets[3].to_dense_matrix()?.data().to_vec(),
        ))
    };

    // Dual cell faces x_{i+1/2}
    let mut x_mid = Vec::with_capacity(nx - 1);
    for i in 0..(nx - 1) {
        x_mid.push(0.5 * (xs[i] + xs[i + 1]));
    }
    let x_pow = |x: f64, p: i32| -> f64 {
        match p {
            0 => 1.0,
            1 => x,
            2 => x * x,
            _ => x.powi(p),
        }
    };
    let vol_cell = |xa: f64, xb: f64| -> f64 {
        let p = m_pow + 1;
        (xb.powi(p) - xa.powi(p)) / (p as f64)
    };

    // Implicit residual G(u_new; u_old, t_new, dt) = 0
    let eval_step_res = |vm: &mut Vm, u_new: &[f64], u_old: &[f64], t_new: f64, dt: f64, stdout: &mut String| -> Result<Vec<f64>> {
        // Midpoint fluxes f_{i+1/2}
        let mut f_mid = Vec::with_capacity(nx - 1);
        for i in 0..(nx - 1) {
            let dx = xs[i + 1] - xs[i];
            let mut um = vec![0.0; dim];
            let mut dm = vec![0.0; dim];
            for k in 0..dim {
                um[k] = 0.5 * (u_new[i * dim + k] + u_new[(i + 1) * dim + k]);
                dm[k] = (u_new[(i + 1) * dim + k] - u_new[i * dim + k]) / dx;
            }
            let (_, fm, _) = call_pde(vm, x_mid[i], t_new, &um, &dm, stdout)?;
            f_mid.push(fm);
        }

        let (pl, ql, pr, qr) = call_bc(
            vm,
            &u_new[0..dim],
            &u_new[(nx - 1) * dim..nx * dim],
            t_new,
            stdout,
        )?;

        let mut res = vec![0.0; n_unk];

        // Left boundary node i = 0
        {
            let du0: Vec<f64> = (0..dim)
                .map(|k| (u_new[dim + k] - u_new[k]) / (xs[1] - xs[0]))
                .collect();
            let (c0, _, s0) = call_pde(vm, xs[0], t_new, &u_new[0..dim], &du0, stdout)?;
            let vol = vol_cell(xs[0], x_mid[0]);
            for k in 0..dim {
                if ql[k].abs() <= 1e-15 {
                    // Dirichlet algebraic BC: p_L = 0
                    res[k] = pl[k];
                } else {
                    let f_l = -pl[k] / ql[k];
                    let flux_diff = x_pow(x_mid[0], m_pow) * f_mid[0][k] - x_pow(xs[0], m_pow) * f_l;
                    let dudt = (u_new[k] - u_old[k]) / dt;
                    res[k] = c0[k] * dudt - (flux_diff / vol + s0[k]);
                }
            }
        }

        // Interior nodes i = 1..nx-2
        for i in 1..(nx - 1) {
            let ui = &u_new[i * dim..(i + 1) * dim];
            let dui: Vec<f64> = (0..dim)
                .map(|k| {
                    (u_new[(i + 1) * dim + k] - u_new[(i - 1) * dim + k]) / (xs[i + 1] - xs[i - 1])
                })
                .collect();
            let (ci, _, si) = call_pde(vm, xs[i], t_new, ui, &dui, stdout)?;
            let vol = vol_cell(x_mid[i - 1], x_mid[i]);
            for k in 0..dim {
                let flux_diff = x_pow(x_mid[i], m_pow) * f_mid[i][k]
                    - x_pow(x_mid[i - 1], m_pow) * f_mid[i - 1][k];
                let dudt = (u_new[i * dim + k] - u_old[i * dim + k]) / dt;
                res[i * dim + k] = ci[k] * dudt - (flux_diff / vol + si[k]);
            }
        }

        // Right boundary node i = nx - 1
        {
            let i = nx - 1;
            let dun: Vec<f64> = (0..dim)
                .map(|k| (u_new[i * dim + k] - u_new[(i - 1) * dim + k]) / (xs[i] - xs[i - 1]))
                .collect();
            let (cn, _, sn) = call_pde(vm, xs[i], t_new, &u_new[i * dim..nx * dim], &dun, stdout)?;
            let vol = vol_cell(x_mid[nx - 2], xs[i]);
            for k in 0..dim {
                if qr[k].abs() <= 1e-15 {
                    res[i * dim + k] = pr[k];
                } else {
                    let f_r = -pr[k] / qr[k];
                    let flux_diff =
                        x_pow(xs[i], m_pow) * f_r - x_pow(x_mid[nx - 2], m_pow) * f_mid[nx - 2][k];
                    let dudt = (u_new[i * dim + k] - u_old[i * dim + k]) / dt;
                    res[i * dim + k] = cn[k] * dudt - (flux_diff / vol + sn[k]);
                }
            }
        }

        Ok(res)
    };

    // Store snapshots for each t in tspan
    let mut snapshots = Vec::with_capacity(nt);
    snapshots.push(u.clone());

    for ti in 1..nt {
        let t_prev = ts[ti - 1];
        let t_target = ts[ti];
        // Sub-step each interval into 4 implicit Backward-Euler sub-steps for high temporal accuracy
        let substeps = 4usize;
        let dt = (t_target - t_prev) / (substeps as f64);
        for s in 1..=substeps {
            let t_now = t_prev + (s as f64) * dt;
            let u_old = u.clone();
            let mut u_cur = u.clone();

            for _newton in 0..8 {
                let r_cur = eval_step_res(vm, &u_cur, &u_old, t_now, dt, stdout)?;
                let rnorm = r_cur.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
                if rnorm < 1e-8 {
                    break;
                }
                // Banded / finite-difference Jacobian (perturbed column-by-column)
                let mut jac = Matrix::zeros(n_unk, n_unk);
                for j in 0..n_unk {
                    let eps = 1e-7 * u_cur[j].abs().max(1.0);
                    let mut u_p = u_cur.clone();
                    u_p[j] += eps;
                    let r_p = eval_step_res(vm, &u_p, &u_old, t_now, dt, stdout)?;
                    for i in 0..n_unk {
                        jac.set(i, j, (r_p[i] - r_cur[i]) / eps);
                    }
                }
                let rhs = Matrix::col_vec(r_cur.iter().map(|&v| -v).collect());
                let du = jac.solve(&rhs)?;
                let mut step_max = 0.0f64;
                for i in 0..n_unk {
                    u_cur[i] += du.data()[i];
                    step_max = step_max.max(du.data()[i].abs());
                }
                if step_max < 1e-9 {
                    break;
                }
            }
            u = u_cur;
        }
        snapshots.push(u.clone());
    }

    if dim == 1 {
        let mut sol = Matrix::zeros(nt, nx);
        for ti in 0..nt {
            for xi in 0..nx {
                sol.set(ti, xi, snapshots[ti][xi]);
            }
        }
        Ok(Value::matrix(sol))
    } else {
        // 3D array [nt, nx, dim] in column-major order
        let mut data = Vec::with_capacity(nt * nx * dim);
        for k in 0..dim {
            for xi in 0..nx {
                for ti in 0..nt {
                    data.push(snapshots[ti][xi * dim + k]);
                }
            }
        }
        let im = vec![0.0; data.len()];
        let sol = Matrix::from_nd_complex_col_major(vec![nt, nx, dim], data, im)?.normalize();
        Ok(Value::matrix(sol))
    }
}

/// Brent's golden-section + parabolic 1D bounded minimizer: `[x, fval] = fminbnd(fun, x1, x2)`.
pub fn solve_fminbnd(
    vm: &mut Vm,
    cl: &ClosureHandle,
    x1: f64,
    x2: f64,
    stdout: &mut String,
) -> Result<(f64, f64)> {
    let mut a = x1.min(x2);
    let mut b = x1.max(x2);
    let golden = 0.3819660112501051f64; // (3 - sqrt(5))/2
    let tol = 1e-8;
    let mut eval_f = |x: f64| -> Result<f64> {
        let r = vm.call_closure(cl, &[Value::scalar(x)], 1, stdout)?;
        r.into_iter()
            .next()
            .unwrap_or_else(Value::empty)
            .as_scalar()
    };

    let mut x = a + golden * (b - a);
    let mut w = x;
    let mut v = x;
    let mut fx = eval_f(x)?;
    let mut fw = fx;
    let mut fv = fx;
    let mut d = 0.0f64;
    let mut e = 0.0f64;

    for _ in 0..200 {
        let xm = 0.5 * (a + b);
        let tol1 = tol * x.abs() + 1e-11;
        let tol2 = 2.0 * tol1;
        if (x - xm).abs() <= tol2 - 0.5 * (b - a) {
            break;
        }
        let mut use_golden = true;
        if e.abs() > tol1 {
            let r = (x - w) * (fx - fv);
            let mut q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            let etemp = e;
            e = d;
            if p.abs() < (0.5 * q * etemp).abs() && p > q * (a - x) && p < q * (b - x) {
                d = p / q;
                let u = x + d;
                if (u - a) < tol2 || (b - u) < tol2 {
                    d = if xm >= x { tol1 } else { -tol1 };
                }
                use_golden = false;
            }
        }
        if use_golden {
            e = if x >= xm { a - x } else { b - x };
            d = golden * e;
        }
        let u = if d.abs() >= tol1 {
            x + d
        } else if d > 0.0 {
            x + tol1
        } else {
            x - tol1
        };
        let fu = eval_f(u)?;
        if fu <= fx {
            if u >= x {
                a = x;
            } else {
                b = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }
    Ok((x, fx))
}

/// 2D Gauss-Legendre composite quadrature: `q = integral2(fun, xmin, xmax, ymin, ymax)`
/// where `ymin` and `ymax` may be scalars or function handles `@(x)`.
pub fn solve_integral2(
    vm: &mut Vm,
    fun_cl: &ClosureHandle,
    xmin: f64,
    xmax: f64,
    ymin_v: &Value,
    ymax_v: &Value,
    stdout: &mut String,
) -> Result<f64> {
    // 5-point Gauss-Legendre nodes & weights on [-1, 1]
    const NODES: [f64; 5] = [
        -0.9061798459386640,
        -0.5384693101056831,
        0.0,
        0.5384693101056831,
        0.9061798459386640,
    ];
    const WEIGHTS: [f64; 5] = [
        0.2369268850561891,
        0.4786286704993665,
        0.5688888888888889,
        0.4786286704993665,
        0.2369268850561891,
    ];

    let eval_bound = |vm: &mut Vm, b: &Value, x: f64, stdout: &mut String| -> Result<f64> {
        match b {
            Value::Closure(cl) => {
                let r = vm.call_closure(cl, &[Value::scalar(x)], 1, stdout)?;
                r.into_iter().next().unwrap_or_else(Value::empty).as_scalar()
            }
            other => other.as_scalar(),
        }
    };

    let nx_panels = 8usize;
    let ny_panels = 8usize;
    let hx = (xmax - xmin) / (nx_panels as f64);
    let mut total = 0.0;

    for ix in 0..nx_panels {
        let xa = xmin + (ix as f64) * hx;
        let xb = xa + hx;
        let xm = 0.5 * (xa + xb);
        let xh = 0.5 * (xb - xa);
        for (&nx, &wx) in NODES.iter().zip(WEIGHTS.iter()) {
            let x = xm + xh * nx;
            let ya_all = eval_bound(vm, ymin_v, x, stdout)?;
            let yb_all = eval_bound(vm, ymax_v, x, stdout)?;
            let hy = (yb_all - ya_all) / (ny_panels as f64);
            let mut inner = 0.0;
            for iy in 0..ny_panels {
                let ya = ya_all + (iy as f64) * hy;
                let yb = ya + hy;
                let ym = 0.5 * (ya + yb);
                let yh = 0.5 * (yb - ya);
                for (&ny, &wy) in NODES.iter().zip(WEIGHTS.iter()) {
                    let y = ym + yh * ny;
                    let r = vm.call_closure(fun_cl, &[Value::scalar(x), Value::scalar(y)], 1, stdout)?;
                    let val = r.into_iter().next().unwrap_or_else(Value::empty).as_scalar()?;
                    inner += wy * yh * val;
                }
            }
            total += wx * xh * inner;
        }
    }
    Ok(total)
}
