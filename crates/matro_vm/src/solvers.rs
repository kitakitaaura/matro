//! Higher-order numerical solvers invoked from `matro_vm`:
//! - `ode45`: Adaptive Dormand-Prince RK4(5) ODE solver (scalar & vector systems)
//! - `ode15s`: Adaptive L-stable TR-BDF2 stiff ODE solver with finite-difference Jacobian
//! - `fzero`: Brent's method root finder
//! - `integral`: Adaptive Simpson's quadrature
//! - `fminsearch`: Nelder-Mead simplex optimizer

use matro_types::{ClosureHandle, MatroError, Matrix, Result, Value};

use crate::Vm;

/// Parsed ODE options struct (`odeset`).
#[derive(Debug, Clone)]
pub struct OdeOptions {
    pub rel_tol: f64,
    pub abs_tol: f64,
    pub max_step: Option<f64>,
    pub initial_step: Option<f64>,
}

impl Default for OdeOptions {
    fn default() -> Self {
        Self {
            rel_tol: 1e-5,
            abs_tol: 1e-7,
            max_step: None,
            initial_step: None,
        }
    }
}

impl OdeOptions {
    pub fn from_val(v: Option<&Value>) -> Self {
        let mut opts = Self::default();
        if let Some(Value::Struct(map)) = v {
            for (k, val) in map {
                if let Ok(sc) = val.as_scalar() {
                    if !sc.is_finite() {
                        continue;
                    }
                    match k.to_ascii_lowercase().as_str() {
                        "reltol" => opts.rel_tol = sc.max(1e-13),
                        "abstol" => opts.abs_tol = sc.max(1e-15),
                        "maxstep" => opts.max_step = Some(sc.abs()),
                        "initialstep" => opts.initial_step = Some(sc.abs()),
                        _ => {}
                    }
                }
            }
        }
        opts
    }
}

fn hermite_interp(t0: f64, y0: &[f64], f0: &[f64], t1: f64, y1: &[f64], f1: &[f64], tq: f64) -> Vec<f64> {
    let h = t1 - t0;
    if h.abs() < 1e-30 {
        return y1.to_vec();
    }
    let s = (tq - t0) / h;
    let s2 = s * s;
    let s3 = s2 * s;
    let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
    let h10 = s3 - 2.0 * s2 + s;
    let h01 = -2.0 * s3 + 3.0 * s2;
    let h11 = s3 - s2;
    (0..y0.len())
        .map(|i| h00 * y0[i] + h10 * h * f0[i] + h01 * y1[i] + h11 * h * f1[i])
        .collect()
}

/// Adaptive Dormand-Prince RK4(5) solver: `[t, y] = ode45(f, tspan, y0 [, options])`.
/// Honors intermediate `tspan` query points via cubic Hermite dense output!
pub fn solve_ode45(
    vm: &mut Vm,
    cl: &ClosureHandle,
    tspan: &Matrix,
    y0: &Matrix,
    opts: &OdeOptions,
    stdout: &mut String,
) -> Result<(Matrix, Matrix)> {
    if tspan.len() < 2 {
        return Err(MatroError::dimension("ode45: tspan must have at least 2 elements [t0, tf]"));
    }
    let t0 = tspan.data()[0];
    let tf = tspan.data()[tspan.len() - 1];
    let dir = if tf >= t0 { 1.0 } else { -1.0 };
    let dim = y0.len();
    if dim == 0 {
        return Err(MatroError::dimension("ode45: y0 cannot be empty"));
    }

    let mut eval_rhs = |t: f64, y: &[f64]| -> Result<Vec<f64>> {
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
                "ode45: derivative function returned {} elements, expected {dim}",
                m.len()
            )));
        }
        Ok(m.data().to_vec())
    };

    let mut t = t0;
    let mut y = y0.data().to_vec();
    let max_step = opts
        .max_step
        .unwrap_or_else(|| (tf - t0).abs().max(1e-6));
    let mut h = dir
        * opts
            .initial_step
            .unwrap_or_else(|| ((tf - t0).abs() * 0.01).clamp(1e-5, 0.1))
            .min(max_step);
    let rtol = opts.rel_tol;
    let atol = opts.abs_tol;

    let fixed_grid = tspan.len() > 2;
    let mut grid_idx = 1usize;
    let mut ts = vec![t0];
    let mut ys = vec![y.clone()];

    let mut k1 = eval_rhs(t, &y)?;
    let max_steps = 20_000usize;

    for _ in 0..max_steps {
        if dir * (tf - t) <= 1e-12 * (tf - t0).abs().max(1.0) {
            break;
        }
        if h.abs() > max_step {
            h = dir * max_step;
        }
        if dir * (t + h - tf) > 0.0 {
            h = tf - t;
        }

        let y2: Vec<f64> = (0..dim).map(|i| y[i] + h * (1.0 / 5.0) * k1[i]).collect();
        let k2 = eval_rhs(t + h * (1.0 / 5.0), &y2)?;

        let y3: Vec<f64> = (0..dim)
            .map(|i| y[i] + h * ((3.0 / 40.0) * k1[i] + (9.0 / 40.0) * k2[i]))
            .collect();
        let k3 = eval_rhs(t + h * (3.0 / 10.0), &y3)?;

        let y4: Vec<f64> = (0..dim)
            .map(|i| {
                y[i] + h
                    * ((44.0 / 45.0) * k1[i] - (56.0 / 15.0) * k2[i] + (32.0 / 9.0) * k3[i])
            })
            .collect();
        let k4 = eval_rhs(t + h * (4.0 / 5.0), &y4)?;

        let y5: Vec<f64> = (0..dim)
            .map(|i| {
                y[i] + h
                    * ((19372.0 / 6561.0) * k1[i] - (25360.0 / 2187.0) * k2[i]
                        + (64448.0 / 6561.0) * k3[i]
                        - (212.0 / 729.0) * k4[i])
            })
            .collect();
        let k5 = eval_rhs(t + h * (8.0 / 9.0), &y5)?;

        let y6: Vec<f64> = (0..dim)
            .map(|i| {
                y[i] + h
                    * ((9017.0 / 3168.0) * k1[i] - (355.0 / 33.0) * k2[i]
                        + (46732.0 / 5247.0) * k3[i]
                        + (49.0 / 176.0) * k4[i]
                        - (5103.0 / 18656.0) * k5[i])
            })
            .collect();
        let k6 = eval_rhs(t + h, &y6)?;

        let y_next: Vec<f64> = (0..dim)
            .map(|i| {
                y[i] + h
                    * ((35.0 / 384.0) * k1[i]
                        + (500.0 / 1113.0) * k3[i]
                        + (125.0 / 192.0) * k4[i]
                        - (2187.0 / 6784.0) * k5[i]
                        + (11.0 / 84.0) * k6[i])
            })
            .collect();
        let k7 = eval_rhs(t + h, &y_next)?;

        let mut err_norm = 0.0f64;
        for i in 0..dim {
            let err_i = h
                * ((71.0 / 57600.0) * k1[i] - (71.0 / 16695.0) * k3[i]
                    + (71.0 / 1920.0) * k4[i]
                    - (17253.0 / 339200.0) * k5[i]
                    + (22.0 / 525.0) * k6[i]
                    - (1.0 / 40.0) * k7[i]);
            let scale = atol + rtol * y[i].abs().max(y_next[i].abs());
            err_norm = err_norm.max((err_i / scale).abs());
        }

        if err_norm <= 1.0 || h.abs() <= 1e-13 {
            let t_prev = t;
            t += h;
            if fixed_grid {
                while grid_idx < tspan.len()
                    && dir * (tspan.data()[grid_idx] - t) <= 1e-12 * (tf - t0).abs().max(1.0)
                {
                    let tq = tspan.data()[grid_idx];
                    ts.push(tq);
                    ys.push(hermite_interp(t_prev, &y, &k1, t, &y_next, &k7, tq));
                    grid_idx += 1;
                }
            } else {
                ts.push(t);
                ys.push(y_next.clone());
            }
            y = y_next;
            k1 = k7;
        }

        let factor = if err_norm == 0.0 {
            2.0
        } else {
            (0.9 * err_norm.powf(-0.2)).clamp(0.2, 2.5)
        };
        h *= factor;
    }

    let n = ts.len();
    let t_mat = Matrix::col_vec(ts);
    let mut y_mat = Matrix::zeros(n, dim);
    for r in 0..n {
        for c in 0..dim {
            y_mat.set(r, c, ys[r][c]);
        }
    }
    Ok((t_mat, y_mat))
}

/// Compact in-place partial-pivoting LU factorization for fast repeated solves in `ode15s`.
struct LuFactor {
    n: usize,
    lu: Vec<f64>, // row-major n x n
    piv: Vec<usize>,
}

impl LuFactor {
    fn factor(m: &Matrix) -> Option<Self> {
        let n = m.rows();
        let mut lu = vec![0.0; n * n];
        for r in 0..n {
            for c in 0..n {
                lu[r * n + c] = m.get(r, c);
            }
        }
        let mut piv: Vec<usize> = (0..n).collect();
        for k in 0..n {
            let mut max_r = k;
            let mut max_v = lu[k * n + k].abs();
            for r in (k + 1)..n {
                let v = lu[r * n + k].abs();
                if v > max_v {
                    max_v = v;
                    max_r = r;
                }
            }
            if max_v < 1e-15 || !max_v.is_finite() {
                return None;
            }
            if max_r != k {
                piv.swap(k, max_r);
                for c in 0..n {
                    lu.swap(k * n + c, max_r * n + c);
                }
            }
            let diag = lu[k * n + k];
            for r in (k + 1)..n {
                let mult = lu[r * n + k] / diag;
                lu[r * n + k] = mult;
                for c in (k + 1)..n {
                    lu[r * n + c] -= mult * lu[k * n + c];
                }
            }
        }
        Some(Self { n, lu, piv })
    }

    fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut x = vec![0.0; n];
        for i in 0..n {
            x[i] = b[self.piv[i]];
        }
        for i in 1..n {
            let mut sum = x[i];
            for j in 0..i {
                sum -= self.lu[i * n + j] * x[j];
            }
            x[i] = sum;
        }
        for i in (0..n).rev() {
            let mut sum = x[i];
            for j in (i + 1)..n {
                sum -= self.lu[i * n + j] * x[j];
            }
            x[i] = sum / self.lu[i * n + i];
        }
        x
    }
}

/// Adaptive L-stable TR-BDF2 stiff ODE solver: `[t, y] = ode15s(f, tspan, y0)`.
///
/// Uses the Bank-Coughran-Fichtner-Grosse-Rose-Smith TR-BDF2 scheme
/// ($\gamma = 2 - \sqrt{2}$, $d = \gamma / 2$) with finite-difference Jacobian,
/// simplified Newton iteration sharing $(I - d h J)$ across both stages, and
/// the Hosea-Shampine filtered divided-difference error estimator.
pub fn solve_ode15s(
    vm: &mut Vm,
    cl: &ClosureHandle,
    tspan: &Matrix,
    y0: &Matrix,
    stdout: &mut String,
) -> Result<(Matrix, Matrix)> {
    if tspan.len() < 2 {
        return Err(MatroError::dimension(
            "ode15s: tspan must have at least 2 elements [t0, tf]",
        ));
    }
    let t0 = tspan.data()[0];
    let tf = tspan.data()[tspan.len() - 1];
    let dir = if tf >= t0 { 1.0 } else { -1.0 };
    let dim = y0.len();
    if dim == 0 {
        return Err(MatroError::dimension("ode15s: y0 cannot be empty"));
    }

    let mut eval_rhs = |t: f64, y: &[f64]| -> Result<Vec<f64>> {
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
            .as_matrix()?
            .clone();
        if m.len() != dim {
            return Err(MatroError::dimension(format!(
                "ode15s: derivative function returned {} elements, expected {dim}",
                m.len()
            )));
        }
        Ok(m.data().to_vec())
    };

    let gamma = 2.0 - std::f64::consts::SQRT_2;
    let d = 0.5 * gamma;
    let w_gamma = 1.0 / (gamma * (2.0 - gamma));
    let w_n = -((1.0 - gamma) * (1.0 - gamma)) / (gamma * (2.0 - gamma));
    let c_err = (3.0 * std::f64::consts::SQRT_2 - 4.0) / 3.0;
    let inv_g = 1.0 / gamma;
    let inv_g1g = 1.0 / (gamma * (1.0 - gamma));
    let inv_1g = 1.0 / (1.0 - gamma);

    let rtol = 1e-4;
    let atol = 1e-6;
    let span_len = (tf - t0).abs();
    let max_h = (span_len * 0.1).max(1e-6);
    let min_h = (span_len * 1e-12).max(1e-14);

    let mut t = t0;
    let mut y = y0.data().to_vec();
    let mut f_n = eval_rhs(t, &y)?;

    // Scale initial step size conservatively when initial derivative is stiff/large
    let f0_norm = f_n
        .iter()
        .zip(y.iter())
        .map(|(&fi, &yi)| (fi / (atol + rtol * yi.abs())).abs())
        .fold(0.0f64, f64::max);
    let h0_est = if f0_norm > 0.0 {
        (0.01 / f0_norm.sqrt()).clamp(min_h * 100.0, max_h * 0.25)
    } else {
        (span_len * 0.01).clamp(1e-5, max_h)
    };
    let mut h = dir * h0_est.min(span_len);

    let mut ts = vec![t];
    let mut ys = vec![y.clone()];

    let mut jac_cache: Option<Matrix> = None;
    let mut jac_age = 0usize;
    let max_steps = 8000usize;

    let build_iter_mat = |jac: &Matrix, dh: f64| -> Matrix {
        let mut m = Matrix::zeros(dim, dim);
        for c in 0..dim {
            for r in 0..dim {
                let id = if r == c { 1.0 } else { 0.0 };
                m.set(r, c, id - dh * jac.get(r, c));
            }
        }
        m
    };

    let scaled_rms = |v: &[f64], ref_a: &[f64], ref_b: &[f64]| -> f64 {
        let mut sum_sq = 0.0;
        for i in 0..dim {
            let sc = atol + rtol * ref_a[i].abs().max(ref_b[i].abs());
            let r = v[i] / sc;
            sum_sq += r * r;
        }
        (sum_sq / dim as f64).sqrt()
    };

    for _ in 0..max_steps {
        if dir * (tf - t) <= 1e-12 * span_len.max(1.0) {
            break;
        }
        if dir * (t + h - tf) > 0.0 {
            h = tf - t;
        }

        let mut recomputed_jac_this_step = false;
        let step_result = loop {
            if jac_cache.is_none() || jac_age >= 5 {
                let mut jac = Matrix::zeros(dim, dim);
                let mut y_pert = y.clone();
                for j in 0..dim {
                    let eps = 1e-7 * y[j].abs().max(1.0);
                    y_pert[j] = y[j] + eps;
                    let fp = eval_rhs(t, &y_pert)?;
                    y_pert[j] = y[j];
                    for i in 0..dim {
                        jac.set(i, j, (fp[i] - f_n[i]) / eps);
                    }
                }
                jac_cache = Some(jac);
                jac_age = 0;
                recomputed_jac_this_step = true;
            }
            let jac = jac_cache.as_ref().unwrap();
            let iter_mat = build_iter_mat(jac, d * h);
            let Some(lu) = LuFactor::factor(&iter_mat) else {
                if !recomputed_jac_this_step {
                    jac_cache = None;
                    continue;
                }
                break None;
            };

            // Helper: solve z - d*h*f(t_s, z) = rhs reusing cached LU factorization of M = I - d*h*J
            let mut solve_stage =
                |t_s: f64, rhs: &[f64], mut z: Vec<f64>| -> Result<Option<(Vec<f64>, Vec<f64>)>> {
                    let mut f_z = eval_rhs(t_s, &z)?;
                    let mut prev_norm = f64::INFINITY;
                    let mut g = vec![0.0; dim];
                    for _ in 0..8 {
                        for i in 0..dim {
                            g[i] = -(z[i] - d * h * f_z[i] - rhs[i]);
                        }
                        let dz = lu.solve(&g);
                        if dz.iter().any(|x| !x.is_finite()) {
                            return Ok(None);
                        }
                        for i in 0..dim {
                            z[i] += dz[i];
                        }
                        f_z = eval_rhs(t_s, &z)?;
                        let norm = scaled_rms(&dz, &y, &z);
                        if norm < 0.05 {
                            return Ok(Some((z, f_z)));
                        }
                        if norm > prev_norm * 2.0 && norm > 1.0 {
                            return Ok(None);
                        }
                        prev_norm = norm;
                    }
                    Ok(None)
                };

            // Stage 1: Trapezoidal step to t + gamma * h
            let rhs1: Vec<f64> = (0..dim).map(|i| y[i] + d * h * f_n[i]).collect();
            let s1 = solve_stage(t + gamma * h, &rhs1, y.clone())?;
            let Some((y_g, f_g)) = s1 else {
                if !recomputed_jac_this_step {
                    jac_cache = None;
                    continue;
                }
                break None;
            };

            // Stage 2: BDF-2 step to t + h
            let rhs2: Vec<f64> = (0..dim).map(|i| w_gamma * y_g[i] + w_n * y[i]).collect();
            let s2 = solve_stage(t + h, &rhs2, y_g.clone())?;
            let Some((y_next, f_next)) = s2 else {
                if !recomputed_jac_this_step {
                    jac_cache = None;
                    continue;
                }
                break None;
            };

            // Hosea-Shampine filtered local error estimate: M * err = e_raw (reusing LU!)
            let e_raw: Vec<f64> = (0..dim)
                .map(|i| c_err * h * (inv_g * f_n[i] - inv_g1g * f_g[i] + inv_1g * f_next[i]))
                .collect();
            let filtered = lu.solve(&e_raw);
            let err_vec = if filtered.iter().all(|x| x.is_finite()) {
                filtered
            } else {
                e_raw
            };
            let err_norm = scaled_rms(&err_vec, &y, &y_next);
            break Some((y_next, f_next, err_norm));
        };

        let Some((y_next, f_next, err_norm)) = step_result else {
            // Newton failed even with fresh Jacobian: cut step size
            h *= 0.25;
            if h.abs() < min_h {
                return Err(MatroError::runtime(
                    "ode15s: step size underflow (Newton iteration failed to converge)",
                ));
            }
            jac_cache = None;
            continue;
        };

        if err_norm <= 1.0 || h.abs() <= min_h {
            t += h;
            y = y_next;
            f_n = f_next;
            ts.push(t);
            ys.push(y.clone());
            jac_age += 1;
        }

        let factor = if err_norm <= 1e-12 {
            3.0
        } else {
            (0.85 * err_norm.powf(-1.0 / 3.0)).clamp(0.2, 3.0)
        };
        let next_h = (h.abs() * factor).clamp(min_h, max_h);
        h = dir * next_h;
    }

    // If tspan specifies more than 2 output points, interpolate onto requested tspan
    if tspan.len() > 2 && ts.len() >= 2 {
        let req = tspan.data();
        let n_out = req.len();
        let mut y_mat = Matrix::zeros(n_out, dim);
        let mut idx = 0usize;
        for (row, &tq) in req.iter().enumerate() {
            while idx + 2 < ts.len() && dir * (ts[idx + 1] - tq) < 0.0 {
                idx += 1;
            }
            let t_lo = ts[idx];
            let t_hi = ts[idx + 1];
            let alpha = if (t_hi - t_lo).abs() > 1e-15 {
                ((tq - t_lo) / (t_hi - t_lo)).clamp(0.0, 1.0)
            } else {
                0.0
            };
            for c in 0..dim {
                let v = ys[idx][c] * (1.0 - alpha) + ys[idx + 1][c] * alpha;
                y_mat.set(row, c, v);
            }
        }
        return Ok((Matrix::col_vec(req.to_vec()), y_mat));
    }

    let n = ts.len();
    let t_mat = Matrix::col_vec(ts);
    let mut y_mat = Matrix::zeros(n, dim);
    for r in 0..n {
        for c in 0..dim {
            y_mat.set(r, c, ys[r][c]);
        }
    }
    Ok((t_mat, y_mat))
}

/// Brent's method root finder: `x = fzero(f, [a, b])` or `fzero(f, x0)`.
pub fn solve_fzero(
    vm: &mut Vm,
    cl: &ClosureHandle,
    guess: &Matrix,
    stdout: &mut String,
) -> Result<f64> {
    let mut eval_f = |x: f64| -> Result<f64> {
        let rets = vm.call_closure(cl, &[Value::scalar(x)], 1, stdout)?;
        rets.into_iter()
            .next()
            .unwrap_or_else(Value::empty)
            .as_scalar()
    };

    let (mut a, mut b) = if guess.len() >= 2 {
        (guess.data()[0], guess.data()[1])
    } else if guess.len() == 1 {
        let x0 = guess.data()[0];
        let mut dx = if x0.abs() > 1e-6 { x0.abs() * 0.05 } else { 0.05 };
        let mut lo = x0;
        let mut hi = x0;
        let f0 = eval_f(x0)?;
        if f0.abs() < 1e-14 {
            return Ok(x0);
        }
        let mut found = false;
        for _ in 0..40 {
            lo = x0 - dx;
            hi = x0 + dx;
            let flo = eval_f(lo)?;
            let fhi = eval_f(hi)?;
            if flo * fhi <= 0.0 {
                found = true;
                break;
            }
            dx *= 1.6;
        }
        if !found {
            return Err(MatroError::runtime("fzero: failed to find sign change near initial guess"));
        }
        (lo, hi)
    } else {
        return Err(MatroError::dimension("fzero: expected scalar guess or [a, b] bracket"));
    };

    let mut fa = eval_f(a)?;
    let mut fb = eval_f(b)?;
    if fa * fb > 0.0 {
        return Err(MatroError::runtime(
            "fzero: function values at bracket endpoints must have opposite signs",
        ));
    }

    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }
    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut mflag = true;

    for _ in 0..200 {
        if fb.abs() < 1e-13 || (b - a).abs() < 1e-13 {
            return Ok(b);
        }
        let mut s = if (fa - fc).abs() > 1e-15 && (fb - fc).abs() > 1e-15 {
            // Inverse quadratic interpolation
            a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
        } else {
            // Secant method
            b - fb * (b - a) / (fb - fa)
        };

        let cond1 = !((s > (3.0 * a + b) / 4.0 && s < b) || (s < (3.0 * a + b) / 4.0 && s > b));
        let cond2 = mflag && (s - b).abs() >= (b - c).abs() * 0.5;
        let cond3 = !mflag && (s - b).abs() >= (c - d).abs() * 0.5;
        let cond4 = mflag && (b - c).abs() < 1e-14;
        let cond5 = !mflag && (c - d).abs() < 1e-14;

        if cond1 || cond2 || cond3 || cond4 || cond5 {
            s = 0.5 * (a + b);
            mflag = true;
        } else {
            mflag = false;
        }

        let fs = eval_f(s)?;
        d = c;
        c = b;
        fc = fb;
        if fa * fs < 0.0 {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }
        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }
    }
    Ok(b)
}

/// Open adaptive Gauss-Kronrod G7-K15 quadrature: `q = integral(f, a, b [, 'RelTol', ..., 'AbsTol', ..., 'Waypoints', ...])`.
/// Handles integrable endpoint singularities (`1./sqrt(x)` on `[0, 1]`), infinite bounds (`-Inf`/`Inf`), and complex integrands!
pub fn solve_integral_opts(
    vm: &mut Vm,
    cl: &ClosureHandle,
    a: f64,
    b: f64,
    rel_tol: f64,
    abs_tol: f64,
    waypoints: &[f64],
    stdout: &mut String,
) -> Result<matro_types::Complex> {
    use matro_types::Complex;
    if (a - b).abs() == 0.0 {
        return Ok(Complex::ZERO);
    }
    if a > b {
        let rev_wp: Vec<f64> = waypoints.iter().rev().copied().collect();
        return Ok(-solve_integral_opts(vm, cl, b, a, rel_tol, abs_tol, &rev_wp, stdout)?);
    }
    if !waypoints.is_empty() {
        let mut pts = vec![a];
        for &w in waypoints {
            if w > a && w < b {
                pts.push(w);
            }
        }
        pts.push(b);
        let mut total = Complex::ZERO;
        for w in pts.windows(2) {
            total = total + solve_integral_opts(vm, cl, w[0], w[1], rel_tol, abs_tol, &[], stdout)?;
        }
        return Ok(total);
    }

    // 15-point Gauss-Kronrod nodes and weights on [-1, 1]
    const XGK: [f64; 8] = [
        0.9914553711208126,
        0.9491079123427585,
        0.8648644233597691,
        0.7415311855993945,
        0.5860872354676911,
        0.4058451513773972,
        0.2077849550078985,
        0.0,
    ];
    const WGK: [f64; 8] = [
        0.02293532201052922,
        0.06309209262997856,
        0.1047900103222502,
        0.1406532597155259,
        0.1690047266392679,
        0.1903505780647854,
        0.2044329400752989,
        0.2094821410847278,
    ];
    const WG: [f64; 4] = [
        0.1294849661688697,
        0.2797053914892767,
        0.3818300505051189,
        0.4179591836734694,
    ];

    // Map infinite limits to finite [-1, 1] or [0, 1] via algebraic transform
    enum BoundTransform {
        Finite,
        PosInf(f64),     // [a, +Inf): x = a + t/(1-t), t in [0, 1), dx = dt/(1-t)^2
        NegInf(f64),     // (-Inf, b]: x = b - t/(1-t), t in [0, 1), dx = dt/(1-t)^2
        BothInf,         // (-Inf, +Inf): x = t/(1-t^2), t in (-1, 1), dx = (1+t^2)/(1-t^2)^2 dt
    }

    let (transform, u_a, u_b) = match (a.is_infinite(), b.is_infinite()) {
        (false, false) => (BoundTransform::Finite, a, b),
        (false, true) => (BoundTransform::PosInf(a), 0.0, 1.0),
        (true, false) => (BoundTransform::NegInf(b), 0.0, 1.0),
        (true, true) => (BoundTransform::BothInf, -1.0, 1.0),
    };

    let eval_u = |vm: &mut Vm, u: f64, stdout: &mut String| -> Result<Complex> {
        let (x, jac) = match transform {
            BoundTransform::Finite => (u, 1.0),
            BoundTransform::PosInf(a0) => {
                let d = 1.0 - u;
                if d.abs() < 1e-15 {
                    return Ok(Complex::ZERO);
                }
                (a0 + u / d, 1.0 / (d * d))
            }
            BoundTransform::NegInf(b0) => {
                let d = 1.0 - u;
                if d.abs() < 1e-15 {
                    return Ok(Complex::ZERO);
                }
                (b0 - u / d, 1.0 / (d * d))
            }
            BoundTransform::BothInf => {
                let d = 1.0 - u * u;
                if d.abs() < 1e-15 {
                    return Ok(Complex::ZERO);
                }
                (u / d, (1.0 + u * u) / (d * d))
            }
        };
        let rets = vm.call_closure(cl, &[Value::scalar(x)], 1, stdout)?;
        let val = rets
            .into_iter()
            .next()
            .unwrap_or_else(Value::empty)
            .to_dense_matrix()?
            .as_complex_scalar()
            .ok_or_else(|| MatroError::dimension("integral: integrand must return scalar"))?;
        Ok(val * Complex::from_real(jac))
    };

    let gk15 = |vm: &mut Vm, lo: f64, hi: f64, stdout: &mut String| -> Result<(Complex, f64)> {
        let mid = 0.5 * (lo + hi);
        let half = 0.5 * (hi - lo);
        let fc = eval_u(vm, mid, stdout)?;
        let mut res_k = fc * Complex::from_real(WGK[7]);
        let mut res_g = fc * Complex::from_real(WG[3]);
        for j in 0..7 {
            let dx = half * XGK[j];
            let fsum = eval_u(vm, mid - dx, stdout)? + eval_u(vm, mid + dx, stdout)?;
            res_k = res_k + fsum * Complex::from_real(WGK[j]);
            if j % 2 == 1 {
                res_g = res_g + fsum * Complex::from_real(WG[j / 2]);
            }
        }
        let qk = res_k * Complex::from_real(half);
        let qg = res_g * Complex::from_real(half);
        let err = (qk - qg).abs();
        Ok((qk, err))
    };

    // Priority-queue-like adaptive panel subdivision (up to 400 subintervals)
    let mut panels: Vec<(f64, f64, Complex, f64)> = Vec::new();
    let (q0, e0) = gk15(vm, u_a, u_b, stdout)?;
    panels.push((u_a, u_b, q0, e0));

    for _ in 0..350 {
        let total_q = panels.iter().fold(Complex::ZERO, |acc, p| acc + p.2);
        let total_err: f64 = panels.iter().map(|p| p.3).sum();
        let target = abs_tol.max(rel_tol * total_q.abs());
        if total_err <= target {
            return Ok(total_q);
        }
        // Subdivide panel with largest error estimate
        let (worst_i, _) = panels
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.3.total_cmp(&b.1.3))
            .unwrap();
        let (lo, hi, _, _) = panels.swap_remove(worst_i);
        if (hi - lo).abs() <= 1e-14 * (hi.abs() + lo.abs()).max(1.0) {
            panels.push((lo, hi, Complex::ZERO, 0.0));
            break;
        }
        let mid = 0.5 * (lo + hi);
        let (ql, el) = gk15(vm, lo, mid, stdout)?;
        let (qr, er) = gk15(vm, mid, hi, stdout)?;
        panels.push((lo, mid, ql, el));
        panels.push((mid, hi, qr, er));
    }

    Ok(panels.iter().fold(Complex::ZERO, |acc, p| acc + p.2))
}

/// Derivative-free Nelder-Mead simplex optimizer: `[x, fval] = fminsearch(f, x0)`.
pub fn solve_fminsearch(
    vm: &mut Vm,
    cl: &ClosureHandle,
    x0: &Matrix,
    stdout: &mut String,
) -> Result<(Matrix, f64)> {
    let n = x0.len();
    if n == 0 {
        return Err(MatroError::dimension("fminsearch: x0 cannot be empty"));
    }
    let is_row = x0.rows() == 1;
    let mut eval_obj = |x: &[f64]| -> Result<f64> {
        let arg = if x.len() == 1 {
            Value::scalar(x[0])
        } else if is_row {
            Value::matrix(Matrix::row_vec(x.to_vec()))
        } else {
            Value::matrix(Matrix::col_vec(x.to_vec()))
        };
        let rets = vm.call_closure(cl, &[arg], 1, stdout)?;
        rets.into_iter()
            .next()
            .unwrap_or_else(Value::empty)
            .as_scalar()
    };

    // Build initial (n + 1)-vertex simplex using MATLAB's standard 5% / 0.00025 perturbation
    let mut simplex: Vec<(Vec<f64>, f64)> = Vec::with_capacity(n + 1);
    let base = x0.data().to_vec();
    let f0 = eval_obj(&base)?;
    simplex.push((base.clone(), f0));
    for i in 0..n {
        let mut v = base.clone();
        v[i] = if v[i].abs() > 1e-12 {
            v[i] * 1.05
        } else {
            0.00025
        };
        let fv = eval_obj(&v)?;
        simplex.push((v, fv));
    }

    let max_iter = 200 * n;
    let tol_x = 1e-7;
    let tol_f = 1e-7;

    for _ in 0..max_iter {
        simplex.sort_by(|a, b| a.1.total_cmp(&b.1));

        // Convergence check
        let f_spread = (simplex[n].1 - simplex[0].1).abs();
        let x_spread = (1..=n)
            .map(|k| {
                (0..n)
                    .map(|i| (simplex[k].0[i] - simplex[0].0[i]).abs())
                    .fold(0.0f64, f64::max)
            })
            .fold(0.0f64, f64::max);
        if f_spread < tol_f && x_spread < tol_x {
            break;
        }

        // Centroid of best n vertices
        let mut centroid = vec![0.0; n];
        for vertex in simplex.iter().take(n) {
            for i in 0..n {
                centroid[i] += vertex.0[i];
            }
        }
        for c in &mut centroid {
            *c /= n as f64;
        }

        // Reflection (alpha = 1)
        let xr: Vec<f64> = (0..n)
            .map(|i| centroid[i] + (centroid[i] - simplex[n].0[i]))
            .collect();
        let fr = eval_obj(&xr)?;

        if fr < simplex[0].1 {
            // Expansion (gamma = 2)
            let xe: Vec<f64> = (0..n)
                .map(|i| centroid[i] + 2.0 * (xr[i] - centroid[i]))
                .collect();
            let fe = eval_obj(&xe)?;
            if fe < fr {
                simplex[n] = (xe, fe);
            } else {
                simplex[n] = (xr, fr);
            }
        } else if fr < simplex[n - 1].1 {
            simplex[n] = (xr, fr);
        } else {
            // Contraction (rho = 0.5)
            let mut shrink = false;
            if fr < simplex[n].1 {
                let xc: Vec<f64> = (0..n)
                    .map(|i| centroid[i] + 0.5 * (xr[i] - centroid[i]))
                    .collect();
                let fc = eval_obj(&xc)?;
                if fc <= fr {
                    simplex[n] = (xc, fc);
                } else {
                    shrink = true;
                }
            } else {
                let xc: Vec<f64> = (0..n)
                    .map(|i| centroid[i] - 0.5 * (centroid[i] - simplex[n].0[i]))
                    .collect();
                let fc = eval_obj(&xc)?;
                if fc < simplex[n].1 {
                    simplex[n] = (xc, fc);
                } else {
                    shrink = true;
                }
            }
            if shrink {
                let best = simplex[0].0.clone();
                for k in 1..=n {
                    let v: Vec<f64> = (0..n)
                        .map(|i| best[i] + 0.5 * (simplex[k].0[i] - best[i]))
                        .collect();
                    let fv = eval_obj(&v)?;
                    simplex[k] = (v, fv);
                }
            }
        }
    }

    simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
    let (best_x, best_f) = simplex.remove(0);
    let mat = if is_row {
        Matrix::row_vec(best_x)
    } else {
        Matrix::col_vec(best_x)
    };
    Ok((mat, best_f))
}

/// Levenberg-Marquardt nonlinear least-squares / system solver (`lsqnonlin`, `fsolve`, `lsqcurvefit`).
/// Minimizes `0.5 * ||r(x)||_2^2` with finite-difference Jacobian, adaptive damping, and optional box bounds.
pub fn solve_lm(
    vm: &mut Vm,
    cl: &ClosureHandle,
    x0: &Matrix,
    stdout: &mut String,
) -> Result<(Matrix, f64, Matrix)> {
    solve_lm_bounded(vm, cl, x0, None, None, None, stdout)
}

pub fn solve_lm_bounded(
    vm: &mut Vm,
    cl: &ClosureHandle,
    x0: &Matrix,
    curve_xy: Option<(&Value, &Matrix)>,
    lb_opt: Option<&Matrix>,
    ub_opt: Option<&Matrix>,
    stdout: &mut String,
) -> Result<(Matrix, f64, Matrix)> {
    let n = x0.len();
    if n == 0 {
        return Err(MatroError::dimension("fsolve/lsqnonlin: x0 cannot be empty"));
    }
    let lb: Vec<f64> = match lb_opt {
        Some(m) if !m.is_empty() => (0..n)
            .map(|i| m.data().get(i).copied().unwrap_or(f64::NEG_INFINITY))
            .collect(),
        _ => vec![f64::NEG_INFINITY; n],
    };
    let ub: Vec<f64> = match ub_opt {
        Some(m) if !m.is_empty() => (0..n)
            .map(|i| m.data().get(i).copied().unwrap_or(f64::INFINITY))
            .collect(),
        _ => vec![f64::INFINITY; n],
    };
    let clamp_vec = |v: &mut [f64]| {
        for i in 0..n {
            if v[i] < lb[i] {
                v[i] = lb[i];
            }
            if v[i] > ub[i] {
                v[i] = ub[i];
            }
        }
    };

    let is_row = x0.rows() == 1;
    let mut eval_r = |x: &[f64]| -> Result<Vec<f64>> {
        let arg = if x.len() == 1 {
            Value::scalar(x[0])
        } else if is_row {
            Value::matrix(Matrix::row_vec(x.to_vec()))
        } else {
            Value::matrix(Matrix::col_vec(x.to_vec()))
        };
        let rets = if let Some((xdata, _)) = curve_xy {
            vm.call_closure(cl, &[arg, xdata.clone()], 1, stdout)?
        } else {
            vm.call_closure(cl, &[arg], 1, stdout)?
        };
        let m = rets
            .into_iter()
            .next()
            .unwrap_or_else(Value::empty)
            .to_dense_matrix()?;
        if let Some((_, ydata)) = curve_xy {
            if m.len() != ydata.len() {
                return Err(MatroError::dimension(format!(
                    "lsqcurvefit: model returned {} elements, expected {}",
                    m.len(),
                    ydata.len()
                )));
            }
            Ok(m.data()
                .iter()
                .zip(ydata.data().iter())
                .map(|(a, b)| a - b)
                .collect())
        } else {
            Ok(m.data().to_vec())
        }
    };

    let mut x = x0.data().to_vec();
    clamp_vec(&mut x);
    let mut r = eval_r(&x)?;
    let m = r.len();
    if m == 0 {
        return Err(MatroError::dimension(
            "fsolve/lsqnonlin: residual function returned empty vector",
        ));
    }
    let mut cost: f64 = r.iter().map(|&v| v * v).sum();
    let mut lambda = 1e-3f64;

    for _ in 0..200 {
        if cost < 1e-22 {
            break;
        }
        // Finite-difference Jacobian J (m x n), clamped inside [lb, ub]
        let mut jac = vec![0.0; m * n]; // col-major: col j at j*m..
        let mut xp = x.clone();
        for j in 0..n {
            let eps = 1e-7 * x[j].abs().max(1.0);
            let step = if x[j] + eps <= ub[j] {
                eps
            } else if x[j] - eps >= lb[j] {
                -eps
            } else {
                0.0
            };
            if step == 0.0 {
                continue;
            }
            xp[j] = x[j] + step;
            let rp = eval_r(&xp)?;
            xp[j] = x[j];
            if rp.len() != m {
                return Err(MatroError::dimension(
                    "fsolve/lsqnonlin: residual length changed across evaluations",
                ));
            }
            for i in 0..m {
                jac[j * m + i] = (rp[i] - r[i]) / step;
            }
        }

        // Normal equations: A = J' * J (n x n), g = -J' * r (n)
        let mut jtj = Matrix::zeros(n, n);
        let mut g = vec![0.0; n];
        for c in 0..n {
            let mut gc = 0.0;
            for i in 0..m {
                gc -= jac[c * m + i] * r[i];
            }
            g[c] = gc;
            for row in c..n {
                let mut s = 0.0;
                for i in 0..m {
                    s += jac[row * m + i] * jac[c * m + i];
                }
                jtj.set(row, c, s);
                jtj.set(c, row, s);
            }
        }
        // Projected gradient check for active bounds
        let g_inf = (0..n)
            .map(|i| {
                if (x[i] <= lb[i] && g[i] < 0.0) || (x[i] >= ub[i] && g[i] > 0.0) {
                    0.0
                } else {
                    g[i].abs()
                }
            })
            .fold(0.0f64, f64::max);
        if g_inf < 1e-11 {
            break;
        }

        let mut accepted = false;
        for _ in 0..12 {
            let mut damped = jtj.clone();
            for j in 0..n {
                let d = damped.get(j, j);
                damped.set(j, j, d + lambda * d.max(1.0));
            }
            let Some(lu) = LuFactor::factor(&damped) else {
                lambda *= 10.0;
                continue;
            };
            let dx = lu.solve(&g);
            if dx.iter().any(|v| !v.is_finite()) {
                lambda *= 10.0;
                continue;
            }
            let mut x_new: Vec<f64> = (0..n).map(|i| x[i] + dx[i]).collect();
            clamp_vec(&mut x_new);
            let r_new = eval_r(&x_new)?;
            let cost_new: f64 = r_new.iter().map(|&v| v * v).sum();
            if cost_new < cost {
                let step_norm = (0..n)
                    .map(|i| (x_new[i] - x[i]).abs())
                    .fold(0.0f64, f64::max);
                x = x_new;
                r = r_new;
                cost = cost_new;
                lambda = (lambda * 0.3).max(1e-12);
                accepted = true;
                if step_norm < 1e-11 * (1.0 + x.iter().map(|v| v.abs()).fold(0.0f64, f64::max)) {
                    let x_mat = if is_row {
                        Matrix::row_vec(x)
                    } else {
                        Matrix::col_vec(x)
                    };
                    return Ok((x_mat, cost, Matrix::col_vec(r)));
                }
                break;
            }
            lambda = (lambda * 5.0).min(1e12);
        }
        if !accepted {
            break;
        }
    }

    let x_mat = if is_row {
        Matrix::row_vec(x)
    } else {
        Matrix::col_vec(x)
    };
    Ok((x_mat, cost, Matrix::col_vec(r)))
}
