//! Higher-order numerical solvers invoked from `matro_vm`:
//! - `ode45`: Adaptive Dormand-Prince RK4(5) ODE solver (scalar & vector systems)
//! - `ode15s`: Adaptive L-stable TR-BDF2 stiff ODE solver with finite-difference Jacobian
//! - `fzero`: Brent's method root finder
//! - `integral`: Adaptive Simpson's quadrature
//! - `fminsearch`: Nelder-Mead simplex optimizer

use matro_types::{ClosureHandle, MatroError, Matrix, Result, Value};

use crate::Vm;

/// Adaptive Dormand-Prince RK4(5) solver: `[t, y] = ode45(f, [t0, tf], y0)`.
pub fn solve_ode45(
    vm: &mut Vm,
    cl: &ClosureHandle,
    tspan: &Matrix,
    y0: &Matrix,
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
            .as_matrix()?
            .clone();
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
    let mut h = dir * ((tf - t0).abs() * 0.01).clamp(1e-5, 0.1);
    let rtol = 1e-5;
    let atol = 1e-7;

    let mut ts = vec![t];
    let mut ys = vec![y.clone()];

    let mut k1 = eval_rhs(t, &y)?;
    let max_steps = 5000usize;

    for _ in 0..max_steps {
        if dir * (tf - t) <= 1e-12 {
            break;
        }
        if dir * (t + h - tf) > 0.0 {
            h = tf - t;
        }

        // Dormand-Prince stages
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

        // 5th-order solution
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

        // 4th-order error estimate
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

        if err_norm <= 1.0 || h.abs() <= 1e-12 {
            t += h;
            y = y_next;
            k1 = k7;
            ts.push(t);
            ys.push(y.clone());
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

/// Adaptive Simpson's quadrature: `q = integral(f, a, b)`.
pub fn solve_integral(
    vm: &mut Vm,
    cl: &ClosureHandle,
    a: f64,
    b: f64,
    stdout: &mut String,
) -> Result<f64> {
    if (a - b).abs() < 1e-15 {
        return Ok(0.0);
    }
    let eval_one = |vm: &mut Vm, x: f64, stdout: &mut String| -> Result<f64> {
        let rets = vm.call_closure(cl, &[Value::scalar(x)], 1, stdout)?;
        rets.into_iter()
            .next()
            .unwrap_or_else(Value::empty)
            .as_scalar()
    };

    #[allow(clippy::too_many_arguments)]
    fn recurse(
        vm: &mut Vm,
        cl: &ClosureHandle,
        stdout: &mut String,
        a: f64,
        b: f64,
        fa: f64,
        fm: f64,
        fb: f64,
        whole: f64,
        tol: f64,
        depth: usize,
    ) -> Result<f64> {
        let m = 0.5 * (a + b);
        let lm = 0.5 * (a + m);
        let rm = 0.5 * (m + b);
        let flm = vm
            .call_closure(cl, &[Value::scalar(lm)], 1, stdout)?
            .into_iter()
            .next()
            .unwrap_or_else(Value::empty)
            .as_scalar()?;
        let frm = vm
            .call_closure(cl, &[Value::scalar(rm)], 1, stdout)?
            .into_iter()
            .next()
            .unwrap_or_else(Value::empty)
            .as_scalar()?;

        let left = (m - a) / 6.0 * (fa + 4.0 * flm + fm);
        let right = (b - m) / 6.0 * (fm + 4.0 * frm + fb);
        let delta = left + right - whole;
        if depth == 0 || delta.abs() <= 15.0 * tol {
            Ok(left + right + delta / 15.0)
        } else {
            let l = recurse(vm, cl, stdout, a, m, fa, flm, fm, left, tol * 0.5, depth - 1)?;
            let r = recurse(vm, cl, stdout, m, b, fm, frm, fb, right, tol * 0.5, depth - 1)?;
            Ok(l + r)
        }
    }

    let fa = eval_one(vm, a, stdout)?;
    let fb = eval_one(vm, b, stdout)?;
    let m = 0.5 * (a + b);
    let fm = eval_one(vm, m, stdout)?;
    let whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb);
    recurse(vm, cl, stdout, a, b, fa, fm, fb, whole, 1e-9, 14)
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
