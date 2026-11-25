//! Pure-Rust, zero-dependency special mathematical functions:
//! `beta`, `betaln`, `erfinv`, `erfcinv`, `besselj`, `bessely`, `besseli`, `besselk`,
//! `legendre`, `ellipke`.

use std::f64::consts::PI;

use matro_types::{MatroError, Matrix, Result, Value};

use crate::stats::erf_f64;
use crate::stats_ml::{gamma_f64, lgamma_f64};
use crate::{as_usize, ok1, value_as_text};

pub fn fn_beta(args: &[Value], is_log: bool) -> Result<Vec<Value>> {
    let name = if is_log { "betaln" } else { "beta" };
    if args.len() != 2 {
        return Err(MatroError::arity(name, "2 (Z, W)", args.len()));
    }
    let z = args[0].to_dense_matrix()?;
    let w = args[1].to_dense_matrix()?;
    let out = z.elementwise(&w, |a, b| {
        let lb = lgamma_f64(a) + lgamma_f64(b) - lgamma_f64(a + b);
        if is_log { lb } else { lb.exp() }
    })?;
    ok1(Value::matrix(out))
}

/// High-precision `erfinv(y)` for `y in [-1, 1]` using Winitzki initial guess + Halley refinement.
pub fn erfinv_f64(y: f64) -> f64 {
    if y.is_nan() || y < -1.0 || y > 1.0 {
        return f64::NAN;
    }
    if y == 0.0 {
        return 0.0;
    }
    if y == 1.0 {
        return f64::INFINITY;
    }
    if y == -1.0 {
        return f64::NEG_INFINITY;
    }
    let sign = y.signum();
    let ay = y.abs();
    let a = 0.147;
    let ln1m = (1.0 - ay * ay).ln();
    let t1 = 2.0 / (PI * a) + 0.5 * ln1m;
    let mut x = sign * ((t1 * t1 - ln1m / a).sqrt() - t1).sqrt();
    let two_over_sqrt_pi = 2.0 / PI.sqrt();
    for _ in 0..6 {
        let err = erf_f64(x) - y;
        let deriv = two_over_sqrt_pi * (-x * x).exp();
        if deriv == 0.0 {
            break;
        }
        let u = err / deriv;
        let step = u / (1.0 + x * u);
        x -= step;
        if step.abs() <= 1e-15 * x.abs().max(1.0) {
            break;
        }
    }
    x
}

pub fn erfcinv_f64(y: f64) -> f64 {
    if y.is_nan() || y < 0.0 || y > 2.0 {
        return f64::NAN;
    }
    if y == 0.0 {
        return f64::INFINITY;
    }
    if y == 2.0 {
        return f64::NEG_INFINITY;
    }
    if y == 1.0 {
        return 0.0;
    }
    if y > 0.5 && y < 1.5 {
        return erfinv_f64(1.0 - y);
    }
    // Tail approximation for small y (or 2 - y) + Halley refinement
    let (sgn, q) = if y < 1.0 { (1.0, y) } else { (-1.0, 2.0 - y) };
    let t = (-q.ln()).sqrt();
    let mut x = t - (0.5 * (t.ln() + 0.5 * PI.ln())) / t;
    let two_over_sqrt_pi = 2.0 / PI.sqrt();
    for _ in 0..8 {
        let erfc_x = 1.0 - erf_f64(x);
        let err = erfc_x - q;
        let deriv = -two_over_sqrt_pi * (-x * x).exp();
        if deriv == 0.0 {
            break;
        }
        let u = err / deriv;
        let step = u / (1.0 + x * u);
        x -= step;
        if step.abs() <= 1e-15 * x.abs().max(1.0) {
            break;
        }
    }
    sgn * x
}

pub fn fn_erfinv(args: &[Value], complementary: bool) -> Result<Vec<Value>> {
    let name = if complementary { "erfcinv" } else { "erfinv" };
    if args.len() != 1 {
        return Err(MatroError::arity(name, "1 (y)", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let out = m.map(|y| if complementary { erfcinv_f64(y) } else { erfinv_f64(y) });
    ok1(Value::matrix(out))
}

/// Bessel function of the first kind `J_nu(x)` for real `nu`, `x >= 0` (and integer `nu` for `x < 0`).
pub fn besselj_f64(nu: f64, x: f64) -> f64 {
    if nu.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        if (nu - nu.round()).abs() < 1e-12 {
            let sgn = if (nu.round() as i64) % 2 == 0 { 1.0 } else { -1.0 };
            return sgn * besselj_f64(nu, -x);
        }
        return f64::NAN;
    }
    if nu < 0.0 && (nu - nu.round()).abs() < 1e-12 {
        let sgn = if ((-nu).round() as i64) % 2 == 0 { 1.0 } else { -1.0 };
        return sgn * besselj_f64(-nu, x);
    }
    if x == 0.0 {
        return if nu == 0.0 { 1.0 } else if nu > 0.0 { 0.0 } else { f64::INFINITY };
    }
    // Asymptotic Hankel expansion for large x
    if x > 25.0 + nu.abs() * 1.5 {
        let mu = 4.0 * nu * nu;
        let z8 = 8.0 * x;
        let mut p = 1.0;
        let mut q = (mu - 1.0) / z8;
        let mut term = 1.0;
        for k in 1..=6 {
            let kf = k as f64;
            let num = (mu - (2.0 * kf - 1.0).powi(2)) * (mu - (2.0 * kf + 1.0).powi(2));
            let _ = num;
            term *= -(mu - (4.0 * kf - 3.0).powi(2)) * (mu - (4.0 * kf - 1.0).powi(2))
                / ((2.0 * kf - 1.0) * (2.0 * kf) * z8 * z8);
            p += term;
            if term.abs() < 1e-16 * p.abs() {
                break;
            }
        }
        let mut qterm = (mu - 1.0) / z8;
        for k in 1..=6 {
            let kf = k as f64;
            qterm *= -(mu - (4.0 * kf - 1.0).powi(2)) * (mu - (4.0 * kf + 1.0).powi(2))
                / ((2.0 * kf) * (2.0 * kf + 1.0) * z8 * z8);
            q += qterm;
            if qterm.abs() < 1e-16 * q.abs().max(1e-30) {
                break;
            }
        }
        let chi = x - (0.5 * nu + 0.25) * PI;
        return (2.0 / (PI * x)).sqrt() * (p * chi.cos() - q * chi.sin());
    }
    // Power series: sum_{k=0}^\infty (-1)^k / (k! Gamma(nu + k + 1)) * (x/2)^(2k + nu)
    let half_x = 0.5 * x;
    let x2_4 = -0.25 * x * x;
    let mut term = half_x.powf(nu) / gamma_f64(nu + 1.0);
    let mut sum = term;
    for k in 1..200 {
        let kf = k as f64;
        term *= x2_4 / (kf * (nu + kf));
        sum += term;
        if term.abs() <= 1e-16 * sum.abs().max(1e-300) {
            break;
        }
    }
    sum
}

/// Bessel function of the second kind `Y_nu(x)` for `x > 0`.
pub fn bessely_f64(nu: f64, x: f64) -> f64 {
    if nu.is_nan() || x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x > 25.0 + nu.abs() * 1.5 {
        let mu = 4.0 * nu * nu;
        let z8 = 8.0 * x;
        let mut p = 1.0;
        let mut term = 1.0;
        for k in 1..=6 {
            let kf = k as f64;
            term *= -(mu - (4.0 * kf - 3.0).powi(2)) * (mu - (4.0 * kf - 1.0).powi(2))
                / ((2.0 * kf - 1.0) * (2.0 * kf) * z8 * z8);
            p += term;
            if term.abs() < 1e-16 * p.abs() {
                break;
            }
        }
        let mut q = (mu - 1.0) / z8;
        let mut qterm = q;
        for k in 1..=6 {
            let kf = k as f64;
            qterm *= -(mu - (4.0 * kf - 1.0).powi(2)) * (mu - (4.0 * kf + 1.0).powi(2))
                / ((2.0 * kf) * (2.0 * kf + 1.0) * z8 * z8);
            q += qterm;
            if qterm.abs() < 1e-16 * q.abs().max(1e-30) {
                break;
            }
        }
        let chi = x - (0.5 * nu + 0.25) * PI;
        return (2.0 / (PI * x)).sqrt() * (p * chi.sin() + q * chi.cos());
    }
    if (nu - nu.round()).abs() > 1e-10 {
        let s = (PI * nu).sin();
        return (besselj_f64(nu, x) * (PI * nu).cos() - besselj_f64(-nu, x)) / s;
    }
    // Integer order n: Y_0, Y_1 via series + upward recurrence
    let n = nu.round() as i64;
    let abs_n = n.unsigned_abs() as usize;
    let gamma_em = 0.5772156649015328606_f64;
    let half_x = 0.5 * x;
    let x2_4 = -0.25 * x * x;
    // Y_0(x) = (2/pi) * ((ln(x/2) + gamma)*J_0(x) - sum_{k=1}^\infty (-1)^k H_k (x/2)^{2k} / (k!)^2)
    let mut sum_h = 0.0;
    let mut h_k = 0.0;
    let mut term = 1.0;
    for k in 1..150 {
        let kf = k as f64;
        h_k += 1.0 / kf;
        term *= x2_4 / (kf * kf);
        let contrib = term * h_k;
        sum_h += contrib;
        if contrib.abs() <= 1e-16 * sum_h.abs().max(1e-300) {
            break;
        }
    }
    let y0 = (2.0 / PI) * ((half_x.ln() + gamma_em) * besselj_f64(0.0, x) - sum_h);
    if abs_n == 0 {
        return y0;
    }
    // Y_1 via Wronskian: J_1(x)*Y_0(x) - J_0(x)*Y_1(x) = 2/(pi*x)
    // Wait: if J_0(x) is near 0, use series for Y_1 instead!
    let j0 = besselj_f64(0.0, x);
    let y1 = if j0.abs() > 0.1 {
        (besselj_f64(1.0, x) * y0 - 2.0 / (PI * x)) / j0
    } else {
        // Direct series for Y_1(x)
        let mut s1 = 0.0;
        let mut hk = 0.0;
        let mut t1 = half_x;
        for k in 0..150 {
            let kf = k as f64;
            if k > 0 {
                hk += 1.0 / kf;
                t1 *= x2_4 / (kf * (kf + 1.0));
            }
            let hkp1 = hk + 1.0 / (kf + 1.0);
            let contrib = t1 * (hk + hkp1);
            s1 += contrib;
            if k > 2 && contrib.abs() <= 1e-16 * s1.abs().max(1e-300) {
                break;
            }
        }
        (2.0 / PI) * ((half_x.ln() + gamma_em) * besselj_f64(1.0, x) - 1.0 / x - 0.5 * s1)
    };
    let mut y_prev = y0;
    let mut y_cur = y1;
    for k in 1..abs_n {
        let y_next = (2.0 * (k as f64) / x) * y_cur - y_prev;
        y_prev = y_cur;
        y_cur = y_next;
    }
    if n < 0 && abs_n % 2 == 1 {
        -y_cur
    } else {
        y_cur
    }
}

/// Modified Bessel function of the first kind `I_nu(x)`.
pub fn besseli_f64(nu: f64, x: f64) -> f64 {
    if nu.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        if (nu - nu.round()).abs() < 1e-12 {
            let sgn = if (nu.round() as i64) % 2 == 0 { 1.0 } else { -1.0 };
            return sgn * besseli_f64(nu, -x);
        }
        return f64::NAN;
    }
    if nu < 0.0 && (nu - nu.round()).abs() < 1e-12 {
        return besseli_f64(-nu, x);
    }
    if x == 0.0 {
        return if nu == 0.0 { 1.0 } else if nu > 0.0 { 0.0 } else { f64::INFINITY };
    }
    if x > 30.0 + nu.abs() {
        let mu = 4.0 * nu * nu;
        let z8 = 8.0 * x;
        let mut s = 1.0;
        let mut term = 1.0;
        for k in 1..=12 {
            let kf = k as f64;
            term *= -(mu - (2.0 * kf - 1.0).powi(2)) / (kf * z8);
            s += term;
            if term.abs() < 1e-16 * s.abs() {
                break;
            }
        }
        return (x.exp() / (2.0 * PI * x).sqrt()) * s;
    }
    let half_x = 0.5 * x;
    let x2_4 = 0.25 * x * x;
    let mut term = half_x.powf(nu) / gamma_f64(nu + 1.0);
    let mut sum = term;
    for k in 1..200 {
        let kf = k as f64;
        term *= x2_4 / (kf * (nu + kf));
        sum += term;
        if term.abs() <= 1e-16 * sum.abs().max(1e-300) {
            break;
        }
    }
    sum
}

/// Modified Bessel function of the second kind `K_nu(x)` for `x > 0`.
pub fn besselk_f64(nu: f64, x: f64) -> f64 {
    if nu.is_nan() || x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    let nu = nu.abs();
    if x > 20.0 + nu {
        let mu = 4.0 * nu * nu;
        let z8 = 8.0 * x;
        let mut s = 1.0;
        let mut term = 1.0;
        for k in 1..=12 {
            let kf = k as f64;
            term *= (mu - (2.0 * kf - 1.0).powi(2)) / (kf * z8);
            s += term;
            if term.abs() < 1e-16 * s.abs() {
                break;
            }
        }
        return (PI / (2.0 * x)).sqrt() * (-x).exp() * s;
    }
    if (nu - nu.round()).abs() > 1e-8 {
        let s = (PI * nu).sin();
        return 0.5 * PI * (besseli_f64(-nu, x) - besseli_f64(nu, x)) / s;
    }
    // Integer order n via K_0, K_1 series + upward recurrence (always stable for K_n!)
    let n = nu.round() as usize;
    let gamma_em = 0.5772156649015328606_f64;
    let half_x = 0.5 * x;
    let x2_4 = 0.25 * x * x;
    let k0 = if x <= 2.0 {
        let mut sum_h = 0.0;
        let mut hk = 0.0;
        let mut term = 1.0;
        for k in 1..60 {
            let kf = k as f64;
            hk += 1.0 / kf;
            term *= x2_4 / (kf * kf);
            let contrib = term * hk;
            sum_h += contrib;
            if contrib.abs() <= 1e-16 * sum_h.abs().max(1e-300) {
                break;
            }
        }
        -(half_x.ln() + gamma_em) * besseli_f64(0.0, x) + sum_h
    } else {
        // Numerical integral K_0(x) = int_0^inf exp(-x cosh(t)) dt via trapezoidal rule (exponentially accurate!)
        let h = 0.15;
        let mut acc = 0.5 * (-x).exp();
        for m in 1..80 {
            let t = (m as f64) * h;
            let v = (-x * t.cosh()).exp();
            acc += v;
            if v < 1e-18 * acc {
                break;
            }
        }
        h * acc
    };
    if n == 0 {
        return k0;
    }
    // K_1 via integral K_1(x) = int_0^inf exp(-x cosh(t)) cosh(t) dt (or Wronskian I_0*K_1 + I_1*K_0 = 1/x)
    let k1 = (1.0 / x - besseli_f64(1.0, x) * k0) / besseli_f64(0.0, x);
    let mut k_prev = k0;
    let mut k_cur = k1;
    for j in 1..n {
        let k_next = (2.0 * (j as f64) / x) * k_cur + k_prev;
        k_prev = k_cur;
        k_cur = k_next;
    }
    k_cur
}

pub fn fn_bessel(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity(name, "2 or 3 (nu, Z [, scale])", args.len()));
    }
    let nu = args[0].to_dense_matrix()?;
    let z = args[1].to_dense_matrix()?;
    let scaled = args.get(2).and_then(|v| v.as_scalar().ok()).unwrap_or(0.0) != 0.0;
    let out = nu.elementwise(&z, |n, x| {
        let val = match name {
            "besselj" => besselj_f64(n, x),
            "bessely" => bessely_f64(n, x),
            "besseli" => besseli_f64(n, x),
            "besselk" => besselk_f64(n, x),
            _ => f64::NAN,
        };
        if scaled {
            match name {
                "besseli" => val * (-x.abs()).exp(),
                "besselk" => val * x.exp(),
                _ => val,
            }
        } else {
            val
        }
    })?;
    ok1(Value::matrix(out))
}

/// Associated Legendre functions `P = legendre(n, X [, 'unnorm'|'sch'|'norm'])`.
pub fn fn_legendre(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity("legendre", "2 or 3 (n, X [, normalize])", args.len()));
    }
    let n = as_usize(&args[0], "legendre")?;
    let x = args[1].to_dense_matrix()?;
    let mode = args
        .get(2)
        .and_then(value_as_text)
        .unwrap_or_else(|| "unnorm".to_string())
        .to_ascii_lowercase();
    let nx = x.len();
    let mut out = Matrix::zeros(n + 1, nx);
    for (col, &xv) in x.data().iter().enumerate() {
        if xv.abs() > 1.0 || xv.is_nan() {
            for m in 0..=n {
                out.set(m, col, f64::NAN);
            }
            continue;
        }
        let s = (1.0 - xv * xv).max(0.0).sqrt();
        // Compute P_n^m(x) for m = 0..=n with Condon-Shortley phase (-1)^m
        for m in 0..=n {
            let mut pmm = 1.0;
            for k in 1..=m {
                pmm *= -(2.0 * (k as f64) - 1.0) * s;
            }
            let val = if n == m {
                pmm
            } else {
                let pmmp1 = xv * (2.0 * (m as f64) + 1.0) * pmm;
                if n == m + 1 {
                    pmmp1
                } else {
                    let mut p_prev2 = pmm;
                    let mut p_prev1 = pmmp1;
                    let mut p_cur = pmmp1;
                    for ll in (m + 2)..=n {
                        let lf = ll as f64;
                        let mf = m as f64;
                        p_cur = ((2.0 * lf - 1.0) * xv * p_prev1 - (lf + mf - 1.0) * p_prev2) / (lf - mf);
                        p_prev2 = p_prev1;
                        p_prev1 = p_cur;
                    }
                    p_cur
                }
            };
            let scaled = match mode.as_str() {
                "sch" => {
                    if m == 0 {
                        val
                    } else {
                        let ratio = (lgamma_f64((n - m + 1) as f64) - lgamma_f64((n + m + 1) as f64)).exp();
                        let sgn = if m % 2 == 1 { -1.0 } else { 1.0 };
                        sgn * (2.0 * ratio).sqrt() * val
                    }
                }
                "norm" => {
                    let ratio = (lgamma_f64((n - m + 1) as f64) - lgamma_f64((n + m + 1) as f64)).exp();
                    let sgn = if m % 2 == 1 { -1.0 } else { 1.0 };
                    sgn * ((n as f64 + 0.5) * ratio).sqrt() * val
                }
                _ => val,
            };
            out.set(m, col, scaled);
        }
    }
    ok1(Value::matrix(out))
}

/// Complete elliptic integrals of the first and second kind `[K, E] = ellipke(M)` via AGM.
pub fn fn_ellipke(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("ellipke", "1 or 2 (M [, tol])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let tol = args.get(1).and_then(|v| v.as_scalar().ok()).unwrap_or(f64::EPSILON);
    let mut k_vals = Vec::with_capacity(m.len());
    let mut e_vals = Vec::with_capacity(m.len());
    for &mv in m.data() {
        if mv.is_nan() || mv > 1.0 {
            k_vals.push(f64::NAN);
            e_vals.push(f64::NAN);
            continue;
        }
        if mv == 1.0 {
            k_vals.push(f64::INFINITY);
            e_vals.push(1.0);
            continue;
        }
        let mut a = 1.0_f64;
        let mut b = (1.0 - mv).sqrt();
        let mut c_sq_sum = mv;
        let mut pow2 = 1.0_f64;
        for _ in 0..50 {
            let c = 0.5 * (a - b);
            let a_new = 0.5 * (a + b);
            let b_new = (a * b).sqrt();
            pow2 *= 2.0;
            c_sq_sum += pow2 * c * c;
            a = a_new;
            b = b_new;
            if c.abs() <= tol * a {
                break;
            }
        }
        let k = 0.5 * PI / a;
        let e = k * (1.0 - 0.5 * c_sq_sum);
        k_vals.push(k);
        e_vals.push(e);
    }
    let k_mat = Matrix::from_nd_col_major(m.dims(), k_vals)?;
    if nargout >= 2 {
        let e_mat = Matrix::from_nd_col_major(m.dims(), e_vals)?;
        Ok(vec![Value::matrix(k_mat), Value::matrix(e_mat)])
    } else {
        ok1(Value::matrix(k_mat))
    }
}
