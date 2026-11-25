//! Statistics & Machine Learning Toolbox (pure zero-dependency Rust):
//! - Special functions: `lgamma`, `gamma`, `beta`, `gammainc`, `betainc`
//! - Distributions: `tpdf`/`tcdf`/`tinv`, `chi2pdf`/`chi2cdf`/`chi2inv`, `fpdf`/`fcdf`,
//!   `exppdf`/`expcdf`/`expinv`, `binopdf`/`binocdf`, `poisspdf`/`poisscdf`
//! - Descriptive stats: `zscore`, `skewness`, `kurtosis`, `iqr`, `mad`
//! - Regression & Hypothesis tests: `fitlm`, `regress`, `anova1`, `ttest2`
//! - Supervised & Unsupervised ML: `fitcknn`, `fitcsvm`, `fitcnb`, `predict`,
//!   `pdist`, `squareform`, `linkage`, `cluster`

use std::collections::BTreeMap;

use matro_types::{MatroError, Matrix, ObjectInstance, Result, Value};

fn ok1(v: Value) -> Result<Vec<Value>> {
    Ok(vec![v])
}

// ---------------------------------------------------------------------------
// Special Functions: Lanczos lgamma, Regularized Incomplete Gamma & Beta
// ---------------------------------------------------------------------------

pub fn lgamma_f64(z: f64) -> f64 {
    const COEF: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if z < 0.5 {
        std::f64::consts::PI.ln()
            - (std::f64::consts::PI * z).sin().abs().ln()
            - lgamma_f64(1.0 - z)
    } else {
        let z = z - 1.0;
        let mut x = COEF[0];
        for (i, &c) in COEF.iter().enumerate().skip(1) {
            x += c / (z + i as f64);
        }
        let t = z + 7.5;
        0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
    }
}

/// Regularized lower incomplete gamma P(a, x) = (1/Gamma(a)) * int_0^x t^{a-1} e^{-t} dt.
pub fn gammainc_lower(x: f64, a: f64) -> f64 {
    if a <= 0.0 || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    let gln = lgamma_f64(a);
    if x < a + 1.0 {
        // Series representation
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..300 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-14 {
                break;
            }
        }
        (sum * (-x + a * x.ln() - gln).exp()).clamp(0.0, 1.0)
    } else {
        // Lentz continued fraction for upper Q(a, x), then 1 - Q
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / 1e-30;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..=300 {
            let an = -(i as f64) * ((i as f64) - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < 1e-30 {
                d = 1e-30;
            }
            c = b + an / c;
            if c.abs() < 1e-30 {
                c = 1e-30;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-14 {
                break;
            }
        }
        let q = ((-x + a * x.ln() - gln).exp() * h).clamp(0.0, 1.0);
        (1.0 - q).clamp(0.0, 1.0)
    }
}

/// Regularized incomplete beta I_x(a, b) via Lentz's continued fraction.
pub fn betainc_reg(x: f64, a: f64, b: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 || x < 0.0 || x > 1.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }
    let bt = (lgamma_f64(a + b) - lgamma_f64(a) - lgamma_f64(b)
        + a * x.ln()
        + b * (1.0 - x).ln())
    .exp();
    let cf = |x: f64, a: f64, b: f64| -> f64 {
        let qab = a + b;
        let qap = a + 1.0;
        let qam = a - 1.0;
        let mut c = 1.0;
        let mut d = 1.0 - qab * x / qap;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        d = 1.0 / d;
        let mut h = d;
        for m in 1..=300 {
            let mf = m as f64;
            let m2 = 2.0 * mf;
            let aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d;
            if d.abs() < 1e-30 {
                d = 1e-30;
            }
            c = 1.0 + aa / c;
            if c.abs() < 1e-30 {
                c = 1e-30;
            }
            d = 1.0 / d;
            h *= d * c;
            let aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d;
            if d.abs() < 1e-30 {
                d = 1e-30;
            }
            c = 1.0 + aa / c;
            if c.abs() < 1e-30 {
                c = 1e-30;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-14 {
                break;
            }
        }
        h
    };
    if x < (a + 1.0) / (a + b + 2.0) {
        (bt * cf(x, a, b) / a).clamp(0.0, 1.0)
    } else {
        (1.0 - bt * cf(1.0 - x, b, a) / b).clamp(0.0, 1.0)
    }
}

pub fn gamma_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        if x == x.floor() {
            return f64::INFINITY;
        }
        let s = (std::f64::consts::PI * x).sin();
        return std::f64::consts::PI / (s * lgamma_f64(1.0 - x).exp());
    }
    lgamma_f64(x).exp()
}

pub fn fn_gamma_fn(args: &[Value], is_log: bool) -> Result<Vec<Value>> {
    let fname = if is_log { "gammaln" } else { "gamma" };
    if args.len() != 1 {
        return Err(MatroError::arity(fname, "1", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let out = m.map(|x| {
        if is_log {
            lgamma_f64(x)
        } else {
            gamma_f64(x)
        }
    });
    ok1(Value::matrix(out))
}

pub fn fn_gammainc(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity("gammainc", "2 or 3 (x, a [, 'lower'|'upper'])", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let a = args[1].to_dense_matrix()?;
    let upper = args
        .get(2)
        .and_then(crate::value_as_text)
        .is_some_and(|s| s.eq_ignore_ascii_case("upper"));
    ok1(Value::matrix(x.elementwise(&a, |xv, av| {
        let p = gammainc_lower(xv, av);
        if upper { 1.0 - p } else { p }
    })?))
}

pub fn fn_betainc(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 3 || args.len() > 4 {
        return Err(MatroError::arity("betainc", "3 or 4 (x, a, b [, 'lower'|'upper'])", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let a = args[1].to_dense_matrix()?;
    let b = args[2].to_dense_matrix()?;
    let upper = args
        .get(3)
        .and_then(crate::value_as_text)
        .is_some_and(|s| s.eq_ignore_ascii_case("upper"));
    let n = x.len().max(a.len()).max(b.len());
    let dims = if x.len() == n {
        x.dims()
    } else if a.len() == n {
        a.dims()
    } else {
        b.dims()
    };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let xv = if x.len() == 1 { x.data()[0] } else { x.data()[i] };
        let av = if a.len() == 1 { a.data()[0] } else { a.data()[i] };
        let bv = if b.len() == 1 { b.data()[0] } else { b.data()[i] };
        let p = betainc_reg(xv, av, bv);
        out.push(if upper { 1.0 - p } else { p });
    }
    ok1(Value::matrix(Matrix::from_nd_col_major(dims, out)?))
}

// ---------------------------------------------------------------------------
// Probability Distributions (t, chi2, F, exp, bino, poiss)
// ---------------------------------------------------------------------------

pub fn tcdf_scalar(t: f64, nu: f64) -> f64 {
    if nu <= 0.0 {
        return f64::NAN;
    }
    let x = nu / (nu + t * t);
    let ib = betainc_reg(x, 0.5 * nu, 0.5);
    if t >= 0.0 {
        1.0 - 0.5 * ib
    } else {
        0.5 * ib
    }
}

pub fn tinv_scalar(p: f64, nu: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 || nu <= 0.0 {
        return if p == 0.0 {
            f64::NEG_INFINITY
        } else if p == 1.0 {
            f64::INFINITY
        } else {
            f64::NAN
        };
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }
    let sign = if p < 0.5 { -1.0 } else { 1.0 };
    let target = if p < 0.5 { 1.0 - p } else { p };
    let mut lo = 0.0;
    let mut hi = 1.0;
    while tcdf_scalar(hi, nu) < target && hi < 1e8 {
        hi *= 2.0;
    }
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if tcdf_scalar(mid, nu) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    sign * 0.5 * (lo + hi)
}

pub fn fcdf_scalar(x: f64, d1: f64, d2: f64) -> f64 {
    if d1 <= 0.0 || d2 <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    let z = (d1 * x) / (d1 * x + d2);
    betainc_reg(z, 0.5 * d1, 0.5 * d2)
}

fn broadcast_2(a: &Matrix, b: &Matrix, f: impl Fn(f64, f64) -> f64) -> Result<Matrix> {
    a.elementwise(b, f)
}

fn broadcast_3(a: &Matrix, b: &Matrix, c: &Matrix, f: impl Fn(f64, f64, f64) -> f64) -> Result<Matrix> {
    let n = a.len().max(b.len()).max(c.len());
    let dims = if a.len() == n {
        a.dims()
    } else if b.len() == n {
        b.dims()
    } else {
        c.dims()
    };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let av = if a.len() == 1 { a.data()[0] } else { a.data()[i] };
        let bv = if b.len() == 1 { b.data()[0] } else { b.data()[i] };
        let cv = if c.len() == 1 { c.data()[0] } else { c.data()[i] };
        out.push(f(av, bv, cv));
    }
    Matrix::from_nd_col_major(dims, out)
}

pub fn fn_t_dist(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity(name, "2 (x, nu)", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let nu = args[1].to_dense_matrix()?;
    let out = match name {
        "tpdf" => broadcast_2(&x, &nu, |t, v| {
            if v <= 0.0 || v.is_nan() {
                f64::NAN
            } else {
                let log_c = lgamma_f64(0.5 * (v + 1.0))
                    - lgamma_f64(0.5 * v)
                    - 0.5 * (v * std::f64::consts::PI).ln();
                (log_c - 0.5 * (v + 1.0) * (1.0 + t * t / v).ln()).exp()
            }
        })?,
        "tcdf" => broadcast_2(&x, &nu, tcdf_scalar)?,
        "tinv" => broadcast_2(&x, &nu, tinv_scalar)?,
        _ => unreachable!(),
    };
    ok1(Value::matrix(out))
}

pub fn chi2inv_scalar(p: f64, k: f64) -> f64 {
    if p.is_nan() || k.is_nan() || k <= 0.0 || p < 0.0 || p > 1.0 {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    let mut lo = 0.0;
    let mut hi = k.max(1.0);
    while gammainc_lower(0.5 * hi, 0.5 * k) < p && hi < 1e9 {
        hi *= 2.0;
    }
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if gammainc_lower(0.5 * mid, 0.5 * k) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

pub fn fn_chi2_dist(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity(name, "2 (x, k)", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let k = args[1].to_dense_matrix()?;
    let out = match name {
        "chi2pdf" => broadcast_2(&x, &k, |v, df| {
            if df <= 0.0 || df.is_nan() || v.is_nan() {
                f64::NAN
            } else if v <= 0.0 {
                0.0
            } else {
                let half_k = 0.5 * df;
                let log_c = -half_k * 2.0f64.ln() - lgamma_f64(half_k);
                (log_c + (half_k - 1.0) * v.ln() - 0.5 * v).exp()
            }
        })?,
        "chi2cdf" => broadcast_2(&x, &k, |v, df| {
            if df <= 0.0 || df.is_nan() || v.is_nan() {
                f64::NAN
            } else if v <= 0.0 {
                0.0
            } else {
                gammainc_lower(0.5 * v, 0.5 * df)
            }
        })?,
        "chi2inv" => broadcast_2(&x, &k, chi2inv_scalar)?,
        _ => unreachable!(),
    };
    ok1(Value::matrix(out))
}

pub fn finv_scalar(p: f64, d1: f64, d2: f64) -> f64 {
    if p.is_nan() || d1 <= 0.0 || d2 <= 0.0 || p < 0.0 || p > 1.0 {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    // Solve betainc_reg(z, d1/2, d2/2) = p via bisection on z in (0, 1), then x = d2*z / (d1*(1-z))
    let mut lo = 0.0;
    let mut hi = 1.0;
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if betainc_reg(mid, 0.5 * d1, 0.5 * d2) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let z = 0.5 * (lo + hi);
    (d2 * z) / (d1 * (1.0 - z).max(1e-300))
}

pub fn fn_f_dist(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 3 {
        return Err(MatroError::arity(name, "3 (x, d1, d2)", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let d1 = args[1].to_dense_matrix()?;
    let d2 = args[2].to_dense_matrix()?;
    let out = match name {
        "fpdf" => broadcast_3(&x, &d1, &d2, |v, v1, v2| {
            if v1 <= 0.0 || v2 <= 0.0 || v.is_nan() {
                f64::NAN
            } else if v <= 0.0 {
                0.0
            } else {
                let log_c = lgamma_f64(0.5 * (v1 + v2))
                    - lgamma_f64(0.5 * v1)
                    - lgamma_f64(0.5 * v2)
                    + 0.5 * v1 * (v1 / v2).ln();
                (log_c + (0.5 * v1 - 1.0) * v.ln() - 0.5 * (v1 + v2) * (1.0 + v1 * v / v2).ln())
                    .exp()
            }
        })?,
        "fcdf" => broadcast_3(&x, &d1, &d2, fcdf_scalar)?,
        "finv" => broadcast_3(&x, &d1, &d2, finv_scalar)?,
        _ => unreachable!(),
    };
    ok1(Value::matrix(out))
}

pub fn fn_exp_dist(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity(name, "1 or 2 (x [, mu])", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let mu = args
        .get(1)
        .map(|v| v.to_dense_matrix())
        .transpose()?
        .unwrap_or_else(|| Matrix::scalar(1.0));
    let out = match name {
        "exppdf" => broadcast_2(&x, &mu, |v, m| {
            if m <= 0.0 || m.is_nan() || v.is_nan() {
                f64::NAN
            } else if v < 0.0 {
                0.0
            } else {
                (-v / m).exp() / m
            }
        })?,
        "expcdf" => broadcast_2(&x, &mu, |v, m| {
            if m <= 0.0 || m.is_nan() || v.is_nan() {
                f64::NAN
            } else if v < 0.0 {
                0.0
            } else {
                1.0 - (-v / m).exp()
            }
        })?,
        "expinv" => broadcast_2(&x, &mu, |p, m| {
            if m <= 0.0 || m.is_nan() || p.is_nan() || p < 0.0 || p > 1.0 {
                f64::NAN
            } else {
                -m * (1.0 - p).ln()
            }
        })?,
        _ => unreachable!(),
    };
    ok1(Value::matrix(out))
}

fn binocdf_scalar(k: f64, n: f64, p: f64) -> f64 {
    if k.is_nan() || n.is_nan() || p.is_nan() || n < 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    let nr = n.round();
    if k < 0.0 {
        return 0.0;
    }
    if k >= nr {
        return 1.0;
    }
    let kf = k.floor();
    betainc_reg(1.0 - p, nr - kf, kf + 1.0)
}

pub fn fn_bino_dist(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 3 {
        return Err(MatroError::arity(name, "3 (x, n, p)", args.len()));
    }
    let kv = args[0].to_dense_matrix()?;
    let nv = args[1].to_dense_matrix()?;
    let pv = args[2].to_dense_matrix()?;
    let out = match name {
        "binopdf" => broadcast_3(&kv, &nv, &pv, |k, n_raw, p| {
            let n = n_raw.round();
            if n < 0.0 || !(0.0..=1.0).contains(&p) {
                return f64::NAN;
            }
            let ki = k.round();
            if (k - ki).abs() > 1e-12 || ki < 0.0 || ki > n {
                return 0.0;
            }
            if p == 0.0 {
                return if ki == 0.0 { 1.0 } else { 0.0 };
            }
            if p == 1.0 {
                return if ki == n { 1.0 } else { 0.0 };
            }
            let log_c = lgamma_f64(n + 1.0) - lgamma_f64(ki + 1.0) - lgamma_f64(n - ki + 1.0);
            (log_c + ki * p.ln() + (n - ki) * (1.0 - p).ln()).exp()
        })?,
        "binocdf" => broadcast_3(&kv, &nv, &pv, binocdf_scalar)?,
        "binoinv" => broadcast_3(&kv, &nv, &pv, |y, n_raw, p| {
            if y.is_nan() || n_raw < 0.0 || !(0.0..=1.0).contains(&p) || y < 0.0 || y > 1.0 {
                return f64::NAN;
            }
            let n = n_raw.round() as usize;
            if y == 0.0 {
                return 0.0;
            }
            if y == 1.0 {
                return n as f64;
            }
            let mut lo = 0usize;
            let mut hi = n;
            while lo < hi {
                let mid = (lo + hi) / 2;
                if binocdf_scalar(mid as f64, n as f64, p) < y {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            lo as f64
        })?,
        _ => unreachable!(),
    };
    ok1(Value::matrix(out))
}

fn poisscdf_scalar(k: f64, lam: f64) -> f64 {
    if k.is_nan() || lam.is_nan() || lam < 0.0 {
        return f64::NAN;
    }
    if k < 0.0 {
        return 0.0;
    }
    if lam == 0.0 {
        return 1.0;
    }
    let kf = k.floor();
    (1.0 - gammainc_lower(lam, kf + 1.0)).clamp(0.0, 1.0)
}

pub fn fn_poiss_dist(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity(name, "2 (x, lambda)", args.len()));
    }
    let kv = args[0].to_dense_matrix()?;
    let lv = args[1].to_dense_matrix()?;
    let out = match name {
        "poisspdf" => broadcast_2(&kv, &lv, |k, lam| {
            if lam < 0.0 || lam.is_nan() || k.is_nan() {
                return f64::NAN;
            }
            let ki = k.round();
            if (k - ki).abs() > 1e-12 || ki < 0.0 {
                return 0.0;
            }
            if lam == 0.0 {
                return if ki == 0.0 { 1.0 } else { 0.0 };
            }
            (-lam + ki * lam.ln() - lgamma_f64(ki + 1.0)).exp()
        })?,
        "poisscdf" => broadcast_2(&kv, &lv, poisscdf_scalar)?,
        "poissinv" => broadcast_2(&kv, &lv, |y, lam| {
            if y.is_nan() || lam.is_nan() || lam < 0.0 || y < 0.0 || y > 1.0 {
                return f64::NAN;
            }
            if y == 0.0 {
                return 0.0;
            }
            if y == 1.0 {
                return f64::INFINITY;
            }
            if lam == 0.0 {
                return 0.0;
            }
            let mut hi = (lam + 10.0 * lam.sqrt() + 20.0).ceil() as usize;
            while poisscdf_scalar(hi as f64, lam) < y && hi < 10_000_000 {
                hi *= 2;
            }
            let mut lo = 0usize;
            while lo < hi {
                let mid = (lo + hi) / 2;
                if poisscdf_scalar(mid as f64, lam) < y {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            lo as f64
        })?,
        _ => unreachable!(),
    };
    ok1(Value::matrix(out))
}

// ---------------------------------------------------------------------------
// Descriptive Stats (`zscore`, `skewness`, `kurtosis`, `iqr`, `mad`)
// ---------------------------------------------------------------------------

pub fn fn_zscore(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity("zscore", "1 to 3 (X [, flag [, dim]])", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let pop = args
        .get(1)
        .map(|v| {
            let m = v.to_dense_matrix()?;
            Ok::<bool, MatroError>(!m.is_empty() && m.data()[0] != 0.0)
        })
        .transpose()?
        .unwrap_or(false);
    let dim_opt = args
        .get(2)
        .map(|v| v.as_scalar().map(|s| s.max(1.0).round() as usize))
        .transpose()?;

    if dim_opt.is_none() && (x.rows() == 1 || x.cols() == 1) {
        let n = x.len() as f64;
        let mu = x.data().iter().sum::<f64>() / n;
        let denom = if pop { n } else { (n - 1.0).max(1.0) };
        let sd = (x.data().iter().map(|&v| (v - mu).powi(2)).sum::<f64>() / denom).sqrt();
        let z = x.map(|v| if sd > 0.0 { (v - mu) / sd } else { 0.0 });
        let mut out = vec![Value::matrix(z)];
        if nargout >= 2 { out.push(Value::scalar(mu)); }
        if nargout >= 3 { out.push(Value::scalar(sd)); }
        return Ok(out);
    }
    let dim = dim_opt.unwrap_or(1);
    let (r, c) = (x.rows(), x.cols());
    if dim == 2 {
        let mut z = Matrix::zeros(r, c);
        let mut mus = Vec::with_capacity(r);
        let mut sds = Vec::with_capacity(r);
        let denom = if pop { c as f64 } else { (c as f64 - 1.0).max(1.0) };
        for row in 0..r {
            let mu = (0..c).map(|col| x.get(row, col)).sum::<f64>() / (c as f64);
            let sd = ((0..c).map(|col| (x.get(row, col) - mu).powi(2)).sum::<f64>() / denom).sqrt();
            mus.push(mu);
            sds.push(sd);
            for col in 0..c {
                z.set(row, col, if sd > 0.0 { (x.get(row, col) - mu) / sd } else { 0.0 });
            }
        }
        let mut out = vec![Value::matrix(z)];
        if nargout >= 2 { out.push(Value::matrix(Matrix::col_vec(mus))); }
        if nargout >= 3 { out.push(Value::matrix(Matrix::col_vec(sds))); }
        return Ok(out);
    }
    let mut z = Matrix::zeros(r, c);
    let mut mus = Vec::with_capacity(c);
    let mut sds = Vec::with_capacity(c);
    let denom = if pop { r as f64 } else { (r as f64 - 1.0).max(1.0) };
    for col in 0..c {
        let mu = (0..r).map(|row| x.get(row, col)).sum::<f64>() / (r as f64);
        let sd = ((0..r).map(|row| (x.get(row, col) - mu).powi(2)).sum::<f64>() / denom).sqrt();
        mus.push(mu);
        sds.push(sd);
        for row in 0..r {
            z.set(row, col, if sd > 0.0 { (x.get(row, col) - mu) / sd } else { 0.0 });
        }
    }
    let mut out = vec![Value::matrix(z)];
    if nargout >= 2 { out.push(Value::matrix(Matrix::row_vec(mus))); }
    if nargout >= 3 { out.push(Value::matrix(Matrix::row_vec(sds))); }
    Ok(out)
}

fn moment_slice(name: &str, raw: &[f64], flag: i32) -> f64 {
    let data: Vec<f64> = raw.iter().copied().filter(|x| !x.is_nan()).collect();
    if data.is_empty() {
        return f64::NAN;
    }
    let n = data.len() as f64;
    match name {
        "skewness" => {
            let mu = data.iter().sum::<f64>() / n;
            let m2 = data.iter().map(|&v| (v - mu).powi(2)).sum::<f64>() / n;
            let m3 = data.iter().map(|&v| (v - mu).powi(3)).sum::<f64>() / n;
            let s1 = m3 / m2.powf(1.5);
            if flag == 0 && data.len() > 2 {
                ((n * (n - 1.0)).sqrt() / (n - 2.0)) * s1
            } else {
                s1
            }
        }
        "kurtosis" => {
            let mu = data.iter().sum::<f64>() / n;
            let m2 = data.iter().map(|&v| (v - mu).powi(2)).sum::<f64>() / n;
            let m4 = data.iter().map(|&v| (v - mu).powi(4)).sum::<f64>() / n;
            let k1 = m4 / (m2 * m2);
            if flag == 0 && data.len() > 3 {
                ((n - 1.0) / ((n - 2.0) * (n - 3.0))) * ((n + 1.0) * k1 - 3.0 * (n - 1.0)) + 3.0
            } else {
                k1
            }
        }
        "mad" => {
            if flag == 1 {
                // Median absolute deviation from median
                let mut s = data.clone();
                s.sort_by(f64::total_cmp);
                let med = if s.len() % 2 == 1 {
                    s[s.len() / 2]
                } else {
                    0.5 * (s[s.len() / 2 - 1] + s[s.len() / 2])
                };
                let mut devs: Vec<f64> = data.iter().map(|&v| (v - med).abs()).collect();
                devs.sort_by(f64::total_cmp);
                if devs.len() % 2 == 1 {
                    devs[devs.len() / 2]
                } else {
                    0.5 * (devs[devs.len() / 2 - 1] + devs[devs.len() / 2])
                }
            } else {
                let mu = data.iter().sum::<f64>() / n;
                data.iter().map(|&v| (v - mu).abs()).sum::<f64>() / n
            }
        }
        "iqr" => {
            crate::stats::slice_quantile(&data, 0.75) - crate::stats::slice_quantile(&data, 0.25)
        }
        _ => unreachable!(),
    }
}

pub fn fn_moment_stat(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity(name, "1 to 3", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let mut all_flag = false;
    let mut flag = if name == "mad" { 0 } else { 1 };
    let mut dim_opt: Option<Vec<usize>> = None;

    if name == "iqr" {
        if let Some(a1) = args.get(1) {
            if crate::value_as_text(a1).is_some_and(|s| s.eq_ignore_ascii_case("all")) {
                all_flag = true;
            } else {
                let dm = a1.to_dense_matrix()?;
                dim_opt = Some(dm.data().iter().map(|&v| v.max(1.0).round() as usize).collect());
            }
        }
    } else {
        if let Some(a1) = args.get(1) {
            if crate::value_as_text(a1).is_some_and(|s| s.eq_ignore_ascii_case("all")) {
                all_flag = true;
            } else {
                let fm = a1.to_dense_matrix()?;
                if !fm.is_empty() {
                    flag = fm.data()[0].round() as i32;
                }
            }
        }
        if let Some(a2) = args.get(2) {
            if crate::value_as_text(a2).is_some_and(|s| s.eq_ignore_ascii_case("all")) {
                all_flag = true;
            } else {
                let dm = a2.to_dense_matrix()?;
                dim_opt = Some(dm.data().iter().map(|&v| v.max(1.0).round() as usize).collect());
            }
        }
    }
    ok1(Value::matrix(crate::stats::reduce_nd_real(
        &x,
        all_flag,
        dim_opt.as_deref(),
        |xs| moment_slice(name, xs, flag),
    )?))
}

// ---------------------------------------------------------------------------
// Regression & ANOVA (`fitlm`, `regress`, `anova1`, `ttest2`)
// ---------------------------------------------------------------------------

/// `mdl = fitlm(X, y)` — Ordinary Least Squares regression with intercept.
pub fn fn_fitlm(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("fitlm", "2 (X, y)", args.len()));
    }
    let x_in = args[0].to_dense_matrix()?;
    let y_in = args[1].to_dense_matrix()?;
    let n = y_in.len();
    let x = if x_in.rows() == 1 && x_in.cols() == n {
        x_in.transpose()
    } else {
        x_in
    };
    if x.rows() != n || n < 2 {
        return Err(MatroError::dimension(
            "fitlm: X must have the same number of rows as length(y)",
        ));
    }
    let p = x.cols();
    let k = p + 1; // intercept + p predictors
    let mut design = Matrix::zeros(n, k);
    for r in 0..n {
        design.set(r, 0, 1.0);
        for c in 0..p {
            design.set(r, c + 1, x.get(r, c));
        }
    }
    let y_col = Matrix::col_vec(y_in.data().to_vec());
    let xt = design.transpose();
    let xtx = xt.matmul(&design)?;
    let xty = xt.matmul(&y_col)?;
    let beta = xtx.solve(&xty)?;
    let fitted = design.matmul(&beta)?;

    let mut resid = Vec::with_capacity(n);
    let mut sse = 0.0;
    let y_mean = y_in.data().iter().sum::<f64>() / (n as f64);
    let mut sst = 0.0;
    for r in 0..n {
        let e = y_in.data()[r] - fitted.get(r, 0);
        resid.push(e);
        sse += e * e;
        sst += (y_in.data()[r] - y_mean).powi(2);
    }
    let df = (n.saturating_sub(k)).max(1) as f64;
    let mse = sse / df;
    let rmse = mse.sqrt();
    let r2 = if sst > 0.0 { 1.0 - sse / sst } else { 1.0 };
    let r2_adj = 1.0 - (1.0 - r2) * ((n as f64) - 1.0) / df;

    // Standard errors, t-stats, and p-values for each coefficient
    let cov_beta = xtx.inv()?;
    let mut se_vec = Vec::with_capacity(k);
    let mut t_vec = Vec::with_capacity(k);
    let mut p_vec = Vec::with_capacity(k);
    for i in 0..k {
        let se = (mse * cov_beta.get(i, i).max(0.0)).sqrt();
        let tstat = if se > 0.0 { beta.get(i, 0) / se } else { 0.0 };
        let pval = 2.0 * (1.0 - tcdf_scalar(tstat.abs(), df));
        se_vec.push(se);
        t_vec.push(tstat);
        p_vec.push(pval);
    }

    let mut coefs = BTreeMap::new();
    coefs.insert("Estimate".to_string(), Value::matrix(beta.clone()));
    coefs.insert("SE".to_string(), Value::matrix(Matrix::col_vec(se_vec)));
    coefs.insert("tStat".to_string(), Value::matrix(Matrix::col_vec(t_vec)));
    coefs.insert("pValue".to_string(), Value::matrix(Matrix::col_vec(p_vec)));

    let mut rsq = BTreeMap::new();
    rsq.insert("Ordinary".to_string(), Value::scalar(r2));
    rsq.insert("Adjusted".to_string(), Value::scalar(r2_adj));

    let mut fields = BTreeMap::new();
    fields.insert("Coefficients".to_string(), Value::Struct(coefs));
    fields.insert("Rsquared".to_string(), Value::Struct(rsq));
    fields.insert("RMSE".to_string(), Value::scalar(rmse));
    fields.insert("SSE".to_string(), Value::scalar(sse));
    fields.insert("DFE".to_string(), Value::scalar(df));
    fields.insert("Fitted".to_string(), Value::matrix(fitted));
    fields.insert("Residuals".to_string(), Value::matrix(Matrix::col_vec(resid)));
    fields.insert("__beta".to_string(), Value::matrix(beta));

    ok1(Value::Object(ObjectInstance {
        class_name: "LinearModel".to_string(),
        fields,
    }))
}

/// `[b, bint, r, rint, stats] = regress(y, X [, alpha])` — OLS without auto-intercept (MATLAB convention).
pub fn fn_regress(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity("regress", "2 or 3 (y, X [, alpha])", args.len()));
    }
    let y = args[0].to_dense_matrix()?;
    let x = args[1].to_dense_matrix()?;
    let alpha = args.get(2).map(|v| v.as_scalar()).transpose()?.unwrap_or(0.05);
    let n = y.len();
    if x.rows() != n {
        return Err(MatroError::dimension("regress: X rows must match length(y)"));
    }
    let p = x.cols();
    let y_col = Matrix::col_vec(y.data().to_vec());
    let xt = x.transpose();
    let xtx = xt.matmul(&x)?;
    let b = xtx.solve(&xt.matmul(&y_col)?)?;
    let yhat = x.matmul(&b)?;
    let y_mean = y.data().iter().sum::<f64>() / (n.max(1) as f64);
    let mut r = Matrix::zeros(n, 1);
    let mut sse = 0.0;
    let mut sst = 0.0;
    for i in 0..n {
        let e = y.data()[i] - yhat.get(i, 0);
        r.set(i, 0, e);
        sse += e * e;
        sst += (y.data()[i] - y_mean).powi(2);
    }
    let df = (n.saturating_sub(p)).max(1) as f64;
    let mse = sse / df;
    let tcrit = tinv_scalar(1.0 - 0.5 * alpha, df);
    let inv_xtx = xtx.inv().unwrap_or_else(|_| Matrix::eye(p, p));

    let mut out = vec![Value::matrix(b.clone())];
    if nargout >= 2 {
        let mut bint = Matrix::zeros(p, 2);
        for i in 0..p {
            let half = tcrit * (mse * inv_xtx.get(i, i).max(0.0)).sqrt();
            bint.set(i, 0, b.get(i, 0) - half);
            bint.set(i, 1, b.get(i, 0) + half);
        }
        out.push(Value::matrix(bint));
    }
    if nargout >= 3 {
        out.push(Value::matrix(r.clone()));
    }
    if nargout >= 4 {
        let mut rint = Matrix::zeros(n, 2);
        for i in 0..n {
            // Leverage h_ii = x_i * (X'X)^-1 * x_i'
            let mut h_ii = 0.0;
            for a in 0..p {
                let xa = x.get(i, a);
                for c in 0..p {
                    h_ii += xa * inv_xtx.get(a, c) * x.get(i, c);
                }
            }
            let half = tcrit * (mse * (1.0 - h_ii).max(0.0)).sqrt();
            let ri = r.get(i, 0);
            rint.set(i, 0, ri - half);
            rint.set(i, 1, ri + half);
        }
        out.push(Value::matrix(rint));
    }
    if nargout >= 5 {
        let r2 = if sst > 0.0 { 1.0 - sse / sst } else { 1.0 };
        let df_reg = (p.saturating_sub(1)).max(1) as f64;
        let f_stat = if mse > 0.0 {
            ((sst - sse).max(0.0) / df_reg) / mse
        } else {
            f64::INFINITY
        };
        let p_val = (1.0 - fcdf_scalar(f_stat, df_reg, df)).clamp(0.0, 1.0);
        out.push(Value::matrix(Matrix::row_vec(vec![r2, f_stat, p_val, mse])));
    }
    Ok(out)
}

/// `[p, tbl, stats] = anova1(X [, group [, displayopt]])` — one-way ANOVA F-test.
pub fn fn_anova1(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity("anova1", "1 to 3 (X [, group [, displayopt]])", args.len()));
    }
    let mut groups: Vec<Vec<f64>> = Vec::new();
    let use_group = args.len() >= 2
        && !matches!(&args[1], Value::Matrix(m) if m.is_empty())
        && crate::value_as_text(&args[1]).is_none();
    if !use_group {
        let x = args[0].to_dense_matrix()?;
        for c in 0..x.cols() {
            groups.push((0..x.rows()).map(|r| x.get(r, c)).filter(|v| !v.is_nan()).collect());
        }
    } else {
        let x = args[0].to_dense_matrix()?;
        let g = args[1].to_dense_matrix()?;
        if x.len() != g.len() {
            return Err(MatroError::dimension("anova1: X and group must have same length"));
        }
        let mut map: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
        for i in 0..x.len() {
            if !x.data()[i].is_nan() {
                map.entry(g.data()[i].round() as i64).or_default().push(x.data()[i]);
            }
        }
        groups = map.into_values().collect();
    }
    let k = groups.len();
    let n_total: usize = groups.iter().map(Vec::len).sum();
    if k < 2 || n_total <= k {
        return Err(MatroError::runtime("anova1: need >= 2 groups and > k total observations"));
    }
    let grand_mean = groups.iter().flatten().sum::<f64>() / (n_total as f64);
    let mut ss_between = 0.0;
    let mut ss_within = 0.0;
    let mut g_means = Vec::with_capacity(k);
    let mut g_counts = Vec::with_capacity(k);
    for g in &groups {
        let gm = g.iter().sum::<f64>() / (g.len() as f64);
        g_means.push(gm);
        g_counts.push(g.len() as f64);
        ss_between += (g.len() as f64) * (gm - grand_mean).powi(2);
        for &v in g {
            ss_within += (v - gm).powi(2);
        }
    }
    let df_b = (k - 1) as f64;
    let df_w = (n_total - k) as f64;
    let ms_b = ss_between / df_b;
    let ms_w = ss_within / df_w;
    let f_stat = if ms_w > 0.0 { ms_b / ms_w } else { f64::INFINITY };
    let p_val = (1.0 - fcdf_scalar(f_stat, df_b, df_w)).clamp(0.0, 1.0);

    let mut st = BTreeMap::new();
    st.insert("F".to_string(), Value::scalar(f_stat));
    st.insert("df".to_string(), Value::scalar(df_w));
    st.insert("df_between".to_string(), Value::scalar(df_b));
    st.insert("df_within".to_string(), Value::scalar(df_w));
    st.insert("SS_between".to_string(), Value::scalar(ss_between));
    st.insert("SS_within".to_string(), Value::scalar(ss_within));
    st.insert("s".to_string(), Value::scalar(ms_w.sqrt()));
    st.insert("means".to_string(), Value::matrix(Matrix::row_vec(g_means)));
    st.insert("n".to_string(), Value::matrix(Matrix::row_vec(g_counts)));

    let mut out = vec![Value::scalar(p_val)];
    if nargout == 2 {
        out.push(Value::Struct(st));
    } else if nargout >= 3 {
        // 4x6 ANOVA table cell array
        let rows = vec![
            vec![
                Value::String("Source".into()),
                Value::String("SS".into()),
                Value::String("df".into()),
                Value::String("MS".into()),
                Value::String("F".into()),
                Value::String("Prob>F".into()),
            ],
            vec![
                Value::String("Columns".into()),
                Value::scalar(ss_between),
                Value::scalar(df_b),
                Value::scalar(ms_b),
                Value::scalar(f_stat),
                Value::scalar(p_val),
            ],
            vec![
                Value::String("Error".into()),
                Value::scalar(ss_within),
                Value::scalar(df_w),
                Value::scalar(ms_w),
                Value::empty(),
                Value::empty(),
            ],
            vec![
                Value::String("Total".into()),
                Value::scalar(ss_between + ss_within),
                Value::scalar(df_b + df_w),
                Value::empty(),
                Value::empty(),
                Value::empty(),
            ],
        ];
        out.push(Value::Cell(matro_types::CellArray::from_rows(rows)?));
        out.push(Value::Struct(st));
    }
    Ok(out)
}

/// `[h, p] = ttest2(x, y)` — two-sample equal-variance Student's t-test.
pub fn fn_ttest2(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("ttest2", "2 (x, y)", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let y = args[1].to_dense_matrix()?;
    let nx = x.len() as f64;
    let ny = y.len() as f64;
    if nx < 2.0 || ny < 2.0 {
        return Err(MatroError::dimension("ttest2: both samples need >= 2 elements"));
    }
    let mx = x.data().iter().sum::<f64>() / nx;
    let my = y.data().iter().sum::<f64>() / ny;
    let vx = x.data().iter().map(|&v| (v - mx).powi(2)).sum::<f64>() / (nx - 1.0);
    let vy = y.data().iter().map(|&v| (v - my).powi(2)).sum::<f64>() / (ny - 1.0);
    let df = nx + ny - 2.0;
    let sp2 = ((nx - 1.0) * vx + (ny - 1.0) * vy) / df;
    let se = (sp2 * (1.0 / nx + 1.0 / ny)).sqrt();
    let tstat = if se > 0.0 { (mx - my) / se } else { 0.0 };
    let p = 2.0 * (1.0 - tcdf_scalar(tstat.abs(), df));
    let h = if p < 0.05 { 1.0 } else { 0.0 };
    if nargout >= 2 {
        Ok(vec![Value::scalar(h), Value::scalar(p)])
    } else {
        ok1(Value::scalar(h))
    }
}

// ---------------------------------------------------------------------------
// Machine Learning (`fitcknn`, `fitcsvm`, `fitcnb`, `predict`, `pdist`, `squareform`, `linkage`, `cluster`)
// ---------------------------------------------------------------------------

pub fn fn_fitcknn(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 {
        return Err(MatroError::arity(
            "fitcknn",
            ">= 2 (X, Y [, 'NumNeighbors', k])",
            args.len(),
        ));
    }
    let x = args[0].to_dense_matrix()?;
    let y = args[1].to_dense_matrix()?;
    if x.rows() != y.len() {
        return Err(MatroError::dimension("fitcknn: rows of X must match length(Y)"));
    }
    let mut k = 1usize;
    let mut i = 2;
    while i + 1 < args.len() {
        if let Ok(key) = args[i].as_str() {
            if key.eq_ignore_ascii_case("NumNeighbors") {
                k = args[i + 1].as_scalar()?.round().max(1.0) as usize;
            }
        }
        i += 2;
    }
    let mut fields = BTreeMap::new();
    fields.insert("X".to_string(), Value::matrix(x));
    fields.insert("Y".to_string(), Value::matrix(Matrix::col_vec(y.data().to_vec())));
    fields.insert("NumNeighbors".to_string(), Value::scalar(k as f64));
    ok1(Value::Object(ObjectInstance {
        class_name: "ClassificationKNN".to_string(),
        fields,
    }))
}

/// Binary linear SVM trained via Pegasos L2-regularized hinge-loss SGD.
pub fn fn_fitcsvm(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 {
        return Err(MatroError::arity("fitcsvm", ">= 2 (X, Y)", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let y_raw = args[1].to_dense_matrix()?;
    let n = x.rows();
    let d = x.cols();
    if n != y_raw.len() || n < 2 {
        return Err(MatroError::dimension("fitcsvm: X rows must match length(Y)"));
    }
    let mut classes: Vec<f64> = Vec::new();
    for &v in y_raw.data() {
        if !classes.iter().any(|&c| (c - v).abs() <= 1e-12) {
            classes.push(v);
        }
    }
    classes.sort_by(|a, b| a.total_cmp(b));
    if classes.len() != 2 {
        return Err(MatroError::runtime("fitcsvm: binary classification requires exactly 2 unique classes"));
    }
    let c_neg = classes[0];
    let c_pos = classes[1];
    let y_bin: Vec<f64> = y_raw
        .data()
        .iter()
        .map(|&v| if (v - c_pos).abs() <= 1e-12 { 1.0 } else { -1.0 })
        .collect();

    // Standardize features for fast, reliable convergence
    let mut mu = vec![0.0; d];
    let mut sd = vec![1.0; d];
    for c in 0..d {
        mu[c] = (0..n).map(|r| x.get(r, c)).sum::<f64>() / (n as f64);
        let var = (0..n).map(|r| (x.get(r, c) - mu[c]).powi(2)).sum::<f64>() / (n as f64);
        sd[c] = if var > 1e-15 { var.sqrt() } else { 1.0 };
    }
    let mut xs = vec![vec![0.0; d]; n];
    for r in 0..n {
        for c in 0..d {
            xs[r][c] = (x.get(r, c) - mu[c]) / sd[c];
        }
    }

    let lambda = 1e-3;
    let mut w = vec![0.0; d];
    let mut b = 0.0f64;
    let epochs = 400usize;
    let mut step = 0usize;
    for _ep in 0..epochs {
        for r in 0..n {
            step += 1;
            let eta = 1.0 / (lambda * (step as f64 + 10.0));
            let dot: f64 = w.iter().zip(xs[r].iter()).map(|(a, b)| a * b).sum::<f64>() + b;
            let margin = y_bin[r] * dot;
            for c in 0..d {
                w[c] *= 1.0 - eta * lambda;
                if margin < 1.0 {
                    w[c] += eta * y_bin[r] * xs[r][c];
                }
            }
            if margin < 1.0 {
                b += eta * y_bin[r];
            }
        }
    }
    // Transform back to original feature coordinates
    let mut beta = vec![0.0; d];
    let mut bias = b;
    for c in 0..d {
        beta[c] = w[c] / sd[c];
        bias -= beta[c] * mu[c];
    }

    let mut fields = BTreeMap::new();
    fields.insert("Beta".to_string(), Value::matrix(Matrix::col_vec(beta)));
    fields.insert("Bias".to_string(), Value::scalar(bias));
    fields.insert("ClassNames".to_string(), Value::matrix(Matrix::col_vec(vec![c_neg, c_pos])));
    ok1(Value::Object(ObjectInstance {
        class_name: "ClassificationSVM".to_string(),
        fields,
    }))
}

/// Gaussian Naive Bayes classifier `mdl = fitcnb(X, Y)`.
pub fn fn_fitcnb(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("fitcnb", "2 (X, Y)", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let y = args[1].to_dense_matrix()?;
    let n = x.rows();
    let d = x.cols();
    if n != y.len() || n < 2 {
        return Err(MatroError::dimension("fitcnb: X rows must match length(Y)"));
    }
    let mut classes: Vec<f64> = Vec::new();
    for &v in y.data() {
        if !classes.iter().any(|&c| (c - v).abs() <= 1e-12) {
            classes.push(v);
        }
    }
    classes.sort_by(|a, b| a.total_cmp(b));
    let nc = classes.len();
    let mut means = Matrix::zeros(nc, d);
    let mut vars = Matrix::zeros(nc, d);
    let mut priors = Vec::with_capacity(nc);
    for (ci, &cls) in classes.iter().enumerate() {
        let idxs: Vec<usize> = (0..n).filter(|&r| (y.data()[r] - cls).abs() <= 1e-12).collect();
        priors.push((idxs.len() as f64) / (n as f64));
        for col in 0..d {
            let mu = idxs.iter().map(|&r| x.get(r, col)).sum::<f64>() / (idxs.len() as f64);
            let v = idxs.iter().map(|&r| (x.get(r, col) - mu).powi(2)).sum::<f64>()
                / (idxs.len() as f64);
            means.set(ci, col, mu);
            vars.set(ci, col, v.max(1e-9));
        }
    }
    let mut fields = BTreeMap::new();
    fields.insert("ClassNames".to_string(), Value::matrix(Matrix::col_vec(classes)));
    fields.insert("Prior".to_string(), Value::matrix(Matrix::row_vec(priors)));
    fields.insert("__means".to_string(), Value::matrix(means));
    fields.insert("__vars".to_string(), Value::matrix(vars));
    ok1(Value::Object(ObjectInstance {
        class_name: "ClassificationNaiveBayes".to_string(),
        fields,
    }))
}

/// Polymorphic `predict(mdl, Xnew)` for `LinearModel`, `ClassificationKNN`, `ClassificationSVM`, `ClassificationNaiveBayes`.
pub fn fn_predict(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("predict", "2 (mdl, Xnew)", args.len()));
    }
    let Value::Object(o) = &args[0] else {
        return Err(MatroError::runtime("predict: first argument must be a fitted model object"));
    };
    let x = args[1].to_dense_matrix()?;
    match o.class_name.as_str() {
        "LinearModel" => {
            let beta = o.fields["__beta"].to_dense_matrix()?;
            let p = beta.len() - 1;
            let x_use = if x.cols() != p && x.rows() == p {
                x.transpose()
            } else {
                x
            };
            let n = x_use.rows();
            let mut out = Vec::with_capacity(n);
            for r in 0..n {
                let mut v = beta.get(0, 0);
                for c in 0..p {
                    v += beta.get(c + 1, 0) * x_use.get(r, c);
                }
                out.push(v);
            }
            ok1(Value::matrix(Matrix::col_vec(out)))
        }
        "ClassificationKNN" => {
            let x_tr = o.fields["X"].to_dense_matrix()?;
            let y_tr = o.fields["Y"].to_dense_matrix()?;
            let k = (o.fields["NumNeighbors"].as_scalar()?.round() as usize).min(x_tr.rows());
            let d = x_tr.cols();
            let mut preds = Vec::with_capacity(x.rows());
            for r in 0..x.rows() {
                let mut dists: Vec<(f64, f64)> = (0..x_tr.rows())
                    .map(|tr| {
                        let dist2: f64 = (0..d)
                            .map(|c| (x.get(r, c) - x_tr.get(tr, c)).powi(2))
                            .sum();
                        (dist2, y_tr.data()[tr])
                    })
                    .collect();
                dists.sort_by(|a, b| a.0.total_cmp(&b.0));
                let mut votes: BTreeMap<i64, (usize, f64)> = BTreeMap::new();
                for &(_, lbl) in dists.iter().take(k) {
                    let key = (lbl * 1e6).round() as i64;
                    let ent = votes.entry(key).or_insert((0, lbl));
                    ent.0 += 1;
                }
                let best = votes
                    .into_values()
                    .max_by_key(|&(cnt, _)| cnt)
                    .map(|(_, lbl)| lbl)
                    .unwrap_or(0.0);
                preds.push(best);
            }
            ok1(Value::matrix(Matrix::col_vec(preds)))
        }
        "ClassificationSVM" => {
            let beta = o.fields["Beta"].to_dense_matrix()?;
            let bias = o.fields["Bias"].as_scalar()?;
            let cnames = o.fields["ClassNames"].to_dense_matrix()?;
            let c_neg = cnames.data()[0];
            let c_pos = cnames.data()[1];
            let d = beta.len();
            let mut preds = Vec::with_capacity(x.rows());
            for r in 0..x.rows() {
                let score: f64 = (0..d).map(|c| x.get(r, c) * beta.data()[c]).sum::<f64>() + bias;
                preds.push(if score >= 0.0 { c_pos } else { c_neg });
            }
            ok1(Value::matrix(Matrix::col_vec(preds)))
        }
        "ClassificationNaiveBayes" => {
            let cnames = o.fields["ClassNames"].to_dense_matrix()?;
            let priors = o.fields["Prior"].to_dense_matrix()?;
            let means = o.fields["__means"].to_dense_matrix()?;
            let vars = o.fields["__vars"].to_dense_matrix()?;
            let nc = cnames.len();
            let d = means.cols();
            let mut preds = Vec::with_capacity(x.rows());
            for r in 0..x.rows() {
                let mut best_lp = f64::NEG_INFINITY;
                let mut best_cls = cnames.data()[0];
                for ci in 0..nc {
                    let mut lp = priors.data()[ci].max(1e-15).ln();
                    for c in 0..d {
                        let mu = means.get(ci, c);
                        let v = vars.get(ci, c);
                        let diff = x.get(r, c) - mu;
                        lp += -0.5 * ((2.0 * std::f64::consts::PI * v).ln() + diff * diff / v);
                    }
                    if lp > best_lp {
                        best_lp = lp;
                        best_cls = cnames.data()[ci];
                    }
                }
                preds.push(best_cls);
            }
            ok1(Value::matrix(Matrix::col_vec(preds)))
        }
        other => Err(MatroError::runtime(format!(
            "predict: unsupported model class '{other}'"
        ))),
    }
}

pub fn pair_distance(a: &[f64], b: &[f64], metric: &str, p_exp: f64) -> f64 {
    let d = a.len();
    match metric {
        "squaredeuclidean" | "seuclidean" => a.iter().zip(b.iter()).map(|(u, v)| (u - v) * (u - v)).sum(),
        "cityblock" | "manhattan" | "l1" => a.iter().zip(b.iter()).map(|(u, v)| (u - v).abs()).sum(),
        "chebychev" | "chebyshev" => a
            .iter()
            .zip(b.iter())
            .map(|(u, v)| (u - v).abs())
            .fold(0.0f64, f64::max),
        "minkowski" => a
            .iter()
            .zip(b.iter())
            .map(|(u, v)| (u - v).abs().powf(p_exp))
            .sum::<f64>()
            .powf(1.0 / p_exp),
        "cosine" => {
            let dot: f64 = a.iter().zip(b.iter()).map(|(u, v)| u * v).sum();
            let na: f64 = a.iter().map(|u| u * u).sum::<f64>().sqrt();
            let nb: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
            if na * nb > 0.0 { (1.0 - dot / (na * nb)).max(0.0) } else { 1.0 }
        }
        "correlation" => {
            let ma = a.iter().sum::<f64>() / d.max(1) as f64;
            let mb = b.iter().sum::<f64>() / d.max(1) as f64;
            let dot: f64 = a.iter().zip(b.iter()).map(|(u, v)| (u - ma) * (v - mb)).sum();
            let na: f64 = a.iter().map(|u| (u - ma).powi(2)).sum::<f64>().sqrt();
            let nb: f64 = b.iter().map(|v| (v - mb).powi(2)).sum::<f64>().sqrt();
            if na * nb > 0.0 { (1.0 - dot / (na * nb)).max(0.0) } else { 1.0 }
        }
        "hamming" => {
            if d == 0 {
                0.0
            } else {
                a.iter().zip(b.iter()).filter(|(u, v)| (**u - **v).abs() > 1e-12).count() as f64 / d as f64
            }
        }
        _ => a.iter().zip(b.iter()).map(|(u, v)| (u - v) * (u - v)).sum::<f64>().sqrt(),
    }
}

/// Pairwise distances `Y = pdist(X [, metric [, p]])` (condensed `1 x n*(n-1)/2` row vector).
pub fn fn_pdist(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity("pdist", "1 to 3 (X [, metric [, p]])", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let metric = args
        .get(1)
        .and_then(crate::value_as_text)
        .unwrap_or_else(|| "euclidean".to_string())
        .to_ascii_lowercase();
    let p_exp = args.get(2).and_then(|v| v.as_scalar().ok()).unwrap_or(2.0);
    let n = x.rows();
    let d = x.cols();
    let rows: Vec<Vec<f64>> = (0..n).map(|r| (0..d).map(|c| x.get(r, c)).collect()).collect();
    let mut out = Vec::with_capacity(n * n.saturating_sub(1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            out.push(pair_distance(&rows[i], &rows[j], &metric, p_exp));
        }
    }
    ok1(Value::matrix(Matrix::row_vec(out)))
}

/// Convert between condensed `pdist` vector and symmetric `n x n` distance matrix.
pub fn fn_squareform(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("squareform", "1 (Y or D)", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    if m.rows() == 1 || m.cols() == 1 {
        let len = m.len();
        let n = ((1.0 + (1.0 + 8.0 * len as f64).sqrt()) / 2.0).round() as usize;
        if n * n.saturating_sub(1) / 2 != len {
            return Err(MatroError::dimension("squareform: invalid condensed distance vector length"));
        }
        let mut out = Matrix::zeros(n, n);
        let mut idx = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                let d = m.data()[idx];
                out.set(i, j, d);
                out.set(j, i, d);
                idx += 1;
            }
        }
        ok1(Value::matrix(out))
    } else {
        let n = m.rows();
        if n != m.cols() {
            return Err(MatroError::dimension("squareform: distance matrix must be square"));
        }
        let mut out = Vec::with_capacity(n * n.saturating_sub(1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                out.push(m.get(i, j));
            }
        }
        ok1(Value::matrix(Matrix::row_vec(out)))
    }
}

/// Agglomerative hierarchical clustering `Z = linkage(Y [, method])` (`'single'`, `'complete'`, `'average'`).
pub fn fn_linkage(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("linkage", "1 or 2 (Y [, method])", args.len()));
    }
    let y = args[0].to_dense_matrix()?;
    let method = args
        .get(1)
        .and_then(|v| v.as_str().ok())
        .unwrap_or("single")
        .to_ascii_lowercase();
    // Accept either condensed pdist vector OR raw observation matrix X
    let (n, mut dist_mat) = if y.rows() == 1 && y.cols() >= 1 {
        let len = y.len();
        let n = ((1.0 + (1.0 + 8.0 * len as f64).sqrt()) / 2.0).round() as usize;
        if n * n.saturating_sub(1) / 2 == len {
            let mut dm = vec![vec![0.0; n]; n];
            let mut idx = 0;
            for i in 0..n {
                for j in (i + 1)..n {
                    dm[i][j] = y.data()[idx];
                    dm[j][i] = y.data()[idx];
                    idx += 1;
                }
            }
            (n, dm)
        } else {
            return Err(MatroError::dimension("linkage: invalid distance vector length"));
        }
    } else {
        let n = y.rows();
        let d = y.cols();
        let mut dm = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let dist = (0..d)
                    .map(|c| (y.get(i, c) - y.get(j, c)).powi(2))
                    .sum::<f64>()
                    .sqrt();
                dm[i][j] = dist;
                dm[j][i] = dist;
            }
        }
        (n, dm)
    };
    if n < 2 {
        return Err(MatroError::dimension("linkage: need at least 2 observations"));
    }

    let mut active: Vec<bool> = vec![true; n];
    let mut cluster_id: Vec<usize> = (1..=n).collect(); // 1-based MATLAB cluster IDs
    let mut cluster_sz: Vec<usize> = vec![1; n];
    let mut z = Matrix::zeros(n - 1, 3);

    for step in 0..(n - 1) {
        let mut best_d = f64::INFINITY;
        let mut bi = 0;
        let mut bj = 1;
        for i in 0..n {
            if !active[i] {
                continue;
            }
            for j in (i + 1)..n {
                if !active[j] {
                    continue;
                }
                if dist_mat[i][j] < best_d {
                    best_d = dist_mat[i][j];
                    bi = i;
                    bj = j;
                }
            }
        }
        let id_a = cluster_id[bi].min(cluster_id[bj]);
        let id_b = cluster_id[bi].max(cluster_id[bj]);
        z.set(step, 0, id_a as f64);
        z.set(step, 1, id_b as f64);
        z.set(step, 2, best_d);

        let sz_i = cluster_sz[bi] as f64;
        let sz_j = cluster_sz[bj] as f64;
        for k in 0..n {
            if !active[k] || k == bi || k == bj {
                continue;
            }
            let d_ik = dist_mat[bi][k];
            let d_jk = dist_mat[bj][k];
            let sz_k = cluster_sz[k] as f64;
            let d_new = match method.as_str() {
                "complete" => d_ik.max(d_jk),
                "average" => (sz_i * d_ik + sz_j * d_jk) / (sz_i + sz_j),
                "ward" => {
                    let denom = sz_i + sz_j + sz_k;
                    let val = ((sz_i + sz_k) * d_ik * d_ik + (sz_j + sz_k) * d_jk * d_jk
                        - sz_k * best_d * best_d)
                        / denom;
                    val.max(0.0).sqrt()
                }
                _ => d_ik.min(d_jk), // single
            };
            dist_mat[bi][k] = d_new;
            dist_mat[k][bi] = d_new;
        }
        active[bj] = false;
        cluster_id[bi] = n + 1 + step;
        cluster_sz[bi] += cluster_sz[bj];
    }
    ok1(Value::matrix(z))
}

/// Cut hierarchical linkage tree `T = cluster(Z, 'maxclust', k)`.
pub fn fn_cluster(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 3 {
        return Err(MatroError::arity("cluster", "3 (Z, 'maxclust', k)", args.len()));
    }
    let z = args[0].to_dense_matrix()?;
    let k = args[2].as_scalar()?.round().max(1.0) as usize;
    let n = z.rows() + 1;
    let merges = n.saturating_sub(k).min(z.rows());

    let mut members: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 1..=n {
        members.insert(i, vec![i - 1]);
    }
    for step in 0..merges {
        let a = z.get(step, 0).round() as usize;
        let b = z.get(step, 1).round() as usize;
        let mut ma = members.remove(&a).unwrap_or_default();
        let mb = members.remove(&b).unwrap_or_default();
        ma.extend(mb);
        members.insert(n + 1 + step, ma);
    }
    let mut labels = vec![0.0; n];
    for (lbl, pts) in members.values().enumerate() {
        for &p in pts {
            labels[p] = (lbl + 1) as f64;
        }
    }
    ok1(Value::matrix(Matrix::col_vec(labels)))
}
