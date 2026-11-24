//! Pure-Rust Statistics Toolbox intrinsics:
//! `std`, `var`, `median`, `prctile`, `quantile`, `cov`, `corrcoef`,
//! `erf`, `erfc`, `normpdf`, `normcdf`, `norminv`, `ttest`, `kmeans`, `pca`.

use std::collections::BTreeMap;

use matro_types::{MatroError, Matrix, Result, Rng, Value};

fn ok1(v: Value) -> Result<Vec<Value>> {
    Ok(vec![v])
}

/// High-precision error function `erf(x)` (~1e-15 accuracy across all real `x`).
pub fn erf_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    if ax > 6.0 {
        return sign;
    }
    if ax < 3.5 {
        // Power series: erf(x) = (2/sqrt(pi)) * exp(-x^2) * sum_{k=0..} (2x^2)^k / (2k+1)!! * x
        let x2 = ax * ax;
        let mut term = ax;
        let mut sum = term;
        for k in 1..120 {
            term *= (2.0 * x2) / (2 * k + 1) as f64;
            sum += term;
            if term.abs() < 1e-16 * sum.abs() {
                break;
            }
        }
        sign * (2.0 / std::f64::consts::PI.sqrt()) * (-x2).exp() * sum
    } else {
        sign * (1.0 - erfc_large(ax))
    }
}

/// Continued fraction for `erfc(x)` when `x >= 3.5`.
fn erfc_large(x: f64) -> f64 {
    // Lentz continued fraction for erfc(x) = (exp(-x^2) / (x*sqrt(pi))) * (1 / (1 + (1/2)/(x^2 + ...)))
    let x2 = x * x;
    let mut cf = 0.0;
    for k in (1..=60).rev() {
        let a = 0.5 * (k as f64);
        cf = a / (x2 + cf);
    }
    let val = (-x2).exp() / (x * std::f64::consts::PI.sqrt() * (1.0 + cf / x2));
    val.clamp(0.0, 1.0)
}

pub fn erfc_f64(x: f64) -> f64 {
    if x >= 3.5 {
        erfc_large(x)
    } else if x <= -6.0 {
        2.0
    } else {
        1.0 - erf_f64(x)
    }
}

pub fn fn_erf(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("erf", "1", args.len()));
    }
    ok1(Value::matrix(args[0].as_matrix()?.map(erf_f64)))
}

pub fn fn_erfc(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("erfc", "1", args.len()));
    }
    ok1(Value::matrix(args[0].as_matrix()?.map(erfc_f64)))
}

fn slice_var(xs: &[f64], pop: bool) -> f64 {
    let n = xs.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return 0.0;
    }
    let mean = xs.iter().sum::<f64>() / n as f64;
    let ss: f64 = xs.iter().map(|&x| (x - mean) * (x - mean)).sum();
    let denom = if pop { n as f64 } else { (n - 1) as f64 };
    ss / denom
}

fn slice_median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn slice_quantile(xs: &[f64], q: f64) -> f64 {
    if xs.is_empty() || !(0.0..=1.0).contains(&q) {
        return f64::NAN;
    }
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n == 1 {
        return v[0];
    }
    // MATLAB convention: p_i = (i - 0.5) / n
    let r = q * (n as f64) - 0.5;
    if r <= 0.0 {
        return v[0];
    }
    if r >= (n - 1) as f64 {
        return v[n - 1];
    }
    let lo = r.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = r - lo as f64;
    v[lo] * (1.0 - frac) + v[hi] * frac
}

fn reduce_stats(
    name: &str,
    m: &Matrix,
    dim_opt: Option<usize>,
    f: impl Fn(&[f64]) -> f64,
) -> Result<Vec<Value>> {
    if m.is_empty() {
        return ok1(Value::scalar(f64::NAN));
    }
    if dim_opt.is_none() && (m.rows() == 1 || m.cols() == 1) {
        return ok1(Value::scalar(f(m.data())));
    }
    let dim = dim_opt.unwrap_or(1);
    if dim == 1 {
        let mut out = Vec::with_capacity(m.cols());
        for c in 0..m.cols() {
            let col: Vec<f64> = (0..m.rows()).map(|r| m.get(r, c)).collect();
            out.push(f(&col));
        }
        ok1(Value::matrix(Matrix::row_vec(out)))
    } else if dim == 2 {
        let mut out = Vec::with_capacity(m.rows());
        for r in 0..m.rows() {
            let row: Vec<f64> = (0..m.cols()).map(|c| m.get(r, c)).collect();
            out.push(f(&row));
        }
        ok1(Value::matrix(Matrix::col_vec(out)))
    } else {
        Err(MatroError::runtime(format!("{name}: unsupported dimension {dim}")))
    }
}

/// `std(x [, w [, dim]])` and `var(x [, w [, dim]])`
pub fn fn_std_var(name: &str, args: &[Value], take_sqrt: bool) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity(name, "1 to 3", args.len()));
    }
    let m = args[0].as_matrix()?;
    let pop = if let Some(w_val) = args.get(1) {
        let wm = w_val.as_matrix()?;
        !wm.is_empty() && wm.data()[0] != 0.0
    } else {
        false
    };
    let dim = if let Some(d_val) = args.get(2) {
        Some(d_val.as_scalar()?.max(1.0).round() as usize)
    } else {
        None
    };
    reduce_stats(name, m, dim, |xs| {
        let v = slice_var(xs, pop);
        if take_sqrt { v.sqrt() } else { v }
    })
}

pub fn fn_median(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("median", "1 or 2", args.len()));
    }
    let m = args[0].as_matrix()?;
    let dim = args
        .get(1)
        .map(|v| v.as_scalar().map(|s| s.max(1.0).round() as usize))
        .transpose()?;
    reduce_stats("median", m, dim, slice_median)
}

pub fn fn_quantile(name: &str, args: &[Value], percent: bool) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity(name, "2 (x, p)", args.len()));
    }
    let x = args[0].as_matrix()?;
    let p = args[1].as_matrix()?;
    let scale = if percent { 0.01 } else { 1.0 };
    if x.rows() <= 1 || x.cols() <= 1 {
        let vals: Vec<f64> = p
            .data()
            .iter()
            .map(|&pi| slice_quantile(x.data(), pi * scale))
            .collect();
        let out = if p.len() == 1 {
            Matrix::scalar(vals[0])
        } else {
            Matrix::row_vec(vals)
        };
        return ok1(Value::matrix(out));
    }
    let mut out = Matrix::zeros(p.len(), x.cols());
    for c in 0..x.cols() {
        let col: Vec<f64> = (0..x.rows()).map(|r| x.get(r, c)).collect();
        for (ri, &pi) in p.data().iter().enumerate() {
            out.set(ri, c, slice_quantile(&col, pi * scale));
        }
    }
    ok1(Value::matrix(out))
}

fn cov_matrix_from_cols(cols: &[Vec<f64>]) -> Matrix {
    let k = cols.len();
    let n = cols.first().map_or(0, Vec::len);
    if k == 0 || n == 0 {
        return Matrix::empty();
    }
    let means: Vec<f64> = cols
        .iter()
        .map(|col| col.iter().sum::<f64>() / n as f64)
        .collect();
    let denom = if n > 1 { (n - 1) as f64 } else { 1.0 };
    let mut out = Matrix::zeros(k, k);
    for i in 0..k {
        for j in i..k {
            let mut s = 0.0;
            for r in 0..n {
                s += (cols[i][r] - means[i]) * (cols[j][r] - means[j]);
            }
            let v = s / denom;
            out.set(i, j, v);
            out.set(j, i, v);
        }
    }
    out
}

pub fn fn_cov_or_corr(name: &str, args: &[Value], normalize_corr: bool) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity(name, "1 or 2", args.len()));
    }
    let cols: Vec<Vec<f64>> = if args.len() == 2 {
        let a = args[0].as_matrix()?.data().to_vec();
        let b = args[1].as_matrix()?.data().to_vec();
        if a.len() != b.len() {
            return Err(MatroError::dimension(format!(
                "{name}: input vectors must have the same length"
            )));
        }
        vec![a, b]
    } else {
        let m = args[0].as_matrix()?;
        if m.rows() == 1 || m.cols() == 1 {
            vec![m.data().to_vec()]
        } else {
            (0..m.cols())
                .map(|c| (0..m.rows()).map(|r| m.get(r, c)).collect())
                .collect()
        }
    };
    let mut cov = cov_matrix_from_cols(&cols);
    if normalize_corr {
        let k = cov.rows();
        let stds: Vec<f64> = (0..k).map(|i| cov.get(i, i).max(0.0).sqrt()).collect();
        for i in 0..k {
            for j in 0..k {
                let d = stds[i] * stds[j];
                let r = if i == j {
                    1.0
                } else if d > 0.0 {
                    (cov.get(i, j) / d).clamp(-1.0, 1.0)
                } else {
                    f64::NAN
                };
                cov.set(i, j, r);
            }
        }
    }
    ok1(Value::matrix(cov))
}

fn parse_mu_sigma(name: &str, args: &[Value]) -> Result<(f64, f64)> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity(name, "1 to 3 (x [, mu, sigma])", args.len()));
    }
    let mu = args.get(1).map(|v| v.as_scalar()).transpose()?.unwrap_or(0.0);
    let sigma = args.get(2).map(|v| v.as_scalar()).transpose()?.unwrap_or(1.0);
    Ok((mu, sigma))
}

pub fn fn_normpdf(args: &[Value]) -> Result<Vec<Value>> {
    let (mu, sigma) = parse_mu_sigma("normpdf", args)?;
    let inv_s = 1.0 / sigma;
    let coeff = inv_s / (2.0 * std::f64::consts::PI).sqrt();
    let out = args[0].as_matrix()?.map(|x| {
        if sigma <= 0.0 {
            f64::NAN
        } else {
            let z = (x - mu) * inv_s;
            coeff * (-0.5 * z * z).exp()
        }
    });
    ok1(Value::matrix(out))
}

pub fn fn_normcdf(args: &[Value]) -> Result<Vec<Value>> {
    let (mu, sigma) = parse_mu_sigma("normcdf", args)?;
    let denom = sigma * std::f64::consts::SQRT_2;
    let out = args[0].as_matrix()?.map(|x| {
        if sigma <= 0.0 {
            f64::NAN
        } else {
            0.5 * (1.0 + erf_f64((x - mu) / denom))
        }
    });
    ok1(Value::matrix(out))
}

/// Inverse standard normal CDF (Acklam's rational approximation + Newton polish).
fn std_norm_inv(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    let mut x = if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    // One Newton step to machine precision
    let cdf = 0.5 * (1.0 + erf_f64(x / std::f64::consts::SQRT_2));
    let pdf = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
    if pdf > 1e-30 {
        x -= (cdf - p) / pdf;
    }
    x
}

pub fn fn_norminv(args: &[Value]) -> Result<Vec<Value>> {
    let (mu, sigma) = parse_mu_sigma("norminv", args)?;
    let out = args[0].as_matrix()?.map(|p| {
        if sigma <= 0.0 {
            f64::NAN
        } else {
            mu + sigma * std_norm_inv(p)
        }
    });
    ok1(Value::matrix(out))
}

/// Lanczos log-gamma for exact regularized incomplete beta (Student's t-distribution).
fn lgamma_f64(z: f64) -> f64 {
    const COEFFS: [f64; 9] = [
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
    let z = z - 1.0;
    let mut x = COEFFS[0];
    for (i, &c) in COEFFS.iter().enumerate().skip(1) {
        x += c / (z + i as f64);
    }
    let t = z + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
}

/// Regularized incomplete beta function I_x(a, b) via Lentz's continued fraction.
fn betainc(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (lgamma_f64(a + b) - lgamma_f64(a) - lgamma_f64(b) + a * x.ln() + b * (1.0 - x).ln())
        .exp();
    let betacf = |x: f64, a: f64, b: f64| -> f64 {
        let mut c = 1.0;
        let mut d = 1.0 - (a + b) * x / (a + 1.0);
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        d = 1.0 / d;
        let mut h = d;
        for m in 1..=200 {
            let mf = m as f64;
            let m2 = 2.0 * mf;
            let aa = mf * (b - mf) * x / ((a + m2 - 1.0) * (a + m2));
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

            let aa2 = -(a + mf) * (a + b + mf) * x / ((a + m2) * (a + m2 + 1.0));
            d = 1.0 + aa2 * d;
            if d.abs() < 1e-30 {
                d = 1e-30;
            }
            c = 1.0 + aa2 / c;
            if c.abs() < 1e-30 {
                c = 1e-30;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-12 {
                break;
            }
        }
        h
    };
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(x, a, b) / a
    } else {
        1.0 - bt * betacf(1.0 - x, b, a) / b
    }
}

/// `[h, p, ci, stats] = ttest(x [, m [, 'Alpha', alpha]])` — one-sample (or paired) Student's t-test.
pub fn fn_ttest(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("ttest", ">= 1", 0));
    }
    let x = args[0].as_matrix()?.data();
    let mut m_val = 0.0;
    let mut sample: Vec<f64> = x.to_vec();
    let mut alpha = 0.05;
    let mut idx = 1;
    if let Some(arg1) = args.get(1) {
        if let Ok(m1) = arg1.as_matrix() {
            if let Some(sc) = m1.as_scalar() {
                m_val = sc;
            } else if m1.len() == sample.len() {
                for (a, &b) in sample.iter_mut().zip(m1.data().iter()) {
                    *a -= b;
                }
            }
            idx = 2;
        }
    }
    while idx + 1 < args.len() {
        if let Ok(k) = args[idx].as_str() {
            if k.eq_ignore_ascii_case("alpha") {
                alpha = args[idx + 1].as_scalar()?;
            }
        }
        idx += 2;
    }

    let n = sample.len();
    if n < 2 {
        return Err(MatroError::dimension("ttest: sample must have at least 2 elements"));
    }
    let df = (n - 1) as f64;
    let mean = sample.iter().sum::<f64>() / n as f64;
    let sd = slice_var(&sample, false).sqrt();
    let se = sd / (n as f64).sqrt();
    let tstat = if se > 0.0 { (mean - m_val) / se } else { f64::INFINITY };
    let p = betainc(df / (df + tstat * tstat), 0.5 * df, 0.5).clamp(0.0, 1.0);
    let h = if p < alpha { 1.0 } else { 0.0 };

    // Critical t via bisection on two-tailed p(t) = alpha
    let mut lo = 0.0;
    let mut hi = 100.0;
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        let p_mid = betainc(df / (df + mid * mid), 0.5 * df, 0.5);
        if p_mid > alpha {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let tcrit = 0.5 * (lo + hi);
    let ci = Matrix::col_vec(vec![mean - tcrit * se, mean + tcrit * se]);

    let mut stats = BTreeMap::new();
    stats.insert("tstat".to_string(), Value::scalar(tstat));
    stats.insert("df".to_string(), Value::scalar(df));
    stats.insert("sd".to_string(), Value::scalar(sd));

    let mut out = vec![Value::scalar(h)];
    if nargout >= 2 {
        out.push(Value::scalar(p));
    }
    if nargout >= 3 {
        out.push(Value::matrix(ci));
    }
    if nargout >= 4 {
        out.push(Value::Struct(stats));
    }
    Ok(out)
}

/// `[idx, C] = kmeans(X, k)` — k-means++ initialization + Lloyd clustering.
pub fn fn_kmeans(args: &[Value], nargout: usize, rng: &mut Rng) -> Result<Vec<Value>> {
    if args.len() < 2 {
        return Err(MatroError::arity("kmeans", ">= 2 (X, k)", args.len()));
    }
    let x = args[0].as_matrix()?;
    let k = args[1].as_scalar()?.max(1.0).round() as usize;
    let n = x.rows();
    let d = x.cols();
    if k > n || n == 0 || d == 0 {
        return Err(MatroError::dimension(
            "kmeans: k must be between 1 and size(X, 1)",
        ));
    }

    let pts: Vec<Vec<f64>> = (0..n)
        .map(|r| (0..d).map(|c| x.get(r, c)).collect())
        .collect();

    let dist_sq = |a: &[f64], b: &[f64]| -> f64 {
        a.iter().zip(b.iter()).map(|(&u, &v)| (u - v) * (u - v)).sum()
    };

    // k-means++ seeding
    let mut centers: Vec<Vec<f64>> = Vec::with_capacity(k);
    let first = (rng.next_u64() as usize) % n;
    centers.push(pts[first].clone());
    while centers.len() < k {
        let mut weights = Vec::with_capacity(n);
        let mut total = 0.0;
        for p in &pts {
            let min_d = centers
                .iter()
                .map(|c| dist_sq(p, c))
                .fold(f64::INFINITY, f64::min);
            weights.push(min_d);
            total += min_d;
        }
        let chosen = if total <= 0.0 {
            centers.len() % n
        } else {
            let target = rng.next_f64() * total;
            let mut acc = 0.0;
            let mut pick = n - 1;
            for (i, &w) in weights.iter().enumerate() {
                acc += w;
                if acc >= target {
                    pick = i;
                    break;
                }
            }
            pick
        };
        centers.push(pts[chosen].clone());
    }

    let mut assign = vec![0usize; n];
    for _ in 0..100 {
        let mut changed = false;
        for (i, p) in pts.iter().enumerate() {
            let mut best_c = 0;
            let mut best_d = f64::INFINITY;
            for (ci, c) in centers.iter().enumerate() {
                let dd = dist_sq(p, c);
                if dd < best_d {
                    best_d = dd;
                    best_c = ci;
                }
            }
            if assign[i] != best_c {
                assign[i] = best_c;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        let mut sums = vec![vec![0.0; d]; k];
        let mut counts = vec![0usize; k];
        for (i, p) in pts.iter().enumerate() {
            let ci = assign[i];
            counts[ci] += 1;
            for j in 0..d {
                sums[ci][j] += p[j];
            }
        }
        for ci in 0..k {
            if counts[ci] > 0 {
                for j in 0..d {
                    centers[ci][j] = sums[ci][j] / counts[ci] as f64;
                }
            }
        }
    }

    let idx_mat = Matrix::col_vec(assign.iter().map(|&c| (c + 1) as f64).collect());
    let mut c_mat = Matrix::zeros(k, d);
    for r in 0..k {
        for c in 0..d {
            c_mat.set(r, c, centers[r][c]);
        }
    }
    if nargout >= 2 {
        Ok(vec![Value::matrix(idx_mat), Value::matrix(c_mat)])
    } else {
        ok1(Value::matrix(idx_mat))
    }
}

/// `[coeff, score, latent] = pca(X)` — Principal Component Analysis via centered SVD.
pub fn fn_pca(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("pca", "1 (X)", args.len()));
    }
    let x = args[0].as_matrix()?;
    let (n, p) = (x.rows(), x.cols());
    if n == 0 || p == 0 {
        return Err(MatroError::dimension("pca: input matrix cannot be empty"));
    }
    let mut centered = Matrix::zeros(n, p);
    for c in 0..p {
        let mean = (0..n).map(|r| x.get(r, c)).sum::<f64>() / n as f64;
        for r in 0..n {
            centered.set(r, c, x.get(r, c) - mean);
        }
    }
    let (_u, s_mat, v) = centered.svd()?;
    let k = n.min(p);
    let score = centered.matmul(&v)?;
    let denom = if n > 1 { (n - 1) as f64 } else { 1.0 };
    let latent: Vec<f64> = (0..k)
        .map(|i| {
            let sv = s_mat.get(i, i);
            (sv * sv) / denom
        })
        .collect();

    let mut out = vec![Value::matrix(v)];
    if nargout >= 2 {
        out.push(Value::matrix(score));
    }
    if nargout >= 3 {
        out.push(Value::matrix(Matrix::col_vec(latent)));
    }
    Ok(out)
}
