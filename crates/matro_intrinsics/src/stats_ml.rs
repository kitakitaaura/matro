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
            lgamma_f64(x).exp()
        }
    });
    ok1(Value::matrix(out))
}

pub fn fn_gammainc(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("gammainc", "2 (x, a)", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let a = args[1].to_dense_matrix()?;
    ok1(Value::matrix(x.elementwise(&a, gammainc_lower)?))
}

pub fn fn_betainc(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 3 {
        return Err(MatroError::arity("betainc", "3 (x, a, b)", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let a = args[1].as_scalar()?;
    let b = args[2].as_scalar()?;
    ok1(Value::matrix(x.map(|v| betainc_reg(v, a, b))))
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

pub fn fn_t_dist(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity(name, "2 (x, nu)", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let nu = args[1].as_scalar()?;
    let out = match name {
        "tpdf" => {
            let log_c = lgamma_f64(0.5 * (nu + 1.0))
                - lgamma_f64(0.5 * nu)
                - 0.5 * (nu * std::f64::consts::PI).ln();
            x.map(|t| (log_c - 0.5 * (nu + 1.0) * (1.0 + t * t / nu).ln()).exp())
        }
        "tcdf" => x.map(|t| tcdf_scalar(t, nu)),
        "tinv" => x.map(|p| tinv_scalar(p, nu)),
        _ => unreachable!(),
    };
    ok1(Value::matrix(out))
}

pub fn fn_chi2_dist(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity(name, "2 (x, k)", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let k = args[1].as_scalar()?;
    let out = match name {
        "chi2pdf" => {
            let half_k = 0.5 * k;
            let log_c = -half_k * 2.0f64.ln() - lgamma_f64(half_k);
            x.map(|v| {
                if v <= 0.0 {
                    0.0
                } else {
                    (log_c + (half_k - 1.0) * v.ln() - 0.5 * v).exp()
                }
            })
        }
        "chi2cdf" => x.map(|v| if v <= 0.0 { 0.0 } else { gammainc_lower(0.5 * v, 0.5 * k) }),
        "chi2inv" => x.map(|p| {
            if p <= 0.0 {
                return 0.0;
            }
            if p >= 1.0 {
                return f64::INFINITY;
            }
            let mut lo = 0.0;
            let mut hi = k.max(1.0);
            while gammainc_lower(0.5 * hi, 0.5 * k) < p && hi < 1e8 {
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
        }),
        _ => unreachable!(),
    };
    ok1(Value::matrix(out))
}

pub fn fn_f_dist(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 3 {
        return Err(MatroError::arity(name, "3 (x, d1, d2)", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let d1 = args[1].as_scalar()?;
    let d2 = args[2].as_scalar()?;
    let out = match name {
        "fpdf" => {
            let log_c = lgamma_f64(0.5 * (d1 + d2))
                - lgamma_f64(0.5 * d1)
                - lgamma_f64(0.5 * d2)
                + 0.5 * d1 * (d1 / d2).ln();
            x.map(|v| {
                if v <= 0.0 {
                    0.0
                } else {
                    (log_c + (0.5 * d1 - 1.0) * v.ln()
                        - 0.5 * (d1 + d2) * (1.0 + d1 * v / d2).ln())
                    .exp()
                }
            })
        }
        "fcdf" => x.map(|v| fcdf_scalar(v, d1, d2)),
        _ => unreachable!(),
    };
    ok1(Value::matrix(out))
}

pub fn fn_exp_dist(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity(name, "1 or 2 (x [, mu])", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let mu = args.get(1).map(|v| v.as_scalar()).transpose()?.unwrap_or(1.0);
    let out = match name {
        "exppdf" => x.map(|v| if v < 0.0 { 0.0 } else { (-v / mu).exp() / mu }),
        "expcdf" => x.map(|v| if v < 0.0 { 0.0 } else { 1.0 - (-v / mu).exp() }),
        "expinv" => x.map(|p| {
            if p < 0.0 || p > 1.0 {
                f64::NAN
            } else {
                -mu * (1.0 - p).ln()
            }
        }),
        _ => unreachable!(),
    };
    ok1(Value::matrix(out))
}

pub fn fn_bino_dist(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 3 {
        return Err(MatroError::arity(name, "3 (k, n, p)", args.len()));
    }
    let kv = args[0].to_dense_matrix()?;
    let n = args[1].as_scalar()?.round().max(0.0);
    let p = args[2].as_scalar()?;
    let pmf = |k: f64| -> f64 {
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
    };
    let out = match name {
        "binopdf" => kv.map(pmf),
        "binocdf" => kv.map(|k| {
            if k < 0.0 {
                0.0
            } else if k >= n {
                1.0
            } else {
                let kf = k.floor() as usize;
                (0..=kf).map(|i| pmf(i as f64)).sum::<f64>().min(1.0)
            }
        }),
        _ => unreachable!(),
    };
    ok1(Value::matrix(out))
}

pub fn fn_poiss_dist(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity(name, "2 (k, lambda)", args.len()));
    }
    let kv = args[0].to_dense_matrix()?;
    let lam = args[1].as_scalar()?;
    let pmf = |k: f64| -> f64 {
        let ki = k.round();
        if (k - ki).abs() > 1e-12 || ki < 0.0 {
            return 0.0;
        }
        (-lam + ki * lam.ln() - lgamma_f64(ki + 1.0)).exp()
    };
    let out = match name {
        "poisspdf" => kv.map(pmf),
        "poisscdf" => kv.map(|k| {
            if k < 0.0 {
                0.0
            } else {
                let kf = k.floor() as usize;
                (0..=kf).map(|i| pmf(i as f64)).sum::<f64>().min(1.0)
            }
        }),
        _ => unreachable!(),
    };
    ok1(Value::matrix(out))
}

// ---------------------------------------------------------------------------
// Descriptive Stats (`zscore`, `skewness`, `kurtosis`, `iqr`, `mad`)
// ---------------------------------------------------------------------------

pub fn fn_zscore(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("zscore", "1 (X)", 0));
    }
    let x = args[0].to_dense_matrix()?;
    if x.rows() == 1 || x.cols() == 1 {
        let n = x.len() as f64;
        let mu = x.data().iter().sum::<f64>() / n;
        let sd = (x.data().iter().map(|&v| (v - mu).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)).sqrt();
        let z = x.map(|v| if sd > 0.0 { (v - mu) / sd } else { 0.0 });
        let mut out = vec![Value::matrix(z)];
        if nargout >= 2 {
            out.push(Value::scalar(mu));
        }
        if nargout >= 3 {
            out.push(Value::scalar(sd));
        }
        return Ok(out);
    }
    let (r, c) = (x.rows(), x.cols());
    let mut z = Matrix::zeros(r, c);
    let mut mus = Vec::with_capacity(c);
    let mut sds = Vec::with_capacity(c);
    for col in 0..c {
        let mu = (0..r).map(|row| x.get(row, col)).sum::<f64>() / (r as f64);
        let sd = ((0..r).map(|row| (x.get(row, col) - mu).powi(2)).sum::<f64>()
            / ((r as f64) - 1.0).max(1.0))
        .sqrt();
        mus.push(mu);
        sds.push(sd);
        for row in 0..r {
            z.set(row, col, if sd > 0.0 { (x.get(row, col) - mu) / sd } else { 0.0 });
        }
    }
    let mut out = vec![Value::matrix(z)];
    if nargout >= 2 {
        out.push(Value::matrix(Matrix::row_vec(mus)));
    }
    if nargout >= 3 {
        out.push(Value::matrix(Matrix::row_vec(sds)));
    }
    Ok(out)
}

pub fn fn_moment_stat(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity(name, "1", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let data = x.data();
    let n = data.len() as f64;
    if data.is_empty() {
        return ok1(Value::scalar(f64::NAN));
    }
    let mu = data.iter().sum::<f64>() / n;
    let m2 = data.iter().map(|&v| (v - mu).powi(2)).sum::<f64>() / n;
    let res = match name {
        "skewness" => {
            let m3 = data.iter().map(|&v| (v - mu).powi(3)).sum::<f64>() / n;
            m3 / m2.powf(1.5)
        }
        "kurtosis" => {
            let m4 = data.iter().map(|&v| (v - mu).powi(4)).sum::<f64>() / n;
            m4 / (m2 * m2)
        }
        "mad" => data.iter().map(|&v| (v - mu).abs()).sum::<f64>() / n,
        "iqr" => {
            let mut sorted = data.to_vec();
            sorted.sort_by(|a, b| a.total_cmp(b));
            let q = |p: f64| -> f64 {
                let pos = p * ((sorted.len() - 1) as f64);
                let lo = pos.floor() as usize;
                let hi = (lo + 1).min(sorted.len() - 1);
                sorted[lo] + (pos - lo as f64) * (sorted[hi] - sorted[lo])
            };
            q(0.75) - q(0.25)
        }
        _ => unreachable!(),
    };
    ok1(Value::scalar(res))
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

/// `[b, bint, r] = regress(y, X)` — OLS without auto-intercept (MATLAB convention).
pub fn fn_regress(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("regress", "2 (y, X)", args.len()));
    }
    let y = args[0].to_dense_matrix()?;
    let x = args[1].to_dense_matrix()?;
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
    let mut r = Matrix::zeros(n, 1);
    let mut sse = 0.0;
    for i in 0..n {
        let e = y.data()[i] - yhat.get(i, 0);
        r.set(i, 0, e);
        sse += e * e;
    }
    let mut out = vec![Value::matrix(b.clone())];
    if nargout >= 2 {
        let df = (n.saturating_sub(p)).max(1) as f64;
        let mse = sse / df;
        let tcrit = tinv_scalar(0.975, df);
        let inv_xtx = xtx.inv()?;
        let mut bint = Matrix::zeros(p, 2);
        for i in 0..p {
            let half = tcrit * (mse * inv_xtx.get(i, i).max(0.0)).sqrt();
            bint.set(i, 0, b.get(i, 0) - half);
            bint.set(i, 1, b.get(i, 0) + half);
        }
        out.push(Value::matrix(bint));
    }
    if nargout >= 3 {
        out.push(Value::matrix(r));
    }
    Ok(out)
}

/// `[p, stats] = anova1(X [, group])` — one-way ANOVA F-test.
pub fn fn_anova1(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("anova1", "1 or 2 (X [, group])", args.len()));
    }
    let mut groups: Vec<Vec<f64>> = Vec::new();
    if args.len() == 1 {
        let x = args[0].to_dense_matrix()?;
        for c in 0..x.cols() {
            groups.push((0..x.rows()).map(|r| x.get(r, c)).collect());
        }
    } else {
        let x = args[0].to_dense_matrix()?;
        let g = args[1].to_dense_matrix()?;
        if x.len() != g.len() {
            return Err(MatroError::dimension("anova1: X and group must have same length"));
        }
        let mut map: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
        for i in 0..x.len() {
            map.entry(g.data()[i].round() as i64).or_default().push(x.data()[i]);
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
    for g in &groups {
        let gm = g.iter().sum::<f64>() / (g.len() as f64);
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

    let mut out = vec![Value::scalar(p_val)];
    if nargout >= 2 {
        let mut st = BTreeMap::new();
        st.insert("F".to_string(), Value::scalar(f_stat));
        st.insert("df_between".to_string(), Value::scalar(df_b));
        st.insert("df_within".to_string(), Value::scalar(df_w));
        st.insert("SS_between".to_string(), Value::scalar(ss_between));
        st.insert("SS_within".to_string(), Value::scalar(ss_within));
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

/// Pairwise Euclidean distances `Y = pdist(X)` (condensed `1 x n*(n-1)/2` row vector).
pub fn fn_pdist(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("pdist", "1 (X)", 0));
    }
    let x = args[0].to_dense_matrix()?;
    let n = x.rows();
    let d = x.cols();
    let mut out = Vec::with_capacity(n * n.saturating_sub(1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = (0..d)
                .map(|c| (x.get(i, c) - x.get(j, c)).powi(2))
                .sum::<f64>()
                .sqrt();
            out.push(dist);
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
            let d_new = match method.as_str() {
                "complete" => d_ik.max(d_jk),
                "average" => (sz_i * d_ik + sz_j * d_jk) / (sz_i + sz_j),
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
