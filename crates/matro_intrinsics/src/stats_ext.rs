//! Extended Statistics & Machine Learning Toolbox intrinsics:
//! - Multivariate normal: `mvnpdf`, `mvnrnd`
//! - Distributions: `lognpdf`/`logncdf`/`logninv`, `gampdf`/`gamcdf`/`gaminv`,
//!   `betapdf`/`betacdf`/`betainv`, `unifpdf`/`unifcdf`/`unifinv`
//! - Random generators: `normrnd`, `unifrnd`, `gamrnd`, `betarnd`, `exprnd`,
//!   `poissrnd`, `binornd`, `trnd`, `chi2rnd`, `frnd`, `lognrnd`, `randsample`, `datasample`
//! - Correlation & distances: `corr`, `pdist2`, `knnsearch`
//! - Nonparametric tests: `kstest`, `kstest2`, `ranksum`, `signrank`
//! - Descriptive & NaN-safe stats: `geomean`, `harmmean`, `trimmean`, `mode`,
//!   `nansum`, `nanmean`, `nanstd`, `nanvar`, `nanmedian`

use matro_types::{MatroError, Matrix, Result, Rng, Value};

use crate::stats::{erf_f64, reduce_nd_real, std_norm_inv};
use crate::stats_ml::{betainc_reg, gammainc_lower, lgamma_f64, pair_distance, tcdf_scalar};

fn ok1(v: Value) -> Result<Vec<Value>> {
    Ok(vec![v])
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

fn parse_rnd_dims(param_dims: &[Vec<usize>], size_args: &[Value]) -> Result<Vec<usize>> {
    if !size_args.is_empty() {
        if size_args.len() == 1 {
            let m = size_args[0].to_dense_matrix()?;
            if m.len() == 1 {
                let n = m.data()[0].max(0.0).round() as usize;
                return Ok(vec![n, n]);
            }
            return Ok(m.data().iter().map(|&v| v.max(0.0).round() as usize).collect());
        }
        let mut dims = Vec::with_capacity(size_args.len());
        for a in size_args {
            dims.push(a.as_scalar()?.max(0.0).round() as usize);
        }
        return Ok(dims);
    }
    for d in param_dims {
        if d.iter().product::<usize>() > 1 {
            return Ok(d.clone());
        }
    }
    Ok(vec![1, 1])
}

fn sample_gamma(a: f64, rng: &mut Rng) -> f64 {
    if a <= 0.0 || a.is_nan() {
        return f64::NAN;
    }
    if a < 1.0 {
        let u = rng.next_f64().max(1e-300);
        return sample_gamma(a + 1.0, rng) * u.powf(1.0 / a);
    }
    // Marsaglia & Tsang (2000)
    let d = a - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = rng.next_normal();
        let v_root = 1.0 + c * x;
        if v_root <= 0.0 {
            continue;
        }
        let v = v_root * v_root * v_root;
        let u = rng.next_f64().max(1e-300);
        let x2 = x * x;
        if u < 1.0 - 0.0331 * x2 * x2 {
            return d * v;
        }
        if u.ln() < 0.5 * x2 + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

fn sample_poisson(lam: f64, rng: &mut Rng) -> f64 {
    if lam < 0.0 || lam.is_nan() {
        return f64::NAN;
    }
    if lam == 0.0 {
        return 0.0;
    }
    if lam < 30.0 {
        // Knuth algorithm
        let l = (-lam).exp();
        let mut k = 0u64;
        let mut p = 1.0;
        loop {
            k += 1;
            p *= rng.next_f64();
            if p <= l {
                return (k - 1) as f64;
            }
        }
    }
    // Ahrens-Dieter / Gaussian-corrected rejection (PTRS) for large lambda
    let smu = lam.sqrt();
    let b = 0.931 + 2.53 * smu;
    let a = -0.059 + 0.02483 * b;
    let inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
    let vr = 0.9277 - 3.6224 / (b - 2.0);
    loop {
        let u = rng.next_f64() - 0.5;
        let v = rng.next_f64();
        let us = 0.5 - u.abs();
        let k = ((2.0 * a / us + b) * u + lam + 0.43).floor();
        if k < 0.0 {
            continue;
        }
        if us >= 0.07 && v <= vr {
            return k;
        }
        if us < 0.013 && v > us {
            continue;
        }
        if (v * inv_alpha / (a / (us * us) + b)).ln()
            <= -lam + k * lam.ln() - lgamma_f64(k + 1.0)
        {
            return k;
        }
    }
}

// ---------------------------------------------------------------------------
// Multivariate Normal (`mvnpdf`, `mvnrnd`)
// ---------------------------------------------------------------------------

pub fn fn_mvnpdf(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity("mvnpdf", "1 to 3 (X [, mu [, Sigma]])", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let (n, d) = (x.rows(), x.cols());
    let mu_m = if let Some(v) = args.get(1) {
        let m = v.to_dense_matrix()?;
        if m.is_empty() {
            Matrix::zeros(1, d)
        } else {
            m
        }
    } else {
        Matrix::zeros(1, d)
    };
    let sigma = if let Some(v) = args.get(2) {
        let s = v.to_dense_matrix()?;
        if s.rows() == 1 && s.cols() == d {
            // Diagonal variance vector
            let mut diag = Matrix::zeros(d, d);
            for i in 0..d {
                diag.set(i, i, s.data()[i]);
            }
            diag
        } else if s.len() == 1 {
            let v0 = s.data()[0];
            let mut diag = Matrix::zeros(d, d);
            for i in 0..d {
                diag.set(i, i, v0);
            }
            diag
        } else {
            s
        }
    } else {
        Matrix::eye(d, d)
    };
    if sigma.rows() != d || sigma.cols() != d {
        return Err(MatroError::dimension("mvnpdf: Sigma must be d x d"));
    }
    // Cholesky L (lower) such that Sigma = L * L'
    let (l_mat, flag) = sigma.chol_with_flag(true)?;
    if flag != 0 {
        return Err(MatroError::runtime("mvnpdf: Sigma must be symmetric positive definite"));
    }
    let mut log_det = 0.0;
    for i in 0..d {
        log_det += 2.0 * l_mat.get(i, i).ln();
    }
    let log_norm = -0.5 * ((d as f64) * (2.0 * std::f64::consts::PI).ln() + log_det);

    let mut out = Vec::with_capacity(n);
    for r in 0..n {
        let mut diff = vec![0.0; d];
        for c in 0..d {
            let mu_rc = if mu_m.rows() == n && mu_m.cols() == d {
                mu_m.get(r, c)
            } else {
                mu_m.data()[c % mu_m.len()]
            };
            diff[c] = x.get(r, c) - mu_rc;
        }
        // Forward substitution L * z = diff
        let mut z = vec![0.0; d];
        for i in 0..d {
            let mut s = diff[i];
            for j in 0..i {
                s -= l_mat.get(i, j) * z[j];
            }
            z[i] = s / l_mat.get(i, i);
        }
        let quad: f64 = z.iter().map(|v| v * v).sum();
        out.push((log_norm - 0.5 * quad).exp());
    }
    if n == 1 {
        ok1(Value::scalar(out[0]))
    } else {
        ok1(Value::matrix(Matrix::col_vec(out)))
    }
}

pub fn fn_mvnrnd(args: &[Value], rng: &mut Rng) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity("mvnrnd", "2 or 3 (mu, Sigma [, n])", args.len()));
    }
    let mu_in = args[0].to_dense_matrix()?;
    let sigma_in = args[1].to_dense_matrix()?;
    let (n, d, mu_is_row_per_sample) = if let Some(nv) = args.get(2) {
        let n = nv.as_scalar()?.max(0.0).round() as usize;
        let d = mu_in.len();
        (n, d, false)
    } else if mu_in.rows() > 1 && mu_in.cols() > 1 {
        (mu_in.rows(), mu_in.cols(), true)
    } else {
        (1, mu_in.len(), false)
    };
    let sigma = if sigma_in.rows() == 1 && sigma_in.cols() == d {
        let mut diag = Matrix::zeros(d, d);
        for i in 0..d {
            diag.set(i, i, sigma_in.data()[i]);
        }
        diag
    } else if sigma_in.len() == 1 {
        let v0 = sigma_in.data()[0];
        let mut diag = Matrix::zeros(d, d);
        for i in 0..d {
            diag.set(i, i, v0);
        }
        diag
    } else {
        sigma_in
    };
    if sigma.rows() != d || sigma.cols() != d {
        return Err(MatroError::dimension("mvnrnd: Sigma must be d x d"));
    }
    // Factor Sigma = T * T': prefer Cholesky, fall back to symmetric SVD for positive semi-definite
    let factor = match sigma.chol_with_flag(true) {
        Ok((l, 0)) => l,
        _ => {
            let (u, s, _) = sigma.svd()?;
            let mut t = Matrix::zeros(d, d);
            for c in 0..d {
                let sv = s.get(c, c).max(0.0).sqrt();
                for r in 0..d {
                    t.set(r, c, u.get(r, c) * sv);
                }
            }
            t
        }
    };

    let mut out = Matrix::zeros(n, d);
    for r in 0..n {
        let z: Vec<f64> = (0..d).map(|_| rng.next_normal()).collect();
        for c in 0..d {
            let mut val = if mu_is_row_per_sample {
                mu_in.get(r, c)
            } else {
                mu_in.data()[c]
            };
            for k in 0..d {
                val += factor.get(c, k) * z[k];
            }
            out.set(r, c, val);
        }
    }
    ok1(Value::matrix(out))
}

// ---------------------------------------------------------------------------
// Additional Continuous Distributions (logn, gam, beta, unif)
// ---------------------------------------------------------------------------

pub fn fn_logn_dist(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity(name, "1 to 3 (x [, mu, sigma])", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let mu = args.get(1).map(|v| v.to_dense_matrix()).transpose()?.unwrap_or_else(|| Matrix::scalar(0.0));
    let sigma = args.get(2).map(|v| v.to_dense_matrix()).transpose()?.unwrap_or_else(|| Matrix::scalar(1.0));
    let out = match name {
        "lognpdf" => broadcast_3(&x, &mu, &sigma, |xv, mv, sv| {
            if sv <= 0.0 || sv.is_nan() || xv.is_nan() {
                f64::NAN
            } else if xv <= 0.0 {
                0.0
            } else {
                let z = (xv.ln() - mv) / sv;
                (-0.5 * z * z).exp() / (xv * sv * (2.0 * std::f64::consts::PI).sqrt())
            }
        })?,
        "logncdf" => broadcast_3(&x, &mu, &sigma, |xv, mv, sv| {
            if sv <= 0.0 || sv.is_nan() || xv.is_nan() {
                f64::NAN
            } else if xv <= 0.0 {
                0.0
            } else {
                0.5 * (1.0 + erf_f64((xv.ln() - mv) / (sv * std::f64::consts::SQRT_2)))
            }
        })?,
        "logninv" => broadcast_3(&x, &mu, &sigma, |pv, mv, sv| {
            if sv <= 0.0 || sv.is_nan() || pv.is_nan() || pv < 0.0 || pv > 1.0 {
                f64::NAN
            } else if pv == 0.0 {
                0.0
            } else if pv == 1.0 {
                f64::INFINITY
            } else {
                (mv + sv * std_norm_inv(pv)).exp()
            }
        })?,
        _ => unreachable!(),
    };
    ok1(Value::matrix(out))
}

pub fn fn_gam_dist(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity(name, "2 or 3 (x, a [, b])", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let a = args[1].to_dense_matrix()?;
    let b = args.get(2).map(|v| v.to_dense_matrix()).transpose()?.unwrap_or_else(|| Matrix::scalar(1.0));
    let out = match name {
        "gampdf" => broadcast_3(&x, &a, &b, |xv, av, bv| {
            if av <= 0.0 || bv <= 0.0 || av.is_nan() || bv.is_nan() || xv.is_nan() {
                f64::NAN
            } else if xv < 0.0 {
                0.0
            } else if xv == 0.0 {
                if av < 1.0 { f64::INFINITY } else if (av - 1.0).abs() < 1e-15 { 1.0 / bv } else { 0.0 }
            } else {
                ((av - 1.0) * (xv / bv).ln() - xv / bv - lgamma_f64(av) - bv.ln()).exp()
            }
        })?,
        "gamcdf" => broadcast_3(&x, &a, &b, |xv, av, bv| {
            if av <= 0.0 || bv <= 0.0 || av.is_nan() || bv.is_nan() || xv.is_nan() {
                f64::NAN
            } else if xv <= 0.0 {
                0.0
            } else {
                gammainc_lower(xv / bv, av)
            }
        })?,
        "gaminv" => broadcast_3(&x, &a, &b, |pv, av, bv| {
            if av <= 0.0 || bv <= 0.0 || pv.is_nan() || pv < 0.0 || pv > 1.0 {
                return f64::NAN;
            }
            if pv == 0.0 { return 0.0; }
            if pv == 1.0 { return f64::INFINITY; }
            let mut lo = 0.0;
            let mut hi = (av + 5.0 * av.sqrt() + 5.0).max(1.0);
            while gammainc_lower(hi, av) < pv && hi < 1e9 {
                hi *= 2.0;
            }
            for _ in 0..80 {
                let mid = 0.5 * (lo + hi);
                if gammainc_lower(mid, av) < pv { lo = mid; } else { hi = mid; }
            }
            0.5 * (lo + hi) * bv
        })?,
        _ => unreachable!(),
    };
    ok1(Value::matrix(out))
}

pub fn fn_beta_dist(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 3 {
        return Err(MatroError::arity(name, "3 (x, a, b)", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let a = args[1].to_dense_matrix()?;
    let b = args[2].to_dense_matrix()?;
    let out = match name {
        "betapdf" => broadcast_3(&x, &a, &b, |xv, av, bv| {
            if av <= 0.0 || bv <= 0.0 || xv.is_nan() {
                f64::NAN
            } else if xv <= 0.0 || xv >= 1.0 {
                0.0
            } else {
                let log_b = lgamma_f64(av) + lgamma_f64(bv) - lgamma_f64(av + bv);
                ((av - 1.0) * xv.ln() + (bv - 1.0) * (1.0 - xv).ln() - log_b).exp()
            }
        })?,
        "betacdf" => broadcast_3(&x, &a, &b, |xv, av, bv| {
            if av <= 0.0 || bv <= 0.0 || xv.is_nan() {
                f64::NAN
            } else if xv <= 0.0 {
                0.0
            } else if xv >= 1.0 {
                1.0
            } else {
                betainc_reg(xv, av, bv)
            }
        })?,
        "betainv" => broadcast_3(&x, &a, &b, |pv, av, bv| {
            if av <= 0.0 || bv <= 0.0 || pv.is_nan() || pv < 0.0 || pv > 1.0 {
                return f64::NAN;
            }
            if pv == 0.0 { return 0.0; }
            if pv == 1.0 { return 1.0; }
            let mut lo = 0.0;
            let mut hi = 1.0;
            for _ in 0..80 {
                let mid = 0.5 * (lo + hi);
                if betainc_reg(mid, av, bv) < pv { lo = mid; } else { hi = mid; }
            }
            0.5 * (lo + hi)
        })?,
        _ => unreachable!(),
    };
    ok1(Value::matrix(out))
}

pub fn fn_unif_dist(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity(name, "1 to 3 (x [, a, b])", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let a = args.get(1).map(|v| v.to_dense_matrix()).transpose()?.unwrap_or_else(|| Matrix::scalar(0.0));
    let b = args.get(2).map(|v| v.to_dense_matrix()).transpose()?.unwrap_or_else(|| Matrix::scalar(1.0));
    let out = match name {
        "unifpdf" => broadcast_3(&x, &a, &b, |xv, av, bv| {
            if av >= bv || xv.is_nan() {
                f64::NAN
            } else if xv >= av && xv <= bv {
                1.0 / (bv - av)
            } else {
                0.0
            }
        })?,
        "unifcdf" => broadcast_3(&x, &a, &b, |xv, av, bv| {
            if av >= bv || xv.is_nan() {
                f64::NAN
            } else if xv <= av {
                0.0
            } else if xv >= bv {
                1.0
            } else {
                (xv - av) / (bv - av)
            }
        })?,
        "unifinv" => broadcast_3(&x, &a, &b, |pv, av, bv| {
            if av >= bv || pv.is_nan() || pv < 0.0 || pv > 1.0 {
                f64::NAN
            } else {
                av + pv * (bv - av)
            }
        })?,
        _ => unreachable!(),
    };
    ok1(Value::matrix(out))
}

// ---------------------------------------------------------------------------
// Distribution Random Number Generators (`*rnd`)
// ---------------------------------------------------------------------------

pub fn fn_dist_rnd(name: &str, args: &[Value], rng: &mut Rng) -> Result<Vec<Value>> {
    match name {
        "normrnd" | "lognrnd" => {
            if args.len() < 2 {
                return Err(MatroError::arity(name, ">= 2 (mu, sigma [, sz...])", args.len()));
            }
            let mu = args[0].to_dense_matrix()?;
            let sd = args[1].to_dense_matrix()?;
            let dims = parse_rnd_dims(&[mu.dims(), sd.dims()], &args[2..])?;
            let total: usize = dims.iter().product();
            let mut out = Vec::with_capacity(total);
            for i in 0..total {
                let m = if mu.len() == 1 { mu.data()[0] } else { mu.data()[i % mu.len()] };
                let s = if sd.len() == 1 { sd.data()[0] } else { sd.data()[i % sd.len()] };
                let v = if s < 0.0 { f64::NAN } else { m + s * rng.next_normal() };
                out.push(if name == "lognrnd" { v.exp() } else { v });
            }
            ok1(Value::matrix(Matrix::from_nd_col_major(dims, out)?))
        }
        "unifrnd" => {
            if args.len() < 2 {
                return Err(MatroError::arity("unifrnd", ">= 2 (a, b [, sz...])", args.len()));
            }
            let a = args[0].to_dense_matrix()?;
            let b = args[1].to_dense_matrix()?;
            let dims = parse_rnd_dims(&[a.dims(), b.dims()], &args[2..])?;
            let total: usize = dims.iter().product();
            let mut out = Vec::with_capacity(total);
            for i in 0..total {
                let av = if a.len() == 1 { a.data()[0] } else { a.data()[i % a.len()] };
                let bv = if b.len() == 1 { b.data()[0] } else { b.data()[i % b.len()] };
                out.push(if av > bv { f64::NAN } else { av + (bv - av) * rng.next_f64() });
            }
            ok1(Value::matrix(Matrix::from_nd_col_major(dims, out)?))
        }
        "exprnd" => {
            if args.is_empty() {
                return Err(MatroError::arity("exprnd", ">= 1 (mu [, sz...])", 0));
            }
            let mu = args[0].to_dense_matrix()?;
            let dims = parse_rnd_dims(&[mu.dims()], &args[1..])?;
            let total: usize = dims.iter().product();
            let mut out = Vec::with_capacity(total);
            for i in 0..total {
                let mv = if mu.len() == 1 { mu.data()[0] } else { mu.data()[i % mu.len()] };
                let u = rng.next_f64().max(1e-300);
                out.push(if mv < 0.0 { f64::NAN } else { -mv * u.ln() });
            }
            ok1(Value::matrix(Matrix::from_nd_col_major(dims, out)?))
        }
        "gamrnd" => {
            if args.len() < 2 {
                return Err(MatroError::arity("gamrnd", ">= 2 (a, b [, sz...])", args.len()));
            }
            let a = args[0].to_dense_matrix()?;
            let b = args[1].to_dense_matrix()?;
            let dims = parse_rnd_dims(&[a.dims(), b.dims()], &args[2..])?;
            let total: usize = dims.iter().product();
            let mut out = Vec::with_capacity(total);
            for i in 0..total {
                let av = if a.len() == 1 { a.data()[0] } else { a.data()[i % a.len()] };
                let bv = if b.len() == 1 { b.data()[0] } else { b.data()[i % b.len()] };
                out.push(if bv <= 0.0 { f64::NAN } else { sample_gamma(av, rng) * bv });
            }
            ok1(Value::matrix(Matrix::from_nd_col_major(dims, out)?))
        }
        "betarnd" => {
            if args.len() < 2 {
                return Err(MatroError::arity("betarnd", ">= 2 (a, b [, sz...])", args.len()));
            }
            let a = args[0].to_dense_matrix()?;
            let b = args[1].to_dense_matrix()?;
            let dims = parse_rnd_dims(&[a.dims(), b.dims()], &args[2..])?;
            let total: usize = dims.iter().product();
            let mut out = Vec::with_capacity(total);
            for i in 0..total {
                let av = if a.len() == 1 { a.data()[0] } else { a.data()[i % a.len()] };
                let bv = if b.len() == 1 { b.data()[0] } else { b.data()[i % b.len()] };
                let x = sample_gamma(av, rng);
                let y = sample_gamma(bv, rng);
                out.push(x / (x + y));
            }
            ok1(Value::matrix(Matrix::from_nd_col_major(dims, out)?))
        }
        "poissrnd" => {
            if args.is_empty() {
                return Err(MatroError::arity("poissrnd", ">= 1 (lambda [, sz...])", 0));
            }
            let lam = args[0].to_dense_matrix()?;
            let dims = parse_rnd_dims(&[lam.dims()], &args[1..])?;
            let total: usize = dims.iter().product();
            let mut out = Vec::with_capacity(total);
            for i in 0..total {
                let lv = if lam.len() == 1 { lam.data()[0] } else { lam.data()[i % lam.len()] };
                out.push(sample_poisson(lv, rng));
            }
            ok1(Value::matrix(Matrix::from_nd_col_major(dims, out)?))
        }
        "binornd" => {
            if args.len() < 2 {
                return Err(MatroError::arity("binornd", ">= 2 (n, p [, sz...])", args.len()));
            }
            let n_m = args[0].to_dense_matrix()?;
            let p_m = args[1].to_dense_matrix()?;
            let dims = parse_rnd_dims(&[n_m.dims(), p_m.dims()], &args[2..])?;
            let total: usize = dims.iter().product();
            let mut out = Vec::with_capacity(total);
            for i in 0..total {
                let nv = if n_m.len() == 1 { n_m.data()[0] } else { n_m.data()[i % n_m.len()] };
                let pv = if p_m.len() == 1 { p_m.data()[0] } else { p_m.data()[i % p_m.len()] };
                if nv < 0.0 || !(0.0..=1.0).contains(&pv) {
                    out.push(f64::NAN);
                } else {
                    let ni = nv.round() as usize;
                    let mut cnt = 0usize;
                    for _ in 0..ni {
                        if rng.next_f64() < pv {
                            cnt += 1;
                        }
                    }
                    out.push(cnt as f64);
                }
            }
            ok1(Value::matrix(Matrix::from_nd_col_major(dims, out)?))
        }
        "chi2rnd" => {
            if args.is_empty() {
                return Err(MatroError::arity("chi2rnd", ">= 1 (v [, sz...])", 0));
            }
            let v = args[0].to_dense_matrix()?;
            let dims = parse_rnd_dims(&[v.dims()], &args[1..])?;
            let total: usize = dims.iter().product();
            let mut out = Vec::with_capacity(total);
            for i in 0..total {
                let df = if v.len() == 1 { v.data()[0] } else { v.data()[i % v.len()] };
                out.push(2.0 * sample_gamma(0.5 * df, rng));
            }
            ok1(Value::matrix(Matrix::from_nd_col_major(dims, out)?))
        }
        "trnd" => {
            if args.is_empty() {
                return Err(MatroError::arity("trnd", ">= 1 (v [, sz...])", 0));
            }
            let v = args[0].to_dense_matrix()?;
            let dims = parse_rnd_dims(&[v.dims()], &args[1..])?;
            let total: usize = dims.iter().product();
            let mut out = Vec::with_capacity(total);
            for i in 0..total {
                let df = if v.len() == 1 { v.data()[0] } else { v.data()[i % v.len()] };
                if df <= 0.0 {
                    out.push(f64::NAN);
                } else {
                    let z = rng.next_normal();
                    let chi = 2.0 * sample_gamma(0.5 * df, rng);
                    out.push(z / (chi / df).sqrt());
                }
            }
            ok1(Value::matrix(Matrix::from_nd_col_major(dims, out)?))
        }
        "frnd" => {
            if args.len() < 2 {
                return Err(MatroError::arity("frnd", ">= 2 (v1, v2 [, sz...])", args.len()));
            }
            let v1 = args[0].to_dense_matrix()?;
            let v2 = args[1].to_dense_matrix()?;
            let dims = parse_rnd_dims(&[v1.dims(), v2.dims()], &args[2..])?;
            let total: usize = dims.iter().product();
            let mut out = Vec::with_capacity(total);
            for i in 0..total {
                let d1 = if v1.len() == 1 { v1.data()[0] } else { v1.data()[i % v1.len()] };
                let d2 = if v2.len() == 1 { v2.data()[0] } else { v2.data()[i % v2.len()] };
                if d1 <= 0.0 || d2 <= 0.0 {
                    out.push(f64::NAN);
                } else {
                    let x1 = 2.0 * sample_gamma(0.5 * d1, rng) / d1;
                    let x2 = 2.0 * sample_gamma(0.5 * d2, rng) / d2;
                    out.push(x1 / x2);
                }
            }
            ok1(Value::matrix(Matrix::from_nd_col_major(dims, out)?))
        }
        _ => unreachable!(),
    }
}

pub fn fn_randsample(args: &[Value], rng: &mut Rng) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 4 {
        return Err(MatroError::arity(
            "randsample",
            "2 to 4 (n_or_pop, k [, replacement [, weights]])",
            args.len(),
        ));
    }
    let pop_m = args[0].to_dense_matrix()?;
    let k = args[1].as_scalar()?.max(0.0).round() as usize;
    let replace = args.get(2).is_some_and(|v| v.is_truthy());
    let weights = args.get(3).map(|v| v.to_dense_matrix()).transpose()?;

    let (pop, is_scalar_n): (Vec<f64>, bool) = if pop_m.len() == 1 {
        let n = pop_m.data()[0].max(0.0).round() as usize;
        ((1..=n).map(|i| i as f64).collect(), true)
    } else {
        (pop_m.data().to_vec(), false)
    };
    let n = pop.len();
    if !replace && k > n {
        return Err(MatroError::runtime("randsample: k cannot exceed population size without replacement"));
    }

    let mut out = Vec::with_capacity(k);
    if let Some(w_m) = weights {
        let mut w: Vec<f64> = w_m.data().iter().map(|&v| v.max(0.0)).collect();
        if w.len() != n {
            return Err(MatroError::dimension("randsample: weights length must match population size"));
        }
        for _ in 0..k {
            let total: f64 = w.iter().sum();
            let target = rng.next_f64() * total;
            let mut acc = 0.0;
            let mut pick = n - 1;
            for (i, &wi) in w.iter().enumerate() {
                acc += wi;
                if acc >= target {
                    pick = i;
                    break;
                }
            }
            out.push(pop[pick]);
            if !replace {
                w[pick] = 0.0;
            }
        }
    } else if replace {
        for _ in 0..k {
            let idx = (rng.next_u64() as usize) % n;
            out.push(pop[idx]);
        }
    } else {
        let mut pool = pop;
        for i in 0..k {
            let j = i + ((rng.next_u64() as usize) % (n - i));
            pool.swap(i, j);
            out.push(pool[i]);
        }
    }
    if is_scalar_n || pop_m.rows() > 1 {
        ok1(Value::matrix(Matrix::col_vec(out)))
    } else {
        ok1(Value::matrix(Matrix::row_vec(out)))
    }
}

pub fn fn_datasample(args: &[Value], nargout: usize, rng: &mut Rng) -> Result<Vec<Value>> {
    if args.len() < 2 {
        return Err(MatroError::arity("datasample", ">= 2 (data, k [, dim] [, NV...])", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let k = args[1].as_scalar()?.max(0.0).round() as usize;
    let mut dim_opt: Option<usize> = None;
    let mut replace = true;
    let mut i = 2;
    if i < args.len() && crate::value_as_text(&args[i]).is_none() {
        dim_opt = Some(args[i].as_scalar()?.max(1.0).round() as usize);
        i += 1;
    }
    while i + 1 < args.len() {
        if let Some(key) = crate::value_as_text(&args[i]) {
            if key.eq_ignore_ascii_case("replace") {
                replace = args[i + 1].is_truthy();
            }
        }
        i += 2;
    }
    let dim = dim_opt.unwrap_or_else(|| if x.rows() == 1 && x.cols() > 1 { 2 } else { 1 });
    let n = if dim == 1 { x.rows() } else { x.cols() };
    if !replace && k > n {
        return Err(MatroError::runtime("datasample: k cannot exceed size along dim without replacement"));
    }
    let mut idx = Vec::with_capacity(k);
    if replace {
        for _ in 0..k {
            idx.push((rng.next_u64() as usize) % n);
        }
    } else {
        let mut pool: Vec<usize> = (0..n).collect();
        for s in 0..k {
            let j = s + ((rng.next_u64() as usize) % (n - s));
            pool.swap(s, j);
            idx.push(pool[s]);
        }
    }
    let out_m = if dim == 1 {
        let mut m = Matrix::zeros(k, x.cols());
        for (ri, &src_r) in idx.iter().enumerate() {
            for c in 0..x.cols() {
                m.set(ri, c, x.get(src_r, c));
            }
        }
        m
    } else {
        let mut m = Matrix::zeros(x.rows(), k);
        for (ci, &src_c) in idx.iter().enumerate() {
            for r in 0..x.rows() {
                m.set(r, ci, x.get(r, src_c));
            }
        }
        m
    };
    if nargout >= 2 {
        let idx_v = Matrix::row_vec(idx.iter().map(|&j| (j + 1) as f64).collect());
        Ok(vec![Value::matrix(out_m), Value::matrix(idx_v)])
    } else {
        ok1(Value::matrix(out_m))
    }
}

// ---------------------------------------------------------------------------
// Correlation (`corr`), Pairwise Distances (`pdist2`), KNN Search (`knnsearch`)
// ---------------------------------------------------------------------------

fn tied_ranks(xs: &[f64]) -> Vec<f64> {
    let n = xs.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| xs[a].total_cmp(&xs[b]));
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (xs[order[j]] - xs[order[i]]).abs() == 0.0 {
            j += 1;
        }
        let avg_rank = 0.5 * ((i + 1) + j) as f64;
        for &idx in &order[i..j] {
            ranks[idx] = avg_rank;
        }
        i = j;
    }
    ranks
}

fn pearson_pair(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len();
    if n < 2 {
        return (f64::NAN, f64::NAN);
    }
    let mx = x.iter().sum::<f64>() / n as f64;
    let my = y.iter().sum::<f64>() / n as f64;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    let denom = (sxx * syy).sqrt();
    if denom <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let r = (sxy / denom).clamp(-1.0, 1.0);
    let pval = if n > 2 {
        let d2 = (1.0 - r * r).max(0.0);
        if d2 <= 1e-30 {
            0.0
        } else {
            let t = r.abs() * ((n - 2) as f64 / d2).sqrt();
            2.0 * (1.0 - tcdf_scalar(t, (n - 2) as f64))
        }
    } else {
        f64::NAN
    };
    (r, pval.clamp(0.0, 1.0))
}

fn kendall_pair(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len();
    if n < 2 {
        return (f64::NAN, f64::NAN);
    }
    let mut conc = 0i64;
    let mut disc = 0i64;
    let mut tx = 0i64;
    let mut ty = 0i64;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[j] - x[i];
            let dy = y[j] - y[i];
            if dx == 0.0 && dy == 0.0 {
                continue;
            } else if dx == 0.0 {
                tx += 1;
            } else if dy == 0.0 {
                ty += 1;
            } else if dx * dy > 0.0 {
                conc += 1;
            } else {
                disc += 1;
            }
        }
    }
    let n0 = (conc + disc) as f64;
    let denom = ((n0 + tx as f64) * (n0 + ty as f64)).sqrt();
    if denom <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let tau = ((conc - disc) as f64 / denom).clamp(-1.0, 1.0);
    let nf = n as f64;
    let var_s = nf * (nf - 1.0) * (2.0 * nf + 5.0) / 18.0;
    let z = if var_s > 0.0 { (conc - disc) as f64 / var_s.sqrt() } else { 0.0 };
    let pval = (1.0 - erf_f64(z.abs() / std::f64::consts::SQRT_2)).clamp(0.0, 1.0);
    (tau, pval)
}

pub fn fn_corr(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("corr", ">= 1 (X [, Y] [, NV...])", 0));
    }
    let x = args[0].to_dense_matrix()?;
    let mut y_opt: Option<Matrix> = None;
    let mut idx = 1;
    if idx < args.len() && crate::value_as_text(&args[idx]).is_none() {
        y_opt = Some(args[idx].to_dense_matrix()?);
        idx += 1;
    }
    let mut corr_type = "pearson".to_string();
    let mut complete_rows = false;
    while idx + 1 < args.len() {
        if let Some(k) = crate::value_as_text(&args[idx]) {
            match k.to_ascii_lowercase().as_str() {
                "type" => corr_type = args[idx + 1].as_str()?.to_ascii_lowercase(),
                "rows" => {
                    let rv = args[idx + 1].as_str()?.to_ascii_lowercase();
                    complete_rows = rv == "complete" || rv == "pairwise";
                }
                _ => {}
            }
        }
        idx += 2;
    }
    let y = y_opt.unwrap_or_else(|| x.clone());
    if x.rows() != y.rows() {
        return Err(MatroError::dimension("corr: X and Y must have the same number of rows"));
    }
    let (n, px, py) = (x.rows(), x.cols(), y.cols());
    let mut rho = Matrix::zeros(px, py);
    let mut pval = Matrix::zeros(px, py);
    for cx in 0..px {
        for cy in 0..py {
            let mut xv = Vec::with_capacity(n);
            let mut yv = Vec::with_capacity(n);
            let mut any_nan = false;
            for r in 0..n {
                let a = x.get(r, cx);
                let b = y.get(r, cy);
                if a.is_nan() || b.is_nan() {
                    if complete_rows {
                        continue;
                    } else {
                        any_nan = true;
                        break;
                    }
                }
                xv.push(a);
                yv.push(b);
            }
            let (r, p) = if any_nan {
                (f64::NAN, f64::NAN)
            } else {
                match corr_type.as_str() {
                    "spearman" => pearson_pair(&tied_ranks(&xv), &tied_ranks(&yv)),
                    "kendall" => kendall_pair(&xv, &yv),
                    _ => pearson_pair(&xv, &yv),
                }
            };
            rho.set(cx, cy, r);
            pval.set(cx, cy, p);
        }
    }
    if nargout >= 2 {
        Ok(vec![Value::matrix(rho), Value::matrix(pval)])
    } else {
        ok1(Value::matrix(rho))
    }
}

pub fn fn_pdist2(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 {
        return Err(MatroError::arity("pdist2", ">= 2 (X, Y [, metric [, p]])", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let y = args[1].to_dense_matrix()?;
    if x.cols() != y.cols() {
        return Err(MatroError::dimension("pdist2: X and Y must have the same number of columns"));
    }
    let metric = args
        .get(2)
        .and_then(crate::value_as_text)
        .unwrap_or_else(|| "euclidean".to_string())
        .to_ascii_lowercase();
    let p_exp = args.get(3).and_then(|v| v.as_scalar().ok()).unwrap_or(2.0);
    let (nx, ny, d) = (x.rows(), y.rows(), x.cols());
    let x_rows: Vec<Vec<f64>> = (0..nx).map(|r| (0..d).map(|c| x.get(r, c)).collect()).collect();
    let y_rows: Vec<Vec<f64>> = (0..ny).map(|r| (0..d).map(|c| y.get(r, c)).collect()).collect();
    let mut out = Matrix::zeros(nx, ny);
    for i in 0..nx {
        for j in 0..ny {
            out.set(i, j, pair_distance(&x_rows[i], &y_rows[j], &metric, p_exp));
        }
    }
    ok1(Value::matrix(out))
}

pub fn fn_knnsearch(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() < 2 {
        return Err(MatroError::arity("knnsearch", ">= 2 (X, Y [, 'K', k, 'Distance', d])", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let y = args[1].to_dense_matrix()?;
    if x.cols() != y.cols() {
        return Err(MatroError::dimension("knnsearch: X and Y must have the same number of columns"));
    }
    let mut k = 1usize;
    let mut metric = "euclidean".to_string();
    let mut i = 2;
    while i + 1 < args.len() {
        if let Some(key) = crate::value_as_text(&args[i]) {
            match key.to_ascii_lowercase().as_str() {
                "k" => k = args[i + 1].as_scalar()?.max(1.0).round() as usize,
                "distance" => metric = args[i + 1].as_str()?.to_ascii_lowercase(),
                _ => {}
            }
        }
        i += 2;
    }
    let (nx, ny, d) = (x.rows(), y.rows(), x.cols());
    let k = k.min(nx);
    let x_rows: Vec<Vec<f64>> = (0..nx).map(|r| (0..d).map(|c| x.get(r, c)).collect()).collect();
    let y_rows: Vec<Vec<f64>> = (0..ny).map(|r| (0..d).map(|c| y.get(r, c)).collect()).collect();
    let mut idx_mat = Matrix::zeros(ny, k);
    let mut dist_mat = Matrix::zeros(ny, k);
    for qi in 0..ny {
        let mut pairs: Vec<(f64, usize)> = (0..nx)
            .map(|xi| (pair_distance(&y_rows[qi], &x_rows[xi], &metric, 2.0), xi))
            .collect();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        for j in 0..k {
            idx_mat.set(qi, j, (pairs[j].1 + 1) as f64);
            dist_mat.set(qi, j, pairs[j].0);
        }
    }
    if nargout >= 2 {
        Ok(vec![Value::matrix(idx_mat), Value::matrix(dist_mat)])
    } else {
        ok1(Value::matrix(idx_mat))
    }
}

// ---------------------------------------------------------------------------
// Nonparametric Hypothesis Tests (`kstest`, `kstest2`, `ranksum`, `signrank`)
// ---------------------------------------------------------------------------

fn ks_asymp_p(z: f64) -> f64 {
    if z <= 0.0 {
        return 1.0;
    }
    let mut sum = 0.0;
    for k in 1..=100 {
        let term = (-2.0 * (k as f64).powi(2) * z * z).exp();
        if k % 2 == 1 {
            sum += term;
        } else {
            sum -= term;
        }
        if term < 1e-16 {
            break;
        }
    }
    (2.0 * sum).clamp(0.0, 1.0)
}

pub fn fn_kstest(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("kstest", ">= 1 (x [, 'Alpha', alpha])", 0));
    }
    let x_m = args[0].to_dense_matrix()?;
    let mut alpha = 0.05;
    let mut i = 1;
    while i + 1 < args.len() {
        if let Some(k) = crate::value_as_text(&args[i]) {
            if k.eq_ignore_ascii_case("alpha") {
                alpha = args[i + 1].as_scalar()?;
            }
        }
        i += 2;
    }
    let mut xs: Vec<f64> = x_m.data().iter().copied().filter(|v| !v.is_nan()).collect();
    xs.sort_by(f64::total_cmp);
    let n = xs.len();
    if n == 0 {
        return Err(MatroError::dimension("kstest: empty sample"));
    }
    let mut d = 0.0f64;
    for (idx, &v) in xs.iter().enumerate() {
        let cdf = 0.5 * (1.0 + erf_f64(v / std::f64::consts::SQRT_2));
        let f_lo = idx as f64 / n as f64;
        let f_hi = (idx + 1) as f64 / n as f64;
        d = d.max((f_hi - cdf).abs()).max((cdf - f_lo).abs());
    }
    let en = (n as f64).sqrt();
    let p = ks_asymp_p((en + 0.12 + 0.11 / en) * d);
    let h = if p < alpha { 1.0 } else { 0.0 };
    let mut out = vec![Value::scalar(h)];
    if nargout >= 2 { out.push(Value::scalar(p)); }
    if nargout >= 3 { out.push(Value::scalar(d)); }
    Ok(out)
}

pub fn fn_kstest2(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() < 2 {
        return Err(MatroError::arity("kstest2", ">= 2 (x1, x2 [, 'Alpha', alpha])", args.len()));
    }
    let x1_m = args[0].to_dense_matrix()?;
    let x2_m = args[1].to_dense_matrix()?;
    let mut alpha = 0.05;
    let mut i = 2;
    while i + 1 < args.len() {
        if let Some(k) = crate::value_as_text(&args[i]) {
            if k.eq_ignore_ascii_case("alpha") {
                alpha = args[i + 1].as_scalar()?;
            }
        }
        i += 2;
    }
    let mut x1: Vec<f64> = x1_m.data().iter().copied().filter(|v| !v.is_nan()).collect();
    let mut x2: Vec<f64> = x2_m.data().iter().copied().filter(|v| !v.is_nan()).collect();
    x1.sort_by(f64::total_cmp);
    x2.sort_by(f64::total_cmp);
    let (n1, n2) = (x1.len(), x2.len());
    if n1 == 0 || n2 == 0 {
        return Err(MatroError::dimension("kstest2: samples cannot be empty"));
    }
    let (mut i1, mut i2) = (0usize, 0usize);
    let mut d = 0.0f64;
    while i1 < n1 || i2 < n2 {
        let v = if i1 < n1 && i2 < n2 {
            x1[i1].min(x2[i2])
        } else if i1 < n1 {
            x1[i1]
        } else {
            x2[i2]
        };
        while i1 < n1 && x1[i1] <= v {
            i1 += 1;
        }
        while i2 < n2 && x2[i2] <= v {
            i2 += 1;
        }
        let diff = (i1 as f64 / n1 as f64 - i2 as f64 / n2 as f64).abs();
        d = d.max(diff);
    }
    let en = ((n1 * n2) as f64 / (n1 + n2) as f64).sqrt();
    let p = ks_asymp_p((en + 0.12 + 0.11 / en) * d);
    let h = if p < alpha { 1.0 } else { 0.0 };
    let mut out = vec![Value::scalar(h)];
    if nargout >= 2 { out.push(Value::scalar(p)); }
    if nargout >= 3 { out.push(Value::scalar(d)); }
    Ok(out)
}

pub fn fn_ranksum(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() < 2 {
        return Err(MatroError::arity("ranksum", ">= 2 (x, y [, 'Alpha', alpha])", args.len()));
    }
    let x: Vec<f64> = args[0].to_dense_matrix()?.data().iter().copied().filter(|v| !v.is_nan()).collect();
    let y: Vec<f64> = args[1].to_dense_matrix()?.data().iter().copied().filter(|v| !v.is_nan()).collect();
    let mut alpha = 0.05;
    let mut i = 2;
    while i + 1 < args.len() {
        if let Some(k) = crate::value_as_text(&args[i]) {
            if k.eq_ignore_ascii_case("alpha") {
                alpha = args[i + 1].as_scalar()?;
            }
        }
        i += 2;
    }
    let (nx, ny) = (x.len(), y.len());
    if nx == 0 || ny == 0 {
        return Err(MatroError::dimension("ranksum: empty sample"));
    }
    let mut combined = Vec::with_capacity(nx + ny);
    combined.extend_from_slice(&x);
    combined.extend_from_slice(&y);
    let ranks = tied_ranks(&combined);
    let w: f64 = ranks[..nx].iter().sum();
    let mu = (nx as f64) * ((nx + ny + 1) as f64) / 2.0;
    let sigma = ((nx * ny) as f64 * ((nx + ny + 1) as f64) / 12.0).sqrt();
    let z = if sigma > 0.0 {
        let diff = w - mu;
        let cc = if diff.abs() > 0.5 { diff - 0.5 * diff.signum() } else { 0.0 };
        cc / sigma
    } else {
        0.0
    };
    let p = (1.0 - erf_f64(z.abs() / std::f64::consts::SQRT_2)).clamp(0.0, 1.0);
    let h = if p < alpha { 1.0 } else { 0.0 };
    let mut out = vec![Value::scalar(p)];
    if nargout >= 2 { out.push(Value::scalar(h)); }
    if nargout >= 3 {
        let mut st = std::collections::BTreeMap::new();
        st.insert("zval".to_string(), Value::scalar(z));
        st.insert("ranksum".to_string(), Value::scalar(w));
        out.push(Value::Struct(st));
    }
    Ok(out)
}

pub fn fn_signrank(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("signrank", ">= 1 (x [, y|m] [, 'Alpha', alpha])", 0));
    }
    let x = args[0].to_dense_matrix()?;
    let mut diffs = x.data().to_vec();
    let mut idx = 1;
    if idx < args.len() && crate::value_as_text(&args[idx]).is_none() {
        let y = args[idx].to_dense_matrix()?;
        if y.len() == 1 {
            let m = y.data()[0];
            for v in &mut diffs {
                *v -= m;
            }
        } else if y.len() == diffs.len() {
            for (v, &yv) in diffs.iter_mut().zip(y.data().iter()) {
                *v -= yv;
            }
        }
        idx += 1;
    }
    let mut alpha = 0.05;
    while idx + 1 < args.len() {
        if let Some(k) = crate::value_as_text(&args[idx]) {
            if k.eq_ignore_ascii_case("alpha") {
                alpha = args[idx + 1].as_scalar()?;
            }
        }
        idx += 2;
    }
    let nz: Vec<f64> = diffs.into_iter().filter(|v| !v.is_nan() && *v != 0.0).collect();
    let n = nz.len();
    if n == 0 {
        let mut out = vec![Value::scalar(1.0)];
        if nargout >= 2 { out.push(Value::scalar(0.0)); }
        return Ok(out);
    }
    let abs_nz: Vec<f64> = nz.iter().map(|v| v.abs()).collect();
    let ranks = tied_ranks(&abs_nz);
    let w_pos: f64 = nz
        .iter()
        .zip(ranks.iter())
        .filter(|(d, _)| **d > 0.0)
        .map(|(_, &r)| r)
        .sum();
    let nf = n as f64;
    let mu = nf * (nf + 1.0) / 4.0;
    let sigma = (nf * (nf + 1.0) * (2.0 * nf + 1.0) / 24.0).sqrt();
    let z = if sigma > 0.0 {
        let diff = w_pos - mu;
        let cc = if diff.abs() > 0.5 { diff - 0.5 * diff.signum() } else { 0.0 };
        cc / sigma
    } else {
        0.0
    };
    let p = (1.0 - erf_f64(z.abs() / std::f64::consts::SQRT_2)).clamp(0.0, 1.0);
    let h = if p < alpha { 1.0 } else { 0.0 };
    let mut out = vec![Value::scalar(p)];
    if nargout >= 2 { out.push(Value::scalar(h)); }
    if nargout >= 3 {
        let mut st = std::collections::BTreeMap::new();
        st.insert("zval".to_string(), Value::scalar(z));
        st.insert("signedrank".to_string(), Value::scalar(w_pos));
        out.push(Value::Struct(st));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Descriptive & NaN-Safe Stats (`geomean`, `harmmean`, `trimmean`, `mode`, `nan*`)
// ---------------------------------------------------------------------------

pub fn fn_extra_means(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity(name, ">= 1", 0));
    }
    let x = args[0].to_dense_matrix()?;
    match name {
        "trimmean" => {
            if args.len() < 2 {
                return Err(MatroError::arity("trimmean", ">= 2 (X, percent [, dim])", args.len()));
            }
            let pct = args[1].as_scalar()?;
            let mut all_flag = false;
            let mut dims_opt: Option<Vec<usize>> = None;
            if let Some(a2) = args.get(2) {
                if crate::value_as_text(a2).is_some_and(|s| s.eq_ignore_ascii_case("all")) {
                    all_flag = true;
                } else {
                    let dm = a2.to_dense_matrix()?;
                    dims_opt = Some(dm.data().iter().map(|&v| v.max(1.0).round() as usize).collect());
                }
            }
            ok1(Value::matrix(reduce_nd_real(&x, all_flag, dims_opt.as_deref(), |xs| {
                let mut v: Vec<f64> = xs.iter().copied().filter(|u| !u.is_nan()).collect();
                if v.is_empty() {
                    return f64::NAN;
                }
                v.sort_by(f64::total_cmp);
                let k = ((v.len() as f64) * (pct / 100.0) * 0.5).floor() as usize;
                let lo = k.min(v.len());
                let hi = v.len().saturating_sub(k);
                if lo >= hi {
                    return f64::NAN;
                }
                v[lo..hi].iter().sum::<f64>() / (hi - lo) as f64
            })?))
        }
        "geomean" | "harmmean" => {
            let mut all_flag = false;
            let mut omit_nan = false;
            let mut dims_opt: Option<Vec<usize>> = None;
            for a in &args[1..] {
                if let Some(s) = crate::value_as_text(a) {
                    match s.to_ascii_lowercase().as_str() {
                        "all" => all_flag = true,
                        "omitnan" | "omitmissing" => omit_nan = true,
                        _ => {}
                    }
                } else {
                    let dm = a.to_dense_matrix()?;
                    dims_opt = Some(dm.data().iter().map(|&v| v.max(1.0).round() as usize).collect());
                }
            }
            ok1(Value::matrix(reduce_nd_real(&x, all_flag, dims_opt.as_deref(), |xs| {
                let vals: Vec<f64> = if omit_nan {
                    xs.iter().copied().filter(|v| !v.is_nan()).collect()
                } else {
                    if xs.iter().any(|v| v.is_nan()) {
                        return f64::NAN;
                    }
                    xs.to_vec()
                };
                if vals.is_empty() {
                    return f64::NAN;
                }
                if name == "geomean" {
                    if vals.iter().any(|&v| v < 0.0) {
                        return f64::NAN;
                    }
                    if vals.iter().any(|&v| v == 0.0) {
                        return 0.0;
                    }
                    (vals.iter().map(|&v| v.ln()).sum::<f64>() / vals.len() as f64).exp()
                } else {
                    let inv_sum: f64 = vals.iter().map(|&v| 1.0 / v).sum();
                    vals.len() as f64 / inv_sum
                }
            })?))
        }
        "nansum" | "nanmean" | "nanmedian" | "nanstd" | "nanvar" => {
            let mut flag_pop = false;
            let mut dim_idx = 1;
            if matches!(name, "nanstd" | "nanvar") && args.len() >= 2 {
                let fm = args[1].to_dense_matrix()?;
                if !fm.is_empty() {
                    flag_pop = fm.data()[0] != 0.0;
                }
                dim_idx = 2;
            }
            let dims_opt = args.get(dim_idx).map(|v| {
                v.to_dense_matrix().map(|dm| {
                    dm.data().iter().map(|&u| u.max(1.0).round() as usize).collect::<Vec<_>>()
                })
            }).transpose()?;
            ok1(Value::matrix(reduce_nd_real(&x, false, dims_opt.as_deref(), |xs| {
                let vals: Vec<f64> = xs.iter().copied().filter(|v| !v.is_nan()).collect();
                match name {
                    "nansum" => vals.iter().sum(),
                    "nanmean" => {
                        if vals.is_empty() { f64::NAN } else { vals.iter().sum::<f64>() / vals.len() as f64 }
                    }
                    "nanmedian" => {
                        if vals.is_empty() {
                            return f64::NAN;
                        }
                        let mut s = vals;
                        s.sort_by(f64::total_cmp);
                        let n = s.len();
                        if n % 2 == 1 { s[n / 2] } else { 0.5 * (s[n / 2 - 1] + s[n / 2]) }
                    }
                    "nanstd" | "nanvar" => {
                        let n = vals.len();
                        if n == 0 { return f64::NAN; }
                        if n == 1 { return 0.0; }
                        let mu = vals.iter().sum::<f64>() / n as f64;
                        let ss: f64 = vals.iter().map(|&v| (v - mu) * (v - mu)).sum();
                        let v = ss / if flag_pop { n as f64 } else { (n - 1) as f64 };
                        if name == "nanstd" { v.sqrt() } else { v }
                    }
                    _ => unreachable!(),
                }
            })?))
        }
        _ => unreachable!(),
    }
}

pub fn fn_mode(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("mode", "1 or 2 (A [, dim|'all'])", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let all_flag = args
        .get(1)
        .and_then(crate::value_as_text)
        .is_some_and(|s| s.eq_ignore_ascii_case("all"));
    let dims_opt = if all_flag {
        None
    } else if let Some(a1) = args.get(1) {
        let dm = a1.to_dense_matrix()?;
        Some(dm.data().iter().map(|&v| v.max(1.0).round() as usize).collect::<Vec<_>>())
    } else {
        None
    };

    let mode_of_slice = |xs: &[f64]| -> (f64, f64) {
        let mut vals: Vec<f64> = xs.iter().copied().filter(|v| !v.is_nan()).collect();
        if vals.is_empty() {
            return (f64::NAN, 0.0);
        }
        vals.sort_by(f64::total_cmp);
        let mut best_val = vals[0];
        let mut best_cnt = 1usize;
        let mut cur_val = vals[0];
        let mut cur_cnt = 1usize;
        for &v in &vals[1..] {
            if v == cur_val {
                cur_cnt += 1;
            } else {
                if cur_cnt > best_cnt {
                    best_cnt = cur_cnt;
                    best_val = cur_val;
                }
                cur_val = v;
                cur_cnt = 1;
            }
        }
        if cur_cnt > best_cnt {
            best_cnt = cur_cnt;
            best_val = cur_val;
        }
        (best_val, best_cnt as f64)
    };

    let m_mat = reduce_nd_real(&x, all_flag, dims_opt.as_deref(), |xs| mode_of_slice(xs).0)?;
    if nargout >= 2 {
        let f_mat = reduce_nd_real(&x, all_flag, dims_opt.as_deref(), |xs| mode_of_slice(xs).1)?;
        Ok(vec![Value::matrix(m_mat), Value::matrix(f_mat)])
    } else {
        ok1(Value::matrix(m_mat))
    }
}
