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

fn slice_var_c(xs: &[(f64, f64)], w: Option<&[f64]>, pop: bool, omit_nan: bool) -> (f64, (f64, f64)) {
    let mut vals: Vec<(f64, f64)> = Vec::with_capacity(xs.len());
    let mut wts: Vec<f64> = Vec::new();
    for (i, &(re, im)) in xs.iter().enumerate() {
        let is_nan = re.is_nan() || im.is_nan();
        if is_nan {
            if omit_nan {
                continue;
            } else {
                return (f64::NAN, (f64::NAN, f64::NAN));
            }
        }
        vals.push((re, im));
        if let Some(wv) = w {
            wts.push(wv[i]);
        }
    }
    let n = vals.len();
    if n == 0 {
        return (f64::NAN, (f64::NAN, 0.0));
    }
    if let Some(_) = w {
        let wsum: f64 = wts.iter().sum();
        if wsum <= 0.0 {
            return (f64::NAN, (f64::NAN, 0.0));
        }
        let mr: f64 = vals.iter().zip(wts.iter()).map(|(v, &wi)| v.0 * wi).sum::<f64>() / wsum;
        let mi: f64 = vals.iter().zip(wts.iter()).map(|(v, &wi)| v.1 * wi).sum::<f64>() / wsum;
        let ss: f64 = vals
            .iter()
            .zip(wts.iter())
            .map(|(v, &wi)| wi * ((v.0 - mr) * (v.0 - mr) + (v.1 - mi) * (v.1 - mi)))
            .sum();
        return (ss / wsum, (mr, mi));
    }
    let mr = vals.iter().map(|v| v.0).sum::<f64>() / n as f64;
    let mi = vals.iter().map(|v| v.1).sum::<f64>() / n as f64;
    if n == 1 {
        return (0.0, (mr, mi));
    }
    let ss: f64 = vals
        .iter()
        .map(|v| (v.0 - mr) * (v.0 - mr) + (v.1 - mi) * (v.1 - mi))
        .sum();
    let denom = if pop { n as f64 } else { (n - 1) as f64 };
    (ss / denom, (mr, mi))
}

fn slice_var(xs: &[f64], pop: bool) -> f64 {
    let pairs: Vec<(f64, f64)> = xs.iter().map(|&x| (x, 0.0)).collect();
    slice_var_c(&pairs, None, pop, false).0
}

fn slice_median_opt(xs: &[f64], omit_nan: bool) -> f64 {
    let mut v = Vec::with_capacity(xs.len());
    for &x in xs {
        if x.is_nan() {
            if omit_nan {
                continue;
            } else {
                return f64::NAN;
            }
        }
        v.push(x);
    }
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

pub fn slice_quantile(xs: &[f64], q: f64) -> f64 {
    if !(0.0..=1.0).contains(&q) {
        return f64::NAN;
    }
    // MATLAB quantile/prctile omit NaNs by default
    let mut v: Vec<f64> = xs.iter().copied().filter(|x| !x.is_nan()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
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

/// N-D reduction along one or more dimensions (0-based `reduce_dims`) or `'all'`.
pub fn reduce_nd_real(
    m: &Matrix,
    all: bool,
    dims_1based: Option<&[usize]>,
    f: impl Fn(&[f64]) -> f64,
) -> Result<Matrix> {
    if all {
        return Ok(Matrix::scalar(f(m.data())));
    }
    let dims = m.dims();
    let target_dims: Vec<usize> = match dims_1based {
        Some(ds) => ds.iter().map(|&d| d.saturating_sub(1)).collect(),
        None => {
            let first = dims.iter().position(|&d| d != 1).unwrap_or(0);
            vec![first]
        }
    };
    let max_d = target_dims.iter().copied().max().unwrap_or(0) + 1;
    let nd = dims.len().max(max_d);
    let mut ext_dims = dims.clone();
    ext_dims.resize(nd, 1);

    let mut out_dims = ext_dims.clone();
    for &td in &target_dims {
        out_dims[td] = 1;
    }
    while out_dims.len() > 2 && *out_dims.last().unwrap() == 1 {
        out_dims.pop();
    }
    let out_total: usize = out_dims.iter().product();
    if out_total == 0 {
        return Ok(Matrix::zeros_nd(out_dims));
    }

    let mut in_strides = vec![1usize; nd];
    for d in 1..nd {
        in_strides[d] = in_strides[d - 1] * ext_dims[d - 1];
    }
    let mut out_ext = ext_dims.clone();
    for &td in &target_dims {
        out_ext[td] = 1;
    }
    let mut out_strides = vec![1usize; nd];
    for d in 1..nd {
        out_strides[d] = out_strides[d - 1] * out_ext[d - 1];
    }

    let mut buckets: Vec<Vec<f64>> = vec![Vec::new(); out_total];
    for lin in 0..m.len() {
        let mut rem = lin;
        let mut out_lin = 0usize;
        for d in 0..nd {
            let coord = rem % ext_dims[d];
            rem /= ext_dims[d];
            if out_ext[d] > 1 {
                out_lin += coord * out_strides[d];
            }
        }
        buckets[out_lin].push(m.data()[lin]);
    }
    let out_data: Vec<f64> = buckets.iter().map(|b| f(b)).collect();
    Matrix::from_nd_col_major(out_dims, out_data)
}

/// `std(x [, w [, dim [, nanflag]]])` and `var(x [, w [, dim [, nanflag]]])` with `[S, M]` outputs.
pub fn fn_std_var(name: &str, args: &[Value], nargout: usize, take_sqrt: bool) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 4 {
        return Err(MatroError::arity(name, "1 to 4", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let mut omit_nan = false;
    let mut all_flag = false;
    let mut pos_args: Vec<&Value> = Vec::new();
    for a in &args[1..] {
        if let Some(s) = crate::value_as_text(a) {
            match s.to_ascii_lowercase().as_str() {
                "omitnan" | "omitmissing" => omit_nan = true,
                "includenan" | "includemissing" => omit_nan = false,
                "all" => all_flag = true,
                other => return Err(MatroError::runtime(format!("{name}: unknown option '{other}'"))),
            }
        } else {
            pos_args.push(a);
        }
    }

    let mut pop = false;
    let mut weight_vec: Option<Vec<f64>> = None;
    if let Some(w_val) = pos_args.first() {
        let wm = w_val.to_dense_matrix()?;
        if !wm.is_empty() {
            if wm.len() == 1 {
                pop = wm.data()[0] != 0.0;
            } else {
                weight_vec = Some(wm.data().to_vec());
            }
        }
    }

    let dims_opt: Option<Vec<usize>> = if let Some(d_val) = pos_args.get(1) {
        let dm = d_val.to_dense_matrix()?;
        Some(dm.data().iter().map(|&v| v.max(1.0).round() as usize).collect())
    } else {
        None
    };

    if all_flag {
        let pairs: Vec<(f64, f64)> = (0..m.len())
            .map(|i| {
                let z = m.get_linear_c(i);
                (z.re, z.im)
            })
            .collect();
        let (v, mu) = slice_var_c(&pairs, weight_vec.as_deref(), pop, omit_nan);
        let s = if take_sqrt { v.sqrt() } else { v };
        if nargout >= 2 {
            let mu_m = Matrix::complex_scalar(matro_types::Complex::new(mu.0, mu.1)).normalize();
            return Ok(vec![Value::scalar(s), Value::matrix(mu_m)]);
        }
        return ok1(Value::scalar(s));
    }

    let dims = m.dims();
    let target_dims: Vec<usize> = match dims_opt {
        Some(ds) => ds.into_iter().map(|d| d.saturating_sub(1)).collect(),
        None => {
            let first = dims.iter().position(|&d| d != 1).unwrap_or(0);
            vec![first]
        }
    };
    let max_d = target_dims.iter().copied().max().unwrap_or(0) + 1;
    let nd = dims.len().max(max_d);
    let mut ext_dims = dims.clone();
    ext_dims.resize(nd, 1);

    let mut out_ext = ext_dims.clone();
    for &td in &target_dims {
        out_ext[td] = 1;
    }
    let mut out_dims = out_ext.clone();
    while out_dims.len() > 2 && *out_dims.last().unwrap() == 1 {
        out_dims.pop();
    }
    let out_total: usize = out_dims.iter().product();
    if out_total == 0 {
        let empty = Matrix::zeros_nd(out_dims);
        return if nargout >= 2 {
            Ok(vec![Value::matrix(empty.clone()), Value::matrix(empty)])
        } else {
            ok1(Value::matrix(empty))
        };
    }

    let mut out_strides = vec![1usize; nd];
    for d in 1..nd {
        out_strides[d] = out_strides[d - 1] * out_ext[d - 1];
    }
    let mut buckets: Vec<Vec<(f64, f64)>> = vec![Vec::new(); out_total];
    for lin in 0..m.len() {
        let mut rem = lin;
        let mut out_lin = 0usize;
        for d in 0..nd {
            let coord = rem % ext_dims[d];
            rem /= ext_dims[d];
            if out_ext[d] > 1 {
                out_lin += coord * out_strides[d];
            }
        }
        let z = m.get_linear_c(lin);
        buckets[out_lin].push((z.re, z.im));
    }

    let mut s_data = Vec::with_capacity(out_total);
    let mut mu_re = Vec::with_capacity(out_total);
    let mut mu_im = Vec::with_capacity(out_total);
    for b in &buckets {
        let w_slice = match &weight_vec {
            Some(wv) if wv.len() == b.len() => Some(wv.as_slice()),
            Some(_) => {
                return Err(MatroError::dimension(format!(
                    "{name}: weight vector length must match operating dimension"
                )));
            }
            None => None,
        };
        let (v, mu) = slice_var_c(b, w_slice, pop, omit_nan);
        s_data.push(if take_sqrt { v.sqrt() } else { v });
        mu_re.push(mu.0);
        mu_im.push(mu.1);
    }
    let s_mat = Matrix::from_nd_col_major(out_dims.clone(), s_data)?;
    if nargout >= 2 {
        let mu_mat = Matrix::from_nd_complex_col_major(out_dims, mu_re, mu_im)?.normalize();
        Ok(vec![Value::matrix(s_mat), Value::matrix(mu_mat)])
    } else {
        ok1(Value::matrix(s_mat))
    }
}

pub fn fn_median(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity("median", "1 to 3", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let mut omit_nan = false;
    let mut all_flag = false;
    let mut dims_opt: Option<Vec<usize>> = None;
    for a in &args[1..] {
        if let Some(s) = crate::value_as_text(a) {
            match s.to_ascii_lowercase().as_str() {
                "omitnan" | "omitmissing" => omit_nan = true,
                "includenan" | "includemissing" => omit_nan = false,
                "all" => all_flag = true,
                other => return Err(MatroError::runtime(format!("median: unknown option '{other}'"))),
            }
        } else {
            let dm = a.to_dense_matrix()?;
            dims_opt = Some(dm.data().iter().map(|&v| v.max(1.0).round() as usize).collect());
        }
    }
    ok1(Value::matrix(reduce_nd_real(
        &m,
        all_flag,
        dims_opt.as_deref(),
        |xs| slice_median_opt(xs, omit_nan),
    )?))
}

pub fn fn_quantile(name: &str, args: &[Value], percent: bool) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity(name, "2 or 3 (x, p [, dim])", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let p_raw = args[1].to_dense_matrix()?;
    let scale = if percent { 0.01 } else { 1.0 };

    // Integer N > 1 in `quantile(x, N)` produces N evenly spaced quantiles (1:N)/(N+1)
    let (probs, p_dims): (Vec<f64>, Vec<usize>) =
        if !percent && p_raw.len() == 1 && p_raw.data()[0] >= 1.0 && p_raw.data()[0] == p_raw.data()[0].floor() && p_raw.data()[0] > 1.0 {
            let nq = p_raw.data()[0] as usize;
            let ps: Vec<f64> = (1..=nq).map(|k| k as f64 / (nq + 1) as f64).collect();
            (ps, vec![1, nq])
        } else {
            let ps: Vec<f64> = p_raw.data().iter().map(|&v| v * scale).collect();
            (ps, p_raw.dims())
        };

    let all_flag = args
        .get(2)
        .and_then(crate::value_as_text)
        .is_some_and(|s| s.eq_ignore_ascii_case("all"));
    let dim_opt = if all_flag {
        None
    } else if let Some(d) = args.get(2) {
        Some(d.as_scalar()?.max(1.0).round() as usize)
    } else {
        None
    };

    if all_flag || (dim_opt.is_none() && (x.rows() <= 1 || x.cols() <= 1) && x.ndims() <= 2) {
        let vals: Vec<f64> = probs.iter().map(|&q| slice_quantile(x.data(), q)).collect();
        let out = if vals.len() == 1 {
            Matrix::scalar(vals[0])
        } else {
            Matrix::from_nd_col_major(p_dims, vals)?
        };
        return ok1(Value::matrix(out));
    }

    let dim = dim_opt.unwrap_or(1);
    if dim == 1 {
        let mut out = Matrix::zeros(probs.len(), x.cols());
        for c in 0..x.cols() {
            let col: Vec<f64> = (0..x.rows()).map(|r| x.get(r, c)).collect();
            for (ri, &q) in probs.iter().enumerate() {
                out.set(ri, c, slice_quantile(&col, q));
            }
        }
        ok1(Value::matrix(out))
    } else if dim == 2 {
        let mut out = Matrix::zeros(x.rows(), probs.len());
        for r in 0..x.rows() {
            let row: Vec<f64> = (0..x.cols()).map(|c| x.get(r, c)).collect();
            for (ci, &q) in probs.iter().enumerate() {
                out.set(r, ci, slice_quantile(&row, q));
            }
        }
        ok1(Value::matrix(out))
    } else {
        Err(MatroError::runtime(format!("{name}: unsupported dimension {dim}")))
    }
}

fn cov_matrix_from_cols(cols: &[Vec<f64>], pop: bool) -> Matrix {
    let k = cols.len();
    let n = cols.first().map_or(0, Vec::len);
    if k == 0 || n == 0 {
        return Matrix::empty();
    }
    let means: Vec<f64> = cols
        .iter()
        .map(|col| col.iter().sum::<f64>() / n as f64)
        .collect();
    let denom = if pop {
        n as f64
    } else if n > 1 {
        (n - 1) as f64
    } else {
        1.0
    };
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

pub fn fn_cov_or_corr(
    name: &str,
    args: &[Value],
    nargout: usize,
    normalize_corr: bool,
) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 5 {
        return Err(MatroError::arity(name, "1 to 4", args.len()));
    }
    let mut omit_rows = false;
    let mut pop = false;
    let mut data_args: Vec<&Value> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if let Some(s) = crate::value_as_text(&args[i]) {
            match s.to_ascii_lowercase().as_str() {
                "omitrows" | "partialrows" => omit_rows = true,
                "includenan" => omit_rows = false,
                "rows" => {
                    if let Some(v) = args.get(i + 1).and_then(crate::value_as_text) {
                        omit_rows = v.eq_ignore_ascii_case("complete")
                            || v.eq_ignore_ascii_case("pairwise");
                    }
                    i += 2;
                    continue;
                }
                _ => {}
            }
        } else {
            data_args.push(&args[i]);
        }
        i += 1;
    }

    // For cov: detect trailing scalar weight w (0 or 1)
    if !normalize_corr && data_args.len() >= 2 {
        let last = data_args.last().unwrap().to_dense_matrix()?;
        if last.len() == 1 && (data_args.len() == 3 || data_args[0].to_dense_matrix()?.len() > 1) {
            pop = last.data()[0] != 0.0;
            data_args.pop();
        }
    }

    let mut cols: Vec<Vec<f64>> = if data_args.len() >= 2 {
        let a = data_args[0].to_dense_matrix()?.data().to_vec();
        let b = data_args[1].to_dense_matrix()?.data().to_vec();
        if a.len() != b.len() {
            return Err(MatroError::dimension(format!(
                "{name}: input vectors must have the same length"
            )));
        }
        vec![a, b]
    } else {
        let m = data_args[0].to_dense_matrix()?;
        if m.rows() == 1 || m.cols() == 1 {
            vec![m.data().to_vec()]
        } else {
            (0..m.cols())
                .map(|c| (0..m.rows()).map(|r| m.get(r, c)).collect())
                .collect()
        }
    };

    if omit_rows && !cols.is_empty() {
        let n = cols[0].len();
        let keep: Vec<usize> = (0..n)
            .filter(|&r| cols.iter().all(|c| !c[r].is_nan()))
            .collect();
        for col in &mut cols {
            *col = keep.iter().map(|&r| col[r]).collect();
        }
    }

    let n_obs = cols.first().map_or(0, Vec::len);
    let mut cov = cov_matrix_from_cols(&cols, pop);
    if normalize_corr {
        let k = cov.rows();
        let stds: Vec<f64> = (0..k).map(|i| cov.get(i, i).max(0.0).sqrt()).collect();
        let mut pmat = Matrix::ones(k, k);
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
                if i != j && n_obs > 2 && r.is_finite() {
                    let denom = (1.0 - r * r).max(0.0);
                    let pval = if denom <= 1e-30 {
                        0.0
                    } else {
                        let t = r.abs() * ((n_obs - 2) as f64 / denom).sqrt();
                        let df = (n_obs - 2) as f64;
                        betainc(df / (df + t * t), 0.5 * df, 0.5).clamp(0.0, 1.0)
                    };
                    pmat.set(i, j, pval);
                } else if i != j {
                    pmat.set(i, j, f64::NAN);
                }
            }
        }
        if nargout >= 2 {
            return Ok(vec![Value::matrix(cov), Value::matrix(pmat)]);
        }
    }
    ok1(Value::matrix(cov))
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

pub fn fn_normpdf(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity("normpdf", "1 to 3 (x [, mu, sigma])", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let mu = args.get(1).map(|v| v.to_dense_matrix()).transpose()?.unwrap_or_else(|| Matrix::scalar(0.0));
    let sigma = args.get(2).map(|v| v.to_dense_matrix()).transpose()?.unwrap_or_else(|| Matrix::scalar(1.0));
    let out = broadcast_3(&x, &mu, &sigma, |xv, mv, sv| {
        if sv <= 0.0 || sv.is_nan() {
            f64::NAN
        } else {
            let z = (xv - mv) / sv;
            (-0.5 * z * z).exp() / (sv * (2.0 * std::f64::consts::PI).sqrt())
        }
    })?;
    ok1(Value::matrix(out))
}

pub fn fn_normcdf(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity("normcdf", "1 to 3 (x [, mu, sigma])", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let mu = args.get(1).map(|v| v.to_dense_matrix()).transpose()?.unwrap_or_else(|| Matrix::scalar(0.0));
    let sigma = args.get(2).map(|v| v.to_dense_matrix()).transpose()?.unwrap_or_else(|| Matrix::scalar(1.0));
    let out = broadcast_3(&x, &mu, &sigma, |xv, mv, sv| {
        if sv <= 0.0 || sv.is_nan() {
            f64::NAN
        } else {
            0.5 * (1.0 + erf_f64((xv - mv) / (sv * std::f64::consts::SQRT_2)))
        }
    })?;
    ok1(Value::matrix(out))
}

/// Inverse standard normal CDF (Acklam's rational approximation + Newton polish).
pub fn std_norm_inv(p: f64) -> f64 {
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
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity("norminv", "1 to 3 (p [, mu, sigma])", args.len()));
    }
    let p = args[0].to_dense_matrix()?;
    let mu = args.get(1).map(|v| v.to_dense_matrix()).transpose()?.unwrap_or_else(|| Matrix::scalar(0.0));
    let sigma = args.get(2).map(|v| v.to_dense_matrix()).transpose()?.unwrap_or_else(|| Matrix::scalar(1.0));
    let out = broadcast_3(&p, &mu, &sigma, |pv, mv, sv| {
        if sv <= 0.0 || sv.is_nan() {
            f64::NAN
        } else {
            mv + sv * std_norm_inv(pv)
        }
    })?;
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

/// `[coeff, score, latent, tsquared, explained, mu] = pca(X [, 'NumComponents', k, 'Centered', tf])`.
pub fn fn_pca(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("pca", ">= 1 (X [, NV...])", 0));
    }
    let x = args[0].to_dense_matrix()?;
    let (n, p) = (x.rows(), x.cols());
    if n == 0 || p == 0 {
        return Err(MatroError::dimension("pca: input matrix cannot be empty"));
    }
    let mut centered_flag = true;
    let mut num_comp: Option<usize> = None;
    let mut i = 1;
    while i + 1 < args.len() {
        if let Some(k) = crate::value_as_text(&args[i]) {
            match k.to_ascii_lowercase().as_str() {
                "centered" => centered_flag = args[i + 1].is_truthy(),
                "numcomponents" => {
                    num_comp = Some(args[i + 1].as_scalar()?.max(1.0).round() as usize);
                }
                _ => {}
            }
        }
        i += 2;
    }

    let mut mu = vec![0.0; p];
    let mut xc = Matrix::zeros(n, p);
    for c in 0..p {
        let mean = if centered_flag {
            (0..n).map(|r| x.get(r, c)).sum::<f64>() / n as f64
        } else {
            0.0
        };
        mu[c] = mean;
        for r in 0..n {
            xc.set(r, c, x.get(r, c) - mean);
        }
    }
    let (_u, s_mat, v_full) = xc.svd()?;
    let k_max = if centered_flag && n > 1 {
        (n - 1).min(p)
    } else {
        n.min(p)
    };
    let k = num_comp.map_or(p.min(v_full.cols()), |nc| nc.min(p).min(v_full.cols()));
    let mut coeff = Matrix::zeros(p, k);
    for c in 0..k {
        for r in 0..p {
            coeff.set(r, c, v_full.get(r, c));
        }
    }
    let score = xc.matmul(&coeff)?;
    let denom = if centered_flag && n > 1 { (n - 1) as f64 } else { n as f64 };
    let latent_all: Vec<f64> = (0..k_max.min(s_mat.rows().min(s_mat.cols())))
        .map(|j| {
            let sv = s_mat.get(j, j);
            (sv * sv) / denom
        })
        .collect();
    let total_var: f64 = latent_all.iter().sum();
    let latent: Vec<f64> = (0..k.min(latent_all.len())).map(|j| latent_all[j]).collect();
    let explained: Vec<f64> = latent
        .iter()
        .map(|&l| if total_var > 0.0 { 100.0 * l / total_var } else { 0.0 })
        .collect();
    let tsquared: Vec<f64> = (0..n)
        .map(|r| {
            let mut sum = 0.0;
            for j in 0..latent.len() {
                if latent[j] > 1e-15 {
                    let s = score.get(r, j);
                    sum += (s * s) / latent[j];
                }
            }
            sum
        })
        .collect();

    let mut out = vec![Value::matrix(coeff)];
    if nargout >= 2 {
        out.push(Value::matrix(score));
    }
    if nargout >= 3 {
        out.push(Value::matrix(Matrix::col_vec(latent)));
    }
    if nargout >= 4 {
        out.push(Value::matrix(Matrix::col_vec(tsquared)));
    }
    if nargout >= 5 {
        out.push(Value::matrix(Matrix::col_vec(explained)));
    }
    if nargout >= 6 {
        out.push(Value::matrix(Matrix::row_vec(mu)));
    }
    Ok(out)
}
