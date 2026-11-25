use std::collections::BTreeMap;

use matro_types::{MatroError, Matrix, ObjectInstance, Result, Value};

fn ok1(v: Value) -> Result<Vec<Value>> {
    Ok(vec![v])
}

/// Sort `(x, y)` pairs by ascending `x`, verifying strictly distinct knots.
fn prepare_1d(x_m: &Matrix, y_m: &Matrix, fname: &str) -> Result<(Vec<f64>, Vec<f64>)> {
    if x_m.len() != y_m.len() || x_m.len() < 2 {
        return Err(MatroError::dimension(format!(
            "{fname}: x and y must have the same length >= 2"
        )));
    }
    let mut pairs: Vec<(f64, f64)> = x_m
        .data()
        .iter()
        .copied()
        .zip(y_m.data().iter().copied())
        .collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    for w in pairs.windows(2) {
        if (w[1].0 - w[0].0).abs() <= 1e-15 {
            return Err(MatroError::runtime(format!(
                "{fname}: sample points x must be unique"
            )));
        }
    }
    let xs = pairs.iter().map(|p| p.0).collect();
    let ys = pairs.iter().map(|p| p.1).collect();
    Ok((xs, ys))
}

fn build_pp_struct(breaks: Vec<f64>, coefs_rows: Vec<[f64; 4]>) -> Value {
    let pieces = coefs_rows.len();
    let mut coefs = Matrix::zeros(pieces, 4);
    for (i, row) in coefs_rows.iter().enumerate() {
        for j in 0..4 {
            coefs.set(i, j, row[j]);
        }
    }
    let mut map = BTreeMap::new();
    map.insert("form".to_string(), Value::String("pp".to_string()));
    map.insert("breaks".to_string(), Value::matrix(Matrix::row_vec(breaks)));
    map.insert("coefs".to_string(), Value::matrix(coefs));
    map.insert("pieces".to_string(), Value::scalar(pieces as f64));
    map.insert("order".to_string(), Value::scalar(4.0));
    map.insert("dim".to_string(), Value::scalar(1.0));
    Value::Struct(map)
}

fn hermite_coefs(x: &[f64], y: &[f64], d: &[f64]) -> Vec<[f64; 4]> {
    let n = x.len();
    let mut coefs = Vec::with_capacity(n - 1);
    for k in 0..(n - 1) {
        let h = x[k + 1] - x[k];
        let delta = (y[k + 1] - y[k]) / h;
        let c0 = y[k];
        let c1 = d[k];
        let c2 = (3.0 * delta - 2.0 * d[k] - d[k + 1]) / h;
        let c3 = (d[k] + d[k + 1] - 2.0 * delta) / (h * h);
        coefs.push([c3, c2, c1, c0]);
    }
    coefs
}

/// Fritsch-Carlson shape-preserving PCHIP slopes (matching MATLAB `pchip`).
fn pchip_slopes(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut h = Vec::with_capacity(n - 1);
    let mut delta = Vec::with_capacity(n - 1);
    for k in 0..(n - 1) {
        let hk = x[k + 1] - x[k];
        h.push(hk);
        delta.push((y[k + 1] - y[k]) / hk);
    }
    if n == 2 {
        return vec![delta[0], delta[0]];
    }
    let mut d = vec![0.0; n];
    for k in 1..(n - 1) {
        if delta[k - 1] * delta[k] > 0.0 {
            let w1 = 2.0 * h[k] + h[k - 1];
            let w2 = h[k] + 2.0 * h[k - 1];
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k]);
        }
    }
    let end_slope = |h0: f64, h1: f64, del0: f64, del1: f64| -> f64 {
        let mut s = ((2.0 * h0 + h1) * del0 - h0 * del1) / (h0 + h1);
        if s.signum() != del0.signum() || del0 == 0.0 {
            s = 0.0;
        } else if del0.signum() != del1.signum() && s.abs() > 3.0 * del0.abs() {
            s = 3.0 * del0;
        }
        s
    };
    d[0] = end_slope(h[0], h[1], delta[0], delta[1]);
    d[n - 1] = end_slope(h[n - 2], h[n - 3], delta[n - 2], delta[n - 3]);
    d
}

/// Solve tridiagonal system `a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = rhs[i]`.
fn solve_tridiag(a: &[f64], b: &[f64], c: &[f64], rhs: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut cp = vec![0.0; n];
    let mut dp = vec![0.0; n];
    cp[0] = c[0] / b[0];
    dp[0] = rhs[0] / b[0];
    for i in 1..n {
        let denom = b[i] - a[i] * cp[i - 1];
        cp[i] = if i + 1 < n { c[i] / denom } else { 0.0 };
        dp[i] = (rhs[i] - a[i] * dp[i - 1]) / denom;
    }
    let mut x = vec![0.0; n];
    x[n - 1] = dp[n - 1];
    for i in (0..(n - 1)).rev() {
        x[i] = dp[i] - cp[i] * x[i + 1];
    }
    x
}

/// Not-a-knot C^2 cubic spline slopes (matching MATLAB `spline`).
fn spline_slopes(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut h = Vec::with_capacity(n - 1);
    let mut delta = Vec::with_capacity(n - 1);
    for k in 0..(n - 1) {
        let hk = x[k + 1] - x[k];
        h.push(hk);
        delta.push((y[k + 1] - y[k]) / hk);
    }
    if n == 2 {
        return vec![delta[0], delta[0]];
    }
    if n == 3 {
        // Unique parabola through 3 points
        let h0 = h[0];
        let h1 = h[1];
        let d0 = ((2.0 * h0 + h1) * delta[0] - h0 * delta[1]) / (h0 + h1);
        let d1 = (h1 * delta[0] + h0 * delta[1]) / (h0 + h1);
        let d2 = ((h0 + 2.0 * h1) * delta[1] - h1 * delta[0]) / (h0 + h1);
        return vec![d0, d1, d2];
    }
    let mut a = vec![0.0; n];
    let mut b = vec![0.0; n];
    let mut c = vec![0.0; n];
    let mut rhs = vec![0.0; n];

    // Left not-a-knot BC
    b[0] = h[1];
    c[0] = h[0] + h[1];
    rhs[0] = ((3.0 * h[0] + 2.0 * h[1]) * h[1] * delta[0] + h[0] * h[0] * delta[1]) / (h[0] + h[1]);

    // Interior C^2 continuity
    for i in 1..(n - 1) {
        a[i] = h[i];
        b[i] = 2.0 * (h[i - 1] + h[i]);
        c[i] = h[i - 1];
        rhs[i] = 3.0 * (h[i] * delta[i - 1] + h[i - 1] * delta[i]);
    }

    // Right not-a-knot BC
    let ha = h[n - 3];
    let hb = h[n - 2];
    a[n - 1] = ha + hb;
    b[n - 1] = ha;
    rhs[n - 1] =
        (hb * hb * delta[n - 3] + (2.0 * ha + 3.0 * hb) * ha * delta[n - 2]) / (ha + hb);

    solve_tridiag(&a, &b, &c, &rhs)
}

fn eval_pp_scalar(breaks: &[f64], coefs: &Matrix, xq: f64) -> f64 {
    let pieces = coefs.rows();
    let order = coefs.cols();
    if pieces == 0 || order == 0 || xq.is_nan() {
        return f64::NAN;
    }
    let idx = if xq <= breaks[0] {
        0
    } else if xq >= breaks[pieces] {
        pieces - 1
    } else {
        match breaks.binary_search_by(|b| b.total_cmp(&xq)) {
            Ok(i) => i.min(pieces - 1),
            Err(i) => i.saturating_sub(1).min(pieces - 1),
        }
    };
    let dx = xq - breaks[idx];
    let mut acc = 0.0;
    for j in 0..order {
        acc = acc * dx + coefs.get(idx, j);
    }
    acc
}

fn eval_pp_matrix(breaks: &[f64], coefs: &Matrix, xq: &Matrix) -> Matrix {
    let mut out = Matrix::zeros(xq.rows(), xq.cols());
    for i in 0..xq.len() {
        out.data_mut()[i] = eval_pp_scalar(breaks, coefs, xq.data()[i]);
    }
    out
}

pub fn fn_pchip(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 && args.len() != 3 {
        return Err(MatroError::arity("pchip", "2 or 3 (x, y [, xq])", args.len()));
    }
    let (xs, ys) = prepare_1d(&args[0].to_dense_matrix()?, &args[1].to_dense_matrix()?, "pchip")?;
    let d = pchip_slopes(&xs, &ys);
    let coefs_rows = hermite_coefs(&xs, &ys, &d);
    let pp = build_pp_struct(xs.clone(), coefs_rows);
    if args.len() == 2 {
        return ok1(pp);
    }
    let Value::Struct(ref map) = pp else { unreachable!() };
    let coefs = map["coefs"].to_dense_matrix()?;
    let xq = args[2].to_dense_matrix()?;
    ok1(Value::matrix(eval_pp_matrix(&xs, &coefs, &xq)))
}

pub fn fn_spline(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 && args.len() != 3 {
        return Err(MatroError::arity("spline", "2 or 3 (x, y [, xq])", args.len()));
    }
    let (xs, ys) = prepare_1d(&args[0].to_dense_matrix()?, &args[1].to_dense_matrix()?, "spline")?;
    let d = spline_slopes(&xs, &ys);
    let coefs_rows = hermite_coefs(&xs, &ys, &d);
    let pp = build_pp_struct(xs.clone(), coefs_rows);
    if args.len() == 2 {
        return ok1(pp);
    }
    let Value::Struct(ref map) = pp else { unreachable!() };
    let coefs = map["coefs"].to_dense_matrix()?;
    let xq = args[2].to_dense_matrix()?;
    ok1(Value::matrix(eval_pp_matrix(&xs, &coefs, &xq)))
}

pub fn fn_mkpp(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("mkpp", "2 (breaks, coefs)", args.len()));
    }
    let breaks = args[0].to_dense_matrix()?;
    let coefs = args[1].to_dense_matrix()?;
    if breaks.len() < 2 || coefs.rows() != breaks.len() - 1 {
        return Err(MatroError::dimension(
            "mkpp: coefs must have length(breaks) - 1 rows",
        ));
    }
    let mut map = BTreeMap::new();
    map.insert("form".to_string(), Value::String("pp".to_string()));
    map.insert("breaks".to_string(), Value::matrix(Matrix::row_vec(breaks.data().to_vec())));
    map.insert("coefs".to_string(), Value::matrix(coefs.clone()));
    map.insert("pieces".to_string(), Value::scalar(coefs.rows() as f64));
    map.insert("order".to_string(), Value::scalar(coefs.cols() as f64));
    map.insert("dim".to_string(), Value::scalar(1.0));
    ok1(Value::Struct(map))
}

pub fn fn_ppval(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("ppval", "2 (pp, xq)", args.len()));
    }
    let Value::Struct(ref map) = args[0] else {
        return Err(MatroError::runtime("ppval: first argument must be a pp struct"));
    };
    let breaks = map
        .get("breaks")
        .ok_or_else(|| MatroError::runtime("ppval: missing 'breaks' field"))?
        .to_dense_matrix()?;
    let coefs = map
        .get("coefs")
        .ok_or_else(|| MatroError::runtime("ppval: missing 'coefs' field"))?
        .to_dense_matrix()?;
    let xq = args[1].to_dense_matrix()?;
    ok1(Value::matrix(eval_pp_matrix(breaks.data(), &coefs, &xq)))
}

pub fn fn_unmkpp(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("unmkpp", "1 (pp)", args.len()));
    }
    let Value::Struct(ref map) = args[0] else {
        return Err(MatroError::runtime("unmkpp: expected pp struct"));
    };
    let breaks = map
        .get("breaks")
        .cloned()
        .ok_or_else(|| MatroError::runtime("unmkpp: missing 'breaks'"))?;
    let coefs = map
        .get("coefs")
        .cloned()
        .ok_or_else(|| MatroError::runtime("unmkpp: missing 'coefs'"))?;
    let pieces = map.get("pieces").cloned().unwrap_or_else(|| Value::scalar(0.0));
    let order = map.get("order").cloned().unwrap_or_else(|| Value::scalar(0.0));
    let dim = map.get("dim").cloned().unwrap_or_else(|| Value::scalar(1.0));
    let all = vec![breaks, coefs, pieces, order, dim];
    Ok(all.into_iter().take(nargout.max(1)).collect())
}

/// Modified Akima (makima) slopes — avoids overshoot while preserving smoothness.
pub fn makima_slopes(xs: &[f64], ys: &[f64]) -> Vec<f64> {
    let n = xs.len();
    if n < 2 {
        return vec![0.0; n];
    }
    if n == 2 {
        let s = (ys[1] - ys[0]) / (xs[1] - xs[0]);
        return vec![s, s];
    }
    let mut m = vec![0.0; n + 3];
    for i in 0..(n - 1) {
        m[i + 2] = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]);
    }
    // Extrapolate 2 ghost slopes on each end per Akima/MATLAB makima
    m[1] = 2.0 * m[2] - m[3];
    m[0] = 2.0 * m[1] - m[2];
    m[n + 1] = 2.0 * m[n] - m[n - 1];
    m[n + 2] = 2.0 * m[n + 1] - m[n];

    let mut d = vec![0.0; n];
    for i in 0..n {
        let dm0 = m[i];
        let dm1 = m[i + 1];
        let dm2 = m[i + 2];
        let dm3 = m[i + 3];
        let w1 = (dm3 - dm2).abs() + 0.5 * (dm3 + dm2).abs();
        let w2 = (dm1 - dm0).abs() + 0.5 * (dm1 + dm0).abs();
        if w1 + w2 == 0.0 {
            d[i] = 0.5 * (dm1 + dm2);
        } else {
            d[i] = (w1 * dm1 + w2 * dm2) / (w1 + w2);
        }
    }
    d
}

pub fn fn_makima(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity("makima", "2 or 3 (x, y [, xq])", args.len()));
    }
    let xm = args[0].to_dense_matrix()?;
    let ym = args[1].to_dense_matrix()?;
    let (xs, ys) = prepare_1d(&xm, &ym, "makima")?;
    let slopes = makima_slopes(&xs, &ys);
    let coefs_rows = hermite_coefs(&xs, &ys, &slopes);
    let mut coefs = Matrix::zeros(coefs_rows.len(), 4);
    for (i, r) in coefs_rows.iter().enumerate() {
        for j in 0..4 {
            coefs.set(i, j, r[j]);
        }
    }
    if args.len() == 2 {
        let mut map = BTreeMap::new();
        map.insert("form".to_string(), Value::String("pp".to_string()));
        map.insert("breaks".to_string(), Value::matrix(Matrix::row_vec(xs)));
        map.insert("coefs".to_string(), Value::matrix(coefs.clone()));
        map.insert("pieces".to_string(), Value::scalar(coefs.rows() as f64));
        map.insert("order".to_string(), Value::scalar(4.0));
        map.insert("dim".to_string(), Value::scalar(1.0));
        return ok1(Value::Struct(map));
    }
    let xq = args[2].to_dense_matrix()?;
    ok1(Value::matrix(eval_pp_matrix(&xs, &coefs, &xq)))
}

#[derive(Clone, Copy)]
enum ExtrapMode {
    Default,
    Extrap,
    Val(f64),
}

/// Full MATLAB `interp1(x, v, xq [, method [, extrap]])` supporting column-wise matrix `v`,
/// implicit `x = 1:size(v, 1)`, `'linear'`, `'nearest'`, `'previous'`, `'next'`, `'pchip'`, `'cubic'`, `'spline'`, `'makima'`.
pub fn fn_interp1(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 5 {
        return Err(MatroError::arity(
            "interp1",
            "2 to 5 (x, y, xq [, method [, extrap]])",
            args.len(),
        ));
    }
    // Detect implicit-x forms: interp1(v, xq), interp1(v, xq, method), interp1(v, xq, method, extrap)
    let implicit_x = args.len() == 2
        || (args.len() >= 3 && crate::value_as_text(&args[2]).is_some());
    let (xm, vm, xq, method_arg, extrap_arg) = if implicit_x {
        let vm = args[0].to_dense_matrix()?;
        let n_pts = if vm.rows() == 1 && vm.cols() > 1 {
            vm.cols()
        } else {
            vm.rows()
        };
        let xm = Matrix::row_vec((1..=n_pts).map(|i| i as f64).collect());
        (xm, vm, args[1].to_dense_matrix()?, args.get(2), args.get(3))
    } else {
        (
            args[0].to_dense_matrix()?,
            args[1].to_dense_matrix()?,
            args[2].to_dense_matrix()?,
            args.get(3),
            args.get(4),
        )
    };

    let method = match method_arg {
        Some(v) => crate::value_as_text(v)
            .ok_or_else(|| MatroError::runtime("interp1: method must be a string"))?
            .to_ascii_lowercase(),
        None => "linear".to_string(),
    };
    let extrap_mode = match extrap_arg {
        None => ExtrapMode::Default,
        Some(v) => {
            if let Some(s) = crate::value_as_text(v) {
                if s.eq_ignore_ascii_case("extrap") {
                    ExtrapMode::Extrap
                } else {
                    return Err(MatroError::runtime(format!(
                        "interp1: unknown extrapolation option '{s}'"
                    )));
                }
            } else {
                ExtrapMode::Val(v.as_scalar()?)
            }
        }
    };
    let do_extrap = match extrap_mode {
        ExtrapMode::Extrap => true,
        ExtrapMode::Default => matches!(method.as_str(), "pchip" | "cubic" | "v5cubic" | "spline" | "makima"),
        ExtrapMode::Val(_) => false,
    };
    let fill_val = match extrap_mode {
        ExtrapMode::Val(v) => v,
        _ => f64::NAN,
    };

    // Matrix V column-wise interpolation when V is 2D with rows == length(x) and cols > 1
    let v_is_vec = vm.ndims() == 2 && (vm.rows() == 1 || vm.cols() == 1);
    if !v_is_vec && vm.rows() == xm.len() {
        let ncols = vm.cols();
        let xq_len = xq.len();
        let mut out = Matrix::zeros(xq_len, ncols);
        let xq_col = Matrix::col_vec(xq.data().to_vec());
        for c in 0..ncols {
            let col_m = Matrix::col_vec((0..vm.rows()).map(|r| vm.get(r, c)).collect());
            let (xs, ys) = prepare_1d(&xm, &col_m, "interp1")?;
            let col_out = eval_1d_method(&xs, &ys, &xq_col, &method, do_extrap, fill_val)?;
            for r in 0..xq_len {
                out.set(r, c, col_out.data()[r]);
            }
        }
        return ok1(Value::matrix(out));
    }

    let (xs, ys) = prepare_1d(&xm, &vm, "interp1")?;
    let out = eval_1d_method(&xs, &ys, &xq, &method, do_extrap, fill_val)?;
    ok1(Value::matrix(out))
}

fn eval_1d_method(
    xs: &[f64],
    ys: &[f64],
    xq: &Matrix,
    method: &str,
    do_extrap: bool,
    fill_val: f64,
) -> Result<Matrix> {
    let n = xs.len();
    let x_min = xs[0];
    let x_max = xs[n - 1];

    match method {
        "pchip" | "cubic" | "v5cubic" | "spline" | "makima" => {
            let slopes = match method {
                "spline" => spline_slopes(xs, ys),
                "makima" => makima_slopes(xs, ys),
                _ => pchip_slopes(xs, ys),
            };
            let coefs_rows = hermite_coefs(xs, ys, &slopes);
            let mut coefs = Matrix::zeros(coefs_rows.len(), 4);
            for (i, r) in coefs_rows.iter().enumerate() {
                for j in 0..4 {
                    coefs.set(i, j, r[j]);
                }
            }
            let mut out = vec![0.0; xq.len()];
            for i in 0..xq.len() {
                let q = xq.data()[i];
                if !do_extrap && (q < x_min || q > x_max) {
                    out[i] = fill_val;
                } else {
                    out[i] = eval_pp_scalar(xs, &coefs, q);
                }
            }
            Matrix::from_nd_col_major(xq.dims(), out)
        }
        "linear" | "nearest" | "previous" | "next" => {
            let mut out = vec![0.0; xq.len()];
            for i in 0..xq.len() {
                let q = xq.data()[i];
                if q.is_nan() {
                    out[i] = f64::NAN;
                    continue;
                }
                if q < x_min || q > x_max {
                    if !do_extrap {
                        out[i] = fill_val;
                        continue;
                    }
                    let val = match method {
                        "linear" => {
                            if q < x_min {
                                let t = (q - xs[0]) / (xs[1] - xs[0]);
                                ys[0] + t * (ys[1] - ys[0])
                            } else {
                                let t = (q - xs[n - 2]) / (xs[n - 1] - xs[n - 2]);
                                ys[n - 2] + t * (ys[n - 1] - ys[n - 2])
                            }
                        }
                        "nearest" => if q < x_min { ys[0] } else { ys[n - 1] },
                        "previous" => if q < x_min { f64::NAN } else { ys[n - 1] },
                        "next" => if q < x_min { ys[0] } else { f64::NAN },
                        _ => unreachable!(),
                    };
                    out[i] = val;
                    continue;
                }
                let k = match xs.binary_search_by(|v| v.total_cmp(&q)) {
                    Ok(idx) => {
                        out[i] = ys[idx];
                        continue;
                    }
                    Err(idx) => idx.saturating_sub(1).min(n - 2),
                };
                let t = (q - xs[k]) / (xs[k + 1] - xs[k]);
                out[i] = match method {
                    "linear" => ys[k] + t * (ys[k + 1] - ys[k]),
                    "nearest" => if t < 0.5 { ys[k] } else { ys[k + 1] },
                    "previous" => ys[k],
                    "next" => ys[k + 1],
                    _ => unreachable!(),
                };
            }
            Matrix::from_nd_col_major(xq.dims(), out)
        }
        other => Err(MatroError::runtime(format!(
            "interp1: unsupported interpolation method '{other}'"
        ))),
    }
}

/// General N-D grid interpolation `interpn(X1, ..., Xn, V, Xq1, ..., Xqn [, method [, extrapval]])`
/// and `interpn(V, Xq1, ..., Xqn [, method [, extrapval]])`.
pub fn fn_interpn(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 {
        return Err(MatroError::arity("interpn", ">= 2", args.len()));
    }
    let mut end = args.len();
    let mut extrapval = f64::NAN;
    let mut method = "linear".to_string();
    if end >= 2 && crate::value_as_text(&args[end - 2]).is_some() {
        method = crate::value_as_text(&args[end - 2]).unwrap().to_ascii_lowercase();
        extrapval = args[end - 1].as_scalar()?;
        end -= 2;
    } else if end >= 1 && crate::value_as_text(&args[end - 1]).is_some() {
        method = crate::value_as_text(&args[end - 1]).unwrap().to_ascii_lowercase();
        end -= 1;
    }
    let num_args = &args[..end];
    let (grids, vm, queries): (Vec<Vec<f64>>, Matrix, Vec<Matrix>) = if num_args.len() >= 3 && (num_args.len() - 1) % 2 == 0 {
        let nd = (num_args.len() - 1) / 2;
        let vm = num_args[nd].to_dense_matrix()?;
        let v_dims = vm.dims();
        let mut grids = Vec::with_capacity(nd);
        for ax in 0..nd {
            let gm = num_args[ax].to_dense_matrix()?;
            if gm.len() == v_dims[ax] {
                grids.push(gm.data().to_vec());
            } else if gm.dims() == v_dims {
                let stride: usize = v_dims[..ax].iter().product();
                grids.push((0..v_dims[ax]).map(|k| gm.data()[k * stride]).collect());
            } else {
                return Err(MatroError::dimension("interpn: grid dimension mismatch"));
            }
        }
        let queries: Vec<Matrix> = num_args[nd + 1..]
            .iter()
            .map(|v| v.to_dense_matrix())
            .collect::<Result<_>>()?;
        (grids, vm, queries)
    } else {
        let vm = num_args[0].to_dense_matrix()?;
        let v_dims = vm.dims();
        let nd = num_args.len() - 1;
        let grids: Vec<Vec<f64>> = (0..nd)
            .map(|ax| (1..=v_dims.get(ax).copied().unwrap_or(1)).map(|i| i as f64).collect())
            .collect();
        let queries: Vec<Matrix> = num_args[1..]
            .iter()
            .map(|v| v.to_dense_matrix())
            .collect::<Result<_>>()?;
        (grids, vm, queries)
    };
    let nd = grids.len();
    let q_dims = queries[0].dims();
    let nq = queries[0].len();
    let v_dims = vm.dims();
    let mut v_strides = vec![1usize; nd];
    for ax in 1..nd {
        v_strides[ax] = v_strides[ax - 1] * v_dims[ax - 1];
    }
    let mut out = vec![extrapval; nq];
    let n_corners = 1usize << nd;
    for qi in 0..nq {
        let mut brackets: Vec<(usize, usize, f64)> = Vec::with_capacity(nd);
        let mut in_bounds = true;
        for ax in 0..nd {
            let g = &grids[ax];
            let qv = queries[ax].data()[qi];
            if !qv.is_finite() || g.len() < 2 {
                in_bounds = false;
                break;
            }
            let asc = g[g.len() - 1] >= g[0];
            let (lo_v, hi_v) = if asc { (g[0], g[g.len() - 1]) } else { (g[g.len() - 1], g[0]) };
            if qv < lo_v || qv > hi_v {
                in_bounds = false;
                break;
            }
            let mut k = 0usize;
            for j in 0..(g.len() - 1) {
                if (asc && qv >= g[j] && qv <= g[j + 1]) || (!asc && qv <= g[j] && qv >= g[j + 1]) {
                    k = j;
                    break;
                }
            }
            let t = if (g[k + 1] - g[k]).abs() < 1e-30 {
                0.0
            } else {
                (qv - g[k]) / (g[k + 1] - g[k])
            };
            brackets.push((k, k + 1, t));
        }
        if !in_bounds {
            out[qi] = extrapval;
            continue;
        }
        if method == "nearest" {
            let mut idx = 0usize;
            for ax in 0..nd {
                let (i0, i1, t) = brackets[ax];
                idx += (if t < 0.5 { i0 } else { i1 }) * v_strides[ax];
            }
            out[qi] = vm.data()[idx];
        } else {
            let mut acc = 0.0;
            for mask in 0..n_corners {
                let mut w = 1.0;
                let mut idx = 0usize;
                for ax in 0..nd {
                    let (i0, i1, t) = brackets[ax];
                    if (mask & (1 << ax)) != 0 {
                        w *= t;
                        idx += i1 * v_strides[ax];
                    } else {
                        w *= 1.0 - t;
                        idx += i0 * v_strides[ax];
                    }
                }
                acc += w * vm.data()[idx];
            }
            out[qi] = acc;
        }
    }
    ok1(Value::matrix(Matrix::from_nd_col_major(q_dims, out)?))
}

/// 3D grid interpolation `interp3(X, Y, Z, V, Xq, Yq, Zq [, method [, extrapval]])`
/// or `interp3(V, Xq, Yq, Zq [, method [, extrapval]])` (meshgrid convention: dims are `[Y, X, Z]`).
pub fn fn_interp3(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 4 {
        return Err(MatroError::arity("interp3", ">= 4", args.len()));
    }
    // Swap X (dim 2) and Y (dim 1) into interpn's ndgrid order [Y, X, Z]
    let mut end = args.len();
    let mut tail: Vec<Value> = Vec::new();
    if end >= 2 && crate::value_as_text(&args[end - 2]).is_some() {
        tail.push(args[end - 2].clone());
        tail.push(args[end - 1].clone());
        end -= 2;
    } else if end >= 1 && crate::value_as_text(&args[end - 1]).is_some() {
        tail.push(args[end - 1].clone());
        end -= 1;
    }
    let mut reordered = if end == 7 {
        vec![
            args[1].clone(),
            args[0].clone(),
            args[2].clone(),
            args[3].clone(),
            args[5].clone(),
            args[4].clone(),
            args[6].clone(),
        ]
    } else if end == 4 {
        vec![
            args[0].clone(),
            args[2].clone(),
            args[1].clone(),
            args[3].clone(),
        ]
    } else {
        return Err(MatroError::arity("interp3", "4..=9", args.len()));
    };
    reordered.extend(tail);
    fn_interpn(&reordered)
}

/// `F = griddedInterpolant(x, v [, method])` or `F = griddedInterpolant({x, y}, V [, method])`
/// or `F = griddedInterpolant(X, Y, V [, method])` (ndgrid convention: `size(V) = [length(x), length(y)]`).
pub fn fn_gridded_interpolant(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 4 {
        return Err(MatroError::arity(
            "griddedInterpolant",
            "2 to 4 (grid, V [, method])",
            args.len(),
        ));
    }
    let mut fields = BTreeMap::new();
    match &args[0] {
        Value::Cell(c) => {
            if c.data.len() == 1 {
                let x = c.data[0].to_dense_matrix()?;
                let v = args[1].to_dense_matrix()?;
                let method = args
                    .get(2)
                    .and_then(|m| m.as_str().ok())
                    .unwrap_or("linear")
                    .to_ascii_lowercase();
                fields.insert("__ndims".to_string(), Value::scalar(1.0));
                fields.insert("__x".to_string(), Value::matrix(x));
                fields.insert("__v".to_string(), Value::matrix(v));
                fields.insert("Method".to_string(), Value::String(method));
            } else if c.data.len() == 2 {
                let x = c.data[0].to_dense_matrix()?;
                let y = c.data[1].to_dense_matrix()?;
                let v = args[1].to_dense_matrix()?;
                if v.rows() != x.len() || v.cols() != y.len() {
                    return Err(MatroError::dimension(
                        "griddedInterpolant: V size must match [length(x), length(y)]",
                    ));
                }
                let method = args
                    .get(2)
                    .and_then(|m| m.as_str().ok())
                    .unwrap_or("linear")
                    .to_ascii_lowercase();
                fields.insert("__ndims".to_string(), Value::scalar(2.0));
                fields.insert("__x".to_string(), Value::matrix(x));
                fields.insert("__y".to_string(), Value::matrix(y));
                fields.insert("__v".to_string(), Value::matrix(v));
                fields.insert("Method".to_string(), Value::String(method));
            } else {
                return Err(MatroError::runtime(
                    "griddedInterpolant: supports 1D and 2D grids",
                ));
            }
        }
        _ => {
            // Check whether 2D `(X, Y, V [, method])` or 1D `(x, v [, method])`
            let is_2d = args.len() >= 3 && matches!(args[2], Value::Matrix(_));
            if is_2d {
                let x_mat = args[0].to_dense_matrix()?;
                let y_mat = args[1].to_dense_matrix()?;
                let v = args[2].to_dense_matrix()?;
                let method = args
                    .get(3)
                    .and_then(|m| m.as_str().ok())
                    .unwrap_or("linear")
                    .to_ascii_lowercase();
                // Extract grid vectors from either ndgrid matrices or 1D vectors
                let x_vec = if x_mat.rows() == v.rows() && x_mat.cols() == v.cols() && x_mat.cols() > 1 {
                    Matrix::col_vec((0..x_mat.rows()).map(|r| x_mat.get(r, 0)).collect())
                } else {
                    x_mat
                };
                let y_vec = if y_mat.rows() == v.rows() && y_mat.cols() == v.cols() && y_mat.rows() > 1 {
                    Matrix::row_vec((0..y_mat.cols()).map(|c| y_mat.get(0, c)).collect())
                } else {
                    y_mat
                };
                if v.rows() != x_vec.len() || v.cols() != y_vec.len() {
                    return Err(MatroError::dimension(
                        "griddedInterpolant: V size must match grid vectors",
                    ));
                }
                fields.insert("__ndims".to_string(), Value::scalar(2.0));
                fields.insert("__x".to_string(), Value::matrix(x_vec));
                fields.insert("__y".to_string(), Value::matrix(y_vec));
                fields.insert("__v".to_string(), Value::matrix(v));
                fields.insert("Method".to_string(), Value::String(method));
            } else {
                let x = args[0].to_dense_matrix()?;
                let v = args[1].to_dense_matrix()?;
                let method = args
                    .get(2)
                    .and_then(|m| m.as_str().ok())
                    .unwrap_or("linear")
                    .to_ascii_lowercase();
                fields.insert("__ndims".to_string(), Value::scalar(1.0));
                fields.insert("__x".to_string(), Value::matrix(x));
                fields.insert("__v".to_string(), Value::matrix(v));
                fields.insert("Method".to_string(), Value::String(method));
            }
        }
    }
    ok1(Value::Object(ObjectInstance {
        class_name: "griddedInterpolant".to_string(),
        fields,
    }))
}

/// Evaluate a `griddedInterpolant` object at `F(xq)` or `F(xq, yq)`.
pub fn eval_gridded_interpolant(o: &ObjectInstance, args: &[Value]) -> Result<Value> {
    let ndims = o
        .fields
        .get("__ndims")
        .and_then(|v| v.as_scalar().ok())
        .unwrap_or(1.0) as usize;
    let method = o
        .fields
        .get("Method")
        .and_then(|v| v.as_str().ok())
        .unwrap_or("linear")
        .to_ascii_lowercase();

    if ndims == 1 {
        if args.len() != 1 {
            return Err(MatroError::arity("griddedInterpolant", "1 (xq)", args.len()));
        }
        let x = o.fields["__x"].to_dense_matrix()?;
        let v = o.fields["__v"].to_dense_matrix()?;
        let (xs, ys) = prepare_1d(&x, &v, "griddedInterpolant")?;
        let xq = args[0].to_dense_matrix()?;
        let out = eval_1d_method(&xs, &ys, &xq, &method, true, f64::NAN)?;
        return Ok(Value::matrix(out));
    }

    if args.len() != 2 {
        return Err(MatroError::arity(
            "griddedInterpolant",
            "2 (xq, yq)",
            args.len(),
        ));
    }
    let xv = o.fields["__x"].to_dense_matrix()?;
    let yv = o.fields["__y"].to_dense_matrix()?;
    let vv = o.fields["__v"].to_dense_matrix()?;
    let xq = args[0].to_dense_matrix()?;
    let yq = args[1].to_dense_matrix()?;
    if xq.rows() != yq.rows() || xq.cols() != yq.cols() {
        return Err(MatroError::dimension(
            "griddedInterpolant: xq and yq must have the same size",
        ));
    }

    let xs = xv.data();
    let ys = yv.data();
    let mut out = Matrix::zeros(xq.rows(), xq.cols());
    for i in 0..xq.len() {
        let px = xq.data()[i];
        let py = yq.data()[i];
        // Interpolate along x for each column of V, then along y (supports linear, nearest, pchip, spline!)
        let mut col_vals = Vec::with_capacity(ys.len());
        let qx_m = Matrix::scalar(px);
        for c in 0..ys.len() {
            let col_y: Vec<f64> = (0..xs.len()).map(|r| vv.get(r, c)).collect();
            let v_at_x = eval_1d_method(xs, &col_y, &qx_m, &method, true, f64::NAN)?.get(0, 0);
            col_vals.push(v_at_x);
        }
        let qy_m = Matrix::scalar(py);
        let val = eval_1d_method(ys, &col_vals, &qy_m, &method, true, f64::NAN)?.get(0, 0);
        out.data_mut()[i] = val;
    }
    Ok(Value::matrix(out))
}
