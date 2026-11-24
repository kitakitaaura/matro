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

/// Full MATLAB `interp1(x, y, xq [, method [, extrap]])` supporting
/// `'linear'`, `'nearest'`, `'previous'`, `'next'`, `'pchip'`, `'cubic'`, `'spline'`.
pub fn fn_interp1(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 5 {
        return Err(MatroError::arity(
            "interp1",
            "2 to 5 (x, y, xq [, method [, extrap]])",
            args.len(),
        ));
    }
    let (xm, ym, xq, method_arg, extrap_arg) = if args.len() == 2 {
        let ym = args[0].to_dense_matrix()?;
        let xm = Matrix::row_vec((1..=ym.len()).map(|i| i as f64).collect());
        (xm, ym, args[1].to_dense_matrix()?, None, None)
    } else {
        (
            args[0].to_dense_matrix()?,
            args[1].to_dense_matrix()?,
            args[2].to_dense_matrix()?,
            args.get(3),
            args.get(4),
        )
    };

    let (xs, ys) = prepare_1d(&xm, &ym, "interp1")?;
    let method = match method_arg {
        Some(v) => v.as_str()?.to_ascii_lowercase(),
        None => "linear".to_string(),
    };
    let extrap_mode: Option<Result<f64>> = match extrap_arg {
        None => None,
        Some(v) => {
            if let Ok(s) = v.as_str() {
                if s.eq_ignore_ascii_case("extrap") {
                    Some(Ok(f64::INFINITY)) // sentinel for extrapolate
                } else {
                    return Err(MatroError::runtime(format!(
                        "interp1: unknown extrapolation option '{s}'"
                    )));
                }
            } else {
                Some(Ok(v.as_scalar()?))
            }
        }
    };
    let do_extrap = matches!(extrap_mode, Some(Ok(v)) if v.is_infinite())
        || (extrap_mode.is_none() && matches!(method.as_str(), "pchip" | "cubic" | "spline"));
    let fill_val = match extrap_mode {
        Some(Ok(v)) if !v.is_infinite() => v,
        _ => f64::NAN,
    };

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
        "pchip" | "cubic" | "spline" => {
            let slopes = if method == "spline" {
                spline_slopes(xs, ys)
            } else {
                pchip_slopes(xs, ys)
            };
            let coefs_rows = hermite_coefs(xs, ys, &slopes);
            let mut coefs = Matrix::zeros(coefs_rows.len(), 4);
            for (i, r) in coefs_rows.iter().enumerate() {
                for j in 0..4 {
                    coefs.set(i, j, r[j]);
                }
            }
            let mut out = Matrix::zeros(xq.rows(), xq.cols());
            for i in 0..xq.len() {
                let q = xq.data()[i];
                if !do_extrap && (q < x_min || q > x_max) {
                    out.data_mut()[i] = fill_val;
                } else {
                    out.data_mut()[i] = eval_pp_scalar(xs, &coefs, q);
                }
            }
            Ok(out)
        }
        "linear" | "nearest" | "previous" | "next" => {
            let mut out = Matrix::zeros(xq.rows(), xq.cols());
            for i in 0..xq.len() {
                let q = xq.data()[i];
                if q.is_nan() {
                    out.data_mut()[i] = f64::NAN;
                    continue;
                }
                if q < x_min || q > x_max {
                    if !do_extrap {
                        out.data_mut()[i] = fill_val;
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
                        "nearest" | "previous" | "next" => {
                            if q < x_min { ys[0] } else { ys[n - 1] }
                        }
                        _ => unreachable!(),
                    };
                    out.data_mut()[i] = val;
                    continue;
                }
                let k = match xs.binary_search_by(|v| v.total_cmp(&q)) {
                    Ok(idx) => {
                        out.data_mut()[i] = ys[idx];
                        continue;
                    }
                    Err(idx) => idx.saturating_sub(1).min(n - 2),
                };
                let t = (q - xs[k]) / (xs[k + 1] - xs[k]);
                out.data_mut()[i] = match method {
                    "linear" => ys[k] + t * (ys[k + 1] - ys[k]),
                    "nearest" => if t < 0.5 { ys[k] } else { ys[k + 1] },
                    "previous" => ys[k],
                    "next" => ys[k + 1],
                    _ => unreachable!(),
                };
            }
            Ok(out)
        }
        other => Err(MatroError::runtime(format!(
            "interp1: unsupported interpolation method '{other}'"
        ))),
    }
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
