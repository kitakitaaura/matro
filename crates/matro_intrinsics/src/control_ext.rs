//! Control System Toolbox (pure zero-dependency Rust):
//! - State-space & conversions: `ss`, `tf2ss`, `ss2tf`, `feedback`, `series`, `parallel`
//! - Time & frequency analysis: `lsim`, `impulse`, `step`, `bode`, `nyquist`, `pzmap`,
//!   `pole`, `zero`, `dcgain`, `margin`
//! - Modern control & estimation: `ctrb`, `obsv`, `place`, `acker`, `lyap`, `care`, `lqr`, `lqe`

use std::collections::BTreeMap;
use std::f64::consts::PI;

use matro_types::{Complex, MatroError, Matrix, ObjectInstance, PlotFigure, PlotKind, Result, Value};

fn ok1(v: Value) -> Result<Vec<Value>> {
    Ok(vec![v])
}

#[derive(Clone)]
pub struct SsModel {
    pub a: Matrix,
    pub b: Matrix,
    pub c: Matrix,
    pub d: Matrix,
}

fn make_ss_obj(m: SsModel) -> Value {
    let mut fields = BTreeMap::new();
    fields.insert("A".to_string(), Value::matrix(m.a));
    fields.insert("B".to_string(), Value::matrix(m.b));
    fields.insert("C".to_string(), Value::matrix(m.c));
    fields.insert("D".to_string(), Value::matrix(m.d));
    Value::Object(ObjectInstance {
        class_name: "ss".to_string(),
        fields,
    })
}

/// Convert SISO transfer function `num/den` to controllable canonical state-space `(A, B, C, D)`.
pub fn tf_to_ss_core(num_in: &[f64], den_in: &[f64]) -> Result<SsModel> {
    // Strip leading zeros in den
    let d_start = den_in.iter().position(|&v| v.abs() > 1e-15).unwrap_or(den_in.len());
    let den = &den_in[d_start..];
    if den.is_empty() {
        return Err(MatroError::runtime("tf2ss: denominator cannot be zero"));
    }
    let lead = den[0];
    let den_n: Vec<f64> = den.iter().map(|&v| v / lead).collect();
    let n = den_n.len() - 1;

    // Pad numerator on left to length n + 1
    let mut num_n = vec![0.0; n + 1];
    let n_start = num_in.iter().position(|&v| v.abs() > 1e-15).unwrap_or(num_in.len());
    let num_trim = &num_in[n_start..];
    if num_trim.len() > n + 1 {
        return Err(MatroError::runtime("tf2ss: transfer function must be proper (deg(num) <= deg(den))"));
    }
    let offset = (n + 1) - num_trim.len();
    for (i, &v) in num_trim.iter().enumerate() {
        num_n[offset + i] = v / lead;
    }

    let d0 = num_n[0];
    if n == 0 {
        return Ok(SsModel {
            a: Matrix::zeros(0, 0),
            b: Matrix::zeros(0, 1),
            c: Matrix::zeros(1, 0),
            d: Matrix::scalar(d0),
        });
    }

    // Controllable canonical form (companion matrix)
    let mut a = Matrix::zeros(n, n);
    for i in 0..(n - 1) {
        a.set(i, i + 1, 1.0);
    }
    for j in 0..n {
        a.set(n - 1, j, -den_n[n - j]);
    }
    let mut b = Matrix::zeros(n, 1);
    b.set(n - 1, 0, 1.0);

    let mut c = Matrix::zeros(1, n);
    for j in 0..n {
        c.set(0, j, num_n[n - j] - d0 * den_n[n - j]);
    }
    Ok(SsModel {
        a,
        b,
        c,
        d: Matrix::scalar(d0),
    })
}

/// Extract `(A, B, C, D)` from either an `ss` object, a `tf` struct, or a scalar gain.
pub fn extract_ss(val: &Value) -> Result<SsModel> {
    match val {
        Value::Object(o) if o.class_name == "ss" => Ok(SsModel {
            a: o.fields["A"].to_dense_matrix()?,
            b: o.fields["B"].to_dense_matrix()?,
            c: o.fields["C"].to_dense_matrix()?,
            d: o.fields["D"].to_dense_matrix()?,
        }),
        Value::Object(o) if o.class_name == "tf" => {
            let num = o.fields["Numerator"].to_dense_matrix()?;
            let den = o.fields["Denominator"].to_dense_matrix()?;
            tf_to_ss_core(num.data(), den.data())
        }
        Value::Struct(map) if map.contains_key("num") && map.contains_key("den") => {
            let num = map["num"].to_dense_matrix()?;
            let den = map["den"].to_dense_matrix()?;
            tf_to_ss_core(num.data(), den.data())
        }
        Value::Matrix(m) if m.is_scalar() => Ok(SsModel {
            a: Matrix::zeros(0, 0),
            b: Matrix::zeros(0, 1),
            c: Matrix::zeros(1, 0),
            d: Matrix::scalar(m.data()[0]),
        }),
        other => Err(MatroError::runtime(format!(
            "Expected tf or ss system model, got {}",
            other.type_name()
        ))),
    }
}

/// Evaluate frequency response `G(s) = C * (sI - A)^{-1} * B + D` for SISO or first channel.
fn eval_ss_freq(sys: &SsModel, s: Complex) -> Result<Complex> {
    let n = sys.a.rows();
    if n == 0 {
        return Ok(Complex::from_real(sys.d.get(0, 0)));
    }
    let mut si_a = Matrix::zeros(n, n);
    for r in 0..n {
        for c in 0..n {
            let v = if r == c {
                s - Complex::from_real(sys.a.get(r, c))
            } else {
                Complex::from_real(-sys.a.get(r, c))
            };
            si_a.set_c(r, c, v);
        }
    }
    let b_col = Matrix::col_vec((0..n).map(|r| sys.b.get(r, 0)).collect());
    let x = si_a.solve(&b_col)?;
    let mut g = Complex::from_real(sys.d.get(0, 0));
    for i in 0..n {
        g = g + Complex::from_real(sys.c.get(0, i)) * x.get_c(i, 0);
    }
    Ok(g)
}

/// Characteristic polynomial `det(sI - A)` via Faddeev-LeVerrier (`1 x (n+1)`).
fn char_poly(a: &Matrix) -> Result<Vec<f64>> {
    let n = a.rows();
    if n == 0 {
        return Ok(vec![1.0]);
    }
    let mut coefs = vec![0.0; n + 1];
    coefs[0] = 1.0;
    let mut m = Matrix::zeros(n, n);
    let eye = Matrix::eye(n, n);
    for k in 1..=n {
        let term = if k == 1 {
            a.clone()
        } else {
            let am = a.matmul(&m)?;
            let mut sum = Matrix::zeros(n, n);
            for r in 0..n {
                for c in 0..n {
                    sum.set(r, c, am.get(r, c) + coefs[k - 1] * eye.get(r, c));
                }
            }
            sum
        };
        let am_k = a.matmul(&term)?;
        let mut tr = 0.0;
        for i in 0..n {
            tr += if k == 1 { a.get(i, i) } else { am_k.get(i, i) };
        }
        // Standard Faddeev-LeVerrier: M_k = A*M_{k-1} + c_{k-1}*I, c_k = -tr(A*M_k)/k
        if k == 1 {
            m = eye.clone();
            let am1 = a.matmul(&m)?;
            let tr1: f64 = (0..n).map(|i| am1.get(i, i)).sum();
            coefs[1] = -tr1;
        } else {
            m = term;
            coefs[k] = -tr / (k as f64);
        }
    }
    Ok(coefs)
}

/// `sys = ss(A, B, C, D)`
pub fn fn_ss(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 4 {
        return Err(MatroError::arity("ss", "4 (A, B, C, D)", args.len()));
    }
    let a = args[0].to_dense_matrix()?;
    let b = args[1].to_dense_matrix()?;
    let c = args[2].to_dense_matrix()?;
    let d = args[3].to_dense_matrix()?;
    if a.rows() != a.cols() {
        return Err(MatroError::dimension("ss: A must be square"));
    }
    ok1(make_ss_obj(SsModel { a, b, c, d }))
}

/// `[A, B, C, D] = tf2ss(num, den)`
pub fn fn_tf2ss(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("tf2ss", "2 (num, den)", args.len()));
    }
    let num = args[0].to_dense_matrix()?;
    let den = args[1].to_dense_matrix()?;
    let sys = tf_to_ss_core(num.data(), den.data())?;
    Ok(vec![
        Value::matrix(sys.a),
        Value::matrix(sys.b),
        Value::matrix(sys.c),
        Value::matrix(sys.d),
    ])
}

/// `[num, den] = ss2tf(A, B, C, D)` (SISO Faddeev-LeVerrier: `num = poly(A - B*C) + (D - 1)*poly(A)`).
pub fn fn_ss2tf(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 4 {
        return Err(MatroError::arity("ss2tf", "4 (A, B, C, D)", args.len()));
    }
    let a = args[0].to_dense_matrix()?;
    let b = args[1].to_dense_matrix()?;
    let c = args[2].to_dense_matrix()?;
    let d = args[3].to_dense_matrix()?;
    let den = char_poly(&a)?;
    let bc = b.matmul(&c)?;
    let a_minus_bc = a.elementwise(&bc, |x, y| x - y)?;
    let p_shifted = char_poly(&a_minus_bc)?;
    let d0 = d.get(0, 0);
    // Identity: det(sI - A + B*C) = det(sI - A) * (1 + C*(sI - A)^{-1}*B)
    // => num(s) = det(sI - A + B*C) + (D - 1)*det(sI - A)
    // Wait: det(sI - (A - B*C)) = det(sI - A + B*C) = det(sI - A)*(1 + C*(sI-A)^{-1}*B)
    let num: Vec<f64> = p_shifted
        .iter()
        .zip(den.iter())
        .map(|(&ps, &pd)| ps + (d0 - 1.0) * pd)
        .collect();
    Ok(vec![
        Value::matrix(Matrix::row_vec(num)),
        Value::matrix(Matrix::row_vec(den)),
    ])
}

/// `sys = series(sys1, sys2)` — Cascade connection `G2(s) * G1(s)`.
pub fn fn_series(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("series", "2 (sys1, sys2)", args.len()));
    }
    let s1 = extract_ss(&args[0])?;
    let s2 = extract_ss(&args[1])?;
    let n1 = s1.a.rows();
    let n2 = s2.a.rows();
    let n = n1 + n2;
    let m1 = s1.b.cols();
    let p2 = s2.c.rows();

    let mut a = Matrix::zeros(n, n);
    let mut b = Matrix::zeros(n, m1);
    let mut c = Matrix::zeros(p2, n);
    for r in 0..n1 {
        for col in 0..n1 {
            a.set(r, col, s1.a.get(r, col));
        }
        for col in 0..m1 {
            b.set(r, col, s1.b.get(r, col));
        }
    }
    if n2 > 0 {
        let b2c1 = if n1 > 0 { s2.b.matmul(&s1.c)? } else { Matrix::zeros(n2, 0) };
        let b2d1 = s2.b.matmul(&s1.d)?;
        for r in 0..n2 {
            for col in 0..n1 {
                a.set(n1 + r, col, b2c1.get(r, col));
            }
            for col in 0..n2 {
                a.set(n1 + r, n1 + col, s2.a.get(r, col));
            }
            for col in 0..m1 {
                b.set(n1 + r, col, b2d1.get(r, col));
            }
        }
    }
    let d2c1 = if n1 > 0 { s2.d.matmul(&s1.c)? } else { Matrix::zeros(p2, 0) };
    for r in 0..p2 {
        for col in 0..n1 {
            c.set(r, col, d2c1.get(r, col));
        }
        for col in 0..n2 {
            c.set(r, n1 + col, s2.c.get(r, col));
        }
    }
    let d = s2.d.matmul(&s1.d)?;
    ok1(make_ss_obj(SsModel { a, b, c, d }))
}

/// `sys = parallel(sys1, sys2)` — Parallel sum `G1(s) + G2(s)`.
pub fn fn_parallel(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("parallel", "2 (sys1, sys2)", args.len()));
    }
    let s1 = extract_ss(&args[0])?;
    let s2 = extract_ss(&args[1])?;
    let n1 = s1.a.rows();
    let n2 = s2.a.rows();
    let n = n1 + n2;
    let m = s1.b.cols();
    let p = s1.c.rows();
    let mut a = Matrix::zeros(n, n);
    let mut b = Matrix::zeros(n, m);
    let mut c = Matrix::zeros(p, n);
    for r in 0..n1 {
        for col in 0..n1 {
            a.set(r, col, s1.a.get(r, col));
        }
        for col in 0..m {
            b.set(r, col, s1.b.get(r, col));
        }
        for row in 0..p {
            c.set(row, r, s1.c.get(row, r));
        }
    }
    for r in 0..n2 {
        for col in 0..n2 {
            a.set(n1 + r, n1 + col, s2.a.get(r, col));
        }
        for col in 0..m {
            b.set(n1 + r, col, s2.b.get(r, col));
        }
        for row in 0..p {
            c.set(row, n1 + r, s2.c.get(row, r));
        }
    }
    let d = s1.d.elementwise(&s2.d, |x, y| x + y)?;
    ok1(make_ss_obj(SsModel { a, b, c, d }))
}

/// `sys = feedback(sys1, sys2 [, sign])` — Feedback interconnection (`sign = -1` negative by default).
pub fn fn_feedback(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity("feedback", "2 or 3 (sys1, sys2 [, sign])", args.len()));
    }
    let s1 = extract_ss(&args[0])?;
    let s2 = extract_ss(&args[1])?;
    let sign = args.get(2).map(|v| v.as_scalar()).transpose()?.unwrap_or(-1.0);
    // Standard state-space feedback for SISO/MIMO with algebraic loop resolution E = (I - sign*D2*D1)^{-1}
    let n1 = s1.a.rows();
    let n2 = s2.a.rows();
    let n = n1 + n2;
    let m = s1.b.cols();
    let p = s1.c.rows();

    let d2d1 = s2.d.matmul(&s1.d)?;
    let mut eye_m = Matrix::eye(m, m);
    for r in 0..m {
        for c in 0..m {
            eye_m.set(r, c, eye_m.get(r, c) - sign * d2d1.get(r, c));
        }
    }
    let e_inv = eye_m.inv()?; // (I - sign*D2*D1)^{-1}

    let d1d2 = s1.d.matmul(&s2.d)?;
    let mut eye_p = Matrix::eye(p, p);
    for r in 0..p {
        for c in 0..p {
            eye_p.set(r, c, eye_p.get(r, c) - sign * d1d2.get(r, c));
        }
    }
    let f_inv = eye_p.inv()?; // (I - sign*D1*D2)^{-1}

    // u1 = E^{-1}*(r + sign*D2*C1*x1 + sign*C2*x2)
    // y  = F^{-1}*(C1*x1 + sign*D1*C2*x2 + D1*r)
    let mut a = Matrix::zeros(n, n);
    let mut b = Matrix::zeros(n, m);
    let mut c = Matrix::zeros(p, n);
    let d = f_inv.matmul(&s1.d)?;

    let b1e = if n1 > 0 { s1.b.matmul(&e_inv)? } else { Matrix::zeros(0, m) };
    let b2f = if n2 > 0 { s2.b.matmul(&f_inv)? } else { Matrix::zeros(0, p) };

    if n1 > 0 {
        let a11 = if n1 > 0 {
            let term = b1e.matmul(&s2.d)?.matmul(&s1.c)?;
            s1.a.elementwise(&term, |x, y| x + sign * y)?
        } else {
            Matrix::zeros(0, 0)
        };
        for r in 0..n1 {
            for col in 0..n1 {
                a.set(r, col, a11.get(r, col));
            }
            for col in 0..m {
                b.set(r, col, b1e.get(r, col));
            }
        }
        let c_left = f_inv.matmul(&s1.c)?;
        for r in 0..p {
            for col in 0..n1 {
                c.set(r, col, c_left.get(r, col));
            }
        }
    }
    if n1 > 0 && n2 > 0 {
        let a12 = b1e.matmul(&s2.c)?.map(|v| sign * v);
        let a21 = b2f.matmul(&s1.c)?;
        for r in 0..n1 {
            for col in 0..n2 {
                a.set(r, n1 + col, a12.get(r, col));
            }
        }
        for r in 0..n2 {
            for col in 0..n1 {
                a.set(n1 + r, col, a21.get(r, col));
            }
        }
    }
    if n2 > 0 {
        let a22_term = b2f.matmul(&s1.d)?.matmul(&s2.c)?;
        let a22 = s2.a.elementwise(&a22_term, |x, y| x + sign * y)?;
        let b2_cl = b2f.matmul(&s1.d)?;
        let c_right = f_inv.matmul(&s1.d)?.matmul(&s2.c)?.map(|v| sign * v);
        for r in 0..n2 {
            for col in 0..n2 {
                a.set(n1 + r, n1 + col, a22.get(r, col));
            }
            for col in 0..m {
                b.set(n1 + r, col, b2_cl.get(r, col));
            }
        }
        for r in 0..p {
            for col in 0..n2 {
                c.set(r, n1 + col, c_right.get(r, col));
            }
        }
    }
    ok1(make_ss_obj(SsModel { a, b, c, d }))
}

/// RK4 state-space simulator for arbitrary input `u(t)`: `[y, t, x] = lsim(sys, u, t [, x0])`.
pub fn simulate_ss(
    sys: &SsModel,
    u_mat: &Matrix,
    t_vec: &[f64],
    x0: &[f64],
) -> Result<(Matrix, Matrix)> {
    let n = sys.a.rows();
    let m = sys.b.cols();
    let p = sys.c.rows();
    let nt = t_vec.len();
    let mut x = x0.to_vec();
    x.resize(n, 0.0);

    let mut y_out = Matrix::zeros(nt, p);
    let mut x_out = Matrix::zeros(nt, n);

    let u_at = |idx: usize| -> Vec<f64> {
        if u_mat.cols() == m && u_mat.rows() == nt {
            (0..m).map(|c| u_mat.get(idx, c)).collect()
        } else {
            vec![u_mat.data()[idx.min(u_mat.len() - 1)]; m]
        }
    };

    let deriv = |xv: &[f64], uv: &[f64]| -> Vec<f64> {
        let mut dx = vec![0.0; n];
        for r in 0..n {
            let mut s = 0.0;
            for c in 0..n {
                s += sys.a.get(r, c) * xv[c];
            }
            for c in 0..m {
                s += sys.b.get(r, c) * uv[c];
            }
            dx[r] = s;
        }
        dx
    };

    let output = |xv: &[f64], uv: &[f64]| -> Vec<f64> {
        let mut yv = vec![0.0; p];
        for r in 0..p {
            let mut s = 0.0;
            for c in 0..n {
                s += sys.c.get(r, c) * xv[c];
            }
            for c in 0..m {
                s += sys.d.get(r, c) * uv[c];
            }
            yv[r] = s;
        }
        yv
    };

    for ti in 0..nt {
        let u0 = u_at(ti);
        let yv = output(&x, &u0);
        for r in 0..p {
            y_out.set(ti, r, yv[r]);
        }
        for r in 0..n {
            x_out.set(ti, r, x[r]);
        }
        if ti + 1 < nt {
            let dt = t_vec[ti + 1] - t_vec[ti];
            let u1 = u_at(ti + 1);
            let um: Vec<f64> = u0.iter().zip(u1.iter()).map(|(a, b)| 0.5 * (a + b)).collect();
            let sub = ((dt.abs() / 0.05).ceil() as usize).clamp(4, 200);
            let h = dt / (sub as f64);
            for _ in 0..sub {
                let k1 = deriv(&x, &um);
                let x2: Vec<f64> = (0..n).map(|i| x[i] + 0.5 * h * k1[i]).collect();
                let k2 = deriv(&x2, &um);
                let x3: Vec<f64> = (0..n).map(|i| x[i] + 0.5 * h * k2[i]).collect();
                let k3 = deriv(&x3, &um);
                let x4: Vec<f64> = (0..n).map(|i| x[i] + h * k3[i]).collect();
                let k4 = deriv(&x4, &um);
                for i in 0..n {
                    x[i] += (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
                }
            }
        }
    }
    Ok((y_out, x_out))
}

fn default_time_grid(sys: &SsModel) -> Vec<f64> {
    let mut t_end = 10.0f64;
    if sys.a.rows() > 0 {
        if let Ok((_, d)) = sys.a.eig() {
            let mut slow = f64::INFINITY;
            for i in 0..d.rows() {
                let re = d.get_c(i, i).re.abs();
                if re > 1e-4 && re < slow {
                    slow = re;
                }
            }
            if slow.is_finite() {
                t_end = (7.0 / slow).clamp(1.0, 50.0);
            }
        }
    }
    let npts = 200usize;
    (0..npts).map(|i| t_end * (i as f64) / ((npts - 1) as f64)).collect()
}

/// `[y, t, x] = lsim(sys, u, t [, x0])`
pub fn fn_lsim(args: &[Value], nargout: usize, fig: &mut PlotFigure) -> Result<Vec<Value>> {
    if args.len() < 3 || args.len() > 4 {
        return Err(MatroError::arity("lsim", "3 or 4 (sys, u, t [, x0])", args.len()));
    }
    let sys = extract_ss(&args[0])?;
    let u = args[1].to_dense_matrix()?;
    let t = args[2].to_dense_matrix()?;
    let x0 = if let Some(v) = args.get(3) {
        v.to_dense_matrix()?.data().to_vec()
    } else {
        vec![0.0; sys.a.rows()]
    };
    let (y, x) = simulate_ss(&sys, &u, t.data(), &x0)?;
    if nargout == 0 {
        let p = fig.panel_mut();
        p.clear();
        p.add_series(
            PlotKind::Line,
            t.data().to_vec(),
            (0..t.len()).map(|r| y.get(r, 0)).collect(),
            None,
        );
        p.title = "Linear Simulation Response (lsim)".to_string();
        p.xlabel = "Time (s)".to_string();
        p.ylabel = "Amplitude".to_string();
        return Ok(vec![]);
    }
    let mut out = vec![Value::matrix(y)];
    if nargout >= 2 {
        out.push(Value::matrix(Matrix::col_vec(t.data().to_vec())));
    }
    if nargout >= 3 {
        out.push(Value::matrix(x));
    }
    Ok(out)
}

/// `[y, t] = impulse(sys [, t])` — Impulse response (`x(0+) = B`, `u(t) = 0`).
pub fn fn_impulse(args: &[Value], nargout: usize, fig: &mut PlotFigure) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("impulse", "1 or 2 (sys [, t])", args.len()));
    }
    let sys = extract_ss(&args[0])?;
    let t_vec = if let Some(tv) = args.get(1) {
        let m = tv.to_dense_matrix()?;
        if m.len() == 1 {
            let tend = m.data()[0].max(1e-3);
            (0..200).map(|i| tend * (i as f64) / 199.0).collect()
        } else {
            m.data().to_vec()
        }
    } else {
        default_time_grid(&sys)
    };
    let n = sys.a.rows();
    let x0: Vec<f64> = (0..n).map(|r| sys.b.get(r, 0)).collect();
    let u_zero = Matrix::zeros(t_vec.len(), sys.b.cols().max(1));
    let (y, _) = simulate_ss(&sys, &u_zero, &t_vec, &x0)?;
    if nargout == 0 {
        let p = fig.panel_mut();
        p.clear();
        p.add_series(
            PlotKind::Line,
            t_vec.clone(),
            (0..t_vec.len()).map(|r| y.get(r, 0)).collect(),
            None,
        );
        p.title = "Impulse Response".to_string();
        p.xlabel = "Time (s)".to_string();
        p.ylabel = "Amplitude".to_string();
        return Ok(vec![]);
    }
    let mut out = vec![Value::matrix(y)];
    if nargout >= 2 {
        out.push(Value::matrix(Matrix::col_vec(t_vec)));
    }
    Ok(out)
}

/// Unified `[y, t] = step(sys [, t])` supporting BOTH `tf` and `ss`!
pub fn fn_step_unified(args: &[Value], nargout: usize, fig: &mut PlotFigure) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("step", "1 or 2 (sys [, t])", args.len()));
    }
    let sys = extract_ss(&args[0])?;
    let t_vec = if let Some(tv) = args.get(1) {
        let m = tv.to_dense_matrix()?;
        if m.len() == 1 {
            let tend = m.data()[0].max(1e-3);
            (0..200).map(|i| tend * (i as f64) / 199.0).collect()
        } else {
            m.data().to_vec()
        }
    } else {
        default_time_grid(&sys)
    };
    let u_ones = Matrix::ones(t_vec.len(), sys.b.cols().max(1));
    let x0 = vec![0.0; sys.a.rows()];
    let (y, _) = simulate_ss(&sys, &u_ones, &t_vec, &x0)?;
    if nargout == 0 {
        let p = fig.panel_mut();
        p.clear();
        p.add_series(
            PlotKind::Line,
            t_vec.clone(),
            (0..t_vec.len()).map(|r| y.get(r, 0)).collect(),
            None,
        );
        p.title = "Step Response".to_string();
        p.xlabel = "Time (s)".to_string();
        p.ylabel = "Amplitude".to_string();
        return Ok(vec![]);
    }
    let mut out = vec![Value::matrix(y)];
    if nargout >= 2 {
        out.push(Value::matrix(Matrix::col_vec(t_vec)));
    }
    Ok(out)
}

/// Unified `[mag, phase, w] = bode(sys [, w])` supporting BOTH `tf` and `ss`!
pub fn fn_bode_unified(args: &[Value], nargout: usize, fig: &mut PlotFigure) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("bode", "1 or 2 (sys [, w])", args.len()));
    }
    let sys = extract_ss(&args[0])?;
    let w_vec: Vec<f64> = if let Some(wv) = args.get(1) {
        wv.to_dense_matrix()?.data().to_vec()
    } else {
        (0..120).map(|i| 10.0f64.powf(-1.0 + 4.0 * (i as f64) / 119.0)).collect()
    };
    let mut mag = Vec::with_capacity(w_vec.len());
    let mut phase = Vec::with_capacity(w_vec.len());
    for &w in &w_vec {
        let g = eval_ss_freq(&sys, Complex::new(0.0, w))?;
        mag.push(g.abs());
        phase.push(g.im.atan2(g.re) * 180.0 / PI);
    }
    // Unwrap phase in degrees
    for i in 1..phase.len() {
        while phase[i] - phase[i - 1] > 180.0 {
            phase[i] -= 360.0;
        }
        while phase[i] - phase[i - 1] < -180.0 {
            phase[i] += 360.0;
        }
    }
    if nargout == 0 {
        let mag_db: Vec<f64> = mag.iter().map(|&v| 20.0 * v.max(1e-12).log10()).collect();
        fig.subplot(2, 1, 1);
        {
            let p = fig.panel_mut();
            p.log_x = true;
            p.add_series(PlotKind::Line, w_vec.clone(), mag_db, None);
            p.title = "Bode Diagram".to_string();
            p.ylabel = "Magnitude (dB)".to_string();
        }
        fig.subplot(2, 1, 2);
        {
            let p = fig.panel_mut();
            p.log_x = true;
            p.add_series(PlotKind::Line, w_vec.clone(), phase.clone(), None);
            p.xlabel = "Frequency (rad/s)".to_string();
            p.ylabel = "Phase (deg)".to_string();
        }
        return Ok(vec![]);
    }
    let mut out = vec![Value::matrix(Matrix::col_vec(mag))];
    if nargout >= 2 {
        out.push(Value::matrix(Matrix::col_vec(phase)));
    }
    if nargout >= 3 {
        out.push(Value::matrix(Matrix::col_vec(w_vec)));
    }
    Ok(out)
}

/// `[re, im, w] = nyquist(sys [, w])`
pub fn fn_nyquist(args: &[Value], nargout: usize, fig: &mut PlotFigure) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("nyquist", "1 or 2 (sys [, w])", args.len()));
    }
    let sys = extract_ss(&args[0])?;
    let w_vec: Vec<f64> = if let Some(wv) = args.get(1) {
        wv.to_dense_matrix()?.data().to_vec()
    } else {
        (0..150).map(|i| 10.0f64.powf(-2.0 + 5.0 * (i as f64) / 149.0)).collect()
    };
    let mut re = Vec::with_capacity(w_vec.len());
    let mut im = Vec::with_capacity(w_vec.len());
    for &w in &w_vec {
        let g = eval_ss_freq(&sys, Complex::new(0.0, w))?;
        re.push(g.re);
        im.push(g.im);
    }
    if nargout == 0 {
        let p = fig.panel_mut();
        p.clear();
        p.add_series(PlotKind::Line, re.clone(), im.clone(), None);
        p.title = "Nyquist Diagram".to_string();
        p.xlabel = "Real Axis".to_string();
        p.ylabel = "Imaginary Axis".to_string();
        return Ok(vec![]);
    }
    let mut out = vec![Value::matrix(Matrix::col_vec(re))];
    if nargout >= 2 {
        out.push(Value::matrix(Matrix::col_vec(im)));
    }
    if nargout >= 3 {
        out.push(Value::matrix(Matrix::col_vec(w_vec)));
    }
    Ok(out)
}

pub fn fn_pole(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("pole", "1 (sys)", args.len()));
    }
    let sys = extract_ss(&args[0])?;
    if sys.a.rows() == 0 {
        return ok1(Value::empty());
    }
    let (_, d) = sys.a.eig()?;
    let poles: Vec<Complex> = (0..d.rows()).map(|i| d.get_c(i, i)).collect();
    ok1(Value::matrix(Matrix::complex_col_vec(poles).normalize()))
}

pub fn fn_zero(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("zero", "1 (sys)", args.len()));
    }
    let sys = extract_ss(&args[0])?;
    let res = fn_ss2tf(&[
        Value::matrix(sys.a),
        Value::matrix(sys.b),
        Value::matrix(sys.c),
        Value::matrix(sys.d),
    ])?;
    let num = res[0].to_dense_matrix()?;
    // Trim leading zeros of num and compute companion-matrix eigenvalues
    let start = num.data().iter().position(|&v| v.abs() > 1e-12).unwrap_or(num.len());
    let trim = &num.data()[start..];
    if trim.len() <= 1 {
        return ok1(Value::empty());
    }
    let deg = trim.len() - 1;
    let lead = trim[0];
    let mut comp = Matrix::zeros(deg, deg);
    for c in 0..deg {
        comp.set(0, c, -trim[c + 1] / lead);
    }
    for r in 1..deg {
        comp.set(r, r - 1, 1.0);
    }
    let (_, d) = comp.eig()?;
    let zeros: Vec<Complex> = (0..deg).map(|i| d.get_c(i, i)).collect();
    ok1(Value::matrix(Matrix::complex_col_vec(zeros).normalize()))
}

pub fn fn_dcgain(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("dcgain", "1 (sys)", args.len()));
    }
    let sys = extract_ss(&args[0])?;
    let g0 = eval_ss_freq(&sys, Complex::ZERO)?;
    ok1(Value::scalar(g0.re))
}

pub fn fn_pzmap(args: &[Value], fig: &mut PlotFigure) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("pzmap", "1 (sys)", args.len()));
    }
    let poles = fn_pole(args)?[0].to_dense_matrix()?;
    let zeros = fn_zero(args)?[0].to_dense_matrix()?;
    let p = fig.panel_mut();
    p.clear();
    p.hold = true;
    if !poles.is_empty() {
        let px: Vec<f64> = (0..poles.len()).map(|i| poles.get_linear_c(i).re).collect();
        let py: Vec<f64> = (0..poles.len()).map(|i| poles.get_linear_c(i).im).collect();
        p.add_series(PlotKind::Scatter, px, py, Some("r"));
    }
    if !zeros.is_empty() {
        let zx: Vec<f64> = (0..zeros.len()).map(|i| zeros.get_linear_c(i).re).collect();
        let zy: Vec<f64> = (0..zeros.len()).map(|i| zeros.get_linear_c(i).im).collect();
        p.add_series(PlotKind::Scatter, zx, zy, Some("b"));
    }
    p.title = "Pole-Zero Map".to_string();
    p.xlabel = "Real Axis".to_string();
    p.ylabel = "Imaginary Axis".to_string();
    Ok(vec![])
}

/// `[Gm, Pm, Wcg, Wcp] = margin(sys)` — Gain margin (linear), Phase margin (deg), crossover frequencies.
pub fn fn_margin(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("margin", "1 (sys)", args.len()));
    }
    let sys = extract_ss(&args[0])?;
    let n = 400usize;
    let ws: Vec<f64> = (0..n).map(|i| 10.0f64.powf(-3.0 + 7.0 * (i as f64) / ((n - 1) as f64))).collect();
    let mut mags = Vec::with_capacity(n);
    let mut phases = Vec::with_capacity(n);
    for &w in &ws {
        let g = eval_ss_freq(&sys, Complex::new(0.0, w))?;
        mags.push(g.abs());
        phases.push(g.im.atan2(g.re) * 180.0 / PI);
    }
    for i in 1..n {
        while phases[i] - phases[i - 1] > 180.0 {
            phases[i] -= 360.0;
        }
        while phases[i] - phases[i - 1] < -180.0 {
            phases[i] += 360.0;
        }
    }

    // Gain crossover Wcp (|G(jw)| = 1) -> Phase margin Pm = 180 + angle(G(j*Wcp))
    let mut pm = f64::INFINITY;
    let mut wcp = f64::NAN;
    for i in 0..(n - 1) {
        let m0 = mags[i] - 1.0;
        let m1 = mags[i + 1] - 1.0;
        if m0 * m1 <= 0.0 && (mags[i] - mags[i + 1]).abs() > 1e-15 {
            let t = -m0 / (m1 - m0);
            let w_cross = ws[i] + t * (ws[i + 1] - ws[i]);
            let ph = phases[i] + t * (phases[i + 1] - phases[i]);
            pm = 180.0 + ph;
            wcp = w_cross;
            break;
        }
    }

    // Phase crossover Wcg (angle = -180 deg) -> Gain margin Gm = 1 / |G(j*Wcg)|
    let mut gm = f64::INFINITY;
    let mut wcg = f64::NAN;
    for i in 0..(n - 1) {
        let p0 = phases[i] + 180.0;
        let p1 = phases[i + 1] + 180.0;
        if p0 * p1 <= 0.0 && (phases[i] - phases[i + 1]).abs() > 1e-15 {
            let t = -p0 / (p1 - p0);
            let w_cross = ws[i] + t * (ws[i + 1] - ws[i]);
            let mg = mags[i] + t * (mags[i + 1] - mags[i]);
            gm = if mg > 0.0 { 1.0 / mg } else { f64::INFINITY };
            wcg = w_cross;
            break;
        }
    }

    let mut out = vec![Value::scalar(gm)];
    if nargout >= 2 {
        out.push(Value::scalar(pm));
    }
    if nargout >= 3 {
        out.push(Value::scalar(wcg));
    }
    if nargout >= 4 {
        out.push(Value::scalar(wcp));
    }
    Ok(out)
}

/// Controllability matrix `Co = ctrb(A, B)`.
pub fn fn_ctrb(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("ctrb", "2 (A, B)", args.len()));
    }
    let a = args[0].to_dense_matrix()?;
    let b = args[1].to_dense_matrix()?;
    let n = a.rows();
    let m = b.cols();
    let mut co = Matrix::zeros(n, n * m);
    let mut cur = b.clone();
    for k in 0..n {
        for r in 0..n {
            for c in 0..m {
                co.set(r, k * m + c, cur.get(r, c));
            }
        }
        if k + 1 < n {
            cur = a.matmul(&cur)?;
        }
    }
    ok1(Value::matrix(co))
}

/// Observability matrix `Ob = obsv(A, C)`.
pub fn fn_obsv(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("obsv", "2 (A, C)", args.len()));
    }
    let a = args[0].to_dense_matrix()?;
    let c = args[1].to_dense_matrix()?;
    let n = a.rows();
    let p = c.rows();
    let mut ob = Matrix::zeros(n * p, n);
    let mut cur = c.clone();
    for k in 0..n {
        for r in 0..p {
            for col in 0..n {
                ob.set(k * p + r, col, cur.get(r, col));
            }
        }
        if k + 1 < n {
            cur = cur.matmul(&a)?;
        }
    }
    ok1(Value::matrix(ob))
}

/// SISO pole placement `K = place(A, B, p)` / `K = acker(A, B, p)` via Ackermann's formula.
pub fn fn_place(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 3 {
        return Err(MatroError::arity("place", "3 (A, B, p)", args.len()));
    }
    let a = args[0].to_dense_matrix()?;
    let b = args[1].to_dense_matrix()?;
    let poles = args[2].to_dense_matrix()?;
    let n = a.rows();
    if b.cols() != 1 || poles.len() != n {
        return Err(MatroError::dimension(
            "place: requires SISO B (n x 1) and n desired poles",
        ));
    }
    // Build desired characteristic polynomial phi(s) = prod(s - p_i)
    let mut poly_c = vec![Complex::ONE];
    for i in 0..n {
        let p = poles.get_linear_c(i);
        let mut next = vec![Complex::ZERO; poly_c.len() + 1];
        for j in 0..poly_c.len() {
            next[j] = next[j] + poly_c[j];
            next[j + 1] = next[j + 1] - poly_c[j] * p;
        }
        poly_c = next;
    }
    // Evaluate matrix polynomial phi(A) = sum c_k A^{n-k} via Horner's method
    let eye = Matrix::eye(n, n);
    let mut phi_a = Matrix::zeros(n, n);
    for (idx, c) in poly_c.iter().enumerate() {
        if idx == 0 {
            phi_a = eye.map(|v| v * c.re);
        } else {
            phi_a = phi_a.matmul(&a)?;
            for i in 0..n {
                phi_a.set(i, i, phi_a.get(i, i) + c.re);
            }
        }
    }
    let co = fn_ctrb(&[Value::matrix(a), Value::matrix(b)])?[0].to_dense_matrix()?;
    let co_inv = co.inv()?;
    let mut e_n = Matrix::zeros(1, n);
    e_n.set(0, n - 1, 1.0);
    let k = e_n.matmul(&co_inv)?.matmul(&phi_a)?;
    ok1(Value::matrix(k))
}

/// Solve continuous Lyapunov equation `A*X + X*A' + Q = 0` via Kronecker linear solve.
pub fn solve_lyap_core(a: &Matrix, q: &Matrix) -> Result<Matrix> {
    let n = a.rows();
    let n2 = n * n;
    let mut kron = Matrix::zeros(n2, n2);
    let mut rhs = Matrix::zeros(n2, 1);
    for j in 0..n {
        for i in 0..n {
            let row = j * n + i;
            rhs.set(row, 0, -q.get(i, j));
            // (I kron A): sum_k A(i, k) X(k, j)
            for k in 0..n {
                let col = j * n + k;
                kron.set(row, col, kron.get(row, col) + a.get(i, k));
            }
            // (A kron I): sum_k A(j, k) X(i, k)
            for k in 0..n {
                let col = k * n + i;
                kron.set(row, col, kron.get(row, col) + a.get(j, k));
            }
        }
    }
    let vec_x = kron.solve(&rhs)?;
    let mut x = Matrix::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            x.set(i, j, vec_x.get(j * n + i, 0));
        }
    }
    Ok(x)
}

pub fn fn_lyap(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("lyap", "2 (A, Q)", args.len()));
    }
    let a = args[0].to_dense_matrix()?;
    let q = args[1].to_dense_matrix()?;
    ok1(Value::matrix(solve_lyap_core(&a, &q)?))
}

/// Solve Continuous Algebraic Riccati Equation `A'P + PA - P*B*R^{-1}*B'*P + Q = 0`
/// using the Hamiltonian stable invariant subspace + Newton-Kleinman polish.
pub fn solve_care_core(a: &Matrix, b: &Matrix, q: &Matrix, r: &Matrix) -> Result<Matrix> {
    let n = a.rows();
    let r_inv = r.inv()?;
    let brinvbt = b.matmul(&r_inv)?.matmul(&b.transpose())?;

    // Build 2n x 2n Hamiltonian matrix H = [A, -B*R^{-1}*B'; -Q, -A']
    let mut h = Matrix::zeros(2 * n, 2 * n);
    for row in 0..n {
        for col in 0..n {
            h.set(row, col, a.get(row, col));
            h.set(row, n + col, -brinvbt.get(row, col));
            h.set(n + row, col, -q.get(row, col));
            h.set(n + row, n + col, -a.get(col, row));
        }
    }
    let (v_ham, d_ham) = h.eig()?;
    // Sort eigenvalues by ascending real part and take the n stable ones (Re(lambda) < 0)
    let mut idxs: Vec<usize> = (0..(2 * n)).collect();
    idxs.sort_by(|&i, &j| d_ham.get_c(i, i).re.total_cmp(&d_ham.get_c(j, j).re));

    let mut v1 = Matrix::zeros(n, n);
    let mut v2 = Matrix::zeros(n, n);
    for (ci, &eig_i) in idxs.iter().take(n).enumerate() {
        for row in 0..n {
            v1.set_c(row, ci, v_ham.get_c(row, eig_i));
            v2.set_c(row, ci, v_ham.get_c(n + row, eig_i));
        }
    }
    // P = Re(V2 * V1^{-1}) -> solve V1' * P' = V2'
    let v1_t = v1.transpose();
    let v2_t = v2.transpose();
    let p_t = v1_t.solve(&v2_t)?;
    let mut p = Matrix::zeros(n, n);
    for row in 0..n {
        for col in 0..n {
            // Symmetrize real part
            let val = 0.5 * (p_t.get_c(col, row).re + p_t.get_c(row, col).re);
            p.set(row, col, val);
        }
    }

    // 3 Newton-Kleinman refinement steps using exact Lyapunov solve for machine-precision polish
    for _ in 0..3 {
        let k = r_inv.matmul(&b.transpose())?.matmul(&p)?;
        let bk = b.matmul(&k)?;
        let a_cl = a.elementwise(&bk, |x, y| x - y)?;
        let kt_r_k = k.transpose().matmul(r)?.matmul(&k)?;
        let q_cl = q.elementwise(&kt_r_k, |x, y| x + y)?;
        if let Ok(p_new) = solve_lyap_core(&a_cl.transpose(), &q_cl) {
            let mut sym = Matrix::zeros(n, n);
            for row in 0..n {
                for col in 0..n {
                    sym.set(row, col, 0.5 * (p_new.get(row, col) + p_new.get(col, row)));
                }
            }
            p = sym;
        } else {
            break;
        }
    }
    Ok(p)
}

pub fn fn_care(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 4 {
        return Err(MatroError::arity("care", "4 (A, B, Q, R)", args.len()));
    }
    let p = solve_care_core(
        &args[0].to_dense_matrix()?,
        &args[1].to_dense_matrix()?,
        &args[2].to_dense_matrix()?,
        &args[3].to_dense_matrix()?,
    )?;
    ok1(Value::matrix(p))
}

/// `[K, S, P] = lqr(A, B, Q, R)` — Continuous-time Linear-Quadratic Regulator.
pub fn fn_lqr(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() != 4 {
        return Err(MatroError::arity("lqr", "4 (A, B, Q, R)", args.len()));
    }
    let a = args[0].to_dense_matrix()?;
    let b = args[1].to_dense_matrix()?;
    let q = args[2].to_dense_matrix()?;
    let r = args[3].to_dense_matrix()?;
    let s = solve_care_core(&a, &b, &q, &r)?;
    let k = r.inv()?.matmul(&b.transpose())?.matmul(&s)?;
    let mut out = vec![Value::matrix(k.clone())];
    if nargout >= 2 {
        out.push(Value::matrix(s));
    }
    if nargout >= 3 {
        let bk = b.matmul(&k)?;
        let a_cl = a.elementwise(&bk, |x, y| x - y)?;
        let (_, d) = a_cl.eig()?;
        let poles: Vec<Complex> = (0..d.rows()).map(|i| d.get_c(i, i)).collect();
        out.push(Value::matrix(Matrix::complex_col_vec(poles).normalize()));
    }
    Ok(out)
}

/// `[L, P, E] = lqe(A, G, C, Q, R)` — Continuous-time Kalman Filter estimator gain via duality.
pub fn fn_lqe(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() != 5 {
        return Err(MatroError::arity("lqe", "5 (A, G, C, Q, R)", args.len()));
    }
    let a = args[0].to_dense_matrix()?;
    let g = args[1].to_dense_matrix()?;
    let c = args[2].to_dense_matrix()?;
    let q = args[3].to_dense_matrix()?;
    let r = args[4].to_dense_matrix()?;
    let gqgt = g.matmul(&q)?.matmul(&g.transpose())?;
    let p = solve_care_core(&a.transpose(), &c.transpose(), &gqgt, &r)?;
    let l = p.matmul(&c.transpose())?.matmul(&r.inv()?)?;
    let mut out = vec![Value::matrix(l.clone())];
    if nargout >= 2 {
        out.push(Value::matrix(p));
    }
    if nargout >= 3 {
        let lc = l.matmul(&c)?;
        let a_est = a.elementwise(&lc, |x, y| x - y)?;
        let (_, d) = a_est.eig()?;
        let poles: Vec<Complex> = (0..d.rows()).map(|i| d.get_c(i, i)).collect();
        out.push(Value::matrix(Matrix::complex_col_vec(poles).normalize()));
    }
    Ok(out)
}
