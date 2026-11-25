//! Extended linear algebra, special matrices, decompositions, and stdlib utilities:
//! - Special matrices: `toeplitz`, `hankel`, `vander`, `compan`, `hilb`, `invhilb`, `pascal`, `gallery`
//! - Matrix decompositions & queries: `hess`, `schur`, `balance`, `rcond`, `bandwidth`, `del2`
//! - Number theory, bitwise & predicates: `factor`, `bitcmp`, `iscellstr`, `ishandle`,
//!   `isobject`, `istimetable`, `nzmax`, `format`
//! - Functional matrix operators: `mtimes`, `mrdivide`, `mldivide`, `mpower`

use matro_types::{Complex, MatroError, Matrix, NumericClass, Result, Value};

fn ok1(v: Value) -> Result<Vec<Value>> {
    Ok(vec![v])
}

// ---------------------------------------------------------------------------
// Special Matrices: `toeplitz`, `hankel`, `vander`, `compan`, `hilb`, `invhilb`, `pascal`, `gallery`
// ---------------------------------------------------------------------------

pub fn fn_toeplitz(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("toeplitz", "1 or 2 (c [, r])", args.len()));
    }
    let cm = args[0].to_dense_matrix()?;
    let c: Vec<Complex> = (0..cm.len()).map(|i| cm.get_linear_c(i)).collect();
    let r: Vec<Complex> = if args.len() == 2 {
        let rm = args[1].to_dense_matrix()?;
        (0..rm.len()).map(|i| rm.get_linear_c(i)).collect()
    } else {
        // Hermitian symmetric when single argument given
        c.iter().map(|z| z.conj()).collect()
    };
    let nr = c.len();
    let nc = r.len();
    let mut out = Matrix::zeros(nr, nc);
    for col in 0..nc {
        for row in 0..nr {
            let v = if row >= col {
                c[row - col]
            } else {
                r[col - row]
            };
            out.set_c(row, col, v);
        }
    }
    ok1(Value::matrix(out.normalize()))
}

pub fn fn_hankel(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("hankel", "1 or 2 (c [, r])", args.len()));
    }
    let cm = args[0].to_dense_matrix()?;
    let c: Vec<Complex> = (0..cm.len()).map(|i| cm.get_linear_c(i)).collect();
    let r: Vec<Complex> = if args.len() == 2 {
        let rm = args[1].to_dense_matrix()?;
        (0..rm.len()).map(|i| rm.get_linear_c(i)).collect()
    } else {
        vec![Complex::ZERO; c.len()]
    };
    let nr = c.len();
    let nc = r.len();
    let mut out = Matrix::zeros(nr, nc);
    for col in 0..nc {
        for row in 0..nr {
            let idx = row + col;
            let v = if idx < nr {
                c[idx]
            } else {
                let ri = idx - nr + 1;
                if ri < nc { r[ri] } else { Complex::ZERO }
            };
            out.set_c(row, col, v);
        }
    }
    ok1(Value::matrix(out.normalize()))
}

pub fn fn_vander(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("vander", "1 or 2 (v [, n])", args.len()));
    }
    let vm = args[0].to_dense_matrix()?;
    let m = vm.len();
    let n = args
        .get(1)
        .map(|v| v.as_scalar().map(|s| s.max(0.0).round() as usize))
        .transpose()?
        .unwrap_or(m);
    let mut out = Matrix::zeros(m, n);
    if n == 0 || m == 0 {
        return ok1(Value::matrix(out));
    }
    for r in 0..m {
        let x = vm.get_linear_c(r);
        let mut pow = Complex::ONE;
        for c in (0..n).rev() {
            out.set_c(r, c, pow);
            pow = pow * x;
        }
    }
    ok1(Value::matrix(out.normalize()))
}

pub fn fn_compan(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("compan", "1 (u)", args.len()));
    }
    let u = args[0].to_dense_matrix()?;
    let n = u.len();
    if n <= 1 {
        return ok1(Value::matrix(Matrix::zeros(0, 0)));
    }
    let lead = u.get_linear_c(0);
    if lead.is_zero() {
        return Err(MatroError::runtime("compan: leading polynomial coefficient must be nonzero"));
    }
    let d = n - 1;
    let mut out = Matrix::zeros(d, d);
    for c in 0..d {
        out.set_c(0, c, -u.get_linear_c(c + 1) / lead);
    }
    for r in 1..d {
        out.set(r, r - 1, 1.0);
    }
    ok1(Value::matrix(out.normalize()))
}

pub fn fn_hilb(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("hilb", "1 (n)", args.len()));
    }
    let n = args[0].as_scalar()?.max(0.0).round() as usize;
    let mut out = Matrix::zeros(n, n);
    for c in 0..n {
        for r in 0..n {
            out.set(r, c, 1.0 / ((r + c + 1) as f64));
        }
    }
    ok1(Value::matrix(out))
}

fn binom_f64(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut res = 1.0;
    for i in 1..=k {
        res = res * ((n - k + i) as f64) / (i as f64);
    }
    res.round()
}

pub fn fn_invhilb(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("invhilb", "1 (n)", args.len()));
    }
    let n = args[0].as_scalar()?.max(0.0).round() as usize;
    let mut out = Matrix::zeros(n, n);
    for j in 1..=n {
        for i in 1..=n {
            let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
            let val = sign
                * ((i + j - 1) as f64)
                * binom_f64(n + i - 1, n - j)
                * binom_f64(n + j - 1, n - i)
                * binom_f64(i + j - 2, i - 1).powi(2);
            out.set(i - 1, j - 1, val);
        }
    }
    ok1(Value::matrix(out))
}

pub fn fn_pascal(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("pascal", "1 or 2 (n [, k])", args.len()));
    }
    let n = args[0].as_scalar()?.max(0.0).round() as usize;
    let k = args.get(1).map(|v| v.as_scalar()).transpose()?.unwrap_or(0.0).round() as i32;
    let mut out = Matrix::zeros(n, n);
    if k == 0 {
        for c in 0..n {
            for r in 0..n {
                out.set(r, c, binom_f64(r + c, r));
            }
        }
    } else if k == 1 || k == 2 {
        for r in 0..n {
            for c in 0..=r {
                let sign = if c % 2 == 0 { 1.0 } else { -1.0 };
                out.set(r, c, sign * binom_f64(r, c));
            }
        }
        if k == 2 {
            // MATLAB: rot90(pascal(n, 1), 3), negated for even n, so P^3 == I
            let mut m2 = Matrix::zeros(n, n);
            let s = if n % 2 == 0 { -1.0 } else { 1.0 };
            for r in 0..n {
                for c in 0..n {
                    m2.set(r, c, s * out.get(n - 1 - c, r));
                }
            }
            out = m2;
        }
    } else {
        return Err(MatroError::runtime("pascal: k must be 0, 1, or 2"));
    }
    ok1(Value::matrix(out))
}

pub fn fn_gallery(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("gallery", ">= 2 (name, n [, ...])", 0));
    }
    let kind = args[0].as_str()?.to_ascii_lowercase();
    if kind == "tridiag" {
        let n = args.get(1).map(|v| v.as_scalar()).transpose()?.unwrap_or(4.0).max(0.0).round() as usize;
        let (c_val, d_val, e_val) = if args.len() >= 5 {
            (args[2].as_scalar()?, args[3].as_scalar()?, args[4].as_scalar()?)
        } else {
            (-1.0, 2.0, -1.0)
        };
        let mut trips = Vec::with_capacity(n * 3);
        for i in 0..n {
            trips.push((i, i, d_val));
            if i + 1 < n {
                trips.push((i + 1, i, c_val));
                trips.push((i, i + 1, e_val));
            }
        }
        return ok1(Value::sparse(matro_types::SparseMatrix::from_triplets(n, n, &trips)?));
    }
    if kind == "poisson" {
        let n = args.get(1).map(|v| v.as_scalar()).transpose()?.unwrap_or(4.0).max(0.0).round() as usize;
        let n2 = n * n;
        let mut trips = Vec::with_capacity(n2 * 5);
        for r in 0..n {
            for c in 0..n {
                let idx = c * n + r;
                trips.push((idx, idx, 4.0));
                if r > 0 { trips.push((idx, idx - 1, -1.0)); }
                if r + 1 < n { trips.push((idx, idx + 1, -1.0)); }
                if c > 0 { trips.push((idx, idx - n, -1.0)); }
                if c + 1 < n { trips.push((idx, idx + n, -1.0)); }
            }
        }
        return ok1(Value::sparse(matro_types::SparseMatrix::from_triplets(n2, n2, &trips)?));
    }
    if kind == "circul" {
        let v_m = args.get(1).ok_or_else(|| MatroError::arity("gallery", "2 ('circul', v)", 1))?.to_dense_matrix()?;
        let v: Vec<f64> = if v_m.len() == 1 {
            let n = v_m.data()[0].max(0.0).round() as usize;
            (1..=n).map(|k| k as f64).collect()
        } else {
            v_m.data().to_vec()
        };
        let len = v.len();
        let mut a = Matrix::zeros(len, len);
        for r in 0..len {
            for c in 0..len {
                a.set(r, c, v[(c + len - r) % len]);
            }
        }
        return ok1(Value::matrix(a));
    }
    let n = args.get(1).map(|v| v.as_scalar()).transpose()?.unwrap_or(4.0).max(0.0).round() as usize;
    let m = match kind.as_str() {
        "lehmer" => {
            let mut a = Matrix::zeros(n, n);
            for c in 0..n {
                for r in 0..n {
                    let lo = (r.min(c) + 1) as f64;
                    let hi = (r.max(c) + 1) as f64;
                    a.set(r, c, lo / hi);
                }
            }
            a
        }
        "minij" => {
            let mut a = Matrix::zeros(n, n);
            for c in 0..n {
                for r in 0..n {
                    a.set(r, c, (r.min(c) + 1) as f64);
                }
            }
            a
        }
        "moler" => {
            let alpha = args.get(2).map(|v| v.as_scalar()).transpose()?.unwrap_or(-1.0);
            let mut l = Matrix::eye(n, n);
            for r in 0..n {
                for c in 0..r {
                    l.set(r, c, alpha);
                }
            }
            l.transpose().matmul(&l)?
        }
        "frank" => {
            let mut a = Matrix::zeros(n, n);
            for r in 0..n {
                for c in 0..n {
                    if c + 1 >= r {
                        a.set(r, c, (n - r.max(c)) as f64);
                    }
                }
            }
            a
        }
        "wilk" => {
            let mut a = Matrix::zeros(n, n);
            let half = (n as f64 - 1.0) * 0.5;
            for i in 0..n {
                a.set(i, i, (i as f64 - half).abs());
                if i + 1 < n {
                    a.set(i, i + 1, 1.0);
                    a.set(i + 1, i, 1.0);
                }
            }
            a
        }
        "orthog" => {
            let mut a = Matrix::zeros(n, n);
            let scale = (2.0 / (n + 1) as f64).sqrt();
            for r in 0..n {
                for c in 0..n {
                    a.set(
                        r,
                        c,
                        scale * (std::f64::consts::PI * ((r + 1) * (c + 1)) as f64 / (n + 1) as f64).sin(),
                    );
                }
            }
            a
        }
        other => {
            return Err(MatroError::runtime(format!(
                "gallery: unsupported test matrix '{other}'"
            )));
        }
    };
    ok1(Value::matrix(m))
}

// ---------------------------------------------------------------------------
// Matrix Decompositions & Queries: `hess`, `schur`, `balance`, `rcond`, `bandwidth`, `del2`
// ---------------------------------------------------------------------------

/// Householder Hessenberg reduction: `A = P * H * P'`, where `P` is unitary and `H` is upper Hessenberg.
fn hessenberg_reduce(a: &Matrix) -> Result<(Matrix, Matrix)> {
    let n = a.rows();
    if n != a.cols() {
        return Err(MatroError::dimension("hess: input must be a square matrix"));
    }
    let mut h = a.clone();
    let mut p = Matrix::eye(n, n);
    if n <= 2 {
        return Ok((p, h));
    }
    for k in 0..(n - 2) {
        // Compute Householder vector v for column k, rows k+1..n
        let m = n - k - 1;
        let mut x = Vec::with_capacity(m);
        let mut norm_sq = 0.0;
        for i in (k + 1)..n {
            let z = h.get_c(i, k);
            x.push(z);
            norm_sq += z.norm_sq();
        }
        let norm_x = norm_sq.sqrt();
        if norm_x < 1e-15 {
            continue;
        }
        let alpha = if x[0].abs() > 0.0 {
            -(x[0] / Complex::from_real(x[0].abs())) * Complex::from_real(norm_x)
        } else {
            Complex::from_real(-norm_x)
        };
        let mut v = x;
        v[0] = v[0] - alpha;
        let v_norm_sq: f64 = v.iter().map(|z| z.norm_sq()).sum();
        if v_norm_sq < 1e-30 {
            continue;
        }
        let beta = 2.0 / v_norm_sq;

        // H[k+1..n, k..n] = (I - beta v v^H) H[k+1..n, k..n]
        for col in k..n {
            let mut dot = Complex::ZERO;
            for i in 0..m {
                dot = dot + v[i].conj() * h.get_c(k + 1 + i, col);
            }
            let s = dot * Complex::from_real(beta);
            for i in 0..m {
                let cur = h.get_c(k + 1 + i, col);
                h.set_c(k + 1 + i, col, cur - v[i] * s);
            }
        }
        // H[0..n, k+1..n] = H[0..n, k+1..n] (I - beta v v^H)
        for row in 0..n {
            let mut dot = Complex::ZERO;
            for i in 0..m {
                dot = dot + h.get_c(row, k + 1 + i) * v[i];
            }
            let s = dot * Complex::from_real(beta);
            for i in 0..m {
                let cur = h.get_c(row, k + 1 + i);
                h.set_c(row, k + 1 + i, cur - s * v[i].conj());
            }
        }
        // P[0..n, k+1..n] = P[0..n, k+1..n] (I - beta v v^H)
        for row in 0..n {
            let mut dot = Complex::ZERO;
            for i in 0..m {
                dot = dot + p.get_c(row, k + 1 + i) * v[i];
            }
            let s = dot * Complex::from_real(beta);
            for i in 0..m {
                let cur = p.get_c(row, k + 1 + i);
                p.set_c(row, k + 1 + i, cur - s * v[i].conj());
            }
        }
        // Zero exact sub-subdiagonal noise
        for i in (k + 2)..n {
            h.set_c(i, k, Complex::ZERO);
        }
    }
    Ok((p.normalize(), h.normalize()))
}

pub fn fn_hess(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("hess", "1 (A)", args.len()));
    }
    let a = args[0].to_dense_matrix()?;
    let (p, h) = hessenberg_reduce(&a)?;
    if nargout >= 2 {
        Ok(vec![Value::matrix(p), Value::matrix(h)])
    } else {
        ok1(Value::matrix(h))
    }
}

/// `[U, T] = schur(A [, 'real'|'complex'])` — Shifted QR iteration on Hessenberg form (`A = U * T * U'`).
pub fn fn_schur(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("schur", "1 or 2 (A [, 'real'|'complex'])", args.len()));
    }
    let a = args[0].to_dense_matrix()?;
    let force_complex = args
        .get(1)
        .and_then(crate::value_as_text)
        .is_some_and(|s| s.eq_ignore_ascii_case("complex"));
    let n = a.rows();
    if n != a.cols() {
        return Err(MatroError::dimension("schur: input must be square"));
    }
    if n <= 1 {
        return if nargout >= 2 {
            Ok(vec![Value::matrix(Matrix::eye(n, n)), Value::matrix(a)])
        } else {
            ok1(Value::matrix(a))
        };
    }

    let (mut u, mut t) = hessenberg_reduce(&a)?;
    let use_complex = force_complex || a.is_complex();

    // Shifted QR iteration with Wilkinson shift (complex shift when use_complex)
    let mut hi = n;
    let max_iter = 40 * n;
    for iter in 0..max_iter {
        if hi <= 1 {
            break;
        }
        // Deflate converged subdiagonal entries
        let sub = t.get_c(hi - 1, hi - 2).abs();
        let diag_scale = (t.get_c(hi - 2, hi - 2).abs() + t.get_c(hi - 1, hi - 1).abs()).max(1e-30);
        if sub <= 1e-13 * diag_scale {
            t.set_c(hi - 1, hi - 2, Complex::ZERO);
            hi -= 1;
            continue;
        }
        // In real Schur mode, stop on an irreducible 2x2 block with complex conjugate eigenvalues
        if !use_complex && hi >= 2 {
            let sub2 = if hi >= 3 { t.get_c(hi - 2, hi - 3).abs() } else { 0.0 };
            let dscale2 = if hi >= 3 {
                (t.get_c(hi - 3, hi - 3).abs() + t.get_c(hi - 2, hi - 2).abs()).max(1e-30)
            } else {
                1.0
            };
            if hi == 2 || sub2 <= 1e-13 * dscale2 {
                let a11 = t.get(hi - 2, hi - 2);
                let a12 = t.get(hi - 2, hi - 1);
                let a21 = t.get(hi - 1, hi - 2);
                let a22 = t.get(hi - 1, hi - 1);
                let tr = a11 + a22;
                let det = a11 * a22 - a12 * a21;
                if tr * tr - 4.0 * det < -1e-14 * (tr * tr + det.abs()).max(1.0) {
                    if hi >= 3 {
                        t.set_c(hi - 2, hi - 3, Complex::ZERO);
                    }
                    hi -= 2;
                    continue;
                }
            }
        }

        // Compute Wilkinson shift from trailing 2x2
        let a11 = t.get_c(hi - 2, hi - 2);
        let a12 = t.get_c(hi - 2, hi - 1);
        let a21 = t.get_c(hi - 1, hi - 2);
        let a22 = t.get_c(hi - 1, hi - 1);
        let tr = a11 + a22;
        let det = a11 * a22 - a12 * a21;
        let disc = tr * tr - det * Complex::from_real(4.0);
        let shift = if iter % 15 == 14 {
            // Exceptional shift to break symmetry
            Complex::new(0.75 * sub, if use_complex { 0.4375 * sub } else { 0.0 })
        } else if use_complex || disc.re >= 0.0 {
            let sd = disc.sqrt();
            let l1 = (tr + sd) * Complex::from_real(0.5);
            let l2 = (tr - sd) * Complex::from_real(0.5);
            if (l1 - a22).abs() <= (l2 - a22).abs() { l1 } else { l2 }
        } else {
            a22
        };

        let mut shifted = Matrix::zeros(hi, hi);
        for c in 0..hi {
            for r in 0..hi {
                let mut v = t.get_c(r, c);
                if r == c {
                    v = v - shift;
                }
                shifted.set_c(r, c, v);
            }
        }
        let (q_sub, r_sub) = shifted.qr()?;
        let rq = r_sub.matmul(&q_sub)?;
        // Update T[0..hi, 0..hi] = R*Q + shift*I, T[0..hi, hi..n] = Q^H * T[0..hi, hi..n]
        for c in 0..hi {
            for r in 0..hi {
                let mut v = rq.get_c(r, c);
                if r == c {
                    v = v + shift;
                }
                t.set_c(r, c, v);
            }
        }
        if hi < n {
            let mut top_right = Matrix::zeros(hi, n - hi);
            for c in 0..(n - hi) {
                for r in 0..hi {
                    top_right.set_c(r, c, t.get_c(r, hi + c));
                }
            }
            let new_tr = q_sub.ctranspose().matmul(&top_right)?;
            for c in 0..(n - hi) {
                for r in 0..hi {
                    t.set_c(r, hi + c, new_tr.get_c(r, c));
                }
            }
        }
        // Update U[:, 0..hi] = U[:, 0..hi] * Q
        let mut u_left = Matrix::zeros(n, hi);
        for c in 0..hi {
            for r in 0..n {
                u_left.set_c(r, c, u.get_c(r, c));
            }
        }
        let new_ul = u_left.matmul(&q_sub)?;
        for c in 0..hi {
            for r in 0..n {
                u.set_c(r, c, new_ul.get_c(r, c));
            }
        }
    }

    // Clean negligible entries strictly below subdiagonal (or below diagonal in complex mode)
    for c in 0..n {
        let start_r = if use_complex { c + 1 } else { c + 2 };
        for r in start_r..n {
            if t.get_c(r, c).abs() < 1e-12 {
                t.set_c(r, c, Complex::ZERO);
            }
        }
    }

    if nargout >= 2 {
        Ok(vec![Value::matrix(u.normalize()), Value::matrix(t.normalize())])
    } else {
        ok1(Value::matrix(t.normalize()))
    }
}

/// `[T, B] = balance(A)` or `B = balance(A)` — Parlett-Reinsch diagonal scaling.
pub fn fn_balance(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("balance", "1 or 2 (A [, 'noperm'])", args.len()));
    }
    let a = args[0].to_dense_matrix()?;
    let n = a.rows();
    if n != a.cols() {
        return Err(MatroError::dimension("balance: input must be square"));
    }
    let mut b = a.clone();
    let mut d = vec![1.0f64; n];
    let radix = 2.0f64;
    let sqrdx = radix * radix;
    for _ in 0..40 {
        let mut converged = true;
        for i in 0..n {
            let mut r_norm = 0.0;
            let mut c_norm = 0.0;
            for j in 0..n {
                if i != j {
                    c_norm += b.get_c(j, i).abs();
                    r_norm += b.get_c(i, j).abs();
                }
            }
            if c_norm == 0.0 || r_norm == 0.0 {
                continue;
            }
            let mut g = r_norm / radix;
            let mut f = 1.0;
            let s = c_norm + r_norm;
            while c_norm < g {
                f *= radix;
                c_norm *= sqrdx;
            }
            g = r_norm * radix;
            while c_norm > g {
                f /= radix;
                c_norm /= sqrdx;
            }
            if (c_norm + r_norm) / f < 0.95 * s {
                converged = false;
                let g_inv = 1.0 / f;
                d[i] *= f;
                for j in 0..n {
                    b.set_c(i, j, b.get_c(i, j) * Complex::from_real(g_inv));
                    b.set_c(j, i, b.get_c(j, i) * Complex::from_real(f));
                }
            }
        }
        if converged {
            break;
        }
    }
    let mut t_mat = Matrix::zeros(n, n);
    for i in 0..n {
        t_mat.set(i, i, d[i]);
    }
    if nargout >= 2 {
        Ok(vec![Value::matrix(t_mat), Value::matrix(b.normalize())])
    } else {
        ok1(Value::matrix(b.normalize()))
    }
}

pub fn fn_rcond(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("rcond", "1 (A)", args.len()));
    }
    let a = args[0].to_dense_matrix()?;
    let n = a.rows();
    if n != a.cols() {
        return Err(MatroError::dimension("rcond: matrix must be square"));
    }
    if n == 0 {
        return ok1(Value::scalar(f64::INFINITY));
    }
    let norm1 = |m: &Matrix| -> f64 {
        (0..m.cols())
            .map(|c| (0..m.rows()).map(|r| m.get_c(r, c).abs()).sum::<f64>())
            .fold(0.0f64, f64::max)
    };
    let n_a = norm1(&a);
    if n_a == 0.0 {
        return ok1(Value::scalar(0.0));
    }
    let Ok(a_inv) = a.inv() else {
        return ok1(Value::scalar(0.0));
    };
    let n_inv = norm1(&a_inv);
    if !n_inv.is_finite() || n_inv == 0.0 {
        return ok1(Value::scalar(0.0));
    }
    ok1(Value::scalar(1.0 / (n_a * n_inv)))
}

pub fn fn_bandwidth(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("bandwidth", "1 or 2 (A [, 'lower'|'upper'])", args.len()));
    }
    let a = args[0].to_dense_matrix()?;
    let mut lower = 0usize;
    let mut upper = 0usize;
    for c in 0..a.cols() {
        for r in 0..a.rows() {
            if !a.get_c(r, c).is_zero() {
                if r > c {
                    lower = lower.max(r - c);
                } else if c > r {
                    upper = upper.max(c - r);
                }
            }
        }
    }
    if let Some(t) = args.get(1).and_then(crate::value_as_text) {
        return match t.to_ascii_lowercase().as_str() {
            "lower" => ok1(Value::scalar(lower as f64)),
            "upper" => ok1(Value::scalar(upper as f64)),
            other => Err(MatroError::runtime(format!("bandwidth: unknown type '{other}'"))),
        };
    }
    if nargout >= 2 {
        Ok(vec![Value::scalar(lower as f64), Value::scalar(upper as f64)])
    } else {
        ok1(Value::matrix(Matrix::row_vec(vec![lower as f64, upper as f64])))
    }
}

/// Discrete Laplacian `L = del2(U [, hx [, hy]])`.
pub fn fn_del2(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity("del2", "1 to 3 (U [, hx [, hy]])", args.len()));
    }
    let u = args[0].to_dense_matrix()?;
    let (nr, nc) = (u.rows(), u.cols());
    let hx = args.get(1).map(|v| v.as_scalar()).transpose()?.unwrap_or(1.0);
    let hy = args.get(2).map(|v| v.as_scalar()).transpose()?.unwrap_or(hx);

    if nr == 1 || nc == 1 {
        let n = u.len();
        let h2 = hx * hx;
        let mut out = vec![0.0; n];
        if n >= 3 {
            for i in 1..(n - 1) {
                out[i] = (u.data()[i - 1] - 2.0 * u.data()[i] + u.data()[i + 1]) / (2.0 * h2);
            }
            out[0] = 2.0 * out[1] - out[2];
            out[n - 1] = 2.0 * out[n - 2] - out[n - 3];
        }
        let m = if nr == 1 { Matrix::row_vec(out) } else { Matrix::col_vec(out) };
        return ok1(Value::matrix(m));
    }

    let d2_axis = |vals: &[f64], h: f64| -> Vec<f64> {
        let n = vals.len();
        let mut d2 = vec![0.0; n];
        if n >= 3 {
            let h2 = h * h;
            for i in 1..(n - 1) {
                d2[i] = (vals[i - 1] - 2.0 * vals[i] + vals[i + 1]) / h2;
            }
            d2[0] = 2.0 * d2[1] - d2[2];
            d2[n - 1] = 2.0 * d2[n - 2] - d2[n - 3];
        }
        d2
    };

    let mut out = Matrix::zeros(nr, nc);
    // Second derivative along y (rows)
    for c in 0..nc {
        let col: Vec<f64> = (0..nr).map(|r| u.get(r, c)).collect();
        let dy2 = d2_axis(&col, hy);
        for r in 0..nr {
            out.set(r, c, out.get(r, c) + dy2[r]);
        }
    }
    // Second derivative along x (cols)
    for r in 0..nr {
        let row: Vec<f64> = (0..nc).map(|c| u.get(r, c)).collect();
        let dx2 = d2_axis(&row, hx);
        for c in 0..nc {
            out.set(r, c, 0.25 * (out.get(r, c) + dx2[c]));
        }
    }
    ok1(Value::matrix(out))
}

// ---------------------------------------------------------------------------
// Number Theory, Bitwise, Predicates & Functional Matrix Operators
// ---------------------------------------------------------------------------

pub fn fn_factor(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("factor", "1 (n)", args.len()));
    }
    let raw = args[0].as_scalar()?;
    if raw < 0.0 || raw.fract() != 0.0 || !raw.is_finite() {
        return Err(MatroError::runtime("factor: input must be a nonnegative integer"));
    }
    let mut n = raw as u64;
    if n <= 1 {
        return ok1(Value::scalar(n as f64));
    }
    let mut factors = Vec::new();
    while n % 2 == 0 {
        factors.push(2.0);
        n /= 2;
    }
    let mut d = 3u64;
    while d * d <= n {
        while n % d == 0 {
            factors.push(d as f64);
            n /= d;
        }
        d += 2;
    }
    if n > 1 {
        factors.push(n as f64);
    }
    ok1(Value::matrix(Matrix::row_vec(factors)))
}

pub fn fn_bitcmp(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("bitcmp", "1 or 2 (A [, assumedtype])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let dt = m.dtype();
    let bits: u32 = if let Some(a1) = args.get(1) {
        if let Some(s) = crate::value_as_text(a1) {
            match s.to_ascii_lowercase().as_str() {
                "uint8" | "int8" => 8,
                "uint16" | "int16" => 16,
                "uint32" | "int32" => 32,
                "uint64" | "int64" => 64,
                other => return Err(MatroError::runtime(format!("bitcmp: unsupported type '{other}'"))),
            }
        } else {
            a1.as_scalar()?.round().clamp(1.0, 53.0) as u32
        }
    } else {
        match dt {
            NumericClass::Uint8 | NumericClass::Int8 => 8,
            NumericClass::Uint16 | NumericClass::Int16 => 16,
            NumericClass::Uint32 | NumericClass::Int32 => 32,
            NumericClass::Uint64 | NumericClass::Int64 => 64,
            _ => 53,
        }
    };
    let mask: u64 = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let out = m.map(|v| ((!(v.round() as u64)) & mask) as f64).cast_to(dt);
    ok1(Value::matrix(out))
}

pub fn fn_iscellstr(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("iscellstr", "1", args.len()));
    }
    let ok = match &args[0] {
        Value::Cell(c) => c.data.iter().all(|v| matches!(v, Value::String(_))),
        _ => false,
    };
    ok1(Value::bool(ok))
}

pub fn fn_ishandle(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("ishandle", "1", args.len()));
    }
    let ok = match &args[0] {
        Value::Handle(_) => true,
        Value::Object(o) => matches!(o.class_name.as_str(), "Figure" | "Axes" | "Line"),
        _ => false,
    };
    ok1(Value::bool(ok))
}

pub fn fn_isobject(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("isobject", "1", args.len()));
    }
    ok1(Value::bool(matches!(&args[0], Value::Object(_) | Value::Handle(_))))
}

pub fn fn_istimetable(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("istimetable", "1", args.len()));
    }
    let ok = match &args[0] {
        Value::Table(t) => t.col_names.first().is_some_and(|n| n == "Time"),
        _ => false,
    };
    ok1(Value::bool(ok))
}

pub fn fn_nzmax(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("nzmax", "1", args.len()));
    }
    let n = match &args[0] {
        Value::Sparse(s) => s.nnz().max(1),
        other => other.to_dense_matrix()?.len(),
    };
    ok1(Value::scalar(n as f64))
}

pub fn fn_matrix_op(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity(name, "2 (A, B)", args.len()));
    }
    let a = args[0].to_dense_matrix()?;
    let b = args[1].to_dense_matrix()?;
    let res = match name {
        "mtimes" => a.matmul(&b)?,
        "mldivide" => a.solve(&b)?,
        "mrdivide" => {
            // A / B = (B' \ A')'
            b.ctranspose().solve(&a.ctranspose())?.ctranspose()
        }
        "mpower" => {
            let p = b.as_scalar().ok_or_else(|| {
                MatroError::runtime("mpower: exponent must be a real scalar")
            })?;
            if a.is_scalar() {
                Matrix::complex_scalar(a.get_c(0, 0).powc(Complex::from_real(p))).normalize()
            } else if p == p.floor() && p.abs() <= 1000.0 {
                let n = a.rows();
                if n != a.cols() {
                    return Err(MatroError::dimension("mpower: matrix must be square"));
                }
                let mut base = if p < 0.0 { a.inv()? } else { a.clone() };
                let mut exp = p.abs() as u64;
                let mut acc = Matrix::eye(n, n);
                while exp > 0 {
                    if exp & 1 == 1 {
                        acc = acc.matmul(&base)?;
                    }
                    exp >>= 1;
                    if exp > 0 {
                        base = base.matmul(&base)?;
                    }
                }
                acc.normalize()
            } else {
                // General fractional power: expm(p * logm(A))
                let la = a.logm()?.map_c(|z| z * Complex::from_real(p));
                la.expm()?
            }
        }
        _ => unreachable!(),
    };
    ok1(Value::matrix(res))
}

pub fn fn_ichol(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("ichol", "1 or 2 (A [, opts])", args.len()));
    }
    let sp = match &args[0] {
        Value::Sparse(s) => (**s).clone(),
        other => matro_types::SparseMatrix::from_dense(&other.to_dense_matrix()?),
    };
    let mut diagcomp = 0.0;
    let mut upper = false;
    if let Some(Value::Struct(opts)) = args.get(1) {
        if let Some(v) = opts.get("diagcomp") {
            diagcomp = v.as_scalar()?.max(0.0);
        }
        if let Some(v) = opts.get("shape") {
            if let Ok(s) = v.as_str() {
                upper = s.eq_ignore_ascii_case("upper");
            }
        }
    }
    let l = sp.ichol0(diagcomp)?;
    if upper {
        ok1(Value::sparse(l.transpose()))
    } else {
        ok1(Value::sparse(l))
    }
}

pub fn fn_ilu(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("ilu", "1 or 2 (A [, opts])", args.len()));
    }
    let sp = match &args[0] {
        Value::Sparse(s) => (**s).clone(),
        other => matro_types::SparseMatrix::from_dense(&other.to_dense_matrix()?),
    };
    let (l, u) = sp.ilu0()?;
    if nargout >= 3 {
        let p = matro_types::SparseMatrix::eye(sp.rows());
        Ok(vec![Value::sparse(l), Value::sparse(u), Value::sparse(p)])
    } else if nargout == 2 {
        Ok(vec![Value::sparse(l), Value::sparse(u)])
    } else {
        // Single output: L + U - I
        let n = sp.rows();
        let eye = matro_types::SparseMatrix::eye(n);
        let lu = l.add_scaled(&u, 1.0)?.add_scaled(&eye, -1.0)?;
        ok1(Value::sparse(lu))
    }
}
