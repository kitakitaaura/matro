//! High-frequency Base MATLAB standard library built-ins:
//! hyperbolic & degree trig, coordinate transforms, `randi`/`randperm`, `blkdiag`, `circshift`,
//! `flip`, `repelem`, `accumarray`, `histcounts`, `discretize`, `spdiags`/`spones`/`spalloc`/`nonzeros`,
//! `pinv`/`null`/`orth`/`rref`/`cross`/`vecnorm`/`rms`/`normalize`/`rescale`/`clip`,
//! `trapz`/`cumtrapz`/`gradient`/`deconv`/`polyder`/`polyint`, `unwrap`/`sinc`/`db2mag`/`mag2db`,
//! moving windows (`movmean`..`movmax`), `cummin`/`cummax`/`bounds`, combinatorics/number theory,
//! structure predicates, and string/struct/cell utilities.

use std::collections::BTreeMap;
use std::f64::consts::PI;

use matro_types::{CellArray, Complex, MatroError, Matrix, Result, Rng, SparseMatrix, Value};

use crate::{as_usize, ok1, value_as_text};

fn parse_dim_opt(args: &[Value], pos: usize, name: &str, m: &Matrix) -> Result<usize> {
    if let Some(a) = args.get(pos) {
        as_usize(a, name)
    } else if m.rows() == 1 && m.cols() > 1 {
        Ok(2)
    } else {
        Ok(1)
    }
}

fn parse_size_args(args: &[Value], fn_name: &str) -> Result<Vec<usize>> {
    if args.is_empty() {
        return Ok(vec![1, 1]);
    }
    if args.len() == 1 {
        let m = args[0].as_matrix()?;
        if m.len() == 1 {
            let n = as_usize(&args[0], fn_name)?;
            return Ok(vec![n, n]);
        }
        return m
            .data()
            .iter()
            .map(|&v| {
                let d = v.round() as isize;
                if d < 0 {
                    Err(MatroError::dimension(format!("{fn_name}: size must be non-negative")))
                } else {
                    Ok(d as usize)
                }
            })
            .collect();
    }
    args.iter().map(|a| as_usize(a, fn_name)).collect()
}

// ---------------------------------------------------------------------------
// Hyperbolic & degree trig, coordinate transforms, rounding & signal helpers
// ---------------------------------------------------------------------------

pub fn fn_unary_c(name: &str, args: &[Value], f: fn(Complex) -> Complex) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity(name, "1", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    ok1(Value::matrix(m.map_c(f).normalize()))
}

pub fn c_sinh(z: Complex) -> Complex {
    Complex::new(z.re.sinh() * z.im.cos(), z.re.cosh() * z.im.sin())
}

pub fn c_cosh(z: Complex) -> Complex {
    Complex::new(z.re.cosh() * z.im.cos(), z.re.sinh() * z.im.sin())
}

pub fn c_tanh(z: Complex) -> Complex {
    c_sinh(z) / c_cosh(z)
}

pub fn c_asinh(z: Complex) -> Complex {
    if z.im == 0.0 {
        return Complex::from_real(z.re.asinh());
    }
    (z + (z * z + Complex::ONE).sqrt()).ln()
}

pub fn c_acosh(z: Complex) -> Complex {
    if z.im == 0.0 && z.re >= 1.0 {
        return Complex::from_real(z.re.acosh());
    }
    (z + (z - Complex::ONE).sqrt() * (z + Complex::ONE).sqrt()).ln()
}

pub fn c_atanh(z: Complex) -> Complex {
    if z.im == 0.0 && z.re.abs() < 1.0 {
        return Complex::from_real(z.re.atanh());
    }
    ((Complex::ONE + z) / (Complex::ONE - z)).ln() * Complex::from_real(0.5)
}

pub fn fn_elem_real(name: &str, args: &[Value], f: fn(f64) -> f64) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity(name, "1", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    ok1(Value::matrix(m.map(f)))
}

pub fn fn_elem_real2(name: &str, args: &[Value], f: fn(f64, f64) -> f64) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity(name, "2", args.len()));
    }
    let a = args[0].to_dense_matrix()?;
    let b = args[1].to_dense_matrix()?;
    ok1(Value::matrix(a.elementwise(&b, f)?))
}

pub fn sind_f(x: f64) -> f64 {
    let rem = x % 180.0;
    if rem == 0.0 {
        0.0
    } else if rem == 90.0 || rem == -270.0 {
        1.0
    } else if rem == -90.0 || rem == 270.0 {
        -1.0
    } else {
        x.to_radians().sin()
    }
}

pub fn cosd_f(x: f64) -> f64 {
    let rem = x % 360.0;
    if rem == 0.0 {
        1.0
    } else if rem.abs() == 90.0 || rem.abs() == 270.0 {
        0.0
    } else if rem.abs() == 180.0 {
        -1.0
    } else {
        x.to_radians().cos()
    }
}

pub fn tand_f(x: f64) -> f64 {
    sind_f(x) / cosd_f(x)
}

pub fn nextpow2_f(x: f64) -> f64 {
    let ax = x.abs();
    if ax == 0.0 || !ax.is_finite() {
        0.0
    } else {
        ax.log2().ceil()
    }
}

pub fn nthroot_f(x: f64, n: f64) -> f64 {
    if x < 0.0 {
        let ni = n.round();
        if (n - ni).abs() < 1e-12 && (ni as i64) % 2 != 0 {
            -(-x).powf(1.0 / n)
        } else {
            f64::NAN
        }
    } else {
        x.powf(1.0 / n)
    }
}

pub fn sinc_f(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let px = PI * x;
        px.sin() / px
    }
}

pub fn fn_pow2(args: &[Value]) -> Result<Vec<Value>> {
    match args.len() {
        1 => fn_elem_real("pow2", args, |x| 2.0f64.powf(x)),
        2 => fn_elem_real2("pow2", args, |f, e| f * 2.0f64.powf(e)),
        n => Err(MatroError::arity("pow2", "1 or 2", n)),
    }
}

pub fn fn_cart2pol(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity("cart2pol", "2 or 3", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let y = args[1].to_dense_matrix()?;
    let theta = y.elementwise(&x, f64::atan2)?;
    let rho = x.elementwise(&y, f64::hypot)?;
    let mut out = vec![Value::matrix(theta)];
    if nargout >= 2 {
        out.push(Value::matrix(rho));
    }
    if args.len() == 3 && nargout >= 3 {
        out.push(args[2].clone());
    }
    Ok(out)
}

pub fn fn_pol2cart(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity("pol2cart", "2 or 3", args.len()));
    }
    let theta = args[0].to_dense_matrix()?;
    let rho = args[1].to_dense_matrix()?;
    let x = rho.elementwise(&theta, |r, t| r * t.cos())?;
    let y = rho.elementwise(&theta, |r, t| r * t.sin())?;
    let mut out = vec![Value::matrix(x)];
    if nargout >= 2 {
        out.push(Value::matrix(y));
    }
    if args.len() == 3 && nargout >= 3 {
        out.push(args[2].clone());
    }
    Ok(out)
}

pub fn fn_cart2sph(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() != 3 {
        return Err(MatroError::arity("cart2sph", "3 (x, y, z)", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let y = args[1].to_dense_matrix()?;
    let z = args[2].to_dense_matrix()?;
    let hxy = x.elementwise(&y, f64::hypot)?;
    let az = y.elementwise(&x, f64::atan2)?;
    let el = z.elementwise(&hxy, f64::atan2)?;
    let r = hxy.elementwise(&z, f64::hypot)?;
    let mut out = vec![Value::matrix(az)];
    if nargout >= 2 {
        out.push(Value::matrix(el));
    }
    if nargout >= 3 {
        out.push(Value::matrix(r));
    }
    Ok(out)
}

pub fn fn_sph2cart(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() != 3 {
        return Err(MatroError::arity("sph2cart", "3 (az, el, r)", args.len()));
    }
    let az = args[0].to_dense_matrix()?;
    let el = args[1].to_dense_matrix()?;
    let r = args[2].to_dense_matrix()?;
    let rcos = r.elementwise(&el, |rv, ev| rv * ev.cos())?;
    let x = rcos.elementwise(&az, |rc, a| rc * a.cos())?;
    let y = rcos.elementwise(&az, |rc, a| rc * a.sin())?;
    let z = r.elementwise(&el, |rv, ev| rv * ev.sin())?;
    let mut out = vec![Value::matrix(x)];
    if nargout >= 2 {
        out.push(Value::matrix(y));
    }
    if nargout >= 3 {
        out.push(Value::matrix(z));
    }
    Ok(out)
}

pub fn fn_unwrap(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity("unwrap", "1 to 3", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let cutoff = if args.len() >= 2 && !args[1].as_matrix()?.is_empty() {
        args[1].as_scalar()?
    } else {
        PI
    };
    let dim = if args.len() == 3 {
        as_usize(&args[2], "unwrap")?
    } else if m.rows() == 1 && m.cols() > 1 {
        2
    } else {
        1
    };
    let unwrap_vec = |xs: &[f64]| -> Vec<f64> {
        if xs.is_empty() {
            return Vec::new();
        }
        let mut out = vec![0.0; xs.len()];
        out[0] = xs[0];
        let mut offset = 0.0;
        let two_pi = 2.0 * PI;
        for i in 1..xs.len() {
            let dp = xs[i] - xs[i - 1];
            if dp > cutoff {
                offset -= two_pi * ((dp + PI) / two_pi).floor();
            } else if dp < -cutoff {
                offset += two_pi * ((-dp + PI) / two_pi).floor();
            }
            out[i] = xs[i] + offset;
        }
        out
    };
    let mut res = m.clone();
    if dim == 1 {
        for c in 0..m.cols() {
            let col: Vec<f64> = (0..m.rows()).map(|r| m.get(r, c)).collect();
            let u = unwrap_vec(&col);
            for (r, &v) in u.iter().enumerate() {
                res.set(r, c, v);
            }
        }
    } else {
        for r in 0..m.rows() {
            let row: Vec<f64> = (0..m.cols()).map(|c| m.get(r, c)).collect();
            let u = unwrap_vec(&row);
            for (c, &v) in u.iter().enumerate() {
                res.set(r, c, v);
            }
        }
    }
    ok1(Value::matrix(res))
}

pub fn wrap_to_pi(x: f64) -> f64 {
    let two_pi = 2.0 * PI;
    let y = (x + PI).rem_euclid(two_pi) - PI;
    if y == -PI && x > 0.0 { PI } else { y }
}

pub fn wrap_to_2pi(x: f64) -> f64 {
    let two_pi = 2.0 * PI;
    let y = x.rem_euclid(two_pi);
    if y == 0.0 && x > 0.0 { two_pi } else { y }
}

pub fn wrap_to_180(x: f64) -> f64 {
    let y = (x + 180.0).rem_euclid(360.0) - 180.0;
    if y == -180.0 && x > 0.0 { 180.0 } else { y }
}

pub fn wrap_to_360(x: f64) -> f64 {
    let y = x.rem_euclid(360.0);
    if y == 0.0 && x > 0.0 { 360.0 } else { y }
}

// ---------------------------------------------------------------------------
// Random: randi, randperm
// ---------------------------------------------------------------------------

pub fn fn_randi(args: &[Value], rng: &mut Rng) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("randi", ">= 1 (imax | [imin, imax], sz...)", 0));
    }
    let lim = args[0].as_matrix()?;
    let (imin, imax) = if lim.len() == 1 {
        (1i64, lim.data()[0].round() as i64)
    } else if lim.len() == 2 {
        (lim.data()[0].round() as i64, lim.data()[1].round() as i64)
    } else {
        return Err(MatroError::runtime("randi: first argument must be scalar imax or [imin, imax]"));
    };
    if imax < imin {
        return Err(MatroError::runtime("randi: imax must be >= imin"));
    }
    let dims = parse_size_args(&args[1..], "randi")?;
    let total: usize = dims.iter().copied().product();
    let span = (imax - imin + 1) as u64;
    let mut data = Vec::with_capacity(total);
    for _ in 0..total {
        let v = imin + (rng.next_u64() % span) as i64;
        data.push(v as f64);
    }
    ok1(Value::matrix(Matrix::from_nd_col_major(dims, data)?))
}

pub fn fn_randperm(args: &[Value], rng: &mut Rng) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("randperm", "1 or 2 (n [, k])", args.len()));
    }
    let n = as_usize(&args[0], "randperm")?;
    let k = if args.len() == 2 {
        as_usize(&args[1], "randperm")?
    } else {
        n
    };
    if k > n {
        return Err(MatroError::runtime("randperm: k cannot exceed n"));
    }
    let mut perm: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    for i in 0..k {
        let j = i + (rng.next_u64() as usize) % (n - i);
        perm.swap(i, j);
    }
    perm.truncate(k);
    ok1(Value::matrix(Matrix::row_vec(perm)))
}

// ---------------------------------------------------------------------------
// Array / Index / Sparse Construction Utilities
// ---------------------------------------------------------------------------

pub fn fn_blkdiag(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() {
        return ok1(Value::empty());
    }
    let any_sparse = args.iter().any(|a| matches!(a, Value::Sparse(_)));
    if any_sparse {
        let mut r_off = 0usize;
        let mut c_off = 0usize;
        let mut triplets = Vec::new();
        for a in args {
            let sp = match a {
                Value::Sparse(s) => (**s).clone(),
                other => SparseMatrix::from_dense(&other.to_dense_matrix()?),
            };
            for (r, c, v) in sp.triplets() {
                triplets.push((r + r_off, c + c_off, v));
            }
            r_off += sp.rows();
            c_off += sp.cols();
        }
        return ok1(Value::sparse(SparseMatrix::from_triplets(r_off, c_off, &triplets)?));
    }
    let mats: Vec<Matrix> = args
        .iter()
        .map(|a| a.to_dense_matrix())
        .collect::<Result<_>>()?;
    let total_r: usize = mats.iter().map(|m| m.rows()).sum();
    let total_c: usize = mats.iter().map(|m| m.cols()).sum();
    let mut out = Matrix::zeros(total_r, total_c);
    let mut r_off = 0usize;
    let mut c_off = 0usize;
    for m in mats {
        for c in 0..m.cols() {
            for r in 0..m.rows() {
                out.set_c(r_off + r, c_off + c, m.get_c(r, c));
            }
        }
        r_off += m.rows();
        c_off += m.cols();
    }
    ok1(Value::matrix(out.normalize()))
}

pub fn fn_circshift(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity("circshift", "2 or 3 (A, K [, dim])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    if m.is_empty() {
        return ok1(Value::matrix(m));
    }
    let (kr, kc) = if args.len() == 3 {
        let k = args[1].as_scalar()?.round() as isize;
        let dim = as_usize(&args[2], "circshift")?;
        if dim == 1 {
            (k, 0)
        } else if dim == 2 {
            (0, k)
        } else {
            (0, 0)
        }
    } else {
        let kv = args[1].as_matrix()?;
        if kv.len() == 1 {
            let k = kv.data()[0].round() as isize;
            if m.rows() == 1 && m.cols() > 1 {
                (0, k)
            } else {
                (k, 0)
            }
        } else {
            (
                kv.data()[0].round() as isize,
                kv.data().get(1).copied().unwrap_or(0.0).round() as isize,
            )
        }
    };
    let (nr, nc) = (m.rows(), m.cols());
    let mut out = Matrix::zeros(nr, nc);
    for c in 0..nc {
        let dst_c = (c as isize + kc).rem_euclid(nc as isize) as usize;
        for r in 0..nr {
            let dst_r = (r as isize + kr).rem_euclid(nr as isize) as usize;
            out.set_c(dst_r, dst_c, m.get_c(r, c));
        }
    }
    ok1(Value::matrix(out.normalize()))
}

pub fn fn_flip(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("flip", "1 or 2 (A [, dim])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let dim = parse_dim_opt(args, 1, "flip", &m)?;
    let (nr, nc) = (m.rows(), m.cols());
    let mut out = Matrix::zeros(nr, nc);
    for c in 0..nc {
        let src_c = if dim == 2 { nc - 1 - c } else { c };
        for r in 0..nr {
            let src_r = if dim == 1 { nr - 1 - r } else { r };
            out.set_c(r, c, m.get_c(src_r, src_c));
        }
    }
    ok1(Value::matrix(out.normalize()))
}

pub fn fn_repelem(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity("repelem", "2 or 3 (A, n | r, c)", args.len()));
    }
    let a = args[0].to_dense_matrix()?;
    if args.len() == 2 {
        let rep = args[1].as_matrix()?;
        let is_row = a.rows() == 1 && a.cols() > 1;
        let mut vals = Vec::new();
        for i in 0..a.len() {
            let cnt = if rep.len() == 1 {
                rep.data()[0].max(0.0).round() as usize
            } else if rep.len() == a.len() {
                rep.data()[i].max(0.0).round() as usize
            } else {
                return Err(MatroError::dimension("repelem: repeat count vector length mismatch"));
            };
            let v = a.get_linear_c(i);
            for _ in 0..cnt {
                vals.push(v);
            }
        }
        let out = if is_row {
            Matrix::complex_row_vec(vals)
        } else {
            Matrix::complex_col_vec(vals)
        };
        return ok1(Value::matrix(out.normalize()));
    }
    let rr = args[1].as_matrix()?;
    let cc = args[2].as_matrix()?;
    let row_counts: Vec<usize> = if rr.len() == 1 {
        vec![rr.data()[0].max(0.0).round() as usize; a.rows()]
    } else if rr.len() == a.rows() {
        rr.data().iter().map(|&x| x.max(0.0).round() as usize).collect()
    } else {
        return Err(MatroError::dimension("repelem: row repeat count length mismatch"));
    };
    let col_counts: Vec<usize> = if cc.len() == 1 {
        vec![cc.data()[0].max(0.0).round() as usize; a.cols()]
    } else if cc.len() == a.cols() {
        cc.data().iter().map(|&x| x.max(0.0).round() as usize).collect()
    } else {
        return Err(MatroError::dimension("repelem: column repeat count length mismatch"));
    };
    let out_r: usize = row_counts.iter().sum();
    let out_c: usize = col_counts.iter().sum();
    let mut out = Matrix::zeros(out_r, out_c);
    let mut dc = 0usize;
    for (c, &cn) in col_counts.iter().enumerate() {
        for _ in 0..cn {
            let mut dr = 0usize;
            for (r, &rn) in row_counts.iter().enumerate() {
                let v = a.get_c(r, c);
                for _ in 0..rn {
                    out.set_c(dr, dc, v);
                    dr += 1;
                }
            }
            dc += 1;
        }
    }
    ok1(Value::matrix(out.normalize()))
}

pub fn fn_accumarray(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 5 {
        return Err(MatroError::arity("accumarray", "2 to 5 (subs, val [, sz])", args.len()));
    }
    let subs = args[0].as_matrix()?;
    let val = args[1].as_matrix()?;
    let is_2d = subs.cols() == 2 && subs.rows() > 0 && !(subs.rows() == 1 && val.len() == 2);
    let n = if is_2d { subs.rows() } else { subs.len() };
    let val_scalar = val.as_scalar();
    if val_scalar.is_none() && val.len() != n {
        return Err(MatroError::dimension("accumarray: val length must match subs"));
    }
    let fill = if args.len() >= 5 {
        args[4].as_scalar()?
    } else {
        0.0
    };
    if is_2d {
        let mut max_r = 0usize;
        let mut max_c = 0usize;
        let mut coords = Vec::with_capacity(n);
        for i in 0..n {
            let r = subs.get(i, 0).round() as isize;
            let c = subs.get(i, 1).round() as isize;
            if r < 1 || c < 1 {
                return Err(MatroError::index("accumarray: subscripts must be >= 1"));
            }
            let (r0, c0) = ((r - 1) as usize, (c - 1) as usize);
            max_r = max_r.max(r0 + 1);
            max_c = max_c.max(c0 + 1);
            coords.push((r0, c0));
        }
        let (nr, nc) = if args.len() >= 3 && !args[2].as_matrix()?.is_empty() {
            let sz = args[2].as_matrix()?;
            (
                sz.data()[0].round() as usize,
                sz.data().get(1).copied().unwrap_or(1.0).round() as usize,
            )
        } else {
            (max_r, max_c)
        };
        let mut out = Matrix::zeros(nr, nc);
        let mut touched = vec![false; nr * nc];
        for (i, &(r0, c0)) in coords.iter().enumerate() {
            if r0 >= nr || c0 >= nc {
                return Err(MatroError::index("accumarray: subscript exceeds sz"));
            }
            let v = val_scalar.unwrap_or_else(|| val.data()[i]);
            let prev = if touched[c0 * nr + r0] { out.get(r0, c0) } else { 0.0 };
            touched[c0 * nr + r0] = true;
            out.set(r0, c0, prev + v);
        }
        if fill != 0.0 {
            for c in 0..nc {
                for r in 0..nr {
                    if !touched[c * nr + r] {
                        out.set(r, c, fill);
                    }
                }
            }
        }
        ok1(Value::matrix(out))
    } else {
        let mut max_i = 0usize;
        let mut idxs = Vec::with_capacity(n);
        for i in 0..n {
            let idx = subs.data()[i].round() as isize;
            if idx < 1 {
                return Err(MatroError::index("accumarray: subscripts must be >= 1"));
            }
            let i0 = (idx - 1) as usize;
            max_i = max_i.max(i0 + 1);
            idxs.push(i0);
        }
        let len = if args.len() >= 3 && !args[2].as_matrix()?.is_empty() {
            args[2].as_matrix()?.data()[0].round() as usize
        } else {
            max_i
        };
        let mut out = vec![0.0; len];
        let mut touched = vec![false; len];
        for (i, &i0) in idxs.iter().enumerate() {
            if i0 >= len {
                return Err(MatroError::index("accumarray: subscript exceeds sz"));
            }
            let v = val_scalar.unwrap_or_else(|| val.data()[i]);
            out[i0] += v;
            touched[i0] = true;
        }
        if fill != 0.0 {
            for i in 0..len {
                if !touched[i] {
                    out[i] = fill;
                }
            }
        }
        ok1(Value::matrix(Matrix::col_vec(out)))
    }
}

pub fn fn_histcounts(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("histcounts", "1 or 2 (X [, nbins | edges])", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let finite: Vec<f64> = x.data().iter().copied().filter(|v| v.is_finite()).collect();
    let edges: Vec<f64> = if args.len() == 2 {
        let e = args[1].as_matrix()?;
        if e.len() == 1 {
            let nb = e.data()[0].max(1.0).round() as usize;
            let xmin = finite.iter().copied().fold(f64::INFINITY, f64::min);
            let xmax = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let (lo, hi) = if !xmin.is_finite() {
                (0.0, 1.0)
            } else if (xmax - xmin).abs() < 1e-15 {
                (xmin - 0.5, xmax + 0.5)
            } else {
                (xmin, xmax)
            };
            (0..=nb)
                .map(|i| lo + (hi - lo) * (i as f64) / (nb as f64))
                .collect()
        } else {
            e.data().to_vec()
        }
    } else {
        let nb = ((finite.len() as f64).sqrt().ceil() as usize).clamp(1, 50);
        let xmin = finite.iter().copied().fold(f64::INFINITY, f64::min);
        let xmax = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let (lo, hi) = if !xmin.is_finite() {
            (0.0, 1.0)
        } else if (xmax - xmin).abs() < 1e-15 {
            (xmin - 0.5, xmax + 0.5)
        } else {
            (xmin, xmax)
        };
        (0..=nb)
            .map(|i| lo + (hi - lo) * (i as f64) / (nb as f64))
            .collect()
    };
    if edges.len() < 2 {
        return Err(MatroError::dimension("histcounts: edges must have at least 2 elements"));
    }
    let nb = edges.len() - 1;
    let mut counts = vec![0.0; nb];
    let mut bins = vec![0.0; x.len()];
    for (idx, &v) in x.data().iter().enumerate() {
        if !v.is_finite() || v < edges[0] || v > edges[nb] {
            continue;
        }
        let b = if v == edges[nb] {
            nb - 1
        } else {
            let pos = edges.partition_point(|&e| e <= v);
            pos.saturating_sub(1).min(nb - 1)
        };
        counts[b] += 1.0;
        bins[idx] = (b + 1) as f64;
    }
    let mut out = vec![Value::matrix(Matrix::row_vec(counts))];
    if nargout >= 2 {
        out.push(Value::matrix(Matrix::row_vec(edges)));
    }
    if nargout >= 3 {
        out.push(Value::matrix(Matrix::from_nd_col_major(x.dims(), bins)?));
    }
    Ok(out)
}

pub fn fn_discretize(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("discretize", "2 (X, edges)", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let edges = args[1].as_matrix()?.data();
    if edges.len() < 2 {
        return Err(MatroError::dimension("discretize: edges must have >= 2 elements"));
    }
    let nb = edges.len() - 1;
    let mut bins = vec![f64::NAN; x.len()];
    for (idx, &v) in x.data().iter().enumerate() {
        if !v.is_finite() || v < edges[0] || v > edges[nb] {
            continue;
        }
        let b = if v == edges[nb] {
            nb - 1
        } else {
            edges.partition_point(|&e| e <= v).saturating_sub(1).min(nb - 1)
        };
        bins[idx] = (b + 1) as f64;
    }
    ok1(Value::matrix(Matrix::from_nd_col_major(x.dims(), bins)?))
}

pub fn fn_ipermute(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("ipermute", "2 (A, order)", args.len()));
    }
    let order = args[1].as_matrix()?;
    let n = order.len();
    let mut inv = vec![0.0; n];
    for (i, &p) in order.data().iter().enumerate() {
        let p0 = p.round() as isize - 1;
        if p0 < 0 || (p0 as usize) >= n {
            return Err(MatroError::dimension("ipermute: invalid permutation order"));
        }
        inv[p0 as usize] = (i + 1) as f64;
    }
    crate::extras::fn_permute(&[args[0].clone(), Value::matrix(Matrix::row_vec(inv))])
}

pub fn fn_horzcat(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() {
        return ok1(Value::empty());
    }
    if args.iter().all(|a| matches!(a, Value::String(_))) {
        let s: String = args.iter().filter_map(|a| a.as_str().ok()).collect();
        return ok1(Value::String(s));
    }
    let mats: Vec<Matrix> = args.iter().map(|a| a.to_dense_matrix()).collect::<Result<_>>()?;
    let refs: Vec<&Matrix> = mats.iter().collect();
    ok1(Value::matrix(Matrix::hcat(&refs)?))
}

pub fn fn_vertcat(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() {
        return ok1(Value::empty());
    }
    let mats: Vec<Matrix> = args.iter().map(|a| a.to_dense_matrix()).collect::<Result<_>>()?;
    let refs: Vec<&Matrix> = mats.iter().collect();
    ok1(Value::matrix(Matrix::vcat(&refs)?))
}

pub fn fn_spdiags(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    match args.len() {
        1 | 2 => {
            // Extract diagonals from matrix A
            let sp = match &args[0] {
                Value::Sparse(s) => (**s).clone(),
                other => SparseMatrix::from_dense(&other.to_dense_matrix()?),
            };
            let (m, n) = (sp.rows(), sp.cols());
            let p = m.min(n);
            let diags: Vec<isize> = if args.len() == 2 {
                args[1]
                    .as_matrix()?
                    .data()
                    .iter()
                    .map(|&x| x.round() as isize)
                    .collect()
            } else {
                let mut ds: Vec<isize> = sp
                    .triplets()
                    .into_iter()
                    .map(|(r, c, _)| c as isize - r as isize)
                    .collect();
                ds.sort_unstable();
                ds.dedup();
                ds
            };
            let mut b = Matrix::zeros(p, diags.len());
            for (k, &d) in diags.iter().enumerate() {
                for col_idx in 0..p {
                    let (r, c) = if m >= n {
                        (col_idx as isize - d, col_idx as isize)
                    } else {
                        (col_idx as isize, col_idx as isize + d)
                    };
                    if r >= 0 && (r as usize) < m && c >= 0 && (c as usize) < n {
                        b.set(col_idx, k, sp.get(r as usize, c as usize));
                    }
                }
            }
            if nargout >= 2 {
                let dv: Vec<f64> = diags.into_iter().map(|d| d as f64).collect();
                Ok(vec![Value::matrix(b), Value::matrix(Matrix::col_vec(dv))])
            } else {
                ok1(Value::matrix(b))
            }
        }
        4 => {
            // Create sparse matrix from diagonals: spdiags(B, d, m, n)
            let b = args[0].to_dense_matrix()?;
            let d_vec = args[1].as_matrix()?.data();
            let m = as_usize(&args[2], "spdiags")?;
            let n = as_usize(&args[3], "spdiags")?;
            if b.cols() != d_vec.len() {
                return Err(MatroError::dimension(
                    "spdiags(B, d, m, n): columns of B must equal length(d)",
                ));
            }
            let mut triplets = Vec::new();
            for (k, &df) in d_vec.iter().enumerate() {
                let d = df.round() as isize;
                // MATLAB rule: for m >= n, B(j, k) goes to (j - d, j); for m < n, B(i, k) goes to (i, i + d)
                for idx in 0..b.rows() {
                    let (r, c) = if m >= n {
                        (idx as isize - d, idx as isize)
                    } else {
                        (idx as isize, idx as isize + d)
                    };
                    if r >= 0 && (r as usize) < m && c >= 0 && (c as usize) < n {
                        let v = b.get(idx, k);
                        if v != 0.0 {
                            triplets.push((r as usize, c as usize, v));
                        }
                    }
                }
            }
            ok1(Value::sparse(SparseMatrix::from_triplets(m, n, &triplets)?))
        }
        n => Err(MatroError::arity("spdiags", "1, 2, or 4", n)),
    }
}

pub fn fn_spones(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("spones", "1", args.len()));
    }
    match &args[0] {
        Value::Sparse(s) => ok1(Value::sparse(s.spones())),
        other => ok1(Value::sparse(SparseMatrix::from_dense(&other.to_dense_matrix()?).spones())),
    }
}

pub fn fn_spalloc(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 3 {
        return Err(MatroError::arity("spalloc", "3 (m, n, nzmax)", args.len()));
    }
    let m = as_usize(&args[0], "spalloc")?;
    let n = as_usize(&args[1], "spalloc")?;
    let _ = as_usize(&args[2], "spalloc")?;
    ok1(Value::sparse(SparseMatrix::zeros(m, n)))
}

pub fn fn_nonzeros(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("nonzeros", "1", args.len()));
    }
    match &args[0] {
        Value::Sparse(s) => ok1(Value::matrix(Matrix::col_vec(s.nonzeros()))),
        other => {
            let m = other.to_dense_matrix()?;
            let nz: Vec<Complex> = (0..m.len())
                .map(|i| m.get_linear_c(i))
                .filter(|c| !c.is_zero())
                .collect();
            ok1(Value::matrix(Matrix::complex_col_vec(nz).normalize()))
        }
    }
}

// ---------------------------------------------------------------------------
// Linear Algebra & Numerical Calculus
// ---------------------------------------------------------------------------

pub fn fn_pinv(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("pinv", "1 or 2 (A [, tol])", args.len()));
    }
    let a = args[0].to_dense_matrix()?;
    if a.is_empty() {
        return ok1(Value::matrix(Matrix::zeros(a.cols(), a.rows())));
    }
    let (u, s, v) = a.svd()?;
    let k = s.rows().min(s.cols());
    let s_max = (0..k).map(|i| s.get(i, i).abs()).fold(0.0f64, f64::max);
    let tol = if args.len() == 2 {
        args[1].as_scalar()?
    } else {
        (a.rows().max(a.cols()) as f64) * f64::EPSILON * s_max
    };
    let mut s_inv = Matrix::zeros(s.cols(), s.rows());
    for i in 0..k {
        let sv = s.get(i, i);
        if sv.abs() > tol {
            s_inv.set(i, i, 1.0 / sv);
        }
    }
    let res = v.matmul(&s_inv)?.matmul(&u.ctranspose())?;
    ok1(Value::matrix(res.normalize()))
}

pub fn fn_null_orth(args: &[Value], is_null: bool) -> Result<Vec<Value>> {
    let name = if is_null { "null" } else { "orth" };
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity(name, "1 or 2 (A [, tol])", args.len()));
    }
    let a = args[0].to_dense_matrix()?;
    let (u, s, v) = a.svd()?;
    let k = s.rows().min(s.cols());
    let s_max = (0..k).map(|i| s.get(i, i).abs()).fold(0.0f64, f64::max);
    let tol = if args.len() == 2 {
        args[1].as_scalar()?
    } else {
        (a.rows().max(a.cols()) as f64) * f64::EPSILON * s_max
    };
    let rank = (0..k).filter(|&i| s.get(i, i).abs() > tol).count();
    if is_null {
        let ncols = v.cols().saturating_sub(rank);
        let mut out = Matrix::zeros(v.rows(), ncols);
        for c in 0..ncols {
            for r in 0..v.rows() {
                out.set_c(r, c, v.get_c(r, rank + c));
            }
        }
        ok1(Value::matrix(out.normalize()))
    } else {
        let mut out = Matrix::zeros(u.rows(), rank);
        for c in 0..rank {
            for r in 0..u.rows() {
                out.set_c(r, c, u.get_c(r, c));
            }
        }
        ok1(Value::matrix(out.normalize()))
    }
}

pub fn fn_rref(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("rref", "1 or 2 (A [, tol])", args.len()));
    }
    let mut a = args[0].to_dense_matrix()?;
    let (m, n) = (a.rows(), a.cols());
    let max_norm = (0..m)
        .map(|r| (0..n).map(|c| a.get(r, c).abs()).sum::<f64>())
        .fold(0.0f64, f64::max);
    let tol = if args.len() == 2 {
        args[1].as_scalar()?
    } else {
        (m.max(n) as f64) * f64::EPSILON * max_norm.max(1.0)
    };
    let mut pivots = Vec::new();
    let mut row = 0usize;
    for col in 0..n {
        if row >= m {
            break;
        }
        let mut piv_r = row;
        let mut piv_v = a.get(row, col).abs();
        for r in (row + 1)..m {
            let v = a.get(r, col).abs();
            if v > piv_v {
                piv_v = v;
                piv_r = r;
            }
        }
        if piv_v <= tol {
            for r in row..m {
                a.set(r, col, 0.0);
            }
            continue;
        }
        if piv_r != row {
            for c in col..n {
                let tmp = a.get(row, c);
                a.set(row, c, a.get(piv_r, c));
                a.set(piv_r, c, tmp);
            }
        }
        let pv = a.get(row, col);
        for c in col..n {
            a.set(row, c, a.get(row, c) / pv);
        }
        for r in 0..m {
            if r != row {
                let f = a.get(r, col);
                if f != 0.0 {
                    for c in col..n {
                        a.set(r, c, a.get(r, c) - f * a.get(row, c));
                    }
                }
            }
        }
        pivots.push((col + 1) as f64);
        row += 1;
    }
    if nargout >= 2 {
        Ok(vec![Value::matrix(a), Value::matrix(Matrix::row_vec(pivots))])
    } else {
        ok1(Value::matrix(a))
    }
}

pub fn fn_cross(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity("cross", "2 or 3 (A, B [, dim])", args.len()));
    }
    let a = args[0].to_dense_matrix()?;
    let b = args[1].to_dense_matrix()?;
    if a.dims() != b.dims() {
        return Err(MatroError::dimension("cross: A and B must have the same size"));
    }
    let dim = if args.len() == 3 {
        as_usize(&args[2], "cross")?
    } else if a.rows() == 3 {
        1
    } else if a.cols() == 3 {
        2
    } else {
        return Err(MatroError::dimension("cross: A and B must have at least one dimension of length 3"));
    };
    let mut out = Matrix::zeros(a.rows(), a.cols());
    if dim == 1 {
        if a.rows() != 3 {
            return Err(MatroError::dimension("cross: dimension 1 must have length 3"));
        }
        for c in 0..a.cols() {
            let (a0, a1, a2) = (a.get_c(0, c), a.get_c(1, c), a.get_c(2, c));
            let (b0, b1, b2) = (b.get_c(0, c), b.get_c(1, c), b.get_c(2, c));
            out.set_c(0, c, a1 * b2 - a2 * b1);
            out.set_c(1, c, a2 * b0 - a0 * b2);
            out.set_c(2, c, a0 * b1 - a1 * b0);
        }
    } else if dim == 2 {
        if a.cols() != 3 {
            return Err(MatroError::dimension("cross: dimension 2 must have length 3"));
        }
        for r in 0..a.rows() {
            let (a0, a1, a2) = (a.get_c(r, 0), a.get_c(r, 1), a.get_c(r, 2));
            let (b0, b1, b2) = (b.get_c(r, 0), b.get_c(r, 1), b.get_c(r, 2));
            out.set_c(r, 0, a1 * b2 - a2 * b1);
            out.set_c(r, 1, a2 * b0 - a0 * b2);
            out.set_c(r, 2, a0 * b1 - a1 * b0);
        }
    } else {
        return Err(MatroError::runtime("cross: dim must be 1 or 2"));
    }
    ok1(Value::matrix(out.normalize()))
}

pub fn fn_vecnorm(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity("vecnorm", "1 to 3 (A [, p [, dim]])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let p = if args.len() >= 2 && !args[1].as_matrix()?.is_empty() {
        args[1].as_scalar()?
    } else {
        2.0
    };
    let dim = parse_dim_opt(args, 2, "vecnorm", &m)?;
    let calc = |vals: &[Complex]| -> f64 {
        if p.is_infinite() && p > 0.0 {
            vals.iter().map(|c| c.abs()).fold(0.0f64, f64::max)
        } else if p.is_infinite() && p < 0.0 {
            vals.iter().map(|c| c.abs()).fold(f64::INFINITY, f64::min)
        } else {
            vals.iter().map(|c| c.abs().powf(p)).sum::<f64>().powf(1.0 / p)
        }
    };
    if dim == 1 {
        let out: Vec<f64> = (0..m.cols())
            .map(|c| {
                let col: Vec<Complex> = (0..m.rows()).map(|r| m.get_c(r, c)).collect();
                calc(&col)
            })
            .collect();
        ok1(Value::matrix(Matrix::row_vec(out)))
    } else {
        let out: Vec<f64> = (0..m.rows())
            .map(|r| {
                let row: Vec<Complex> = (0..m.cols()).map(|c| m.get_c(r, c)).collect();
                calc(&row)
            })
            .collect();
        ok1(Value::matrix(Matrix::col_vec(out)))
    }
}

pub fn fn_rms(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("rms", "1 or 2 (x [, dim])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let dim = parse_dim_opt(args, 1, "rms", &m)?;
    if dim == 1 {
        let out: Vec<f64> = (0..m.cols())
            .map(|c| {
                let ss: f64 = (0..m.rows()).map(|r| m.get_c(r, c).abs().powi(2)).sum();
                (ss / (m.rows().max(1) as f64)).sqrt()
            })
            .collect();
        ok1(Value::matrix(Matrix::row_vec(out)))
    } else {
        let out: Vec<f64> = (0..m.rows())
            .map(|r| {
                let ss: f64 = (0..m.cols()).map(|c| m.get_c(r, c).abs().powi(2)).sum();
                (ss / (m.cols().max(1) as f64)).sqrt()
            })
            .collect();
        ok1(Value::matrix(Matrix::col_vec(out)))
    }
}

pub fn fn_normalize(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity("normalize", "1 to 3 (A [, dim | method])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let mut dim = if m.rows() == 1 && m.cols() > 1 { 2 } else { 1 };
    let mut method = "zscore".to_string();
    for a in &args[1..] {
        if let Some(s) = value_as_text(a) {
            method = s.trim().to_ascii_lowercase();
        } else {
            dim = as_usize(a, "normalize")?;
        }
    }
    let norm_vec = |xs: &[f64]| -> Vec<f64> {
        let n = xs.len() as f64;
        if xs.is_empty() {
            return Vec::new();
        }
        match method.as_str() {
            "norm" => {
                let l2 = xs.iter().map(|x| x * x).sum::<f64>().sqrt();
                if l2 == 0.0 { xs.to_vec() } else { xs.iter().map(|x| x / l2).collect() }
            }
            "range" => {
                let lo = xs.iter().copied().fold(f64::INFINITY, f64::min);
                let hi = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let span = hi - lo;
                if span == 0.0 { vec![0.0; xs.len()] } else { xs.iter().map(|x| (x - lo) / span).collect() }
            }
            "center" => {
                let mu = xs.iter().sum::<f64>() / n;
                xs.iter().map(|x| x - mu).collect()
            }
            _ => {
                let mu = xs.iter().sum::<f64>() / n;
                let denom = (n - 1.0).max(1.0);
                let sd = (xs.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / denom).sqrt();
                if sd == 0.0 { vec![0.0; xs.len()] } else { xs.iter().map(|x| (x - mu) / sd).collect() }
            }
        }
    };
    let mut out = m.clone();
    if dim == 1 {
        for c in 0..m.cols() {
            let col: Vec<f64> = (0..m.rows()).map(|r| m.get(r, c)).collect();
            let nv = norm_vec(&col);
            for (r, &v) in nv.iter().enumerate() {
                out.set(r, c, v);
            }
        }
    } else {
        for r in 0..m.rows() {
            let row: Vec<f64> = (0..m.cols()).map(|c| m.get(r, c)).collect();
            let nv = norm_vec(&row);
            for (c, &v) in nv.iter().enumerate() {
                out.set(r, c, v);
            }
        }
    }
    ok1(Value::matrix(out))
}

pub fn fn_rescale(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity("rescale", "1 to 3 (A [, l, u])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let (l, u) = if args.len() == 3 {
        (args[1].as_scalar()?, args[2].as_scalar()?)
    } else {
        (0.0, 1.0)
    };
    let lo = m.data().iter().copied().fold(f64::INFINITY, f64::min);
    let hi = m.data().iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = hi - lo;
    let out = if span == 0.0 {
        m.map(|_| l)
    } else {
        m.map(|x| l + (x - lo) * (u - l) / span)
    };
    ok1(Value::matrix(out))
}

pub fn fn_clip(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 3 {
        return Err(MatroError::arity("clip", "3 (X, lower, upper)", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let lo = args[1].as_scalar()?;
    let hi = args[2].as_scalar()?;
    ok1(Value::matrix(x.map(|v| v.clamp(lo, hi))))
}

pub fn fn_trapz(args: &[Value], cumulative: bool) -> Result<Vec<Value>> {
    let name = if cumulative { "cumtrapz" } else { "trapz" };
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity(name, "1 to 3", args.len()));
    }
    let (x_opt, y, dim) = match args.len() {
        1 => {
            let y = args[0].to_dense_matrix()?;
            let d = if y.rows() == 1 && y.cols() > 1 { 2 } else { 1 };
            (None, y, d)
        }
        2 => {
            let a = args[0].to_dense_matrix()?;
            let b = args[1].to_dense_matrix()?;
            if b.len() == 1 && (b.data()[0] == 1.0 || b.data()[0] == 2.0) && a.len() > 1 {
                (None, a, b.data()[0] as usize)
            } else {
                let d = if b.rows() == 1 && b.cols() > 1 { 2 } else { 1 };
                (Some(a), b, d)
            }
        }
        3 => (
            Some(args[0].to_dense_matrix()?),
            args[1].to_dense_matrix()?,
            as_usize(&args[2], name)?,
        ),
        _ => unreachable!(),
    };
    let integrate = |ys: &[f64]| -> Vec<f64> {
        let n = ys.len();
        let mut acc = vec![0.0; n];
        for i in 1..n {
            let dx = match &x_opt {
                Some(xm) if xm.len() == 1 => xm.data()[0],
                Some(xm) => xm.data()[i] - xm.data()[i - 1],
                None => 1.0,
            };
            acc[i] = acc[i - 1] + 0.5 * dx * (ys[i - 1] + ys[i]);
        }
        acc
    };
    if cumulative {
        let mut out = y.clone();
        if dim == 1 {
            for c in 0..y.cols() {
                let col: Vec<f64> = (0..y.rows()).map(|r| y.get(r, c)).collect();
                let res = integrate(&col);
                for (r, &v) in res.iter().enumerate() {
                    out.set(r, c, v);
                }
            }
        } else {
            for r in 0..y.rows() {
                let row: Vec<f64> = (0..y.cols()).map(|c| y.get(r, c)).collect();
                let res = integrate(&row);
                for (c, &v) in res.iter().enumerate() {
                    out.set(r, c, v);
                }
            }
        }
        ok1(Value::matrix(out))
    } else if dim == 1 {
        let vals: Vec<f64> = (0..y.cols())
            .map(|c| {
                let col: Vec<f64> = (0..y.rows()).map(|r| y.get(r, c)).collect();
                integrate(&col).last().copied().unwrap_or(0.0)
            })
            .collect();
        ok1(Value::matrix(Matrix::row_vec(vals)))
    } else {
        let vals: Vec<f64> = (0..y.rows())
            .map(|r| {
                let row: Vec<f64> = (0..y.cols()).map(|c| y.get(r, c)).collect();
                integrate(&row).last().copied().unwrap_or(0.0)
            })
            .collect();
        ok1(Value::matrix(Matrix::col_vec(vals)))
    }
}

pub fn fn_gradient(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity("gradient", "1 to 3 (F [, hx [, hy]])", args.len()));
    }
    let f = args[0].to_dense_matrix()?;
    let diff_1d = |vals: &[f64], spacing: Option<&Matrix>| -> Vec<f64> {
        let n = vals.len();
        if n <= 1 {
            return vec![0.0; n];
        }
        let coords: Vec<f64> = match spacing {
            Some(s) if s.len() == n => s.data().to_vec(),
            Some(s) if s.len() == 1 => (0..n).map(|i| (i as f64) * s.data()[0]).collect(),
            _ => (0..n).map(|i| i as f64).collect(),
        };
        let mut g = vec![0.0; n];
        g[0] = (vals[1] - vals[0]) / (coords[1] - coords[0]);
        g[n - 1] = (vals[n - 1] - vals[n - 2]) / (coords[n - 1] - coords[n - 2]);
        for i in 1..(n - 1) {
            g[i] = (vals[i + 1] - vals[i - 1]) / (coords[i + 1] - coords[i - 1]);
        }
        g
    };

    // 1D vector
    if f.rows() == 1 || f.cols() == 1 {
        let sp = args.get(1).map(|v| v.as_matrix()).transpose()?;
        let g = diff_1d(f.data(), sp);
        let mat = if f.rows() == 1 {
            Matrix::row_vec(g)
        } else {
            Matrix::col_vec(g)
        };
        return ok1(Value::matrix(mat));
    }

    // 2D matrix: [FX, FY] = gradient(F, hx, hy)
    let hx = args.get(1).map(|v| v.as_matrix()).transpose()?;
    let hy = args.get(2).map(|v| v.as_matrix()).transpose()?.or(hx);
    let mut fx = Matrix::zeros(f.rows(), f.cols());
    for r in 0..f.rows() {
        let row: Vec<f64> = (0..f.cols()).map(|c| f.get(r, c)).collect();
        let g = diff_1d(&row, hx);
        for (c, &v) in g.iter().enumerate() {
            fx.set(r, c, v);
        }
    }
    if nargout <= 1 {
        return ok1(Value::matrix(fx));
    }
    let mut fy = Matrix::zeros(f.rows(), f.cols());
    for c in 0..f.cols() {
        let col: Vec<f64> = (0..f.rows()).map(|r| f.get(r, c)).collect();
        let g = diff_1d(&col, hy);
        for (r, &v) in g.iter().enumerate() {
            fy.set(r, c, v);
        }
    }
    Ok(vec![Value::matrix(fx), Value::matrix(fy)])
}

pub fn fn_deconv(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("deconv", "2 (u, v)", args.len()));
    }
    let u = args[0].to_dense_matrix()?;
    let v = args[1].to_dense_matrix()?;
    if v.is_empty() || v.get_linear_c(0).is_zero() {
        return Err(MatroError::runtime("deconv: first element of divisor v must be non-zero"));
    }
    let is_col = u.cols() == 1 && u.rows() > 1;
    let nu = u.len();
    let nv = v.len();
    let mut rem: Vec<Complex> = (0..nu).map(|i| u.get_linear_c(i)).collect();
    if nu < nv {
        let q = Matrix::complex_scalar(Complex::ZERO).normalize();
        let r = if is_col {
            Matrix::complex_col_vec(rem).normalize()
        } else {
            Matrix::complex_row_vec(rem).normalize()
        };
        return if nargout >= 2 {
            Ok(vec![Value::matrix(q), Value::matrix(r)])
        } else {
            ok1(Value::matrix(q))
        };
    }
    let nq = nu - nv + 1;
    let mut q = vec![Complex::ZERO; nq];
    let v0 = v.get_linear_c(0);
    for i in 0..nq {
        let qi = rem[i] / v0;
        q[i] = qi;
        for j in 0..nv {
            rem[i + j] = rem[i + j] - qi * v.get_linear_c(j);
        }
    }
    let q_mat = if is_col {
        Matrix::complex_col_vec(q).normalize()
    } else {
        Matrix::complex_row_vec(q).normalize()
    };
    let r_mat = if is_col {
        Matrix::complex_col_vec(rem).normalize()
    } else {
        Matrix::complex_row_vec(rem).normalize()
    };
    if nargout >= 2 {
        Ok(vec![Value::matrix(q_mat), Value::matrix(r_mat)])
    } else {
        ok1(Value::matrix(q_mat))
    }
}

pub fn fn_polyder(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("polyder", "1 (p)", args.len()));
    }
    let p = args[0].as_matrix()?.data();
    if p.len() <= 1 {
        return ok1(Value::scalar(0.0));
    }
    let n = p.len() - 1;
    let d: Vec<f64> = (0..n).map(|i| p[i] * ((n - i) as f64)).collect();
    ok1(Value::matrix(Matrix::row_vec(d)))
}

pub fn fn_polyint(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("polyint", "1 or 2 (p [, k])", args.len()));
    }
    let p = args[0].as_matrix()?.data();
    let k = if args.len() == 2 { args[1].as_scalar()? } else { 0.0 };
    let n = p.len();
    let mut out = Vec::with_capacity(n + 1);
    for (i, &c) in p.iter().enumerate() {
        out.push(c / ((n - i) as f64));
    }
    out.push(k);
    ok1(Value::matrix(Matrix::row_vec(out)))
}

// ---------------------------------------------------------------------------
// Combinatorics & Number Theory
// ---------------------------------------------------------------------------

pub fn fn_factorial(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("factorial", "1", args.len()));
    }
    let m = args[0].as_matrix()?;
    let out = m.map(|x| {
        if x < 0.0 || (x - x.round()).abs() > 1e-12 {
            f64::NAN
        } else {
            let n = x.round() as u64;
            let mut acc = 1.0f64;
            for i in 2..=n {
                acc *= i as f64;
            }
            acc
        }
    });
    ok1(Value::matrix(out))
}

pub fn fn_nchoosek(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("nchoosek", "2 (n, k | v, k)", args.len()));
    }
    let a = args[0].as_matrix()?;
    let k = as_usize(&args[1], "nchoosek")?;
    if a.len() == 1 {
        let n = as_usize(&args[0], "nchoosek")?;
        if k > n {
            return ok1(Value::scalar(0.0));
        }
        let kk = k.min(n - k);
        let mut res = 1.0f64;
        for i in 1..=kk {
            res = res * ((n - kk + i) as f64) / (i as f64);
        }
        return ok1(Value::scalar(res.round()));
    }
    // Combinations of elements of vector v taken k at a time
    let v = a.data();
    let n = v.len();
    if k > n {
        return ok1(Value::matrix(Matrix::zeros(0, k)));
    }
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut cur = Vec::with_capacity(k);
    fn gen_comb(v: &[f64], start: usize, k: usize, cur: &mut Vec<f64>, out: &mut Vec<Vec<f64>>) {
        if cur.len() == k {
            out.push(cur.clone());
            return;
        }
        let rem = k - cur.len();
        for i in start..=(v.len() - rem) {
            cur.push(v[i]);
            gen_comb(v, i + 1, k, cur, out);
            cur.pop();
        }
    }
    gen_comb(v, 0, k, &mut cur, &mut rows);
    let mut mat = Matrix::zeros(rows.len(), k);
    for (r, row) in rows.iter().enumerate() {
        for (c, &val) in row.iter().enumerate() {
            mat.set(r, c, val);
        }
    }
    ok1(Value::matrix(mat))
}

fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

pub fn fn_gcd_lcm(args: &[Value], is_lcm: bool) -> Result<Vec<Value>> {
    let name = if is_lcm { "lcm" } else { "gcd" };
    if args.len() != 2 {
        return Err(MatroError::arity(name, "2", args.len()));
    }
    let a = args[0].as_matrix()?;
    let b = args[1].as_matrix()?;
    let out = a.elementwise(b, |x, y| {
        let (xi, yi) = (x.round() as i64, y.round() as i64);
        let g = gcd_i64(xi, yi);
        if is_lcm {
            if g == 0 { 0.0 } else { ((xi / g) * yi).abs() as f64 }
        } else {
            g as f64
        }
    })?;
    ok1(Value::matrix(out))
}

fn is_prime_u64(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }
    let mut i = 5u64;
    while i.saturating_mul(i) <= n {
        if n % i == 0 || n % (i + 2) == 0 {
            return false;
        }
        i += 6;
    }
    true
}

pub fn fn_isprime(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("isprime", "1", args.len()));
    }
    let m = args[0].as_matrix()?;
    let out = m
        .map(|x| {
            if x >= 2.0 && (x - x.round()).abs() < 1e-12 && is_prime_u64(x.round() as u64) {
                1.0
            } else {
                0.0
            }
        })
        .with_logical(true);
    ok1(Value::matrix(out))
}

pub fn fn_primes(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("primes", "1", args.len()));
    }
    let n = args[0].as_scalar()?.floor() as isize;
    if n < 2 {
        return ok1(Value::matrix(Matrix::zeros(1, 0)));
    }
    let n = n as usize;
    let mut sieve = vec![true; n + 1];
    sieve[0] = false;
    sieve[1] = false;
    let mut p = 2usize;
    while p * p <= n {
        if sieve[p] {
            let mut m = p * p;
            while m <= n {
                sieve[m] = false;
                m += p;
            }
        }
        p += 1;
    }
    let out: Vec<f64> = (2..=n).filter(|&i| sieve[i]).map(|i| i as f64).collect();
    ok1(Value::matrix(Matrix::row_vec(out)))
}

// ---------------------------------------------------------------------------
// Structure Predicates & Moving Windows / Cumulative Extrema
// ---------------------------------------------------------------------------

pub fn fn_shape_pred(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity(name, "1", args.len()));
    }
    let dims = crate::value_dims(&args[0]);
    let r = dims.first().copied().unwrap_or(1);
    let c = dims.get(1).copied().unwrap_or(1);
    let nd = dims.len();
    let res = match name {
        "isscalar" => nd <= 2 && r == 1 && c == 1,
        "isvector" => nd <= 2 && (r == 1 || c == 1),
        "isrow" => nd <= 2 && r == 1,
        "iscolumn" => nd <= 2 && c == 1,
        "ismatrix" => nd <= 2,
        _ => false,
    };
    ok1(Value::bool(res))
}

pub fn fn_struct_pred(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity(name, "1 or 2", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let (nr, nc) = (m.rows(), m.cols());
    let res = match name {
        "issymmetric" => {
            nr == nc
                && (0..nr).all(|r| {
                    (r..nc).all(|c| {
                        let a = m.get_c(r, c);
                        let b = m.get_c(c, r);
                        (a - b).abs() <= 1e-12
                    })
                })
        }
        "ishermitian" => {
            nr == nc
                && (0..nr).all(|r| {
                    (r..nc).all(|c| {
                        let a = m.get_c(r, c);
                        let b = m.get_c(c, r).conj();
                        (a - b).abs() <= 1e-12
                    })
                })
        }
        "isdiag" => (0..nc).all(|c| (0..nr).all(|r| r == c || m.get_c(r, c).is_zero())),
        "istril" => (0..nc).all(|c| (0..nr).all(|r| r >= c || m.get_c(r, c).is_zero())),
        "istriu" => (0..nc).all(|c| (0..nr).all(|r| r <= c || m.get_c(r, c).is_zero())),
        "issorted" => {
            let desc = args
                .get(1)
                .and_then(value_as_text)
                .is_some_and(|s| s.eq_ignore_ascii_case("descend"));
            let d = m.data();
            d.windows(2).all(|w| if desc { w[0] >= w[1] } else { w[0] <= w[1] })
        }
        _ => false,
    };
    ok1(Value::bool(res))
}

pub fn fn_mov_window(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity(name, "2 or 3 (A, k [, dim])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let k_mat = args[1].as_matrix()?;
    let (kb, kf) = if k_mat.len() == 1 {
        let k = k_mat.data()[0].max(1.0).round() as usize;
        (k / 2, (k - 1) / 2)
    } else {
        (
            k_mat.data()[0].max(0.0).round() as usize,
            k_mat.data()[1].max(0.0).round() as usize,
        )
    };
    let dim = parse_dim_opt(args, 2, name, &m)?;
    let apply_vec = |xs: &[f64]| -> Vec<f64> {
        let n = xs.len();
        let mut out = vec![0.0; n];
        for i in 0..n {
            let lo = i.saturating_sub(kb);
            let hi = (i + kf).min(n - 1);
            let win = &xs[lo..=hi];
            let wn = win.len() as f64;
            out[i] = match name {
                "movsum" => win.iter().sum(),
                "movmean" => win.iter().sum::<f64>() / wn,
                "movmin" => win.iter().copied().fold(f64::INFINITY, f64::min),
                "movmax" => win.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                "movmedian" => {
                    let mut s = win.to_vec();
                    s.sort_by(f64::total_cmp);
                    let mid = s.len() / 2;
                    if s.len() % 2 == 1 { s[mid] } else { 0.5 * (s[mid - 1] + s[mid]) }
                }
                "movstd" | "movvar" => {
                    let mu = win.iter().sum::<f64>() / wn;
                    let denom = (wn - 1.0).max(1.0);
                    let v = if win.len() <= 1 {
                        0.0
                    } else {
                        win.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / denom
                    };
                    if name == "movstd" { v.sqrt() } else { v }
                }
                _ => 0.0,
            };
        }
        out
    };
    let mut out = m.clone();
    if dim == 1 {
        for c in 0..m.cols() {
            let col: Vec<f64> = (0..m.rows()).map(|r| m.get(r, c)).collect();
            let res = apply_vec(&col);
            for (r, &v) in res.iter().enumerate() {
                out.set(r, c, v);
            }
        }
    } else {
        for r in 0..m.rows() {
            let row: Vec<f64> = (0..m.cols()).map(|c| m.get(r, c)).collect();
            let res = apply_vec(&row);
            for (c, &v) in res.iter().enumerate() {
                out.set(r, c, v);
            }
        }
    }
    ok1(Value::matrix(out))
}

pub fn fn_cum_extrema(args: &[Value], is_max: bool) -> Result<Vec<Value>> {
    let name = if is_max { "cummax" } else { "cummin" };
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity(name, "1 or 2", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let dim = parse_dim_opt(args, 1, name, &m)?;
    let mut out = m.clone();
    if dim == 1 {
        for c in 0..m.cols() {
            let mut cur = if is_max { f64::NEG_INFINITY } else { f64::INFINITY };
            for r in 0..m.rows() {
                let v = m.get(r, c);
                cur = if is_max { cur.max(v) } else { cur.min(v) };
                out.set(r, c, cur);
            }
        }
    } else {
        for r in 0..m.rows() {
            let mut cur = if is_max { f64::NEG_INFINITY } else { f64::INFINITY };
            for c in 0..m.cols() {
                let v = m.get(r, c);
                cur = if is_max { cur.max(v) } else { cur.min(v) };
                out.set(r, c, cur);
            }
        }
    }
    ok1(Value::matrix(out))
}

pub fn fn_bounds(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("bounds", "1 or 2 (A [, dim | 'all'])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let all = args
        .get(1)
        .and_then(value_as_text)
        .is_some_and(|s| s.eq_ignore_ascii_case("all"));
    let (lo, hi) = if all || (args.len() == 1 && (m.rows() == 1 || m.cols() == 1)) {
        let mn = m.data().iter().copied().fold(f64::INFINITY, f64::min);
        let mx = m.data().iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (Matrix::scalar(mn), Matrix::scalar(mx))
    } else {
        let dim = parse_dim_opt(args, 1, "bounds", &m)?;
        if dim == 1 {
            let mn: Vec<f64> = (0..m.cols())
                .map(|c| (0..m.rows()).map(|r| m.get(r, c)).fold(f64::INFINITY, f64::min))
                .collect();
            let mx: Vec<f64> = (0..m.cols())
                .map(|c| (0..m.rows()).map(|r| m.get(r, c)).fold(f64::NEG_INFINITY, f64::max))
                .collect();
            (Matrix::row_vec(mn), Matrix::row_vec(mx))
        } else {
            let mn: Vec<f64> = (0..m.rows())
                .map(|r| (0..m.cols()).map(|c| m.get(r, c)).fold(f64::INFINITY, f64::min))
                .collect();
            let mx: Vec<f64> = (0..m.rows())
                .map(|r| (0..m.cols()).map(|c| m.get(r, c)).fold(f64::NEG_INFINITY, f64::max))
                .collect();
            (Matrix::col_vec(mn), Matrix::col_vec(mx))
        }
    };
    if nargout >= 2 {
        Ok(vec![Value::matrix(lo), Value::matrix(hi)])
    } else {
        ok1(Value::matrix(lo))
    }
}

// ---------------------------------------------------------------------------
// Strings, Structs, Cells & Misc Utilities
// ---------------------------------------------------------------------------

pub fn fn_str_cmp_ext(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    let eq = match name {
        "strcmpi" => {
            if args.len() != 2 {
                return Err(MatroError::arity("strcmpi", "2", args.len()));
            }
            let a = value_as_text(&args[0]).unwrap_or_default();
            let b = value_as_text(&args[1]).unwrap_or_default();
            a.eq_ignore_ascii_case(&b)
        }
        "strncmp" | "strncmpi" => {
            if args.len() != 3 {
                return Err(MatroError::arity(name, "3 (s1, s2, n)", args.len()));
            }
            let a = value_as_text(&args[0]).unwrap_or_default();
            let b = value_as_text(&args[1]).unwrap_or_default();
            let n = as_usize(&args[2], name)?;
            let pa: String = a.chars().take(n).collect();
            let pb: String = b.chars().take(n).collect();
            if name == "strncmpi" {
                pa.eq_ignore_ascii_case(&pb)
            } else {
                pa == pb
            }
        }
        _ => false,
    };
    ok1(Value::bool(eq))
}

pub fn fn_strtrim_deblank(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity(name, "1", args.len()));
    }
    let s = value_as_text(&args[0])
        .ok_or_else(|| MatroError::runtime(format!("{name}: expected string")))?;
    let res = if name == "deblank" {
        s.trim_end().to_string()
    } else {
        s.trim().to_string()
    };
    ok1(Value::String(res))
}

pub fn fn_int2str(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("int2str", "1", args.len()));
    }
    let m = args[0].as_matrix()?;
    if m.is_scalar() {
        return ok1(Value::String(format!("{}", m.data()[0].round() as i64)));
    }
    let mut lines = Vec::new();
    for r in 0..m.rows() {
        let row: Vec<String> = (0..m.cols())
            .map(|c| format!("{}", m.get(r, c).round() as i64))
            .collect();
        lines.push(row.join("  "));
    }
    ok1(Value::String(lines.join("\n")))
}

pub fn fn_mat2str(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("mat2str", "1 or 2", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    if m.is_scalar() {
        return ok1(Value::String(format!("{}", m.data()[0])));
    }
    let mut rows = Vec::new();
    for r in 0..m.rows() {
        let row: Vec<String> = (0..m.cols()).map(|c| format!("{}", m.get(r, c))).collect();
        rows.push(row.join(" "));
    }
    ok1(Value::String(format!("[{}]", rows.join(";"))))
}

pub fn fn_base_conv(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    match name {
        "dec2bin" => {
            if args.is_empty() || args.len() > 2 {
                return Err(MatroError::arity("dec2bin", "1 or 2", args.len()));
            }
            let n = args[0].as_scalar()?.max(0.0).round() as u64;
            let width = args.get(1).map(|v| as_usize(v, "dec2bin")).transpose()?.unwrap_or(0);
            ok1(Value::String(format!("{n:0>width$b}")))
        }
        "dec2hex" => {
            if args.is_empty() || args.len() > 2 {
                return Err(MatroError::arity("dec2hex", "1 or 2", args.len()));
            }
            let n = args[0].as_scalar()?.max(0.0).round() as u64;
            let width = args.get(1).map(|v| as_usize(v, "dec2hex")).transpose()?.unwrap_or(0);
            ok1(Value::String(format!("{n:0>width$X}")))
        }
        "bin2dec" => {
            if args.len() != 1 {
                return Err(MatroError::arity("bin2dec", "1", args.len()));
            }
            let s = value_as_text(&args[0]).unwrap_or_default();
            let v = u64::from_str_radix(s.trim(), 2)
                .map_err(|_| MatroError::runtime(format!("bin2dec: invalid binary string '{s}'")))?;
            ok1(Value::scalar(v as f64))
        }
        "hex2dec" => {
            if args.len() != 1 {
                return Err(MatroError::arity("hex2dec", "1", args.len()));
            }
            let s = value_as_text(&args[0]).unwrap_or_default();
            let cleaned = s.trim().trim_start_matches("0x").trim_start_matches("0X");
            let v = u64::from_str_radix(cleaned, 16)
                .map_err(|_| MatroError::runtime(format!("hex2dec: invalid hex string '{s}'")))?;
            ok1(Value::scalar(v as f64))
        }
        "blanks" => {
            if args.len() != 1 {
                return Err(MatroError::arity("blanks", "1", args.len()));
            }
            let n = as_usize(&args[0], "blanks")?;
            ok1(Value::String(" ".repeat(n)))
        }
        _ => unreachable!(),
    }
}

pub fn fn_rmfield(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("rmfield", "2 (s, field)", args.len()));
    }
    let mut to_remove: Vec<String> = Vec::new();
    match &args[1] {
        Value::Cell(c) => {
            for v in &c.data {
                if let Some(s) = value_as_text(v) {
                    to_remove.push(s);
                }
            }
        }
        other => {
            if let Some(s) = value_as_text(other) {
                to_remove.push(s);
            }
        }
    }
    match &args[0] {
        Value::Struct(map) => {
            let mut m = map.clone();
            for f in &to_remove {
                m.remove(f);
            }
            ok1(Value::Struct(m))
        }
        Value::StructArray(sa) => {
            let mut copy = sa.clone();
            for item in &mut copy.items {
                for f in &to_remove {
                    item.remove(f);
                }
            }
            ok1(Value::StructArray(copy))
        }
        other => Err(MatroError::runtime(format!(
            "rmfield: expected struct, got {}",
            other.type_name()
        ))),
    }
}

pub fn fn_orderfields(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("orderfields", "1", 0));
    }
    // BTreeMap already stores fields in lexicographical order
    ok1(args[0].clone())
}

pub fn fn_struct2cell(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("struct2cell", "1", args.len()));
    }
    let Value::Struct(map) = &args[0] else {
        return Err(MatroError::runtime("struct2cell: expected scalar struct"));
    };
    let vals: Vec<Value> = map.values().cloned().collect();
    let n = vals.len();
    ok1(Value::Cell(CellArray {
        rows: n,
        cols: if n == 0 { 0 } else { 1 },
        data: vals,
    }))
}

pub fn fn_cell2mat(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("cell2mat", "1", args.len()));
    }
    let Value::Cell(c) = &args[0] else {
        return Err(MatroError::runtime("cell2mat: expected cell array"));
    };
    if c.data.is_empty() {
        return ok1(Value::empty());
    }
    let mut row_mats = Vec::with_capacity(c.rows);
    for r in 0..c.rows {
        let mut cols = Vec::with_capacity(c.cols);
        for col in 0..c.cols {
            cols.push(c.get(r, col).to_dense_matrix()?);
        }
        let refs: Vec<&Matrix> = cols.iter().collect();
        row_mats.push(Matrix::hcat(&refs)?);
    }
    let refs: Vec<&Matrix> = row_mats.iter().collect();
    ok1(Value::matrix(Matrix::vcat(&refs)?))
}

pub fn fn_num2cell(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("num2cell", "1 or 2", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    if args.len() == 1 {
        let mut data = Vec::with_capacity(m.len());
        for c in 0..m.cols() {
            for r in 0..m.rows() {
                data.push(Value::matrix(Matrix::complex_scalar(m.get_c(r, c)).normalize()));
            }
        }
        // CellArray is row-major; build via from_rows
        let mut rows = Vec::with_capacity(m.rows());
        for r in 0..m.rows() {
            let mut row = Vec::with_capacity(m.cols());
            for c in 0..m.cols() {
                row.push(Value::matrix(Matrix::complex_scalar(m.get_c(r, c)).normalize()));
            }
            rows.push(row);
        }
        let _ = data;
        return ok1(Value::Cell(CellArray::from_rows(rows)?));
    }
    let dim = as_usize(&args[1], "num2cell")?;
    if dim == 1 {
        let mut cols = Vec::with_capacity(m.cols());
        for c in 0..m.cols() {
            let v: Vec<Complex> = (0..m.rows()).map(|r| m.get_c(r, c)).collect();
            cols.push(Value::matrix(Matrix::complex_col_vec(v).normalize()));
        }
        ok1(Value::Cell(CellArray::from_rows(vec![cols])?))
    } else {
        let mut rows = Vec::with_capacity(m.rows());
        for r in 0..m.rows() {
            let v: Vec<Complex> = (0..m.cols()).map(|c| m.get_c(r, c)).collect();
            rows.push(vec![Value::matrix(Matrix::complex_row_vec(v).normalize())]);
        }
        ok1(Value::Cell(CellArray::from_rows(rows)?))
    }
}

pub fn fn_deal(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    let n = nargout.max(1);
    if args.len() == 1 {
        Ok(vec![args[0].clone(); n])
    } else if args.len() >= n {
        Ok(args[..n].to_vec())
    } else {
        Err(MatroError::arity("deal", format!("1 or {n}"), args.len()))
    }
}

pub fn fn_warning(args: &[Value], stdout: &mut String) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Ok(vec![]);
    }
    if let Some(first) = value_as_text(&args[0]) {
        let low = first.trim().to_ascii_lowercase();
        if matches!(low.as_str(), "off" | "on" | "query") {
            return Ok(vec![]);
        }
        // If 2+ args and first looks like an identifier (`A:B`), message is 2nd arg
        let msg = if args.len() >= 2 && first.contains(':') && !first.contains(' ') {
            value_as_text(&args[1]).unwrap_or(first)
        } else {
            first
        };
        stdout.push_str(&format!("Warning: {msg}\n"));
    }
    Ok(vec![])
}

/// Helper to create a `BTreeMap` if needed elsewhere.
#[allow(dead_code)]
fn _keep_btree() -> BTreeMap<String, Value> {
    BTreeMap::new()
}
