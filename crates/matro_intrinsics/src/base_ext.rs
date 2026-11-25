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
    let dims = m.dims();
    let nd = dims.len();
    let mut shifts = vec![0isize; nd];
    if args.len() == 3 {
        let k = args[1].as_scalar()?.round() as isize;
        let dim = as_usize(&args[2], "circshift")?;
        if dim >= 1 && dim <= nd {
            shifts[dim - 1] = k;
        }
    } else {
        let kv = args[1].to_dense_matrix()?;
        if kv.len() == 1 {
            let k = kv.data()[0].round() as isize;
            let ax = dims.iter().position(|&d| d != 1).unwrap_or(0);
            shifts[ax] = k;
        } else {
            for (ax, &v) in kv.data().iter().enumerate().take(nd) {
                shifts[ax] = v.round() as isize;
            }
        }
    }
    let mut strides = vec![1usize; nd];
    for ax in 1..nd {
        strides[ax] = strides[ax - 1] * dims[ax - 1];
    }
    let total = m.len();
    let mut re = vec![0.0; total];
    let mut im = vec![0.0; total];
    let mut coord = vec![0usize; nd];
    for lin in 0..total {
        let mut dst = 0usize;
        for ax in 0..nd {
            let d = dims[ax] as isize;
            let nc = (coord[ax] as isize + shifts[ax]).rem_euclid(d) as usize;
            dst += nc * strides[ax];
        }
        let v = m.get_linear_c(lin);
        re[dst] = v.re;
        im[dst] = v.im;
        for ax in 0..nd {
            coord[ax] += 1;
            if coord[ax] < dims[ax] {
                break;
            }
            coord[ax] = 0;
        }
    }
    let out = Matrix::from_nd_complex_col_major(dims, re, im)?.normalize();
    let out = if m.is_logical() { out.with_logical(true) } else { out };
    ok1(Value::matrix(out))
}

pub fn fn_flip(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("flip", "1 or 2 (A [, dim])", args.len()));
    }
    if let Value::String(s) = &args[0] {
        let dim = args.get(1).map(|v| as_usize(v, "flip")).transpose()?.unwrap_or(2);
        return ok1(Value::String(if dim == 2 {
            s.chars().rev().collect()
        } else {
            s.clone()
        }));
    }
    let m = args[0].to_dense_matrix()?;
    let dims = m.dims();
    let dim = if args.len() == 2 {
        as_usize(&args[1], "flip")?
    } else {
        dims.iter().position(|&d| d != 1).map_or(1, |i| i + 1)
    };
    if dim == 0 {
        return Err(MatroError::runtime("flip: dimension must be >= 1"));
    }
    ok1(Value::matrix(crate::extras::flip_axis_nd(&m, dim - 1)?))
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
    if args.len() < 2 || args.len() > 6 {
        return Err(MatroError::arity("accumarray", "2 to 6 (subs, val [, sz, fun, fillval, issparse])", args.len()));
    }
    let val = args[1].to_dense_matrix()?;
    // Support cell array subs `{i, j, ...}` or matrix `subs`
    let (coords_nd, nd): (Vec<Vec<usize>>, usize) = match &args[0] {
        Value::Cell(c) => {
            let k = c.data.len();
            if k == 0 {
                return ok1(Value::empty());
            }
            let mats: Vec<Matrix> = c
                .data
                .iter()
                .map(|v| v.to_dense_matrix())
                .collect::<Result<_>>()?;
            let n = mats[0].len();
            let mut coords = Vec::with_capacity(n);
            for i in 0..n {
                let mut pt = Vec::with_capacity(k);
                for m in &mats {
                    let v = if m.len() == 1 { m.data()[0] } else { m.data()[i] };
                    let idx = v.round() as isize;
                    if idx < 1 {
                        return Err(MatroError::index("accumarray: subscripts must be >= 1"));
                    }
                    pt.push((idx - 1) as usize);
                }
                coords.push(pt);
            }
            (coords, k)
        }
        _ => {
            let subs = args[0].to_dense_matrix()?;
            let is_1d = subs.rows() == 1 || subs.cols() == 1;
            if is_1d {
                let n = subs.len();
                let mut coords = Vec::with_capacity(n);
                for &v in subs.data() {
                    let idx = v.round() as isize;
                    if idx < 1 {
                        return Err(MatroError::index("accumarray: subscripts must be >= 1"));
                    }
                    coords.push(vec![(idx - 1) as usize]);
                }
                (coords, 1)
            } else {
                let n = subs.rows();
                let k = subs.cols();
                let mut coords = Vec::with_capacity(n);
                for i in 0..n {
                    let mut pt = Vec::with_capacity(k);
                    for j in 0..k {
                        let idx = subs.get(i, j).round() as isize;
                        if idx < 1 {
                            return Err(MatroError::index("accumarray: subscripts must be >= 1"));
                        }
                        pt.push((idx - 1) as usize);
                    }
                    coords.push(pt);
                }
                (coords, k)
            }
        }
    };
    let n = coords_nd.len();
    let val_scalar = val.as_complex_scalar();
    if val_scalar.is_none() && val.len() != n {
        return Err(MatroError::dimension("accumarray: val length must match subs"));
    }
    let fill = if args.len() >= 5 && !args[4].to_dense_matrix().is_ok_and(|m| m.is_empty()) {
        args[4].to_dense_matrix()?.as_complex_scalar().unwrap_or(Complex::ZERO)
    } else {
        Complex::ZERO
    };
    let make_sparse = args.get(5).is_some_and(Value::is_truthy);

    let mut max_dims = vec![0usize; nd];
    for pt in &coords_nd {
        for ax in 0..nd {
            max_dims[ax] = max_dims[ax].max(pt[ax] + 1);
        }
    }
    let out_dims: Vec<usize> = if args.len() >= 3 && !args[2].to_dense_matrix().is_ok_and(|m| m.is_empty()) {
        let sz = args[2].to_dense_matrix()?;
        let mut d: Vec<usize> = sz.data().iter().map(|&x| x.max(0.0).round() as usize).collect();
        if d.len() == 1 {
            d.push(1);
        }
        d
    } else if nd == 1 {
        vec![max_dims[0], 1]
    } else {
        max_dims
    };
    let mut strides = vec![1usize; out_dims.len()];
    for ax in 1..out_dims.len() {
        strides[ax] = strides[ax - 1] * out_dims[ax - 1];
    }
    let total: usize = out_dims.iter().product();
    let mut acc = vec![Complex::ZERO; total];
    let mut touched = vec![false; total];
    for (i, pt) in coords_nd.iter().enumerate() {
        let mut lin = 0usize;
        for ax in 0..pt.len() {
            if ax >= out_dims.len() || pt[ax] >= out_dims[ax] {
                return Err(MatroError::index("accumarray: subscript exceeds sz"));
            }
            lin += pt[ax] * strides[ax];
        }
        let v = val_scalar.unwrap_or_else(|| val.get_linear_c(i));
        acc[lin] = acc[lin] + v;
        touched[lin] = true;
    }
    if !fill.is_zero() {
        for i in 0..total {
            if !touched[i] {
                acc[i] = fill;
            }
        }
    }
    let re: Vec<f64> = acc.iter().map(|c| c.re).collect();
    let im: Vec<f64> = acc.iter().map(|c| c.im).collect();
    let mat = Matrix::from_nd_complex_col_major(out_dims, re, im)?.normalize();
    if make_sparse && mat.ndims() == 2 {
        ok1(Value::sparse(SparseMatrix::from_dense(&mat)))
    } else {
        ok1(Value::matrix(mat))
    }
}

pub fn fn_histcounts(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("histcounts", ">= 1 (X [, nbins | edges, NV...])", 0));
    }
    let x = args[0].to_dense_matrix()?;
    let finite: Vec<f64> = x.data().iter().copied().filter(|v| v.is_finite()).collect();
    let mut nbins_opt: Option<usize> = None;
    let mut edges_opt: Option<Vec<f64>> = None;
    let mut bin_width: Option<f64> = None;
    let mut bin_limits: Option<(f64, f64)> = None;
    let mut norm_mode = "count".to_string();

    let mut i = 1;
    if i < args.len() && value_as_text(&args[i]).is_none() {
        let e = args[i].to_dense_matrix()?;
        if e.len() == 1 {
            nbins_opt = Some(e.data()[0].max(1.0).round() as usize);
        } else if e.len() >= 2 {
            edges_opt = Some(e.data().to_vec());
        }
        i += 1;
    }
    while i + 1 < args.len() {
        if let Some(k) = value_as_text(&args[i]) {
            match k.trim().to_ascii_lowercase().as_str() {
                "normalization" => {
                    if let Some(s) = value_as_text(&args[i + 1]) {
                        norm_mode = s.trim().to_ascii_lowercase();
                    }
                }
                "numbins" => nbins_opt = Some(as_usize(&args[i + 1], "histcounts")?.max(1)),
                "binwidth" => bin_width = Some(args[i + 1].as_scalar()?),
                "binlimits" => {
                    let lim = args[i + 1].to_dense_matrix()?;
                    if lim.len() >= 2 {
                        bin_limits = Some((lim.data()[0], lim.data()[1]));
                    }
                }
                _ => {}
            }
        }
        i += 2;
    }

    let edges: Vec<f64> = if let Some(e) = edges_opt {
        e
    } else {
        let (lo, hi) = if let Some(lim) = bin_limits {
            lim
        } else {
            let xmin = finite.iter().copied().fold(f64::INFINITY, f64::min);
            let xmax = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if !xmin.is_finite() {
                (0.0, 1.0)
            } else if (xmax - xmin).abs() < 1e-15 {
                (xmin - 0.5, xmax + 0.5)
            } else {
                (xmin, xmax)
            }
        };
        if let Some(bw) = bin_width {
            let w = bw.abs().max(1e-15);
            let nb = ((hi - lo) / w).ceil().max(1.0) as usize;
            (0..=nb).map(|k| lo + (k as f64) * w).collect()
        } else {
            let nb = nbins_opt
                .unwrap_or_else(|| ((finite.len() as f64).sqrt().ceil() as usize).clamp(1, 50));
            (0..=nb)
                .map(|k| lo + (hi - lo) * (k as f64) / (nb as f64))
                .collect()
        }
    };
    if edges.len() < 2 {
        return Err(MatroError::dimension("histcounts: edges must have at least 2 elements"));
    }
    let nb = edges.len() - 1;
    let mut counts = vec![0.0; nb];
    let mut bins = vec![0.0; x.len()];
    let mut total_in = 0.0;
    for (idx, &v) in x.data().iter().enumerate() {
        if v.is_nan() || v < edges[0] || v > edges[nb] {
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
        total_in += 1.0;
    }
    match norm_mode.as_str() {
        "probability" if total_in > 0.0 => {
            for c in &mut counts {
                *c /= total_in;
            }
        }
        "pdf" if total_in > 0.0 => {
            for (k, c) in counts.iter_mut().enumerate() {
                let w = (edges[k + 1] - edges[k]).abs();
                if w > 0.0 {
                    *c /= total_in * w;
                }
            }
        }
        "countdensity" => {
            for (k, c) in counts.iter_mut().enumerate() {
                let w = (edges[k + 1] - edges[k]).abs();
                if w > 0.0 {
                    *c /= w;
                }
            }
        }
        "cumcount" => {
            let mut acc = 0.0;
            for c in &mut counts {
                acc += *c;
                *c = acc;
            }
        }
        "cdf" if total_in > 0.0 => {
            let mut acc = 0.0;
            for c in &mut counts {
                acc += *c;
                *c = acc / total_in;
            }
        }
        _ => {}
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

pub fn fn_histc(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity("histc", "2 or 3 (X, edges [, dim])", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let edges = args[1].to_dense_matrix()?.data().to_vec();
    let ne = edges.len();
    let mut counts = vec![0.0; ne];
    let mut bins = vec![0.0; x.len()];
    if ne > 0 {
        for (idx, &v) in x.data().iter().enumerate() {
            if v.is_nan() || v < edges[0] || v > edges[ne - 1] {
                continue;
            }
            let b = if v == edges[ne - 1] {
                ne - 1
            } else {
                edges.partition_point(|&e| e <= v).saturating_sub(1).min(ne - 1)
            };
            counts[b] += 1.0;
            bins[idx] = (b + 1) as f64;
        }
    }
    let is_row = x.rows() == 1 && x.cols() > 1;
    let c_mat = if is_row {
        Matrix::row_vec(counts)
    } else {
        Matrix::col_vec(counts)
    };
    if nargout >= 2 {
        Ok(vec![
            Value::matrix(c_mat),
            Value::matrix(Matrix::from_nd_col_major(x.dims(), bins)?),
        ])
    } else {
        ok1(Value::matrix(c_mat))
    }
}

pub fn fn_discretize(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() < 2 {
        return Err(MatroError::arity("discretize", ">= 2 (X, edges|N [, values, NV...])", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let e_arg = args[1].to_dense_matrix()?;
    let edges: Vec<f64> = if e_arg.len() == 1 {
        let nb = e_arg.data()[0].max(1.0).round() as usize;
        let finite: Vec<f64> = x.data().iter().copied().filter(|v| v.is_finite()).collect();
        let lo = finite.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let (lo, hi) = if !lo.is_finite() { (0.0, 1.0) } else if (hi - lo).abs() < 1e-15 { (lo - 0.5, hi + 0.5) } else { (lo, hi) };
        (0..=nb).map(|k| lo + (hi - lo) * (k as f64) / (nb as f64)).collect()
    } else {
        e_arg.data().to_vec()
    };
    if edges.len() < 2 {
        return Err(MatroError::dimension("discretize: edges must have >= 2 elements"));
    }
    let mut right_inc = false;
    let mut values_opt: Option<Matrix> = None;
    let mut i = 2;
    if i < args.len() && value_as_text(&args[i]).is_none() {
        values_opt = Some(args[i].to_dense_matrix()?);
        i += 1;
    }
    while i + 1 < args.len() {
        if value_as_text(&args[i]).is_some_and(|s| s.eq_ignore_ascii_case("includededge")) {
            if value_as_text(&args[i + 1]).is_some_and(|s| s.eq_ignore_ascii_case("right")) {
                right_inc = true;
            }
        }
        i += 2;
    }
    let nb = edges.len() - 1;
    let mut bins = vec![f64::NAN; x.len()];
    for (idx, &v) in x.data().iter().enumerate() {
        if v.is_nan() || v < edges[0] || v > edges[nb] {
            continue;
        }
        let b = if right_inc {
            if v == edges[0] {
                0
            } else {
                edges.partition_point(|&e| e < v).saturating_sub(1).min(nb - 1)
            }
        } else if v == edges[nb] {
            nb - 1
        } else {
            edges.partition_point(|&e| e <= v).saturating_sub(1).min(nb - 1)
        };
        bins[idx] = if let Some(ref vm) = values_opt {
            vm.data().get(b).copied().unwrap_or(f64::NAN)
        } else {
            (b + 1) as f64
        };
    }
    let y = Matrix::from_nd_col_major(x.dims(), bins)?;
    if nargout >= 2 {
        Ok(vec![Value::matrix(y), Value::matrix(Matrix::row_vec(edges))])
    } else {
        ok1(Value::matrix(y))
    }
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
        3 => {
            // S = spdiags(B, d, A): replace diagonals d of A with columns of B
            let b = args[0].to_dense_matrix()?;
            let d_mat = args[1].to_dense_matrix()?;
            let d_vec: Vec<isize> = d_mat.data().iter().map(|&df| df.round() as isize).collect();
            let a_sp = match &args[2] {
                Value::Sparse(s) => (**s).clone(),
                other => SparseMatrix::from_dense(&other.to_dense_matrix()?),
            };
            let (m, n) = (a_sp.rows(), a_sp.cols());
            let mut triplets: Vec<(usize, usize, f64)> = a_sp
                .triplets()
                .into_iter()
                .filter(|&(r, c, _)| !d_vec.contains(&((c as isize) - (r as isize))))
                .collect();
            for (k, &d) in d_vec.iter().enumerate() {
                let col_k = k.min(b.cols().saturating_sub(1));
                for idx in 0..b.rows() {
                    let (r, c) = if m >= n {
                        (idx as isize - d, idx as isize)
                    } else {
                        (idx as isize, idx as isize + d)
                    };
                    if r >= 0 && (r as usize) < m && c >= 0 && (c as usize) < n {
                        let v = b.get(idx, col_k);
                        if v != 0.0 {
                            triplets.push((r as usize, c as usize, v));
                        }
                    }
                }
            }
            ok1(Value::sparse(SparseMatrix::from_triplets(m, n, &triplets)?))
        }
        n => Err(MatroError::arity("spdiags", "1, 2, 3, or 4", n)),
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
        return Err(MatroError::arity(name, "1 or 2 (A [, tol | 'r'])", args.len()));
    }
    let a = args[0].to_dense_matrix()?;
    if is_null
        && args.len() == 2
        && crate::value_as_text(&args[1]).is_some_and(|s| s.eq_ignore_ascii_case("r"))
    {
        // Rational nullspace basis from RREF
        let rref_res = fn_rref(&[args[0].clone()], 2)?;
        let r_mat = rref_res[0].to_dense_matrix()?;
        let piv_mat = rref_res[1].to_dense_matrix()?;
        let pivs: Vec<usize> = piv_mat.data().iter().map(|&x| (x.round() as usize) - 1).collect();
        let n = a.cols();
        let free_cols: Vec<usize> = (0..n).filter(|c| !pivs.contains(c)).collect();
        let mut out = Matrix::zeros(n, free_cols.len());
        for (k, &fc) in free_cols.iter().enumerate() {
            out.set_c(fc, k, Complex::ONE);
            for (row_idx, &pc) in pivs.iter().enumerate() {
                out.set_c(pc, k, -r_mat.get_c(row_idx, fc));
            }
        }
        return ok1(Value::matrix(out.normalize()));
    }
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
        .map(|r| (0..n).map(|c| a.get_c(r, c).abs()).sum::<f64>())
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
        let mut piv_v = a.get_c(row, col).abs();
        for r in (row + 1)..m {
            let v = a.get_c(r, col).abs();
            if v > piv_v {
                piv_v = v;
                piv_r = r;
            }
        }
        if piv_v <= tol {
            for r in row..m {
                a.set_c(r, col, Complex::ZERO);
            }
            continue;
        }
        if piv_r != row {
            for c in col..n {
                let tmp = a.get_c(row, c);
                a.set_c(row, c, a.get_c(piv_r, c));
                a.set_c(piv_r, c, tmp);
            }
        }
        let pv = a.get_c(row, col);
        for c in col..n {
            a.set_c(row, c, a.get_c(row, c) / pv);
        }
        for r in 0..m {
            if r != row {
                let f = a.get_c(r, col);
                if !f.is_zero() {
                    for c in col..n {
                        a.set_c(r, c, a.get_c(r, c) - f * a.get_c(row, c));
                    }
                }
            }
        }
        pivots.push((col + 1) as f64);
        row += 1;
    }
    let a = a.normalize();
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
    let a_is_vec3 = a.ndims() == 2 && (a.rows() == 1 || a.cols() == 1) && a.len() == 3;
    let b_is_vec3 = b.ndims() == 2 && (b.rows() == 1 || b.cols() == 1) && b.len() == 3;
    if args.len() == 2 && a_is_vec3 && b_is_vec3 {
        let (a0, a1, a2) = (a.get_linear_c(0), a.get_linear_c(1), a.get_linear_c(2));
        let (b0, b1, b2) = (b.get_linear_c(0), b.get_linear_c(1), b.get_linear_c(2));
        let vals = vec![a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0];
        let out = if a.rows() == 1 {
            Matrix::complex_row_vec(vals)
        } else {
            Matrix::complex_col_vec(vals)
        };
        return ok1(Value::matrix(out.normalize()));
    }
    if a.dims() != b.dims() {
        return Err(MatroError::dimension("cross: A and B must have the same size"));
    }
    let dims = a.dims();
    let dim = if args.len() == 3 {
        let d = as_usize(&args[2], "cross")?;
        if d == 0 || d > dims.len() || dims[d - 1] != 3 {
            return Err(MatroError::dimension(format!(
                "cross: dimension {d} must have length 3"
            )));
        }
        d
    } else if let Some(ax) = dims.iter().position(|&d| d == 3) {
        ax + 1
    } else {
        return Err(MatroError::dimension(
            "cross: A and B must have at least one dimension of length 3",
        ));
    };
    let ax = dim - 1;
    let inner: usize = dims[..ax].iter().product();
    let outer: usize = dims[ax + 1..].iter().product();
    let total = a.len();
    let mut re = vec![0.0; total];
    let mut im = vec![0.0; total];
    for o in 0..outer {
        let block = o * 3 * inner;
        for i in 0..inner {
            let i0 = block + i;
            let i1 = block + inner + i;
            let i2 = block + 2 * inner + i;
            let (a0, a1, a2) = (a.get_linear_c(i0), a.get_linear_c(i1), a.get_linear_c(i2));
            let (b0, b1, b2) = (b.get_linear_c(i0), b.get_linear_c(i1), b.get_linear_c(i2));
            let c0 = a1 * b2 - a2 * b1;
            let c1 = a2 * b0 - a0 * b2;
            let c2 = a0 * b1 - a1 * b0;
            re[i0] = c0.re;
            im[i0] = c0.im;
            re[i1] = c1.re;
            im[i1] = c1.im;
            re[i2] = c2.re;
            im[i2] = c2.im;
        }
    }
    ok1(Value::matrix(
        Matrix::from_nd_complex_col_major(dims, re, im)?.normalize(),
    ))
}

pub fn fn_vecnorm(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity("vecnorm", "1 to 3 (A [, p [, dim]])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let p = if args.len() >= 2 && !args[1].to_dense_matrix()?.is_empty() {
        args[1].as_scalar()?
    } else {
        2.0
    };
    let dims = m.dims();
    let dim = if args.len() >= 3 {
        let d = as_usize(&args[2], "vecnorm")?;
        if d == 0 {
            return Err(MatroError::runtime("vecnorm: dimension must be >= 1"));
        }
        d
    } else {
        dims.iter().position(|&d| d != 1).map_or(1, |i| i + 1)
    };
    let ax = dim - 1;
    if ax >= dims.len() {
        return ok1(Value::matrix(m.map_c_to_real(Complex::abs)));
    }
    let inner: usize = dims[..ax].iter().product();
    let axis_len = dims[ax];
    let outer: usize = dims[ax + 1..].iter().product();
    let mut out_dims = dims.clone();
    out_dims[ax] = 1;
    let mut out = Vec::with_capacity(inner * outer);
    let mut buf = Vec::with_capacity(axis_len);
    for o in 0..outer {
        let block = o * axis_len * inner;
        for i in 0..inner {
            buf.clear();
            for k in 0..axis_len {
                buf.push(m.get_linear_c(block + k * inner + i).abs());
            }
            let val = if p.is_infinite() && p > 0.0 {
                buf.iter().copied().fold(0.0f64, f64::max)
            } else if p.is_infinite() && p < 0.0 {
                buf.iter().copied().fold(f64::INFINITY, f64::min)
            } else {
                buf.iter().map(|x| x.powf(p)).sum::<f64>().powf(1.0 / p)
            };
            out.push(val);
        }
    }
    ok1(Value::matrix(Matrix::from_nd_col_major(out_dims, out)?))
}

pub fn fn_rms(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("rms", "1 or 2 (x [, dim])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let dims = m.dims();
    let dim = if args.len() >= 2 {
        as_usize(&args[1], "rms")?
    } else {
        dims.iter().position(|&d| d != 1).map_or(1, |i| i + 1)
    };
    if dim == 0 {
        return Err(MatroError::runtime("rms: dimension must be >= 1"));
    }
    let ax = dim - 1;
    if ax >= dims.len() {
        return ok1(Value::matrix(m.map_c_to_real(Complex::abs)));
    }
    let inner: usize = dims[..ax].iter().product();
    let axis_len = dims[ax];
    let outer: usize = dims[ax + 1..].iter().product();
    let mut out_dims = dims.clone();
    out_dims[ax] = 1;
    let mut out = Vec::with_capacity(inner * outer);
    for o in 0..outer {
        let block = o * axis_len * inner;
        for i in 0..inner {
            let mut ss = 0.0;
            for k in 0..axis_len {
                ss += m.get_linear_c(block + k * inner + i).norm_sq();
            }
            out.push((ss / (axis_len.max(1) as f64)).sqrt());
        }
    }
    ok1(Value::matrix(Matrix::from_nd_col_major(out_dims, out)?))
}

pub fn fn_normalize(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 4 {
        return Err(MatroError::arity("normalize", "1 to 4 (A [, dim, method, methodtype])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let dims = m.dims();
    let mut dim = dims.iter().position(|&d| d != 1).map_or(1, |i| i + 1);
    let mut method = "zscore".to_string();
    let mut method_param: Option<Value> = None;
    let mut i = 1;
    while i < args.len() {
        if let Some(s) = value_as_text(&args[i]) {
            method = s.trim().to_ascii_lowercase();
            if i + 1 < args.len() {
                method_param = Some(args[i + 1].clone());
                i += 2;
                continue;
            }
        } else if i == 1 {
            dim = as_usize(&args[i], "normalize")?;
        }
        i += 1;
    }
    let norm_vec = |xs: &[f64]| -> (Vec<f64>, f64, f64) {
        let valid: Vec<f64> = xs.iter().copied().filter(|x| !x.is_nan()).collect();
        if valid.is_empty() {
            return (vec![f64::NAN; xs.len()], f64::NAN, f64::NAN);
        }
        let n = valid.len() as f64;
        match method.as_str() {
            "norm" => {
                let p = method_param
                    .as_ref()
                    .and_then(|v| v.as_scalar().ok())
                    .unwrap_or(2.0);
                let sc = if p.is_infinite() && p > 0.0 {
                    valid.iter().map(|x| x.abs()).fold(0.0f64, f64::max)
                } else {
                    valid.iter().map(|x| x.abs().powf(p)).sum::<f64>().powf(1.0 / p)
                };
                let res = if sc == 0.0 { xs.to_vec() } else { xs.iter().map(|x| x / sc).collect() };
                (res, 0.0, sc)
            }
            "range" => {
                let (a, b) = method_param
                    .as_ref()
                    .and_then(|v| v.to_dense_matrix().ok())
                    .filter(|r| r.len() >= 2)
                    .map(|r| (r.data()[0], r.data()[1]))
                    .unwrap_or((0.0, 1.0));
                let lo = valid.iter().copied().fold(f64::INFINITY, f64::min);
                let hi = valid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let span = hi - lo;
                let res = if span == 0.0 {
                    xs.iter().map(|x| if x.is_nan() { f64::NAN } else { a }).collect()
                } else {
                    xs.iter().map(|x| a + (x - lo) * (b - a) / span).collect()
                };
                (res, lo, span / (b - a))
            }
            "center" => {
                let use_med = method_param
                    .as_ref()
                    .and_then(value_as_text)
                    .is_some_and(|s| s.eq_ignore_ascii_case("median"));
                let c = if use_med {
                    let mut sv = valid.clone();
                    sv.sort_by(f64::total_cmp);
                    let mid = sv.len() / 2;
                    if sv.len() % 2 == 1 { sv[mid] } else { 0.5 * (sv[mid - 1] + sv[mid]) }
                } else {
                    valid.iter().sum::<f64>() / n
                };
                (xs.iter().map(|x| x - c).collect(), c, 1.0)
            }
            "scale" => {
                let sc = {
                    let mu = valid.iter().sum::<f64>() / n;
                    (valid.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)).sqrt()
                };
                let res = if sc == 0.0 { xs.to_vec() } else { xs.iter().map(|x| x / sc).collect() };
                (res, 0.0, sc)
            }
            _ => {
                let robust = method_param
                    .as_ref()
                    .and_then(value_as_text)
                    .is_some_and(|s| s.eq_ignore_ascii_case("robust"));
                if robust {
                    let mut sv = valid.clone();
                    sv.sort_by(f64::total_cmp);
                    let mid = sv.len() / 2;
                    let med = if sv.len() % 2 == 1 { sv[mid] } else { 0.5 * (sv[mid - 1] + sv[mid]) };
                    let mut devs: Vec<f64> = sv.iter().map(|x| (x - med).abs()).collect();
                    devs.sort_by(f64::total_cmp);
                    let mad = if devs.len() % 2 == 1 { devs[mid] } else { 0.5 * (devs[mid - 1] + devs[mid]) };
                    let res = if mad == 0.0 { vec![0.0; xs.len()] } else { xs.iter().map(|x| (x - med) / mad).collect() };
                    (res, med, mad)
                } else {
                    let mu = valid.iter().sum::<f64>() / n;
                    let denom = (n - 1.0).max(1.0);
                    let sd = (valid.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / denom).sqrt();
                    let res = if sd == 0.0 { vec![0.0; xs.len()] } else { xs.iter().map(|x| (x - mu) / sd).collect() };
                    (res, mu, sd)
                }
            }
        }
    };
    let ax = dim.saturating_sub(1);
    if ax >= dims.len() {
        return ok1(Value::matrix(m));
    }
    let inner: usize = dims[..ax].iter().product();
    let axis_len = dims[ax];
    let outer: usize = dims[ax + 1..].iter().product();
    let mut out_data = vec![0.0; m.len()];
    let mut stat_dims = dims.clone();
    stat_dims[ax] = 1;
    let mut centers = Vec::with_capacity(inner * outer);
    let mut scales = Vec::with_capacity(inner * outer);
    let mut buf = Vec::with_capacity(axis_len);
    for o in 0..outer {
        let block = o * axis_len * inner;
        for ii in 0..inner {
            buf.clear();
            for k in 0..axis_len {
                buf.push(m.data()[block + k * inner + ii]);
            }
            let (nv, c_val, s_val) = norm_vec(&buf);
            for (k, &v) in nv.iter().enumerate() {
                out_data[block + k * inner + ii] = v;
            }
            centers.push(c_val);
            scales.push(s_val);
        }
    }
    let out_m = Matrix::from_nd_col_major(dims, out_data)?;
    if nargout >= 3 {
        Ok(vec![
            Value::matrix(out_m),
            Value::matrix(Matrix::from_nd_col_major(stat_dims.clone(), centers)?),
            Value::matrix(Matrix::from_nd_col_major(stat_dims, scales)?),
        ])
    } else if nargout == 2 {
        Ok(vec![
            Value::matrix(out_m),
            Value::matrix(Matrix::from_nd_col_major(stat_dims, centers)?),
        ])
    } else {
        ok1(Value::matrix(out_m))
    }
}

pub fn fn_rescale(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("rescale", ">= 1 (A [, l, u] [, 'InputMin', ...])", 0));
    }
    let m = args[0].to_dense_matrix()?;
    let mut l = 0.0;
    let mut u = 1.0;
    let mut in_min: Option<f64> = None;
    let mut in_max: Option<f64> = None;
    let mut i = 1;
    while i < args.len() {
        if let Some(s) = value_as_text(&args[i]) {
            if i + 1 < args.len() {
                let val = args[i + 1].as_scalar()?;
                if s.eq_ignore_ascii_case("inputmin") {
                    in_min = Some(val);
                } else if s.eq_ignore_ascii_case("inputmax") {
                    in_max = Some(val);
                }
                i += 2;
                continue;
            }
        } else if i == 1 && i + 1 < args.len() && value_as_text(&args[i + 1]).is_none() {
            l = args[1].as_scalar()?;
            u = args[2].as_scalar()?;
            i += 2;
            continue;
        }
        i += 1;
    }
    let lo = in_min.unwrap_or_else(|| {
        m.data()
            .iter()
            .copied()
            .filter(|x| !x.is_nan())
            .fold(f64::INFINITY, f64::min)
    });
    let hi = in_max.unwrap_or_else(|| {
        m.data()
            .iter()
            .copied()
            .filter(|x| !x.is_nan())
            .fold(f64::NEG_INFINITY, f64::max)
    });
    let span = hi - lo;
    let out = if span == 0.0 {
        m.map(|x| if x.is_nan() { f64::NAN } else { l })
    } else {
        m.map(|x| {
            if x.is_nan() {
                f64::NAN
            } else {
                let xc = x.max(lo).min(hi);
                l + (xc - lo) * (u - l) / span
            }
        })
    };
    ok1(Value::matrix(out))
}

pub fn fn_clip(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 3 {
        return Err(MatroError::arity("clip", "3 (X, lower, upper)", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let lo_m = args[1].to_dense_matrix()?;
    let hi_m = args[2].to_dense_matrix()?;
    let step1 = x.elementwise(&lo_m, |v, lo| {
        if v.is_nan() || lo.is_nan() { f64::NAN } else if v < lo { lo } else { v }
    })?;
    let out = step1.elementwise(&hi_m, |v, hi| {
        if v.is_nan() || hi.is_nan() { f64::NAN } else if v > hi { hi } else { v }
    })?;
    ok1(Value::matrix(out))
}

pub fn fn_trapz(args: &[Value], cumulative: bool) -> Result<Vec<Value>> {
    let name = if cumulative { "cumtrapz" } else { "trapz" };
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity(name, "1 to 3", args.len()));
    }
    let (x_opt, y, dim) = match args.len() {
        1 => {
            let y = args[0].to_dense_matrix()?;
            let d = y.dims().iter().position(|&sz| sz != 1).map_or(1, |i| i + 1);
            (None, y, d)
        }
        2 => {
            let a = args[0].to_dense_matrix()?;
            let b = args[1].to_dense_matrix()?;
            if b.len() == 1
                && b.data()[0] >= 1.0
                && (b.data()[0] - b.data()[0].round()).abs() < 1e-12
                && a.len() > 1
            {
                (None, a, b.data()[0].round() as usize)
            } else {
                let d = b.dims().iter().position(|&sz| sz != 1).map_or(1, |i| i + 1);
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
    if dim == 0 {
        return Err(MatroError::runtime(format!("{name}: dimension must be >= 1")));
    }
    let dims = y.dims();
    let ax = dim - 1;
    if ax >= dims.len() {
        return if cumulative {
            ok1(Value::matrix(Matrix::zeros_nd(dims)))
        } else {
            ok1(Value::matrix(Matrix::zeros_nd(dims)))
        };
    }
    let axis_len = dims[ax];
    if let Some(ref xm) = x_opt {
        if xm.len() != 1 && xm.len() != axis_len {
            return Err(MatroError::dimension(format!(
                "{name}: X length {} does not match dimension {dim} length {axis_len}",
                xm.len()
            )));
        }
    }
    let inner: usize = dims[..ax].iter().product();
    let outer: usize = dims[ax + 1..].iter().product();
    let integrate = |ys: &[Complex]| -> Vec<Complex> {
        let n = ys.len();
        let mut acc = vec![Complex::ZERO; n];
        for i in 1..n {
            let dx = match &x_opt {
                Some(xm) if xm.len() == 1 => xm.get_linear_c(0),
                Some(xm) => xm.get_linear_c(i) - xm.get_linear_c(i - 1),
                None => Complex::ONE,
            };
            acc[i] = acc[i - 1] + dx * Complex::from_real(0.5) * (ys[i - 1] + ys[i]);
        }
        acc
    };

    if cumulative {
        let mut out_re = vec![0.0; y.len()];
        let mut out_im = vec![0.0; y.len()];
        let mut buf = Vec::with_capacity(axis_len);
        for o in 0..outer {
            let block = o * axis_len * inner;
            for i in 0..inner {
                buf.clear();
                for k in 0..axis_len {
                    buf.push(y.get_linear_c(block + k * inner + i));
                }
                let res = integrate(&buf);
                for (k, v) in res.into_iter().enumerate() {
                    let pos = block + k * inner + i;
                    out_re[pos] = v.re;
                    out_im[pos] = v.im;
                }
            }
        }
        ok1(Value::matrix(
            Matrix::from_nd_complex_col_major(dims, out_re, out_im)?.normalize(),
        ))
    } else {
        let mut out_dims = dims.clone();
        out_dims[ax] = 1;
        let mut out_re = Vec::with_capacity(inner * outer);
        let mut out_im = Vec::with_capacity(inner * outer);
        let mut buf = Vec::with_capacity(axis_len);
        for o in 0..outer {
            let block = o * axis_len * inner;
            for i in 0..inner {
                buf.clear();
                for k in 0..axis_len {
                    buf.push(y.get_linear_c(block + k * inner + i));
                }
                let val = integrate(&buf).last().copied().unwrap_or(Complex::ZERO);
                out_re.push(val.re);
                out_im.push(val.im);
            }
        }
        ok1(Value::matrix(
            Matrix::from_nd_complex_col_major(out_dims, out_re, out_im)?.normalize(),
        ))
    }
}

pub fn fn_gradient(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("gradient", ">= 1 (F [, h1, h2, ...])", 0));
    }
    let f = args[0].to_dense_matrix()?;
    let diff_1d_c = |vals: &[Complex], spacing: Option<&Matrix>| -> Vec<Complex> {
        let n = vals.len();
        if n <= 1 {
            return vec![Complex::ZERO; n];
        }
        let coords: Vec<f64> = match spacing {
            Some(s) if s.len() == n => s.data().to_vec(),
            Some(s) if s.len() == 1 => (0..n).map(|i| (i as f64) * s.data()[0]).collect(),
            _ => (0..n).map(|i| i as f64).collect(),
        };
        let mut g = vec![Complex::ZERO; n];
        g[0] = (vals[1] - vals[0]) / Complex::from_real(coords[1] - coords[0]);
        g[n - 1] = (vals[n - 1] - vals[n - 2]) / Complex::from_real(coords[n - 1] - coords[n - 2]);
        for i in 1..(n - 1) {
            g[i] = (vals[i + 1] - vals[i - 1]) / Complex::from_real(coords[i + 1] - coords[i - 1]);
        }
        g
    };

    let dims = f.dims();
    let spacings: Vec<Matrix> = args[1..]
        .iter()
        .map(|v| v.to_dense_matrix())
        .collect::<Result<_>>()?;

    // 1D vector
    if f.ndims() == 2 && (f.rows() == 1 || f.cols() == 1) {
        let vals: Vec<Complex> = (0..f.len()).map(|i| f.get_linear_c(i)).collect();
        let g = diff_1d_c(&vals, spacings.first());
        let mat = if f.rows() == 1 {
            Matrix::complex_row_vec(g)
        } else {
            Matrix::complex_col_vec(g)
        };
        return ok1(Value::matrix(mat.normalize()));
    }

    // MATLAB axis order for N-D gradient outputs: [FX (dim 2), FY (dim 1), FZ (dim 3), F4 (dim 4), ...]
    let nd = dims.len();
    let want = nargout.max(1).min(nd);
    let mut outputs = Vec::with_capacity(want);
    for out_k in 0..want {
        let ax = match out_k {
            0 => 1,
            1 => 0,
            k => k,
        };
        let sp = spacings.get(out_k).or_else(|| {
            if spacings.len() == 1 && spacings[0].len() == 1 {
                spacings.first()
            } else {
                None
            }
        });
        let inner: usize = dims[..ax].iter().product();
        let axis_len = dims[ax];
        let outer: usize = dims[ax + 1..].iter().product();
        let mut out_re = vec![0.0; f.len()];
        let mut out_im = vec![0.0; f.len()];
        let mut buf = Vec::with_capacity(axis_len);
        for o in 0..outer {
            let block = o * axis_len * inner;
            for i in 0..inner {
                buf.clear();
                for k in 0..axis_len {
                    buf.push(f.get_linear_c(block + k * inner + i));
                }
                let g = diff_1d_c(&buf, sp);
                for (k, v) in g.into_iter().enumerate() {
                    let pos = block + k * inner + i;
                    out_re[pos] = v.re;
                    out_im[pos] = v.im;
                }
            }
        }
        outputs.push(Value::matrix(
            Matrix::from_nd_complex_col_major(dims.clone(), out_re, out_im)?.normalize(),
        ));
    }
    Ok(outputs)
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

fn poly_conv_c(a: &[Complex], b: &[Complex]) -> Vec<Complex> {
    if a.is_empty() || b.is_empty() {
        return vec![Complex::ZERO];
    }
    let mut out = vec![Complex::ZERO; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            out[i + j] = out[i + j] + ai * bj;
        }
    }
    out
}

fn poly_deriv_c(p: &[Complex]) -> Vec<Complex> {
    if p.len() <= 1 {
        return vec![Complex::ZERO];
    }
    let n = p.len() - 1;
    let mut d: Vec<Complex> = (0..n).map(|i| p[i] * Complex::from_real((n - i) as f64)).collect();
    while d.len() > 1 && d[0].is_zero() {
        d.remove(0);
    }
    d
}

pub fn fn_polyder(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("polyder", "1 or 2 (p | a, b)", args.len()));
    }
    let a_m = args[0].to_dense_matrix()?;
    let a: Vec<Complex> = (0..a_m.len()).map(|i| a_m.get_linear_c(i)).collect();
    if args.len() == 1 {
        let d = poly_deriv_c(&a);
        return ok1(Value::matrix(Matrix::complex_row_vec(d).normalize()));
    }
    let b_m = args[1].to_dense_matrix()?;
    let b: Vec<Complex> = (0..b_m.len()).map(|i| b_m.get_linear_c(i)).collect();
    if nargout >= 2 {
        // Quotient rule: d/dx(a/b) = (a'*b - a*b') / b^2
        let da_b = poly_conv_c(&poly_deriv_c(&a), &b);
        let a_db = poly_conv_c(&a, &poly_deriv_c(&b));
        let max_len = da_b.len().max(a_db.len());
        let mut q = vec![Complex::ZERO; max_len];
        for i in 0..da_b.len() {
            q[max_len - da_b.len() + i] = q[max_len - da_b.len() + i] + da_b[i];
        }
        for i in 0..a_db.len() {
            q[max_len - a_db.len() + i] = q[max_len - a_db.len() + i] - a_db[i];
        }
        while q.len() > 1 && q[0].is_zero() {
            q.remove(0);
        }
        let mut d = poly_conv_c(&b, &b);
        while d.len() > 1 && d[0].is_zero() {
            d.remove(0);
        }
        Ok(vec![
            Value::matrix(Matrix::complex_row_vec(q).normalize()),
            Value::matrix(Matrix::complex_row_vec(d).normalize()),
        ])
    } else {
        // Product rule: d/dx(a*b)
        let prod = poly_conv_c(&a, &b);
        let d = poly_deriv_c(&prod);
        ok1(Value::matrix(Matrix::complex_row_vec(d).normalize()))
    }
}

pub fn fn_polyint(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("polyint", "1 or 2 (p [, k])", args.len()));
    }
    let p = args[0].to_dense_matrix()?;
    let k = if args.len() == 2 {
        args[1]
            .to_dense_matrix()?
            .as_complex_scalar()
            .ok_or_else(|| MatroError::runtime("polyint: constant k must be scalar"))?
    } else {
        Complex::ZERO
    };
    let n = p.len();
    let mut out = Vec::with_capacity(n + 1);
    for i in 0..n {
        out.push(p.get_linear_c(i) / Complex::from_real((n - i) as f64));
    }
    out.push(k);
    ok1(Value::matrix(Matrix::complex_row_vec(out).normalize()))
}

// ---------------------------------------------------------------------------
// Combinatorics & Number Theory
// ---------------------------------------------------------------------------

pub fn fn_factorial(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("factorial", "1", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let out = m.map(|x| {
        if x.is_nan() || x < 0.0 || !x.is_finite() || (x - x.round()).abs() > 1e-12 {
            if x.is_infinite() && x > 0.0 {
                f64::INFINITY
            } else {
                f64::NAN
            }
        } else if x > 170.0 {
            f64::INFINITY
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
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity(name, "1 to 3", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let (nr, nc) = (m.rows(), m.cols());
    let skew = args
        .get(1)
        .and_then(value_as_text)
        .is_some_and(|s| s.eq_ignore_ascii_case("skew"));
    let res = match name {
        "issymmetric" => {
            nr == nc
                && (0..nr).all(|r| {
                    (r..nc).all(|c| {
                        let a = m.get_c(r, c);
                        let b = if skew { -m.get_c(c, r) } else { m.get_c(c, r) };
                        a == b
                    })
                })
        }
        "ishermitian" => {
            nr == nc
                && (0..nr).all(|r| {
                    (r..nc).all(|c| {
                        let a = m.get_c(r, c);
                        let b = if skew { -m.get_c(c, r).conj() } else { m.get_c(c, r).conj() };
                        a == b
                    })
                })
        }
        "isdiag" => (0..nc).all(|c| (0..nr).all(|r| r == c || m.get_c(r, c).is_zero())),
        "istril" => (0..nc).all(|c| (0..nr).all(|r| r >= c || m.get_c(r, c).is_zero())),
        "istriu" => (0..nc).all(|c| (0..nr).all(|r| r <= c || m.get_c(r, c).is_zero())),
        "issorted" => {
            let mut dir = "ascend".to_string();
            let mut dim_opt: Option<usize> = None;
            let mut rows_mode = false;
            for a in &args[1..] {
                if let Some(s) = value_as_text(a) {
                    let low = s.trim().to_ascii_lowercase();
                    if low == "rows" {
                        rows_mode = true;
                    } else {
                        dir = low;
                    }
                } else {
                    dim_opt = Some(as_usize(a, "issorted")?);
                }
            }
            let check_vec = |xs: &[f64]| -> bool {
                // Ignore trailing NaNs (default MissingPlacement='auto' puts NaNs at end)
                let clean: Vec<f64> = xs.iter().copied().filter(|x| !x.is_nan()).collect();
                match dir.as_str() {
                    "descend" => clean.windows(2).all(|w| w[0] >= w[1]),
                    "strictascend" => clean.windows(2).all(|w| w[0] < w[1]),
                    "strictdescend" => clean.windows(2).all(|w| w[0] > w[1]),
                    "monotonic" => {
                        clean.windows(2).all(|w| w[0] <= w[1])
                            || clean.windows(2).all(|w| w[0] >= w[1])
                    }
                    "strictmonotonic" => {
                        clean.windows(2).all(|w| w[0] < w[1])
                            || clean.windows(2).all(|w| w[0] > w[1])
                    }
                    _ => clean.windows(2).all(|w| w[0] <= w[1]),
                }
            };
            if rows_mode {
                let rows_v: Vec<Vec<f64>> = (0..nr)
                    .map(|r| (0..nc).map(|c| m.get(r, c)).collect())
                    .collect();
                rows_v.windows(2).all(|w| {
                    let mut ord = std::cmp::Ordering::Equal;
                    for (&a, &b) in w[0].iter().zip(w[1].iter()) {
                        ord = a.total_cmp(&b);
                        if ord != std::cmp::Ordering::Equal {
                            break;
                        }
                    }
                    match dir.as_str() {
                        "descend" => ord != std::cmp::Ordering::Less,
                        "strictascend" => ord == std::cmp::Ordering::Less,
                        "strictdescend" => ord == std::cmp::Ordering::Greater,
                        _ => ord != std::cmp::Ordering::Greater,
                    }
                })
            } else {
                let dim = dim_opt.unwrap_or(if nr == 1 && nc > 1 { 2 } else { 1 });
                if dim == 1 {
                    (0..nc).all(|c| {
                        let col: Vec<f64> = (0..nr).map(|r| m.get(r, c)).collect();
                        check_vec(&col)
                    })
                } else {
                    (0..nr).all(|r| {
                        let row: Vec<f64> = (0..nc).map(|c| m.get(r, c)).collect();
                        check_vec(&row)
                    })
                }
            }
        }
        _ => false,
    };
    ok1(Value::bool(res))
}

pub fn fn_mov_window(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 6 {
        return Err(MatroError::arity(name, "2 to 6", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let k_mat = args[1].to_dense_matrix()?;
    let (kb, kf) = if k_mat.len() == 1 {
        let k = k_mat.data()[0].max(1.0).round() as usize;
        (k / 2, (k - 1) / 2)
    } else {
        (
            k_mat.data()[0].max(0.0).round() as usize,
            k_mat.data()[1].max(0.0).round() as usize,
        )
    };
    let is_std_var = name == "movstd" || name == "movvar";
    let mut pop_norm = false; // w = 0 => N-1, w = 1 => N
    let mut omit_nan = false;
    let mut dim_opt: Option<usize> = None;
    let mut num_pos = 0usize;
    for a in &args[2..] {
        if let Some(s) = value_as_text(a) {
            match s.trim().to_ascii_lowercase().as_str() {
                "omitnan" | "omitmissing" => omit_nan = true,
                "includenan" | "includemissing" => omit_nan = false,
                _ => {}
            }
        } else {
            let am = a.to_dense_matrix()?;
            if is_std_var && num_pos == 0 {
                if !am.is_empty() && am.data()[0] != 0.0 {
                    pop_norm = true;
                }
            } else if !am.is_empty() {
                dim_opt = Some(as_usize(a, name)?);
            }
            num_pos += 1;
        }
    }
    let dims = m.dims();
    let dim = dim_opt.unwrap_or_else(|| dims.iter().position(|&d| d != 1).map_or(1, |i| i + 1));
    let ax = dim.saturating_sub(1);
    if ax >= dims.len() || m.is_empty() {
        return ok1(Value::matrix(m));
    }
    let apply_vec = |xs: &[f64]| -> Vec<f64> {
        let n = xs.len();
        let mut out = vec![0.0; n];
        for i in 0..n {
            let lo = i.saturating_sub(kb);
            let hi = (i + kf).min(n - 1);
            let raw = &xs[lo..=hi];
            let has_nan = raw.iter().any(|x| x.is_nan());
            let filtered: Vec<f64>;
            let win: &[f64] = if omit_nan {
                filtered = raw.iter().copied().filter(|x| !x.is_nan()).collect();
                &filtered
            } else {
                raw
            };
            if win.is_empty() {
                out[i] = match name {
                    "movsum" => 0.0,
                    _ => f64::NAN,
                };
                continue;
            }
            if !omit_nan && has_nan {
                out[i] = f64::NAN;
                continue;
            }
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
                    let denom = if pop_norm { wn } else { (wn - 1.0).max(1.0) };
                    let v = if !pop_norm && win.len() <= 1 {
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
    let inner: usize = dims[..ax].iter().product();
    let axis_len = dims[ax];
    let outer: usize = dims[ax + 1..].iter().product();
    let mut out_data = vec![0.0; m.len()];
    let mut buf = Vec::with_capacity(axis_len);
    for o in 0..outer {
        let block = o * axis_len * inner;
        for ii in 0..inner {
            buf.clear();
            for k in 0..axis_len {
                buf.push(m.data()[block + k * inner + ii]);
            }
            let res = apply_vec(&buf);
            for (k, &v) in res.iter().enumerate() {
                out_data[block + k * inner + ii] = v;
            }
        }
    }
    ok1(Value::matrix(Matrix::from_nd_col_major(dims, out_data)?))
}

pub fn fn_cum_extrema(args: &[Value], is_max: bool) -> Result<Vec<Value>> {
    let name = if is_max { "cummax" } else { "cummin" };
    if args.is_empty() || args.len() > 4 {
        return Err(MatroError::arity(name, "1 to 4 (A [, dim [, dir [, nanflag]]])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let mut reverse = false;
    let mut omit_nan = true; // MATLAB default for cummin/cummax is 'omitnan'
    let mut dim_opt: Option<usize> = None;
    for a in &args[1..] {
        if let Some(s) = value_as_text(a) {
            match s.trim().to_ascii_lowercase().as_str() {
                "reverse" => reverse = true,
                "forward" => reverse = false,
                "omitnan" | "omitmissing" => omit_nan = true,
                "includenan" | "includemissing" => omit_nan = false,
                _ => {}
            }
        } else {
            dim_opt = Some(as_usize(a, name)?);
        }
    }
    let dims = m.dims();
    if m.is_empty() {
        return ok1(Value::matrix(m));
    }
    let dim = dim_opt.unwrap_or_else(|| dims.iter().position(|&d| d != 1).map_or(1, |i| i + 1));
    let ax = dim.saturating_sub(1);
    if ax >= dims.len() {
        return ok1(Value::matrix(m));
    }
    let inner: usize = dims[..ax].iter().product();
    let axis_len = dims[ax];
    let outer: usize = dims[ax + 1..].iter().product();
    let mut out_data = vec![0.0; m.len()];
    let step = |cur: &mut f64, have: &mut bool, v: f64| -> f64 {
        if !omit_nan {
            if !*have {
                *cur = v;
                *have = true;
            } else if cur.is_nan() || v.is_nan() {
                *cur = f64::NAN;
            } else if (is_max && v > *cur) || (!is_max && v < *cur) {
                *cur = v;
            }
            *cur
        } else if v.is_nan() {
            if *have { *cur } else { f64::NAN }
        } else {
            if !*have || (is_max && v > *cur) || (!is_max && v < *cur) {
                *cur = v;
                *have = true;
            }
            *cur
        }
    };
    for o in 0..outer {
        let block = o * axis_len * inner;
        for ii in 0..inner {
            let mut cur = f64::NAN;
            let mut have = false;
            if reverse {
                for k in (0..axis_len).rev() {
                    let pos = block + k * inner + ii;
                    out_data[pos] = step(&mut cur, &mut have, m.data()[pos]);
                }
            } else {
                for k in 0..axis_len {
                    let pos = block + k * inner + ii;
                    out_data[pos] = step(&mut cur, &mut have, m.data()[pos]);
                }
            }
        }
    }
    ok1(Value::matrix(Matrix::from_nd_col_major(dims, out_data)?))
}

pub fn fn_bounds(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity("bounds", "1 to 3 (A [, dim | 'all', nanflag])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    if m.is_empty() {
        return if nargout >= 2 {
            Ok(vec![Value::matrix(m.clone()), Value::matrix(m)])
        } else {
            ok1(Value::matrix(m))
        };
    }
    let mut all = false;
    let mut omit_nan = true;
    let mut dim_opt: Option<usize> = None;
    for a in &args[1..] {
        if let Some(s) = value_as_text(a) {
            match s.trim().to_ascii_lowercase().as_str() {
                "all" => all = true,
                "omitnan" | "omitmissing" => omit_nan = true,
                "includenan" | "includemissing" => omit_nan = false,
                _ => {}
            }
        } else {
            dim_opt = Some(as_usize(a, "bounds")?);
        }
    }
    let slice_bounds = |xs: &[f64]| -> (f64, f64) {
        if !omit_nan && xs.iter().any(|x| x.is_nan()) {
            return (f64::NAN, f64::NAN);
        }
        let mut mn = f64::INFINITY;
        let mut mx = f64::NEG_INFINITY;
        let mut any_valid = false;
        for &x in xs {
            if x.is_nan() {
                continue;
            }
            any_valid = true;
            if x < mn { mn = x; }
            if x > mx { mx = x; }
        }
        if any_valid { (mn, mx) } else { (f64::NAN, f64::NAN) }
    };
    let dims = m.dims();
    let is_vec = m.ndims() == 2 && (m.rows() == 1 || m.cols() == 1);
    let (lo, hi) = if all || (dim_opt.is_none() && is_vec) {
        let (mn, mx) = slice_bounds(m.data());
        (Matrix::scalar(mn), Matrix::scalar(mx))
    } else {
        let dim = dim_opt.unwrap_or_else(|| dims.iter().position(|&d| d != 1).map_or(1, |i| i + 1));
        let ax = dim.saturating_sub(1);
        if ax >= dims.len() {
            (m.clone(), m)
        } else {
            let inner: usize = dims[..ax].iter().product();
            let axis_len = dims[ax];
            let outer: usize = dims[ax + 1..].iter().product();
            let mut out_dims = dims.clone();
            out_dims[ax] = 1;
            let mut mins = Vec::with_capacity(inner * outer);
            let mut maxs = Vec::with_capacity(inner * outer);
            let mut buf = Vec::with_capacity(axis_len);
            for o in 0..outer {
                let block = o * axis_len * inner;
                for ii in 0..inner {
                    buf.clear();
                    for k in 0..axis_len {
                        buf.push(m.data()[block + k * inner + ii]);
                    }
                    let (mn, mx) = slice_bounds(&buf);
                    mins.push(mn);
                    maxs.push(mx);
                }
            }
            (
                Matrix::from_nd_col_major(out_dims.clone(), mins)?,
                Matrix::from_nd_col_major(out_dims, maxs)?,
            )
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

fn extract_str_arr_ext(v: &Value) -> Option<(Vec<String>, usize, usize)> {
    match v {
        Value::String(s) => Some((vec![s.clone()], 1, 1)),
        Value::Cell(c) => Some((
            c.data
                .iter()
                .map(|item| value_as_text(item).unwrap_or_default())
                .collect(),
            c.rows,
            c.cols,
        )),
        Value::Object(o) if o.class_name == "string" => {
            if let Some(Value::Cell(c)) = o.fields.get("Data") {
                return Some((
                    c.data
                        .iter()
                        .map(|item| value_as_text(item).unwrap_or_default())
                        .collect(),
                    c.rows,
                    c.cols,
                ));
            }
            None
        }
        _ => None,
    }
}

pub fn fn_str_cmp_ext(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    let (n_opt, expected_arity) = match name {
        "strcmpi" => (None, 2usize),
        "strncmp" | "strncmpi" => (Some(as_usize(args.get(2).ok_or_else(|| MatroError::arity(name, "3 (s1, s2, n)", args.len()))?, name)?), 3usize),
        _ => (None, 2usize),
    };
    if args.len() != expected_arity {
        return Err(MatroError::arity(name, &expected_arity.to_string(), args.len()));
    }
    let Some((sa, ra, ca)) = extract_str_arr_ext(&args[0]) else {
        return ok1(Value::bool(false));
    };
    let Some((sb, rb, cb)) = extract_str_arr_ext(&args[1]) else {
        return ok1(Value::bool(false));
    };
    let cmp_one = |a: &str, b: &str| -> bool {
        if let Some(n) = n_opt {
            let pa: String = a.chars().take(n).collect();
            let pb: String = b.chars().take(n).collect();
            if name == "strncmpi" {
                pa.eq_ignore_ascii_case(&pb)
            } else {
                pa == pb
            }
        } else {
            a.eq_ignore_ascii_case(b)
        }
    };
    if sa.len() == 1 && sb.len() == 1 {
        return ok1(Value::bool(cmp_one(&sa[0], &sb[0])));
    }
    if sa.len() == 1 {
        let vals: Vec<f64> = sb.iter().map(|b| if cmp_one(&sa[0], b) { 1.0 } else { 0.0 }).collect();
        return ok1(Value::matrix(Matrix::from_col_major(rb, cb, vals)?.with_logical(true)));
    }
    if sb.len() == 1 {
        let vals: Vec<f64> = sa.iter().map(|a| if cmp_one(a, &sb[0]) { 1.0 } else { 0.0 }).collect();
        return ok1(Value::matrix(Matrix::from_col_major(ra, ca, vals)?.with_logical(true)));
    }
    if ra == rb && ca == cb {
        let vals: Vec<f64> = sa
            .iter()
            .zip(sb.iter())
            .map(|(a, b)| if cmp_one(a, b) { 1.0 } else { 0.0 })
            .collect();
        return ok1(Value::matrix(Matrix::from_col_major(ra, ca, vals)?.with_logical(true)));
    }
    Err(MatroError::dimension(format!("{name}: incompatible array sizes")))
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
        "dec2bin" | "dec2hex" => {
            if args.is_empty() || args.len() > 2 {
                return Err(MatroError::arity(name, "1 or 2", args.len()));
            }
            let m = args[0].to_dense_matrix()?;
            let min_w = args.get(1).map(|v| as_usize(v, name)).transpose()?.unwrap_or(0);
            let raw: Vec<String> = m
                .data()
                .iter()
                .map(|&x| {
                    let n = x.max(0.0).round() as u64;
                    if name == "dec2bin" {
                        format!("{n:0>min_w$b}")
                    } else {
                        format!("{n:0>min_w$X}")
                    }
                })
                .collect();
            if raw.len() == 1 {
                return ok1(Value::String(raw.into_iter().next().unwrap()));
            }
            let w = raw.iter().map(|s| s.len()).max().unwrap_or(min_w).max(min_w);
            let padded: Vec<Value> = raw
                .into_iter()
                .map(|s| Value::String(format!("{s:0>w$}")))
                .collect();
            ok1(Value::Cell(CellArray::from_rows(
                padded.into_iter().map(|v| vec![v]).collect(),
            )?))
        }
        "bin2dec" | "hex2dec" => {
            if args.len() != 1 {
                return Err(MatroError::arity(name, "1", args.len()));
            }
            let radix = if name == "bin2dec" { 2 } else { 16 };
            let parse_one = |s: &str| -> Result<f64> {
                let mut cleaned = s.trim();
                if radix == 16 {
                    cleaned = cleaned.trim_start_matches("0x").trim_start_matches("0X");
                }
                let v = u64::from_str_radix(cleaned, radix)
                    .map_err(|_| MatroError::runtime(format!("{name}: invalid string '{s}'")))?;
                Ok(v as f64)
            };
            if let Value::String(s) = &args[0] {
                return ok1(Value::scalar(parse_one(s)?));
            }
            if let Some((items, r, c)) = extract_str_arr_ext(&args[0]) {
                let vals: Vec<f64> = items
                    .iter()
                    .map(|s| parse_one(s))
                    .collect::<Result<_>>()?;
                return ok1(Value::matrix(Matrix::from_col_major(r, c, vals)?));
            }
            Err(MatroError::runtime(format!("{name}: expected string or cellstr")))
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
    match &args[0] {
        Value::Struct(map) => {
            let vals: Vec<Value> = map.values().cloned().collect();
            let n = vals.len();
            ok1(Value::Cell(CellArray {
                rows: n,
                cols: if n == 0 { 0 } else { 1 },
                data: vals,
            }))
        }
        Value::StructArray(sa) => {
            let keys: Vec<String> = sa
                .items
                .first()
                .map(|m| m.keys().cloned().collect())
                .unwrap_or_default();
            let nf = keys.len();
            let mut rows = Vec::with_capacity(nf);
            for k in &keys {
                let row: Vec<Value> = sa
                    .items
                    .iter()
                    .map(|m| m.get(k).cloned().unwrap_or_else(Value::empty))
                    .collect();
                rows.push(row);
            }
            ok1(Value::Cell(CellArray::from_rows(rows)?))
        }
        other => Err(MatroError::runtime(format!(
            "struct2cell: expected struct, got {}",
            other.type_name()
        ))),
    }
}

pub fn fn_cell2struct(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity("cell2struct", "2 or 3 (C, fields [, dim])", args.len()));
    }
    let Value::Cell(c) = &args[0] else {
        return Err(MatroError::runtime("cell2struct: first arg must be a cell array"));
    };
    let (fields, _, _) = extract_str_arr_ext(&args[1])
        .ok_or_else(|| MatroError::runtime("cell2struct: fields must be cellstr or string array"))?;
    let dim = args.get(2).map(|v| as_usize(v, "cell2struct")).transpose()?.unwrap_or(1);
    if dim == 1 {
        if fields.len() != c.rows {
            return Err(MatroError::dimension("cell2struct: field count must match size(C, 1)"));
        }
        let mut items = Vec::with_capacity(c.cols);
        for col in 0..c.cols {
            let mut map = BTreeMap::new();
            for (row, fname) in fields.iter().enumerate() {
                map.insert(fname.clone(), c.get(row, col).clone());
            }
            items.push(map);
        }
        if items.len() == 1 {
            ok1(Value::Struct(items.into_iter().next().unwrap()))
        } else {
            ok1(Value::StructArray(matro_types::StructArray {
                rows: 1,
                cols: items.len(),
                items,
            }))
        }
    } else {
        if fields.len() != c.cols {
            return Err(MatroError::dimension("cell2struct: field count must match size(C, 2)"));
        }
        let mut items = Vec::with_capacity(c.rows);
        for row in 0..c.rows {
            let mut map = BTreeMap::new();
            for (col, fname) in fields.iter().enumerate() {
                map.insert(fname.clone(), c.get(row, col).clone());
            }
            items.push(map);
        }
        if items.len() == 1 {
            ok1(Value::Struct(items.into_iter().next().unwrap()))
        } else {
            ok1(Value::StructArray(matro_types::StructArray {
                rows: items.len(),
                cols: 1,
                items,
            }))
        }
    }
}

pub fn fn_mat2cell(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 {
        return Err(MatroError::arity("mat2cell", ">= 2 (A, rowDist [, colDist])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let r_dist: Vec<usize> = args[1]
        .to_dense_matrix()?
        .data()
        .iter()
        .map(|&x| x.max(0.0).round() as usize)
        .collect();
    let c_dist: Vec<usize> = if args.len() >= 3 {
        args[2]
            .to_dense_matrix()?
            .data()
            .iter()
            .map(|&x| x.max(0.0).round() as usize)
            .collect()
    } else {
        vec![m.cols()]
    };
    if r_dist.iter().sum::<usize>() != m.rows() || c_dist.iter().sum::<usize>() != m.cols() {
        return Err(MatroError::dimension("mat2cell: partition sizes must sum to array dimensions"));
    }
    let mut rows = Vec::with_capacity(r_dist.len());
    let mut r0 = 0usize;
    for &rh in &r_dist {
        let mut row_cells = Vec::with_capacity(c_dist.len());
        let mut c0 = 0usize;
        for &cw in &c_dist {
            let mut block = Matrix::zeros(rh, cw);
            for cc in 0..cw {
                for rr in 0..rh {
                    block.set_c(rr, cc, m.get_c(r0 + rr, c0 + cc));
                }
            }
            row_cells.push(Value::matrix(block.normalize()));
            c0 += cw;
        }
        rows.push(row_cells);
        r0 += rh;
    }
    ok1(Value::Cell(CellArray::from_rows(rows)?))
}

pub fn fn_getfield(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 {
        return Err(MatroError::arity("getfield", ">= 2 (S, field, ...)", args.len()));
    }
    let mut cur = args[0].clone();
    for arg in &args[1..] {
        let f = value_as_text(arg)
            .ok_or_else(|| MatroError::runtime("getfield: field name must be a string"))?;
        cur = match cur {
            Value::Struct(map) => map
                .get(&f)
                .cloned()
                .ok_or_else(|| MatroError::runtime(format!("Reference to non-existent field '{f}'")))?,
            Value::StructArray(sa) => sa
                .items
                .first()
                .and_then(|m| m.get(&f))
                .cloned()
                .ok_or_else(|| MatroError::runtime(format!("Reference to non-existent field '{f}'")))?,
            Value::Object(o) => o
                .fields
                .get(&f)
                .cloned()
                .ok_or_else(|| MatroError::runtime(format!("Reference to non-existent property '{f}'")))?,
            other => {
                return Err(MatroError::runtime(format!(
                    "getfield: expected struct, got {}",
                    other.type_name()
                )))
            }
        };
    }
    ok1(cur)
}

pub fn fn_setfield(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 3 {
        return Err(MatroError::arity("setfield", "3 (S, field, value)", args.len()));
    }
    let f = value_as_text(&args[1])
        .ok_or_else(|| MatroError::runtime("setfield: field name must be a string"))?;
    match &args[0] {
        Value::Struct(map) => {
            let mut m = map.clone();
            m.insert(f, args[2].clone());
            ok1(Value::Struct(m))
        }
        other => Err(MatroError::runtime(format!(
            "setfield: expected struct, got {}",
            other.type_name()
        ))),
    }
}

pub fn fn_shiftdim(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("shiftdim", "1 or 2 (A [, n])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let dims = m.dims();
    let reshape_mat = |mat: &Matrix, nd: Vec<usize>| -> Result<Matrix> {
        if let Some(im) = mat.imag_data() {
            Matrix::from_nd_complex_col_major(nd, mat.data().to_vec(), im.to_vec())
        } else {
            Matrix::from_nd_col_major(nd, mat.data().to_vec())
        }
    };
    if args.len() == 1 {
        let lead = dims.iter().take_while(|&&d| d == 1).count();
        let mut new_dims = dims[lead..].to_vec();
        while new_dims.len() < 2 {
            new_dims.push(1);
        }
        let out = reshape_mat(&m, new_dims)?;
        return if nargout >= 2 {
            Ok(vec![Value::matrix(out), Value::scalar(lead as f64)])
        } else {
            ok1(Value::matrix(out))
        };
    }
    let n = args[1].as_scalar()?.round() as isize;
    if n < 0 {
        let mut new_dims = vec![1usize; (-n) as usize];
        new_dims.extend_from_slice(&dims);
        ok1(Value::matrix(reshape_mat(&m, new_dims)?))
    } else if n == 0 {
        ok1(Value::matrix(m))
    } else {
        let shift = (n as usize) % dims.len();
        let mut perm: Vec<usize> = (shift..dims.len()).chain(0..shift).collect();
        let order: Vec<f64> = perm.drain(..).map(|ax| (ax + 1) as f64).collect();
        crate::extras::fn_permute(&[Value::matrix(m), Value::matrix(Matrix::row_vec(order))])
    }
}

pub fn fn_pagetranspose(args: &[Value], conj: bool) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity(if conj { "pagectranspose" } else { "pagetranspose" }, "1", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let dims = m.dims();
    let (r, c) = (dims[0], dims[1]);
    let pages: usize = dims[2..].iter().product();
    let mut out_dims = dims.clone();
    out_dims[0] = c;
    out_dims[1] = r;
    let page_sz = r * c;
    let mut re = vec![0.0; m.len()];
    let mut im = vec![0.0; m.len()];
    for p in 0..pages {
        let off = p * page_sz;
        for col in 0..c {
            for row in 0..r {
                let mut v = m.get_linear_c(off + col * r + row);
                if conj {
                    v = v.conj();
                }
                let dst = off + row * c + col;
                re[dst] = v.re;
                im[dst] = v.im;
            }
        }
    }
    ok1(Value::matrix(
        Matrix::from_nd_complex_col_major(out_dims, re, im)?.normalize(),
    ))
}

pub fn fn_pagemtimes(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("pagemtimes", "2 (X, Y)", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let y = args[1].to_dense_matrix()?;
    let dx = x.dims();
    let dy = y.dims();
    let (mx, kx) = (dx[0], dx[1]);
    let (ky, ny) = (dy[0], dy[1]);
    if kx != ky {
        return Err(MatroError::dimension(format!(
            "pagemtimes: inner dimensions must agree ({mx}x{kx} * {ky}x{ny})"
        )));
    }
    let nd = dx.len().max(dy.len());
    let mut page_dims = Vec::with_capacity(nd.saturating_sub(2));
    for ax in 2..nd {
        let a = dx.get(ax).copied().unwrap_or(1);
        let b = dy.get(ax).copied().unwrap_or(1);
        if a != b && a != 1 && b != 1 {
            return Err(MatroError::dimension("pagemtimes: incompatible page dimensions"));
        }
        page_dims.push(a.max(b));
    }
    let mut out_dims = vec![mx, ny];
    out_dims.extend_from_slice(&page_dims);
    let n_pages: usize = page_dims.iter().product::<usize>().max(1);
    let mut sx = vec![0usize; page_dims.len()];
    let mut sy = vec![0usize; page_dims.len()];
    let mut cx = 1usize;
    let mut cy = 1usize;
    for (i, _) in page_dims.iter().enumerate() {
        let a = dx.get(2 + i).copied().unwrap_or(1);
        let b = dy.get(2 + i).copied().unwrap_or(1);
        sx[i] = if a == 1 { 0 } else { cx };
        sy[i] = if b == 1 { 0 } else { cy };
        cx *= a;
        cy *= b;
    }
    let x_psz = mx * kx;
    let y_psz = ky * ny;
    let o_psz = mx * ny;
    let mut re = vec![0.0; o_psz * n_pages];
    let mut im = vec![0.0; o_psz * n_pages];
    let mut coord = vec![0usize; page_dims.len()];
    for p in 0..n_pages {
        let mut px = 0usize;
        let mut py = 0usize;
        for i in 0..page_dims.len() {
            px += coord[i] * sx[i];
            py += coord[i] * sy[i];
        }
        let x_off = px * x_psz;
        let y_off = py * y_psz;
        let o_off = p * o_psz;
        for j in 0..ny {
            for i in 0..mx {
                let mut sum = Complex::ZERO;
                for kk in 0..kx {
                    sum = sum
                        + x.get_linear_c(x_off + kk * mx + i)
                            * y.get_linear_c(y_off + j * ky + kk);
                }
                re[o_off + j * mx + i] = sum.re;
                im[o_off + j * mx + i] = sum.im;
            }
        }
        for i in 0..page_dims.len() {
            coord[i] += 1;
            if coord[i] < page_dims[i] {
                break;
            }
            coord[i] = 0;
        }
    }
    ok1(Value::matrix(
        Matrix::from_nd_complex_col_major(out_dims, re, im)?.normalize(),
    ))
}

pub fn fn_mink_maxk(args: &[Value], nargout: usize, is_max: bool) -> Result<Vec<Value>> {
    let name = if is_max { "maxk" } else { "mink" };
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity(name, "2 or 3 (A, k [, dim])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let k_req = as_usize(&args[1], name)?;
    let dims = m.dims();
    let dim = if args.len() == 3 {
        as_usize(&args[2], name)?
    } else {
        dims.iter().position(|&d| d != 1).map_or(1, |i| i + 1)
    };
    let ax = dim.saturating_sub(1);
    if ax >= dims.len() {
        return ok1(Value::matrix(m));
    }
    let inner: usize = dims[..ax].iter().product();
    let axis_len = dims[ax];
    let outer: usize = dims[ax + 1..].iter().product();
    let k_out = k_req.min(axis_len);
    let mut out_dims = dims.clone();
    out_dims[ax] = k_out;
    let mut out_val = vec![0.0; inner * k_out * outer];
    let mut out_idx = vec![0.0; inner * k_out * outer];
    let mut pairs = Vec::with_capacity(axis_len);
    for o in 0..outer {
        let s_block = o * axis_len * inner;
        let d_block = o * k_out * inner;
        for ii in 0..inner {
            pairs.clear();
            for j in 0..axis_len {
                pairs.push((m.data()[s_block + j * inner + ii], j + 1));
            }
            pairs.sort_by(|a, b| match (a.0.is_nan(), b.0.is_nan()) {
                (true, true) => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                (false, false) => {
                    let ord = a.0.total_cmp(&b.0);
                    if is_max { ord.reverse() } else { ord }
                }
            });
            for (j, &(v, idx)) in pairs.iter().take(k_out).enumerate() {
                out_val[d_block + j * inner + ii] = v;
                out_idx[d_block + j * inner + ii] = idx as f64;
            }
        }
    }
    let v_mat = Matrix::from_nd_col_major(out_dims.clone(), out_val)?;
    if nargout >= 2 {
        Ok(vec![
            Value::matrix(v_mat),
            Value::matrix(Matrix::from_nd_col_major(out_dims, out_idx)?),
        ])
    } else {
        ok1(Value::matrix(v_mat))
    }
}

pub fn fn_topkrows(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 4 {
        return Err(MatroError::arity("topkrows", "2 to 4 (A, k [, col [, dir]])", args.len()));
    }
    let k = as_usize(&args[1], "topkrows")?;
    let mut sr_args = vec![args[0].clone()];
    if args.len() >= 3 {
        sr_args.push(args[2].clone());
    }
    if args.len() >= 4 {
        sr_args.push(args[3].clone());
    } else if args.len() == 2 {
        sr_args.push(Value::String("descend".to_string()));
    }
    let sorted = crate::tables::fn_sortrows(&sr_args, 2)?;
    let sm = sorted[0].to_dense_matrix()?;
    let si = sorted[1].to_dense_matrix()?;
    let take = k.min(sm.rows());
    let mut out = Matrix::zeros(take, sm.cols());
    for c in 0..sm.cols() {
        for r in 0..take {
            out.set_c(r, c, sm.get_c(r, c));
        }
    }
    let idx: Vec<f64> = si.data().iter().copied().take(take).collect();
    if nargout >= 2 {
        Ok(vec![Value::matrix(out.normalize()), Value::matrix(Matrix::col_vec(idx))])
    } else {
        ok1(Value::matrix(out.normalize()))
    }
}

pub fn fn_perms(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("perms", "1 (v)", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let n = m.len();
    if n > 10 {
        return Err(MatroError::runtime("perms: vector length > 10 exceeds safe size"));
    }
    let mut vals: Vec<f64> = m.data().to_vec();
    let mut rows: Vec<Vec<f64>> = Vec::new();
    fn build_perms(k: usize, arr: &mut [f64], out: &mut Vec<Vec<f64>>) {
        if k == arr.len() {
            out.push(arr.to_vec());
            return;
        }
        for i in k..arr.len() {
            arr.swap(k, i);
            build_perms(k + 1, arr, out);
            arr.swap(k, i);
        }
    }
    build_perms(0, &mut vals, &mut rows);
    let mut out = Matrix::zeros(rows.len(), n);
    for (r, row) in rows.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            out.set(r, c, v);
        }
    }
    ok1(Value::matrix(out))
}

pub fn fn_islocal_extrema(args: &[Value], is_max: bool) -> Result<Vec<Value>> {
    let name = if is_max { "islocalmax" } else { "islocalmin" };
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity(name, "1 or 2 (A [, dim])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let dims = m.dims();
    let dim = if args.len() == 2 {
        as_usize(&args[1], name)?
    } else {
        dims.iter().position(|&d| d != 1).map_or(1, |i| i + 1)
    };
    let ax = dim.saturating_sub(1);
    if ax >= dims.len() || dims[ax] < 3 {
        return ok1(Value::matrix(Matrix::zeros_nd(dims).with_logical(true)));
    }
    let inner: usize = dims[..ax].iter().product();
    let axis_len = dims[ax];
    let outer: usize = dims[ax + 1..].iter().product();
    let mut out = vec![0.0; m.len()];
    for o in 0..outer {
        let block = o * axis_len * inner;
        for ii in 0..inner {
            for k in 1..(axis_len - 1) {
                let prev = m.data()[block + (k - 1) * inner + ii];
                let cur = m.data()[block + k * inner + ii];
                let next = m.data()[block + (k + 1) * inner + ii];
                let hit = if is_max {
                    cur > prev && cur > next
                } else {
                    cur < prev && cur < next
                };
                if hit {
                    out[block + k * inner + ii] = 1.0;
                }
            }
        }
    }
    ok1(Value::matrix(Matrix::from_nd_col_major(dims, out)?.with_logical(true)))
}

pub fn fn_smoothdata(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("smoothdata", ">= 1 (A [, dim | method, window])", 0));
    }
    let mut method = "movmean".to_string();
    let mut win: Option<Value> = None;
    let mut dim_val: Option<Value> = None;
    for a in &args[1..] {
        if let Some(s) = value_as_text(a) {
            let low = s.trim().to_ascii_lowercase();
            if matches!(low.as_str(), "movmean" | "movmedian" | "gaussian" | "sgolay" | "lowess") {
                method = low;
            }
        } else if win.is_none() && method != "movmean" {
            win = Some(a.clone());
        } else if dim_val.is_none() {
            dim_val = Some(a.clone());
        } else if win.is_none() {
            win = Some(a.clone());
        }
    }
    let k = win.unwrap_or_else(|| Value::scalar(5.0));
    let mut call_a = vec![args[0].clone(), k];
    if let Some(d) = dim_val {
        call_a.push(d);
    }
    let target = if method == "movmedian" { "movmedian" } else { "movmean" };
    fn_mov_window(target, &call_a)
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
