//! Extended MATLAB/Octave standard library intrinsics:
//! N-D array manipulation, polynomials, set operations, matrix utilities,
//! pure-Rust JSON encode/decode, string utilities, and histogram/contour plots.

use std::collections::BTreeMap;

use matro_types::plot::json_str;
use matro_types::{CellArray, Complex, MatroError, Matrix, PlotFigure, PlotKind, Result, Value};

fn ok1(v: Value) -> Result<Vec<Value>> {
    Ok(vec![v])
}

fn as_usize(v: &Value, ctx: &str) -> Result<usize> {
    let s = v.as_scalar()?;
    if s < 0.0 || !s.is_finite() {
        return Err(MatroError::runtime(format!(
            "{ctx}: expected non-negative integer, got {s}"
        )));
    }
    Ok(s.round() as usize)
}

/// `ndims(A)`
pub fn fn_ndims(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("ndims", "1", args.len()));
    }
    let d = match &args[0] {
        Value::Matrix(m) => m.ndims(),
        _ => 2,
    };
    ok1(Value::scalar(d as f64))
}

/// `squeeze(A)` — remove singleton dimensions (preserving at least 2D).
pub fn fn_squeeze(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("squeeze", "1", args.len()));
    }
    let m = args[0].as_matrix()?;
    if m.ndims() <= 2 {
        return ok1(Value::matrix(m.clone()));
    }
    let mut new_dims: Vec<usize> = m.dims().into_iter().filter(|&d| d != 1).collect();
    while new_dims.len() < 2 {
        new_dims.push(1);
    }
    ok1(Value::matrix(m.clone().with_dims(new_dims)?))
}

/// `permute(A, order)` — permute N-D array dimensions (1-based `order`).
pub fn fn_permute(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("permute", "2", args.len()));
    }
    let m = args[0].as_matrix()?;
    let ord_m = args[1].as_matrix()?;
    let order: Vec<usize> = ord_m
        .data()
        .iter()
        .map(|&x| {
            let i = x.round() as isize;
            if i < 1 {
                Err(MatroError::index("permute: order indices must be >= 1"))
            } else {
                Ok((i - 1) as usize)
            }
        })
        .collect::<Result<_>>()?;

    let nd = order.len().max(m.ndims());
    let mut in_dims = m.dims();
    in_dims.resize(nd, 1);

    let mut out_dims = vec![1usize; order.len()];
    for (new_ax, &old_ax) in order.iter().enumerate() {
        out_dims[new_ax] = in_dims.get(old_ax).copied().unwrap_or(1);
    }

    let in_strides = col_major_strides(&in_dims);
    let total: usize = out_dims.iter().product();
    let mut out_re = vec![0.0; total];
    let mut out_im = if m.is_complex() {
        Some(vec![0.0; total])
    } else {
        None
    };

    let mut out_coord = vec![0usize; out_dims.len()];
    for out_idx in 0..total {
        // Map out_coord -> in_coord
        let mut in_idx = 0usize;
        for (new_ax, &old_ax) in order.iter().enumerate() {
            if old_ax < in_strides.len() {
                in_idx += out_coord[new_ax] * in_strides[old_ax];
            }
        }
        let c = m.get_linear_c(in_idx);
        out_re[out_idx] = c.re;
        if let Some(im_vec) = &mut out_im {
            im_vec[out_idx] = c.im;
        }
        // Increment column-major out_coord
        for ax in 0..out_dims.len() {
            out_coord[ax] += 1;
            if out_coord[ax] < out_dims[ax] {
                break;
            }
            out_coord[ax] = 0;
        }
    }

    let res = if let Some(im) = out_im {
        Matrix::from_nd_complex_col_major(out_dims, out_re, im)?
    } else {
        Matrix::from_nd_col_major(out_dims, out_re)?
    };
    ok1(Value::matrix(res))
}

/// `cat(dim, A, B, ...)` — concatenate arrays along arbitrary 1-based dimension `dim`.
pub fn fn_cat(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 {
        return Err(MatroError::arity("cat", ">= 2 (dim, A, ...)", args.len()));
    }
    let dim = as_usize(&args[0], "cat")?;
    if dim < 1 {
        return Err(MatroError::index("cat: dim must be >= 1"));
    }
    let axis = dim - 1;
    let mats: Vec<Matrix> = args[1..]
        .iter()
        .map(|v| v.to_dense_matrix())
        .collect::<Result<_>>()?;
    let non_empty: Vec<&Matrix> = mats.iter().filter(|m| !m.is_empty()).collect();
    if non_empty.is_empty() {
        return ok1(Value::empty());
    }
    if axis == 0 && non_empty.iter().all(|m| m.ndims() <= 2) {
        return ok1(Value::matrix(Matrix::vcat(&non_empty)?));
    }
    if axis == 1 && non_empty.iter().all(|m| m.ndims() <= 2) {
        return ok1(Value::matrix(Matrix::hcat(&non_empty)?));
    }

    let nd = non_empty
        .iter()
        .map(|m| m.ndims())
        .max()
        .unwrap_or(2)
        .max(axis + 1);
    let mut base_dims = non_empty[0].dims();
    base_dims.resize(nd, 1);

    let mut total_axis = 0usize;
    let mut any_complex = false;
    for m in &non_empty {
        let mut d = m.dims();
        d.resize(nd, 1);
        for ax in 0..nd {
            if ax != axis && d[ax] != base_dims[ax] {
                return Err(MatroError::dimension(format!(
                    "cat: dimension mismatch along axis {} ({} vs {})",
                    ax + 1,
                    d[ax],
                    base_dims[ax]
                )));
            }
        }
        total_axis += d[axis];
        any_complex |= m.is_complex();
    }

    let mut out_dims = base_dims;
    out_dims[axis] = total_axis;
    let out_strides = col_major_strides(&out_dims);
    let total: usize = out_dims.iter().product();
    let mut out_re = vec![0.0; total];
    let mut out_im = if any_complex {
        Some(vec![0.0; total])
    } else {
        None
    };

    let mut axis_offset = 0usize;
    for m in &non_empty {
        let mut d = m.dims();
        d.resize(nd, 1);
        let mut coord = vec![0usize; nd];
        for idx in 0..m.len() {
            let mut dst = 0usize;
            for ax in 0..nd {
                let c = coord[ax] + if ax == axis { axis_offset } else { 0 };
                dst += c * out_strides[ax];
            }
            let val = m.get_linear_c(idx);
            out_re[dst] = val.re;
            if let Some(im) = &mut out_im {
                im[dst] = val.im;
            }
            for ax in 0..nd {
                coord[ax] += 1;
                if coord[ax] < d[ax] {
                    break;
                }
                coord[ax] = 0;
            }
        }
        axis_offset += d[axis];
    }

    let res = if let Some(im) = out_im {
        Matrix::from_nd_complex_col_major(out_dims, out_re, im)?
    } else {
        Matrix::from_nd_col_major(out_dims, out_re)?
    };
    ok1(Value::matrix(res))
}

pub fn col_major_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(dims.len());
    let mut s = 1usize;
    for &d in dims {
        strides.push(s);
        s = s.saturating_mul(d);
    }
    strides
}

/// `roots(c)` — polynomial roots via Frobenius companion matrix eigenvalues.
pub fn fn_roots(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("roots", "1", args.len()));
    }
    let raw = args[0].as_matrix()?.data();
    // Strip leading zeros
    let first_nz = raw.iter().position(|&v| v.abs() > 0.0);
    let Some(start) = first_nz else {
        return ok1(Value::empty());
    };
    let mut coeffs = raw[start..].to_vec();
    // Count and strip trailing zeros (roots at 0)
    let mut zero_roots = 0usize;
    while coeffs.len() > 1 && coeffs.last().copied().unwrap_or(0.0).abs() == 0.0 {
        coeffs.pop();
        zero_roots += 1;
    }
    let deg = coeffs.len().saturating_sub(1);
    let mut roots = Vec::with_capacity(deg + zero_roots);
    if deg == 1 {
        roots.push(Complex::from_real(-coeffs[1] / coeffs[0]));
    } else if deg >= 2 {
        let lead = coeffs[0];
        let mut comp = Matrix::zeros(deg, deg);
        for c in 0..deg {
            comp.set(0, c, -coeffs[c + 1] / lead);
        }
        for i in 1..deg {
            comp.set(i, i - 1, 1.0);
        }
        let (_, d) = comp.eig()?;
        for i in 0..deg {
            roots.push(d.get_c(i, i));
        }
    }
    for _ in 0..zero_roots {
        roots.push(Complex::ZERO);
    }
    ok1(Value::matrix(Matrix::complex_col_vec(roots).normalize()))
}

/// `poly(r)` — polynomial coefficients from roots vector or square matrix characteristic polynomial.
pub fn fn_poly(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("poly", "1", args.len()));
    }
    let m = args[0].as_matrix()?;
    let roots: Vec<Complex> = if m.rows() == m.cols() && m.rows() > 1 {
        let (_, d) = m.eig()?;
        (0..m.rows()).map(|i| d.get_c(i, i)).collect()
    } else {
        (0..m.len()).map(|i| m.get_linear_c(i)).collect()
    };
    let mut p = vec![Complex::ONE];
    for r in roots {
        let mut next = vec![Complex::ZERO; p.len() + 1];
        for i in 0..p.len() {
            next[i] = next[i] + p[i];
            next[i + 1] = next[i + 1] - p[i] * r;
        }
        p = next;
    }
    ok1(Value::matrix(Matrix::complex_row_vec(p).normalize()))
}

/// Set operations: `unique`, `ismember`, `setdiff`, `intersect`, `union`.
pub fn fn_unique(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("unique", "1", args.len()));
    }
    let m = args[0].as_matrix()?;
    let mut vals = m.data().to_vec();
    vals.sort_by(f64::total_cmp);
    vals.dedup_by(|a, b| (*a - *b).abs() == 0.0);
    let out = if m.rows() == 1 {
        Matrix::row_vec(vals)
    } else {
        Matrix::col_vec(vals)
    };
    ok1(Value::matrix(out))
}

pub fn fn_ismember(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("ismember", "2", args.len()));
    }
    let a = args[0].as_matrix()?;
    let b = args[1].as_matrix()?.data();
    let out = a
        .map(|x| if b.iter().any(|&y| x == y) { 1.0 } else { 0.0 })
        .with_logical(true);
    ok1(Value::matrix(out))
}

pub fn fn_set_binary(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity(name, "2", args.len()));
    }
    let a = args[0].as_matrix()?;
    let b = args[1].as_matrix()?;
    let mut av = a.data().to_vec();
    let mut bv = b.data().to_vec();
    av.sort_by(f64::total_cmp);
    av.dedup();
    bv.sort_by(f64::total_cmp);
    bv.dedup();

    let res: Vec<f64> = match name {
        "setdiff" => av.into_iter().filter(|x| !bv.contains(x)).collect(),
        "intersect" => av.into_iter().filter(|x| bv.contains(x)).collect(),
        "union" => {
            av.extend(bv);
            av.sort_by(f64::total_cmp);
            av.dedup();
            av
        }
        _ => unreachable!(),
    };
    let out = if a.rows() == 1 {
        Matrix::row_vec(res)
    } else {
        Matrix::col_vec(res)
    };
    ok1(Value::matrix(out))
}

/// Matrix reshaping & structural utilities: `repmat`, `fliplr`, `flipud`, `rot90`, `tril`, `triu`.
pub fn fn_repmat(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity("repmat", "2 or 3", args.len()));
    }
    let m = args[0].as_matrix()?;
    let (rm, cm) = if args.len() == 2 {
        let rep = args[1].as_matrix()?;
        if rep.len() == 1 {
            let k = as_usize(&args[1], "repmat")?;
            (k, k)
        } else {
            (rep.data()[0].round() as usize, rep.data()[1].round() as usize)
        }
    } else {
        (as_usize(&args[1], "repmat")?, as_usize(&args[2], "repmat")?)
    };
    let mut out = Matrix::zeros(m.rows() * rm, m.cols() * cm);
    for br in 0..rm {
        for bc in 0..cm {
            for r in 0..m.rows() {
                for c in 0..m.cols() {
                    out.set_c(br * m.rows() + r, bc * m.cols() + c, m.get_c(r, c));
                }
            }
        }
    }
    ok1(Value::matrix(out.normalize()))
}

pub fn fn_fliplr(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("fliplr", "1", args.len()));
    }
    let m = args[0].as_matrix()?;
    let mut out = Matrix::zeros(m.rows(), m.cols());
    for r in 0..m.rows() {
        for c in 0..m.cols() {
            out.set_c(r, m.cols() - 1 - c, m.get_c(r, c));
        }
    }
    ok1(Value::matrix(out.normalize()))
}

pub fn fn_flipud(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("flipud", "1", args.len()));
    }
    let m = args[0].as_matrix()?;
    let mut out = Matrix::zeros(m.rows(), m.cols());
    for r in 0..m.rows() {
        for c in 0..m.cols() {
            out.set_c(m.rows() - 1 - r, c, m.get_c(r, c));
        }
    }
    ok1(Value::matrix(out.normalize()))
}

pub fn fn_rot90(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("rot90", "1 or 2", args.len()));
    }
    let mut m = args[0].as_matrix()?.clone();
    let k = if args.len() == 2 {
        args[1].as_scalar()?.round() as i64
    } else {
        1
    };
    let steps = k.rem_euclid(4) as usize;
    for _ in 0..steps {
        let mut next = Matrix::zeros(m.cols(), m.rows());
        for r in 0..m.rows() {
            for c in 0..m.cols() {
                next.set_c(m.cols() - 1 - c, r, m.get_c(r, c));
            }
        }
        m = next.normalize();
    }
    ok1(Value::matrix(m))
}

pub fn fn_tri(name: &str, args: &[Value], upper: bool) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity(name, "1 or 2", args.len()));
    }
    let m = args[0].as_matrix()?;
    let k = if args.len() == 2 {
        args[1].as_scalar()?.round() as isize
    } else {
        0
    };
    let mut out = Matrix::zeros(m.rows(), m.cols());
    for r in 0..m.rows() {
        for c in 0..m.cols() {
            let keep = if upper {
                (c as isize) - (r as isize) >= k
            } else {
                (c as isize) - (r as isize) <= k
            };
            if keep {
                out.set_c(r, c, m.get_c(r, c));
            }
        }
    }
    ok1(Value::matrix(out.normalize()))
}

/// String search / replace / prefix / suffix helpers.
pub fn fn_strfind(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("strfind", "2", args.len()));
    }
    let s = args[0].as_str()?;
    let pat = args[1].as_str()?;
    if pat.is_empty() {
        return ok1(Value::empty());
    }
    let idxs: Vec<f64> = s.match_indices(pat).map(|(i, _)| (i + 1) as f64).collect();
    ok1(Value::matrix(Matrix::row_vec(idxs)))
}

pub fn fn_strrep(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 3 {
        return Err(MatroError::arity("strrep", "3", args.len()));
    }
    let s = args[0].as_str()?;
    let old = args[1].as_str()?;
    let new = args[2].as_str()?;
    ok1(Value::String(s.replace(old, new)))
}

pub fn fn_str_pred(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity(name, "2", args.len()));
    }
    let s = args[0].as_str()?;
    let pat = args[1].as_str()?;
    let b = match name {
        "startsWith" => s.starts_with(pat),
        "endsWith" => s.ends_with(pat),
        "contains" => s.contains(pat),
        _ => false,
    };
    ok1(Value::bool(b))
}

/// `jsonencode(val)` — zero-dependency JSON serializer.
pub fn fn_jsonencode(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("jsonencode", "1", args.len()));
    }
    ok1(Value::String(encode_json(&args[0])))
}

fn encode_json(val: &Value) -> String {
    match val {
        Value::String(s) => json_str(s),
        Value::Matrix(m) => {
            if let Some(s) = m.as_scalar() {
                if m.is_logical() {
                    return if s != 0.0 { "true" } else { "false" }.to_string();
                }
                if s.is_finite() {
                    return format!("{s}");
                }
                return "null".to_string();
            }
            if m.rows() <= 1 || m.cols() <= 1 {
                let mut out = String::from("[");
                for (i, &v) in m.data().iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    if m.is_logical() {
                        out.push_str(if v != 0.0 { "true" } else { "false" });
                    } else if v.is_finite() {
                        out.push_str(&format!("{v}"));
                    } else {
                        out.push_str("null");
                    }
                }
                out.push(']');
                out
            } else {
                let mut out = String::from("[");
                for r in 0..m.rows() {
                    if r > 0 {
                        out.push(',');
                    }
                    out.push('[');
                    for c in 0..m.cols() {
                        if c > 0 {
                            out.push(',');
                        }
                        let v = m.get(r, c);
                        if v.is_finite() {
                            out.push_str(&format!("{v}"));
                        } else {
                            out.push_str("null");
                        }
                    }
                    out.push(']');
                }
                out.push(']');
                out
            }
        }
        Value::Sparse(s) => encode_json(&Value::matrix(s.to_dense())),
        Value::Struct(map) => encode_json_map(map),
        Value::Object(o) => encode_json_map(&o.fields),
        Value::StructArray(sa) => {
            let mut out = String::from("[");
            for (i, item) in sa.items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&encode_json_map(item));
            }
            out.push(']');
            out
        }
        Value::Cell(c) => {
            let mut out = String::from("[");
            for (i, item) in c.data.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&encode_json(item));
            }
            out.push(']');
            out
        }
        Value::Handle(h) => json_str(&format!("<{} handle>", h.class_name)),
        Value::Closure(cl) => json_str(&format!("@({})", cl.params.join(","))),
        Value::Table(t) => {
            let mut rows = Vec::with_capacity(t.nrows);
            for r in 0..t.nrows {
                let mut fields = Vec::with_capacity(t.width());
                for (c, name) in t.col_names.iter().enumerate() {
                    let elem = match &t.columns[c] {
                        Value::Matrix(m) => Value::scalar(m.get(r, 0)),
                        Value::Cell(cell) => cell.data[r].clone(),
                        other => other.clone(),
                    };
                    fields.push(format!("{}:{}", json_str(name), encode_json(&elem)));
                }
                rows.push(format!("{{{}}}", fields.join(",")));
            }
            format!("[{}]", rows.join(","))
        }
        Value::Map(m) => {
            let mut entries = Vec::with_capacity(m.entries.len());
            for (k, v) in &m.entries {
                entries.push(format!("{}:{}", json_str(k), encode_json(v)));
            }
            format!("{{{}}}", entries.join(","))
        }
    }
}

fn encode_json_map(map: &BTreeMap<String, Value>) -> String {
    let mut out = String::from("{");
    for (i, (k, v)) in map.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&json_str(k));
        out.push(':');
        out.push_str(&encode_json(v));
    }
    out.push('}');
    out
}

/// `jsondecode(text)` — zero-dependency recursive-descent JSON parser.
pub fn fn_jsondecode(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("jsondecode", "1", args.len()));
    }
    let text = args[0].as_str()?;
    let mut p = JsonParser::new(text);
    let val = p.parse_value()?;
    ok1(val)
}

struct JsonParser<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            src: s.as_bytes(),
            pos: 0,
        }
    }

    fn skip_ws(&mut self) {
        while let Some(&b) = self.src.get(self.pos) {
            if matches!(b, b' ' | b'\t' | b'\r' | b'\n') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn parse_value(&mut self) -> Result<Value> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => Ok(Value::String(self.parse_string()?)),
            Some(b't') if self.src[self.pos..].starts_with(b"true") => {
                self.pos += 4;
                Ok(Value::bool(true))
            }
            Some(b'f') if self.src[self.pos..].starts_with(b"false") => {
                self.pos += 5;
                Ok(Value::bool(false))
            }
            Some(b'n') if self.src[self.pos..].starts_with(b"null") => {
                self.pos += 4;
                Ok(Value::empty())
            }
            Some(b'-' | b'0'..=b'9') => self.parse_number(),
            _ => Err(MatroError::runtime("jsondecode: invalid JSON syntax")),
        }
    }

    fn parse_string(&mut self) -> Result<String> {
        self.pos += 1; // opening `"`
        let mut s = String::new();
        while let Some(&b) = self.src.get(self.pos) {
            self.pos += 1;
            if b == b'"' {
                return Ok(s);
            }
            if b == b'\\' {
                let esc = *self
                    .src
                    .get(self.pos)
                    .ok_or_else(|| MatroError::runtime("jsondecode: unterminated escape"))?;
                self.pos += 1;
                match esc {
                    b'"' => s.push('"'),
                    b'\\' => s.push('\\'),
                    b'/' => s.push('/'),
                    b'n' => s.push('\n'),
                    b'r' => s.push('\r'),
                    b't' => s.push('\t'),
                    other => s.push(other as char),
                }
            } else {
                s.push(b as char);
            }
        }
        Err(MatroError::runtime("jsondecode: unterminated string"))
    }

    fn parse_number(&mut self) -> Result<Value> {
        let start = self.pos;
        while let Some(&b) = self.src.get(self.pos) {
            if matches!(b, b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                self.pos += 1;
            } else {
                break;
            }
        }
        let s = std::str::from_utf8(&self.src[start..self.pos])
            .map_err(|_| MatroError::runtime("jsondecode: invalid UTF-8"))?;
        let n = s
            .parse::<f64>()
            .map_err(|_| MatroError::runtime(format!("jsondecode: invalid number '{s}'")))?;
        Ok(Value::scalar(n))
    }

    fn parse_object(&mut self) -> Result<Value> {
        self.pos += 1; // `{`
        let mut map = BTreeMap::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Value::Struct(map));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(MatroError::runtime("jsondecode: expected object key string"));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(MatroError::runtime("jsondecode: expected ':'"));
            }
            self.pos += 1;
            let val = self.parse_value()?;
            map.insert(key, val);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(MatroError::runtime("jsondecode: expected ',' or '}'")),
            }
        }
        Ok(Value::Struct(map))
    }

    fn parse_array(&mut self) -> Result<Value> {
        self.pos += 1; // `[`
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Value::empty());
        }
        loop {
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(MatroError::runtime("jsondecode: expected ',' or ']'")),
            }
        }
        // MATLAB convention: homogeneous numeric array -> Matrix; otherwise CellArray
        if items.iter().all(|v| v.as_scalar().is_ok()) {
            let nums: Vec<f64> = items.iter().map(|v| v.as_scalar().unwrap()).collect();
            return Ok(Value::matrix(Matrix::col_vec(nums)));
        }
        Ok(Value::Cell(CellArray::from_rows(vec![items])?))
    }
}

/// `histogram(x [, nbins])` — compute bin counts and plot as a bar chart.
pub fn fn_histogram(args: &[Value], figure: &mut PlotFigure) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("histogram", "1 or 2", args.len()));
    }
    let data = args[0].as_matrix()?.data();
    if data.is_empty() {
        return Ok(vec![]);
    }
    let nbins = if args.len() == 2 {
        as_usize(&args[1], "histogram")?.max(1)
    } else {
        ((data.len() as f64).sqrt().ceil() as usize).clamp(5, 30)
    };
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in data {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if (hi - lo).abs() < 1e-12 {
        lo -= 0.5;
        hi += 0.5;
    }
    let step = (hi - lo) / nbins as f64;
    let mut counts = vec![0.0; nbins];
    for &v in data {
        if v.is_finite() {
            let idx = (((v - lo) / step).floor() as isize).clamp(0, (nbins - 1) as isize) as usize;
            counts[idx] += 1.0;
        }
    }
    let centers: Vec<f64> = (0..nbins).map(|i| lo + (i as f64 + 0.5) * step).collect();
    figure.panel_mut().add_series(PlotKind::Bar, centers, counts, None);
    Ok(vec![])
}

/// `contour(Z [, levels])` or `contour(X, Y, Z [, levels])`.
pub fn fn_contour(args: &[Value], figure: &mut PlotFigure) -> Result<Vec<Value>> {
    match args.len() {
        1 | 2 => {
            let z = args[0].as_matrix()?;
            let levels = args
                .get(1)
                .map(|v| as_usize(v, "contour"))
                .transpose()?
                .unwrap_or(8);
            let x: Vec<f64> = (1..=z.cols()).map(|i| i as f64).collect();
            let y: Vec<f64> = (1..=z.rows()).map(|i| i as f64).collect();
            figure.panel_mut().set_contour(x, y, z, levels);
            Ok(vec![])
        }
        3 | 4 => {
            let xm = args[0].as_matrix()?;
            let ym = args[1].as_matrix()?;
            let z = args[2].as_matrix()?;
            let levels = args
                .get(3)
                .map(|v| as_usize(v, "contour"))
                .transpose()?
                .unwrap_or(8);
            let x: Vec<f64> = if xm.rows() == z.rows() && xm.cols() == z.cols() {
                (0..z.cols()).map(|c| xm.get(0, c)).collect()
            } else {
                xm.data().to_vec()
            };
            let y: Vec<f64> = if ym.rows() == z.rows() && ym.cols() == z.cols() {
                (0..z.rows()).map(|r| ym.get(r, 0)).collect()
            } else {
                ym.data().to_vec()
            };
            figure.panel_mut().set_contour(x, y, z, levels);
            Ok(vec![])
        }
        n => Err(MatroError::arity("contour", "1..=4", n)),
    }
}

/// `filter(b, a, x)` — MATLAB Direct Form II Transposed 1D digital filter (IIR & FIR).
pub fn fn_filter(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 3 {
        return Err(MatroError::arity("filter", "3 (b, a, x)", args.len()));
    }
    let b_raw = args[0].as_matrix()?.data();
    let a_raw = args[1].as_matrix()?.data();
    let x_mat = args[2].as_matrix()?;
    if a_raw.is_empty() || a_raw[0] == 0.0 {
        return Err(MatroError::runtime("filter: a(1) must be non-zero"));
    }
    let a0 = a_raw[0];
    let n_ord = b_raw.len().max(a_raw.len());
    let mut b = vec![0.0; n_ord];
    let mut a = vec![0.0; n_ord];
    for (i, &v) in b_raw.iter().enumerate() {
        b[i] = v / a0;
    }
    for (i, &v) in a_raw.iter().enumerate() {
        a[i] = v / a0;
    }

    let x = x_mat.data();
    let mut y = vec![0.0; x.len()];
    let mut z = vec![0.0; n_ord];

    for (n, &xn) in x.iter().enumerate() {
        let yn = b[0] * xn + z[0];
        y[n] = yn;
        for i in 1..n_ord {
            z[i - 1] = b[i] * xn + z[i] - a[i] * yn;
        }
    }
    let out = if x_mat.rows() == 1 {
        Matrix::row_vec(y)
    } else {
        Matrix::col_vec(y)
    };
    ok1(Value::matrix(out))
}

/// `interp2(X, Y, Z, Xq, Yq)` — 2D bilinear grid interpolation.
pub fn fn_interp2(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 5 {
        return Err(MatroError::arity("interp2", "5 (X, Y, Z, Xq, Yq)", args.len()));
    }
    let xm = args[0].as_matrix()?;
    let ym = args[1].as_matrix()?;
    let zm = args[2].as_matrix()?;
    let xqm = args[3].as_matrix()?;
    let yqm = args[4].as_matrix()?;
    if xqm.dims() != yqm.dims() {
        return Err(MatroError::dimension("interp2: Xq and Yq must have matching dimensions"));
    }

    let xs: Vec<f64> = if xm.rows() == zm.rows() && xm.cols() == zm.cols() {
        (0..zm.cols()).map(|c| xm.get(0, c)).collect()
    } else {
        xm.data().to_vec()
    };
    let ys: Vec<f64> = if ym.rows() == zm.rows() && ym.cols() == zm.cols() {
        (0..zm.rows()).map(|r| ym.get(r, 0)).collect()
    } else {
        ym.data().to_vec()
    };
    if xs.len() != zm.cols() || ys.len() != zm.rows() || xs.len() < 2 || ys.len() < 2 {
        return Err(MatroError::dimension("interp2: grid dimensions do not match Z"));
    }

    let find_bracket = |arr: &[f64], val: f64| -> Option<(usize, usize, f64)> {
        if val < arr[0] || val > arr[arr.len() - 1] {
            return None;
        }
        let mut lo = 0usize;
        let mut hi = arr.len() - 1;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if arr[mid] <= val {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let t = (val - arr[lo]) / (arr[hi] - arr[lo]);
        Some((lo, hi, t))
    };

    let mut out = Matrix::zeros(xqm.rows(), xqm.cols());
    for c in 0..xqm.cols() {
        for r in 0..xqm.rows() {
            let qx = xqm.get(r, c);
            let qy = yqm.get(r, c);
            let val = match (find_bracket(&xs, qx), find_bracket(&ys, qy)) {
                (Some((c0, c1, tx)), Some((r0, r1, ty))) => {
                    let z00 = zm.get(r0, c0);
                    let z10 = zm.get(r0, c1);
                    let z01 = zm.get(r1, c0);
                    let z11 = zm.get(r1, c1);
                    let top = z00 + tx * (z10 - z00);
                    let bot = z01 + tx * (z11 - z01);
                    top + ty * (bot - top)
                }
                _ => f64::NAN,
            };
            out.set(r, c, val);
        }
    }
    ok1(Value::matrix(out))
}


/// 2D convolution: `conv2(A, B [, 'full'|'same'|'valid'])`.
pub fn fn_conv2(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity("conv2", "2 or 3", args.len()));
    }
    let a = args[0].as_matrix()?;
    let b = args[1].as_matrix()?;
    let shape = args
        .get(2)
        .map(|v| v.as_str())
        .transpose()?
        .unwrap_or("full");
    let (ar, ac) = (a.rows(), a.cols());
    let (br, bc) = (b.rows(), b.cols());
    if ar == 0 || ac == 0 || br == 0 || bc == 0 {
        return ok1(Value::empty());
    }
    let fr = ar + br - 1;
    let fc = ac + bc - 1;
    let mut full = Matrix::zeros(fr, fc);
    for i in 0..ar {
        for j in 0..ac {
            let av = a.get_c(i, j);
            for p in 0..br {
                for q in 0..bc {
                    let cur = full.get_c(i + p, j + q);
                    full.set_c(i + p, j + q, cur + av * b.get_c(p, q));
                }
            }
        }
    }
    match shape {
        "full" => ok1(Value::matrix(full.normalize())),
        "same" => {
            let r0 = br / 2;
            let c0 = bc / 2;
            let mut out = Matrix::zeros(ar, ac);
            for r in 0..ar {
                for c in 0..ac {
                    out.set_c(r, c, full.get_c(r + r0, c + c0));
                }
            }
            ok1(Value::matrix(out.normalize()))
        }
        "valid" => {
            if ar < br || ac < bc {
                return ok1(Value::empty());
            }
            let vr = ar - br + 1;
            let vc = ac - bc + 1;
            let mut out = Matrix::zeros(vr, vc);
            for r in 0..vr {
                for c in 0..vc {
                    out.set_c(r, c, full.get_c(r + br - 1, c + bc - 1));
                }
            }
            ok1(Value::matrix(out.normalize()))
        }
        other => Err(MatroError::runtime(format!(
            "conv2: unknown shape '{other}' (expected 'full', 'same', or 'valid')"
        ))),
    }
}

/// Cumulative product: `cumprod(x)`.
pub fn fn_cumprod(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity("cumprod", "1 to 3 (A [, dim [, direction]])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let mut reverse = false;
    let mut dim_opt: Option<usize> = None;
    for a in &args[1..] {
        if let Some(s) = crate::value_as_text(a) {
            if s.eq_ignore_ascii_case("reverse") {
                reverse = true;
            }
        } else {
            dim_opt = Some(as_usize(a, "cumprod")?);
        }
    }
    let dim = dim_opt.unwrap_or(if m.rows() == 1 && m.cols() > 1 { 2 } else { 1 });
    let mut out = Matrix::zeros(m.rows(), m.cols());
    if dim == 1 {
        for c in 0..m.cols() {
            let mut prod = Complex::ONE;
            if reverse {
                for r in (0..m.rows()).rev() {
                    prod = prod * m.get_c(r, c);
                    out.set_c(r, c, prod);
                }
            } else {
                for r in 0..m.rows() {
                    prod = prod * m.get_c(r, c);
                    out.set_c(r, c, prod);
                }
            }
        }
    } else if dim == 2 {
        for r in 0..m.rows() {
            let mut prod = Complex::ONE;
            if reverse {
                for c in (0..m.cols()).rev() {
                    prod = prod * m.get_c(r, c);
                    out.set_c(r, c, prod);
                }
            } else {
                for c in 0..m.cols() {
                    prod = prod * m.get_c(r, c);
                    out.set_c(r, c, prod);
                }
            }
        }
    } else {
        return Err(MatroError::runtime(format!("cumprod: unsupported dimension {dim}")));
    }
    ok1(Value::matrix(out.normalize()))
}
