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

/// `roots(c)` — polynomial roots (real or complex) via Frobenius companion matrix eigenvalues.
pub fn fn_roots(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("roots", "1", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let raw: Vec<Complex> = (0..m.len()).map(|i| m.get_linear_c(i)).collect();
    let first_nz = raw.iter().position(|c| c.abs() > 0.0);
    let Some(start) = first_nz else {
        return ok1(Value::empty());
    };
    let mut coeffs = raw[start..].to_vec();
    let mut zero_roots = 0usize;
    while coeffs.len() > 1 && coeffs.last().copied().unwrap_or(Complex::ZERO).abs() == 0.0 {
        coeffs.pop();
        zero_roots += 1;
    }
    let deg = coeffs.len().saturating_sub(1);
    let mut roots = Vec::with_capacity(deg + zero_roots);
    if deg == 1 {
        roots.push(-coeffs[1] / coeffs[0]);
    } else if deg >= 2 {
        let lead = coeffs[0];
        let mut comp = Matrix::zeros(deg, deg);
        for c in 0..deg {
            comp.set_c(0, c, -coeffs[c + 1] / lead);
        }
        for i in 1..deg {
            comp.set_c(i, i - 1, Complex::ONE);
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

fn extract_text_collection(v: &Value) -> Option<(Vec<String>, usize, usize)> {
    match v {
        Value::String(s) => Some((vec![s.clone()], 1, 1)),
        Value::Cell(c) => {
            if c.data.iter().all(|item| crate::value_as_text(item).is_some()) {
                let items = c
                    .data
                    .iter()
                    .map(|item| crate::value_as_text(item).unwrap_or_default())
                    .collect();
                Some((items, c.rows, c.cols))
            } else {
                None
            }
        }
        Value::Object(o) if o.class_name == "string" => {
            if let Some(Value::Cell(c)) = o.fields.get("Data") {
                let items = c
                    .data
                    .iter()
                    .map(|item| crate::value_as_text(item).unwrap_or_default())
                    .collect();
                Some((items, c.rows, c.cols))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn parse_set_flags(args: &[Value]) -> (bool, bool) {
    let mut rows_mode = false;
    let mut stable = false;
    for a in args {
        if let Some(s) = crate::value_as_text(a) {
            match s.trim().to_ascii_lowercase().as_str() {
                "rows" => rows_mode = true,
                "stable" => stable = true,
                _ => {}
            }
        }
    }
    (rows_mode, stable)
}

fn cmp_f64_row(a: &[f64], b: &[f64]) -> std::cmp::Ordering {
    for (&x, &y) in a.iter().zip(b.iter()) {
        let ord = x.total_cmp(&y);
        if ord != std::cmp::Ordering::Equal {
            return ord;
        }
    }
    a.len().cmp(&b.len())
}

fn eq_f64_row(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(&x, &y)| x == y)
}

/// Set operations: `unique`, `ismember`, `setdiff`, `intersect`, `union`, `setxor`.
pub fn fn_unique(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 4 {
        return Err(MatroError::arity("unique", "1 to 4 (A [, 'rows'|'sorted'|'stable'])", args.len()));
    }
    let (rows_mode, stable) = parse_set_flags(&args[1..]);

    if let Some((items, r, c)) = extract_text_collection(&args[0]) {
        let n = items.len();
        let mut order: Vec<usize> = (0..n).collect();
        if !stable {
            order.sort_by(|&i, &j| items[i].cmp(&items[j]));
        }
        let mut uniq_str: Vec<String> = Vec::new();
        let mut ia: Vec<f64> = Vec::new();
        let mut ic = vec![0.0; n];
        for idx in order {
            let s = &items[idx];
            if let Some(pos) = uniq_str.iter().position(|u| u == s) {
                ic[idx] = (pos + 1) as f64;
            } else {
                let pos = uniq_str.len();
                uniq_str.push(s.clone());
                ia.push((idx + 1) as f64);
                ic[idx] = (pos + 1) as f64;
            }
        }
        let is_row = r == 1 && c > 1;
        let cells: Vec<Value> = uniq_str.into_iter().map(Value::String).collect();
        let cell = if is_row {
            CellArray::from_rows(vec![cells])?
        } else {
            CellArray::from_rows(cells.into_iter().map(|v| vec![v]).collect())?
        };
        let c_val = if matches!(&args[0], Value::Object(o) if o.class_name == "string") {
            let mut fields = BTreeMap::new();
            fields.insert("Data".to_string(), Value::Cell(cell));
            Value::Object(matro_types::ObjectInstance {
                class_name: "string".to_string(),
                fields,
            })
        } else {
            Value::Cell(cell)
        };
        let mut out = vec![c_val];
        if nargout >= 2 {
            out.push(Value::matrix(Matrix::col_vec(ia)));
        }
        if nargout >= 3 {
            out.push(Value::matrix(Matrix::col_vec(ic)));
        }
        return Ok(out);
    }

    let m = args[0].to_dense_matrix()?;
    if rows_mode {
        let nr = m.rows();
        let nc = m.cols();
        let rows_vec: Vec<Vec<f64>> = (0..nr)
            .map(|r| (0..nc).map(|c| m.get(r, c)).collect())
            .collect();
        let mut order: Vec<usize> = (0..nr).collect();
        if !stable {
            order.sort_by(|&i, &j| cmp_f64_row(&rows_vec[i], &rows_vec[j]));
        }
        let mut uniq_rows: Vec<Vec<f64>> = Vec::new();
        let mut ia: Vec<f64> = Vec::new();
        let mut ic = vec![0.0; nr];
        for idx in order {
            let row = &rows_vec[idx];
            if let Some(pos) = uniq_rows.iter().position(|u| eq_f64_row(u, row)) {
                ic[idx] = (pos + 1) as f64;
            } else {
                let pos = uniq_rows.len();
                uniq_rows.push(row.clone());
                ia.push((idx + 1) as f64);
                ic[idx] = (pos + 1) as f64;
            }
        }
        let mut c_mat = Matrix::zeros(uniq_rows.len(), nc);
        for (r, row) in uniq_rows.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                c_mat.set(r, c, v);
            }
        }
        let mut out = vec![Value::matrix(c_mat)];
        if nargout >= 2 {
            out.push(Value::matrix(Matrix::col_vec(ia)));
        }
        if nargout >= 3 {
            out.push(Value::matrix(Matrix::col_vec(ic)));
        }
        return Ok(out);
    }

    let data = m.data();
    let n = data.len();
    let mut order: Vec<usize> = (0..n).collect();
    if !stable {
        order.sort_by(|&i, &j| data[i].total_cmp(&data[j]));
    }
    let mut uniq_vals: Vec<f64> = Vec::new();
    let mut ia: Vec<f64> = Vec::new();
    let mut ic = vec![0.0; n];
    for idx in order {
        let v = data[idx];
        let found = if v.is_nan() {
            None
        } else {
            uniq_vals.iter().position(|&u| u == v)
        };
        if let Some(pos) = found {
            ic[idx] = (pos + 1) as f64;
        } else {
            let pos = uniq_vals.len();
            uniq_vals.push(v);
            ia.push((idx + 1) as f64);
            ic[idx] = (pos + 1) as f64;
        }
    }
    let c_mat = if m.rows() == 1 && m.cols() > 1 {
        Matrix::row_vec(uniq_vals)
    } else {
        Matrix::col_vec(uniq_vals)
    };
    let mut out = vec![Value::matrix(c_mat)];
    if nargout >= 2 {
        out.push(Value::matrix(Matrix::col_vec(ia)));
    }
    if nargout >= 3 {
        out.push(Value::matrix(Matrix::col_vec(ic)));
    }
    Ok(out)
}

pub fn fn_ismember(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 4 {
        return Err(MatroError::arity("ismember", "2 to 4 (A, B [, 'rows'])", args.len()));
    }
    let (rows_mode, _) = parse_set_flags(&args[2..]);

    if let (Some((sa, ra, ca)), Some((sb, _, _))) = (
        extract_text_collection(&args[0]),
        extract_text_collection(&args[1]),
    ) {
        let mut lia = Vec::with_capacity(sa.len());
        let mut locb = Vec::with_capacity(sa.len());
        for s in &sa {
            if let Some(pos) = sb.iter().position(|b| b == s) {
                lia.push(1.0);
                locb.push((pos + 1) as f64);
            } else {
                lia.push(0.0);
                locb.push(0.0);
            }
        }
        let lia_mat = Matrix::from_col_major(ra, ca, lia)?.with_logical(true);
        if nargout >= 2 {
            let locb_mat = Matrix::from_col_major(ra, ca, locb)?;
            return Ok(vec![Value::matrix(lia_mat), Value::matrix(locb_mat)]);
        }
        return ok1(Value::matrix(lia_mat));
    }

    let a = args[0].to_dense_matrix()?;
    let b = args[1].to_dense_matrix()?;
    if rows_mode {
        if a.cols() != b.cols() {
            return Err(MatroError::dimension("ismember(..., 'rows'): column count mismatch"));
        }
        let b_rows: Vec<Vec<f64>> = (0..b.rows())
            .map(|r| (0..b.cols()).map(|c| b.get(r, c)).collect())
            .collect();
        let mut lia = Vec::with_capacity(a.rows());
        let mut locb = Vec::with_capacity(a.rows());
        for r in 0..a.rows() {
            let a_row: Vec<f64> = (0..a.cols()).map(|c| a.get(r, c)).collect();
            if let Some(pos) = b_rows.iter().position(|br| eq_f64_row(br, &a_row)) {
                lia.push(1.0);
                locb.push((pos + 1) as f64);
            } else {
                lia.push(0.0);
                locb.push(0.0);
            }
        }
        let lia_mat = Matrix::col_vec(lia).with_logical(true);
        if nargout >= 2 {
            return Ok(vec![Value::matrix(lia_mat), Value::matrix(Matrix::col_vec(locb))]);
        }
        return ok1(Value::matrix(lia_mat));
    }

    let bdata = b.data();
    let mut lia = Vec::with_capacity(a.len());
    let mut locb = Vec::with_capacity(a.len());
    for &x in a.data() {
        if let Some(pos) = bdata.iter().position(|&y| x == y) {
            lia.push(1.0);
            locb.push((pos + 1) as f64);
        } else {
            lia.push(0.0);
            locb.push(0.0);
        }
    }
    let lia_mat = Matrix::from_nd_col_major(a.dims(), lia)?.with_logical(true);
    if nargout >= 2 {
        let locb_mat = Matrix::from_nd_col_major(a.dims(), locb)?;
        Ok(vec![Value::matrix(lia_mat), Value::matrix(locb_mat)])
    } else {
        ok1(Value::matrix(lia_mat))
    }
}

pub fn fn_set_binary(name: &str, args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 4 {
        return Err(MatroError::arity(name, "2 to 4 (A, B [, 'rows'|'sorted'|'stable'])", args.len()));
    }
    let (rows_mode, stable) = parse_set_flags(&args[2..]);

    if let (Some((sa, ra, ca)), Some((sb, rb, cb))) = (
        extract_text_collection(&args[0]),
        extract_text_collection(&args[1]),
    ) {
        let mut items: Vec<(String, Option<usize>, Option<usize>)> = Vec::new();
        match name {
            "setdiff" => {
                for (i, s) in sa.iter().enumerate() {
                    if !sb.contains(s) && !items.iter().any(|(x, _, _)| x == s) {
                        items.push((s.clone(), Some(i + 1), None));
                    }
                }
            }
            "intersect" => {
                for (i, s) in sa.iter().enumerate() {
                    if let Some(j) = sb.iter().position(|b| b == s) {
                        if !items.iter().any(|(x, _, _)| x == s) {
                            items.push((s.clone(), Some(i + 1), Some(j + 1)));
                        }
                    }
                }
            }
            "union" => {
                for (i, s) in sa.iter().enumerate() {
                    if !items.iter().any(|(x, _, _)| x == s) {
                        items.push((s.clone(), Some(i + 1), None));
                    }
                }
                for (j, s) in sb.iter().enumerate() {
                    if !items.iter().any(|(x, _, _)| x == s) {
                        items.push((s.clone(), None, Some(j + 1)));
                    }
                }
            }
            "setxor" => {
                for (i, s) in sa.iter().enumerate() {
                    if !sb.contains(s) && !items.iter().any(|(x, _, _)| x == s) {
                        items.push((s.clone(), Some(i + 1), None));
                    }
                }
                for (j, s) in sb.iter().enumerate() {
                    if !sa.contains(s) && !items.iter().any(|(x, _, _)| x == s) {
                        items.push((s.clone(), None, Some(j + 1)));
                    }
                }
            }
            _ => unreachable!(),
        }
        if !stable {
            items.sort_by(|x, y| x.0.cmp(&y.0));
        }
        let is_row = ra == 1 && ca > 1 && rb == 1 && cb > 1;
        let cells: Vec<Value> = items.iter().map(|(s, _, _)| Value::String(s.clone())).collect();
        let cell = if is_row {
            CellArray::from_rows(vec![cells])?
        } else {
            CellArray::from_rows(cells.into_iter().map(|v| vec![v]).collect())?
        };
        let mut out = vec![Value::Cell(cell)];
        if nargout >= 2 {
            let ia: Vec<f64> = items.iter().filter_map(|(_, i, _)| i.map(|k| k as f64)).collect();
            out.push(Value::matrix(Matrix::col_vec(ia)));
        }
        if nargout >= 3 {
            let ib: Vec<f64> = items.iter().filter_map(|(_, _, j)| j.map(|k| k as f64)).collect();
            out.push(Value::matrix(Matrix::col_vec(ib)));
        }
        return Ok(out);
    }

    let a = args[0].to_dense_matrix()?;
    let b = args[1].to_dense_matrix()?;
    if rows_mode {
        if a.cols() != b.cols() {
            return Err(MatroError::dimension(format!("{name}(..., 'rows'): column count mismatch")));
        }
        let nc = a.cols();
        let a_rows: Vec<Vec<f64>> = (0..a.rows()).map(|r| (0..nc).map(|c| a.get(r, c)).collect()).collect();
        let b_rows: Vec<Vec<f64>> = (0..b.rows()).map(|r| (0..nc).map(|c| b.get(r, c)).collect()).collect();
        let mut items: Vec<(Vec<f64>, Option<usize>, Option<usize>)> = Vec::new();
        match name {
            "setdiff" => {
                for (i, row) in a_rows.iter().enumerate() {
                    if !b_rows.iter().any(|br| eq_f64_row(br, row))
                        && !items.iter().any(|(x, _, _)| eq_f64_row(x, row))
                    {
                        items.push((row.clone(), Some(i + 1), None));
                    }
                }
            }
            "intersect" => {
                for (i, row) in a_rows.iter().enumerate() {
                    if let Some(j) = b_rows.iter().position(|br| eq_f64_row(br, row)) {
                        if !items.iter().any(|(x, _, _)| eq_f64_row(x, row)) {
                            items.push((row.clone(), Some(i + 1), Some(j + 1)));
                        }
                    }
                }
            }
            "union" => {
                for (i, row) in a_rows.iter().enumerate() {
                    if !items.iter().any(|(x, _, _)| eq_f64_row(x, row)) {
                        items.push((row.clone(), Some(i + 1), None));
                    }
                }
                for (j, row) in b_rows.iter().enumerate() {
                    if !items.iter().any(|(x, _, _)| eq_f64_row(x, row)) {
                        items.push((row.clone(), None, Some(j + 1)));
                    }
                }
            }
            "setxor" => {
                for (i, row) in a_rows.iter().enumerate() {
                    if !b_rows.iter().any(|br| eq_f64_row(br, row))
                        && !items.iter().any(|(x, _, _)| eq_f64_row(x, row))
                    {
                        items.push((row.clone(), Some(i + 1), None));
                    }
                }
                for (j, row) in b_rows.iter().enumerate() {
                    if !a_rows.iter().any(|ar| eq_f64_row(ar, row))
                        && !items.iter().any(|(x, _, _)| eq_f64_row(x, row))
                    {
                        items.push((row.clone(), None, Some(j + 1)));
                    }
                }
            }
            _ => unreachable!(),
        }
        if !stable {
            items.sort_by(|x, y| cmp_f64_row(&x.0, &y.0));
        }
        let mut c_mat = Matrix::zeros(items.len(), nc);
        for (r, (row, _, _)) in items.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                c_mat.set(r, c, v);
            }
        }
        let mut out = vec![Value::matrix(c_mat)];
        if nargout >= 2 {
            let ia: Vec<f64> = items.iter().filter_map(|(_, i, _)| i.map(|k| k as f64)).collect();
            out.push(Value::matrix(Matrix::col_vec(ia)));
        }
        if nargout >= 3 {
            let ib: Vec<f64> = items.iter().filter_map(|(_, _, j)| j.map(|k| k as f64)).collect();
            out.push(Value::matrix(Matrix::col_vec(ib)));
        }
        return Ok(out);
    }

    let ad = a.data();
    let bd = b.data();
    let mut items: Vec<(f64, Option<usize>, Option<usize>)> = Vec::new();
    match name {
        "setdiff" => {
            for (i, &x) in ad.iter().enumerate() {
                if !bd.iter().any(|&y| y == x) && !items.iter().any(|&(v, _, _)| v == x) {
                    items.push((x, Some(i + 1), None));
                }
            }
        }
        "intersect" => {
            for (i, &x) in ad.iter().enumerate() {
                if let Some(j) = bd.iter().position(|&y| y == x) {
                    if !items.iter().any(|&(v, _, _)| v == x) {
                        items.push((x, Some(i + 1), Some(j + 1)));
                    }
                }
            }
        }
        "union" => {
            for (i, &x) in ad.iter().enumerate() {
                if !items.iter().any(|&(v, _, _)| v == x) {
                    items.push((x, Some(i + 1), None));
                }
            }
            for (j, &y) in bd.iter().enumerate() {
                if !items.iter().any(|&(v, _, _)| v == y) {
                    items.push((y, None, Some(j + 1)));
                }
            }
        }
        "setxor" => {
            for (i, &x) in ad.iter().enumerate() {
                if !bd.iter().any(|&y| y == x) && !items.iter().any(|&(v, _, _)| v == x) {
                    items.push((x, Some(i + 1), None));
                }
            }
            for (j, &y) in bd.iter().enumerate() {
                if !ad.iter().any(|&x| x == y) && !items.iter().any(|&(v, _, _)| v == y) {
                    items.push((y, None, Some(j + 1)));
                }
            }
        }
        _ => unreachable!(),
    }
    if !stable {
        items.sort_by(|x, y| x.0.total_cmp(&y.0));
    }
    let vals: Vec<f64> = items.iter().map(|&(v, _, _)| v).collect();
    let is_row = match name {
        "setdiff" => a.rows() == 1 && a.cols() > 1,
        _ => a.rows() == 1 && b.rows() == 1 && (a.cols() > 1 || b.cols() > 1),
    };
    let c_mat = if is_row {
        Matrix::row_vec(vals)
    } else {
        Matrix::col_vec(vals)
    };
    let mut out = vec![Value::matrix(c_mat)];
    if nargout >= 2 {
        let ia: Vec<f64> = items.iter().filter_map(|(_, i, _)| i.map(|k| k as f64)).collect();
        out.push(Value::matrix(Matrix::col_vec(ia)));
    }
    if nargout >= 3 {
        let ib: Vec<f64> = items.iter().filter_map(|(_, _, j)| j.map(|k| k as f64)).collect();
        out.push(Value::matrix(Matrix::col_vec(ib)));
    }
    Ok(out)
}

/// Matrix reshaping & structural utilities: N-D `repmat`, `fliplr`, `flipud`, `rot90`, `tril`, `triu`.
pub fn fn_repmat(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 {
        return Err(MatroError::arity("repmat", ">= 2 (A, r1, r2, ... | A, [r1 r2 ...])", args.len()));
    }
    let reps: Vec<usize> = if args.len() == 2 {
        let rep = args[1].to_dense_matrix()?;
        if rep.len() == 1 {
            let k = as_usize(&args[1], "repmat")?;
            vec![k, k]
        } else {
            rep.data().iter().map(|&x| x.max(0.0).round() as usize).collect()
        }
    } else {
        args[1..]
            .iter()
            .map(|a| as_usize(a, "repmat"))
            .collect::<Result<_>>()?
    };

    if let Value::Cell(cell) = &args[0] {
        let rm = reps.first().copied().unwrap_or(1);
        let cm = reps.get(1).copied().unwrap_or(1);
        let out_r = cell.rows * rm;
        let out_c = cell.cols * cm;
        let mut data = Vec::with_capacity(out_r * out_c);
        for c in 0..out_c {
            for r in 0..out_r {
                data.push(cell.get(r % cell.rows, c % cell.cols).clone());
            }
        }
        return ok1(Value::Cell(CellArray {
            rows: out_r,
            cols: out_c,
            data,
        }));
    }
    if let Value::String(s) = &args[0] {
        let cm = reps.get(1).copied().unwrap_or(1);
        return ok1(Value::String(s.repeat(cm)));
    }

    let m = args[0].to_dense_matrix()?;
    let in_dims = m.dims();
    let nd = in_dims.len().max(reps.len());
    let mut out_dims = Vec::with_capacity(nd);
    let mut in_ext = Vec::with_capacity(nd);
    for ax in 0..nd {
        let d = in_dims.get(ax).copied().unwrap_or(1);
        let r = reps.get(ax).copied().unwrap_or(1);
        in_ext.push(d);
        out_dims.push(d * r);
    }
    let mut in_strides = vec![1usize; nd];
    for ax in 1..nd {
        in_strides[ax] = in_strides[ax - 1] * in_ext[ax - 1];
    }
    let total: usize = out_dims.iter().product();
    let mut re = Vec::with_capacity(total);
    let mut im = Vec::with_capacity(total);
    let mut coord = vec![0usize; nd];
    for _ in 0..total {
        let mut src = 0usize;
        for ax in 0..nd {
            src += (coord[ax] % in_ext[ax]) * in_strides[ax];
        }
        let v = m.get_linear_c(src);
        re.push(v.re);
        im.push(v.im);
        for ax in 0..nd {
            coord[ax] += 1;
            if coord[ax] < out_dims[ax] {
                break;
            }
            coord[ax] = 0;
        }
    }
    let out = Matrix::from_nd_complex_col_major(out_dims, re, im)?.normalize();
    let out = if m.is_logical() { out.with_logical(true) } else { out };
    ok1(Value::matrix(out))
}

pub fn flip_axis_nd(m: &Matrix, ax: usize) -> Result<Matrix> {
    let dims = m.dims();
    if ax >= dims.len() || dims[ax] <= 1 || m.is_empty() {
        return Ok(m.clone());
    }
    let inner: usize = dims[..ax].iter().product();
    let axis_len = dims[ax];
    let outer: usize = dims[ax + 1..].iter().product();
    let total = m.len();
    let mut re = vec![0.0; total];
    let mut im = vec![0.0; total];
    for o in 0..outer {
        let block = o * axis_len * inner;
        for k in 0..axis_len {
            let src_base = block + (axis_len - 1 - k) * inner;
            let dst_base = block + k * inner;
            for i in 0..inner {
                let v = m.get_linear_c(src_base + i);
                re[dst_base + i] = v.re;
                im[dst_base + i] = v.im;
            }
        }
    }
    let out = Matrix::from_nd_complex_col_major(dims, re, im)?.normalize();
    Ok(if m.is_logical() { out.with_logical(true) } else { out })
}

pub fn fn_fliplr(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("fliplr", "1", args.len()));
    }
    if let Value::String(s) = &args[0] {
        return ok1(Value::String(s.chars().rev().collect()));
    }
    let m = args[0].to_dense_matrix()?;
    ok1(Value::matrix(flip_axis_nd(&m, 1)?))
}

pub fn fn_flipud(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("flipud", "1", args.len()));
    }
    if let Value::String(s) = &args[0] {
        return ok1(Value::String(s.clone()));
    }
    let m = args[0].to_dense_matrix()?;
    ok1(Value::matrix(flip_axis_nd(&m, 0)?))
}

pub fn fn_rot90(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("rot90", "1 or 2", args.len()));
    }
    let mut m = args[0].to_dense_matrix()?;
    let orig_logical = m.is_logical();
    let k = if args.len() == 2 {
        args[1].as_scalar()?.round() as i64
    } else {
        1
    };
    let steps = k.rem_euclid(4) as usize;
    for _ in 0..steps {
        let dims = m.dims();
        let (rows, cols) = (dims[0], dims[1]);
        let pages: usize = dims[2..].iter().product();
        let mut new_dims = dims.clone();
        new_dims[0] = cols;
        new_dims[1] = rows;
        let mut re = vec![0.0; m.len()];
        let mut im = vec![0.0; m.len()];
        let page_sz = rows * cols;
        for p in 0..pages {
            let p_off = p * page_sz;
            for r in 0..rows {
                for c in 0..cols {
                    let v = m.get_linear_c(p_off + c * rows + r);
                    let nr = cols - 1 - c;
                    let nc = r;
                    let dst = p_off + nc * cols + nr;
                    re[dst] = v.re;
                    im[dst] = v.im;
                }
            }
        }
        m = Matrix::from_nd_complex_col_major(new_dims, re, im)?.normalize();
    }
    if orig_logical {
        m = m.with_logical(true);
    }
    ok1(Value::matrix(m))
}

pub fn fn_tri(name: &str, args: &[Value], upper: bool) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity(name, "1 or 2", args.len()));
    }
    let k = if args.len() == 2 {
        args[1].as_scalar()?.round() as isize
    } else {
        0
    };
    if let Value::Sparse(sp) = &args[0] {
        let trips: Vec<(usize, usize, f64)> = sp
            .triplets()
            .into_iter()
            .filter(|&(r, c, _)| {
                if upper {
                    (c as isize) - (r as isize) >= k
                } else {
                    (c as isize) - (r as isize) <= k
                }
            })
            .collect();
        return ok1(Value::sparse(matro_types::SparseMatrix::from_triplets(
            sp.rows(),
            sp.cols(),
            &trips,
        )?));
    }
    let m = args[0].to_dense_matrix()?;
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

fn overlapping_char_find(s: &str, pat: &str) -> Vec<f64> {
    if pat.is_empty() {
        return Vec::new();
    }
    let sc: Vec<char> = s.chars().collect();
    let pc: Vec<char> = pat.chars().collect();
    if pc.len() > sc.len() {
        return Vec::new();
    }
    let mut idxs = Vec::new();
    for i in 0..=(sc.len() - pc.len()) {
        if sc[i..i + pc.len()] == pc[..] {
            idxs.push((i + 1) as f64);
        }
    }
    idxs
}

/// String search / replace / prefix / suffix helpers.
pub fn fn_strfind(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 4 {
        return Err(MatroError::arity("strfind", "2 to 4 (str, pat [, 'ForceCellOutput', tf])", args.len()));
    }
    let pat = crate::value_as_text(&args[1])
        .ok_or_else(|| MatroError::runtime("strfind: pattern must be a string"))?;
    let mut force_cell = false;
    let mut i = 2;
    while i + 1 < args.len() {
        if crate::value_as_text(&args[i]).is_some_and(|s| s.eq_ignore_ascii_case("forcecelloutput")) {
            force_cell = args[i + 1].is_truthy();
        }
        i += 2;
    }
    if let Value::String(s) = &args[0] {
        let m = Matrix::row_vec(overlapping_char_find(s, &pat));
        return if force_cell {
            ok1(Value::Cell(CellArray::from_rows(vec![vec![Value::matrix(m)]])?))
        } else {
            ok1(Value::matrix(m))
        };
    }
    if let Some((items, r, c)) = extract_text_collection(&args[0]) {
        let mut data = Vec::with_capacity(items.len());
        for s in items {
            data.push(Value::matrix(Matrix::row_vec(overlapping_char_find(&s, &pat))));
        }
        return ok1(Value::Cell(CellArray { rows: r, cols: c, data }));
    }
    Err(MatroError::runtime("strfind: expected char vector, string, or cellstr"))
}

pub fn fn_strrep(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 3 {
        return Err(MatroError::arity("strrep", "3", args.len()));
    }
    let old = crate::value_as_text(&args[1])
        .ok_or_else(|| MatroError::runtime("strrep: old substring must be text"))?;
    let new = crate::value_as_text(&args[2])
        .ok_or_else(|| MatroError::runtime("strrep: new substring must be text"))?;
    if let Value::String(s) = &args[0] {
        return ok1(Value::String(s.replace(&old, &new)));
    }
    if let Some((items, r, c)) = extract_text_collection(&args[0]) {
        let data: Vec<Value> = items
            .into_iter()
            .map(|s| Value::String(s.replace(&old, &new)))
            .collect();
        let cell = CellArray { rows: r, cols: c, data };
        if matches!(&args[0], Value::Object(o) if o.class_name == "string") {
            let mut fields = BTreeMap::new();
            fields.insert("Data".to_string(), Value::Cell(cell));
            return ok1(Value::Object(matro_types::ObjectInstance {
                class_name: "string".to_string(),
                fields,
            }));
        }
        return ok1(Value::Cell(cell));
    }
    Err(MatroError::runtime("strrep: expected text input"))
}

pub fn fn_str_pred(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 4 {
        return Err(MatroError::arity(name, "2 to 4 (str, pat [, 'IgnoreCase', tf])", args.len()));
    }
    let mut ignore_case = false;
    let mut i = 2;
    while i + 1 < args.len() {
        if crate::value_as_text(&args[i]).is_some_and(|s| s.eq_ignore_ascii_case("ignorecase")) {
            ignore_case = args[i + 1].is_truthy();
        }
        i += 2;
    }
    let (pats, _, _) = extract_text_collection(&args[1])
        .ok_or_else(|| MatroError::runtime(format!("{name}: pattern must be text or cellstr")))?;
    let check = |s: &str| -> bool {
        if ignore_case {
            let sl = s.to_lowercase();
            pats.iter().any(|p| {
                let pl = p.to_lowercase();
                match name {
                    "startsWith" => sl.starts_with(&pl),
                    "endsWith" => sl.ends_with(&pl),
                    "contains" => sl.contains(&pl),
                    _ => false,
                }
            })
        } else {
            pats.iter().any(|p| match name {
                "startsWith" => s.starts_with(p.as_str()),
                "endsWith" => s.ends_with(p.as_str()),
                "contains" => s.contains(p.as_str()),
                _ => false,
            })
        }
    };
    if let Value::String(s) = &args[0] {
        return ok1(Value::bool(check(s)));
    }
    if let Some((items, r, c)) = extract_text_collection(&args[0]) {
        let vals: Vec<f64> = items
            .iter()
            .map(|s| if check(s) { 1.0 } else { 0.0 })
            .collect();
        return ok1(Value::matrix(
            Matrix::from_col_major(r, c, vals)?.with_logical(true),
        ));
    }
    Err(MatroError::runtime(format!("{name}: expected text input")))
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

/// `filter(b, a, x [, zi [, dim]])` — Direct Form II Transposed digital filter (N-D, complex, `[y, zf]`).
pub fn fn_filter(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() < 3 || args.len() > 5 {
        return Err(MatroError::arity("filter", "3 to 5 (b, a, x [, zi [, dim]])", args.len()));
    }
    let bm = args[0].to_dense_matrix()?;
    let am = args[1].to_dense_matrix()?;
    let xm = args[2].to_dense_matrix()?;
    if am.is_empty() || am.get_linear_c(0).is_zero() {
        return Err(MatroError::runtime("filter: a(1) must be non-zero"));
    }
    let a0 = am.get_linear_c(0);
    let n_ord = bm.len().max(am.len());
    let mut b = vec![Complex::ZERO; n_ord];
    let mut a = vec![Complex::ZERO; n_ord];
    for i in 0..bm.len() {
        b[i] = bm.get_linear_c(i) / a0;
    }
    for i in 0..am.len() {
        a[i] = am.get_linear_c(i) / a0;
    }
    let zi_opt: Option<Matrix> = match args.get(3) {
        Some(v) => {
            let zm = v.to_dense_matrix()?;
            if zm.is_empty() { None } else { Some(zm) }
        }
        None => None,
    };
    let dims = xm.dims();
    let dim = if let Some(d_val) = args.get(4) {
        as_usize(d_val, "filter")?
    } else {
        dims.iter().position(|&d| d != 1).map_or(1, |i| i + 1)
    };
    let ax = dim.saturating_sub(1);
    let state_len = n_ord.saturating_sub(1);
    if xm.is_empty() || ax >= dims.len() {
        let zf = Matrix::zeros(state_len, 1);
        return if nargout >= 2 {
            Ok(vec![Value::matrix(xm), Value::matrix(zf)])
        } else {
            ok1(Value::matrix(xm))
        };
    }

    let inner: usize = dims[..ax].iter().product();
    let axis_len = dims[ax];
    let outer: usize = dims[ax + 1..].iter().product();
    let mut y_re = vec![0.0; xm.len()];
    let mut y_im = vec![0.0; xm.len()];
    let mut zf_dims = dims.clone();
    zf_dims[ax] = state_len;
    let zf_total = inner * state_len * outer;
    let mut zf_re = vec![0.0; zf_total];
    let mut zf_im = vec![0.0; zf_total];

    let n_channels = inner * outer;
    let mut ch = 0usize;
    for o in 0..outer {
        let block = o * axis_len * inner;
        let z_block = o * state_len * inner;
        for ii in 0..inner {
            let mut z = vec![Complex::ZERO; n_ord];
            if let Some(ref zi) = zi_opt {
                for k in 0..state_len {
                    let zi_idx = if zi.len() == state_len {
                        k
                    } else if zi.len() == state_len * n_channels {
                        z_block + k * inner + ii
                    } else {
                        (ch * state_len + k).min(zi.len() - 1)
                    };
                    z[k] = zi.get_linear_c(zi_idx);
                }
            }
            for k in 0..axis_len {
                let pos = block + k * inner + ii;
                let xn = xm.get_linear_c(pos);
                let yn = b[0] * xn + z[0];
                y_re[pos] = yn.re;
                y_im[pos] = yn.im;
                for m in 1..n_ord {
                    z[m - 1] = b[m] * xn + z[m] - a[m] * yn;
                }
            }
            for k in 0..state_len {
                let zpos = z_block + k * inner + ii;
                zf_re[zpos] = z[k].re;
                zf_im[zpos] = z[k].im;
            }
            ch += 1;
        }
    }
    let y_mat = Matrix::from_nd_complex_col_major(dims, y_re, y_im)?.normalize();
    if nargout >= 2 {
        let zf_mat = Matrix::from_nd_complex_col_major(zf_dims, zf_re, zf_im)?.normalize();
        Ok(vec![Value::matrix(y_mat), Value::matrix(zf_mat)])
    } else {
        ok1(Value::matrix(y_mat))
    }
}

/// `interp2`: supports 3..=7 args (`interp2(V, Xq, Yq [, method [, extrapval]])` and
/// `interp2(X, Y, V, Xq, Yq [, method [, extrapval]])`), `'nearest'`/`'linear'`/`'cubic'`/`'spline'`/`'makima'`,
/// implicit grid vectors (`xq_row`, `yq_col`), and descending grids.
pub fn fn_interp2(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 3 || args.len() > 7 {
        return Err(MatroError::arity("interp2", "3 to 7", args.len()));
    }
    let mut end = args.len();
    let mut method = "linear".to_string();
    let mut extrapval = f64::NAN;
    if end >= 2 && crate::value_as_text(&args[end - 2]).is_some() {
        method = crate::value_as_text(&args[end - 2]).unwrap().to_ascii_lowercase();
        extrapval = args[end - 1].as_scalar()?;
        end -= 2;
    } else if end >= 1 && crate::value_as_text(&args[end - 1]).is_some() {
        method = crate::value_as_text(&args[end - 1]).unwrap().to_ascii_lowercase();
        end -= 1;
    }

    let (xs, ys, zm, xqm_raw, yqm_raw) = if end == 3 {
        let zm = args[0].to_dense_matrix()?;
        let xs: Vec<f64> = (1..=zm.cols()).map(|i| i as f64).collect();
        let ys: Vec<f64> = (1..=zm.rows()).map(|i| i as f64).collect();
        (xs, ys, zm, args[1].to_dense_matrix()?, args[2].to_dense_matrix()?)
    } else if end == 5 {
        let xm = args[0].to_dense_matrix()?;
        let ym = args[1].to_dense_matrix()?;
        let zm = args[2].to_dense_matrix()?;
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
        (xs, ys, zm, args[3].to_dense_matrix()?, args[4].to_dense_matrix()?)
    } else {
        return Err(MatroError::arity("interp2", "3 to 7", args.len()));
    };

    if xs.len() != zm.cols() || ys.len() != zm.rows() || xs.len() < 2 || ys.len() < 2 {
        return Err(MatroError::dimension("interp2: grid dimensions do not match Z"));
    }

    // Handle implicit grid vector queries (xq row vec + yq col vec)
    let (xqm, yqm) = if xqm_raw.dims() == yqm_raw.dims() {
        (xqm_raw, yqm_raw)
    } else if (xqm_raw.rows() == 1 || xqm_raw.cols() == 1)
        && (yqm_raw.rows() == 1 || yqm_raw.cols() == 1)
    {
        let nr = yqm_raw.len();
        let nc = xqm_raw.len();
        let mut xq = Matrix::zeros(nr, nc);
        let mut yq = Matrix::zeros(nr, nc);
        for c in 0..nc {
            for r in 0..nr {
                xq.set(r, c, xqm_raw.data()[c]);
                yq.set(r, c, yqm_raw.data()[r]);
            }
        }
        (xq, yq)
    } else {
        return Err(MatroError::dimension("interp2: Xq and Yq must have matching dimensions"));
    };

    let find_bracket = |arr: &[f64], val: f64| -> Option<(usize, usize, f64)> {
        if !val.is_finite() {
            return None;
        }
        let asc = arr[arr.len() - 1] >= arr[0];
        let (min_v, max_v) = if asc { (arr[0], arr[arr.len() - 1]) } else { (arr[arr.len() - 1], arr[0]) };
        if val < min_v || val > max_v {
            return None;
        }
        let mut lo = 0usize;
        let mut hi = arr.len() - 1;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if (asc && arr[mid] <= val) || (!asc && arr[mid] >= val) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let t = (val - arr[lo]) / (arr[hi] - arr[lo]);
        Some((lo, hi, t))
    };

    let mut out = Matrix::zeros(xqm.rows(), xqm.cols());
    if method == "cubic" || method == "spline" || method == "makima" {
        // Separable 1D cubic/spline/makima along X then Y
        let xm_vec = Matrix::row_vec(xs.clone());
        let ym_vec = Matrix::col_vec(ys.clone());
        for c in 0..xqm.cols() {
            for r in 0..xqm.rows() {
                let qx = xqm.get(r, c);
                let qy = yqm.get(r, c);
                if find_bracket(&xs, qx).is_none() || find_bracket(&ys, qy).is_none() {
                    out.set(r, c, extrapval);
                    continue;
                }
                let qx_v = Value::scalar(qx);
                let mut col_vals = Vec::with_capacity(ys.len());
                for rr in 0..ys.len() {
                    let row_v = Value::matrix(Matrix::row_vec((0..xs.len()).map(|cc| zm.get(rr, cc)).collect()));
                    let iv = crate::interp::fn_interp1(&[
                        Value::matrix(xm_vec.clone()),
                        row_v,
                        qx_v.clone(),
                        Value::String(method.clone()),
                    ])?;
                    col_vals.push(iv[0].as_scalar()?);
                }
                let iv = crate::interp::fn_interp1(&[
                    Value::matrix(ym_vec.clone()),
                    Value::matrix(Matrix::col_vec(col_vals)),
                    Value::scalar(qy),
                    Value::String(method.clone()),
                ])?;
                out.set(r, c, iv[0].as_scalar()?);
            }
        }
        return ok1(Value::matrix(out));
    }

    for c in 0..xqm.cols() {
        for r in 0..xqm.rows() {
            let qx = xqm.get(r, c);
            let qy = yqm.get(r, c);
            let val = match (find_bracket(&xs, qx), find_bracket(&ys, qy)) {
                (Some((c0, c1, tx)), Some((r0, r1, ty))) => {
                    if method == "nearest" {
                        let rr = if ty < 0.5 { r0 } else { r1 };
                        let cc = if tx < 0.5 { c0 } else { c1 };
                        zm.get(rr, cc)
                    } else {
                        let z00 = zm.get(r0, c0);
                        let z10 = zm.get(r0, c1);
                        let z01 = zm.get(r1, c0);
                        let z11 = zm.get(r1, c1);
                        let top = z00 + tx * (z10 - z00);
                        let bot = z01 + tx * (z11 - z01);
                        top + ty * (bot - top)
                    }
                }
                _ => extrapval,
            };
            out.set(r, c, val);
        }
    }
    ok1(Value::matrix(out))
}

/// 2D convolution: `conv2(A, B [, shape])` and separable `conv2(u, v, A [, shape])`.
pub fn fn_conv2(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 4 {
        return Err(MatroError::arity("conv2", "2 to 4 (A, B [, shape] | u, v, A [, shape])", args.len()));
    }
    let is_sep = args.len() == 4
        || (args.len() == 3 && crate::value_as_text(&args[2]).is_none());
    let (a, b, shape) = if is_sep {
        let u = args[0].to_dense_matrix()?;
        let v = args[1].to_dense_matrix()?;
        let a = args[2].to_dense_matrix()?;
        let shape = args
            .get(3)
            .and_then(crate::value_as_text)
            .unwrap_or_else(|| "full".to_string())
            .to_ascii_lowercase();
        // Outer product kernel B = u(:) * v(:).'
        let mut k = Matrix::zeros(u.len(), v.len());
        for j in 0..v.len() {
            for i in 0..u.len() {
                k.set_c(i, j, u.get_linear_c(i) * v.get_linear_c(j));
            }
        }
        (a, k.normalize(), shape)
    } else {
        let a = args[0].to_dense_matrix()?;
        let b = args[1].to_dense_matrix()?;
        let shape = args
            .get(2)
            .and_then(crate::value_as_text)
            .unwrap_or_else(|| "full".to_string())
            .to_ascii_lowercase();
        (a, b, shape)
    };
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
    match shape.as_str() {
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

/// Cumulative product: N-D `cumprod(A [, dim [, direction [, nanflag]]])`.
pub fn fn_cumprod(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 4 {
        return Err(MatroError::arity("cumprod", "1 to 4 (A [, dim [, direction [, nanflag]]])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let mut reverse = false;
    let mut omitnan = false;
    let mut dim_opt: Option<usize> = None;
    for a in &args[1..] {
        if let Some(s) = crate::value_as_text(a) {
            if s.eq_ignore_ascii_case("reverse") {
                reverse = true;
            } else if s.eq_ignore_ascii_case("forward") {
                reverse = false;
            } else if s.eq_ignore_ascii_case("omitnan") || s.eq_ignore_ascii_case("omitmissing") {
                omitnan = true;
            } else if s.eq_ignore_ascii_case("includenan") || s.eq_ignore_ascii_case("includemissing") {
                omitnan = false;
            }
        } else {
            let d = as_usize(a, "cumprod")?;
            if d == 0 {
                return Err(MatroError::runtime("cumprod: dimension must be >= 1"));
            }
            dim_opt = Some(d);
        }
    }
    let dims = m.dims();
    let total = m.len();
    if total == 0 {
        return ok1(Value::matrix(m));
    }
    let dim = dim_opt.unwrap_or_else(|| dims.iter().position(|&d| d != 1).map_or(1, |i| i + 1));
    let ax = dim - 1;
    let inner: usize = if ax < dims.len() { dims[..ax].iter().product() } else { total };
    let axis_len = if ax < dims.len() { dims[ax] } else { 1 };
    let outer: usize = if ax < dims.len() { dims[ax + 1..].iter().product() } else { 1 };
    let is_complex = m.is_complex();
    let mut out_re = vec![0.0; total];
    let mut out_im = if is_complex { vec![0.0; total] } else { Vec::new() };
    for o in 0..outer {
        let block = o * axis_len * inner;
        for i in 0..inner {
            let mut acc = Complex::ONE;
            if reverse {
                for k in (0..axis_len).rev() {
                    let pos = block + k * inner + i;
                    let v = m.get_linear_c(pos);
                    if !(omitnan && (v.re.is_nan() || v.im.is_nan())) {
                        acc = acc * v;
                    }
                    out_re[pos] = acc.re;
                    if is_complex {
                        out_im[pos] = acc.im;
                    }
                }
            } else {
                for k in 0..axis_len {
                    let pos = block + k * inner + i;
                    let v = m.get_linear_c(pos);
                    if !(omitnan && (v.re.is_nan() || v.im.is_nan())) {
                        acc = acc * v;
                    }
                    out_re[pos] = acc.re;
                    if is_complex {
                        out_im[pos] = acc.im;
                    }
                }
            }
        }
    }
    let res = if is_complex {
        Matrix::from_nd_complex_col_major(dims, out_re, out_im)?.normalize()
    } else {
        Matrix::from_nd_col_major(dims, out_re)?
    };
    ok1(Value::matrix(res))
}
