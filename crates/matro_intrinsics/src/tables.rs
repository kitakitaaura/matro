//! MATLAB `table`, `containers.Map`/`dictionary`, `sortrows`, and operator functions
//! (`plus`, `minus`, `times`, `rdivide`, `power`, etc. for `bsxfun` / `@plus`).

use matro_loader::ModuleLoader;
use matro_types::{CellArray, Complex, MapData, MatroError, Matrix, Result, TableData, Value};

fn ok1(v: Value) -> Result<Vec<Value>> {
    Ok(vec![v])
}

/// `table(col1, col2, ..., 'VariableNames', {'A', 'B'})`
pub fn fn_table(args: &[Value]) -> Result<Vec<Value>> {
    let mut cols = Vec::new();
    let mut custom_names: Option<Vec<String>> = None;
    let mut i = 0;
    while i < args.len() {
        if let Ok(s) = args[i].as_str() {
            if s.eq_ignore_ascii_case("VariableNames") && i + 1 < args.len() {
                let Value::Cell(c) = &args[i + 1] else {
                    return Err(MatroError::runtime(
                        "table: 'VariableNames' must be a cell array of strings",
                    ));
                };
                let mut names = Vec::with_capacity(c.data.len());
                for item in &c.data {
                    names.push(item.as_str()?.to_string());
                }
                custom_names = Some(names);
                i += 2;
                continue;
            }
        }
        cols.push(args[i].clone());
        i += 1;
    }
    let names = match custom_names {
        Some(n) => n,
        None => (1..=cols.len()).map(|k| format!("Var{k}")).collect(),
    };
    ok1(Value::Table(TableData::new(names, cols)?))
}

pub fn fn_height(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("height", "1", args.len()));
    }
    let h = match &args[0] {
        Value::Table(t) => t.height(),
        Value::Matrix(m) => m.rows(),
        Value::Cell(c) => c.rows,
        other => {
            return Err(MatroError::runtime(format!(
                "height: unsupported type {}",
                other.type_name()
            )));
        }
    };
    ok1(Value::scalar(h as f64))
}

pub fn fn_width(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("width", "1", args.len()));
    }
    let w = match &args[0] {
        Value::Table(t) => t.width(),
        Value::Matrix(m) => m.cols(),
        Value::Cell(c) => c.cols,
        other => {
            return Err(MatroError::runtime(format!(
                "width: unsupported type {}",
                other.type_name()
            )));
        }
    };
    ok1(Value::scalar(w as f64))
}

pub fn fn_istable(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("istable", "1", args.len()));
    }
    ok1(Value::bool(matches!(&args[0], Value::Table(_))))
}

pub fn fn_summary(args: &[Value], stdout: &mut String) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("summary", "1", args.len()));
    }
    let Value::Table(t) = &args[0] else {
        return Err(MatroError::runtime("summary: expected a table"));
    };
    stdout.push_str(&format!(
        "Variables ({} rows x {} cols):\n",
        t.nrows,
        t.col_names.len()
    ));
    for (name, col) in t.col_names.iter().zip(t.columns.iter()) {
        match col {
            Value::Matrix(m) if !m.is_empty() => {
                let mut mn = f64::INFINITY;
                let mut mx = f64::NEG_INFINITY;
                let mut sum = 0.0;
                for &x in m.data() {
                    if x < mn {
                        mn = x;
                    }
                    if x > mx {
                        mx = x;
                    }
                    sum += x;
                }
                let mean = sum / m.len() as f64;
                stdout.push_str(&format!(
                    "  {name}: {}x1 double  [Min={mn:.4}, Mean={mean:.4}, Max={mx:.4}]\n",
                    m.rows()
                ));
            }
            Value::Cell(c) => {
                stdout.push_str(&format!("  {name}: {}x1 cell\n", c.data.len()));
            }
            other => {
                stdout.push_str(&format!("  {name}: {}\n", other.type_name()));
            }
        }
    }
    Ok(vec![])
}

/// `sortrows(A [, col [, 'ascend'|'descend']])` — works for both `table` and numeric `Matrix`.
pub fn fn_sortrows(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity("sortrows", "1 to 3", args.len()));
    }
    let desc = args
        .get(2)
        .and_then(|v| v.as_str().ok())
        .is_some_and(|s| s.eq_ignore_ascii_case("descend"));

    match &args[0] {
        Value::Table(t) => {
            let col_idx = match args.get(1) {
                None => 0usize,
                Some(Value::String(name)) => t.col_index(name).ok_or_else(|| {
                    MatroError::runtime(format!("sortrows: unknown table variable '{name}'"))
                })?,
                Some(v) => {
                    let s = v.as_scalar()?.round() as isize;
                    if s == 0 {
                        return Err(MatroError::index("sortrows: column index must be non-zero"));
                    }
                    let idx = (s.unsigned_abs()) - 1;
                    if idx >= t.width() {
                        return Err(MatroError::index("sortrows: column index out of bounds"));
                    }
                    idx
                }
            };
            let neg_col = args
                .get(1)
                .and_then(|v| v.as_scalar().ok())
                .is_some_and(|s| s < 0.0);
            let descending = desc ^ neg_col;

            let mut order: Vec<usize> = (0..t.nrows).collect();
            match &t.columns[col_idx] {
                Value::Matrix(m) => {
                    order.sort_by(|&a, &b| {
                        let cmp = m.get(a, 0).total_cmp(&m.get(b, 0));
                        if descending { cmp.reverse() } else { cmp }
                    });
                }
                Value::Cell(c) => {
                    order.sort_by(|&a, &b| {
                        let sa = format!("{}", c.data[a]);
                        let sb = format!("{}", c.data[b]);
                        let cmp = sa.cmp(&sb);
                        if descending { cmp.reverse() } else { cmp }
                    });
                }
                _ => {}
            }
            let all_cols: Vec<usize> = (0..t.width()).collect();
            let sorted = t.slice(&order, &all_cols)?;
            if nargout >= 2 {
                let idx_vec: Vec<f64> = order.iter().map(|&i| (i + 1) as f64).collect();
                Ok(vec![
                    Value::Table(sorted),
                    Value::matrix(Matrix::col_vec(idx_vec)),
                ])
            } else {
                ok1(Value::Table(sorted))
            }
        }
        Value::Matrix(m) => {
            let mut key_cols: Vec<(usize, bool)> = Vec::new();
            if let Some(col_arg) = args.get(1) {
                for &v in col_arg.as_matrix()?.data() {
                    let i = v.round() as isize;
                    let idx = i.unsigned_abs().saturating_sub(1);
                    if idx >= m.cols() {
                        return Err(MatroError::index("sortrows: column index out of bounds"));
                    }
                    key_cols.push((idx, (i < 0) ^ desc));
                }
            } else {
                for c in 0..m.cols() {
                    key_cols.push((c, desc));
                }
            }
            let mut order: Vec<usize> = (0..m.rows()).collect();
            order.sort_by(|&a, &b| {
                for &(c, rev) in &key_cols {
                    let cmp = m.get(a, c).total_cmp(&m.get(b, c));
                    if cmp != std::cmp::Ordering::Equal {
                        return if rev { cmp.reverse() } else { cmp };
                    }
                }
                std::cmp::Ordering::Equal
            });
            let mut out = Matrix::zeros(m.rows(), m.cols());
            for (nr, &or) in order.iter().enumerate() {
                for c in 0..m.cols() {
                    out.set_c(nr, c, m.get_c(or, c));
                }
            }
            if nargout >= 2 {
                let idx_vec: Vec<f64> = order.iter().map(|&i| (i + 1) as f64).collect();
                Ok(vec![
                    Value::matrix(out.normalize()),
                    Value::matrix(Matrix::col_vec(idx_vec)),
                ])
            } else {
                ok1(Value::matrix(out.normalize()))
            }
        }
        other => Err(MatroError::runtime(format!(
            "sortrows: expected table or matrix, got {}",
            other.type_name()
        ))),
    }
}

/// `T = readtable(filename)` — reads a header-row CSV from VFS into a `table`.
pub fn fn_readtable(args: &[Value], loader: &ModuleLoader) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("readtable", "1", args.len()));
    }
    let path = args[0].as_str()?;
    let resolved = loader
        .resolve_path(path)
        .unwrap_or_else(|| path.to_string());
    let text = loader
        .vfs()
        .read_to_string(&resolved)
        .ok_or_else(|| MatroError::runtime(format!("readtable: file not found '{path}'")))?;

    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let Some(header) = lines.next() else {
        return ok1(Value::Table(TableData::new(vec![], vec![])?));
    };
    let col_names: Vec<String> = header.split(',').map(|s| s.trim().to_string()).collect();
    let ncols = col_names.len();
    let mut raw_cols: Vec<Vec<String>> = vec![Vec::new(); ncols];

    for line in lines {
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        for c in 0..ncols {
            raw_cols[c].push(fields.get(c).copied().unwrap_or("").to_string());
        }
    }

    let mut columns = Vec::with_capacity(ncols);
    for col in raw_cols {
        if !col.is_empty() && col.iter().all(|s| s.parse::<f64>().is_ok()) {
            let nums: Vec<f64> = col.iter().map(|s| s.parse::<f64>().unwrap()).collect();
            columns.push(Value::matrix(Matrix::col_vec(nums)));
        } else {
            let vals: Vec<Value> = col.into_iter().map(Value::String).collect();
            let n = vals.len();
            columns.push(Value::Cell(CellArray {
                rows: n,
                cols: if n == 0 { 0 } else { 1 },
                data: vals,
            }));
        }
    }
    ok1(Value::Table(TableData::new(col_names, columns)?))
}

/// `writetable(T, filename)` — writes a `table` to CSV in the VFS.
pub fn fn_writetable(args: &[Value], loader: &mut ModuleLoader) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("writetable", "2 (T, filename)", args.len()));
    }
    let Value::Table(t) = &args[0] else {
        return Err(MatroError::runtime("writetable: first argument must be a table"));
    };
    let path = args[1].as_str()?;
    let mut out = t.col_names.join(",");
    out.push('\n');
    for r in 0..t.nrows {
        for c in 0..t.width() {
            if c > 0 {
                out.push(',');
            }
            out.push_str(&t.row_elem_str(r, c));
        }
        out.push('\n');
    }
    loader.mount_file(path, &out)?;
    Ok(vec![])
}

/// Construct a `containers.Map` / `dictionary` from optional `(keys, values)`.
pub fn build_map(args: &[Value]) -> Result<Value> {
    let mut map = MapData::default();
    if args.is_empty() {
        return Ok(Value::Map(map));
    }
    if args.len() != 2 {
        return Err(MatroError::arity("containers.Map", "0 or 2", args.len()));
    }
    let key_vals: Vec<Value> = match &args[0] {
        Value::Cell(c) => c.data.clone(),
        Value::Matrix(m) => m.data().iter().map(|&x| Value::scalar(x)).collect(),
        Value::String(s) => vec![Value::String(s.clone())],
        other => {
            return Err(MatroError::runtime(format!(
                "containers.Map: invalid keys type {}",
                other.type_name()
            )));
        }
    };
    let val_list: Vec<Value> = match &args[1] {
        Value::Cell(c) => c.data.clone(),
        Value::Matrix(m) => m.data().iter().map(|&x| Value::scalar(x)).collect(),
        Value::String(s) => vec![Value::String(s.clone())],
        other => vec![other.clone()],
    };
    if key_vals.len() != val_list.len() {
        return Err(MatroError::dimension(
            "containers.Map: keys and values must have the same length",
        ));
    }
    for (k, v) in key_vals.into_iter().zip(val_list.into_iter()) {
        map.entries.insert(MapData::key_of(&k)?, v);
    }
    Ok(Value::Map(map))
}

pub fn fn_dictionary(args: &[Value]) -> Result<Vec<Value>> {
    ok1(build_map(args)?)
}

pub fn fn_is_key(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("isKey", "2 (map, key)", args.len()));
    }
    let Value::Map(m) = &args[0] else {
        return Err(MatroError::runtime("isKey: first argument must be containers.Map"));
    };
    let k = MapData::key_of(&args[1])?;
    ok1(Value::bool(m.entries.contains_key(&k)))
}

pub fn fn_keys(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("keys", "1", args.len()));
    }
    let Value::Map(m) = &args[0] else {
        return Err(MatroError::runtime("keys: argument must be containers.Map"));
    };
    let ks: Vec<Value> = m.entries.keys().cloned().map(Value::String).collect();
    let cell = if ks.is_empty() {
        CellArray::empty()
    } else {
        CellArray::from_rows(vec![ks])?
    };
    ok1(Value::Cell(cell))
}

pub fn fn_values(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("values", "1", args.len()));
    }
    let Value::Map(m) = &args[0] else {
        return Err(MatroError::runtime("values: argument must be containers.Map"));
    };
    let vs: Vec<Value> = m.entries.values().cloned().collect();
    let cell = if vs.is_empty() {
        CellArray::empty()
    } else {
        CellArray::from_rows(vec![vs])?
    };
    ok1(Value::Cell(cell))
}

pub fn fn_binop_c(
    name: &str,
    args: &[Value],
    op: impl Fn(Complex, Complex) -> Complex,
) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity(name, "2", args.len()));
    }
    let a = args[0].to_dense_matrix()?;
    let b = args[1].to_dense_matrix()?;
    ok1(Value::matrix(a.elementwise_c(&b, op)?))
}

pub fn fn_binop_cmp(
    name: &str,
    args: &[Value],
    op: impl Fn(f64, f64) -> bool,
) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity(name, "2", args.len()));
    }
    let a = args[0].to_dense_matrix()?;
    let b = args[1].to_dense_matrix()?;
    ok1(Value::matrix(
        a.elementwise(&b, |x, y| if op(x, y) { 1.0 } else { 0.0 })?
            .with_logical(true),
    ))
}
