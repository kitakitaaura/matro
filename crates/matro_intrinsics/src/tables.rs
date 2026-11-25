//! MATLAB `table`, `containers.Map`/`dictionary`, `sortrows`, and operator functions
//! (`plus`, `minus`, `times`, `rdivide`, `power`, etc. for `bsxfun` / `@plus`).

use matro_loader::ModuleLoader;
use matro_types::{CellArray, Complex, MapData, MatroError, Matrix, Result, TableData, Value};

fn ok1(v: Value) -> Result<Vec<Value>> {
    Ok(vec![v])
}

/// `table(col1, col2, ..., 'VariableNames', {'A', 'B'})`
/// or `table('Size', [nr nc], 'VariableTypes', {...}, 'VariableNames', {...})`
pub fn fn_table(args: &[Value]) -> Result<Vec<Value>> {
    let mut cols = Vec::new();
    let mut custom_names: Option<Vec<String>> = None;
    let mut prealloc_size: Option<(usize, usize)> = None;
    let mut var_types: Option<Vec<String>> = None;
    let mut i = 0;
    while i < args.len() {
        if let Ok(s) = args[i].as_str() {
            let low = s.to_ascii_lowercase();
            if low == "variablenames" && i + 1 < args.len() {
                let names = extract_name_list(&args[i + 1])?;
                custom_names = Some(names);
                i += 2;
                continue;
            } else if low == "size" && i + 1 < args.len() {
                let sz = args[i + 1].to_dense_matrix()?;
                if sz.len() >= 2 {
                    prealloc_size = Some((
                        sz.data()[0].max(0.0).round() as usize,
                        sz.data()[1].max(0.0).round() as usize,
                    ));
                }
                i += 2;
                continue;
            } else if low == "variabletypes" && i + 1 < args.len() {
                var_types = Some(extract_name_list(&args[i + 1])?);
                i += 2;
                continue;
            } else if low == "rownames" && i + 1 < args.len() {
                i += 2;
                continue;
            }
        }
        cols.push(args[i].clone());
        i += 1;
    }
    if let Some((nr, nc)) = prealloc_size {
        if cols.is_empty() {
            for c in 0..nc {
                let vt = var_types
                    .as_ref()
                    .and_then(|v| v.get(c))
                    .map(|s| s.to_ascii_lowercase())
                    .unwrap_or_else(|| "double".to_string());
                let col = match vt.as_str() {
                    "string" | "cell" | "cellstr" | "char" => {
                        let data = vec![Value::String(String::new()); nr];
                        Value::Cell(CellArray {
                            rows: nr,
                            cols: if nr == 0 { 0 } else { 1 },
                            data,
                        })
                    }
                    "logical" => Value::matrix(Matrix::zeros(nr, 1).with_logical(true)),
                    _ => Value::matrix(Matrix::zeros(nr, 1)),
                };
                cols.push(col);
            }
        }
    }
    let names = match custom_names {
        Some(n) => n,
        None => (1..=cols.len()).map(|k| format!("Var{k}")).collect(),
    };
    ok1(Value::Table(TableData::new(names, cols)?))
}

fn extract_name_list(v: &Value) -> Result<Vec<String>> {
    match v {
        Value::String(s) => Ok(vec![s.clone()]),
        Value::Cell(c) => c.data.iter().map(|x| Ok(x.as_str()?.to_string())).collect(),
        Value::Object(o) if o.class_name == "string" => match o.fields.get("Data") {
            Some(Value::Cell(c)) => c.data.iter().map(|x| Ok(x.as_str()?.to_string())).collect(),
            _ => Ok(Vec::new()),
        },
        _ => Err(MatroError::runtime("Expected string, cellstr, or string array")),
    }
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
    // Handle `sortrows(A, 'descend')` / `sortrows(A, 'ascend')` where 2nd arg is direction!
    let (col_spec, dir_arg) = if args.len() == 2 {
        if let Some(s) = crate::value_as_text(&args[1]) {
            if s.eq_ignore_ascii_case("ascend") || s.eq_ignore_ascii_case("descend") {
                (None, Some(args[1].clone()))
            } else {
                (Some(&args[1]), None)
            }
        } else {
            (Some(&args[1]), None)
        }
    } else {
        (args.get(1), args.get(2).cloned())
    };
    let dir_list: Vec<bool> = match &dir_arg {
        None => vec![false],
        Some(v) => {
            if let Ok(names) = extract_name_list(v) {
                names.iter().map(|s| s.eq_ignore_ascii_case("descend")).collect()
            } else {
                vec![false]
            }
        }
    };
    let dir_for = |k: usize| -> bool {
        if dir_list.len() == 1 {
            dir_list[0]
        } else {
            dir_list.get(k).copied().unwrap_or(false)
        }
    };

    match &args[0] {
        Value::Table(t) => {
            let mut key_cols: Vec<(usize, bool)> = Vec::new();
            match col_spec {
                None => {
                    for c in 0..t.width() {
                        key_cols.push((c, dir_for(c)));
                    }
                }
                Some(Value::String(name)) => {
                    let idx = t.col_index(name).ok_or_else(|| {
                        MatroError::runtime(format!("sortrows: unknown table variable '{name}'"))
                    })?;
                    key_cols.push((idx, dir_for(0)));
                }
                Some(Value::Cell(_)) | Some(Value::Object(_)) => {
                    let names = extract_name_list(col_spec.unwrap())?;
                    for (k, name) in names.iter().enumerate() {
                        let idx = t.col_index(name).ok_or_else(|| {
                            MatroError::runtime(format!("sortrows: unknown table variable '{name}'"))
                        })?;
                        key_cols.push((idx, dir_for(k)));
                    }
                }
                Some(v) => {
                    for (k, &val) in v.to_dense_matrix()?.data().iter().enumerate() {
                        let s = val.round() as isize;
                        if s == 0 {
                            return Err(MatroError::index("sortrows: column index must be non-zero"));
                        }
                        let idx = s.unsigned_abs() - 1;
                        if idx >= t.width() {
                            return Err(MatroError::index("sortrows: column index out of bounds"));
                        }
                        key_cols.push((idx, (s < 0) ^ dir_for(k)));
                    }
                }
            }

            let mut order: Vec<usize> = (0..t.nrows).collect();
            order.sort_by(|&a, &b| {
                for &(c, descending) in &key_cols {
                    let cmp = match &t.columns[c] {
                        Value::Matrix(m) => m.get(a, 0).total_cmp(&m.get(b, 0)),
                        Value::Cell(cell) => format!("{}", cell.data[a]).cmp(&format!("{}", cell.data[b])),
                        _ => std::cmp::Ordering::Equal,
                    };
                    if cmp != std::cmp::Ordering::Equal {
                        return if descending { cmp.reverse() } else { cmp };
                    }
                }
                std::cmp::Ordering::Equal
            });
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
            if let Some(col_arg) = col_spec {
                for (k, &v) in col_arg.to_dense_matrix()?.data().iter().enumerate() {
                    let i = v.round() as isize;
                    let idx = i.unsigned_abs().saturating_sub(1);
                    if idx >= m.cols() {
                        return Err(MatroError::index("sortrows: column index out of bounds"));
                    }
                    key_cols.push((idx, (i < 0) ^ dir_for(k)));
                }
            } else {
                for c in 0..m.cols() {
                    key_cols.push((c, dir_for(c)));
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

// ---------------------------------------------------------------------------
// Relational Table Operations (`head`, `tail`, `array2table`, `table2array`,
// `renamevars`, `removevars`, `join`, `innerjoin`, `outerjoin`, `groupsummary`)
// ---------------------------------------------------------------------------

pub fn fn_head_tail(args: &[Value], is_tail: bool) -> Result<Vec<Value>> {
    let fname = if is_tail { "tail" } else { "head" };
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity(fname, "1 or 2 (T [, k])", args.len()));
    }
    let k = if args.len() == 2 {
        args[1].as_scalar()?.max(0.0).round() as usize
    } else {
        8
    };
    match &args[0] {
        Value::Table(t) => {
            let take = k.min(t.nrows);
            let rows: Vec<usize> = if is_tail {
                ((t.nrows - take)..t.nrows).collect()
            } else {
                (0..take).collect()
            };
            let cols: Vec<usize> = (0..t.width()).collect();
            ok1(Value::Table(t.slice(&rows, &cols)?))
        }
        Value::Matrix(m) => {
            let take = k.min(m.rows());
            let start = if is_tail { m.rows() - take } else { 0 };
            let mut out = Matrix::zeros(take, m.cols());
            for c in 0..m.cols() {
                for r in 0..take {
                    out.set_c(r, c, m.get_c(start + r, c));
                }
            }
            ok1(Value::matrix(out.normalize()))
        }
        other => Err(MatroError::runtime(format!(
            "{fname}: expected table or matrix, got {}",
            other.type_name()
        ))),
    }
}

pub fn fn_array2table(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("array2table", ">= 1 (A [, 'VariableNames', {...}])", 0));
    }
    let m = args[0].to_dense_matrix()?;
    let mut names: Vec<String> = (1..=m.cols()).map(|c| format!("Var{c}")).collect();
    let mut i = 1;
    while i + 1 < args.len() {
        if let Ok(k) = args[i].as_str() {
            if k.eq_ignore_ascii_case("VariableNames") {
                if let Value::Cell(c) = &args[i + 1] {
                    names = c
                        .data
                        .iter()
                        .map(|v| v.as_str().unwrap_or("Var").to_string())
                        .collect();
                }
            }
        }
        i += 2;
    }
    let mut cols = Vec::with_capacity(m.cols());
    for c in 0..m.cols() {
        let col_data: Vec<Complex> = (0..m.rows()).map(|r| m.get_c(r, c)).collect();
        cols.push(Value::matrix(Matrix::complex_col_vec(col_data).normalize()));
    }
    ok1(Value::Table(TableData::new(names, cols)?))
}

pub fn fn_table2array(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("table2array", "1 (T)", args.len()));
    }
    let Value::Table(t) = &args[0] else {
        return Err(MatroError::runtime("table2array: expected a table"));
    };
    let mut out = Matrix::zeros(t.nrows, t.width());
    for (c, col) in t.columns.iter().enumerate() {
        let m = col.to_dense_matrix()?;
        for r in 0..t.nrows {
            out.set_c(r, c, m.get_c(r, 0));
        }
    }
    ok1(Value::matrix(out.normalize()))
}

pub fn fn_renamevars(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 3 {
        return Err(MatroError::arity("renamevars", "3 (T, oldNames, newNames)", args.len()));
    }
    let Value::Table(t) = &args[0] else {
        return Err(MatroError::runtime("renamevars: first argument must be a table"));
    };
    let extract_names = |v: &Value| -> Result<Vec<String>> {
        match v {
            Value::String(s) => Ok(vec![s.clone()]),
            Value::Cell(c) => c.data.iter().map(|x| Ok(x.as_str()?.to_string())).collect(),
            _ => Err(MatroError::runtime("renamevars: variable names must be strings or cellstr")),
        }
    };
    let olds = extract_names(&args[1])?;
    let news = extract_names(&args[2])?;
    if olds.len() != news.len() {
        return Err(MatroError::dimension("renamevars: oldNames and newNames must have same length"));
    }
    let mut out = t.clone();
    for (o, n) in olds.iter().zip(news.iter()) {
        let idx = out.col_index(o).ok_or_else(|| {
            MatroError::runtime(format!("renamevars: unknown variable '{o}'"))
        })?;
        out.col_names[idx] = n.clone();
    }
    ok1(Value::Table(out))
}

pub fn fn_removevars(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("removevars", "2 (T, vars)", args.len()));
    }
    let Value::Table(t) = &args[0] else {
        return Err(MatroError::runtime("removevars: first argument must be a table"));
    };
    let drop_list: Vec<String> = match &args[1] {
        Value::String(s) => vec![s.clone()],
        Value::Cell(c) => c.data.iter().map(|x| Ok(x.as_str()?.to_string())).collect::<Result<_>>()?,
        _ => return Err(MatroError::runtime("removevars: vars must be string or cellstr")),
    };
    let keep: Vec<usize> = (0..t.width())
        .filter(|&c| !drop_list.contains(&t.col_names[c]))
        .collect();
    let rows: Vec<usize> = (0..t.nrows).collect();
    ok1(Value::Table(t.slice(&rows, &keep)?))
}

fn table_cell_key(col: &Value, row: usize) -> String {
    match col {
        Value::Matrix(m) => {
            let x = m.get(row, 0);
            if x.fract() == 0.0 && x.abs() < 1e15 {
                format!("{}", x as i64)
            } else {
                format!("{x:.12}")
            }
        }
        Value::Cell(c) => c
            .data
            .get(row)
            .map(|v| v.as_str().map(ToString::to_string).unwrap_or_else(|_| format!("{v}")))
            .unwrap_or_default(),
        other => format!("{other}"),
    }
}

fn table_col_elem_opt(col: &Value, row: Option<usize>) -> Value {
    match (col, row) {
        (Value::Matrix(m), Some(r)) => Value::matrix(Matrix::complex_scalar(m.get_c(r, 0)).normalize()),
        (Value::Matrix(_), None) => Value::scalar(f64::NAN),
        (Value::Cell(c), Some(r)) => c.data.get(r).cloned().unwrap_or_else(Value::empty),
        (Value::Cell(_), None) => Value::String(String::new()),
        (other, Some(_)) => other.clone(),
        (_, None) => Value::empty(),
    }
}

fn build_col_from_rows(template: &Value, rows: &[Option<usize>]) -> Result<Value> {
    match template {
        Value::Matrix(_) => {
            let mut vals = Vec::with_capacity(rows.len());
            for &opt_r in rows {
                let v = table_col_elem_opt(template, opt_r);
                vals.push(v.to_dense_matrix()?.get_c(0, 0));
            }
            Ok(Value::matrix(Matrix::complex_col_vec(vals).normalize()))
        }
        Value::Cell(_) => {
            let items: Vec<Vec<Value>> = rows
                .iter()
                .map(|&opt_r| vec![table_col_elem_opt(template, opt_r)])
                .collect();
            if items.is_empty() {
                Ok(Value::Cell(CellArray::empty()))
            } else {
                Ok(Value::Cell(CellArray::from_rows(items)?))
            }
        }
        _ => {
            let items: Vec<Vec<Value>> = rows
                .iter()
                .map(|&opt_r| vec![table_col_elem_opt(template, opt_r)])
                .collect();
            if items.is_empty() {
                Ok(Value::Cell(CellArray::empty()))
            } else {
                Ok(Value::Cell(CellArray::from_rows(items)?))
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum JoinKind {
    Left,
    Inner,
    Outer,
}

pub fn fn_table_join(args: &[Value], kind: JoinKind) -> Result<Vec<Value>> {
    if args.len() < 2 {
        return Err(MatroError::arity("join", ">= 2 (T1, T2 [, 'Keys', key])", args.len()));
    }
    let (Value::Table(t1), Value::Table(t2)) = (&args[0], &args[1]) else {
        return Err(MatroError::runtime("join: first two arguments must be tables"));
    };
    let mut left_key_name: Option<String> = None;
    let mut right_key_name: Option<String> = None;
    let mut outer_type = "full".to_string();
    let mut i = 2;
    while i + 1 < args.len() {
        if let Ok(k) = args[i].as_str() {
            match k.to_ascii_lowercase().as_str() {
                "keys" => {
                    let kn = args[i + 1].as_str()?.to_string();
                    left_key_name = Some(kn.clone());
                    right_key_name = Some(kn);
                }
                "leftkeys" => left_key_name = Some(args[i + 1].as_str()?.to_string()),
                "rightkeys" => right_key_name = Some(args[i + 1].as_str()?.to_string()),
                "type" => outer_type = args[i + 1].as_str()?.to_ascii_lowercase(),
                _ => {}
            }
        }
        i += 2;
    }
    let (lkey, rkey) = match (left_key_name, right_key_name) {
        (Some(lk), Some(rk)) => (lk, rk),
        (Some(lk), None) => (lk.clone(), lk),
        (None, Some(rk)) => (rk.clone(), rk),
        (None, None) => {
            let common = t1
                .col_names
                .iter()
                .find(|n| t2.col_names.contains(*n))
                .cloned()
                .ok_or_else(|| MatroError::runtime("join: no common key variable between tables"))?;
            (common.clone(), common)
        }
    };
    let k1 = t1.col_index(&lkey).ok_or_else(|| {
        MatroError::runtime(format!("join: key '{lkey}' not found in left table"))
    })?;
    let k2 = t2.col_index(&rkey).ok_or_else(|| {
        MatroError::runtime(format!("join: key '{rkey}' not found in right table"))
    })?;

    let mut pairs: Vec<(Option<usize>, Option<usize>)> = Vec::new();
    let mut matched_r2 = vec![false; t2.nrows];

    for r1 in 0..t1.nrows {
        let kv = table_cell_key(&t1.columns[k1], r1);
        let mut found = false;
        for r2 in 0..t2.nrows {
            if table_cell_key(&t2.columns[k2], r2) == kv {
                pairs.push((Some(r1), Some(r2)));
                matched_r2[r2] = true;
                found = true;
                if kind == JoinKind::Left {
                    break;
                }
            }
        }
        if !found {
            match kind {
                JoinKind::Left => {
                    return Err(MatroError::runtime(format!(
                        "join: key '{kv}' in left table not found in right table (use outerjoin or innerjoin)"
                    )));
                }
                JoinKind::Outer if outer_type == "full" || outer_type == "left" => {
                    pairs.push((Some(r1), None));
                }
                _ => {}
            }
        }
    }
    if kind == JoinKind::Outer && (outer_type == "full" || outer_type == "right") {
        for (r2, &m) in matched_r2.iter().enumerate() {
            if !m {
                pairs.push((None, Some(r2)));
            }
        }
    }

    let r1_rows: Vec<Option<usize>> = pairs.iter().map(|p| p.0).collect();
    let r2_rows: Vec<Option<usize>> = pairs.iter().map(|p| p.1).collect();

    let mut out_names = Vec::new();
    let mut out_cols = Vec::new();

    out_names.push(lkey.clone());
    match &t1.columns[k1] {
        Value::Matrix(_) => {
            let mut vals = Vec::with_capacity(pairs.len());
            for &(o1, o2) in &pairs {
                let v = if let Some(r) = o1 {
                    table_col_elem_opt(&t1.columns[k1], Some(r))
                } else {
                    table_col_elem_opt(&t2.columns[k2], o2)
                };
                vals.push(v.to_dense_matrix()?.get_c(0, 0));
            }
            out_cols.push(Value::matrix(Matrix::complex_col_vec(vals).normalize()));
        }
        _ => {
            let items: Vec<Vec<Value>> = pairs
                .iter()
                .map(|&(o1, o2)| {
                    let v = if let Some(r) = o1 {
                        table_col_elem_opt(&t1.columns[k1], Some(r))
                    } else {
                        table_col_elem_opt(&t2.columns[k2], o2)
                    };
                    vec![v]
                })
                .collect();
            let cell = if items.is_empty() {
                CellArray::empty()
            } else {
                CellArray::from_rows(items)?
            };
            out_cols.push(Value::Cell(cell));
        }
    }

    for (c, cname) in t1.col_names.iter().enumerate() {
        if c == k1 {
            continue;
        }
        let col_name = if t2.col_names.contains(cname) && cname != &rkey {
            format!("{cname}_left")
        } else {
            cname.clone()
        };
        out_names.push(col_name);
        out_cols.push(build_col_from_rows(&t1.columns[c], &r1_rows)?);
    }
    for (c, cname) in t2.col_names.iter().enumerate() {
        if c == k2 {
            continue;
        }
        let col_name = if t1.col_names.contains(cname) && cname != &lkey {
            format!("{cname}_right")
        } else {
            cname.clone()
        };
        out_names.push(col_name);
        out_cols.push(build_col_from_rows(&t2.columns[c], &r2_rows)?);
    }

    ok1(Value::Table(TableData::new(out_names, out_cols)?))
}

#[derive(Clone, PartialEq)]
enum GroupKeyAtom {
    Num(f64),
    Str(String),
}
impl Eq for GroupKeyAtom {}
impl PartialOrd for GroupKeyAtom {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for GroupKeyAtom {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Self::Num(a), Self::Num(b)) => a.total_cmp(b),
            (Self::Str(a), Self::Str(b)) => a.cmp(b),
            (Self::Num(_), Self::Str(_)) => std::cmp::Ordering::Less,
            (Self::Str(_), Self::Num(_)) => std::cmp::Ordering::Greater,
        }
    }
}

/// `groupsummary(T, groupvars, method [, datavars])`
pub fn fn_groupsummary(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 4 {
        return Err(MatroError::arity(
            "groupsummary",
            "2 to 4 (T, groupvars [, method [, datavars]])",
            args.len(),
        ));
    }
    let Value::Table(t) = &args[0] else {
        return Err(MatroError::runtime("groupsummary: first argument must be a table"));
    };
    let gvars = extract_name_list(&args[1])?;
    let g_idxs: Vec<usize> = gvars
        .iter()
        .map(|gvar| {
            t.col_index(gvar).ok_or_else(|| {
                MatroError::runtime(format!("groupsummary: unknown grouping variable '{gvar}'"))
            })
        })
        .collect::<Result<_>>()?;
    let method = args
        .get(2)
        .and_then(|v| v.as_str().ok())
        .unwrap_or("count")
        .to_ascii_lowercase();

    let data_cols: Vec<usize> = if let Some(dv) = args.get(3) {
        let names = extract_name_list(dv)?;
        names
            .iter()
            .map(|n| {
                t.col_index(n).ok_or_else(|| {
                    MatroError::runtime(format!("groupsummary: unknown data variable '{n}'"))
                })
            })
            .collect::<Result<_>>()?
    } else {
        (0..t.width())
            .filter(|&c| !g_idxs.contains(&c) && matches!(t.columns[c], Value::Matrix(_)))
            .collect()
    };

    let mut groups: std::collections::BTreeMap<Vec<GroupKeyAtom>, Vec<usize>> =
        std::collections::BTreeMap::new();
    for r in 0..t.nrows {
        let key: Vec<GroupKeyAtom> = g_idxs
            .iter()
            .map(|&gi| match &t.columns[gi] {
                Value::Matrix(m) => GroupKeyAtom::Num(m.get(r, 0)),
                other => GroupKeyAtom::Str(table_cell_key(other, r)),
            })
            .collect();
        groups.entry(key).or_default().push(r);
    }

    let first_rows: Vec<Option<usize>> = groups.values().map(|rs| rs.first().copied()).collect();
    let counts: Vec<f64> = groups.values().map(|rs| rs.len() as f64).collect();

    let mut out_names = gvars.clone();
    let mut out_cols: Vec<Value> = g_idxs
        .iter()
        .map(|&gi| build_col_from_rows(&t.columns[gi], &first_rows))
        .collect::<Result<_>>()?;
    out_names.push("GroupCount".to_string());
    out_cols.push(Value::matrix(Matrix::col_vec(counts)));

    if method != "count" && method != "none" {
        for &dc in &data_cols {
            let m = t.columns[dc].to_dense_matrix()?;
            let mut agg = Vec::with_capacity(groups.len());
            for rs in groups.values() {
                let vals: Vec<f64> = rs.iter().map(|&r| m.get(r, 0)).collect();
                let v = match method.as_str() {
                    "mean" => vals.iter().sum::<f64>() / (vals.len() as f64),
                    "sum" => vals.iter().sum::<f64>(),
                    "min" => vals.iter().copied().fold(f64::INFINITY, f64::min),
                    "max" => vals.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                    "median" => {
                        let mut s = vals;
                        s.sort_by(f64::total_cmp);
                        let n = s.len();
                        if n % 2 == 1 { s[n / 2] } else { 0.5 * (s[n / 2 - 1] + s[n / 2]) }
                    }
                    "std" | "var" => {
                        if vals.len() <= 1 {
                            0.0
                        } else {
                            let mu = vals.iter().sum::<f64>() / (vals.len() as f64);
                            let var = vals.iter().map(|&x| (x - mu).powi(2)).sum::<f64>()
                                / ((vals.len() - 1) as f64);
                            if method == "std" { var.sqrt() } else { var }
                        }
                    }
                    other => {
                        return Err(MatroError::runtime(format!(
                            "groupsummary: unsupported method '{other}'"
                        )));
                    }
                };
                agg.push(v);
            }
            out_names.push(format!("{method}_{}", t.col_names[dc]));
            out_cols.push(Value::matrix(Matrix::col_vec(agg)));
        }
    }

    ok1(Value::Table(TableData::new(out_names, out_cols)?))
}

pub fn fn_addvars(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 {
        return Err(MatroError::arity("addvars", ">= 2 (T, var1, ...)", args.len()));
    }
    let Value::Table(t) = &args[0] else {
        return Err(MatroError::runtime("addvars: first argument must be a table"));
    };
    let mut new_cols = Vec::new();
    let mut new_names: Option<Vec<String>> = None;
    let mut before: Option<String> = None;
    let mut after: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        if let Ok(k) = args[i].as_str() {
            if i + 1 < args.len() {
                match k.to_ascii_lowercase().as_str() {
                    "newvariablenames" => {
                        new_names = Some(extract_name_list(&args[i + 1])?);
                        i += 2;
                        continue;
                    }
                    "before" => {
                        before = Some(args[i + 1].as_str()?.to_string());
                        i += 2;
                        continue;
                    }
                    "after" => {
                        after = Some(args[i + 1].as_str()?.to_string());
                        i += 2;
                        continue;
                    }
                    _ => {}
                }
            }
        }
        new_cols.push(args[i].clone());
        i += 1;
    }
    let mut names = t.col_names.clone();
    let mut cols = t.columns.clone();
    let insert_idx = if let Some(b) = before {
        t.col_index(&b).unwrap_or(cols.len())
    } else if let Some(a) = after {
        t.col_index(&a).map(|idx| idx + 1).unwrap_or(cols.len())
    } else {
        cols.len()
    };
    let add_names: Vec<String> = match new_names {
        Some(ns) if ns.len() == new_cols.len() => ns,
        _ => (1..=new_cols.len())
            .map(|k| format!("Var{}", t.width() + k))
            .collect(),
    };
    for (offset, (nm, col)) in add_names.into_iter().zip(new_cols.into_iter()).enumerate() {
        names.insert(insert_idx + offset, nm);
        cols.insert(insert_idx + offset, col);
    }
    ok1(Value::Table(TableData::new(names, cols)?))
}

pub fn fn_movevars(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 4 {
        return Err(MatroError::arity("movevars", "4 (T, vars, 'Before'|'After', loc)", args.len()));
    }
    let Value::Table(t) = &args[0] else {
        return Err(MatroError::runtime("movevars: first argument must be a table"));
    };
    let to_move = extract_name_list(&args[1])?;
    let rel = args[2].as_str()?.to_ascii_lowercase();
    let loc_name = args[3].as_str()?;

    let mut moved_names = Vec::new();
    let mut moved_cols = Vec::new();
    let mut rem_names = Vec::new();
    let mut rem_cols = Vec::new();
    for (nm, col) in t.col_names.iter().zip(t.columns.iter()) {
        if to_move.contains(nm) {
            moved_names.push(nm.clone());
            moved_cols.push(col.clone());
        } else {
            rem_names.push(nm.clone());
            rem_cols.push(col.clone());
        }
    }
    let pos = rem_names
        .iter()
        .position(|n| n == loc_name)
        .map(|idx| if rel == "after" { idx + 1 } else { idx })
        .unwrap_or(rem_names.len());
    for (off, (nm, col)) in moved_names.into_iter().zip(moved_cols.into_iter()).enumerate() {
        rem_names.insert(pos + off, nm);
        rem_cols.insert(pos + off, col);
    }
    ok1(Value::Table(TableData::new(rem_names, rem_cols)?))
}

pub fn fn_table2struct(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("table2struct", ">= 1 (T [, 'ToScalar', tf])", 0));
    }
    let Value::Table(t) = &args[0] else {
        return Err(MatroError::runtime("table2struct: expected a table"));
    };
    let mut to_scalar = false;
    let mut i = 1;
    while i + 1 < args.len() {
        if let Ok(k) = args[i].as_str() {
            if k.eq_ignore_ascii_case("ToScalar") {
                to_scalar = args[i + 1].is_truthy();
            }
        }
        i += 2;
    }
    if to_scalar {
        let mut st = std::collections::BTreeMap::new();
        for (nm, col) in t.col_names.iter().zip(t.columns.iter()) {
            st.insert(nm.clone(), col.clone());
        }
        return ok1(Value::Struct(st));
    }
    let mut items = Vec::with_capacity(t.nrows);
    for r in 0..t.nrows {
        let mut st = std::collections::BTreeMap::new();
        for (nm, col) in t.col_names.iter().zip(t.columns.iter()) {
            st.insert(nm.clone(), table_col_elem_opt(col, Some(r)));
        }
        items.push(st);
    }
    ok1(Value::StructArray(matro_types::StructArray {
        rows: t.nrows,
        cols: if t.nrows == 0 { 0 } else { 1 },
        items,
    }))
}

pub fn fn_struct2table(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("struct2table", "1 (S)", 0));
    }
    match &args[0] {
        Value::Struct(map) => {
            let mut names = Vec::with_capacity(map.len());
            let mut cols = Vec::with_capacity(map.len());
            for (k, v) in map {
                names.push(k.clone());
                cols.push(v.clone());
            }
            ok1(Value::Table(TableData::new(names, cols)?))
        }
        Value::StructArray(sa) => {
            if sa.items.is_empty() {
                return ok1(Value::Table(TableData::new(vec![], vec![])?));
            }
            let names: Vec<String> = sa.items[0].keys().cloned().collect();
            let mut cols = Vec::with_capacity(names.len());
            for nm in &names {
                let all_scalar = sa
                    .items
                    .iter()
                    .all(|st| st.get(nm).and_then(|v| v.as_scalar().ok()).is_some());
                if all_scalar {
                    let vals: Vec<f64> = sa
                        .items
                        .iter()
                        .map(|st| st[nm].as_scalar().unwrap_or(f64::NAN))
                        .collect();
                    cols.push(Value::matrix(Matrix::col_vec(vals)));
                } else {
                    let data: Vec<Value> = sa
                        .items
                        .iter()
                        .map(|st| st.get(nm).cloned().unwrap_or_else(Value::empty))
                        .collect();
                    cols.push(Value::Cell(CellArray {
                        rows: data.len(),
                        cols: if data.is_empty() { 0 } else { 1 },
                        data,
                    }));
                }
            }
            ok1(Value::Table(TableData::new(names, cols)?))
        }
        other => Err(MatroError::runtime(format!(
            "struct2table: expected struct or struct array, got {}",
            other.type_name()
        ))),
    }
}

pub fn fn_cell2table(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("cell2table", ">= 1 (C [, 'VariableNames', {...}])", 0));
    }
    let Value::Cell(c) = &args[0] else {
        return Err(MatroError::runtime("cell2table: first argument must be a cell array"));
    };
    let mut names: Vec<String> = (1..=c.cols).map(|k| format!("Var{k}")).collect();
    let mut i = 1;
    while i + 1 < args.len() {
        if let Ok(k) = args[i].as_str() {
            if k.eq_ignore_ascii_case("VariableNames") {
                names = extract_name_list(&args[i + 1])?;
            }
        }
        i += 2;
    }
    let mut cols = Vec::with_capacity(c.cols);
    for col in 0..c.cols {
        let mut all_num = c.rows > 0;
        let mut nums = Vec::with_capacity(c.rows);
        let mut vals = Vec::with_capacity(c.rows);
        for row in 0..c.rows {
            let v = c.get(row, col).clone();
            if let Ok(sc) = v.as_scalar() {
                nums.push(sc);
            } else {
                all_num = false;
            }
            vals.push(v);
        }
        if all_num {
            cols.push(Value::matrix(Matrix::col_vec(nums)));
        } else {
            cols.push(Value::Cell(CellArray {
                rows: c.rows,
                cols: if c.rows == 0 { 0 } else { 1 },
                data: vals,
            }));
        }
    }
    ok1(Value::Table(TableData::new(names, cols)?))
}

pub fn fn_table2cell(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("table2cell", "1 (T)", args.len()));
    }
    let Value::Table(t) = &args[0] else {
        return Err(MatroError::runtime("table2cell: expected a table"));
    };
    let mut rows = Vec::with_capacity(t.nrows);
    for r in 0..t.nrows {
        let mut row = Vec::with_capacity(t.width());
        for c in 0..t.width() {
            row.push(table_col_elem_opt(&t.columns[c], Some(r)));
        }
        rows.push(row);
    }
    let cell = if rows.is_empty() {
        CellArray::empty()
    } else {
        CellArray::from_rows(rows)?
    };
    ok1(Value::Cell(cell))
}

