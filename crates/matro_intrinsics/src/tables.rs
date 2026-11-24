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
    let mut key_name: Option<String> = None;
    let mut i = 2;
    while i + 1 < args.len() {
        if let Ok(k) = args[i].as_str() {
            if k.eq_ignore_ascii_case("Keys") {
                key_name = Some(args[i + 1].as_str()?.to_string());
            }
        }
        i += 2;
    }
    let key = match key_name {
        Some(k) => k,
        None => t1
            .col_names
            .iter()
            .find(|n| t2.col_names.contains(*n))
            .cloned()
            .ok_or_else(|| MatroError::runtime("join: no common key variable between tables"))?,
    };
    let k1 = t1.col_index(&key).ok_or_else(|| {
        MatroError::runtime(format!("join: key '{key}' not found in left table"))
    })?;
    let k2 = t2.col_index(&key).ok_or_else(|| {
        MatroError::runtime(format!("join: key '{key}' not found in right table"))
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
                JoinKind::Outer => pairs.push((Some(r1), None)),
                JoinKind::Inner => {}
            }
        }
    }
    if kind == JoinKind::Outer {
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

    // Key column (merge left/right for outerjoin)
    out_names.push(key.clone());
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
        let col_name = if t2.col_names.contains(cname) {
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
        let col_name = if t1.col_names.contains(cname) {
            format!("{cname}_right")
        } else {
            cname.clone()
        };
        out_names.push(col_name);
        out_cols.push(build_col_from_rows(&t2.columns[c], &r2_rows)?);
    }

    ok1(Value::Table(TableData::new(out_names, out_cols)?))
}

/// `groupsummary(T, groupvar, method [, datavars])`
pub fn fn_groupsummary(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 4 {
        return Err(MatroError::arity(
            "groupsummary",
            "2 to 4 (T, groupvar [, method [, datavars]])",
            args.len(),
        ));
    }
    let Value::Table(t) = &args[0] else {
        return Err(MatroError::runtime("groupsummary: first argument must be a table"));
    };
    let gvar = args[1].as_str()?;
    let g_idx = t.col_index(gvar).ok_or_else(|| {
        MatroError::runtime(format!("groupsummary: unknown grouping variable '{gvar}'"))
    })?;
    let method = args
        .get(2)
        .and_then(|v| v.as_str().ok())
        .unwrap_or("count")
        .to_ascii_lowercase();

    let data_cols: Vec<usize> = if let Some(dv) = args.get(3) {
        let names: Vec<String> = match dv {
            Value::String(s) => vec![s.clone()],
            Value::Cell(c) => c.data.iter().map(|v| Ok(v.as_str()?.to_string())).collect::<Result<_>>()?,
            _ => return Err(MatroError::runtime("groupsummary: datavars must be string or cellstr")),
        };
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
            .filter(|&c| c != g_idx && matches!(t.columns[c], Value::Matrix(_)))
            .collect()
    };

    // Group row indices preserving sorted unique key order
    let mut groups: std::collections::BTreeMap<String, Vec<usize>> = std::collections::BTreeMap::new();
    for r in 0..t.nrows {
        let k = table_cell_key(&t.columns[g_idx], r);
        groups.entry(k).or_default().push(r);
    }

    let first_rows: Vec<Option<usize>> = groups.values().map(|rs| rs.first().copied()).collect();
    let counts: Vec<f64> = groups.values().map(|rs| rs.len() as f64).collect();

    let mut out_names = vec![gvar.to_string(), "GroupCount".to_string()];
    let mut out_cols = vec![
        build_col_from_rows(&t.columns[g_idx], &first_rows)?,
        Value::matrix(Matrix::col_vec(counts)),
    ];

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
                    "std" => {
                        if vals.len() <= 1 {
                            0.0
                        } else {
                            let mu = vals.iter().sum::<f64>() / (vals.len() as f64);
                            let var = vals.iter().map(|&x| (x - mu).powi(2)).sum::<f64>()
                                / ((vals.len() - 1) as f64);
                            var.sqrt()
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

