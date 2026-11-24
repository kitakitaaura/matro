//! Modern MATLAB data types:
//! `datetime`, `duration` (`seconds`/`minutes`/`hours`/`days`), `categorical`,
//! `timetable`, and `string` arrays (`strlength`, `split`, `join`, `startsWith`,
//! `endsWith`, `erase`, `strip`, `replace`).

use std::collections::BTreeMap;

use matro_types::{CellArray, MatroError, Matrix, ObjectInstance, Result, TableData, Value};

fn ok1(v: Value) -> Result<Vec<Value>> {
    Ok(vec![v])
}

/// Howard Hinnant's exact civil-to-days algorithm (days since 1970-01-01).
pub fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = y - if m <= 2 { 1 } else { 0 };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// Inverse of `days_from_civil`: days since 1970-01-01 -> `(year, month, day)`.
pub fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    (y + if m <= 2 { 1 } else { 0 }, m, d)
}

pub fn epoch_to_parts(epoch_sec: f64) -> (i64, i64, i64, i64, i64, f64) {
    let total_days = (epoch_sec / 86400.0).floor() as i64;
    let rem = epoch_sec - (total_days as f64) * 86400.0;
    let (y, m, d) = civil_from_days(total_days);
    let h = (rem / 3600.0).floor() as i64;
    let mi = ((rem - (h as f64) * 3600.0) / 60.0).floor() as i64;
    let s = rem - (h as f64) * 3600.0 - (mi as f64) * 60.0;
    (y, m, d, h, mi, s)
}

pub fn format_epoch(epoch_sec: f64) -> String {
    let (y, m, d, h, mi, s) = epoch_to_parts(epoch_sec);
    if h == 0 && mi == 0 && s.abs() < 1e-6 {
        format!("{y:04}-{m:02}-{d:02}")
    } else {
        format!("{y:04}-{m:02}-{d:02} {h:02}:{mi:02}:{:02}", s.round() as i64)
    }
}

pub fn make_datetime_from_epochs(epochs: Vec<f64>) -> Value {
    let mut ys = Vec::with_capacity(epochs.len());
    let mut ms = Vec::with_capacity(epochs.len());
    let mut ds = Vec::with_capacity(epochs.len());
    let mut hs = Vec::with_capacity(epochs.len());
    let mut mins = Vec::with_capacity(epochs.len());
    let mut secs = Vec::with_capacity(epochs.len());
    for &e in &epochs {
        let (y, m, d, h, mi, s) = epoch_to_parts(e);
        ys.push(y as f64);
        ms.push(m as f64);
        ds.push(d as f64);
        hs.push(h as f64);
        mins.push(mi as f64);
        secs.push(s);
    }
    let wrap = |v: Vec<f64>| -> Value {
        if v.len() == 1 {
            Value::scalar(v[0])
        } else {
            Value::matrix(Matrix::row_vec(v))
        }
    };
    let mut fields = BTreeMap::new();
    fields.insert("EpochSec".to_string(), wrap(epochs));
    fields.insert("Year".to_string(), wrap(ys));
    fields.insert("Month".to_string(), wrap(ms));
    fields.insert("Day".to_string(), wrap(ds));
    fields.insert("Hour".to_string(), wrap(hs));
    fields.insert("Minute".to_string(), wrap(mins));
    fields.insert("Second".to_string(), wrap(secs));
    Value::Object(ObjectInstance {
        class_name: "datetime".to_string(),
        fields,
    })
}

pub fn make_duration_from_secs(secs: Vec<f64>) -> Value {
    let mut fields = BTreeMap::new();
    let mat = if secs.len() == 1 {
        Matrix::scalar(secs[0])
    } else {
        Matrix::row_vec(secs)
    };
    fields.insert("Seconds".to_string(), Value::matrix(mat));
    Value::Object(ObjectInstance {
        class_name: "duration".to_string(),
        fields,
    })
}

fn parse_iso_datetime(s: &str) -> Result<f64> {
    let trimmed = s.trim();
    if trimmed.eq_ignore_ascii_case("now") || trimmed.eq_ignore_ascii_case("today") {
        // Deterministic epoch fallback on wasm32; real wall-clock on native
        #[cfg(not(target_arch = "wasm32"))]
        {
            let secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0);
            if trimmed.eq_ignore_ascii_case("today") {
                return Ok((secs / 86400.0).floor() * 86400.0);
            }
            return Ok(secs);
        }
        #[cfg(target_arch = "wasm32")]
        {
            return Ok(days_from_civil(2026, 1, 1) as f64 * 86400.0);
        }
    }
    let (date_part, time_part) = trimmed
        .split_once([' ', 'T'])
        .unwrap_or((trimmed, "00:00:00"));
    let dparts: Vec<i64> = date_part
        .split(['-', '/'])
        .map(|p| p.parse::<i64>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| MatroError::runtime(format!("datetime: invalid date string '{s}'")))?;
    if dparts.len() != 3 {
        return Err(MatroError::runtime(format!(
            "datetime: expected YYYY-MM-DD, got '{s}'"
        )));
    }
    let tparts: Vec<f64> = time_part
        .trim_end_matches('Z')
        .split(':')
        .map(|p| p.parse::<f64>().unwrap_or(0.0))
        .collect();
    let h = tparts.first().copied().unwrap_or(0.0);
    let mi = tparts.get(1).copied().unwrap_or(0.0);
    let sec = tparts.get(2).copied().unwrap_or(0.0);
    let days = days_from_civil(dparts[0], dparts[1], dparts[2]);
    Ok((days as f64) * 86400.0 + h * 3600.0 + mi * 60.0 + sec)
}

/// `datetime('2026-01-15')` | `datetime(Y, M, D [, H, MI, S])` | `datetime('now')`
pub fn fn_datetime(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() {
        return ok1(make_datetime_from_epochs(vec![parse_iso_datetime("now")?]));
    }
    if args.len() == 1 {
        match &args[0] {
            Value::String(s) => return ok1(make_datetime_from_epochs(vec![parse_iso_datetime(s)?])),
            Value::Cell(c) => {
                let mut epochs = Vec::with_capacity(c.data.len());
                for item in &c.data {
                    epochs.push(parse_iso_datetime(item.as_str()?)?);
                }
                return ok1(make_datetime_from_epochs(epochs));
            }
            Value::Object(o) if o.class_name == "string" => {
                if let Some(Value::Cell(c)) = o.fields.get("Data") {
                    let mut epochs = Vec::with_capacity(c.data.len());
                    for item in &c.data {
                        epochs.push(parse_iso_datetime(item.as_str()?)?);
                    }
                    return ok1(make_datetime_from_epochs(epochs));
                }
            }
            _ => {}
        }
    }
    if args.len() >= 3 && args.len() <= 6 {
        let y = args[0].as_matrix()?.data();
        let m = args[1].as_matrix()?.data();
        let d = args[2].as_matrix()?.data();
        let n = y.len().max(m.len()).max(d.len());
        let h_data = args.get(3).map(|v| v.as_matrix().map(|x| x.data())).transpose()?;
        let mi_data = args.get(4).map(|v| v.as_matrix().map(|x| x.data())).transpose()?;
        let s_data = args.get(5).map(|v| v.as_matrix().map(|x| x.data())).transpose()?;
        let get = |sl: &[f64], i: usize| if sl.len() == 1 { sl[0] } else { sl[i] };
        let mut epochs = Vec::with_capacity(n);
        for i in 0..n {
            let days = days_from_civil(
                get(y, i).round() as i64,
                get(m, i).round() as i64,
                get(d, i).round() as i64,
            );
            let hv = h_data.map_or(0.0, |sl| get(sl, i));
            let mv = mi_data.map_or(0.0, |sl| get(sl, i));
            let sv = s_data.map_or(0.0, |sl| get(sl, i));
            epochs.push((days as f64) * 86400.0 + hv * 3600.0 + mv * 60.0 + sv);
        }
        return ok1(make_datetime_from_epochs(epochs));
    }
    Err(MatroError::arity("datetime", "0, 1, or 3..=6", args.len()))
}

pub fn fn_duration(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 3 {
        return Err(MatroError::arity("duration", "3 (H, MI, S)", args.len()));
    }
    let h = args[0].as_matrix()?.data();
    let m = args[1].as_matrix()?.data();
    let s = args[2].as_matrix()?.data();
    let n = h.len().max(m.len()).max(s.len());
    let get = |sl: &[f64], i: usize| if sl.len() == 1 { sl[0] } else { sl[i] };
    let secs: Vec<f64> = (0..n)
        .map(|i| get(h, i) * 3600.0 + get(m, i) * 60.0 + get(s, i))
        .collect();
    ok1(make_duration_from_secs(secs))
}

pub fn fn_duration_unit(name: &str, args: &[Value], factor_sec: f64) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity(name, "1", args.len()));
    }
    // If given a duration object, convert to numeric in that unit (MATLAB: `hours(d)`)
    if let Value::Object(o) = &args[0] {
        if o.class_name == "duration" {
            let secs = o.fields["Seconds"].as_matrix()?;
            return ok1(Value::matrix(secs.map(|s| s / factor_sec)));
        }
    }
    let vals: Vec<f64> = args[0]
        .as_matrix()?
        .data()
        .iter()
        .map(|&v| v * factor_sec)
        .collect();
    ok1(make_duration_from_secs(vals))
}

pub fn fn_datetime_part(args: &[Value], field: &str) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity(field, "1 (datetime)", args.len()));
    }
    let Value::Object(o) = &args[0] else {
        return Err(MatroError::runtime(format!("{field}: expected a datetime object")));
    };
    if o.class_name != "datetime" {
        return Err(MatroError::runtime(format!("{field}: expected a datetime object")));
    }
    if field == "datestr" {
        let epochs = o.fields["EpochSec"].as_matrix()?.data();
        if epochs.len() == 1 {
            return ok1(Value::String(format_epoch(epochs[0])));
        }
        let items: Vec<Value> = epochs.iter().map(|&e| Value::String(format_epoch(e))).collect();
        return ok1(Value::Cell(CellArray::from_rows(vec![items])?));
    }
    ok1(o.fields[field].clone())
}

/// `categorical(vals)`
pub fn fn_categorical(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("categorical", "1", args.len()));
    }
    let raw_labels: Vec<String> = match &args[0] {
        Value::Cell(c) => c.data.iter().map(|v| format!("{v}")).collect(),
        Value::Matrix(m) => m.data().iter().map(|&x| format!("{x}")).collect(),
        Value::String(s) => vec![s.clone()],
        Value::Object(o) if o.class_name == "string" => match o.fields.get("Data") {
            Some(Value::Cell(c)) => c.data.iter().map(|v| format!("{v}")).collect(),
            _ => Vec::new(),
        },
        other => {
            return Err(MatroError::runtime(format!(
                "categorical: unsupported input type {}",
                other.type_name()
            )));
        }
    };
    let mut cats: Vec<String> = Vec::new();
    for lbl in &raw_labels {
        if !cats.contains(lbl) {
            cats.push(lbl.clone());
        }
    }
    cats.sort();
    let codes: Vec<f64> = raw_labels
        .iter()
        .map(|lbl| (cats.iter().position(|c| c == lbl).unwrap() + 1) as f64)
        .collect();
    let cat_vals: Vec<Value> = cats.into_iter().map(Value::String).collect();
    let cat_cell = if cat_vals.is_empty() {
        CellArray::empty()
    } else {
        CellArray::from_rows(vec![cat_vals])?
    };
    let mut fields = BTreeMap::new();
    fields.insert("Categories".to_string(), Value::Cell(cat_cell));
    fields.insert("Codes".to_string(), Value::matrix(Matrix::col_vec(codes)));
    ok1(Value::Object(ObjectInstance {
        class_name: "categorical".to_string(),
        fields,
    }))
}

pub fn fn_categories(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("categories", "1", args.len()));
    }
    let Value::Object(o) = &args[0] else {
        return Err(MatroError::runtime("categories: expected a categorical array"));
    };
    if o.class_name != "categorical" {
        return Err(MatroError::runtime("categories: expected a categorical array"));
    }
    ok1(o.fields["Categories"].clone())
}

pub fn fn_countcats(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("countcats", "1", args.len()));
    }
    let Value::Object(o) = &args[0] else {
        return Err(MatroError::runtime("countcats: expected a categorical array"));
    };
    if o.class_name != "categorical" {
        return Err(MatroError::runtime("countcats: expected a categorical array"));
    }
    let Value::Cell(cats) = &o.fields["Categories"] else {
        return ok1(Value::empty());
    };
    let codes = o.fields["Codes"].as_matrix()?.data();
    let mut counts = vec![0.0; cats.data.len()];
    for &c in codes {
        let idx = c.round() as usize;
        if idx >= 1 && idx <= counts.len() {
            counts[idx - 1] += 1.0;
        }
    }
    ok1(Value::matrix(Matrix::col_vec(counts)))
}

/// `timetable(times, col1, col2, ..., 'VariableNames', {...})`
pub fn fn_timetable(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() {
        return ok1(Value::Table(TableData::new(vec![], vec![])?));
    }
    let time_col = match &args[0] {
        Value::Object(o) if o.class_name == "datetime" => {
            let epochs = o.fields["EpochSec"].as_matrix()?.data();
            let labels: Vec<Value> = epochs.iter().map(|&e| Value::String(format_epoch(e))).collect();
            let n = labels.len();
            Value::Cell(CellArray {
                rows: n,
                cols: if n == 0 { 0 } else { 1 },
                data: labels,
            })
        }
        Value::Object(o) if o.class_name == "duration" => o.fields["Seconds"].clone(),
        other => other.clone(),
    };
    let mut cols = vec![time_col];
    let mut custom_names: Option<Vec<String>> = None;
    let mut i = 1;
    while i < args.len() {
        if let Ok(s) = args[i].as_str() {
            if s.eq_ignore_ascii_case("VariableNames") && i + 1 < args.len() {
                let Value::Cell(c) = &args[i + 1] else {
                    return Err(MatroError::runtime(
                        "timetable: 'VariableNames' must be a cell array of strings",
                    ));
                };
                let mut names = vec!["Time".to_string()];
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
        None => {
            let mut ns = vec!["Time".to_string()];
            for k in 1..cols.len() {
                ns.push(format!("Var{k}"));
            }
            ns
        }
    };
    ok1(Value::Table(TableData::new(names, cols)?))
}

/// Construct a MATLAB `string` array (`string({'a', 'b'})`, `string([1 2])`, `string('hi')`).
pub fn make_string_array(items: Vec<String>) -> Result<Value> {
    let vals: Vec<Value> = items.into_iter().map(Value::String).collect();
    let cell = if vals.is_empty() {
        CellArray::empty()
    } else {
        CellArray::from_rows(vec![vals])?
    };
    let mut fields = BTreeMap::new();
    fields.insert("Data".to_string(), Value::Cell(cell));
    Ok(Value::Object(ObjectInstance {
        class_name: "string".to_string(),
        fields,
    }))
}

pub fn extract_strings(v: &Value) -> Result<Vec<String>> {
    match v {
        Value::String(s) => Ok(vec![s.clone()]),
        Value::Cell(c) => Ok(c.data.iter().map(|x| format!("{x}")).collect()),
        Value::Matrix(m) => Ok(m.data().iter().map(|&x| format!("{x}")).collect()),
        Value::Object(o) if o.class_name == "string" => match o.fields.get("Data") {
            Some(Value::Cell(c)) => Ok(c.data.iter().map(|x| format!("{x}")).collect()),
            _ => Ok(Vec::new()),
        },
        other => Ok(vec![format!("{other}")]),
    }
}

pub fn fn_string(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() {
        return ok1(make_string_array(vec![String::new()])?);
    }
    if args.len() != 1 {
        return Err(MatroError::arity("string", "0 or 1", args.len()));
    }
    ok1(make_string_array(extract_strings(&args[0])?)?)
}

pub fn fn_strlength(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("strlength", "1", args.len()));
    }
    let strs = extract_strings(&args[0])?;
    let lens: Vec<f64> = strs.iter().map(|s| s.chars().count() as f64).collect();
    let out = if lens.len() == 1 {
        Matrix::scalar(lens[0])
    } else {
        Matrix::row_vec(lens)
    };
    ok1(Value::matrix(out))
}

pub fn fn_split(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("split", "1 or 2", args.len()));
    }
    let strs = extract_strings(&args[0])?;
    let s = strs.first().map(String::as_str).unwrap_or("");
    let parts: Vec<String> = if let Some(d) = args.get(1) {
        let ds = extract_strings(d)?;
        let delim = ds.first().map(String::as_str).unwrap_or(" ");
        s.split(delim).map(ToString::to_string).collect()
    } else {
        s.split_whitespace().map(ToString::to_string).collect()
    };
    ok1(make_string_array(parts)?)
}

pub fn fn_join(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("join", "1 or 2", args.len()));
    }
    let strs = extract_strings(&args[0])?;
    let delim = args.get(1).map(|v| v.as_str()).transpose()?.unwrap_or(" ");
    ok1(Value::String(strs.join(delim)))
}

pub fn fn_starts_ends_with(name: &str, args: &[Value], is_end: bool) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity(name, "2 (str, pat)", args.len()));
    }
    let strs = extract_strings(&args[0])?;
    let pat = args[1].as_str()?;
    let flags: Vec<f64> = strs
        .iter()
        .map(|s| {
            let hit = if is_end { s.ends_with(pat) } else { s.starts_with(pat) };
            if hit { 1.0 } else { 0.0 }
        })
        .collect();
    let out = if flags.len() == 1 {
        Matrix::logical_scalar(flags[0] != 0.0)
    } else {
        Matrix::row_vec(flags).with_logical(true)
    };
    ok1(Value::matrix(out))
}

pub fn fn_erase(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("erase", "2 (str, match)", args.len()));
    }
    let pat = args[1].as_str()?;
    match &args[0] {
        Value::String(s) => ok1(Value::String(s.replace(pat, ""))),
        other => {
            let strs: Vec<String> = extract_strings(other)?
                .into_iter()
                .map(|s| s.replace(pat, ""))
                .collect();
            ok1(make_string_array(strs)?)
        }
    }
}

pub fn fn_strip(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("strip", "1", args.len()));
    }
    match &args[0] {
        Value::String(s) => ok1(Value::String(s.trim().to_string())),
        other => {
            let strs: Vec<String> = extract_strings(other)?
                .into_iter()
                .map(|s| s.trim().to_string())
                .collect();
            ok1(make_string_array(strs)?)
        }
    }
}
