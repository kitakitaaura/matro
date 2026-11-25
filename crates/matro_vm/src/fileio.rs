//! VFS-backed MATLAB File I/O:
//! `fopen`, `fclose`, `fprintf(fid, ...)`, `fgetl`, `fgets`, `fread`, `fwrite`,
//! `feof`, `ftell`, `frewind`, and `textscan`.

use std::collections::HashMap;

use matro_intrinsics::format_printf;
use matro_loader::ModuleLoader;
use matro_types::{CellArray, MatroError, Matrix, Result, Value};

#[derive(Debug, Clone)]
pub struct OpenFile {
    pub path: String,
    pub mode: String,
    pub buf: Vec<u8>,
    pub pos: usize,
    pub dirty: bool,
}

impl OpenFile {
    fn is_writable(&self) -> bool {
        self.mode.contains('w') || self.mode.contains('a') || self.mode.contains('+')
    }

    fn write_slice(&mut self, bytes: &[u8]) {
        if self.mode.starts_with('a') {
            self.pos = self.buf.len();
        }
        let end = self.pos + bytes.len();
        if end > self.buf.len() {
            self.buf.resize(end, 0);
        }
        self.buf[self.pos..end].copy_from_slice(bytes);
        self.pos = end;
        self.dirty = true;
    }
}

#[derive(Debug)]
pub struct FileTable {
    files: HashMap<i32, OpenFile>,
    next_fid: i32,
}

impl Default for FileTable {
    fn default() -> Self {
        Self::new()
    }
}

impl FileTable {
    pub fn new() -> Self {
        Self {
            files: HashMap::new(),
            next_fid: 3,
        }
    }

    pub fn clear(&mut self) {
        self.files.clear();
        self.next_fid = 3;
    }

    fn get(&self, fid: i32) -> Result<&OpenFile> {
        self.files
            .get(&fid)
            .ok_or_else(|| MatroError::runtime(format!("Invalid file identifier {fid}")))
    }

    fn get_mut(&mut self, fid: i32) -> Result<&mut OpenFile> {
        self.files
            .get_mut(&fid)
            .ok_or_else(|| MatroError::runtime(format!("Invalid file identifier {fid}")))
    }

    pub fn fopen(
        &mut self,
        args: &[Value],
        nargout: usize,
        loader: &ModuleLoader,
    ) -> Result<Vec<Value>> {
        if args.is_empty() || args.len() > 2 {
            return Err(MatroError::arity("fopen", "1 or 2", args.len()));
        }
        let path = args[0].as_str()?;
        let raw_mode = if args.len() == 2 {
            args[1].as_str()?.trim()
        } else {
            "r"
        };
        let mode: String = raw_mode
            .chars()
            .filter(|&c| c != 'b' && c != 't')
            .collect();
        let mode = if mode.is_empty() {
            "r".to_string()
        } else {
            mode
        };

        let (resolved_path, buf, pos, dirty) = match mode.as_str() {
            "r" | "r+" => {
                let resolved = loader
                    .resolve_path(path)
                    .unwrap_or_else(|| path.to_string());
                match loader.vfs().read_bytes(&resolved) {
                    Some(bytes) => (resolved, bytes, 0, false),
                    None => {
                        return Ok(if nargout >= 2 {
                            vec![
                                Value::scalar(-1.0),
                                Value::String(format!("No such file or directory: '{path}'")),
                            ]
                        } else {
                            vec![Value::scalar(-1.0)]
                        });
                    }
                }
            }
            "w" | "w+" => (path.to_string(), Vec::new(), 0, true),
            "a" | "a+" => {
                let resolved = loader
                    .resolve_path(path)
                    .unwrap_or_else(|| path.to_string());
                let bytes = loader.vfs().read_bytes(&resolved).unwrap_or_default();
                let pos = bytes.len();
                (resolved, bytes, pos, true)
            }
            other => {
                return Err(MatroError::runtime(format!(
                    "fopen: unsupported permission '{other}'"
                )));
            }
        };

        let fid = self.next_fid;
        self.next_fid += 1;
        self.files.insert(
            fid,
            OpenFile {
                path: resolved_path,
                mode,
                buf,
                pos,
                dirty,
            },
        );
        Ok(if nargout >= 2 {
            vec![Value::scalar(fid as f64), Value::String(String::new())]
        } else {
            vec![Value::scalar(fid as f64)]
        })
    }

    pub fn fclose(&mut self, args: &[Value], loader: &mut ModuleLoader) -> Result<Vec<Value>> {
        if args.len() != 1 {
            return Err(MatroError::arity("fclose", "1", args.len()));
        }
        if let Ok("all") = args[0].as_str() {
            let entries: Vec<OpenFile> = self.files.drain().map(|(_, f)| f).collect();
            for f in entries {
                if f.dirty {
                    loader.mount_bytes(&f.path, &f.buf)?;
                }
            }
            return Ok(vec![Value::scalar(0.0)]);
        }
        let fid = args[0].as_scalar()?.round() as i32;
        if let Some(f) = self.files.remove(&fid) {
            if f.dirty {
                loader.mount_bytes(&f.path, &f.buf)?;
            }
            Ok(vec![Value::scalar(0.0)])
        } else {
            Ok(vec![Value::scalar(-1.0)])
        }
    }

    pub fn fprintf(
        &mut self,
        args: &[Value],
        nargout: usize,
        stdout: &mut String,
    ) -> Result<Vec<Value>> {
        if args.is_empty() {
            return Err(MatroError::arity("fprintf", ">= 1", 0));
        }
        // Check if first argument is a scalar file identifier
        if let Value::Matrix(m) = &args[0] {
            if let Some(fid_f) = m.as_scalar() {
                if args.len() < 2 {
                    return Err(MatroError::arity("fprintf", ">= 2 (fid, fmt, ...)", args.len()));
                }
                let fid = fid_f.round() as i32;
                let formatted = format_printf(&args[1..])?;
                let nbytes = formatted.len();
                if fid == 1 || fid == 2 {
                    stdout.push_str(&formatted);
                } else {
                    let f = self.get_mut(fid)?;
                    if !f.is_writable() {
                        return Err(MatroError::runtime(format!(
                            "fprintf: file {fid} is not open for writing"
                        )));
                    }
                    f.write_slice(formatted.as_bytes());
                }
                return Ok(if nargout > 0 {
                    vec![Value::scalar(nbytes as f64)]
                } else {
                    vec![]
                });
            }
        }
        let formatted = format_printf(args)?;
        let nbytes = formatted.len();
        stdout.push_str(&formatted);
        Ok(if nargout > 0 {
            vec![Value::scalar(nbytes as f64)]
        } else {
            vec![]
        })
    }

    pub fn fgetl(&mut self, args: &[Value], keep_newline: bool) -> Result<Vec<Value>> {
        let fname = if keep_newline { "fgets" } else { "fgetl" };
        if args.len() != 1 {
            return Err(MatroError::arity(fname, "1", args.len()));
        }
        let fid = args[0].as_scalar()?.round() as i32;
        let f = self.get_mut(fid)?;
        if f.pos >= f.buf.len() {
            return Ok(vec![Value::scalar(-1.0)]);
        }
        let start = f.pos;
        let mut end = start;
        while end < f.buf.len() && f.buf[end] != b'\n' {
            end += 1;
        }
        let next_pos = if end < f.buf.len() && f.buf[end] == b'\n' {
            end + 1
        } else {
            end
        };
        f.pos = next_pos;

        let slice = if keep_newline {
            &f.buf[start..next_pos]
        } else {
            let mut line_end = end;
            if line_end > start && f.buf[line_end - 1] == b'\r' {
                line_end -= 1;
            }
            &f.buf[start..line_end]
        };
        Ok(vec![Value::String(
            String::from_utf8_lossy(slice).into_owned(),
        )])
    }

    pub fn feof(&self, args: &[Value]) -> Result<Vec<Value>> {
        if args.len() != 1 {
            return Err(MatroError::arity("feof", "1", args.len()));
        }
        let fid = args[0].as_scalar()?.round() as i32;
        let f = self.get(fid)?;
        Ok(vec![Value::bool(f.pos >= f.buf.len())])
    }

    pub fn ftell(&self, args: &[Value]) -> Result<Vec<Value>> {
        if args.len() != 1 {
            return Err(MatroError::arity("ftell", "1", args.len()));
        }
        let fid = args[0].as_scalar()?.round() as i32;
        let f = self.get(fid)?;
        Ok(vec![Value::scalar(f.pos as f64)])
    }

    pub fn frewind(&mut self, args: &[Value]) -> Result<Vec<Value>> {
        if args.len() != 1 {
            return Err(MatroError::arity("frewind", "1", args.len()));
        }
        let fid = args[0].as_scalar()?.round() as i32;
        let f = self.get_mut(fid)?;
        f.pos = 0;
        Ok(vec![])
    }

    pub fn fseek(&mut self, args: &[Value]) -> Result<Vec<Value>> {
        if args.len() != 3 {
            return Err(MatroError::arity("fseek", "3 (fid, offset, origin)", args.len()));
        }
        let fid = args[0].as_scalar()?.round() as i32;
        let offset = args[1].as_scalar()?.round() as isize;
        let origin_code = if let Ok(s) = args[2].as_str() {
            match s.to_ascii_lowercase().as_str() {
                "bof" => -1,
                "cof" => 0,
                "eof" => 1,
                _ => return Err(MatroError::runtime(format!("fseek: unknown origin '{s}'"))),
            }
        } else {
            args[2].as_scalar()?.round() as i32
        };
        let f = self.get_mut(fid)?;
        let base = match origin_code {
            -1 => 0isize,
            0 => f.pos as isize,
            1 => f.buf.len() as isize,
            _ => return Ok(vec![Value::scalar(-1.0)]),
        };
        let target = base + offset;
        if target < 0 || (target as usize) > f.buf.len() {
            Ok(vec![Value::scalar(-1.0)])
        } else {
            f.pos = target as usize;
            Ok(vec![Value::scalar(0.0)])
        }
    }

    pub fn ferror(&self, args: &[Value], nargout: usize) -> Result<Vec<Value>> {
        if args.is_empty() || args.len() > 2 {
            return Err(MatroError::arity("ferror", "1 or 2", args.len()));
        }
        if nargout >= 2 {
            Ok(vec![Value::String(String::new()), Value::scalar(0.0)])
        } else {
            Ok(vec![Value::String(String::new())])
        }
    }

    pub fn fscanf(&mut self, args: &[Value], nargout: usize) -> Result<Vec<Value>> {
        if args.len() < 2 || args.len() > 3 {
            return Err(MatroError::arity("fscanf", "2 or 3 (fid, fmt [, sizeA])", args.len()));
        }
        let fid = args[0].as_scalar()?.round() as i32;
        let fmt = args[1].as_str()?.to_string();
        let f = self.get_mut(fid)?;
        let remaining = String::from_utf8_lossy(&f.buf[f.pos..]).into_owned();
        let (mat, count, consumed) = scan_formatted_text(&remaining, &fmt, args.get(2))?;
        f.pos = (f.pos + consumed).min(f.buf.len());
        if nargout >= 2 {
            Ok(vec![mat, Value::scalar(count as f64)])
        } else {
            Ok(vec![mat])
        }
    }

    pub fn sscanf(&self, args: &[Value], nargout: usize) -> Result<Vec<Value>> {
        if args.len() < 2 || args.len() > 3 {
            return Err(MatroError::arity("sscanf", "2 or 3 (str, fmt [, sizeA])", args.len()));
        }
        let text = args[0].as_str()?;
        let fmt = args[1].as_str()?;
        let (mat, count, consumed) = scan_formatted_text(text, fmt, args.get(2))?;
        let mut out = vec![mat];
        if nargout >= 2 {
            out.push(Value::scalar(count as f64));
        }
        if nargout >= 3 {
            out.push(Value::String(String::new()));
        }
        if nargout >= 4 {
            out.push(Value::scalar((consumed + 1) as f64));
        }
        Ok(out)
    }

    pub fn fwrite(&mut self, args: &[Value]) -> Result<Vec<Value>> {
        if args.len() < 2 || args.len() > 3 {
            return Err(MatroError::arity("fwrite", "2 or 3", args.len()));
        }
        let fid = args[0].as_scalar()?.round() as i32;
        let prec_raw = if args.len() == 3 {
            args[2].as_str()?
        } else {
            "uint8"
        };
        let prec = normalize_precision(prec_raw);

        let mut bytes = Vec::new();
        let count: usize = match &args[1] {
            Value::String(s) => {
                for &b in s.as_bytes() {
                    encode_elem(&mut bytes, b as f64, prec)?;
                }
                s.len()
            }
            other => {
                let m = other.as_matrix()?;
                for &v in m.data() {
                    encode_elem(&mut bytes, v, prec)?;
                }
                m.len()
            }
        };

        let f = self.get_mut(fid)?;
        if !f.is_writable() {
            return Err(MatroError::runtime(format!(
                "fwrite: file {fid} is not open for writing"
            )));
        }
        f.write_slice(&bytes);
        Ok(vec![Value::scalar(count as f64)])
    }

    pub fn fread(&mut self, args: &[Value], nargout: usize) -> Result<Vec<Value>> {
        if args.is_empty() || args.len() > 3 {
            return Err(MatroError::arity("fread", "1 to 3", args.len()));
        }
        let fid = args[0].as_scalar()?.round() as i32;

        // Parse optional sizeA and precision (handles `fread(fid, '*char')` as well as `fread(fid, n, prec)`)
        let (size_spec, prec_raw) = match args.len() {
            1 => (None, "uint8"),
            2 => {
                if let Ok(p) = args[1].as_str() {
                    (None, p)
                } else {
                    (Some(args[1].as_matrix()?.clone()), "uint8")
                }
            }
            _ => (Some(args[1].as_matrix()?.clone()), args[2].as_str()?),
        };

        let as_char_str = prec_raw == "*char" || prec_raw == "char=>char";
        let prec = normalize_precision(prec_raw);
        let elem_bytes = precision_size(prec)?;

        let (max_elems, shape): (usize, Option<(usize, Option<usize>)>) = match &size_spec {
            None => (usize::MAX, None),
            Some(m) if m.len() == 1 => {
                let v = m.data()[0];
                if v.is_infinite() && v > 0.0 {
                    (usize::MAX, None)
                } else {
                    (v.max(0.0).round() as usize, None)
                }
            }
            Some(m) if m.len() >= 2 => {
                let r = m.data()[0].max(0.0).round() as usize;
                let c_raw = m.data()[1];
                if c_raw.is_infinite() && c_raw > 0.0 {
                    (usize::MAX, Some((r, None)))
                } else {
                    let c = c_raw.max(0.0).round() as usize;
                    (r.saturating_mul(c), Some((r, Some(c))))
                }
            }
            Some(_) => (0, None),
        };

        let f = self.get_mut(fid)?;
        let avail = f.buf.len().saturating_sub(f.pos) / elem_bytes;
        let to_read = max_elems.min(avail);
        let mut vals = Vec::with_capacity(to_read);
        for _ in 0..to_read {
            let chunk = &f.buf[f.pos..f.pos + elem_bytes];
            vals.push(decode_elem(chunk, prec));
            f.pos += elem_bytes;
        }

        let result_val = if as_char_str {
            let bytes: Vec<u8> = vals.iter().map(|&v| v as u8).collect();
            Value::String(String::from_utf8_lossy(&bytes).into_owned())
        } else if let Some((rows, cols_opt)) = shape {
            if rows == 0 || vals.is_empty() {
                Value::matrix(Matrix::zeros(0, 0))
            } else {
                let cols = cols_opt.unwrap_or_else(|| vals.len().div_ceil(rows));
                let mut padded = vals;
                padded.resize(rows * cols, 0.0);
                Value::matrix(Matrix::from_col_major(rows, cols, padded)?)
            }
        } else {
            Value::matrix(Matrix::col_vec(vals))
        };

        Ok(if nargout >= 2 {
            vec![result_val, Value::scalar(to_read as f64)]
        } else {
            vec![result_val]
        })
    }

    pub fn textscan(&mut self, args: &[Value]) -> Result<Vec<Value>> {
        if args.len() < 2 {
            return Err(MatroError::arity("textscan", ">= 2 (src, fmt)", args.len()));
        }
        let text = match &args[0] {
            Value::String(s) => s.clone(),
            other => {
                let fid = other.as_scalar()?.round() as i32;
                let f = self.get_mut(fid)?;
                let slice = &f.buf[f.pos..];
                let s = String::from_utf8_lossy(slice).into_owned();
                f.pos = f.buf.len();
                s
            }
        };
        let fmt = args[1].as_str()?;

        let mut delimiter: Option<char> = None;
        let mut header_lines: usize = 0;
        let mut i = 2;
        while i + 1 < args.len() {
            let key = args[i].as_str()?.to_ascii_lowercase();
            match key.as_str() {
                "delimiter" => {
                    delimiter = args[i + 1].as_str()?.chars().next();
                }
                "headerlines" => {
                    header_lines = args[i + 1].as_scalar()?.max(0.0).round() as usize;
                }
                _ => {}
            }
            i += 2;
        }

        #[derive(Clone, Copy)]
        enum ColSpec {
            Float,
            Str,
            Skip,
        }

        let mut specs = Vec::new();
        let fchars: Vec<char> = fmt.chars().collect();
        let mut fi = 0;
        while fi < fchars.len() {
            if fchars[fi] == '%' {
                fi += 1;
                let mut skip = false;
                if fi < fchars.len() && fchars[fi] == '*' {
                    skip = true;
                    fi += 1;
                }
                while fi < fchars.len() && (fchars[fi].is_ascii_digit() || fchars[fi] == '.') {
                    fi += 1;
                }
                if fi < fchars.len() {
                    let kind = match fchars[fi] {
                        's' | 'q' | 'c' => {
                            if skip {
                                ColSpec::Skip
                            } else {
                                ColSpec::Str
                            }
                        }
                        'f' | 'd' | 'i' | 'u' | 'n' | 'g' | 'e' => {
                            if skip {
                                ColSpec::Skip
                            } else {
                                ColSpec::Float
                            }
                        }
                        _ => ColSpec::Skip,
                    };
                    specs.push(kind);
                    fi += 1;
                }
            } else {
                fi += 1;
            }
        }

        enum ColBuf {
            Float(Vec<f64>),
            Str(Vec<Value>),
        }

        let mut cols: Vec<ColBuf> = specs
            .iter()
            .filter_map(|&s| match s {
                ColSpec::Float => Some(ColBuf::Float(Vec::new())),
                ColSpec::Str => Some(ColBuf::Str(Vec::new())),
                ColSpec::Skip => None,
            })
            .collect();

        for line in text.lines().skip(header_lines) {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let tokens: Vec<&str> = if let Some(d) = delimiter {
                trimmed.split(d).map(str::trim).collect()
            } else {
                trimmed.split_whitespace().collect()
            };
            if tokens.is_empty() {
                continue;
            }
            let mut tok_idx = 0usize;
            let mut col_idx = 0usize;
            for &spec in &specs {
                let tok = tokens.get(tok_idx).copied().unwrap_or("");
                tok_idx += 1;
                match spec {
                    ColSpec::Skip => {}
                    ColSpec::Float => {
                        let v = tok.parse::<f64>().unwrap_or(f64::NAN);
                        if let ColBuf::Float(vec) = &mut cols[col_idx] {
                            vec.push(v);
                        }
                        col_idx += 1;
                    }
                    ColSpec::Str => {
                        if let ColBuf::Str(vec) = &mut cols[col_idx] {
                            vec.push(Value::String(tok.to_string()));
                        }
                        col_idx += 1;
                    }
                }
            }
        }

        let cell_items: Vec<Value> = cols
            .into_iter()
            .map(|c| match c {
                ColBuf::Float(v) => Value::matrix(Matrix::col_vec(v)),
                ColBuf::Str(v) => {
                    let rows = v.len();
                    Value::Cell(CellArray {
                        rows,
                        cols: if rows == 0 { 0 } else { 1 },
                        data: v,
                    })
                }
            })
            .collect();

        Ok(vec![Value::Cell(CellArray::from_rows(vec![cell_items])?)])
    }
}

fn normalize_precision(raw: &str) -> &str {
    let s = raw.trim().trim_start_matches('*');
    if let Some((lhs, _)) = s.split_once("=>") {
        lhs.trim()
    } else {
        s
    }
}

fn precision_size(prec: &str) -> Result<usize> {
    match prec {
        "uint8" | "uchar" | "char" | "int8" | "schar" => Ok(1),
        "uint16" | "int16" | "short" => Ok(2),
        "uint32" | "int32" | "single" | "float32" | "float" => Ok(4),
        "double" | "float64" | "int64" | "uint64" => Ok(8),
        other => Err(MatroError::runtime(format!(
            "Unsupported fread/fwrite precision '{other}'"
        ))),
    }
}

fn encode_elem(out: &mut Vec<u8>, v: f64, prec: &str) -> Result<()> {
    match prec {
        "uint8" | "uchar" | "char" => out.push(v.clamp(0.0, 255.0).round() as u8),
        "int8" | "schar" => out.push((v.clamp(-128.0, 127.0).round() as i8) as u8),
        "uint16" => out.extend_from_slice(&(v.clamp(0.0, 65535.0).round() as u16).to_le_bytes()),
        "int16" | "short" => {
            out.extend_from_slice(&(v.clamp(-32768.0, 32767.0).round() as i16).to_le_bytes())
        }
        "uint32" => out.extend_from_slice(&(v.max(0.0).round() as u32).to_le_bytes()),
        "int32" => out.extend_from_slice(&(v.round() as i32).to_le_bytes()),
        "single" | "float32" | "float" => out.extend_from_slice(&(v as f32).to_le_bytes()),
        "double" | "float64" => out.extend_from_slice(&v.to_le_bytes()),
        "int64" => out.extend_from_slice(&(v.round() as i64).to_le_bytes()),
        "uint64" => out.extend_from_slice(&(v.max(0.0).round() as u64).to_le_bytes()),
        other => {
            return Err(MatroError::runtime(format!(
                "Unsupported fwrite precision '{other}'"
            )));
        }
    }
    Ok(())
}

fn decode_elem(bytes: &[u8], prec: &str) -> f64 {
    match prec {
        "uint8" | "uchar" | "char" => bytes[0] as f64,
        "int8" | "schar" => (bytes[0] as i8) as f64,
        "uint16" => u16::from_le_bytes([bytes[0], bytes[1]]) as f64,
        "int16" | "short" => i16::from_le_bytes([bytes[0], bytes[1]]) as f64,
        "uint32" => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64,
        "int32" => i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64,
        "single" | "float32" | "float" => {
            f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64
        }
        "double" | "float64" => f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]),
        "int64" => i64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]) as f64,
        "uint64" => u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]) as f64,
        _ => 0.0,
    }
}

#[derive(Clone, Debug)]
enum ScanToken {
    Whitespace,
    Literal(char),
    Spec {
        suppress: bool,
        width: Option<usize>,
        conv: char,
    },
}

fn parse_scan_format(fmt: &str) -> Vec<ScanToken> {
    let chars: Vec<char> = fmt.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if ch == '\\' && i + 1 < chars.len() {
            match chars[i + 1] {
                'n' | 'r' | 't' => {
                    out.push(ScanToken::Whitespace);
                    i += 2;
                    continue;
                }
                '\\' => {
                    out.push(ScanToken::Literal('\\'));
                    i += 2;
                    continue;
                }
                _ => {}
            }
        }
        if ch.is_whitespace() {
            while i < chars.len() && chars[i].is_whitespace() {
                i += 1;
            }
            out.push(ScanToken::Whitespace);
        } else if ch == '%' {
            i += 1;
            if i < chars.len() && chars[i] == '%' {
                out.push(ScanToken::Literal('%'));
                i += 1;
                continue;
            }
            let mut suppress = false;
            if i < chars.len() && chars[i] == '*' {
                suppress = true;
                i += 1;
            }
            let w_start = i;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            let width = if i > w_start {
                chars[w_start..i].iter().collect::<String>().parse::<usize>().ok()
            } else {
                None
            };
            // Skip length modifiers like l, h, ll
            while i < chars.len() && matches!(chars[i], 'l' | 'h' | 'z' | 't') {
                i += 1;
            }
            if i < chars.len() {
                let conv = chars[i];
                i += 1;
                out.push(ScanToken::Spec { suppress, width, conv });
            }
        } else {
            out.push(ScanToken::Literal(ch));
            i += 1;
        }
    }
    out
}

fn scan_formatted_text(
    text: &str,
    fmt: &str,
    size_arg: Option<&Value>,
) -> Result<(Value, usize, usize)> {
    let tokens = parse_scan_format(fmt);
    if tokens.is_empty() {
        return Ok((Value::matrix(Matrix::zeros(0, 0)), 0, 0));
    }

    let (max_items, shape): (usize, Option<(usize, Option<usize>)>) = match size_arg {
        None => (usize::MAX, None),
        Some(v) => {
            let m = v.to_dense_matrix()?;
            if m.is_empty() {
                (0, None)
            } else if m.len() == 1 {
                let n = m.data()[0];
                if n.is_infinite() && n > 0.0 {
                    (usize::MAX, None)
                } else {
                    (n.max(0.0).round() as usize, None)
                }
            } else {
                let r = m.data()[0].max(0.0).round() as usize;
                let c_raw = m.data()[1];
                if c_raw.is_infinite() && c_raw > 0.0 {
                    (usize::MAX, Some((r, None)))
                } else {
                    let c = c_raw.max(0.0).round() as usize;
                    (r.saturating_mul(c), Some((r, Some(c))))
                }
            }
        }
    };

    let all_char_specs = tokens.iter().all(|t| match t {
        ScanToken::Spec { suppress: false, conv, .. } => matches!(conv, 's' | 'c'),
        _ => true,
    }) && tokens
        .iter()
        .any(|t| matches!(t, ScanToken::Spec { suppress: false, .. }));

    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut pos = 0usize;
    let mut nums: Vec<f64> = Vec::new();
    let mut out_str = String::new();
    let mut count = 0usize;

    'outer: while pos < len && count < max_items {
        let iter_start = pos;
        let mut matched_any = false;
        for tok in &tokens {
            if count >= max_items {
                break 'outer;
            }
            match tok {
                ScanToken::Whitespace => {
                    while pos < len && (bytes[pos] as char).is_whitespace() {
                        pos += 1;
                    }
                }
                ScanToken::Literal(expected) => {
                    let rem = &text[pos..];
                    if let Some(ch) = rem.chars().next() {
                        if ch == *expected {
                            pos += ch.len_utf8();
                        } else {
                            break 'outer;
                        }
                    } else {
                        break 'outer;
                    }
                }
                ScanToken::Spec { suppress, width, conv } => {
                    let max_w = width.unwrap_or(usize::MAX);
                    match conv {
                        'c' => {
                            let w = width.unwrap_or(1);
                            let mut taken = String::new();
                            for _ in 0..w {
                                if pos >= len {
                                    break;
                                }
                                let ch = text[pos..].chars().next().unwrap();
                                pos += ch.len_utf8();
                                taken.push(ch);
                            }
                            if taken.is_empty() {
                                break 'outer;
                            }
                            matched_any = true;
                            if !*suppress {
                                if all_char_specs {
                                    out_str.push_str(&taken);
                                    count += taken.chars().count();
                                } else {
                                    for ch in taken.chars() {
                                        nums.push(ch as u32 as f64);
                                        count += 1;
                                    }
                                }
                            }
                        }
                        's' => {
                            while pos < len && (bytes[pos] as char).is_whitespace() {
                                pos += 1;
                            }
                            let mut taken = String::new();
                            while pos < len && taken.chars().count() < max_w {
                                let ch = text[pos..].chars().next().unwrap();
                                if ch.is_whitespace() {
                                    break;
                                }
                                pos += ch.len_utf8();
                                taken.push(ch);
                            }
                            if taken.is_empty() {
                                break 'outer;
                            }
                            matched_any = true;
                            if !*suppress {
                                if all_char_specs {
                                    out_str.push_str(&taken);
                                    count += 1;
                                } else {
                                    for ch in taken.chars() {
                                        nums.push(ch as u32 as f64);
                                    }
                                    count += 1;
                                }
                            }
                        }
                        'd' | 'i' | 'u' | 'x' | 'X' | 'o' => {
                            while pos < len && (bytes[pos] as char).is_whitespace() {
                                pos += 1;
                            }
                            if pos >= len {
                                break 'outer;
                            }
                            let start = pos;
                            let limit = len.min(start.saturating_add(max_w));
                            if pos < limit && matches!(bytes[pos], b'+' | b'-') && *conv != 'u' {
                                pos += 1;
                            }
                            let radix: u32 = match conv {
                                'x' | 'X' => {
                                    if pos + 1 < limit
                                        && bytes[pos] == b'0'
                                        && matches!(bytes[pos + 1], b'x' | b'X')
                                    {
                                        pos += 2;
                                    }
                                    16
                                }
                                'o' => 8,
                                'i' => {
                                    if pos + 1 < limit
                                        && bytes[pos] == b'0'
                                        && matches!(bytes[pos + 1], b'x' | b'X')
                                    {
                                        pos += 2;
                                        16
                                    } else {
                                        10
                                    }
                                }
                                _ => 10,
                            };
                            let d_start = pos;
                            while pos < limit && (bytes[pos] as char).is_digit(radix) {
                                pos += 1;
                            }
                            if pos == d_start {
                                pos = start;
                                break 'outer;
                            }
                            let neg = bytes[start] == b'-';
                            let digits = &text[d_start..pos];
                            let parsed = if radix == 10 {
                                text[start..pos].parse::<f64>().ok()
                            } else {
                                i64::from_str_radix(digits, radix)
                                    .ok()
                                    .map(|v| if neg { -(v as f64) } else { v as f64 })
                            };
                            let Some(val) = parsed else {
                                pos = start;
                                break 'outer;
                            };
                            matched_any = true;
                            if !*suppress {
                                nums.push(val);
                                count += 1;
                            }
                        }
                        'f' | 'e' | 'E' | 'g' | 'G' => {
                            while pos < len && (bytes[pos] as char).is_whitespace() {
                                pos += 1;
                            }
                            if pos >= len {
                                break 'outer;
                            }
                            let start = pos;
                            let limit = len.min(start.saturating_add(max_w));
                            if pos < limit && matches!(bytes[pos], b'+' | b'-') {
                                pos += 1;
                            }
                            // Check Inf / NaN
                            let rem_low = text[pos..limit].to_ascii_lowercase();
                            if rem_low.starts_with("inf") {
                                pos += 3;
                                let val = if bytes[start] == b'-' {
                                    f64::NEG_INFINITY
                                } else {
                                    f64::INFINITY
                                };
                                matched_any = true;
                                if !*suppress {
                                    nums.push(val);
                                    count += 1;
                                }
                                continue;
                            } else if rem_low.starts_with("nan") {
                                pos += 3;
                                matched_any = true;
                                if !*suppress {
                                    nums.push(f64::NAN);
                                    count += 1;
                                }
                                continue;
                            }
                            let mut has_digit = false;
                            while pos < limit && bytes[pos].is_ascii_digit() {
                                has_digit = true;
                                pos += 1;
                            }
                            if pos < limit && bytes[pos] == b'.' {
                                pos += 1;
                                while pos < limit && bytes[pos].is_ascii_digit() {
                                    has_digit = true;
                                    pos += 1;
                                }
                            }
                            if !has_digit {
                                pos = start;
                                break 'outer;
                            }
                            if pos < limit && matches!(bytes[pos], b'e' | b'E') {
                                let exp_pos = pos;
                                pos += 1;
                                if pos < limit && matches!(bytes[pos], b'+' | b'-') {
                                    pos += 1;
                                }
                                let ed_start = pos;
                                while pos < limit && bytes[pos].is_ascii_digit() {
                                    pos += 1;
                                }
                                if pos == ed_start {
                                    pos = exp_pos;
                                }
                            }
                            let Ok(val) = text[start..pos].parse::<f64>() else {
                                pos = start;
                                break 'outer;
                            };
                            matched_any = true;
                            if !*suppress {
                                nums.push(val);
                                count += 1;
                            }
                        }
                        _ => break 'outer,
                    }
                }
            }
        }
        if !matched_any || pos == iter_start {
            break;
        }
    }

    if all_char_specs {
        return Ok((Value::String(out_str), count, pos));
    }

    let val = if let Some((rows, cols_opt)) = shape {
        if rows == 0 || nums.is_empty() {
            Value::matrix(Matrix::zeros(0, 0))
        } else {
            let cols = cols_opt.unwrap_or_else(|| nums.len().div_ceil(rows));
            let mut padded = nums;
            padded.resize(rows * cols, 0.0);
            Value::matrix(Matrix::from_col_major(rows, cols, padded)?)
        }
    } else if nums.is_empty() {
        Value::matrix(Matrix::zeros(0, 1))
    } else {
        Value::matrix(Matrix::col_vec(nums))
    };

    Ok((val, count, pos))
}
