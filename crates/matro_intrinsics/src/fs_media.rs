//! Handle graphics (`gcf`, `gca`, `set`, `get`), VFS/OS filesystem utilities
//! (`fullfile`, `fileparts`, `isfile`, `isfolder`, `dir`, `ls`, `pwd`, `cd`,
//! `mkdir`, `delete`, `copyfile`, `movefile`), and zero-dependency pure-Rust
//! media I/O (`imread`/`imwrite` BMP + PNG, `audioread`/`audiowrite` 16-bit PCM WAV).

use std::collections::BTreeMap;

use matro_loader::ModuleLoader;
use matro_loader::inflate::{inflate_zlib, zlib_wrap_stored};
use matro_types::{
    MatroError, Matrix, NumericClass, ObjectInstance, PlotFigure, Result, StructArray, Value,
};

fn ok1(v: Value) -> Result<Vec<Value>> {
    Ok(vec![v])
}

// ---------------------------------------------------------------------------
// Handle Graphics (`gcf`, `gca`, `set`, `get`)
// ---------------------------------------------------------------------------

pub fn make_line_handle(panel_idx: usize, series_idx: usize) -> Value {
    let mut fields = BTreeMap::new();
    fields.insert("__panel".to_string(), Value::scalar(panel_idx as f64));
    fields.insert("__series".to_string(), Value::scalar(series_idx as f64));
    Value::Object(ObjectInstance {
        class_name: "Line".to_string(),
        fields,
    })
}

pub fn fn_gcf(_args: &[Value]) -> Result<Vec<Value>> {
    let mut fields = BTreeMap::new();
    fields.insert("Number".to_string(), Value::scalar(1.0));
    ok1(Value::Object(ObjectInstance {
        class_name: "Figure".to_string(),
        fields,
    }))
}

pub fn fn_gca(_args: &[Value], fig: &PlotFigure) -> Result<Vec<Value>> {
    let mut fields = BTreeMap::new();
    fields.insert("__panel".to_string(), Value::scalar(fig.active as f64));
    ok1(Value::Object(ObjectInstance {
        class_name: "Axes".to_string(),
        fields,
    }))
}

pub fn parse_color_val(val: &Value) -> Result<String> {
    if let Ok(s) = val.as_str() {
        let lower = s.trim().to_ascii_lowercase();
        let hex = match lower.as_str() {
            "r" | "red" => "#d95319",
            "g" | "green" => "#77ac30",
            "b" | "blue" => "#0072bd",
            "k" | "black" => "#111827",
            "m" | "magenta" => "#7e2f8e",
            "c" | "cyan" => "#4dbeee",
            "y" | "yellow" => "#edb120",
            "w" | "white" => "#ffffff",
            _ if s.starts_with('#') => s,
            _ => s,
        };
        return Ok(hex.to_string());
    }
    let m = val.to_dense_matrix()?;
    if m.len() == 3 {
        let r = (m.data()[0].clamp(0.0, 1.0) * 255.0).round() as u8;
        let g = (m.data()[1].clamp(0.0, 1.0) * 255.0).round() as u8;
        let b = (m.data()[2].clamp(0.0, 1.0) * 255.0).round() as u8;
        return Ok(format!("#{r:02x}{g:02x}{b:02x}"));
    }
    Err(MatroError::runtime(
        "Color must be a color name, hex string, or 1x3 RGB triple in [0, 1]",
    ))
}

pub fn fn_set(args: &[Value], fig: &mut PlotFigure) -> Result<Vec<Value>> {
    if args.len() < 3 || (args.len() - 1) % 2 != 0 {
        return Err(MatroError::arity(
            "set",
            "h, 'Prop1', val1, ...",
            args.len(),
        ));
    }
    let Value::Object(o) = &args[0] else {
        return Err(MatroError::runtime("set: first argument must be a graphics handle"));
    };
    let panel_idx = o
        .fields
        .get("__panel")
        .and_then(|v| v.as_scalar().ok())
        .unwrap_or(fig.active as f64) as usize;
    let panel = fig
        .panels
        .get_mut(panel_idx)
        .ok_or_else(|| MatroError::runtime("set: invalid axes handle"))?;

    let mut i = 1;
    while i + 1 < args.len() {
        let prop = args[i].as_str()?.to_ascii_lowercase();
        let val = &args[i + 1];
        match o.class_name.as_str() {
            "Line" => {
                let s_idx = o
                    .fields
                    .get("__series")
                    .and_then(|v| v.as_scalar().ok())
                    .unwrap_or(0.0) as usize;
                let ser = panel
                    .series
                    .get_mut(s_idx)
                    .ok_or_else(|| MatroError::runtime("set: invalid Line handle"))?;
                match prop.as_str() {
                    "color" => ser.color = parse_color_val(val)?,
                    "linewidth" => ser.line_width = val.as_scalar()?.max(0.2),
                    "linestyle" => {
                        let ls = val.as_str()?;
                        ser.dashed = ls.contains("--") || ls.contains(':');
                    }
                    "displayname" => ser.label = Some(val.as_str()?.to_string()),
                    "xdata" => ser.x = val.to_dense_matrix()?.data().to_vec(),
                    "ydata" => ser.y = val.to_dense_matrix()?.data().to_vec(),
                    _ => {
                        return Err(MatroError::runtime(format!(
                            "set: unknown Line property '{}'",
                            args[i].as_str().unwrap_or("")
                        )));
                    }
                }
            }
            "Axes" => match prop.as_str() {
                "xlim" => {
                    let m = val.to_dense_matrix()?;
                    if m.len() != 2 {
                        return Err(MatroError::dimension("XLim must be [min, max]"));
                    }
                    panel.xlim = Some((m.data()[0], m.data()[1]));
                }
                "ylim" => {
                    let m = val.to_dense_matrix()?;
                    if m.len() != 2 {
                        return Err(MatroError::dimension("YLim must be [min, max]"));
                    }
                    panel.ylim = Some((m.data()[0], m.data()[1]));
                }
                "zlim" => {
                    let m = val.to_dense_matrix()?;
                    if m.len() != 2 {
                        return Err(MatroError::dimension("ZLim must be [min, max]"));
                    }
                    panel.zlim = Some((m.data()[0], m.data()[1]));
                }
                "title" => panel.title = val.as_str()?.to_string(),
                "xlabel" => panel.xlabel = val.as_str()?.to_string(),
                "ylabel" => panel.ylabel = val.as_str()?.to_string(),
                "zlabel" => panel.zlabel = val.as_str()?.to_string(),
                "xscale" => panel.log_x = val.as_str()?.eq_ignore_ascii_case("log"),
                "yscale" => panel.log_y = val.as_str()?.eq_ignore_ascii_case("log"),
                "xgrid" | "ygrid" | "grid" => {
                    panel.grid = val.as_str()?.eq_ignore_ascii_case("on");
                }
                _ => {
                    return Err(MatroError::runtime(format!(
                        "set: unknown Axes property '{}'",
                        args[i].as_str().unwrap_or("")
                    )));
                }
            },
            _ => {}
        }
        i += 2;
    }
    Ok(vec![])
}

pub fn fn_get(args: &[Value], fig: &PlotFigure) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("get", "2 (h, prop)", args.len()));
    }
    let Value::Object(o) = &args[0] else {
        return Err(MatroError::runtime("get: first argument must be a graphics handle"));
    };
    let prop = args[1].as_str()?.to_ascii_lowercase();
    let panel_idx = o
        .fields
        .get("__panel")
        .and_then(|v| v.as_scalar().ok())
        .unwrap_or(fig.active as f64) as usize;
    let panel = fig
        .panels
        .get(panel_idx)
        .ok_or_else(|| MatroError::runtime("get: invalid axes handle"))?;

    match o.class_name.as_str() {
        "Line" => {
            let s_idx = o
                .fields
                .get("__series")
                .and_then(|v| v.as_scalar().ok())
                .unwrap_or(0.0) as usize;
            let ser = panel
                .series
                .get(s_idx)
                .ok_or_else(|| MatroError::runtime("get: invalid Line handle"))?;
            match prop.as_str() {
                "color" => ok1(Value::String(ser.color.clone())),
                "linewidth" => ok1(Value::scalar(ser.line_width)),
                "linestyle" => ok1(Value::String(if ser.dashed { "--" } else { "-" }.to_string())),
                "displayname" => ok1(Value::String(ser.label.clone().unwrap_or_default())),
                "xdata" => ok1(Value::matrix(Matrix::row_vec(ser.x.clone()))),
                "ydata" => ok1(Value::matrix(Matrix::row_vec(ser.y.clone()))),
                "type" => ok1(Value::String("line".to_string())),
                _ => Err(MatroError::runtime(format!("get: unknown Line property '{prop}'"))),
            }
        }
        "Axes" => match prop.as_str() {
            "xlim" => {
                let (a, b) = panel.xlim.unwrap_or((0.0, 1.0));
                ok1(Value::matrix(Matrix::row_vec(vec![a, b])))
            }
            "ylim" => {
                let (a, b) = panel.ylim.unwrap_or((0.0, 1.0));
                ok1(Value::matrix(Matrix::row_vec(vec![a, b])))
            }
            "zlim" => {
                let (a, b) = panel.zlim.unwrap_or((0.0, 1.0));
                ok1(Value::matrix(Matrix::row_vec(vec![a, b])))
            }
            "title" => ok1(Value::String(panel.title.clone())),
            "xlabel" => ok1(Value::String(panel.xlabel.clone())),
            "ylabel" => ok1(Value::String(panel.ylabel.clone())),
            "zlabel" => ok1(Value::String(panel.zlabel.clone())),
            "xscale" => ok1(Value::String(if panel.log_x { "log" } else { "linear" }.to_string())),
            "yscale" => ok1(Value::String(if panel.log_y { "log" } else { "linear" }.to_string())),
            "xgrid" | "ygrid" => ok1(Value::String(if panel.grid { "on" } else { "off" }.to_string())),
            "type" => ok1(Value::String("axes".to_string())),
            _ => Err(MatroError::runtime(format!("get: unknown Axes property '{prop}'"))),
        },
        "Figure" => match prop.as_str() {
            "number" => ok1(Value::scalar(1.0)),
            "type" => ok1(Value::String("figure".to_string())),
            _ => Err(MatroError::runtime(format!("get: unknown Figure property '{prop}'"))),
        },
        _ => Err(MatroError::runtime("get: unsupported handle type")),
    }
}

// ---------------------------------------------------------------------------
// Filesystem Utilities (`fullfile`, `fileparts`, `isfile`, `isfolder`, `dir`, ...)
// ---------------------------------------------------------------------------

pub fn fn_fullfile(args: &[Value]) -> Result<Vec<Value>> {
    let mut parts = Vec::new();
    for a in args {
        let s = a.as_str()?.replace('\\', "/");
        if s.is_empty() {
            continue;
        }
        if parts.is_empty() {
            parts.push(s.trim_end_matches('/').to_string());
        } else {
            parts.push(s.trim_matches('/').to_string());
        }
    }
    ok1(Value::String(parts.join("/")))
}

pub fn fn_fileparts(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("fileparts", "1 (path)", args.len()));
    }
    let raw = args[0].as_str()?.replace('\\', "/");
    let (dir, file) = match raw.rfind('/') {
        Some(pos) => (&raw[..pos], &raw[pos + 1..]),
        None => ("", raw.as_str()),
    };
    let (name, ext) = match file.rfind('.') {
        Some(pos) if pos > 0 => (&file[..pos], &file[pos..]),
        _ => (file, ""),
    };
    let mut out = vec![Value::String(dir.to_string())];
    if nargout >= 2 {
        out.push(Value::String(name.to_string()));
    }
    if nargout >= 3 {
        out.push(Value::String(ext.to_string()));
    }
    Ok(out)
}

pub fn fn_isfile(args: &[Value], loader: &ModuleLoader) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("isfile", "1 (path)", args.len()));
    }
    let p = args[0].as_str()?;
    ok1(Value::bool(loader.vfs().exists(p)))
}

pub fn fn_isfolder(args: &[Value], loader: &ModuleLoader) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("isfolder", "1 (path)", args.len()));
    }
    let p = args[0].as_str()?;
    ok1(Value::bool(loader.vfs().is_dir(p)))
}

fn glob_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let (mut pi, mut ti) = (0, 0);
    let (mut star, mut match_i) = (None, 0);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            match_i = ti;
            pi += 1;
        } else if let Some(s) = star {
            pi = s + 1;
            match_i += 1;
            ti = match_i;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

pub fn fn_dir(
    args: &[Value],
    nargout: usize,
    loader: &ModuleLoader,
    stdout: &mut String,
) -> Result<Vec<Value>> {
    let filter = args.first().and_then(|v| v.as_str().ok()).unwrap_or("*");
    let files = loader.vfs().list_files();
    let matched: Vec<String> = files
        .into_iter()
        .filter(|f| {
            if filter == "*" || filter == "." || filter.is_empty() {
                true
            } else if filter.contains('*') || filter.contains('?') {
                let base = f.rsplit('/').next().unwrap_or(f);
                glob_match(filter, f) || glob_match(filter, base)
            } else {
                let norm = filter.trim_matches('/');
                f == norm || f.starts_with(&format!("{norm}/"))
            }
        })
        .collect();

    if nargout == 0 {
        for f in &matched {
            stdout.push_str(f);
            stdout.push('\n');
        }
        return Ok(vec![]);
    }

    let mut items = Vec::with_capacity(matched.len());
    for f in matched {
        let mut entry = BTreeMap::new();
        let bytes = loader.vfs().read_bytes(&f).map_or(0, |b| b.len());
        entry.insert("name".to_string(), Value::String(f.clone()));
        entry.insert("folder".to_string(), Value::String(loader.vfs().cwd()));
        entry.insert("bytes".to_string(), Value::scalar(bytes as f64));
        entry.insert("isdir".to_string(), Value::bool(loader.vfs().is_dir(&f)));
        items.push(entry);
    }
    let len = items.len();
    ok1(Value::StructArray(StructArray {
        rows: len,
        cols: if len == 0 { 0 } else { 1 },
        items,
    }))
}

pub fn fn_ls(
    args: &[Value],
    nargout: usize,
    loader: &ModuleLoader,
    stdout: &mut String,
) -> Result<Vec<Value>> {
    let filter = args.first().and_then(|v| v.as_str().ok()).unwrap_or("*");
    let files: Vec<String> = loader
        .vfs()
        .list_files()
        .into_iter()
        .filter(|f| filter == "*" || glob_match(filter, f))
        .collect();
    let joined = files.join("  ");
    if nargout == 0 {
        if !joined.is_empty() {
            stdout.push_str(&joined);
            stdout.push('\n');
        }
        Ok(vec![])
    } else {
        ok1(Value::String(joined))
    }
}

pub fn fn_pwd(loader: &ModuleLoader) -> Result<Vec<Value>> {
    ok1(Value::String(loader.vfs().cwd()))
}

pub fn fn_cd(args: &[Value], loader: &mut ModuleLoader) -> Result<Vec<Value>> {
    let prev = loader.vfs().cwd();
    if let Some(arg) = args.first() {
        loader.vfs_mut().set_cwd(arg.as_str()?)?;
    }
    ok1(Value::String(prev))
}

pub fn fn_mkdir(args: &[Value], loader: &mut ModuleLoader) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("mkdir", "1 (folderName)", 0));
    }
    loader.vfs_mut().mkdir(args[0].as_str()?)?;
    ok1(Value::bool(true))
}

pub fn fn_delete(args: &[Value], loader: &mut ModuleLoader) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("delete", "1 (filename)", 0));
    }
    for a in args {
        loader.vfs_mut().remove(a.as_str()?);
    }
    Ok(vec![])
}

pub fn fn_copyfile(args: &[Value], loader: &mut ModuleLoader) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("copyfile", "2 (src, dst)", args.len()));
    }
    let src = args[0].as_str()?;
    let dst = args[1].as_str()?;
    let bytes = loader
        .vfs()
        .read_bytes(src)
        .ok_or_else(|| MatroError::runtime(format!("copyfile: source '{src}' not found")))?;
    loader.vfs_mut().write_bytes(dst, &bytes)?;
    ok1(Value::bool(true))
}

pub fn fn_movefile(args: &[Value], loader: &mut ModuleLoader) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("movefile", "2 (src, dst)", args.len()));
    }
    let src = args[0].as_str()?;
    let dst = args[1].as_str()?;
    let bytes = loader
        .vfs()
        .read_bytes(src)
        .ok_or_else(|| MatroError::runtime(format!("movefile: source '{src}' not found")))?;
    loader.vfs_mut().write_bytes(dst, &bytes)?;
    loader.vfs_mut().remove(src);
    ok1(Value::bool(true))
}

// ---------------------------------------------------------------------------
// Pure-Rust Image I/O (`imread` / `imwrite`: BMP & PNG)
// ---------------------------------------------------------------------------

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

fn png_chunk(chunk_type: &[u8; 4], payload: &[u8], out: &mut Vec<u8>) {
    out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(chunk_type);
    out.extend_from_slice(payload);
    let c = crc32(&out[start..]);
    out.extend_from_slice(&c.to_be_bytes());
}

fn encode_png(h: usize, w: usize, channels: usize, rgb_row_major: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]); // PNG signature
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(w as u32).to_be_bytes());
    ihdr.extend_from_slice(&(h as u32).to_be_bytes());
    ihdr.push(8); // bit depth
    ihdr.push(if channels == 1 { 0 } else { 2 }); // 0 = Grayscale, 2 = RGB
    ihdr.push(0); // compression
    ihdr.push(0); // filter
    ihdr.push(0); // interlace
    png_chunk(b"IHDR", &ihdr, &mut out);

    // Raw scanlines with filter byte 0 (None)
    let stride = w * channels;
    let mut raw = Vec::with_capacity(h * (stride + 1));
    for r in 0..h {
        raw.push(0);
        raw.extend_from_slice(&rgb_row_major[r * stride..(r + 1) * stride]);
    }
    let zlib_data = zlib_wrap_stored(&raw);
    png_chunk(b"IDAT", &zlib_data, &mut out);
    png_chunk(b"IEND", &[], &mut out);
    out
}

fn paeth_predictor(a: u8, b: u8, c: u8) -> u8 {
    let a = a as i32;
    let b = b as i32;
    let c = c as i32;
    let p = a + b - c;
    let pa = (p - a).abs();
    let pb = (p - b).abs();
    let pc = (p - c).abs();
    if pa <= pb && pa <= pc {
        a as u8
    } else if pb <= pc {
        b as u8
    } else {
        c as u8
    }
}

fn decode_png(bytes: &[u8]) -> Result<(usize, usize, usize, Vec<u8>)> {
    if bytes.len() < 8 || &bytes[..8] != b"\x89PNG\r\n\x1a\n" {
        return Err(MatroError::runtime("imread: invalid PNG signature"));
    }
    let mut pos = 8usize;
    let mut w = 0usize;
    let mut h = 0usize;
    let mut color_type = 0u8;
    let mut idat = Vec::new();
    while pos + 12 <= bytes.len() {
        let len = u32::from_be_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        let ctype = &bytes[pos + 4..pos + 8];
        pos += 8;
        if pos + len + 4 > bytes.len() {
            break;
        }
        let payload = &bytes[pos..pos + len];
        pos += len + 4;
        if ctype == b"IHDR" && payload.len() >= 13 {
            w = u32::from_be_bytes(payload[0..4].try_into().unwrap()) as usize;
            h = u32::from_be_bytes(payload[4..8].try_into().unwrap()) as usize;
            let bit_depth = payload[8];
            color_type = payload[9];
            if bit_depth != 8 {
                return Err(MatroError::runtime(format!(
                    "imread: unsupported PNG bit depth {bit_depth} (expected 8)"
                )));
            }
        } else if ctype == b"IDAT" {
            idat.extend_from_slice(payload);
        } else if ctype == b"IEND" {
            break;
        }
    }
    let bpp = match color_type {
        0 => 1usize,
        2 => 3usize,
        6 => 4usize,
        other => {
            return Err(MatroError::runtime(format!(
                "imread: unsupported PNG color type {other} (supported: 0 gray, 2 RGB, 6 RGBA)"
            )));
        }
    };
    let inflated = inflate_zlib(&idat)?;
    let stride = w * bpp;
    if inflated.len() < h * (stride + 1) {
        return Err(MatroError::runtime("imread: truncated PNG image data"));
    }
    let mut recon = vec![0u8; h * stride];
    for r in 0..h {
        let ftype = inflated[r * (stride + 1)];
        let src = &inflated[r * (stride + 1) + 1..r * (stride + 1) + 1 + stride];
        for i in 0..stride {
            let left = if i >= bpp { recon[r * stride + i - bpp] } else { 0 };
            let up = if r > 0 { recon[(r - 1) * stride + i] } else { 0 };
            let up_left = if r > 0 && i >= bpp {
                recon[(r - 1) * stride + i - bpp]
            } else {
                0
            };
            let v = match ftype {
                0 => src[i],
                1 => src[i].wrapping_add(left),
                2 => src[i].wrapping_add(up),
                3 => src[i].wrapping_add((((left as u16) + (up as u16)) / 2) as u8),
                4 => src[i].wrapping_add(paeth_predictor(left, up, up_left)),
                _ => {
                    return Err(MatroError::runtime(format!(
                        "imread: unsupported PNG scanline filter {ftype}"
                    )));
                }
            };
            recon[r * stride + i] = v;
        }
    }
    if bpp == 4 {
        // Strip alpha into RGB
        let mut rgb = Vec::with_capacity(h * w * 3);
        for px in recon.chunks_exact(4) {
            rgb.push(px[0]);
            rgb.push(px[1]);
            rgb.push(px[2]);
        }
        Ok((h, w, 3, rgb))
    } else {
        Ok((h, w, bpp, recon))
    }
}

fn encode_bmp(h: usize, w: usize, channels: usize, row_major: &[u8]) -> Vec<u8> {
    let row_bytes = w * 3;
    let pad = (4 - (row_bytes % 4)) % 4;
    let img_size = (row_bytes + pad) * h;
    let file_size = 54 + img_size;
    let mut out = Vec::with_capacity(file_size);
    // BITMAPFILEHEADER (14 bytes)
    out.extend_from_slice(b"BM");
    out.extend_from_slice(&(file_size as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&54u32.to_le_bytes());
    // BITMAPINFOHEADER (40 bytes)
    out.extend_from_slice(&40u32.to_le_bytes());
    out.extend_from_slice(&(w as i32).to_le_bytes());
    out.extend_from_slice(&(h as i32).to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // planes
    out.extend_from_slice(&24u16.to_le_bytes()); // 24-bit BGR
    out.extend_from_slice(&0u32.to_le_bytes()); // BI_RGB
    out.extend_from_slice(&(img_size as u32).to_le_bytes());
    out.extend_from_slice(&2835u32.to_le_bytes());
    out.extend_from_slice(&2835u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());

    // Bottom-up BGR rows
    for r in (0..h).rev() {
        for c in 0..w {
            let (red, grn, blu) = if channels == 1 {
                let g = row_major[r * w + c];
                (g, g, g)
            } else {
                let idx = (r * w + c) * 3;
                (row_major[idx], row_major[idx + 1], row_major[idx + 2])
            };
            out.push(blu);
            out.push(grn);
            out.push(red);
        }
        for _ in 0..pad {
            out.push(0);
        }
    }
    out
}

fn decode_bmp(bytes: &[u8]) -> Result<(usize, usize, usize, Vec<u8>)> {
    if bytes.len() < 54 || &bytes[0..2] != b"BM" {
        return Err(MatroError::runtime("imread: invalid BMP header"));
    }
    let data_offset = u32::from_le_bytes(bytes[10..14].try_into().unwrap()) as usize;
    let w = i32::from_le_bytes(bytes[18..22].try_into().unwrap()).unsigned_abs() as usize;
    let raw_h = i32::from_le_bytes(bytes[22..26].try_into().unwrap());
    let top_down = raw_h < 0;
    let h = raw_h.unsigned_abs() as usize;
    let bpp = u16::from_le_bytes(bytes[28..30].try_into().unwrap());
    if bpp != 24 && bpp != 32 {
        return Err(MatroError::runtime(format!(
            "imread: unsupported BMP bit depth {bpp} (supported: 24, 32)"
        )));
    }
    let px_bytes = (bpp / 8) as usize;
    let row_stride = (w * px_bytes + 3) & !3;
    let mut rgb = vec![0u8; h * w * 3];
    for r in 0..h {
        let src_r = if top_down { r } else { h - 1 - r };
        let row_start = data_offset + src_r * row_stride;
        if row_start + w * px_bytes > bytes.len() {
            return Err(MatroError::runtime("imread: truncated BMP pixel data"));
        }
        for c in 0..w {
            let p = row_start + c * px_bytes;
            let dst = (r * w + c) * 3;
            rgb[dst] = bytes[p + 2];
            rgb[dst + 1] = bytes[p + 1];
            rgb[dst + 2] = bytes[p];
        }
    }
    Ok((h, w, 3, rgb))
}

pub fn fn_imwrite(args: &[Value], loader: &mut ModuleLoader) -> Result<Vec<Value>> {
    if args.len() < 2 {
        return Err(MatroError::arity("imwrite", "2 (img, filename)", args.len()));
    }
    let img = args[0].to_dense_matrix()?;
    let path = args[1].as_str()?;
    let dims = img.dims();
    let h = dims[0];
    let w = dims.get(1).copied().unwrap_or(1);
    let ch = dims.get(2).copied().unwrap_or(1);
    if ch != 1 && ch != 3 {
        return Err(MatroError::dimension(
            "imwrite: image must be HxW (grayscale) or HxWx3 (RGB)",
        ));
    }
    let is_float = img.dtype().is_float();
    let to_u8 = |v: f64| -> u8 {
        if is_float {
            (v.clamp(0.0, 1.0) * 255.0).round() as u8
        } else {
            v.clamp(0.0, 255.0).round() as u8
        }
    };
    let mut row_major = vec![0u8; h * w * ch];
    for r in 0..h {
        for c in 0..w {
            if ch == 1 {
                row_major[r * w + c] = to_u8(img.get(r, c));
            } else {
                for k in 0..3 {
                    let lin = k * (h * w) + c * h + r;
                    row_major[(r * w + c) * 3 + k] = to_u8(img.get_linear_c(lin).re);
                }
            }
        }
    }
    let lower = path.to_ascii_lowercase();
    let bytes = if lower.ends_with(".bmp") {
        encode_bmp(h, w, ch, &row_major)
    } else {
        encode_png(h, w, ch, &row_major)
    };
    loader.vfs_mut().write_bytes(path, &bytes)?;
    Ok(vec![])
}

pub fn fn_imread(args: &[Value], loader: &ModuleLoader) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("imread", "1 (filename)", 0));
    }
    let path = args[0].as_str()?;
    let bytes = loader
        .vfs()
        .read_bytes(path)
        .ok_or_else(|| MatroError::runtime(format!("imread: file '{path}' not found")))?;
    let (h, w, ch, row_major) = if bytes.len() >= 8 && &bytes[..8] == b"\x89PNG\r\n\x1a\n" {
        decode_png(&bytes)?
    } else if bytes.len() >= 2 && &bytes[..2] == b"BM" {
        decode_bmp(&bytes)?
    } else {
        return Err(MatroError::runtime(
            "imread: unrecognized image format (supported: PNG, BMP)",
        ));
    };

    if ch == 1 {
        let mut m = Matrix::zeros(h, w);
        for r in 0..h {
            for c in 0..w {
                m.set(r, c, row_major[r * w + c] as f64);
            }
        }
        return ok1(Value::matrix(m.cast_to(NumericClass::Uint8)));
    }
    let mut re = vec![0.0; h * w * 3];
    for k in 0..3 {
        for c in 0..w {
            for r in 0..h {
                let lin = k * (h * w) + c * h + r;
                re[lin] = row_major[(r * w + c) * 3 + k] as f64;
            }
        }
    }
    let im = vec![0.0; re.len()];
    let m = Matrix::from_nd_complex_col_major(vec![h, w, 3], re, im)?
        .normalize()
        .cast_to(NumericClass::Uint8);
    ok1(Value::matrix(m))
}

// ---------------------------------------------------------------------------
// Pure-Rust Audio I/O (`audioread` / `audiowrite`: 16-bit PCM WAV)
// ---------------------------------------------------------------------------

pub fn fn_audiowrite(args: &[Value], loader: &mut ModuleLoader) -> Result<Vec<Value>> {
    if args.len() < 3 {
        return Err(MatroError::arity(
            "audiowrite",
            "3 (filename, y, Fs)",
            args.len(),
        ));
    }
    let path = args[0].as_str()?;
    let y = args[1].to_dense_matrix()?;
    let fs = args[2].as_scalar()?.round().max(1.0) as u32;

    // Treat row vector as single-channel column
    let (nsamples, nch) = if y.rows() == 1 && y.cols() > 1 {
        (y.cols(), 1usize)
    } else {
        (y.rows(), y.cols())
    };
    let bits_per_sample = 16u16;
    let block_align = (nch as u16) * (bits_per_sample / 8);
    let byte_rate = fs * (block_align as u32);
    let data_bytes = (nsamples * nch * 2) as u32;
    let riff_size = 36 + data_bytes;

    let mut out = Vec::with_capacity((44 + data_bytes) as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&(nch as u16).to_le_bytes());
    out.extend_from_slice(&fs.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());

    for s in 0..nsamples {
        for ch in 0..nch {
            let v = if y.rows() == 1 && y.cols() > 1 {
                y.get(0, s)
            } else {
                y.get(s, ch)
            };
            let q = (v.clamp(-1.0, 1.0) * 32767.0).round() as i16;
            out.extend_from_slice(&q.to_le_bytes());
        }
    }

    loader.vfs_mut().write_bytes(path, &out)?;
    Ok(vec![])
}

pub fn fn_audioread(args: &[Value], nargout: usize, loader: &ModuleLoader) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("audioread", "1 (filename)", 0));
    }
    let path = args[0].as_str()?;
    let bytes = loader
        .vfs()
        .read_bytes(path)
        .ok_or_else(|| MatroError::runtime(format!("audioread: file '{path}' not found")))?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(MatroError::runtime("audioread: invalid WAV header"));
    }
    let mut pos = 12usize;
    let mut nch = 1usize;
    let mut fs = 44100u32;
    let mut bps = 16u16;
    let mut pcm: &[u8] = &[];
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let sz = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        if pos + sz > bytes.len() {
            break;
        }
        if id == b"fmt " && sz >= 16 {
            nch = u16::from_le_bytes(bytes[pos + 2..pos + 4].try_into().unwrap()) as usize;
            fs = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap());
            bps = u16::from_le_bytes(bytes[pos + 14..pos + 16].try_into().unwrap());
        } else if id == b"data" {
            pcm = &bytes[pos..pos + sz];
        }
        pos += sz + (sz & 1);
    }
    if bps != 16 || nch == 0 {
        return Err(MatroError::runtime(format!(
            "audioread: unsupported WAV format ({bps}-bit, {nch} ch; expected 16-bit PCM)"
        )));
    }
    let nsamples = pcm.len() / (2 * nch);
    let mut mat = Matrix::zeros(nsamples, nch);
    for s in 0..nsamples {
        for ch in 0..nch {
            let off = (s * nch + ch) * 2;
            let v = i16::from_le_bytes([pcm[off], pcm[off + 1]]) as f64 / 32767.0;
            mat.set(s, ch, v);
        }
    }
    if nargout >= 2 {
        Ok(vec![Value::matrix(mat), Value::scalar(fs as f64)])
    } else {
        ok1(Value::matrix(mat))
    }
}
