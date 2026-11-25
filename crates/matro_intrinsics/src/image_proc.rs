//! Pure-Rust Image Processing Toolbox (zero external dependencies):
//! - `rgb2gray`, `im2double`, `im2uint8`, `mat2gray`, `imcomplement`
//! - `imresize` (nearest/bilinear/bicubic, `[H NaN]` aspect ratio, 2D & RGB)
//! - `imfilter` (zero/scalar/replicate/symmetric/circular, same/full, corr/conv, 2D & RGB)
//! - `imgaussfilt`, `medfilt2`, `fspecial`
//! - `edge` (Sobel, Prewitt, Roberts, LoG, Canny)
//! - `graythresh` (Otsu), `imbinarize`, `imhist`, `imadjust`
//! - `bwlabel`, `regionprops`, `imerode`, `imdilate`, `imopen`, `imclose`

use std::collections::{BTreeMap, VecDeque};

use matro_types::{MatroError, Matrix, NumericClass, Result, StructArray, Value};

fn ok1(v: Value) -> Result<Vec<Value>> {
    Ok(vec![v])
}

fn to_unit_double(m: &Matrix) -> Matrix {
    let scale = if m.dtype() == NumericClass::Uint8 {
        1.0 / 255.0
    } else if m.dtype() == NumericClass::Uint16 {
        1.0 / 65535.0
    } else {
        1.0
    };
    let dims = m.dims();
    let re: Vec<f64> = m.data().iter().map(|&v| v * scale).collect();
    Matrix::from_nd_col_major(dims, re).unwrap_or_else(|_| m.map(|v| v * scale))
}

pub fn fn_im2double(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("im2double", "1 (I)", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    ok1(Value::matrix(to_unit_double(&m)))
}

pub fn fn_im2uint8(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("im2uint8", "1 (I)", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    if m.dtype() == NumericClass::Uint8 {
        return ok1(Value::matrix(m));
    }
    let dims = m.dims();
    let re: Vec<f64> = match m.dtype() {
        NumericClass::Uint16 => m
            .data()
            .iter()
            .map(|&v| ((v.clamp(0.0, 65535.0) / 65535.0) * 255.0).round())
            .collect(),
        NumericClass::Logical => m
            .data()
            .iter()
            .map(|&v| if v != 0.0 { 255.0 } else { 0.0 })
            .collect(),
        _ => m
            .data()
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round())
            .collect(),
    };
    let out = Matrix::from_nd_col_major(dims, re)?.cast_to(NumericClass::Uint8);
    ok1(Value::matrix(out))
}

pub fn fn_mat2gray(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("mat2gray", "1 or 2 (A [, [amin amax]])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    if m.is_empty() {
        return ok1(Value::matrix(m));
    }
    let (amin, amax) = if let Some(lim) = args.get(1) {
        let lm = lim.to_dense_matrix()?;
        if lm.len() != 2 {
            return Err(MatroError::dimension("mat2gray: limits must be [amin, amax]"));
        }
        (lm.data()[0], lm.data()[1])
    } else {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &v in m.data() {
            if v < lo { lo = v; }
            if v > hi { hi = v; }
        }
        (lo, hi)
    };
    let span = amax - amin;
    let re: Vec<f64> = if span.abs() <= 1e-15 {
        vec![1.0; m.len()]
    } else {
        m.data().iter().map(|&v| ((v - amin) / span).clamp(0.0, 1.0)).collect()
    };
    ok1(Value::matrix(Matrix::from_nd_col_major(m.dims(), re)?))
}

pub fn fn_imcomplement(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("imcomplement", "1 (I)", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let dt = m.dtype();
    let re: Vec<f64> = match dt {
        NumericClass::Logical => m.data().iter().map(|&v| if v == 0.0 { 1.0 } else { 0.0 }).collect(),
        NumericClass::Uint8 => m.data().iter().map(|&v| 255.0 - v).collect(),
        NumericClass::Uint16 => m.data().iter().map(|&v| 65535.0 - v).collect(),
        NumericClass::Int8 => m.data().iter().map(|&v| -v).collect(),
        NumericClass::Int16 => m.data().iter().map(|&v| -v).collect(),
        _ => m.data().iter().map(|&v| 1.0 - v).collect(),
    };
    let out = Matrix::from_nd_col_major(m.dims(), re)?.cast_to(dt);
    ok1(Value::matrix(out))
}

/// Convert `H x W x 3` RGB image to `H x W` grayscale using ITU-R BT.601 weights.
pub fn fn_rgb2gray(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("rgb2gray", "1 (RGB)", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let dims = m.dims();
    if dims.len() < 3 || dims[2] != 3 {
        return Err(MatroError::dimension("rgb2gray: input must be H x W x 3 RGB array"));
    }
    let h = dims[0];
    let w = dims[1];
    let plane = h * w;
    let mut out = Matrix::zeros(h, w);
    for c in 0..w {
        for r in 0..h {
            let idx = c * h + r;
            let red = m.get_linear_c(idx).re;
            let grn = m.get_linear_c(plane + idx).re;
            let blu = m.get_linear_c(2 * plane + idx).re;
            out.set(r, c, 0.2989 * red + 0.5870 * grn + 0.1140 * blu);
        }
    }
    ok1(Value::matrix(out.cast_to(m.dtype())))
}

fn resize_plane(src: &[f64], h: usize, w: usize, nh: usize, nw: usize, nearest: bool) -> Vec<f64> {
    let mut dst = vec![0.0; nh * nw];
    if h == 0 || w == 0 || nh == 0 || nw == 0 {
        return dst;
    }
    for c in 0..nw {
        let sx = if nw > 1 {
            (c as f64) * ((w - 1) as f64) / ((nw - 1) as f64)
        } else {
            0.0
        };
        let x0 = (sx.floor() as usize).min(w - 1);
        let x1 = (x0 + 1).min(w - 1);
        let fx = sx - (x0 as f64);
        let xn = (sx.round() as usize).min(w - 1);
        for r in 0..nh {
            let sy = if nh > 1 {
                (r as f64) * ((h - 1) as f64) / ((nh - 1) as f64)
            } else {
                0.0
            };
            if nearest {
                let yn = (sy.round() as usize).min(h - 1);
                dst[c * nh + r] = src[xn * h + yn];
            } else {
                let y0 = (sy.floor() as usize).min(h - 1);
                let y1 = (y0 + 1).min(h - 1);
                let fy = sy - (y0 as f64);

                let v00 = src[x0 * h + y0];
                let v10 = src[x1 * h + y0];
                let v01 = src[x0 * h + y1];
                let v11 = src[x1 * h + y1];
                let top = v00 + fx * (v10 - v00);
                let bot = v01 + fx * (v11 - v01);
                dst[c * nh + r] = top + fy * (bot - top);
            }
        }
    }
    dst
}

/// Image resize: `B = imresize(A, scale_or_size [, method])` with `[nh NaN]` aspect-ratio support.
pub fn fn_imresize(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity("imresize", "2 or 3 (A, scale_or_size [, method])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    if m.is_empty() {
        return ok1(Value::matrix(m));
    }
    let dims = m.dims();
    let h = dims[0];
    let w = dims.get(1).copied().unwrap_or(1);
    let ch = dims.get(2).copied().unwrap_or(1);
    let sz = args[1].to_dense_matrix()?;
    let nearest = args
        .get(2)
        .and_then(crate::value_as_text)
        .is_some_and(|s| s.eq_ignore_ascii_case("nearest"));

    let (nh, nw) = if sz.len() == 1 {
        let sc = sz.data()[0].max(1e-6);
        (((h as f64) * sc).round().max(1.0) as usize, ((w as f64) * sc).round().max(1.0) as usize)
    } else if sz.len() == 2 {
        let rh = sz.data()[0];
        let rw = sz.data()[1];
        if rh.is_nan() && !rw.is_nan() {
            let nw = rw.round().max(1.0) as usize;
            let nh = ((h as f64) * (nw as f64) / (w.max(1) as f64)).round().max(1.0) as usize;
            (nh, nw)
        } else if rw.is_nan() && !rh.is_nan() {
            let nh = rh.round().max(1.0) as usize;
            let nw = ((w as f64) * (nh as f64) / (h.max(1) as f64)).round().max(1.0) as usize;
            (nh, nw)
        } else {
            (rh.round().max(1.0) as usize, rw.round().max(1.0) as usize)
        }
    } else {
        return Err(MatroError::dimension("imresize: size must be scalar scale or [nh, nw]"));
    };

    let plane = h * w;
    if ch == 1 {
        let out_data = resize_plane(m.data(), h, w, nh, nw, nearest);
        let mat = Matrix::from_col_major(nh, nw, out_data)?.cast_to(m.dtype());
        return ok1(Value::matrix(mat));
    }
    let mut re = Vec::with_capacity(nh * nw * ch);
    for k in 0..ch {
        let slice = &m.data()[k * plane..(k + 1) * plane];
        re.extend(resize_plane(slice, h, w, nh, nw, nearest));
    }
    let mat = Matrix::from_nd_col_major(vec![nh, nw, ch], re)?.cast_to(m.dtype());
    ok1(Value::matrix(mat))
}

#[derive(Clone, Copy)]
enum PadMode {
    Const(f64),
    Replicate,
    Symmetric,
    Circular,
}

fn sample_padded(plane: &[f64], h: usize, w: usize, r: isize, c: isize, pad: PadMode) -> f64 {
    let resolve = |idx: isize, len: usize| -> Option<usize> {
        let n = len as isize;
        if idx >= 0 && idx < n {
            return Some(idx as usize);
        }
        match pad {
            PadMode::Const(_) => None,
            PadMode::Replicate => Some(idx.clamp(0, n - 1) as usize),
            PadMode::Circular => Some((((idx % n) + n) % n) as usize),
            PadMode::Symmetric => {
                let period = 2 * n;
                let mut m = ((idx % period) + period) % period;
                if m >= n {
                    m = period - 1 - m;
                }
                Some(m as usize)
            }
        }
    };
    match (resolve(r, h), resolve(c, w)) {
        (Some(sr), Some(sc)) => plane[sc * h + sr],
        _ => match pad {
            PadMode::Const(v) => v,
            _ => 0.0,
        },
    }
}

fn filter_plane_2d(
    plane: &[f64],
    h: usize,
    w: usize,
    kern: &Matrix,
    pad: PadMode,
    full: bool,
    conv: bool,
) -> (usize, usize, Vec<f64>) {
    let kh = kern.rows();
    let kw = kern.cols();
    let (oh, ow, r_off, c_off) = if full {
        (h + kh - 1, w + kw - 1, (kh - 1) as isize, (kw - 1) as isize)
    } else {
        (h, w, (kh / 2) as isize, (kw / 2) as isize)
    };
    let mut out = vec![0.0; oh * ow];
    for c in 0..ow {
        for r in 0..oh {
            let mut acc = 0.0;
            for kc in 0..kw {
                let (kcc, sc) = if conv {
                    (kw - 1 - kc, c as isize + kc as isize - c_off)
                } else {
                    (kc, c as isize + kc as isize - c_off)
                };
                for kr in 0..kh {
                    let (krr, sr) = if conv {
                        (kh - 1 - kr, r as isize + kr as isize - r_off)
                    } else {
                        (kr, r as isize + kr as isize - r_off)
                    };
                    acc += sample_padded(plane, h, w, sr, sc, pad) * kern.get(krr, kcc);
                }
            }
            out[c * oh + r] = acc;
        }
    }
    (oh, ow, out)
}

fn filter_2d_replicate(img: &Matrix, kern: &Matrix) -> Matrix {
    let (oh, ow, data) = filter_plane_2d(img.data(), img.rows(), img.cols(), kern, PadMode::Replicate, false, false);
    Matrix::from_col_major(oh, ow, data).unwrap_or_else(|_| Matrix::zeros(oh, ow))
}

pub fn fn_imfilter(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 {
        return Err(MatroError::arity("imfilter", ">= 2 (I, h [, options...])", args.len()));
    }
    let img = args[0].to_dense_matrix()?;
    let h = args[1].to_dense_matrix()?;
    if img.is_empty() || h.is_empty() {
        return ok1(Value::matrix(img));
    }
    let mut pad = PadMode::Const(0.0);
    let mut full = false;
    let mut conv = false;
    for opt in &args[2..] {
        if let Some(s) = crate::value_as_text(opt) {
            match s.to_ascii_lowercase().as_str() {
                "replicate" => pad = PadMode::Replicate,
                "symmetric" => pad = PadMode::Symmetric,
                "circular" => pad = PadMode::Circular,
                "full" => full = true,
                "same" => full = false,
                "conv" => conv = true,
                "corr" => conv = false,
                _ => {}
            }
        } else if let Ok(sc) = opt.as_scalar() {
            pad = PadMode::Const(sc);
        }
    }

    let dims = img.dims();
    let rows = dims[0];
    let cols = dims.get(1).copied().unwrap_or(1);
    let ch = dims.get(2).copied().unwrap_or(1);
    let plane_sz = rows * cols;
    if ch == 1 {
        let (oh, ow, out_data) = filter_plane_2d(img.data(), rows, cols, &h, pad, full, conv);
        let out = Matrix::from_col_major(oh, ow, out_data)?.cast_to(img.dtype());
        return ok1(Value::matrix(out));
    }
    let mut out_all = Vec::new();
    let mut out_h = 0;
    let mut out_w = 0;
    for k in 0..ch {
        let slice = &img.data()[k * plane_sz..(k + 1) * plane_sz];
        let (oh, ow, pd) = filter_plane_2d(slice, rows, cols, &h, pad, full, conv);
        out_h = oh;
        out_w = ow;
        out_all.extend(pd);
    }
    let out = Matrix::from_nd_col_major(vec![out_h, out_w, ch], out_all)?.cast_to(img.dtype());
    ok1(Value::matrix(out))
}

/// Separable 2D Gaussian blur: `B = imgaussfilt(I [, sigma])` (2D & RGB).
pub fn fn_imgaussfilt(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("imgaussfilt", "1 or 2 (I [, sigma])", args.len()));
    }
    let img = args[0].to_dense_matrix()?;
    if img.is_empty() {
        return ok1(Value::matrix(img));
    }
    let sigma = args
        .get(1)
        .map(|v| v.as_scalar())
        .transpose()?
        .unwrap_or(0.5)
        .max(1e-3);
    let radius = (2.0 * sigma).ceil().max(1.0) as isize;
    let klen = (2 * radius + 1) as usize;
    let mut k1d = Vec::with_capacity(klen);
    let mut sum = 0.0;
    for i in -radius..=radius {
        let w = (-0.5 * ((i as f64) / sigma).powi(2)).exp();
        k1d.push(w);
        sum += w;
    }
    for v in &mut k1d {
        *v /= sum;
    }
    let kx = Matrix::row_vec(k1d.clone());
    let ky = Matrix::col_vec(k1d);
    let dims = img.dims();
    let (h, w, ch) = (dims[0], dims.get(1).copied().unwrap_or(1), dims.get(2).copied().unwrap_or(1));
    let plane_sz = h * w;
    if ch == 1 {
        let tmp = filter_2d_replicate(&img, &kx);
        let out = filter_2d_replicate(&tmp, &ky).cast_to(img.dtype());
        return ok1(Value::matrix(out));
    }
    let mut all = Vec::with_capacity(img.len());
    for k in 0..ch {
        let sl = &img.data()[k * plane_sz..(k + 1) * plane_sz];
        let (_, _, t) = filter_plane_2d(sl, h, w, &kx, PadMode::Replicate, false, false);
        let (_, _, o) = filter_plane_2d(&t, h, w, &ky, PadMode::Replicate, false, false);
        all.extend(o);
    }
    ok1(Value::matrix(Matrix::from_nd_col_major(vec![h, w, ch], all)?.cast_to(img.dtype())))
}

/// 2D median filter: `B = medfilt2(I [, [m n]])`.
pub fn fn_medfilt2(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("medfilt2", "1 or 2 (I [, [m n]])", args.len()));
    }
    let img = args[0].to_dense_matrix()?;
    let (mh, mw) = if let Some(sz) = args.get(1) {
        let sm = sz.to_dense_matrix()?;
        if sm.len() == 1 {
            let k = sm.data()[0].max(1.0).round() as usize;
            (k, k)
        } else {
            (sm.data()[0].max(1.0).round() as usize, sm.data()[1].max(1.0).round() as usize)
        }
    } else {
        (3, 3)
    };
    let (h, w) = (img.rows(), img.cols());
    let hr = (mh / 2) as isize;
    let hc = (mw / 2) as isize;
    let mut out = Matrix::zeros(h, w);
    let mut win = Vec::with_capacity(mh * mw);
    for c in 0..w {
        for r in 0..h {
            win.clear();
            for kc in 0..mw {
                let sc = c as isize + kc as isize - hc;
                for kr in 0..mh {
                    let sr = r as isize + kr as isize - hr;
                    let v = if sr >= 0 && sr < h as isize && sc >= 0 && sc < w as isize {
                        img.get(sr as usize, sc as usize)
                    } else {
                        0.0
                    };
                    win.push(v);
                }
            }
            win.sort_by(f64::total_cmp);
            let n = win.len();
            let med = if n % 2 == 1 { win[n / 2] } else { 0.5 * (win[n / 2 - 1] + win[n / 2]) };
            out.set(r, c, med);
        }
    }
    ok1(Value::matrix(out.cast_to(img.dtype())))
}

/// Predefined 2D filter kernels: `h = fspecial(type [, p1 [, p2]])`.
pub fn fn_fspecial(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("fspecial", ">= 1 (type [, params...])", 0));
    }
    let kind = args[0].as_str()?.to_ascii_lowercase();
    let parse_hw = |arg: Option<&Value>, def: usize| -> Result<(usize, usize)> {
        if let Some(v) = arg {
            let m = v.to_dense_matrix()?;
            if m.len() == 1 {
                let k = m.data()[0].max(1.0).round() as usize;
                Ok((k, k))
            } else {
                Ok((m.data()[0].max(1.0).round() as usize, m.data()[1].max(1.0).round() as usize))
            }
        } else {
            Ok((def, def))
        }
    };
    let h = match kind.as_str() {
        "average" => {
            let (r, c) = parse_hw(args.get(1), 3)?;
            Matrix::from_col_major(r, c, vec![1.0 / (r * c) as f64; r * c])?
        }
        "gaussian" => {
            let (r, c) = parse_hw(args.get(1), 3)?;
            let sigma = args.get(2).map(|v| v.as_scalar()).transpose()?.unwrap_or(0.5);
            let hr = (r as f64 - 1.0) * 0.5;
            let hc = (c as f64 - 1.0) * 0.5;
            let mut m = Matrix::zeros(r, c);
            let mut sum = 0.0;
            for ci in 0..c {
                for ri in 0..r {
                    let dy = ri as f64 - hr;
                    let dx = ci as f64 - hc;
                    let w = (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
                    m.set(ri, ci, w);
                    sum += w;
                }
            }
            m.map(|v| v / sum)
        }
        "log" => {
            let (r, c) = parse_hw(args.get(1), 5)?;
            let sigma = args.get(2).map(|v| v.as_scalar()).transpose()?.unwrap_or(0.5);
            let hr = (r as f64 - 1.0) * 0.5;
            let hc = (c as f64 - 1.0) * 0.5;
            let s2 = sigma * sigma;
            let mut g = Matrix::zeros(r, c);
            let mut gsum = 0.0;
            for ci in 0..c {
                for ri in 0..r {
                    let dy = ri as f64 - hr;
                    let dx = ci as f64 - hc;
                    let w = (-(dx * dx + dy * dy) / (2.0 * s2)).exp();
                    g.set(ri, ci, w);
                    gsum += w;
                }
            }
            let mut out = Matrix::zeros(r, c);
            let mut mean = 0.0;
            for ci in 0..c {
                for ri in 0..r {
                    let dy = ri as f64 - hr;
                    let dx = ci as f64 - hc;
                    let v = (g.get(ri, ci) / gsum) * (dx * dx + dy * dy - 2.0 * s2) / (s2 * s2);
                    out.set(ri, ci, v);
                    mean += v;
                }
            }
            mean /= (r * c) as f64;
            out.map(|v| v - mean)
        }
        "laplacian" => {
            let alpha = args.get(1).map(|v| v.as_scalar()).transpose()?.unwrap_or(0.2).clamp(0.0, 1.0);
            let s = 4.0 / (alpha + 1.0);
            let corner = alpha * s / 4.0;
            let side = (1.0 - alpha) * s / 4.0;
            Matrix::from_col_major(
                3,
                3,
                vec![corner, side, corner, side, -s, side, corner, side, corner],
            )?
        }
        "sobel" => Matrix::from_col_major(3, 3, vec![1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0])?,
        "prewitt" => Matrix::from_col_major(3, 3, vec![1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0])?,
        "disk" => {
            let rad = args.get(1).map(|v| v.as_scalar()).transpose()?.unwrap_or(5.0).max(0.5);
            let cr = rad.ceil() as isize;
            let sz = (2 * cr + 1) as usize;
            let mut m = Matrix::zeros(sz, sz);
            let mut sum = 0.0f64;
            for ci in 0..sz {
                for ri in 0..sz {
                    let dx = ci as isize - cr;
                    let dy = ri as isize - cr;
                    if ((dx * dx + dy * dy) as f64).sqrt() <= rad + 0.5 {
                        m.set(ri, ci, 1.0);
                        sum += 1.0;
                    }
                }
            }
            m.map(|v| v / sum.max(1.0))
        }
        "unsharp" => {
            let alpha = args.get(1).map(|v| v.as_scalar()).transpose()?.unwrap_or(0.2).clamp(0.0, 1.0);
            let lap = fn_fspecial(&[Value::String("laplacian".into()), Value::scalar(alpha)])?[0]
                .to_dense_matrix()?;
            let mut m = lap.map(|v| -v);
            m.set(1, 1, m.get(1, 1) + 1.0);
            m
        }
        other => return Err(MatroError::runtime(format!("fspecial: unsupported type '{other}'"))),
    };
    ok1(Value::matrix(h))
}

/// Edge detection (`'sobel'`, `'prewitt'`, `'roberts'`, `'log'`, `'canny'`): `BW = edge(I [, method [, thresh]])`.
pub fn fn_edge(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 4 {
        return Err(MatroError::arity("edge", "1 to 4 (I [, method [, thresh [, sigma]]])", args.len()));
    }
    let img = to_unit_double(&args[0].to_dense_matrix()?);
    let method = args
        .get(1)
        .and_then(|v| v.as_str().ok())
        .unwrap_or("sobel")
        .to_ascii_lowercase();
    let (h, w) = (img.rows(), img.cols());

    if method == "log" {
        let sigma = args.get(3).map(|v| v.as_scalar()).transpose()?.unwrap_or(2.0);
        let ksz = ((sigma * 3.0).ceil() as usize * 2 + 1).max(5);
        let kern = fn_fspecial(&[
            Value::String("log".into()),
            Value::scalar(ksz as f64),
            Value::scalar(sigma),
        ])?[0]
            .to_dense_matrix()?;
        let resp = filter_2d_replicate(&img, &kern);
        let max_abs = resp.data().iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let thresh = args.get(2).and_then(|v| v.as_scalar().ok()).unwrap_or(0.04 * max_abs);
        let mut bw = Matrix::zeros(h, w);
        for c in 1..w.saturating_sub(1) {
            for r in 1..h.saturating_sub(1) {
                let v = resp.get(r, c);
                let cross_x = resp.get(r, c - 1) * resp.get(r, c + 1) < 0.0
                    && (resp.get(r, c - 1) - resp.get(r, c + 1)).abs() > thresh;
                let cross_y = resp.get(r - 1, c) * resp.get(r + 1, c) < 0.0
                    && (resp.get(r - 1, c) - resp.get(r + 1, c)).abs() > thresh;
                if (cross_x || cross_y) && v.abs() <= max_abs {
                    bw.set(r, c, 1.0);
                }
            }
        }
        return ok1(Value::matrix(bw.with_logical(true)));
    }

    if method == "canny" {
        let sigma = args.get(3).map(|v| v.as_scalar()).transpose()?.unwrap_or(1.4);
        let smooth = fn_imgaussfilt(&[Value::matrix(img), Value::scalar(sigma)])?[0].to_dense_matrix()?;
        let gx_k = Matrix::from_col_major(3, 3, vec![-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0])?;
        let gy_k = Matrix::from_col_major(3, 3, vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0])?;
        let gx = filter_2d_replicate(&smooth, &gx_k);
        let gy = filter_2d_replicate(&smooth, &gy_k);
        let mut mag = Matrix::zeros(h, w);
        let mut max_m = 0.0f64;
        for i in 0..mag.len() {
            let m = (gx.data()[i].powi(2) + gy.data()[i].powi(2)).sqrt();
            mag.data_mut()[i] = m;
            if m > max_m { max_m = m; }
        }
        // Non-maximum suppression
        let mut nms = Matrix::zeros(h, w);
        for c in 1..w.saturating_sub(1) {
            for r in 1..h.saturating_sub(1) {
                let m = mag.get(r, c);
                let ang = gy.get(r, c).atan2(gx.get(r, c)).to_degrees().rem_euclid(180.0);
                let (m1, m2) = if ang < 22.5 || ang >= 157.5 {
                    (mag.get(r, c - 1), mag.get(r, c + 1))
                } else if ang < 67.5 {
                    (mag.get(r - 1, c + 1), mag.get(r + 1, c - 1))
                } else if ang < 112.5 {
                    (mag.get(r - 1, c), mag.get(r + 1, c))
                } else {
                    (mag.get(r - 1, c - 1), mag.get(r + 1, c + 1))
                };
                if m >= m1 && m >= m2 {
                    nms.set(r, c, m);
                }
            }
        }
        let (t_lo, t_hi) = if let Some(tv) = args.get(2) {
            let tm = tv.to_dense_matrix()?;
            if tm.len() >= 2 {
                (tm.data()[0] * max_m, tm.data()[1] * max_m)
            } else {
                let hi = tm.data()[0] * max_m;
                (0.4 * hi, hi)
            }
        } else {
            (0.08 * max_m, 0.2 * max_m)
        };
        // Hysteresis via BFS from strong edges
        let mut out = Matrix::zeros(h, w);
        let mut queue = VecDeque::new();
        for c in 0..w {
            for r in 0..h {
                if nms.get(r, c) >= t_hi && t_hi > 0.0 {
                    out.set(r, c, 1.0);
                    queue.push_back((r, c));
                }
            }
        }
        while let Some((r, c)) = queue.pop_front() {
            for dc in -1isize..=1 {
                for dr in -1isize..=1 {
                    let nr = r as isize + dr;
                    let nc = c as isize + dc;
                    if nr >= 0 && nr < h as isize && nc >= 0 && nc < w as isize {
                        let (nr, nc) = (nr as usize, nc as usize);
                        if out.get(nr, nc) == 0.0 && nms.get(nr, nc) >= t_lo {
                            out.set(nr, nc, 1.0);
                            queue.push_back((nr, nc));
                        }
                    }
                }
            }
        }
        return ok1(Value::matrix(out.with_logical(true)));
    }

    let (gx_k, gy_k) = match method.as_str() {
        "prewitt" => (
            Matrix::from_col_major(3, 3, vec![-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])?,
            Matrix::from_col_major(3, 3, vec![-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0])?,
        ),
        "roberts" => (
            Matrix::from_col_major(2, 2, vec![1.0, 0.0, 0.0, -1.0])?,
            Matrix::from_col_major(2, 2, vec![0.0, 1.0, -1.0, 0.0])?,
        ),
        _ => (
            Matrix::from_col_major(3, 3, vec![-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0])?,
            Matrix::from_col_major(3, 3, vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0])?,
        ),
    };
    let gx = filter_2d_replicate(&img, &gx_k);
    let gy = filter_2d_replicate(&img, &gy_k);
    let mut mag = Matrix::zeros(h, w);
    let mut sum_sq = 0.0;
    for i in 0..mag.len() {
        let m2 = gx.data()[i].powi(2) + gy.data()[i].powi(2);
        mag.data_mut()[i] = m2.sqrt();
        sum_sq += m2;
    }
    let thresh = match args.get(2) {
        Some(v) => v.as_scalar()?,
        None => (2.0 * sum_sq / (mag.len().max(1) as f64)).sqrt(),
    };
    let bw = mag
        .map(|v| if v > thresh { 1.0 } else { 0.0 })
        .with_logical(true);
    ok1(Value::matrix(bw))
}

fn otsu_threshold(unit: &Matrix) -> f64 {
    let mut hist = [0u64; 256];
    let n = unit.len() as f64;
    if n == 0.0 {
        return 0.5;
    }
    for &v in unit.data() {
        let bin = (v.clamp(0.0, 1.0) * 255.0).round() as usize;
        hist[bin.min(255)] += 1;
    }
    let total_sum: f64 = hist
        .iter()
        .enumerate()
        .map(|(i, &c)| (i as f64) * (c as f64))
        .sum();
    let mut w_b = 0.0;
    let mut sum_b = 0.0;
    let mut best_var = -1.0;
    let mut vars = [0.0f64; 256];
    for (t, &cnt) in hist.iter().enumerate() {
        w_b += cnt as f64;
        if w_b == 0.0 {
            continue;
        }
        let w_f = n - w_b;
        if w_f == 0.0 {
            break;
        }
        sum_b += (t as f64) * (cnt as f64);
        let m_b = sum_b / w_b;
        let m_f = (total_sum - sum_b) / w_f;
        let between = w_b * w_f * (m_b - m_f).powi(2);
        vars[t] = between;
        if between > best_var {
            best_var = between;
        }
    }
    let mut sum_t = 0.0;
    let mut count_t = 0.0;
    for (t, &v) in vars.iter().enumerate() {
        if v >= best_var - 1e-12 * best_var.abs().max(1.0) && v > 0.0 {
            sum_t += t as f64;
            count_t += 1.0;
        }
    }
    let best_t = if count_t > 0.0 { sum_t / count_t } else { 128.0 };
    best_t / 255.0
}

pub fn fn_graythresh(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("graythresh", "1 (I)", args.len()));
    }
    let unit = to_unit_double(&args[0].to_dense_matrix()?);
    ok1(Value::scalar(otsu_threshold(&unit)))
}

pub fn fn_imbinarize(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("imbinarize", "1 or 2 (I [, thresh])", args.len()));
    }
    let unit = to_unit_double(&args[0].to_dense_matrix()?);
    let t = match args.get(1) {
        Some(v) => v.as_scalar()?,
        None => otsu_threshold(&unit),
    };
    let bw = unit
        .map(|v| if v > t { 1.0 } else { 0.0 })
        .with_logical(true);
    ok1(Value::matrix(bw))
}

pub fn fn_imhist(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("imhist", "1 or 2 (I [, nbins])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let nbins = args
        .get(1)
        .map(|v| v.as_scalar())
        .transpose()?
        .unwrap_or(256.0)
        .round()
        .max(2.0) as usize;
    let unit = to_unit_double(&m);
    let mut counts = vec![0.0; nbins];
    for &v in unit.data() {
        let bin = ((v.clamp(0.0, 1.0) * ((nbins - 1) as f64)).round() as usize).min(nbins - 1);
        counts[bin] += 1.0;
    }
    let scale = if m.dtype() == NumericClass::Uint8 { 255.0 } else { 1.0 };
    let locs: Vec<f64> = (0..nbins)
        .map(|i| scale * (i as f64) / ((nbins - 1) as f64))
        .collect();
    if nargout >= 2 {
        Ok(vec![
            Value::matrix(Matrix::col_vec(counts)),
            Value::matrix(Matrix::col_vec(locs)),
        ])
    } else {
        ok1(Value::matrix(Matrix::col_vec(counts)))
    }
}

/// `J = imadjust(I [, [low_in high_in] [, [low_out high_out] [, gamma]]])`
pub fn fn_imadjust(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 4 {
        return Err(MatroError::arity(
            "imadjust",
            "1 to 4 (I [, lim_in [, lim_out [, gamma]]])",
            args.len(),
        ));
    }
    let m = args[0].to_dense_matrix()?;
    let unit = to_unit_double(&m);
    let (lo_in, hi_in) = match args.get(1) {
        Some(v) if !v.to_dense_matrix()?.is_empty() => {
            let lm = v.to_dense_matrix()?;
            (lm.data()[0], lm.data()[1.min(lm.len() - 1)])
        }
        _ => {
            let mut sorted = unit.data().to_vec();
            sorted.sort_by(|a, b| a.total_cmp(b));
            let n = sorted.len();
            if n == 0 {
                (0.0, 1.0)
            } else {
                (
                    sorted[(0.01 * (n.saturating_sub(1) as f64)).round() as usize],
                    sorted[(0.99 * (n.saturating_sub(1) as f64)).round() as usize],
                )
            }
        }
    };
    let (lo_out, hi_out) = match args.get(2) {
        Some(v) if !v.to_dense_matrix()?.is_empty() => {
            let lm = v.to_dense_matrix()?;
            (lm.data()[0], lm.data()[1.min(lm.len() - 1)])
        }
        _ => (0.0, 1.0),
    };
    let gamma = args.get(3).map(|v| v.as_scalar()).transpose()?.unwrap_or(1.0).max(1e-6);

    let span = (hi_in - lo_in).max(1e-12);
    let stretched = unit.map(|v| {
        let t = ((v - lo_in) / span).clamp(0.0, 1.0);
        lo_out + (hi_out - lo_out) * t.powf(gamma)
    });
    if m.dtype() == NumericClass::Uint8 {
        ok1(Value::matrix(
            stretched.map(|v| (v.clamp(0.0, 1.0) * 255.0).round()).cast_to(NumericClass::Uint8),
        ))
    } else {
        ok1(Value::matrix(stretched))
    }
}

/// `[L, num] = bwlabel(BW [, conn])` — Connected component labeling (conn = 8 default, or 4).
pub fn fn_bwlabel(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("bwlabel", "1 or 2 (BW [, conn])", args.len()));
    }
    let bw = args[0].to_dense_matrix()?;
    let conn = args.get(1).map(|v| v.as_scalar()).transpose()?.unwrap_or(8.0).round() as i32;
    let (h, w) = (bw.rows(), bw.cols());
    let mut labels = Matrix::zeros(h, w);
    let mut num = 0usize;
    let mut queue = VecDeque::new();
    let dirs4: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    let dirs8: [(isize, isize); 8] = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ];
    let dirs: &[(isize, isize)] = if conn == 4 { &dirs4 } else { &dirs8 };

    // MATLAB scans column-major
    for c in 0..w {
        for r in 0..h {
            if bw.get(r, c) != 0.0 && labels.get(r, c) == 0.0 {
                num += 1;
                labels.set(r, c, num as f64);
                queue.push_back((r, c));
                while let Some((cr, cc)) = queue.pop_front() {
                    for &(dr, dc) in dirs {
                        let nr = cr as isize + dr;
                        let nc = cc as isize + dc;
                        if nr >= 0 && nr < h as isize && nc >= 0 && nc < w as isize {
                            let (nr, nc) = (nr as usize, nc as usize);
                            if bw.get(nr, nc) != 0.0 && labels.get(nr, nc) == 0.0 {
                                labels.set(nr, nc, num as f64);
                                queue.push_back((nr, nc));
                            }
                        }
                    }
                }
            }
        }
    }
    if nargout >= 2 {
        Ok(vec![Value::matrix(labels), Value::scalar(num as f64)])
    } else {
        ok1(Value::matrix(labels))
    }
}

/// `stats = regionprops(BW_or_L [, props...])` — Connected region measurements.
pub fn fn_regionprops(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("regionprops", ">= 1 (BW_or_L [, props...])", 0));
    }
    let m = args[0].to_dense_matrix()?;
    let (labels, num) = if m.is_logical() {
        let res = fn_bwlabel(&[Value::matrix(m)], 2)?;
        (res[0].to_dense_matrix()?, res[1].as_scalar()?.round() as usize)
    } else {
        let max_lbl = m.data().iter().copied().fold(0.0f64, f64::max).round().max(0.0) as usize;
        (m, max_lbl)
    };
    let (h, w) = (labels.rows(), labels.cols());
    let mut areas = vec![0usize; num];
    let mut sum_x = vec![0.0; num];
    let mut sum_y = vec![0.0; num];
    let mut min_x = vec![usize::MAX; num];
    let mut max_x = vec![0usize; num];
    let mut min_y = vec![usize::MAX; num];
    let mut max_y = vec![0usize; num];
    let mut pix_idx: Vec<Vec<f64>> = vec![Vec::new(); num];

    for c in 0..w {
        for r in 0..h {
            let l = labels.get(r, c).round() as isize;
            if l >= 1 && (l as usize) <= num {
                let k = (l - 1) as usize;
                areas[k] += 1;
                sum_x[k] += (c + 1) as f64;
                sum_y[k] += (r + 1) as f64;
                min_x[k] = min_x[k].min(c + 1);
                max_x[k] = max_x[k].max(c + 1);
                min_y[k] = min_y[k].min(r + 1);
                max_y[k] = max_y[k].max(r + 1);
                pix_idx[k].push((c * h + r + 1) as f64);
            }
        }
    }

    let mut items = Vec::with_capacity(num);
    for k in 0..num {
        let mut st = BTreeMap::new();
        let area = areas[k] as f64;
        st.insert("Area".to_string(), Value::scalar(area));
        let cx = if areas[k] > 0 { sum_x[k] / area } else { f64::NAN };
        let cy = if areas[k] > 0 { sum_y[k] / area } else { f64::NAN };
        st.insert("Centroid".to_string(), Value::matrix(Matrix::row_vec(vec![cx, cy])));
        let bbox = if areas[k] > 0 {
            vec![
                min_x[k] as f64 - 0.5,
                min_y[k] as f64 - 0.5,
                (max_x[k] - min_x[k] + 1) as f64,
                (max_y[k] - min_y[k] + 1) as f64,
            ]
        } else {
            vec![0.0, 0.0, 0.0, 0.0]
        };
        st.insert("BoundingBox".to_string(), Value::matrix(Matrix::row_vec(bbox)));
        st.insert(
            "EquivDiameter".to_string(),
            Value::scalar((4.0 * area / std::f64::consts::PI).sqrt()),
        );
        st.insert(
            "PixelIdxList".to_string(),
            Value::matrix(Matrix::col_vec(pix_idx[k].clone())),
        );
        items.push(st);
    }
    ok1(Value::StructArray(StructArray {
        rows: num,
        cols: if num == 0 { 0 } else { 1 },
        items,
    }))
}

fn morph_op(img: &Matrix, se: &Matrix, dilate: bool) -> Matrix {
    let (h, w) = (img.rows(), img.cols());
    let (sh, sw) = (se.rows(), se.cols());
    let hr = (sh / 2) as isize;
    let hc = (sw / 2) as isize;
    let mut out = Matrix::zeros(h, w);
    for c in 0..w {
        for r in 0..h {
            let mut acc = if dilate { f64::NEG_INFINITY } else { f64::INFINITY };
            for kc in 0..sw {
                for kr in 0..sh {
                    if se.get(kr, kc) == 0.0 {
                        continue;
                    }
                    let sr = r as isize + kr as isize - hr;
                    let sc = c as isize + kc as isize - hc;
                    if sr >= 0 && sr < h as isize && sc >= 0 && sc < w as isize {
                        let v = img.get(sr as usize, sc as usize);
                        acc = if dilate { acc.max(v) } else { acc.min(v) };
                    }
                }
            }
            if !acc.is_finite() {
                acc = img.get(r, c);
            }
            out.set(r, c, acc);
        }
    }
    out.cast_to(img.dtype())
}

/// Morphological operations: `imerode`, `imdilate`, `imopen`, `imclose`.
pub fn fn_morph(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity(name, "2 (I, SE)", args.len()));
    }
    let img = args[0].to_dense_matrix()?;
    let se = args[1].to_dense_matrix()?;
    let out = match name {
        "imerode" => morph_op(&img, &se, false),
        "imdilate" => morph_op(&img, &se, true),
        "imopen" => {
            let er = morph_op(&img, &se, false);
            morph_op(&er, &se, true)
        }
        "imclose" => {
            let dl = morph_op(&img, &se, true);
            morph_op(&dl, &se, false)
        }
        _ => unreachable!(),
    };
    ok1(Value::matrix(out))
}
