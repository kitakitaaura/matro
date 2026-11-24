//! Pure-Rust Image Processing Toolbox basics (zero external dependencies):
//! - `rgb2gray`, `im2double`, `im2uint8`
//! - `imresize` (bilinear, 2D & RGB)
//! - `imfilter`, `imgaussfilt` (separable Gaussian)
//! - `edge` (Sobel & Prewitt)
//! - `graythresh` (Otsu), `imbinarize`, `imhist`, `imadjust`

use matro_types::{MatroError, Matrix, NumericClass, Result, Value};

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
    let im = vec![0.0; re.len()];
    Matrix::from_nd_complex_col_major(dims, re, im)
        .unwrap_or_else(|_| m.map(|v| v * scale))
        .normalize()
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
    let re: Vec<f64> = m.data().iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0).round()).collect();
    let im = vec![0.0; re.len()];
    let out = Matrix::from_nd_complex_col_major(dims, re, im)?
        .normalize()
        .cast_to(NumericClass::Uint8);
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

fn resize_plane(src: &[f64], h: usize, w: usize, nh: usize, nw: usize) -> Vec<f64> {
    let mut dst = vec![0.0; nh * nw];
    for c in 0..nw {
        let sx = if nw > 1 {
            (c as f64) * ((w - 1) as f64) / ((nw - 1) as f64)
        } else {
            0.0
        };
        let x0 = (sx.floor() as usize).min(w - 1);
        let x1 = (x0 + 1).min(w - 1);
        let fx = sx - (x0 as f64);
        for r in 0..nh {
            let sy = if nh > 1 {
                (r as f64) * ((h - 1) as f64) / ((nh - 1) as f64)
            } else {
                0.0
            };
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
    dst
}

/// Bilinear image resize: `B = imresize(A, scale)` or `B = imresize(A, [numrows, numcols])`.
pub fn fn_imresize(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("imresize", "2 (A, scale_or_size)", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let dims = m.dims();
    let h = dims[0];
    let w = dims.get(1).copied().unwrap_or(1);
    let ch = dims.get(2).copied().unwrap_or(1);
    let sz = args[1].to_dense_matrix()?;
    let (nh, nw) = if sz.len() == 1 {
        let sc = sz.data()[0].max(1e-6);
        (((h as f64) * sc).round().max(1.0) as usize, ((w as f64) * sc).round().max(1.0) as usize)
    } else if sz.len() == 2 {
        (sz.data()[0].round().max(1.0) as usize, sz.data()[1].round().max(1.0) as usize)
    } else {
        return Err(MatroError::dimension("imresize: size must be scalar scale or [nh, nw]"));
    };

    let plane = h * w;
    if ch == 1 {
        let out_data = resize_plane(m.data(), h, w, nh, nw);
        let mat = Matrix::from_col_major(nh, nw, out_data)?.cast_to(m.dtype());
        return ok1(Value::matrix(mat));
    }
    let mut re = Vec::with_capacity(nh * nw * ch);
    for k in 0..ch {
        let slice = &m.data()[k * plane..(k + 1) * plane];
        re.extend(resize_plane(slice, h, w, nh, nw));
    }
    let im = vec![0.0; re.len()];
    let mat = Matrix::from_nd_complex_col_major(vec![nh, nw, ch], re, im)?
        .normalize()
        .cast_to(m.dtype());
    ok1(Value::matrix(mat))
}

fn filter_2d_replicate(img: &Matrix, kern: &Matrix) -> Matrix {
    let h = img.rows();
    let w = img.cols();
    let kh = kern.rows();
    let kw = kern.cols();
    let hr = (kh / 2) as isize;
    let hc = (kw / 2) as isize;
    let mut out = Matrix::zeros(h, w);
    for c in 0..w {
        for r in 0..h {
            let mut acc = 0.0;
            for kc in 0..kw {
                let sc = (c as isize + kc as isize - hc).clamp(0, (w - 1) as isize) as usize;
                for kr in 0..kh {
                    let sr = (r as isize + kr as isize - hr).clamp(0, (h - 1) as isize) as usize;
                    acc += img.get(sr, sc) * kern.get(kr, kc);
                }
            }
            out.set(r, c, acc);
        }
    }
    out
}

pub fn fn_imfilter(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("imfilter", "2 (I, h)", args.len()));
    }
    let img = args[0].to_dense_matrix()?;
    let h = args[1].to_dense_matrix()?;
    let out = filter_2d_replicate(&img, &h).cast_to(img.dtype());
    ok1(Value::matrix(out))
}

/// Separable 2D Gaussian blur: `B = imgaussfilt(I [, sigma])`.
pub fn fn_imgaussfilt(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("imgaussfilt", "1 or 2 (I [, sigma])", args.len()));
    }
    let img = args[0].to_dense_matrix()?;
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
    let tmp = filter_2d_replicate(&img, &kx);
    let out = filter_2d_replicate(&tmp, &ky).cast_to(img.dtype());
    ok1(Value::matrix(out))
}

/// Edge detection (`'sobel'` or `'prewitt'`): `BW = edge(I [, method [, thresh]])`.
pub fn fn_edge(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity("edge", "1 to 3 (I [, method [, thresh]])", args.len()));
    }
    let img = to_unit_double(&args[0].to_dense_matrix()?);
    let method = args
        .get(1)
        .and_then(|v| v.as_str().ok())
        .unwrap_or("sobel")
        .to_ascii_lowercase();
    let (gx_k, gy_k) = if method == "prewitt" {
        (
            Matrix::from_col_major(3, 3, vec![-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])?,
            Matrix::from_col_major(3, 3, vec![-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0])?,
        )
    } else {
        (
            Matrix::from_col_major(3, 3, vec![-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0])?,
            Matrix::from_col_major(3, 3, vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0])?,
        )
    };
    let gx = filter_2d_replicate(&img, &gx_k);
    let gy = filter_2d_replicate(&img, &gy_k);
    let mut mag = Matrix::zeros(img.rows(), img.cols());
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
    // MATLAB graythresh averages all tied maximum-variance bins
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

pub fn fn_imadjust(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("imadjust", "1 (I)", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let unit = to_unit_double(&m);
    let mut sorted = unit.data().to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    let lo = sorted[(0.01 * (n.saturating_sub(1) as f64)).round() as usize];
    let hi = sorted[(0.99 * (n.saturating_sub(1) as f64)).round() as usize];
    let span = (hi - lo).max(1e-12);
    let stretched = unit.map(|v| ((v - lo) / span).clamp(0.0, 1.0));
    if m.dtype() == NumericClass::Uint8 {
        ok1(Value::matrix(
            stretched.map(|v| (v * 255.0).round()).cast_to(NumericClass::Uint8),
        ))
    } else {
        ok1(Value::matrix(stretched))
    }
}
