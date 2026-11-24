//! Pure-Rust Signal Processing Toolbox intrinsics:
//! `fftshift`, `ifftshift`, `hann`, `hamming`, `blackman`, `xcorr`,
//! `findpeaks`, `butter`, `filtfilt`, `pwelch`.

use std::f64::consts::PI;

use matro_types::{Complex, MatroError, Matrix, Result, Value};

fn ok1(v: Value) -> Result<Vec<Value>> {
    Ok(vec![v])
}

/// `fftshift(x)` / `ifftshift(x)` — shifts zero-frequency component to center (1D or 2D).
pub fn fn_fftshift(args: &[Value], inverse: bool) -> Result<Vec<Value>> {
    let fname = if inverse { "ifftshift" } else { "fftshift" };
    if args.len() != 1 {
        return Err(MatroError::arity(fname, "1", args.len()));
    }
    let m = args[0].as_matrix()?;
    if m.is_empty() {
        return ok1(Value::matrix(m.clone()));
    }
    let shift_1d = |n: usize| -> usize { if inverse { (n + 1) / 2 } else { n / 2 } };
    if m.rows() == 1 || m.cols() == 1 {
        let n = m.len();
        let sh = shift_1d(n);
        let vals: Vec<Complex> = (0..n).map(|i| m.get_linear_c((i + n - sh) % n)).collect();
        let out = if m.rows() == 1 {
            Matrix::complex_row_vec(vals)
        } else {
            Matrix::complex_col_vec(vals)
        };
        return ok1(Value::matrix(out.normalize()));
    }
    let (rows, cols) = (m.rows(), m.cols());
    let sr = shift_1d(rows);
    let sc = shift_1d(cols);
    let mut out = Matrix::zeros(rows, cols);
    for c in 0..cols {
        for r in 0..rows {
            out.set_c(r, c, m.get_c((r + rows - sr) % rows, (c + cols - sc) % cols));
        }
    }
    ok1(Value::matrix(out.normalize()))
}

/// Symmetric window functions: `hann(N)`, `hamming(N)`, `blackman(N)`.
pub fn fn_window(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity(name, "1 (N)", args.len()));
    }
    let n = args[0].as_scalar()?.max(0.0).round() as usize;
    if n == 0 {
        return ok1(Value::empty());
    }
    if n == 1 {
        return ok1(Value::scalar(1.0));
    }
    let denom = (n - 1) as f64;
    let w: Vec<f64> = (0..n)
        .map(|k| {
            let phase = 2.0 * PI * (k as f64) / denom;
            match name {
                "hann" => 0.5 * (1.0 - phase.cos()),
                "hamming" => 0.54 - 0.46 * phase.cos(),
                "blackman" => 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos(),
                _ => 1.0,
            }
        })
        .collect();
    ok1(Value::matrix(Matrix::col_vec(w)))
}

/// `[r, lags] = xcorr(x [, y])` — raw cross-correlation / autocorrelation.
pub fn fn_xcorr(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("xcorr", "1 or 2", args.len()));
    }
    let x = args[0].as_matrix()?.data();
    let y = if args.len() == 2 {
        args[1].as_matrix()?.data()
    } else {
        x
    };
    let n = x.len().max(y.len());
    if n == 0 {
        return ok1(Value::empty());
    }
    let m = 2 * n - 1;
    let mut r = vec![0.0; m];
    let mut lags = Vec::with_capacity(m);
    for idx in 0..m {
        let lag = idx as isize - (n as isize - 1);
        lags.push(lag as f64);
        let mut sum = 0.0;
        for i in 0..x.len() {
            let j = i as isize - lag;
            if j >= 0 && (j as usize) < y.len() {
                sum += x[i] * y[j as usize];
            }
        }
        r[idx] = sum;
    }
    let is_row = args[0].as_matrix()?.rows() == 1;
    let r_mat = if is_row {
        Matrix::row_vec(r)
    } else {
        Matrix::col_vec(r)
    };
    if nargout >= 2 {
        Ok(vec![Value::matrix(r_mat), Value::matrix(Matrix::row_vec(lags))])
    } else {
        ok1(Value::matrix(r_mat))
    }
}

/// `[pks, locs] = findpeaks(x [, 'MinPeakHeight', h [, 'MinPeakDistance', d]])`
pub fn fn_findpeaks(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("findpeaks", ">= 1", 0));
    }
    let x = args[0].as_matrix()?;
    let data = x.data();
    let mut min_height = f64::NEG_INFINITY;
    let mut min_dist: usize = 1;
    let mut i = 1;
    while i + 1 < args.len() {
        let key = args[i].as_str()?.to_ascii_lowercase();
        match key.as_str() {
            "minpeakheight" => min_height = args[i + 1].as_scalar()?,
            "minpeakdistance" => min_dist = args[i + 1].as_scalar()?.max(1.0).round() as usize,
            _ => {}
        }
        i += 2;
    }

    let mut cands: Vec<(usize, f64)> = Vec::new();
    if data.len() >= 3 {
        for k in 1..(data.len() - 1) {
            if data[k] > data[k - 1] && data[k] >= data[k + 1] && data[k] >= min_height {
                cands.push((k, data[k]));
            }
        }
    }

    // Enforce MinPeakDistance greedily in descending peak height order (MATLAB convention)
    let mut kept: Vec<(usize, f64)> = if min_dist > 1 {
        let mut by_height = cands.clone();
        by_height.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut chosen: Vec<(usize, f64)> = Vec::new();
        for (pos, val) in by_height {
            if chosen
                .iter()
                .all(|&(cpos, _)| pos.abs_diff(cpos) >= min_dist)
            {
                chosen.push((pos, val));
            }
        }
        chosen.sort_by_key(|&(pos, _)| pos);
        chosen
    } else {
        cands
    };

    let pks: Vec<f64> = kept.iter().map(|&(_, v)| v).collect();
    let locs: Vec<f64> = kept.drain(..).map(|(p, _)| (p + 1) as f64).collect();
    let is_row = x.rows() == 1;
    let pks_mat = if is_row {
        Matrix::row_vec(pks)
    } else {
        Matrix::col_vec(pks)
    };
    let locs_mat = if is_row {
        Matrix::row_vec(locs)
    } else {
        Matrix::col_vec(locs)
    };
    if nargout >= 2 {
        Ok(vec![Value::matrix(pks_mat), Value::matrix(locs_mat)])
    } else {
        ok1(Value::matrix(pks_mat))
    }
}

/// Multiply polynomial `p` (descending or ascending powers in `z^-1`) by `(1 - r * z^-1)`.
fn poly_from_roots_c(roots: &[Complex]) -> Vec<f64> {
    let mut poly = vec![Complex::ONE];
    for &r in roots {
        let mut next = vec![Complex::ZERO; poly.len() + 1];
        for (i, &c) in poly.iter().enumerate() {
            next[i] = next[i] + c;
            next[i + 1] = next[i + 1] - c * r;
        }
        poly = next;
    }
    poly.into_iter().map(|c| c.re).collect()
}

fn eval_poly_at(coeffs: &[f64], z: Complex) -> Complex {
    let mut acc = Complex::ZERO;
    let mut zp = Complex::ONE;
    for &c in coeffs {
        acc = acc + Complex::from_real(c) * zp;
        zp = zp * z;
    }
    acc
}

/// `[b, a] = butter(n, Wn [, 'low'|'high'|'bandpass'])` — digital Butterworth IIR filter design.
pub fn fn_butter(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity("butter", "2 or 3 (n, Wn [, ftype])", args.len()));
    }
    let n = args[0].as_scalar()?.round() as usize;
    if n == 0 || n > 12 {
        return Err(MatroError::runtime("butter: filter order n must be in 1..=12"));
    }
    let wn_mat = args[1].as_matrix()?;
    let ftype = args
        .get(2)
        .map(|v| v.as_str().map(str::to_ascii_lowercase))
        .transpose()?
        .unwrap_or_else(|| {
            if wn_mat.len() == 2 {
                "bandpass".to_string()
            } else {
                "low".to_string()
            }
        });

    // Analog normalized Butterworth poles in left half-plane
    let proto_poles: Vec<Complex> = (0..n)
        .map(|k| {
            let theta = PI * (2 * k + n + 1) as f64 / (2 * n) as f64;
            Complex::new(theta.cos(), theta.sin())
        })
        .collect();

    let (z_zeros, z_poles, norm_z): (Vec<Complex>, Vec<Complex>, Complex) = match ftype.as_str() {
        "low" => {
            let wn = wn_mat.data()[0];
            if !(0.0..1.0).contains(&wn) {
                return Err(MatroError::runtime("butter: Wn must be strictly in (0, 1)"));
            }
            let wc = 2.0 * (PI * wn * 0.5).tan();
            let zp: Vec<Complex> = proto_poles
                .iter()
                .map(|&p| {
                    let sp = p * Complex::from_real(wc);
                    (Complex::from_real(2.0) + sp) / (Complex::from_real(2.0) - sp)
                })
                .collect();
            let zz = vec![Complex::from_real(-1.0); n];
            (zz, zp, Complex::ONE) // unity at DC (z = 1)
        }
        "high" => {
            let wn = wn_mat.data()[0];
            if !(0.0..1.0).contains(&wn) {
                return Err(MatroError::runtime("butter: Wn must be strictly in (0, 1)"));
            }
            let wc = 2.0 * (PI * wn * 0.5).tan();
            let zp: Vec<Complex> = proto_poles
                .iter()
                .map(|&p| {
                    let sp = Complex::from_real(wc) / p;
                    (Complex::from_real(2.0) + sp) / (Complex::from_real(2.0) - sp)
                })
                .collect();
            let zz = vec![Complex::ONE; n];
            (zz, zp, Complex::from_real(-1.0)) // unity at Nyquist (z = -1)
        }
        "bandpass" => {
            if wn_mat.len() != 2 {
                return Err(MatroError::dimension(
                    "butter bandpass requires 2-element Wn = [w1, w2]",
                ));
            }
            let w1 = wn_mat.data()[0];
            let w2 = wn_mat.data()[1];
            if !(0.0..1.0).contains(&w1) || !(0.0..1.0).contains(&w2) || w1 >= w2 {
                return Err(MatroError::runtime(
                    "butter: bandpass requires 0 < Wn(1) < Wn(2) < 1",
                ));
            }
            let o1 = 2.0 * (PI * w1 * 0.5).tan();
            let o2 = 2.0 * (PI * w2 * 0.5).tan();
            let bw = o2 - o1;
            let w0_sq = o1 * o2;
            let mut sp = Vec::with_capacity(2 * n);
            for &p in &proto_poles {
                let half_b = p * Complex::from_real(0.5 * bw);
                let disc = (half_b * half_b - Complex::from_real(w0_sq)).sqrt();
                sp.push(half_b + disc);
                sp.push(half_b - disc);
            }
            let zp: Vec<Complex> = sp
                .into_iter()
                .map(|s| (Complex::from_real(2.0) + s) / (Complex::from_real(2.0) - s))
                .collect();
            let mut zz = vec![Complex::ONE; n];
            zz.extend(vec![Complex::from_real(-1.0); n]);
            let w_mid = 0.5 * (w1 + w2) * PI;
            (zz, zp, Complex::new(w_mid.cos(), -w_mid.sin()))
        }
        other => {
            return Err(MatroError::runtime(format!(
                "butter: unsupported filter type '{other}'"
            )));
        }
    };

    let a = poly_from_roots_c(&z_poles);
    let mut b = poly_from_roots_c(&z_zeros);
    // Normalize gain at `norm_z` (note eval_poly_at uses z^-1 powers)
    let h_num = eval_poly_at(&b, norm_z);
    let h_den = eval_poly_at(&a, norm_z);
    let gain = (h_den / h_num).norm_sq().sqrt();
    for bi in &mut b {
        *bi *= gain;
    }

    Ok(vec![
        Value::matrix(Matrix::row_vec(b)),
        Value::matrix(Matrix::row_vec(a)),
    ])
}

/// Direct Form II Transposed filter helper over a raw `f64` slice.
fn apply_iir(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
    let a0 = a[0];
    let nb = b.len();
    let na = a.len();
    let order = nb.max(na).saturating_sub(1);
    let mut z = vec![0.0; order + 1];
    let mut y = vec![0.0; x.len()];
    for (n, &xn) in x.iter().enumerate() {
        let yn = (b[0] * xn + z[0]) / a0;
        y[n] = yn;
        for k in 1..=order {
            let bk = if k < nb { b[k] } else { 0.0 };
            let ak = if k < na { a[k] } else { 0.0 };
            let z_next = if k < order { z[k] } else { 0.0 };
            z[k - 1] = bk * xn - ak * yn + z_next;
        }
    }
    y
}

/// `y = filtfilt(b, a, x)` — zero-phase forward-backward digital IIR/FIR filtering.
pub fn fn_filtfilt(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 3 {
        return Err(MatroError::arity("filtfilt", "3 (b, a, x)", args.len()));
    }
    let b = args[0].as_matrix()?.data();
    let a = args[1].as_matrix()?.data();
    let xm = args[2].as_matrix()?;
    let x = xm.data();
    if b.is_empty() || a.is_empty() || a[0] == 0.0 {
        return Err(MatroError::runtime("filtfilt: invalid filter coefficients"));
    }
    if x.is_empty() {
        return ok1(Value::empty());
    }
    let edge = (3 * (b.len().max(a.len()).saturating_sub(1))).min(x.len().saturating_sub(1));
    // Odd-reflected padding at both ends to suppress edge transients
    let mut padded = Vec::with_capacity(x.len() + 2 * edge);
    for i in (1..=edge).rev() {
        padded.push(2.0 * x[0] - x[i]);
    }
    padded.extend_from_slice(x);
    for i in 1..=edge {
        padded.push(2.0 * x[x.len() - 1] - x[x.len() - 1 - i]);
    }
    // Forward pass, reverse, backward pass, reverse, unpad
    let mut y = apply_iir(b, a, &padded);
    y.reverse();
    y = apply_iir(b, a, &y);
    y.reverse();
    let trimmed = y[edge..edge + x.len()].to_vec();
    let out = if xm.rows() == 1 {
        Matrix::row_vec(trimmed)
    } else {
        Matrix::col_vec(trimmed)
    };
    ok1(Value::matrix(out))
}

/// `[pxx, f] = pwelch(x [, win_len [, noverlap [, nfft [, fs]]]])` — Welch PSD estimator.
pub fn fn_pwelch(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 5 {
        return Err(MatroError::arity("pwelch", "1 to 5", args.len()));
    }
    let x = args[0].as_matrix()?.data();
    if x.is_empty() {
        return Err(MatroError::dimension("pwelch: input signal cannot be empty"));
    }
    let win: Vec<f64> = match args.get(1) {
        Some(v) if !v.as_matrix()?.is_empty() => {
            let wm = v.as_matrix()?;
            if wm.len() == 1 {
                let n = (wm.data()[0].round() as usize).clamp(2, x.len());
                (0..n)
                    .map(|k| 0.54 - 0.46 * (2.0 * PI * k as f64 / (n - 1) as f64).cos())
                    .collect()
            } else {
                wm.data().to_vec()
            }
        }
        _ => {
            let n = (x.len() / 4).clamp(8, 256).min(x.len());
            (0..n)
                .map(|k| {
                    if n <= 1 {
                        1.0
                    } else {
                        0.54 - 0.46 * (2.0 * PI * k as f64 / (n - 1) as f64).cos()
                    }
                })
                .collect()
        }
    };
    let wlen = win.len();
    let noverlap = match args.get(2) {
        Some(v) if !v.as_matrix()?.is_empty() => (v.as_scalar()?.round() as usize).min(wlen - 1),
        _ => wlen / 2,
    };
    let nfft = match args.get(3) {
        Some(v) if !v.as_matrix()?.is_empty() => (v.as_scalar()?.round() as usize).max(wlen),
        _ => wlen.next_power_of_two().max(256),
    };
    let fs = match args.get(4) {
        Some(v) => Some(v.as_scalar()?),
        None => None,
    };

    let step = (wlen - noverlap).max(1);
    let win_energy: f64 = win.iter().map(|&w| w * w).sum();
    let n_onesided = nfft / 2 + 1;
    let mut pxx = vec![0.0; n_onesided];
    let mut n_segs = 0usize;

    let mut start = 0usize;
    while start + wlen <= x.len() {
        let mut seg = vec![0.0; nfft];
        for i in 0..wlen {
            seg[i] = x[start + i] * win[i];
        }
        let spec = Matrix::row_vec(seg).fft();
        for k in 0..n_onesided {
            let mag2 = spec.get_linear_c(k).norm_sq();
            let scale = if k == 0 || (nfft % 2 == 0 && k == n_onesided - 1) {
                1.0
            } else {
                2.0
            };
            pxx[k] += scale * mag2;
        }
        n_segs += 1;
        start += step;
    }
    if n_segs == 0 {
        // Fallback single zero-padded segment if signal < wlen
        let mut seg = vec![0.0; nfft];
        for i in 0..x.len().min(wlen) {
            seg[i] = x[i] * win[i];
        }
        let spec = Matrix::row_vec(seg).fft();
        for k in 0..n_onesided {
            let scale = if k == 0 || (nfft % 2 == 0 && k == n_onesided - 1) {
                1.0
            } else {
                2.0
            };
            pxx[k] = scale * spec.get_linear_c(k).norm_sq();
        }
        n_segs = 1;
    }

    let norm = match fs {
        Some(f) => f * win_energy * n_segs as f64,
        None => 2.0 * PI * win_energy * n_segs as f64,
    };
    for v in &mut pxx {
        *v /= norm.max(1e-30);
    }

    let freqs: Vec<f64> = (0..n_onesided)
        .map(|k| match fs {
            Some(f) => (k as f64) * f / (nfft as f64),
            None => (k as f64) * 2.0 * PI / (nfft as f64),
        })
        .collect();

    if nargout >= 2 {
        Ok(vec![
            Value::matrix(Matrix::col_vec(pxx)),
            Value::matrix(Matrix::col_vec(freqs)),
        ])
    } else {
        ok1(Value::matrix(Matrix::col_vec(pxx)))
    }
}
