//! Pure-Rust Signal Processing Toolbox intrinsics:
//! `fftshift`, `ifftshift`, `hann`, `hamming`, `blackman`, `xcorr`,
//! `findpeaks`, `butter`, `filtfilt`, `pwelch`.

use std::f64::consts::PI;

use matro_types::{Complex, MatroError, Matrix, Result, Value};

fn ok1(v: Value) -> Result<Vec<Value>> {
    Ok(vec![v])
}

/// `fftshift(x [, dim])` / `ifftshift(x [, dim])` — N-D zero-frequency shift.
pub fn fn_fftshift(args: &[Value], inverse: bool) -> Result<Vec<Value>> {
    let fname = if inverse { "ifftshift" } else { "fftshift" };
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity(fname, "1 or 2 (X [, dim])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    if m.is_empty() {
        return ok1(Value::matrix(m));
    }
    let dims = m.dims();
    let shift_1d = |n: usize| -> usize { if inverse { (n + 1) / 2 } else { n / 2 } };
    let axes: Vec<usize> = if args.len() == 2 {
        let d = args[1].as_scalar()?.round() as isize;
        if d < 1 {
            return Err(MatroError::runtime(format!("{fname}: dim must be >= 1")));
        }
        let ax = (d - 1) as usize;
        if ax >= dims.len() {
            return ok1(Value::matrix(m));
        }
        vec![ax]
    } else {
        (0..dims.len()).collect()
    };

    let mut cur_re = m.data().to_vec();
    let mut cur_im = m.imag_data().map(|s| s.to_vec());
    let total = cur_re.len();
    for ax in axes {
        let axis_len = dims[ax];
        if axis_len <= 1 {
            continue;
        }
        let sh = shift_1d(axis_len);
        let inner: usize = dims[..ax].iter().product();
        let outer: usize = dims[ax + 1..].iter().product();
        let mut next_re = vec![0.0; total];
        let mut next_im = cur_im.as_ref().map(|_| vec![0.0; total]);
        for o in 0..outer {
            let block = o * axis_len * inner;
            for k in 0..axis_len {
                let src_k = (k + axis_len - sh) % axis_len;
                let dst = block + k * inner;
                let src = block + src_k * inner;
                next_re[dst..dst + inner].copy_from_slice(&cur_re[src..src + inner]);
                if let (Some(si), Some(di)) = (cur_im.as_ref(), next_im.as_mut()) {
                    di[dst..dst + inner].copy_from_slice(&si[src..src + inner]);
                }
            }
        }
        cur_re = next_re;
        cur_im = next_im;
    }
    let out = if let Some(im) = cur_im {
        Matrix::from_nd_complex_col_major(dims, cur_re, im)?.normalize()
    } else {
        Matrix::from_nd_col_major(dims, cur_re)?
    };
    ok1(Value::matrix(out))
}

/// Window functions: `hann(N [, 'symmetric'|'periodic'])`, `hamming(...)`, `blackman(...)`.
pub fn fn_window(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity(name, "1 or 2 (N [, sflag])", args.len()));
    }
    let n = args[0].as_scalar()?.max(0.0).round() as usize;
    let periodic = args
        .get(1)
        .and_then(crate::value_as_text)
        .is_some_and(|s| s.eq_ignore_ascii_case("periodic"));
    if n == 0 {
        return ok1(Value::empty());
    }
    if n == 1 {
        return ok1(Value::scalar(1.0));
    }
    let denom = if periodic { n as f64 } else { (n - 1) as f64 };
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

/// `[r, lags] = xcorr(x [, y] [, maxlag] [, scaleopt])` and `xcov(...)`.
pub fn fn_xcorr_or_xcov(args: &[Value], nargout: usize, subtract_mean: bool) -> Result<Vec<Value>> {
    let fname = if subtract_mean { "xcov" } else { "xcorr" };
    if args.is_empty() || args.len() > 4 {
        return Err(MatroError::arity(fname, "1 to 4", args.len()));
    }
    let xm = args[0].to_dense_matrix()?;
    let is_row = xm.rows() == 1 && xm.cols() > 1;
    let mut y_opt: Option<Matrix> = None;
    let mut maxlag_opt: Option<usize> = None;
    let mut scaleopt = "none".to_string();

    for (idx, a) in args[1..].iter().enumerate() {
        if let Some(s) = crate::value_as_text(a) {
            scaleopt = s.to_ascii_lowercase();
        } else {
            let m = a.to_dense_matrix()?;
            if idx == 0 && m.len() == 1 && xm.len() > 1 && args.len() <= 3 {
                // `xcorr(x, maxlag)` or `xcorr(x, maxlag, scaleopt)`
                let next_is_num = args.get(2).is_some_and(|v| crate::value_as_text(v).is_none());
                if !next_is_num {
                    maxlag_opt = Some(m.data()[0].max(0.0).round() as usize);
                    continue;
                }
            }
            if y_opt.is_none() && idx == 0 {
                y_opt = Some(m);
            } else if maxlag_opt.is_none() && !m.is_empty() {
                maxlag_opt = Some(m.data()[0].max(0.0).round() as usize);
            }
        }
    }

    let mut x_c: Vec<Complex> = (0..xm.len()).map(|i| xm.get_linear_c(i)).collect();
    let mut y_c: Vec<Complex> = match &y_opt {
        Some(ym) => (0..ym.len()).map(|i| ym.get_linear_c(i)).collect(),
        None => x_c.clone(),
    };
    let n = x_c.len().max(y_c.len());
    if n == 0 {
        return ok1(Value::empty());
    }
    if subtract_mean {
        let mx = x_c.iter().fold(Complex::ZERO, |a, &b| a + b) * Complex::from_real(1.0 / x_c.len().max(1) as f64);
        let my = y_c.iter().fold(Complex::ZERO, |a, &b| a + b) * Complex::from_real(1.0 / y_c.len().max(1) as f64);
        for v in &mut x_c { *v = *v - mx; }
        for v in &mut y_c { *v = *v - my; }
    }
    let maxlag = maxlag_opt.unwrap_or(n - 1);
    let m = 2 * maxlag + 1;
    let mut r = vec![Complex::ZERO; m];
    let mut lags = Vec::with_capacity(m);
    for idx in 0..m {
        let lag = idx as isize - maxlag as isize;
        lags.push(lag as f64);
        let mut sum = Complex::ZERO;
        for i in 0..x_c.len() {
            let j = i as isize - lag;
            if j >= 0 && (j as usize) < y_c.len() {
                sum = sum + x_c[i] * y_c[j as usize].conj();
            }
        }
        r[idx] = sum;
    }

    match scaleopt.as_str() {
        "biased" => {
            let inv = 1.0 / n as f64;
            for v in &mut r { *v = *v * Complex::from_real(inv); }
        }
        "unbiased" => {
            for (idx, v) in r.iter_mut().enumerate() {
                let abslag = (idx as isize - maxlag as isize).unsigned_abs();
                let denom = n.saturating_sub(abslag);
                if denom > 0 {
                    *v = *v * Complex::from_real(1.0 / denom as f64);
                } else {
                    *v = Complex::ZERO;
                }
            }
        }
        "normalized" | "coeff" => {
            let ex: f64 = x_c.iter().map(|z| z.norm_sq()).sum();
            let ey: f64 = y_c.iter().map(|z| z.norm_sq()).sum();
            let denom = (ex * ey).sqrt();
            if denom > 0.0 {
                let inv = 1.0 / denom;
                for v in &mut r { *v = *v * Complex::from_real(inv); }
            }
        }
        _ => {}
    }

    let r_mat = if is_row {
        Matrix::complex_row_vec(r).normalize()
    } else {
        Matrix::complex_col_vec(r).normalize()
    };
    if nargout >= 2 {
        Ok(vec![Value::matrix(r_mat), Value::matrix(Matrix::row_vec(lags))])
    } else {
        ok1(Value::matrix(r_mat))
    }
}

/// `[pks, locs, w, p] = findpeaks(y [, x|Fs] [, NV...])`
pub fn fn_findpeaks(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("findpeaks", ">= 1", 0));
    }
    let xm = args[0].to_dense_matrix()?;
    let data = xm.data();
    let n = data.len();
    let mut x_coords: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let mut i = 1;
    if i < args.len() && crate::value_as_text(&args[i]).is_none() {
        let loc_m = args[i].to_dense_matrix()?;
        if loc_m.len() == 1 {
            let fs = loc_m.data()[0];
            if fs > 0.0 {
                x_coords = (0..n).map(|k| k as f64 / fs).collect();
            }
        } else if loc_m.len() == n {
            x_coords = loc_m.data().to_vec();
        }
        i += 1;
    }

    let mut min_height = f64::NEG_INFINITY;
    let mut min_dist = 0.0f64;
    let mut min_prom = 0.0f64;
    let mut threshold = 0.0f64;
    let mut npeaks: Option<usize> = None;
    let mut sort_str = "none".to_string();

    while i + 1 < args.len() {
        let key = args[i].as_str()?.to_ascii_lowercase();
        match key.as_str() {
            "minpeakheight" => min_height = args[i + 1].as_scalar()?,
            "minpeakdistance" => min_dist = args[i + 1].as_scalar()?.max(0.0),
            "minpeakprominence" => min_prom = args[i + 1].as_scalar()?.max(0.0),
            "threshold" => threshold = args[i + 1].as_scalar()?.max(0.0),
            "npeaks" => npeaks = Some(args[i + 1].as_scalar()?.max(0.0).round() as usize),
            "sortstr" => sort_str = args[i + 1].as_str()?.to_ascii_lowercase(),
            _ => {}
        }
        i += 2;
    }

    // Detect true local maxima, properly handling flat plateaus (no rising-stair false peaks!)
    let mut raw_peaks: Vec<usize> = Vec::new();
    if n >= 3 {
        let mut k = 1usize;
        while k < n - 1 {
            if data[k] > data[k - 1] {
                let mut end_k = k;
                while end_k + 1 < n - 1 && data[end_k + 1] == data[k] {
                    end_k += 1;
                }
                if data[end_k] > data[end_k + 1] {
                    let diff_l = data[k] - data[k - 1];
                    let diff_r = data[end_k] - data[end_k + 1];
                    if data[k] >= min_height && diff_l >= threshold && diff_r >= threshold {
                        raw_peaks.push(k);
                    }
                }
                k = end_k + 1;
            } else {
                k += 1;
            }
        }
    }

    // Compute prominence and half-prominence width for each candidate peak
    struct PeakInfo {
        idx: usize,
        val: f64,
        loc: f64,
        prom: f64,
        width: f64,
    }
    let mut cands: Vec<PeakInfo> = Vec::with_capacity(raw_peaks.len());
    for &idx in &raw_peaks {
        let val = data[idx];
        // Scan left until a higher sample or start of signal
        let mut min_l = val;
        let mut j = idx;
        while j > 0 {
            j -= 1;
            if data[j] < min_l {
                min_l = data[j];
            }
            if data[j] > val {
                break;
            }
        }
        // Scan right until a higher sample or end of signal
        let mut min_r = val;
        let mut j = idx;
        while j + 1 < n {
            j += 1;
            if data[j] < min_r {
                min_r = data[j];
            }
            if data[j] > val {
                break;
            }
        }
        let ref_base = min_l.max(min_r);
        let prom = (val - ref_base).max(0.0);
        if prom < min_prom {
            continue;
        }
        // Half-prominence width via linear interpolation
        let half_lvl = val - 0.5 * prom;
        let mut x_left = x_coords[0];
        for jl in (0..idx).rev() {
            if data[jl] <= half_lvl {
                let d = data[jl + 1] - data[jl];
                let frac = if d.abs() > 1e-15 { (half_lvl - data[jl]) / d } else { 0.0 };
                x_left = x_coords[jl] + frac * (x_coords[jl + 1] - x_coords[jl]);
                break;
            }
        }
        let mut x_right = x_coords[n - 1];
        for jr in (idx + 1)..n {
            if data[jr] <= half_lvl {
                let d = data[jr - 1] - data[jr];
                let frac = if d.abs() > 1e-15 { (half_lvl - data[jr]) / d } else { 0.0 };
                x_right = x_coords[jr] - frac * (x_coords[jr] - x_coords[jr - 1]);
                break;
            }
        }
        cands.push(PeakInfo {
            idx,
            val,
            loc: x_coords[idx],
            prom,
            width: (x_right - x_left).max(0.0),
        });
    }

    // Enforce MinPeakDistance greedily in descending peak height order
    let mut kept: Vec<PeakInfo> = if min_dist > 0.0 {
        cands.sort_by(|a, b| b.val.total_cmp(&a.val));
        let mut chosen: Vec<PeakInfo> = Vec::new();
        for p in cands {
            if chosen.iter().all(|c| (p.loc - c.loc).abs() >= min_dist) {
                chosen.push(p);
            }
        }
        chosen.sort_by_key(|p| p.idx);
        chosen
    } else {
        cands
    };

    match sort_str.as_str() {
        "descend" => kept.sort_by(|a, b| b.val.total_cmp(&a.val)),
        "ascend" => kept.sort_by(|a, b| a.val.total_cmp(&b.val)),
        _ => {}
    }
    if let Some(np) = npeaks {
        kept.truncate(np);
    }

    let is_row = xm.rows() == 1;
    let pack = |v: Vec<f64>| {
        if is_row {
            Value::matrix(Matrix::row_vec(v))
        } else {
            Value::matrix(Matrix::col_vec(v))
        }
    };
    let mut out = vec![pack(kept.iter().map(|p| p.val).collect())];
    if nargout >= 2 {
        out.push(pack(kept.iter().map(|p| p.loc).collect()));
    }
    if nargout >= 3 {
        out.push(pack(kept.iter().map(|p| p.width).collect()));
    }
    if nargout >= 4 {
        out.push(pack(kept.iter().map(|p| p.prom).collect()));
    }
    Ok(out)
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

/// `[b, a]` or `[z, p, k] = butter(n, Wn [, ftype] [, 's'])` — Butterworth filter design.
pub fn fn_butter(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 4 {
        return Err(MatroError::arity("butter", "2 to 4 (n, Wn [, ftype [, 's']])", args.len()));
    }
    let n = args[0].as_scalar()?.round() as usize;
    if n == 0 || n > 16 {
        return Err(MatroError::runtime("butter: filter order n must be in 1..=16"));
    }
    let wn_mat = args[1].to_dense_matrix()?;
    let mut ftype: Option<String> = None;
    let mut analog = false;
    for a in &args[2..] {
        if let Some(s) = crate::value_as_text(a) {
            let low = s.to_ascii_lowercase();
            if low == "s" {
                analog = true;
            } else {
                ftype = Some(low);
            }
        }
    }
    let ftype = ftype.unwrap_or_else(|| {
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

    if analog {
        let (zs, ps, k_gain) = crate::signal_ext::transform_analog_zpk(
            &[],
            &proto_poles,
            1.0,
            wn_mat.data(),
            &ftype,
            true,
        )?;
        if nargout >= 3 {
            return Ok(vec![
                Value::matrix(Matrix::complex_col_vec(zs).normalize()),
                Value::matrix(Matrix::complex_col_vec(ps).normalize()),
                Value::scalar(k_gain),
            ]);
        }
        let a = poly_from_roots_c(&ps);
        let mut b = poly_from_roots_c(&zs);
        for bi in &mut b {
            *bi *= k_gain;
        }
        return Ok(vec![
            Value::matrix(Matrix::row_vec(b)),
            Value::matrix(Matrix::row_vec(a)),
        ]);
    }

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
            (zz, zp, Complex::ONE)
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
            (zz, zp, Complex::from_real(-1.0))
        }
        "bandpass" => {
            if wn_mat.len() != 2 {
                return Err(MatroError::dimension(
                    "butter bandpass requires 2-element Wn = [w1, w2]",
                ));
            }
            let (w1, w2) = (wn_mat.data()[0], wn_mat.data()[1]);
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
        "stop" => {
            if wn_mat.len() != 2 {
                return Err(MatroError::dimension(
                    "butter stop requires 2-element Wn = [w1, w2]",
                ));
            }
            let (w1, w2) = (wn_mat.data()[0], wn_mat.data()[1]);
            if !(0.0..1.0).contains(&w1) || !(0.0..1.0).contains(&w2) || w1 >= w2 {
                return Err(MatroError::runtime(
                    "butter: stop requires 0 < Wn(1) < Wn(2) < 1",
                ));
            }
            let o1 = 2.0 * (PI * w1 * 0.5).tan();
            let o2 = 2.0 * (PI * w2 * 0.5).tan();
            let bw = o2 - o1;
            let w0_sq = o1 * o2;
            let mut sp = Vec::with_capacity(2 * n);
            for &p in &proto_poles {
                let half_b = Complex::from_real(0.5 * bw) / p;
                let disc = (half_b * half_b - Complex::from_real(w0_sq)).sqrt();
                sp.push(half_b + disc);
                sp.push(half_b - disc);
            }
            let zp: Vec<Complex> = sp
                .into_iter()
                .map(|s| (Complex::from_real(2.0) + s) / (Complex::from_real(2.0) - s))
                .collect();
            let w0 = w0_sq.sqrt();
            let sz_pos = Complex::new(0.0, w0);
            let sz_neg = Complex::new(0.0, -w0);
            let z_pos = (Complex::from_real(2.0) + sz_pos) / (Complex::from_real(2.0) - sz_pos);
            let z_neg = (Complex::from_real(2.0) + sz_neg) / (Complex::from_real(2.0) - sz_neg);
            let mut zz = Vec::with_capacity(2 * n);
            for _ in 0..n {
                zz.push(z_pos);
                zz.push(z_neg);
            }
            (zz, zp, Complex::ONE) // unity at DC
        }
        other => {
            return Err(MatroError::runtime(format!(
                "butter: unsupported filter type '{other}'"
            )));
        }
    };

    let a = poly_from_roots_c(&z_poles);
    let mut b = poly_from_roots_c(&z_zeros);
    let h_num = eval_poly_at(&b, norm_z);
    let h_den = eval_poly_at(&a, norm_z);
    let gain = (h_den / h_num).norm_sq().sqrt();
    if nargout >= 3 {
        return Ok(vec![
            Value::matrix(Matrix::complex_col_vec(z_zeros).normalize()),
            Value::matrix(Matrix::complex_col_vec(z_poles).normalize()),
            Value::scalar(gain),
        ]);
    }
    for bi in &mut b {
        *bi *= gain;
    }

    Ok(vec![
        Value::matrix(Matrix::row_vec(b)),
        Value::matrix(Matrix::row_vec(a)),
    ])
}

/// Direct Form II Transposed filter helper over a raw `f64` slice.
pub fn apply_iir(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
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

fn filtfilt_1d(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }
    let edge = (3 * (b.len().max(a.len()).saturating_sub(1))).min(x.len().saturating_sub(1));
    let mut padded = Vec::with_capacity(x.len() + 2 * edge);
    for i in (1..=edge).rev() {
        padded.push(2.0 * x[0] - x[i]);
    }
    padded.extend_from_slice(x);
    for i in 1..=edge {
        padded.push(2.0 * x[x.len() - 1] - x[x.len() - 1 - i]);
    }
    let mut y = apply_iir(b, a, &padded);
    y.reverse();
    y = apply_iir(b, a, &y);
    y.reverse();
    y[edge..edge + x.len()].to_vec()
}

/// `y = filtfilt(b, a, x)` or `y = filtfilt(sos, g, x)` — zero-phase forward-backward digital IIR/FIR filtering.
pub fn fn_filtfilt(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 3 {
        return Err(MatroError::arity("filtfilt", "3 (b, a, x) or (sos, g, x)", args.len()));
    }
    let bm = args[0].to_dense_matrix()?;
    let am = args[1].to_dense_matrix()?;
    let xm = args[2].to_dense_matrix()?;
    if xm.is_empty() {
        return ok1(Value::matrix(xm));
    }

    // SOS overload: `filtfilt(sos, g, x)` when `sos` is L x 6 and `g` is scalar or gain vector
    let is_sos = bm.cols() == 6 && (bm.rows() > 1 || (am.len() == 1 && bm.rows() == 1 && bm.get(0, 3) != 0.0));
    let (b_vec, a_vec): (Vec<f64>, Vec<f64>) = if is_sos {
        let (mut b_sos, a_sos) = crate::signal_ext::sos_to_tf(&bm)?;
        let g = am.data().iter().product::<f64>();
        for bi in &mut b_sos {
            *bi *= g;
        }
        (b_sos, a_sos)
    } else {
        (bm.data().to_vec(), am.data().to_vec())
    };
    if b_vec.is_empty() || a_vec.is_empty() || a_vec[0] == 0.0 {
        return Err(MatroError::runtime("filtfilt: invalid filter coefficients"));
    }

    if xm.rows() == 1 || xm.cols() == 1 {
        let trimmed = filtfilt_1d(&b_vec, &a_vec, xm.data());
        let out = if xm.rows() == 1 {
            Matrix::row_vec(trimmed)
        } else {
            Matrix::col_vec(trimmed)
        };
        return ok1(Value::matrix(out));
    }

    let (nr, nc) = (xm.rows(), xm.cols());
    let mut out = Matrix::zeros(nr, nc);
    for c in 0..nc {
        let col: Vec<f64> = (0..nr).map(|r| xm.get(r, c)).collect();
        let filtered = filtfilt_1d(&b_vec, &a_vec, &col);
        for r in 0..nr {
            out.set(r, c, filtered[r]);
        }
    }
    ok1(Value::matrix(out))
}

/// `[pxx, f] = periodogram(x [, win [, nfft [, fs]]])` — Periodogram PSD estimator.
pub fn fn_periodogram(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 4 {
        return Err(MatroError::arity("periodogram", "1 to 4 (x [, win [, nfft [, fs]]])", args.len()));
    }
    let xm = args[0].to_dense_matrix()?;
    let n = xm.len();
    if n == 0 {
        return Err(MatroError::dimension("periodogram: input signal cannot be empty"));
    }
    let win: Vec<f64> = match args.get(1) {
        Some(v) if !v.to_dense_matrix()?.is_empty() => v.to_dense_matrix()?.data().to_vec(),
        _ => vec![1.0; n],
    };
    // Delegate directly to pwelch with single full-length segment (noverlap = 0)
    let mut pw_args = vec![
        Value::matrix(xm),
        Value::matrix(Matrix::col_vec(win)),
        Value::scalar(0.0),
    ];
    if let Some(nfft) = args.get(2) {
        pw_args.push(nfft.clone());
    } else {
        pw_args.push(Value::scalar(n.next_power_of_two().max(256) as f64));
    }
    if let Some(fs) = args.get(3) {
        pw_args.push(fs.clone());
    }
    fn_pwelch(&pw_args, nargout)
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
