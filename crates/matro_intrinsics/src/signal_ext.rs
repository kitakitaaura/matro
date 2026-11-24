//! Signal Processing Toolbox extensions (pure zero-dependency Rust):
//! - IIR/FIR design: `cheby1`, `cheby2`, `fir1`, `freqz`, `grpdelay`
//! - Time-frequency & analytic signal: `spectrogram`, `stft`, `hilbert`, `envelope`
//! - Filtering & resampling: `medfilt1`, `sgolayfilt`, `detrend`, `downsample`, `upsample`, `decimate`, `resample`
//! - Windows: `bartlett`, `kaiser`, `flattopwin`

use std::f64::consts::PI;

use matro_types::{Complex, MatroError, Matrix, PlotFigure, PlotKind, Result, Value};

fn ok1(v: Value) -> Result<Vec<Value>> {
    Ok(vec![v])
}

fn poly_from_roots_c(roots: &[Complex]) -> Vec<Complex> {
    let mut p = vec![Complex::ONE];
    for &r in roots {
        let mut next = vec![Complex::ZERO; p.len() + 1];
        for i in 0..p.len() {
            next[i] = next[i] + p[i];
            next[i + 1] = next[i + 1] - p[i] * r;
        }
        p = next;
    }
    p
}

fn eval_poly_c(p: &[f64], z: Complex) -> Complex {
    let mut acc = Complex::ZERO;
    for &c in p {
        acc = acc * z + Complex::from_real(c);
    }
    acc
}

/// Bilinear transform from analog (zeros, poles, gain) with prewarped Fs=2 to digital `(b, a)`.
fn bilinear_zpk(zeros: &[Complex], poles: &[Complex], gain: f64) -> (Vec<f64>, Vec<f64>) {
    let fs2 = 2.0;
    let mut z_d = Vec::with_capacity(poles.len());
    let mut p_d = Vec::with_capacity(poles.len());
    let mut num_scale = Complex::from_real(gain);
    let mut den_scale = Complex::ONE;

    for &z in zeros {
        z_d.push((Complex::from_real(fs2) + z) / (Complex::from_real(fs2) - z));
        num_scale = num_scale * (Complex::from_real(fs2) - z);
    }
    for &p in poles {
        p_d.push((Complex::from_real(fs2) + p) / (Complex::from_real(fs2) - p));
        den_scale = den_scale * (Complex::from_real(fs2) - p);
    }
    // Zeros at s = inf map to z = -1
    for _ in zeros.len()..poles.len() {
        z_d.push(Complex::from_real(-1.0));
    }
    let k_d = (num_scale / den_scale).re;
    let b_c = poly_from_roots_c(&z_d);
    let a_c = poly_from_roots_c(&p_d);
    let b: Vec<f64> = b_c.iter().map(|c| c.re * k_d).collect();
    let a: Vec<f64> = a_c.iter().map(|c| c.re).collect();
    (b, a)
}

/// Apply analog lowpass-to-lowpass / highpass / bandpass frequency transform to normalized prototype.
fn transform_analog_zpk(
    proto_z: &[Complex],
    proto_p: &[Complex],
    proto_k: f64,
    wn: &[f64],
    ftype: &str,
) -> Result<(Vec<Complex>, Vec<Complex>, f64)> {
    let n = proto_p.len();
    let m = proto_z.len();
    let prewarp = |w: f64| -> Result<f64> {
        if w <= 0.0 || w >= 1.0 {
            return Err(MatroError::runtime("Cutoff frequency Wn must be in (0, 1)"));
        }
        Ok(2.0 * (0.5 * PI * w).tan())
    };
    match ftype {
        "low" => {
            let wo = prewarp(wn[0])?;
            let zs: Vec<Complex> = proto_z.iter().map(|&z| z * Complex::from_real(wo)).collect();
            let ps: Vec<Complex> = proto_p.iter().map(|&p| p * Complex::from_real(wo)).collect();
            let k = proto_k * wo.powi((n - m) as i32);
            Ok((zs, ps, k))
        }
        "high" => {
            let wo = prewarp(wn[0])?;
            let mut zs: Vec<Complex> = proto_z.iter().map(|&z| Complex::from_real(wo) / z).collect();
            for _ in m..n {
                zs.push(Complex::ZERO);
            }
            let ps: Vec<Complex> = proto_p.iter().map(|&p| Complex::from_real(wo) / p).collect();
            let prod_z = proto_z.iter().fold(Complex::ONE, |acc, &z| acc * (-z));
            let prod_p = proto_p.iter().fold(Complex::ONE, |acc, &p| acc * (-p));
            let k = proto_k * (prod_z / prod_p).re;
            Ok((zs, ps, k))
        }
        "bandpass" => {
            if wn.len() != 2 || wn[0] >= wn[1] {
                return Err(MatroError::runtime("bandpass requires Wn = [w1, w2] with 0 < w1 < w2 < 1"));
            }
            let w1 = prewarp(wn[0])?;
            let w2 = prewarp(wn[1])?;
            let bw = w2 - w1;
            let w0 = (w1 * w2).sqrt();
            let mut zs = Vec::new();
            for &z in proto_z {
                let b = z * Complex::from_real(0.5 * bw);
                let disc = (b * b - Complex::from_real(w0 * w0)).sqrt();
                zs.push(b + disc);
                zs.push(b - disc);
            }
            for _ in m..n {
                zs.push(Complex::ZERO);
            }
            let mut ps = Vec::new();
            for &p in proto_p {
                let b = p * Complex::from_real(0.5 * bw);
                let disc = (b * b - Complex::from_real(w0 * w0)).sqrt();
                ps.push(b + disc);
                ps.push(b - disc);
            }
            let k = proto_k * bw.powi((n - m) as i32);
            Ok((zs, ps, k))
        }
        other => Err(MatroError::runtime(format!("Unsupported filter type '{other}'"))),
    }
}

/// `[b, a] = cheby1(n, Rp, Wn [, ftype])` — Chebyshev Type I digital IIR filter.
pub fn fn_cheby1(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 3 || args.len() > 4 {
        return Err(MatroError::arity("cheby1", "3 or 4 (n, Rp, Wn [, ftype])", args.len()));
    }
    let n = args[0].as_scalar()?.round().max(1.0) as usize;
    let rp = args[1].as_scalar()?;
    if rp <= 0.0 {
        return Err(MatroError::runtime("cheby1: passband ripple Rp must be > 0 dB"));
    }
    let wn_m = args[2].to_dense_matrix()?;
    let wn = wn_m.data();
    let ftype = args
        .get(3)
        .and_then(|v| v.as_str().ok())
        .map(str::to_ascii_lowercase)
        .unwrap_or_else(|| if wn.len() == 2 { "bandpass".into() } else { "low".into() });

    let eps = (10.0f64.powf(0.1 * rp) - 1.0).sqrt();
    let mu = (1.0 / eps).asinh() / (n as f64);
    let mut poles = Vec::with_capacity(n);
    for k in 1..=n {
        let theta = PI * (2 * k - 1) as f64 / (2.0 * n as f64);
        poles.push(Complex::new(-mu.sinh() * theta.sin(), mu.cosh() * theta.cos()));
    }
    let prod_p = poles.iter().fold(Complex::ONE, |acc, &p| acc * (-p));
    let k0 = prod_p.re / if n % 2 == 0 { (1.0 + eps * eps).sqrt() } else { 1.0 };

    let (zs, ps, k) = transform_analog_zpk(&[], &poles, k0, wn, &ftype)?;
    let (b, a) = bilinear_zpk(&zs, &ps, k);
    Ok(vec![
        Value::matrix(Matrix::row_vec(b)),
        Value::matrix(Matrix::row_vec(a)),
    ])
}

/// `[b, a] = cheby2(n, Rs, Wn [, ftype])` — Chebyshev Type II digital IIR filter.
pub fn fn_cheby2(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 3 || args.len() > 4 {
        return Err(MatroError::arity("cheby2", "3 or 4 (n, Rs, Wn [, ftype])", args.len()));
    }
    let n = args[0].as_scalar()?.round().max(1.0) as usize;
    let rs = args[1].as_scalar()?;
    if rs <= 0.0 {
        return Err(MatroError::runtime("cheby2: stopband attenuation Rs must be > 0 dB"));
    }
    let wn_m = args[2].to_dense_matrix()?;
    let wn = wn_m.data();
    let ftype = args
        .get(3)
        .and_then(|v| v.as_str().ok())
        .map(str::to_ascii_lowercase)
        .unwrap_or_else(|| if wn.len() == 2 { "bandpass".into() } else { "low".into() });

    let eps = 1.0 / (10.0f64.powf(0.1 * rs) - 1.0).sqrt();
    let mu = (1.0 / eps).asinh() / (n as f64);
    let mut zeros = Vec::new();
    let mut poles = Vec::with_capacity(n);
    for k in 1..=n {
        let theta = PI * (2 * k - 1) as f64 / (2.0 * n as f64);
        let p_c1 = Complex::new(-mu.sinh() * theta.sin(), mu.cosh() * theta.cos());
        poles.push(Complex::ONE / p_c1);
        let c = theta.cos();
        if c.abs() > 1e-12 {
            zeros.push(Complex::new(0.0, 1.0 / c));
        }
    }
    let prod_z = zeros.iter().fold(Complex::ONE, |acc, &z| acc * (-z));
    let prod_p = poles.iter().fold(Complex::ONE, |acc, &p| acc * (-p));
    let k0 = (prod_p / prod_z).re;

    let (zs, ps, k) = transform_analog_zpk(&zeros, &poles, k0, wn, &ftype)?;
    let (b, a) = bilinear_zpk(&zs, &ps, k);
    Ok(vec![
        Value::matrix(Matrix::row_vec(b)),
        Value::matrix(Matrix::row_vec(a)),
    ])
}

/// `b = fir1(n, Wn [, ftype])` — Windowed-sinc (Hamming) FIR filter design.
pub fn fn_fir1(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity("fir1", "2 or 3 (n, Wn [, ftype])", args.len()));
    }
    let mut n = args[0].as_scalar()?.round().max(1.0) as usize;
    let wn_m = args[1].to_dense_matrix()?;
    let wn = wn_m.data();
    let ftype = args
        .get(2)
        .and_then(|v| v.as_str().ok())
        .map(str::to_ascii_lowercase)
        .unwrap_or_else(|| if wn.len() == 2 { "bandpass".into() } else { "low".into() });

    // Type I FIR (even order) required for highpass & bandstop
    if (ftype == "high" || ftype == "stop") && n % 2 == 1 {
        n += 1;
    }
    let len = n + 1;
    let m = 0.5 * (n as f64);
    let sinc = |x: f64| -> f64 {
        if x.abs() < 1e-15 { 1.0 } else { (PI * x).sin() / (PI * x) }
    };
    let lp_kernel = |fc: f64| -> Vec<f64> {
        (0..len)
            .map(|i| {
                let k = (i as f64) - m;
                let w = 0.54 - 0.46 * (2.0 * PI * (i as f64) / (n as f64)).cos();
                fc * sinc(fc * k) * w
            })
            .collect()
    };

    let mut h = match ftype.as_str() {
        "low" => lp_kernel(wn[0]),
        "high" => {
            let mut lp = lp_kernel(wn[0]);
            let s: f64 = lp.iter().sum();
            for v in &mut lp {
                *v = -*v / s;
            }
            lp[n / 2] += 1.0;
            lp
        }
        "bandpass" => {
            if wn.len() != 2 {
                return Err(MatroError::runtime("fir1 bandpass requires Wn = [w1, w2]"));
            }
            let h2 = lp_kernel(wn[1]);
            let h1 = lp_kernel(wn[0]);
            h2.iter().zip(h1.iter()).map(|(a, b)| a - b).collect()
        }
        "stop" => {
            if wn.len() != 2 {
                return Err(MatroError::runtime("fir1 stop requires Wn = [w1, w2]"));
            }
            let h2 = lp_kernel(wn[1]);
            let h1 = lp_kernel(wn[0]);
            let mut out: Vec<f64> = h1.iter().zip(h2.iter()).map(|(a, b)| a - b).collect();
            out[n / 2] += 1.0;
            out
        }
        other => return Err(MatroError::runtime(format!("fir1: unknown filter type '{other}'"))),
    };

    // Normalize gain at center of passband (DC for low/stop, Nyquist for high, midpoint for bandpass)
    let f0 = match ftype.as_str() {
        "high" => 1.0,
        "bandpass" => 0.5 * (wn[0] + wn[1]),
        _ => 0.0,
    };
    let mut gain = Complex::ZERO;
    for (i, &c) in h.iter().enumerate() {
        let ang = -PI * f0 * (i as f64);
        gain = gain + Complex::new(c * ang.cos(), c * ang.sin());
    }
    let g_abs = gain.abs();
    if g_abs > 1e-15 {
        for v in &mut h {
            *v /= g_abs;
        }
    }
    ok1(Value::matrix(Matrix::row_vec(h)))
}

/// `[H, w] = freqz(b [, a [, n [, fs]]])` — Digital filter complex frequency response.
pub fn fn_freqz(args: &[Value], nargout: usize, fig: &mut PlotFigure) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 4 {
        return Err(MatroError::arity("freqz", "1 to 4 (b [, a [, n [, fs]]])", args.len()));
    }
    let b = args[0].to_dense_matrix()?;
    let a = if args.len() >= 2 {
        args[1].to_dense_matrix()?
    } else {
        Matrix::scalar(1.0)
    };
    let n = if args.len() >= 3 {
        args[2].as_scalar()?.round().max(2.0) as usize
    } else {
        512
    };
    let fs = if args.len() >= 4 {
        Some(args[3].as_scalar()?)
    } else {
        None
    };

    let mut h_vals = Vec::with_capacity(n);
    let mut w_vals = Vec::with_capacity(n);
    for k in 0..n {
        let w = PI * (k as f64) / (n as f64);
        let z_inv = Complex::new(w.cos(), -w.sin());
        // Evaluate sum b[j] z^{-j} / sum a[j] z^{-j}
        let mut num = Complex::ZERO;
        let mut zp = Complex::ONE;
        for &c in b.data() {
            num = num + Complex::from_real(c) * zp;
            zp = zp * z_inv;
        }
        let mut den = Complex::ZERO;
        let mut zp = Complex::ONE;
        for &c in a.data() {
            den = den + Complex::from_real(c) * zp;
            zp = zp * z_inv;
        }
        h_vals.push(num / den);
        w_vals.push(if let Some(sr) = fs { w * sr / (2.0 * PI) } else { w });
    }

    if nargout == 0 {
        let mag_db: Vec<f64> = h_vals
            .iter()
            .map(|c| 20.0 * c.abs().max(1e-12).log10())
            .collect();
        let p = fig.panel_mut();
        p.clear();
        p.add_series(PlotKind::Line, w_vals, mag_db, None);
        p.title = "Frequency Response (freqz)".to_string();
        p.xlabel = if fs.is_some() { "Frequency (Hz)" } else { "Normalized Frequency (rad/sample)" }.to_string();
        p.ylabel = "Magnitude (dB)".to_string();
        return Ok(vec![]);
    }

    let mut out = vec![Value::matrix(Matrix::complex_col_vec(h_vals).normalize())];
    if nargout >= 2 {
        out.push(Value::matrix(Matrix::col_vec(w_vals)));
    }
    Ok(out)
}

/// `[gd, w] = grpdelay(b [, a [, n]])` — Digital filter group delay.
pub fn fn_grpdelay(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity("grpdelay", "1 to 3 (b [, a [, n]])", args.len()));
    }
    let b = args[0].to_dense_matrix()?;
    let a = if args.len() >= 2 {
        args[1].to_dense_matrix()?
    } else {
        Matrix::scalar(1.0)
    };
    let n = if args.len() >= 3 {
        args[2].as_scalar()?.round().max(2.0) as usize
    } else {
        512
    };
    let mut gd = Vec::with_capacity(n);
    let mut ws = Vec::with_capacity(n);
    let eval_gd_poly = |p: &[f64], w: f64| -> f64 {
        let z_inv = Complex::new(w.cos(), -w.sin());
        let mut val = Complex::ZERO;
        let mut dval = Complex::ZERO;
        let mut zp = Complex::ONE;
        for (k, &c) in p.iter().enumerate() {
            val = val + Complex::from_real(c) * zp;
            dval = dval + Complex::from_real(c * (k as f64)) * zp;
            zp = zp * z_inv;
        }
        if val.norm_sq() < 1e-24 { 0.0 } else { (dval / val).re }
    };
    for k in 0..n {
        let w = PI * (k as f64) / (n as f64);
        gd.push(eval_gd_poly(b.data(), w) - eval_gd_poly(a.data(), w));
        ws.push(w);
    }
    if nargout >= 2 {
        Ok(vec![Value::matrix(Matrix::col_vec(gd)), Value::matrix(Matrix::col_vec(ws))])
    } else {
        ok1(Value::matrix(Matrix::col_vec(gd)))
    }
}

/// `z = hilbert(x)` — Discrete-time analytic signal via Marple's FFT algorithm.
pub fn fn_hilbert(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("hilbert", "1 (x)", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let n = x.len();
    if n == 0 {
        return ok1(Value::empty());
    }
    let is_row = x.rows() == 1 && x.cols() > 1;
    let col = Matrix::col_vec(x.data().to_vec());
    let mut xf = col.fft();
    let half = n / 2;
    for k in 1..n {
        let scale = if n % 2 == 0 {
            if k < half { 2.0 } else if k == half { 1.0 } else { 0.0 }
        } else if k <= half {
            2.0
        } else {
            0.0
        };
        let v = xf.get_c(k, 0) * Complex::from_real(scale);
        xf.set_c(k, 0, v);
    }
    let analytic = xf.ifft();
    let out = if is_row { analytic.transpose() } else { analytic };
    ok1(Value::matrix(out))
}

/// `[yupper, ylower] = envelope(x)` — Analytic signal amplitude envelope.
pub fn fn_envelope(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("envelope", "1 (x)", 0));
    }
    let x = args[0].to_dense_matrix()?;
    let mu = x.data().iter().sum::<f64>() / (x.len().max(1) as f64);
    let centered = x.map(|v| v - mu);
    let analytic = fn_hilbert(&[Value::matrix(centered)])?[0].to_dense_matrix()?;
    let amp = analytic.map_c_to_real(Complex::abs);
    let upper = amp.map(|v| mu + v);
    let lower = amp.map(|v| mu - v);
    if nargout >= 2 {
        Ok(vec![Value::matrix(upper), Value::matrix(lower)])
    } else {
        ok1(Value::matrix(upper))
    }
}

/// `[S, F, T] = spectrogram(x [, window [, noverlap [, nfft [, fs]]]])`
pub fn fn_spectrogram(args: &[Value], nargout: usize, fig: &mut PlotFigure) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 5 {
        return Err(MatroError::arity(
            "spectrogram",
            "1 to 5 (x [, win [, noverlap [, nfft [, fs]]]])",
            args.len(),
        ));
    }
    let x_m = args[0].to_dense_matrix()?;
    let x = x_m.data();
    let nx = x.len();
    let win: Vec<f64> = match args.get(1) {
        None => {
            let wlen = (nx / 8).clamp(8, 256).min(nx);
            (0..wlen)
                .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / ((wlen - 1).max(1) as f64)).cos())
                .collect()
        }
        Some(v) => {
            let m = v.to_dense_matrix()?;
            if m.len() == 1 {
                let wlen = (m.data()[0].round() as usize).clamp(4, nx);
                (0..wlen)
                    .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / ((wlen - 1).max(1) as f64)).cos())
                    .collect()
            } else {
                m.data().to_vec()
            }
        }
    };
    let wlen = win.len();
    let noverlap = match args.get(2) {
        Some(v) => (v.as_scalar()?.round() as usize).min(wlen.saturating_sub(1)),
        None => wlen / 2,
    };
    let hop = (wlen - noverlap).max(1);
    let nfft = match args.get(3) {
        Some(v) => (v.as_scalar()?.round() as usize).max(wlen),
        None => wlen.next_power_of_two().max(64),
    };
    let fs = match args.get(4) {
        Some(v) => v.as_scalar()?,
        None => 2.0 * PI,
    };

    let nframes = if nx >= wlen { (nx - wlen) / hop + 1 } else { 1 };
    let nfreq = nfft / 2 + 1;
    let mut s_mat = Matrix::zeros(nfreq, nframes);
    let mut times = Vec::with_capacity(nframes);

    for f_idx in 0..nframes {
        let start = f_idx * hop;
        let mut buf = vec![0.0; nfft];
        for i in 0..wlen {
            if start + i < nx {
                buf[i] = x[start + i] * win[i];
            }
        }
        let spec = Matrix::col_vec(buf).fft();
        for k in 0..nfreq {
            s_mat.set_c(k, f_idx, spec.get_c(k, 0));
        }
        times.push((start as f64 + 0.5 * wlen as f64) / fs);
    }
    let freqs: Vec<f64> = (0..nfreq).map(|k| (k as f64) * fs / (nfft as f64)).collect();

    if nargout == 0 {
        let mut db = Matrix::zeros(nfreq, nframes);
        for c in 0..nframes {
            for r in 0..nfreq {
                // Flip vertically so low frequencies sit at bottom of heatmap
                let val = 20.0 * s_mat.get_c(nfreq - 1 - r, c).abs().max(1e-12).log10();
                db.set(r, c, val);
            }
        }
        let p = fig.panel_mut();
        p.clear();
        p.set_heatmap(&db);
        p.title = "Spectrogram (dB)".to_string();
        return Ok(vec![]);
    }

    let mut out = vec![Value::matrix(s_mat.normalize())];
    if nargout >= 2 {
        out.push(Value::matrix(Matrix::col_vec(freqs)));
    }
    if nargout >= 3 {
        out.push(Value::matrix(Matrix::row_vec(times)));
    }
    Ok(out)
}

/// 1D running median filter `y = medfilt1(x [, n])`.
pub fn fn_medfilt1(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("medfilt1", "1 or 2 (x [, n])", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let n = args
        .get(1)
        .map(|v| v.as_scalar())
        .transpose()?
        .unwrap_or(3.0)
        .round()
        .max(1.0) as usize;
    let data = x.data();
    let len = data.len();
    let half = n / 2;
    let mut out = vec![0.0; len];
    let mut win = Vec::with_capacity(n);
    for i in 0..len {
        win.clear();
        for offset in 0..n {
            let idx = (i as isize) + (offset as isize) - (half as isize);
            let v = if idx < 0 || (idx as usize) >= len { 0.0 } else { data[idx as usize] };
            win.push(v);
        }
        win.sort_by(|a, b| a.total_cmp(b));
        out[i] = if n % 2 == 1 {
            win[n / 2]
        } else {
            0.5 * (win[n / 2 - 1] + win[n / 2])
        };
    }
    let m = if x.rows() == 1 { Matrix::row_vec(out) } else { Matrix::col_vec(out) };
    ok1(Value::matrix(m))
}

/// Savitzky-Golay polynomial smoothing filter `y = sgolayfilt(x, order, framelen)`.
pub fn fn_sgolayfilt(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 3 {
        return Err(MatroError::arity("sgolayfilt", "3 (x, order, framelen)", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let order = args[1].as_scalar()?.round() as usize;
    let flen = args[2].as_scalar()?.round() as usize;
    if flen % 2 == 0 || flen < 3 || order >= flen {
        return Err(MatroError::runtime(
            "sgolayfilt: framelen must be odd >= 3 and order < framelen",
        ));
    }
    let half = (flen / 2) as isize;
    // Build Vandermonde matrix A (flen x (order+1)) for t = -half..=half
    let k = order + 1;
    let mut a = Matrix::zeros(flen, k);
    for r in 0..flen {
        let t = (r as isize - half) as f64;
        let mut p = 1.0;
        for c in 0..k {
            a.set(r, c, p);
            p *= t;
        }
    }
    // Center row of (A'A)^{-1} A' gives the smoothing weights for interior points
    let at = a.transpose();
    let ata = at.matmul(&a)?;
    let ata_inv = ata.inv()?;
    let filt = ata_inv.matmul(&at)?; // k x flen; row 0 is value at t=0

    let data = x.data();
    let n = data.len();
    let mut out = vec![0.0; n];
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..flen {
            let idx = (i as isize + j as isize - half).clamp(0, (n - 1) as isize) as usize;
            acc += filt.get(0, j) * data[idx];
        }
        out[i] = acc;
    }
    let m = if x.rows() == 1 { Matrix::row_vec(out) } else { Matrix::col_vec(out) };
    ok1(Value::matrix(m))
}

/// Remove mean (`'constant'`) or linear trend (`'linear'`, default): `y = detrend(x [, type])`.
pub fn fn_detrend(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("detrend", "1 or 2 (x [, 'linear'|'constant'])", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let mode = args
        .get(1)
        .and_then(|v| v.as_str().ok())
        .unwrap_or("linear")
        .to_ascii_lowercase();
    let data = x.data();
    let n = data.len();
    if n == 0 {
        return ok1(Value::matrix(x));
    }
    let out: Vec<f64> = if mode == "constant" || n == 1 {
        let mu = data.iter().sum::<f64>() / (n as f64);
        data.iter().map(|&v| v - mu).collect()
    } else {
        let nf = n as f64;
        let mx = 0.5 * (nf - 1.0);
        let my = data.iter().sum::<f64>() / nf;
        let mut num = 0.0;
        let mut den = 0.0;
        for (i, &v) in data.iter().enumerate() {
            let dx = (i as f64) - mx;
            num += dx * (v - my);
            den += dx * dx;
        }
        let slope = if den > 0.0 { num / den } else { 0.0 };
        let intercept = my - slope * mx;
        data.iter()
            .enumerate()
            .map(|(i, &v)| v - (intercept + slope * (i as f64)))
            .collect()
    };
    let m = if x.rows() == 1 { Matrix::row_vec(out) } else { Matrix::col_vec(out) };
    ok1(Value::matrix(m))
}

pub fn fn_downsample(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 {
        return Err(MatroError::arity("downsample", "2 (x, n)", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let step = (args[1].as_scalar()?.round() as usize).max(1);
    let out: Vec<f64> = x.data().iter().step_by(step).copied().collect();
    let m = if x.rows() == 1 { Matrix::row_vec(out) } else { Matrix::col_vec(out) };
    ok1(Value::matrix(m))
}

pub fn fn_upsample(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 {
        return Err(MatroError::arity("upsample", "2 (x, n)", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let factor = (args[1].as_scalar()?.round() as usize).max(1);
    let mut out = vec![0.0; x.len() * factor];
    for (i, &v) in x.data().iter().enumerate() {
        out[i * factor] = v;
    }
    let m = if x.rows() == 1 { Matrix::row_vec(out) } else { Matrix::col_vec(out) };
    ok1(Value::matrix(m))
}

/// Resample signal by rational factor `p / q` using windowed-sinc anti-aliasing interpolation.
pub fn fn_resample(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 3 {
        return Err(MatroError::arity("resample", "3 (x, p, q)", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let p = args[1].as_scalar()?.round().max(1.0) as usize;
    let q = args[2].as_scalar()?.round().max(1.0) as usize;
    let data = x.data();
    let n = data.len();
    let out_len = ((n * p) as f64 / (q as f64)).ceil() as usize;
    let ratio = (p as f64) / (q as f64);
    let cutoff = ratio.min(1.0);
    let half_w = 12usize;
    let sinc = |t: f64| if t.abs() < 1e-15 { 1.0 } else { (PI * t).sin() / (PI * t) };
    let mut out = Vec::with_capacity(out_len);
    for k in 0..out_len {
        let center = (k as f64) / ratio;
        let lo = (center.floor() as isize - half_w as isize).max(0) as usize;
        let hi = (center.ceil() as isize + half_w as isize).min((n - 1) as isize) as usize;
        let mut acc = 0.0;
        let mut wsum = 0.0;
        for i in lo..=hi {
            let dt = center - (i as f64);
            let win = 0.54 + 0.46 * (PI * dt / (half_w as f64)).cos();
            let w = cutoff * sinc(cutoff * dt) * win;
            acc += data[i] * w;
            wsum += w;
        }
        out.push(if wsum.abs() > 1e-12 { acc / wsum } else { 0.0 });
    }
    let m = if x.rows() == 1 { Matrix::row_vec(out) } else { Matrix::col_vec(out) };
    ok1(Value::matrix(m))
}

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let xh2 = 0.25 * x * x;
    for k in 1..=40 {
        term *= xh2 / ((k * k) as f64);
        sum += term;
        if term < 1e-15 * sum {
            break;
        }
    }
    sum
}

pub fn fn_extra_window(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity(name, ">= 1 (N)", 0));
    }
    let n = args[0].as_scalar()?.round().max(1.0) as usize;
    let nm1 = (n.saturating_sub(1)).max(1) as f64;
    let vals: Vec<f64> = match name {
        "bartlett" => (0..n)
            .map(|i| 1.0 - ((2.0 * (i as f64) - nm1) / nm1).abs())
            .collect(),
        "kaiser" => {
            let beta = args.get(1).map(|v| v.as_scalar()).transpose()?.unwrap_or(0.5);
            let denom = bessel_i0(beta);
            (0..n)
                .map(|i| {
                    let r = 2.0 * (i as f64) / nm1 - 1.0;
                    bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / denom
                })
                .collect()
        }
        "flattopwin" => (0..n)
            .map(|i| {
                let t = 2.0 * PI * (i as f64) / nm1;
                0.21557895 - 0.41663158 * t.cos() + 0.277263158 * (2.0 * t).cos()
                    - 0.083578947 * (3.0 * t).cos()
                    + 0.006947368 * (4.0 * t).cos()
            })
            .collect(),
        _ => unreachable!(),
    };
    ok1(Value::matrix(Matrix::col_vec(vals)))
}

#[allow(dead_code)]
fn _unused_eval_poly(p: &[f64], z: Complex) -> Complex {
    eval_poly_c(p, z)
}
