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

/// Bilinear transform from analog (zeros, poles, gain) with prewarped Fs=2 to digital `(z, p, k)`.
fn bilinear_zpk_roots(zeros: &[Complex], poles: &[Complex], gain: f64) -> (Vec<Complex>, Vec<Complex>, f64) {
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
    for _ in zeros.len()..poles.len() {
        z_d.push(Complex::from_real(-1.0));
    }
    let k_d = (num_scale / den_scale).re;
    (z_d, p_d, k_d)
}

fn bilinear_zpk(zeros: &[Complex], poles: &[Complex], gain: f64) -> (Vec<f64>, Vec<f64>) {
    let (z_d, p_d, k_d) = bilinear_zpk_roots(zeros, poles, gain);
    let b_c = poly_from_roots_c(&z_d);
    let a_c = poly_from_roots_c(&p_d);
    let b: Vec<f64> = b_c.iter().map(|c| c.re * k_d).collect();
    let a: Vec<f64> = a_c.iter().map(|c| c.re).collect();
    (b, a)
}

/// Apply analog lowpass-to-lowpass / highpass / bandpass / bandstop frequency transform to normalized prototype.
pub fn transform_analog_zpk(
    proto_z: &[Complex],
    proto_p: &[Complex],
    proto_k: f64,
    wn: &[f64],
    ftype: &str,
    analog: bool,
) -> Result<(Vec<Complex>, Vec<Complex>, f64)> {
    let n = proto_p.len();
    let m = proto_z.len();
    let prewarp = |w: f64| -> Result<f64> {
        if analog {
            if w <= 0.0 {
                return Err(MatroError::runtime("Analog cutoff frequency Wn must be > 0"));
            }
            return Ok(w);
        }
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
                return Err(MatroError::runtime("bandpass requires Wn = [w1, w2] with w1 < w2"));
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
        "stop" => {
            if wn.len() != 2 || wn[0] >= wn[1] {
                return Err(MatroError::runtime("stop requires Wn = [w1, w2] with w1 < w2"));
            }
            let w1 = prewarp(wn[0])?;
            let w2 = prewarp(wn[1])?;
            let bw = w2 - w1;
            let w0 = (w1 * w2).sqrt();
            let mut zs = Vec::new();
            for &z in proto_z {
                let b = Complex::from_real(0.5 * bw) / z;
                let disc = (b * b - Complex::from_real(w0 * w0)).sqrt();
                zs.push(b + disc);
                zs.push(b - disc);
            }
            for _ in m..n {
                zs.push(Complex::new(0.0, w0));
                zs.push(Complex::new(0.0, -w0));
            }
            let mut ps = Vec::new();
            for &p in proto_p {
                let b = Complex::from_real(0.5 * bw) / p;
                let disc = (b * b - Complex::from_real(w0 * w0)).sqrt();
                ps.push(b + disc);
                ps.push(b - disc);
            }
            let prod_z = proto_z.iter().fold(Complex::ONE, |acc, &z| acc * (-z));
            let prod_p = proto_p.iter().fold(Complex::ONE, |acc, &p| acc * (-p));
            let k = proto_k * (prod_z / prod_p).re;
            Ok((zs, ps, k))
        }
        other => Err(MatroError::runtime(format!("Unsupported filter type '{other}'"))),
    }
}

fn parse_ftype_and_analog(args: &[Value], wn_len: usize) -> (String, bool) {
    let mut ftype: Option<String> = None;
    let mut analog = false;
    for a in args {
        if let Some(s) = crate::value_as_text(a) {
            let low = s.to_ascii_lowercase();
            if low == "s" {
                analog = true;
            } else {
                ftype = Some(low);
            }
        }
    }
    let ft = ftype.unwrap_or_else(|| if wn_len == 2 { "bandpass".into() } else { "low".into() });
    (ft, analog)
}

/// `[b, a]` or `[z, p, k] = cheby1(n, Rp, Wn [, ftype [, 's']])` — Chebyshev Type I filter.
pub fn fn_cheby1(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() < 3 || args.len() > 5 {
        return Err(MatroError::arity("cheby1", "3 to 5 (n, Rp, Wn [, ftype [, 's']])", args.len()));
    }
    let n = args[0].as_scalar()?.round().max(1.0) as usize;
    let rp = args[1].as_scalar()?;
    if rp <= 0.0 {
        return Err(MatroError::runtime("cheby1: passband ripple Rp must be > 0 dB"));
    }
    let wn_m = args[2].to_dense_matrix()?;
    let wn = wn_m.data();
    let (ftype, analog) = parse_ftype_and_analog(&args[3..], wn.len());

    let eps = (10.0f64.powf(0.1 * rp) - 1.0).sqrt();
    let mu = (1.0 / eps).asinh() / (n as f64);
    let mut poles = Vec::with_capacity(n);
    for k in 1..=n {
        let theta = PI * (2 * k - 1) as f64 / (2.0 * n as f64);
        poles.push(Complex::new(-mu.sinh() * theta.sin(), mu.cosh() * theta.cos()));
    }
    let prod_p = poles.iter().fold(Complex::ONE, |acc, &p| acc * (-p));
    let k0 = prod_p.re / if n % 2 == 0 { (1.0 + eps * eps).sqrt() } else { 1.0 };

    let (zs, ps, k) = transform_analog_zpk(&[], &poles, k0, wn, &ftype, analog)?;
    if analog {
        if nargout >= 3 {
            return Ok(vec![
                Value::matrix(Matrix::complex_col_vec(zs).normalize()),
                Value::matrix(Matrix::complex_col_vec(ps).normalize()),
                Value::scalar(k),
            ]);
        }
        let b: Vec<f64> = poly_from_roots_c(&zs).iter().map(|c| c.re * k).collect();
        let a: Vec<f64> = poly_from_roots_c(&ps).iter().map(|c| c.re).collect();
        return Ok(vec![Value::matrix(Matrix::row_vec(b)), Value::matrix(Matrix::row_vec(a))]);
    }
    if nargout >= 3 {
        let (zd, pd, kd) = bilinear_zpk_roots(&zs, &ps, k);
        return Ok(vec![
            Value::matrix(Matrix::complex_col_vec(zd).normalize()),
            Value::matrix(Matrix::complex_col_vec(pd).normalize()),
            Value::scalar(kd),
        ]);
    }
    let (b, a) = bilinear_zpk(&zs, &ps, k);
    Ok(vec![Value::matrix(Matrix::row_vec(b)), Value::matrix(Matrix::row_vec(a))])
}

/// `[b, a]` or `[z, p, k] = cheby2(n, Rs, Wn [, ftype [, 's']])` — Chebyshev Type II filter.
pub fn fn_cheby2(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() < 3 || args.len() > 5 {
        return Err(MatroError::arity("cheby2", "3 to 5 (n, Rs, Wn [, ftype [, 's']])", args.len()));
    }
    let n = args[0].as_scalar()?.round().max(1.0) as usize;
    let rs = args[1].as_scalar()?;
    if rs <= 0.0 {
        return Err(MatroError::runtime("cheby2: stopband attenuation Rs must be > 0 dB"));
    }
    let wn_m = args[2].to_dense_matrix()?;
    let wn = wn_m.data();
    let (ftype, analog) = parse_ftype_and_analog(&args[3..], wn.len());

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

    let (zs, ps, k) = transform_analog_zpk(&zeros, &poles, k0, wn, &ftype, analog)?;
    if analog {
        if nargout >= 3 {
            return Ok(vec![
                Value::matrix(Matrix::complex_col_vec(zs).normalize()),
                Value::matrix(Matrix::complex_col_vec(ps).normalize()),
                Value::scalar(k),
            ]);
        }
        let b: Vec<f64> = poly_from_roots_c(&zs).iter().map(|c| c.re * k).collect();
        let a: Vec<f64> = poly_from_roots_c(&ps).iter().map(|c| c.re).collect();
        return Ok(vec![Value::matrix(Matrix::row_vec(b)), Value::matrix(Matrix::row_vec(a))]);
    }
    if nargout >= 3 {
        let (zd, pd, kd) = bilinear_zpk_roots(&zs, &ps, k);
        return Ok(vec![
            Value::matrix(Matrix::complex_col_vec(zd).normalize()),
            Value::matrix(Matrix::complex_col_vec(pd).normalize()),
            Value::scalar(kd),
        ]);
    }
    let (b, a) = bilinear_zpk(&zs, &ps, k);
    Ok(vec![Value::matrix(Matrix::row_vec(b)), Value::matrix(Matrix::row_vec(a))])
}

/// Group complex roots into real second-order quadratic factors `[1, c1, c2]`.
fn roots_to_quadratics(roots: &[Complex]) -> Vec<[f64; 3]> {
    let mut used = vec![false; roots.len()];
    let mut reals = Vec::new();
    let mut sections = Vec::new();
    for i in 0..roots.len() {
        if used[i] {
            continue;
        }
        if roots[i].im.abs() <= 1e-9 {
            used[i] = true;
            reals.push(roots[i].re);
        } else {
            used[i] = true;
            // Find closest conjugate partner
            let mut best = None;
            let mut best_d = f64::INFINITY;
            for j in (i + 1)..roots.len() {
                if !used[j] {
                    let d = (roots[j] - roots[i].conj()).abs();
                    if d < best_d {
                        best_d = d;
                        best = Some(j);
                    }
                }
            }
            if let Some(j) = best {
                used[j] = true;
                let r = roots[i];
                sections.push([1.0, -2.0 * r.re, r.norm_sq()]);
            } else {
                reals.push(roots[i].re);
            }
        }
    }
    let mut k = 0;
    while k + 1 < reals.len() {
        let (r1, r2) = (reals[k], reals[k + 1]);
        sections.push([1.0, -(r1 + r2), r1 * r2]);
        k += 2;
    }
    if k < reals.len() {
        sections.push([1.0, -reals[k], 0.0]);
    }
    sections
}

pub fn fn_zp2sos(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity("zp2sos", "2 or 3 (z, p [, k])", args.len()));
    }
    let zm = args[0].to_dense_matrix()?;
    let pm = args[1].to_dense_matrix()?;
    let k = args.get(2).map(|v| v.as_scalar()).transpose()?.unwrap_or(1.0);
    let zs: Vec<Complex> = (0..zm.len()).map(|i| zm.get_linear_c(i)).collect();
    let ps: Vec<Complex> = (0..pm.len()).map(|i| pm.get_linear_c(i)).collect();
    let mut b_quads = roots_to_quadratics(&zs);
    let mut a_quads = roots_to_quadratics(&ps);
    let l = b_quads.len().max(a_quads.len()).max(1);
    while b_quads.len() < l {
        b_quads.push([1.0, 0.0, 0.0]);
    }
    while a_quads.len() < l {
        a_quads.push([1.0, 0.0, 0.0]);
    }
    let mut sos = Matrix::zeros(l, 6);
    for r in 0..l {
        let scale = if r == 0 && nargout <= 1 { k } else { 1.0 };
        sos.set(r, 0, b_quads[r][0] * scale);
        sos.set(r, 1, b_quads[r][1] * scale);
        sos.set(r, 2, b_quads[r][2] * scale);
        sos.set(r, 3, a_quads[r][0]);
        sos.set(r, 4, a_quads[r][1]);
        sos.set(r, 5, a_quads[r][2]);
    }
    if nargout >= 2 {
        Ok(vec![Value::matrix(sos), Value::scalar(k)])
    } else {
        ok1(Value::matrix(sos))
    }
}

fn convolve_poly(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            out[i + j] += ai * bj;
        }
    }
    out
}

pub fn sos_to_tf(sos: &Matrix) -> Result<(Vec<f64>, Vec<f64>)> {
    if sos.cols() != 6 || sos.rows() == 0 {
        return Err(MatroError::dimension("sos2tf: SOS matrix must have size L x 6"));
    }
    let mut b = vec![1.0];
    let mut a = vec![1.0];
    for r in 0..sos.rows() {
        let br = [sos.get(r, 0), sos.get(r, 1), sos.get(r, 2)];
        let ar = [sos.get(r, 3), sos.get(r, 4), sos.get(r, 5)];
        b = convolve_poly(&b, &br);
        a = convolve_poly(&a, &ar);
    }
    // Trim trailing exact zeros past order
    while b.len() > 1 && a.len() > 1 && b.last() == Some(&0.0) && a.last() == Some(&0.0) {
        b.pop();
        a.pop();
    }
    Ok((b, a))
}

pub fn fn_sos2tf(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("sos2tf", "1 or 2 (sos [, g])", args.len()));
    }
    let sos = args[0].to_dense_matrix()?;
    let g = args
        .get(1)
        .map(|v| Ok::<f64, MatroError>(v.to_dense_matrix()?.data().iter().product()))
        .transpose()?
        .unwrap_or(1.0);
    let (mut b, a) = sos_to_tf(&sos)?;
    for bi in &mut b {
        *bi *= g;
    }
    Ok(vec![
        Value::matrix(Matrix::row_vec(b)),
        Value::matrix(Matrix::row_vec(a)),
    ])
}

pub fn fn_tf2sos(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("tf2sos", "2 (b, a)", args.len()));
    }
    let bm = args[0].to_dense_matrix()?;
    let am = args[1].to_dense_matrix()?;
    let b_lead = bm.data().iter().copied().find(|v| v.abs() > 1e-15).unwrap_or(1.0);
    let a_lead = am.data().iter().copied().find(|v| v.abs() > 1e-15).unwrap_or(1.0);
    let k = b_lead / a_lead;
    let zs = crate::extras::fn_roots(&[Value::matrix(bm)])?[0].clone();
    let ps = crate::extras::fn_roots(&[Value::matrix(am)])?[0].clone();
    fn_zp2sos(&[zs, ps, Value::scalar(k)], nargout)
}

pub fn fn_sosfilt(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("sosfilt", "2 (sos, x)", args.len()));
    }
    let sos = args[0].to_dense_matrix()?;
    if sos.cols() != 6 {
        return Err(MatroError::dimension("sosfilt: sos must have 6 columns"));
    }
    let x = args[1].to_dense_matrix()?;
    if x.is_empty() {
        return ok1(Value::matrix(x));
    }
    let filter_1d = |mut sig: Vec<f64>| -> Vec<f64> {
        for r in 0..sos.rows() {
            let b = [sos.get(r, 0), sos.get(r, 1), sos.get(r, 2)];
            let a = [sos.get(r, 3), sos.get(r, 4), sos.get(r, 5)];
            sig = crate::signal::apply_iir(&b, &a, &sig);
        }
        sig
    };
    if x.rows() == 1 || x.cols() == 1 {
        let y = filter_1d(x.data().to_vec());
        let out = if x.rows() == 1 { Matrix::row_vec(y) } else { Matrix::col_vec(y) };
        return ok1(Value::matrix(out));
    }
    let mut out = Matrix::zeros(x.rows(), x.cols());
    for c in 0..x.cols() {
        let col: Vec<f64> = (0..x.rows()).map(|r| x.get(r, c)).collect();
        let y = filter_1d(col);
        for r in 0..x.rows() {
            out.set(r, c, y[r]);
        }
    }
    ok1(Value::matrix(out))
}

/// `b = fir1(n, Wn [, ftype] [, window])` — Windowed-sinc FIR filter design.
pub fn fn_fir1(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 4 {
        return Err(MatroError::arity("fir1", "2 to 4 (n, Wn [, ftype] [, window])", args.len()));
    }
    let mut n = args[0].as_scalar()?.round().max(1.0) as usize;
    let wn_m = args[1].to_dense_matrix()?;
    let wn = wn_m.data();
    let mut ftype: Option<String> = None;
    let mut custom_win: Option<Vec<f64>> = None;
    for a in &args[2..] {
        if let Some(s) = crate::value_as_text(a) {
            let low = s.to_ascii_lowercase();
            if low != "scale" && low != "noscale" {
                ftype = Some(low);
            }
        } else {
            let wm = a.to_dense_matrix()?;
            if !wm.is_empty() {
                custom_win = Some(wm.data().to_vec());
            }
        }
    }
    let ftype = ftype.unwrap_or_else(|| if wn.len() == 2 { "bandpass".into() } else { "low".into() });

    if (ftype == "high" || ftype == "stop") && n % 2 == 1 {
        n += 1;
    }
    let len = n + 1;
    let win: Vec<f64> = match custom_win {
        Some(w) if w.len() == len => w,
        _ => (0..len)
            .map(|i| 0.54 - 0.46 * (2.0 * PI * (i as f64) / (n as f64)).cos())
            .collect(),
    };
    let m = 0.5 * (n as f64);
    let sinc = |x: f64| -> f64 {
        if x.abs() < 1e-15 { 1.0 } else { (PI * x).sin() / (PI * x) }
    };
    let lp_kernel = |fc: f64| -> Vec<f64> {
        (0..len)
            .map(|i| {
                let k = (i as f64) - m;
                fc * sinc(fc * k) * win[i]
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

/// `[H, w] = freqz(b [, a [, n|w [, fs]]])` — Digital filter complex frequency response.
pub fn fn_freqz(args: &[Value], nargout: usize, fig: &mut PlotFigure) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 4 {
        return Err(MatroError::arity("freqz", "1 to 4 (b [, a [, n|w [, fs]]])", args.len()));
    }
    let b = args[0].to_dense_matrix()?;
    let a = if args.len() >= 2 && !args[1].to_dense_matrix()?.is_empty() {
        args[1].to_dense_matrix()?
    } else {
        Matrix::scalar(1.0)
    };
    let fs = if args.len() >= 4 {
        Some(args[3].as_scalar()?)
    } else {
        None
    };

    let (omegas, w_out): (Vec<f64>, Vec<f64>) = if let Some(a2) = args.get(2) {
        let m2 = a2.to_dense_matrix()?;
        if m2.len() > 1 {
            let raw = m2.data().to_vec();
            let rad: Vec<f64> = if let Some(sr) = fs {
                raw.iter().map(|&f| 2.0 * PI * f / sr).collect()
            } else {
                raw.clone()
            };
            (rad, raw)
        } else {
            let n = if m2.is_empty() {
                512
            } else {
                m2.data()[0].round().max(2.0) as usize
            };
            let rad: Vec<f64> = (0..n).map(|k| PI * (k as f64) / (n as f64)).collect();
            let out: Vec<f64> = rad
                .iter()
                .map(|&w| if let Some(sr) = fs { w * sr / (2.0 * PI) } else { w })
                .collect();
            (rad, out)
        }
    } else {
        let n = 512usize;
        let rad: Vec<f64> = (0..n).map(|k| PI * (k as f64) / (n as f64)).collect();
        let out: Vec<f64> = rad
            .iter()
            .map(|&w| if let Some(sr) = fs { w * sr / (2.0 * PI) } else { w })
            .collect();
        (rad, out)
    };

    let mut h_vals = Vec::with_capacity(omegas.len());
    for &w in &omegas {
        let z_inv = Complex::new(w.cos(), -w.sin());
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
    }

    if nargout == 0 {
        let mag_db: Vec<f64> = h_vals
            .iter()
            .map(|c| 20.0 * c.abs().max(1e-12).log10())
            .collect();
        let p = fig.panel_mut();
        p.clear();
        p.add_series(PlotKind::Line, w_out, mag_db, None);
        p.title = "Frequency Response (freqz)".to_string();
        p.xlabel = if fs.is_some() { "Frequency (Hz)" } else { "Normalized Frequency (rad/sample)" }.to_string();
        p.ylabel = "Magnitude (dB)".to_string();
        return Ok(vec![]);
    }

    let mut out = vec![Value::matrix(Matrix::complex_col_vec(h_vals).normalize())];
    if nargout >= 2 {
        out.push(Value::matrix(Matrix::col_vec(w_out)));
    }
    Ok(out)
}

/// `[gd, w] = grpdelay(b [, a [, n|w [, fs]]])` — Digital filter group delay.
pub fn fn_grpdelay(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 4 {
        return Err(MatroError::arity("grpdelay", "1 to 4 (b [, a [, n|w [, fs]]])", args.len()));
    }
    let b = args[0].to_dense_matrix()?;
    let a = if args.len() >= 2 && !args[1].to_dense_matrix()?.is_empty() {
        args[1].to_dense_matrix()?
    } else {
        Matrix::scalar(1.0)
    };
    let fs = args.get(3).map(|v| v.as_scalar()).transpose()?;
    let (omegas, ws): (Vec<f64>, Vec<f64>) = if let Some(a2) = args.get(2) {
        let m2 = a2.to_dense_matrix()?;
        if m2.len() > 1 {
            let raw = m2.data().to_vec();
            let rad: Vec<f64> = if let Some(sr) = fs {
                raw.iter().map(|&f| 2.0 * PI * f / sr).collect()
            } else {
                raw.clone()
            };
            (rad, raw)
        } else {
            let n = if m2.is_empty() {
                512
            } else {
                m2.data()[0].round().max(2.0) as usize
            };
            let rad: Vec<f64> = (0..n).map(|k| PI * (k as f64) / (n as f64)).collect();
            let out: Vec<f64> = rad
                .iter()
                .map(|&w| if let Some(sr) = fs { w * sr / (2.0 * PI) } else { w })
                .collect();
            (rad, out)
        }
    } else {
        let n = 512usize;
        let rad: Vec<f64> = (0..n).map(|k| PI * (k as f64) / (n as f64)).collect();
        (rad.clone(), rad)
    };

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
    let gd: Vec<f64> = omegas
        .iter()
        .map(|&w| eval_gd_poly(b.data(), w) - eval_gd_poly(a.data(), w))
        .collect();
    if nargout >= 2 {
        Ok(vec![Value::matrix(Matrix::col_vec(gd)), Value::matrix(Matrix::col_vec(ws))])
    } else {
        ok1(Value::matrix(Matrix::col_vec(gd)))
    }
}

fn hilbert_1d(data: &[f64], n_fft: usize) -> Vec<Complex> {
    if n_fft == 0 {
        return vec![];
    }
    let mut padded = vec![0.0; n_fft];
    let copy_len = data.len().min(n_fft);
    padded[..copy_len].copy_from_slice(&data[..copy_len]);
    let mut xf = Matrix::col_vec(padded).fft();
    let half = n_fft / 2;
    for k in 1..n_fft {
        let scale = if n_fft % 2 == 0 {
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
    (0..n_fft).map(|k| analytic.get_c(k, 0)).collect()
}

/// `z = hilbert(x [, n])` — Discrete-time analytic signal via Marple's FFT algorithm.
pub fn fn_hilbert(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("hilbert", "1 or 2 (x [, n])", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    if x.is_empty() {
        return ok1(Value::empty());
    }
    let n_opt = args.get(1).map(|v| v.as_scalar().map(|s| s.max(1.0).round() as usize)).transpose()?;
    if x.rows() == 1 || x.cols() == 1 {
        let is_row = x.rows() == 1 && x.cols() > 1;
        let n = n_opt.unwrap_or(x.len());
        let h = hilbert_1d(x.data(), n);
        let out = if is_row {
            Matrix::complex_row_vec(h)
        } else {
            Matrix::complex_col_vec(h)
        };
        return ok1(Value::matrix(out));
    }
    let n = n_opt.unwrap_or(x.rows());
    let mut out = Matrix::zeros(n, x.cols());
    for c in 0..x.cols() {
        let col: Vec<f64> = (0..x.rows()).map(|r| x.get(r, c)).collect();
        let h = hilbert_1d(&col, n);
        for r in 0..n {
            out.set_c(r, c, h[r]);
        }
    }
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
    let default_win = || {
        let wlen = (nx / 8).clamp(8, 256).min(nx).max(1);
        (0..wlen)
            .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / ((wlen - 1).max(1) as f64)).cos())
            .collect::<Vec<f64>>()
    };
    let win: Vec<f64> = match args.get(1) {
        None => default_win(),
        Some(v) => {
            let m = v.to_dense_matrix()?;
            if m.is_empty() {
                default_win()
            } else if m.len() == 1 {
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
        Some(v) if !v.to_dense_matrix()?.is_empty() => {
            (v.as_scalar()?.round() as usize).min(wlen.saturating_sub(1))
        }
        _ => wlen / 2,
    };
    let hop = (wlen - noverlap).max(1);
    let nfft = match args.get(3) {
        Some(v) if !v.to_dense_matrix()?.is_empty() => {
            (v.as_scalar()?.round() as usize).max(wlen)
        }
        _ => wlen.next_power_of_two().max(64),
    };
    let fs = match args.get(4) {
        Some(v) if !v.to_dense_matrix()?.is_empty() => v.as_scalar()?,
        _ => 2.0 * PI,
    };
    let win_energy: f64 = win.iter().map(|&w| w * w).sum::<f64>().max(1e-30);

    let nframes = if nx >= wlen { (nx - wlen) / hop + 1 } else { 1 };
    let nfreq = nfft / 2 + 1;
    let mut s_mat = Matrix::zeros(nfreq, nframes);
    let mut p_mat = Matrix::zeros(nfreq, nframes);
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
            let val = spec.get_c(k, 0);
            s_mat.set_c(k, f_idx, val);
            let scale = if k == 0 || (nfft % 2 == 0 && k == nfreq - 1) { 1.0 } else { 2.0 };
            p_mat.set(k, f_idx, scale * val.norm_sq() / (fs * win_energy));
        }
        times.push((start as f64 + 0.5 * wlen as f64) / fs);
    }
    let freqs: Vec<f64> = (0..nfreq).map(|k| (k as f64) * fs / (nfft as f64)).collect();

    if nargout == 0 {
        let mut db = Matrix::zeros(nfreq, nframes);
        for c in 0..nframes {
            for r in 0..nfreq {
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
    if nargout >= 4 {
        out.push(Value::matrix(p_mat));
    }
    Ok(out)
}

fn apply_colwise(x: &Matrix, f: impl Fn(&[f64]) -> Result<Vec<f64>>) -> Result<Matrix> {
    if x.is_empty() {
        return Ok(x.clone());
    }
    if x.rows() == 1 || x.cols() == 1 {
        let y = f(x.data())?;
        return Ok(if x.rows() == 1 { Matrix::row_vec(y) } else { Matrix::col_vec(y) });
    }
    let nr = x.rows();
    let nc = x.cols();
    let mut cols_out: Vec<Vec<f64>> = Vec::with_capacity(nc);
    for c in 0..nc {
        let col: Vec<f64> = (0..nr).map(|r| x.get(r, c)).collect();
        cols_out.push(f(&col)?);
    }
    let out_r = cols_out[0].len();
    let mut out = Matrix::zeros(out_r, nc);
    for (c, col) in cols_out.iter().enumerate() {
        for r in 0..out_r {
            out.set(r, c, col[r]);
        }
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
    let half = n / 2;
    let out = apply_colwise(&x, |data| {
        let len = data.len();
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
        Ok(out)
    })?;
    ok1(Value::matrix(out))
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
    let at = a.transpose();
    let ata = at.matmul(&a)?;
    let ata_inv = ata.inv()?;
    let filt = ata_inv.matmul(&at)?;

    let out = apply_colwise(&x, |data| {
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
        Ok(out)
    })?;
    ok1(Value::matrix(out))
}

/// Remove constant (`0`/`'constant'`), linear (`1`/`'linear'`), or degree-`n` polynomial trend: `y = detrend(x [, n])`.
pub fn fn_detrend(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("detrend", "1 or 2 (x [, n|'linear'|'constant'])", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let degree: usize = match args.get(1) {
        None => 1,
        Some(v) => {
            if let Some(s) = crate::value_as_text(v) {
                if s.eq_ignore_ascii_case("constant") { 0 } else { 1 }
            } else {
                v.as_scalar()?.max(0.0).round() as usize
            }
        }
    };
    let out = apply_colwise(&x, |data| {
        let n = data.len();
        if n == 0 {
            return Ok(vec![]);
        }
        if degree == 0 || n == 1 {
            let mu = data.iter().sum::<f64>() / (n as f64);
            return Ok(data.iter().map(|&v| v - mu).collect());
        }
        if degree == 1 {
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
            return Ok(data
                .iter()
                .enumerate()
                .map(|(i, &v)| v - (intercept + slope * (i as f64)))
                .collect());
        }
        // General polynomial detrend via normalized Vandermonde least squares
        let p = (degree + 1).min(n);
        let scale = ((n - 1) as f64).max(1.0);
        let mut vand = Matrix::zeros(n, p);
        for i in 0..n {
            let t = 2.0 * (i as f64) / scale - 1.0;
            let mut tp = 1.0;
            for c in 0..p {
                vand.set(i, c, tp);
                tp *= t;
            }
        }
        let y_col = Matrix::col_vec(data.to_vec());
        let coef = vand.solve(&y_col)?;
        let trend = vand.matmul(&coef)?;
        Ok((0..n).map(|i| data[i] - trend.get(i, 0)).collect())
    })?;
    ok1(Value::matrix(out))
}

pub fn fn_downsample(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity("downsample", "2 or 3 (x, n [, phase])", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let step = (args[1].as_scalar()?.round() as usize).max(1);
    let phase = args
        .get(2)
        .map(|v| v.as_scalar().map(|s| (s.max(0.0).round() as usize) % step))
        .transpose()?
        .unwrap_or(0);
    let out = apply_colwise(&x, |data| {
        Ok(data.iter().skip(phase).step_by(step).copied().collect())
    })?;
    ok1(Value::matrix(out))
}

pub fn fn_upsample(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity("upsample", "2 or 3 (x, n [, phase])", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let factor = (args[1].as_scalar()?.round() as usize).max(1);
    let phase = args
        .get(2)
        .map(|v| v.as_scalar().map(|s| (s.max(0.0).round() as usize) % factor))
        .transpose()?
        .unwrap_or(0);
    let out = apply_colwise(&x, |data| {
        let mut out = vec![0.0; data.len() * factor];
        for (i, &v) in data.iter().enumerate() {
            out[i * factor + phase] = v;
        }
        Ok(out)
    })?;
    ok1(Value::matrix(out))
}

/// `y = decimate(x, r [, n [, 'fir']])` — Anti-aliased integer-factor decimation.
pub fn fn_decimate(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 4 {
        return Err(MatroError::arity("decimate", "2 to 4 (x, r [, n [, 'fir']])", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let r = (args[1].as_scalar()?.round() as usize).max(1);
    if r == 1 {
        return ok1(Value::matrix(x));
    }
    let is_fir = args
        .iter()
        .skip(2)
        .any(|v| crate::value_as_text(v).is_some_and(|s| s.eq_ignore_ascii_case("fir")));
    let order = args
        .get(2)
        .and_then(|v| v.as_scalar().ok())
        .map(|s| s.max(1.0).round() as usize)
        .unwrap_or(if is_fir { 30 } else { 8 });
    let wn = (0.8 / r as f64).clamp(1e-4, 0.99);
    let (b, a) = if is_fir {
        let bv = fn_fir1(&[Value::scalar(order as f64), Value::scalar(wn)])?[0].to_dense_matrix()?;
        (bv.data().to_vec(), vec![1.0])
    } else {
        let ba = fn_cheby1(&[Value::scalar(order.min(8) as f64), Value::scalar(0.05), Value::scalar(wn)], 2)?;
        (ba[0].to_dense_matrix()?.data().to_vec(), ba[1].to_dense_matrix()?.data().to_vec())
    };
    let filtered = crate::signal::fn_filtfilt(&[
        Value::matrix(Matrix::row_vec(b)),
        Value::matrix(Matrix::row_vec(a)),
        Value::matrix(x),
    ])?[0].to_dense_matrix()?;
    fn_downsample(&[Value::matrix(filtered), Value::scalar(r as f64)])
}

/// Resample signal by rational factor `p / q` using windowed-sinc anti-aliasing interpolation.
pub fn fn_resample(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 3 {
        return Err(MatroError::arity("resample", "3 (x, p, q)", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let p = args[1].as_scalar()?.round().max(1.0) as usize;
    let q = args[2].as_scalar()?.round().max(1.0) as usize;
    let ratio = (p as f64) / (q as f64);
    let cutoff = ratio.min(1.0);
    let half_w = 12usize;
    let sinc = |t: f64| if t.abs() < 1e-15 { 1.0 } else { (PI * t).sin() / (PI * t) };
    let out = apply_colwise(&x, |data| {
        let n = data.len();
        let out_len = ((n * p) as f64 / (q as f64)).ceil() as usize;
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
        Ok(out)
    })?;
    ok1(Value::matrix(out))
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
    let n = args[0].as_scalar()?.round().max(0.0) as usize;
    if n == 0 {
        return ok1(Value::empty());
    }
    if n == 1 {
        return ok1(Value::scalar(1.0));
    }
    let nm1 = (n - 1) as f64;
    let vals: Vec<f64> = match name {
        "bartlett" => (0..n)
            .map(|i| 1.0 - ((2.0 * (i as f64) - nm1) / nm1).abs())
            .collect(),
        "triang" => {
            if n % 2 == 1 {
                let half = (n + 1) / 2;
                (1..=n)
                    .map(|k| {
                        if k <= half {
                            2.0 * k as f64 / (n + 1) as f64
                        } else {
                            2.0 * (n + 1 - k) as f64 / (n + 1) as f64
                        }
                    })
                    .collect()
            } else {
                let half = n / 2;
                (1..=n)
                    .map(|k| {
                        if k <= half {
                            (2 * k - 1) as f64 / n as f64
                        } else {
                            (2 * (n - k) + 1) as f64 / n as f64
                        }
                    })
                    .collect()
            }
        }
        "rectwin" => vec![1.0; n],
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
        "blackmanharris" => (0..n)
            .map(|i| {
                let t = 2.0 * PI * (i as f64) / nm1;
                0.35875 - 0.48829 * t.cos() + 0.14128 * (2.0 * t).cos() - 0.01168 * (3.0 * t).cos()
            })
            .collect(),
        "tukeywin" => {
            let r = args.get(1).map(|v| v.as_scalar()).transpose()?.unwrap_or(0.5).clamp(0.0, 1.0);
            if r <= 0.0 {
                vec![1.0; n]
            } else {
                (0..n)
                    .map(|i| {
                        let x = i as f64 / nm1;
                        if x < 0.5 * r {
                            0.5 * (1.0 + (2.0 * PI / r * (x - 0.5 * r)).cos())
                        } else if x <= 1.0 - 0.5 * r {
                            1.0
                        } else {
                            0.5 * (1.0 + (2.0 * PI / r * (x - 1.0 + 0.5 * r)).cos())
                        }
                    })
                    .collect()
            }
        }
        "gausswin" => {
            let alpha = args.get(1).map(|v| v.as_scalar()).transpose()?.unwrap_or(2.5);
            let half = 0.5 * nm1;
            (0..n)
                .map(|i| {
                    let t = alpha * ((i as f64 - half) / half);
                    (-0.5 * t * t).exp()
                })
                .collect()
        }
        _ => unreachable!(),
    };
    ok1(Value::matrix(Matrix::col_vec(vals)))
}

#[allow(dead_code)]
fn _unused_eval_poly(p: &[f64], z: Complex) -> Complex {
    eval_poly_c(p, z)
}
