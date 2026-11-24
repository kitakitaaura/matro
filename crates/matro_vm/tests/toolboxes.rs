//! Conformance tests for Matro's pure-Rust Statistics, Signal Processing,
//! and Optimization & Control toolboxes.

use matro_vm::Vm;

#[test]
fn toolbox_statistics() {
    let mut vm = Vm::new();
    vm.eval(
        r#"
        x = [2, 4, 4, 4, 5, 5, 7, 9];
        assert(abs(mean(x) - 5.0) < 1e-12);
        assert(abs(var(x, 1) - 4.0) < 1e-12);
        assert(abs(std(x, 1) - 2.0) < 1e-12);
        assert(abs(var(x) - 32/7) < 1e-12);
        assert(abs(median(x) - 4.5) < 1e-12);
        assert(abs(prctile(x, 50) - 4.5) < 1e-12);
        assert(abs(quantile(x, 0.5) - 4.5) < 1e-12);

        % Covariance & correlation
        a = [1; 2; 3; 4; 5];
        b = [2; 4; 6; 8; 10];
        C = cov(a, b);
        R = corrcoef(a, b);
        assert(abs(C(1, 2) - 5.0) < 1e-12);
        assert(abs(R(1, 2) - 1.0) < 1e-12);

        % Normal distribution + erf/erfc round-trip
        assert(abs(erf(0)) < 1e-15 && abs(erfc(0) - 1) < 1e-15);
        assert(abs(normpdf(0) - 1/sqrt(2*pi)) < 1e-12);
        assert(abs(normcdf(0) - 0.5) < 1e-12);
        assert(abs(normcdf(1.959963984540054) - 0.975) < 1e-10);
        assert(abs(norminv(0.975) - 1.959963984540054) < 1e-10);
        assert(abs(norminv(normcdf(2.3, 10, 3), 10, 3) - 2.3) < 1e-10);

        % One-sample Student's t-test
        sample = [10.2, 9.8, 10.1, 10.4, 9.9, 10.3, 10.0, 10.5];
        [h0, p0, ci0, st0] = ttest(sample, 10.0);
        assert(h0 == 0 && p0 > 0.05 && st0.df == 7);
        [h1, p1] = ttest(sample, 0.0);
        assert(h1 == 1 && p1 < 1e-6);

        % k-means clustering on 2 clearly separated 2D blobs
        rng(7);
        X = [0.1, 0.2; 0.0, -0.1; -0.2, 0.1; 10.1, 10.0; 9.9, 10.2; 10.2, 9.8];
        [idx, centers] = kmeans(X, 2);
        assert(idx(1) == idx(2) && idx(2) == idx(3));
        assert(idx(4) == idx(5) && idx(5) == idx(6) && idx(1) ~= idx(4));
        assert(abs(sum(centers(:, 1)) - 10.0) < 0.3);

        % PCA on correlated 2D data along y = 2x
        t = (-5:5)';
        Xp = [t, 2*t];
        [coeff, score, latent] = pca(Xp);
        assert(latent(1) > 10 && latent(2) < 1e-10);
        assert(abs(abs(coeff(2, 1) / coeff(1, 1)) - 2.0) < 1e-10);
        "#,
    )
    .unwrap();
}

#[test]
fn toolbox_signal_processing() {
    let mut vm = Vm::new();
    vm.eval(
        r#"
        % fftshift / ifftshift round-trip (even & odd lengths)
        v = [1, 2, 3, 4, 5, 6];
        assert(norm(fftshift(v) - [4, 5, 6, 1, 2, 3]) == 0);
        assert(norm(ifftshift(fftshift(v)) - v) == 0);
        v_odd = [1, 2, 3, 4, 5];
        assert(norm(ifftshift(fftshift(v_odd)) - v_odd) == 0);

        % Window functions
        wh = hann(5);
        assert(abs(wh(1)) < 1e-12 && abs(wh(3) - 1.0) < 1e-12 && abs(wh(5)) < 1e-12);
        wm = hamming(5);
        assert(abs(wm(3) - 1.0) < 1e-12 && abs(wm(1) - 0.08) < 1e-12);
        wb = blackman(5);
        assert(abs(wb(3) - 1.0) < 1e-12);

        % Cross-correlation peak at exact lag
        [r, lags] = xcorr([0, 0, 1, 2, 1], [1, 2, 1, 0, 0]);
        [mx, imx] = max(r);
        assert(abs(mx - 6.0) < 1e-12 && lags(imx) == 2);

        % findpeaks with MinPeakHeight & MinPeakDistance
        sig = [0, 1, 3, 1, 0, 2, 5, 2, 0, 4, 1];
        [pks, locs] = findpeaks(sig, 'MinPeakHeight', 3.5);
        assert(length(pks) == 2 && pks(1) == 5 && locs(1) == 7 && pks(2) == 4 && locs(2) == 10);

        % Butterworth IIR design + zero-phase filtfilt
        fs = 200;
        t = (0:399) / fs;
        low_tone = sin(2 * pi * 5 * t);
        noisy = low_tone + 0.8 * sin(2 * pi * 70 * t);
        [b, a] = butter(4, 20 / (fs / 2));
        clean = filtfilt(b, a, noisy);
        % Interior of filtered signal should closely match pure 5 Hz tone with zero phase lag
        err = norm(clean(40:360) - low_tone(40:360)) / norm(low_tone(40:360));
        assert(err < 0.05, 'filtfilt Butterworth lowpass error too large');

        % Welch PSD peak frequency detection
        [pxx, f] = pwelch(sin(2 * pi * 25 * t), 128, 64, 256, fs);
        [~, peak_i] = max(pxx);
        assert(abs(f(peak_i) - 25.0) < 2.0, 'pwelch peak frequency mismatch');
        "#,
    )
    .unwrap();
}

#[test]
fn toolbox_optimization_and_control() {
    let mut vm = Vm::new();
    vm.eval(
        r#"
        % 1. fsolve: nonlinear system x^2 + y^2 = 5, x - y = 1 => (2, 1)
        [sol, fval] = fsolve(@(v) [v(1)^2 + v(2)^2 - 5; v(1) - v(2) - 1], [1.5; 0.5]);
        assert(norm(sol - [2; 1]) < 1e-8 && norm(fval) < 1e-8);

        % 2. lsqnonlin: nonlinear exponential curve fitting y = a * exp(-b * x)
        xd = linspace(0, 3, 15);
        yd = 2.5 * exp(-0.8 * xd);
        [p, resnorm] = lsqnonlin(@(p) p(1) * exp(-p(2) * xd) - yd, [1.0, 0.2]);
        assert(abs(p(1) - 2.5) < 1e-6 && abs(p(2) - 0.8) < 1e-6 && resnorm < 1e-12);

        % 3. linprog: minimize -3*x1 - 5*x2 s.t. x1 <= 4, 2*x2 <= 12, 3*x1 + 2*x2 <= 18, x >= 0
        % Exact optimum: x = [2; 6], fval = -36
        f = [-3; -5];
        A = [1, 0; 0, 2; 3, 2];
        b = [4; 12; 18];
        [xlp, flp] = linprog(f, A, b, [], [], [0; 0]);
        assert(abs(xlp(1) - 2.0) < 1e-8 && abs(xlp(2) - 6.0) < 1e-8 && abs(flp - (-36.0)) < 1e-8);

        % 4. Control systems: tf, step, bode
        % First-order system G(s) = 1 / (s + 1) => exact step y(t) = 1 - exp(-t)
        sys1 = tf(1, [1, 1]);
        t_eval = [0, 1, 2, 5];
        [ys, ts] = step(sys1, t_eval);
        assert(abs(ys(2) - (1 - exp(-1))) < 1e-4);
        assert(abs(ys(4) - (1 - exp(-5))) < 1e-4);

        % Bode at w = 1 rad/s: |1/(1+i)| = 1/sqrt(2), phase = -45 deg
        [mag, phase, w] = bode(sys1, [0.1, 1.0, 10.0]);
        assert(abs(mag(2) - 1/sqrt(2)) < 1e-12 && abs(phase(2) - (-45.0)) < 1e-10);

        % Auto-plotting Bode diagram
        sys2 = tf(4, [1, 1.2, 4]);
        bode(sys2);
        "#,
    )
    .unwrap();
    let svg = vm.figure().to_svg(640.0, 400.0);
    assert!(svg.contains("Bode Diagram") && svg.contains("Magnitude (dB)"));
}
