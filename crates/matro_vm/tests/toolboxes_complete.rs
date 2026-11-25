use matro_vm::Vm;

#[test]
fn base_niche_packages_import_subsasgn_tables_fminbnd_integral2() {
    let mut vm = Vm::new();
    vm.mount_file(
        "+mymath/cube.m",
        "function y = cube(x)\n  y = x.^3;\nend\n",
    )
    .unwrap();
    vm.mount_file(
        "+mymath/+geom/Box.m",
        r#"
        classdef Box
            properties
                w = 1
                h = 1
            end
            methods
                function obj = Box(w, h)
                    if nargin >= 1, obj.w = w; end
                    if nargin >= 2, obj.h = h; end
                end
                function a = area(obj)
                    a = obj.w * obj.h;
                end
            end
        end
        "#,
    )
    .unwrap();

    vm.eval(
        r#"
        % 1. +package namespaces & import
        assert(mymath.cube(3) == 27);
        b = mymath.geom.Box(4, 5);
        assert(b.area() == 20);
        import mymath.*
        assert(cube(4) == 64);

        % 2. subsasgn & subsref overloading
        classdef RingBuf
            properties
                Data = [0, 0, 0, 0]
            end
            methods
                function v = subsref(obj, S)
                    idx = mod(S.subs{1} - 1, length(obj.Data)) + 1;
                    v = obj.Data(idx);
                end
                function obj = subsasgn(obj, S, val)
                    idx = mod(S.subs{1} - 1, length(obj.Data)) + 1;
                    obj.Data(idx) = val;
                end
            end
        end
        rb = RingBuf();
        rb(1) = 10;
        rb(6) = 99; % wraps around to slot 2!
        assert(rb(1) == 10 && rb(2) == 99 && rb(6) == 99);

        % 3. Relational table operations
        T1 = table([1; 2; 3], {'a'; 'b'; 'c'}, [10; 20; 30], 'VariableNames', {'ID', 'Grp', 'Val'});
        T2 = table([2; 3; 4], [200; 300; 400], 'VariableNames', {'ID', 'Score'});
        Tin = innerjoin(T1, T2, 'Keys', 'ID');
        assert(height(Tin) == 2 && Tin.ID(1) == 2 && Tin.Score(2) == 300);
        Tout = outerjoin(T1, T2, 'Keys', 'ID');
        assert(height(Tout) == 4 && isnan(Tout.Score(1)) && isnan(Tout.Val(4)));

        T3 = table({'A'; 'B'; 'A'; 'B'; 'A'}, [10; 20; 30; 40; 50], 'VariableNames', {'Cat', 'X'});
        G = groupsummary(T3, 'Cat', 'mean', 'X');
        assert(height(G) == 2 && G.GroupCount(1) == 3 && abs(G.mean_X(1) - 30) < 1e-12);
        assert(G.GroupCount(2) == 2 && abs(G.mean_X(2) - 30) < 1e-12);

        assert(height(head(Tout, 2)) == 2 && height(tail(Tout, 2)) == 2);
        A = [1, 2; 3, 4];
        TA = array2table(A, 'VariableNames', {'C1', 'C2'});
        TA = renamevars(TA, 'C1', 'X1');
        assert(norm(table2array(TA) - A) < 1e-12);
        TA2 = removevars(TA, 'C2');
        assert(width(TA2) == 1);

        % 4. fminbnd & integral2
        [xmin, fmin] = fminbnd(@(x) (x - 2.3)^2 + 1.5, 0, 5);
        assert(abs(xmin - 2.3) < 1e-6 && abs(fmin - 1.5) < 1e-10);

        % int_0^1 int_0^x (x + y) dy dx = int_0^1 (3/2)*x^2 dx = 0.5
        q2 = integral2(@(x, y) x + y, 0, 1, 0, @(x) x);
        assert(abs(q2 - 0.5) < 1e-10);
        "#,
    )
    .unwrap();
}

#[test]
fn toolbox_statistics_and_ml_complete() {
    let mut vm = Vm::new();
    vm.eval(
        r#"
        % 1. Special functions & probability distributions
        assert(abs(gamma(5) - 24) < 1e-10);
        assert(abs(gammaln(6) - log(120)) < 1e-10);
        assert(abs(tcdf(0, 10) - 0.5) < 1e-12);
        assert(abs(tinv(tcdf(2.1, 15), 15) - 2.1) < 1e-6);
        assert(abs(chi2inv(chi2cdf(5.5, 4), 4) - 5.5) < 1e-6);
        assert(fcdf(1.0, 5, 5) > 0.49 && fcdf(1.0, 5, 5) < 0.51);
        assert(abs(expinv(expcdf(3.0, 2.0), 2.0) - 3.0) < 1e-10);
        assert(abs(binopdf(2, 4, 0.5) - 0.375) < 1e-10);
        assert(abs(binocdf(4, 4, 0.5) - 1.0) < 1e-10);
        assert(abs(poisspdf(0, 2.0) - exp(-2.0)) < 1e-10);
        assert(poisscdf(5, 2.0) > 0.98);

        % Descriptive stats
        [z, mu, sd] = zscore([2, 4, 6, 8]);
        assert(abs(mu - 5) < 1e-12 && abs(mean(z)) < 1e-12 && abs(std(z) - 1) < 1e-12);
        assert(abs(skewness([-2, -1, 0, 1, 2])) < 1e-12);
        assert(abs(iqr([1, 2, 3, 4, 5, 6, 7, 8, 9]) - 4.5) < 1e-12);

        % 2. fitlm, regress, anova1, ttest2
        X = [1; 2; 3; 4; 5; 6];
        y = 3.0 + 2.5 * X;
        mdl = fitlm(X, y);
        assert(abs(mdl.Coefficients.Estimate(1) - 3.0) < 1e-10);
        assert(abs(mdl.Coefficients.Estimate(2) - 2.5) < 1e-10);
        assert(abs(mdl.Rsquared.Ordinary - 1.0) < 1e-12);
        yp = predict(mdl, [10; 20]);
        assert(abs(yp(1) - 28.0) < 1e-10 && abs(yp(2) - 53.0) < 1e-10);
        yp2 = mdl.predict([10]);
        assert(abs(yp2(1) - 28.0) < 1e-10);

        [b, bint, r] = regress(y, [ones(6, 1), X]);
        assert(abs(b(1) - 3.0) < 1e-10 && abs(b(2) - 2.5) < 1e-10 && norm(r) < 1e-10);

        % One-way ANOVA on 3 clearly separated groups -> p << 0.001
        Gmat = [1, 10, 20; 2, 11, 21; 1.5, 10.5, 20.5; 1.2, 9.8, 19.7];
        [p_anova, st] = anova1(Gmat);
        assert(p_anova < 1e-5 && st.F > 100);
        [h2, p2] = ttest2([1, 2, 1.5, 1.8], [10, 11, 10.5, 10.8]);
        assert(h2 == 1 && p2 < 1e-4);

        % 3. Supervised ML: fitcknn, fitcsvm, fitcnb
        Xtr = [-3, -2; -2, -3; -2.5, -2.5; -1.8, -2.2; ...
                3,  2;  2,  3;  2.5,  2.5;  1.8,  2.2];
        Ytr = [0; 0; 0; 0; 1; 1; 1; 1];
        Xte = [-2.2, -2.1; 2.4, 2.6];

        knn = fitcknn(Xtr, Ytr, 'NumNeighbors', 3);
        p_knn = predict(knn, Xte);
        assert(p_knn(1) == 0 && p_knn(2) == 1);

        svm = fitcsvm(Xtr, Ytr);
        p_svm = predict(svm, Xte);
        assert(p_svm(1) == 0 && p_svm(2) == 1);

        nb = fitcnb(Xtr, Ytr);
        p_nb = nb.predict(Xte);
        assert(p_nb(1) == 0 && p_nb(2) == 1);

        % 4. Unsupervised hierarchical clustering: pdist, squareform, linkage, cluster
        pts = [0, 0; 0.1, 0.1; 10, 10; 10.2, 9.9];
        D = pdist(pts);
        Dm = squareform(D);
        assert(size(Dm, 1) == 4 && abs(Dm(1, 3) - sqrt(200)) < 1e-10);
        Z = linkage(D, 'average');
        T = cluster(Z, 'maxclust', 2);
        assert(T(1) == T(2) && T(3) == T(4) && T(1) ~= T(3));
        "#,
    )
    .unwrap();
}

#[test]
fn toolbox_signal_processing_complete() {
    let mut vm = Vm::new();
    vm.eval(
        r#"
        % 1. cheby1, cheby2, fir1, freqz, grpdelay
        [b1, a1] = cheby1(4, 1.0, 0.3);
        [H1, w1] = freqz(b1, a1, 128);
        assert(abs(H1(1)) > 0.85 && abs(H1(end)) < 0.05);

        [b2, a2] = cheby2(4, 40, 0.3);
        [H2, ~] = freqz(b2, a2, 128);
        assert(abs(H2(1)) > 0.95 && abs(H2(end)) < 0.05);

        bf = fir1(20, 0.4);
        [Hf, ~] = freqz(bf, 1, 128);
        assert(abs(Hf(1) - 1.0) < 1e-6 && abs(Hf(end)) < 0.05);
        [gd, ~] = grpdelay(bf, 1, 64);
        % Linear-phase Type I FIR of order 20 has constant group delay = 10 samples in passband!
        assert(abs(gd(5) - 10.0) < 1e-6);

        % 2. hilbert & envelope on pure cosine -> |hilbert(cos)| == 1
        t = (0:127) / 128;
        sig = cos(2 * pi * 8 * t);
        z = hilbert(sig);
        assert(max(abs(abs(z) - 1.0)) < 1e-10);
        [up, lo] = envelope(sig);
        assert(abs(mean(up) - 1.0) < 1e-6 && abs(mean(lo) + 1.0) < 1e-6);

        % 3. spectrogram
        [S, F, T] = spectrogram(sig, 32, 16, 64, 128);
        assert(size(S, 1) == 33 && size(S, 2) == length(T) && length(F) == 33);

        % 4. medfilt1, sgolayfilt, detrend, downsample, upsample, resample, windows
        spiky = [1, 1, 100, 1, 1, 1];
        cleaned = medfilt1(spiky, 3);
        assert(cleaned(3) == 1);

        xs = linspace(0, 4, 25);
        quad = xs.^2;
        % Order-2 Savitzky-Golay reproduces interior quadratic points accurately
        sg = sgolayfilt(quad, 2, 5);
        assert(max(abs(sg(4:22) - quad(4:22))) < 1e-10);

        dt = detrend(3 + 2 * (0:9));
        assert(max(abs(dt)) < 1e-10);

        assert(length(downsample(1:10, 2)) == 5);
        assert(length(upsample([1, 2, 3], 3)) == 9);
        rs = resample(sin(2*pi*0.05*(0:39)), 2, 1);
        assert(length(rs) == 80);

        wk = kaiser(11, 5.0);
        assert(abs(wk(6) - 1.0) < 1e-12 && abs(wk(1) - wk(11)) < 1e-12);
        wb = bartlett(5);
        assert(abs(wb(1)) < 1e-12 && abs(wb(3) - 1.0) < 1e-12);
        "#,
    )
    .unwrap();
}

#[test]
fn toolbox_optimization_complete() {
    let mut vm = Vm::new();
    vm.eval(
        r#"
        % 1. fminunc (BFGS) on 2D Rosenbrock function -> minimum at (1, 1), f = 0
        [x_u, f_u] = fminunc(@(x) (1 - x(1))^2 + 100*(x(2) - x(1)^2)^2, [-1.2; 1.0]);
        assert(norm(x_u - [1; 1]) < 1e-4 && f_u < 1e-8);

        % 2. quadprog: min 0.5*(x1^2 + x2^2) - 2*x1 - 3*x2 s.t. x1 + x2 <= 3, x >= 0
        % Unconstrained min is (2, 3), sum=5 > 3 => active boundary x1+x2=3 => x1=1, x2=2!
        H = [1, 0; 0, 1];
        f = [-2; -3];
        [x_q, f_q] = quadprog(H, f, [1, 1], 3, [], [], [0; 0], []);
        assert(abs(x_q(1) - 1.0) < 1e-4 && abs(x_q(2) - 2.0) < 1e-4);
        assert(abs(f_q - (0.5*(1+4) - 2 - 6)) < 1e-3);

        % 3. lsqlin: min ||C*x - d||^2 s.t. x >= 0
        C = [1, 0; 0, 1];
        d = [2; -5];
        [x_ls, resn] = lsqlin(C, d, [], [], [], [], [0; 0], []);
        assert(abs(x_ls(1) - 2.0) < 1e-4 && abs(x_ls(2)) < 1e-4 && abs(resn - 25.0) < 1e-3);

        % 4. fmincon with nonlinear inequality constraint:
        % min (x-2)^2 + (y-2)^2 s.t. x^2 + y^2 <= 1 => optimum at (1/sqrt(2), 1/sqrt(2))!
        function [c, ceq] = unitdisk(v)
            c = v(1)^2 + v(2)^2 - 1;
            ceq = [];
        end
        [x_c, f_c] = fmincon(@(v) (v(1)-2)^2 + (v(2)-2)^2, [0.5; 0.5], [], [], [], [], [], [], @unitdisk);
        target = 1 / sqrt(2);
        assert(abs(x_c(1) - target) < 1e-3 && abs(x_c(2) - target) < 1e-3);
        assert(x_c(1)^2 + x_c(2)^2 <= 1.001);
        "#,
    )
    .unwrap();
}

#[test]
fn toolbox_control_and_image_processing_complete() {
    let mut vm = Vm::new();
    vm.eval(
        r#"
        % --- CONTROL SYSTEM TOOLBOX ---
        % 1. ss, tf2ss, ss2tf, dcgain, pole, zero
        [A, B, C, D] = tf2ss([2, 6], [1, 5, 6]); % G(s) = 2(s+3)/((s+2)(s+3)) = 2/(s+2)
        sys = ss(A, B, C, D);
        assert(abs(dcgain(sys) - 1.0) < 1e-10);
        [num2, den2] = ss2tf(A, B, C, D);
        assert(abs(num2(end) - 6.0) < 1e-10 && abs(den2(end) - 6.0) < 1e-10);
        ps = sort(real(pole(sys)));
        assert(abs(ps(1) - (-3)) < 1e-8 && abs(ps(2) - (-2)) < 1e-8);
        zs = zero(sys);
        assert(abs(real(zs(1)) - (-3)) < 1e-8);

        % 2. feedback, series, parallel, impulse, lsim, margin
        G = tf(1, [1, 1]);
        CL = feedback(G, 1); % 1 / (s + 2) => dcgain = 0.5, pole = -2
        assert(abs(dcgain(CL) - 0.5) < 1e-10);
        assert(abs(real(pole(CL)) - (-2.0)) < 1e-8);

        t = linspace(0, 4, 81);
        [yi, ~] = impulse(CL, t); % exact: exp(-2*t)
        assert(abs(yi(1) - 1.0) < 1e-6 && abs(yi(end) - exp(-8)) < 1e-4);

        [yl, ~] = lsim(G, ones(size(t)), t); % unit step on 1/(s+1): 1 - exp(-t)
        assert(abs(yl(end) - (1 - exp(-4))) < 1e-4);

        % Third-order plant G3 = 1/(s+1)^3 has finite gain margin Gm = 8 at Wcg = sqrt(3)
        G3 = tf(1, [1, 3, 3, 1]);
        [Gm, Pm, Wcg, Wcp] = margin(G3);
        assert(abs(Gm - 8.0) < 0.25 && abs(Wcg - sqrt(3)) < 0.05);

        % 3. ctrb, obsv, place, lyap, care, lqr, lqe
        % Double integrator: A = [0 1; 0 0], B = [0; 1], C = [1 0]
        Adi = [0, 1; 0, 0];
        Bdi = [0; 1];
        Cdi = [1, 0];
        assert(rank(ctrb(Adi, Bdi)) == 2 && rank(obsv(Adi, Cdi)) == 2);

        K_place = place(Adi, Bdi, [-2, -3]);
        assert(abs(K_place(1) - 6.0) < 1e-8 && abs(K_place(2) - 5.0) < 1e-8);

        % LQR on double integrator with Q = I, R = 1 => exact S = [sqrt(3), 1; 1, sqrt(3)], K = [1, sqrt(3)]
        [K_lqr, S_lqr, P_lqr] = lqr(Adi, Bdi, eye(2), 1);
        assert(abs(K_lqr(1) - 1.0) < 1e-8 && abs(K_lqr(2) - sqrt(3)) < 1e-8);
        assert(all(real(P_lqr) < -0.5));

        % Kalman filter gain (lqe) on double integrator
        [L_kal, ~, E_kal] = lqe(Adi, eye(2), Cdi, eye(2), 1);
        assert(length(L_kal) == 2 && all(real(E_kal) < -0.5));

        % --- IMAGE PROCESSING TOOLBOX BASICS ---
        rgb = uint8(zeros(8, 8, 3));
        rgb(:, 5:8, 1) = 200;
        rgb(:, 5:8, 2) = 200;
        rgb(:, 5:8, 3) = 200;
        gray = rgb2gray(rgb);
        assert(strcmp(class(gray), 'uint8') && size(gray, 1) == 8 && size(gray, 2) == 8);
        assert(double(gray(1, 1)) == 0 && abs(double(gray(1, 6)) - 200) <= 1);

        % Otsu binarization separates dark left half from bright right half
        th = graythresh(gray);
        assert(th > 0.05 && th < 0.8);
        bw = imbinarize(gray);
        assert(all(all(bw(:, 1:4) == 0)) && all(all(bw(:, 5:8) == 1)));

        % Sobel edge detects the vertical boundary at columns 4-5
        ed = edge(gray, 'sobel');
        assert(any(ed(:, 4)) || any(ed(:, 5)));
        assert(~any(ed(:, 1)) && ~any(ed(:, 8)));

        % Resize, Gaussian blur, imhist, imadjust
        big = imresize(gray, 2);
        assert(size(big, 1) == 16 && size(big, 2) == 16);
        blurred = imgaussfilt(im2double(gray), 1.0);
        assert(blurred(4, 4) > 0.05 && blurred(4, 4) < 0.95);
        [cnts, locs] = imhist(gray, 16);
        assert(sum(cnts) == 64 && length(locs) == 16);
        adj = imadjust(gray);
        assert(double(max(max(adj))) == 255);
        "#,
    )
    .unwrap();
}
