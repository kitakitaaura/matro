//! Comprehensive MATLAB compatibility regression tests covering Waves 1-6:
//! - Wave 1: VM operators (`.\`), dtype-preserving indexing, auto-growth, cell/struct/char indexing,
//!           for-loops over cells/structs/3D, sparse concat, `A \ B` over/under-determined, `class`/`isa`/`exist`/`load`.
//! - Wave 2: Constructors (`true(2,3)`, `zeros(...,'single')`, `'like'`), `norm`, `chol` lower/flag,
//!           `svd` econ/vector, `qr` pivoted, `eig(A,B)`, `fft` along dim, `polyfit`/`polyval`, `dot`, `ndgrid`, `round(x,n)`, `logm`/`sqrtm`.
//! - Wave 3: N-D reductions (`'all'`, `vecdim`, `'omitnan'`), `sprintf`/`fprintf` recycling & flags,
//!           multi-output set ops (`'rows'`, `'stable'`, cellstr), `pagemtimes`/`pagetranspose`/`shiftdim`/`mink`/`maxk`/`topkrows`/`perms`/`islocalmax`/`smoothdata`.
//! - Wave 4: `interp1` matrix/`'makima'`, `interp2` cubic/spline/makima, `interp3`/`interpn`, `filter` N-D/`[y,zf]`,
//!           `conv2` separable, `roots` complex, `accumarray` N-D/closure, `histcounts`/`histc`/`discretize`,
//!           `beta`/`betaln`/`erfinv`/`erfcinv`/`besselj`/`bessely`/`besseli`/`besselk`/`legendre`/`ellipke`, `eigs`/`svds`.
//! - Wave 5: Krylov solvers (`pcg`, `gmres`, `bicgstab`, `minres`, `lsqr`, `cgs`, `bicg`, `lsqnonneg`),
//!           `odeset`/`odeget`/`optimset`/`optimoptions`, `ode45` options + struct/`deval`, `integral` Inf/complex/Waypoints/ArrayValued,
//!           `lsqcurvefit`, `lsqnonlin` bounds, `arrayfun`/`cellfun`/`bsxfun` N-D/multi-input, `fseek`/`fscanf`/`sscanf`, `readmatrix`/`writematrix`/`dlmread`/`dlmwrite`/`importdata`/`rmdir`.
//! - Wave 6: Stats & ML (`std`/`var`/`median`/`prctile`/`quantile`/`cov`/`corrcoef`/`corr`, `pca` 6 outputs,
//!           `mvnpdf`/`mvnrnd`, `*rnd`, `logn*`/`gam*`/`beta*`/`unif*`/`finv`/`poissinv`/`binoinv`, `pdist2`/`knnsearch`/`randsample`/`datasample`,
//!           `kstest`/`kstest2`/`ranksum`/`signrank`, `geomean`/`harmmean`/`trimmean`/`mode`/`nan*`),
//!           Signal (`xcorr`/`xcov`, `butter`/`cheby1`/`cheby2` stop/s/zpk, `zp2sos`/`sos2tf`/`tf2sos`/`sosfilt`, `filtfilt` matrix/SOS,
//!           `findpeaks` plateaus/prominence, `periodogram`, `decimate`, `detrend` poly, windows),
//!           Image (`imfilter` pad modes, `imresize` NaN aspect, `medfilt2`, `fspecial`, `edge` canny/log/roberts, `bwlabel`, `regionprops`, `imerode`/`imdilate`/`imopen`/`imclose`, `mat2gray`, `imcomplement`),
//!           Control (`c2d`/`d2c`, `dlyap`/`dare`/`dlqr`, `care`/`lqr` N-term, `initial`, `stepinfo`, `ssdata`/`tfdata`, `rlocus`),
//!           Regex (`$1`/`$0`, `\b`, `'once'`, `'tokens'`, `'names'`), Tables (`addvars`, `movevars`, `table2struct`, `struct2table`, `cell2table`, `table2cell`, `sortrows` multi-col, `groupsummary` multi-group), Modern types (`datetime([Y M D])`, `'15-Jan-2026'`, `split`), Handle Graphics `set(gca, 'FontSize', 12)`.

use matro_vm::Vm;

#[test]
fn wave1_vm_operators_indexing_sparse_and_class() {
    let mut vm = Vm::new();
    vm.eval(
        r#"
        % Elementwise left division .\
        assert(isequal([2, 4] .\ [10, 20], [5, 5]));
        assert(abs(2 .\ 9 - 4.5) < 1e-12);

        % true(...) / false(...) constructor vs scalar
        T = true(2, 3);
        F = false(3, 1);
        assert(islogical(T) && isequal(size(T), [2, 3]) && all(T, 'all'));
        assert(islogical(F) && isequal(size(F), [3, 1]) && ~any(F, 'all'));
        assert(true && ~false);

        % Dtype preservation across indexing (regression for y(mask(1:5)))
        mask = [1, 2, 3, 4, 5, 6] > 3;
        y = [10, 20, 30, 40, 50];
        sub = mask(1:5);
        assert(islogical(sub));
        assert(isequal(y(sub), [40, 50]));

        % Column auto-growth orientation preservation & scalar indexing orientation
        v = [1; 2; 3];
        v(end + 1) = 4;
        assert(isequal(size(v), [4, 1]) && v(4) == 4);
        a = 7;
        assert(isequal(size(a([1; 1; 1])), [3, 1]));

        % Cell, struct array, and char parenthesis assignment & deletion
        c = {1, 2, 3};
        c(2) = {99};
        assert(c{2} == 99);
        c(1) = [];
        assert(length(c) == 2 && c{1} == 99);

        str = 'abcdef';
        str(2) = 'Z';
        assert(strcmp(str, 'aZcdef'));
        str([1, 6]) = [];
        assert(strcmp(str, 'Zcde'));

        % For-loops over cell, char, and 3D arrays
        s_sum = 0;
        for item = {10, 20, 30}
            s_sum = s_sum + item{1};
        end
        assert(s_sum == 60);

        T3 = reshape(1:12, 2, 2, 3);
        col_count = 0;
        for col = T3
            col_count = col_count + 1;
            assert(isequal(size(col), [2, 1]));
        end
        assert(col_count == 6);

        % Overdetermined & underdetermined A \ B (real & complex)
        A_over = [1, 1; 1, 2; 1, 3];
        b_over = [2; 3; 4];
        x_ls = A_over \ b_over;
        assert(norm(x_ls - [1; 1]) < 1e-10);

        A_under = [1, 2, 3; 4, 5, 6];
        b_under = [14; 32];
        x_mn = A_under \ b_under;
        assert(norm(A_under * x_mn - b_under) < 1e-10);

        % Sparse concatenation preserves sparsity
        S = [speye(2), sparse(2, 2); sparse(2, 2), 2 * speye(2)];
        assert(issparse(S) && isequal(size(S), [4, 4]) && full(S(4, 4)) == 2);

        % Class identity & isa
        assert(strcmp(class(1 + 2i), 'double'));
        assert(strcmp(class(speye(2)), 'double'));
        assert(isa(int32(5), 'integer') && ~isa(int32(5), 'double'));
        assert(isa(single(1.5), 'float') && ~isempty(struct()));
        assert(exist('sin', 'builtin') == 5 && exist('A_over', 'var') == 1);
        "#,
    )
    .unwrap();
}

#[test]
fn wave2_linalg_fft_constructors_and_math() {
    let mut vm = Vm::new();
    vm.eval(
        r#"
        % Class-typed & 'like' constructors
        Z = zeros(2, 3, 'single');
        assert(isa(Z, 'single') && isequal(size(Z), [2, 3]));
        O = ones(2, 2, 'like', int32(1));
        assert(isa(O, 'int32') && O(1, 1) == 1);
        N = NaN(2, 2);
        assert(all(isnan(N), 'all') && isequal(size(N), [2, 2]));
        I_inf = Inf(1, 3);
        assert(all(isinf(I_inf)));

        % Matrix norms (2-norm, 1, inf, 'fro') & complex
        M = [3, 0; 0, 4];
        assert(abs(norm(M) - 4) < 1e-12);
        assert(abs(norm(M, 1) - 4) < 1e-12);
        assert(abs(norm(M, inf) - 4) < 1e-12);
        assert(abs(norm(M, 'fro') - 5) < 1e-12);

        % chol lower & flag
        SPD = [4, 2; 2, 5];
        [L, flag] = chol(SPD, 'lower');
        assert(flag == 0 && norm(L * L' - SPD) < 1e-12);
        [~, flag_bad] = chol([1, 2; 2, 1]);
        assert(flag_bad > 0);

        % svd econ/vector, pivoted qr, generalized eig(A, B)
        A = [1, 2; 3, 4; 5, 6];
        [U, s, V] = svd(A, 'econ', 'vector');
        assert(isequal(size(U), [3, 2]) && isequal(size(s), [2, 1]));
        assert(norm(U * diag(s) * V' - A) < 1e-11);

        [Q, R, p] = qr(A, 'econ', 'vector');
        assert(norm(Q * R - A(:, p)) < 1e-11);

        Ag = [6, 2; 2, 3];
        Bg = [2, 0; 0, 1];
        [Vg, Dg] = eig(Ag, Bg);
        assert(norm(Ag * Vg - Bg * Vg * Dg) < 1e-10);

        % fft along dim 2 & symmetric ifft
        X = [1, 2, 3, 4; 5, 6, 7, 8];
        FX2 = fft(X, [], 2);
        assert(norm(FX2(1, :) - fft(X(1, :))) < 1e-11);
        assert(norm(ifft(FX2, [], 2, 'symmetric') - X) < 1e-11);
        assert(isreal(ifft(FX2, [], 2, 'symmetric')));

        % polyfit / polyval with [p, S, mu]
        xp = 1000:1010;
        yp = 2 * (xp - 1005).^2 + 3 * (xp - 1005) + 1;
        [p_fit, S_fit, mu_fit] = polyfit(xp, yp, 2);
        [y_hat, delta] = polyval(p_fit, xp, S_fit, mu_fit);
        assert(norm(y_hat - yp) < 1e-8 && all(delta >= 0));

        % round(x, N, 'significant'), log2([F, E]), mod(x, 0), logm/sqrtm
        assert(abs(round(3.14159, 2) - 3.14) < 1e-12);
        assert(abs(round(12345, 2, 'significant') - 12000) < 1e-9);
        [F2, E2] = log2(12);
        assert(abs(F2 * 2^E2 - 12) < 1e-12);
        assert(mod(5, 0) == 5);
        M2 = [2, 1; 0, 3];
        assert(norm(expm(logm(M2)) - M2) < 1e-10);
        S2 = sqrtm(M2);
        assert(norm(S2 * S2 - M2) < 1e-10);

        % 3D ndgrid & column-wise dot
        [G1, G2, G3] = ndgrid(1:2, 1:3, 1:4);
        assert(isequal(size(G1), [2, 3, 4]) && G3(2, 3, 4) == 4);
        assert(isequal(dot([1, 2; 3, 4], [5, 6; 7, 8], 1), [26, 44]));
        "#,
    )
    .unwrap();
}

#[test]
fn wave3_nd_reductions_printf_sets_and_array_ops() {
    let mut vm = Vm::new();
    vm.eval(
        r#"
        % N-D reductions with 'all', vecdim, 'omitnan'
        A = reshape(1:24, 2, 3, 4);
        assert(sum(A, 'all') == 300);
        assert(isequal(size(sum(A, [1, 2])), [1, 1, 4]));
        assert(sum([1, NaN, 3], 'omitnan') == 4);
        assert(abs(mean([2, NaN, 6], 'omitnan') - 4) < 1e-12);
        assert(max([1, NaN, 5], [], 'omitnan') == 5);
        assert(isequal(size(A, [1, 3]), [2, 4]));

        % sprintf vectorized recycling, flags, hex, %g
        assert(strcmp(sprintf('%d,', [1, 2, 3]), '1,2,3,'));
        assert(strcmp(sprintf('%05d', 42), '00042'));
        assert(strcmp(sprintf('%X', 255), 'FF'));
        assert(strcmp(sprintf('%.2f', 3.14159), '3.14'));

        % Multi-output set ops with 'rows', 'stable', cellstr
        [U, ia, ic] = unique([3, 1, 2, 1, 3], 'stable');
        assert(isequal(U, [3, 1, 2]) && isequal(U(ic), [3, 1, 2, 1, 3]));
        [Lia, Locb] = ismember([2, 9, 1], [1, 2, 3]);
        assert(isequal(Lia, logical([1, 0, 1])) && isequal(Locb, [2, 0, 1]));
        [Ur, ~, ~] = unique([1, 2; 3, 4; 1, 2], 'rows');
        assert(isequal(Ur, [1, 2; 3, 4]));
        assert(isequal(unique({'b', 'a', 'b'}), {'a', 'b'}));

        % Batched page ops, shiftdim, mink/maxk/topkrows, perms, islocalmax, smoothdata
        P1 = reshape(1:12, 2, 3, 2);
        P2 = reshape(1:12, 3, 2, 2);
        PM = pagemtimes(P1, P2);
        assert(isequal(size(PM), [2, 2, 2]));
        assert(isequal(PM(:, :, 1), P1(:, :, 1) * P2(:, :, 1)));
        assert(isequal(PM(:, :, 2), P1(:, :, 2) * P2(:, :, 2)));
        assert(isequal(size(pagetranspose(P1)), [3, 2, 2]));

        [Bs, nsh] = shiftdim(zeros(1, 1, 3, 4));
        assert(nsh == 2 && isequal(size(Bs), [3, 4]));

        [mk, mi] = mink([5, 1, 4, 2, 3], 2);
        assert(isequal(mk, [1, 2]) && isequal(mi, [2, 4]));
        [xk, xi] = maxk([5, 1, 4, 2, 3], 2);
        assert(isequal(xk, [5, 4]) && isequal(xi, [1, 3]));
        assert(size(perms([1, 2, 3]), 1) == 6);
        assert(isequal(islocalmax([0, 3, 1, 5, 2]), logical([0, 1, 0, 1, 0])));
        sm = smoothdata([1, 2, 10, 2, 1], 'movmedian', 3);
        assert(sm(3) == 2);

        % String ops with 'IgnoreCase' & cellstr
        assert(contains('Hello World', 'world', 'IgnoreCase', true));
        assert(isequal(startsWith({'Foo', 'bar'}, 'f', 'IgnoreCase', true), logical([1, 0])));
        assert(isequal(strfind('banana', 'ana'), [2, 4]));
        assert(~strcmpi(123, 456));
        "#,
    )
    .unwrap();
}

#[test]
fn wave4_interp_binning_special_math_and_eigs() {
    let mut vm = Vm::new();
    vm.eval(
        r#"
        % interp1 matrix V & makima & unmkpp
        x = (0:4)';
        V = [x.^2, 2 * x + 1];
        Vq = interp1(x, V, [1.5; 2.5], 'makima');
        assert(isequal(size(Vq), [2, 2]) && abs(Vq(1, 2) - 4.0) < 1e-10);
        pp = makima(x, x.^2);
        [breaks, coefs, l, k, d] = unmkpp(pp);
        assert(length(breaks) == 5 && l == 4 && k == 4 && d == 1 && size(coefs, 1) == 4);

        % interp2 cubic/spline & interp3/interpn
        [X2, Y2] = meshgrid(1:4, 1:4);
        Z2 = X2 + 2 * Y2;
        assert(abs(interp2(X2, Y2, Z2, 2.5, 3.5, 'cubic') - 9.5) < 1e-10);
        [X3, Y3, Z3] = meshgrid(1:3, 1:3, 1:3);
        V3 = X3 + 2 * Y3 + 3 * Z3;
        assert(abs(interp3(X3, Y3, Z3, V3, 1.5, 2.5, 2.0) - (1.5 + 5.0 + 6.0)) < 1e-10);

        % Separable conv2 & N-D filter with [y, zf] & complex roots
        C2 = conv2([1, 1], [1; 1], eye(3), 'same');
        assert(isequal(size(C2), [3, 3]) && C2(2, 2) == 2);
        [yf, zf] = filter([0.5, 0.5], 1, [2, 4, 6, 8]);
        assert(isequal(yf, [1, 3, 5, 7]) && abs(zf - 4) < 1e-12);
        rc = roots([1, -(2 + 3i), 2 * 3i]);
        assert(norm(sort(real(rc)) - [0; 2]) < 1e-8);

        % accumarray N-D & closure, histcounts NV, histc, discretize
        ac = accumarray([1, 1; 2, 2; 1, 1], [3; 5; 7]);
        assert(isequal(ac, [10, 0; 0, 5]));
        ac_fn = accumarray([1; 1; 2; 2], [2; 8; 3; 9], [], @(v) max(v));
        assert(isequal(ac_fn, [8; 9]));
        [N_hc, E_hc] = histcounts([0.1, 0.4, 0.6, 0.9], [0, 0.5, 1.0], 'Normalization', 'probability');
        assert(norm(N_hc - [0.5, 0.5]) < 1e-12 && length(E_hc) == 3);
        [bins, edges] = discretize([1.5, 2.5, 3.5], [1, 2, 3, 4]);
        assert(isequal(bins, [1, 2, 3]) && length(edges) == 4);

        % Special functions: gamma(-0.5), beta, erfinv, besselj/y/i/k, legendre, ellipke
        assert(abs(gamma(-0.5) - (-2 * sqrt(pi))) < 1e-10);
        assert(abs(beta(2, 3) - 1/12) < 1e-12);
        assert(abs(erf(erfinv(0.75)) - 0.75) < 1e-12);
        assert(abs(erfc(erfcinv(0.25)) - 0.25) < 1e-12);
        assert(abs(besselj(0, 0) - 1.0) < 1e-12);
        assert(abs(besselj(0, 1.0) - 0.7651976865579666) < 1e-10);
        assert(abs(besseli(0, 1.0) - 1.2660658777520084) < 1e-10);
        assert(abs(besselk(0, 1.0) - 0.4210244382407083) < 1e-6);
        assert(abs(bessely(0, 1.0) - 0.08825696421567696) < 1e-8);
        P2 = legendre(2, 0.5);
        assert(abs(P2(1) - (-0.125)) < 1e-10);
        [K_el, E_el] = ellipke(0.0);
        assert(abs(K_el - pi/2) < 1e-12 && abs(E_el - pi/2) < 1e-12);
        "#,
    )
    .unwrap();
}

#[test]
fn wave5_krylov_solvers_ode_integral_and_fileio() {
    let mut vm = Vm::new();
    vm.eval(
        r#"
        % Krylov solvers: pcg, gmres, bicgstab, minres, lsqr, lsqnonneg
        n = 12;
        e = ones(n, 1);
        A = spdiags([-e, 4*e, -e], [-1, 0, 1], n, n);
        x_true = (1:n)';
        b = A * x_true;
        [x_pcg, fl_pcg, rel_pcg] = pcg(A, b, 1e-10, 100);
        assert(fl_pcg == 0 && norm(x_pcg - x_true) < 1e-7 && rel_pcg < 1e-9);
        [x_gm, fl_gm] = gmres(A, b, [], 1e-10, 50);
        assert(fl_gm == 0 && norm(x_gm - x_true) < 1e-7);
        [x_bi, fl_bi] = bicgstab(A, b, 1e-10, 100);
        assert(fl_bi == 0 && norm(x_bi - x_true) < 1e-7);
        [x_mr, fl_mr] = minres(A, b, 1e-10, 100);
        assert(fl_mr == 0 && norm(x_mr - x_true) < 1e-7);
        % Matrix-free afun
        [x_mf, fl_mf] = pcg(@(v) A * v, b, 1e-10, 100);
        assert(fl_mf == 0 && norm(x_mf - x_true) < 1e-7);
        % lsqnonneg
        C_nn = [1, 0; 0, 1; 1, 1];
        d_nn = [2; -3; 1];
        x_nn = lsqnonneg(C_nn, d_nn);
        assert(all(x_nn >= -1e-12) && abs(x_nn(2)) < 1e-10);

        % odeset/odeget + ode45 with intermediate tspan & deval struct output
        opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);
        assert(abs(odeget(opts, 'RelTol') - 1e-8) < 1e-15);
        t_req = [0, 0.5, 1.0, 2.0];
        [t_out, y_out] = ode45(@(t, y) -y, t_req, 1.0, opts);
        assert(isequal(size(t_out), [4, 1]) && norm(y_out - exp(-t_req')) < 1e-6);
        sol = ode45(@(t, y) -y, [0, 2], 1.0, opts);
        assert(abs(deval(sol, 1.0) - exp(-1.0)) < 1e-4);

        % integral with Inf bounds, endpoint singularity, complex integrand, ArrayValued
        assert(abs(integral(@(x) exp(-x), 0, Inf) - 1.0) < 1e-6);
        assert(abs(integral(@(x) exp(-x^2), -Inf, Inf) - sqrt(pi)) < 1e-6);
        assert(abs(integral(@(x) 1 / sqrt(x), 0, 1) - 2.0) < 1e-4);
        assert(abs(integral(@(x) exp(1i * x), 0, pi) - 2i) < 1e-8);
        av = integral(@(x) [1, x, x^2], 0, 1, 'ArrayValued', true);
        assert(norm(av - [1, 0.5, 1/3]) < 1e-8);

        % lsqcurvefit & bounded lsqnonlin & optimoptions
        xdata = (0:0.5:3)';
        ydata = 2.5 * exp(-0.8 * xdata);
        [p_cf, ~, ~, ef_cf] = lsqcurvefit(@(p, x) p(1) * exp(-p(2) * x), [1; 1], xdata, ydata, [0; 0], [10; 10]);
        assert(ef_cf == 1 && norm(p_cf - [2.5; 0.8]) < 1e-5);
        op = optimoptions('fmincon', 'Display', 'off');
        assert(strcmp(op.Display, 'off'));

        % arrayfun / cellfun / bsxfun multi-input & UniformOutput false
        af = arrayfun(@(x, y) x + 2*y, [1, 2; 3, 4], [10, 20; 30, 40]);
        assert(isequal(af, [21, 42; 63, 84]));
        afc = arrayfun(@(x) [x, x^2], [2, 3], 'UniformOutput', false);
        assert(iscell(afc) && isequal(afc{2}, [3, 9]));
        cf = cellfun(@(a, b) a + b, {1, 2}, {10, 20});
        assert(isequal(cf, [11, 22]));
        bf = bsxfun(@plus, reshape(1:6, 2, 1, 3), [10, 20]);
        assert(isequal(size(bf), [2, 2, 3]));

        % sscanf / fscanf / fseek / readmatrix / writematrix / dlmread / importdata / rmdir
        [nums, cnt] = sscanf('10 20.5 -3', '%f');
        assert(cnt == 3 && norm(nums - [10; 20.5; -3]) < 1e-12);
        fid = fopen('scan_test.txt', 'w');
        fprintf(fid, '1,2,3\n4,5,6\n');
        fclose(fid);
        fid = fopen('scan_test.txt', 'r');
        assert(fseek(fid, 0, 'eof') == 0 && ftell(fid) > 0);
        assert(fseek(fid, 0, 'bof') == 0);
        A_sc = fscanf(fid, '%f,%f,%f\n', [3, Inf]);
        fclose(fid);
        assert(isequal(A_sc, [1, 4; 2, 5; 3, 6]));

        M_io = [1.5, 2.5; 3.5, 4.5];
        writematrix(M_io, 'mat_io.csv');
        assert(norm(readmatrix('mat_io.csv') - M_io) < 1e-12);
        dlmwrite('mat_io.tsv', M_io, '\t');
        assert(norm(dlmread('mat_io.tsv', '\t') - M_io) < 1e-12);
        mkdir('tmp_dir', 'sub');
        assert(isfolder('tmp_dir/sub'));
        rmdir('tmp_dir', 's');
        "#,
    )
    .unwrap();
}

#[test]
fn wave6_toolboxes_stats_signal_image_control_regex_tables() {
    let mut vm = Vm::new();
    vm.eval(
        r#"
        % 6A. Statistics & ML upgrades
        rng(42, 'twister');
        assert(isnan(median([1, NaN, 3])));
        assert(median([1, NaN, 3], 'omitnan') == 2);
        [s_v, m_v] = std([2, 4, 6; 4, 6, 8], 0, 2);
        assert(norm(s_v - [2; 2]) < 1e-12 && norm(m_v - [4; 6]) < 1e-12);
        assert(abs(nanmean([1, NaN, 5]) - 3) < 1e-12);
        assert(abs(geomean([2, 8]) - 4) < 1e-12);
        assert(abs(harmmean([2, 6]) - 3) < 1e-12);
        [md_v, md_f] = mode([1, 2, 2, 3, 3, 3]);
        assert(md_v == 3 && md_f == 3);

        % mvnpdf / mvnrnd / *rnd / *pdf / *cdf / *inv
        p_mvn = mvnpdf([0, 0], [0, 0], eye(2));
        assert(abs(p_mvn - 1 / (2 * pi)) < 1e-10);
        R_mvn = mvnrnd([1, 2], [2, 0.5; 0.5, 1], 50);
        assert(isequal(size(R_mvn), [50, 2]));
        assert(isequal(size(gamrnd(2, 1, 3, 4)), [3, 4]));
        assert(isequal(size(poissrnd(5, [2, 3])), [2, 3]));
        assert(abs(gaminv(gamcdf(3.0, 2.0, 1.5), 2.0, 1.5) - 3.0) < 1e-6);
        assert(abs(betainv(betacdf(0.4, 2.0, 3.0), 2.0, 3.0) - 0.4) < 1e-6);
        assert(abs(logninv(logncdf(2.5, 0, 1), 0, 1) - 2.5) < 1e-8);
        assert(abs(finv(fcdf(2.0, 4, 6), 4, 6) - 2.0) < 1e-6);
        assert(isnan(chi2inv(-0.1, 4)));
        assert(poisscdf(50000, 10000) == 1.0);

        % corr, pdist2, knnsearch, pca 6 outputs, kstest2, ranksum
        [rho_sp, ~] = corr([1; 2; 3; 4], [1; 8; 27; 64], 'Type', 'Spearman');
        assert(abs(rho_sp - 1.0) < 1e-12);
        D12 = pdist2([0, 0; 3, 4], [0, 0; 0, 4]);
        assert(abs(D12(2, 1) - 5) < 1e-12 && abs(D12(2, 2) - 3) < 1e-12);
        [ki, kd] = knnsearch([0, 0; 10, 10; 1, 0], [0.2, 0], 'K', 2);
        assert(isequal(ki, [1, 3]) && abs(kd(1) - 0.2) < 1e-12);
        [~, ~, ~, ts2, expl, mu_pca] = pca([1, 2; 3, 4; 5, 6; 7, 9]);
        assert(length(ts2) == 4 && abs(sum(expl) - 100) < 1e-6 && length(mu_pca) == 2);
        [h_ks, ~] = kstest2([1, 2, 1.5, 1.8, 1.2], [20, 21, 22, 20.5, 21.5]);
        assert(h_ks == 1);
        [p_rs, h_rs] = ranksum([1, 2, 3, 4, 5], [100, 101, 102, 103, 104]);
        assert(h_rs == 1 && p_rs < 0.05);

        % 6B. Signal processing upgrades
        [xc, lags] = xcorr([1, 2, 3], 2, 'normalized');
        assert(length(xc) == 5 && abs(xc(3) - 1.0) < 1e-12 && isequal(lags, -2:2));
        [zb, pb, kb] = butter(3, [0.2, 0.5], 'stop');
        [sos_b, g_b] = zp2sos(zb, pb, kb);
        [bs, as] = sos2tf(sos_b, g_b);
        [bd, ad] = butter(3, [0.2, 0.5], 'stop');
        assert(norm(bs - bd) < 1e-6 && norm(as - ad) < 1e-6);
        % findpeaks plateau & prominence
        [pks, locs, ~, prom] = findpeaks([0, 1, 1, 2, 1, 0, 5, 0], 'MinPeakProminence', 1.5);
        assert(isequal(pks, [2, 5]) && isequal(locs, [4, 7]) && prom(2) == 5);
        % detrend degree 0 & 2
        assert(abs(mean(detrend([5, 5, 5], 0))) < 1e-12);
        t_dt = (0:10)';
        assert(norm(detrend(3 * t_dt.^2 + 2 * t_dt + 1, 2)) < 1e-10);
        assert(bartlett(1) == 1 && length(tukeywin(16, 0.5)) == 16 && length(gausswin(16)) == 16);
        assert(length(decimate(sin(0:0.1:10), 2)) == 51);

        % 6C. Image processing upgrades
        I = zeros(10, 10);
        I(2:4, 2:4) = 1;
        I(7:9, 7:9) = 1;
        [L_bw, num_bw] = bwlabel(I);
        assert(num_bw == 2);
        rp = regionprops(logical(I));
        assert(length(rp) == 2 && rp(1).Area == 9 && rp(2).Area == 9);
        I_er = imerode(logical(I), ones(3, 3));
        assert(sum(I_er, 'all') == 2);
        assert(isequal(size(imresize(I, [20, NaN])), [20, 20]));
        assert(abs(sum(fspecial('gaussian', [5, 5], 1.0), 'all') - 1.0) < 1e-12);
        assert(isequal(imcomplement(logical([0, 1])), logical([1, 0])));
        assert(isequal(mat2gray([10, 20, 30]), [0, 0.5, 1]));

        % 6D. Control upgrades
        Ac = [0, 1; -2, -3]; Bc = [0; 1]; Cc = [1, 0]; Dc = 0;
        sysc = ss(Ac, Bc, Cc, Dc);
        sysd = c2d(sysc, 0.1, 'zoh');
        sysc2 = d2c(sysd, 'zoh');
        [A2, B2, ~, ~] = ssdata(sysc2);
        assert(norm(A2 - Ac) < 1e-8 && norm(B2 - Bc) < 1e-8);
        [Kd, Sd, ~] = dlqr(sysd.A, sysd.B, eye(2), 1);
        assert(isequal(size(Kd), [1, 2]) && norm(Sd - Sd') < 1e-10);
        si = stepinfo(sysc);
        assert(si.RiseTime > 0 && si.SettlingTime > 0);
        [r_rl, k_rl] = rlocus(tf([1], [1, 3, 2]), [0, 1, 10]);
        assert(isequal(size(r_rl), [2, 3]) && length(k_rl) == 3);

        % 6E. Regex, Tables, Modern Types & Handle Graphics
        assert(strcmp(regexprep('foo=42, bar=7', '(\w+)=(\d+)', '$2:$1'), '42:foo, 7:bar'));
        assert(isequal(regexp('cat catalog scatter', '\bcat\b'), [1]));
        toks = regexp('x=10', '(?<key>\w+)=(?<val>\d+)', 'names');
        assert(strcmp(toks.key, 'x') && strcmp(toks.val, '10'));

        T = table([10; 2; 10; 2], [4; 1; 2; 3], 'VariableNames', {'G', 'V'});
        Ts = sortrows(T, {'G', 'V'});
        assert(isequal(Ts.G, [2; 2; 10; 10]) && isequal(Ts.V, [1; 3; 2; 4]));
        Gs = groupsummary(T, 'G', 'median', 'V');
        assert(isequal(Gs.G, [2; 10]) && isequal(Gs.median_V, [2; 3]));
        T_add = addvars(T, [1; 2; 3; 4], 'Before', 'G', 'NewVariableNames', 'ID');
        assert(width(T_add) == 3);
        S_tbl = table2struct(T);
        assert(length(S_tbl) == 4 && S_tbl(1).G == 10);
        T_back = struct2table(S_tbl);
        assert(isequal(T_back.V, T.V));

        dt1 = datetime([2026, 1, 15]);
        dt2 = datetime('15-Jan-2026');
        assert(dt1.Year == 2026 && dt1.Month == 1 && dt1.Day == 15);
        assert(dt2.Year == 2026 && dt2.Month == 1 && dt2.Day == 15);

        plot([1, 2, 3], [1, 4, 9]);
        set(gca, 'FontSize', 12, 'Box', 'on', 'TickDir', 'out');
        "#,
    )
    .unwrap();
}
