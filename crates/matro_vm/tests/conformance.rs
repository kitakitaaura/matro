//! End-to-end MATLAB/Octave conformance test suite written in `.m` scripts using `assert()`.

use matro_vm::Vm;

fn run_m(script: &str) {
    let mut vm = Vm::new();
    vm.set_step_limit(Some(5_000_000));
    if let Err(e) = vm.eval(script) {
        panic!("Conformance script failed: {e}");
    }
}

#[test]
fn conformance_cow_value_semantics() {
    run_m(
        r#"
        A = [1 2 3; 4 5 6];
        B = A;
        B(1, 2) = 99;
        assert(A(1, 2) == 2, 'Copy-on-Write leaked mutation to original matrix');
        assert(B(1, 2) == 99);

        function M = mutate_copy(M)
            M(1, 1) = -100;
        end
        C = mutate_copy(A);
        assert(A(1, 1) == 1, 'Function call must preserve MATLAB value semantics');
        assert(C(1, 1) == -100);
        "#,
    );
}

#[test]
fn conformance_logical_masks_and_vector_ops() {
    run_m(
        r#"
        x = -5:5;
        evens = x(mod(x, 2) == 0);
        assert(length(evens) == 5);
        assert(all(mod(evens, 2) == 0));

        y = x;
        y(y < 0) = 0;
        assert(sum(y) == 15);
        assert(any(x < 0) && ~any(y < 0));

        idx = find(x > 2);
        assert(length(idx) == 3 && idx(1) == 9);

        [sorted, order] = sort([30, 10, 20]);
        assert(sorted(1) == 10 && sorted(3) == 30);
        assert(order(1) == 2 && order(2) == 3 && order(3) == 1);

        assert(norm(conv([1 2], [3 4]) - [3 10 8]) < 1e-12);
        assert(abs(interp1([0 1 2], [0 10 20], 1.5) - 15) < 1e-12);
        "#,
    );
}

#[test]
fn conformance_try_catch_varargin_strings() {
    run_m(
        r#"
        function [out, count] = combine(sep, varargin)
            assert(nargin >= 1);
            out = strjoin(varargin, sep);
            count = length(varargin);
        end

        [joined, n] = combine('-', 'a', 'b', 'c');
        assert(strcmp(joined, 'a-b-c'));
        assert(n == 3);
        parts = strsplit(joined, '-');
        assert(length(parts) == 3 && strcmp(parts{2}, 'b'));

        recovered = 0;
        try
            assert(1 == 2, 'bad math');
        catch e
            recovered = 1;
            assert(isfield(e, 'message'));
            assert(isfield(e, 'identifier'));
        end
        assert(recovered == 1);
        "#,
    );
}

#[test]
fn conformance_solvers_and_subplots() {
    let mut vm = Vm::new();
    vm.eval(
        r#"
        % Harmonic oscillator y'' = -y as 2D system: [y1; y2]' = [y2; -y1]
        % Initial condition [1; 0] => y1(t) = cos(t), y2(t) = -sin(t)
        [t, y] = ode45(@(t, u) [u(2); -u(1)], [0, pi], [1; 0]);
        assert(abs(y(end, 1) - (-1)) < 1e-3);
        assert(abs(y(end, 2) - 0) < 1e-3);

        % Root finding and quadrature
        root = fzero(@(x) x.^3 - 2*x - 5, 2);
        assert(abs(root.^3 - 2*root - 5) < 1e-9);
        area = integral(@(x) exp(-x.^2), 0, 1);
        assert(abs(area - 0.7468241328) < 1e-6);

        % Subplots + imagesc + legend + xlim/ylim
        subplot(2, 1, 1);
        plot(t, y(:, 1), 'b');
        hold('on');
        plot(t, y(:, 2), 'r--');
        legend('cos(t)', '-sin(t)');
        xlim([0, pi]);
        ylim([-1.2, 1.2]);
        title('Harmonic Oscillator');

        subplot(2, 1, 2);
        [X, Y] = meshgrid(linspace(-2, 2, 12));
        Z = exp(-(X.^2 + Y.^2));
        imagesc(Z);
        title('2D Gaussian');
        "#,
    )
    .unwrap();

    let svg = vm.figure().to_svg(640.0, 300.0);
    assert!(svg.contains("Harmonic Oscillator"));
    assert!(svg.contains("2D Gaussian"));
    assert!(svg.contains("cos(t)"));
}

#[test]
fn conformance_multifile_vfs_modules() {
    let mut vm = Vm::new();
    vm.mount_file(
        "utils/norm2.m",
        "function n = norm2(x, y)\n  n = sqrt(x.^2 + y.^2);\nend\n",
    )
    .unwrap();
    vm.mount_file("setup.m", "alpha = 42;\nbeta = norm2(6, 8);\n")
        .unwrap();
    vm.eval(
        r#"
        addpath('utils');
        assert(exist('norm2') == 2);
        run('setup.m');
        assert(alpha == 42);
        assert(abs(beta - 10) < 1e-12);
        "#,
    )
    .unwrap();
}

#[test]
fn conformance_sparse_matfile_classdef_and_profiler() {
    let mut vm = Vm::new();
    vm.mount_file(
        "Counter.m",
        r#"
        classdef Counter
            properties
                val = 0
            end
            methods
                function obj = Counter(start)
                    if nargin >= 1
                        obj.val = start;
                    end
                end
                function obj = inc(obj, by)
                    obj.val = obj.val + by;
                end
            end
        end
        "#,
    )
    .unwrap();

    let out = vm
        .eval(
            r#"
            % Auto-loaded classdef from Counter.m
            c = Counter(10);
            c = c.inc(5);
            assert(isa(c, 'Counter') && c.val == 15);

            % Sparse 60x60 1D Poisson system solved via CG
            n = 60;
            S = sparse([1:n, 1:(n-1), 2:n], [1:n, 2:n, 1:(n-1)], [2*ones(1, n), -ones(1, n-1), -ones(1, n-1)], n, n);
            b = ones(n, 1);
            x = pcg(S, b, 1e-10, 500);
            assert(norm(S * x - b) < 1e-8);

            % .mat v5 binary save & load roundtrip
            save('state.mat', 'x', 'n');
            s = load('state.mat');
            assert(s.n == 60 && norm(s.x - x) < 1e-12);

            % Profiler + fast local-slot recursion
            profile('on');
            function r = fib(k)
                if k <= 1
                    r = k;
                else
                    r = fib(k - 1) + fib(k - 2);
                end
            end
            assert(fib(10) == 55);
            profile('report');
            "#,
        )
        .unwrap();
    assert!(out.stdout.contains("fib"));
}

#[test]
fn conformance_nd_arrays_deletion_command_syntax_and_scopes() {
    let mut vm = Vm::new();
    vm.eval(
        r#"
        % N-D arrays (3D)
        Z = zeros(2, 3, 4);
        assert(ndims(Z) == 3 && size(Z, 3) == 4 && numel(Z) == 24);
        Z(2, 3, 4) = 77;
        assert(Z(2, 3, 4) == 77);

        A = [1 2; 3 4];
        B = [5 6; 7 8];
        C = cat(3, A, B);
        assert(ndims(C) == 3 && C(1, 2, 1) == 2 && C(1, 2, 2) == 6);
        P = permute(C, [2, 1, 3]);
        assert(P(2, 1, 2) == 6);
        S = squeeze(zeros(2, 1, 3));
        assert(ndims(S) == 2 && size(S, 1) == 2 && size(S, 2) == 3);

        % Empty-bracket deletion `[]`
        v = [-2, 1, -5, 4, 3];
        v(v < 0) = [];
        assert(length(v) == 3 && sum(v) == 8);
        M = [1 2 3; 4 5 6];
        M(:, 2) = [];
        assert(size(M, 1) == 2 && size(M, 2) == 2 && M(1, 2) == 3);
        M(1, :) = [];
        assert(size(M, 1) == 1 && M(1, 1) == 4 && M(1, 2) == 6);

        % Command syntax (`save`, `load`, `clear`, `hold on`, `grid on`)
        keep_val = 123;
        save cmd_test.mat keep_val
        clear keep_val
        assert(exist('keep_val') == 0);
        load cmd_test.mat
        assert(keep_val == 123);

        % global, persistent, and varargout
        global SHARED_CNT
        SHARED_CNT = 10;

        function varargout = bump_counters(step)
            global SHARED_CNT
            persistent LOCAL_CNT
            if isempty(LOCAL_CNT)
                LOCAL_CNT = 0;
            end
            SHARED_CNT = SHARED_CNT + step;
            LOCAL_CNT = LOCAL_CNT + 1;
            varargout{1} = SHARED_CNT;
            varargout{2} = LOCAL_CNT;
        end

        [g1, p1] = bump_counters(5);
        [g2, p2] = bump_counters(7);
        assert(g1 == 15 && p1 == 1 && g2 == 22 && p2 == 2 && SHARED_CNT == 22);
        "#,
    )
    .unwrap();
}

#[test]
fn conformance_handle_classes_inheritance_structs_and_stdlib() {
    let mut vm = Vm::new();
    vm.eval(
        r#"
        % Handle class + inheritance
        classdef BaseTracker < handle
            properties
                count = 0
            end
            methods
                function tick(obj)
                    obj.count = obj.count + 1;
                end
            end
        end

        classdef OptTracker < BaseTracker
            properties
                label = 'nelder-mead'
            end
            methods
                function obj = OptTracker(lbl)
                    obj.label = lbl;
                end
            end
        end

        tr = OptTracker('rosenbrock');
        assert(isa(tr, 'OptTracker') && isa(tr, 'BaseTracker') && isa(tr, 'handle'));
        tr_alias = tr;
        tr.tick();
        assert(tr_alias.count == 1, 'Handle class must share mutable state');

        % Nelder-Mead fminsearch on Rosenbrock banana function f(x,y)=(1-x)^2 + 100*(y-x^2)^2
        function val = rosen(p, tracker)
            tracker.tick();
            val = (1 - p(1))^2 + 100 * (p(2) - p(1)^2)^2;
        end
        [xopt, fval] = fminsearch(@(p) rosen(p, tr), [-1.2, 1.0]);
        assert(norm(xopt - [1, 1]) < 1e-3 && fval < 1e-6);
        assert(tr.count > 20);

        % Struct arrays
        s(1).name = 'alpha';
        s(1).val = 10;
        s(2).name = 'beta';
        s(2).val = 20;
        assert(length(s) == 2 && strcmp(s(2).name, 'beta'));
        assert(sum(s.val) == 30);

        % Roots, poly, fft2/ifft2, expm, sets, JSON, strings
        r = roots([1, 0, 1]); % x^2 + 1 = 0 => +/- i
        assert(abs(abs(r(1)) - 1) < 1e-10 && abs(imag(r(1))) > 0.99);
        p_rec = poly([1, 2, 3]);
        assert(norm(p_rec - [1, -6, 11, -6]) < 1e-10);

        F = [1 2 3 4; 5 6 7 8; 2 1 0 3; 4 3 2 1];
        assert(norm(ifft2(fft2(F)) - F) < 1e-10);
        E = expm([0 1; 0 0]); % [1 1; 0 1]
        assert(norm(E - [1 1; 0 1]) < 1e-12);

        assert(norm(unique([3 1 2 1 3]) - [1 2 3]) == 0);
        assert(all(ismember([2 5], [1 2 3]) == [1 0]));
        assert(norm(setdiff([1 2 3 4], [2 4]) - [1 3]) == 0);
        assert(norm(intersect([1 2 3], [2 3 4]) - [2 3]) == 0);
        assert(norm(union([1 2], [2 3]) - [1 2 3]) == 0);
        assert(norm(fliplr([1 2 3]) - [3 2 1]) == 0);
        assert(size(repmat([1 2], 3, 2), 1) == 3 && size(repmat([1 2], 3, 2), 2) == 4);

        obj.a = 1;
        obj.b = 'ok';
        obj.c = [10 20];
        j = jsonencode(obj);
        dec = jsondecode(j);
        assert(dec.a == 1 && strcmp(dec.b, 'ok') && sum(dec.c) == 30);
        assert(contains('hello matro', 'matro') && strcmp(strrep('a-b', '-', '_'), 'a_b'));

        % New plots: contour, stem, histogram
        subplot(2, 1, 1);
        [X, Y] = meshgrid(linspace(-2, 2, 15));
        contour(X, Y, X.^2 + Y.^2, 6);
        title('Contours');
        subplot(2, 1, 2);
        stem(1:5, [1 3 2 5 4]);
        hold on
        grid on
        title('Stem');
        "#,
    )
    .unwrap();
    let svg = vm.figure().to_svg(600.0, 300.0);
    assert!(svg.contains("Contours") && svg.contains("Stem"));
}

#[test]
fn conformance_complex_linalg_ode15s_filter_interp2_regexp() {
    let mut vm = Vm::new();
    vm.eval(
        r#"
        % 1. Complex linear algebra: lu, qr, chol, svd, eig
        A = [2+1i, 1-2i, 0.5i; 1+1i, 3-0.5i, 2; -1i, 0.5+1i, 4-1i];
        [L, U, P] = lu(A);
        assert(norm(P * A - L * U) < 1e-10, 'Complex LU mismatch');

        [Q, R] = qr(A);
        assert(norm(Q' * Q - eye(3)) < 1e-10, 'Complex Q not unitary');
        assert(norm(Q * R - A) < 1e-10, 'Complex QR mismatch');

        H = A' * A + 2 * eye(3); % Hermitian positive definite
        Rc = chol(H);
        assert(norm(Rc' * Rc - H) < 1e-10, 'Complex Cholesky mismatch');

        [Us, Ss, Vs] = svd(A);
        assert(norm(Us * Ss * Vs' - A) < 1e-9, 'Complex SVD mismatch');
        assert(norm(Us' * Us - eye(3)) < 1e-9 && norm(Vs' * Vs - eye(3)) < 1e-9);

        [V, D] = eig(A);
        assert(norm(A * V - V * D) < 1e-8, 'Complex eig mismatch');

        % 2. Stiff ODE solver: ode15s
        % Scalar stiff test: y' = -1000*(y - cos(t)), y(0) = 0
        % Exact steady-state solution for t >> 0.01: (1e6*cos(t) + 1000*sin(t))/(1e6 + 1)
        [ts, ys] = ode15s(@(t, y) -1000 * (y - cos(t)), [0, 1], 0);
        y_exact = (1e6 * cos(1) + 1000 * sin(1)) / (1e6 + 1);
        assert(abs(ys(end) - y_exact) < 5e-3, 'ode15s scalar stiff accuracy');
        assert(length(ts) < 250, 'ode15s should take far fewer steps than explicit RK');

        % Vector stiff test: Van der Pol oscillator with mu = 50
        mu = 50;
        [tv, yv] = ode15s(@(t, y) [y(2); mu * (1 - y(1)^2) * y(2) - y(1)], [0, 5], [2; 0]);
        assert(abs(tv(end) - 5) < 1e-9 && size(yv, 2) == 2 && all(isfinite(yv(:))));

        % Interpolated output on multi-point tspan
        [ti, yi] = ode15s(@(t, y) -10 * y, [0, 0.2, 0.5, 1.0], 1);
        assert(length(ti) == 4 && abs(yi(4) - exp(-10)) < 1e-3);

        % 3. Digital filter (FIR + IIR)
        x = [1, 2, 3, 4, 5];
        y_fir = filter([0.5, 0.5], 1, x);
        assert(norm(y_fir - [0.5, 1.5, 2.5, 3.5, 4.5]) < 1e-12);
        % First-order IIR: y[n] = x[n] + 0.5*y[n-1] for unit impulse
        y_iir = filter(1, [1, -0.5], [1, 0, 0, 0]);
        assert(norm(y_iir - [1, 0.5, 0.25, 0.125]) < 1e-12);

        % 4. 2D bilinear interpolation: interp2
        [X, Y] = meshgrid(1:4, 1:3);
        Z = 2 * X + 3 * Y; % exact bilinear surface
        zq = interp2(X, Y, Z, [1.5, 3.25], [2.5, 1.75]);
        assert(abs(zq(1) - (2 * 1.5 + 3 * 2.5)) < 1e-12);
        assert(abs(zq(2) - (2 * 3.25 + 3 * 1.75)) < 1e-12);
        assert(isnan(interp2(X, Y, Z, 5.0, 2.0)));

        % 5. Pure-Rust regular expressions: regexp, regexpi, regexprep
        txt = 'alpha=12, Beta=345, gamma=6';
        starts = regexp(txt, '\d+');
        assert(norm(starts - [7, 16, 27]) == 0);
        [st, en, matches] = regexp(txt, '[a-z]+=\d+', 'start', 'end', 'match');
        assert(length(st) == 3 && st(1) == 1 && en(1) == 8);
        assert(strcmp(matches{1}, 'alpha=12') && strcmp(matches{2}, 'eta=345') && strcmp(matches{3}, 'gamma=6'));

        im = regexpi(txt, 'beta=\d+', 'match');
        assert(length(im) == 1 && strcmp(im{1}, 'Beta=345'));

        cleaned = regexprep('foo_123 bar_456', '\d+', 'NUM');
        assert(strcmp(cleaned, 'foo_NUM bar_NUM'));
        "#,
    )
    .unwrap();
}

#[test]
fn conformance_comma_separated_lists_fileio_and_compressed_mat() {
    let mut vm = Vm::new();

    // Build a genuine miCOMPRESSED .mat v5 file using zlib-wrapped miMATRIX element
    let w_val = matro_types::Value::matrix(
        matro_types::Matrix::from_col_major(2, 2, vec![10.0, 30.0, 20.0, 40.0]).unwrap(),
    );
    let tag_val = matro_types::Value::String("compressed-ok".to_string());
    let plain = matro_loader::matfile::write_mat_v5(&[("W", &w_val), ("tag", &tag_val)]).unwrap();
    let mut compressed_mat = plain[..128].to_vec();
    let mut pos = 128usize;
    while pos + 8 <= plain.len() {
        let len = u32::from_le_bytes(plain[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let padded = (len + 7) & !7;
        let elem = &plain[pos..pos + 8 + padded];
        let zbytes = matro_loader::inflate::zlib_wrap_stored(elem);
        compressed_mat.extend_from_slice(&15u32.to_le_bytes()); // miCOMPRESSED
        compressed_mat.extend_from_slice(&(zbytes.len() as u32).to_le_bytes());
        compressed_mat.extend_from_slice(&zbytes);
        let zpad = (8 - (zbytes.len() % 8)) % 8;
        compressed_mat.resize(compressed_mat.len() + zpad, 0);
        pos += 8 + padded;
    }
    vm.mount_bytes("compressed.mat", &compressed_mat).unwrap();

    vm.eval(
        r#"
        % 1. Compressed Level 5 .mat loading
        S = load('compressed.mat');
        assert(norm(S.W - [10 20; 30 40]) < 1e-12);
        assert(strcmp(S.tag, 'compressed-ok'));

        % 2. Comma-separated lists: c{:}
        c = {10, 20, 30};
        v = [c{:}];
        assert(norm(v - [10, 20, 30]) == 0);

        [a, b, d] = c{:};
        assert(a == 10 && b == 20 && d == 30);

        function s = add3(x, y, z)
            s = x + 2 * y + 3 * z;
        end
        assert(add3(c{:}) == 10 + 40 + 90);

        fmt_args = {'item', 42};
        assert(strcmp(sprintf('%s=%d', fmt_args{:}), 'item=42'));

        sc = {'hello', ' ', 'matro'};
        assert(strcmp([sc{:}], 'hello matro'));

        c2 = {c{:}, 40};
        assert(length(c2) == 4 && c2{4} == 40);

        % 3. VFS-backed File I/O: text write/read + textscan
        fid = fopen('report.csv', 'w');
        assert(fid >= 3);
        fprintf(fid, 'name,val,count\n');
        fprintf(fid, 'alpha,1.5,10\n');
        fprintf(fid, 'beta,2.75,20\n');
        assert(fclose(fid) == 0);

        fid = fopen('report.csv', 'r');
        hdr = fgetl(fid);
        assert(strcmp(hdr, 'name,val,count'));
        line1 = fgets(fid);
        assert(strcmp(line1, sprintf('alpha,1.5,10\n')));
        assert(~feof(fid));
        line2 = fgetl(fid);
        assert(strcmp(line2, 'beta,2.75,20'));
        assert(feof(fid));
        assert(fgetl(fid) == -1);

        frewind(fid);
        assert(ftell(fid) == 0);
        cols = textscan(fid, '%s %f %d', 'Delimiter', ',', 'HeaderLines', 1);
        fclose(fid);
        assert(strcmp(cols{1}{1}, 'alpha') && strcmp(cols{1}{2}, 'beta'));
        assert(abs(cols{2}(1) - 1.5) < 1e-12 && abs(cols{2}(2) - 2.75) < 1e-12);
        assert(cols{3}(1) == 10 && cols{3}(2) == 20);

        % Missing file returns -1
        assert(fopen('does_not_exist.txt', 'r') == -1);

        % 4. Binary File I/O: fwrite / fread
        bf = fopen('vals.bin', 'wb');
        nw = fwrite(bf, [1.25, -3.5, 7.0], 'double');
        assert(nw == 3);
        fclose(bf);

        bf = fopen('vals.bin', 'rb');
        [raw_d, nr] = fread(bf, Inf, 'double');
        fclose(bf);
        assert(nr == 3 && norm(raw_d' - [1.25, -3.5, 7.0]) < 1e-12);
        "#,
    )
    .unwrap();
}
