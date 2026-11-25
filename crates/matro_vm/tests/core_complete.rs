use matro_vm::Vm;

#[test]
fn core_typed_integers_single_char_and_bitwise() {
    let mut vm = Vm::new();
    vm.eval(
        r#"
        % Saturating int8 / uint8 arithmetic
        a = uint8(200);
        b = uint8(100);
        assert(strcmp(class(a), 'uint8') && isa(a, 'uint8') && isa(a, 'integer') && isa(a, 'numeric'));
        assert(isinteger(a) && ~isfloat(a) && isnumeric(a));
        c = a + b;
        assert(strcmp(class(c), 'uint8') && double(c) == 255);
        d = uint8(10) - uint8(50);
        assert(double(d) == 0);

        s1 = int8(100) + int8(100);
        s2 = int8(-100) - int8(100);
        assert(strcmp(class(s1), 'int8') && double(s1) == 127 && double(s2) == -128);

        % Single precision & type promotion
        f = single(3.5);
        assert(strcmp(class(f), 'single') && isfloat(f) && ~isinteger(f) && isa(f, 'float'));
        f2 = f + 1.5;
        assert(strcmp(class(f2), 'single') && abs(double(f2) - 5.0) < 1e-6);

        % intmin / intmax / cast / char
        assert(double(intmin('int8')) == -128 && double(intmax('int8')) == 127);
        assert(double(intmin('uint16')) == 0 && double(intmax('uint16')) == 65535);
        assert(strcmp(class(cast(42.7, 'int16')), 'int16') && double(cast(42.7, 'int16')) == 43);
        assert(strcmp(char([77, 97, 116, 114, 111]), 'Matro'));

        % Bitwise ops
        assert(bitand(uint8(12), uint8(10)) == 8);
        assert(bitor(uint8(12), uint8(10)) == 14);
        assert(bitxor(uint8(12), uint8(10)) == 6);
        assert(bitshift(uint8(1), 4) == 16);
        assert(bitshift(uint8(16), -2) == 4);
        assert(bitget(uint8(13), 1) == 1 && bitget(uint8(13), 2) == 0 && bitget(uint8(13), 3) == 1);
        assert(bitset(uint8(8), 1, 1) == 9);
        "#,
    )
    .unwrap();
}

#[test]
fn core_modern_types_datetime_duration_categorical_timetable_string() {
    let mut vm = Vm::new();
    vm.eval(
        r#"
        % datetime & duration arithmetic + comparisons
        t1 = datetime(2026, 11, 24, 10, 30, 0);
        t2 = t1 + hours(2) + minutes(15);
        assert(isdatetime(t1) && isa(t1, 'datetime'));
        assert(year(t1) == 2026 && month(t1) == 11 && day(t1) == 24);
        assert(hour(t2) == 12 && minute(t2) == 45 && second(t2) == 0);
        dt = t2 - t1;
        assert(isduration(dt) && abs(seconds(dt) - (2*3600 + 15*60)) < 1e-6);
        assert(t2 > t1 && t1 < t2 && t1 == datetime('2026-11-24 10:30:00'));

        % Vectorized duration scaling
        d = seconds([10, 20, 30]) * 2;
        assert(all(seconds(d) == [20, 40, 60]));

        % categorical
        cats = categorical({'red', 'blue', 'red', 'green', 'blue', 'red'});
        assert(iscategorical(cats));
        cnames = categories(cats);
        assert(length(cnames) == 3);
        cnts = countcats(cats);
        assert(sum(cnts) == 6);
        mask_red = (cats == 'red');
        assert(sum(mask_red) == 3);

        % timetable
        times = datetime(2026, 1, [1; 2; 3]);
        TT = timetable(times, [10; 20; 30], 'VariableNames', {'Temp'});
        assert(height(TT) == 3 && width(TT) == 2);
        assert(TT.Temp(2) == 20);

        % string arrays & vectorized string ops
        S = string({'hello', 'world', 'matro'});
        assert(isstring(S));
        assert(all(strlength(S) == [5, 5, 5]));
        S2 = S + '!';
        assert(all(endsWith(S2, '!')));
        assert(strcmp(S2{3}, 'matro!'));
        assert(all(S == string({'hello', 'world', 'matro'})));
        assert(strcmp(char(join(S, '-')), 'hello-world-matro'));
        parts = split(string('a,b,c'), ',');
        assert(strcmp(parts{2}, 'b'));
        assert(strcmp(char(strip(string('  hi  '))), 'hi'));
        assert(strcmp(char(erase(string('foobar'), 'bar')), 'foo'));
        "#,
    )
    .unwrap();
}

#[test]
fn core_oop_multiple_inheritance_static_dependent_enum_events_overloads() {
    let mut vm = Vm::new();
    vm.eval(
        r#"
        % Mixin + Multiple inheritance + Static + Constant + Dependent get/set + Operator overloading + subsref
        classdef Named
            properties
                Tag = 'default'
            end
            methods
                function s = getTag(obj)
                    s = obj.Tag;
                end
            end
        end

        classdef Vec2 < Named
            properties
                x = 0
                y = 0
            end
            properties (Constant)
                DIM = 2
            end
            properties (Dependent)
                Norm
            end
            methods (Static)
                function v = unitX()
                    v = Vec2(1, 0);
                end
                function s = addScalars(a, b)
                    s = a + b;
                end
            end
            methods
                function obj = Vec2(x, y)
                    if nargin >= 1, obj.x = x; end
                    if nargin >= 2, obj.y = y; end
                end
                function n = get.Norm(obj)
                    n = sqrt(obj.x^2 + obj.y^2);
                end
                function obj = set.x(obj, val)
                    obj.x = val;
                end
                function r = plus(a, b)
                    r = Vec2(a.x + b.x, a.y + b.y);
                end
                function r = minus(a, b)
                    r = Vec2(a.x - b.x, a.y - b.y);
                end
                function r = uminus(a)
                    r = Vec2(-a.x, -a.y);
                end
                function r = mtimes(a, k)
                    r = Vec2(a.x * k, a.y * k);
                end
                function v = subsref(obj, S)
                    idx = S.subs{1};
                    if idx == 1
                        v = obj.x;
                    else
                        v = obj.y;
                    end
                end
            end
        end

        assert(Vec2.DIM == 2);
        assert(Vec2.addScalars(7, 8) == 15);
        u = Vec2.unitX();
        v = Vec2(3, 4);
        assert(isa(v, 'Vec2') && isa(v, 'Named'));
        assert(strcmp(v.getTag(), 'default'));
        assert(abs(v.Norm - 5.0) < 1e-12);
        w = u + v;
        assert(w.x == 4 && w.y == 4);
        w2 = v * 3;
        assert(w2.x == 9 && w2.y == 12);
        neg_v = -v;
        assert(neg_v.x == -3 && neg_v.y == -4);
        assert(v(1) == 3 && v(2) == 4);
        v.x = 6;
        assert(v.x == 6);

        % Enumeration with constructor arguments
        classdef Compass
            properties
                Angle = 0
            end
            methods
                function obj = Compass(deg)
                    if nargin >= 1, obj.Angle = deg; end
                end
            end
            enumeration
                North (0)
                East  (90)
                South (180)
                West  (270)
            end
        end

        d1 = Compass.East;
        d2 = Compass.West;
        assert(d1.Angle == 90 && d2.Angle == 270);
        assert(d1 == Compass.East && d1 ~= d2);

        % Handle class + Events + addlistener + notify + multiple inheritance
        classdef Sensor < handle & Named
            properties
                Value = 0
            end
            events
                ValueChanged
            end
            methods
                function setValue(obj, v)
                    obj.Value = v;
                    notify(obj, 'ValueChanged');
                end
            end
        end

        classdef LogSink < handle
            properties
                Count = 0
                LastVal = 0
            end
            methods
                function onEvent(obj, src, ev)
                    obj.Count = obj.Count + 1;
                    obj.LastVal = src.Value;
                end
            end
        end

        sen = Sensor();
        sink = LogSink();
        assert(isa(sen, 'Sensor') && isa(sen, 'handle') && isa(sen, 'Named'));
        addlistener(sen, 'ValueChanged', @(src, ev) sink.onEvent(src, ev));
        sen.setValue(42);
        sen.setValue(99);
        assert(sink.Count == 2 && sink.LastVal == 99);
        "#,
    )
    .unwrap();
}

#[test]
fn core_arguments_validation_and_key_value_syntax() {
    let mut vm = Vm::new();
    vm.eval(
        r#"
        function y = scaleShift(x, scale, opts)
            arguments
                x (1,:) double {mustBeNumeric, mustBeFinite}
                scale (1,1) double {mustBePositive} = 2
                opts.Offset (1,1) double = 0
                opts.Mode char {mustBeMember(opts.Mode, {'linear', 'square'})} = 'linear'
            end
            if strcmp(opts.Mode, 'square')
                y = scale * (x.^2) + opts.Offset;
            else
                y = scale * x + opts.Offset;
            end
        end

        % Default optional & name-value
        r1 = scaleShift([1, 2, 3]);
        assert(all(r1 == [2, 4, 6]));

        % Key=Val R2021a+ syntax!
        r2 = scaleShift([1, 2, 3], 3, Offset=10);
        assert(all(r2 == [13, 16, 19]));

        % Omitting optional positional and jumping straight to Key=Val!
        r3 = scaleShift([2, 3], Mode='square', Offset=1);
        assert(all(r3 == [9, 19]));

        % Validation failure caught via try/catch
        caught_neg = false;
        try
            scaleShift([1, 2], -5);
        catch
            caught_neg = true;
        end
        assert(caught_neg);

        caught_member = false;
        try
            scaleShift([1, 2], 1, Mode='invalid');
        catch
            caught_member = true;
        end
        assert(caught_member);

        % Repeating arguments
        function total = sumPairs(xs, ys)
            arguments (Repeating)
                xs double
                ys double
            end
            total = 0;
            for k = 1:length(xs)
                total = total + xs{k} * ys{k};
            end
        end
        assert(sumPairs(2, 3, 4, 5, 6, 7) == (6 + 20 + 42));
        "#,
    )
    .unwrap();
}

#[test]
fn stdlib_interpolation_suite_and_gridded_interpolant() {
    let mut vm = Vm::new();
    vm.eval(
        r#"
        % Not-a-knot cubic spline is EXACT on any cubic polynomial!
        x = [0, 1, 2.5, 4, 6];
        y = x.^3 - 2*x.^2 + 3*x - 1;
        xq = [0.5, 1.7, 3.2, 5.1];
        ys = spline(x, y, xq);
        y_true = xq.^3 - 2*xq.^2 + 3*xq - 1;
        assert(max(abs(ys - y_true)) < 1e-10);

        % pp struct + ppval + mkpp
        pp = spline(x, y);
        assert(strcmp(pp.form, 'pp') && pp.pieces == 4 && pp.order == 4);
        assert(max(abs(ppval(pp, xq) - y_true)) < 1e-10);

        % PCHIP preserves monotonicity with zero overshoot on step data
        xs = [0, 1, 2, 3, 4, 5];
        ystep = [0, 0, 0, 1, 1, 1];
        xq2 = linspace(0, 5, 51);
        yp = pchip(xs, ystep, xq2);
        assert(min(yp) >= -1e-12 && max(yp) <= 1.0 + 1e-12);
        assert(all(diff(yp) >= -1e-12));

        % interp1 methods: linear, nearest, previous, next, pchip, spline
        assert(abs(interp1([0, 1, 2], [0, 10, 20], 0.3, 'nearest') - 0) < 1e-12);
        assert(abs(interp1([0, 1, 2], [0, 10, 20], 0.7, 'nearest') - 10) < 1e-12);
        assert(abs(interp1([0, 1, 2], [0, 10, 20], 0.7, 'previous') - 0) < 1e-12);
        assert(abs(interp1([0, 1, 2], [0, 10, 20], 0.3, 'next') - 10) < 1e-12);
        assert(max(abs(interp1(x, y, xq, 'spline') - y_true)) < 1e-10);
        assert(abs(interp1([0, 1], [0, 2], 3, 'linear', 'extrap') - 6) < 1e-12);

        % griddedInterpolant 1D & 2D
        F1 = griddedInterpolant(x, y, 'spline');
        assert(max(abs(F1(xq) - y_true)) < 1e-10);

        [X, Y] = ndgrid(0:4, 0:3);
        V = 2*X + 3*Y + 1;
        F2 = griddedInterpolant(X, Y, V, 'linear');
        assert(abs(F2(1.5, 2.25) - (2*1.5 + 3*2.25 + 1)) < 1e-12);
        "#,
    )
    .unwrap();
}

#[test]
fn stdlib_ode23_ode113_bvp4c_pdepe_eigs_svds() {
    let mut vm = Vm::new();
    vm.eval(
        r#"
        % ode23 & ode113 on dy/dt = -2*y, y(0) = 1 -> y(1) = exp(-2)
        [t23, y23] = ode23(@(t, y) -2*y, [0, 1], 1);
        assert(abs(y23(end) - exp(-2)) < 1e-3);

        [t113, y113] = ode113(@(t, y) -2*y, [0, 1], 1);
        assert(abs(y113(end) - exp(-2)) < 1e-5);

        % bvp4c + bvpinit + deval on y'' + y = 0, y(0) = 0, y(pi/2) = 1 -> y(x) = sin(x)
        odefun = @(x, y) [y(2); -y(1)];
        bcfun = @(ya, yb) [ya(1); yb(1) - 1];
        solinit = bvpinit(linspace(0, pi/2, 11), [0.5; 0.5]);
        sol = bvp4c(odefun, bcfun, solinit);
        assert(max(abs(sol.y(1, :) - sin(sol.x))) < 1e-5);
        val_mid = deval(sol, pi/4);
        assert(abs(val_mid(1) - sin(pi/4)) < 1e-5);

        % pdepe on 1D heat equation u_t = u_xx, u(x,0) = sin(pi*x), u(0,t)=u(1,t)=0
        % Exact solution: u(x, t) = exp(-pi^2 * t) * sin(pi * x)
        function [c, f, s] = heatpde(x, t, u, dudx)
            c = 1;
            f = dudx;
            s = 0;
        end
        function u0 = heatic(x)
            u0 = sin(pi * x);
        end
        function [pl, ql, pr, qr] = heatbc(xl, ul, xr, ur, t)
            pl = ul; ql = 0;
            pr = ur; qr = 0;
        end
        xm = linspace(0, 1, 15);
        tm = linspace(0, 0.1, 6);
        u_sol = pdepe(0, @heatpde, @heatic, @heatbc, xm, tm);
        exact_end = exp(-pi^2 * 0.1) * sin(pi * xm);
        assert(max(abs(u_sol(end, :) - exact_end)) < 0.03);

        % eigs on sparse 60x60 1D Laplacian (largest and smallest eigenvalues)
        n = 60;
        L = sparse([1:n, 1:(n-1), 2:n], [1:n, 2:n, 1:(n-1)], [2*ones(1,n), -ones(1,n-1), -ones(1,n-1)], n, n);
        [Vmax, Dmax] = eigs(L, 2, 'largestabs');
        lam_max_exact = 2 - 2*cos(n*pi/(n+1));
        assert(abs(Dmax(1,1) - lam_max_exact) < 1e-3);
        assert(norm(L * Vmax(:,1) - Dmax(1,1) * Vmax(:,1)) < 1e-2);

        d_min = eigs(L, 2, 'smallestabs');
        lam_min_exact = 2 - 2*cos(pi/(n+1));
        assert(abs(d_min(1) - lam_min_exact) < 1e-4);

        % svds on sparse 60x40 low-rank-ish matrix
        A = sparse(1:40, 1:40, 40:-1:1, 60, 40);
        [U, S, V] = svds(A, 3);
        assert(abs(S(1,1) - 40) < 1e-4 && abs(S(2,2) - 39) < 1e-4 && abs(S(3,3) - 38) < 1e-4);
        "#,
    )
    .unwrap();
}

#[test]
fn stdlib_handle_graphics_fs_utils_and_media_io() {
    let mut vm = Vm::new();
    vm.eval(
        r#"
        % Handle graphics: h = plot(...), set/get, gca/gcf, zlim, Key=Val
        x = linspace(0, 2*pi, 30);
        h = plot(x, sin(x), LineWidth=2.5, Color=[1, 0, 0]);
        assert(abs(get(h, 'LineWidth') - 2.5) < 1e-12);
        assert(strcmp(get(h, 'Color'), '#ff0000'));
        set(h, 'LineWidth', 3.5, 'LineStyle', '--', 'DisplayName', 'wave');
        assert(abs(get(h, 'LineWidth') - 3.5) < 1e-12);
        assert(strcmp(get(h, 'LineStyle'), '--'));
        assert(strcmp(get(h, 'DisplayName'), 'wave'));

        ax = gca();
        set(ax, 'XLim', [0, 6], 'Title', 'Handle Plot');
        xl = get(ax, 'XLim');
        assert(xl(1) == 0 && xl(2) == 6);
        assert(strcmp(get(ax, 'Title'), 'Handle Plot'));
        zlim([-1, 1]);
        zl = get(ax, 'ZLim');
        assert(zl(1) == -1 && zl(2) == 1);
        assert(get(gcf(), 'Number') == 1);

        % Filesystem utilities (VFS)
        p = fullfile('data', 'sub', 'sample.csv');
        assert(strcmp(p, 'data/sub/sample.csv'));
        [fp, fn, fext] = fileparts(p);
        assert(strcmp(fp, 'data/sub') && strcmp(fn, 'sample') && strcmp(fext, '.csv'));
        mkdir('scratch');
        assert(isfolder('scratch'));
        fid = fopen('scratch/hello.txt', 'w');
        fprintf(fid, 'matro');
        fclose(fid);
        assert(isfile('scratch/hello.txt'));
        copyfile('scratch/hello.txt', 'scratch/copy.txt');
        assert(isfile('scratch/copy.txt'));
        d = dir('scratch/*.txt');
        assert(length(d) == 2);
        movefile('scratch/copy.txt', 'scratch/moved.txt');
        assert(~isfile('scratch/copy.txt') && isfile('scratch/moved.txt'));
        delete('scratch/moved.txt');
        assert(~isfile('scratch/moved.txt'));

        % Pure-Rust PNG and BMP image round-trip (uint8 HxWx3)
        img = uint8(zeros(4, 5, 3));
        img(1, 1, 1) = 255;
        img(2, 3, 2) = 128;
        img(4, 5, 3) = 200;
        imwrite(img, 'test.png');
        img_png = imread('test.png');
        assert(strcmp(class(img_png), 'uint8'));
        assert(size(img_png, 1) == 4 && size(img_png, 2) == 5 && size(img_png, 3) == 3);
        assert(double(img_png(1, 1, 1)) == 255 && double(img_png(2, 3, 2)) == 128 && double(img_png(4, 5, 3)) == 200);

        imwrite(img, 'test.bmp');
        img_bmp = imread('test.bmp');
        assert(strcmp(class(img_bmp), 'uint8'));
        assert(double(img_bmp(1, 1, 1)) == 255 && double(img_bmp(2, 3, 2)) == 128 && double(img_bmp(4, 5, 3)) == 200);

        % Pure-Rust 16-bit PCM WAV stereo round-trip
        fs = 8000;
        t = (0:99)' / fs;
        sig = [0.5 * sin(2*pi*440*t), 0.25 * cos(2*pi*880*t)];
        audiowrite('tone.wav', sig, fs);
        [y_read, fs_read] = audioread('tone.wav');
        assert(fs_read == fs);
        assert(size(y_read, 1) == 100 && size(y_read, 2) == 2);
        assert(max(max(abs(y_read - sig))) < 1e-4);
        "#,
    )
    .unwrap();
    let svg = vm.figure().to_svg(640.0, 400.0);
    assert!(svg.contains("stroke-width=\"3.50\""));
    assert!(svg.contains("#ff0000"));
}

#[test]
fn core_end_keyword_universal_indexing() {
    let mut vm = Vm::new();
    vm.eval(
        r#"
        classdef EndBox
            properties
                Data = [10, 20, 30, 40, 50];
            end
            properties (Dependent)
                Scaled
            end
            methods
                function v = get.Scaled(obj)
                    v = obj.Data * 2;
                end
                function idx = pick(~, n)
                    idx = n - 1;
                end
                function sub = tail_slice(obj)
                    sub = obj.Data(floor(end/2):end);
                end
            end
        end

        function v = make_vec(n)
            v = (1:n) * 10;
        end

        function out = local_end_check(A, s, c)
            p1 = A(floor(end/2):end);
            p2 = s.vec(2:end-1);
            p3 = c{end}(2:end);
            out = sum(p1) + sum(p2) + sum(p3);
        end

        % 1. Struct field & struct array indexing with 'end'
        s.vec = [10, 20, 30, 40, 50, 60];
        s.mat = [1, 2, 3, 4; 5, 6, 7, 8; 9, 10, 11, 12];
        assert(isequal(s.vec(1:end), [10, 20, 30, 40, 50, 60]));
        assert(isequal(s.vec(2:end-1), [20, 30, 40, 50]));
        assert(isequal(s.vec(floor(end/2):end), [30, 40, 50, 60]));
        assert(isequal(s.mat(2:end, floor(end/2):end), [6, 7, 8; 10, 11, 12]));

        arr(1).v = [1, 2, 3];
        arr(2).v = [4, 5, 6, 7, 8];
        assert(isequal(arr(end).v(2:end), [5, 6, 7, 8]));

        % 2. Classdef property, dependent property, and method inside indexing
        box = EndBox();
        assert(isequal(box.Data(2:end-1), [20, 30, 40]));
        assert(isequal(box.Scaled(3:end), [60, 80, 100]));
        assert(isequal(box.tail_slice(), [20, 30, 40, 50]));
        M = [1, 2, 3, 4; 5, 6, 7, 8];
        assert(M(1, box.pick(end)) == 3);

        % 3. Table & table column indexing with 'end'
        T = table([1; 2; 3; 4], [10; 20; 30; 40], 'VariableNames', {'Id', 'Val'});
        assert(isequal(T.Val(2:end), [20; 30; 40]));
        assert(height(T(2:end, 1:end)) == 3 && width(T(2:end, 1:end)) == 2);

        % 4. DynamicCall indexing with 'end' (cell{...}(...), fn()(...), expr(...))
        c = {[1, 2, 3, 4], [10, 20, 30, 40, 50]};
        assert(isequal(c{1}(2:end), [2, 3, 4]));
        assert(isequal(c{end}(floor(end/2):end), [20, 30, 40, 50]));
        assert(isequal(make_vec(5)(2:end-1), [20, 30, 40]));
        A = [1, 2, 3, 4];
        B = [10, 20, 30, 40];
        assert(isequal((A + B)(2:end), [22, 33, 44]));
        assert(isequal(([10, 20, 30, 40])(end-1:end), [30, 40]));

        % 5. Compound CellIndex with 'end'
        s.cells = {'alpha', 'beta', 'gamma'};
        assert(strcmp(s.cells{end}, 'gamma'));
        nested = {{100, 200, 300}};
        assert(nested{1}{end} == 300);

        % 6. Nested function calls, closures, and nested arrays inside indexing
        V = [10, 20, 30, 40, 50, 60];
        assert(isequal(V(floor(end/2):end), [30, 40, 50, 60]));
        assert(M(1, min(3, end)) == 3);
        assert(isequal(M(1, max(1, end-2):end), [2, 3, 4]));
        idx_vec = [1, 2];
        assert(M(2, max(end - 1, idx_vec(end))) == 7);
        assert(M(2, min(end, idx_vec(end))) == 6);
        fn = @(k) k - 1;
        assert(V(fn(end)) == 50);
        s.fn = @(k) k - 2;
        assert(V(s.fn(end)) == 40);
        fs = {@(k) k - 1};
        assert(V(fs{1}(end)) == 50);
        assert(isequal(V([1, end]), [10, 60]));

        % 7. Char vector and string array indexing with 'end'
        str = 'hello world';
        assert(strcmp(str(7:end), 'world'));
        assert(strcmp(str(1, 2:end-1), 'ello worl'));
        assert(strcmp(s.cells{end}(2:end), 'amma'));
        sa = string({'a', 'b', 'c', 'd'});
        sa_sub = sa(2:end);
        assert(length(sa_sub) == 3 && strcmp(char(sa_sub(1)), 'b') && strcmp(char(sa_sub(end)), 'd'));

        % 8. LHS assignments with 'end' (including field-index + growing)
        V(end+1) = 70;
        assert(length(V) == 7 && V(end) == 70);
        s.vec(floor(end/2):end) = 0;
        assert(isequal(s.vec, [10, 20, 0, 0, 0, 0]));
        s.vec(end+1) = 99;
        assert(length(s.vec) == 7 && s.vec(end) == 99);

        % 9. Function local-frame 'end' resolution
        s2.vec = [1, 2, 3, 4, 5];
        assert(local_end_check([10, 20, 30, 40], s2, c) == (20+30+40) + (2+3+4) + (20+30+40+50));

        % 10. Clean try/catch unwinding if error occurs inside indexing arg
        caught = 0;
        try
            bad = V(error('boom'), end);
        catch
            caught = 1;
        end
        assert(caught == 1);
        assert(V(end) == 70);
        "#,
    )
    .unwrap();

    // 11. 'end' outside any indexing expression must still be rejected
    let compile_err = vm.eval("x = end;").unwrap_err();
    assert!(
        compile_err.to_string().contains("indexing"),
        "expected indexing error, got: {compile_err}"
    );
    let runtime_err = vm.eval("y = floor(end / 2);").unwrap_err();
    assert!(
        runtime_err.to_string().contains("indexing"),
        "expected indexing error, got: {runtime_err}"
    );
}

#[test]
fn core_multifile_function_resolution() {
    let mut vm = Vm::new();

    // 1. Two files with colliding file-local helper names + function handle to file-local helper
    vm.mount_file(
        "fileA.m",
        r#"
        function [y, fh] = fileA(x)
            y = helper(x);
            fh = @helper;
        end
        function z = helper(v)
            z = v + 1;
        end
        "#,
    )
    .unwrap();
    vm.mount_file(
        "fileB.m",
        r#"
        function y = fileB(x)
            y = helper(x);
        end
        function z = helper(v)
            z = v * 10;
        end
        "#,
    )
    .unwrap();

    // 2. Private directory scoping (`pkg_dir/private/secret_val.m`)
    vm.mount_file(
        "pkg_dir/pub_fn.m",
        r#"
        function y = pub_fn(x)
            y = x + secret_val();
        end
        "#,
    )
    .unwrap();
    vm.mount_file(
        "pkg_dir/private/secret_val.m",
        r#"
        function k = secret_val()
            k = 42;
        end
        "#,
    )
    .unwrap();
    vm.mount_file(
        "other_dir/try_priv.m",
        r#"
        function y = try_priv()
            y = secret_val();
        end
        "#,
    )
    .unwrap();

    // 3. Filename stem overrides internal header name (with recursion via internal name)
    vm.mount_file(
        "renamed_fact.m",
        r#"
        function y = internal_name(n)
            if n <= 1
                y = 1;
            else
                y = n * internal_name(n - 1);
            end
        end
        "#,
    )
    .unwrap();

    // 4. Bare script file mutating caller's workspace
    vm.mount_file(
        "setup_vars.m",
        r#"
        alpha = 11;
        beta = alpha * 3;
        "#,
    )
    .unwrap();

    // 5. Cross-file classdef inheritance + @ClassName folder with separate method file
    vm.mount_file(
        "oop_lib/BaseItem.m",
        r#"
        classdef BaseItem
            properties
                Val = 0
            end
            methods
                function obj = BaseItem(v)
                    if nargin >= 1, obj.Val = v; end
                end
                function y = baseVal(obj)
                    y = obj.Val;
                end
            end
        end
        "#,
    )
    .unwrap();
    vm.mount_file(
        "oop_lib/ChildItem.m",
        r#"
        classdef ChildItem < BaseItem
            methods
                function obj = ChildItem(v)
                    obj.Val = v * 2;
                end
            end
        end
        "#,
    )
    .unwrap();
    vm.mount_file(
        "oop_lib/@BoxVal/BoxVal.m",
        r#"
        classdef BoxVal
            properties
                N = 0
            end
            methods
                function obj = BoxVal(n)
                    if nargin >= 1, obj.N = n; end
                end
            end
        end
        "#,
    )
    .unwrap();
    vm.mount_file(
        "oop_lib/@BoxVal/addN.m",
        r#"
        function y = addN(obj, k)
            y = obj.N + k;
        end
        "#,
    )
    .unwrap();

    // 6. Path precedence + genpath / rmpath
    vm.mount_file("p1/shadow.m", "function y = shadow(); y = 1; end\n").unwrap();
    vm.mount_file("p2/shadow.m", "function y = shadow(); y = 2; end\n").unwrap();
    vm.mount_file("tree/sub/deep_fn.m", "function y = deep_fn(x); y = x + 7; end\n")
        .unwrap();
    vm.mount_file("tree/private/hidden_fn.m", "function y = hidden_fn(); y = 9; end\n")
        .unwrap();

    vm.eval(
        r#"
        % File-local helper isolation + handle capture of file-local helper
        [a1, fhA] = fileA(5);
        b1 = fileB(5);
        a2 = fileA(5); % still calls fileA's helper after fileB loaded!
        assert(a1 == 6 && b1 == 50 && a2 == 6);
        assert(fhA(9) == 10);

        % Calling private local helper from outside must fail
        local_err = 0;
        try
            helper(5);
        catch
            local_err = 1;
        end
        assert(local_err == 1);

        % Private directory visible to pkg_dir/pub_fn.m, hidden from outside & other_dir
        addpath('pkg_dir', 'other_dir');
        assert(pub_fn(8) == 50);
        priv_err = 0;
        try
            secret_val();
        catch
            priv_err = 1;
        end
        assert(priv_err == 1);
        other_priv_err = 0;
        try
            try_priv();
        catch
            other_priv_err = 1;
        end
        assert(other_priv_err == 1);

        % Filename stem beats internal function name
        assert(renamed_fact(5) == 120);

        % Bare script execution in caller workspace
        setup_vars;
        assert(alpha == 11 && beta == 33);

        % Cross-file class inheritance & @ClassName folder separate method file
        addpath('oop_lib');
        ch = ChildItem(7);
        assert(ch.baseVal() == 14);
        bv = BoxVal(20);
        assert(bv.addN(5) == 25);
        assert(addN(bv, 3) == 23);

        % addpath (prepend), addpath('-end'), rmpath, genpath, which, exist
        addpath('p1');
        assert(shadow() == 1);
        assert(strcmp(which('shadow'), 'p1/shadow.m'));
        addpath('p2'); % prepends p2 ahead of p1
        assert(shadow() == 2);
        assert(strcmp(which('shadow'), 'p2/shadow.m'));
        rmpath('p2');
        assert(shadow() == 1);
        addpath('p2', '-end');
        assert(shadow() == 1);

        gp = genpath('tree');
        assert(contains(gp, 'tree/sub'));
        assert(~contains(gp, 'private'));
        addpath(gp);
        assert(deep_fn(10) == 17);
        assert(exist('deep_fn') == 2);

        % Bottom-of-script local function (MATLAB R2016b+)
        assert(bottom_sq(6) == 36);
        function r = bottom_sq(x)
            r = x * x;
        end
        "#,
    )
    .unwrap();

    // 7. Hot-reload when a VFS .m file is modified (without vm.clear())
    vm.mount_file("dyn_fn.m", "function y = dyn_fn(); y = 100; end\n")
        .unwrap();
    vm.eval("assert(dyn_fn() == 100);").unwrap();
    vm.mount_file("dyn_fn.m", "function y = dyn_fn(); y = 200; end\n")
        .unwrap();
    vm.eval("assert(dyn_fn() == 200);").unwrap();

    // 8. eval_file: sibling & private/ resolution without addpath + 0-arg function auto-invocation
    vm.mount_file("proj/sibling.m", "function y = sibling(x); y = x + 3; end\n")
        .unwrap();
    vm.mount_file(
        "proj/private/priv_mul.m",
        "function y = priv_mul(x); y = x * 4; end\n",
    )
    .unwrap();
    let entry_src = r#"
        function out = entry_main()
            out = priv_mul(sibling(7));
            fprintf('entry_out=%d\n', out);
        end
    "#;
    let out = vm
        .eval_file(Some("proj/entry_main.m"), entry_src, true)
        .unwrap();
    assert!(out.stdout.contains("entry_out=40"));
    assert_eq!(out.last_value.unwrap().as_scalar().unwrap(), 40.0);
}

#[test]
fn core_sparse_fem_kron_reshape_and_2d_index() {
    let mut vm = Vm::new();
    vm.eval(
        r#"
        % 1. kron (dense real, dense complex, sparse)
        A = [1, 2; 3, 4];
        B = [0, 5; 6, 7];
        K_ab = kron(A, B);
        assert(isequal(K_ab, [0, 5, 0, 10; 6, 7, 12, 14; 0, 15, 0, 20; 18, 21, 24, 28]));
        K_sp = kron(sparse(eye(2)), sparse([2, 1; 0, 3]));
        assert(issparse(K_sp) && isequal(full(K_sp), [2, 1, 0, 0; 0, 3, 0, 0; 0, 0, 2, 1; 0, 0, 0, 3]));

        % 2. reshape with [] auto-inferred dim and vector size form
        M = [1, 2, 3; 4, 5, 6];
        c1 = reshape(M, [], 1);
        assert(size(c1, 1) == 6 && size(c1, 2) == 1 && isequal(c1, [1; 4; 2; 5; 3; 6]));
        r2 = reshape(1:12, 3, []);
        assert(size(r2, 1) == 3 && size(r2, 2) == 4);
        r3 = reshape(1:12, [4, 3]);
        assert(size(r3, 1) == 4 && size(r3, 2) == 3);

        % 3. 1D indexing with 2D index matrix preserves index matrix shape
        U = (10:10:60)';
        idx2d = [1, 3, 5; 2, 4, 6];
        U2d = U(idx2d);
        assert(size(U2d, 1) == 2 && size(U2d, 2) == 3);
        assert(isequal(U2d, [10, 30, 50; 20, 40, 60]));
        % Vector indexed by vector preserves source vector orientation
        row = [10, 20, 30, 40];
        assert(size(row([1, 3]), 1) == 1 && size(row([1, 3]), 2) == 2);

        % 4. Sparse indexed assignment, sum(H, dim), sparse submatrix, Banded Cholesky & Banded LU
        H = sparse(5, 5);
        for i = 1:5
            H(i, i) = 4;
            if i < 5
                H(i, i+1) = -1;
                H(i+1, i) = -1;
            end
        end
        assert(issparse(H) && H(2, 2) == 4 && H(2, 3) == -1);
        hs = sum(H, 2);
        assert(size(hs, 1) == 5 && size(hs, 2) == 1 && isequal(hs, [3; 2; 2; 2; 3]));
        Hsub = H([2, 3, 4], [2, 3, 4]);
        assert(issparse(Hsub) && size(Hsub, 1) == 3 && size(Hsub, 2) == 3);

        % Banded SPD (n=40, bw=2) solved via Banded Cholesky
        n = 40;
        Spd = sparse(n, n);
        for i = 1:n
            Spd(i, i) = 6;
            if i + 1 <= n, Spd(i, i+1) = -2; Spd(i+1, i) = -2; end
            if i + 2 <= n, Spd(i, i+2) = 0.5; Spd(i+2, i) = 0.5; end
        end
        rhs = (1:n)';
        sol = Spd \ rhs;
        assert(norm(Spd * sol - rhs) < 1e-10);

        % Non-symmetric banded (n=30, kl=1, ku=2) solved via Banded LU
        n2 = 30;
        Ns = sparse(n2, n2);
        for i = 1:n2
            Ns(i, i) = 5;
            if i + 1 <= n2, Ns(i+1, i) = -1.5; Ns(i, i+1) = 0.8; end
            if i + 2 <= n2, Ns(i, i+2) = -0.3; end
        end
        rhs2 = sin((1:n2)');
        sol2 = Ns \ rhs2;
        assert(norm(Ns * sol2 - rhs2) < 1e-10);
        "#,
    )
    .unwrap();

    // 5. Compact 2D SIMP cantilever FEM topology optimization (zero-arg function auto-run)
    let fem_src = r#"
        function res = mini_topopt()
            nelx = 12; nely = 4; volfrac = 0.4; penal = 3.0; rmin = 1.5;
            E0 = 1.0; Emin = 1e-9; nu = 0.3;
            A11 = [12 3 -6 -3; 3 12 3 0; -6 3 12 -3; -3 0 -3 12];
            A12 = [-6 -3 0 3; -3 -6 -3 -6; 0 -3 -6 3; 3 -6 3 -6];
            B11 = [-4 3 -2 9; 3 -4 -9 4; -2 -9 -4 -3; 9 4 -3 -4];
            B12 = [2 -3 4 -9; -3 2 9 -2; 4 9 2 3; -9 -2 3 2];
            KE = ([A11 A12; A12' A11] + nu*[B11 B12; B12' B11]) / (24*(1-nu^2));
            nodenrs = reshape(1:(nelx+1)*(nely+1), nely+1, nelx+1);
            edofVec = reshape(2*nodenrs(1:end-1, 1:end-1)+1, nelx*nely, 1);
            edofMat = repmat(edofVec, 1, 8) + repmat([0 1 2*nely+[2 3 0 1] -2 -1], nelx*nely, 1);
            ndof = 2*(nelx+1)*(nely+1);
            nel = nelx*nely;
            iK = reshape(kron(edofMat, ones(8, 1))', 64*nel, 1);
            jK = reshape(kron(edofMat, ones(1, 8))', 64*nel, 1);
            leftNodes = nodenrs(:, 1);
            fixedDofs = reshape([2*leftNodes-1, 2*leftNodes]', [], 1);
            freeDofs = setdiff((1:ndof)', fixedDofs);
            F = zeros(ndof, 1);
            F(2*nodenrs(end, end)) = -1;
            H = sparse(nel, nel);
            for ex = 1:nelx
                for ey = 1:nely
                    e1 = (ex-1)*nely + ey;
                    for fx = max(1, ex-floor(rmin)):min(nelx, ex+floor(rmin))
                        for fy = max(1, ey-floor(rmin)):min(nely, ey+floor(rmin))
                            e2 = (fx-1)*nely + fy;
                            w = rmin - sqrt((ex-fx)^2 + (ey-fy)^2);
                            if w > 0, H(e1, e2) = w; end
                        end
                    end
                end
            end
            Hs = sum(H, 2);
            x = volfrac * ones(nely, nelx);
            xPhys = x;
            U = zeros(ndof, 1);
            c_hist = zeros(8, 1);
            for it = 1:8
                E = Emin + xPhys(:)'.^penal * (E0 - Emin);
                sK = reshape(KE(:) * E, 64*nel, 1);
                K = sparse(iK, jK, sK, ndof, ndof);
                K = (K + K') / 2;
                U(:) = 0;
                U(freeDofs) = K(freeDofs, freeDofs) \ F(freeDofs);
                elU = U(edofMat);
                elE = sum((elU * KE) .* elU, 2);
                c_hist(it) = F' * U;
                dc = -penal*(E0-Emin)*xPhys(:).^(penal-1).*elE;
                dc = H*(x(:).*dc)./Hs./max(1e-3, x(:));
                lo = 0; hi = 1e9;
                while (hi - lo)/(hi + lo + eps) > 1e-3
                    mid = 0.5*(lo + hi);
                    cand = max(0, max(x(:)-0.2, min(1, min(x(:)+0.2, x(:).*sqrt(-dc/mid)))));
                    if mean(cand) > volfrac, lo = mid; else, hi = mid; end
                end
                x = reshape(cand, nely, nelx);
                xPhys(:) = (H * x(:)) ./ Hs;
            end
            assert(c_hist(end) < c_hist(1) * 0.7 && c_hist(end) < c_hist(2));
            assert(abs(mean(x(:)) - volfrac) < 2e-3);
            assert(abs(mean(xPhys(:)) - volfrac) < 0.05);
            res = c_hist(end);
        end
    "#;
    let out = vm.eval_file(Some("mini_topopt.m"), fem_src, true).unwrap();
    let final_c = out.last_value.unwrap().as_scalar().unwrap();
    assert!(final_c > 0.0 && final_c.is_finite());

    // 6. Full MATLAB rng() API: 0-arg query, 1-arg seed/default/shuffle/struct, 2-arg (seed, generator)
    vm.eval(
        r#"
        rng(42, 'twister');
        a1 = rand(1, 5);
        s_mid = rng();
        assert(strcmp(s_mid.Type, 'twister') && s_mid.Seed == 42);
        a2 = rand(1, 5);

        % Restoring saved state struct reproduces exact sequence
        rng(s_mid);
        a2_again = rand(1, 5);
        assert(isequal(a2, a2_again));

        % Re-seeding with 2-arg form reproduces exact sequence
        old = rng(42, 'twister');
        assert(strcmp(old.Type, 'twister'));
        assert(isequal(rand(1, 5), a1));

        % Different generator names accepted (case-insensitive & string object)
        rng(42, 'philox');
        sp = rng;
        assert(strcmp(sp.Type, 'philox') && sp.Seed == 42);
        p1 = rand(1, 4);
        rng(42, string('philox'));
        assert(isequal(rand(1, 4), p1));

        rng('shuffle', 'combRecursive');
        sc = rng();
        assert(strcmp(sc.Type, 'combRecursive'));

        rng('default');
        sd = rng();
        assert(strcmp(sd.Type, 'twister') && sd.Seed == 0);
        "#,
    )
    .unwrap();
}
