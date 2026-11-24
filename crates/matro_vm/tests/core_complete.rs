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
