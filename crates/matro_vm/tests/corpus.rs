//! Real-world MATLAB/Octave algorithm corpus and language edge-case suite.
//! Tests Pillar 1 features (tables, containers.Map, eval/evalc, cellfun/structfun/bsxfun,
//! inputParser, 3D plotting, mutable nested functions) alongside realistic numerical algorithms.

use matro_vm::Vm;

#[test]
fn corpus_tables_maps_dynamic_and_nested_captures() {
    let mut vm = Vm::new();
    let out = vm
        .eval(
            r#"
            % 1. MATLAB tables: construction, dot access, slicing, sortrows, CSV round-trip
            names = {'Alice'; 'Bob'; 'Carol'};
            age = [31; 25; 29];
            score = [92.5; 88.0; 95.0];
            T = table(names, age, score, 'VariableNames', {'Name', 'Age', 'Score'});
            assert(istable(T) && height(T) == 3 && width(T) == 3);
            assert(sum(T.Age) == 85 && T.Score(3) == 95.0);
            T.Passed = T.Score >= 90;
            assert(width(T) == 4 && sum(T.Passed) == 2);

            T2 = sortrows(T, 'Age');
            assert(T2.Age(1) == 25 && strcmp(T2.Name{1}, 'Bob'));
            sub = T(1:2, {'Name', 'Score'});
            assert(height(sub) == 2 && width(sub) == 2);
            mat_scores = T{:, {'Age', 'Score'}};
            assert(size(mat_scores, 1) == 3 && size(mat_scores, 2) == 2 && mat_scores(1, 2) == 92.5);

            writetable(T, 'people.csv');
            Tr = readtable('people.csv');
            assert(height(Tr) == 3 && abs(Tr.Score(1) - 92.5) < 1e-12);
            summary(Tr);

            % 2. containers.Map & dictionary
            m = containers.Map({'pi', 'e'}, {3.14159, 2.71828});
            assert(m.Count == 2 && isKey(m, 'pi') && ~isKey(m, 'phi'));
            m('phi') = 1.61803;
            assert(m.Count == 3 && abs(m('phi') - 1.61803) < 1e-12);
            m = remove(m, 'e');
            assert(m.Count == 2 && length(keys(m)) == 2 && length(values(m)) == 2);

            % 3. eval & evalc
            eval('dyn_x = 6 * 7;');
            assert(dyn_x == 42);
            [txt, val] = evalc('fprintf("hi=%d", 99); 10 + 5;');
            assert(strcmp(txt, 'hi=99') && val == 15);

            % 4. cellfun, structfun, bsxfun, multi-arity @handles
            C = {[1 2], [3 4 5], [6]};
            lens = cellfun(@length, C);
            assert(norm(lens - [2, 3, 1]) == 0);
            doubled = cellfun(@(v) 2 * v, C, 'UniformOutput', false);
            assert(norm(doubled{2} - [6 8 10]) == 0);

            S.a = [1 2 3];
            S.b = [4 5];
            sums = structfun(@sum, S);
            assert(norm(sums' - [6, 9]) == 0);

            M = bsxfun(@plus, [1; 2; 3], [10, 20, 30, 40]);
            assert(size(M, 1) == 3 && size(M, 2) == 4 && M(2, 3) == 32);

            % 5. narginchk & inputParser
            function y = scaled_add(a, varargin)
                narginchk(1, 5);
                p = inputParser;
                addRequired(p, 'a');
                addOptional(p, 'b', 10);
                addParameter(p, 'Scale', 2);
                parse(p, a, varargin{:});
                y = (p.Results.a + p.Results.b) * p.Results.Scale;
            end
            assert(scaled_add(5) == 30);
            assert(scaled_add(5, 3, 'Scale', 4) == 32);
            assert(scaled_add(5, 'Scale', 10) == 150);

            % 6. Mutable outer-scope captures in nested functions
            function total = run_acc()
                acc = 0;
                function add_val(v)
                    acc = acc + v;
                end
                add_val(10);
                add_val(25);
                add_val(7);
                total = acc;
            end
            assert(run_acc() == 42);

            function inc = make_counter(start)
                count = start;
                function n = next()
                    count = count + 1;
                    n = count;
                end
                inc = @next;
            end
            c1 = make_counter(10);
            c2 = make_counter(100);
            assert(c1() == 11 && c1() == 12);
            assert(c2() == 101 && c1() == 13 && c2() == 102);
            "#,
        )
        .unwrap();
    assert!(out.stdout.contains("Variables (3 rows x 4 cols)"));
}

#[test]
fn corpus_3d_plotting_surf_mesh_plot3_quiver() {
    let mut vm = Vm::new();
    vm.eval(
        r#"
        subplot(2, 2, 1);
        [X, Y] = meshgrid(linspace(-3, 3, 18));
        R = sqrt(X.^2 + Y.^2) + eps;
        Z = sin(R) ./ R;
        surf(X, Y, Z);
        view(-35, 28);
        title('Sinc Surface');
        xlabel('x'); ylabel('y'); zlabel('z');

        subplot(2, 2, 2);
        mesh(X, Y, cos(X) .* sin(Y));
        title('Wireframe Mesh');

        subplot(2, 2, 3);
        t = linspace(0, 6*pi, 120);
        plot3(cos(t), sin(t), t / (6*pi), 'r');
        title('3D Helix');

        subplot(2, 2, 4);
        [Qx, Qy] = meshgrid(-2:0.8:2);
        quiver(Qx, Qy, -Qy, Qx);
        title('Vortex Quiver');
        "#,
    )
    .unwrap();
    let svg = vm.figure().to_svg(800.0, 600.0);
    assert!(svg.contains("Sinc Surface"));
    assert!(svg.contains("Wireframe Mesh"));
    assert!(svg.contains("3D Helix"));
    assert!(svg.contains("Vortex Quiver"));
    assert!(svg.contains("<polygon"));
    assert!(svg.contains("<polyline"));
}

#[test]
fn corpus_real_world_algorithms_and_edge_cases() {
    let mut vm = Vm::new();
    vm.eval(
        r#"
        % === Algorithm 1: 1D Constant-Velocity Kalman Filter ===
        rng(42);
        dt = 0.1;
        F = [1 dt; 0 1];
        H = [1 0];
        Q = 1e-3 * eye(2);
        R = 0.25;
        x_hat = [0; 0];
        P = eye(2);
        true_pos = 0; true_vel = 2.0;
        for k = 1:60
            true_pos = true_pos + true_vel * dt;
            z = true_pos + 0.5 * randn();
            % Predict
            x_pred = F * x_hat;
            P_pred = F * P * F' + Q;
            % Update
            y_inn = z - H * x_pred;
            S = H * P_pred * H' + R;
            K = P_pred * H' / S;
            x_hat = x_pred + K * y_inn;
            P = (eye(2) - K * H) * P_pred;
        end
        assert(abs(x_hat(1) - true_pos) < 0.6, 'Kalman position estimate drifted');
        assert(abs(x_hat(2) - true_vel) < 0.5, 'Kalman velocity estimate drifted');

        % === Algorithm 2: 2D Heat Equation (Explicit FTCS) ===
        nx = 15; ny = 15;
        U = zeros(ny, nx);
        U(8, 8) = 100;
        alpha = 0.2;
        for step = 1:30
            Un = U;
            Un(2:end-1, 2:end-1) = U(2:end-1, 2:end-1) + alpha * ( ...
                U(1:end-2, 2:end-1) + U(3:end, 2:end-1) + ... % vertical neighbors
                U(2:end-1, 1:end-2) + U(2:end-1, 3:end) - ...
                4 * U(2:end-1, 2:end-1));
            U = Un;
        end
        assert(U(8, 8) < 20 && U(8, 8) > 1 && abs(U(7, 8) - U(9, 8)) < 1e-12);

        % === Algorithm 3: 2D Image Convolution (Sobel edge filter) ===
        img = [zeros(6, 3), ones(6, 3)];
        sobel_x = [1 0 -1; 2 0 -2; 1 0 -1];
        gx_same = conv2(img, sobel_x, 'same');
        gx_valid = conv2(img, sobel_x, 'valid');
        assert(size(gx_same, 1) == 6 && size(gx_same, 2) == 6);
        assert(size(gx_valid, 1) == 4 && size(gx_valid, 2) == 4);
        assert(abs(gx_valid(2, 2)) > 3.5 && abs(gx_valid(2, 1)) < 1e-12);

        % === Algorithm 4: Modified Gram-Schmidt QR in pure MATLAB ===
        function [Q, R] = mgs_qr(A)
            [m, n] = size(A);
            Q = zeros(m, n);
            R = zeros(n, n);
            for j = 1:n
                v = A(:, j);
                for i = 1:(j-1)
                    R(i, j) = Q(:, i)' * v;
                    v = v - R(i, j) * Q(:, i);
                end
                R(j, j) = norm(v);
                Q(:, j) = v / R(j, j);
            end
        end
        M = [1 2 3; 4 5 6; 7 8 10; 2 1 0];
        [Qm, Rm] = mgs_qr(M);
        assert(norm(Qm' * Qm - eye(3)) < 1e-12);
        assert(norm(Qm * Rm - M) < 1e-12);

        % === Edge Cases: N-D implicit broadcasting, space-minus, cumprod, @func in ode45 ===
        T3 = ones(3, 4, 2);
        row4 = [1, 2, 3, 4];
        B3 = T3 + row4;
        assert(ndims(B3) == 3 && B3(2, 4, 2) == 5);

        v_space = [1 -2 3];
        v_sub = [1 - 2 3];
        assert(length(v_space) == 3 && v_space(2) == -2);
        assert(length(v_sub) == 2 && v_sub(1) == -1 && v_sub(2) == 3);

        assert(norm(cumprod([1 2 3 4]) - [1 2 6 24]) == 0);

        function dy = decay_rhs(t, y)
            dy = -2 * y;
        end
        [to, yo] = ode45(@decay_rhs, [0, 1], 1);
        assert(abs(yo(end) - exp(-2)) < 1e-4);
        "#,
    )
    .unwrap();
}
