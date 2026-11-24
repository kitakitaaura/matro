% Statistics & Optimization: PCA, k-means, t-test, fsolve & linprog
rng(12);
cluster1 = [randn(15, 1) * 0.6, randn(15, 1) * 0.6];
cluster2 = [4 + randn(15, 1) * 0.6, 3 + randn(15, 1) * 0.6];
X = [cluster1; cluster2];

[idx, C] = kmeans(X, 2);
[coeff, score, latent] = pca(X);
fprintf('PCA variance explained: %.1f%%, %.1f%%\n', ...
        100 * latent(1) / sum(latent), 100 * latent(2) / sum(latent));

[h, p] = ttest(cluster1(:, 1), 0);
fprintf('One-sample t-test on cluster1 mean=0: h=%d, p=%.4f\n', h, p);

[sol, ~] = fsolve(@(v) [v(1)^2 + v(2)^2 - 5; v(1) - v(2) - 1], [1.5; 0.5]);
fprintf('fsolve root: (%.6f, %.6f)\n', sol(1), sol(2));

[xlp, flp] = linprog([-3; -5], [1, 0; 0, 2; 3, 2], [4; 12; 18], [], [], [0; 0]);
fprintf('linprog optimum: x = [%.2f, %.2f], fval = %.2f\n', xlp(1), xlp(2), flp);
