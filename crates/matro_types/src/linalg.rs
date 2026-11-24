//! Pure-Rust numerical linear algebra decompositions for BOTH real and complex matrices
//! (LU, QR, Cholesky, SVD, Eig, FFT/FFT2, Expm).

use crate::complex::Complex;
use crate::{MatroError, Matrix, Result};

impl Matrix {
    /// LU decomposition with partial pivoting: `P * A = L * U` (real and complex).
    pub fn lu(&self) -> Result<(Matrix, Matrix, Matrix)> {
        let m = self.rows();
        let n = self.cols();
        let k_max = m.min(n);

        let mut u = self.clone();
        let mut l = Matrix::eye(m, k_max);
        let mut perm: Vec<usize> = (0..m).collect();

        for k in 0..k_max {
            let mut pivot_row = k;
            let mut max_val = u.get_c(k, k).abs();
            for i in (k + 1)..m {
                let v = u.get_c(i, k).abs();
                if v > max_val {
                    max_val = v;
                    pivot_row = i;
                }
            }
            if pivot_row != k {
                perm.swap(k, pivot_row);
                for j in k..n {
                    let tmp = u.get_c(k, j);
                    u.set_c(k, j, u.get_c(pivot_row, j));
                    u.set_c(pivot_row, j, tmp);
                }
                for j in 0..k {
                    let tmp = l.get_c(k, j);
                    l.set_c(k, j, l.get_c(pivot_row, j));
                    l.set_c(pivot_row, j, tmp);
                }
            }
            let pivot = u.get_c(k, k);
            if pivot.abs() < 1e-15 {
                continue;
            }
            for i in (k + 1)..m {
                let factor = u.get_c(i, k) / pivot;
                l.set_c(i, k, factor);
                u.set_c(i, k, Complex::ZERO);
                for j in (k + 1)..n {
                    let v = u.get_c(i, j) - factor * u.get_c(k, j);
                    u.set_c(i, j, v);
                }
            }
        }

        let mut u_out = if u.is_complex() {
            Matrix::complex_zeros(k_max, n)
        } else {
            Matrix::zeros(k_max, n)
        };
        for r in 0..k_max {
            for c in 0..n {
                u_out.set_c(r, c, u.get_c(r, c));
            }
        }
        let mut p = Matrix::zeros(m, m);
        for (row, &orig_col) in perm.iter().enumerate() {
            p.set(row, orig_col, 1.0);
        }
        Ok((l.normalize(), u_out.normalize(), p))
    }

    /// Householder QR decomposition: `A = Q * R`, where `Q` is `m x m` unitary/orthogonal
    /// and `R` is `m x n` upper triangular (real and complex).
    pub fn qr(&self) -> Result<(Matrix, Matrix)> {
        let m = self.rows();
        let n = self.cols();
        let mut r = self.clone();
        let mut q = if self.is_complex() {
            let mut e = Matrix::complex_zeros(m, m);
            for i in 0..m {
                e.set_c(i, i, Complex::ONE);
            }
            e
        } else {
            Matrix::eye(m, m)
        };

        for k in 0..m.min(n) {
            let mut norm_sq = 0.0;
            for i in k..m {
                norm_sq += r.get_c(i, k).norm_sq();
            }
            let norm = norm_sq.sqrt();
            if norm < 1e-15 {
                continue;
            }
            let x0 = r.get_c(k, k);
            // Phase-aligned alpha = -e^{i*arg(x0)} * norm to avoid cancellation in complex arithmetic
            let alpha = if x0.abs() > 1e-15 {
                let phase = x0 / Complex::from_real(x0.abs());
                -phase * Complex::from_real(norm)
            } else {
                Complex::from_real(-norm)
            };

            let mut v = vec![Complex::ZERO; m - k];
            v[0] = x0 - alpha;
            for i in (k + 1)..m {
                v[i - k] = r.get_c(i, k);
            }
            let v_norm_sq: f64 = v.iter().map(|c| c.norm_sq()).sum();
            if v_norm_sq < 1e-30 {
                continue;
            }
            let beta = Complex::from_real(2.0 / v_norm_sq);

            // Apply H = I - beta * v * v^H to R from left
            for j in k..n {
                let mut dot = Complex::ZERO;
                for i in k..m {
                    dot = dot + v[i - k].conj() * r.get_c(i, j);
                }
                let scale = beta * dot;
                for i in k..m {
                    let cur = r.get_c(i, j);
                    r.set_c(i, j, cur - scale * v[i - k]);
                }
            }
            for i in (k + 1)..m {
                r.set_c(i, k, Complex::ZERO);
            }

            // Apply H to Q from right: Q[:, k:] -= beta * (Q[:, k:] * v) * v^H
            for i in 0..m {
                let mut dot = Complex::ZERO;
                for j in k..m {
                    dot = dot + q.get_c(i, j) * v[j - k];
                }
                let scale = beta * dot;
                for j in k..m {
                    let cur = q.get_c(i, j);
                    q.set_c(i, j, cur - scale * v[j - k].conj());
                }
            }
        }
        Ok((q.normalize(), r.normalize()))
    }

    /// Upper-triangular Cholesky factorization `R' * R = A` for symmetric/Hermitian positive definite matrices.
    pub fn chol(&self) -> Result<Matrix> {
        if self.rows() != self.cols() {
            return Err(MatroError::dimension("chol() requires a square matrix"));
        }
        let n = self.rows();
        let mut r = if self.is_complex() {
            Matrix::complex_zeros(n, n)
        } else {
            Matrix::zeros(n, n)
        };
        for i in 0..n {
            for j in i..n {
                let mut sum = self.get_c(i, j);
                for k in 0..i {
                    sum = sum - r.get_c(k, i).conj() * r.get_c(k, j);
                }
                if i == j {
                    if sum.re <= 0.0 || sum.im.abs() > 1e-10 {
                        return Err(MatroError::runtime(
                            "Matrix is not positive definite for chol()",
                        ));
                    }
                    r.set_c(i, i, Complex::from_real(sum.re.sqrt()));
                } else {
                    r.set_c(i, j, sum / r.get_c(i, i));
                }
            }
        }
        Ok(r.normalize())
    }

    /// Singular Value Decomposition `A = U * S * V'` using one-sided Jacobi rotations (real & complex).
    pub fn svd(&self) -> Result<(Matrix, Matrix, Matrix)> {
        let m = self.rows();
        let n = self.cols();
        if m == 0 || n == 0 {
            return Ok((Matrix::eye(m, m), Matrix::zeros(m, n), Matrix::eye(n, n)));
        }

        if m < n {
            let (v_t, s_t, u_t) = self.ctranspose().svd()?;
            return Ok((u_t, s_t.transpose(), v_t));
        }

        // Now m >= n. Orthogonalize columns of B = A * V via one-sided complex Jacobi
        let mut b = self.clone();
        let mut v = if self.is_complex() {
            let mut e = Matrix::complex_zeros(n, n);
            for i in 0..n {
                e.set_c(i, i, Complex::ONE);
            }
            e
        } else {
            Matrix::eye(n, n)
        };

        for _ in 0..60 {
            let mut converged = true;
            for p in 0..n {
                for q in (p + 1)..n {
                    let mut app = 0.0;
                    let mut aqq = 0.0;
                    let mut apq = Complex::ZERO;
                    for i in 0..m {
                        let bp = b.get_c(i, p);
                        let bq = b.get_c(i, q);
                        app += bp.norm_sq();
                        aqq += bq.norm_sq();
                        apq = apq + bp.conj() * bq;
                    }
                    let apq_abs = apq.abs();
                    let tol = 1e-13 * (app * aqq).sqrt();
                    if apq_abs <= tol {
                        continue;
                    }
                    converged = false;
                    // Phase factor e^{i*phi} = apq / |apq| makes off-diagonal real
                    let phase = apq / Complex::from_real(apq_abs);
                    let tau = (aqq - app) / (2.0 * apq_abs);
                    let t = if tau >= 0.0 {
                        1.0 / (tau + (1.0 + tau * tau).sqrt())
                    } else {
                        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                    };
                    let c = Complex::from_real(1.0 / (1.0 + t * t).sqrt());
                    let s_real = t / (1.0 + t * t).sqrt();
                    let s_p = Complex::from_real(s_real) * phase;
                    let s_m = Complex::from_real(s_real) * phase.conj();

                    for i in 0..m {
                        let bp = b.get_c(i, p);
                        let bq = b.get_c(i, q);
                        b.set_c(i, p, c * bp - s_m * bq);
                        b.set_c(i, q, s_p * bp + c * bq);
                    }
                    for i in 0..n {
                        let vp = v.get_c(i, p);
                        let vq = v.get_c(i, q);
                        v.set_c(i, p, c * vp - s_m * vq);
                        v.set_c(i, q, s_p * vp + c * vq);
                    }
                }
            }
            if converged {
                break;
            }
        }

        let mut sigmas: Vec<(f64, usize)> = (0..n)
            .map(|j| {
                let norm = (0..m).map(|i| b.get_c(i, j).norm_sq()).sum::<f64>().sqrt();
                (norm, j)
            })
            .collect();
        sigmas.sort_by(|a, b| b.0.total_cmp(&a.0));

        let mut u = if self.is_complex() {
            Matrix::complex_zeros(m, m)
        } else {
            Matrix::zeros(m, m)
        };
        let mut s_mat = Matrix::zeros(m, n);
        let mut v_sorted = if self.is_complex() {
            Matrix::complex_zeros(n, n)
        } else {
            Matrix::zeros(n, n)
        };

        let mut basis_count = 0usize;
        for (new_col, &(sigma, old_col)) in sigmas.iter().enumerate() {
            s_mat.set(new_col, new_col, sigma);
            for i in 0..n {
                v_sorted.set_c(i, new_col, v.get_c(i, old_col));
            }
            if sigma > 1e-14 {
                for i in 0..m {
                    u.set_c(i, new_col, b.get_c(i, old_col) / Complex::from_real(sigma));
                }
                basis_count = new_col + 1;
            }
        }

        // Complete orthonormal basis of U via modified Gram-Schmidt
        for col in basis_count..m {
            let mut chosen = vec![Complex::ZERO; m];
            for cand in 0..m {
                chosen.fill(Complex::ZERO);
                chosen[cand] = Complex::ONE;
                for prev in 0..col {
                    let mut dot = Complex::ZERO;
                    for i in 0..m {
                        dot = dot + u.get_c(i, prev).conj() * chosen[i];
                    }
                    for i in 0..m {
                        chosen[i] = chosen[i] - dot * u.get_c(i, prev);
                    }
                }
                let norm = chosen.iter().map(|c| c.norm_sq()).sum::<f64>().sqrt();
                if norm > 1e-8 {
                    for i in 0..m {
                        u.set_c(i, col, chosen[i] / Complex::from_real(norm));
                    }
                    break;
                }
            }
        }

        Ok((u.normalize(), s_mat, v_sorted.normalize()))
    }

    /// Eigenvalues and eigenvectors `(V, D)` such that `A * V = V * D` (real & complex).
    pub fn eig(&self) -> Result<(Matrix, Matrix)> {
        if self.rows() != self.cols() {
            return Err(MatroError::dimension("eig() requires a square matrix"));
        }
        let n = self.rows();
        if n == 0 {
            return Ok((Matrix::empty(), Matrix::empty()));
        }
        if n == 1 {
            return Ok((Matrix::scalar(1.0), self.clone()));
        }

        if !self.is_complex() {
            let mut is_sym = true;
            'sym: for i in 0..n {
                for j in (i + 1)..n {
                    if (self.get(i, j) - self.get(j, i)).abs() > 1e-12 {
                        is_sym = false;
                        break 'sym;
                    }
                }
            }
            if is_sym {
                return self.eig_symmetric_jacobi();
            }
            return self.eig_general_qr();
        }
        self.eig_complex_qr()
    }

    fn eig_symmetric_jacobi(&self) -> Result<(Matrix, Matrix)> {
        let n = self.rows();
        let mut a = self.clone();
        let mut v = Matrix::eye(n, n);

        for _ in 0..100 {
            let mut max_off = 0.0;
            let mut p = 0;
            let mut q = 1;
            for i in 0..n {
                for j in (i + 1)..n {
                    let val = a.get(i, j).abs();
                    if val > max_off {
                        max_off = val;
                        p = i;
                        q = j;
                    }
                }
            }
            if max_off < 1e-14 {
                break;
            }
            let app = a.get(p, p);
            let aqq = a.get(q, q);
            let apq = a.get(p, q);
            let tau = (aqq - app) / (2.0 * apq);
            let t = if tau >= 0.0 {
                1.0 / (tau + (1.0 + tau * tau).sqrt())
            } else {
                -1.0 / (-tau + (1.0 + tau * tau).sqrt())
            };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;

            for i in 0..n {
                if i != p && i != q {
                    let aip = a.get(i, p);
                    let aiq = a.get(i, q);
                    let np = c * aip - s * aiq;
                    let nq = s * aip + c * aiq;
                    a.set(i, p, np);
                    a.set(p, i, np);
                    a.set(i, q, nq);
                    a.set(q, i, nq);
                }
            }
            a.set(p, p, c * c * app - 2.0 * s * c * apq + s * s * aqq);
            a.set(q, q, s * s * app + 2.0 * s * c * apq + c * c * aqq);
            a.set(p, q, 0.0);
            a.set(q, p, 0.0);

            for i in 0..n {
                let vip = v.get(i, p);
                let viq = v.get(i, q);
                v.set(i, p, c * vip - s * viq);
                v.set(i, q, s * vip + c * viq);
            }
        }

        let mut d = Matrix::zeros(n, n);
        for i in 0..n {
            d.set(i, i, a.get(i, i));
        }
        Ok((v, d))
    }

    fn eig_general_qr(&self) -> Result<(Matrix, Matrix)> {
        let n = self.rows();
        let mut h = self.clone();

        for k in 0..n.saturating_sub(2) {
            let mut norm_sq = 0.0;
            for i in (k + 1)..n {
                norm_sq += h.get(i, k).powi(2);
            }
            let norm = norm_sq.sqrt();
            if norm < 1e-15 {
                continue;
            }
            let x0 = h.get(k + 1, k);
            let alpha = if x0 >= 0.0 { -norm } else { norm };
            let len = n - (k + 1);
            let mut v = vec![0.0; len];
            v[0] = x0 - alpha;
            for i in (k + 2)..n {
                v[i - (k + 1)] = h.get(i, k);
            }
            let v_norm_sq: f64 = v.iter().map(|x| x * x).sum();
            if v_norm_sq < 1e-30 {
                continue;
            }
            let beta = 2.0 / v_norm_sq;
            for j in k..n {
                let mut dot = 0.0;
                for i in (k + 1)..n {
                    dot += v[i - (k + 1)] * h.get(i, j);
                }
                let sc = beta * dot;
                for i in (k + 1)..n {
                    h.set(i, j, h.get(i, j) - sc * v[i - (k + 1)]);
                }
            }
            for i in 0..n {
                let mut dot = 0.0;
                for j in (k + 1)..n {
                    dot += h.get(i, j) * v[j - (k + 1)];
                }
                let sc = beta * dot;
                for j in (k + 1)..n {
                    h.set(i, j, h.get(i, j) - sc * v[j - (k + 1)]);
                }
            }
        }

        let mut m = n;
        let mut iter = 0;
        let max_iter = 200 * n;
        let mut eigenvalues: Vec<Complex> = Vec::with_capacity(n);

        while m > 0 && iter < max_iter {
            iter += 1;
            if m == 1 {
                eigenvalues.push(Complex::from_real(h.get(0, 0)));
                break;
            }
            let sub = h.get(m - 1, m - 2).abs();
            let scale = h.get(m - 2, m - 2).abs() + h.get(m - 1, m - 1).abs();
            if sub <= 1e-12 * scale.max(1.0) {
                eigenvalues.push(Complex::from_real(h.get(m - 1, m - 1)));
                m -= 1;
                continue;
            }
            let a = h.get(m - 2, m - 2);
            let b = h.get(m - 2, m - 1);
            let c = h.get(m - 1, m - 2);
            let d = h.get(m - 1, m - 1);
            let tr = a + d;
            let det = a * d - b * c;
            let disc = tr * tr - 4.0 * det;

            if m == 2
                || h.get(m - 2, m - 3).abs()
                    <= 1e-12 * (h.get(m - 3, m - 3).abs() + a.abs()).max(1.0)
            {
                if disc < 0.0 {
                    let im = (-disc).sqrt() * 0.5;
                    eigenvalues.push(Complex::new(tr * 0.5, im));
                    eigenvalues.push(Complex::new(tr * 0.5, -im));
                    m -= 2;
                    continue;
                } else if m == 2 {
                    let sq = disc.sqrt() * 0.5;
                    eigenvalues.push(Complex::from_real(tr * 0.5 + sq));
                    eigenvalues.push(Complex::from_real(tr * 0.5 - sq));
                    break;
                }
            }

            let shift = if iter % 15 == 0 {
                h.get(m - 1, m - 1) + 0.75 * h.get(m - 1, m - 2).abs()
            } else if disc >= 0.0 {
                let sq = disc.sqrt() * 0.5;
                let r1 = tr * 0.5 + sq;
                let r2 = tr * 0.5 - sq;
                if (r1 - d).abs() < (r2 - d).abs() {
                    r1
                } else {
                    r2
                }
            } else {
                d
            };

            let mut sub_h = Matrix::zeros(m, m);
            for i in 0..m {
                for j in 0..m {
                    sub_h.set(i, j, h.get(i, j) - if i == j { shift } else { 0.0 });
                }
            }
            let (q, r) = sub_h.qr()?;
            let next = r.matmul(&q)?;
            for i in 0..m {
                for j in 0..m {
                    h.set(i, j, next.get(i, j) + if i == j { shift } else { 0.0 });
                }
            }
        }

        while eigenvalues.len() < n {
            let idx = eigenvalues.len();
            eigenvalues.push(Complex::from_real(h.get(idx, idx)));
        }

        self.eigenvectors_from_eigenvalues(&eigenvalues)
    }

    /// Complex shifted QR algorithm (handles both Hermitian and general non-Hermitian complex matrices).
    fn eig_complex_qr(&self) -> Result<(Matrix, Matrix)> {
        let n = self.rows();
        let mut h = self.clone();
        let mut m = n;
        let mut iter = 0;
        let max_iter = 250 * n;
        let mut eigenvalues = Vec::with_capacity(n);

        while m > 0 && iter < max_iter {
            iter += 1;
            if m == 1 {
                eigenvalues.push(h.get_c(0, 0));
                break;
            }
            let sub = h.get_c(m - 1, m - 2).abs();
            let scale = h.get_c(m - 2, m - 2).abs() + h.get_c(m - 1, m - 1).abs();
            if sub <= 1e-12 * scale.max(1.0) {
                eigenvalues.push(h.get_c(m - 1, m - 1));
                m -= 1;
                continue;
            }
            let a = h.get_c(m - 2, m - 2);
            let b = h.get_c(m - 2, m - 1);
            let c = h.get_c(m - 1, m - 2);
            let d = h.get_c(m - 1, m - 1);
            let tr = a + d;
            let det = a * d - b * c;
            let disc = (tr * tr - Complex::from_real(4.0) * det).sqrt();
            let r1 = (tr + disc) / Complex::from_real(2.0);
            let r2 = (tr - disc) / Complex::from_real(2.0);
            if m == 2 {
                eigenvalues.push(r1);
                eigenvalues.push(r2);
                break;
            }
            let shift = if iter % 15 == 0 {
                d + Complex::new(0.5 * sub, 0.3 * sub)
            } else if (r1 - d).abs() < (r2 - d).abs() {
                r1
            } else {
                r2
            };

            let mut sub_h = Matrix::complex_zeros(m, m);
            for i in 0..m {
                for j in 0..m {
                    sub_h.set_c(
                        i,
                        j,
                        h.get_c(i, j) - if i == j { shift } else { Complex::ZERO },
                    );
                }
            }
            let (q, r) = sub_h.qr()?;
            let next = r.matmul(&q)?;
            for i in 0..m {
                for j in 0..m {
                    h.set_c(
                        i,
                        j,
                        next.get_c(i, j) + if i == j { shift } else { Complex::ZERO },
                    );
                }
            }
        }

        while eigenvalues.len() < n {
            let idx = eigenvalues.len();
            eigenvalues.push(h.get_c(idx, idx));
        }

        self.eigenvectors_from_eigenvalues(&eigenvalues)
    }

    fn eigenvectors_from_eigenvalues(&self, eigenvalues: &[Complex]) -> Result<(Matrix, Matrix)> {
        let n = self.rows();
        let mut d_mat = Matrix::complex_zeros(n, n);
        let mut v_mat = Matrix::complex_zeros(n, n);

        for (col, &lam) in eigenvalues.iter().enumerate() {
            d_mat.set_c(col, col, lam);
            let mut shifted = Matrix::complex_zeros(n, n);
            let eps = Complex::new(1e-9, 1e-11);
            for i in 0..n {
                for j in 0..n {
                    let mut val = self.get_c(i, j);
                    if i == j {
                        val = val - (lam + eps);
                    }
                    shifted.set_c(i, j, val);
                }
            }
            let mut vec = Matrix::complex_col_vec(
                (0..n)
                    .map(|i| {
                        Complex::new(
                            ((i + col + 1) as f64).sin() + 1.0,
                            ((i * 2 + 1) as f64).cos() * 0.1,
                        )
                    })
                    .collect(),
            );
            for _ in 0..4 {
                if let Ok(next) = shifted.solve(&vec) {
                    let norm = (0..n)
                        .map(|i| next.get_c(i, 0).norm_sq())
                        .sum::<f64>()
                        .sqrt();
                    if norm > 1e-15 {
                        vec = next.map_c(|c| c / Complex::from_real(norm));
                    }
                } else {
                    break;
                }
            }
            for i in 0..n {
                v_mat.set_c(i, col, vec.get_c(i, 0));
            }
        }

        Ok((v_mat.normalize(), d_mat.normalize()))
    }

    /// 1D Discrete Fourier Transform (`fft`) — radix-2 Cooley-Tukey for powers of 2, exact DFT fallback otherwise.
    pub fn fft(&self) -> Matrix {
        let is_row = self.rows() == 1 && self.cols() > 1;
        let vals: Vec<Complex> = (0..self.len()).map(|i| self.get_linear_c(i)).collect();
        let out = fft_1d(&vals, false);
        if is_row {
            Matrix::complex_row_vec(out)
        } else {
            Matrix::complex_col_vec(out)
        }
    }

    /// 1D Inverse Discrete Fourier Transform (`ifft`).
    pub fn ifft(&self) -> Matrix {
        let is_row = self.rows() == 1 && self.cols() > 1;
        let vals: Vec<Complex> = (0..self.len()).map(|i| self.get_linear_c(i)).collect();
        let out = fft_1d(&vals, true);
        let res = if is_row {
            Matrix::complex_row_vec(out)
        } else {
            Matrix::complex_col_vec(out)
        };
        res.normalize()
    }

    /// 2D Discrete Fourier Transform (`fft2`).
    pub fn fft2(&self) -> Matrix {
        let r = self.rows();
        let c = self.cols();
        let mut tmp = Matrix::complex_zeros(r, c);
        for j in 0..c {
            let col: Vec<Complex> = (0..r).map(|i| self.get_c(i, j)).collect();
            let fcol = fft_1d(&col, false);
            for (i, val) in fcol.into_iter().enumerate() {
                tmp.set_c(i, j, val);
            }
        }
        let mut out = Matrix::complex_zeros(r, c);
        for i in 0..r {
            let row: Vec<Complex> = (0..c).map(|j| tmp.get_c(i, j)).collect();
            let frow = fft_1d(&row, false);
            for (j, val) in frow.into_iter().enumerate() {
                out.set_c(i, j, val);
            }
        }
        out.normalize()
    }

    /// 2D Inverse Discrete Fourier Transform (`ifft2`).
    pub fn ifft2(&self) -> Matrix {
        let r = self.rows();
        let c = self.cols();
        let mut tmp = Matrix::complex_zeros(r, c);
        for j in 0..c {
            let col: Vec<Complex> = (0..r).map(|i| self.get_c(i, j)).collect();
            let fcol = fft_1d(&col, true);
            for (i, val) in fcol.into_iter().enumerate() {
                tmp.set_c(i, j, val);
            }
        }
        let mut out = Matrix::complex_zeros(r, c);
        for i in 0..r {
            let row: Vec<Complex> = (0..c).map(|j| tmp.get_c(i, j)).collect();
            let frow = fft_1d(&row, true);
            for (j, val) in frow.into_iter().enumerate() {
                out.set_c(i, j, val);
            }
        }
        out.normalize()
    }

    /// Matrix exponential `expm(A)` via scaling-and-squaring with Taylor series (accurate to ~1e-14).
    pub fn expm(&self) -> Result<Matrix> {
        if self.rows() != self.cols() {
            return Err(MatroError::dimension("expm() requires a square matrix"));
        }
        let n = self.rows();
        if n == 0 {
            return Ok(Matrix::empty());
        }
        let norm_inf = (0..n)
            .map(|r| (0..n).map(|c| self.get_c(r, c).abs()).sum::<f64>())
            .fold(0.0f64, f64::max);
        let s = if norm_inf > 0.5 {
            (norm_inf / 0.5).log2().ceil().max(0.0) as u32
        } else {
            0
        };
        let scale = 2.0f64.powi(-(s as i32));
        let a_scaled = self.map_c(|c| c * Complex::from_real(scale));

        let mut acc = Matrix::eye(n, n);
        let mut term = Matrix::eye(n, n);
        for k in 1..=22 {
            term = term
                .matmul(&a_scaled)?
                .map_c(|c| c / Complex::from_real(k as f64));
            acc = acc.elementwise_c(&term, |x, y| x + y)?;
            let tnorm: f64 = (0..term.len()).map(|i| term.get_linear_c(i).abs()).sum();
            if tnorm < 1e-16 {
                break;
            }
        }
        for _ in 0..s {
            acc = acc.matmul(&acc)?;
        }
        Ok(acc.normalize())
    }
}

fn fft_1d(input: &[Complex], inverse: bool) -> Vec<Complex> {
    let n = input.len();
    if n <= 1 {
        return input.to_vec();
    }
    let sign = if inverse { 1.0 } else { -1.0 };

    if n.is_power_of_two() {
        let mut a = input.to_vec();
        let mut j = 0usize;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if i < j {
                a.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let ang = sign * 2.0 * std::f64::consts::PI / (len as f64);
            let wlen = Complex::new(ang.cos(), ang.sin());
            let mut i = 0;
            while i < n {
                let mut w = Complex::ONE;
                for k in 0..(len / 2) {
                    let u = a[i + k];
                    let v = a[i + k + len / 2] * w;
                    a[i + k] = u + v;
                    a[i + k + len / 2] = u - v;
                    w = w * wlen;
                }
                i += len;
            }
            len <<= 1;
        }
        if inverse {
            let inv_n = Complex::from_real(n as f64);
            for x in &mut a {
                *x = *x / inv_n;
            }
        }
        return a;
    }

    let mut out = vec![Complex::ZERO; n];
    let two_pi = sign * 2.0 * std::f64::consts::PI / (n as f64);
    for (k, slot) in out.iter_mut().enumerate() {
        let mut sum = Complex::ZERO;
        for (t, &x) in input.iter().enumerate() {
            let angle = two_pi * ((k * t) as f64);
            sum = sum + x * Complex::new(angle.cos(), angle.sin());
        }
        *slot = if inverse {
            sum / Complex::from_real(n as f64)
        } else {
            sum
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lu_qr_svd_eig_chol_roundtrip() {
        let a = Matrix::from_row_major(3, 3, &[4.0, 1.0, 1.0, 1.0, 3.0, 0.0, 1.0, 0.0, 2.0]).unwrap();
        let (l, u, p) = a.lu().unwrap();
        let pa = p.matmul(&a).unwrap();
        let lu = l.matmul(&u).unwrap();
        for i in 0..9 {
            assert!((pa.data()[i] - lu.data()[i]).abs() < 1e-10);
        }
        let (q, r) = a.qr().unwrap();
        let qr = q.matmul(&r).unwrap();
        for i in 0..9 {
            assert!((a.data()[i] - qr.data()[i]).abs() < 1e-10);
        }
        let rc = a.chol().unwrap();
        let rtr = rc.transpose().matmul(&rc).unwrap();
        for i in 0..9 {
            assert!((a.data()[i] - rtr.data()[i]).abs() < 1e-10);
        }
        let (su, ss, sv) = a.svd().unwrap();
        let recon = su.matmul(&ss).unwrap().matmul(&sv.transpose()).unwrap();
        for i in 0..9 {
            assert!((a.data()[i] - recon.data()[i]).abs() < 1e-9);
        }
        let (ev, ed) = a.eig().unwrap();
        let av = a.matmul(&ev).unwrap();
        let vd = ev.matmul(&ed).unwrap();
        for i in 0..9 {
            assert!((av.data()[i] - vd.data()[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn complex_lu_qr_svd_eig() {
        let mut a = Matrix::complex_zeros(2, 2);
        a.set_c(0, 0, Complex::new(2.0, 1.0));
        a.set_c(0, 1, Complex::new(1.0, -1.0));
        a.set_c(1, 0, Complex::new(1.0, 1.0));
        a.set_c(1, 1, Complex::new(3.0, -2.0));

        let (l, u, p) = a.lu().unwrap();
        let diff_lu = p.matmul(&a).unwrap().elementwise_c(&l.matmul(&u).unwrap(), |x, y| x - y).unwrap();
        assert!((0..4).all(|i| diff_lu.get_linear_c(i).abs() < 1e-10));

        let (q, r) = a.qr().unwrap();
        let diff_qr = a.elementwise_c(&q.matmul(&r).unwrap(), |x, y| x - y).unwrap();
        assert!((0..4).all(|i| diff_qr.get_linear_c(i).abs() < 1e-10));

        let (su, ss, sv) = a.svd().unwrap();
        let recon = su.matmul(&ss).unwrap().matmul(&sv.ctranspose()).unwrap();
        let diff_svd = a.elementwise_c(&recon, |x, y| x - y).unwrap();
        assert!((0..4).all(|i| diff_svd.get_linear_c(i).abs() < 1e-9));

        let (ev, ed) = a.eig().unwrap();
        let diff_eig = a.matmul(&ev).unwrap().elementwise_c(&ev.matmul(&ed).unwrap(), |x, y| x - y).unwrap();
        assert!((0..4).all(|i| diff_eig.get_linear_c(i).abs() < 1e-8));
    }

    #[test]
    fn fft_ifft_roundtrip() {
        let sig = Matrix::row_vec(vec![1.0, 2.0, 3.0, 4.0, 0.0, -1.0, 2.0, 1.0]);
        let back = sig.fft().ifft();
        for i in 0..sig.len() {
            assert!((sig.data()[i] - back.data()[i]).abs() < 1e-10);
        }
    }
}
