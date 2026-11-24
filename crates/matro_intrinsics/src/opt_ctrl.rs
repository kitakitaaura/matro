//! Pure-Rust Optimization & Control Toolbox intrinsics:
//! - `linprog`: Two-phase Simplex Linear Programming solver
//! - `tf`: Continuous-time transfer function object
//! - `step`: Time-domain unit step response (state-space RK4)
//! - `bode`: Frequency-domain magnitude & phase response

use std::collections::BTreeMap;

use matro_types::{MatroError, Matrix, ObjectInstance, Result, Value};

fn ok1(v: Value) -> Result<Vec<Value>> {
    Ok(vec![v])
}

/// `[x, fval] = linprog(f, A, b [, Aeq, beq [, lb [, ub]]])`
/// Minimizes `f' * x` subject to `A*x <= b`, `Aeq*x == beq`, `lb <= x <= ub`.
pub fn fn_linprog(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() < 3 {
        return Err(MatroError::arity("linprog", ">= 3 (f, A, b, ...)", args.len()));
    }
    let f_vec = args[0].as_matrix()?.data();
    let n = f_vec.len();
    if n == 0 {
        return Err(MatroError::dimension("linprog: objective f cannot be empty"));
    }

    let a_mat = args[1].as_matrix()?;
    let b_mat = args[2].as_matrix()?;
    let aeq_mat = args.get(3).map(|v| v.as_matrix()).transpose()?;
    let beq_mat = args.get(4).map(|v| v.as_matrix()).transpose()?;
    let lb_mat = args.get(5).map(|v| v.as_matrix()).transpose()?;
    let ub_mat = args.get(6).map(|v| v.as_matrix()).transpose()?;

    // Map each original variable x_j to non-negative variables:
    // - If finite lb_j: x_j = lb_j + u_j (u_j >= 0)
    // - Else (-inf): x_j = u_j+ - u_j- (u_j+, u_j- >= 0)
    struct VarMap {
        shift: f64,
        pos_col: usize,
        neg_col: Option<usize>,
    }
    let mut vmaps = Vec::with_capacity(n);
    let mut num_u = 0usize;
    let mut const_obj_shift = 0.0;
    for j in 0..n {
        let lb_j = lb_mat
            .filter(|m| !m.is_empty())
            .and_then(|m| m.data().get(j).copied())
            .unwrap_or(f64::NEG_INFINITY);
        if lb_j.is_finite() {
            vmaps.push(VarMap {
                shift: lb_j,
                pos_col: num_u,
                neg_col: None,
            });
            const_obj_shift += f_vec[j] * lb_j;
            num_u += 1;
        } else {
            vmaps.push(VarMap {
                shift: 0.0,
                pos_col: num_u,
                neg_col: Some(num_u + 1),
            });
            num_u += 2;
        }
    }

    // Collect constraints in standard form: row * u <= rhs (sense = -1), == rhs (0), >= rhs (+1)
    let mut rows: Vec<(Vec<f64>, f64, i8)> = Vec::new();
    let mut push_orig_row = |coeffs: &[f64], rhs_orig: f64, sense: i8| {
        let mut u_row = vec![0.0; num_u];
        let mut rhs = rhs_orig;
        for (j, &c) in coeffs.iter().enumerate() {
            if c != 0.0 {
                u_row[vmaps[j].pos_col] += c;
                if let Some(nc) = vmaps[j].neg_col {
                    u_row[nc] -= c;
                }
                rhs -= c * vmaps[j].shift;
            }
        }
        if rhs < 0.0 {
            for v in &mut u_row {
                *v = -*v;
            }
            rows.push((u_row, -rhs, -sense));
        } else {
            rows.push((u_row, rhs, sense));
        }
    };

    if !a_mat.is_empty() {
        for r in 0..a_mat.rows() {
            let coeffs: Vec<f64> = (0..n).map(|c| a_mat.get(r, c)).collect();
            let rhs = b_mat.data().get(r).copied().unwrap_or(0.0);
            push_orig_row(&coeffs, rhs, -1);
        }
    }
    if let (Some(aeq), Some(beq)) = (aeq_mat, beq_mat) {
        if !aeq.is_empty() {
            for r in 0..aeq.rows() {
                let coeffs: Vec<f64> = (0..n).map(|c| aeq.get(r, c)).collect();
                let rhs = beq.data().get(r).copied().unwrap_or(0.0);
                push_orig_row(&coeffs, rhs, 0);
            }
        }
    }
    if let Some(ub) = ub_mat {
        if !ub.is_empty() {
            for j in 0..n {
                if let Some(&ub_j) = ub.data().get(j) {
                    if ub_j.is_finite() {
                        let mut coeffs = vec![0.0; n];
                        coeffs[j] = 1.0;
                        push_orig_row(&coeffs, ub_j, -1);
                    }
                }
            }
        }
    }

    // Build Phase-I / Phase-II Simplex tableau
    let m = rows.len();
    let mut num_slack = 0usize;
    let mut num_art = 0usize;
    for &(_, _, sense) in &rows {
        if sense == -1 || sense == 1 {
            num_slack += 1;
        }
        if sense == 0 || sense == 1 {
            num_art += 1;
        }
    }
    let total_vars = num_u + num_slack + num_art;
    let rhs_col = total_vars;
    let mut tab = vec![vec![0.0; total_vars + 1]; m + 1];
    let mut basis = vec![0usize; m];

    let mut slack_col = num_u;
    let mut art_col = num_u + num_slack;
    let mut art_cols = Vec::with_capacity(num_art);

    for (r, (u_row, rhs, sense)) in rows.iter().enumerate() {
        for j in 0..num_u {
            tab[r][j] = u_row[j];
        }
        tab[r][rhs_col] = *rhs;
        match *sense {
            -1 => {
                tab[r][slack_col] = 1.0;
                basis[r] = slack_col;
                slack_col += 1;
            }
            0 => {
                tab[r][art_col] = 1.0;
                basis[r] = art_col;
                art_cols.push(art_col);
                art_col += 1;
            }
            1 => {
                tab[r][slack_col] = -1.0;
                slack_col += 1;
                tab[r][art_col] = 1.0;
                basis[r] = art_col;
                art_cols.push(art_col);
                art_col += 1;
            }
            _ => {}
        }
    }

    let pivot = |tab: &mut [Vec<f64>], basis: &mut [usize], pr: usize, pc: usize| {
        let cols = tab[0].len();
        let pv = tab[pr][pc];
        for c in 0..cols {
            tab[pr][c] /= pv;
        }
        for r in 0..tab.len() {
            if r != pr {
                let factor = tab[r][pc];
                if factor.abs() > 1e-15 {
                    for c in 0..cols {
                        let v = tab[pr][c];
                        tab[r][c] -= factor * v;
                    }
                }
            }
        }
        basis[pr] = pc;
    };

    let run_simplex = |tab: &mut [Vec<f64>], basis: &mut [usize], max_col: usize| -> Result<()> {
        let m_rows = basis.len();
        let rhs_c = tab[0].len() - 1;
        for _ in 0..5000 {
            // Bland's rule / most-negative reduced cost for minimization (c_bar < -1e-9)
            let mut enter: Option<usize> = None;
            let mut best_rc = -1e-9;
            for c in 0..max_col {
                if tab[m_rows][c] < best_rc {
                    best_rc = tab[m_rows][c];
                    enter = Some(c);
                }
            }
            let Some(pc) = enter else {
                return Ok(());
            };
            let mut leave: Option<usize> = None;
            let mut best_ratio = f64::INFINITY;
            for r in 0..m_rows {
                let a_rc = tab[r][pc];
                if a_rc > 1e-10 {
                    let ratio = tab[r][rhs_c] / a_rc;
                    if ratio < best_ratio - 1e-12
                        || ((ratio - best_ratio).abs() <= 1e-12
                            && leave.is_some_and(|l| basis[r] < basis[l]))
                    {
                        best_ratio = ratio;
                        leave = Some(r);
                    }
                }
            }
            let Some(pr) = leave else {
                return Err(MatroError::runtime("linprog: problem is unbounded"));
            };
            pivot(tab, basis, pr, pc);
        }
        Err(MatroError::runtime("linprog: simplex iteration limit reached"))
    };

    // Phase I: minimize sum of artificial variables if any
    if !art_cols.is_empty() {
        for &ac in &art_cols {
            tab[m][ac] = 1.0;
        }
        for r in 0..m {
            if art_cols.contains(&basis[r]) {
                for c in 0..=rhs_col {
                    let v = tab[r][c];
                    tab[m][c] -= v;
                }
            }
        }
        run_simplex(&mut tab, &mut basis, total_vars)?;
        if (-tab[m][rhs_col]).abs() > 1e-6 {
            return Err(MatroError::runtime("linprog: no feasible solution found"));
        }
        // Drive any remaining artificials out of basis
        for r in 0..m {
            if art_cols.contains(&basis[r]) {
                if let Some(pc) = (0..num_u + num_slack).find(|&c| tab[r][c].abs() > 1e-9) {
                    pivot(&mut tab, &mut basis, r, pc);
                }
            }
        }
    }

    // Phase II: set original objective row
    for v in &mut tab[m] {
        *v = 0.0;
    }
    for j in 0..n {
        tab[m][vmaps[j].pos_col] = f_vec[j];
        if let Some(nc) = vmaps[j].neg_col {
            tab[m][nc] = -f_vec[j];
        }
    }
    for r in 0..m {
        let bc = basis[r];
        let coeff = tab[m][bc];
        if coeff.abs() > 1e-15 {
            for c in 0..=rhs_col {
                let v = tab[r][c];
                tab[m][c] -= coeff * v;
            }
        }
    }
    run_simplex(&mut tab, &mut basis, num_u + num_slack)?;

    let mut u_sol = vec![0.0; total_vars];
    for r in 0..m {
        u_sol[basis[r]] = tab[r][rhs_col].max(0.0);
    }
    let mut x = vec![0.0; n];
    for j in 0..n {
        let mut val = vmaps[j].shift + u_sol[vmaps[j].pos_col];
        if let Some(nc) = vmaps[j].neg_col {
            val -= u_sol[nc];
        }
        x[j] = val;
    }
    let fval: f64 = f_vec.iter().zip(x.iter()).map(|(&c, &xi)| c * xi).sum();
    let _ = const_obj_shift;

    if nargout >= 2 {
        Ok(vec![
            Value::matrix(Matrix::col_vec(x)),
            Value::scalar(fval),
        ])
    } else {
        ok1(Value::matrix(Matrix::col_vec(x)))
    }
}

/// `sys = tf(num, den)` — continuous-time transfer function `num(s) / den(s)`.
pub fn fn_tf(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("tf", "2 (num, den)", args.len()));
    }
    let num = args[0].as_matrix()?.data().to_vec();
    let den = args[1].as_matrix()?.data().to_vec();
    if den.is_empty() || den.iter().all(|&c| c == 0.0) {
        return Err(MatroError::runtime("tf: denominator polynomial cannot be zero"));
    }
    let mut fields = BTreeMap::new();
    fields.insert("Numerator".to_string(), Value::matrix(Matrix::row_vec(num)));
    fields.insert("Denominator".to_string(), Value::matrix(Matrix::row_vec(den)));
    ok1(Value::Object(ObjectInstance {
        class_name: "tf".to_string(),
        fields,
    }))
}
