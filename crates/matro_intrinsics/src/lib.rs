use std::collections::BTreeMap;

use matro_loader::ModuleLoader;
use matro_types::{
    CellArray, Complex, MatroError, Matrix, NumericClass, PlotFigure, PlotKind, Result, Rng,
    SparseMatrix, Value,
};

mod base_ext;
mod control_ext;
mod extras;
pub mod fs_media;
mod image_proc;
pub mod interp;
mod linalg_ext;
pub mod modern_types;
mod opt_ctrl;
mod opt_ext;
mod regex;
mod signal;
mod signal_ext;
mod sparse_eig;
pub mod special_math;
mod stats;
mod stats_ext;
pub mod stats_ml;
pub mod tables;

/// Mutable host context passed to intrinsic functions.
pub struct IntrinsicContext<'a> {
    pub stdout: &'a mut String,
    pub figure: &'a mut PlotFigure,
    pub rng: &'a mut Rng,
    pub loader: &'a mut ModuleLoader,
}

/// Attempt to invoke a built-in MATLAB/Octave intrinsic function.
/// Returns `None` if `name` is not a known intrinsic.
pub fn call_intrinsic(
    name: &str,
    args: &[Value],
    nargout: usize,
    ctx: &mut IntrinsicContext<'_>,
) -> Option<Result<Vec<Value>>> {
    let res = match name {
        // Constants (callable as 0-arg or identifier when not shadowed)
        "pi" => ok1(Value::scalar(std::f64::consts::PI)),
        "e" => ok1(Value::scalar(std::f64::consts::E)),
        "i" | "j" => ok1(Value::complex(0.0, 1.0)),
        "inf" | "Inf" => fn_const_array("Inf", args, f64::INFINITY, NumericClass::Double),
        "nan" | "NaN" => fn_const_array("NaN", args, f64::NAN, NumericClass::Double),
        "true" => fn_const_array("true", args, 1.0, NumericClass::Logical),
        "false" => fn_const_array("false", args, 0.0, NumericClass::Logical),
        "eps" => fn_eps(args),
        "realmax" => ok1(Value::scalar(f64::MAX)),
        "realmin" => ok1(Value::scalar(f64::MIN_POSITIVE)),
        "flintmax" => ok1(Value::scalar((1u64 << 53) as f64)),
        "newline" => ok1(Value::String("\n".to_string())),
        "filesep" => ok1(Value::String("/".to_string())),
        "pathsep" => ok1(Value::String(":".to_string())),

        // Errors, warnings & assertions
        "error" => fn_error(args),
        "warning" => base_ext::fn_warning(args, ctx.stdout),
        "assert" => fn_assert(args),

        // Complex utilities
        "complex" => fn_complex(args),
        "real" => match args {
            [Value::Sparse(s)] => ok1(Value::sparse(s.real_part())),
            _ => fn_unary_mat(name, args, |m| Ok(m.real_part())),
        },
        "imag" => match args {
            [Value::Sparse(s)] => ok1(Value::sparse(s.imag_part())),
            _ => fn_unary_mat(name, args, |m| Ok(m.imag_part())),
        },
        "conj" => match args {
            [Value::Sparse(s)] => ok1(Value::sparse(s.conj())),
            _ => fn_unary_mat(name, args, |m| Ok(m.conj())),
        },
        "angle" => fn_unary_mat(name, args, |m| Ok(m.map_c_to_real(Complex::arg))),
        "isreal" => fn_isreal(args),

        // Logical, numeric class casting & type predicates
        "logical" => fn_cast_class("logical", args, NumericClass::Logical),
        "double" => fn_cast_class("double", args, NumericClass::Double),
        "char" => fn_char(args),
        "single" => fn_cast_class("single", args, NumericClass::Single),
        "int8" => fn_cast_class("int8", args, NumericClass::Int8),
        "int16" => fn_cast_class("int16", args, NumericClass::Int16),
        "int32" => fn_cast_class("int32", args, NumericClass::Int32),
        "int64" => fn_cast_class("int64", args, NumericClass::Int64),
        "uint8" => fn_cast_class("uint8", args, NumericClass::Uint8),
        "uint16" => fn_cast_class("uint16", args, NumericClass::Uint16),
        "uint32" => fn_cast_class("uint32", args, NumericClass::Uint32),
        "uint64" => fn_cast_class("uint64", args, NumericClass::Uint64),
        "cast" => fn_cast_named(args),
        "intmin" => fn_int_bound("intmin", args, false),
        "intmax" => fn_int_bound("intmax", args, true),
        "isinteger" => fn_val_pred("isinteger", args, |v| {
            matches!(v, Value::Matrix(m) if m.dtype().is_integer())
        }),
        "isfloat" => fn_val_pred("isfloat", args, |v| {
            matches!(v, Value::Matrix(m) if m.dtype().is_float()) || matches!(v, Value::Sparse(_))
        }),
        "isnumeric" => fn_val_pred("isnumeric", args, |v| {
            matches!(v, Value::Matrix(m) if m.dtype() != NumericClass::Logical)
                || matches!(v, Value::Sparse(_))
        }),
        "islogical" => fn_val_pred("islogical", args, |v| {
            matches!(v, Value::Matrix(m) if m.dtype() == NumericClass::Logical)
        }),
        "ischar" => fn_val_pred("ischar", args, |v| matches!(v, Value::String(_))),
        "iscell" => fn_val_pred("iscell", args, |v| matches!(v, Value::Cell(_))),
        "iscellstr" => linalg_ext::fn_iscellstr(args),
        "isstruct" => fn_val_pred("isstruct", args, |v| {
            matches!(v, Value::Struct(_) | Value::StructArray(_))
        }),
        "ishandle" => linalg_ext::fn_ishandle(args),
        "isobject" => linalg_ext::fn_isobject(args),
        "istimetable" => linalg_ext::fn_istimetable(args),
        "bitand" => fn_bit_bin("bitand", args, |a, b| a & b),
        "bitor" => fn_bit_bin("bitor", args, |a, b| a | b),
        "bitxor" => fn_bit_bin("bitxor", args, |a, b| a ^ b),
        "bitcmp" => linalg_ext::fn_bitcmp(args),
        "bitshift" => fn_bitshift(args),
        "bitget" => fn_bitget(args),
        "bitset" => fn_bitset(args),
        "find" => fn_find(args, nargout),
        "any" => fn_any_all(name, args, false),
        "all" => fn_any_all(name, args, true),
        "isnan" => fn_pred_mat(name, args, |c| c.re.is_nan() || c.im.is_nan()),
        "isinf" => fn_pred_mat(name, args, |c| c.re.is_infinite() || c.im.is_infinite()),
        "isfinite" => fn_pred_mat(name, args, |c| c.re.is_finite() && c.im.is_finite()),
        "isempty" => fn_isempty(args),
        "isequal" => fn_isequal(args, false),
        "isequaln" => fn_isequal(args, true),

        // Constructors & RNG
        "zeros" => fn_fill(name, args, Matrix::zeros),
        "ones" => fn_fill(name, args, Matrix::ones),
        "eye" => fn_fill(name, args, Matrix::eye),
        "rand" => fn_rand(args, ctx.rng, false),
        "randn" => fn_rand(args, ctx.rng, true),
        "rng" => fn_rng(args, nargout, ctx.rng),
        "linspace" => fn_linspace(args),
        "logspace" => fn_logspace(args),
        "meshgrid" => fn_meshgrid(args, nargout),
        "ndgrid" => fn_ndgrid(args, nargout),
        "diag" => fn_diag(args),
        "magic" => fn_magic(args),
        "toeplitz" => linalg_ext::fn_toeplitz(args),
        "hankel" => linalg_ext::fn_hankel(args),
        "vander" => linalg_ext::fn_vander(args),
        "compan" => linalg_ext::fn_compan(args),
        "hilb" => linalg_ext::fn_hilb(args),
        "invhilb" => linalg_ext::fn_invhilb(args),
        "pascal" => linalg_ext::fn_pascal(args),
        "gallery" => linalg_ext::fn_gallery(args),
        "cell" => fn_cell(args),
        "struct" => fn_struct(args),
        "sparse" => fn_sparse(args),
        "speye" => fn_speye(args),
        "sprand" => fn_sprand(args, ctx.rng, false, false),
        "sprandn" => fn_sprand(args, ctx.rng, true, false),
        "sprandsym" => fn_sprand(args, ctx.rng, true, true),
        "full" => fn_full(args),
        "issparse" => fn_issparse(args),
        "nnz" => fn_nnz(args),
        "nzmax" => linalg_ext::fn_nzmax(args),
        "pcg" => fn_pcg(args),

        // Struct / object / string / type helpers
        "fieldnames" => fn_fieldnames(args),
        "isfield" => fn_isfield(args),
        "isa" => fn_isa(args),
        "strcmp" => fn_strcmp(args),
        "class" => fn_class(args),
        "sprintf" => fn_sprintf(args),
        "num2str" => fn_num2str(args),
        "str2double" => fn_str2double(args),
        "strsplit" => fn_strsplit(args),
        "strjoin" => fn_strjoin(args),
        "strcat" => fn_strcat(args),
        "upper" => fn_str1(name, args, str::to_uppercase),
        "lower" => fn_str1(name, args, str::to_lowercase),
        "strfind" => extras::fn_strfind(args),
        "strrep" => extras::fn_strrep(args),
        "contains" => extras::fn_str_pred(name, args),
        "regexp" => regex::fn_regexp(args, nargout, false),
        "regexpi" => regex::fn_regexp(args, nargout, true),
        "regexprep" => regex::fn_regexprep(args),
        "jsonencode" => extras::fn_jsonencode(args),
        "jsondecode" => extras::fn_jsondecode(args),

        // Shape, N-D arrays & queries
        "size" => fn_size(args, nargout),
        "ndims" => extras::fn_ndims(args),
        "squeeze" => extras::fn_squeeze(args),
        "permute" => extras::fn_permute(args),
        "cat" => extras::fn_cat(args),
        "repmat" => extras::fn_repmat(args),
        "kron" => fn_kron(args),
        "fliplr" => extras::fn_fliplr(args),
        "flipud" => extras::fn_flipud(args),
        "rot90" => extras::fn_rot90(args),
        "tril" => extras::fn_tri("tril", args, false),
        "triu" => extras::fn_tri("triu", args, true),
        "length" => fn_length(args),
        "numel" => fn_numel(args),
        "rows" => fn_rows(args),
        "cols" => fn_cols(args),
        "reshape" => fn_reshape(args),
        "ind2sub" => fn_ind2sub(args, nargout),
        "sub2ind" => fn_sub2ind(args),
        "transpose" => match args {
            [Value::Sparse(s)] => ok1(Value::sparse(s.transpose())),
            _ => fn_unary_mat(name, args, |m| Ok(m.transpose())),
        },
        "ctranspose" => match args {
            [Value::Sparse(s)] => ok1(Value::sparse(s.ctranspose())),
            _ => fn_unary_mat(name, args, |m| Ok(m.ctranspose())),
        },

        // Linear algebra, signal processing & numerics
        "det" => fn_unary_mat(name, args, |m| Ok(Matrix::complex_scalar(m.det_c()?).normalize())),
        "inv" => fn_unary_mat(name, args, |m| m.inv()),
        "expm" => fn_unary_mat(name, args, |m| m.expm()),
        "logm" => fn_unary_mat(name, args, |m| m.logm()),
        "sqrtm" => fn_unary_mat(name, args, |m| m.sqrtm()),
        "lu" => fn_lu(args, nargout),
        "ilu" => linalg_ext::fn_ilu(args, nargout),
        "qr" => fn_qr(args, nargout),
        "svd" => fn_svd(args, nargout),
        "eig" => fn_eig(args, nargout),
        "eigs" => sparse_eig::fn_eigs(args, nargout, ctx.rng),
        "svds" => sparse_eig::fn_svds(args, nargout, ctx.rng),
        "chol" => fn_chol(args, nargout),
        "ichol" => linalg_ext::fn_ichol(args),
        "hess" => linalg_ext::fn_hess(args, nargout),
        "schur" => linalg_ext::fn_schur(args, nargout),
        "balance" => linalg_ext::fn_balance(args, nargout),
        "rank" => fn_rank(args),
        "cond" => fn_cond(args),
        "rcond" => linalg_ext::fn_rcond(args),
        "bandwidth" => linalg_ext::fn_bandwidth(args, nargout),
        "del2" => linalg_ext::fn_del2(args),
        "trace" => fn_trace(args),
        "dot" => fn_dot(args),
        "norm" => fn_norm(args),
        "fft" => fn_fft(args, false),
        "ifft" => fn_fft(args, true),
        "fft2" => fn_fft2(args, false),
        "ifft2" => fn_fft2(args, true),
        "fftn" => fn_fftn(args, false),
        "ifftn" => fn_fftn(args, true),
        "roots" => extras::fn_roots(args),
        "poly" => extras::fn_poly(args),
        "polyfit" => fn_polyfit(args, nargout),
        "polyval" => fn_polyval(args, nargout),
        "conv" => fn_conv(args),
        "conv2" => extras::fn_conv2(args),
        "filter" => extras::fn_filter(args, nargout),
        "interp1" => interp::fn_interp1(args),
        "interp2" => extras::fn_interp2(args),
        "interp3" => interp::fn_interp3(args),
        "interpn" => interp::fn_interpn(args),
        "pchip" => interp::fn_pchip(args),
        "spline" => interp::fn_spline(args),
        "makima" => interp::fn_makima(args),
        "mkpp" => interp::fn_mkpp(args),
        "ppval" => interp::fn_ppval(args),
        "unmkpp" => interp::fn_unmkpp(args, nargout),
        "griddedInterpolant" => interp::fn_gridded_interpolant(args),
        "sort" => fn_sort(args, nargout),
        "unique" => extras::fn_unique(args, nargout),
        "ismember" => extras::fn_ismember(args, nargout),
        "setdiff" | "intersect" | "union" | "setxor" => extras::fn_set_binary(name, args, nargout),
        "cumsum" => fn_cumsum(args),
        "cumprod" => extras::fn_cumprod(args),
        "diff" => fn_diff(args),

        // Reductions
        "sum" => fn_sum(args),
        "prod" => fn_reduce_c(name, args, |slice| {
            slice.iter().copied().fold(Complex::ONE, |a, b| a * b)
        }),
        "mean" => fn_reduce_c(name, args, |slice| {
            if slice.is_empty() {
                Complex::from_real(f64::NAN)
            } else {
                let s = slice.iter().copied().fold(Complex::ZERO, |a, b| a + b);
                s / Complex::from_real(slice.len() as f64)
            }
        }),
        "min" => fn_minmax("min", args, nargout, false),
        "max" => fn_minmax("max", args, nargout, true),

        // Element-wise math
        "abs" => fn_unary_mat(name, args, |m| Ok(m.map_c_to_real(Complex::abs))),
        "sqrt" => fn_unary_mat(name, args, |m| Ok(m.map_c(Complex::sqrt).normalize())),
        "exp" => fn_unary_mat(name, args, |m| Ok(m.map_c(Complex::exp).normalize())),
        "log" => fn_unary_mat(name, args, |m| Ok(m.map_c(Complex::ln).normalize())),
        "sin" => fn_unary_mat(name, args, |m| Ok(m.map_c(Complex::sin).normalize())),
        "cos" => fn_unary_mat(name, args, |m| Ok(m.map_c(Complex::cos).normalize())),
        "tan" => fn_unary_mat(name, args, |m| Ok(m.map_c(Complex::tan).normalize())),
        "asin" => fn_unary_mat(name, args, |m| Ok(m.map_c(Complex::asin).normalize())),
        "acos" => fn_unary_mat(name, args, |m| Ok(m.map_c(Complex::acos).normalize())),
        "atan" => fn_unary_mat(name, args, |m| Ok(m.map_c(Complex::atan).normalize())),
        "atan2" => fn_elem2(name, args, f64::atan2),
        "log10" => fn_unary_mat(name, args, |m| Ok(m.map_c(Complex::log10).normalize())),
        "log2" => fn_log2(args, nargout),
        "floor" => fn_unary_mat(name, args, |m| {
            Ok(m.map_c(|c| Complex::new(c.re.floor(), c.im.floor())).normalize())
        }),
        "ceil" => fn_unary_mat(name, args, |m| {
            Ok(m.map_c(|c| Complex::new(c.re.ceil(), c.im.ceil())).normalize())
        }),
        "round" => fn_round(args),
        "sign" => fn_unary_mat(name, args, |m| Ok(m.map_c(Complex::sign).normalize())),
        "mod" => fn_elem2(name, args, |a, b| {
            if b == 0.0 {
                a
            } else if b.is_infinite() {
                if a.is_finite() && (a == 0.0 || a.signum() == b.signum()) {
                    a
                } else {
                    f64::NAN
                }
            } else {
                let r = a - (a / b).floor() * b;
                if r == b { 0.0 } else { r }
            }
        }),
        "rem" => fn_elem2(name, args, |a, b| a % b),

        // Output & session commands
        "disp" => fn_disp(args, ctx.stdout),
        "fprintf" | "printf" => fn_fprintf(args, nargout, ctx.stdout),
        "format" => Ok(vec![]),
        "clc" => Ok(vec![]),
        "pause" => Ok(vec![]),

        // Statistics toolbox
        "std" => stats::fn_std_var("std", args, nargout, true),
        "var" => stats::fn_std_var("var", args, nargout, false),
        "median" => stats::fn_median(args),
        "prctile" => stats::fn_quantile("prctile", args, true),
        "quantile" => stats::fn_quantile("quantile", args, false),
        "cov" => stats::fn_cov_or_corr("cov", args, nargout, false),
        "corrcoef" => stats::fn_cov_or_corr("corrcoef", args, nargout, true),
        "corr" => stats_ext::fn_corr(args, nargout),
        "geomean" | "harmmean" | "trimmean" | "nansum" | "nanmean" | "nanmedian" | "nanstd"
        | "nanvar" => stats_ext::fn_extra_means(name, args),
        "mode" => stats_ext::fn_mode(args, nargout),
        "erf" => stats::fn_erf(args),
        "erfc" => stats::fn_erfc(args),
        "normpdf" => stats::fn_normpdf(args),
        "normcdf" => stats::fn_normcdf(args),
        "norminv" => stats::fn_norminv(args),
        "mvnpdf" => stats_ext::fn_mvnpdf(args),
        "mvnrnd" => stats_ext::fn_mvnrnd(args, ctx.rng),
        "lognpdf" | "logncdf" | "logninv" => stats_ext::fn_logn_dist(name, args),
        "gampdf" | "gamcdf" | "gaminv" => stats_ext::fn_gam_dist(name, args),
        "betapdf" | "betacdf" | "betainv" => stats_ext::fn_beta_dist(name, args),
        "unifpdf" | "unifcdf" | "unifinv" => stats_ext::fn_unif_dist(name, args),
        "normrnd" | "lognrnd" | "unifrnd" | "exprnd" | "gamrnd" | "betarnd" | "poissrnd"
        | "binornd" | "chi2rnd" | "trnd" | "frnd" => stats_ext::fn_dist_rnd(name, args, ctx.rng),
        "randsample" => stats_ext::fn_randsample(args, ctx.rng),
        "datasample" => stats_ext::fn_datasample(args, nargout, ctx.rng),
        "ttest" => stats::fn_ttest(args, nargout),
        "ttest2" => stats_ml::fn_ttest2(args, nargout),
        "kstest" => stats_ext::fn_kstest(args, nargout),
        "kstest2" => stats_ext::fn_kstest2(args, nargout),
        "ranksum" => stats_ext::fn_ranksum(args, nargout),
        "signrank" => stats_ext::fn_signrank(args, nargout),
        "kmeans" => stats::fn_kmeans(args, nargout, ctx.rng),
        "pca" => stats::fn_pca(args, nargout),
        "gamma" => stats_ml::fn_gamma_fn(args, false),
        "gammaln" => stats_ml::fn_gamma_fn(args, true),
        "gammainc" => stats_ml::fn_gammainc(args),
        "betainc" => stats_ml::fn_betainc(args),
        "tpdf" | "tcdf" | "tinv" => stats_ml::fn_t_dist(name, args),
        "chi2pdf" | "chi2cdf" | "chi2inv" => stats_ml::fn_chi2_dist(name, args),
        "fpdf" | "fcdf" | "finv" => stats_ml::fn_f_dist(name, args),
        "exppdf" | "expcdf" | "expinv" => stats_ml::fn_exp_dist(name, args),
        "binopdf" | "binocdf" | "binoinv" => stats_ml::fn_bino_dist(name, args),
        "poisspdf" | "poisscdf" | "poissinv" => stats_ml::fn_poiss_dist(name, args),
        "zscore" => stats_ml::fn_zscore(args, nargout),
        "skewness" | "kurtosis" | "iqr" | "mad" => stats_ml::fn_moment_stat(name, args),
        "fitlm" => stats_ml::fn_fitlm(args),
        "regress" => stats_ml::fn_regress(args, nargout),
        "anova1" => stats_ml::fn_anova1(args, nargout),
        "fitcknn" => stats_ml::fn_fitcknn(args),
        "fitcsvm" => stats_ml::fn_fitcsvm(args),
        "fitcnb" => stats_ml::fn_fitcnb(args),
        "predict" => stats_ml::fn_predict(args),
        "pdist" => stats_ml::fn_pdist(args),
        "pdist2" => stats_ext::fn_pdist2(args),
        "knnsearch" => stats_ext::fn_knnsearch(args, nargout),
        "squareform" => stats_ml::fn_squareform(args),
        "linkage" => stats_ml::fn_linkage(args),
        "cluster" => stats_ml::fn_cluster(args),

        // Signal processing toolbox
        "fftshift" => signal::fn_fftshift(args, false),
        "ifftshift" => signal::fn_fftshift(args, true),
        "hann" | "hamming" | "blackman" => signal::fn_window(name, args),
        "xcorr" => signal::fn_xcorr_or_xcov(args, nargout, false),
        "xcov" => signal::fn_xcorr_or_xcov(args, nargout, true),
        "findpeaks" => signal::fn_findpeaks(args, nargout),
        "butter" => signal::fn_butter(args, nargout),
        "cheby1" => signal_ext::fn_cheby1(args, nargout),
        "cheby2" => signal_ext::fn_cheby2(args, nargout),
        "zp2sos" => signal_ext::fn_zp2sos(args, nargout),
        "sos2tf" => signal_ext::fn_sos2tf(args),
        "tf2sos" => signal_ext::fn_tf2sos(args, nargout),
        "sosfilt" => signal_ext::fn_sosfilt(args),
        "fir1" => signal_ext::fn_fir1(args),
        "freqz" => signal_ext::fn_freqz(args, nargout, ctx.figure),
        "grpdelay" => signal_ext::fn_grpdelay(args, nargout),
        "filtfilt" => signal::fn_filtfilt(args),
        "pwelch" => signal::fn_pwelch(args, nargout),
        "periodogram" => signal::fn_periodogram(args, nargout),
        "spectrogram" | "stft" => signal_ext::fn_spectrogram(args, nargout, ctx.figure),
        "hilbert" => signal_ext::fn_hilbert(args),
        "envelope" => signal_ext::fn_envelope(args, nargout),
        "medfilt1" => signal_ext::fn_medfilt1(args),
        "sgolayfilt" => signal_ext::fn_sgolayfilt(args),
        "detrend" => signal_ext::fn_detrend(args),
        "downsample" => signal_ext::fn_downsample(args),
        "upsample" => signal_ext::fn_upsample(args),
        "decimate" => signal_ext::fn_decimate(args),
        "resample" => signal_ext::fn_resample(args),
        "bartlett" | "kaiser" | "flattopwin" | "tukeywin" | "gausswin" | "rectwin" | "triang"
        | "blackmanharris" => signal_ext::fn_extra_window(name, args),

        // Optimization & Control toolbox
        "linprog" => opt_ctrl::fn_linprog(args, nargout),
        "quadprog" => opt_ext::fn_quadprog(args, nargout),
        "lsqlin" => opt_ext::fn_lsqlin(args, nargout),
        "tf" => opt_ctrl::fn_tf(args),
        "ss" => control_ext::fn_ss(args),
        "tf2ss" => control_ext::fn_tf2ss(args),
        "ss2tf" => control_ext::fn_ss2tf(args),
        "series" => control_ext::fn_series(args),
        "parallel" => control_ext::fn_parallel(args),
        "feedback" => control_ext::fn_feedback(args),
        "step" => control_ext::fn_step_unified(args, nargout, ctx.figure),
        "impulse" => control_ext::fn_impulse(args, nargout, ctx.figure),
        "lsim" => control_ext::fn_lsim(args, nargout, ctx.figure),
        "initial" => control_ext::fn_initial(args, nargout, ctx.figure),
        "stepinfo" => control_ext::fn_stepinfo(args),
        "rlocus" => control_ext::fn_rlocus(args, nargout, ctx.figure),
        "bode" => control_ext::fn_bode_unified(args, nargout, ctx.figure),
        "nyquist" => control_ext::fn_nyquist(args, nargout, ctx.figure),
        "pzmap" => control_ext::fn_pzmap(args, ctx.figure),
        "pole" => control_ext::fn_pole(args),
        "zero" => control_ext::fn_zero(args),
        "dcgain" => control_ext::fn_dcgain(args),
        "margin" => control_ext::fn_margin(args, nargout),
        "ctrb" => control_ext::fn_ctrb(args),
        "obsv" => control_ext::fn_obsv(args),
        "place" | "acker" => control_ext::fn_place(args),
        "lyap" => control_ext::fn_lyap(args),
        "dlyap" => control_ext::fn_dlyap(args),
        "care" => control_ext::fn_care(args, nargout),
        "dare" => control_ext::fn_dare(args, nargout),
        "lqr" => control_ext::fn_lqr(args, nargout),
        "dlqr" => control_ext::fn_dlqr(args, nargout),
        "lqe" => control_ext::fn_lqe(args, nargout),
        "c2d" => control_ext::fn_c2d(args),
        "d2c" => control_ext::fn_d2c(args),
        "ssdata" => control_ext::fn_ssdata(args, nargout),
        "tfdata" => control_ext::fn_tfdata(args, nargout),

        // VFS & module path helpers
        "addpath" => fn_addpath(args, ctx.loader),
        "rmpath" => fn_rmpath(args, ctx.loader),
        "genpath" => fn_genpath(args, ctx.loader),
        "path" => fn_path(args, ctx.loader),
        "csvread" => fn_csvread(args, ctx.loader),
        "csvwrite" => fn_csvwrite(args, ctx.loader),
        "readmatrix" => fs_media::fn_readmatrix(args, ctx.loader),
        "writematrix" => fs_media::fn_writematrix(args, ctx.loader),
        "dlmread" => fs_media::fn_dlmread(args, ctx.loader),
        "dlmwrite" => fs_media::fn_dlmwrite(args, ctx.loader),
        "importdata" => fs_media::fn_importdata(args, ctx.loader),
        "fileread" => fn_fileread(args, ctx.loader),

        // Tables & dictionaries
        "table" => tables::fn_table(args),
        "height" => tables::fn_height(args),
        "width" => tables::fn_width(args),
        "istable" => tables::fn_istable(args),
        "summary" => tables::fn_summary(args, ctx.stdout),
        "sortrows" => tables::fn_sortrows(args, nargout),
        "readtable" => tables::fn_readtable(args, ctx.loader),
        "writetable" => tables::fn_writetable(args, ctx.loader),
        "head" => tables::fn_head_tail(args, false),
        "tail" => tables::fn_head_tail(args, true),
        "array2table" => tables::fn_array2table(args),
        "table2array" => tables::fn_table2array(args),
        "table2struct" => tables::fn_table2struct(args),
        "struct2table" => tables::fn_struct2table(args),
        "cell2table" => tables::fn_cell2table(args),
        "table2cell" => tables::fn_table2cell(args),
        "renamevars" => tables::fn_renamevars(args),
        "removevars" => tables::fn_removevars(args),
        "addvars" => tables::fn_addvars(args),
        "movevars" => tables::fn_movevars(args),
        "innerjoin" => tables::fn_table_join(args, tables::JoinKind::Inner),
        "outerjoin" => tables::fn_table_join(args, tables::JoinKind::Outer),
        "groupsummary" => tables::fn_groupsummary(args),
        "dictionary" => tables::fn_dictionary(args),
        "isKey" => tables::fn_is_key(args),
        "keys" => tables::fn_keys(args),
        "values" => tables::fn_values(args),

        // Modern MATLAB data types: datetime, duration, categorical, timetable, string arrays
        "datetime" => modern_types::fn_datetime(args),
        "duration" => modern_types::fn_duration(args),
        "seconds" => modern_types::fn_duration_unit("seconds", args, 1.0),
        "milliseconds" => modern_types::fn_duration_unit("milliseconds", args, 1e-3),
        "minutes" => modern_types::fn_duration_unit("minutes", args, 60.0),
        "hours" => modern_types::fn_duration_unit("hours", args, 3600.0),
        "days" => modern_types::fn_duration_unit("days", args, 86400.0),
        "year" => modern_types::fn_datetime_part(args, "Year"),
        "month" => modern_types::fn_datetime_part(args, "Month"),
        "day" => modern_types::fn_datetime_part(args, "Day"),
        "hour" => modern_types::fn_datetime_part(args, "Hour"),
        "minute" => modern_types::fn_datetime_part(args, "Minute"),
        "second" => modern_types::fn_datetime_part(args, "Second"),
        "posixtime" => modern_types::fn_datetime_part(args, "EpochSec"),
        "datestr" => modern_types::fn_datetime_part(args, "datestr"),
        "isdatetime" => fn_val_pred("isdatetime", args, |v| {
            matches!(v, Value::Object(o) if o.class_name == "datetime")
        }),
        "isduration" => fn_val_pred("isduration", args, |v| {
            matches!(v, Value::Object(o) if o.class_name == "duration")
        }),
        "categorical" => modern_types::fn_categorical(args),
        "categories" => modern_types::fn_categories(args),
        "countcats" => modern_types::fn_countcats(args),
        "iscategorical" => fn_val_pred("iscategorical", args, |v| {
            matches!(v, Value::Object(o) if o.class_name == "categorical")
        }),
        "timetable" => modern_types::fn_timetable(args),
        "string" => modern_types::fn_string(args),
        "isstring" => fn_val_pred("isstring", args, |v| {
            matches!(v, Value::Object(o) if o.class_name == "string")
        }),
        "strlength" => modern_types::fn_strlength(args),
        "split" => modern_types::fn_split(args),
        "join" => {
            if matches!(args.first(), Some(Value::Table(_))) {
                tables::fn_table_join(args, tables::JoinKind::Left)
            } else {
                modern_types::fn_join(args)
            }
        }
        "startsWith" => extras::fn_str_pred("startsWith", args),
        "endsWith" => extras::fn_str_pred("endsWith", args),
        "erase" => modern_types::fn_erase(args),
        "strip" => modern_types::fn_strip(args),
        "replace" => extras::fn_strrep(args),

        // Base MATLAB extended stdlib (hyperbolic/degree trig, coords, sparse/array/linalg/calculus/moving/strings/structs)
        "sinh" => base_ext::fn_unary_c(name, args, base_ext::c_sinh),
        "cosh" => base_ext::fn_unary_c(name, args, base_ext::c_cosh),
        "tanh" => base_ext::fn_unary_c(name, args, base_ext::c_tanh),
        "asinh" => base_ext::fn_unary_c(name, args, base_ext::c_asinh),
        "acosh" => base_ext::fn_unary_c(name, args, base_ext::c_acosh),
        "atanh" => base_ext::fn_unary_c(name, args, base_ext::c_atanh),
        "sind" => base_ext::fn_elem_real(name, args, base_ext::sind_f),
        "cosd" => base_ext::fn_elem_real(name, args, base_ext::cosd_f),
        "tand" => base_ext::fn_elem_real(name, args, base_ext::tand_f),
        "asind" => base_ext::fn_elem_real(name, args, |x| x.asin().to_degrees()),
        "acosd" => base_ext::fn_elem_real(name, args, |x| x.acos().to_degrees()),
        "atand" => base_ext::fn_elem_real(name, args, |x| x.atan().to_degrees()),
        "atan2d" => base_ext::fn_elem_real2(name, args, |y, x| y.atan2(x).to_degrees()),
        "sec" => base_ext::fn_unary_c(name, args, |z| Complex::ONE / z.cos()),
        "csc" => base_ext::fn_unary_c(name, args, |z| Complex::ONE / z.sin()),
        "cot" => base_ext::fn_unary_c(name, args, |z| z.cos() / z.sin()),
        "secd" => base_ext::fn_elem_real(name, args, |x| 1.0 / base_ext::cosd_f(x)),
        "cscd" => base_ext::fn_elem_real(name, args, |x| 1.0 / base_ext::sind_f(x)),
        "cotd" => base_ext::fn_elem_real(name, args, |x| base_ext::cosd_f(x) / base_ext::sind_f(x)),
        "asec" => base_ext::fn_unary_c(name, args, |z| (Complex::ONE / z).acos()),
        "acsc" => base_ext::fn_unary_c(name, args, |z| (Complex::ONE / z).asin()),
        "acot" => base_ext::fn_unary_c(name, args, |z| {
            if z.re == 0.0 && z.im == 0.0 {
                Complex::from_real(std::f64::consts::FRAC_PI_2)
            } else {
                (Complex::ONE / z).atan()
            }
        }),
        "asecd" => base_ext::fn_elem_real(name, args, |x| (1.0 / x).acos().to_degrees()),
        "acscd" => base_ext::fn_elem_real(name, args, |x| (1.0 / x).asin().to_degrees()),
        "acotd" => base_ext::fn_elem_real(name, args, |x| {
            if x == 0.0 { 90.0 } else { (1.0 / x).atan().to_degrees() }
        }),
        "sech" => base_ext::fn_unary_c(name, args, |z| Complex::ONE / base_ext::c_cosh(z)),
        "csch" => base_ext::fn_unary_c(name, args, |z| Complex::ONE / base_ext::c_sinh(z)),
        "coth" => base_ext::fn_unary_c(name, args, |z| base_ext::c_cosh(z) / base_ext::c_sinh(z)),
        "asech" => base_ext::fn_unary_c(name, args, |z| base_ext::c_acosh(Complex::ONE / z)),
        "acsch" => base_ext::fn_unary_c(name, args, |z| base_ext::c_asinh(Complex::ONE / z)),
        "acoth" => base_ext::fn_unary_c(name, args, |z| base_ext::c_atanh(Complex::ONE / z)),
        "deg2rad" => base_ext::fn_elem_real(name, args, f64::to_radians),
        "rad2deg" => base_ext::fn_elem_real(name, args, f64::to_degrees),
        "cart2pol" => base_ext::fn_cart2pol(args, nargout),
        "pol2cart" => base_ext::fn_pol2cart(args, nargout),
        "cart2sph" => base_ext::fn_cart2sph(args, nargout),
        "sph2cart" => base_ext::fn_sph2cart(args, nargout),
        "hypot" => base_ext::fn_elem_real2(name, args, f64::hypot),
        "fix" => base_ext::fn_elem_real(name, args, f64::trunc),
        "nextpow2" => base_ext::fn_elem_real(name, args, base_ext::nextpow2_f),
        "log1p" => base_ext::fn_elem_real(name, args, f64::ln_1p),
        "expm1" => base_ext::fn_elem_real(name, args, f64::exp_m1),
        "nthroot" => base_ext::fn_elem_real2(name, args, base_ext::nthroot_f),
        "pow2" => base_ext::fn_pow2(args),
        "unwrap" => base_ext::fn_unwrap(args),
        "wrapToPi" => base_ext::fn_elem_real(name, args, base_ext::wrap_to_pi),
        "wrapTo2Pi" => base_ext::fn_elem_real(name, args, base_ext::wrap_to_2pi),
        "wrapTo180" => base_ext::fn_elem_real(name, args, base_ext::wrap_to_180),
        "wrapTo360" => base_ext::fn_elem_real(name, args, base_ext::wrap_to_360),
        "sinc" => base_ext::fn_elem_real(name, args, base_ext::sinc_f),
        "db2mag" => base_ext::fn_elem_real(name, args, |db| 10.0f64.powf(db / 20.0)),
        "mag2db" => base_ext::fn_elem_real(name, args, |m| 20.0 * m.log10()),
        "db2pow" => base_ext::fn_elem_real(name, args, |db| 10.0f64.powf(db / 10.0)),
        "pow2db" => base_ext::fn_elem_real(name, args, |p| 10.0 * p.log10()),
        "randi" => base_ext::fn_randi(args, ctx.rng),
        "randperm" => base_ext::fn_randperm(args, ctx.rng),
        "blkdiag" => base_ext::fn_blkdiag(args),
        "circshift" => base_ext::fn_circshift(args),
        "flip" => base_ext::fn_flip(args),
        "repelem" => base_ext::fn_repelem(args),
        "accumarray" => base_ext::fn_accumarray(args),
        "histcounts" => base_ext::fn_histcounts(args, nargout),
        "histc" => base_ext::fn_histc(args, nargout),
        "discretize" => base_ext::fn_discretize(args, nargout),
        "beta" => special_math::fn_beta(args, false),
        "betaln" => special_math::fn_beta(args, true),
        "erfinv" => special_math::fn_erfinv(args, false),
        "erfcinv" => special_math::fn_erfinv(args, true),
        "besselj" | "bessely" | "besseli" | "besselk" => special_math::fn_bessel(name, args),
        "legendre" => special_math::fn_legendre(args),
        "ellipke" => special_math::fn_ellipke(args, nargout),
        "ipermute" => base_ext::fn_ipermute(args),
        "horzcat" => base_ext::fn_horzcat(args),
        "vertcat" => base_ext::fn_vertcat(args),
        "spdiags" => base_ext::fn_spdiags(args, nargout),
        "spones" => base_ext::fn_spones(args),
        "spalloc" => base_ext::fn_spalloc(args),
        "nonzeros" => base_ext::fn_nonzeros(args),
        "pinv" => base_ext::fn_pinv(args),
        "null" => base_ext::fn_null_orth(args, true),
        "orth" => base_ext::fn_null_orth(args, false),
        "rref" => base_ext::fn_rref(args, nargout),
        "cross" => base_ext::fn_cross(args),
        "vecnorm" => base_ext::fn_vecnorm(args),
        "rms" => base_ext::fn_rms(args),
        "normalize" => base_ext::fn_normalize(args, nargout),
        "rescale" => base_ext::fn_rescale(args),
        "clip" => base_ext::fn_clip(args),
        "trapz" => base_ext::fn_trapz(args, false),
        "cumtrapz" => base_ext::fn_trapz(args, true),
        "gradient" => base_ext::fn_gradient(args, nargout),
        "deconv" => base_ext::fn_deconv(args, nargout),
        "polyder" => base_ext::fn_polyder(args, nargout),
        "polyint" => base_ext::fn_polyint(args),
        "factorial" => base_ext::fn_factorial(args),
        "nchoosek" => base_ext::fn_nchoosek(args),
        "perms" => base_ext::fn_perms(args),
        "gcd" => base_ext::fn_gcd_lcm(args, false),
        "lcm" => base_ext::fn_gcd_lcm(args, true),
        "isprime" => base_ext::fn_isprime(args),
        "primes" => base_ext::fn_primes(args),
        "factor" => linalg_ext::fn_factor(args),
        "isscalar" | "isvector" | "isrow" | "iscolumn" | "ismatrix" => {
            base_ext::fn_shape_pred(name, args)
        }
        "issymmetric" | "ishermitian" | "isdiag" | "istril" | "istriu" | "issorted" => {
            base_ext::fn_struct_pred(name, args)
        }
        "movmean" | "movsum" | "movstd" | "movvar" | "movmedian" | "movmin" | "movmax" => {
            base_ext::fn_mov_window(name, args)
        }
        "smoothdata" => base_ext::fn_smoothdata(args),
        "islocalmin" => base_ext::fn_islocal_extrema(args, false),
        "islocalmax" => base_ext::fn_islocal_extrema(args, true),
        "mink" => base_ext::fn_mink_maxk(args, nargout, false),
        "maxk" => base_ext::fn_mink_maxk(args, nargout, true),
        "topkrows" => base_ext::fn_topkrows(args, nargout),
        "shiftdim" => base_ext::fn_shiftdim(args, nargout),
        "pagetranspose" => base_ext::fn_pagetranspose(args, false),
        "pagectranspose" => base_ext::fn_pagetranspose(args, true),
        "pagemtimes" => base_ext::fn_pagemtimes(args),
        "cummin" => base_ext::fn_cum_extrema(args, false),
        "cummax" => base_ext::fn_cum_extrema(args, true),
        "bounds" => base_ext::fn_bounds(args, nargout),
        "strcmpi" | "strncmp" | "strncmpi" => base_ext::fn_str_cmp_ext(name, args),
        "strtrim" | "deblank" => base_ext::fn_strtrim_deblank(name, args),
        "str2num" => fn_str2double(args),
        "int2str" => base_ext::fn_int2str(args),
        "mat2str" => base_ext::fn_mat2str(args),
        "dec2bin" | "dec2hex" | "bin2dec" | "hex2dec" | "blanks" => {
            base_ext::fn_base_conv(name, args)
        }
        "rmfield" => base_ext::fn_rmfield(args),
        "orderfields" => base_ext::fn_orderfields(args),
        "getfield" => base_ext::fn_getfield(args),
        "setfield" => base_ext::fn_setfield(args),
        "struct2cell" => base_ext::fn_struct2cell(args),
        "cell2struct" => base_ext::fn_cell2struct(args),
        "cell2mat" => base_ext::fn_cell2mat(args),
        "mat2cell" => base_ext::fn_mat2cell(args),
        "num2cell" => base_ext::fn_num2cell(args),
        "deal" => base_ext::fn_deal(args, nargout),

        // Functional operator builtins (useful with bsxfun / @plus / etc.)
        "plus" => tables::fn_binop_c("plus", args, |a, b| a + b),
        "minus" => tables::fn_binop_c("minus", args, |a, b| a - b),
        "times" => tables::fn_binop_c("times", args, |a, b| a * b),
        "rdivide" => tables::fn_binop_c("rdivide", args, |a, b| a / b),
        "ldivide" => tables::fn_binop_c("ldivide", args, |a, b| b / a),
        "power" => tables::fn_binop_c("power", args, Complex::powc),
        "mtimes" | "mrdivide" | "mldivide" | "mpower" => linalg_ext::fn_matrix_op(name, args),
        "eq" => tables::fn_binop_cmp("eq", args, |a, b| a == b),
        "ne" => tables::fn_binop_cmp("ne", args, |a, b| a != b),
        "lt" => tables::fn_binop_cmp("lt", args, |a, b| a < b),
        "le" => tables::fn_binop_cmp("le", args, |a, b| a <= b),
        "gt" => tables::fn_binop_cmp("gt", args, |a, b| a > b),
        "ge" => tables::fn_binop_cmp("ge", args, |a, b| a >= b),
        "and" => tables::fn_binop_cmp("and", args, |a, b| a != 0.0 && b != 0.0),
        "or" => tables::fn_binop_cmp("or", args, |a, b| a != 0.0 || b != 0.0),
        "xor" => tables::fn_binop_cmp("xor", args, |a, b| (a != 0.0) ^ (b != 0.0)),
        "not" => fn_pred_mat("not", args, |c| c.is_zero()),

        // Plotting & Handle Graphics
        "plot" => fn_plot(PlotKind::Line, args, nargout, ctx.figure, false, false),
        "semilogx" => fn_plot(PlotKind::Line, args, nargout, ctx.figure, true, false),
        "semilogy" => fn_plot(PlotKind::Line, args, nargout, ctx.figure, false, true),
        "loglog" => fn_plot(PlotKind::Line, args, nargout, ctx.figure, true, true),
        "scatter" => fn_plot(PlotKind::Scatter, args, nargout, ctx.figure, false, false),
        "bar" => fn_plot(PlotKind::Bar, args, nargout, ctx.figure, false, false),
        "stem" => fn_plot(PlotKind::Stem, args, nargout, ctx.figure, false, false),
        "errorbar" => fn_errorbar(args, nargout, ctx.figure),
        "fill" | "patch" => fn_fill_plot(name, args, nargout, ctx.figure),
        "hist" => fn_hist(args, nargout, ctx.figure),
        "spy" => fn_spy(args, ctx.figure),
        "histogram" => extras::fn_histogram(args, ctx.figure),
        "contour" => extras::fn_contour(args, ctx.figure),
        "imagesc" => fn_imagesc(args, ctx.figure),
        "quiver" => fn_quiver(args, ctx.figure),
        "plot3" => fn_plot3(args, ctx.figure),
        "surf" => fn_surface("surf", args, ctx.figure, false),
        "mesh" => fn_surface("mesh", args, ctx.figure, true),
        "view" => fn_view(args, ctx.figure),
        "subplot" => fn_subplot(args, ctx.figure),
        "legend" => fn_legend(args, ctx.figure),
        "xlim" => fn_axis_lim("xlim", args, &mut ctx.figure.panel_mut().xlim),
        "ylim" => fn_axis_lim("ylim", args, &mut ctx.figure.panel_mut().ylim),
        "zlim" => fn_axis_lim("zlim", args, &mut ctx.figure.panel_mut().zlim),
        "axis" => fn_axis(args, nargout, ctx.figure),
        "colorbar" => fn_colorbar(args, ctx.figure),
        "colormap" => fn_colormap(args, nargout, ctx.figure),
        "parula" | "viridis" | "jet" | "turbo" | "gray" | "hot" | "cool" | "bone" => {
            fn_colormap_palette(name, args)
        }
        "gcf" => fs_media::fn_gcf(args),
        "gca" => fs_media::fn_gca(args, ctx.figure),
        "set" => fs_media::fn_set(args, ctx.figure),
        "get" => fs_media::fn_get(args, ctx.figure),
        "title" => fn_plot_label("title", args, |s| ctx.figure.panel_mut().title = s),
        "xlabel" => fn_plot_label("xlabel", args, |s| ctx.figure.panel_mut().xlabel = s),
        "ylabel" => fn_plot_label("ylabel", args, |s| ctx.figure.panel_mut().ylabel = s),
        "zlabel" => fn_plot_label("zlabel", args, |s| ctx.figure.panel_mut().zlabel = s),
        "grid" => fn_plot_toggle("grid", args, &mut ctx.figure.panel_mut().grid),
        "hold" => fn_plot_toggle("hold", args, &mut ctx.figure.panel_mut().hold),
        "clf" => {
            ctx.figure.clear_current();
            Ok(vec![])
        }
        "figure" => {
            let target = args
                .first()
                .and_then(|v| v.as_scalar().ok())
                .map(|s| s.max(1.0).round() as usize);
            let id = ctx.figure.switch_figure(target);
            if nargout >= 1 {
                ok1(Value::scalar(id as f64))
            } else {
                Ok(vec![])
            }
        }
        "close" => {
            if let Some(a0) = args.first() {
                if let Ok(s) = a0.as_str() {
                    if s.eq_ignore_ascii_case("all") {
                        ctx.figure.close_figure(true, None);
                    } else {
                        ctx.figure.close_figure(false, None);
                    }
                } else if let Ok(n) = a0.as_scalar() {
                    ctx.figure.close_figure(false, Some(n.max(1.0).round() as usize));
                } else {
                    ctx.figure.close_figure(false, None);
                }
            } else {
                ctx.figure.close_figure(false, None);
            }
            Ok(vec![])
        }

        // Filesystem utilities & pure-Rust Media I/O (BMP, PNG, WAV)
        "fullfile" => fs_media::fn_fullfile(args),
        "fileparts" => fs_media::fn_fileparts(args, nargout),
        "isfile" => fs_media::fn_isfile(args, ctx.loader),
        "isfolder" => fs_media::fn_isfolder(args, ctx.loader),
        "dir" => fs_media::fn_dir(args, nargout, ctx.loader, ctx.stdout),
        "ls" => fs_media::fn_ls(args, nargout, ctx.loader, ctx.stdout),
        "pwd" => fs_media::fn_pwd(ctx.loader),
        "cd" => fs_media::fn_cd(args, ctx.loader),
        "mkdir" => fs_media::fn_mkdir(args, ctx.loader),
        "rmdir" => fs_media::fn_rmdir(args, ctx.loader),
        "delete" => fs_media::fn_delete(args, ctx.loader),
        "copyfile" => fs_media::fn_copyfile(args, ctx.loader),
        "movefile" => fs_media::fn_movefile(args, ctx.loader),
        "imread" => fs_media::fn_imread(args, ctx.loader),
        "imwrite" => fs_media::fn_imwrite(args, ctx.loader),
        "audioread" => fs_media::fn_audioread(args, nargout, ctx.loader),
        "audiowrite" => fs_media::fn_audiowrite(args, ctx.loader),

        // Image Processing Toolbox
        "rgb2gray" => image_proc::fn_rgb2gray(args),
        "im2double" => image_proc::fn_im2double(args),
        "im2uint8" => image_proc::fn_im2uint8(args),
        "mat2gray" => image_proc::fn_mat2gray(args),
        "imcomplement" => image_proc::fn_imcomplement(args),
        "imresize" => image_proc::fn_imresize(args),
        "imfilter" => image_proc::fn_imfilter(args),
        "imgaussfilt" => image_proc::fn_imgaussfilt(args),
        "medfilt2" => image_proc::fn_medfilt2(args),
        "fspecial" => image_proc::fn_fspecial(args),
        "edge" => image_proc::fn_edge(args),
        "graythresh" => image_proc::fn_graythresh(args),
        "imbinarize" => image_proc::fn_imbinarize(args),
        "imhist" => image_proc::fn_imhist(args, nargout),
        "imadjust" => image_proc::fn_imadjust(args),
        "bwlabel" => image_proc::fn_bwlabel(args, nargout),
        "regionprops" => image_proc::fn_regionprops(args),
        "imerode" | "imdilate" | "imopen" | "imclose" => image_proc::fn_morph(name, args),

        _ => return None,
    };
    Some(res)
}

fn ok1(v: Value) -> Result<Vec<Value>> {
    Ok(vec![v])
}

fn as_usize(v: &Value, ctx: &str) -> Result<usize> {
    let s = v.as_scalar()?;
    if s < 0.0 || !s.is_finite() {
        return Err(MatroError::runtime(format!(
            "{ctx}: expected non-negative integer, got {s}"
        )));
    }
    Ok(s.round() as usize)
}

fn is_matlab_err_id(s: &str) -> bool {
    s.contains(':')
        && !s.contains('%')
        && !s.contains(char::is_whitespace)
        && s.split(':').all(|part| {
            !part.is_empty()
                && part
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_')
        })
}

fn fn_error(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::runtime("Error raised"));
    }
    if args.len() >= 2 {
        if let Ok(id) = args[0].as_str() {
            if is_matlab_err_id(id) {
                let msg = format_printf(&args[1..])?;
                return Err(MatroError::custom(id, msg));
            }
        }
    }
    let msg = format_printf(args)?;
    Err(MatroError::runtime(msg))
}

fn fn_assert(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("assert", ">= 1", 0));
    }
    if !args[0].is_truthy() {
        let msg = if args.len() > 1 {
            format_printf(&args[1..])?
        } else {
            "Assertion failed".to_string()
        };
        return Err(MatroError::runtime(msg));
    }
    Ok(vec![])
}

/// Returns true if `name` is a registered Matro intrinsic function.
pub fn is_intrinsic(name: &str) -> bool {
    matches!(
        name,
        "pi" | "eps" | "inf" | "Inf" | "nan" | "NaN" | "realmax" | "realmin" | "intmax" | "intmin"
            | "true" | "false" | "disp" | "clc" | "format" | "pause" | "tic" | "toc" | "error" | "assert"
            | "warning" | "isreal" | "isempty" | "isnan" | "isinf" | "isfinite" | "issparse"
            | "isscalar" | "isvector" | "isrow" | "iscolumn" | "ismatrix" | "isnumeric"
            | "islogical" | "ischar" | "isstring" | "iscell" | "iscellstr" | "isstruct"
            | "isfloat" | "isinteger" | "istable" | "istimetable" | "iscategorical"
            | "isdatetime" | "isduration" | "isobject" | "ishandle" | "isa" | "class"
            | "find" | "any" | "all" | "complex" | "real" | "imag" | "conj" | "angle"
            | "zeros" | "ones" | "eye" | "rand" | "randn" | "randi" | "randperm" | "rng"
            | "linspace" | "logspace" | "meshgrid" | "ndgrid" | "diag" | "blkdiag" | "kron"
            | "magic" | "pascal" | "hilb" | "invhilb" | "vander" | "toeplitz" | "hankel"
            | "compan" | "gallery" | "cat" | "horzcat" | "vertcat" | "reshape" | "repmat"
            | "repelem" | "permute" | "ipermute" | "squeeze" | "shiftdim" | "flip" | "fliplr"
            | "flipud" | "rot90" | "circshift" | "tril" | "triu" | "size" | "length" | "numel"
            | "ndims" | "sub2ind" | "ind2sub" | "sparse" | "speye" | "spdiags" | "spones"
            | "sprand" | "sprandn" | "sprandsym" | "full" | "nnz" | "nonzeros" | "nzmax"
            | "spalloc" | "spfun" | "spy" | "sum" | "prod" | "mean" | "median" | "std" | "var"
            | "min" | "max" | "cumsum" | "cumprod" | "cummin" | "cummax" | "diff" | "sort"
            | "sortrows" | "issorted" | "unique" | "ismember" | "intersect" | "union"
            | "setdiff" | "setxor" | "abs" | "sqrt" | "exp" | "expm1" | "log" | "log1p"
            | "log2" | "log10" | "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "atan2"
            | "sinh" | "cosh" | "tanh" | "asinh" | "acosh" | "atanh" | "sind" | "cosd" | "tand"
            | "asind" | "acosd" | "atand" | "atan2d" | "sec" | "csc" | "cot" | "secd" | "cscd"
            | "cotd" | "asec" | "acsc" | "acot" | "asecd" | "acscd" | "acotd" | "sech" | "csch" | "coth" | "asech" | "acsch"
            | "acoth" | "floor" | "ceil" | "round" | "fix" | "sign" | "mod" | "rem" | "hypot"
            | "nthroot" | "pow2" | "nextpow2" | "sinc" | "deg2rad" | "rad2deg" | "wrapToPi"
            | "wrapTo2Pi" | "wrapTo180" | "wrapTo360" | "unwrap" | "inv" | "det" | "lu" | "ilu" | "qr"
            | "svd" | "eig" | "eigs" | "svds" | "chol" | "ichol" | "pinv" | "rank" | "cond" | "rcond"
            | "null" | "orth" | "rref" | "trace" | "dot" | "cross" | "norm" | "vecnorm"
            | "expm" | "logm" | "sqrtm" | "schur" | "hess" | "balance" | "issymmetric"
            | "ishermitian" | "bandwidth" | "fft" | "ifft" | "fft2" | "ifft2" | "fftn" | "ifftn"
            | "fftshift" | "ifftshift" | "conv" | "conv2" | "deconv" | "filter" | "filtfilt"
            | "xcorr" | "xcov" | "pwelch" | "periodogram" | "spectrogram" | "stft" | "hilbert"
            | "envelope" | "detrend" | "findpeaks" | "butter" | "cheby1" | "cheby2" | "fir1"
            | "freqz" | "grpdelay" | "decimate" | "resample" | "downsample" | "upsample"
            | "medfilt1" | "sgolayfilt" | "hann" | "hamming" | "blackman" | "kaiser"
            | "bartlett" | "flattopwin" | "tukeywin" | "gausswin" | "rectwin" | "triang"
            | "blackmanharris" | "zp2sos" | "sos2tf" | "tf2sos" | "sosfilt" | "polyfit"
            | "polyval" | "polyder" | "polyint" | "roots" | "poly" | "interp1" | "interp2"
            | "interp3" | "interpn" | "pchip" | "spline" | "makima" | "mkpp" | "ppval"
            | "unmkpp" | "griddedInterpolant" | "trapz" | "cumtrapz" | "gradient" | "del2"
            | "histcounts" | "histc" | "discretize" | "accumarray" | "movmean" | "movsum"
            | "movstd" | "movvar" | "movmedian" | "movmin" | "movmax" | "smoothdata"
            | "normalize" | "rescale" | "clip" | "bounds" | "rms" | "mink" | "maxk"
            | "topkrows" | "pagemtimes" | "pagetranspose" | "islocalmin" | "islocalmax"
            | "cov" | "corrcoef" | "corr" | "prctile" | "quantile" | "iqr" | "mad" | "zscore"
            | "skewness" | "kurtosis" | "geomean" | "harmmean" | "trimmean" | "mode"
            | "nansum" | "nanmean" | "nanstd" | "nanvar" | "nanmedian" | "normpdf" | "normcdf"
            | "norminv" | "normrnd" | "mvnpdf" | "mvnrnd" | "lognpdf" | "logncdf" | "logninv"
            | "lognrnd" | "gampdf" | "gamcdf" | "gaminv" | "gamrnd" | "betapdf" | "betacdf"
            | "betainv" | "betarnd" | "unifpdf" | "unifcdf" | "unifinv" | "unifrnd"
            | "exppdf" | "expcdf" | "expinv" | "exprnd" | "tpdf" | "tcdf" | "tinv" | "trnd"
            | "chi2pdf" | "chi2cdf" | "chi2inv" | "chi2rnd" | "fpdf" | "fcdf" | "finv" | "frnd"
            | "poisspdf" | "poisscdf" | "poissinv" | "poissrnd" | "binopdf" | "binocdf"
            | "binoinv" | "binornd" | "gamma" | "gammaln" | "gammainc" | "beta" | "betaln"
            | "betainc" | "erf" | "erfc" | "erfinv" | "erfcinv" | "besselj" | "bessely"
            | "besseli" | "besselk" | "legendre" | "ellipke" | "factorial" | "nchoosek"
            | "perms" | "gcd" | "lcm" | "primes" | "isprime" | "factor" | "pca" | "kmeans"
            | "fitlm" | "regress" | "anova1" | "ttest" | "ttest2" | "kstest" | "kstest2"
            | "ranksum" | "signrank" | "pdist" | "pdist2" | "squareform" | "linkage"
            | "cluster" | "knnsearch" | "randsample" | "datasample" | "sprintf" | "fprintf"
            | "num2str" | "str2double" | "str2num" | "mat2str" | "strcmp" | "strcmpi"
            | "strncmp" | "strncmpi" | "strfind" | "strrep" | "replace" | "contains"
            | "startsWith" | "endsWith" | "strsplit" | "strjoin" | "strcat" | "strtrim"
            | "deblank" | "strip" | "erase" | "upper" | "lower" | "regexp" | "regexpi"
            | "regexprep" | "split" | "join" | "string" | "char" | "double" | "single"
            | "logical" | "int8" | "int16" | "int32" | "int64" | "uint8" | "uint16" | "uint32"
            | "uint64" | "cast" | "cell" | "struct" | "fieldnames" | "isfield" | "rmfield"
            | "orderfields" | "getfield" | "setfield" | "deal" | "cell2mat" | "mat2cell"
            | "num2cell" | "struct2cell" | "cell2struct" | "table" | "timetable" | "datetime"
            | "addvars" | "movevars" | "removevars" | "renamevars" | "array2table" | "table2array"
            | "table2struct" | "struct2table" | "cell2table" | "table2cell" | "innerjoin"
            | "outerjoin" | "groupsummary" | "head" | "tail" | "height" | "width"
            | "summary" | "duration" | "categorical" | "readtable" | "writetable"
            | "readmatrix"
            | "writematrix" | "csvread" | "csvwrite" | "dlmread" | "dlmwrite" | "importdata"
            | "jsonencode" | "jsondecode" | "linprog" | "quadprog" | "lsqlin" | "lsqnonneg"
            | "optimset" | "optimoptions" | "tf" | "ss" | "bode" | "nyquist" | "margin"
            | "step" | "impulse" | "lsim" | "initial" | "stepinfo" | "rlocus" | "pzmap"
            | "lqr" | "dlqr" | "lqe" | "care" | "dare" | "lyap" | "dlyap" | "place" | "acker"
            | "ctrb" | "obsv" | "c2d" | "d2c" | "ssdata" | "tfdata" | "ss2tf" | "tf2ss"
            | "imfilter" | "medfilt2" | "imgaussfilt" | "imresize" | "rgb2gray" | "edge"
            | "bwlabel" | "regionprops" | "fspecial" | "imerode" | "imdilate" | "imopen"
            | "imclose" | "mat2gray" | "imcomplement" | "imbinarize" | "imadjust" | "graythresh"
            | "imhist" | "im2double" | "im2uint8" | "imread" | "imwrite" | "audioread"
            | "audiowrite" | "plot" | "plot3" | "scatter" | "bar" | "stem" | "errorbar" | "fill" | "patch" | "hist" | "histogram"
            | "contour" | "surf" | "mesh" | "quiver" | "imagesc" | "figure" | "subplot"
            | "hold" | "grid" | "title" | "xlabel" | "ylabel" | "zlabel" | "legend" | "xlim"
            | "ylim" | "zlim" | "colorbar" | "colormap" | "axis" | "clf" | "close" | "gca" | "gcf"
            | "parula" | "viridis" | "jet" | "turbo" | "gray" | "hot" | "cool" | "bone"
            | "get" | "set" | "dir" | "pwd" | "cd" | "mkdir" | "rmdir" | "delete" | "copyfile"
            | "movefile" | "isfile" | "isfolder" | "fullfile" | "fileparts" | "plus" | "minus"
            | "times" | "rdivide" | "ldivide" | "power" | "mtimes" | "mrdivide" | "mldivide"
            | "mpower" | "eq" | "ne" | "lt" | "le" | "gt" | "ge" | "and" | "or" | "xor" | "not"
            | "bitand" | "bitor" | "bitxor" | "bitshift" | "bitget" | "bitset" | "bitcmp"
            | "dec2bin" | "bin2dec" | "dec2hex" | "hex2dec" | "cart2pol" | "pol2cart"
            | "cart2sph" | "sph2cart"
    )
}

fn fn_isreal(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("isreal", "1", args.len()));
    }
    let real = match &args[0] {
        Value::Matrix(m) => !m.is_complex(),
        Value::Sparse(s) => !s.is_complex(),
        _ => true,
    };
    ok1(Value::bool(real))
}

fn fn_isempty(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("isempty", "1", args.len()));
    }
    let empty = match &args[0] {
        Value::Matrix(m) => m.is_empty(),
        Value::Sparse(s) => s.rows() == 0 || s.cols() == 0,
        Value::String(s) => s.is_empty(),
        Value::Cell(c) => c.data.is_empty(),
        Value::Struct(_) => false,
        Value::StructArray(sa) => sa.items.is_empty(),
        Value::Table(t) => t.nrows == 0 || t.columns.is_empty(),
        Value::Map(m) => m.entries.is_empty(),
        Value::Object(_) | Value::Handle(_) | Value::Closure(_) => false,
    };
    ok1(Value::bool(empty))
}

fn fn_pred_mat(name: &str, args: &[Value], pred: impl Fn(Complex) -> bool) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity(name, "1", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let out = m
        .map_c_to_real(|c| if pred(c) { 1.0 } else { 0.0 })
        .with_logical(true);
    ok1(Value::matrix(out))
}

fn fn_eps(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() {
        return ok1(Value::scalar(f64::EPSILON));
    }
    if let Some(s) = value_as_text(&args[0]) {
        if s.eq_ignore_ascii_case("single") {
            return ok1(Value::scalar(f32::EPSILON as f64));
        }
        return ok1(Value::scalar(f64::EPSILON));
    }
    let m = args[0].to_dense_matrix()?;
    let out = m.map(|x| {
        let ax = x.abs();
        if !ax.is_finite() {
            f64::NAN
        } else if ax <= f64::MIN_POSITIVE {
            f64::from_bits(1)
        } else {
            let bits = ax.to_bits();
            f64::from_bits(bits + 1) - ax
        }
    });
    ok1(Value::matrix(out))
}

fn fn_find(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity("find", "1 to 3 (X [, k [, 'first'|'last']])", args.len()));
    }
    let k_limit = if args.len() >= 2 {
        Some(as_usize(&args[1], "find")?)
    } else {
        None
    };
    let take_last = if args.len() == 3 {
        let s = value_as_text(&args[2])
            .ok_or_else(|| MatroError::runtime("find: direction must be 'first' or 'last'"))?;
        if s.eq_ignore_ascii_case("last") {
            true
        } else if s.eq_ignore_ascii_case("first") {
            false
        } else {
            return Err(MatroError::runtime(format!("find: unknown direction '{s}'")));
        }
    } else {
        false
    };

    let (nrows, is_row_vec, mut entries): (usize, bool, Vec<(usize, usize, Complex)>) =
        match &args[0] {
            Value::Sparse(s) => {
                let trips = s
                    .triplets()
                    .into_iter()
                    .map(|(r, c, v)| (r, c, Complex::from_real(v)))
                    .collect();
                (s.rows(), s.rows() == 1 && s.cols() > 1, trips)
            }
            other => {
                let m = other.to_dense_matrix()?;
                let nrows = m.rows();
                let mut e = Vec::new();
                if nrows > 0 {
                    for lin in 0..m.len() {
                        let v = m.get_linear_c(lin);
                        if !v.is_zero() {
                            let r = lin % nrows;
                            let c = lin / nrows;
                            e.push((r, c, v));
                        }
                    }
                }
                (nrows, m.ndims() == 2 && m.rows() == 1 && m.cols() > 1, e)
            }
        };

    if let Some(k) = k_limit {
        if entries.len() > k {
            if take_last {
                let start = entries.len() - k;
                entries = entries.split_off(start);
            } else {
                entries.truncate(k);
            }
        }
    }

    if nargout >= 2 {
        let rows: Vec<f64> = entries.iter().map(|&(r, _, _)| (r + 1) as f64).collect();
        let cols: Vec<f64> = entries.iter().map(|&(_, c, _)| (c + 1) as f64).collect();
        let r_mat = if is_row_vec {
            Matrix::row_vec(rows)
        } else {
            Matrix::col_vec(rows)
        };
        let c_mat = if is_row_vec {
            Matrix::row_vec(cols)
        } else {
            Matrix::col_vec(cols)
        };
        let mut out = vec![Value::matrix(r_mat), Value::matrix(c_mat)];
        if nargout >= 3 {
            let vals: Vec<Complex> = entries.iter().map(|&(_, _, v)| v).collect();
            let v_mat = if is_row_vec {
                Matrix::complex_row_vec(vals)
            } else {
                Matrix::complex_col_vec(vals)
            };
            out.push(Value::matrix(v_mat.normalize()));
        }
        return Ok(out);
    }

    let idxs: Vec<f64> = entries
        .iter()
        .map(|&(r, c, _)| (c * nrows + r + 1) as f64)
        .collect();
    let out = if is_row_vec {
        Matrix::row_vec(idxs)
    } else {
        Matrix::col_vec(idxs)
    };
    ok1(Value::matrix(out))
}

fn parse_vecdim(val: &Value, ctx: &str) -> Result<Vec<usize>> {
    let m = val.to_dense_matrix()?;
    let mut dims = Vec::with_capacity(m.len());
    for &x in m.data() {
        let d = x.round() as isize;
        if !x.is_finite() || (x - d as f64).abs() > 1e-12 || d < 1 {
            return Err(MatroError::runtime(format!(
                "{ctx}: dimension must be positive integer, got {x}"
            )));
        }
        let du = d as usize;
        if !dims.contains(&du) {
            dims.push(du);
        }
    }
    Ok(dims)
}

fn fn_any_all(name: &str, args: &[Value], is_all: bool) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity(name, "1 or 2 (A [, dim | 'all'])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    // MATLAB ignores NaN in any/all (NaN is neither true nor false; all([1 NaN]) == true, any([0 NaN]) == false)
    let is_nan_c = |c: Complex| c.re.is_nan() || c.im.is_nan();
    let is_nonzero = |c: Complex| !c.is_zero() && !is_nan_c(c);
    let reduce_slice = |slice: &[Complex]| -> bool {
        if is_all {
            slice.iter().all(|&c| is_nan_c(c) || !c.is_zero())
        } else {
            slice.iter().any(|&c| is_nonzero(c))
        }
    };
    let all_mode = args
        .get(1)
        .and_then(value_as_text)
        .is_some_and(|s| s.eq_ignore_ascii_case("all"));
    let m_dims = m.dims();
    let is_vec = m.ndims() == 2 && (m.rows() <= 1 || m.cols() <= 1);
    if all_mode || (args.len() == 1 && is_vec) {
        let vals: Vec<Complex> = (0..m.len()).map(|i| m.get_linear_c(i)).collect();
        return ok1(Value::bool(reduce_slice(&vals)));
    }
    let target_dims = if let Some(a) = args.get(1) {
        parse_vecdim(a, name)?
    } else {
        vec![m_dims.iter().position(|&d| d != 1).map_or(1, |i| i + 1)]
    };
    let mut cur = m;
    for dim in target_dims {
        let cur_dims = cur.dims();
        let ax = dim - 1;
        if ax >= cur_dims.len() {
            let out = cur.map_c_to_real(|c| {
                let b = if is_all { is_nan_c(c) || !c.is_zero() } else { is_nonzero(c) };
                if b { 1.0 } else { 0.0 }
            });
            cur = out.with_logical(true);
            continue;
        }
        let inner: usize = cur_dims[..ax].iter().product();
        let axis_len = cur_dims[ax];
        let outer: usize = cur_dims[ax + 1..].iter().product();
        let mut out_dims = cur_dims.clone();
        out_dims[ax] = 1;
        let out_total = inner * outer;
        let mut out_data = Vec::with_capacity(out_total);
        let mut buf = Vec::with_capacity(axis_len);
        for o in 0..outer {
            let block = o * axis_len * inner;
            for i in 0..inner {
                buf.clear();
                for k in 0..axis_len {
                    buf.push(cur.get_linear_c(block + k * inner + i));
                }
                out_data.push(if reduce_slice(&buf) { 1.0 } else { 0.0 });
            }
        }
        cur = Matrix::from_nd_col_major(out_dims, out_data)?.with_logical(true);
    }
    ok1(Value::matrix(cur.with_logical(true)))
}

fn fn_complex(args: &[Value]) -> Result<Vec<Value>> {
    match args.len() {
        1 => {
            let a = args[0].as_matrix()?;
            ok1(Value::matrix(a.clone()))
        }
        2 => {
            let a = args[0].as_matrix()?;
            let b = args[1].as_matrix()?;
            let m = a.elementwise_c(b, |x, y| Complex::new(x.re, y.re))?;
            ok1(Value::matrix(m))
        }
        n => Err(MatroError::arity("complex", "1 or 2", n)),
    }
}

fn parse_class_name(s: &str) -> Option<NumericClass> {
    match s.trim().to_ascii_lowercase().as_str() {
        "double" => Some(NumericClass::Double),
        "single" => Some(NumericClass::Single),
        "logical" => Some(NumericClass::Logical),
        "int8" => Some(NumericClass::Int8),
        "int16" => Some(NumericClass::Int16),
        "int32" => Some(NumericClass::Int32),
        "int64" => Some(NumericClass::Int64),
        "uint8" => Some(NumericClass::Uint8),
        "uint16" => Some(NumericClass::Uint16),
        "uint32" => Some(NumericClass::Uint32),
        "uint64" => Some(NumericClass::Uint64),
        _ => None,
    }
}

/// Parses MATLAB array constructor args `(sz1, sz2, ..., [typename | 'like', prototype])`.
fn parse_dims_and_class(
    name: &str,
    args: &[Value],
    default_class: NumericClass,
) -> Result<(Vec<usize>, NumericClass, bool)> {
    let mut class = default_class;
    let mut make_sparse = false;
    let mut end = args.len();
    if end >= 2 && value_as_text(&args[end - 2]).is_some_and(|s| s.eq_ignore_ascii_case("like")) {
        match &args[end - 1] {
            Value::Sparse(_) => {
                make_sparse = true;
                class = NumericClass::Double;
            }
            Value::Matrix(m) => class = m.dtype(),
            _ => {}
        }
        end -= 2;
    } else if end >= 1 {
        if let Some(s) = value_as_text(&args[end - 1]) {
            if let Some(c) = parse_class_name(&s) {
                class = c;
                end -= 1;
            }
        }
    }
    let dim_args = &args[..end];
    let dims: Vec<usize> = match dim_args.len() {
        0 => vec![1, 1],
        1 => {
            let m = dim_args[0].to_dense_matrix()?;
            if m.len() >= 2 {
                m.data().iter().map(|&x| x.max(0.0).round() as usize).collect()
            } else if m.len() == 1 {
                let n = m.data()[0].max(0.0).round() as usize;
                vec![n, n]
            } else {
                vec![0, 0]
            }
        }
        _ => dim_args
            .iter()
            .map(|a| {
                let s = a.as_scalar()?;
                if !s.is_finite() {
                    return Err(MatroError::runtime(format!("{name}: size must be finite")));
                }
                Ok(s.max(0.0).round() as usize)
            })
            .collect::<Result<_>>()?,
    };
    Ok((dims, class, make_sparse))
}

fn finish_constructed_array(m: Matrix, class: NumericClass, make_sparse: bool) -> Result<Vec<Value>> {
    if make_sparse && m.ndims() == 2 {
        return ok1(Value::sparse(SparseMatrix::from_dense(&m)));
    }
    let out = if class == NumericClass::Logical {
        m.with_logical(true)
    } else if class != NumericClass::Double {
        m.cast_to(class)
    } else {
        m
    };
    ok1(Value::matrix(out))
}

fn fn_const_array(
    name: &str,
    args: &[Value],
    fill_val: f64,
    default_class: NumericClass,
) -> Result<Vec<Value>> {
    let (dims, class, make_sparse) = parse_dims_and_class(name, args, default_class)?;
    let total: usize = dims.iter().product();
    let m = Matrix::from_nd_col_major(dims, vec![fill_val; total])?;
    finish_constructed_array(m, class, make_sparse)
}

fn fn_fill(name: &str, args: &[Value], make: fn(usize, usize) -> Matrix) -> Result<Vec<Value>> {
    let (dims, class, make_sparse) = parse_dims_and_class(name, args, NumericClass::Double)?;
    if name == "eye" {
        if dims.len() > 2 {
            return Err(MatroError::dimension("eye: only supports 2D dimensions"));
        }
        let m = make(dims[0], dims[1]);
        return finish_constructed_array(m, class, make_sparse);
    }
    let m = if dims.len() == 2 {
        make(dims[0], dims[1])
    } else if name == "ones" {
        Matrix::ones_nd(dims)
    } else {
        Matrix::zeros_nd(dims)
    };
    finish_constructed_array(m, class, make_sparse)
}

fn fn_rand(args: &[Value], rng: &mut Rng, normal: bool) -> Result<Vec<Value>> {
    let name = if normal { "randn" } else { "rand" };
    let (dims, class, make_sparse) = parse_dims_and_class(name, args, NumericClass::Double)?;
    let total: usize = dims.iter().product();
    let mut data = Vec::with_capacity(total);
    for _ in 0..total {
        data.push(if normal {
            rng.next_normal()
        } else {
            rng.next_f64()
        });
    }
    let m = Matrix::from_nd_col_major(dims, data)?;
    finish_constructed_array(m, class, make_sparse)
}

fn rng_state_struct(rng: &Rng) -> Value {
    let (gen_type, seed_val, state) = rng.state_snapshot();
    let mut map = BTreeMap::new();
    map.insert("Type".to_string(), Value::String(gen_type.to_string()));
    map.insert("Seed".to_string(), Value::scalar(seed_val as f64));
    let lo = (state & 0xffff_ffff) as f64;
    let hi = ((state >> 32) & 0xffff_ffff) as f64;
    map.insert(
        "State".to_string(),
        Value::matrix(Matrix::col_vec(vec![lo, hi])),
    );
    Value::Struct(map)
}

fn value_as_text(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Object(o) if o.class_name == "string" => {
            if let Some(Value::Cell(c)) = o.fields.get("Data") {
                c.data.first().and_then(|x| x.as_str().ok()).map(String::from)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn fn_rng(args: &[Value], nargout: usize, rng: &mut Rng) -> Result<Vec<Value>> {
    let prev = rng_state_struct(rng);
    match args.len() {
        0 => ok1(prev),
        1 => {
            match &args[0] {
                Value::Struct(map) => {
                    let gen_raw = map
                        .get("Type")
                        .and_then(value_as_text)
                        .unwrap_or_else(|| "twister".to_string());
                    let gen_name = Rng::canonical_generator(&gen_raw).ok_or_else(|| {
                        MatroError::runtime(format!("rng: unknown generator '{gen_raw}'"))
                    })?;
                    let seed_val = map
                        .get("Seed")
                        .and_then(|v| v.as_scalar().ok())
                        .unwrap_or(0.0)
                        .max(0.0)
                        .round() as u64;
                    if let Some(st_val) = map.get("State") {
                        let m = st_val.as_matrix()?;
                        if m.len() >= 2 {
                            let lo = m.data()[0].max(0.0).round() as u64 & 0xffff_ffff;
                            let hi = m.data()[1].max(0.0).round() as u64 & 0xffff_ffff;
                            rng.restore_state(gen_name.to_string(), seed_val, (hi << 32) | lo);
                        } else if m.len() == 1 {
                            rng.restore_state(
                                gen_name.to_string(),
                                seed_val,
                                m.data()[0].max(0.0).round() as u64,
                            );
                        } else {
                            rng.seed_with_generator(seed_val, gen_name);
                        }
                    } else {
                        rng.seed_with_generator(seed_val, gen_name);
                    }
                }
                other => {
                    if let Some(txt) = value_as_text(other) {
                        match txt.trim().to_ascii_lowercase().as_str() {
                            "default" => rng.reset_default(),
                            "shuffle" => rng.shuffle(None),
                            _ => {
                                return Err(MatroError::runtime(format!(
                                    "rng: invalid option '{txt}' (expected 'default', 'shuffle', seed, or state struct)"
                                )));
                            }
                        }
                    } else {
                        let s = other.as_scalar()?;
                        if !s.is_finite() || s < 0.0 {
                            return Err(MatroError::runtime(
                                "rng: seed must be a non-negative integer",
                            ));
                        }
                        rng.seed(s.round() as u64);
                    }
                }
            }
            if nargout > 0 { ok1(prev) } else { Ok(vec![]) }
        }
        2 => {
            let gen_txt = value_as_text(&args[1]).ok_or_else(|| {
                MatroError::runtime("rng: second argument must be a generator name string")
            })?;
            let gen_name = Rng::canonical_generator(&gen_txt).ok_or_else(|| {
                MatroError::runtime(format!("rng: unknown generator '{gen_txt}'"))
            })?;
            if let Some(seed_txt) = value_as_text(&args[0]) {
                match seed_txt.trim().to_ascii_lowercase().as_str() {
                    "shuffle" => rng.shuffle(Some(gen_name)),
                    "default" => rng.seed_with_generator(0, gen_name),
                    _ => {
                        return Err(MatroError::runtime(format!(
                            "rng: invalid seed option '{seed_txt}'"
                        )));
                    }
                }
            } else {
                let s = args[0].as_scalar()?;
                if !s.is_finite() || s < 0.0 {
                    return Err(MatroError::runtime(
                        "rng: seed must be a non-negative integer",
                    ));
                }
                rng.seed_with_generator(s.round() as u64, gen_name);
            }
            if nargout > 0 { ok1(prev) } else { Ok(vec![]) }
        }
        n => Err(MatroError::arity("rng", "0, 1, or 2", n)),
    }
}

fn fn_meshgrid(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity("meshgrid", "1, 2, or 3", args.len()));
    }
    let xm = args[0].to_dense_matrix()?;
    let ym = if args.len() >= 2 {
        args[1].to_dense_matrix()?
    } else {
        xm.clone()
    };
    let want_3d = args.len() == 3 || nargout >= 3;
    if !want_3d {
        let rows = ym.len();
        let cols = xm.len();
        let mut x_mat = Matrix::zeros(rows, cols);
        let mut y_mat = Matrix::zeros(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                x_mat.set_c(r, c, xm.get_linear_c(c));
                y_mat.set_c(r, c, ym.get_linear_c(r));
            }
        }
        return Ok(vec![
            Value::matrix(x_mat.normalize()),
            Value::matrix(y_mat.normalize()),
        ]);
    }
    let zm = if args.len() == 3 {
        args[2].to_dense_matrix()?
    } else {
        xm.clone()
    };
    let nx = xm.len();
    let ny = ym.len();
    let nz = zm.len();
    let dims = vec![ny, nx, nz];
    let total = ny * nx * nz;
    let mut x_re = vec![0.0; total];
    let mut x_im = vec![0.0; total];
    let mut y_re = vec![0.0; total];
    let mut y_im = vec![0.0; total];
    let mut z_re = vec![0.0; total];
    let mut z_im = vec![0.0; total];
    for iz in 0..nz {
        let zv = zm.get_linear_c(iz);
        for ix in 0..nx {
            let xv = xm.get_linear_c(ix);
            for iy in 0..ny {
                let yv = ym.get_linear_c(iy);
                let pos = iz * (ny * nx) + ix * ny + iy;
                x_re[pos] = xv.re;
                x_im[pos] = xv.im;
                y_re[pos] = yv.re;
                y_im[pos] = yv.im;
                z_re[pos] = zv.re;
                z_im[pos] = zv.im;
            }
        }
    }
    Ok(vec![
        Value::matrix(Matrix::from_nd_complex_col_major(dims.clone(), x_re, x_im)?.normalize()),
        Value::matrix(Matrix::from_nd_complex_col_major(dims.clone(), y_re, y_im)?.normalize()),
        Value::matrix(Matrix::from_nd_complex_col_major(dims, z_re, z_im)?.normalize()),
    ])
}

fn fn_cell(args: &[Value]) -> Result<Vec<Value>> {
    match args.len() {
        0 => ok1(Value::Cell(CellArray::new_empty_cells(0, 0))),
        1 => {
            let m = args[0].to_dense_matrix()?;
            if m.len() == 1 {
                let n = as_usize(&args[0], "cell")?;
                ok1(Value::Cell(CellArray::new_empty_cells(n, n)))
            } else {
                let mut dims = Vec::with_capacity(m.len());
                for &x in m.data() {
                    if x < 0.0 || !x.is_finite() {
                        return Err(MatroError::runtime("cell: dimensions must be non-negative integers"));
                    }
                    dims.push(x.round() as usize);
                }
                ok1(Value::Cell(CellArray::new_empty_nd(dims)))
            }
        }
        2 => {
            let r = as_usize(&args[0], "cell")?;
            let c = as_usize(&args[1], "cell")?;
            ok1(Value::Cell(CellArray::new_empty_cells(r, c)))
        }
        _ => {
            let mut dims = Vec::with_capacity(args.len());
            for a in args {
                dims.push(as_usize(a, "cell")?);
            }
            ok1(Value::Cell(CellArray::new_empty_nd(dims)))
        }
    }
}

fn fn_struct(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() % 2 != 0 {
        return Err(MatroError::arity(
            "struct",
            "even number of key/value args",
            args.len(),
        ));
    }
    let mut map = BTreeMap::new();
    for pair in args.chunks_exact(2) {
        let key = pair[0].as_str()?.to_string();
        map.insert(key, pair[1].clone());
    }
    ok1(Value::Struct(map))
}

fn fn_fieldnames(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("fieldnames", "1", args.len()));
    }
    match &args[0] {
        Value::Struct(map) => {
            let rows: Vec<Vec<Value>> = map
                .keys()
                .map(|k| vec![Value::String(k.clone())])
                .collect();
            ok1(Value::Cell(CellArray::from_rows(rows)?))
        }
        Value::StructArray(sa) => {
            let rows: Vec<Vec<Value>> = sa
                .items
                .first()
                .map(|m| m.keys().map(|k| vec![Value::String(k.clone())]).collect())
                .unwrap_or_default();
            ok1(Value::Cell(CellArray::from_rows(rows)?))
        }
        Value::Object(o) => {
            let rows: Vec<Vec<Value>> = o
                .fields
                .keys()
                .map(|k| vec![Value::String(k.clone())])
                .collect();
            ok1(Value::Cell(CellArray::from_rows(rows)?))
        }
        other => Err(MatroError::runtime(format!(
            "fieldnames: expected struct or object, got {}",
            other.type_name()
        ))),
    }
}

fn fn_isfield(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("isfield", "2", args.len()));
    }
    let has_field = |f: &str| -> bool {
        match &args[0] {
            Value::Struct(map) => map.contains_key(f),
            Value::StructArray(sa) => sa.items.first().is_some_and(|m| m.contains_key(f)),
            Value::Object(o) => o.fields.contains_key(f),
            _ => false,
        }
    };
    if let Some(field) = value_as_text(&args[1]) {
        return ok1(Value::bool(has_field(&field)));
    }
    if let Value::Cell(c) = &args[1] {
        let mut out = Matrix::zeros(c.rows, c.cols);
        for col in 0..c.cols {
            for row in 0..c.rows {
                let b = value_as_text(c.get(row, col)).is_some_and(|f| has_field(&f));
                out.set(row, col, if b { 1.0 } else { 0.0 });
            }
        }
        return ok1(Value::matrix(out.with_logical(true)));
    }
    if let Value::Object(o) = &args[1] {
        if o.class_name == "string" {
            if let Some(Value::Cell(c)) = o.fields.get("Data") {
                let mut out = Matrix::zeros(c.rows, c.cols);
                for col in 0..c.cols {
                    for row in 0..c.rows {
                        let b = value_as_text(c.get(row, col)).is_some_and(|f| has_field(&f));
                        out.set(row, col, if b { 1.0 } else { 0.0 });
                    }
                }
                return ok1(Value::matrix(out.with_logical(true)));
            }
        }
    }
    ok1(Value::bool(false))
}

fn fn_isa(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("isa", "2", args.len()));
    }
    let expected = value_as_text(&args[1])
        .ok_or_else(|| MatroError::runtime("isa: expected class name string"))?;
    let matched = match expected.as_str() {
        "numeric" => matches!(&args[0], Value::Matrix(m) if !m.is_logical())
            || matches!(&args[0], Value::Sparse(_)),
        "float" => matches!(
            &args[0],
            Value::Matrix(m) if matches!(m.dtype(), matro_types::NumericClass::Double | matro_types::NumericClass::Single)
        ) || matches!(&args[0], Value::Sparse(_)),
        "integer" => matches!(
            &args[0],
            Value::Matrix(m) if m.dtype().is_integer()
        ),
        "double" => matches!(
            &args[0],
            Value::Matrix(m) if m.dtype() == matro_types::NumericClass::Double
        ) || matches!(&args[0], Value::Sparse(_)),
        other => args[0].type_name() == other,
    };
    ok1(Value::bool(matched))
}

fn fn_sparse(args: &[Value]) -> Result<Vec<Value>> {
    match args.len() {
        1 => match &args[0] {
            Value::Sparse(s) => ok1(Value::Sparse(s.clone())),
            Value::Matrix(m) => ok1(Value::sparse(SparseMatrix::from_dense(m))),
            other => Err(MatroError::runtime(format!(
                "sparse: expected matrix, got {}",
                other.type_name()
            ))),
        },
        2 => {
            let m = as_usize(&args[0], "sparse")?;
            let n = as_usize(&args[1], "sparse")?;
            ok1(Value::sparse(SparseMatrix::zeros(m, n)))
        }
        3 | 5 | 6 => {
            let im = args[0].to_dense_matrix()?;
            let jm = args[1].to_dense_matrix()?;
            let vv = args[2].to_dense_matrix()?;
            let n_entries = im.len().max(jm.len()).max(vv.len());
            if im.len() != 1 && im.len() != n_entries {
                return Err(MatroError::dimension("sparse(i, j, v): i length mismatch"));
            }
            if jm.len() != 1 && jm.len() != n_entries {
                return Err(MatroError::dimension("sparse(i, j, v): j length mismatch"));
            }
            if vv.len() != 1 && vv.len() != n_entries {
                return Err(MatroError::dimension("sparse(i, j, v): v length mismatch"));
            }
            let iv = im.data();
            let jv = jm.data();
            let mut max_r = 0usize;
            let mut max_c = 0usize;
            let is_complex = vv.is_complex();
            let mut real_triplets = if is_complex { Vec::new() } else { Vec::with_capacity(n_entries) };
            let mut cplx_triplets = if is_complex { Vec::with_capacity(n_entries) } else { Vec::new() };
            for k in 0..n_entries {
                let r = (if iv.len() == 1 { iv[0] } else { iv[k] }).round() as isize;
                let c = (if jv.len() == 1 { jv[0] } else { jv[k] }).round() as isize;
                if r < 1 || c < 1 {
                    return Err(MatroError::index("sparse: indices must be >= 1"));
                }
                let r0 = (r - 1) as usize;
                let c0 = (c - 1) as usize;
                max_r = max_r.max(r0 + 1);
                max_c = max_c.max(c0 + 1);
                if is_complex {
                    let val = if vv.len() == 1 { vv.get_linear_c(0) } else { vv.get_linear_c(k) };
                    cplx_triplets.push((r0, c0, val));
                } else {
                    let val = if vv.len() == 1 { vv.data()[0] } else { vv.data()[k] };
                    real_triplets.push((r0, c0, val));
                }
            }
            let (rows, cols) = if args.len() >= 5 {
                (as_usize(&args[3], "sparse")?, as_usize(&args[4], "sparse")?)
            } else {
                (max_r, max_c)
            };
            if args.len() == 6 {
                let _nzmax = as_usize(&args[5], "sparse")?;
            }
            let sp = if is_complex {
                SparseMatrix::from_complex_triplets(rows, cols, &cplx_triplets)?
            } else {
                SparseMatrix::from_triplets(rows, cols, &real_triplets)?
            };
            ok1(Value::sparse(sp))
        }
        n => Err(MatroError::arity("sparse", "1, 2, 3, 5, or 6", n)),
    }
}

fn fn_speye(args: &[Value]) -> Result<Vec<Value>> {
    let (m, n) = match args.len() {
        0 => (1, 1),
        1 => {
            let mat = args[0].to_dense_matrix()?;
            if mat.len() >= 2 {
                (
                    mat.data()[0].max(0.0).round() as usize,
                    mat.data()[1].max(0.0).round() as usize,
                )
            } else {
                let k = as_usize(&args[0], "speye")?;
                (k, k)
            }
        }
        2 => (as_usize(&args[0], "speye")?, as_usize(&args[1], "speye")?),
        k => return Err(MatroError::arity("speye", "0, 1, or 2", k)),
    };
    let d = m.min(n);
    let triplets: Vec<(usize, usize, f64)> = (0..d).map(|i| (i, i, 1.0)).collect();
    ok1(Value::sparse(SparseMatrix::from_triplets(m, n, &triplets)?))
}

fn fn_sprand(args: &[Value], rng: &mut Rng, normal: bool, sym: bool) -> Result<Vec<Value>> {
    let name = if sym {
        "sprandsym"
    } else if normal {
        "sprandn"
    } else {
        "sprand"
    };
    if sym {
        if args.is_empty() || args.len() > 3 {
            return Err(MatroError::arity(name, "1..=3 (n, density [, rc])", args.len()));
        }
        if args.len() == 1 {
            let s = match &args[0] {
                Value::Sparse(sp) => (**sp).clone(),
                other => SparseMatrix::from_dense(&other.to_dense_matrix()?),
            };
            let n = s.rows().min(s.cols());
            let mut trips = Vec::new();
            for (r, c, _) in s.triplets() {
                if r <= c && r < n && c < n {
                    let v = rng.next_normal();
                    trips.push((r, c, v));
                    if r != c {
                        trips.push((c, r, v));
                    }
                }
            }
            return ok1(Value::sparse(SparseMatrix::from_triplets(n, n, &trips)?));
        }
        let n = as_usize(&args[0], name)?;
        let density = args[1].as_scalar()?.clamp(0.0, 1.0);
        let nnz_target = ((n * n) as f64 * density * 0.5).round() as usize;
        let mut trips = Vec::with_capacity(nnz_target * 2);
        for _ in 0..nnz_target {
            let r = (rng.next_f64() * (n as f64)).floor().min((n.saturating_sub(1)) as f64) as usize;
            let c = (rng.next_f64() * (n as f64)).floor().min((n.saturating_sub(1)) as f64) as usize;
            let v = rng.next_normal();
            trips.push((r, c, v));
            if r != c {
                trips.push((c, r, v));
            }
        }
        return ok1(Value::sparse(SparseMatrix::from_triplets(n, n, &trips)?));
    }
    if args.len() == 1 {
        let s = match &args[0] {
            Value::Sparse(sp) => (**sp).clone(),
            other => SparseMatrix::from_dense(&other.to_dense_matrix()?),
        };
        let trips: Vec<(usize, usize, f64)> = s
            .triplets()
            .into_iter()
            .map(|(r, c, _)| (r, c, if normal { rng.next_normal() } else { rng.next_f64() }))
            .collect();
        return ok1(Value::sparse(SparseMatrix::from_triplets(s.rows(), s.cols(), &trips)?));
    }
    if args.len() != 3 && args.len() != 4 {
        return Err(MatroError::arity(name, "1, 3, or 4 (m, n, density [, rc])", args.len()));
    }
    let m = as_usize(&args[0], name)?;
    let n = as_usize(&args[1], name)?;
    let density = args[2].as_scalar()?.clamp(0.0, 1.0);
    let nnz_target = ((m * n) as f64 * density).round() as usize;
    let mut trips = Vec::with_capacity(nnz_target);
    if m > 0 && n > 0 {
        for _ in 0..nnz_target {
            let r = (rng.next_f64() * (m as f64)).floor().min((m - 1) as f64) as usize;
            let c = (rng.next_f64() * (n as f64)).floor().min((n - 1) as f64) as usize;
            let v = if normal { rng.next_normal() } else { rng.next_f64() };
            trips.push((r, c, v));
        }
    }
    ok1(Value::sparse(SparseMatrix::from_triplets(m, n, &trips)?))
}

fn fn_full(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("full", "1", args.len()));
    }
    ok1(Value::matrix(args[0].to_dense_matrix()?))
}

fn fn_issparse(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("issparse", "1", args.len()));
    }
    ok1(Value::bool(matches!(&args[0], Value::Sparse(_))))
}

fn fn_nnz(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("nnz", "1", args.len()));
    }
    let count = match &args[0] {
        Value::Sparse(s) => s.nnz(),
        Value::Matrix(m) => (0..m.len()).filter(|&i| !m.get_linear_c(i).is_zero()).count(),
        other => {
            return Err(MatroError::runtime(format!(
                "nnz: expected matrix, got {}",
                other.type_name()
            )));
        }
    };
    ok1(Value::scalar(count as f64))
}

fn fn_pcg(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 4 {
        return Err(MatroError::arity("pcg", "2..=4 (A, b [, tol, maxit])", args.len()));
    }
    let b = args[1].as_matrix()?.data();
    let tol = args.get(2).map(|v| v.as_scalar()).transpose()?.unwrap_or(1e-8);
    let maxit = args
        .get(3)
        .map(|v| as_usize(v, "pcg"))
        .transpose()?
        .unwrap_or_else(|| (b.len() * 4).max(200));
    let x = match &args[0] {
        Value::Sparse(s) => s.solve_cg(b, tol, maxit)?,
        Value::Matrix(m) => SparseMatrix::from_dense(m).solve_cg(b, tol, maxit)?,
        other => {
            return Err(MatroError::runtime(format!(
                "pcg: expected sparse or dense matrix, got {}",
                other.type_name()
            )));
        }
    };
    ok1(Value::matrix(Matrix::col_vec(x)))
}

fn fn_strcmp(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("strcmp", "2", args.len()));
    }
    fn extract_str_arr(v: &Value) -> Option<(Vec<String>, usize, usize)> {
        match v {
            Value::String(s) => Some((vec![s.clone()], 1, 1)),
            Value::Cell(c) => {
                let mut out = Vec::with_capacity(c.data.len());
                for item in &c.data {
                    out.push(value_as_text(item).unwrap_or_default());
                }
                Some((out, c.rows, c.cols))
            }
            Value::Object(o) if o.class_name == "string" => {
                if let Some(Value::Cell(c)) = o.fields.get("Data") {
                    let mut out = Vec::with_capacity(c.data.len());
                    for item in &c.data {
                        out.push(value_as_text(item).unwrap_or_default());
                    }
                    return Some((out, c.rows, c.cols));
                }
                None
            }
            _ => None,
        }
    }
    let Some((sa, ra, ca)) = extract_str_arr(&args[0]) else {
        return ok1(Value::bool(false));
    };
    let Some((sb, rb, cb)) = extract_str_arr(&args[1]) else {
        return ok1(Value::bool(false));
    };
    if sa.len() == 1 && sb.len() == 1 {
        return ok1(Value::bool(sa[0] == sb[0]));
    }
    if sa.len() == 1 {
        let vals: Vec<f64> = sb.iter().map(|b| if sa[0] == *b { 1.0 } else { 0.0 }).collect();
        return ok1(Value::matrix(Matrix::from_col_major(rb, cb, vals)?.with_logical(true)));
    }
    if sb.len() == 1 {
        let vals: Vec<f64> = sa.iter().map(|a| if *a == sb[0] { 1.0 } else { 0.0 }).collect();
        return ok1(Value::matrix(Matrix::from_col_major(ra, ca, vals)?.with_logical(true)));
    }
    if ra == rb && ca == cb {
        let vals: Vec<f64> = sa
            .iter()
            .zip(sb.iter())
            .map(|(a, b)| if a == b { 1.0 } else { 0.0 })
            .collect();
        return ok1(Value::matrix(Matrix::from_col_major(ra, ca, vals)?.with_logical(true)));
    }
    Err(MatroError::dimension("strcmp: string arrays must have compatible sizes"))
}

fn fn_class(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("class", "1", args.len()));
    }
    ok1(Value::String(args[0].type_name().to_string()))
}

fn fn_sprintf(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("sprintf", ">= 1", 0));
    }
    ok1(Value::String(format_printf(args)?))
}

fn fn_num2str(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("num2str", "1 or 2 (A [, precision | formatSpec])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    if args.len() == 2 {
        if let Some(fmt_spec) = value_as_text(&args[1]) {
            let mut lines = Vec::with_capacity(m.rows());
            for r in 0..m.rows() {
                let mut row_vals = Vec::with_capacity(m.cols());
                for c in 0..m.cols() {
                    row_vals.push(Value::scalar(m.get(r, c)));
                }
                let mut call_a = vec![Value::String(fmt_spec.clone())];
                call_a.extend(row_vals);
                lines.push(format_printf(&call_a)?);
            }
            return ok1(Value::String(lines.join("\n")));
        }
        let prec = as_usize(&args[1], "num2str")?;
        let mut lines = Vec::with_capacity(m.rows());
        for r in 0..m.rows() {
            let parts: Vec<String> = (0..m.cols())
                .map(|c| format_g_value(m.get(r, c), prec, false))
                .collect();
            lines.push(parts.join("  "));
        }
        return ok1(Value::String(lines.join("\n")));
    }
    ok1(Value::String(format!("{}", args[0])))
}

fn fn_str2double(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("str2double", "1", args.len()));
    }
    let s = args[0].as_str()?.trim();
    let v = s.parse::<f64>().unwrap_or(f64::NAN);
    ok1(Value::scalar(v))
}

fn fn_strsplit(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("strsplit", "1 or 2", args.len()));
    }
    let s = args[0].as_str()?;
    let parts: Vec<Value> = if args.len() == 2 {
        let delim = args[1].as_str()?;
        s.split(delim).map(|p| Value::String(p.to_string())).collect()
    } else {
        s.split_whitespace()
            .map(|p| Value::String(p.to_string()))
            .collect()
    };
    ok1(Value::Cell(CellArray::from_rows(vec![parts])?))
}

fn fn_strjoin(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("strjoin", "1 or 2", args.len()));
    }
    let Value::Cell(cell) = &args[0] else {
        return Err(MatroError::runtime("strjoin: first argument must be a cell array"));
    };
    let sep = if args.len() == 2 {
        args[1].as_str()?
    } else {
        " "
    };
    let parts: Vec<String> = cell.data.iter().map(|v| format!("{v}")).collect();
    ok1(Value::String(parts.join(sep)))
}

fn fn_strcat(args: &[Value]) -> Result<Vec<Value>> {
    let mut out = String::new();
    for a in args {
        out.push_str(&format!("{a}"));
    }
    ok1(Value::String(out))
}

fn fn_str1(name: &str, args: &[Value], f: fn(&str) -> String) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity(name, "1", args.len()));
    }
    ok1(Value::String(f(args[0].as_str()?)))
}

fn fn_linspace(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity("linspace", "2 or 3", args.len()));
    }
    let start = args[0].as_scalar()?;
    let stop = args[1].as_scalar()?;
    let n = if args.len() == 3 {
        as_usize(&args[2], "linspace")?
    } else {
        100
    };
    if n == 0 {
        return ok1(Value::empty());
    }
    if n == 1 {
        return ok1(Value::scalar(stop));
    }
    let mut data = Vec::with_capacity(n);
    let step = (stop - start) / ((n - 1) as f64);
    for i in 0..n {
        data.push(if i + 1 == n {
            stop
        } else {
            start + (i as f64) * step
        });
    }
    ok1(Value::matrix(Matrix::row_vec(data)))
}

fn fn_logspace(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity("logspace", "2 or 3", args.len()));
    }
    let a = args[0].as_scalar()?;
    let b = args[1].as_scalar()?;
    let n = if args.len() == 3 {
        as_usize(&args[2], "logspace")?
    } else {
        50
    };
    if n == 0 {
        return ok1(Value::empty());
    }
    if n == 1 {
        return ok1(Value::scalar(10f64.powf(b)));
    }
    let step = (b - a) / ((n - 1) as f64);
    let data: Vec<f64> = (0..n)
        .map(|i| 10f64.powf(if i + 1 == n { b } else { a + (i as f64) * step }))
        .collect();
    ok1(Value::matrix(Matrix::row_vec(data)))
}

fn fn_diag(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("diag", "1 or 2 (v|A [, k])", args.len()));
    }
    let k: isize = if args.len() == 2 {
        let kf = args[1].as_scalar()?;
        if !kf.is_finite() || (kf - kf.round()).abs() > 1e-12 {
            return Err(MatroError::runtime("diag: diagonal offset k must be an integer"));
        }
        kf.round() as isize
    } else {
        0
    };
    let is_sp = matches!(&args[0], Value::Sparse(_));
    let m = args[0].to_dense_matrix()?;
    let k_abs = k.unsigned_abs();
    if m.rows() == 1 || m.cols() == 1 {
        let len = m.len();
        let n = len + k_abs;
        if is_sp && !m.is_complex() {
            let mut trips = Vec::with_capacity(len);
            for i in 0..len {
                let (r, c) = if k >= 0 { (i, i + k_abs) } else { (i + k_abs, i) };
                let v = m.data()[i];
                if v != 0.0 {
                    trips.push((r, c, v));
                }
            }
            return ok1(Value::sparse(SparseMatrix::from_triplets(n, n, &trips)?));
        }
        let mut out = Matrix::zeros(n, n);
        for i in 0..len {
            let (r, c) = if k >= 0 { (i, i + k_abs) } else { (i + k_abs, i) };
            out.set_c(r, c, m.get_linear_c(i));
        }
        ok1(Value::matrix(out.normalize()))
    } else {
        let (r0, c0) = if k >= 0 { (0usize, k_abs) } else { (k_abs, 0usize) };
        if r0 >= m.rows() || c0 >= m.cols() {
            return ok1(Value::matrix(Matrix::zeros(0, 1)));
        }
        let len = (m.rows() - r0).min(m.cols() - c0);
        if is_sp && !m.is_complex() {
            let mut trips = Vec::new();
            for i in 0..len {
                let v = m.get(r0 + i, c0 + i);
                if v != 0.0 {
                    trips.push((i, 0, v));
                }
            }
            return ok1(Value::sparse(SparseMatrix::from_triplets(len, 1, &trips)?));
        }
        let vals: Vec<Complex> = (0..len).map(|i| m.get_c(r0 + i, c0 + i)).collect();
        ok1(Value::matrix(Matrix::complex_col_vec(vals).normalize()))
    }
}

fn fn_magic(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("magic", "1", args.len()));
    }
    let n = as_usize(&args[0], "magic")?;
    if n == 0 {
        return ok1(Value::empty());
    }
    if n == 1 {
        return ok1(Value::scalar(1.0));
    }
    if n % 2 == 1 {
        let mut m = Matrix::zeros(n, n);
        let mut r = 0usize;
        let mut c = n / 2;
        for k in 1..=(n * n) {
            m.set(r, c, k as f64);
            let nr = (r + n - 1) % n;
            let nc = (c + 1) % n;
            if m.get(nr, nc) != 0.0 {
                r = (r + 1) % n;
            } else {
                r = nr;
                c = nc;
            }
        }
        return ok1(Value::matrix(m));
    }
    if n % 4 == 0 {
        let mut m = Matrix::zeros(n, n);
        let total = (n * n + 1) as f64;
        for r in 0..n {
            for c in 0..n {
                let v = (r * n + c + 1) as f64;
                let diag = (r % 4 == c % 4) || ((r % 4) + (c % 4) == 3);
                m.set(r, c, if diag { total - v } else { v });
            }
        }
        return ok1(Value::matrix(m));
    }
    Err(MatroError::runtime(
        "magic(n) currently supports odd n and multiples of 4",
    ))
}

fn value_dims(v: &Value) -> Vec<usize> {
    match v {
        Value::Matrix(m) => m.dims(),
        Value::Sparse(s) => vec![s.rows(), s.cols()],
        Value::Cell(c) => c.dims(),
        Value::StructArray(sa) => vec![sa.rows, sa.cols],
        Value::String(s) => vec![1, s.chars().count()],
        Value::Table(t) => vec![t.nrows, t.col_names.len()],
        Value::Map(m) => vec![m.entries.len(), 1],
        Value::Object(o) => match o.class_name.as_str() {
            "string" => o.fields.get("Data").map_or_else(|| vec![1, 1], value_dims),
            "datetime" => o.fields.get("EpochSec").map_or_else(|| vec![1, 1], value_dims),
            "duration" => o.fields.get("Seconds").map_or_else(|| vec![1, 1], value_dims),
            "categorical" => o.fields.get("Codes").map_or_else(|| vec![1, 1], value_dims),
            _ => vec![1, 1],
        },
        Value::Struct(_) | Value::Handle(_) | Value::Closure(_) => vec![1, 1],
    }
}

fn fn_size(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("size", ">= 1", 0));
    }
    let dims = value_dims(&args[0]);
    if args.len() >= 2 {
        let mut requested = Vec::new();
        for a in &args[1..] {
            let dm = a.to_dense_matrix()?;
            for &x in dm.data() {
                let d = x.round() as isize;
                if d < 1 {
                    return Err(MatroError::runtime(format!(
                        "size: dimension must be >= 1, got {x}"
                    )));
                }
                requested.push(dims.get((d - 1) as usize).copied().unwrap_or(1) as f64);
            }
        }
        if nargout >= 2 {
            let mut out = Vec::with_capacity(nargout);
            for k in 0..nargout {
                out.push(Value::scalar(requested.get(k).copied().unwrap_or(1.0)));
            }
            return Ok(out);
        }
        return if requested.len() == 1 {
            ok1(Value::scalar(requested[0]))
        } else {
            ok1(Value::matrix(Matrix::row_vec(requested)))
        };
    }
    if nargout >= 2 {
        let mut out = Vec::with_capacity(nargout);
        for k in 0..nargout {
            let val = if k + 1 == nargout && dims.len() > nargout {
                dims[k..].iter().product()
            } else {
                dims.get(k).copied().unwrap_or(1)
            };
            out.push(Value::scalar(val as f64));
        }
        Ok(out)
    } else {
        let d: Vec<f64> = dims.into_iter().map(|x| x as f64).collect();
        ok1(Value::matrix(Matrix::row_vec(d)))
    }
}

fn fn_length(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("length", "1", args.len()));
    }
    let dims = value_dims(&args[0]);
    let l = if dims.iter().any(|&d| d == 0) {
        0
    } else {
        dims.into_iter().max().unwrap_or(0)
    };
    ok1(Value::scalar(l as f64))
}

fn fn_numel(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("numel", "1", args.len()));
    }
    let n: usize = value_dims(&args[0]).into_iter().product();
    ok1(Value::scalar(n as f64))
}

fn fn_rows(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("rows", "1", args.len()));
    }
    ok1(Value::scalar(args[0].as_matrix()?.rows() as f64))
}

fn fn_cols(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("cols", "1", args.len()));
    }
    ok1(Value::scalar(args[0].as_matrix()?.cols() as f64))
}

fn fn_kron(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("kron", "2 (A, B)", args.len()));
    }
    match (&args[0], &args[1]) {
        (Value::Sparse(sa), Value::Sparse(sb)) => ok1(Value::sparse(sa.kron(sb))),
        (Value::Sparse(sa), Value::Matrix(mb)) => {
            ok1(Value::sparse(sa.kron(&SparseMatrix::from_dense(mb))))
        }
        (Value::Matrix(ma), Value::Sparse(sb)) => {
            ok1(Value::sparse(SparseMatrix::from_dense(ma).kron(sb)))
        }
        _ => {
            let a = args[0].as_matrix()?;
            let b = args[1].as_matrix()?;
            let (ra, ca) = (a.rows(), a.cols());
            let (rb, cb) = (b.rows(), b.cols());
            let out_r = ra * rb;
            let out_c = ca * cb;
            if !a.is_complex() && !b.is_complex() {
                let mut data = vec![0.0; out_r * out_c];
                for ja in 0..ca {
                    for jb in 0..cb {
                        let col = ja * cb + jb;
                        let col_off = col * out_r;
                        for ia in 0..ra {
                            let av = a.get(ia, ja);
                            let row_off = col_off + ia * rb;
                            for ib in 0..rb {
                                data[row_off + ib] = av * b.get(ib, jb);
                            }
                        }
                    }
                }
                ok1(Value::matrix(Matrix::from_col_major(out_r, out_c, data)?))
            } else {
                let mut re = vec![0.0; out_r * out_c];
                let mut im = vec![0.0; out_r * out_c];
                for ja in 0..ca {
                    for jb in 0..cb {
                        let col = ja * cb + jb;
                        let col_off = col * out_r;
                        for ia in 0..ra {
                            let av = a.get_c(ia, ja);
                            let row_off = col_off + ia * rb;
                            for ib in 0..rb {
                                let p = av * b.get_c(ib, jb);
                                re[row_off + ib] = p.re;
                                im[row_off + ib] = p.im;
                            }
                        }
                    }
                }
                ok1(Value::matrix(
                    Matrix::from_complex_col_major(out_r, out_c, re, im)?.normalize(),
                ))
            }
        }
    }
}

fn fn_sum(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 4 {
        return Err(MatroError::arity("sum", "1 to 4", args.len()));
    }
    if let Value::Sparse(s) = &args[0] {
        let mut all_mode = false;
        let mut omit_nan = false;
        let mut vecdim: Option<Vec<usize>> = None;
        for a in &args[1..] {
            if let Some(txt) = value_as_text(a) {
                match txt.trim().to_ascii_lowercase().as_str() {
                    "all" => all_mode = true,
                    "omitnan" | "omitmissing" => omit_nan = true,
                    _ => {}
                }
            } else {
                vecdim = Some(parse_vecdim(a, "sum")?);
            }
        }
        if all_mode {
            let total: f64 = s
                .nonzeros()
                .into_iter()
                .filter(|v| !omit_nan || !v.is_nan())
                .sum();
            return ok1(Value::scalar(total));
        }
        if !omit_nan && vecdim.as_ref().is_none_or(|v| v.len() == 1) {
            let dim = vecdim
                .and_then(|v| v.first().copied())
                .unwrap_or(if s.rows() == 1 { 2 } else { 1 });
            if dim == 1 || dim == 2 {
                return ok1(Value::matrix(s.sum_dim(dim)));
            }
        }
    }
    fn_reduce_c("sum", args, |slice| {
        slice.iter().copied().fold(Complex::ZERO, |a, b| a + b)
    })
}

fn fn_reshape(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 {
        return Err(MatroError::arity("reshape", ">= 2 (A, sz | d1, d2, ...)", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let total = m.len();
    let dims: Vec<usize> = if args.len() == 2 {
        let sz = args[1].as_matrix()?;
        if sz.len() < 2 {
            return Err(MatroError::dimension("reshape: size vector must have at least 2 elements"));
        }
        sz.data()
            .iter()
            .map(|&v| {
                let n = v.round() as isize;
                if n < 0 {
                    Err(MatroError::dimension("reshape: size must be non-negative"))
                } else {
                    Ok(n as usize)
                }
            })
            .collect::<Result<_>>()?
    } else {
        let mut parsed: Vec<Option<usize>> = Vec::with_capacity(args.len() - 1);
        let mut auto_idx: Option<usize> = None;
        for (i, a) in args[1..].iter().enumerate() {
            let am = a.as_matrix()?;
            if am.is_empty() {
                if auto_idx.is_some() {
                    return Err(MatroError::dimension(
                        "reshape: size can only have one unknown dimension []",
                    ));
                }
                auto_idx = Some(i);
                parsed.push(None);
            } else {
                parsed.push(Some(as_usize(a, "reshape")?));
            }
        }
        if let Some(ai) = auto_idx {
            let known_prod: usize = parsed.iter().flatten().product();
            if known_prod == 0 || total % known_prod != 0 {
                return Err(MatroError::dimension(format!(
                    "reshape: number of elements ({total}) is not divisible by {known_prod}"
                )));
            }
            parsed[ai] = Some(total / known_prod);
        }
        parsed.into_iter().map(Option::unwrap).collect()
    };
    let out = if let Some(im) = m.imag_data() {
        Matrix::from_nd_complex_col_major(dims, m.data().to_vec(), im.to_vec())?
    } else {
        Matrix::from_nd_col_major(dims, m.data().to_vec())?
    };
    ok1(Value::matrix(out))
}

fn fn_ind2sub(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("ind2sub", "2 (sz, ind)", args.len()));
    }
    let sz_mat = args[0].as_matrix()?;
    if sz_mat.is_empty() {
        return Err(MatroError::dimension("ind2sub: size vector cannot be empty"));
    }
    let mut dims: Vec<usize> = Vec::with_capacity(sz_mat.len());
    for &v in sz_mat.data() {
        let d = v.round() as isize;
        if !v.is_finite() || (v - d as f64).abs() > 1e-12 || d < 0 {
            return Err(MatroError::dimension(
                "ind2sub: size vector must contain non-negative integers",
            ));
        }
        dims.push(d as usize);
    }
    let total: usize = dims.iter().copied().product();
    let ind_mat = args[1].as_matrix()?;
    let nout = nargout.max(1);
    let out_shape = ind_mat.dims();
    let n_elem = ind_mat.len();

    let mut outputs: Vec<Vec<f64>> = (0..nout).map(|_| vec![1.0; n_elem]).collect();
    for (idx_pos, &raw) in ind_mat.data().iter().enumerate() {
        let one_based = raw.round() as isize;
        if !raw.is_finite()
            || (raw - one_based as f64).abs() > 1e-12
            || one_based < 1
            || (one_based as usize) > total
        {
            return Err(MatroError::index(format!(
                "ind2sub: linear index {raw} out of bounds for size {:?} (numel = {total})",
                dims
            )));
        }
        let mut rem = (one_based - 1) as usize;
        if nout == 1 {
            outputs[0][idx_pos] = one_based as f64;
        } else {
            // Decompose along first `min(nout - 1, dims.len())` dimensions;
            // the last requested output absorbs all remaining trailing dimensions (MATLAB rule),
            // and any outputs beyond `dims.len()` remain 1 (trailing singleton dims).
            let last_active = (nout - 1).min(dims.len());
            for k in 0..last_active {
                let dk = dims[k];
                let sub = rem % dk;
                rem /= dk;
                outputs[k][idx_pos] = (sub + 1) as f64;
            }
            if nout - 1 < dims.len() {
                outputs[nout - 1][idx_pos] = (rem + 1) as f64;
            }
        }
    }

    outputs
        .into_iter()
        .map(|buf| Ok(Value::matrix(Matrix::from_nd_col_major(out_shape.clone(), buf)?)))
        .collect()
}

fn fn_sub2ind(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 {
        return Err(MatroError::arity("sub2ind", ">= 2 (sz, i1, i2, ...)", args.len()));
    }
    let sz_mat = args[0].as_matrix()?;
    if sz_mat.is_empty() {
        return Err(MatroError::dimension("sub2ind: size vector cannot be empty"));
    }
    let mut dims: Vec<usize> = Vec::with_capacity(sz_mat.len());
    for &v in sz_mat.data() {
        let d = v.round() as isize;
        if !v.is_finite() || (v - d as f64).abs() > 1e-12 || d < 0 {
            return Err(MatroError::dimension(
                "sub2ind: size vector must contain non-negative integers",
            ));
        }
        dims.push(d as usize);
    }
    let k = args.len() - 1;
    let mut eff_dims = Vec::with_capacity(k);
    for ax in 0..k {
        if ax + 1 < k {
            eff_dims.push(dims.get(ax).copied().unwrap_or(1));
        } else if ax < dims.len() {
            eff_dims.push(dims[ax..].iter().copied().product());
        } else {
            eff_dims.push(1);
        }
    }
    let mut strides = Vec::with_capacity(k);
    let mut s = 1usize;
    for &d in &eff_dims {
        strides.push(s);
        s = s.saturating_mul(d);
    }

    let sub_mats: Vec<&Matrix> = args[1..]
        .iter()
        .map(|v| v.as_matrix())
        .collect::<Result<_>>()?;
    let mut out_shape: Option<Vec<usize>> = None;
    for m in &sub_mats {
        if m.len() != 1 {
            if let Some(ref prev) = out_shape {
                if *prev != m.dims() {
                    return Err(MatroError::dimension(
                        "sub2ind: subscript arrays must have matching dimensions",
                    ));
                }
            } else {
                out_shape = Some(m.dims());
            }
        }
    }
    let out_shape = out_shape.unwrap_or_else(|| sub_mats[0].dims());
    let n_elem: usize = out_shape.iter().copied().product();
    let mut out = vec![0.0; n_elem];

    for pos in 0..n_elem {
        let mut lin = 1usize;
        for (ax, m) in sub_mats.iter().enumerate() {
            let raw = if m.len() == 1 { m.data()[0] } else { m.data()[pos] };
            let sub = raw.round() as isize;
            if !raw.is_finite()
                || (raw - sub as f64).abs() > 1e-12
                || sub < 1
                || (sub as usize) > eff_dims[ax]
            {
                return Err(MatroError::index(format!(
                    "sub2ind: subscript {raw} out of bounds for dimension {} (size {})",
                    ax + 1,
                    eff_dims[ax]
                )));
            }
            lin += ((sub - 1) as usize) * strides[ax];
        }
        out[pos] = lin as f64;
    }

    ok1(Value::matrix(Matrix::from_nd_col_major(out_shape, out)?))
}

fn fn_unary_mat(
    name: &str,
    args: &[Value],
    f: impl FnOnce(&Matrix) -> Result<Matrix>,
) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity(name, "1", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    ok1(Value::matrix(f(&m)?))
}

fn fn_lu(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("lu", "1 or 2 (A [, 'vector'|'matrix'])", args.len()));
    }
    let vec_perm = args
        .get(1)
        .and_then(value_as_text)
        .is_some_and(|s| s.eq_ignore_ascii_case("vector"));
    if let Value::Sparse(sp) = &args[0] {
        if sp.rows() == sp.cols() {
            let (l, u, perm) = if sp.is_complex() {
                sp.lu_sparse_complex()?
            } else {
                sp.lu_sparse()?
            };
            let n = perm.len();
            if nargout >= 3 {
                let p_val = if vec_perm {
                    Value::matrix(Matrix::col_vec(
                        perm.iter().map(|&idx| (idx + 1) as f64).collect(),
                    ))
                } else {
                    let p_trips: Vec<(usize, usize, f64)> =
                        perm.iter().enumerate().map(|(i, &j)| (i, j, 1.0)).collect();
                    Value::sparse(SparseMatrix::from_triplets(n, n, &p_trips)?)
                };
                return Ok(vec![Value::sparse(l), Value::sparse(u), p_val]);
            } else if nargout == 2 {
                let pt_trips: Vec<(usize, usize, f64)> =
                    perm.iter().enumerate().map(|(i, &j)| (j, i, 1.0)).collect();
                let pt = SparseMatrix::from_triplets(n, n, &pt_trips)?;
                let pl = pt.matmul_sparse(&l)?;
                return Ok(vec![Value::sparse(pl), Value::sparse(u)]);
            } else {
                return ok1(Value::sparse(u));
            }
        }
    }
    let (l, u, p) = args[0].to_dense_matrix()?.lu()?;
    if nargout >= 3 {
        let p_out = if vec_perm {
            let n = p.rows();
            let mut pv = vec![0.0; n];
            for r in 0..n {
                for c in 0..p.cols() {
                    if p.get(r, c) != 0.0 {
                        pv[r] = (c + 1) as f64;
                        break;
                    }
                }
            }
            Matrix::col_vec(pv)
        } else {
            p
        };
        Ok(vec![Value::matrix(l), Value::matrix(u), Value::matrix(p_out)])
    } else if nargout == 2 {
        let pl = p.transpose().matmul(&l)?;
        Ok(vec![Value::matrix(pl), Value::matrix(u)])
    } else {
        ok1(Value::matrix(u))
    }
}

fn slice_submatrix(m: &Matrix, r_end: usize, c_end: usize) -> Matrix {
    let mut out = if m.is_complex() {
        Matrix::complex_zeros(r_end, c_end)
    } else {
        Matrix::zeros(r_end, c_end)
    };
    for c in 0..c_end {
        for r in 0..r_end {
            out.set_c(r, c, m.get_c(r, c));
        }
    }
    out.normalize()
}

fn fn_qr(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity("qr", "1 to 3 (A [, 'econ'|0, 'vector'|'matrix'])", args.len()));
    }
    let a = args[0].to_dense_matrix()?;
    let mut econ = false;
    let mut vec_perm = false;
    for arg in &args[1..] {
        if let Some(s) = value_as_text(arg) {
            match s.trim().to_ascii_lowercase().as_str() {
                "econ" => econ = true,
                "vector" => vec_perm = true,
                "matrix" => vec_perm = false,
                _ => {}
            }
        } else if let Ok(v) = arg.as_scalar() {
            if v == 0.0 {
                econ = true;
                if nargout >= 3 {
                    vec_perm = true;
                }
            }
        }
    }
    let m = a.rows();
    let n = a.cols();
    let k = m.min(n);
    if nargout >= 3 {
        let (q_full, r_full, perm) = a.qr_pivoted()?;
        let (q, r) = if econ && m > n {
            (slice_submatrix(&q_full, m, k), slice_submatrix(&r_full, k, n))
        } else {
            (q_full, r_full)
        };
        let e = if vec_perm {
            Matrix::row_vec(perm.iter().map(|&p| (p + 1) as f64).collect())
        } else {
            let mut em = Matrix::zeros(n, n);
            for (col, &p) in perm.iter().enumerate() {
                em.set(p, col, 1.0);
            }
            em
        };
        Ok(vec![Value::matrix(q), Value::matrix(r), Value::matrix(e)])
    } else {
        let (q_full, r_full) = a.qr()?;
        let (q, r) = if econ && m > n {
            (slice_submatrix(&q_full, m, k), slice_submatrix(&r_full, k, n))
        } else {
            (q_full, r_full)
        };
        if nargout >= 2 {
            Ok(vec![Value::matrix(q), Value::matrix(r)])
        } else {
            ok1(Value::matrix(r))
        }
    }
}

fn fn_svd(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity("svd", "1 to 3 (A [, 'econ'|0 [, 'vector'|'matrix']])", args.len()));
    }
    let a = args[0].to_dense_matrix()?;
    let mut econ = false;
    let mut s_vec = false;
    for arg in &args[1..] {
        if let Some(s) = value_as_text(arg) {
            match s.trim().to_ascii_lowercase().as_str() {
                "econ" => econ = true,
                "vector" => s_vec = true,
                "matrix" => s_vec = false,
                _ => {}
            }
        } else if let Ok(v) = arg.as_scalar() {
            if v == 0.0 {
                econ = true;
            }
        }
    }
    let (u_full, s_full, v_full) = a.svd()?;
    let m = a.rows();
    let n = a.cols();
    let k = m.min(n);
    if nargout >= 3 {
        let (u, s, v) = if econ {
            (
                slice_submatrix(&u_full, m, k),
                slice_submatrix(&s_full, k, k),
                slice_submatrix(&v_full, n, k),
            )
        } else {
            (u_full, s_full, v_full)
        };
        let s_out = if s_vec {
            Matrix::col_vec((0..k).map(|i| s.get(i, i)).collect())
        } else {
            s
        };
        Ok(vec![Value::matrix(u), Value::matrix(s_out), Value::matrix(v)])
    } else {
        let diag: Vec<f64> = (0..k).map(|i| s_full.get(i, i)).collect();
        ok1(Value::matrix(Matrix::col_vec(diag)))
    }
}

fn fn_chol(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity("chol", "1 to 3 (A [, 'upper'|'lower'])", args.len()));
    }
    let mut lower = false;
    let mut vec_perm = false;
    for arg in &args[1..] {
        if let Some(s) = value_as_text(arg) {
            match s.trim().to_ascii_lowercase().as_str() {
                "lower" => lower = true,
                "upper" => lower = false,
                "vector" => vec_perm = true,
                _ => {}
            }
        }
    }
    if let Value::Sparse(sp) = &args[0] {
        if !sp.is_complex() && sp.rows() == sp.cols() {
            match sp.chol_sparse() {
                Ok(l) => {
                    let factor = if lower { l } else { l.transpose() };
                    if nargout >= 3 {
                        let n = sp.rows();
                        let p_val = if vec_perm {
                            Value::matrix(Matrix::row_vec((1..=n).map(|i| i as f64).collect()))
                        } else {
                            Value::sparse(SparseMatrix::eye(n))
                        };
                        return Ok(vec![Value::sparse(factor), Value::scalar(0.0), p_val]);
                    } else if nargout == 2 {
                        return Ok(vec![Value::sparse(factor), Value::scalar(0.0)]);
                    } else {
                        return ok1(Value::sparse(factor));
                    }
                }
                Err(flag) => {
                    if nargout >= 2 {
                        let empty = Value::sparse(SparseMatrix::zeros(0, 0));
                        if nargout >= 3 {
                            let n = sp.rows();
                            let p_val = if vec_perm {
                                Value::matrix(Matrix::row_vec((1..=n).map(|i| i as f64).collect()))
                            } else {
                                Value::sparse(SparseMatrix::eye(n))
                            };
                            return Ok(vec![empty, Value::scalar(flag as f64), p_val]);
                        }
                        return Ok(vec![empty, Value::scalar(flag as f64)]);
                    }
                    return Err(MatroError::runtime("Matrix is not positive definite for chol()"));
                }
            }
        }
    }
    let a = args[0].to_dense_matrix()?;
    let (factor, flag) = a.chol_with_flag(lower)?;
    if nargout >= 2 {
        Ok(vec![Value::matrix(factor), Value::scalar(flag as f64)])
    } else if flag != 0 {
        Err(MatroError::runtime("Matrix is not positive definite for chol()"))
    } else {
        ok1(Value::matrix(factor))
    }
}

fn fn_eig(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 4 {
        return Err(MatroError::arity("eig", "1 to 4 (A [, B] [, 'vector'|'matrix'])", args.len()));
    }
    let a = args[0].to_dense_matrix()?;
    let mut b_opt: Option<Matrix> = None;
    let mut d_mode: Option<&str> = None;
    for arg in &args[1..] {
        if let Some(s) = value_as_text(arg) {
            match s.trim().to_ascii_lowercase().as_str() {
                "vector" => d_mode = Some("vector"),
                "matrix" => d_mode = Some("matrix"),
                _ => {}
            }
        } else if b_opt.is_none() {
            b_opt = Some(arg.to_dense_matrix()?);
        }
    }
    let (v, d) = if let Some(b) = b_opt {
        if b.rows() != a.rows() || b.cols() != a.cols() {
            return Err(MatroError::dimension("eig(A, B): A and B must have the same square dimensions"));
        }
        // If B is SPD, solve symmetric generalized problem via Cholesky: L^{-1} A L^{-T} y = lambda y
        if let Ok((l, 0)) = b.chol_with_flag(true) {
            let linv_a = l.solve(&a)?;
            let sym_m = l.solve(&linv_a.ctranspose())?.ctranspose();
            let (y, d_mat) = sym_m.eig()?;
            let v_mat = l.ctranspose().solve(&y)?;
            (v_mat.normalize(), d_mat.normalize())
        } else {
            let m = b.solve(&a)?;
            m.eig()?
        }
    } else {
        a.eig()?
    };

    let n = d.rows().min(d.cols());
    let d_vec = || {
        let diag: Vec<Complex> = (0..n).map(|i| d.get_c(i, i)).collect();
        Matrix::complex_col_vec(diag).normalize()
    };

    if nargout >= 3 {
        // Left eigenvectors W satisfy W' * A = D * W' => columns of W are eigenvectors of A'
        let (w, _) = a.ctranspose().eig()?;
        let d_out = if d_mode == Some("vector") { d_vec() } else { d };
        Ok(vec![Value::matrix(v), Value::matrix(d_out), Value::matrix(w)])
    } else if nargout == 2 {
        let d_out = if d_mode == Some("vector") { d_vec() } else { d };
        Ok(vec![Value::matrix(v), Value::matrix(d_out)])
    } else if d_mode == Some("matrix") {
        ok1(Value::matrix(d))
    } else {
        ok1(Value::matrix(d_vec()))
    }
}

fn fn_rank(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("rank", "1 or 2 (A [, tol])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let (_, s, _) = m.svd()?;
    let k = s.rows().min(s.cols());
    if k == 0 {
        return ok1(Value::scalar(0.0));
    }
    let max_s = s.get(0, 0);
    let tol = if args.len() == 2 {
        args[1].as_scalar()?
    } else {
        (m.rows().max(m.cols()) as f64) * f64::EPSILON * max_s
    };
    let r = (0..k).filter(|&i| s.get(i, i) > tol).count();
    ok1(Value::scalar(r as f64))
}

fn fn_cond(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("cond", "1 or 2 (A [, p])", args.len()));
    }
    let a = args[0].to_dense_matrix()?;
    if a.is_empty() {
        return ok1(Value::scalar(0.0));
    }
    if args.len() == 2 {
        let is_two = args[1].as_scalar().ok() == Some(2.0);
        if !is_two {
            let n_a = matrix_norm_p(&a, &args[1])?;
            if let Ok(ainv) = a.inv() {
                let n_inv = matrix_norm_p(&ainv, &args[1])?;
                return ok1(Value::scalar(n_a * n_inv));
            }
            return ok1(Value::scalar(f64::INFINITY));
        }
    }
    let (_, s, _) = a.svd()?;
    let k = s.rows().min(s.cols());
    if k == 0 {
        return ok1(Value::scalar(0.0));
    }
    let smax = s.get(0, 0);
    let smin = s.get(k - 1, k - 1);
    ok1(Value::scalar(if smin == 0.0 || (smax > 0.0 && smin / smax < f64::EPSILON * 0.1) {
        f64::INFINITY
    } else {
        smax / smin
    }))
}

fn fn_fft(args: &[Value], inverse: bool) -> Result<Vec<Value>> {
    let name = if inverse { "ifft" } else { "fft" };
    if args.is_empty() || args.len() > 4 {
        return Err(MatroError::arity(name, "1 to 4 (X [, n [, dim [, symflag]]])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let mut symmetric = false;
    let mut num_args: Vec<&Value> = Vec::new();
    for a in &args[1..] {
        if let Some(s) = value_as_text(a) {
            if s.eq_ignore_ascii_case("symmetric") {
                symmetric = true;
            } else if s.eq_ignore_ascii_case("nonsymmetric") {
                symmetric = false;
            } else {
                return Err(MatroError::runtime(format!("{name}: unknown flag '{s}'")));
            }
        } else {
            num_args.push(a);
        }
    }
    let n_opt = if let Some(v) = num_args.first() {
        if v.to_dense_matrix().is_ok_and(|nm| nm.is_empty()) {
            None
        } else {
            Some(as_usize(v, name)?)
        }
    } else {
        None
    };
    let dim_opt = if let Some(v) = num_args.get(1) {
        Some(as_usize(v, name)?)
    } else {
        None
    };
    ok1(Value::matrix(m.fft_along_dim(n_opt, dim_opt, inverse, symmetric)?))
}

fn fn_fft2(args: &[Value], inverse: bool) -> Result<Vec<Value>> {
    let name = if inverse { "ifft2" } else { "fft2" };
    if args.is_empty() || args.len() > 4 {
        return Err(MatroError::arity(name, "1 to 4 (X [, m, n [, symflag]])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let mut symmetric = false;
    let mut num_args: Vec<&Value> = Vec::new();
    for a in &args[1..] {
        if let Some(s) = value_as_text(a) {
            if s.eq_ignore_ascii_case("symmetric") {
                symmetric = true;
            } else if s.eq_ignore_ascii_case("nonsymmetric") {
                symmetric = false;
            }
        } else {
            num_args.push(a);
        }
    }
    let m_opt = num_args.first().map(|v| as_usize(v, name)).transpose()?;
    let n_opt = num_args.get(1).map(|v| as_usize(v, name)).transpose()?;
    ok1(Value::matrix(m.fft2_sized(m_opt, n_opt, inverse, symmetric)?))
}

fn fn_fftn(args: &[Value], inverse: bool) -> Result<Vec<Value>> {
    let name = if inverse { "ifftn" } else { "fftn" };
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity(name, "1 to 3 (X [, sz [, symflag]])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let mut symmetric = false;
    let mut sz_opt: Option<Vec<usize>> = None;
    for a in &args[1..] {
        if let Some(s) = value_as_text(a) {
            if s.eq_ignore_ascii_case("symmetric") {
                symmetric = true;
            }
        } else {
            let sm = a.to_dense_matrix()?;
            if !sm.is_empty() {
                sz_opt = Some(sm.data().iter().map(|&x| x.max(0.0).round() as usize).collect());
            }
        }
    }
    ok1(Value::matrix(m.fftn_sized(sz_opt.as_deref(), inverse, symmetric)?))
}

fn fn_log2(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("log2", "1", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    if nargout >= 2 {
        let mut f_data = Vec::with_capacity(m.len());
        let mut e_data = Vec::with_capacity(m.len());
        for &x in m.data() {
            if x == 0.0 || !x.is_finite() {
                f_data.push(x);
                e_data.push(0.0);
            } else {
                let ax = x.abs();
                let exp = ax.log2().floor() as i32 + 1;
                let mant = x / (2.0_f64).powi(exp);
                f_data.push(mant);
                e_data.push(exp as f64);
            }
        }
        let dims = m.dims();
        Ok(vec![
            Value::matrix(Matrix::from_nd_col_major(dims.clone(), f_data)?),
            Value::matrix(Matrix::from_nd_col_major(dims, e_data)?),
        ])
    } else {
        ok1(Value::matrix(m.map_c(Complex::log2).normalize()))
    }
}

fn round_scalar_n(x: f64, n: i32, sig: bool) -> f64 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    if sig {
        if n <= 0 {
            return 0.0;
        }
        let mag = x.abs().log10().floor() as i32;
        let scale = 10.0_f64.powi(n - 1 - mag);
        (x * scale).round() / scale
    } else if n == 0 {
        x.round()
    } else if n > 0 {
        let scale = 10.0_f64.powi(n);
        (x * scale).round() / scale
    } else {
        let scale = 10.0_f64.powi(-n);
        (x / scale).round() * scale
    }
}

fn fn_round(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity("round", "1 to 3 (X [, N [, 'decimals'|'significant']])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let n = if let Some(nv) = args.get(1) {
        nv.as_scalar()?.round() as i32
    } else {
        0
    };
    let sig = args
        .get(2)
        .and_then(value_as_text)
        .is_some_and(|s| s.eq_ignore_ascii_case("significant"));
    let out = m
        .map_c(|c| Complex::new(round_scalar_n(c.re, n, sig), round_scalar_n(c.im, n, sig)))
        .normalize();
    ok1(Value::matrix(out))
}

fn fn_polyfit(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() != 3 {
        return Err(MatroError::arity("polyfit", "3", args.len()));
    }
    let x = args[0].to_dense_matrix()?;
    let y = args[1].to_dense_matrix()?;
    let deg = as_usize(&args[2], "polyfit")?;
    if x.len() != y.len() || x.len() <= deg {
        return Err(MatroError::dimension(
            "polyfit: x and y must have equal length > degree",
        ));
    }
    let n = x.len();
    let cols = deg + 1;
    let use_scale = nargout >= 3;
    let mean_x = x.data().iter().sum::<f64>() / (n as f64);
    let std_x = if n > 1 {
        (x.data().iter().map(|&v| (v - mean_x).powi(2)).sum::<f64>() / ((n - 1) as f64)).sqrt()
    } else {
        1.0
    };
    let scale = if use_scale && std_x > 0.0 { std_x } else { 1.0 };
    let shift = if use_scale { mean_x } else { 0.0 };

    let mut vander = Matrix::zeros(n, cols);
    for r in 0..n {
        let xv = (x.data()[r] - shift) / scale;
        for c in 0..cols {
            let p = (deg - c) as i32;
            vander.set(r, c, xv.powi(p));
        }
    }
    let y_vals: Vec<Complex> = (0..n).map(|i| y.get_linear_c(i)).collect();
    let y_col = Matrix::complex_col_vec(y_vals).normalize();
    let coeff = vander.solve(&y_col)?;
    let p_row = coeff.transpose().normalize();
    if nargout <= 1 {
        return ok1(Value::matrix(p_row));
    }
    let (_, r_full) = vander.qr()?;
    let r_mat = slice_submatrix(&r_full, cols, cols);
    let resid = vander.matmul(&coeff)?.elementwise_c(&y_col, |a, b| a - b)?;
    let normr: f64 = (0..resid.len())
        .map(|i| resid.get_linear_c(i).norm_sq())
        .sum::<f64>()
        .sqrt();
    let mut s_map = BTreeMap::new();
    s_map.insert("R".to_string(), Value::matrix(r_mat));
    s_map.insert("df".to_string(), Value::scalar((n.saturating_sub(cols)) as f64));
    s_map.insert("normr".to_string(), Value::scalar(normr));
    if nargout >= 3 {
        let mu = Matrix::col_vec(vec![mean_x, if std_x > 0.0 { std_x } else { 1.0 }]);
        Ok(vec![
            Value::matrix(p_row),
            Value::Struct(s_map),
            Value::matrix(mu),
        ])
    } else {
        Ok(vec![Value::matrix(p_row), Value::Struct(s_map)])
    }
}

fn fn_polyval(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 4 {
        return Err(MatroError::arity("polyval", "2 to 4 (p, x [, S [, mu]])", args.len()));
    }
    let p = args[0].to_dense_matrix()?;
    let x = args[1].to_dense_matrix()?;
    let (shift, scale) = if let Some(mu_v) = args.get(3) {
        let mu = mu_v.to_dense_matrix()?;
        if mu.len() >= 2 {
            (mu.data()[0], mu.data()[1])
        } else {
            (0.0, 1.0)
        }
    } else {
        (0.0, 1.0)
    };
    let n_coeffs = p.len();
    let out = x
        .map_c(|xv| {
            let xhat = (xv - Complex::from_real(shift)) / Complex::from_real(scale);
            let mut acc = Complex::ZERO;
            for i in 0..n_coeffs {
                acc = acc * xhat + p.get_linear_c(i);
            }
            acc
        })
        .normalize();
    if nargout >= 2 {
        let mut delta = Matrix::zeros_nd(x.dims());
        if let Some(Value::Struct(s_map)) = args.get(2) {
            if let (Some(r_val), Some(df_val), Some(normr_val)) =
                (s_map.get("R"), s_map.get("df"), s_map.get("normr"))
            {
                let r_mat = r_val.to_dense_matrix()?;
                let df = df_val.as_scalar()?.max(1.0);
                let normr = normr_val.as_scalar()?;
                let deg = n_coeffs.saturating_sub(1);
                let mut v = Matrix::zeros(x.len(), n_coeffs);
                for row in 0..x.len() {
                    let xh = (x.data()[row] - shift) / scale;
                    for col in 0..n_coeffs {
                        v.set(row, col, xh.powi((deg - col) as i32));
                    }
                }
                if let Ok(e_mat) = v.rdivide(&r_mat) {
                    let mut d_vals = Vec::with_capacity(x.len());
                    for row in 0..x.len() {
                        let e2: f64 = (0..n_coeffs).map(|c| e_mat.get(row, c).powi(2)).sum();
                        d_vals.push((normr / df.sqrt()) * (1.0 + e2).sqrt());
                    }
                    delta = Matrix::from_nd_col_major(x.dims(), d_vals)?;
                }
            }
        }
        Ok(vec![Value::matrix(out), Value::matrix(delta)])
    } else {
        ok1(Value::matrix(out))
    }
}

fn fn_conv(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity("conv", "2 or 3 (u, v [, shape])", args.len()));
    }
    let a = args[0].as_matrix()?;
    let b = args[1].as_matrix()?;
    let shape = if args.len() == 3 {
        value_as_text(&args[2])
            .ok_or_else(|| MatroError::runtime("conv: shape must be 'full', 'same', or 'valid'"))?
            .trim()
            .to_ascii_lowercase()
    } else {
        "full".to_string()
    };
    if !matches!(shape.as_str(), "full" | "same" | "valid") {
        return Err(MatroError::runtime(format!(
            "conv: unknown shape '{shape}' (expected 'full', 'same', or 'valid')"
        )));
    }
    let a_is_col = a.cols() == 1 && a.rows() > 1;
    let b_is_col = b.cols() == 1 && b.rows() > 1;
    let is_col = match shape.as_str() {
        "same" => a_is_col,
        _ => {
            if a.len() >= b.len() {
                a_is_col
            } else {
                b_is_col
            }
        }
    };
    if a.is_empty() || b.is_empty() {
        return ok1(Value::empty());
    }
    let full_n = a.len() + b.len() - 1;
    let mut full = vec![Complex::ZERO; full_n];
    for i in 0..a.len() {
        let ai = a.get_linear_c(i);
        for j in 0..b.len() {
            full[i + j] = full[i + j] + ai * b.get_linear_c(j);
        }
    }
    let out: Vec<Complex> = match shape.as_str() {
        "full" => full,
        "same" => {
            // MATLAB central slice of length a.len(): start = ceil((b.len() - 1) / 2) = b.len() / 2
            let start = b.len() / 2;
            full[start..start + a.len()].to_vec()
        }
        "valid" => {
            if a.len() < b.len() {
                return ok1(Value::empty());
            }
            let start = b.len() - 1;
            let len = a.len() - b.len() + 1;
            full[start..start + len].to_vec()
        }
        _ => unreachable!(),
    };
    let mat = if is_col {
        Matrix::complex_col_vec(out)
    } else {
        Matrix::complex_row_vec(out)
    };
    ok1(Value::matrix(mat.normalize()))
}

fn fn_ndgrid(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("ndgrid", ">= 1", 0));
    }
    let vecs: Vec<Matrix> = if args.len() == 1 {
        let v = args[0].to_dense_matrix()?;
        let nd = nargout.max(2);
        vec![v; nd]
    } else {
        args.iter()
            .map(|a| a.to_dense_matrix())
            .collect::<Result<_>>()?
    };
    let nd = vecs.len();
    let dims: Vec<usize> = vecs.iter().map(Matrix::len).collect();
    let total: usize = dims.iter().product();
    let mut results = Vec::with_capacity(nd);
    for ax in 0..nd {
        let inner: usize = dims[..ax].iter().product();
        let axis_len = dims[ax];
        let outer: usize = dims[ax + 1..].iter().product();
        let mut re = vec![0.0; total];
        let mut im = vec![0.0; total];
        for o in 0..outer {
            let block = o * axis_len * inner;
            for k in 0..axis_len {
                let val = vecs[ax].get_linear_c(k);
                let base = block + k * inner;
                for i in 0..inner {
                    re[base + i] = val.re;
                    im[base + i] = val.im;
                }
            }
        }
        results.push(Value::matrix(
            Matrix::from_nd_complex_col_major(dims.clone(), re, im)?.normalize(),
        ));
    }
    Ok(results)
}

fn matlab_f64_cmp(a: f64, b: f64, desc: bool) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => {
            if desc {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        }
        (false, true) => {
            if desc {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Less
            }
        }
        (false, false) => {
            let ord = a.total_cmp(&b);
            if desc { ord.reverse() } else { ord }
        }
    }
}

fn fn_sort(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity(
            "sort",
            "1 to 3 (A [, dim [, direction]])",
            args.len(),
        ));
    }
    let m = args[0].to_dense_matrix()?;
    let mut desc = false;
    let mut dim_opt: Option<usize> = None;
    for a in &args[1..] {
        if let Some(s) = value_as_text(a) {
            if s.eq_ignore_ascii_case("descend") {
                desc = true;
            } else if s.eq_ignore_ascii_case("ascend") {
                desc = false;
            } else {
                return Err(MatroError::runtime(format!(
                    "sort: unknown direction '{s}' (expected 'ascend' or 'descend')"
                )));
            }
        } else {
            let d = as_usize(a, "sort")?;
            if d == 0 {
                return Err(MatroError::runtime("sort: dimension must be >= 1"));
            }
            dim_opt = Some(d);
        }
    }
    let dims = m.dims();
    let dim = dim_opt.unwrap_or_else(|| dims.iter().position(|&d| d != 1).map_or(1, |i| i + 1));
    let ax = dim - 1;
    let total = m.len();
    if ax >= dims.len() || dims[ax] <= 1 || total == 0 {
        let i_mat = Matrix::ones_nd(dims);
        return if nargout >= 2 {
            Ok(vec![Value::matrix(m), Value::matrix(i_mat)])
        } else {
            ok1(Value::matrix(m))
        };
    }

    let inner: usize = dims[..ax].iter().product();
    let axis_len = dims[ax];
    let outer: usize = dims[ax + 1..].iter().product();
    let is_complex = m.is_complex();
    let mut out_re = vec![0.0; total];
    let mut out_im = if is_complex { vec![0.0; total] } else { Vec::new() };
    let mut out_idx = vec![0.0; total];

    if is_complex {
        let mut pairs: Vec<(Complex, usize)> = Vec::with_capacity(axis_len);
        for o in 0..outer {
            let block = o * axis_len * inner;
            for i in 0..inner {
                pairs.clear();
                for k in 0..axis_len {
                    pairs.push((m.get_linear_c(block + k * inner + i), k + 1));
                }
                pairs.sort_by(|a, b| {
                    let a_nan = a.0.re.is_nan() || a.0.im.is_nan();
                    let b_nan = b.0.re.is_nan() || b.0.im.is_nan();
                    match (a_nan, b_nan) {
                        (true, true) => std::cmp::Ordering::Equal,
                        (true, false) => {
                            if desc {
                                std::cmp::Ordering::Less
                            } else {
                                std::cmp::Ordering::Greater
                            }
                        }
                        (false, true) => {
                            if desc {
                                std::cmp::Ordering::Greater
                            } else {
                                std::cmp::Ordering::Less
                            }
                        }
                        (false, false) => {
                            let ord = a
                                .0
                                .norm_sq()
                                .total_cmp(&b.0.norm_sq())
                                .then_with(|| a.0.im.atan2(a.0.re).total_cmp(&b.0.im.atan2(b.0.re)));
                            if desc { ord.reverse() } else { ord }
                        }
                    }
                });
                for (k, &(val, idx)) in pairs.iter().enumerate() {
                    let pos = block + k * inner + i;
                    out_re[pos] = val.re;
                    out_im[pos] = val.im;
                    out_idx[pos] = idx as f64;
                }
            }
        }
    } else {
        let src = m.data();
        let mut pairs: Vec<(f64, usize)> = Vec::with_capacity(axis_len);
        for o in 0..outer {
            let block = o * axis_len * inner;
            for i in 0..inner {
                pairs.clear();
                for k in 0..axis_len {
                    pairs.push((src[block + k * inner + i], k + 1));
                }
                pairs.sort_by(|a, b| matlab_f64_cmp(a.0, b.0, desc));
                for (k, &(val, idx)) in pairs.iter().enumerate() {
                    let pos = block + k * inner + i;
                    out_re[pos] = val;
                    out_idx[pos] = idx as f64;
                }
            }
        }
    }

    let v_mat = if is_complex {
        Matrix::from_nd_complex_col_major(dims.clone(), out_re, out_im)?
    } else {
        Matrix::from_nd_col_major(dims.clone(), out_re)?
    };
    if nargout >= 2 {
        let i_mat = Matrix::from_nd_col_major(dims, out_idx)?;
        Ok(vec![Value::matrix(v_mat), Value::matrix(i_mat)])
    } else {
        ok1(Value::matrix(v_mat))
    }
}

fn fn_cumsum(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 4 {
        return Err(MatroError::arity(
            "cumsum",
            "1 to 4 (A [, dim [, direction [, nanflag]]])",
            args.len(),
        ));
    }
    let m = args[0].to_dense_matrix()?;
    let mut reverse = false;
    let mut omitnan = false;
    let mut dim_opt: Option<usize> = None;
    for a in &args[1..] {
        if let Some(s) = value_as_text(a) {
            if s.eq_ignore_ascii_case("reverse") {
                reverse = true;
            } else if s.eq_ignore_ascii_case("forward") {
                reverse = false;
            } else if s.eq_ignore_ascii_case("omitnan") || s.eq_ignore_ascii_case("omitmissing") {
                omitnan = true;
            } else if s.eq_ignore_ascii_case("includenan")
                || s.eq_ignore_ascii_case("includemissing")
            {
                omitnan = false;
            } else {
                return Err(MatroError::runtime(format!("cumsum: unknown option '{s}'")));
            }
        } else {
            let d = as_usize(a, "cumsum")?;
            if d == 0 {
                return Err(MatroError::runtime("cumsum: dimension must be >= 1"));
            }
            dim_opt = Some(d);
        }
    }
    let dims = m.dims();
    let dim = dim_opt.unwrap_or_else(|| dims.iter().position(|&d| d != 1).map_or(1, |i| i + 1));
    let ax = dim - 1;
    let total = m.len();
    if total == 0 {
        return ok1(Value::matrix(m));
    }
    if ax >= dims.len() || dims[ax] <= 1 {
        if !omitnan {
            return ok1(Value::matrix(m));
        }
    }

    let inner: usize = if ax < dims.len() {
        dims[..ax].iter().product()
    } else {
        total
    };
    let axis_len = if ax < dims.len() { dims[ax] } else { 1 };
    let outer: usize = if ax < dims.len() {
        dims[ax + 1..].iter().product()
    } else {
        1
    };
    let is_complex = m.is_complex();
    let mut out_re = vec![0.0; total];
    let mut out_im = if is_complex { vec![0.0; total] } else { Vec::new() };

    for o in 0..outer {
        let block = o * axis_len * inner;
        for i in 0..inner {
            let mut acc = Complex::ZERO;
            if reverse {
                for k in (0..axis_len).rev() {
                    let pos = block + k * inner + i;
                    let v = m.get_linear_c(pos);
                    if !(omitnan && (v.re.is_nan() || v.im.is_nan())) {
                        acc = acc + v;
                    }
                    out_re[pos] = acc.re;
                    if is_complex {
                        out_im[pos] = acc.im;
                    }
                }
            } else {
                for k in 0..axis_len {
                    let pos = block + k * inner + i;
                    let v = m.get_linear_c(pos);
                    if !(omitnan && (v.re.is_nan() || v.im.is_nan())) {
                        acc = acc + v;
                    }
                    out_re[pos] = acc.re;
                    if is_complex {
                        out_im[pos] = acc.im;
                    }
                }
            }
        }
    }

    let res = if is_complex {
        Matrix::from_nd_complex_col_major(dims, out_re, out_im)?
    } else {
        Matrix::from_nd_col_major(dims, out_re)?
    };
    ok1(Value::matrix(res))
}

fn fn_diff(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity(
            "diff",
            "1 to 3 (X [, n [, dim]])",
            args.len(),
        ));
    }
    let m = args[0].to_dense_matrix()?;
    let order = if let Some(n_val) = args.get(1) {
        if n_val.to_dense_matrix().is_ok_and(|nm| nm.is_empty()) {
            1
        } else {
            as_usize(n_val, "diff")?
        }
    } else {
        1
    };
    let dim_opt = if let Some(d_val) = args.get(2) {
        let d = as_usize(d_val, "diff")?;
        if d == 0 {
            return Err(MatroError::runtime("diff: dimension must be >= 1"));
        }
        Some(d)
    } else {
        None
    };

    if order == 0 {
        return ok1(Value::matrix(m));
    }

    let mut cur_dims = m.dims();
    let mut cur_re = m.data().to_vec();
    let mut cur_im = m.imag_data().map(|s| s.to_vec());

    for _ in 0..order {
        let ax = if let Some(d) = dim_opt {
            d - 1
        } else if let Some(i) = cur_dims.iter().position(|&sz| sz != 1) {
            i
        } else {
            return ok1(Value::empty());
        };

        if ax >= cur_dims.len() {
            while cur_dims.len() <= ax {
                cur_dims.push(1);
            }
            cur_dims[ax] = 0;
            cur_re.clear();
            if let Some(im) = &mut cur_im {
                im.clear();
            }
            break;
        }

        let axis_len = cur_dims[ax];
        if axis_len <= 1 {
            cur_dims[ax] = 0;
            cur_re.clear();
            if let Some(im) = &mut cur_im {
                im.clear();
            }
            break;
        }

        let inner: usize = cur_dims[..ax].iter().product();
        let outer: usize = cur_dims[ax + 1..].iter().product();
        let new_axis_len = axis_len - 1;
        let new_total = inner * new_axis_len * outer;

        let mut next_re = vec![0.0; new_total];
        if let Some(im) = &cur_im {
            let mut next_im = vec![0.0; new_total];
            for o in 0..outer {
                let src_o = o * axis_len * inner;
                let dst_o = o * new_axis_len * inner;
                for k in 0..new_axis_len {
                    let s0 = src_o + k * inner;
                    let s1 = s0 + inner;
                    let d0 = dst_o + k * inner;
                    for i in 0..inner {
                        next_re[d0 + i] = cur_re[s1 + i] - cur_re[s0 + i];
                        next_im[d0 + i] = im[s1 + i] - im[s0 + i];
                    }
                }
            }
            cur_im = Some(next_im);
        } else {
            for o in 0..outer {
                let src_o = o * axis_len * inner;
                let dst_o = o * new_axis_len * inner;
                for k in 0..new_axis_len {
                    let s0 = src_o + k * inner;
                    let s1 = s0 + inner;
                    let d0 = dst_o + k * inner;
                    for i in 0..inner {
                        next_re[d0 + i] = cur_re[s1 + i] - cur_re[s0 + i];
                    }
                }
            }
        }
        cur_re = next_re;
        cur_dims[ax] = new_axis_len;
    }

    let res = if let Some(im) = cur_im {
        Matrix::from_nd_complex_col_major(cur_dims, cur_re, im)?
    } else {
        Matrix::from_nd_col_major(cur_dims, cur_re)?
    };
    ok1(Value::matrix(res))
}

fn fn_trace(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("trace", "1", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let n = m.rows().min(m.cols());
    let mut s = Complex::ZERO;
    for i in 0..n {
        s = s + m.get_c(i, i);
    }
    ok1(Value::matrix(Matrix::complex_scalar(s).normalize()))
}

fn fn_dot(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity("dot", "2 or 3 (A, B [, dim])", args.len()));
    }
    let a = args[0].to_dense_matrix()?;
    let b = args[1].to_dense_matrix()?;
    let a_is_vec = a.ndims() == 2 && (a.rows() == 1 || a.cols() == 1);
    let b_is_vec = b.ndims() == 2 && (b.rows() == 1 || b.cols() == 1);
    if args.len() == 2 && a_is_vec && b_is_vec {
        if a.len() != b.len() {
            return Err(MatroError::dimension("dot: vectors must have the same length"));
        }
        let mut s = Complex::ZERO;
        for i in 0..a.len() {
            s = s + a.get_linear_c(i).conj() * b.get_linear_c(i);
        }
        return ok1(Value::matrix(Matrix::complex_scalar(s).normalize()));
    }
    if a.dims() != b.dims() {
        return Err(MatroError::dimension("dot: A and B must have the same dimensions"));
    }
    let dims = a.dims();
    let dim = if args.len() == 3 {
        let d = as_usize(&args[2], "dot")?;
        if d == 0 {
            return Err(MatroError::runtime("dot: dimension must be >= 1"));
        }
        d
    } else {
        dims.iter().position(|&d| d != 1).map_or(1, |i| i + 1)
    };
    let ax = dim - 1;
    if ax >= dims.len() {
        let out = a.elementwise_c(&b, |x, y| x.conj() * y)?;
        return ok1(Value::matrix(out));
    }
    let inner: usize = dims[..ax].iter().product();
    let axis_len = dims[ax];
    let outer: usize = dims[ax + 1..].iter().product();
    let mut out_dims = dims.clone();
    out_dims[ax] = 1;
    let out_total = inner * outer;
    let mut re = vec![0.0; out_total];
    let mut im = vec![0.0; out_total];
    for o in 0..outer {
        let src_block = o * axis_len * inner;
        let dst_block = o * inner;
        for i in 0..inner {
            let mut s = Complex::ZERO;
            for k in 0..axis_len {
                let idx = src_block + k * inner + i;
                s = s + a.get_linear_c(idx).conj() * b.get_linear_c(idx);
            }
            re[dst_block + i] = s.re;
            im[dst_block + i] = s.im;
        }
    }
    ok1(Value::matrix(
        Matrix::from_nd_complex_col_major(out_dims, re, im)?.normalize(),
    ))
}

fn matrix_norm_p(m: &Matrix, p_val: &Value) -> Result<f64> {
    if m.is_empty() {
        return Ok(0.0);
    }
    let is_vec = m.ndims() == 2 && (m.rows() == 1 || m.cols() == 1);
    if let Some(s) = value_as_text(p_val) {
        match s.trim().to_ascii_lowercase().as_str() {
            "fro" => {
                let sq: f64 = (0..m.len()).map(|i| m.get_linear_c(i).norm_sq()).sum();
                return Ok(sq.sqrt());
            }
            "inf" => {
                if is_vec {
                    let mx = (0..m.len())
                        .map(|i| m.get_linear_c(i).abs())
                        .fold(0.0_f64, f64::max);
                    return Ok(mx);
                }
                // Matrix Inf-norm = max row sum of |a_ij|
                let mut mx = 0.0_f64;
                for r in 0..m.rows() {
                    let rs: f64 = (0..m.cols()).map(|c| m.get_c(r, c).abs()).sum();
                    if rs > mx || rs.is_nan() {
                        mx = rs;
                    }
                }
                return Ok(mx);
            }
            other => {
                return Err(MatroError::runtime(format!(
                    "norm: unknown norm option '{other}'"
                )));
            }
        }
    }
    let p = p_val.as_scalar()?;
    if is_vec {
        if p == 2.0 {
            let sq: f64 = (0..m.len()).map(|i| m.get_linear_c(i).norm_sq()).sum();
            Ok(sq.sqrt())
        } else if p == 1.0 {
            Ok((0..m.len()).map(|i| m.get_linear_c(i).abs()).sum())
        } else if p == f64::INFINITY {
            Ok((0..m.len())
                .map(|i| m.get_linear_c(i).abs())
                .fold(0.0_f64, f64::max))
        } else if p == f64::NEG_INFINITY {
            Ok((0..m.len())
                .map(|i| m.get_linear_c(i).abs())
                .fold(f64::INFINITY, f64::min))
        } else {
            let s: f64 = (0..m.len())
                .map(|i| m.get_linear_c(i).abs().powf(p))
                .sum();
            Ok(s.powf(1.0 / p))
        }
    } else if p == 1.0 {
        let mut mx = 0.0_f64;
        for c in 0..m.cols() {
            let cs: f64 = (0..m.rows()).map(|r| m.get_c(r, c).abs()).sum();
            if cs > mx || cs.is_nan() {
                mx = cs;
            }
        }
        Ok(mx)
    } else if p == f64::INFINITY {
        let mut mx = 0.0_f64;
        for r in 0..m.rows() {
            let rs: f64 = (0..m.cols()).map(|c| m.get_c(r, c).abs()).sum();
            if rs > mx || rs.is_nan() {
                mx = rs;
            }
        }
        Ok(mx)
    } else if p == 2.0 {
        let (_, s, _) = m.svd()?;
        let k = s.rows().min(s.cols());
        Ok(if k == 0 { 0.0 } else { s.get(0, 0) })
    } else {
        Err(MatroError::runtime(format!(
            "norm: matrix norm only supports p = 1, 2, Inf, or 'fro' (got {p})"
        )))
    }
}

fn fn_norm(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("norm", "1 or 2 (A [, p])", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    if args.len() == 2 {
        return ok1(Value::scalar(matrix_norm_p(&m, &args[1])?));
    }
    if m.is_empty() {
        return ok1(Value::scalar(0.0));
    }
    let is_vec = m.ndims() == 2 && (m.rows() == 1 || m.cols() == 1);
    if is_vec {
        let sq: f64 = (0..m.len()).map(|i| m.get_linear_c(i).norm_sq()).sum();
        ok1(Value::scalar(sq.sqrt()))
    } else {
        let (_, s, _) = m.svd()?;
        let k = s.rows().min(s.cols());
        ok1(Value::scalar(if k == 0 { 0.0 } else { s.get(0, 0) }))
    }
}

fn complex_gt(a: Complex, b: Complex) -> bool {
    if a.im == 0.0 && b.im == 0.0 {
        a.re > b.re
    } else {
        let ma = a.abs();
        let mb = b.abs();
        if (ma - mb).abs() > 1e-15 * ma.max(mb).max(1.0) {
            ma > mb
        } else {
            a.arg() > b.arg()
        }
    }
}

fn complex_lt(a: Complex, b: Complex) -> bool {
    if a.im == 0.0 && b.im == 0.0 {
        a.re < b.re
    } else {
        let ma = a.abs();
        let mb = b.abs();
        if (ma - mb).abs() > 1e-15 * ma.max(mb).max(1.0) {
            ma < mb
        } else {
            a.arg() < b.arg()
        }
    }
}

fn fn_minmax(name: &str, args: &[Value], nargout: usize, is_max: bool) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 5 {
        return Err(MatroError::arity(name, "1 to 5", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    // Element-wise min(A, B [, nanflag]) when 2nd arg is non-empty numeric
    let second_is_data = args.len() >= 2
        && value_as_text(&args[1]).is_none()
        && !args[1].to_dense_matrix()?.is_empty();
    if second_is_data {
        let b = args[1].to_dense_matrix()?;
        let mut omit_nan = true;
        for a in &args[2..] {
            if let Some(s) = value_as_text(a) {
                match s.trim().to_ascii_lowercase().as_str() {
                    "includenan" | "includemissing" => omit_nan = false,
                    "omitnan" | "omitmissing" => omit_nan = true,
                    _ => {}
                }
            }
        }
        let out = m.elementwise_c(&b, |x, y| {
            let xn = x.re.is_nan() || x.im.is_nan();
            let yn = y.re.is_nan() || y.im.is_nan();
            if xn && yn {
                return Complex::from_real(f64::NAN);
            }
            if xn {
                return if omit_nan { y } else { Complex::from_real(f64::NAN) };
            }
            if yn {
                return if omit_nan { x } else { Complex::from_real(f64::NAN) };
            }
            if (is_max && complex_gt(x, y)) || (!is_max && complex_lt(x, y)) {
                x
            } else {
                y
            }
        })?;
        return ok1(Value::matrix(out));
    }

    let mut all_mode = false;
    let mut omit_nan = true; // MATLAB default for min/max is 'omitnan'
    let mut vecdim_opt: Option<Vec<usize>> = None;
    for (idx, a) in args.iter().enumerate().skip(1) {
        if let Some(s) = value_as_text(a) {
            match s.trim().to_ascii_lowercase().as_str() {
                "all" => all_mode = true,
                "omitnan" | "omitmissing" => omit_nan = true,
                "includenan" | "includemissing" => omit_nan = false,
                _ => {}
            }
        } else if idx >= 2 && !a.to_dense_matrix()?.is_empty() {
            vecdim_opt = Some(parse_vecdim(a, name)?);
        }
    }

    if m.is_empty() {
        return if nargout >= 2 {
            Ok(vec![Value::matrix(m.clone()), Value::matrix(Matrix::zeros_nd(m.dims()))])
        } else {
            ok1(Value::matrix(m))
        };
    }

    let best_in_slice = |xs: &[Complex]| -> (Complex, f64) {
        if !omit_nan {
            if let Some(nan_i) = xs.iter().position(|c| c.re.is_nan() || c.im.is_nan()) {
                return (Complex::from_real(f64::NAN), (nan_i + 1) as f64);
            }
        }
        let mut best_v = Complex::from_real(f64::NAN);
        let mut best_i = 0usize;
        let mut have = false;
        for (i, &v) in xs.iter().enumerate() {
            if v.re.is_nan() || v.im.is_nan() {
                continue;
            }
            if !have || (is_max && complex_gt(v, best_v)) || (!is_max && complex_lt(v, best_v)) {
                best_v = v;
                best_i = i;
                have = true;
            }
        }
        (best_v, (best_i + 1) as f64)
    };

    let m_dims = m.dims();
    let is_vec = m.ndims() == 2 && (m.rows() == 1 || m.cols() == 1);
    if all_mode || (vecdim_opt.is_none() && is_vec) {
        let vals: Vec<Complex> = (0..m.len()).map(|i| m.get_linear_c(i)).collect();
        let (v, idx) = best_in_slice(&vals);
        return if nargout >= 2 {
            Ok(vec![
                Value::matrix(Matrix::complex_scalar(v).normalize()),
                Value::scalar(idx),
            ])
        } else {
            ok1(Value::matrix(Matrix::complex_scalar(v).normalize()))
        };
    }

    let target_dims = vecdim_opt
        .unwrap_or_else(|| vec![m_dims.iter().position(|&d| d != 1).map_or(1, |i| i + 1)]);
    if target_dims.len() == 1 {
        let dim = target_dims[0];
        let ax = dim - 1;
        if ax >= m_dims.len() {
            return if nargout >= 2 {
                Ok(vec![
                    Value::matrix(m),
                    Value::matrix(Matrix::ones_nd(m_dims)),
                ])
            } else {
                ok1(Value::matrix(m))
            };
        }
        let inner: usize = m_dims[..ax].iter().product();
        let axis_len = m_dims[ax];
        let outer: usize = m_dims[ax + 1..].iter().product();
        let mut out_dims = m_dims.clone();
        out_dims[ax] = 1;
        let out_total = inner * outer;
        let mut out_re = Vec::with_capacity(out_total);
        let mut out_im = Vec::with_capacity(out_total);
        let mut out_idx = Vec::with_capacity(out_total);
        let mut buf = Vec::with_capacity(axis_len);
        for o in 0..outer {
            let block = o * axis_len * inner;
            for i in 0..inner {
                buf.clear();
                for k in 0..axis_len {
                    buf.push(m.get_linear_c(block + k * inner + i));
                }
                let (v, idx) = best_in_slice(&buf);
                out_re.push(v.re);
                out_im.push(v.im);
                out_idx.push(idx);
            }
        }
        let v_mat = Matrix::from_nd_complex_col_major(out_dims.clone(), out_re, out_im)?.normalize();
        if nargout >= 2 {
            let i_mat = Matrix::from_nd_col_major(out_dims, out_idx)?;
            return Ok(vec![Value::matrix(v_mat), Value::matrix(i_mat)]);
        }
        return ok1(Value::matrix(v_mat));
    }

    // Multi-axis vecdim reduction
    let mut cur = m;
    for dim in target_dims {
        let cur_dims = cur.dims();
        let ax = dim - 1;
        if ax >= cur_dims.len() {
            continue;
        }
        let inner: usize = cur_dims[..ax].iter().product();
        let axis_len = cur_dims[ax];
        let outer: usize = cur_dims[ax + 1..].iter().product();
        let mut out_dims = cur_dims.clone();
        out_dims[ax] = 1;
        let out_total = inner * outer;
        let mut out_re = Vec::with_capacity(out_total);
        let mut out_im = Vec::with_capacity(out_total);
        let mut buf = Vec::with_capacity(axis_len);
        for o in 0..outer {
            let block = o * axis_len * inner;
            for i in 0..inner {
                buf.clear();
                for k in 0..axis_len {
                    buf.push(cur.get_linear_c(block + k * inner + i));
                }
                let (v, _) = best_in_slice(&buf);
                out_re.push(v.re);
                out_im.push(v.im);
            }
        }
        cur = Matrix::from_nd_complex_col_major(out_dims, out_re, out_im)?.normalize();
    }
    ok1(Value::matrix(cur))
}

fn fn_reduce_c(name: &str, args: &[Value], op: impl Fn(&[Complex]) -> Complex) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 4 {
        return Err(MatroError::arity(name, "1 to 4", args.len()));
    }
    let m = args[0].to_dense_matrix()?;
    let mut all_mode = false;
    let mut omit_nan = false;
    let mut vecdim_opt: Option<Vec<usize>> = None;
    for a in &args[1..] {
        if let Some(s) = value_as_text(a) {
            match s.trim().to_ascii_lowercase().as_str() {
                "all" => all_mode = true,
                "omitnan" | "omitmissing" => omit_nan = true,
                "includenan" | "includemissing" => omit_nan = false,
                "double" | "default" | "native" => {}
                _ => {}
            }
        } else {
            vecdim_opt = Some(parse_vecdim(a, name)?);
        }
    }
    let apply = |vals: &[Complex]| -> Complex {
        if omit_nan {
            let filtered: Vec<Complex> = vals
                .iter()
                .copied()
                .filter(|c| !c.re.is_nan() && !c.im.is_nan())
                .collect();
            op(&filtered)
        } else {
            op(vals)
        }
    };
    let m_dims = m.dims();
    if all_mode || (vecdim_opt.is_none() && m.ndims() == 2 && (m.rows() == 1 || m.cols() == 1 || (m.rows() == 0 && m.cols() == 0))) {
        let vals: Vec<Complex> = (0..m.len()).map(|i| m.get_linear_c(i)).collect();
        return ok1(Value::matrix(Matrix::complex_scalar(apply(&vals)).normalize()));
    }
    let target_dims = vecdim_opt
        .unwrap_or_else(|| vec![m_dims.iter().position(|&d| d != 1).map_or(1, |i| i + 1)]);
    // For mean over multiple axes, compute sum / count in one pass across all reduced axes so empty/omitnan weights stay exact
    let nd = m_dims.len();
    let mut reduce_mask = vec![false; nd];
    for &d in &target_dims {
        if d >= 1 && d <= nd {
            reduce_mask[d - 1] = true;
        }
    }
    let mut out_dims = m_dims.clone();
    for ax in 0..nd {
        if reduce_mask[ax] {
            out_dims[ax] = 1;
        }
    }
    let out_total: usize = out_dims.iter().product();
    let mut buckets: Vec<Vec<Complex>> = vec![Vec::new(); out_total];
    let mut out_strides = vec![1usize; nd];
    for ax in 1..nd {
        out_strides[ax] = out_strides[ax - 1] * out_dims[ax - 1];
    }
    let mut coord = vec![0usize; nd];
    for lin in 0..m.len() {
        let mut dst = 0usize;
        for ax in 0..nd {
            if !reduce_mask[ax] {
                dst += coord[ax] * out_strides[ax];
            }
        }
        buckets[dst].push(m.get_linear_c(lin));
        for ax in 0..nd {
            coord[ax] += 1;
            if coord[ax] < m_dims[ax] {
                break;
            }
            coord[ax] = 0;
        }
    }
    let mut out_re = Vec::with_capacity(out_total);
    let mut out_im = Vec::with_capacity(out_total);
    for b in &buckets {
        let v = apply(b);
        out_re.push(v.re);
        out_im.push(v.im);
    }
    ok1(Value::matrix(
        Matrix::from_nd_complex_col_major(out_dims, out_re, out_im)?.normalize(),
    ))
}

fn fn_elem2(name: &str, args: &[Value], f: fn(f64, f64) -> f64) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity(name, "2", args.len()));
    }
    let a = args[0].to_dense_matrix()?;
    let b = args[1].to_dense_matrix()?;
    ok1(Value::matrix(a.elementwise(&b, f)?))
}

fn fn_disp(args: &[Value], stdout: &mut String) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("disp", "1", args.len()));
    }
    stdout.push_str(&format!("{}\n", args[0]));
    Ok(vec![])
}

fn fn_fprintf(args: &[Value], nargout: usize, stdout: &mut String) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("fprintf", ">= 1", 0));
    }
    // Support fprintf(1, fmt, ...) / fprintf(2, fmt, ...)
    let fmt_args = if args.len() >= 2 && value_as_text(&args[0]).is_none() {
        let fid = args[0].as_scalar()?;
        if (fid - 1.0).abs() < 1e-12 || (fid - 2.0).abs() < 1e-12 {
            &args[1..]
        } else {
            return Err(MatroError::runtime(format!(
                "fprintf: invalid file identifier {fid}"
            )));
        }
    } else {
        args
    };
    let s = format_printf(fmt_args)?;
    let nbytes = s.len();
    stdout.push_str(&s);
    if nargout >= 1 {
        ok1(Value::scalar(nbytes as f64))
    } else {
        Ok(vec![])
    }
}

fn format_g_value(v: f64, sig: usize, upper: bool) -> String {
    if !v.is_finite() {
        return format!("{v}");
    }
    let p = sig.max(1);
    if v == 0.0 {
        return "0".to_string();
    }
    let exp = v.abs().log10().floor() as i32;
    let s = if exp < -4 || exp >= (p as i32) {
        let dec = p.saturating_sub(1);
        let raw = if upper {
            format!("{v:.dec$E}")
        } else {
            format!("{v:.dec$e}")
        };
        // Strip trailing zeros in mantissa before e/E
        let e_char = if upper { 'E' } else { 'e' };
        if let Some(pos) = raw.find(e_char) {
            let mant = raw[..pos].trim_end_matches('0').trim_end_matches('.');
            let exp_part = &raw[pos + 1..];
            let exp_num: i32 = exp_part.parse().unwrap_or(0);
            format!("{mant}{e_char}{exp_num:+03}")
        } else {
            raw
        }
    } else {
        let dec = (p as i32 - 1 - exp).max(0) as usize;
        let raw = format!("{v:.dec$}");
        if raw.contains('.') {
            raw.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            raw
        }
    };
    s
}

enum PrintfAtom {
    Num(f64),
    Str(String),
}

fn flatten_printf_atoms(args: &[Value]) -> Vec<PrintfAtom> {
    let mut atoms = Vec::new();
    for v in args {
        match v {
            Value::Matrix(m) => {
                for &x in m.data() {
                    atoms.push(PrintfAtom::Num(x));
                }
            }
            Value::Sparse(s) => {
                let d = s.to_dense();
                for &x in d.data() {
                    atoms.push(PrintfAtom::Num(x));
                }
            }
            Value::String(s) => atoms.push(PrintfAtom::Str(s.clone())),
            Value::Object(o) if o.class_name == "string" => {
                if let Some(Value::Cell(c)) = o.fields.get("Data") {
                    for item in &c.data {
                        atoms.push(PrintfAtom::Str(
                            value_as_text(item).unwrap_or_else(|| format!("{item}")),
                        ));
                    }
                }
            }
            other => atoms.push(PrintfAtom::Str(format!("{other}"))),
        }
    }
    atoms
}

fn apply_width_pad(s: String, width: Option<usize>, left_align: bool, zero_pad: bool) -> String {
    let Some(w) = width else {
        return s;
    };
    let len = s.chars().count();
    if len >= w {
        return s;
    }
    let pad = w - len;
    if left_align {
        format!("{s}{}", " ".repeat(pad))
    } else if zero_pad && !s.contains("NaN") && !s.contains("Inf") {
        if let Some(rest) = s.strip_prefix('-').or_else(|| s.strip_prefix('+')).or_else(|| s.strip_prefix(' ')) {
            let sign = &s[..1];
            format!("{sign}{}{rest}", "0".repeat(pad))
        } else {
            format!("{}{s}", "0".repeat(pad))
        }
    } else {
        format!("{}{s}", " ".repeat(pad))
    }
}

pub fn format_printf(args: &[Value]) -> Result<String> {
    let fmt_str = value_as_text(&args[0])
        .ok_or_else(|| MatroError::runtime("printf: format specifier must be a string"))?;
    let atoms = flatten_printf_atoms(&args[1..]);
    let chars: Vec<char> = fmt_str.chars().collect();
    let mut out = String::new();
    let mut atom_idx = 0usize;

    loop {
        let start_atom = atom_idx;
        let mut i = 0usize;
        while i < chars.len() {
            if chars[i] == '\\' && i + 1 < chars.len() {
                match chars[i + 1] {
                    'n' => out.push('\n'),
                    't' => out.push('\t'),
                    'r' => out.push('\r'),
                    '\\' => out.push('\\'),
                    other => {
                        out.push('\\');
                        out.push(other);
                    }
                }
                i += 2;
            } else if chars[i] == '%' && i + 1 < chars.len() {
                if chars[i + 1] == '%' {
                    out.push('%');
                    i += 2;
                    continue;
                }
                let mut j = i + 1;
                let mut left_align = false;
                let mut plus_sign = false;
                let mut space_sign = false;
                let mut zero_pad = false;
                let mut alt_form = false;
                while j < chars.len() {
                    match chars[j] {
                        '-' => left_align = true,
                        '+' => plus_sign = true,
                        ' ' => space_sign = true,
                        '0' => zero_pad = true,
                        '#' => alt_form = true,
                        _ => break,
                    }
                    j += 1;
                }
                let mut width: Option<usize> = None;
                if j < chars.len() && chars[j] == '*' {
                    if let Some(PrintfAtom::Num(w)) = atoms.get(atom_idx) {
                        atom_idx += 1;
                        let wi = w.round() as isize;
                        if wi < 0 {
                            left_align = true;
                            width = Some((-wi) as usize);
                        } else {
                            width = Some(wi as usize);
                        }
                    }
                    j += 1;
                } else {
                    let w_start = j;
                    while j < chars.len() && chars[j].is_ascii_digit() {
                        j += 1;
                    }
                    if j > w_start {
                        let s: String = chars[w_start..j].iter().collect();
                        width = s.parse().ok();
                    }
                }
                let mut precision: Option<usize> = None;
                if j < chars.len() && chars[j] == '.' {
                    j += 1;
                    if j < chars.len() && chars[j] == '*' {
                        if let Some(PrintfAtom::Num(p)) = atoms.get(atom_idx) {
                            atom_idx += 1;
                            precision = Some(p.max(0.0).round() as usize);
                        }
                        j += 1;
                    } else {
                        let p_start = j;
                        while j < chars.len() && chars[j].is_ascii_digit() {
                            j += 1;
                        }
                        if j > p_start {
                            let s: String = chars[p_start..j].iter().collect();
                            precision = s.parse().ok();
                        } else {
                            precision = Some(0);
                        }
                    }
                }
                // Optional length modifier (l, h, etc.)
                while j < chars.len() && matches!(chars[j], 'l' | 'h' | 'b' | 't') {
                    j += 1;
                }
                let spec = chars.get(j).copied().unwrap_or('s');
                if atom_idx >= atoms.len() {
                    return Ok(out);
                }
                let atom = &atoms[atom_idx];
                atom_idx += 1;
                let sign_prefix = |v: f64, body: String| -> String {
                    if v >= 0.0 && !v.is_nan() && !body.starts_with('-') {
                        if plus_sign {
                            return format!("+{body}");
                        } else if space_sign {
                            return format!(" {body}");
                        }
                    }
                    body
                };
                let formatted = match spec {
                    'd' | 'i' => {
                        let v = match atom {
                            PrintfAtom::Num(x) => *x,
                            PrintfAtom::Str(s) => s.chars().next().map_or(0.0, |c| c as u32 as f64),
                        };
                        let iv = v.round() as i64;
                        let body = sign_prefix(v, format!("{iv}"));
                        apply_width_pad(body, width, left_align, zero_pad)
                    }
                    'u' => {
                        let v = match atom {
                            PrintfAtom::Num(x) => *x,
                            PrintfAtom::Str(s) => s.chars().next().map_or(0.0, |c| c as u32 as f64),
                        };
                        apply_width_pad(format!("{}", v.max(0.0).round() as u64), width, left_align, zero_pad)
                    }
                    'x' | 'X' => {
                        let v = match atom {
                            PrintfAtom::Num(x) => *x,
                            PrintfAtom::Str(s) => s.chars().next().map_or(0.0, |c| c as u32 as f64),
                        };
                        let u = v.max(0.0).round() as u64;
                        let raw = if spec == 'X' {
                            if alt_form && u != 0 { format!("0X{u:X}") } else { format!("{u:X}") }
                        } else if alt_form && u != 0 {
                            format!("0x{u:x}")
                        } else {
                            format!("{u:x}")
                        };
                        apply_width_pad(raw, width, left_align, zero_pad)
                    }
                    'o' => {
                        let v = match atom {
                            PrintfAtom::Num(x) => *x,
                            PrintfAtom::Str(s) => s.chars().next().map_or(0.0, |c| c as u32 as f64),
                        };
                        let u = v.max(0.0).round() as u64;
                        let raw = if alt_form && u != 0 { format!("0{u:o}") } else { format!("{u:o}") };
                        apply_width_pad(raw, width, left_align, zero_pad)
                    }
                    'f' | 'F' => {
                        let v = match atom {
                            PrintfAtom::Num(x) => *x,
                            PrintfAtom::Str(s) => s.parse().unwrap_or(f64::NAN),
                        };
                        let p = precision.unwrap_or(6);
                        let body = sign_prefix(v, format!("{v:.p$}"));
                        apply_width_pad(body, width, left_align, zero_pad)
                    }
                    'g' | 'G' => {
                        let v = match atom {
                            PrintfAtom::Num(x) => *x,
                            PrintfAtom::Str(s) => s.parse().unwrap_or(f64::NAN),
                        };
                        let p = precision.unwrap_or(6);
                        let body = sign_prefix(v, format_g_value(v, p, spec == 'G'));
                        apply_width_pad(body, width, left_align, zero_pad)
                    }
                    'e' | 'E' => {
                        let v = match atom {
                            PrintfAtom::Num(x) => *x,
                            PrintfAtom::Str(s) => s.parse().unwrap_or(f64::NAN),
                        };
                        let p = precision.unwrap_or(6);
                        let raw = if spec == 'E' {
                            format!("{v:.p$E}")
                        } else {
                            format!("{v:.p$e}")
                        };
                        let e_ch = if spec == 'E' { 'E' } else { 'e' };
                        let nice = if let Some(pos) = raw.find(e_ch) {
                            let mant = &raw[..pos];
                            let exp_num: i32 = raw[pos + 1..].parse().unwrap_or(0);
                            format!("{mant}{e_ch}{exp_num:+03}")
                        } else {
                            raw
                        };
                        let body = sign_prefix(v, nice);
                        apply_width_pad(body, width, left_align, zero_pad)
                    }
                    'c' => {
                        let ch = match atom {
                            PrintfAtom::Num(x) => char::from_u32(x.max(0.0).round() as u32).unwrap_or(' '),
                            PrintfAtom::Str(s) => s.chars().next().unwrap_or(' '),
                        };
                        apply_width_pad(ch.to_string(), width, left_align, false)
                    }
                    's' => {
                        let mut s = match atom {
                            PrintfAtom::Str(st) => st.clone(),
                            PrintfAtom::Num(x) => format!("{x}"),
                        };
                        if let Some(p) = precision {
                            s = s.chars().take(p).collect();
                        }
                        apply_width_pad(s, width, left_align, false)
                    }
                    _ => match atom {
                        PrintfAtom::Str(st) => st.clone(),
                        PrintfAtom::Num(x) => format!("{x}"),
                    },
                };
                out.push_str(&formatted);
                i = j + 1;
            } else {
                out.push(chars[i]);
                i += 1;
            }
        }
        if atom_idx >= atoms.len() || atom_idx == start_atom {
            break;
        }
    }
    Ok(out)
}

fn fn_addpath(args: &[Value], loader: &mut ModuleLoader) -> Result<Vec<Value>> {
    let prev = loader
        .search_paths()
        .iter()
        .filter(|d| !d.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join(":");
    let mut at_end = false;
    let mut dirs: Vec<&str> = Vec::new();
    for a in args {
        let s = a.as_str()?;
        match s {
            "-end" | "-END" => at_end = true,
            "-begin" | "-BEGIN" | "-frozen" => {}
            _ => dirs.push(s),
        }
    }
    if at_end {
        for d in dirs {
            loader.add_path_end(d);
        }
    } else {
        for d in dirs.into_iter().rev() {
            loader.add_path(d);
        }
    }
    ok1(Value::String(prev))
}

fn fn_rmpath(args: &[Value], loader: &mut ModuleLoader) -> Result<Vec<Value>> {
    let prev = loader
        .search_paths()
        .iter()
        .filter(|d| !d.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join(":");
    for a in args {
        loader.rm_path(a.as_str()?);
    }
    ok1(Value::String(prev))
}

fn fn_genpath(args: &[Value], loader: &ModuleLoader) -> Result<Vec<Value>> {
    let root = if let Some(a) = args.first() {
        a.as_str()?.to_string()
    } else {
        loader.vfs().cwd()
    };
    let dirs = loader.gen_path(&root);
    ok1(Value::String(dirs.join(":")))
}

fn fn_path(args: &[Value], loader: &mut ModuleLoader) -> Result<Vec<Value>> {
    if let Some(a) = args.first() {
        let new_p = a.as_str()?;
        loader.reset_search_paths();
        loader.add_path_end(new_p);
        if let Some(b) = args.get(1) {
            loader.add_path_end(b.as_str()?);
        }
    }
    let out = loader
        .search_paths()
        .iter()
        .filter(|d| !d.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join(":");
    ok1(Value::String(out))
}

fn fn_csvread(args: &[Value], loader: &ModuleLoader) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 4 {
        return Err(MatroError::arity("csvread", "1 to 4 (file [, R1, C1 [, range]])", args.len()));
    }
    if args.len() == 1 {
        let m = loader.read_csv(args[0].as_str()?)?;
        return ok1(Value::matrix(m));
    }
    let mut dargs = Vec::with_capacity(args.len() + 1);
    dargs.push(args[0].clone());
    dargs.push(Value::String(",".to_string()));
    dargs.extend_from_slice(&args[1..]);
    fs_media::fn_dlmread(&dargs, loader)
}

fn fn_csvwrite(args: &[Value], loader: &mut ModuleLoader) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 4 {
        return Err(MatroError::arity("csvwrite", "2 to 4 (file, M [, R1, C1])", args.len()));
    }
    if args.len() == 2 {
        loader.write_csv(args[0].as_str()?, &args[1].to_dense_matrix()?)?;
        return Ok(vec![]);
    }
    let mut dargs = vec![
        args[0].clone(),
        args[1].clone(),
        Value::String("delimiter".to_string()),
        Value::String(",".to_string()),
        Value::String("roffset".to_string()),
        args[2].clone(),
    ];
    if args.len() >= 4 {
        dargs.push(Value::String("coffset".to_string()));
        dargs.push(args[3].clone());
    }
    fs_media::fn_dlmwrite(&dargs, loader)
}

fn fn_fileread(args: &[Value], loader: &ModuleLoader) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("fileread", "1", args.len()));
    }
    let path = args[0].as_str()?;
    let resolved = loader.resolve_path(path).unwrap_or_else(|| path.to_string());
    let text = loader
        .vfs()
        .read_to_string(&resolved)
        .ok_or_else(|| MatroError::runtime(format!("File not found: '{path}'")))?;
    ok1(Value::String(text))
}

fn is_plot_prop_key(s: &str) -> bool {
    matches!(
        s.to_ascii_lowercase().as_str(),
        "color" | "linewidth" | "linestyle" | "displayname"
    )
}

fn fn_plot(
    kind: PlotKind,
    args: &[Value],
    nargout: usize,
    figure: &mut PlotFigure,
    log_x: bool,
    log_y: bool,
) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity(kind.as_str(), ">= 1", 0));
    }
    let panel_idx = figure.active;
    let panel = figure.panel_mut();
    panel.log_x = log_x;
    panel.log_y = log_y;

    let mut added_indices = Vec::new();
    let mut prop_pairs: Vec<(&str, &Value)> = Vec::new();

    if args.len() == 1 {
        let y = args[0].to_dense_matrix()?.data().to_vec();
        let x: Vec<f64> = (1..=y.len()).map(|i| i as f64).collect();
        panel.add_series(kind, x, y, None);
        added_indices.push(panel.series.len() - 1);
    } else {
        let orig_hold = panel.hold;
        let mut i = 0;
        let mut first = true;
        while i < args.len() {
            if let Ok(k) = args[i].as_str() {
                if is_plot_prop_key(k) && i + 1 < args.len() {
                    prop_pairs.push((k, &args[i + 1]));
                    i += 2;
                    continue;
                }
            }
            let x = args[i].to_dense_matrix()?.data().to_vec();
            let Some(y_val) = args.get(i + 1) else {
                return Err(MatroError::arity(
                    kind.as_str(),
                    "paired x, y vectors",
                    args.len(),
                ));
            };
            let y = y_val.to_dense_matrix()?.data().to_vec();
            i += 2;
            let style = if let Some(Value::String(s)) = args.get(i) {
                if !is_plot_prop_key(s) {
                    i += 1;
                    Some(s.as_str())
                } else {
                    None
                }
            } else {
                None
            };
            if !first {
                panel.hold = true;
            }
            panel.add_series(kind, x, y, style);
            added_indices.push(panel.series.len() - 1);
            first = false;
        }
        panel.hold = orig_hold;
    }

    for &(k, val) in &prop_pairs {
        let key = k.to_ascii_lowercase();
        for &si in &added_indices {
            if let Some(ser) = panel.series.get_mut(si) {
                match key.as_str() {
                    "color" => ser.color = fs_media::parse_color_val(val)?,
                    "linewidth" => ser.line_width = val.as_scalar()?.max(0.2),
                    "linestyle" => {
                        let ls = val.as_str()?;
                        ser.dashed = ls.contains("--") || ls.contains(':');
                    }
                    "displayname" => ser.label = Some(val.as_str()?.to_string()),
                    _ => {}
                }
            }
        }
    }

    if nargout >= 1 {
        if let Some(&last_si) = added_indices.last() {
            return Ok(vec![fs_media::make_line_handle(panel_idx, last_si)]);
        }
    }
    Ok(vec![])
}

fn fn_imagesc(args: &[Value], figure: &mut PlotFigure) -> Result<Vec<Value>> {
    match args.len() {
        1 => {
            figure.panel_mut().set_heatmap(&args[0].to_dense_matrix()?);
        }
        2 => {
            figure.panel_mut().set_heatmap(&args[0].to_dense_matrix()?);
        }
        3 | 4 => {
            let xm = args[0].to_dense_matrix()?;
            let ym = args[1].to_dense_matrix()?;
            let cm = args[2].to_dense_matrix()?;
            let x_ext = if !xm.is_empty() {
                let x0 = xm.data().first().copied().unwrap_or(1.0);
                let x1 = xm.data().last().copied().unwrap_or(cm.cols() as f64);
                Some((x0.min(x1), x0.max(x1)))
            } else {
                None
            };
            let y_ext = if !ym.is_empty() {
                let y0 = ym.data().first().copied().unwrap_or(1.0);
                let y1 = ym.data().last().copied().unwrap_or(cm.rows() as f64);
                Some((y0.min(y1), y0.max(y1)))
            } else {
                None
            };
            figure
                .panel_mut()
                .set_heatmap_with_extents(&cm, x_ext, y_ext);
        }
        n => return Err(MatroError::arity("imagesc", "1 to 4", n)),
    }
    Ok(vec![])
}

fn fn_axis(args: &[Value], nargout: usize, figure: &mut PlotFigure) -> Result<Vec<Value>> {
    let panel = figure.panel_mut();
    if args.is_empty() {
        if nargout >= 1 {
            let (x0, x1) = panel.xlim.unwrap_or_else(|| {
                let mut lo = f64::INFINITY;
                let mut hi = f64::NEG_INFINITY;
                for s in &panel.series {
                    for &x in &s.x {
                        if x.is_finite() {
                            lo = lo.min(x);
                            hi = hi.max(x);
                        }
                    }
                }
                if lo.is_finite() && hi.is_finite() { (lo, hi) } else { (0.0, 1.0) }
            });
            let (y0, y1) = panel.ylim.unwrap_or_else(|| {
                let mut lo = f64::INFINITY;
                let mut hi = f64::NEG_INFINITY;
                for s in &panel.series {
                    for &y in &s.y {
                        if y.is_finite() {
                            lo = lo.min(y);
                            hi = hi.max(y);
                        }
                    }
                }
                if lo.is_finite() && hi.is_finite() { (lo, hi) } else { (0.0, 1.0) }
            });
            return ok1(Value::matrix(Matrix::row_vec(vec![x0, x1, y0, y1])));
        }
        return Ok(vec![]);
    }
    for a in args {
        match a {
            Value::String(s) => match s.to_ascii_lowercase().as_str() {
                "equal" | "square" => panel.axis_equal = true,
                "tight" => {
                    panel.axis_tight = true;
                    panel.xlim = None;
                    panel.ylim = None;
                }
                "image" => {
                    panel.axis_equal = true;
                    panel.axis_tight = true;
                }
                "normal" | "auto" => {
                    panel.axis_equal = false;
                    panel.axis_tight = false;
                    panel.xlim = None;
                    panel.ylim = None;
                }
                "off" => panel.axis_off = true,
                "on" => panel.axis_off = false,
                "xy" | "ij" | "vis3d" | "fill" | "manual" | "padded" => {}
                other => {
                    return Err(MatroError::runtime(format!("axis: unknown option '{other}'")));
                }
            },
            other => {
                let m = other.to_dense_matrix()?;
                if m.len() == 4 || m.len() == 6 {
                    panel.xlim = Some((m.data()[0], m.data()[1]));
                    panel.ylim = Some((m.data()[2], m.data()[3]));
                    if m.len() == 6 {
                        panel.zlim = Some((m.data()[4], m.data()[5]));
                    }
                } else {
                    return Err(MatroError::dimension(
                        "axis: limits vector must have 4 or 6 elements",
                    ));
                }
            }
        }
    }
    Ok(vec![])
}

fn fn_colorbar(args: &[Value], figure: &mut PlotFigure) -> Result<Vec<Value>> {
    let panel = figure.panel_mut();
    if let Some(first) = args.first() {
        if let Ok(s) = first.as_str() {
            let low = s.to_ascii_lowercase();
            if low == "off" || low == "hide" {
                panel.show_colorbar = false;
                return Ok(vec![]);
            }
        }
    }
    panel.show_colorbar = true;
    Ok(vec![])
}

fn colormap_matrix(name: &str, n: usize) -> Matrix {
    let n = n.max(1);
    let mut m = Matrix::zeros(n, 3);
    for i in 0..n {
        let t = if n == 1 { 0.5 } else { i as f64 / (n - 1) as f64 };
        let (r, g, b) = match name.to_ascii_lowercase().as_str() {
            "gray" | "grey" => (t, t, t),
            "hot" => ((3.0 * t).clamp(0.0, 1.0), (3.0 * t - 1.0).clamp(0.0, 1.0), (3.0 * t - 2.0).clamp(0.0, 1.0)),
            "cool" => (t, 1.0 - t, 1.0),
            "bone" => (
                (7.0 * t + (3.0 * t - 2.0).clamp(0.0, 1.0)) / 8.0,
                (7.0 * t + (3.0 * t - 1.0).clamp(0.0, 1.0)) / 8.0,
                (7.0 * t + (3.0 * t).clamp(0.0, 1.0)) / 8.0,
            ),
            "jet" | "turbo" => (
                (1.5 - (4.0 * t - 3.0).abs()).clamp(0.0, 1.0),
                (1.5 - (4.0 * t - 2.0).abs()).clamp(0.0, 1.0),
                (1.5 - (4.0 * t - 1.0).abs()).clamp(0.0, 1.0),
            ),
            "parula" => (
                (0.24 + 0.74 * t.powf(1.4)).clamp(0.0, 1.0),
                (0.15 + 0.80 * (t * 1.15).min(1.0)).clamp(0.0, 1.0),
                (0.66 * (1.0 - t).powf(0.8) + 0.14 * t).clamp(0.0, 1.0),
            ),
            _ => {
                // viridis
                let r = (0.267 + t * (0.005 + t * 0.721)).clamp(0.0, 1.0);
                let g = (0.005 + t * (0.873 - t * 0.03)).clamp(0.0, 1.0);
                let b = (0.329 + t * (0.8 - t * 0.985)).clamp(0.0, 1.0);
                (r, g, b)
            }
        };
        m.set(i, 0, r);
        m.set(i, 1, g);
        m.set(i, 2, b);
    }
    m
}

fn fn_colormap_palette(name: &str, args: &[Value]) -> Result<Vec<Value>> {
    let n = if args.is_empty() {
        64
    } else if args.len() == 1 {
        as_usize(&args[0], name)?.max(1)
    } else {
        return Err(MatroError::arity(name, "0 or 1", args.len()));
    };
    ok1(Value::matrix(colormap_matrix(name, n)))
}

fn fn_colormap(args: &[Value], nargout: usize, figure: &mut PlotFigure) -> Result<Vec<Value>> {
    let panel = figure.panel_mut();
    if let Some(first) = args.first() {
        if let Ok(s) = first.as_str() {
            let low = s.to_ascii_lowercase();
            panel.colormap = if low == "default" {
                "parula".to_string()
            } else {
                low
            };
        }
    }
    if nargout >= 1 {
        return ok1(Value::matrix(colormap_matrix(&panel.colormap, 64)));
    }
    Ok(vec![])
}

fn fn_hist(args: &[Value], nargout: usize, figure: &mut PlotFigure) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("hist", "1 or 2", args.len()));
    }
    let ym = args[0].to_dense_matrix()?;
    let vals: Vec<f64> = ym.data().iter().copied().filter(|x| x.is_finite()).collect();
    if vals.is_empty() {
        if nargout >= 2 {
            return Ok(vec![
                Value::matrix(Matrix::zeros(1, 0)),
                Value::matrix(Matrix::zeros(1, 0)),
            ]);
        }
        return ok1(Value::matrix(Matrix::zeros(1, 0)));
    }

    let centers: Vec<f64> = if let Some(b) = args.get(1) {
        let bm = b.to_dense_matrix()?;
        if bm.len() == 1 {
            let m = (bm.data()[0].round() as usize).max(1);
            let lo = vals.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if (hi - lo).abs() < 1e-15 {
                let start = lo - (m as f64 - 1.0) * 0.5;
                (0..m).map(|i| start + i as f64).collect()
            } else {
                let step = (hi - lo) / m as f64;
                (0..m).map(|i| lo + (i as f64 + 0.5) * step).collect()
            }
        } else {
            bm.data().to_vec()
        }
    } else {
        let m = 10usize;
        let lo = vals.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if (hi - lo).abs() < 1e-15 {
            (0..m).map(|i| lo - 4.5 + i as f64).collect()
        } else {
            let step = (hi - lo) / m as f64;
            (0..m).map(|i| lo + (i as f64 + 0.5) * step).collect()
        }
    };

    let nb = centers.len().max(1);
    let mut counts = vec![0.0; nb];
    for &v in &vals {
        // Assign to nearest bin center (MATLAB hist semantics)
        let mut best_idx = 0usize;
        let mut best_dist = (v - centers[0]).abs();
        for (i, &c) in centers.iter().enumerate().skip(1) {
            let d = (v - c).abs();
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        counts[best_idx] += 1.0;
    }

    if nargout == 0 {
        figure
            .panel_mut()
            .add_series(PlotKind::Bar, centers, counts, None);
        return Ok(vec![]);
    }
    if nargout == 1 {
        return ok1(Value::matrix(Matrix::row_vec(counts)));
    }
    Ok(vec![
        Value::matrix(Matrix::row_vec(counts)),
        Value::matrix(Matrix::row_vec(centers)),
    ])
}

fn fn_spy(args: &[Value], figure: &mut PlotFigure) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("spy", "1 or 2", args.len()));
    }
    let style = args.get(1).and_then(|v| v.as_str().ok());
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let (nrows, ncols) = match &args[0] {
        Value::Sparse(sp) => {
            for (r, c, v) in sp.triplets() {
                if v != 0.0 {
                    xs.push((c + 1) as f64);
                    ys.push(-((r + 1) as f64));
                }
            }
            (sp.rows(), sp.cols())
        }
        other => {
            let m = other.to_dense_matrix()?;
            for c in 0..m.cols() {
                for r in 0..m.rows() {
                    if !m.get_c(r, c).is_zero() {
                        xs.push((c + 1) as f64);
                        ys.push(-((r + 1) as f64));
                    }
                }
            }
            (m.rows(), m.cols())
        }
    };
    let nnz = xs.len();
    let panel = figure.panel_mut();
    panel.add_series(PlotKind::Scatter, xs, ys, style);
    panel.xlim = Some((0.0, (ncols + 1) as f64));
    panel.ylim = Some((-((nrows + 1) as f64), 0.0));
    panel.xlabel = format!("nz = {nnz}");
    Ok(vec![])
}

fn fn_errorbar(args: &[Value], nargout: usize, figure: &mut PlotFigure) -> Result<Vec<Value>> {
    if args.len() < 2 {
        return Err(MatroError::arity("errorbar", ">= 2", args.len()));
    }
    let panel_idx = figure.active;
    let mut num_args = Vec::new();
    let mut style: Option<&str> = None;
    let mut props: Vec<(&str, &Value)> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if let Ok(k) = args[i].as_str() {
            if is_plot_prop_key(k) && i + 1 < args.len() {
                props.push((k, &args[i + 1]));
                i += 2;
                continue;
            }
            style = Some(k);
            i += 1;
            continue;
        }
        num_args.push(args[i].to_dense_matrix()?.data().to_vec());
        i += 1;
    }
    let (x, y, err) = match num_args.len() {
        2 => {
            let y = num_args.remove(0);
            let e = num_args.remove(0);
            let x: Vec<f64> = (1..=y.len()).map(|k| k as f64).collect();
            (x, y, e)
        }
        3 => {
            let x = num_args.remove(0);
            let y = num_args.remove(0);
            let e = num_args.remove(0);
            (x, y, e)
        }
        4 => {
            let x = num_args.remove(0);
            let y = num_args.remove(0);
            let neg = num_args.remove(0);
            let pos = num_args.remove(0);
            let e: Vec<f64> = neg
                .iter()
                .zip(pos.iter())
                .map(|(a, b)| 0.5 * (a.abs() + b.abs()))
                .collect();
            (x, y, e)
        }
        n => return Err(MatroError::arity("errorbar", "2, 3, or 4 numeric arrays", n)),
    };
    let panel = figure.panel_mut();
    panel.add_errorbar(x, y, err, style);
    let si = panel.series.len() - 1;
    for (k, val) in props {
        if let Some(ser) = panel.series.get_mut(si) {
            match k.to_ascii_lowercase().as_str() {
                "color" => ser.color = fs_media::parse_color_val(val)?,
                "linewidth" => ser.line_width = val.as_scalar()?.max(0.2),
                "displayname" => ser.label = Some(val.as_str()?.to_string()),
                _ => {}
            }
        }
    }
    if nargout >= 1 {
        return Ok(vec![fs_media::make_line_handle(panel_idx, si)]);
    }
    Ok(vec![])
}

fn fn_fill_plot(
    name: &str,
    args: &[Value],
    nargout: usize,
    figure: &mut PlotFigure,
) -> Result<Vec<Value>> {
    if args.len() < 3 {
        return Err(MatroError::arity(name, ">= 3 (x, y, c)", args.len()));
    }
    let panel_idx = figure.active;
    let panel = figure.panel_mut();
    let orig_hold = panel.hold;
    let mut i = 0;
    let mut first = true;
    let mut last_si = None;
    while i + 2 < args.len() {
        if let Ok(k) = args[i].as_str() {
            if is_plot_prop_key(k) || k.eq_ignore_ascii_case("facecolor") || k.eq_ignore_ascii_case("edgecolor") {
                i += 2;
                continue;
            }
        }
        let x = args[i].to_dense_matrix()?.data().to_vec();
        let y = args[i + 1].to_dense_matrix()?.data().to_vec();
        let color_val = &args[i + 2];
        i += 3;
        if !first {
            panel.hold = true;
        }
        let style_str = color_val.as_str().ok();
        panel.add_series(PlotKind::Fill, x, y, style_str);
        let si = panel.series.len() - 1;
        if style_str.is_none() {
            if let Ok(col) = fs_media::parse_color_val(color_val) {
                panel.series[si].color = col;
            }
        }
        last_si = Some(si);
        first = false;
    }
    panel.hold = orig_hold;
    if nargout >= 1 {
        if let Some(si) = last_si {
            return Ok(vec![fs_media::make_line_handle(panel_idx, si)]);
        }
    }
    Ok(vec![])
}

fn fn_subplot(args: &[Value], figure: &mut PlotFigure) -> Result<Vec<Value>> {
    if args.len() != 3 {
        return Err(MatroError::arity("subplot", "3", args.len()));
    }
    let r = as_usize(&args[0], "subplot")?;
    let c = as_usize(&args[1], "subplot")?;
    let idx = as_usize(&args[2], "subplot")?;
    figure.subplot(r, c, idx);
    Ok(vec![])
}

fn fn_legend(args: &[Value], figure: &mut PlotFigure) -> Result<Vec<Value>> {
    let mut labels = Vec::new();
    for a in args {
        match a {
            Value::Cell(c) => {
                for item in &c.data {
                    labels.push(format!("{item}"));
                }
            }
            other => labels.push(format!("{other}")),
        }
    }
    figure.panel_mut().set_legend(labels);
    Ok(vec![])
}

fn fn_axis_lim(name: &str, args: &[Value], slot: &mut Option<(f64, f64)>) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity(name, "1", args.len()));
    }
    let m = args[0].as_matrix()?;
    if m.len() != 2 {
        return Err(MatroError::dimension(format!(
            "{name}: expected 2-element vector [min, max]"
        )));
    }
    *slot = Some((m.data()[0], m.data()[1]));
    Ok(vec![])
}

fn fn_plot_label(name: &str, args: &[Value], set: impl FnOnce(String)) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity(name, "1", args.len()));
    }
    set(args[0].as_str()?.to_string());
    Ok(vec![])
}

fn fn_plot_toggle(name: &str, args: &[Value], flag: &mut bool) -> Result<Vec<Value>> {
    if args.is_empty() {
        *flag = !*flag;
        return Ok(vec![]);
    }
    if args.len() != 1 {
        return Err(MatroError::arity(name, "0 or 1", args.len()));
    }
    match args[0].as_str()? {
        "on" => *flag = true,
        "off" => *flag = false,
        other => {
            return Err(MatroError::runtime(format!(
                "{name}: expected 'on' or 'off', got '{other}'"
            )));
        }
    }
    Ok(vec![])
}

fn fn_quiver(args: &[Value], figure: &mut PlotFigure) -> Result<Vec<Value>> {
    let (x, y, u, v) = match args.len() {
        2 => {
            let um = args[0].as_matrix()?;
            let vm = args[1].as_matrix()?;
            let mut xs = Vec::with_capacity(um.len());
            let mut ys = Vec::with_capacity(um.len());
            for c in 0..um.cols() {
                for r in 0..um.rows() {
                    xs.push((c + 1) as f64);
                    ys.push((r + 1) as f64);
                }
            }
            (xs, ys, um.data().to_vec(), vm.data().to_vec())
        }
        4 => (
            args[0].as_matrix()?.data().to_vec(),
            args[1].as_matrix()?.data().to_vec(),
            args[2].as_matrix()?.data().to_vec(),
            args[3].as_matrix()?.data().to_vec(),
        ),
        n => return Err(MatroError::arity("quiver", "2 or 4 (X, Y, U, V)", n)),
    };
    figure.panel_mut().set_quiver(x, y, u, v);
    Ok(vec![])
}

fn fn_plot3(args: &[Value], figure: &mut PlotFigure) -> Result<Vec<Value>> {
    if args.len() < 3 {
        return Err(MatroError::arity("plot3", ">= 3 (x, y, z [, style])", args.len()));
    }
    let panel = figure.panel_mut();
    let orig_hold = panel.hold;
    let mut i = 0;
    let mut first = true;
    while i + 2 < args.len() {
        let x = args[i].as_matrix()?.data().to_vec();
        let y = args[i + 1].as_matrix()?.data().to_vec();
        let z = args[i + 2].as_matrix()?.data().to_vec();
        i += 3;
        let style = if let Some(Value::String(s)) = args.get(i) {
            i += 1;
            Some(s.as_str())
        } else {
            None
        };
        if !first {
            panel.hold = true;
        }
        panel.add_curve_3d(x, y, z, style);
        first = false;
    }
    panel.hold = orig_hold;
    Ok(vec![])
}

fn fn_surface(
    name: &str,
    args: &[Value],
    figure: &mut PlotFigure,
    wireframe_only: bool,
) -> Result<Vec<Value>> {
    match args.len() {
        1 => {
            let z = args[0].as_matrix()?;
            let mut x = Matrix::zeros(z.rows(), z.cols());
            let mut y = Matrix::zeros(z.rows(), z.cols());
            for r in 0..z.rows() {
                for c in 0..z.cols() {
                    x.set(r, c, (c + 1) as f64);
                    y.set(r, c, (r + 1) as f64);
                }
            }
            figure.panel_mut().set_surface(&x, &y, z, wireframe_only);
            Ok(vec![])
        }
        3 => {
            let xm = args[0].as_matrix()?;
            let ym = args[1].as_matrix()?;
            let zm = args[2].as_matrix()?;
            if xm.rows() == zm.rows() && xm.cols() == zm.cols() {
                figure.panel_mut().set_surface(xm, ym, zm, wireframe_only);
            } else {
                // Allow 1D vector x, y with 2D Z
                let mut xg = Matrix::zeros(zm.rows(), zm.cols());
                let mut yg = Matrix::zeros(zm.rows(), zm.cols());
                for r in 0..zm.rows() {
                    for c in 0..zm.cols() {
                        xg.set(r, c, xm.data().get(c).copied().unwrap_or((c + 1) as f64));
                        yg.set(r, c, ym.data().get(r).copied().unwrap_or((r + 1) as f64));
                    }
                }
                figure.panel_mut().set_surface(&xg, &yg, zm, wireframe_only);
            }
            Ok(vec![])
        }
        n => Err(MatroError::arity(name, "1 or 3 (X, Y, Z)", n)),
    }
}

fn fn_view(args: &[Value], figure: &mut PlotFigure) -> Result<Vec<Value>> {
    let (az, el) = match args.len() {
        1 => {
            let m = args[0].as_matrix()?;
            if m.len() == 1 {
                let d = m.data()[0].round() as i32;
                if d == 2 { (0.0, 90.0) } else { (-37.5, 30.0) }
            } else if m.len() == 2 {
                (m.data()[0], m.data()[1])
            } else {
                return Err(MatroError::dimension("view: expected 2, 3, or [az, el]"));
            }
        }
        2 => (args[0].as_scalar()?, args[1].as_scalar()?),
        n => return Err(MatroError::arity("view", "1 or 2", n)),
    };
    let panel = figure.panel_mut();
    panel.view_az = az;
    panel.view_el = el;
    Ok(vec![])
}


fn fn_cast_class(name: &str, args: &[Value], target: NumericClass) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity(name, "1", args.len()));
    }
    match &args[0] {
        Value::String(s) => {
            let codes: Vec<f64> = s.chars().map(|c| c as u32 as f64).collect();
            ok1(Value::matrix(Matrix::row_vec(codes).cast_to(target)))
        }
        other => ok1(Value::matrix(other.to_dense_matrix()?.cast_to(target))),
    }
}

fn fn_char(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("char", "1", args.len()));
    }
    match &args[0] {
        Value::String(s) => ok1(Value::String(s.clone())),
        Value::Object(o) if o.class_name == "string" => {
            if let Some(Value::Cell(c)) = o.fields.get("Data") {
                if let Some(first) = c.data.first() {
                    if let Ok(s) = first.as_str() {
                        return ok1(Value::String(s.to_string()));
                    }
                }
            }
            ok1(Value::String(String::new()))
        }
        Value::Matrix(m) => {
            let s: String = m
                .data()
                .iter()
                .filter_map(|&x| char::from_u32(x.max(0.0).round() as u32))
                .collect();
            ok1(Value::String(s))
        }
        other => ok1(Value::String(format!("{other}"))),
    }
}

fn fn_cast_named(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("cast", "2 (A, newclass)", args.len()));
    }
    let cname = args[1].as_str()?;
    if cname == "char" {
        return fn_char(&args[0..1]);
    }
    let target = NumericClass::from_name(cname)
        .ok_or_else(|| MatroError::runtime(format!("cast: unknown class '{cname}'")))?;
    fn_cast_class("cast", &args[0..1], target)
}

fn fn_int_bound(name: &str, args: &[Value], is_max: bool) -> Result<Vec<Value>> {
    let cname = if args.is_empty() {
        "int32"
    } else if args.len() == 1 {
        args[0].as_str()?
    } else {
        return Err(MatroError::arity(name, "0 or 1", args.len()));
    };
    let cls = NumericClass::from_name(cname)
        .ok_or_else(|| MatroError::runtime(format!("{name}: unknown integer class '{cname}'")))?;
    let Some((lo, hi)) = cls.bounds() else {
        return Err(MatroError::runtime(format!(
            "{name}: '{cname}' is not an integer class"
        )));
    };
    let val = if is_max { hi } else { lo };
    ok1(Value::matrix(Matrix::scalar(val).cast_to(cls)))
}

fn fn_val_pred(name: &str, args: &[Value], pred: impl Fn(&Value) -> bool) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity(name, "1", args.len()));
    }
    ok1(Value::bool(pred(&args[0])))
}

fn fn_bit_bin(name: &str, args: &[Value], op: impl Fn(u64, u64) -> u64) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity(name, "2", args.len()));
    }
    let a = args[0].as_matrix()?;
    let b = args[1].as_matrix()?;
    let out_dtype = NumericClass::combine(a.dtype(), b.dtype())?;
    let res = a.elementwise(b, |x, y| op(x.max(0.0).round() as u64, y.max(0.0).round() as u64) as f64)?;
    ok1(Value::matrix(res.cast_to(out_dtype)))
}

fn fn_bitshift(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity("bitshift", "2 or 3", args.len()));
    }
    let a = args[0].as_matrix()?;
    let k = args[1].as_matrix()?;
    let dtype = a.dtype();
    let res = a.elementwise(k, |x, sh| {
        let v = x.max(0.0).round() as u64;
        let s = sh.round() as i32;
        let shifted = if s >= 0 {
            v.checked_shl(s as u32).unwrap_or(0)
        } else {
            v.checked_shr((-s) as u32).unwrap_or(0)
        };
        shifted as f64
    })?;
    ok1(Value::matrix(res.cast_to(dtype)))
}

fn fn_bitget(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("bitget", "2 (A, bit)", args.len()));
    }
    let a = args[0].as_matrix()?;
    let bit = args[1].as_matrix()?;
    let res = a.elementwise(bit, |x, b| {
        let v = x.max(0.0).round() as u64;
        let pos = (b.round() as isize - 1).max(0) as u32;
        if pos < 64 { ((v >> pos) & 1) as f64 } else { 0.0 }
    })?;
    ok1(Value::matrix(res))
}

fn fn_bitset(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity("bitset", "2 or 3 (A, bit [, val])", args.len()));
    }
    let a = args[0].as_matrix()?;
    let bit = args[1].as_matrix()?;
    let set_one = args
        .get(2)
        .map(|v| v.as_scalar().map(|s| s != 0.0))
        .transpose()?
        .unwrap_or(true);
    let dtype = a.dtype();
    let res = a.elementwise(bit, |x, b| {
        let v = x.max(0.0).round() as u64;
        let pos = (b.round() as isize - 1).max(0) as u32;
        let out = if pos < 64 {
            if set_one { v | (1u64 << pos) } else { v & !(1u64 << pos) }
        } else {
            v
        };
        out as f64
    })?;
    ok1(Value::matrix(res.cast_to(dtype)))
}

fn fn_isequal(args: &[Value], nan_eq: bool) -> Result<Vec<Value>> {
    if args.len() < 2 {
        return Err(MatroError::arity(
            if nan_eq { "isequaln" } else { "isequal" },
            "at least 2",
            args.len(),
        ));
    }
    let first = &args[0];
    let eq = args[1..].iter().all(|other| values_equal(first, other, nan_eq));
    ok1(Value::bool(eq))
}

fn values_equal(a: &Value, b: &Value, nan_eq: bool) -> bool {
    let f_eq = |x: f64, y: f64| -> bool {
        if nan_eq && x.is_nan() && y.is_nan() {
            true
        } else {
            x == y
        }
    };
    match (a, b) {
        (Value::Matrix(ma), Value::Matrix(mb)) => {
            if ma.dims() != mb.dims() {
                return false;
            }
            if !ma.is_complex() && !mb.is_complex() {
                ma.data()
                    .iter()
                    .zip(mb.data().iter())
                    .all(|(&x, &y)| f_eq(x, y))
            } else {
                let re_ok = ma
                    .data()
                    .iter()
                    .zip(mb.data().iter())
                    .all(|(&x, &y)| f_eq(x, y));
                if !re_ok {
                    return false;
                }
                let im_a = ma.imag_data();
                let im_b = mb.imag_data();
                (0..ma.len()).all(|i| {
                    let xa = im_a.map_or(0.0, |v| v[i]);
                    let xb = im_b.map_or(0.0, |v| v[i]);
                    f_eq(xa, xb)
                })
            }
        }
        (Value::Sparse(sa), Value::Sparse(sb)) => {
            values_equal(&Value::matrix(sa.to_dense()), &Value::matrix(sb.to_dense()), nan_eq)
        }
        (Value::Matrix(_), Value::Sparse(sb)) => {
            values_equal(a, &Value::matrix(sb.to_dense()), nan_eq)
        }
        (Value::Sparse(sa), Value::Matrix(_)) => {
            values_equal(&Value::matrix(sa.to_dense()), b, nan_eq)
        }
        (Value::String(sa), Value::String(sb)) => sa == sb,
        (Value::Cell(ca), Value::Cell(cb)) => {
            ca.rows == cb.rows
                && ca.cols == cb.cols
                && ca
                    .data
                    .iter()
                    .zip(cb.data.iter())
                    .all(|(x, y)| values_equal(x, y, nan_eq))
        }
        (Value::Struct(sa), Value::Struct(sb)) => {
            sa.len() == sb.len()
                && sa
                    .iter()
                    .all(|(k, v)| sb.get(k).is_some_and(|bv| values_equal(v, bv, nan_eq)))
        }
        (Value::StructArray(sa), Value::StructArray(sb)) => {
            sa.rows == sb.rows
                && sa.cols == sb.cols
                && sa.items.iter().zip(sb.items.iter()).all(|(ia, ib)| {
                    ia.len() == ib.len()
                        && ia
                            .iter()
                            .all(|(k, v)| ib.get(k).is_some_and(|bv| values_equal(v, bv, nan_eq)))
                })
        }
        (Value::Object(oa), Value::Object(ob)) => {
            oa.class_name == ob.class_name
                && oa.fields.len() == ob.fields.len()
                && oa.fields.iter().all(|(k, v)| {
                    ob.fields
                        .get(k)
                        .is_some_and(|bv| values_equal(v, bv, nan_eq))
                })
        }
        (Value::Table(ta), Value::Table(tb)) => {
            ta.nrows == tb.nrows
                && ta.col_names == tb.col_names
                && ta
                    .columns
                    .iter()
                    .zip(tb.columns.iter())
                    .all(|(ca, cb)| values_equal(ca, cb, nan_eq))
        }
        _ => false,
    }
}
