use std::collections::BTreeMap;

use matro_loader::ModuleLoader;
use matro_types::{
    CellArray, Complex, MatroError, Matrix, NumericClass, PlotFigure, PlotKind, Result, Rng,
    SparseMatrix, Value,
};

mod control_ext;
mod extras;
pub mod fs_media;
mod image_proc;
pub mod interp;
pub mod modern_types;
mod opt_ctrl;
mod opt_ext;
mod regex;
mod signal;
mod signal_ext;
mod sparse_eig;
mod stats;
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
        "inf" | "Inf" => ok1(Value::scalar(f64::INFINITY)),
        "nan" | "NaN" => ok1(Value::scalar(f64::NAN)),
        "eps" => ok1(Value::scalar(f64::EPSILON)),

        // Errors & assertions
        "error" => fn_error(args),
        "assert" => fn_assert(args),

        // Complex utilities
        "complex" => fn_complex(args),
        "real" => fn_unary_mat(name, args, |m| Ok(m.real_part())),
        "imag" => fn_unary_mat(name, args, |m| Ok(m.imag_part())),
        "conj" => fn_unary_mat(name, args, |m| Ok(m.conj())),
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
        "isstruct" => fn_val_pred("isstruct", args, |v| {
            matches!(v, Value::Struct(_) | Value::StructArray(_))
        }),
        "bitand" => fn_bit_bin("bitand", args, |a, b| a & b),
        "bitor" => fn_bit_bin("bitor", args, |a, b| a | b),
        "bitxor" => fn_bit_bin("bitxor", args, |a, b| a ^ b),
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

        // Constructors & RNG
        "zeros" => fn_fill(name, args, Matrix::zeros),
        "ones" => fn_fill(name, args, Matrix::ones),
        "eye" => fn_fill(name, args, Matrix::eye),
        "rand" => fn_rand(args, ctx.rng, false),
        "randn" => fn_rand(args, ctx.rng, true),
        "rng" => fn_rng(args, ctx.rng),
        "linspace" => fn_linspace(args),
        "logspace" => fn_logspace(args),
        "meshgrid" => fn_meshgrid(args),
        "ndgrid" => fn_ndgrid(args),
        "diag" => fn_diag(args),
        "magic" => fn_magic(args),
        "cell" => fn_cell(args),
        "struct" => fn_struct(args),
        "sparse" => fn_sparse(args),
        "speye" => fn_speye(args),
        "full" => fn_full(args),
        "issparse" => fn_issparse(args),
        "nnz" => fn_nnz(args),
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
        "regexp" => regex::fn_regexp(args, false),
        "regexpi" => regex::fn_regexp(args, true),
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
        "transpose" => fn_unary_mat(name, args, |m| Ok(m.transpose())),
        "ctranspose" => fn_unary_mat(name, args, |m| Ok(m.ctranspose())),

        // Linear algebra, signal processing & numerics
        "det" => fn_unary_mat(name, args, |m| Ok(Matrix::complex_scalar(m.det_c()?).normalize())),
        "inv" => fn_unary_mat(name, args, |m| m.inv()),
        "expm" => fn_unary_mat(name, args, |m| m.expm()),
        "lu" => fn_lu(args, nargout),
        "qr" => fn_qr(args, nargout),
        "svd" => fn_svd(args, nargout),
        "eig" => fn_eig(args, nargout),
        "eigs" => sparse_eig::fn_eigs(args, nargout, ctx.rng),
        "svds" => sparse_eig::fn_svds(args, nargout, ctx.rng),
        "chol" => fn_unary_mat(name, args, |m| m.chol()),
        "rank" => fn_rank(args),
        "cond" => fn_cond(args),
        "trace" => fn_trace(args),
        "dot" => fn_dot(args),
        "norm" => fn_norm(args),
        "fft" => fn_unary_mat(name, args, |m| Ok(m.fft())),
        "ifft" => fn_unary_mat(name, args, |m| Ok(m.ifft())),
        "fft2" => fn_unary_mat(name, args, |m| Ok(m.fft2())),
        "ifft2" => fn_unary_mat(name, args, |m| Ok(m.ifft2())),
        "roots" => extras::fn_roots(args),
        "poly" => extras::fn_poly(args),
        "polyfit" => fn_polyfit(args),
        "polyval" => fn_polyval(args),
        "conv" => fn_conv(args),
        "conv2" => extras::fn_conv2(args),
        "filter" => extras::fn_filter(args),
        "interp1" => interp::fn_interp1(args),
        "interp2" => extras::fn_interp2(args),
        "pchip" => interp::fn_pchip(args),
        "spline" => interp::fn_spline(args),
        "mkpp" => interp::fn_mkpp(args),
        "ppval" => interp::fn_ppval(args),
        "griddedInterpolant" => interp::fn_gridded_interpolant(args),
        "sort" => fn_sort(args, nargout),
        "unique" => extras::fn_unique(args),
        "ismember" => extras::fn_ismember(args),
        "setdiff" | "intersect" | "union" => extras::fn_set_binary(name, args),
        "cumsum" => fn_cumsum(args),
        "cumprod" => extras::fn_cumprod(args),
        "diff" => fn_diff(args),

        // Reductions
        "sum" => fn_reduce_c(name, args, |slice| {
            slice.iter().copied().fold(Complex::ZERO, |a, b| a + b)
        }),
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
        "asin" => fn_elem1(name, args, f64::asin),
        "acos" => fn_elem1(name, args, f64::acos),
        "atan" => fn_elem1(name, args, f64::atan),
        "atan2" => fn_elem2(name, args, f64::atan2),
        "log10" => fn_elem1(name, args, f64::log10),
        "log2" => fn_elem1(name, args, f64::log2),
        "floor" => fn_elem1(name, args, f64::floor),
        "ceil" => fn_elem1(name, args, f64::ceil),
        "round" => fn_elem1(name, args, f64::round),
        "sign" => fn_elem1(name, args, |x| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }),
        "mod" => fn_elem2(name, args, |a, b| ((a % b) + b) % b),
        "rem" => fn_elem2(name, args, |a, b| a % b),

        // Output
        "disp" => fn_disp(args, ctx.stdout),
        "fprintf" | "printf" => fn_fprintf(args, ctx.stdout),

        // Statistics toolbox
        "std" => stats::fn_std_var("std", args, true),
        "var" => stats::fn_std_var("var", args, false),
        "median" => stats::fn_median(args),
        "prctile" => stats::fn_quantile("prctile", args, true),
        "quantile" => stats::fn_quantile("quantile", args, false),
        "cov" => stats::fn_cov_or_corr("cov", args, false),
        "corrcoef" => stats::fn_cov_or_corr("corrcoef", args, true),
        "erf" => stats::fn_erf(args),
        "erfc" => stats::fn_erfc(args),
        "normpdf" => stats::fn_normpdf(args),
        "normcdf" => stats::fn_normcdf(args),
        "norminv" => stats::fn_norminv(args),
        "ttest" => stats::fn_ttest(args, nargout),
        "ttest2" => stats_ml::fn_ttest2(args, nargout),
        "kmeans" => stats::fn_kmeans(args, nargout, ctx.rng),
        "pca" => stats::fn_pca(args, nargout),
        "gamma" => stats_ml::fn_gamma_fn(args, false),
        "gammaln" => stats_ml::fn_gamma_fn(args, true),
        "gammainc" => stats_ml::fn_gammainc(args),
        "betainc" => stats_ml::fn_betainc(args),
        "tpdf" | "tcdf" | "tinv" => stats_ml::fn_t_dist(name, args),
        "chi2pdf" | "chi2cdf" | "chi2inv" => stats_ml::fn_chi2_dist(name, args),
        "fpdf" | "fcdf" => stats_ml::fn_f_dist(name, args),
        "exppdf" | "expcdf" | "expinv" => stats_ml::fn_exp_dist(name, args),
        "binopdf" | "binocdf" => stats_ml::fn_bino_dist(name, args),
        "poisspdf" | "poisscdf" => stats_ml::fn_poiss_dist(name, args),
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
        "squareform" => stats_ml::fn_squareform(args),
        "linkage" => stats_ml::fn_linkage(args),
        "cluster" => stats_ml::fn_cluster(args),

        // Signal processing toolbox
        "fftshift" => signal::fn_fftshift(args, false),
        "ifftshift" => signal::fn_fftshift(args, true),
        "hann" | "hamming" | "blackman" => signal::fn_window(name, args),
        "xcorr" => signal::fn_xcorr(args, nargout),
        "findpeaks" => signal::fn_findpeaks(args, nargout),
        "butter" => signal::fn_butter(args),
        "cheby1" => signal_ext::fn_cheby1(args),
        "cheby2" => signal_ext::fn_cheby2(args),
        "fir1" => signal_ext::fn_fir1(args),
        "freqz" => signal_ext::fn_freqz(args, nargout, ctx.figure),
        "grpdelay" => signal_ext::fn_grpdelay(args, nargout),
        "filtfilt" => signal::fn_filtfilt(args),
        "pwelch" => signal::fn_pwelch(args, nargout),
        "spectrogram" | "stft" => signal_ext::fn_spectrogram(args, nargout, ctx.figure),
        "hilbert" => signal_ext::fn_hilbert(args),
        "envelope" => signal_ext::fn_envelope(args, nargout),
        "medfilt1" => signal_ext::fn_medfilt1(args),
        "sgolayfilt" => signal_ext::fn_sgolayfilt(args),
        "detrend" => signal_ext::fn_detrend(args),
        "downsample" => signal_ext::fn_downsample(args),
        "upsample" => signal_ext::fn_upsample(args),
        "resample" => signal_ext::fn_resample(args),
        "bartlett" | "kaiser" | "flattopwin" => signal_ext::fn_extra_window(name, args),

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
        "care" => control_ext::fn_care(args),
        "lqr" => control_ext::fn_lqr(args, nargout),
        "lqe" => control_ext::fn_lqe(args, nargout),

        // VFS & module path helpers
        "addpath" => fn_addpath(args, ctx.loader),
        "csvread" => fn_csvread(args, ctx.loader),
        "csvwrite" => fn_csvwrite(args, ctx.loader),
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
        "renamevars" => tables::fn_renamevars(args),
        "removevars" => tables::fn_removevars(args),
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
        "startsWith" => modern_types::fn_starts_ends_with("startsWith", args, false),
        "endsWith" => modern_types::fn_starts_ends_with("endsWith", args, true),
        "erase" => modern_types::fn_erase(args),
        "strip" => modern_types::fn_strip(args),
        "replace" => extras::fn_strrep(args),

        // Functional operator builtins (useful with bsxfun / @plus / etc.)
        "plus" => tables::fn_binop_c("plus", args, |a, b| a + b),
        "minus" => tables::fn_binop_c("minus", args, |a, b| a - b),
        "times" => tables::fn_binop_c("times", args, |a, b| a * b),
        "rdivide" => tables::fn_binop_c("rdivide", args, |a, b| a / b),
        "ldivide" => tables::fn_binop_c("ldivide", args, |a, b| b / a),
        "power" => tables::fn_binop_c("power", args, Complex::powc),
        "eq" => tables::fn_binop_cmp("eq", args, |a, b| a == b),
        "ne" => tables::fn_binop_cmp("ne", args, |a, b| a != b),
        "lt" => tables::fn_binop_cmp("lt", args, |a, b| a < b),
        "le" => tables::fn_binop_cmp("le", args, |a, b| a <= b),
        "gt" => tables::fn_binop_cmp("gt", args, |a, b| a > b),
        "ge" => tables::fn_binop_cmp("ge", args, |a, b| a >= b),

        // Plotting & Handle Graphics
        "plot" => fn_plot(PlotKind::Line, args, nargout, ctx.figure, false, false),
        "semilogx" => fn_plot(PlotKind::Line, args, nargout, ctx.figure, true, false),
        "semilogy" => fn_plot(PlotKind::Line, args, nargout, ctx.figure, false, true),
        "loglog" => fn_plot(PlotKind::Line, args, nargout, ctx.figure, true, true),
        "scatter" => fn_plot(PlotKind::Scatter, args, nargout, ctx.figure, false, false),
        "bar" => fn_plot(PlotKind::Bar, args, nargout, ctx.figure, false, false),
        "stem" => fn_plot(PlotKind::Stem, args, nargout, ctx.figure, false, false),
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
        "clf" | "figure" => {
            ctx.figure.clear();
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
        "delete" => fs_media::fn_delete(args, ctx.loader),
        "copyfile" => fs_media::fn_copyfile(args, ctx.loader),
        "movefile" => fs_media::fn_movefile(args, ctx.loader),
        "imread" => fs_media::fn_imread(args, ctx.loader),
        "imwrite" => fs_media::fn_imwrite(args, ctx.loader),
        "audioread" => fs_media::fn_audioread(args, nargout, ctx.loader),
        "audiowrite" => fs_media::fn_audiowrite(args, ctx.loader),

        // Image Processing Toolbox basics
        "rgb2gray" => image_proc::fn_rgb2gray(args),
        "im2double" => image_proc::fn_im2double(args),
        "im2uint8" => image_proc::fn_im2uint8(args),
        "imresize" => image_proc::fn_imresize(args),
        "imfilter" => image_proc::fn_imfilter(args),
        "imgaussfilt" => image_proc::fn_imgaussfilt(args),
        "edge" => image_proc::fn_edge(args),
        "graythresh" => image_proc::fn_graythresh(args),
        "imbinarize" => image_proc::fn_imbinarize(args),
        "imhist" => image_proc::fn_imhist(args, nargout),
        "imadjust" => image_proc::fn_imadjust(args),

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

fn fn_error(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::runtime("Error raised"));
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

fn fn_isreal(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("isreal", "1", args.len()));
    }
    ok1(Value::bool(!args[0].as_matrix()?.is_complex()))
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
        Value::Struct(m) => m.is_empty(),
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
    let m = args[0].as_matrix()?;
    let out = m
        .map_c_to_real(|c| if pred(c) { 1.0 } else { 0.0 })
        .with_logical(true);
    ok1(Value::matrix(out))
}

fn fn_find(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("find", "1", args.len()));
    }
    let m = args[0].as_matrix()?;
    if nargout >= 2 {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        for c in 0..m.cols() {
            for r in 0..m.rows() {
                if !m.get_c(r, c).is_zero() {
                    rows.push((r + 1) as f64);
                    cols.push((c + 1) as f64);
                }
            }
        }
        return Ok(vec![
            Value::matrix(Matrix::col_vec(rows)),
            Value::matrix(Matrix::col_vec(cols)),
        ]);
    }
    let mut idxs = Vec::new();
    for i in 0..m.len() {
        if !m.get_linear_c(i).is_zero() {
            idxs.push((i + 1) as f64);
        }
    }
    let out = if m.rows() == 1 {
        Matrix::row_vec(idxs)
    } else {
        Matrix::col_vec(idxs)
    };
    ok1(Value::matrix(out))
}

fn fn_any_all(name: &str, args: &[Value], is_all: bool) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity(name, "1", args.len()));
    }
    let m = args[0].as_matrix()?;
    let pred = |c: Complex| !c.is_zero() && !c.re.is_nan() && !c.im.is_nan();
    if m.rows() <= 1 || m.cols() <= 1 {
        let b = if is_all {
            (0..m.len()).all(|i| pred(m.get_linear_c(i)))
        } else {
            (0..m.len()).any(|i| pred(m.get_linear_c(i)))
        };
        return ok1(Value::bool(b));
    }
    let mut out = Vec::with_capacity(m.cols());
    for c in 0..m.cols() {
        let b = if is_all {
            (0..m.rows()).all(|r| pred(m.get_c(r, c)))
        } else {
            (0..m.rows()).any(|r| pred(m.get_c(r, c)))
        };
        out.push(if b { 1.0 } else { 0.0 });
    }
    ok1(Value::matrix(Matrix::row_vec(out).with_logical(true)))
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

fn fn_fill(name: &str, args: &[Value], make: fn(usize, usize) -> Matrix) -> Result<Vec<Value>> {
    if args.len() == 1 {
        if let Ok(m) = args[0].to_dense_matrix() {
            if m.len() >= 2 {
                let dims: Vec<usize> = m
                    .data()
                    .iter()
                    .map(|&x| x.max(0.0).round() as usize)
                    .collect();
                if dims.len() == 2 {
                    return ok1(Value::matrix(make(dims[0], dims[1])));
                }
                if name != "eye" {
                    let mat = if name == "ones" {
                        Matrix::ones_nd(dims)
                    } else {
                        Matrix::zeros_nd(dims)
                    };
                    return ok1(Value::matrix(mat));
                }
            }
        }
    }
    if args.len() > 2 && name != "eye" {
        let dims: Vec<usize> = args
            .iter()
            .map(|a| as_usize(a, name))
            .collect::<Result<_>>()?;
        let m = if name == "ones" {
            Matrix::ones_nd(dims)
        } else {
            Matrix::zeros_nd(dims)
        };
        return ok1(Value::matrix(m));
    }
    let (r, c) = match args.len() {
        0 => (1, 1),
        1 => {
            let n = as_usize(&args[0], name)?;
            (n, n)
        }
        2 => (as_usize(&args[0], name)?, as_usize(&args[1], name)?),
        n => return Err(MatroError::arity(name, "0..=2", n)),
    };
    ok1(Value::matrix(make(r, c)))
}

fn fn_rand(args: &[Value], rng: &mut Rng, normal: bool) -> Result<Vec<Value>> {
    let name = if normal { "randn" } else { "rand" };
    let dims: Vec<usize> = match args.len() {
        0 => vec![1, 1],
        1 => {
            let m = args[0].to_dense_matrix()?;
            if m.len() >= 2 {
                m.data().iter().map(|&x| x.max(0.0).round() as usize).collect()
            } else {
                let n = as_usize(&args[0], name)?;
                vec![n, n]
            }
        }
        _ => args
            .iter()
            .map(|a| as_usize(a, name))
            .collect::<Result<_>>()?,
    };
    let total: usize = dims.iter().product();
    let mut data = Vec::with_capacity(total);
    for _ in 0..total {
        data.push(if normal {
            rng.next_normal()
        } else {
            rng.next_f64()
        });
    }
    ok1(Value::matrix(Matrix::from_nd_col_major(dims, data)?))
}

fn fn_rng(args: &[Value], rng: &mut Rng) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("rng", "1", args.len()));
    }
    let seed = match &args[0] {
        Value::String(s) if s == "default" => 0,
        other => other.as_scalar()?.max(0.0).round() as u64,
    };
    rng.seed(seed);
    Ok(vec![])
}

fn fn_meshgrid(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("meshgrid", "1 or 2", args.len()));
    }
    let xv = args[0].as_matrix()?.data();
    let yv = if args.len() == 2 {
        args[1].as_matrix()?.data()
    } else {
        xv
    };
    let rows = yv.len();
    let cols = xv.len();
    let mut x_mat = Matrix::zeros(rows, cols);
    let mut y_mat = Matrix::zeros(rows, cols);
    for r in 0..rows {
        for c in 0..cols {
            x_mat.set(r, c, xv[c]);
            y_mat.set(r, c, yv[r]);
        }
    }
    Ok(vec![Value::matrix(x_mat), Value::matrix(y_mat)])
}

fn fn_cell(args: &[Value]) -> Result<Vec<Value>> {
    let (r, c) = match args.len() {
        0 => (0, 0),
        1 => {
            let n = as_usize(&args[0], "cell")?;
            (n, n)
        }
        2 => (as_usize(&args[0], "cell")?, as_usize(&args[1], "cell")?),
        n => return Err(MatroError::arity("cell", "0..=2", n)),
    };
    ok1(Value::Cell(CellArray::new_empty_cells(r, c)))
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
    let field = args[1].as_str()?;
    let present = match &args[0] {
        Value::Struct(map) => map.contains_key(field),
        Value::Object(o) => o.fields.contains_key(field),
        _ => false,
    };
    ok1(Value::bool(present))
}

fn fn_isa(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("isa", "2", args.len()));
    }
    let expected = args[1].as_str()?;
    let matched = match expected {
        "numeric" => matches!(&args[0], Value::Matrix(m) if !m.is_logical())
            || matches!(&args[0], Value::Sparse(_)),
        "double" => matches!(&args[0], Value::Matrix(m) if !m.is_logical())
            || matches!(&args[0], Value::Sparse(_)),
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
        3 | 5 => {
            let iv = args[0].as_matrix()?.data();
            let jv = args[1].as_matrix()?.data();
            let vv = args[2].as_matrix()?;
            if iv.len() != jv.len() {
                return Err(MatroError::dimension("sparse(i, j, v): i and j must have equal length"));
            }
            let n_entries = iv.len();
            let scalar_v = vv.as_scalar();
            if scalar_v.is_none() && vv.len() != n_entries {
                return Err(MatroError::dimension("sparse(i, j, v): v length must match i and j"));
            }
            let mut max_r = 0usize;
            let mut max_c = 0usize;
            let mut triplets = Vec::with_capacity(n_entries);
            for k in 0..n_entries {
                let r = iv[k].round() as isize;
                let c = jv[k].round() as isize;
                if r < 1 || c < 1 {
                    return Err(MatroError::index("sparse: indices must be >= 1"));
                }
                let r0 = (r - 1) as usize;
                let c0 = (c - 1) as usize;
                max_r = max_r.max(r0 + 1);
                max_c = max_c.max(c0 + 1);
                let val = scalar_v.unwrap_or_else(|| vv.data()[k]);
                triplets.push((r0, c0, val));
            }
            let (rows, cols) = if args.len() == 5 {
                (as_usize(&args[3], "sparse")?, as_usize(&args[4], "sparse")?)
            } else {
                (max_r, max_c)
            };
            ok1(Value::sparse(SparseMatrix::from_triplets(rows, cols, &triplets)?))
        }
        n => Err(MatroError::arity("sparse", "1, 2, 3, or 5", n)),
    }
}

fn fn_speye(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("speye", "1", args.len()));
    }
    let n = as_usize(&args[0], "speye")?;
    ok1(Value::sparse(SparseMatrix::eye(n)))
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
    let eq = match (&args[0], &args[1]) {
        (Value::String(a), Value::String(b)) => a == b,
        _ => false,
    };
    ok1(Value::bool(eq))
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
    if args.len() != 1 {
        return Err(MatroError::arity("num2str", "1", args.len()));
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
    if args.len() != 1 {
        return Err(MatroError::arity("diag", "1", args.len()));
    }
    let m = args[0].as_matrix()?;
    if m.rows() == 1 || m.cols() == 1 {
        let n = m.len();
        let mut out = Matrix::zeros(n, n);
        for i in 0..n {
            out.set_c(i, i, m.get_linear_c(i));
        }
        ok1(Value::matrix(out.normalize()))
    } else {
        let n = m.rows().min(m.cols());
        let vals: Vec<Complex> = (0..n).map(|i| m.get_c(i, i)).collect();
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

fn fn_size(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("size", "1 or 2", args.len()));
    }
    let dims: Vec<usize> = match &args[0] {
        Value::Matrix(m) => m.dims(),
        Value::Sparse(s) => vec![s.rows(), s.cols()],
        Value::Cell(c) => vec![c.rows, c.cols],
        Value::StructArray(sa) => vec![sa.rows, sa.cols],
        Value::String(s) => vec![1, s.chars().count()],
        Value::Table(t) => vec![t.nrows, t.col_names.len()],
        Value::Map(m) => vec![m.entries.len(), 1],
        Value::Struct(_) | Value::Object(_) | Value::Handle(_) | Value::Closure(_) => vec![1, 1],
    };
    if args.len() == 2 {
        let dim = as_usize(&args[1], "size")?;
        let val = if dim >= 1 {
            dims.get(dim - 1).copied().unwrap_or(1)
        } else {
            1
        };
        return ok1(Value::scalar(val as f64));
    }
    if nargout >= 2 {
        let mut out = Vec::with_capacity(nargout);
        for k in 0..nargout {
            out.push(Value::scalar(dims.get(k).copied().unwrap_or(1) as f64));
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
    match &args[0] {
        Value::Matrix(m) => {
            let l = if m.is_empty() {
                0
            } else {
                m.dims().into_iter().max().unwrap_or(0)
            };
            ok1(Value::scalar(l as f64))
        }
        Value::Sparse(s) => {
            let l = if s.rows() == 0 || s.cols() == 0 {
                0
            } else {
                s.rows().max(s.cols())
            };
            ok1(Value::scalar(l as f64))
        }
        Value::Cell(c) => {
            let l = if c.data.is_empty() { 0 } else { c.rows.max(c.cols) };
            ok1(Value::scalar(l as f64))
        }
        Value::StructArray(sa) => {
            let l = if sa.items.is_empty() { 0 } else { sa.rows.max(sa.cols) };
            ok1(Value::scalar(l as f64))
        }
        Value::String(s) => ok1(Value::scalar(s.chars().count() as f64)),
        Value::Table(t) => ok1(Value::scalar(t.nrows.max(t.col_names.len()) as f64)),
        Value::Map(m) => ok1(Value::scalar(m.entries.len() as f64)),
        Value::Struct(_) | Value::Object(_) | Value::Handle(_) | Value::Closure(_) => {
            ok1(Value::scalar(1.0))
        }
    }
}

fn fn_numel(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("numel", "1", args.len()));
    }
    let n = match &args[0] {
        Value::Matrix(m) => m.len(),
        Value::Sparse(s) => s.rows().saturating_mul(s.cols()),
        Value::Cell(c) => c.data.len(),
        Value::StructArray(sa) => sa.items.len(),
        Value::String(s) => s.chars().count(),
        Value::Table(t) => t.nrows * t.col_names.len(),
        Value::Map(m) => m.entries.len(),
        Value::Struct(_) | Value::Object(_) | Value::Handle(_) | Value::Closure(_) => 1,
    };
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

fn fn_reshape(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 3 {
        return Err(MatroError::arity("reshape", ">= 3 (A, d1, d2, ...)", args.len()));
    }
    let m = args[0].as_matrix()?;
    let dims: Vec<usize> = args[1..]
        .iter()
        .map(|a| as_usize(a, "reshape"))
        .collect::<Result<_>>()?;
    let out = if let Some(im) = m.imag_data() {
        Matrix::from_nd_complex_col_major(dims, m.data().to_vec(), im.to_vec())?
    } else {
        Matrix::from_nd_col_major(dims, m.data().to_vec())?
    };
    ok1(Value::matrix(out))
}

fn fn_unary_mat(
    name: &str,
    args: &[Value],
    f: impl FnOnce(&Matrix) -> Result<Matrix>,
) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity(name, "1", args.len()));
    }
    let m = args[0].as_matrix()?;
    ok1(Value::matrix(f(m)?))
}

fn fn_lu(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("lu", "1", args.len()));
    }
    let (l, u, p) = args[0].as_matrix()?.lu()?;
    if nargout >= 3 {
        Ok(vec![Value::matrix(l), Value::matrix(u), Value::matrix(p)])
    } else if nargout == 2 {
        let pl = p.transpose().matmul(&l)?;
        Ok(vec![Value::matrix(pl), Value::matrix(u)])
    } else {
        ok1(Value::matrix(u))
    }
}

fn fn_qr(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("qr", "1", args.len()));
    }
    let (q, r) = args[0].as_matrix()?.qr()?;
    if nargout >= 2 {
        Ok(vec![Value::matrix(q), Value::matrix(r)])
    } else {
        ok1(Value::matrix(r))
    }
}

fn fn_svd(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("svd", "1", args.len()));
    }
    let (u, s, v) = args[0].as_matrix()?.svd()?;
    if nargout >= 3 {
        Ok(vec![Value::matrix(u), Value::matrix(s), Value::matrix(v)])
    } else {
        let k = s.rows().min(s.cols());
        let diag: Vec<f64> = (0..k).map(|i| s.get(i, i)).collect();
        ok1(Value::matrix(Matrix::col_vec(diag)))
    }
}

fn fn_eig(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("eig", "1", args.len()));
    }
    let (v, d) = args[0].as_matrix()?.eig()?;
    if nargout >= 2 {
        Ok(vec![Value::matrix(v), Value::matrix(d)])
    } else {
        let n = d.rows().min(d.cols());
        let diag: Vec<Complex> = (0..n).map(|i| d.get_c(i, i)).collect();
        ok1(Value::matrix(Matrix::complex_col_vec(diag).normalize()))
    }
}

fn fn_rank(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("rank", "1", args.len()));
    }
    let m = args[0].as_matrix()?;
    let (_, s, _) = m.svd()?;
    let k = s.rows().min(s.cols());
    if k == 0 {
        return ok1(Value::scalar(0.0));
    }
    let max_s = s.get(0, 0);
    let tol = (m.rows().max(m.cols()) as f64) * f64::EPSILON * max_s.max(1.0);
    let r = (0..k).filter(|&i| s.get(i, i) > tol).count();
    ok1(Value::scalar(r as f64))
}

fn fn_cond(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("cond", "1", args.len()));
    }
    let (_, s, _) = args[0].as_matrix()?.svd()?;
    let k = s.rows().min(s.cols());
    if k == 0 {
        return ok1(Value::scalar(0.0));
    }
    let smax = s.get(0, 0);
    let smin = s.get(k - 1, k - 1);
    ok1(Value::scalar(if smin < 1e-15 {
        f64::INFINITY
    } else {
        smax / smin
    }))
}

fn fn_polyfit(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 3 {
        return Err(MatroError::arity("polyfit", "3", args.len()));
    }
    let x = args[0].as_matrix()?;
    let y = args[1].as_matrix()?;
    let deg = as_usize(&args[2], "polyfit")?;
    if x.len() != y.len() || x.len() <= deg {
        return Err(MatroError::dimension(
            "polyfit: x and y must have equal length > degree",
        ));
    }
    let n = x.len();
    let cols = deg + 1;
    let mut vander = Matrix::zeros(n, cols);
    for r in 0..n {
        let xv = x.data()[r];
        for c in 0..cols {
            let p = (deg - c) as i32;
            vander.set(r, c, xv.powi(p));
        }
    }
    let y_col = Matrix::col_vec(y.data().to_vec());
    let coeff = vander.solve(&y_col)?;
    ok1(Value::matrix(coeff.transpose()))
}

fn fn_polyval(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("polyval", "2", args.len()));
    }
    let p = args[0].as_matrix()?;
    let x = args[1].as_matrix()?;
    let coeffs = p.data();
    let out = x.map(|xv| {
        let mut acc = 0.0;
        for &c in coeffs {
            acc = acc * xv + c;
        }
        acc
    });
    ok1(Value::matrix(out))
}

fn fn_conv(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("conv", "2", args.len()));
    }
    let a = args[0].as_matrix()?;
    let b = args[1].as_matrix()?;
    if a.is_empty() || b.is_empty() {
        return ok1(Value::empty());
    }
    let n = a.len() + b.len() - 1;
    let mut out = vec![Complex::ZERO; n];
    for i in 0..a.len() {
        let ai = a.get_linear_c(i);
        for j in 0..b.len() {
            out[i + j] = out[i + j] + ai * b.get_linear_c(j);
        }
    }
    ok1(Value::matrix(Matrix::complex_row_vec(out).normalize()))
}

fn fn_ndgrid(args: &[Value]) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("ndgrid", "1 or 2", args.len()));
    }
    let xv = args[0].to_dense_matrix()?;
    let yv = if args.len() == 2 {
        args[1].to_dense_matrix()?
    } else {
        xv.clone()
    };
    let nx = xv.len();
    let ny = yv.len();
    let mut x_out = Matrix::zeros(nx, ny);
    let mut y_out = Matrix::zeros(nx, ny);
    for c in 0..ny {
        for r in 0..nx {
            x_out.set(r, c, xv.data()[r]);
            y_out.set(r, c, yv.data()[c]);
        }
    }
    Ok(vec![Value::matrix(x_out), Value::matrix(y_out)])
}

fn fn_sort(args: &[Value], nargout: usize) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity("sort", "1 or 2", args.len()));
    }
    let m = args[0].as_matrix()?;
    let desc = args
        .get(1)
        .and_then(|v| v.as_str().ok())
        .is_some_and(|s| s == "descend");
    let mut pairs: Vec<(f64, usize)> = m
        .data()
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i + 1))
        .collect();
    pairs.sort_by(|a, b| {
        let ord = a.0.total_cmp(&b.0);
        if desc { ord.reverse() } else { ord }
    });
    let vals: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    let idxs: Vec<f64> = pairs.iter().map(|p| p.1 as f64).collect();
    let is_row = m.rows() == 1;
    let v_mat = if is_row {
        Matrix::row_vec(vals)
    } else {
        Matrix::col_vec(vals)
    };
    if nargout >= 2 {
        let i_mat = if is_row {
            Matrix::row_vec(idxs)
        } else {
            Matrix::col_vec(idxs)
        };
        Ok(vec![Value::matrix(v_mat), Value::matrix(i_mat)])
    } else {
        ok1(Value::matrix(v_mat))
    }
}

fn fn_cumsum(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("cumsum", "1", args.len()));
    }
    let m = args[0].as_matrix()?;
    let mut acc = 0.0;
    let out: Vec<f64> = m
        .data()
        .iter()
        .map(|&v| {
            acc += v;
            acc
        })
        .collect();
    let res = if m.rows() == 1 {
        Matrix::row_vec(out)
    } else {
        Matrix::col_vec(out)
    };
    ok1(Value::matrix(res))
}

fn fn_diff(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("diff", "1", args.len()));
    }
    let m = args[0].as_matrix()?;
    if m.len() < 2 {
        return ok1(Value::empty());
    }
    let out: Vec<f64> = m.data().windows(2).map(|w| w[1] - w[0]).collect();
    let res = if m.rows() == 1 {
        Matrix::row_vec(out)
    } else {
        Matrix::col_vec(out)
    };
    ok1(Value::matrix(res))
}

fn fn_trace(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("trace", "1", args.len()));
    }
    let m = args[0].as_matrix()?;
    let n = m.rows().min(m.cols());
    let mut s = Complex::ZERO;
    for i in 0..n {
        s = s + m.get_c(i, i);
    }
    ok1(Value::matrix(Matrix::complex_scalar(s).normalize()))
}

fn fn_dot(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("dot", "2", args.len()));
    }
    let a = args[0].as_matrix()?;
    let b = args[1].as_matrix()?;
    if a.len() != b.len() {
        return Err(MatroError::dimension("dot: vectors must have the same length"));
    }
    let mut s = Complex::ZERO;
    for i in 0..a.len() {
        s = s + a.get_linear_c(i).conj() * b.get_linear_c(i);
    }
    ok1(Value::matrix(Matrix::complex_scalar(s).normalize()))
}

fn fn_norm(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("norm", "1", args.len()));
    }
    let m = args[0].as_matrix()?;
    let s: f64 = (0..m.len()).map(|i| m.get_linear_c(i).norm_sq()).sum();
    ok1(Value::scalar(s.sqrt()))
}

fn fn_minmax(name: &str, args: &[Value], nargout: usize, is_max: bool) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 3 {
        return Err(MatroError::arity(name, "1 to 3", args.len()));
    }
    let m = args[0].as_matrix()?;
    // Element-wise min(A, B) / max(A, B) when 2nd arg is non-empty and args.len() == 2
    if args.len() == 2 && !args[1].as_matrix()?.is_empty() {
        let b = args[1].as_matrix()?;
        let out = m.elementwise(b, |x, y| if is_max { x.max(y) } else { x.min(y) })?;
        return ok1(Value::matrix(out));
    }
    if m.is_empty() {
        return if nargout >= 2 {
            Ok(vec![Value::empty(), Value::empty()])
        } else {
            ok1(Value::empty())
        };
    }
    let best_in_slice = |xs: &[f64]| -> (f64, f64) {
        let mut best_v = xs[0];
        let mut best_i = 0usize;
        for (i, &v) in xs.iter().enumerate().skip(1) {
            if (is_max && v > best_v) || (!is_max && v < best_v) {
                best_v = v;
                best_i = i;
            }
        }
        (best_v, (best_i + 1) as f64)
    };
    if args.len() == 1 && (m.rows() == 1 || m.cols() == 1) {
        let (v, idx) = best_in_slice(m.data());
        return if nargout >= 2 {
            Ok(vec![Value::scalar(v), Value::scalar(idx)])
        } else {
            ok1(Value::scalar(v))
        };
    }
    let dim = if args.len() == 3 {
        as_usize(&args[2], name)?
    } else {
        1
    };
    if dim == 1 {
        let mut vals = Vec::with_capacity(m.cols());
        let mut idxs = Vec::with_capacity(m.cols());
        for c in 0..m.cols() {
            let col: Vec<f64> = (0..m.rows()).map(|r| m.get(r, c)).collect();
            let (v, idx) = best_in_slice(&col);
            vals.push(v);
            idxs.push(idx);
        }
        if nargout >= 2 {
            Ok(vec![
                Value::matrix(Matrix::row_vec(vals)),
                Value::matrix(Matrix::row_vec(idxs)),
            ])
        } else {
            ok1(Value::matrix(Matrix::row_vec(vals)))
        }
    } else if dim == 2 {
        let mut vals = Vec::with_capacity(m.rows());
        let mut idxs = Vec::with_capacity(m.rows());
        for r in 0..m.rows() {
            let row: Vec<f64> = (0..m.cols()).map(|c| m.get(r, c)).collect();
            let (v, idx) = best_in_slice(&row);
            vals.push(v);
            idxs.push(idx);
        }
        if nargout >= 2 {
            Ok(vec![
                Value::matrix(Matrix::col_vec(vals)),
                Value::matrix(Matrix::col_vec(idxs)),
            ])
        } else {
            ok1(Value::matrix(Matrix::col_vec(vals)))
        }
    } else {
        Err(MatroError::runtime(format!("{name}: unsupported dimension {dim}")))
    }
}

fn fn_reduce_c(name: &str, args: &[Value], op: impl Fn(&[Complex]) -> Complex) -> Result<Vec<Value>> {
    if args.is_empty() || args.len() > 2 {
        return Err(MatroError::arity(name, "1 or 2", args.len()));
    }
    let m = args[0].as_matrix()?;
    if m.is_empty() {
        return ok1(Value::scalar(0.0));
    }
    if args.len() == 1 && (m.rows() == 1 || m.cols() == 1) {
        let vals: Vec<Complex> = (0..m.len()).map(|i| m.get_linear_c(i)).collect();
        return ok1(Value::matrix(Matrix::complex_scalar(op(&vals)).normalize()));
    }
    let dim = if args.len() == 2 {
        as_usize(&args[1], name)?
    } else {
        1
    };
    if dim == 1 {
        let mut out = Vec::with_capacity(m.cols());
        for c in 0..m.cols() {
            let col: Vec<Complex> = (0..m.rows()).map(|r| m.get_c(r, c)).collect();
            out.push(op(&col));
        }
        ok1(Value::matrix(Matrix::complex_row_vec(out).normalize()))
    } else if dim == 2 {
        let mut out = Vec::with_capacity(m.rows());
        for r in 0..m.rows() {
            let row: Vec<Complex> = (0..m.cols()).map(|c| m.get_c(r, c)).collect();
            out.push(op(&row));
        }
        ok1(Value::matrix(Matrix::complex_col_vec(out).normalize()))
    } else {
        Err(MatroError::runtime(format!("{name}: unsupported dimension {dim}")))
    }
}

fn fn_elem1(name: &str, args: &[Value], f: fn(f64) -> f64) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity(name, "1", args.len()));
    }
    let m = args[0].as_matrix()?;
    ok1(Value::matrix(m.map(f)))
}

fn fn_elem2(name: &str, args: &[Value], f: fn(f64, f64) -> f64) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity(name, "2", args.len()));
    }
    let a = args[0].as_matrix()?;
    let b = args[1].as_matrix()?;
    ok1(Value::matrix(a.elementwise(b, f)?))
}

fn fn_disp(args: &[Value], stdout: &mut String) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("disp", "1", args.len()));
    }
    stdout.push_str(&format!("{}\n", args[0]));
    Ok(vec![])
}

fn fn_fprintf(args: &[Value], stdout: &mut String) -> Result<Vec<Value>> {
    if args.is_empty() {
        return Err(MatroError::arity("fprintf", ">= 1", 0));
    }
    stdout.push_str(&format_printf(args)?);
    Ok(vec![])
}

pub fn format_printf(args: &[Value]) -> Result<String> {
    let fmt_str = args[0].as_str()?;
    let mut out = String::new();
    let mut arg_idx = 1usize;
    let chars: Vec<char> = fmt_str.chars().collect();
    let mut i = 0;
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
            let mut precision: Option<usize> = None;
            while j < chars.len()
                && (chars[j].is_ascii_digit() || chars[j] == '.' || chars[j] == '-')
            {
                if chars[j] == '.' {
                    let p_start = j + 1;
                    let mut p_end = p_start;
                    while p_end < chars.len() && chars[p_end].is_ascii_digit() {
                        p_end += 1;
                    }
                    if p_end > p_start {
                        let s: String = chars[p_start..p_end].iter().collect();
                        precision = s.parse().ok();
                    }
                    j = p_end;
                    break;
                }
                j += 1;
            }
            let spec = chars.get(j).copied().unwrap_or('s');
            if let Some(val) = args.get(arg_idx) {
                arg_idx += 1;
                match spec {
                    'd' | 'i' => out.push_str(&format!("{}", val.as_scalar()?.round() as i64)),
                    'f' | 'g' => {
                        let v = val.as_scalar()?;
                        if let Some(p) = precision {
                            out.push_str(&format!("{v:.p$}"));
                        } else {
                            out.push_str(&format!("{v}"));
                        }
                    }
                    'e' => {
                        let v = val.as_scalar()?;
                        if let Some(p) = precision {
                            out.push_str(&format!("{v:.p$e}"));
                        } else {
                            out.push_str(&format!("{v:e}"));
                        }
                    }
                    'E' => {
                        let v = val.as_scalar()?;
                        if let Some(p) = precision {
                            out.push_str(&format!("{v:.p$E}"));
                        } else {
                            out.push_str(&format!("{v:E}"));
                        }
                    }
                    _ => out.push_str(&format!("{val}")),
                }
            }
            i = j + 1;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    Ok(out)
}

fn fn_addpath(args: &[Value], loader: &mut ModuleLoader) -> Result<Vec<Value>> {
    for a in args {
        loader.add_path(a.as_str()?);
    }
    Ok(vec![])
}

fn fn_csvread(args: &[Value], loader: &ModuleLoader) -> Result<Vec<Value>> {
    if args.len() != 1 {
        return Err(MatroError::arity("csvread", "1", args.len()));
    }
    let m = loader.read_csv(args[0].as_str()?)?;
    ok1(Value::matrix(m))
}

fn fn_csvwrite(args: &[Value], loader: &mut ModuleLoader) -> Result<Vec<Value>> {
    if args.len() != 2 {
        return Err(MatroError::arity("csvwrite", "2", args.len()));
    }
    loader.write_csv(args[0].as_str()?, args[1].as_matrix()?)?;
    Ok(vec![])
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
    if args.len() != 1 {
        return Err(MatroError::arity("imagesc", "1", args.len()));
    }
    figure.panel_mut().set_heatmap(args[0].as_matrix()?);
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
