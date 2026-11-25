use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

pub mod complex;
pub mod linalg;
pub mod matrix;
pub mod plot;
pub mod sparse;

pub use complex::Complex;
pub use matrix::{Matrix, NumericClass};
pub use plot::{PlotFigure, PlotKind, PlotPanel, PlotSeries};
pub use sparse::SparseMatrix;

/// Deterministic, zero-dependency xorshift64* PRNG (works identically on native & Wasm).
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
    seed_val: u64,
    gen_type: String,
    shuffle_seq: u64,
}

impl Default for Rng {
    fn default() -> Self {
        Self {
            state: 0x9e37_79b9_7f4a_7c15,
            seed_val: 0,
            gen_type: "twister".to_string(),
            shuffle_seq: 0,
        }
    }
}

impl Rng {
    pub fn canonical_generator(name: &str) -> Option<&'static str> {
        match name.trim().to_ascii_lowercase().as_str() {
            "twister" | "mt19937ar" => Some("twister"),
            "simdtwister" | "dsfmt19937" => Some("simdTwister"),
            "combrecursive" | "mrg32k3a" => Some("combRecursive"),
            "multfibonacci" | "mlfg6331_64" => Some("multFibonacci"),
            "philox" | "philox4x32_10" => Some("philox"),
            "threefry" | "threefry4x64_20" => Some("threefry"),
            "v5uniform" | "mcg16807" => Some("v5uniform"),
            "v5normal" => Some("v5normal"),
            "v4" => Some("v4"),
            _ => None,
        }
    }

    pub fn seed(&mut self, s: u64) {
        self.seed_val = s;
        let salt = Self::gen_salt(&self.gen_type);
        let mixed = s ^ salt;
        self.state = if mixed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            mixed
        };
    }

    pub fn seed_with_generator(&mut self, s: u64, gen_type: &str) {
        self.gen_type = gen_type.to_string();
        self.seed(s);
    }

    pub fn reset_default(&mut self) {
        self.gen_type = "twister".to_string();
        self.seed(0);
    }

    pub fn shuffle(&mut self, gen_override: Option<&str>) {
        if let Some(g) = gen_override {
            self.gen_type = g.to_string();
        }
        self.shuffle_seq = self.shuffle_seq.wrapping_add(1);
        #[cfg(not(target_arch = "wasm32"))]
        let base = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x517c_c1b7_2722_0a95);
        #[cfg(target_arch = "wasm32")]
        let base = self.state.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let s = (base ^ self.shuffle_seq.wrapping_mul(0x9e37_79b9_7f4a_7c15)) & 0xffff_ffff;
        self.seed(s);
    }

    pub fn state_snapshot(&self) -> (&str, u64, u64) {
        (&self.gen_type, self.seed_val, self.state)
    }

    pub fn restore_state(&mut self, gen_type: String, seed_val: u64, state: u64) {
        self.gen_type = gen_type;
        self.seed_val = seed_val;
        self.state = if state == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            state
        };
    }

    fn gen_salt(gen_type: &str) -> u64 {
        match gen_type {
            "twister" => 0,
            "simdTwister" => 0x1111_2222_3333_4444,
            "combRecursive" => 0x2222_3333_4444_5555,
            "multFibonacci" => 0x3333_4444_5555_6666,
            "philox" => 0x4444_5555_6666_7777,
            "threefry" => 0x5555_6666_7777_8888,
            "v5uniform" => 0x6666_7777_8888_9999,
            "v5normal" => 0x7777_8888_9999_aaaa,
            "v4" => 0x8888_9999_aaaa_bbbb,
            _ => 0,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform random float in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Standard normal `N(0, 1)` via Box-Muller transform.
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Source span (1-based line and column) for diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Span {
    pub line: usize,
    pub col: usize,
}

impl Span {
    pub const fn new(line: usize, col: usize) -> Self {
        Self { line, col }
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

/// Unified error type across the Matro runtime.
#[derive(Debug, Clone, PartialEq)]
pub enum MatroError {
    Syntax { span: Span, message: String },
    Compile { span: Span, message: String },
    Runtime { message: String },
    Custom { identifier: String, message: String },
    Dimension { message: String },
    Index { message: String },
    Undefined { name: String },
    Arity { name: String, expected: String, got: usize },
    StepLimit { limit: u64 },
    WithLine { line: u32, inner: Box<MatroError> },
}

impl MatroError {
    pub fn syntax(span: Span, msg: impl Into<String>) -> Self {
        Self::Syntax { span, message: msg.into() }
    }

    pub fn compile(span: Span, msg: impl Into<String>) -> Self {
        Self::Compile { span, message: msg.into() }
    }

    pub fn runtime(msg: impl Into<String>) -> Self {
        Self::Runtime { message: msg.into() }
    }

    pub fn custom(identifier: impl Into<String>, msg: impl Into<String>) -> Self {
        Self::Custom {
            identifier: identifier.into(),
            message: msg.into(),
        }
    }

    pub fn dimension(msg: impl Into<String>) -> Self {
        Self::Dimension { message: msg.into() }
    }

    pub fn index(msg: impl Into<String>) -> Self {
        Self::Index { message: msg.into() }
    }

    pub fn undefined(name: impl Into<String>) -> Self {
        Self::Undefined { name: name.into() }
    }

    pub fn arity(name: impl Into<String>, expected: impl Into<String>, got: usize) -> Self {
        Self::Arity {
            name: name.into(),
            expected: expected.into(),
            got,
        }
    }

    /// Attach a 1-based source line number if not already attached.
    pub fn at_line(self, line: u32) -> Self {
        if line == 0 || matches!(self, Self::WithLine { .. } | Self::Syntax { .. } | Self::Compile { .. }) {
            self
        } else {
            Self::WithLine {
                line,
                inner: Box::new(self),
            }
        }
    }

    pub fn is_step_limit(&self) -> bool {
        match self {
            Self::StepLimit { .. } => true,
            Self::WithLine { inner, .. } => inner.is_step_limit(),
            _ => false,
        }
    }

    pub fn identifier(&self) -> &str {
        match self {
            Self::Syntax { .. } => "Matro:SyntaxError",
            Self::Compile { .. } => "Matro:CompileError",
            Self::Runtime { .. } => "Matro:RuntimeError",
            Self::Custom { identifier, .. } => identifier.as_str(),
            Self::Dimension { .. } => "Matro:DimensionMismatch",
            Self::Index { .. } => "Matro:IndexOutOfBounds",
            Self::Undefined { .. } => "Matro:UndefinedIdentifier",
            Self::Arity { .. } => "Matro:WrongArity",
            Self::StepLimit { .. } => "Matro:StepLimitExceeded",
            Self::WithLine { inner, .. } => inner.identifier(),
        }
    }

    /// Underlying message without the `Line N:` wrapper (useful for `err.message` in `catch`).
    pub fn bare_message(&self) -> String {
        match self {
            Self::WithLine { inner, .. } => inner.bare_message(),
            Self::Custom { message, .. } => message.clone(),
            other => other.to_string(),
        }
    }
}

impl fmt::Display for MatroError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax { span, message } => write!(f, "Syntax error at {span}: {message}"),
            Self::Compile { span, message } => write!(f, "Compile error at {span}: {message}"),
            Self::Runtime { message } => write!(f, "Runtime error: {message}"),
            Self::Custom { identifier, message } => write!(f, "{identifier}: {message}"),
            Self::Dimension { message } => write!(f, "Dimension mismatch: {message}"),
            Self::Index { message } => write!(f, "Index error: {message}"),
            Self::Undefined { name } => write!(f, "Undefined variable or function '{name}'"),
            Self::Arity { name, expected, got } => {
                write!(f, "Wrong number of arguments to '{name}': expected {expected}, got {got}")
            }
            Self::StepLimit { limit } => {
                write!(f, "Execution aborted: exceeded step limit of {limit} instructions")
            }
            Self::WithLine { line, inner } => write!(f, "Line {line}: {inner}"),
        }
    }
}

impl std::error::Error for MatroError {}

pub type Result<T> = std::result::Result<T, MatroError>;

/// 2D column-major MATLAB cell array `{a, b; c, d}`.
#[derive(Debug, Clone, PartialEq)]
pub struct CellArray {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Value>,
    /// Extra dimensions beyond 2D (empty for standard 2D cell arrays, e.g. `[4]` for `2x3x4`).
    pub extra_dims: Vec<usize>,
}

impl CellArray {
    pub fn empty() -> Self {
        Self {
            rows: 0,
            cols: 0,
            data: Vec::new(),
            extra_dims: Vec::new(),
        }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<Value>) -> Self {
        Self {
            rows,
            cols,
            data,
            extra_dims: Vec::new(),
        }
    }

    pub fn dims(&self) -> Vec<usize> {
        let mut d = Vec::with_capacity(2 + self.extra_dims.len());
        d.push(self.rows);
        d.push(self.cols);
        d.extend_from_slice(&self.extra_dims);
        while d.len() > 2 && d.last() == Some(&1) {
            d.pop();
        }
        d
    }

    pub fn ndims(&self) -> usize {
        self.dims().len()
    }

    pub fn dim_size(&self, dim: usize) -> usize {
        match dim {
            0 => self.rows,
            1 => self.cols,
            k => self.extra_dims.get(k - 2).copied().unwrap_or(1),
        }
    }

    pub fn new_empty_nd(mut dims: Vec<usize>) -> Self {
        while dims.len() < 2 {
            dims.push(1);
        }
        while dims.len() > 2 && dims.last() == Some(&1) {
            dims.pop();
        }
        let total: usize = dims.iter().product();
        let rows = dims[0];
        let cols = dims[1];
        let extra_dims = dims[2..].to_vec();
        Self {
            rows,
            cols,
            data: vec![Value::empty(); total],
            extra_dims,
        }
    }

    pub fn linear_index(&self, coords: &[usize]) -> Option<usize> {
        let dims = self.dims();
        let mut stride = 1usize;
        let mut lin = 0usize;
        for (i, &c) in coords.iter().enumerate() {
            let d = dims.get(i).copied().unwrap_or(1);
            if c >= d {
                return None;
            }
            lin += c * stride;
            stride *= d;
        }
        if lin < self.data.len() { Some(lin) } else { None }
    }

    pub fn grow_to_nd(&mut self, req_dims: &[usize]) {
        let old_dims = self.dims();
        let nd = old_dims.len().max(req_dims.len()).max(2);
        let mut new_dims = Vec::with_capacity(nd);
        for i in 0..nd {
            let od = old_dims.get(i).copied().unwrap_or(1);
            let rd = req_dims.get(i).copied().unwrap_or(1);
            new_dims.push(od.max(rd));
        }
        while new_dims.len() > 2 && new_dims.last() == Some(&1) {
            new_dims.pop();
        }
        if new_dims == old_dims {
            return;
        }
        let mut grown = Self::new_empty_nd(new_dims.clone());
        let old_total = self.data.len();
        let mut old_strides = vec![1usize; nd];
        let mut new_strides = vec![1usize; nd];
        for i in 1..nd {
            old_strides[i] = old_strides[i - 1] * old_dims.get(i - 1).copied().unwrap_or(1);
            new_strides[i] = new_strides[i - 1] * new_dims.get(i - 1).copied().unwrap_or(1);
        }
        for lin in 0..old_total {
            let mut rem = lin;
            let mut dst = 0usize;
            for i in 0..nd {
                let od = old_dims.get(i).copied().unwrap_or(1);
                let coord = if od > 0 { rem % od } else { 0 };
                if od > 0 {
                    rem /= od;
                }
                dst += coord * new_strides[i];
            }
            if dst < grown.data.len() {
                grown.data[dst] = std::mem::replace(&mut self.data[lin], Value::empty());
            }
        }
        *self = grown;
    }

    pub fn from_rows(row_items: Vec<Vec<Value>>) -> Result<Self> {
        if row_items.is_empty() {
            return Ok(Self::empty());
        }
        let rows = row_items.len();
        let cols = row_items[0].len();
        for r in &row_items {
            if r.len() != cols {
                return Err(MatroError::dimension(
                    "Cell array rows must have identical column counts",
                ));
            }
        }
        let mut data = vec![Value::empty(); rows * cols];
        for (r, row) in row_items.into_iter().enumerate() {
            for (c, val) in row.into_iter().enumerate() {
                data[c * rows + r] = val;
            }
        }
        Ok(Self {
            rows,
            cols,
            data,
            extra_dims: Vec::new(),
        })
    }

    pub fn new_empty_cells(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![Value::empty(); rows * cols],
            extra_dims: Vec::new(),
        }
    }

    pub fn get(&self, r: usize, c: usize) -> &Value {
        &self.data[c * self.rows + r]
    }

    pub fn set(&mut self, r: usize, c: usize, val: Value) {
        self.data[c * self.rows + r] = val;
    }
}

/// First-class anonymous function / closure handle (`@(x, y) expr` or nested function).
#[derive(Debug, Clone, PartialEq)]
pub struct ClosureHandle {
    pub id: usize,
    pub params: Vec<String>,
    pub captures: BTreeMap<String, Value>,
    pub env_id: Option<usize>,
}

/// Instance of a MATLAB `classdef` value class.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectInstance {
    pub class_name: String,
    pub fields: BTreeMap<String, Value>,
}

/// Reference handle to a mutable MATLAB `< handle` object instance in the VM heap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandleRef {
    pub id: usize,
    pub class_name: String,
}

/// 1D/2D MATLAB struct array (`s(1).a = 1; s(2).a = 2`).
#[derive(Debug, Clone, PartialEq)]
pub struct StructArray {
    pub rows: usize,
    pub cols: usize,
    pub items: Vec<BTreeMap<String, Value>>,
}

/// Column-oriented MATLAB `table` with named variables.
#[derive(Debug, Clone, PartialEq)]
pub struct TableData {
    pub col_names: Vec<String>,
    pub columns: Vec<Value>,
    pub nrows: usize,
}

impl TableData {
    pub fn new(col_names: Vec<String>, columns: Vec<Value>) -> Result<Self> {
        if col_names.len() != columns.len() {
            return Err(MatroError::dimension(
                "table: VariableNames length must match number of columns",
            ));
        }
        let mut nrows: Option<usize> = None;
        let mut norm_cols = Vec::with_capacity(columns.len());
        for col in columns {
            let (col_val, len) = match col {
                Value::Matrix(m) => {
                    let v = if m.rows() == 1 && m.cols() > 1 {
                        Value::matrix(m.transpose())
                    } else {
                        Value::Matrix(m)
                    };
                    let n = v.as_matrix().unwrap().rows();
                    (v, n)
                }
                Value::Cell(c) => {
                    let n = c.data.len();
                    let col_c = CellArray::from_vec(n, if n == 0 { 0 } else { 1 }, c.data);
                    (Value::Cell(col_c), n)
                }
                Value::String(s) => {
                    let col_c = CellArray {
                        rows: 1,
                        cols: 1,
                        data: vec![Value::String(s)],
                        extra_dims: Vec::new(),
                    };
                    (Value::Cell(col_c), 1)
                }
                other => {
                    return Err(MatroError::runtime(format!(
                        "table: unsupported column type {}",
                        other.type_name()
                    )));
                }
            };
            if let Some(expected) = nrows {
                if len != expected {
                    return Err(MatroError::dimension(format!(
                        "table: all variables must have the same number of rows (expected {expected}, got {len})"
                    )));
                }
            } else {
                nrows = Some(len);
            }
            norm_cols.push(col_val);
        }
        Ok(Self {
            col_names,
            columns: norm_cols,
            nrows: nrows.unwrap_or(0),
        })
    }

    pub fn height(&self) -> usize {
        self.nrows
    }

    pub fn width(&self) -> usize {
        self.col_names.len()
    }

    pub fn col_index(&self, name: &str) -> Option<usize> {
        self.col_names.iter().position(|n| n == name)
    }

    pub fn get_col(&self, name: &str) -> Option<&Value> {
        self.col_index(name).map(|i| &self.columns[i])
    }

    pub fn set_col(&mut self, name: String, val: Value) -> Result<()> {
        let (norm, len) = match val {
            Value::Matrix(m) => {
                let v = if m.rows() == 1 && m.cols() > 1 {
                    Value::matrix(m.transpose())
                } else {
                    Value::Matrix(m)
                };
                let n = v.as_matrix().unwrap().rows();
                (v, n)
            }
            Value::Cell(c) => {
                let n = c.data.len();
                (
                    Value::Cell(CellArray::from_vec(n, if n == 0 { 0 } else { 1 }, c.data)),
                    n,
                )
            }
            other => {
                return Err(MatroError::runtime(format!(
                    "table: unsupported column type {}",
                    other.type_name()
                )));
            }
        };
        if !self.columns.is_empty() && len != self.nrows {
            return Err(MatroError::dimension(format!(
                "table: new column has {len} rows, expected {}",
                self.nrows
            )));
        }
        if self.columns.is_empty() {
            self.nrows = len;
        }
        if let Some(idx) = self.col_index(&name) {
            self.columns[idx] = norm;
        } else {
            self.col_names.push(name);
            self.columns.push(norm);
        }
        Ok(())
    }

    pub fn row_elem_str(&self, row: usize, col: usize) -> String {
        match &self.columns[col] {
            Value::Matrix(m) => {
                let c = m.get_c(row, 0);
                if c.im == 0.0 {
                    format!("{}", c.re)
                } else {
                    format!("{c}")
                }
            }
            Value::Cell(c) => match &c.data[row] {
                Value::String(s) => s.clone(),
                other => format!("{other}"),
            },
            other => format!("{other}"),
        }
    }

    pub fn slice(&self, rows: &[usize], cols: &[usize]) -> Result<Self> {
        let mut names = Vec::with_capacity(cols.len());
        let mut out_cols = Vec::with_capacity(cols.len());
        for &c in cols {
            names.push(self.col_names[c].clone());
            let col_val = match &self.columns[c] {
                Value::Matrix(m) => {
                    let vals: Vec<Complex> = rows.iter().map(|&r| m.get_c(r, 0)).collect();
                    Value::matrix(Matrix::complex_col_vec(vals).normalize())
                }
                Value::Cell(cell) => {
                    let vals: Vec<Value> = rows.iter().map(|&r| cell.data[r].clone()).collect();
                    let n = vals.len();
                    Value::Cell(CellArray::from_vec(n, if n == 0 { 0 } else { 1 }, vals))
                }
                other => other.clone(),
            };
            out_cols.push(col_val);
        }
        Ok(Self {
            col_names: names,
            columns: out_cols,
            nrows: rows.len(),
        })
    }
}

/// Key-value dictionary (`containers.Map` / `dictionary`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MapData {
    pub entries: BTreeMap<String, Value>,
}

impl MapData {
    pub fn key_of(v: &Value) -> Result<String> {
        match v {
            Value::String(s) => Ok(s.clone()),
            Value::Matrix(m) => {
                let s = m.as_scalar().ok_or_else(|| {
                    MatroError::runtime("containers.Map key must be a string or scalar number")
                })?;
                if (s - s.round()).abs() < 1e-12 && s.abs() < 1e15 {
                    Ok(format!("{}", s.round() as i64))
                } else {
                    Ok(format!("{s}"))
                }
            }
            other => Err(MatroError::runtime(format!(
                "containers.Map key must be string or scalar, got {}",
                other.type_name()
            ))),
        }
    }
}

/// Core runtime value in Matro.
/// Matrices and sparse matrices use `Arc` for zero-cost Copy-on-Write MATLAB value semantics.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Matrix(Arc<Matrix>),
    Sparse(Arc<SparseMatrix>),
    String(String),
    Struct(BTreeMap<String, Value>),
    StructArray(StructArray),
    Object(ObjectInstance),
    Handle(HandleRef),
    Cell(CellArray),
    Closure(ClosureHandle),
    Table(TableData),
    Map(MapData),
}

impl Value {
    pub fn matrix(m: Matrix) -> Self {
        Self::Matrix(Arc::new(m))
    }

    pub fn sparse(s: SparseMatrix) -> Self {
        Self::Sparse(Arc::new(s))
    }

    pub fn scalar(v: f64) -> Self {
        Self::matrix(Matrix::scalar(v))
    }

    pub fn complex(re: f64, im: f64) -> Self {
        Self::matrix(Matrix::complex_scalar(Complex::new(re, im)))
    }

    pub fn bool(b: bool) -> Self {
        Self::matrix(Matrix::logical_scalar(b))
    }

    pub fn empty() -> Self {
        Self::matrix(Matrix::empty())
    }

    pub fn type_name(&self) -> &str {
        match self {
            Self::Matrix(m) if m.dtype() == NumericClass::Logical => "logical",
            Self::Matrix(m) => m.dtype().name(),
            Self::Sparse(_) => "double",
            Self::String(_) => "char",
            Self::Struct(_) | Self::StructArray(_) => "struct",
            Self::Object(o) => &o.class_name,
            Self::Handle(h) => &h.class_name,
            Self::Cell(_) => "cell",
            Self::Closure(_) => "function_handle",
            Self::Table(_) => "table",
            Self::Map(_) => "containers.Map",
        }
    }

    pub fn dims_str(&self) -> String {
        match self {
            Self::Matrix(m) => m
                .dims()
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("x"),
            Self::Sparse(s) => format!("{}x{} (nnz={})", s.rows(), s.cols(), s.nnz()),
            Self::String(s) => format!("1x{}", s.chars().count()),
            Self::Struct(_) | Self::Object(_) | Self::Handle(_) | Self::Closure(_) => {
                "1x1".to_string()
            }
            Self::StructArray(sa) => format!("{}x{}", sa.rows, sa.cols),
            Self::Cell(c) => format!("{}x{}", c.rows, c.cols),
            Self::Table(t) => format!("{}x{}", t.nrows, t.col_names.len()),
            Self::Map(m) => format!("{}x1", m.entries.len()),
        }
    }

    pub fn as_matrix(&self) -> Result<&Matrix> {
        match self {
            Self::Matrix(m) => Ok(m.as_ref()),
            other => Err(MatroError::runtime(format!(
                "Expected dense numeric matrix, got {}",
                other.type_name()
            ))),
        }
    }

    /// Convert dense or sparse numeric value into an owned dense `Matrix`.
    pub fn to_dense_matrix(&self) -> Result<Matrix> {
        match self {
            Self::Matrix(m) => Ok(m.as_ref().clone()),
            Self::Sparse(s) => Ok(s.to_dense()),
            other => Err(MatroError::runtime(format!(
                "Expected numeric matrix, got {}",
                other.type_name()
            ))),
        }
    }

    /// Copy-on-Write mutable access to the underlying `Matrix`.
    pub fn as_matrix_mut(&mut self) -> Result<&mut Matrix> {
        match self {
            Self::Matrix(m) => Ok(Arc::make_mut(m)),
            other => Err(MatroError::runtime(format!(
                "Expected dense numeric matrix, got {}",
                other.type_name()
            ))),
        }
    }

    pub fn as_scalar(&self) -> Result<f64> {
        if let Self::Sparse(s) = self {
            if s.rows() == 1 && s.cols() == 1 {
                return Ok(s.get(0, 0));
            }
            return Err(MatroError::dimension(format!(
                "Expected real scalar (1x1), got {}x{} sparse",
                s.rows(),
                s.cols()
            )));
        }
        let m = self.as_matrix()?;
        m.as_scalar().ok_or_else(|| {
            MatroError::dimension(format!(
                "Expected real scalar (1x1), got {}x{}{}",
                m.rows(),
                m.cols(),
                if m.is_complex() { " complex" } else { "" }
            ))
        })
    }

    pub fn as_str(&self) -> Result<&str> {
        match self {
            Self::String(s) => Ok(s),
            other => Err(MatroError::runtime(format!(
                "Expected string, got {}",
                other.type_name()
            ))),
        }
    }

    pub fn is_truthy(&self) -> bool {
        match self {
            Self::Matrix(m) => {
                !m.is_empty()
                    && (0..m.len()).all(|i| {
                        let c = m.get_linear_c(i);
                        !c.is_zero() && !c.re.is_nan() && !c.im.is_nan()
                    })
            }
            Self::Sparse(s) => s.nnz() == s.rows().saturating_mul(s.cols()) && s.nnz() > 0,
            Self::String(s) => !s.is_empty(),
            Self::Struct(map) => !map.is_empty(),
            Self::StructArray(sa) => !sa.items.is_empty(),
            Self::Object(_) | Self::Handle(_) | Self::Closure(_) => true,
            Self::Cell(c) => !c.data.is_empty(),
            Self::Table(t) => t.nrows > 0 && !t.columns.is_empty(),
            Self::Map(m) => !m.entries.is_empty(),
        }
    }
}

impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Self::scalar(v)
    }
}

impl From<Matrix> for Value {
    fn from(m: Matrix) -> Self {
        Self::matrix(m)
    }
}

impl From<String> for Value {
    fn from(s: String) -> Self {
        Self::String(s)
    }
}

fn fmt_fields(f: &mut fmt::Formatter<'_>, header: &str, map: &BTreeMap<String, Value>) -> fmt::Result {
    if map.is_empty() {
        return write!(f, "{header} with no properties");
    }
    writeln!(f, "{header} with properties:")?;
    for (i, (k, v)) in map.iter().enumerate() {
        if i > 0 {
            writeln!(f)?;
        }
        let summary = match v {
            Value::Matrix(m) if m.is_scalar() => format!("{m}"),
            Value::Matrix(m) => format!("[{}x{} {}]", m.rows(), m.cols(), v.type_name()),
            Value::Sparse(s) => format!("[{}x{} sparse, nnz={}]", s.rows(), s.cols(), s.nnz()),
            Value::String(s) => format!("'{s}'"),
            Value::Struct(s) => format!("[1x1 struct ({} fields)]", s.len()),
            Value::StructArray(sa) => format!("[{}x{} struct array]", sa.rows, sa.cols),
            Value::Object(o) => format!("[1x1 {}]", o.class_name),
            Value::Handle(h) => format!("[1x1 {} handle]", h.class_name),
            Value::Cell(c) => format!("{{{}x{} cell}}", c.rows, c.cols),
            Value::Closure(cl) => format!("@({})", cl.params.join(",")),
            Value::Table(t) => format!("[{}x{} table]", t.nrows, t.col_names.len()),
            Value::Map(m) => format!("[containers.Map ({} keys)]", m.entries.len()),
        };
        write!(f, "  {k}: {summary}")?;
    }
    Ok(())
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Matrix(m) => write!(f, "{m}"),
            Self::Sparse(s) => write!(f, "{s}"),
            Self::String(s) => write!(f, "{s}"),
            Self::Struct(map) => fmt_fields(f, "struct", map),
            Self::StructArray(sa) => {
                write!(f, "{}x{} struct array", sa.rows, sa.cols)
            }
            Self::Object(o) => fmt_fields(f, &o.class_name, &o.fields),
            Self::Handle(h) => write!(f, "{} handle (#{})", h.class_name, h.id),
            Self::Cell(c) => {
                if c.data.is_empty() {
                    return write!(f, "{{}}");
                }
                for r in 0..c.rows {
                    if r > 0 {
                        writeln!(f)?;
                    }
                    write!(f, "  {{")?;
                    for col in 0..c.cols {
                        if col > 0 {
                            write!(f, ", ")?;
                        }
                        match c.get(r, col) {
                            Value::String(s) => write!(f, "'{s}'")?,
                            Value::Matrix(m) if !m.is_scalar() => write!(
                                f,
                                "[{}x{} {}]",
                                m.rows(),
                                m.cols(),
                                c.get(r, col).type_name()
                            )?,
                            other => write!(f, "{other}")?,
                        }
                    }
                    write!(f, "}}")?;
                }
                Ok(())
            }
            Self::Closure(cl) => write!(f, "@({}) <anonymous>", cl.params.join(", ")),
            Self::Table(t) => {
                if t.col_names.is_empty() {
                    return write!(f, "0x0 empty table");
                }
                let mut widths: Vec<usize> = t.col_names.iter().map(|s| s.len().max(4)).collect();
                let show_rows = t.nrows.min(20);
                for r in 0..show_rows {
                    for c in 0..t.col_names.len() {
                        widths[c] = widths[c].max(t.row_elem_str(r, c).len());
                    }
                }
                for (c, name) in t.col_names.iter().enumerate() {
                    if c > 0 {
                        write!(f, "  ")?;
                    }
                    write!(f, "{name:>w$}", w = widths[c])?;
                }
                writeln!(f)?;
                for (c, &w) in widths.iter().enumerate() {
                    if c > 0 {
                        write!(f, "  ")?;
                    }
                    for _ in 0..w {
                        write!(f, "_")?;
                    }
                }
                for r in 0..show_rows {
                    writeln!(f)?;
                    for (c, &w) in widths.iter().enumerate() {
                        if c > 0 {
                            write!(f, "  ")?;
                        }
                        let s = t.row_elem_str(r, c);
                        write!(f, "{s:>w$}")?;
                    }
                }
                if t.nrows > show_rows {
                    write!(f, "\n  ... ({} more rows)", t.nrows - show_rows)?;
                }
                Ok(())
            }
            Self::Map(m) => {
                write!(f, "containers.Map with {} entries", m.entries.len())
            }
        }
    }
}
