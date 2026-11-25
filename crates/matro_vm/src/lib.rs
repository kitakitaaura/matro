use std::collections::{BTreeMap, HashMap};

use matro_bytecode::{ChainStep, Chunk, ClassProto, FunctionProto, OpCode, compile};
use matro_intrinsics::{IntrinsicContext, call_intrinsic};
use matro_loader::{ModuleLoader, Vfs};
use matro_types::plot::json_str;
use matro_types::{
    CellArray, ClosureHandle, Complex, HandleRef, MapData, MatroError, Matrix, ObjectInstance,
    PlotFigure, Result, Rng, SparseMatrix, StructArray, TableData, Value,
};

mod fileio;
mod krylov;
mod oop;
mod opt_solvers;
mod solvers;
mod solvers_ext;

/// Result of evaluating a script or REPL line.
#[derive(Debug, Clone, Default)]
pub struct EvalOutput {
    pub stdout: String,
    pub last_value: Option<Value>,
}

/// Internal stack slot (regular value, fast unboxed scalar, `:` sentinel, `c{:}` spread, or void script result).
#[derive(Debug, Clone)]
enum StackItem {
    Val(Value),
    Scalar(f64),
    Colon,
    Spread(Vec<Value>),
    Void,
}

impl StackItem {
    #[inline]
    fn from_value_fast(v: &Value) -> Self {
        if let Value::Matrix(m) = v {
            if let Some(x) = m.is_real_non_logical_scalar() {
                return Self::Scalar(x);
            }
        }
        Self::Val(v.clone())
    }

    #[inline]
    fn as_fast_scalar(&self) -> Option<f64> {
        match self {
            Self::Scalar(x) => Some(*x),
            Self::Val(Value::Matrix(m)) => m.is_real_non_logical_scalar(),
            _ => None,
        }
    }

    fn into_val(self) -> Result<Value> {
        match self {
            Self::Val(v) => Ok(v),
            Self::Scalar(x) => Ok(Value::scalar(x)),
            Self::Colon => Err(MatroError::runtime("Unexpected ':' outside indexing")),
            Self::Spread(mut vs) => {
                if vs.is_empty() {
                    Ok(Value::empty())
                } else {
                    Ok(vs.remove(0))
                }
            }
            Self::Void => Ok(Value::empty()),
        }
    }
}

#[derive(Debug, Clone)]
enum ForLoopState {
    Matrix { matrix: Matrix, col_idx: usize, eff_cols: usize },
    Range { start: f64, step: f64, idx: usize, count: usize },
    Values { items: Vec<Value>, idx: usize },
}

#[inline(never)]
fn exec_build_row(stack: &mut Vec<StackItem>, n: usize) -> Result<()> {
    let raw = pop_items(stack, n)?;
    let items = flatten_call_args(raw)?;
    if items.is_empty() {
        stack.push(StackItem::Val(Value::empty()));
    } else if items.iter().all(|v| matches!(v, Value::String(_))) {
        let mut s = String::new();
        for v in items {
            if let Value::String(part) = v {
                s.push_str(&part);
            }
        }
        stack.push(StackItem::Val(Value::String(s)));
    } else if items.iter().any(|v| matches!(v, Value::Sparse(_))) {
        let mut sps: Vec<SparseMatrix> = Vec::with_capacity(items.len());
        for v in &items {
            match v {
                Value::Sparse(s) => {
                    if s.rows() > 0 || s.cols() > 0 {
                        sps.push((**s).clone());
                    }
                }
                other => {
                    let m = other.to_dense_matrix()?;
                    if m.rows() > 0 || m.cols() > 0 {
                        sps.push(SparseMatrix::from_dense(&m));
                    }
                }
            }
        }
        let refs: Vec<&SparseMatrix> = sps.iter().collect();
        let row = SparseMatrix::horzcat(&refs)?;
        stack.push(StackItem::Val(Value::sparse(row)));
    } else {
        let mats: Vec<Matrix> = items
            .iter()
            .map(|v| v.to_dense_matrix())
            .collect::<Result<_>>()?;
        let refs: Vec<&Matrix> = mats.iter().collect();
        let row = Matrix::hcat(&refs)?;
        stack.push(StackItem::Val(Value::matrix(row)));
    }
    Ok(())
}

#[inline(never)]
fn exec_build_matrix(stack: &mut Vec<StackItem>, n: usize) -> Result<()> {
    let items = pop_vals(stack, n)?;
    if items.len() == 1 && matches!(items[0], Value::String(_)) {
        stack.push(StackItem::Val(items.into_iter().next().unwrap()));
    } else if items.iter().any(|v| matches!(v, Value::Sparse(_))) {
        let mut sps: Vec<SparseMatrix> = Vec::with_capacity(items.len());
        for v in &items {
            match v {
                Value::Sparse(s) => {
                    if s.rows() > 0 || s.cols() > 0 {
                        sps.push((**s).clone());
                    }
                }
                other => {
                    let m = other.to_dense_matrix()?;
                    if m.rows() > 0 || m.cols() > 0 {
                        sps.push(SparseMatrix::from_dense(&m));
                    }
                }
            }
        }
        let refs: Vec<&SparseMatrix> = sps.iter().collect();
        let mat = SparseMatrix::vertcat(&refs)?;
        stack.push(StackItem::Val(Value::sparse(mat)));
    } else {
        let mats: Vec<Matrix> = items
            .iter()
            .map(|v| v.to_dense_matrix())
            .collect::<Result<_>>()?;
        let refs: Vec<&Matrix> = mats.iter().collect();
        let mat = Matrix::vcat(&refs)?;
        stack.push(StackItem::Val(Value::matrix(mat)));
    }
    Ok(())
}

#[inline(never)]
fn init_for_loop(val: Value) -> Result<Option<(Value, ForLoopState)>> {
    match val {
        Value::Matrix(m) => {
            let mat = (*m).clone();
            let eff_cols = if mat.rows() == 0 {
                0
            } else if mat.ndims() > 2 {
                mat.len() / mat.rows()
            } else {
                mat.cols()
            };
            if eff_cols == 0 {
                Ok(None)
            } else {
                let first = extract_col(&mat, 0);
                Ok(Some((
                    Value::matrix(first),
                    ForLoopState::Matrix {
                        matrix: mat,
                        col_idx: 1,
                        eff_cols,
                    },
                )))
            }
        }
        Value::Sparse(s) => {
            let mat = s.to_dense();
            let eff_cols = mat.cols();
            if eff_cols == 0 {
                Ok(None)
            } else {
                let first = extract_col(&mat, 0);
                Ok(Some((
                    Value::matrix(first),
                    ForLoopState::Matrix {
                        matrix: mat,
                        col_idx: 1,
                        eff_cols,
                    },
                )))
            }
        }
        Value::Cell(c) => {
            if c.cols == 0 {
                return Ok(None);
            }
            let mut items = Vec::with_capacity(c.cols);
            for col in 0..c.cols {
                let mut rows = Vec::with_capacity(c.rows);
                for row in 0..c.rows {
                    rows.push(vec![c.get(row, col).clone()]);
                }
                items.push(Value::Cell(CellArray::from_rows(rows)?));
            }
            let first = items[0].clone();
            Ok(Some((first, ForLoopState::Values { items, idx: 1 })))
        }
        Value::StructArray(sa) => {
            if sa.cols == 0 {
                return Ok(None);
            }
            let mut items = Vec::with_capacity(sa.cols);
            for col in 0..sa.cols {
                if sa.rows == 1 {
                    items.push(Value::Struct(sa.items[col].clone()));
                } else {
                    let mut col_items = Vec::with_capacity(sa.rows);
                    for row in 0..sa.rows {
                        col_items.push(sa.items[col * sa.rows + row].clone());
                    }
                    items.push(Value::StructArray(matro_types::StructArray {
                        rows: sa.rows,
                        cols: 1,
                        items: col_items,
                    }));
                }
            }
            let first = items[0].clone();
            Ok(Some((first, ForLoopState::Values { items, idx: 1 })))
        }
        Value::Struct(s) => Ok(Some((
            Value::Struct(s),
            ForLoopState::Values {
                items: Vec::new(),
                idx: 1,
            },
        ))),
        Value::String(s) => {
            let chars: Vec<Value> = s.chars().map(|ch| Value::String(ch.to_string())).collect();
            if chars.is_empty() {
                Ok(None)
            } else {
                let first = chars[0].clone();
                Ok(Some((first, ForLoopState::Values { items: chars, idx: 1 })))
            }
        }
        other => {
            let mat = other.to_dense_matrix()?;
            let eff_cols = mat.cols();
            if eff_cols == 0 {
                Ok(None)
            } else {
                let first = extract_col(&mat, 0);
                Ok(Some((
                    Value::matrix(first),
                    ForLoopState::Matrix {
                        matrix: mat,
                        col_idx: 1,
                        eff_cols,
                    },
                )))
            }
        }
    }
}

fn range_count(start: f64, step: f64, stop: f64) -> usize {
    if step == 0.0 || (step > 0.0 && start > stop) || (step < 0.0 && start < stop) {
        return 0;
    }
    let eps = 1e-12 * step.abs().max(1.0);
    let raw = (((stop - start) / step) + eps / step.abs()).floor() as isize + 1;
    if raw <= 0 {
        return 0;
    }
    let mut n = raw as usize;
    while n > 0 {
        let last = start + ((n - 1) as f64) * step;
        if (step > 0.0 && last <= stop + eps) || (step < 0.0 && last >= stop - eps) {
            break;
        }
        n -= 1;
    }
    n
}

#[derive(Debug, Clone, Copy)]
struct TryHandler {
    catch_ip: usize,
    err_var: u32,
    err_is_local: bool,
    stack_depth: usize,
    loop_depth: usize,
    end_depth: usize,
}

#[derive(Debug, Clone)]
struct EndFrame {
    val: Value,
    dim: u8,
    total_dims: u8,
    inherited: bool,
}

/// Fast indexed local frame for functions & closures (no HashMap per call!).
struct LocalFrame<'a> {
    func_name: &'a str,
    source_file: Option<&'a str>,
    slots: Vec<Option<Value>>,
    names: &'a [String],
    captures: Option<&'a BTreeMap<String, Value>>,
    env_id: Option<usize>,
}

impl<'a> LocalFrame<'a> {
    fn new(
        func_name: &'a str,
        source_file: Option<&'a str>,
        names: &'a [String],
        captures: Option<&'a BTreeMap<String, Value>>,
        env_id: Option<usize>,
    ) -> Self {
        Self {
            func_name,
            source_file,
            slots: vec![None; names.len()],
            names,
            captures,
            env_id,
        }
    }

    fn sync_to_env(&self, envs: &mut [BTreeMap<String, Value>]) {
        if let Some(eid) = self.env_id {
            if let Some(env) = envs.get_mut(eid) {
                for (name, slot) in self.names.iter().zip(self.slots.iter()) {
                    if name == "nargin" || name == "nargout" || name == "ans" {
                        continue;
                    }
                    if let Some(v) = slot {
                        env.insert(name.clone(), v.clone());
                    }
                }
            }
        }
    }

    fn sync_from_env(&mut self, envs: &[BTreeMap<String, Value>]) {
        if let Some(eid) = self.env_id {
            if let Some(env) = envs.get(eid) {
                for (name, slot) in self.names.iter().zip(self.slots.iter_mut()) {
                    if name == "nargin" || name == "nargout" || name == "ans" {
                        continue;
                    }
                    if let Some(v) = env.get(name) {
                        *slot = Some(v.clone());
                    }
                }
            }
        }
    }

    #[inline]
    fn get(&self, slot: u16) -> Option<&Value> {
        self.slots.get(slot as usize).and_then(Option::as_ref)
    }

    #[inline]
    fn set(&mut self, slot: u16, val: Value) {
        if let Some(cell) = self.slots.get_mut(slot as usize) {
            *cell = Some(val);
        }
    }

    /// Zero-allocation scalar update when the local slot uniquely owns a scalar `Arc<Matrix>`.
    #[inline]
    fn set_scalar(&mut self, slot: u16, x: f64) {
        if let Some(cell) = self.slots.get_mut(slot as usize) {
            if let Some(Value::Matrix(arc)) = cell {
                if let Some(m) = std::sync::Arc::get_mut(arc) {
                    if m.set_scalar_in_place(x) {
                        return;
                    }
                }
            }
            *cell = Some(Value::scalar(x));
        }
    }

    #[inline]
    fn get_mut(&mut self, slot: u16) -> Option<&mut Value> {
        self.slots.get_mut(slot as usize).and_then(Option::as_mut)
    }

    fn take(&mut self, slot: u16) -> Option<Value> {
        self.slots.get_mut(slot as usize).and_then(Option::take)
    }

    fn get_by_name(&self, name: &str) -> Option<&Value> {
        if let Some(pos) = self.names.iter().position(|n| n == name) {
            if let Some(v) = &self.slots[pos] {
                return Some(v);
            }
        }
        self.captures.and_then(|c| c.get(name))
    }

    fn set_by_name(&mut self, name: &str, val: Value) -> bool {
        if let Some(pos) = self.names.iter().position(|n| n == name) {
            self.slots[pos] = Some(val);
            return true;
        }
        false
    }

    fn snapshot_captures(&self) -> BTreeMap<String, Value> {
        let mut map = BTreeMap::new();
        if let Some(caps) = self.captures {
            for (k, v) in caps {
                map.insert(k.clone(), v.clone());
            }
        }
        for (name, slot) in self.names.iter().zip(self.slots.iter()) {
            if let Some(v) = slot {
                map.insert(name.clone(), v.clone());
            }
        }
        map
    }
}

#[derive(Debug, Default, Clone)]
pub struct ProfileEntry {
    pub calls: u64,
    pub instructions: u64,
}

#[derive(Debug, Clone)]
enum TicMark {
    #[cfg(not(target_arch = "wasm32"))]
    Instant(std::time::Instant),
    #[cfg(target_arch = "wasm32")]
    Steps(u64),
}

/// Compiled `.m` file metadata with file-scoped local subfunctions.
#[derive(Debug, Clone)]
struct LoadedFile {
    path: String,
    primary_name: String,
    primary_func: Option<FunctionProto>,
    primary_class: Option<ClassProto>,
    local_funcs: HashMap<String, FunctionProto>,
    is_script: bool,
    chunk: Chunk,
}

/// Embeddable, pure-Rust MATLAB/Octave bytecode Virtual Machine.
#[derive(Debug)]
pub struct Vm {
    globals: HashMap<String, Value>,
    persistents: HashMap<(String, String), Value>,
    functions: HashMap<String, FunctionProto>,
    classes: HashMap<String, ClassProto>,
    loaded_files: HashMap<String, LoadedFile>,
    active_file: Option<String>,
    handles: Vec<ObjectInstance>,
    listeners: HashMap<(usize, String), Vec<ClosureHandle>>,
    imports: Vec<String>,
    closures: Vec<FunctionProto>,
    envs: Vec<BTreeMap<String, Value>>,
    figure: PlotFigure,
    rng: Rng,
    loader: ModuleLoader,
    files: fileio::FileTable,
    step_limit: Option<u64>,
    steps: u64,
    tic_stack: Vec<TicMark>,
    profiling: bool,
    profile: HashMap<String, ProfileEntry>,
    last_error: Option<(String, String, u32)>,
}

impl Default for Vm {
    fn default() -> Self {
        Self::new()
    }
}

impl Vm {
    pub fn new() -> Self {
        Self {
            globals: HashMap::new(),
            persistents: HashMap::new(),
            functions: HashMap::new(),
            classes: HashMap::new(),
            loaded_files: HashMap::new(),
            active_file: None,
            handles: Vec::new(),
            listeners: HashMap::new(),
            imports: Vec::new(),
            closures: Vec::new(),
            envs: Vec::new(),
            figure: PlotFigure::default(),
            rng: Rng::default(),
            loader: ModuleLoader::default(),
            files: fileio::FileTable::new(),
            step_limit: None,
            steps: 0,
            tic_stack: Vec::new(),
            profiling: false,
            profile: HashMap::new(),
            last_error: None,
        }
    }

    pub fn with_vfs(vfs: Box<dyn Vfs>) -> Self {
        Self {
            loader: ModuleLoader::with_vfs(vfs),
            ..Self::new()
        }
    }

    pub fn set_step_limit(&mut self, limit: Option<u64>) {
        self.step_limit = limit;
    }

    pub fn step_limit(&self) -> Option<u64> {
        self.step_limit
    }

    pub fn loader(&self) -> &ModuleLoader {
        &self.loader
    }

    pub fn loader_mut(&mut self) -> &mut ModuleLoader {
        &mut self.loader
    }

    fn invalidate_file(&mut self, path: &str) {
        let key = matro_loader::normalize_preserve_abs(path);
        self.loaded_files.remove(&key);
        self.functions
            .retain(|_, f| f.source_file.as_deref() != Some(key.as_str()));
        self.classes
            .retain(|_, c| c.source_file.as_deref() != Some(key.as_str()));
    }

    pub fn mount_file(&mut self, path: &str, content: &str) -> Result<()> {
        self.invalidate_file(path);
        self.loader.mount_file(path, content)
    }

    pub fn mount_bytes(&mut self, path: &str, data: &[u8]) -> Result<()> {
        self.invalidate_file(path);
        self.loader.mount_bytes(path, data)
    }

    pub fn add_path(&mut self, dir: &str) {
        self.loader.add_path(dir);
    }

    pub fn active_file(&self) -> Option<&str> {
        self.active_file.as_deref()
    }

    pub fn set_active_file(&mut self, path: Option<&str>) {
        self.active_file = path
            .map(matro_loader::MemoryVfs::normalize)
            .filter(|s| !s.is_empty());
    }

    pub fn globals(&self) -> &HashMap<String, Value> {
        &self.globals
    }

    pub fn figure(&self) -> &PlotFigure {
        &self.figure
    }

    pub fn figure_mut(&mut self) -> &mut PlotFigure {
        &mut self.figure
    }

    pub fn get_var(&self, name: &str) -> Option<&Value> {
        self.globals.get(name)
    }

    pub fn set_var(&mut self, name: impl Into<String>, val: Value) {
        self.globals.insert(name.into(), val);
    }

    pub fn clear(&mut self) {
        self.globals.clear();
        self.persistents.clear();
        self.functions.clear();
        self.classes.clear();
        self.loaded_files.clear();
        self.active_file = None;
        self.handles.clear();
        self.listeners.clear();
        self.imports.clear();
        self.closures.clear();
        self.envs.clear();
        self.figure.clear();
        self.loader.reset_search_paths();
        self.loader.clear_cache();
        self.files.clear();
        self.tic_stack.clear();
        self.profile.clear();
        self.profiling = false;
    }

    /// Serialize workspace variables as JSON for IDE / browser playground inspectors.
    pub fn vars_json(&self) -> String {
        let mut names: Vec<&String> = self.globals.keys().collect();
        names.sort();
        let mut out = String::from("[");
        for (i, name) in names.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let val = &self.globals[*name];
            let preview = match val {
                Value::Matrix(m) if m.is_scalar() => format!("{m}"),
                Value::Matrix(m) if m.len() <= 6 => format!("{m}").replace('\n', "; "),
                Value::Matrix(_) => format!("[{} {}]", val.dims_str(), val.type_name()),
                Value::Sparse(s) => format!("[{}x{} sparse, nnz={}]", s.rows(), s.cols(), s.nnz()),
                Value::String(s) => format!("'{s}'"),
                Value::Struct(map) => format!("struct ({} fields)", map.len()),
                Value::StructArray(sa) => format!("[{}x{} struct]", sa.rows, sa.cols),
                Value::Object(o) => format!("{} ({} props)", o.class_name, o.fields.len()),
                Value::Handle(h) => format!("{} handle", h.class_name),
                Value::Cell(c) => format!("{{{}x{} cell}}", c.rows, c.cols),
                Value::Closure(cl) => format!("@({})", cl.params.join(",")),
                Value::Table(t) => format!("[{}x{} table]", t.nrows, t.col_names.len()),
                Value::Map(m) => format!("[containers.Map ({} keys)]", m.entries.len()),
            };
            out.push('{');
            out.push_str(&format!("\"name\":{},", json_str(name)));
            out.push_str(&format!("\"size\":{},", json_str(&val.dims_str())));
            out.push_str(&format!("\"class\":{},", json_str(val.type_name())));
            out.push_str(&format!("\"preview\":{}", json_str(&preview)));
            out.push('}');
        }
        out.push(']');
        out
    }

    /// Serialize VFS file list as JSON.
    pub fn files_json(&self) -> String {
        let files = self.loader.vfs().list_files();
        let mut out = String::from("[");
        for (i, f) in files.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&json_str(f));
        }
        out.push(']');
        out
    }

    /// Compile and execute a MATLAB/Octave source snippet in this VM session.
    pub fn eval(&mut self, source: &str) -> Result<EvalOutput> {
        let chunk = compile(source)?;
        self.execute_chunk(&chunk)
    }

    /// Compile and execute a `.m` file with full file-path context (sibling & `private/` resolution).
    /// When `auto_invoke_zero_arg` is `true` and the file is a pure function file whose primary
    /// function takes 0 parameters (e.g. `function r = main()`), it is automatically invoked.
    pub fn eval_file(
        &mut self,
        path: Option<&str>,
        source: &str,
        auto_invoke_zero_arg: bool,
    ) -> Result<EvalOutput> {
        self.steps = 0;
        let mut chunk = compile(source)?;
        let norm_path = path
            .map(matro_loader::normalize_preserve_abs)
            .filter(|s| !s.is_empty());
        let file_key = norm_path
            .clone()
            .unwrap_or_else(|| "__editor_buffer__.m".to_string());
        chunk.set_source_file(&file_key);
        if let Some(ref np) = norm_path {
            self.active_file = Some(np.clone());
        } else {
            self.active_file = Some(file_key.clone());
        }

        let mut stdout = String::new();
        let mut last_value = None;

        if chunk.is_definition_only() && !chunk.functions.is_empty() && chunk.classes.is_empty() {
            let primary = chunk.functions[0].clone();
            let stem = if norm_path.is_some() {
                matro_loader::file_stem(&file_key)
            } else {
                primary.name.clone()
            };
            let mut local_funcs = HashMap::new();
            if primary.name != stem {
                local_funcs.insert(primary.name.clone(), primary.clone());
            }
            for sub in chunk.functions.iter().skip(1) {
                local_funcs.insert(sub.name.clone(), sub.clone());
            }
            let lf = LoadedFile {
                path: file_key.clone(),
                primary_name: stem.clone(),
                primary_func: Some(primary.clone()),
                primary_class: None,
                local_funcs,
                is_script: false,
                chunk,
            };
            self.loaded_files.insert(file_key, lf);
            // Also register primary function so follow-up REPL calls work by name.
            self.functions.insert(stem, primary.clone());
            if auto_invoke_zero_arg && primary.params.is_empty() {
                let want = if primary.returns.is_empty() { 0 } else { 1 };
                let rets = self.call_user_func(&primary, &[], want, None, &mut stdout)?;
                if let Some(v) = rets.into_iter().next() {
                    last_value = Some(v.clone());
                    self.globals.insert("ans".to_string(), v);
                }
            }
            return Ok(EvalOutput { stdout, last_value });
        }

        // Script or classdef file: register any script-local functions scoped to `file_key`.
        if !chunk.functions.is_empty() {
            let mut local_funcs = HashMap::new();
            for f in &chunk.functions {
                local_funcs.insert(f.name.clone(), f.clone());
            }
            let lf = LoadedFile {
                path: file_key.clone(),
                primary_name: matro_loader::file_stem(&file_key),
                primary_func: None,
                primary_class: chunk.classes.first().cloned(),
                local_funcs,
                is_script: !chunk.is_definition_only(),
                chunk: chunk.clone(),
            };
            self.loaded_files.insert(file_key, lf);
        }

        self.run_frame(&chunk, None, &mut stdout, &mut last_value)?;
        Ok(EvalOutput { stdout, last_value })
    }

    /// Execute a precompiled bytecode `Chunk` at top-level scope.
    pub fn execute_chunk(&mut self, chunk: &Chunk) -> Result<EvalOutput> {
        self.steps = 0;
        let mut stdout = String::new();
        let mut last_value = None;
        self.run_frame(chunk, None, &mut stdout, &mut last_value)?;
        Ok(EvalOutput { stdout, last_value })
    }

    fn run_frame(
        &mut self,
        chunk: &Chunk,
        mut frame: Option<&mut LocalFrame<'_>>,
        stdout: &mut String,
        last_value: &mut Option<Value>,
    ) -> Result<()> {
        let mut stack: Vec<StackItem> = Vec::with_capacity(64);
        let mut for_loops: Vec<ForLoopState> = Vec::new();
        let mut try_stack: Vec<TryHandler> = Vec::new();
        let mut end_targets: Vec<Option<EndFrame>> = Vec::new();
        let mut ip: usize = 0;

        while ip < chunk.code.len() {
            self.steps += 1;
            if let Some(limit) = self.step_limit {
                if self.steps > limit {
                    return Err(MatroError::StepLimit { limit }.at_line(chunk.line_at(ip)));
                }
            }

            let op_ip = ip;
            let op = chunk.code[op_ip];
            ip += 1;

            let step_res = self.exec_op(
                op,
                chunk,
                &mut ip,
                &mut stack,
                &mut for_loops,
                &mut try_stack,
                &mut end_targets,
                frame.as_deref_mut(),
                stdout,
                last_value,
            );

            match step_res {
                Ok(true) => {}
                Ok(false) => break,
                Err(err) => {
                    let line = chunk.line_at(op_ip);
                    if err.is_step_limit() {
                        return Err(err.at_line(line));
                    }
                    let err_id = err.identifier().to_string();
                    let err_msg = err.bare_message();
                    self.last_error = Some((err_id.clone(), err_msg.clone(), line));
                    if let Some(handler) = try_stack.pop() {
                        stack.truncate(handler.stack_depth);
                        for_loops.truncate(handler.loop_depth);
                        end_targets.truncate(handler.end_depth);
                        if handler.err_var != u32::MAX {
                            let err_val = make_mexception(&err_id, &err_msg, line);
                            if handler.err_is_local {
                                if let Some(f) = frame.as_deref_mut() {
                                    f.set(handler.err_var as u16, err_val);
                                }
                            } else {
                                let name = chunk.names[handler.err_var as usize].clone();
                                self.globals.insert(name, err_val);
                            }
                        }
                        ip = handler.catch_ip;
                    } else {
                        return Err(err.at_line(line));
                    }
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn exec_op(
        &mut self,
        op: OpCode,
        chunk: &Chunk,
        ip: &mut usize,
        stack: &mut Vec<StackItem>,
        for_loops: &mut Vec<ForLoopState>,
        try_stack: &mut Vec<TryHandler>,
        end_targets: &mut Vec<Option<EndFrame>>,
        mut frame: Option<&mut LocalFrame<'_>>,
        stdout: &mut String,
        last_value: &mut Option<Value>,
    ) -> Result<bool> {
        match op {
            OpCode::LoadConst(idx) => {
                stack.push(StackItem::from_value_fast(&chunk.constants[idx as usize]));
            }
            OpCode::LoadLocal(slot) => {
                let f = frame
                    .as_deref()
                    .ok_or_else(|| MatroError::runtime("LoadLocal outside function frame"))?;
                if let Some(val) = f.get(slot) {
                    stack.push(StackItem::from_value_fast(val));
                } else {
                    let name = f.names[slot as usize].as_str();
                    return Err(MatroError::undefined(name));
                }
            }
            OpCode::StoreLocal(slot) => {
                let item = stack
                    .pop()
                    .ok_or_else(|| MatroError::runtime("VM stack underflow"))?;
                let f = frame
                    .as_deref_mut()
                    .ok_or_else(|| MatroError::runtime("StoreLocal outside function frame"))?;
                match item {
                    StackItem::Scalar(x) => f.set_scalar(slot, x),
                    other => {
                        let val = other.into_val()?;
                        f.set(slot, val);
                    }
                }
            }
            OpCode::InitGlobal(idx) => {
                let name = chunk.names[idx as usize].clone();
                self.globals.entry(name).or_insert_with(Value::empty);
            }
            OpCode::LoadGlobal(idx) => {
                let name = &chunk.names[idx as usize];
                let val = self.globals.get(name).cloned().unwrap_or_else(Value::empty);
                stack.push(StackItem::Val(val));
            }
            OpCode::StoreGlobal(idx) => {
                let name = chunk.names[idx as usize].clone();
                let val = pop_val(stack)?;
                *last_value = Some(val.clone());
                self.globals.insert(name, val);
            }
            OpCode::InitPersistent(idx) => {
                let fn_name = frame.as_deref().map_or("<main>", |f| f.func_name).to_string();
                let var_name = chunk.names[idx as usize].clone();
                self.persistents
                    .entry((fn_name, var_name))
                    .or_insert_with(Value::empty);
            }
            OpCode::LoadPersistent(idx) => {
                let fn_name = frame.as_deref().map_or("<main>", |f| f.func_name).to_string();
                let var_name = chunk.names[idx as usize].clone();
                let val = self
                    .persistents
                    .get(&(fn_name, var_name))
                    .cloned()
                    .unwrap_or_else(Value::empty);
                stack.push(StackItem::Val(val));
            }
            OpCode::StorePersistent(idx) => {
                let fn_name = frame.as_deref().map_or("<main>", |f| f.func_name).to_string();
                let var_name = chunk.names[idx as usize].clone();
                let val = pop_val(stack)?;
                *last_value = Some(val.clone());
                self.persistents.insert((fn_name, var_name), val);
            }
            OpCode::PrintLocal(slot) => {
                let val = pop_val(stack)?;
                let name = frame
                    .as_deref()
                    .and_then(|f| f.names.get(slot as usize))
                    .map(String::as_str)
                    .unwrap_or("ans");
                format_assignment(stdout, name, &val);
            }
            OpCode::LoadVar(idx) => {
                let name = chunk.names[idx as usize].clone();
                let env_val = frame
                    .as_deref()
                    .and_then(|f| f.env_id)
                    .and_then(|eid| self.envs.get(eid))
                    .and_then(|env| env.get(&name))
                    .cloned();
                if let Some(val) = lookup_var(&name, frame.as_deref(), &self.globals) {
                    stack.push(StackItem::Val(val.clone()));
                } else if let Some(val) = env_val {
                    stack.push(StackItem::Val(val));
                } else if name == "tic" {
                    self.do_tic();
                    stack.push(StackItem::Val(Value::empty()));
                } else if name == "toc" {
                    let elapsed = self.do_toc();
                    stack.push(StackItem::Val(Value::scalar(elapsed)));
                } else if name == "containers" {
                    stack.push(StackItem::Val(Value::Object(ObjectInstance {
                        class_name: "__ns_containers".to_string(),
                        fields: BTreeMap::new(),
                    })));
                } else if name == "inputParser" {
                    let handle = self.new_input_parser();
                    stack.push(StackItem::Val(handle));
                } else {
                    let item = self.resolve_bare_ident(&name, frame.as_deref_mut(), stdout)?;
                    stack.push(item);
                }
            }
            OpCode::StoreVar(idx) => {
                let name = chunk.names[idx as usize].clone();
                let val = pop_val(stack)?;
                *last_value = Some(val.clone());
                if let Some(f) = frame.as_deref_mut() {
                    if f.set_by_name(&name, val.clone()) {
                        f.sync_to_env(&mut self.envs);
                    } else if let Some(eid) = f.env_id {
                        if let Some(env) = self.envs.get_mut(eid) {
                            env.insert(name, val);
                        }
                    } else {
                        self.globals.insert(name, val);
                    }
                } else {
                    self.globals.insert(name, val);
                }
            }
            OpCode::Pop => {
                stack.pop();
            }
            OpCode::Dup => {
                let top = stack
                    .last()
                    .ok_or_else(|| MatroError::runtime("Stack underflow on Dup"))?
                    .clone();
                stack.push(top);
            }
            OpCode::PrintVar(idx) => {
                let name = &chunk.names[idx as usize];
                let val = pop_val(stack)?;
                format_assignment(stdout, name, &val);
            }
            OpCode::PrintAns => {
                if matches!(stack.last(), Some(StackItem::Void)) {
                    stack.pop();
                    return Ok(true);
                }
                let val = pop_val(stack)?;
                *last_value = Some(val.clone());
                store_var(
                    "ans".to_string(),
                    val.clone(),
                    frame.as_deref_mut(),
                    &mut self.globals,
                );
                format_assignment(stdout, "ans", &val);
            }
            OpCode::BuildRow(n) => exec_build_row(stack, n as usize)?,
            OpCode::BuildMatrix(n) => exec_build_matrix(stack, n as usize)?,
            OpCode::BuildCell { rows, cols } => {
                let total = (rows as usize) * (cols as usize);
                let raw = pop_items(stack, total)?;
                let cell = if rows == 1 {
                    let flat = flatten_call_args(raw)?;
                    if flat.is_empty() {
                        CellArray::empty()
                    } else {
                        CellArray::from_rows(vec![flat])?
                    }
                } else {
                    let flat: Vec<Value> = raw
                        .into_iter()
                        .map(StackItem::into_val)
                        .collect::<Result<_>>()?;
                    let mut row_vecs = Vec::with_capacity(rows as usize);
                    if rows > 0 {
                        for chunk_row in flat.chunks(cols as usize) {
                            row_vecs.push(chunk_row.to_vec());
                        }
                    }
                    CellArray::from_rows(row_vecs)?
                };
                stack.push(StackItem::Val(Value::Cell(cell)));
            }
            OpCode::MakeRange { has_step } => {
                let stop = pop_val(stack)?.as_scalar()?;
                let step = if has_step {
                    pop_val(stack)?.as_scalar()?
                } else {
                    1.0
                };
                let start = pop_val(stack)?.as_scalar()?;
                stack.push(StackItem::Val(Value::matrix(Matrix::range(
                    start, step, stop,
                ))));
            }
            OpCode::Neg => {
                if let Some(StackItem::Scalar(x)) = stack.last_mut() {
                    *x = -*x;
                } else {
                    self.exec_unop_slow(op, stack, stdout)?;
                }
            }
            OpCode::Pos => {
                if !matches!(stack.last(), Some(StackItem::Scalar(_))) {
                    let v = pop_val(stack)?;
                    let out = match v {
                        Value::String(s) => {
                            let codes: Vec<f64> = s.chars().map(|c| c as u32 as f64).collect();
                            Value::matrix(Matrix::row_vec(codes))
                        }
                        Value::Matrix(m) if m.is_logical() => {
                            Value::matrix(m.as_ref().clone().with_logical(false))
                        }
                        other => other,
                    };
                    stack.push(StackItem::Val(out));
                }
            }
            OpCode::Not => {
                self.exec_unop_slow(op, stack, stdout)?;
            }
            OpCode::CTranspose | OpCode::Transpose => {
                if !matches!(stack.last(), Some(StackItem::Scalar(_))) {
                    self.exec_unop_slow(op, stack, stdout)?;
                }
            }
            OpCode::Add
            | OpCode::Sub
            | OpCode::ElemMul
            | OpCode::ElemDiv
            | OpCode::ElemLeftDiv
            | OpCode::Mul
            | OpCode::Div => {
                let b_item = stack.pop().ok_or_else(|| MatroError::runtime("VM stack underflow"))?;
                let a_item = stack.pop().ok_or_else(|| MatroError::runtime("VM stack underflow"))?;
                if let (Some(x), Some(y)) = (a_item.as_fast_scalar(), b_item.as_fast_scalar()) {
                    let v = match op {
                        OpCode::Add => x + y,
                        OpCode::Sub => x - y,
                        OpCode::ElemMul | OpCode::Mul => x * y,
                        OpCode::ElemLeftDiv => y / x,
                        _ => x / y,
                    };
                    stack.push(StackItem::Scalar(v));
                } else {
                    self.exec_binop_slow(op, a_item, b_item, stack, stdout)?;
                }
            }
            OpCode::ElemPow | OpCode::LeftDiv | OpCode::Pow | OpCode::Eq | OpCode::Ne => {
                let b_item = stack.pop().ok_or_else(|| MatroError::runtime("VM stack underflow"))?;
                let a_item = stack.pop().ok_or_else(|| MatroError::runtime("VM stack underflow"))?;
                self.exec_binop_slow(op, a_item, b_item, stack, stdout)?;
            }
            OpCode::Lt => self.exec_cmp_op(stack, "lt", |a, b| a < b, stdout)?,
            OpCode::Le => self.exec_cmp_op(stack, "le", |a, b| a <= b, stdout)?,
            OpCode::Gt => self.exec_cmp_op(stack, "gt", |a, b| a > b, stdout)?,
            OpCode::Ge => self.exec_cmp_op(stack, "ge", |a, b| a >= b, stdout)?,
            OpCode::And => bin_logical(stack, |a, b| a != 0.0 && b != 0.0)?,
            OpCode::Or => bin_logical(stack, |a, b| a != 0.0 || b != 0.0)?,
            OpCode::Jump(target) => {
                *ip = target as usize;
            }
            OpCode::JumpIfFalse(target) => {
                let truthy = match stack.pop() {
                    Some(StackItem::Scalar(x)) => x != 0.0 && !x.is_nan(),
                    Some(other) => other.into_val()?.is_truthy(),
                    None => return Err(MatroError::runtime("VM stack underflow")),
                };
                if !truthy {
                    *ip = target as usize;
                }
            }
            OpCode::JumpIfFalsePeek(target) => {
                let truthy = match stack.last() {
                    Some(StackItem::Scalar(x)) => *x != 0.0 && !x.is_nan(),
                    Some(StackItem::Val(v)) => v.is_truthy(),
                    _ => return Err(MatroError::runtime("Stack underflow")),
                };
                if !truthy {
                    *ip = target as usize;
                }
            }
            OpCode::JumpIfTruePeek(target) => {
                let truthy = match stack.last() {
                    Some(StackItem::Scalar(x)) => *x != 0.0 && !x.is_nan(),
                    Some(StackItem::Val(v)) => v.is_truthy(),
                    _ => return Err(MatroError::runtime("Stack underflow")),
                };
                if truthy {
                    *ip = target as usize;
                }
            }
            OpCode::SwitchMatch => {
                let case_val = pop_val(stack)?;
                let matched = match stack.last() {
                    Some(StackItem::Val(v)) => switch_matches(v, &case_val),
                    Some(StackItem::Scalar(x)) => switch_matches(&Value::scalar(*x), &case_val),
                    _ => return Err(MatroError::runtime("Stack underflow in switch")),
                };
                stack.push(StackItem::Val(Value::bool(matched)));
            }
            OpCode::TryEnter {
                catch_ip,
                err_var,
                err_is_local,
            } => {
                try_stack.push(TryHandler {
                    catch_ip: catch_ip as usize,
                    err_var,
                    err_is_local,
                    stack_depth: stack.len(),
                    loop_depth: for_loops.len(),
                    end_depth: end_targets.len(),
                });
            }
            OpCode::TryExit => {
                try_stack.pop();
            }
            OpCode::ForInit { var_idx, exit_ip } => {
                let iter_val = pop_val(stack)?;
                if let Some((first, state)) = init_for_loop(iter_val)? {
                    let name = chunk.names[var_idx as usize].clone();
                    store_var(name, first, frame.as_deref_mut(), &mut self.globals);
                    for_loops.push(state);
                } else {
                    *ip = exit_ip as usize;
                }
            }
            OpCode::ForRangeInit {
                var_idx,
                exit_ip,
                has_step,
            } => {
                let stop = pop_val(stack)?.as_scalar()?;
                let step = if has_step {
                    pop_val(stack)?.as_scalar()?
                } else {
                    1.0
                };
                let start = pop_val(stack)?.as_scalar()?;
                let count = range_count(start, step, stop);
                if count == 0 {
                    *ip = exit_ip as usize;
                } else {
                    let name = chunk.names[var_idx as usize].clone();
                    store_var(
                        name,
                        Value::scalar(start),
                        frame.as_deref_mut(),
                        &mut self.globals,
                    );
                    for_loops.push(ForLoopState::Range {
                        start,
                        step,
                        idx: 1,
                        count,
                    });
                }
            }
            OpCode::ForNext { var_idx, loop_ip } => {
                let state = for_loops
                    .last_mut()
                    .ok_or_else(|| MatroError::runtime("ForNext without active loop"))?;
                let next_val = match state {
                    ForLoopState::Range {
                        start,
                        step,
                        idx,
                        count,
                    } => {
                        if *idx < *count {
                            let v = Value::scalar(*start + (*idx as f64) * (*step));
                            *idx += 1;
                            Some(v)
                        } else {
                            None
                        }
                    }
                    ForLoopState::Matrix {
                        matrix,
                        col_idx,
                        eff_cols,
                    } => {
                        if *col_idx < *eff_cols {
                            let col = extract_col(matrix, *col_idx);
                            *col_idx += 1;
                            Some(Value::matrix(col))
                        } else {
                            None
                        }
                    }
                    ForLoopState::Values { items, idx } => {
                        if *idx < items.len() {
                            let v = items[*idx].clone();
                            *idx += 1;
                            Some(v)
                        } else {
                            None
                        }
                    }
                };
                if let Some(v) = next_val {
                    let name = chunk.names[var_idx as usize].clone();
                    store_var(name, v, frame.as_deref_mut(), &mut self.globals);
                    *ip = loop_ip as usize;
                } else {
                    for_loops.pop();
                }
            }
            OpCode::ForInitLocal { slot, exit_ip } => {
                let iter_val = pop_val(stack)?;
                if let Some((first, state)) = init_for_loop(iter_val)? {
                    let f = frame
                        .as_deref_mut()
                        .ok_or_else(|| MatroError::runtime("ForInitLocal outside function"))?;
                    f.set(slot, first);
                    for_loops.push(state);
                } else {
                    *ip = exit_ip as usize;
                }
            }
            OpCode::ForRangeInitLocal {
                slot,
                exit_ip,
                has_step,
            } => {
                let stop = pop_val(stack)?.as_scalar()?;
                let step = if has_step {
                    pop_val(stack)?.as_scalar()?
                } else {
                    1.0
                };
                let start = pop_val(stack)?.as_scalar()?;
                let count = range_count(start, step, stop);
                if count == 0 {
                    *ip = exit_ip as usize;
                } else {
                    let f = frame
                        .as_deref_mut()
                        .ok_or_else(|| MatroError::runtime("ForRangeInitLocal outside function"))?;
                    f.set(slot, Value::scalar(start));
                    for_loops.push(ForLoopState::Range {
                        start,
                        step,
                        idx: 1,
                        count,
                    });
                }
            }
            OpCode::ForNextLocal { slot, loop_ip } => {
                let state = for_loops
                    .last_mut()
                    .ok_or_else(|| MatroError::runtime("ForNextLocal without active loop"))?;
                match state {
                    ForLoopState::Range {
                        start,
                        step,
                        idx,
                        count,
                    } => {
                        if *idx < *count {
                            let val = *start + (*idx as f64) * (*step);
                            *idx += 1;
                            let f = frame.as_deref_mut().ok_or_else(|| {
                                MatroError::runtime("ForNextLocal outside function")
                            })?;
                            f.set_scalar(slot, val);
                            *ip = loop_ip as usize;
                        } else {
                            for_loops.pop();
                        }
                    }
                    ForLoopState::Matrix {
                        matrix,
                        col_idx,
                        eff_cols,
                    } => {
                        if *col_idx < *eff_cols {
                            let col = extract_col(matrix, *col_idx);
                            *col_idx += 1;
                            let f = frame.as_deref_mut().ok_or_else(|| {
                                MatroError::runtime("ForNextLocal outside function")
                            })?;
                            f.set(slot, Value::matrix(col));
                            *ip = loop_ip as usize;
                        } else {
                            for_loops.pop();
                        }
                    }
                    ForLoopState::Values { items, idx } => {
                        if *idx < items.len() {
                            let v = items[*idx].clone();
                            *idx += 1;
                            let f = frame.as_deref_mut().ok_or_else(|| {
                                MatroError::runtime("ForNextLocal outside function")
                            })?;
                            f.set(slot, v);
                            *ip = loop_ip as usize;
                        } else {
                            for_loops.pop();
                        }
                    }
                }
            }
            OpCode::ForPop => {
                for_loops.pop();
            }
            OpCode::PushColon => {
                stack.push(StackItem::Colon);
            }
            OpCode::PushEndTarget { .. }
            | OpCode::PushEndFieldTarget { .. }
            | OpCode::PushEndVarTarget { .. }
            | OpCode::PushEndLocalTarget { .. } => {
                self.exec_push_end_target(op, chunk, stack, end_targets, frame.as_deref(), stdout)?;
            }
            OpCode::SetEndDim(dim) => {
                if let Some(Some(top)) = end_targets.last_mut() {
                    if !top.inherited {
                        top.dim = dim;
                    }
                }
            }
            OpCode::PushEndTop => {
                let top = end_targets
                    .last()
                    .and_then(|o| o.as_ref())
                    .ok_or_else(|| {
                        MatroError::runtime("'end' can only be used inside an indexing expression")
                    })?;
                let n = end_for_value(&top.val, top.dim, top.total_dims);
                stack.push(StackItem::Scalar(n as f64));
            }
            OpCode::PopEndTarget => {
                end_targets.pop();
            }
            OpCode::CallOrIndexLocal {
                slot,
                argc,
                nargout,
            } => {
                // Fast-path: 1D or 2D scalar indexing into a real double 2D local Matrix without boxing or Vec alloc
                if nargout <= 1 && (argc == 1 || argc == 2) {
                    if let Some(Value::Matrix(m)) = frame.as_deref().and_then(|f| f.get(slot)) {
                        if !m.is_complex()
                            && m.dtype() == matro_types::NumericClass::Double
                            && m.ndims() == 2
                        {
                            let slen = stack.len();
                            if argc == 1 && slen >= 1 {
                                if let StackItem::Scalar(x) = stack[slen - 1] {
                                    if x >= 1.0 && x.fract() == 0.0 && (x as usize) <= m.len() {
                                        let v = m.data()[x as usize - 1];
                                        stack.pop();
                                        if nargout == 1 {
                                            stack.push(StackItem::Scalar(v));
                                        }
                                        return Ok(true);
                                    }
                                }
                            } else if argc == 2 && slen >= 2 {
                                if let (StackItem::Scalar(r), StackItem::Scalar(c)) =
                                    (&stack[slen - 2], &stack[slen - 1])
                                {
                                    let (r, c) = (*r, *c);
                                    if r >= 1.0 && r.fract() == 0.0 && c >= 1.0 && c.fract() == 0.0
                                    {
                                        let ri = r as usize - 1;
                                        let ci = c as usize - 1;
                                        if ri < m.rows() && ci < m.cols() {
                                            let v = m.get(ri, ci);
                                            stack.truncate(slen - 2);
                                            if nargout == 1 {
                                                stack.push(StackItem::Scalar(v));
                                            }
                                            return Ok(true);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                let raw_args = pop_items(stack, argc as usize)?;
                let val = frame
                    .as_deref()
                    .and_then(|f| f.get(slot))
                    .cloned();
                if let Some(var_val) = val {
                    if let Some(f) = frame.as_deref() {
                        f.sync_to_env(&mut self.envs);
                    }
                    self.apply_value_call_or_index(var_val, raw_args, nargout as usize, stack, stdout)?;
                    if let Some(f) = frame.as_deref_mut() {
                        f.sync_from_env(&self.envs);
                    }
                } else {
                    let name = frame
                        .as_deref()
                        .map(|f| f.names[slot as usize].clone())
                        .unwrap_or_default();
                    let call_args = flatten_call_args(raw_args)?;
                    let results = self.invoke_named_callable(
                        &name,
                        &call_args,
                        nargout as usize,
                        frame.as_deref_mut(),
                        stdout,
                    )?;
                    push_call_results(stack, results, nargout as usize);
                }
            }
            OpCode::CallOrIndex {
                name_idx,
                argc,
                nargout,
            } => {
                // Fast-path: 1D or 2D scalar indexing into a real double 2D Matrix without cloning name/args
                if nargout <= 1 && (argc == 1 || argc == 2) {
                    let name_ref = &chunk.names[name_idx as usize];
                    if let Some(Value::Matrix(m)) =
                        lookup_var(name_ref, frame.as_deref(), &self.globals)
                    {
                        if !m.is_complex()
                            && m.dtype() == matro_types::NumericClass::Double
                            && m.ndims() == 2
                        {
                            let slen = stack.len();
                            if argc == 1 && slen >= 1 {
                                if let StackItem::Scalar(x) = stack[slen - 1] {
                                    if x >= 1.0 && x.fract() == 0.0 && (x as usize) <= m.len() {
                                        let v = m.data()[x as usize - 1];
                                        stack.pop();
                                        if nargout == 1 {
                                            stack.push(StackItem::Scalar(v));
                                        }
                                        return Ok(true);
                                    }
                                }
                            } else if argc == 2 && slen >= 2 {
                                if let (StackItem::Scalar(r), StackItem::Scalar(c)) =
                                    (&stack[slen - 2], &stack[slen - 1])
                                {
                                    let (r, c) = (*r, *c);
                                    if r >= 1.0 && r.fract() == 0.0 && c >= 1.0 && c.fract() == 0.0
                                    {
                                        let ri = r as usize - 1;
                                        let ci = c as usize - 1;
                                        if ri < m.rows() && ci < m.cols() {
                                            let v = m.get(ri, ci);
                                            stack.truncate(slen - 2);
                                            if nargout == 1 {
                                                stack.push(StackItem::Scalar(v));
                                            }
                                            return Ok(true);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                let name = chunk.names[name_idx as usize].clone();
                let raw_args = pop_items(stack, argc as usize)?;
                let env_val = frame
                    .as_deref()
                    .and_then(|f| f.env_id)
                    .and_then(|eid| self.envs.get(eid))
                    .and_then(|env| env.get(&name))
                    .cloned();

                if let Some(var_val) = lookup_var(&name, frame.as_deref(), &self.globals)
                    .cloned()
                    .or(env_val)
                {
                    if let Some(f) = frame.as_deref() {
                        f.sync_to_env(&mut self.envs);
                    }
                    self.apply_value_call_or_index(var_val, raw_args, nargout as usize, stack, stdout)?;
                    if let Some(f) = frame.as_deref_mut() {
                        f.sync_from_env(&self.envs);
                    }
                } else {
                    let call_args = flatten_call_args(raw_args)?;
                    let results = self.invoke_named_callable(
                        &name,
                        &call_args,
                        nargout as usize,
                        frame.as_deref_mut(),
                        stdout,
                    )?;
                    push_call_results(stack, results, nargout as usize);
                }
            }
            OpCode::DynamicCall { argc, nargout } => {
                let raw_args = pop_items(stack, argc as usize)?;
                let target = pop_val(stack)?;
                self.apply_value_call_or_index(target, raw_args, nargout as usize, stack, stdout)?;
            }
            OpCode::MethodCall {
                method_idx,
                argc,
                nargout,
            } => {
                let raw_args = pop_items(stack, argc as usize)?;
                let receiver = pop_val(stack)?;
                let method_name = chunk.names[method_idx as usize].clone();

                // 0. Built-in namespace / class static / native handle / map methods
                if let Value::Object(ref o) = receiver {
                    if o.class_name == "__ns_containers" && method_name == "Map" {
                        let call_args = flatten_call_args(raw_args)?;
                        let map_val = matro_intrinsics::tables::build_map(&call_args)?;
                        push_call_results(stack, vec![map_val], nargout as usize);
                        return Ok(true);
                    }
                    if o.class_name == "MException" {
                        match method_name.as_str() {
                            "throw" | "rethrow" | "throwAsCaller" => {
                                return Err(mexception_to_error(&receiver));
                            }
                            "getReport" => {
                                let id = o
                                    .fields
                                    .get("identifier")
                                    .and_then(|v| v.as_str().ok())
                                    .unwrap_or("Matro:Error");
                                let msg = o
                                    .fields
                                    .get("message")
                                    .and_then(|v| v.as_str().ok())
                                    .unwrap_or("");
                                let rep = if id.is_empty() {
                                    msg.to_string()
                                } else {
                                    format!("{id}: {msg}")
                                };
                                push_call_results(stack, vec![Value::String(rep)], nargout as usize);
                                return Ok(true);
                            }
                            "addCause" => {
                                let call_args = flatten_call_args(raw_args)?;
                                let mut updated = o.clone();
                                let mut causes = match updated.fields.get("cause") {
                                    Some(Value::Cell(c)) => c.data.clone(),
                                    _ => Vec::new(),
                                };
                                if let Some(c) = call_args.first() {
                                    causes.push(c.clone());
                                }
                                let n = causes.len();
                                updated.fields.insert(
                                    "cause".to_string(),
                                    Value::Cell(CellArray::from_vec(
                                        n,
                                        if n == 0 { 0 } else { 1 },
                                        causes,
                                    )),
                                );
                                push_call_results(
                                    stack,
                                    vec![Value::Object(updated)],
                                    nargout as usize,
                                );
                                return Ok(true);
                            }
                            _ => {}
                        }
                    }
                    if o.class_name == "__class_ref" {
                        let cname = o
                            .fields
                            .get("__name")
                            .and_then(|v| v.as_str().ok())
                            .unwrap_or("")
                            .to_string();
                        if let Some(proto) = self.find_class_method(&cname, &method_name) {
                            let call_args = flatten_call_args(raw_args)?;
                            let results = self.call_user_func(
                                &proto,
                                &call_args,
                                nargout as usize,
                                None,
                                stdout,
                            )?;
                            push_call_results(stack, results, nargout as usize);
                            return Ok(true);
                        }
                        return Err(MatroError::runtime(format!(
                            "No static method '{method_name}' on class '{cname}'"
                        )));
                    }
                    if o.class_name == "__pkg_ref" {
                        let pkg_path = o
                            .fields
                            .get("__path")
                            .and_then(|v| v.as_str().ok())
                            .unwrap_or("")
                            .to_string();
                        let call_args = flatten_call_args(raw_args)?;
                        let results = self.call_package_member(
                            &pkg_path,
                            &method_name,
                            &call_args,
                            nargout as usize,
                            stdout,
                        )?;
                        push_call_results(stack, results, nargout as usize);
                        return Ok(true);
                    }
                    if method_name == "predict"
                        && matches!(
                            o.class_name.as_str(),
                            "LinearModel"
                                | "ClassificationKNN"
                                | "ClassificationSVM"
                                | "ClassificationNaiveBayes"
                        )
                    {
                        let mut call_args = vec![receiver];
                        call_args.extend(flatten_call_args(raw_args)?);
                        let results = matro_intrinsics::stats_ml::fn_predict(&call_args)?;
                        push_call_results(stack, results, nargout as usize);
                        return Ok(true);
                    }
                }
                if let Value::Handle(ref h) = receiver {
                    if h.class_name == "inputParser" {
                        let call_args = flatten_call_args(raw_args)?;
                        let res = self.call_input_parser_method(h.id, &method_name, &call_args)?;
                        push_call_results(stack, res, nargout as usize);
                        return Ok(true);
                    }
                    if method_name == "addlistener" {
                        let call_args = flatten_call_args(raw_args)?;
                        self.add_event_listener(h.id, &call_args)?;
                        push_call_results(stack, vec![], nargout as usize);
                        return Ok(true);
                    }
                    if method_name == "notify" {
                        let call_args = flatten_call_args(raw_args)?;
                        self.notify_event(h, &call_args, stdout)?;
                        push_call_results(stack, vec![], nargout as usize);
                        return Ok(true);
                    }
                }
                if let Value::Map(ref m) = receiver {
                    let call_args = flatten_call_args(raw_args)?;
                    let res = match method_name.as_str() {
                        "isKey" => {
                            let mut a = vec![Value::Map(m.clone())];
                            a.extend(call_args);
                            matro_intrinsics::tables::fn_is_key(&a)?
                        }
                        "keys" => matro_intrinsics::tables::fn_keys(&[Value::Map(m.clone())])?,
                        "values" => matro_intrinsics::tables::fn_values(&[Value::Map(m.clone())])?,
                        "remove" => {
                            let mut updated = m.clone();
                            for k in &call_args {
                                updated.entries.remove(&MapData::key_of(k)?);
                            }
                            vec![Value::Map(updated)]
                        }
                        _ => {
                            return Err(MatroError::runtime(format!(
                                "No method '{method_name}' on containers.Map"
                            )));
                        }
                    };
                    push_call_results(stack, res, nargout as usize);
                    return Ok(true);
                }

                // 1. If receiver is a `classdef` Object or Handle and its class (or superclass) defines `method_name`, dispatch it!
                let class_name = match &receiver {
                    Value::Object(obj) => Some(obj.class_name.clone()),
                    Value::Handle(h) => Some(h.class_name.clone()),
                    _ => None,
                };
                if let Some(cname) = class_name {
                    let caller_dir = frame
                        .as_deref()
                        .and_then(|f| f.source_file)
                        .or(self.active_file.as_deref())
                        .map(matro_loader::parent_dir);
                    if let Some(proto) =
                        self.resolve_class_method(&cname, &method_name, caller_dir.as_deref())?
                    {
                        let expanded = flatten_call_args(raw_args)?;
                        let call_args = if self.is_static_method(&cname, &method_name) {
                            expanded
                        } else {
                            let mut v = Vec::with_capacity(expanded.len() + 1);
                            v.push(receiver);
                            v.extend(expanded);
                            v
                        };
                        let results =
                            self.call_user_func(&proto, &call_args, nargout as usize, None, stdout)?;
                        push_call_results(stack, results, nargout as usize);
                        return Ok(true);
                    }
                }

                // 2. Otherwise fall back to field lookup (e.g. `s.fn(x)` or `obj.vec(1)`)
                let cur_fn = frame.as_deref().map(|f| f.func_name);
                let field_val = match &receiver {
                    Value::Struct(map) => map.get(&method_name).cloned(),
                    Value::Object(obj) => {
                        let getter = format!("get.{method_name}");
                        if cur_fn != Some(getter.as_str()) {
                            if let Some(proto) = self.find_class_method(&obj.class_name, &getter) {
                                let rets = self.call_user_func(
                                    &proto,
                                    &[receiver.clone()],
                                    1,
                                    None,
                                    stdout,
                                )?;
                                Some(rets.into_iter().next().unwrap_or_else(Value::empty))
                            } else {
                                obj.fields.get(&method_name).cloned()
                            }
                        } else {
                            obj.fields.get(&method_name).cloned()
                        }
                    }
                    Value::Handle(h) => {
                        let getter = format!("get.{method_name}");
                        if cur_fn != Some(getter.as_str()) {
                            if let Some(proto) = self.find_class_method(&h.class_name, &getter) {
                                let rets = self.call_user_func(
                                    &proto,
                                    &[receiver.clone()],
                                    1,
                                    None,
                                    stdout,
                                )?;
                                Some(rets.into_iter().next().unwrap_or_else(Value::empty))
                            } else {
                                self.handles
                                    .get(h.id)
                                    .and_then(|o| o.fields.get(&method_name).cloned())
                            }
                        } else {
                            self.handles
                                .get(h.id)
                                .and_then(|o| o.fields.get(&method_name).cloned())
                        }
                    }
                    Value::Table(t) => t.get_col(&method_name).cloned(),
                    other => {
                        return Err(MatroError::runtime(format!(
                            "Cannot call method '.{method_name}' on value of type {}",
                            other.type_name()
                        )));
                    }
                }
                .ok_or_else(|| {
                    MatroError::runtime(format!(
                        "No method or field '{method_name}' on {}",
                        receiver.type_name()
                    ))
                })?;

                self.apply_value_call_or_index(
                    field_val,
                    raw_args,
                    nargout as usize,
                    stack,
                    stdout,
                )?;
            }
            OpCode::StoreIndex { name_idx, argc } => {
                // Drop previous last_value reference first so COW `Arc::make_mut` sees strong_count == 1
                *last_value = None;
                // Fast-path: 1D/2D scalar write directly into global real double 2D Matrix in-place
                if frame.is_none() && (argc == 1 || argc == 2) {
                    let slen = stack.len();
                    let name_ref = &chunk.names[name_idx as usize];
                    if argc == 1 && slen >= 2 {
                        if let (StackItem::Scalar(x), StackItem::Scalar(rhs_f)) =
                            (&stack[slen - 2], &stack[slen - 1])
                        {
                            let (x, rhs_f) = (*x, *rhs_f);
                            if x >= 1.0 && x.fract() == 0.0 {
                                let idx = x as usize - 1;
                                if let Some(Value::Matrix(arc)) = self.globals.get_mut(name_ref) {
                                    if !arc.is_complex()
                                        && arc.dtype() == matro_types::NumericClass::Double
                                        && arc.ndims() == 2
                                        && idx < arc.len()
                                    {
                                        std::sync::Arc::make_mut(arc).data_mut()[idx] = rhs_f;
                                        *last_value = Some(Value::Matrix(arc.clone()));
                                        stack.truncate(slen - 2);
                                        return Ok(true);
                                    }
                                }
                            }
                        }
                    } else if argc == 2 && slen >= 3 {
                        if let (
                            StackItem::Scalar(r),
                            StackItem::Scalar(c),
                            StackItem::Scalar(rhs_f),
                        ) = (&stack[slen - 3], &stack[slen - 2], &stack[slen - 1])
                        {
                            let (r, c, rhs_f) = (*r, *c, *rhs_f);
                            if r >= 1.0 && r.fract() == 0.0 && c >= 1.0 && c.fract() == 0.0 {
                                let ri = r as usize - 1;
                                let ci = c as usize - 1;
                                if let Some(Value::Matrix(arc)) = self.globals.get_mut(name_ref) {
                                    if !arc.is_complex()
                                        && arc.dtype() == matro_types::NumericClass::Double
                                        && arc.ndims() == 2
                                        && ri < arc.rows()
                                        && ci < arc.cols()
                                    {
                                        std::sync::Arc::make_mut(arc).set(ri, ci, rhs_f);
                                        *last_value = Some(Value::Matrix(arc.clone()));
                                        stack.truncate(slen - 3);
                                        return Ok(true);
                                    }
                                }
                            }
                        }
                    }
                }
                let rhs = pop_val(stack)?;
                let raw_args = pop_items(stack, argc as usize)?;
                let name = chunk.names[name_idx as usize].clone();
                let mut val = if frame.is_none() {
                    self.globals.remove(&name).unwrap_or_else(Value::empty)
                } else {
                    let v = lookup_var(&name, frame.as_deref(), &self.globals)
                        .cloned()
                        .unwrap_or_else(Value::empty);
                    remove_var(&name, frame.as_deref_mut(), &mut self.globals);
                    v
                };
                self.assign_value_index(&mut val, &raw_args, rhs, stdout)?;
                if frame.is_none() {
                    *last_value = Some(val.clone());
                }
                store_var(name, val, frame.as_deref_mut(), &mut self.globals);
            }
            OpCode::StoreIndexLocal { slot, argc } => {
                // Fast-path: 1D/2D scalar write into local real double 2D Matrix in-place (zero alloc, true COW)
                if argc == 1 || argc == 2 {
                    let slen = stack.len();
                    if argc == 1 && slen >= 2 {
                        if let (StackItem::Scalar(x), StackItem::Scalar(rhs_f)) =
                            (&stack[slen - 2], &stack[slen - 1])
                        {
                            let (x, rhs_f) = (*x, *rhs_f);
                            if x >= 1.0 && x.fract() == 0.0 {
                                let idx = x as usize - 1;
                                if let Some(Value::Matrix(arc)) =
                                    frame.as_deref_mut().and_then(|f| f.get_mut(slot))
                                {
                                    if !arc.is_complex()
                                        && arc.dtype() == matro_types::NumericClass::Double
                                        && arc.ndims() == 2
                                        && idx < arc.len()
                                    {
                                        std::sync::Arc::make_mut(arc).data_mut()[idx] = rhs_f;
                                        stack.truncate(slen - 2);
                                        return Ok(true);
                                    }
                                }
                            }
                        }
                    } else if argc == 2 && slen >= 3 {
                        if let (
                            StackItem::Scalar(r),
                            StackItem::Scalar(c),
                            StackItem::Scalar(rhs_f),
                        ) = (&stack[slen - 3], &stack[slen - 2], &stack[slen - 1])
                        {
                            let (r, c, rhs_f) = (*r, *c, *rhs_f);
                            if r >= 1.0 && r.fract() == 0.0 && c >= 1.0 && c.fract() == 0.0 {
                                let ri = r as usize - 1;
                                let ci = c as usize - 1;
                                if let Some(Value::Matrix(arc)) =
                                    frame.as_deref_mut().and_then(|f| f.get_mut(slot))
                                {
                                    if !arc.is_complex()
                                        && arc.dtype() == matro_types::NumericClass::Double
                                        && arc.ndims() == 2
                                        && ri < arc.rows()
                                        && ci < arc.cols()
                                    {
                                        std::sync::Arc::make_mut(arc).set(ri, ci, rhs_f);
                                        stack.truncate(slen - 3);
                                        return Ok(true);
                                    }
                                }
                            }
                        }
                    }
                }
                let rhs = pop_val(stack)?;
                let raw_args = pop_items(stack, argc as usize)?;
                let mut val = frame
                    .as_deref_mut()
                    .ok_or_else(|| MatroError::runtime("StoreIndexLocal outside function"))?
                    .take(slot)
                    .unwrap_or_else(Value::empty);
                self.assign_value_index(&mut val, &raw_args, rhs, stdout)?;
                if let Some(f) = frame.as_deref_mut() {
                    f.set(slot, val);
                }
            }
            OpCode::CellIndex { argc, nargout } => {
                let raw_args = pop_items(stack, argc as usize)?;
                let target = pop_val(stack)?;
                match target {
                    Value::Cell(cell) => {
                        let vals = index_cell_contents(&cell, &raw_args)?;
                        if nargout > 1 {
                            push_call_results(stack, vals, nargout as usize);
                        } else if vals.len() == 1 {
                            stack.push(StackItem::Val(vals.into_iter().next().unwrap()));
                        } else {
                            stack.push(StackItem::Spread(vals));
                        }
                    }
                    Value::Table(t) => {
                        let extracted = extract_table_braces(&t, &raw_args)?;
                        stack.push(StackItem::Val(extracted));
                    }
                    Value::Object(ref o) if o.class_name == "string" => {
                        let vals = oop::index_string_braces(o, &raw_args)?;
                        if nargout > 1 {
                            push_call_results(stack, vals, nargout as usize);
                        } else if vals.len() == 1 {
                            stack.push(StackItem::Val(vals.into_iter().next().unwrap()));
                        } else {
                            stack.push(StackItem::Spread(vals));
                        }
                    }
                    other => {
                        return Err(MatroError::runtime(format!(
                            "Brace indexing '{{}}' requires a cell array or table, got {}",
                            other.type_name()
                        )));
                    }
                }
            }
            OpCode::StoreCellIndex { name_idx, argc } => {
                let rhs = pop_val(stack)?;
                let raw_args = pop_items(stack, argc as usize)?;
                let name = chunk.names[name_idx as usize].clone();
                let mut cell = match lookup_var(&name, frame.as_deref(), &self.globals) {
                    Some(Value::Cell(c)) => c.clone(),
                    _ => CellArray::empty(),
                };
                assign_cell_index(&mut cell, &raw_args, rhs)?;
                *last_value = Some(Value::Cell(cell.clone()));
                store_var(
                    name,
                    Value::Cell(cell),
                    frame.as_deref_mut(),
                    &mut self.globals,
                );
            }
            OpCode::StoreCellIndexLocal { slot, argc } => {
                let rhs = pop_val(stack)?;
                let raw_args = pop_items(stack, argc as usize)?;
                let f = frame
                    .as_deref_mut()
                    .ok_or_else(|| MatroError::runtime("StoreCellIndexLocal outside function"))?;
                let mut cell = match f.take(slot) {
                    Some(Value::Cell(c)) => c,
                    _ => CellArray::empty(),
                };
                assign_cell_index(&mut cell, &raw_args, rhs)?;
                *last_value = Some(Value::Cell(cell.clone()));
                f.set(slot, Value::Cell(cell));
            }
            OpCode::GetField(field_idx) => {
                let field = chunk.names[field_idx as usize].clone();
                let obj = pop_val(stack)?;
                let cur_fn = frame.as_deref().map(|f| f.func_name);
                let val = self.exec_get_field_by_name(obj, field, cur_fn, stdout)?;
                stack.push(StackItem::Val(val));
            }
            OpCode::GetDynamicField => {
                let field_val = pop_val(stack)?;
                let field = match &field_val {
                    Value::String(s) => s.clone(),
                    Value::Object(o) if o.class_name == "string" => {
                        if let Some(Value::Cell(c)) = o.fields.get("Data") {
                            c.data
                                .first()
                                .and_then(|v| v.as_str().ok())
                                .unwrap_or("")
                                .to_string()
                        } else {
                            return Err(MatroError::runtime("Dynamic field name must be a string"));
                        }
                    }
                    other => {
                        return Err(MatroError::runtime(format!(
                            "Dynamic field name must be a character vector or string, got {}",
                            other.type_name()
                        )));
                    }
                };
                let obj = pop_val(stack)?;
                let cur_fn = frame.as_deref().map(|f| f.func_name);
                let val = self.exec_get_field_by_name(obj, field, cur_fn, stdout)?;
                stack.push(StackItem::Val(val));
            }
            OpCode::StoreField { name_idx, path_idx } => {
                let rhs = pop_val(stack)?;
                let name = chunk.names[name_idx as usize].clone();
                let path = &chunk.field_paths[path_idx as usize];
                let cur_fn = frame.as_deref().map(|f| f.func_name.to_string());
                let mut root = lookup_var(&name, frame.as_deref(), &self.globals)
                    .cloned()
                    .unwrap_or_else(|| Value::Struct(BTreeMap::new()));
                if path.len() == 1 {
                    let setter = format!("set.{}", path[0]);
                    if cur_fn.as_deref() != Some(setter.as_str()) {
                        let cname = match &root {
                            Value::Object(o) => Some(o.class_name.clone()),
                            Value::Handle(h) => Some(h.class_name.clone()),
                            _ => None,
                        };
                        if let Some(cls) = cname {
                            if let Some(proto) = self.find_class_method(&cls, &setter) {
                                let is_handle = matches!(root, Value::Handle(_));
                                let rets = self.call_user_func(
                                    &proto,
                                    &[root.clone(), rhs],
                                    if is_handle { 0 } else { 1 },
                                    None,
                                    stdout,
                                )?;
                                if !is_handle {
                                    let updated = rets.into_iter().next().unwrap_or(root);
                                    *last_value = Some(updated.clone());
                                    store_var(name, updated, frame.as_deref_mut(), &mut self.globals);
                                } else {
                                    *last_value = Some(root);
                                }
                                return Ok(true);
                            }
                        }
                    }
                }
                if let Value::Handle(ref h) = root {
                    let inst = self
                        .handles
                        .get_mut(h.id)
                        .ok_or_else(|| MatroError::runtime("Invalid object handle"))?;
                    set_nested_field_map(&mut inst.fields, path, rhs)?;
                    *last_value = Some(root);
                } else {
                    set_nested_field(&mut root, path, rhs)?;
                    *last_value = Some(root.clone());
                    store_var(name, root, frame.as_deref_mut(), &mut self.globals);
                }
            }
            OpCode::StoreFieldLocal { slot, path_idx } => {
                let rhs = pop_val(stack)?;
                let path = &chunk.field_paths[path_idx as usize];
                let cur_fn = frame.as_deref().map(|f| f.func_name.to_string());
                let mut root = frame
                    .as_deref_mut()
                    .ok_or_else(|| MatroError::runtime("StoreFieldLocal outside function"))?
                    .take(slot)
                    .unwrap_or_else(|| Value::Struct(BTreeMap::new()));
                if path.len() == 1 {
                    let setter = format!("set.{}", path[0]);
                    if cur_fn.as_deref() != Some(setter.as_str()) {
                        let cname = match &root {
                            Value::Object(o) => Some(o.class_name.clone()),
                            Value::Handle(h) => Some(h.class_name.clone()),
                            _ => None,
                        };
                        if let Some(cls) = cname {
                            if let Some(proto) = self.find_class_method(&cls, &setter) {
                                let is_handle = matches!(root, Value::Handle(_));
                                let rets = self.call_user_func(
                                    &proto,
                                    &[root.clone(), rhs],
                                    if is_handle { 0 } else { 1 },
                                    None,
                                    stdout,
                                )?;
                                let updated = if is_handle {
                                    root
                                } else {
                                    rets.into_iter().next().unwrap_or(root)
                                };
                                *last_value = Some(updated.clone());
                                if let Some(f) = frame.as_deref_mut() {
                                    f.set(slot, updated);
                                }
                                return Ok(true);
                            }
                        }
                    }
                }
                if let Value::Handle(ref h) = root {
                    let inst = self
                        .handles
                        .get_mut(h.id)
                        .ok_or_else(|| MatroError::runtime("Invalid object handle"))?;
                    set_nested_field_map(&mut inst.fields, path, rhs)?;
                    *last_value = Some(root.clone());
                    if let Some(f) = frame.as_deref_mut() {
                        f.set(slot, root);
                    }
                } else {
                    set_nested_field(&mut root, path, rhs)?;
                    *last_value = Some(root.clone());
                    if let Some(f) = frame.as_deref_mut() {
                        f.set(slot, root);
                    }
                }
            }
            OpCode::StoreIndexedField {
                name_idx,
                argc,
                path_idx,
            } => {
                let rhs = pop_val(stack)?;
                let raw_args = pop_items(stack, argc as usize)?;
                let name = chunk.names[name_idx as usize].clone();
                let path = &chunk.field_paths[path_idx as usize];
                let cur = lookup_var(&name, frame.as_deref(), &self.globals).cloned();
                let updated = assign_struct_array_field(cur, &raw_args, path, rhs)?;
                *last_value = Some(updated.clone());
                store_var(name, updated, frame.as_deref_mut(), &mut self.globals);
            }
            OpCode::StoreIndexedFieldLocal {
                slot,
                argc,
                path_idx,
            } => {
                let rhs = pop_val(stack)?;
                let raw_args = pop_items(stack, argc as usize)?;
                let path = &chunk.field_paths[path_idx as usize];
                let f = frame
                    .as_deref_mut()
                    .ok_or_else(|| MatroError::runtime("StoreIndexedFieldLocal outside function"))?;
                let cur = f.take(slot);
                let updated = assign_struct_array_field(cur, &raw_args, path, rhs)?;
                *last_value = Some(updated.clone());
                f.set(slot, updated);
            }
            OpCode::StoreChain { name_idx, spec_idx } => {
                let name = chunk.names[name_idx as usize].clone();
                self.exec_store_chain(
                    Some(&name),
                    None,
                    spec_idx,
                    chunk,
                    stack,
                    frame,
                    last_value,
                )?;
            }
            OpCode::StoreChainLocal { slot, spec_idx } => {
                self.exec_store_chain(
                    None,
                    Some(slot),
                    spec_idx,
                    chunk,
                    stack,
                    frame,
                    last_value,
                )?;
            }
            OpCode::DefineFunc(idx) => {
                let mut proto = chunk.functions[idx as usize].clone();
                if proto.source_file.is_none() {
                    proto.source_file = self.active_file.clone();
                }
                let scoped_to_file = proto
                    .source_file
                    .as_deref()
                    .is_some_and(|sf| self.loaded_files.contains_key(sf));
                if !scoped_to_file {
                    self.functions.insert(proto.name.clone(), proto);
                }
            }
            OpCode::DefineNestedFunc { func_idx, slot } => {
                let mut proto = chunk.functions[func_idx as usize].clone();
                if proto.source_file.is_none() {
                    proto.source_file = frame
                        .as_deref()
                        .and_then(|f| f.source_file)
                        .map(String::from)
                        .or_else(|| self.active_file.clone());
                }
                let params = proto.params.clone();
                let id = self.closures.len();
                self.closures.push(proto);
                let env_id = frame.as_deref().and_then(|f| f.env_id);
                let cl = Value::Closure(ClosureHandle {
                    id,
                    params,
                    captures: BTreeMap::new(),
                    env_id,
                });
                if let Some(f) = frame.as_deref_mut() {
                    f.set(slot, cl);
                    f.sync_to_env(&mut self.envs);
                }
            }
            OpCode::DefineClass(idx) => {
                let mut cls = chunk.classes[idx as usize].clone();
                if cls.source_file.is_none() {
                    cls.source_file = self.active_file.clone();
                }
                let caller_dir = cls.source_file.as_deref().map(matro_loader::parent_dir);
                let supers = cls.superclasses.clone();
                self.classes.insert(cls.name.clone(), cls);
                self.ensure_superclasses_loaded(&supers, caller_dir.as_deref())?;
            }
            OpCode::MakeClosure(idx) => {
                let mut proto = chunk.functions[idx as usize].clone();
                if proto.source_file.is_none() {
                    proto.source_file = frame
                        .as_deref()
                        .and_then(|f| f.source_file)
                        .map(String::from)
                        .or_else(|| self.active_file.clone());
                }
                let params = proto.params.clone();
                let id = self.closures.len();
                self.closures.push(proto);

                let env_id = frame.as_deref().and_then(|f| f.env_id);
                let captures = if let Some(f) = frame.as_deref() {
                    f.sync_to_env(&mut self.envs);
                    f.snapshot_captures()
                } else {
                    self.globals
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect()
                };
                stack.push(StackItem::Val(Value::Closure(ClosureHandle {
                    id,
                    params,
                    captures,
                    env_id,
                })));
            }
            OpCode::Return => {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn apply_value_call_or_index(
        &mut self,
        var_val: Value,
        raw_args: Vec<StackItem>,
        nargout: usize,
        stack: &mut Vec<StackItem>,
        stdout: &mut String,
    ) -> Result<()> {
        match var_val {
            Value::Closure(cl) => {
                let call_args = flatten_call_args(raw_args)?;
                let results = self.call_closure(&cl, &call_args, nargout, stdout)?;
                push_call_results(stack, results, nargout);
            }
            Value::Matrix(m) => {
                let indexed = index_matrix(&m, &raw_args)?;
                stack.push(StackItem::Val(Value::matrix(indexed)));
            }
            Value::String(s) => {
                let chars: Vec<char> = s.chars().collect();
                let out: String = match raw_args.len() {
                    1 => {
                        let idxs = resolve_indices(&raw_args[0], chars.len())?;
                        idxs.into_iter().map(|i| chars[i]).collect()
                    }
                    2 => {
                        let r_idxs = resolve_indices(&raw_args[0], 1)?;
                        let c_idxs = resolve_indices(&raw_args[1], chars.len())?;
                        if r_idxs.is_empty() || c_idxs.is_empty() {
                            String::new()
                        } else {
                            c_idxs.into_iter().map(|j| chars[j]).collect()
                        }
                    }
                    _ => {
                        return Err(MatroError::index(
                            "Character vector indexing expects 1 or 2 subscripts",
                        ));
                    }
                };
                stack.push(StackItem::Val(Value::String(out)));
            }
            Value::Sparse(s) => {
                let indexed = index_sparse(&s, &raw_args)?;
                stack.push(StackItem::Val(indexed));
            }
            Value::Cell(c) => {
                let sub = index_cell_subarray(&c, &raw_args)?;
                stack.push(StackItem::Val(Value::Cell(sub)));
            }
            Value::StructArray(sa) => {
                let val = index_struct_array(&sa, &raw_args)?;
                stack.push(StackItem::Val(val));
            }
            Value::Struct(map) => {
                let idxs = resolve_indices(&raw_args[0], 1)?;
                if idxs.len() == 1 && idxs[0] == 0 {
                    stack.push(StackItem::Val(Value::Struct(map)));
                } else {
                    return Err(MatroError::index("Struct index out of bounds"));
                }
            }
            Value::Table(t) => {
                let sub = index_table(&t, &raw_args)?;
                stack.push(StackItem::Val(Value::Table(sub)));
            }
            Value::Map(m) => {
                if raw_args.len() != 1 {
                    return Err(MatroError::index("containers.Map lookup expects 1 key"));
                }
                let key = MapData::key_of(&raw_args[0].clone().into_val()?)?;
                let val = m.entries.get(&key).cloned().ok_or_else(|| {
                    MatroError::index(format!("containers.Map: key '{key}' not found"))
                })?;
                stack.push(StackItem::Val(val));
            }
            Value::Object(o) => {
                if let Some(res) = oop::index_modern_object(&o, &raw_args)? {
                    stack.push(StackItem::Val(res));
                } else if let Some(proto) = self.find_class_method(&o.class_name, "subsref") {
                    let subs = flatten_call_args(raw_args)?;
                    let mut s_map = BTreeMap::new();
                    s_map.insert("type".to_string(), Value::String("()".to_string()));
                    let cell = if subs.is_empty() {
                        CellArray::empty()
                    } else {
                        CellArray::from_rows(vec![subs])?
                    };
                    s_map.insert("subs".to_string(), Value::Cell(cell));
                    let results = self.call_user_func(
                        &proto,
                        &[Value::Object(o), Value::Struct(s_map)],
                        nargout,
                        None,
                        stdout,
                    )?;
                    push_call_results(stack, results, nargout);
                } else {
                    return Err(MatroError::runtime(format!(
                        "Cannot call or index object of class '{}'",
                        o.class_name
                    )));
                }
            }
            Value::Handle(h) => {
                if let Some(proto) = self.find_class_method(&h.class_name, "subsref") {
                    let subs = flatten_call_args(raw_args)?;
                    let mut s_map = BTreeMap::new();
                    s_map.insert("type".to_string(), Value::String("()".to_string()));
                    let cell = if subs.is_empty() {
                        CellArray::empty()
                    } else {
                        CellArray::from_rows(vec![subs])?
                    };
                    s_map.insert("subs".to_string(), Value::Cell(cell));
                    let results = self.call_user_func(
                        &proto,
                        &[Value::Handle(h), Value::Struct(s_map)],
                        nargout,
                        None,
                        stdout,
                    )?;
                    push_call_results(stack, results, nargout);
                } else {
                    return Err(MatroError::runtime(format!(
                        "Cannot call or index handle of class '{}'",
                        h.class_name
                    )));
                }
            }
        }
        Ok(())
    }

    #[inline(never)]
    fn exec_push_end_target(
        &mut self,
        op: OpCode,
        chunk: &Chunk,
        stack: &[StackItem],
        end_targets: &mut Vec<Option<EndFrame>>,
        frame: Option<&LocalFrame<'_>>,
        stdout: &mut String,
    ) -> Result<()> {
        let inherit_outer = |end_targets: &mut Vec<Option<EndFrame>>| {
            if let Some(Some(outer)) = end_targets.last().cloned() {
                end_targets.push(Some(EndFrame {
                    val: outer.val,
                    dim: outer.dim,
                    total_dims: outer.total_dims,
                    inherited: true,
                }));
            } else {
                end_targets.push(None);
            }
        };

        match op {
            OpCode::PushEndTarget { total_dims } => {
                let target_val = match stack.last() {
                    Some(StackItem::Val(v)) => v.clone(),
                    Some(StackItem::Scalar(x)) => Value::scalar(*x),
                    _ => {
                        return Err(MatroError::runtime(
                            "VM stack underflow resolving 'end' target",
                        ));
                    }
                };
                if matches!(target_val, Value::Closure(_)) {
                    inherit_outer(end_targets);
                } else {
                    end_targets.push(Some(EndFrame {
                        val: target_val,
                        dim: 0,
                        total_dims,
                        inherited: false,
                    }));
                }
            }
            OpCode::PushEndFieldTarget {
                field_idx,
                total_dims,
            } => {
                let field_name = chunk.names[field_idx as usize].clone();
                let receiver = match stack.last() {
                    Some(StackItem::Val(v)) => v.clone(),
                    Some(StackItem::Scalar(x)) => Value::scalar(*x),
                    _ => {
                        return Err(MatroError::runtime(
                            "VM stack underflow resolving 'end' target",
                        ));
                    }
                };
                // Check if this dot-call is a method/namespace/package call rather than field indexing.
                let is_method_call = match &receiver {
                    Value::Object(o) => {
                        matches!(
                            o.class_name.as_str(),
                            "__ns_containers" | "__class_ref" | "__pkg_ref"
                        ) || (field_name == "predict"
                            && matches!(
                                o.class_name.as_str(),
                                "LinearModel"
                                    | "ClassificationKNN"
                                    | "ClassificationSVM"
                                    | "ClassificationNaiveBayes"
                            ))
                            || self.find_class_method(&o.class_name, &field_name).is_some()
                    }
                    Value::Handle(h) => {
                        h.class_name == "inputParser"
                            || matches!(field_name.as_str(), "addlistener" | "notify")
                            || self.find_class_method(&h.class_name, &field_name).is_some()
                    }
                    Value::Map(_) => true,
                    _ => false,
                };
                if is_method_call {
                    inherit_outer(end_targets);
                    return Ok(());
                }
                let cur_fn = frame.map(|f| f.func_name);
                let field_val = match &receiver {
                    Value::Struct(map) => map.get(&field_name).cloned(),
                    Value::Object(obj) => {
                        let getter = format!("get.{field_name}");
                        if cur_fn != Some(getter.as_str()) {
                            if let Some(proto) = self.find_class_method(&obj.class_name, &getter) {
                                let rets = self.call_user_func(
                                    &proto,
                                    &[receiver.clone()],
                                    1,
                                    None,
                                    stdout,
                                )?;
                                Some(rets.into_iter().next().unwrap_or_else(Value::empty))
                            } else {
                                obj.fields.get(&field_name).cloned()
                            }
                        } else {
                            obj.fields.get(&field_name).cloned()
                        }
                    }
                    Value::Handle(h) => {
                        let getter = format!("get.{field_name}");
                        if cur_fn != Some(getter.as_str()) {
                            if let Some(proto) = self.find_class_method(&h.class_name, &getter) {
                                let rets = self.call_user_func(
                                    &proto,
                                    &[receiver.clone()],
                                    1,
                                    None,
                                    stdout,
                                )?;
                                Some(rets.into_iter().next().unwrap_or_else(Value::empty))
                            } else {
                                self.handles
                                    .get(h.id)
                                    .and_then(|o| o.fields.get(&field_name).cloned())
                            }
                        } else {
                            self.handles
                                .get(h.id)
                                .and_then(|o| o.fields.get(&field_name).cloned())
                        }
                    }
                    Value::Table(t) => t.get_col(&field_name).cloned(),
                    _ => None,
                };
                if let Some(val) = field_val {
                    if matches!(val, Value::Closure(_)) {
                        inherit_outer(end_targets);
                    } else {
                        end_targets.push(Some(EndFrame {
                            val,
                            dim: 0,
                            total_dims,
                            inherited: false,
                        }));
                    }
                } else {
                    inherit_outer(end_targets);
                }
            }
            OpCode::PushEndVarTarget {
                var_idx,
                total_dims,
            } => {
                let name = &chunk.names[var_idx as usize];
                let env_val = frame
                    .and_then(|f| f.env_id)
                    .and_then(|eid| self.envs.get(eid))
                    .and_then(|env| env.get(name))
                    .cloned();
                let persistent_val = {
                    let fn_name = frame.map_or("<main>", |f| f.func_name);
                    self.persistents
                        .get(&(fn_name.to_string(), name.clone()))
                        .cloned()
                };
                let found = lookup_var(name, frame, &self.globals)
                    .cloned()
                    .or(env_val)
                    .or(persistent_val);
                if let Some(val) = found {
                    if matches!(val, Value::Closure(_)) {
                        inherit_outer(end_targets);
                    } else {
                        end_targets.push(Some(EndFrame {
                            val,
                            dim: 0,
                            total_dims,
                            inherited: false,
                        }));
                    }
                } else {
                    inherit_outer(end_targets);
                }
            }
            OpCode::PushEndLocalTarget { slot, total_dims } => {
                let found = frame.and_then(|f| f.get(slot)).cloned();
                if let Some(val) = found {
                    if matches!(val, Value::Closure(_)) {
                        inherit_outer(end_targets);
                    } else {
                        end_targets.push(Some(EndFrame {
                            val,
                            dim: 0,
                            total_dims,
                            inherited: false,
                        }));
                    }
                } else {
                    inherit_outer(end_targets);
                }
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    #[inline(never)]
    fn exec_unop_slow(
        &mut self,
        op: OpCode,
        stack: &mut Vec<StackItem>,
        stdout: &mut String,
    ) -> Result<()> {
        let v = pop_val(stack)?;
        let out = match op {
            OpCode::Neg => {
                if let Some(ov) = oop::try_object_unop(self, "uminus", &v, stdout)? {
                    ov
                } else {
                    match v {
                        Value::Sparse(s) => Value::sparse(s.scale(-1.0)),
                        other => Value::matrix(other.as_matrix()?.map_c(|c| -c)),
                    }
                }
            }
            OpCode::Not => Value::matrix(
                v.to_dense_matrix()?
                    .map_c_to_real(|c| if c.is_zero() { 1.0 } else { 0.0 })
                    .with_logical(true),
            ),
            OpCode::CTranspose => {
                if let Some(ov) = oop::try_object_unop(self, "ctranspose", &v, stdout)? {
                    ov
                } else {
                    match v {
                        Value::Sparse(s) => Value::sparse(s.ctranspose()),
                        other => Value::matrix(other.as_matrix()?.ctranspose()),
                    }
                }
            }
            OpCode::Transpose => {
                if let Some(ov) = oop::try_object_unop(self, "transpose", &v, stdout)? {
                    ov
                } else {
                    match v {
                        Value::Sparse(s) => Value::sparse(s.transpose()),
                        other => Value::matrix(other.as_matrix()?.transpose()),
                    }
                }
            }
            _ => unreachable!(),
        };
        stack.push(StackItem::Val(out));
        Ok(())
    }

    #[inline(never)]
    fn exec_binop_slow(
        &mut self,
        op: OpCode,
        a_item: StackItem,
        b_item: StackItem,
        stack: &mut Vec<StackItem>,
        stdout: &mut String,
    ) -> Result<()> {
        let a = a_item.into_val()?;
        let b = b_item.into_val()?;
        let op_name = match op {
            OpCode::Add => "plus",
            OpCode::Sub => "minus",
            OpCode::ElemMul => "times",
            OpCode::ElemDiv => "rdivide",
            OpCode::ElemLeftDiv => "ldivide",
            OpCode::ElemPow => "power",
            OpCode::Mul => "mtimes",
            OpCode::Div => "mrdivide",
            OpCode::LeftDiv => "mldivide",
            OpCode::Pow => "mpower",
            OpCode::Eq => "eq",
            OpCode::Ne => "ne",
            _ => unreachable!(),
        };
        if let Some(ov) = oop::try_object_binop(self, op_name, &a, &b, stdout)? {
            stack.push(StackItem::Val(ov));
            return Ok(());
        }
        let out = match op {
            OpCode::Add => match (&a, &b) {
                (Value::Sparse(sa), Value::Sparse(sb)) => Value::sparse(sa.add_scaled(sb, 1.0)?),
                _ => Value::matrix(
                    a.to_dense_matrix()?
                        .elementwise_c(&b.to_dense_matrix()?, |x, y| x + y)?,
                ),
            },
            OpCode::Sub => match (&a, &b) {
                (Value::Sparse(sa), Value::Sparse(sb)) => Value::sparse(sa.add_scaled(sb, -1.0)?),
                _ => Value::matrix(
                    a.to_dense_matrix()?
                        .elementwise_c(&b.to_dense_matrix()?, |x, y| x - y)?,
                ),
            },
            OpCode::ElemMul => match (&a, &b) {
                (Value::Sparse(sa), Value::Sparse(sb)) => Value::sparse(sa.elem_mul(sb)?),
                (Value::Sparse(sa), Value::Matrix(mb)) if mb.as_complex_scalar().is_some() => {
                    Value::sparse(sa.scale_c(mb.as_complex_scalar().unwrap()))
                }
                (Value::Matrix(ma), Value::Sparse(sb)) if ma.as_complex_scalar().is_some() => {
                    Value::sparse(sb.scale_c(ma.as_complex_scalar().unwrap()))
                }
                _ => Value::matrix(
                    a.to_dense_matrix()?
                        .elementwise_c(&b.to_dense_matrix()?, |x, y| x * y)?,
                ),
            },
            OpCode::ElemDiv => match (&a, &b) {
                (Value::Sparse(sa), Value::Matrix(mb)) if mb.as_complex_scalar().is_some() => {
                    let sc = mb.as_complex_scalar().unwrap();
                    if !sc.is_zero() && sc.re.is_finite() && sc.im.is_finite() {
                        Value::sparse(sa.scale_c(Complex::ONE / sc))
                    } else {
                        Value::matrix(
                            sa.to_dense()
                                .elementwise_c(mb, |x, y| x / y)?,
                        )
                    }
                }
                _ => Value::matrix(
                    a.to_dense_matrix()?
                        .elementwise_c(&b.to_dense_matrix()?, |x, y| x / y)?,
                ),
            },
            OpCode::ElemLeftDiv => match (&a, &b) {
                (Value::Matrix(ma), Value::Sparse(sb)) if ma.as_complex_scalar().is_some() => {
                    let sc = ma.as_complex_scalar().unwrap();
                    if !sc.is_zero() && sc.re.is_finite() && sc.im.is_finite() {
                        Value::sparse(sb.scale_c(Complex::ONE / sc))
                    } else {
                        Value::matrix(
                            ma.elementwise_c(&sb.to_dense(), |x, y| y / x)?,
                        )
                    }
                }
                _ => Value::matrix(
                    a.to_dense_matrix()?
                        .elementwise_c(&b.to_dense_matrix()?, |x, y| y / x)?,
                ),
            },
            OpCode::ElemPow => Value::matrix(
                a.to_dense_matrix()?
                    .elementwise_c(&b.to_dense_matrix()?, Complex::powc)?,
            ),
            OpCode::Mul => match (&a, &b) {
                (Value::Sparse(sa), Value::Sparse(sb)) => Value::sparse(sa.matmul_sparse(sb)?),
                (Value::Sparse(sa), Value::Matrix(mb)) => {
                    if let Some(sc) = mb.as_complex_scalar() {
                        Value::sparse(sa.scale_c(sc))
                    } else {
                        Value::matrix(sa.matmul_dense(mb)?)
                    }
                }
                (Value::Matrix(ma), Value::Sparse(sb)) => {
                    if let Some(sc) = ma.as_complex_scalar() {
                        Value::sparse(sb.scale_c(sc))
                    } else {
                        Value::matrix(ma.matmul(&sb.to_dense())?)
                    }
                }
                _ => Value::matrix(a.as_matrix()?.matmul(b.as_matrix()?)?),
            },
            OpCode::Div => match (&a, &b) {
                (Value::Sparse(sa), Value::Matrix(mb)) if mb.as_complex_scalar().is_some() => {
                    let sc = mb.as_complex_scalar().unwrap();
                    Value::sparse(sa.scale_c(Complex::ONE / sc))
                }
                _ => Value::matrix(a.to_dense_matrix()?.rdivide(&b.to_dense_matrix()?)?),
            },
            OpCode::LeftDiv => match (&a, &b) {
                (Value::Sparse(sa), _) => Value::matrix(sa.solve_dense(&b.to_dense_matrix()?)?),
                _ => Value::matrix(a.to_dense_matrix()?.solve(&b.to_dense_matrix()?)?),
            },
            OpCode::Pow => Value::matrix(a.to_dense_matrix()?.matpow(&b.to_dense_matrix()?)?),
            OpCode::Eq => match (&a, &b) {
                (Value::String(sa), Value::String(sb)) => Value::bool(sa == sb),
                _ => Value::matrix(
                    a.to_dense_matrix()?
                        .elementwise_c(&b.to_dense_matrix()?, |x, y| {
                            Complex::from_real(if x == y { 1.0 } else { 0.0 })
                        })?
                        .with_logical(true),
                ),
            },
            OpCode::Ne => match (&a, &b) {
                (Value::String(sa), Value::String(sb)) => Value::bool(sa != sb),
                _ => Value::matrix(
                    a.to_dense_matrix()?
                        .elementwise_c(&b.to_dense_matrix()?, |x, y| {
                            Complex::from_real(if x != y { 1.0 } else { 0.0 })
                        })?
                        .with_logical(true),
                ),
            },
            _ => unreachable!(),
        };
        stack.push(StackItem::Val(out));
        Ok(())
    }

    #[inline(never)]
    fn exec_cmp_op(
        &mut self,
        stack: &mut Vec<StackItem>,
        op: &str,
        f: impl Fn(f64, f64) -> bool,
        stdout: &mut String,
    ) -> Result<()> {
        let b = pop_val(stack)?;
        let a = pop_val(stack)?;
        if let Some(ov) = oop::try_object_binop(self, op, &a, &b, stdout)? {
            stack.push(StackItem::Val(ov));
            return Ok(());
        }
        let res = a
            .to_dense_matrix()?
            .elementwise(&b.to_dense_matrix()?, |x, y| if f(x, y) { 1.0 } else { 0.0 })?
            .with_logical(true);
        stack.push(StackItem::Val(Value::matrix(res)));
        Ok(())
    }

    fn all_ancestors(&self, class_name: &str) -> Vec<String> {
        let mut visited = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(class_name.to_string());
        while let Some(name) = queue.pop_front() {
            if visited.contains(&name) || visited.len() > 64 {
                continue;
            }
            visited.push(name.clone());
            if let Some(cls) = self.classes.get(&name) {
                for sup in &cls.superclasses {
                    queue.push_back(sup.clone());
                }
            }
        }
        visited
    }

    fn is_handle_class(&self, class_name: &str) -> bool {
        self.all_ancestors(class_name)
            .iter()
            .any(|n| n == "handle")
    }

    fn class_has_static_or_enum_or_const(&self, class_name: &str) -> bool {
        for anc in self.all_ancestors(class_name) {
            if let Some(cls) = self.classes.get(&anc) {
                if !cls.static_methods.is_empty()
                    || !cls.enumerations.is_empty()
                    || cls.properties.iter().any(|p| p.constant)
                {
                    return true;
                }
            }
        }
        false
    }

    fn is_static_method(&self, class_name: &str, method_name: &str) -> bool {
        for anc in self.all_ancestors(class_name) {
            if let Some(cls) = self.classes.get(&anc) {
                if cls.static_methods.iter().any(|m| m == method_name) {
                    return true;
                }
                if cls.methods.iter().any(|m| m.name == method_name) {
                    return false;
                }
            }
        }
        false
    }

    fn resolve_class_ref_field(
        &mut self,
        class_name: &str,
        field: &str,
        stdout: &mut String,
    ) -> Result<Option<Value>> {
        // 1. Enumeration member
        if let Some(cls) = self.classes.get(class_name).cloned() {
            if let Some((_, arg_chunks)) = cls.enumerations.iter().find(|(n, _)| n == field) {
                let mut eval_args = Vec::with_capacity(arg_chunks.len());
                for ch in arg_chunks {
                    eval_args.push(oop::eval_standalone_chunk(self, ch, "__enum_arg", stdout)?);
                }
                let mut obj = self.instantiate_class(&cls, &eval_args, stdout)?;
                if let Value::Object(ref mut inst) = obj {
                    inst.fields.insert("__enum".to_string(), Value::String(field.to_string()));
                }
                return Ok(Some(obj));
            }
        }
        // 2. Constant property on class or ancestor
        for anc in self.all_ancestors(class_name) {
            if let Some(cls) = self.classes.get(&anc).cloned() {
                if let Some(prop) = cls.properties.iter().find(|p| p.name == field && p.constant) {
                    let val = if let Some(ch) = &prop.default {
                        oop::eval_standalone_chunk(self, ch, "__prop_default", stdout)?
                    } else {
                        Value::empty()
                    };
                    return Ok(Some(val));
                }
            }
        }
        Ok(None)
    }

    fn add_event_listener(&mut self, handle_id: usize, args: &[Value]) -> Result<()> {
        if args.len() < 2 {
            return Err(MatroError::arity("addlistener", "2 (eventName, callback)", args.len()));
        }
        let ev = args[0].as_str()?.to_string();
        let Value::Closure(cl) = &args[1] else {
            return Err(MatroError::runtime(
                "addlistener: callback must be a function handle",
            ));
        };
        self.listeners
            .entry((handle_id, ev))
            .or_default()
            .push(cl.clone());
        Ok(())
    }

    fn notify_event(
        &mut self,
        handle: &HandleRef,
        args: &[Value],
        stdout: &mut String,
    ) -> Result<()> {
        if args.is_empty() {
            return Err(MatroError::arity("notify", ">= 1 (eventName [, data])", 0));
        }
        let ev = args[0].as_str()?.to_string();
        let data = if args.len() >= 2 {
            args[1].clone()
        } else {
            let mut m = BTreeMap::new();
            m.insert("EventName".to_string(), Value::String(ev.clone()));
            m.insert("Source".to_string(), Value::Handle(handle.clone()));
            Value::Struct(m)
        };
        let cbs = self
            .listeners
            .get(&(handle.id, ev))
            .cloned()
            .unwrap_or_default();
        for cb in cbs {
            let cb_args = if cb.params.len() >= 2 {
                vec![Value::Handle(handle.clone()), data.clone()]
            } else if cb.params.len() == 1 {
                vec![Value::Handle(handle.clone())]
            } else {
                vec![]
            };
            self.call_closure(&cb, &cb_args, 0, stdout)?;
        }
        Ok(())
    }

    pub(crate) fn isa_class(&self, val: &Value, expected: &str) -> bool {
        let cname = match val {
            Value::Object(o) => Some(o.class_name.as_str()),
            Value::Handle(h) => Some(h.class_name.as_str()),
            _ => None,
        };
        if let Some(name) = cname {
            if self.all_ancestors(name).iter().any(|n| n == expected) {
                return true;
            }
        }
        match expected {
            "numeric" => {
                matches!(val, Value::Matrix(m) if !m.is_logical()) || matches!(val, Value::Sparse(_))
            }
            "float" => {
                matches!(val, Value::Matrix(m) if m.dtype().is_float())
                    || matches!(val, Value::Sparse(_))
            }
            "integer" => matches!(val, Value::Matrix(m) if m.dtype().is_integer()),
            "double" => {
                matches!(val, Value::Matrix(m) if m.dtype() == matro_types::NumericClass::Double)
                    || matches!(val, Value::Sparse(_))
            }
            _ => val.type_name() == expected,
        }
    }

    pub(crate) fn find_class_method(&self, class_name: &str, method_name: &str) -> Option<FunctionProto> {
        for anc in self.all_ancestors(class_name) {
            if let Some(cls) = self.classes.get(&anc) {
                if let Some(m) = cls.methods.iter().find(|m| m.name == method_name) {
                    return Some(m.clone());
                }
            }
        }
        None
    }

    fn resolve_class_method(
        &mut self,
        class_name: &str,
        method_name: &str,
        caller_dir: Option<&str>,
    ) -> Result<Option<FunctionProto>> {
        if let Some(m) = self.find_class_method(class_name, method_name) {
            return Ok(Some(m));
        }
        for anc in self.all_ancestors(class_name) {
            let class_dir = self
                .classes
                .get(&anc)
                .and_then(|c| c.source_file.as_deref())
                .map(|sf| {
                    let p = matro_loader::parent_dir(sf);
                    // If `sf` is `@ClassName/ClassName.m`, its parent is `@ClassName`; go one level up for sibling `@ClassName/` probes.
                    if p.ends_with(&format!("@{anc}")) {
                        matro_loader::parent_dir(&p)
                    } else {
                        p
                    }
                });
            let probe_dir = class_dir.as_deref().or(caller_dir);
            if let Some(mpath) = self
                .loader
                .resolve_class_method_file(&anc, method_name, probe_dir)
            {
                if let Some(lf) = self.load_file(&mpath)? {
                    if let Some(func) = lf.primary_func {
                        return Ok(Some(func));
                    }
                }
            }
        }
        Ok(None)
    }

    fn ensure_superclasses_loaded(
        &mut self,
        superclasses: &[String],
        caller_dir: Option<&str>,
    ) -> Result<()> {
        for sup in superclasses {
            if sup == "handle" || self.classes.contains_key(sup) {
                continue;
            }
            if let Some(m) = self.loader.resolve_in_context(sup, caller_dir) {
                let _ = self.load_file(&m.path)?;
            } else {
                let imports = self.imports.clone();
                for imp in imports {
                    if let Some(pkg) = imp.strip_suffix(".*") {
                        if let Some(mpath) = self.loader.resolve_package_member(pkg, sup) {
                            let _ = self.load_file(&mpath)?;
                            break;
                        }
                    } else if let Some((pkg, member)) = imp.rsplit_once('.') {
                        if member == sup {
                            if let Some(mpath) = self.loader.resolve_package_member(pkg, sup) {
                                let _ = self.load_file(&mpath)?;
                                break;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn collect_class_hierarchy(&self, class_name: &str) -> Vec<ClassProto> {
        let mut chain = Vec::new();
        for anc in self.all_ancestors(class_name) {
            if let Some(cls) = self.classes.get(&anc) {
                chain.push(cls.clone());
            }
        }
        chain.reverse(); // base -> derived
        chain
    }

    fn instantiate_class(
        &mut self,
        cls: &ClassProto,
        args: &[Value],
        stdout: &mut String,
    ) -> Result<Value> {
        let chain = self.collect_class_hierarchy(&cls.name);
        let mut fields = BTreeMap::new();
        for c in &chain {
            for prop in &c.properties {
                if prop.dependent {
                    continue;
                }
                let val = if let Some(ch) = &prop.default {
                    oop::eval_standalone_chunk(self, ch, "__prop_default", stdout)?
                } else {
                    Value::empty()
                };
                fields.insert(prop.name.clone(), val);
            }
        }

        let is_handle = self.is_handle_class(&cls.name);
        let obj = if is_handle {
            let id = self.handles.len();
            self.handles.push(ObjectInstance {
                class_name: cls.name.clone(),
                fields,
            });
            Value::Handle(HandleRef {
                id,
                class_name: cls.name.clone(),
            })
        } else {
            Value::Object(ObjectInstance {
                class_name: cls.name.clone(),
                fields,
            })
        };

        if let Some(ctor) = cls.methods.iter().find(|m| m.name == cls.name).cloned() {
            let expects_self = ctor.params.first().is_some_and(|p| p == "obj");
            let mut ctor_args = Vec::new();
            if expects_self {
                ctor_args.push(obj.clone());
            }
            ctor_args.extend_from_slice(args);

            let mut f = LocalFrame::new(
                &ctor.name,
                ctor.source_file.as_deref(),
                &ctor.local_names,
                None,
                None,
            );
            if let Some(&ret_slot) = ctor.return_slots.first() {
                f.set(ret_slot, obj);
            }
            let results = self.exec_user_func_with_frame(&ctor, &ctor_args, 1, f, stdout)?;
            return Ok(results.into_iter().next().unwrap_or_else(Value::empty));
        }
        Ok(obj)
    }

    #[inline(never)]
    fn assign_value_index(
        &mut self,
        val: &mut Value,
        raw_args: &[StackItem],
        rhs: Value,
        stdout: &mut String,
    ) -> Result<()> {
        if let Value::Map(map) = val {
            if raw_args.len() != 1 {
                return Err(MatroError::index("containers.Map indexing expects 1 key"));
            }
            let key = MapData::key_of(&raw_args[0].clone().into_val()?)?;
            map.entries.insert(key, rhs);
            return Ok(());
        }
        if let Value::Sparse(sp) = val {
            assign_sparse_index(
                std::sync::Arc::make_mut(sp),
                raw_args,
                &rhs.to_dense_matrix()?,
            )?;
            return Ok(());
        }
        if let Value::Cell(cell) = val {
            return assign_cell_parens(cell, raw_args, rhs);
        }
        if matches!(val, Value::Struct(_) | Value::StructArray(_)) {
            return assign_struct_parens(val, raw_args, rhs);
        }
        if let Value::String(s) = val {
            return assign_string_parens(s, raw_args, rhs);
        }
        if let Some(updated) = self.try_subsasgn(val, "()", raw_args, rhs.clone(), stdout)? {
            *val = updated;
            return Ok(());
        }
        if matches!(val, Value::Matrix(m) if m.is_empty()) {
            if let Value::Struct(_) | Value::StructArray(_) = rhs {
                *val = Value::StructArray(matro_types::StructArray {
                    rows: 0,
                    cols: 0,
                    items: Vec::new(),
                });
                return assign_struct_parens(val, raw_args, rhs);
            }
            if let Value::Cell(_) = rhs {
                *val = Value::Cell(CellArray::empty());
                if let Value::Cell(cell) = val {
                    return assign_cell_parens(cell, raw_args, rhs);
                }
            }
        }
        assign_matrix_index(val.as_matrix_mut()?, raw_args, &rhs.to_dense_matrix()?)
    }

    #[inline(never)]
    fn try_subsasgn(
        &mut self,
        val: &Value,
        kind: &str,
        raw_args: &[StackItem],
        rhs: Value,
        stdout: &mut String,
    ) -> Result<Option<Value>> {
        let cname = match val {
            Value::Object(o) => Some(o.class_name.clone()),
            Value::Handle(h) => Some(h.class_name.clone()),
            _ => None,
        };
        let Some(cls) = cname else {
            return Ok(None);
        };
        let Some(proto) = self.find_class_method(&cls, "subsasgn") else {
            return Ok(None);
        };
        let subs = flatten_call_args(raw_args.to_vec())?;
        let mut s_map = BTreeMap::new();
        s_map.insert("type".to_string(), Value::String(kind.to_string()));
        let cell = if subs.is_empty() {
            CellArray::empty()
        } else {
            CellArray::from_rows(vec![subs])?
        };
        s_map.insert("subs".to_string(), Value::Cell(cell));
        let is_handle = matches!(val, Value::Handle(_));
        let rets = self.call_user_func(
            &proto,
            &[val.clone(), Value::Struct(s_map), rhs],
            if is_handle { 0 } else { 1 },
            None,
            stdout,
        )?;
        if is_handle {
            Ok(Some(val.clone()))
        } else {
            Ok(Some(rets.into_iter().next().unwrap_or_else(|| val.clone())))
        }
    }

    /// Load and index a `.m` file by normalized VFS path, keeping its local subfunctions strictly file-scoped.
    #[inline(never)]
    fn load_file(&mut self, resolved_path: &str) -> Result<Option<LoadedFile>> {
        let key = matro_loader::normalize_preserve_abs(resolved_path);
        if let Some(lf) = self.loaded_files.get(&key) {
            return Ok(Some(lf.clone()));
        }
        let Some(chunk) = self.loader.load_module_at(&key)? else {
            return Ok(None);
        };
        let dir = matro_loader::parent_dir(&key);
        let stem = matro_loader::file_stem(&key);

        let lf = if chunk.is_definition_only() && !chunk.classes.is_empty() {
            let cls = chunk.classes[0].clone();
            let supers = cls.superclasses.clone();
            let mut local_funcs = HashMap::new();
            for f in &chunk.functions {
                local_funcs.insert(f.name.clone(), f.clone());
            }
            self.classes.insert(cls.name.clone(), cls.clone());
            let lf = LoadedFile {
                path: key.clone(),
                primary_name: cls.name.clone(),
                primary_func: None,
                primary_class: Some(cls),
                local_funcs,
                is_script: false,
                chunk,
            };
            self.loaded_files.insert(key, lf.clone());
            self.ensure_superclasses_loaded(&supers, Some(&dir))?;
            lf
        } else if chunk.is_definition_only() && !chunk.functions.is_empty() {
            let primary = chunk.functions[0].clone();
            let mut local_funcs = HashMap::new();
            if primary.name != stem {
                local_funcs.insert(primary.name.clone(), primary.clone());
            }
            for sub in chunk.functions.iter().skip(1) {
                local_funcs.insert(sub.name.clone(), sub.clone());
            }
            let lf = LoadedFile {
                path: key.clone(),
                primary_name: stem,
                primary_func: Some(primary),
                primary_class: None,
                local_funcs,
                is_script: false,
                chunk,
            };
            self.loaded_files.insert(key, lf.clone());
            lf
        } else {
            // Script file (possibly with local functions at the bottom)
            let mut local_funcs = HashMap::new();
            for f in &chunk.functions {
                local_funcs.insert(f.name.clone(), f.clone());
            }
            let lf = LoadedFile {
                path: key.clone(),
                primary_name: stem,
                primary_func: None,
                primary_class: None,
                local_funcs,
                is_script: true,
                chunk,
            };
            self.loaded_files.insert(key, lf.clone());
            lf
        };
        Ok(Some(lf))
    }

    fn resolve_imported_file(&mut self, name: &str) -> Result<Option<LoadedFile>> {
        let imports = self.imports.clone();
        for imp in imports {
            if let Some(pkg) = imp.strip_suffix(".*") {
                if let Some(mpath) = self.loader.resolve_package_member(pkg, name) {
                    if let Some(lf) = self.load_file(&mpath)? {
                        return Ok(Some(lf));
                    }
                }
            } else if let Some((pkg, member)) = imp.rsplit_once('.') {
                if member == name {
                    if let Some(mpath) = self.loader.resolve_package_member(pkg, name) {
                        if let Some(lf) = self.load_file(&mpath)? {
                            return Ok(Some(lf));
                        }
                    }
                }
            }
        }
        Ok(None)
    }

    fn exec_script_file(
        &mut self,
        lf: &LoadedFile,
        frame: Option<&mut LocalFrame<'_>>,
        stdout: &mut String,
    ) -> Result<()> {
        let prev_active = self.active_file.replace(lf.path.clone());
        let mut ignored = None;
        let res = self.run_frame(&lf.chunk, frame, stdout, &mut ignored);
        self.active_file = prev_active;
        res
    }

    #[inline(never)]
    fn resolve_bare_ident(
        &mut self,
        name: &str,
        mut frame: Option<&mut LocalFrame<'_>>,
        stdout: &mut String,
    ) -> Result<StackItem> {
        let caller_file: Option<String> = frame
            .as_deref()
            .and_then(|f| f.source_file)
            .map(String::from)
            .or_else(|| self.active_file.clone());
        let caller_dir = caller_file.as_deref().map(matro_loader::parent_dir);

        // 1. File-local subfunction in caller's file
        if let Some(ref cf) = caller_file {
            if let Some(lf) = self.loaded_files.get(cf) {
                if let Some(func) = lf.local_funcs.get(name).cloned() {
                    let mut rets = self.call_user_func(&func, &[], 1, None, stdout)?;
                    return Ok(StackItem::Val(rets.pop().unwrap_or_else(Value::empty)));
                }
                if lf.primary_name == name {
                    if let Some(func) = lf.primary_func.clone() {
                        let mut rets = self.call_user_func(&func, &[], 1, None, stdout)?;
                        return Ok(StackItem::Val(rets.pop().unwrap_or_else(Value::empty)));
                    }
                }
            }
        }

        // 2. Private function in caller directory's `private/` folder
        if let Some(cd) = caller_dir.as_deref() {
            if let Some(pm) = self.loader.resolve_private(name, cd) {
                if let Some(lf) = self.load_file(&pm.path)? {
                    if let Some(func) = lf.primary_func {
                        let mut rets = self.call_user_func(&func, &[], 1, None, stdout)?;
                        return Ok(StackItem::Val(rets.pop().unwrap_or_else(Value::empty)));
                    }
                }
            }
        }

        // 3. Inline workspace functions / classes
        if let Some(func) = self.functions.get(name).cloned() {
            let mut rets = self.call_user_func(&func, &[], 1, None, stdout)?;
            return Ok(StackItem::Val(rets.pop().unwrap_or_else(Value::empty)));
        }
        if let Some(cls) = self.classes.get(name).cloned() {
            if self.class_has_static_or_enum_or_const(name) {
                let mut f = BTreeMap::new();
                f.insert("__name".to_string(), Value::String(name.to_string()));
                return Ok(StackItem::Val(Value::Object(ObjectInstance {
                    class_name: "__class_ref".to_string(),
                    fields: f,
                })));
            }
            let obj = self.instantiate_class(&cls, &[], stdout)?;
            return Ok(StackItem::Val(obj));
        }

        // 4. External `.m` file (sibling / cwd / search_paths / @Class / imported +package)
        let resolved_lf = if let Some(m) = self.loader.resolve_in_context(name, caller_dir.as_deref()) {
            self.load_file(&m.path)?
        } else {
            self.resolve_imported_file(name)?
        };
        if let Some(lf) = resolved_lf {
            if let Some(func) = lf.primary_func {
                let mut rets = self.call_user_func(&func, &[], 1, None, stdout)?;
                return Ok(StackItem::Val(rets.pop().unwrap_or_else(Value::empty)));
            }
            if let Some(cls) = lf.primary_class {
                if self.class_has_static_or_enum_or_const(&cls.name) {
                    let mut f = BTreeMap::new();
                    f.insert("__name".to_string(), Value::String(cls.name));
                    return Ok(StackItem::Val(Value::Object(ObjectInstance {
                        class_name: "__class_ref".to_string(),
                        fields: f,
                    })));
                }
                let obj = self.instantiate_class(&cls, &[], stdout)?;
                return Ok(StackItem::Val(obj));
            }
            if lf.is_script {
                self.exec_script_file(&lf, frame.as_deref_mut(), stdout)?;
                return Ok(StackItem::Void);
            }
        }

        // 5. Built-in 0-arg intrinsics (pi, eps, i, j, inf, nan, true, false, ans, pwd, ...)
        if let Some(res) = call_intrinsic(
            name,
            &[],
            1,
            &mut IntrinsicContext {
                stdout,
                figure: &mut self.figure,
                rng: &mut self.rng,
                loader: &mut self.loader,
            },
        ) {
            let vals = res?;
            return Ok(StackItem::Val(
                vals.into_iter().next().unwrap_or_else(Value::empty),
            ));
        }

        // 6. MATLAB `+package` namespace root
        if self.loader.has_package(name) {
            let mut f = BTreeMap::new();
            f.insert("__path".to_string(), Value::String(name.to_string()));
            return Ok(StackItem::Val(Value::Object(ObjectInstance {
                class_name: "__pkg_ref".to_string(),
                fields: f,
            })));
        }

        Err(MatroError::undefined(name))
    }

    #[inline(never)]
    fn resolve_pkg_field(
        &mut self,
        pkg_path: &str,
        field: &str,
        stdout: &mut String,
    ) -> Result<Option<Value>> {
        let sub_pkg = format!("{pkg_path}.{field}");
        if self.loader.has_package(&sub_pkg) {
            let mut f = BTreeMap::new();
            f.insert("__path".to_string(), Value::String(sub_pkg));
            return Ok(Some(Value::Object(ObjectInstance {
                class_name: "__pkg_ref".to_string(),
                fields: f,
            })));
        }
        if let Some(mpath) = self.loader.resolve_package_member(pkg_path, field) {
            let Some(lf) = self.load_file(&mpath)? else {
                return Ok(None);
            };
            if let Some(cls) = lf.primary_class {
                if self.class_has_static_or_enum_or_const(&cls.name) {
                    let mut f = BTreeMap::new();
                    f.insert("__name".to_string(), Value::String(cls.name));
                    return Ok(Some(Value::Object(ObjectInstance {
                        class_name: "__class_ref".to_string(),
                        fields: f,
                    })));
                }
                return Ok(Some(self.instantiate_class(&cls, &[], stdout)?));
            }
            if let Some(func) = lf.primary_func {
                let rets = self.call_user_func(&func, &[], 1, None, stdout)?;
                return Ok(Some(rets.into_iter().next().unwrap_or_else(Value::empty)));
            }
        }
        Ok(None)
    }

    #[inline(never)]
    fn call_package_member(
        &mut self,
        pkg_path: &str,
        member: &str,
        args: &[Value],
        nargout: usize,
        stdout: &mut String,
    ) -> Result<Vec<Value>> {
        let Some(mpath) = self.loader.resolve_package_member(pkg_path, member) else {
            return Err(MatroError::runtime(format!(
                "Unable to resolve package member '{pkg_path}.{member}'"
            )));
        };
        let Some(lf) = self.load_file(&mpath)? else {
            return Err(MatroError::runtime(format!(
                "Unable to load package member '{pkg_path}.{member}'"
            )));
        };
        if let Some(func) = lf.primary_func {
            return self.call_user_func(&func, args, nargout, None, stdout);
        }
        if let Some(cls) = lf.primary_class {
            return Ok(vec![self.instantiate_class(&cls, args, stdout)?]);
        }
        Err(MatroError::runtime(format!(
            "Package file '{mpath}' did not define function or class '{member}'"
        )))
    }

    fn do_tic(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        self.tic_stack.push(TicMark::Instant(std::time::Instant::now()));
        #[cfg(target_arch = "wasm32")]
        self.tic_stack.push(TicMark::Steps(self.steps));
    }

    fn do_toc(&mut self) -> f64 {
        match self.tic_stack.pop() {
            #[cfg(not(target_arch = "wasm32"))]
            Some(TicMark::Instant(t0)) => t0.elapsed().as_secs_f64(),
            #[cfg(target_arch = "wasm32")]
            Some(TicMark::Steps(s0)) => {
                // On wasm32-unknown-unknown without host clock import, report approximate seconds (~1e7 ops/s)
                self.steps.saturating_sub(s0) as f64 / 1e7
            }
            None => 0.0,
        }
    }

    fn invoke_named_callable(
        &mut self,
        name: &str,
        call_args: &[Value],
        nargout: usize,
        frame: Option<&mut LocalFrame<'_>>,
        stdout: &mut String,
    ) -> Result<Vec<Value>> {
        self.invoke_named_callable_in_file(name, call_args, nargout, frame, None, stdout)
    }

    fn invoke_named_callable_in_file(
        &mut self,
        name: &str,
        call_args: &[Value],
        nargout: usize,
        mut frame: Option<&mut LocalFrame<'_>>,
        explicit_source_file: Option<&str>,
        stdout: &mut String,
    ) -> Result<Vec<Value>> {
        match name {
            "tic" => {
                self.do_tic();
                return Ok(vec![]);
            }
            "toc" => {
                let secs = self.do_toc();
                return Ok(vec![Value::scalar(secs)]);
            }
            "profile" => {
                let cmd = call_args
                    .first()
                    .and_then(|v| v.as_str().ok())
                    .unwrap_or("report");
                match cmd {
                    "on" => {
                        self.profiling = true;
                        self.profile.clear();
                    }
                    "off" => self.profiling = false,
                    "clear" => self.profile.clear(),
                    "report" => {
                        let mut rows: Vec<(&String, &ProfileEntry)> =
                            self.profile.iter().collect();
                        rows.sort_by(|a, b| b.1.instructions.cmp(&a.1.instructions));
                        stdout.push_str("Function                 Calls    Instructions\n");
                        stdout.push_str("----------------------------------------------\n");
                        for (fname, entry) in rows {
                            stdout.push_str(&format!(
                                "{fname:<24} {:<8} {}\n",
                                entry.calls, entry.instructions
                            ));
                        }
                    }
                    other => {
                        return Err(MatroError::runtime(format!(
                            "profile: expected 'on', 'off', 'clear', or 'report', got '{other}'"
                        )));
                    }
                }
                return Ok(vec![]);
            }
            "save" => {
                if call_args.is_empty() {
                    return Err(MatroError::arity("save", ">= 1 (filename [, vars...])", 0));
                }
                let path = call_args[0].as_str()?;
                let mut pairs: Vec<(String, Value)> = Vec::new();
                if call_args.len() == 1 {
                    let mut names: Vec<&String> = self.globals.keys().collect();
                    names.sort();
                    for n in names {
                        let v = &self.globals[n];
                        if matches!(v, Value::Matrix(_) | Value::Sparse(_) | Value::String(_)) {
                            pairs.push((n.clone(), v.clone()));
                        }
                    }
                } else {
                    for arg in &call_args[1..] {
                        let vname = arg.as_str()?;
                        let val = lookup_var(vname, frame.as_deref(), &self.globals)
                            .cloned()
                            .ok_or_else(|| MatroError::undefined(vname))?;
                        pairs.push((vname.to_string(), val));
                    }
                }
                let refs: Vec<(&str, &Value)> =
                    pairs.iter().map(|(k, v)| (k.as_str(), v)).collect();
                self.loader.save_mat(path, &refs)?;
                return Ok(vec![]);
            }
            "load" => {
                if call_args.is_empty() {
                    return Err(MatroError::arity("load", ">= 1 (filename.mat [, vars...])", 0));
                }
                let path = call_args[0].as_str()?;
                let mut filter_vars: Vec<String> = Vec::new();
                let mut is_ascii = false;
                for a in &call_args[1..] {
                    let s = a.as_str()?;
                    if s.eq_ignore_ascii_case("-ascii") {
                        is_ascii = true;
                    } else if !s.starts_with('-') {
                        filter_vars.push(s.to_string());
                    }
                }
                let mut vars = if is_ascii
                    || (!path.to_ascii_lowercase().ends_with(".mat")
                        && self.loader.vfs().read_to_string(path).is_some())
                {
                    let text = self
                        .loader
                        .vfs()
                        .read_to_string(path)
                        .ok_or_else(|| MatroError::runtime(format!("load: file '{path}' not found")))?;
                    let mut rows: Vec<Vec<f64>> = Vec::new();
                    for line in text.lines() {
                        let trimmed = line.trim();
                        if trimmed.is_empty() || trimmed.starts_with('%') || trimmed.starts_with('#') {
                            continue;
                        }
                        let vals: Vec<f64> = trimmed
                            .split(|c: char| c.is_whitespace() || c == ',' || c == ';')
                            .filter(|t| !t.is_empty())
                            .map(|t| t.parse::<f64>().unwrap_or(f64::NAN))
                            .collect();
                        if !vals.is_empty() {
                            rows.push(vals);
                        }
                    }
                    let mat = if rows.is_empty() {
                        Matrix::empty()
                    } else {
                        let nc = rows[0].len();
                        let nr = rows.len();
                        let mut m = Matrix::zeros(nr, nc);
                        for (r, row) in rows.iter().enumerate() {
                            for (c, &v) in row.iter().take(nc).enumerate() {
                                m.set(r, c, v);
                            }
                        }
                        m
                    };
                    if nargout >= 1 {
                        return Ok(vec![Value::matrix(mat)]);
                    }
                    let stem = path
                        .rsplit(['/', '\\'])
                        .next()
                        .unwrap_or(path)
                        .split('.')
                        .next()
                        .unwrap_or("data")
                        .to_string();
                    let mut map = BTreeMap::new();
                    map.insert(stem, Value::matrix(mat));
                    map
                } else {
                    self.loader.load_mat(path)?
                };
                if !filter_vars.is_empty() {
                    vars.retain(|k, _| filter_vars.iter().any(|f| f == k));
                }
                if nargout == 0 {
                    for (k, v) in &vars {
                        store_var(k.clone(), v.clone(), frame.as_deref_mut(), &mut self.globals);
                    }
                    return Ok(vec![]);
                }
                return Ok(vec![Value::Struct(vars)]);
            }
            "feval" => {
                if call_args.is_empty() {
                    return Err(MatroError::arity("feval", ">= 1", 0));
                }
                return match &call_args[0] {
                    Value::Closure(cl) => self.call_closure(cl, &call_args[1..], nargout, stdout),
                    Value::String(fname) => {
                        self.invoke_named_callable(fname, &call_args[1..], nargout, frame, stdout)
                    }
                    other => Err(MatroError::runtime(format!(
                        "feval: expected function handle or name, got {}",
                        other.type_name()
                    ))),
                };
            }
            "arrayfun" => return self.exec_arrayfun(call_args, stdout),
            "accumarray" => {
                if call_args.len() >= 4 && matches!(&call_args[3], Value::Closure(_)) {
                    return self.exec_accumarray_closure(call_args, stdout);
                }
            }
            "fopen" => return self.files.fopen(call_args, nargout, &self.loader),
            "fclose" => return self.files.fclose(call_args, &mut self.loader),
            "fprintf" => return self.files.fprintf(call_args, nargout, stdout),
            "fscanf" => return self.files.fscanf(call_args, nargout),
            "sscanf" => return self.files.sscanf(call_args, nargout),
            "fgetl" => return self.files.fgetl(call_args, false),
            "fgets" => return self.files.fgetl(call_args, true),
            "feof" => return self.files.feof(call_args),
            "ftell" => return self.files.ftell(call_args),
            "fseek" => return self.files.fseek(call_args),
            "frewind" => return self.files.frewind(call_args),
            "ferror" => return self.files.ferror(call_args, nargout),
            "fwrite" => return self.files.fwrite(call_args),
            "fread" => return self.files.fread(call_args, nargout),
            "textscan" => return self.files.textscan(call_args),
            "pcg" | "gmres" | "bicgstab" | "minres" | "lsqr" | "cgs" | "bicg" => {
                return krylov::solve_krylov(self, name, call_args, nargout, stdout);
            }
            "lsqnonneg" => return krylov::solve_lsqnonneg(call_args, nargout),
            "odeset" => return Ok(vec![exec_odeset(call_args)?]),
            "odeget" => return Ok(vec![exec_odeget(call_args)?]),
            "optimset" | "optimoptions" => return Ok(vec![exec_optimset(name, call_args)?]),
            "inputParser" => return Ok(vec![self.new_input_parser()]),
            "remove" => {
                if call_args.len() != 2 {
                    return Err(MatroError::arity("remove", "2 (map, key)", call_args.len()));
                }
                let Value::Map(mut m) = call_args[0].clone() else {
                    return Err(MatroError::runtime("remove: first argument must be containers.Map"));
                };
                m.entries.remove(&MapData::key_of(&call_args[1])?);
                return Ok(vec![Value::Map(m)]);
            }
            "eval" | "evalc" => {
                if call_args.len() != 1 {
                    return Err(MatroError::arity(name, "1 (code string)", call_args.len()));
                }
                let code = call_args[0].as_str()?;
                let eval_chunk = compile(code)?;
                let mut captured = String::new();
                let mut eval_last = None;
                let target_out = if name == "evalc" {
                    &mut captured
                } else {
                    stdout
                };
                self.run_frame(&eval_chunk, frame, target_out, &mut eval_last)?;
                if name == "evalc" {
                    let mut out = vec![Value::String(captured)];
                    if nargout >= 2 {
                        out.push(eval_last.unwrap_or_else(Value::empty));
                    }
                    return Ok(out);
                }
                return Ok(eval_last.map(|v| vec![v]).unwrap_or_default());
            }
            "cellfun" => return self.exec_cellfun(call_args, stdout),
            "structfun" => {
                if call_args.len() < 2 {
                    return Err(MatroError::arity("structfun", ">= 2 (fn, S)", call_args.len()));
                }
                let Value::Struct(map) = &call_args[1] else {
                    return Err(MatroError::runtime("structfun: second argument must be a struct"));
                };
                let mut uniform = true;
                let mut i = 2;
                while i + 1 < call_args.len() {
                    if let Ok(k) = call_args[i].as_str() {
                        if k.eq_ignore_ascii_case("UniformOutput") {
                            uniform = call_args[i + 1].is_truthy();
                        }
                    }
                    i += 2;
                }
                if uniform {
                    let mut vals = Vec::with_capacity(map.len());
                    let mut all_logical = !map.is_empty();
                    for v in map.values() {
                        let r = self.invoke_callable_val(&call_args[0], &[v.clone()], stdout)?;
                        let rm = r.to_dense_matrix()?;
                        if rm.dtype() != matro_types::NumericClass::Logical {
                            all_logical = false;
                        }
                        let sc = rm.as_complex_scalar().ok_or_else(|| {
                            MatroError::runtime(
                                "structfun: non-scalar output; pass 'UniformOutput', false",
                            )
                        })?;
                        vals.push(sc);
                    }
                    let mut out = Matrix::complex_col_vec(vals).normalize();
                    if all_logical {
                        out = out.into_logical();
                    }
                    return Ok(vec![Value::matrix(out)]);
                } else {
                    let mut out = BTreeMap::new();
                    for (k, v) in map {
                        let r = self.invoke_callable_val(&call_args[0], &[v.clone()], stdout)?;
                        out.insert(k.clone(), r);
                    }
                    return Ok(vec![Value::Struct(out)]);
                }
            }
            "bsxfun" => return self.exec_bsxfun(call_args, stdout),
            "spfun" => return self.exec_spfun(call_args, stdout),
            "MException" => return self.exec_mexception(call_args),
            "throw" | "rethrow" => {
                if call_args.len() != 1 {
                    return Err(MatroError::arity(name, "1 (ME)", call_args.len()));
                }
                return Err(mexception_to_error(&call_args[0]));
            }
            "lasterror" => return self.exec_lasterror(call_args),
            "functions" => return self.exec_functions(call_args),
            "func2str" => return self.exec_func2str(call_args),
            "str2func" => return self.exec_str2func(call_args, stdout),
            "input" => {
                let is_str = call_args
                    .get(1)
                    .and_then(|v| v.as_str().ok())
                    .is_some_and(|s| s.eq_ignore_ascii_case("s"));
                if is_str {
                    return Ok(vec![Value::String(String::new())]);
                }
                return Ok(vec![Value::empty()]);
            }
            "narginchk" | "nargoutchk" => {
                if call_args.len() != 2 {
                    return Err(MatroError::arity(name, "2 (min, max)", call_args.len()));
                }
                let lo = call_args[0].as_scalar()?;
                let hi = call_args[1].as_scalar()?;
                let var = if name == "narginchk" {
                    "nargin"
                } else {
                    "nargout"
                };
                if let Some(actual) = frame
                    .as_deref()
                    .and_then(|f| f.get_by_name(var))
                    .and_then(|v| v.as_scalar().ok())
                {
                    if actual < lo {
                        return Err(MatroError::runtime(format!(
                            "Not enough {} arguments (got {}, expected >= {})",
                            if name == "narginchk" { "input" } else { "output" },
                            actual,
                            lo
                        )));
                    }
                    if actual > hi {
                        return Err(MatroError::runtime(format!(
                            "Too many {} arguments (got {}, expected <= {})",
                            if name == "narginchk" { "input" } else { "output" },
                            actual,
                            hi
                        )));
                    }
                }
                return Ok(vec![]);
            }
            "ode45" | "ode15s" | "ode23" | "ode113" | "ode23s" | "ode23t" | "ode23tb" => {
                return self.exec_ode_solver(name, call_args, nargout, stdout);
            }
            "bvpinit" => {
                if call_args.len() != 2 {
                    return Err(MatroError::arity("bvpinit", "2 (x, yinit)", call_args.len()));
                }
                let x = call_args[0].to_dense_matrix()?;
                let solinit = solvers_ext::eval_bvpinit(self, &x, &call_args[1], stdout)?;
                return Ok(vec![solinit]);
            }
            "bvp4c" => {
                if call_args.len() != 3 {
                    return Err(MatroError::arity(
                        "bvp4c",
                        "3 (odefun, bcfun, solinit)",
                        call_args.len(),
                    ));
                }
                let Value::Closure(ode_cl) = &call_args[0] else {
                    return Err(MatroError::runtime(
                        "bvp4c: first argument must be an ODE function handle @(x, y) ...",
                    ));
                };
                let Value::Closure(bc_cl) = &call_args[1] else {
                    return Err(MatroError::runtime(
                        "bvp4c: second argument must be a BC function handle @(ya, yb) ...",
                    ));
                };
                let sol = solvers_ext::solve_bvp4c(self, ode_cl, bc_cl, &call_args[2], stdout)?;
                return Ok(vec![sol]);
            }
            "deval" => {
                if call_args.len() != 2 {
                    return Err(MatroError::arity("deval", "2 (sol, xint)", call_args.len()));
                }
                let xint = call_args[1].to_dense_matrix()?;
                let val = solvers_ext::eval_deval(&call_args[0], &xint)?;
                return Ok(vec![val]);
            }
            "pdepe" => {
                if call_args.len() != 6 {
                    return Err(MatroError::arity(
                        "pdepe",
                        "6 (m, pdefun, icfun, bcfun, xmesh, tspan)",
                        call_args.len(),
                    ));
                }
                let m_sym = call_args[0].as_scalar()?;
                let Value::Closure(pde_cl) = &call_args[1] else {
                    return Err(MatroError::runtime("pdepe: pdefun must be a function handle"));
                };
                let Value::Closure(ic_cl) = &call_args[2] else {
                    return Err(MatroError::runtime("pdepe: icfun must be a function handle"));
                };
                let Value::Closure(bc_cl) = &call_args[3] else {
                    return Err(MatroError::runtime("pdepe: bcfun must be a function handle"));
                };
                let xmesh = call_args[4].to_dense_matrix()?;
                let tspan = call_args[5].to_dense_matrix()?;
                let sol = solvers_ext::solve_pdepe(
                    self, m_sym, pde_cl, ic_cl, bc_cl, &xmesh, &tspan, stdout,
                )?;
                return Ok(vec![sol]);
            }
            "fzero" => return self.exec_fzero(call_args, nargout, stdout),
            "clear" => {
                if call_args.is_empty()
                    || call_args.iter().any(|a| a.as_str().ok() == Some("all"))
                {
                    self.globals.clear();
                    self.persistents.clear();
                } else {
                    for arg in call_args {
                        let vname = arg.as_str()?;
                        self.globals.remove(vname);
                        self.persistents.retain(|(f, _), _| f != vname);
                    }
                }
                return Ok(vec![]);
            }
            "isa" => {
                if call_args.len() != 2 {
                    return Err(MatroError::arity("isa", "2", call_args.len()));
                }
                let expected = call_args[1].as_str()?;
                return Ok(vec![Value::bool(self.isa_class(&call_args[0], expected))]);
            }
            "addlistener" => {
                if call_args.len() < 3 {
                    return Err(MatroError::arity(
                        "addlistener",
                        "3 (handleObj, eventName, callback)",
                        call_args.len(),
                    ));
                }
                let Value::Handle(ref h) = call_args[0] else {
                    return Err(MatroError::runtime(
                        "addlistener: first argument must be a handle object",
                    ));
                };
                self.add_event_listener(h.id, &call_args[1..])?;
                return Ok(vec![]);
            }
            "notify" => {
                if call_args.len() < 2 {
                    return Err(MatroError::arity(
                        "notify",
                        ">= 2 (handleObj, eventName [, data])",
                        call_args.len(),
                    ));
                }
                let Value::Handle(h) = call_args[0].clone() else {
                    return Err(MatroError::runtime(
                        "notify: first argument must be a handle object",
                    ));
                };
                self.notify_event(&h, &call_args[1..], stdout)?;
                return Ok(vec![]);
            }
            "builtin" => {
                if call_args.is_empty() {
                    return Err(MatroError::arity("builtin", ">= 1 (funcName, args...)", 0));
                }
                let fname = call_args[0].as_str()?;
                if let Some(res) = call_intrinsic(
                    fname,
                    &call_args[1..],
                    nargout,
                    &mut IntrinsicContext {
                        stdout,
                        figure: &mut self.figure,
                        rng: &mut self.rng,
                        loader: &mut self.loader,
                    },
                ) {
                    return res;
                }
                return Err(MatroError::undefined(fname));
            }
            "fminsearch" | "fminunc" | "fmincon" | "fsolve" | "lsqnonlin" | "lsqcurvefit"
            | "fminbnd" => {
                return self.exec_opt_solver(name, call_args, nargout, stdout);
            }
            "integral" | "quad" | "quadl" | "quadgk" => {
                return self.exec_integral(name, call_args, stdout);
            }
            "integral2" => {
                if call_args.len() < 5 {
                    return Err(MatroError::arity(
                        "integral2",
                        ">= 5 (fun, xmin, xmax, ymin, ymax)",
                        call_args.len(),
                    ));
                }
                let Value::Closure(cl) = &call_args[0] else {
                    return Err(MatroError::runtime(
                        "integral2: first argument must be a function handle @(x, y) ...",
                    ));
                };
                let xmin = call_args[1].as_scalar()?;
                let xmax = call_args[2].as_scalar()?;
                let q = solvers_ext::solve_integral2(
                    self,
                    cl,
                    xmin,
                    xmax,
                    &call_args[3],
                    &call_args[4],
                    stdout,
                )?;
                return Ok(vec![Value::scalar(q)]);
            }
            "import" => {
                for a in call_args {
                    let pat = a.as_str()?.trim().to_string();
                    if !pat.is_empty() && !self.imports.contains(&pat) {
                        self.imports.push(pat);
                    }
                }
                return Ok(vec![]);
            }
            "run" => {
                if call_args.len() != 1 {
                    return Err(MatroError::arity("run", "1", call_args.len()));
                }
                let path = call_args[0].as_str()?;
                let caller_dir = explicit_source_file
                    .or_else(|| frame.as_deref().and_then(|f| f.source_file))
                    .or(self.active_file.as_deref())
                    .map(matro_loader::parent_dir);
                let resolved = self
                    .loader
                    .resolve_in_context(path, caller_dir.as_deref())
                    .map(|m| m.path)
                    .or_else(|| self.loader.resolve_path(path))
                    .ok_or_else(|| MatroError::runtime(format!("Script not found: '{path}'")))?;
                let Some(lf) = self.load_file(&resolved)? else {
                    return Err(MatroError::runtime(format!("Script not found: '{path}'")));
                };
                if lf.is_script {
                    self.exec_script_file(&lf, frame.as_deref_mut(), stdout)?;
                } else if let Some(func) = lf.primary_func {
                    if func.params.is_empty() {
                        let _ = self.call_user_func(&func, &[], 0, None, stdout)?;
                    } else {
                        return Err(MatroError::runtime(format!(
                            "Cannot run function '{}' without input arguments",
                            lf.primary_name
                        )));
                    }
                }
                return Ok(vec![]);
            }
            "which" => {
                if call_args.is_empty() {
                    return Err(MatroError::arity("which", "1", 0));
                }
                let q = call_args[0].as_str()?;
                let caller_file: Option<String> = explicit_source_file
                    .map(String::from)
                    .or_else(|| {
                        frame
                            .as_deref()
                            .and_then(|f| f.source_file)
                            .map(String::from)
                    })
                    .or_else(|| self.active_file.clone());
                let caller_dir = caller_file.as_deref().map(matro_loader::parent_dir);

                let found = if lookup_var(q, frame.as_deref(), &self.globals).is_some() {
                    "variable".to_string()
                } else if caller_file.as_deref().is_some_and(|cf| {
                    self.loaded_files
                        .get(cf)
                        .is_some_and(|lf| lf.local_funcs.contains_key(q))
                }) {
                    format!("{} (local function)", caller_file.as_deref().unwrap_or(""))
                } else if let Some(m) = self.loader.resolve_in_context(q, caller_dir.as_deref()) {
                    m.path
                } else if let Some(lf) = self.resolve_imported_file(q)? {
                    lf.path
                } else if self.functions.contains_key(q) {
                    format!("{q} (workspace function)")
                } else if self.classes.contains_key(q) {
                    format!("{q} (workspace class)")
                } else {
                    String::new()
                };
                return Ok(vec![Value::String(found)]);
            }
            "exist" => {
                if call_args.is_empty() || call_args.len() > 2 {
                    return Err(MatroError::arity("exist", "1 or 2 (name [, searchType])", call_args.len()));
                }
                let q = call_args[0].as_str()?;
                let kind = call_args
                    .get(1)
                    .map(|v| v.as_str())
                    .transpose()?
                    .map(|s| s.to_ascii_lowercase());
                let caller_file: Option<String> = explicit_source_file
                    .map(String::from)
                    .or_else(|| {
                        frame
                            .as_deref()
                            .and_then(|f| f.source_file)
                            .map(String::from)
                    })
                    .or_else(|| self.active_file.clone());
                let caller_dir = caller_file.as_deref().map(matro_loader::parent_dir);

                let is_var = lookup_var(q, frame.as_deref(), &self.globals).is_some();
                let is_class = self.classes.contains_key(q);
                let is_file = self.functions.contains_key(q)
                    || caller_file.as_deref().is_some_and(|cf| {
                        self.loaded_files
                            .get(cf)
                            .is_some_and(|lf| lf.local_funcs.contains_key(q))
                    })
                    || self
                        .loader
                        .resolve_in_context(q, caller_dir.as_deref())
                        .is_some()
                    || self.loader.resolve_path(q).is_some();
                let is_dir = self.loader.vfs().is_dir(q);
                let is_builtin = is_vm_or_intrinsic_builtin(q);

                let code = match kind.as_deref() {
                    Some("var") => if is_var { 1.0 } else { 0.0 },
                    Some("file") => if is_file { 2.0 } else if is_dir { 7.0 } else { 0.0 },
                    Some("dir") => if is_dir { 7.0 } else { 0.0 },
                    Some("builtin") => if is_builtin { 5.0 } else { 0.0 },
                    Some("class") => if is_class { 8.0 } else { 0.0 },
                    _ => {
                        if is_var {
                            1.0
                        } else if is_class {
                            8.0
                        } else if is_file {
                            2.0
                        } else if is_builtin {
                            5.0
                        } else if is_dir {
                            7.0
                        } else {
                            0.0
                        }
                    }
                };
                return Ok(vec![Value::scalar(code)]);
            }
            _ => {}
        }

        let caller_file: Option<String> = explicit_source_file
            .map(String::from)
            .or_else(|| {
                frame
                    .as_deref()
                    .and_then(|f| f.source_file)
                    .map(String::from)
            })
            .or_else(|| self.active_file.clone());
        let caller_dir = caller_file.as_deref().map(matro_loader::parent_dir);

        // 1. Method dispatch when first argument is a `classdef` Object or Handle (MATLAB syntax `method(obj, ...)`)
        if let Some(Value::Handle(h)) = call_args.first() {
            if h.class_name == "inputParser" {
                return self.call_input_parser_method(h.id, name, &call_args[1..]);
            }
        }
        let first_class = match call_args.first() {
            Some(Value::Object(obj)) => Some(obj.class_name.clone()),
            Some(Value::Handle(h)) => Some(h.class_name.clone()),
            _ => None,
        };
        if let Some(cname) = first_class {
            if let Some(proto) =
                self.resolve_class_method(&cname, name, caller_dir.as_deref())?
            {
                return self.call_user_func(&proto, call_args, nargout, None, stdout);
            }
        }

        // 2. File-local subfunction (or primary self-recursion) in caller's `.m` file
        if let Some(ref cf) = caller_file {
            if let Some(lf) = self.loaded_files.get(cf) {
                if let Some(func) = lf.local_funcs.get(name).cloned() {
                    return self.call_user_func(&func, call_args, nargout, None, stdout);
                }
                if lf.primary_name == name {
                    if let Some(func) = lf.primary_func.clone() {
                        return self.call_user_func(&func, call_args, nargout, None, stdout);
                    }
                }
            }
        }

        // 3. Private function in `<caller_dir>/private/<name>.m`
        if let Some(cd) = caller_dir.as_deref() {
            if let Some(pm) = self.loader.resolve_private(name, cd) {
                if let Some(lf) = self.load_file(&pm.path)? {
                    if let Some(func) = lf.primary_func {
                        return self.call_user_func(&func, call_args, nargout, None, stdout);
                    }
                }
            }
        }

        // 4. Inline workspace functions / classes
        if let Some(func) = self.functions.get(name).cloned() {
            return self.call_user_func(&func, call_args, nargout, None, stdout);
        }
        if let Some(cls) = self.classes.get(name).cloned() {
            return Ok(vec![self.instantiate_class(&cls, call_args, stdout)?]);
        }

        // 5. External `.m` file (sibling dir -> cwd -> search_paths -> @ClassName -> imported +package)
        let resolved_lf =
            if let Some(m) = self.loader.resolve_in_context(name, caller_dir.as_deref()) {
                self.load_file(&m.path)?
            } else {
                self.resolve_imported_file(name)?
            };
        if let Some(lf) = resolved_lf {
            if let Some(func) = lf.primary_func {
                return self.call_user_func(&func, call_args, nargout, None, stdout);
            }
            if let Some(cls) = lf.primary_class {
                return Ok(vec![self.instantiate_class(&cls, call_args, stdout)?]);
            }
            if lf.is_script && call_args.is_empty() {
                self.exec_script_file(&lf, frame.as_deref_mut(), stdout)?;
                return Ok(vec![]);
            }
        }

        // 6. Built-in intrinsics
        if let Some(res) = call_intrinsic(
            name,
            call_args,
            nargout,
            &mut IntrinsicContext {
                stdout,
                figure: &mut self.figure,
                rng: &mut self.rng,
                loader: &mut self.loader,
            },
        ) {
            res
        } else {
            Err(MatroError::undefined(name))
        }
    }

    fn invoke_callable_val(
        &mut self,
        callable: &Value,
        args: &[Value],
        stdout: &mut String,
    ) -> Result<Value> {
        let rets = match callable {
            Value::Closure(cl) => self.call_closure(cl, args, 1, stdout)?,
            Value::String(fname) => self.invoke_named_callable(fname, args, 1, None, stdout)?,
            other => {
                return Err(MatroError::runtime(format!(
                    "Expected function handle or function name, got {}",
                    other.type_name()
                )));
            }
        };
        Ok(rets.into_iter().next().unwrap_or_else(Value::empty))
    }

    fn new_input_parser(&mut self) -> Value {
        let mut fields = BTreeMap::new();
        fields.insert("__required".to_string(), Value::Cell(CellArray::empty()));
        fields.insert("__optional".to_string(), Value::Cell(CellArray::empty()));
        fields.insert("__params".to_string(), Value::Struct(BTreeMap::new()));
        fields.insert("Results".to_string(), Value::Struct(BTreeMap::new()));
        let id = self.handles.len();
        self.handles.push(ObjectInstance {
            class_name: "inputParser".to_string(),
            fields,
        });
        Value::Handle(HandleRef {
            id,
            class_name: "inputParser".to_string(),
        })
    }

    fn call_input_parser_method(
        &mut self,
        handle_id: usize,
        method: &str,
        args: &[Value],
    ) -> Result<Vec<Value>> {
        let inst = self
            .handles
            .get_mut(handle_id)
            .ok_or_else(|| MatroError::runtime("Invalid inputParser handle"))?;
        match method {
            "addRequired" => {
                if args.is_empty() {
                    return Err(MatroError::arity("addRequired", ">= 1 (name)", 0));
                }
                let name = args[0].as_str()?.to_string();
                if let Some(Value::Cell(c)) = inst.fields.get_mut("__required") {
                    c.data.push(Value::String(name));
                    c.rows = 1;
                    c.cols = c.data.len();
                }
                Ok(vec![])
            }
            "addOptional" => {
                if args.len() < 2 {
                    return Err(MatroError::arity("addOptional", ">= 2 (name, default)", args.len()));
                }
                let name = args[0].as_str()?.to_string();
                let def = args[1].clone();
                if let Some(Value::Cell(c)) = inst.fields.get_mut("__optional") {
                    let mut entry = BTreeMap::new();
                    entry.insert("name".to_string(), Value::String(name));
                    entry.insert("default".to_string(), def);
                    c.data.push(Value::Struct(entry));
                    c.rows = 1;
                    c.cols = c.data.len();
                }
                Ok(vec![])
            }
            "addParameter" | "addParamValue" => {
                if args.len() < 2 {
                    return Err(MatroError::arity(method, ">= 2 (name, default)", args.len()));
                }
                let name = args[0].as_str()?.to_string();
                let def = args[1].clone();
                if let Some(Value::Struct(pmap)) = inst.fields.get_mut("__params") {
                    pmap.insert(name, def);
                }
                Ok(vec![])
            }
            "parse" => {
                let req_names: Vec<String> = match inst.fields.get("__required") {
                    Some(Value::Cell(c)) => c
                        .data
                        .iter()
                        .filter_map(|v| v.as_str().ok().map(ToString::to_string))
                        .collect(),
                    _ => Vec::new(),
                };
                let opt_defs: Vec<(String, Value)> = match inst.fields.get("__optional") {
                    Some(Value::Cell(c)) => c
                        .data
                        .iter()
                        .filter_map(|v| {
                            if let Value::Struct(m) = v {
                                let n = m.get("name")?.as_str().ok()?.to_string();
                                let d = m.get("default")?.clone();
                                Some((n, d))
                            } else {
                                None
                            }
                        })
                        .collect(),
                    _ => Vec::new(),
                };
                let param_defs: BTreeMap<String, Value> = match inst.fields.get("__params") {
                    Some(Value::Struct(m)) => m.clone(),
                    _ => BTreeMap::new(),
                };

                if args.len() < req_names.len() {
                    return Err(MatroError::runtime(format!(
                        "inputParser.parse: expected at least {} required arguments, got {}",
                        req_names.len(),
                        args.len()
                    )));
                }
                let mut results = BTreeMap::new();
                let mut idx = 0usize;
                for rname in &req_names {
                    results.insert(rname.clone(), args[idx].clone());
                    idx += 1;
                }
                for (oname, odef) in &opt_defs {
                    let is_param_key = args
                        .get(idx)
                        .and_then(|v| v.as_str().ok())
                        .is_some_and(|s| param_defs.contains_key(s));
                    if idx < args.len() && !is_param_key {
                        results.insert(oname.clone(), args[idx].clone());
                        idx += 1;
                    } else {
                        results.insert(oname.clone(), odef.clone());
                    }
                }
                for (pname, pdef) in &param_defs {
                    results.insert(pname.clone(), pdef.clone());
                }
                while idx + 1 < args.len() {
                    let key = args[idx].as_str()?;
                    let val = args[idx + 1].clone();
                    if let Some(canon) = param_defs
                        .keys()
                        .find(|k| k.eq_ignore_ascii_case(key))
                        .cloned()
                    {
                        results.insert(canon, val);
                    } else {
                        return Err(MatroError::runtime(format!(
                            "inputParser.parse: unrecognized parameter '{key}'"
                        )));
                    }
                    idx += 2;
                }
                inst.fields.insert("Results".to_string(), Value::Struct(results));
                Ok(vec![])
            }
            _ => Err(MatroError::runtime(format!(
                "No method '{method}' on inputParser"
            ))),
        }
    }

    pub(crate) fn call_closure(
        &mut self,
        cl: &ClosureHandle,
        args: &[Value],
        nargout: usize,
        stdout: &mut String,
    ) -> Result<Vec<Value>> {
        let proto = self
            .closures
            .get(cl.id)
            .cloned()
            .ok_or_else(|| MatroError::runtime("Invalid closure handle"))?;
        if let Some(target_name) = proto.name.strip_prefix('@') {
            if let Some(Value::Closure(inner)) = cl.captures.get(target_name) {
                return self.call_closure(inner, args, nargout, stdout);
            }
            if let Some(eid) = cl.env_id {
                if let Some(Value::Closure(inner)) = self.envs.get(eid).and_then(|e| e.get(target_name)).cloned() {
                    return self.call_closure(&inner, args, nargout, stdout);
                }
            }
            return self.invoke_named_callable_in_file(
                target_name,
                args,
                nargout,
                None,
                proto.source_file.as_deref(),
                stdout,
            );
        }
        let env_id = cl.env_id.or_else(|| {
            if proto.has_nested {
                let eid = self.envs.len();
                self.envs.push(BTreeMap::new());
                Some(eid)
            } else {
                None
            }
        });
        let frame = LocalFrame::new(
            &proto.name,
            proto.source_file.as_deref(),
            &proto.local_names,
            Some(&cl.captures),
            env_id,
        );
        self.exec_user_func_with_frame(&proto, args, nargout, frame, stdout)
    }

    fn call_user_func(
        &mut self,
        proto: &FunctionProto,
        args: &[Value],
        nargout: usize,
        captures: Option<&BTreeMap<String, Value>>,
        stdout: &mut String,
    ) -> Result<Vec<Value>> {
        let env_id = if proto.has_nested {
            let eid = self.envs.len();
            self.envs.push(BTreeMap::new());
            Some(eid)
        } else {
            None
        };
        let frame = LocalFrame::new(
            &proto.name,
            proto.source_file.as_deref(),
            &proto.local_names,
            captures,
            env_id,
        );
        self.exec_user_func_with_frame(proto, args, nargout, frame, stdout)
    }

    fn exec_user_func_with_frame(
        &mut self,
        proto: &FunctionProto,
        args: &[Value],
        nargout: usize,
        mut frame: LocalFrame<'_>,
        stdout: &mut String,
    ) -> Result<Vec<Value>> {
        if let Some(s) = proto.nargin_slot {
            frame.set(s, Value::scalar(args.len() as f64));
        }
        if let Some(s) = proto.nargout_slot {
            frame.set(s, Value::scalar(nargout as f64));
        }

        if !proto.arg_specs.is_empty() {
            oop::bind_arguments_block(self, proto, args, &mut frame, stdout)?;
        } else {
            let has_varargin = proto.varargin_slot.is_some();
            let fixed_params = if has_varargin {
                proto.params.len() - 1
            } else {
                proto.params.len()
            };

            if !has_varargin && args.len() > fixed_params {
                return Err(MatroError::arity(
                    &proto.name,
                    format!("at most {}", proto.params.len()),
                    args.len(),
                ));
            }

            for (&slot, v) in proto.param_slots.iter().take(fixed_params).zip(args.iter()) {
                frame.set(slot, v.clone());
            }
            if let Some(vslot) = proto.varargin_slot {
                let extra: Vec<Value> = args.iter().skip(fixed_params).cloned().collect();
                let cell = if extra.is_empty() {
                    CellArray::empty()
                } else {
                    CellArray::from_rows(vec![extra])?
                };
                frame.set(vslot, Value::Cell(cell));
            }
        }
        if let Some(voslot) = proto.varargout_slot {
            frame.set(voslot, Value::Cell(CellArray::empty()));
        }

        if proto.has_nested {
            frame.sync_to_env(&mut self.envs);
        }

        let start_steps = self.steps;
        let mut ignored_last = None;
        let res = self.run_frame(&proto.chunk, Some(&mut frame), stdout, &mut ignored_last);
        if proto.has_nested {
            frame.sync_to_env(&mut self.envs);
        }
        if self.profiling {
            let delta = self.steps.saturating_sub(start_steps);
            let entry = self.profile.entry(proto.name.clone()).or_default();
            entry.calls += 1;
            entry.instructions += delta;
        }
        res?;

        if proto.return_slots.is_empty() {
            return Ok(vec![]);
        }

        // Unpack fixed return slots + optional trailing `varargout` cell array
        let wanted = nargout.max(1);
        let mut out = Vec::with_capacity(wanted);
        for (&rslot, rname) in proto.return_slots.iter().zip(proto.returns.iter()) {
            if rname == "varargout" {
                if let Some(Value::Cell(cell)) = frame.get(rslot) {
                    for item in &cell.data {
                        if out.len() >= wanted {
                            break;
                        }
                        out.push(item.clone());
                    }
                }
            } else if out.len() < wanted {
                let v = frame.get(rslot).cloned().unwrap_or_else(Value::empty);
                out.push(v);
            }
        }
        Ok(out)
    }

    #[inline(never)]
    fn exec_arrayfun(&mut self, call_args: &[Value], stdout: &mut String) -> Result<Vec<Value>> {
        if call_args.len() < 2 {
            return Err(MatroError::arity("arrayfun", ">= 2 (fun, A, ...)", call_args.len()));
        }
        let mut uniform = true;
        let mut end = call_args.len();
        while end >= 4 {
            if let Ok(k) = call_args[end - 2].as_str() {
                if k.eq_ignore_ascii_case("UniformOutput") {
                    uniform = call_args[end - 1].is_truthy();
                    end -= 2;
                    continue;
                } else if k.eq_ignore_ascii_case("ErrorHandler") {
                    end -= 2;
                    continue;
                }
            }
            break;
        }
        let arr_vals = &call_args[1..end];
        if arr_vals.is_empty() {
            return Err(MatroError::arity("arrayfun", ">= 2 (fun, A, ...)", call_args.len()));
        }
        let mats: Vec<Matrix> = arr_vals
            .iter()
            .map(|v| v.to_dense_matrix())
            .collect::<Result<_>>()?;
        let dims = mats[0].dims().to_vec();
        let total = mats[0].len();
        for m in &mats[1..] {
            if m.dims() != dims.as_slice() {
                return Err(MatroError::dimension(
                    "arrayfun: all input arrays must have the same size",
                ));
            }
        }
        let mut results = Vec::with_capacity(total);
        for idx in 0..total {
            let mut elem_args = Vec::with_capacity(mats.len());
            for m in &mats {
                elem_args.push(Value::matrix(
                    Matrix::complex_scalar(m.get_linear_c(idx)).normalize(),
                ));
            }
            let r = self.invoke_callable_val(&call_args[0], &elem_args, stdout)?;
            results.push(r);
        }
        if uniform {
            let mut out = Matrix::zeros_nd(dims);
            let mut all_logical = !results.is_empty();
            for (idx, v) in results.iter().enumerate() {
                let rm = v.to_dense_matrix()?;
                if rm.dtype() != matro_types::NumericClass::Logical {
                    all_logical = false;
                }
                let sc = rm.as_complex_scalar().ok_or_else(|| {
                    MatroError::dimension(
                        "arrayfun: non-scalar output; pass 'UniformOutput', false",
                    )
                })?;
                out.set_linear_c(idx, sc);
            }
            let mut out = out.normalize();
            if all_logical {
                out = out.into_logical();
            }
            Ok(vec![Value::matrix(out)])
        } else {
            let rows = dims.first().copied().unwrap_or(0);
            let cols = if rows == 0 { 0 } else { total / rows };
            Ok(vec![Value::Cell(CellArray {
                rows,
                cols,
                data: results,
                extra_dims: Vec::new(),
            })])
        }
    }

    #[inline(never)]
    fn exec_cellfun(&mut self, call_args: &[Value], stdout: &mut String) -> Result<Vec<Value>> {
        if call_args.len() < 2 {
            return Err(MatroError::arity("cellfun", ">= 2 (fn, C, ...)", call_args.len()));
        }
        let mut uniform = true;
        let mut end = call_args.len();
        while end >= 4 {
            if let Ok(k) = call_args[end - 2].as_str() {
                if k.eq_ignore_ascii_case("UniformOutput") {
                    uniform = call_args[end - 1].is_truthy();
                    end -= 2;
                    continue;
                } else if k.eq_ignore_ascii_case("ErrorHandler") {
                    end -= 2;
                    continue;
                }
            }
            break;
        }
        let cell_vals = &call_args[1..end];
        let mut cells: Vec<&CellArray> = Vec::with_capacity(cell_vals.len());
        for v in cell_vals {
            let Value::Cell(c) = v else {
                return Err(MatroError::runtime("cellfun: input arguments must be cell arrays"));
            };
            cells.push(c);
        }
        let rows = cells[0].rows;
        let cols = cells[0].cols;
        let total = cells[0].data.len();
        for c in &cells[1..] {
            if c.rows != rows || c.cols != cols {
                return Err(MatroError::dimension(
                    "cellfun: all input cell arrays must have the same size",
                ));
            }
        }
        let shortcut = call_args[0].as_str().ok().map(|s| s.to_ascii_lowercase());
        let mut results = Vec::with_capacity(total);
        for idx in 0..total {
            let elem_args: Vec<Value> = cells.iter().map(|c| c.data[idx].clone()).collect();
            let v = match shortcut.as_deref() {
                Some("prodofsize") => self.invoke_callable_val(
                    &Value::String("numel".to_string()),
                    &elem_args,
                    stdout,
                )?,
                _ => self.invoke_callable_val(&call_args[0], &elem_args, stdout)?,
            };
            results.push(v);
        }
        if uniform {
            let mut out = Matrix::zeros(rows, cols);
            let mut all_logical = !results.is_empty();
            for (idx, v) in results.iter().enumerate() {
                let rm = v.to_dense_matrix()?;
                if rm.dtype() != matro_types::NumericClass::Logical {
                    all_logical = false;
                }
                let sc = rm.as_complex_scalar().ok_or_else(|| {
                    MatroError::runtime(
                        "cellfun: non-scalar output; pass 'UniformOutput', false",
                    )
                })?;
                out.set_linear_c(idx, sc);
            }
            let mut out = out.normalize();
            if all_logical {
                out = out.into_logical();
            }
            Ok(vec![Value::matrix(out)])
        } else {
            Ok(vec![Value::Cell(CellArray {
                rows,
                cols,
                data: results,
                extra_dims: Vec::new(),
            })])
        }
    }

    #[inline(never)]
    fn exec_bsxfun(&mut self, call_args: &[Value], stdout: &mut String) -> Result<Vec<Value>> {
        if call_args.len() != 3 {
            return Err(MatroError::arity("bsxfun", "3 (fn, A, B)", call_args.len()));
        }
        let a = call_args[1].to_dense_matrix()?;
        let b = call_args[2].to_dense_matrix()?;
        let a_dims = a.dims();
        let b_dims = b.dims();
        let nd = a_dims.len().max(b_dims.len());
        let mut out_dims = Vec::with_capacity(nd);
        for d in 0..nd {
            let da = a_dims.get(d).copied().unwrap_or(1);
            let db = b_dims.get(d).copied().unwrap_or(1);
            if da == db {
                out_dims.push(da);
            } else if da == 1 {
                out_dims.push(db);
            } else if db == 1 {
                out_dims.push(da);
            } else {
                return Err(MatroError::dimension(
                    "bsxfun: incompatible array dimensions for singleton expansion",
                ));
            }
        }
        let total: usize = out_dims.iter().product();
        let mut out = Matrix::zeros_nd(out_dims.clone());
        let mut all_logical = total > 0;
        let mut strides_a = vec![1usize; nd];
        let mut strides_b = vec![1usize; nd];
        for d in 1..nd {
            strides_a[d] = strides_a[d - 1] * a_dims.get(d - 1).copied().unwrap_or(1);
            strides_b[d] = strides_b[d - 1] * b_dims.get(d - 1).copied().unwrap_or(1);
        }
        for lin in 0..total {
            let mut rem = lin;
            let mut idx_a = 0usize;
            let mut idx_b = 0usize;
            for d in 0..nd {
                let coord = rem % out_dims[d];
                rem /= out_dims[d];
                if a_dims.get(d).copied().unwrap_or(1) > 1 {
                    idx_a += coord * strides_a[d];
                }
                if b_dims.get(d).copied().unwrap_or(1) > 1 {
                    idx_b += coord * strides_b[d];
                }
            }
            let va = Value::matrix(Matrix::complex_scalar(a.get_linear_c(idx_a)).normalize());
            let vb = Value::matrix(Matrix::complex_scalar(b.get_linear_c(idx_b)).normalize());
            let res = self.invoke_callable_val(&call_args[0], &[va, vb], stdout)?;
            let rm = res.to_dense_matrix()?;
            if rm.dtype() != matro_types::NumericClass::Logical {
                all_logical = false;
            }
            let sc = rm.as_complex_scalar().ok_or_else(|| {
                MatroError::dimension("bsxfun: callback must return scalar")
            })?;
            out.set_linear_c(lin, sc);
        }
        let mut out = out.normalize();
        if all_logical {
            out = out.into_logical();
        }
        Ok(vec![Value::matrix(out)])
    }

    #[inline(never)]
    fn exec_get_field_by_name(
        &mut self,
        obj: Value,
        field: String,
        cur_fn: Option<&str>,
        stdout: &mut String,
    ) -> Result<Value> {
        match obj {
            Value::Struct(map) => map.get(&field).cloned(),
            Value::Object(ref inst) if inst.class_name == "__class_ref" => {
                let cname = inst
                    .fields
                    .get("__name")
                    .and_then(|v| v.as_str().ok())
                    .unwrap_or("")
                    .to_string();
                self.resolve_class_ref_field(&cname, &field, stdout)?
            }
            Value::Object(ref inst) if inst.class_name == "__pkg_ref" => {
                let pkg_path = inst
                    .fields
                    .get("__path")
                    .and_then(|v| v.as_str().ok())
                    .unwrap_or("")
                    .to_string();
                self.resolve_pkg_field(&pkg_path, &field, stdout)?
            }
            Value::Object(inst) => {
                let getter = format!("get.{field}");
                if cur_fn != Some(getter.as_str()) {
                    if let Some(proto) = self.find_class_method(&inst.class_name, &getter) {
                        let rets = self.call_user_func(
                            &proto,
                            &[Value::Object(inst)],
                            1,
                            None,
                            stdout,
                        )?;
                        Some(rets.into_iter().next().unwrap_or_else(Value::empty))
                    } else {
                        inst.fields.get(&field).cloned()
                    }
                } else {
                    inst.fields.get(&field).cloned()
                }
            }
            Value::Handle(h) => {
                let getter = format!("get.{field}");
                if cur_fn != Some(getter.as_str()) {
                    if let Some(proto) = self.find_class_method(&h.class_name, &getter) {
                        let rets = self.call_user_func(
                            &proto,
                            &[Value::Handle(h)],
                            1,
                            None,
                            stdout,
                        )?;
                        Some(rets.into_iter().next().unwrap_or_else(Value::empty))
                    } else {
                        self.handles
                            .get(h.id)
                            .and_then(|inst| inst.fields.get(&field).cloned())
                    }
                } else {
                    self.handles
                        .get(h.id)
                        .and_then(|inst| inst.fields.get(&field).cloned())
                }
            }
            Value::StructArray(sa) => {
                let items: Vec<Value> = sa
                    .items
                    .iter()
                    .map(|m| m.get(&field).cloned().unwrap_or_else(Value::empty))
                    .collect();
                if items.iter().all(|v| v.as_scalar().is_ok()) {
                    let nums: Vec<f64> = items.iter().map(|v| v.as_scalar().unwrap()).collect();
                    Some(Value::matrix(Matrix::row_vec(nums)))
                } else {
                    Some(Value::Cell(CellArray::from_rows(vec![items])?))
                }
            }
            Value::Table(t) => {
                if field == "Properties" {
                    let mut props = BTreeMap::new();
                    let vnames: Vec<Value> = t
                        .col_names
                        .iter()
                        .cloned()
                        .map(Value::String)
                        .collect();
                    let cell = if vnames.is_empty() {
                        CellArray::empty()
                    } else {
                        CellArray::from_rows(vec![vnames])?
                    };
                    props.insert("VariableNames".to_string(), Value::Cell(cell));
                    Some(Value::Struct(props))
                } else {
                    t.get_col(&field).cloned()
                }
            }
            Value::Map(m) => match field.as_str() {
                "Count" => Some(Value::scalar(m.entries.len() as f64)),
                _ => None,
            },
            other => {
                return Err(MatroError::runtime(format!(
                    "Dot indexing requires a struct or object, got {}",
                    other.type_name()
                )));
            }
        }
        .ok_or_else(|| MatroError::runtime(format!("No field or property '{field}'")))
    }

    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn exec_store_chain(
        &mut self,
        var_name: Option<&str>,
        local_slot: Option<u16>,
        spec_idx: u16,
        chunk: &Chunk,
        stack: &mut Vec<StackItem>,
        mut frame: Option<&mut LocalFrame>,
        last_value: &mut Option<Value>,
    ) -> Result<()> {
        let spec = &chunk.chain_specs[spec_idx as usize];
        // Pop index args in reverse order of steps
        let mut step_args: Vec<Vec<StackItem>> = vec![Vec::new(); spec.len()];
        for (i, step) in spec.iter().enumerate().rev() {
            match *step {
                ChainStep::Field(_) => {}
                ChainStep::DynamicField => {
                    step_args[i] = pop_items(stack, 1)?;
                }
                ChainStep::Paren(argc) | ChainStep::Brace(argc) => {
                    step_args[i] = pop_items(stack, argc as usize)?;
                }
            }
        }
        let rhs = pop_val(stack)?;

        let mut root = if let Some(slot) = local_slot {
            frame
                .as_deref_mut()
                .ok_or_else(|| MatroError::runtime("StoreChainLocal outside function"))?
                .take(slot)
                .unwrap_or_else(Value::empty)
        } else if let Some(name) = var_name {
            lookup_var(name, frame.as_deref(), &self.globals)
                .cloned()
                .unwrap_or_else(Value::empty)
        } else {
            Value::empty()
        };

        assign_chain_step(&mut root, spec, &step_args, &chunk.names, rhs, &mut self.handles)?;
        *last_value = Some(root.clone());

        if let Some(slot) = local_slot {
            if let Some(f) = frame {
                f.set(slot, root);
            }
        } else if let Some(name) = var_name {
            store_var(name.to_string(), root, frame, &mut self.globals);
        }
        Ok(())
    }

    #[inline(never)]
    fn exec_spfun(&mut self, call_args: &[Value], stdout: &mut String) -> Result<Vec<Value>> {
        if call_args.len() != 2 {
            return Err(MatroError::arity("spfun", "2 (fun, S)", call_args.len()));
        }
        let sp = match &call_args[1] {
            Value::Sparse(s) => (**s).clone(),
            other => SparseMatrix::from_dense(&other.to_dense_matrix()?),
        };
        let triplets = sp.triplets_c();
        if triplets.is_empty() {
            return Ok(vec![Value::sparse(SparseMatrix::zeros(sp.rows(), sp.cols()))]);
        }
        let nz_col = if sp.is_complex() {
            let nz_vals: Vec<Complex> = triplets.iter().map(|&(_, _, v)| v).collect();
            Value::matrix(Matrix::complex_col_vec(nz_vals).normalize())
        } else {
            let nz_vals: Vec<f64> = triplets.iter().map(|&(_, _, v)| v.re).collect();
            Value::matrix(Matrix::col_vec(nz_vals))
        };
        let mapped = self.invoke_callable_val(&call_args[0], &[nz_col], stdout)?;
        let mapped_mat = mapped.to_dense_matrix()?;
        if mapped_mat.len() != triplets.len() {
            return Err(MatroError::dimension(format!(
                "spfun: function returned {} elements, expected {}",
                mapped_mat.len(),
                triplets.len()
            )));
        }
        let out = if mapped_mat.is_complex() {
            let new_triplets: Vec<(usize, usize, Complex)> = triplets
                .iter()
                .enumerate()
                .filter_map(|(k, &(r, c, _))| {
                    let v = mapped_mat.get_linear_c(k);
                    if !v.is_zero() { Some((r, c, v)) } else { None }
                })
                .collect();
            SparseMatrix::from_complex_triplets(sp.rows(), sp.cols(), &new_triplets)?
        } else {
            let new_triplets: Vec<(usize, usize, f64)> = triplets
                .iter()
                .zip(mapped_mat.data().iter())
                .filter_map(|(&(r, c, _), &v)| if v != 0.0 { Some((r, c, v)) } else { None })
                .collect();
            SparseMatrix::from_triplets(sp.rows(), sp.cols(), &new_triplets)?
        };
        Ok(vec![Value::sparse(out)])
    }

    #[inline(never)]
    fn exec_mexception(&mut self, call_args: &[Value]) -> Result<Vec<Value>> {
        if call_args.is_empty() {
            return Ok(vec![make_mexception("", "", 0)]);
        }
        let id = call_args[0].as_str()?.to_string();
        let msg = if call_args.len() == 1 {
            String::new()
        } else if call_args.len() == 2 {
            call_args[1].as_str()?.to_string()
        } else {
            let mut tmp_out = String::new();
            let mut ctx = matro_intrinsics::IntrinsicContext {
                stdout: &mut tmp_out,
                figure: &mut self.figure,
                rng: &mut self.rng,
                loader: &mut self.loader,
            };
            let formatted = matro_intrinsics::call_intrinsic("sprintf", &call_args[1..], 1, &mut ctx)
                .transpose()?
                .and_then(|v| v.into_iter().next())
                .and_then(|v| v.as_str().ok().map(String::from))
                .unwrap_or_default();
            formatted
        };
        Ok(vec![make_mexception(&id, &msg, 0)])
    }

    #[inline(never)]
    fn exec_lasterror(&mut self, call_args: &[Value]) -> Result<Vec<Value>> {
        if let Some(arg) = call_args.first() {
            if let Ok(s) = arg.as_str() {
                if s.eq_ignore_ascii_case("reset") {
                    self.last_error = None;
                }
            } else if let Value::Struct(map) | Value::Object(ObjectInstance { fields: map, .. }) = arg {
                let id = map
                    .get("identifier")
                    .and_then(|v| v.as_str().ok())
                    .unwrap_or("")
                    .to_string();
                let msg = map
                    .get("message")
                    .and_then(|v| v.as_str().ok())
                    .unwrap_or("")
                    .to_string();
                if id.is_empty() && msg.is_empty() {
                    self.last_error = None;
                } else {
                    self.last_error = Some((id, msg, 0));
                }
            }
        }
        let mut out = BTreeMap::new();
        if let Some((id, msg, line)) = &self.last_error {
            out.insert("message".to_string(), Value::String(msg.clone()));
            out.insert("identifier".to_string(), Value::String(id.clone()));
            out.insert("line".to_string(), Value::scalar(*line as f64));
        } else {
            out.insert("message".to_string(), Value::String(String::new()));
            out.insert("identifier".to_string(), Value::String(String::new()));
            out.insert("line".to_string(), Value::scalar(0.0));
        }
        out.insert(
            "stack".to_string(),
            Value::StructArray(matro_types::StructArray {
                rows: 0,
                cols: 1,
                items: Vec::new(),
            }),
        );
        Ok(vec![Value::Struct(out)])
    }

    #[inline(never)]
    fn exec_functions(&self, call_args: &[Value]) -> Result<Vec<Value>> {
        if call_args.len() != 1 {
            return Err(MatroError::arity("functions", "1 (fh)", call_args.len()));
        }
        let Value::Closure(cl) = &call_args[0] else {
            return Err(MatroError::runtime("functions: expected a function handle"));
        };
        let proto = self
            .closures
            .get(cl.id)
            .ok_or_else(|| MatroError::runtime("functions: invalid closure handle"))?;
        let mut info = BTreeMap::new();
        if let Some(fname) = proto.name.strip_prefix('@') {
            info.insert("function".to_string(), Value::String(fname.to_string()));
            info.insert("type".to_string(), Value::String("simple".to_string()));
            info.insert(
                "file".to_string(),
                Value::String(proto.source_file.clone().unwrap_or_default()),
            );
        } else if proto.name == "<anonymous>" {
            let desc = format!("@({})<anonymous>", cl.params.join(","));
            info.insert("function".to_string(), Value::String(desc));
            info.insert("type".to_string(), Value::String("anonymous".to_string()));
            info.insert(
                "file".to_string(),
                Value::String(proto.source_file.clone().unwrap_or_default()),
            );
            let ws = Value::Struct(cl.captures.clone());
            info.insert(
                "workspace".to_string(),
                Value::Cell(CellArray {
                    rows: 1,
                    cols: 1,
                    data: vec![ws],
                    extra_dims: Vec::new(),
                }),
            );
        } else {
            info.insert("function".to_string(), Value::String(proto.name.clone()));
            info.insert("type".to_string(), Value::String("nested".to_string()));
            info.insert(
                "file".to_string(),
                Value::String(proto.source_file.clone().unwrap_or_default()),
            );
        }
        Ok(vec![Value::Struct(info)])
    }

    #[inline(never)]
    fn exec_func2str(&self, call_args: &[Value]) -> Result<Vec<Value>> {
        if call_args.len() != 1 {
            return Err(MatroError::arity("func2str", "1 (fh)", call_args.len()));
        }
        let Value::Closure(cl) = &call_args[0] else {
            return Err(MatroError::runtime("func2str: expected a function handle"));
        };
        let proto = self
            .closures
            .get(cl.id)
            .ok_or_else(|| MatroError::runtime("func2str: invalid closure handle"))?;
        let name = if let Some(fname) = proto.name.strip_prefix('@') {
            fname.to_string()
        } else if proto.name == "<anonymous>" {
            format!("@({})<anonymous>", cl.params.join(","))
        } else {
            proto.name.clone()
        };
        Ok(vec![Value::String(name)])
    }

    #[inline(never)]
    fn exec_str2func(&mut self, call_args: &[Value], stdout: &mut String) -> Result<Vec<Value>> {
        if call_args.len() != 1 {
            return Err(MatroError::arity("str2func", "1 (str)", call_args.len()));
        }
        let raw = call_args[0].as_str()?.trim();
        let src = if raw.starts_with('@') {
            format!("__str2func_res = {raw};")
        } else {
            format!("__str2func_res = @{raw};")
        };
        let chunk = compile(&src)?;
        let prev = self.globals.remove("__str2func_res");
        let mut last = None;
        let run_res = self.run_frame(&chunk, None, stdout, &mut last);
        let out = self.globals.remove("__str2func_res");
        if let Some(p) = prev {
            self.globals.insert("__str2func_res".to_string(), p);
        }
        run_res?;
        let fh = out.ok_or_else(|| MatroError::runtime("str2func: failed to create handle"))?;
        Ok(vec![fh])
    }

    #[inline(never)]
    fn exec_accumarray_closure(
        &mut self,
        call_args: &[Value],
        stdout: &mut String,
    ) -> Result<Vec<Value>> {
        let (n_entries, n_dims, coord_cols): (usize, usize, Vec<Vec<usize>>) = match &call_args[0]
        {
            Value::Cell(c) => {
                let mut cols = Vec::with_capacity(c.data.len());
                for item in &c.data {
                    let m = item.to_dense_matrix()?;
                    let mut col = Vec::with_capacity(m.len());
                    for &v in m.data() {
                        if v < 1.0 || !v.is_finite() {
                            return Err(MatroError::index("accumarray: subscripts must be positive integers"));
                        }
                        col.push(v.round() as usize - 1);
                    }
                    cols.push(col);
                }
                let n = cols.first().map_or(0, |v| v.len());
                (n, cols.len().max(1), cols)
            }
            other => {
                let s = other.to_dense_matrix()?;
                if s.rows() == 1 && s.cols() > 1 {
                    let mut col = Vec::with_capacity(s.len());
                    for &v in s.data() {
                        if v < 1.0 || !v.is_finite() {
                            return Err(MatroError::index("accumarray: subscripts must be positive integers"));
                        }
                        col.push(v.round() as usize - 1);
                    }
                    (col.len(), 1, vec![col])
                } else {
                    let nr = s.rows();
                    let nc = s.cols().max(1);
                    let mut cols = vec![Vec::with_capacity(nr); nc];
                    for c in 0..nc {
                        for r in 0..nr {
                            let v = s.get(r, c);
                            if v < 1.0 || !v.is_finite() {
                                return Err(MatroError::index("accumarray: subscripts must be positive integers"));
                            }
                            cols[c].push(v.round() as usize - 1);
                        }
                    }
                    (nr, nc, cols)
                }
            }
        };

        let val_m = call_args[1].to_dense_matrix()?;
        let val_scalar = val_m.len() == 1;
        if !val_scalar && val_m.len() != n_entries {
            return Err(MatroError::dimension(
                "accumarray: val length must match subs row count or be scalar",
            ));
        }

        let mut dims: Vec<usize> = if call_args.len() >= 3 && !matches!(&call_args[2], Value::Matrix(m) if m.is_empty()) {
            let sm = call_args[2].to_dense_matrix()?;
            sm.data().iter().map(|&v| v.max(0.0).round() as usize).collect()
        } else {
            let mut d: Vec<usize> = coord_cols
                .iter()
                .map(|c| c.iter().copied().max().map_or(0, |m| m + 1))
                .collect();
            if d.len() == 1 {
                d.push(1);
            }
            d
        };
        if dims.len() == 1 {
            dims.push(1);
        }
        let fill = if call_args.len() >= 5 && !matches!(&call_args[4], Value::Matrix(m) if m.is_empty()) {
            call_args[4].as_scalar()?
        } else {
            0.0
        };

        let total: usize = dims.iter().product();
        let mut strides = vec![1usize; dims.len()];
        for d in 1..dims.len() {
            strides[d] = strides[d - 1] * dims[d - 1];
        }
        let mut buckets: Vec<Vec<Complex>> = vec![Vec::new(); total];
        for i in 0..n_entries {
            let mut lin = 0usize;
            for d in 0..n_dims {
                let coord = coord_cols[d][i];
                if coord >= dims[d] {
                    return Err(MatroError::index("accumarray: subscript exceeds output size"));
                }
                lin += coord * strides[d];
            }
            let v = if val_scalar {
                val_m.get_linear_c(0)
            } else {
                val_m.get_linear_c(i)
            };
            buckets[lin].push(v);
        }

        let mut out = Matrix::zeros_nd(dims);
        for (lin, grp) in buckets.into_iter().enumerate() {
            if grp.is_empty() {
                out.set_linear_c(lin, Complex::from_real(fill));
            } else {
                let arg = Value::matrix(Matrix::complex_col_vec(grp).normalize());
                let res = self.invoke_callable_val(&call_args[3], &[arg], stdout)?;
                let sc = res
                    .to_dense_matrix()?
                    .as_complex_scalar()
                    .ok_or_else(|| MatroError::dimension("accumarray: fun must return scalar"))?;
                out.set_linear_c(lin, sc);
            }
        }
        Ok(vec![Value::matrix(out.normalize())])
    }

    #[inline(never)]
    fn exec_ode_solver(
        &mut self,
        name: &str,
        call_args: &[Value],
        nargout: usize,
        stdout: &mut String,
    ) -> Result<Vec<Value>> {
        if call_args.len() < 3 || call_args.len() > 4 {
            return Err(MatroError::arity(name, "3 or 4 (f, tspan, y0 [, options])", call_args.len()));
        }
        let Value::Closure(cl) = &call_args[0] else {
            return Err(MatroError::runtime(format!(
                "{name}: first argument must be a function handle @(t, y) ..."
            )));
        };
        let tspan = call_args[1].to_dense_matrix()?;
        let y0 = call_args[2].to_dense_matrix()?;
        let opts = solvers::OdeOptions::from_val(call_args.get(3));
        let (t_mat, y_mat) = match name {
            "ode45" => solvers::solve_ode45(self, cl, &tspan, &y0, &opts, stdout)?,
            "ode23" => solvers_ext::solve_ode23(self, cl, &tspan, &y0, stdout)?,
            "ode113" => solvers_ext::solve_ode113(self, cl, &tspan, &y0, stdout)?,
            "ode15s" | "ode23s" | "ode23t" | "ode23tb" => {
                solvers::solve_ode15s(self, cl, &tspan, &y0, stdout)?
            }
            _ => unreachable!(),
        };
        if nargout <= 1 {
            let mut sol = BTreeMap::new();
            sol.insert("solver".to_string(), Value::String(name.to_string()));
            sol.insert("x".to_string(), Value::matrix(t_mat.transpose()));
            sol.insert("y".to_string(), Value::matrix(y_mat.transpose()));
            return Ok(vec![Value::Struct(sol)]);
        }
        Ok(vec![Value::matrix(t_mat), Value::matrix(y_mat)])
    }

    #[inline(never)]
    fn exec_fzero(
        &mut self,
        call_args: &[Value],
        nargout: usize,
        stdout: &mut String,
    ) -> Result<Vec<Value>> {
        if call_args.len() < 2 || call_args.len() > 3 {
            return Err(MatroError::arity("fzero", "2 or 3 (f, x0 [, options])", call_args.len()));
        }
        let Value::Closure(cl) = &call_args[0] else {
            return Err(MatroError::runtime(
                "fzero: first argument must be a function handle",
            ));
        };
        let guess = call_args[1].to_dense_matrix()?;
        let root = solvers::solve_fzero(self, cl, &guess, stdout)?;
        let mut out = vec![Value::scalar(root)];
        if nargout >= 2 {
            let fval = self
                .call_closure(cl, &[Value::scalar(root)], 1, stdout)?
                .into_iter()
                .next()
                .and_then(|v| v.as_scalar().ok())
                .unwrap_or(0.0);
            out.push(Value::scalar(fval));
        }
        if nargout >= 3 {
            out.push(Value::scalar(if root.is_finite() { 1.0 } else { -1.0 }));
        }
        if nargout >= 4 {
            let mut info = BTreeMap::new();
            info.insert("algorithm".to_string(), Value::String("bisection, interpolation".into()));
            out.push(Value::Struct(info));
        }
        Ok(out)
    }

    #[inline(never)]
    fn exec_integral(
        &mut self,
        name: &str,
        call_args: &[Value],
        stdout: &mut String,
    ) -> Result<Vec<Value>> {
        if call_args.len() < 3 {
            return Err(MatroError::arity(name, ">= 3 (f, a, b [, opts...])", call_args.len()));
        }
        let Value::Closure(cl) = &call_args[0] else {
            return Err(MatroError::runtime(format!(
                "{name}: first argument must be a function handle"
            )));
        };
        let a = call_args[1].as_scalar()?;
        let b = call_args[2].as_scalar()?;
        let mut rel_tol = 1e-6;
        let mut abs_tol = 1e-10;
        let mut waypoints: Vec<f64> = Vec::new();
        let mut array_valued = false;

        if call_args.len() == 4 && matches!(name, "quad" | "quadl") {
            if let Ok(t) = call_args[3].as_scalar() {
                abs_tol = t.abs().max(1e-15);
            }
        } else {
            let mut i = 3;
            while i + 1 < call_args.len() {
                if let Ok(k) = call_args[i].as_str() {
                    match k.to_ascii_lowercase().as_str() {
                        "reltol" => rel_tol = call_args[i + 1].as_scalar()?.max(1e-14),
                        "abstol" => abs_tol = call_args[i + 1].as_scalar()?.max(1e-15),
                        "waypoints" => {
                            let wm = call_args[i + 1].to_dense_matrix()?;
                            waypoints = wm.data().to_vec();
                        }
                        "arrayvalued" => array_valued = call_args[i + 1].is_truthy(),
                        _ => {}
                    }
                }
                i += 2;
            }
        }

        if array_valued {
            let probe_x = if a.is_finite() && b.is_finite() {
                0.5 * (a + b)
            } else if a.is_finite() {
                a + 1.0
            } else if b.is_finite() {
                b - 1.0
            } else {
                0.0
            };
            let probe = self
                .call_closure(cl, &[Value::scalar(probe_x)], 1, stdout)?
                .into_iter()
                .next()
                .unwrap_or_else(Value::empty)
                .to_dense_matrix()?;
            let dims = probe.dims().to_vec();
            let total = probe.len();
            // Integrate component-wise via fixed 64-panel Gauss-Kronrod or adaptive GK15
            let n_panels = 64usize;
            let mut acc = vec![Complex::ZERO; total];
            const XGK: [f64; 8] = [
                0.9914553711208126, 0.9491079123427585, 0.8648644233597691, 0.7415311855993945,
                0.5860872354676911, 0.4058451513773972, 0.2077849550078985, 0.0,
            ];
            const WGK: [f64; 8] = [
                0.02293532201052922, 0.06309209262997856, 0.1047900103222502, 0.1406532597155259,
                0.1690047266392679, 0.1903505780647854, 0.2044329400752989, 0.2094821410847278,
            ];
            let h = (b - a) / n_panels as f64;
            for p in 0..n_panels {
                let lo = a + p as f64 * h;
                let hi = lo + h;
                let mid = 0.5 * (lo + hi);
                let half = 0.5 * (hi - lo);
                let eval_pt = |vm: &mut Vm, x: f64, w: f64, acc: &mut [Complex], stdout: &mut String| -> Result<()> {
                    let m = vm
                        .call_closure(cl, &[Value::scalar(x)], 1, stdout)?
                        .into_iter()
                        .next()
                        .unwrap_or_else(Value::empty)
                        .to_dense_matrix()?;
                    for k in 0..total.min(m.len()) {
                        acc[k] = acc[k] + m.get_linear_c(k) * Complex::from_real(w);
                    }
                    Ok(())
                };
                eval_pt(self, mid, half * WGK[7], &mut acc, stdout)?;
                for j in 0..7 {
                    let dx = half * XGK[j];
                    eval_pt(self, mid - dx, half * WGK[j], &mut acc, stdout)?;
                    eval_pt(self, mid + dx, half * WGK[j], &mut acc, stdout)?;
                }
            }
            let re: Vec<f64> = acc.iter().map(|z| z.re).collect();
            let im: Vec<f64> = acc.iter().map(|z| z.im).collect();
            let out = Matrix::from_nd_complex_col_major(dims, re, im)?.normalize();
            return Ok(vec![Value::matrix(out)]);
        }

        let q = solvers::solve_integral_opts(
            self,
            cl,
            a,
            b,
            rel_tol,
            abs_tol,
            &waypoints,
            stdout,
        )?;
        Ok(vec![Value::matrix(Matrix::complex_scalar(q).normalize())])
    }

    #[inline(never)]
    fn exec_opt_solver(
        &mut self,
        name: &str,
        call_args: &[Value],
        nargout: usize,
        stdout: &mut String,
    ) -> Result<Vec<Value>> {
        let make_out_struct = |algo: &str| {
            let mut m = BTreeMap::new();
            m.insert("algorithm".to_string(), Value::String(algo.to_string()));
            Value::Struct(m)
        };
        match name {
            "fminsearch" => {
                if call_args.len() < 2 || call_args.len() > 3 {
                    return Err(MatroError::arity("fminsearch", "2 or 3 (f, x0 [, options])", call_args.len()));
                }
                let Value::Closure(cl) = &call_args[0] else {
                    return Err(MatroError::runtime("fminsearch: first argument must be a function handle"));
                };
                let x0 = call_args[1].to_dense_matrix()?;
                let (best_x, fval) = solvers::solve_fminsearch(self, cl, &x0, stdout)?;
                let mut out = vec![Value::matrix(best_x)];
                if nargout >= 2 { out.push(Value::scalar(fval)); }
                if nargout >= 3 { out.push(Value::scalar(1.0)); }
                if nargout >= 4 { out.push(make_out_struct("Nelder-Mead simplex")); }
                Ok(out)
            }
            "fminunc" => {
                if call_args.len() < 2 || call_args.len() > 3 {
                    return Err(MatroError::arity("fminunc", "2 or 3 (fun, x0 [, options])", call_args.len()));
                }
                let Value::Closure(cl) = &call_args[0] else {
                    return Err(MatroError::runtime("fminunc: first argument must be a function handle"));
                };
                let x0 = call_args[1].to_dense_matrix()?;
                let (best_x, fval) = opt_solvers::solve_fminunc(self, cl, &x0, stdout)?;
                let mut out = vec![Value::matrix(best_x)];
                if nargout >= 2 { out.push(Value::scalar(fval)); }
                if nargout >= 3 { out.push(Value::scalar(1.0)); }
                if nargout >= 4 { out.push(make_out_struct("quasi-newton")); }
                Ok(out)
            }
            "fmincon" => {
                let (best_x, fval) = opt_solvers::solve_fmincon(self, call_args, stdout)?;
                let mut out = vec![Value::matrix(best_x)];
                if nargout >= 2 { out.push(Value::scalar(fval)); }
                if nargout >= 3 { out.push(Value::scalar(1.0)); }
                if nargout >= 4 { out.push(make_out_struct("interior-point")); }
                Ok(out)
            }
            "fminbnd" => {
                if call_args.len() < 3 || call_args.len() > 4 {
                    return Err(MatroError::arity("fminbnd", "3 or 4 (fun, x1, x2 [, options])", call_args.len()));
                }
                let Value::Closure(cl) = &call_args[0] else {
                    return Err(MatroError::runtime("fminbnd: first argument must be a function handle"));
                };
                let x1 = call_args[1].as_scalar()?;
                let x2 = call_args[2].as_scalar()?;
                let (x, fval) = solvers_ext::solve_fminbnd(self, cl, x1, x2, stdout)?;
                let mut out = vec![Value::scalar(x)];
                if nargout >= 2 { out.push(Value::scalar(fval)); }
                if nargout >= 3 { out.push(Value::scalar(1.0)); }
                if nargout >= 4 { out.push(make_out_struct("golden section search, parabolic interpolation")); }
                Ok(out)
            }
            "fsolve" => {
                if call_args.len() < 2 || call_args.len() > 3 {
                    return Err(MatroError::arity("fsolve", "2 or 3 (f, x0 [, options])", call_args.len()));
                }
                let Value::Closure(cl) = &call_args[0] else {
                    return Err(MatroError::runtime("fsolve: first argument must be a function handle"));
                };
                let x0 = call_args[1].to_dense_matrix()?;
                let (sol, _resnorm, residual) = solvers::solve_lm(self, cl, &x0, stdout)?;
                let mut out = vec![Value::matrix(sol)];
                if nargout >= 2 { out.push(Value::matrix(residual)); }
                if nargout >= 3 { out.push(Value::scalar(1.0)); }
                if nargout >= 4 { out.push(make_out_struct("levenberg-marquardt")); }
                Ok(out)
            }
            "lsqnonlin" => {
                if call_args.len() < 2 || call_args.len() > 5 {
                    return Err(MatroError::arity(
                        "lsqnonlin",
                        "2 to 5 (f, x0 [, lb, ub, options])",
                        call_args.len(),
                    ));
                }
                let Value::Closure(cl) = &call_args[0] else {
                    return Err(MatroError::runtime("lsqnonlin: first argument must be a function handle"));
                };
                let x0 = call_args[1].to_dense_matrix()?;
                let lb = call_args.get(2).and_then(|v| v.to_dense_matrix().ok());
                let ub = call_args.get(3).and_then(|v| v.to_dense_matrix().ok());
                let (sol, resnorm, residual) = solvers::solve_lm_bounded(
                    self,
                    cl,
                    &x0,
                    None,
                    lb.as_ref(),
                    ub.as_ref(),
                    stdout,
                )?;
                let mut out = vec![Value::matrix(sol)];
                if nargout >= 2 { out.push(Value::scalar(resnorm)); }
                if nargout >= 3 { out.push(Value::matrix(residual)); }
                if nargout >= 4 { out.push(Value::scalar(1.0)); }
                if nargout >= 5 { out.push(make_out_struct("levenberg-marquardt")); }
                Ok(out)
            }
            "lsqcurvefit" => {
                if call_args.len() < 4 || call_args.len() > 7 {
                    return Err(MatroError::arity(
                        "lsqcurvefit",
                        "4 to 7 (fun, x0, xdata, ydata [, lb, ub, options])",
                        call_args.len(),
                    ));
                }
                let Value::Closure(cl) = &call_args[0] else {
                    return Err(MatroError::runtime("lsqcurvefit: first argument must be a function handle"));
                };
                let x0 = call_args[1].to_dense_matrix()?;
                let ydata = call_args[3].to_dense_matrix()?;
                let lb = call_args.get(4).and_then(|v| v.to_dense_matrix().ok());
                let ub = call_args.get(5).and_then(|v| v.to_dense_matrix().ok());
                let (sol, resnorm, residual) = solvers::solve_lm_bounded(
                    self,
                    cl,
                    &x0,
                    Some((&call_args[2], &ydata)),
                    lb.as_ref(),
                    ub.as_ref(),
                    stdout,
                )?;
                let mut out = vec![Value::matrix(sol)];
                if nargout >= 2 { out.push(Value::scalar(resnorm)); }
                if nargout >= 3 { out.push(Value::matrix(residual)); }
                if nargout >= 4 { out.push(Value::scalar(1.0)); }
                if nargout >= 5 { out.push(make_out_struct("levenberg-marquardt")); }
                Ok(out)
            }
            _ => unreachable!(),
        }
    }
}

fn exec_odeset(args: &[Value]) -> Result<Value> {
    let mut map = BTreeMap::new();
    let mut i = 0;
    if let Some(Value::Struct(existing)) = args.first() {
        map = existing.clone();
        i = 1;
    }
    while i + 1 < args.len() {
        let k = args[i].as_str()?.to_string();
        map.insert(k, args[i + 1].clone());
        i += 2;
    }
    Ok(Value::Struct(map))
}

fn exec_odeget(args: &[Value]) -> Result<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(MatroError::arity("odeget", "2 or 3 (options, name [, default])", args.len()));
    }
    let key = args[1].as_str()?;
    let default = args.get(2).cloned().unwrap_or_else(Value::empty);
    if let Value::Struct(map) = &args[0] {
        for (k, v) in map {
            if k.eq_ignore_ascii_case(key) {
                return Ok(v.clone());
            }
        }
    }
    Ok(default)
}

fn exec_optimset(name: &str, args: &[Value]) -> Result<Value> {
    let mut map = BTreeMap::new();
    let mut i = 0;
    if name == "optimoptions" && !args.is_empty() {
        if let Ok(solver) = args[0].as_str() {
            map.insert("SolverName".to_string(), Value::String(solver.to_string()));
            i = 1;
        } else if let Value::Struct(existing) = &args[0] {
            map = existing.clone();
            i = 1;
        }
    } else if let Some(Value::Struct(existing)) = args.first() {
        map = existing.clone();
        i = 1;
    } else if args.len() == 1 {
        if let Ok(solver) = args[0].as_str() {
            map.insert("SolverName".to_string(), Value::String(solver.to_string()));
            return Ok(Value::Struct(map));
        }
    }
    while i + 1 < args.len() {
        let k = args[i].as_str()?.to_string();
        map.insert(k, args[i + 1].clone());
        i += 2;
    }
    Ok(Value::Struct(map))
}

fn make_mexception(identifier: &str, message: &str, line: u32) -> Value {
    let mut fields = BTreeMap::new();
    fields.insert("identifier".to_string(), Value::String(identifier.to_string()));
    fields.insert("message".to_string(), Value::String(message.to_string()));
    fields.insert("line".to_string(), Value::scalar(line as f64));
    fields.insert(
        "cause".to_string(),
        Value::Cell(CellArray {
            rows: 0,
            cols: 1,
            data: Vec::new(),
            extra_dims: Vec::new(),
        }),
    );
    fields.insert(
        "stack".to_string(),
        Value::StructArray(matro_types::StructArray {
            rows: 0,
            cols: 1,
            items: Vec::new(),
        }),
    );
    Value::Object(ObjectInstance {
        class_name: "MException".to_string(),
        fields,
    })
}

fn mexception_to_error(val: &Value) -> MatroError {
    let map = match val {
        Value::Object(o) => Some(&o.fields),
        Value::Struct(m) => Some(m),
        _ => None,
    };
    if let Some(fields) = map {
        let id = fields
            .get("identifier")
            .and_then(|v| v.as_str().ok())
            .unwrap_or("Matro:Error");
        let msg = fields
            .get("message")
            .and_then(|v| v.as_str().ok())
            .unwrap_or("Exception thrown");
        if id.is_empty() {
            MatroError::runtime(msg)
        } else {
            MatroError::custom(id, msg)
        }
    } else if let Ok(s) = val.as_str() {
        MatroError::runtime(s)
    } else {
        MatroError::runtime("Exception thrown")
    }
}

fn is_vm_or_intrinsic_builtin(name: &str) -> bool {
    matches!(
        name,
        "feval"
            | "eval"
            | "evalc"
            | "arrayfun"
            | "cellfun"
            | "structfun"
            | "bsxfun"
            | "spfun"
            | "MException"
            | "throw"
            | "rethrow"
            | "lasterror"
            | "functions"
            | "func2str"
            | "str2func"
            | "input"
            | "load"
            | "save"
            | "exist"
            | "which"
            | "addpath"
            | "rmpath"
            | "genpath"
            | "path"
            | "isa"
            | "class"
            | "methods"
            | "properties"
            | "superclasses"
            | "builtin"
            | "ode45"
            | "ode23"
            | "ode15s"
            | "ode113"
            | "ode23s"
            | "ode23t"
            | "ode23tb"
            | "odeset"
            | "odeget"
            | "optimset"
            | "optimoptions"
            | "deval"
            | "bvpinit"
            | "bvp4c"
            | "pdepe"
            | "fzero"
            | "fsolve"
            | "fminsearch"
            | "fminbnd"
            | "fminunc"
            | "fmincon"
            | "lsqnonlin"
            | "lsqcurvefit"
            | "lsqnonneg"
            | "integral"
            | "integral2"
            | "quad"
            | "quadl"
            | "quadgk"
            | "pcg"
            | "gmres"
            | "bicgstab"
            | "minres"
            | "lsqr"
            | "cgs"
            | "bicg"
            | "fopen"
            | "fclose"
            | "fgetl"
            | "fgets"
            | "fprintf"
            | "fscanf"
            | "sscanf"
            | "fread"
            | "fwrite"
            | "feof"
            | "ftell"
            | "fseek"
            | "frewind"
            | "ferror"
            | "textscan"
            | "nargin"
            | "nargout"
            | "narginchk"
            | "nargoutchk"
            | "who"
            | "whos"
            | "clear"
            | "clc"
    ) || matro_intrinsics::is_intrinsic(name)
}

fn assign_cell_parens(cell: &mut CellArray, args: &[StackItem], rhs: Value) -> Result<()> {
    let is_delete = matches!(&rhs, Value::Matrix(m) if m.is_empty());
    if is_delete && args.len() == 1 {
        let del = resolve_indices(&args[0], cell.data.len())?;
        let mut keep = vec![true; cell.data.len()];
        for i in del {
            if i < keep.len() {
                keep[i] = false;
            }
        }
        let new_data: Vec<Value> = cell
            .data
            .iter()
            .enumerate()
            .filter(|(i, _)| keep[*i])
            .map(|(_, v)| v.clone())
            .collect();
        let n = new_data.len();
        if cell.cols == 1 && cell.rows > 1 {
            cell.rows = n;
            cell.cols = if n == 0 { 0 } else { 1 };
        } else {
            cell.rows = if n == 0 { 0 } else { 1 };
            cell.cols = n;
        }
        cell.data = new_data;
        return Ok(());
    }
    let rhs_cells: Vec<Value> = match rhs {
        Value::Cell(c) => c.data,
        other => vec![other],
    };
    match args.len() {
        1 => {
            let idxs: Vec<usize> = match &args[0] {
                StackItem::Colon => (0..cell.data.len()).collect(),
                StackItem::Scalar(x) => {
                    let i = x.round() as isize;
                    if i < 1 {
                        return Err(MatroError::index(format!("Index {x} must be >= 1")));
                    }
                    vec![(i - 1) as usize]
                }
                StackItem::Val(v) => {
                    let m = v.to_dense_matrix()?;
                    if m.is_logical() {
                        m.data()
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &f)| (f != 0.0).then_some(i))
                            .collect()
                    } else {
                        m.data()
                            .iter()
                            .map(|&x| {
                                let i = x.round() as isize;
                                if i < 1 {
                                    Err(MatroError::index(format!("Index {x} must be >= 1")))
                                } else {
                                    Ok((i - 1) as usize)
                                }
                            })
                            .collect::<Result<_>>()?
                    }
                }
                _ => Vec::new(),
            };
            let needed = idxs.iter().copied().max().map(|i| i + 1).unwrap_or(0);
            if needed > cell.data.len() {
                let was_col = cell.cols == 1 && cell.rows > 1;
                cell.data.resize(needed, Value::empty());
                if was_col {
                    cell.rows = needed;
                    cell.cols = 1;
                } else {
                    cell.rows = 1;
                    cell.cols = needed;
                }
            }
            if rhs_cells.len() == 1 {
                for &i in &idxs {
                    cell.data[i] = rhs_cells[0].clone();
                }
            } else if rhs_cells.len() == idxs.len() {
                for (k, &i) in idxs.iter().enumerate() {
                    cell.data[i] = rhs_cells[k].clone();
                }
            } else {
                return Err(MatroError::dimension("Cell indexed assignment size mismatch"));
            }
            Ok(())
        }
        2 => {
            let parse_dim = |item: &StackItem, cur: usize| -> Result<Vec<usize>> {
                match item {
                    StackItem::Colon => Ok((0..cur).collect()),
                    StackItem::Scalar(x) => {
                        let i = x.round() as isize;
                        if i < 1 {
                            Err(MatroError::index(format!("Index {x} must be >= 1")))
                        } else {
                            Ok(vec![(i - 1) as usize])
                        }
                    }
                    StackItem::Val(v) => {
                        let m = v.to_dense_matrix()?;
                        if m.is_logical() {
                            Ok(m.data()
                                .iter()
                                .enumerate()
                                .filter_map(|(i, &f)| (f != 0.0).then_some(i))
                                .collect())
                        } else {
                            m.data()
                                .iter()
                                .map(|&x| {
                                    let i = x.round() as isize;
                                    if i < 1 {
                                        Err(MatroError::index(format!("Index {x} must be >= 1")))
                                    } else {
                                        Ok((i - 1) as usize)
                                    }
                                })
                                .collect()
                        }
                    }
                    _ => Ok(Vec::new()),
                }
            };
            let rows = parse_dim(&args[0], cell.rows)?;
            let cols = parse_dim(&args[1], cell.cols)?;
            let new_r = cell.rows.max(rows.iter().copied().max().map(|i| i + 1).unwrap_or(0));
            let new_c = cell.cols.max(cols.iter().copied().max().map(|i| i + 1).unwrap_or(0));
            if new_r != cell.rows || new_c != cell.cols {
                let mut grown = CellArray {
                    rows: new_r,
                    cols: new_c,
                    data: vec![Value::empty(); new_r * new_c],
                    extra_dims: Vec::new(),
                };
                for c in 0..cell.cols {
                    for r in 0..cell.rows {
                        grown.data[c * new_r + r] = cell.get(r, c).clone();
                    }
                }
                *cell = grown;
            }
            let total = rows.len() * cols.len();
            if rhs_cells.len() == 1 {
                for &c in &cols {
                    for &r in &rows {
                        cell.data[c * cell.rows + r] = rhs_cells[0].clone();
                    }
                }
            } else if rhs_cells.len() == total {
                let mut k = 0;
                for &c in &cols {
                    for &r in &rows {
                        cell.data[c * cell.rows + r] = rhs_cells[k].clone();
                        k += 1;
                    }
                }
            } else {
                return Err(MatroError::dimension("Cell indexed assignment size mismatch"));
            }
            Ok(())
        }
        n => Err(MatroError::index(format!(
            "Cell assignment expects 1 or 2 indices, got {n}"
        ))),
    }
}

fn assign_struct_parens(val: &mut Value, args: &[StackItem], rhs: Value) -> Result<()> {
    let mut sa = match std::mem::replace(val, Value::empty()) {
        Value::Struct(m) => matro_types::StructArray {
            rows: 1,
            cols: 1,
            items: vec![m],
        },
        Value::StructArray(sa) => sa,
        other => {
            *val = other;
            return Err(MatroError::runtime("Expected struct array"));
        }
    };
    let is_delete = matches!(&rhs, Value::Matrix(m) if m.is_empty());
    if is_delete && args.len() == 1 {
        let del = resolve_indices(&args[0], sa.items.len())?;
        let mut keep = vec![true; sa.items.len()];
        for i in del {
            if i < keep.len() {
                keep[i] = false;
            }
        }
        sa.items = sa
            .items
            .into_iter()
            .enumerate()
            .filter(|(i, _)| keep[*i])
            .map(|(_, s)| s)
            .collect();
        sa.rows = if sa.items.is_empty() { 0 } else { 1 };
        sa.cols = sa.items.len();
        *val = if sa.items.len() == 1 {
            Value::Struct(sa.items.into_iter().next().unwrap())
        } else {
            Value::StructArray(sa)
        };
        return Ok(());
    }
    let rhs_items: Vec<BTreeMap<String, Value>> = match rhs {
        Value::Struct(m) => vec![m],
        Value::StructArray(s) => s.items,
        other => {
            *val = Value::StructArray(sa);
            return Err(MatroError::runtime(format!(
                "Cannot assign {} into struct array",
                other.type_name()
            )));
        }
    };
    if args.len() != 1 {
        *val = Value::StructArray(sa);
        return Err(MatroError::index("Struct array assignment currently expects 1 index"));
    }
    let idxs: Vec<usize> = match &args[0] {
        StackItem::Colon => (0..sa.items.len()).collect(),
        StackItem::Scalar(x) => {
            let i = x.round() as isize;
            if i < 1 {
                return Err(MatroError::index(format!("Index {x} must be >= 1")));
            }
            vec![(i - 1) as usize]
        }
        StackItem::Val(v) => {
            let m = v.to_dense_matrix()?;
            if m.is_logical() {
                m.data()
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &f)| (f != 0.0).then_some(i))
                    .collect()
            } else {
                m.data()
                    .iter()
                    .map(|&x| {
                        let i = x.round() as isize;
                        if i < 1 {
                            Err(MatroError::index(format!("Index {x} must be >= 1")))
                        } else {
                            Ok((i - 1) as usize)
                        }
                    })
                    .collect::<Result<_>>()?
            }
        }
        _ => Vec::new(),
    };
    let needed = idxs.iter().copied().max().map(|i| i + 1).unwrap_or(0);
    if needed > sa.items.len() {
        let template: BTreeMap<String, Value> = sa
            .items
            .first()
            .or_else(|| rhs_items.first())
            .map(|m| m.keys().map(|k| (k.clone(), Value::empty())).collect())
            .unwrap_or_default();
        sa.items.resize(needed, template);
        sa.rows = 1;
        sa.cols = needed;
    }
    if rhs_items.len() == 1 {
        for &i in &idxs {
            sa.items[i] = rhs_items[0].clone();
        }
    } else if rhs_items.len() == idxs.len() {
        for (k, &i) in idxs.iter().enumerate() {
            sa.items[i] = rhs_items[k].clone();
        }
    } else {
        return Err(MatroError::dimension("Struct array assignment size mismatch"));
    }
    *val = if sa.items.len() == 1 {
        Value::Struct(sa.items.into_iter().next().unwrap())
    } else {
        Value::StructArray(sa)
    };
    Ok(())
}

fn assign_string_parens(s: &mut String, args: &[StackItem], rhs: Value) -> Result<()> {
    let mut chars: Vec<char> = s.chars().collect();
    let idx_item = match args.len() {
        1 => &args[0],
        2 => &args[1],
        n => {
            return Err(MatroError::index(format!(
                "Char array assignment expects 1 or 2 indices, got {n}"
            )))
        }
    };
    let is_delete = matches!(&rhs, Value::Matrix(m) if m.is_empty())
        || matches!(&rhs, Value::String(rs) if rs.is_empty());
    if is_delete {
        let del = resolve_indices(idx_item, chars.len())?;
        let mut keep = vec![true; chars.len()];
        for i in del {
            if i < keep.len() {
                keep[i] = false;
            }
        }
        *s = chars
            .into_iter()
            .enumerate()
            .filter(|(i, _)| keep[*i])
            .map(|(_, ch)| ch)
            .collect();
        return Ok(());
    }
    let idxs: Vec<usize> = match idx_item {
        StackItem::Colon => (0..chars.len()).collect(),
        StackItem::Scalar(x) => {
            let i = x.round() as isize;
            if i < 1 {
                return Err(MatroError::index(format!("Index {x} must be >= 1")));
            }
            vec![(i - 1) as usize]
        }
        StackItem::Val(v) => {
            let m = v.to_dense_matrix()?;
            if m.is_logical() {
                m.data()
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &f)| (f != 0.0).then_some(i))
                    .collect()
            } else {
                m.data()
                    .iter()
                    .map(|&x| {
                        let i = x.round() as isize;
                        if i < 1 {
                            Err(MatroError::index(format!("Index {x} must be >= 1")))
                        } else {
                            Ok((i - 1) as usize)
                        }
                    })
                    .collect::<Result<_>>()?
            }
        }
        _ => Vec::new(),
    };
    let rhs_chars: Vec<char> = match rhs {
        Value::String(rs) => rs.chars().collect(),
        Value::Matrix(m) => m
            .data()
            .iter()
            .map(|&x| char::from_u32(x.round().max(0.0) as u32).unwrap_or('\0'))
            .collect(),
        other => {
            return Err(MatroError::runtime(format!(
                "Cannot assign {} into char vector",
                other.type_name()
            )))
        }
    };
    let needed = idxs.iter().copied().max().map(|i| i + 1).unwrap_or(0);
    if needed > chars.len() {
        chars.resize(needed, ' ');
    }
    if rhs_chars.len() == 1 {
        for &i in &idxs {
            chars[i] = rhs_chars[0];
        }
    } else if rhs_chars.len() == idxs.len() {
        for (k, &i) in idxs.iter().enumerate() {
            chars[i] = rhs_chars[k];
        }
    } else {
        return Err(MatroError::dimension("Char assignment size mismatch"));
    }
    *s = chars.into_iter().collect();
    Ok(())
}

fn end_for_value(val: &Value, dim: u8, total_dims: u8) -> usize {
    match val {
        Value::Matrix(m) => {
            if total_dims == 1 {
                m.len()
            } else if (dim as usize) + 1 == (total_dims as usize) && m.ndims() > (total_dims as usize) {
                let dims = m.dims();
                dims[(dim as usize)..].iter().product()
            } else {
                m.dim_size(dim as usize)
            }
        }
        Value::Sparse(s) => {
            if total_dims == 1 {
                s.rows().saturating_mul(s.cols())
            } else if dim == 0 {
                s.rows()
            } else {
                s.cols()
            }
        }
        Value::StructArray(sa) => {
            if total_dims == 1 {
                sa.items.len()
            } else if dim == 0 {
                sa.rows
            } else {
                sa.cols
            }
        }
        Value::Cell(c) => {
            if total_dims == 1 {
                c.data.len()
            } else if (dim as usize) == (total_dims as usize).saturating_sub(1) {
                let dims = c.dims();
                dims[(dim as usize)..].iter().product()
            } else {
                c.dim_size(dim as usize)
            }
        }
        Value::Table(t) => {
            if dim == 0 {
                t.nrows
            } else {
                t.col_names.len()
            }
        }
        Value::String(s) => {
            if total_dims == 1 || dim == 1 {
                s.chars().count()
            } else {
                1
            }
        }
        Value::Object(o) => match o.class_name.as_str() {
            "string" => o
                .fields
                .get("Data")
                .map_or(1, |v| end_for_value(v, dim, total_dims)),
            "datetime" => o
                .fields
                .get("EpochSec")
                .map_or(1, |v| end_for_value(v, dim, total_dims)),
            "duration" => o
                .fields
                .get("Seconds")
                .map_or(1, |v| end_for_value(v, dim, total_dims)),
            "categorical" => o
                .fields
                .get("Codes")
                .map_or(1, |v| end_for_value(v, dim, total_dims)),
            _ => 1,
        },
        Value::Map(m) => m.entries.len(),
        _ => 1,
    }
}

fn push_call_results(stack: &mut Vec<StackItem>, results: Vec<Value>, wanted: usize) {
    if wanted == 0 {
        return;
    }
    if results.is_empty() {
        for _ in 0..wanted {
            stack.push(StackItem::Val(Value::empty()));
        }
    } else {
        for i in 0..wanted {
            let v = results.get(i).cloned().unwrap_or_else(Value::empty);
            stack.push(StackItem::Val(v));
        }
    }
}

fn switch_matches(scrutinee: &Value, case_val: &Value) -> bool {
    if let Value::Cell(cell) = case_val {
        return cell.data.iter().any(|item| switch_matches(scrutinee, item));
    }
    match (scrutinee, case_val) {
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Matrix(a), Value::Matrix(b)) => {
            if let (Some(ca), Some(cb)) = (a.as_complex_scalar(), b.as_complex_scalar()) {
                ca == cb
            } else {
                a == b
            }
        }
        _ => false,
    }
}

fn set_nested_field_map(
    map: &mut BTreeMap<String, Value>,
    fields: &[String],
    val: Value,
) -> Result<()> {
    let Some((first, rest)) = fields.split_first() else {
        return Ok(());
    };
    if rest.is_empty() {
        map.insert(first.clone(), val);
    } else {
        let entry = map
            .entry(first.clone())
            .or_insert_with(|| Value::Struct(BTreeMap::new()));
        set_nested_field(entry, rest, val)?;
    }
    Ok(())
}

fn set_nested_field(target: &mut Value, fields: &[String], val: Value) -> Result<()> {
    let Some((first, rest)) = fields.split_first() else {
        *target = val;
        return Ok(());
    };
    match target {
        Value::Struct(map) => set_nested_field_map(map, fields, val)?,
        Value::Object(obj) => set_nested_field_map(&mut obj.fields, fields, val)?,
        Value::Table(t) => {
            if first == "Properties" && rest.len() == 1 && rest[0] == "VariableNames" {
                let Value::Cell(c) = val else {
                    return Err(MatroError::runtime(
                        "table.Properties.VariableNames must be a cell array of strings",
                    ));
                };
                if c.data.len() != t.col_names.len() {
                    return Err(MatroError::dimension(
                        "VariableNames length must match table width",
                    ));
                }
                t.col_names = c
                    .data
                    .iter()
                    .map(|v| v.as_str().map(ToString::to_string))
                    .collect::<Result<_>>()?;
            } else if rest.is_empty() {
                t.set_col(first.clone(), val)?;
            } else {
                return Err(MatroError::runtime("Nested table property assignment not supported"));
            }
        }
        other => {
            *other = Value::Struct(BTreeMap::new());
            set_nested_field(other, fields, val)?;
        }
    }
    Ok(())
}

fn resolve_table_cols(t: &TableData, item: &StackItem) -> Result<Vec<usize>> {
    match item {
        StackItem::Colon => Ok((0..t.width()).collect()),
        StackItem::Val(Value::String(name)) => {
            let idx = t.col_index(name).ok_or_else(|| {
                MatroError::index(format!("Unknown table variable '{name}'"))
            })?;
            Ok(vec![idx])
        }
        StackItem::Val(Value::Cell(c)) => {
            let mut cols = Vec::with_capacity(c.data.len());
            for v in &c.data {
                let name = v.as_str()?;
                let idx = t.col_index(name).ok_or_else(|| {
                    MatroError::index(format!("Unknown table variable '{name}'"))
                })?;
                cols.push(idx);
            }
            Ok(cols)
        }
        other => resolve_indices(other, t.width()),
    }
}

fn index_table(t: &TableData, args: &[StackItem]) -> Result<TableData> {
    if args.len() != 2 {
        return Err(MatroError::index(
            "Table indexing requires 2 subscripts T(rows, cols)",
        ));
    }
    let rows = resolve_indices(&args[0], t.nrows)?;
    let cols = resolve_table_cols(t, &args[1])?;
    t.slice(&rows, &cols)
}

fn extract_table_braces(t: &TableData, args: &[StackItem]) -> Result<Value> {
    if args.len() != 2 {
        return Err(MatroError::index(
            "Table brace indexing requires 2 subscripts T{rows, cols}",
        ));
    }
    let rows = resolve_indices(&args[0], t.nrows)?;
    let cols = resolve_table_cols(t, &args[1])?;
    let sub = t.slice(&rows, &cols)?;
    let all_numeric = sub.columns.iter().all(|v| matches!(v, Value::Matrix(_)));
    if all_numeric && !sub.columns.is_empty() {
        let mut out = Matrix::zeros(sub.nrows, sub.width());
        for (c, col) in sub.columns.iter().enumerate() {
            let m = col.as_matrix()?;
            for r in 0..sub.nrows {
                out.set_c(r, c, m.get_c(r, 0));
            }
        }
        Ok(Value::matrix(out.normalize()))
    } else if sub.width() == 1 && sub.nrows == 1 {
        match &sub.columns[0] {
            Value::Cell(c) => Ok(c.data[0].clone()),
            other => Ok(other.clone()),
        }
    } else {
        let mut grid = Vec::with_capacity(sub.nrows);
        for r in 0..sub.nrows {
            let mut row = Vec::with_capacity(sub.width());
            for c in 0..sub.width() {
                match &sub.columns[c] {
                    Value::Matrix(m) => {
                        row.push(Value::matrix(Matrix::complex_scalar(m.get_c(r, 0)).normalize()));
                    }
                    Value::Cell(cell) => row.push(cell.data[r].clone()),
                    other => row.push(other.clone()),
                }
            }
            grid.push(row);
        }
        Ok(Value::Cell(CellArray::from_rows(grid)?))
    }
}

fn index_struct_array(sa: &StructArray, args: &[StackItem]) -> Result<Value> {
    if args.len() != 1 {
        return Err(MatroError::index("Struct array indexing currently supports 1D indexing"));
    }
    let idxs = resolve_indices(&args[0], sa.items.len())?;
    if idxs.len() == 1 {
        Ok(Value::Struct(sa.items[idxs[0]].clone()))
    } else {
        let items: Vec<BTreeMap<String, Value>> = idxs.iter().map(|&i| sa.items[i].clone()).collect();
        let len = items.len();
        Ok(Value::StructArray(StructArray {
            rows: if len == 0 { 0 } else { 1 },
            cols: len,
            items,
        }))
    }
}

fn assign_struct_array_field(
    cur: Option<Value>,
    args: &[StackItem],
    path: &[String],
    rhs: Value,
) -> Result<Value> {
    if args.len() != 1 {
        return Err(MatroError::index(
            "Struct array field assignment currently supports 1D indexing",
        ));
    }
    let idx = args[0].clone().into_val()?.as_scalar()?.round() as isize;
    if idx < 1 {
        return Err(MatroError::index("Struct array index must be >= 1"));
    }
    let pos = (idx - 1) as usize;
    let mut sa = match cur {
        Some(Value::StructArray(sa)) => sa,
        Some(Value::Struct(map)) => StructArray {
            rows: 1,
            cols: 1,
            items: vec![map],
        },
        _ => StructArray {
            rows: 0,
            cols: 0,
            items: Vec::new(),
        },
    };
    if pos >= sa.items.len() {
        sa.items.resize_with(pos + 1, BTreeMap::new);
        sa.rows = 1;
        sa.cols = sa.items.len();
    }
    set_nested_field_map(&mut sa.items[pos], path, rhs)?;
    Ok(Value::StructArray(sa))
}

fn lookup_var<'a>(
    name: &str,
    frame: Option<&'a LocalFrame<'_>>,
    globals: &'a HashMap<String, Value>,
) -> Option<&'a Value> {
    if let Some(f) = frame {
        f.get_by_name(name)
    } else {
        globals.get(name)
    }
}

fn store_var(
    name: String,
    val: Value,
    frame: Option<&mut LocalFrame<'_>>,
    globals: &mut HashMap<String, Value>,
) {
    if let Some(f) = frame {
        if !f.set_by_name(&name, val.clone()) {
            globals.insert(name, val);
        }
    } else {
        globals.insert(name, val);
    }
}

fn remove_var(
    name: &str,
    frame: Option<&mut LocalFrame<'_>>,
    globals: &mut HashMap<String, Value>,
) {
    if let Some(f) = frame {
        if let Some(pos) = f.names.iter().position(|n| n == name) {
            f.slots[pos] = None;
            return;
        }
    }
    globals.remove(name);
}

fn pop_val(stack: &mut Vec<StackItem>) -> Result<Value> {
    stack
        .pop()
        .ok_or_else(|| MatroError::runtime("VM stack underflow"))?
        .into_val()
}

fn pop_vals(stack: &mut Vec<StackItem>, n: usize) -> Result<Vec<Value>> {
    if stack.len() < n {
        return Err(MatroError::runtime("VM stack underflow"));
    }
    let start = stack.len() - n;
    stack.drain(start..).map(StackItem::into_val).collect()
}

fn pop_items(stack: &mut Vec<StackItem>, n: usize) -> Result<Vec<StackItem>> {
    if stack.len() < n {
        return Err(MatroError::runtime("VM stack underflow"));
    }
    let start = stack.len() - n;
    Ok(stack.drain(start..).collect())
}

fn flatten_call_args(raw_args: Vec<StackItem>) -> Result<Vec<Value>> {
    let mut out = Vec::with_capacity(raw_args.len());
    for item in raw_args {
        match item {
            StackItem::Val(v) => out.push(v),
            StackItem::Scalar(x) => out.push(Value::scalar(x)),
            StackItem::Spread(vs) => out.extend(vs),
            StackItem::Void => out.push(Value::empty()),
            StackItem::Colon => {
                return Err(MatroError::runtime("Unexpected ':' outside indexing"));
            }
        }
    }
    Ok(out)
}

fn bin_logical(stack: &mut Vec<StackItem>, f: impl Fn(f64, f64) -> bool) -> Result<()> {
    let b = pop_val(stack)?;
    let a = pop_val(stack)?;
    let res = a
        .to_dense_matrix()?
        .elementwise_c(&b.to_dense_matrix()?, |x, y| {
            let ax = if !x.is_zero() && !x.re.is_nan() && !x.im.is_nan() { 1.0 } else { 0.0 };
            let by = if !y.is_zero() && !y.re.is_nan() && !y.im.is_nan() { 1.0 } else { 0.0 };
            Complex::from_real(if f(ax, by) { 1.0 } else { 0.0 })
        })?
        .with_logical(true);
    stack.push(StackItem::Val(Value::matrix(res)));
    Ok(())
}

fn extract_col(m: &Matrix, col: usize) -> Matrix {
    let orig_dtype = m.dtype();
    let out = if m.rows() == 1 {
        Matrix::complex_scalar(m.get_linear_c(col))
    } else {
        let base = col * m.rows();
        let vals = (0..m.rows()).map(|r| m.get_linear_c(base + r)).collect();
        Matrix::complex_col_vec(vals)
    };
    preserve_dtype(out.normalize(), orig_dtype)
}

fn preserve_dtype(m: Matrix, dtype: matro_types::NumericClass) -> Matrix {
    if dtype == matro_types::NumericClass::Logical {
        m.with_logical(true)
    } else if dtype != matro_types::NumericClass::Double {
        m.cast_to(dtype)
    } else {
        m
    }
}

fn resolve_indices(item: &StackItem, max_len: usize) -> Result<Vec<usize>> {
    match item {
        StackItem::Colon => Ok((0..max_len).collect()),
        StackItem::Scalar(x) => {
            let idx = x.round() as isize;
            if (x - idx as f64).abs() > 1e-12 || idx < 1 || (idx as usize) > max_len {
                return Err(MatroError::index(format!(
                    "Index {x} out of bounds for length {max_len} (1-based)"
                )));
            }
            Ok(vec![(idx - 1) as usize])
        }
        StackItem::Spread(vs) => {
            let first = vs.first().ok_or_else(|| {
                MatroError::index("Empty comma-separated list cannot be used as index")
            })?;
            resolve_indices(&StackItem::Val(first.clone()), max_len)
        }
        StackItem::Void => Ok(Vec::new()),
        StackItem::Val(v) => {
            let m = v.to_dense_matrix()?;
            if m.is_logical() {
                let mut out = Vec::new();
                for (i, &flag) in m.data().iter().enumerate() {
                    if flag != 0.0 {
                        if i >= max_len {
                            return Err(MatroError::index(format!(
                                "Logical index position {} exceeds dimension {}",
                                i + 1,
                                max_len
                            )));
                        }
                        out.push(i);
                    }
                }
                return Ok(out);
            }
            let mut out = Vec::with_capacity(m.len());
            for &x in m.data() {
                let idx = x.round() as isize;
                if (x - idx as f64).abs() > 1e-12 || idx < 1 || (idx as usize) > max_len {
                    return Err(MatroError::index(format!(
                        "Index {x} out of bounds for length {max_len} (1-based)"
                    )));
                }
                out.push((idx - 1) as usize);
            }
            Ok(out)
        }
    }
}

fn index_sparse(s: &SparseMatrix, args: &[StackItem]) -> Result<Value> {
    if args.len() == 2 {
        let rows = resolve_indices(&args[0], s.rows())?;
        let cols = resolve_indices(&args[1], s.cols())?;
        return Ok(Value::sparse(s.submatrix(&rows, &cols)));
    }
    Ok(Value::matrix(index_matrix(&s.to_dense(), args)?))
}

fn assign_sparse_index(s: &mut SparseMatrix, args: &[StackItem], rhs: &Matrix) -> Result<()> {
    match args.len() {
        2 => {
            let parse_dim = |item: &StackItem, cur: usize| -> Result<Vec<usize>> {
                match item {
                    StackItem::Colon => Ok((0..cur).collect()),
                    StackItem::Scalar(x) => {
                        let i = x.round() as isize;
                        if i < 1 {
                            Err(MatroError::index(format!("Index {x} must be >= 1")))
                        } else {
                            Ok(vec![(i - 1) as usize])
                        }
                    }
                    StackItem::Val(v) => {
                        let idx_m = v.as_matrix()?;
                        if idx_m.is_logical() {
                            Ok(idx_m
                                .data()
                                .iter()
                                .enumerate()
                                .filter_map(|(i, &f)| (f != 0.0).then_some(i))
                                .collect())
                        } else {
                            idx_m
                                .data()
                                .iter()
                                .map(|&x| {
                                    let i = x.round() as isize;
                                    if i < 1 {
                                        Err(MatroError::index(format!("Index {x} must be >= 1")))
                                    } else {
                                        Ok((i - 1) as usize)
                                    }
                                })
                                .collect()
                        }
                    }
                    StackItem::Spread(_) => Err(MatroError::index(
                        "Comma-separated list cannot be used as assignment index",
                    )),
                    StackItem::Void => Ok(Vec::new()),
                }
            };
            let rows = parse_dim(&args[0], s.rows())?;
            let cols = parse_dim(&args[1], s.cols())?;
            if let Some(sc) = rhs.as_complex_scalar() {
                for &c in &cols {
                    for &r in &rows {
                        s.set_c(r, c, sc);
                    }
                }
            } else if rhs.rows() == rows.len() && rhs.cols() == cols.len() {
                for (ci, &c) in cols.iter().enumerate() {
                    for (ri, &r) in rows.iter().enumerate() {
                        s.set_c(r, c, rhs.get_c(ri, ci));
                    }
                }
            } else if (rows.len() == 1 || cols.len() == 1)
                && (rhs.rows() == 1 || rhs.cols() == 1)
                && rhs.len() == rows.len() * cols.len()
            {
                let mut k = 0usize;
                for &c in &cols {
                    for &r in &rows {
                        s.set_c(r, c, rhs.get_linear_c(k));
                        k += 1;
                    }
                }
            } else {
                return Err(MatroError::dimension("Sparse indexed assignment size mismatch"));
            }
            Ok(())
        }
        1 => {
            let mut dense = s.to_dense();
            assign_matrix_index(&mut dense, args, rhs)?;
            *s = SparseMatrix::from_dense(&dense);
            Ok(())
        }
        n => Err(MatroError::index(format!(
            "Sparse assignment expects 1 or 2 indices, got {n}"
        ))),
    }
}

fn nd_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(dims.len());
    let mut s = 1usize;
    for &d in dims {
        strides.push(s);
        s = s.saturating_mul(d);
    }
    strides
}

fn index_matrix(m: &Matrix, args: &[StackItem]) -> Result<Matrix> {
    let orig_dtype = m.dtype();
    let res = match args.len() {
        1 => {
            let idxs = resolve_indices(&args[0], m.len())?;
            let vals: Vec<Complex> = idxs.iter().map(|&i| m.get_linear_c(i)).collect();
            if let StackItem::Val(v) = &args[0] {
                if let Ok(idx_m) = v.to_dense_matrix() {
                    if !idx_m.is_logical() {
                        let a_is_vec =
                            m.ndims() == 2 && (m.rows() == 1 || m.cols() == 1) && m.len() != 1;
                        let idx_is_vec =
                            idx_m.ndims() == 2 && (idx_m.rows() == 1 || idx_m.cols() == 1);
                        if !a_is_vec || !idx_is_vec {
                            let re: Vec<f64> = vals.iter().map(|c| c.re).collect();
                            let im: Vec<f64> = vals.iter().map(|c| c.im).collect();
                            return Matrix::from_nd_complex_col_major(idx_m.dims(), re, im)
                                .map(|mat| preserve_dtype(mat.normalize(), orig_dtype));
                        }
                    }
                }
            }
            if m.rows() == 1 && m.ndims() == 2 && !matches!(args[0], StackItem::Colon) {
                Matrix::complex_row_vec(vals).normalize()
            } else {
                Matrix::complex_col_vec(vals).normalize()
            }
        }
        2 => {
            let rows = resolve_indices(&args[0], m.rows())?;
            // In MATLAB, 2D indexing of a 3D+ array collapses trailing dims into dim 2
            let eff_cols = if m.ndims() > 2 && m.rows() > 0 {
                m.len() / m.rows()
            } else {
                m.cols()
            };
            let cols = resolve_indices(&args[1], eff_cols)?;
            let mut out = Matrix::zeros(rows.len(), cols.len());
            for (ci, &c) in cols.iter().enumerate() {
                for (ri, &r) in rows.iter().enumerate() {
                    out.set_c(ri, ci, m.get_linear_c(c * m.rows() + r));
                }
            }
            out.normalize()
        }
        n => {
            // General N-D indexing `A(i1, i2, ..., in)` with MATLAB trailing-dim collapse when n < ndims
            let in_dims = m.dims();
            let mut eff_dims = Vec::with_capacity(n);
            for ax in 0..n {
                if ax + 1 == n && in_dims.len() > n {
                    eff_dims.push(in_dims[ax..].iter().product());
                } else {
                    eff_dims.push(in_dims.get(ax).copied().unwrap_or(1));
                }
            }
            let mut dim_idxs = Vec::with_capacity(n);
            let mut out_dims = Vec::with_capacity(n);
            for (ax, arg) in args.iter().enumerate() {
                let dlen = eff_dims[ax];
                let idxs = resolve_indices(arg, dlen)?;
                out_dims.push(idxs.len());
                dim_idxs.push(idxs);
            }
            let in_strides = nd_strides(&eff_dims);
            let total: usize = out_dims.iter().product();
            let mut re = Vec::with_capacity(total);
            let mut im = Vec::with_capacity(total);
            let mut coord = vec![0usize; n];
            for _ in 0..total {
                let mut src = 0usize;
                for ax in 0..n {
                    let c = dim_idxs[ax][coord[ax]];
                    src += c * in_strides[ax];
                }
                let val = m.get_linear_c(src);
                re.push(val.re);
                im.push(val.im);
                for ax in 0..n {
                    coord[ax] += 1;
                    if coord[ax] < out_dims[ax] {
                        break;
                    }
                    coord[ax] = 0;
                }
            }
            Matrix::from_nd_complex_col_major(out_dims, re, im)?.normalize()
        }
    };
    Ok(preserve_dtype(res, orig_dtype))
}

fn effective_nd_dims(in_dims: &[usize], n: usize) -> Vec<usize> {
    let mut eff = Vec::with_capacity(n);
    for ax in 0..n {
        if ax + 1 == n && in_dims.len() > n {
            eff.push(in_dims[ax..].iter().product());
        } else {
            eff.push(in_dims.get(ax).copied().unwrap_or(1));
        }
    }
    eff
}

fn index_cell_contents(cell: &CellArray, args: &[StackItem]) -> Result<Vec<Value>> {
    match args.len() {
        1 => {
            let idxs = resolve_indices(&args[0], cell.data.len())?;
            Ok(idxs.iter().map(|&i| cell.data[i].clone()).collect())
        }
        2 if cell.extra_dims.is_empty() => {
            let rows = resolve_indices(&args[0], cell.rows)?;
            let cols = resolve_indices(&args[1], cell.cols)?;
            let mut out = Vec::with_capacity(rows.len() * cols.len());
            for &c in &cols {
                for &r in &rows {
                    out.push(cell.get(r, c).clone());
                }
            }
            Ok(out)
        }
        n => {
            let eff_dims = effective_nd_dims(&cell.dims(), n);
            let mut dim_idxs = Vec::with_capacity(n);
            let mut out_dims = Vec::with_capacity(n);
            for ax in 0..n {
                let idxs = resolve_indices(&args[ax], eff_dims[ax])?;
                out_dims.push(idxs.len());
                dim_idxs.push(idxs);
            }
            let in_strides = nd_strides(&eff_dims);
            let total: usize = out_dims.iter().product();
            let mut out = Vec::with_capacity(total);
            let mut coord = vec![0usize; n];
            for _ in 0..total {
                let mut src = 0usize;
                for ax in 0..n {
                    src += dim_idxs[ax][coord[ax]] * in_strides[ax];
                }
                out.push(cell.data[src].clone());
                for ax in 0..n {
                    coord[ax] += 1;
                    if coord[ax] < out_dims[ax] {
                        break;
                    }
                    coord[ax] = 0;
                }
            }
            Ok(out)
        }
    }
}

fn index_cell_subarray(cell: &CellArray, args: &[StackItem]) -> Result<CellArray> {
    match args.len() {
        1 => {
            let idxs = resolve_indices(&args[0], cell.data.len())?;
            let data: Vec<Value> = idxs.iter().map(|&i| cell.data[i].clone()).collect();
            let len = data.len();
            Ok(CellArray::from_vec(if len == 0 { 0 } else { 1 }, len, data))
        }
        2 if cell.extra_dims.is_empty() => {
            let rows = resolve_indices(&args[0], cell.rows)?;
            let cols = resolve_indices(&args[1], cell.cols)?;
            let mut row_items = Vec::with_capacity(rows.len());
            for &r in &rows {
                let mut row = Vec::with_capacity(cols.len());
                for &c in &cols {
                    row.push(cell.get(r, c).clone());
                }
                row_items.push(row);
            }
            CellArray::from_rows(row_items)
        }
        n => {
            let eff_dims = effective_nd_dims(&cell.dims(), n);
            let mut dim_idxs = Vec::with_capacity(n);
            let mut out_dims = Vec::with_capacity(n);
            for ax in 0..n {
                let idxs = resolve_indices(&args[ax], eff_dims[ax])?;
                out_dims.push(idxs.len());
                dim_idxs.push(idxs);
            }
            let in_strides = nd_strides(&eff_dims);
            let total: usize = out_dims.iter().product();
            let mut data = Vec::with_capacity(total);
            let mut coord = vec![0usize; n];
            for _ in 0..total {
                let mut src = 0usize;
                for ax in 0..n {
                    src += dim_idxs[ax][coord[ax]] * in_strides[ax];
                }
                data.push(cell.data[src].clone());
                for ax in 0..n {
                    coord[ax] += 1;
                    if coord[ax] < out_dims[ax] {
                        break;
                    }
                    coord[ax] = 0;
                }
            }
            while out_dims.len() > 2 && out_dims.last() == Some(&1) {
                out_dims.pop();
            }
            let rows = out_dims.first().copied().unwrap_or(0);
            let cols = out_dims.get(1).copied().unwrap_or(1);
            let extra_dims = if out_dims.len() > 2 {
                out_dims[2..].to_vec()
            } else {
                Vec::new()
            };
            Ok(CellArray {
                rows,
                cols,
                data,
                extra_dims,
            })
        }
    }
}

fn assign_chain_step(
    cur: &mut Value,
    steps: &[ChainStep],
    step_args: &[Vec<StackItem>],
    names: &[String],
    rhs: Value,
    handles: &mut [ObjectInstance],
) -> Result<()> {
    if steps.is_empty() {
        *cur = rhs;
        return Ok(());
    }
    let resolve_field_name = |step: &ChainStep, args: &[StackItem]| -> Result<Option<String>> {
        match *step {
            ChainStep::Field(f_idx) => Ok(Some(names[f_idx as usize].clone())),
            ChainStep::DynamicField => {
                let v = args
                    .first()
                    .ok_or_else(|| MatroError::runtime("Missing dynamic field name"))?
                    .clone()
                    .into_val()?;
                match v {
                    Value::String(s) => Ok(Some(s)),
                    Value::Object(o) if o.class_name == "string" => {
                        if let Some(Value::Cell(c)) = o.fields.get("Data") {
                            Ok(Some(
                                c.data
                                    .first()
                                    .and_then(|x| x.as_str().ok())
                                    .unwrap_or("")
                                    .to_string(),
                            ))
                        } else {
                            Err(MatroError::runtime("Dynamic field name must be a string"))
                        }
                    }
                    other => Err(MatroError::runtime(format!(
                        "Dynamic field name must be a character vector or string, got {}",
                        other.type_name()
                    ))),
                }
            }
            _ => Ok(None),
        }
    };

    // Handle object reference mutation via VM heap
    if let Value::Handle(h) = cur {
        if let Some(fname) = resolve_field_name(&steps[0], &step_args[0])? {
            let hid = h.id;
            if steps.len() == 1 {
                let inst = handles
                    .get_mut(hid)
                    .ok_or_else(|| MatroError::runtime("Invalid object handle"))?;
                inst.fields.insert(fname, rhs);
                return Ok(());
            } else {
                let mut child = handles
                    .get_mut(hid)
                    .ok_or_else(|| MatroError::runtime("Invalid object handle"))?
                    .fields
                    .remove(&fname)
                    .unwrap_or_else(Value::empty);
                assign_chain_step(&mut child, &steps[1..], &step_args[1..], names, rhs, handles)?;
                if let Some(inst) = handles.get_mut(hid) {
                    inst.fields.insert(fname, child);
                }
                return Ok(());
            }
        }
    }

    if let Some(fname) = resolve_field_name(&steps[0], &step_args[0])? {
        if steps.len() == 1 {
            match cur {
                Value::Struct(map) => {
                    map.insert(fname, rhs);
                }
                Value::Object(o) => {
                    o.fields.insert(fname, rhs);
                }
                Value::StructArray(sa) if sa.items.len() == 1 => {
                    sa.items[0].insert(fname, rhs);
                }
                _ => {
                    let mut map = BTreeMap::new();
                    map.insert(fname, rhs);
                    *cur = Value::Struct(map);
                }
            }
            return Ok(());
        } else {
            match cur {
                Value::Struct(map) => {
                    let mut child = map.remove(&fname).unwrap_or_else(Value::empty);
                    assign_chain_step(&mut child, &steps[1..], &step_args[1..], names, rhs, handles)?;
                    map.insert(fname, child);
                }
                Value::Object(o) => {
                    let mut child = o.fields.remove(&fname).unwrap_or_else(Value::empty);
                    assign_chain_step(&mut child, &steps[1..], &step_args[1..], names, rhs, handles)?;
                    o.fields.insert(fname, child);
                }
                Value::StructArray(sa) if sa.items.len() == 1 => {
                    let mut child = sa.items[0].remove(&fname).unwrap_or_else(Value::empty);
                    assign_chain_step(&mut child, &steps[1..], &step_args[1..], names, rhs, handles)?;
                    sa.items[0].insert(fname, child);
                }
                _ => {
                    let mut child = Value::empty();
                    assign_chain_step(&mut child, &steps[1..], &step_args[1..], names, rhs, handles)?;
                    let mut map = BTreeMap::new();
                    map.insert(fname, child);
                    *cur = Value::Struct(map);
                }
            }
            return Ok(());
        }
    }

    match steps[0] {
        ChainStep::Field(_) | ChainStep::DynamicField => unreachable!(),
        ChainStep::Paren(_) => {
            let args = &step_args[0];
            if steps.len() == 1 {
                match cur {
                    Value::Sparse(sp) => {
                        let m_rhs = rhs.to_dense_matrix()?;
                        assign_sparse_index(std::sync::Arc::make_mut(sp), args, &m_rhs)?;
                    }
                    Value::Matrix(m) => {
                        let m_rhs = rhs.to_dense_matrix()?;
                        assign_matrix_index(std::sync::Arc::make_mut(m), args, &m_rhs)?;
                    }
                    Value::StructArray(sa) => {
                        let idx = resolve_struct_array_pos(sa, args)?;
                        if let Value::Struct(map) = rhs {
                            sa.items[idx] = map;
                        } else {
                            return Err(MatroError::runtime("Expected struct for struct array assignment"));
                        }
                    }
                    Value::Cell(c) => {
                        let item = match rhs {
                            Value::Cell(rc) if rc.data.len() == 1 => rc.data.into_iter().next().unwrap(),
                            other => other,
                        };
                        assign_cell_index(c, args, item)?;
                    }
                    _ => {
                        if let Value::Struct(map) = rhs {
                            let mut sa = StructArray { rows: 0, cols: 0, items: Vec::new() };
                            let idx = resolve_struct_array_pos(&mut sa, args)?;
                            sa.items[idx] = map;
                            *cur = Value::StructArray(sa);
                        } else {
                            let mut m = Matrix::zeros(0, 0);
                            let m_rhs = rhs.to_dense_matrix()?;
                            assign_matrix_index(&mut m, args, &m_rhs)?;
                            *cur = Value::matrix(m);
                        }
                    }
                }
                Ok(())
            } else {
                // Intermediate `(...)` step (most commonly struct array `s(k).field...`)
                match cur {
                    Value::Struct(map) => {
                        let mut sa = StructArray {
                            rows: 1,
                            cols: 1,
                            items: vec![std::mem::take(map)],
                        };
                        let idx = resolve_struct_array_pos(&mut sa, args)?;
                        let mut elem = Value::Struct(std::mem::take(&mut sa.items[idx]));
                        assign_chain_step(&mut elem, &steps[1..], &step_args[1..], names, rhs, handles)?;
                        if let Value::Struct(m) = elem {
                            sa.items[idx] = m;
                        }
                        *cur = Value::StructArray(sa);
                        Ok(())
                    }
                    Value::StructArray(sa) => {
                        let idx = resolve_struct_array_pos(sa, args)?;
                        let mut elem = Value::Struct(std::mem::take(&mut sa.items[idx]));
                        assign_chain_step(&mut elem, &steps[1..], &step_args[1..], names, rhs, handles)?;
                        if let Value::Struct(m) = elem {
                            sa.items[idx] = m;
                        }
                        Ok(())
                    }
                    Value::Cell(c) => {
                        let pos = ensure_cell_pos(c, args)?;
                        let mut child = std::mem::replace(&mut c.data[pos], Value::empty());
                        assign_chain_step(&mut child, &steps[1..], &step_args[1..], names, rhs, handles)?;
                        c.data[pos] = child;
                        Ok(())
                    }
                    _ => {
                        let mut sa = StructArray { rows: 0, cols: 0, items: Vec::new() };
                        let idx = resolve_struct_array_pos(&mut sa, args)?;
                        let mut elem = Value::Struct(std::mem::take(&mut sa.items[idx]));
                        assign_chain_step(&mut elem, &steps[1..], &step_args[1..], names, rhs, handles)?;
                        if let Value::Struct(m) = elem {
                            sa.items[idx] = m;
                        }
                        *cur = Value::StructArray(sa);
                        Ok(())
                    }
                }
            }
        }
        ChainStep::Brace(_) => {
            let args = &step_args[0];
            if !matches!(cur, Value::Cell(_)) {
                *cur = Value::Cell(CellArray::empty());
            }
            let Value::Cell(c) = cur else { unreachable!() };
            if steps.len() == 1 {
                assign_cell_index(c, args, rhs)
            } else {
                let pos = ensure_cell_pos(c, args)?;
                let mut child = std::mem::replace(&mut c.data[pos], Value::empty());
                assign_chain_step(&mut child, &steps[1..], &step_args[1..], names, rhs, handles)?;
                c.data[pos] = child;
                Ok(())
            }
        }
    }
}

fn resolve_struct_array_pos(sa: &mut StructArray, args: &[StackItem]) -> Result<usize> {
    match args.len() {
        1 => {
            let idx = args[0].clone().into_val()?.as_scalar()?.round() as isize;
            if idx < 1 {
                return Err(MatroError::index("Struct array index must be >= 1"));
            }
            let i = (idx - 1) as usize;
            if i >= sa.items.len() {
                sa.items.resize(i + 1, BTreeMap::new());
                if sa.rows == 0 {
                    sa.rows = 1;
                }
                sa.cols = sa.items.len().div_ceil(sa.rows);
            }
            Ok(i)
        }
        2 => {
            let r = args[0].clone().into_val()?.as_scalar()?.round() as isize;
            let c = args[1].clone().into_val()?.as_scalar()?.round() as isize;
            if r < 1 || c < 1 {
                return Err(MatroError::index("Struct array indices must be >= 1"));
            }
            let r = (r - 1) as usize;
            let c = (c - 1) as usize;
            let new_r = sa.rows.max(r + 1);
            let new_c = sa.cols.max(c + 1);
            if new_r != sa.rows || new_c != sa.cols {
                let mut grown = vec![BTreeMap::new(); new_r * new_c];
                for cc in 0..sa.cols {
                    for rr in 0..sa.rows {
                        grown[cc * new_r + rr] = std::mem::take(&mut sa.items[cc * sa.rows + rr]);
                    }
                }
                sa.rows = new_r;
                sa.cols = new_c;
                sa.items = grown;
            }
            Ok(c * sa.rows + r)
        }
        n => Err(MatroError::index(format!(
            "Unsupported struct array indexing with {n} indices"
        ))),
    }
}

fn ensure_cell_pos(cell: &mut CellArray, args: &[StackItem]) -> Result<usize> {
    match args.len() {
        1 => {
            let idx = args[0].clone().into_val()?.as_scalar()?.round() as isize;
            if idx < 1 {
                return Err(MatroError::index("Cell index must be >= 1"));
            }
            let i = (idx - 1) as usize;
            if i >= cell.data.len() {
                cell.data.resize(i + 1, Value::empty());
                if cell.rows == 0 {
                    cell.rows = 1;
                }
                cell.cols = cell.data.len();
                cell.extra_dims.clear();
            }
            Ok(i)
        }
        n => {
            let mut coords = Vec::with_capacity(n);
            let mut req = Vec::with_capacity(n);
            for a in args {
                let idx = a.clone().into_val()?.as_scalar()?.round() as isize;
                if idx < 1 {
                    return Err(MatroError::index("Cell indices must be >= 1"));
                }
                let c0 = (idx - 1) as usize;
                coords.push(c0);
                req.push(c0 + 1);
            }
            cell.grow_to_nd(&req);
            cell.linear_index(&coords)
                .ok_or_else(|| MatroError::index("Cell index out of bounds"))
        }
    }
}

fn assign_cell_index(cell: &mut CellArray, args: &[StackItem], rhs: Value) -> Result<()> {
    let pos = ensure_cell_pos(cell, args)?;
    cell.data[pos] = rhs;
    Ok(())
}

fn assign_matrix_index(m: &mut Matrix, args: &[StackItem], rhs: &Matrix) -> Result<()> {
    // MATLAB deletion via empty brackets `[]`: `v(mask) = []`, `A(:, 2) = []`, `A(1, :) = []`
    if rhs.is_empty() {
        match args.len() {
            1 => {
                let to_del = resolve_indices(&args[0], m.len())?;
                let mut keep = vec![true; m.len()];
                for idx in to_del {
                    if idx < keep.len() {
                        keep[idx] = false;
                    }
                }
                let orig_dtype = m.dtype();
                let vals: Vec<Complex> = (0..m.len())
                    .filter(|&i| keep[i])
                    .map(|i| m.get_linear_c(i))
                    .collect();
                *m = preserve_dtype(
                    if m.cols() == 1 && m.rows() > 1 {
                        Matrix::complex_col_vec(vals).normalize()
                    } else {
                        Matrix::complex_row_vec(vals).normalize()
                    },
                    orig_dtype,
                );
                return Ok(());
            }
            2 => {
                let orig_dtype = m.dtype();
                if matches!(args[0], StackItem::Colon) {
                    let del_cols = resolve_indices(&args[1], m.cols())?;
                    let keep_cols: Vec<usize> = (0..m.cols())
                        .filter(|c| !del_cols.contains(c))
                        .collect();
                    let mut out = Matrix::zeros(m.rows(), keep_cols.len());
                    for (nc, &oc) in keep_cols.iter().enumerate() {
                        for r in 0..m.rows() {
                            out.set_c(r, nc, m.get_c(r, oc));
                        }
                    }
                    *m = preserve_dtype(out.normalize(), orig_dtype);
                    return Ok(());
                } else if matches!(args[1], StackItem::Colon) {
                    let del_rows = resolve_indices(&args[0], m.rows())?;
                    let keep_rows: Vec<usize> = (0..m.rows())
                        .filter(|r| !del_rows.contains(r))
                        .collect();
                    let mut out = Matrix::zeros(keep_rows.len(), m.cols());
                    for c in 0..m.cols() {
                        for (nr, &or) in keep_rows.iter().enumerate() {
                            out.set_c(nr, c, m.get_c(or, c));
                        }
                    }
                    *m = preserve_dtype(out.normalize(), orig_dtype);
                    return Ok(());
                }
                return Err(MatroError::index(
                    "Null assignment `[]` can only have one non-colon subscript",
                ));
            }
            _ => {
                return Err(MatroError::index(
                    "Null assignment `[]` currently supports 1D and 2D arrays",
                ));
            }
        }
    }

    match args.len() {
        1 => {
            let max_len = m.len();
            let idxs: Vec<usize> = match &args[0] {
                StackItem::Colon => (0..max_len).collect(),
                StackItem::Scalar(x) => {
                    let i = x.round() as isize;
                    if i < 1 {
                        return Err(MatroError::index(format!("Index {x} must be >= 1")));
                    }
                    vec![(i - 1) as usize]
                }
                StackItem::Spread(_) => {
                    return Err(MatroError::index(
                        "Comma-separated list cannot be used as assignment index",
                    ));
                }
                StackItem::Void => Vec::new(),
                StackItem::Val(v) => {
                    let idx_m = v.as_matrix()?;
                    if idx_m.is_logical() {
                        idx_m
                            .data()
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &f)| (f != 0.0).then_some(i))
                            .collect()
                    } else {
                        idx_m
                            .data()
                            .iter()
                            .map(|&x| {
                                let i = x.round() as isize;
                                if i < 1 {
                                    Err(MatroError::index(format!("Index {x} must be >= 1")))
                                } else {
                                    Ok((i - 1) as usize)
                                }
                            })
                            .collect::<Result<_>>()?
                    }
                }
            };
            let needed = idxs.iter().copied().max().map(|i| i + 1).unwrap_or(0);
            if needed > m.len() {
                let orig_dtype = m.dtype();
                let was_col = m.cols() == 1 && m.rows() > 1;
                let mut vals: Vec<Complex> = (0..m.len()).map(|i| m.get_linear_c(i)).collect();
                vals.resize(needed, Complex::ZERO);
                *m = preserve_dtype(
                    if was_col {
                        Matrix::complex_col_vec(vals)
                    } else {
                        Matrix::complex_row_vec(vals)
                    },
                    orig_dtype,
                );
            }
            if let Some(s) = rhs.as_complex_scalar() {
                for &i in &idxs {
                    m.set_linear_c(i, s);
                }
            } else if rhs.len() == idxs.len() {
                for (k, &i) in idxs.iter().enumerate() {
                    m.set_linear_c(i, rhs.get_linear_c(k));
                }
            } else {
                return Err(MatroError::dimension("Indexed assignment size mismatch"));
            }
            Ok(())
        }
        2 => {
            let parse_dim = |item: &StackItem, cur: usize| -> Result<Vec<usize>> {
                match item {
                    StackItem::Colon => Ok((0..cur).collect()),
                    StackItem::Scalar(x) => {
                        let i = x.round() as isize;
                        if i < 1 {
                            Err(MatroError::index(format!("Index {x} must be >= 1")))
                        } else {
                            Ok(vec![(i - 1) as usize])
                        }
                    }
                    StackItem::Spread(_) => Err(MatroError::index(
                        "Comma-separated list cannot be used as assignment index",
                    )),
                    StackItem::Void => Ok(Vec::new()),
                    StackItem::Val(v) => {
                        let idx_m = v.as_matrix()?;
                        if idx_m.is_logical() {
                            Ok(idx_m
                                .data()
                                .iter()
                                .enumerate()
                                .filter_map(|(i, &f)| (f != 0.0).then_some(i))
                                .collect())
                        } else {
                            idx_m
                                .data()
                                .iter()
                                .map(|&x| {
                                    let i = x.round() as isize;
                                    if i < 1 {
                                        Err(MatroError::index(format!("Index {x} must be >= 1")))
                                    } else {
                                        Ok((i - 1) as usize)
                                    }
                                })
                                .collect()
                        }
                    }
                }
            };
            let rows = parse_dim(&args[0], m.rows())?;
            let cols = parse_dim(&args[1], m.cols())?;
            let new_r = m
                .rows()
                .max(rows.iter().copied().max().map(|i| i + 1).unwrap_or(0));
            let new_c = m
                .cols()
                .max(cols.iter().copied().max().map(|i| i + 1).unwrap_or(0));
            if new_r != m.rows() || new_c != m.cols() {
                let orig_dtype = m.dtype();
                let mut grown = Matrix::zeros(new_r, new_c);
                for c in 0..m.cols() {
                    for r in 0..m.rows() {
                        grown.set_c(r, c, m.get_c(r, c));
                    }
                }
                *m = preserve_dtype(grown, orig_dtype);
            }
            if let Some(s) = rhs.as_complex_scalar() {
                for &c in &cols {
                    for &r in &rows {
                        m.set_c(r, c, s);
                    }
                }
            } else if rhs.rows() == rows.len() && rhs.cols() == cols.len() {
                for (ci, &c) in cols.iter().enumerate() {
                    for (ri, &r) in rows.iter().enumerate() {
                        m.set_c(r, c, rhs.get_c(ri, ci));
                    }
                }
            } else if (rows.len() == 1 || cols.len() == 1)
                && (rhs.rows() == 1 || rhs.cols() == 1)
                && rhs.len() == rows.len() * cols.len()
            {
                let mut k = 0usize;
                for &c in &cols {
                    for &r in &rows {
                        m.set_c(r, c, rhs.get_linear_c(k));
                        k += 1;
                    }
                }
            } else {
                return Err(MatroError::dimension("Indexed assignment size mismatch"));
            }
            Ok(())
        }
        n => {
            // General N-D assignment `A(i1, i2, ..., in) = rhs`
            let mut dim_idxs = Vec::with_capacity(n);
            let mut new_dims = m.dims();
            new_dims.resize(n.max(new_dims.len()), 1);
            for (ax, arg) in args.iter().enumerate() {
                let cur = m.dim_size(ax);
                let idxs = match arg {
                    StackItem::Colon => (0..cur).collect(),
                    StackItem::Scalar(x) => {
                        let i = x.round() as isize;
                        if i < 1 {
                            return Err(MatroError::index(format!("Index {x} must be >= 1")));
                        }
                        vec![(i - 1) as usize]
                    }
                    StackItem::Spread(_) => {
                        return Err(MatroError::index(
                            "Comma-separated list cannot be used as assignment index",
                        ));
                    }
                    StackItem::Void => Vec::new(),
                    StackItem::Val(v) => {
                        let idx_m = v.to_dense_matrix()?;
                        if idx_m.is_logical() {
                            idx_m
                                .data()
                                .iter()
                                .enumerate()
                                .filter_map(|(i, &f)| (f != 0.0).then_some(i))
                                .collect()
                        } else {
                            idx_m
                                .data()
                                .iter()
                                .map(|&x| {
                                    let i = x.round() as isize;
                                    if i < 1 {
                                        Err(MatroError::index(format!("Index {x} must be >= 1")))
                                    } else {
                                        Ok((i - 1) as usize)
                                    }
                                })
                                .collect::<Result<Vec<usize>>>()?
                        }
                    }
                };
                if let Some(&mx) = idxs.iter().max() {
                    new_dims[ax] = new_dims[ax].max(mx + 1);
                }
                dim_idxs.push(idxs);
            }
            let old_dims = m.dims();
            if new_dims != old_dims {
                let orig_dtype = m.dtype();
                let mut grown = Matrix::zeros_nd(new_dims.clone());
                let new_strides = nd_strides(&new_dims);
                let mut coord = vec![0usize; old_dims.len()];
                for idx in 0..m.len() {
                    let mut dst = 0usize;
                    for ax in 0..old_dims.len() {
                        dst += coord[ax] * new_strides[ax];
                    }
                    grown.set_linear_c(dst, m.get_linear_c(idx));
                    for ax in 0..old_dims.len() {
                        coord[ax] += 1;
                        if coord[ax] < old_dims[ax] {
                            break;
                        }
                        coord[ax] = 0;
                    }
                }
                *m = preserve_dtype(grown, orig_dtype);
            }
            let strides = nd_strides(&m.dims());
            let target_lens: Vec<usize> = dim_idxs.iter().map(Vec::len).collect();
            let total: usize = target_lens.iter().product();
            if rhs.as_complex_scalar().is_none() && rhs.len() != total {
                return Err(MatroError::dimension("N-D indexed assignment size mismatch"));
            }
            let mut coord = vec![0usize; n];
            for k in 0..total {
                let mut dst = 0usize;
                for ax in 0..n {
                    dst += dim_idxs[ax][coord[ax]] * strides[ax];
                }
                let val = rhs
                    .as_complex_scalar()
                    .unwrap_or_else(|| rhs.get_linear_c(k));
                m.set_linear_c(dst, val);
                for ax in 0..n {
                    coord[ax] += 1;
                    if coord[ax] < target_lens[ax] {
                        break;
                    }
                    coord[ax] = 0;
                }
            }
            Ok(())
        }
    }
}

fn format_assignment(stdout: &mut String, name: &str, val: &Value) {
    match val {
        Value::Matrix(m) if !m.is_scalar() && !m.is_empty() => {
            stdout.push_str(&format!("{name} =\n{m}\n"));
        }
        Value::Sparse(_) | Value::Struct(_) | Value::Object(_) | Value::Cell(_) => {
            stdout.push_str(&format!("{name} =\n{val}\n"));
        }
        _ => {
            stdout.push_str(&format!("{name} = {val}\n"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_system_and_functions() {
        let mut vm = Vm::new();
        let out = vm
            .eval(
                r#"
                A = [3 1; 1 2];
                b = [9; 8];
                x = A \ b;

                function [s, p] = stats(v)
                    s = sum(v);
                    p = prod(v);
                end

                [total, product] = stats(1:5);
                "#,
            )
            .unwrap();
        assert!(out.stdout.is_empty());
        let x = vm.get_var("x").unwrap().as_matrix().unwrap();
        assert!((x.get(0, 0) - 2.0).abs() < 1e-10);
        assert!((x.get(1, 0) - 3.0).abs() < 1e-10);
        assert_eq!(vm.get_var("total").unwrap().as_scalar().unwrap(), 15.0);
        assert_eq!(vm.get_var("product").unwrap().as_scalar().unwrap(), 120.0);
    }

    #[test]
    fn line_numbered_errors_and_profiler() {
        let mut vm = Vm::new();
        let err = vm.eval("a = 1;\nb = 2;\nc = [1 2] * [1 2];\n").unwrap_err();
        let msg = err.to_string();
        assert!(msg.starts_with("Line 3:"), "unexpected error: {msg}");

        let out = vm
            .eval(
                r#"
                profile('on');
                function y = sq(x)
                    y = x * x;
                end
                tic;
                s = 0;
                for k = 1:20
                    s = s + sq(k);
                end
                dt = toc;
                profile('report');
                "#,
            )
            .unwrap();
        assert!(out.stdout.contains("sq"));
        assert!(vm.get_var("dt").unwrap().as_scalar().unwrap() >= 0.0);
    }

    #[test]
    fn sparse_matfile_and_classdef() {
        let mut vm = Vm::new();
        vm.eval(
            r#"
            % Sparse 1D Poisson system (40x40 tridiagonal)
            n = 40;
            i = [1:n, 1:(n-1), 2:n];
            j = [1:n, 2:n, 1:(n-1)];
            v = [2*ones(1, n), -ones(1, n-1), -ones(1, n-1)];
            S = sparse(i, j, v, n, n);
            assert(issparse(S) && nnz(S) == 3*n - 2);
            b = ones(n, 1);
            x = S \ b;
            assert(norm(S * x - b) < 1e-8);

            % Binary .mat v5 save & load
            M = [1 2; 3 4];
            z = 2 + 5i;
            save('snapshot.mat', 'M', 'z');
            clear_m = 0;
            loaded = load('snapshot.mat');
            assert(norm(loaded.M - M) < 1e-12);
            assert(abs(loaded.z - z) < 1e-12);

            % classdef OOP
            classdef Vec2
                properties
                    x = 0
                    y = 0
                end
                methods
                    function obj = Vec2(x, y)
                        obj.x = x;
                        obj.y = y;
                    end
                    function r = len(obj)
                        r = sqrt(obj.x^2 + obj.y^2);
                    end
                    function out = scale(obj, k)
                        out = Vec2(obj.x * k, obj.y * k);
                    end
                end
            end

            p = Vec2(3, 4);
            assert(isa(p, 'Vec2') && strcmp(class(p), 'Vec2'));
            assert(abs(p.len() - 5) < 1e-12);
            assert(abs(len(p) - 5) < 1e-12);
            p2 = p.scale(2);
            assert(p2.x == 6 && p2.y == 8);
            "#,
        )
        .unwrap();
    }
}
