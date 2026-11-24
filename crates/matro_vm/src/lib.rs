use std::collections::{BTreeMap, HashMap};

use matro_bytecode::{Chunk, ClassProto, FunctionProto, OpCode, compile};
use matro_intrinsics::{IntrinsicContext, call_intrinsic};
use matro_loader::{ModuleLoader, Vfs};
use matro_types::plot::json_str;
use matro_types::{
    CellArray, ClosureHandle, Complex, HandleRef, MapData, MatroError, Matrix, ObjectInstance,
    PlotFigure, Result, Rng, SparseMatrix, StructArray, TableData, Value,
};

mod fileio;
mod solvers;

/// Result of evaluating a script or REPL line.
#[derive(Debug, Clone, Default)]
pub struct EvalOutput {
    pub stdout: String,
    pub last_value: Option<Value>,
}

/// Internal stack slot (regular value, fast unboxed scalar, `:` sentinel, or `c{:}` spread).
#[derive(Debug, Clone)]
enum StackItem {
    Val(Value),
    Scalar(f64),
    Colon,
    Spread(Vec<Value>),
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
        }
    }
}

#[derive(Debug, Clone)]
enum ForLoopState {
    Matrix { matrix: Matrix, col_idx: usize },
    Range { start: f64, step: f64, idx: usize, count: usize },
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
}

/// Fast indexed local frame for functions & closures (no HashMap per call!).
struct LocalFrame<'a> {
    func_name: &'a str,
    slots: Vec<Option<Value>>,
    names: &'a [String],
    captures: Option<&'a BTreeMap<String, Value>>,
    env_id: Option<usize>,
}

impl<'a> LocalFrame<'a> {
    fn new(
        func_name: &'a str,
        names: &'a [String],
        captures: Option<&'a BTreeMap<String, Value>>,
        env_id: Option<usize>,
    ) -> Self {
        Self {
            func_name,
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

/// Embeddable, pure-Rust MATLAB/Octave bytecode Virtual Machine.
#[derive(Debug)]
pub struct Vm {
    globals: HashMap<String, Value>,
    persistents: HashMap<(String, String), Value>,
    functions: HashMap<String, FunctionProto>,
    classes: HashMap<String, ClassProto>,
    handles: Vec<ObjectInstance>,
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
            handles: Vec::new(),
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

    pub fn mount_file(&mut self, path: &str, content: &str) -> Result<()> {
        self.loader.mount_file(path, content)
    }

    pub fn mount_bytes(&mut self, path: &str, data: &[u8]) -> Result<()> {
        self.loader.mount_bytes(path, data)
    }

    pub fn add_path(&mut self, dir: &str) {
        self.loader.add_path(dir);
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
        self.handles.clear();
        self.closures.clear();
        self.envs.clear();
        self.figure.clear();
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
                    if let Some(handler) = try_stack.pop() {
                        stack.truncate(handler.stack_depth);
                        for_loops.truncate(handler.loop_depth);
                        if handler.err_var != u32::MAX {
                            let mut info = BTreeMap::new();
                            info.insert("message".to_string(), Value::String(err.bare_message()));
                            info.insert(
                                "identifier".to_string(),
                                Value::String(err.identifier().to_string()),
                            );
                            info.insert("line".to_string(), Value::scalar(line as f64));
                            let err_val = Value::Struct(info);
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
                } else if let Some(res) = call_intrinsic(
                    &name,
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
                    stack.push(StackItem::Val(
                        vals.into_iter().next().unwrap_or_else(Value::empty),
                    ));
                } else if let Some(func) = self.functions.get(&name).cloned() {
                    let mut rets = self.call_user_func(&func, &[], 1, None, stdout)?;
                    stack.push(StackItem::Val(rets.pop().unwrap_or_else(Value::empty)));
                } else if let Some(cls) = self.classes.get(&name).cloned() {
                    let obj = self.instantiate_class(&cls, &[], stdout)?;
                    stack.push(StackItem::Val(obj));
                } else if self.try_autoload_symbol(&name, stdout)? {
                    if let Some(func) = self.functions.get(&name).cloned() {
                        let mut rets = self.call_user_func(&func, &[], 1, None, stdout)?;
                        stack.push(StackItem::Val(rets.pop().unwrap_or_else(Value::empty)));
                    } else if let Some(cls) = self.classes.get(&name).cloned() {
                        let obj = self.instantiate_class(&cls, &[], stdout)?;
                        stack.push(StackItem::Val(obj));
                    } else {
                        return Err(MatroError::undefined(name));
                    }
                } else {
                    return Err(MatroError::undefined(name));
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
            OpCode::BuildRow(n) => {
                let raw = pop_items(stack, n as usize)?;
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
                } else {
                    let mats: Vec<Matrix> = items
                        .iter()
                        .map(|v| v.to_dense_matrix())
                        .collect::<Result<_>>()?;
                    let refs: Vec<&Matrix> = mats.iter().collect();
                    let row = Matrix::hcat(&refs)?;
                    stack.push(StackItem::Val(Value::matrix(row)));
                }
            }
            OpCode::BuildMatrix(n) => {
                let items = pop_vals(stack, n as usize)?;
                if items.len() == 1 && matches!(items[0], Value::String(_)) {
                    stack.push(StackItem::Val(items.into_iter().next().unwrap()));
                } else {
                    let mats: Vec<Matrix> = items
                        .iter()
                        .map(|v| v.to_dense_matrix())
                        .collect::<Result<_>>()?;
                    let refs: Vec<&Matrix> = mats.iter().collect();
                    let mat = Matrix::vcat(&refs)?;
                    stack.push(StackItem::Val(Value::matrix(mat)));
                }
            }
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
                    let v = pop_val(stack)?;
                    let out = match v {
                        Value::Sparse(s) => Value::sparse(s.scale(-1.0)),
                        other => Value::matrix(other.as_matrix()?.map_c(|c| -c)),
                    };
                    stack.push(StackItem::Val(out));
                }
            }
            OpCode::Pos => {
                // unary `+` is a no-op on numeric scalars/matrices
            }
            OpCode::Not => {
                let m = pop_val(stack)?
                    .to_dense_matrix()?
                    .map_c_to_real(|c| if c.is_zero() { 1.0 } else { 0.0 })
                    .with_logical(true);
                stack.push(StackItem::Val(Value::matrix(m)));
            }
            OpCode::CTranspose => {
                if matches!(stack.last(), Some(StackItem::Scalar(_))) {
                    // real scalar conjugate transpose is itself
                } else {
                    let v = pop_val(stack)?;
                    let out = match v {
                        Value::Sparse(s) => Value::sparse(s.transpose()),
                        other => Value::matrix(other.as_matrix()?.ctranspose()),
                    };
                    stack.push(StackItem::Val(out));
                }
            }
            OpCode::Transpose => {
                if matches!(stack.last(), Some(StackItem::Scalar(_))) {
                    // real scalar transpose is itself
                } else {
                    let v = pop_val(stack)?;
                    let out = match v {
                        Value::Sparse(s) => Value::sparse(s.transpose()),
                        other => Value::matrix(other.as_matrix()?.transpose()),
                    };
                    stack.push(StackItem::Val(out));
                }
            }
            OpCode::Add => {
                let b_item = stack.pop().ok_or_else(|| MatroError::runtime("VM stack underflow"))?;
                let a_item = stack.pop().ok_or_else(|| MatroError::runtime("VM stack underflow"))?;
                if let (Some(x), Some(y)) = (a_item.as_fast_scalar(), b_item.as_fast_scalar()) {
                    stack.push(StackItem::Scalar(x + y));
                } else {
                    let a = a_item.into_val()?;
                    let b = b_item.into_val()?;
                    let out = match (&a, &b) {
                        (Value::Sparse(sa), Value::Sparse(sb)) => Value::sparse(sa.add_scaled(sb, 1.0)?),
                        _ => Value::matrix(
                            a.to_dense_matrix()?
                                .elementwise_c(&b.to_dense_matrix()?, |x, y| x + y)?,
                        ),
                    };
                    stack.push(StackItem::Val(out));
                }
            }
            OpCode::Sub => {
                let b_item = stack.pop().ok_or_else(|| MatroError::runtime("VM stack underflow"))?;
                let a_item = stack.pop().ok_or_else(|| MatroError::runtime("VM stack underflow"))?;
                if let (Some(x), Some(y)) = (a_item.as_fast_scalar(), b_item.as_fast_scalar()) {
                    stack.push(StackItem::Scalar(x - y));
                } else {
                    let a = a_item.into_val()?;
                    let b = b_item.into_val()?;
                    let out = match (&a, &b) {
                        (Value::Sparse(sa), Value::Sparse(sb)) => {
                            Value::sparse(sa.add_scaled(sb, -1.0)?)
                        }
                        _ => Value::matrix(
                            a.to_dense_matrix()?
                                .elementwise_c(&b.to_dense_matrix()?, |x, y| x - y)?,
                        ),
                    };
                    stack.push(StackItem::Val(out));
                }
            }
            OpCode::ElemMul => {
                let b_item = stack.pop().ok_or_else(|| MatroError::runtime("VM stack underflow"))?;
                let a_item = stack.pop().ok_or_else(|| MatroError::runtime("VM stack underflow"))?;
                if let (Some(x), Some(y)) = (a_item.as_fast_scalar(), b_item.as_fast_scalar()) {
                    stack.push(StackItem::Scalar(x * y));
                } else {
                    let a = a_item.into_val()?;
                    let b = b_item.into_val()?;
                    let out = match (&a, &b) {
                        (Value::Sparse(sa), Value::Sparse(sb)) => Value::sparse(sa.elem_mul(sb)?),
                        (Value::Sparse(sa), Value::Matrix(mb)) if mb.as_scalar().is_some() => {
                            Value::sparse(sa.scale(mb.as_scalar().unwrap()))
                        }
                        (Value::Matrix(ma), Value::Sparse(sb)) if ma.as_scalar().is_some() => {
                            Value::sparse(sb.scale(ma.as_scalar().unwrap()))
                        }
                        _ => Value::matrix(
                            a.to_dense_matrix()?
                                .elementwise_c(&b.to_dense_matrix()?, |x, y| x * y)?,
                        ),
                    };
                    stack.push(StackItem::Val(out));
                }
            }
            OpCode::ElemDiv => {
                let b_item = stack.pop().ok_or_else(|| MatroError::runtime("VM stack underflow"))?;
                let a_item = stack.pop().ok_or_else(|| MatroError::runtime("VM stack underflow"))?;
                if let (Some(x), Some(y)) = (a_item.as_fast_scalar(), b_item.as_fast_scalar()) {
                    stack.push(StackItem::Scalar(x / y));
                } else {
                    let a = a_item.into_val()?;
                    let b = b_item.into_val()?;
                    let res = a.to_dense_matrix()?.elementwise_c(&b.to_dense_matrix()?, |x, y| x / y)?;
                    stack.push(StackItem::Val(Value::matrix(res)));
                }
            }
            OpCode::ElemPow => bin_elem_c(stack, Complex::powc)?,
            OpCode::Mul => {
                let b_item = stack.pop().ok_or_else(|| MatroError::runtime("VM stack underflow"))?;
                let a_item = stack.pop().ok_or_else(|| MatroError::runtime("VM stack underflow"))?;
                if let (Some(x), Some(y)) = (a_item.as_fast_scalar(), b_item.as_fast_scalar()) {
                    stack.push(StackItem::Scalar(x * y));
                } else {
                    let a = a_item.into_val()?;
                    let b = b_item.into_val()?;
                    let out = match (&a, &b) {
                        (Value::Sparse(sa), Value::Sparse(sb)) => Value::sparse(sa.matmul_sparse(sb)?),
                        (Value::Sparse(sa), Value::Matrix(mb)) => {
                            if let Some(sc) = mb.as_scalar() {
                                Value::sparse(sa.scale(sc))
                            } else {
                                Value::matrix(sa.matmul_dense(mb)?)
                            }
                        }
                        (Value::Matrix(ma), Value::Sparse(sb)) => {
                            if let Some(sc) = ma.as_scalar() {
                                Value::sparse(sb.scale(sc))
                            } else {
                                Value::matrix(ma.matmul(&sb.to_dense())?)
                            }
                        }
                        _ => Value::matrix(a.as_matrix()?.matmul(b.as_matrix()?)?),
                    };
                    stack.push(StackItem::Val(out));
                }
            }
            OpCode::Div => {
                let b_item = stack.pop().ok_or_else(|| MatroError::runtime("VM stack underflow"))?;
                let a_item = stack.pop().ok_or_else(|| MatroError::runtime("VM stack underflow"))?;
                if let (Some(x), Some(y)) = (a_item.as_fast_scalar(), b_item.as_fast_scalar()) {
                    stack.push(StackItem::Scalar(x / y));
                } else {
                    let a = a_item.into_val()?;
                    let b = b_item.into_val()?;
                    let out = match (&a, &b) {
                        (Value::Sparse(sa), Value::Matrix(mb)) if mb.as_scalar().is_some() => {
                            Value::sparse(sa.scale(1.0 / mb.as_scalar().unwrap()))
                        }
                        _ => Value::matrix(a.to_dense_matrix()?.rdivide(&b.to_dense_matrix()?)?),
                    };
                    stack.push(StackItem::Val(out));
                }
            }
            OpCode::LeftDiv => {
                let b = pop_val(stack)?;
                let a = pop_val(stack)?;
                let out = match (&a, &b) {
                    (Value::Sparse(sa), _) => Value::matrix(sa.solve_dense(&b.to_dense_matrix()?)?),
                    _ => Value::matrix(a.to_dense_matrix()?.solve(&b.to_dense_matrix()?)?),
                };
                stack.push(StackItem::Val(out));
            }
            OpCode::Pow => {
                let b = pop_val(stack)?;
                let a = pop_val(stack)?;
                let res = a.to_dense_matrix()?.matpow(&b.to_dense_matrix()?)?;
                stack.push(StackItem::Val(Value::matrix(res)));
            }
            OpCode::Eq => {
                let b = pop_val(stack)?;
                let a = pop_val(stack)?;
                let res = match (&a, &b) {
                    (Value::String(sa), Value::String(sb)) => Value::bool(sa == sb),
                    _ => Value::matrix(
                        a.to_dense_matrix()?
                            .elementwise_c(&b.to_dense_matrix()?, |x, y| {
                                Complex::from_real(if x == y { 1.0 } else { 0.0 })
                            })?
                            .with_logical(true),
                    ),
                };
                stack.push(StackItem::Val(res));
            }
            OpCode::Ne => {
                let b = pop_val(stack)?;
                let a = pop_val(stack)?;
                let res = match (&a, &b) {
                    (Value::String(sa), Value::String(sb)) => Value::bool(sa != sb),
                    _ => Value::matrix(
                        a.to_dense_matrix()?
                            .elementwise_c(&b.to_dense_matrix()?, |x, y| {
                                Complex::from_real(if x != y { 1.0 } else { 0.0 })
                            })?
                            .with_logical(true),
                    ),
                };
                stack.push(StackItem::Val(res));
            }
            OpCode::Lt => bin_logical(stack, |a, b| a < b)?,
            OpCode::Le => bin_logical(stack, |a, b| a <= b)?,
            OpCode::Gt => bin_logical(stack, |a, b| a > b)?,
            OpCode::Ge => bin_logical(stack, |a, b| a >= b)?,
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
                });
            }
            OpCode::TryExit => {
                try_stack.pop();
            }
            OpCode::ForInit { var_idx, exit_ip } => {
                let iter_mat = pop_val(stack)?.to_dense_matrix()?;
                if iter_mat.cols() == 0 {
                    *ip = exit_ip as usize;
                } else {
                    let col = extract_col(&iter_mat, 0);
                    let name = chunk.names[var_idx as usize].clone();
                    store_var(
                        name,
                        Value::matrix(col),
                        frame.as_deref_mut(),
                        &mut self.globals,
                    );
                    for_loops.push(ForLoopState::Matrix {
                        matrix: iter_mat,
                        col_idx: 1,
                    });
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
                    ForLoopState::Matrix { matrix, col_idx } => {
                        if *col_idx < matrix.cols() {
                            let col = extract_col(matrix, *col_idx);
                            *col_idx += 1;
                            Some(Value::matrix(col))
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
                let iter_mat = pop_val(stack)?.to_dense_matrix()?;
                if iter_mat.cols() == 0 {
                    *ip = exit_ip as usize;
                } else {
                    let col = extract_col(&iter_mat, 0);
                    let f = frame
                        .as_deref_mut()
                        .ok_or_else(|| MatroError::runtime("ForInitLocal outside function"))?;
                    f.set(slot, Value::matrix(col));
                    for_loops.push(ForLoopState::Matrix {
                        matrix: iter_mat,
                        col_idx: 1,
                    });
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
                    ForLoopState::Matrix { matrix, col_idx } => {
                        if *col_idx < matrix.cols() {
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
                }
            }
            OpCode::ForPop => {
                for_loops.pop();
            }
            OpCode::PushColon => {
                stack.push(StackItem::Colon);
            }
            OpCode::PushEnd {
                var_idx,
                dim,
                total_dims,
            } => {
                let name = &chunk.names[var_idx as usize];
                let val = lookup_var(name, frame.as_deref(), &self.globals)
                    .ok_or_else(|| MatroError::undefined(name))?;
                stack.push(StackItem::Val(Value::scalar(end_for_value(val, dim, total_dims) as f64)));
            }
            OpCode::PushEndLocal {
                slot,
                dim,
                total_dims,
            } => {
                let f = frame
                    .as_deref()
                    .ok_or_else(|| MatroError::runtime("PushEndLocal outside function"))?;
                let val = f
                    .get(slot)
                    .ok_or_else(|| MatroError::undefined(&f.names[slot as usize]))?;
                stack.push(StackItem::Val(Value::scalar(end_for_value(val, dim, total_dims) as f64)));
            }
            OpCode::CallOrIndexLocal {
                slot,
                argc,
                nargout,
            } => {
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

                // 0. Built-in namespace / native handle / map methods
                if let Value::Object(ref o) = receiver {
                    if o.class_name == "__ns_containers" && method_name == "Map" {
                        let call_args = flatten_call_args(raw_args)?;
                        let map_val = matro_intrinsics::tables::build_map(&call_args)?;
                        push_call_results(stack, vec![map_val], nargout as usize);
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
                    if let Some(proto) = self.find_class_method(&cname, &method_name) {
                        let expanded = flatten_call_args(raw_args)?;
                        let mut call_args = Vec::with_capacity(expanded.len() + 1);
                        call_args.push(receiver);
                        call_args.extend(expanded);
                        let results =
                            self.call_user_func(&proto, &call_args, nargout as usize, None, stdout)?;
                        push_call_results(stack, results, nargout as usize);
                        return Ok(true);
                    }
                }

                // 2. Otherwise fall back to field lookup (e.g. `s.fn(x)` or `obj.vec(1)`)
                let field_val = match &receiver {
                    Value::Struct(map) => map.get(&method_name).cloned(),
                    Value::Object(obj) => obj.fields.get(&method_name).cloned(),
                    Value::Handle(h) => self
                        .handles
                        .get(h.id)
                        .and_then(|o| o.fields.get(&method_name).cloned()),
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
                let rhs = pop_val(stack)?;
                let raw_args = pop_items(stack, argc as usize)?;
                let name = chunk.names[name_idx as usize].clone();
                let mut val = lookup_var(&name, frame.as_deref(), &self.globals)
                    .cloned()
                    .unwrap_or_else(Value::empty);
                remove_var(&name, frame.as_deref_mut(), &mut self.globals);
                if let Value::Map(ref mut map) = val {
                    if raw_args.len() != 1 {
                        return Err(MatroError::index("containers.Map indexing expects 1 key"));
                    }
                    let key = MapData::key_of(&raw_args[0].clone().into_val()?)?;
                    map.entries.insert(key, rhs);
                } else {
                    assign_matrix_index(val.as_matrix_mut()?, &raw_args, &rhs.to_dense_matrix()?)?;
                }
                *last_value = Some(val.clone());
                store_var(name, val, frame.as_deref_mut(), &mut self.globals);
            }
            OpCode::StoreIndexLocal { slot, argc } => {
                let rhs = pop_val(stack)?;
                let raw_args = pop_items(stack, argc as usize)?;
                let f = frame
                    .as_deref_mut()
                    .ok_or_else(|| MatroError::runtime("StoreIndexLocal outside function"))?;
                let mut val = f.take(slot).unwrap_or_else(Value::empty);
                if let Value::Map(ref mut map) = val {
                    if raw_args.len() != 1 {
                        return Err(MatroError::index("containers.Map indexing expects 1 key"));
                    }
                    let key = MapData::key_of(&raw_args[0].clone().into_val()?)?;
                    map.entries.insert(key, rhs);
                } else {
                    assign_matrix_index(val.as_matrix_mut()?, &raw_args, &rhs.to_dense_matrix()?)?;
                }
                *last_value = Some(val.clone());
                f.set(slot, val);
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
                let field = &chunk.names[field_idx as usize];
                let obj = pop_val(stack)?;
                let val = match obj {
                    Value::Struct(map) => map.get(field).cloned(),
                    Value::Object(inst) => inst.fields.get(field).cloned(),
                    Value::Handle(h) => self
                        .handles
                        .get(h.id)
                        .and_then(|inst| inst.fields.get(field).cloned()),
                    Value::StructArray(sa) => {
                        let items: Vec<Value> = sa
                            .items
                            .iter()
                            .map(|m| m.get(field).cloned().unwrap_or_else(Value::empty))
                            .collect();
                        if items.iter().all(|v| v.as_scalar().is_ok()) {
                            let nums: Vec<f64> =
                                items.iter().map(|v| v.as_scalar().unwrap()).collect();
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
                            t.get_col(field).cloned()
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
                .ok_or_else(|| MatroError::runtime(format!("No field or property '{field}'")))?;
                stack.push(StackItem::Val(val));
            }
            OpCode::StoreField { name_idx, path_idx } => {
                let rhs = pop_val(stack)?;
                let name = chunk.names[name_idx as usize].clone();
                let path = &chunk.field_paths[path_idx as usize];
                let mut root = lookup_var(&name, frame.as_deref(), &self.globals)
                    .cloned()
                    .unwrap_or_else(|| Value::Struct(BTreeMap::new()));
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
                let f = frame
                    .as_deref_mut()
                    .ok_or_else(|| MatroError::runtime("StoreFieldLocal outside function"))?;
                let mut root = f
                    .take(slot)
                    .unwrap_or_else(|| Value::Struct(BTreeMap::new()));
                if let Value::Handle(ref h) = root {
                    let inst = self
                        .handles
                        .get_mut(h.id)
                        .ok_or_else(|| MatroError::runtime("Invalid object handle"))?;
                    set_nested_field_map(&mut inst.fields, path, rhs)?;
                    *last_value = Some(root.clone());
                    f.set(slot, root);
                } else {
                    set_nested_field(&mut root, path, rhs)?;
                    *last_value = Some(root.clone());
                    f.set(slot, root);
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
            OpCode::DefineFunc(idx) => {
                let proto = chunk.functions[idx as usize].clone();
                self.functions.insert(proto.name.clone(), proto);
            }
            OpCode::DefineNestedFunc { func_idx, slot } => {
                let proto = chunk.functions[func_idx as usize].clone();
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
                let cls = chunk.classes[idx as usize].clone();
                self.classes.insert(cls.name.clone(), cls);
            }
            OpCode::MakeClosure(idx) => {
                let proto = chunk.functions[idx as usize].clone();
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
            Value::Sparse(s) => {
                let indexed = index_sparse(&s, &raw_args)?;
                stack.push(StackItem::Val(Value::matrix(indexed)));
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
            other => {
                return Err(MatroError::runtime(format!(
                    "Cannot call or index value of type {}",
                    other.type_name()
                )));
            }
        }
        Ok(())
    }

    fn is_handle_class(&self, class_name: &str) -> bool {
        let mut cur = Some(class_name);
        let mut guard = 0;
        while let Some(name) = cur {
            if name == "handle" {
                return true;
            }
            guard += 1;
            if guard > 32 {
                break;
            }
            cur = self
                .classes
                .get(name)
                .and_then(|c| c.superclass.as_deref());
        }
        false
    }

    fn isa_class(&self, val: &Value, expected: &str) -> bool {
        let mut cur = match val {
            Value::Object(o) => Some(o.class_name.as_str()),
            Value::Handle(h) => Some(h.class_name.as_str()),
            _ => None,
        };
        let mut guard = 0;
        while let Some(name) = cur {
            if name == expected {
                return true;
            }
            guard += 1;
            if guard > 32 {
                break;
            }
            cur = self
                .classes
                .get(name)
                .and_then(|c| c.superclass.as_deref());
        }
        match expected {
            "numeric" | "double" => {
                matches!(val, Value::Matrix(m) if !m.is_logical()) || matches!(val, Value::Sparse(_))
            }
            _ => val.type_name() == expected,
        }
    }

    fn find_class_method(&self, class_name: &str, method_name: &str) -> Option<FunctionProto> {
        let mut cur = Some(class_name);
        let mut guard = 0;
        while let Some(name) = cur {
            if let Some(cls) = self.classes.get(name) {
                if let Some(m) = cls.methods.iter().find(|m| m.name == method_name) {
                    return Some(m.clone());
                }
                cur = cls.superclass.as_deref();
            } else {
                break;
            }
            guard += 1;
            if guard > 32 {
                break;
            }
        }
        None
    }

    fn collect_class_hierarchy(&self, class_name: &str) -> Vec<ClassProto> {
        let mut chain = Vec::new();
        let mut cur = Some(class_name.to_string());
        let mut guard = 0;
        while let Some(name) = cur {
            if let Some(cls) = self.classes.get(&name) {
                cur = cls.superclass.clone();
                chain.push(cls.clone());
            } else {
                break;
            }
            guard += 1;
            if guard > 32 {
                break;
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
            for (prop_name, default_chunk) in &c.properties {
                let val = if let Some(ch) = default_chunk {
                    let mut tmp_val = None;
                    let prev = self.globals.remove("__prop_default");
                    self.run_frame(ch, None, stdout, &mut tmp_val)?;
                    let v = self
                        .globals
                        .remove("__prop_default")
                        .unwrap_or_else(Value::empty);
                    if let Some(p) = prev {
                        self.globals.insert("__prop_default".to_string(), p);
                    }
                    v
                } else {
                    Value::empty()
                };
                fields.insert(prop_name.clone(), val);
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

            let mut f = LocalFrame::new(&ctor.name, &ctor.local_names, None, None);
            if let Some(&ret_slot) = ctor.return_slots.first() {
                f.set(ret_slot, obj);
            }
            let results = self.exec_user_func_with_frame(&ctor, &ctor_args, 1, f, stdout)?;
            return Ok(results.into_iter().next().unwrap_or_else(Value::empty));
        }
        Ok(obj)
    }

    fn try_autoload_symbol(&mut self, name: &str, stdout: &mut String) -> Result<bool> {
        let Some(mod_chunk) = self.loader.load_module(name)? else {
            return Ok(false);
        };
        let mut ignored = None;
        self.run_frame(&mod_chunk, None, stdout, &mut ignored)?;
        Ok(self.functions.contains_key(name) || self.classes.contains_key(name))
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
        mut frame: Option<&mut LocalFrame<'_>>,
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
                if call_args.len() != 1 {
                    return Err(MatroError::arity("load", "1 (filename.mat)", call_args.len()));
                }
                let path = call_args[0].as_str()?;
                let vars = self.loader.load_mat(path)?;
                for (k, v) in &vars {
                    store_var(k.clone(), v.clone(), frame.as_deref_mut(), &mut self.globals);
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
            "arrayfun" => {
                if call_args.len() != 2 {
                    return Err(MatroError::arity("arrayfun", "2", call_args.len()));
                }
                let Value::Closure(cl) = &call_args[0] else {
                    return Err(MatroError::runtime(
                        "arrayfun: first argument must be a function handle",
                    ));
                };
                let m = call_args[1].to_dense_matrix()?;
                let mut out = Matrix::zeros(m.rows(), m.cols());
                for c in 0..m.cols() {
                    for r in 0..m.rows() {
                        let elem = Value::matrix(Matrix::complex_scalar(m.get_c(r, c)));
                        let rets = self.call_closure(cl, &[elem], 1, stdout)?;
                        let res = rets
                            .into_iter()
                            .next()
                            .unwrap_or_else(Value::empty)
                            .as_matrix()?
                            .as_complex_scalar()
                            .ok_or_else(|| {
                                MatroError::dimension("arrayfun: callback must return scalar")
                            })?;
                        out.set_c(r, c, res);
                    }
                }
                return Ok(vec![Value::matrix(out.normalize())]);
            }
            "fopen" => return self.files.fopen(call_args, nargout, &self.loader),
            "fclose" => return self.files.fclose(call_args, &mut self.loader),
            "fprintf" => return self.files.fprintf(call_args, nargout, stdout),
            "fgetl" => return self.files.fgetl(call_args, false),
            "fgets" => return self.files.fgetl(call_args, true),
            "feof" => return self.files.feof(call_args),
            "ftell" => return self.files.ftell(call_args),
            "frewind" => return self.files.frewind(call_args),
            "fwrite" => return self.files.fwrite(call_args),
            "fread" => return self.files.fread(call_args, nargout),
            "textscan" => return self.files.textscan(call_args),
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
            "cellfun" => {
                if call_args.len() < 2 {
                    return Err(MatroError::arity("cellfun", ">= 2 (fn, C)", call_args.len()));
                }
                let Value::Cell(c) = &call_args[1] else {
                    return Err(MatroError::runtime("cellfun: second argument must be a cell array"));
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
                let mut results = Vec::with_capacity(c.data.len());
                for item in &c.data {
                    let v = self.invoke_callable_val(&call_args[0], &[item.clone()], stdout)?;
                    results.push(v);
                }
                if uniform {
                    let mut out = Matrix::zeros(c.rows, c.cols);
                    for (idx, v) in results.iter().enumerate() {
                        let sc = v.as_matrix()?.as_complex_scalar().ok_or_else(|| {
                            MatroError::runtime(
                                "cellfun: non-scalar output; pass 'UniformOutput', false",
                            )
                        })?;
                        out.set_linear_c(idx, sc);
                    }
                    return Ok(vec![Value::matrix(out.normalize())]);
                } else {
                    return Ok(vec![Value::Cell(CellArray {
                        rows: c.rows,
                        cols: c.cols,
                        data: results,
                    })]);
                }
            }
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
                    for v in map.values() {
                        let r = self.invoke_callable_val(&call_args[0], &[v.clone()], stdout)?;
                        let sc = r.as_matrix()?.as_complex_scalar().ok_or_else(|| {
                            MatroError::runtime(
                                "structfun: non-scalar output; pass 'UniformOutput', false",
                            )
                        })?;
                        vals.push(sc);
                    }
                    return Ok(vec![Value::matrix(Matrix::complex_col_vec(vals).normalize())]);
                } else {
                    let mut out = BTreeMap::new();
                    for (k, v) in map {
                        let r = self.invoke_callable_val(&call_args[0], &[v.clone()], stdout)?;
                        out.insert(k.clone(), r);
                    }
                    return Ok(vec![Value::Struct(out)]);
                }
            }
            "bsxfun" => {
                if call_args.len() != 3 {
                    return Err(MatroError::arity("bsxfun", "3 (fn, A, B)", call_args.len()));
                }
                let a = call_args[1].to_dense_matrix()?;
                let b = call_args[2].to_dense_matrix()?;
                let out_r = a.rows().max(b.rows());
                let out_c = a.cols().max(b.cols());
                if (a.rows() != out_r && a.rows() != 1)
                    || (b.rows() != out_r && b.rows() != 1)
                    || (a.cols() != out_c && a.cols() != 1)
                    || (b.cols() != out_c && b.cols() != 1)
                {
                    return Err(MatroError::dimension(
                        "bsxfun: incompatible matrix dimensions for singleton expansion",
                    ));
                }
                let mut out = Matrix::zeros(out_r, out_c);
                for c in 0..out_c {
                    let ac = if a.cols() == 1 { 0 } else { c };
                    let bc = if b.cols() == 1 { 0 } else { c };
                    for r in 0..out_r {
                        let ar = if a.rows() == 1 { 0 } else { r };
                        let br = if b.rows() == 1 { 0 } else { r };
                        let va = Value::matrix(Matrix::complex_scalar(a.get_c(ar, ac)).normalize());
                        let vb = Value::matrix(Matrix::complex_scalar(b.get_c(br, bc)).normalize());
                        let res = self.invoke_callable_val(&call_args[0], &[va, vb], stdout)?;
                        let sc = res.as_matrix()?.as_complex_scalar().ok_or_else(|| {
                            MatroError::dimension("bsxfun: callback must return scalar")
                        })?;
                        out.set_c(r, c, sc);
                    }
                }
                return Ok(vec![Value::matrix(out.normalize())]);
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
            "ode45" => {
                if call_args.len() != 3 {
                    return Err(MatroError::arity("ode45", "3 (f, tspan, y0)", call_args.len()));
                }
                let Value::Closure(cl) = &call_args[0] else {
                    return Err(MatroError::runtime(
                        "ode45: first argument must be a function handle @(t, y) ...",
                    ));
                };
                let tspan = call_args[1].as_matrix()?.clone();
                let y0 = call_args[2].as_matrix()?.clone();
                let (t_mat, y_mat) = solvers::solve_ode45(self, cl, &tspan, &y0, stdout)?;
                return Ok(vec![Value::matrix(t_mat), Value::matrix(y_mat)]);
            }
            "ode15s" => {
                if call_args.len() != 3 {
                    return Err(MatroError::arity("ode15s", "3 (f, tspan, y0)", call_args.len()));
                }
                let Value::Closure(cl) = &call_args[0] else {
                    return Err(MatroError::runtime(
                        "ode15s: first argument must be a function handle @(t, y) ...",
                    ));
                };
                let tspan = call_args[1].as_matrix()?.clone();
                let y0 = call_args[2].as_matrix()?.clone();
                let (t_mat, y_mat) = solvers::solve_ode15s(self, cl, &tspan, &y0, stdout)?;
                return Ok(vec![Value::matrix(t_mat), Value::matrix(y_mat)]);
            }
            "fzero" => {
                if call_args.len() != 2 {
                    return Err(MatroError::arity("fzero", "2 (f, x0)", call_args.len()));
                }
                let Value::Closure(cl) = &call_args[0] else {
                    return Err(MatroError::runtime(
                        "fzero: first argument must be a function handle",
                    ));
                };
                let guess = call_args[1].as_matrix()?.clone();
                let root = solvers::solve_fzero(self, cl, &guess, stdout)?;
                return Ok(vec![Value::scalar(root)]);
            }
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
            "fminsearch" => {
                if call_args.len() != 2 {
                    return Err(MatroError::arity("fminsearch", "2 (f, x0)", call_args.len()));
                }
                let Value::Closure(cl) = &call_args[0] else {
                    return Err(MatroError::runtime(
                        "fminsearch: first argument must be a function handle",
                    ));
                };
                let x0 = call_args[1].as_matrix()?.clone();
                let (best_x, fval) = solvers::solve_fminsearch(self, cl, &x0, stdout)?;
                if nargout >= 2 {
                    return Ok(vec![Value::matrix(best_x), Value::scalar(fval)]);
                }
                return Ok(vec![Value::matrix(best_x)]);
            }
            "fsolve" => {
                if call_args.len() != 2 {
                    return Err(MatroError::arity("fsolve", "2 (f, x0)", call_args.len()));
                }
                let Value::Closure(cl) = &call_args[0] else {
                    return Err(MatroError::runtime(
                        "fsolve: first argument must be a function handle",
                    ));
                };
                let x0 = call_args[1].as_matrix()?.clone();
                let (sol, _resnorm, residual) = solvers::solve_lm(self, cl, &x0, stdout)?;
                if nargout >= 2 {
                    return Ok(vec![Value::matrix(sol), Value::matrix(residual)]);
                }
                return Ok(vec![Value::matrix(sol)]);
            }
            "lsqnonlin" => {
                if call_args.len() != 2 {
                    return Err(MatroError::arity("lsqnonlin", "2 (f, x0)", call_args.len()));
                }
                let Value::Closure(cl) = &call_args[0] else {
                    return Err(MatroError::runtime(
                        "lsqnonlin: first argument must be a function handle",
                    ));
                };
                let x0 = call_args[1].as_matrix()?.clone();
                let (sol, resnorm, residual) = solvers::solve_lm(self, cl, &x0, stdout)?;
                let mut out = vec![Value::matrix(sol)];
                if nargout >= 2 {
                    out.push(Value::scalar(resnorm));
                }
                if nargout >= 3 {
                    out.push(Value::matrix(residual));
                }
                return Ok(out);
            }
            "integral" => {
                if call_args.len() != 3 {
                    return Err(MatroError::arity("integral", "3 (f, a, b)", call_args.len()));
                }
                let Value::Closure(cl) = &call_args[0] else {
                    return Err(MatroError::runtime(
                        "integral: first argument must be a function handle",
                    ));
                };
                let a = call_args[1].as_scalar()?;
                let b = call_args[2].as_scalar()?;
                let q = solvers::solve_integral(self, cl, a, b, stdout)?;
                return Ok(vec![Value::scalar(q)]);
            }
            "run" => {
                if call_args.len() != 1 {
                    return Err(MatroError::arity("run", "1", call_args.len()));
                }
                let path = call_args[0].as_str()?;
                let Some(script_chunk) = self.loader.load_module(path)? else {
                    return Err(MatroError::runtime(format!("Script not found: '{path}'")));
                };
                let mut ignored = None;
                self.run_frame(&script_chunk, frame.as_deref_mut(), stdout, &mut ignored)?;
                return Ok(vec![]);
            }
            "exist" => {
                if call_args.is_empty() {
                    return Err(MatroError::arity("exist", "1", 0));
                }
                let q = call_args[0].as_str()?;
                let code = if lookup_var(q, frame.as_deref(), &self.globals).is_some() {
                    1.0
                } else if self.classes.contains_key(q) {
                    8.0
                } else if self.functions.contains_key(q) || self.loader.resolve_path(q).is_some() {
                    2.0
                } else {
                    0.0
                };
                return Ok(vec![Value::scalar(code)]);
            }
            _ => {}
        }

        // Method dispatch when first argument is a `classdef` Object or Handle (MATLAB syntax `method(obj, ...)`)
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
            if let Some(proto) = self.find_class_method(&cname, name) {
                return self.call_user_func(&proto, call_args, nargout, None, stdout);
            }
        }

        if let Some(func) = self.functions.get(name).cloned() {
            self.call_user_func(&func, call_args, nargout, None, stdout)
        } else if let Some(cls) = self.classes.get(name).cloned() {
            Ok(vec![self.instantiate_class(&cls, call_args, stdout)?])
        } else if let Some(res) = call_intrinsic(
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
        } else if self.try_autoload_symbol(name, stdout)? {
            if let Some(func) = self.functions.get(name).cloned() {
                self.call_user_func(&func, call_args, nargout, None, stdout)
            } else if let Some(cls) = self.classes.get(name).cloned() {
                Ok(vec![self.instantiate_class(&cls, call_args, stdout)?])
            } else {
                Err(MatroError::undefined(name))
            }
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
        let env_id = cl.env_id.or_else(|| {
            if proto.has_nested {
                let eid = self.envs.len();
                self.envs.push(BTreeMap::new());
                Some(eid)
            } else {
                None
            }
        });
        let frame = LocalFrame::new(&proto.name, &proto.local_names, Some(&cl.captures), env_id);
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
        let frame = LocalFrame::new(&proto.name, &proto.local_names, captures, env_id);
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

        if let Some(s) = proto.nargin_slot {
            frame.set(s, Value::scalar(args.len() as f64));
        }
        if let Some(s) = proto.nargout_slot {
            frame.set(s, Value::scalar(nargout as f64));
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
}

fn end_for_value(val: &Value, dim: u8, total_dims: u8) -> usize {
    match val {
        Value::Matrix(m) => {
            if total_dims == 1 {
                m.len()
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
            } else if dim == 0 {
                c.rows
            } else {
                c.cols
            }
        }
        Value::Table(t) => {
            if dim == 0 {
                t.nrows
            } else {
                t.col_names.len()
            }
        }
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
        .elementwise(&b.to_dense_matrix()?, |x, y| if f(x, y) { 1.0 } else { 0.0 })?
        .with_logical(true);
    stack.push(StackItem::Val(Value::matrix(res)));
    Ok(())
}

fn bin_elem_c(stack: &mut Vec<StackItem>, f: impl Fn(Complex, Complex) -> Complex) -> Result<()> {
    let b = pop_val(stack)?;
    let a = pop_val(stack)?;
    let res = a
        .to_dense_matrix()?
        .elementwise_c(&b.to_dense_matrix()?, f)?;
    stack.push(StackItem::Val(Value::matrix(res)));
    Ok(())
}

fn extract_col(m: &Matrix, col: usize) -> Matrix {
    if m.rows() == 1 {
        Matrix::complex_scalar(m.get_c(0, col))
    } else {
        let vals = (0..m.rows()).map(|r| m.get_c(r, col)).collect();
        Matrix::complex_col_vec(vals)
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
        StackItem::Val(v) => {
            let m = v.as_matrix()?;
            if m.is_logical() {
                if m.len() > max_len {
                    return Err(MatroError::index(format!(
                        "Logical index length {} exceeds dimension {}",
                        m.len(),
                        max_len
                    )));
                }
                let mut out = Vec::new();
                for (i, &flag) in m.data().iter().enumerate() {
                    if flag != 0.0 {
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

fn index_sparse(s: &SparseMatrix, args: &[StackItem]) -> Result<Matrix> {
    if args.len() == 2 {
        let rows = resolve_indices(&args[0], s.rows())?;
        let cols = resolve_indices(&args[1], s.cols())?;
        let mut out = Matrix::zeros(rows.len(), cols.len());
        for (ci, &c) in cols.iter().enumerate() {
            for (ri, &r) in rows.iter().enumerate() {
                out.set(ri, ci, s.get(r, c));
            }
        }
        return Ok(out);
    }
    index_matrix(&s.to_dense(), args)
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
    match args.len() {
        1 => {
            let idxs = resolve_indices(&args[0], m.len())?;
            let vals: Vec<Complex> = idxs.iter().map(|&i| m.get_linear_c(i)).collect();
            if m.rows() == 1 && m.ndims() == 2 && !matches!(args[0], StackItem::Colon) {
                Ok(Matrix::complex_row_vec(vals).normalize())
            } else {
                Ok(Matrix::complex_col_vec(vals).normalize())
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
            Ok(out.normalize())
        }
        n => {
            // General N-D indexing `A(i1, i2, ..., in)`
            let mut dim_idxs = Vec::with_capacity(n);
            let mut out_dims = Vec::with_capacity(n);
            for (ax, arg) in args.iter().enumerate() {
                let dlen = m.dim_size(ax);
                let idxs = resolve_indices(arg, dlen)?;
                out_dims.push(idxs.len());
                dim_idxs.push(idxs);
            }
            let in_dims = m.dims();
            let in_strides = nd_strides(&in_dims);
            let total: usize = out_dims.iter().product();
            let mut re = Vec::with_capacity(total);
            let mut im = Vec::with_capacity(total);
            let mut coord = vec![0usize; n];
            for _ in 0..total {
                let mut src = 0usize;
                for ax in 0..n {
                    let c = dim_idxs[ax][coord[ax]];
                    if ax < in_strides.len() {
                        src += c * in_strides[ax];
                    }
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
            Matrix::from_nd_complex_col_major(out_dims, re, im).map(Matrix::normalize)
        }
    }
}

fn index_cell_contents(cell: &CellArray, args: &[StackItem]) -> Result<Vec<Value>> {
    match args.len() {
        1 => {
            let idxs = resolve_indices(&args[0], cell.data.len())?;
            Ok(idxs.iter().map(|&i| cell.data[i].clone()).collect())
        }
        2 => {
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
        n => Err(MatroError::index(format!(
            "Only 1D and 2D cell indexing supported, got {n} indices"
        ))),
    }
}

fn index_cell_subarray(cell: &CellArray, args: &[StackItem]) -> Result<CellArray> {
    match args.len() {
        1 => {
            let idxs = resolve_indices(&args[0], cell.data.len())?;
            let data: Vec<Value> = idxs.iter().map(|&i| cell.data[i].clone()).collect();
            let len = data.len();
            Ok(CellArray {
                rows: if len == 0 { 0 } else { 1 },
                cols: len,
                data,
            })
        }
        2 => {
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
        n => Err(MatroError::index(format!(
            "Only 1D and 2D cell indexing supported, got {n} indices"
        ))),
    }
}

fn assign_cell_index(cell: &mut CellArray, args: &[StackItem], rhs: Value) -> Result<()> {
    match args.len() {
        1 => {
            let idx = args[0].clone().into_val()?.as_scalar()?.round() as isize;
            if idx < 1 {
                return Err(MatroError::index("Cell index must be >= 1"));
            }
            let i = (idx - 1) as usize;
            if i >= cell.data.len() {
                cell.data.resize(i + 1, Value::empty());
                cell.rows = 1;
                cell.cols = cell.data.len();
            }
            cell.data[i] = rhs;
            Ok(())
        }
        2 => {
            let r = args[0].clone().into_val()?.as_scalar()?.round() as isize;
            let c = args[1].clone().into_val()?.as_scalar()?.round() as isize;
            if r < 1 || c < 1 {
                return Err(MatroError::index("Cell indices must be >= 1"));
            }
            let r = (r - 1) as usize;
            let c = (c - 1) as usize;
            let new_r = cell.rows.max(r + 1);
            let new_c = cell.cols.max(c + 1);
            if new_r != cell.rows || new_c != cell.cols {
                let mut grown = CellArray::new_empty_cells(new_r, new_c);
                for cc in 0..cell.cols {
                    for rr in 0..cell.rows {
                        grown.set(rr, cc, cell.get(rr, cc).clone());
                    }
                }
                *cell = grown;
            }
            cell.set(r, c, rhs);
            Ok(())
        }
        n => Err(MatroError::index(format!(
            "Only 1D and 2D cell assignment supported, got {n} indices"
        ))),
    }
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
                let vals: Vec<Complex> = (0..m.len())
                    .filter(|&i| keep[i])
                    .map(|i| m.get_linear_c(i))
                    .collect();
                *m = if m.cols() == 1 && m.rows() > 1 {
                    Matrix::complex_col_vec(vals).normalize()
                } else {
                    Matrix::complex_row_vec(vals).normalize()
                };
                return Ok(());
            }
            2 => {
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
                    *m = out.normalize();
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
                    *m = out.normalize();
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
                let mut vals: Vec<Complex> = (0..m.len()).map(|i| m.get_linear_c(i)).collect();
                vals.resize(needed, Complex::ZERO);
                *m = Matrix::complex_row_vec(vals);
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
                let mut grown = Matrix::zeros(new_r, new_c);
                for c in 0..m.cols() {
                    for r in 0..m.rows() {
                        grown.set_c(r, c, m.get_c(r, c));
                    }
                }
                *m = grown;
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
                    StackItem::Val(v) => v
                        .as_matrix()?
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
                        .collect::<Result<Vec<usize>>>()?,
                };
                if let Some(&mx) = idxs.iter().max() {
                    new_dims[ax] = new_dims[ax].max(mx + 1);
                }
                dim_idxs.push(idxs);
            }
            let old_dims = m.dims();
            if new_dims != old_dims {
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
                *m = grown;
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
