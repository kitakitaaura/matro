use std::fmt;

use matro_types::Value;

/// Compact stack-based bytecode instructions for the Matro VM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpCode {
    LoadConst(u32),
    LoadVar(u32),
    StoreVar(u32),
    LoadLocal(u16),
    StoreLocal(u16),
    LoadGlobal(u32),
    StoreGlobal(u32),
    InitGlobal(u32),
    LoadPersistent(u32),
    StorePersistent(u32),
    InitPersistent(u32),
    Pop,
    Dup,

    PrintVar(u32),
    PrintLocal(u16),
    PrintAns,

    BuildRow(u32),
    BuildMatrix(u32),
    BuildCell { rows: u16, cols: u16 },
    MakeRange { has_step: bool },

    Neg,
    Pos,
    Not,
    CTranspose,
    Transpose,

    Add,
    Sub,
    Mul,
    ElemMul,
    Div,
    ElemDiv,
    LeftDiv,
    Pow,
    ElemPow,

    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,

    Jump(u32),
    JumpIfFalse(u32),
    JumpIfFalsePeek(u32),
    JumpIfTruePeek(u32),
    SwitchMatch,

    TryEnter { catch_ip: u32, err_var: u32, err_is_local: bool },
    TryExit,

    ForInit { var_idx: u32, exit_ip: u32 },
    ForNext { var_idx: u32, loop_ip: u32 },
    ForInitLocal { slot: u16, exit_ip: u32 },
    ForNextLocal { slot: u16, loop_ip: u32 },
    ForRangeInit { var_idx: u32, exit_ip: u32, has_step: bool },
    ForRangeInitLocal { slot: u16, exit_ip: u32, has_step: bool },
    ForPop,

    PushColon,
    PushEndTarget { total_dims: u8 },
    PushEndFieldTarget { field_idx: u32, total_dims: u8 },
    PushEndVarTarget { var_idx: u32, total_dims: u8 },
    PushEndLocalTarget { slot: u16, total_dims: u8 },
    SetEndDim(u8),
    PushEndTop,
    PopEndTarget,

    CallOrIndex { name_idx: u32, argc: u8, nargout: u8 },
    CallOrIndexLocal { slot: u16, argc: u8, nargout: u8 },
    DynamicCall { argc: u8, nargout: u8 },
    MethodCall { method_idx: u32, argc: u8, nargout: u8 },
    StoreIndex { name_idx: u32, argc: u8 },
    StoreIndexLocal { slot: u16, argc: u8 },

    CellIndex { argc: u8, nargout: u8 },
    StoreCellIndex { name_idx: u32, argc: u8 },
    StoreCellIndexLocal { slot: u16, argc: u8 },

    GetField(u32),
    StoreField { name_idx: u32, path_idx: u32 },
    StoreFieldLocal { slot: u16, path_idx: u32 },
    StoreIndexedField { name_idx: u32, argc: u8, path_idx: u32 },
    StoreIndexedFieldLocal { slot: u16, argc: u8, path_idx: u32 },

    DefineFunc(u32),
    DefineNestedFunc { func_idx: u32, slot: u16 },
    DefineClass(u32),
    MakeClosure(u32),
    Return,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompiledArgSpec {
    pub target: String,
    pub target_slot: Option<u16>,
    pub field: Option<String>,
    pub repeating: bool,
    pub class_name: Option<String>,
    pub validators: Vec<(String, Vec<Chunk>)>,
    pub default: Option<Chunk>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionProto {
    pub name: String,
    pub params: Vec<String>,
    pub returns: Vec<String>,
    /// Slot indices for `params` (in order).
    pub param_slots: Vec<u16>,
    /// Slot indices for `returns` (in order).
    pub return_slots: Vec<u16>,
    /// Optional slots for `nargin`, `nargout`, `varargin`, `varargout`.
    pub nargin_slot: Option<u16>,
    pub nargout_slot: Option<u16>,
    pub varargin_slot: Option<u16>,
    pub varargout_slot: Option<u16>,
    /// True if this function defines nested functions that share its mutable frame env.
    pub has_nested: bool,
    /// Optional MATLAB `arguments ... end` validation & default specifications.
    pub arg_specs: Vec<CompiledArgSpec>,
    /// Names of all local slots `0..local_names.len()` (for disassembly & closure capture).
    pub local_names: Vec<String>,
    pub chunk: Chunk,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PropProto {
    pub name: String,
    pub default: Option<Chunk>,
    pub dependent: bool,
    pub constant: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClassProto {
    pub name: String,
    pub superclasses: Vec<String>,
    pub properties: Vec<PropProto>,
    pub methods: Vec<FunctionProto>,
    pub static_methods: Vec<String>,
    pub enumerations: Vec<(String, Vec<Chunk>)>,
    pub events: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Chunk {
    pub code: Vec<OpCode>,
    /// Parallel 1-based source line number for each instruction in `code`.
    pub lines: Vec<u32>,
    pub constants: Vec<Value>,
    pub names: Vec<String>,
    pub functions: Vec<FunctionProto>,
    pub classes: Vec<ClassProto>,
    pub field_paths: Vec<Vec<String>>,
}

impl Chunk {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn emit(&mut self, op: OpCode, line: u32) -> usize {
        let idx = self.code.len();
        self.code.push(op);
        self.lines.push(line);
        idx
    }

    pub fn line_at(&self, ip: usize) -> u32 {
        self.lines.get(ip).copied().unwrap_or(0)
    }

    pub fn add_const(&mut self, val: Value) -> u32 {
        self.constants.push(val);
        (self.constants.len() - 1) as u32
    }

    pub fn add_name(&mut self, name: &str) -> u32 {
        if let Some(pos) = self.names.iter().position(|n| n == name) {
            return pos as u32;
        }
        self.names.push(name.to_string());
        (self.names.len() - 1) as u32
    }

    pub fn add_function(&mut self, proto: FunctionProto) -> u32 {
        self.functions.push(proto);
        (self.functions.len() - 1) as u32
    }

    pub fn add_class(&mut self, cls: ClassProto) -> u32 {
        self.classes.push(cls);
        (self.classes.len() - 1) as u32
    }

    pub fn add_field_path(&mut self, path: Vec<String>) -> u32 {
        self.field_paths.push(path);
        (self.field_paths.len() - 1) as u32
    }

    /// Format a human-readable disassembly of this bytecode chunk (with source line numbers).
    pub fn disassemble(&self, label: &str) -> String {
        self.disassemble_with_locals(label, &[])
    }

    pub fn disassemble_with_locals(&self, label: &str, local_names: &[String]) -> String {
        let mut out = String::new();
        out.push_str(&format!("== {label} ==\n"));
        let local_label = |s: u16| -> &str {
            local_names
                .get(s as usize)
                .map(String::as_str)
                .unwrap_or("?")
        };
        for (ip, op) in self.code.iter().enumerate() {
            let ln = self.line_at(ip);
            let line = match *op {
                OpCode::LoadConst(i) => format!("LOAD_CONST   {i} ({})", self.constants[i as usize]),
                OpCode::LoadVar(i) => format!("LOAD_VAR     {i} ({})", self.names[i as usize]),
                OpCode::StoreVar(i) => format!("STORE_VAR    {i} ({})", self.names[i as usize]),
                OpCode::LoadLocal(s) => format!("LOAD_LOCAL   {s} ({})", local_label(s)),
                OpCode::StoreLocal(s) => format!("STORE_LOCAL  {s} ({})", local_label(s)),
                OpCode::PrintVar(i) => format!("PRINT_VAR    {i} ({})", self.names[i as usize]),
                OpCode::PrintLocal(s) => format!("PRINT_LOCAL  {s} ({})", local_label(s)),
                OpCode::GetField(i) => format!("GET_FIELD    .{}", self.names[i as usize]),
                OpCode::StoreField { name_idx, path_idx } => format!(
                    "STORE_FIELD  {}.{}",
                    self.names[name_idx as usize],
                    self.field_paths[path_idx as usize].join(".")
                ),
                OpCode::StoreFieldLocal { slot, path_idx } => format!(
                    "STORE_FLD_L  {}.{}",
                    local_label(slot),
                    self.field_paths[path_idx as usize].join(".")
                ),
                OpCode::CallOrIndex {
                    name_idx,
                    argc,
                    nargout,
                } => format!(
                    "CALL_OR_IDX  {} (argc={argc}, nargout={nargout})",
                    self.names[name_idx as usize]
                ),
                OpCode::CallOrIndexLocal { slot, argc, nargout } => format!(
                    "CALL_IDX_L   {} (argc={argc}, nargout={nargout})",
                    local_label(slot)
                ),
                OpCode::MethodCall {
                    method_idx,
                    argc,
                    nargout,
                } => format!(
                    "METHOD_CALL  .{} (argc={argc}, nargout={nargout})",
                    self.names[method_idx as usize]
                ),
                OpCode::DynamicCall { argc, nargout } => {
                    format!("DYN_CALL     (argc={argc}, nargout={nargout})")
                }
                OpCode::StoreIndex { name_idx, argc } => {
                    format!("STORE_IDX    {} (argc={argc})", self.names[name_idx as usize])
                }
                OpCode::StoreIndexLocal { slot, argc } => {
                    format!("STORE_IDX_L  {} (argc={argc})", local_label(slot))
                }
                OpCode::StoreCellIndex { name_idx, argc } => {
                    format!("STORE_CELL   {} (argc={argc})", self.names[name_idx as usize])
                }
                OpCode::StoreCellIndexLocal { slot, argc } => {
                    format!("STORE_CELL_L {} (argc={argc})", local_label(slot))
                }
                OpCode::DefineFunc(i) => {
                    format!("DEFINE_FUNC  {i} ({})", self.functions[i as usize].name)
                }
                OpCode::DefineClass(i) => {
                    format!("DEFINE_CLASS {i} ({})", self.classes[i as usize].name)
                }
                OpCode::MakeClosure(i) => {
                    let f = &self.functions[i as usize];
                    format!("MAKE_CLOSURE {i} (@({}))", f.params.join(","))
                }
                OpCode::ForInit { var_idx, exit_ip } => format!(
                    "FOR_INIT     {} -> exit {exit_ip:04}",
                    self.names[var_idx as usize]
                ),
                OpCode::ForNext { var_idx, loop_ip } => format!(
                    "FOR_NEXT     {} -> loop {loop_ip:04}",
                    self.names[var_idx as usize]
                ),
                OpCode::ForInitLocal { slot, exit_ip } => {
                    format!("FOR_INIT_L   {} -> exit {exit_ip:04}", local_label(slot))
                }
                OpCode::ForNextLocal { slot, loop_ip } => {
                    format!("FOR_NEXT_L   {} -> loop {loop_ip:04}", local_label(slot))
                }
                other => format!("{other:?}"),
            };
            out.push_str(&format!("{ip:04} [L{ln:<3}] {line}\n"));
        }
        for func in &self.functions {
            out.push('\n');
            out.push_str(
                &func
                    .chunk
                    .disassemble_with_locals(&format!("fn {}", func.name), &func.local_names),
            );
        }
        for cls in &self.classes {
            for m in &cls.methods {
                out.push('\n');
                out.push_str(&m.chunk.disassemble_with_locals(
                    &format!("method {}.{}", cls.name, m.name),
                    &m.local_names,
                ));
            }
        }
        out
    }
}

impl fmt::Display for Chunk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.disassemble("main"))
    }
}
