use matro_types::Span;

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub statements: Vec<Stmt>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
    /// True if terminated by `;` (suppresses automatic output in MATLAB/Octave).
    pub suppressed: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StmtKind {
    Expr(Expr),
    Assign {
        target: AssignTarget,
        value: Expr,
    },
    MultiAssign {
        targets: Vec<String>,
        value: Expr,
    },
    If {
        cond: Expr,
        then_body: Vec<Stmt>,
        elseifs: Vec<(Expr, Vec<Stmt>)>,
        else_body: Option<Vec<Stmt>>,
    },
    Switch {
        expr: Expr,
        cases: Vec<(Expr, Vec<Stmt>)>,
        otherwise: Option<Vec<Stmt>>,
    },
    TryCatch {
        try_body: Vec<Stmt>,
        catch_var: Option<String>,
        catch_body: Vec<Stmt>,
    },
    While {
        cond: Expr,
        body: Vec<Stmt>,
    },
    For {
        var: String,
        iter: Expr,
        body: Vec<Stmt>,
    },
    Function {
        name: String,
        params: Vec<String>,
        returns: Vec<String>,
        body: Vec<Stmt>,
    },
    ClassDef {
        name: String,
        superclasses: Vec<String>,
        properties: Vec<PropDef>,
        methods: Vec<MethodDef>,
        enumerations: Vec<(String, Vec<Expr>)>,
        events: Vec<String>,
    },
    Arguments {
        repeating: bool,
        entries: Vec<ArgSpec>,
    },
    Global(Vec<String>),
    Persistent(Vec<String>),
    Break,
    Continue,
    Return,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PropDef {
    pub name: String,
    pub default: Option<Expr>,
    pub dependent: bool,
    pub constant: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArgSpec {
    pub target: String,          // e.g. "x" or "opts"
    pub field: Option<String>,   // e.g. Some("Color") for opts.Color
    pub class_name: Option<String>,
    pub validators: Vec<(String, Vec<Expr>)>, // e.g. ("mustBePositive", []) or ("mustBeMember", [allowed])
    pub default: Option<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MethodDef {
    pub name: String,
    pub params: Vec<String>,
    pub returns: Vec<String>,
    pub body: Vec<Stmt>,
    pub is_static: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LValueStep {
    Field(String),
    DynamicField(Expr),
    Paren(Vec<IndexArg>),
    Brace(Vec<IndexArg>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum AssignTarget {
    Var(String),
    Index { name: String, indices: Vec<IndexArg> },
    CellIndex { name: String, indices: Vec<IndexArg> },
    Field { name: String, fields: Vec<String> },
    IndexedField {
        name: String,
        indices: Vec<IndexArg>,
        fields: Vec<String>,
    },
    /// Indexing into a struct/object field, e.g. `obj.Data(idx) = val`.
    FieldIndex {
        name: String,
        fields: Vec<String>,
        indices: Vec<IndexArg>,
    },
    /// Arbitrary-depth chained L-value target, e.g. `s(2).arr{1}(3, :) = val` or `c{1}(2) = val`.
    Chain {
        name: String,
        steps: Vec<LValueStep>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    Number(f64),
    Bool(bool),
    ImagNumber(f64),
    String(String),
    Ident(String),
    EndKeyword,
    Matrix(Vec<Vec<Expr>>),
    Cell(Vec<Vec<Expr>>),
    AnonFunc {
        params: Vec<String>,
        body: Box<Expr>,
    },
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Range {
        start: Box<Expr>,
        step: Option<Box<Expr>>,
        stop: Box<Expr>,
    },
    /// Hermitian / conjugate transpose (`'`).
    CTranspose(Box<Expr>),
    /// Non-conjugate transpose (`.'`).
    Transpose(Box<Expr>),
    /// In MATLAB, `f(a, b)` is syntactically ambiguous between a function call and matrix indexing.
    CallOrIndex {
        callee: String,
        args: Vec<IndexArg>,
    },
    /// General postfix call/index on an arbitrary expression (e.g., `s.fn(x)` or `c{1}(2)`).
    DynamicCall {
        target: Box<Expr>,
        args: Vec<IndexArg>,
    },
    /// Cell content indexing `c{1, 2}`.
    CellIndex {
        target: Box<Expr>,
        indices: Vec<IndexArg>,
    },
    /// Struct field or object property access `s.field`.
    FieldAccess {
        object: Box<Expr>,
        field: String,
    },
    /// Dynamic struct/object field access `s.(expr)`.
    DynamicFieldAccess {
        object: Box<Expr>,
        field_expr: Box<Expr>,
    },
    /// Dot-method or field-closure call `obj.method(args...)`.
    MethodCall {
        object: Box<Expr>,
        method: String,
        args: Vec<IndexArg>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum IndexArg {
    Expr(Expr),
    Colon,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Pos,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    ElemMul,
    Div,
    ElemDiv,
    LeftDiv,
    ElemLeftDiv,
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
    ShortAnd,
    ShortOr,
}
