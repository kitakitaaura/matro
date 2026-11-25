use std::collections::{HashMap, HashSet};

use matro_parser::{
    AssignTarget, BinaryOp, Expr, ExprKind, IndexArg, LValueStep, Program, Stmt, StmtKind, UnaryOp,
};
use matro_types::{MatroError, Result, Value};

use crate::op::{ChainStep, Chunk, ClassProto, CompiledArgSpec, FunctionProto, OpCode, PropProto};

struct LoopContext {
    is_for: bool,
    continue_target: Option<usize>,
    continue_patches: Vec<usize>,
    break_patches: Vec<usize>,
}

pub struct Compiler {
    chunk: Chunk,
    loops: Vec<LoopContext>,
    locals: Option<HashMap<String, u16>>,
    local_names: Vec<String>,
    globals_declared: HashSet<String>,
    persistents_declared: HashSet<String>,
    outer_locals: HashSet<String>,
    has_nested: bool,
}

impl Compiler {
    pub fn compile_program(program: &Program) -> Result<Chunk> {
        let mut compiler = Self {
            chunk: Chunk::new(),
            loops: Vec::new(),
            locals: None,
            local_names: Vec::new(),
            globals_declared: HashSet::new(),
            persistents_declared: HashSet::new(),
            outer_locals: HashSet::new(),
            has_nested: false,
        };
        // Hoist top-level function and classdef definitions (MATLAB R2016b+ allows local functions at bottom of scripts)
        for stmt in &program.statements {
            if matches!(stmt.kind, StmtKind::Function { .. } | StmtKind::ClassDef { .. }) {
                compiler.compile_stmt(stmt)?;
            }
        }
        for stmt in &program.statements {
            if !matches!(stmt.kind, StmtKind::Function { .. } | StmtKind::ClassDef { .. }) {
                compiler.compile_stmt(stmt)?;
            }
        }
        compiler.chunk.emit(OpCode::Return, 0);
        Ok(compiler.chunk)
    }

    fn alloc_local(&mut self, name: &str) -> u16 {
        if let Some(map) = &mut self.locals {
            if let Some(&slot) = map.get(name) {
                return slot;
            }
            let slot = self.local_names.len() as u16;
            self.local_names.push(name.to_string());
            map.insert(name.to_string(), slot);
            slot
        } else {
            unreachable!("alloc_local called outside function scope")
        }
    }

    fn lookup_local(&self, name: &str) -> Option<u16> {
        if self.globals_declared.contains(name) || self.persistents_declared.contains(name) {
            return None;
        }
        self.locals.as_ref().and_then(|m| m.get(name).copied())
    }

    fn collect_scope_decls(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            match &s.kind {
                StmtKind::Global(names) => {
                    for n in names {
                        self.globals_declared.insert(n.clone());
                    }
                }
                StmtKind::Persistent(names) => {
                    for n in names {
                        self.persistents_declared.insert(n.clone());
                    }
                }
                StmtKind::If {
                    then_body,
                    elseifs,
                    else_body,
                    ..
                } => {
                    self.collect_scope_decls(then_body);
                    for (_, b) in elseifs {
                        self.collect_scope_decls(b);
                    }
                    if let Some(eb) = else_body {
                        self.collect_scope_decls(eb);
                    }
                }
                StmtKind::Switch {
                    cases, otherwise, ..
                } => {
                    for (_, b) in cases {
                        self.collect_scope_decls(b);
                    }
                    if let Some(ob) = otherwise {
                        self.collect_scope_decls(ob);
                    }
                }
                StmtKind::TryCatch {
                    try_body,
                    catch_body,
                    ..
                } => {
                    self.collect_scope_decls(try_body);
                    self.collect_scope_decls(catch_body);
                }
                StmtKind::While { body, .. } | StmtKind::For { body, .. } => {
                    self.collect_scope_decls(body);
                }
                _ => {}
            }
        }
    }

    fn collect_locals_in_stmts(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            match &s.kind {
                StmtKind::Assign { target, .. } => {
                    let n = match target {
                        AssignTarget::Var(n)
                        | AssignTarget::Index { name: n, .. }
                        | AssignTarget::CellIndex { name: n, .. }
                        | AssignTarget::Field { name: n, .. }
                        | AssignTarget::IndexedField { name: n, .. }
                        | AssignTarget::FieldIndex { name: n, .. }
                        | AssignTarget::Chain { name: n, .. } => n,
                    };
                    if matches!(target, AssignTarget::FieldIndex { .. }) {
                        self.alloc_local("__matro_fi_tmp");
                    }
                    if !self.globals_declared.contains(n)
                        && !self.persistents_declared.contains(n)
                        && !self.outer_locals.contains(n)
                    {
                        self.alloc_local(n);
                    }
                }
                StmtKind::MultiAssign { targets, .. } => {
                    for t in targets {
                        if t != "~"
                            && !self.globals_declared.contains(t)
                            && !self.persistents_declared.contains(t)
                            && !self.outer_locals.contains(t)
                        {
                            self.alloc_local(t);
                        }
                    }
                }
                StmtKind::If {
                    then_body,
                    elseifs,
                    else_body,
                    ..
                } => {
                    self.collect_locals_in_stmts(then_body);
                    for (_, b) in elseifs {
                        self.collect_locals_in_stmts(b);
                    }
                    if let Some(eb) = else_body {
                        self.collect_locals_in_stmts(eb);
                    }
                }
                StmtKind::Switch {
                    cases, otherwise, ..
                } => {
                    for (_, b) in cases {
                        self.collect_locals_in_stmts(b);
                    }
                    if let Some(ob) = otherwise {
                        self.collect_locals_in_stmts(ob);
                    }
                }
                StmtKind::TryCatch {
                    try_body,
                    catch_var,
                    catch_body,
                } => {
                    self.collect_locals_in_stmts(try_body);
                    if let Some(v) = catch_var {
                        if !self.globals_declared.contains(v)
                            && !self.persistents_declared.contains(v)
                        {
                            self.alloc_local(v);
                        }
                    }
                    self.collect_locals_in_stmts(catch_body);
                }
                StmtKind::While { body, .. } => self.collect_locals_in_stmts(body),
                StmtKind::For { var, body, .. } => {
                    if !self.globals_declared.contains(var)
                        && !self.persistents_declared.contains(var)
                        && !self.outer_locals.contains(var)
                    {
                        self.alloc_local(var);
                    }
                    self.collect_locals_in_stmts(body);
                }
                StmtKind::Function { name, .. } => {
                    self.has_nested = true;
                    self.alloc_local(name);
                }
                StmtKind::Arguments { entries, .. } => {
                    for e in entries {
                        self.alloc_local(&e.target);
                    }
                }
                StmtKind::Expr(_)
                | StmtKind::ClassDef { .. }
                | StmtKind::Global(_)
                | StmtKind::Persistent(_)
                | StmtKind::Break
                | StmtKind::Continue
                | StmtKind::Return => {}
            }
        }
    }

    fn compile_function_body(
        name: String,
        params: Vec<String>,
        returns: Vec<String>,
        body: &[Stmt],
    ) -> Result<FunctionProto> {
        Self::compile_function_body_with_outer(name, params, returns, body, HashSet::new())
    }

    fn compile_function_body_with_outer(
        name: String,
        params: Vec<String>,
        returns: Vec<String>,
        body: &[Stmt],
        outer_locals: HashSet<String>,
    ) -> Result<FunctionProto> {
        let mut sub = Self {
            chunk: Chunk::new(),
            loops: Vec::new(),
            locals: Some(HashMap::new()),
            local_names: Vec::new(),
            globals_declared: HashSet::new(),
            persistents_declared: HashSet::new(),
            outer_locals,
            has_nested: false,
        };

        sub.collect_scope_decls(body);

        let nargin_slot = Some(sub.alloc_local("nargin"));
        let nargout_slot = Some(sub.alloc_local("nargout"));
        sub.alloc_local("ans");

        let mut param_slots = Vec::with_capacity(params.len());
        let mut varargin_slot = None;
        for p in &params {
            let slot = sub.alloc_local(p);
            if p == "varargin" {
                varargin_slot = Some(slot);
            }
            param_slots.push(slot);
        }
        let mut return_slots = Vec::with_capacity(returns.len());
        let mut varargout_slot = None;
        for r in &returns {
            let slot = sub.alloc_local(r);
            if r == "varargout" {
                varargout_slot = Some(slot);
            }
            return_slots.push(slot);
        }

        sub.collect_locals_in_stmts(body);

        let mut arg_specs = Vec::new();
        for s in body {
            if let StmtKind::Arguments { repeating, entries } = &s.kind {
                for e in entries {
                    let target_slot = sub.lookup_local(&e.target);
                    let default = if let Some(expr) = &e.default {
                        Some(Self::compile_standalone_expr(expr, "__arg_default")?)
                    } else {
                        None
                    };
                    let mut validators = Vec::with_capacity(e.validators.len());
                    for (vname, vargs) in &e.validators {
                        let mut chunks = Vec::with_capacity(vargs.len());
                        for (idx, va) in vargs.iter().enumerate() {
                            // In MATLAB `arguments`, `mustBeMember(opts.Mode, {...})` passes the target as arg 1
                            if idx == 0 {
                                let is_self_ref = match &va.kind {
                                    ExprKind::Ident(id) => id == &e.target && e.field.is_none(),
                                    ExprKind::FieldAccess { object, field } => {
                                        matches!(&object.kind, ExprKind::Ident(id) if id == &e.target)
                                            && e.field.as_deref() == Some(field.as_str())
                                    }
                                    _ => false,
                                };
                                if is_self_ref {
                                    continue;
                                }
                            }
                            chunks.push(Self::compile_standalone_expr(va, "__arg_default")?);
                        }
                        validators.push((vname.clone(), chunks));
                    }
                    arg_specs.push(CompiledArgSpec {
                        target: e.target.clone(),
                        target_slot,
                        field: e.field.clone(),
                        repeating: *repeating,
                        class_name: e.class_name.clone(),
                        validators,
                        default,
                    });
                }
            } else {
                sub.compile_stmt(s)?;
            }
        }
        sub.chunk.emit(OpCode::Return, 0);

        Ok(FunctionProto {
            name,
            params,
            returns,
            param_slots,
            return_slots,
            nargin_slot,
            nargout_slot,
            varargin_slot,
            varargout_slot,
            has_nested: sub.has_nested,
            arg_specs,
            local_names: sub.local_names,
            source_file: None,
            chunk: sub.chunk,
        })
    }

    fn compile_standalone_expr(expr: &Expr, target_var: &str) -> Result<Chunk> {
        let mut sub = Self {
            chunk: Chunk::new(),
            loops: Vec::new(),
            locals: None,
            local_names: Vec::new(),
            globals_declared: HashSet::new(),
            persistents_declared: HashSet::new(),
            outer_locals: HashSet::new(),
            has_nested: false,
        };
        sub.compile_expr(expr, 1, false)?;
        let idx = sub.chunk.add_name(target_var);
        sub.chunk.emit(OpCode::StoreVar(idx), expr.span.line as u32);
        sub.chunk.emit(OpCode::Return, expr.span.line as u32);
        Ok(sub.chunk)
    }

    fn compile_stmt(&mut self, stmt: &Stmt) -> Result<()> {
        let line = stmt.span.line as u32;
        match &stmt.kind {
            StmtKind::Expr(expr) => {
                if matches!(
                    &expr.kind,
                    ExprKind::CallOrIndex { callee, .. }
                        if matches!(
                            callee.as_str(),
                            "bode"
                                | "step"
                                | "impulse"
                                | "nyquist"
                                | "pzmap"
                                | "freqz"
                                | "spectrogram"
                                | "plot"
                                | "semilogx"
                                | "semilogy"
                                | "loglog"
                                | "scatter"
                                | "bar"
                                | "stem"
                                | "errorbar"
                                | "fill"
                                | "patch"
                                | "hist"
                                | "spy"
                                | "axis"
                                | "colorbar"
                                | "colormap"
                                | "format"
                                | "clc"
                                | "pause"
                                | "dir"
                                | "ls"
                                | "load"
                                | "save"
                                | "margin"
                                | "rlocus"
                                | "initial"
                                | "imhist"
                        )
                ) {
                    self.compile_expr(expr, 0, false)?;
                    return Ok(());
                }
                self.compile_expr(expr, 1, false)?;
                if stmt.suppressed {
                    if let Some(slot) = self.lookup_local("ans") {
                        self.chunk.emit(OpCode::StoreLocal(slot), line);
                    } else {
                        let ans_idx = self.chunk.add_name("ans");
                        self.chunk.emit(OpCode::StoreVar(ans_idx), line);
                    }
                } else {
                    self.chunk.emit(OpCode::PrintAns, line);
                }
            }
            StmtKind::Assign { target, value } => match target {
                AssignTarget::Var(name) => {
                    self.compile_expr(value, 1, false)?;
                    if self.globals_declared.contains(name) {
                        let idx = self.chunk.add_name(name);
                        if !stmt.suppressed {
                            self.chunk.emit(OpCode::Dup, line);
                        }
                        self.chunk.emit(OpCode::StoreGlobal(idx), line);
                        if !stmt.suppressed {
                            self.chunk.emit(OpCode::PrintVar(idx), line);
                        }
                    } else if self.persistents_declared.contains(name) {
                        let idx = self.chunk.add_name(name);
                        if !stmt.suppressed {
                            self.chunk.emit(OpCode::Dup, line);
                        }
                        self.chunk.emit(OpCode::StorePersistent(idx), line);
                        if !stmt.suppressed {
                            self.chunk.emit(OpCode::PrintVar(idx), line);
                        }
                    } else if let Some(slot) = self.lookup_local(name) {
                        if !stmt.suppressed {
                            self.chunk.emit(OpCode::Dup, line);
                        }
                        self.chunk.emit(OpCode::StoreLocal(slot), line);
                        if !stmt.suppressed {
                            self.chunk.emit(OpCode::PrintLocal(slot), line);
                        }
                    } else {
                        let idx = self.chunk.add_name(name);
                        if !stmt.suppressed {
                            self.chunk.emit(OpCode::Dup, line);
                        }
                        self.chunk.emit(OpCode::StoreVar(idx), line);
                        if !stmt.suppressed {
                            self.chunk.emit(OpCode::PrintVar(idx), line);
                        }
                    }
                }
                AssignTarget::Index { name, indices } => {
                    let total_dims = indices.len() as u8;
                    if let Some(slot) = self.lookup_local(name) {
                        self.compile_index_args_stack_end(
                            indices,
                            OpCode::PushEndLocalTarget { slot, total_dims },
                            line,
                        )?;
                        self.compile_expr(value, 1, false)?;
                        self.chunk.emit(
                            OpCode::StoreIndexLocal {
                                slot,
                                argc: total_dims,
                            },
                            line,
                        );
                        if !stmt.suppressed {
                            self.chunk.emit(OpCode::LoadLocal(slot), line);
                            self.chunk.emit(OpCode::PrintLocal(slot), line);
                        }
                    } else {
                        let name_idx = self.chunk.add_name(name);
                        self.compile_index_args_stack_end(
                            indices,
                            OpCode::PushEndVarTarget {
                                var_idx: name_idx,
                                total_dims,
                            },
                            line,
                        )?;
                        self.compile_expr(value, 1, false)?;
                        self.chunk.emit(
                            OpCode::StoreIndex {
                                name_idx,
                                argc: total_dims,
                            },
                            line,
                        );
                        if !stmt.suppressed {
                            self.chunk.emit(OpCode::LoadVar(name_idx), line);
                            self.chunk.emit(OpCode::PrintVar(name_idx), line);
                        }
                    }
                }
                AssignTarget::CellIndex { name, indices } => {
                    let total_dims = indices.len() as u8;
                    if let Some(slot) = self.lookup_local(name) {
                        self.compile_index_args_stack_end(
                            indices,
                            OpCode::PushEndLocalTarget { slot, total_dims },
                            line,
                        )?;
                        self.compile_expr(value, 1, false)?;
                        self.chunk.emit(
                            OpCode::StoreCellIndexLocal {
                                slot,
                                argc: total_dims,
                            },
                            line,
                        );
                        if !stmt.suppressed {
                            self.chunk.emit(OpCode::LoadLocal(slot), line);
                            self.chunk.emit(OpCode::PrintLocal(slot), line);
                        }
                    } else {
                        let name_idx = self.chunk.add_name(name);
                        self.compile_index_args_stack_end(
                            indices,
                            OpCode::PushEndVarTarget {
                                var_idx: name_idx,
                                total_dims,
                            },
                            line,
                        )?;
                        self.compile_expr(value, 1, false)?;
                        self.chunk.emit(
                            OpCode::StoreCellIndex {
                                name_idx,
                                argc: total_dims,
                            },
                            line,
                        );
                        if !stmt.suppressed {
                            self.chunk.emit(OpCode::LoadVar(name_idx), line);
                            self.chunk.emit(OpCode::PrintVar(name_idx), line);
                        }
                    }
                }
                AssignTarget::Field { name, fields } => {
                    self.compile_expr(value, 1, false)?;
                    let path_idx = self.chunk.add_field_path(fields.clone());
                    if let Some(slot) = self.lookup_local(name) {
                        self.chunk
                            .emit(OpCode::StoreFieldLocal { slot, path_idx }, line);
                        if !stmt.suppressed {
                            self.chunk.emit(OpCode::LoadLocal(slot), line);
                            self.chunk.emit(OpCode::PrintLocal(slot), line);
                        }
                    } else {
                        let name_idx = self.chunk.add_name(name);
                        self.chunk
                            .emit(OpCode::StoreField { name_idx, path_idx }, line);
                        if !stmt.suppressed {
                            self.chunk.emit(OpCode::LoadVar(name_idx), line);
                            self.chunk.emit(OpCode::PrintVar(name_idx), line);
                        }
                    }
                }
                AssignTarget::IndexedField {
                    name,
                    indices,
                    fields,
                } => {
                    let total_dims = indices.len() as u8;
                    let path_idx = self.chunk.add_field_path(fields.clone());
                    if let Some(slot) = self.lookup_local(name) {
                        self.compile_index_args_stack_end(
                            indices,
                            OpCode::PushEndLocalTarget { slot, total_dims },
                            line,
                        )?;
                        self.compile_expr(value, 1, false)?;
                        self.chunk.emit(
                            OpCode::StoreIndexedFieldLocal {
                                slot,
                                argc: total_dims,
                                path_idx,
                            },
                            line,
                        );
                        if !stmt.suppressed {
                            self.chunk.emit(OpCode::LoadLocal(slot), line);
                            self.chunk.emit(OpCode::PrintLocal(slot), line);
                        }
                    } else {
                        let name_idx = self.chunk.add_name(name);
                        self.compile_index_args_stack_end(
                            indices,
                            OpCode::PushEndVarTarget {
                                var_idx: name_idx,
                                total_dims,
                            },
                            line,
                        )?;
                        self.compile_expr(value, 1, false)?;
                        self.chunk.emit(
                            OpCode::StoreIndexedField {
                                name_idx,
                                argc: total_dims,
                                path_idx,
                            },
                            line,
                        );
                        if !stmt.suppressed {
                            self.chunk.emit(OpCode::LoadVar(name_idx), line);
                            self.chunk.emit(OpCode::PrintVar(name_idx), line);
                        }
                    }
                }
                AssignTarget::FieldIndex {
                    name,
                    fields,
                    indices,
                } => {
                    // Desugar `obj.a.b(idx) = val` -> tmp = obj.a.b; tmp(idx) = val; obj.a.b = tmp
                    let total_dims = indices.len() as u8;
                    let path_idx = self.chunk.add_field_path(fields.clone());
                    if let Some(slot) = self.lookup_local(name) {
                        self.chunk.emit(OpCode::LoadLocal(slot), line);
                    } else {
                        let name_idx = self.chunk.add_name(name);
                        self.chunk.emit(OpCode::LoadVar(name_idx), line);
                    }
                    for f in fields {
                        let f_idx = self.chunk.add_name(f);
                        self.chunk.emit(OpCode::GetField(f_idx), line);
                    }
                    if let Some(tmp_slot) = self.lookup_local("__matro_fi_tmp") {
                        self.chunk.emit(OpCode::StoreLocal(tmp_slot), line);
                        self.compile_index_args_stack_end(
                            indices,
                            OpCode::PushEndLocalTarget {
                                slot: tmp_slot,
                                total_dims,
                            },
                            line,
                        )?;
                        self.compile_expr(value, 1, false)?;
                        self.chunk.emit(
                            OpCode::StoreIndexLocal {
                                slot: tmp_slot,
                                argc: total_dims,
                            },
                            line,
                        );
                        self.chunk.emit(OpCode::LoadLocal(tmp_slot), line);
                    } else {
                        let tmp_idx = self.chunk.add_name("__matro_fi_tmp");
                        self.chunk.emit(OpCode::StoreVar(tmp_idx), line);
                        self.compile_index_args_stack_end(
                            indices,
                            OpCode::PushEndVarTarget {
                                var_idx: tmp_idx,
                                total_dims,
                            },
                            line,
                        )?;
                        self.compile_expr(value, 1, false)?;
                        self.chunk.emit(
                            OpCode::StoreIndex {
                                name_idx: tmp_idx,
                                argc: total_dims,
                            },
                            line,
                        );
                        self.chunk.emit(OpCode::LoadVar(tmp_idx), line);
                    }
                    if let Some(slot) = self.lookup_local(name) {
                        self.chunk
                            .emit(OpCode::StoreFieldLocal { slot, path_idx }, line);
                        if !stmt.suppressed {
                            self.chunk.emit(OpCode::LoadLocal(slot), line);
                            self.chunk.emit(OpCode::PrintLocal(slot), line);
                        }
                    } else {
                        let name_idx = self.chunk.add_name(name);
                        self.chunk
                            .emit(OpCode::StoreField { name_idx, path_idx }, line);
                        if !stmt.suppressed {
                            self.chunk.emit(OpCode::LoadVar(name_idx), line);
                            self.chunk.emit(OpCode::PrintVar(name_idx), line);
                        }
                    }
                }
                AssignTarget::Chain { name, steps } => {
                    self.compile_expr(value, 1, false)?;
                    let mut compiled_steps = Vec::with_capacity(steps.len());
                    for step in steps {
                        match step {
                            LValueStep::Field(f) => {
                                let f_idx = self.chunk.add_name(f);
                                compiled_steps.push(ChainStep::Field(f_idx));
                            }
                            LValueStep::Paren(indices) => {
                                for arg in indices {
                                    match arg {
                                        IndexArg::Colon => {
                                            self.chunk.emit(OpCode::PushColon, line);
                                        }
                                        IndexArg::Expr(e) => {
                                            self.compile_expr(e, 1, true)?;
                                        }
                                    }
                                }
                                compiled_steps.push(ChainStep::Paren(indices.len() as u8));
                            }
                            LValueStep::Brace(indices) => {
                                for arg in indices {
                                    match arg {
                                        IndexArg::Colon => {
                                            self.chunk.emit(OpCode::PushColon, line);
                                        }
                                        IndexArg::Expr(e) => {
                                            self.compile_expr(e, 1, true)?;
                                        }
                                    }
                                }
                                compiled_steps.push(ChainStep::Brace(indices.len() as u8));
                            }
                        }
                    }
                    let spec_idx = self.chunk.add_chain_spec(compiled_steps);
                    if let Some(slot) = self.lookup_local(name) {
                        self.chunk
                            .emit(OpCode::StoreChainLocal { slot, spec_idx }, line);
                        if !stmt.suppressed {
                            self.chunk.emit(OpCode::LoadLocal(slot), line);
                            self.chunk.emit(OpCode::PrintLocal(slot), line);
                        }
                    } else {
                        let name_idx = self.chunk.add_name(name);
                        self.chunk
                            .emit(OpCode::StoreChain { name_idx, spec_idx }, line);
                        if !stmt.suppressed {
                            self.chunk.emit(OpCode::LoadVar(name_idx), line);
                            self.chunk.emit(OpCode::PrintVar(name_idx), line);
                        }
                    }
                }
            },
            StmtKind::MultiAssign { targets, value } => {
                let nargout = targets.len() as u8;
                self.compile_expr(value, nargout, false)?;
                for target in targets.iter().rev() {
                    if target == "~" {
                        self.chunk.emit(OpCode::Pop, line);
                    } else if let Some(slot) = self.lookup_local(target) {
                        if !stmt.suppressed {
                            self.chunk.emit(OpCode::Dup, line);
                        }
                        self.chunk.emit(OpCode::StoreLocal(slot), line);
                        if !stmt.suppressed {
                            self.chunk.emit(OpCode::PrintLocal(slot), line);
                        }
                    } else {
                        let idx = self.chunk.add_name(target);
                        if !stmt.suppressed {
                            self.chunk.emit(OpCode::Dup, line);
                        }
                        self.chunk.emit(OpCode::StoreVar(idx), line);
                        if !stmt.suppressed {
                            self.chunk.emit(OpCode::PrintVar(idx), line);
                        }
                    }
                }
            }
            StmtKind::If {
                cond,
                then_body,
                elseifs,
                else_body,
            } => {
                let mut end_jumps = Vec::new();
                self.compile_expr(cond, 1, false)?;
                let mut false_jump = self.chunk.emit(OpCode::JumpIfFalse(0), line);

                for s in then_body {
                    self.compile_stmt(s)?;
                }
                end_jumps.push(self.chunk.emit(OpCode::Jump(0), line));

                for (econd, ebody) in elseifs {
                    let target = self.chunk.code.len() as u32;
                    self.chunk.code[false_jump] = OpCode::JumpIfFalse(target);

                    let eline = econd.span.line as u32;
                    self.compile_expr(econd, 1, false)?;
                    false_jump = self.chunk.emit(OpCode::JumpIfFalse(0), eline);
                    for s in ebody {
                        self.compile_stmt(s)?;
                    }
                    end_jumps.push(self.chunk.emit(OpCode::Jump(0), eline));
                }

                let else_start = self.chunk.code.len() as u32;
                self.chunk.code[false_jump] = OpCode::JumpIfFalse(else_start);
                if let Some(ebody) = else_body {
                    for s in ebody {
                        self.compile_stmt(s)?;
                    }
                }

                let end_ip = self.chunk.code.len() as u32;
                for j in end_jumps {
                    self.chunk.code[j] = OpCode::Jump(end_ip);
                }
            }
            StmtKind::Switch {
                expr,
                cases,
                otherwise,
            } => {
                self.compile_expr(expr, 1, false)?;
                let mut end_jumps = Vec::new();

                for (case_expr, body) in cases {
                    let cline = case_expr.span.line as u32;
                    self.compile_expr(case_expr, 1, false)?;
                    self.chunk.emit(OpCode::SwitchMatch, cline);
                    let next_case = self.chunk.emit(OpCode::JumpIfFalse(0), cline);
                    self.chunk.emit(OpCode::Pop, cline);
                    for s in body {
                        self.compile_stmt(s)?;
                    }
                    end_jumps.push(self.chunk.emit(OpCode::Jump(0), cline));
                    let next_ip = self.chunk.code.len() as u32;
                    self.chunk.code[next_case] = OpCode::JumpIfFalse(next_ip);
                }

                self.chunk.emit(OpCode::Pop, line);
                if let Some(body) = otherwise {
                    for s in body {
                        self.compile_stmt(s)?;
                    }
                }

                let end_ip = self.chunk.code.len() as u32;
                for j in end_jumps {
                    self.chunk.code[j] = OpCode::Jump(end_ip);
                }
            }
            StmtKind::TryCatch {
                try_body,
                catch_var,
                catch_body,
            } => {
                let (err_var, err_is_local) = match catch_var.as_deref() {
                    Some(n) => {
                        if let Some(slot) = self.lookup_local(n) {
                            (slot as u32, true)
                        } else {
                            (self.chunk.add_name(n), false)
                        }
                    }
                    None => (u32::MAX, false),
                };
                let enter_ip = self.chunk.emit(
                    OpCode::TryEnter {
                        catch_ip: 0,
                        err_var,
                        err_is_local,
                    },
                    line,
                );
                for s in try_body {
                    self.compile_stmt(s)?;
                }
                self.chunk.emit(OpCode::TryExit, line);
                let skip_catch = self.chunk.emit(OpCode::Jump(0), line);

                let catch_ip = self.chunk.code.len() as u32;
                self.chunk.code[enter_ip] = OpCode::TryEnter {
                    catch_ip,
                    err_var,
                    err_is_local,
                };
                for s in catch_body {
                    self.compile_stmt(s)?;
                }
                let end_ip = self.chunk.code.len() as u32;
                self.chunk.code[skip_catch] = OpCode::Jump(end_ip);
            }
            StmtKind::While { cond, body } => {
                let loop_start = self.chunk.code.len();
                self.compile_expr(cond, 1, false)?;
                let exit_jump = self.chunk.emit(OpCode::JumpIfFalse(0), line);

                self.loops.push(LoopContext {
                    is_for: false,
                    continue_target: Some(loop_start),
                    continue_patches: Vec::new(),
                    break_patches: Vec::new(),
                });

                for s in body {
                    self.compile_stmt(s)?;
                }
                self.chunk.emit(OpCode::Jump(loop_start as u32), line);

                let exit_ip = self.chunk.code.len() as u32;
                self.chunk.code[exit_jump] = OpCode::JumpIfFalse(exit_ip);

                let ctx = self.loops.pop().unwrap();
                for bp in ctx.break_patches {
                    self.chunk.code[bp] = OpCode::Jump(exit_ip);
                }
            }
            StmtKind::For { var, iter, body } => {
                let local_slot = self.lookup_local(var);
                let range_has_step = if let ExprKind::Range { start, step, stop } = &iter.kind {
                    self.compile_expr(start, 1, false)?;
                    let hs = if let Some(st) = step {
                        self.compile_expr(st, 1, false)?;
                        true
                    } else {
                        false
                    };
                    self.compile_expr(stop, 1, false)?;
                    Some(hs)
                } else {
                    self.compile_expr(iter, 1, false)?;
                    None
                };

                let init_ip = match (local_slot, range_has_step) {
                    (Some(slot), Some(has_step)) => self.chunk.emit(
                        OpCode::ForRangeInitLocal {
                            slot,
                            exit_ip: 0,
                            has_step,
                        },
                        line,
                    ),
                    (None, Some(has_step)) => {
                        let var_idx = self.chunk.add_name(var);
                        self.chunk.emit(
                            OpCode::ForRangeInit {
                                var_idx,
                                exit_ip: 0,
                                has_step,
                            },
                            line,
                        )
                    }
                    (Some(slot), None) => self
                        .chunk
                        .emit(OpCode::ForInitLocal { slot, exit_ip: 0 }, line),
                    (None, None) => {
                        let var_idx = self.chunk.add_name(var);
                        self.chunk
                            .emit(OpCode::ForInit { var_idx, exit_ip: 0 }, line)
                    }
                };
                let body_start = self.chunk.code.len();

                self.loops.push(LoopContext {
                    is_for: true,
                    continue_target: None,
                    continue_patches: Vec::new(),
                    break_patches: Vec::new(),
                });

                for s in body {
                    self.compile_stmt(s)?;
                }

                let next_ip = self.chunk.code.len() as u32;
                if let Some(slot) = local_slot {
                    self.chunk.emit(
                        OpCode::ForNextLocal {
                            slot,
                            loop_ip: body_start as u32,
                        },
                        line,
                    );
                } else {
                    let var_idx = self.chunk.add_name(var);
                    self.chunk.emit(
                        OpCode::ForNext {
                            var_idx,
                            loop_ip: body_start as u32,
                        },
                        line,
                    );
                }

                let exit_ip = self.chunk.code.len() as u32;
                match (local_slot, range_has_step) {
                    (Some(slot), Some(has_step)) => {
                        self.chunk.code[init_ip] = OpCode::ForRangeInitLocal {
                            slot,
                            exit_ip,
                            has_step,
                        };
                    }
                    (None, Some(has_step)) => {
                        let var_idx = self.chunk.add_name(var);
                        self.chunk.code[init_ip] = OpCode::ForRangeInit {
                            var_idx,
                            exit_ip,
                            has_step,
                        };
                    }
                    (Some(slot), None) => {
                        self.chunk.code[init_ip] = OpCode::ForInitLocal { slot, exit_ip };
                    }
                    (None, None) => {
                        let var_idx = self.chunk.add_name(var);
                        self.chunk.code[init_ip] = OpCode::ForInit { var_idx, exit_ip };
                    }
                }

                let ctx = self.loops.pop().unwrap();
                for cp in ctx.continue_patches {
                    self.chunk.code[cp] = OpCode::Jump(next_ip);
                }
                for bp in ctx.break_patches {
                    self.chunk.code[bp] = OpCode::Jump(exit_ip);
                }
            }
            StmtKind::Break => {
                let Some(ctx) = self.loops.last_mut() else {
                    return Err(MatroError::compile(stmt.span, "'break' outside of loop"));
                };
                if ctx.is_for {
                    self.chunk.emit(OpCode::ForPop, line);
                }
                let patch = self.chunk.emit(OpCode::Jump(0), line);
                self.loops.last_mut().unwrap().break_patches.push(patch);
            }
            StmtKind::Continue => {
                let Some(ctx) = self.loops.last() else {
                    return Err(MatroError::compile(stmt.span, "'continue' outside of loop"));
                };
                if let Some(target) = ctx.continue_target {
                    self.chunk.emit(OpCode::Jump(target as u32), line);
                } else {
                    let patch = self.chunk.emit(OpCode::Jump(0), line);
                    self.loops.last_mut().unwrap().continue_patches.push(patch);
                }
            }
            StmtKind::Return => {
                self.chunk.emit(OpCode::Return, line);
            }
            StmtKind::Function {
                name,
                params,
                returns,
                body,
            } => {
                if let Some(slot) = self.lookup_local(name) {
                    let mut outer: HashSet<String> = self.outer_locals.clone();
                    for ln in &self.local_names {
                        if ln != "nargin" && ln != "nargout" && ln != "ans" {
                            outer.insert(ln.clone());
                        }
                    }
                    let proto = Self::compile_function_body_with_outer(
                        name.clone(),
                        params.clone(),
                        returns.clone(),
                        body,
                        outer,
                    )?;
                    let func_idx = self.chunk.add_function(proto);
                    self.chunk
                        .emit(OpCode::DefineNestedFunc { func_idx, slot }, line);
                } else {
                    let proto = Self::compile_function_body(
                        name.clone(),
                        params.clone(),
                        returns.clone(),
                        body,
                    )?;
                    let idx = self.chunk.add_function(proto);
                    self.chunk.emit(OpCode::DefineFunc(idx), line);
                }
            }
            StmtKind::ClassDef {
                name,
                superclasses,
                properties,
                methods,
                enumerations,
                events,
            } => {
                let mut compiled_props = Vec::with_capacity(properties.len());
                for p in properties {
                    let default_chunk = if let Some(expr) = &p.default {
                        Some(Self::compile_standalone_expr(expr, "__prop_default")?)
                    } else {
                        None
                    };
                    compiled_props.push(PropProto {
                        name: p.name.clone(),
                        default: default_chunk,
                        dependent: p.dependent,
                        constant: p.constant,
                    });
                }

                let mut compiled_methods = Vec::with_capacity(methods.len());
                let mut static_methods = Vec::new();
                for m in methods {
                    if m.is_static {
                        static_methods.push(m.name.clone());
                    }
                    let proto = Self::compile_function_body(
                        m.name.clone(),
                        m.params.clone(),
                        m.returns.clone(),
                        &m.body,
                    )?;
                    compiled_methods.push(proto);
                }

                let mut compiled_enums = Vec::with_capacity(enumerations.len());
                for (ename, eargs) in enumerations {
                    let mut arg_chunks = Vec::with_capacity(eargs.len());
                    for ea in eargs {
                        arg_chunks.push(Self::compile_standalone_expr(ea, "__enum_arg")?);
                    }
                    compiled_enums.push((ename.clone(), arg_chunks));
                }

                let idx = self.chunk.add_class(ClassProto {
                    name: name.clone(),
                    superclasses: superclasses.clone(),
                    properties: compiled_props,
                    methods: compiled_methods,
                    static_methods,
                    enumerations: compiled_enums,
                    events: events.clone(),
                    source_file: None,
                });
                self.chunk.emit(OpCode::DefineClass(idx), line);
            }
            StmtKind::Arguments { .. } => {}
            StmtKind::Global(names) => {
                for n in names {
                    self.globals_declared.insert(n.clone());
                    let idx = self.chunk.add_name(n);
                    self.chunk.emit(OpCode::InitGlobal(idx), line);
                }
            }
            StmtKind::Persistent(names) => {
                for n in names {
                    self.persistents_declared.insert(n.clone());
                    let idx = self.chunk.add_name(n);
                    self.chunk.emit(OpCode::InitPersistent(idx), line);
                }
            }
        }
        Ok(())
    }

    fn compile_index_args_stack_end(
        &mut self,
        args: &[IndexArg],
        push_op: OpCode,
        line: u32,
    ) -> Result<()> {
        let has_end = args_has_end(args);
        if has_end {
            self.chunk.emit(push_op, line);
        }
        for (dim, arg) in args.iter().enumerate() {
            if has_end && index_arg_has_end(arg) {
                self.chunk.emit(OpCode::SetEndDim(dim as u8), line);
                self.compile_index_arg(arg, true)?;
            } else {
                self.compile_index_arg(arg, false)?;
            }
        }
        if has_end {
            self.chunk.emit(OpCode::PopEndTarget, line);
        }
        Ok(())
    }

    fn compile_index_arg(&mut self, arg: &IndexArg, in_index: bool) -> Result<()> {
        match arg {
            IndexArg::Colon => {
                self.chunk.emit(OpCode::PushColon, 0);
                Ok(())
            }
            IndexArg::Expr(e) => self.compile_expr(e, 1, in_index),
        }
    }

    fn compile_expr(&mut self, expr: &Expr, nargout: u8, in_index: bool) -> Result<()> {
        let line = expr.span.line as u32;

        // Compile-time constant folding for pure numeric literal expressions
        if let Some(folded) = try_fold_const(expr) {
            let idx = self.chunk.add_const(Value::scalar(folded));
            self.chunk.emit(OpCode::LoadConst(idx), line);
            return Ok(());
        }

        match &expr.kind {
            ExprKind::Number(n) => {
                let idx = self.chunk.add_const(Value::scalar(*n));
                self.chunk.emit(OpCode::LoadConst(idx), line);
            }
            ExprKind::Bool(b) => {
                let idx = self.chunk.add_const(Value::bool(*b));
                self.chunk.emit(OpCode::LoadConst(idx), line);
            }
            ExprKind::ImagNumber(im) => {
                let idx = self.chunk.add_const(Value::complex(0.0, *im));
                self.chunk.emit(OpCode::LoadConst(idx), line);
            }
            ExprKind::String(s) => {
                let idx = self.chunk.add_const(Value::String(s.clone()));
                self.chunk.emit(OpCode::LoadConst(idx), line);
            }
            ExprKind::Ident(name) => {
                if self.globals_declared.contains(name) {
                    let idx = self.chunk.add_name(name);
                    self.chunk.emit(OpCode::LoadGlobal(idx), line);
                } else if self.persistents_declared.contains(name) {
                    let idx = self.chunk.add_name(name);
                    self.chunk.emit(OpCode::LoadPersistent(idx), line);
                } else if let Some(slot) = self.lookup_local(name) {
                    self.chunk.emit(OpCode::LoadLocal(slot), line);
                } else {
                    let idx = self.chunk.add_name(name);
                    self.chunk.emit(OpCode::LoadVar(idx), line);
                }
            }
            ExprKind::EndKeyword => {
                if !in_index {
                    return Err(MatroError::compile(
                        expr.span,
                        "'end' can only be used inside an indexing expression",
                    ));
                }
                self.chunk.emit(OpCode::PushEndTop, line);
            }
            ExprKind::Matrix(rows) => {
                for row in rows {
                    for item in row {
                        self.compile_expr(item, 1, in_index)?;
                    }
                    self.chunk.emit(OpCode::BuildRow(row.len() as u32), line);
                }
                self.chunk
                    .emit(OpCode::BuildMatrix(rows.len() as u32), line);
            }
            ExprKind::Cell(rows) => {
                let r_count = rows.len();
                let c_count = rows.first().map_or(0, Vec::len);
                for row in rows {
                    if row.len() != c_count {
                        return Err(MatroError::compile(
                            expr.span,
                            "Cell array rows must have the same number of columns",
                        ));
                    }
                    for item in row {
                        self.compile_expr(item, 1, in_index)?;
                    }
                }
                self.chunk.emit(
                    OpCode::BuildCell {
                        rows: r_count as u16,
                        cols: c_count as u16,
                    },
                    line,
                );
            }
            ExprKind::AnonFunc { params, body } => {
                let mut sub = Self {
                    chunk: Chunk::new(),
                    loops: Vec::new(),
                    locals: Some(HashMap::new()),
                    local_names: Vec::new(),
                    globals_declared: HashSet::new(),
                    persistents_declared: HashSet::new(),
                    outer_locals: HashSet::new(),
                    has_nested: false,
                };
                let mut param_slots = Vec::with_capacity(params.len());
                for p in params {
                    param_slots.push(sub.alloc_local(p));
                }
                let varargin_slot = if params.last().map(String::as_str) == Some("varargin") {
                    param_slots.last().copied()
                } else {
                    None
                };
                let ret_slot = sub.alloc_local("__anon_ret");
                sub.compile_expr(body, 1, false)?;
                sub.chunk.emit(OpCode::StoreLocal(ret_slot), line);
                sub.chunk.emit(OpCode::Return, line);

                let proto_name = if params.len() == 1 && params[0] == "varargin" {
                    if let ExprKind::CallOrIndex { callee, args } = &body.kind {
                        if args.len() == 1
                            && matches!(
                                &args[0],
                                IndexArg::Expr(Expr {
                                    kind: ExprKind::CellIndex { target, indices },
                                    ..
                                }) if matches!(&target.kind, ExprKind::Ident(id) if id == "varargin")
                                    && matches!(indices.as_slice(), [IndexArg::Colon])
                            )
                        {
                            format!("@{callee}")
                        } else {
                            "<anonymous>".to_string()
                        }
                    } else {
                        "<anonymous>".to_string()
                    }
                } else {
                    "<anonymous>".to_string()
                };

                let proto = FunctionProto {
                    name: proto_name,
                    params: params.clone(),
                    returns: vec!["__anon_ret".to_string()],
                    param_slots,
                    return_slots: vec![ret_slot],
                    nargin_slot: None,
                    nargout_slot: None,
                    varargin_slot,
                    varargout_slot: None,
                    has_nested: false,
                    arg_specs: Vec::new(),
                    local_names: sub.local_names,
                    source_file: None,
                    chunk: sub.chunk,
                };
                let idx = self.chunk.add_function(proto);
                self.chunk.emit(OpCode::MakeClosure(idx), line);
            }
            ExprKind::Unary { op, expr: inner } => {
                self.compile_expr(inner, 1, in_index)?;
                let code = match op {
                    UnaryOp::Neg => OpCode::Neg,
                    UnaryOp::Pos => OpCode::Pos,
                    UnaryOp::Not => OpCode::Not,
                };
                self.chunk.emit(code, line);
            }
            ExprKind::CTranspose(inner) => {
                self.compile_expr(inner, 1, in_index)?;
                self.chunk.emit(OpCode::CTranspose, line);
            }
            ExprKind::Transpose(inner) => {
                self.compile_expr(inner, 1, in_index)?;
                self.chunk.emit(OpCode::Transpose, line);
            }
            ExprKind::Range { start, step, stop } => {
                self.compile_expr(start, 1, in_index)?;
                let has_step = if let Some(st) = step {
                    self.compile_expr(st, 1, in_index)?;
                    true
                } else {
                    false
                };
                self.compile_expr(stop, 1, in_index)?;
                self.chunk.emit(OpCode::MakeRange { has_step }, line);
            }
            ExprKind::Binary { op, left, right } => match op {
                BinaryOp::ShortAnd => {
                    self.compile_expr(left, 1, in_index)?;
                    let skip = self.chunk.emit(OpCode::JumpIfFalsePeek(0), line);
                    self.chunk.emit(OpCode::Pop, line);
                    self.compile_expr(right, 1, in_index)?;
                    let end = self.chunk.code.len() as u32;
                    self.chunk.code[skip] = OpCode::JumpIfFalsePeek(end);
                }
                BinaryOp::ShortOr => {
                    self.compile_expr(left, 1, in_index)?;
                    let skip = self.chunk.emit(OpCode::JumpIfTruePeek(0), line);
                    self.chunk.emit(OpCode::Pop, line);
                    self.compile_expr(right, 1, in_index)?;
                    let end = self.chunk.code.len() as u32;
                    self.chunk.code[skip] = OpCode::JumpIfTruePeek(end);
                }
                _ => {
                    self.compile_expr(left, 1, in_index)?;
                    self.compile_expr(right, 1, in_index)?;
                    let code = match op {
                        BinaryOp::Add => OpCode::Add,
                        BinaryOp::Sub => OpCode::Sub,
                        BinaryOp::Mul => OpCode::Mul,
                        BinaryOp::ElemMul => OpCode::ElemMul,
                        BinaryOp::Div => OpCode::Div,
                        BinaryOp::ElemDiv => OpCode::ElemDiv,
                        BinaryOp::LeftDiv => OpCode::LeftDiv,
                        BinaryOp::ElemLeftDiv => OpCode::ElemLeftDiv,
                        BinaryOp::Pow => OpCode::Pow,
                        BinaryOp::ElemPow => OpCode::ElemPow,
                        BinaryOp::Eq => OpCode::Eq,
                        BinaryOp::Ne => OpCode::Ne,
                        BinaryOp::Lt => OpCode::Lt,
                        BinaryOp::Le => OpCode::Le,
                        BinaryOp::Gt => OpCode::Gt,
                        BinaryOp::Ge => OpCode::Ge,
                        BinaryOp::And => OpCode::And,
                        BinaryOp::Or => OpCode::Or,
                        BinaryOp::ShortAnd | BinaryOp::ShortOr => unreachable!(),
                    };
                    self.chunk.emit(code, line);
                }
            },
            ExprKind::CallOrIndex { callee, args } => {
                let total_dims = args.len() as u8;
                if let Some(slot) = self.lookup_local(callee) {
                    self.compile_index_args_stack_end(
                        args,
                        OpCode::PushEndLocalTarget { slot, total_dims },
                        line,
                    )?;
                    self.chunk.emit(
                        OpCode::CallOrIndexLocal {
                            slot,
                            argc: total_dims,
                            nargout,
                        },
                        line,
                    );
                } else {
                    let name_idx = self.chunk.add_name(callee);
                    self.compile_index_args_stack_end(
                        args,
                        OpCode::PushEndVarTarget {
                            var_idx: name_idx,
                            total_dims,
                        },
                        line,
                    )?;
                    self.chunk.emit(
                        OpCode::CallOrIndex {
                            name_idx,
                            argc: total_dims,
                            nargout,
                        },
                        line,
                    );
                }
            }
            ExprKind::DynamicCall { target, args } => {
                self.compile_expr(target, 1, in_index)?;
                let total_dims = args.len() as u8;
                self.compile_index_args_stack_end(
                    args,
                    OpCode::PushEndTarget { total_dims },
                    line,
                )?;
                self.chunk.emit(
                    OpCode::DynamicCall {
                        argc: total_dims,
                        nargout,
                    },
                    line,
                );
            }
            ExprKind::MethodCall {
                object,
                method,
                args,
            } => {
                self.compile_expr(object, 1, in_index)?;
                let total_dims = args.len() as u8;
                let method_idx = self.chunk.add_name(method);
                self.compile_index_args_stack_end(
                    args,
                    OpCode::PushEndFieldTarget {
                        field_idx: method_idx,
                        total_dims,
                    },
                    line,
                )?;
                self.chunk.emit(
                    OpCode::MethodCall {
                        method_idx,
                        argc: total_dims,
                        nargout,
                    },
                    line,
                );
            }
            ExprKind::CellIndex { target, indices } => {
                self.compile_expr(target, 1, in_index)?;
                let total_dims = indices.len() as u8;
                self.compile_index_args_stack_end(
                    indices,
                    OpCode::PushEndTarget { total_dims },
                    line,
                )?;
                self.chunk.emit(
                    OpCode::CellIndex {
                        argc: total_dims,
                        nargout,
                    },
                    line,
                );
            }
            ExprKind::FieldAccess { object, field } => {
                self.compile_expr(object, 1, in_index)?;
                let f_idx = self.chunk.add_name(field);
                self.chunk.emit(OpCode::GetField(f_idx), line);
            }
        }
        Ok(())
    }
}

fn expr_has_end(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::EndKeyword => true,
        ExprKind::Number(_)
        | ExprKind::Bool(_)
        | ExprKind::ImagNumber(_)
        | ExprKind::String(_)
        | ExprKind::Ident(_)
        | ExprKind::AnonFunc { .. } => false,
        ExprKind::Unary { expr: inner, .. }
        | ExprKind::CTranspose(inner)
        | ExprKind::Transpose(inner)
        | ExprKind::FieldAccess { object: inner, .. } => expr_has_end(inner),
        ExprKind::Binary { left, right, .. } => expr_has_end(left) || expr_has_end(right),
        ExprKind::Range { start, step, stop } => {
            expr_has_end(start)
                || step.as_deref().is_some_and(expr_has_end)
                || expr_has_end(stop)
        }
        ExprKind::Matrix(rows) | ExprKind::Cell(rows) => {
            rows.iter().any(|row| row.iter().any(expr_has_end))
        }
        ExprKind::CallOrIndex { args, .. } => args_has_end(args),
        ExprKind::DynamicCall { target, args } => expr_has_end(target) || args_has_end(args),
        ExprKind::MethodCall { object, args, .. } => expr_has_end(object) || args_has_end(args),
        ExprKind::CellIndex { target, indices } => expr_has_end(target) || args_has_end(indices),
    }
}

fn index_arg_has_end(arg: &IndexArg) -> bool {
    match arg {
        IndexArg::Colon => false,
        IndexArg::Expr(e) => expr_has_end(e),
    }
}

fn args_has_end(args: &[IndexArg]) -> bool {
    args.iter().any(index_arg_has_end)
}

/// Pure scalar constant folder (only folds compound unary/binary expressions of numeric literals).
fn try_fold_const(expr: &Expr) -> Option<f64> {
    match &expr.kind {
        ExprKind::Unary { op, expr: inner } => {
            let v = eval_const_leaf(inner)?;
            match op {
                UnaryOp::Neg => Some(-v),
                UnaryOp::Pos => Some(v),
                UnaryOp::Not => None, // preserve logical flag
            }
        }
        ExprKind::Binary { op, left, right } => {
            let a = eval_const_leaf(left)?;
            let b = eval_const_leaf(right)?;
            match op {
                BinaryOp::Add => Some(a + b),
                BinaryOp::Sub => Some(a - b),
                BinaryOp::Mul | BinaryOp::ElemMul => Some(a * b),
                BinaryOp::Div | BinaryOp::ElemDiv => Some(a / b),
                BinaryOp::LeftDiv | BinaryOp::ElemLeftDiv => Some(b / a),
                BinaryOp::Pow | BinaryOp::ElemPow if a >= 0.0 || b.fract() == 0.0 => Some(a.powf(b)),
                _ => None,
            }
        }
        _ => None,
    }
}

fn eval_const_leaf(expr: &Expr) -> Option<f64> {
    match &expr.kind {
        ExprKind::Number(n) => Some(*n),
        _ => try_fold_const(expr),
    }
}
