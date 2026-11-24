use matro_types::{MatroError, Result};

use crate::ast::*;
use crate::lexer::{Lexer, Token, TokenKind};

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    /// When true, unary `+`/`-` after whitespace inside `[ ... ]` or `{ ... }` starts a new element.
    in_matrix: bool,
}

impl Parser {
    pub fn parse_str(input: &str) -> Result<Program> {
        let tokens = Lexer::new(input).tokenize()?;
        let mut parser = Self {
            tokens,
            pos: 0,
            in_matrix: false,
        };
        parser.parse_program()
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.pos.min(self.tokens.len() - 1)]
    }

    fn peek_kind(&self) -> &TokenKind {
        &self.peek().kind
    }

    fn peek_at(&self, offset: usize) -> &Token {
        &self.tokens[(self.pos + offset).min(self.tokens.len() - 1)]
    }

    fn bump(&mut self) -> Token {
        let tok = self.peek().clone();
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        tok
    }

    fn skip_separators(&mut self) {
        while matches!(
            self.peek_kind(),
            TokenKind::Newline | TokenKind::Semicolon | TokenKind::Comma
        ) {
            self.bump();
        }
    }

    fn skip_newlines(&mut self) {
        while matches!(self.peek_kind(), TokenKind::Newline) {
            self.bump();
        }
    }

    fn expect(&mut self, expected: TokenKind, label: &str) -> Result<Token> {
        if *self.peek_kind() == expected {
            Ok(self.bump())
        } else {
            Err(MatroError::syntax(
                self.peek().span,
                format!("Expected {label}, found {:?}", self.peek_kind()),
            ))
        }
    }

    fn parse_program(&mut self) -> Result<Program> {
        let mut statements = Vec::new();
        self.skip_separators();
        while *self.peek_kind() != TokenKind::Eof {
            statements.push(self.parse_stmt()?);
            self.skip_separators();
        }
        Ok(Program { statements })
    }

    fn parse_block(&mut self, stop_tokens: &[TokenKind]) -> Result<Vec<Stmt>> {
        let mut stmts = Vec::new();
        self.skip_separators();
        while *self.peek_kind() != TokenKind::Eof && !stop_tokens.contains(self.peek_kind()) {
            stmts.push(self.parse_stmt()?);
            self.skip_separators();
        }
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> Result<Stmt> {
        let span = self.peek().span;
        let kind = match self.peek_kind() {
            TokenKind::If => self.parse_if()?,
            TokenKind::Switch => self.parse_switch()?,
            TokenKind::Try => self.parse_try()?,
            TokenKind::While => self.parse_while()?,
            TokenKind::For => self.parse_for()?,
            TokenKind::Function => self.parse_function()?,
            TokenKind::ClassDef => self.parse_classdef()?,
            TokenKind::Global => self.parse_scope_decl(false)?,
            TokenKind::Persistent => self.parse_scope_decl(true)?,
            TokenKind::Break => {
                self.bump();
                StmtKind::Break
            }
            TokenKind::Continue => {
                self.bump();
                StmtKind::Continue
            }
            TokenKind::Return => {
                self.bump();
                StmtKind::Return
            }
            TokenKind::Ident(kw)
                if kw == "arguments"
                    && matches!(
                        self.peek_at(1).kind(),
                        TokenKind::Newline | TokenKind::Semicolon | TokenKind::LParen
                    )
                    && self.looks_like_arguments_block() =>
            {
                self.parse_arguments_block()?
            }
            _ => self.parse_expr_or_assign()?,
        };

        let suppressed = if *self.peek_kind() == TokenKind::Semicolon {
            self.bump();
            true
        } else {
            if *self.peek_kind() == TokenKind::Comma {
                self.bump();
            }
            false
        };

        Ok(Stmt {
            kind,
            span,
            suppressed,
        })
    }

    fn parse_if(&mut self) -> Result<StmtKind> {
        self.bump(); // `if`
        let cond = self.parse_expr()?;
        let then_body = self.parse_block(&[TokenKind::ElseIf, TokenKind::Else, TokenKind::End])?;

        let mut elseifs = Vec::new();
        while *self.peek_kind() == TokenKind::ElseIf {
            self.bump();
            let c = self.parse_expr()?;
            let b = self.parse_block(&[TokenKind::ElseIf, TokenKind::Else, TokenKind::End])?;
            elseifs.push((c, b));
        }

        let else_body = if *self.peek_kind() == TokenKind::Else {
            self.bump();
            Some(self.parse_block(&[TokenKind::End])?)
        } else {
            None
        };

        self.expect(TokenKind::End, "'end'")?;
        Ok(StmtKind::If {
            cond,
            then_body,
            elseifs,
            else_body,
        })
    }

    fn parse_switch(&mut self) -> Result<StmtKind> {
        self.bump(); // `switch`
        let expr = self.parse_expr()?;
        self.skip_separators();

        let mut cases = Vec::new();
        while *self.peek_kind() == TokenKind::Case {
            self.bump();
            let case_expr = self.parse_expr()?;
            let body = self.parse_block(&[TokenKind::Case, TokenKind::Otherwise, TokenKind::End])?;
            cases.push((case_expr, body));
        }

        let otherwise = if *self.peek_kind() == TokenKind::Otherwise {
            self.bump();
            Some(self.parse_block(&[TokenKind::End])?)
        } else {
            None
        };

        self.expect(TokenKind::End, "'end'")?;
        Ok(StmtKind::Switch {
            expr,
            cases,
            otherwise,
        })
    }

    fn parse_try(&mut self) -> Result<StmtKind> {
        self.bump(); // `try`
        let try_body = self.parse_block(&[TokenKind::Catch, TokenKind::End])?;
        let (catch_var, catch_body) = if *self.peek_kind() == TokenKind::Catch {
            self.bump(); // `catch`
            // Optional exception identifier if immediately followed by newline/semicolon/comma/end
            let var = if let TokenKind::Ident(name) = self.peek_kind().clone() {
                if matches!(
                    self.peek_at(1).kind(),
                    TokenKind::Newline | TokenKind::Semicolon | TokenKind::Comma | TokenKind::End
                ) {
                    self.bump();
                    Some(name)
                } else {
                    None
                }
            } else {
                None
            };
            let body = self.parse_block(&[TokenKind::End])?;
            (var, body)
        } else {
            (None, Vec::new())
        };
        self.expect(TokenKind::End, "'end'")?;
        Ok(StmtKind::TryCatch {
            try_body,
            catch_var,
            catch_body,
        })
    }

    fn parse_while(&mut self) -> Result<StmtKind> {
        self.bump(); // `while`
        let cond = self.parse_expr()?;
        let body = self.parse_block(&[TokenKind::End])?;
        self.expect(TokenKind::End, "'end'")?;
        Ok(StmtKind::While { cond, body })
    }

    fn parse_for(&mut self) -> Result<StmtKind> {
        self.bump(); // `for`
        let var = match self.bump() {
            Token {
                kind: TokenKind::Ident(name),
                ..
            } => name,
            other => {
                return Err(MatroError::syntax(
                    other.span,
                    "Expected loop variable identifier after 'for'",
                ));
            }
        };
        self.expect(TokenKind::Assign, "'='")?;
        let iter = self.parse_expr()?;
        let body = self.parse_block(&[TokenKind::End])?;
        self.expect(TokenKind::End, "'end'")?;
        Ok(StmtKind::For { var, iter, body })
    }

    fn parse_function(&mut self) -> Result<StmtKind> {
        self.bump(); // `function`
        let mut returns = Vec::new();

        if *self.peek_kind() == TokenKind::LBracket {
            self.bump();
            while *self.peek_kind() != TokenKind::RBracket && *self.peek_kind() != TokenKind::Eof {
                match self.bump() {
                    Token {
                        kind: TokenKind::Ident(r),
                        ..
                    } => returns.push(r),
                    other => {
                        return Err(MatroError::syntax(
                            other.span,
                            "Expected return identifier in function signature",
                        ));
                    }
                }
                if *self.peek_kind() == TokenKind::Comma {
                    self.bump();
                }
            }
            self.expect(TokenKind::RBracket, "']'")?;
            self.expect(TokenKind::Assign, "'='")?;
        } else if matches!(self.peek_kind(), TokenKind::Ident(_))
            && *self.peek_at(1).kind() == TokenKind::Assign
        {
            if let TokenKind::Ident(r) = self.bump().kind {
                returns.push(r);
            }
            self.bump(); // `=`
        }

        let mut name = match self.bump() {
            Token {
                kind: TokenKind::Ident(n),
                ..
            } => n,
            other => {
                return Err(MatroError::syntax(other.span, "Expected function name"));
            }
        };
        // Support `get.Prop` and `set.Prop` property accessor method names
        if *self.peek_kind() == TokenKind::Dot && (name == "get" || name == "set") {
            self.bump();
            if let TokenKind::Ident(prop) = self.bump().kind {
                name.push('.');
                name.push_str(&prop);
            }
        }

        let mut params = Vec::new();
        if *self.peek_kind() == TokenKind::LParen {
            self.bump();
            while *self.peek_kind() != TokenKind::RParen && *self.peek_kind() != TokenKind::Eof {
                match self.bump() {
                    Token {
                        kind: TokenKind::Ident(p),
                        ..
                    } => params.push(p),
                    other => {
                        return Err(MatroError::syntax(other.span, "Expected parameter name"));
                    }
                }
                if *self.peek_kind() == TokenKind::Comma {
                    self.bump();
                }
            }
            self.expect(TokenKind::RParen, "')'")?;
        }

        let body = self.parse_block(&[TokenKind::End])?;
        self.expect(TokenKind::End, "'end'")?;

        Ok(StmtKind::Function {
            name,
            params,
            returns,
            body,
        })
    }

    fn parse_classdef(&mut self) -> Result<StmtKind> {
        self.bump(); // `classdef`
        if *self.peek_kind() == TokenKind::LParen {
            self.skip_paren_group()?;
        }
        let name = match self.bump() {
            Token {
                kind: TokenKind::Ident(n),
                ..
            } => n,
            other => {
                return Err(MatroError::syntax(
                    other.span,
                    "Expected class name after 'classdef'",
                ));
            }
        };
        // Optional single or multiple inheritance: `< Base1 & Base2 & pkg.Mixin`
        let mut superclasses = Vec::new();
        if *self.peek_kind() == TokenKind::Lt {
            self.bump();
            loop {
                let mut sup = match self.bump() {
                    Token {
                        kind: TokenKind::Ident(s),
                        ..
                    } => s,
                    other => {
                        return Err(MatroError::syntax(
                            other.span,
                            "Expected superclass name after '<' or '&'",
                        ));
                    }
                };
                while *self.peek_kind() == TokenKind::Dot {
                    self.bump();
                    if let TokenKind::Ident(part) = self.bump().kind {
                        sup.push('.');
                        sup.push_str(&part);
                    }
                }
                superclasses.push(sup);
                if *self.peek_kind() == TokenKind::Amp {
                    self.bump();
                } else {
                    break;
                }
            }
        }
        self.skip_separators();

        let mut properties = Vec::new();
        let mut methods = Vec::new();
        let mut enumerations = Vec::new();
        let mut events = Vec::new();

        while *self.peek_kind() != TokenKind::End && *self.peek_kind() != TokenKind::Eof {
            match self.peek_kind().clone() {
                TokenKind::Ident(ref kw) if kw == "properties" => {
                    self.bump();
                    let attrs = if *self.peek_kind() == TokenKind::LParen {
                        self.parse_attr_names()?
                    } else {
                        Vec::new()
                    };
                    let dependent = attrs.iter().any(|a| a.eq_ignore_ascii_case("Dependent"));
                    let constant = attrs.iter().any(|a| a.eq_ignore_ascii_case("Constant"));
                    self.skip_separators();
                    while *self.peek_kind() != TokenKind::End && *self.peek_kind() != TokenKind::Eof
                    {
                        let prop_name = match self.bump() {
                            Token {
                                kind: TokenKind::Ident(p),
                                ..
                            } => p,
                            other => {
                                return Err(MatroError::syntax(
                                    other.span,
                                    "Expected property name inside 'properties' block",
                                ));
                            }
                        };
                        // Optional MATLAB property size/type/validator annotations before `=`
                        if *self.peek_kind() == TokenKind::LParen {
                            self.skip_paren_group()?;
                        }
                        if matches!(self.peek_kind(), TokenKind::Ident(_)) {
                            self.bump();
                        }
                        if *self.peek_kind() == TokenKind::LBrace {
                            self.skip_brace_group()?;
                        }
                        let default = if *self.peek_kind() == TokenKind::Assign {
                            self.bump();
                            Some(self.parse_expr()?)
                        } else {
                            None
                        };
                        properties.push(PropDef {
                            name: prop_name,
                            default,
                            dependent,
                            constant,
                        });
                        self.skip_separators();
                    }
                    self.expect(TokenKind::End, "'end' for properties block")?;
                }
                TokenKind::Ident(ref kw) if kw == "methods" => {
                    self.bump();
                    let attrs = if *self.peek_kind() == TokenKind::LParen {
                        self.parse_attr_names()?
                    } else {
                        Vec::new()
                    };
                    let is_static = attrs.iter().any(|a| a.eq_ignore_ascii_case("Static"));
                    self.skip_separators();
                    while *self.peek_kind() != TokenKind::End && *self.peek_kind() != TokenKind::Eof
                    {
                        if *self.peek_kind() == TokenKind::Function {
                            if let StmtKind::Function {
                                name,
                                params,
                                returns,
                                body,
                            } = self.parse_function()?
                            {
                                methods.push(MethodDef {
                                    name,
                                    params,
                                    returns,
                                    body,
                                    is_static,
                                });
                            }
                        } else {
                            return Err(MatroError::syntax(
                                self.peek().span,
                                "Expected 'function' inside 'methods' block",
                            ));
                        }
                        self.skip_separators();
                    }
                    self.expect(TokenKind::End, "'end' for methods block")?;
                }
                TokenKind::Ident(ref kw) if kw == "enumeration" => {
                    self.bump();
                    self.skip_separators();
                    while *self.peek_kind() != TokenKind::End && *self.peek_kind() != TokenKind::Eof
                    {
                        let ename = match self.bump() {
                            Token {
                                kind: TokenKind::Ident(e),
                                ..
                            } => e,
                            other => {
                                return Err(MatroError::syntax(
                                    other.span,
                                    "Expected enumeration member name",
                                ));
                            }
                        };
                        let mut eargs = Vec::new();
                        if *self.peek_kind() == TokenKind::LParen {
                            self.bump();
                            while *self.peek_kind() != TokenKind::RParen
                                && *self.peek_kind() != TokenKind::Eof
                            {
                                eargs.push(self.parse_expr()?);
                                if *self.peek_kind() == TokenKind::Comma {
                                    self.bump();
                                } else {
                                    break;
                                }
                            }
                            self.expect(TokenKind::RParen, "')'")?;
                        }
                        enumerations.push((ename, eargs));
                        self.skip_separators();
                    }
                    self.expect(TokenKind::End, "'end' for enumeration block")?;
                }
                TokenKind::Ident(ref kw) if kw == "events" => {
                    self.bump();
                    if *self.peek_kind() == TokenKind::LParen {
                        self.skip_paren_group()?;
                    }
                    self.skip_separators();
                    while *self.peek_kind() != TokenKind::End && *self.peek_kind() != TokenKind::Eof
                    {
                        if let TokenKind::Ident(ev) = self.bump().kind {
                            events.push(ev);
                        }
                        self.skip_separators();
                    }
                    self.expect(TokenKind::End, "'end' for events block")?;
                }
                _ => {
                    return Err(MatroError::syntax(
                        self.peek().span,
                        "Expected 'properties', 'methods', 'enumeration', 'events', or 'end' in classdef",
                    ));
                }
            }
            self.skip_separators();
        }
        self.expect(TokenKind::End, "'end' for classdef")?;
        Ok(StmtKind::ClassDef {
            name,
            superclasses,
            properties,
            methods,
            enumerations,
            events,
        })
    }

    fn parse_attr_names(&mut self) -> Result<Vec<String>> {
        self.expect(TokenKind::LParen, "'('")?;
        let mut names = Vec::new();
        let mut depth = 1usize;
        while depth > 0 && *self.peek_kind() != TokenKind::Eof {
            match self.bump().kind {
                TokenKind::LParen => depth += 1,
                TokenKind::RParen => depth -= 1,
                TokenKind::Ident(n) if depth == 1 => names.push(n),
                _ => {}
            }
        }
        Ok(names)
    }

    fn skip_brace_group(&mut self) -> Result<()> {
        self.expect(TokenKind::LBrace, "'{'")?;
        let mut depth = 1usize;
        while depth > 0 && *self.peek_kind() != TokenKind::Eof {
            match self.bump().kind {
                TokenKind::LBrace => depth += 1,
                TokenKind::RBrace => depth -= 1,
                _ => {}
            }
        }
        Ok(())
    }

    fn looks_like_arguments_block(&self) -> bool {
        match self.peek_at(1).kind() {
            TokenKind::Newline | TokenKind::Semicolon => true,
            TokenKind::LParen => matches!(
                self.peek_at(2).kind(),
                TokenKind::Ident(s) if s.eq_ignore_ascii_case("Repeating")
                    || s.eq_ignore_ascii_case("Input")
                    || s.eq_ignore_ascii_case("Output")
            ),
            _ => false,
        }
    }

    fn parse_arguments_block(&mut self) -> Result<StmtKind> {
        self.bump(); // `arguments`
        let attrs = if *self.peek_kind() == TokenKind::LParen {
            self.parse_attr_names()?
        } else {
            Vec::new()
        };
        let repeating = attrs.iter().any(|a| a.eq_ignore_ascii_case("Repeating"));
        self.skip_separators();

        let mut entries = Vec::new();
        while *self.peek_kind() != TokenKind::End && *self.peek_kind() != TokenKind::Eof {
            let target = match self.bump() {
                Token {
                    kind: TokenKind::Ident(n),
                    ..
                } => n,
                other => {
                    return Err(MatroError::syntax(
                        other.span,
                        "Expected argument name inside 'arguments' block",
                    ));
                }
            };
            let field = if *self.peek_kind() == TokenKind::Dot {
                self.bump();
                match self.bump() {
                    Token {
                        kind: TokenKind::Ident(f),
                        ..
                    } => Some(f),
                    other => {
                        return Err(MatroError::syntax(
                            other.span,
                            "Expected field name after '.' in 'arguments' block",
                        ));
                    }
                }
            } else {
                None
            };
            // Optional size spec `(1, 1)` or `(1, :)`
            if *self.peek_kind() == TokenKind::LParen {
                self.skip_paren_group()?;
            }
            // Optional class name (`double`, `string`, `char`, etc.)
            let class_name = if let TokenKind::Ident(cn) = self.peek_kind().clone() {
                self.bump();
                Some(cn)
            } else {
                None
            };
            // Optional validator cell `{mustBePositive, mustBeMember(x, {'a','b'})}`
            let mut validators = Vec::new();
            if *self.peek_kind() == TokenKind::LBrace {
                self.bump();
                while *self.peek_kind() != TokenKind::RBrace && *self.peek_kind() != TokenKind::Eof
                {
                    if let TokenKind::Ident(vname) = self.peek_kind().clone() {
                        self.bump();
                        let mut vargs = Vec::new();
                        if *self.peek_kind() == TokenKind::LParen {
                            self.bump();
                            while *self.peek_kind() != TokenKind::RParen
                                && *self.peek_kind() != TokenKind::Eof
                            {
                                vargs.push(self.parse_expr()?);
                                if *self.peek_kind() == TokenKind::Comma {
                                    self.bump();
                                } else {
                                    break;
                                }
                            }
                            self.expect(TokenKind::RParen, "')'")?;
                        }
                        validators.push((vname, vargs));
                    } else {
                        self.bump();
                    }
                    if *self.peek_kind() == TokenKind::Comma {
                        self.bump();
                    }
                }
                self.expect(TokenKind::RBrace, "'}'")?;
            }
            let default = if *self.peek_kind() == TokenKind::Assign {
                self.bump();
                Some(self.parse_expr()?)
            } else {
                None
            };
            entries.push(ArgSpec {
                target,
                field,
                class_name,
                validators,
                default,
            });
            self.skip_separators();
        }
        self.expect(TokenKind::End, "'end' for arguments block")?;
        Ok(StmtKind::Arguments { repeating, entries })
    }

    fn parse_scope_decl(&mut self, is_persistent: bool) -> Result<StmtKind> {
        self.bump(); // `global` or `persistent`
        let mut names = Vec::new();
        while let TokenKind::Ident(n) = self.peek_kind().clone() {
            self.bump();
            names.push(n);
            if *self.peek_kind() == TokenKind::Comma {
                self.bump();
            }
        }
        if names.is_empty() {
            return Err(MatroError::syntax(
                self.peek().span,
                "Expected at least one variable name after 'global'/'persistent'",
            ));
        }
        Ok(if is_persistent {
            StmtKind::Persistent(names)
        } else {
            StmtKind::Global(names)
        })
    }

    fn skip_paren_group(&mut self) -> Result<()> {
        self.expect(TokenKind::LParen, "'('")?;
        let mut depth = 1usize;
        while depth > 0 && *self.peek_kind() != TokenKind::Eof {
            match self.bump().kind {
                TokenKind::LParen => depth += 1,
                TokenKind::RParen => depth -= 1,
                _ => {}
            }
        }
        Ok(())
    }

    fn parse_expr_or_assign(&mut self) -> Result<StmtKind> {
        // MATLAB Command Syntax: `hold on`, `grid off`, `save data.mat A B`, `clear x`, `profile report`
        if let TokenKind::Ident(cmd) = self.peek_kind().clone() {
            let next = self.peek_at(1);
            if next.preceded_by_space
                && matches!(
                    next.kind(),
                    TokenKind::Ident(_)
                        | TokenKind::String(_)
                        | TokenKind::Number(_)
                        | TokenKind::True
                        | TokenKind::False
                )
            {
                let span = self.bump().span; // consume command name
                let mut args = Vec::new();
                while !matches!(
                    self.peek_kind(),
                    TokenKind::Newline
                        | TokenKind::Semicolon
                        | TokenKind::Comma
                        | TokenKind::Eof
                ) {
                    let arg_span = self.peek().span;
                    let mut word = match self.bump().kind {
                        TokenKind::String(s) => s,
                        TokenKind::Ident(s) => s,
                        TokenKind::Number(n) => format!("{n}"),
                        TokenKind::True => "true".to_string(),
                        TokenKind::False => "false".to_string(),
                        TokenKind::Minus => "-".to_string(),
                        TokenKind::Slash => "/".to_string(),
                        TokenKind::Dot => ".".to_string(),
                        other => {
                            return Err(MatroError::syntax(
                                arg_span,
                                format!("Unexpected token in command syntax: {other:?}"),
                            ));
                        }
                    };
                    // Join contiguous unspaced tokens into a single command argument (e.g., `data.mat`, `lib/foo.m`, `-v6`)
                    while !self.peek().preceded_by_space
                        && !matches!(
                            self.peek_kind(),
                            TokenKind::Newline
                                | TokenKind::Semicolon
                                | TokenKind::Comma
                                | TokenKind::Eof
                        )
                    {
                        match self.bump().kind {
                            TokenKind::Ident(s) => word.push_str(&s),
                            TokenKind::Number(n) => word.push_str(&format!("{n}")),
                            TokenKind::Dot => word.push('.'),
                            TokenKind::DotStar => word.push_str(".*"),
                            TokenKind::Star => word.push('*'),
                            TokenKind::Slash => word.push('/'),
                            TokenKind::Minus => word.push('-'),
                            TokenKind::Plus => word.push('+'),
                            TokenKind::String(s) => word.push_str(&s),
                            _ => break,
                        }
                    }
                    args.push(IndexArg::Expr(Expr {
                        kind: ExprKind::String(word),
                        span: arg_span,
                    }));
                }
                return Ok(StmtKind::Expr(Expr {
                    kind: ExprKind::CallOrIndex { callee: cmd, args },
                    span,
                }));
            }
        }

        // Check for multi-return assignment `[a, b] = expr`
        if *self.peek_kind() == TokenKind::LBracket {
            let saved = self.pos;
            if let Some(targets) = self.try_parse_multi_assign_lhs() {
                if *self.peek_kind() == TokenKind::Assign {
                    self.bump();
                    let value = self.parse_expr()?;
                    return Ok(StmtKind::MultiAssign { targets, value });
                }
            }
            self.pos = saved;
        }

        let expr = self.parse_expr()?;
        if *self.peek_kind() == TokenKind::Assign {
            let span = self.peek().span;
            self.bump();
            let value = self.parse_expr()?;
            let target = match expr.kind {
                ExprKind::Ident(name) => AssignTarget::Var(name),
                ExprKind::CallOrIndex { callee, args } => AssignTarget::Index {
                    name: callee,
                    indices: args,
                },
                ExprKind::CellIndex { target, indices } => match target.kind {
                    ExprKind::Ident(name) => AssignTarget::CellIndex { name, indices },
                    _ => return Err(MatroError::syntax(span, "Invalid cell assignment target")),
                },
                ExprKind::FieldAccess { .. } => {
                    let mut fields = Vec::new();
                    let mut cur = expr;
                    loop {
                        match cur.kind {
                            ExprKind::FieldAccess { object, field } => {
                                fields.push(field);
                                cur = *object;
                            }
                            ExprKind::Ident(name) => {
                                fields.reverse();
                                break AssignTarget::Field { name, fields };
                            }
                            ExprKind::CallOrIndex { callee, args } => {
                                fields.reverse();
                                break AssignTarget::IndexedField {
                                    name: callee,
                                    indices: args,
                                    fields,
                                };
                            }
                            _ => {
                                return Err(MatroError::syntax(
                                    span,
                                    "Invalid struct field assignment target",
                                ));
                            }
                        }
                    }
                }
                ExprKind::MethodCall {
                    object,
                    method,
                    args,
                } => {
                    let mut fields = vec![method];
                    let mut cur = *object;
                    loop {
                        match cur.kind {
                            ExprKind::Ident(n) => {
                                fields.reverse();
                                break AssignTarget::FieldIndex {
                                    name: n,
                                    fields,
                                    indices: args,
                                };
                            }
                            ExprKind::FieldAccess { object: o, field: f } => {
                                fields.push(f);
                                cur = *o;
                            }
                            _ => {
                                return Err(MatroError::syntax(
                                    span,
                                    "Invalid field-index assignment target",
                                ));
                            }
                        }
                    }
                }
                _ => {
                    return Err(MatroError::syntax(span, "Invalid assignment target"));
                }
            };
            Ok(StmtKind::Assign { target, value })
        } else {
            Ok(StmtKind::Expr(expr))
        }
    }

    fn try_parse_multi_assign_lhs(&mut self) -> Option<Vec<String>> {
        if *self.peek_kind() != TokenKind::LBracket {
            return None;
        }
        self.bump();
        let mut names = Vec::new();
        while *self.peek_kind() != TokenKind::RBracket {
            match self.peek_kind() {
                TokenKind::Ident(name) => {
                    names.push(name.clone());
                    self.bump();
                }
                TokenKind::Tilde => {
                    names.push("~".to_string());
                    self.bump();
                }
                _ => return None,
            }
            if *self.peek_kind() == TokenKind::Comma {
                self.bump();
            }
        }
        self.bump(); // `]`
        if names.is_empty() {
            None
        } else {
            Some(names)
        }
    }

    pub fn parse_expr(&mut self) -> Result<Expr> {
        if *self.peek_kind() == TokenKind::At {
            return self.parse_anon_func();
        }
        self.parse_short_or()
    }

    fn parse_anon_func(&mut self) -> Result<Expr> {
        let span = self.bump().span; // `@`
        // `@(x, y) expr` or `@sin` (function handle to existing function)
        if *self.peek_kind() == TokenKind::LParen {
            self.bump();
            let mut params = Vec::new();
            while *self.peek_kind() != TokenKind::RParen && *self.peek_kind() != TokenKind::Eof {
                match self.bump() {
                    Token {
                        kind: TokenKind::Ident(p),
                        ..
                    } => params.push(p),
                    other => {
                        return Err(MatroError::syntax(
                            other.span,
                            "Expected parameter name in anonymous function",
                        ));
                    }
                }
                if *self.peek_kind() == TokenKind::Comma {
                    self.bump();
                }
            }
            self.expect(TokenKind::RParen, "')'")?;
            let body = self.parse_expr()?;
            Ok(Expr {
                kind: ExprKind::AnonFunc {
                    params,
                    body: Box::new(body),
                },
                span,
            })
        } else if let TokenKind::Ident(fname) = self.peek_kind().clone() {
            // Shorthand `@fname` -> `@(varargin) fname(varargin{:})` (supports any arity!)
            self.bump();
            let param = "varargin".to_string();
            let spread_arg = Expr {
                kind: ExprKind::CellIndex {
                    target: Box::new(Expr {
                        kind: ExprKind::Ident(param.clone()),
                        span,
                    }),
                    indices: vec![IndexArg::Colon],
                },
                span,
            };
            let body = Expr {
                kind: ExprKind::CallOrIndex {
                    callee: fname,
                    args: vec![IndexArg::Expr(spread_arg)],
                },
                span,
            };
            Ok(Expr {
                kind: ExprKind::AnonFunc {
                    params: vec![param],
                    body: Box::new(body),
                },
                span,
            })
        } else {
            Err(MatroError::syntax(
                span,
                "Expected '(' or function name after '@'",
            ))
        }
    }

    fn parse_short_or(&mut self) -> Result<Expr> {
        let mut left = self.parse_short_and()?;
        while *self.peek_kind() == TokenKind::PipePipe {
            let span = self.bump().span;
            let right = self.parse_short_and()?;
            left = Expr {
                kind: ExprKind::Binary {
                    op: BinaryOp::ShortOr,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            };
        }
        Ok(left)
    }

    fn parse_short_and(&mut self) -> Result<Expr> {
        let mut left = self.parse_elem_or()?;
        while *self.peek_kind() == TokenKind::AmpAmp {
            let span = self.bump().span;
            let right = self.parse_elem_or()?;
            left = Expr {
                kind: ExprKind::Binary {
                    op: BinaryOp::ShortAnd,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            };
        }
        Ok(left)
    }

    fn parse_elem_or(&mut self) -> Result<Expr> {
        let mut left = self.parse_elem_and()?;
        while *self.peek_kind() == TokenKind::Pipe {
            let span = self.bump().span;
            let right = self.parse_elem_and()?;
            left = Expr {
                kind: ExprKind::Binary {
                    op: BinaryOp::Or,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            };
        }
        Ok(left)
    }

    fn parse_elem_and(&mut self) -> Result<Expr> {
        let mut left = self.parse_comparison()?;
        while *self.peek_kind() == TokenKind::Amp {
            let span = self.bump().span;
            let right = self.parse_comparison()?;
            left = Expr {
                kind: ExprKind::Binary {
                    op: BinaryOp::And,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            };
        }
        Ok(left)
    }

    fn parse_comparison(&mut self) -> Result<Expr> {
        let mut left = self.parse_range()?;
        loop {
            let op = match self.peek_kind() {
                TokenKind::EqEq => BinaryOp::Eq,
                TokenKind::NotEq => BinaryOp::Ne,
                TokenKind::Lt => BinaryOp::Lt,
                TokenKind::Le => BinaryOp::Le,
                TokenKind::Gt => BinaryOp::Gt,
                TokenKind::Ge => BinaryOp::Ge,
                _ => break,
            };
            let span = self.bump().span;
            let right = self.parse_range()?;
            left = Expr {
                kind: ExprKind::Binary {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            };
        }
        Ok(left)
    }

    fn parse_range(&mut self) -> Result<Expr> {
        let start = self.parse_additive()?;
        if *self.peek_kind() == TokenKind::Colon {
            let span = self.bump().span;
            let second = self.parse_additive()?;
            if *self.peek_kind() == TokenKind::Colon {
                self.bump();
                let stop = self.parse_additive()?;
                return Ok(Expr {
                    kind: ExprKind::Range {
                        start: Box::new(start),
                        step: Some(Box::new(second)),
                        stop: Box::new(stop),
                    },
                    span,
                });
            }
            return Ok(Expr {
                kind: ExprKind::Range {
                    start: Box::new(start),
                    step: None,
                    stop: Box::new(second),
                },
                span,
            });
        }
        Ok(start)
    }

    fn parse_additive(&mut self) -> Result<Expr> {
        let mut left = self.parse_multiplicative()?;
        loop {
            if self.in_matrix
                && matches!(self.peek_kind(), TokenKind::Plus | TokenKind::Minus)
                && self.peek().preceded_by_space
                && !self.peek_at(1).preceded_by_space
            {
                break;
            }
            let op = match self.peek_kind() {
                TokenKind::Plus => BinaryOp::Add,
                TokenKind::Minus => BinaryOp::Sub,
                _ => break,
            };
            let span = self.bump().span;
            let right = self.parse_multiplicative()?;
            left = Expr {
                kind: ExprKind::Binary {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            };
        }
        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> Result<Expr> {
        let mut left = self.parse_unary()?;
        loop {
            let op = match self.peek_kind() {
                TokenKind::Star => BinaryOp::Mul,
                TokenKind::DotStar => BinaryOp::ElemMul,
                TokenKind::Slash => BinaryOp::Div,
                TokenKind::DotSlash => BinaryOp::ElemDiv,
                TokenKind::Backslash => BinaryOp::LeftDiv,
                _ => break,
            };
            let span = self.bump().span;
            let right = self.parse_unary()?;
            left = Expr {
                kind: ExprKind::Binary {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            };
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr> {
        let op = match self.peek_kind() {
            TokenKind::Minus => Some(UnaryOp::Neg),
            TokenKind::Plus => Some(UnaryOp::Pos),
            TokenKind::Tilde => Some(UnaryOp::Not),
            _ => None,
        };
        if let Some(op) = op {
            let span = self.bump().span;
            let expr = self.parse_unary()?;
            return Ok(Expr {
                kind: ExprKind::Unary {
                    op,
                    expr: Box::new(expr),
                },
                span,
            });
        }
        self.parse_power()
    }

    fn parse_power(&mut self) -> Result<Expr> {
        let mut left = self.parse_postfix()?;
        loop {
            let op = match self.peek_kind() {
                TokenKind::Caret => BinaryOp::Pow,
                TokenKind::DotCaret => BinaryOp::ElemPow,
                _ => break,
            };
            let span = self.bump().span;
            let right = self.parse_unary()?;
            left = Expr {
                kind: ExprKind::Binary {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            };
        }
        Ok(left)
    }

    fn parse_postfix(&mut self) -> Result<Expr> {
        let mut expr = self.parse_primary()?;
        loop {
            match self.peek_kind() {
                TokenKind::CTranspose => {
                    let span = self.bump().span;
                    expr = Expr {
                        kind: ExprKind::CTranspose(Box::new(expr)),
                        span,
                    };
                }
                TokenKind::Transpose => {
                    let span = self.bump().span;
                    expr = Expr {
                        kind: ExprKind::Transpose(Box::new(expr)),
                        span,
                    };
                }
                TokenKind::Dot => {
                    let span = self.bump().span;
                    let field = match self.bump() {
                        Token {
                            kind: TokenKind::Ident(f),
                            ..
                        } => f,
                        other => {
                            return Err(MatroError::syntax(
                                other.span,
                                "Expected field name after '.'",
                            ));
                        }
                    };
                    expr = Expr {
                        kind: ExprKind::FieldAccess {
                            object: Box::new(expr),
                            field,
                        },
                        span,
                    };
                }
                TokenKind::LBrace if !self.in_matrix || !self.peek().preceded_by_space => {
                    let span = self.bump().span;
                    let args = self.parse_index_args(TokenKind::RBrace, "'}'")?;
                    expr = Expr {
                        kind: ExprKind::CellIndex {
                            target: Box::new(expr),
                            indices: args,
                        },
                        span,
                    };
                }
                TokenKind::LParen if !self.in_matrix || !self.peek().preceded_by_space => {
                    let span = self.bump().span;
                    let args = self.parse_index_args(TokenKind::RParen, "')'")?;
                    expr = match expr.kind {
                        ExprKind::Ident(callee) => Expr {
                            kind: ExprKind::CallOrIndex { callee, args },
                            span,
                        },
                        ExprKind::FieldAccess { object, field } => Expr {
                            kind: ExprKind::MethodCall {
                                object,
                                method: field,
                                args,
                            },
                            span,
                        },
                        _ => Expr {
                            kind: ExprKind::DynamicCall {
                                target: Box::new(expr),
                                args,
                            },
                            span,
                        },
                    };
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_index_args(&mut self, close: TokenKind, close_label: &str) -> Result<Vec<IndexArg>> {
        let prev_in_matrix = self.in_matrix;
        self.in_matrix = false;
        let mut args = Vec::new();
        while *self.peek_kind() != close && *self.peek_kind() != TokenKind::Eof {
            if *self.peek_kind() == TokenKind::Colon
                && (*self.peek_at(1).kind() == TokenKind::Comma
                    || *self.peek_at(1).kind() == close)
            {
                self.bump();
                args.push(IndexArg::Colon);
            } else if close == TokenKind::RParen
                && matches!(self.peek_kind(), TokenKind::Ident(_))
                && *self.peek_at(1).kind() == TokenKind::Assign
            {
                // MATLAB R2021a+ Name=Value argument syntax: desugar `Key=Val` -> `'Key', Val`
                let key_tok = self.bump();
                self.bump(); // `=`
                let val_expr = self.parse_expr()?;
                if let TokenKind::Ident(k) = key_tok.kind {
                    args.push(IndexArg::Expr(Expr {
                        kind: ExprKind::String(k),
                        span: key_tok.span,
                    }));
                }
                args.push(IndexArg::Expr(val_expr));
            } else {
                args.push(IndexArg::Expr(self.parse_expr()?));
            }
            if *self.peek_kind() == TokenKind::Comma {
                self.bump();
            } else {
                break;
            }
        }
        self.expect(close, close_label)?;
        self.in_matrix = prev_in_matrix;
        Ok(args)
    }

    fn parse_primary(&mut self) -> Result<Expr> {
        let tok = self.peek().clone();
        match tok.kind {
            TokenKind::Number(n) => {
                self.bump();
                Ok(Expr {
                    kind: ExprKind::Number(n),
                    span: tok.span,
                })
            }
            TokenKind::ImagNumber(im) => {
                self.bump();
                Ok(Expr {
                    kind: ExprKind::ImagNumber(im),
                    span: tok.span,
                })
            }
            TokenKind::True => {
                self.bump();
                Ok(Expr {
                    kind: ExprKind::Bool(true),
                    span: tok.span,
                })
            }
            TokenKind::False => {
                self.bump();
                Ok(Expr {
                    kind: ExprKind::Bool(false),
                    span: tok.span,
                })
            }
            TokenKind::String(s) => {
                self.bump();
                Ok(Expr {
                    kind: ExprKind::String(s),
                    span: tok.span,
                })
            }
            TokenKind::End => {
                self.bump();
                Ok(Expr {
                    kind: ExprKind::EndKeyword,
                    span: tok.span,
                })
            }
            TokenKind::Ident(name) => {
                self.bump();
                Ok(Expr {
                    kind: ExprKind::Ident(name),
                    span: tok.span,
                })
            }
            TokenKind::LParen => {
                let prev_in_matrix = self.in_matrix;
                self.in_matrix = false;
                self.bump();
                let expr = self.parse_expr()?;
                self.expect(TokenKind::RParen, "')'")?;
                self.in_matrix = prev_in_matrix;
                Ok(expr)
            }
            TokenKind::LBracket => self.parse_matrix_literal(),
            TokenKind::LBrace => self.parse_cell_literal(),
            TokenKind::At => self.parse_anon_func(),
            _ => Err(MatroError::syntax(
                tok.span,
                format!("Unexpected token {:?}", tok.kind),
            )),
        }
    }

    fn parse_matrix_literal(&mut self) -> Result<Expr> {
        let span = self.bump().span; // `[`
        let rows = self.parse_grid_body(TokenKind::RBracket, "']'")?;
        Ok(Expr {
            kind: ExprKind::Matrix(rows),
            span,
        })
    }

    fn parse_cell_literal(&mut self) -> Result<Expr> {
        let span = self.bump().span; // `{`
        let rows = self.parse_grid_body(TokenKind::RBrace, "'}'")?;
        Ok(Expr {
            kind: ExprKind::Cell(rows),
            span,
        })
    }

    fn parse_grid_body(&mut self, close: TokenKind, label: &str) -> Result<Vec<Vec<Expr>>> {
        let prev_in_matrix = self.in_matrix;
        self.in_matrix = true;

        let mut rows: Vec<Vec<Expr>> = Vec::new();
        let mut current_row: Vec<Expr> = Vec::new();

        self.skip_newlines();
        while *self.peek_kind() != close && *self.peek_kind() != TokenKind::Eof {
            if matches!(self.peek_kind(), TokenKind::Semicolon | TokenKind::Newline) {
                self.bump();
                if !current_row.is_empty() {
                    rows.push(std::mem::take(&mut current_row));
                }
                self.skip_newlines();
                continue;
            }
            current_row.push(self.parse_expr()?);
            if *self.peek_kind() == TokenKind::Comma {
                self.bump();
            }
        }
        if !current_row.is_empty() {
            rows.push(current_row);
        }

        self.expect(close, label)?;
        self.in_matrix = prev_in_matrix;
        Ok(rows)
    }
}

impl Token {
    pub fn kind(&self) -> &TokenKind {
        &self.kind
    }
}
