use matro_types::{MatroError, Result, Span};

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    Number(f64),
    ImagNumber(f64),
    String(String),
    Ident(String),

    // Keywords
    If,
    ElseIf,
    Else,
    Switch,
    Case,
    Otherwise,
    Try,
    Catch,
    End,
    While,
    For,
    Function,
    ClassDef,
    Global,
    Persistent,
    Break,
    Continue,
    Return,
    True,
    False,

    // Operators
    Plus,
    Minus,
    Star,
    DotStar,
    Slash,
    DotSlash,
    Backslash,
    DotBackslash,
    Caret,
    DotCaret,
    CTranspose, // '
    Transpose,  // .'
    Dot,        // .
    At,         // @
    EqEq,
    NotEq,
    Lt,
    Le,
    Gt,
    Ge,
    Amp,
    AmpAmp,
    Pipe,
    PipePipe,
    Tilde,
    Assign,

    // Delimiters
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Semicolon,
    Colon,
    Newline,
    Eof,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    pub preceded_by_space: bool,
}

pub struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
    line: usize,
    col: usize,
    /// Tracks whether the previous token can be followed by a transpose `'` rather than a string literal.
    can_transpose: bool,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            src: input.as_bytes(),
            pos: 0,
            line: 1,
            col: 1,
            can_transpose: false,
        }
    }

    pub fn tokenize(mut self) -> Result<Vec<Token>> {
        let mut tokens = Vec::new();
        loop {
            let tok = self.next_token()?;
            let is_eof = tok.kind == TokenKind::Eof;
            tokens.push(tok);
            if is_eof {
                break;
            }
        }
        Ok(tokens)
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn peek_at(&self, offset: usize) -> Option<u8> {
        self.src.get(self.pos + offset).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        if b == b'\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(b)
    }

    fn span(&self) -> Span {
        Span::new(self.line, self.col)
    }

    fn next_token(&mut self) -> Result<Token> {
        let mut preceded_by_space = false;

        loop {
            match self.peek() {
                Some(b' ' | b'\t' | b'\r') => {
                    preceded_by_space = true;
                    self.bump();
                }
                // Line continuation `...`
                Some(b'.') if self.peek_at(1) == Some(b'.') && self.peek_at(2) == Some(b'.') => {
                    preceded_by_space = true;
                    self.bump();
                    self.bump();
                    self.bump();
                    while let Some(c) = self.peek() {
                        self.bump();
                        if c == b'\n' {
                            break;
                        }
                    }
                }
                // Comments: `%` or `#` (Octave)
                Some(b'%' | b'#') => {
                    while let Some(c) = self.peek() {
                        if c == b'\n' {
                            break;
                        }
                        self.bump();
                    }
                }
                _ => break,
            }
        }

        let span = self.span();
        let Some(ch) = self.peek() else {
            return Ok(Token {
                kind: TokenKind::Eof,
                span,
                preceded_by_space,
            });
        };

        let kind = match ch {
            b'\n' => {
                self.bump();
                self.can_transpose = false;
                TokenKind::Newline
            }
            b'(' => {
                self.bump();
                self.can_transpose = false;
                TokenKind::LParen
            }
            b')' => {
                self.bump();
                self.can_transpose = true;
                TokenKind::RParen
            }
            b'[' => {
                self.bump();
                self.can_transpose = false;
                TokenKind::LBracket
            }
            b']' => {
                self.bump();
                self.can_transpose = true;
                TokenKind::RBracket
            }
            b'{' => {
                self.bump();
                self.can_transpose = false;
                TokenKind::LBrace
            }
            b'}' => {
                self.bump();
                self.can_transpose = true;
                TokenKind::RBrace
            }
            b'@' => {
                self.bump();
                self.can_transpose = false;
                TokenKind::At
            }
            b',' => {
                self.bump();
                self.can_transpose = false;
                TokenKind::Comma
            }
            b';' => {
                self.bump();
                self.can_transpose = false;
                TokenKind::Semicolon
            }
            b':' => {
                self.bump();
                self.can_transpose = false;
                TokenKind::Colon
            }
            b'+' => {
                self.bump();
                self.can_transpose = false;
                TokenKind::Plus
            }
            b'-' => {
                self.bump();
                self.can_transpose = false;
                TokenKind::Minus
            }
            b'*' => {
                self.bump();
                self.can_transpose = false;
                TokenKind::Star
            }
            b'/' => {
                self.bump();
                self.can_transpose = false;
                TokenKind::Slash
            }
            b'\\' => {
                self.bump();
                self.can_transpose = false;
                TokenKind::Backslash
            }
            b'^' => {
                self.bump();
                self.can_transpose = false;
                TokenKind::Caret
            }
            b'.' => match self.peek_at(1) {
                Some(b'*') => {
                    self.bump();
                    self.bump();
                    self.can_transpose = false;
                    TokenKind::DotStar
                }
                Some(b'/') => {
                    self.bump();
                    self.bump();
                    self.can_transpose = false;
                    TokenKind::DotSlash
                }
                Some(b'\\') => {
                    self.bump();
                    self.bump();
                    self.can_transpose = false;
                    TokenKind::DotBackslash
                }
                Some(b'^') => {
                    self.bump();
                    self.bump();
                    self.can_transpose = false;
                    TokenKind::DotCaret
                }
                Some(b'\'') => {
                    self.bump();
                    self.bump();
                    self.can_transpose = true;
                    TokenKind::Transpose
                }
                Some(d) if d.is_ascii_digit() => self.lex_number(span)?,
                _ => {
                    self.bump();
                    self.can_transpose = false;
                    TokenKind::Dot
                }
            },
            b'\'' => {
                if self.can_transpose {
                    self.bump();
                    self.can_transpose = true;
                    TokenKind::CTranspose
                } else {
                    self.lex_string(b'\'', span)?
                }
            }
            b'"' => self.lex_string(b'"', span)?,
            b'=' => {
                self.bump();
                self.can_transpose = false;
                if self.peek() == Some(b'=') {
                    self.bump();
                    TokenKind::EqEq
                } else {
                    TokenKind::Assign
                }
            }
            b'~' | b'!' => {
                self.bump();
                self.can_transpose = false;
                if self.peek() == Some(b'=') {
                    self.bump();
                    TokenKind::NotEq
                } else {
                    TokenKind::Tilde
                }
            }
            b'<' => {
                self.bump();
                self.can_transpose = false;
                if self.peek() == Some(b'=') {
                    self.bump();
                    TokenKind::Le
                } else {
                    TokenKind::Lt
                }
            }
            b'>' => {
                self.bump();
                self.can_transpose = false;
                if self.peek() == Some(b'=') {
                    self.bump();
                    TokenKind::Ge
                } else {
                    TokenKind::Gt
                }
            }
            b'&' => {
                self.bump();
                self.can_transpose = false;
                if self.peek() == Some(b'&') {
                    self.bump();
                    TokenKind::AmpAmp
                } else {
                    TokenKind::Amp
                }
            }
            b'|' => {
                self.bump();
                self.can_transpose = false;
                if self.peek() == Some(b'|') {
                    self.bump();
                    TokenKind::PipePipe
                } else {
                    TokenKind::Pipe
                }
            }
            c if c.is_ascii_digit() => self.lex_number(span)?,
            c if c.is_ascii_alphabetic() || c == b'_' => self.lex_ident(),
            other => {
                return Err(MatroError::syntax(
                    span,
                    format!("Unexpected character '{}'", other as char),
                ));
            }
        };

        Ok(Token {
            kind,
            span,
            preceded_by_space,
        })
    }

    fn lex_number(&mut self, span: Span) -> Result<TokenKind> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                self.bump();
            } else {
                break;
            }
        }
        // Decimal point (but not `.*`, `./`, `.^`, `.'`, `..`)
        if self.peek() == Some(b'.')
            && !matches!(
                self.peek_at(1),
                Some(b'*' | b'/' | b'^' | b'\'' | b'.')
            )
        {
            self.bump();
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    self.bump();
                } else {
                    break;
                }
            }
        }
        // Scientific notation `1e-5`, `2.5E+3`
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.bump();
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.bump();
            }
            let exp_start = self.pos;
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    self.bump();
                } else {
                    break;
                }
            }
            if self.pos == exp_start {
                return Err(MatroError::syntax(span, "Invalid numeric exponent"));
            }
        }
        let text = std::str::from_utf8(&self.src[start..self.pos])
            .map_err(|_| MatroError::syntax(span, "Invalid UTF-8 in number"))?;
        let val = text
            .parse::<f64>()
            .map_err(|_| MatroError::syntax(span, format!("Invalid number '{text}'")))?;

        // MATLAB imaginary suffix: `4i`, `2.5j` (not followed by alphanumeric/`_`)
        let is_imag = matches!(self.peek(), Some(b'i' | b'j'))
            && !self
                .peek_at(1)
                .is_some_and(|c| c.is_ascii_alphanumeric() || c == b'_');
        if is_imag {
            self.bump();
            self.can_transpose = true;
            return Ok(TokenKind::ImagNumber(val));
        }

        self.can_transpose = true;
        Ok(TokenKind::Number(val))
    }

    fn lex_string(&mut self, quote: u8, span: Span) -> Result<TokenKind> {
        self.bump(); // opening quote
        let mut s = String::new();
        loop {
            match self.bump() {
                None => return Err(MatroError::syntax(span, "Unterminated string literal")),
                Some(b'\n') => return Err(MatroError::syntax(span, "Unterminated string literal")),
                Some(c) if c == quote => {
                    if self.peek() == Some(quote) {
                        self.bump();
                        s.push(quote as char);
                    } else {
                        break;
                    }
                }
                Some(b'\\') if quote == b'"' => match self.bump() {
                    Some(b'n') => s.push('\n'),
                    Some(b't') => s.push('\t'),
                    Some(b'r') => s.push('\r'),
                    Some(b'\\') => s.push('\\'),
                    Some(b'"') => s.push('"'),
                    Some(other) => {
                        s.push('\\');
                        s.push(other as char);
                    }
                    None => return Err(MatroError::syntax(span, "Unterminated escape sequence")),
                },
                Some(c) => s.push(c as char),
            }
        }
        self.can_transpose = true;
        Ok(TokenKind::String(s))
    }

    fn lex_ident(&mut self) -> TokenKind {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == b'_' {
                self.bump();
            } else {
                break;
            }
        }
        let text = std::str::from_utf8(&self.src[start..self.pos]).unwrap_or("");
        let (kind, can_transpose) = match text {
            "if" => (TokenKind::If, false),
            "elseif" => (TokenKind::ElseIf, false),
            "else" => (TokenKind::Else, false),
            "switch" => (TokenKind::Switch, false),
            "case" => (TokenKind::Case, false),
            "otherwise" => (TokenKind::Otherwise, false),
            "try" => (TokenKind::Try, false),
            "catch" => (TokenKind::Catch, false),
            "end" | "endif" | "endfor" | "endwhile" | "endfunction" | "endswitch"
            | "end_try_catch" | "endclassdef" | "endproperties" | "endmethods" => {
                (TokenKind::End, true)
            }
            "while" => (TokenKind::While, false),
            "for" => (TokenKind::For, false),
            "function" => (TokenKind::Function, false),
            "classdef" => (TokenKind::ClassDef, false),
            "global" => (TokenKind::Global, false),
            "persistent" => (TokenKind::Persistent, false),
            "break" => (TokenKind::Break, false),
            "continue" => (TokenKind::Continue, false),
            "return" => (TokenKind::Return, false),
            "true" => (TokenKind::True, true),
            "false" => (TokenKind::False, true),
            _ => (TokenKind::Ident(text.to_string()), true),
        };
        self.can_transpose = can_transpose;
        kind
    }
}
