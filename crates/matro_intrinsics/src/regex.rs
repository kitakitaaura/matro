//! Zero-dependency pure-Rust regular expression engine for MATLAB `regexp`, `regexpi`, `regexprep`.
//!
//! Supports: literals, `.`, `^`, `$`, `[a-z0-9]`, `[^...]`, `\d`, `\D`, `\w`, `\W`, `\s`, `\S`,
//! quantifiers `*`, `+`, `?`, `*?`, `+?`, `{m}`, `{m,n}`, grouping `(...)`, and alternation `a|b`.

use matro_types::{CellArray, MatroError, Matrix, Result, Value};

#[derive(Debug, Clone)]
enum Atom {
    Literal(char),
    Any,
    Digit(bool),
    Word(bool),
    Space(bool),
    Class { negated: bool, ranges: Vec<(char, char)> },
    Group(Vec<Vec<Piece>>),
    AnchorStart,
    AnchorEnd,
}

#[derive(Debug, Clone)]
struct Piece {
    atom: Atom,
    min: usize,
    max: Option<usize>,
    lazy: bool,
}

struct Parser<'a> {
    chars: Vec<char>,
    pos: usize,
    _pat: &'a str,
}

impl<'a> Parser<'a> {
    fn new(pat: &'a str) -> Self {
        Self {
            chars: pat.chars().collect(),
            pos: 0,
            _pat: pat,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += 1;
        Some(c)
    }

    fn parse_alts(&mut self) -> Result<Vec<Vec<Piece>>> {
        let mut alts = vec![self.parse_seq()?];
        while self.peek() == Some('|') {
            self.bump();
            alts.push(self.parse_seq()?);
        }
        Ok(alts)
    }

    fn parse_seq(&mut self) -> Result<Vec<Piece>> {
        let mut seq = Vec::new();
        while let Some(c) = self.peek() {
            if c == '|' || c == ')' {
                break;
            }
            let atom = self.parse_atom()?;
            let (min, max, lazy) = self.parse_quantifier()?;
            seq.push(Piece {
                atom,
                min,
                max,
                lazy,
            });
        }
        Ok(seq)
    }

    fn parse_atom(&mut self) -> Result<Atom> {
        let ch = self.bump().ok_or_else(|| MatroError::runtime("regexp: unexpected end"))?;
        match ch {
            '.' => Ok(Atom::Any),
            '^' => Ok(Atom::AnchorStart),
            '$' => Ok(Atom::AnchorEnd),
            '(' => {
                // Skip optional non-capturing `(?:`
                if self.peek() == Some('?') && self.chars.get(self.pos + 1) == Some(&':') {
                    self.bump();
                    self.bump();
                }
                let inner = self.parse_alts()?;
                if self.bump() != Some(')') {
                    return Err(MatroError::runtime("regexp: unclosed '('"));
                }
                Ok(Atom::Group(inner))
            }
            '[' => self.parse_class(),
            '\\' => {
                let esc = self
                    .bump()
                    .ok_or_else(|| MatroError::runtime("regexp: trailing '\\'"))?;
                Ok(match esc {
                    'd' => Atom::Digit(false),
                    'D' => Atom::Digit(true),
                    'w' => Atom::Word(false),
                    'W' => Atom::Word(true),
                    's' => Atom::Space(false),
                    'S' => Atom::Space(true),
                    'n' => Atom::Literal('\n'),
                    't' => Atom::Literal('\t'),
                    'r' => Atom::Literal('\r'),
                    other => Atom::Literal(other),
                })
            }
            other => Ok(Atom::Literal(other)),
        }
    }

    fn parse_class(&mut self) -> Result<Atom> {
        let negated = if self.peek() == Some('^') {
            self.bump();
            true
        } else {
            false
        };
        let mut ranges = Vec::new();
        while let Some(c) = self.bump() {
            if c == ']' {
                return Ok(Atom::Class { negated, ranges });
            }
            let start = if c == '\\' {
                match self.bump() {
                    Some('d') => {
                        ranges.push(('0', '9'));
                        continue;
                    }
                    Some('w') => {
                        ranges.push(('a', 'z'));
                        ranges.push(('A', 'Z'));
                        ranges.push(('0', '9'));
                        ranges.push(('_', '_'));
                        continue;
                    }
                    Some('s') => {
                        ranges.push((' ', ' '));
                        ranges.push(('\t', '\t'));
                        ranges.push(('\n', '\n'));
                        ranges.push(('\r', '\r'));
                        continue;
                    }
                    Some(x) => x,
                    None => return Err(MatroError::runtime("regexp: unclosed '['")),
                }
            } else {
                c
            };
            if self.peek() == Some('-') && self.chars.get(self.pos + 1) != Some(&']') {
                self.bump(); // `-`
                let end = self
                    .bump()
                    .ok_or_else(|| MatroError::runtime("regexp: unclosed '['"))?;
                ranges.push((start, end));
            } else {
                ranges.push((start, start));
            }
        }
        Err(MatroError::runtime("regexp: unclosed '['"))
    }

    fn parse_quantifier(&mut self) -> Result<(usize, Option<usize>, bool)> {
        let (min, max) = match self.peek() {
            Some('*') => {
                self.bump();
                (0, None)
            }
            Some('+') => {
                self.bump();
                (1, None)
            }
            Some('?') => {
                self.bump();
                (0, Some(1))
            }
            Some('{') => {
                self.bump();
                let m = self.read_uint()?;
                let max = if self.peek() == Some(',') {
                    self.bump();
                    if self.peek() == Some('}') {
                        None
                    } else {
                        Some(self.read_uint()?)
                    }
                } else {
                    Some(m)
                };
                if self.bump() != Some('}') {
                    return Err(MatroError::runtime("regexp: unclosed '{' quantifier"));
                }
                (m, max)
            }
            _ => return Ok((1, Some(1), false)),
        };
        let lazy = if self.peek() == Some('?') {
            self.bump();
            true
        } else {
            false
        };
        Ok((min, max, lazy))
    }

    fn read_uint(&mut self) -> Result<usize> {
        let mut s = String::new();
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                s.push(c);
                self.bump();
            } else {
                break;
            }
        }
        s.parse::<usize>()
            .map_err(|_| MatroError::runtime("regexp: invalid quantifier number"))
    }
}

fn char_eq(a: char, b: char, icase: bool) -> bool {
    if icase {
        a.to_ascii_lowercase() == b.to_ascii_lowercase()
    } else {
        a == b
    }
}

fn atom_matches_char(atom: &Atom, c: char, icase: bool) -> bool {
    match atom {
        Atom::Literal(lit) => char_eq(*lit, c, icase),
        Atom::Any => c != '\n',
        Atom::Digit(neg) => c.is_ascii_digit() ^ *neg,
        Atom::Word(neg) => (c.is_ascii_alphanumeric() || c == '_') ^ *neg,
        Atom::Space(neg) => c.is_whitespace() ^ *neg,
        Atom::Class { negated, ranges } => {
            let hit = ranges.iter().any(|&(lo, hi)| {
                if icase {
                    let lc = c.to_ascii_lowercase();
                    lc >= lo.to_ascii_lowercase() && lc <= hi.to_ascii_lowercase()
                } else {
                    c >= lo && c <= hi
                }
            });
            hit ^ *negated
        }
        _ => false,
    }
}

fn match_alts(alts: &[Vec<Piece>], text: &[char], pos: usize, icase: bool) -> Option<usize> {
    for seq in alts {
        if let Some(end) = match_seq(seq, text, pos, icase) {
            return Some(end);
        }
    }
    None
}

fn match_seq(seq: &[Piece], text: &[char], pos: usize, icase: bool) -> Option<usize> {
    let Some((first, rest)) = seq.split_first() else {
        return Some(pos);
    };

    match &first.atom {
        Atom::AnchorStart => {
            if pos == 0 {
                return match_seq(rest, text, pos, icase);
            }
            return None;
        }
        Atom::AnchorEnd => {
            if pos == text.len() {
                return match_seq(rest, text, pos, icase);
            }
            return None;
        }
        Atom::Group(inner_alts) => {
            let mut positions = vec![pos];
            let max_reps = first.max.unwrap_or(text.len().saturating_sub(pos) + 1);
            let mut cur = pos;
            for _ in 0..max_reps {
                let Some(next) = match_alts(inner_alts, text, cur, icase) else {
                    break;
                };
                if next == cur && positions.len() > first.min {
                    break;
                }
                cur = next;
                positions.push(cur);
            }
            if positions.len() <= first.min {
                return None;
            }
            let candidates = &positions[first.min..];
            if first.lazy {
                for &p in candidates {
                    if let Some(end) = match_seq(rest, text, p, icase) {
                        return Some(end);
                    }
                }
            } else {
                for &p in candidates.iter().rev() {
                    if let Some(end) = match_seq(rest, text, p, icase) {
                        return Some(end);
                    }
                }
            }
            return None;
        }
        _ => {}
    }

    // Single-character atom repetition
    let max_reps = first.max.unwrap_or(text.len().saturating_sub(pos));
    let mut count = 0usize;
    while count < max_reps
        && pos + count < text.len()
        && atom_matches_char(&first.atom, text[pos + count], icase)
    {
        count += 1;
    }
    if count < first.min {
        return None;
    }
    if first.lazy {
        for k in first.min..=count {
            if let Some(end) = match_seq(rest, text, pos + k, icase) {
                return Some(end);
            }
        }
    } else {
        for k in (first.min..=count).rev() {
            if let Some(end) = match_seq(rest, text, pos + k, icase) {
                return Some(end);
            }
        }
    }
    None
}

fn find_all_matches(text_str: &str, pat: &str, icase: bool) -> Result<Vec<(usize, usize)>> {
    let alts = Parser::new(pat).parse_alts()?;
    let chars: Vec<char> = text_str.chars().collect();
    let mut matches = Vec::new();
    let mut pos = 0usize;
    while pos <= chars.len() {
        if let Some(end) = match_alts(&alts, &chars, pos, icase) {
            matches.push((pos, end));
            pos = if end > pos { end } else { pos + 1 };
        } else {
            pos += 1;
        }
    }
    Ok(matches)
}

/// `regexp(str, pat [, out1, out2, ...])` / `regexpi(...)`
/// Supports output keys: `'start'` (default), `'end'`, `'match'`, `'split'`.
pub fn fn_regexp(args: &[Value], icase: bool) -> Result<Vec<Value>> {
    if args.len() < 2 {
        return Err(MatroError::arity(
            if icase { "regexpi" } else { "regexp" },
            ">= 2",
            args.len(),
        ));
    }
    let text = args[0].as_str()?;
    let pat = args[1].as_str()?;
    let matches = find_all_matches(text, pat, icase)?;
    let chars: Vec<char> = text.chars().collect();

    let keys: Vec<&str> = if args.len() == 2 {
        vec!["start"]
    } else {
        args[2..]
            .iter()
            .map(|v| v.as_str())
            .collect::<Result<_>>()?
    };

    let mut outs = Vec::with_capacity(keys.len());
    for key in keys {
        let val = match key {
            "match" => {
                let items: Vec<Value> = matches
                    .iter()
                    .map(|&(s, e)| Value::String(chars[s..e].iter().collect()))
                    .collect();
                let cell = if items.is_empty() {
                    CellArray::empty()
                } else {
                    CellArray::from_rows(vec![items])?
                };
                Value::Cell(cell)
            }
            "split" => {
                let mut parts = Vec::new();
                let mut last = 0usize;
                for &(s, e) in &matches {
                    if s >= last {
                        parts.push(Value::String(chars[last..s].iter().collect()));
                        last = e;
                    }
                }
                parts.push(Value::String(chars[last..].iter().collect()));
                Value::Cell(CellArray::from_rows(vec![parts])?)
            }
            "end" => {
                let ends: Vec<f64> = matches.iter().map(|&(_, e)| e as f64).collect();
                Value::matrix(Matrix::row_vec(ends))
            }
            _ => {
                let starts: Vec<f64> = matches.iter().map(|&(s, _)| (s + 1) as f64).collect();
                Value::matrix(Matrix::row_vec(starts))
            }
        };
        outs.push(val);
    }
    Ok(outs)
}

/// `regexprep(str, pat, replacement [, 'ignorecase'])`
pub fn fn_regexprep(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 3 || args.len() > 4 {
        return Err(MatroError::arity("regexprep", "3 or 4", args.len()));
    }
    let text = args[0].as_str()?;
    let pat = args[1].as_str()?;
    let rep = args[2].as_str()?;
    let icase = args
        .get(3)
        .and_then(|v| v.as_str().ok())
        .is_some_and(|s| s.eq_ignore_ascii_case("ignorecase"));
    let matches = find_all_matches(text, pat, icase)?;
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut last = 0usize;
    for (s, e) in matches {
        if s >= last {
            for &ch in &chars[last..s] {
                out.push(ch);
            }
            out.push_str(rep);
            last = e;
        }
    }
    for &ch in &chars[last..] {
        out.push(ch);
    }
    Ok(vec![Value::String(out)])
}
