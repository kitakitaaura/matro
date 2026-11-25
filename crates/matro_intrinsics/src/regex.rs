//! Zero-dependency pure-Rust regular expression engine for MATLAB `regexp`, `regexpi`, `regexprep`.
//!
//! Supports: literals, `.`, `^`, `$`, `\b`, `\B`, `[a-z0-9\t\n\r\d\D\w\W\s\S]`, `[^...]`,
//! `\d`, `\D`, `\w`, `\W`, `\s`, `\S`, quantifiers `*`, `+`, `?`, `*?`, `+?`, `{m}`, `{m,n}`,
//! capturing `(...)`, non-capturing `(?:...)`, named `(?<name>...)`, alternation `a|b`,
//! `'start'`, `'end'`, `'match'`, `'split'`, `'tokens'`, `'names'`, `'once'`, `'ignorecase'`,
//! and `$0`/`$&`/`$1`..`$9`/`$<name>` backreferences in `regexprep`.

use std::collections::BTreeMap;

use matro_types::{CellArray, MatroError, Matrix, Result, StructArray, Value};

#[derive(Debug, Clone)]
enum ClassItem {
    Range(char, char),
    Digit(bool),
    Word(bool),
    Space(bool),
}

#[derive(Debug, Clone)]
enum Atom {
    Literal(char),
    Any,
    Digit(bool),
    Word(bool),
    Space(bool),
    WordBoundary(bool), // false = \b, true = \B
    Class { negated: bool, items: Vec<ClassItem> },
    Group { cap_idx: Option<usize>, alts: Vec<Vec<Piece>> },
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

struct CompiledRegex {
    alts: Vec<Vec<Piece>>,
    num_caps: usize,
    cap_names: Vec<(usize, String)>,
}

struct Parser<'a> {
    chars: Vec<char>,
    pos: usize,
    num_caps: usize,
    cap_names: Vec<(usize, String)>,
    _pat: &'a str,
}

impl<'a> Parser<'a> {
    fn new(pat: &'a str) -> Self {
        Self {
            chars: pat.chars().collect(),
            pos: 0,
            num_caps: 0,
            cap_names: Vec::new(),
            _pat: pat,
        }
    }

    fn compile(mut self) -> Result<CompiledRegex> {
        let alts = self.parse_alts()?;
        Ok(CompiledRegex {
            alts,
            num_caps: self.num_caps,
            cap_names: self.cap_names,
        })
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
                let cap_idx = if self.peek() == Some('?') {
                    if self.chars.get(self.pos + 1) == Some(&':') {
                        self.bump();
                        self.bump();
                        None
                    } else if self.chars.get(self.pos + 1) == Some(&'<') {
                        self.bump();
                        self.bump();
                        let mut name = String::new();
                        while let Some(c) = self.bump() {
                            if c == '>' {
                                break;
                            }
                            name.push(c);
                        }
                        let idx = self.num_caps;
                        self.num_caps += 1;
                        self.cap_names.push((idx, name));
                        Some(idx)
                    } else {
                        let idx = self.num_caps;
                        self.num_caps += 1;
                        Some(idx)
                    }
                } else {
                    let idx = self.num_caps;
                    self.num_caps += 1;
                    Some(idx)
                };
                let inner = self.parse_alts()?;
                if self.bump() != Some(')') {
                    return Err(MatroError::runtime("regexp: unclosed '('"));
                }
                Ok(Atom::Group {
                    cap_idx,
                    alts: inner,
                })
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
                    'b' => Atom::WordBoundary(false),
                    'B' => Atom::WordBoundary(true),
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
        let mut items = Vec::new();
        while let Some(c) = self.bump() {
            if c == ']' && !items.is_empty() {
                return Ok(Atom::Class { negated, items });
            }
            let start = if c == '\\' {
                match self.bump() {
                    Some('d') => {
                        items.push(ClassItem::Digit(false));
                        continue;
                    }
                    Some('D') => {
                        items.push(ClassItem::Digit(true));
                        continue;
                    }
                    Some('w') => {
                        items.push(ClassItem::Word(false));
                        continue;
                    }
                    Some('W') => {
                        items.push(ClassItem::Word(true));
                        continue;
                    }
                    Some('s') => {
                        items.push(ClassItem::Space(false));
                        continue;
                    }
                    Some('S') => {
                        items.push(ClassItem::Space(true));
                        continue;
                    }
                    Some('n') => '\n',
                    Some('t') => '\t',
                    Some('r') => '\r',
                    Some(x) => x,
                    None => return Err(MatroError::runtime("regexp: unclosed '['")),
                }
            } else {
                c
            };
            if self.peek() == Some('-') && self.chars.get(self.pos + 1) != Some(&']') {
                self.bump(); // `-`
                let end = match self.bump() {
                    Some('\\') => match self.bump() {
                        Some('n') => '\n',
                        Some('t') => '\t',
                        Some('r') => '\r',
                        Some(x) => x,
                        None => return Err(MatroError::runtime("regexp: unclosed '['")),
                    },
                    Some(x) => x,
                    None => return Err(MatroError::runtime("regexp: unclosed '['")),
                };
                items.push(ClassItem::Range(start, end));
            } else {
                items.push(ClassItem::Range(start, start));
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

fn is_word_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
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
        Atom::Word(neg) => is_word_char(c) ^ *neg,
        Atom::Space(neg) => c.is_whitespace() ^ *neg,
        Atom::Class { negated, items } => {
            let hit = items.iter().any(|it| match *it {
                ClassItem::Range(lo, hi) => {
                    if icase {
                        let lc = c.to_ascii_lowercase();
                        lc >= lo.to_ascii_lowercase() && lc <= hi.to_ascii_lowercase()
                    } else {
                        c >= lo && c <= hi
                    }
                }
                ClassItem::Digit(neg) => c.is_ascii_digit() ^ neg,
                ClassItem::Word(neg) => is_word_char(c) ^ neg,
                ClassItem::Space(neg) => c.is_whitespace() ^ neg,
            });
            hit ^ *negated
        }
        _ => false,
    }
}

type Captures = Vec<Option<(usize, usize)>>;

fn match_alts(
    alts: &[Vec<Piece>],
    text: &[char],
    pos: usize,
    icase: bool,
    caps: &mut Captures,
) -> Option<usize> {
    for seq in alts {
        let saved = caps.clone();
        if let Some(end) = match_seq(seq, text, pos, icase, caps) {
            return Some(end);
        }
        *caps = saved;
    }
    None
}

fn match_seq(
    seq: &[Piece],
    text: &[char],
    pos: usize,
    icase: bool,
    caps: &mut Captures,
) -> Option<usize> {
    let Some((first, rest)) = seq.split_first() else {
        return Some(pos);
    };

    match &first.atom {
        Atom::AnchorStart => {
            if pos == 0 {
                return match_seq(rest, text, pos, icase, caps);
            }
            return None;
        }
        Atom::AnchorEnd => {
            if pos == text.len() {
                return match_seq(rest, text, pos, icase, caps);
            }
            return None;
        }
        Atom::WordBoundary(neg) => {
            let left_w = pos > 0 && is_word_char(text[pos - 1]);
            let right_w = pos < text.len() && is_word_char(text[pos]);
            let is_b = left_w != right_w;
            if is_b ^ *neg {
                return match_seq(rest, text, pos, icase, caps);
            }
            return None;
        }
        Atom::Group { cap_idx, alts: inner_alts } => {
            let mut states: Vec<(usize, Captures)> = vec![(pos, caps.clone())];
            let max_reps = first.max.unwrap_or(text.len().saturating_sub(pos) + 1);
            let mut cur_pos = pos;
            let mut cur_caps = caps.clone();
            for _ in 0..max_reps {
                let mut next_caps = cur_caps.clone();
                let Some(next) = match_alts(inner_alts, text, cur_pos, icase, &mut next_caps) else {
                    break;
                };
                if let Some(ci) = *cap_idx {
                    next_caps[ci] = Some((cur_pos, next));
                }
                if next == cur_pos && states.len() > first.min {
                    break;
                }
                cur_pos = next;
                cur_caps = next_caps;
                states.push((cur_pos, cur_caps.clone()));
            }
            if states.len() <= first.min {
                return None;
            }
            let candidates = &states[first.min..];
            if first.lazy {
                for (p, c_state) in candidates {
                    let mut try_caps = c_state.clone();
                    if let Some(end) = match_seq(rest, text, *p, icase, &mut try_caps) {
                        *caps = try_caps;
                        return Some(end);
                    }
                }
            } else {
                for (p, c_state) in candidates.iter().rev() {
                    let mut try_caps = c_state.clone();
                    if let Some(end) = match_seq(rest, text, *p, icase, &mut try_caps) {
                        *caps = try_caps;
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
            let saved = caps.clone();
            if let Some(end) = match_seq(rest, text, pos + k, icase, caps) {
                return Some(end);
            }
            *caps = saved;
        }
    } else {
        for k in (first.min..=count).rev() {
            let saved = caps.clone();
            if let Some(end) = match_seq(rest, text, pos + k, icase, caps) {
                return Some(end);
            }
            *caps = saved;
        }
    }
    None
}

struct MatchEntry {
    start: usize,
    end: usize,
    caps: Captures,
}

fn find_matches(
    text_str: &str,
    compiled: &CompiledRegex,
    icase: bool,
    once: bool,
) -> Vec<MatchEntry> {
    let chars: Vec<char> = text_str.chars().collect();
    let mut matches = Vec::new();
    let mut pos = 0usize;
    while pos <= chars.len() {
        let mut caps = vec![None; compiled.num_caps];
        if let Some(end) = match_alts(&compiled.alts, &chars, pos, icase, &mut caps) {
            matches.push(MatchEntry { start: pos, end, caps });
            if once {
                break;
            }
            pos = if end > pos { end } else { pos + 1 };
        } else {
            pos += 1;
        }
    }
    matches
}

fn eval_regexp_single(
    text: &str,
    compiled: &CompiledRegex,
    keys: &[String],
    icase: bool,
    once: bool,
) -> Result<Vec<Value>> {
    let matches = find_matches(text, compiled, icase, once);
    let chars: Vec<char> = text.chars().collect();
    let mut outs = Vec::with_capacity(keys.len());
    for key in keys {
        let val = match key.as_str() {
            "match" => {
                if once {
                    if let Some(m) = matches.first() {
                        Value::String(chars[m.start..m.end].iter().collect())
                    } else {
                        Value::String(String::new())
                    }
                } else {
                    let items: Vec<Value> = matches
                        .iter()
                        .map(|m| Value::String(chars[m.start..m.end].iter().collect()))
                        .collect();
                    let cell = if items.is_empty() {
                        CellArray::empty()
                    } else {
                        CellArray::from_rows(vec![items])?
                    };
                    Value::Cell(cell)
                }
            }
            "tokens" => {
                if once {
                    if let Some(m) = matches.first() {
                        let toks: Vec<Value> = m
                            .caps
                            .iter()
                            .map(|opt| match opt {
                                Some((s, e)) => Value::String(chars[*s..*e].iter().collect()),
                                None => Value::String(String::new()),
                            })
                            .collect();
                        let cell = if toks.is_empty() {
                            CellArray::empty()
                        } else {
                            CellArray::from_rows(vec![toks])?
                        };
                        Value::Cell(cell)
                    } else {
                        Value::Cell(CellArray::empty())
                    }
                } else {
                    let mut all = Vec::with_capacity(matches.len());
                    for m in &matches {
                        let toks: Vec<Value> = m
                            .caps
                            .iter()
                            .map(|opt| match opt {
                                Some((s, e)) => Value::String(chars[*s..*e].iter().collect()),
                                None => Value::String(String::new()),
                            })
                            .collect();
                        let inner = if toks.is_empty() {
                            CellArray::empty()
                        } else {
                            CellArray::from_rows(vec![toks])?
                        };
                        all.push(Value::Cell(inner));
                    }
                    let cell = if all.is_empty() {
                        CellArray::empty()
                    } else {
                        CellArray::from_rows(vec![all])?
                    };
                    Value::Cell(cell)
                }
            }
            "names" => {
                let mut items = Vec::with_capacity(matches.len());
                for m in &matches {
                    let mut st = BTreeMap::new();
                    for (ci, name) in &compiled.cap_names {
                        let s = match m.caps.get(*ci).copied().flatten() {
                            Some((s, e)) => chars[s..e].iter().collect(),
                            None => String::new(),
                        };
                        st.insert(name.clone(), Value::String(s));
                    }
                    items.push(st);
                }
                if items.len() == 1 {
                    Value::Struct(items.remove(0))
                } else {
                    let len = items.len();
                    Value::StructArray(StructArray {
                        rows: if len == 0 { 0 } else { 1 },
                        cols: len,
                        items,
                    })
                }
            }
            "split" => {
                let mut parts = Vec::new();
                let mut last = 0usize;
                for m in &matches {
                    if m.start >= last {
                        parts.push(Value::String(chars[last..m.start].iter().collect()));
                        last = m.end;
                    }
                }
                parts.push(Value::String(chars[last..].iter().collect()));
                Value::Cell(CellArray::from_rows(vec![parts])?)
            }
            "end" => {
                let ends: Vec<f64> = matches.iter().map(|m| m.end as f64).collect();
                Value::matrix(Matrix::row_vec(ends))
            }
            _ => {
                let starts: Vec<f64> = matches.iter().map(|m| (m.start + 1) as f64).collect();
                Value::matrix(Matrix::row_vec(starts))
            }
        };
        outs.push(val);
    }
    Ok(outs)
}

/// `regexp(str, pat [, out1, ... [, 'once'|'ignorecase']])` / `regexpi(...)`
pub fn fn_regexp(args: &[Value], nargout: usize, default_icase: bool) -> Result<Vec<Value>> {
    if args.len() < 2 {
        return Err(MatroError::arity(
            if default_icase { "regexpi" } else { "regexp" },
            ">= 2",
            args.len(),
        ));
    }
    let pat = args[1].as_str()?;
    let compiled = Parser::new(pat).compile()?;

    let mut icase = default_icase;
    let mut once = false;
    let mut keys: Vec<String> = Vec::new();
    for a in &args[2..] {
        let s = a.as_str()?.to_ascii_lowercase();
        match s.as_str() {
            "once" => once = true,
            "ignorecase" => icase = true,
            "matchcase" => icase = false,
            "start" | "end" | "match" | "split" | "tokens" | "names" => keys.push(s),
            _ => {}
        }
    }
    if keys.is_empty() {
        keys.push("start".to_string());
        if nargout >= 2 {
            keys.push("end".to_string());
        }
        if nargout >= 3 {
            keys.push("match".to_string());
        }
    }

    // Handle cell array or string array input
    if let Value::Cell(c) = &args[0] {
        let mut per_out: Vec<Vec<Value>> = vec![Vec::with_capacity(c.data.len()); keys.len()];
        for item in &c.data {
            let s = item.as_str()?;
            let res = eval_regexp_single(s, &compiled, &keys, icase, once)?;
            for (k, v) in res.into_iter().enumerate() {
                per_out[k].push(v);
            }
        }
        let mut final_outs = Vec::with_capacity(keys.len());
        for items in per_out {
            final_outs.push(Value::Cell(CellArray {
                rows: c.rows,
                cols: c.cols,
                data: items,
            }));
        }
        return Ok(final_outs);
    }

    let text = args[0].as_str()?;
    eval_regexp_single(text, &compiled, &keys, icase, once)
}

fn expand_replacement(
    rep: &str,
    chars: &[char],
    m: &MatchEntry,
    cap_names: &[(usize, String)],
) -> String {
    let rchars: Vec<char> = rep.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < rchars.len() {
        if rchars[i] == '$' && i + 1 < rchars.len() {
            let next = rchars[i + 1];
            if next == '&' || next == '0' {
                for &c in &chars[m.start..m.end] {
                    out.push(c);
                }
                i += 2;
                continue;
            } else if next.is_ascii_digit() {
                let idx = (next as u8 - b'1') as usize;
                if let Some(Some((s, e))) = m.caps.get(idx) {
                    for &c in &chars[*s..*e] {
                        out.push(c);
                    }
                }
                i += 2;
                continue;
            } else if next == '<' {
                if let Some(close) = rchars[i + 2..].iter().position(|&c| c == '>') {
                    let name: String = rchars[i + 2..i + 2 + close].iter().collect();
                    if let Some((ci, _)) = cap_names.iter().find(|(_, n)| n == &name) {
                        if let Some(Some((s, e))) = m.caps.get(*ci) {
                            for &c in &chars[*s..*e] {
                                out.push(c);
                            }
                        }
                    }
                    i += 3 + close;
                    continue;
                }
            }
        }
        out.push(rchars[i]);
        i += 1;
    }
    out
}

fn regexprep_single(
    text: &str,
    compiled: &CompiledRegex,
    rep: &str,
    icase: bool,
    once: bool,
) -> String {
    let matches = find_matches(text, compiled, icase, once);
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut last = 0usize;
    for m in &matches {
        if m.start >= last {
            for &ch in &chars[last..m.start] {
                out.push(ch);
            }
            out.push_str(&expand_replacement(rep, &chars, m, &compiled.cap_names));
            last = m.end;
        }
    }
    for &ch in &chars[last..] {
        out.push(ch);
    }
    out
}

/// `regexprep(str, pat, replacement [, 'ignorecase'|'once'])`
pub fn fn_regexprep(args: &[Value]) -> Result<Vec<Value>> {
    if args.len() < 3 {
        return Err(MatroError::arity("regexprep", ">= 3 (str, pat, rep [, opts...])", args.len()));
    }
    let pat = args[1].as_str()?;
    let rep = args[2].as_str()?;
    let compiled = Parser::new(pat).compile()?;
    let mut icase = false;
    let mut once = false;
    for a in &args[3..] {
        if let Some(s) = crate::value_as_text(a) {
            match s.to_ascii_lowercase().as_str() {
                "ignorecase" => icase = true,
                "once" => once = true,
                _ => {}
            }
        }
    }
    if let Value::Cell(c) = &args[0] {
        let mut data = Vec::with_capacity(c.data.len());
        for item in &c.data {
            let s = item.as_str()?;
            data.push(Value::String(regexprep_single(s, &compiled, rep, icase, once)));
        }
        return Ok(vec![Value::Cell(CellArray {
            rows: c.rows,
            cols: c.cols,
            data,
        })]);
    }
    let text = args[0].as_str()?;
    Ok(vec![Value::String(regexprep_single(text, &compiled, rep, icase, once))])
}
