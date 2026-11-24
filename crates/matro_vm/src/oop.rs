use std::collections::BTreeMap;

use matro_bytecode::{Chunk, CompiledArgSpec, FunctionProto};
use matro_types::{
    CellArray, Complex, MatroError, Matrix, NumericClass, ObjectInstance, Result, Value,
};

use crate::{LocalFrame, StackItem, Vm, index_matrix, resolve_indices};

/// Evaluate a standalone expression chunk (`__prop_default`, `__enum_arg`, `__arg_default`).
pub(crate) fn eval_standalone_chunk(
    vm: &mut Vm,
    ch: &Chunk,
    var_name: &str,
    stdout: &mut String,
) -> Result<Value> {
    let mut tmp_val = None;
    let prev = vm.globals.remove(var_name);
    let res = vm.run_frame(ch, None, stdout, &mut tmp_val);
    let v = vm.globals.remove(var_name).unwrap_or_else(Value::empty);
    if let Some(p) = prev {
        vm.globals.insert(var_name.to_string(), p);
    }
    res?;
    Ok(v)
}

fn is_obj_of<'a>(v: &'a Value, cls: &str) -> Option<&'a ObjectInstance> {
    match v {
        Value::Object(o) if o.class_name == cls => Some(o),
        _ => None,
    }
}

fn make_obj(class_name: &str, mut fields: BTreeMap<String, Value>) -> Value {
    let _ = &mut fields;
    Value::Object(ObjectInstance {
        class_name: class_name.to_string(),
        fields,
    })
}

fn secs_of(o: &ObjectInstance) -> Result<Matrix> {
    let key = if o.class_name == "datetime" {
        "EpochSec"
    } else {
        "Seconds"
    };
    o.fields
        .get(key)
        .ok_or_else(|| MatroError::runtime(format!("Malformed {} object", o.class_name)))?
        .to_dense_matrix()
}

fn string_elems(v: &Value) -> Option<(usize, usize, Vec<String>)> {
    match v {
        Value::String(s) => Some((1, 1, vec![s.clone()])),
        Value::Object(o) if o.class_name == "string" => {
            if let Some(Value::Cell(c)) = o.fields.get("Data") {
                let elems: Vec<String> = c
                    .data
                    .iter()
                    .map(|x| x.as_str().unwrap_or("").to_string())
                    .collect();
                Some((c.rows, c.cols, elems))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn make_string_obj(rows: usize, cols: usize, elems: Vec<String>) -> Value {
    let cell = CellArray {
        rows,
        cols,
        data: elems.into_iter().map(Value::String).collect(),
    };
    let mut fields = BTreeMap::new();
    fields.insert("Data".to_string(), Value::Cell(cell));
    make_obj("string", fields)
}

fn cat_labels(v: &Value) -> Option<(usize, usize, Vec<String>)> {
    match v {
        Value::Object(o) if o.class_name == "categorical" => {
            let codes = o.fields.get("Codes")?.to_dense_matrix().ok()?;
            let cats = match o.fields.get("Categories")? {
                Value::Cell(c) => c
                    .data
                    .iter()
                    .map(|x| x.as_str().unwrap_or("").to_string())
                    .collect::<Vec<_>>(),
                _ => return None,
            };
            let mut out = Vec::with_capacity(codes.len());
            for &code in codes.data() {
                let idx = code.round() as isize;
                if idx >= 1 && (idx as usize) <= cats.len() {
                    out.push(cats[(idx - 1) as usize].clone());
                } else {
                    out.push(String::new());
                }
            }
            Some((codes.rows(), codes.cols(), out))
        }
        Value::String(s) => Some((1, 1, vec![s.clone()])),
        Value::Object(o) if o.class_name == "string" => string_elems(v),
        _ => None,
    }
}

fn cmp_f64(op: &str, x: f64, y: f64) -> bool {
    match op {
        "eq" => (x - y).abs() <= 1e-12 * x.abs().max(y.abs()).max(1.0),
        "ne" => (x - y).abs() > 1e-12 * x.abs().max(y.abs()).max(1.0),
        "lt" => x < y,
        "le" => x <= y,
        "gt" => x > y,
        "ge" => x >= y,
        _ => false,
    }
}

fn cmp_str(op: &str, a: &str, b: &str) -> bool {
    match op {
        "eq" => a == b,
        "ne" => a != b,
        "lt" => a < b,
        "le" => a <= b,
        "gt" => a > b,
        "ge" => a >= b,
        _ => false,
    }
}

fn broadcast_str_op<T>(
    a: (usize, usize, Vec<String>),
    b: (usize, usize, Vec<String>),
    f: impl Fn(&str, &str) -> T,
) -> Result<(usize, usize, Vec<T>)> {
    let (ar, ac, av) = a;
    let (br, bc, bv) = b;
    let (or, oc) = if av.len() == 1 {
        (br, bc)
    } else if bv.len() == 1 {
        (ar, ac)
    } else if ar == br && ac == bc {
        (ar, ac)
    } else {
        return Err(MatroError::dimension(format!(
            "Incompatible string array dimensions: {ar}x{ac} vs {br}x{bc}"
        )));
    };
    let n = or * oc;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let sa = if av.len() == 1 { &av[0] } else { &av[i] };
        let sb = if bv.len() == 1 { &bv[0] } else { &bv[i] };
        out.push(f(sa, sb));
    }
    Ok((or, oc, out))
}

fn try_modern_binop(op: &str, a: &Value, b: &Value) -> Result<Option<Value>> {
    // 1. datetime & duration arithmetic and comparisons
    if let (Some(da), Some(db)) = (is_obj_of(a, "datetime"), is_obj_of(b, "datetime")) {
        let sa = secs_of(da)?;
        let sb = secs_of(db)?;
        match op {
            "minus" => {
                let diff = sa.elementwise(&sb, |x, y| x - y)?;
                return Ok(Some(matro_intrinsics::modern_types::make_duration_from_secs(
                    diff.data().to_vec(),
                )));
            }
            "eq" | "ne" | "lt" | "le" | "gt" | "ge" => {
                let res = sa
                    .elementwise(&sb, |x, y| if cmp_f64(op, x, y) { 1.0 } else { 0.0 })?
                    .with_logical(true);
                return Ok(Some(Value::matrix(res)));
            }
            _ => {}
        }
    }
    if let (Some(da), Some(du)) = (is_obj_of(a, "datetime"), is_obj_of(b, "duration")) {
        if op == "plus" || op == "minus" {
            let sa = secs_of(da)?;
            let su = secs_of(du)?;
            let res = if op == "plus" {
                sa.elementwise(&su, |x, y| x + y)?
            } else {
                sa.elementwise(&su, |x, y| x - y)?
            };
            return Ok(Some(
                matro_intrinsics::modern_types::make_datetime_from_epochs(res.data().to_vec()),
            ));
        }
    }
    if let (Some(du), Some(da)) = (is_obj_of(a, "duration"), is_obj_of(b, "datetime")) {
        if op == "plus" {
            let su = secs_of(du)?;
            let sa = secs_of(da)?;
            let res = sa.elementwise(&su, |x, y| x + y)?;
            return Ok(Some(
                matro_intrinsics::modern_types::make_datetime_from_epochs(res.data().to_vec()),
            ));
        }
    }
    if let (Some(ua), Some(ub)) = (is_obj_of(a, "duration"), is_obj_of(b, "duration")) {
        let sa = secs_of(ua)?;
        let sb = secs_of(ub)?;
        match op {
            "plus" | "minus" => {
                let res = if op == "plus" {
                    sa.elementwise(&sb, |x, y| x + y)?
                } else {
                    sa.elementwise(&sb, |x, y| x - y)?
                };
                return Ok(Some(matro_intrinsics::modern_types::make_duration_from_secs(
                    res.data().to_vec(),
                )));
            }
            "rdivide" | "mrdivide" => {
                let res = sa.elementwise(&sb, |x, y| x / y)?;
                return Ok(Some(Value::matrix(res)));
            }
            "eq" | "ne" | "lt" | "le" | "gt" | "ge" => {
                let res = sa
                    .elementwise(&sb, |x, y| if cmp_f64(op, x, y) { 1.0 } else { 0.0 })?
                    .with_logical(true);
                return Ok(Some(Value::matrix(res)));
            }
            _ => {}
        }
    }
    if let (Some(du), Value::Matrix(mb)) = (is_obj_of(a, "duration"), b) {
        if op == "mtimes" || op == "times" || op == "mrdivide" || op == "rdivide" {
            let su = secs_of(du)?;
            let res = if op == "mtimes" || op == "times" {
                su.elementwise(mb, |x, y| x * y)?
            } else {
                su.elementwise(mb, |x, y| x / y)?
            };
            return Ok(Some(matro_intrinsics::modern_types::make_duration_from_secs(
                res.data().to_vec(),
            )));
        }
    }
    if let (Value::Matrix(ma), Some(du)) = (a, is_obj_of(b, "duration")) {
        if op == "mtimes" || op == "times" {
            let su = secs_of(du)?;
            let res = ma.elementwise(&su, |x, y| x * y)?;
            return Ok(Some(matro_intrinsics::modern_types::make_duration_from_secs(
                res.data().to_vec(),
            )));
        }
    }

    // 2. string array concatenation (`+`) and comparisons
    if is_obj_of(a, "string").is_some() || is_obj_of(b, "string").is_some() {
        if let (Some(sa), Some(sb)) = (string_elems(a), string_elems(b)) {
            if op == "plus" {
                let (r, c, elems) = broadcast_str_op(sa, sb, |x, y| format!("{x}{y}"))?;
                return Ok(Some(make_string_obj(r, c, elems)));
            }
            if matches!(op, "eq" | "ne" | "lt" | "le" | "gt" | "ge") {
                let (r, c, flags) = broadcast_str_op(sa, sb, |x, y| {
                    if cmp_str(op, x, y) { 1.0 } else { 0.0 }
                })?;
                return Ok(Some(Value::matrix(
                    Matrix::from_col_major(r, c, flags)?.with_logical(true),
                )));
            }
        }
    }

    // 3. categorical comparisons (`==`, `~=`)
    if is_obj_of(a, "categorical").is_some() || is_obj_of(b, "categorical").is_some() {
        if matches!(op, "eq" | "ne") {
            if let (Some(ca), Some(cb)) = (cat_labels(a), cat_labels(b)) {
                let (r, c, flags) = broadcast_str_op(ca, cb, |x, y| {
                    if cmp_str(op, x, y) { 1.0 } else { 0.0 }
                })?;
                return Ok(Some(Value::matrix(
                    Matrix::from_col_major(r, c, flags)?.with_logical(true),
                )));
            }
        }
    }

    Ok(None)
}

/// Dispatch native modern-type binary operators OR `classdef` operator overloading (`plus`, `mtimes`, etc.).
pub(crate) fn try_object_binop(
    vm: &mut Vm,
    op: &str,
    a: &Value,
    b: &Value,
    stdout: &mut String,
) -> Result<Option<Value>> {
    if let Some(res) = try_modern_binop(op, a, b)? {
        return Ok(Some(res));
    }
    let cname = match a {
        Value::Object(o) => Some(o.class_name.clone()),
        Value::Handle(h) => Some(h.class_name.clone()),
        _ => match b {
            Value::Object(o) => Some(o.class_name.clone()),
            Value::Handle(h) => Some(h.class_name.clone()),
            _ => None,
        },
    };
    if let Some(cls) = cname {
        if let Some(proto) = vm.find_class_method(&cls, op) {
            let rets = vm.call_user_func(&proto, &[a.clone(), b.clone()], 1, None, stdout)?;
            return Ok(Some(rets.into_iter().next().unwrap_or_else(Value::empty)));
        }
        // Fallback: if `a` is an enum / object without `eq`/`ne` overload, compare class + fields!
        if op == "eq" || op == "ne" {
            if let (Value::Object(oa), Value::Object(ob)) = (a, b) {
                let eq = oa.class_name == ob.class_name && oa.fields == ob.fields;
                return Ok(Some(Value::bool(if op == "eq" { eq } else { !eq })));
            }
            if let (Value::Handle(ha), Value::Handle(hb)) = (a, b) {
                let eq = ha.id == hb.id;
                return Ok(Some(Value::bool(if op == "eq" { eq } else { !eq })));
            }
        }
    }
    Ok(None)
}

/// Dispatch unary operators (`uminus`, `uplus`, `ctranspose`, `transpose`) on modern types or `classdef` objects.
pub(crate) fn try_object_unop(
    vm: &mut Vm,
    op: &str,
    v: &Value,
    stdout: &mut String,
) -> Result<Option<Value>> {
    if op == "uminus" {
        if let Some(du) = is_obj_of(v, "duration") {
            let su = secs_of(du)?;
            return Ok(Some(matro_intrinsics::modern_types::make_duration_from_secs(
                su.data().iter().map(|&x| -x).collect(),
            )));
        }
    }
    let cname = match v {
        Value::Object(o) => Some(o.class_name.clone()),
        Value::Handle(h) => Some(h.class_name.clone()),
        _ => None,
    };
    if let Some(cls) = cname {
        if let Some(proto) = vm.find_class_method(&cls, op) {
            let rets = vm.call_user_func(&proto, &[v.clone()], 1, None, stdout)?;
            return Ok(Some(rets.into_iter().next().unwrap_or_else(Value::empty)));
        }
    }
    Ok(None)
}

/// Parenthesis indexing `obj(...)` for built-in modern object types (`string`, `datetime`, `duration`, `categorical`).
pub(crate) fn index_modern_object(
    o: &ObjectInstance,
    raw_args: &[StackItem],
) -> Result<Option<Value>> {
    match o.class_name.as_str() {
        "string" => {
            let Some(Value::Cell(c)) = o.fields.get("Data") else {
                return Ok(None);
            };
            let sub = crate::index_cell_subarray(c, raw_args)?;
            let mut fields = BTreeMap::new();
            fields.insert("Data".to_string(), Value::Cell(sub));
            Ok(Some(make_obj("string", fields)))
        }
        "datetime" => {
            let secs = secs_of(o)?;
            let sub = index_matrix(&secs, raw_args)?;
            Ok(Some(
                matro_intrinsics::modern_types::make_datetime_from_epochs(sub.data().to_vec()),
            ))
        }
        "duration" => {
            let secs = secs_of(o)?;
            let sub = index_matrix(&secs, raw_args)?;
            Ok(Some(matro_intrinsics::modern_types::make_duration_from_secs(
                sub.data().to_vec(),
            )))
        }
        "categorical" => {
            let codes = o
                .fields
                .get("Codes")
                .ok_or_else(|| MatroError::runtime("Malformed categorical object"))?
                .to_dense_matrix()?;
            let sub = index_matrix(&codes, raw_args)?;
            let mut fields = o.fields.clone();
            fields.insert("Codes".to_string(), Value::matrix(sub));
            Ok(Some(make_obj("categorical", fields)))
        }
        "griddedInterpolant" => {
            let call_args = crate::flatten_call_args(raw_args.to_vec())?;
            let val = matro_intrinsics::interp::eval_gridded_interpolant(o, &call_args)?;
            Ok(Some(val))
        }
        _ => Ok(None),
    }
}

/// Brace indexing `s{...}` on `string` arrays (returns char `Value::String`).
pub(crate) fn index_string_braces(o: &ObjectInstance, raw_args: &[StackItem]) -> Result<Vec<Value>> {
    let Some(Value::Cell(c)) = o.fields.get("Data") else {
        return Err(MatroError::runtime("Malformed string array"));
    };
    if raw_args.len() == 1 {
        let idxs = resolve_indices(&raw_args[0], c.data.len())?;
        Ok(idxs.iter().map(|&i| c.data[i].clone()).collect())
    } else if raw_args.len() == 2 {
        let rows = resolve_indices(&raw_args[0], c.rows)?;
        let cols = resolve_indices(&raw_args[1], c.cols)?;
        let mut out = Vec::with_capacity(rows.len() * cols.len());
        for &col in &cols {
            for &row in &rows {
                out.push(c.get(row, col).clone());
            }
        }
        Ok(out)
    } else {
        Err(MatroError::index("string brace indexing expects 1 or 2 indices"))
    }
}

/// Bind and validate MATLAB `arguments ... end` blocks on function entry.
pub(crate) fn bind_arguments_block(
    vm: &mut Vm,
    proto: &FunctionProto,
    args: &[Value],
    frame: &mut LocalFrame<'_>,
    stdout: &mut String,
) -> Result<()> {
    let mut pos_specs: Vec<&CompiledArgSpec> = Vec::new();
    let mut rep_specs: Vec<&CompiledArgSpec> = Vec::new();
    let mut nv_specs: Vec<&CompiledArgSpec> = Vec::new();

    for sp in &proto.arg_specs {
        if sp.repeating {
            rep_specs.push(sp);
        } else if sp.field.is_some() {
            nv_specs.push(sp);
        } else {
            pos_specs.push(sp);
        }
    }

    // Collect known name-value field names so we know where positional/repeating args stop
    let nv_field_names: Vec<String> = nv_specs
        .iter()
        .filter_map(|s| s.field.clone())
        .collect();

    let mut arg_idx = 0usize;

    // 1. Positional specs
    for sp in &pos_specs {
        let starts_nv = !nv_field_names.is_empty()
            && sp.default.is_some()
            && arg_idx + 1 < args.len()
            && args[arg_idx]
                .as_str()
                .ok()
                .is_some_and(|k| nv_field_names.iter().any(|f| f.eq_ignore_ascii_case(k)));

        let raw_val = if arg_idx < args.len() && !starts_nv {
            let v = args[arg_idx].clone();
            arg_idx += 1;
            v
        } else if let Some(def_ch) = &sp.default {
            eval_standalone_chunk(vm, def_ch, "__arg_default", stdout)?
        } else {
            return Err(MatroError::runtime(format!(
                "{}: missing required argument '{}'",
                proto.name, sp.target
            )));
        };

        let val = validate_and_coerce_arg(vm, &proto.name, &sp.target, raw_val, sp, stdout)?;
        if let Some(slot) = sp.target_slot {
            frame.set(slot, val);
        }
    }

    // 2. Repeating specs
    if !rep_specs.is_empty() {
        let stride = rep_specs.len();
        let mut buckets: Vec<Vec<Value>> = vec![Vec::new(); stride];
        let mut rem_end = args.len();
        if !nv_field_names.is_empty() {
            // Find earliest start of trailing name-value pairs
            let mut scan = arg_idx;
            while scan + 1 < args.len() {
                if args[scan]
                    .as_str()
                    .ok()
                    .is_some_and(|k| nv_field_names.iter().any(|f| f.eq_ignore_ascii_case(k)))
                {
                    rem_end = scan;
                    break;
                }
                scan += stride;
            }
        }
        let rem_count = rem_end.saturating_sub(arg_idx);
        if rem_count % stride != 0 {
            return Err(MatroError::runtime(format!(
                "{}: repeating arguments must be passed in groups of {stride}",
                proto.name
            )));
        }
        while arg_idx < rem_end {
            for (k, sp) in rep_specs.iter().enumerate() {
                let v = validate_and_coerce_arg(
                    vm,
                    &proto.name,
                    &sp.target,
                    args[arg_idx].clone(),
                    sp,
                    stdout,
                )?;
                buckets[k].push(v);
                arg_idx += 1;
            }
        }
        for (sp, items) in rep_specs.iter().zip(buckets.into_iter()) {
            let cell = if items.is_empty() {
                CellArray::empty()
            } else {
                CellArray::from_rows(vec![items])?
            };
            if let Some(slot) = sp.target_slot {
                frame.set(slot, Value::Cell(cell));
            }
        }
    }

    // 3. Name-Value specs (`opts.Field = default`)
    if !nv_specs.is_empty() {
        let mut structs: BTreeMap<String, (Option<u16>, BTreeMap<String, Value>)> = BTreeMap::new();
        for sp in &nv_specs {
            let entry = structs
                .entry(sp.target.clone())
                .or_insert_with(|| (sp.target_slot, BTreeMap::new()));
            let fname = sp.field.as_ref().unwrap().clone();
            let def_val = if let Some(def_ch) = &sp.default {
                eval_standalone_chunk(vm, def_ch, "__arg_default", stdout)?
            } else {
                Value::empty()
            };
            entry.1.insert(fname, def_val);
        }

        if (args.len() - arg_idx) % 2 != 0 {
            return Err(MatroError::runtime(format!(
                "{}: name-value arguments must come in pairs",
                proto.name
            )));
        }
        while arg_idx + 1 < args.len() {
            let key = args[arg_idx].as_str().map_err(|_| {
                MatroError::runtime(format!(
                    "{}: name-value argument name must be a string",
                    proto.name
                ))
            })?;
            let raw_val = args[arg_idx + 1].clone();
            arg_idx += 2;

            let Some(sp) = nv_specs
                .iter()
                .find(|s| s.field.as_deref().is_some_and(|f| f.eq_ignore_ascii_case(key)))
            else {
                return Err(MatroError::runtime(format!(
                    "{}: unrecognized name-value argument '{key}'",
                    proto.name
                )));
            };
            let fname = sp.field.as_ref().unwrap();
            let label = format!("{}.{fname}", sp.target);
            let val = validate_and_coerce_arg(vm, &proto.name, &label, raw_val, sp, stdout)?;
            if let Some((_, map)) = structs.get_mut(&sp.target) {
                map.insert(fname.clone(), val);
            }
        }

        for (_tname, (slot, map)) in structs {
            if let Some(s) = slot {
                frame.set(s, Value::Struct(map));
            }
        }
    } else if arg_idx < args.len() && proto.varargin_slot.is_none() {
        return Err(MatroError::arity(
            &proto.name,
            format!("at most {}", pos_specs.len()),
            args.len(),
        ));
    }

    Ok(())
}

fn validate_and_coerce_arg(
    vm: &mut Vm,
    func_name: &str,
    arg_label: &str,
    mut val: Value,
    sp: &CompiledArgSpec,
    stdout: &mut String,
) -> Result<Value> {
    if let Some(cname) = &sp.class_name {
        val = coerce_or_check_class(vm, func_name, arg_label, val, cname)?;
    }
    for (vname, varg_chunks) in &sp.validators {
        let mut vargs = Vec::with_capacity(varg_chunks.len());
        for ch in varg_chunks {
            vargs.push(eval_standalone_chunk(vm, ch, "__arg_default", stdout)?);
        }
        run_validator(vm, func_name, arg_label, &val, vname, &vargs, stdout)?;
    }
    Ok(val)
}

fn coerce_or_check_class(
    vm: &Vm,
    func_name: &str,
    arg_label: &str,
    val: Value,
    cname: &str,
) -> Result<Value> {
    if let Some(nc) = NumericClass::from_name(cname) {
        if let Ok(m) = val.to_dense_matrix() {
            return Ok(Value::matrix(m.cast_to(nc)));
        }
        return Err(MatroError::runtime(format!(
            "{func_name}: argument '{arg_label}' must be convertible to {cname}, got {}",
            val.type_name()
        )));
    }
    match cname {
        "char" => match &val {
            Value::String(_) => Ok(val),
            Value::Object(o) if o.class_name == "string" => {
                if let Some(Value::Cell(c)) = o.fields.get("Data") {
                    if c.data.len() == 1 {
                        if let Ok(s) = c.data[0].as_str() {
                            return Ok(Value::String(s.to_string()));
                        }
                    }
                }
                Err(MatroError::runtime(format!(
                    "{func_name}: argument '{arg_label}' must be char"
                )))
            }
            _ => Err(MatroError::runtime(format!(
                "{func_name}: argument '{arg_label}' must be char, got {}",
                val.type_name()
            ))),
        },
        "string" => match &val {
            Value::Object(o) if o.class_name == "string" => Ok(val),
            Value::String(s) => Ok(make_string_obj(1, 1, vec![s.clone()])),
            _ => Err(MatroError::runtime(format!(
                "{func_name}: argument '{arg_label}' must be string, got {}",
                val.type_name()
            ))),
        },
        _ => {
            if vm.isa_class(&val, cname) {
                Ok(val)
            } else {
                Err(MatroError::runtime(format!(
                    "{func_name}: argument '{arg_label}' must be of class '{cname}', got '{}'",
                    val.type_name()
                )))
            }
        }
    }
}

fn run_validator(
    vm: &mut Vm,
    func_name: &str,
    arg_label: &str,
    val: &Value,
    vname: &str,
    vargs: &[Value],
    stdout: &mut String,
) -> Result<()> {
    let fail = |msg: &str| {
        Err(MatroError::runtime(format!(
            "{func_name}: validation failed for '{arg_label}' ({vname}): {msg}"
        )))
    };

    match vname {
        "mustBeNumeric" => {
            let ok = matches!(val, Value::Matrix(m) if !m.is_logical()) || matches!(val, Value::Sparse(_));
            if !ok {
                return fail("value must be numeric");
            }
        }
        "mustBeNumericOrLogical" => {
            let ok = matches!(val, Value::Matrix(_) | Value::Sparse(_));
            if !ok {
                return fail("value must be numeric or logical");
            }
        }
        "mustBeReal" => {
            let m = val.to_dense_matrix()?;
            if m.is_complex() {
                return fail("value must be real");
            }
        }
        "mustBeFinite" => {
            let m = val.to_dense_matrix()?;
            for i in 0..m.len() {
                let c = m.get_linear_c(i);
                if !c.re.is_finite() || !c.im.is_finite() {
                    return fail("value must be finite");
                }
            }
        }
        "mustBeNonNan" => {
            let m = val.to_dense_matrix()?;
            for i in 0..m.len() {
                let c = m.get_linear_c(i);
                if c.re.is_nan() || c.im.is_nan() {
                    return fail("value must not be NaN");
                }
            }
        }
        "mustBePositive" => {
            let m = val.to_dense_matrix()?;
            if m.is_complex() || m.data().iter().any(|&x| !(x > 0.0)) {
                return fail("value must be positive");
            }
        }
        "mustBeNonnegative" => {
            let m = val.to_dense_matrix()?;
            if m.is_complex() || m.data().iter().any(|&x| !(x >= 0.0)) {
                return fail("value must be nonnegative");
            }
        }
        "mustBeNegative" => {
            let m = val.to_dense_matrix()?;
            if m.is_complex() || m.data().iter().any(|&x| !(x < 0.0)) {
                return fail("value must be negative");
            }
        }
        "mustBeNonpositive" => {
            let m = val.to_dense_matrix()?;
            if m.is_complex() || m.data().iter().any(|&x| !(x <= 0.0)) {
                return fail("value must be nonpositive");
            }
        }
        "mustBeInteger" => {
            let m = val.to_dense_matrix()?;
            if m.is_complex() || m.data().iter().any(|&x| !x.is_finite() || x.fract() != 0.0) {
                return fail("value must be integer");
            }
        }
        "mustBeNonempty" => {
            let empty = match val {
                Value::Matrix(m) => m.is_empty(),
                Value::Sparse(s) => s.rows() == 0 || s.cols() == 0,
                Value::Cell(c) => c.data.is_empty(),
                Value::String(s) => s.is_empty(),
                _ => false,
            };
            if empty {
                return fail("value must not be empty");
            }
        }
        "mustBeVector" => {
            let m = val.to_dense_matrix()?;
            if m.rows() != 1 && m.cols() != 1 {
                return fail("value must be a vector");
            }
        }
        "mustBeText" | "mustBeTextScalar" => {
            let ok = match val {
                Value::String(_) => true,
                Value::Object(o) if o.class_name == "string" => {
                    if vname == "mustBeTextScalar" {
                        matches!(o.fields.get("Data"), Some(Value::Cell(c)) if c.data.len() == 1)
                    } else {
                        true
                    }
                }
                Value::Cell(c) if vname == "mustBeText" => {
                    c.data.iter().all(|v| matches!(v, Value::String(_)))
                }
                _ => false,
            };
            if !ok {
                return fail("value must be text");
            }
        }
        "mustBeGreaterThan" | "mustBeGreaterThanOrEqual" | "mustBeLessThan" | "mustBeLessThanOrEqual" => {
            let bound = vargs
                .first()
                .ok_or_else(|| MatroError::runtime(format!("{vname} requires a bound argument")))?
                .as_scalar()?;
            let m = val.to_dense_matrix()?;
            let ok = m.data().iter().all(|&x| match vname {
                "mustBeGreaterThan" => x > bound,
                "mustBeGreaterThanOrEqual" => x >= bound,
                "mustBeLessThan" => x < bound,
                _ => x <= bound,
            });
            if !ok {
                return fail(&format!("value out of bound relative to {bound}"));
            }
        }
        "mustBeInRange" => {
            if vargs.len() < 2 {
                return fail("mustBeInRange requires lo and hi bounds");
            }
            let lo = vargs[0].as_scalar()?;
            let hi = vargs[1].as_scalar()?;
            let m = val.to_dense_matrix()?;
            if m.data().iter().any(|&x| !(x >= lo && x <= hi)) {
                return fail(&format!("value must be in range [{lo}, {hi}]"));
            }
        }
        "mustBeMember" => {
            let allowed = vargs
                .first()
                .ok_or_else(|| MatroError::runtime("mustBeMember requires an allowed-set argument"))?;
            let ok = match (val, allowed) {
                (Value::String(s), Value::Cell(c)) => c
                    .data
                    .iter()
                    .any(|item| item.as_str().ok() == Some(s.as_str())),
                (Value::String(s), Value::Object(o)) if o.class_name == "string" => {
                    if let Some(Value::Cell(c)) = o.fields.get("Data") {
                        c.data.iter().any(|item| item.as_str().ok() == Some(s.as_str()))
                    } else {
                        false
                    }
                }
                (Value::Object(vo), Value::Cell(c)) if vo.class_name == "string" => {
                    if let Some(Value::Cell(vc)) = vo.fields.get("Data") {
                        vc.data.iter().all(|v| {
                            v.as_str()
                                .ok()
                                .is_some_and(|s| c.data.iter().any(|item| item.as_str().ok() == Some(s)))
                        })
                    } else {
                        false
                    }
                }
                (Value::Matrix(m), Value::Matrix(set)) => m
                    .data()
                    .iter()
                    .all(|&x| set.data().iter().any(|&y| (x - y).abs() <= 1e-12)),
                _ => false,
            };
            if !ok {
                return fail("value is not a member of the allowed set");
            }
        }
        other => {
            // Custom user-defined validator function
            let mut call_args = Vec::with_capacity(1 + vargs.len());
            call_args.push(val.clone());
            call_args.extend_from_slice(vargs);
            if let Some(func) = vm.functions.get(other).cloned() {
                vm.call_user_func(&func, &call_args, 0, None, stdout)?;
            } else {
                return fail(&format!("unknown validator '{other}'"));
            }
        }
    }
    Ok(())
}

/// Ensure complex scalar helper doesn't trigger unused warnings if needed.
#[allow(dead_code)]
fn _complex_unused(_: Complex) {}
