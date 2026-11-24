//! Zero-dependency MATLAB Level 5 (`.mat`) binary file reader & writer.
//!
//! Supports uncompressed `miMATRIX` variables: real/complex/logical double matrices
//! and `mxCHAR_CLASS` strings, in both little-endian (`IM`) and big-endian (`MI`) files.

use std::collections::BTreeMap;

use matro_types::{MatroError, Matrix, Result, Value};

const MI_INT8: u32 = 1;
const MI_UINT8: u32 = 2;
const MI_INT16: u32 = 3;
const MI_UINT16: u32 = 4;
const MI_INT32: u32 = 5;
const MI_UINT32: u32 = 6;
const MI_SINGLE: u32 = 7;
const MI_DOUBLE: u32 = 9;
const MI_MATRIX: u32 = 14;
const MI_COMPRESSED: u32 = 15;
const MI_UTF8: u32 = 16;
const MI_UTF16: u32 = 17;

const MX_CHAR_CLASS: u8 = 4;
const MX_DOUBLE_CLASS: u8 = 6;

const FLAG_COMPLEX: u32 = 0x0800;
const FLAG_LOGICAL: u32 = 0x0200;

/// Serialize named Matro variables into a standard MATLAB Level 5 `.mat` binary buffer.
pub fn write_mat_v5(vars: &[(&str, &Value)]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(256);

    // 128-byte MATLAB v5 header
    let mut header = [b' '; 116];
    let desc = b"MATLAB 5.0 MAT-file, Created by Matro";
    header[..desc.len()].copy_from_slice(desc);
    out.extend_from_slice(&header);
    out.extend_from_slice(&[0u8; 8]); // subsystem data offset
    out.extend_from_slice(&0x0100u16.to_le_bytes()); // version
    out.extend_from_slice(b"IM"); // little-endian

    for &(name, val) in vars {
        match val {
            Value::Matrix(m) => write_matrix_element(&mut out, name, m),
            Value::Sparse(s) => write_matrix_element(&mut out, name, &s.to_dense()),
            Value::String(s) => write_string_element(&mut out, name, s),
            other => {
                return Err(MatroError::runtime(format!(
                    "save(.mat): unsupported type '{}' for variable '{name}'",
                    other.type_name()
                )));
            }
        }
    }

    Ok(out)
}

fn pad8(buf: &mut Vec<u8>) {
    let rem = buf.len() % 8;
    if rem != 0 {
        buf.resize(buf.len() + (8 - rem), 0);
    }
}

fn write_element(buf: &mut Vec<u8>, dtype: u32, payload: &[u8]) {
    buf.extend_from_slice(&dtype.to_le_bytes());
    buf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    buf.extend_from_slice(payload);
    pad8(buf);
}

fn write_matrix_element(out: &mut Vec<u8>, name: &str, m: &Matrix) {
    let mut body = Vec::new();

    // 1. Array Flags
    let mut flags = MX_DOUBLE_CLASS as u32;
    if m.is_complex() {
        flags |= FLAG_COMPLEX;
    }
    if m.is_logical() {
        flags |= FLAG_LOGICAL;
    }
    let mut flags_bytes = [0u8; 8];
    flags_bytes[0..4].copy_from_slice(&flags.to_le_bytes());
    write_element(&mut body, MI_UINT32, &flags_bytes);

    // 2. Dimensions Array
    let mut dims = [0u8; 8];
    dims[0..4].copy_from_slice(&(m.rows() as i32).to_le_bytes());
    dims[4..8].copy_from_slice(&(m.cols() as i32).to_le_bytes());
    write_element(&mut body, MI_INT32, &dims);

    // 3. Array Name
    write_element(&mut body, MI_INT8, name.as_bytes());

    // 4. Real Part (column-major f64)
    let mut re_bytes = Vec::with_capacity(m.data().len() * 8);
    for &v in m.data() {
        re_bytes.extend_from_slice(&v.to_le_bytes());
    }
    write_element(&mut body, MI_DOUBLE, &re_bytes);

    // 5. Optional Imaginary Part
    if let Some(im) = m.imag_data() {
        let mut im_bytes = Vec::with_capacity(im.len() * 8);
        for &v in im {
            im_bytes.extend_from_slice(&v.to_le_bytes());
        }
        write_element(&mut body, MI_DOUBLE, &im_bytes);
    }

    write_element(out, MI_MATRIX, &body);
}

fn write_string_element(out: &mut Vec<u8>, name: &str, s: &str) {
    let mut body = Vec::new();
    let mut flags_bytes = [0u8; 8];
    flags_bytes[0..4].copy_from_slice(&(MX_CHAR_CLASS as u32).to_le_bytes());
    write_element(&mut body, MI_UINT32, &flags_bytes);

    let utf16: Vec<u16> = s.encode_utf16().collect();
    let mut dims = [0u8; 8];
    dims[0..4].copy_from_slice(&1i32.to_le_bytes());
    dims[4..8].copy_from_slice(&(utf16.len() as i32).to_le_bytes());
    write_element(&mut body, MI_INT32, &dims);

    write_element(&mut body, MI_INT8, name.as_bytes());

    let mut ch_bytes = Vec::with_capacity(utf16.len() * 2);
    for u in utf16 {
        ch_bytes.extend_from_slice(&u.to_le_bytes());
    }
    write_element(&mut body, MI_UTF16, &ch_bytes);

    write_element(out, MI_MATRIX, &body);
}

/// Parse an uncompressed MATLAB Level 5 `.mat` binary buffer into a map of named `Value`s.
pub fn read_mat_v5(bytes: &[u8]) -> Result<BTreeMap<String, Value>> {
    if bytes.len() < 128 {
        return Err(MatroError::runtime("Invalid .mat file: shorter than 128-byte header"));
    }
    let endian = &bytes[126..128];
    let le = match endian {
        b"IM" => true,
        b"MI" => false,
        _ => {
            return Err(MatroError::runtime(
                "Invalid .mat file: unrecognized endian marker (expected MATLAB v5 'IM' or 'MI')",
            ));
        }
    };

    let mut pos = 128usize;
    let mut vars = BTreeMap::new();

    while pos + 8 <= bytes.len() {
        let (dtype, payload, next_pos) = read_element(bytes, pos, le)?;
        pos = next_pos;

        if dtype == MI_COMPRESSED {
            let decompressed = crate::inflate::inflate_zlib(payload)?;
            let mut inner_pos = 0usize;
            while inner_pos + 8 <= decompressed.len() {
                let (inner_type, inner_payload, inner_next) =
                    read_element(&decompressed, inner_pos, le)?;
                inner_pos = inner_next;
                if inner_type == MI_MATRIX {
                    let (name, val) = parse_matrix_element(inner_payload, le)?;
                    if !name.is_empty() {
                        vars.insert(name, val);
                    }
                }
            }
            continue;
        }
        if dtype != MI_MATRIX {
            continue;
        }
        let (name, val) = parse_matrix_element(payload, le)?;
        if !name.is_empty() {
            vars.insert(name, val);
        }
    }

    Ok(vars)
}

fn read_u32(b: &[u8], le: bool) -> u32 {
    let arr: [u8; 4] = [b[0], b[1], b[2], b[3]];
    if le { u32::from_le_bytes(arr) } else { u32::from_be_bytes(arr) }
}

fn read_u16(b: &[u8], le: bool) -> u16 {
    let arr: [u8; 2] = [b[0], b[1]];
    if le { u16::from_le_bytes(arr) } else { u16::from_be_bytes(arr) }
}

fn read_i32(b: &[u8], le: bool) -> i32 {
    read_u32(b, le) as i32
}

fn read_f64(b: &[u8], le: bool) -> f64 {
    let arr: [u8; 8] = [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]];
    if le { f64::from_le_bytes(arr) } else { f64::from_be_bytes(arr) }
}

fn read_f32(b: &[u8], le: bool) -> f32 {
    let arr: [u8; 4] = [b[0], b[1], b[2], b[3]];
    if le { f32::from_le_bytes(arr) } else { f32::from_be_bytes(arr) }
}

fn read_element<'a>(buf: &'a [u8], pos: usize, le: bool) -> Result<(u32, &'a [u8], usize)> {
    if pos + 8 > buf.len() {
        return Err(MatroError::runtime("Unexpected EOF in .mat element tag"));
    }
    let first = read_u32(&buf[pos..pos + 4], le);
    let upper = first >> 16;
    if upper != 0 {
        // Small Data Element Format (8 bytes total)
        let dtype = first & 0xFFFF;
        let len = upper as usize;
        if len > 4 {
            return Err(MatroError::runtime("Invalid small data element length in .mat file"));
        }
        return Ok((dtype, &buf[pos + 4..pos + 4 + len], pos + 8));
    }
    let dtype = first;
    let len = read_u32(&buf[pos + 4..pos + 8], le) as usize;
    let start = pos + 8;
    let end = start + len;
    if end > buf.len() {
        return Err(MatroError::runtime("Unexpected EOF in .mat element payload"));
    }
    let padded_end = (end + 7) & !7;
    Ok((dtype, &buf[start..end], padded_end.min(buf.len())))
}

fn decode_numeric(dtype: u32, payload: &[u8], count: usize, le: bool) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(count);
    match dtype {
        MI_DOUBLE => {
            for chunk in payload.chunks_exact(8).take(count) {
                out.push(read_f64(chunk, le));
            }
        }
        MI_SINGLE => {
            for chunk in payload.chunks_exact(4).take(count) {
                out.push(read_f32(chunk, le) as f64);
            }
        }
        MI_INT32 => {
            for chunk in payload.chunks_exact(4).take(count) {
                out.push(read_i32(chunk, le) as f64);
            }
        }
        MI_UINT32 => {
            for chunk in payload.chunks_exact(4).take(count) {
                out.push(read_u32(chunk, le) as f64);
            }
        }
        MI_INT16 => {
            for chunk in payload.chunks_exact(2).take(count) {
                out.push(read_u16(chunk, le) as i16 as f64);
            }
        }
        MI_UINT16 => {
            for chunk in payload.chunks_exact(2).take(count) {
                out.push(read_u16(chunk, le) as f64);
            }
        }
        MI_INT8 => {
            for &b in payload.iter().take(count) {
                out.push(b as i8 as f64);
            }
        }
        MI_UINT8 => {
            for &b in payload.iter().take(count) {
                out.push(b as f64);
            }
        }
        other => {
            return Err(MatroError::runtime(format!(
                "Unsupported numeric data type {other} in .mat file"
            )));
        }
    }
    if out.len() != count {
        return Err(MatroError::runtime("Truncated numeric array in .mat file"));
    }
    Ok(out)
}

fn parse_matrix_element(body: &[u8], le: bool) -> Result<(String, Value)> {
    let mut pos = 0;

    // 1. Array Flags
    let (_, flags_payload, next) = read_element(body, pos, le)?;
    pos = next;
    if flags_payload.len() < 4 {
        return Err(MatroError::runtime("Invalid array flags in .mat file"));
    }
    let flags_word = read_u32(&flags_payload[0..4], le);
    let mx_class = (flags_word & 0xFF) as u8;
    let is_complex = (flags_word & FLAG_COMPLEX) != 0;
    let is_logical = (flags_word & FLAG_LOGICAL) != 0;

    // 2. Dimensions
    let (_, dims_payload, next) = read_element(body, pos, le)?;
    pos = next;
    let mut dims = Vec::new();
    for chunk in dims_payload.chunks_exact(4) {
        dims.push(read_i32(chunk, le).max(0) as usize);
    }
    let rows = dims.first().copied().unwrap_or(0);
    let cols = dims.get(1).copied().unwrap_or(1);
    let total = rows.saturating_mul(cols);

    // 3. Name
    let (_, name_payload, next) = read_element(body, pos, le)?;
    pos = next;
    let name = String::from_utf8_lossy(name_payload)
        .trim_end_matches('\0')
        .to_string();

    // 4. Data
    let (pr_type, pr_payload, next) = read_element(body, pos, le)?;
    pos = next;

    if mx_class == MX_CHAR_CLASS {
        let s = match pr_type {
            MI_UTF16 | MI_UINT16 => {
                let units: Vec<u16> = pr_payload
                    .chunks_exact(2)
                    .map(|c| read_u16(c, le))
                    .collect();
                String::from_utf16_lossy(&units)
            }
            MI_UTF8 | MI_UINT8 | MI_INT8 => String::from_utf8_lossy(pr_payload).to_string(),
            _ => String::new(),
        };
        return Ok((name, Value::String(s)));
    }

    let re = decode_numeric(pr_type, pr_payload, total, le)?;
    let mat = if is_complex && pos + 8 <= body.len() {
        let (pi_type, pi_payload, _) = read_element(body, pos, le)?;
        let im = decode_numeric(pi_type, pi_payload, total, le)?;
        Matrix::from_complex_col_major(rows, cols, re, im)?
    } else {
        Matrix::from_col_major(rows, cols, re)?.with_logical(is_logical)
    };

    Ok((name, Value::matrix(mat)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use matro_types::Complex;

    #[test]
    fn mat_v5_roundtrip() {
        let a = Value::matrix(Matrix::from_row_major(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
        let z = Value::matrix(Matrix::complex_scalar(Complex::new(3.0, -4.0)));
        let s = Value::String("hello matro".to_string());
        let bytes = write_mat_v5(&[("A", &a), ("z", &z), ("label", &s)]).unwrap();
        let loaded = read_mat_v5(&bytes).unwrap();
        assert_eq!(loaded.get("A"), Some(&a));
        assert_eq!(loaded.get("z"), Some(&z));
        assert_eq!(loaded.get("label"), Some(&s));

        // Wrap the miMATRIX elements inside an miCOMPRESSED (zlib) element and verify decompression
        let mut comp_mat = bytes[..128].to_vec();
        let compressed_body = crate::inflate::zlib_wrap_stored(&bytes[128..]);
        write_element(&mut comp_mat, MI_COMPRESSED, &compressed_body);
        let loaded_comp = read_mat_v5(&comp_mat).unwrap();
        assert_eq!(loaded_comp.get("A"), Some(&a));
        assert_eq!(loaded_comp.get("z"), Some(&z));
        assert_eq!(loaded_comp.get("label"), Some(&s));
    }
}
