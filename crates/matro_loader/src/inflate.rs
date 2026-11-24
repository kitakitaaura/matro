//! Zero-dependency RFC 1950 (zlib) + RFC 1951 (DEFLATE) decompressor
//! for reading compressed MATLAB Level 5 `.mat` (`miCOMPRESSED`) data elements.

use matro_types::{MatroError, Result};

/// Decompress a zlib-wrapped (RFC 1950) byte stream.
pub fn inflate_zlib(input: &[u8]) -> Result<Vec<u8>> {
    if input.len() < 6 {
        return Err(MatroError::runtime("zlib: stream too short"));
    }
    let cmf = input[0];
    let flg = input[1];
    if (cmf & 0x0f) != 8 {
        return Err(MatroError::runtime("zlib: unsupported compression method (expected DEFLATE)"));
    }
    if ((cmf as u16) * 256 + (flg as u16)) % 31 != 0 {
        return Err(MatroError::runtime("zlib: invalid header checksum"));
    }
    let mut start = 2usize;
    if (flg & 0x20) != 0 {
        start += 4; // FDICT preset dictionary
        if start >= input.len() {
            return Err(MatroError::runtime("zlib: truncated FDICT header"));
        }
    }
    inflate_raw(&input[start..])
}

/// Decompress a raw DEFLATE (RFC 1951) bit stream.
pub fn inflate_raw(data: &[u8]) -> Result<Vec<u8>> {
    let mut br = BitReader::new(data);
    let mut out = Vec::new();

    loop {
        let bfinal = br.read_bits(1)?;
        let btype = br.read_bits(2)?;
        match btype {
            0 => {
                // Stored (uncompressed) block
                br.align_byte();
                let len = br.read_u16_le()? as usize;
                let nlen = br.read_u16_le()? as usize;
                if (len ^ 0xffff) != nlen {
                    return Err(MatroError::runtime("deflate: stored block length mismatch"));
                }
                let slice = br.read_bytes(len)?;
                out.extend_from_slice(slice);
            }
            1 => {
                // Fixed Huffman codes
                let (lit_tree, dist_tree) = fixed_trees();
                decode_huffman_block(&mut br, &lit_tree, &dist_tree, &mut out)?;
            }
            2 => {
                // Dynamic Huffman codes
                let (lit_tree, dist_tree) = read_dynamic_trees(&mut br)?;
                decode_huffman_block(&mut br, &lit_tree, &dist_tree, &mut out)?;
            }
            _ => return Err(MatroError::runtime("deflate: invalid block type 3")),
        }
        if bfinal != 0 {
            break;
        }
    }
    Ok(out)
}

struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_buf: u32,
    bit_cnt: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_buf: 0,
            bit_cnt: 0,
        }
    }

    fn fill(&mut self, needed: u32) -> Result<()> {
        while self.bit_cnt < needed {
            let b = *self
                .data
                .get(self.byte_pos)
                .ok_or_else(|| MatroError::runtime("deflate: unexpected EOF"))?;
            self.byte_pos += 1;
            self.bit_buf |= (b as u32) << self.bit_cnt;
            self.bit_cnt += 8;
        }
        Ok(())
    }

    fn read_bits(&mut self, n: u32) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        self.fill(n)?;
        let mask = (1u32 << n) - 1;
        let val = self.bit_buf & mask;
        self.bit_buf >>= n;
        self.bit_cnt -= n;
        Ok(val)
    }

    fn align_byte(&mut self) {
        let drop = self.bit_cnt % 8;
        self.bit_buf >>= drop;
        self.bit_cnt -= drop;
        // Push back any whole buffered bytes into byte_pos
        let whole = (self.bit_cnt / 8) as usize;
        self.byte_pos -= whole;
        self.bit_buf = 0;
        self.bit_cnt = 0;
    }

    fn read_u16_le(&mut self) -> Result<u16> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.byte_pos + n > self.data.len() {
            return Err(MatroError::runtime("deflate: unexpected EOF in stored block"));
        }
        let s = &self.data[self.byte_pos..self.byte_pos + n];
        self.byte_pos += n;
        Ok(s)
    }
}

struct HuffTree {
    counts: [u16; 16],
    symbols: Vec<u16>,
}

impl HuffTree {
    fn from_lengths(lengths: &[u8]) -> Result<Self> {
        let mut counts = [0u16; 16];
        for &len in lengths {
            if len as usize >= 16 {
                return Err(MatroError::runtime("deflate: Huffman code length > 15"));
            }
            if len > 0 {
                counts[len as usize] += 1;
            }
        }
        let mut offsets = [0usize; 16];
        let mut total = 0usize;
        for bits in 1..16 {
            offsets[bits] = total;
            total += counts[bits] as usize;
        }
        let mut symbols = vec![0u16; total];
        for (sym, &len) in lengths.iter().enumerate() {
            if len > 0 {
                let idx = offsets[len as usize];
                symbols[idx] = sym as u16;
                offsets[len as usize] += 1;
            }
        }
        Ok(Self { counts, symbols })
    }

    fn decode(&self, br: &mut BitReader<'_>) -> Result<u16> {
        let mut code = 0i32;
        let mut first = 0i32;
        let mut index = 0usize;
        for bits in 1..16 {
            code |= br.read_bits(1)? as i32;
            let count = self.counts[bits] as i32;
            if code - count < first {
                return Ok(self.symbols[index + (code - first) as usize]);
            }
            index += count as usize;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err(MatroError::runtime("deflate: invalid Huffman symbol"))
    }
}

fn fixed_trees() -> (HuffTree, HuffTree) {
    let mut lit_lens = vec![0u8; 288];
    for l in &mut lit_lens[0..=143] {
        *l = 8;
    }
    for l in &mut lit_lens[144..=255] {
        *l = 9;
    }
    for l in &mut lit_lens[256..=279] {
        *l = 7;
    }
    for l in &mut lit_lens[280..=287] {
        *l = 8;
    }
    let dist_lens = vec![5u8; 32];
    (
        HuffTree::from_lengths(&lit_lens).unwrap(),
        HuffTree::from_lengths(&dist_lens).unwrap(),
    )
}

fn read_dynamic_trees(br: &mut BitReader<'_>) -> Result<(HuffTree, HuffTree)> {
    const ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];
    let hlit = br.read_bits(5)? as usize + 257;
    let hdist = br.read_bits(5)? as usize + 1;
    let hclen = br.read_bits(4)? as usize + 4;

    let mut cl_lens = [0u8; 19];
    for i in 0..hclen {
        cl_lens[ORDER[i]] = br.read_bits(3)? as u8;
    }
    let cl_tree = HuffTree::from_lengths(&cl_lens)?;

    let total = hlit + hdist;
    let mut lens = Vec::with_capacity(total);
    while lens.len() < total {
        let sym = cl_tree.decode(br)?;
        match sym {
            0..=15 => lens.push(sym as u8),
            16 => {
                let prev = *lens
                    .last()
                    .ok_or_else(|| MatroError::runtime("deflate: repeat code 16 with no prev"))?;
                let rep = br.read_bits(2)? as usize + 3;
                for _ in 0..rep {
                    lens.push(prev);
                }
            }
            17 => {
                let rep = br.read_bits(3)? as usize + 3;
                for _ in 0..rep {
                    lens.push(0);
                }
            }
            18 => {
                let rep = br.read_bits(7)? as usize + 11;
                for _ in 0..rep {
                    lens.push(0);
                }
            }
            _ => return Err(MatroError::runtime("deflate: invalid code-length symbol")),
        }
    }
    if lens.len() > total {
        return Err(MatroError::runtime("deflate: dynamic code lengths overflow"));
    }
    let lit_tree = HuffTree::from_lengths(&lens[..hlit])?;
    let dist_tree = HuffTree::from_lengths(&lens[hlit..])?;
    Ok((lit_tree, dist_tree))
}

fn decode_huffman_block(
    br: &mut BitReader<'_>,
    lit_tree: &HuffTree,
    dist_tree: &HuffTree,
    out: &mut Vec<u8>,
) -> Result<()> {
    const LEN_BASE: [usize; 29] = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115,
        131, 163, 195, 227, 258,
    ];
    const LEN_EXTRA: [u32; 29] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
    ];
    const DIST_BASE: [usize; 30] = [
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
        2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
    ];
    const DIST_EXTRA: [u32; 30] = [
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
        13, 13,
    ];

    loop {
        let sym = lit_tree.decode(br)? as usize;
        if sym < 256 {
            out.push(sym as u8);
        } else if sym == 256 {
            break;
        } else {
            let lidx = sym - 257;
            if lidx >= 29 {
                return Err(MatroError::runtime("deflate: invalid length symbol"));
            }
            let length = LEN_BASE[lidx] + br.read_bits(LEN_EXTRA[lidx])? as usize;
            let didx = dist_tree.decode(br)? as usize;
            if didx >= 30 {
                return Err(MatroError::runtime("deflate: invalid distance symbol"));
            }
            let dist = DIST_BASE[didx] + br.read_bits(DIST_EXTRA[didx])? as usize;
            if dist > out.len() {
                return Err(MatroError::runtime("deflate: distance beyond output start"));
            }
            for _ in 0..length {
                let b = out[out.len() - dist];
                out.push(b);
            }
        }
    }
    Ok(())
}

/// Wrap raw bytes in a valid uncompressed zlib (RFC 1950) stream (useful for tests & writers).
pub fn zlib_wrap_stored(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() + 16);
    out.push(0x78); // CMF: DEFLATE, 32K window
    out.push(0x01); // FLG: no dict, fastest
    let mut pos = 0;
    while pos < raw.len() || pos == 0 {
        let end = (pos + 65535).min(raw.len());
        let is_final = end == raw.len();
        out.push(if is_final { 1 } else { 0 });
        let len = (end - pos) as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(&raw[pos..end]);
        pos = end;
        if raw.is_empty() {
            break;
        }
    }
    // Adler-32 checksum
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in raw {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    let adler = (b << 16) | a;
    out.extend_from_slice(&adler.to_be_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inflate_fixed_and_dynamic_zlib() {
        // 1. Stored block (BTYPE=0)
        let raw = b"hello hello hello matro matro matro!";
        let stored = zlib_wrap_stored(raw);
        assert_eq!(inflate_zlib(&stored).unwrap(), raw);

        // 2. Real zlib fixed Huffman stream (BTYPE=1) for "hello world! hello world!"
        let fixed: &[u8] = &[
            0x78, 0x9c, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0x28, 0xcf, 0x2f, 0xca, 0x49, 0x51,
            0x54, 0xc8, 0x40, 0xe2, 0x00, 0x00, 0x79, 0x0c, 0x09, 0x1b,
        ];
        assert_eq!(inflate_zlib(fixed).unwrap(), b"hello world! hello world!");

        // 3. Real zlib dynamic Huffman stream (BTYPE=2) for b"aaabbbccde" * 30
        let dynamic: &[u8] = &[
            0x78, 0x01, 0x05, 0xc1, 0x01, 0x01, 0x00, 0x00, 0x08, 0x83, 0xb0, 0xac, 0xc0, 0xed,
            0x5f, 0xc1, 0x0d, 0x50, 0x6b, 0x07, 0xa8, 0xb5, 0x03, 0xd4, 0xda, 0x01, 0x6a, 0xed,
            0x00, 0xb5, 0x76, 0x80, 0x5a, 0x3b, 0x40, 0xad, 0x1d, 0xa0, 0xd6, 0x0e, 0x50, 0x6b,
            0x07, 0xa8, 0xb5, 0x03, 0xd4, 0xda, 0x01, 0x6a, 0xed, 0x00, 0xb5, 0x76, 0x80, 0x5a,
            0x3b, 0x40, 0xad, 0x1d, 0xa0, 0xd6, 0x0e, 0x50, 0x6b, 0x07, 0xa8, 0xb5, 0x03, 0xd4,
            0xda, 0x01, 0x6a, 0xed, 0x00, 0xb5, 0x76, 0x80, 0x5a, 0x3b, 0x40, 0xad, 0x1d, 0xa0,
            0xd6, 0x0e, 0x50, 0x6b, 0x07, 0xa8, 0xb5, 0x03, 0xd4, 0xda, 0x01, 0x6a, 0xed, 0x00,
            0xb5, 0x76, 0x80, 0x5a, 0xbb, 0x07, 0xcb, 0x87, 0x73, 0x51,
        ];
        let expected: Vec<u8> = b"aaabbbccde".repeat(30);
        assert_eq!(inflate_zlib(dynamic).unwrap(), expected);
    }
}
