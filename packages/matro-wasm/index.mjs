/**
 * @matro/wasm — Zero-dependency universal (Browser + Node.js) wrapper for Matro's WebAssembly VM.
 */
export class MatroSession {
  constructor(instance) {
    this.exports = instance.exports;
    this.encoder = new TextEncoder();
    this.decoder = new TextDecoder();
  }

  _readOutput() {
    const ptr = this.exports.matro_output_ptr();
    const len = this.exports.matro_output_len();
    if (len === 0) return "";
    const view = new Uint8Array(this.exports.memory.buffer, ptr, len);
    return this.decoder.decode(view);
  }

  _allocBytes(bytes) {
    const ptr = this.exports.matro_alloc(bytes.length);
    new Uint8Array(this.exports.memory.buffer, ptr, bytes.length).set(bytes);
    return { ptr, len: bytes.length };
  }

  _allocStr(text) {
    return this._allocBytes(this.encoder.encode(text));
  }

  _callWithString(fn, text) {
    const { ptr, len } = this._allocStr(text);
    const status = fn(ptr, len);
    this.exports.matro_free(ptr, len);
    return { ok: status === 0, output: this._readOutput() };
  }

  /** Evaluate MATLAB/Octave code in the persistent session workspace. */
  eval(code) {
    return this._callWithString(this.exports.matro_eval, code);
  }

  /** Disassemble MATLAB/Octave source into Matro bytecode without executing it. */
  disasm(code) {
    return this._callWithString(this.exports.matro_disasm, code);
  }

  /** Mount a UTF-8 text file (`.m`, `.csv`, `.txt`) into the session's in-memory VFS. */
  mountFile(path, content) {
    const p = this._allocStr(path);
    const c = this._allocStr(content);
    const status = this.exports.matro_mount_file(p.ptr, p.len, c.ptr, c.len);
    this.exports.matro_free(p.ptr, p.len);
    this.exports.matro_free(c.ptr, c.len);
    return { ok: status === 0, output: this._readOutput() };
  }

  /** Mount arbitrary binary bytes (e.g. Level 5 `.mat` files) into the in-memory VFS. */
  mountBytes(path, bytes) {
    const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    const p = this._allocStr(path);
    const d = this._allocBytes(data);
    const status = this.exports.matro_mount_bytes(p.ptr, p.len, d.ptr, d.len);
    this.exports.matro_free(p.ptr, p.len);
    this.exports.matro_free(d.ptr, d.len);
    return { ok: status === 0, output: this._readOutput() };
  }

  /** List all mounted/created VFS file paths. */
  files() {
    this.exports.matro_files_json();
    const raw = this._readOutput();
    return raw ? JSON.parse(raw) : [];
  }

  /** Remove all files from the in-memory VFS. */
  clearFiles() {
    this.exports.matro_clear_files();
  }

  /** Set the VM instruction fuel limit (0 disables the limit). */
  setStepLimit(limit) {
    this.exports.matro_set_step_limit(BigInt(limit));
  }

  /** Inspect current workspace variables. */
  vars() {
    this.exports.matro_vars_json();
    const raw = this._readOutput();
    return raw ? JSON.parse(raw) : [];
  }

  /** Return the current figure as structured JSON, or `null` if no plot exists. */
  plotJson() {
    this.exports.matro_plot_json();
    const raw = this._readOutput();
    return raw ? JSON.parse(raw) : null;
  }

  /** Render the current figure (2D, heatmap, contour, quiver, or 3D surf/mesh/plot3) to SVG. */
  plotSvg(width = 640, height = 360) {
    this.exports.matro_plot_svg(width, height);
    return this._readOutput();
  }

  /** Clear workspace variables, functions, classes, open files, and plots. */
  reset() {
    this.exports.matro_reset();
  }
}

/**
 * Instantiate a fresh `MatroSession` from a URL, `ArrayBuffer`/`Uint8Array`, `Response`, or default sibling `.wasm`.
 */
export async function loadMatro(source) {
  let bytes;
  if (source instanceof ArrayBuffer) {
    bytes = source;
  } else if (ArrayBuffer.isView(source)) {
    bytes = source.buffer.slice(source.byteOffset, source.byteOffset + source.byteLength);
  } else if (typeof Response !== "undefined" && source instanceof Response) {
    bytes = await source.arrayBuffer();
  } else {
    const url = source ?? new URL("./matro_wasm.wasm", import.meta.url);
    if (typeof url === "string" || url instanceof URL) {
      const urlStr = url.toString();
      if (urlStr.startsWith("file://") && typeof process !== "undefined" && process.versions?.node) {
        const { readFile } = await import("node:fs/promises");
        bytes = await readFile(new URL(urlStr));
      } else {
        const resp = await fetch(url);
        if (!resp.ok) {
          throw new Error(`Failed to fetch Matro Wasm (${urlStr}): HTTP ${resp.status}`);
        }
        bytes = await resp.arrayBuffer();
      }
    } else {
      throw new TypeError("loadMatro: unsupported wasm source");
    }
  }
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return new MatroSession(instance);
}
