/**
 * Zero-dependency JS wrapper around `matro_wasm.wasm`.
 */
export class MatroRuntime {
  constructor(instance) {
    this.exports = instance.exports;
    this.encoder = new TextEncoder();
    this.decoder = new TextDecoder();
  }

  static async load(wasmUrl = "./matro_wasm.wasm") {
    const resp = await fetch(wasmUrl);
    if (!resp.ok) {
      throw new Error(`Failed to fetch ${wasmUrl}: HTTP ${resp.status}`);
    }
    const bytes = await resp.arrayBuffer();
    const { instance } = await WebAssembly.instantiate(bytes, {});
    return new MatroRuntime(instance);
  }

  _readOutput() {
    const ptr = this.exports.matro_output_ptr();
    const len = this.exports.matro_output_len();
    if (len === 0) return "";
    const view = new Uint8Array(this.exports.memory.buffer, ptr, len);
    return this.decoder.decode(view);
  }

  _allocStr(text) {
    const encoded = this.encoder.encode(text);
    const ptr = this.exports.matro_alloc(encoded.length);
    new Uint8Array(this.exports.memory.buffer, ptr, encoded.length).set(encoded);
    return { ptr, len: encoded.length };
  }

  _callWithString(fn, text) {
    const { ptr, len } = this._allocStr(text);
    const status = fn(ptr, len);
    this.exports.matro_free(ptr, len);
    return { ok: status === 0, output: this._readOutput() };
  }

  eval(code) {
    return this._callWithString(this.exports.matro_eval, code);
  }

  disasm(code) {
    return this._callWithString(this.exports.matro_disasm, code);
  }

  mountFile(path, content) {
    const p = this._allocStr(path);
    const c = this._allocStr(content);
    const status = this.exports.matro_mount_file(p.ptr, p.len, c.ptr, c.len);
    this.exports.matro_free(p.ptr, p.len);
    this.exports.matro_free(c.ptr, c.len);
    return { ok: status === 0, output: this._readOutput() };
  }

  files() {
    this.exports.matro_files_json();
    const raw = this._readOutput();
    return raw ? JSON.parse(raw) : [];
  }

  clearFiles() {
    this.exports.matro_clear_files();
  }

  setStepLimit(limit) {
    this.exports.matro_set_step_limit(BigInt(limit));
  }

  vars() {
    this.exports.matro_vars_json();
    const raw = this._readOutput();
    return raw ? JSON.parse(raw) : [];
  }

  plotJson() {
    this.exports.matro_plot_json();
    const raw = this._readOutput();
    return raw ? JSON.parse(raw) : null;
  }

  plotSvg(width = 640, height = 360) {
    this.exports.matro_plot_svg(width, height);
    return this._readOutput();
  }

  reset() {
    this.exports.matro_reset();
  }
}
