//! Zero-dependency WebAssembly (`wasm32-unknown-unknown` / WASI) bridge for `matro_vm`.
//!
//! Exports a persistent VM session over linear memory with:
//! - Safe default instruction fuel limit (prevents infinite loops from freezing browser tabs)
//! - In-memory VFS file mounting (`matro_mount_file`, `matro_files_json`)
//! - Bytecode disassembly (`matro_disasm`)
//! - Workspace JSON (`matro_vars_json`) & Plot JSON / SVG (`matro_plot_json`, `matro_plot_svg`)

use std::cell::RefCell;
use std::slice;

use matro_bytecode::compile;
use matro_vm::Vm;

const DEFAULT_WASM_STEP_LIMIT: u64 = 5_000_000;

thread_local! {
    static SESSION: RefCell<Vm> = RefCell::new({
        let mut vm = Vm::new();
        vm.set_step_limit(Some(DEFAULT_WASM_STEP_LIMIT));
        vm
    });
    static LAST_OUTPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Allocate `len` bytes in Wasm linear memory for the host to write UTF-8 data into.
#[unsafe(no_mangle)]
pub extern "C" fn matro_alloc(len: usize) -> *mut u8 {
    let mut buf = Vec::<u8>::with_capacity(len);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

/// Free a buffer previously allocated by `matro_alloc`.
///
/// # Safety
/// `ptr` must have been returned by `matro_alloc(len)` and not already freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn matro_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        let _ = unsafe { Vec::from_raw_parts(ptr, 0, len) };
    }
}

/// Reset all variables, user functions, and plots in the persistent Wasm VM session.
/// (Preserves mounted VFS files unless `matro_clear_files()` is called.)
#[unsafe(no_mangle)]
pub extern "C" fn matro_reset() {
    SESSION.with(|vm| vm.borrow_mut().clear());
}

/// Clear all mounted files in the in-memory VFS.
#[unsafe(no_mangle)]
pub extern "C" fn matro_clear_files() {
    SESSION.with(|vm| vm.borrow_mut().loader_mut().vfs_mut().clear());
}

/// Configure instruction fuel limit (`0` disables the limit).
#[unsafe(no_mangle)]
pub extern "C" fn matro_set_step_limit(limit: u64) {
    SESSION.with(|vm| {
        vm.borrow_mut()
            .set_step_limit(if limit == 0 { None } else { Some(limit) });
    });
}

/// Mount a UTF-8 file (e.g. `.m` function or `.csv` data) into the VM's in-memory VFS.
/// Returns `0` on success, `1` on error.
///
/// # Safety
/// Both pointer/length pairs must reference valid UTF-8 in Wasm linear memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn matro_mount_file(
    path_ptr: *const u8,
    path_len: usize,
    content_ptr: *const u8,
    content_len: usize,
) -> i32 {
    let p_bytes = unsafe { slice::from_raw_parts(path_ptr, path_len) };
    let c_bytes = unsafe { slice::from_raw_parts(content_ptr, content_len) };
    let (Ok(path), Ok(content)) = (std::str::from_utf8(p_bytes), std::str::from_utf8(c_bytes))
    else {
        set_output("Invalid UTF-8 in file path or content");
        return 1;
    };
    SESSION.with(|vm| match vm.borrow_mut().mount_file(path, content) {
        Ok(()) => 0,
        Err(e) => {
            set_output(&e.to_string());
            1
        }
    })
}

/// Mount a raw binary file (e.g., a `.mat` v5 binary dataset) into the VM's in-memory VFS.
/// Returns `0` on success, `1` on error.
///
/// # Safety
/// `path_ptr..path_ptr+path_len` must be valid UTF-8 and `data_ptr..data_ptr+data_len` valid bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn matro_mount_bytes(
    path_ptr: *const u8,
    path_len: usize,
    data_ptr: *const u8,
    data_len: usize,
) -> i32 {
    let p_bytes = unsafe { slice::from_raw_parts(path_ptr, path_len) };
    let data = unsafe { slice::from_raw_parts(data_ptr, data_len) };
    let Ok(path) = std::str::from_utf8(p_bytes) else {
        set_output("Invalid UTF-8 in file path");
        return 1;
    };
    SESSION.with(|vm| match vm.borrow_mut().mount_bytes(path, data) {
        Ok(()) => 0,
        Err(e) => {
            set_output(&e.to_string());
            1
        }
    })
}

/// Write the JSON array of mounted VFS filenames into the output buffer.
#[unsafe(no_mangle)]
pub extern "C" fn matro_files_json() {
    SESSION.with(|vm| {
        let json = vm.borrow().files_json();
        set_output(&json);
    });
}

/// Evaluate a UTF-8 MATLAB/Octave snippet in the persistent VM session.
/// Returns `0` on success or `1` on error.
///
/// # Safety
/// `ptr..ptr+len` must point to valid UTF-8 bytes in Wasm linear memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn matro_eval(ptr: *const u8, len: usize) -> i32 {
    let bytes = unsafe { slice::from_raw_parts(ptr, len) };
    let Ok(src) = std::str::from_utf8(bytes) else {
        set_output("Invalid UTF-8 input");
        return 1;
    };

    SESSION.with(|vm| match vm.borrow_mut().eval(src) {
        Ok(out) => {
            set_output(&out.stdout);
            0
        }
        Err(err) => {
            set_output(&err.to_string());
            1
        }
    })
}

/// Evaluate a UTF-8 MATLAB/Octave file with active-file path context (enables sibling & `private/` resolution,
/// and automatically invokes a 0-argument primary function file).
/// Returns `0` on success or `1` on error.
///
/// # Safety
/// Both pointer/length pairs must point to valid UTF-8 bytes in Wasm linear memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn matro_eval_file(
    path_ptr: *const u8,
    path_len: usize,
    src_ptr: *const u8,
    src_len: usize,
) -> i32 {
    let p_bytes = unsafe { slice::from_raw_parts(path_ptr, path_len) };
    let s_bytes = unsafe { slice::from_raw_parts(src_ptr, src_len) };
    let (Ok(path), Ok(src)) = (std::str::from_utf8(p_bytes), std::str::from_utf8(s_bytes)) else {
        set_output("Invalid UTF-8 input");
        return 1;
    };
    let opt_path = if path.trim().is_empty() {
        None
    } else {
        Some(path)
    };
    SESSION.with(|vm| match vm.borrow_mut().eval_file(opt_path, src, true) {
        Ok(out) => {
            set_output(&out.stdout);
            0
        }
        Err(err) => {
            set_output(&err.to_string());
            1
        }
    })
}

/// Compile a UTF-8 MATLAB/Octave snippet and place its bytecode disassembly in the output buffer.
///
/// # Safety
/// `ptr..ptr+len` must point to valid UTF-8 bytes in Wasm linear memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn matro_disasm(ptr: *const u8, len: usize) -> i32 {
    let bytes = unsafe { slice::from_raw_parts(ptr, len) };
    let Ok(src) = std::str::from_utf8(bytes) else {
        set_output("Invalid UTF-8 input");
        return 1;
    };
    match compile(src) {
        Ok(chunk) => {
            set_output(&chunk.disassemble("main"));
            0
        }
        Err(err) => {
            set_output(&err.to_string());
            1
        }
    }
}

/// Write the workspace variables JSON array into the output buffer.
#[unsafe(no_mangle)]
pub extern "C" fn matro_vars_json() {
    SESSION.with(|vm| {
        let json = vm.borrow().vars_json();
        set_output(&json);
    });
}

/// Write the current plot figure JSON into the output buffer.
#[unsafe(no_mangle)]
pub extern "C" fn matro_plot_json() {
    SESSION.with(|vm| {
        let json = vm.borrow().figure().to_json();
        set_output(&json);
    });
}

/// Render the current plot figure as SVG into the output buffer (empty if no series plotted).
#[unsafe(no_mangle)]
pub extern "C" fn matro_plot_svg(width: f64, height: f64) {
    SESSION.with(|vm| {
        let svg = vm.borrow().figure().to_svg(width, height);
        set_output(&svg);
    });
}

/// Render all non-empty plot figures as a JSON array `[{"id":1,"active":true,"svg":"..."}]`.
#[unsafe(no_mangle)]
pub extern "C" fn matro_plots_json(width: f64, height: f64) {
    SESSION.with(|vm| {
        let json = vm.borrow().figure().all_figures_svg_json(width, height);
        set_output(&json);
    });
}

/// Pointer to the UTF-8 output buffer from the last call.
#[unsafe(no_mangle)]
pub extern "C" fn matro_output_ptr() -> *const u8 {
    LAST_OUTPUT.with(|buf| buf.borrow().as_ptr())
}

/// Byte length of the UTF-8 output buffer from the last call.
#[unsafe(no_mangle)]
pub extern "C" fn matro_output_len() -> usize {
    LAST_OUTPUT.with(|buf| buf.borrow().len())
}

fn set_output(s: &str) {
    LAST_OUTPUT.with(|buf| {
        let mut b = buf.borrow_mut();
        b.clear();
        b.extend_from_slice(s.as_bytes());
    });
}
