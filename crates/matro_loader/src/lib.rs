//! Pluggable virtual filesystem (`Vfs`), MATLAB Level 5 `.mat` I/O, and `.m` module loader.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Debug;

use matro_bytecode::{Chunk, compile};
use matro_types::{MatroError, Matrix, Result, Value};

pub mod inflate;
pub mod matfile;

/// Abstract filesystem interface for Matro sessions (supports both UTF-8 text and raw binary files).
pub trait Vfs: Debug {
    fn read_bytes(&self, path: &str) -> Option<Vec<u8>>;
    fn write_bytes(&mut self, path: &str, data: &[u8]) -> Result<()>;

    fn read_to_string(&self, path: &str) -> Option<String> {
        let bytes = self.read_bytes(path)?;
        String::from_utf8(bytes).ok()
    }

    fn write_string(&mut self, path: &str, content: &str) -> Result<()> {
        self.write_bytes(path, content.as_bytes())
    }

    fn exists(&self, path: &str) -> bool {
        self.read_bytes(path).is_some()
    }

    fn is_dir(&self, _path: &str) -> bool {
        false
    }

    fn mkdir(&mut self, _path: &str) -> Result<()> {
        Ok(())
    }

    fn cwd(&self) -> String {
        "/".to_string()
    }

    fn set_cwd(&mut self, _path: &str) -> Result<()> {
        Ok(())
    }

    fn list_files(&self) -> Vec<String>;
    fn remove(&mut self, path: &str);
    fn clear(&mut self);
}

/// Pure in-memory virtual filesystem (default for Wasm & embedded sessions).
#[derive(Debug, Default, Clone)]
pub struct MemoryVfs {
    files: BTreeMap<String, Vec<u8>>,
    dirs: BTreeSet<String>,
    cwd: String,
}

impl MemoryVfs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn normalize(path: &str) -> String {
        let p = path.trim().replace('\\', "/");
        p.trim_start_matches("./").trim_start_matches('/').trim_end_matches('/').to_string()
    }
}

impl Vfs for MemoryVfs {
    fn read_bytes(&self, path: &str) -> Option<Vec<u8>> {
        let key = Self::normalize(path);
        self.files.get(&key).cloned()
    }

    fn write_bytes(&mut self, path: &str, data: &[u8]) -> Result<()> {
        let key = Self::normalize(path);
        if let Some(pos) = key.rfind('/') {
            self.dirs.insert(key[..pos].to_string());
        }
        self.files.insert(key, data.to_vec());
        Ok(())
    }

    fn exists(&self, path: &str) -> bool {
        let key = Self::normalize(path);
        self.files.contains_key(&key)
    }

    fn is_dir(&self, path: &str) -> bool {
        let key = Self::normalize(path);
        if key.is_empty() || key == "." {
            return true;
        }
        if self.dirs.contains(&key) {
            return true;
        }
        let prefix = format!("{key}/");
        self.files.keys().any(|k| k.starts_with(&prefix))
    }

    fn mkdir(&mut self, path: &str) -> Result<()> {
        let key = Self::normalize(path);
        if !key.is_empty() {
            self.dirs.insert(key);
        }
        Ok(())
    }

    fn cwd(&self) -> String {
        if self.cwd.is_empty() {
            "/".to_string()
        } else {
            format!("/{}", self.cwd)
        }
    }

    fn set_cwd(&mut self, path: &str) -> Result<()> {
        let key = Self::normalize(path);
        if key.is_empty() || key == "." {
            self.cwd.clear();
            return Ok(());
        }
        if key == ".." {
            if let Some(pos) = self.cwd.rfind('/') {
                self.cwd.truncate(pos);
            } else {
                self.cwd.clear();
            }
            return Ok(());
        }
        self.cwd = key;
        Ok(())
    }

    fn list_files(&self) -> Vec<String> {
        self.files.keys().cloned().collect()
    }

    fn remove(&mut self, path: &str) {
        let key = Self::normalize(path);
        self.files.remove(&key);
        self.dirs.remove(&key);
    }

    fn clear(&mut self) {
        self.files.clear();
        self.dirs.clear();
        self.cwd.clear();
    }
}

/// OS-backed filesystem (for `matro_cli` and `wasm32-wasip1`), with an in-memory overlay.
#[derive(Debug, Default)]
pub struct OsVfs {
    overlay: MemoryVfs,
}

impl OsVfs {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Vfs for OsVfs {
    fn read_bytes(&self, path: &str) -> Option<Vec<u8>> {
        if let Some(b) = self.overlay.read_bytes(path) {
            return Some(b);
        }
        std::fs::read(path).ok()
    }

    fn write_bytes(&mut self, path: &str, data: &[u8]) -> Result<()> {
        std::fs::write(path, data)
            .map_err(|e| MatroError::runtime(format!("Failed writing '{path}': {e}")))
    }

    fn exists(&self, path: &str) -> bool {
        self.overlay.exists(path) || std::path::Path::new(path).is_file()
    }

    fn is_dir(&self, path: &str) -> bool {
        self.overlay.is_dir(path) || std::path::Path::new(path).is_dir()
    }

    fn mkdir(&mut self, path: &str) -> Result<()> {
        std::fs::create_dir_all(path)
            .map_err(|e| MatroError::runtime(format!("mkdir '{path}' failed: {e}")))
    }

    fn cwd(&self) -> String {
        std::env::current_dir()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|_| "/".to_string())
    }

    fn set_cwd(&mut self, path: &str) -> Result<()> {
        std::env::set_current_dir(path)
            .map_err(|e| MatroError::runtime(format!("cd '{path}' failed: {e}")))
    }

    fn list_files(&self) -> Vec<String> {
        let mut out = self.overlay.list_files();
        if let Ok(rd) = std::fs::read_dir(".") {
            for ent in rd.flatten() {
                if let Some(name) = ent.file_name().to_str() {
                    let s = name.to_string();
                    if !out.contains(&s) {
                        out.push(s);
                    }
                }
            }
        }
        out.sort();
        out
    }

    fn remove(&mut self, path: &str) {
        self.overlay.remove(path);
        let _ = std::fs::remove_file(path);
    }

    fn clear(&mut self) {
        self.overlay.clear();
    }
}

/// Resolves and compiles `.m` scripts and function files along MATLAB's `addpath` search path.
#[derive(Debug)]
pub struct ModuleLoader {
    vfs: Box<dyn Vfs>,
    search_paths: Vec<String>,
    cache: HashMap<String, Chunk>,
}

impl Default for ModuleLoader {
    fn default() -> Self {
        Self::with_vfs(Box::new(MemoryVfs::new()))
    }
}

impl ModuleLoader {
    pub fn with_vfs(vfs: Box<dyn Vfs>) -> Self {
        Self {
            vfs,
            search_paths: vec![String::new()],
            cache: HashMap::new(),
        }
    }

    pub fn vfs(&self) -> &dyn Vfs {
        self.vfs.as_ref()
    }

    pub fn vfs_mut(&mut self) -> &mut dyn Vfs {
        self.cache.clear();
        self.vfs.as_mut()
    }

    pub fn mount_file(&mut self, path: &str, content: &str) -> Result<()> {
        self.cache.clear();
        self.vfs.write_string(path, content)
    }

    pub fn mount_bytes(&mut self, path: &str, data: &[u8]) -> Result<()> {
        self.cache.clear();
        self.vfs.write_bytes(path, data)
    }

    pub fn add_path(&mut self, dir: &str) {
        let clean = MemoryVfs::normalize(dir);
        if !self.search_paths.contains(&clean) {
            self.search_paths.push(clean);
        }
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Convert a dotted package name `"a.b"` to `"+a/+b"`.
    pub fn package_dir(dotted: &str) -> String {
        dotted
            .split('.')
            .filter(|s| !s.is_empty())
            .map(|seg| format!("+{seg}"))
            .collect::<Vec<_>>()
            .join("/")
    }

    /// Check whether a MATLAB `+package` directory exists on disk/VFS or along `search_paths`.
    pub fn has_package(&self, dotted: &str) -> bool {
        let pdir = Self::package_dir(dotted);
        if pdir.is_empty() {
            return false;
        }
        if self.vfs.is_dir(&pdir) {
            return true;
        }
        for dir in &self.search_paths {
            if dir.is_empty() {
                continue;
            }
            let joined = format!("{dir}/{pdir}");
            if self.vfs.is_dir(&joined) {
                return true;
            }
        }
        false
    }

    /// Resolve a member `.m` file inside a dotted `+package` namespace (e.g. `pkg.sub`, `foo` -> `+pkg/+sub/foo.m`).
    pub fn resolve_package_member(&self, dotted_pkg: &str, member: &str) -> Option<String> {
        let pdir = Self::package_dir(dotted_pkg);
        if pdir.is_empty() {
            return None;
        }
        let rel = format!("{pdir}/{member}.m");
        if self.vfs.exists(&rel) {
            return Some(rel);
        }
        for dir in &self.search_paths {
            if dir.is_empty() {
                continue;
            }
            let joined = format!("{dir}/{rel}");
            if self.vfs.exists(&joined) {
                return Some(joined);
            }
        }
        None
    }

    /// Resolve a bare module or script path across `search_paths`.
    pub fn resolve_path(&self, name_or_path: &str) -> Option<String> {
        let candidates = if name_or_path.contains('.') {
            vec![name_or_path.to_string()]
        } else {
            vec![format!("{name_or_path}.m"), name_or_path.to_string()]
        };
        for cand in &candidates {
            if self.vfs.exists(cand) {
                return Some(cand.clone());
            }
            for dir in &self.search_paths {
                if dir.is_empty() {
                    continue;
                }
                let joined = format!("{dir}/{cand}");
                if self.vfs.exists(&joined) {
                    return Some(joined);
                }
            }
        }
        None
    }

    /// Load and compile a `.m` file by function/script name (cached by resolved path).
    pub fn load_module(&mut self, name_or_path: &str) -> Result<Option<Chunk>> {
        let Some(resolved) = self.resolve_path(name_or_path) else {
            return Ok(None);
        };
        if let Some(chunk) = self.cache.get(&resolved) {
            return Ok(Some(chunk.clone()));
        }
        let Some(src) = self.vfs.read_to_string(&resolved) else {
            return Ok(None);
        };
        let chunk = compile(&src)?;
        self.cache.insert(resolved, chunk.clone());
        Ok(Some(chunk))
    }

    /// Save variables to a MATLAB Level 5 `.mat` file in the VFS.
    pub fn save_mat(&mut self, path: &str, vars: &[(&str, &Value)]) -> Result<()> {
        let bytes = matfile::write_mat_v5(vars)?;
        self.vfs.write_bytes(path, &bytes)
    }

    /// Load variables from a MATLAB Level 5 `.mat` file in the VFS.
    pub fn load_mat(&self, path: &str) -> Result<BTreeMap<String, Value>> {
        let resolved = self
            .resolve_path(path)
            .or_else(|| self.resolve_path(&format!("{path}.mat")))
            .unwrap_or_else(|| path.to_string());
        let bytes = self
            .vfs
            .read_bytes(&resolved)
            .ok_or_else(|| MatroError::runtime(format!("MAT-file not found: '{path}'")))?;
        matfile::read_mat_v5(&bytes)
    }

    /// Read a numeric CSV file into a 2D `Matrix`.
    pub fn read_csv(&self, path: &str) -> Result<Matrix> {
        let resolved = self
            .resolve_path(path)
            .unwrap_or_else(|| path.to_string());
        let src = self
            .vfs
            .read_to_string(&resolved)
            .ok_or_else(|| MatroError::runtime(format!("File not found: '{path}'")))?;

        let mut rows: Vec<Vec<f64>> = Vec::new();
        for line in src.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let vals: Vec<f64> = trimmed
                .split(',')
                .map(|cell| {
                    cell.trim().parse::<f64>().map_err(|_| {
                        MatroError::runtime(format!("Invalid numeric CSV cell: '{cell}'"))
                    })
                })
                .collect::<Result<_>>()?;
            rows.push(vals);
        }
        if rows.is_empty() {
            return Ok(Matrix::empty());
        }
        let r = rows.len();
        let c = rows[0].len();
        let mut flat = Vec::with_capacity(r * c);
        for row in rows {
            if row.len() != c {
                return Err(MatroError::dimension("CSV rows have inconsistent column counts"));
            }
            flat.extend(row);
        }
        Matrix::from_row_major(r, c, &flat)
    }

    /// Write a 2D `Matrix` to a CSV file in the VFS.
    pub fn write_csv(&mut self, path: &str, m: &Matrix) -> Result<()> {
        let mut out = String::new();
        for r in 0..m.rows() {
            for c in 0..m.cols() {
                if c > 0 {
                    out.push(',');
                }
                out.push_str(&format!("{}", m.get(r, c)));
            }
            out.push('\n');
        }
        self.vfs.write_string(path, &out)
    }
}
