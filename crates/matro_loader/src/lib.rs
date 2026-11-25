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
    fn list_dirs(&self) -> Vec<String> {
        Vec::new()
    }
    fn is_empty_fs(&self) -> bool {
        false
    }
    fn remove(&mut self, path: &str);
    fn clear(&mut self);
}

pub fn normalize_preserve_abs(path: &str) -> String {
    let p = path.trim().replace('\\', "/");
    let is_abs = p.starts_with('/');
    let trimmed = p.trim_start_matches("./").trim_start_matches('/').trim_end_matches('/');
    if is_abs && !trimmed.is_empty() {
        format!("/{trimmed}")
    } else {
        trimmed.to_string()
    }
}

/// Parent directory of a normalized VFS path (empty string for root-level files).
pub fn parent_dir(path: &str) -> String {
    let norm = normalize_preserve_abs(path);
    match norm.rfind('/') {
        Some(0) => "/".to_string(),
        Some(pos) => norm[..pos].to_string(),
        None => String::new(),
    }
}

/// Filename stem (without directory or `.m` extension).
pub fn file_stem(path: &str) -> String {
    let norm = normalize_preserve_abs(path);
    let base = match norm.rfind('/') {
        Some(pos) => &norm[pos + 1..],
        None => norm.as_str(),
    };
    base.strip_suffix(".m").unwrap_or(base).to_string()
}

/// Join a normalized directory (possibly empty for root) and a relative path.
pub fn join_dir(dir: &str, rel: &str) -> String {
    let d = normalize_preserve_abs(dir);
    let r = MemoryVfs::normalize(rel);
    if d.is_empty() {
        r
    } else if r.is_empty() {
        d
    } else if d == "/" {
        format!("/{r}")
    } else {
        format!("{d}/{r}")
    }
}

/// Result of resolving a `.m` function, script, or class file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModule {
    pub path: String,
    pub dir: String,
    pub is_private: bool,
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
        let mut cur = key.as_str();
        while let Some(pos) = cur.rfind('/') {
            let parent = &cur[..pos];
            if parent.is_empty() {
                break;
            }
            self.dirs.insert(parent.to_string());
            cur = parent;
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

    fn list_dirs(&self) -> Vec<String> {
        self.dirs.iter().cloned().collect()
    }

    fn is_empty_fs(&self) -> bool {
        self.files.is_empty()
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

    fn list_dirs(&self) -> Vec<String> {
        let mut out: BTreeSet<String> = self.overlay.list_dirs().into_iter().collect();
        fn walk_os(dir: &std::path::Path, prefix: &str, out: &mut BTreeSet<String>, depth: usize) {
            if depth > 12 {
                return;
            }
            let Ok(rd) = std::fs::read_dir(dir) else {
                return;
            };
            for ent in rd.flatten() {
                let p = ent.path();
                if p.is_dir() {
                    if let Some(name) = ent.file_name().to_str() {
                        if name.starts_with('.') {
                            continue;
                        }
                        let rel = if prefix.is_empty() {
                            name.to_string()
                        } else {
                            format!("{prefix}/{name}")
                        };
                        out.insert(rel.clone());
                        walk_os(&p, &rel, out, depth + 1);
                    }
                }
            }
        }
        walk_os(std::path::Path::new("."), "", &mut out, 0);
        out.into_iter().collect()
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
    resolve_cache: HashMap<(Option<String>, String), Option<ResolvedModule>>,
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
            resolve_cache: HashMap::new(),
        }
    }

    pub fn vfs(&self) -> &dyn Vfs {
        self.vfs.as_ref()
    }

    pub fn vfs_mut(&mut self) -> &mut dyn Vfs {
        self.cache.clear();
        self.resolve_cache.clear();
        self.vfs.as_mut()
    }

    pub fn mount_file(&mut self, path: &str, content: &str) -> Result<()> {
        self.cache.clear();
        self.resolve_cache.clear();
        self.vfs.write_string(path, content)
    }

    pub fn mount_bytes(&mut self, path: &str, data: &[u8]) -> Result<()> {
        self.cache.clear();
        self.resolve_cache.clear();
        self.vfs.write_bytes(path, data)
    }

    /// Prepend one or more (`:` / `;` separated) directories to the front of MATLAB's search path.
    pub fn add_path(&mut self, dirs: &str) {
        self.resolve_cache.clear();
        let parts: Vec<String> = dirs
            .split([':', ';'])
            .map(normalize_preserve_abs)
            .filter(|s| !s.is_empty())
            .collect();
        for clean in parts.into_iter().rev() {
            self.search_paths.retain(|d| d != &clean);
            self.search_paths.insert(0, clean);
        }
    }

    /// Append one or more (`:` / `;` separated) directories to the end of MATLAB's search path.
    pub fn add_path_end(&mut self, dirs: &str) {
        self.resolve_cache.clear();
        for raw in dirs.split([':', ';']) {
            let clean = normalize_preserve_abs(raw);
            if !clean.is_empty() && !self.search_paths.contains(&clean) {
                self.search_paths.push(clean);
            }
        }
    }

    /// Remove one or more (`:` / `;` separated) directories from MATLAB's search path.
    pub fn rm_path(&mut self, dirs: &str) {
        self.resolve_cache.clear();
        for raw in dirs.split([':', ';']) {
            let clean = normalize_preserve_abs(raw);
            if !clean.is_empty() {
                self.search_paths.retain(|d| d != &clean);
            }
        }
    }

    pub fn search_paths(&self) -> &[String] {
        &self.search_paths
    }

    pub fn reset_search_paths(&mut self) {
        self.search_paths = vec![String::new()];
        self.resolve_cache.clear();
    }

    /// MATLAB `genpath(root)`: return `root` and all non-`private`, non-`@`, non-`+` subdirectories.
    pub fn gen_path(&self, root: &str) -> Vec<String> {
        let norm_root = MemoryVfs::normalize(root);
        let is_excluded = |d: &str| -> bool {
            d.split('/').any(|seg| {
                seg == "private" || seg.starts_with('@') || seg.starts_with('+') || seg.starts_with('.')
            })
        };
        let mut out = Vec::new();
        if !norm_root.is_empty() && !is_excluded(&norm_root) && self.vfs.is_dir(&norm_root) {
            out.push(norm_root.clone());
        }
        let prefix = if norm_root.is_empty() {
            String::new()
        } else {
            format!("{norm_root}/")
        };
        for d in self.vfs.list_dirs() {
            let dn = MemoryVfs::normalize(&d);
            if dn.is_empty() || is_excluded(&dn) {
                continue;
            }
            if norm_root.is_empty() || dn.starts_with(&prefix) {
                if !out.contains(&dn) {
                    out.push(dn);
                }
            }
        }
        out
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.resolve_cache.clear();
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

    /// Check whether a MATLAB `+package` directory exists on disk/VFS, in `cwd`, or along `search_paths`.
    pub fn has_package(&self, dotted: &str) -> bool {
        let pdir = Self::package_dir(dotted);
        if pdir.is_empty() {
            return false;
        }
        let cwd = normalize_preserve_abs(&self.vfs.cwd());
        if !cwd.is_empty() && self.vfs.is_dir(&join_dir(&cwd, &pdir)) {
            return true;
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
        let cwd = normalize_preserve_abs(&self.vfs.cwd());
        if !cwd.is_empty() {
            let cwd_path = join_dir(&cwd, &rel);
            if self.vfs.exists(&cwd_path) {
                return Some(cwd_path);
            }
        }
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

    fn probe_dir_for_name(&self, dir: &str, name: &str) -> Option<ResolvedModule> {
        let direct = join_dir(dir, &format!("{name}.m"));
        if self.vfs.exists(&direct) {
            return Some(ResolvedModule {
                dir: parent_dir(&direct),
                path: direct,
                is_private: false,
            });
        }
        let class_folder = join_dir(dir, &format!("@{name}/{name}.m"));
        if self.vfs.exists(&class_folder) {
            return Some(ResolvedModule {
                dir: parent_dir(&class_folder),
                path: class_folder,
                is_private: false,
            });
        }
        None
    }

    /// Resolve a `private/<name>.m` helper scoped strictly to `caller_dir`.
    pub fn resolve_private(&self, name: &str, caller_dir: &str) -> Option<ResolvedModule> {
        if self.vfs.is_empty_fs() || name.contains('/') || name.contains('\\') || name.contains('.') {
            return None;
        }
        let priv_path = join_dir(&join_dir(caller_dir, "private"), &format!("{name}.m"));
        if self.vfs.exists(&priv_path) {
            return Some(ResolvedModule {
                dir: parent_dir(&priv_path),
                path: priv_path,
                is_private: true,
            });
        }
        None
    }

    /// Resolve a function, script, or class `.m` file in full MATLAB precedence order:
    /// 1. `<caller_dir>/private/<name>.m`
    /// 2. `<caller_dir>/<name>.m` or `<caller_dir>/@<name>/<name>.m`
    /// 3. `<cwd>/<name>.m` or `<cwd>/@<name>/<name>.m`
    /// 4. `<search_paths[i]>/<name>.m` or `<search_paths[i]>/@<name>/<name>.m`
    pub fn resolve_in_context(
        &mut self,
        name: &str,
        caller_dir: Option<&str>,
    ) -> Option<ResolvedModule> {
        if self.vfs.is_empty_fs() || name.is_empty() {
            return None;
        }
        let norm_caller = caller_dir.map(normalize_preserve_abs);
        let key = (norm_caller.clone(), name.to_string());
        if let Some(cached) = self.resolve_cache.get(&key) {
            return cached.clone();
        }
        let found = self.resolve_in_context_uncached(name, norm_caller.as_deref());
        self.resolve_cache.insert(key, found.clone());
        found
    }

    fn resolve_in_context_uncached(
        &self,
        name: &str,
        caller_dir: Option<&str>,
    ) -> Option<ResolvedModule> {
        // Explicit relative/nested path (e.g. "sub/foo.m")
        if name.contains('/') || name.contains('\\') || name.ends_with(".m") {
            let target = if name.ends_with(".m") {
                name.to_string()
            } else {
                format!("{name}.m")
            };
            if let Some(cd) = caller_dir {
                let p = join_dir(cd, &target);
                if self.vfs.exists(&p) {
                    return Some(ResolvedModule {
                        dir: parent_dir(&p),
                        path: p,
                        is_private: false,
                    });
                }
            }
            let cwd = normalize_preserve_abs(&self.vfs.cwd());
            if !cwd.is_empty() {
                let p = join_dir(&cwd, &target);
                if self.vfs.exists(&p) {
                    return Some(ResolvedModule {
                        dir: parent_dir(&p),
                        path: p,
                        is_private: false,
                    });
                }
            }
            let norm = normalize_preserve_abs(&target);
            if self.vfs.exists(&norm) {
                return Some(ResolvedModule {
                    dir: parent_dir(&norm),
                    path: norm,
                    is_private: false,
                });
            }
            for dir in &self.search_paths {
                let p = join_dir(dir, &target);
                if self.vfs.exists(&p) {
                    return Some(ResolvedModule {
                        dir: parent_dir(&p),
                        path: p,
                        is_private: false,
                    });
                }
            }
            return None;
        }

        // 1. Caller directory's `private/` folder (highest file-resolution priority)
        if let Some(cd) = caller_dir {
            if let Some(m) = self.resolve_private(name, cd) {
                return Some(m);
            }
            // 2. Sibling file / @Class folder in caller's directory
            if let Some(m) = self.probe_dir_for_name(cd, name) {
                return Some(m);
            }
        }

        // 3. Current working directory
        let cwd = normalize_preserve_abs(&self.vfs.cwd());
        if caller_dir != Some(cwd.as_str()) {
            if let Some(m) = self.probe_dir_for_name(&cwd, name) {
                return Some(m);
            }
        }

        // 4. Configured `search_paths` (in MATLAB precedence order)
        for dir in &self.search_paths {
            if let Some(m) = self.probe_dir_for_name(dir, name) {
                return Some(m);
            }
        }
        None
    }

    /// Resolve a separate method file inside an `@ClassName/` class folder (`@ClassName/<method>.m`).
    pub fn resolve_class_method_file(
        &self,
        class_name: &str,
        method_name: &str,
        caller_dir: Option<&str>,
    ) -> Option<String> {
        if self.vfs.is_empty_fs() {
            return None;
        }
        let rel = format!("@{class_name}/{method_name}.m");
        if let Some(cd) = caller_dir {
            let p = join_dir(cd, &rel);
            if self.vfs.exists(&p) {
                return Some(p);
            }
        }
        let cwd = normalize_preserve_abs(&self.vfs.cwd());
        if !cwd.is_empty() {
            let p = join_dir(&cwd, &rel);
            if self.vfs.exists(&p) {
                return Some(p);
            }
        }
        if self.vfs.exists(&rel) {
            return Some(rel);
        }
        for dir in &self.search_paths {
            if dir.is_empty() {
                continue;
            }
            let p = join_dir(dir, &rel);
            if self.vfs.exists(&p) {
                return Some(p);
            }
        }
        None
    }

    /// Resolve a bare module or script path across `cwd` and `search_paths`.
    pub fn resolve_path(&self, name_or_path: &str) -> Option<String> {
        let candidates = if name_or_path.contains('.') {
            vec![name_or_path.to_string()]
        } else {
            vec![format!("{name_or_path}.m"), name_or_path.to_string()]
        };
        let cwd = normalize_preserve_abs(&self.vfs.cwd());
        for cand in &candidates {
            if !cwd.is_empty() {
                let cwd_p = join_dir(&cwd, cand);
                if self.vfs.exists(&cwd_p) {
                    return Some(cwd_p);
                }
            }
            if self.vfs.exists(cand) {
                return Some(normalize_preserve_abs(cand));
            }
            for dir in &self.search_paths {
                if dir.is_empty() {
                    continue;
                }
                let joined = format!("{dir}/{cand}");
                if self.vfs.exists(&joined) {
                    return Some(normalize_preserve_abs(&joined));
                }
            }
        }
        None
    }

    /// Load and compile a `.m` file at an already-resolved path (cached & stamped with `source_file`).
    pub fn load_module_at(&mut self, resolved_path: &str) -> Result<Option<Chunk>> {
        let key = normalize_preserve_abs(resolved_path);
        if let Some(chunk) = self.cache.get(&key) {
            return Ok(Some(chunk.clone()));
        }
        let Some(src) = self.vfs.read_to_string(&key) else {
            return Ok(None);
        };
        let mut chunk = compile(&src)?;
        chunk.set_source_file(&key);
        self.cache.insert(key, chunk.clone());
        Ok(Some(chunk))
    }

    /// Load and compile a `.m` file by function/script name (cached by resolved path).
    pub fn load_module(&mut self, name_or_path: &str) -> Result<Option<Chunk>> {
        let Some(resolved) = self.resolve_path(name_or_path) else {
            return Ok(None);
        };
        self.load_module_at(&resolved)
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
