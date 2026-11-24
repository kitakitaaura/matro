# Matro

**Matro** is a modular, pure-Rust, **zero-external-dependency** MATLAB/Octave bytecode virtual machine targeting native CLI, WebAssembly in the browser (`wasm32-unknown-unknown`, **0 Wasm imports**), and portable WASI (`wasm32-wasip1`).

Inspired by the crate-layered architecture of [Ristretto](https://github.com/theseus-rs/ristretto), Matro cleanly separates lexing/parsing, bytecode compilation, virtual filesystem loading, numerical intrinsics, and execution across an 8-crate Cargo workspace.

## Architecture

```
MATLAB/Octave (.m)
       │
       ▼
  matro_parser      Lexer & recursive-descent AST (space-separated matrices, ' vs .',
       │            classdef, closures, structs, cells, ... continuations, command syntax)
       ▼
  matro_bytecode    Stack-based bytecode compiler, constant folding, indexed local slots,
       │            zero-alloc range loop ops, line-number tables, disassembler
       ▼
  matro_vm ◄──────► matro_intrinsics  (180+ built-ins: linalg, FFT, DSP, regex, tables, 2D/3D plots)
       │   ◄──────► matro_loader      (VFS, addpath/.m autoload, CSV, Level 5 .mat + zlib inflate)
       │   ◄──────► matro_types       (CoW N-D Matrix, Complex, Sparse CSC, Table, Map, SVG PlotFigure)
       ▼
  ┌──────────────┬───────────────────────────┬──────────────────────────┐
  │  matro_cli   │        matro_wasm         │     packages/matro-wasm  │
  │ Native & WASI│ 0-import Wasm cdylib (1MB)│  ESM + TypeScript SDK    │
  └──────────────┴───────────────────────────┴──────────────────────────┘
```

## Key Features

- **Core MATLAB/Octave Language**
  - Column-major N-D real, complex (`3 + 4i`), and logical matrices with Copy-on-Write (`Arc<Matrix>`) value semantics and R2016b+ N-D implicit broadcasting
  - Control flow: `if`/`elseif`/`else`, `for` (zero-alloc `a:step:b` ranges & column iteration), `while`, `switch`/`case`/`otherwise`, `break`, `continue`, `try`/`catch`
  - Functions: multi-return `[a, b] = f(x)`, `varargin`/`varargout`, `nargin`/`nargout`, `narginchk`/`nargoutchk`, `inputParser`, `global`/`persistent`, `eval`/`evalc`
  - Closures & OOP: `@(x) ...`, multi-arity `@func` handles, **mutable outer-scope nested functions**, `classdef` value & `< handle` classes with inheritance, struct arrays, cell arrays, comma-separated lists (`c{:}`)
  - Modern data containers: `table` (`T.col`, `T(r,c)`, `T{r,c}`, `height`, `width`, `summary`, `sortrows`, `readtable`/`writetable`) and `containers.Map` / `dictionary`
- **Linear Algebra, Sparse & Solvers**
  - Real & **complex** `lu`, `qr`, `chol`, `svd`, `eig`, `inv`, `pinv`, `det`, `rank`, `cond`, `null`, `orth`, `expm`, `A \ b`
  - Sparse CSC matrices (`sparse`, `speye`, `full`, `nnz`, `issparse`, `S \ b`, `pcg`)
  - Numerical solvers: `ode45` (Dormand-Prince RK4(5)), `ode15s` (adaptive L-stable TR-BDF2 stiff solver with cached LU), `fzero` (Brent), `integral` (adaptive Simpson), `fminsearch` (Nelder-Mead)
- **Signal Processing, Regex & File I/O**
  - `fft`/`ifft`, `fft2`/`ifft2`, `filter` (FIR/IIR), `conv`/`conv2`, `interp1`/`interp2`, `roots`/`poly`/`polyfit`/`polyval`
  - Pure-Rust `regexp`, `regexpi`, `regexprep`, `jsondecode`/`jsonencode`, set operations (`unique`, `ismember`, `setdiff`, `intersect`, `union`)
  - VFS-backed File I/O (`fopen`, `fclose`, `fprintf(fid, ...)`, `fgetl`, `fgets`, `fread`, `fwrite`, `feof`, `ftell`, `frewind`, `textscan`) + Level 5 `.mat` reader/writer with built-in RFC 1950/1951 zlib/DEFLATE `miCOMPRESSED` support
- **Pure-Rust 2D & 3D SVG Plotting**
  - `plot`, `semilogy`, `loglog`, `scatter`, `bar`, `stem`, `histogram`, `imagesc`, `contour`, `quiver`, `subplot`, `legend`, `xlim`/`ylim`, `hold`, `grid`
  - 3D perspective SVG renderer with painter's algorithm: `surf`, `mesh`, `plot3`, `view(az, el)`, `zlabel`

## Quick Start

### Native CLI & REPL
```bash
# Run tests across all 8 crates
cargo test --workspace

# Build release CLI
cargo build -p matro_cli --release

# Interactive multiline REPL
./target/release/matro

# Run a script and export figure to SVG or workspace to .mat
./target/release/matro script.m --svg plot.svg --export-mat out.mat

# Evaluate inline code, syntax-check, disassemble bytecode, or benchmark
./target/release/matro -e "A = [3 1; 1 2]; disp(A \ [9; 8]);"
./target/release/matro --check script.m
./target/release/matro --disasm "s = 0; for k = 1:10, s = s + k; end"
./target/release/matro --bench
```

### Browser Playground
```bash
# Rebuild the zero-import WebAssembly bundle
cargo build -p matro_wasm --target wasm32-unknown-unknown --release
cp target/wasm32-unknown-unknown/release/matro_wasm.wasm web/

# Serve locally and open http://localhost:8080
python3 -m http.server 8080 -d web
```

### Embed with `@matro/wasm` (Browser & Node.js)
```js
import { loadMatro } from "./packages/matro-wasm/index.mjs";

const matro = await loadMatro();
matro.mountFile("lib/hypot2.m", "function h = hypot2(a, b)\n h = sqrt(a.^2 + b.^2);\nend\n");

const res = matro.eval(`
  addpath('lib');
  fprintf('hypot = %.2f\\n', hypot2(3, 4));
  [X, Y] = meshgrid(linspace(-4, 4, 20));
  R = sqrt(X.^2 + Y.^2) + eps;
  surf(X, Y, sin(R) ./ R);
`);

console.log(res.output);       // "hypot = 5.00\n"
const svg = matro.plotSvg();   // Pure SVG string
```

### Portable WASI CLI
```bash
cargo build -p matro_cli --target wasm32-wasip1 --release
wasmtime target/wasm32-wasip1/release/matro.wasm -e "disp(eig([0 -1; 1 0]))"
```

## Benchmarks

Measured with `./target/release/matro --bench` (Apple Silicon, `--release`, zero external dependencies):

| Benchmark | Time |
|---|---:|
| Zero-alloc range loop (100,000 iters) | `20.79 ms` |
| Blocked dense `matmul` (128×128) | `1.18 ms` |
| Dense linear solve `A \ b` (80×80) | `0.45 ms` |
| Radix-2 `fft` (4,096 pts) | `0.35 ms` |
| Sparse Poisson `pcg` (400×400) | `0.76 ms` |
| Stiff `ode15s` (Van der Pol $\mu=50$, $t \in [0,120]$) | `11.57 ms` |

- `web/matro_wasm.wasm`: **~1.1 MB**, **0 Wasm imports**
- `target/wasm32-wasip1/release/matro.wasm`: **~1.1 MB**

## License

MIT — see [LICENSE](LICENSE).
