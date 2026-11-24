# Matro

**Matro** is a modular, pure-Rust, **zero-external-dependency** MATLAB/Octave bytecode virtual machine targeting native CLI, WebAssembly in the browser (`wasm32-unknown-unknown`, **0 Wasm imports**, `~380 KB` gzipped), and portable WASI (`wasm32-wasip1`).

**[Try the Live Browser Playground →](https://kitakitaaura.github.io/matro/)**

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
  matro_vm ◄──────► matro_intrinsics  (250+ built-ins: linalg, interp, solvers, DSP, stats, media I/O)
       │   ◄──────► matro_loader      (VFS, addpath/.m autoload, CSV, Level 5 .mat + zlib inflate)
       │   ◄──────► matro_types       (CoW N-D typed Matrix, Complex, Sparse CSC, Table, Map, SVG PlotFigure)
       ▼
  ┌──────────────┬───────────────────────────┬──────────────────────────┐
  │  matro_cli   │        matro_wasm         │     packages/matro-wasm  │
  │ Native & WASI│ 0-import Wasm cdylib      │  ESM + TypeScript SDK    │
  └──────────────┴───────────────────────────┴──────────────────────────┘
```

## Key Features

- **Core MATLAB/Octave Language**
  - Column-major N-D real, complex (`3 + 4i`), logical, `single`, and **saturating typed integer arrays** (`int8`–`uint64`, `intmin`/`intmax`, `cast`, `char`, bitwise `bitand`/`bitor`/`bitxor`/`bitshift`/`bitget`/`bitset`) with Copy-on-Write (`Arc<Matrix>`) value semantics and R2016b+ N-D implicit broadcasting
  - Control flow: `if`/`elseif`/`else`, `for` (zero-alloc `a:step:b` ranges & column iteration), `while`, `switch`/`case`/`otherwise`, `break`, `continue`, `try`/`catch`
  - Functions: multi-return `[a, b] = f(x)`, `varargin`/`varargout`, `nargin`/`nargout`, `narginchk`/`nargoutchk`, `inputParser`, **`arguments ... end` validation blocks** (positional defaults, `arguments (Repeating)`, name-value `opts.Field`, `mustBe*` validators), R2021a+ **`Key=Val` call syntax**, `global`/`persistent`, `eval`/`evalc`
  - Closures & Full `classdef` OOP: `@(x) ...`, multi-arity & multi-output `@func` handles, **mutable outer-scope nested functions**, value & `< handle` classes with **multiple inheritance (`< A & B`)**, `methods (Static)`, `properties (Constant)` & `(Dependent)` (`get.Prop`/`set.Prop`), **`enumeration`**, **`events` + `addlistener`/`notify`**, **operator overloading** (`plus`/`minus`/`mtimes`/`uminus`/`subsref`/etc.), struct arrays, cell arrays, comma-separated lists (`c{:}`)
  - Modern data types: **`datetime`**, **`duration`**, **`categorical`**, **`timetable`**, **`string` arrays** (`strlength`, `split`, `join`, `startsWith`, `endsWith`, `erase`, `strip`, `replace`, `+`), **`table`** (`T.col`, `T(r,c)`, `T{r,c}`, `height`, `width`, `summary`, `sortrows`, `readtable`/`writetable`), and `containers.Map` / `dictionary`
- **Linear Algebra, Sparse, Interpolation & Solvers**
  - Real & **complex** `lu`, `qr`, `chol`, `svd`, `eig` (cyclic Jacobi & QR), `inv`, `pinv`, `det`, `rank`, `cond`, `null`, `orth`, `expm`, `A \ b`
  - Sparse CSC matrices & large-scale iterative solvers: `sparse`, `speye`, `full`, `nnz`, `issparse`, `S \ b`, `pcg`, **`eigs`** (restarted Arnoldi / shift-and-invert), **`svds`** (subspace iteration)
  - Interpolation suite: **`pchip`** (Fritsch-Carlson monotone Hermite), **`spline`** (not-a-knot $C^2$ cubic spline), `mkpp`/`ppval`, **`interp1`** (`linear`/`nearest`/`previous`/`next`/`pchip`/`spline`), `interp2`, `ndgrid`, callable 1D/2D **`griddedInterpolant`**
  - ODE, BVP & PDE solvers: **`ode23`** (Bogacki-Shampine), **`ode45`** (Dormand-Prince RK4(5)), **`ode113`** (Adams-Bashforth-Moulton PECE), **`ode15s`** (adaptive L-stable TR-BDF2 stiff solver with cached LU), **`bvpinit`/`bvp4c`/`deval`** (4th-order Lobatto IIIA collocation), **`pdepe`** (1D parabolic/elliptic finite-volume PDE solver), `fzero` (Brent), `integral` (adaptive Simpson), `fminsearch` (Nelder-Mead)
- **Built-In Scientific Toolboxes (Pure Zero-Dep Rust)**
  - **Statistics**: `std`, `var`, `median`, `prctile`, `quantile`, `cov`, `corrcoef`, `erf`, `erfc`, `normpdf`, `normcdf`, `norminv`, `ttest`, `kmeans` (k-means++), `pca`
  - **Signal Processing**: `fft`/`ifft`, `fft2`/`ifft2`, `fftshift`/`ifftshift`, `hann`, `hamming`, `blackman`, `xcorr`, `findpeaks`, `butter` (digital Butterworth IIR), `filter`, `filtfilt` (zero-phase), `pwelch` (Welch PSD), `conv`/`conv2`
  - **Optimization & Control**: `fsolve` & `lsqnonlin` (Levenberg-Marquardt), `linprog` (two-phase Simplex), `tf`, `step`, `bode`, `logspace`
- **Strings, Regex, Filesystem & Pure-Rust Media / File I/O**
  - Pure-Rust `regexp`, `regexpi`, `regexprep`, `jsondecode`/`jsonencode`, set operations (`unique`, `ismember`, `setdiff`, `intersect`, `union`)
  - VFS/OS Filesystem utilities: `fullfile`, `fileparts`, `isfile`, `isfolder`, `dir`, `ls`, `pwd`, `cd`, `mkdir`, `delete`, `copyfile`, `movefile`
  - File I/O (`fopen`, `fclose`, `fprintf(fid, ...)`, `fgetl`, `fgets`, `fread`, `fwrite`, `feof`, `ftell`, `frewind`, `textscan`) + Level 5 `.mat` reader/writer with built-in RFC 1950/1951 zlib/DEFLATE `miCOMPRESSED` support
  - **Zero-dependency Media I/O**: `imread`/`imwrite` (**PNG** & **BMP** returning `uint8` RGB/grayscale arrays), `audioread`/`audiowrite` (**16-bit PCM WAV**)
- **Pure-Rust 2D & 3D SVG Plotting & Handle Graphics**
  - `plot`, `semilogx`, `semilogy`, `loglog`, `scatter`, `bar`, `stem`, `histogram`, `imagesc`, `contour`, `quiver`, `subplot`, `legend`, `xlim`/`ylim`/`zlim`, `hold`, `grid`
  - **Handle graphics**: `h = plot(..., LineWidth=2.5, Color=...)`, `set`/`get`, `gcf`, `gca`
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
# Rebuild the size-optimized zero-import WebAssembly bundle
cargo build -p matro_wasm --target wasm32-unknown-unknown --profile wasm-release
cp target/wasm32-unknown-unknown/wasm-release/matro_wasm.wasm web/

# Serve locally and open http://localhost:8080
python3 -m http.server 8080 -d web
```

### Runnable Examples
```bash
./target/release/matro examples/signal_butter_bode.m --svg signal.svg
./target/release/matro examples/stats_pca_kmeans.m
./target/release/matro examples/control_step_bode.m --svg step.svg
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

## Benchmarks & Footprint

Measured with `./target/release/matro --bench` (Apple Silicon, `--release`, zero external dependencies):

| Benchmark | Time |
|---|---:|
| Zero-alloc range loop (100,000 iters, unboxed scalar fast-path) | `10.78 ms` |
| Blocked dense `matmul` (128×128) | `3.70 ms` |
| Dense linear solve `A \ b` (80×80) | `1.15 ms` |
| Radix-2 `fft` (4,096 pts) | `0.99 ms` |
| Sparse Poisson `pcg` (400×400) | `1.06 ms` |
| Stiff `ode15s` (Van der Pol $\mu=50$, $t \in [0,120]$) | `28.58 ms` |

- `web/matro_wasm.wasm` (`--profile wasm-release`): **~1.27 MB** raw (**`380 KB` gzipped**), **0 Wasm imports**
- `target/wasm32-wasip1/wasm-release/matro.wasm`: **~1.2 MB**

## License

MIT — see [LICENSE](LICENSE).
