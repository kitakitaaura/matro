# Matro

**Matro** is a modular, pure-Rust, **zero-external-dependency** MATLAB/Octave bytecode virtual machine targeting native CLI, WebAssembly in the browser (`wasm32-unknown-unknown`, **0 Wasm imports**, `~445 KB` gzipped), and portable WASI (`wasm32-wasip1`).

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
  matro_vm ◄──────► matro_intrinsics  (330+ built-ins: linalg, interp, solvers, 5 toolboxes, media I/O)
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
  - Closures & Full `classdef` OOP: `@(x) ...`, multi-arity & multi-output `@func` handles, **mutable outer-scope nested functions**, **`+package` namespaces & `import`**, value & `< handle` classes with **multiple inheritance (`< A & B`)**, `methods (Static)`, `properties (Constant)` & `(Dependent)` (`get.Prop`/`set.Prop`), **`enumeration`**, **`events` + `addlistener`/`notify`**, **operator overloading** (`plus`/`minus`/`mtimes`/`uminus`/`subsref`/`subsasgn`/etc.), struct arrays, cell arrays, comma-separated lists (`c{:}`)
  - Modern data types: **`datetime`**, **`duration`**, **`categorical`**, **`timetable`**, **`string` arrays** (`strlength`, `split`, `join`, `startsWith`, `endsWith`, `erase`, `strip`, `replace`, `+`), **`table`** (`T.col`, `T(r,c)`, `T{r,c}`, `height`, `width`, `summary`, `sortrows`, `join`/`innerjoin`/`outerjoin`, `groupsummary`, `head`/`tail`, `array2table`/`table2array`, `renamevars`/`removevars`, `readtable`/`writetable`), and `containers.Map` / `dictionary`
- **Linear Algebra, Sparse, Interpolation & Solvers**
  - Real & **complex** `lu`, `qr`, `chol`, `svd`, `eig` (cyclic Jacobi & QR), `inv`, `pinv`, `det`, `rank`, `cond`, `null`, `orth`, `expm`, `A \ b`
  - Sparse CSC matrices & large-scale iterative solvers: `sparse`, `speye`, `full`, `nnz`, `issparse`, `S \ b`, `pcg`, **`eigs`** (restarted Arnoldi / shift-and-invert), **`svds`** (subspace iteration)
  - Interpolation suite: **`pchip`** (Fritsch-Carlson monotone Hermite), **`spline`** (not-a-knot $C^2$ cubic spline), `mkpp`/`ppval`, **`interp1`** (`linear`/`nearest`/`previous`/`next`/`pchip`/`spline`), `interp2`, `ndgrid`, callable 1D/2D **`griddedInterpolant`**
  - ODE, BVP, PDE & Quadrature solvers: **`ode23`**, **`ode45`**, **`ode113`**, **`ode15s`** (stiff TR-BDF2), **`bvpinit`/`bvp4c`/`deval`**, **`pdepe`**, `fzero`, `fminbnd`, `integral`, **`integral2`**, `fminsearch`
- **Built-In Scientific Toolboxes (Pure Zero-Dep Rust)**
  - **Statistics & Machine Learning**: `std`, `var`, `median`, `prctile`, `quantile`, `iqr`, `mad`, `zscore`, `skewness`, `kurtosis`, `cov`, `corrcoef`, `erf`/`erfc`, `gamma`/`gammaln`/`gammainc`/`betainc`, distributions (`norm*`, `tpdf`/`tcdf`/`tinv`, `chi2pdf`/`chi2cdf`/`chi2inv`, `fpdf`/`fcdf`, `exppdf`/`expcdf`/`expinv`, `binopdf`/`binocdf`, `poisspdf`/`poisscdf`), `ttest`/`ttest2`, `anova1`, `regress`, **`fitlm`**, **`fitcknn`**, **`fitcsvm`**, **`fitcnb`**, `predict`, `kmeans`, `pca`, `pdist`/`squareform`/`linkage`/`cluster`
  - **Signal Processing**: `fft`/`ifft`, `fft2`/`ifft2`, `fftshift`/`ifftshift`, windows (`hann`, `hamming`, `blackman`, `bartlett`, `kaiser`, `flattopwin`), `xcorr`, `findpeaks`, IIR/FIR design (`butter`, **`cheby1`**, **`cheby2`**, **`fir1`**, **`freqz`**, **`grpdelay`**), `filter`, `filtfilt`, `pwelch`, **`spectrogram`/`stft`**, **`hilbert`**, **`envelope`**, `medfilt1`, **`sgolayfilt`**, `detrend`, `downsample`/`upsample`/`resample`, `conv`/`conv2`
  - **Optimization**: `fminunc` (BFGS), `fminbnd` (Brent), **`quadprog`** (QP), **`lsqlin`**, **`fmincon`** (Augmented Lagrangian + box-projected BFGS with linear & nonlinear constraints), `fsolve`, `lsqnonlin`, `linprog` (two-phase Simplex)
  - **Control Systems**: `tf`, **`ss`**, `tf2ss`/`ss2tf`, `series`/`parallel`/`feedback`, `step`, **`impulse`**, **`lsim`**, `bode`, **`nyquist`**, `pzmap`, `pole`/`zero`/`dcgain`, **`margin`**, `ctrb`/`obsv`, **`place`/`acker`**, `lyap`, **`care`**, **`lqr`**, **`lqe`** (Kalman estimator), `logspace`
  - **Image Processing**: `rgb2gray`, `im2double`/`im2uint8`, `imresize`, `imfilter`, `imgaussfilt`, `edge` (Sobel/Prewitt), `graythresh` (Otsu), `imbinarize`, `imhist`, `imadjust`
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

- `web/matro_wasm.wasm` (`--profile wasm-release`): **~1.52 MB** raw (**`445 KB` gzipped**), **0 Wasm imports**
- `target/wasm32-wasip1/wasm-release/matro.wasm`: **~1.5 MB**

## License

MIT — see [LICENSE](LICENSE).
