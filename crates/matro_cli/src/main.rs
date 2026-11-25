use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::process;
use std::time::Instant;

use matro_bytecode::compile;
use matro_loader::OsVfs;
use matro_vm::Vm;

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn print_usage() {
    eprintln!(
        "matro {VERSION} — pure-Rust MATLAB/Octave bytecode VM\n\
         \n\
         Usage:\n\
           matro                               Start interactive REPL\n\
           matro <file.m> [options]            Execute a MATLAB/Octave script\n\
           matro -e <code> [options]           Evaluate inline code\n\
           matro --check <file.m>              Parse & compile check without executing\n\
           matro --disasm <code>               Print compiled bytecode disassembly\n\
           matro --bench                       Run built-in VM & numerical benchmarks\n\
           matro --version                     Print version\n\
           matro --help                        Show this help message\n\
         \n\
         Options:\n\
           --svg <out.svg>                     Export generated figure as SVG\n\
           --export-mat <out.mat>              Save final workspace variables to .mat v5"
    );
}

fn new_cli_vm() -> Vm {
    Vm::with_vfs(Box::new(OsVfs::new()))
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        run_repl();
        return;
    }

    let mut svg_out: Option<String> = None;
    let mut mat_out: Option<String> = None;
    let mut positional: Vec<String> = Vec::new();
    let mut eval_mode = false;
    let mut disasm_mode = false;
    let mut check_mode = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_usage();
                return;
            }
            "-V" | "--version" => {
                println!("matro {VERSION}");
                return;
            }
            "--bench" => {
                run_benchmarks();
                return;
            }
            "-e" | "--eval" => eval_mode = true,
            "--disasm" => disasm_mode = true,
            "--check" => check_mode = true,
            "--svg" => {
                i += 1;
                let Some(path) = args.get(i) else {
                    eprintln!("Error: expected output file path after --svg");
                    process::exit(1);
                };
                svg_out = Some(path.clone());
            }
            "--export-mat" => {
                i += 1;
                let Some(path) = args.get(i) else {
                    eprintln!("Error: expected output file path after --export-mat");
                    process::exit(1);
                };
                mat_out = Some(path.clone());
            }
            other => positional.push(other.to_string()),
        }
        i += 1;
    }

    if disasm_mode {
        let Some(code) = positional.first() else {
            eprintln!("Error: expected code string for --disasm");
            process::exit(1);
        };
        match compile(code) {
            Ok(chunk) => print!("{chunk}"),
            Err(e) => {
                eprintln!("{e}");
                process::exit(1);
            }
        }
        return;
    }

    if check_mode {
        let Some(path) = positional.first() else {
            eprintln!("Error: expected file path after --check");
            process::exit(1);
        };
        let src = match fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error reading '{path}': {e}");
                process::exit(1);
            }
        };
        match compile(&src) {
            Ok(chunk) => {
                println!(
                    "OK: '{path}' ({} opcodes, {} functions, {} classes)",
                    chunk.code.len(),
                    chunk.functions.len(),
                    chunk.classes.len()
                );
            }
            Err(e) => {
                eprintln!("{e}");
                process::exit(1);
            }
        }
        return;
    }

    let (src, script_path, script_dir) = if eval_mode {
        let Some(code) = positional.first() else {
            eprintln!("Error: expected code string after -e");
            process::exit(1);
        };
        (code.clone(), None, None)
    } else {
        let Some(path) = positional.first() else {
            run_repl();
            return;
        };
        match fs::read_to_string(path) {
            Ok(s) => {
                let dir = std::path::Path::new(path)
                    .parent()
                    .and_then(|p| p.to_str())
                    .filter(|s| !s.is_empty())
                    .map(String::from);
                (s, Some(path.clone()), dir)
            }
            Err(e) => {
                eprintln!("Error reading '{path}': {e}");
                process::exit(1);
            }
        }
    };

    run_source(
        &src,
        script_path.as_deref(),
        script_dir.as_deref(),
        svg_out.as_deref(),
        mat_out.as_deref(),
    );
}

fn run_source(
    src: &str,
    script_path: Option<&str>,
    script_dir: Option<&str>,
    svg_out: Option<&str>,
    mat_out: Option<&str>,
) {
    let mut vm = new_cli_vm();
    if let Some(dir) = script_dir {
        vm.add_path(dir);
    }
    let res = if script_path.is_some() {
        vm.eval_file(script_path, src, true)
    } else {
        vm.eval(src)
    };
    match res {
        Ok(out) => {
            if !out.stdout.is_empty() {
                print!("{}", out.stdout);
            }
            if let Some(svg_path) = svg_out {
                let svg = vm.figure().to_svg(720.0, 420.0);
                if svg.is_empty() {
                    eprintln!("Warning: no plot series generated, skipping SVG export");
                } else if let Err(e) = fs::write(svg_path, svg) {
                    eprintln!("Error writing SVG '{svg_path}': {e}");
                    process::exit(1);
                }
            }
            if let Some(mat_path) = mat_out {
                let mut pairs: Vec<(&str, &matro_types::Value)> = vm
                    .globals()
                    .iter()
                    .map(|(k, v)| (k.as_str(), v))
                    .collect();
                pairs.sort_by_key(|(k, _)| *k);
                match matro_loader::matfile::write_mat_v5(&pairs) {
                    Ok(bytes) => {
                        if let Err(e) = fs::write(mat_path, bytes) {
                            eprintln!("Error writing MAT '{mat_path}': {e}");
                            process::exit(1);
                        }
                    }
                    Err(e) => {
                        eprintln!("Error encoding MAT '{mat_path}': {e}");
                        process::exit(1);
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("{e}");
            process::exit(1);
        }
    }
}

fn run_benchmarks() {
    let cases: &[(&str, &str)] = &[
        (
            "Zero-alloc range loop (100k iters)",
            "function s = f()\n s = 0;\n for k = 1:100000\n  s = s + k;\n end\nend\nr = f();",
        ),
        (
            "Blocked dense matmul (128x128)",
            "rng(1); A = randn(128, 128); B = randn(128, 128); C = A * B;",
        ),
        (
            "Dense linear solve A\\b (80x80)",
            "rng(2); A = randn(80, 80) + 5*eye(80); b = ones(80, 1); x = A \\ b;",
        ),
        (
            "Radix-2 FFT (4096 pts)",
            "x = sin(linspace(0, 40*pi, 4096)); X = fft(x);",
        ),
        (
            "Sparse Poisson PCG (400x400)",
            "n = 400; i = [1:n, 1:n-1, 2:n]; j = [1:n, 2:n, 1:n-1]; v = [2*ones(1,n), -ones(1,n-1), -ones(1,n-1)]; S = sparse(i, j, v, n, n); x = pcg(S, ones(n, 1), 1e-8, 1000);",
        ),
        (
            "Stiff ode15s (Van der Pol mu=50)",
            "mu = 50; [t, y] = ode15s(@(t, y) [y(2); mu*(1-y(1)^2)*y(2) - y(1)], [0, 120], [2; 0]);",
        ),
    ];

    println!("Matro {VERSION} Benchmark Suite");
    println!("{:<38} {:>10}", "Benchmark", "Time");
    println!("{}", "-".repeat(50));
    for (label, code) in cases {
        let mut vm = Vm::new();
        let t0 = Instant::now();
        if let Err(e) = vm.eval(code) {
            eprintln!("Benchmark '{label}' failed: {e}");
            process::exit(1);
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!("{label:<38} {:>8.2} ms", ms);
    }
}

fn run_repl() {
    println!("Matro 0.1.0 — pure-Rust MATLAB/Octave VM (type 'exit' to leave, 'whos' for vars)");
    let mut vm = new_cli_vm();
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let mut buffer = String::new();

    loop {
        let prompt = if buffer.is_empty() { "matro>> " } else { "     .. " };
        print!("{prompt}");
        let _ = io::stdout().flush();

        let Some(Ok(line)) = lines.next() else {
            println!();
            break;
        };

        let trimmed = line.trim();
        if buffer.is_empty() {
            match trimmed {
                "" => continue,
                "exit" | "quit" => break,
                "clear" => {
                    vm.clear();
                    continue;
                }
                "whos" => {
                    let mut names: Vec<_> = vm.globals().keys().collect();
                    names.sort();
                    for name in names {
                        let val = &vm.globals()[name];
                        println!(
                            "  {name:<16} {:<10} {}",
                            val.dims_str(),
                            val.type_name()
                        );
                    }
                    continue;
                }
                _ => {}
            }
        }

        buffer.push_str(&line);
        buffer.push('\n');

        if block_depth(&buffer) > 0 {
            continue;
        }

        let code = std::mem::take(&mut buffer);
        match vm.eval(&code) {
            Ok(out) => {
                if !out.stdout.is_empty() {
                    print!("{}", out.stdout);
                }
            }
            Err(e) => eprintln!("{e}"),
        }
    }
}

fn block_depth(src: &str) -> isize {
    let mut depth: isize = 0;
    for line in src.lines() {
        let code = line.split(['%', '#']).next().unwrap_or("");
        for word in code.split(|c: char| !c.is_ascii_alphanumeric() && c != '_') {
            match word {
                "if" | "for" | "while" | "function" | "switch" | "try" | "classdef"
                | "properties" | "methods" => depth += 1,
                "end" | "endif" | "endfor" | "endwhile" | "endfunction" | "endswitch"
                | "end_try_catch" | "endclassdef" | "endproperties" | "endmethods" => depth -= 1,
                _ => {}
            }
        }
    }
    depth
}
