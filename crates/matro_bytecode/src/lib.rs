pub mod compiler;
pub mod op;

pub use compiler::Compiler;
pub use op::{Chunk, ClassProto, FunctionProto, OpCode};

use matro_parser::parse;
use matro_types::Result;

/// Parse and compile MATLAB/Octave source into a bytecode `Chunk`.
pub fn compile(source: &str) -> Result<Chunk> {
    let program = parse(source)?;
    Compiler::compile_program(&program)
}
