pub mod ast;
pub mod lexer;
pub mod parser;

pub use ast::*;
pub use parser::Parser;

use matro_types::Result;

/// Parse a MATLAB/Octave source string into an AST `Program`.
pub fn parse(input: &str) -> Result<Program> {
    Parser::parse_str(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_matrix_space_and_transpose() {
        let prog = parse("A = [1 2 -3; 4, 5, 6]'; B = A.';").unwrap();
        assert_eq!(prog.statements.len(), 2);
        assert!(prog.statements[0].suppressed);
    }

    #[test]
    fn parse_string_vs_transpose() {
        let prog = parse("s = 'hello'; B = A';").unwrap();
        assert_eq!(prog.statements.len(), 2);
    }

    #[test]
    fn parse_complex_closures_structs_cells_switch() {
        let prog = parse(
            r#"
            z = 3 + 4i;
            f = @(x, y) x.^2 + y;
            s.a = 1;
            c = {1, 'hi', [2 3]};
            v = c{2};
            switch v
                case {'hi', 'hello'}
                    x = f(2, 3);
                otherwise
                    x = 0;
            end
            "#,
        )
        .unwrap();
        assert_eq!(prog.statements.len(), 6);
    }

    #[test]
    fn parse_classdef() {
        let prog = parse(
            r#"
            classdef Vec2
                properties
                    x = 0
                    y = 0
                end
                methods
                    function obj = Vec2(x, y)
                        obj.x = x;
                        obj.y = y;
                    end
                    function r = norm2(obj)
                        r = sqrt(obj.x^2 + obj.y^2);
                    end
                end
            end
            v = Vec2(3, 4);
            d = v.norm2();
            "#,
        )
        .unwrap();
        assert_eq!(prog.statements.len(), 3);
    }
}
