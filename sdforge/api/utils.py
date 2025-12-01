from collections.abc import Iterable

class Expr:
    """Represents a GLSL expression that may contain Param objects."""
    def __init__(self, glsl_str: str, params=None):
        self.glsl_str = glsl_str
        self.params = params or set()

    def to_glsl(self):
        return self.glsl_str
    
    def __str__(self):
        return self.glsl_str

def _glsl_format(val):
    """Formats a Python value for injection into a GLSL string."""
    if hasattr(val, 'to_glsl'):
        return val.to_glsl()
    if isinstance(val, str):
        return val  # Assume it's a raw GLSL expression
    return f"{float(val)}"
    
def _combine_expr(a, b, op):
    """Helper to combine two values (Expr, Param, or number) into a new Expr."""
    from .params import Param # avoid circular import
    
    a_params = set()
    if isinstance(a, Param): a_params.add(a)
    elif isinstance(a, Expr): a_params.update(a.params)
    
    b_params = set()
    if isinstance(b, Param): b_params.add(b)
    elif isinstance(b, Expr): b_params.update(b.params)
    
    new_params = a_params.union(b_params)
    new_glsl = f"({_glsl_format(a)} {op} {_glsl_format(b)})"
    return Expr(new_glsl, new_params)

# Add arithmetic methods to the Expr class dynamically
for op in ['add', 'sub', 'mul', 'truediv']:
    op_symbol = {'add': '+', 'sub': '-', 'mul': '*', 'truediv': '/'}[op]
    
    # Forward operation: expr + other
    def make_op(op_symbol):
        return lambda self, other: _combine_expr(self, other, op_symbol)
    setattr(Expr, f"__{op}__", make_op(op_symbol))

    # Reverse operation: other + expr
    def make_rop(op_symbol):
        return lambda self, other: _combine_expr(other, self, op_symbol)
    setattr(Expr, f"__r{op}__", make_rop(op_symbol))

def _neg_expr(self):
    return Expr(f"(-{self.glsl_str})", self.params)
setattr(Expr, "__neg__", _neg_expr)