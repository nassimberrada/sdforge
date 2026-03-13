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

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Intercepts NumPy math functions (like np.sin) and converts them
        into GLSL expression strings instead of crashing!
        """
        if method != '__call__':
            return NotImplemented

        # Get the name of the NumPy function (e.g., 'sin', 'cos', 'exp')
        func_name = ufunc.__name__
        
        # We only support 1-argument functions right now (sin, cos, abs, etc.)
        if len(inputs) == 1:
            obj = inputs[0]
            if hasattr(obj, 'to_glsl'):
                # Wrap the GLSL string in the math function name
                new_glsl = f"{func_name}({obj.to_glsl()})"
                return Expr(new_glsl, obj.params)
                
        # If it's a multi-argument function or something weird, fail gracefully
        return NotImplemented

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