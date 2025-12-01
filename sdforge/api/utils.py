import numpy as np
import uuid

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
        return val
    if isinstance(val, (list, tuple, np.ndarray)):
        components = [_glsl_format(v) for v in np.array(val).flatten()]
        if len(components) == 2: return f"vec2({components[0]}, {components[1]})"
        if len(components) == 3: return f"vec3({components[0]}, {components[1]}, {components[2]})"
        if len(components) == 4: return f"vec4({components[0]}, {components[1]}, {components[2]}, {components[3]})"
    return f"{float(val)}"
    
def _combine_expr(a, b, op):
    """Helper to combine two values (Expr, Param, or number) into a new Expr."""
    a_params = set()
    if isinstance(a, Param): a_params.add(a)
    elif isinstance(a, Expr): a_params.update(a.params)
    
    b_params = set()
    if isinstance(b, Param): b_params.add(b)
    elif isinstance(b, Expr): b_params.update(b.params)
    
    new_params = a_params.union(b_params)
    new_glsl = f"({_glsl_format(a)} {op} {_glsl_format(b)})"
    return Expr(new_glsl, new_params)

for op in ['add', 'sub', 'mul', 'truediv']:
    op_symbol = {'add': '+', 'sub': '-', 'mul': '*', 'truediv': '/'}[op]
    def make_op(op_symbol):
        return lambda self, other: _combine_expr(self, other, op_symbol)
    setattr(Expr, f"__{op}__", make_op(op_symbol))
    def make_rop(op_symbol):
        return lambda self, other: _combine_expr(other, self, op_symbol)
    setattr(Expr, f"__r{op}__", make_rop(op_symbol))
    def _neg_expr(self):
        return Expr(f"(-{self.glsl_str})", self.params)
    setattr(Expr, "__neg__", _neg_expr)

def _smoothstep(edge0, edge1, x):
    """NumPy implementation of GLSL smoothstep."""
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

class Param:
    """
    An interactive, real-time parameter for an SDF model.
    """
    def __init__(self, name: str, default: float, min_val: float, max_val: float):
        self.name = name
        self.value = default
        self.min_val = min_val
        self.max_val = max_val
        sanitized_name = ''.join(c if c.isalnum() else '_' for c in name)
        self.uniform_name = f"u_param_{sanitized_name}_{uuid.uuid4().hex[:6]}"

    def __str__(self):
        """Returns the GLSL uniform name for use in shader code."""
        return self.uniform_name

    def to_glsl(self):
        """Returns the GLSL uniform name."""
        return self.uniform_name

    def __add__(self, other): return _combine_expr(self, other, '+')
    def __radd__(self, other): return _combine_expr(other, self, '+')
    def __sub__(self, other): return _combine_expr(self, other, '-')
    def __rsub__(self, other): return _combine_expr(other, self, '-')
    def __mul__(self, other): return _combine_expr(self, other, '*')
    def __rmul__(self, other): return _combine_expr(other, self, '*')
    def __truediv__(self, other): return _combine_expr(self, other, '/')
    def __rtruediv__(self, other): return _combine_expr(other, self, '/')
    def __neg__(self): return Expr(f"(-{self.to_glsl()})", {self})

class Debug:
    """
    Represents a debug visualization mode for the renderer.
    """
    def __init__(self, mode: str, plane: str = 'xy', slice_height: float = 0.0, view_scale: float = 4.0):
        self.mode = mode
        self.plane = plane.lower()
        self.slice_height = slice_height
        self.view_scale = view_scale