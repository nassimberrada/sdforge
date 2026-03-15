import uuid
from ..utils.helpers import _combine_expr, Expr

class Param:
    """
    An interactive, real-time parameter for an SDF model.
    Can be used to bind Python variables to GLSL uniforms for animations.
    """
    def __init__(self, name: str, default: float, min_val: float = None, max_val: float = None):
        """
        Initializes an interactive parameter.

        Args:
            name (str): The display name for UI sliders.
            default (float): The initial value of the parameter.
            min_val (float, optional): The minimum value for UI sliders.
            max_val (float, optional): The maximum value for UI sliders.
        
        Example:
            >>> from sdforge import Param, box
            >>> # Create a parameter to control corner rounding
            >>> p_radius = Param("Corner Radius", 0.1, 0.0, 0.5)
            >>> # Use the parameter just like a number
            >>> scene = box(size=1.5, radius=p_radius)
            >>> # When rendered, a slider for "Corner Radius" will appear.
            >>> scene.render()
        """
        self.value = default
        self.name = name or f"param_{uuid.uuid4().hex[:6]}"
        self.min_val = min_val
        self.max_val = max_val
        
        sanitized_name = ''.join(c if c.isalnum() else '_' for c in self.name)
        self.uniform_name = f"u_param_{sanitized_name}_{uuid.uuid4().hex[:6]}"

    def __str__(self):
        """Returns the GLSL uniform name for use in shader code."""
        return self.uniform_name

    def to_glsl(self):
        """Returns the GLSL uniform name."""
        return self.uniform_name

    # --- Arithmetic Operations ---
    # These methods allow Params to be used in expressions (e.g., `p / 2`).
    # They return an Expr object that tracks the Param dependency.
    def __add__(self, other):
        return _combine_expr(self, other, '+')

    def __radd__(self, other):
        return _combine_expr(other, self, '+')

    def __sub__(self, other):
        return _combine_expr(self, other, '-')

    def __rsub__(self, other):
        return _combine_expr(other, self, '-')

    def __mul__(self, other):
        return _combine_expr(self, other, '*')

    def __rmul__(self, other):
        return _combine_expr(other, self, '*')

    def __truediv__(self, other):
        return _combine_expr(self, other, '/')

    def __rtruediv__(self, other):
        return _combine_expr(other, self, '/')
        
    def __neg__(self):
        return Expr(f"(-{self.to_glsl()})", {self})

class _TimeExpr(Expr):
    """
    A special expression representing the global elapsed animation time.
    Compiles directly to the 'u_time' GLSL uniform.
    """
    def __init__(self):
        super().__init__("u_time", set())

Time = _TimeExpr()