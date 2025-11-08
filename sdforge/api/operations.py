import numpy as np
from functools import reduce
from ..core import SDFNode, GLSLContext
from ..utils import _glsl_format
from ..params import Param

class Union(SDFNode):
    """Represents the union of multiple SDF objects."""
    glsl_dependencies = {"operations"}

    def __init__(self, children: list, k: float = 0.0):
        super().__init__()
        self.children = children
        self.k = k

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        
        child_vars = [c.to_glsl(ctx) for c in self.children]
        
        use_smooth = False
        if isinstance(self.k, (int, float)):
            if self.k > 1e-6:
                use_smooth = True
        else: # Param or string
            use_smooth = True

        if use_smooth:
            op = lambda a, b: f"sUnion({a}, {b}, {_glsl_format(self.k)})"
        else:
            op = lambda a, b: f"opU({a}, {b})"
        
        result_expr = reduce(op, child_vars)
        return ctx.new_variable('vec4', result_expr)

    def to_callable(self):
        if isinstance(self.k, (str, Param)):
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")
        
        child_callables = [c.to_callable() for c in self.children]
        k = self.k

        if k <= 1e-6:
            def _callable(points: np.ndarray) -> np.ndarray:
                return reduce(np.minimum, [c(points) for c in child_callables])
            return _callable
        else:
            def _callable_smooth(points: np.ndarray) -> np.ndarray:
                dists = [c(points) for c in child_callables]
                res = dists[0]
                for i in range(1, len(dists)):
                    d1, d2 = res, dists[i]
                    h = np.clip(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
                    res = d2 * (1.0 - h) + d1 * h - k * h * (1.0 - h)
                return res
            return _callable_smooth

class Intersection(SDFNode):
    """Represents the intersection of multiple SDF objects."""
    glsl_dependencies = {"operations"}

    def __init__(self, children: list, k: float = 0.0):
        super().__init__()
        self.children = children
        self.k = k

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        child_vars = [c.to_glsl(ctx) for c in self.children]
        
        use_smooth = False
        if isinstance(self.k, (int, float)):
            if self.k > 1e-6:
                use_smooth = True
        else: # Param or string
            use_smooth = True

        if use_smooth:
            op = lambda a, b: f"sIntersect({a}, {b}, {_glsl_format(self.k)})"
        else:
            op = lambda a, b: f"opI({a}, {b})"
        result_expr = reduce(op, child_vars)
        return ctx.new_variable('vec4', result_expr)

    def to_callable(self):
        if isinstance(self.k, (str, Param)):
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")
        
        child_callables = [c.to_callable() for c in self.children]
        k = self.k
        if k <= 1e-6:
            return lambda p: reduce(np.maximum, [c(p) for c in child_callables])
        else:
            def _callable_smooth(p):
                dists = [c(p) for c in child_callables]
                res = dists[0]
                for i in range(1, len(dists)):
                    d1, d2 = res, dists[i]
                    h = np.clip(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0)
                    res = d2 * (1.0 - h) + d1 * h + k * h * (1.0 - h)
                return res
            return _callable_smooth

class Difference(SDFNode):
    """Represents the subtraction of one SDF object from another."""
    glsl_dependencies = {"operations"}

    def __init__(self, a: SDFNode, b: SDFNode, k: float = 0.0):
        super().__init__()
        self.a, self.b, self.k = a, b, k

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        a_var, b_var = self.a.to_glsl(ctx), self.b.to_glsl(ctx)
        
        use_smooth = False
        if isinstance(self.k, (int, float)):
            if self.k > 1e-6:
                use_smooth = True
        else: # Param or string
            use_smooth = True
        
        if use_smooth:
            result_expr = f"sDifference({a_var}, {b_var}, {_glsl_format(self.k)})"
        else:
            result_expr = f"opS({a_var}, {b_var})"
        return ctx.new_variable('vec4', result_expr)

    def to_callable(self):
        if isinstance(self.k, (str, Param)):
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")
            
        a_call, b_call = self.a.to_callable(), self.b.to_callable()
        k = self.k
        if k <= 1e-6:
            return lambda p: np.maximum(a_call(p), -b_call(p))
        else:
            def _callable_smooth(p):
                d1, d2 = a_call(p), -b_call(p)
                h = np.clip(0.5 - 0.5 * (d1 - d2) / k, 0.0, 1.0)
                return d1 * (1.0 - h) + d2 * h + k * h * (1.0 - h)
            return _callable_smooth

    def _collect_materials(self, materials: list):
        self.a._collect_materials(materials)
        self.b._collect_materials(materials)