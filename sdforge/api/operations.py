import numpy as np
from functools import reduce
from ..core import SDFNode, GLSLContext
from ..utils import _glsl_format
from .params import Param

class Union(SDFNode):
    """
    Internal node representing the union of multiple SDF objects.
    """
    glsl_dependencies = {"operations"}

    def __init__(self, children: list, blend: float = 0.0, blend_type: str = 'smooth'):
        super().__init__()
        self.children = children
        self.blend = blend
        self.blend_type = blend_type

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)

        child_vars = [c.to_glsl(ctx) for c in self.children]

        is_blending = (isinstance(self.blend, (int, float)) and self.blend > 1e-6) or isinstance(self.blend, (str, Param))

        if is_blending:
            if self.blend_type == 'linear':
                op = lambda a, b: f"cUnion({a}, {b}, {_glsl_format(self.blend)})"
            else:
                op = lambda a, b: f"sUnion({a}, {b}, {_glsl_format(self.blend)})"
        else:
            op = lambda a, b: f"opU({a}, {b})"

        result_expr = reduce(op, child_vars)
        return ctx.new_variable('vec4', result_expr)

    def to_callable(self):
        if isinstance(self.blend, (str, Param)):
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")

        child_callables = [c.to_callable() for c in self.children]
        blend = self.blend

        if blend <= 1e-6:
            def _callable(points: np.ndarray) -> np.ndarray:
                return reduce(np.minimum, [c(points) for c in child_callables])
            return _callable
        else:
            is_linear = self.blend_type == 'linear'

            def _callable_smooth(points: np.ndarray) -> np.ndarray:
                dists = [c(points) for c in child_callables]
                res = dists[0]
                for i in range(1, len(dists)):
                    d1, d2 = res, dists[i]
                    h = np.clip(0.5 + 0.5 * (d2 - d1) / blend, 0.0, 1.0)
                    res = d2 * (1.0 - h) + d1 * h
                    if not is_linear:
                        res -= blend * h * (1.0 - h)
                return res
            return _callable_smooth

class Intersection(SDFNode):
    """
    Internal node representing the intersection of multiple SDF objects.
    """
    glsl_dependencies = {"operations"}

    def __init__(self, children: list, blend: float = 0.0, blend_type: str = 'smooth'):
        super().__init__()
        self.children = children
        self.blend = blend
        self.blend_type = blend_type

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        child_vars = [c.to_glsl(ctx) for c in self.children]

        is_blending = (isinstance(self.blend, (int, float)) and self.blend > 1e-6) or isinstance(self.blend, (str, Param))

        if is_blending:
            if self.blend_type == 'linear':
                op = lambda a, b: f"cIntersect({a}, {b}, {_glsl_format(self.blend)})"
            else:
                op = lambda a, b: f"sIntersect({a}, {b}, {_glsl_format(self.blend)})"
        else:
            op = lambda a, b: f"opI({a}, {b})"
        result_expr = reduce(op, child_vars)
        return ctx.new_variable('vec4', result_expr)

    def to_callable(self):
        if isinstance(self.blend, (str, Param)):
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")

        child_callables = [c.to_callable() for c in self.children]
        blend = self.blend

        if blend <= 1e-6:
            return lambda p: reduce(np.maximum, [c(p) for c in child_callables])
        else:
            is_linear = self.blend_type == 'linear'

            def _callable_smooth(p):
                dists = [c(p) for c in child_callables]
                res = dists[0]
                for i in range(1, len(dists)):
                    d1, d2 = res, dists[i]
                    h = np.clip(0.5 - 0.5 * (d2 - d1) / blend, 0.0, 1.0)
                    res = d2 * (1.0 - h) + d1 * h
                    if not is_linear:
                         res += blend * h * (1.0 - h)
                return res
            return _callable_smooth

class Difference(SDFNode):
    """
    Internal node representing the subtraction of one SDF object from another.
    """
    glsl_dependencies = {"operations"}

    def __init__(self, a: SDFNode, b: SDFNode, blend: float = 0.0, blend_type: str = 'smooth'):
        super().__init__()
        self.a = a
        self.b = b
        self.blend = blend
        self.blend_type = blend_type

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        a_var, b_var = self.a.to_glsl(ctx), self.b.to_glsl(ctx)

        is_blending = (isinstance(self.blend, (int, float)) and self.blend > 1e-6) or isinstance(self.blend, (str, Param))

        if is_blending:
            if self.blend_type == 'linear':
                result_expr = f"cDifference({a_var}, {b_var}, {_glsl_format(self.blend)})"
            else:
                result_expr = f"sDifference({a_var}, {b_var}, {_glsl_format(self.blend)})"
        else:
            result_expr = f"opS({a_var}, {b_var})"
        return ctx.new_variable('vec4', result_expr)

    def to_callable(self):
        if isinstance(self.blend, (str, Param)):
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")

        a_call, b_call = self.a.to_callable(), self.b.to_callable()
        blend = self.blend

        if blend <= 1e-6:
            return lambda p: np.maximum(a_call(p), -b_call(p))
        else:
            is_linear = self.blend_type == 'linear'

            def _callable_smooth(p):
                d1, d2 = a_call(p), -b_call(p)
                h = np.clip(0.5 - 0.5 * (d1 - d2) / blend, 0.0, 1.0)
                res = d1 * (1.0 - h) + d2 * h
                if not is_linear:
                    res += blend * h * (1.0 - h)
                return res
            return _callable_smooth

    def _collect_materials(self, materials: list):
        self.a._collect_materials(materials)
        self.b._collect_materials(materials)