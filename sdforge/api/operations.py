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

    def _base_to_glsl(self, ctx: GLSLContext, profile_mode: bool) -> str:
        ctx.dependencies.update(self.glsl_dependencies)

        if profile_mode:
            child_vars = [c.to_profile_glsl(ctx) for c in self.children]
        else:
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

    def to_glsl(self, ctx: GLSLContext) -> str:
        return self._base_to_glsl(ctx, profile_mode=False)

    def to_profile_glsl(self, ctx: GLSLContext) -> str:
        return self._base_to_glsl(ctx, profile_mode=True)

    def _make_callable(self, child_callables):
        if isinstance(self.blend, (str, Param)): raise TypeError("Cannot save mesh...")
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

    def to_callable(self):
        return self._make_callable([c.to_callable() for c in self.children])

    def to_profile_callable(self):
        return self._make_callable([c.to_profile_callable() for c in self.children])

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

    def _base_to_glsl(self, ctx: GLSLContext, profile_mode: bool) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        if profile_mode:
            child_vars = [c.to_profile_glsl(ctx) for c in self.children]
        else:
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

    def to_glsl(self, ctx: GLSLContext) -> str: return self._base_to_glsl(ctx, False)
    def to_profile_glsl(self, ctx: GLSLContext) -> str: return self._base_to_glsl(ctx, True)

    def _make_callable(self, child_callables):
        if isinstance(self.blend, (str, Param)): raise TypeError("Cannot save mesh...")
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

    def to_callable(self): return self._make_callable([c.to_callable() for c in self.children])
    def to_profile_callable(self): return self._make_callable([c.to_profile_callable() for c in self.children])


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

    def _base_to_glsl(self, ctx: GLSLContext, profile_mode: bool) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        if profile_mode:
            a_var, b_var = self.a.to_profile_glsl(ctx), self.b.to_profile_glsl(ctx)
        else:
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

    def to_glsl(self, ctx: GLSLContext) -> str: return self._base_to_glsl(ctx, False)
    def to_profile_glsl(self, ctx: GLSLContext) -> str: return self._base_to_glsl(ctx, True)

    def _make_callable(self, a_call, b_call):
        if isinstance(self.blend, (str, Param)): raise TypeError("Cannot save mesh...")
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

    def to_callable(self): return self._make_callable(self.a.to_callable(), self.b.to_callable())
    def to_profile_callable(self): return self._make_callable(self.a.to_profile_callable(), self.b.to_profile_callable())

    def _collect_materials(self, materials: list):
        self.a._collect_materials(materials)
        self.b._collect_materials(materials)

class Morph(SDFNode):
    """
    Internal node representing the linear interpolation (morph) between two SDF objects.
    """
    glsl_dependencies = {"operations"}

    def __init__(self, a: SDFNode, b: SDFNode, factor: float = 0.5):
        super().__init__()
        self.a = a
        self.b = b
        self.factor = factor

    def _base_to_glsl(self, ctx: GLSLContext, profile_mode: bool) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        if profile_mode:
            a_var, b_var = self.a.to_profile_glsl(ctx), self.b.to_profile_glsl(ctx)
        else:
            a_var, b_var = self.a.to_glsl(ctx), self.b.to_glsl(ctx)

        result_expr = f"opMorph({a_var}, {b_var}, {_glsl_format(self.factor)})"
        return ctx.new_variable('vec4', result_expr)

    def to_glsl(self, ctx: GLSLContext) -> str: 
        return self._base_to_glsl(ctx, False)

    def to_profile_glsl(self, ctx: GLSLContext) -> str: 
        return self._base_to_glsl(ctx, True)

    def _make_callable(self, a_call, b_call):
        if isinstance(self.factor, (str, Param)):
             raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")

        t = np.clip(self.factor, 0.0, 1.0)

        def _callable(p):
            d1 = a_call(p)
            d2 = b_call(p)
            return (1.0 - t) * d1 + t * d2
        return _callable

    def to_callable(self):
        return self._make_callable(self.a.to_callable(), self.b.to_callable())

    def to_profile_callable(self):
        return self._make_callable(self.a.to_profile_callable(), self.b.to_profile_callable())

    def _collect_materials(self, materials: list):
        self.a._collect_materials(materials)
        self.b._collect_materials(materials)