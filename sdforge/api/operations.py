import numpy as np
from functools import reduce
from ..core import SDFNode, GLSLContext
from ..utils import _glsl_format
from .params import Param
from .transforms import _smoothstep

class Union(SDFNode):
    """
    Internal node representing the union of multiple SDF objects.
    Supports masking for variable blend strength.
    """
    glsl_dependencies = {"operations"}

    def __init__(self, children: list, blend: float = 0.0, blend_type: str = 'smooth', mask: SDFNode = None, mask_falloff: float = 0.0):
        super().__init__()
        self.children = children
        self.blend = blend
        self.blend_type = blend_type
        self.mask = mask
        self.mask_falloff = mask_falloff

    def _base_to_glsl(self, ctx: GLSLContext, profile_mode: bool) -> str:
        ctx.dependencies.update(self.glsl_dependencies)

        if profile_mode:
            child_vars = [c.to_profile_glsl(ctx) for c in self.children]
        else:
            child_vars = [c.to_glsl(ctx) for c in self.children]

        is_blending = (isinstance(self.blend, (int, float)) and self.blend > 1e-6) or isinstance(self.blend, (str, Param))

        if is_blending:
            blend_expr = _glsl_format(self.blend)
            if self.mask:
                mask_var = self.mask.to_glsl(ctx)
                falloff_str = _glsl_format(self.mask_falloff)
                factor_expr = f"(1.0 - smoothstep(0.0, max({falloff_str}, 1e-4), {mask_var}.x))"
                blend_expr = f"({blend_expr} * {factor_expr})"

            if self.blend_type == 'linear':
                op = lambda a, b: f"cUnion({a}, {b}, {blend_expr})"
            else:
                op = lambda a, b: f"sUnion({a}, {b}, {blend_expr})"
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
        
        base_blend = self.blend
        if base_blend <= 1e-6 and not self.mask:
            def _callable(points: np.ndarray) -> np.ndarray:
                return reduce(np.minimum, [c(points) for c in child_callables])
            return _callable
        
        is_linear = self.blend_type == 'linear'
        mask_callable = self.mask.to_callable() if self.mask else None
        falloff = max(self.mask_falloff, 1e-4)

        def _callable_smooth(points: np.ndarray) -> np.ndarray:
            dists = [c(points) for c in child_callables]
            
            # Determine blend factor for each point
            blend = base_blend
            if mask_callable:
                d_mask = mask_callable(points)
                factor = 1.0 - _smoothstep(0.0, falloff, d_mask)
                blend = blend * factor
                # Avoid div by zero issues if blend becomes 0
                blend = np.maximum(blend, 1e-6)
            
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
    Supports masking for variable blend strength.
    """
    glsl_dependencies = {"operations"}

    def __init__(self, children: list, blend: float = 0.0, blend_type: str = 'smooth', mask: SDFNode = None, mask_falloff: float = 0.0):
        super().__init__()
        self.children = children
        self.blend = blend
        self.blend_type = blend_type
        self.mask = mask
        self.mask_falloff = mask_falloff

    def _base_to_glsl(self, ctx: GLSLContext, profile_mode: bool) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        if profile_mode:
            child_vars = [c.to_profile_glsl(ctx) for c in self.children]
        else:
            child_vars = [c.to_glsl(ctx) for c in self.children]

        is_blending = (isinstance(self.blend, (int, float)) and self.blend > 1e-6) or isinstance(self.blend, (str, Param))

        if is_blending:
            blend_expr = _glsl_format(self.blend)
            if self.mask:
                mask_var = self.mask.to_glsl(ctx)
                falloff_str = _glsl_format(self.mask_falloff)
                factor_expr = f"(1.0 - smoothstep(0.0, max({falloff_str}, 1e-4), {mask_var}.x))"
                blend_expr = f"({blend_expr} * {factor_expr})"

            if self.blend_type == 'linear':
                op = lambda a, b: f"cIntersect({a}, {b}, {blend_expr})"
            else:
                op = lambda a, b: f"sIntersect({a}, {b}, {blend_expr})"
        else:
            op = lambda a, b: f"opI({a}, {b})"
        result_expr = reduce(op, child_vars)
        return ctx.new_variable('vec4', result_expr)

    def to_glsl(self, ctx: GLSLContext) -> str: return self._base_to_glsl(ctx, False)
    def to_profile_glsl(self, ctx: GLSLContext) -> str: return self._base_to_glsl(ctx, True)

    def _make_callable(self, child_callables):
        if isinstance(self.blend, (str, Param)): raise TypeError("Cannot save mesh...")
        
        base_blend = self.blend
        if base_blend <= 1e-6 and not self.mask:
            return lambda p: reduce(np.maximum, [c(p) for c in child_callables])
        
        is_linear = self.blend_type == 'linear'
        mask_callable = self.mask.to_callable() if self.mask else None
        falloff = max(self.mask_falloff, 1e-4)

        def _callable_smooth(p):
            dists = [c(p) for c in child_callables]
            
            blend = base_blend
            if mask_callable:
                d_mask = mask_callable(p)
                factor = 1.0 - _smoothstep(0.0, falloff, d_mask)
                blend = blend * factor
                blend = np.maximum(blend, 1e-6)

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
    Supports masking for variable blend strength.
    """
    glsl_dependencies = {"operations"}

    def __init__(self, a: SDFNode, b: SDFNode, blend: float = 0.0, blend_type: str = 'smooth', mask: SDFNode = None, mask_falloff: float = 0.0):
        super().__init__()
        self.a = a
        self.b = b
        self.blend = blend
        self.blend_type = blend_type
        self.mask = mask
        self.mask_falloff = mask_falloff

    def _base_to_glsl(self, ctx: GLSLContext, profile_mode: bool) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        if profile_mode:
            a_var, b_var = self.a.to_profile_glsl(ctx), self.b.to_profile_glsl(ctx)
        else:
            a_var, b_var = self.a.to_glsl(ctx), self.b.to_glsl(ctx)

        is_blending = (isinstance(self.blend, (int, float)) and self.blend > 1e-6) or isinstance(self.blend, (str, Param))

        if is_blending:
            blend_expr = _glsl_format(self.blend)
            if self.mask:
                mask_var = self.mask.to_glsl(ctx)
                falloff_str = _glsl_format(self.mask_falloff)
                factor_expr = f"(1.0 - smoothstep(0.0, max({falloff_str}, 1e-4), {mask_var}.x))"
                blend_expr = f"({blend_expr} * {factor_expr})"

            if self.blend_type == 'linear':
                result_expr = f"cDifference({a_var}, {b_var}, {blend_expr})"
            else:
                result_expr = f"sDifference({a_var}, {b_var}, {blend_expr})"
        else:
            result_expr = f"opS({a_var}, {b_var})"
        return ctx.new_variable('vec4', result_expr)

    def to_glsl(self, ctx: GLSLContext) -> str: return self._base_to_glsl(ctx, False)
    def to_profile_glsl(self, ctx: GLSLContext) -> str: return self._base_to_glsl(ctx, True)

    def _make_callable(self, a_call, b_call):
        if isinstance(self.blend, (str, Param)): raise TypeError("Cannot save mesh...")
        
        base_blend = self.blend
        if base_blend <= 1e-6 and not self.mask:
            return lambda p: np.maximum(a_call(p), -b_call(p))
        
        is_linear = self.blend_type == 'linear'
        mask_callable = self.mask.to_callable() if self.mask else None
        falloff = max(self.mask_falloff, 1e-4)

        def _callable_smooth(p):
            d1, d2 = a_call(p), -b_call(p)
            
            blend = base_blend
            if mask_callable:
                d_mask = mask_callable(p)
                factor = 1.0 - _smoothstep(0.0, falloff, d_mask)
                blend = blend * factor
                blend = np.maximum(blend, 1e-6)

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
        if self.mask: self.mask._collect_materials(materials)

class Morph(SDFNode):
    """
    Internal node representing the linear interpolation (morph) between two SDF objects.
    Supports masking for variable morph factor.
    """
    glsl_dependencies = {"operations"}

    def __init__(self, a: SDFNode, b: SDFNode, factor: float = 0.5, mask: SDFNode = None, mask_falloff: float = 0.0):
        super().__init__()
        self.a = a
        self.b = b
        self.factor = factor
        self.mask = mask
        self.mask_falloff = mask_falloff

    def _base_to_glsl(self, ctx: GLSLContext, profile_mode: bool) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        if profile_mode:
            a_var, b_var = self.a.to_profile_glsl(ctx), self.b.to_profile_glsl(ctx)
        else:
            a_var, b_var = self.a.to_glsl(ctx), self.b.to_glsl(ctx)

        factor_expr = _glsl_format(self.factor)
        if self.mask:
            mask_var = self.mask.to_glsl(ctx)
            falloff_str = _glsl_format(self.mask_falloff)
            mask_factor_expr = f"(1.0 - smoothstep(0.0, max({falloff_str}, 1e-4), {mask_var}.x))"
            # Multiply user factor by mask factor? Or something else?
            # Morphing usually goes 0->1. If mask is active, we probably want to apply the factor.
            # If mask is inactive, factor should be 0 (state A)?
            factor_expr = f"({factor_expr} * {mask_factor_expr})"

        result_expr = f"opMorph({a_var}, {b_var}, {factor_expr})"
        return ctx.new_variable('vec4', result_expr)

    def to_glsl(self, ctx: GLSLContext) -> str: 
        return self._base_to_glsl(ctx, False)

    def to_profile_glsl(self, ctx: GLSLContext) -> str: 
        return self._base_to_glsl(ctx, True)

    def _make_callable(self, a_call, b_call):
        if isinstance(self.factor, (str, Param)):
             raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")

        base_t = np.clip(self.factor, 0.0, 1.0)
        mask_callable = self.mask.to_callable() if self.mask else None
        falloff = max(self.mask_falloff, 1e-4)

        def _callable(p):
            d1 = a_call(p)
            d2 = b_call(p)
            
            t = base_t
            if mask_callable:
                d_mask = mask_callable(p)
                factor = 1.0 - _smoothstep(0.0, falloff, d_mask)
                t = t * factor
                
            return (1.0 - t) * d1 + t * d2
        return _callable

    def to_callable(self):
        return self._make_callable(self.a.to_callable(), self.b.to_callable())

    def to_profile_callable(self):
        return self._make_callable(self.a.to_profile_callable(), self.b.to_profile_callable())

    def _collect_materials(self, materials: list):
        self.a._collect_materials(materials)
        self.b._collect_materials(materials)
        if self.mask: self.mask._collect_materials(materials)