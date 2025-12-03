import numpy as np
from functools import reduce
from .core import SDFNode, GLSLContext
from .utils import _glsl_format
from .params import Param

class Union(SDFNode):
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

class Intersection(SDFNode):
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

class Difference(SDFNode):
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
    
    def _collect_materials(self, materials: list):
        self.a._collect_materials(materials)
        self.b._collect_materials(materials)
        if self.mask: self.mask._collect_materials(materials)

class Morph(SDFNode):
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
            factor_expr = f"({factor_expr} * {mask_factor_expr})"

        result_expr = f"opMorph({a_var}, {b_var}, {factor_expr})"
        return ctx.new_variable('vec4', result_expr)

    def to_glsl(self, ctx: GLSLContext) -> str: 
        return self._base_to_glsl(ctx, False)

    def to_profile_glsl(self, ctx: GLSLContext) -> str: 
        return self._base_to_glsl(ctx, True)

    def _collect_materials(self, materials: list):
        self.a._collect_materials(materials)
        self.b._collect_materials(materials)
        if self.mask: self.mask._collect_materials(materials)