import numpy as np
from .core import SDFNode, GLSLContext
from .utils import _glsl_format
from .params import Param

class Round(SDFNode):
    glsl_dependencies = {"shaping"}
    def __init__(self, child: SDFNode, radius: float, mask: SDFNode = None, mask_falloff: float = 0.0):
        super().__init__()
        self.child = child
        self.radius = radius
        self.mask = mask
        self.mask_falloff = mask_falloff

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        child_var = self.child.to_glsl(ctx)
        r_expr = _glsl_format(self.radius)
        if self.mask:
            mask_var = self.mask.to_glsl(ctx)
            falloff_str = _glsl_format(self.mask_falloff)
            factor_expr = f"(1.0 - smoothstep(0.0, max({falloff_str}, 1e-4), {mask_var}.x))"
            r_expr = f"({r_expr} * {factor_expr})"
        result_expr = f"opRound({child_var}, {r_expr})"
        return ctx.new_variable('vec4', result_expr)
    
    def to_profile_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        child_var = self.child.to_profile_glsl(ctx)
        r_expr = _glsl_format(self.radius)
        if self.mask:
            mask_var = self.mask.to_glsl(ctx)
            falloff_str = _glsl_format(self.mask_falloff)
            factor_expr = f"(1.0 - smoothstep(0.0, max({falloff_str}, 1e-4), {mask_var}.x))"
            r_expr = f"({r_expr} * {factor_expr})"
        result_expr = f"opRound({child_var}, {r_expr})"
        return ctx.new_variable('vec4', result_expr)

class Shell(SDFNode):
    glsl_dependencies = {"shaping"}
    def __init__(self, child: SDFNode, thickness: float, mask: SDFNode = None, mask_falloff: float = 0.0):
        super().__init__()
        self.child = child
        self.thickness = thickness
        self.mask = mask
        self.mask_falloff = mask_falloff

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        child_var = self.child.to_glsl(ctx)
        th_expr = _glsl_format(self.thickness)
        if self.mask:
            mask_var = self.mask.to_glsl(ctx)
            falloff_str = _glsl_format(self.mask_falloff)
            factor_expr = f"(1.0 - smoothstep(0.0, max({falloff_str}, 1e-4), {mask_var}.x))"
            th_expr = f"({th_expr} * {factor_expr})"
        result_expr = f"opShell({child_var}, {th_expr})"
        return ctx.new_variable('vec4', result_expr)

    def to_profile_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        child_var = self.child.to_profile_glsl(ctx)
        th_expr = _glsl_format(self.thickness)
        if self.mask:
            mask_var = self.mask.to_glsl(ctx)
            falloff_str = _glsl_format(self.mask_falloff)
            factor_expr = f"(1.0 - smoothstep(0.0, max({falloff_str}, 1e-4), {mask_var}.x))"
            th_expr = f"({th_expr} * {factor_expr})"
        result_expr = f"opShell({child_var}, {th_expr})"
        return ctx.new_variable('vec4', result_expr)

class Extrude(SDFNode):
    glsl_dependencies = {"shaping"}
    def __init__(self, child: SDFNode, height: float):
        super().__init__()
        self.child = child
        self.height = height
    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        child_var = self.child.to_profile_glsl(ctx)
        result_expr = f"opExtrude({child_var}, {ctx.p}, {_glsl_format(self.height)})"
        return ctx.new_variable('vec4', result_expr)
    def to_profile_glsl(self, ctx: GLSLContext) -> str:
        return self.to_glsl(ctx)

class Revolve(SDFNode):
    def to_glsl(self, ctx: GLSLContext) -> str:
        revolved_p_xy = f"vec2(length({ctx.p}.xz), {ctx.p}.y)"
        transformed_p = ctx.new_variable('vec3', f"vec3({revolved_p_xy}, 0.0)")
        sub_ctx = ctx.with_p(transformed_p)
        child_var = self.child.to_profile_glsl(sub_ctx)
        ctx.merge_from(sub_ctx)
        return child_var
    def to_profile_glsl(self, ctx: GLSLContext) -> str:
        return self.to_glsl(ctx)