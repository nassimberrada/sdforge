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
    glsl_dependencies = {"shaping"}

    def __init__(self, child: SDFNode, axis=(0, 1, 0)):
        super().__init__()
        self.child = child
        self.axis = axis

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        
        if hasattr(self.axis, 'to_glsl'):
            axis_str = self.axis.to_glsl(ctx)
        else:
            axis_str = f"vec3({float(self.axis[0])}, {float(self.axis[1])}, {float(self.axis[2])})"

        axis_var = ctx.new_variable("vec3", f"normalize({axis_str})")
        
        # Evaluate side 1
        q1_var = ctx.new_variable('vec3', f"opRevolve({ctx.p}, {axis_var}, 1.0)")
        sub_ctx1 = ctx.with_p(q1_var)
        d1 = self.child.to_profile_glsl(sub_ctx1)
        ctx.merge_from(sub_ctx1)  # Fix: Merge statements back to main context
        
        # Evaluate side 2
        q2_var = ctx.new_variable('vec3', f"opRevolve({ctx.p}, {axis_var}, -1.0)")
        sub_ctx2 = ctx.with_p(q2_var)
        d2 = self.child.to_profile_glsl(sub_ctx2)
        ctx.merge_from(sub_ctx2)  # Fix: Merge statements back to main context
        
        # Fix: Inline the union (min distance, keep associated material)
        union_expr = f"({d1}.x < {d2}.x) ? {d1} : {d2}"
        return ctx.new_variable('vec4', union_expr)

    def to_profile_glsl(self, ctx: GLSLContext) -> str:
        return self.to_glsl(ctx)