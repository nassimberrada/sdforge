import numpy as np
from .core import SDFNode, GLSLContext
from .utils import _glsl_format
from .params import Param

class Displace(SDFNode):
    glsl_dependencies = {"shaping"}
    def __init__(self, child: SDFNode, displacement_glsl: str, mask: SDFNode = None, mask_falloff: float = 0.0):
        super().__init__()
        self.child = child
        self.displacement_glsl = displacement_glsl
        self.mask = mask
        self.mask_falloff = mask_falloff

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        child_var = self.child.to_glsl(ctx)
        
        disp_expr = self.displacement_glsl.replace('p', ctx.p)
        if self.mask:
            mask_var = self.mask.to_glsl(ctx)
            falloff_str = _glsl_format(self.mask_falloff)
            factor_expr = f"(1.0 - smoothstep(0.0, max({falloff_str}, 1e-4), {mask_var}.x))"
            disp_expr = f"({disp_expr}) * {factor_expr}"
            
        result_expr = f"opDisplace({child_var}, {disp_expr})"
        return ctx.new_variable('vec4', result_expr)

class DisplaceByNoise(Displace):
    glsl_dependencies = {"shaping", "noise"}
    def __init__(self, child: SDFNode, scale: float = 10.0, strength: float = 0.1, mask: SDFNode = None, mask_falloff: float = 0.0):
        glsl_expr = f"snoise(p * {_glsl_format(scale)}) * {_glsl_format(strength)}"
        super().__init__(child, glsl_expr, mask, mask_falloff)
        self.scale = scale
        self.strength = strength