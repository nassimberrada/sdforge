import numpy as np
import re
from ..core import SDFNode, GLSLContext
from ..utils import _glsl_format
from .params import Param

class Round(SDFNode):
    """Rounds the edges of a child object."""
    glsl_dependencies = {"shaping"}
    def __init__(self, child: SDFNode, radius: float):
        super().__init__()
        self.child = child
        self.radius = radius
    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        child_var = self.child.to_glsl(ctx)
        result_expr = f"opRound({child_var}, {_glsl_format(self.radius)})"
        return ctx.new_variable('vec4', result_expr)
    def to_callable(self):
        if isinstance(self.radius, (str, Param)):
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")
        child_callable = self.child.to_callable()
        return lambda p: child_callable(p) - self.radius

class Bevel(SDFNode):
    """Creates a shell or outline of a child object."""
    glsl_dependencies = {"shaping"}
    def __init__(self, child: SDFNode, thickness: float):
        super().__init__()
        self.child = child
        self.thickness = thickness
    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        child_var = self.child.to_glsl(ctx)
        result_expr = f"opBevel({child_var}, {_glsl_format(self.thickness)})"
        return ctx.new_variable('vec4', result_expr)
    def to_callable(self):
        if isinstance(self.thickness, (str, Param)):
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")
        child_callable = self.child.to_callable()
        return lambda p: np.abs(child_callable(p)) - self.thickness

class Extrude(SDFNode):
    """Extrudes a 2D SDF shape."""
    glsl_dependencies = {"shaping"}
    def __init__(self, child: SDFNode, height: float):
        super().__init__()
        self.child = child
        self.height = height
    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        child_var = self.child.to_glsl(ctx)
        result_expr = f"opExtrude({child_var}, {ctx.p}, {_glsl_format(self.height)})"
        return ctx.new_variable('vec4', result_expr)
    def to_callable(self):
        if isinstance(self.height, (str, Param)):
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")
        child_callable_2d = self.child.to_callable()
        h = self.height
        def _callable(p_3d):
            d = child_callable_2d(p_3d)
            w = np.stack([d, np.abs(p_3d[:, 2]) - h], axis=-1)
            return np.minimum(np.maximum(w[:,0], w[:,1]), 0.0) + np.linalg.norm(np.maximum(w, 0.0), axis=-1)
        return _callable

class Revolve(SDFNode):
    """Revolves a 2D SDF shape around the Y-axis."""
    def to_glsl(self, ctx: GLSLContext) -> str:
        # Create a new point variable for the revolved coordinate space
        revolved_p_xy = f"vec2(length({ctx.p}.xz), {ctx.p}.y)"
        
        # This is a special transform that modifies the coordinate space for its child.
        # We manually create the transformed 'p' and pass it to a sub-context.
        # Note: we need a vec3, so we add a 0.0 z-component.
        transformed_p = ctx.new_variable('vec3', f"vec3({revolved_p_xy}, 0.0)")
        
        sub_ctx = ctx.with_p(transformed_p)
        child_var = self.child.to_glsl(sub_ctx)
        
        ctx.merge_from(sub_ctx)
        return child_var

    def to_callable(self):
        child_callable_2d = self.child.to_callable()
        def _callable_3d(p_3d):
            # Create a 2D point (as a vec3 for the callable) from the 3D point
            p_2d_x = np.linalg.norm(p_3d[:, [0, 2]], axis=-1)
            p_2d_y = p_3d[:, 1]
            p_2d = np.stack([p_2d_x, p_2d_y, np.zeros(p_3d.shape[0])], axis=-1)
            return child_callable_2d(p_2d)
        return _callable_3d