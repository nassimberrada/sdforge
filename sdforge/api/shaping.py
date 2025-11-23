import numpy as np
from ..core import SDFNode, GLSLContext
from ..utils import _glsl_format
from .params import Param

class Round(SDFNode):
    """
    Internal node to round the edges of a child object.
    """
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
    
    def to_profile_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        # Propagate to child's profile (e.g. rounding the corners of a 2D rectangle before extruding)
        child_var = self.child.to_profile_glsl(ctx)
        result_expr = f"opRound({child_var}, {_glsl_format(self.radius)})"
        return ctx.new_variable('vec4', result_expr)

    def to_callable(self):
        if isinstance(self.radius, (str, Param)):
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")
        child_callable = self.child.to_callable()
        return lambda p: child_callable(p) - self.radius

    def to_profile_callable(self):
        if isinstance(self.radius, (str, Param)): raise TypeError("Cannot save mesh with animated params.")
        child_func = self.child.to_profile_callable()
        return lambda p: child_func(p) - self.radius

class Shell(SDFNode):
    """
    Internal node to create a hollow shell or outline of a child object.
    """
    glsl_dependencies = {"shaping"}
    def __init__(self, child: SDFNode, thickness: float):
        super().__init__()
        self.child = child
        self.thickness = thickness
    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        child_var = self.child.to_glsl(ctx)
        result_expr = f"opShell({child_var}, {_glsl_format(self.thickness)})"
        return ctx.new_variable('vec4', result_expr)

    def to_profile_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        child_var = self.child.to_profile_glsl(ctx)
        result_expr = f"opShell({child_var}, {_glsl_format(self.thickness)})"
        return ctx.new_variable('vec4', result_expr)

    def to_callable(self):
        if isinstance(self.thickness, (str, Param)):
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")
        child_callable = self.child.to_callable()
        return lambda p: np.abs(child_callable(p)) - self.thickness

    def to_profile_callable(self):
        if isinstance(self.thickness, (str, Param)): raise TypeError("Cannot save mesh with animated params.")
        child_func = self.child.to_profile_callable()
        return lambda p: np.abs(child_func(p)) - self.thickness

class Extrude(SDFNode):
    """
    Internal node to extrude a 2D SDF shape.
    """
    glsl_dependencies = {"shaping"}
    def __init__(self, child: SDFNode, height: float):
        super().__init__()
        self.child = child
        self.height = height
    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        # Use to_profile_glsl to retrieve the infinite 2D profile
        child_var = self.child.to_profile_glsl(ctx)
        result_expr = f"opExtrude({child_var}, {ctx.p}, {_glsl_format(self.height)})"
        return ctx.new_variable('vec4', result_expr)
    
    def to_profile_glsl(self, ctx: GLSLContext) -> str:
        # If used as a profile, it returns its 3D representation (effectively chaining extrusion?)
        # Or more likely, Extrude is a terminal 3D op. 
        # Default to to_glsl behavior.
        return self.to_glsl(ctx)

    def to_callable(self):
        if isinstance(self.height, (str, Param)):
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")
        child_callable_2d = self.child.to_profile_callable()
        h = self.height
        def _callable(p_3d):
            d = child_callable_2d(p_3d)
            w = np.stack([d, np.abs(p_3d[:, 2]) - h], axis=-1)
            return np.minimum(np.maximum(w[:,0], w[:,1]), 0.0) + np.linalg.norm(np.maximum(w, 0.0), axis=-1)
        return _callable

    def to_profile_callable(self):
        return self.to_callable()

class Revolve(SDFNode):
    """
    Internal node to revolve a 2D SDF shape around the Y-axis.
    """
    def to_glsl(self, ctx: GLSLContext) -> str:
        revolved_p_xy = f"vec2(length({ctx.p}.xz), {ctx.p}.y)"
        transformed_p = ctx.new_variable('vec3', f"vec3({revolved_p_xy}, 0.0)")

        sub_ctx = ctx.with_p(transformed_p)
        child_var = self.child.to_profile_glsl(sub_ctx)

        ctx.merge_from(sub_ctx)
        return child_var

    def to_profile_glsl(self, ctx: GLSLContext) -> str:
        return self.to_glsl(ctx)

    def to_callable(self):
        child_callable_2d = self.child.to_profile_callable()
        def _callable_3d(p_3d):
            p_2d_x = np.linalg.norm(p_3d[:, [0, 2]], axis=-1)
            p_2d_y = p_3d[:, 1]
            p_2d = np.stack([p_2d_x, p_2d_y, np.zeros(p_3d.shape[0])], axis=-1)
            return child_callable_2d(p_2d)
        return _callable_3d

    def to_profile_callable(self):
        return self.to_callable()