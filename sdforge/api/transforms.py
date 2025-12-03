import numpy as np
from .core import SDFNode, GLSLContext, X, Y, Z
from .utils import _glsl_format
from .params import Param

class _Transform(SDFNode):
    """Base class for transforms to reduce boilerplate."""
    def __init__(self, child: SDFNode, mask: SDFNode = None, mask_falloff: float = 0.0):
        super().__init__()
        self.child = child
        self.mask = mask
        self.mask_falloff = mask_falloff

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        raw_transform_expr = self._get_transform_glsl_expr(ctx.p)
        if self.mask:
            mask_var = self.mask.to_glsl(ctx)
            falloff_str = _glsl_format(self.mask_falloff)
            factor_expr = f"(1.0 - smoothstep(0.0, max({falloff_str}, 1e-4), {mask_var}.x))"
            final_p_expr = f"mix({ctx.p}, {raw_transform_expr}, {factor_expr})"
            transformed_p = ctx.new_variable('vec3', final_p_expr)
        else:
            transformed_p = ctx.new_variable('vec3', raw_transform_expr)
        sub_ctx = ctx.with_p(transformed_p)
        child_var = self.child.to_glsl(sub_ctx)
        ctx.merge_from(sub_ctx)
        return child_var

    def to_profile_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        raw_transform_expr = self._get_transform_glsl_expr(ctx.p)
        if self.mask:
            mask_var = self.mask.to_glsl(ctx)
            falloff_str = _glsl_format(self.mask_falloff)
            factor_expr = f"(1.0 - smoothstep(0.0, max({falloff_str}, 1e-4), {mask_var}.x))"
            final_p_expr = f"mix({ctx.p}, {raw_transform_expr}, {factor_expr})"
            transformed_p = ctx.new_variable('vec3', final_p_expr)
        else:
            transformed_p = ctx.new_variable('vec3', raw_transform_expr)
        sub_ctx = ctx.with_p(transformed_p)
        child_var = self.child.to_profile_glsl(sub_ctx)
        ctx.merge_from(sub_ctx)
        return child_var

    def _get_transform_glsl_expr(self, p_expr: str) -> str:
        raise NotImplementedError

class Translate(_Transform):
    glsl_dependencies = {"transforms"}
    def __init__(self, child: SDFNode, offset: tuple, mask=None, mask_falloff=0.0):
        super().__init__(child, mask, mask_falloff)
        self.offset = np.array(offset)
    def _get_transform_glsl_expr(self, p_expr: str) -> str:
        o = self.offset
        offset_str = f"vec3({_glsl_format(o[0])}, {_glsl_format(o[1])}, {_glsl_format(o[2])})"
        return f"opTranslate({p_expr}, {offset_str})"

class Scale(SDFNode):
    glsl_dependencies = {"transforms"}
    def __init__(self, child: SDFNode, factor, mask=None, mask_falloff=0.0):
        super().__init__()
        self.child = child
        self.mask = mask
        self.mask_falloff = mask_falloff
        if isinstance(factor, (int, float, str, Param)):
            self.factor = np.array([factor, factor, factor])
        else:
            self.factor = np.array(factor)

    def _base_to_glsl(self, ctx: GLSLContext, profile_mode: bool) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        f = self.factor
        factor_str = f"vec3({_glsl_format(f[0])}, {_glsl_format(f[1])}, {_glsl_format(f[2])})"
        if self.mask:
            mask_var = self.mask.to_glsl(ctx)
            falloff_str = _glsl_format(self.mask_falloff)
            factor_expr = f"(1.0 - smoothstep(0.0, max({falloff_str}, 1e-4), {mask_var}.x))"
            raw_transformed_p = f"opScale({ctx.p}, {factor_str})"
            mixed_p = f"mix({ctx.p}, {raw_transformed_p}, {factor_expr})"
            transformed_p = ctx.new_variable('vec3', mixed_p)
            if isinstance(self.factor[0], (int, float)) and isinstance(self.factor[1], (int, float)) and isinstance(self.factor[2], (int, float)):
                avg_scale = np.mean(self.factor)
                scale_val = _glsl_format(avg_scale)
            else:
                scale_val = f"({_glsl_format(self.factor[0])} + {_glsl_format(self.factor[1])} + {_glsl_format(self.factor[2])}) / 3.0"
            correction_expr = f"mix(1.0, {scale_val}, {factor_expr})"
        else:
            transformed_p = ctx.new_variable('vec3', f"opScale({ctx.p}, {factor_str})")
            if isinstance(self.factor[0], (int, float)) and isinstance(self.factor[1], (int, float)) and isinstance(self.factor[2], (int, float)):
                correction_expr = _glsl_format(np.mean(self.factor))
            else:
                correction_expr = f"({_glsl_format(self.factor[0])} + {_glsl_format(self.factor[1])} + {_glsl_format(self.factor[2])}) / 3.0"
        sub_ctx = ctx.with_p(transformed_p)
        if profile_mode: child_var = self.child.to_profile_glsl(sub_ctx)
        else: child_var = self.child.to_glsl(sub_ctx)
        ctx.merge_from(sub_ctx)
        result_expr = f"vec4({child_var}.x * ({correction_expr}), {child_var}.yzw)"
        return ctx.new_variable('vec4', result_expr)

    def to_glsl(self, ctx: GLSLContext) -> str: return self._base_to_glsl(ctx, profile_mode=False)
    def to_profile_glsl(self, ctx: GLSLContext) -> str: return self._base_to_glsl(ctx, profile_mode=True)

class Rotate(_Transform):
    glsl_dependencies = {"transforms"}
    def __init__(self, child: SDFNode, axis: tuple, angle: float, mask=None, mask_falloff=0.0):
        super().__init__(child, mask, mask_falloff)
        self.axis = np.array(axis, dtype=float)
        if np.linalg.norm(self.axis) == 0: raise ValueError("Rotation axis cannot be zero vector")
        self.axis /= np.linalg.norm(self.axis)
        self.angle = angle
    def _get_transform_glsl_expr(self, p_expr: str) -> str:
        if np.allclose(self.axis, X): func = "opRotateX"
        elif np.allclose(self.axis, Y): func = "opRotateY"
        elif np.allclose(self.axis, Z): func = "opRotateZ"
        else:
            ax = self.axis
            axis_str = f"vec3({_glsl_format(ax[0])}, {_glsl_format(ax[1])}, {_glsl_format(ax[2])})"
            return f"opRotateAxis({p_expr}, {axis_str}, {_glsl_format(self.angle)})"
        return f"{func}({p_expr}, {_glsl_format(self.angle)})"

class Orient(_Transform):
    glsl_dependencies = set()
    def __init__(self, child: SDFNode, axis: tuple):
        super().__init__(child)
        self.axis = axis
    def _get_transform_glsl_expr(self, p_expr: str) -> str:
        if np.allclose(self.axis, X): return f"{p_expr}.zyx"
        if np.allclose(self.axis, Y): return f"{p_expr}.xzy"
        return p_expr 

class Twist(_Transform):
    glsl_dependencies = {"transforms"}
    def __init__(self, child: SDFNode, strength: float, mask=None, mask_falloff=0.0):
        super().__init__(child, mask, mask_falloff)
        self.strength = strength
    def _get_transform_glsl_expr(self, p_expr: str) -> str:
        return f"opTwist({p_expr}, {_glsl_format(self.strength)})"

class Bend(_Transform):
    glsl_dependencies = {"transforms"}
    def __init__(self, child: SDFNode, axis: np.ndarray, curvature: float, mask=None, mask_falloff=0.0):
        super().__init__(child, mask, mask_falloff)
        self.axis = axis
        self.curvature = curvature
    def _get_transform_glsl_expr(self, p_expr: str) -> str:
        if np.allclose(self.axis, X): func = "opBendX"
        elif np.allclose(self.axis, Y): func = "opBendY"
        else: func = "opBendZ"
        return f"{func}({p_expr}, {_glsl_format(self.curvature)})"

class Warp(_Transform):
    glsl_dependencies = {"transforms", "noise"}
    def __init__(self, child: SDFNode, frequency: float, strength: float, mask=None, mask_falloff=0.0):
        super().__init__(child, mask, mask_falloff)
        self.frequency = frequency
        self.strength = strength
    def _get_transform_glsl_expr(self, p_expr: str) -> str:
        return f"opWarp({p_expr}, {_glsl_format(self.frequency)}, {_glsl_format(self.strength)})"

class Repeat(SDFNode):
    glsl_dependencies = {"transforms"}
    def __init__(self, child, spacing=None, count=None, axis=None):
        super().__init__()
        self.child = child
        self.spacing = np.array(spacing) if spacing is not None else None
        self.count = np.array(count) if count is not None else None
        self.axis = np.array(axis) if axis is not None else np.array([0, 1, 0])
        if self.spacing is None and self.count is None:
            raise ValueError("Repeat requires at least 'spacing' (linear) or 'count' (polar).")

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        if self.spacing is not None and self.count is not None:
            s_str = f"vec3({_glsl_format(self.spacing[0])}, {_glsl_format(self.spacing[1])}, {_glsl_format(self.spacing[2])})"
            if self.count.size == 1: cx, cy, cz = self.count, self.count, self.count
            else: cx, cy, cz = self.count[0], self.count[1], self.count[2]
            l_str = f"vec3({_glsl_format(cx)}, {_glsl_format(cy)}, {_glsl_format(cz)})"
            p_expr = f"opLimitedRepeat({ctx.p}, {s_str}, {l_str})"
        elif self.spacing is not None:
            s_str = f"vec3({_glsl_format(self.spacing[0])}, {_glsl_format(self.spacing[1])}, {_glsl_format(self.spacing[2])})"
            p_expr = f"opRepeat({ctx.p}, {s_str})"
        else:
            reps = _glsl_format(self.count if self.count.size == 1 else self.count[0])
            if np.allclose(self.axis, [1, 0, 0]):
                p_in = f"vec3({ctx.p}.y, {ctx.p}.x, {ctx.p}.z)"
                trans = f"opPolarRepeat({p_in}, {reps})"
                p_expr = f"vec3({trans}.y, {trans}.x, {trans}.z)"
            elif np.allclose(self.axis, [0, 0, 1]):
                p_in = f"vec3({ctx.p}.x, {ctx.p}.z, {ctx.p}.y)"
                trans = f"opPolarRepeat({p_in}, {reps})"
                p_expr = f"vec3({trans}.x, {trans}.z, {trans}.y)"
            else:
                p_expr = f"opPolarRepeat({ctx.p}, {reps})"
        transformed_p = ctx.new_variable('vec3', p_expr)
        sub_ctx = ctx.with_p(transformed_p)
        child_var = self.child.to_glsl(sub_ctx)
        ctx.merge_from(sub_ctx)
        return child_var

class Mirror(_Transform):
    glsl_dependencies = {"transforms"}
    def __init__(self, child, axes):
        super().__init__(child)
        self.axes = np.array(axes)
    def _get_transform_glsl_expr(self, p_expr: str) -> str:
        a = self.axes
        a_str = f"vec3({_glsl_format(a[0])}, {_glsl_format(a[1])}, {_glsl_format(a[2])})"
        return f"opMirror({p_expr}, {a_str})"