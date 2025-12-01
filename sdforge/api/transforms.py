import numpy as np
from .core import SDFNode, GLSLContext, X, Y, Z
from .utils import _glsl_format
from .params import Param

def _smoothstep(edge0, edge1, x):
    """
    NumPy implementation of GLSL smoothstep.
    """
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

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
            # Evaluate mask in the current coordinate space (ctx.p)
            mask_var = self.mask.to_glsl(ctx)
            
            falloff_str = _glsl_format(self.mask_falloff)
            # smoothstep(edge0, edge1, x): 0 if x <= edge0, 1 if x >= edge1
            # factor = 1.0 inside mask (d<0), 0.0 outside mask (d>falloff)
            factor_expr = f"(1.0 - smoothstep(0.0, max({falloff_str}, 1e-4), {mask_var}.x))"
            
            # Mix the coordinate spaces
            final_p_expr = f"mix({ctx.p}, {raw_transform_expr}, {factor_expr})"
            transformed_p = ctx.new_variable('vec3', final_p_expr)
        else:
            transformed_p = ctx.new_variable('vec3', raw_transform_expr)

        sub_ctx = ctx.with_p(transformed_p)
        child_var = self.child.to_glsl(sub_ctx)

        ctx.merge_from(sub_ctx)
        return child_var

    def to_profile_glsl(self, ctx: GLSLContext) -> str:
        """Propagates profile intent down the transform chain."""
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

    def to_profile_callable(self):
        # Default: Reuse to_callable logic but apply to child's profile callable
        # This works because _make_callable usually wraps any function 'child_func'
        return self._make_callable(self.child.to_profile_callable())

    def _get_transform_glsl_expr(self, p_expr: str) -> str:
        """Subclasses must implement this to return the GLSL transform expression."""
        raise NotImplementedError

    def _make_callable(self, child_func):
        raise NotImplementedError
        
    def _apply_mask_to_callable(self, raw_transform_func, child_func):
        """Helper to create a masked callable if self.mask is present."""
        if not self.mask:
            return lambda p: child_func(raw_transform_func(p))
            
        mask_callable = self.mask.to_callable()
        falloff = max(self.mask_falloff, 1e-4) # Avoid div by zero, match GLSL max(..., 1e-4)
        
        def _masked_transform(p):
            # Calculate mix factor
            d_mask = mask_callable(p)
            
            # Use smoothstep to match GLSL interpolation
            # smoothstep(0.0, falloff, d_mask) -> 0 inside (d<=0), 1 outside (d>=falloff)
            # We want factor 1 inside, 0 outside, so we invert.
            factor = 1.0 - _smoothstep(0.0, falloff, d_mask)
            
            # Broadcast factor for vec3 mixing
            f = factor[:, np.newaxis]
            
            p_transformed = raw_transform_func(p)
            p_mixed = p * (1.0 - f) + p_transformed * f
            
            return child_func(p_mixed)
            
        return _masked_transform

class Translate(_Transform):
    """Internal node to translate a child object."""
    glsl_dependencies = {"transforms"}
    def __init__(self, child: SDFNode, offset: tuple, mask=None, mask_falloff=0.0):
        super().__init__(child, mask, mask_falloff)
        self.offset = np.array(offset)
    def _get_transform_glsl_expr(self, p_expr: str) -> str:
        o = self.offset
        offset_str = f"vec3({_glsl_format(o[0])}, {_glsl_format(o[1])}, {_glsl_format(o[2])})"
        return f"opTranslate({p_expr}, {offset_str})"
    def _make_callable(self, child_func):
        offset = self.offset
        transform = lambda p: p - offset
        return self._apply_mask_to_callable(transform, child_func)
    def to_callable(self): return self._make_callable(self.child.to_callable())

class Scale(SDFNode):
    """Internal node to scale a child object."""
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
        
        # Scale transform Logic
        # Regular Scale applies transform AND corrects distance: d * scale
        # Masked Scale is complex because distance correction must also be masked.
        
        if self.mask:
            # Evaluate mask
            mask_var = self.mask.to_glsl(ctx)
            falloff_str = _glsl_format(self.mask_falloff)
            factor_expr = f"(1.0 - smoothstep(0.0, max({falloff_str}, 1e-4), {mask_var}.x))"
            
            # Coordinate Mix
            raw_transformed_p = f"opScale({ctx.p}, {factor_str})"
            mixed_p = f"mix({ctx.p}, {raw_transformed_p}, {factor_expr})"
            transformed_p = ctx.new_variable('vec3', mixed_p)
            
            # Determine scale correction factor
            if isinstance(self.factor[0], (int, float)) and isinstance(self.factor[1], (int, float)) and isinstance(self.factor[2], (int, float)):
                avg_scale = np.mean(self.factor)
                scale_val = _glsl_format(avg_scale)
            else:
                scale_val = f"({_glsl_format(self.factor[0])} + {_glsl_format(self.factor[1])} + {_glsl_format(self.factor[2])}) / 3.0"
            
            # Mix the distance correction multiplier: 1.0 (no scale) vs avg_scale (scaled)
            # If we scale up by 2, we divide P by 2, distance shrinks, so we multiply d by 2.
            correction_expr = f"mix(1.0, {scale_val}, {factor_expr})"
            
        else:
            transformed_p = ctx.new_variable('vec3', f"opScale({ctx.p}, {factor_str})")
            if isinstance(self.factor[0], (int, float)) and isinstance(self.factor[1], (int, float)) and isinstance(self.factor[2], (int, float)):
                correction_expr = _glsl_format(np.mean(self.factor))
            else:
                correction_expr = f"({_glsl_format(self.factor[0])} + {_glsl_format(self.factor[1])} + {_glsl_format(self.factor[2])}) / 3.0"

        sub_ctx = ctx.with_p(transformed_p)

        if profile_mode:
            child_var = self.child.to_profile_glsl(sub_ctx)
        else:
            child_var = self.child.to_glsl(sub_ctx)

        ctx.merge_from(sub_ctx)
        
        result_expr = f"vec4({child_var}.x * ({correction_expr}), {child_var}.yzw)"
        return ctx.new_variable('vec4', result_expr)

    def to_glsl(self, ctx: GLSLContext) -> str:
        return self._base_to_glsl(ctx, profile_mode=False)

    def to_profile_glsl(self, ctx: GLSLContext) -> str:
        return self._base_to_glsl(ctx, profile_mode=True)

    def _make_callable(self, child_func):
        if any(isinstance(v, (str, Param)) for v in self.factor): raise TypeError("Cannot save mesh...")
        factor = self.factor
        scale_correction = np.mean(factor)
        
        if self.mask:
            mask_callable = self.mask.to_callable()
            falloff = max(self.mask_falloff, 1e-4)
            def _masked_scale(p):
                d_mask = mask_callable(p)
                # Use smoothstep
                mix_f = 1.0 - _smoothstep(0.0, falloff, d_mask)
                f_vec = mix_f[:, np.newaxis]
                
                # Mix coordinates: p vs p/factor
                p_trans = p / factor
                p_mixed = p * (1.0 - f_vec) + p_trans * f_vec
                
                # Mix correction: 1.0 vs scale_correction
                corr = 1.0 * (1.0 - mix_f) + scale_correction * mix_f
                
                return child_func(p_mixed) * corr
            return _masked_scale
        else:
            return lambda points: child_func(points / factor) * scale_correction

    def to_callable(self): return self._make_callable(self.child.to_callable())
    def to_profile_callable(self): return self._make_callable(self.child.to_profile_callable())

class Rotate(_Transform):
    """Internal node to rotate a child object."""
    glsl_dependencies = {"transforms"}
    def __init__(self, child: SDFNode, axis: tuple, angle: float, mask=None, mask_falloff=0.0):
        super().__init__(child, mask, mask_falloff)
        self.axis = np.array(axis, dtype=float)
        if np.linalg.norm(self.axis) == 0:
            raise ValueError("Rotation axis cannot be zero vector")
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

    def _make_callable(self, child_func):
        if isinstance(self.angle, (str, Param)): raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")
        axis, angle = self.axis, self.angle
        c, s = np.cos(angle), np.sin(angle)
        if np.allclose(axis, X): rot_matrix = np.array([[1,0,0],[0,c,s],[0,-s,c]])
        elif np.allclose(axis, Y): rot_matrix = np.array([[c,0,-s],[0,1,0],[s,0,c]])
        elif np.allclose(axis, Z): rot_matrix = np.array([[c,s,0],[-s,c,0],[0,0,1]])
        else:
            kx, ky, kz = axis
            K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
            rot_matrix = np.eye(3) + s * K + (1 - c) * (K @ K)
        
        transform = lambda points: points @ rot_matrix.T
        return self._apply_mask_to_callable(transform, child_func)

    def to_callable(self): return self._make_callable(self.child.to_callable())

class Orient(_Transform):
    """Internal node to orient a child object."""
    glsl_dependencies = set()
    def __init__(self, child: SDFNode, axis: tuple):
        super().__init__(child)
        self.axis = axis
    def _get_transform_glsl_expr(self, p_expr: str) -> str:
        if np.allclose(self.axis, X): return f"{p_expr}.zyx"
        if np.allclose(self.axis, Y): return f"{p_expr}.xzy"
        return p_expr 
    def _make_callable(self, child_func):
        if np.allclose(self.axis, X): return lambda p: child_func(p[:, [2,1,0]])
        if np.allclose(self.axis, Y): return lambda p: child_func(p[:, [0,2,1]])
        return child_func
    def to_callable(self): return self._make_callable(self.child.to_callable())

class Twist(_Transform):
    """Internal node to twist a child object."""
    glsl_dependencies = {"transforms"}
    def __init__(self, child: SDFNode, strength: float, mask=None, mask_falloff=0.0):
        super().__init__(child, mask, mask_falloff)
        self.strength = strength
    def _get_transform_glsl_expr(self, p_expr: str) -> str:
        return f"opTwist({p_expr}, {_glsl_format(self.strength)})"
    def _make_callable(self, child_func):
        if isinstance(self.strength, (str, Param)): raise TypeError("Cannot save mesh...")
        s = self.strength
        def transform(p):
            c, s_val = np.cos(s * p[:,1]), np.sin(s * p[:,1])
            x_new, z_new = p[:,0]*c - p[:,2]*s_val, p[:,0]*s_val + p[:,2]*c
            q = np.stack([x_new, p[:,1], z_new], axis=-1)
            return q
        return self._apply_mask_to_callable(transform, child_func)
    def to_callable(self): return self._make_callable(self.child.to_callable())

class Bend(_Transform):
    """Internal node to bend a child object."""
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
    def _make_callable(self, child_func):
        if isinstance(self.curvature, (str, Param)): raise TypeError("Cannot save mesh...")
        k = self.curvature
        def transform(p):
            if np.allclose(self.axis, X):
                c, s = np.cos(k * p[:,0]), np.sin(k * p[:,0])
                y_new, z_new = c * p[:,1] + s * p[:,2], -s * p[:,1] + c * p[:,2]
                q = np.stack([p[:,0], y_new, z_new], axis=-1)
            elif np.allclose(self.axis, Y):
                c, s = np.cos(k * p[:,1]), np.sin(k * p[:,1])
                x_new, z_new = c * p[:,0] - s * p[:,2], s * p[:,0] + c * p[:,2]
                q = np.stack([x_new, p[:,1], z_new], axis=-1)
            else: # Z
                c, s = np.cos(k * p[:,2]), np.sin(k * p[:,2])
                x_new, y_new = c * p[:,0] + s * p[:,1], -s * p[:,0] + c * p[:,1]
                q = np.stack([x_new, y_new, p[:,2]], axis=-1)
            return q
        return self._apply_mask_to_callable(transform, child_func)
    def to_callable(self): return self._make_callable(self.child.to_callable())

class Warp(_Transform):
    """Internal node to apply domain warping using vector noise."""
    glsl_dependencies = {"transforms", "noise"}
    def __init__(self, child: SDFNode, frequency: float, strength: float, mask=None, mask_falloff=0.0):
        super().__init__(child, mask, mask_falloff)
        self.frequency = frequency
        self.strength = strength

    def _get_transform_glsl_expr(self, p_expr: str) -> str:
        return f"opWarp({p_expr}, {_glsl_format(self.frequency)}, {_glsl_format(self.strength)})"

    def _make_callable(self, child_func):
        # We don't have a numpy implementation of Simplex Noise in the core library yet.
        raise TypeError("Cannot create a callable (save mesh) for an object with procedural Domain Warping.")

    def to_callable(self): return self._make_callable(None)


class Repeat(_Transform):
    """Internal node to repeat a child object."""
    glsl_dependencies = {"transforms"}
    def __init__(self, child, spacing):
        super().__init__(child)
        self.spacing = np.array(spacing)
    def _get_transform_glsl_expr(self, p_expr: str) -> str:
        s = self.spacing
        s_str = f"vec3({_glsl_format(s[0])}, {_glsl_format(s[1])}, {_glsl_format(s[2])})"
        return f"opRepeat({p_expr}, {s_str})"
    def _make_callable(self, child_func):
        s = self.spacing
        def _callable(p):
            q = p.copy()
            mask = s != 0
            q[:, mask] = np.mod(p[:, mask] + 0.5 * s[mask], s[mask]) - 0.5 * s[mask]
            return child_func(q)
        return _callable
    def to_callable(self): return self._make_callable(self.child.to_callable())

class LimitedRepeat(_Transform):
    """Internal node to repeat a child object."""
    glsl_dependencies = {"transforms"}
    def __init__(self, child, spacing, limits):
        super().__init__(child)
        self.spacing = np.array(spacing)
        self.limits = np.array(limits)
    def _get_transform_glsl_expr(self, p_expr: str) -> str:
        s, l = self.spacing, self.limits
        s_str = f"vec3({_glsl_format(s[0])}, {_glsl_format(s[1])}, {_glsl_format(s[2])})"
        l_str = f"vec3({_glsl_format(l[0])}, {_glsl_format(l[1])}, {_glsl_format(l[2])})"
        return f"opLimitedRepeat({p_expr}, {s_str}, {l_str})"
    def _make_callable(self, child_func):
        s, l = self.spacing, self.limits
        def _callable(p):
            q = p.copy()
            mask = s != 0
            s_masked = s[mask]
            p_masked = p[:, mask]
            l_masked = l[mask]
            rounded = np.round(p_masked / (s_masked + 1e-9))
            q[:, mask] = p_masked - s_masked * np.clip(rounded, -l_masked, l_masked)
            return child_func(q)
        return _callable
    def to_callable(self): return self._make_callable(self.child.to_callable())

class PolarRepeat(_Transform):
    """Internal node to repeat a child object."""
    glsl_dependencies = {"transforms"}
    def __init__(self, child, repetitions):
        super().__init__(child)
        self.repetitions = repetitions
    def _get_transform_glsl_expr(self, p_expr: str) -> str:
        return f"opPolarRepeat({p_expr}, {_glsl_format(self.repetitions)})"
    def _make_callable(self, child_func):
        if isinstance(self.repetitions, (str, Param)): raise TypeError("Cannot save mesh...")
        n = self.repetitions
        def _callable(p):
            a = np.arctan2(p[:,0], p[:,2])
            r = np.linalg.norm(p[:,[0,2]], axis=-1)
            angle = 2 * np.pi / n
            newA = np.mod(a, angle) - 0.5 * angle
            q = np.stack([r * np.sin(newA), p[:,1], r * np.cos(newA)], axis=-1)
            return child_func(q)
        return _callable
    def to_callable(self): return self._make_callable(self.child.to_callable())

class Mirror(_Transform):
    """Internal node to mirror a child object."""
    glsl_dependencies = {"transforms"}
    def __init__(self, child, axes):
        super().__init__(child)
        self.axes = np.array(axes)
    def _get_transform_glsl_expr(self, p_expr: str) -> str:
        a = self.axes
        a_str = f"vec3({_glsl_format(a[0])}, {_glsl_format(a[1])}, {_glsl_format(a[2])})"
        return f"opMirror({p_expr}, {a_str})"
    def _make_callable(self, child_func):
        a = self.axes
        def _callable(p):
            q = p.copy()
            if a[0] > 0.5: q[:,0] = np.abs(q[:,0])
            if a[1] > 0.5: q[:,1] = np.abs(q[:,1])
            if a[2] > 0.5: q[:,2] = np.abs(q[:,2])
            return child_func(q)
        return _callable
    def to_callable(self): return self._make_callable(self.child.to_callable())