import numpy as np
from ..core import SDFNode, GLSLContext, X, Y, Z

class _Transform(SDFNode):
    """Base class for transforms to reduce boilerplate."""
    def __init__(self, child: SDFNode):
        super().__init__()
        self.child = child

    def to_glsl(self, ctx: GLSLContext) -> str:
        # This is the core transform logic pattern.
        # 1. Get the GLSL expression for the transformed point.
        # 2. Create a new variable in the GLSL context for it.
        # 3. Create a sub-context where 'p' refers to this new variable.
        # 4. Evaluate the child node within that sub-context.
        # 5. Merge the sub-context's statements back into the main one.
        # 6. Return the variable name of the child's final result.
        ctx.dependencies.update(self.glsl_dependencies)
        
        transform_expr = self._get_transform_glsl_expr(ctx.p)
        transformed_p = ctx.new_variable('vec3', transform_expr)
        
        sub_ctx = ctx.with_p(transformed_p)
        child_var = self.child.to_glsl(sub_ctx)
        
        ctx.merge_from(sub_ctx)
        return child_var

    def _get_transform_glsl_expr(self, p_expr: str) -> str:
        """Subclasses must implement this to return the GLSL transform expression."""
        raise NotImplementedError

class Translate(_Transform):
    """Translates a child object."""
    glsl_dependencies = {"transforms"}

    def __init__(self, child: SDFNode, offset: tuple):
        super().__init__(child)
        self.offset = np.array(offset)

    def _get_transform_glsl_expr(self, p_expr: str) -> str:
        o = self.offset
        offset_str = f"vec3({float(o[0])}, {float(o[1])}, {float(o[2])})"
        return f"opTranslate({p_expr}, {offset_str})"
        
    def to_callable(self):
        child_callable = self.child.to_callable()
        offset = self.offset
        return lambda points: child_callable(points - offset)
        
class Scale(SDFNode):
    """Scales a child object."""
    glsl_dependencies = {"transforms"}

    def __init__(self, child: SDFNode, factor):
        super().__init__()
        self.child = child
        if isinstance(factor, (int, float)):
            self.factor = np.array([factor, factor, factor])
        else:
            self.factor = np.array(factor)
            
    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        f = self.factor
        factor_str = f"vec3({float(f[0])}, {float(f[1])}, {float(f[2])})"
        
        transformed_p = ctx.new_variable('vec3', f"opScale({ctx.p}, {factor_str})")
        
        sub_ctx = ctx.with_p(transformed_p)
        child_var = self.child.to_glsl(sub_ctx)
        ctx.merge_from(sub_ctx)
        
        scale_correction = np.mean(self.factor)
        
        result_expr = f"vec4({child_var}.x * {scale_correction}, {child_var}.yzw)"
        return ctx.new_variable('vec4', result_expr)
        
    def to_callable(self):
        child_callable = self.child.to_callable()
        factor = self.factor
        scale_correction = np.mean(factor)
        return lambda points: child_callable(points / factor) * scale_correction

class Rotate(_Transform):
    """Rotates a child object around a cardinal axis."""
    glsl_dependencies = {"transforms"}

    def __init__(self, child: SDFNode, axis: tuple, angle: float):
        super().__init__(child)
        self.axis = np.array(axis)
        self.angle = angle
        
    def _get_transform_glsl_expr(self, p_expr: str) -> str:
        if np.allclose(self.axis, X): func = "opRotateX"
        elif np.allclose(self.axis, Y): func = "opRotateY"
        elif np.allclose(self.axis, Z): func = "opRotateZ"
        else: raise ValueError(f"Rotation axis {self.axis} must be a cardinal axis.")
        return f"{func}({p_expr}, {float(self.angle)})"

    def to_callable(self):
        child_callable = self.child.to_callable()
        axis, angle = self.axis, self.angle
        
        c, s = np.cos(angle), np.sin(angle)
        if np.allclose(axis, X): rot_matrix = np.array([[1,0,0],[0,c,s],[0,-s,c]])
        elif np.allclose(axis, Y): rot_matrix = np.array([[c,0,-s],[0,1,0],[s,0,c]])
        else: rot_matrix = np.array([[c,s,0],[-s,c,0],[0,0,1]])
            
        return lambda points: child_callable(points @ rot_matrix.T)

class Orient(_Transform):
    """Orients a child object along an axis by swizzling coordinates."""
    glsl_dependencies = set()

    def __init__(self, child: SDFNode, axis: tuple):
        super().__init__(child)
        self.axis = axis

    def _get_transform_glsl_expr(self, p_expr: str) -> str:
        if np.allclose(self.axis, X): return f"{p_expr}.zyx"
        if np.allclose(self.axis, Y): return f"{p_expr}.xzy"
        return p_expr # Z is default, no-op
        
    def to_callable(self):
        child_callable = self.child.to_callable()
        if np.allclose(self.axis, X): return lambda p: child_callable(p[:, [2,1,0]])
        if np.allclose(self.axis, Y): return lambda p: child_callable(p[:, [0,2,1]])
        return child_callable

class Twist(_Transform):
    """Twists a child object."""
    glsl_dependencies = {"transforms"}
    def __init__(self, child: SDFNode, k: float):
        super().__init__(child)
        self.k = k
    def _get_transform_glsl_expr(self, p_expr: str) -> str:
        return f"opTwist({p_expr}, {float(self.k)})"
    def to_callable(self):
        child_callable, k = self.child.to_callable(), self.k
        def _callable(p):
            c, s = np.cos(k * p[:,1]), np.sin(k * p[:,1])
            x_new, z_new = p[:,0]*c - p[:,2]*s, p[:,0]*s + p[:,2]*c
            q = np.stack([x_new, p[:,1], z_new], axis=-1)
            return child_callable(q)
        return _callable

class Bend(_Transform):
    """Bends a child object along a cardinal axis."""
    glsl_dependencies = {"transforms"}
    def __init__(self, child: SDFNode, axis: np.ndarray, k: float):
        super().__init__(child)
        self.axis = axis
        self.k = k

    def _get_transform_glsl_expr(self, p_expr: str) -> str:
        if np.allclose(self.axis, X): func = "opBendX"
        elif np.allclose(self.axis, Y): func = "opBendY"
        else: func = "opBendZ"
        return f"{func}({p_expr}, {float(self.k)})"
        
    def to_callable(self):
        child_callable, k = self.child.to_callable(), self.k
        def _callable(p):
            if np.allclose(self.axis, X):
                c, s = np.cos(k * p[:,0]), np.sin(k * p[:,0])
                # Inverse bend transformation
                y_new, z_new = c * p[:,1] + s * p[:,2], -s * p[:,1] + c * p[:,2]
                q = np.stack([p[:,0], y_new, z_new], axis=-1)
            elif np.allclose(self.axis, Y):
                c, s = np.cos(k * p[:,1]), np.sin(k * p[:,1])
                # Inverse bend transformation
                x_new, z_new = c * p[:,0] - s * p[:,2], s * p[:,0] + c * p[:,2]
                q = np.stack([x_new, p[:,1], z_new], axis=-1)
            else: # Z
                c, s = np.cos(k * p[:,2]), np.sin(k * p[:,2])
                # Inverse bend transformation
                x_new, y_new = c * p[:,0] + s * p[:,1], -s * p[:,0] + c * p[:,1]
                q = np.stack([x_new, y_new, p[:,2]], axis=-1)
            return child_callable(q)
        return _callable

class Repeat(_Transform):
    """Repeats a child object infinitely."""
    glsl_dependencies = {"transforms"}
    def __init__(self, child, spacing):
        super().__init__(child)
        self.spacing = np.array(spacing)
    def _get_transform_glsl_expr(self, p_expr: str) -> str:
        s = self.spacing
        s_str = f"vec3({float(s[0])}, {float(s[1])}, {float(s[2])})"
        return f"opRepeat({p_expr}, {s_str})"
    def to_callable(self):
        child_callable, s = self.child.to_callable(), self.spacing
        def _callable(p):
            q = p.copy()
            # Only apply modulo where spacing is non-zero to avoid division by zero
            mask = s != 0
            q[:, mask] = np.mod(p[:, mask] + 0.5 * s[mask], s[mask]) - 0.5 * s[mask]
            return child_callable(q)
        return _callable

class LimitedRepeat(_Transform):
    """Repeats a child object a limited number of times."""
    glsl_dependencies = {"transforms"}
    def __init__(self, child, spacing, limits):
        super().__init__(child)
        self.spacing = np.array(spacing)
        self.limits = np.array(limits)
    def _get_transform_glsl_expr(self, p_expr: str) -> str:
        s, l = self.spacing, self.limits
        s_str = f"vec3({float(s[0])}, {float(s[1])}, {float(s[2])})"
        l_str = f"vec3({float(l[0])}, {float(l[1])}, {float(l[2])})"
        return f"opLimitedRepeat({p_expr}, {s_str}, {l_str})"
    def to_callable(self):
        child_callable, s, l = self.child.to_callable(), self.spacing, self.limits
        def _callable(p):
            q = p.copy()
            # Only apply repeat logic where spacing is non-zero
            mask = s != 0
            s_masked = s[mask]
            p_masked = p[:, mask]
            l_masked = l[mask]
            # Add a small epsilon to avoid division by zero if s_masked still contains zero (shouldn't happen with mask)
            rounded = np.round(p_masked / (s_masked + 1e-9))
            q[:, mask] = p_masked - s_masked * np.clip(rounded, -l_masked, l_masked)
            return child_callable(q)
        return _callable

class PolarRepeat(_Transform):
    """Repeats a child object in a circle."""
    glsl_dependencies = {"transforms"}
    def __init__(self, child, repetitions):
        super().__init__(child)
        self.repetitions = repetitions
    def _get_transform_glsl_expr(self, p_expr: str) -> str:
        return f"opPolarRepeat({p_expr}, {float(self.repetitions)})"
    def to_callable(self):
        child_callable, n = self.child.to_callable(), self.repetitions
        def _callable(p):
            a = np.arctan2(p[:,0], p[:,2])
            r = np.linalg.norm(p[:,[0,2]], axis=-1)
            angle = 2 * np.pi / n
            newA = np.mod(a, angle) - 0.5 * angle
            q = np.stack([r * np.sin(newA), p[:,1], r * np.cos(newA)], axis=-1)
            return child_callable(q)
        return _callable

class Mirror(_Transform):
    """Mirrors a child object."""
    glsl_dependencies = {"transforms"}
    def __init__(self, child, axes):
        super().__init__(child)
        self.axes = np.array(axes)
    def _get_transform_glsl_expr(self, p_expr: str) -> str:
        a = self.axes
        a_str = f"vec3({float(a[0])}, {float(a[1])}, {float(a[2])})"
        return f"opMirror({p_expr}, {a_str})"
    def to_callable(self):
        child_callable, a = self.child.to_callable(), self.axes
        def _callable(p):
            q = p.copy()
            if a[0] > 0.5: q[:,0] = np.abs(q[:,0])
            if a[1] > 0.5: q[:,1] = np.abs(q[:,1])
            if a[2] > 0.5: q[:,2] = np.abs(q[:,2])
            return child_callable(q)
        return _callable