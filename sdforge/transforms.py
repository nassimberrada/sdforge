import numpy as np
import re
from .core import SDFObject, _get_glsl_content, _glsl_format, X, Y, Z

# --- Basic Transformations ---
class _Transform(SDFObject):
    """Base class for robust transformations."""
    def __init__(self, child):
        super().__init__()
        self.child = child

    def _get_transform_glsl(self, p_expr: str) -> str:
        raise NotImplementedError("Subclasses must implement _get_transform_glsl")

    def to_glsl(self) -> str:
        transformed_p = f"({self._get_transform_glsl('p')})"
        child_glsl = self.child.to_glsl()
        child_glsl = re.sub(r'\bp\b', transformed_p, child_glsl)
        return child_glsl

    def get_glsl_definitions(self):
        return [_get_glsl_content("transforms.glsl")] + self.child.get_glsl_definitions()

    def _collect_materials(self, materials):
        self.child._collect_materials(materials)


class Translate(_Transform):
    """Translates a child object."""
    def __init__(self, child, offset):
        super().__init__(child)
        self.offset = offset
    def _get_transform_glsl(self, p_expr: str) -> str:
        o = self.offset
        return f"opTranslate({p_expr}, vec3({_glsl_format(o[0])}, {_glsl_format(o[1])}, {_glsl_format(o[2])}))"
    def to_callable(self):
        child_call = self.child.to_callable()
        return lambda p: child_call(p - self.offset)


class Scale(SDFObject):
    """Scales a child object."""
    def __init__(self, child, factor):
        super().__init__()
        self.child, self.factor = child, factor
    def to_glsl(self) -> str:
        f = self.factor
        if isinstance(f, (tuple, list, np.ndarray)):
            factor_str = f"vec3({_glsl_format(f[0])}, {_glsl_format(f[1])}, {_glsl_format(f[2])})"
            len_str = f"dot(vec3(1.0), vec3({_glsl_format(f[0])}, {_glsl_format(f[1])}, {_glsl_format(f[2])}))/3.0"
        else:
            factor_str = f"vec3({_glsl_format(f)})"
            len_str = _glsl_format(f)
       
        transformed_p = f"opScale(p, {factor_str})"
        child_glsl = self.child.to_glsl()
        child_glsl = re.sub(r'\bp\b', transformed_p, child_glsl)

        child_expr = f"({child_glsl})"
        return f"vec4({child_expr}.x * ({len_str}), {child_expr}.y, {child_expr}.z, {child_expr}.w)"

    def to_callable(self):
        if isinstance(self.factor, str): 
            raise TypeError("Animated parameters not supported for mesh export.")
        child_call = self.child.to_callable()
        f = np.array(self.factor if isinstance(self.factor, (tuple, list, np.ndarray)) else (self.factor,))
        return lambda p: child_call(p / f) * np.mean(f)  # approximate isotropic correction
    def get_glsl_definitions(self):
        return [_get_glsl_content("transforms.glsl")] + self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)


class Rotate(_Transform):
    """Rotates a child object."""
    def __init__(self, child, axis, angle):
        super().__init__(child)
        self.axis, self.angle = axis, angle
    def _get_transform_glsl(self, p_expr: str) -> str:
        if np.allclose(self.axis, X): func = "opRotateX"
        elif np.allclose(self.axis, Y): func = "opRotateY"
        else: func = "opRotateZ"
        return f"{func}({p_expr}, {_glsl_format(self.angle)})"
    def to_callable(self):
        if isinstance(self.angle, str): 
            raise TypeError("Animated parameters not supported for mesh export.")
        child_call, angle = self.child.to_callable(), float(self.angle)
        c, s = np.cos(angle), np.sin(angle)
        if np.allclose(self.axis, X):
            R = np.array([[1,0,0],[0,c,-s],[0,s,c]])
        elif np.allclose(self.axis, Y):
            R = np.array([[c,0,s],[0,1,0],[-s,0,c]])
        else:
            R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
        return lambda p: child_call(p @ R.T)


class Orient(_Transform):
    """Orients a child object along an axis."""
    def __init__(self, child, axis):
        super().__init__(child)
        self.axis = axis

    def to_glsl(self) -> str:
        # If the orientation is along the Z axis, it's a no-op.
        # Return the child's GLSL directly to avoid a redundant wrapper.
        if np.allclose(self.axis, Z):
            return self.child.to_glsl()
        # For other axes, use the robust scoped transformation from the parent class.
        return super().to_glsl()
    def _get_transform_glsl(self, p_expr: str) -> str:
        if np.allclose(self.axis, X): return f"{p_expr}.zyx"
        if np.allclose(self.axis, Y): return f"{p_expr}.xzy"
        return p_expr # Z is default, no-op
    def to_callable(self):
        child_call = self.child.to_callable()
        if np.allclose(self.axis, X): return lambda p: child_call(p[:, [2,1,0]])
        elif np.allclose(self.axis, Y): return lambda p: child_call(p[:, [0,2,1]])
        return child_call


class Twist(_Transform):
    """Twists a child object."""
    def __init__(self, child, k):
        super().__init__(child)
        self.k = k
    def _get_transform_glsl(self, p_expr: str) -> str:
        return f"opTwist({p_expr}, {_glsl_format(self.k)})"
    def to_callable(self):
        if isinstance(self.k, str): raise TypeError("Animated parameters not supported for mesh export.")
        child_call, k = self.child.to_callable(), float(self.k)
        def _callable(p):
            c, s = np.cos(k * p[:,1]), np.sin(k * p[:,1])
            x_new, z_new = p[:,0]*c - p[:,2]*s, p[:,0]*s + p[:,2]*c
            q = np.stack([x_new, p[:,1], z_new], axis=-1)
            return child_call(q)
        return _callable

class ShearXY(_Transform):
    """Shears a child object."""
    def __init__(self, child, shear):
        super().__init__(child)
        self.shear = shear
    def _get_transform_glsl(self, p_expr: str) -> str:
        sh = self.shear
        return f"opShearXY({p_expr}, vec2({_glsl_format(sh[0])}, {_glsl_format(sh[1])}))"
    def to_callable(self):
        child_call, sh = self.child.to_callable(), np.array(self.shear)
        def _callable(p):
            q = p.copy()
            q[:,0] += sh[0] * p[:,2]
            q[:,1] += sh[1] * p[:,2]
            return child_call(q)
        return _callable

class ShearXZ(_Transform):
    """Shears a child object."""
    def __init__(self, child, shear):
        super().__init__(child)
        self.shear = shear
    def _get_transform_glsl(self, p_expr: str) -> str:
        sh = self.shear
        return f"opShearXZ({p_expr}, vec2({_glsl_format(sh[0])}, {_glsl_format(sh[1])}))"
    def to_callable(self):
        child_call, sh = self.child.to_callable(), np.array(self.shear)
        def _callable(p):
            q = p.copy()
            q[:,0] += sh[0] * p[:,1]
            q[:,2] += sh[1] * p[:,1]
            return child_call(q)
        return _callable

class ShearYZ(_Transform):
    """Shears a child object."""
    def __init__(self, child, shear):
        super().__init__(child)
        self.shear = shear
    def _get_transform_glsl(self, p_expr: str) -> str:
        sh = self.shear
        return f"opShearYZ({p_expr}, vec2({_glsl_format(sh[0])}, {_glsl_format(sh[1])}))"
    def to_callable(self):
        child_call, sh = self.child.to_callable(), np.array(self.shear)
        def _callable(p):
            q = p.copy()
            q[:,1] += sh[0] * p[:,0]
            q[:,2] += sh[1] * p[:,0]
            return child_call(q)
        return _callable

class BendX(_Transform):
    """Bends a child object."""
    def __init__(self, child, k):
        super().__init__(child)
        self.k = k
    def _get_transform_glsl(self, p_expr: str) -> str:
        return f"opBendX({p_expr}, {_glsl_format(self.k)})"
    def to_callable(self):
        if isinstance(self.k, str): raise TypeError("Animated bend not supported for mesh export.")
        child_call, k = self.child.to_callable(), float(self.k)
        def _callable(p):
            c, s = np.cos(k * p[:,0]), np.sin(k * p[:,0])
            y_new = c * p[:,1] - s * p[:,2]
            z_new = s * p[:,1] + c * p[:,2]
            q = np.stack([p[:,0], y_new, z_new], axis=-1)
            return child_call(q)
        return _callable

class BendY(_Transform):
    """Bends a child object."""
    def __init__(self, child, k):
        super().__init__(child)
        self.k = k
    def _get_transform_glsl(self, p_expr: str) -> str:
        return f"opBendY({p_expr}, {_glsl_format(self.k)})"
    def to_callable(self):
        if isinstance(self.k, str): raise TypeError("Animated bend not supported for mesh export.")
        child_call, k = self.child.to_callable(), float(self.k)
        def _callable(p):
            c, s = np.cos(k * p[:,1]), np.sin(k * p[:,1])
            x_new = c * p[:,0] + s * p[:,2]
            z_new = -s * p[:,0] + c * p[:,2]
            q = np.stack([x_new, p[:,1], z_new], axis=-1)
            return child_call(q)
        return _callable

class BendZ(_Transform):
    """Bends a child object."""
    def __init__(self, child, k):
        super().__init__(child)
        self.k = k
    def _get_transform_glsl(self, p_expr: str) -> str:
        return f"opBendZ({p_expr}, {_glsl_format(self.k)})"
    def to_callable(self):
        if isinstance(self.k, str): raise TypeError("Animated bend not supported for mesh export.")
        child_call, k = self.child.to_callable(), float(self.k)
        def _callable(p):
            c, s = np.cos(k * p[:,2]), np.sin(k * p[:,2])
            x_new = c * p[:,0] - s * p[:,1]
            y_new = s * p[:,0] + c * p[:,1]
            q = np.stack([x_new, y_new, p[:,2]], axis=-1)
            return child_call(q)
        return _callable


class Repeat(_Transform):
    """Repeats a child object infinitely."""
    def __init__(self, child, spacing):
        super().__init__(child)
        self.spacing = spacing
    def _get_transform_glsl(self, p_expr: str) -> str:
        s = self.spacing
        return f"opRepeat({p_expr}, vec3({_glsl_format(s[0])}, {_glsl_format(s[1])}, {_glsl_format(s[2])}))"
    def to_callable(self):
        child_call, s = self.child.to_callable(), self.spacing
        def _callable(p):
            q = p.copy()
            mask = s != 0
            if np.any(mask):
                p_masked = p[:, mask]
                s_masked = s[mask]
                q[:, mask] = np.mod(p_masked + 0.5 * s_masked, s_masked) - 0.5 * s_masked
            return child_call(q)
        return _callable

class LimitedRepeat(_Transform):
    """Repeats a child object a limited number of times."""
    def __init__(self, child, spacing, limits):
        super().__init__(child)
        self.spacing, self.limits = spacing, limits
    def _get_transform_glsl(self, p_expr: str) -> str:
        s = self.spacing
        l = self.limits
        s_str = f"vec3({_glsl_format(s[0])}, {_glsl_format(s[1])}, {_glsl_format(s[2])})"
        l_str = f"vec3({_glsl_format(l[0])}, {_glsl_format(l[1])}, {_glsl_format(l[2])})"
        return f"opLimitedRepeat({p_expr}, {s_str}, {l_str})"
    def to_callable(self):
        child_call = self.child.to_callable()
        s = self.spacing
        l = self.limits
        def _callable(p):
            q = p.copy()
            mask = s != 0
            if np.any(mask):
                p_masked = p[:, mask]
                s_masked = s[mask]
                l_masked = l[mask]
                q[:, mask] = p_masked - s_masked * np.clip(np.round(p_masked / s_masked), -l_masked, l_masked)
            return child_call(q)
        return _callable


class PolarRepeat(_Transform):
    """Repeats a child object in a circle."""
    def __init__(self, child, repetitions):
        super().__init__(child)
        self.repetitions = repetitions
    def _get_transform_glsl(self, p_expr: str) -> str:
        return f"opPolarRepeat({p_expr}, {_glsl_format(self.repetitions)})"
    def to_callable(self):
        if isinstance(self.repetitions, str):
            raise TypeError("Animated polar repeat not supported for mesh export.")
        child_call, n = self.child.to_callable(), float(self.repetitions)
        def _callable(p):
            a = np.arctan2(p[:,0], p[:,2])
            r = np.linalg.norm(p[:,[0,2]], axis=-1)
            angle = 2*np.pi/n
            newA = np.mod(a, angle) - 0.5*angle
            x_new = r * np.sin(newA)
            z_new = r * np.cos(newA)
            q = np.stack([x_new, p[:,1], z_new], axis=-1)
            return child_call(q)
        return _callable


class Mirror(_Transform):
    """Mirrors a child object."""
    def __init__(self, child, axes):
        super().__init__(child)
        self.axes = axes
    def _get_transform_glsl(self, p_expr: str) -> str:
        a = self.axes
        return f"opMirror({p_expr}, vec3({_glsl_format(a[0])}, {_glsl_format(a[1])}, {_glsl_format(a[2])}))"
    def to_callable(self):
        child_call, a = self.child.to_callable(), self.axes
        def _callable(p):
            q = p.copy()
            if a[0] > 0.5: q[:,0] = np.abs(q[:,0])
            if a[1] > 0.5: q[:,1] = np.abs(q[:,1])
            if a[2] > 0.5: q[:,2] = np.abs(q[:,2])
            return child_call(q)
        return _callable
