import numpy as np
from ..core import SDFNode, GLSLContext
from ..utils import _glsl_format
from .params import Param

class Sphere(SDFNode):
    """Represents a sphere primitive."""
    glsl_dependencies = {"primitives"}

    def __init__(self, radius: float = 1.0):
        super().__init__()
        self.radius = radius

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        dist_expr = f"sdSphere({ctx.p}, {_glsl_format(self.radius)})"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

    def to_callable(self):
        if isinstance(self.radius, (str, Param)):
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")
        r_val = self.radius
        def _callable(points: np.ndarray) -> np.ndarray:
            return np.linalg.norm(points, axis=-1) - r_val
        return _callable

def sphere(radius: float = 1.0) -> SDFNode:
    """
    Creates a sphere centered at the origin.

    Args:
        radius (float, optional): The radius of the sphere. Defaults to 1.0.
    """
    return Sphere(radius)

class Box(SDFNode):
    """Represents a box primitive."""
    glsl_dependencies = {"primitives"}

    def __init__(self, size: tuple = (1.0, 1.0, 1.0)):
        super().__init__()
        self.size = size

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        
        s = []
        for v in self.size:
            if isinstance(v, (int, float)):
                s.append(_glsl_format(v / 2.0))
            else:
                s.append(f"({_glsl_format(v)} / 2.0)")
        size_vec = f"vec3({s[0]}, {s[1]}, {s[2]})"
        
        dist_expr = f"sdBox({ctx.p}, {size_vec})"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

    def to_callable(self):
        is_dynamic = any(isinstance(v, (str, Param)) for v in self.size)
        if is_dynamic:
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")
        
        half_size = np.array(self.size) / 2.0
        
        def _callable(points: np.ndarray) -> np.ndarray:
            q = np.abs(points) - half_size
            dist = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
            dist += np.minimum(np.max(q, axis=-1), 0.0)
            return dist
        return _callable

def box(size=1.0) -> SDFNode:
    """
    Creates a box centered at the origin.

    Args:
        size (float or tuple, optional): The size of the box. If a float,
                                         creates a cube. If a tuple, specifies
                                         (width, height, depth). Defaults to 1.0.
    """
    if isinstance(size, (int, float, str, Param)):
        size = (size, size, size)
    return Box(size=tuple(size))

class HexPrism(SDFNode):
    """Represents a hexagonal prism."""
    glsl_dependencies = {"primitives"}

    def __init__(self, radius: float = 1.0, height: float = 1.0):
        super().__init__()
        self.radius = radius
        self.height = height

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        # h.x = radius, h.y = half-height
        r = _glsl_format(self.radius)
        h = _glsl_format(self.height / 2.0) if not isinstance(self.height, (str, Param)) else f"({_glsl_format(self.height)})/2.0"
        
        dist_expr = f"sdHexPrism({ctx.p}, vec2({r}, {h}))"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

    def to_callable(self):
        if isinstance(self.radius, (str, Param)) or isinstance(self.height, (str, Param)):
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")
        
        radius = self.radius
        half_height = self.height / 2.0
        k = np.array([-0.8660254, 0.5, 0.57735026])

        def _callable(points: np.ndarray) -> np.ndarray:
            # Convert to float to allow in-place operations on integer inputs
            p = np.abs(points).astype(float)
            
            dot_k_p = p[:,0] * k[0] + p[:,1] * k[1]
            min_dot = np.minimum(dot_k_p, 0.0)
            p[:,0] -= 2.0 * min_dot * k[0]
            p[:,1] -= 2.0 * min_dot * k[1]
            
            clamp_val = np.clip(p[:,0], -k[2] * radius, k[2] * radius)
            vec_sub = p[:, :2] - np.stack([clamp_val, np.full_like(p[:,0], radius)], axis=-1)
            len_xy = np.linalg.norm(vec_sub, axis=-1)
            d_x = len_xy * np.sign(p[:,1] - radius)
            d_y = p[:,2] - half_height
            
            d = np.stack([d_x, d_y], axis=-1)
            return np.minimum(np.maximum(d[:,0], d[:,1]), 0.0) + np.linalg.norm(np.maximum(d, 0.0), axis=-1)
        return _callable

def hex_prism(radius: float = 1.0, height: float = 1.0) -> SDFNode:
    """
    Creates a hexagonal prism oriented along the Z axis.

    Args:
        radius (float): The radius (apothem) of the hexagon.
        height (float): The total height (depth) of the prism.
    """
    return HexPrism(radius, height)

class Pyramid(SDFNode):
    """Represents a pyramid with a square base."""
    glsl_dependencies = {"primitives"}

    def __init__(self, height: float = 1.0):
        super().__init__()
        self.height = height

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        dist_expr = f"sdPyramid({ctx.p}, {_glsl_format(self.height)})"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

    def to_callable(self):
        if isinstance(self.height, (str, Param)):
             raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")
        
        h = self.height
        m2 = h*h + 0.25
        
        def _callable(points: np.ndarray) -> np.ndarray:
            # Cast to float to support in-place subtraction
            p = points.astype(float)
            
            # Center vertically logic (matching GLSL shift)
            p[:, 1] += h * 0.5

            p[:, [0, 2]] = np.abs(p[:, [0, 2]])
            
            mask = p[:,2] > p[:,0]
            p[mask] = p[mask][:, [2, 1, 0]] # Swap x and z
            
            p[:, [0, 2]] -= 0.5
            
            q = np.stack([p[:,2], h*p[:,1] - 0.5*p[:,0], h*p[:,0] + 0.5*p[:,1]], axis=-1)
            
            s = np.maximum(-q[:,0], 0.0)
            t = np.clip((q[:,1] - 0.5*p[:,2]) / (m2 + 0.25), 0.0, 1.0)
            
            a = m2 * (q[:,0] + s)**2 + q[:,1]**2
            b = m2 * (q[:,0] + 0.5*t)**2 + (q[:,1] - m2*t)**2
            
            d2_cond = np.minimum(q[:,1], -q[:,0]*m2 - q[:,1]*0.5) > 0.0
            d2 = np.where(d2_cond, 0.0, np.minimum(a, b))
            
            return np.sqrt((d2 + q[:,2]**2) / m2) * np.sign(np.maximum(q[:,2], -p[:,1]))

        return _callable

def pyramid(height: float = 1.0) -> SDFNode:
    """
    Creates a pyramid with a square base centered at (0,0,0).

    Args:
        height (float): The vertical height of the pyramid.
    """
    return Pyramid(height)

class Torus(SDFNode):
    """Represents a torus primitive."""
    glsl_dependencies = {"primitives"}

    def __init__(self, radius_major: float = 1.0, radius_minor: float = 0.25):
        super().__init__()
        self.radius_major, self.radius_minor = radius_major, radius_minor
        
    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        dist_expr = f"sdTorus({ctx.p}, vec2({_glsl_format(self.radius_major)}, {_glsl_format(self.radius_minor)}))"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

    def to_callable(self):
        if isinstance(self.radius_major, (str, Param)) or isinstance(self.radius_minor, (str, Param)):
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")
        major, minor = self.radius_major, self.radius_minor
        def _callable(points: np.ndarray) -> np.ndarray:
            q = np.array([np.linalg.norm(points[:, [0, 2]], axis=-1) - major, points[:, 1]]).T
            return np.linalg.norm(q, axis=-1) - minor
        return _callable

def torus(radius_major: float = 1.0, radius_minor: float = 0.25) -> SDFNode:
    """
    Creates a torus centered at the origin, oriented in the XZ plane.

    Args:
        radius_major (float): Distance from the origin to the center of the tube.
        radius_minor (float): Radius of the tube itself.
    """
    return Torus(radius_major, radius_minor)

class Line(SDFNode):
    """Represents a line segment primitive with a radius."""
    glsl_dependencies = {"primitives"}

    def __init__(self, start, end, radius: float = 0.1, rounded_caps: bool = True):
        super().__init__()
        self.start, self.end, self.radius = np.array(start), np.array(end), radius
        self.rounded_caps = rounded_caps

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        a, b, r = self.start, self.end, _glsl_format(self.radius)
        a_str = f"vec3({a[0]},{a[1]},{a[2]})"
        b_str = f"vec3({b[0]},{b[1]},{b[2]})"
        func = "sdCapsule" if self.rounded_caps else "sdCappedCylinder"
        dist_expr = f"{func}({ctx.p}, {a_str}, {b_str}, {r})"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

    def to_callable(self):
        if isinstance(self.radius, (str, Param)):
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")
        a, b, r = self.start, self.end, self.radius
        if self.rounded_caps:
            def _callable(points: np.ndarray) -> np.ndarray:
                pa = points - a; ba = b - a
                h = np.clip(np.dot(pa, ba) / np.dot(ba, ba), 0.0, 1.0)
                return np.linalg.norm(pa - ba * h[:, np.newaxis], axis=-1) - r
            return _callable
        else: # Capped cylinder
            def _callable(points: np.ndarray) -> np.ndarray:
                ba = b - a
                pa = points - a
                baba = np.dot(ba, ba)
                paba = np.dot(pa, ba)
                x = np.linalg.norm(pa * baba - ba * paba[:, np.newaxis], axis=-1) - r * baba
                y = np.abs(paba - baba * 0.5) - baba * 0.5
                x2 = x*x
                y2 = y*y*baba
                d_inner = np.where(np.maximum(x, y) < 0.0, -np.minimum(x2, y2), (np.where(x > 0.0, x2, 0.0) + np.where(y > 0.0, y2, 0.0)))
                return np.sign(d_inner) * np.sqrt(np.abs(d_inner)) / baba
            return _callable

def line(start, end, radius: float = 0.1, rounded_caps: bool = True) -> SDFNode:
    """
    Creates a line segment between two points.

    Args:
        start (tuple): The starting point (x, y, z).
        end (tuple): The ending point (x, y, z).
        radius (float, optional): The thickness of the line. Defaults to 0.1.
        rounded_caps (bool, optional): Whether to round the ends. Defaults to True.
    """
    return Line(start, end, radius, rounded_caps)

class Bezier(SDFNode):
    """Represents a Quadratic Bezier tube."""
    glsl_dependencies = {"primitives"}

    def __init__(self, p0, p1, p2, radius: float = 0.1):
        super().__init__()
        self.p0, self.p1, self.p2 = np.array(p0), np.array(p1), np.array(p2)
        self.radius = radius

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        A_str = f"vec3({self.p0[0]},{self.p0[1]},{self.p0[2]})"
        B_str = f"vec3({self.p1[0]},{self.p1[1]},{self.p1[2]})"
        C_str = f"vec3({self.p2[0]},{self.p2[1]},{self.p2[2]})"
        dist_expr = f"sdBezier({ctx.p}, {A_str}, {B_str}, {C_str}, {_glsl_format(self.radius)})"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

    def to_callable(self):
        if isinstance(self.radius, (str, Param)):
             raise TypeError("Cannot save mesh with animated parameters.")
        
        A, B, C, r = self.p0, self.p1, self.p2, self.radius
        
        # Precompute constant curve vectors
        a = B - A
        b = A - 2.0*B + C
        c = a * 2.0
        
        def _callable(points: np.ndarray) -> np.ndarray:
            # d = A - pos
            d = A[np.newaxis, :] - points
            
            kk = 1.0 / np.dot(b, b)
            kx = kk * np.dot(a, b)
            
            dot_d_b = np.dot(d, b)
            dot_d_a = np.dot(d, a)
            
            ky = kk * (2.0 * np.dot(a, a) + dot_d_b) / 3.0
            kz = kk * dot_d_a
            
            # Align coefficients with standard cubic t^3 + kx_full*t^2 + ...
            kx_full = 3.0 * kx
            ky_full = 3.0 * ky
            kz_full = kz

            # Depressed cubic: x^3 + p x + q = 0, where t = x - kx_full/3
            p = ky_full - kx_full*kx_full / 3.0
            q = 2.0*kx_full*kx_full*kx_full/27.0 - kx_full*ky_full/3.0 + kz_full
            
            p3 = p*p*p
            discriminant = q*q/4.0 + p3/27.0
            
            t_candidates = np.zeros((points.shape[0], 3))
            num_roots = np.zeros(points.shape[0], dtype=int)
            
            # Case 1: D > 0 (One real root)
            mask_d_pos = discriminant > 0
            if np.any(mask_d_pos):
                d_sqrt = np.sqrt(discriminant[mask_d_pos])
                q_masked = q[mask_d_pos]
                
                u1 = -q_masked/2.0 + d_sqrt
                u2 = -q_masked/2.0 - d_sqrt
                
                # Use cbrt to handle negative bases correctly
                u1 = np.cbrt(u1)
                u2 = np.cbrt(u2)
                
                t_candidates[mask_d_pos, 0] = u1 + u2 - kx_full/3.0
                num_roots[mask_d_pos] = 1

            # Case 2: D <= 0 (Three real roots)
            mask_d_neg = ~mask_d_pos
            if np.any(mask_d_neg):
                p3_masked = p3[mask_d_neg]
                q_masked = q[mask_d_neg]
                p_masked = p[mask_d_neg]
                kx_neg = kx_full if np.isscalar(kx_full) else kx_full
                
                val = -27.0 / (p3_masked + 1e-14)
                val_sqrt = np.sqrt(np.abs(val)) 
                
                arg = -val_sqrt * q_masked / 2.0
                arg = np.clip(arg, -1.0, 1.0)
                
                v = np.arccos(arg) / 3.0
                m = np.cos(v)
                n = np.sin(v) * 1.732050808
                
                sqrt_neg_p_3 = np.sqrt(np.abs(-p_masked/3.0))
                
                offset = -kx_neg/3.0
                t_candidates[mask_d_neg, 0] = (m + m) * sqrt_neg_p_3 + offset
                t_candidates[mask_d_neg, 1] = (-n - m) * sqrt_neg_p_3 + offset
                t_candidates[mask_d_neg, 2] = (n - m) * sqrt_neg_p_3 + offset
                num_roots[mask_d_neg] = 3

            t0 = np.clip(t_candidates[:, 0], 0.0, 1.0)
            # Fixed distance calc: d + c*t + b*t^2
            q0 = d + np.outer(t0, c) + np.outer(t0**2, b)
            dist_sq = np.sum(q0*q0, axis=1)
            
            mask_3 = num_roots == 3
            if np.any(mask_3):
                t1 = np.clip(t_candidates[mask_3, 1], 0.0, 1.0)
                q1 = d[mask_3] + np.outer(t1, c) + np.outer(t1**2, b)
                dist_sq[mask_3] = np.minimum(dist_sq[mask_3], np.sum(q1*q1, axis=1))
                
                t2 = np.clip(t_candidates[mask_3, 2], 0.0, 1.0)
                q2 = d[mask_3] + np.outer(t2, c) + np.outer(t2**2, b)
                dist_sq[mask_3] = np.minimum(dist_sq[mask_3], np.sum(q2*q2, axis=1))

            return np.sqrt(dist_sq) - r

        return _callable

def curve(p0, p1, p2, radius: float = 0.1) -> SDFNode:
    """
    Creates a Quadratic Bezier tube (curve) defined by 3 points.

    Args:
        p0 (tuple): The starting point (x, y, z).
        p1 (tuple): The control point (x, y, z).
        p2 (tuple): The ending point (x, y, z).
        radius (float, optional): The thickness of the tube. Defaults to 0.1.
    """
    return Bezier(p0, p1, p2, radius)

class Cylinder(SDFNode):
    """Represents a cylinder primitive."""
    glsl_dependencies = {"primitives"}

    def __init__(self, radius: float = 0.5, height: float = 1.0):
        super().__init__()
        self.radius, self.height = radius, height

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        h_half = _glsl_format(self.height / 2.0) if not isinstance(self.height, (str, Param)) else f"({_glsl_format(self.height)})/2.0"
        dist_expr = f"sdCylinder({ctx.p}, vec2({_glsl_format(self.radius)}, {h_half}))"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

    def to_callable(self):
        is_dynamic = isinstance(self.radius, (str, Param)) or isinstance(self.height, (str, Param))
        if is_dynamic:
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")

        radius, height = self.radius, self.height
        h_half = height / 2.0
        def _callable_sharp(points: np.ndarray) -> np.ndarray:
            d = np.abs(np.array([np.linalg.norm(points[:, [0, 2]], axis=-1), points[:, 1]]).T) - np.array([radius, h_half])
            return np.minimum(np.maximum(d[:, 0], d[:, 1]), 0.0) + np.linalg.norm(np.maximum(d, 0.0), axis=-1)
        return _callable_sharp

def cylinder(radius: float = 0.5, height: float = 1.0) -> SDFNode:
    """
    Creates a cylinder centered at the origin, oriented along the Y-axis.

    Args:
        radius (float): The radius of the cylinder.
        height (float): The total height of the cylinder.
    """
    return Cylinder(radius, height)

class Cone(SDFNode):
    """Represents a cone or frustum primitive."""
    glsl_dependencies = {"primitives"}

    def __init__(self, height: float = 1.0, radius_base: float = 0.5, radius_top: float = 0.0):
        super().__init__()
        self.height, self.radius_base, self.radius_top = height, radius_base, radius_top
        
    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        h, r1 = _glsl_format(self.height), _glsl_format(self.radius_base)
        
        use_capped = False
        if isinstance(self.radius_top, (int, float)):
            if self.radius_top > 1e-6:
                use_capped = True
        else:
            use_capped = True
            
        if use_capped:
            dist_expr = f"sdCappedCone({ctx.p}, {h}, {r1}, {_glsl_format(self.radius_top)})"
        else:
            dist_expr = f"sdCone({ctx.p}, vec2({h}, {r1}))"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

    def to_callable(self):
        is_dynamic = isinstance(self.height, (str, Param)) or isinstance(self.radius_base, (str, Param)) or isinstance(self.radius_top, (str, Param))
        if is_dynamic:
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")
            
        h, r1, r2 = self.height, self.radius_base, self.radius_top

        use_capped = False
        if isinstance(r2, (int, float)):
            if r2 > 1e-6:
                use_capped = True
        else: 
            use_capped = True

        if use_capped:
            def _callable_capped(points: np.ndarray) -> np.ndarray:
                q_x = np.linalg.norm(points[:, [0, 2]], axis=-1)
                q = np.stack([q_x, points[:, 1]], axis=-1)
                k1 = np.array([r2, h])
                k2 = np.array([r2 - r1, 2.0 * h])
                ca_x_min = np.where(q[:, 1] < 0.0, r1, r2)
                ca_x = q[:, 0] - np.minimum(q[:, 0], ca_x_min)
                ca_y = np.abs(q[:, 1]) - h
                ca = np.stack([ca_x, ca_y], axis=-1)
                k1_q = k1 - q
                dot_k2k2 = np.dot(k2, k2)
                clamp_val = np.clip(np.sum(k1_q * k2, axis=-1) / (dot_k2k2 + 1e-9), 0.0, 1.0)
                cb = q - k1 + k2 * clamp_val[:, np.newaxis]
                s = np.where((cb[:, 0] < 0.0) & (ca[:, 1] < 0.0), -1.0, 1.0)
                return s * np.sqrt(np.minimum(np.sum(ca * ca, axis=-1), np.sum(cb * cb, axis=-1)))
            return _callable_capped
        else: # Sharp cone
            def _callable_sharp(points: np.ndarray) -> np.ndarray:
                q = np.stack([np.linalg.norm(points[:, [0, 2]], axis=-1), points[:, 1]], axis=-1)
                w = np.array([r1, h])
                a = q - w * np.clip(np.dot(q, w) / np.dot(w, w), 0.0, 1.0)[:, np.newaxis]
                b = q - np.stack([np.zeros(len(q)), np.clip(q[:, 1], 0.0, h)], axis=-1)
                k = np.sign(r1)
                d = np.minimum(np.sum(a*a, axis=-1), np.sum(b*b, axis=-1))
                s = np.maximum(k * (q[:, 0] * w[1] - q[:, 1] * w[0]), k * (q[:, 1] - h))
                return np.sqrt(d) * np.sign(s)
            return _callable_sharp

def cone(height: float = 1.0, radius_base: float = 0.5, radius_top: float = 0.0) -> SDFNode:
    """
    Creates a cone or frustum centered at the origin, oriented along the Y-axis.

    Args:
        height (float): Total height.
        radius_base (float): Bottom radius.
        radius_top (float): Top radius.
    """
    return Cone(height, radius_base, radius_top)

class Plane(SDFNode):
    """Represents an infinite plane."""
    glsl_dependencies = {"primitives"}

    def __init__(self, normal, point=(0,0,0)):
        super().__init__()
        self.normal = np.array(normal)
        if np.linalg.norm(self.normal) > 0:
            self.normal = self.normal / np.linalg.norm(self.normal)
        self.point = np.array(point)
        self.offset = -np.dot(self.point, self.normal)
        
    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        n = self.normal
        dist_expr = f"sdPlane({ctx.p}, vec4({n[0]}, {n[1]}, {n[2]}, {_glsl_format(self.offset)}))"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

    def to_callable(self):
        if isinstance(self.offset, (str, Param)):
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")
        normal, offset = self.normal, self.offset
        def _callable(points: np.ndarray) -> np.ndarray:
            return np.dot(points, normal) + offset
        return _callable

def plane(normal, point=(0,0,0)) -> SDFNode:
    """
    Creates an infinite plane defined by a normal and a point.

    Args:
        normal (tuple): Normal vector.
        point (tuple): A point on the plane. Defaults to origin.
    """
    return Plane(normal, point)

class Octahedron(SDFNode):
    """Represents an octahedron."""
    glsl_dependencies = {"primitives"}

    def __init__(self, size: float = 1.0):
        super().__init__()
        self.size = size
        
    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        dist_expr = f"sdOctahedron({ctx.p}, {_glsl_format(self.size)})"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

    def to_callable(self):
        if isinstance(self.size, (str, Param)):
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")
        size = self.size
        def _callable(points: np.ndarray) -> np.ndarray:
            return (np.sum(np.abs(points), axis=-1) - size) * 0.57735027
        return _callable

def octahedron(size: float = 1.0) -> SDFNode:
    """
    Creates an octahedron centered at the origin.

    Args:
        size (float, optional): The size of the octahedron, corresponding to the
                                distance from the center to the center of a face.
                                Defaults to 1.0.    
    """
    return Octahedron(size)

class Ellipsoid(SDFNode):
    """Represents an ellipsoid."""
    glsl_dependencies = {"primitives"}

    def __init__(self, radii: tuple = (1.0, 0.5, 0.5)):
        super().__init__()
        self.radii = np.array(radii)
        
    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        r = [_glsl_format(v) for v in self.radii]
        dist_expr = f"sdEllipsoid({ctx.p}, vec3({r[0]}, {r[1]}, {r[2]}))"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

    def to_callable(self):
        if any(isinstance(v, (str, Param)) for v in self.radii):
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")
        radii = self.radii
        def _callable(points: np.ndarray) -> np.ndarray:
            k0 = np.linalg.norm(points / radii, axis=-1)
            k1 = np.linalg.norm(points / (radii * radii), axis=-1)
            return k0 * (k0 - 1.0) / (k1 + 1e-9)
        return _callable

def ellipsoid(radii=(1.0, 0.5, 0.5)) -> SDFNode:
    """
    Creates an ellipsoid centered at the origin.

    Args:
        radii (tuple): Radii along (X, Y, Z).
    """
    return Ellipsoid(tuple(radii))
    
class Circle(SDFNode):
    """Represents a 2D circle primitive."""
    glsl_dependencies = {"primitives"}

    def __init__(self, radius: float = 1.0):
        super().__init__()
        self.radius = radius

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        dist_expr = f"sdCircle({ctx.p}.xy, {_glsl_format(self.radius)})"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

    def to_callable(self):
        if isinstance(self.radius, (str, Param)):
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")
        r_val = self.radius
        def _callable(points: np.ndarray) -> np.ndarray:
            return np.linalg.norm(points[:, :2], axis=-1) - r_val
        return _callable

def circle(radius: float = 1.0) -> SDFNode:
    """
    Creates a 2D circle in the XY plane.

    Args:
        radius (float): Radius of the circle.
    """
    return Circle(radius)

class Rectangle(SDFNode):
    """Represents a 2D rectangle primitive."""
    glsl_dependencies = {"primitives"}

    def __init__(self, size: tuple = (1.0, 1.0)):
        super().__init__()
        self.size = np.array(size)

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        s = [_glsl_format(v) for v in self.size]
        size_vec = f"vec2({s[0]}/2.0, {s[1]}/2.0)"
        dist_expr = f"sdRectangle({ctx.p}.xy, {size_vec})"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

    def to_callable(self):
        if any(isinstance(v, (str, Param)) for v in self.size):
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")
        half_size = self.size / 2.0
        def _callable(points: np.ndarray) -> np.ndarray:
            q = np.abs(points[:, :2]) - half_size
            return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.maximum(q[:, 0], q[:, 1]), 0.0)
        return _callable

def rectangle(size=1.0) -> SDFNode:
    """
    Creates a 2D rectangle in the XY plane.

    Args:
        size (float): Size of the rectangle.
    """
    if isinstance(size, (int, float, str, Param)):
        size = (size, size)
    return Rectangle(tuple(size))