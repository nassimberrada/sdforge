import numpy as np
import uuid
import atexit
from .core import SDFNode, GLSLContext
from .utils import _glsl_format, Param
from .compositors import Compositor

_MODERNGL_AVAILABLE = False
try:
    import moderngl
    import glfw
    _MODERNGL_AVAILABLE = True
except ImportError:
    pass

class Primitive(SDFNode):
    """
    A generic class for all standard SDF primitives.
    Wraps GLSL generation and NumPy execution in a unified structure.
    """
    glsl_dependencies = {"primitives"}

    def __init__(self, name, glsl_func, params, py_func_generator, ndim=3):
        super().__init__()
        self.name = name
        self.glsl_func = glsl_func
        self.params = params
        self._func_generator = py_func_generator
        self.ndim = ndim
        self._param_values = params

    def _get_glsl_args(self, profile_mode=False):
        """Formats parameters for GLSL."""
        formatted = []
        for p in self.params:
            if isinstance(p, (tuple, list, np.ndarray)):
                components = [_glsl_format(v) for v in p]
                if len(components) == 2: formatted.append(f"vec2({components[0]}, {components[1]})")
                elif len(components) == 3: formatted.append(f"vec3({components[0]}, {components[1]}, {components[2]})")
                elif len(components) == 4: formatted.append(f"vec4({components[0]}, {components[1]}, {components[2]}, {components[3]})")
            else:
                formatted.append(_glsl_format(p))
        return formatted

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        args = ", ".join(self._get_glsl_args())
        
        if self.ndim == 2:
            profile_expr = f"{self.glsl_func}({ctx.p}.xy, {args})"
            dist_expr = f"max({profile_expr}, abs({ctx.p}.z) - 0.001)"
        else:
            dist_expr = f"{self.glsl_func}({ctx.p}, {args})"
            
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

    def to_profile_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        args = ", ".join(self._get_glsl_args())
        
        if self.ndim == 2:
            dist_expr = f"{self.glsl_func}({ctx.p}.xy, {args})"
        else:
            p_flat = f"vec3({ctx.p}.x, {ctx.p}.y, 0.0)"
            dist_expr = f"{self.glsl_func}({p_flat}, {args})"
            
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

    def to_callable(self):
        def check_dynamic(val):
            if isinstance(val, (str, Param)): return True
            if isinstance(val, (list, tuple, np.ndarray)):
                return any(check_dynamic(x) for x in val)
            return False

        if any(check_dynamic(p) for p in self.params):
            raise TypeError("Cannot save mesh with animated parameters.")

        py_func = self._func_generator(*self.params)
        
        if self.ndim == 2:
            return lambda points: np.maximum(py_func(points), np.abs(points[:, 2]) - 0.001)
        return py_func

    def to_profile_callable(self):
        if any(isinstance(p, (str, Param)) for p in self.params):
            raise TypeError("Cannot save mesh with animated parameters.")

        py_func = self._func_generator(*self.params)

        if self.ndim == 2:
            return lambda points: py_func(np.column_stack([points[:,0], points[:,1], np.zeros(len(points))]))
        else:
            return lambda points: py_func(np.column_stack([points[:,0], points[:,1], np.zeros(len(points))]))

def _sphere(radius):
    return lambda p: np.linalg.norm(p, axis=-1) - radius

def _box(half_size):
    hs = np.array(half_size)
    def f(p):
        q = np.abs(p) - hs
        return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.max(q, axis=-1), 0.0)
    return f

def _hex_prism(params):
    radius, half_height = params
    k = np.array([-0.8660254, 0.5, 0.57735026])
    def f(points):
        p = np.abs(points).astype(float)
        dot = p[:,0]*k[0] + p[:,1]*k[1]
        min_dot = np.minimum(dot, 0.0)
        p[:,0] -= 2.0 * min_dot * k[0]
        p[:,1] -= 2.0 * min_dot * k[1]
        clamp_x = np.clip(p[:,0], -k[2]*radius, k[2]*radius)
        vec_sub = p[:, :2] - np.stack([clamp_x, np.full_like(p[:,0], radius)], axis=-1)
        d_x = np.linalg.norm(vec_sub, axis=-1) * np.sign(p[:,1]-radius)
        d_y = p[:,2] - half_height
        return np.minimum(np.maximum(d_x, d_y), 0.0) + np.linalg.norm(np.maximum(np.stack([d_x, d_y], axis=-1), 0.0), axis=-1)
    return f

def ramid(height):
    h = height
    m2 = h*h + 0.25
    def f(points):
        p = points.astype(float); p[:, 1] += h * 0.5; p[:, [0, 2]] = np.abs(p[:, [0, 2]])
        mask = p[:,2] > p[:,0]; p[mask] = p[mask][:, [2, 1, 0]]; p[:, [0, 2]] -= 0.5
        q = np.stack([p[:,2], h*p[:,1] - 0.5*p[:,0], h*p[:,0] + 0.5*p[:,1]], axis=-1)
        s = np.maximum(-q[:,0], 0.0)
        t = np.clip((q[:,1] - 0.5*p[:,2]) / (m2 + 0.25), 0.0, 1.0)
        a = m2 * (q[:,0] + s)**2 + q[:,1]**2
        b = m2 * (q[:,0] + 0.5*t)**2 + (q[:,1] - m2*t)**2
        d2 = np.where(np.minimum(q[:,1], -q[:,0]*m2 - q[:,1]*0.5) > 0.0, 0.0, np.minimum(a, b))
        return np.sqrt((d2 + q[:,2]**2) / m2) * np.sign(np.maximum(q[:,2], -p[:,1]))
    return f

def _torus(t):
    maj, min_ = t
    def f(p):
        q = np.array([np.linalg.norm(p[:, [0, 2]], axis=-1) - maj, p[:, 1]]).T
        return np.linalg.norm(q, axis=-1) - min_
    return f

def _line(start, end, radius, rounded):
    a, b = np.array(start), np.array(end)
    def f(p):
        pa, ba = p - a, b - a
        h = np.clip(np.dot(pa, ba) / np.dot(ba, ba), 0.0, 1.0)
        return np.linalg.norm(pa - ba * h[:, np.newaxis], axis=-1) - radius
    
    def f_capped(p):
        ba = b - a; pa = p - a
        baba = np.dot(ba, ba); paba = np.dot(pa, ba)
        x = np.linalg.norm(pa * baba - ba * paba[:, np.newaxis], axis=-1) - radius * baba
        y = np.abs(paba - baba * 0.5) - baba * 0.5
        d = np.where(np.maximum(x, y) < 0.0, -np.minimum(x*x, y*y*baba), (np.where(x > 0.0, x*x, 0.0) + np.where(y > 0.0, y*y*baba, 0.0)))
        return np.sign(d) * np.sqrt(np.abs(d)) / baba
        
    return f if rounded else f_capped

def _cylinder(params):
    radius, h = params
    def f(p):
        d = np.abs(np.array([np.linalg.norm(p[:, [0, 2]], axis=-1), p[:, 1]]).T) - np.array([radius, h])
        return np.minimum(np.maximum(d[:, 0], d[:, 1]), 0.0) + np.linalg.norm(np.maximum(d, 0.0), axis=-1)
    return f

def _cone(height, r1, r2=None):
    if r2 is not None:
        def f(p):
            q = np.stack([np.linalg.norm(p[:, [0, 2]], axis=-1), p[:, 1]], axis=-1)
            k1, k2 = np.array([r2, height]), np.array([r2 - r1, 2.0 * height])
            ca = np.stack([q[:, 0] - np.minimum(q[:, 0], np.where(q[:, 1] < 0.0, r1, r2)), np.abs(q[:, 1]) - height], axis=-1)
            cb = q - k1 + k2 * np.clip(np.sum((k1 - q) * k2, axis=-1) / (np.dot(k2, k2) + 1e-9), 0.0, 1.0)[:, np.newaxis]
            s = np.where((cb[:, 0] < 0.0) & (ca[:, 1] < 0.0), -1.0, 1.0)
            return s * np.sqrt(np.minimum(np.sum(ca * ca, axis=-1), np.sum(cb * cb, axis=-1)))
        return f
    else:
        def f(p):
            q = np.stack([np.linalg.norm(p[:, [0, 2]], axis=-1), p[:, 1]], axis=-1)
            w = np.array([r1, height])
            a = q - w * np.clip(np.dot(q, w) / np.dot(w, w), 0.0, 1.0)[:, np.newaxis]
            b = q - np.stack([np.zeros(len(q)), np.clip(q[:, 1], 0.0, height)], axis=-1)
            s = np.maximum(np.sign(r1) * (q[:, 0] * w[1] - q[:, 1] * w[0]), np.sign(r1) * (q[:, 1] - height))
            return np.sqrt(np.minimum(np.sum(a*a, axis=-1), np.sum(b*b, axis=-1))) * np.sign(s)
        return f
    
def _cone_vec2(params):
    h, r = params
    return _cone(h, r)

def _plane(normal, offset):
    n = np.array(normal)
    return lambda p: np.dot(p, n) + offset

def _octahedron(size):
    return lambda p: (np.sum(np.abs(p), axis=-1) - size) * 0.57735027

def _ellipsoid(radii):
    r = np.array(radii)
    def f(p):
        k0 = np.linalg.norm(p / r, axis=-1)
        k1 = np.linalg.norm(p / (r * r), axis=-1)
        return k0 * (k0 - 1.0) / (k1 + 1e-9)
    return f

def _bezier(p0, p1, p2, radius):

    def _bezier_distance_callable(A, B, C, r):
        if np.allclose(A, B) or np.allclose(B, C):
            ba = C - A
            baba = np.dot(ba, ba)
            def _callable_linear(points: np.ndarray) -> np.ndarray:
                pa = points - A
                h = np.clip(np.dot(pa, ba) / (baba + 1e-9), 0.0, 1.0)
                return np.linalg.norm(pa - ba * h[:, np.newaxis], axis=-1) - r
            return _callable_linear

        a = B - A
        b = A - 2.0*B + C
        c = a * 2.0
        
        def _callable(points: np.ndarray) -> np.ndarray:
            d = A[np.newaxis, :] - points
            kk = 1.0 / np.dot(b, b)
            kx = kk * np.dot(a, b)
            dot_d_b = np.dot(d, b)
            dot_d_a = np.dot(d, a)
            ky = kk * (2.0 * np.dot(a, a) + dot_d_b) / 3.0
            kz = kk * dot_d_a
            
            kx_full = 3.0 * kx
            ky_full = 3.0 * ky
            kz_full = kz

            p = ky_full - kx_full*kx_full / 3.0
            q = 2.0*kx_full*kx_full*kx_full/27.0 - kx_full*ky_full/3.0 + kz_full
            
            p3 = p*p*p
            discriminant = q*q/4.0 + p3/27.0
            
            t_candidates = np.zeros((points.shape[0], 3))
            num_roots = np.zeros(points.shape[0], dtype=int)
            
            mask_d_pos = discriminant > 0
            if np.any(mask_d_pos):
                d_sqrt = np.sqrt(discriminant[mask_d_pos])
                q_masked = q[mask_d_pos]
                u1 = -q_masked/2.0 + d_sqrt
                u2 = -q_masked/2.0 - d_sqrt
                u1 = np.cbrt(u1)
                u2 = np.cbrt(u2)
                t_candidates[mask_d_pos, 0] = u1 + u2 - kx_full/3.0
                num_roots[mask_d_pos] = 1

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


    return _bezier_distance_callable(np.array(p0), np.array(p1), np.array(p2), radius)

def _circle(radius):
    return lambda p: np.linalg.norm(p[:, :2], axis=-1) - radius

def _rectangle(half_size):
    hs = np.array(half_size)
    def f(p):
        q = np.abs(p[:, :2]) - hs
        return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.maximum(q[:, 0], q[:, 1]), 0.0)
    return f

def _triangle(radius):
    r, k = radius, np.sqrt(3.0)
    def f(points):
        p = points[:, :2].astype(np.float64).copy(); p[:,0] = np.abs(p[:,0]) - r; p[:,1] += r/k
        cond = (p[:,0] + k*p[:,1]) > 0.0
        px_new, py_new = (p[:,0] - k*p[:,1])/2.0, (-k*p[:,0] - p[:,1])/2.0
        p[cond, 0] = px_new[cond]; p[cond, 1] = py_new[cond]
        p[:,0] -= np.clip(p[:,0], -2.0*r, 0.0)
        return -np.linalg.norm(p, axis=-1) * np.sign(p[:,1])
    return f

def _trapezoid(r1, r2, he):
    k1, k2 = np.array([r2, he]), np.array([r2 - r1, 2.0 * he])
    def f(points):
        p = points[:, :2].astype(np.float64).copy(); p[:,0] = np.abs(p[:,0])
        ca = np.stack([p[:,0] - np.minimum(p[:,0], np.where(p[:,1] < 0.0, r1, r2)), np.abs(p[:,1]) - he], axis=-1)
        cb = p - k1 + np.outer(np.clip((np.dot(k1-p, k2)) / np.dot(k2, k2), 0.0, 1.0), k2)
        s = np.where((cb[:,0] < 0.0) & (ca[:,1] < 0.0), -1.0, 1.0)
        return s * np.sqrt(np.minimum(np.sum(ca**2, axis=1), np.sum(cb**2, axis=1)))
    return f

def sphere(radius: float = 1.0) -> Primitive:
    """
    Creates a sphere centered at the origin.

    Args:
        radius (float, optional): The radius of the sphere. Defaults to 1.0.
    """
    prim = Primitive("Sphere", "sdSphere", [radius], _sphere)
    prim.radius = radius
    return prim

def box(size=1.0) -> Primitive:
    """
    Creates a box centered at the origin.

    Args:
        size (float or tuple, optional): The size of the box. If a float,
                                         creates a cube. If a tuple, specifies
                                         (width, height, depth). Defaults to 1.0.
    """
    if isinstance(size, (int, float, str, Param)): size = (size, size, size)
    s = np.array(size)
    prim = Primitive("Box", "sdBox", [tuple(s/2.0)], _box)
    prim.size = size
    return prim

def hex_prism(radius: float = 1.0, height: float = 1.0) -> Primitive: 
    """
    Creates a hexagonal prism oriented along the Z axis.

    Args:
        radius (float): The radius (apothem) of the hexagon.
        height (float): The total height (depth) of the prism.
    """
    prim = Primitive("HexPrism", "sdHexPrism", [(radius, height/2.0)], _hex_prism)
    prim.radius, prim.height = radius, height
    return prim

def pyramid(height: float = 1.0) -> Primitive:
    """
    Creates a pyramid with a square base centered at (0,0,0).

    Args:
        height (float): The vertical height of the pyramid.
    """
    prim = Primitive("Pyramid", "sdPyramid", [height], ramid)
    prim.height = height
    return prim

def torus(radius_major: float = 1.0, radius_minor: float = 0.25) -> Primitive:
    """
    Creates a torus centered at the origin, oriented in the XZ plane.

    Args:
        radius_major (float): Distance from the origin to the center of the tube.
        radius_minor (float): Radius of the tube itself.
    """
    prim = Primitive("Torus", "sdTorus", [(radius_major, radius_minor)], _torus)
    prim.radius_major, prim.radius_minor = radius_major, radius_minor
    return prim

def line(start, end, radius: float = 0.1, rounded_caps: bool = True) -> Primitive:
    """
    Creates a line segment between two points.

    Args:
        start (tuple): The starting point (x, y, z).
        end (tuple): The ending point (x, y, z).
        radius (float, optional): The thickness of the line. Defaults to 0.1.
        rounded_caps (bool, optional): Whether to round the ends. Defaults to True.
    """
    func = "sdCapsule" if rounded_caps else "sdCappedCylinder"
    prim = Primitive("Line", func, [start, end, radius], lambda s, e, r: _line(s, e, r, rounded_caps))
    prim.rounded_caps = rounded_caps
    return prim

def cylinder(radius: float = 0.5, height: float = 1.0) -> Primitive:
    """
    Creates a cylinder centered at the origin, oriented along the Y-axis.

    Args:
        radius (float): The radius of the cylinder.
        height (float): The total height of the cylinder.
    """
    prim = Primitive("Cylinder", "sdCylinder", [(radius, height/2.0)], _cylinder)
    prim.radius, prim.height = radius, height
    return prim

def cone(height: float = 1.0, radius_base: float = 0.5, radius_top: float = 0.0) -> Primitive:
    """
    Creates a cone or frustum centered at the origin, oriented along the Y-axis.

    Args:
        height (float): Total height.
        radius_base (float): Bottom radius.
        radius_top (float): Top radius.
    """
    use_capped = not isinstance(radius_top, (int, float)) or radius_top > 1e-6
    if use_capped:
        prim = Primitive("Cone", "sdCappedCone", [height, radius_base, radius_top], _cone)
    else:
        prim = Primitive("Cone", "sdCone", [(height, radius_base)], _cone_vec2)
    prim.height, prim.radius_base, prim.radius_top = height, radius_base, radius_top
    return prim

def plane(normal, point=(0,0,0)) -> Primitive:
    """
    Creates an infinite plane defined by a normal and a point.

    Args:
        normal (tuple): Normal vector.
        point (tuple): A point on the plane. Defaults to origin.
    """
    n = np.array(normal)
    if np.linalg.norm(n) > 0: n = n / np.linalg.norm(n)
    offset = -np.dot(np.array(point), n)
    prim = Primitive("Plane", "sdPlane", [(n[0], n[1], n[2], offset)], lambda p: _plane(n, offset))
    prim.offset = offset
    return prim

def octahedron(size: float = 1.0) -> Primitive:
    """
    Creates an octahedron centered at the origin.

    Args:
        size (float, optional): The size of the octahedron, corresponding to the
                                distance from the center to the center of a face.
                                Defaults to 1.0.    
    """
    return Primitive("Octahedron", "sdOctahedron", [size], _octahedron)

def ellipsoid(radii=(1.0, 0.5, 0.5)) -> Primitive:
    """
    Creates an ellipsoid centered at the origin.

    Args:
        radii (tuple): Radii along (X, Y, Z).
    """
    return Primitive("Ellipsoid", "sdEllipsoid", [tuple(radii)], _ellipsoid)

def curve(p0, p1, p2, radius: float = 0.1) -> Primitive:
    """
    Creates a Quadratic Bezier tube (curve) defined by 3 points.

    Args:
        p0 (tuple): The starting point (x, y, z).
        p1 (tuple): The control point (x, y, z).
        p2 (tuple): The ending point (x, y, z).
        radius (float, optional): The thickness of the tube. Defaults to 0.1.
    """
    return Primitive("Bezier", "sdBezier", [p0, p1, p2, radius], _bezier)

def circle(radius: float = 1.0) -> Primitive:
    """
    Creates a 2D circle in the XY plane. 
    By default, this renders as a thin disc in 3D. 
    Use .extrude() to create a cylinder.

    Args:
        radius (float): Radius of the circle.
    """
    return Primitive("Circle", "sdCircle", [radius], _circle, ndim=2)

def rectangle(size=1.0) -> Primitive:
    """
    Creates a 2D rectangle in the XY plane.
    By default, this renders as a thin plate in 3D.
    Use .extrude() to create a box.

    Args:
        size (float): Size of the rectangle.
    """
    if isinstance(size, (int, float, str, Param)): size = (size, size)
    s = np.array(size)
    return Primitive("Rectangle", "sdRectangle", [tuple(s/2.0)], _rectangle, ndim=2)

def triangle(radius: float = 1.0) -> Primitive:
    """
    Creates a 2D equilateral triangle.
    
    Args:
        radius (float): The radius (apothem) of the triangle.
    """
    return Primitive("Triangle", "sdEquilateralTriangle", [radius], _triangle, ndim=2)

def trapezoid(bottom_width: float = 1.0, top_width: float = 0.5, height: float = 0.5) -> Primitive:
    """
    Creates an isosceles trapezoid.

    Args:
        bottom_width (float): Width at the bottom.
        top_width (float): Width at the top.
        height (float): Height of the trapezoid.
    """
    return Primitive("Trapezoid", "sdTrapezoid", [bottom_width/2.0, top_width/2.0, height/2.0], _trapezoid, ndim=2)

def polyline(points, radius: float = 0.1, closed: bool = False) -> SDFNode:
    """
    Creates a continuous chain of line segments (capsules) connecting the points.

    Args:
        points (list): A list of (x, y, z) coordinates.
        radius (float, optional): The thickness of the line. Defaults to 0.1.
        closed (bool, optional): If True, connects the last point back to the first. Defaults to False.
    """
    pts = np.array(points)
    num = len(pts) if closed else len(pts) - 1
    segments = []
    if num < 1:
        return sphere(0).translate((1e6, 0, 0))
    
    for i in range(num):
        p0 = pts[i]
        p1 = pts[(i + 1) % len(pts)]
        segments.append(line(p0, p1, radius))
    
    return Compositor(segments, op_type='union')

def polycurve(points, radius: float = 0.1, closed: bool = False) -> SDFNode:
    """
    Creates a smooth continuous curve passing near the points (except endpoints).
    
    Uses a Corner-Cutting / Chaikin / Quadratic B-Spline approach.
    - For open curves: Passes through the first and last points exactly.
    - For closed curves: Creates a smooth loop controlled by the points.

    Args:
        points (list): A list of (x, y, z) control points.
        radius (float, optional): The thickness of the curve tube. Defaults to 0.1.
        closed (bool, optional): If True, creates a closed loop. Defaults to False.
    """
    pts = np.array(points)
    n = len(pts)
    segments = []
    
    segs_data = []
    if n < 2: 
        return sphere(0).translate((1e6, 0, 0))
    if n == 2 and not closed:
        segs_data.append((pts[0], (pts[0] + pts[1]) / 2.0, pts[1]))
    elif closed:
        for i in range(n):
            segs_data.append(((pts[(i - 1) % n] + pts[i]) / 2.0, pts[i], (pts[i] + pts[(i + 1) % n]) / 2.0))
    else:
        segs_data.append((pts[0], pts[1], (pts[1] + pts[2]) / 2.0))
        for i in range(1, n - 2):
            segs_data.append(((pts[i] + pts[i+1]) / 2.0, pts[i+1], (pts[i+1] + pts[i+2]) / 2.0))
        segs_data.append(((pts[-2] + pts[-1]) / 2.0, pts[-1], pts[-1]))
        
    for A, B, C in segs_data:
        if np.allclose(B, C) or np.allclose(A, B):
            segments.append(line(A, C, radius))
        else:
            segments.append(curve(A, B, C, radius))
            
    return Compositor(segments, op_type='union')

class Forge(SDFNode):
    """An SDF object defined by a raw GLSL code snippet."""
    def __init__(self, glsl_code_body: str, uniforms: dict = None):
        super().__init__()
        if "return" not in glsl_code_body: glsl_code_body = f"return {glsl_code_body};"
        self.glsl_code_body = glsl_code_body
        self.uniforms = uniforms or {}
        self.unique_id = "forge_func_" + uuid.uuid4().hex[:8]
        self.glsl_dependencies = set()
    def _get_glsl_definition(self) -> str:
        uniform_params = "".join([f", in float {name}" for name in self.uniforms.keys()])
        return f"float {self.unique_id}(vec3 p{uniform_params}){{ {self.glsl_code_body} }}"
    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        ctx.definitions.add(self._get_glsl_definition())
        uniform_args = "".join([f", {name}" for name in self.uniforms.keys()])
        return ctx.new_variable('vec4', f"vec4({self.unique_id}({ctx.p}{uniform_args}), -1.0, 0.0, 0.0)")
    def _collect_uniforms(self, uniforms_dict: dict):
        uniforms_dict.update(self.uniforms)
        super()._collect_uniforms(uniforms_dict)
    def to_callable(self):
        if not _MODERNGL_AVAILABLE: raise ImportError("To create a callable for Forge objects, 'moderngl' and 'glfw' are required.")
        if self.uniforms: raise TypeError("Cannot create a callable for a Forge object with uniforms.")
        cls = self.__class__
        if not hasattr(cls, '_mgl_context'):
            if not glfw.init(): raise RuntimeError("glfw.init() failed")
            atexit.register(glfw.terminate)
            glfw.window_hint(glfw.VISIBLE, False)
            win = glfw.create_window(1, 1, "", None, None)
            glfw.make_context_current(win)
            cls._mgl_context = moderngl.create_context(require=430)
        ctx = cls._mgl_context
        func_def = self._get_glsl_definition()
        call_expr = f"{self.unique_id}(p[gid])"
        from .io import get_glsl_definitions
        library_code = get_glsl_definitions(frozenset(self.glsl_dependencies))
        compute_shader = ctx.compute_shader(f"""
        #version 430
        layout(local_size_x=256) in;
        layout(std430, binding=0) buffer points {{ vec3 p[]; }};
        layout(std430, binding=1) buffer distances {{ float d[]; }};
        {library_code}
        {func_def}
        void main() {{
            uint gid = gl_GlobalInvocationID.x;
            d[gid] = {call_expr};
        }}
        """)
        def _gpu_evaluator(points_np):
            points_np = np.array(points_np, dtype='f4')
            num_points = len(points_np)
            padded_points = np.zeros((num_points, 4), dtype='f4')
            padded_points[:, :3] = points_np
            point_buffer = ctx.buffer(padded_points.tobytes())
            dist_buffer = ctx.buffer(reserve=num_points * 4)
            point_buffer.bind_to_storage_buffer(0)
            dist_buffer.bind_to_storage_buffer(1)
            compute_shader.run(group_x=(num_points + 255) // 256)
            return np.frombuffer(dist_buffer.read(), dtype='f4')
        return _gpu_evaluator

class Sketch:
    """Builder for 2D profiles using a path-based interface."""
    def __init__(self, start=(0.0, 0.0)):
        self._segments = []
        self._current_pos = np.array(start, dtype=float)
        self._start_pos = self._current_pos.copy()
    def _to_vec3(self, p2d):
        if len(p2d) == 2: return np.array([p2d[0], p2d[1], 0.0])
        return np.array(p2d)
    def move_to(self, x, y):
        self._current_pos = np.array([x, y], dtype=float)
        if not self._segments: self._start_pos = self._current_pos
        return self
    def line_to(self, x, y):
        end_pos = np.array([x, y], dtype=float)
        self._segments.append({'type': 'line', 'start': self._to_vec3(self._current_pos), 'end': self._to_vec3(end_pos)})
        self._current_pos = end_pos
        return self
    def curve_to(self, x, y, control):
        end_pos = np.array([x, y], dtype=float)
        self._segments.append({'type': 'bezier', 'start': self._to_vec3(self._current_pos), 'control': self._to_vec3(control), 'end': self._to_vec3(end_pos)})
        self._current_pos = end_pos
        return self
    def close(self):
        if not np.allclose(self._current_pos, self._start_pos): self.line_to(self._start_pos[0], self._start_pos[1])
        return self
    def to_sdf(self, stroke_radius=0.05) -> SDFNode:
        nodes = []
        for seg in self._segments:
            if seg['type'] == 'line': nodes.append(line(seg['start'], seg['end'], radius=stroke_radius))
            elif seg['type'] == 'bezier': nodes.append(curve(seg['start'], seg['control'], seg['end'], radius=stroke_radius))
        if not nodes: return sphere(0).translate((1e5, 0, 0))
        return Compositor(nodes, op_type='union')