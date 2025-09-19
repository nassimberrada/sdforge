import numpy as np
import uuid
from functools import reduce, lru_cache
from pathlib import Path
import atexit

# --- Constants ---
X = np.array([1, 0, 0])
Y = np.array([0, 1, 0])
Z = np.array([0, 0, 1])

# --- Optional GPU Dependency Check for Forge ---
_MODERNGL_AVAILABLE = False
try:
    import moderngl
    import glfw
    _MODERNGL_AVAILABLE = True
except ImportError:
    pass


# --- GLSL File Loader Utility ---
@lru_cache(maxsize=None)
def _get_glsl_content(filename: str) -> str:
    """Cached reader for GLSL library files."""
    glsl_dir = Path(__file__).parent / 'glsl' / 'sdf'
    try:
        with open(glsl_dir / filename, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return ""

# --- Helper for formatting GLSL parameters ---
def _glsl_format(val):
    """Formats a Python value for injection into a GLSL string."""
    if isinstance(val, str):
        return val  # Assume it's a raw GLSL expression
    return f"{float(val)}"


# --- Camera ---

class Camera:
    """
    Represents a camera in the scene, allowing for static or animated positioning.
    """
    def __init__(self, position=(5, 4, 5), target=(0, 0, 0), zoom=1.0):
        """
        Initializes the camera.

        Args:
            position (tuple, optional): The position of the camera in 3D space.
                                        Components can be numbers or GLSL expressions (str).
                                        Defaults to (5, 4, 5).
            target (tuple, optional): The point the camera is looking at.
                                      Components can be numbers or GLSL expressions (str).
                                      Defaults to (0, 0, 0).
            zoom (float or str, optional): The zoom level. Defaults to 1.0.
        """
        self.position = position
        self.target = target
        self.zoom = zoom


# --- Light ---

class Light:
    """
    Represents light and shadow properties for the scene.
    """
    def __init__(self, position=None, ambient_strength=0.1, shadow_softness=8.0, ao_strength=3.0):
        """
        Initializes the scene light.

        Args:
            position (tuple, optional): The position of the light source.
                                        Components can be numbers or GLSL expressions (str).
                                        If None, the light is positioned at the camera (headlight).
                                        Defaults to None.
            ambient_strength (float or str, optional): The minimum brightness for surfaces. Defaults to 0.1.
            shadow_softness (float or str, optional): How soft the shadows are. Higher is softer. Defaults to 8.0.
            ao_strength (float or str, optional): Strength of ambient occlusion. Defaults to 3.0.
        """
        self.position = position
        self.ambient_strength = ambient_strength
        self.shadow_softness = shadow_softness
        self.ao_strength = ao_strength


# --- Base Class ---

class SDFObject:
    """Base class for all SDF objects, defining the core interface."""
    def __init__(self):
        self.uuid = uuid.uuid4()

    def to_glsl(self) -> str: raise NotImplementedError
    def to_callable(self): raise NotImplementedError
    def get_glsl_definitions(self) -> list: return []
    def _collect_materials(self, materials): pass
    def __or__(self, other): return Union(self, other)
    def __and__(self, other): return Intersection(self, other)
    def __sub__(self, other): return Difference(self, other)
    def xor(self, other): return Xor(self, other)
    def translate(self, offset): return Translate(self, np.array(offset))
    def scale(self, factor): return Scale(self, factor)
    def orient(self, axis): return Orient(self, np.array(axis))
    def rotate(self, axis, angle): return Rotate(self, np.array(axis), angle)
    def twist(self, k): return Twist(self, k)
    def shear_xy(self, shear): return ShearXY(self, np.array(shear))
    def shear_xz(self, shear): return ShearXZ(self, np.array(shear))
    def shear_yz(self, shear): return ShearYZ(self, np.array(shear))
    def bend_x(self, k): return BendX(self, k)
    def bend_y(self, k): return BendY(self, k)
    def bend_z(self, k): return BendZ(self, k)
    def repeat(self, spacing): return Repeat(self, np.array(spacing))
    def limited_repeat(self, spacing, limits): return LimitedRepeat(self, np.array(spacing), np.array(limits))
    def polar_repeat(self, repetitions): return PolarRepeat(self, repetitions)
    def mirror(self, axes): return Mirror(self, np.array(axes))
    def smooth_union(self, other, k): return SmoothUnion(self, other, k)
    def smooth_intersection(self, other, k): return SmoothIntersection(self, other, k)
    def smooth_difference(self, other, k): return SmoothDifference(self, other, k)
    def color(self, r, g, b): return Material(self, (r, g, b))
    def round(self, radius): return Round(self, radius)
    def onion(self, thickness): return Onion(self, thickness)
    def elongate(self, h): return Elongate(self, np.array(h))
    def displace(self, displacement_glsl): return Displace(self, displacement_glsl)
    def extrude(self, height): return Extrude(self, height)


# --- Primitives ---

class Sphere(SDFObject):
    def __init__(self, r=1.0):
        super().__init__()
        self.r = r
    def to_glsl(self) -> str: return f"vec4(sdSphere(p, {_glsl_format(self.r)}), -1.0, 0.0, 0.0)"
    def to_callable(self):
        if isinstance(self.r, str): raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        return lambda p: np.linalg.norm(p, axis=-1) - self.r

def sphere(r=1.0) -> SDFObject: return Sphere(r)

class Box(SDFObject):
    def __init__(self, size=1.0):
        super().__init__()
        if isinstance(size, (int, float, str)): size = (size, size, size)
        self.size = size
    def to_glsl(self) -> str:
        s = []
        for v in self.size:
            if isinstance(v, str): s.append(f"({v})")
            else: s.append(_glsl_format(v / 2.0))
        return f"vec4(sdBox(p, vec3({s[0]}, {s[1]}, {s[2]})), -1.0, 0.0, 0.0)"
    def to_callable(self):
        if any(isinstance(v, str) for v in self.size): raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        size_arr = np.array(self.size)
        def _callable(p):
            q = np.abs(p) - size_arr / 2.0
            return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(np.max(q, axis=-1), 0)
        return _callable

def box(size=1.0) -> SDFObject: return Box(size)

class RoundedBox(SDFObject):
    def __init__(self, size=1.0, radius=0.1):
        super().__init__()
        if isinstance(size, (int, float, str)): size = (size, size, size)
        self.size, self.radius = size, radius
    def to_glsl(self) -> str:
        s = []
        for v in self.size:
            if isinstance(v, str): s.append(f"({v})")
            else: s.append(_glsl_format(v / 2.0))
        r = _glsl_format(self.radius)
        return f"vec4(sdRoundedBox(p, vec3({s[0]}, {s[1]}, {s[2]}), {r}), -1.0, 0.0, 0.0)"
    def to_callable(self):
        if any(isinstance(v, str) for v in self.size) or isinstance(self.radius, str):
            raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        size_arr = np.array(self.size)
        def _callable(p):
            q = np.abs(p) - size_arr / 2.0
            return np.linalg.norm(np.maximum(q, 0), axis=-1) - self.radius
        return _callable

def rounded_box(size=1.0, radius=0.1) -> SDFObject: return RoundedBox(size, radius)

class Torus(SDFObject):
    def __init__(self, major=1.0, minor=0.25):
        super().__init__()
        self.major, self.minor = major, minor
    def to_glsl(self) -> str: return f"vec4(sdTorus(p, vec2({_glsl_format(self.major)}, {_glsl_format(self.minor)})), -1.0, 0.0, 0.0)"
    def to_callable(self):
        if isinstance(self.major, str) or isinstance(self.minor, str):
            raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        def _callable(p):
            q = np.array([np.linalg.norm(p[:, [0, 2]], axis=-1) - self.major, p[:, 1]]).T
            return np.linalg.norm(q, axis=-1) - self.minor
        return _callable

def torus(major=1.0, minor=0.25) -> SDFObject: return Torus(major, minor)

class Capsule(SDFObject):
    def __init__(self, a, b, radius=0.1):
        super().__init__()
        self.a, self.b, self.radius = np.array(a), np.array(b), radius
    def to_glsl(self) -> str:
        a, b, r = self.a, self.b, _glsl_format(self.radius)
        return f"vec4(sdCapsule(p, vec3({a[0]},{a[1]},{a[2]}), vec3({b[0]},{b[1]},{b[2]}), {r}), -1.0, 0.0, 0.0)"
    def to_callable(self):
        if isinstance(self.radius, str): raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        def _callable(p):
            pa = p - self.a; ba = self.b - self.a
            h = np.clip(np.dot(pa, ba) / np.dot(ba, ba), 0.0, 1.0)
            return np.linalg.norm(pa - ba * h[:, np.newaxis], axis=-1) - self.radius
        return _callable

def capsule(a, b, radius=0.1) -> SDFObject: return Capsule(a, b, radius)

class Cylinder(SDFObject):
    def __init__(self, radius=0.5, height=1.0):
        super().__init__()
        self.radius, self.height = radius, height
    def to_glsl(self) -> str:
        r = _glsl_format(self.radius)
        h = _glsl_format(self.height / 2.0) if not isinstance(self.height, str) else f"({self.height})/2.0"
        return f"vec4(sdCylinder(p, vec2({r}, {h})), -1.0, 0.0, 0.0)"
    def to_callable(self):
        if isinstance(self.radius, str) or isinstance(self.height, str):
            raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        def _callable(p):
            r, h_half = self.radius, self.height / 2.0
            d = np.abs(np.array([np.linalg.norm(p[:, [0, 2]], axis=-1), p[:, 1]]).T) - np.array([r, h_half])
            return np.minimum(np.maximum(d[:, 0], d[:, 1]), 0.0) + np.linalg.norm(np.maximum(d, 0.0), axis=-1)
        return _callable

def cylinder(radius=0.5, height=1.0) -> SDFObject: return Cylinder(radius, height)

class Cone(SDFObject):
    def __init__(self, height=1.0, radius=0.5):
        super().__init__()
        self.height, self.radius = height, radius
    def to_glsl(self) -> str: return f"vec4(sdCone(p, vec2({_glsl_format(self.height)}, {_glsl_format(self.radius)})), -1.0, 0.0, 0.0)"
    def to_callable(self):
        if isinstance(self.height, str) or isinstance(self.radius, str):
            raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        def _callable(p):
            c = np.array([self.height, self.radius])
            q = np.array([np.linalg.norm(p[:, [0, 2]], axis=-1), p[:, 1]]).T; a = c - q; b = q - c * np.array([1, -1])
            k = np.sign(c[1]); d = np.minimum(np.sum(a*a, axis=-1), np.sum(b*b, axis=-1))
            s = np.maximum(k * (q[:,0]*c[1] - q[:,1]*c[0]), k * (q[:,1] - c[1]))
            return np.sqrt(d) * np.sign(s)
        return _callable

def cone(height=1.0, radius=0.5) -> SDFObject: return Cone(height, radius)

class Plane(SDFObject):
    def __init__(self, normal=Y, offset=0):
        super().__init__()
        self.normal, self.offset = np.array(normal), offset
    def to_glsl(self) -> str: n = self.normal; return f"vec4(sdPlane(p, vec4({n[0]}, {n[1]}, {n[2]}, {_glsl_format(self.offset)})), -1.0, 0.0, 0.0)"
    def to_callable(self):
        if isinstance(self.offset, str): raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        return lambda p: np.dot(p, self.normal) + self.offset

def plane(normal=Y, offset=0) -> SDFObject: return Plane(normal, offset)

class HexPrism(SDFObject):
    def __init__(self, radius=1.0, height=0.1):
        super().__init__()
        self.radius, self.height = radius, height
    def to_glsl(self) -> str: return f"vec4(sdHexPrism(p, vec2({_glsl_format(self.radius)}, {_glsl_format(self.height)})), -1.0, 0.0, 0.0)"
    def to_callable(self):
        if isinstance(self.radius, str) or isinstance(self.height, str):
            raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        h_x, h_y = self.radius, self.height
        k = np.array([-0.8660254, 0.5, 0.57735])
        def _callable(p):
            p = np.abs(p)
            dot_val = np.dot(p[:, :2], k[:2])
            min_dot = np.minimum(dot_val, 0.0)
            p[:, :2] -= 2.0 * min_dot[:, np.newaxis] * k[:2]
            clamped_x = np.clip(p[:, 0], -k[2] * h_x, k[2] * h_x)
            vec_to_len = p[:, :2] - np.stack([clamped_x, np.full_like(clamped_x, h_x)], axis=-1)
            len_val = np.linalg.norm(vec_to_len, axis=-1)
            d_x = len_val * np.sign(p[:, 1] - h_x)
            d_y = p[:, 2] - h_y
            d = np.stack([d_x, d_y], axis=-1)
            max_d = np.maximum(d[:, 0], d[:, 1])
            return np.minimum(max_d, 0.0) + np.linalg.norm(np.maximum(d, 0.0), axis=-1)
        return _callable

def hex_prism(radius=1.0, height=0.1) -> SDFObject: return HexPrism(radius, height)

class Octahedron(SDFObject):
    def __init__(self, size=1.0):
        super().__init__()
        self.size = size
    def to_glsl(self) -> str: return f"vec4(sdOctahedron(p, {_glsl_format(self.size)}), -1.0, 0.0, 0.0)"
    def to_callable(self):
        if isinstance(self.size, str): raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        return lambda p: (np.sum(np.abs(p), axis=-1) - self.size) * 0.57735027

def octahedron(size=1.0) -> SDFObject: return Octahedron(size)

class Ellipsoid(SDFObject):
    def __init__(self, radii):
        super().__init__()
        self.radii = radii
    def to_glsl(self) -> str:
        r = [_glsl_format(v) for v in self.radii]
        return f"vec4(sdEllipsoid(p, vec3({r[0]}, {r[1]}, {r[2]})), -1.0, 0.0, 0.0)"
    def to_callable(self):
        if any(isinstance(v, str) for v in self.radii):
            raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        radii_arr = np.array(self.radii)
        def _callable(p):
            k0 = np.linalg.norm(p / radii_arr, axis=-1)
            k1 = np.linalg.norm(p / (radii_arr * radii_arr), axis=-1)
            return k0 * (k0 - 1.0) / (k1 + 1e-9)
        return _callable

def ellipsoid(radii) -> SDFObject: return Ellipsoid(radii)

class BoxFrame(SDFObject):
    def __init__(self, size, edge_radius=0.1):
        super().__init__()
        if isinstance(size, (int, float, str)): size = (size, size, size)
        self.size, self.edge_radius = size, edge_radius
    def to_glsl(self) -> str:
        s = 'vec3(' + ','.join([_glsl_format(v) for v in self.size]) + ')'
        e = _glsl_format(self.edge_radius)
        return f"vec4(sdBoxFrame(p, {s}, {e}), -1.0, 0.0, 0.0)"
    def to_callable(self):
        if any(isinstance(v, str) for v in self.size) or isinstance(self.edge_radius, str):
            raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        b = np.array(self.size)
        e = self.edge_radius
        def _sdBox_np(p, b_inner):
            q = np.abs(p) - b_inner
            return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.max(q, axis=-1), 0.0)
        def _callable(p):
            p_abs = np.abs(p) - b
            q_abs = np.abs(p_abs + e) - e
            d1 = _sdBox_np(np.stack([p_abs[:,0], q_abs[:,1], q_abs[:,2]], axis=-1), np.array([0,0,0]))
            d2 = _sdBox_np(np.stack([q_abs[:,0], p_abs[:,1], q_abs[:,2]], axis=-1), np.array([0,0,0]))
            d3 = _sdBox_np(np.stack([q_abs[:,0], q_abs[:,1], p_abs[:,2]], axis=-1), np.array([0,0,0]))
            return np.minimum(np.minimum(d1, d2), d3)
        return _callable

def box_frame(size, edge_radius=0.1) -> SDFObject: return BoxFrame(size, edge_radius)

class CappedTorus(SDFObject):
    def __init__(self, angle_sc, major_radius, minor_radius):
        super().__init__()
        self.angle_sc = np.array(angle_sc)
        self.major_radius = major_radius
        self.minor_radius = minor_radius
    def to_glsl(self) -> str:
        sc = f"vec2({_glsl_format(self.angle_sc[0])}, {_glsl_format(self.angle_sc[1])})"
        ra = _glsl_format(self.major_radius)
        rb = _glsl_format(self.minor_radius)
        return f"vec4(sdCappedTorus(p, {sc}, {ra}, {rb}), -1.0, 0.0, 0.0)"
    def to_callable(self):
        if isinstance(self.major_radius, str) or isinstance(self.minor_radius, str):
            raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        sc, ra, rb = self.angle_sc, self.major_radius, self.minor_radius
        def _callable(p):
            p_x_abs = np.abs(p[:, 0])
            p_xy = np.stack([p_x_abs, p[:, 1]], axis=-1)
            cond = sc[1] * p_x_abs > sc[0] * p[:, 1]
            k = np.where(cond, np.dot(p_xy, sc), np.linalg.norm(p_xy, axis=-1))
            p_with_abs = np.stack([p_x_abs, p[:, 1], p[:, 2]], axis=-1)
            dot_p = np.sum(p_with_abs * p_with_abs, axis=-1)
            return np.sqrt(dot_p + ra*ra - 2.0*ra*k) - rb
        return _callable

def capped_torus(angle_sc, major_radius=1.0, minor_radius=0.25) -> SDFObject: return CappedTorus(angle_sc, major_radius, minor_radius)

class Link(SDFObject):
    def __init__(self, length, radius1, radius2):
        super().__init__()
        self.length, self.radius1, self.radius2 = length, radius1, radius2
    def to_glsl(self) -> str:
        return f"vec4(sdLink(p, {_glsl_format(self.length)}, {_glsl_format(self.radius1)}, {_glsl_format(self.radius2)}), -1.0, 0.0, 0.0)"
    def to_callable(self):
        if isinstance(self.length, str) or isinstance(self.radius1, str) or isinstance(self.radius2, str):
            raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        le, r1, r2 = self.length, self.radius1, self.radius2
        def _callable(p):
            q_y = np.maximum(np.abs(p[:, 1]) - le, 0.0)
            q_xy = np.stack([p[:, 0], q_y], axis=-1)
            q_xy_len = np.linalg.norm(q_xy, axis=-1)
            vec = np.stack([q_xy_len - r1, p[:, 2]], axis=-1)
            return np.linalg.norm(vec, axis=-1) - r2
        return _callable

def link(length=1.0, radius1=0.3, radius2=0.1) -> SDFObject: return Link(length, radius1, radius2)

class CappedCylinder(SDFObject):
    def __init__(self, a, b, radius):
        super().__init__()
        self.a, self.b, self.radius = np.array(a), np.array(b), radius
    def to_glsl(self) -> str:
        a, b, r = self.a, self.b, _glsl_format(self.radius)
        return f"vec4(sdCappedCylinder(p, vec3({a[0]},{a[1]},{a[2]}), vec3({b[0]},{b[1]},{b[2]}), {r}), -1.0, 0.0, 0.0)"
    def to_callable(self):
        if isinstance(self.radius, str): raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        a, b, r = self.a, self.b, self.radius
        def _callable(p):
            ba = b - a
            pa = p - a
            baba = np.dot(ba, ba)
            paba = np.dot(pa, ba)
            x = np.linalg.norm(pa * baba - ba * paba[:, np.newaxis], axis=-1) - r * baba
            y = np.abs(paba - baba * 0.5) - baba * 0.5
            x2 = x*x
            y2 = y*y*baba
            d_inner = np.where(np.maximum(x, y) < 0.0, -np.minimum(x2, y2), (np.where(x > 0.0, x2, 0.0) + np.where(y > 0.0, y2, 0.0)))
            return np.sign(d_inner) * np.sqrt(np.abs(d_inner)) / baba
        return _callable

def capped_cylinder(a, b, radius=0.1) -> SDFObject: return CappedCylinder(a, b, radius)

class RoundedCylinder(SDFObject):
    def __init__(self, radius, round_radius, height):
        super().__init__()
        self.radius, self.round_radius, self.height = radius, round_radius, height
    def to_glsl(self) -> str:
        ra = _glsl_format(self.radius)
        rb = _glsl_format(self.round_radius)
        h = _glsl_format(self.height)
        return f"vec4(sdRoundedCylinder(p, {ra}, {rb}, {h}), -1.0, 0.0, 0.0)"
    def to_callable(self):
        if isinstance(self.radius, str) or isinstance(self.round_radius, str) or isinstance(self.height, str):
            raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        ra, rb, h = self.radius, self.round_radius, self.height
        def _callable(p):
            d_x = np.linalg.norm(p[:, [0, 2]], axis=-1) - 2.0 * ra + rb
            d_y = np.abs(p[:, 1]) - h
            d = np.stack([d_x, d_y], axis=-1)
            return np.minimum(np.maximum(d[:, 0], d[:, 1]), 0.0) + np.linalg.norm(np.maximum(d, 0.0), axis=-1) - rb
        return _callable

def rounded_cylinder(radius=1.0, round_radius=0.1, height=2.0) -> SDFObject: return RoundedCylinder(radius, round_radius, height)

class CappedCone(SDFObject):
    def __init__(self, height, radius1, radius2):
        super().__init__()
        self.height, self.radius1, self.radius2 = height, radius1, radius2
    def to_glsl(self) -> str:
        h = _glsl_format(self.height)
        r1 = _glsl_format(self.radius1)
        r2 = _glsl_format(self.radius2)
        return f"vec4(sdCappedCone(p, {h}, {r1}, {r2}), -1.0, 0.0, 0.0)"
    def to_callable(self):
        if isinstance(self.height, str) or isinstance(self.radius1, str) or isinstance(self.radius2, str):
            raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        h, r1, r2 = self.height, self.radius1, self.radius2
        def _callable(p):
            q_x = np.linalg.norm(p[:, [0, 2]], axis=-1)
            q = np.stack([q_x, p[:, 1]], axis=-1)
            k1 = np.array([r2, h])
            k2 = np.array([r2 - r1, 2.0 * h])
            ca_x_min_operand = np.where(q[:, 1] < 0.0, r1, r2)
            ca_x = q[:, 0] - np.minimum(q[:, 0], ca_x_min_operand)
            ca_y = np.abs(q[:, 1]) - h
            ca = np.stack([ca_x, ca_y], axis=-1)
            k1_minus_q = k1 - q
            dot_val = np.sum(k1_minus_q * k2, axis=-1)
            dot2_k2 = np.dot(k2, k2)
            clamp_val = np.clip(dot_val / dot2_k2, 0.0, 1.0)
            cb = q - k1 + k2 * clamp_val[:, np.newaxis]
            s = np.where((cb[:, 0] < 0.0) & (ca[:, 1] < 0.0), -1.0, 1.0)
            dot2_ca = np.sum(ca * ca, axis=-1)
            dot2_cb = np.sum(cb * cb, axis=-1)
            return s * np.sqrt(np.minimum(dot2_ca, dot2_cb))
        return _callable

def capped_cone(height=1.0, radius1=0.5, radius2=0.2) -> SDFObject: return CappedCone(height, radius1, radius2)

class RoundCone(SDFObject):
    def __init__(self, radius1, radius2, height):
        super().__init__()
        self.radius1, self.radius2, self.height = radius1, radius2, height
    def to_glsl(self) -> str:
        r1 = _glsl_format(self.radius1)
        r2 = _glsl_format(self.radius2)
        h = _glsl_format(self.height)
        return f"vec4(sdRoundCone(p, {r1}, {r2}, {h}), -1.0, 0.0, 0.0)"
    def to_callable(self):
        if isinstance(self.radius1, str) or isinstance(self.radius2, str) or isinstance(self.height, str):
            raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        r1, r2, h = self.radius1, self.radius2, self.height
        def _callable(p):
            b = (r1 - r2) / h
            a = np.sqrt(1.0 - b * b)
            q = np.stack([np.linalg.norm(p[:, [0, 2]], axis=-1), p[:, 1]], axis=-1)
            k = np.dot(q, np.array([-b, a]))
            cond1 = k < 0.0
            cond2 = k > a * h
            dist1 = np.linalg.norm(q, axis=-1) - r1
            dist2 = np.linalg.norm(q - np.array([0.0, h]), axis=-1) - r2
            dist3 = np.dot(q, np.array([a, b])) - r1
            return np.where(cond1, dist1, np.where(cond2, dist2, dist3))
        return _callable

def round_cone(radius1=0.5, radius2=0.2, height=1.0) -> SDFObject: return RoundCone(radius1, radius2, height)

class Pyramid(SDFObject):
    def __init__(self, height):
        super().__init__()
        self.height = height
    def to_glsl(self) -> str: return f"vec4(sdPyramid(p, {_glsl_format(self.height)}), -1.0, 0.0, 0.0)"
    def to_callable(self):
        if isinstance(self.height, str): raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        h = self.height
        m2 = h*h + 0.25
        def _callable(p):
            p_xz = np.abs(p[:, [0, 2]])
            p_z_gt_x = p_xz[:, 1] > p_xz[:, 0]
            p_xz_swapped = p_xz[:, ::-1]
            p_xz = np.where(p_z_gt_x[:, np.newaxis], p_xz_swapped, p_xz)
            p_xz -= 0.5
            q = np.stack([
                p_xz[:, 1],
                h * p[:, 1] - 0.5 * p_xz[:, 0],
                h * p_xz[:, 0] + 0.5 * p[:, 1]
            ], axis=-1)
            s = np.maximum(-q[:, 0], 0.0)
            t = np.clip((q[:, 1] - 0.5 * p_xz[:, 1]) / (m2 + 0.25), 0.0, 1.0)
            a = m2 * (q[:, 0] + s)**2 + q[:, 1]**2
            b = m2 * (q[:, 0] + 0.5 * t)**2 + (q[:, 1] - m2 * t)**2
            cond = np.minimum(q[:, 1], -q[:, 0] * m2 - q[:, 1] * 0.5) > 0.0
            d2 = np.where(cond, 0.0, np.minimum(a, b))
            return np.sqrt((d2 + q[:, 2]**2) / m2) * np.sign(np.maximum(q[:, 2], -p[:, 1]))
        return _callable

def pyramid(height=1.0) -> SDFObject: return Pyramid(height)


# --- Material ---

class Material(SDFObject):
    def __init__(self, child, color):
        super().__init__()
        self.child = child
        self.color = color
        self.material_id = -1 # Will be set by the renderer
    
    def to_glsl(self) -> str:
        child_glsl = self.child.to_glsl()
        # The child returns a vec4. We need to overwrite the material ID.
        return f"vec4(({child_glsl}).x, {float(self.material_id)}, 0.0, 0.0)"

    def to_callable(self):
        # Materials are a render-time concept; for mesh generation, we use the child's shape.
        return self.child.to_callable()

    def _collect_materials(self, materials):
        if self not in materials:
            self.material_id = len(materials)
            materials.append(self)
        self.child._collect_materials(materials)

    def get_glsl_definitions(self) -> list:
        return self.child.get_glsl_definitions()


# --- Standard Operations ---

class Union(SDFObject):
    def __init__(self, *children):
        super().__init__()
        self.children = children
    def to_glsl(self) -> str: return reduce(lambda a, b: f"opU({a}, {b})", [c.to_glsl() for c in self.children])
    def to_callable(self):
        callables = [c.to_callable() for c in self.children]; return lambda p: reduce(np.minimum, [c(p) for c in callables])
    def get_glsl_definitions(self) -> list: return [_get_glsl_content('operations.glsl')] + sum([c.get_glsl_definitions() for c in self.children], [])
    def _collect_materials(self, materials):
        for c in self.children: c._collect_materials(materials)

class Intersection(SDFObject):
    def __init__(self, *children):
        super().__init__()
        self.children = children
    def to_glsl(self) -> str: return reduce(lambda a, b: f"opI({a}, {b})", [c.to_glsl() for c in self.children])
    def to_callable(self):
        callables = [c.to_callable() for c in self.children]; return lambda p: reduce(np.maximum, [c(p) for c in callables])
    def get_glsl_definitions(self) -> list: return [_get_glsl_content('operations.glsl')] + sum([c.get_glsl_definitions() for c in self.children], [])
    def _collect_materials(self, materials):
        for c in self.children: c._collect_materials(materials)

class Difference(SDFObject):
    def __init__(self, a, b):
        super().__init__()
        self.a, self.b = a, b
    def to_glsl(self) -> str: return f"opS({self.a.to_glsl()}, {self.b.to_glsl()})"
    def to_callable(self):
        a_call, b_call = self.a.to_callable(), self.b.to_callable(); return lambda p: np.maximum(a_call(p), -b_call(p))
    def get_glsl_definitions(self) -> list: return [_get_glsl_content('operations.glsl')] + self.a.get_glsl_definitions() + self.b.get_glsl_definitions()
    def _collect_materials(self, materials):
        self.a._collect_materials(materials)
        self.b._collect_materials(materials)

class Xor(SDFObject):
    def __init__(self, a, b):
        super().__init__()
        self.a, self.b = a, b
    def to_glsl(self) -> str: return f"opX({self.a.to_glsl()}, {self.b.to_glsl()})"
    def to_callable(self):
        a_call, b_call = self.a.to_callable(), self.b.to_callable(); return lambda p: np.maximum(np.minimum(a_call(p), b_call(p)), -np.maximum(a_call(p), b_call(p)))
    def get_glsl_definitions(self) -> list: return [_get_glsl_content('operations.glsl')] + self.a.get_glsl_definitions() + self.b.get_glsl_definitions()
    def _collect_materials(self, materials):
        self.a._collect_materials(materials)
        self.b._collect_materials(materials)


# --- Smooth Operations ---

class SmoothUnion(SDFObject):
    def __init__(self, a, b, k):
        super().__init__()
        self.a, self.b, self.k = a, b, k
    def to_glsl(self) -> str: return f"sUnion({self.a.to_glsl()}, {self.b.to_glsl()}, {_glsl_format(self.k)})"
    def to_callable(self):
        if isinstance(self.k, str): raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        a_call, b_call, k = self.a.to_callable(), self.b.to_callable(), float(self.k)
        def _callable(p):
            d1, d2 = a_call(p), b_call(p); h = np.clip(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
            return d2 * (1.0 - h) + d1 * h - k * h * (1.0 - h)
        return _callable
    def get_glsl_definitions(self) -> list: return [_get_glsl_content('operations.glsl')] + self.a.get_glsl_definitions() + self.b.get_glsl_definitions()
    def _collect_materials(self, materials):
        self.a._collect_materials(materials)
        self.b._collect_materials(materials)

class SmoothIntersection(SDFObject):
    def __init__(self, a, b, k):
        super().__init__()
        self.a, self.b, self.k = a, b, k
    def to_glsl(self) -> str: return f"sIntersect({self.a.to_glsl()}, {self.b.to_glsl()}, {_glsl_format(self.k)})"
    def to_callable(self):
        if isinstance(self.k, str): raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        a_call, b_call, k = self.a.to_callable(), self.b.to_callable(), float(self.k)
        def _callable(p):
            d1, d2 = a_call(p), b_call(p)
            h = np.clip(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0)
            return d2 * (1.0 - h) + d1 * h + k * h * (1.0 - h)
        return _callable
    def get_glsl_definitions(self) -> list: return [_get_glsl_content('operations.glsl')] + self.a.get_glsl_definitions() + self.b.get_glsl_definitions()
    def _collect_materials(self, materials):
        self.a._collect_materials(materials)
        self.b._collect_materials(materials)

class SmoothDifference(SDFObject):
    def __init__(self, a, b, k):
        super().__init__()
        self.a, self.b, self.k = a, b, k
    def to_glsl(self) -> str: return f"sDifference({self.a.to_glsl()}, {self.b.to_glsl()}, {_glsl_format(self.k)})"
    def to_callable(self):
        if isinstance(self.k, str): raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        a_call, b_call, k = self.a.to_callable(), self.b.to_callable(), float(self.k)
        def _callable(p):
            d1, d2 = a_call(p), -b_call(p)
            h = np.clip(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
            return d1 * (1.0 - h) + d2 * h + k * h * (1.0 - h)
        return _callable
    def get_glsl_definitions(self) -> list: return [_get_glsl_content('operations.glsl')] + self.a.get_glsl_definitions() + self.b.get_glsl_definitions()
    def _collect_materials(self, materials):
        self.a._collect_materials(materials)
        self.b._collect_materials(materials)


# --- Shaping Operations ---

class Round(SDFObject):
    def __init__(self, child, radius):
        super().__init__()
        self.child, self.radius = child, radius
    def to_glsl(self) -> str:
        return f"opRound({self.child.to_glsl()}, {_glsl_format(self.radius)})"
    def to_callable(self):
        if isinstance(self.radius, str): raise TypeError("Animated parameters not supported for mesh export.")
        child_call = self.child.to_callable()
        return lambda p: child_call(p) - self.radius
    def get_glsl_definitions(self) -> list: return [_get_glsl_content("operations.glsl")] + self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)

class Onion(SDFObject):
    def __init__(self, child, thickness):
        super().__init__()
        self.child, self.thickness = child, thickness
    def to_glsl(self) -> str:
        return f"opOnion({self.child.to_glsl()}, {_glsl_format(self.thickness)})"
    def to_callable(self):
        if isinstance(self.thickness, str): raise TypeError("Animated parameters not supported for mesh export.")
        child_call = self.child.to_callable()
        return lambda p: np.abs(child_call(p)) - self.thickness
    def get_glsl_definitions(self) -> list: return [_get_glsl_content("operations.glsl")] + self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)

class Elongate(SDFObject):
    def __init__(self, child, h):
        super().__init__()
        self.child, self.h = child, h
    def to_glsl(self) -> str:
        h_str = f"vec3({_glsl_format(self.h[0])}, {_glsl_format(self.h[1])}, {_glsl_format(self.h[2])})"
        return self.child.to_glsl().replace("p", f"opElongate(p, {h_str})")
    def to_callable(self):
        child_call = self.child.to_callable()
        return lambda p: child_call(p - np.clip(p, -self.h, self.h))
    def get_glsl_definitions(self): return [_get_glsl_content("transforms.glsl")] + self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)

class Displace(SDFObject):
    def __init__(self, child, displacement_glsl):
        super().__init__()
        self.child, self.displacement_glsl = child, displacement_glsl
    def to_glsl(self) -> str:
        return f"opDisplace({self.child.to_glsl()}, {self.displacement_glsl})"
    def to_callable(self):
        raise TypeError("Cannot save mesh of an object with GLSL-based displacement.")
    def get_glsl_definitions(self) -> list: return [_get_glsl_content("operations.glsl")] + self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)

class Extrude(SDFObject):
    def __init__(self, child, height):
        super().__init__()
        self.child, self.height = child, height
    def to_glsl(self) -> str:
        h = _glsl_format(self.height)
        return f"opExtrude({self.child.to_glsl()}, p, {h})"
    def to_callable(self):
        if isinstance(self.height, str): raise TypeError("Animated parameters not supported for mesh export.")
        child_call = self.child.to_callable()
        h = self.height
        def _callable(p):
            d = child_call(p)
            w = np.stack([d, np.abs(p[:, 2]) - h], axis=-1)
            return np.minimum(np.maximum(w[:,0], w[:,1]), 0.0) + np.linalg.norm(np.maximum(w, 0.0), axis=-1)
        return _callable
    def get_glsl_definitions(self) -> list: return [_get_glsl_content("operations.glsl")] + self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)


# --- Basic Transformations ---

class Translate(SDFObject):
    def __init__(self, child, offset):
        super().__init__()
        self.child, self.offset = child, offset
    def to_glsl(self) -> str:
        o = self.offset
        return self.child.to_glsl().replace(
            "p", f"opTranslate(p, vec3({_glsl_format(o[0])}, {_glsl_format(o[1])}, {_glsl_format(o[2])}))"
        )
    def to_callable(self):
        child_call = self.child.to_callable()
        return lambda p: child_call(p - self.offset)
    def get_glsl_definitions(self): 
        return [_get_glsl_content("transforms.glsl")] + self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)


class Scale(SDFObject):
    def __init__(self, child, factor):
        super().__init__()
        self.child, self.factor = child, factor
    def to_glsl(self) -> str:
        f = self.factor
        if isinstance(f, (tuple, list, np.ndarray)):
            factor_str = f"vec3({_glsl_format(f[0])}, {_glsl_format(f[1])}, {_glsl_format(f[2])})"
        else:
            factor_str = f"vec3({_glsl_format(f)})"
        return f"((){{ vec4 res = {self.child.to_glsl().replace('p', f'opScale(p, {factor_str})')}; res.x *= length({factor_str}); return res; }})()"
    def to_callable(self):
        if isinstance(self.factor, str): 
            raise TypeError("Animated parameters not supported for mesh export.")
        child_call = self.child.to_callable()
        f = np.array(self.factor if isinstance(self.factor, (tuple, list, np.ndarray)) else (self.factor,))
        return lambda p: child_call(p / f) * np.mean(f)  # approximate isotropic correction
    def get_glsl_definitions(self):
        return [_get_glsl_content("transforms.glsl")] + self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)


class Rotate(SDFObject):
    def __init__(self, child, axis, angle):
        super().__init__()
        self.child, self.axis, self.angle = child, axis, angle
    def to_glsl(self) -> str:
        if np.allclose(self.axis, X): func = "opRotateX"
        elif np.allclose(self.axis, Y): func = "opRotateY"
        else: func = "opRotateZ"
        return self.child.to_glsl().replace("p", f"{func}(p, {_glsl_format(self.angle)})")
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
    def get_glsl_definitions(self):
        return [_get_glsl_content("transforms.glsl")] + self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)


class Orient(SDFObject):
    def __init__(self, child, axis):
        super().__init__()
        self.child, self.axis = child, axis
    def to_glsl(self) -> str:
        if np.allclose(self.axis, X): return self.child.to_glsl().replace("p", "opOrient(p, 0)")
        elif np.allclose(self.axis, Y): return self.child.to_glsl().replace("p", "opOrient(p, 1)")
        else: return self.child.to_glsl()
    def to_callable(self):
        child_call = self.child.to_callable()
        if np.allclose(self.axis, X): return lambda p: child_call(p[:, [2,1,0]])
        elif np.allclose(self.axis, Y): return lambda p: child_call(p[:, [0,2,1]])
        return child_call
    def get_glsl_definitions(self):
        return [_get_glsl_content("transforms.glsl")] + self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)


class Twist(SDFObject):
    def __init__(self, child, k):
        super().__init__()
        self.child, self.k = child, k
    def to_glsl(self) -> str:
        return self.child.to_glsl().replace("p", f"opTwist(p, {_glsl_format(self.k)})")
    def to_callable(self):
        if isinstance(self.k, str): raise TypeError("Animated parameters not supported for mesh export.")
        child_call, k = self.child.to_callable(), float(self.k)
        def _callable(p):
            c, s = np.cos(k * p[:,1]), np.sin(k * p[:,1])
            x_new, z_new = p[:,0]*c - p[:,2]*s, p[:,0]*s + p[:,2]*c
            q = np.stack([x_new, p[:,1], z_new], axis=-1)
            return child_call(q)
        return _callable
    def get_glsl_definitions(self):
        return [_get_glsl_content("transforms.glsl")] + self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)

class ShearXY(SDFObject):
    def __init__(self, child, shear):
        super().__init__()
        self.child, self.shear = child, shear
    def to_glsl(self) -> str:
        sh = self.shear
        return self.child.to_glsl().replace("p", f"opShearXY(p, vec2({_glsl_format(sh[0])}, {_glsl_format(sh[1])}))")
    def to_callable(self):
        child_call, sh = self.child.to_callable(), np.array(self.shear)
        def _callable(p):
            q = p.copy()
            q[:,0] += sh[0] * p[:,2]
            q[:,1] += sh[1] * p[:,2]
            return child_call(q)
        return _callable
    def get_glsl_definitions(self): return [_get_glsl_content("transforms.glsl")] + self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)

class ShearXZ(SDFObject):
    def __init__(self, child, shear):
        super().__init__()
        self.child, self.shear = child, shear
    def to_glsl(self) -> str:
        sh = self.shear
        return self.child.to_glsl().replace("p", f"opShearXZ(p, vec2({_glsl_format(sh[0])}, {_glsl_format(sh[1])}))")
    def to_callable(self):
        child_call, sh = self.child.to_callable(), np.array(self.shear)
        def _callable(p):
            q = p.copy()
            q[:,0] += sh[0] * p[:,1]
            q[:,2] += sh[1] * p[:,1]
            return child_call(q)
        return _callable
    def get_glsl_definitions(self): return [_get_glsl_content("transforms.glsl")] + self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)

class ShearYZ(SDFObject):
    def __init__(self, child, shear):
        super().__init__()
        self.child, self.shear = child, shear
    def to_glsl(self) -> str:
        sh = self.shear
        return self.child.to_glsl().replace("p", f"opShearYZ(p, vec2({_glsl_format(sh[0])}, {_glsl_format(sh[1])}))")
    def to_callable(self):
        child_call, sh = self.child.to_callable(), np.array(self.shear)
        def _callable(p):
            q = p.copy()
            q[:,1] += sh[0] * p[:,0]
            q[:,2] += sh[1] * p[:,0]
            return child_call(q)
        return _callable
    def get_glsl_definitions(self): return [_get_glsl_content("transforms.glsl")] + self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)

class BendX(SDFObject):
    def __init__(self, child, k):
        super().__init__()
        self.child, self.k = child, k
    def to_glsl(self) -> str:
        return self.child.to_glsl().replace("p", f"opBendX(p, {_glsl_format(self.k)})")
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
    def get_glsl_definitions(self): return [_get_glsl_content("transforms.glsl")] + self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)

class BendY(SDFObject):
    def __init__(self, child, k):
        super().__init__()
        self.child, self.k = child, k
    def to_glsl(self) -> str:
        return self.child.to_glsl().replace("p", f"opBendY(p, {_glsl_format(self.k)})")
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
    def get_glsl_definitions(self): return [_get_glsl_content("transforms.glsl")] + self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)

class BendZ(SDFObject):
    def __init__(self, child, k):
        super().__init__()
        self.child, self.k = child, k
    def to_glsl(self) -> str:
        return self.child.to_glsl().replace("p", f"opBendZ(p, {_glsl_format(self.k)})")
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
    def get_glsl_definitions(self): return [_get_glsl_content("transforms.glsl")] + self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)


class Repeat(SDFObject):
    def __init__(self, child, spacing):
        super().__init__()
        self.child, self.spacing = child, spacing
    def to_glsl(self) -> str:
        s = self.spacing
        return self.child.to_glsl().replace("p", f"opRepeat(p, vec3({_glsl_format(s[0])}, {_glsl_format(s[1])}, {_glsl_format(s[2])}))")
    def to_callable(self):
        child_call, s = self.child.to_callable(), self.spacing
        active_spacing = np.where(s == 0, np.inf, s)
        return lambda p: child_call(np.mod(p + 0.5*active_spacing, active_spacing) - 0.5*active_spacing)
    def get_glsl_definitions(self):
        return [_get_glsl_content("transforms.glsl")] + self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)

class LimitedRepeat(SDFObject):
    def __init__(self, child, spacing, limits):
        super().__init__()
        self.child, self.spacing, self.limits = child, spacing, limits
    def to_glsl(self) -> str:
        s = self.spacing
        l = self.limits
        s_str = f"vec3({_glsl_format(s[0])}, {_glsl_format(s[1])}, {_glsl_format(s[2])})"
        l_str = f"vec3({_glsl_format(l[0])}, {_glsl_format(l[1])}, {_glsl_format(l[2])})"
        return self.child.to_glsl().replace("p", f"opLimitedRepeat(p, {s_str}, {l_str})")
    def to_callable(self):
        child_call = self.child.to_callable()
        s = self.spacing
        l = self.limits
        def _callable(p):
            q = p - s * np.clip(np.round(p / s), -l, l)
            return child_call(q)
        return _callable
    def get_glsl_definitions(self): return [_get_glsl_content("transforms.glsl")] + self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)


class PolarRepeat(SDFObject):
    def __init__(self, child, repetitions):
        super().__init__()
        self.child, self.repetitions = child, repetitions
    def to_glsl(self) -> str:
        return self.child.to_glsl().replace("p", f"opPolarRepeat(p, {_glsl_format(self.repetitions)})")
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
    def get_glsl_definitions(self):
        return [_get_glsl_content("transforms.glsl")] + self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)


class Mirror(SDFObject):
    def __init__(self, child, axes):
        super().__init__()
        self.child, self.axes = child, axes
    def to_glsl(self) -> str:
        a = self.axes
        return self.child.to_glsl().replace("p", f"opMirror(p, vec3({_glsl_format(a[0])}, {_glsl_format(a[1])}, {_glsl_format(a[2])}))")
    def to_callable(self):
        child_call, a = self.child.to_callable(), self.axes
        def _callable(p):
            q = p.copy()
            if a[0] > 0.5: q[:,0] = np.abs(q[:,0])
            if a[1] > 0.5: q[:,1] = np.abs(q[:,1])
            if a[2] > 0.5: q[:,2] = np.abs(q[:,2])
            return child_call(q)
        return _callable
    def get_glsl_definitions(self):
        return [_get_glsl_content("transforms.glsl")] + self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)


# --- Custom GLSL ---

class Forge(SDFObject):
    def __init__(self, glsl_code_body: str):
        super().__init__()
        self.glsl_code_body = glsl_code_body
        self.unique_id = "forge_func_" + uuid.uuid4().hex[:8]
    def to_glsl(self) -> str: return f"vec4({self.unique_id}(p), -1.0, 0.0, 0.0)"
    def get_glsl_definitions(self) -> list:
        return [f"float {self.unique_id}(vec3 p){{ {self.glsl_code_body} }}"]
    def to_callable(self):
        if not _MODERNGL_AVAILABLE: raise ImportError("To save meshes with Forge objects, 'moderngl' and 'glfw' are required.")
        cls = self.__class__
        if not hasattr(cls, '_mgl_context'):
            if not glfw.init(): raise RuntimeError("glfw.init() failed")
            atexit.register(glfw.terminate)
            glfw.window_hint(glfw.VISIBLE, False); win = glfw.create_window(1, 1, "", None, None)
            glfw.make_context_current(win); cls._mgl_context = moderngl.create_context(require=430)
        ctx = cls._mgl_context
        compute_shader = ctx.compute_shader(f"""
        #version 430
        layout(local_size_x=256, local_size_y=1, local_size_z=1) in;
        layout(std430, binding=0) buffer points {{ vec3 p[]; }};
        layout(std430, binding=1) buffer distances {{ float d[]; }};
        {self.get_glsl_definitions()[0]}
        void main() {{ uint gid = gl_GlobalInvocationID.x; d[gid] = {self.to_glsl().replace('p', 'p[gid]').replace('vec4', '').strip('()').split(',')[0]}; }}""")
        def _gpu_evaluator(points_np):
            points_np = np.array(points_np, dtype='f4'); num_points = len(points_np)
            point_buffer = ctx.buffer(points_np.tobytes()); dist_buffer = ctx.buffer(reserve=num_points * 4)
            point_buffer.bind_to_storage_buffer(0); dist_buffer.bind_to_storage_buffer(1)
            group_size = (num_points + 255) // 256; compute_shader.run(group_x=group_size)
            return np.frombuffer(dist_buffer.read(), dtype='f4')
        return _gpu_evaluator
