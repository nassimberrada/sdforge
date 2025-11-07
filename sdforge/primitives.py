import numpy as np
from .core import SDFObject, _glsl_format

# --- Primitives ---

class Sphere(SDFObject):
    """Represents a sphere primitive."""
    def __init__(self, r=1.0):
        super().__init__()
        self.r = r
    def to_glsl(self) -> str: return f"vec4(sdSphere(p, {_glsl_format(self.r)}), -1.0, 0.0, 0.0)"
    def to_callable(self):
        if isinstance(self.r, str): raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        return lambda p: np.linalg.norm(p, axis=-1) - self.r

def sphere(r=1.0) -> SDFObject:
    """Creates a sphere.

    Args:
        r (float or str, optional): The radius of the sphere. Defaults to 1.0.
    """
    return Sphere(r)

class Box(SDFObject):
    """Represents a box primitive."""
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

class RoundedBox(SDFObject):
    """Represents a box with rounded edges."""
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

def box(size=1.0, radius=0.0) -> SDFObject:
    """
    Creates a box, optionally with rounded edges.

    Args:
        size (float, tuple, or str, optional): The size of the box along each axis.
                                               If a float, creates a uniform box. Defaults to 1.0.
        radius (float or str, optional): The radius of the rounded edges. If > 0,
                                         a rounded box is created. Defaults to 0.0.
    """
    if (isinstance(radius, str) and radius != "0.0") or (isinstance(radius, (int, float)) and radius > 0):
        return RoundedBox(size, radius)
    return Box(size)

class Torus(SDFObject):
    """Represents a torus primitive."""
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

def torus(major=1.0, minor=0.25) -> SDFObject:
    """Creates a torus.

    Args:
        major (float or str, optional): The major radius (from center to tube center). Defaults to 1.0.
        minor (float or str, optional): The minor radius (radius of the tube). Defaults to 0.25.
    """
    return Torus(major, minor)

class Capsule(SDFObject):
    """Represents a capsule primitive."""
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

class CappedCylinder(SDFObject):
    """Represents a cylinder defined by two end points."""
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

def line(a, b, radius=0.1, rounded_caps=True) -> SDFObject:
    """
    Creates a line segment with a radius, with either rounded or flat caps.

    Args:
        a (tuple or np.ndarray): The start point of the line's spine.
        b (tuple or np.ndarray): The end point of the line's spine.
        radius (float or str, optional): The radius of the line. Defaults to 0.1.
        rounded_caps (bool, optional): If True, creates rounded ends (a capsule).
                                       If False, creates flat ends (a capped cylinder).
                                       Defaults to True.
    """
    if rounded_caps:
        return Capsule(a, b, radius)
    else:
        return CappedCylinder(a, b, radius)

class Cylinder(SDFObject):
    """Represents a cylinder primitive."""
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

class RoundedCylinder(SDFObject):
    """Represents a cylinder with rounded edges."""
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

def cylinder(radius=0.5, height=1.0, round_radius=0.0) -> SDFObject:
    """
    Creates a cylinder, optionally with rounded edges, oriented along the Y-axis.

    Args:
        radius (float or str, optional): The radius of the cylinder. Defaults to 0.5.
        height (float or str, optional): The total height of the cylinder. Defaults to 1.0.
        round_radius (float or str, optional): The radius of the rounding. If > 0,
                                              a rounded cylinder is created. Defaults to 0.0.
    """
    if (isinstance(round_radius, str) and round_radius != "0.0") or (isinstance(round_radius, (int, float)) and round_radius > 0):
        return RoundedCylinder(radius, round_radius, height)
    return Cylinder(radius, height)

class Cone(SDFObject):
    """Represents a cone with caps (a frustum)."""
    def __init__(self, height=1.0, radius1=0.5, radius2=0.0):
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

def cone(height=1.0, radius1=0.5, radius2=0.0) -> SDFObject:
    """
    Creates a capped cone (a frustum).

    Args:
        height (float, optional): The height of the cone. Defaults to 1.0.
        radius1 (float, optional): The radius of the base at Y=0. Defaults to 0.5.
        radius2 (float, optional): The radius of the top at Y=height. If 0, a standard
                                   pointed cone is created. Defaults to 0.0.
    """
    return Cone(height, radius1, radius2)

class Plane(SDFObject):
    """Represents an infinite plane."""
    def __init__(self, normal, offset=0):
        super().__init__()
        self.normal, self.offset = np.array(normal), offset
    def to_glsl(self) -> str: n = self.normal; return f"vec4(sdPlane(p, vec4({n[0]}, {n[1]}, {n[2]}, {_glsl_format(self.offset)})), -1.0, 0.0, 0.0)"
    def to_callable(self):
        if isinstance(self.offset, str): raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        return lambda p: np.dot(p, self.normal) + self.offset

def plane(normal, offset=0) -> SDFObject:
    """Creates an infinite plane.

    Args:
        normal (tuple or np.ndarray): The normal vector of the plane.
        offset (float or str, optional): The offset of the plane from the origin. Defaults to 0.
    """
    return Plane(normal, offset)

class HexPrism(SDFObject):
    """Represents a hexagonal prism."""
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

def hex_prism(radius=1.0, height=0.1) -> SDFObject:
    """Creates a hexagonal prism oriented along the Z-axis.

    Args:
        radius (float or str, optional): The radius (distance from center to vertex). Defaults to 1.0.
        height (float or str, optional): The height of the prism. Defaults to 0.1.
    """
    return HexPrism(radius, height)

class Octahedron(SDFObject):
    """Represents an octahedron."""
    def __init__(self, size=1.0):
        super().__init__()
        self.size = size
    def to_glsl(self) -> str: return f"vec4(sdOctahedron(p, {_glsl_format(self.size)}), -1.0, 0.0, 0.0)"
    def to_callable(self):
        if isinstance(self.size, str): raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        return lambda p: (np.sum(np.abs(p), axis=-1) - self.size) * 0.57735027

def octahedron(size=1.0) -> SDFObject:
    """Creates an octahedron.

    Args:
        size (float or str, optional): The size of the octahedron. Defaults to 1.0.
    """
    return Octahedron(size)

class Ellipsoid(SDFObject):
    """Represents an ellipsoid."""
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

def ellipsoid(radii) -> SDFObject:
    """Creates an ellipsoid.

    Args:
        radii (tuple or list): The radii of the ellipsoid along the X, Y, and Z axes.
    """
    return Ellipsoid(radii)

class BoxFrame(SDFObject):
    """Represents the frame of a box."""
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

def box_frame(size, edge_radius=0.1) -> SDFObject:
    """Creates the frame of a box.

    Args:
        size (tuple or float): The size of the box frame.
        edge_radius (float, optional): The radius of the edges. Defaults to 0.1.
    """
    return BoxFrame(size, edge_radius)

class CappedTorus(SDFObject):
    """Represents a torus capped at a specific angle."""
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

def capped_torus(angle_sc, major_radius=1.0, minor_radius=0.25) -> SDFObject:
    """Creates a capped torus, like a 'C' shape.

    Args:
        angle_sc (tuple): A 2D vector representing (sin(angle), cos(angle)) of the cap.
        major_radius (float, optional): The major radius. Defaults to 1.0.
        minor_radius (float, optional): The minor radius. Defaults to 0.25.
    """
    return CappedTorus(angle_sc, major_radius, minor_radius)

class Link(SDFObject):
    """Represents two cylinders connected by a torus section."""
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

def link(length=1.0, radius1=0.3, radius2=0.1) -> SDFObject:
    """Creates a link shape.

    Args:
        length (float, optional): The length of the straight sections. Defaults to 1.0.
        radius1 (float, optional): The major radius of the curved section. Defaults to 0.3.
        radius2 (float, optional): The minor radius of the shape. Defaults to 0.1.
    """
    return Link(length, radius1, radius2)

class RoundCone(SDFObject):
    """Represents a cone with rounded edges."""
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

def round_cone(radius1=0.5, radius2=0.2, height=1.0) -> SDFObject:
    """Creates a cone with rounded edges.

    Args:
        radius1 (float, optional): The radius of the bottom sphere. Defaults to 0.5.
        radius2 (float, optional): The radius of the top sphere. Defaults to 0.2.
        height (float, optional): The height between the sphere centers. Defaults to 1.0.
    """
    return RoundCone(radius1, radius2, height)

class Pyramid(SDFObject):
    """Represents a 4-sided pyramid."""
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

def pyramid(height=1.0) -> SDFObject:
    """Creates a 4-sided pyramid.

    Args:
        height (float, optional): The height of the pyramid. Defaults to 1.0.
    """
    return Pyramid(height)