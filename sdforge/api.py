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
    def translate(self, offset): return Translate(self, np.array(offset))
    def scale(self, factor): return Scale(self, factor)
    def orient(self, axis): return Orient(self, np.array(axis))
    def rotate(self, axis, angle): return Rotate(self, np.array(axis), angle)
    def twist(self, k): return Twist(self, k)
    def repeat(self, spacing): return Repeat(self, np.array(spacing))
    def mirror(self, axes): return Mirror(self, np.array(axes))
    def smooth_union(self, other, k): return SmoothUnion(self, other, k)
    def smooth_intersection(self, other, k): return SmoothIntersection(self, other, k)
    def smooth_difference(self, other, k): return SmoothDifference(self, other, k)
    def color(self, r, g, b): return Material(self, (r, g, b))


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
        raise NotImplementedError("HexPrism for mesh generation is not yet implemented.")

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
        raise NotImplementedError("Smooth Intersection for mesh generation is not yet implemented.")
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
        raise NotImplementedError("Smooth Difference for mesh generation is not yet implemented.")
    def get_glsl_definitions(self) -> list: return [_get_glsl_content('operations.glsl')] + self.a.get_glsl_definitions() + self.b.get_glsl_definitions()
    def _collect_materials(self, materials):
        self.a._collect_materials(materials)
        self.b._collect_materials(materials)


# --- Basic Transformations ---

class Translate(SDFObject):
    def __init__(self, child, offset):
        super().__init__()
        self.child, self.offset = child, offset
    def to_glsl(self) -> str: o = self.offset; return self.child.to_glsl().replace('p', f'(p - vec3({o[0]}, {o[1]}, {o[2]}))')
    def to_callable(self): child_call = self.child.to_callable(); return lambda p: child_call(p - self.offset)
    def get_glsl_definitions(self) -> list: return self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)

class Scale(SDFObject):
    def __init__(self, child, factor):
        super().__init__()
        self.child, self.factor = child, factor
    def to_glsl(self) -> str:
        f = _glsl_format(self.factor)
        # Use an IIFE to avoid evaluating the child GLSL twice
        return f"(() {{ vec4 res = {self.child.to_glsl().replace('p', f'(p / ({f}))')}; res.x *= ({f}); return res; }})()"
    def to_callable(self):
        if isinstance(self.factor, str): raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        child_call = self.child.to_callable(); return lambda p: child_call(p / self.factor) * self.factor
    def get_glsl_definitions(self) -> list: return self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)

class Orient(SDFObject):
    def __init__(self, child, axis):
        super().__init__()
        self.child, self.axis = child, axis
    def to_glsl(self) -> str:
        if np.allclose(self.axis, X): return self.child.to_glsl().replace('p', 'p.zyx')
        elif np.allclose(self.axis, Y): return self.child.to_glsl().replace('p', 'p.xzy')
        return self.child.to_glsl()
    def to_callable(self):
        child_call = self.child.to_callable()
        if np.allclose(self.axis, X): return lambda p: child_call(p[:, [2, 1, 0]])
        elif np.allclose(self.axis, Y): return lambda p: child_call(p[:, [0, 2, 1]])
        return child_call
    def get_glsl_definitions(self) -> list: return self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)


# --- Advanced Transformations ---

class Rotate(SDFObject):
    def __init__(self, child, axis, angle):
        super().__init__()
        self.child, self.axis, self.angle = child, axis, angle
    def to_glsl(self) -> str:
        if np.allclose(self.axis, X): func = 'opRotateX'
        elif np.allclose(self.axis, Y): func = 'opRotateY'
        else: func = 'opRotateZ'
        return self.child.to_glsl().replace('p', f"{func}(p, {_glsl_format(self.angle)})")
    def to_callable(self):
        if isinstance(self.angle, str): raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        child_call, angle = self.child.to_callable(), self.angle; c, s = np.cos(angle), np.sin(angle)
        if np.allclose(self.axis, X): R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif np.allclose(self.axis, Y): R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        else: R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return lambda p: child_call(p @ R.T)
    def get_glsl_definitions(self) -> list: return [_get_glsl_content('transforms.glsl')] + self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)

class Twist(SDFObject):
    def __init__(self, child, k):
        super().__init__()
        self.child, self.k = child, k
    def to_glsl(self) -> str:
        k_str = _glsl_format(self.k)
        # The opTwist function now modifies p in place and returns the child's result
        return f"opTwist({self.child.to_glsl()}, p, {k_str})"
    def to_callable(self):
        if isinstance(self.k, str): raise TypeError("Cannot save mesh of an object with animated (string) parameters.")
        child_call, k = self.child.to_callable(), float(self.k)
        def _callable(p):
            c, s = np.cos(k * p[:, 1]), np.sin(k * p[:, 1])
            x_new, z_new = p[:, 0] * c - p[:, 2] * s, p[:, 0] * s + p[:, 2] * c
            q = np.stack([x_new, p[:, 1], z_new], axis=-1); return child_call(q)
        return _callable
    def get_glsl_definitions(self) -> list:
        return [_get_glsl_content('transforms.glsl')] + self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)

class Repeat(SDFObject):
    def __init__(self, child, spacing):
        super().__init__()
        self.child, self.spacing = child, spacing
    def to_glsl(self) -> str: s = self.spacing; return self.child.to_glsl().replace('p', f"opRepeat(p, vec3({s[0]}, {s[1]}, {s[2]}))")
    def to_callable(self):
        child_call, s = self.child.to_callable(), self.spacing
        active_spacing = np.where(s == 0, np.inf, s)
        def _callable(p): return child_call(np.mod(p + 0.5 * active_spacing, active_spacing) - 0.5 * active_spacing)
        return _callable
    def get_glsl_definitions(self) -> list: return [_get_glsl_content('transforms.glsl')] + self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)

class Mirror(SDFObject):
    def __init__(self, child, axes):
        super().__init__()
        self.child, self.axes = child, axes
    def to_glsl(self) -> str: a = self.axes; return self.child.to_glsl().replace('p', f"opMirror(p, vec3({a[0]}, {a[1]}, {a[2]}))")
    def to_callable(self):
        child_call = self.child.to_callable(); a = self.axes
        def _callable(p):
            q = p.copy()
            if a[0] > 0.5: q[:,0] = np.abs(q[:,0])
            if a[1] > 0.5: q[:,1] = np.abs(q[:,1])
            if a[2] > 0.5: q[:,2] = np.abs(q[:,2])
            return child_call(q)
        return _callable
    def get_glsl_definitions(self) -> list: return [_get_glsl_content('transforms.glsl')] + self.child.get_glsl_definitions()
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