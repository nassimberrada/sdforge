import numpy as np
from ..core import SDFNode, GLSLContext
from ..utils import _glsl_format
from .params import Param

class Sphere(SDFNode):
    """Represents a sphere primitive."""
    glsl_dependencies = {"primitives"}

    def __init__(self, r: float = 1.0):
        super().__init__()
        self.r = r

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        dist_expr = f"sdSphere({ctx.p}, {_glsl_format(self.r)})"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

    def to_callable(self):
        if isinstance(self.r, (str, Param)):
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")
        r_val = self.r
        def _callable(points: np.ndarray) -> np.ndarray:
            return np.linalg.norm(points, axis=-1) - r_val
        return _callable

def sphere(r: float = 1.0) -> SDFNode:
    """
    Creates a sphere centered at the origin.

    Args:
        r (float, optional): The radius of the sphere. Defaults to 1.0.

    Returns:
        SDFNode: A sphere primitive.
    
    Example:
        >>> s = sphere(r=1.5)
    """
    return Sphere(r)

class Box(SDFNode):
    """Represents a box primitive, possibly with rounded edges."""
    glsl_dependencies = {"primitives"}

    def __init__(self, size: tuple = (1.0, 1.0, 1.0), radius: float = 0.0):
        super().__init__()
        self.size = size
        self.radius = radius

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        
        s = []
        for v in self.size:
            if isinstance(v, (int, float)):
                s.append(_glsl_format(v / 2.0))
            else:
                s.append(f"({_glsl_format(v)} / 2.0)")
        size_vec = f"vec3({s[0]}, {s[1]}, {s[2]})"
        
        use_rounding = False
        if isinstance(self.radius, (int, float)):
            if self.radius > 1e-6:
                use_rounding = True
        else: # Param or string
            use_rounding = True
        
        if use_rounding:
            dist_expr = f"sdRoundedBox({ctx.p}, {size_vec}, {_glsl_format(self.radius)})"
        else:
            dist_expr = f"sdBox({ctx.p}, {size_vec})"
            
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

    def to_callable(self):
        is_dynamic = any(isinstance(v, (str, Param)) for v in self.size) or isinstance(self.radius, (str, Param))
        if is_dynamic:
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")
        
        half_size = np.array(self.size) / 2.0
        radius = self.radius
        
        def _callable(points: np.ndarray) -> np.ndarray:
            q = np.abs(points) - half_size
            dist = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
            if radius <= 1e-6:
                dist += np.minimum(np.max(q, axis=-1), 0.0)
            else:
                dist -= radius
            return dist
        return _callable

def box(size=1.0, radius: float = 0.0, x: float = None, y: float = None, z: float = None) -> SDFNode:
    """
    Creates a box centered at the origin, optionally with rounded edges.

    Args:
        size (float or tuple, optional): The size of the box. If a float,
                                         creates a cube. If a tuple, specifies
                                         (width, height, depth). Defaults to 1.0.
        radius (float, optional): The radius for rounding the box's edges.
                                  Defaults to 0.0 (sharp edges).
        x (float, optional): A convenience argument to specify the x-size.
                             If x, y, and z are all provided, they override `size`.
        y (float, optional): Convenience for y-size.
        z (float, optional): Convenience for z-size.

    Returns:
        SDFNode: A box primitive.
    
    Example:
        >>> # A 2x2x2 cube
        >>> cube = box(2.0)
        >>> # A 1x2x3 cuboid with rounded corners
        >>> rounded_box = box(size=(1, 2, 3), radius=0.1)
    """
    if x is not None and y is not None and z is not None:
        size = (x, y, z)
    elif isinstance(size, (int, float, str, Param)):
        size = (size, size, size)
    return Box(size=tuple(size), radius=radius)

class Torus(SDFNode):
    """Represents a torus primitive."""
    glsl_dependencies = {"primitives"}

    def __init__(self, major: float = 1.0, minor: float = 0.25):
        super().__init__()
        self.major, self.minor = major, minor
        
    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        dist_expr = f"sdTorus({ctx.p}, vec2({_glsl_format(self.major)}, {_glsl_format(self.minor)}))"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

    def to_callable(self):
        if isinstance(self.major, (str, Param)) or isinstance(self.minor, (str, Param)):
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")
        major, minor = self.major, self.minor
        def _callable(points: np.ndarray) -> np.ndarray:
            q = np.array([np.linalg.norm(points[:, [0, 2]], axis=-1) - major, points[:, 1]]).T
            return np.linalg.norm(q, axis=-1) - minor
        return _callable

def torus(major: float = 1.0, minor: float = 0.25) -> SDFNode:
    """
    Creates a torus centered at the origin, oriented in the XZ plane.

    Args:
        major (float, optional): The major radius (from the center of the
                                 torus to the center of the tube). Defaults to 1.0.
        minor (float, optional): The minor radius (the radius of the tube
                                 itself). Defaults to 0.25.

    Returns:
        SDFNode: A torus primitive.
        
    Example:
        >>> t = torus(major=2.0, minor=0.1)
    """
    return Torus(major, minor)

class Line(SDFNode):
    """Represents a line segment primitive with a radius."""
    glsl_dependencies = {"primitives"}

    def __init__(self, a, b, radius: float = 0.1, rounded_caps: bool = True):
        super().__init__()
        self.a, self.b, self.radius = np.array(a), np.array(b), radius
        self.rounded_caps = rounded_caps

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        a, b, r = self.a, self.b, _glsl_format(self.radius)
        a_str = f"vec3({a[0]},{a[1]},{a[2]})"
        b_str = f"vec3({b[0]},{b[1]},{b[2]})"
        func = "sdCapsule" if self.rounded_caps else "sdCappedCylinder"
        dist_expr = f"{func}({ctx.p}, {a_str}, {b_str}, {r})"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

    def to_callable(self):
        if isinstance(self.radius, (str, Param)):
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")
        a, b, r = self.a, self.b, self.radius
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

def line(a, b, radius: float = 0.1, rounded_caps: bool = True) -> SDFNode:
    """
    Creates a line segment between two points with a given radius.

    Args:
        a (tuple or np.ndarray): The starting point of the line segment.
        b (tuple or np.ndarray): The ending point of the line segment.
        radius (float, optional): The radius of the line. Defaults to 0.1.
        rounded_caps (bool, optional): If True, creates a capsule with hemispherical
                                       ends. If False, creates a cylinder with flat
                                       ends. Defaults to True.

    Returns:
        SDFNode: A capsule or cylinder primitive.
    
    Example:
        >>> from sdforge import X
        >>> # A capsule from the origin to a point on the X axis
        >>> l = line(a=(0,0,0), b=X*2, radius=0.1)
    """
    return Line(a, b, radius, rounded_caps)

class Cylinder(SDFNode):
    """Represents a cylinder primitive, possibly with rounded edges."""
    glsl_dependencies = {"primitives"}

    def __init__(self, radius: float = 0.5, height: float = 1.0, round_radius: float = 0.0):
        super().__init__()
        self.radius, self.height, self.round_radius = radius, height, round_radius

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        
        use_rounding = False
        if isinstance(self.round_radius, (int, float)):
            if self.round_radius > 1e-6:
                use_rounding = True
        else: # Param or string
            use_rounding = True

        if use_rounding:
            dist_expr = f"sdRoundedCylinder({ctx.p}, {_glsl_format(self.radius)}, {_glsl_format(self.round_radius)}, {_glsl_format(self.height)})"
        else:
            h_half = _glsl_format(self.height / 2.0) if not isinstance(self.height, (str, Param)) else f"({_glsl_format(self.height)})/2.0"
            dist_expr = f"sdCylinder({ctx.p}, vec2({_glsl_format(self.radius)}, {h_half}))"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

    def to_callable(self):
        is_dynamic = isinstance(self.radius, (str, Param)) or isinstance(self.height, (str, Param)) or isinstance(self.round_radius, (str, Param))
        if is_dynamic:
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")

        radius, height, round_radius = self.radius, self.height, self.round_radius
        if round_radius > 1e-6:
            def _callable_rounded(points: np.ndarray) -> np.ndarray:
                d_x = np.linalg.norm(points[:, [0, 2]], axis=-1) - 2.0 * radius + round_radius
                d_y = np.abs(points[:, 1]) - height
                d = np.stack([d_x, d_y], axis=-1)
                return np.minimum(np.maximum(d[:, 0], d[:, 1]), 0.0) + np.linalg.norm(np.maximum(d, 0.0), axis=-1) - round_radius
            return _callable_rounded
        else:
            h_half = height / 2.0
            def _callable_sharp(points: np.ndarray) -> np.ndarray:
                d = np.abs(np.array([np.linalg.norm(points[:, [0, 2]], axis=-1), points[:, 1]]).T) - np.array([radius, h_half])
                return np.minimum(np.maximum(d[:, 0], d[:, 1]), 0.0) + np.linalg.norm(np.maximum(d, 0.0), axis=-1)
            return _callable_sharp

def cylinder(radius: float = 0.5, height: float = 1.0, round_radius: float = 0.0) -> SDFNode:
    """
    Creates a cylinder centered at the origin, oriented along the Y-axis.

    Args:
        radius (float, optional): The radius of the cylinder. Defaults to 0.5.
        height (float, optional): The total height of the cylinder. Defaults to 1.0.
        round_radius (float, optional): If > 0, rounds the top and bottom edges
                                        of the cylinder. Defaults to 0.0.

    Returns:
        SDFNode: A cylinder primitive.
        
    Example:
        >>> # A tall, thin cylinder
        >>> c = cylinder(radius=0.2, height=3.0)
    """
    return Cylinder(radius, height, round_radius)

class Cone(SDFNode):
    """Represents a cone or frustum primitive."""
    glsl_dependencies = {"primitives"}

    def __init__(self, height: float = 1.0, radius1: float = 0.5, radius2: float = 0.0):
        super().__init__()
        self.height, self.radius1, self.radius2 = height, radius1, radius2
        
    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        h, r1 = _glsl_format(self.height), _glsl_format(self.radius1)
        
        use_capped = False
        if isinstance(self.radius2, (int, float)):
            if self.radius2 > 1e-6:
                use_capped = True
        else:
            use_capped = True
            
        if use_capped:
            dist_expr = f"sdCappedCone({ctx.p}, {h}, {r1}, {_glsl_format(self.radius2)})"
        else:
            dist_expr = f"sdCone({ctx.p}, vec2({h}, {r1}))"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

    def to_callable(self):
        is_dynamic = isinstance(self.height, (str, Param)) or isinstance(self.radius1, (str, Param)) or isinstance(self.radius2, (str, Param))
        if is_dynamic:
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")
            
        h, r1, r2 = self.height, self.radius1, self.radius2

        use_capped = False
        if isinstance(r2, (int, float)):
            if r2 > 1e-6:
                use_capped = True
        else: # Should already be caught by is_dynamic check, but for safety
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
                # Add epsilon to prevent division by zero if h=0 and r1=r2
                clamp_val = np.clip(np.sum(k1_q * k2, axis=-1) / (dot_k2k2 + 1e-9), 0.0, 1.0)
                cb = q - k1 + k2 * clamp_val[:, np.newaxis]
                s = np.where((cb[:, 0] < 0.0) & (ca[:, 1] < 0.0), -1.0, 1.0)
                return s * np.sqrt(np.minimum(np.sum(ca * ca, axis=-1), np.sum(cb * cb, axis=-1)))
            return _callable_capped
        else: # Sharp cone logic
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


def cone(height: float = 1.0, radius1: float = 0.5, radius2: float = 0.0) -> SDFNode:
    """
    Creates a cone or frustum centered at the origin, oriented along the Y-axis.

    The cone's base (radius1) is at y = -height/2 and its top (radius2) is at
    y = +height/2.

    Args:
        height (float, optional): The total height of the cone. Defaults to 1.0.
        radius1 (float, optional): The radius of the base. Defaults to 0.5.
        radius2 (float, optional): The radius of the top. A value of 0 creates a
                                   sharp point. Defaults to 0.0.

    Returns:
        SDFNode: A cone or frustum primitive.
        
    Example:
        >>> # A sharp cone
        >>> c1 = cone(height=2.0, radius1=0.8, radius2=0.0)
        >>> # A frustum (a cone with the top cut off)
        >>> c2 = cone(height=1.5, radius1=0.6, radius2=0.2)
    """
    return Cone(height, radius1, radius2)

class Plane(SDFNode):
    """Represents an infinite plane."""
    glsl_dependencies = {"primitives"}

    def __init__(self, normal, offset: float = 0.0):
        super().__init__()
        self.normal, self.offset = np.array(normal), offset
        
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

def plane(normal, offset: float = 0.0) -> SDFNode:
    """
    Creates an infinite plane.

    Args:
        normal (tuple or np.ndarray): The normal vector of the plane, indicating
                                      which direction it faces.
        offset (float, optional): The distance of the plane from the origin
                                  along its normal. Defaults to 0.0.

    Returns:
        SDFNode: An infinite plane primitive.
        
    Example:
        >>> from sdforge import Y, box
        >>> # Create a floor plane below a box
        >>> floor = plane(normal=Y, offset=-1.0)
        >>> b = box(1.0)
        >>> scene = b | floor
    """
    return Plane(normal, offset)

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

    Returns:
        SDFNode: An octahedron primitive.
        
    Example:
        >>> o = octahedron(size=1.5)
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

def ellipsoid(radii=(1.0, 0.5, 0.5), x: float = None, y: float = None, z: float = None) -> SDFNode:
    """
    Creates an ellipsoid centered at the origin.

    Args:
        radii (tuple, optional): A tuple of the radii along the (X, Y, Z) axes.
                                 Defaults to (1.0, 0.5, 0.5).
        x (float, optional): A convenience argument to specify the x-radius.
                             If x, y, and z are all provided, they override `radii`.
        y (float, optional): Convenience for y-radius.
        z (float, optional): Convenience for z-radius.

    Returns:
        SDFNode: An ellipsoid primitive.
        
    Example:
        >>> # A tall, thin ellipsoid
        >>> e = ellipsoid(radii=(0.2, 1.5, 0.2))
    """
    if x is not None and y is not None and z is not None:
        radii = (x, y, z)
    return Ellipsoid(tuple(radii))
    
class Circle(SDFNode):
    """Represents a 2D circle primitive for extrusion or revolution."""
    glsl_dependencies = {"primitives"}

    def __init__(self, r: float = 1.0):
        super().__init__()
        self.r = r

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        dist_expr = f"sdCircle({ctx.p}.xy, {_glsl_format(self.r)})"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

    def to_callable(self):
        if isinstance(self.r, (str, Param)):
            raise TypeError("Cannot save mesh of an object with animated or interactive parameters.")
        r_val = self.r
        def _callable(points: np.ndarray) -> np.ndarray:
            return np.linalg.norm(points[:, :2], axis=-1) - r_val
        return _callable

def circle(r: float = 1.0) -> SDFNode:
    """
    Creates a 2D circle in the XY plane.

    This is a 2D primitive, primarily used as a profile for 3D shaping
    operations like `.extrude()` or `.revolve()`.

    Args:
        r (float, optional): The radius of the circle. Defaults to 1.0.

    Returns:
        SDFNode: A 2D circle primitive.
    
    Example:
        >>> # Extrude a circle to create a cylinder
        >>> c = circle(r=0.5).extrude(height=2.0)
    """
    return Circle(r)

class Rectangle(SDFNode):
    """Represents a 2D rectangle primitive for extrusion or revolution."""
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

    This is a 2D primitive, primarily used as a profile for 3D shaping
    operations like `.extrude()` or `.revolve()`.

    Args:
        size (float or tuple, optional): The size of the rectangle. If a float,
                                         creates a square. If a tuple, specifies
                                         (width, height). Defaults to 1.0.

    Returns:
        SDFNode: A 2D rectangle primitive.
    
    Example:
        >>> from sdforge import X
        >>> # Revolve a rectangle offset from the Y-axis to create a washer
        >>> washer = rectangle(size=(0.2, 0.1)).translate(X*1.0).revolve()
    """
    if isinstance(size, (int, float, str, Param)):
        size = (size, size)
    return Rectangle(tuple(size))