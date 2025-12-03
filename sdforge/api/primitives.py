import numpy as np
from .core import SDFNode, GLSLContext
from .utils import _glsl_format
from .params import Param

# --- Primitive Classes ---

class Sphere(SDFNode):
    glsl_dependencies = {"primitives"}
    def __init__(self, radius: float = 1.0):
        super().__init__()
        self.radius = radius
    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        dist_expr = f"sdSphere({ctx.p}, {_glsl_format(self.radius)})"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

def sphere(radius: float = 1.0) -> SDFNode:
    """
    Creates a sphere centered at the origin.

    Args:
        radius (float, optional): The radius of the sphere. Defaults to 1.0.
    """
    return Sphere(radius)

class Box(SDFNode):
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
    glsl_dependencies = {"primitives"}
    def __init__(self, radius: float = 1.0, height: float = 1.0):
        super().__init__()
        self.radius = radius
        self.height = height
    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        r = _glsl_format(self.radius)
        h = _glsl_format(self.height / 2.0) if not isinstance(self.height, (str, Param)) else f"({_glsl_format(self.height)})/2.0"
        dist_expr = f"sdHexPrism({ctx.p}, vec2({r}, {h}))"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

def hex_prism(radius: float = 1.0, height: float = 1.0) -> SDFNode:
    """
    Creates a hexagonal prism oriented along the Z axis.

    Args:
        radius (float): The radius (apothem) of the hexagon.
        height (float): The total height (depth) of the prism.
    """
    return HexPrism(radius, height)

class Pyramid(SDFNode):
    glsl_dependencies = {"primitives"}
    def __init__(self, height: float = 1.0):
        super().__init__()
        self.height = height
    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        dist_expr = f"sdPyramid({ctx.p}, {_glsl_format(self.height)})"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

def pyramid(height: float = 1.0) -> SDFNode:
    """
    Creates a pyramid with a square base centered at (0,0,0).

    Args:
        height (float): The vertical height of the pyramid.
    """
    return Pyramid(height)

class Torus(SDFNode):
    glsl_dependencies = {"primitives"}
    def __init__(self, radius_major: float = 1.0, radius_minor: float = 0.25):
        super().__init__()
        self.radius_major, self.radius_minor = radius_major, radius_minor
    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        dist_expr = f"sdTorus({ctx.p}, vec2({_glsl_format(self.radius_major)}, {_glsl_format(self.radius_minor)}))"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

def torus(radius_major: float = 1.0, radius_minor: float = 0.25) -> SDFNode:
    """
    Creates a torus centered at the origin, oriented in the XZ plane.

    Args:
        radius_major (float): Distance from the origin to the center of the tube.
        radius_minor (float): Radius of the tube itself.
    """
    return Torus(radius_major, radius_minor)

class Line(SDFNode):
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

    def to_profile_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        a, b, r = self.start, self.end, _glsl_format(self.radius)
        a_str = f"vec3({_glsl_format(a[0])}, {_glsl_format(a[1])}, 0.0)"
        b_str = f"vec3({_glsl_format(b[0])}, {_glsl_format(b[1])}, 0.0)"
        p_flat = f"vec3({ctx.p}.x, {ctx.p}.y, 0.0)"
        func = "sdCapsule" if self.rounded_caps else "sdCappedCylinder"
        dist_expr = f"{func}({p_flat}, {a_str}, {b_str}, {r})"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

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

class Polyline(SDFNode):
    glsl_dependencies = {"primitives", "operations"}
    def __init__(self, points, radius: float = 0.1, closed: bool = False):
        super().__init__()
        self.points = np.array(points)
        if self.points.shape[1] != 3:
            raise ValueError("Points must be a list of 3D coordinates.")
        self.radius = radius
        self.closed = closed

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        r = _glsl_format(self.radius)
        points = self.points
        num_segments = len(points) if self.closed else len(points) - 1
        if num_segments < 1: return "vec4(1e9, -1.0, 0.0, 0.0)"

        dist_exprs = []
        for i in range(num_segments):
            p0 = points[i]
            p1 = points[(i + 1) % len(points)]
            p0_str = f"vec3({_glsl_format(p0[0])},{_glsl_format(p0[1])},{_glsl_format(p0[2])})"
            p1_str = f"vec3({_glsl_format(p1[0])},{_glsl_format(p1[1])},{_glsl_format(p1[2])})"
            dist_exprs.append(f"sdCapsule({ctx.p}, {p0_str}, {p1_str}, {r})")
        
        current_expr = dist_exprs[0]
        for expr in dist_exprs[1:]:
            current_expr = f"min({current_expr}, {expr})"
        return ctx.new_variable('vec4', f"vec4({current_expr}, -1.0, 0.0, 0.0)")

    def to_profile_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        r = _glsl_format(self.radius)
        points = self.points
        num_segments = len(points) if self.closed else len(points) - 1
        p_flat = f"vec3({ctx.p}.x, {ctx.p}.y, 0.0)"
        dist_exprs = []
        for i in range(num_segments):
            p0 = points[i]
            p1 = points[(i + 1) % len(points)]
            p0_str = f"vec3({_glsl_format(p0[0])},{_glsl_format(p0[1])}, 0.0)"
            p1_str = f"vec3({_glsl_format(p1[0])},{_glsl_format(p1[1])}, 0.0)"
            dist_exprs.append(f"sdCapsule({p_flat}, {p0_str}, {p1_str}, {r})")
        
        current_expr = dist_exprs[0]
        for expr in dist_exprs[1:]:
            current_expr = f"min({current_expr}, {expr})"
        return ctx.new_variable('vec4', f"vec4({current_expr}, -1.0, 0.0, 0.0)")

def polyline(points, radius: float = 0.1, closed: bool = False) -> SDFNode:
    """
    Creates a continuous chain of line segments (capsules) connecting the points.

    Args:
        points (list): A list of (x, y, z) coordinates.
        radius (float, optional): The thickness of the line. Defaults to 0.1.
        closed (bool, optional): If True, connects the last point back to the first. Defaults to False.
    """
    return Polyline(points, radius, closed)

class Polycurve(SDFNode):
    glsl_dependencies = {"primitives", "operations"}
    def __init__(self, points, radius: float = 0.1, closed: bool = False):
        super().__init__()
        self.points = np.array(points)
        if self.points.shape[1] != 3: raise ValueError("Points must be a list of 3D coordinates.")
        self.radius = radius
        self.closed = closed

    def _generate_segments(self):
        points = self.points
        n = len(points)
        segments = []
        if n < 2: return []
        if n == 2 and not self.closed:
            mid = (points[0] + points[1]) / 2.0
            return [(points[0], mid, points[1])]
        if self.closed:
            for i in range(n):
                p_prev, p_curr, p_next = points[(i - 1) % n], points[i], points[(i + 1) % n]
                segments.append(((p_prev + p_curr) / 2.0, p_curr, (p_curr + p_next) / 2.0))
        else:
            segments = [(points[0], points[1], (points[1] + points[2]) / 2.0)]
            for i in range(1, n - 2):
                segments.append(((points[i] + points[i+1]) / 2.0, points[i+1], (points[i+1] + points[i+2]) / 2.0))
            segments.append(((points[-2] + points[-1]) / 2.0, points[-1], points[-1]))
        return segments

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        r_str = _glsl_format(self.radius)
        segments = self._generate_segments()
        if not segments: return "vec4(1e9, -1.0, 0.0, 0.0)"
        dist_exprs = []
        for A, B, C in segments:
            A_str = f"vec3({_glsl_format(A[0])},{_glsl_format(A[1])},{_glsl_format(A[2])})"
            B_str = f"vec3({_glsl_format(B[0])},{_glsl_format(B[1])},{_glsl_format(B[2])})"
            C_str = f"vec3({_glsl_format(C[0])},{_glsl_format(C[1])},{_glsl_format(C[2])})"
            if np.allclose(B, C) or np.allclose(A, B):
                dist_exprs.append(f"sdCapsule({ctx.p}, {A_str}, {C_str}, {r_str})")
            else:
                dist_exprs.append(f"sdBezier({ctx.p}, {A_str}, {B_str}, {C_str}, {r_str})")
        current_expr = dist_exprs[0]
        for expr in dist_exprs[1:]:
            current_expr = f"min({current_expr}, {expr})"
        return ctx.new_variable('vec4', f"vec4({current_expr}, -1.0, 0.0, 0.0)")

    def to_profile_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        r_str = _glsl_format(self.radius)
        segments = self._generate_segments()
        if not segments: return "vec4(1e9, -1.0, 0.0, 0.0)"
        p_flat = f"vec3({ctx.p}.x, {ctx.p}.y, 0.0)"
        dist_exprs = []
        for A, B, C in segments:
            A_str = f"vec3({_glsl_format(A[0])},{_glsl_format(A[1])}, 0.0)"
            B_str = f"vec3({_glsl_format(B[0])},{_glsl_format(B[1])}, 0.0)"
            C_str = f"vec3({_glsl_format(C[0])},{_glsl_format(C[1])}, 0.0)"
            if np.allclose(B, C) or np.allclose(A, B):
                dist_exprs.append(f"sdCapsule({p_flat}, {A_str}, {C_str}, {r_str})")
            else:
                dist_exprs.append(f"sdBezier({p_flat}, {A_str}, {B_str}, {C_str}, {r_str})")
        current_expr = dist_exprs[0]
        for expr in dist_exprs[1:]:
            current_expr = f"min({current_expr}, {expr})"
        return ctx.new_variable('vec4', f"vec4({current_expr}, -1.0, 0.0, 0.0)")

def polycurve(points, radius: float = 0.1, closed: bool = False) -> SDFNode:
    """
    Creates a smooth continuous curve passing near the points (except endpoints).

    Args:
        points (list): A list of (x, y, z) control points.
        radius (float, optional): The thickness of the curve tube. Defaults to 0.1.
        closed (bool, optional): If True, creates a closed loop. Defaults to False.
    """
    return Polycurve(points, radius, closed)

class Bezier(SDFNode):
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
    def to_profile_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        A_str = f"vec3({self.p0[0]},{self.p0[1]}, 0.0)"
        B_str = f"vec3({self.p1[0]},{self.p1[1]}, 0.0)"
        C_str = f"vec3({self.p2[0]},{self.p2[1]}, 0.0)"
        p_flat = f"vec3({ctx.p}.x, {ctx.p}.y, 0.0)"
        dist_expr = f"sdBezier({p_flat}, {A_str}, {B_str}, {C_str}, {_glsl_format(self.radius)})"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

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
    glsl_dependencies = {"primitives"}
    def __init__(self, radius: float = 0.5, height: float = 1.0):
        super().__init__()
        self.radius, self.height = radius, height
    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        h_half = _glsl_format(self.height / 2.0) if not isinstance(self.height, (str, Param)) else f"({_glsl_format(self.height)})/2.0"
        dist_expr = f"sdCylinder({ctx.p}, vec2({_glsl_format(self.radius)}, {h_half}))"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

def cylinder(radius: float = 0.5, height: float = 1.0) -> SDFNode:
    """
    Creates a cylinder centered at the origin, oriented along the Y-axis.

    Args:
        radius (float): The radius of the cylinder.
        height (float): The total height of the cylinder.
    """
    return Cylinder(radius, height)

class Cone(SDFNode):
    glsl_dependencies = {"primitives"}
    def __init__(self, height: float = 1.0, radius_base: float = 0.5, radius_top: float = 0.0):
        super().__init__()
        self.height, self.radius_base, self.radius_top = height, radius_base, radius_top
    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        h, r1 = _glsl_format(self.height), _glsl_format(self.radius_base)
        use_capped = isinstance(self.radius_top, (int, float)) and self.radius_top > 1e-6 or not isinstance(self.radius_top, (int, float))
        if use_capped:
            dist_expr = f"sdCappedCone({ctx.p}, {h}, {r1}, {_glsl_format(self.radius_top)})"
        else:
            dist_expr = f"sdCone({ctx.p}, vec2({h}, {r1}))"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

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
    glsl_dependencies = {"primitives"}
    def __init__(self, normal, point=(0,0,0)):
        super().__init__()
        self.normal = np.array(normal)
        if np.linalg.norm(self.normal) > 0: self.normal = self.normal / np.linalg.norm(self.normal)
        self.point = np.array(point)
        self.offset = -np.dot(self.point, self.normal)
    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        n = self.normal
        dist_expr = f"sdPlane({ctx.p}, vec4({n[0]}, {n[1]}, {n[2]}, {_glsl_format(self.offset)}))"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

def plane(normal, point=(0,0,0)) -> SDFNode:
    """
    Creates an infinite plane defined by a normal and a point.

    Args:
        normal (tuple): Normal vector.
        point (tuple): A point on the plane. Defaults to origin.
    """
    return Plane(normal, point)

class Octahedron(SDFNode):
    glsl_dependencies = {"primitives"}
    def __init__(self, size: float = 1.0):
        super().__init__()
        self.size = size
    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        dist_expr = f"sdOctahedron({ctx.p}, {_glsl_format(self.size)})"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

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
    glsl_dependencies = {"primitives"}
    def __init__(self, radii: tuple = (1.0, 0.5, 0.5)):
        super().__init__()
        self.radii = np.array(radii)
    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        r = [_glsl_format(v) for v in self.radii]
        dist_expr = f"sdEllipsoid({ctx.p}, vec3({r[0]}, {r[1]}, {r[2]}))"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

def ellipsoid(radii=(1.0, 0.5, 0.5)) -> SDFNode:
    """
    Creates an ellipsoid centered at the origin.

    Args:
        radii (tuple): Radii along (X, Y, Z).
    """
    return Ellipsoid(tuple(radii))
    
class Circle(SDFNode):
    glsl_dependencies = {"primitives"}
    def __init__(self, radius: float = 1.0):
        super().__init__()
        self.radius = radius
    def to_profile_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        dist_expr = f"sdCircle({ctx.p}.xy, {_glsl_format(self.radius)})"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")
    def to_glsl(self, ctx: GLSLContext) -> str:
        profile_var = self.to_profile_glsl(ctx)
        dist_expr = f"max({profile_var}.x, abs({ctx.p}.z) - 0.001)"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, {profile_var}.yzw)")

def circle(radius: float = 1.0) -> SDFNode:
    """
    Creates a 2D circle in the XY plane. 
    By default, this renders as a thin disc in 3D. 
    Use .extrude() to create a cylinder.

    Args:
        radius (float): Radius of the circle.
    """
    return Circle(radius)

class Rectangle(SDFNode):
    glsl_dependencies = {"primitives"}
    def __init__(self, size: tuple = (1.0, 1.0)):
        super().__init__()
        self.size = np.array(size)
    def to_profile_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        s = [_glsl_format(v) for v in self.size]
        size_vec = f"vec2({s[0]}/2.0, {s[1]}/2.0)"
        dist_expr = f"sdRectangle({ctx.p}.xy, {size_vec})"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")
    def to_glsl(self, ctx: GLSLContext) -> str:
        profile_var = self.to_profile_glsl(ctx)
        dist_expr = f"max({profile_var}.x, abs({ctx.p}.z) - 0.001)"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, {profile_var}.yzw)")

def rectangle(size=1.0) -> SDFNode:
    """
    Creates a 2D rectangle in the XY plane.
    By default, this renders as a thin plate in 3D.
    Use .extrude() to create a box.

    Args:
        size (float): Size of the rectangle.
    """
    if isinstance(size, (int, float, str, Param)):
        size = (size, size)
    return Rectangle(tuple(size))

class Triangle(SDFNode):
    glsl_dependencies = {"primitives"}
    def __init__(self, radius: float = 1.0):
        super().__init__()
        self.radius = radius
    def to_profile_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        dist_expr = f"sdEquilateralTriangle({ctx.p}.xy, {_glsl_format(self.radius)})"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")
    def to_glsl(self, ctx: GLSLContext) -> str:
        profile_var = self.to_profile_glsl(ctx)
        dist_expr = f"max({profile_var}.x, abs({ctx.p}.z) - 0.001)"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, {profile_var}.yzw)")

def triangle(radius: float = 1.0) -> SDFNode:
    """
    Creates a 2D equilateral triangle.
    
    Args:
        radius (float): The radius (apothem) of the triangle.
    """
    return Triangle(radius)

class Trapezoid(SDFNode):
    glsl_dependencies = {"primitives"}
    def __init__(self, bottom_width: float = 1.0, top_width: float = 0.5, height: float = 0.5):
        super().__init__()
        self.bottom_width, self.top_width, self.height = bottom_width, top_width, height
    def to_profile_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        r1 = _glsl_format(self.bottom_width / 2.0)
        r2 = _glsl_format(self.top_width / 2.0)
        he = _glsl_format(self.height / 2.0)
        dist_expr = f"sdTrapezoid({ctx.p}.xy, {r1}, {r2}, {he})"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")
    def to_glsl(self, ctx: GLSLContext) -> str:
        profile_var = self.to_profile_glsl(ctx)
        dist_expr = f"max({profile_var}.x, abs({ctx.p}.z) - 0.001)"
        return ctx.new_variable('vec4', f"vec4({dist_expr}, {profile_var}.yzw)")

def trapezoid(bottom_width: float = 1.0, top_width: float = 0.5, height: float = 0.5) -> SDFNode:
    """
    Creates an isosceles trapezoid.

    Args:
        bottom_width (float): Width at the bottom.
        top_width (float): Width at the top.
        height (float): Height of the trapezoid.
    """
    return Trapezoid(bottom_width, top_width, height)