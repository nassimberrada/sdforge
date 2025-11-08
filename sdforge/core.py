import numpy as np
from abc import ABC, abstractmethod

# Cardinal axis constants
X, Y, Z = np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])

class GLSLContext:
    """Manages the state of the GLSL compilation process for a scene."""
    def __init__(self, compiler):
        self.compiler = compiler
        self.p = "p"  # The name of the current point variable being evaluated
        self.statements = []
        self.dependencies = set()
        self._var_counter = 0

    def add_statement(self, line: str):
        """Adds a line of code to the current function body."""
        self.statements.append(line)

    def new_variable(self, type: str, expression: str) -> str:
        """Declares a new GLSL variable and returns its name."""
        name = f"var_{self._var_counter}"
        self._var_counter += 1
        self.add_statement(f"{type} {name} = {expression};")
        return name

    def with_p(self, new_p_name: str) -> 'GLSLContext':
        """Creates a sub-context for a child node with a transformed point."""
        new_ctx = GLSLContext(self.compiler)
        new_ctx.p = new_p_name
        # Inherit dependencies and counter state from parent
        new_ctx.dependencies = self.dependencies.copy()
        new_ctx._var_counter = self._var_counter
        return new_ctx

    def merge_from(self, sub_context: 'GLSLContext'):
        """Merges statements and state from a sub-context into this one."""
        self.statements.extend(sub_context.statements)
        self.dependencies.update(sub_context.dependencies)
        self._var_counter = sub_context._var_counter


class SDFNode(ABC):
    """Abstract base class for all SDF objects in the scene graph."""
    
    glsl_dependencies = set() # Default empty set

    def __init__(self):
        super().__init__()
        # Special case for Revolve, which has no child in __init__
        if not hasattr(self, 'child'):
            self.child = None

    @abstractmethod
    def to_glsl(self, ctx: GLSLContext) -> str:
        """
        Contributes to the GLSL compilation and returns the name of the
        GLSL variable holding the vec4 result (dist, mat_id, 0, 0).
        """
        raise NotImplementedError

    @abstractmethod
    def to_callable(self):
        """
        Returns a Python function that takes a NumPy array of points (N, 3)
        and returns an array of distances (N,).
        """
        raise NotImplementedError

    def render(self, **kwargs):
        """Renders the SDF object in a live-updating viewer."""
        from .engine import render as render_func
        render_func(self, **kwargs)

    # --- Boolean Operations ---
    def union(self, *others, k: float = 0.0) -> 'SDFNode':
        """Creates a union of this object and others, with optional smoothness."""
        from .api.operations import Union
        return Union(children=[self] + list(others), k=k)

    def intersection(self, *others, k: float = 0.0) -> 'SDFNode':
        """Creates an intersection of this object and others, with optional smoothness."""
        from .api.operations import Intersection
        return Intersection(children=[self] + list(others), k=k)

    def difference(self, other, k: float = 0.0) -> 'SDFNode':
        """Subtracts another object from this one, with optional smoothness."""
        from .api.operations import Difference
        return Difference(self, other, k=k)

    def __or__(self, other):
        """Operator overload for a simple union: `shape1 | shape2`."""
        return self.union(other)

    def __and__(self, other):
        """Operator overload for a simple intersection: `shape1 & shape2`."""
        return self.intersection(other)

    def __sub__(self, other):
        """Operator overload for a simple difference: `shape1 - shape2`."""
        return self.difference(other)
        
    # --- Transformations ---
    def translate(self, offset) -> 'SDFNode':
        """Moves the object in space."""
        from .api.transforms import Translate
        return Translate(self, offset)

    def scale(self, factor) -> 'SDFNode':
        """Scales the object. Can be a uniform float or a (x, y, z) tuple."""
        from .api.transforms import Scale
        return Scale(self, factor)

    def rotate(self, axis, angle: float) -> 'SDFNode':
        """Rotates the object around a cardinal axis by an angle in radians."""
        from .api.transforms import Rotate
        return Rotate(self, axis, angle)
        
    def __add__(self, offset):
        """Operator overload for translation: `shape + (x, y, z)`."""
        return self.translate(offset)
        
    def __mul__(self, factor):
        """Operator overload for uniform scaling: `shape * 2.0`."""
        return self.scale(factor)

    def __rmul__(self, factor):
        """Operator overload for uniform scaling: `2.0 * shape`."""
        return self.scale(factor)

    def orient(self, axis) -> 'SDFNode':
        """Orients the object along a primary axis (e.g., 'x', 'y', 'z' or vector)."""
        from .api.transforms import Orient
        axis_map = {'x': X, 'y': Y, 'z': Z}
        if isinstance(axis, str) and axis.lower() in axis_map:
            axis = axis_map[axis.lower()]
        return Orient(self, axis)

    def twist(self, k: float) -> 'SDFNode':
        """Twists the object around the Y-axis."""
        from .api.transforms import Twist
        return Twist(self, k)

    def bend(self, axis, k: float) -> 'SDFNode':
        """Bends the object around a cardinal axis."""
        from .api.transforms import Bend
        return Bend(self, axis, k)
        
    def repeat(self, spacing) -> 'SDFNode':
        """Repeats the object infinitely with a given spacing vector."""
        from .api.transforms import Repeat
        return Repeat(self, spacing)

    def limited_repeat(self, spacing, limits) -> 'SDFNode':
        """Repeats the object a limited number of times along each axis."""
        from .api.transforms import LimitedRepeat
        return LimitedRepeat(self, spacing, limits)

    def polar_repeat(self, repetitions: int) -> 'SDFNode':
        """Repeats the object in a circle around the Y-axis."""
        from .api.transforms import PolarRepeat
        return PolarRepeat(self, repetitions)

    def mirror(self, axes) -> 'SDFNode':
        """Mirrors the object across one or more axes (e.g., X, Y, X|Z)."""
        from .api.transforms import Mirror
        return Mirror(self, axes)

    # --- Shaping Operations ---
    def round(self, radius: float) -> 'SDFNode':
        """Rounds all edges of the object by a given radius."""
        from .api.shaping import Round
        return Round(self, radius)

    def shell(self, thickness: float) -> 'SDFNode':
        """Creates a shell or outline of the object with a given thickness."""
        from .api.shaping import Bevel
        return Bevel(self, thickness)

    def bevel(self, thickness: float) -> 'SDFNode':
        """Alias for .shell(). Creates an outline of the object."""
        return self.shell(thickness)

    def extrude(self, height: float) -> 'SDFNode':
        """Extrudes a 2D SDF shape along the Z-axis."""
        from .api.shaping import Extrude
        return Extrude(self, height)

    def revolve(self) -> 'SDFNode':
        """Revolves a 2D SDF shape around the Y-axis."""
        from .api.shaping import Revolve
        # Revolve is special: it becomes the parent of the current node
        r = Revolve()
        r.child = self
        return r

    # --- Surface Displacement ---
    def displace(self, displacement_glsl: str) -> 'SDFNode':
        """Displaces the surface of the object using a GLSL expression."""
        from .api.noise import Displace
        return Displace(self, displacement_glsl)

    def displace_by_noise(self, scale: float = 10.0, strength: float = 0.1) -> 'SDFNode':
        """Displaces the surface using a procedural noise function."""
        from .api.noise import DisplaceByNoise
        return DisplaceByNoise(self, scale, strength)

    # --- Stubs for future functionality ---
    def color(self, r, g, b): raise NotImplementedError("Materials not implemented yet.")