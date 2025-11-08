import numpy as np
from abc import ABC, abstractmethod
import sys

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
        self.definitions = set() # For node-specific GLSL function definitions

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
        new_ctx.definitions = self.definitions.copy()
        new_ctx._var_counter = self._var_counter
        return new_ctx

    def merge_from(self, sub_context: 'GLSLContext'):
        """Merges statements and state from a sub-context into this one."""
        self.statements.extend(sub_context.statements)
        self.dependencies.update(sub_context.dependencies)
        self.definitions.update(sub_context.definitions)
        self._var_counter = sub_context._var_counter


class SDFNode(ABC):
    """Abstract base class for all SDF objects in the scene graph."""
    
    glsl_dependencies = set() # Default empty set

    def __init__(self):
        super().__init__()
        # Special case for Revolve, which has no child in __init__
        if not hasattr(self, 'child'):
            self.child = None

    def _collect_params(self, params: dict):
        """Recursively collects Param objects from the scene graph."""
        from .api.params import Param
        # Inspect all public attributes of the current object.
        for attr_name in dir(self):
            if attr_name.startswith('_') or attr_name in ['child', 'children']:
                continue
            try:
                attr_val = getattr(self, attr_name)
                if isinstance(attr_val, Param):
                    params[attr_val.uniform_name] = attr_val
                elif isinstance(attr_val, (list, tuple, np.ndarray)):
                    for item in attr_val:
                        if isinstance(item, Param):
                            params[item.uniform_name] = item
            except Exception:
                continue # Gracefully skip attributes that might fail

        # Recurse into children
        if hasattr(self, 'child') and self.child:
            self.child._collect_params(params)
        if hasattr(self, 'children'):
            for child in self.children:
                child._collect_params(params)
    
    def _collect_materials(self, materials: list):
        """Recursively collects Material objects from the scene graph."""
        if hasattr(self, 'child') and self.child:
            self.child._collect_materials(materials)
        if hasattr(self, 'children'):
            for child in self.children:
                child._collect_materials(materials)

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

    def render(self, camera=None, light=None, debug=None, **kwargs):
        """Renders the SDF object in a live-updating viewer."""
        from .render import render as render_func
        render_func(self, camera=camera, light=light, debug=debug, **kwargs)

    def save(self, path, bounds=None, samples=2**22, verbose=True, algorithm='marching_cubes', adaptive=False, vertex_colors=False):
        """
        Generates a mesh and saves it to a file.

        Args:
            path (str): The file path to save to (e.g., 'model.stl', 'model.glb').
            bounds (tuple, optional): The bounding box to mesh within. If None, it will be automatically estimated.
            samples (int, optional): The number of points to sample in the volume.
            verbose (bool, optional): Whether to print progress information.
            algorithm (str, optional): Meshing algorithm to use. Currently only 'marching_cubes' is supported.
            adaptive (bool, optional): Whether to use adaptive meshing. Not currently implemented.
            vertex_colors (bool, optional): Whether to include vertex colors in the export (for .glb/.gltf). Not currently implemented.
        """
        if bounds is None:
            if verbose:
                print("INFO: No bounds provided to .save(), estimating automatically.", file=sys.stderr)
            bounds = self.estimate_bounds(verbose=verbose)

        from . import mesh
        mesh.save(self, path, bounds, samples, verbose, algorithm, adaptive, vertex_colors)

    def save_frame(self, path, camera=None, light=None, **kwargs):
        """Renders a single frame and saves it to an image file (e.g., '.png')."""
        self.render(save_frame=path, watch=False, camera=camera, light=light, **kwargs)

    def estimate_bounds(self, resolution=64, search_bounds=((-2, -2, -2), (2, 2, 2)), padding=0.1, verbose=True):
        """
        Estimates the bounding box of the SDF object by sampling a grid.

        Args:
            resolution (int, optional): The number of points to sample along each axis.
            search_bounds (tuple, optional): The initial cube volume to search for the object.
            padding (float, optional): A padding factor to add to the estimated bounds.
            verbose (bool, optional): If True, prints progress information.

        Returns:
            tuple: A tuple of ((min_x, min_y, min_z), (max_x, max_y, max_z)).
        """
        from .mesh import _cartesian_product
        if verbose:
            print(f"INFO: Estimating bounds with {resolution**3} samples...", file=sys.stderr)

        sdf_callable = self.to_callable()

        axes = [np.linspace(search_bounds[0][i], search_bounds[1][i], resolution) for i in range(3)]
        points_grid = _cartesian_product(*axes).astype('f4')

        distances = sdf_callable(points_grid)
        inside_mask = distances <= 1e-4
        inside_points = points_grid[inside_mask]

        if inside_points.shape[0] < 2:
            if verbose:
                print(f"WARNING: No object surface found within the search bounds {search_bounds}. Returning search_bounds.", file=sys.stderr)
            return search_bounds

        min_coords = np.min(inside_points, axis=0)
        max_coords = np.max(inside_points, axis=0)
        
        size = max_coords - min_coords
        size[size < 1e-6] = padding
        min_coords -= size * padding
        max_coords += size * padding

        bounds = (tuple(min_coords), tuple(max_coords))
        if verbose:
            print(f"SUCCESS: Estimated bounds: {bounds}", file=sys.stderr)
            
        return bounds

    def export_shader(self, path: str):
        """
        Exports a complete, self-contained GLSL fragment shader for the current scene.
        
        Args:
            path (str): The file path to save the GLSL shader to (e.g., 'my_scene.glsl').
        """
        from .export import assemble_standalone_shader
        shader_code = assemble_standalone_shader(self)
        with open(path, 'w') as f:
            f.write(shader_code)
        print(f"SUCCESS: Shader exported to '{path}'.")

    def _collect_uniforms(self, uniforms: dict):
        """Recursively collects uniforms from the scene graph."""
        if hasattr(self, 'child') and self.child:
            self.child._collect_uniforms(uniforms)
        if hasattr(self, 'children'):
            for child in self.children:
                child._collect_uniforms(uniforms)

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

    # --- Material ---
    def color(self, r: float, g: float, b: float) -> 'SDFNode':
        """Applies a color material to the object."""
        from .api.material import Material
        return Material(self, (r, g, b))

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