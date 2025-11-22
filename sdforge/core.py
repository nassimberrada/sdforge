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
        from .utils import Expr
        # Inspect all public attributes of the current object.
        for attr_name in dir(self):
            if attr_name.startswith('_') or attr_name in ['child', 'children']:
                continue
            try:
                attr_val = getattr(self, attr_name)
                if isinstance(attr_val, Param):
                    params[attr_val.uniform_name] = attr_val
                elif isinstance(attr_val, Expr): 
                    for p in attr_val.params:
                        params[p.uniform_name] = p
                elif isinstance(attr_val, (list, tuple, np.ndarray)):
                    for item in attr_val:
                        if isinstance(item, Param):
                            params[item.uniform_name] = item
                        elif isinstance(item, Expr):
                             for p in item.params:
                                 params[p.uniform_name] = p
            except Exception:
                continue

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

    def save(self, path, bounds=None, samples=2**22, verbose=True, algorithm='marching_cubes', adaptive=False, octree_depth=8, vertex_colors=False, decimate_ratio=None):
        """
        Generates a mesh and saves it to a file.

        Args:
            path (str): The file path to save to (e.g., 'model.stl', 'model.glb').
            bounds (tuple, optional): The bounding box to mesh within. If None, it will be automatically estimated.
            samples (int, optional): The number of points to sample for uniform grid meshing.
                                     Higher is more detailed but slower. Ignored if `adaptive=True`.
            verbose (bool, optional): Whether to print progress information.
            algorithm (str, optional): Meshing algorithm to use. Supported options:
                                       'marching_cubes', 'dual_contouring'. Defaults to 'marching_cubes'.
            adaptive (bool, optional): If True, uses an octree-based adaptive meshing algorithm which is often
                                       faster and more memory-efficient than a uniform grid. Defaults to False.
                                       Note: Not currently compatible with 'dual_contouring'.
            octree_depth (int, optional): The maximum subdivision depth for adaptive meshing.
                                          Higher is more detailed. Effective resolution is 2**octree_depth.
                                          Ignored if `adaptive=False`. Defaults to 8.
            vertex_colors (bool, optional): Whether to include vertex colors in the export (for .glb/.gltf). Not currently implemented.
            decimate_ratio (float, optional): If specified, simplifies the mesh to reduce triangle count.
                                           A value of 0.9 aims to remove 90% of the triangles.
                                           Requires the 'trimesh' library. Defaults to None (no simplification).
        """
        if bounds is None:
            if verbose:
                print("INFO: No bounds provided to .save(), estimating automatically.", file=sys.stderr)
            bounds = self.estimate_bounds(verbose=verbose)

        from . import mesh
        mesh.save(self, path, bounds, samples, verbose, algorithm, adaptive, vertex_colors, decimate_ratio, octree_depth=octree_depth)

    def save_frame(self, path, camera=None, light=None, **kwargs):
        """
        Renders a single frame and saves it to an image file.

        This is a convenience method that calls `.render()` with the appropriate
        arguments to produce a single image output instead of launching the
        interactive viewer.

        Args:
            path (str): The file path to save the image to (e.g., 'frame.png').
            camera (Camera, optional): A Camera object to define the view.
                                       Defaults to None (uses default view).
            light (Light, optional): A Light object to define the scene lighting.
                                     Defaults to None (uses default lighting).
            **kwargs: Additional keyword arguments passed to the renderer,
                      such as `width` and `height`.
        """
        self.render(save_frame=path, watch=False, camera=camera, light=light, **kwargs)

    def estimate_bounds(self, resolution=64, search_bounds=((-2, -2, -2), (2, 2, 2)), padding=0.1, verbose=True):
        """
        Estimates the bounding box of the SDF object by sampling a grid.
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
    def union(self, *others, blend: float = 0.0, blend_type: str = 'smooth') -> 'SDFNode':
        """
        Creates the union (join) of this object and one or more other objects.

        Args:
            *others (SDFNode): A variable number of other SDF objects to union with.
            blend (float, optional): The amount of blending at the seam. Defaults to 0.0 (sharp).
            blend_type (str, optional): The type of blend ('smooth' for fillet, 'linear' for chamfer).
                                        Defaults to 'smooth'.
        """
        from .api.operations import Union
        return Union(children=[self] + list(others), blend=blend, blend_type=blend_type)

    def intersection(self, *others, blend: float = 0.0, blend_type: str = 'smooth') -> 'SDFNode':
        """
        Creates the intersection (common area) of this object and others.

        Args:
            *others (SDFNode): A variable number of other SDF objects to intersect with.
            blend (float, optional): The amount of blending at the intersection. Defaults to 0.0.
            blend_type (str, optional): The type of blend ('smooth' for fillet, 'linear' for chamfer).
                                        Defaults to 'smooth'.
        """
        from .api.operations import Intersection
        return Intersection(children=[self] + list(others), blend=blend, blend_type=blend_type)

    def difference(self, other, blend: float = 0.0, blend_type: str = 'smooth') -> 'SDFNode':
        """
        Subtracts another object from this one.

        Args:
            other (SDFNode): The object to subtract.
            blend (float, optional): The amount of blending at the cut. Defaults to 0.0.
            blend_type (str, optional): The type of blend ('smooth' for fillet, 'linear' for chamfer).
                                        Defaults to 'smooth'.
        """
        from .api.operations import Difference
        return Difference(self, other, blend=blend, blend_type=blend_type)

    def __or__(self, other):
        """Operator overload for a simple union: `shape1 | shape2`."""
        return self.union(other)

    def __and__(self, other):
        """Operator overload for a simple intersection: `shape1 & shape2`."""
        return self.intersection(other)

    def __sub__(self, other):
        """Operator overload for a simple difference: `shape1 - shape2`."""
        return self.difference(other)

    # --- Fluent Constraints ---
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
        from .utils import Expr
        # Inspect all public attributes of the current object.
        for attr_name in dir(self):
            if attr_name.startswith('_') or attr_name in ['child', 'children']:
                continue
            try:
                attr_val = getattr(self, attr_name)
                if isinstance(attr_val, Param):
                    params[attr_val.uniform_name] = attr_val
                elif isinstance(attr_val, Expr): 
                    for p in attr_val.params:
                        params[p.uniform_name] = p
                elif isinstance(attr_val, (list, tuple, np.ndarray)):
                    for item in attr_val:
                        if isinstance(item, Param):
                            params[item.uniform_name] = item
                        elif isinstance(item, Expr):
                             for p in item.params:
                                 params[p.uniform_name] = p
            except Exception:
                continue

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

    def save(self, path, bounds=None, samples=2**22, verbose=True, algorithm='marching_cubes', adaptive=False, octree_depth=8, vertex_colors=False, decimate_ratio=None):
        """
        Generates a mesh and saves it to a file.
        """
        if bounds is None:
            if verbose:
                print("INFO: No bounds provided to .save(), estimating automatically.", file=sys.stderr)
            bounds = self.estimate_bounds(verbose=verbose)

        from . import mesh
        mesh.save(self, path, bounds, samples, verbose, algorithm, adaptive, vertex_colors, decimate_ratio, octree_depth=octree_depth)

    def save_frame(self, path, camera=None, light=None, **kwargs):
        """
        Renders a single frame and saves it to an image file.
        """
        self.render(save_frame=path, watch=False, camera=camera, light=light, **kwargs)

    def estimate_bounds(self, resolution=64, search_bounds=((-2, -2, -2), (2, 2, 2)), padding=0.1, verbose=True):
        """
        Estimates the bounding box of the SDF object by sampling a grid.
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
    def union(self, *others, blend: float = 0.0, blend_type: str = 'smooth') -> 'SDFNode':
        """
        Creates the union (join) of this object and one or more other objects.

        Args:
            *others (SDFNode): A variable number of other SDF objects to union with.
            blend (float, optional): The amount of blending at the seam. Defaults to 0.0 (sharp).
            blend_type (str, optional): The type of blend ('smooth' for fillet, 'chamfer' for linear).
                                        Defaults to 'smooth'.
        """
        from .api.operations import Union
        return Union(children=[self] + list(others), blend=blend, blend_type=blend_type)

    def intersection(self, *others, blend: float = 0.0, blend_type: str = 'smooth') -> 'SDFNode':
        """
        Creates the intersection (common area) of this object and others.

        Args:
            *others (SDFNode): A variable number of other SDF objects to intersect with.
            blend (float, optional): The amount of blending at the intersection. Defaults to 0.0.
            blend_type (str, optional): The type of blend ('smooth' for fillet, 'chamfer' for linear).
                                        Defaults to 'smooth'.
        """
        from .api.operations import Intersection
        return Intersection(children=[self] + list(others), blend=blend, blend_type=blend_type)

    def difference(self, other, blend: float = 0.0, blend_type: str = 'smooth') -> 'SDFNode':
        """
        Subtracts another object from this one.

        Args:
            other (SDFNode): The object to subtract.
            blend (float, optional): The amount of blending at the cut. Defaults to 0.0.
            blend_type (str, optional): The type of blend ('smooth' for fillet, 'chamfer' for linear).
                                        Defaults to 'smooth'.
        """
        from .api.operations import Difference
        return Difference(self, other, blend=blend, blend_type=blend_type)

    def __or__(self, other):
        """Operator overload for a simple union: `shape1 | shape2`."""
        return self.union(other)

    def __and__(self, other):
        """Operator overload for a simple intersection: `shape1 & shape2`."""
        return self.intersection(other)

    def __sub__(self, other):
        """Operator overload for a simple difference: `shape1 - shape2`."""
        return self.difference(other)

    # --- Fluent Constraints ---
    def align_to(self, reference_point, face_normal, offset: float = 0.0) -> 'SDFNode':
        """
        Aligns the object's primary axis (Y) to a specific face normal at a target point.

        Args:
            reference_point (tuple or np.ndarray): The point on the target face where the object
                                                   will be placed.
            face_normal (tuple or np.ndarray): The normal vector of the target face (e.g., `sdforge.X`).
                                               Currently supports cardinal axes.
            offset (float, optional): An additional distance to move the object along the normal.
                                      Defaults to 0.0.
        """
        from .api.constraints import align_to_face
        return align_to_face(self, reference_point, face_normal, offset)

    def place_at_angle(self, pivot_point, axis, angle_rad, distance) -> 'SDFNode':
        """
        Places the object at a specific angle and distance from a pivot point.

        Args:
            pivot_point (tuple or np.ndarray): The center of rotation.
            axis (tuple or np.ndarray): The axis of rotation (e.g., `sdforge.Y`).
            angle_rad (float): The angle in radians to rotate the object around the pivot.
            distance (float): The radial distance from the pivot point to place the object.
        """
        from .api.constraints import place_at_angle
        return place_at_angle(self, pivot_point, axis, angle_rad, distance)

    def offset_along(self, reference_point, direction, distance) -> 'SDFNode':
        """
        Translates the object relative to a reference point along a specific direction vector.

        Args:
            reference_point (tuple or np.ndarray): The starting point for the calculation.
            direction (tuple or np.ndarray): The direction vector (will be normalized).
            distance (float): The distance to move along the direction vector.
        """
        from .api.constraints import offset_along
        return offset_along(self, reference_point, direction, distance)

    def bounding_box(self, padding: float = 0.0) -> 'SDFNode':
        """
        Creates a new box primitive that completely encloses this object.

        Args:
            padding (float, optional): Extra space to add to all sides of the bounding box.
                                       Defaults to 0.0.
        """
        from .api.constraints import bounding_box
        return bounding_box(self, padding)

    # --- Material ---
    def color(self, r, g=None, b=None) -> 'SDFNode':
        """
        Applies a color material to the object.

        Args:
            r: Red component OR a tuple/list of (r, g, b).
            g: Green component (optional if r is a tuple).
            b: Blue component (optional if r is a tuple).
        """
        from .api.material import Material
        if isinstance(r, (list, tuple, np.ndarray)):
            return Material(self, tuple(r))
        return Material(self, (r, g, b))

    # --- Transformations ---
    def translate(self, offset) -> 'SDFNode':
        """
        Moves the object in space by a given offset vector.

        Args:
            offset (tuple or np.ndarray): The (x, y, z) vector to move the object by.
        """
        from .api.transforms import Translate
        return Translate(self, offset)

    def scale(self, factor) -> 'SDFNode':
        """
        Scales the object.

        The scaling can be uniform (a single float) or non-uniform
        (a tuple of factors for x, y, and z).

        Args:
            factor (float or tuple): The scaling factor. If a float, scaling is
                                     uniform. If a tuple, it is (sx, sy, sz).
        """
        from .api.transforms import Scale
        return Scale(self, factor)

    def rotate(self, axis, angle: float) -> 'SDFNode':
        """
        Rotates the object around an axis.

        Args:
            axis (tuple or np.ndarray): The axis of rotation (x, y, z).
                                        Can be a cardinal axis like `sdforge.X` 
                                        or an arbitrary vector like `(1, 1, 0)`.
            angle (float): The angle of rotation in radians.
        """
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
        """
        Orients the object's longest dimension along a primary axis.

        Args:
            axis (str or np.ndarray): The target axis, e.g., 'x', 'y', 'z',
                                      or the corresponding vectors `X`, `Y`, `Z`.
        """
        from .api.transforms import Orient
        axis_map = {'x': X, 'y': Y, 'z': Z}
        if isinstance(axis, str) and axis.lower() in axis_map:
            axis = axis_map[axis.lower()]
        return Orient(self, axis)

    def twist(self, strength: float) -> 'SDFNode':
        """
        Twists the object around its Y-axis.

        Args:
            strength (float): The amount of twist in radians per unit of height.
        """
        from .api.transforms import Twist
        return Twist(self, strength)

    def bend(self, axis, curvature: float) -> 'SDFNode':
        """
        Bends the object into an arc around a cardinal axis.

        Args:
            axis (np.ndarray): The axis to bend around (`X`, `Y`, or `Z`).
            curvature (float): The curvature amount.
        """
        from .api.transforms import Bend
        return Bend(self, axis, curvature)

    def repeat(self, spacing) -> 'SDFNode':
        """
        Repeats the object infinitely across a grid.

        Args:
            spacing (tuple or np.ndarray): The (x, y, z) spacing of the grid.
                                           A spacing of 0 along an axis means
                                           no repetition on that axis.
        """
        from .api.transforms import Repeat
        return Repeat(self, spacing)

    def limited_repeat(self, spacing, limits) -> 'SDFNode':
        """
        Repeats the object a finite number of times.

        Args:
            spacing (tuple): The (x, y, z) spacing of the grid.
            limits (tuple): The (nx, ny, nz) integer number of repetitions in
                            each positive and negative direction. A limit of
                            (2, 0, 0) creates 5 instances: -2, -1, 0, 1, 2.
        """
        from .api.transforms import LimitedRepeat
        return LimitedRepeat(self, spacing, limits)

    def polar_repeat(self, repetitions: int) -> 'SDFNode':
        """
        Repeats the object in a circle around the Y-axis.

        Args:
            repetitions (int): The number of times to repeat the object.
        """
        from .api.transforms import PolarRepeat
        return PolarRepeat(self, repetitions)

    def mirror(self, axes) -> 'SDFNode':
        """
        Mirrors the object across one or more axes to create symmetry.

        Args:
            axes (np.ndarray): The axes to mirror across. Can be combined with `|`,
                               e.g., `X | Y` to mirror across both X and Y.
        """
        from .api.transforms import Mirror
        return Mirror(self, axes)

    # --- Shaping Operations ---
    def round(self, radius: float) -> 'SDFNode':
        """
        Rounds all edges of the object by a given radius.

        Args:
            radius (float): The radius of the rounding effect.
        """
        from .api.shaping import Round
        return Round(self, radius)

    def shell(self, thickness: float) -> 'SDFNode':
        """
        Creates a hollow shell or outline of the object.

        Args:
            thickness (float): The thickness of the shell wall.
        """
        from .api.shaping import Shell
        return Shell(self, thickness)

    def extrude(self, height: float) -> 'SDFNode':
        """
        Extrudes a 2D SDF shape along the Z-axis to create a 3D object.

        Args:
            height (float): The total height of the extrusion.
        """
        from .api.shaping import Extrude
        return Extrude(self, height)

    def revolve(self) -> 'SDFNode':
        """
        Revolves a 2D SDF shape around the Y-axis to create a 3D object.
        """
        from .api.shaping import Revolve
        r = Revolve()
        r.child = self
        return r

    # --- Surface Displacement ---
    def displace(self, displacement_glsl: str) -> 'SDFNode':
        """
        Displaces the surface of the object using a custom GLSL expression.

        Args:
            displacement_glsl (str): A GLSL expression that evaluates to a float.
        """
        from .api.noise import Displace
        return Displace(self, displacement_glsl)

    def displace_by_noise(self, scale: float = 10.0, strength: float = 0.1) -> 'SDFNode':
        """
        Displaces the surface using a built-in procedural noise function.

        Args:
            scale (float, optional): The frequency of the noise.
            strength (float, optional): The amplitude of the displacement.
        """
        from .api.noise import DisplaceByNoise
        return DisplaceByNoise(self, scale, strength)

    # --- Material ---
    def color(self, r, g=None, b=None) -> 'SDFNode':
        """
        Applies a color material to the object.

        Args:
            r: Red component OR a tuple/list of (r, g, b).
            g: Green component (optional if r is a tuple).
            b: Blue component (optional if r is a tuple).
        """
        from .api.material import Material
        if isinstance(r, (list, tuple, np.ndarray)):
            return Material(self, tuple(r))
        return Material(self, (r, g, b))

    # --- Transformations ---
    def translate(self, offset) -> 'SDFNode':
        """
        Moves the object in space by a given offset vector.

        Args:
            offset (tuple or np.ndarray): The (x, y, z) vector to move the object by.
        """
        from .api.transforms import Translate
        return Translate(self, offset)

    def scale(self, factor) -> 'SDFNode':
        """
        Scales the object.

        The scaling can be uniform (a single float) or non-uniform
        (a tuple of factors for x, y, and z).

        Args:
            factor (float or tuple): The scaling factor. If a float, scaling is
                                     uniform. If a tuple, it is (sx, sy, sz).
        """
        from .api.transforms import Scale
        return Scale(self, factor)

    def rotate(self, axis, angle: float) -> 'SDFNode':
        """
        Rotates the object around an axis.

        Args:
            axis (tuple or np.ndarray): The axis of rotation (x, y, z).
                                        Can be a cardinal axis like `sdforge.X` 
                                        or an arbitrary vector like `(1, 1, 0)`.
            angle (float): The angle of rotation in radians.
        """
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
        """
        Orients the object's longest dimension along a primary axis.

        Args:
            axis (str or np.ndarray): The target axis, e.g., 'x', 'y', 'z',
                                      or the corresponding vectors `X`, `Y`, `Z`.
        """
        from .api.transforms import Orient
        axis_map = {'x': X, 'y': Y, 'z': Z}
        if isinstance(axis, str) and axis.lower() in axis_map:
            axis = axis_map[axis.lower()]
        return Orient(self, axis)

    def twist(self, strength: float) -> 'SDFNode':
        """
        Twists the object around its Y-axis.

        Args:
            strength (float): The amount of twist in radians per unit of height.
        """
        from .api.transforms import Twist
        return Twist(self, strength)

    def bend(self, axis, curvature: float) -> 'SDFNode':
        """
        Bends the object into an arc around a cardinal axis.

        Args:
            axis (np.ndarray): The axis to bend around (`X`, `Y`, or `Z`).
            curvature (float): The curvature amount.
        """
        from .api.transforms import Bend
        return Bend(self, axis, curvature)

    def repeat(self, spacing) -> 'SDFNode':
        """
        Repeats the object infinitely across a grid.

        Args:
            spacing (tuple or np.ndarray): The (x, y, z) spacing of the grid.
                                           A spacing of 0 along an axis means
                                           no repetition on that axis.
        """
        from .api.transforms import Repeat
        return Repeat(self, spacing)

    def limited_repeat(self, spacing, limits) -> 'SDFNode':
        """
        Repeats the object a finite number of times.

        Args:
            spacing (tuple): The (x, y, z) spacing of the grid.
            limits (tuple): The (nx, ny, nz) integer number of repetitions in
                            each positive and negative direction. A limit of
                            (2, 0, 0) creates 5 instances: -2, -1, 0, 1, 2.
        """
        from .api.transforms import LimitedRepeat
        return LimitedRepeat(self, spacing, limits)

    def polar_repeat(self, repetitions: int) -> 'SDFNode':
        """
        Repeats the object in a circle around the Y-axis.

        Args:
            repetitions (int): The number of times to repeat the object.
        """
        from .api.transforms import PolarRepeat
        return PolarRepeat(self, repetitions)

    def mirror(self, axes) -> 'SDFNode':
        """
        Mirrors the object across one or more axes to create symmetry.

        Args:
            axes (np.ndarray): The axes to mirror across. Can be combined with `|`,
                               e.g., `X | Y` to mirror across both X and Y.
        """
        from .api.transforms import Mirror
        return Mirror(self, axes)

    # --- Shaping Operations ---
    def round(self, radius: float) -> 'SDFNode':
        """
        Rounds all edges of the object by a given radius.

        Args:
            radius (float): The radius of the rounding effect.
        """
        from .api.shaping import Round
        return Round(self, radius)

    def shell(self, thickness: float) -> 'SDFNode':
        """
        Creates a hollow shell or outline of the object.

        Args:
            thickness (float): The thickness of the shell wall.
        """
        from .api.shaping import Shell
        return Shell(self, thickness)

    def extrude(self, height: float) -> 'SDFNode':
        """
        Extrudes a 2D SDF shape along the Z-axis to create a 3D object.

        Args:
            height (float): The total height of the extrusion.
        """
        from .api.shaping import Extrude
        return Extrude(self, height)

    def revolve(self) -> 'SDFNode':
        """
        Revolves a 2D SDF shape around the Y-axis to create a 3D object.
        """
        from .api.shaping import Revolve
        r = Revolve()
        r.child = self
        return r

    # --- Surface Displacement ---
    def displace(self, displacement_glsl: str) -> 'SDFNode':
        """
        Displaces the surface of the object using a custom GLSL expression.

        Args:
            displacement_glsl (str): A GLSL expression that evaluates to a float.
        """
        from .api.noise import Displace
        return Displace(self, displacement_glsl)

    def displace_by_noise(self, scale: float = 10.0, strength: float = 0.1) -> 'SDFNode':
        """
        Displaces the surface using a built-in procedural noise function.

        Args:
            scale (float, optional): The frequency of the noise.
            strength (float, optional): The amplitude of the displacement.
        """
        from .api.noise import DisplaceByNoise
        return DisplaceByNoise(self, scale, strength)