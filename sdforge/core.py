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
                elif isinstance(attr_val, Expr): # <-- NEW: Check for expressions
                    for p in attr_val.params:
                        params[p.uniform_name] = p
                elif isinstance(attr_val, (list, tuple, np.ndarray)):
                    for item in attr_val:
                        if isinstance(item, Param):
                            params[item.uniform_name] = item
                        elif isinstance(item, Expr): # <-- NEW: Check for expressions in lists
                             for p in item.params:
                                 params[p.uniform_name] = p
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
        
        Example:
            >>> from sdforge import sphere, Camera
            >>> scene = sphere(1.0)
            >>> cam = Camera(position=(3,2,3))
            >>> # scene.save_frame("output.png", camera=cam, width=800, height=600)
        """
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
    def union(self, *others, k: float = 0.0, fillet: float = 0.0, chamfer: float = 0.0) -> 'SDFNode':
        """
        Creates the union (join) of this object and one or more other objects.

        This operation combines multiple shapes into a single one. An optional
        blending parameter can be used to create smooth fillets or linear
        chamfers at the seams. Only one blending type can be used at a time.

        Args:
            *others (SDFNode): A variable number of other SDF objects to union with.
            k (float, optional): A generic smoothing factor for fillets. Higher
                                 values create larger, smoother blends. `fillet` is
                                 a more intuitive alias for this. Defaults to 0.0.
            fillet (float, optional): The radius of a smooth, rounded blend at the
                                      seam between objects. Defaults to 0.0 (no blend).
            chamfer (float, optional): The distance of a linear, 45-degree blend
                                       (a bevel) at the seam. Defaults to 0.0.

        Returns:
            SDFNode: A new object representing the union.
        
        Example:
            >>> from sdforge import sphere, box, X
            >>> s1 = sphere(0.7).translate(-X * 0.5)
            >>> s2 = sphere(0.7).translate(X * 0.5)
            >>> # Smoothly blend two spheres together
            >>> scene = s1.union(s2, fillet=0.3)
        """
        from .api.operations import Union
        num_ops = (k > 1e-6) + (fillet > 1e-6) + (chamfer > 1e-6)
        if num_ops > 1:
            raise ValueError("Cannot specify more than one of 'k', 'fillet', or 'chamfer' at the same time.")
        if fillet > 1e-6: k = fillet
        return Union(children=[self] + list(others), k=k, chamfer=chamfer)

    def intersection(self, *others, k: float = 0.0, fillet: float = 0.0, chamfer: float = 0.0) -> 'SDFNode':
        """
        Creates the intersection (common area) of this object and others.

        This operation results in a shape that is only the volume shared by all
        the input objects. Optional blending can be applied.

        Args:
            *others (SDFNode): A variable number of other SDF objects to intersect with.
            k (float, optional): A generic smoothing factor. Defaults to 0.0.
            fillet (float, optional): The radius of a smooth, rounded blend at the
                                      intersection edge. Defaults to 0.0.
            chamfer (float, optional): The distance of a linear blend at the
                                       intersection edge. Defaults to 0.0.

        Returns:
            SDFNode: A new object representing the intersection.
            
        Example:
            >>> from sdforge import sphere, X
            >>> # Create a lens shape by intersecting two spheres
            >>> s1 = sphere(1.0).translate(-X * 0.5)
            >>> s2 = sphere(1.0).translate(X * 0.5)
            >>> lens = s1.intersection(s2)
        """
        from .api.operations import Intersection
        num_ops = (k > 1e-6) + (fillet > 1e-6) + (chamfer > 1e-6)
        if num_ops > 1:
            raise ValueError("Cannot specify more than one of 'k', 'fillet', or 'chamfer' at the same time.")
        if fillet > 1e-6: k = fillet
        return Intersection(children=[self] + list(others), k=k, chamfer=chamfer)

    def difference(self, other, k: float = 0.0, fillet: float = 0.0, chamfer: float = 0.0) -> 'SDFNode':
        """
        Subtracts another object from this one.

        This operation carves the volume of the `other` object out of this one.
        Optional blending can be used to create rounded (filleted) or linear
        (chamfered) cuts.

        Args:
            other (SDFNode): The object to subtract.
            k (float, optional): A generic smoothing factor. Defaults to 0.0.
            fillet (float, optional): The radius of a smooth, rounded blend at the
                                      cut edge. Defaults to 0.0.
            chamfer (float, optional): The distance of a linear blend at the
                                       cut edge. Defaults to 0.0.

        Returns:
            SDFNode: A new object representing the difference.
            
        Example:
            >>> from sdforge import box, cylinder
            >>> # Create a plate with a filleted hole
            >>> plate = box((2, 0.2, 2))
            >>> hole = cylinder(radius=0.5, height=0.3)
            >>> scene = plate.difference(hole, fillet=0.05)
        """
        from .api.operations import Difference
        num_ops = (k > 1e-6) + (fillet > 1e-6) + (chamfer > 1e-6)
        if num_ops > 1:
            raise ValueError("Cannot specify more than one of 'k', 'fillet', or 'chamfer' at the same time.")
        if fillet > 1e-6: k = fillet
        return Difference(self, other, k=k, chamfer=chamfer)

    def __or__(self, other):
        """Operator overload for a simple union: `shape1 | shape2`."""
        return self.union(other)

    def __and__(self, other):
        """Operator overload for a simple intersection: `shape1 & shape2`."""
        return self.intersection(other)

    def __sub__(self, other):
        """Operator overload for a simple difference: `shape1 - shape2`."""
        return self.difference(other)
        
    # --- CAD-style Boolean Operations ---
    def fillet_union(self, *others, radius: float) -> 'SDFNode':
        """
        Creates a union with a rounded (filleted) seam.

        This is a convenience alias for `.union(*others, fillet=radius)`.

        Args:
            *others (SDFNode): The other objects to union with.
            radius (float): The radius of the fillet at the seam.

        Returns:
            SDFNode: The filleted union object.
        """
        return self.union(*others, fillet=radius)

    def chamfer_union(self, *others, distance: float) -> 'SDFNode':
        """
        Creates a union with a linear (chamfered) seam.

        This is a convenience alias for `.union(*others, chamfer=distance)`.

        Args:
            *others (SDFNode): The other objects to union with.
            distance (float): The distance of the chamfer.

        Returns:
            SDFNode: The chamfered union object.
        """
        return self.union(*others, chamfer=distance)
        
    def fillet_intersection(self, *others, radius: float) -> 'SDFNode':
        """
        Creates an intersection with a rounded (filleted) edge.

        This is a convenience alias for `.intersection(*others, fillet=radius)`.

        Args:
            *others (SDFNode): The other objects to intersect with.
            radius (float): The radius of the fillet.

        Returns:
            SDFNode: The filleted intersection object.
        """
        return self.intersection(*others, fillet=radius)

    def chamfer_intersection(self, *others, distance: float) -> 'SDFNode':
        """
        Creates an intersection with a linear (chamfered) edge.

        This is a convenience alias for `.intersection(*others, chamfer=distance)`.

        Args:
            *others (SDFNode): The other objects to intersect with.
            distance (float): The distance of the chamfer.

        Returns:
            SDFNode: The chamfered intersection object.
        """
        return self.intersection(*others, chamfer=distance)

    def fillet_difference(self, other, radius: float) -> 'SDFNode':
        """
        Subtracts a shape with a rounded (filleted) edge.

        This is a convenience alias for `.difference(other, fillet=radius)`.

        Args:
            other (SDFNode): The object to subtract.
            radius (float): The radius of the fillet at the cut.

        Returns:
            SDFNode: The filleted difference object.
        """
        return self.difference(other, fillet=radius)

    def chamfer_difference(self, other, distance: float) -> 'SDFNode':
        """
        Subtracts a shape with a linear (chamfered) edge.

        This is a convenience alias for `.difference(other, chamfer=distance)`.

        Args:
            other (SDFNode): The object to subtract.
            distance (float): The distance of the chamfer at the cut.

        Returns:
            SDFNode: The chamfered difference object.
        """
        return self.difference(other, chamfer=distance)

    # --- Material ---
    def color(self, r: float, g: float, b: float) -> 'SDFNode':
        """
        Applies a color material to the object.

        Colors are specified in RGB format with values from 0.0 to 1.0.

        Args:
            r (float): The red component (0.0 to 1.0).
            g (float): The green component (0.0 to 1.0).
            b (float): The blue component (0.0 to 1.0).

        Returns:
            SDFNode: A new object with the color applied.
            
        Example:
            >>> from sdforge import box, sphere
            >>> red_box = box(1.5).color(1.0, 0.2, 0.2)
            >>> blue_sphere = sphere(1.2).color(0.3, 0.5, 1.0)
            >>> scene = red_box - blue_sphere
        """
        from .api.material import Material
        return Material(self, (r, g, b))

    # --- Transformations ---
    def translate(self, offset) -> 'SDFNode':
        """
        Moves the object in space by a given offset vector.

        Args:
            offset (tuple or np.ndarray): The (x, y, z) vector to move the object by.

        Returns:
            SDFNode: The translated object.
            
        Example:
            >>> from sdforge import sphere, X
            >>> # Move a sphere 2 units along the positive X axis
            >>> s = sphere(1.0).translate(X * 2.0)
            >>> # The '+' operator is a convenient shortcut:
            >>> s_alt = sphere(1.0) + (2, 0, 0)
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

        Returns:
            SDFNode: The scaled object.
            
        Example:
            >>> from sdforge import box
            >>> # Make a box twice as large uniformly
            >>> b1 = box(1.0).scale(2.0)
            >>> # The '*' operator is a shortcut for uniform scaling:
            >>> b1_alt = box(1.0) * 2.0
            >>> # Stretch a box to be tall and thin
            >>> b2 = box(1.0).scale((0.5, 2.0, 0.5))
        """
        from .api.transforms import Scale
        return Scale(self, factor)

    def rotate(self, axis, angle: float) -> 'SDFNode':
        """
        Rotates the object around a cardinal axis.

        Args:
            axis (np.ndarray): The axis of rotation. Must be one of `sdforge.X`,
                               `sdforge.Y`, or `sdforge.Z`.
            angle (float): The angle of rotation in radians.

        Returns:
            SDFNode: The rotated object.
            
        Example:
            >>> import numpy as np
            >>> from sdforge import box, Z
            >>> # A non-symmetrical box to make rotation obvious
            >>> b = box(size=(1.5, 0.5, 0.2))
            >>> # Rotate it 45 degrees around the Z axis
            >>> scene = b.rotate(Z, np.pi / 4)
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

        This is a convenience transform that swaps coordinate axes to quickly
        re-orient a shape. For example, a cylinder is Y-aligned by default;
        `.orient('x')` will make it X-aligned.

        Args:
            axis (str or np.ndarray): The target axis, e.g., 'x', 'y', 'z',
                                      or the corresponding vectors `X`, `Y`, `Z`.

        Returns:
            SDFNode: The re-oriented object.
            
        Example:
            >>> from sdforge import cylinder
            >>> # A standard cylinder is oriented along the Y axis
            >>> y_cyl = cylinder(radius=0.2, height=1.0)
            >>> # Re-orient it to lie along the X axis
            >>> x_cyl = y_cyl.orient('x')
        """
        from .api.transforms import Orient
        axis_map = {'x': X, 'y': Y, 'z': Z}
        if isinstance(axis, str) and axis.lower() in axis_map:
            axis = axis_map[axis.lower()]
        return Orient(self, axis)

    def twist(self, k: float) -> 'SDFNode':
        """
        Twists the object around its Y-axis.

        Args:
            k (float): The amount of twist in radians per unit of height.

        Returns:
            SDFNode: The twisted object.
            
        Example:
            >>> from sdforge import box
            >>> # Create a tall, twisted column
            >>> column = box(size=(0.5, 3.0, 0.5)).twist(k=3.0)
        """
        from .api.transforms import Twist
        return Twist(self, k)

    def bend(self, axis, k: float) -> 'SDFNode':
        """
        Bends the object into an arc around a cardinal axis.

        Args:
            axis (np.ndarray): The axis to bend around (`X`, `Y`, or `Z`).
            k (float): The curvature amount. Higher values result in a
                       tighter bend.

        Returns:
            SDFNode: The bent object.
            
        Example:
            >>> from sdforge import box, Y
            >>> # Bend a long plank into an arch
            >>> plank = box(size=(4.0, 0.2, 0.5))
            >>> arch = plank.bend(Y, k=0.5)
        """
        from .api.transforms import Bend
        return Bend(self, axis, k)
        
    def repeat(self, spacing) -> 'SDFNode':
        """
        Repeats the object infinitely across a grid.

        Args:
            spacing (tuple or np.ndarray): The (x, y, z) spacing of the grid.
                                           A spacing of 0 along an axis means
                                           no repetition on that axis.

        Returns:
            SDFNode: An infinitely repeating object.
            
        Example:
            >>> from sdforge import sphere
            >>> # Create an infinite field of spheres
            >>> s = sphere(0.4)
            >>> scene = s.repeat((2.0, 2.0, 2.0))
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

        Returns:
            SDFNode: A finitely repeating object.
            
        Example:
            >>> from sdforge import sphere
            >>> # Create a line of 5 spheres along the X axis
            >>> s = sphere(0.4)
            >>> scene = s.limited_repeat(spacing=(1.0, 0, 0), limits=(2, 0, 0))
        """
        from .api.transforms import LimitedRepeat
        return LimitedRepeat(self, spacing, limits)

    def polar_repeat(self, repetitions: int) -> 'SDFNode':
        """
        Repeats the object in a circle around the Y-axis.

        The object should be created with an initial offset from the origin,
        as this offset determines the radius of the circular pattern.

        Args:
            repetitions (int): The number of times to repeat the object.

        Returns:
            SDFNode: The patterned object.
            
        Example:
            >>> from sdforge import box, X
            >>> # Start with a box offset from the origin
            >>> b = box((0.5, 0.2, 0.1)).translate(X * 1.5)
            >>> # Repeat it 8 times in a circle
            >>> scene = b.polar_repeat(8)
        """
        from .api.transforms import PolarRepeat
        return PolarRepeat(self, repetitions)

    def mirror(self, axes) -> 'SDFNode':
        """
        Mirrors the object across one or more axes to create symmetry.

        Args:
            axes (np.ndarray): The axes to mirror across. Can be combined with `|`,
                               e.g., `X | Y` to mirror across both X and Y.

        Returns:
            SDFNode: The mirrored, symmetrical object.
            
        Example:
            >>> from sdforge import sphere, X, Y
            >>> # Model one quadrant of a shape
            >>> quadrant = sphere(0.5).translate((0.8, 0.8, 0))
            >>> # Mirror it to create the full symmetrical shape
            >>> scene = quadrant.mirror(X | Y)
        """
        from .api.transforms import Mirror
        return Mirror(self, axes)

    # --- Shaping Operations ---
    def round(self, radius: float) -> 'SDFNode':
        """
        Rounds all edges of the object by a given radius.

        Args:
            radius (float): The radius of the rounding effect.

        Returns:
            SDFNode: The rounded object.
            
        Example:
            >>> from sdforge import box
            >>> sharp_box = box(1.5)
            >>> rounded_box = sharp_box.round(0.2)
        """
        from .api.shaping import Round
        return Round(self, radius)

    def shell(self, thickness: float) -> 'SDFNode':
        """
        Creates a hollow shell or outline of the object.

        This operation takes the absolute distance to the surface, resulting
        in a new surface both inside and outside the original.

        Args:
            thickness (float): The thickness of the shell wall.

        Returns:
            SDFNode: The shell object.
            
        Example:
            >>> from sdforge import sphere
            >>> # Create a hollow sphere with a 0.05 unit thick wall
            >>> hollow_sphere = sphere(1.0).shell(0.05)
        """
        from .api.shaping import Bevel
        return Bevel(self, thickness)

    def bevel(self, thickness: float) -> 'SDFNode':
        """
        Alias for .shell(). Creates an outline of the object.

        Args:
            thickness (float): The thickness of the bevel/shell.

        Returns:
            SDFNode: The shell object.
        """
        return self.shell(thickness)

    def extrude(self, height: float) -> 'SDFNode':
        """
        Extrudes a 2D SDF shape along the Z-axis to create a 3D object.

        This method should be called on a 2D primitive like `circle` or `rectangle`.

        Args:
            height (float): The total height of the extrusion.

        Returns:
            SDFNode: The 3D extruded object.
            
        Example:
            >>> from sdforge import circle
            >>> # Extrude a circle to create a cylinder
            >>> cylinder = circle(r=0.8).extrude(1.5)
        """
        from .api.shaping import Extrude
        return Extrude(self, height)

    def revolve(self) -> 'SDFNode':
        """
        Revolves a 2D SDF shape around the Y-axis to create a 3D object.

        The 2D shape, defined in the XY plane, acts as the cross-section profile.
        It should be offset from the Y-axis to create a valid 3D shape.

        Returns:
            SDFNode: The 3D revolved object.
            
        Example:
            >>> from sdforge import rectangle, X
            >>> # A rectangular profile offset from the Y-axis
            >>> profile = rectangle((0.2, 1.0)).translate(X * 0.8)
            >>> # Revolve it to create a thick ring
            >>> ring = profile.revolve()
        """
        from .api.shaping import Revolve
        # Revolve is special: it becomes the parent of the current node
        r = Revolve()
        r.child = self
        return r

    # --- Surface Displacement ---
    def displace(self, displacement_glsl: str) -> 'SDFNode':
        """
        Displaces the surface of the object using a custom GLSL expression.

        This allows for complex procedural texturing and shaping. The operation
        cannot be exported to a mesh file.

        Args:
            displacement_glsl (str): A GLSL expression that evaluates to a float.
                                     The `vec3 p` variable is available, representing
                                     the point in space being sampled. A positive
                                     result moves the surface outwards.

        Returns:
            SDFNode: The displaced object.
            
        Example:
            >>> from sdforge import box
            >>> # Create a sine wave pattern on a box
            >>> glsl = "sin(p.x * 20.0) * 0.05"
            >>> scene = box(1.5).displace(glsl)
        """
        from .api.noise import Displace
        return Displace(self, displacement_glsl)

    def displace_by_noise(self, scale: float = 10.0, strength: float = 0.1) -> 'SDFNode':
        """
        Displaces the surface using a built-in procedural noise function.

        This is a convenient way to add texture or organic variation to a shape.
        This operation cannot be exported to a mesh file.

        Args:
            scale (float, optional): The frequency of the noise. Higher values
                                     create finer, more detailed patterns.
                                     Defaults to 10.0.
            strength (float, optional): The amplitude of the displacement, i.e.,
                                        how much the surface moves. Defaults to 0.1.

        Returns:
            SDFNode: The displaced object.
            
        Example:
            >>> from sdforge import sphere
            >>> # Create a bumpy, organic-looking sphere
            >>> noisy_sphere = sphere(1.2).displace_by_noise(scale=8.0, strength=0.1)
        """
        from .api.noise import DisplaceByNoise
        return DisplaceByNoise(self, scale, strength)