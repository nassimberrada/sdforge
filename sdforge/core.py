import numpy as np
import uuid
from pathlib import Path
from functools import lru_cache
import sys
import inspect

# --- Constants ---
X = np.array([1, 0, 0])
Y = np.array([0, 1, 0])
Z = np.array([0, 0, 1])

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
    from .ui import Param
    if isinstance(val, Param):
        return val.to_glsl()
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
        self._init_args = self._capture_init_args()

    def _capture_init_args(self):
        """Helper to capture __init__ arguments for parameter traversal."""
        # This is a bit of magic to get the arguments passed to a subclass's __init__
        try:
            frame = inspect.currentframe()
            outer_frames = inspect.getouterframes(frame)
            # Find the frame corresponding to the subclass __init__ call
            for f in outer_frames:
                if f.function == '__init__' and 'self' in f.frame.f_locals:
                    instance = f.frame.f_locals['self']
                    if isinstance(instance, self.__class__) and instance is self:
                        args, _, _, values = inspect.getargvalues(f.frame)
                        return {arg: values[arg] for arg in args if arg != 'self'}
        except Exception:
            pass # Fails gracefully if something goes wrong
        return {}


    def render(self, camera=None, light=None, watch=True, record=None, save_frame=None, bg_color=(0.1, 0.12, 0.15), debug=None, **kwargs):
        """
        Renders the SDF object in a live-updating viewer or saves a single frame.

        Args:
            camera (Camera, optional): A camera object for scene viewing. Defaults to a mouse-orbit camera.
            light (Light, optional): A light object for the scene. Defaults to a headlight.
            watch (bool, optional): If True, enables hot-reloading. Defaults to True.
            record (str, optional): Path to save a video recording (e.g., "output.mp4"). Defaults to None.
            save_frame (str, optional): Path to save a single frame image (e.g., "frame.png"). Defaults to None.
            bg_color (tuple, optional): The background color as an (r, g, b) tuple. Defaults to (0.1, 0.12, 0.15).
            debug (str, optional): Enables a debug visualization mode.
                                   Options: 'normals', 'steps'. Defaults to None.
        """
        from .render import render as render_func
        render_func(self, camera, light, watch, record, save_frame, bg_color, debug, **kwargs)

    def save(self, path, bounds=None, samples=2**22, verbose=True):
        """
        Generates a mesh and saves it to a file (e.g., '.stl', '.obj').
        
        Args:
            path (str): The file path to save to.
            bounds (tuple, optional): The bounding box to mesh within. If None,
                                      it will be automatically estimated.
            samples (int, optional): The number of points to sample in the volume.
            verbose (bool, optional): Whether to print progress information.
        """
        if bounds is None:
            if verbose:
                print("INFO: No bounds provided to .save(), estimating automatically.", file=sys.stderr)
            bounds = self.estimate_bounds(verbose=verbose)
        from .mesh import save as save_func
        save_func(self, path, bounds, samples, verbose)

    def save_frame(self, path, camera=None, light=None, **kwargs):
        """Renders a single frame and saves it to an image file (e.g., '.png')."""
        self.render(save_frame=path, camera=camera, light=light, watch=False, **kwargs)

    def estimate_bounds(self, resolution=64, search_bounds=((-2, -2, -2), (2, 2, 2)), padding=0.1, verbose=True):
        """
        Estimates the bounding box of the SDF object by sampling a grid.

        This method is useful for automatically framing a camera or setting the
        meshing bounds for the `.save()` method.

        Args:
            resolution (int, optional): The number of points to sample along each axis.
                                        Higher values are more accurate but slower. Defaults to 64.
            search_bounds (tuple, optional): The initial cube volume to search for the object.
                                             Defaults to ((-2, -2, -2), (2, 2, 2)).
            padding (float, optional): A padding factor to add to the estimated bounds,
                                       proportional to the object's size. Defaults to 0.1.
            verbose (bool, optional): If True, prints progress information to the console. Defaults to True.

        Returns:
            tuple: A tuple of ((min_x, min_y, min_z), (max_x, max_y, max_z)) representing the bounds,
                   or the original search_bounds if the object is not found.
        """
        from .mesh import _cartesian_product
        if verbose:
            print(f"INFO: Estimating bounds with {resolution**3} samples...", file=sys.stderr)

        try:
            sdf_callable = self.to_callable()
        except (TypeError, NotImplementedError, ImportError) as e:
            if verbose:
                print(f"ERROR: Could not estimate bounds. The object may contain animated or un-callable parts. {e}", file=sys.stderr)
            raise

        # Create the grid points
        axes = [np.linspace(search_bounds[0][i], search_bounds[1][i], resolution) for i in range(3)]
        points_grid = _cartesian_product(*axes).astype('f4')

        # Evaluate SDF and find points inside the surface
        distances = sdf_callable(points_grid)
        inside_mask = distances <= 1e-4 # Use a small epsilon to catch the surface
        inside_points = points_grid[inside_mask]

        if inside_points.shape[0] < 2: # Need at least 2 points to define a non-zero volume
            if verbose:
                print(f"WARNING: No object surface found within the search bounds {search_bounds}. Returning search_bounds.", file=sys.stderr)
            return search_bounds

        # Find min and max coordinates
        min_coords = np.min(inside_points, axis=0)
        max_coords = np.max(inside_points, axis=0)
        
        # Add padding
        size = max_coords - min_coords
        # Handle case where a dimension is flat
        size[size < 1e-6] = padding 
        min_coords -= size * padding
        max_coords += size * padding

        bounds = (tuple(min_coords), tuple(max_coords))
        if verbose:
            print(f"SUCCESS: Estimated bounds: {bounds}", file=sys.stderr)
            
        return bounds
        
    def to_glsl(self) -> str: raise NotImplementedError
    def to_callable(self): raise NotImplementedError
    def get_glsl_definitions(self) -> list: return []
    def _collect_materials(self, materials): pass
    def _collect_uniforms(self, uniforms): pass
    def _collect_params(self, params):
        from .ui import Param
        if hasattr(self, '_init_args'):
            for arg_val in self._init_args.values():
                if isinstance(arg_val, Param):
                    params[arg_val.uniform_name] = arg_val
                elif isinstance(arg_val, (list, tuple)):
                    for item in arg_val:
                        if isinstance(item, Param):
                           params[item.uniform_name] = item
        if hasattr(self, 'child'):
            self.child._collect_params(params)
        if hasattr(self, 'children'):
            for child in self.children:
                child._collect_params(params)


    def union(self, other, k=0.0):
        """Creates a union of this object and another, with optional smoothness."""
        from .operations import Union
        return Union(self, other, k=k)

    def intersection(self, other, k=0.0):
        """Creates an intersection of this object and another, with optional smoothness."""
        from .operations import Intersection
        return Intersection(self, other, k=k)

    def difference(self, other, k=0.0):
        """Subtracts another object from this one, with optional smoothness."""
        from .operations import Difference
        return Difference(self, other, k=k)
    
    def bounded_by(self, bounding_shape):
        """
        Optimizes rendering by intersecting this object with a simpler bounding shape.
        The raymarcher can take larger steps when outside the bounds.
        This is an alias for intersection.
        """
        return self.intersection(bounding_shape)

    def __or__(self, other):
        """Creates a union of this object and another."""
        return self.union(other)

    def __and__(self, other):
        """Creates an intersection of this object and another."""
        return self.intersection(other)

    def __sub__(self, other):
        """Subtracts another object from this one."""
        return self.difference(other)

    def xor(self, other):
        """Creates an exclusive-or (XOR) of this object and another."""
        from .operations import Xor
        return Xor(self, other)

    def __add__(self, offset):
        """Moves the object in space using the '+' operator."""
        return self.translate(offset)

    def __mul__(self, factor):
        """Scales the object using the '*' operator."""
        return self.scale(factor)

    def __rmul__(self, factor):
        """Scales the object using the '*' operator (e.g., `2 * shape`)."""
        return self.scale(factor)

    def translate(self, offset):
        """Moves the object in space."""
        from .transforms import Translate
        return Translate(self, np.array(offset))

    def scale(self, factor):
        """Scales the object. Can be a uniform factor or per-axis."""
        from .transforms import Scale
        return Scale(self, factor)

    def orient(self, axis):
        """Orients the object along a primary axis (e.g., 'x', 'y', 'z' or vector)."""
        from .transforms import Orient
        axis_map = {'x': X, 'y': Y, 'z': Z}
        if isinstance(axis, str) and axis.lower() in axis_map:
            axis = axis_map[axis.lower()]
        return Orient(self, np.array(axis))

    def rotate(self, axis, angle):
        """Rotates the object around an axis by a given angle in radians."""
        from .transforms import Rotate
        axis_map = {'x': X, 'y': Y, 'z': Z}
        if isinstance(axis, str) and axis.lower() in axis_map:
            axis = axis_map[axis.lower()]
        return Rotate(self, np.array(axis), angle)

    def twist(self, k):
        """Twists the object around the Y-axis."""
        from .transforms import Twist
        return Twist(self, k)

    def shear_xy(self, shear):
        """Shears the object in the XY plane based on the Z coordinate."""
        from .transforms import ShearXY
        return ShearXY(self, np.array(shear))

    def shear_xz(self, shear):
        """Shears the object in the XZ plane based on the Y coordinate."""
        from .transforms import ShearXZ
        return ShearXZ(self, np.array(shear))

    def shear_yz(self, shear):
        """Shears the object in the YZ plane based on the X coordinate."""
        from .transforms import ShearYZ
        return ShearYZ(self, np.array(shear))

    def bend_x(self, k):
        """Bends the object around the X-axis."""
        from .transforms import BendX
        return BendX(self, k)

    def bend_y(self, k):
        """Bends the object around the Y-axis."""
        from .transforms import BendY
        return BendY(self, k)

    def bend_z(self, k):
        """Bends the object around the Z-axis."""
        from .transforms import BendZ
        return BendZ(self, k)

    def repeat(self, spacing):
        """Repeats the object infinitely with a given spacing vector."""
        from .transforms import Repeat
        return Repeat(self, np.array(spacing))

    def limited_repeat(self, spacing, limits):
        """Repeats the object a limited number of times along each axis."""
        from .transforms import LimitedRepeat
        return LimitedRepeat(self, np.array(spacing), np.array(limits))

    def polar_repeat(self, repetitions):
        """Repeats the object in a circle around the Y-axis."""
        from .transforms import PolarRepeat
        return PolarRepeat(self, repetitions)

    def mirror(self, axes):
        """Mirrors the object across one or more axes (e.g., X, Y, X|Z)."""
        from .transforms import Mirror
        return Mirror(self, np.array(axes))

    def color(self, r, g, b):
        """Applies a color material to the object."""
        from .shaping import Material
        return Material(self, (r, g, b))

    def round(self, radius):
        """Rounds all edges of the object by a given radius."""
        from .shaping import Round
        return Round(self, radius)

    def shell(self, thickness):
        """Creates a shell or outline of the object with a given thickness."""
        from .shaping import Bevel
        return Bevel(self, thickness)

    def bevel(self, thickness):
        """Alias for .shell(). Creates an outline of the object."""
        return self.shell(thickness)

    def elongate(self, h):
        """Elongates the object along its axes."""
        from .shaping import Elongate
        return Elongate(self, np.array(h))

    def displace(self, displacement_glsl):
        """Displaces the surface of the object using a GLSL expression."""
        from .shaping import Displace
        return Displace(self, displacement_glsl)

    def displace_by_noise(self, scale=10.0, strength=0.1):
        """
        Displaces the surface of the object using a procedural noise function.

        Args:
            scale (float or str, optional): The frequency/scale of the noise pattern.
                                            Higher values result in smaller, more detailed patterns.
                                            Defaults to 10.0.
            strength (float or str, optional): The amplitude of the displacement.
                                               Defaults to 0.1.
        """
        from .shaping import DisplaceByNoise
        return DisplaceByNoise(self, scale, strength)

    def extrude(self, height):
        """Extrudes a 2D SDF shape along the Z-axis."""
        from .shaping import Extrude
        return Extrude(self, height)

    def revolve(self):
        """Revolves a 2D SDF shape around the Y-axis."""
        from .shaping import Revolve
        return Revolve(self)