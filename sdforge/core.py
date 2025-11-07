import numpy as np
import uuid
from pathlib import Path
from functools import lru_cache

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

    def render(self, camera=None, light=None, watch=True, record=None, save_frame=None, bg_color=(0.1, 0.12, 0.15), **kwargs):
        """Renders the SDF object in a live-updating viewer or saves a single frame."""
        from .render import render as render_func
        render_func(self, camera, light, watch, record, save_frame, bg_color, **kwargs)

    def save(self, path, bounds=((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)), samples=2**22, verbose=True):
        """Generates a mesh and saves it to a file (e.g., '.stl', '.obj')."""
        from .mesh import save as save_func
        save_func(self, path, bounds, samples, verbose)

    def save_frame(self, path, camera=None, light=None, **kwargs):
        """Renders a single frame and saves it to an image file (e.g., '.png')."""
        self.render(save_frame=path, camera=camera, light=light, watch=False, **kwargs)
        
    def to_glsl(self) -> str: raise NotImplementedError
    def to_callable(self): raise NotImplementedError
    def get_glsl_definitions(self) -> list: return []
    def _collect_materials(self, materials): pass

    def __or__(self, other):
        """Creates a union of this object and another."""
        from .operations import Union
        return Union(self, other)

    def __and__(self, other):
        """Creates an intersection of this object and another."""
        from .operations import Intersection
        return Intersection(self, other)

    def __sub__(self, other):
        """Subtracts another object from this one."""
        from .operations import Difference
        return Difference(self, other)

    def xor(self, other):
        """Creates an exclusive-or (XOR) of this object and another."""
        from .operations import Xor
        return Xor(self, other)

    def translate(self, offset):
        """Moves the object in space."""
        from .transforms import Translate
        return Translate(self, np.array(offset))

    def scale(self, factor):
        """Scales the object. Can be a uniform factor or per-axis."""
        from .transforms import Scale
        return Scale(self, factor)

    def orient(self, axis):
        """Orients the object along a primary axis (X, Y, or Z)."""
        from .transforms import Orient
        return Orient(self, np.array(axis))

    def rotate(self, axis, angle):
        """Rotates the object around an axis by a given angle in radians."""
        from .transforms import Rotate
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

    def union(self, other, k=0.0):
        """
        Creates a union of this object and another, with optional smoothing.

        Args:
            other (SDFObject): The other object to union with.
            k (float or str, optional): The smoothness factor. If > 0, a smooth union is performed.
                                        Defaults to 0.0 (a hard union).
        """
        if (isinstance(k, str) and k != "0.0") or (isinstance(k, (int, float)) and k > 0):
            from .operations import SmoothUnion
            return SmoothUnion(self, other, k)
        from .operations import Union
        return Union(self, other)

    def intersection(self, other, k=0.0):
        """
        Creates an intersection of this object and another, with optional smoothing.

        Args:
            other (SDFObject): The other object to intersect with.
            k (float or str, optional): The smoothness factor. If > 0, a smooth intersection is performed.
                                        Defaults to 0.0 (a hard intersection).
        """
        if (isinstance(k, str) and k != "0.0") or (isinstance(k, (int, float)) and k > 0):
            from .operations import SmoothIntersection
            return SmoothIntersection(self, other, k)
        from .operations import Intersection
        return Intersection(self, other)

    def difference(self, other, k=0.0):
        """
        Subtracts another object from this one, with optional smoothing.

        Args:
            other (SDFObject): The other object to subtract.
            k (float or str, optional): The smoothness factor. If > 0, a smooth difference is performed.
                                        Defaults to 0.0 (a hard difference).
        """
        if (isinstance(k, str) and k != "0.0") or (isinstance(k, (int, float)) and k > 0):
            from .operations import SmoothDifference
            return SmoothDifference(self, other, k)
        from .operations import Difference
        return Difference(self, other)

    def color(self, r, g, b):
        """Applies a color material to the object."""
        from .shaping import Material
        return Material(self, (r, g, b))

    def round(self, radius):
        """Rounds all edges of the object by a given radius."""
        from .shaping import Round
        return Round(self, radius)

    def bevel(self, thickness):
        """Creates a shell or outline of the object with a given thickness."""
        from .shaping import Bevel
        return Bevel(self, thickness)

    def elongate(self, h):
        """Elongates the object along its axes."""
        from .shaping import Elongate
        return Elongate(self, np.array(h))

    def displace(self, displacement_glsl):
        """Displaces the surface of the object using a GLSL expression."""
        from .shaping import Displace
        return Displace(self, displacement_glsl)

    def extrude(self, height):
        """Extrudes a 2D SDF shape along the Z-axis."""
        from .shaping import Extrude
        return Extrude(self, height)