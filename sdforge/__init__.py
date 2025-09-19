from .api import (
    # Base class (for type hinting or extension)
    SDFObject,

    # Primitives
    sphere,
    box,
    rounded_box,
    cylinder,
    torus,
    capsule,
    cone,
    plane,
    hex_prism,
    octahedron,
    ellipsoid,
    box_frame,
    capped_torus,
    link,
    capped_cylinder,
    rounded_cylinder,
    capped_cone,
    round_cone,
    pyramid,

    # Custom GLSL
    Forge,

    # Camera
    Camera,

    # Light
    Light,

    # Constants
    X, Y, Z,
)

# By attaching render and save to the base class, any created object
# can call them directly, e.g., sphere(1).render()
from .render import render
from .mesh import save
from .api import SDFObject
SDFObject.render = render
SDFObject.save = save