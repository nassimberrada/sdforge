import numpy as np
import re
from .core import SDFObject, _get_glsl_content, _glsl_format

# --- Material ---

class Material(SDFObject):
    """Applies a color material to a child object."""
    def __init__(self, child, color):
        super().__init__()
        self.child = child
        self.color = color
        self.material_id = -1 # Will be set by the renderer
    
    def to_glsl(self) -> str:
        child_glsl = self.child.to_glsl()
        return (
            f"(vec4(({child_glsl}).x, {float(self.material_id)}, "
            f"({child_glsl}).z, ({child_glsl}).w))"
        )

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

# --- Shaping Operations ---

class Round(SDFObject):
    """Rounds the edges of a child object."""
    def __init__(self, child, radius):
        super().__init__()
        self.child, self.radius = child, radius
    def to_glsl(self) -> str:
        return f"opRound({self.child.to_glsl()}, {_glsl_format(self.radius)})"
    def to_callable(self):
        if isinstance(self.radius, str): raise TypeError("Animated parameters not supported for mesh export.")
        child_call = self.child.to_callable()
        return lambda p: child_call(p) - self.radius
    def get_glsl_definitions(self) -> list: return [_get_glsl_content("operations.glsl")] + self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)

class Bevel(SDFObject):
    """Creates a shell or outline of a child object."""
    def __init__(self, child, thickness):
        super().__init__()
        self.child, self.thickness = child, thickness
    def to_glsl(self) -> str:
        return f"opBevel({self.child.to_glsl()}, {_glsl_format(self.thickness)})"
    def to_callable(self):
        if isinstance(self.thickness, str): raise TypeError("Animated parameters not supported for mesh export.")
        child_call = self.child.to_callable()
        return lambda p: np.abs(child_call(p)) - self.thickness
    def get_glsl_definitions(self) -> list: return [_get_glsl_content("operations.glsl")] + self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)

class Elongate(SDFObject):
    """Elongates a child object."""
    def __init__(self, child, h):
        super().__init__()
        self.child, self.h = child, h
    def to_glsl(self) -> str:
        h_str = f"vec3({_glsl_format(self.h[0])}, {_glsl_format(self.h[1])}, {_glsl_format(self.h[2])})"
        transformed_p = f"opElongate(p, {h_str})"
        child_glsl = self.child.to_glsl()
        return re.sub(r'\bp\b', transformed_p, child_glsl)
    def to_callable(self):
        child_call = self.child.to_callable()
        return lambda p: child_call(p - np.clip(p, -self.h, self.h))
    def get_glsl_definitions(self): return [_get_glsl_content("transforms.glsl")] + self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)

class Displace(SDFObject):
    """Displaces the surface of a child object."""
    def __init__(self, child, displacement_glsl):
        super().__init__()
        self.child, self.displacement_glsl = child, displacement_glsl
    def to_glsl(self) -> str:
        return f"opDisplace({self.child.to_glsl()}, {self.displacement_glsl})"
    def to_callable(self):
        raise TypeError("Cannot save mesh of an object with GLSL-based displacement.")
    def get_glsl_definitions(self) -> list: return [_get_glsl_content("operations.glsl")] + self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)

class Extrude(SDFObject):
    """Extrudes a 2D SDF shape."""
    def __init__(self, child, height):
        super().__init__()
        self.child, self.height = child, height
    def to_glsl(self) -> str:
        h = _glsl_format(self.height)
        return f"opExtrude({self.child.to_glsl()}, p, {h})"
    def to_callable(self):
        if isinstance(self.height, str): raise TypeError("Animated parameters not supported for mesh export.")
        child_call = self.child.to_callable()
        h = self.height
        def _callable(p):
            d = child_call(p)
            w = np.stack([d, np.abs(p[:, 2]) - h], axis=-1)
            return np.minimum(np.maximum(w[:,0], w[:,1]), 0.0) + np.linalg.norm(np.maximum(w, 0.0), axis=-1)
        return _callable
    def get_glsl_definitions(self) -> list: return [_get_glsl_content("operations.glsl")] + self.child.get_glsl_definitions()
    def _collect_materials(self, materials): self.child._collect_materials(materials)
