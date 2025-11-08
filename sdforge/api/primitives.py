import numpy as np
from ..core import SDFNode, GLSLContext

class Sphere(SDFNode):
    """Represents a sphere primitive."""
    
    # This class attribute is the explicit link to the required GLSL code.
    glsl_dependencies = {"sdSphere"}

    def __init__(self, r: float = 1.0):
        super().__init__()
        self.r = r

    def to_glsl(self, ctx: GLSLContext) -> str:
        # 1. Add our dependencies to the context for the compiler to find.
        ctx.dependencies.update(self.glsl_dependencies)
        
        # 2. Generate the specific GLSL expression for this node.
        dist_expr = f"sdSphere({ctx.p}, {float(self.r)})"
        
        # 3. Return the result in a new vec4 variable.
        #    The vec4 format (dist, mat_id, 0, 0) is for future compatibility.
        return ctx.new_variable('vec4', f"vec4({dist_expr}, -1.0, 0.0, 0.0)")

    def to_callable(self):
        r_val = self.r
        def _callable(points: np.ndarray) -> np.ndarray:
            return np.linalg.norm(points, axis=-1) - r_val
        return _callable

def sphere(r: float = 1.0) -> SDFNode:
    """Creates a sphere SDF node."""
    return Sphere(r)