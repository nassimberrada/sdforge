from ..core import SDFNode, GLSLContext
from ..utils import _glsl_format
from .params import Param

class Displace(SDFNode):
    """
    Internal node to displace the surface of a child object using a raw GLSL expression.
    
    Note: This class is not typically instantiated directly. Use the
    `.displace()` method on an SDFNode object instead.
    """
    glsl_dependencies = {"shaping"}

    def __init__(self, child: SDFNode, displacement_glsl: str):
        """
        Initializes the Displace node.

        Args:
            child (SDFNode): The object whose surface will be displaced.
            displacement_glsl (str): A GLSL expression that evaluates to a float.
                                     The expression can use `vec3 p` to get the
                                     current sample point in space.
        """
        super().__init__()
        self.child = child
        self.displacement_glsl = displacement_glsl

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        child_var = self.child.to_glsl(ctx)
        # The GLSL expression can use 'p', so we use the current context's 'p'.
        result_expr = f"opDisplace({child_var}, {self.displacement_glsl.replace('p', ctx.p)})"
        return ctx.new_variable('vec4', result_expr)

    def to_callable(self):
        raise TypeError("Cannot create a callable for an object with raw GLSL displacement.")

class DisplaceByNoise(Displace):
    """
    Internal node to displace a surface using a procedural noise function.

    Note: This class is not typically instantiated directly. Use the
    `.displace_by_noise()` method on an SDFNode object instead.
    """
    glsl_dependencies = {"shaping", "noise"}

    def __init__(self, child: SDFNode, scale: float = 10.0, strength: float = 0.1):
        """
        Initializes the DisplaceByNoise node.

        Args:
            child (SDFNode): The object whose surface will be displaced.
            scale (float): The frequency/scale of the noise. Higher values
                           result in finer, more detailed noise.
            strength (float): The amplitude of the displacement.
        """
        glsl_expr = f"snoise(p * {_glsl_format(scale)}) * {_glsl_format(strength)}"
        super().__init__(child, glsl_expr)
        self.scale = scale
        self.strength = strength

    def to_callable(self):
        is_dynamic = isinstance(self.scale, (str, Param)) or isinstance(self.strength, (str, Param))
        if is_dynamic:
            raise TypeError("Cannot create a callable for an object with animated or interactive parameters.")
        # Even with static values, noise is GPU-only and has no NumPy equivalent.
        raise TypeError("Cannot create a callable for an object with procedural noise displacement.")