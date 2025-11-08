from ..core import SDFNode, GLSLContext

class Displace(SDFNode):
    """Displaces the surface of a child object using a raw GLSL expression."""
    glsl_dependencies = {"opDisplace"}

    def __init__(self, child: SDFNode, displacement_glsl: str):
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
    """Displaces the surface of a child object using a procedural noise function."""
    glsl_dependencies = {"opDisplace", "snoise"}

    def __init__(self, child: SDFNode, scale: float = 10.0, strength: float = 0.1):
        glsl_expr = f"snoise(p * {float(scale)}) * {float(strength)}"
        super().__init__(child, glsl_expr)