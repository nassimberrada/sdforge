import uuid
from ..core import SDFNode, GLSLContext
from ..utils.helpers import _glsl_format

class Function(SDFNode):
    """
    An SDF object defined by an implicit algebraic equation (f(x,y,z) = 0).
    Automatically calculates the numerical gradient to create a valid pseudo-SDF.
    """
    def __init__(self, glsl_expr: str, safety: float = 0.5, uniforms: dict = None):
        """
        Args:
            glsl_expr (str): The algebraic expression, e.g., "p.x*p.x + p.y*p.y - 1.0".
            safety (float): Multiplier to prevent ray overshoot. Lower is safer but slower. 
                            Defaults to 0.5.
            uniforms (dict, optional): Custom uniforms to pass into the equation.
        """
        super().__init__()
        if "return" in glsl_expr:
            self.glsl_code_body = glsl_expr
        else:
            self.glsl_code_body = f"return {glsl_expr};"
            
        self.safety = safety
        self.uniforms = uniforms or {}
        self.unique_id = "implicit_func_" + uuid.uuid4().hex[:8]
        self.glsl_dependencies = set()

    def _get_glsl_definition(self) -> str:
        uniform_params = "".join([f", in float {name}" for name in self.uniforms.keys()])
        return f"float {self.unique_id}(vec3 p{uniform_params}){{\n    {self.glsl_code_body}\n}}"

    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        ctx.definitions.add(self._get_glsl_definition())
        
        uniform_args = "".join([f", {name}" for name in self.uniforms.keys()])
        func_call = f"{self.unique_id}({{pos}}{uniform_args})"
        
        val_var = ctx.new_variable("float", func_call.format(pos=ctx.p))
        e_var = ctx.new_variable("vec2", "vec2(0.001, 0.0)")
        
        grad_x = f"({func_call.format(pos=f'({ctx.p} + {e_var}.xyy)')} - {func_call.format(pos=f'({ctx.p} - {e_var}.xyy)')})"
        grad_y = f"({func_call.format(pos=f'({ctx.p} + {e_var}.yxy)')} - {func_call.format(pos=f'({ctx.p} - {e_var}.yxy)')})"
        grad_z = f"({func_call.format(pos=f'({ctx.p} + {e_var}.yyx)')} - {func_call.format(pos=f'({ctx.p} - {e_var}.yyx)')})"
        
        grad_var = ctx.new_variable("vec3", f"vec3({grad_x}, {grad_y}, {grad_z}) / (2.0 * {e_var}.x)")
        
        dist_expr = f"({val_var} / (length({grad_var}) + 0.0001)) * {_glsl_format(self.safety)}"
        dist_var = ctx.new_variable("float", f"clamp({dist_expr}, -1.0, 1.0)")
        
        return ctx.new_variable('vec4', f"vec4({dist_var}, -1.0, 0.0, 0.0)")

    def to_profile_glsl(self, ctx: GLSLContext) -> str:
        return self.to_glsl(ctx)

    def _collect_uniforms(self, uniforms_dict: dict):
        uniforms_dict.update(self.uniforms)
        super()._collect_uniforms(uniforms_dict)