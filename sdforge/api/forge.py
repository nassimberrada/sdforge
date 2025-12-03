import uuid
import numpy as np
from .core import SDFNode, GLSLContext

class Forge(SDFNode):
    """
    An SDF object defined by a raw GLSL code snippet.
    """
    def __init__(self, glsl_code_body: str, uniforms: dict = None):
        """
        Initializes the Forge object with a GLSL expression.

        Args:
            glsl_code_body (str): A string of GLSL code that returns a float
                                  distance. The point in space is available as
                                  the `vec3 p` variable. If the code does not
                                  contain 'return', it will be added.
            uniforms (dict, optional): A dictionary of uniforms to be passed to
                                       the GLSL code. Keys are uniform names
                                       (e.g., 'u_radius') and values are the
                                       corresponding floats. Defaults to None.
        """
        super().__init__()
        if "return" not in glsl_code_body:
            glsl_code_body = f"return {glsl_code_body};"
        self.glsl_code_body = glsl_code_body
        self.uniforms = uniforms or {}
        self.unique_id = "forge_func_" + uuid.uuid4().hex[:8]
        self.glsl_dependencies = set()
    
    def _get_glsl_definition(self) -> str:
        uniform_params = "".join([f", in float {name}" for name in self.uniforms.keys()])
        return f"float {self.unique_id}(vec3 p{uniform_params}){{ {self.glsl_code_body} }}"
    
    def to_glsl(self, ctx: GLSLContext) -> str:
        ctx.dependencies.update(self.glsl_dependencies)
        ctx.definitions.add(self._get_glsl_definition())
        
        uniform_args = "".join([f", {name}" for name in self.uniforms.keys()])
        result_expr = f"vec4({self.unique_id}({ctx.p}{uniform_args}), -1.0, 0.0, 0.0)"
        return ctx.new_variable('vec4', result_expr)

    def _collect_uniforms(self, uniforms_dict: dict):
        uniforms_dict.update(self.uniforms)
        super()._collect_uniforms(uniforms_dict)