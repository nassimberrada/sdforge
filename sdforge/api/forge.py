import uuid
import atexit
import numpy as np
from .core import SDFNode, GLSLContext

# --- Optional GPU Dependency Check for Forge ---
_MODERNGL_AVAILABLE = False
try:
    import moderngl
    import glfw
    _MODERNGL_AVAILABLE = True
except ImportError:
    pass

class Forge(SDFNode):
    """
    An SDF object defined by a raw GLSL code snippet.

    Forge provides an escape hatch to define custom shapes or operations
    directly in GLSL, which can be seamlessly integrated with other `sdforge`
    objects.
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
        
        Example:
            >>> from sdforge import Forge, box
            >>> # Create a sphere using a raw GLSL expression
            >>> custom_sphere = Forge("length(p) - 1.0")
            >>>
            >>> # Create a box with a controllable size uniform
            >>> glsl_box = '''
            ...     vec3 q = abs(p) - vec3(u_size);
            ...     return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
            ... '''
            >>> custom_box = Forge(glsl_box, uniforms={'u_size': 0.8})
            >>>
            >>> scene = custom_sphere | custom_box.translate((2, 0, 0))
        """
        super().__init__()
        if "return" not in glsl_code_body:
            glsl_code_body = f"return {glsl_code_body};"
        self.glsl_code_body = glsl_code_body
        self.uniforms = uniforms or {}
        self.unique_id = "forge_func_" + uuid.uuid4().hex[:8]
        self.glsl_dependencies = set()
    
    def _get_glsl_definition(self) -> str:
        """Helper to generate the full GLSL function and uniform text."""
        # The renderer is responsible for declaring uniforms globally.
        # This function should only define the helper function itself.
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

    def to_callable(self):
        if not _MODERNGL_AVAILABLE:
            raise ImportError("To create a callable for Forge objects, 'moderngl' and 'glfw' are required.")
        if self.uniforms:
            raise TypeError("Cannot create a callable for a Forge object with uniforms.")

        cls = self.__class__
        if not hasattr(cls, '_mgl_context'):
            if not glfw.init(): raise RuntimeError("glfw.init() failed")
            atexit.register(glfw.terminate)
            glfw.window_hint(glfw.VISIBLE, False)
            win = glfw.create_window(1, 1, "", None, None)
            glfw.make_context_current(win)
            cls._mgl_context = moderngl.create_context(require=430)
        
        ctx = cls._mgl_context
        
        func_def = self._get_glsl_definition()
        # The GLSL call needs the current point `p`, but in the compute shader `p` is an array.
        # We replace `p` with `p[gid]` for the call.
        call_expr = f"{self.unique_id}(p[gid])"
        
        # We need to manually include dependencies for the callable's compute shader
        from .loader import get_glsl_definitions
        library_code = get_glsl_definitions(frozenset(self.glsl_dependencies))
        
        compute_shader_src = f"""
        #version 430
        layout(local_size_x=256) in;
        layout(std430, binding=0) buffer points {{ vec3 p[]; }};
        layout(std430, binding=1) buffer distances {{ float d[]; }};
        
        {library_code}
        {func_def}

        void main() {{
            uint gid = gl_GlobalInvocationID.x;
            d[gid] = {call_expr};
        }}
        """
        compute_shader = ctx.compute_shader(compute_shader_src)

        def _gpu_evaluator(points_np: np.ndarray) -> np.ndarray:
            points_np = np.array(points_np, dtype='f4')
            num_points = len(points_np)
            
            padded_points = np.zeros((num_points, 4), dtype='f4')
            padded_points[:, :3] = points_np
            
            point_buffer = ctx.buffer(padded_points.tobytes())
            dist_buffer = ctx.buffer(reserve=num_points * 4)
            
            point_buffer.bind_to_storage_buffer(0)
            dist_buffer.bind_to_storage_buffer(1)
            
            group_size = (num_points + 255) // 256
            compute_shader.run(group_x=group_size)
            return np.frombuffer(dist_buffer.read(), dtype='f4')
        
        return _gpu_evaluator