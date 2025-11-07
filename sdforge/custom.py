import uuid
import atexit
import numpy as np
from .core import SDFObject

# --- Optional GPU Dependency Check for Forge ---
_MODERNGL_AVAILABLE = False
try:
    import moderngl
    import glfw
    _MODERNGL_AVAILABLE = True
except ImportError:
    pass

# --- Custom GLSL ---

class Forge(SDFObject):
    """
    An SDF object defined by a raw GLSL code snippet.
    """
    def __init__(self, glsl_code_body: str):
        """
        Initializes the Forge object.

        Args:
            glsl_code_body (str): A string of GLSL code that returns a float.
                                  The point in space is available as the `vec3 p` variable.
                                  Example: "return length(p) - 1.0;"
        """
        super().__init__()
        if "return" not in glsl_code_body:
            glsl_code_body = f"return {glsl_code_body};"
        self.glsl_code_body = glsl_code_body
        self.unique_id = "forge_func_" + uuid.uuid4().hex[:8]
    def to_glsl(self) -> str: return f"vec4({self.unique_id}(p), -1.0, 0.0, 0.0)"
    def get_glsl_definitions(self) -> list:
        return [f"float {self.unique_id}(vec3 p){{ {self.glsl_code_body} }}"]
    def to_callable(self):
        if not _MODERNGL_AVAILABLE: raise ImportError("To save meshes with Forge objects, 'moderngl' and 'glfw' are required.")
        cls = self.__class__
        if not hasattr(cls, '_mgl_context'):
            if not glfw.init(): raise RuntimeError("glfw.init() failed")
            atexit.register(glfw.terminate)
            glfw.window_hint(glfw.VISIBLE, False); win = glfw.create_window(1, 1, "", None, None)
            glfw.make_context_current(win); cls._mgl_context = moderngl.create_context(require=430)
        ctx = cls._mgl_context
        compute_shader = ctx.compute_shader(f"""
        #version 430
        layout(local_size_x=256, local_size_y=1, local_size_z=1) in;
        layout(std430, binding=0) buffer points {{ vec3 p[]; }};
        layout(std430, binding=1) buffer distances {{ float d[]; }};
        uniform int u_num_points;
        {self.get_glsl_definitions()[0]}
        void main() {{
            uint gid = gl_GlobalInvocationID.x;
            if (gid >= u_num_points) return;
            d[gid] = {self.to_glsl().replace('p', 'p[gid]').replace('vec4', '').strip('()').split(',')[0]};
        }}""")
        def _gpu_evaluator(points_np):
            points_np = np.array(points_np, dtype='f4')
            num_points = len(points_np)
            if 'u_num_points' in compute_shader:
                compute_shader['u_num_points'].value = num_points
            
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
