import numpy as np
import atexit
import sys
from .core import GLSLContext
from .loader import get_glsl_definitions

class HeadlessContext:
    """
    Manages a hidden OpenGL context for compute operations.
    Singleton pattern to prevent creating multiple contexts.
    """
    _instance = None
    _ctx = None

    @classmethod
    def get(cls):
        if cls._ctx is not None:
            return cls._ctx

        try:
            import glfw
            import moderngl
        except ImportError:
            # Caller handles the ImportError (e.g. falling back to CPU)
            return None

        if not glfw.init():
            return None

        glfw.window_hint(glfw.VISIBLE, False)
        # Request a context capable of Compute Shaders (OpenGL 4.3+)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        window = glfw.create_window(1, 1, "SDForge Headless", None, None)
        if not window:
            glfw.terminate()
            return None

        glfw.make_context_current(window)
        cls._ctx = moderngl.create_context()
        
        # Cleanup on exit
        def cleanup():
            glfw.terminate()
        atexit.register(cleanup)

        return cls._ctx

def get_callable(node, profile_mode=False):
    """
    Compiles the SDFNode to a GLSL Compute Shader and returns a 
    Python callable that evaluates it on the GPU.
    """
    ctx = HeadlessContext.get()
    if not ctx:
        raise RuntimeError("GPU backend requested but moderngl/glfw could not be initialized.")

    # 1. Generate the scene GLSL manually to support profile_mode toggle
    # We don't use SceneCompiler directly because we need to inject specific logic
    # depending on whether we want 3D distance or 2D profile distance.
    glsl_ctx = GLSLContext(compiler=None)
    
    if profile_mode:
        result_var = node.to_profile_glsl(glsl_ctx)
    else:
        result_var = node.to_glsl(glsl_ctx)

    library_code = get_glsl_definitions(frozenset(glsl_ctx.dependencies))
    custom_definitions = "\n".join(glsl_ctx.definitions)
    function_body = "\n    ".join(glsl_ctx.statements)

    scene_function = f"""
    vec4 Scene(in vec3 p) {{
        {function_body}
        return {result_var};
    }}
    """
    
    # 2. Collect Uniforms and Params
    uniforms, params = {}, {}
    node._collect_uniforms(uniforms)
    node._collect_params(params)
    
    uniform_decls = [f"uniform float {name};" for name in uniforms.keys()]
    uniform_decls += [f"uniform float {p.uniform_name};" for p in params.values()]
    uniform_block = "\n".join(uniform_decls)

    # 3. Assemble Compute Shader
    # Input points are packed as vec4 (x,y,z,pad) for std430 alignment safety
    cs_source = f"""
    #version 430
    layout(local_size_x=256) in;
    
    layout(std430, binding=0) buffer InputPoints {{ 
        vec4 points[]; 
    }};
    
    layout(std430, binding=1) buffer OutputDists {{ 
        float dists[]; 
    }};
    
    {library_code}
    {custom_definitions}
    {uniform_block}
    {scene_function}

    void main() {{
        uint gid = gl_GlobalInvocationID.x;
        if (gid >= points.length()) return;
        
        vec3 p = points[gid].xyz;
        dists[gid] = Scene(p).x;
    }}
    """
    
    try:
        program = ctx.compute_shader(cs_source)
    except Exception as e:
        # Include source in error for debugging
        raise RuntimeError(f"Failed to compile GPU shader:\n{e}")
    
    # 4. Create the execution closure
    def _gpu_evaluator(points_np: np.ndarray) -> np.ndarray:
        points_np = np.array(points_np, dtype='f4')
        num_points = len(points_np)
        
        if num_points == 0:
            return np.array([], dtype='f4')
        
        # Pack (N,3) into (N,4)
        padded_points = np.zeros((num_points, 4), dtype='f4')
        padded_points[:, :3] = points_np
        
        # Update Uniforms
        for name, val in uniforms.items():
            if name in program: program[name].value = float(val)
        for p in params.values():
            if p.uniform_name in program: program[p.uniform_name].value = float(p.value)

        # Buffer Management
        # In a high-performance scenario, we might cache these buffers if size doesn't change,
        # but for safety/simplicity in this API, we create/release per call.
        in_buf = ctx.buffer(padded_points.tobytes())
        out_buf = ctx.buffer(reserve=num_points * 4) # float32 = 4 bytes
        
        in_buf.bind_to_storage_buffer(0)
        out_buf.bind_to_storage_buffer(1)
        
        # Dispatch
        group_size = (num_points + 255) // 256
        program.run(group_x=group_size)
        
        # Readback
        results = np.frombuffer(out_buf.read(), dtype='f4')
        
        in_buf.release()
        out_buf.release()
        
        return results
        
    return _gpu_evaluator