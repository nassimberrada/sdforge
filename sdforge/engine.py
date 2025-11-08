import sys
import numpy as np
from .core import SDFNode, GLSLContext
from .loader import get_glsl_definitions

class SceneCompiler:
    """Compiles an SDFNode tree into a complete GLSL Scene function."""
    def compile(self, root_node: SDFNode) -> str:
        """
        Generates the complete GLSL code for the scene, including all
        dependencies and the main Scene(p) function.
        """
        ctx = GLSLContext(compiler=self)
        
        # This call populates ctx.statements and ctx.dependencies
        result_var = root_node.to_glsl(ctx)

        # Get the GLSL source for all required library functions
        library_code = get_glsl_definitions(frozenset(ctx.dependencies))

        # Assemble the body of the Scene(p) function
        function_body = "\n    ".join(ctx.statements)
        
        scene_function = f"""
vec4 Scene(in vec3 p) {{
    {function_body}
    return {result_var};
}}
"""
        return library_code + "\n" + scene_function

class NativeRenderer:
    """A minimal renderer for displaying the raw SDF distance field."""
    def __init__(self, sdf_obj: SDFNode, width=1280, height=720):
        self.sdf_obj = sdf_obj
        self.width = width
        self.height = height
        self.window = None
        self.ctx = None
        self.program = None
        self.vao = None
        
    def run(self):
        import glfw
        import moderngl

        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")

        self.window = glfw.create_window(self.width, self.height, "SDF Forge", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create GLFW window.")
        glfw.make_context_current(self.window)
        
        self.ctx = moderngl.create_context()
        
        # --- Shader Generation ---
        scene_code = SceneCompiler().compile(self.sdf_obj)

        vertex_shader = """
            #version 330 core
            in vec2 in_vert;
            void main() { gl_Position = vec4(in_vert, 0.0, 1.0); }
        """
        
        fragment_shader = f"""
            #version 330 core
            uniform vec2 u_resolution;
            out vec4 f_color;
            
            {scene_code}

            // Simple raymarcher
            float raymarch(vec3 ro, vec3 rd) {{
                float t = 0.0;
                for (int i = 0; i < 100; i++) {{
                    vec3 p = ro + t * rd;
                    float d = Scene(p).x;
                    if (d < 0.001) return t;
                    t += d;
                    if (t > 100.0) break;
                }}
                return -1.0;
            }}

            void main() {{
                vec2 st = (2.0 * gl_FragCoord.xy - u_resolution.xy) / u_resolution.y;
                
                // --- THIS IS THE CORRECTED CAMERA LOGIC ---
                vec3 ro = vec3(2.5, 2.0, 2.5);
                vec3 target = vec3(0.0, 0.0, 0.0);
                float zoom = 1.5;
                
                vec3 f = normalize(target - ro); // forward vector
                vec3 r = normalize(cross(vec3(0.0, 1.0, 0.0), f)); // right vector
                vec3 u = cross(f, r); // up vector
                vec3 rd = normalize(st.x * r + st.y * u + zoom * f); // final ray direction
                
                float t = raymarch(ro, rd);
                
                // Simple black and white visualization based on hit
                vec3 color = vec3(0.1); // Background color
                if (t > 0.0) {{
                    color = vec3(0.9); // Object color
                }}
                
                f_color = vec4(color, 1.0);
            }}
        """
        
        try:
            self.program = self.ctx.program(
                vertex_shader=vertex_shader, fragment_shader=fragment_shader
            )
            self.program['u_resolution'].value = (self.width, self.height)
        except Exception as e:
            print(f"ERROR: Shader compilation failed:\n{e}", file=sys.stderr)
            glfw.terminate()
            return
            
        vertices = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0], dtype='f4')
        vbo = self.ctx.buffer(vertices)
        self.vao = self.ctx.simple_vertex_array(self.program, vbo, 'in_vert')
        
        while not glfw.window_should_close(self.window):
            self.ctx.clear(0.1, 0.1, 0.1)
            self.vao.render(mode=moderngl.TRIANGLE_STRIP)
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        glfw.terminate()

def render(sdf_obj: SDFNode, **kwargs):
    """Public API to launch the renderer."""
    try:
        import moderngl, glfw
    except ImportError:
        print("ERROR: Live rendering requires 'moderngl' and 'glfw'.", file=sys.stderr)
        return
    renderer = NativeRenderer(sdf_obj, **kwargs)
    renderer.run()