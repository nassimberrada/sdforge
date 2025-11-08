import sys
import numpy as np
from .core import SDFNode, GLSLContext
from .loader import get_glsl_definitions
from .api.camera import Camera

class SceneCompiler:
    """Compiles an SDFNode tree into a complete GLSL Scene function."""
    def compile(self, root_node: SDFNode) -> str:
        ctx = GLSLContext(compiler=self)
        result_var = root_node.to_glsl(ctx)
        
        library_code = get_glsl_definitions(frozenset(ctx.dependencies))
        custom_definitions = "\n".join(ctx.definitions)
        function_body = "\n    ".join(ctx.statements)
        
        scene_function = f"""
vec4 Scene(in vec3 p) {{
    {function_body}
    return {result_var};
}}
"""
        return library_code + "\n" + custom_definitions + "\n" + scene_function

class NativeRenderer:
    """A minimal renderer for displaying the raw SDF distance field."""
    def __init__(self, sdf_obj: SDFNode, camera: Camera = None, width=1280, height=720):
        self.sdf_obj = sdf_obj
        self.camera = camera
        self.width = width
        self.height = height
        self.window = None
        self.ctx = None
        self.program = None
        self.vao = None
        self.uniforms = {}
        
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
        
        # --- Collect Uniforms ---
        self.sdf_obj._collect_uniforms(self.uniforms)
        
        # --- Shader Generation ---
        scene_code = SceneCompiler().compile(self.sdf_obj)
        camera_code = get_glsl_definitions(frozenset(['camera']))

        # Determine which camera function to use in GLSL
        if self.camera:
            cam = self.camera
            pos = f"vec3({float(cam.position[0])}, {float(cam.position[1])}, {float(cam.position[2])})"
            tgt = f"vec3({float(cam.target[0])}, {float(cam.target[1])}, {float(cam.target[2])})"
            camera_logic_glsl = f"cameraStatic(st, {pos}, {tgt}, {float(cam.zoom)}, ro, rd);"
        else:
            camera_logic_glsl = "cameraOrbit(st, u_mouse.xy, u_resolution, 1.0, ro, rd);"
            
        custom_uniforms_glsl = "\n".join([f"uniform float {name};" for name in self.uniforms.keys()])

        vertex_shader = """
            #version 330 core
            in vec2 in_vert;
            void main() { gl_Position = vec4(in_vert, 0.0, 1.0); }
        """
        
        fragment_shader = f"""
            #version 330 core
            uniform vec2 u_resolution;
            uniform vec4 u_mouse;
            {custom_uniforms_glsl}
            out vec4 f_color;
            
            {camera_code}
            {scene_code}

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
            
            vec3 estimateNormal(vec3 p) {{
                float eps = 0.001;
                vec2 e = vec2(1.0, -1.0) * 0.5773 * eps;
                return normalize(
                    e.xyy * Scene(p + e.xyy).x +
                    e.yyx * Scene(p + e.yyx).x +
                    e.yxy * Scene(p + e.yxy).x +
                    e.xxx * Scene(p + e.xxx).x
                );
            }}

            void main() {{
                vec2 st = (2.0 * gl_FragCoord.xy - u_resolution.xy) / u_resolution.y;
                vec3 ro, rd;
                {camera_logic_glsl}
                
                float t = raymarch(ro, rd);
                
                vec3 color = vec3(0.1, 0.12, 0.15); // Background color
                if (t > 0.0) {{
                    vec3 p = ro + t * rd;
                    vec3 normal = estimateNormal(p);
                    // Simple diffuse lighting from a fixed point
                    vec3 lightDir = normalize(vec3(0.8, 0.7, 0.6));
                    float diffuse = max(dot(normal, lightDir), 0.2);
                    color = vec3(0.9) * diffuse;
                }}
                
                f_color = vec4(color, 1.0);
            }}
        """
        
        try:
            self.program = self.ctx.program(
                vertex_shader=vertex_shader, fragment_shader=fragment_shader
            )
        except Exception as e:
            print(f"ERROR: Shader compilation failed:\n{e}", file=sys.stderr)
            glfw.terminate()
            return
            
        vertices = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0], dtype='f4')
        vbo = self.ctx.buffer(vertices)
        self.vao = self.ctx.simple_vertex_array(self.program, vbo, 'in_vert')
        
        while not glfw.window_should_close(self.window):
            width, height = glfw.get_framebuffer_size(self.window)
            self.ctx.viewport = (0, 0, width, height)
            
            try: self.program['u_resolution'].value = (width, height)
            except KeyError: pass
            
            if not self.camera: # Only update mouse uniform for orbit camera
                try:
                    mx, my = glfw.get_cursor_pos(self.window)
                    self.program['u_mouse'].value = (mx, my, 0, 0)
                except KeyError: pass
            
            # Upload custom uniforms
            for name, value in self.uniforms.items():
                try:
                    self.program[name].value = float(value)
                except KeyError:
                    pass # Uniform may have been optimized out by the GLSL compiler

            self.ctx.clear(0.1, 0.12, 0.15)
            self.vao.render(mode=moderngl.TRIANGLE_STRIP)
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        glfw.terminate()

def render(sdf_obj: SDFNode, camera: Camera = None, **kwargs):
    """Public API to launch the renderer."""
    try:
        import moderngl, glfw
    except ImportError:
        print("ERROR: Live rendering requires 'moderngl' and 'glfw'.", file=sys.stderr)
        return
    renderer = NativeRenderer(sdf_obj, camera=camera, **kwargs)
    renderer.run()