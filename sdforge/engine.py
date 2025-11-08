import sys
import os
import time
from pathlib import Path
import importlib.util
import numpy as np
from .core import SDFNode, GLSLContext
from .loader import get_glsl_definitions
from .api.camera import Camera

# NEW: Add watchdog imports
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False


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
    def __init__(self, sdf_obj: SDFNode, camera: Camera = None, watch=True, width=1280, height=720, **kwargs):
        self.sdf_obj = sdf_obj
        self.camera = camera
        self.watching = watch and WATCHDOG_AVAILABLE
        self.width = width
        self.height = height
        self.window = None
        self.ctx = None
        self.program = None
        self.vao = None
        self.vbo = None # NEW: Make VBO an instance attribute
        self.uniforms = {}
        # NEW ATTRIBUTES for hot-reloading
        self.script_path = os.path.abspath(sys.argv[0])
        self.reload_pending = False
        
    def _reload_script(self):
        """Dynamically reloads the user's script and updates the scene."""
        print(f"INFO: Change detected in '{Path(self.script_path).name}'. Reloading...")
        try:
            spec = importlib.util.spec_from_file_location("user_script", self.script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, 'main') and callable(module.main):
                result = module.main()
                new_sdf_obj, new_cam_obj = None, None

                if isinstance(result, SDFNode):
                    new_sdf_obj = result
                elif isinstance(result, tuple):
                    for item in result:
                        if isinstance(item, SDFNode): new_sdf_obj = item
                        if isinstance(item, Camera): new_cam_obj = item
                
                if new_sdf_obj:
                    self.sdf_obj = new_sdf_obj
                    self.camera = new_cam_obj
                    # Re-compile shader and vertex array
                    self.program = self._compile_shader()
                    if self.program:
                        self.vao = self.ctx.simple_vertex_array(self.program, self.vbo, 'in_vert')
            else:
                print("WARNING: No valid `main` function found in script. Cannot reload.")
        except Exception as e:
            print(f"ERROR: Failed to reload script: {e}")

    def _start_watcher(self):
        """Initializes and starts the watchdog file observer."""
        if not self.watching:
            if not WATCHDOG_AVAILABLE:
                print("INFO: Hot-reloading disabled. `watchdog` not installed. Run 'pip install watchdog'.")
            return

        class ChangeHandler(FileSystemEventHandler):
            def __init__(self, renderer_instance):
                self.renderer = renderer_instance
            def on_modified(self, event):
                if event.src_path == self.renderer.script_path:
                    self.renderer.reload_pending = True
        
        observer = Observer()
        observer.schedule(ChangeHandler(self), str(Path(self.script_path).parent), recursive=False)
        observer.daemon = True
        observer.start()
        print(f"INFO: Watching '{Path(self.script_path).name}' for changes...")
        
    def _compile_shader(self):
        """Compiles the full fragment shader for the current scene."""
        # This helper function encapsulates the shader string generation
        self.uniforms = {}
        self.sdf_obj._collect_uniforms(self.uniforms)
        
        scene_code = SceneCompiler().compile(self.sdf_obj)
        camera_code = get_glsl_definitions(frozenset(['camera']))

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
            new_program = self.ctx.program(
                vertex_shader=vertex_shader, fragment_shader=fragment_shader
            )
            print("INFO: Shader compiled successfully.")
            return new_program
        except Exception as e:
            print(f"ERROR: Shader compilation failed. Keeping previous shader. Details:\n{e}", file=sys.stderr)
            return self.program # Return old program on failure

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
        self.program = self._compile_shader()
        
        vertices = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0], dtype='f4')
        self.vbo = self.ctx.buffer(vertices) # Assign to instance
        self.vao = self.ctx.simple_vertex_array(self.program, self.vbo, 'in_vert')
        
        self._start_watcher() # NEW: Start the file watcher

        while not glfw.window_should_close(self.window):
            # NEW: Check for reload flag at start of loop
            if self.reload_pending:
                self._reload_script()
                self.reload_pending = False

            width, height = glfw.get_framebuffer_size(self.window)
            self.ctx.viewport = (0, 0, width, height)
            
            try: self.program['u_resolution'].value = (width, height)
            except KeyError: pass
            
            if not self.camera:
                try:
                    mx, my = glfw.get_cursor_pos(self.window)
                    self.program['u_mouse'].value = (mx, my, 0, 0)
                except KeyError: pass
            
            for name, value in self.uniforms.items():
                try: self.program[name].value = float(value)
                except KeyError: pass

            self.ctx.clear(0.1, 0.12, 0.15)
            self.vao.render(mode=moderngl.TRIANGLE_STRIP)
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        glfw.terminate()

def render(sdf_obj: SDFNode, camera: Camera = None, watch=True, **kwargs):
    """Public API to launch the renderer."""
    try:
        import moderngl, glfw
    except ImportError:
        print("ERROR: Live rendering requires 'moderngl' and 'glfw'.", file=sys.stderr)
        return
    # Pass `watch` parameter to the renderer
    renderer = NativeRenderer(sdf_obj, camera=camera, watch=watch, **kwargs)
    renderer.run()