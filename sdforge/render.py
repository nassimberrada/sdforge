import sys
import os
import time
from pathlib import Path
import importlib.util
import numpy as np
from .core import SDFNode, GLSLContext
from .loader import get_glsl_definitions
from .api.camera import Camera
from .api.light import Light
from .debug import Debug

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

MAX_MATERIALS = 64

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
    def __init__(self, sdf_obj: SDFNode, camera: Camera = None, light: Light = None, debug: Debug = None, watch=True, width=1280, height=720, **kwargs):
        self.sdf_obj = sdf_obj
        self.camera = camera
        self.light = light
        self.debug = debug
        self.watching = watch and WATCHDOG_AVAILABLE
        self.width = width
        self.height = height
        self.window = None
        self.ctx = None
        self.program = None
        self.vao = None
        self.vbo = None
        self.uniforms = {}
        self.params = {}
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
                new_sdf_obj, new_cam_obj, new_light_obj, new_debug_obj = None, None, None, None

                if isinstance(result, SDFNode):
                    new_sdf_obj = result
                elif isinstance(result, tuple):
                    for item in result:
                        if isinstance(item, SDFNode): new_sdf_obj = item
                        if isinstance(item, Camera): new_cam_obj = item
                        if isinstance(item, Light): new_light_obj = item
                        if isinstance(item, Debug): new_debug_obj = item
                
                if new_sdf_obj:
                    self.sdf_obj = new_sdf_obj
                    self.camera = new_cam_obj
                    self.light = new_light_obj
                    self.debug = new_debug_obj
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
        # --- Collect scene components ---
        materials = []
        self.sdf_obj._collect_materials(materials)
        if len(materials) > MAX_MATERIALS:
            print(f"WARNING: Exceeded maximum of {MAX_MATERIALS} materials. Truncating.")
            materials = materials[:MAX_MATERIALS]

        self.uniforms = {}
        self.sdf_obj._collect_uniforms(self.uniforms)
        
        self.params = {}
        self.sdf_obj._collect_params(self.params)
        
        scene_code = SceneCompiler().compile(self.sdf_obj)
        
        # --- GLSL Library Imports ---
        glsl_deps = {'camera', 'raymarching', 'light'}
        if self.debug:
            glsl_deps.add('debug')
        renderer_library_code = get_glsl_definitions(frozenset(glsl_deps))

        # --- Camera Logic ---
        if self.camera:
            cam = self.camera
            pos = f"vec3({float(cam.position[0])}, {float(cam.position[1])}, {float(cam.position[2])})"
            tgt = f"vec3({float(cam.target[0])}, {float(cam.target[1])}, {float(cam.target[2])})"
            camera_logic_glsl = f"cameraStatic(st, {pos}, {tgt}, {float(cam.zoom)}, ro, rd);"
        else:
            camera_logic_glsl = "cameraOrbit(st, u_mouse.xy, u_resolution, 1.0, ro, rd);"
        
        # --- Lighting Logic ---
        light = self.light or Light()
        light_pos_str = f"vec3({light.position[0]}, {light.position[1]}, {light.position[2]})" if light.position else "ro"
        
        # --- Material Logic ---
        material_struct_glsl = "struct MaterialInfo { vec3 color; };\n"
        material_uniform_glsl = f"uniform MaterialInfo u_materials[{max(1, len(materials))}];\n"
        material_lookup_glsl = """
            int material_id = int(hit.y);
            vec3 material_color = vec3(0.8); // Default color
            if (material_id >= 0 && material_id < {material_count}) {{
                material_color = u_materials[material_id].color;
            }}
        """.format(material_count=len(materials))


        # --- Debug Logic ---
        final_color_logic = """
            vec3 lightPos = {light_pos};
            vec3 lightDir = normalize(lightPos - p);
            float diffuse = max(dot(normal, lightDir), {ambient});
            float shadow = softShadow(p + normal * 0.01, lightDir, {shadow_softness});
            diffuse *= shadow;
            float ao = ambientOcclusion(p, normal, {ao_strength});
            {material_lookup}
            color = material_color * diffuse * ao;
        """.format(
            light_pos=light_pos_str,
            ambient=light.ambient_strength,
            shadow_softness=light.shadow_softness,
            ao_strength=light.ao_strength,
            material_lookup=material_lookup_glsl
        )
        if self.debug:
            if self.debug.mode == 'normals':
                final_color_logic = "color = debugNormals(normal);"
            elif self.debug.mode == 'steps':
                final_color_logic = "color = debugSteps(hit.z, 100.0);"
            else:
                print(f"WARNING: Unknown debug mode '{self.debug.mode}'. Ignoring.")

        all_uniforms = list(self.uniforms.keys()) + [p.uniform_name for p in self.params.values()]
        custom_uniforms_glsl = "\n".join([f"uniform float {name};" for name in all_uniforms])
        
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
            {material_struct_glsl}
            {material_uniform_glsl}
            out vec4 f_color;

            // Forward declare Scene() so renderer functions can find it.
            vec4 Scene(in vec3 p);
            
            {renderer_library_code}
            {scene_code}

            void main() {{
                vec2 st = (2.0 * gl_FragCoord.xy - u_resolution.xy) / u_resolution.y;
                vec3 ro, rd;
                {camera_logic_glsl}
                
                vec4 hit = raymarch(ro, rd);
                float t = hit.x;
                
                vec3 color = vec3(0.1, 0.12, 0.15); // Background color
                if (t > 0.0) {{
                    vec3 p = ro + t * rd;
                    vec3 normal = estimateNormal(p);
                    {final_color_logic}
                }}
                
                f_color = vec4(color, 1.0);
            }}
        """
        
        try:
            new_program = self.ctx.program(
                vertex_shader=vertex_shader, fragment_shader=fragment_shader
            )
            print("INFO: Shader compiled successfully.")
            
            # Upload material data
            for i, mat in enumerate(materials):
                new_program[f'u_materials[{i}].color'].value = mat.color

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
        
        self._start_watcher()

        while not glfw.window_should_close(self.window):
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
            
            # NEW: Upload Param uniforms
            for p in self.params.values():
                try: self.program[p.uniform_name].value = p.value
                except KeyError: pass

            self.ctx.clear(0.1, 0.12, 0.15)
            self.vao.render(mode=moderngl.TRIANGLE_STRIP)
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        glfw.terminate()

def render(sdf_obj: SDFNode, camera: Camera = None, light: Light = None, watch=True, debug: Debug = None, **kwargs):
    """Public API to launch the renderer."""
    try:
        import moderngl, glfw
    except ImportError:
        print("ERROR: Live rendering requires 'moderngl' and 'glfw'.", file=sys.stderr)
        return
    # Pass `watch` parameter to the renderer
    renderer = NativeRenderer(sdf_obj, camera=camera, light=light, watch=watch, debug=debug, **kwargs)
    renderer.run()