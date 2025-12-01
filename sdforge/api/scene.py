import sys
import os
import importlib.util
from pathlib import Path
import numpy as np
from .core import SDFNode, GLSLContext
from .io import get_glsl_definitions, generate
from .utils import Debug

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

try:
    get_ipython()
    _IS_NOTEBOOK = True
except NameError:
    _IS_NOTEBOOK = False

MAX_MATERIALS = 64
_IS_RELOADING = False

class Camera:
    """Represents a camera in the scene for static positioning."""
    def __init__(self, position=(5, 4, 5), target=(0, 0, 0), zoom=1.0):
        self.position = np.array(position, dtype=float)
        self.target = np.array(target, dtype=float)
        self.zoom = zoom

class Light:
    """Represents light and shadow properties for the scene."""
    def __init__(self, position=None, ambient_strength=0.1, shadow_softness=8.0, ao_strength=3.0):
        self.position = np.array(position, dtype=float) if position is not None else None
        self.ambient_strength = ambient_strength
        self.shadow_softness = shadow_softness
        self.ao_strength = ao_strength

class Scene:
    """
    Manages the composition, camera, lighting, materials, and rendering of an SDF scene.
    """
    def __init__(self, root: SDFNode = None, camera: Camera = None, light: Light = None, debug: Debug = None):
        self.root = root
        self.camera = camera if camera else Camera()
        self.light = light if light else Light()
        self.debug = debug
        self.materials = []
        self.uniforms = {}
        self.params = {}

    def compile(self, extra_dependencies=None) -> str:
        """
        Compiles the scene graph into GLSL code.
        Returns the concatenated string of Library Code + Custom Definitions + Scene Function.
        """
        if self.root is None:
            return ""
        
        self.materials = []
        self.root._collect_materials(self.materials)
        if len(self.materials) > MAX_MATERIALS:
            print(f"WARNING: Exceeded maximum of {MAX_MATERIALS} materials. Truncating.")
            self.materials = self.materials[:MAX_MATERIALS]

        self.uniforms = {}
        self.root._collect_uniforms(self.uniforms)
        self.params = {}
        self.root._collect_params(self.params)

        ctx = GLSLContext(compiler=self)
        result_var = self.root.to_glsl(ctx)
        
        deps = ctx.dependencies.copy()
        if extra_dependencies:
            deps.update(extra_dependencies)
        
        library_code = get_glsl_definitions(frozenset(deps))
        custom_definitions = "\n".join(ctx.definitions)
        function_body = "\n    ".join(ctx.statements)
        
        scene_function = f"""
vec4 Scene(in vec3 p) {{
    {function_body}
    return {result_var};
}}
"""
        return library_code + "\n" + custom_definitions + "\n" + scene_function

    def render(self, mode='auto', watch=True, width=1280, height=720, save_frame=None):
        """
        Renders the scene.
        """
        if self.root is None:
            print("Scene has no root object.")
            return

        if save_frame:
            try:
                renderer = NativeRenderer(self, width, height, watch=False, headless=True)
                renderer.render_to_file(save_frame)
            except Exception as e:
                print(f"ERROR: Failed to save frame: {e}", file=sys.stderr)
            return

        if mode == 'auto':
            mode = 'mesh' if _IS_NOTEBOOK else 'window'

        if mode == 'mesh':
            self._render_mesh()
        elif mode == 'window':
            if not os.environ.get("DISPLAY") and sys.platform == 'linux': 
                print("WARNING: No display detected. Window creation may fail.", file=sys.stderr)
            renderer = NativeRenderer(self, width, height, watch=watch)
            try:
                renderer.run()
            except Exception as e:
                print(f"ERROR: Failed to launch window: {e}", file=sys.stderr)
        else:
            print(f"ERROR: Unknown render mode '{mode}'.", file=sys.stderr)

    def _render_mesh(self):
        try: import trimesh
        except ImportError: print("ERROR: Mesh rendering requires 'trimesh'.", file=sys.stderr); return
        verts, faces = generate(self.root, verbose=True)
        if len(verts) == 0: print("WARNING: No geometry generated.", file=sys.stderr); return
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)    
        mesh.visual.face_colors = [200, 200, 220, 255]    
        return mesh.show()

class SceneCompiler:
    """
    Legacy helper to compile a node without an explicit Scene object.
    Used by tests and io.export_shader.
    """
    def compile(self, root_node: SDFNode) -> str:
        scene = Scene(root_node)
        return scene.compile()

class NativeRenderer:
    """
    OpenGL renderer using ModernGL and GLFW.
    """
    def __init__(self, scene: Scene, width=1280, height=720, watch=True, headless=False):
        self.scene = scene
        self.width = width
        self.height = height
        self.watching = watch and WATCHDOG_AVAILABLE
        self.headless = headless
        
        self.window = None
        self.ctx = None
        self.program = None
        self.vao = None
        self.vbo = None
        
        self.script_path = os.path.abspath(sys.argv[0])
        self.reload_pending = False

    def _init_context(self):
        import moderngl
        import glfw
        if not glfw.init(): raise RuntimeError("Could not initialize GLFW")
        
        glfw.window_hint(glfw.VISIBLE, not self.headless)
        glfw.window_hint(glfw.RESIZABLE, False if self.headless else True)
        
        self.window = glfw.create_window(self.width, self.height, "SDF Forge", None, None)
        if not self.window: glfw.terminate(); raise RuntimeError("Could not create GLFW window.")
        
        glfw.make_context_current(self.window)
        self.ctx = moderngl.create_context()

    def _compile_shader(self):
        extra_deps = {'utils'}
        is_slice_debug = (self.scene.debug and self.scene.debug.mode == 'slice')
        if not is_slice_debug:
            extra_deps.add('scene')

        scene_code = self.scene.compile(extra_dependencies=extra_deps)
        
        all_uniforms = list(self.scene.uniforms.keys()) + [p.uniform_name for p in self.scene.params.values()]
        custom_uniforms_glsl = "\n".join([f"uniform float {name};" for name in all_uniforms])
        
        materials = self.scene.materials
        material_struct_glsl = "struct MaterialInfo { vec3 color; };\n"
        material_uniform_glsl = f"uniform MaterialInfo u_materials[{max(1, len(materials))}];\n"

        debug = self.scene.debug
        if debug and debug.mode == 'slice':

            scale, h = debug.view_scale, debug.slice_height
            if debug.plane == 'xz': p_expr = f"vec3(st.x * {scale/2.0}, {h}, st.y * {scale/2.0})"
            elif debug.plane == 'yz': p_expr = f"vec3({h}, st.x * {scale/2.0}, st.y * {scale/2.0})"
            else: p_expr = f"vec3(st * {scale/2.0}, {h})"
            
            return self.ctx.program(
                vertex_shader="#version 330 core\nin vec2 in_vert; void main() { gl_Position = vec4(in_vert, 0.0, 1.0); }",
                fragment_shader=f"""
                    #version 330 core
                    uniform vec2 u_resolution; uniform vec4 u_mouse;
                    {custom_uniforms_glsl}
                    {material_struct_glsl} {material_uniform_glsl}
                    out vec4 f_color;
                    vec4 Scene(in vec3 p);
                    {scene_code}
                    void main() {{
                        vec2 st = (2.0 * gl_FragCoord.xy - u_resolution.xy) / u_resolution.y;
                        vec3 p = {p_expr};
                        float d = Scene(p).x;
                        f_color = vec4(debugDistanceField(d), 1.0);
                    }}
                """
            )

        cam = self.scene.camera
        if cam:
            camera_logic = f"cameraStatic(st, vec3({cam.position[0]}, {cam.position[1]}, {cam.position[2]}), vec3({cam.target[0]}, {cam.target[1]}, {cam.target[2]}), {cam.zoom}, ro, rd);"
        else:
            camera_logic = "cameraOrbit(st, u_mouse.xy, u_resolution, 1.0, ro, rd);"

        light = self.scene.light
        light_pos = f"vec3({light.position[0]}, {light.position[1]}, {light.position[2]})" if light.position is not None else "ro"
        
        mat_lookup = "int material_id = int(hit.y); vec3 material_color = vec3(0.8); if (material_id >= 0 && material_id < {count}) {{ material_color = u_materials[material_id].color; }}".format(count=len(materials))
        
        final_color_logic = f"""
            vec3 lightPos = {light_pos}; vec3 lightDir = normalize(lightPos - p);
            float diffuse = max(dot(normal, lightDir), {light.ambient_strength});
            float shadow = softShadow(p + normal * 0.01, lightDir, {light.shadow_softness});
            diffuse *= shadow; float ao = ambientOcclusion(p, normal, {light.ao_strength});
            {mat_lookup}
            color = material_color * diffuse * ao;
        """

        if debug:
            if debug.mode == 'normals': final_color_logic = "color = debugNormals(normal);"
            elif debug.mode == 'steps': final_color_logic = "color = debugSteps(hit.z, 100.0);"

        fragment_shader = f"""
            #version 330 core
            uniform vec2 u_resolution; uniform vec4 u_mouse;
            {custom_uniforms_glsl}
            {material_struct_glsl} {material_uniform_glsl}
            out vec4 f_color;
            vec4 Scene(in vec3 p);
            {scene_code}
            void main() {{
                vec2 st = (2.0 * gl_FragCoord.xy - u_resolution.xy) / u_resolution.y;
                vec3 ro, rd;
                {camera_logic}
                vec4 hit = raymarch(ro, rd);
                float t = hit.x;
                vec3 color = vec3(0.1, 0.12, 0.15);
                if (t > 0.0) {{
                    vec3 p = ro + t * rd;
                    vec3 normal = estimateNormal(p);
                    {final_color_logic}
                }}
                f_color = vec4(color, 1.0);
            }}
        """
        
        try:
            prog = self.ctx.program(vertex_shader="#version 330 core\nin vec2 in_vert; void main() { gl_Position = vec4(in_vert, 0.0, 1.0); }", fragment_shader=fragment_shader)
            for i, mat in enumerate(materials):
                try: prog[f'u_materials[{i}].color'].value = mat.rgb
                except KeyError: pass
            return prog
        except Exception as e:
            print(f"ERROR: Shader compilation failed. Details:\n{e}", file=sys.stderr)
            return self.program

    def run(self):
        import glfw, moderngl
        self._init_context()
        self.program = self._compile_shader()
        if self.program is None: glfw.terminate(); return
        
        vertices = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0], dtype='f4')
        self.vbo = self.ctx.buffer(vertices)
        self.vao = self.ctx.simple_vertex_array(self.program, self.vbo, 'in_vert')
        
        self._start_watcher()
        
        while not glfw.window_should_close(self.window):
            if self.reload_pending: self._reload_script(); self.reload_pending = False
            
            width, height = glfw.get_framebuffer_size(self.window)
            self.ctx.viewport = (0, 0, width, height)
            
            if self.program:
                try: self.program['u_resolution'].value = (width, height)
                except KeyError: pass
                if not self.scene.camera:
                    try: mx, my = glfw.get_cursor_pos(self.window); self.program['u_mouse'].value = (mx, my, 0, 0)
                    except KeyError: pass
                
                # Update Uniforms
                for name, value in self.scene.uniforms.items():
                    try: self.program[name].value = float(value)
                    except KeyError: pass
                for p in self.scene.params.values():
                    try: self.program[p.uniform_name].value = p.value
                    except KeyError: pass
            
            self.ctx.clear(0.1, 0.12, 0.15)
            if self.vao: self.vao.render(mode=moderngl.TRIANGLE_STRIP)
            glfw.swap_buffers(self.window); glfw.poll_events()
        
        glfw.terminate()

    def render_to_file(self, path):
        import moderngl, glfw
        from PIL import Image
        self._init_context()
        self.program = self._compile_shader()
        if self.program is None: glfw.terminate(); return
        
        # Setup Geometry
        vertices = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0], dtype='f4')
        self.vbo = self.ctx.buffer(vertices)
        self.vao = self.ctx.simple_vertex_array(self.program, self.vbo, 'in_vert')
        
        # Framebuffer
        fbo = self.ctx.framebuffer(color_attachments=[self.ctx.texture((self.width, self.height), 4)])
        fbo.use()
        
        # Uniforms
        try: self.program['u_resolution'].value = (self.width, self.height)
        except KeyError: pass
        if not self.scene.camera:
             try: self.program['u_mouse'].value = (0, 0, 0, 0)
             except KeyError: pass
        for name, value in self.scene.uniforms.items():
            try: self.program[name].value = float(value)
            except KeyError: pass
        for p in self.scene.params.values():
            try: self.program[p.uniform_name].value = p.value
            except KeyError: pass

        self.ctx.clear(0.1, 0.12, 0.15)
        self.vao.render(mode=moderngl.TRIANGLE_STRIP)
        
        # Save
        image = Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, -1)
        image.save(path)
        print(f"Saved frame to {path}")
        glfw.terminate()

    def _start_watcher(self):
        if not self.watching: return
        class ChangeHandler(FileSystemEventHandler):
            def __init__(self, renderer): self.renderer = renderer
            def on_modified(self, event):
                if event.src_path == self.renderer.script_path: self.renderer.reload_pending = True
        observer = Observer()
        observer.schedule(ChangeHandler(self), str(Path(self.script_path).parent), recursive=False)
        observer.daemon = True
        observer.start()
        print(f"INFO: Watching '{Path(self.script_path).name}' for changes...")

    def _reload_script(self):
        global _IS_RELOADING
        print(f"INFO: Reloading...")
        _IS_RELOADING = True
        try:
            spec = importlib.util.spec_from_file_location("user_script", self.script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, 'main') and callable(module.main):
                result = module.main()
                new_scene = None
                if isinstance(result, Scene): new_scene = result
                elif isinstance(result, SDFNode): new_scene = Scene(result)
                elif isinstance(result, tuple):
                    s, c, l, d = None, None, None, None
                    for item in result:
                        if isinstance(item, SDFNode): s = item
                        elif isinstance(item, Camera): c = item
                        elif isinstance(item, Light): l = item
                        elif isinstance(item, Debug): d = item
                    if s: new_scene = Scene(s, c, l, d)
                
                if new_scene:
                    self.scene = new_scene
                    new_prog = self._compile_shader()
                    if new_prog:
                        self.program = new_prog
                        self.vao = self.ctx.simple_vertex_array(self.program, self.vbo, 'in_vert')
        except Exception as e:
            print(f"ERROR: Reload failed: {e}")
        finally:
            _IS_RELOADING = False

def render(sdf_obj: SDFNode, camera: Camera = None, light: Light = None, watch=True, debug: Debug = None, mode='auto', **kwargs):
    """
    Main entry point for rendering. Creates a Scene and renders it.
    """
    if _IS_RELOADING: return
    scene = Scene(sdf_obj, camera, light, debug)
    scene.render(mode=mode, watch=watch, **kwargs)