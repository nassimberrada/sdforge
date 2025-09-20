import sys
import os
import time
from pathlib import Path
import importlib.util
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .api import Camera, Light

# The maximum number of unique materials supported in a scene.
MAX_MATERIALS = 64

def _glsl_format(val):
    """Formats a Python value for injection into a GLSL string."""
    if isinstance(val, str):
        return val
    return f"{float(val)}"

def is_in_colab():
    """Checks if the code is running in a Google Colab environment."""
    return 'google.colab' in sys.modules

def get_glsl_content(filename: str) -> str:
    """Reads the content of a GLSL file from the package."""
    glsl_dir = Path(__file__).parent / 'glsl'
    try:
        with open(glsl_dir / filename, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"ERROR: Could not find GLSL file: {filename}")
        return ""

def assemble_shader_code(sdf_obj) -> str:
    """Assembles the final GLSL scene function from an SDF object."""
    # The Scene function now returns a vec4: (distance, material_id, 0, 0)
    scene_glsl = sdf_obj.to_glsl()
    inline_definitions = []
    # Use a set to prevent duplicates, which cause redefinition errors
    unique_defs = set(sdf_obj.get_glsl_definitions())
    for d in unique_defs:
        inline_definitions.append(d)
    joined_defs = '\n'.join(inline_definitions)
    return f"""
    {joined_defs}
    vec4 Scene(in vec3 p) {{ return {scene_glsl}; }}
    """

class NativeRenderer:
    """Handles the creation of a native window and renders the SDF."""

    def __init__(self, sdf_obj, camera=None, light=None, watch=False, width=1280, height=720, record=None, bg_color=(0.1, 0.12, 0.15)):
        self.sdf_obj = sdf_obj
        self.camera = camera
        self.light = light
        self.watching = watch
        self.width = width
        self.height = height
        self.record_path = record
        self.bg_color = bg_color
        self.window = None
        self.ctx = None
        self.program = None
        self.vao = None
        self.vbo = None
        self.script_path = os.path.abspath(sys.argv[0])
        self.reload_pending = False

    def _init_window(self):
        import glfw
        try:
            if not glfw.init():
                raise RuntimeError("Could not initialize GLFW")
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
            self.window = glfw.create_window(self.width, self.height, "SDF Forge Viewer", None, None)
            if not self.window:
                glfw.terminate()
                raise RuntimeError("Could not create GLFW window. Your GPU may not be supported or drivers may be missing.")
            glfw.make_context_current(self.window)
        except Exception as e:
            print(f"FATAL: Failed to create an OpenGL context. {e}", file=sys.stderr)
            print("INFO: This may be due to missing graphics drivers or running in an environment without a display server (e.g., a raw terminal).", file=sys.stderr)
            sys.exit(1)

    def _compile_shader(self):
        import moderngl
        
        materials = []
        self.sdf_obj._collect_materials(materials)
        if len(materials) > MAX_MATERIALS:
            print(f"WARNING: Exceeded maximum of {MAX_MATERIALS} materials. Truncating.")
            materials = materials[:MAX_MATERIALS]

        material_struct_glsl = "struct MaterialInfo { vec3 color; };\n"
        material_uniform_glsl = f"uniform MaterialInfo u_materials[{max(1, len(materials))}];\n"

        # --- Light ---
        light_pos_str = "ro"
        ambient_strength_str = "0.1"
        shadow_softness_str = "8.0"
        ao_strength_str = "3.0"

        if self.light:
            if self.light.position:
                pos = self.light.position
                light_pos_str = f"vec3({_glsl_format(pos[0])}, {_glsl_format(pos[1])}, {_glsl_format(pos[2])})"
            ambient_strength_str = _glsl_format(self.light.ambient_strength)
            shadow_softness_str = _glsl_format(self.light.shadow_softness)
            ao_strength_str = _glsl_format(self.light.ao_strength)
        
        # --- Camera ---
        camera_logic_glsl = ""
        if self.camera:
            pos = self.camera.position
            pos_str = f"vec3({_glsl_format(pos[0])}, {_glsl_format(pos[1])}, {_glsl_format(pos[2])})"
            
            target = self.camera.target
            target_str = f"vec3({_glsl_format(target[0])}, {_glsl_format(target[1])}, {_glsl_format(target[2])})"

            zoom_str = _glsl_format(self.camera.zoom)
            
            camera_logic_glsl = f"cameraStatic(st, {pos_str}, {target_str}, {zoom_str}, ro, rd);"
        else:
            camera_logic_glsl = "cameraOrbit(st, u_mouse.xy, u_resolution, 1.0, ro, rd);"

        scene_code = assemble_shader_code(self.sdf_obj)
        
        full_fragment_shader = f"""
            #version 330 core
            
            uniform vec2 u_resolution;
            uniform float u_time;
            uniform vec4 u_mouse;
            uniform vec3 u_bg_color;
            
            {material_struct_glsl}
            {material_uniform_glsl}

            out vec4 f_color;
            
            {get_glsl_content('sdf/primitives.glsl')}
            {get_glsl_content('scene/camera.glsl')}
            {get_glsl_content('scene/raymarching.glsl')}
            {get_glsl_content('scene/light.glsl')}
            
            {scene_code}

            void main() {{
                vec2 st = (2.0 * gl_FragCoord.xy - u_resolution.xy) / u_resolution.y;
                vec3 ro, rd;
                {camera_logic_glsl}
                
                vec3 color = u_bg_color;
                vec4 hit = raymarch(ro, rd);
                float t = hit.x;
                int material_id = int(hit.y);
                
                if (t > 0.0) {{
                    vec3 p = ro + t * rd;
                    vec3 normal = estimateNormal(p);
                    vec3 lightPos = {light_pos_str};
                    vec3 lightDir = normalize(lightPos - p);
                    
                    float diffuse = max(dot(normal, lightDir), {ambient_strength_str});
                    float shadow = softShadow(p + normal * 0.01, lightDir, {shadow_softness_str});
                    diffuse *= shadow;
                    float ao = ambientOcclusion(p, normal, {ao_strength_str});
                    
                    vec3 material_color = vec3(0.8); // Default color if no material
                    if (material_id >= 0 && material_id < {len(materials)}) {{
                        material_color = u_materials[material_id].color;
                    }}

                    color = material_color * diffuse * ao;
                }}
                f_color = vec4(color, 1.0);
            }}
        """

        vertex_shader = """
            #version 330 core
            in vec2 in_vert;
            void main() { gl_Position = vec4(in_vert, 0.0, 1.0); }
        """
        try:
            program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=full_fragment_shader)
            print("INFO: Shader compiled successfully.")
            
            # Upload material data
            for i, mat in enumerate(materials):
                program[f'u_materials[{i}].color'].value = mat.color
            
            program['u_bg_color'].value = self.bg_color

            return program
        except Exception as e:
            print(f"ERROR: Shader compilation failed. Keeping previous shader. Details:\n{e}")
            return self.program

    def _setup_gl(self):
        import moderngl
        self.ctx = moderngl.create_context()
        self.program = self._compile_shader()
        if self.program is None:
            raise RuntimeError("Failed to compile initial shader. Cannot continue.")
        vertices = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0], dtype='f4')
        self.vbo = self.ctx.buffer(vertices)
        self.vao = self.ctx.simple_vertex_array(self.program, self.vbo, 'in_vert')

    def _reload_script(self):
        print(f"INFO: Change detected in '{Path(self.script_path).name}'. Reloading...")
        try:
            spec = importlib.util.spec_from_file_location("user_script", self.script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, 'main') and callable(module.main):
                result = module.main()
                new_sdf_obj, new_cam_obj, new_light_obj = None, None, None
                
                if isinstance(result, tuple):
                    new_sdf_obj = result[0]
                    for item in result[1:]:
                        if isinstance(item, Camera):
                            new_cam_obj = item
                        elif isinstance(item, Light):
                            new_light_obj = item
                else:
                    new_sdf_obj = result
                
                if new_sdf_obj:
                    self.sdf_obj = new_sdf_obj
                    self.camera = new_cam_obj
                    self.light = new_light_obj
                    self.program = self._compile_shader()
                    self.vao = self.ctx.simple_vertex_array(self.program, self.vbo, 'in_vert')
            else:
                print("WARNING: No valid `main` function found. Cannot reload.")
        except Exception as e:
            print(f"ERROR: Failed to reload script: {e}")

    def _start_watcher(self):
        class ChangeHandler(FileSystemEventHandler):
            def __init__(self, renderer_instance): self.renderer = renderer_instance
            def on_modified(self, event):
                if event.src_path == self.renderer.script_path:
                    self.renderer.reload_pending = True
        
        observer = Observer()
        observer.schedule(ChangeHandler(self), str(Path(self.script_path).parent), recursive=False)
        observer.daemon = True
        observer.start()
        print(f"INFO: Watching '{Path(self.script_path).name}' for changes...")

    def run(self):
        import glfw
        import moderngl

        writer = None
        if self.record_path:
            try:
                import imageio
                fps = 30
                writer = imageio.get_writer(self.record_path, fps=fps)
                print(f"INFO: Recording video to '{self.record_path}' at {fps} FPS.")
            except ImportError:
                print("ERROR: 'imageio' is required for video recording.\nPlease install it via: pip install sdforge[record]")
                self.record_path = None

        self._init_window()
        self._setup_gl()
        
        if self.watching: self._start_watcher()

        while not glfw.window_should_close(self.window):
            if self.reload_pending:
                self._reload_script()
                self.reload_pending = False

            width, height = glfw.get_framebuffer_size(self.window)
            self.ctx.viewport = (0, 0, width, height)
            
            try: self.program['u_resolution'].value = (width, height)
            except KeyError: pass
            try: self.program['u_time'].value = glfw.get_time()
            except KeyError: pass
            try:
                mx, my = glfw.get_cursor_pos(self.window)
                self.program['u_mouse'].value = (mx, height - my, 0, 0)
            except KeyError: pass

            self.vao.render(mode=moderngl.TRIANGLE_STRIP)
            
            if writer:
                frame_bytes = self.ctx.fbo.read(components=3, alignment=1)
                frame_np = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3))
                writer.append_data(np.flipud(frame_np))
            
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        if writer: writer.close()
        print("INFO: Viewer window closed.")
        glfw.terminate()

def render(sdf_obj, camera=None, light=None, watch=True, record=None, bg_color=(0.1, 0.12, 0.15), **kwargs):
    if is_in_colab():
        from IPython.display import IFrame

        materials = []
        sdf_obj._collect_materials(materials)
        if len(materials) > MAX_MATERIALS:
            materials = materials[:MAX_MATERIALS]

        material_struct_glsl = "struct MaterialInfo { vec3 color; };\n"
        
        colors_glsl = ", ".join([f"vec3({c.color[0]}, {c.color[1]}, {c.color[2]})" for c in materials])
        if not materials:
            material_array_glsl = f"const MaterialInfo u_materials[1];\n"
        else:
            material_array_glsl = f"const MaterialInfo u_materials[{len(materials)}] = MaterialInfo[]({colors_glsl});\n"
        
        # --- Light ---
        light_pos_str = "ro"
        ambient_strength_str = "0.1"
        shadow_softness_str = "8.0"
        ao_strength_str = "3.0"

        if light:
            if light.position:
                pos = light.position
                light_pos_str = f"vec3({_glsl_format(pos[0])}, {_glsl_format(pos[1])}, {_glsl_format(pos[2])})"
            ambient_strength_str = _glsl_format(light.ambient_strength)
            shadow_softness_str = _glsl_format(light.shadow_softness)
            ao_strength_str = _glsl_format(light.ao_strength)

        # --- Camera ---
        camera_logic_glsl = ""
        if camera:
            pos = camera.position
            pos_str = f"vec3({_glsl_format(pos[0])}, {_glsl_format(pos[1])}, {_glsl_format(pos[2])})"
            target = camera.target
            target_str = f"vec3({_glsl_format(target[0])}, {_glsl_format(target[1])}, {_glsl_format(target[2])})"
            zoom_str = _glsl_format(camera.zoom)
            camera_logic_glsl = f"cameraStatic(st, {pos_str}, {target_str}, {zoom_str}, ro, rd);"
        else:
            camera_logic_glsl = "cameraOrbit(st, u_mouse.xy, u_resolution, 1.0, ro, rd);"

        shader_code = assemble_shader_code(sdf_obj)
        
        width, height = 800, 600

        html_template = f"""
        <!DOCTYPE html><html><head><title>SDF Forge Viewer</title>
        <style>body{{margin:0;overflow:hidden}}canvas{{display:block}}</style></head>
        <body><script type="importmap">{{"imports":{{"three":"https://unpkg.com/three@0.157.0/build/three.module.js"}}}}</script>
        <script type="module">
        import * as THREE from 'three';
        const fragmentShader = 
            #version 300 es
            precision mediump float;`
            varying vec2 vUv;
            uniform float u_time;
            uniform vec4 u_mouse;
            uniform vec2 u_resolution;
            #define u_bg_color vec3({bg_color[0]}, {bg_color[1]}, {bg_color[2]})
            
            {material_struct_glsl}
            {material_array_glsl}

            {get_glsl_content('sdf/primitives.glsl')}
            {get_glsl_content('scene/camera.glsl')}
            {get_glsl_content('scene/raymarching.glsl')}
            {get_glsl_content('scene/light.glsl')}
            {shader_code}
            void main() {{
                vec2 st = (2.0*vUv - 1.0) * vec2(u_resolution.x/u_resolution.y, 1.0);
                vec3 ro, rd;
                {camera_logic_glsl}
                vec3 color = u_bg_color;
                vec4 hit = raymarch(ro, rd);
                float t = hit.x;
                int material_id = int(hit.y);
                if (t > 0.0) {{
                    vec3 p = ro + t * rd;
                    vec3 normal = estimateNormal(p);
                    vec3 lightPos = {light_pos_str};
                    vec3 lightDir = normalize(lightPos - p);
                    float diffuse = max(dot(normal, lightDir), {ambient_strength_str});
                    float shadow = softShadow(p+normal*0.01, lightDir, {shadow_softness_str});
                    diffuse *= shadow;
                    float ao = ambientOcclusion(p, normal, {ao_strength_str});
                    vec3 material_color = vec3(0.8);
                    if (material_id >= 0 && material_id < {len(materials)}) {{
                        material_color = u_materials[material_id].color;
                    }}
                    color = material_color * diffuse * ao;
                }}
                gl_FragColor = vec4(color, 1.0);
            }}
        `;
        const scene = new THREE.Scene();
        const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
        const renderer = new THREE.WebGLRenderer({{antialias: true}});
        document.body.appendChild(renderer.domElement);

        const uniforms = {{
            u_time:{{value:0}}, 
            u_mouse:{{value:new THREE.Vector4()}},
            u_resolution: {{value: new THREE.Vector2({width}.0, {height}.0)}}
        }};

        const material = new THREE.ShaderMaterial({{vertexShader:`varying vec2 vUv; void main(){{vUv=uv;gl_Position=vec4(position,1.0);}}`, fragmentShader, uniforms}});
        scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2), material));
        renderer.setSize({width}, {height});
        document.addEventListener('mousemove', e=>{{uniforms.u_mouse.value.x=e.clientX;uniforms.u_mouse.value.y={height}-e.clientY;}});
        function animate(t){{requestAnimationFrame(animate);uniforms.u_time.value=t*0.001;renderer.render(scene,camera);}}
        animate();
        </script></body></html>
        """
        return IFrame(srcdoc=html_template, width=width + 20, height=height + 20)

    try:
        import moderngl, glfw
    except ImportError:
        print("ERROR: Live rendering requires 'moderngl' and 'glfw'.", file=sys.stderr)
        print("Please install them via: pip install sdforge[full] or pip install moderngl glfw", file=sys.stderr)
        return
        
    renderer = NativeRenderer(sdf_obj, camera=camera, light=light, watch=watch, record=record, bg_color=bg_color, **kwargs)
    renderer.run()