import sys
import os
import time
from pathlib import Path
import importlib.util

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

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
    scene_glsl = sdf_obj.to_glsl()
    # Preserve order while deduplicating inline defs
    inline_definitions = []
    for d in sdf_obj.get_glsl_definitions():
        if d not in inline_definitions:
            inline_definitions.append(d)
    joined_defs = '\n'.join(inline_definitions)
    return f"""
    {joined_defs}
    float Scene(in vec3 p) {{ return {scene_glsl}; }}
    """

class NativeRenderer:
    """Handles the creation of a native window and renders the SDF."""

    def __init__(self, sdf_obj, watch=False, width=1280, height=720):
        self.sdf_obj = sdf_obj
        self.watching = watch
        self.width = width
        self.height = height
        self.window = None
        self.ctx = None
        self.program = None
        self.vao = None
        self.vbo = None
        self.script_path = os.path.abspath(sys.argv[0])
        self.reload_pending = False # Flag to signal a reload is needed

    def _init_window(self):
        """Initializes GLFW and creates a window."""
        import glfw
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        
        self.window = glfw.create_window(self.width, self.height, "SDF Forge Viewer", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create GLFW window")
        
        glfw.make_context_current(self.window)

    def _compile_shader(self):
        """Compiles the vertex and fragment shaders."""
        import moderngl
        
        scene_code = assemble_shader_code(self.sdf_obj)
        
        full_fragment_shader = f"""
            #version 330 core
            
            uniform vec2 u_resolution;
            uniform float u_time;
            uniform vec4 u_mouse; // .xy is pixel pos, .zw is click state
            
            out vec4 f_color;
            
            {get_glsl_content('sdf/primitives.glsl')}
            {get_glsl_content('scene/camera.glsl')}
            {get_glsl_content('scene/raymarching.glsl')}
            {get_glsl_content('scene/lighting.glsl')}
            
            {scene_code}

            void main() {{
                vec2 st = (2.0 * gl_FragCoord.xy - u_resolution.xy) / u_resolution.y;
                vec3 ro, rd;
                cameraOrbit(st, u_mouse.xy, u_resolution, 1.0, ro, rd);
                
                vec3 color = vec3(0.1, 0.12, 0.15); // Background color
                float t = raymarch(ro, rd);
                
                if (t > 0.0) {{
                    vec3 p = ro + t * rd;
                    vec3 normal = estimateNormal(p);
                    vec3 lightPos = ro; // Camera as light source
                    vec3 lightDir = normalize(lightPos - p);
                    
                    float diffuse = max(dot(normal, lightDir), 0.1);
                    float shadow = softShadow(p + normal * 0.01, lightDir);
                    diffuse *= shadow;
                    float ao = ambientOcclusion(p, normal);
                    
                    color = vec3(0.2, 0.5, 0.85) * diffuse * ao;
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
            return program
        except Exception as e:
            print(f"ERROR: Shader compilation failed. Keeping previous shader. Details:\n{e}")
            return self.program # Return old program if compilation fails

    def _setup_gl(self):
        """Sets up ModernGL context and shaders."""
        import moderngl
        import numpy as np

        self.ctx = moderngl.create_context()
        self.program = self._compile_shader()

        vertices = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0], dtype='f4')
        self.vbo = self.ctx.buffer(vertices)
        self.vao = self.ctx.simple_vertex_array(self.program, self.vbo, 'in_vert')

    def _reload_script(self):
        """Hot-reloads the user script to get the new SDF object."""
        print(f"INFO: Change detected in '{Path(self.script_path).name}'. Reloading...")
        try:
            spec = importlib.util.spec_from_file_location("user_script", self.script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, 'main') and callable(module.main):
                new_sdf_obj = module.main()
                if new_sdf_obj:
                    self.sdf_obj = new_sdf_obj
                    self.program = self._compile_shader()
                    self.vao = self.ctx.simple_vertex_array(self.program, self.vbo, 'in_vert')
            else:
                print("WARNING: No valid `main` function found. Cannot reload.")
        except Exception as e:
            print(f"ERROR: Failed to reload script: {e}")

    def _start_watcher(self):
        """Starts the watchdog file observer for hot-reloading."""
        # --- THE FIX IS HERE (Part 1) ---
        # The event handler now only sets a flag. It does not do any GL work.
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

    def run(self):
        """Runs the main rendering loop."""
        import glfw
        import moderngl

        self._init_window()
        self._setup_gl()
        
        if self.watching:
            self._start_watcher()

        while not glfw.window_should_close(self.window):
            # --- THE FIX IS HERE (Part 2) ---
            # Check the flag on the main thread before rendering each frame.
            if self.reload_pending:
                self._reload_script()
                self.reload_pending = False # Reset the flag

            width, height = glfw.get_framebuffer_size(self.window)
            self.ctx.viewport = (0, 0, width, height)
            
            try:
                self.program['u_resolution'].value = (width, height)
            except KeyError: pass

            try:
                self.program['u_time'].value = glfw.get_time()
            except KeyError: pass

            try:
                mx, my = glfw.get_cursor_pos(self.window)
                self.program['u_mouse'].value = (mx, height - my, 0, 0)
            except KeyError: pass

            self.vao.render(mode=moderngl.TRIANGLE_STRIP)
            
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        print("INFO: Viewer window closed.")
        glfw.terminate()

def render(sdf_obj, watch=True, **kwargs):
    """
    Renders an SDF object.
    - In a native window on desktop.
    - As an embedded iframe in Google Colab.
    """
    if is_in_colab():
        from IPython.display import display, HTML
        shader_code = assemble_shader_code(sdf_obj)
        # ... (Colab rendering code remains the same) ...
        html_template = f"""
        <!DOCTYPE html><html><head><title>SDF Forge Viewer</title>
        <style>body{{margin:0;overflow:hidden}}canvas{{display:block}}</style></head>
        <body><script type="importmap">{{"imports":{{"three":"https://unpkg.com/three@0.157.0/build/three.module.js"}}}}</script>
        <script type="module">
        import * as THREE from 'three';
        const fragmentShader = `
            varying vec2 vUv;
            uniform vec2 u_resolution;
            uniform float u_time;
            uniform vec4 u_mouse;
            {get_glsl_content('sdf/primitives.glsl')}
            {get_glsl_content('scene/camera.glsl')}
            {get_glsl_content('scene/raymarching.glsl')}
            {get_glsl_content('scene/lighting.glsl')}
            {shader_code}
            void main() {{
                vec2 st = (2.0*vUv - 1.0) * vec2(u_resolution.x/u_resolution.y, 1.0);
                vec3 ro, rd;
                cameraOrbit(st, u_mouse.xy, u_resolution, 1.0, ro, rd);
                vec3 color = vec3(0.1, 0.12, 0.15);
                float t = raymarch(ro, rd);
                if (t > 0.0) {{
                    vec3 p = ro + t * rd;
                    vec3 normal = estimateNormal(p);
                    vec3 lightPos = ro;
                    vec3 lightDir = normalize(lightPos - p);
                    float diffuse = max(dot(normal, lightDir), 0.1);
                    float shadow = softShadow(p+normal*0.01, lightDir);
                    diffuse *= shadow;
                    float ao = ambientOcclusion(p, normal);
                    color = vec3(0.2, 0.5, 0.85) * diffuse * ao;
                }}
                gl_FragColor = vec4(color, 1.0);
            }}
        `;
        const scene = new THREE.Scene();
        const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
        const renderer = new THREE.WebGLRenderer({{antialias: true}});
        document.body.appendChild(renderer.domElement);
        const uniforms = {{u_time:{{value:0}},u_resolution:{{value:new THREE.Vector2()}},u_mouse:{{value:new THREE.Vector4()}}}};
        const material = new THREE.ShaderMaterial({{vertexShader:`varying vec2 vUv; void main(){{vUv=uv;gl_Position=vec4(position,1.0);}}`, fragmentShader, uniforms}});
        scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2), material));
        function onResize(){{renderer.setSize(800,600);uniforms.u_resolution.value.set(800,600);}}
        document.addEventListener('mousemove', e=>{{uniforms.u_mouse.value.x=e.clientX;uniforms.u_mouse.value.y=600-e.clientY;}});
        function animate(t){{requestAnimationFrame(animate);uniforms.u_time.value=t*0.001;renderer.render(scene,camera);}}
        onResize(); animate();
        </script></body></html>
        """
        escaped_html = html_template.replace('"', "&quot;")
        display(HTML(f'<iframe srcdoc="{escaped_html}" width="800" height="600" style="border:1px solid #ccc"></iframe>'))
        return

    # Desktop rendering
    try:
        import moderngl
        import glfw
    except ImportError:
        print("ERROR: Live rendering requires 'moderngl' and 'glfw'.")
        print("Please install them via: pip install moderngl glfw")
        return
        
    renderer = NativeRenderer(sdf_obj, watch=watch, **kwargs)
    renderer.run()