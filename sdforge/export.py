from pathlib import Path

def _get_glsl_from_lib(rel_path: str) -> str:
    """Reads the content of a GLSL file from the library."""
    try:
        glsl_path = Path(__file__).parent / 'glsl' / rel_path
        with open(glsl_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"ERROR: Could not find GLSL library file: {rel_path}")
        return ""

def assemble_standalone_shader(sdf_obj) -> str:
    """
    Assembles a complete, self-contained GLSL fragment shader for an SDF object.
    The resulting shader can be used in other applications like Godot or Three.js.
    """
    from .engine import SceneCompiler

    # 1. Collect user-defined Uniforms and Params from the SDF tree
    uniforms = {}
    sdf_obj._collect_uniforms(uniforms)
    
    params = {}
    sdf_obj._collect_params(params)
    
    all_user_uniforms = list(uniforms.keys()) + [p.uniform_name for p in params.values()]
    custom_uniforms_glsl = "\n".join([f"uniform float {name};" for name in all_user_uniforms])

    # 2. Compile the core Scene(p) function and its dependencies
    scene_compiler = SceneCompiler()
    scene_code = scene_compiler.compile(sdf_obj)
    
    # 3. Assemble the final shader string using a template
    shader = f"""
#version 330 core

// --- Uniforms ---
uniform vec2 u_resolution; // The viewport resolution (in pixels)
uniform float u_time;      // Time in seconds

// Generic camera uniforms (can be controlled by your application)
uniform vec3 u_cam_pos = vec3(5.0, 4.0, 5.0);
uniform vec3 u_cam_target = vec3(0.0, 0.0, 0.0);
uniform float u_cam_zoom = 1.0;

// Generic lighting uniforms
uniform vec3 u_light_pos = vec3(4.0, 5.0, 6.0);
uniform float u_ambient_strength = 0.2;
uniform float u_shadow_softness = 8.0;

// User-defined uniforms from Forge and Param objects
{custom_uniforms_glsl}

out vec4 f_color; // Output fragment color

// --- SDForge GLSL Library ---
// NOTE: The Scene() function generated below already includes the necessary
// sdf/noise.glsl, sdf/operations.glsl, sdf/primitives.glsl, and sdf/transforms.glsl
// This section includes the rendering-specific functions.
{_get_glsl_from_lib('camera.glsl')}
// The Scene() function must be declared before it is used by the functions below.
vec4 Scene(in vec3 p);

// --- Raymarching & Normals ---
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

float softShadow(vec3 ro, vec3 rd, float softness) {{
    float res = 1.0;
    float t = 0.02;
    for (int i = 0; i < 32; i++) {{
        float h = Scene(ro + rd * t).x;
        if (h < 0.001) return 0.0;
        res = min(res, softness * h / t);
        t += h;
        if (t > 10.0) break;
    }}
    return clamp(res, 0.0, 1.0);
}}


// --- Scene Definition (Generated from Python) ---
{scene_code}

// --- Main Render Loop ---
void main() {{
    // 1. Setup camera ray
    vec2 st = (2.0 * gl_FragCoord.xy - u_resolution.xy) / u_resolution.y;
    vec3 ro, rd;
    cameraStatic(st, u_cam_pos, u_cam_target, u_cam_zoom, ro, rd);
    
    // 2. Raymarch the scene
    float t = raymarch(ro, rd);
    
    vec3 color = vec3(0.1, 0.12, 0.15); // Default background color

    if (t > 0.0) {{
        // 3. We hit something, calculate surface properties
        vec3 p = ro + t * rd;
        vec3 normal = estimateNormal(p);
        vec3 lightDir = normalize(u_light_pos - p);
        
        // 4. Calculate lighting and shadows
        float diffuse = max(dot(normal, lightDir), u_ambient_strength);
        float shadow = softShadow(p + normal * 0.01, lightDir, u_shadow_softness);
        
        // 5. Apply lighting to a default material color
        color = vec3(0.9) * diffuse * shadow;
    }}
    
    f_color = vec4(color, 1.0);
}}
"""
    return shader