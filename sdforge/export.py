from pathlib import Path

def get_glsl_content_from_path(glsl_path: str) -> str:
    """Reads the content of a GLSL file."""
    try:
        with open(Path(__file__).parent / 'glsl' / glsl_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"ERROR: Could not find GLSL file: {glsl_path}")
        return ""

def assemble_standalone_shader(sdf_obj) -> str:
    """
    Assembles a complete, self-contained GLSL fragment shader for an SDF object.
    The resulting shader can be used in other applications like Godot or Three.js.
    """
    from .render import assemble_shader_code as assemble_scene
    from .core import _glsl_format

    # 1. Collect Materials and Uniforms from the SDF tree
    materials = []
    sdf_obj._collect_materials(materials)
    
    uniforms = {}
    sdf_obj._collect_uniforms(uniforms)

    params = {}
    sdf_obj._collect_params(params)

    # 2. Build Material and Uniform GLSL declarations
    material_struct_glsl = "struct MaterialInfo { vec3 color; };\n"
    material_uniform_glsl = f"uniform MaterialInfo u_materials[{max(1, len(materials))}];\n"
    
    all_user_uniforms = list(uniforms.keys()) + [p.uniform_name for p in params.values()]
    custom_uniforms_glsl = "\n".join([f"uniform float {name};" for name in all_user_uniforms])

    # 3. Assemble the core Scene(p) function from the SDF object
    scene_code = assemble_scene(sdf_obj)
    
    # 4. Assemble the final shader string
    shader = f"""
#version 330 core

// --- Uniforms ---
uniform vec2 u_resolution; // The viewport resolution (in pixels)
uniform float u_time;      // Time in seconds

// Camera uniforms (can be controlled by your application)
uniform vec3 u_cam_pos = vec3(5.0, 4.0, 5.0);
uniform vec3 u_cam_target = vec3(0.0, 0.0, 0.0);
uniform float u_cam_zoom = 1.0;

// Lighting uniforms
uniform vec3 u_light_pos = vec3(4.0, 5.0, 6.0);
uniform float u_ambient_strength = 0.1;
uniform float u_shadow_softness = 8.0;

// User-defined uniforms from Forge and Param objects
{custom_uniforms_glsl}

// --- Materials ---
{material_struct_glsl}
{material_uniform_glsl}

out vec4 f_color; // Output fragment color

// --- SDForge GLSL Library ---
{get_glsl_content_from_path('sdf/noise.glsl')}
{get_glsl_content_from_path('sdf/primitives.glsl')}
{get_glsl_content_from_path('sdf/operations.glsl')}
{get_glsl_content_from_path('sdf/transforms.glsl')}
{get_glsl_content_from_path('scene/camera.glsl')}
{get_glsl_content_from_path('scene/light.glsl')}
{get_glsl_content_from_path('scene/raymarching.glsl')}

// --- Scene Definition (Generated from Python) ---
{scene_code}

// --- Main Render Loop ---
void main() {{
    // 1. Setup camera ray
    vec2 st = (2.0 * gl_FragCoord.xy - u_resolution.xy) / u_resolution.y;
    vec3 ro, rd;
    cameraStatic(st, u_cam_pos, u_cam_target, u_cam_zoom, ro, rd);
    
    // 2. Raymarch the scene
    vec4 hit = raymarch(ro, rd);
    float t = hit.x;
    
    vec3 color = vec3(0.1, 0.12, 0.15); // Default background color

    if (t > 0.0) {{
        // 3. We hit something, calculate surface properties
        vec3 p = ro + t * rd;
        vec3 normal = estimateNormal(p);
        vec3 lightDir = normalize(u_light_pos - p);
        
        // 4. Calculate lighting and shadows
        float diffuse = max(dot(normal, lightDir), u_ambient_strength);
        float shadow = softShadow(p + normal * 0.01, lightDir, u_shadow_softness);
        diffuse *= shadow;
        
        // 5. Get material color
        int material_id = int(hit.y);
        vec3 material_color = vec3(0.8); // Default color
        if (material_id >= 0 && material_id < {len(materials)}) {{
            material_color = u_materials[material_id].color;
        }}

        color = material_color * diffuse;
    }}
    
    f_color = vec4(color, 1.0);
}}
"""
    return shader