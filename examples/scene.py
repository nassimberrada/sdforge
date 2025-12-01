import sys
from sdforge import box, sphere, Camera, Light, Debug, Param

def interactive_camera_example(): return box(1.5).round(0.1) | sphere(1.2)
def static_camera_example():
    scene = box(1.5).round(0.1) | sphere(1.2)
    cam = Camera(position=(4, 3, 4), target=(0, 0, 0), zoom=1.5)
    return scene, cam

def headlight_example(): return box(1.5).round(0.1) | sphere(1.2), Light(shadow_softness=32.0, ao_strength=5.0)
def static_light_example():
    scene = box(1.5).round(0.1) | sphere(1.2)
    light = Light(position=(4, 5, 3), shadow_softness=16.0)
    return scene, light

def full_scene_example():
    scene = box(1.5).round(0.1) | sphere(1.2)
    cam = Camera(position=(-5, 2, 5), target=(0, 0, 0), zoom=1.2)
    light = Light(position=(-2, 5, 2), shadow_softness=4.0, ambient_strength=0.3)
    return scene, cam, light

def normals_debug_example():
    scene = box(1.5).round(0.1) - sphere(1.2)
    return scene, Debug('normals')

def steps_debug_example():
    scene = box(1.5).round(0.1) - sphere(1.2)
    return scene, Debug('steps')

def slice_debug_example():
    scene = sphere(1.0).shell(0.1) - box(1.0).translate((0.5, 0, 0))
    return scene, Debug('slice', plane='xy', slice_height=0.0, view_scale=3.0)

def params_example():
    p_size = Param("Box Size", 0.8, 0.2, 2.0)
    p_radius = Param("Corner Radius", 0.1, 0.0, 0.5)
    p_twist = Param("Twist", 0.0, -10.0, 10.0)
    return box(size=p_size).round(p_radius).twist(strength=p_twist)

def export_shader_example():
    scene = box(1.5).round(0.1) - sphere(0.8)
    path = "exported_shader.glsl"
    print(f"Exporting to {path}...")
    scene.export_shader(path)
    return scene

def auto_bounds_save_example():
    scene = box(1.5).round(0.1) | sphere(1.2)
    scene.save("auto_bounds.stl", samples=2**18)

def adaptive_save_example():
    scene = box(2.0).round(0.1).shell(0.05)
    scene.save("adaptive.stl", adaptive=True, octree_depth=7)

def voxel_save_example():
    scene = sphere(1.0)
    scene.save("voxel_res.stl", voxel_size=0.1, adaptive=True)

def dual_contouring_save_example():
    scene = box(1.5)
    scene.save("dc_sharp.stl", samples=2**16, algorithm='dual_contouring')

def main():
    print("--- SDForge Scene & IO Examples ---")
    render_examples = {
        "static_cam": static_camera_example, "interactive_cam": interactive_camera_example,
        "static_light": static_light_example, "headlight": headlight_example, "full": full_scene_example,
        "debug_normals": normals_debug_example, "debug_steps": steps_debug_example, "debug_slice": slice_debug_example,
        "params": params_example, "export_shader": export_shader_example,
    }
    save_examples = {
        "save_auto": auto_bounds_save_example, "save_adaptive": adaptive_save_example,
        "save_voxel": voxel_save_example, "save_dc": dual_contouring_save_example,
    }

    if len(sys.argv) < 2:
        print("Render Examples:", ", ".join(render_examples.keys()))
        print("Save Examples:", ", ".join(save_examples.keys()))
        return

    name = sys.argv[1]
    if name in save_examples:
        save_examples[name]()
        return

    func = render_examples.get(name)
    if not func: print(f"Example '{name}' not found."); return

    result = func()
    scene, cam, light, debug = None, None, None, None
    
    if isinstance(result, tuple):
        for item in result:
            from sdforge import SDFNode
            if isinstance(item, SDFNode): scene = item
            elif isinstance(item, Camera): cam = item
            elif isinstance(item, Light): light = item
            elif isinstance(item, Debug): debug = item
    else: scene = result

    if scene: scene.render(camera=cam, light=light, debug=debug)

if __name__ == "__main__":
    main()