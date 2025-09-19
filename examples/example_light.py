from sdforge import *

def main():
    """
    This example demonstrates scene light controls.
    The `main` function can return a tuple containing the sdf object,
    a camera object, and a light object.
    """

    # A simple shape
    shape = sphere(1) & box(1.5)
    shape -= cylinder(0.2)

    # --- Camera ---
    cam = Camera(position=(3, 4, 5), target=(0, 0, 0))

    # --- Light ---
    # You can define light and animate its properties using GLSL expressions.
    # The `u_time` uniform is available for animations.
    
    # 1. Default headlight (if you return `shape, cam` or just `shape`)
    
    # 2. A static, hard-shadowed light from the side (uncomment to use)
    # light = Light(position=(4, 4, -3), shadow_softness=32.0, ambient_strength=0.2)

    # 3. An animated light that orbits the scene with animated shadow softness
    light = Light(
        position=(
            "8.0 * sin(u_time * 0.3)",
            "5.0",
            "8.0 * cos(u_time * 0.3)"
        ),
        ambient_strength=0.05,
        shadow_softness="12.0 + 10.0 * (1.0 + sin(u_time * 0.7))",
        ao_strength=2.5,
    )
    
    # The order of camera and light objects in the returned tuple does not matter for hot-reloading.
    return shape, cam, light

if __name__ == "__main__":
    result = main()
    if result:
        # A robust way to unpack the scene objects that also works with hot-reloading
        sdf_obj, cam_obj, light_obj = result
        sdf_obj.render(camera=cam_obj, light=light_obj, watch=True)