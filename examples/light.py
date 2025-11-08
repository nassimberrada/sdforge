import sys
import numpy as np
from sdforge import box, sphere, Light, Camera

def static_light_example():
    """A scene lit by a single, fixed-position light."""
    scene = box(1.5, radius=0.1) | sphere(1.2)
    light = Light(position=(4, 5, 3), shadow_softness=16.0)
    return scene, light

def headlight_example():
    """A scene lit by a headlight attached to the camera."""
    scene = box(1.5, radius=0.1) | sphere(1.2)
    # When Light() is created with no position, it defaults to the camera's position.
    light = Light(shadow_softness=32.0, ao_strength=5.0)
    return scene, light

def main():
    """
    Renders an example based on a command-line argument.
    """
    print("--- SDForge Lighting Examples ---")
    
    examples = {
        "static": static_light_example,
        "headlight": headlight_example,
    }
    
    if len(sys.argv) < 2:
        print("\nPlease provide the name of an example to run.")
        print("Available examples:")
        for key in examples:
            print(f"  - {key}")
        print(f"\nUsage: python {sys.argv[0]} <example_name>")
        return

    example_name = sys.argv[1]
    scene_func = examples.get(example_name)
    
    if not scene_func:
        print(f"\nError: Example '{example_name}' not found.")
        return

    print(f"Rendering: {example_name.replace('_', ' ').title()} Example")
    result = scene_func()
    
    # Unpack the result tuple which may contain scene, light, and/or camera
    scene, light, cam = None, None, None
    from sdforge.core import SDFNode
    for item in result:
        if isinstance(item, SDFNode): scene = item
        if isinstance(item, Light): light = item
        if isinstance(item, Camera): cam = item

    if scene:
        scene.render(camera=cam, light=light)
    else:
        print("Error: Example function did not return a scene object.")


if __name__ == "__main__":
    main()