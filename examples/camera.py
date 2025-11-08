import sys
from sdforge import box, sphere, Camera

def static_camera_example():
    """
    Returns a scene and a fixed Camera object.
    The renderer will use this camera's position and target.
    """
    scene = box(1.5, radius=0.1) | sphere(1.2)
    
    # Define a camera positioned at (4, 3, 4), looking at the origin.
    cam = Camera(position=(4, 3, 4), target=(0, 0, 0), zoom=1.5)
    
    return scene, cam

def interactive_camera_example():
    """
    Returns only a scene.
    When no camera is provided to the render function, it defaults
    to an interactive orbit camera controlled by the mouse.
    """
    scene = box(1.5, radius=0.1) | sphere(1.2)
    return scene

def main():
    """
    Renders an example based on a command-line argument.
    """
    print("--- SDForge Camera Examples ---")
    
    examples = {
        "static": static_camera_example,
        "interactive": interactive_camera_example,
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
        print("Available examples are:")
        for key in examples:
            print(f"  - {key}")
        return

    print(f"Rendering: {example_name.replace('_', ' ').title()} Example")
    result = scene_func()
    
    # Handle both return types: (scene, camera) or just scene
    if isinstance(result, tuple):
        scene, cam = result
        scene.render(camera=cam)
    else:
        scene = result
        scene.render() # camera=None, defaults to interactive

if __name__ == "__main__":
    main()