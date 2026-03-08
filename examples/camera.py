import sys
from sdforge import box, sphere, Camera

def default_camera_example():
    """
    Returns only a scene.
    When no camera is provided to the render function, it defaults
    to a standard perspective orbit camera starting at (5, 4, 5).
    """
    scene = box(1.5).round(0.1).color((0.2, 0.6, 1.0)) | sphere(radius=1.2).color((1.0, 0.2, 0.2))
    return scene

def custom_perspective_example():
    """
    Returns a scene and a custom Camera object.
    The renderer will use this camera's position and target as the starting
    point for the interactive orbit.
    """
    scene = box(1.5).round(0.1).color((0.2, 0.6, 1.0)) | sphere(radius=1.2).color((1.0, 0.2, 0.2))
    
    # Define a camera positioned far away and low to the ground
    cam = Camera(position=(8, 1, 8), target=(0, 0, 0), zoom=2.0, type='perspective')
    
    return scene, cam

def isometric_camera_example():
    """
    Returns a scene and an Orthographic Camera object.
    Because the default camera position is (5, 4, 5), simply setting the type 
    to 'orthographic' creates a perfect classic isometric projection!
    (You can still orbit it with the mouse).
    """
    scene = box(1.5).round(0.1).color((0.2, 0.6, 1.0)) | sphere(radius=1.2).color((1.0, 0.2, 0.2))
    
    # Create an isometric camera
    cam = Camera(type='orthographic', zoom=1.5)
    
    return scene, cam

def main():
    """
    Renders an example based on a command-line argument.
    """
    print("--- SDForge Camera Examples ---")
    print("Note: All cameras are interactive! Click and drag to orbit.")
    
    examples = {
        "default": default_camera_example,
        "custom": custom_perspective_example,
        "isometric": isometric_camera_example,
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
        scene.render() # camera=None, defaults to standard orbit

if __name__ == "__main__":
    main()