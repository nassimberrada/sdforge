import sys
from sdforge import box, sphere, Debug

def normals_debug_example():
    """Visualizes the surface normals as colors."""
    scene = box(1.5, radius=0.1) - sphere(1.2)
    debug = Debug('normals')
    return scene, debug

def steps_debug_example():
    """Visualizes the number of raymarching steps."""
    scene = box(1.5, radius=0.1) - sphere(1.2)
    debug = Debug('steps')
    return scene, debug

def main():
    """Renders a debug example based on a command-line argument."""
    print("--- SDForge Debug Examples ---")
    
    examples = {
        "normals": normals_debug_example,
        "steps": steps_debug_example,
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
    
    scene, debug = None, None
    for item in result:
        from sdforge.core import SDFNode
        if isinstance(item, SDFNode): scene = item
        if isinstance(item, Debug): debug = item

    if scene:
        scene.render(debug=debug)
    else:
        print("Error: Example function did not return a scene object.")


if __name__ == "__main__":
    main()